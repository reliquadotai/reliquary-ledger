from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib import request

import torch

from ..constants import LAYER_INDEX, PROOF_VERSION
from ..protocol.artifacts import make_artifact
from ..protocol.signatures import sign_commit_binding
from ..protocol.sketch_verifier import SketchProofVerifier
from ..shared.forward import forward_single_layer
from ..shared.hf_compat import resolve_hidden_size
from ..shared.modeling import (
    compute_completion_logprobs,
    decode_completion,
    load_model_bundle,
)


class MiningEngine:
    def __init__(self, *, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.bundle = load_model_bundle(
            str(cfg["model_ref"]),
            device=str(cfg["device"]),
            dtype_name=str(cfg.get("load_dtype", "auto")),
            require_flash_attention=bool(cfg.get("require_flash_attention", False)),
        )
        self.model = self.bundle["model"]
        self.tokenizer = self.bundle["tokenizer"]
        self.hidden_dim = resolve_hidden_size(self.model)
        self.verifier = SketchProofVerifier(hidden_dim=self.hidden_dim)
        self.model_name = getattr(self.model, "name_or_path", str(cfg["model_ref"]))
        self.wallet = self._load_wallet()

    def generate_m_completions(
        self,
        *,
        task: dict[str, Any],
        window_context: dict[str, Any],
        registry,
        miner_id: str,
        num_samples: int,
    ) -> list[dict[str, Any]]:
        """Generate ``num_samples`` completions in ONE batched ``model.generate()``.

        Serial generation of M=8 rollouts leaves the GPU ~15% utilised —
        matmul tiling is far more efficient when M sequences share a
        single forward pass. We tile the prompt to shape ``(M, L)`` and
        slice each output row, truncating at the first post-prompt EOS
        so HF's batch padding (pad_token_id) never leaks into downstream
        GRAIL verification.

        Per-sample determinism is preserved by seeding the batch with a
        hash over (window_id, task_id, miner_id) — the M sampled rows
        diverge via independent sampling within the batch. A validator
        does NOT need to re-sample; it only verifies the miner-supplied
        tokens via GRAIL commitments.

        Falls back to the single-sample path for num_samples==1 and for
        the dual-engine vLLM path (which already batches server-side).

        Expected speedup on H100-class GPUs: 5-7× vs serial, based on
        upstream observations (romain13190/reliquary@f7510fb).
        """
        if num_samples == 1 or (
            str(self.cfg["miner_mode"]) == "dual_engine" and self.cfg.get("vllm_base_url")
        ):
            return [
                self.generate_completion(
                    task=task,
                    window_context=window_context,
                    registry=registry,
                    miner_id=miner_id,
                    sample_index=i,
                )
                for i in range(num_samples)
            ]

        prompt_text = task["prompt"]
        if hasattr(self.tokenizer, "__call__") and not str(self.cfg["model_ref"]).startswith("toy://"):
            encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            prompt_ids_single = encoded["input_ids"].to(next(self.model.parameters()).device)
        else:
            prompt_ids_single = self.tokenizer.encode(prompt_text, return_tensors="pt").to(
                next(self.model.parameters()).device
            )
        prompt_length = int(prompt_ids_single.shape[1])
        # Tile prompt to (M, L) for batched generation.
        prompt_ids_batch = prompt_ids_single.expand(num_samples, -1).contiguous()
        attention_mask = torch.ones_like(prompt_ids_batch, device=prompt_ids_batch.device)

        # Seed once per batch — individual rows diverge through independent
        # sampling, so the batch as a whole is deterministic given the
        # same (window, task, miner) triple.
        batch_seed_material = (
            f"{window_context['window_id']}|{task['task_id']}|{miner_id}|batch"
        ).encode("utf-8")
        batch_seed = int(hashlib.sha256(batch_seed_material).hexdigest()[:8], 16)
        torch.manual_seed(batch_seed)

        with torch.no_grad():
            generate_kwargs = {
                "attention_mask": attention_mask,
                "max_new_tokens": int(self.cfg["max_new_tokens"]),
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": True,
                "temperature": float(self.cfg.get("generation_temperature", 0.9)),
                "top_p": float(self.cfg.get("generation_top_p", 1.0)),
            }
            try:
                outputs = self.model.generate(prompt_ids_batch, **generate_kwargs)
            except TypeError:
                generate_kwargs.pop("attention_mask", None)
                outputs = self.model.generate(prompt_ids_batch, **generate_kwargs)

        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        # Slice each row and trim at first post-prompt EOS so HF's padding
        # doesn't trail into the tokens the validator GRAIL-verifies.
        completions: list[dict[str, Any]] = []
        for sample_index in range(num_samples):
            seq = outputs[sample_index].tolist()
            gen = seq[prompt_length:]
            for idx, tok in enumerate(gen):
                if tok == eos:
                    gen = gen[: idx + 1]
                    break
                if pad is not None and tok == pad and eos != pad:
                    gen = gen[:idx]
                    break
            full_tokens = seq[:prompt_length] + gen
            completions.append(
                self._finalize_completion(
                    task=task,
                    window_context=window_context,
                    registry=registry,
                    miner_id=miner_id,
                    sample_index=sample_index,
                    full_tokens=full_tokens,
                    prompt_length=prompt_length,
                )
            )
        return completions

    def generate_completion(
        self,
        *,
        task: dict[str, Any],
        window_context: dict[str, Any],
        registry,
        miner_id: str,
        sample_index: int,
    ) -> dict[str, Any]:
        prompt_text = task["prompt"]
        if hasattr(self.tokenizer, "__call__") and not str(self.cfg["model_ref"]).startswith("toy://"):
            encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            prompt_ids = encoded["input_ids"].to(next(self.model.parameters()).device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(prompt_ids.device)
        else:
            prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
            prompt_ids = prompt_ids.to(next(self.model.parameters()).device)
            attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
        if str(self.cfg["miner_mode"]) == "dual_engine" and self.cfg.get("vllm_base_url"):
            full_tokens = self._generate_with_vllm(prompt_text, prompt_ids)
        else:
            # GRPO needs multiple DISTINCT rollouts per prompt — greedy (do_sample=False)
            # would make all M samples identical, collapsing group σ to 0 and
            # starving the zone filter. We switch to sampling with the DAPO-
            # canonical T_PROTO=0.9 whenever samples_per_task>1, and seed per
            # (window_id, task_id, miner, sample_index) so a validator
            # re-running the same sample gets the same tokens.
            samples_per_task = int(self.cfg.get("samples_per_task", 1))
            use_sampling = samples_per_task > 1
            with torch.no_grad():
                generate_kwargs = {
                    "attention_mask": attention_mask,
                    "max_new_tokens": int(self.cfg["max_new_tokens"]),
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if use_sampling:
                    # Per-rollout seed so each of the M samples takes a different
                    # branch through the sampling distribution.
                    seed_material = (
                        f"{window_context['window_id']}|{task['task_id']}|"
                        f"{miner_id}|{sample_index}"
                    ).encode("utf-8")
                    sample_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
                    torch.manual_seed(sample_seed)
                    generate_kwargs.update(
                        do_sample=True,
                        temperature=float(self.cfg.get("generation_temperature", 0.9)),
                        top_p=float(self.cfg.get("generation_top_p", 1.0)),
                    )
                else:
                    generate_kwargs["do_sample"] = False
                try:
                    outputs = self.model.generate(
                        prompt_ids,
                        **generate_kwargs,
                    )
                except TypeError:
                    generate_kwargs.pop("attention_mask", None)
                    outputs = self.model.generate(
                        prompt_ids,
                        **generate_kwargs,
                    )
            full_tokens = outputs[0].tolist()
        prompt_length = int(prompt_ids.shape[1])
        return self._finalize_completion(
            task=task,
            window_context=window_context,
            registry=registry,
            miner_id=miner_id,
            sample_index=sample_index,
            full_tokens=full_tokens,
            prompt_length=prompt_length,
        )

    def _finalize_completion(
        self,
        *,
        task: dict[str, Any],
        window_context: dict[str, Any],
        registry,
        miner_id: str,
        sample_index: int,
        full_tokens: list[int],
        prompt_length: int,
    ) -> dict[str, Any]:
        """Post-generation bookkeeping: GRAIL proof, logprobs, artifact.

        Shared by both the single-rollout path (``generate_completion``)
        and the batched M-rollout path (``generate_m_completions``). The
        only input is the finished token sequence + how many prompt
        tokens are on the left.
        """
        device = next(self.model.parameters()).device
        proof_input = torch.tensor([full_tokens], device=device)
        proof_attention_mask = torch.ones_like(proof_input, device=proof_input.device)
        with torch.no_grad():
            hidden_states, _ = forward_single_layer(self.model, proof_input, proof_attention_mask, LAYER_INDEX)
        hidden_states = hidden_states[0]
        randomness = hashlib.sha256(
            f"{window_context['window_id']}|{task['task_id']}|{miner_id}|{sample_index}".encode("utf-8")
        ).hexdigest()
        r_vec = self.verifier.generate_r_vec(randomness).to(hidden_states.device)
        commitments = self.verifier.create_commitments_batch(hidden_states, r_vec)
        signature, signer_id, signature_scheme = sign_commit_binding(
            full_tokens,
            randomness,
            self.model_name,
            LAYER_INDEX,
            commitments,
            scheme=str(self.cfg["signature_scheme"]),
            signer_id=self._signer_id(miner_id),
            secret=str(self.cfg["signing_secret"]),
            wallet=self.wallet,
        )
        prompt_token_ids = full_tokens[:prompt_length]
        completion_token_ids = full_tokens[prompt_length:]
        completion_logprobs = compute_completion_logprobs(
            model=self.model,
            full_token_ids=full_tokens,
            prompt_length=prompt_length,
            device=str(device),
        )
        upload_ref = registry.predicted_completion_bundle_ref(
            window_id=int(window_context["window_id"]),
            miner_id=miner_id,
        )
        return make_artifact(
            artifact_type="completion",
            producer_id=miner_id,
            producer_role="miner",
            window_id=int(window_context["window_id"]),
            payload={
                "task_id": task["task_id"],
                "task_source": window_context["task_source"],
                "task_index": task.get("dataset_index", task.get("order_index")),
                "prompt_hash": task.get("prompt_hash"),
                "tokens": full_tokens,
                "prompt_token_ids": prompt_token_ids,
                "prompt_length": prompt_length,
                "completion_token_ids": completion_token_ids,
                "completion_logprobs": completion_logprobs,
                "old_sum_logprob": float(sum(completion_logprobs)),
                "completion_text": decode_completion(self.tokenizer, completion_token_ids),
                "completion_digest": hashlib.sha256(json.dumps(completion_token_ids).encode("utf-8")).hexdigest(),
                "model_name": self.model_name,
                "nonce": int(hashlib.sha256(f"{task['task_id']}|{sample_index}|{miner_id}".encode("utf-8")).hexdigest()[:8], 16),
                "proof_version": PROOF_VERSION,
                "layer_index": LAYER_INDEX,
                "commitments": commitments,
                "signature": signature,
                "signature_scheme": signature_scheme,
                "signer_id": signer_id,
                "randomness": randomness,
                "upload_ref": upload_ref,
                "sample_index": sample_index,
                "contamination_tags": [],
            },
            parent_ids=[],
        )

    def _generate_with_vllm(self, prompt_text: str, prompt_ids: torch.Tensor) -> list[int]:
        # vLLM path: keep temperature configurable so GRPO groups also get
        # variance when the dual-engine path is on. Defaults to 0.9 matching
        # DAPO T_PROTO. Set RELIQUARY_INFERENCE_GENERATION_TEMPERATURE=0 to
        # revert to greedy.
        temperature = float(self.cfg.get("generation_temperature", 0.9))
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt_text,
                "max_tokens": int(self.cfg["max_new_tokens"]),
                "temperature": temperature,
                "top_p": float(self.cfg.get("generation_top_p", 1.0)),
            }
        ).encode("utf-8")
        req = request.Request(
            f"{str(self.cfg['vllm_base_url']).rstrip('/')}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["text"]
        completion_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return prompt_ids[0].tolist() + completion_ids

    def _load_wallet(self):
        if str(self.cfg["signature_scheme"]) != "bittensor_hotkey":
            return None
        from bittensor_wallet import Wallet

        return Wallet(
            name=str(self.cfg["wallet_name"]),
            hotkey=str(self.cfg["hotkey_name"]),
            path=str(self.cfg["wallet_path"]),
        )

    def _signer_id(self, miner_id: str) -> str:
        if str(self.cfg["signature_scheme"]) == "local_hmac":
            return miner_id
        if self.wallet is not None:
            return str(self.wallet.hotkey.ss58_address)
        return str(self.cfg.get("local_signer_id", miner_id))
