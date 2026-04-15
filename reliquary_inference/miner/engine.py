from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib import request

import torch

from ..constants import PROOF_VERSION, LAYER_INDEX
from ..protocol.artifacts import make_artifact
from ..protocol.sketch_verifier import SketchProofVerifier
from ..protocol.signatures import sign_commit_binding
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
        )
        self.model = self.bundle["model"]
        self.tokenizer = self.bundle["tokenizer"]
        self.hidden_dim = resolve_hidden_size(self.model)
        self.verifier = SketchProofVerifier(hidden_dim=self.hidden_dim)
        self.model_name = getattr(self.model, "name_or_path", str(cfg["model_ref"]))
        self.wallet = self._load_wallet()

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
            with torch.no_grad():
                generate_kwargs = {
                    "attention_mask": attention_mask,
                    "max_new_tokens": int(self.cfg["max_new_tokens"]),
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
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
        proof_input = torch.tensor([full_tokens], device=next(self.model.parameters()).device)
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
        prompt_length = int(prompt_ids.shape[1])
        prompt_token_ids = prompt_ids[0].detach().cpu().tolist()
        completion_token_ids = full_tokens[prompt_length:]
        completion_logprobs = compute_completion_logprobs(
            model=self.model,
            full_token_ids=full_tokens,
            prompt_length=prompt_length,
            device=str(next(self.model.parameters()).device),
        )
        upload_ref = registry.predicted_completion_bundle_ref(
            window_id=int(window_context["window_id"]),
            miner_id=miner_id,
        )
        artifact = make_artifact(
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
        return artifact

    def _generate_with_vllm(self, prompt_text: str, prompt_ids: torch.Tensor) -> list[int]:
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt_text,
                "max_tokens": int(self.cfg["max_new_tokens"]),
                "temperature": 0.0,
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
