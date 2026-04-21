from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        import torch

        ids = [ord(ch) + 1 for ch in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id <= 0:
                continue
            chars.append(chr(token_id - 1))
        return "".join(chars)


def _toy_completion_text(prompt_text: str) -> str:
    import re

    lower = prompt_text.lower()
    numbers = [int(value) for value in re.findall(r"[-+]?\d+", prompt_text)]
    if "subtract" in lower and len(numbers) >= 2:
        answer = numbers[0] - numbers[1]
    elif "multiply" in lower and "then add" not in lower and "then multiply" not in lower and len(numbers) >= 2:
        answer = numbers[0] * numbers[1]
    elif "larger" in lower and len(numbers) >= 2:
        answer = max(numbers[0], numbers[1])
    elif "smaller" in lower and len(numbers) >= 2:
        answer = min(numbers[0], numbers[1])
    elif "absolute difference" in lower and len(numbers) >= 2:
        answer = abs(numbers[0] - numbers[1])
    elif "first multiply" in lower and "then add" in lower and len(numbers) >= 3:
        answer = (numbers[0] * numbers[1]) + numbers[2]
    elif "first add" in lower and "then multiply" in lower and len(numbers) >= 3:
        answer = (numbers[0] + numbers[1]) * numbers[2]
    elif "add" in lower and len(numbers) >= 2:
        answer = numbers[0] + numbers[1]
    else:
        return " Verified output."
    return f"Reasoning: toy.\nFinal Answer: {answer}"


def _build_toy_bundle(model_ref: str, device: str) -> dict[str, Any]:
    import torch

    class ToyBackbone(torch.nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.float32)
            features = []
            for offset in range(self.hidden_size):
                features.append(((input_ids.float() + positions + offset) % 17) / 8.0)
            hidden = torch.stack(features, dim=-1)
            return type("ToyBackboneOutput", (), {"last_hidden_state": hidden})

    class ToyModel(torch.nn.Module):
        base_model_prefix = "backbone"

        def __init__(self, name_or_path: str) -> None:
            super().__init__()
            self.name_or_path = name_or_path
            self.config = type(
                "ToyConfig",
                (),
                {
                    "hidden_size": 8,
                    "vocab_size": 2048,
                    "max_position_embeddings": 4096,
                },
            )()
            self.backbone = ToyBackbone(self.config.hidden_size)
            self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            torch.nn.init.uniform_(self.lm_head.weight, -0.05, 0.05)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
            backbone_output = self.backbone(input_ids, attention_mask=attention_mask, use_cache=use_cache)
            hidden = backbone_output.last_hidden_state
            logits = self.lm_head(hidden)
            if output_hidden_states:
                hidden_states = tuple(hidden for _ in range(2))
                return type("ToyOutput", (), {"hidden_states": hidden_states, "logits": logits})
            return type("ToyOutput", (), {"logits": logits})

        def generate(self, input_ids, max_new_tokens=48, do_sample=False, pad_token_id=0):
            tokenizer = ToyTokenizer()
            prompt_text = tokenizer.decode(input_ids[0].tolist())
            completion_text = _toy_completion_text(prompt_text)[:max_new_tokens]
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
            full_ids = input_ids[0].tolist() + completion_ids
            return torch.tensor([full_ids], dtype=torch.long, device=input_ids.device)

    tokenizer = ToyTokenizer()
    model = ToyModel(model_ref)
    model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device, "model_ref": model_ref}


def _resolve_pretrained_ref(model_ref: str) -> tuple[str, dict[str, Any]]:
    path_obj = Path(model_ref).expanduser()
    if path_obj.exists():
        return str(path_obj.resolve()), {"local_files_only": True}
    return model_ref, {}


def _resolve_torch_dtype(device: str, dtype_name: str):
    import torch

    normalized = (dtype_name or "auto").strip().lower()
    if normalized == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16
        return torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(normalized)


@lru_cache(maxsize=8)
def _load_tokenizer(model_ref: str):
    from transformers import AutoTokenizer

    resolved_ref, kwargs = _resolve_pretrained_ref(model_ref)
    tokenizer = AutoTokenizer.from_pretrained(resolved_ref, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tokenizer_for_model(model_ref: str):
    if model_ref.startswith("toy://"):
        return ToyTokenizer()
    return _load_tokenizer(model_ref)


@lru_cache(maxsize=4)
def _load_eval_bundle(
    model_ref: str,
    device: str,
    dtype_name: str,
    require_flash_attention: bool,
) -> tuple[Any, Any, str]:
    from transformers import AutoModelForCausalLM

    from ..protocol.constants import ATTN_IMPLEMENTATION
    from .flash_attention import require_flash_attention_2

    resolved_ref, kwargs = _resolve_pretrained_ref(model_ref)
    tokenizer = load_tokenizer_for_model(model_ref)
    model_kwargs = dict(kwargs)
    model_kwargs["low_cpu_mem_usage"] = True
    torch_dtype = _resolve_torch_dtype(device, dtype_name)
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    if require_flash_attention:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    model = AutoModelForCausalLM.from_pretrained(resolved_ref, **model_kwargs)
    model.to(device)
    model.eval()
    if require_flash_attention:
        require_flash_attention_2(model)
    return model, tokenizer, resolved_ref


# Global model-bundle cache + mutation registry.
#
# The verifier calls load_model_bundle() from inside verify_completion, which
# runs once per completion per window — in a 4-validator 8-completions-per-
# window mesh that's 32 loads/window. Without caching, each load is a 6GB
# bfloat16 read of the Qwen2.5-3B weights; with caching it's a dict lookup.
#
# The second purpose of the cache is to support in-place weight mutation by
# the closed-loop PolicyConsumer: when a delta bundle is applied, we mutate
# each cached bundle's model tensors so subsequent verify_completion calls
# run against the updated weights. The ReloadingPolicyApplier on the miner
# side and the validator-side hook (see cli.py) both call into
# apply_delta_to_cached_bundles() below.

_BUNDLE_CACHE: dict[tuple, dict[str, Any]] = {}


def _bundle_cache_key(
    model_ref: str,
    device: str,
    trainable: bool,
    dtype_name: str,
    require_flash_attention: bool,
) -> tuple:
    return (model_ref, device, bool(trainable), dtype_name, bool(require_flash_attention))


def load_model_bundle(
    model_ref: str,
    device: str = "cpu",
    trainable: bool = False,
    dtype_name: str = "auto",
    require_flash_attention: bool = False,
) -> dict[str, Any]:
    # Fast path: return a cached bundle if this exact (ref, device, dtype,
    # trainable, flash_attn) combination has been loaded.  Prevents the
    # per-completion 6GB re-load that was previously happening on the
    # validator side.
    key = _bundle_cache_key(
        model_ref, device, trainable, dtype_name, require_flash_attention
    )
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached

    if model_ref.startswith("toy://"):
        bundle = _build_toy_bundle(model_ref, device)
        _BUNDLE_CACHE[key] = bundle
        return bundle
    if not trainable:
        model, tokenizer, resolved_ref = _load_eval_bundle(
            model_ref, device, dtype_name, require_flash_attention
        )
        bundle = {"model": model, "tokenizer": tokenizer, "device": device, "model_ref": resolved_ref}
        _BUNDLE_CACHE[key] = bundle
        return bundle
    from transformers import AutoModelForCausalLM

    from ..protocol.constants import ATTN_IMPLEMENTATION
    from .flash_attention import require_flash_attention_2

    resolved_ref, kwargs = _resolve_pretrained_ref(model_ref)
    tokenizer = load_tokenizer_for_model(model_ref)
    model_kwargs = dict(kwargs)
    torch_dtype = _resolve_torch_dtype(device, dtype_name)
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    if require_flash_attention:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    model = AutoModelForCausalLM.from_pretrained(resolved_ref, **model_kwargs)
    model.to(device)
    if require_flash_attention:
        require_flash_attention_2(model)
    bundle = {"model": model, "tokenizer": tokenizer, "device": device, "model_ref": resolved_ref}
    _BUNDLE_CACHE[key] = bundle
    return bundle


def clear_bundle_cache() -> None:
    """Drop every cached bundle. Intended for tests + explicit operator reset."""
    _BUNDLE_CACHE.clear()


def cached_bundles() -> list[dict[str, Any]]:
    """Return a list of every currently-cached bundle dict.

    The PolicyConsumer applier uses this to mutate every running model
    at once when a delta is applied. A mesh validator that serves
    multiple model_refs will have multiple cached bundles; each gets
    its own delta applied (scoped by tensor-name match).
    """
    return list(_BUNDLE_CACHE.values())


def apply_delta_to_cached_bundles(delta_bundle) -> int:
    """Apply a DeltaBundle in-place to every cached bundle's model.

    Delegates to the ReloadingPolicyApplier via a per-bundle dispatch
    so the same fp32-dequant path is used across miner + validator.
    Returns the number of bundles mutated.
    """
    # Lazy import: policy_applier imports modeling (circular), so keep
    # the reference chain broken until this is called.
    from .policy_applier import ReloadingPolicyApplier
    from .policy_consumer import LoadedDelta

    mutated = 0
    for bundle in cached_bundles():
        class _EngineShim:
            """Adapter: ReloadingPolicyApplier expects an object with
            ``.model`` — our cached bundle dict has the model under
            the "model" key."""
            def __init__(self, b):
                self.model = b["model"]
        applier = ReloadingPolicyApplier(_EngineShim(bundle))
        stub = LoadedDelta(
            run_id="",
            window_id=0,
            merkle_root_hex="",
            raw_manifest_bytes=b"",
            shard_digests=(),
            extra={"bundle": delta_bundle},
        )
        applier(stub)
        mutated += 1
    return mutated


def save_model_bundle(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def compute_completion_logprobs(
    model,
    full_token_ids: list[int],
    prompt_length: int,
    device: str,
) -> list[float]:
    import torch

    if len(full_token_ids) <= prompt_length:
        return []

    input_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids).logits[0]

    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    completion_target_ids = input_ids[0, prompt_length:]
    start = prompt_length - 1
    end = start + completion_target_ids.shape[0]
    selected = log_probs[start:end, :].gather(1, completion_target_ids.unsqueeze(1)).squeeze(1)
    return [float(x) for x in selected.detach().cpu().tolist()]


def decode_completion(tokenizer, completion_token_ids: list[int]) -> str:
    return tokenizer.decode(completion_token_ids, skip_special_tokens=True)
