from __future__ import annotations

from typing import Any

import torch

from ..constants import CHALLENGE_K, PROOF_VERSION, LAYER_INDEX
from ..dataset.task_sources import build_task_source
from ..protocol.sketch_verifier import SketchProofVerifier
from ..protocol.signatures import verify_commit_signature
from ..protocol.tokens import hash_tokens, verify_tokens
from ..shared.forward import forward_single_layer
from ..shared.hf_compat import resolve_hidden_size
from ..shared.modeling import load_model_bundle


def verify_completion(
    *,
    cfg: dict[str, Any],
    completion: dict[str, Any],
    task_batch: dict[str, Any],
    seen_nonces: set[tuple[str, int]],
) -> dict[str, Any]:
    bundle = load_model_bundle(
        str(task_batch["payload"]["model_ref"]),
        device=str(cfg["device"]),
        dtype_name=str(cfg.get("load_dtype", "auto")),
    )
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    payload = completion["payload"]
    task_source = build_task_source(payload["task_source"])
    report = {
        "accepted": False,
        "hard_fail_reason": None,
        "soft_fail_reason": None,
        "checked_positions": 0,
        "passed_positions": 0,
        "task_source": payload["task_source"],
        "copycat_status": "pending",
        "signature_status": "pending",
        "task_binding_summary": {},
        "proof_summary": {},
    }
    if payload.get("proof_version") != PROOF_VERSION:
        report["hard_fail_reason"] = "invalid_proof_version"
        return report
    if not verify_tokens(payload["tokens"], model.config):
        report["hard_fail_reason"] = "invalid_tokens"
        return report
    nonce_key = (completion["producer_id"], int(payload["nonce"]))
    if nonce_key in seen_nonces:
        report["hard_fail_reason"] = "duplicate_nonce"
        return report
    seen_nonces.add(nonce_key)
    signature_ok = verify_commit_signature(
        {
            "tokens": payload["tokens"],
            "commitments": payload["commitments"],
            "signature": payload["signature"],
            "signature_scheme": payload.get("signature_scheme", "local_hmac"),
            "signer_id": payload.get("signer_id"),
            "beacon": {"randomness": payload["randomness"]},
            "model": {"name": payload["model_name"], "layer_index": payload["layer_index"]},
            "proof_version": payload["proof_version"],
        },
        wallet_address=str(payload.get("signer_id") or completion["producer_id"]),
        secret=cfg.get("signing_secret") if payload.get("signature_scheme") == "local_hmac" else None,
    )
    report["signature_status"] = "ok" if signature_ok else "invalid_signature"
    if not signature_ok:
        report["hard_fail_reason"] = "invalid_signature"
        return report
    task_binding_ok, binding_summary = task_source.verify_task_binding(completion, task_batch, tokenizer)
    report["task_binding_summary"] = binding_summary
    if not task_binding_ok:
        report["hard_fail_reason"] = binding_summary["reason"]
        return report
    hidden_dim = resolve_hidden_size(model)
    verifier = SketchProofVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(payload["randomness"]).to(next(model.parameters()).device)
    input_ids = torch.tensor([payload["tokens"]], device=next(model.parameters()).device)
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    with torch.no_grad():
        hidden_states, _ = forward_single_layer(model, input_ids, attention_mask, LAYER_INDEX)
    hidden_states = hidden_states[0]
    checked = 0
    passed = 0
    for idx in verifier_indices(payload["tokens"], payload["randomness"], len(payload["tokens"])):
        if idx >= len(payload["commitments"]):
            continue
        checked += 1
        valid, diag = verifier.verify_commitment(hidden_states[idx], payload["commitments"][idx], r_vec, len(payload["tokens"]), idx)
        if valid:
            passed += 1
        report["proof_summary"] = diag
    report["checked_positions"] = checked
    report["passed_positions"] = passed
    if checked == 0 or passed != checked:
        report["hard_fail_reason"] = "proof_failed"
        return report
    task = binding_summary["task"]
    semantic_result = task_source.evaluate_completion(completion, task)
    if not semantic_result["accepted"]:
        report["soft_fail_reason"] = semantic_result["reason"]
        return report
    report["accepted"] = True
    return report


def verifier_indices(tokens: list[int], randomness: str, seq_len: int) -> list[int]:
    from ..protocol.crypto import indices_from_root

    return indices_from_root(tokens, randomness, seq_len, min(CHALLENGE_K, seq_len))
