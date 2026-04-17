"""Stage 4: sketch-proof replay.

Runs the validator's forward pass on the claimed tokens, then replays the
sketch commitment at CHALLENGE_K deterministically-selected positions. Any
per-position failure short-circuits the rollout.

Upstream (pre-pipeline) must verify the commit signature so this stage trusts
``payload["commitments"]`` to be miner-bound.
"""

from __future__ import annotations

from .base import RejectReason, StageContext, StageResult, accept, reject


class ProofStage:
    name: str = "proof"

    def check(self, context: StageContext) -> StageResult:
        import torch

        from ...protocol.constants import CHALLENGE_K, LAYER_INDEX
        from ...protocol.crypto import indices_from_root
        from ...protocol.sketch_verifier import SketchProofVerifier
        from ...shared.forward import forward_single_layer
        from ...shared.hf_compat import resolve_hidden_size

        model = context.model
        payload = context.payload
        tokens = payload.get("tokens", [])
        commitments = payload.get("commitments", [])
        randomness = payload.get("randomness", context.randomness)

        if model is None:
            return reject(
                self.name,
                RejectReason.PIPELINE_MODEL_NOT_LOADED,
                {"reason": "stage_requires_model"},
            )

        hidden_dim = resolve_hidden_size(model)
        verifier = SketchProofVerifier(hidden_dim=hidden_dim)
        device = next(model.parameters()).device
        r_vec = verifier.generate_r_vec(randomness).to(device)

        input_ids = torch.tensor([tokens], device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            hidden_states, _ = forward_single_layer(
                model, input_ids, attention_mask, LAYER_INDEX
            )
        hidden_states = hidden_states[0]

        seq_len = len(tokens)
        indices = indices_from_root(tokens, randomness, seq_len, min(CHALLENGE_K, seq_len))

        checked = 0
        passed = 0
        last_diag: dict | None = None
        for idx in indices:
            if idx >= len(commitments):
                continue
            checked += 1
            valid, diag = verifier.verify_commitment(
                hidden_states[idx], commitments[idx], r_vec, seq_len, idx
            )
            last_diag = diag
            if valid:
                passed += 1

        if checked == 0:
            return reject(
                self.name,
                RejectReason.PROOF_NO_POSITIONS_CHECKED,
                {"commitments": len(commitments), "seq_len": seq_len},
            )

        context.extras["proof_summary"] = last_diag or {}
        context.extras["checked_positions"] = checked
        context.extras["passed_positions"] = passed
        if passed != checked:
            return reject(
                self.name,
                RejectReason.PROOF_SKETCH_MISMATCH,
                {"checked": checked, "passed": passed, "last_diag": last_diag},
            )

        return accept(self.name, {"checked": checked, "passed": passed})
