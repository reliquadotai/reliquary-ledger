import torch

from reliquary_inference.protocol.crypto import (
    indices_from_root,
    prf,
    r_vec_from_randomness,
)
from reliquary_inference.protocol.sketch_verifier import SketchProofVerifier


def test_prf_is_deterministic() -> None:
    left = prf(b"label", b"alpha", out_bytes=64)
    right = prf(b"label", b"alpha", out_bytes=64)
    assert left == right
    assert len(left) == 64


def test_r_vec_from_randomness_is_stable() -> None:
    vector = r_vec_from_randomness("deadbeef", 16)
    assert vector.shape[0] == 16
    assert torch.equal(vector, r_vec_from_randomness("deadbeef", 16))


def test_indices_from_root_is_stable() -> None:
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    indices = indices_from_root(tokens, "cafebabe", len(tokens), 4)
    assert indices == indices_from_root(tokens, "cafebabe", len(tokens), 4)
    assert indices == sorted(indices)


def test_sketch_verifier_roundtrip() -> None:
    torch.manual_seed(42)
    hidden = torch.randn(32, 64)
    verifier = SketchProofVerifier(hidden_dim=64)
    r_vec = verifier.generate_r_vec("0123abcd")
    commitments = verifier.create_commitments_batch(hidden, r_vec)
    for index in range(8):
        valid, _ = verifier.verify_commitment(hidden[index], commitments[index], r_vec, hidden.shape[0], index)
        assert valid is True


def test_sketch_verifier_caps_topk_to_hidden_dim() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(4, 8)
    verifier = SketchProofVerifier(hidden_dim=8)
    assert verifier.topk == 8
    r_vec = verifier.generate_r_vec("feedface")
    commitments = verifier.create_commitments_batch(hidden, r_vec)
    assert len(commitments) == hidden.shape[0]


def test_sketch_verifier_verify_commitment_on_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(11)
    hidden = torch.randn(4, 16, device="cuda")
    verifier = SketchProofVerifier(hidden_dim=16)
    r_vec = verifier.generate_r_vec("abc123").to("cuda")
    commitments = verifier.create_commitments_batch(hidden, r_vec)
    valid, _ = verifier.verify_commitment(hidden[0], commitments[0], r_vec, hidden.shape[0], 0)
    assert valid is True
