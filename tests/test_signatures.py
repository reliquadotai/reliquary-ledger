from reliquary_inference.protocol.signatures import (
    LOCAL_HMAC_SCHEME,
    sign_commit_binding,
    verify_commit_signature,
)


def test_local_hmac_signature_roundtrip() -> None:
    signature, signer_id, scheme = sign_commit_binding(
        [1, 2, 3],
        "deadbeef",
        "toy://local-inference-v1",
        -1,
        [{"sketch": 10}, {"sketch": 20}],
        scheme=LOCAL_HMAC_SCHEME,
        signer_id="miner-a",
        secret="secret-a",
    )
    commit = {
        "tokens": [1, 2, 3],
        "commitments": [{"sketch": 10}, {"sketch": 20}],
        "signature": signature,
        "signature_scheme": scheme,
        "proof_version": "v5",
        "beacon": {"randomness": "deadbeef"},
        "model": {"name": "toy://local-inference-v1", "layer_index": -1},
    }
    assert signer_id == "miner-a"
    assert verify_commit_signature(commit, wallet_address="miner-a", secret="secret-a") is True
    assert verify_commit_signature(commit, wallet_address="miner-a", secret="wrong") is False
