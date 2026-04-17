from reliquary_inference.validator.copycat import detect_index_copycats
from reliquary_inference.validator.weights import compute_weights


def test_detect_index_copycats_rejects_later_uploader() -> None:
    rejected = detect_index_copycats(
        {
            "miner-a": {"indices": {10, 11}, "upload_time": 1.0},
            "miner-b": {"indices": {11, 12}, "upload_time": 5.0},
        }
    )
    assert rejected == {"miner-b": {11}}


def test_compute_weights_prefers_unique_valid_work() -> None:
    weights = compute_weights(
        {
            "miner-a": {"unique": 4, "valid": 4},
            "miner-b": {"unique": 2, "valid": 2},
        }
    )
    assert weights["miner-a"] > weights["miner-b"]
