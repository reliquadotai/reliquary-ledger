"""Unit tests for reliquary_inference.protocol.tokens pure helpers.

``protocol.tokens`` imports ``shared.hf_compat`` which requires
transformers. This module is skipped cleanly when transformers is
absent (matches the dev-without-GPU flow) but verifies the pure-python
helpers exhaustively when transformers is present.
"""

from __future__ import annotations

import hashlib
import struct

import pytest

pytest.importorskip(
    "transformers",
    reason="transformers required to import reliquary_inference.protocol.tokens",
)

from reliquary_inference.protocol.tokens import (
    _validate_token_ids,
    hash_tokens,
    int_to_bytes,
)


def test_int_to_bytes_big_endian() -> None:
    assert int_to_bytes(1) == b"\x00\x00\x00\x01"
    assert int_to_bytes(256) == b"\x00\x00\x01\x00"


def test_int_to_bytes_matches_struct() -> None:
    for value in (0, 1, 2**16, 2**31 - 1, 0xFFFFFFFF):
        assert int_to_bytes(value) == struct.pack(">I", value & 0xFFFFFFFF)


def test_int_to_bytes_truncates_upper_bits() -> None:
    assert int_to_bytes(2**32) == b"\x00\x00\x00\x00"


def test_hash_tokens_matches_manual_sha() -> None:
    tokens = [1, 2, 3, 4, 5]
    expected = hashlib.sha256(b"".join(int_to_bytes(t) for t in tokens)).digest()
    assert hash_tokens(tokens) == expected


def test_hash_tokens_length_is_32() -> None:
    assert len(hash_tokens([0, 1, 2])) == 32


def test_hash_tokens_empty() -> None:
    assert hash_tokens([]) == hashlib.sha256(b"").digest()


def test_hash_tokens_order_sensitive() -> None:
    assert hash_tokens([1, 2, 3]) != hash_tokens([3, 2, 1])


def test_validate_token_ids_within_bounds() -> None:
    assert _validate_token_ids([0, 1, 2, 100], vocab_size=101) is True


def test_validate_token_ids_rejects_negative() -> None:
    assert _validate_token_ids([0, -1, 2], vocab_size=100) is False


def test_validate_token_ids_rejects_above_vocab() -> None:
    assert _validate_token_ids([0, 1, 100], vocab_size=100) is False


def test_validate_token_ids_rejects_non_int() -> None:
    assert _validate_token_ids([0, "not-int", 1], vocab_size=100) is False  # type: ignore[list-item]


def test_validate_token_ids_empty_is_true() -> None:
    assert _validate_token_ids([], vocab_size=100) is True
