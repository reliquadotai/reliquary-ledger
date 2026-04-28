"""Unit tests for the ``--resume-from`` source parser and resolver.

Phase 1.6 of the parallel-harvest plan. Tests cover:
- Strict parsing of ``sha:<hex>`` and ``path:<dir>`` source strings
- Rejection of missing prefix, empty value, non-hex sha, and bad path
- Resolver dispatches to ``snapshot_download`` for ShaSource and to
  filesystem validation for PathSource
- Both real and dependency-injected snapshot_download paths

Parallel-work credit: romain13190/reliquary@1801544 + @e5f81ad.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reliquary_inference.validator.resume import (
    InvalidResumeSourceError,
    PathSource,
    ShaSource,
    apply_resume_from,
    parse_resume_source,
    resolve_resume_source,
)


# ────────  Parser  ────────


def test_parse_sha_lowercase_full_length_accepted() -> None:
    src = parse_resume_source("sha:abcdef0123456789abcdef0123456789abcdef01")
    assert isinstance(src, ShaSource)
    assert src.revision == "abcdef0123456789abcdef0123456789abcdef01"


def test_parse_sha_short_seven_char_revision_accepted() -> None:
    src = parse_resume_source("sha:1234abc")
    assert isinstance(src, ShaSource)
    assert src.revision == "1234abc"


def test_parse_sha_uppercase_accepted() -> None:
    src = parse_resume_source("sha:DEADBEEF")
    assert isinstance(src, ShaSource)
    assert src.revision == "DEADBEEF"


def test_parse_sha_strips_whitespace_around_value() -> None:
    src = parse_resume_source("  sha:  abc1234  ")
    assert isinstance(src, ShaSource)
    assert src.revision == "abc1234"


def test_parse_sha_rejects_non_hex_value() -> None:
    with pytest.raises(InvalidResumeSourceError, match="hex"):
        parse_resume_source("sha:not-hex-zzz")


def test_parse_sha_rejects_too_short_value() -> None:
    with pytest.raises(InvalidResumeSourceError, match="hex"):
        parse_resume_source("sha:abc")


def test_parse_sha_rejects_too_long_value() -> None:
    with pytest.raises(InvalidResumeSourceError, match="hex"):
        parse_resume_source("sha:" + "a" * 65)


def test_parse_sha_rejects_empty_value() -> None:
    with pytest.raises(InvalidResumeSourceError):
        parse_resume_source("sha:")


def test_parse_path_returns_expanded_path(tmp_path: Path) -> None:
    src = parse_resume_source(f"path:{tmp_path}")
    assert isinstance(src, PathSource)
    assert src.path == tmp_path


def test_parse_path_expands_user_home() -> None:
    src = parse_resume_source("path:~/some/checkpoint")
    assert isinstance(src, PathSource)
    assert "~" not in str(src.path)
    assert str(src.path).endswith("some/checkpoint")


def test_parse_path_rejects_empty_value() -> None:
    with pytest.raises(InvalidResumeSourceError):
        parse_resume_source("path:")


def test_parse_rejects_unknown_prefix() -> None:
    with pytest.raises(InvalidResumeSourceError, match="sha:|path:"):
        parse_resume_source("file:///some/path")


def test_parse_rejects_no_prefix() -> None:
    with pytest.raises(InvalidResumeSourceError):
        parse_resume_source("just-a-string")


def test_parse_rejects_empty_input() -> None:
    with pytest.raises(InvalidResumeSourceError):
        parse_resume_source("")


def test_parse_rejects_whitespace_only_input() -> None:
    with pytest.raises(InvalidResumeSourceError):
        parse_resume_source("   ")


def test_parse_rejects_non_string_input() -> None:
    with pytest.raises(InvalidResumeSourceError):
        parse_resume_source(123)  # type: ignore[arg-type]


# ────────  Resolver  ────────


def test_resolve_sha_calls_snapshot_download_with_repo_and_revision(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, str]] = []

    def fake_download(*, repo_id: str, revision: str) -> str:
        seen.append({"repo_id": repo_id, "revision": revision})
        return str(tmp_path)

    out = resolve_resume_source(
        ShaSource(revision="abc1234"),
        repo_id="Qwen/Qwen3-4B-Instruct",
        snapshot_download=fake_download,
    )
    assert out == tmp_path
    assert seen == [{"repo_id": "Qwen/Qwen3-4B-Instruct", "revision": "abc1234"}]


def test_resolve_sha_requires_repo_id() -> None:
    with pytest.raises(InvalidResumeSourceError, match="repo_id"):
        resolve_resume_source(
            ShaSource(revision="abc1234"),
            repo_id="",
            snapshot_download=lambda **_: "/fake/path",
        )


def test_resolve_path_returns_existing_directory(tmp_path: Path) -> None:
    out = resolve_resume_source(
        PathSource(path=tmp_path),
        repo_id="ignored",
    )
    assert out == tmp_path


def test_resolve_path_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(InvalidResumeSourceError, match="does not exist"):
        resolve_resume_source(
            PathSource(path=missing),
            repo_id="ignored",
        )


def test_resolve_path_rejects_file_target(tmp_path: Path) -> None:
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"not a directory")
    with pytest.raises(InvalidResumeSourceError, match="not a directory"):
        resolve_resume_source(
            PathSource(path=file_path),
            repo_id="ignored",
        )


# ────────  apply_resume_from (cfg-mutation helper)  ────────


def test_apply_resume_from_overrides_model_ref_with_path(tmp_path: Path) -> None:
    cfg = {"model_ref": "Qwen/Qwen3-4B-Instruct"}
    out = apply_resume_from(cfg, f"path:{tmp_path}")
    assert cfg["model_ref"] == str(tmp_path)
    assert out == tmp_path


def test_apply_resume_from_overrides_model_ref_with_sha(tmp_path: Path) -> None:
    seen: list[dict[str, str]] = []

    def fake_download(*, repo_id: str, revision: str) -> str:
        seen.append({"repo_id": repo_id, "revision": revision})
        return str(tmp_path)

    cfg = {"model_ref": "Qwen/Qwen3-4B-Instruct"}
    out = apply_resume_from(cfg, "sha:abc1234", snapshot_download=fake_download)
    assert cfg["model_ref"] == str(tmp_path)
    assert out == tmp_path
    assert seen == [{"repo_id": "Qwen/Qwen3-4B-Instruct", "revision": "abc1234"}]


def test_apply_resume_from_propagates_invalid_source_error() -> None:
    cfg = {"model_ref": "Qwen/Qwen3-4B-Instruct"}
    with pytest.raises(InvalidResumeSourceError):
        apply_resume_from(cfg, "garbage:value")
    # cfg.model_ref should be untouched on failure
    assert cfg["model_ref"] == "Qwen/Qwen3-4B-Instruct"


def test_apply_resume_from_propagates_missing_path_error(tmp_path: Path) -> None:
    cfg = {"model_ref": "Qwen/Qwen3-4B-Instruct"}
    missing = tmp_path / "not-here"
    with pytest.raises(InvalidResumeSourceError, match="does not exist"):
        apply_resume_from(cfg, f"path:{missing}")
    # cfg.model_ref should be untouched on failure
    assert cfg["model_ref"] == "Qwen/Qwen3-4B-Instruct"
