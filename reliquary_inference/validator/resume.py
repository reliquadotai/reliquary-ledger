"""Resume-from-checkpoint source resolution for the validator.

Phase 1.6 of the parallel-harvest plan: lets operators boot a validator
pointed at a specific HF revision (``sha:<hex>``) or a local snapshot
directory (``path:<dir>``) instead of the default ``cfg["model_ref"]``.

Parallel-work credit: romain13190/reliquary@1801544 + @e5f81ad. The
upstream flow operates on a single-process trainer; the resolver shape
adopted here (parse strict, dispatch by prefix, fail loud on bad input)
follows that pattern. The actual download/path-validation logic is
independent.

Usage:

    source = parse_resume_source("sha:1234abcd...")
    local_dir = resolve_resume_source(
        source,
        repo_id=cfg["model_ref"],
    )
    cfg["model_ref"] = str(local_dir)

The resolver is dependency-injectable for tests: pass a fake
``snapshot_download`` to avoid hitting Hugging Face during unit tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class InvalidResumeSourceError(ValueError):
    """Raised when ``--resume-from`` is parseable as neither sha: nor path:."""


@dataclass(frozen=True)
class ShaSource:
    """A specific revision of the configured HF repo, identified by hex."""

    revision: str


@dataclass(frozen=True)
class PathSource:
    """A local directory containing a HF-format model snapshot."""

    path: Path


ResumeSource = ShaSource | PathSource


_SHA_PREFIX = "sha:"
_PATH_PREFIX = "path:"
# 6+ hex chars covers short HF revisions (typically 7) and full-length git SHAs.
_SHA_RE = re.compile(r"^[0-9a-fA-F]{6,64}$")


def parse_resume_source(raw: str) -> ResumeSource:
    """Parse a ``--resume-from`` source string. Strict; raises on bad input.

    Accepted shapes:
    - ``sha:<6-64 hex chars>``  — checkpoint revision on the configured HF repo
    - ``path:<dir>``            — local directory holding a HF-format snapshot

    No other prefixes are accepted. Whitespace around the prefix is stripped;
    whitespace inside the value is treated as part of the value (and almost
    certainly a typo, hence the strict regex/path validation).
    """
    if not isinstance(raw, str):
        raise InvalidResumeSourceError(f"resume-from source must be a string, got {type(raw)!r}")
    stripped = raw.strip()
    if not stripped:
        raise InvalidResumeSourceError("resume-from source is empty")
    if stripped.startswith(_SHA_PREFIX):
        rev = stripped[len(_SHA_PREFIX):].strip()
        if not _SHA_RE.match(rev):
            raise InvalidResumeSourceError(
                f"resume-from sha must be 6-64 hex chars, got {rev!r}"
            )
        return ShaSource(revision=rev)
    if stripped.startswith(_PATH_PREFIX):
        raw_path = stripped[len(_PATH_PREFIX):].strip()
        if not raw_path:
            raise InvalidResumeSourceError("resume-from path is empty")
        path = Path(raw_path).expanduser()
        # No existence check here — the resolver does that, with a clearer error.
        return PathSource(path=path)
    raise InvalidResumeSourceError(
        f"resume-from must start with 'sha:' or 'path:', got {stripped!r}"
    )


class _SnapshotDownloader(Protocol):
    def __call__(self, *, repo_id: str, revision: str) -> str: ...


def _default_snapshot_download(*, repo_id: str, revision: str) -> str:
    """Production snapshot_download — wraps huggingface_hub if installed.

    Kept as a small wrapper so tests can substitute a fake without needing
    huggingface_hub installed.
    """
    try:
        from huggingface_hub import snapshot_download as _real
    except ImportError as exc:  # pragma: no cover - exercised when hub missing
        raise RuntimeError(
            "resume-from sha:<rev> requires the huggingface_hub package; "
            "install with `pip install huggingface_hub`"
        ) from exc
    return _real(repo_id=repo_id, revision=revision)


def resolve_resume_source(
    source: ResumeSource,
    *,
    repo_id: str,
    snapshot_download: _SnapshotDownloader | None = None,
) -> Path:
    """Resolve a ``ResumeSource`` to a local directory containing the model.

    For ``ShaSource``, this calls ``snapshot_download(repo_id, revision)``
    using ``repo_id`` as the canonical HF repo. For ``PathSource``, this
    validates that the directory exists and is non-empty.

    ``snapshot_download`` is dependency-injected so unit tests don't need
    huggingface_hub installed. In production, leave it ``None`` and the
    real ``huggingface_hub.snapshot_download`` is used.
    """
    if isinstance(source, ShaSource):
        if not repo_id:
            raise InvalidResumeSourceError(
                "resume-from sha requires a repo_id (RELIQUARY_INFERENCE_MODEL_REF)"
            )
        downloader = snapshot_download or _default_snapshot_download
        local_dir = Path(downloader(repo_id=repo_id, revision=source.revision))
        return local_dir
    # PathSource
    if not source.path.exists():
        raise InvalidResumeSourceError(
            f"resume-from path does not exist: {source.path}"
        )
    if not source.path.is_dir():
        raise InvalidResumeSourceError(
            f"resume-from path is not a directory: {source.path}"
        )
    return source.path


class ChecksumMismatchError(Exception):
    """Raised when the resolved snapshot's checksum does not match the
    operator-supplied ``--checksum-expected`` value.

    The audit (#11) recommends comparing the resumed weights against
    an on-chain registry (e.g. an attestation merkle_root). When an
    operator passes ``--checksum-expected sha256:<hex>``, the resume
    flow computes a canonical digest over the resolved model
    directory and refuses to advance unless the digest matches.
    """


def compute_resume_checksum(local_path: Path) -> str:
    """Compute a canonical sha256 over the resolved model snapshot.

    Algorithm:
      1. Enumerate all ``*.safetensors`` files under ``local_path``,
         sorted by relative path for determinism.
      2. For each file, append ``"<rel_path>:<sha256_of_bytes>\\n"``
         to a canonical string.
      3. Final digest = ``sha256(canonical_string).hexdigest()``.

    No safetensors files → ``""`` (caller should treat as a hard
    error since there's nothing to verify).
    """
    import hashlib

    base = Path(local_path)
    files = sorted(base.rglob("*.safetensors"))
    if not files:
        return ""
    canonical_lines: list[str] = []
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        rel = f.relative_to(base).as_posix()
        canonical_lines.append(f"{rel}:{h.hexdigest()}")
    canonical = "\n".join(canonical_lines)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalise_expected_checksum(raw: str) -> str:
    """Strip an optional ``sha256:`` prefix and lowercase the hex.

    Operator may pass either ``--checksum-expected sha256:abc…`` or
    just the hex; both forms accepted.
    """
    cleaned = raw.strip().lower()
    if cleaned.startswith("sha256:"):
        cleaned = cleaned[len("sha256:") :]
    return cleaned


def apply_resume_from(
    cfg: dict,
    raw_source: str,
    *,
    snapshot_download: _SnapshotDownloader | None = None,
    expected_checksum: str | None = None,
) -> Path:
    """Parse + resolve a ``--resume-from`` source and override ``cfg["model_ref"]``.

    Mutates ``cfg`` in place; returns the resolved local path so callers
    can log or further inspect it. Raises ``InvalidResumeSourceError`` if
    the source is malformed; raises ``RuntimeError`` if a sha source needs
    huggingface_hub but it isn't installed.

    When ``expected_checksum`` is supplied, the resolved snapshot is
    digested via :func:`compute_resume_checksum` and compared; on
    mismatch raises :class:`ChecksumMismatchError` BEFORE the cfg
    mutation lands (so a poisoned snapshot can't be silently
    activated). Closes audit finding #11 (``--resume-from`` poisoning).
    """
    source = parse_resume_source(raw_source)
    repo_id = str(cfg.get("model_ref", "")).strip()
    local_path = resolve_resume_source(
        source,
        repo_id=repo_id,
        snapshot_download=snapshot_download,
    )
    if expected_checksum is not None:
        expected_hex = _normalise_expected_checksum(expected_checksum)
        if not expected_hex:
            raise ChecksumMismatchError(
                "expected_checksum is empty; pass sha256:<hex> or <hex>"
            )
        actual_hex = compute_resume_checksum(local_path)
        if not actual_hex:
            raise ChecksumMismatchError(
                f"resume snapshot at {local_path} contains no .safetensors "
                "files — refusing to apply unverified weights"
            )
        if actual_hex != expected_hex:
            raise ChecksumMismatchError(
                f"checksum mismatch on resume snapshot at {local_path}: "
                f"expected sha256:{expected_hex}, got sha256:{actual_hex}. "
                "Refusing to swap to potentially-poisoned weights."
            )
    cfg["model_ref"] = str(local_path)
    return local_path


__all__ = [
    "ChecksumMismatchError",
    "InvalidResumeSourceError",
    "PathSource",
    "ResumeSource",
    "ShaSource",
    "apply_resume_from",
    "compute_resume_checksum",
    "parse_resume_source",
    "resolve_resume_source",
]
