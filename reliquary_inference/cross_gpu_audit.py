"""Cross-GPU determinism harness for the proof sketch.

Produces a structured JSON document listing sketch outputs for a fixed
set of ``(seed, scale, hidden_dim)`` inputs plus metadata identifying the
GPU that generated it. Running this on multiple GPUs and diffing the
outputs answers the spec question: do independent GPU instances agree
bit-exactly on sketch values?

Spec reference: private/reliquary-plan/notes/spec-proof-protocol.md
Security Properties (cross-GPU determinism) + 02_TIER2_PRD.md Epic 6
cross-GPU acceptance test.
"""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .protocol.sketch_verifier import SketchProofVerifier

DEFAULT_SEEDS = (0, 1, 7, 42, 100, 256, 1024, 2048, 4096, 9999)
DEFAULT_SCALES = (1.0, 2.5, 8.0)
DEFAULT_HIDDEN_DIMS = (128, 256, 512)
DEFAULT_RANDOMNESS_HEX = "00000000000000000000000000000000000000000000000000000000deadbeef"


@dataclass(frozen=True)
class SketchSample:
    seed: int
    scale: float
    hidden_dim: int
    sketch: int


@dataclass
class CrossGpuReport:
    timestamp: str
    host: str
    torch_version: str
    cuda_available: bool
    gpu_name: str
    gpu_memory_total: int
    cuda_capability: tuple[int, int] | None
    randomness_hex: str
    samples: list[SketchSample] = field(default_factory=list)
    samples_digest: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "host": self.host,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_total": self.gpu_memory_total,
            "cuda_capability": list(self.cuda_capability) if self.cuda_capability else None,
            "randomness_hex": self.randomness_hex,
            "samples_digest": self.samples_digest,
            "samples": [asdict(s) for s in self.samples],
        }


def _digest_samples(samples: Iterable[SketchSample]) -> str:
    hasher = hashlib.sha256()
    for s in samples:
        hasher.update(f"{s.seed}|{s.scale}|{s.hidden_dim}|{s.sketch}\n".encode("utf-8"))
    return hasher.hexdigest()


def collect_samples(
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    scales: Iterable[float] = DEFAULT_SCALES,
    hidden_dims: Iterable[int] = DEFAULT_HIDDEN_DIMS,
    randomness_hex: str = DEFAULT_RANDOMNESS_HEX,
) -> list[SketchSample]:
    """Generate a deterministic fixed-seed sample set."""
    import torch

    samples: list[SketchSample] = []
    for hidden_dim in hidden_dims:
        verifier = SketchProofVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness_hex)
        for seed in seeds:
            for scale in scales:
                gen = torch.Generator().manual_seed(seed)
                hidden = torch.randn(hidden_dim, generator=gen, dtype=torch.float32) * scale
                sketch = int(verifier.create_commitment(hidden, r_vec)["sketch"])
                samples.append(SketchSample(seed=seed, scale=scale, hidden_dim=hidden_dim, sketch=sketch))
    return samples


def _collect_gpu_metadata():
    import torch

    gpu_name = "cpu"
    gpu_memory_total = 0
    cuda_capability = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory_total = int(props.total_memory)
        cuda_capability = (props.major, props.minor)
    return gpu_name, gpu_memory_total, cuda_capability


def run_cross_gpu_campaign(
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    scales: Iterable[float] = DEFAULT_SCALES,
    hidden_dims: Iterable[int] = DEFAULT_HIDDEN_DIMS,
    randomness_hex: str = DEFAULT_RANDOMNESS_HEX,
    device: str | None = None,
) -> CrossGpuReport:
    """Run the fixed-seed sample sweep and return a structured report.

    ``device`` optionally forces CPU with ``"cpu"`` (useful as a baseline).
    Default behavior uses the active torch device; tensor seed state is on
    CPU for cross-device reproducibility.
    """
    from datetime import datetime, timezone

    import torch

    if device is not None:
        torch.set_default_device(device)

    samples = collect_samples(
        seeds=seeds,
        scales=scales,
        hidden_dims=hidden_dims,
        randomness_hex=randomness_hex,
    )
    gpu_name, gpu_memory_total, cuda_capability = _collect_gpu_metadata()
    report = CrossGpuReport(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        host=socket.gethostname(),
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        gpu_name=gpu_name,
        gpu_memory_total=gpu_memory_total,
        cuda_capability=cuda_capability,
        randomness_hex=randomness_hex,
        samples=samples,
    )
    report.samples_digest = _digest_samples(samples)
    return report


def compare_reports(reports: list[CrossGpuReport]) -> dict:
    """Structural + bit-level diff across a list of reports from different GPUs.

    Returns a dict with:
      - ``matching_digests``: True iff every report's samples_digest is identical
        (implies bit-exact sketch agreement across the whole sample sweep).
      - ``digest_by_host``: {host → samples_digest}.
      - ``mismatches``: list of (seed, scale, hidden_dim, [sketch_per_host])
        for input keys that disagree between reports.
    """
    if not reports:
        raise ValueError("need at least one report")

    digests = {r.host: r.samples_digest for r in reports}
    matching = len(set(digests.values())) == 1

    indexed: dict[tuple[int, float, int], dict[str, int]] = {}
    for report in reports:
        for s in report.samples:
            key = (s.seed, s.scale, s.hidden_dim)
            indexed.setdefault(key, {})[report.host] = s.sketch

    mismatches: list[dict] = []
    for key, by_host in indexed.items():
        if len(set(by_host.values())) > 1:
            mismatches.append(
                {"seed": key[0], "scale": key[1], "hidden_dim": key[2], "sketches": dict(by_host)}
            )

    return {
        "matching_digests": matching,
        "digest_by_host": digests,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "total_samples_per_host": {r.host: len(r.samples) for r in reports},
    }


def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="reliquary-inference-cross-gpu-audit",
        description="Capture deterministic sketch samples for cross-GPU comparison.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help='force "cpu" for a CPU baseline')
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma-separated integer seeds",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SCALES),
        help="comma-separated float scales",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default=",".join(str(s) for s in DEFAULT_HIDDEN_DIMS),
        help="comma-separated integer hidden dims",
    )
    args = parser.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(","))
    scales = tuple(float(x) for x in args.scales.split(","))
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    report = run_cross_gpu_campaign(
        seeds=seeds,
        scales=scales,
        hidden_dims=hidden_dims,
        device=args.device,
    )
    payload = json.dumps(report.to_dict(), indent=2)
    if args.output:
        Path(args.output).write_text(payload)
        print(f"cross-gpu report written to {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
