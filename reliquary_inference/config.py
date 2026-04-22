from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _git_sha() -> str:
    package_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _derived_local_signer(miner_id: str, secret: str) -> str:
    material = f"{miner_id}:{secret}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def load_config() -> dict[str, object]:
    state_root = _env_str("RELIQUARY_INFERENCE_STATE_ROOT", "./data")
    artifact_dir = _env_str("RELIQUARY_INFERENCE_ARTIFACT_DIR", f"{state_root}/artifacts")
    export_dir = _env_str("RELIQUARY_INFERENCE_EXPORT_DIR", f"{state_root}/exports")
    log_dir = _env_str("RELIQUARY_INFERENCE_LOG_DIR", f"{state_root}/logs")
    wallet_public_file = _env_str(
        "RELIQUARY_INFERENCE_WALLET_PUBLIC_FILE",
        f"{state_root}/bootstrap/wallet-public.json",
    )
    miner_id = _env_str("RELIQUARY_INFERENCE_MINER_ID", "local-miner")
    signing_secret = _env_str("RELIQUARY_INFERENCE_SIGNING_SECRET", "local-secret")
    return {
        "state_root": state_root,
        "artifact_dir": artifact_dir,
        "export_dir": export_dir,
        "log_dir": log_dir,
        "wallet_public_file": wallet_public_file,
        "model_ref": _env_str("RELIQUARY_INFERENCE_MODEL_REF", "toy://local-inference-v1"),
        "task_source": _env_str("RELIQUARY_INFERENCE_TASK_SOURCE", "reasoning_tasks"),
        "task_count": _env_int("RELIQUARY_INFERENCE_TASK_COUNT", 8),
        "poll_interval": _env_int("RELIQUARY_INFERENCE_POLL_INTERVAL", 10),
        "metrics_bind": _env_str("RELIQUARY_INFERENCE_METRICS_BIND", "127.0.0.1"),
        "metrics_port": _env_int("RELIQUARY_INFERENCE_METRICS_PORT", 9108),
        "metrics_refresh_interval": _env_int("RELIQUARY_INFERENCE_METRICS_REFRESH_INTERVAL", 15),
        "metrics_window_count": _env_int("RELIQUARY_INFERENCE_METRICS_WINDOW_COUNT", 10),
        "samples_per_task": _env_int("RELIQUARY_INFERENCE_SAMPLES_PER_TASK", 1),
        "max_new_tokens": _env_int("RELIQUARY_INFERENCE_MAX_NEW_TOKENS", 48),
        "device": _env_str("RELIQUARY_INFERENCE_DEVICE", "cpu"),
        "load_dtype": _env_str("RELIQUARY_INFERENCE_LOAD_DTYPE", "auto"),
        "miner_mode": _env_str("RELIQUARY_INFERENCE_MINER_MODE", "single_gpu_hf"),
        "vllm_base_url": _env_str("RELIQUARY_INFERENCE_VLLM_BASE_URL", ""),
        "miner_id": miner_id,
        "validator_id": _env_str("RELIQUARY_INFERENCE_VALIDATOR_ID", "local-validator"),
        "miner_ss58": _env_str("RELIQUARY_INFERENCE_MINER_SS58", ""),
        "validator_ss58": _env_str("RELIQUARY_INFERENCE_VALIDATOR_SS58", ""),
        "signature_scheme": _env_str("RELIQUARY_INFERENCE_SIGNATURE_SCHEME", "local_hmac"),
        "signing_secret": signing_secret,
        "local_signer_id": _env_str(
            "RELIQUARY_INFERENCE_LOCAL_SIGNER_ID",
            _derived_local_signer(miner_id, signing_secret),
        ),
        "storage_backend": _env_str("RELIQUARY_INFERENCE_STORAGE_BACKEND", "local"),
        "network": _env_str("RELIQUARY_INFERENCE_NETWORK", "local"),
        "chain_endpoint": _env_str(
            "RELIQUARY_INFERENCE_CHAIN_ENDPOINT",
            _env_str("BT_CHAIN_ENDPOINT", _env_str("BT_SUBTENSOR_CHAIN_ENDPOINT", "")),
        ),
        "audit_bucket": _env_str("RELIQUARY_INFERENCE_AUDIT_BUCKET", ""),
        "audit_endpoint_url": _env_str("RELIQUARY_INFERENCE_AUDIT_ENDPOINT_URL", ""),
        "audit_access_key_id": _env_str("RELIQUARY_INFERENCE_AUDIT_ACCESS_KEY_ID", ""),
        "audit_secret_access_key": _env_str("RELIQUARY_INFERENCE_AUDIT_SECRET_ACCESS_KEY", ""),
        "public_audit_base_url": _env_str(
            "RELIQUARY_INFERENCE_PUBLIC_AUDIT_BASE_URL",
            "",
        ),
        "expose_public_artifact_urls": _env_bool(
            "RELIQUARY_INFERENCE_EXPOSE_PUBLIC_ARTIFACT_URLS",
            False,
        ),
        "audit_prefix": _env_str("RELIQUARY_INFERENCE_AUDIT_PREFIX", "audit"),
        "netuid": _env_int("RELIQUARY_INFERENCE_NETUID", 1),
        "wallet_name": _env_str("WALLET_NAME", "default"),
        "hotkey_name": _env_str("HOTKEY_NAME", "default"),
        "wallet_path": _env_str("BT_WALLET_PATH", "~/.bittensor/wallets"),
        "use_drand": _env_bool("RELIQUARY_INFERENCE_USE_DRAND", False),
        "r2_bucket": _env_str("RELIQUARY_INFERENCE_R2_BUCKET", ""),
        "r2_endpoint_url": _env_str("RELIQUARY_INFERENCE_R2_ENDPOINT_URL", ""),
        "r2_access_key_id": _env_str("RELIQUARY_INFERENCE_R2_ACCESS_KEY_ID", ""),
        "r2_secret_access_key": _env_str("RELIQUARY_INFERENCE_R2_SECRET_ACCESS_KEY", ""),
        # REST-mode R2 (reuses reliquary-protocol's R2ObjectBackend; authenticates
        # via a single CF account API token rather than S3 access keys).
        "r2_rest_account_id": _env_str("RELIQUARY_INFERENCE_R2_ACCOUNT_ID", ""),
        "r2_rest_bucket": _env_str("RELIQUARY_INFERENCE_R2_BUCKET", ""),
        "r2_rest_cf_api_token": _env_str("RELIQUARY_INFERENCE_R2_CF_API_TOKEN", ""),
        "r2_rest_public_url": _env_str("RELIQUARY_INFERENCE_R2_PUBLIC_URL", ""),
        "dataset_name": _env_str("RELIQUARY_INFERENCE_DATASET_NAME", "karpathy/climbmix-400b-shuffle"),
        "dataset_split": _env_str("RELIQUARY_INFERENCE_DATASET_SPLIT", "train"),
        # DAPO zone filter — σ_min threshold relaxed to 0.33 during bootstrap
        # so a new fleet with a weak base model can still collect enough
        # training signal to start the learning loop.
        "zone_filter_bootstrap": _env_bool("RELIQUARY_INFERENCE_ZONE_FILTER_BOOTSTRAP", False),
        "cooldown_windows": _env_int("RELIQUARY_INFERENCE_COOLDOWN_WINDOWS", 50),
        # Miner sampling — used when samples_per_task > 1 so the M rollouts
        # in a GRPO group aren't all identical. Default 0.9 matches DAPO
        # T_PROTO. top_p=1.0 = no nucleus truncation (DAPO default).
        "generation_temperature": float(os.getenv("RELIQUARY_INFERENCE_GENERATION_TEMPERATURE", "0.9")),
        "generation_top_p": float(os.getenv("RELIQUARY_INFERENCE_GENERATION_TOP_P", "1.0")),
        # MATH difficulty cap during bootstrap — restrict sampling pool to
        # Level 1..N problems (Hendrycks 1=easiest, 5=hardest). Qwen2.5-3B
        # on Level 4-5 gets 0/8 ~100% of the time, producing σ=0 groups
        # that starve the zone filter. Setting max_level=2 gives the base
        # model a realistic shot at partial credit. Unset / None = no cap.
        "math_max_level": (
            int(os.environ["RELIQUARY_INFERENCE_MATH_MAX_LEVEL"])
            if os.environ.get("RELIQUARY_INFERENCE_MATH_MAX_LEVEL")
            else None
        ),
        # Validator backfill: how many historical windows to scan each loop
        # for unprocessed completion bundles. 10 windows × 30 blocks × 12 s
        # ≈ 60 min — covers typical R2-rate-limit-induced lag.
        "validator_backfill_horizon_windows": _env_int(
            "RELIQUARY_INFERENCE_VALIDATOR_BACKFILL_HORIZON_WINDOWS", 10,
        ),
        "window_stride_blocks": _env_int(
            "RELIQUARY_INFERENCE_WINDOW_STRIDE_BLOCKS", 30,
        ),
        "git_sha": _git_sha(),
    }
