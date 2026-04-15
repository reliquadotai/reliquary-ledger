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
        "dataset_name": _env_str("RELIQUARY_INFERENCE_DATASET_NAME", "karpathy/climbmix-400b-shuffle"),
        "dataset_split": _env_str("RELIQUARY_INFERENCE_DATASET_SPLIT", "train"),
        "git_sha": _git_sha(),
    }
