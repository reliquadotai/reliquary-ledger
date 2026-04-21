#!/usr/bin/env python3
"""
Autonomous Reliquary subnet bootstrap daemon (v2 — persistent WS + 429 backoff).

Phases:
  1. Wait for testnet NetworkRateLimit window, then `btcli subnet create`.
     Retry on NetworkTxRateLimitExceeded (we lost the race to another user).
  2. `btcli subnet start` (dTAO activation).
  3. Register reliquary-validator hotkey.
  4. Register reliquary-miner hotkey.
  5. Update env file with new NETUID.

State is persisted to /save/state/subnet-bootstrap.json so re-launching
resumes from the last completed phase.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BTCLI = "/opt/reliquary-venv/bin/btcli"
WALLET_PATH = "/save/.bittensor/wallets"
NETWORK = "test"
SUBNET_NAME = "Reliquary"

LOG_DIR = Path("/save/logs")
STATE_DIR = Path("/save/state")
STATE_FILE = STATE_DIR / "subnet-bootstrap.json"
LOG_FILE = LOG_DIR / "subnet-bootstrap.log"
LOCK_FILE = STATE_DIR / "subnet-bootstrap.lock"
ENV_FILE = Path("/save/state/reliquary-inference.env")

OWNER_WALLET = "reliquary-inference"
OWNER_HOTKEY = "default"
VALIDATOR_WALLET = "reliquary-validator"
VALIDATOR_HOTKEY = "default"
MINER_WALLET = "reliquary-miner"
MINER_HOTKEY = "default"

GITHUB_REPO = "https://github.com/0xgrizz/reliquary-inference"
CONTACT_EMAIL = "reliquary@protonmail.com"
SUBNET_URL = "https://github.com/0xgrizz/reliquary-inference"
DISCORD_HANDLE = "0xgrizz"
LOGO_URL = "https://avatars.githubusercontent.com/u/114535944"
DESCRIPTION = "Proof-carrying verifiable inference subnet"
ADDITIONAL_INFO = "Reliquary Ledger proof-carrying verifiable inference"

CREATE_MAX_ATTEMPTS = 8
REGISTER_MAX_ATTEMPTS = 5
POLL_FAR_S = 90
POLL_NEAR_S = 20
POLL_VERY_NEAR_S = 6
NEAR_WINDOW_BLOCKS = 40
VERY_NEAR_WINDOW_BLOCKS = 8
BLOCK_TIME_S = 12
WINDOW_SAFETY_BLOCKS = 2
NETWORK_RATE_LIMIT = 720

# Connection recovery
MAX_CONN_BACKOFF_S = 300
INITIAL_CONN_BACKOFF_S = 30


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    state["last_update"] = utcnow()
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


class Chain:
    """Persistent Subtensor connection with auto-reconnect on failure."""

    def __init__(self):
        self._sub = None
        self._backoff = INITIAL_CONN_BACKOFF_S

    def _connect(self):
        import bittensor as bt
        logging.info("connecting Subtensor(network=%s) ...", NETWORK)
        self._sub = bt.Subtensor(network=NETWORK)
        self._backoff = INITIAL_CONN_BACKOFF_S
        logging.info("connected")

    def _ensure(self):
        if self._sub is None:
            self._connect()

    def _close(self):
        try:
            if self._sub is not None:
                try:
                    self._sub.substrate.close()
                except Exception:
                    pass
        finally:
            self._sub = None

    def call(self, fn, *args, **kwargs):
        """Run a callable with the Subtensor instance, reconnecting on failure."""
        last_exc = None
        for attempt in range(4):
            try:
                self._ensure()
                return fn(self._sub, *args, **kwargs)
            except Exception as e:
                last_exc = e
                self._close()
                msg = str(e)
                if "429" in msg or "InvalidStatus" in msg:
                    backoff = self._backoff
                    logging.warning("HTTP 429 / ws rejected — backoff %ds (attempt %d)", backoff, attempt + 1)
                    time.sleep(backoff)
                    self._backoff = min(int(self._backoff * 2), MAX_CONN_BACKOFF_S)
                else:
                    logging.warning("chain call failed: %s — reconnecting after 20s (attempt %d)", e, attempt + 1)
                    time.sleep(20)
        raise last_exc


def snapshot(chain: Chain) -> dict:
    def _fn(sub):
        total = int(sub.get_total_subnets())
        block = int(sub.block)
        last_netuid = total - 1
        r = sub.substrate.query("SubtensorModule", "NetworkRegisteredAt", [last_netuid])
        last_reg_block = int(getattr(r, "value", r))
        burn = sub.get_subnet_burn_cost()
        return {
            "block": block,
            "total_subnets": total,
            "last_netuid": last_netuid,
            "last_reg_block": last_reg_block,
            "next_window_block": last_reg_block + NETWORK_RATE_LIMIT + WINDOW_SAFETY_BLOCKS,
            "burn_cost": str(burn),
        }
    return chain.call(_fn)


def wait_for_window(chain: Chain) -> dict:
    """Poll until current block >= target window. Target is recomputed on
    each iteration so if a competitor creates we re-arm automatically."""
    while True:
        snap = snapshot(chain)
        delta = snap["next_window_block"] - snap["block"]
        if delta <= 0:
            logging.info(
                "WINDOW OPEN: block=%s target=%s (last_netuid=%s burn=%s)",
                snap["block"], snap["next_window_block"], snap["last_netuid"], snap["burn_cost"],
            )
            return snap
        if delta <= VERY_NEAR_WINDOW_BLOCKS:
            interval = POLL_VERY_NEAR_S
        elif delta <= NEAR_WINDOW_BLOCKS:
            interval = POLL_NEAR_S
        else:
            interval = POLL_FAR_S
        eta_s = delta * BLOCK_TIME_S
        logging.info(
            "wait: block=%s target=%s delta=%s (~%.0fs) burn=%s poll=%ss",
            snap["block"], snap["next_window_block"], delta, eta_s,
            snap["burn_cost"], interval,
        )
        time.sleep(interval)


def run(cmd: list, timeout: int = 300) -> tuple:
    logging.info("RUN: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (res.stdout or "") + (res.stderr or "")
    logging.info("exit=%s", res.returncode)
    if res.stdout:
        logging.info("stdout:\n%s", res.stdout[-4000:])
    if res.stderr:
        logging.info("stderr:\n%s", res.stderr[-4000:])
    return res.returncode, out, res.stdout


def parse_netuid(stdout: str) -> int | None:
    for line in stdout.splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            j = json.loads(s)
        except Exception:
            continue
        if j.get("success") and j.get("netuid") is not None:
            return int(j["netuid"])
    return None


def cmd_create() -> list:
    return [
        BTCLI, "subnet", "create",
        "--subnet-name", SUBNET_NAME,
        "--wallet-path", WALLET_PATH,
        "--wallet-name", OWNER_WALLET,
        "--hotkey", OWNER_HOTKEY,
        "--network", NETWORK,
        "--description", DESCRIPTION,
        "--github-repo", GITHUB_REPO,
        "--subnet-contact", CONTACT_EMAIL,
        "--subnet-url", SUBNET_URL,
        "--discord-handle", DISCORD_HANDLE,
        "--logo-url", LOGO_URL,
        "--additional-info", ADDITIONAL_INFO,
        "--json-output",
        "--no-prompt",
    ]


def cmd_start(netuid: int) -> list:
    return [
        BTCLI, "subnet", "start",
        "--netuid", str(netuid),
        "--wallet-path", WALLET_PATH,
        "--wallet-name", OWNER_WALLET,
        "--hotkey", OWNER_HOTKEY,
        "--network", NETWORK,
        "--no-prompt",
    ]


def cmd_register(netuid: int, wallet: str, hotkey: str) -> list:
    return [
        BTCLI, "subnet", "register",
        "--netuid", str(netuid),
        "--wallet-path", WALLET_PATH,
        "--wallet-name", wallet,
        "--hotkey", hotkey,
        "--network", NETWORK,
        "--json-output",
        "--no-prompt",
    ]


def phase_create(state: dict, chain: Chain) -> dict:
    logging.info("== Phase 1: subnet create ==")
    attempts = state.get("create_attempts", 0)
    while attempts < CREATE_MAX_ATTEMPTS:
        wait_for_window(chain)
        code, out, stdout = run(cmd_create(), timeout=300)
        attempts += 1
        state["create_attempts"] = attempts
        state["last_create_tail"] = out[-2000:]
        netuid = parse_netuid(stdout)
        if code == 0 and netuid is not None:
            state["netuid"] = netuid
            state["phase"] = "created"
            state["created_at"] = utcnow()
            save_state(state)
            logging.info("✅ subnet created: netuid=%s", netuid)
            return state
        if "NetworkTxRateLimitExceeded" in out:
            logging.warning(
                "Lost the race (attempt %d) — competitor created a subnet. "
                "Rearming for next window.", attempts,
            )
            time.sleep(6)
            save_state(state)
            continue
        logging.error("create failed (attempt %d) — non-rate-limit error. Sleeping 30s.", attempts)
        save_state(state)
        time.sleep(30)
    raise RuntimeError(f"Exhausted create retries after {attempts} attempts")


def phase_start(state: dict) -> dict:
    netuid = state["netuid"]
    logging.info("== Phase 2: subnet start (netuid=%s) ==", netuid)
    for attempt in range(REGISTER_MAX_ATTEMPTS):
        code, out, _ = run(cmd_start(netuid), timeout=180)
        state["start_tail"] = out[-2000:]
        if code == 0:
            state["started"] = True
            state["phase"] = "started"
            state["started_at_time"] = utcnow()
            save_state(state)
            logging.info("✅ subnet started")
            return state
        logging.warning("start attempt %d failed (exit=%d). Sleeping 30s.", attempt + 1, code)
        save_state(state)
        time.sleep(30)
    logging.error("subnet start exhausted retries — continuing to registrations anyway")
    state["start_failed"] = True
    save_state(state)
    return state


def phase_register(state: dict, role: str, wallet: str, hotkey: str) -> dict:
    netuid = state["netuid"]
    key = f"{role}_registered"
    tail_key = f"{role}_register_tail"
    logging.info("== Phase: register %s on netuid=%s ==", role, netuid)
    for attempt in range(REGISTER_MAX_ATTEMPTS):
        code, out, _ = run(cmd_register(netuid, wallet, hotkey), timeout=300)
        state[tail_key] = out[-2000:]
        if code == 0:
            state[key] = True
            state["phase"] = f"{role}_registered"
            save_state(state)
            logging.info("✅ %s registered", role)
            return state
        logging.warning("%s register attempt %d failed (exit=%d). Sleeping 30s.", role, attempt + 1, code)
        save_state(state)
        time.sleep(30)
    logging.error("%s registration exhausted retries", role)
    state[f"{role}_register_failed"] = True
    save_state(state)
    return state


def phase_write_env(state: dict) -> None:
    if state.get("env_written"):
        return
    netuid = state.get("netuid")
    if netuid is None:
        return
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("RELIQUARY_INFERENCE_NETUID="):
                continue
            if line.startswith("RELIQUARY_INFERENCE_NETWORK="):
                continue
            if line.startswith("WALLET_NAME="):
                continue
            if line.startswith("BT_SUBTENSOR_NETWORK=") or line.startswith("BT_WALLET_PATH="):
                continue
            lines.append(line)
    lines.append(f"RELIQUARY_INFERENCE_NETUID={netuid}")
    lines.append("RELIQUARY_INFERENCE_NETWORK=test")
    lines.append(f"WALLET_NAME={OWNER_WALLET}")
    lines.append("BT_SUBTENSOR_NETWORK=test")
    lines.append("BT_WALLET_PATH=/save/.bittensor/wallets")
    ENV_FILE.write_text("\n".join(lines).rstrip() + "\n")
    state["env_written"] = True
    save_state(state)
    logging.info("✅ env file written: %s", ENV_FILE)


def acquire_lock() -> None:
    if LOCK_FILE.exists():
        pid_txt = LOCK_FILE.read_text().strip()
        if pid_txt.isdigit() and Path(f"/proc/{pid_txt}").exists():
            raise SystemExit(f"lock held by pid {pid_txt}")
        logging.info("stale lock (pid %s gone) — removing", pid_txt)
        LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def on_signal(signum, _frame):
    logging.warning("signal %s received — releasing lock and exiting", signum)
    release_lock()
    sys.exit(0)


def main() -> int:
    setup_logging()
    logging.info("=== subnet-bootstrap-daemon v2 starting ===")
    acquire_lock()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        state = load_state()
        state.setdefault("started_at", utcnow())
        state.setdefault("phase", "init")
        save_state(state)

        chain = Chain()
        snap0 = snapshot(chain)
        logging.info("initial chain snapshot: %s", snap0)
        state["initial_snapshot_v2"] = snap0
        save_state(state)

        if not state.get("netuid"):
            state = phase_create(state, chain)
        if not state.get("started") and not state.get("start_failed"):
            state = phase_start(state)
        if not state.get("validator_registered"):
            state = phase_register(state, "validator", VALIDATOR_WALLET, VALIDATOR_HOTKEY)
        if not state.get("miner_registered"):
            state = phase_register(state, "miner", MINER_WALLET, MINER_HOTKEY)
        phase_write_env(state)

        state["phase"] = "done"
        state["done_at"] = utcnow()
        save_state(state)
        logging.info("🎉 all phases complete — netuid=%s", state.get("netuid"))
        return 0
    except Exception:
        logging.exception("fatal error")
        return 1
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
