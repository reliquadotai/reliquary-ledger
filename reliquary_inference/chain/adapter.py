from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import request

from ..constants import WINDOW_LENGTH


@dataclass
class WindowContext:
    window_id: int
    block_hash: str
    public_randomness: str
    task_source: str
    model_ref: str
    dataset_name: str
    dataset_split: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "block_hash": self.block_hash,
            "public_randomness": self.public_randomness,
            "task_source": self.task_source,
            "model_ref": self.model_ref,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
        }


def _combine_randomness(block_hash: str, drand_randomness: str | None = None) -> str:
    clean_hash = block_hash.replace("0x", "")
    if drand_randomness:
        return hashlib.sha256(bytes.fromhex(clean_hash) + bytes.fromhex(drand_randomness)).hexdigest()
    return clean_hash


def _fetch_drand_randomness() -> str | None:
    try:
        with request.urlopen("https://api.drand.sh/public/latest", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        randomness = payload.get("randomness")
        return randomness if isinstance(randomness, str) else None
    except Exception:
        return None


class LocalChainAdapter:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def get_window_context(self, *, cfg: dict[str, Any], window_id: int | None = None) -> WindowContext:
        current_window = window_id if window_id is not None else int(time.time() // WINDOW_LENGTH)
        block_hash = hashlib.sha256(f"local-block-{current_window}".encode("utf-8")).hexdigest()
        randomness = _combine_randomness(block_hash)
        return WindowContext(
            window_id=current_window,
            block_hash=block_hash,
            public_randomness=randomness,
            task_source=str(cfg["task_source"]),
            model_ref=str(cfg["model_ref"]),
            dataset_name=str(cfg["dataset_name"]),
            dataset_split=str(cfg["dataset_split"]),
        )

    def publish_weights(self, *, window_id: int, weights: dict[str, float]) -> dict[str, Any]:
        event = {
            "mode": "local",
            "event_type": "weights",
            "window_id": window_id,
            "weights": dict(weights),
            "event_index": len(self._events),
        }
        self._events.append(event)
        return event

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)


class BittensorChainAdapter:
    def __init__(
        self,
        *,
        network: str,
        netuid: int,
        wallet_name: str,
        hotkey_name: str,
        wallet_path: str,
        use_drand: bool,
        chain_endpoint: str = "",
    ) -> None:
        self.network = network
        self.netuid = netuid
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.wallet_path = wallet_path
        self.use_drand = use_drand
        self.chain_endpoint = chain_endpoint
        self._subtensor_client = None

    def _subtensor(self):
        import bittensor as bt

        if self._subtensor_client is None:
            if self.chain_endpoint:
                config = bt.Subtensor.config()
                config.subtensor.chain_endpoint = self.chain_endpoint
                self._subtensor_client = bt.Subtensor(network=None, config=config)
            else:
                self._subtensor_client = bt.Subtensor(network=self.network)
        return self._subtensor_client

    def _with_subtensor(self, callback):
        subtensor = self._subtensor()
        try:
            return callback(subtensor)
        except Exception:
            self._subtensor_client = None
            raise

    def get_current_block(self) -> int:
        return int(self._with_subtensor(lambda subtensor: subtensor.block))

    def get_block_hash(self, block_number: int) -> str:
        return str(self._with_subtensor(lambda subtensor: subtensor.get_block_hash(block_number)))

    def get_metagraph(self):
        return self._with_subtensor(lambda subtensor: subtensor.metagraph(self.netuid))

    def describe_hotkeys(self, hotkeys: dict[str, str]) -> dict[str, dict[str, Any]]:
        metagraph = self.get_metagraph()
        hotkey_to_uid = {
            str(hotkey): int(uid)
            for hotkey, uid in zip(metagraph.hotkeys, metagraph.uids)
        }
        details: dict[str, dict[str, Any]] = {}
        for role, hotkey in hotkeys.items():
            normalized = str(hotkey).strip()
            uid = hotkey_to_uid.get(normalized)
            details[role] = {
                "hotkey": normalized,
                "registered": uid is not None,
                "uid": int(uid) if uid is not None else -1,
            }
        return details

    def get_window_context(self, *, cfg: dict[str, Any], window_id: int | None = None) -> WindowContext:
        current_block = self.get_current_block()
        target_window = window_id if window_id is not None else (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
        block_hash = self.get_block_hash(target_window)
        drand_randomness = _fetch_drand_randomness() if self.use_drand else None
        randomness = _combine_randomness(block_hash, drand_randomness)
        return WindowContext(
            window_id=target_window,
            block_hash=block_hash,
            public_randomness=randomness,
            task_source=str(cfg["task_source"]),
            model_ref=str(cfg["model_ref"]),
            dataset_name=str(cfg["dataset_name"]),
            dataset_split=str(cfg["dataset_split"]),
        )

    def publish_weights(self, *, window_id: int, weights: dict[str, float]) -> dict[str, Any]:
        from bittensor_wallet import Wallet

        wallet = Wallet(name=self.wallet_name, hotkey=self.hotkey_name, path=self.wallet_path)
        metagraph = self.get_metagraph()
        hotkey_to_uid = dict(zip(metagraph.hotkeys, metagraph.uids))
        uids = []
        weight_values = []
        for hotkey, weight in weights.items():
            if hotkey in hotkey_to_uid and weight > 0:
                uids.append(int(hotkey_to_uid[hotkey]))
                weight_values.append(float(weight))
        if not uids:
            return {
                "mode": "bittensor",
                "window_id": window_id,
                "uids": [],
                "weights": [],
                "success": False,
                "reason": "no_matching_hotkeys",
            }
        success = self._with_subtensor(
            lambda subtensor: subtensor.set_weights(
                wallet=wallet,
                netuid=self.netuid,
                uids=uids,
                weights=weight_values,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
        )
        return {
            "mode": "bittensor",
            "window_id": window_id,
            "uids": uids,
            "weights": weight_values,
            "success": bool(success),
        }
