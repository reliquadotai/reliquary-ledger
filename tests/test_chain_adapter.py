import sys
import types

from reliquary_inference.chain.adapter import BittensorChainAdapter


class _DummyWallet:
    def __init__(self, *, name: str, hotkey: str, path: str) -> None:
        self.name = name
        self.hotkey = hotkey
        self.path = path


class _DummySubtensor:
    def __init__(self) -> None:
        self.called = False

    def set_weights(self, **kwargs):
        self.called = True
        raise AssertionError("set_weights should not be called when no hotkeys match")


class _FactorySubtensor:
    def __init__(self, calls: list[tuple[str, str | None]]) -> None:
        self.calls = calls
        self.block = 321

    @staticmethod
    def config():
        return types.SimpleNamespace(
            subtensor=types.SimpleNamespace(
                chain_endpoint="",
            )
        )

    def __call__(self, *, network: str | None = None, config=None, **_kwargs):
        chain_endpoint = None
        if config is not None:
            chain_endpoint = getattr(getattr(config, "subtensor", None), "chain_endpoint", None)
        self.calls.append((network, chain_endpoint))
        return self

    def get_block_hash(self, block_number: int) -> str:
        return f"0x{block_number:064x}"


def test_publish_weights_skips_chain_call_when_no_hotkeys_match(monkeypatch) -> None:
    dummy_subtensor = _DummySubtensor()
    monkeypatch.setitem(sys.modules, "bittensor_wallet", types.SimpleNamespace(Wallet=_DummyWallet))

    adapter = BittensorChainAdapter(
        network="test",
        netuid=1,
        wallet_name="wallet",
        hotkey_name="validator",
        wallet_path="/tmp/wallets",
        use_drand=False,
    )
    monkeypatch.setattr(adapter, "_subtensor", lambda: dummy_subtensor)
    monkeypatch.setattr(
        adapter,
        "get_metagraph",
        lambda: types.SimpleNamespace(hotkeys=["5known"], uids=[7]),
    )

    result = adapter.publish_weights(window_id=123, weights={"5unknown": 1.0})

    assert result == {
        "mode": "bittensor",
        "window_id": 123,
        "uids": [],
        "weights": [],
        "success": False,
        "reason": "no_matching_hotkeys",
    }
    assert dummy_subtensor.called is False


def test_bittensor_adapter_reuses_cached_subtensor(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []
    factory = _FactorySubtensor(calls)
    monkeypatch.setitem(sys.modules, "bittensor", types.SimpleNamespace(Subtensor=factory))

    adapter = BittensorChainAdapter(
        network="test",
        netuid=1,
        wallet_name="wallet",
        hotkey_name="validator",
        wallet_path="/tmp/wallets",
        use_drand=False,
        chain_endpoint="wss://example.test",
    )

    assert adapter.get_current_block() == 321
    assert adapter.get_block_hash(120) == f"0x{120:064x}"
    assert calls == [(None, "wss://example.test")]
