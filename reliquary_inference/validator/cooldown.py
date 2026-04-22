"""Per-prompt cooldown map — DAPO curriculum diversity guard.

Adapted from romain13190/reliquary (MIT). A prompt that entered the training
batch at window N is ineligible for the next ``cooldown_windows`` windows.
This forces the curriculum to rotate so the policy has time to shift between
re-uses of the same prompt — prevents the trainer from fixating on a single
easy-zone prompt.

The validator publishes the current cooldown set via the task-batch
``window_context["cooldown_indices"]`` — ``MathTasksSource.build_window_batch``
respects it by skipping those dataset indices when rolling the per-window
sample.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

# Default horizon: 50 windows. With 6-min windows this is ~5h — long enough
# for the policy to actually shift before revisiting a prompt.
DEFAULT_COOLDOWN_WINDOWS = 50


class CooldownMap:
    """Per-prompt "last batched at window N" store + eligibility predicate.

    A prompt is in cooldown iff ``current_window - last_batched < cooldown_windows``.
    At ``current_window == last_batched + cooldown_windows`` the prompt becomes
    eligible again (half-open interval).
    """

    def __init__(self, cooldown_windows: int = DEFAULT_COOLDOWN_WINDOWS) -> None:
        if cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        self._cooldown_windows = int(cooldown_windows)
        self._last_batched: dict[int, int] = {}

    @property
    def cooldown_windows(self) -> int:
        return self._cooldown_windows

    def record_batched(self, prompt_idx: int, window: int) -> None:
        """Mark *prompt_idx* as having entered the batch at *window*."""
        if prompt_idx < 0:
            raise ValueError("prompt_idx must be non-negative")
        if window < 0:
            raise ValueError("window must be non-negative")
        self._last_batched[int(prompt_idx)] = int(window)

    def record_batched_many(
        self,
        prompt_indices: Iterable[int],
        window: int,
    ) -> None:
        """Mark every prompt_idx in *prompt_indices* as batched at *window*."""
        for idx in prompt_indices:
            self.record_batched(int(idx), window)

    def is_in_cooldown(self, prompt_idx: int, current_window: int) -> bool:
        """True iff *prompt_idx* was batched within the cooldown horizon."""
        if self._cooldown_windows == 0:
            return False
        last = self._last_batched.get(int(prompt_idx))
        if last is None:
            return False
        return current_window - last < self._cooldown_windows

    def current_cooldown_set(self, current_window: int) -> set[int]:
        """All prompt_idx currently in cooldown at *current_window*."""
        if self._cooldown_windows == 0:
            return set()
        return {
            idx for idx, last in self._last_batched.items()
            if current_window - last < self._cooldown_windows
        }

    def __len__(self) -> int:
        return len(self._last_batched)

    # ---------- persistence ----------

    def save(self, path: os.PathLike[str] | str) -> None:
        """Serialise to JSON at *path*. Atomic via tmp-file + rename."""
        path = str(path)
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".cooldown.", dir=parent)
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {
                        "cooldown_windows": self._cooldown_windows,
                        "last_batched": self._last_batched,
                    },
                    f,
                )
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def load(self, path: os.PathLike[str] | str) -> None:
        """Load state from JSON at *path*. No-op if file doesn't exist."""
        path = str(path)
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self._load_dict(data)

    def _load_dict(self, data: dict) -> None:
        # JSON object keys are strings — coerce back to int.
        self._last_batched = {
            int(k): int(v) for k, v in data.get("last_batched", {}).items()
        }
        if "cooldown_windows" in data:
            self._cooldown_windows = int(data["cooldown_windows"])

    def as_dict(self) -> dict:
        return {
            "cooldown_windows": self._cooldown_windows,
            "last_batched": self._last_batched,
        }

    # ---------- R2 disaster-recovery mirror ----------
    #
    # Cooldown is a load-bearing curriculum guard — if the map is lost
    # (node rebuild, disk wipe) the task builder will immediately re-
    # sample prompts that were supposed to be in the 50-window cooldown,
    # starving the curriculum of diversity. Best-effort R2 mirroring on
    # every save ensures we can recover.

    def save_r2(self, backend, *, r2_key: str) -> None:
        """Best-effort R2 upload of the current map. Swallows failures."""
        try:
            backend.put(r2_key, json.dumps(self.as_dict(), sort_keys=True).encode("utf-8"))
        except Exception:
            # Local file is authoritative; R2 is disaster recovery only.
            # A transient R2 error should not fail the validator loop.
            pass

    def load_r2(self, backend, *, r2_key: str) -> bool:
        """Pull map state from R2. Returns True if loaded successfully."""
        try:
            raw = backend.get(r2_key)
            if raw is None:
                return False
            self._load_dict(json.loads(raw.decode("utf-8")))
            return True
        except Exception:
            return False

    # ---------- helpers ----------

    def prune(self, current_window: int) -> int:
        """Drop entries older than *current_window - cooldown_windows*.

        Returns the number of entries dropped. Production validators can
        call this once per window to keep the map from growing unbounded
        on long-running processes.
        """
        if self._cooldown_windows == 0:
            dropped = len(self._last_batched)
            self._last_batched.clear()
            return dropped
        horizon = current_window - self._cooldown_windows
        keep = {
            idx: last for idx, last in self._last_batched.items()
            if last > horizon
        }
        dropped = len(self._last_batched) - len(keep)
        self._last_batched = keep
        return dropped


def default_cooldown_path(state_dir: os.PathLike[str] | str) -> Path:
    """Conventional on-disk location for the validator's cooldown map."""
    return Path(state_dir) / "cooldown.json"


__all__ = [
    "CooldownMap",
    "DEFAULT_COOLDOWN_WINDOWS",
    "default_cooldown_path",
]
