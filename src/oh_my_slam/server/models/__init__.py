"""Model adapters loaded by the inference server.

Every adapter method runs on the single GPU worker thread. ``load`` may raise; every adapter is
required, so one failure puts the server in ``error``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def select_device() -> str:
    """MPS when available (spec §4: preferred, not mandatory), else the CPU."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class LoadState:
    loaded: bool = False
    error: str | None = None
    detail: str | None = None


@dataclass
class Registry:
    """Holds the adapters and their load state; filled on the GPU thread."""

    adapters: dict[str, Any] = field(default_factory=dict)
    state: dict[str, LoadState] = field(default_factory=dict)
    device: str = "cpu"
    precision: str = "fp32"

    def add(self, adapter: Any) -> None:
        self.adapters[adapter.key] = adapter
        self.state[adapter.key] = LoadState()

    def get(self, key: str) -> Any | None:
        st = self.state.get(key)
        if st is None or not st.loaded:
            return None
        return self.adapters[key]

    def load_all(self, device: str, log: Any = None) -> None:
        self.device = device
        for key, adapter in self.adapters.items():
            st = self.state[key]
            try:
                if log:
                    log(f"loading {adapter.name} on {device}")
                adapter.load(device)
                adapter.warmup()
                st.loaded = True
                st.detail = getattr(adapter, "detail", None)
            except Exception as exc:
                st.error = f"{type(exc).__name__}: {exc}"
                if log:
                    log(f"failed to load {adapter.name}: {st.error}")
        precision = [getattr(a, "precision", None) for a in self.adapters.values()]
        self.precision = next((p for p in precision if p), "fp32")

    def status(self) -> str:
        return "ready" if all(st.loaded for st in self.state.values()) else "error"


def build_registry() -> Registry:
    reg = Registry()
    from oh_my_slam.server.models.geometry_moge import MoGeGeometry
    from oh_my_slam.server.models.gravity_geocalib import GeoCalibGravity
    from oh_my_slam.server.models.multiview_mapanything import MapAnythingMultiview
    from oh_my_slam.server.models.seg_yoloe import YoloeSegmenter

    reg.add(MoGeGeometry())
    reg.add(GeoCalibGravity())
    reg.add(YoloeSegmenter())
    reg.add(MapAnythingMultiview())
    return reg
