"""Model adapters loaded by the inference server.

Every adapter method runs on the single GPU worker thread. ``load`` may raise; required adapters
put the server in ``error``, optional ones (SAM 3) put it in ``degraded``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


class ModelAdapter(Protocol):
    key: str
    name: str
    required: bool

    def load(self, device: str) -> None: ...

    def warmup(self) -> None: ...


def select_device() -> str:
    """``OH_MY_SLAM_DEVICE`` (cpu|mps) or MPS when available."""
    forced = os.environ.get("OH_MY_SLAM_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps"}:
        return forced
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
        required_ok = all(
            self.state[k].loaded for k, a in self.adapters.items() if getattr(a, "required", True)
        )
        if not required_ok:
            return "error"
        if all(st.loaded for st in self.state.values()):
            return "ready"
        return "degraded"


def build_registry(stub: bool = False) -> Registry:
    reg = Registry()
    if stub:
        from oh_my_slam.server.models.stub import stub_adapters

        for a in stub_adapters():
            reg.add(a)
        return reg
    from oh_my_slam.server.models.geometry_moge import MoGeGeometry
    from oh_my_slam.server.models.gravity_geocalib import GeoCalibGravity
    from oh_my_slam.server.models.multiview_mapanything import MapAnythingMultiview
    from oh_my_slam.server.models.seg_sam3 import Sam3Refiner
    from oh_my_slam.server.models.seg_yoloe import YoloeSegmenter

    reg.add(MoGeGeometry())
    reg.add(GeoCalibGravity())
    reg.add(YoloeSegmenter())
    reg.add(Sam3Refiner())
    reg.add(MapAnythingMultiview())
    return reg
