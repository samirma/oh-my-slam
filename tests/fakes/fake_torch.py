"""A NumPy stand-in for the few parts of ``torch`` the server's model adapters use.

The adapters (``oh_my_slam.server.models``) import ``torch`` and their model libraries inside their
methods. The unit tests run offline in a process that also imports Open3D, where real torch must
never be imported (duplicate libomp aborts the process), so they put this module — and fake model
libraries built on it — into ``sys.modules`` (:func:`installed`) and run the adapters' real pre- and
post-processing without weights or a device.

Like torch, a tensor on a device other than the CPU refuses ``.numpy()``: an adapter that forgets
``.cpu()`` before reading a result back fails here as it would on MPS.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np


class Device:
    def __init__(self, name: str | Device) -> None:
        self.type = str(name)

    def __str__(self) -> str:
        return self.type

    def __eq__(self, other: object) -> bool:
        return str(other) == self.type

    def __hash__(self) -> int:
        return hash(self.type)


class Tensor:
    def __init__(self, data: Any, device: str | Device = "cpu") -> None:
        self.a = np.asarray(data)
        self.device = str(device)

    def __repr__(self) -> str:
        return f"Tensor({self.a!r}, device={self.device!r})"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.a.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.a.dtype

    def _new(self, a: Any) -> Tensor:
        return Tensor(a, self.device)

    def to(self, device: str | Device) -> Tensor:
        return Tensor(self.a, device)

    def cpu(self) -> Tensor:
        return Tensor(self.a, "cpu")

    def detach(self) -> Tensor:
        return self

    def float(self) -> Tensor:
        return self._new(self.a.astype(np.float32))

    def div_(self, x: float) -> Tensor:
        self.a = self.a / np.float32(x)
        return self

    def permute(self, *dims: int) -> Tensor:
        return self._new(self.a.transpose(dims))

    def reshape(self, *shape: int) -> Tensor:
        return self._new(self.a.reshape(*shape))

    def numpy(self) -> np.ndarray:
        if self.device != "cpu":
            raise TypeError(f"can't convert {self.device} device type tensor to numpy. "
                            "Use Tensor.cpu() to copy the tensor to host memory first.")
        return self.a

    def __float__(self) -> float:
        return float(self.a)

    def element_size(self) -> int:
        return int(self.a.itemsize)

    def nelement(self) -> int:
        return int(self.a.size)

    def __getitem__(self, index: Any) -> Tensor:
        return self._new(self.a[index])

    def __gt__(self, other: Any) -> Tensor:
        return self._new(self.a > (other.a if isinstance(other, Tensor) else other))

    def __mul__(self, other: Any) -> Tensor:
        return self._new(self.a * (other.a if isinstance(other, Tensor) else other))

    __rmul__ = __mul__


def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def make_torch(mps_available: bool = False) -> types.ModuleType:
    """A fresh fake ``torch``; ``torch.calls`` records the device-level calls in order."""
    calls: list[tuple[Any, ...]] = []
    state = {"inference_mode": False}

    @contextmanager
    def inference_mode() -> Iterator[None]:
        state["inference_mode"] = True
        try:
            yield
        finally:
            state["inference_mode"] = False

    def rand(*shape: int, device: str | Device = "cpu") -> Tensor:
        return Tensor(np.random.default_rng(0).random(shape, dtype=np.float32), device)

    def tensor(data: Any, device: str | Device = "cpu") -> Tensor:
        return Tensor(np.asarray(data), device)

    def normalize(t: Tensor, dim: int = -1) -> Tensor:
        return t._new(t.a / np.linalg.norm(t.a, axis=dim, keepdims=True))

    mps = _module("torch.mps",
                  empty_cache=lambda: calls.append(("empty_cache",)),
                  set_per_process_memory_fraction=lambda f: calls.append(("memory_fraction", f)))
    backends = _module("torch.backends",
                       mps=_module("torch.backends.mps", is_available=lambda: mps_available))
    functional = _module("torch.nn.functional", normalize=normalize)
    return _module("torch", Tensor=Tensor, device=Device, from_numpy=lambda a: Tensor(a),
                   tensor=tensor, rand=rand, inference_mode=inference_mode, mps=mps,
                   backends=backends, nn=_module("torch.nn", functional=functional),
                   calls=calls, state=state)


def libraries(contents: dict[str, dict[str, Any]]) -> dict[str, types.ModuleType]:
    """Fake modules as ``sys.modules`` entries: ``{"moge.model.v2": {"MoGeModel": cls}}`` gives
    ``moge.model.v2`` holding ``MoGeModel``, with each parent package (``moge``, ``moge.model``),
    every module an attribute of its parent."""
    mods: dict[str, types.ModuleType] = {}
    for dotted, attrs in contents.items():
        parts = dotted.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[:i + 1])
            if name not in mods:
                mods[name] = _module(name)
                if i:
                    setattr(mods[".".join(parts[:i])], parts[i], mods[name])
        mods[dotted].__dict__.update(attrs)
    return mods


@contextmanager
def installed(modules: dict[str, types.ModuleType]) -> Iterator[None]:
    """Put ``modules`` into ``sys.modules`` for the block, then restore whatever was there."""
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
