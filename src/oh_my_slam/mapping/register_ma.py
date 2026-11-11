"""MapAnything registration of an anchored map's camera-only frames (prior ``none``; README
"Camera-only frames"), ported from the MapAnything candidate (branch ``cand/mapanything-priors``,
``service/register.py``) to the frames the prior-first engine hands it.

``OH_MY_SLAM_NONE_ENGINE`` picks the engine of an anchored map's frames without a prior:
``mapanything`` (this module) or ``geometric`` (``mapping.anchored``, as frames with priors).
Frames with priors always take the prior-first engine first; the frames without one come after,
against the map those placed (``service.prior_first``).

The new frames are registered in request order, in chunks. One multi-view call
(``Backend.geometry_views``) per chunk runs over:

* a bounded set of stored *anchor* keyframes (``MAX_ANCHORS``): the most similar to the chunk by
  thumbnail among the ``RECENT_POOL`` most recent, the most recent first on ties, with their
  stored map-frame poses as pose inputs (the first view of the call, MapAnything's reference);
* the chunk's frames with their intrinsics, unposed.

A call whose anchors' centres (nearly) coincide, for frames without depth images — a camera
turning in place — runs without pose inputs (``unposed``): pose inputs that close leave
MapAnything's translation normalisation degenerate and its depths inconsistent.

Each call's output is brought into the map frame by a rigid fit on its anchors (their stored
poses) and one scale: the frames' depth images when they have some, else the anchors' stored
depth (the map's own depth scale: its camera heights, ``mapping.anchored.ScaleState``, never
MapAnything's metres), else MapAnything's metres.

A frame is **registered** at MapAnything's map-frame pose when its depth (its depth image, else
MapAnything's, scaled) agrees with the map there: at least ``MIN_OVERLAP`` of its points land on a
reference view's depth and ``MIN_AGREEMENT`` of those agree within ``DEPTH_TOL``
(``mapping.anchored.agreement``) (``CALL_DEPTH_TOL``
between MapAnything's own depths of one call). The references are the anchors and the frames this
update already registered. A frame no reference confirms is rejected (``no_overlap``, or
``few_features`` with fewer than 150 features). Each placement carries the frame's median scene
depth (``Placement.scene_m``): the caller applies the distance limit, which a frame with enough
inliers escapes (api.md §4)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchor
from oh_my_slam.mapping.anchor import Anchor, Sim3
from oh_my_slam.mapping.anchored import DepthRef as Ref
from oh_my_slam.mapping.anchored import agreement, backproject

Array = NDArray[np.float64]

ENGINES = ("mapanything", "geometric")
MAX_VIEWS = 8  # views per MapAnything call (anchors + new): 18 GB beside the other models
MAX_ANCHORS = 4
MIN_ANCHORS = 2  # an out-of-memory retry keeps at least this many
RECENT_POOL = 60  # anchors among the most recent registered keyframes
POSED_BASELINE_M = 0.30  # anchors closer than this leave a call without depth images unposed
MA_SIGMA_POS_M = 0.05  # floors of MapAnything's pose sigma
MA_SIGMA_ROT_DEG = 1.5
MA_SIGMA_POS_REL = 0.03  # of the frame's median depth
UNPOSED_SIGMA_POS_REL = 0.10  # an unposed call's, whose positions drift in a turn
UNPOSED_SIGMA_ROT_DEG = 5.0  # an unposed call's rotations in a turn: up to 15 deg at 120 deg
DEPTH_TOL = 0.10  # relative depth agreement against depth images
CALL_DEPTH_TOL = 0.15  # between MapAnything's depths of one call (0.76 / 0.88 at 10 / 20 %)
MIN_DEPTH_M = 0.05
SIGMA_POS_FLOOR_M = 0.01
SIGMA_ROT_FLOOR_DEG = 0.1
MIN_SCALE_SIGMA_REL = 0.005
FEW_FEATURES = 150



@dataclass
class MvView:
    """One view of a multi-view call (the backend's ``geometry_views``): its pose (4x4
    camera-to-world in the call's frame), its z-depth (0 = none) and the 3x3 intrinsics of the
    depth's grid."""

    pose: NDArray[np.float64]
    depth: NDArray[np.float32]
    K: NDArray[np.float64]


def out_of_memory(exc: BaseException) -> bool:
    """Whether a multi-view call failed for want of device memory: the service's typed error
    ``out_of_memory`` (its ``code``)."""
    return getattr(exc, "code", None) == "out_of_memory"


Multiview = Callable[[list[Path], list[Intrinsics], list[Array | None]], list[MvView]]


@dataclass
class Stored:
    """A keyframe of the current version, as an anchor and a reference."""

    name: str
    image: Path
    K: Intrinsics  # of the image
    T: Array  # T_map_cam
    order: int  # its position in the map (later = more recent)
    depth: Callable[[], NDArray[np.float32] | None] = lambda: None  # metric, on its grid


@dataclass
class New:
    """A frame to register (canonical pinhole)."""

    index: int
    image: Path
    K: Intrinsics
    depth: NDArray[np.float32] | None = None  # metric, on the image's pixels
    features: int = FEW_FEATURES


@dataclass
class Placement:
    status: str  # registered / rejected
    reason: str | None = None
    T: Array | None = None
    sigma_pos_m: float | None = None
    sigma_rot_deg: float | None = None
    inliers: int = 0  # agreeing depth points
    depth: NDArray[np.float32] | None = None  # metric depth for fusion, on the grid of ``K_depth``
    K_depth: Array | None = None  # 3x3
    scene_m: float | None = None  # the frame's median scene depth (the distance limit)


@dataclass
class Result:
    placements: dict[int, Placement]
    calls: int = 0
    views: int = 0
    notes: dict[str, Any] = field(default_factory=dict)


# -- geometry ------------------------------------------------------------------------------------


def resized_K(K: Intrinsics, shape: tuple[int, ...]) -> Array:
    """The 3x3 intrinsics of ``K``'s image resampled to ``shape`` (rows, cols)."""
    sx, sy = shape[1] / K.width, shape[0] / K.height
    return np.array([[K.fx * sx, 0.0, K.cx * sx], [0.0, K.fy * sy, K.cy * sy], [0.0, 0.0, 1.0]])


def robust_ratio(ratios: list[float], floor_rel: float = 0.0) -> tuple[float, float]:
    """The median of ``ratios`` and its relative sigma (MAD of the logs over sqrt(n), combined
    with ``floor_rel``)."""
    r = np.log(np.array(ratios))
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    sigma = math.sqrt(mad ** 2 + floor_rel ** 2) / math.sqrt(len(r))
    return float(np.exp(np.median(r))), max(sigma, MIN_SCALE_SIGMA_REL)


def depth_ratio(target: NDArray[np.float32], depth: NDArray[np.float32]) -> float | None:
    """The median ratio of ``target`` to ``depth`` (same image, any grids) where both have one."""
    t = cv2.resize(target, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    ok = (t > MIN_DEPTH_M) & (depth > MIN_DEPTH_M) & np.isfinite(t) & np.isfinite(depth)
    if int(ok.sum()) < 100:
        return None
    return float(np.median(t[ok] / depth[ok]))


def baseline(poses: list[Array]) -> float:
    """The largest distance between two of the poses' centres (0 with fewer than two)."""
    if len(poses) < 2:
        return 0.0
    C = np.array([np.asarray(T)[:3, 3] for T in poses])
    return float(np.max(np.linalg.norm(C[:, None] - C[None], axis=2)))


def median_depth(d: NDArray[np.float32]) -> float | None:
    v = d[np.isfinite(d) & (d > 0)]
    return float(np.median(v)) if v.size else None


def _thumb(path: Path) -> NDArray[np.float32] | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    t = cv2.resize(img, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    t -= t.mean()
    n = float(np.linalg.norm(t))
    return t / n if n > 1e-6 else t


# -- the registration ----------------------------------------------------------------------------


@dataclass
class _Posed:
    """A view placed by this update or stored: a candidate anchor and reference."""

    key: str
    image: Path
    K: Intrinsics
    T: Array
    order: int
    depth: Callable[[], tuple[NDArray[np.float32], Array] | None]  # metric depth, grid's K
    _cache: list[tuple[NDArray[np.float32], Array] | None] = field(default_factory=list)

    def ref(self) -> Ref | None:
        if not self._cache:
            self._cache.append(self.depth())
        d = self._cache[0]
        return None if d is None else Ref(self.T, d[0], d[1])


def _stored_depth(s: Stored) -> Callable[[], tuple[NDArray[np.float32], Array] | None]:
    def load() -> tuple[NDArray[np.float32], Array] | None:
        d = s.depth()
        return None if d is None else (d, resized_K(s.K, d.shape))
    return load


class Registration:
    """The camera-only frames ``new`` against the map's keyframes ``stored`` (at least one)."""

    def __init__(self, multiview: Multiview, new: list[New], stored: list[Stored]) -> None:
        self.views_of = multiview
        self.new = new
        self.pool: list[_Posed] = [
            _Posed(s.name, s.image, s.K, s.T, s.order, _stored_depth(s)) for s in stored]
        self.placements: dict[int, Placement] = {}
        self.scenes: dict[int, float | None] = {}  # each new frame's median scene depth
        self.calls = 0
        self.views = 0
        self.max_new = MAX_VIEWS - MAX_ANCHORS
        self.max_anchors = MAX_ANCHORS
        self.unposed = False  # the last call ran without pose inputs
        self.notes: dict[str, Any] = {"calls": []}
        self._thumbs: dict[str, NDArray[np.float32] | None] = {}

    def _thumb(self, key: str, path: Path) -> NDArray[np.float32] | None:
        if key not in self._thumbs:
            self._thumbs[key] = _thumb(path)
        return self._thumbs[key]

    def anchors(self, chunk: list[New]) -> list[_Posed]:
        """The pool views most similar to the chunk by thumbnail among the most recent, the
        most recent first on ties."""
        recent = sorted(self.pool, key=lambda v: -v.order)[:RECENT_POOL]
        thumbs = [self._thumb(f"new{f.index}", f.image) for f in chunk]

        def sim(v: _Posed) -> float:
            t = self._thumb(v.key, v.image)
            vals = [float((t * c).sum()) for c in thumbs if c is not None and t is not None]
            return max(vals) if vals else 0.0

        return sorted(recent, key=lambda v: (-sim(v), -v.order))[:self.max_anchors]

    def _call(self, anchors: list[_Posed], chunk: list[New]) -> list[MvView]:
        images = [a.image for a in anchors] + [f.image for f in chunk]
        Ks = [a.K for a in anchors] + [f.K for f in chunk]
        poses: list[Array | None] = [a.T for a in anchors] + [None] * len(chunk)
        self.unposed = all(f.depth is None for f in chunk) \
            and baseline([a.T for a in anchors]) < POSED_BASELINE_M
        if self.unposed:
            poses = [None] * len(poses)
        self.calls += 1
        self.views += len(images)
        return self.views_of(images, Ks, poses)

    def _scale(self, anchors: list[_Posed], chunk: list[New], out: list[MvView]
               ) -> tuple[float, str]:
        """The call's scale (map metres per MapAnything unit) and its source: the frames' depth
        images, else the anchors' stored depth, else MapAnything's metres."""
        na = len(anchors)
        ratios = [r for i, f in enumerate(chunk) if f.depth is not None
                  for r in [depth_ratio(f.depth, out[na + i].depth)] if r]
        if ratios:
            return robust_ratio(ratios, 0.01)[0], "depth"
        stored = [r for i, a in enumerate(anchors) for ref in [a.ref()] if ref is not None
                  for r in [depth_ratio(ref.depth, out[i].depth)] if r]
        if stored:
            return robust_ratio(stored, 0.01)[0], "map"
        return 1.0, "mapanything"

    def _align(self, anchors: list[_Posed], chunk: list[New], out: list[MvView]
               ) -> tuple[Sim3, str, tuple[float, float]]:
        """The call's similarity onto the map, its scale source, and MapAnything's pose sigmas
        (position m, rotation deg) from the anchors' residuals."""
        refs = [Anchor(a.key, "previous", out[i].pose, a.T) for i, a in enumerate(anchors)]
        s, src = self._scale(anchors, chunk, out)
        S = anchor.fit(refs, scale=s).sim
        res = [anchor.residual(S, a) for a in refs]
        rp = float(np.sqrt(np.mean([r[0] ** 2 for r in res])))
        rr = float(np.sqrt(np.mean([r[1] ** 2 for r in res])))
        floor_r = UNPOSED_SIGMA_ROT_DEG if self.unposed else MA_SIGMA_ROT_DEG
        return S, src, (max(rp, MA_SIGMA_POS_M), max(rr, floor_r))

    def _place_chunk(self, chunk: list[New], out: list[MvView], anchors: list[_Posed], S: Sim3,
                     ma_sigma: tuple[float, float]) -> None:
        """Place the call's new frames. A frame with a depth image is checked against the
        anchors' stored depth; one without, against the anchors' depth from the same call (one
        network pass predicts consistent depths), each at its map-frame pose."""
        na = len(anchors)
        sensor = [r for a in anchors for r in [a.ref()] if r is not None]
        call = [Ref(a.T, (out[i].depth * S.s).astype(np.float32), out[i].K)
                for i, a in enumerate(anchors)]
        cand: dict[int, tuple[New, Array, tuple[float, float], NDArray[np.float32], Array,
                              NDArray[np.float32], Array]] = {}
        for i, f in enumerate(chunk):
            v = out[na + i]
            T = S.pose(v.pose)
            ma_depth = (v.depth * S.s).astype(np.float32)
            depth, Kd = (f.depth, resized_K(f.K, f.depth.shape)) if f.depth is not None \
                else (ma_depth, v.K)
            med = median_depth(depth)
            self.scenes[f.index] = med
            rel = UNPOSED_SIGMA_POS_REL if self.unposed else MA_SIGMA_POS_REL
            sig = (max(ma_sigma[0], rel * (med or 0.0)), ma_sigma[1])
            cand[f.index] = (f, T, sig, depth, Kd, ma_depth, v.K)
        changed = True
        while changed and cand:
            changed = False
            for idx in list(cand):
                f, T, sig, depth, Kd, ma_depth, Kma = cand[idx]
                refs = sensor if f.depth is not None else call
                tol = DEPTH_TOL if f.depth is not None else CALL_DEPTH_TOL
                a = agreement(backproject(depth, Kd), T, refs, tol)
                if not a.confirmed:
                    continue
                self._register(f, T, sig, depth, Kd, a.agree)
                sensor.append(Ref(T, depth, Kd))
                call.append(Ref(T, ma_depth, Kma))
                del cand[idx]
                changed = True
        for idx, (f, *_rest) in cand.items():
            self.placements[idx] = Placement(
                "rejected", "few_features" if f.features < FEW_FEATURES else "no_overlap",
                scene_m=self.scenes.get(idx))

    def _register(self, f: New, T: Array, sig: tuple[float, float], depth: NDArray[np.float32],
                  Kd: Array, inliers: int) -> None:
        self.placements[f.index] = Placement(
            "registered", None, T, round(max(sig[0], SIGMA_POS_FLOOR_M), 4),
            round(max(sig[1], SIGMA_ROT_FLOOR_DEG), 3), inliers, depth, Kd,
            self.scenes.get(f.index))
        got = (depth, Kd)
        self.pool.append(_Posed(f"new{f.index}", f.image, f.K, T, 10 ** 9 + f.index,
                                lambda: got))

    def run(self) -> Result:
        todo = list(self.new)
        while todo:
            chunk = todo[:self.max_new]
            anchors = self.anchors(chunk)
            try:
                out = self._call(anchors, chunk)
            except Exception as exc:
                if not out_of_memory(exc) or (len(chunk) == 1
                                                   and self.max_anchors <= MIN_ANCHORS):
                    raise
                if len(chunk) > 1:
                    self.max_new = max(1, len(chunk) // 2)
                else:
                    self.max_anchors = MIN_ANCHORS
                continue
            S, src, ma_sigma = self._align(anchors, chunk, out)
            self._place_chunk(chunk, out, anchors, S, ma_sigma)
            self.notes["calls"].append({"anchors": len(anchors), "new": len(chunk),
                                        "scale": round(S.s, 5), "scale_source": src,
                                        "unposed": self.unposed})
            todo = todo[len(chunk):]
        return Result(self.placements, self.calls, self.views, self.notes)
