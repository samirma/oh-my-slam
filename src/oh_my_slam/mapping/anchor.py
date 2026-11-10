"""Similarity transforms and the robust fit that anchors the map frame (map.md §1).

The mapper keeps its own internal frame, which a rebuild may define again. The service frame is
fixed by a map's first update and reached from the internal frame by one similarity per version,
``S`` (``x_map = s R x_internal + t``). Each update fits ``S`` to its *anchors*: camera poses
whose map-frame value is known — the previous version's poses of the frames both versions hold,
and the metric priors (or, on a map's first update without metric priors, the drifting ones). An
anchor constrains the camera centre (``sigma_pos_m``) and its orientation (``sigma_rot_deg``).

The fit is robust: hypotheses from single anchors (rotation and translation) and pairs (scale),
scored by the anchors within ``K_SIGMA`` of them, then a weighted Procrustes refinement of
centres and orientations together on the inliers. A pose that disagrees with the majority
cannot pull the map: its residual is reported (``prior_conflict``)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

K_SIGMA = 3.0  # an anchor whose residual exceeds this many sigmas is an outlier
MIN_SIGMA_POS_M = 0.01  # floors of the anchors' sigmas: a perfect prior still has noise
MIN_SIGMA_ROT_DEG = 0.5
DEFAULT_SIGMA_POS_M = 0.05  # a metric prior without sigmas (api.md §3)
DEFAULT_SIGMA_ROT_DEG = 2.0
DRIFTING_SIGMA_POS_M = 0.20  # a drifting prior without sigmas (api.md §3)
DRIFTING_SIGMA_ROT_DEG = 5.0
SCALE_OBSERVABLE_SIGMAS = 5.0  # centres must spread over this many position sigmas to fix scale
MAX_HYPOTHESES = 400
REFINE_ITERATIONS = 20


def default_sigmas(trust: str) -> tuple[float, float]:
    """The fixed sigmas (m, deg) of a prior sent without any (api.md §3): ``drifting`` 0.20 m /
    5°, ``metric`` 0.05 m / 2°."""
    if trust == "drifting":
        return DRIFTING_SIGMA_POS_M, DRIFTING_SIGMA_ROT_DEG
    return DEFAULT_SIGMA_POS_M, DEFAULT_SIGMA_ROT_DEG


def prior_sigmas(p: dict[str, Any]) -> tuple[float, float]:
    """A request ``Prior``'s sigmas (m, deg): its own, else its trust's defaults."""
    dp, dr = default_sigmas(str(p.get("trust", "metric")))
    return float(p.get("sigma_pos_m", dp)), float(p.get("sigma_rot_deg", dr))


def rotation_deg(R: Array) -> float:
    """The angle of rotation ``R``, in degrees."""
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def nearest_rotation(M: Array) -> Array:
    """The rotation closest to ``M`` (Frobenius)."""
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt))) or 1.0])
    out: Array = U @ D @ Vt
    return out


@dataclass(frozen=True)
class Sim3:
    """``x' = s R x + t``."""

    s: float = 1.0
    R: Array = field(default_factory=lambda: np.eye(3))
    t: Array = field(default_factory=lambda: np.zeros(3))

    def point(self, x: Array) -> Array:
        out: Array = self.s * (np.asarray(x, np.float64) @ self.R.T) + self.t
        return out

    def pose(self, T: Array) -> Array:
        """A camera pose ``T_internal_cam`` (4x4) as ``T_map_cam``: rotation and centre move,
        the camera's own axes and units do not."""
        T = np.asarray(T, np.float64)
        out = np.eye(4)
        out[:3, :3] = self.R @ T[:3, :3]
        out[:3, 3] = self.point(T[:3, 3])
        return out

    def compose(self, other: Sim3) -> Sim3:
        """``self ∘ other`` (``other`` first)."""
        return Sim3(self.s * other.s, self.R @ other.R, self.s * self.R @ other.t + self.t)

    def inverse(self) -> Sim3:
        Ri = self.R.T
        return Sim3(1.0 / self.s, Ri, -(Ri @ self.t) / self.s)

    def to_dict(self) -> dict[str, Any]:
        return {"s": self.s, "R": self.R.tolist(), "t": self.t.tolist()}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Sim3:
        return Sim3(float(d["s"]), np.asarray(d["R"], np.float64), np.asarray(d["t"], np.float64))


@dataclass
class Anchor:
    """A camera whose internal pose ``src`` should land on ``dst`` in the map frame."""

    name: str
    kind: str  # "prior" (metric), "drifting" or "previous"
    src: Array  # 4x4 internal pose
    dst: Array  # 4x4 map-frame pose
    sigma_pos_m: float = DEFAULT_SIGMA_POS_M
    sigma_rot_deg: float = DEFAULT_SIGMA_ROT_DEG

    @property
    def sp(self) -> float:
        return max(self.sigma_pos_m, MIN_SIGMA_POS_M)

    @property
    def sr(self) -> float:
        return math.radians(max(self.sigma_rot_deg, MIN_SIGMA_ROT_DEG))


def residual(S: Sim3, a: Anchor) -> tuple[float, float]:
    """Position (m) and rotation (deg) residuals of anchor ``a`` under ``S``."""
    T = S.pose(a.src)
    return (float(np.linalg.norm(T[:3, 3] - a.dst[:3, 3])),
            rotation_deg(a.dst[:3, :3].T @ T[:3, :3]))


def chi(S: Sim3, a: Anchor) -> float:
    """The larger of an anchor's position and rotation residuals, in sigmas."""
    dp, dr = residual(S, a)
    return max(dp / a.sp, math.radians(dr) / a.sr)


@dataclass
class Fit:
    sim: Sim3
    inliers: list[bool]
    scale_observable: bool
    scale_sigma_rel: float | None  # from the residuals, when the fit fixed the scale


def _refine(anchors: list[Anchor], use: list[bool], S: Sim3, free_scale: bool) -> Sim3:
    """Weighted Procrustes of centres and orientations over the anchors ``use`` marks."""
    A = [a for a, u in zip(anchors, use, strict=True) if u]
    wc = np.array([1.0 / a.sp ** 2 for a in A])
    wr = np.array([1.0 / (2.0 * a.sr ** 2) for a in A])
    P = np.array([a.src[:3, 3] for a in A])
    Q = np.array([a.dst[:3, 3] for a in A])
    pm = (wc[:, None] * P).sum(0) / wc.sum()
    qm = (wc[:, None] * Q).sum(0) / wc.sum()
    Pc, Qc = P - pm, Q - qm
    N = sum((w * a.dst[:3, :3] @ a.src[:3, :3].T for w, a in zip(wr, A, strict=True)),
            np.zeros((3, 3)))
    C = (wc[:, None, None] * (Qc[:, :, None] @ Pc[:, None, :])).sum(0)
    s, R = S.s, S.R
    for _ in range(REFINE_ITERATIONS):
        R = nearest_rotation(s * C + N)
        if free_scale:
            den = float((wc * (Pc ** 2).sum(1)).sum())
            num = float((wc * np.einsum("ij,ij->i", Qc, Pc @ R.T)).sum())
            if den > 1e-12 and num > 0:
                s = num / den
    return Sim3(s, R, qm - s * R @ pm)


def _spread(anchors: list[Anchor], use: list[bool]) -> float:
    """RMS distance of the used anchors' map-frame centres from their mean."""
    Q = np.array([a.dst[:3, 3] for a, u in zip(anchors, use, strict=True) if u])
    return float(np.sqrt(((Q - Q.mean(0)) ** 2).sum(1).mean()))


def _hypotheses(anchors: list[Anchor], scale: float | None, rng: np.random.Generator
                ) -> list[Sim3]:
    n = len(anchors)
    out = []
    singles = range(n) if n <= MAX_HYPOTHESES // 2 else rng.choice(n, MAX_HYPOTHESES // 2,
                                                                   replace=False)
    for i in singles:
        a = anchors[int(i)]
        R = a.dst[:3, :3] @ a.src[:3, :3].T
        s = 1.0 if scale is None else scale
        out.append(Sim3(s, R, a.dst[:3, 3] - s * R @ a.src[:3, 3]))
    if scale is None and n >= 2:
        for _ in range(min(MAX_HYPOTHESES // 2, n * (n - 1) // 2)):
            i, j = (int(v) for v in rng.choice(n, 2, replace=False))
            a, b = anchors[i], anchors[j]
            d_src = float(np.linalg.norm(a.src[:3, 3] - b.src[:3, 3]))
            if d_src < 1e-9:
                continue
            s = float(np.linalg.norm(a.dst[:3, 3] - b.dst[:3, 3])) / d_src
            if s <= 0:
                continue
            R = nearest_rotation(a.dst[:3, :3] @ a.src[:3, :3].T + b.dst[:3, :3] @ b.src[:3, :3].T)
            t = (a.dst[:3, 3] + b.dst[:3, 3]) / 2 - s * R @ (a.src[:3, 3] + b.src[:3, 3]) / 2
            out.append(Sim3(s, R, t))
    return out


def fit(anchors: list[Anchor], scale: float | None = None, seed: int = 0) -> Fit:
    """The robust similarity bringing ``anchors`` onto their map-frame poses. With ``scale``
    the scale is held; without it, it is fitted when the inliers' centres spread over
    ``SCALE_OBSERVABLE_SIGMAS`` position sigmas, else held at the best hypothesis' estimate and
    ``scale_observable`` is false (the caller then holds another scale)."""
    if not anchors:
        raise ValueError("no anchors to fit")
    rng = np.random.default_rng(seed)
    best: tuple[int, float, Sim3] | None = None
    for S in _hypotheses(anchors, scale, rng):
        c = [chi(S, a) for a in anchors]
        n_in = sum(v <= K_SIGMA for v in c)
        cost = float(sum(min(v, K_SIGMA) ** 2 for v in c))
        if best is None or (n_in, -cost) > (best[0], -best[1]):
            best = (n_in, cost, S)
    assert best is not None
    S = best[2]
    use = [chi(S, a) <= K_SIGMA for a in anchors]  # the hypothesis' own anchor at least
    free = scale is None
    sp = float(np.median([a.sp for a in anchors]))
    observable = free and sum(use) >= 2 and _spread(anchors, use) > SCALE_OBSERVABLE_SIGMAS * sp
    k = 0
    while True:  # refine, then again on the inliers of the refined fit (three times at most)
        S = _refine(anchors, use, S, observable)
        new = [chi(S, a) <= K_SIGMA for a in anchors]
        if new == use or not any(new) or k == 2:
            break
        use, k = new, k + 1
    sigma = None
    if observable:
        dp = np.array([residual(S, a)[0] for a, u in zip(anchors, use, strict=True) if u])
        rms = max(float(np.sqrt((dp ** 2).mean())), 0.1 * sp)
        sigma = rms / (_spread(anchors, use) * math.sqrt(max(sum(use) - 1, 1)))
    return Fit(S, use, observable, sigma)


def yaw_frame(T_first: Array, scale: float) -> Sim3:
    """The first-camera rule (map.md §1) on a gravity-aligned internal frame: the origin at the
    camera centre, +x its viewing direction projected onto the horizontal plane, +z up, and the
    internal metres times ``scale``."""
    z = T_first[:3, 2]
    yaw = math.atan2(float(z[1]), float(z[0])) if abs(z[0]) + abs(z[1]) > 1e-9 else 0.0
    c, s_ = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    return Sim3(scale, R, -scale * R @ T_first[:3, 3])
