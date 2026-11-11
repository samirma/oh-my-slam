"""Calibrated pose sigmas and the conflict with a prior (api.md §4 "Pose sigmas", "Conflict with
the prior", §6; README "Calibrated sigmas").

**Calibration.** A registration's own covariance (thousands of correlated matches, depth that is
misregistered to the colour, references whose poses are themselves uncertain) is far tighter
than its real error. The client measured, on replay truth: with depth images a reported
1.7 mm / 0.24° against an actual 7.2 mm / 1.06°; without depth a reported 0.031° against an
actual 2.1°. So every image pose's sigma is ``SIGMA_SCALE`` times its own covariance sigma,
combined in quadrature with the *image-registration noise floor* of its data kind
(``NOISE_FLOORS``: depth image or not, and the trust of the frame's prior: ``metric``,
``drifting``, ``none``). The floors are the documented model, tunable without code:
``OH_MY_SLAM_SIGMA_FLOORS`` (JSON ``{"<depth>/<trust>": [m, deg], ...}``, the keys of
``NOISE_FLOORS``) and ``OH_MY_SLAM_SIGMA_SCALE``.

**Convention.** Every sigma is 3-D RMS (api.md §6 "Sigma convention"): σ² is the trace of the
position or rotation covariance, for update and locate and every engine; the noise floors are
3-D RMS errors too (the client's measurements are error norms).

**Conflict.** A hypothesis conflicts with a prior when its distance from the prior exceeds
``K_SIGMA`` × sqrt(σ_prior² + σ_hypothesis²), position and rotation tested separately; σ_hypothesis
is the hypothesis's image-only calibrated sigma (before fusion with the prior, no rival
inflation), σ_prior the prior's (drift-grown for a ``drifting`` prior). Hypotheses are ranked by
hypot(dp / sqrt(σp² + σh,p²), dr / sqrt(σr² + σh,r²)), the same normalisation."""

from __future__ import annotations

import json
import math
import os

K_SIGMA = 3.0
SIGMA_SCALE_ENV = "OH_MY_SLAM_SIGMA_SCALE"
SIGMA_FLOORS_ENV = "OH_MY_SLAM_SIGMA_FLOORS"
LOCATE_MAP_SIGMA_ENV = "OH_MY_SLAM_LOCATE_MAP_SIGMA"
SIGMA_SCALE = 1.0  # the covariance sigma's factor before the floor is added
DEPTH, NO_DEPTH = "depth", "nodepth"
TRUSTS = ("metric", "drifting", "none")
# (m, deg): the image-registration noise of one image pose, by data kind (module docstring).
# With depth: the measured 7.2 mm / 1.06° of a pose fused with a metric prior of 2° (the default)
# needs an image noise of about 8 mm / 1.25° (1 / 1.06² = 1 / 1.25² + 1 / 2²). Without depth:
# the measured 2.1° is the image pose's own, plus monocular scale (10 %) on a 0.5 m baseline.
# Without a prior, nothing bounds the chain: half again.
NOISE_FLOORS: dict[str, tuple[float, float]] = {
    f"{DEPTH}/metric": (0.008, 1.25),
    f"{DEPTH}/drifting": (0.008, 1.25),
    f"{DEPTH}/none": (0.012, 1.25),
    f"{NO_DEPTH}/metric": (0.05, 2.1),
    f"{NO_DEPTH}/drifting": (0.05, 2.1),
    f"{NO_DEPTH}/none": (0.075, 2.1),
}


def floors() -> dict[str, tuple[float, float]]:
    """``NOISE_FLOORS`` with the overrides of ``OH_MY_SLAM_SIGMA_FLOORS``."""
    out = dict(NOISE_FLOORS)
    raw = os.environ.get(SIGMA_FLOORS_ENV, "").strip()
    if raw:
        given = json.loads(raw)
        for k, v in given.items():
            if k not in NOISE_FLOORS or len(v) != 2 or min(float(x) for x in v) < 0:
                raise ValueError(f"{SIGMA_FLOORS_ENV}: {k!r}: expected one of "
                                 f"{sorted(NOISE_FLOORS)} with [m, deg] >= 0")
            out[k] = (float(v[0]), float(v[1]))
    return out


def scale() -> float:
    """``SIGMA_SCALE``, or ``OH_MY_SLAM_SIGMA_SCALE``."""
    raw = os.environ.get(SIGMA_SCALE_ENV, "").strip()
    v = float(raw) if raw else SIGMA_SCALE
    if not v > 0:
        raise ValueError(f"{SIGMA_SCALE_ENV}={raw!r}: expected a positive number")
    return v


def locate_map_sigma() -> bool:
    """Whether a located pose's image-only sigma is combined with its reference keyframes' map
    sigmas (api.md §6 "Locate sigmas": only while ``locate.sigma_containment`` and
    ``locate.sigma_tightness_bound`` both hold, which only a replay shows):
    ``OH_MY_SLAM_LOCATE_MAP_SIGMA`` ``1``; off by default (σ_rel alone)."""
    raw = os.environ.get(LOCATE_MAP_SIGMA_ENV, "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{LOCATE_MAP_SIGMA_ENV}={raw!r}: expected 0 or 1")


def kind(depth: bool, trust: str | None) -> str:
    """The data kind of an image pose: ``<depth|nodepth>/<metric|drifting|none>``."""
    t = trust if trust in ("metric", "drifting") else "none"
    return f"{DEPTH if depth else NO_DEPTH}/{t}"


def floor_of(depth: bool, trust: str | None) -> tuple[float, float]:
    """The noise floor (m, deg) of a data kind."""
    return floors()[kind(depth, trust)]


def calibrated(sp: float, sr: float, depth: bool, trust: str | None) -> tuple[float, float]:
    """An image pose's calibrated sigmas (m, deg) from its own covariance sigmas."""
    fp, fr = floor_of(depth, trust)
    k = scale()
    return math.hypot(k * sp, fp), math.hypot(k * sr, fr)


def bounds(prior: tuple[float, float], hyp: tuple[float, float]) -> tuple[float, float]:
    """The conflict bounds (m, deg): ``K_SIGMA`` × sqrt(σ_prior² + σ_hypothesis²) each."""
    return K_SIGMA * math.hypot(prior[0], hyp[0]), K_SIGMA * math.hypot(prior[1], hyp[1])


def conflicts(dp: float, dr: float, prior: tuple[float, float], hyp: tuple[float, float],
              position: bool = True) -> bool:
    """Whether a hypothesis ``dp`` m and ``dr`` deg from the prior conflicts with it; without
    ``position`` only its rotation is tested."""
    bp, br = bounds(prior, hyp)
    return (position and dp > bp) or dr > br


def distance(dp: float, dr: float, prior: tuple[float, float], hyp: tuple[float, float]) -> float:
    """The ranking distance to the prior over the combined sigmas."""
    return math.hypot(dp / max(math.hypot(prior[0], hyp[0]), 1e-9),
                      dr / max(math.hypot(prior[1], hyp[1]), 1e-9))
