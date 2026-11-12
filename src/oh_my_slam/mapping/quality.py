"""The quality report of an update and its gate (map.md §1, §4), shared by both engines and the
dry run: residual statistics, the new frames' displacement, the distance limit, the
re-registrations, and ``accept_if``."""

from __future__ import annotations

from typing import Any

import numpy as np

from oh_my_slam.mapping.anchored import TOO_CLOSE_INLIERS, TOO_CLOSE_M, Placement


def residual_stats(res: list[tuple[float, float]]) -> dict[str, Any] | None:
    """The ``Residuals`` object of (m, deg) residuals, or None without any."""
    if not res:
        return None
    m = np.array([r[0] for r in res])
    d = np.array([r[1] for r in res])
    return {"median_m": round(float(np.median(m)), 4), "max_m": round(float(m.max()), 4),
            "median_deg": round(float(np.median(d)), 3), "max_deg": round(float(d.max()), 3),
            "count": len(res)}


def too_close(scene_m: float | None, inliers: int) -> bool:
    """The distance limit (api.md §4): a median scene depth below ``TOO_CLOSE_M``, unless the
    frame has at least ``TOO_CLOSE_INLIERS`` inliers."""
    return scene_m is not None and scene_m < TOO_CLOSE_M and inliers < TOO_CLOSE_INLIERS


def displacements(pls: list[Placement | None]) -> list[tuple[float | None, float]]:
    """The new frames' displacement (map.md §1 "New-frame displacement"), as (m or None, deg)
    per registered frame: a frame with a prior, its committed pose against the prior as sent (a
    drifting prior as placed through the drift estimate: its prior residual); a frame without
    one, its committed pose against its image-only pose (``Placement.map_residual``), only its
    rotation (None in metres) when that pose is unconstrained in translation — rotation-only,
    or below the two-view parallax limit against every reference it registers to
    (``Placement.unconstrained``). Frames kept at their prior or rejected are left out."""
    out: list[tuple[float | None, float]] = []
    for pl in pls:
        if pl is None or pl.status != "registered":
            continue
        if pl.has_prior:  # a registered frame with a prior has its residual (origin: 0)
            out.append(pl.prior_residual or (0.0, 0.0))
            continue
        if pl.map_residual is None:
            continue
        dp = None if pl.method == "rotation" or pl.unconstrained else pl.map_residual[0]
        out.append((dp, pl.map_residual[1]))
    return out


def displacement_quality(moved: list[tuple[float | None, float]]) -> dict[str, float | None]:
    """``max_displacement_m`` and ``max_displacement_deg`` (map.md §4), null without any."""
    m = [d for d, _ in moved if d is not None]
    return {"max_displacement_m": round(max(m), 4) if m else None,
            "max_displacement_deg": round(max(r for _, r in moved), 3) if moved else None}


def drift_correction(moved: list[tuple[float, float]], odo_scale: float
                     ) -> dict[str, float] | None:
    """``quality.drift_correction`` (map.md §4, oms/2.5): the largest translation and rotation
    over the job's frames with a drifting prior between the prior as sent and as placed (one
    (m, deg) pair per frame: ``moved``) and the estimate's odometry scale; None without such a
    frame."""
    if not moved:
        return None
    return {"translation_m": round(max(m for m, _ in moved), 4),
            "rotation_deg": round(max(d for _, d in moved), 3),
            "odo_scale": round(odo_scale, 5)}


def reregistered(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``quality.reregistered`` (map.md §4) of the engine's re-registrations."""
    return [{k: e[k] for k in ("name", "displacement_m", "displacement_deg")} for e in entries]


def gate(q: dict[str, Any], kept_any: bool, accept_if: dict[str, float] | None,
         prior_frames_rejected: bool = False, rotation_inconsistent: bool = False,
         depth_inconsistent: bool = False) -> dict[str, Any]:
    """``accepted`` and ``failed_checks`` of a quality report (map.md §4): nothing kept is the
    hard limit, a map's first update whose frames with a prior are all rejected is
    ``prior_frames_rejected`` (map.md §1), and an update without priors whose committed relative
    rotations disagree with the images' is ``rotation_inconsistent`` (``mapping.standpoint``),
    an update two of whose committed views contradict each other's sensor depth images is
    ``depth_inconsistent`` (oms/2.6, ``standpoint.depth_consistency``). ``accept_if`` checks,
    each on its own and only when
    given: ``max_residual_m`` on ``previous_residuals.max_m`` (stored keyframes only; prior
    corrections never fail it), ``max_gauge_shift_m``, ``max_displacement_m`` and
    ``max_displacement_deg`` (a null value passes), ``max_prior_disagreement_deg`` and
    ``max_rejected_share``."""
    prev_res, gauge = q["previous_residuals"], q["gauge_shift"]
    failed = []
    if not kept_any:
        failed.append("hard_limit")
    if prior_frames_rejected:
        failed.append("prior_frames_rejected")
    if rotation_inconsistent:
        failed.append("rotation_inconsistent")
    if depth_inconsistent:
        failed.append("depth_inconsistent")
    a = accept_if or {}
    if "max_residual_m" in a and prev_res and prev_res["max_m"] > a["max_residual_m"]:
        failed.append("max_residual_m")
    if "max_gauge_shift_m" in a and gauge and gauge["translation_m"] > a["max_gauge_shift_m"]:
        failed.append("max_gauge_shift_m")
    for key in ("max_displacement_m", "max_displacement_deg"):
        if key in a and q[key] is not None and q[key] > a[key]:
            failed.append(key)
    if "max_prior_disagreement_deg" in a \
            and q["prior_disagreement_deg"] > a["max_prior_disagreement_deg"]:
        failed.append("max_prior_disagreement_deg")
    if "max_rejected_share" in a and q["rejected_share"] > a["max_rejected_share"]:
        failed.append("max_rejected_share")
    q["accepted"] = not failed
    q["failed_checks"] = failed
    return q
