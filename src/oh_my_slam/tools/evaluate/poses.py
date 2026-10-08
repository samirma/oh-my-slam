"""Pose accuracy of a map against the commanded motion encoded in the capture names.

Poses are the ``camera_*_to_map`` transforms of the mapper's ``-t full`` OpenLABEL output; the map
frame is z-up (gravity-aligned). Yaw is the heading of the viewing direction about the map's up
axis, positive to the left. ``ainex-captures`` (``pose_metrics``): yaw measured relative to frame
001 (the map frame aligned to frame 001) and compared with the commanded yaw after wrapping to
±180°. ``camera`` (``pan_pose_metrics``), whose names give the pan order but not the step angle:
the heading must turn left from one pan position to the next and agree across the tilts of a pan
position. Pitch direction compares the elevation of each ``up`` / ``down`` frame with the
untilted frame of the same motion or pan position (``level`` / ``mid``; sign only)."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.types import Pose
from oh_my_slam.mapping.store import MapReader
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import (
    AnyCapture,
    Capture,
    PanCapture,
    level_siblings,
    mid_siblings,
    pan_positions,
    same_heading_pairs,
    tilt_pairs,
    wrap_deg,
)
from oh_my_slam.tools.evaluate.scene import Json, frame_poses, frame_property, pitch_deg, yaw_deg

POSE_METRICS = ("registered_fraction", "yaw_err_median_deg", "yaw_err_max_deg",
                "pitch_direction_fraction", "same_heading_yaw_diff_max_deg")
PAN_POSE_METRICS = ("registered_fraction", "pan_order_fraction", "tilt_yaw_diff_median_deg",
                    "tilt_yaw_diff_max_deg", "pitch_direction_fraction")


def capture_sources(doc: Json, map_dir: Path) -> dict[int, str]:
    """Frame key → capture file name: the frames' ``source`` property when exported, else the
    map's keyframe records (read-only)."""
    exported = frame_property(doc, "source")
    if exported:
        return {k: Path(str(v)).name for k, v in exported.items()}
    return {r.index: Path(r.source).name for r in MapReader(map_dir).frames}


def capture_poses(doc: Json, map_dir: Path) -> dict[str, Pose]:
    """Camera-to-map pose of every registered capture, by capture file name."""
    sources = capture_sources(doc, map_dir)
    return {sources[k]: p for k, p in frame_poses(doc).items() if k in sources}


def relative_yaws(poses: dict[str, Pose], reference: str) -> dict[str, float]:
    """Yaw of each pose relative to ``reference``'s, wrapped to ±180°."""
    y0 = yaw_deg(poses[reference].R)
    return {n: wrap_deg(yaw_deg(p.R) - y0) for n, p in poses.items()}


def circular_mean_deg(angles: list[float]) -> float:
    """The mean direction of ``angles`` (degrees), wrapped to ±180°."""
    return float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(angles))))))


def pitch_directions(poses: dict[str, Pose], tilted: list[tuple[str, str, str]]
                     ) -> list[dict[str, Any]]:
    """For every registered ``up`` / ``down`` capture of ``tilted`` (capture, tilt, its untilted
    sibling) whose sibling is registered: the pitch difference to the sibling and whether its sign
    is the named direction."""
    rows = []
    for name, tilt, sibling in tilted:
        if name not in poses or sibling not in poses:
            continue
        d = pitch_deg(poses[name].R) - pitch_deg(poses[sibling].R)
        rows.append({"capture": name, "sibling": sibling, "tilt": tilt,
                     "pitch_delta_deg": round(d, 2), "ok": d > 0 if tilt == "up" else d < 0})
    return rows


def _registered(m: Metrics, mid: str, poses: dict[str, Pose],
                captures: Sequence[AnyCapture]) -> None:
    reg = [c for c in captures if c.name in poses]
    m.add(mid, len(reg) / len(captures) if captures else None,
          {"registered": len(reg), "captures": len(captures),
           "missing": [c.name for c in captures if c.name not in poses]},
          error=None if captures else "no captures")


def _pitch(m: Metrics, mid: str, rows: list[dict[str, Any]], pitch: list[dict[str, Any]],
           untilted: str) -> None:
    """``pitch_direction_fraction``; the per-capture rows get each capture's pitch."""
    by_name = {r["capture"]: r for r in rows}
    for p in pitch:
        by_name[p["capture"]].update(pitch_delta_deg=p["pitch_delta_deg"], pitch_ok=p["ok"])
    m.add(mid, sum(p["ok"] for p in pitch) / len(pitch) if pitch else None,
          {"evaluated": len(pitch), "wrong": [p["capture"] for p in pitch if not p["ok"]]},
          error=f"no up/down capture with a registered {untilted} sibling")


def pose_metrics(m: Metrics, prefix: str, poses: dict[str, Pose], captures: list[Capture]
                 ) -> list[dict[str, Any]]:
    """Record the pose metrics ``<prefix>.<name>`` (see ``POSE_METRICS``); returns per-capture
    rows for the report."""
    ids = {k: f"{prefix}.{k}" for k in POSE_METRICS}
    _registered(m, ids["registered_fraction"], poses, captures)
    rows: list[dict[str, Any]] = [{"capture": c.name, "commanded_yaw_deg": c.yaw_deg,
                                   "registered": c.name in poses} for c in captures]
    ref = captures[0].name if captures else ""
    if ref in poses:
        rel = relative_yaws(poses, ref)
        errs = []
        for row in rows:
            if row["registered"]:
                est = rel[row["capture"]]
                err = wrap_deg(est - row["commanded_yaw_deg"])
                row.update(yaw_deg=round(est, 2), yaw_err_deg=round(err, 2))
                errs.append(abs(err))
        m.add(ids["yaw_err_median_deg"], float(np.median(errs)))
        m.add(ids["yaw_err_max_deg"], float(np.max(errs)),
              {"worst": max(rows, key=lambda r: abs(r.get("yaw_err_deg", 0.0)))["capture"]})
        diffs = {f"{a.name}~{b.name}": round(abs(wrap_deg(rel[a.name] - rel[b.name])), 2)
                 for a, b in same_heading_pairs(captures) if a.name in poses and b.name in poses}
        m.add(ids["same_heading_yaw_diff_max_deg"], max(diffs.values()) if diffs else None,
              diffs, error=None if diffs else "no same-heading pair registered")
    else:
        m.fail([ids["yaw_err_median_deg"], ids["yaw_err_max_deg"],
                ids["same_heading_yaw_diff_max_deg"]], f"reference frame {ref} is not registered")
    siblings = level_siblings(captures)
    _pitch(m, ids["pitch_direction_fraction"], rows, pitch_directions(
        poses, [(c.name, c.tilt, s.name) for c in captures if (s := siblings.get(c.name))]),
        "level")
    return rows


def pan_pose_metrics(m: Metrics, prefix: str, poses: dict[str, Pose], captures: list[PanCapture]
                     ) -> list[dict[str, Any]]:
    """Record the pose metrics ``<prefix>.<name>`` of a pan-tilt sequence (see
    ``PAN_POSE_METRICS``): the heading of each pan position (the circular mean of its registered
    captures' yaws) turns left from one pan position to the next (``pan_order_fraction``: the
    share of such steps), the tilts of one pan position share its heading (the yaw differences of
    ``tilt_pairs``), and the ``up`` / ``down`` captures are pitched that way from the ``mid`` one.
    Returns per-capture rows for the report, yaw relative to the first registered capture."""
    ids = {k: f"{prefix}.{k}" for k in PAN_POSE_METRICS}
    _registered(m, ids["registered_fraction"], poses, captures)
    reg = [c for c in captures if c.name in poses]
    rel = relative_yaws(poses, reg[0].name) if reg else {}
    rows: list[dict[str, Any]] = [
        {"capture": c.name, "pan": c.pan, "tilt": c.tilt, "registered": c.name in poses,
         **({"yaw_deg": round(rel[c.name], 2)} if c.name in poses else {})} for c in captures]
    heading = {pan: circular_mean_deg([rel[c.name] for c in group])
               for pan, group in pan_positions(reg).items()}
    steps = [(a, b, wrap_deg(heading[b] - heading[a])) for a, b in pairwise(heading)]
    m.add(ids["pan_order_fraction"], sum(d > 0 for *_, d in steps) / len(steps) if steps else None,
          {"steps_deg": {f"p{a:02d}~p{b:02d}": round(d, 2) for a, b, d in steps},
           "wrong": [f"p{a:02d}~p{b:02d}" for a, b, d in steps if d <= 0]},
          error="fewer than two pan positions registered")
    diffs = {f"{a.name}~{b.name}": abs(wrap_deg(rel[a.name] - rel[b.name]))
             for a, b in tilt_pairs(captures) if a.name in poses and b.name in poses}
    none = "no pan position has two registered tilts"
    m.add(ids["tilt_yaw_diff_median_deg"],
          float(np.median(list(diffs.values()))) if diffs else None, error=none)
    m.add(ids["tilt_yaw_diff_max_deg"], max(diffs.values()) if diffs else None,
          {k: round(v, 2) for k, v in diffs.items()}, error=none)
    mids = mid_siblings(captures)
    _pitch(m, ids["pitch_direction_fraction"], rows, pitch_directions(
        poses, [(c.name, c.tilt, s.name) for c in captures if (s := mids.get(c.name))]), "mid")
    return rows
