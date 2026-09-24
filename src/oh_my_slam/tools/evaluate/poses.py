"""Pose accuracy of a map against the commanded head motion encoded in the capture names.

Poses are the ``camera_*_to_map`` transforms of the mapper's ``-t full`` OpenLABEL output; the map
frame is z-up (gravity-aligned). Yaw is the heading of the viewing direction about the map's up
axis, measured relative to frame 001 (the map frame aligned to frame 001) and compared with the
commanded yaw after wrapping to ±180°. Pitch direction compares the elevation of each ``up`` /
``down`` frame with the ``level`` frame of the same motion (sign only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.types import Pose
from oh_my_slam.mapping.store import MapReader
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import (
    Capture,
    level_siblings,
    same_heading_pairs,
    wrap_deg,
)
from oh_my_slam.tools.evaluate.scene import Json, frame_poses, frame_property, pitch_deg, yaw_deg

POSE_METRICS = ("registered_fraction", "yaw_err_median_deg", "yaw_err_max_deg",
                "pitch_direction_fraction", "same_heading_yaw_diff_max_deg", "centre_radius_m")


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


def pitch_directions(poses: dict[str, Pose], captures: list[Capture]
                     ) -> list[dict[str, Any]]:
    """For every registered ``up`` / ``down`` capture whose level sibling is registered: the pitch
    difference to the sibling and whether its sign is the named direction."""
    siblings = level_siblings(captures)
    rows = []
    for c in captures:
        level = siblings.get(c.name)
        if level is None or c.name not in poses or level.name not in poses:
            continue
        d = pitch_deg(poses[c.name].R) - pitch_deg(poses[level.name].R)
        rows.append({"capture": c.name, "level": level.name, "tilt": c.tilt,
                     "pitch_delta_deg": round(d, 2), "ok": d > 0 if c.tilt == "up" else d < 0})
    return rows


def pose_metrics(m: Metrics, prefix: str, poses: dict[str, Pose], captures: list[Capture]
                 ) -> list[dict[str, Any]]:
    """Record the pose metrics ``<prefix>.<name>`` (see ``POSE_METRICS``); returns per-capture
    rows for the report."""
    ids = {k: f"{prefix}.{k}" for k in POSE_METRICS}
    reg = [c for c in captures if c.name in poses]
    m.add(ids["registered_fraction"], len(reg) / len(captures) if captures else None,
          {"registered": len(reg), "captures": len(captures),
           "missing": [c.name for c in captures if c.name not in poses]},
          error=None if captures else "no captures")
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
    pitch = pitch_directions(poses, captures)
    by_name = {r["capture"]: r for r in rows}
    for p in pitch:
        by_name[p["capture"]].update(pitch_delta_deg=p["pitch_delta_deg"], pitch_ok=p["ok"])
    m.add(ids["pitch_direction_fraction"],
          sum(p["ok"] for p in pitch) / len(pitch) if pitch else None,
          {"evaluated": len(pitch), "wrong": [p["capture"] for p in pitch if not p["ok"]]},
          error=None if pitch else "no up/down capture with a registered level sibling")
    centres = np.array([poses[c.name].t for c in reg]).reshape(-1, 3)
    m.add(ids["centre_radius_m"],
          float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) if len(reg) else None,
          error=None if len(reg) else "no registered capture")
    return rows
