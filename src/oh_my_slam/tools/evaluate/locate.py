"""``mapper.sh locate`` (specs/mapper.md): the located cameras of its OpenLABEL output, and pose
accuracy of held-out captures (spec §5 "Pose accuracy": estimated camera yaw against the headings
encoded in the capture names, and the fraction of frames successfully registered).

The held-out captures are the images of the split map's second update, located against the map
after its first update (which has not seen them). Their yaw relative to the map's pose of the
sequence's first capture (``001``) is compared with a heading relative to the same capture: the
commanded one (``ainex-captures``), as for the map's own keyframes (``poses``), or — for
``examples/camera``, whose step angle is not recorded — the yaw the split map gives the capture
once an update has added it. The detail also gives, per image, the difference between the located
pose and the pose the map gives the same capture once an update has added it."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.types import Pose
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import AnyCapture, Capture, wrap_deg
from oh_my_slam.tools.evaluate.poses import relative_yaws
from oh_my_slam.tools.evaluate.scene import Json, frame_poses, frame_property, yaw_deg

LOCATE_METRICS = ("located_fraction", "yaw_err_median_deg", "yaw_err_max_deg")


def located_poses(doc: Json) -> dict[str, Pose]:
    """Camera-to-map pose of every located input image, by file name (frames marked
    ``located``)."""
    marked = frame_property(doc, "located")
    images = frame_property(doc, "image")
    poses = frame_poses(doc)
    return {Path(str(images[k])).name: poses[k] for k, v in marked.items()
            if v is True and k in poses and k in images}


def located_problems(doc: Json, asked: int) -> list[str]:
    """Located cameras distinguishable from the map's own frames (mapper.md): each located frame
    is marked ``located``, names its image and has its own pose; no more of them than inputs."""
    marked = frame_property(doc, "located")
    images = frame_property(doc, "image")
    poses = frame_poses(doc)
    out = [f"located frame {k} has no image or pose" for k, v in marked.items()
           if v is True and (k not in images or k not in poses)]
    n = sum(v is True for v in marked.values())
    if n > asked:
        out.append(f"{n} located frames for {asked} input images")
    return out


def rotation_deg(a: Pose, b: Pose) -> float:
    c = (np.trace(a.R.T @ b.R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def headings(held_out: Sequence[AnyCapture], mapped_later: dict[str, Pose] | None,
             reference_name: str | None) -> dict[str, float]:
    """The heading each held-out capture's located yaw is judged against, relative to the
    reference capture's: the commanded one (``Capture``), else (a ``PanCapture``, whose step angle
    is not recorded) the yaw the map gives it once an update has added it (``mapped_later``)."""
    later = mapped_later or {}
    rel = relative_yaws(later, reference_name) if reference_name in later else {}
    return {c.name: c.yaw_deg if isinstance(c, Capture) else rel[c.name] for c in held_out
            if isinstance(c, Capture) or c.name in rel}


def held_out_metrics(m: Metrics, prefix: str, located: dict[str, Pose] | None,
                     reference: Pose | None, held_out: Sequence[AnyCapture],
                     mapped_later: dict[str, Pose] | None, reference_name: str | None = None
                     ) -> list[dict[str, Any]]:
    """``<prefix>.located_fraction`` and the yaw errors of the located ``held_out`` captures
    (``reference``: the map's pose of the reference capture ``reference_name``, the sequence's
    first; ``mapped_later``: the poses the map gives them once added, for the detail and, for a
    pan-tilt capture, its heading: ``headings``)."""
    ids = [f"{prefix}.{k}" for k in LOCATE_METRICS]
    if located is None:
        m.fail(ids, "mapper.sh locate failed")
        return []
    m.add(ids[0], len(located) / len(held_out) if held_out else None,
          {"located": len(located), "asked": len(held_out)}, error="no held-out capture")
    heading = headings(held_out, mapped_later, reference_name)
    rows: list[dict[str, Any]] = []
    for c in held_out:
        row: dict[str, Any] = {"capture": c.name}
        if isinstance(c, Capture):
            row["commanded_yaw_deg"] = c.yaw_deg
        p = located.get(c.name)
        if p is None:
            row["located"] = False
        elif reference is not None:
            yaw = wrap_deg(yaw_deg(p.R) - yaw_deg(reference.R))
            row.update(located=True, yaw_deg=round(yaw, 2))
            if c.name in heading:
                row["yaw_err_deg"] = round(abs(wrap_deg(yaw - heading[c.name])), 2)
            later = (mapped_later or {}).get(c.name)
            if later is not None:
                row.update(vs_mapped_rot_deg=round(rotation_deg(p, later), 2),
                           vs_mapped_m=round(float(np.linalg.norm(p.t - later.t)), 3))
        rows.append(row)
    errs = [r["yaw_err_deg"] for r in rows if "yaw_err_deg" in r]
    why = ("the map has no pose of the sequence's first capture" if reference is None else
           "no capture was located with a heading to judge it by")
    m.add(ids[1], float(np.median(errs)) if errs else None, error=why)
    m.add(ids[2], float(max(errs)) if errs else None, error=why)
    return rows
