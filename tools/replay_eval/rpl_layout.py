"""The episode layout: every file name, JSON key and convention the readers rely on, in one table.

The values below are the layout the robot-guide conductor described on 2026-11-09
(``~/oh-my-slam-data/robot-guide/episode_layout.md``). Any of them can be overridden without a code
change with ``--layout FILE.json``: a JSON object deep-merged into ``LAYOUT``. ``--dump-layout``
prints what an episode folder actually holds, so a wrong key shows up in one run.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

LAYOUT: dict[str, Any] = {
    "files": {
        "truth": "truth.jsonl",
        "requests": "oms/agent_requests.jsonl",
        "runs": "agent_files/runs",  # runs/<run>/<set>/ holds one frame set
        "views": "views.json",  # beside the frames of a set
        "frame_suffixes": [".jpg", ".jpeg", ".png"],
    },
    # <set>-YYYYMMDD-HHMMSS-NN.jpg; the time is local time at a fixed UTC offset
    "frame_name": {
        "pattern": r"^(?P<set>.+)-(?P<date>\d{8})-(?P<time>\d{6})-(?P<nn>\d+)\.[A-Za-z]+$",
        "utc_offset_h": 1.0,
    },
    "truth": {
        "kind": "kind",
        "t": "t",
        "kind_meta": "meta",
        "kind_robot": "robot",
        "kind_objects": "objects",
        "meta_robot": "robot",
        "meta_hello": "hello",  # meta.hello holds scene, floor_z, worktop
        "scene": "scene",
        "floor_z": "floor_z",
        "worktop": "worktop",
        "worktop_z": "z",
        "pos": "pos",
        "quat": "quat",
        "quat_order": "wxyz",
        "cams": "cams",  # robot line: {camera name: optical centre [x, y, z] in the world}
        "bodies": "bodies",  # robot line: links; objects line: objects; each {pos, quat}
    },
    # views.json: {file name: value}; value is {"pose": 4x4, ...} or the bare 4x4 (arm)
    "views": {"pose": "pose", "stamp": "stamp"},
    "requests": {
        # a request line is classified by the first of these words found in its text
        "locate_words": ["locate"],
        "update_words": ["update"],
        "skip_words": ["segment", "reconstruct", "analyse", "delete", "revert", "health"],
        "dedupe_updates": True,  # a set sent again as an update is skipped (the agent's retry)
    },
    "robots": {
        "ainex": {
            "kind": "humanoid",
            "camera": "camera",
            "intrinsics": {"width": 640, "height": 480, "fx": 270.0, "fy": 270.0,
                           "cx": 319.5, "cy": 239.5},
        },
        "rebot_b601_rs": {
            "kind": "arm",
            "camera": "camera_color_optical_frame",
            "intrinsics": {"width": 640, "height": 360, "fx": 337.2, "fy": 337.2,
                           "cx": 319.5, "cy": 179.5},
        },
        "rosmaster_x3_plus": {
            "kind": "base",
            "camera": "camera_color_optical_frame",
            "intrinsics": "exif",  # fx from each frame's EXIF, cx, cy at the image centre
        },
    },
    # the replays each robot kind gets with --priors both (evaluation.md: metric priors for the
    # base and the arm, trust none for camera-only robots; the arm replays with priors only)
    "default_modes": {"base": ["metric", "none"], "humanoid": ["none"], "arm": ["metric"]},
    # robots standing on the floor: height above floor = views pose[2][3] (start frame on the floor)
    "floor_robots": ["base", "humanoid"],
    # robot kind -> truth body link whose quaternion is the camera's optical orientation; when
    # unset the truth orientation is the views.json rotation composed with the run's start pose
    "rotation_body": {},
    # EXIF FocalPlaneResolutionUnit when the tag is absent: the conductor writes px per mm
    "exif_default_unit": "mm",
    "prior_sigma_pos_m": 0.02,
    "prior_sigma_rot_deg": 1.0,
    "height_sigma_m": 0.02,
    # scenes without a floor plan (no room polygons): objects come from truth.jsonl
    "no_floorplan_scene_prefixes": ["ithor"],
    # THOR object types whose children are inside them rather than on them
    "container_types": ["Box", "Bowl", "Cup", "Mug", "Pot", "Pan", "Drawer", "Cabinet", "Fridge",
                        "Microwave", "Safe", "GarbageCan", "Sink", "SinkBasin", "Bathtub",
                        "BathtubBasin", "Toilet", "LaundryHamper", "Dresser"],
}


class LayoutError(ValueError):
    """The episode does not match the layout: the message names the file and the missing key."""


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_layout(path: Path | None) -> dict[str, Any]:
    """``LAYOUT``, with ``path`` (a JSON object) merged into it."""
    if path is None:
        return copy.deepcopy(LAYOUT)
    override = json.loads(path.read_text())
    if not isinstance(override, dict):
        raise LayoutError(f"{path}: a layout override must be a JSON object")
    return merge(LAYOUT, override)


def need(obj: Any, key: str, where: str) -> Any:
    """``obj[key]``, or a ``LayoutError`` that names the key, where it was looked for and the keys
    that are there."""
    if not isinstance(obj, dict):
        raise LayoutError(f"{where}: expected a JSON object holding '{key}', got "
                          f"{type(obj).__name__}")
    if key not in obj:
        have = ", ".join(sorted(map(str, obj))[:30])
        raise LayoutError(f"{where}: missing key '{key}' (keys present: {have}); fix the key in "
                          f"--layout")
    return obj[key]
