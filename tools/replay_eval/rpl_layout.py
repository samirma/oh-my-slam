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
        "requests": "oms/agent_requests.jsonl",  # a shell-command log: only counted by the dump
        "meta": "meta.json",
        "runs": "agent_files/runs",  # runs/<run>/<set>/ holds one frame set
        "views": "views.json",  # beside the frames of a set
        "frame_suffixes": [".jpg", ".jpeg", ".png"],
    },
    # a date-time anywhere in a frame's file name (not every name has one, nor -NN); the agents
    # ran in Docker on UTC, so the offset to truth `t` (epoch) is 0
    "frame_time": {"pattern": r"(\d{8})-(\d{6})", "utc_offset_h": 0.0},
    # a frame whose set holds one name time per frame is matched within this many seconds of it;
    # frames sharing their set's time use the window between the neighbouring sets
    "frame_window_s": 3.0,
    "replay": {
        # sets whose name starts with these are locate-only (turn-in-place frames)
        "locate_prefixes": ["loc", "look"],
        # every n-th frame of an update set is held out of the update and located after it
        "holdout_every": 5,
    },
    "truth": {
        "kind": "kind",
        "t": "t",
        "kind_meta": "meta",
        "kind_robot": "robot",
        "kind_objects": "objects",
        # meta: {cameras, hello, kind, objects, placement, robot, robot_row, t, t0}
        "meta_robot": "robot",
        "meta_hello": "hello",  # hello: {engine, floor_z, rtf, scene, sim_time, starting, worktop}
        "scene": "scene",
        "floor_z": "floor_z",
        "worktop": "worktop",
        "worktop_z": "z",
        # robot lines: {bodies, cams, kind, pos, quat, sim_time, t} (+ joints on the arm)
        "pos": "pos",
        "quat": "quat",
        "quat_order": "wxyz",
        "cams": "cams",  # {camera name: optical centre [x, y, z] in the world}
        "bodies": "bodies",  # robot line: links; objects line: objects; each {pos, quat}
    },
    # views.json: {file name: value}; value is {"pose": 4x4, ...} or the bare 4x4 (arm)
    "views": {"pose": "pose", "stamp": "stamp", "base": "base"},
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
            "intrinsics": {"width": 640, "height": 480, "fx": 655.8, "fy": 655.8,
                           "cx": 319.5, "cy": 239.5},
        },
    },
    # the replays each robot kind gets with --priors both (evaluation.md: metric priors for the
    # base and the arm, trust none for camera-only robots; the arm replays with priors only)
    "default_modes": {"base": ["metric", "none"], "humanoid": ["none"], "arm": ["metric"]},
    # robots standing on the floor: height above floor = views pose[2][3] (start frame on the floor)
    "floor_robots": ["base", "humanoid"],
    # robot kind -> truth link (x forward, y left, z up) whose quaternion orients the camera; the
    # optical frame is that link rotated (optical x = -link y, y = -link z, z = link x). A kind
    # without one takes the views.json rotation composed with the run's start pose.
    "rotation_body": {"humanoid": "head_tilt_link", "base": "camera_link"},
    # robot kinds whose start pose is refined by a rigid fit of the views positions to the matched
    # truth camera positions (exact kinematics)
    "refine_start": ["arm"],
    "refine_iterations": 100,
    # robot kind -> truth body whose position is the start frame's origin, with identity rotation
    # (the arm's base link); when the body is missing, the base pose is refined as above
    "start_link": {"arm": "base_link"},
    # robot kinds whose start pose is fitted (yaw, translation and odometry scale, 2-D) from the
    # views odometry against the truth base track at the frames' name times, not the base quat
    "odometry_fit": ["base"],
    "odometry_fit_min_spread_m": 0.3,
    # the prior trust of the `metric` replay per robot kind (default metric): the base's
    # odometry over-counts by 5-10 %, so its priors are drifting, with sigma_pos_m the larger of
    # prior_sigma_pos_m and this share of the distance travelled since the run's start
    "prior_trust": {"base": "drifting"},
    "prior_sigma_travel_share": {"base": 0.10},
    # odometry matching of the base: cost = hypot(position error m, yaw_weight_m x yaw error rad)
    "odometry_yaw_weight_m": 0.5,
    # EXIF FocalPlaneResolutionUnit when the tag is absent: the conductor writes px per mm
    "exif_default_unit": "mm",
    # pass/fail: meta.json[verdict_key] when present (a bool, or text read with the words below),
    # else verdict.md: the earliest of the pass or fail words in it decides
    "verdict": {"file": "verdict.md", "meta_key": "verdict",
                "pass_words": ["pass", "success", "succeeded"],
                "fail_words": ["fail", "abort", "timeout", "timed out", "error"]},
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
