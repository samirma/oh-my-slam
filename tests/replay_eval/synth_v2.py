"""A synthetic episode in the skill-v2 layout, with the sidecar, jobs and intrinsics shapes of
the conductor's e051 (values from it where they matter): runs/<run>/frames/<name>.{png,
depth.png,json}, frames/_select.json, oms/jobs.json, calibrate_camera.json or
camera_info_<id>.json. The truth track and poses come from the v1 builder."""

from __future__ import annotations

import json
import shutil
import struct
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import synth

SETS = [  # set name -> v2 frame prefix (letters of the set name)
    synth.SetSpec("r1", "look", [(0, 0, 0), (0, 0, 25), (0, 0, 50), (0, 0, 75)]),
    synth.SetSpec("r1", "probe", [(0.4, 0, 0), (0.8, 0, 0), (1.2, 0, 0), (1.6, 0, 0)]),
    synth.SetSpec("r1", "loc", [(0.4, 0, 5), (0.8, 0, 0)]),
    synth.SetSpec("r1", "close", [(1.2, 0, 180)]),
]
E051_CAMERA = {"fx_px": 655.8, "fy_px": 655.8, "cx_px": 319.5, "cy_px": 239.5,
               "width_px": 640, "height_px": 480}


def png16(w: int, h: int) -> bytes:
    """A 16-bit grey PNG (the depth image's shape; contents irrelevant)."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(
            ">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@dataclass
class SynthV2:
    root: Path
    truth: dict[str, np.ndarray] = field(default_factory=dict)  # v2 frame name -> T_world_cam
    names: dict[str, list[str]] = field(default_factory=dict)  # set -> v2 frame names
    start: np.ndarray = field(default_factory=lambda: np.eye(4))
    api: dict[str, str] = field(default_factory=dict)  # v2 frame name -> API frame name


def build_v2(root: Path, eid: str = "e051", camera: str = "calibrate",
             sets: list[synth.SetSpec] | None = None, run: str = "r1") -> SynthV2:
    """Episode ``eid`` in the v2 layout. ``camera``: calibrate (calibrate_camera.json),
    camera_info (camera_info_astra.json only) or none."""
    tmp = root / f".v1_{eid}"
    syn = synth.build(tmp, eid, sets=sets or SETS)
    ep = root / eid
    ep.mkdir(parents=True)
    for name in ("truth.jsonl", "meta.json", "verdict.md"):
        shutil.copy(syn.root / name, ep / name)
    out = SynthV2(ep, start=syn.start["r1"])
    run_dir = ep / "agent_files" / "runs" / run
    frames = run_dir / "frames"
    frames.mkdir(parents=True)
    for key, v1_names in syn.names.items():
        set_name = key.split("/")[1]
        prefix = "".join(c for c in set_name if c.isalpha())
        out.names[set_name] = []
        views = json.loads((syn.root / "agent_files" / "runs" / key / "views.json").read_text())
        for k, v1 in enumerate(v1_names):
            t = syn.times[v1]
            stamp = datetime.fromtimestamp(t, UTC).strftime("%Y%m%d-%H%M%S")
            name = f"{prefix}{k + 1:03d}-{stamp}"
            pose = next(v["pose"] for f, v in views.items() if Path(f).stem == Path(v1).name)
            (frames / f"{name}.png").write_bytes(synth.png(640, 480))
            (frames / f"{name}.depth.png").write_bytes(png16(640, 480))
            prior: Any = {"T_map_cam": pose, "sigma_pos_m": 0.0372, "sigma_rot_deg": 0.636,
                          "trust": "metric"}
            if set_name == "close":
                prior = None  # pose_state unknown: no prior
            (frames / f"{name}.json").write_text(json.dumps({
                "T_body_cam": [[0, 0, 1, -0.043645], [-1, 0, 0, 0], [0, -1, 0, 0.49555],
                               [0, 0, 0, 1]],
                "T_body_cam_source": "calibration", "camera_id": "astra",
                "depth": f"{name}.depth.png", "height_above_floor_m": 0.5,
                "height_px": 480, "height_sigma_m": None, "image": f"{name}.png", "name": name,
                "pose_source": "locate" if prefix == "loc" else "odometry",
                "pose_state": "unknown" if prior is None else "known", "prior": prior,
                "ros_stamp": t + 0.1, "stamp": t, "state_ros_stamp": t + 0.05,
                "width_px": 640}))
            out.truth[name] = syn.truth[v1]
            out.names[set_name].append(name)
            out.api[name] = f"{run}/frames/{name}"
    (frames / "_select.json").write_text(json.dumps({
        "counts": {"drop": 1, "gaps": 0, "send": 3}, "drop": [
            {"name": out.names["probe"][0], "reason": "prior_conflict"}],
        "send": out.names["probe"][1:], "status": "send_some"}))
    oms = run_dir / "oms"
    oms.mkdir()
    look, probe = out.names["look"], out.names["probe"]
    t_look = syn.times[syn.names["r1/look"][-1]] + 1
    t_probe = syn.times[syn.names["r1/probe"][-1]] + 1
    jobs = [{"accept_if": {"max_gauge_shift_m": 0.05}, "frames": look,
             "job_id": "2dab2e4523d4486a",
             "measured": {"n": len(look), "recorded": True, "s": 4.958, "source": "ended_at"},
             "seen_at": t_look + 17, "state": "committed", "submitted_at": t_look},
            {"accept_if": {"max_gauge_shift_m": 0.05, "max_rejected_share": 0.5},
             "frames": probe + ["ghost-20261110-000000"], "job_id": "77aa",
             "measured": {"n": len(probe), "recorded": True, "s": 6.1, "source": "ended_at"},
             "state": "committed",
             "submitted_at": datetime.fromtimestamp(t_probe, UTC).isoformat()}]
    (oms / "jobs.json").write_text(json.dumps({"jobs": jobs}))
    for name in ("dry_run.json", "frames.json", "grid.json", "health.json", "locate.json",
                 "objects.json"):
        (oms / name).write_text("{}")
    (run_dir / "log.jsonl").write_text("\n".join(json.dumps(
        {"by": "skill", "script": s, "command": f"{s} /home/u/x", "status": "ok", "code": 0,
         "seconds": 1.0, "t": 1.0}) for s in ("map.sh", "locate.sh")) + "\nnot json\n")
    if camera == "calibrate":
        (run_dir / "calibrate_camera.json").write_text(json.dumps(
            {"intrinsics": E051_CAMERA, "source": "camera_info"}))
    if camera in ("calibrate", "camera_info"):
        (run_dir / "camera_info_astra.json").write_text(json.dumps({"msg": {
            "K": [600.0, 0, 320.0, 0, 601.0, 240.0, 0, 0, 1], "width": 640, "height": 480,
            "D": [0.1, -0.05, 0.0, 0.0, 0.0], "distortion_model": "plumb_bob"}}))
    shutil.rmtree(tmp)
    return out
