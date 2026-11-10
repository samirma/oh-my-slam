"""The skill-v2 episode layout: ``agent_files/runs/<run>/frames/<name>.{png,depth.png,json}``.

Each frame has a JSON sidecar: ``{name, image, depth, camera_id, width_px, height_px, stamp,
T_body_cam, height_above_floor_m, height_sigma_m, pose_source, pose_state, prior {T_map_cam,
sigma_pos_m, sigma_rot_deg, trust}}``; any field may be null or missing. ``oms/jobs.json`` lists
the update jobs ``{job_id, frames, accept_if, state, submitted_at, measured}``.

Replay plan: every job, in submission order, with its frames, priors, depth and accept_if;
the ``loc*`` frames (not in a job) are located at their time, grouped by stamp gaps.

Intrinsics, per run and camera: ``calibrate_camera.json`` ``intrinsics``, else
``camera_info_<camera_id>.json`` (ROS CameraInfo ``msg``), else ``--camera``.

Truth: the camera position ``cams[<camera>]`` at the frame's ``stamp``; the orientation from the
robot's link (``rotation_body``); the run's start frame (the priors' map frame) from a yaw-only
similarity fit of the prior positions to those truth positions, else the base pose at the first
stamp (the arm: ``bodies.base_link``).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rpl_episode import (
    Episode,
    Event,
    Frame,
    FrameSet,
    hello_of,
    orient,
    read_jsonl,
    read_track,
    read_verdict,
    start_pose,
)
from rpl_geometry import Arr, make_T, mat4, umeyama
from rpl_images import ImageError, image_size
from rpl_layout import LayoutError, need

NAN_POSE = np.full((4, 4), np.nan)


def _num(value: Any) -> float | None:
    return None if value is None else float(value)


def stamp_of(f: Frame) -> float:
    """A v2 frame's stamp (always set: ``read_frame`` refuses a sidecar without one)."""
    assert f.stamp is not None
    return f.stamp


def read_frame(run: str, sidecar: Path) -> Frame:
    """One frame from its sidecar; errors name the file and the key."""
    data = json.loads(sidecar.read_text())
    if not isinstance(data, dict):
        raise LayoutError(f"{sidecar}: expected a JSON object")
    name = str(data.get("name") or sidecar.stem)
    image = str(data.get("image") or f"{name}.png")
    path = sidecar.parent / image
    if not path.is_file():
        raise LayoutError(f"{sidecar}: key 'image' names '{image}', which is not in "
                          f"{sidecar.parent}")
    try:
        width, height = image_size(path.read_bytes())
    except ImageError as e:
        raise LayoutError(f"{path}: {e}") from e
    stamp = data.get("stamp")
    if stamp is None:
        raise LayoutError(f"{sidecar}: key 'stamp' is missing or null")
    prior = data.get("prior")
    pose = NAN_POSE.copy()
    if isinstance(prior, dict) and prior.get("T_map_cam") is not None:
        pose = mat4(prior["T_map_cam"])
    else:
        prior = None
    depth = data.get("depth")
    frame = Frame(run, "frames", image, path, None, float(stamp), pose, width, height,
                  prior=prior, height_sigma_m=_num(data.get("height_sigma_m")))
    frame.height_m = _num(data.get("height_above_floor_m"))
    frame.camera_id = str(data.get("camera_id") or "cam0")
    frame.intrinsics = {"width_px": float(data.get("width_px") or width),
                        "height_px": float(data.get("height_px") or height)}
    if depth:
        dpath = sidecar.parent / str(depth)
        frame.depth_path = dpath if dpath.is_file() else None
    return frame


def camera_from_calibrate(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    k = need(json.loads(path.read_text()), "intrinsics", str(path))
    where = f"{path} intrinsics"
    return {"fx": float(need(k, "fx_px", where)), "fy": float(need(k, "fy_px", where)),
            "cx": float(need(k, "cx_px", where)), "cy": float(need(k, "cy_px", where)),
            "width": int(need(k, "width_px", where)), "height": int(need(k, "height_px", where))}


def camera_from_info(path: Path) -> dict[str, Any] | None:
    """A camera from a ROS CameraInfo (``msg``: K row-major, width, height, D,
    distortion_model); plumb_bob / rational_polynomial with non-zero D give OpenCV distortion."""
    if not path.is_file():
        return None
    msg = need(json.loads(path.read_text()), "msg", str(path))
    where = f"{path} msg"
    K = [float(v) for v in need(msg, "K", where)]
    if len(K) != 9:
        raise LayoutError(f"{where}: key 'K' must hold 9 numbers, has {len(K)}")
    cam: dict[str, Any] = {"fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5],
                           "width": int(need(msg, "width", where)),
                           "height": int(need(msg, "height", where))}
    D = [float(v) for v in msg.get("D") or []]
    if msg.get("distortion_model") in ("plumb_bob", "rational_polynomial") and any(D):
        cam["distortion"] = {"model": "opencv", "coefficients": (D + [0.0] * 4)[:max(4, min(8,
                                                                                    len(D)))]}
    return cam


def run_camera(run_dir: Path, camera_id: str, size: tuple[int, int], layout: dict[str, Any],
               sources: dict[str, str]) -> dict[str, Any]:
    """The intrinsics of one camera of a run, in the order calibrate_camera.json (when its size
    matches the frames), camera_info_<id>.json, --camera; an error lists what was tried."""
    v2 = layout["v2"]
    cal = run_dir / v2["calibrate"]
    info = run_dir / v2["camera_info"].format(camera_id=camera_id)
    cam = camera_from_calibrate(cal)
    source = str(cal)
    if cam is None or (cam["width"], cam["height"]) != size:
        cam, source = camera_from_info(info), str(info)
    if cam is None and layout.get("camera_override"):
        cam, source = dict(layout["camera_override"]), "--camera"
    if cam is None:
        raise LayoutError(f"run {run_dir.name}, camera '{camera_id}' ({size[0]}x{size[1]}): no "
                          f"intrinsics: {cal} is missing or another size, {info} is missing, "
                          f"and no --camera was given")
    sources[camera_id] = source
    return cam


def parse_time(value: Any) -> float | None:
    """Epoch seconds from a number or an ISO-8601 string; None otherwise."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def read_jobs(run_dir: Path, layout: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    for rel in layout["v2"]["jobs"]:
        path = (run_dir / rel).resolve()
        if path.is_file():
            data = json.loads(path.read_text())
            jobs = data.get("jobs") if isinstance(data, dict) else data
            if not isinstance(jobs, list):
                raise LayoutError(f"{path}: key 'jobs' must be a list")
            return jobs, str(path)
    return [], None


def plan_v2(run: str, frames: list[Frame], jobs: list[dict[str, Any]], jobs_path: str | None,
            layout: dict[str, Any], warnings: list[str]) -> list[tuple[float, int, Event]]:
    """(time, order, event) for the run's jobs and its locate groups."""
    by_name: dict[str, Frame] = {}
    for f in frames:
        for key in (Path(f.file).stem, f.file):
            by_name[key] = f
    key = f"{run}/frames"
    out: list[tuple[float, int, Event]] = []
    in_jobs: set[str] = set()
    last = -math.inf
    for i, job in enumerate(jobs):
        where = f"{jobs_path} jobs[{i}]"
        names = need(job, "frames", where)
        found = [by_name[n] for n in names if n in by_name]
        for n in names:
            if n not in by_name:
                warnings.append(f"{where}: frame '{n}' has no sidecar in {run}/frames; skipped")
        if not found:
            continue
        in_jobs.update(f.file for f in found)
        t = parse_time(job.get("submitted_at"))
        last = t if t is not None else last
        measured = job.get("measured")
        out.append((last, i, Event(
            "update", key, [f.file for f in found], accept_if=job.get("accept_if"),
            job_id=None if job.get("job_id") is None else str(job["job_id"]),
            measured_s=_num(measured.get("s")) if isinstance(measured, dict) else None)))
    prefixes = tuple(layout["v2"]["locate_prefixes"])
    loc = [f for f in frames if f.file not in in_jobs and Path(f.file).stem.startswith(prefixes)]
    groups: list[list[Frame]] = []
    for f in loc:
        if groups and stamp_of(f) - stamp_of(groups[-1][-1]) <= layout["v2"]["locate_gap_s"]:
            groups[-1].append(f)
        else:
            groups.append([f])
    for g in groups:
        out.append((stamp_of(g[0]), len(jobs) + len(out),
                    Event("locate", key, [f.file for f in g])))
    return out


def start_from_priors(ep: Episode, frames: list[Frame],
                      layout: dict[str, Any]) -> tuple[Arr, float] | None:
    """The priors' map frame in the world (yaw and (x, y) at floor_z, z kept) and the scale,
    from a yaw-only similarity fit of the prior positions to the truth camera positions."""
    rows = [f for f in frames if f.has_pose and f.truth is not None
            and (f.prior or {}).get("trust") != "none"]
    if len(rows) < 2:
        return None
    src = np.array([f.pose[:3, 3] for f in rows])
    dst = np.array([f.truth[:3, 3] for f in rows])  # type: ignore[index]
    if np.linalg.norm(dst[:, :2].max(axis=0) - dst[:, :2].min(axis=0)) < \
            layout["odometry_fit_min_spread_m"]:
        return None
    try:
        sim, _ = umeyama(src, dst, min_ratio=math.inf)
    except ValueError:
        return None
    return make_T(sim.R, sim.t), sim.s


def load_v2(root: Path, layout: dict[str, Any]) -> Episode:
    warnings: list[str] = []
    keys = layout["truth"]
    lines = read_jsonl(root / layout["files"]["truth"])
    metas = [x for x in lines if x.get(keys["kind"]) == keys["kind_meta"]]
    if not metas:
        raise LayoutError(f"{root}/truth.jsonl: no line with {keys['kind']} = "
                          f"'{keys['kind_meta']}'")
    meta = metas[0]
    robot = str(need(meta, keys["meta_robot"], f"{root}/truth.jsonl meta line"))
    if robot not in layout["robots"]:
        raise LayoutError(f"robot '{robot}' is not in the layout's robots table "
                          f"({', '.join(layout['robots'])}); add it with --layout")
    spec = layout["robots"][robot]
    hello = hello_of(meta, keys)
    worktop, floor_z = hello.get(keys["worktop"]), hello.get(keys["floor_z"])
    sets: dict[str, FrameSet] = {}
    cameras: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    timeline: list[tuple[float, int, Event]] = []
    runs = sorted(p.parent for p in (root / layout["files"]["runs"]).glob(
        f"*/{layout['v2']['frames']}") if p.is_dir())
    for index, run_dir in enumerate(runs):
        run = run_dir.name
        sidecars = sorted(p for p in (run_dir / layout["v2"]["frames"]).glob("*.json")
                          if not p.name.startswith("_"))
        frames = sorted((read_frame(run, p) for p in sidecars), key=lambda f: (stamp_of(f), f.file))
        run_keys: dict[tuple[str, int, int], str] = {}  # (camera id, w, h) -> API camera id
        for f in frames:
            wanted = (f.camera_id, f.width, f.height)
            if wanted not in run_keys:
                cam = run_camera(run_dir, f.camera_id, (f.width, f.height), layout, sources)
                cam_key = f.camera_id
                if cam_key in cameras and cameras[cam_key] != cam:
                    cam_key = f"{f.camera_id}@{run}"
                    warnings.append(f"run {run}: camera '{f.camera_id}' differs from an earlier "
                                    f"run's; sent as '{cam_key}'")
                cameras.setdefault(cam_key, cam)
                run_keys[wanted] = cam_key
            f.camera_id = run_keys[wanted]
        jobs, jobs_path = read_jobs(run_dir, layout)
        if jobs_path is None:
            warnings.append(f"run {run}: no {' or '.join(layout['v2']['jobs'])}; nothing to "
                            f"update")
        sets[f"{run}/frames"] = FrameSet(run, "frames", "v2", frames, index)
        timeline += plan_v2(run, frames, jobs, jobs_path, layout, warnings)
    events = [e for _, _, e in sorted(timeline, key=lambda x: (x[0], x[1]))]
    ep = Episode(
        id=root.name, root=root, robot=robot, kind=str(spec["kind"]),
        camera_name=str(spec["camera"]), scene=str(hello.get(keys["scene"], "")),
        floor_z=None if floor_z is None else float(floor_z),
        worktop_z=float(worktop[keys["worktop_z"]])
        if isinstance(worktop, dict) and keys["worktop_z"] in worktop else None,
        meta=meta, sets=sets, events=events, cameras=cameras,
        track=read_track(lines, str(spec["camera"]), keys),
        objects_lines=[x for x in lines if x.get(keys["kind"]) == keys["kind_objects"]],
        start_poses={}, warnings=warnings, passed=read_verdict(root, layout),
        placement=str(meta.get(keys["meta_placement"]) or ""), version=2)
    ep.meta = {**meta, "camera_sources": sources}
    truth_v2(ep, layout)
    return ep


def truth_v2(ep: Episode, layout: dict[str, Any]) -> None:
    link = layout["rotation_body"].get(ep.kind)
    for run, fs in ((s.run, s.frames) for s in ep.sets.values()):
        for f in fs:
            f.truth_t = f.stamp
            f.truth = make_T(np.full((3, 3), np.nan), ep.track.cam_at(stamp_of(f)))
            f.truth_source = "stamp"
        t0 = min((stamp_of(f) for f in fs), default=None)
        body0 = layout["start_link"].get(ep.kind)
        first = ep.track.bodies[0 if t0 is None else ep.track.index(t0)]
        fitted = start_from_priors(ep, fs, layout)
        if body0 is not None and body0 in first:
            T_ws = make_T(np.eye(3), np.asarray(need(first[body0], layout["truth"]["pos"],
                                                     f"bodies.{body0}"), dtype=float))
        elif fitted is not None:
            T_ws, ep.odom_scale[run] = fitted
        else:
            T_ws = start_pose(ep.track, t0, ep.kind, layout, ep.floor_z)
        ep.start_poses[run] = T_ws
        for f in fs:
            if f.has_pose:
                assert f.truth is not None
                f.truth[:3, :3] = (T_ws @ f.pose)[:3, :3]
    if link:
        orient(ep, link, layout)
    bad = [f for f in ep.all_frames() if f.truth is not None and not np.isfinite(f.truth).all()]
    for f in bad:
        assert f.truth is not None
        f.truth[:3, :3] = np.eye(3)
    if bad:
        ep.warnings.append(f"{len(bad)} frames without a prior nor a truth link orientation: "
                           f"identity orientation")


def _prefix(stem: str) -> str:
    return "".join(c for c in stem.split("-")[0] if c.isalpha()) or "?"


def dump_v2(root: Path, layout: dict[str, Any]) -> list[dict[str, Any]]:
    """Keys and counts of each v2 run (never values that could carry paths)."""
    v2 = layout["v2"]
    out: list[dict[str, Any]] = []
    for frames_dir in sorted((root / layout["files"]["runs"]).glob(f"*/{v2['frames']}")):
        run_dir = frames_dir.parent
        jsons = sorted(frames_dir.glob("*.json"))
        sidecars = [p for p in jsons if not p.name.startswith("_")]
        keys: set[str] = set()
        prefixes: dict[str, int] = {}
        trust: dict[str, int] = {}
        cams: set[str] = set()
        depth = 0
        for p in sidecars:
            d = json.loads(p.read_text())
            keys |= set(d)
            prefixes[_prefix(p.stem)] = prefixes.get(_prefix(p.stem), 0) + 1
            t = str((d.get("prior") or {}).get("trust"))
            trust[t] = trust.get(t, 0) + 1
            cams.add(str(d.get("camera_id")))
            depth += bool(d.get("depth")) and (frames_dir / str(d["depth"])).is_file()
        jobs, jobs_path = read_jobs(run_dir, layout)
        entry: dict[str, Any] = {
            "run": run_dir.name, "sidecars": len(sidecars),
            "skipped": [p.name for p in jsons if p.name.startswith("_")],
            "by_prefix": prefixes, "sidecar_keys": sorted(keys), "prior_trust": trust,
            "with_depth": depth, "camera_ids": sorted(cams),
            "jobs": {"found": jobs_path is not None, "n": len(jobs),
                     "keys": sorted({k for j in jobs if isinstance(j, dict) for k in j})},
            "oms_files": sorted(p.name for p in (run_dir / "oms").glob("*"))
            if (run_dir / "oms").is_dir() else [],
            "intrinsics": {}}
        for cam in sorted(cams):
            frame = next(p for p in sidecars if str(json.loads(p.read_text()).get("camera_id"))
                         == cam)
            d = json.loads(frame.read_text())
            size = (int(d.get("width_px") or 0), int(d.get("height_px") or 0))
            sources: dict[str, str] = {}
            try:
                run_camera(run_dir, cam, size, layout, sources)
                entry["intrinsics"][cam] = Path(sources[cam]).name
            except LayoutError as e:
                entry["intrinsics"][cam] = f"error: {e}"
        log = run_dir / v2["log"]
        if log.is_file():
            lines = [x for x in log.read_text().splitlines() if x.strip()]
            parsed = []
            for x in lines:
                try:
                    parsed.append(json.loads(x))
                except json.JSONDecodeError:
                    continue
            entry["log"] = {"lines": len(lines),
                            "keys": sorted({k for x in parsed if isinstance(x, dict) for k in x})}
        out.append(entry)
    return out
