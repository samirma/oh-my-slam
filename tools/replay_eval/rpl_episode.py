"""Read one robot-guide episode folder: frame sets, the replay plan, cameras and the simulator truth
of every frame.

Replay plan: every set with a views.json, in time order (humanoid ``stamp``, else file-name time,
else set folder order), is one update, except every ``holdout_every``-th frame, which is located
right after it. Sets named ``loc*`` / ``look*`` are locate-only. ``oms/agent_requests.jsonl`` is a
log of shell commands, not of API requests, and is not used.

Truth pose of a frame (README "Assumptions"):
* start frame -> world: the truth base pose at the run's first frame (floor robots: its yaw and
  (x, y) at ``floor_z``; the arm: the full base pose, refined by a rigid fit of the views positions
  to the matched truth camera positions, since its kinematics are exact);
* time: the humanoid's ``stamp``; the base's time is where the truth base track, relative to the
  start pose, best matches the views ``base`` odometry [x, y, yaw] within the set's time window;
  the arm's is where the truth camera track is nearest the views position within that window;
* position: the truth camera optical centre ``cams[<camera>]`` at that time;
* orientation: the truth link ``rotation_body[<kind>]`` turned into the optical frame, else the
  views.json rotation composed with the start pose.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rpl_floorplan import TruthObject
from rpl_geometry import (
    R_LINK_OPT,
    Arr,
    inv_T,
    make_T,
    mat4,
    quat_to_R,
    rigid_fit,
    rot_z,
    yaw_of_R,
)
from rpl_images import ImageError, exif_focal_px, image_size
from rpl_layout import LayoutError, need


@dataclass
class Frame:
    run: str
    set: str
    file: str
    path: Path
    name_time: float | None
    stamp: float | None
    pose: Arr  # views.json: optical pose in the run's start frame
    width: int
    height: int
    odom: tuple[float, float, float] | None = None  # views "base": [x, y, yaw rad]
    intrinsics: dict[str, float] = field(default_factory=dict)
    camera_id: str = ""
    truth: Arr | None = None  # T_world_cam
    truth_source: str = ""
    height_m: float | None = None

    @property
    def name(self) -> str:
        """The frame's name in API requests: unique within the episode."""
        return f"{self.run}/{self.set}/{Path(self.file).stem}"

    @property
    def time(self) -> float | None:
        return self.stamp if self.stamp is not None else self.name_time


@dataclass
class FrameSet:
    run: str
    name: str
    variant: str
    frames: list[Frame]
    folder_index: int = 0

    @property
    def key(self) -> str:
        return f"{self.run}/{self.name}"

    @property
    def time(self) -> float | None:
        times = [f.time for f in self.frames if f.time is not None]
        return min(times) if times else None


@dataclass
class Event:
    op: str  # update | locate
    set_key: str
    files: list[str] | None  # a subset of the set's files, or None for all


@dataclass
class Track:
    """The truth ``robot`` lines as arrays."""

    t: Arr
    base_pos: Arr
    base_R: Arr  # (n, 3, 3)
    cam: Arr  # (n, 3)
    bodies: list[dict[str, Any]]

    def index(self, t: float) -> int:
        i = int(np.searchsorted(self.t, t))
        if i == 0:
            return 0
        if i >= len(self.t):
            return len(self.t) - 1
        return i if self.t[i] - t < t - self.t[i - 1] else i - 1

    def window(self, lo: float, hi: float) -> Any:
        return np.nonzero((self.t >= lo) & (self.t <= hi))[0]

    def cam_at(self, t: float) -> Arr:
        return np.array([np.interp(t, self.t, self.cam[:, k]) for k in range(3)])

    def match(self, p_hat: Arr, lo: float, hi: float) -> tuple[Arr, float]:
        """The truth camera sample within [lo, hi] nearest ``p_hat``, and its time (frames are
        taken while still, so a sample sits where the frame was taken)."""
        idx = self.window(lo, hi)
        if len(idx) == 0:
            mid = (lo + hi) / 2.0
            return self.cam_at(mid), mid
        k = int(idx[int(np.argmin(np.linalg.norm(self.cam[idx] - p_hat, axis=1)))])
        return self.cam[k].copy(), float(self.t[k])

    def match_odometry(self, T_ws: Arr, odom: tuple[float, float, float], lo: float, hi: float,
                       yaw_weight_m: float) -> float:
        """The time in [lo, hi] whose base pose, in the start frame, best matches ``odom``."""
        idx = self.window(lo, hi)
        if len(idx) == 0:
            return (lo + hi) / 2.0
        R0, t0 = T_ws[:3, :3], T_ws[:3, 3]
        rel = (self.base_pos[idx] - t0) @ R0  # R0^T (p - t0), row-wise
        yaw0 = yaw_of_R(R0)
        best, best_t = math.inf, float(self.t[idx[0]])
        for k, i in enumerate(idx):
            dyaw = (yaw_of_R(self.base_R[i]) - yaw0 - odom[2] + math.pi) % (2 * math.pi) - math.pi
            cost = math.hypot(math.hypot(rel[k, 0] - odom[0], rel[k, 1] - odom[1]),
                              yaw_weight_m * dyaw)
            if cost < best:
                best, best_t = cost, float(self.t[i])
        return best_t


@dataclass
class Episode:
    id: str
    root: Path
    robot: str
    kind: str
    camera_name: str
    scene: str
    floor_z: float | None
    worktop_z: float | None
    meta: dict[str, Any]
    sets: dict[str, FrameSet]
    events: list[Event]
    cameras: dict[str, dict[str, Any]]
    track: Track
    objects_lines: list[dict[str, Any]]
    start_poses: dict[str, Arr]
    warnings: list[str]
    passed: bool | None = None  # from meta.json or verdict.md; None when neither says

    def frames(self, event: Event) -> list[Frame]:
        fs = self.sets[event.set_key].frames
        if event.files is None:
            return list(fs)
        keep = set(event.files)
        return [f for f in fs if f.file in keep]

    def all_frames(self) -> list[Frame]:
        return [f for s in self.sets.values() for f in s.frames]

    def truth_objects(self, layout: dict[str, Any]) -> list[TruthObject]:
        """Objects of the last ``objects`` line (arm scenes, which have no floor plan)."""
        if not self.objects_lines:
            return []
        keys = layout["truth"]
        bodies = need(self.objects_lines[-1], keys["bodies"], "truth.jsonl objects line")
        out: list[TruthObject] = []
        for name, body in bodies.items():
            pos = np.asarray(need(body, keys["pos"], f"objects bodies.{name}"), dtype=float)
            support = None
            if self.worktop_z is not None and 0.0 <= pos[2] - self.worktop_z <= 0.3:
                support = self.worktop_z
            label = re.sub(r"[\d_\-.]+$", "", name) or name
            out.append(TruthObject(name, label, pos, support_z=support))
        return out


def parse_name_time(file: str, layout: dict[str, Any]) -> float | None:
    """Epoch seconds of the first ``YYYYMMDD-HHMMSS`` (local time) in a file name, or None."""
    m = re.search(layout["frame_time"]["pattern"], file)
    if m is None:
        return None
    tz = timezone(timedelta(hours=float(layout["frame_time"]["utc_offset_h"])))
    try:
        when = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return when.replace(tzinfo=tz).timestamp()


def read_jsonl(path: Path) -> list[Any]:
    out: list[Any] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise LayoutError(f"{path}:{n}: not JSON ({e})") from e
    return out


def view_entry(value: Any, keys: dict[str, str],
               where: str) -> tuple[Arr, float | None, tuple[float, float, float] | None, str]:
    """(pose, stamp, odometry, variant) of one views.json value."""
    if isinstance(value, list):
        return mat4(value), None, None, "arm"
    pose = mat4(need(value, keys["pose"], where))
    stamp = value.get(keys["stamp"])
    base = value.get(keys["base"])
    odom = None if base is None else (float(base[0]), float(base[1]), float(base[2]))
    variant = "humanoid" if stamp is not None else "base" if odom is not None else "dict"
    return pose, None if stamp is None else float(stamp), odom, variant


def read_sets(root: Path, layout: dict[str, Any], warnings: list[str]) -> dict[str, FrameSet]:
    files = layout["files"]
    runs = root / files["runs"]
    if not runs.is_dir():
        raise LayoutError(f"{root}: no folder {files['runs']}")
    sets: dict[str, FrameSet] = {}
    for index, views_path in enumerate(sorted(runs.glob(f"*/*/{files['views']}"))):
        set_dir = views_path.parent
        run, name = set_dir.parent.name, set_dir.name
        views = json.loads(views_path.read_text())
        if not isinstance(views, dict):
            raise LayoutError(f"{views_path}: expected an object keyed by image file name")
        on_disk = {p.name for p in set_dir.iterdir()
                   if p.suffix.lower() in files["frame_suffixes"]}
        frames: list[Frame] = []
        variant = "empty"
        for file, value in views.items():
            if file not in on_disk:
                warnings.append(f"{views_path}: '{file}' has no image file; skipped")
                continue
            pose, stamp, odom, variant = view_entry(value, layout["views"],
                                                    f"{views_path}['{file}']")
            path = set_dir / file
            data = path.read_bytes()
            try:
                width, height = image_size(data)
            except ImageError as e:
                raise LayoutError(f"{path}: {e}") from e
            frame = Frame(run, name, file, path, parse_name_time(file, layout), stamp, pose,
                          width, height, odom)
            fxy = exif_focal_px(data, layout["exif_default_unit"])
            if fxy is not None:
                frame.intrinsics = {"exif_fx": fxy[0], "exif_fy": fxy[1]}
            frames.append(frame)
        for extra in sorted(on_disk - set(views)):
            warnings.append(f"{set_dir}: '{extra}' is not in views.json; skipped")
        frames.sort(key=lambda f: (f.stamp if f.stamp is not None else math.inf, f.file))
        sets[f"{run}/{name}"] = FrameSet(run, name, variant, frames, index)
    if not sets:
        raise LayoutError(f"{runs}: no <run>/<set>/{files['views']}")
    return sets


def assign_cameras(sets: dict[str, FrameSet], spec: dict[str, Any],
                   warnings: list[str]) -> dict[str, dict[str, Any]]:
    """The API camera of each frame: the robot's intrinsics table at its image size, else the
    frame's EXIF (fx from FocalLength x FocalPlaneXResolution, c at the image centre), else the
    table scaled to the frame's size."""
    cameras: dict[str, dict[str, Any]] = {}
    ids: dict[tuple[float, ...], str] = {}
    table = spec.get("intrinsics")
    table = table if isinstance(table, dict) else None
    for fs in sets.values():
        for f in fs.frames:
            w, h = f.width, f.height
            exif = f.intrinsics
            if table is not None and (w, h) == (table["width"], table["height"]):
                fx, fy = float(table["fx"]), float(table["fy"])
                cx, cy = float(table["cx"]), float(table["cy"])
            elif "exif_fx" in exif:
                fx, fy = exif["exif_fx"], exif["exif_fy"]
                cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            elif table is not None:
                sx, sy = w / float(table["width"]), h / float(table["height"])
                warnings.append(f"{f.path}: {w}x{h} differs from the intrinsics table "
                                f"{table['width']}x{table['height']}; scaled")
                fx, fy = float(table["fx"]) * sx, float(table["fy"]) * sy
                cx = (float(table["cx"]) + 0.5) * sx - 0.5
                cy = (float(table["cy"]) + 0.5) * sy - 0.5
            else:
                raise LayoutError(f"{f.path}: no intrinsics in the robots table and no EXIF "
                                  f"FocalLength/FocalPlaneXResolution")
            key = (float(w), float(h), round(fx, 3), round(fy, 3), round(cx, 3), round(cy, 3))
            if key not in ids:
                ids[key] = f"cam{len(ids)}"
                cameras[ids[key]] = {"fx": key[2], "fy": key[3], "cx": key[4], "cy": key[5],
                                     "width": w, "height": h}
            f.camera_id = ids[key]
            f.intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return cameras


def ordered_sets(sets: dict[str, FrameSet]) -> list[FrameSet]:
    """Sets in time order; a set without a time keeps its folder-order place after the set
    before it."""
    folder = sorted(sets.values(), key=lambda s: s.folder_index)
    keyed: list[tuple[float, int, FrameSet]] = []
    last = -math.inf
    for s in folder:
        t = s.time
        last = t if t is not None else last
        keyed.append((last, s.folder_index, s))
    return [s for _, _, s in sorted(keyed, key=lambda k: (k[0], k[1]))]


def plan_events(sets: dict[str, FrameSet], layout: dict[str, Any]) -> list[Event]:
    """One update per set in time order, with every n-th frame held out and located after it;
    locate-only sets are located whole."""
    rp = layout["replay"]
    every = int(rp["holdout_every"])
    events: list[Event] = []
    for s in ordered_sets(sets):
        if not s.frames:
            continue
        if any(s.name.startswith(p) for p in rp["locate_prefixes"]):
            events.append(Event("locate", s.key, None))
            continue
        held = [f.file for i, f in enumerate(s.frames) if every > 0 and (i + 1) % every == 0]
        kept = [f.file for f in s.frames if f.file not in held]
        events.append(Event("update", s.key, kept if held else None))
        if held:
            events.append(Event("locate", s.key, held))
    return events


def read_track(lines: list[Any], camera: str, keys: dict[str, str]) -> Track:
    robot = [x for x in lines if x.get(keys["kind"]) == keys["kind_robot"]]
    if not robot:
        raise LayoutError(f"truth.jsonl: no line with {keys['kind']} = '{keys['kind_robot']}'")
    robot.sort(key=lambda x: float(need(x, keys["t"], "truth.jsonl robot line")))
    ts, pos, rot, cam, bodies = [], [], [], [], []
    for x in robot:
        where = f"truth.jsonl robot line t={x.get(keys['t'])}"
        ts.append(float(x[keys["t"]]))
        pos.append([float(v) for v in need(x, keys["pos"], where)])
        rot.append(quat_to_R(need(x, keys["quat"], where), keys["quat_order"]))
        cam.append([float(v) for v in need(need(x, keys["cams"], where), camera,
                                           f"{where} {keys['cams']}")])
        bodies.append(x.get(keys["bodies"]) or {})
    return Track(np.array(ts), np.array(pos), np.array(rot), np.array(cam), bodies)


def start_pose(track: Track, t: float | None, kind: str, layout: dict[str, Any],
               floor_z: float | None) -> Arr:
    i = 0 if t is None else track.index(t)
    p, R = track.base_pos[i], track.base_R[i]
    if kind in layout["floor_robots"]:
        z = 0.0 if floor_z is None else floor_z
        return make_T(rot_z(yaw_of_R(R)), np.array([p[0], p[1], z]))
    return make_T(R, p)


def set_windows(sets: dict[str, FrameSet]) -> dict[str, tuple[float, float]]:
    """The time window of each timed set: from the set before it to the set after it (2 s
    slack), or 300 s on an open side."""
    timed = sorted((t, s.key) for s in sets.values() if (t := s.time) is not None)
    out: dict[str, tuple[float, float]] = {}
    for k, (t, key) in enumerate(timed):
        lo = timed[k - 1][0] - 2.0 if k > 0 else t - 300.0
        hi = timed[k + 1][0] + 2.0 if k + 1 < len(timed) else t + 300.0
        out[key] = (lo, hi)
    return out


def _match_run(ep: Episode, frames: list[Frame], T_ws: Arr, windows: dict[str, tuple[float, float]],
               layout: dict[str, Any]) -> list[tuple[Arr, float | None, str]]:
    out: list[tuple[Arr, float | None, str]] = []
    for f in frames:
        p_hat = (T_ws @ f.pose)[:3, 3]
        win = windows.get(f"{f.run}/{f.set}")
        if f.stamp is not None:
            out.append((ep.track.cam_at(f.stamp), f.stamp, "stamp"))
        elif win is not None and f.odom is not None:
            t = ep.track.match_odometry(T_ws, f.odom, *win, layout["odometry_yaw_weight_m"])
            out.append((ep.track.cam_at(t), t, "odometry"))
        elif win is not None:
            pos, t = ep.track.match(p_hat, *win)
            out.append((pos, t, "views position"))
        else:
            out.append((p_hat, None, "views"))
    return out


def _best_shift(ep: Episode, frames: list[Frame], T_ws: Arr,
                windows: dict[str, tuple[float, float]]) -> Arr:
    """``T_ws`` translated by the shift that best lays the views positions on the truth camera
    samples of their sets' windows: each (frame, sample) pair proposes a shift, scored by the
    median distance of every frame to its nearest sample."""
    timed = [f for f in frames if f"{f.run}/{f.set}" in windows]

    def spread(items: list[Frame], n: int) -> list[Frame]:  # at most n, evenly spaced
        return [items[int(i)] for i in np.unique(np.linspace(0, len(items) - 1, n).round())]

    pairs = [((T_ws @ f.pose)[:3, 3], ep.track.cam[ep.track.window(*windows[f"{f.run}/{f.set}"])])
             for f in spread(timed, 30)]
    pairs = [(x, s) for x, s in pairs if len(s)]
    best, best_shift = math.inf, np.zeros(3)
    for f in spread(timed, 5):
        p = (T_ws @ f.pose)[:3, 3]
        samples = ep.track.cam[ep.track.window(*windows[f"{f.run}/{f.set}"])]
        for q in samples:
            shift = q - p
            score = float(np.median([np.min(np.linalg.norm(s - (x + shift), axis=1))
                                     for x, s in pairs]))
            if score < best:
                best, best_shift = score, shift
    return make_T(T_ws[:3, :3], T_ws[:3, 3] + best_shift)


def assign_truth(ep: Episode, layout: dict[str, Any]) -> None:
    """Truth pose and height above the floor of every frame (module docstring)."""
    windows = set_windows(ep.sets)
    link = layout["rotation_body"].get(ep.kind)
    for run in sorted({s.run for s in ep.sets.values()}):
        frames = [f for s in ep.sets.values() if s.run == run for f in s.frames]
        times = [f.time for f in frames if f.time is not None]
        if not times:
            ep.warnings.append(f"run {run}: no frame time; its start pose is the first truth line")
        T_ws = start_pose(ep.track, min(times) if times else None, ep.kind, layout, ep.floor_z)
        matched = _match_run(ep, frames, T_ws, windows, layout)
        if ep.kind in layout["refine_start"] and any(m[2] == "views position" for m in matched):
            T_ws = _best_shift(ep, frames, T_ws, windows)
            matched = _match_run(ep, frames, T_ws, windows, layout)
            for _ in range(int(layout["refine_iterations"])):  # ICP until the pose settles
                pairs = [(f.pose[:3, 3], m[0]) for f, m in zip(frames, matched, strict=True)
                         if m[2] == "views position"]
                T_new = rigid_fit(np.array([p for p, _ in pairs]),
                                  np.array([q for _, q in pairs]), T_ws[:3, :3])
                settled = bool(np.abs(T_new - T_ws).max() < 1e-10)
                T_ws = T_new
                matched = _match_run(ep, frames, T_ws, windows, layout)
                if settled:
                    break
        ep.start_poses[run] = T_ws
        for f, (pos, t, src) in zip(frames, matched, strict=True):
            R = (T_ws @ f.pose)[:3, :3]
            if link and t is not None and link in ep.track.bodies[ep.track.index(t)]:
                b = ep.track.bodies[ep.track.index(t)][link]
                R = quat_to_R(need(b, layout["truth"]["quat"], f"bodies.{link}"),
                              layout["truth"]["quat_order"]) @ R_LINK_OPT
            f.truth = make_T(R, pos)
            f.truth_source = src
            if ep.kind in layout["floor_robots"]:
                f.height_m = float(f.pose[2, 3])


def words_verdict(text: str, v: dict[str, Any]) -> bool | None:
    low = text.lower()
    hits = [(low.find(w), ok) for ok, key in ((True, "pass_words"), (False, "fail_words"))
            for w in v[key] if w in low]
    return min(hits)[1] if hits else None


def read_verdict(root: Path, layout: dict[str, Any]) -> bool | None:
    """meta.json's verdict field when present, else the earliest pass or fail word of
    verdict.md; None when neither says."""
    v = layout["verdict"]
    meta = root / layout["files"]["meta"]
    if meta.is_file():
        data = json.loads(meta.read_text())
        if isinstance(data, dict) and v["meta_key"] in data:
            value = data[v["meta_key"]]
            return value if isinstance(value, bool) else words_verdict(str(value), v)
    path = root / v["file"]
    return words_verdict(path.read_text(), v) if path.is_file() else None


def load_episode(root: Path, layout: dict[str, Any]) -> Episode:
    warnings: list[str] = []
    keys = layout["truth"]
    lines = read_jsonl(root / layout["files"]["truth"])
    metas = [x for x in lines if x.get(keys["kind"]) == keys["kind_meta"]]
    if not metas:
        raise LayoutError(f"{root}/truth.jsonl: no line with {keys['kind']} = "
                          f"'{keys['kind_meta']}'")
    meta = metas[0]
    robot = str(need(meta, keys["meta_robot"], "truth.jsonl meta line"))
    hello = meta.get(keys["meta_hello"]) or meta
    robots = layout["robots"]
    if robot not in robots:
        raise LayoutError(f"robot '{robot}' is not in the layout's robots table "
                          f"({', '.join(robots)}); add it with --layout")
    spec = robots[robot]
    worktop = hello.get(keys["worktop"])
    floor_z = hello.get(keys["floor_z"])
    sets = read_sets(root, layout, warnings)
    cameras = assign_cameras(sets, spec, warnings)
    ep = Episode(
        id=root.name, root=root, robot=robot, kind=str(spec["kind"]),
        camera_name=str(spec["camera"]), scene=str(hello.get(keys["scene"], "")),
        floor_z=None if floor_z is None else float(floor_z),
        worktop_z=float(worktop[keys["worktop_z"]])
        if isinstance(worktop, dict) and keys["worktop_z"] in worktop else None,
        meta=meta, sets=sets, events=plan_events(sets, layout),
        cameras=cameras, track=read_track(lines, str(spec["camera"]), keys),
        objects_lines=[x for x in lines if x.get(keys["kind"]) == keys["kind_objects"]],
        start_poses={}, warnings=warnings, passed=read_verdict(root, layout))
    assign_truth(ep, layout)
    return ep


def _log_summary(path: Path) -> dict[str, Any]:
    """Line count and keys of a JSONL log, never its values."""
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    parsed = []
    for x in lines:
        try:
            parsed.append(json.loads(x))
        except json.JSONDecodeError:
            continue
    return {"lines": len(lines), "json_lines": len(parsed), "keys": _keys(parsed), "used": False}


def _keys(lines: list[Any]) -> list[str]:
    return sorted({k for x in lines if isinstance(x, dict) for k in x})


def dump_layout(root: Path, layout: dict[str, Any]) -> dict[str, Any]:
    """What an episode folder holds, key by key (keys and counts only, never values that could
    carry local paths), without assuming the layout is right."""
    out: dict[str, Any] = {"episode": root.name,
                           "top": sorted(p.name for p in root.iterdir())}
    keys = layout["truth"]
    truth = root / layout["files"]["truth"]
    if truth.is_file():
        lines = read_jsonl(truth)
        kinds: dict[str, int] = {}
        first: dict[str, Any] = {}
        for x in lines:
            k = str(x.get(keys["kind"]))
            kinds[k] = kinds.get(k, 0) + 1
            first.setdefault(k, x)
        out["truth"] = {"kinds": kinds}
        for k, x in first.items():
            info: dict[str, Any] = {"keys": sorted(x)}
            for sub in (keys["meta_hello"], keys["cams"], keys["bodies"]):
                if isinstance(x.get(sub), dict):
                    info[sub] = sorted(x[sub])[:40]
            out["truth"][k] = info
    else:
        out["truth"] = f"missing {layout['files']['truth']}"
    sets: list[dict[str, Any]] = []
    runs = root / layout["files"]["runs"]
    for views_path in sorted(runs.glob(f"*/*/{layout['files']['views']}")):
        views = json.loads(views_path.read_text())
        entry: dict[str, Any] = {"set": f"{views_path.parent.parent.name}/"
                                        f"{views_path.parent.name}",
                                 "views_type": type(views).__name__}
        if isinstance(views, dict) and views:
            file, value = next(iter(views.items()))
            entry.update(frames=len(views),
                         name_time_parsed=sum(parse_name_time(f, layout) is not None
                                              for f in views),
                         value=sorted(value) if isinstance(value, dict)
                         else f"{type(value).__name__} of {len(value)}")
            img = views_path.parent / file
            if img.is_file():
                data = img.read_bytes()
                try:
                    entry["size"] = image_size(data)
                    entry["exif_fx_fy"] = exif_focal_px(data, layout["exif_default_unit"])
                except ImageError as e:
                    entry["image_error"] = str(e)
        sets.append(entry)
    out["sets"] = sets
    req = root / layout["files"]["requests"]
    out["requests_log"] = _log_summary(req) if req.is_file() else None
    meta = root / layout["files"]["meta"]
    out["meta_json_keys"] = _keys([json.loads(meta.read_text())]) if meta.is_file() else None
    out["verdict_passed"] = read_verdict(root, layout)
    rc = root / "robot_control"
    out["robot_control"] = sorted(str(p.relative_to(rc)) for p in rc.rglob("*")) \
        if rc.is_dir() else []
    try:
        ep = load_episode(root, layout)
    except (LayoutError, ValueError, KeyError, OSError) as e:
        out["load"] = {"error": str(e)}
    else:
        out["load"] = {
            "robot": ep.robot, "kind": ep.kind, "scene": ep.scene, "floor_z": ep.floor_z,
            "worktop_z": ep.worktop_z, "cameras": ep.cameras,
            "events": [f"{e.op} {e.set_key}" + ("" if e.files is None else f" ({len(e.files)})")
                       for e in ep.events],
            "frames": len(ep.all_frames()),
            "truth_sources": sorted({f.truth_source for f in ep.all_frames()}),
            "views_vs_truth_m": _views_vs_truth(ep), "warnings": ep.warnings[:50]}
    return out


def _views_vs_truth(ep: Episode) -> dict[str, float | None]:
    """Median position and rotation difference between the views poses (via the start pose)
    and the truth poses: a check of the start-frame, time and orientation assumptions."""
    pos, rot = [], []
    for f in ep.all_frames():
        if f.truth is None:
            continue
        V = ep.start_poses[f.run] @ f.pose
        pos.append(float(np.linalg.norm(V[:3, 3] - f.truth[:3, 3])))
        D = inv_T(V)[:3, :3] @ f.truth[:3, :3]
        rot.append(math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(D)) - 1) / 2)))))
    if not pos:
        return {"pos_m": None, "rot_deg": None}
    return {"pos_m": float(np.median(pos)), "rot_deg": float(np.median(rot))}
