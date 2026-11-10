"""Read one robot-guide episode folder: frame sets, the agent's request order, cameras and the
simulator truth of every frame.

Truth pose of a frame (see README "Assumptions"):
* position: the truth camera optical centre (``robot`` lines, ``cams[<camera>]``) at the frame's
  time: the humanoid's ``stamp`` (interpolated), else the file-name time, refined to the point of
  the truth track nearest the views.json position composed with the run's start pose, within the
  time window between the neighbouring sets;
* orientation: the truth body named in ``rotation_body`` when given, else the views.json rotation
  composed with the run's start pose;
* start frame -> world: the truth base pose at the run's first frame; for floor robots it is the
  base's yaw and its (x, y) on the floor (``floor_z``), for the arm the full base pose.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rpl_floorplan import TruthObject
from rpl_geometry import Arr, make_T, mat4, quat_to_R, rot_z, yaw_of_R
from rpl_images import ImageError, exif_focal_px, image_size
from rpl_layout import LayoutError, need


@dataclass
class Frame:
    run: str
    set: str
    file: str
    path: Path
    nn: int
    name_time: float | None
    stamp: float | None
    pose: Arr  # views.json: optical pose in the run's start frame
    width: int
    height: int
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

    @property
    def key(self) -> str:
        return f"{self.run}/{self.name}"


@dataclass
class Event:
    op: str  # update | locate
    set_key: str
    files: list[str] | None  # a subset of the set's files, or None for all
    line: int


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

    def cam_at(self, t: float) -> Arr:
        return np.array([np.interp(t, self.t, self.cam[:, k]) for k in range(3)])

    def match(self, p_hat: Arr, lo: float, hi: float) -> tuple[Arr, float]:
        """The point of the camera track within [lo, hi] nearest ``p_hat``, and its time."""
        idx = np.nonzero((self.t >= lo) & (self.t <= hi))[0]
        if len(idx) == 0:
            mid = (lo + hi) / 2.0
            return self.cam_at(mid), mid
        if len(idx) == 1:
            return self.cam[idx[0]].copy(), float(self.t[idx[0]])
        A, B = self.cam[idx[:-1]], self.cam[idx[1:]]
        d = B - A
        den = np.maximum((d * d).sum(axis=1), 1e-12)
        s = np.clip(((p_hat - A) * d).sum(axis=1) / den, 0.0, 1.0)
        Q = A + s[:, None] * d
        k = int(np.argmin(np.linalg.norm(Q - p_hat, axis=1)))
        t0, t1 = self.t[idx[k]], self.t[idx[k + 1]]
        return Q[k], float(t0 + s[k] * (t1 - t0))


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


def parse_name_time(file: str, layout: dict[str, Any]) -> tuple[int, float | None]:
    """(NN, epoch seconds) from ``<set>-YYYYMMDD-HHMMSS-NN.jpg``; (0, None) if it does not match."""
    m = re.match(layout["frame_name"]["pattern"], file)
    if m is None:
        return 0, None
    tz = timezone(timedelta(hours=float(layout["frame_name"]["utc_offset_h"])))
    when = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S").replace(tzinfo=tz)
    return int(m["nn"]), when.timestamp()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise LayoutError(f"{path}:{n}: not JSON ({e})") from e
    return out


def view_entry(value: Any, keys: dict[str, str], where: str) -> tuple[Arr, float | None, str]:
    """(pose, stamp, variant) of one views.json value."""
    if isinstance(value, list):
        return mat4(value), None, "arm"
    pose = mat4(need(value, keys["pose"], where))
    stamp = value.get(keys["stamp"])
    variant = "humanoid" if stamp is not None else "base" if "base" in value else "dict"
    return pose, None if stamp is None else float(stamp), variant


def read_sets(root: Path, layout: dict[str, Any], warnings: list[str]) -> dict[str, FrameSet]:
    files = layout["files"]
    runs = root / files["runs"]
    if not runs.is_dir():
        raise LayoutError(f"{root}: no folder {files['runs']}")
    sets: dict[str, FrameSet] = {}
    for views_path in sorted(runs.glob(f"*/*/{files['views']}")):
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
            pose, stamp, variant = view_entry(value, layout["views"], f"{views_path}['{file}']")
            path = set_dir / file
            data = path.read_bytes()
            try:
                width, height = image_size(data)
            except ImageError as e:
                raise LayoutError(f"{path}: {e}") from e
            nn, name_time = parse_name_time(file, layout)
            frame = Frame(run, name, file, path, nn, name_time, stamp, pose, width, height)
            fxy = exif_focal_px(data, layout["exif_default_unit"])
            if fxy is not None:
                frame.intrinsics = {"exif_fx": fxy[0], "exif_fy": fxy[1]}
            frames.append(frame)
        for extra in sorted(on_disk - set(views)):
            warnings.append(f"{set_dir}: '{extra}' is not in views.json; skipped")
        frames.sort(key=lambda f: (f.nn, f.file))
        sets[f"{run}/{name}"] = FrameSet(run, name, variant, frames)
    if not sets:
        raise LayoutError(f"{runs}: no <run>/<set>/{files['views']}")
    return sets


def assign_cameras(sets: dict[str, FrameSet], spec: dict[str, Any],
                   warnings: list[str]) -> dict[str, dict[str, Any]]:
    """The API camera of each frame, from the robot's intrinsics table or the frame's EXIF."""
    cameras: dict[str, dict[str, Any]] = {}
    ids: dict[tuple[float, ...], str] = {}
    table = spec["intrinsics"]
    for fs in sets.values():
        for f in fs.frames:
            w, h = f.width, f.height
            if table == "exif":
                if "exif_fx" not in f.intrinsics:
                    raise LayoutError(f"{f.path}: no EXIF FocalLength/FocalPlaneXResolution, and "
                                      f"the robot's intrinsics are 'exif'")
                fx, fy = f.intrinsics["exif_fx"], f.intrinsics["exif_fy"]
                cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            else:
                sx, sy = w / float(table["width"]), h / float(table["height"])
                if (sx, sy) != (1.0, 1.0):
                    warnings.append(f"{f.path}: {w}x{h} differs from the intrinsics table "
                                    f"{table['width']}x{table['height']}; scaled")
                fx, fy = float(table["fx"]) * sx, float(table["fy"]) * sy
                cx = (float(table["cx"]) + 0.5) * sx - 0.5
                cy = (float(table["cy"]) + 0.5) * sy - 0.5
            key = (float(w), float(h), round(fx, 3), round(fy, 3), round(cx, 3), round(cy, 3))
            if key not in ids:
                ids[key] = f"cam{len(ids)}"
                cameras[ids[key]] = {"fx": key[2], "fy": key[3], "cx": key[4], "cy": key[5],
                                     "width": w, "height": h}
            f.camera_id = ids[key]
            f.intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return cameras


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in [str(k), *_strings(v)]]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def classify(text: str, words: dict[str, Any]) -> str:
    """update, locate or other: the earliest of the layout's words in the line."""
    low = text.lower()
    best = ("other", len(low) + 1)
    for op, key in (("locate", "locate_words"), ("update", "update_words"),
                    ("other", "skip_words")):
        for w in words[key]:
            i = low.find(w)
            if 0 <= i < best[1]:
                best = (op, i)
    return best[0]


def read_events(root: Path, sets: dict[str, FrameSet], layout: dict[str, Any],
                warnings: list[str]) -> list[Event]:
    """The agent's update and locate requests, in order, each naming one set."""
    path = root / layout["files"]["requests"]
    if not path.is_file():
        warnings.append(f"{path} is missing: every set is sent as an update, in time order")
        order = sorted(sets.values(), key=lambda s: min(
            (f.time for f in s.frames if f.time is not None), default=0.0))
        return [Event("update", s.key, None, 0) for s in order]
    by_file = {f.file: s.key for s in sets.values() for f in s.frames}
    by_name: dict[str, list[str]] = {}
    for s in sets.values():
        by_name.setdefault(s.name, []).append(s.key)
    events: list[Event] = []
    sent_updates: set[str] = set()
    for n, line in enumerate(read_jsonl(path), 1):
        text = json.dumps(line)
        op = classify(text, layout["requests"])
        if op == "other":
            continue
        refs: list[tuple[int, str, str | None]] = []  # (position, set key, file)
        for file, key in by_file.items():
            i = text.find(file)
            if i >= 0:
                refs.append((i, key, file))
        if not refs:
            for key in sets:
                i = text.find(key)
                if i >= 0:
                    refs.append((i, key, None))
        if not refs:
            for leaf in _strings(line):
                for part in re.split(r"[/\\]", leaf):
                    keys = by_name.get(part, [])
                    choice = [k for k in keys if k.split("/")[0] in text] or keys
                    if choice:
                        refs.append((text.find(leaf), choice[0], None))
        if not refs:
            warnings.append(f"{path}:{n}: {op} request names no known set; skipped")
            continue
        grouped: dict[str, list[str] | None] = {}
        for _, key, fname in sorted(refs, key=lambda r: r[0]):
            files = grouped.setdefault(key, [] if fname else None)
            if fname and files is not None:
                files.append(fname)
        for key, files in grouped.items():
            if files is not None and len(set(files)) == len(sets[key].frames):
                files = None
            if op == "update" and layout["requests"]["dedupe_updates"]:
                if key in sent_updates:
                    warnings.append(f"{path}:{n}: set {key} sent again as an update; skipped")
                    continue
                sent_updates.add(key)
            events.append(Event(op, key, files, n))
    return events


def read_track(lines: list[dict[str, Any]], camera: str, keys: dict[str, str]) -> Track:
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


def assign_truth(ep: Episode, layout: dict[str, Any]) -> None:
    """Truth pose and height above the floor of every frame (module docstring)."""
    timed = sorted({f.name_time for f in ep.all_frames() if f.name_time is not None})
    body = layout["rotation_body"].get(ep.kind)
    for run in sorted({s.run for s in ep.sets.values()}):
        frames = [f for s in ep.sets.values() if s.run == run for f in s.frames]
        times = [f.time for f in frames if f.time is not None]
        if not times:
            ep.warnings.append(f"run {run}: no frame time; its start pose is the first truth line")
        ep.start_poses[run] = start_pose(ep.track, min(times) if times else None, ep.kind,
                                         layout, ep.floor_z)
    for f in ep.all_frames():
        T_ws = ep.start_poses[f.run]
        T_wc = T_ws @ f.pose
        p_hat = T_wc[:3, 3]
        if f.stamp is not None:
            pos, t, src = ep.track.cam_at(f.stamp), f.stamp, "stamp"
        elif f.name_time is not None:
            k = timed.index(f.name_time)
            lo = timed[k - 1] if k > 0 else f.name_time - 300.0
            hi = timed[k + 1] if k + 1 < len(timed) else f.name_time + 300.0
            pos, t = ep.track.match(p_hat, lo - 2.0, hi + 2.0)
            src = "file-name time"
        else:
            pos, t, src = p_hat, None, "views"
        R = T_wc[:3, :3]
        if body and t is not None and body in ep.track.bodies[ep.track.index(t)]:
            b = ep.track.bodies[ep.track.index(t)][body]
            R = quat_to_R(need(b, layout["truth"]["quat"], f"bodies.{body}"),
                          layout["truth"]["quat_order"])
        f.truth = make_T(R, pos)
        f.truth_source = src
        if ep.kind in layout["floor_robots"]:
            f.height_m = float(f.pose[2, 3])


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
        meta=meta, sets=sets, events=read_events(root, sets, layout, warnings),
        cameras=cameras, track=read_track(lines, str(spec["camera"]), keys),
        objects_lines=[x for x in lines if x.get(keys["kind"]) == keys["kind_objects"]],
        start_poses={}, warnings=warnings)
    assign_truth(ep, layout)
    return ep


def dump_layout(root: Path, layout: dict[str, Any]) -> dict[str, Any]:
    """What an episode folder holds, key by key, without assuming the layout is right."""
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
            entry.update(frames=len(views), first_file=file,
                         name_parsed=parse_name_time(file, layout)[1] is not None,
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
    if req.is_file():
        reqs = read_jsonl(req)
        ops: dict[str, int] = {}
        for x in reqs:
            op = classify(json.dumps(x), layout["requests"])
            ops[op] = ops.get(op, 0) + 1
        out["requests"] = {"lines": len(reqs), "ops": ops,
                           "first": [json.dumps(x)[:400] for x in reqs[:3]]}
    else:
        out["requests"] = f"missing {layout['files']['requests']}"
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


def _views_vs_truth(ep: Episode) -> float | None:
    """Median distance between views.json positions (via the start pose) and truth positions: a
    check of the start-frame and time assumptions."""
    d = [float(np.linalg.norm((ep.start_poses[f.run] @ f.pose)[:3, 3] - f.truth[:3, 3]))
         for f in ep.all_frames() if f.truth is not None]
    return float(np.median(d)) if d else None
