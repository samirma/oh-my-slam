"""A tiny synthetic robot-guide episode: two box rooms joined by a door, a robot that drives,
turns and locates, and image files that carry only headers (and EXIF) — nothing is rendered."""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

LOCAL = UTC  # the agents ran in Docker on UTC: file-name times are UTC
ODOM_SCALE = 1.08  # the base's odometry over-counts distances
MODEL_YAW = math.pi / 2  # the base body's own frame is turned against its heading
ARM_LINE_OFFSET = (0.2, -0.1)  # the arm's robot-line pos is another body than its base link
ARM_CAM_ERROR = np.array([0.003, -0.002, 0.001])  # simulator camera vs the arm's kinematics

T0 = datetime(2026, 11, 9, 10, 0, 0, tzinfo=LOCAL).timestamp()
# optical axes in a forward-looking body frame (x forward, y left, z up)
R_BODY_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
CAM = {"rosmaster_x3_plus": "camera_color_optical_frame", "ainex": "camera",
       "rebot_b601_rs": "camera_color_optical_frame"}
HEIGHT = {"rosmaster_x3_plus": 0.5, "ainex": 0.34}
SIZE = {"rosmaster_x3_plus": (640, 480), "ainex": (640, 480), "rebot_b601_rs": (640, 360)}
FX_EXIF = 655.8
DOOR_Y = (1.0, 1.9)  # door gap in the wall x = 4 (world)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# the wrist camera's fixed mount: optical = gripper_end link rotated by this
ARM_MOUNT = _rot_x(math.radians(15)) @ rot_z(math.radians(-100))


def make_T(R: np.ndarray, t: Any) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def quat_wxyz(R: np.ndarray) -> list[float]:
    """Quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    tr = float(np.trace(R))
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    return [float(x) for x in q]


# ---- image files ---------------------------------------------------------------------------
def tiff_exif(entries: list[tuple[int, int, Any]], order: str = "<",
              exif_ifd: bool = True) -> bytes:
    """A TIFF block whose IFD0 points to an Exif IFD holding ``entries`` (tag, type, value);
    rationals are (num, den). With ``exif_ifd`` False the entries sit in IFD0."""
    bo = b"II" if order == "<" else b"MM"
    head = bo + struct.pack(order + "HI", 42, 8)

    def ifd(items: list[tuple[int, int, Any]], base: int) -> bytes:
        body = struct.pack(order + "H", len(items))
        extra = b""
        extra_at = base + 2 + 12 * len(items) + 4
        for tag, kind, value in items:
            if kind in (5, 10):
                body += struct.pack(order + "HHII", tag, kind, 1, extra_at + len(extra))
                extra += struct.pack(order + ("II" if kind == 5 else "ii"), *value)
            elif kind == 3:
                body += struct.pack(order + "HHIHH", tag, kind, 1, value, 0)
            elif kind in (4, 9):
                body += struct.pack(order + "HHI" + ("I" if kind == 4 else "i"), tag, kind, 1,
                                    value)
            else:  # an ASCII or zero-count entry the reader skips
                body += struct.pack(order + "HHI4s", tag, kind, value, b"ab\x00\x00")
        return body + struct.pack(order + "I", 0) + extra

    if not exif_ifd:
        return head + ifd(entries, 8)
    ifd0_len = 2 + 12 + 4
    ifd0 = struct.pack(order + "H", 1) + struct.pack(order + "HHII", 0x8769, 4, 1,
                                                     8 + ifd0_len) + struct.pack(order + "I", 0)
    return head + ifd0 + ifd(entries, 8 + ifd0_len)


def focal_entries(fx: float, unit: int | None = 4) -> list[tuple[int, int, Any]]:
    e: list[tuple[int, int, Any]] = [(0x920A, 5, (1, 1)), (0xA20E, 5, (int(fx * 1000), 1000)),
                                     (0xA20F, 5, (int(fx * 1000), 1000))]
    if unit is not None:
        e.append((0xA210, 3, unit))
    return e


def jpeg(w: int, h: int, exif: bytes | None = None) -> bytes:
    out = b"\xff\xd8"
    if exif is not None:
        app1 = b"Exif\x00\x00" + exif
        out += b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
    out += b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = struct.pack(">BHHB", 8, h, w, 3) + b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    out += b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof
    out += b"\xff\xda" + struct.pack(">H", 2) + b"\x00\x3f\x00\xff\xd9"
    return out


def png(w: int, h: int, exif: bytes | None = None) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(
            ">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    if exif is not None:
        out += chunk(b"eXIf", exif)
    raw = b"".join(b"\x00" + b"\x00" * (3 * w) for _ in range(h))
    return out + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


# ---- the house ------------------------------------------------------------------------------
def thor(x: float, y: float, h: float = 0.0) -> dict[str, float]:
    """World (x, y, height) -> THOR {x, y (up), z}."""
    return {"x": x, "y": h, "z": y}


def wall(room: int, a: tuple[float, float], b: tuple[float, float]) -> dict[str, Any]:
    return {"id": f"wall|{room}|{a[0]:.2f}|{a[1]:.2f}|{b[0]:.2f}|{b[1]:.2f}", "roomId": room,
            "polygon": [thor(*a), thor(*b), thor(*b, 2.5), thor(*a, 2.5)]}


def house() -> dict[str, Any]:
    rooms = [((0, 0), (4, 0), (4, 3), (0, 3)), ((4, 0), (7, 0), (7, 3), (4, 3))]
    walls = []
    for i, poly in enumerate(rooms, 1):
        for k in range(4):
            walls.append(wall(i, poly[k], poly[(k + 1) % 4]))
    door_wall = next(w for w in walls if w["id"] == "wall|1|4.00|0.00|4.00|3.00")
    other = next(w for w in walls if w["id"] == "wall|2|4.00|3.00|4.00|0.00")
    return {
        "rooms": [{"id": i, "roomType": "Kitchen",
                   "floorPolygon": [thor(*p) for p in poly]} for i, poly in enumerate(rooms, 1)],
        "walls": walls,
        "doors": [{"id": "door|1|2", "wall0": door_wall["id"], "wall1": other["id"],
                   "holePolygon": [{"x": DOOR_Y[0], "y": 0, "z": 0},
                                   {"x": DOOR_Y[1], "y": 2.0, "z": 0}]}],
        "windows": [],
        "objects": [
            {"id": "DiningTable|1|1", "position": thor(2.0, 2.5, 0.4), "children": [
                {"id": "Mug|1|2", "position": thor(2.0, 2.5, 0.85)},
                {"id": "Bowl|1|3", "position": thor(2.3, 2.5, 0.83), "children": [
                    {"id": "Apple|1|4", "position": thor(2.3, 2.5, 0.86)}]}]},
            {"id": "Sofa|2|1", "position": thor(6.0, 2.5, 0.4)},
        ],
    }


# ---- the episode ----------------------------------------------------------------------------
@dataclass
class SetSpec:
    run: str
    name: str
    poses: list[tuple[float, float, float]]  # base (x, y, yaw deg) in the run's start frame


@dataclass
class Synth:
    root: Path
    robot: str
    truth: dict[str, np.ndarray] = field(default_factory=dict)  # frame name -> T_world_cam
    start: dict[str, np.ndarray] = field(default_factory=dict)  # run -> T_world_start
    names: dict[str, list[str]] = field(default_factory=dict)  # set key -> frame names


RUN_STARTS = {"r1": (0.8, 1.45, 0.0), "r2": (6.0, 1.0, 180.0)}
DEFAULT_SETS = [
    SetSpec("r1", "start", [(0, 0, 0), (0.4, 0, 0), (0.8, 0, 0), (1.2, 0, 0)]),
    SetSpec("r1", "scan1", [(1.2, 0, 20), (1.2, 0, 40), (1.2, 0, -20), (1.2, 0, -40)]),
    SetSpec("r1", "chain1", [(1.6, 0, 0), (2.0, 0, 0), (2.4, 0, 0), (2.6, 0, 0),
                             (2.5, 0, 0)]),  # the fifth frame is held out and located
    SetSpec("r1", "loc", [(0.4, 0, 0), (0.8, 0, 5), (2.0, 0, 0), (1.2, 0, 180)]),
    # facing the wall x = 0: 0.3 m away (too close), 1.0 m away (a bare wall)
    SetSpec("r1", "wall", [(0.0, 0, 180), (-0.5, 0, 180), (0.2, 0, 180)]),
    SetSpec("r2", "start", [(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0), (1.0, 0.3, 0)]),
]


def camera_body(robot: str, x: float, y: float, yaw_deg: float) -> np.ndarray:
    """The optical pose of the camera of a base pose (in the frame of that base pose's
    parent): floor robots look forward, horizontal; the arm looks down 50° at a worktop."""
    if robot == "rebot_b601_rs":
        tilt = math.radians(50)
        R = rot_z(math.radians(yaw_deg)) @ np.array(
            [[math.cos(tilt), 0, math.sin(tilt)], [0, 1, 0],
             [-math.sin(tilt), 0, math.cos(tilt)]]) @ R_BODY_OPT
        return make_T(R, [x, y, 0.45])
    return make_T(rot_z(math.radians(yaw_deg)) @ R_BODY_OPT, [x, y, HEIGHT[robot]])


ARM_LABELS = ["approach", "grasp", "lift", "place", "pregrasp", "release", "retreat"]
LINK = {"rosmaster_x3_plus": "camera_link", "ainex": "head_tilt_link"}
WORKTOP_Z = {"house": 0.776, "arm": 1.10001}


def frame_file(robot: str, set_name: str, stamp: str, k: int, ext: str) -> str:
    """File names as the robots write them: only the base's follow <set>-date-time-NN."""
    if robot == "ainex":
        return f"{set_name}-{stamp}{'' if k == 0 else f'mp{k}'}.{ext}"
    if robot == "rebot_b601_rs":
        return f"{set_name}-{stamp}-{ARM_LABELS[k]}.{ext}"
    return f"{set_name}-{stamp}-{k + 1:02d}.{ext}"


def build(root: Path, eid: str = "e041", robot: str = "rosmaster_x3_plus",
          sets: list[SetSpec] | None = None, image: str = "jpeg", scene: str = "procthor:7",
          floor_z: float = 0.0, objects_lines: bool = False,
          meta_verdict: Any = None) -> Synth:
    """Write episode ``eid`` under ``root`` and return its truth."""
    sets = sets if sets is not None else DEFAULT_SETS
    arm = robot == "rebot_b601_rs"
    worktop = WORKTOP_Z["arm" if arm else "house"]
    ep = root / eid
    syn = Synth(ep, robot)
    w, h = SIZE[robot]
    # t, base T (world), camera position, camera optical R (world)
    track: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    t_set = T0
    for spec in sets:
        sx, sy, syaw = RUN_STARTS[spec.run]
        T_ws = make_T(rot_z(math.radians(syaw)), [sx, sy, worktop if arm else floor_z])
        T_ws = syn.start.setdefault(spec.run, T_ws)
        stamp = datetime.fromtimestamp(t_set, LOCAL).strftime("%Y%m%d-%H%M%S")
        folder = ep / "agent_files" / "runs" / spec.run / spec.name
        folder.mkdir(parents=True)
        views: dict[str, Any] = {}
        key = f"{spec.run}/{spec.name}"
        syn.names[key] = []
        for k, (x, y, yaw) in enumerate(spec.poses):
            t_f = t_set + 2 * k  # the first frame at the file-name time
            if robot == "rosmaster_x3_plus":  # the base names each frame with its own time
                stamp = datetime.fromtimestamp(t_f, LOCAL).strftime("%Y%m%d-%H%M%S")
            file = frame_file(robot, spec.name, stamp, k, "jpg" if image == "jpeg" else "png")
            T_sc = camera_body(robot, x, y, yaw)
            exif = tiff_exif(focal_entries(FX_EXIF)) if robot == "rosmaster_x3_plus" else None
            data = jpeg(w, h, exif) if image == "jpeg" else png(w, h, exif)
            (folder / file).write_bytes(data)
            if arm:
                views[file] = T_sc.tolist()
            elif robot == "ainex":
                views[file] = {"pose": T_sc.tolist(), "pan": 0.0, "tilt": 0.0, "pitch": 0.0,
                               "roll": 0.0, "imu_yaw": 0.0, "stamp": t_f}
            else:
                odo = T_sc.copy()
                odo[:2, 3] *= ODOM_SCALE
                views[file] = {"pose": odo.tolist(),
                               "base": [x * ODOM_SCALE, y * ODOM_SCALE, math.radians(yaw)]}
            T_wc = T_ws @ T_sc
            if arm:  # the simulator's camera sits a few mm off the believed kinematics
                T_wc[:3, 3] += ARM_CAM_ERROR
            name = f"{spec.run}/{spec.name}/{Path(file).stem}"
            syn.truth[name] = T_wc
            syn.names[key].append(name)
            base = T_ws if arm else T_ws @ make_T(rot_z(math.radians(yaw)), [x, y, 0.0])
            track.append((t_f, base, T_wc[:3, 3].copy(), T_wc[:3, :3].copy()))
        (folder / "views.json").write_text(json.dumps(views))
        t_set += 2 * len(spec.poses) + 4
    hello = {"engine": "mujoco", "scene": scene, "floor_z": floor_z, "rtf": 1.0,
             "starting": True,
             "worktop": {"name": "table", "z": worktop, "area_m2": 1.0,
                         "centroid": [0.0, 0.0], "bounds": [[-1, -1], [1, 1]]}}
    meta: dict[str, Any] = {"kind": "meta", "t": T0 - 5, "t0": T0 - 5, "robot": robot,
                            "robot_row": 1, "cameras": [CAM[robot]], "objects": []}
    if arm:  # the arm episodes' real nesting: meta.sim.hello, placement "worktop", no kind
        meta.update(placement="worktop", sim={"hello": hello, "sim_time": 0.0})
    else:
        meta.update(placement="p1", hello={**hello, "sim_time": 0.0})
    lines: list[dict[str, Any]] = [meta]
    # the truth track: still at each frame for 1 s, moving linearly in between (5 Hz)
    times = np.arange(T0 - 2, t_set + 2, 0.2)
    for t in times:
        k = int(np.searchsorted([x[0] for x in track], t))
        if k == 0 or k >= len(track):
            _, base, cam, R = track[0 if k == 0 else -1]
        else:
            (ta, ba, ca, Ra), (tb, bb, cb, Rb) = track[k - 1], track[k]
            if t - ta <= 0.5:
                base, cam, R = ba, ca, Ra
            elif tb - t <= 0.5:
                base, cam, R = bb, cb, Rb
            else:  # in motion: halfway poses, matching nothing the views hold
                u = (t - ta - 0.5) / (tb - ta - 1.0)
                side = np.array([0.0, 0.05, 0.0])  # off the straight line between frames
                base = make_T(ba[:3, :3], ba[:3, 3] + u * (bb[:3, 3] - ba[:3, 3])
                              + (0.0 if arm else side))
                cam, R = ca + u * (cb - ca) + side, Ra
        line_pos, line_R = base[:3, 3].copy(), base[:3, :3]
        if robot == "rosmaster_x3_plus":
            line_R = base[:3, :3] @ rot_z(MODEL_YAW)
        if arm:
            line_pos[:2] += ARM_LINE_OFFSET
            line_R = rot_z(math.radians(4.0))
        bodies: dict[str, Any] = {"base_link": {"pos": base[:3, 3].tolist(),
                                                "quat": quat_wxyz(base[:3, :3])}}
        if robot in LINK:
            bodies[LINK[robot]] = {"pos": cam.tolist(), "quat": quat_wxyz(R @ R_BODY_OPT.T)}
        if arm:
            bodies["gripper_end"] = {"pos": cam.tolist(), "quat": quat_wxyz(R @ ARM_MOUNT.T)}
        line: dict[str, Any] = {"kind": "robot", "t": float(t), "sim_time": float(t - T0),
                                "pos": line_pos.tolist(), "quat": quat_wxyz(line_R),
                                "cams": {CAM[robot]: cam.tolist(), "usb_cam": [0, 0, 0]},
                                "bodies": bodies}
        if arm:
            line["joints"] = {"j1": 0.0}
        lines.append(line)
    if objects_lines:
        lines.append({"kind": "objects", "t": t_set, "sim_time": 1.0, "bodies": {
            "apple_1": {"pos": [1.2, 1.5, worktop + 0.05], "quat": [1, 0, 0, 0]},
            "box_2": {"pos": [1.5, 1.2, worktop - 0.5], "quat": [1, 0, 0, 0]}}})
    lines.append({"kind": "render", "t": t_set, "file": "r.png"})
    lines.append({"kind": "end", "t": t_set + 1, "samples": len(times), "errors": 0})
    (ep / "truth.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    # the agent's shell-command log: not used by the replay
    (ep / "oms").mkdir(parents=True, exist_ok=True)
    (ep / "oms" / "agent_requests.jsonl").write_text("\n".join(json.dumps(
        {"id": i, "command": f"mapper.sh update /home/u/{k}", "result": "ok", "is_error": False,
         "subagent": None}) for i, k in enumerate(syn.names)) + "\n")
    rc = ep / "robot_control" / robot
    rc.mkdir(parents=True)
    (rc / "calibration.md").write_text("The camera looks forward.\n")
    meta_json: dict[str, Any] = {"robot": robot}
    if meta_verdict is not None:
        meta_json["verdict"] = meta_verdict
    (ep / "meta.json").write_text(json.dumps(meta_json))
    (ep / "verdict.md").write_text("pass\n")
    return syn


def files_of(syn: Synth, key: str) -> list[str]:
    folder = syn.root / "agent_files" / "runs" / key
    return sorted(p.name for p in folder.iterdir() if p.name != "views.json")
