"""Body links (oms/2.15; api.md §4 "Body links", map.md §1 "Body links"): a frame's camera
mount on the robot's body and the body's motion since an earlier frame, composed into a metric
camera-to-camera relative pose with its sigmas.

* **Records** (``Body``): a frame's ``body`` as the request sent it, its mount resolved (its own,
  else its camera's ``body_mount``), the sigmas defaulted and raised to their floors, the mount
  fresh or stale by its stamps. Kept with the frame in the mapper's store (``to_json``).
* **Paths** (``find_path``, ``compose``): links form trees (one ``from`` per frame); two frames
  of one camera joined through at most ``LINK_PATH_MAX`` links compose their ``T_from_body``,
  the sigmas propagated linearly (each link's rotation error pivots about the child body's
  origin, which is where its ``T_from_body`` puts it; its translation error moves everything
  past it alike).
* **Camera relative pose** (``camera_link``): T_camA_camB = T_body_camA⁻¹ · T_bodyA_bodyB ·
  T_body_camB, the mounts' errors common (the same reading: they cancel but for the turn) or
  independent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# -- constants (README "Body links", oms/2.15) --------------------------------------------------
MOUNT_STAMP_TOL_S = 0.03  # a joint reading this close to the exposure is fresh
MOUNT_STAMP_STANDING_MAX_S = 0.5  # ... and this close for a standing frame
# mount sigmas (position m, rotation deg) by source: the default when none is sent, the floor
MOUNT_DEFAULT = {"measured": (0.002, 0.5), "commanded": (0.010, 3.0)}
MOUNT_FLOOR = {"measured": (0.001, 0.2), "commanded": (0.005, 1.5)}
# relative-motion floors by source: yaw (deg), translation (m; see ``translation_floor``)
YAW_FLOOR_DEG = {"imu": 0.5, "lidar_match": 0.5, "odometry": 1.0, "dead_reckoning": 1.0,
                 "none": 0.1}
TRANSLATION_FLOOR_M = {"lidar_match": 0.005, "odometry": 0.005, "dead_reckoning": 0.010,
                       "imu": 0.05, "none": 0.001}
ODOMETRY_SCALE_REL = 0.10  # odometry: at least this share of its translation
DEAD_RECKONING_M_PER_DEG = 0.001  # dead reckoning: at least this per degree turned
TILT_DEFAULT_DEG = 1.0  # sigma_tilt_deg when absent
NONE_TILT_DEG = 0.1  # ... for a body that did not move (``none``)
NONE_MAX_M = 0.001  # a ``none`` motion's T_from_body is the identity within these
NONE_MAX_DEG = 0.1
SAME_MOUNT_M = 0.001  # two mounts this close, of one source, are the same reading
SAME_MOUNT_DEG = 0.1
LINK_PATH_MAX = 4  # links in a path (the oms/2.14 order gap)
LINK_SCALE_MIN = 3  # link-measured keyframe scales that make scale evidence


def rot_angle_deg(R: Array) -> float:
    """The angle of a rotation matrix, degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


def yaw_deg(R: Array) -> float:
    """The yaw (about z) of a rotation matrix, degrees."""
    return math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))


@dataclass
class Mount:
    """A camera's pose in the body frame and its sigmas (3-D RMS, m and deg), whether it is the
    camera's fixed ``body_mount`` and whether it was read close enough to the exposure."""

    T: Array
    source: str
    sigma_pos_m: float
    sigma_rot_deg: float
    fresh: bool = True
    fixed: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"T": self.T.tolist(), "source": self.source, "sigma_pos_m": self.sigma_pos_m,
                "sigma_rot_deg": self.sigma_rot_deg, "fresh": self.fresh, "fixed": self.fixed}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Mount:
        return Mount(np.asarray(d["T"], np.float64), d["source"], float(d["sigma_pos_m"]),
                     float(d["sigma_rot_deg"]), bool(d["fresh"]), bool(d["fixed"]))


@dataclass
class Motion:
    """The body's motion from the ``from`` frame (``T``: the pose of this frame's body in the
    ``from`` frame's), its sigmas raised to their sources' floors."""

    source_name: str
    T: Array
    yaw_source: str
    translation_source: str
    sigma_yaw_deg: float
    sigma_tilt_deg: float
    sigma_pos_m: float

    def to_json(self) -> dict[str, Any]:
        return {"from": self.source_name, "T": self.T.tolist(), "yaw_source": self.yaw_source,
                "translation_source": self.translation_source,
                "sigma_yaw_deg": self.sigma_yaw_deg, "sigma_tilt_deg": self.sigma_tilt_deg,
                "sigma_pos_m": self.sigma_pos_m}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Motion:
        return Motion(d["from"], np.asarray(d["T"], np.float64), d["yaw_source"],
                      d["translation_source"], float(d["sigma_yaw_deg"]),
                      float(d["sigma_tilt_deg"]), float(d["sigma_pos_m"]))


@dataclass
class Body:
    """A frame's body record (``FrameBody`` resolved)."""

    name: str
    camera_id: str
    standing: bool
    mount: Mount | None
    motion: Motion | None

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "camera_id": self.camera_id, "standing": self.standing,
                "mount": None if self.mount is None else self.mount.to_json(),
                "motion": None if self.motion is None else self.motion.to_json()}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Body:
        return Body(d["name"], d["camera_id"], bool(d["standing"]),
                    None if d.get("mount") is None else Mount.from_json(d["mount"]),
                    None if d.get("motion") is None else Motion.from_json(d["motion"]))


def mount_of(m: dict[str, Any], fixed: bool, fresh: bool = True) -> Mount:
    """A request's ``BodyMount`` with its sigmas defaulted by source and raised to the floor."""
    src = str(m["source"])
    dp, dr = MOUNT_DEFAULT[src]
    fp, fr = MOUNT_FLOOR[src]
    return Mount(np.asarray(m["T_body_cam"], np.float64), src,
                 max(float(m.get("sigma_pos_m", dp)), fp),
                 max(float(m.get("sigma_rot_deg", dr)), fr), fresh, fixed)


def translation_floor(source: str, T: Array) -> float:
    """A relative motion's translation sigma floor (README "Body links"): a source's constant;
    odometry at least ``ODOMETRY_SCALE_REL`` of its translation besides (its scale error); dead
    reckoning at least ``DEAD_RECKONING_M_PER_DEG`` per degree turned (a stepping body slides
    as it turns on the spot)."""
    base = TRANSLATION_FLOOR_M[source]
    if source == "odometry":
        return math.hypot(base, ODOMETRY_SCALE_REL * float(np.linalg.norm(T[:3, 3])))
    if source == "dead_reckoning":
        return max(base, DEAD_RECKONING_M_PER_DEG * abs(yaw_deg(T[:3, :3])))
    return base


def is_still(T: Array) -> bool:
    """Whether a ``none`` motion's ``T_from_body`` is the identity within ``NONE_MAX_M`` and
    ``NONE_MAX_DEG``."""
    return float(np.linalg.norm(T[:3, 3])) <= NONE_MAX_M \
        and rot_angle_deg(T[:3, :3]) <= NONE_MAX_DEG


def motion_of(m: dict[str, Any]) -> Motion:
    """A request's ``BodyMotion`` with its sigmas raised to their sources' floors."""
    T = np.asarray(m["T_from_body"], np.float64)
    ys, ts = str(m["yaw_source"]), str(m["translation_source"])
    still = "none" in (ys, ts)
    tilt = float(m.get("sigma_tilt_deg", NONE_TILT_DEG if still else TILT_DEFAULT_DEG))
    return Motion(str(m["from"]), T, ys, ts, max(float(m["sigma_yaw_deg"]), YAW_FLOOR_DEG[ys]),
                  max(tilt, NONE_TILT_DEG if still else 0.0),
                  max(float(m["sigma_pos_m"]), translation_floor(ts, T)))


def body_of(name: str, camera_id: str, b: dict[str, Any],
            camera_mount: dict[str, Any] | None) -> Body:
    """A frame's resolved body record: its own mount (fresh when its joint reading lies within
    ``MOUNT_STAMP_TOL_S`` of the exposure, ``MOUNT_STAMP_STANDING_MAX_S`` when standing), else
    its camera's fixed mount (always fresh); its motion."""
    standing = bool(b["standing"])
    mount = None
    if b.get("mount") is not None:
        dt = abs(float(b["mount_stamp_s"]) - float(b["image_stamp_s"]))
        fresh = dt <= (MOUNT_STAMP_STANDING_MAX_S if standing else MOUNT_STAMP_TOL_S)
        mount = mount_of(b["mount"], False, fresh)
    elif camera_mount is not None:
        mount = mount_of(camera_mount, True)
    motion = None if b.get("motion") is None else motion_of(b["motion"])
    return Body(name, camera_id, standing, mount, motion)


# -- paths ---------------------------------------------------------------------------------------


@dataclass
class Step:
    """One link of a path: the child's motion and whether the path walks it from child to
    parent (inverted)."""

    motion: Motion
    inverted: bool


def find_path(bodies: dict[str, Body], a: str, b: str, max_links: int = LINK_PATH_MAX
              ) -> list[Step] | None:
    """The links from frame ``a`` to frame ``b`` (breadth first, at most ``max_links``) through
    ``bodies`` (each a frame's record, the frames of one camera only); None without one."""
    if a == b or a not in bodies or b not in bodies:
        return None
    cam = bodies[a].camera_id
    nbrs: dict[str, list[tuple[str, Step]]] = {}
    for name, body in bodies.items():
        m = body.motion
        if m is None or body.camera_id != cam or m.source_name not in bodies \
                or bodies[m.source_name].camera_id != cam:
            continue
        nbrs.setdefault(m.source_name, []).append((name, Step(m, False)))
        nbrs.setdefault(name, []).append((m.source_name, Step(m, True)))
    seen: dict[str, list[Step]] = {a: []}
    frontier = [a]
    for _ in range(max_links):
        nxt = []
        for u in frontier:
            for v, step in nbrs.get(u, []):
                if v in seen:
                    continue
                seen[v] = [*seen[u], step]
                if v == b:
                    return seen[v]
                nxt.append(v)
        frontier = nxt
    return None


@dataclass
class Composed:
    """A path's body motion ``T`` (body B in body A) and, per link, its rotation error's pivot
    (the child body's origin, in body A), that body's rotation (in body A) and its sigmas."""

    T: Array
    pivots: list[tuple[Array, Array, float, float]]  # (pivot, R_A_child, yaw deg, tilt deg)
    var_pos: float  # the links' own translation variances, summed
    var_rot_deg2: float  # yaw and tilt (both axes) variances, summed


def compose(steps: list[Step]) -> Composed:
    """The body motion along ``steps`` and what its sigma propagation needs (``Composed``)."""
    T = np.eye(4)
    pivots = []
    var_pos = var_rot = 0.0
    for s in steps:
        m = s.motion
        if s.inverted:
            # the child at the start: its rotation error pivots there
            child = T.copy()
            T = T @ np.linalg.inv(m.T)
        else:
            T = T @ m.T
            child = T.copy()
        pivots.append((child[:3, 3].copy(), child[:3, :3].copy(), m.sigma_yaw_deg,
                       m.sigma_tilt_deg))
        var_pos += m.sigma_pos_m ** 2
        var_rot += m.sigma_yaw_deg ** 2 + 2.0 * m.sigma_tilt_deg ** 2
    return Composed(T, pivots, var_pos, var_rot)


def same_reading(a: Mount, b: Mount) -> bool:
    """Whether two mounts are one reading (both the camera's fixed mount, or equal within
    ``SAME_MOUNT_M`` and ``SAME_MOUNT_DEG`` with one source): their errors are common."""
    if a.fixed and b.fixed:
        return True
    return a.source == b.source \
        and float(np.linalg.norm(a.T[:3, 3] - b.T[:3, 3])) <= SAME_MOUNT_M \
        and rot_angle_deg(a.T[:3, :3].T @ b.T[:3, :3]) <= SAME_MOUNT_DEG


@dataclass
class CameraLink:
    """The camera relative pose of a path (camera B in camera A's optical frame), its
    translation sigma (m) and rotation sigma (deg)."""

    T: Array
    sigma_t: float
    sigma_rot_deg: float

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T[:3, 3]))

    @property
    def rotation_deg(self) -> float:
        return rot_angle_deg(self.T[:3, :3])


def camera_link(ma: Mount, mb: Mount, c: Composed) -> CameraLink:
    """The camera relative pose of a composed path between frames whose mounts are ``ma`` and
    ``mb`` (map.md §1 "Body links", camera relative pose)."""
    T = np.linalg.inv(ma.T) @ c.T @ mb.T
    cb = (c.T @ mb.T)[:3, 3]  # camera B's centre in body A
    var = c.var_pos
    for pivot, R, sy, st in c.pivots:
        lever = cb - pivot
        z, x, y = R[:, 2], R[:, 0], R[:, 1]
        var += math.radians(sy) ** 2 * float(np.linalg.norm(np.cross(z, lever))) ** 2
        var += math.radians(st) ** 2 * (float(np.linalg.norm(np.cross(x, lever))) ** 2
                                        + float(np.linalg.norm(np.cross(y, lever))) ** 2)
    psi = rot_angle_deg(c.T[:3, :3])
    turn = 2.0 * math.sin(math.radians(psi) / 2.0)
    if same_reading(ma, mb):
        mp = turn * max(ma.sigma_pos_m, mb.sigma_pos_m)
        mr = turn * max(ma.sigma_rot_deg, mb.sigma_rot_deg)
    else:
        mp = math.hypot(ma.sigma_pos_m, mb.sigma_pos_m)
        mr = math.hypot(ma.sigma_rot_deg, mb.sigma_rot_deg)
    t = float(np.linalg.norm(T[:3, 3]))
    var += mp ** 2 + (t * math.radians(ma.sigma_rot_deg)) ** 2
    return CameraLink(T, math.sqrt(var), math.sqrt(c.var_rot_deg2 + mr ** 2))


def path_link(bodies: dict[str, Body], a: str, b: str, max_links: int = LINK_PATH_MAX
              ) -> CameraLink | None:
    """The camera relative pose of frame ``b`` in frame ``a`` through a path (``find_path``),
    both mounts fresh; None without one."""
    ma = bodies[a].mount if a in bodies else None
    mb = bodies[b].mount if b in bodies else None
    if ma is None or mb is None or not ma.fresh or not mb.fresh:
        return None
    steps = find_path(bodies, a, b, max_links)
    if steps is None:
        return None
    return camera_link(ma, mb, compose(steps))
