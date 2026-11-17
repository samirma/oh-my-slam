"""Body links (oms/2.15; api.md §4 "Body links", map.md §1 "Body links"): the records, the
floors, paths and the camera relative pose with its sigmas (``mapping.body``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oh_my_slam.mapping import body as bl

# a camera looking along body +x (optical z forward, x right, y down)
R_BC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def Rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def T(R: np.ndarray | None = None, t: tuple[float, ...] | np.ndarray = (0.0, 0.0, 0.0)
      ) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = np.eye(3) if R is None else R
    M[:3, 3] = t
    return M


def mount(lever: tuple[float, float, float] = (0.08, 0.0, 0.3), source: str = "measured",
          **sig: float) -> dict:
    return {"T_body_cam": T(R_BC, lever).tolist(), "source": source, **sig}


def motion(frm: str, M: np.ndarray, ys: str = "imu", sy: float = 2.0,
           ts: str = "dead_reckoning", sp: float = 0.008, **kw: float) -> dict:
    return {"from": frm, "T_from_body": M.tolist(), "yaw_source": ys, "sigma_yaw_deg": sy,
            "translation_source": ts, "sigma_pos_m": sp, **kw}


def test_turn_in_place_baseline_and_sigma() -> None:
    a = bl.body_of("a", "c", {"image_stamp_s": 0.0, "standing": True}, mount())
    b = bl.body_of("b", "c", {"image_stamp_s": 2.0, "standing": True,
                             "motion": motion("a", T(Rz(40.0)))}, mount())
    L = bl.path_link({"a": a, "b": b}, "a", "b")
    assert L is not None
    # |t| = 2 r sin(20 deg), r 0.08
    assert L.baseline_m == pytest.approx(2 * 0.08 * math.sin(math.radians(20.0)), abs=1e-9)
    assert L.rotation_deg == pytest.approx(40.0)
    # dead reckoning's floor is turn-dependent: 0.001 m/deg x 40 deg = 0.04 m > 0.008 sent
    assert b.motion is not None and b.motion.sigma_pos_m == pytest.approx(0.04)
    assert L.sigma_t / L.baseline_m > 0.20  # a stepping turn of 40 deg: no scale baseline
    # the fixed mount is one reading: its error cancels but for the turn
    assert bl.same_reading(a.mount, b.mount)  # type: ignore[arg-type]
    assert L.sigma_rot_deg == pytest.approx(math.sqrt(2.0 ** 2 + 2 * 1.0 ** 2
                                                      + (2 * math.sin(math.radians(20)) * 0.5)
                                                      ** 2))


def test_lidar_turn_is_usable() -> None:
    a = bl.body_of("a", "c", {"image_stamp_s": 0.0, "standing": True}, mount())
    b = bl.body_of("b", "c", {"image_stamp_s": 2.0, "standing": True,
                             "motion": motion("a", T(Rz(40.0)), ts="lidar_match", sp=0.006)},
                  mount())
    L = bl.path_link({"a": a, "b": b}, "a", "b")
    assert L is not None and L.sigma_t / L.baseline_m < 0.20


def test_head_pan_with_body_still_is_usable() -> None:
    """A pan of +-30 deg about a head axis 0.03 m behind the camera, the body still (``none``),
    a measured mount: the baseline is the camera's lever about the pan axis, and with no slip
    the pair is a usable scale baseline (the client's case 6)."""
    def head(deg: float) -> dict:
        Rp = Rz(deg)
        c = np.array([0.05, 0.0, 0.4]) + Rp @ np.array([0.03, 0.0, 0.0])
        return {"T_body_cam": T(Rp @ R_BC, c).tolist(), "source": "measured"}

    still = motion("a", T(), ys="none", sy=0.0, ts="none", sp=0.0)
    a = bl.body_of("a", "c", {"image_stamp_s": 0.0, "standing": True, "mount": head(-30.0),
                             "mount_stamp_s": 0.2}, None)
    b = bl.body_of("b", "c", {"image_stamp_s": 3.0, "standing": True, "mount": head(30.0),
                             "mount_stamp_s": 2.6, "motion": still}, None)
    assert a.mount is not None and b.mount is not None and a.mount.fresh and b.mount.fresh
    assert not bl.same_reading(a.mount, b.mount)
    L = bl.path_link({"a": a, "b": b}, "a", "b")
    assert L is not None
    assert L.baseline_m == pytest.approx(2 * 0.03 * math.sin(math.radians(30.0)), abs=1e-9)
    assert L.rotation_deg == pytest.approx(60.0)
    assert L.sigma_t / L.baseline_m <= 0.20
    assert b.motion is not None
    assert (b.motion.sigma_yaw_deg, b.motion.sigma_tilt_deg, b.motion.sigma_pos_m) == (
        0.1, 0.1, 0.001)


def test_floors_and_defaults() -> None:
    m = bl.mount_of({"T_body_cam": np.eye(4).tolist(), "source": "commanded"}, False)
    assert (m.sigma_pos_m, m.sigma_rot_deg) == (0.010, 3.0)
    m = bl.mount_of({"T_body_cam": np.eye(4).tolist(), "source": "commanded",
                    "sigma_pos_m": 0.0, "sigma_rot_deg": 0.0}, False)
    assert (m.sigma_pos_m, m.sigma_rot_deg) == (0.005, 1.5)
    m = bl.mount_of({"T_body_cam": np.eye(4).tolist(), "source": "measured"}, True)
    assert (m.sigma_pos_m, m.sigma_rot_deg, m.fixed) == (0.002, 0.5, True)
    straight = T(None, (1.0, 0.0, 0.0))
    assert bl.translation_floor("odometry", straight) == pytest.approx(math.hypot(0.005, 0.1))
    assert bl.translation_floor("lidar_match", straight) == 0.005
    assert bl.translation_floor("imu", straight) == 0.05
    assert bl.translation_floor("dead_reckoning", T(Rz(5.0))) == 0.010
    assert bl.translation_floor("dead_reckoning", T(Rz(-90.0))) == pytest.approx(0.09)
    mo = bl.motion_of(motion("a", T(), ys="odometry", sy=0.0, ts="imu", sp=0.0))
    assert (mo.sigma_yaw_deg, mo.sigma_tilt_deg, mo.sigma_pos_m) == (1.0, 1.0, 0.05)
    mo = bl.motion_of(motion("a", T(), sigma_tilt_deg=0.3))
    assert mo.sigma_tilt_deg == 0.3
    assert bl.is_still(T()) and not bl.is_still(T(None, (0.002, 0.0, 0.0)))
    assert not bl.is_still(T(Rz(0.2)))


def test_stamps() -> None:
    def b(stamp: float, standing: bool) -> bl.Body:
        return bl.body_of("x", "c", {"image_stamp_s": 10.0, "standing": standing,
                                    "mount": mount(), "mount_stamp_s": stamp}, None)

    assert b(10.03, False).mount.fresh  # type: ignore[union-attr]
    assert not b(10.05, False).mount.fresh  # type: ignore[union-attr]
    assert b(9.5, True).mount.fresh  # type: ignore[union-attr]
    assert not b(9.4, True).mount.fresh  # type: ignore[union-attr]
    none = bl.body_of("x", "c", {"image_stamp_s": 0.0, "standing": False}, None)
    assert none.mount is None and none.motion is None


def test_records_round_trip() -> None:
    b = bl.body_of("b", "c", {"image_stamp_s": 2.0, "standing": True, "mount": mount(),
                             "mount_stamp_s": 2.0, "motion": motion("a", T(Rz(10.0)))}, None)
    back = bl.Body.from_json(b.to_json())
    assert back.to_json() == b.to_json()
    bare = bl.Body("n", "c", False, None, None)
    assert bl.Body.from_json(bare.to_json()).to_json() == bare.to_json()


def test_paths() -> None:
    def rec(name: str, frm: str | None, M: np.ndarray, cam: str = "c",
            m: dict | None = None) -> bl.Body:
        d: dict = {"image_stamp_s": 0.0, "standing": True}
        if frm is not None:
            d["motion"] = motion(frm, M, ts="lidar_match", sp=0.005)
        return bl.body_of(name, cam, d, m or mount())

    # a - b - c (b from a, c from b), d from a, e of another camera, f from a missing frame
    bodies = {"a": rec("a", None, T()), "b": rec("b", "a", T(Rz(30.0))),
              "c": rec("c", "b", T(Rz(30.0))), "d": rec("d", "a", T(None, (0.5, 0.0, 0.0))),
              "e": rec("e", "a", T(), cam="other"), "f": rec("f", "zz", T())}
    steps = bl.find_path(bodies, "c", "d")
    assert steps is not None and [s.inverted for s in steps] == [True, True, False]
    assert bl.find_path(bodies, "a", "a") is None
    assert bl.find_path(bodies, "a", "e") is None  # another camera
    assert bl.find_path(bodies, "a", "f") is None  # its from is unknown
    assert bl.find_path(bodies, "a", "nope") is None
    assert bl.find_path(bodies, "c", "d", max_links=2) is None
    # composing c -> d equals the poses' product
    comp = bl.compose(steps)
    want = np.linalg.inv(T(Rz(30.0)) @ T(Rz(30.0))) @ T(None, (0.5, 0.0, 0.0))
    assert np.allclose(comp.T, want)
    # a forward-then-back path agrees with its Monte Carlo spread
    L = bl.path_link(bodies, "c", "d")
    assert L is not None
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(4000):
        def noisy(m: bl.Motion) -> np.ndarray:
            th = rng.normal(0.0, [math.radians(m.sigma_tilt_deg)] * 2
                            + [math.radians(m.sigma_yaw_deg)])
            ang = float(np.linalg.norm(th))
            k = th / ang
            Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            dR = np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * Kx @ Kx
            M = m.T.copy()
            M[:3, :3] = M[:3, :3] @ dR
            M[:3, 3] += rng.normal(0.0, m.sigma_pos_m / math.sqrt(3.0), 3)
            return M

        Tb = noisy(bodies["b"].motion)  # type: ignore[arg-type]
        Tc = noisy(bodies["c"].motion)  # type: ignore[arg-type]
        Td = noisy(bodies["d"].motion)  # type: ignore[arg-type]
        Tcd = np.linalg.inv(Tb @ Tc) @ Td
        M = bodies["c"].mount.T  # type: ignore[union-attr]
        errs.append(np.linalg.inv(M) @ Tcd @ M)
    t = np.array([e[:3, 3] for e in errs])
    mc = math.sqrt(float(np.trace(np.cov(t.T))))
    mount_part = math.hypot(bl.camera_link(bodies["c"].mount, bodies["d"].mount,  # type: ignore[arg-type]
                                          comp).sigma_t, 0.0)
    # the propagated sigma covers the body part (within 15 %) plus the mount terms
    assert mc <= mount_part * 1.15
    assert mount_part <= math.hypot(mc, 0.0) * 1.15 + 0.004
    # stale or missing mounts give no link
    stale = bl.body_of("s", "c", {"image_stamp_s": 0.0, "standing": False, "mount": mount(),
                                 "mount_stamp_s": 1.0, "motion": motion("a", T())}, None)
    assert stale.mount is not None and not stale.mount.fresh
    assert bl.path_link({**bodies, "s": stale}, "a", "s") is None
    bare = bl.Body("g", "c", True, None, bl.motion_of(motion("a", T())))
    assert bl.path_link({**bodies, "g": bare}, "a", "g") is None
    assert bl.path_link(bodies, "a", "f") is None
    assert bl.path_link(bodies, "x", "a") is None
