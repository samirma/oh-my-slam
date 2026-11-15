"""Which plane is a frame's own floor, and which the map's (map.md §2 "Floor check", §6):

* a plane's support is counted on all its points, never on the sample it was fitted on (under
  which every plane larger than the sample counted the same, and a sliver passed for the floor);
* a frame's own floor is a plane with real support (``FLOOR_MIN_SHARE`` of its points): the one
  nearest the sent height when one lies near it, else the widest plausible one. The client's
  low camera (the model's floor 0.9 m below a camera sent as 0.37 m up) took a 0.3 % sliver that
  lay at 0.37 m; its ``scale_check`` then read 1.0;
* a robot on a worktop that sees wider counters 0.2 m below it: its frames' own floor is the
  worktop, and the map's floor stays z 0 (it was the counters, −0.2 m: the worktop held less
  than half the counters' points and was no candidate; the frames sent from the worktop then
  failed the height gate)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import analysis, quality
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import AnchoredMap
from oh_my_slam.reconstruction.pointcloud import pixel_points
from oh_my_slam.service.app import Service, create_app
from oh_my_slam.service.derived import Derived
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.store import Store
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import Updater
from tests.fakes.service import FakeBackend, FakeInference
from tests.synth.room import Room, base_pose, jpeg, look, png_depth, render
from tests.synth.table import K_TABLE, Item, TableScene
from tests.unit import test_round14
from tests.unit.test_anchored_objects import detections
from tests.unit.test_round14 import frame, prior
from tests.unit.test_service_api import Client
from tests.unit.test_service_nav import decode

K640 = Intrinsics(270.0, 270.0, 320.0, 240.0, 640, 480)  # the client's camera
# the model's floor 0.9 m below the camera, a 15 cm box top 0.37 m below it (0.3 % of the frame)
LOW = Room(half=(20.0, 20.0), boxes=[((0.8, -0.3, 0.0), (0.95, -0.15, 0.53))])
# a 0.55 x 0.6 m worktop 0.5 m below the camera, among counters 0.2 m below the top
TOP = Room(half=(20.0, 20.0), boxes=[((0.45, -0.3, 0.0), (1.0, 0.3, 1.1)),
                                     ((1.0, -6.0, 0.0), (8.0, 6.0, 0.9)),
                                     ((-1.0, 0.3, 0.0), (1.0, 6.0, 0.9)),
                                     ((-1.0, -6.0, 0.0), (1.0, -0.3, 0.9))])
# a wall with two ledges below the camera, each under 2 % of the frame
LEDGES = Room(half=(1.2, 20.0), height=3.0, boxes=[((0.9, -0.5, 0.0), (1.2, -0.25, 0.5)),
                                                   ((0.9, 0.2, 0.0), (1.2, 0.45, 0.62))])


def _view(room: Room, height: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    T = look(0.0, (0.0, 0.0, height), pitch)
    return render(room, T, K640, supersample=1)[1], T


def test_a_plane_s_support_is_all_its_points() -> None:
    """``horizontal_plane`` counts every point on the plane: the floor of the low view holds
    three quarters of the frame, not the 5000-point sample's ~2000 inliers."""
    z, T = _view(LOW, 0.9, -30.0)
    p = pixel_points(z, K640, an.edge_free(z))[0] @ T[:3, :3].T
    q = p[p[:, 2] < -an.MIN_CAMERA_HEIGHT]
    floor = an.horizontal_plane(q, 0.9)
    box = an.horizontal_plane(q, 0.37)
    assert floor is not None and box is not None
    assert floor[2] > 0.7 * len(p) > 40 * an.FLOOR_SAMPLE  # (4991 before)
    assert 400 < box[2] < 0.01 * len(p)
    assert floor[4] <= an.FLOOR_SAMPLE  # the fit's sample: a camera-only map's floor count
    assert [round(h, 2) for h in an.floor_candidates(p, sampled=True)] == [0.9, 0.37]
    # the planes of a frame: every one with a share of the best one's points and of the frame's
    assert [round(h, 2) for h in an.floor_candidates(p)] == [0.9]  # (0.9 and 0.37 before)
    every = [round(h, 2) for h in an.floor_candidates(p, 0.0)]
    assert every[0] == 0.9 and 0.37 in every
    assert an.floor_candidates(p, 0.0, min_share=an.FLOOR_MIN_SHARE) == an.floor_candidates(p)
    assert an.floor_candidates(p, 0.0, min_share=0.9) == []


def test_a_sliver_at_the_sent_height_is_not_the_floor() -> None:
    """The client's low camera: sent 0.37 m, the model's floor 0.9 m below it. The floor is the
    wide plane, the scale 0.41, and ``scale_check`` says so (before: the box top, a ratio of 1.0
    on 0.5 % of the frame); on a map at that depth scale the frame is not judged."""
    z, T = _view(LOW, 0.9, -30.0)
    R = T[:3, :3]
    assert an.own_floor(z, K640, R, 0.37, an.floor_band("mono", 0.37, 0.03)) == pytest.approx(
        0.9, abs=0.01)
    assert analysis.height_factor(z, K640, R, 0.37, 0.03) == pytest.approx(0.37 / 0.9, rel=0.01)
    check = analysis.scale_check(z, K640, R, 0.37, "monocular", 0.03)
    assert check["model_height_m"] == pytest.approx(0.9, abs=0.01)
    assert check["ratio"] == pytest.approx(0.411, abs=0.01) and check["floor_inlier_share"] > 0.7
    assert not an.floor_verdict("mono", 0.37, 0.03, T, 0.05, z, K640).judged
    # the same view as a depth image: its floor disagrees with the sent height (the frame is
    # left out), and at the true height it agrees
    assert not an.floor_verdict("sensor", 0.37, 0.03, T, 0.01, z, K640).ok
    assert an.floor_verdict("sensor", 0.9, 0.03, T, 0.01, z, K640) == an.FloorCheck()
    assert analysis.scale_check(z, K640, R, 0.9, "sensor")["ratio"] == pytest.approx(1.0, abs=0.01)
    # a monocular depth near its true scale takes the plane near the height
    got = an.floor_verdict("mono", 0.9, 0.03, T, 0.05, z * 1.2, K640)
    assert got.judged and got.factor == pytest.approx(1 / 1.2, rel=0.02)


def test_quality_scale_check_reads_the_floor_not_the_sliver(tmp_path: Path) -> None:
    """``quality.scale_check`` (api.md §4, oms/2.8; ``AnchoredMap.floor_ratios``) and
    ``quality.scale_check``'s statistics take the same plane as the floor check: the low view's
    ratio is its sent height over the wide floor, 0.37 / 0.9 (before: 1.0, the box top that
    lay at the sent height)."""
    T = look(0.0, (0.0, 0.0, 0.9), -30.0)
    img, z = render(LOW, T, K640, supersample=1)[:2]
    mono = (z.astype(np.float32), K640, None, np.array([0.0, -1.0, 0.0]))
    m = AnchoredMap(tmp_path / "m")
    m.update([an.FrameIn("s0", "cam", img, K640, prior(T), None, None, 0.37, 0.03, 1000,
                         lambda: mono)])
    assert m.floor_ratios() == [pytest.approx(0.37 / 0.9, abs=0.01)]
    got = quality.scale_check(m.floor_ratios())
    assert got is not None and got["median_ratio"] == pytest.approx(0.411, abs=0.01)


def test_the_worktop_under_the_robot_is_its_floor_among_wider_counters() -> None:
    """A camera 0.5 m above a worktop that holds 7 % of the frame, counters 0.2 m lower that
    hold 59 %: the plane near the sent height is the floor (never simply the widest), for a
    depth image and for a monocular depth 20 % off; without a height, the lowest."""
    z, T = _view(TOP, 1.6, -30.0)
    R = T[:3, :3]
    p = pixel_points(z, K640, an.edge_free(z))[0] @ R.T
    share = {h: float((np.abs(-p[:, 2] - h) < 0.02 * h).mean()) for h in (0.5, 0.7)}
    assert 0.05 < share[0.5] < 0.1 and share[0.7] > 0.5
    assert an.own_floor(z, K640, R, 0.5, an.floor_band("sensor", 0.5, 0.01)) == pytest.approx(
        0.5, abs=0.005)
    assert an.floor_verdict("sensor", 0.5, 0.01, T, 0.01, z, K640) == an.FloorCheck()
    got = an.floor_verdict("mono", 0.5, 0.01, T, 0.05, z * 1.2, K640)
    assert got.judged and got.factor == pytest.approx(1 / 1.2, rel=0.02)
    assert analysis.height_factor(z * 1.2, K640, R, 0.5, 0.01) == pytest.approx(1 / 1.2, rel=0.02)
    assert an.own_floor(z, K640, R, None, 0.0) == pytest.approx(0.7, abs=0.005)
    assert analysis.own_floor_z(z, K640, T, None, "sensor") == pytest.approx(0.9, abs=0.005)
    assert analysis.own_floor_z(z, K640, T, 0.5, "sensor", 0.01) == pytest.approx(1.1, abs=0.005)
    # no plane near the height: the widest plausible one, and none when none is plausible
    assert an.own_floor(z, K640, R, 0.6, 0.03) == pytest.approx(0.7, abs=0.005)
    assert an.own_floor(z, K640, R, 0.05, 0.01) is None
    # the band a plane is near its height in: the floor check's bound, a monocular depth's bound
    assert an.floor_band("sensor", 0.5, 0.01) == pytest.approx(3 * np.hypot(0.01, 0.005))
    assert an.floor_band("sensor", 0.5, None, 0.03) == pytest.approx(
        3 * np.sqrt(0.02 ** 2 + 0.005 ** 2 + 0.03 ** 2 / 3))
    assert an.floor_band("mono", 0.5, 0.01) == pytest.approx(an.MONO_FLOOR_BOUND_REL * 0.5)
    assert an.floor_band("mono", 0.1, 0.03) == pytest.approx(3 * np.hypot(0.03, 0.001))


def test_the_widest_plausible_plane_when_the_widest_is_implausible() -> None:
    """map.md §2: with no plane near the expected height, the widest plane whose implied scale
    lies within ``SCALE_RANGE`` (0.2-5); a widest one beyond it gives way to the next widest.
    The worktop view: counters 0.7 m below (the widest), the worktop 0.5 m; a height of 0.12 m
    implies 0.17 on the counters (implausible) and 0.24 on the worktop."""
    z, T = _view(TOP, 1.6, -30.0)
    R = T[:3, :3]
    assert an.SCALE_RANGE == (0.2, 5.0)
    assert 0.12 / 0.7 < an.SCALE_RANGE[0] <= 0.12 / 0.5
    assert an.own_floor(z, K640, R, 0.6, 0.03) == pytest.approx(0.7, abs=0.005)  # the widest
    assert an.own_floor(z, K640, R, 0.12, 0.01) == pytest.approx(0.5, abs=0.005)  # the next


def test_the_frame_s_floor_and_its_scale_check_take_one_plane() -> None:
    """Frame analysis (api.md §7): its own floor (``own_floor_z``) takes the plane in the same
    band (``floor_band`` of the depth's kind and the height's sigma) as ``scale_check``. A depth
    image of the worktop view sent 0.57 m up: with a 0.01 m sigma no plane is that near and both
    take the widest (the counters, 0.7 m); with 0.1 m both take the nearest (the worktop, 0.5
    m). Before, the floor took a fixed 0.06 m band: the counters, whatever the sigma."""
    z, T = _view(TOP, 1.6, -30.0)
    for sigma, h in ((0.01, 0.7), (0.1, 0.5)):
        check = analysis.scale_check(z, K640, T[:3, :3], 0.57, "sensor", sigma)
        assert check["model_height_m"] == pytest.approx(h, abs=0.005)
        floor = analysis.own_floor_z(z, K640, T, 0.57, "sensor", sigma)
        assert floor == pytest.approx(float(T[2, 3]) - check["model_height_m"], abs=1e-4)


def test_a_frame_with_slivers_only_has_no_floor() -> None:
    """A wall and two ledges of 1-2 % of the frame: no measurable floor plane, so no scale from
    the camera's height and an empty ``scale_check`` (before: the ledge nearest the height)."""
    z, T = _view(LEDGES, 0.9, 12.0)
    R = T[:3, :3]
    assert an.own_floor(z, K640, R, 0.37, 0.06) is None
    assert an.own_floor(z, K640, R, None, 0.0) is None
    assert analysis.height_factor(z, K640, R, 0.37) is None
    check = analysis.scale_check(z, K640, R, 0.37, "monocular")
    assert check["ratio"] is None and check["floor_inlier_share"] == 0.0
    assert an.floor_verdict("mono", 0.37, 0.03, T, 0.05, z, K640) == an.FloorCheck()


# -- the map's floor: a robot on a raised surface --------------------------------------------------


def _plane(rng: np.random.Generator, x: tuple[float, float], y: tuple[float, float], z: float
           ) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(*x, 0.02), np.arange(*y, 0.02))
    return np.column_stack([xs.ravel(), ys.ravel(), z + rng.normal(0, 0.002, xs.size)])


def _raised(tmp_path: Path, top_z: float) -> AnchoredMap:
    """A map whose frame priors fixed: a 0.5 m worktop at ``top_z`` seen from a base camera 0.5 m
    and a wrist camera 0.3 m above z 0, counters 0.2 m below the top with 30 times its area, the
    room's floor 1.1 m below it."""
    rng = np.random.default_rng(1)
    m = AnchoredMap(tmp_path / f"m{top_z}")
    m.created_with = "metric"
    for i, c in enumerate(((0.0, 0.0, 0.5), (0.2, 0.0, 0.3))):
        T = np.eye(4)
        T[:3, 3] = c
        m.keyframes.append(an.Keyframe(f"k{i}", "a", "cam", K640, T, 0.02, 1.0, "fused", None,
                                       "sensor"))
    pts = np.vstack([_plane(rng, (-0.25, 0.25), (-0.25, 0.25), top_z),
                     _plane(rng, (-3.0, 3.0), (-2.5, -1.0), top_z - 0.2),
                     _plane(rng, (1.0, 3.0), (-1.0, 2.5), top_z - 0.2),
                     _plane(rng, (-3.0, 1.0), (1.0, 2.5), top_z - 1.1)])
    m.cloud.add(pts, np.zeros((len(pts), 3)))
    return m


def test_the_map_s_floor_is_the_plane_nearest_z_0_whatever_its_size(tmp_path: Path) -> None:
    """map.md §6, priors fixed the frame: the large horizontal plane nearest z 0, also when
    another one holds 30 times its points (before: no candidate under half the best one's
    points, so the counters, −0.2 m). An arm whose origin is 0.15 m above its worktop still gets
    the worktop (before: the counters, −0.35 m)."""
    m = _raised(tmp_path, 0.0)
    assert m._floor_estimate()[0] == 0.0
    m.begin_job()
    assert (m.floor_z, m.floor_z_sigma) == (0.0, 0.0)
    assert m.height_gate(0.3, 0.01, m.keyframes[1].T, 0.02)  # the wrist frame's sent height
    assert _raised(tmp_path, -0.15)._floor_estimate()[0] == pytest.approx(-0.15, abs=0.005)


def test_a_surface_seen_only_past_the_edge_is_not_the_floor(tmp_path: Path) -> None:
    """map.md §6 "Other planes, such as the room floor seen past a worktop's edge, are support
    surfaces": the floor is looked for under the robot, within ``FLOOR_NEAR_M`` of its cameras.
    A base camera that sees only the counters 1.5 m away and 0.2 m lower keeps z 0 (before:
    the counters, the plane nearest z 0, −0.2 m); a ramp under the robot with no plane is its
    lowest dense layer."""
    rng = np.random.default_rng(2)
    m = _raised(tmp_path, 0.0)
    far = np.vstack([_plane(rng, (-3.0, 3.0), (-2.5, -1.5), -0.2),
                     _plane(rng, (1.5, 3.0), (-1.5, 2.5), -0.2)])
    m.cloud = an.Cloud()
    m.cloud.add(far, np.zeros((len(far), 3)))
    assert m._floor_estimate()[0] == 0.0
    xs, ys = np.meshgrid(np.arange(-0.6, 0.6, 0.02), np.arange(-0.6, 0.6, 0.02))
    ramp = np.column_stack([xs.ravel(), ys.ravel(), -0.5 + 0.5 * xs.ravel()])  # 27 deg
    m.cloud.add(ramp, np.zeros((len(ramp), 3)))
    z = m._floor_estimate()[0]
    assert z is not None and -0.8 < z < -0.2


# the room of the engine test: the robot on a 0.9 x 0.8 m worktop 1.1 m up, counters 0.9 m high
# along the four walls
test_round14.ROOMS["worktop"] = Room(boxes=[((-0.45, -0.4, 0.0), (0.45, 0.4, 1.1)),
                                            ((1.5, -2.5, 0.0), (3.0, 2.5, 0.9)),
                                            ((-3.0, -2.5, 0.0), (-1.5, 2.5, 0.9)),
                                            ((-1.5, 1.4, 0.0), (1.5, 2.5, 0.9)),
                                            ((-1.5, -2.5, 0.0), (1.5, -1.4, 0.9))])
UP = 1.1  # the worktop's height in the room: the map's z 0 (the odometry's) is its top


def _on_top(name: str, T: np.ndarray, height: float | None) -> an.FrameIn:
    P = T.copy()
    P[2, 3] -= UP
    return frame(name, T, p=prior(P), height=height, hs=0.01, room="worktop")


def test_a_robot_on_a_worktop_keeps_its_floor_and_its_wrist_frames(tmp_path: Path) -> None:
    """The raised-surface scene through the engine (depth images, metric priors with the
    worktop at z 0): a base camera 0.5 m up turning in place sees the counters widely, a wrist
    camera 0.35 m up (sent as its height) sees the top. The floor is z 0, and a wrist frame of a
    later update passes the height gate and gives content (before: floor −0.2 m, the frame
    0.2 m off its sent height, left out, so nothing on the top reached the map)."""
    m = AnchoredMap(tmp_path / "m")
    base = [_on_top(f"b{i}", base_pose(yaw, height=UP + 0.5, pitch_deg=0.0), None)
            for i, yaw in enumerate((0.0, 30.0, 60.0, 90.0))]
    wrist = [_on_top(f"w{i}", look(yaw, (0.0, 0.0, UP + 0.35), -40.0), 0.35)
             for i, yaw in enumerate((15.0, 45.0, 75.0))]
    pls = m.update([*base, *wrist])
    assert [p.status for p in pls] == ["registered"] * 7
    assert m.floor()[0] == 0.0
    assert m.height_inconsistent == [] and m.floor_inconsistent == []
    m.save()
    m = AnchoredMap.load(m.root)
    later = m.update([_on_top("w9", look(100.0, (0.0, 0.0, UP + 0.35), -40.0), 0.35)])[0]
    assert later.status == "registered" and later.keyframe is not None
    assert m.height_inconsistent == [] and m.floor_inconsistent == []
    kf = m.keyframe(later.keyframe)
    assert kf is not None and kf.height_ok and kf.contributes() and kf.detects()
    assert m.floor()[0] == 0.0


# -- the raised-surface scene through the service: objects on the worktop ----------------------

# the robot on a 0.9 x 0.8 m worktop 1.1 m up (the priors' z 0), counters 0.9 m high along the
# walls; a mug and a box on the worktop 0.2-0.4 m from the base, a kettle and a pot on the
# counters
WORKTOP = TableScene(
    items=[Item("mug", [((0.25, 0.1, UP), (0.35, 0.2, UP + 0.12))], (200, 40, 40)),
           Item("box", [((0.2, -0.3, UP), (0.38, -0.15, UP + 0.1))], (40, 60, 200)),
           Item("kettle", [((2.0, -0.2, 0.9), (2.25, 0.05, 1.15))], (60, 190, 50)),
           Item("pot", [((-2.25, 0.3, 0.9), (-1.95, 0.6, 1.05))], (200, 200, 40))],
    room=test_round14.ROOMS["worktop"])
TABLE_CAM = {"fx": K_TABLE.fx, "fy": K_TABLE.fy, "cx": K_TABLE.cx, "cy": K_TABLE.cy,
             "width": K_TABLE.width, "height": K_TABLE.height}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Client]:
    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend()
    timings = Timings()
    updater = Updater(store, backend, timings)
    jobs = Jobs(store.jobs, updater, timings.predict_update)
    jobs.start()
    derived = Derived(store, backend.read)
    derived.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", dry_run=updater.dry_run, derived=derived)
    with TestClient(create_app(service)) as http:
        yield Client(http, backend, service)
    jobs.stop(timeout=10)
    derived.stop()


def _worktop_frames(c: Client, tag: str, poses: list[np.ndarray], height: float | None
                    ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Depth-image frames of ``WORKTOP`` with metric priors in the odometry's frame (its z 0 the
    worktop) and their detections; ``height``: the sent camera height above the worktop."""
    out, files = [], {}
    for i, T in enumerate(poses):
        rgb, z, idx = WORKTOP.render(T)
        img = jpeg(rgb, 95)
        c.backend.detections[hashlib.sha256(img).hexdigest()] = detections(WORKTOP, idx)
        P = T.copy()
        P[2, 3] -= UP
        name = f"{tag}{i}"
        files[name], files[name + "d"] = img, png_depth(z)
        f: dict[str, Any] = {"name": name, "image": name, "depth": name + "d", "camera_id": "cam",
                             "prior": {"trust": "metric", "T_map_cam": P.tolist(),
                                       "sigma_pos_m": 0.02, "sigma_rot_deg": 1.0}}
        if height is not None:
            f["height_above_floor_m"], f["height_sigma_m"] = height, 0.01
        out.append(f)
    return out, files


def test_the_objects_on_the_worktop_reach_the_map(client: Client) -> None:
    """The client's raised-surface case through the API: the base camera 0.5 m above the
    worktop, level, looks around first and sees only the counters 0.2 m lower (the worktop is
    below its view); then the wrist camera 0.35 m above it (its sent height) looks at the
    worktop. ``floor_z_m`` is the worktop's z 0 in both versions, the wrist frames pass the
    height gate, the mug and the box on the worktop are objects of the map beside the counters'
    kettle and pot, and the grid observes the floor on the worktop and marks the counters as
    drops (oms/2.9). Before: the counters were the floor (−0.2 m, the plane nearest z 0 among
    those seen), the wrist frames 0.2 m off their sent height above it were left out (two even
    rejected as prior conflicts), and nothing on the worktop reached the map."""
    base = [look(yaw, (0.0, 0.0, UP + 0.5), 0.0) for yaw in range(0, 360, 45)]
    wrist = [look(yaw, (0.05, 0.0, UP + 0.35), -45.0) for yaw in (-60.0, -30.0, 0.0, 30.0, 60.0)]
    for tag, poses, height in (("b", base, None), ("w", wrist, 0.35)):
        frames, files = _worktop_frames(client, tag, poses, height)
        job = client.update("top", frames, files, cameras={"cam": TABLE_CAM})
        assert job["state"] == "committed", job
        assert {i["status"] for i in job["inputs"]} == {"registered"}
        assert job["quality"]["height_inconsistent"] == []
        assert client.get("/api/maps/top", "MapSummary")["floor_z_m"] == 0.0
    doc = client.get("/api/maps/top/objects", "Objects")
    assert sorted(o["label"] for o in doc["objects"]) == ["box", "kettle", "mug", "pot"]
    g = client.get("/api/maps/top/grid?z_max=1.0", "Grid")  # below the ceiling
    floor = decode(g["layers"]["floor_observed"])

    def at(x: float, y: float) -> int:
        return int(floor[int((y - g["origin_m"][1]) / g["cell_m"]),
                         int((x - g["origin_m"][0]) / g["cell_m"])])

    assert at(0.3, -0.05) == 1  # the worktop between the mug and the box
    assert at(2.6, -1.0) == 3  # a counter: a drop below the floor band
