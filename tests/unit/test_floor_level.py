"""The map floor of a camera-height map whose cameras are far from the map's origin (map.md §2
"Sent height in registration", §6 ``floor_z_m``; the client's ``dev_ainex_loop2``): a floor
plane is never exactly level in the map frame (a monocular depth's bend, the origin prior's
rotation error), so its height is the height of its points, never its distance to the map's
origin. Before, a floor tilted by 1.8° and seen 3 m from the origin read 0.11 m low: the sent
heights were then measured from that floor, every later camera sagged toward it, the floor band
missed the real floor, and a frame analysed on the map was scaled by its height above it.
Also: the camera heights' own sigma is common to the frames (``scale_sigma_rel``)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from oh_my_slam.mapping import analysis
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import AnchoredMap, FrameIn, ScaleState
from oh_my_slam.service.analyse import Analyser
from oh_my_slam.service.app import Service, create_app
from oh_my_slam.service.derived import Derived
from oh_my_slam.service.jobs import Jobs
from oh_my_slam.service.store import Store
from oh_my_slam.service.timings import Timings
from oh_my_slam.service.update import Updater
from tests.fakes.service import FakeBackend, FakeInference
from tests.synth.room import base_pose
from tests.unit.test_round14 import H, K, frame, prior
from tests.unit.test_service_analyse import An, decode
from tests.unit.test_service_prior_first import CAM
from tests.unit.test_service_prior_first import shot as service_shot

# the priors' frame: its origin is not where the robot is (odometry started elsewhere)
OFF = np.array([10.0, -6.0, 0.0])
TILT_DEG = 1.0  # the origin prior's pitch error: the map's floor tilts by it
BIAS = 1.3  # the monocular model's depth over the true depth, the same for every frame
HS = 0.03  # the sent height's sigma


AT = (-4.0, 0.35)  # where the cameras of the unit maps stand, 4 m from the map's origin


def _tilted_floor(n: int = 6000, tilt_deg: float = 1.8, z0: float = 0.0) -> np.ndarray:
    """Floor points seen around a camera at ``AT``: a plane through ``z0`` under the camera,
    tilted by ``tilt_deg`` about the map's y axis, falling toward the map's origin."""
    rng = np.random.default_rng(3)
    xy = np.asarray(AT) + rng.uniform(-1.5, 1.5, (n, 2))
    z = z0 - math.tan(math.radians(tilt_deg)) * (xy[:, 0] - AT[0]) + rng.normal(0, 0.003, n)
    return np.column_stack([xy, z])


def test_a_plane_s_level_is_the_height_of_its_points() -> None:
    """``horizontal_plane``: the distance to the origin (a camera's height above its own floor)
    and the level (how far the plane's points lie below z 0), which differ as soon as the plane
    tilts and the origin is not among its points."""
    floor = _tilted_floor()
    q = floor - np.array([0.0, 0.0, 0.37])  # below a camera 0.37 m up, the origin 4 m away
    got = an.horizontal_plane(q, 0.37)
    assert got is not None
    normal, dist, n, level, _ = got
    assert n > 1000 and math.degrees(math.acos(normal[2])) == pytest.approx(1.8, abs=0.3)
    assert level == pytest.approx(0.37, abs=0.04)  # where it was seen: within its own heights
    assert dist == pytest.approx(0.37 + 4.0 * math.tan(math.radians(1.8)), abs=0.02)  # at 0, 0
    # every slice of it offered as a candidate: at its own height, or at the origin's
    levels, dists = an.floor_candidates(q, level=True), an.floor_candidates(q)
    assert levels and all(abs(h - 0.37) < 0.05 for h in levels)
    assert len(dists) == len(levels) and all(abs(h - dist) < 0.02 for h in dists)
    assert [h for h, _ in an.floor_planes(q, level=True)] == levels
    # a camera's own points: the two agree (the camera is above its floor)
    own = floor - np.array([*AT, 0.37])
    d, _, lv, _ = an.horizontal_plane(own, 0.37)[1:]  # type: ignore[index]
    assert d == pytest.approx(0.37, abs=0.005) and lv == pytest.approx(0.37, abs=0.04)


def _map_with(tmp_path: Path, cloud: np.ndarray, created: str, height: float) -> AnchoredMap:
    """A map created with ``created`` priors: one confirmed camera ``height`` up at ``AT`` and
    its cloud."""
    m = AnchoredMap(tmp_path / created)
    m.created_with = created
    T = np.eye(4)
    T[:3, 3] = (*AT, height)
    m.keyframes = [an.Keyframe("k000000", "a", "cam", K, T, 0.05, 1.0, "fused", None, "mono")]
    m.cloud.add(cloud, np.zeros((len(cloud), 3)))
    return m


@pytest.mark.parametrize("created", ["drifting", "metric"])
def test_a_tilted_floor_far_from_the_origin_is_the_floor(tmp_path: Path, created: str) -> None:
    """Priors fixed the frame, the cameras stand 4 m from its origin, and the floor they see
    tilts by 1.8° (the client's first frame): it is the floor at z 0 (within the floor band),
    not a plane 0.13 m lower, and a real surface 0.15 m up (an arm's worktop) is still found at
    its own height."""
    m = _map_with(tmp_path, _tilted_floor(), created, 0.37)
    z, sigma = m._floor_estimate()
    assert z == 0.0 and sigma < 0.05
    m.begin_job()
    assert (m.floor_z, m.floor_z_sigma, m.floor_z_rel) == (0.0, 0.0, 0.0)
    f = FrameIn("b", "cam", np.zeros((K.height, K.width, 3), np.uint8), K, None, None, None,
                0.37, 0.03)
    assert m._height_z(f) == (pytest.approx(0.37), pytest.approx(0.03))
    up = _map_with(tmp_path / "up", _tilted_floor(z0=0.15), created, 0.52)
    assert up._floor_estimate()[0] == pytest.approx(0.15, abs=0.05)  # (0.0 before)


def _prior(T: np.ndarray, pitch_deg: float = 0.0) -> dict[str, object]:
    """A drifting prior of the true pose ``T`` in the priors' frame (``OFF``), its pitch off by
    ``pitch_deg``; its sigmas the client's (0.05 m, 5°, tilt 10°)."""
    P = T.copy()
    P[:3, :3] = T[:3, :3] @ an.exp_so3(np.array([math.radians(pitch_deg), 0.0, 0.0]))
    P[:3, 3] += OFF
    return prior(P, "drifting", 0.05, 5.0, 10.0)


def _mono(name: str, yaw: float, pitch_deg: float = 0.0) -> FrameIn:
    T = base_pose(yaw)
    return frame(name, T, p=_prior(T, pitch_deg), depth=None, mono=BIAS, hs=HS)


def _loop(tmp_path: Path) -> tuple[AnchoredMap, list[float], list[float]]:
    """The client's session on the synthetic room: a monocular camera 0.5 m up, its model depth
    1.3 × the truth, one frame per update while it turns, drifting priors in a frame whose
    origin is 11.7 m away, the origin frame's prior pitched by 1°. Returns the map, each
    version's floor and each frame's committed camera height above it."""
    floors, above = [], []
    for i, yaw in enumerate((0.0, 20.0, -20.0, 40.0, -40.0)):
        m = AnchoredMap.load(tmp_path / "m") if i else AnchoredMap(tmp_path / "m")
        pl = m.update([_mono(f"f{i}", yaw, TILT_DEG if i == 0 else 0.0)])[0]
        assert pl.status == "registered" and pl.T is not None
        z, _ = m.floor()
        assert z is not None
        m.save()
        floors.append(z)
        above.append(float(pl.T[2, 3]) - z)
    return m, floors, above


def test_cameras_stay_at_their_sent_height_above_the_floor(tmp_path: Path) -> None:
    """The regression of ``dev_ainex_loop2``: every version's floor is z 0, every committed
    camera lies at its sent height above it within the height's sigma, nothing is left out, and
    each frame's content takes the scale its own floor gives (1 / 1.3). Before, the first
    version's floor read 0.18 m low (the tilted plane at the map's origin), the following frames
    were measured from it and sagged."""
    m, floors, above = _loop(tmp_path)
    assert floors == [0.0] * 5
    assert all(abs(h - H) <= HS for h in above), above
    assert m.height_inconsistent == [] and m.floor_inconsistent == []
    assert not any(k.floor_unjudged for k in m.keyframes)
    for k in m.keyframes:
        assert k.depth_scale == pytest.approx(1.0 / BIAS, rel=0.03)
    # the scale: the camera heights', its sigma never below the sent height's own
    source, sigma = m.scale_summary()
    assert source == "camera_height" and sigma >= HS / H - 1e-9
    assert m.scale.k == pytest.approx(1.0 / BIAS, rel=0.03)


def test_a_frame_analysed_on_the_map_takes_the_scale_of_its_sent_height(tmp_path: Path) -> None:
    """Frame analysis in the map frame (api.md §7; ``service.analyse``): a monocular depth
    brought onto the map by the camera's height above the map's floor has the scale its sent
    height gives outside the map, within the height's sigma. Before, the floor 0.18 m low made
    it 1.3 × too large."""
    m = AnchoredMap.load(_loop(tmp_path)[0].root)
    floor = m.floor()[0]
    assert floor is not None
    k = m.keyframes[-1]
    raw = m.depth_of(k.id)
    assert raw is not None and k.K_grid is not None and k.grid_scale is not None
    raw = (raw / (k.grid_scale * m._depth_factor(k))).astype(np.float32)
    inline = analysis.height_factor(raw, k.K_grid, k.T[:3, :3], H)
    on_map = analysis.height_factor(raw, k.K_grid, k.T[:3, :3], float(k.T[2, 3]) - floor)
    assert inline == pytest.approx(1.0 / BIAS, rel=0.02)
    assert on_map is not None and abs(on_map / inline - 1.0) <= HS / H
    # and against the map's cloud as the pose sees it
    cloud = analysis.map_factor(raw, k.K_grid, k.T, m.cloud.points())
    assert cloud is not None and abs(cloud / inline - 1.0) <= HS / H


def test_the_sent_heights_own_sigma_is_common_to_the_frames() -> None:
    """``scale_sigma_rel`` of camera heights (map.md §1: any horizontal distance within 3
    sigma): a height sent with every frame is one measurement, so nine frames that agree leave
    its relative sigma as it is; their spread about the median is averaged, the spread between
    viewing directions is not."""
    s = ScaleState(heights=[[0.08, 0.75]] * 9, height_yaws=[40.0 * i for i in range(9)])
    s.estimate(0.0)
    assert s.source == "camera_height" and s.k == pytest.approx(0.75)
    assert s.sigma_rel == pytest.approx(0.08)
    one = ScaleState(heights=[[0.08, 0.75]], height_yaws=[0.0])
    one.estimate(0.0)
    assert one.sigma_rel == pytest.approx(0.08)
    # a spread between the frames adds to it, over the root of their count
    spread = ScaleState(heights=[[0.08, v] for v in (0.70, 0.75, 0.80, 0.75)],
                        height_yaws=[0.0, 1.0, 2.0, 3.0])
    spread.estimate(0.0)
    lr = np.log([0.70, 0.75, 0.80, 0.75])
    mad = 1.4826 * float(np.median(np.abs(lr - np.median(lr))))
    assert mad > 0 and spread.sigma_rel == pytest.approx(math.hypot(0.08, mad / 2.0))


# -- through the service: the client's session and its analysis ------------------------------------


@pytest.fixture
def svc(tmp_path: Path) -> Iterator[An]:
    """A service with an analyser whose monocular model answers ``BIAS`` × the true depth."""
    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend()
    backend.mono_scale = BIAS
    timings = Timings()
    updater = Updater(store, backend, timings)
    jobs = Jobs(store.jobs, updater, timings.predict_update)
    jobs.start()
    derived = Derived(store, backend.read)
    derived.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test", dry_run=updater.dry_run, derived=derived,
                      analyser=Analyser(store, backend))
    with TestClient(create_app(service)) as http:
        yield An(http, backend, service)
    jobs.stop(timeout=10)
    derived.stop()


def _sent(c: An, name: str, yaw: float, pitch_deg: float = 0.0
          ) -> tuple[dict[str, Any], dict[str, bytes], np.ndarray, np.ndarray]:
    """A monocular frame of the room with its drifting prior (``_prior``): the request's frame,
    its files, its true depth and its true pose in the priors' frame."""
    T = base_pose(yaw)
    img, _, z = service_shot(T)
    c.backend.mono_depth[hashlib.sha256(img).hexdigest()] = z
    P = T.copy()
    P[:3, 3] += OFF
    return ({"name": name, "image": name, "camera_id": "cam", "prior": _prior(T, pitch_deg)},
            {name: img}, z, P)


def test_the_service_keeps_the_floor_and_analyses_at_the_sent_height(
        svc: An, monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's report end to end: one monocular frame per update with a camera height and
    drifting priors far from their origin. The map's floor stays z 0 and its cameras at the
    sent height above it; ``scale_sigma_rel`` is at least the height's own; and a view analysed
    on the map (by its camera's height, by the map's cloud, or by its height above the map's
    floor) has the scale of the same view analysed without the map, within that sigma."""
    cam = {**CAM, "height_above_floor_m": H, "height_sigma_m": HS}
    sigma = HS / H
    for i, yaw in enumerate((0.0, 20.0, -20.0, 40.0)):
        f, files, _, _ = _sent(svc, f"f{i}", yaw, TILT_DEG if i == 0 else 0.0)
        job = svc.update("m", [f], files, cameras={"cam": cam} if i == 0 else None)
        assert job["state"] == "committed", job
        assert job["inputs"][0]["status"] == "registered"
        assert job["quality"]["height_inconsistent"] == []
        assert svc.get("/api/maps/m", "MapSummary")["floor_z_m"] == 0.0
        assert abs(job["inputs"][0]["T_map_cam"][2][3] - H) <= HS
        assert job["scale_source"] == "camera_height"
        assert job["scale_sigma_rel"] >= round(sigma, 5)
    # a view the map does not hold, at the pose the client's odometry gives it
    _, files, z, P = _sent(svc, "q", 10.0)
    img = files["q"]
    svc.backend.up = base_pose(10.0)[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    alone = svc.analyse({"image": "i", "camera": cam, "outputs": ["depth"]}, {"i": img})
    assert (alone["frame"], alone["scale_source"]) == ("gravity_camera", "camera_height")
    d0 = decode(alone["depth"]) * 0.001
    assert np.median(d0 / z) == pytest.approx(1.0, abs=0.02)

    def on_map(camera: dict[str, Any]) -> float:
        doc = svc.analyse({"image": "i", "camera": camera, "map": "m", "T_map_cam": P.tolist(),
                           "outputs": ["depth", "range_scan"]}, {"i": img})
        assert (doc["frame"], doc["scale_source"]) == ("map", "camera_height")
        assert doc["scale_sigma_rel"] >= round(sigma, 5)
        return float(np.median(decode(doc["depth"]) * 0.001 / d0))

    # by the camera's sent height: the scale of the analysis without the map
    assert on_map(cam) == pytest.approx(1.0, abs=0.01)
    # a camera that gives no height: by the map's cloud as the pose sees it ...
    assert abs(on_map(CAM) - 1.0) <= sigma
    # ... and where the pose sees too little of it, by its height above the map's floor
    monkeypatch.setattr(analysis, "MAP_ALIGN_MIN_POINTS", 10 ** 9)
    assert abs(on_map(CAM) - 1.0) <= sigma
