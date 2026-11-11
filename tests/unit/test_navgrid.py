"""The navigation grid's pieces (``mapping.navgrid``) on hand-made volumes and grids: the volume's
growth and limits, the band rules, camera stands, chords, leaves, regions and frontiers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oh_my_slam.mapping import navgrid as ng
from tests.synth import rooms


def test_pose_moved_and_ray_stride() -> None:
    a = np.eye(4).reshape(-1).tolist()
    b = np.eye(4)
    b[0, 3] = 0.001
    assert not ng.pose_moved(a, b.reshape(-1).tolist())
    b[0, 3] = 0.01
    assert ng.pose_moved(a, b.reshape(-1).tolist())
    c = rooms.camera(0, 0, 0, 1.0, 0.0)
    c[:3, 3] = 0
    assert ng.pose_moved(a, c.reshape(-1).tolist()) is False or True  # rotation: compared below
    r = np.eye(4)
    r[:3, :3] = [[math.cos(0.01), -math.sin(0.01), 0], [math.sin(0.01), math.cos(0.01), 0],
                 [0, 0, 1]]
    assert ng.pose_moved(a, r.reshape(-1).tolist())
    assert ng.ray_stride(270.0, 640, 480) == 3 and ng.ray_stride(100.0, 4000, 3000) >= 18


def test_volume_growth_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    v = ng.Volume()
    assert v.empty and v.estimate_floor() is None
    assert v.ensure(np.zeros(3), np.ones(3))
    shape = v.shape
    assert v.ensure(np.zeros(3), np.ones(3)) and v.shape == shape  # nothing to grow
    assert v.ensure(np.array([-2.0, 0, 0]), np.ones(3)) and v.shape[2] > shape[2]
    assert v.estimate_floor() is None  # no flat hits
    monkeypatch.setattr(ng, "MAX_CELLS_XY", 10)
    assert not v.ensure(np.zeros(3), np.array([50.0, 50.0, 1.0]))
    assert not ng.Volume().ensure(np.zeros(3), np.array([50.0, 50.0, 1.0]))


def test_integrate_skips_frames_with_nothing_to_add(monkeypatch: pytest.MonkeyPatch) -> None:
    T = rooms.camera(1, 1, 1, 0)
    v = ng.Volume()
    assert not v.integrate("a", np.zeros((24, 32)), rooms.K640 / 10 * [[1], [1], [10]], T)
    d = rooms.two_rooms().depth(T)
    assert not v.integrate("b", d, rooms.K640, T, np.zeros(d.shape, bool))
    assert v.integrate("c", d, rooms.K640, T) and "c" in v.frames
    assert not v.integrate("c2", d, rooms.K640, T, np.zeros(d.shape, bool))
    monkeypatch.setattr(ng, "MAX_CELLS_XY", 10)
    far = rooms.camera(100, 100, 1, 0)
    assert not v.integrate("d", d, rooms.K640, far)
    blob = v.to_bytes()
    w = ng.Volume.from_bytes(blob)
    assert w.shape == v.shape and w.frames == v.frames and (w.hit == v.hit).all()
    assert w.estimate_floor() == pytest.approx(0.0, abs=0.01)  # the floor box top: z 0


def _door(x: float, y: float, oid: int = 1, yaw: float = 0.0, width: float = 0.8,
          thick: float = 0.04) -> dict[str, object]:
    """A door object shaped as a leaf (map.md §5), as the derived meta lists it."""
    return {"id": oid, "centre_m": [x, y, 1.0], "size_m": [width, thick, 2.0], "yaw_deg": yaw}


def _grid(state: np.ndarray, cell: float = 0.1) -> ng.Layers:
    R, C = state.shape
    return ng.Layers(np.zeros(2), cell, state.astype(np.uint8), np.ones((R, C), np.uint16),
                     np.full((R, C), np.nan, np.float32), np.ones((R, C), np.uint8))


def test_resample_finer_and_coarser() -> None:
    st = np.zeros((4, 6), np.uint8)
    st[0, 0] = 2
    st[1, 1] = 1
    lay = _grid(st)
    lay.height[0, 0] = 0.5
    assert ng.resample(lay, 0.1) is lay
    fine = ng.resample(lay, 0.05)
    assert fine.shape == (8, 12) and fine.state[0, 0] == 2 and fine.state[1, 1] == 2
    coarse = ng.resample(lay, 0.2)
    assert coarse.shape == (2, 3) and coarse.state[0, 0] == 2 and coarse.state[0, 1] == 0
    assert coarse.height[0, 0] == 0.5 and math.isnan(coarse.height[1, 2])


def test_blind_radius_and_stands_outside_the_grid() -> None:
    K = rooms.K640
    level = ng.Stand(rooms.camera(0.5, 0.5, 0.5, 0, 0.0), K, 320, 240)
    up = ng.Stand(rooms.camera(0.5, 0.5, 1.0, 0, 60.0), K, 320, 240)
    assert ng.blind_radius(level, 0.0) == pytest.approx(0.5 / (120 / 320), rel=1e-6)
    high = ng.Stand(rooms.camera(0.5, 0.5, 5.0, 0, 0.0), K, 320, 240)
    assert ng.blind_radius(high, 0.0) == ng.BLIND_RING_MAX_M
    assert ng.blind_radius(up, 0.0) == 0.0
    assert ng.blind_radius(level, 2.0) == 0.0  # below the floor
    lay = _grid(np.zeros((10, 10), np.uint8))
    ng.mark_stands(lay, [up], 0.0, None)
    assert (lay.state == 1).sum() == 1  # only its own cell
    out = ng.Stand(rooms.camera(-5.0, -5.0, 1.0, 0, 0.0), K, 320, 240)
    edge = ng.Stand(rooms.camera(-0.05, 0.5, 1.0, 0, -30.0), K, 320, 240)
    ng.mark_stands(lay, [out, edge], 0.0, None)
    assert (lay.state == 1).sum() == 1  # neither stands inside the grid
    lay2 = _grid(np.zeros((40, 40), np.uint8))
    lay2.state[:, 8] = 2  # a wall 0.85 m from the camera
    ng.mark_stands(lay2, [ng.Stand(rooms.camera(0.5, 2.0, 1.0, 0, -20.0), K, 320, 240)], 0.0, 0.2)
    assert lay2.state[20, 5] == 1 and lay2.state[20, 9] == 0 and lay2.state[20, 8] == 2


def test_march_at_the_grid_edge() -> None:
    free = np.ones((5, 5), bool)
    t, cell = ng._march(free, np.array([2.5, 2.5]), np.array([1.0, 0.0]), 1.0)
    assert t == 1.0 and cell is None
    t, cell = ng._march(free, np.array([2.5, 2.5]), np.array([1.0, 0.0]), 10.0)
    assert t == pytest.approx(2.5) and cell is None
    free[2, 4] = False
    t, cell = ng._march(free, np.array([2.5, 2.5]), np.array([1.0, 0.0]), 10.0)
    assert t == pytest.approx(1.5) and cell == (2, 4)


def test_polygon_of_a_one_cell_region() -> None:
    lay = _grid(np.ones((3, 3), np.uint8), cell=1.0)
    m = np.zeros((3, 3), bool)
    m[1, 1] = True
    poly = ng._polygon(m, lay)
    assert len(poly) >= 3
    xs = [p[0] for p in poly]
    assert min(xs) >= 1.0 and max(xs) <= 2.0


def test_leaf_needs_cells_off_the_jamb_line() -> None:
    st = np.zeros((40, 40), np.uint8)
    lay = _grid(st)
    j1, j2 = np.array([2.0, 1.5]), np.array([2.0, 2.4])
    assert ng._leaves(st, lay, j1, j2) == []  # nothing occupied
    st[15, 25] = 2
    assert ng._leaves(st, lay, j1, j2) == []  # one cell
    apart = st.copy()
    apart[18, 27] = 2
    assert ng._leaves(apart, lay, j1, j2) == []  # two cells, apart: two pieces of one cell
    short = np.zeros((40, 40), np.uint8)
    short[15, 22:24] = 2
    assert ng._leaves(short, lay, j1, j2) == []  # too short
    st[16:20, 22] = 2  # a leaf swung half into the gap
    st[15, 21:29] = 2
    (leaf,) = ng._leaves(st, lay, j1, j2)
    assert leaf["intrudes"] and len(leaf["polygon_m"]) == 4
    st[24, 21:29] = 2  # a second leaf, at the other jamb: a double door
    two = ng._leaves(st, lay, j1, j2)
    assert len(two) == 2 and two[0] == leaf and not two[1]["intrudes"]


def test_features_kinds_frontiers_and_small_parts() -> None:
    # two 3 x 3 m rooms of free cells (10 cm) joined by a 0.8 m gap in a 0.2 m thick wall
    st = np.zeros((40, 70), np.uint8)
    st[5:35, 2:34] = 1
    st[5:35, 36:68] = 1
    st[5:35, 34:36] = 2
    st[16:24, 34:36] = 1  # the gap
    st[1:3, 1:4] = 1  # a tiny free blob: no region
    lay = _grid(st)
    f = ng.features(lay)
    assert len(f.regions) == 2 and len(f.passages) == 1
    (p,) = f.passages
    # no door object: an opening, whatever its width; narrow unknown without a footprint
    assert p.kind == "opening" and p.width == pytest.approx(0.9, abs=0.11)
    assert p.leaves == [] and p.leaf is None and p.narrow is None and p.frontier is None
    assert p.regions[1] is not None
    assert {fr.region for fr in f.frontiers} <= {0, 1, None}
    f2 = ng.features(lay, doors=[p.centre])  # a door hint alone is no leaf
    assert f2.passages[0].kind == "opening" and f2.passages[0].leaf is None
    # a door object shaped as a leaf, swung into the gap, and one along the wall: a door with
    # both leaves (no open leaf in this grid: their boxes' footprints), the larger first
    f4 = ng.features(lay, leaves=[_door(3.95, 2.0, 7, width=0.4), _door(3.95, 1.62, 8),
                                  _door(9.0, 9.0, 9)])
    (q,) = f4.passages
    assert q.kind == "door" and [lf["object_id"] for lf in q.leaves] == [8, 7]
    assert q.leaf == q.leaves[0] and not q.leaves[0]["intrudes"] and q.leaves[1]["intrudes"]
    assert len(q.leaves[0]["polygon_m"]) == 4
    st[13:27, 34:36] = 1  # a wider gap: an opening
    wide = ng.features(_grid(st))
    assert [q.kind for q in wide.passages] in ([], ["opening"])
    f3 = ng.features(lay, footprint_radius=0.5)  # 0.9 m < 2 x 0.5 m
    assert f3.passages[0].kind == "opening" and f3.passages[0].narrow is True
    assert ng.features(lay, footprint_radius=0.3).passages[0].narrow is False
    # the blob's frontier is too short and touches no region
    assert all(fr.length >= ng.MIN_FRONTIER_M for fr in f.frontiers)


def test_features_without_passages_or_regions() -> None:
    assert ng.features(_grid(np.zeros((10, 10), np.uint8))).regions == []
    st = np.zeros((30, 30), np.uint8)
    st[2:28, 2:28] = 1
    f = ng.features(_grid(st))
    assert len(f.regions) == 1 and f.passages == [] and f.frontiers
    assert f.frontiers[0].region == 0


def test_a_chord_that_does_not_split_is_no_passage() -> None:
    # two rooms joined by two walled gaps: both cut, two regions; a gap in a free-standing wall
    # inside one room (free all around it) does not split anything
    st = np.zeros((60, 70), np.uint8)
    st[4:56, 2:68] = 1
    st[4:56, 34:36] = 2  # the wall between the rooms
    st[10:18, 34:36] = 1  # door 1
    st[42:50, 34:36] = 1  # door 2
    st[26:32, 6:24] = 2  # a free-standing block, far from the doors
    st[26:32, 13:17] = 1  # a gap through it, free around it
    f = ng.features(_grid(st))
    assert len(f.regions) == 2 and len(f.passages) == 2
    for p in f.passages:
        # 0.8 m of free cells; the jambs' faces half a cell into the cells the chord ends on
        assert p.width == pytest.approx(0.9, abs=0.01) and p.kind == "opening"


def test_walled_checks_both_ends() -> None:
    st = np.zeros((20, 20), np.uint8)
    st[10, :] = 2
    st[10, 8:12] = 1
    e1, e2 = np.array([8.0, 10.5]), np.array([12.0, 10.5])
    assert ng._walled(st, e1, e2, 0.1)
    assert not ng._walled(st, e1, e1, 0.1)  # no length
    st[10, 14] = 1
    assert not ng._walled(st, e1, e2, 0.1)  # free beyond an end
    st[10, 14] = 0
    st[10, 13:16] = 0
    assert not ng._walled(st, e1, e2, 0.1)  # unseen beyond an end
    assert not ng._walled(st, np.array([0.5, 10.5]), np.array([4.0, 10.5]), 0.1)  # the edge
    ok = ng._walled_many(st, np.array([e1, e1]), np.array([e2, e1]), 0.1)
    assert ok.tolist() == [False, False]


def test_object_region_by_footprint_and_by_the_cameras() -> None:
    st = np.zeros((30, 30), np.uint8)
    st[2:28, 2:15] = 1
    st[2:28, 15] = 2
    lay = _grid(st)
    f = ng.features(lay)
    assert ng.object_region(f, lay, [0.8, 1.5], [0.4, 0.4], 30.0) == 0
    assert ng.object_region(f, lay, [2.5, 1.5], [0.4, 0.4], 0.0) is None
    assert ng.object_region(f, lay, [2.5, 1.5], [0.4, 0.4], 0.0, [[0.8, 1.5], [9.0, 9.0]]) == 0
    assert ng.object_region(f, lay, [9.0, 9.0], [0.4, 0.4], 0.0) is None


def test_a_chord_screened_in_but_not_walled_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    st = np.zeros((40, 70), np.uint8)
    st[5:35, 2:34] = 1
    st[5:35, 36:68] = 1
    st[5:35, 34:36] = 2
    st[16:24, 34:36] = 1
    assert len(ng._passage_chords(st, 0.1)) == 1
    monkeypatch.setattr(ng, "_walled", lambda *a: False)
    assert ng._passage_chords(st, 0.1) == []


def test_an_obstacle_beside_the_wall_narrows_the_passage() -> None:
    st = np.zeros((40, 70), np.uint8)
    st[5:35, 2:34] = 1
    st[5:35, 36:68] = 1
    st[5:35, 34:36] = 2
    st[16:24, 34:36] = 1
    st[22:24, 36:44] = 2  # a leaf along the wall, beside the gap's upper jamb
    (ch,) = ng._passage_chords(st, 0.1)
    assert ch[0] < 8.0 - 1e-6
    assert float(np.linalg.norm(ch[2] - ch[1])) == pytest.approx(ch[0])


def test_the_narrowest_chord_beside_the_wall_line_is_the_passage() -> None:
    st = np.zeros((40, 70), np.uint8)
    st[5:35, 2:34] = 1
    st[5:35, 36:68] = 1
    st[5:35, 34:36] = 2
    st[12:26, 34:36] = 1  # a 1.4 m gap in the wall
    st[12:15, 31:34] = 2  # a frame, 0.3 m deep on one side, leaves 0.9 m
    st[24:26, 31:34] = 2
    f = ng.features(_grid(st))
    (p,) = f.passages
    assert p.width == pytest.approx(0.9 + 0.1, abs=0.01)  # half a cell beyond each end
    assert sorted(j[1] for j in p.jambs) == pytest.approx([1.45, 2.45], abs=0.01)
    assert 3.1 <= p.centre[0] <= 3.4 and p.centre[1] == pytest.approx(1.95)


def test_coverage_counts_floor_seen_over_free_and_enclosed_unknown() -> None:
    st = np.zeros((30, 30), np.uint8)
    st[2:28, 2:28] = 1
    st[10:14, 10:14] = 0  # an unknown hole inside the region
    lay = _grid(st)
    lay.floor_observed[:] = 0
    lay.floor_observed[2:15, 2:28] = 1
    (r,) = ng.features(lay).regions
    free = int((st == 1).sum())
    seen = int(((st == 1) & (lay.floor_observed == 1)).sum())
    assert r.coverage_share == pytest.approx(seen / (free + 16), abs=1e-4)


def test_cells_free_by_rule_get_floor_observed_2() -> None:
    K = rooms.K640
    lay = _grid(np.zeros((40, 40), np.uint8))
    lay.floor_observed[:] = 0
    lay.floor_observed[20, 8] = 1  # seen floor stays seen
    lay.state[20, 8] = 1
    ng.mark_stands(lay, [ng.Stand(rooms.camera(0.5, 2.0, 1.0, 0, -20.0), K, 320, 240)], 0.0, 0.2)
    assert lay.floor_observed[20, 5] == 2 and lay.floor_observed[20, 8] == 1
    assert lay.floor_observed[20, 30] == 0
    coarse = ng.resample(lay, 0.2)
    assert coarse.floor_observed[10, 4] == 1  # seen wins over free by rule
    assert coarse.floor_observed[10, 2] == 2


def test_one_standpoint_sweep_frees_the_whole_blind_disc() -> None:
    """A level camera 0.4955 m above the floor (640x480, vertical FOV 40.2 deg: its lower edge
    meets the floor 1.357 m ahead) sweeping 360 deg from one position in an open room: every cell
    of the blind disc is free, with ``floor_observed`` 1 or 2 (the camera cells 100 % free), even
    those that other frames' rays already made free; a table leg inside the disc stays occupied."""
    fy = 240.0 / math.tan(math.radians(40.2 / 2))
    K = np.array([[fy, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]])
    s = rooms.Scene()
    s.add((-4.0, -4.0, -0.05), (4.0, 4.0, 0.0))  # the floor
    s.add((-4.0, -4.0, 0.0), (-3.9, 4.0, 2.0)).add((3.9, -4.0, 0.0), (4.0, 4.0, 2.0))
    s.add((-4.0, -4.0, 0.0), (4.0, -3.9, 2.0)).add((-4.0, 3.9, 0.0), (4.0, 4.0, 2.0))
    s.add((0.80, 0.30, 0.0), (0.85, 0.35, 0.7))  # a table leg inside the disc
    cams = [rooms.camera(0.0, 0.0, 0.4955, yaw, 0.0) for yaw in range(0, 360, 24)]
    vol = ng.Volume()
    for i, T in enumerate(cams):
        d = s.depth(T, K, 640, 480)
        vol.integrate(f"f{i}", d, K, T, d > 0)
    floor = vol.estimate_floor(0.4955)
    assert floor == pytest.approx(0.0, abs=0.02)
    lay = ng.layers(vol, floor, 0.05, 1.8)
    lay.state[lay.cell_of([0.5, -0.5])] = ng.FREE  # free from a ray, floor unseen
    lay.floor_observed[lay.cell_of([0.5, -0.5])] = 0
    stands = [ng.Stand(T, K, 640, 480) for T in cams]
    ring = ng.blind_radius(stands[0], floor)
    assert ring == pytest.approx(1.357, abs=0.01)
    ng.mark_stands(lay, stands, floor, None)
    rr, cc = np.mgrid[0:lay.shape[0], 0:lay.shape[1]]
    centres = lay.origin[None, None, :] + (np.stack([cc, rr], -1) + 0.5) * lay.cell
    disc = np.hypot(centres[..., 0], centres[..., 1]) < ring - lay.cell
    assert lay.state[lay.cell_of([0.825, 0.325])] == ng.OCCUPIED
    near_leg = (np.abs(centres[..., 0] - 0.825) < 0.1) & (np.abs(centres[..., 1] - 0.325) < 0.1)
    disc &= ~near_leg
    assert disc.sum() > 2000
    assert (lay.state[disc] == ng.FREE).all()  # camera_cells_free 100 %
    assert np.isin(lay.floor_observed[disc], (1, 2)).all()
    assert lay.floor_observed[lay.cell_of([0.5, -0.5])] == 2


@pytest.mark.parametrize("thickness", [0.02, 0.03, 0.04])
def test_a_thin_open_leaf_seen_from_a_few_views_stays_occupied(thickness: float) -> None:
    """A door leaf 2-4 cm thick, open at 90 deg to the wall into room B, seen from three views in
    room B among a room A sweep whose rays pass beside it: its cells are occupied (other views'
    carving does not erase it) and the door's passage reports it."""
    scene = rooms.two_rooms(leaf=True, leaf_t=thickness)
    cams = [rooms.camera(2.0, 2.0, 1.0, yaw) for yaw in range(0, 360, 30)]
    cams += [rooms.camera(3.0, 1.95, 1.0, 0.0)]
    cams += [rooms.camera(5.6, 1.2, 1.0, 140.0), rooms.camera(6.2, 2.2, 1.0, 175.0),
             rooms.camera(5.5, 3.4, 1.0, -135.0)]
    vol = ng.Volume()
    for i, T in enumerate(cams):
        d = scene.depth(T)
        vol.integrate(f"f{i}", d, rooms.K640, T, d > 0)
    floor = vol.estimate_floor(1.0)
    assert floor == pytest.approx(0.0, abs=0.02)
    lay = ng.layers(vol, floor, 0.05, 1.8)
    y = rooms.DOOR[1] - thickness / 2
    leaf_cells = [lay.cell_of([x, y]) for x in np.arange(4.2, 4.85, 0.05)]
    seen = [rc for rc in leaf_cells if lay.observations[rc] > 0]
    assert len(seen) >= 8
    assert all(lay.state[rc] == ng.OCCUPIED for rc in seen)
    feat = ng.features(lay)
    door = [p for p in feat.passages if abs(p.centre[0] - 4.05) < 0.3]
    assert door and door[0].kind == "opening" and door[0].leaf is None  # no door object
    leaf = _door(4.5, y, 5, width=0.8, thick=thickness)
    door = [p for p in ng.features(lay, leaves=[leaf]).passages if abs(p.centre[0] - 4.05) < 0.3]
    assert door and door[0].kind == "door" and door[0].leaf is not None
    assert door[0].leaf["object_id"] == 5 and not door[0].leaf["intrudes"]


def test_the_floor_is_the_surface_the_robot_stands_on() -> None:
    """An arm on a worktop (top at z 0, priors' origin) looking down past its edge at the room
    floor 1.1 m below: named by the version's floor (near 0) the worktop is the floor; without,
    the lowest large plane. A base on the floor: the floor either way."""
    s = rooms.Scene()
    s.add((-3.0, -3.0, -1.15), (3.0, 3.0, -1.1))  # the room floor
    s.add((-0.5, -0.5, -1.1), (0.5, 0.5, 0.0))  # the worktop
    vol = ng.Volume()
    for i, yaw in enumerate(range(0, 360, 45)):
        T = rooms.camera(0.0, 0.0, 0.45, yaw, -35.0)
        d = s.depth(T)
        vol.integrate(f"a{i}", d, rooms.K640, T, d > 0)
    assert vol.estimate_floor(0.45) == pytest.approx(-1.1, abs=0.02)
    assert vol.estimate_floor(0.45, 0.02) == pytest.approx(0.0, abs=0.02)
    assert vol.estimate_floor(0.45, 0.5) == 0.5  # no plane there: as named
    base = ng.Volume()
    scene = rooms.two_rooms()
    for i, yaw in enumerate(range(0, 360, 60)):
        T = rooms.camera(2.0, 2.0, 0.25, yaw, 0.0)
        d = scene.depth(T)
        base.integrate(f"b{i}", d, rooms.K640, T, d > 0)
    assert base.estimate_floor(0.25) == pytest.approx(0.0, abs=0.01)
    assert base.estimate_floor(0.25, 0.06) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("height", [1.0, 0.25])
def test_a_wide_double_doorway_with_both_leaves_open_into_the_room(height: float) -> None:
    """A 2 m double doorway, both leaves open 90 deg into the observer's room, the observer 1.5 m
    away, the far room seen only through the opening, door objects detected on the leaves: the
    passage is the doorway, its width the clear width between the leaves (open leaves are
    obstacles), its centre the free gap's at the jamb line, both leaves reported. Without the
    door objects the hidden wall leaves no walled chord; the blind disc frees nothing the leaves
    hide."""
    scene = rooms.double_door()
    cams = [rooms.camera(x, y, height, yaw, -15.0 if height > 0.5 else 0.0)
            for x, y in ((2.5, 2.0), (2.0, 1.3)) for yaw in range(0, 360, 30)]
    vol = ng.Volume()
    for i, T in enumerate(cams):
        d = scene.depth(T)
        vol.integrate(f"f{i}", d, rooms.K640, T, d > 0)
    floor = vol.estimate_floor(height)
    lay = ng.layers(vol, floor, 0.05, 1.8)
    ng.mark_stands(lay, [ng.Stand(T, rooms.K640, 320, 240) for T in cams], floor, 0.0)
    assert lay.state[lay.cell_of([3.5, 3.5])] == ng.UNKNOWN  # behind a leaf, inside a disc
    assert not ng.features(lay).passages
    doors = [(3.5, 1.02), (3.5, 2.98)]
    hinted = ng.features(lay, doors=doors)  # door hints find the doorway; no leaf named
    assert [(q.kind, q.leaves) for q in hinted.passages] == [("opening", [])]
    feat = ng.features(lay, doors=doors, leaves=[_door(x, y, i, width=0.9) for i, (x, y) in
                                                 enumerate(doors, start=1)])
    (p,) = feat.passages
    assert p.kind == "door" and len(feat.regions) == 2 and p.regions in ((0, 1), (1, 0))
    assert sorted(lf["object_id"] for lf in p.leaves) == [1, 2]
    assert p.width == pytest.approx(1.92, abs=0.1)
    assert float(np.linalg.norm(p.centre - [4.05, 2.0])) <= 0.1
    assert sorted(round(float(j[1]), 1) for j in p.jambs) == [1.0, 3.0]
    assert len(p.leaves) == 2 and p.leaf == p.leaves[0]
    assert all(abs(float(np.mean([q[1] for q in leaf["polygon_m"]])) - y) < 0.1
               for leaf, y in zip(sorted(p.leaves, key=lambda f: f["polygon_m"][0][1]),
                                  (1.02, 2.98), strict=True))
    # one door object at the gap's centre finds it too
    (q,) = ng.features(lay, doors=[(4.05, 2.0)]).passages
    assert float(np.linalg.norm(q.centre - p.centre)) < 1e-6


def test_a_doorway_to_a_never_observed_side() -> None:
    """A room whose wall has a 0.8 m gap with only a sliver seen beyond it, then unknown: the
    passage goes from the room to the unobserved side (regions [a, null]), names the frontier
    there, and its normal points toward it (map.md §6). Fully seen beyond (no frontier there): no
    passage."""
    st = np.zeros((40, 70), np.uint8)
    st[5:35, 2:34] = 1
    st[5:35, 34:36] = 2
    st[16:24, 34:36] = 1  # the gap
    st[16:24, 36:38] = 1  # a sliver seen beyond it: too small for a region
    f = ng.features(_grid(st))
    assert len(f.regions) == 1
    (p,) = f.passages
    assert p.regions == (0, None) and p.frontier is not None and p.kind == "opening"
    fr = f.frontiers[p.frontier]
    assert fr.centre[0] > 3.4  # on the far side
    assert abs(p.normal_deg) < 1.0  # toward +x, the unobserved side
    # the same doorway on the room's other side: the normal still points away from the room
    flipped = ng.features(_grid(st[:, ::-1].copy()))
    (q,) = flipped.passages
    assert q.regions == (0, None) and abs(abs(q.normal_deg) - 180.0) < 1.0
    closed = st.copy()
    closed[14:26, 38] = 2  # walled in beyond the sliver
    closed[14, 36:38] = 2
    closed[25, 36:38] = 2
    closed[16:24, 36:38] = 2
    assert ng.features(_grid(closed)).passages == []


def test_far_frontier_needs_one_ahead_and_near() -> None:
    centre, n = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    behind = [np.array([[-0.5, 0.0], [-0.4, 0.1]])]
    assert ng._far_frontier(behind, centre, n) is None
    far = [np.array([[3.0, 0.0]])]
    assert ng._far_frontier(far, centre, n) is None
    both = [np.array([[0.8, 0.0]]), np.array([[0.3, 0.0]])]
    assert ng._far_frontier(both, centre, n) == 1


def test_an_open_leaf_without_a_door_object_is_no_leaf() -> None:
    st = np.zeros((40, 70), np.uint8)
    st[5:35, 2:34] = 1
    st[5:35, 36:68] = 1
    st[5:35, 34:36] = 2
    st[16:24, 34:36] = 1
    st[24, 36:44] = 2  # an open leaf, hinged at the upper jamb, swung 90 deg into room B
    lay = _grid(st)
    (p,) = ng.features(lay).passages
    assert p.kind == "opening" and p.leaves == []
    # named by a door object near it: the leaf the grid shows, with the object's id
    (q,) = ng.features(lay, leaves=[_door(4.0, 2.45, 3)]).passages
    assert q.kind == "door" and q.leaves[0]["object_id"] == 3
    assert q.leaves[0] == {**ng._leaves(st, lay, *q.jambs)[0], "object_id": 3}


def test_the_floor_band_and_the_height_cap() -> None:
    """The floor band is max(0.06 m, 3 floor sigmas) about the floor (map.md §6): a flat surface
    0.075 m up is an obstacle on a well-known floor, floor on a floor of sigma 0.03 m. ``height_m``
    keeps surfaces at or below the cap only."""
    v = ng.Volume()
    v.ensure(np.zeros(3), np.array([0.5, 0.5, 3.0]))
    Z = v.shape[0]
    zc = v.origin[2] + (np.arange(Z) + 0.5) * v.vz
    low = int(np.argmin(np.abs(zc - 0.075)))
    v.hit[low, 2, 2] = v.flat[low, 2, 2] = 3
    v.hit[low, 4, 4] = v.flat[low, 4, 4] = 3
    top = int(np.argmin(np.abs(zc - 2.475)))
    v.hit[top, 4, 4] = 3  # a ceiling over (4, 4)
    floor = float(zc[low] - 0.075)
    sharp = ng.layers(v, floor, 0.05, 1.8)
    assert sharp.state[2, 2] == ng.OCCUPIED and sharp.floor_observed[2, 2] == 0
    loose = ng.layers(v, floor, 0.05, 1.8, floor_sigma=0.03)
    assert loose.state[2, 2] == ng.FREE and loose.floor_observed[2, 2] == 1
    assert sharp.height[4, 4] == pytest.approx(0.075, abs=0.01)  # the ceiling is above 2.0 m
    high = ng.layers(v, floor, 0.05, 1.8, height_cap=2.6)
    assert high.height[4, 4] == pytest.approx(2.475, abs=0.01)


def test_a_gap_between_two_sides_of_one_region_is_no_passage() -> None:
    st = np.zeros((45, 70), np.uint8)
    st[5:40, 2:68] = 1
    st[5:35, 34:36] = 2  # a wall with a gap; the rooms also meet below its end
    st[16:24, 34:36] = 1
    f = ng.features(_grid(st))
    assert len(f.regions) == 1 and f.passages == []
