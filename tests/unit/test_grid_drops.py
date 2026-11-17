"""Drops and the blind disc's extent in the navigation grid (map.md §6, oms/2.9;
``mapping.navgrid``): a raised top over a room floor 1.1 m below, hand-made grids for the rules'
edges, and a flat floor whose far sheets sag below the floor band (no drop)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oh_my_slam.mapping import navgrid as ng
from tests.synth import rooms

K = rooms.K640


def _volume(scene: rooms.Scene, cams: list[np.ndarray]) -> ng.Volume:
    vol = ng.Volume()
    for i, T in enumerate(cams):
        d = scene.depth(T)
        vol.integrate(f"f{i}", d, K, T, d > 0)
    return vol


def _xy(lay: ng.Layers) -> tuple[np.ndarray, np.ndarray]:
    rr, cc = np.mgrid[0:lay.shape[0], 0:lay.shape[1]]
    return lay.origin[0] + (cc + 0.5) * lay.cell, lay.origin[1] + (rr + 0.5) * lay.cell


def _past_edge(lay: ng.Layers) -> np.ndarray:
    """Each cell centre's distance beyond the raised top's edge (negative: over the top)."""
    x, y = _xy(lay)
    return np.maximum(np.abs(x) - rooms.TOP[0], np.abs(y) - rooms.TOP[1])


def _grid(state: np.ndarray, cell: float = 0.1, floor: int = 0) -> ng.Layers:
    R, C = state.shape
    return ng.Layers(np.zeros(2), cell, state.astype(np.uint8), np.ones((R, C), np.uint16),
                     np.full((R, C), np.nan, np.float32), np.full((R, C), floor, np.uint8))


def _drop(lay: ng.Layers, rows: slice, cols: slice, height: float = -1.1) -> None:
    lay.state[rows, cols] = ng.OCCUPIED
    lay.floor_observed[rows, cols] = ng.FLOOR_DROP
    lay.height[rows, cols] = height


@pytest.fixture(scope="module")
def top() -> tuple[ng.Volume, list[ng.Stand]]:
    cams = rooms.top_tour()
    return _volume(rooms.raised_top(), cams), [ng.Stand(T, K, 320, 240) for T in cams]


# -- drops ---------------------------------------------------------------------------------------


def test_the_room_floor_past_a_raised_top_is_a_drop(top: tuple[ng.Volume, list[ng.Stand]]
                                                    ) -> None:
    """A camera 0.5 m above a 1.1 m x 2 m top (z 0), the room floor 1.1 m below: every cell
    over the room floor the cameras saw is occupied whatever the band, ``floor_observed`` 3,
    ``height_m`` the room floor's (negative); the top is seen floor, never a drop."""
    vol, _ = top
    assert vol.estimate_floor(0.5, 0.0) == pytest.approx(0.0, abs=0.01)
    lay = ng.layers(vol, 0.0, 0.05, 1.8)
    past = _past_edge(lay)
    drop = lay.floor_observed == ng.FLOOR_DROP
    # the visible room floor: from 0.45 m past the edge (the edge hides a strip, below) outward
    seen = (past > 0.45) & (past < 2.5)
    assert seen.sum() > 10000 and drop[seen].mean() > 0.999
    assert (lay.state[drop] == ng.OCCUPIED).all()
    assert lay.height[drop] == pytest.approx(-rooms.TOP_DROP, abs=0.05)
    for xy in ([2.5, 0.0], [0.0, 3.0], [-2.5, 0.5], [1.5, -2.0]):
        rc = lay.cell_of(xy)
        assert (lay.state[rc], lay.floor_observed[rc]) == (ng.OCCUPIED, ng.FLOOR_DROP)
        assert lay.height[rc] == pytest.approx(-rooms.TOP_DROP, abs=0.05)
    # whatever the band: one that ends below the room floor's height marks them all the same
    high = ng.layers(vol, 0.0, 0.5, 1.0)
    assert ((high.floor_observed == ng.FLOOR_DROP) == drop).all()
    assert (high.state[drop] == ng.OCCUPIED).all()
    # the top itself: its floor was seen, so no cell of it is a drop, and it is free
    on = past < -0.05
    assert (lay.floor_observed[on] == 1).all() and (lay.state[on] == ng.FREE).all()
    assert not drop[past < 0].any()
    # the strip the top's edge hides from every camera (the room floor is seen from 0.33 m past
    # the edge here) holds no surface: seen through, floor not seen, no drop (map.md §6 asks a
    # surface below the band)
    hidden = (past > 0.1) & (past < 0.3)
    assert (lay.state[hidden] == ng.FREE).all() and (lay.floor_observed[hidden] == 0).all()
    assert np.isnan(lay.height[hidden]).all()


def _sight_clear(drop: np.ndarray, lay: ng.Layers, a: np.ndarray, b: np.ndarray) -> bool:
    """Whether the straight segment a-b (map x, y) crosses no drop cell (sampled every tenth of
    a cell)."""
    n = max(2, math.ceil(float(np.linalg.norm(b - a)) / (0.1 * lay.cell)))
    p = a[None] + (b - a)[None] * np.linspace(0.0, 1.0, n)[:, None]
    r = np.floor((p[:, 1] - lay.origin[1]) / lay.cell).astype(int)
    c = np.floor((p[:, 0] - lay.origin[0]) / lay.cell).astype(int)
    return not drop[r, c].any()


def test_no_cell_over_or_past_a_drop_is_free_by_rule(top: tuple[ng.Volume, list[ng.Stand]]
                                                     ) -> None:
    """The level camera at the top's centre has a blind disc of 1.33 m, wider than the top:
    whatever its extent, no drop cell is freed, and every cell free by rule has a camera whose
    straight line to it crosses no drop cell."""
    vol, stands = top
    assert ng.blind_radius(stands[0], 0.0) == pytest.approx(1.333, abs=0.01)
    plain = ng.layers(vol, 0.0, 0.05, 1.8)
    drop = plain.floor_observed == ng.FLOOR_DROP
    counts = {}
    for disc in ng.BLIND_DISCS:
        lay = ng.layers(vol, 0.0, 0.05, 1.8)
        ng.mark_stands(lay, stands, 0.0, 0.2, disc)
        assert ((lay.floor_observed == ng.FLOOR_DROP) == drop).all()  # a drop stays a drop
        assert (lay.state[drop] == ng.OCCUPIED).all()
        by_rule = lay.floor_observed == ng.FLOOR_BY_RULE
        counts[disc] = int(by_rule.sum())
        x, y = _xy(lay)
        cams = [np.asarray(s.T_map_cam)[:2, 3] for s in stands]
        for r, c in np.argwhere(by_rule):
            p = np.array([x[r, c], y[r, c]])
            near = [p + lay.cell * np.array(o) for o in ((0, 0), (.25, 0), (-.25, 0), (0, .25),
                                                         (0, -.25))]
            assert any(_sight_clear(drop, lay, s, q) for s in cams for q in near), (r, c)
        # past the edge, nothing is free but the strip the edge hides
        assert _past_edge(lay)[lay.state == ng.FREE].max() < 0.35
    # the top's floor was seen wherever a camera stood: only the full disc adds cells by rule,
    # those of the hidden strip between the edge and the first drop cell
    assert counts["none"] == counts["footprint"] == 0 and counts["full"] > 500
    # the features: one region, no passage across the top (a drop is no wall), no frontier
    # along the drop (it is not unknown)
    lay = ng.layers(vol, 0.0, 0.05, 1.8)
    ng.mark_stands(lay, stands, 0.0, None)
    feat = ng.features(lay, 0.2)
    assert len(feat.regions) == 1 and feat.passages == [] and feat.frontiers == []
    assert feat.regions[0].area_m2 == pytest.approx(2.2 + 2.49, abs=0.3)  # the top, the strip


def test_a_drop_cuts_the_blind_disc_along_straight_lines() -> None:
    """On a hand-made grid (0.1 m cells): a short row of drop cells 0.5 m in front of a level
    camera 1 m up (blind disc 2 m at most: here 2.0). The cells behind the row, on straight
    lines from the camera, are not freed, though the disc reaches them around its ends; the
    cells beside its shadow are."""
    lay = _grid(np.zeros((60, 60), np.uint8))
    _drop(lay, slice(27, 34), slice(35, 36))  # x 3.5-3.6, y 2.7-3.4
    stand = ng.Stand(rooms.camera(3.05, 3.05, 1.0, 0.0, 0.0), K, 320, 240)
    assert ng.blind_radius(stand, 0.0) == ng.BLIND_RING_MAX_M
    ng.mark_stands(lay, [stand], 0.0, None)
    assert (lay.state[27:34, 35] == ng.OCCUPIED).all()
    assert (lay.floor_observed[27:34, 35] == ng.FLOOR_DROP).all()
    assert (lay.state[28:33, 36:45] == ng.UNKNOWN).all()  # behind the row
    assert (lay.floor_observed[28:33, 36:45] == 0).all()
    assert (lay.state[25:36, 31:35] == ng.FREE).all()  # in front of it
    assert (lay.floor_observed[25:36, 31:35] == ng.FLOOR_BY_RULE).all()
    assert lay.state[40, 36] == ng.FREE and lay.state[20, 36] == ng.FREE  # beside the shadow
    assert lay.state[30, 20] == ng.FREE  # and behind the camera
    # the shadow widens with the distance: at x 4.5 it spans more rows than the row of drops
    assert (lay.state[24:37, 45] == ng.UNKNOWN).all() and lay.state[42, 42] == ng.FREE


def test_a_camera_over_a_drop_frees_nothing() -> None:
    """An arm's camera past the worktop's edge stands over a drop: neither its cell, nor its
    footprint, nor its blind disc is free by rule."""
    lay = _grid(np.zeros((40, 40), np.uint8))
    _drop(lay, slice(0, 40), slice(20, 40))
    before = lay.state.copy()
    over = ng.Stand(rooms.camera(2.55, 2.05, 1.0, 180.0, 0.0), K, 320, 240)
    for disc in ng.BLIND_DISCS:
        ng.mark_stands(lay, [over], 0.0, 0.3, disc)
        assert (lay.state == before).all() and not (lay.floor_observed == 2).any()
    # beside the drop, its footprint and disc stop at the drop
    beside = ng.Stand(rooms.camera(1.85, 2.05, 1.0, 180.0, 0.0), K, 320, 240)
    ng.mark_stands(lay, [beside], 0.0, 0.3, "footprint")
    assert (lay.state[:, 20:] == ng.OCCUPIED).all() and lay.state[20, 19] == ng.FREE
    assert (lay.floor_observed[:, 20:] == ng.FLOOR_DROP).all()


def test_the_blind_disc_extents() -> None:
    """``blind_disc`` (map.md §6): ``none`` frees the camera's own cell only, whatever the
    footprint; ``footprint`` the footprint's disc; ``full`` the whole blind disc."""
    stand = ng.Stand(rooms.camera(3.05, 3.05, 0.5, 0.0, 0.0), K, 320, 240)
    ring = ng.blind_radius(stand, 0.0)
    assert ring == pytest.approx(1.333, abs=0.01)
    free = {}
    for disc in ng.BLIND_DISCS:
        lay = _grid(np.zeros((60, 60), np.uint8))
        ng.mark_stands(lay, [stand], 0.0, 0.31, disc)
        assert ((lay.state == ng.FREE) == (lay.floor_observed == ng.FLOOR_BY_RULE)).all()
        x, y = _xy(lay)
        dist = np.hypot(x - 3.05, y - 3.05)
        free[disc] = lay.state == ng.FREE
        reach = {"none": 0.0, "footprint": 0.31, "full": ring}[disc]
        assert (free[disc] == ((dist <= reach) | (dist < 0.01))).all()
    assert free["none"].sum() == 1 and free["footprint"].sum() == 29
    assert free["full"].sum() > 500
    # without a footprint, ``none`` and the default differ by the disc alone
    lay = _grid(np.zeros((60, 60), np.uint8))
    ng.mark_stands(lay, [stand], 0.0, None, "none")
    assert (lay.state == ng.FREE).sum() == 1


@pytest.mark.parametrize("rise, treads", [(0.17, 6), (0.08, 1)])
def test_stairs_down_and_a_kerb_are_drops(rise: float, treads: int) -> None:
    """A staircase going down from the floor's edge (0.17 m a step, treads of 0.28 m), and a
    kerb of 0.08 m, just below the 0.06 m floor band, seen from a camera 1 m up on the floor:
    what is seen of every tread is a drop at its own height; the floor above is seen floor."""
    s = rooms.Scene()
    s.add((-3.0, -2.0, -0.05), (1.0, 2.0, 0.0))  # the floor, its edge at x 1
    for i in range(treads):
        x0 = 1.0 + 0.28 * i
        s.add((x0, -2.0, -rise * (i + 1) - 0.05),
              (x0 + 0.28 if treads > 1 else 4.0, 2.0, -rise * (i + 1)))
    lay = ng.layers(_volume(s, [rooms.camera(x, y, 1.0, yaw, -35.0) for x in (-1.0, 0.2)
                                for y in (-0.8, 0.8) for yaw in (-40, -20, 0, 20, 40, 180)]),
                    0.0, 0.05, 1.8)
    row = lay.cell_of([0.0, 0.0])[0]
    c0 = lay.cell_of([1.0, 0.0])[1]
    assert (lay.floor_observed[row, c0 - 10:c0 - 1] == 1).all()
    assert (lay.state[row, c0 - 10:c0 - 1] == ng.FREE).all()
    for i in range(treads):
        cells = slice(lay.cell_of([1.0 + 0.28 * i, 0.0])[1] + 1,
                      lay.cell_of([1.0 + 0.28 * (i + 1), 0.0])[1] + 1)
        drop = lay.floor_observed[row, cells] == ng.FLOOR_DROP
        assert drop.any(), i
        assert (lay.state[row, cells][drop] == ng.OCCUPIED).all()
        assert lay.height[row, cells][drop] == pytest.approx(-rise * (i + 1), abs=0.04)
    assert not (lay.floor_observed[:, :c0] == ng.FLOOR_DROP).any()  # none on the floor


# -- what is no drop -----------------------------------------------------------------------------


def test_an_obstacle_of_the_band_over_a_drop_reads_as_the_obstacle() -> None:
    """A cell with a surface below the floor band, no seen floor, and an obstacle of the band
    above the floor (a wall that rises from below the floor): occupied by the obstacle as
    before oms/2.9, ``floor_observed`` 0, ``height_m`` the obstacle's. A band that leaves the
    obstacle out makes it a drop like its neighbours."""
    v = ng.Volume()
    v.ensure(np.array([0.0, 0.0, -1.5]), np.array([1.0, 1.0, 1.5]))
    zc = v.origin[2] + (np.arange(v.shape[0]) + 0.5) * v.vz
    low, up = int(np.argmin(np.abs(zc + 1.125))), int(np.argmin(np.abs(zc - 0.525)))
    y, x = v.shape[1] // 2 - 5, v.shape[2] // 2 - 5
    v.hit[low, y:y + 10, x:x + 10] = 3  # the room floor under 10 x 10 cells
    v.hit[up, y:y + 10, x:x + 4] = 3  # an obstacle 0.5 m up over four of their columns
    lay = ng.layers(v, 0.0, 0.05, 1.8)
    beside, under = (slice(y, y + 10), slice(x + 4, x + 10)), (slice(y, y + 10), slice(x, x + 4))
    assert (lay.floor_observed[beside] == ng.FLOOR_DROP).all()
    assert lay.height[beside] == pytest.approx(-1.125, abs=0.01)
    assert (lay.state[under] == ng.OCCUPIED).all() and (lay.floor_observed[under] == 0).all()
    assert lay.height[under] == pytest.approx(0.525, abs=0.01)
    assert (lay.floor_observed == ng.FLOOR_DROP).sum() == 60
    low_band = ng.layers(v, 0.0, 0.05, 0.3)
    assert (low_band.floor_observed == ng.FLOOR_DROP).sum() == 100
    assert (low_band.state[under] == ng.OCCUPIED).all()
    assert low_band.height[under] == pytest.approx(-1.125, abs=0.01)  # the drop's, not 0.525


def test_the_drop_rule_on_hand_made_columns() -> None:
    """``_drops``: a surface below the band is a drop unless it continues the seen floor
    without a step, or is too small."""
    nan = np.nan
    cand = np.zeros((12, 30), bool)
    below = np.full((12, 30), nan)
    floor = np.full((12, 30), nan)
    floor[:, :10] = -0.02  # seen floor on the left
    # rows 0-5: a sheet sagging from the floor's height down to -0.5 m, 3 cm a cell
    cand[:6, 10:26] = True
    below[:6, 10:26] = -0.08 - 0.03 * np.arange(16)
    # rows 7-11: the floor of a room 1.1 m below, from the next cell on
    cand[7:, 10:26] = True
    below[7:, 10:26] = -1.1
    got = ng._drops(cand, below, floor, 0.06, 0.05)
    assert not got[:6].any() and got[7:, 10:26].all() and not got[:, :10].any()
    # a step of 5 cm a cell is a slope the 0.06 m step still follows; one of 8 cm is a stair
    below[:6, 10:26] = -0.08 - 0.08 * np.arange(16)
    got = ng._drops(cand, below, floor, 0.06, 0.05)
    assert not got[:6, 10].any() and got[:6, 11:26].all()
    # a sheet under seen floor carries the continuity on: floor seen (by a near camera) up to
    # column 14, a far camera's sheet under it, which leaves the band at column 3 (where it is
    # the lowest floor-like surface before) and sags on, 0.43 m down where the seen floor ends
    cand[:], below[:], floor[:] = False, nan, nan
    floor[:, :3], floor[:, 3:15] = -0.05, 0.0
    below[:, 3:26] = -0.07 - 0.03 * (np.arange(3, 26) - 3)
    cand[:, 15:26] = True
    assert below[0, 15] < -0.4 and not ng._drops(cand, below, floor, 0.06, 0.05).any()
    # ... but a lower floor under the edge's own cell does not: the room floor seen under the
    # top's last cell (an arm's camera looking straight down) is no continuation of the top
    below[:, 3:26] = -1.1
    assert ng._drops(cand, below, floor, 0.06, 0.05)[:, 15:26].all()
    # fewer cells than DROP_MIN_M2 together are no drop: 4 x 4 cells of 5 cm are 0.04 m2
    cand[:], below[:], floor[:] = False, nan, nan
    cand[2:6, 2:6] = cand[2:7, 12:16] = True
    below[cand] = -0.5
    got = ng._drops(cand, below, floor, 0.06, 0.05)
    assert not got[2:6, 2:6].any() and got[2:7, 12:16].all()
    assert ng.DROP_MIN_M2 == 0.05
    # the step follows the floor's noise: half the floor band, 0.06 m at least
    assert ng.drop_step_m(0.02) == 0.06 and ng.drop_step_m(0.08) == pytest.approx(0.12)


def _sagging_floor() -> tuple[ng.Volume, list[np.ndarray]]:
    """A flat floor (z 0) in a walled room, seen from a camera 0.37 m up turning at two spots;
    half the frames' poses are tilted down by 4 deg about the camera, as a walking robot's
    priors are, so their floor sheets sag below the true floor with the distance (0.28 m at
    4 m), and a patch of every image's depth is 30 % too long (flying pixels below the floor)."""
    s = rooms.Scene()
    s.add((-4.0, -4.0, -0.05), (4.0, 4.0, 0.0))
    s.add((-4.1, -4.1, 0.0), (-4.0, 4.1, 2.0)).add((4.0, -4.1, 0.0), (4.1, 4.1, 2.0))
    s.add((-4.1, -4.1, 0.0), (4.1, -4.0, 2.0)).add((-4.1, 4.0, 0.0), (4.1, 4.1, 2.0))
    vol = ng.Volume()
    cams = []
    for i, (x, y, yaw) in enumerate((x, y, yaw) for x, y in ((-2.0, -2.0), (1.5, 1.0))
                                    for yaw in range(0, 360, 30)):
        true = rooms.camera(x, y, 0.37, yaw, -10.0)
        d = s.depth(true)
        d[200:206, 100 + 10 * (i % 12):106 + 10 * (i % 12)] *= 1.3
        T = rooms.camera(x, y, 0.37, yaw, -14.0) if i % 2 else true
        vol.integrate(f"f{i}", d, K, T, d > 0)
        cams.append(T)
    return vol, cams


def test_a_flat_floor_with_sagging_sheets_and_flying_pixels_has_no_drop(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Below-floor noise on a normal floor makes no drop cell (the evaluation's
    ``grid.floor_false_obstacles`` stays 0): the sheets of tilted frames continue the seen floor
    without a step, flying pixels are too few together. Without that test of ``_drops`` the same
    volume gives hundreds of drop cells, as the floor of a real room did."""
    vol, cams = _sagging_floor()
    floor = vol.estimate_floor(0.37, 0.0)
    assert floor == pytest.approx(0.0, abs=0.03)
    sigma = 0.03  # a floor band of 0.09 m
    lay = ng.layers(vol, floor, 0.05, 1.8, sigma)
    assert not (lay.floor_observed == ng.FLOOR_DROP).any()
    x, y = _xy(lay)
    inside = (np.abs(x) < 3.9) & (np.abs(y) < 3.9)
    assert not (lay.state[inside] == ng.OCCUPIED).any()  # no obstacle on the open floor
    # the blind discs are whole: nothing cuts them
    ng.mark_stands(lay, [ng.Stand(T, K, 320, 240) for T in cams], floor, None)
    assert lay.state[lay.cell_of([-2.0, -1.0])] == ng.FREE
    # the same volume with every candidate kept: the noise is there, well below the band
    monkeypatch.setattr(ng, "_drops", lambda cand, *_: cand)
    raw = ng.layers(vol, floor, 0.05, 1.8, sigma)
    noise = raw.floor_observed == ng.FLOOR_DROP
    assert noise.sum() > 300 and raw.height[noise].min() < -0.2
    assert (raw.state[noise] == ng.OCCUPIED).all()


# -- resampling, walls, features -----------------------------------------------------------------


def test_resampled_drops() -> None:
    """A coarse cell that covers a drop cell is a drop (its worst state is occupied), with the
    drop's height, not the seen floor's beside it; finer cells repeat theirs."""
    lay = _grid(np.ones((8, 8), np.uint8), floor=1)
    lay.height[:] = 0.0
    lay.top = np.full((8, 8), np.nan, np.float32)
    _drop(lay, slice(0, 8), slice(5, 8), -1.1)
    lay.height[3, 5] = -0.6
    coarse = ng.resample(lay, 0.2)
    assert (coarse.floor_observed[:, 2:] == ng.FLOOR_DROP).all()  # column 2 covers cells 4, 5
    assert (coarse.state[:, 2:] == ng.OCCUPIED).all()
    assert (coarse.floor_observed[:, :2] == 1).all() and (coarse.state[:, :2] == ng.FREE).all()
    assert coarse.height[0, 2] == pytest.approx(-1.1) and coarse.height[1, 2] == pytest.approx(-0.6)
    assert coarse.height[0, 3] == pytest.approx(-1.1) and coarse.height[0, 0] == 0.0
    fine = ng.resample(lay, 0.05)
    assert (fine.floor_observed[:, 10:] == ng.FLOOR_DROP).all()
    assert (fine.floor_observed[:, :10] == 1).all() and fine.height[6, 10] == pytest.approx(-0.6)


def test_a_drop_is_no_wall() -> None:
    """``inflate_walls`` grows thin walls by a cell; a drop, however long, neither grows nor is
    grown into."""
    st = np.ones((40, 40), np.uint8)
    lay = _grid(st, floor=1)
    lay.top = np.full((40, 40), np.nan, np.float32)
    _drop(lay, slice(0, 40), slice(30, 32))  # a strip of drop 4 m long, two cells wide
    lay.state[5:35, 10] = ng.OCCUPIED  # a thin wall 3 m long
    lay.top[5:35, 10] = 1.5
    lay.state[5:35, 29] = ng.OCCUPIED  # and one right beside the drop
    lay.top[5:35, 29] = 1.5
    out = ng.inflate_walls(lay)
    assert (out.state[10:30, 9] == ng.OCCUPIED).all() and (out.state[10:30, 11] == ng.OCCUPIED).all()
    assert (out.state[:, 32] == ng.FREE).all()  # the drop did not grow
    assert (out.state[0:4, 29] == ng.FREE).all()
    assert (out.state[10:30, 28] == ng.OCCUPIED).all()  # the wall beside it did, away from it
    assert np.isnan(out.top[10:30, 30]).all()  # ... and not into the drop cells
    assert (out.floor_observed[:, 30:32] == ng.FLOOR_DROP).all()


def test_a_drop_bounds_regions_but_is_no_jamb() -> None:
    """A free top 1.1 m wide between drops: as walls the drops would make a passage across it
    (a chord walled at both ends); as drops they bound one region, with no passage and no
    frontier along them. A door in a real wall beside a drop is still found."""
    st = np.zeros((140, 140), np.uint8)
    st[20:120, 20:120] = ng.OCCUPIED
    st[50:90, 60:82] = ng.FREE
    walls = ng.features(_grid(st, 0.05, floor=1))  # chords across it, as along a corridor
    assert walls.passages and len(walls.regions) == len(walls.passages) + 1
    lay = _grid(st, 0.05, floor=1)
    lay.floor_observed[st == ng.OCCUPIED] = ng.FLOOR_DROP
    drops = ng.features(lay)
    assert drops.passages == [] and len(drops.regions) == 1 and drops.frontiers == []
    assert drops.regions[0].area_m2 == pytest.approx(2.2, abs=0.01)
    # two rooms joined by a door in a wall, a drop along one side of both
    st = np.zeros((60, 100), np.uint8)
    st[5:55, 5:95] = ng.FREE
    st[5:55, 49:51] = ng.OCCUPIED
    st[25:35, 49:51] = ng.FREE
    lay = _grid(st, 0.1, floor=1)
    _drop(lay, slice(0, 5), slice(0, 100))
    feat = ng.features(lay)
    assert len(feat.passages) == 1 and len(feat.regions) == 2
    assert feat.passages[0].width == pytest.approx(1.1, abs=0.11)
