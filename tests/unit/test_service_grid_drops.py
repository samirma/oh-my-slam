"""The grid's drops and ``blind_disc`` through the API (map.md §6, oms/2.9): a wheeled robot's
camera 0.5 m above a raised top of 1.1 m x 2 m (the priors' z 0), the room floor 1.1 m below
(``tests.synth.rooms.raised_top``), the mapper replaced by ``tests.fakes.service.FakeBackend``.
Every body is checked against api.schema.json."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from oh_my_slam.service import schema
from tests.synth import rooms
from tests.unit.test_service_nav import CAM320, Nav, decode, nav, scene_frames  # noqa: F401

DROP, BY_RULE = 3, 2


def _map(nav: Nav, name: str, poses: list[np.ndarray]) -> Nav:  # noqa: F811
    nav.backend.floor_z = 0.0
    frames, files = scene_frames(nav, rooms.raised_top(), poses)
    job = nav.update(name, frames, files, {"cam": CAM320})
    assert job["state"] == "committed", (job["inputs"], job["quality"])
    return nav


@pytest.fixture
def top_map(nav: Nav) -> Nav:  # noqa: F811
    return _map(nav, "top", rooms.top_tour())


def _layers(g: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(state, floor_observed, height_m, each cell centre's distance past the top's edge)."""
    state, floor, height = (decode(g["layers"][k]) for k in ("state", "floor_observed",
                                                              "height_m"))
    rows, cols = g["shape"]
    x = g["origin_m"][0] + (np.arange(cols) + 0.5) * g["cell_m"]
    y = g["origin_m"][1] + (np.arange(rows) + 0.5) * g["cell_m"]
    past = np.maximum(np.abs(x)[None, :] - rooms.TOP[0], np.abs(y)[:, None] - rooms.TOP[1])
    return state, floor, height, past


def _cell(g: dict[str, Any], x: float, y: float) -> tuple[int, int]:
    return (int((y - g["origin_m"][1]) / g["cell_m"]), int((x - g["origin_m"][0]) / g["cell_m"]))


def test_the_room_floor_past_the_top_is_occupied_as_a_drop(top_map: Nav) -> None:
    g = top_map.get("/api/maps/top/grid", "Grid")
    assert g["schema"] == schema.SCHEMA_VERSION == "oms/2.17"
    assert g["blind_disc"] == "full"  # the default, echoed
    assert g["floor_z_m"] == pytest.approx(0.0, abs=0.01)  # the top: where the priors' origin is
    state, floor, height, past = _layers(g)
    assert set(np.unique(floor)) <= {0, 1, BY_RULE, DROP}
    drop = floor == DROP
    # every cell over the room floor the cameras saw, past the strip the edge hides
    seen = (past > 0.45) & (past < 2.5)
    assert seen.sum() > 10000 and drop[seen].mean() > 0.99 and (state[drop] == 2).all()
    assert height[drop] == pytest.approx(-rooms.TOP_DROP, abs=0.05)
    for x, y in ((2.5, 0.0), (0.0, 3.0), (-2.5, 0.5), (1.5, -2.0)):
        rc = _cell(g, x, y)
        assert (state[rc], floor[rc]) == (2, DROP)
        assert height[rc] == pytest.approx(-rooms.TOP_DROP, abs=0.05)
    # the top is free, its floor seen; no cell of it is a drop
    on = past < -0.05
    assert (state[on] == 1).all() and np.isin(floor[on], (1, BY_RULE)).all()
    assert (floor[on] == 1).mean() > 0.95 and not drop[past < 0].any()
    assert height[_cell(g, 0.2, 0.3)] == pytest.approx(0.0, abs=0.05)
    # no drop cell is free by rule (the layer has one value a cell), and no cell is free farther
    # past the edge than the strip it hides, which no confirmed frame saw (0.33 m here, up to
    # 0.48 m beside the seven frames this update keeps at their priors: they add no content)
    assert past[state == 1].max() < 0.5 and past[floor == BY_RULE].max() < 0.5
    # the drop bounds one region and makes neither a passage across the top nor a frontier
    assert len(g["regions"]) == 1 and g["passages"] == [] and g["frontiers"] == []
    # whatever the band: one wholly above the room floor's height marks the same drops
    band = top_map.get("/api/maps/top/grid?z_min=0.5&z_max=1.0", "Grid")
    s2, f2, h2, _ = _layers(band)
    assert ((f2 == DROP) == drop).all() and (s2[drop] == 2).all()
    assert h2[drop] == pytest.approx(-rooms.TOP_DROP, abs=0.05)
    # coarser and finer cells: a drop is occupied with its negative height in both
    for cell in (0.2, 0.025):
        r = top_map.get(f"/api/maps/top/grid?cell_m={cell}", "Grid")
        s3, f3, h3, p3 = _layers(r)
        d3 = f3 == DROP
        assert d3[(p3 > 0.5) & (p3 < 2.5)].mean() > 0.99 and (s3[d3] == 2).all()
        assert h3[d3] == pytest.approx(-rooms.TOP_DROP, abs=0.05)
        assert not d3[p3 < 0].any() and (s3[p3 < -cell] == 1).all()
        assert r["passages"] == g["passages"] and len(r["regions"]) == 1


def test_blind_disc_limits_the_cells_free_by_rule(nav: Nav) -> None:  # noqa: F811
    """A top whose middle no frame saw (the level turn at its centre, and the looks past its
    edges): ``none`` frees the cameras' own cells, ``footprint`` their footprints, ``full`` the
    blind disc, which reaches past the top's edge into the strip the edge hides; none of them
    a drop cell or one past it."""
    tour = rooms.top_tour()
    _map(nav, "part", tour[:12] + tour[36:])
    by_rule = {}
    for q in ("blind_disc=none", "blind_disc=none&footprint_radius_m=0.2",
              "blind_disc=footprint&footprint_radius_m=0.2", "blind_disc=full",
              "footprint_radius_m=0.2", ""):
        g = nav.get(f"/api/maps/part/grid?{q}", "Grid")
        assert g["blind_disc"] == (q.split("blind_disc=")[1].split("&")[0]
                                   if "blind_disc" in q else "full")
        state, floor, height, past = _layers(g)
        drop = floor == DROP
        # the looks past the edges see the room floor up to 0.8 m from the top
        assert drop[(past > 0.45) & (past < 0.8)].mean() > 0.99 and (state[drop] == 2).all()
        assert (state[floor == BY_RULE] == 1).all()
        by_rule[q] = floor == BY_RULE
        assert past[by_rule[q]].max() < 0.5
        assert state[_cell(g, 0.0, 0.0)] == 1  # where the camera stood
    none, none_foot, foot, full, full_foot, default = by_rule.values()
    assert 1 <= none.sum() <= 19 and (none == none_foot).all()  # the cameras' own cells
    assert foot.sum() > 10 * none.sum() and full.sum() > 3 * foot.sum()
    assert not (none & ~foot).any() and not (foot & ~full_foot).any()
    assert (full == default).all()
    g = nav.get("/api/maps/part/grid?blind_disc=footprint&footprint_radius_m=0.2", "Grid")
    _, floor, _, past = _layers(g)
    cams = np.array([T[:2, 3] for T in tour[:12] + tour[36:]])
    rows, cols = np.nonzero(floor == BY_RULE)
    xy = np.column_stack([g["origin_m"][0] + (cols + 0.5) * 0.05,
                          g["origin_m"][1] + (rows + 0.5) * 0.05])
    reach = np.min(np.linalg.norm(xy[:, None] - cams[None], axis=-1), axis=1)
    assert reach.max() <= 0.2 + 1e-6  # the footprint's disc, no farther
    # the footprint's passage flag is unchanged by the disc, and features follow the cells
    assert len(g["regions"]) >= 1 and g["passages"] == []


def test_blind_disc_is_validated_and_documented(top_map: Nav) -> None:
    doc = top_map.get("/api/maps/top/grid?blind_disc=footprint", status=400)
    assert doc["error"]["code"] == "invalid_request"
    assert doc["error"]["details"]["field"] == "footprint_radius_m"
    assert "footprint_radius_m" in doc["error"]["message"]
    for word in ("wide", "", "FULL", "0"):
        doc = top_map.get(f"/api/maps/top/grid?blind_disc={word}&footprint_radius_m=0.2",
                          status=400)
        assert doc["error"]["code"] == "invalid_request"
        assert doc["error"]["details"]["field"] == "blind_disc"
    # a footprint of radius 0 is given: the cameras' own cells
    g = top_map.get("/api/maps/top/grid?blind_disc=footprint&footprint_radius_m=0", "Grid")
    assert g["blind_disc"] == "footprint"
    params = {p["name"]: p for p in top_map.get("/api/openapi.json")["paths"][
        "/api/maps/{name}/grid"]["get"]["parameters"]}
    assert params["blind_disc"]["schema"] == {"type": "string",
                                              "enum": ["full", "footprint", "none"]}
    assert params["blind_disc"]["in"] == "query" and not params["blind_disc"]["required"]
    assert "enum" not in params["cell_m"]["schema"]
