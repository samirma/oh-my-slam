"""The viewer's modules in a real browser (``-m browser``): the §2.5 display-budget notice over a map
above the point budget, and its wording when points are omitted with no voxel grid."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.cloud import map_cloud_source
from oh_my_slam.segmentation.colors import color_for_id, color_hex_for_id
from oh_my_slam.viewer.bundle import ViewBundle
from tests.browser.conftest import View
from tests.browser.scenes import running
from tests.synth.scene import look_at

pytestmark = [pytest.mark.browser]

K = Intrinsics(320.0, 330.0, 320.0, 240.0, 640, 480)
BUDGET = 1000
N = 5000


def scene_doc() -> dict[str, Any]:
    """A map scene: two objects, three keyframes."""
    objects = {}
    for k, (c, s, label) in enumerate([((0.5, 0.2, 0.3), (0.4, 0.3, 0.6), "chair"),
                                       ((-0.6, -0.3, 0.25), (0.5, 0.5, 0.5), "box")], start=1):
        size = np.array(s)
        cub = ol.cuboid(np.array(c), np.eye(3), size, "map", attributes_num=[
            ol.num("width_m", s[0]), ol.num("depth_m", s[1]), ol.num("height_m", s[2]),
            ol.num("volume_m3", float(np.prod(size)))])
        objects[str(k)] = ol.object_entry(f"{label} {k}", label, "map", cub, nums=[ol.num("score", 0.9)],
                                          texts=[ol.text("color_hex", color_hex_for_id(k))],
                                          vecs=[ol.vec("color", list(color_for_id(k)))])
    frames = {str(k): ol.frame(float(k), stream_uris={"camera_0": f"frames/f{k:06d}.jpg"},
                               transforms={"camera_0_to_map": ol.transform(
                                   "camera_0", "map", look_at(np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 1.2]),
                                                              np.array([0.0, 0.0, 0.5])))},
                               keyframe=f"f{k:06d}", update_id=1)
              for k, a in enumerate((0.0, 2.0, 4.0))}
    return ol.document(ol.metadata("modules"), objects,
                       coordinate_systems={"map": ol.map_cs(["camera_0"]),
                                           "camera_0": ol.sensor_cs("map")},
                       streams={"camera_0": ol.camera_stream(K)}, frames=frames)


def cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    xyz = np.c_[rng.uniform(-1.5, 1.5, N), rng.uniform(-1.5, 1.5, N), rng.uniform(0, 1, N) ** 3]
    labels = np.zeros(N, np.int32)
    labels[(np.abs(xyz[:, 0] - 0.5) < 0.2) & (np.abs(xyz[:, 1] - 0.2) < 0.15)] = 1
    labels[(np.abs(xyz[:, 0] + 0.6) < 0.25) & (np.abs(xyz[:, 1] + 0.3) < 0.25)] = 2
    return xyz, rng.integers(40, 220, (N, 3), np.uint8), labels


@pytest.fixture(scope="module")
def over_budget(browser: Any) -> Iterator[View]:
    """A 5000-point map with a display budget of 1000 points, served as view.sh serves it."""
    xyz, rgb, labels = cloud()
    source = map_cloud_source(xyz, rgb, labels, {1, 2}, np.array([[0.0, -3.0, 1.5]]))
    bundle = ViewBundle(mode="map", title="modules", scene=scene_doc(), source=source, catalog=[],
                        point_budget=BUDGET)
    with running(bundle) as url:
        v = View(browser, bundle, url)
        yield v
        v.pg.close()


def module(v: View, name: str, body: str, arg: Any = None) -> Any:
    """Run ``body`` (an async function of ``m``, the module, and ``arg``) in the page."""
    return v.js(f"""async (arg) => {{
      const m = await import(new URL('static/lib/{name}', document.baseURI).href);
      return await (async (m, arg) => {{ {body} }})(m, arg);
    }}""", arg)


# ------------------------------------------------------------------------------------------------


def test_display_budget_notice(over_budget: View) -> None:
    """Spec §2.5: above the budget, one point per voxel, "showing X of Y points" with the voxel
    edge; the segmentation layer draws the same subset."""
    v = over_budget
    head = v.js("() => window.__viewer.cloud")
    expected = v.bundle.cloud(v.bundle.parse_attrs([]))
    assert head["total"] == N and head["count"] == len(expected.cloud) <= BUDGET
    assert head["voxel"] == expected.voxel > 0
    note = v.pg.inner_text("#cloud-note")
    m = re.match(r"Showing ([\d,]+) of ([\d,]+) points: one per voxel of ([\d.]+) (mm|cm|m) edge", note)
    assert m, note
    assert int(m[1].replace(",", "")) == head["count"] and int(m[2].replace(",", "")) == N
    scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[m[4]]
    assert float(m[3]) * scale == pytest.approx(head["voxel"], rel=0.01)
    same = v.js("""() => {
      const g = window.__viewerGroups, p = g.points.children[0].geometry, s = g.segments.children[0].geometry;
      const idx = s.index.array;
      return {shared: p.attributes.position === s.attributes.position, n: p.attributes.position.count,
              max: idx.length ? Math.max(...idx) : -1, segmented: idx.length};
    }""")
    assert same["shared"] and same["n"] == head["count"] and same["max"] < head["count"]
    assert same["segmented"] == int((expected.cloud.label > 0).sum()) > 0
    assert v.errors == []


def test_budget_note_without_a_grid(over_budget: View) -> None:
    """Points omitted with no grid (edge 0: duplicates or non-finite points) are not described
    as a "0.00 mm" voxel; a complete cloud has no notice."""
    v = over_budget
    note = module(v, "controls.js", "return m.budgetNote({count: 5, total: 7, voxel: 0});")
    assert note.startswith("Showing 5 of 7 points: one per distinct position") and "mm" not in note
    assert module(v, "controls.js", "return m.budgetNote({count: 7, total: 7, voxel: 0});") == ""
