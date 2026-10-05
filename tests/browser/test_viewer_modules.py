"""The viewer's reusable modules in a real browser (``-m browser``): the page mounted under another
server's prefix, the §2.5 display-budget notice, the in-browser PLY parser and the cameras of a
``mapper.sh locate`` PLY header, the mirror of ``scene_cameras`` for scene files, located cameras
drawn distinguishably from a map's frames, and the selection API (http_server.md: the 3D scene
viewer reuses this drawing; selecting an object highlights it everywhere)."""

from __future__ import annotations

import base64
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.locate import Located, located_blocks, pose_comment
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.cloud import map_cloud_source
from oh_my_slam.segmentation.colors import color_for_id, color_hex_for_id
from oh_my_slam.viewer.bundle import ViewBundle, scene_cameras
from tests.browser.conftest import View
from tests.browser.scenes import running_mounted
from tests.synth.scene import look_at

pytestmark = [pytest.mark.browser]

K = Intrinsics(320.0, 330.0, 320.0, 240.0, 640, 480)
PREFIX = "/viewer/abc/"
BUDGET = 1000
N = 5000


def located(k: int, at: tuple[float, float, float]) -> Located:
    return Located(Path(f"/in/query_{k}.jpg"), k, look_at(np.array(at), np.array([0.0, 0.0, 0.5])), K,
                   inliers=50)


def scene_doc() -> dict[str, Any]:
    """A map scene (two objects, three keyframes) plus two located cameras, as ``mapper.sh locate
    -t full`` returns it."""
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
    css, streams, located_frames = located_blocks([located(0, (1.5, -1.5, 1.0)), located(1, (-1.5, 1.5, 1.4))],
                                                  base=3)
    return ol.document(ol.metadata("modules"), objects,
                       coordinate_systems={"map": ol.map_cs(["camera_0", *css]),
                                           "camera_0": ol.sensor_cs("map"), **css},
                       streams={"camera_0": ol.camera_stream(K), **streams},
                       frames={**frames, **located_frames})


def cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    xyz = np.c_[rng.uniform(-1.5, 1.5, N), rng.uniform(-1.5, 1.5, N), rng.uniform(0, 1, N) ** 3]
    labels = np.zeros(N, np.int32)
    labels[(np.abs(xyz[:, 0] - 0.5) < 0.2) & (np.abs(xyz[:, 1] - 0.2) < 0.15)] = 1
    labels[(np.abs(xyz[:, 0] + 0.6) < 0.25) & (np.abs(xyz[:, 1] + 0.3) < 0.25)] = 2
    return xyz, rng.integers(40, 220, (N, 3), np.uint8), labels


@pytest.fixture(scope="module")
def mounted(browser: Any) -> Iterator[View]:
    """A 5000-point map with a display budget of 1000 points, served under a prefix by another
    server."""
    xyz, rgb, labels = cloud()
    source = map_cloud_source(xyz, rgb, labels, {1, 2}, np.array([[0.0, -3.0, 1.5]]))
    bundle = ViewBundle(mode="map", title="modules", scene=scene_doc(), source=source, catalog=[],
                        point_budget=BUDGET)
    with running_mounted(bundle, PREFIX) as url:
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


def test_the_page_works_under_another_servers_prefix(mounted: View) -> None:
    v = mounted
    assert v.url.endswith(PREFIX) and v.js("() => window.__viewer.ready")
    urls = v.js("() => performance.getEntriesByType('resource').map(e => e.name)")
    outside = [u for u in urls if not u.startswith(v.url) and not u.endswith("/favicon.ico")]
    assert urls and not outside, outside  # nothing outside the prefix
    assert v.errors == []


def test_display_budget_notice(mounted: View) -> None:
    """Spec §2.5: above the budget, one point per voxel, "showing X of Y points" with the voxel
    edge; the segmentation layer draws the same subset."""
    v = mounted
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


def ply_fixture(encoding: str) -> tuple[bytes, PointCloud]:
    rng = np.random.default_rng(1)
    n = 40
    normals = rng.normal(size=(n, 3))
    pc = PointCloud(rng.normal(size=(n, 3)), rng.integers(0, 255, (n, 3), np.uint8),
                    rng.integers(0, 4, n).astype(np.int32),
                    normals / np.linalg.norm(normals, axis=1, keepdims=True))
    comments = ["oh-my-slam map frame (z up), metres",
                f"attributes {CloudAttrs(color='segment', normals=True, label=True).describe(CloudScope.MAP)}",
                pose_comment(located(0, (1.0, -2.0, 1.5))),
                pose_comment(Located(Path("/in/far.jpg"), 1, reason="no overlap"))]
    return ply_bytes(pc, encoding=encoding, comments=comments), pc


@pytest.mark.parametrize("encoding", ["binary", "ascii"])
def test_ply_parser_and_located_cameras(mounted: View, encoding: str) -> None:
    v = mounted
    data, pc = ply_fixture(encoding)
    out = module(v, "ply.js", """
      const bytes = Uint8Array.from(atob(arg), (c) => c.charCodeAt(0));
      const c = m.parsePly(bytes.buffer);
      const cams = (await import(new URL('static/lib/cameras.js', document.baseURI).href)).plyCameras(c.header.comments);
      return {header: c.header, position: [...c.arrays.position], color: [...c.arrays.color],
              normal: [...c.arrays.normal], label: [...c.arrays.label], cams};
    """, base64.b64encode(data).decode())
    h = out["header"]
    assert (h["count"], h["total"], h["voxel"]) == (40, 40, 0)
    assert h["attrs"] == "color=segment,voxel=0,normals=on,label=on,encoding=binary"
    np.testing.assert_array_equal(np.float32(out["position"]), pc.xyz.ravel())
    np.testing.assert_array_equal(out["color"], pc.rgb.ravel())
    np.testing.assert_array_equal(np.float32(out["normal"]), pc.normals.ravel())
    np.testing.assert_array_equal(out["label"], pc.label)
    (cam,) = out["cams"]["cameras"]
    expected = located(0, (1.0, -2.0, 1.5))
    assert cam["name"] == "located_0" and cam["located"] is True and cam["source"] == "query_0.jpg"
    np.testing.assert_allclose(cam["T"], expected.T_map_cam.matrix(), atol=1e-6)
    assert cam["position"] == [float(round(x, 6)) for x in expected.T_map_cam.t]
    assert cam["K"] == [K.fx, K.fy, K.cx, K.cy] and cam["size"] == [K.width, K.height]
    assert out["cams"]["unlocated"] == ["/in/far.jpg"]
    assert v.errors == []


def test_ply_parser_refuses_what_it_cannot_draw(mounted: View) -> None:
    v = mounted
    bad = {"hello": "not a PLY file",
           "ply\nformat binary_big_endian 1.0\nelement vertex 0\nproperty float x\nend_header\n":
               "unsupported PLY format",
           "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n": "no x y z",
           "ply\nformat ascii 1.0\nelement face 1\nend_header\n": "first PLY element",
           "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
           "property float z\nend_header\n1 2 3\n": "1 vertices, not 2",
           "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
           "property float z\nend_header\n1 two 3\n": '"two" is not a number'}
    for text, message in bad.items():
        err = module(v, "ply.js", "try { m.parsePly(new TextEncoder().encode(arg).buffer); return null; }"
                     " catch (e) { return e.message; }", text)
        assert err and message in err, (text, err)


def test_scene_cameras_mirror_the_server(mounted: View) -> None:
    """JS ``sceneCameras`` (scene files opened in the browser) gives what bundle.scene_cameras
    serves, the located frames of ``mapper.sh locate -t full`` included."""
    v = mounted
    single = ol.document(ol.metadata("x"), {}, coordinate_systems={"camera": ol.sensor_cs()},
                         streams={"camera": ol.camera_stream(K)},
                         frames={"0": ol.frame(0.0, stream_uris={"camera": "dir/x.jpg"})})
    odd = scene_doc()  # present-but-null values, as Python's dict.get reads them
    odd["openlabel"]["frames"]["0"]["frame_properties"]["keyframe"] = None
    for doc in (scene_doc(), single, odd):
        js = module(v, "cameras.js", "return m.sceneCameras(arg);", doc)
        py = scene_cameras(doc)
        assert len(js) == len(py) > 0
        for a, b in zip(js, py, strict=True):
            for key in ("name", "frame", "position", "K", "size", "update", "source", "located"):
                assert a[key] == b[key], key
            np.testing.assert_allclose(a["T"], b["T"], atol=1e-9)
    assert [c["located"] for c in scene_cameras(scene_doc())] == [False] * 3 + [True] * 2


def test_located_cameras_are_distinguishable(mounted: View) -> None:
    """Located cameras: their own colour, dashed frustums and a "located" label (in the view and in
    the camera list), so colour is never the only cue."""
    v = mounted
    v.js("() => window.__viewerResetView()")
    v.settle()
    drawn = v.js("""() => {
      return window.__viewerGroups.cameras.children.map(l => ({name: l.name, type: l.material.type, color: '#' + l.material.color.getHexString(),
                                   vertices: l.geometry.attributes.position.count}));
    }""")
    by = {d["name"]: d for d in drawn}
    assert by["frustums"]["type"] == "LineBasicMaterial" and by["frustums"]["vertices"] == 3 * 16
    assert by["located-frustums"]["type"] == "LineDashedMaterial" and by["located-frustums"]["vertices"] == 2 * 16
    assert by["frustums"]["color"] != by["located-frustums"]["color"]
    tags = v.js("""() => [...document.querySelectorAll('.obj-label[data-group="cameras"]')]
      .filter(d => !d.hidden).map(d => d.innerText)""")
    assert tags and all(t.startswith("located") for t in tags), tags
    v.pg.click('#tabs button[data-tab="cameras"]')
    rows = v.js("""() => [...document.querySelectorAll('#cameras tbody tr[data-index]')].map(tr => ({
      located: tr.hasAttribute('data-located'), text: tr.children[0].innerText}))""")
    assert [r["located"] for r in rows] == [False] * 3 + [True] * 2
    assert all(("located" in r["text"]) == r["located"] for r in rows)
    v.pg.click('#tabs button[data-tab="controls"]')
    assert v.errors == []


def test_selection_api(mounted: View) -> None:
    """``select(id)`` highlights the object's box and label and tells every ``onSelect`` listener
    (how a host page highlights it in its other views); ``select(null)`` clears it."""
    v = mounted
    out = v.js("""() => {
      const viewer = window.__viewerApp, seen = [];
      const off = viewer.onSelect((id) => seen.push(id));
      const state = () => viewer.objects.map(o => [o.id, o.line.material.linewidth, o.div.classList.contains('selected')]);
      viewer.select(2); viewer.select(2);
      const on = state();
      viewer.select(null);
      const cleared = state();
      off(); viewer.select(1); viewer.select(null);
      return {seen, on, cleared};
    }""")
    assert out["seen"] == [2, None]
    widths = {i: w for i, w, _ in out["on"]}
    assert widths[2] > widths[1] and {i for i, _, sel in out["on"] if sel} == {2}
    assert len({w for _, w, _ in out["cleared"]}) == 1 and not any(s for *_, s in out["cleared"])
    assert v.errors == []
