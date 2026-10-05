"""Viewer in a real browser (``-m browser``; Microsoft Edge/Chromium through Playwright), for an
image (fake inference client) and a map (real mapper, COLMAP): the ``data-rendered`` signal, layer
toggles that affect only their layer, attribute controls taken from the shared attribute table that
re-derive the cloud without inference, exact §2.4 colours on screen, camera frustums at the scene's
poses, camera centres with "Go to", legible labels, the catalogue and the segmented image, errors
shown in the page, no console errors, layouts at desktop and phone widths, frames drawn only when
something changes, and the map unchanged by viewing.

Set ``OH_MY_SLAM_VIEWER_SHOTS=<folder>`` to keep the layout screenshots."""

from __future__ import annotations

import base64
import io
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping.store import full_tree_hash
from oh_my_slam.viewer.bundle import ViewBundle, image_bundle, map_bundle
from tests.browser.conftest import View
from tests.browser.scenes import running, synthetic_image, synthetic_map

pytestmark = [pytest.mark.browser]

LAYERS = ("points", "segments", "cameras", "labels", "obbs")
IMAGE_KEYS = ["color", "stride", "min-depth", "max-depth", "edge", "voxel", "normals"]
MAP_KEYS = ["color", "voxel", "normals"]
# Chromium logs every non-2xx fetch as a console error; only the invalid-input test causes one
BAD_REQUEST = "the server responded with a status of 400"


@pytest.fixture(scope="module")
def image_view(browser: Any, tmp_path_factory: pytest.TempPathFactory) -> Iterator[View]:
    img, client = synthetic_image(tmp_path_factory.mktemp("viewimg"))
    bundle = image_bundle(img, client)
    with running(bundle) as url:
        v = View(browser, bundle, url)
        v.client = client  # type: ignore[attr-defined]
        v.calls = dict(client.calls)  # type: ignore[attr-defined]
        yield v
        v.pg.close()


@pytest.fixture(scope="module")
def map_view(browser: Any, tmp_path_factory: pytest.TempPathFactory) -> Iterator[View]:
    if shutil.which("colmap") is None:
        pytest.skip("needs Homebrew colmap")
    root = synthetic_map(tmp_path_factory.mktemp("viewmap"))
    before = full_tree_hash(root)
    bundle = map_bundle(root)
    with running(bundle) as url:
        v = View(browser, bundle, url)
        v.root, v.before = root, before  # type: ignore[attr-defined]
        yield v
        v.pg.close()


@pytest.fixture(params=["image", "map"])
def view(request: pytest.FixtureRequest) -> View:
    return request.getfixturevalue(f"{request.param}_view")


def visibility(v: View) -> dict[str, bool]:
    return v.js("""() => {
      const g = window.__viewerGroups;
      return {points: g.points.visible, segments: g.segments.visible,
              cameras: g.cameras.visible, labels: g.labels.visible, obbs: g.obbs.visible};
    }""")


def set_only(v: View, on: set[str]) -> None:
    vis = visibility(v)
    for layer in LAYERS:
        if vis[layer] != (layer in on):
            v.pg.click(f'[data-layer="{layer}"] input')
    v.js("() => { window.__viewerGroups.points.parent.parent.background.setRGB(0, 0, 0);"
         " window.__viewerInvalidate(); }")  # changed behind the page's back: ask for a frame
    v.settle()


def canvas_image(v: View) -> np.ndarray:
    """The WebGL canvas only (no HTML overlays such as labels), as last drawn."""
    from PIL import Image

    url = v.js("() => document.querySelector('#canvas-host canvas').toDataURL('image/png')")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))


def canvas_pixels(v: View) -> set[tuple[int, int, int]]:
    return {tuple(p) for p in canvas_image(v).reshape(-1, 3).tolist()}


def object_colours(v: View) -> list[tuple[int, int, int]]:
    hexes = v.js("() => window.__viewer.objects.map(o => o.hex)")
    return [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in hexes]


def reset_view(v: View) -> None:
    v.js("() => window.__viewerResetView()")
    v.settle()


def restore_defaults(v: View) -> None:
    """A reload: the page comes back with every attribute and layer at its default (the bundle is
    in memory, so nothing is inferred again)."""
    v.pg.reload()
    v.pg.wait_for_selector('body[data-rendered="true"]', timeout=120000)
    v.settle()


# ------------------------------------------------------------------------------------------------


def test_rendered_signal_and_page_contents(view: View) -> None:
    v = view
    assert v.js("() => document.body.dataset.rendered") == "true"
    assert v.js("() => window.__viewer.ready") is True
    assert v.js("() => window.__viewerGroups.points.children[0].geometry"
                ".attributes.position.count") == v.js("() => window.__viewer.cloud.count") > 0
    assert v.pg.is_hidden("#loading") and v.pg.is_hidden("#cloud-error")
    assert v.pg.is_hidden("#cloud-note")  # within the display budget: every point, no notice
    assert v.js("() => window.__viewer.cloud.count") == v.js("() => window.__viewer.cloud.total")
    image = v.bundle.mode == "image"
    # spec §2.5: the catalogue and the segmented image are shown for an image only
    assert v.pg.locator("#catalogue tbody tr[data-id]").count() == (
        len(v.bundle.catalog) if image else 0)
    assert v.pg.is_visible("#tab-catalogue-btn") == image
    assert v.pg.is_visible("#tab-image-btn") == image
    assert v.errors == []


def test_each_toggle_affects_only_its_layer(view: View) -> None:
    v = view
    for layer in LAYERS:
        before = visibility(v)
        v.pg.click(f'[data-layer="{layer}"] input')
        after = visibility(v)
        changed = {k for k in LAYERS if before[k] != after[k]}
        assert changed == {layer}, (layer, changed)
        v.pg.click(f'[data-layer="{layer}"] input')  # restore
        assert visibility(v) == before
    assert v.errors == []


FRAMES = "() => window.__viewer.frames"


def frames_while_idle(v: View, ms: int = 2000) -> int:
    """Frames drawn while the page is left alone for ``ms``."""
    before = v.js(FRAMES)
    v.pg.wait_for_timeout(ms)
    return int(v.js(FRAMES) - before)


def wait_until_idle(v: View, timeout_s: float = 15.0) -> None:
    """Until no frame is drawn for 500 ms (e.g. once an orbit's damping has settled)."""
    deadline = time.monotonic() + timeout_s
    while frames_while_idle(v, 500):
        assert time.monotonic() < deadline, "the page keeps drawing"


def canvas_point(v: View) -> tuple[float, float]:
    """A point of the view where no label, header or help line covers the canvas."""
    at = v.js("""() => {
      const c = document.querySelector('#canvas-host canvas'), r = c.getBoundingClientRect();
      for (let fy = 0.5; fy < 0.9; fy += 0.02) for (let fx = 0.2; fx < 0.7; fx += 0.02) {
        const x = r.left + fx * r.width, y = r.top + fy * r.height;
        if (document.elementFromPoint(x, y) === c) return [x, y];
      }
      return null;
    }""")
    assert at, "no free point on the canvas"
    return at[0], at[1]


def test_frames_are_drawn_only_when_something_changes(view: View) -> None:
    """An idle page draws nothing (a large cloud would keep the GPU busy, next to the inference
    server); orbiting, a layer toggle and a colour change each draw, and the canvas shows them."""
    v = view
    v.pg.click('#tabs button[data-tab="controls"]')
    reset_view(v)
    wait_until_idle(v)
    assert frames_while_idle(v) == 0
    assert v.js("() => document.body.dataset.rendered") == "true"
    # orbiting: drawn while the view moves and its damping settles, then nothing
    x, y = canvas_point(v)
    before, f0 = canvas_image(v), v.js(FRAMES)
    v.pg.mouse.move(x, y)
    v.pg.mouse.down()
    for i in range(1, 11):
        v.pg.mouse.move(x + 6 * i, y + 2 * i)
    v.pg.mouse.up()
    wait_until_idle(v)
    assert v.js(FRAMES) - f0 >= 10
    assert not np.array_equal(canvas_image(v), before)
    assert frames_while_idle(v) == 0
    # a layer toggle
    before, f0 = canvas_image(v), v.js(FRAMES)
    v.pg.click('[data-layer="points"] input')
    v.settle()
    assert v.js(FRAMES) > f0 and not np.array_equal(canvas_image(v), before)
    v.pg.click('[data-layer="points"] input')
    v.settle()
    # a colour change: drawn again once the re-derived cloud has arrived, not only on the input
    before, f0 = canvas_image(v), v.js(FRAMES)
    v.pg.select_option("#attr-color", "height")
    v.settle()
    assert "color=height" in v.js("() => window.__viewer.cloud.attrs")
    assert v.js(FRAMES) > f0 and not np.array_equal(canvas_image(v), before)
    wait_until_idle(v)
    assert frames_while_idle(v) == 0
    restore_defaults(v)
    reset_view(v)
    assert v.errors == []


def test_controls_are_the_applicable_attributes(view: View) -> None:
    v = view
    keys = v.js("() => [...document.querySelectorAll('[data-attr]')].map(e => e.dataset.attr)")
    assert keys == (IMAGE_KEYS if v.bundle.mode == "image" else MAP_KEYS)
    assert v.pg.inner_text("#attr-max-depth + output" if v.bundle.mode == "image"
                           else "#attr-voxel + output") in ("∞", "off")


def test_exact_obb_and_segment_colours(view: View) -> None:
    v = view
    colours = object_colours(v)
    assert colours
    # material colours read back as the scene's hex values …
    assert v.js("() => window.__viewer.objects.every("
                "o => '#' + o.line.material.color.getHexString() === o.hex)")
    # … and drawn as exactly those sRGB triples
    set_only(v, {"obbs"})
    px = canvas_pixels(v)
    for rgb in colours:
        assert rgb in px, rgb
    # segmentation layer: only segmented points, each in its object's colour
    set_only(v, {"segments"})
    px = canvas_pixels(v)
    assert any(rgb in px for rgb in colours) and (128, 128, 128) not in px
    # point-cloud layer with color=segment: object colours, unsegmented points mid-grey
    v.pg.select_option("#attr-color", "segment")
    v.settle()
    assert "color=segment" in v.js("() => window.__viewer.cloud.attrs")
    set_only(v, {"points"})
    px = canvas_pixels(v)
    assert (128, 128, 128) in px and any(rgb in px for rgb in colours)
    restore_defaults(v)
    assert v.errors == []


def test_controls_rederive_without_inference(image_view: View) -> None:
    v = image_view
    full = v.js("() => window.__viewer.cloud.count")
    v.pg.fill("#attr-stride", "2")
    assert v.js("() => window.__viewer.busy")  # at once, while the change is debounced
    v.settle()
    assert v.pg.inner_text("#attr-stride + output") == "2 px"
    head = v.js("() => window.__viewer.cloud")
    expected = v.bundle.cloud(v.bundle.parse_attrs([("stride", "2")]))
    assert head["count"] == len(expected.cloud) < full / 3
    assert "stride=2" in head["attrs"]
    for sel, value in (("#attr-voxel", "0.05"), ("#attr-edge", "0"), ("#attr-min-depth", "2")):
        v.pg.fill(sel, value)
    v.pg.select_option("#attr-color", "height")
    v.settle()
    assert v.js("() => window.__viewer.cloud.attrs") == \
        "color=height,stride=2,min-depth=2,max-depth=inf,edge=0,voxel=0.05,normals=off"
    assert v.client.calls == v.calls  # type: ignore[attr-defined]
    restore_defaults(v)
    assert v.js("() => window.__viewer.cloud.count") == full
    assert v.client.calls == v.calls  # type: ignore[attr-defined]
    assert v.errors == []


def test_normals_are_shown(view: View) -> None:
    """normals=on derives the normals and the points are shaded by them, so the attribute shows."""
    v = view
    set_only(v, {"points"})
    flat = canvas_pixels(v)
    v.pg.click("#attr-normals")
    assert v.pg.inner_text("#attr-normals + output") == "on"
    v.settle()
    assert v.js("() => !!window.__viewerGroups.points.children[0].geometry.attributes.normal")
    assert canvas_pixels(v) != flat
    restore_defaults(v)
    assert v.errors == []


def test_invalid_input_is_reported_in_the_page(image_view: View) -> None:
    v = image_view
    count = v.js("() => window.__viewer.cloud.count")
    v.pg.fill("#attr-min-depth", "3")
    v.pg.fill("#attr-max-depth", "2")
    v.settle()
    msg = v.pg.inner_text("#cloud-error")
    assert "min-depth (3) must be smaller than max-depth (2)" in msg
    assert "invalid" in (v.pg.get_attribute('[data-attr="min-depth"]', "class") or "")
    assert v.js("() => window.__viewer.cloud.count") == count  # the last good cloud stays
    restore_defaults(v)
    assert v.pg.is_hidden("#cloud-error")
    assert [e for e in v.errors if BAD_REQUEST not in e] == []
    v.errors.clear()


def test_camera_frustums_at_the_scene_poses(view: View) -> None:
    v = view
    cams = v.js("() => window.__viewer.meta.cameras")
    verts = v.js("() => window.__viewerGroups.cameras.children[0].geometry.attributes.position.count")
    assert verts == len(cams) * 16 > 0  # 8 segments (2 vertices each) per frustum
    assert [c["T"] for c in cams] == [c["T"] for c in v.bundle.cameras]
    if v.bundle.mode == "image":
        assert len(cams) == 1 and v.pg.inner_text('[data-layer="cameras"]').startswith(
            "Camera pose")
    else:
        from oh_my_slam.mapping.store import MapReader

        assert len(cams) == len(MapReader(v.root).frames)  # type: ignore[attr-defined]
    assert visibility(v)["cameras"]


def test_catalogue_lists_every_object_in_its_colour(image_view: View) -> None:
    v = image_view
    v.pg.click('#tabs button[data-tab="catalogue"]')
    rows = v.js("""() => [...document.querySelectorAll('#catalogue tbody tr[data-id]')].map(tr => ({
      id: Number(tr.dataset.id), label: tr.children[2].textContent,
      hex: tr.querySelector('.swatch').title}))""")
    objects = v.js("() => window.__viewer.objects.map(o => ({id: o.id, label: o.label, hex: o.hex}))")
    assert sorted(rows, key=lambda r: r["id"]) == sorted(objects, key=lambda r: r["id"])
    v.pg.click('#tabs button[data-tab="controls"]')
    assert v.errors == []


def test_layouts(view: View, tmp_path: Path) -> None:
    v = view
    out = Path(os.environ.get("OH_MY_SLAM_VIEWER_SHOTS") or tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in {"desktop": (1440, 900), "narrow": (820, 700), "phone": (390, 844)}.items():
        v.pg.set_viewport_size({"width": w, "height": h})
        v.settle()
        v.pg.wait_for_timeout(250)
        box = v.js("""() => {
          const r = (s) => document.querySelector(s).getBoundingClientRect();
          return {scroll: document.documentElement.scrollWidth, canvas: r('#canvas-host canvas'),
                  panel: r('#panel')};
        }""")
        assert box["scroll"] <= w, name  # no horizontal overflow
        assert box["canvas"]["width"] > 0.3 * w and box["canvas"]["height"] > 0.3 * h, name
        assert box["panel"]["right"] <= w + 1 and box["panel"]["width"] > 0, name
        # every catalogue column fits the panel: no sideways scrolling, numbers not clipped
        if v.bundle.mode == "image":
            v.pg.click('#tabs button[data-tab="catalogue"]')
            cat = v.js("""() => {
              const w = document.querySelector('#tab-catalogue .table-wrap');
              const cells = [...document.querySelectorAll('#catalogue td.num')];
              return {scroll: w.scrollWidth, client: w.clientWidth,
                      clipped: cells.filter(c => c.offsetParent && c.scrollWidth > c.clientWidth + 1).length};
            }""")
            assert cat["scroll"] <= cat["client"] and cat["clipped"] == 0, (name, cat)
            v.pg.click('#tabs button[data-tab="controls"]')
        v.pg.screenshot(path=str(out / f"{v.bundle.mode}-{name}.png"))
    v.pg.set_viewport_size({"width": 1280, "height": 800})
    v.settle()
    assert v.errors == []


# ------------------------------------------------------------------------------------------------
# cameras: listed with their centres, "Go to" moves the viewpoint there (spec §2.5)


def coord(x: float) -> str:
    return f"{0.0 if abs(x) < 5e-4 else x:.3f}"


def display_frame(v: View, cam: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre and optical axis of a served camera, in the page's display frame."""
    M = np.asarray(v.js("() => window.__viewer.meta.display_transform")) @ np.asarray(cam["T"])
    return M[:3, 3], M[:3, 2]


def viewer_camera(v: View) -> dict[str, Any]:
    return v.js("""() => {
      const c = window.__viewerCamera, d = c.getWorldDirection(c.position.clone());
      return {pos: c.position.toArray(), dir: d.toArray(), fov: c.fov};
    }""")


def go_to(v: View, index: int) -> dict[str, Any]:
    v.pg.click('#tabs button[data-tab="cameras"]')
    v.pg.click(f'#cameras tr[data-index="{index}"] button.goto')
    v.settle()
    return viewer_camera(v)


def assert_at_camera(v: View, index: int, cam_state: dict[str, Any]) -> None:
    cam = v.js("() => window.__viewer.meta.cameras")[index]
    centre, axis = display_frame(v, cam)
    assert np.linalg.norm(np.asarray(cam_state["pos"]) - centre) < 1e-3  # within 1 mm
    assert float(np.dot(cam_state["dir"], axis)) > 0.9999  # looking along the optical axis
    fx, fy = cam["K"][:2]
    w, h = cam["size"]
    aspect = v.js("() => window.__viewerCamera.aspect")
    fov = np.degrees(2 * np.arctan(max(h / (2 * fy), w / (2 * fx) / aspect)))
    assert abs(cam_state["fov"] - fov) < 1e-6  # the camera's whole field in view


def test_cameras_are_listed_with_their_centres(view: View) -> None:
    v = view
    v.pg.click('#tabs button[data-tab="cameras"]')
    cams = v.js("() => window.__viewer.meta.cameras")
    rows = v.pg.locator("#cameras tbody tr[data-index]")
    assert rows.count() == len(cams) == len(v.bundle.cameras) > 0
    for i, cam in enumerate(cams):
        cells = rows.nth(i).locator("td")
        assert cells.nth(0).inner_text().splitlines()[0] == cam["name"]
        assert [cells.nth(k).inner_text() for k in (1, 2, 3)] == [coord(x) for x in cam["position"]]
        assert cam["position"] == v.bundle.cameras[i]["position"]
    note = v.pg.inner_text("#cam-note")
    assert ("map frame" in note) if v.bundle.mode == "map" else ("camera frame" in note)
    v.pg.click('#tabs button[data-tab="controls"]')
    assert v.errors == []


def test_go_to_camera(view: View, tmp_path: Path) -> None:
    v = view
    reset_view(v)
    index = len(v.bundle.cameras) // 2
    assert_at_camera(v, index, go_to(v, index))
    out = Path(os.environ.get("OH_MY_SLAM_VIEWER_SHOTS") or tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    v.pg.screenshot(path=str(out / f"{v.bundle.mode}-go-to-camera-desktop.png"))
    # phone width: the list sits under the view; going to a camera fits the new aspect
    v.pg.set_viewport_size({"width": 390, "height": 844})
    v.settle()
    assert_at_camera(v, index, go_to(v, index))
    v.pg.screenshot(path=str(out / f"{v.bundle.mode}-go-to-camera-phone.png"))
    v.pg.set_viewport_size({"width": 1280, "height": 800})
    reset_view(v)
    v.pg.click('#tabs button[data-tab="controls"]')
    v.settle()
    assert v.errors == []


@pytest.fixture(scope="module")
def ring_view(browser: Any) -> Iterator[View]:
    """100 cameras on a circle around a small cloud (no objects), to exercise a long list."""
    from oh_my_slam.core.types import Intrinsics
    from oh_my_slam.schema import openlabel as ol
    from oh_my_slam.segmentation.cloud import map_cloud_source
    from tests.synth.scene import look_at

    K = Intrinsics(320.0, 320.0, 320.0, 240.0, 640, 480)
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    poses = [look_at(np.array([3 * np.cos(a), 3 * np.sin(a), 1.2]), np.array([0.0, 0.0, 0.8]))
             for a in angles]
    frames = {str(k): ol.frame(float(k), stream_uris={"camera_0": f"frames/f{k:06d}.jpg"},
                               transforms={"camera_0_to_map": ol.transform("camera_0", "map", p)},
                               keyframe=f"f{k:06d}", update_id=1)
              for k, p in enumerate(poses)}
    scene = ol.document(ol.metadata("ring"), {},
                        coordinate_systems={"map": ol.map_cs(["camera_0"]),
                                            "camera_0": ol.sensor_cs("map")},
                        streams={"camera_0": ol.camera_stream(K)}, frames=frames)
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(5000, 3)) * [1.0, 1.0, 0.4] + [0.0, 0.0, 0.8]
    source = map_cloud_source(xyz, rng.integers(0, 255, (5000, 3), np.uint8), None, set(),
                              np.zeros((1, 3)))
    bundle = ViewBundle(mode="map", title="ring", scene=scene, source=source, catalog=[])
    with running(bundle) as url:
        v = View(browser, bundle, url)
        yield v
        v.pg.close()


def test_long_camera_list(ring_view: View) -> None:
    v = ring_view
    v.pg.click('#tabs button[data-tab="cameras"]')
    wrap = "#tab-cameras .table-wrap"
    assert v.js(f"() => {{ const w = document.querySelector('{wrap}'); "
                "return w.scrollHeight > 2 * w.clientHeight; }")  # scrolls, the page does not
    assert v.js("() => document.documentElement.scrollHeight") <= 800
    assert_at_camera(v, 90, go_to(v, 90))
    reset_view(v)
    assert v.js("() => window.__viewerCamera.fov") == 55
    assert v.errors == []


def test_map_unchanged_after_viewing(map_view: View) -> None:
    assert full_tree_hash(map_view.root) == map_view.before  # type: ignore[attr-defined]


def test_empty_map_still_renders(browser: Any, tmp_path: Path) -> None:
    """A map without points, frames or objects: the page loads, signals and logs no errors."""
    from oh_my_slam.mapping import store

    root = tmp_path / "m"
    with store.MapTransaction(root) as tx:
        tx.write_json(store.FRAMES_JSON, {"frames": []})
        tx.commit({"update_count": 1})
    bundle = map_bundle(root)
    with running(bundle) as url:
        v = View(browser, bundle, url)
        assert v.js("() => window.__viewer.cloud.count") == 0
        assert v.pg.is_disabled("#layer-cameras")
        v.pg.fill("#attr-voxel", "0.1")
        v.settle()
        assert v.errors == []
        v.pg.close()


# ------------------------------------------------------------------------------------------------
# frustums, the default views, labels, the segmented image and the two object-colour displays


def test_map_overview_looks_down_on_the_scene_and_its_cameras(map_view: View) -> None:
    v = map_view
    reset_view(v)
    cam = viewer_camera(v)
    assert np.degrees(np.arcsin(-cam["dir"][2])) == pytest.approx(60, abs=0.5)  # bird's-eye
    inside = v.js("""() => window.__viewer.meta.cameras.every(f => {
      const p = new window.__viewerCamera.position.constructor(f.T[0][3], f.T[1][3], f.T[2][3])
        .applyMatrix4(window.__viewerGroups.points.parent.matrixWorld).project(window.__viewerCamera);
      return Math.abs(p.x) < 1 && Math.abs(p.y) < 1 && p.z < 1;
    })""")
    assert inside  # every camera centre is in view
    assert v.errors == []


def test_a_stray_camera_does_not_shrink_the_overview(map_view: View, browser: Any,
                                                     tmp_path: Path) -> None:
    """A keyframe posed far off (a failed registration) is drawn, but the overview frames the
    cloud: framing it too would shrink the whole cloud to a dot."""
    import json

    root = tmp_path / "stray"
    shutil.copytree(map_view.root, root)  # type: ignore[attr-defined]
    doc = json.loads((root / "frames.json").read_text())
    doc["frames"][0]["T_map_cam"]["translation"] = [10000.0, 0.0, 0.0]
    (root / "frames.json").write_text(json.dumps(doc))
    bundle = map_bundle(root)
    with running(bundle) as url:
        v = View(browser, bundle, url)
        v.settle()
        cams = v.js("() => window.__viewer.meta.cameras.map(f => [f.T[0][3], f.T[1][3], f.T[2][3]])")
        assert max(abs(c[0]) for c in cams) == pytest.approx(10000.0)  # still there, still drawn
        reset_view(map_view)
        cam, home = viewer_camera(v), viewer_camera(map_view)
        dist = np.linalg.norm(np.subtract(cam["pos"], v.js(
            "() => window.__viewerControls.target.toArray()")))
        assert dist < 100  # the scene is a few metres across: framed, not seen from 10 km
        assert dist == pytest.approx(np.linalg.norm(np.subtract(home["pos"], map_view.js(
            "() => window.__viewerControls.target.toArray()"))), rel=0.5)
        assert v.errors == []
        v.pg.close()


def label_boxes(v: View) -> list[dict[str, Any]]:
    return v.js("""() => {
      const host = document.querySelector('#canvas-host').getBoundingClientRect();
      const cam = window.__viewerCamera;
      return window.__viewer.objects.map(o => {
        const p = o.anchor.clone().project(cam);
        const inView = p.z < 1 && p.z > -1 && Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1;
        const r = o.div.getBoundingClientRect();
        return {id: o.id, inView, shown: !o.div.hidden, mode: o.mode, text: o.div.innerText,
                rect: [r.left, r.top, r.right, r.bottom],
                inside: r.left >= host.left - 0.5 && r.right <= host.right + 0.5
                        && r.top >= host.top - 0.5 && r.bottom <= host.bottom + 0.5};
      });
    }""")


def intersects(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_every_box_in_view_is_labelled(view: View) -> None:
    """Spec §2.5 "labelled OBBs": each box whose top is in view shows its id and label inside the
    view, and no label covers another (these scenes have room for every label)."""
    v = view
    for w, h in ((1280, 800), (390, 844)):
        v.pg.set_viewport_size({"width": w, "height": h})
        reset_view(v)
        boxes = label_boxes(v)
        assert boxes and any(b["inView"] for b in boxes)
        for b in boxes:
            assert b["shown"] == b["inView"], b
            if b["shown"]:
                assert b["inside"] and b["text"].split()[0] == str(b["id"]), b
        assert any(b["mode"] == "full" for b in boxes)
        shown = [b for b in boxes if b["shown"]]
        for i, a in enumerate(shown):
            assert not any(intersects(a["rect"], b["rect"]) for b in shown[i + 1:]), (w, a)
    v.pg.set_viewport_size({"width": 1280, "height": 800})
    v.settle()
    assert v.errors == []


DENSE_LABELS = ("cup", "chair", "dining table", "bottle", "potted plant", "person", "plate", "lamp")


@pytest.fixture(scope="module")
def dense_view(browser: Any) -> Iterator[View]:
    """120 small boxes crowded in the middle of a 6 x 6 m floor: more labels than fit next to
    their boxes at desktop and at phone width."""
    from oh_my_slam.schema import openlabel as ol
    from oh_my_slam.segmentation.cloud import map_cloud_source
    from oh_my_slam.segmentation.colors import color_for_id, color_hex_for_id

    rng = np.random.default_rng(5)
    objects = {}
    for k in range(1, 121):
        centre = np.array([rng.uniform(-0.9, 0.9), rng.uniform(-0.6, 0.6), 0.0])
        size = rng.uniform(0.05, 0.14, 3)
        centre[2] = size[2] / 2
        w, d, h = (float(s) for s in size)
        cub = ol.cuboid(centre, np.eye(3), size, "map", attributes_num=[
            ol.num("width_m", w), ol.num("depth_m", d), ol.num("height_m", h),
            ol.num("volume_m3", w * d * h)])
        label = DENSE_LABELS[k % len(DENSE_LABELS)]
        objects[str(k)] = ol.object_entry(
            f"{label} {k}", label, "map", cub, nums=[ol.num("score", 0.9)],
            texts=[ol.text("color_hex", color_hex_for_id(k))], vecs=[ol.vec("color", list(color_for_id(k)))])
    scene = ol.document(ol.metadata("dense"), objects, coordinate_systems={"map": ol.map_cs([])})
    xyz = np.c_[rng.uniform(-3, 3, 40_000), rng.uniform(-3, 3, 40_000), rng.normal(0, 0.005, 40_000)]
    source = map_cloud_source(xyz, rng.integers(60, 200, (len(xyz), 3), np.uint8), None, set(),
                              np.array([[0.0, -4.0, 1.5]]))
    bundle = ViewBundle(mode="map", title="dense", scene=scene, source=source, catalog=[])
    with running(bundle) as url:
        v = View(browser, bundle, url)
        yield v
        v.pg.close()


def test_dense_labels_stay_legible(dense_view: View) -> None:
    """More boxes than room for their labels (120 boxes, desktop and phone): no shown label
    intersects another or leaves the view; every box whose top is in view either shows its id or,
    when no free place was left near it, is listed (id and label) under the Labels layer."""
    v = dense_view
    for w, h in ((1440, 900), (390, 844)):
        v.pg.set_viewport_size({"width": w, "height": h})
        reset_view(v)
        boxes = label_boxes(v)
        in_view = [b for b in boxes if b["inView"]]
        assert len(boxes) == 120 and len(in_view) >= 100, (w, len(in_view))
        shown = [b for b in boxes if b["shown"]]
        assert len(shown) >= 20 and all(b["inView"] for b in shown), w
        for i, a in enumerate(shown):
            assert a["inside"] and a["text"].split()[0] == str(a["id"]), a
            assert not any(intersects(a["rect"], b["rect"]) for b in shown[i + 1:]), (w, a)
        hidden = sorted(b["id"] for b in in_view if not b["shown"])
        note = v.pg.inner_text("#labels-note") if hidden else ""
        assert bool(hidden) == v.pg.is_visible("#labels-note"), w
        listed = f" {note.removeprefix('No room for:')} ".replace(",", " ")
        for oid in hidden[:20]:  # the first 20, then how many more
            assert f" {oid} " in listed, oid
        for oid in hidden[20:]:
            assert f" {oid} " not in listed, oid
        assert (f"…and {len(hidden) - 20} more" in note) == (len(hidden) > 20), note
        # not a live region: re-placing labels on every orbit announces nothing
        assert v.pg.get_attribute("#labels-note", "aria-live") == "off"
        assert v.pg.get_attribute("#labels-note", "role") is None
    v.pg.set_viewport_size({"width": 1280, "height": 800})
    v.settle()
    assert v.errors == []


def test_segment_colours_stay_exact_with_normals(view: View) -> None:
    """color=segment with normals=on: the object colours and the unsegmented grey reach the
    canvas exactly (normals shade the other colourings only)."""
    v = view
    set_only(v, {"points"})
    v.pg.select_option("#attr-color", "segment")
    v.settle()
    pre = [rgb for rgb in object_colours(v) if rgb in canvas_pixels(v)]
    v.pg.click("#attr-normals")
    v.settle()
    attrs = v.js("() => window.__viewer.cloud.attrs")
    assert "color=segment" in attrs and "normals=on" in attrs
    assert v.js("() => !!window.__viewerGroups.points.children[0].geometry.attributes.normal")
    assert v.js("() => window.__viewerGroups.points.children[0].material.uniforms.shade.value") == 0
    px = canvas_pixels(v)
    assert (128, 128, 128) in px
    seen = [rgb for rgb in object_colours(v) if rgb in px]
    assert seen and seen == pre, (seen, pre)  # every object colour seen without normals
    restore_defaults(v)
    assert v.errors == []


def test_segmented_image_is_shown(image_view: View) -> None:
    v = image_view
    v.pg.click("#tab-image-btn")
    assert v.pg.is_visible("#segmented")
    v.pg.wait_for_function("() => document.querySelector('#segmented').naturalWidth > 0")
    assert v.js("() => document.querySelector('#segmented').naturalWidth") == 320
    v.pg.click('#tabs button[data-tab="controls"]')
    assert v.errors == []
