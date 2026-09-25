"""Viewer in a real browser (``-m browser``; Microsoft Edge/Chromium through Playwright), for an
image (fake inference client) and a map (real mapper, COLMAP): the ``data-rendered`` signal, layer
toggles that affect only their layer, attribute controls taken from the shared attribute table that
re-derive the cloud without inference, exact §2.4 colours on screen, camera frustums at the scene's
poses, catalogue ↔ 3D selection, errors shown in the page, no console errors, layouts at desktop
and phone widths, and the map unchanged by viewing.

Set ``OH_MY_SLAM_VIEWER_SHOTS=<folder>`` to keep the layout screenshots."""

from __future__ import annotations

import base64
import io
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping.store import full_tree_hash
from oh_my_slam.viewer.bundle import ViewBundle, image_bundle, map_bundle
from tests.browser.scenes import running, synthetic_image, synthetic_map

pytestmark = [pytest.mark.browser]

CHANNELS = ("msedge", "chrome")
LAYERS = ("points", "segments", "cameras", "labels", "obbs")
IMAGE_KEYS = ["color", "stride", "min-depth", "max-depth", "edge", "voxel", "normals"]
MAP_KEYS = ["color", "voxel", "normals"]
# Chromium logs every non-2xx fetch as a console error; only the invalid-input test causes one
BAD_REQUEST = "the server responded with a status of 400"


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = None
        for ch in CHANNELS:
            try:
                b = p.chromium.launch(channel=ch, headless=True)
                break
            except Exception:
                continue
        if b is None:
            pytest.skip("no Chromium-based browser (Edge/Chrome) available")
        yield b
        b.close()


class View:
    """A page showing one bundle, with the console errors it logged."""

    def __init__(self, browser: Any, bundle: ViewBundle, url: str) -> None:
        self.bundle, self.url, self.errors = bundle, url, []
        self.pg = browser.new_page(viewport={"width": 1280, "height": 800})
        self.pg.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.pg.on("pageerror", lambda e: self.errors.append(str(e)))
        self.pg.goto(url)
        self.pg.wait_for_selector('body[data-rendered="true"]', timeout=120000)

    def js(self, expr: str) -> Any:
        return self.pg.evaluate(expr)

    def settle(self) -> None:
        """Wait for any pending re-derivation and two drawn frames."""
        self.pg.wait_for_function("() => !window.__viewer.busy", timeout=60000)
        self.js("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


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
    v.js("() => { window.__viewerGroups.points.parent.parent.background.setRGB(0, 0, 0); }")
    v.settle()


def canvas_pixels(v: View) -> set[tuple[int, int, int]]:
    """The WebGL canvas only (no HTML overlays such as labels)."""
    from PIL import Image

    url = v.js("() => document.querySelector('#canvas-host canvas').toDataURL('image/png')")
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    return {tuple(p) for p in np.asarray(img).reshape(-1, 3).tolist()}


def object_colours(v: View) -> list[tuple[int, int, int]]:
    hexes = v.js("() => window.__viewer.objects.map(o => o.hex)")
    return [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in hexes]


def restore_defaults(v: View) -> None:
    v.pg.click("#cloud-defaults")
    v.settle()
    set_only(v, {"points", "cameras", "labels", "obbs"})


# ------------------------------------------------------------------------------------------------


def test_rendered_signal_and_page_contents(view: View) -> None:
    v = view
    assert v.js("() => document.body.dataset.rendered") == "true"
    assert v.js("() => window.__viewer.ready") is True
    assert v.js("() => window.__viewerGroups.points.children[0].geometry"
                ".attributes.position.count") == v.js("() => window.__viewer.cloud.count") > 0
    stats = v.pg.inner_text("#stats")
    assert "points" in stats and "objects" in stats and "frame" in stats
    assert v.pg.is_hidden("#loading") and v.pg.is_hidden("#cloud-error")
    n = len(v.bundle.catalog)
    assert v.pg.locator("#catalogue tbody tr[data-id]").count() == n
    assert v.pg.is_visible("#tab-image-btn") == (v.bundle.mode == "image")
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


def test_controls_are_the_applicable_attributes(view: View) -> None:
    v = view
    keys = v.js("() => [...document.querySelectorAll('[data-attr]')].map(e => e.dataset.attr)")
    assert keys == (IMAGE_KEYS if v.bundle.mode == "image" else MAP_KEYS)
    assert v.pg.inner_text("#cloud-cli") == "default attributes"
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
    assert v.pg.is_visible("#busy")  # shown at once, while the change is debounced
    v.settle()
    assert v.pg.is_hidden("#busy") and v.pg.inner_text("#attr-stride + output") == "2 px"
    head = v.js("() => window.__viewer.cloud")
    expected = v.bundle.cloud(v.bundle.parse_attrs([("stride", "2")]))
    assert head["count"] == len(expected.cloud) < full / 3
    assert "stride=2" in head["attrs"] and v.pg.inner_text("#cloud-cli") == "-p stride=2"
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
    v = view
    assert v.pg.is_disabled("#display-normals")
    v.pg.click("#attr-normals")
    assert v.pg.inner_text("#attr-normals + output") == "on"
    v.settle()
    assert v.js("() => !!window.__viewerGroups.points.children[0].geometry.attributes.normal")
    assert v.pg.is_enabled("#display-normals")
    set_only(v, {"points"})
    shaded = canvas_pixels(v)
    v.pg.select_option("#display-normals", "normals")
    v.settle()
    assert v.js("() => window.__viewerGroups.points.children[0].material.uniforms.shade.value") == 2
    assert canvas_pixels(v) != shaded
    v.pg.select_option("#display-normals", "shade")
    restore_defaults(v)
    assert v.pg.is_disabled("#display-normals")
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


def test_catalogue_selects_in_3d(view: View) -> None:
    v = view
    v.pg.click('#tabs button[data-tab="catalogue"]')
    first = v.pg.locator("#catalogue tbody tr").first
    oid = int(first.get_attribute("data-id"))
    first.click()
    assert v.js("() => window.__viewer.selected") == oid
    assert "selected" in (first.get_attribute("class") or "")
    assert v.js(f"() => window.__viewer.objects.find(o => o.id === {oid}).line.material.linewidth") > 2
    v.pg.keyboard.press("Escape")
    v.pg.keyboard.press("r")
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
        # the title and stats sit on their own scrim and never run under the buttons
        head = v.js("""() => {
          const r = (s) => document.querySelector(s).getBoundingClientRect();
          return {info: r('#info'), actions: r('#actions'),
                  scrim: getComputedStyle(document.querySelector('#info')).backgroundColor};
        }""")
        i, a = head["info"], head["actions"]
        assert i["right"] <= a["left"] or i["bottom"] <= a["top"], name
        assert head["scrim"] not in ("rgba(0, 0, 0, 0)", "transparent"), name
        # every catalogue column fits the panel: no sideways scrolling, numbers not clipped
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
    v.pg.wait_for_function("() => !window.__viewer.flying", timeout=10000)
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
    assert v.js("() => window.__viewer.cameraIndex") == index
    assert v.js("() => window.__viewerGroups.cameras.children.some(c => c.name === 'selected-camera')")


def test_cameras_are_listed_with_their_centres(view: View) -> None:
    v = view
    v.pg.click('#tabs button[data-tab="cameras"]')
    cams = v.js("() => window.__viewer.meta.cameras")
    rows = v.pg.locator("#cameras tbody tr[data-index]")
    assert rows.count() == len(cams) == len(v.bundle.cameras) > 0
    assert v.pg.inner_text("#cam-count") == str(len(cams))
    for i, cam in enumerate(cams):
        cells = rows.nth(i).locator("td")
        assert cells.nth(0).inner_text().splitlines()[0] == cam["name"]
        assert [cells.nth(k).inner_text() for k in (1, 2, 3)] == [coord(x) for x in cam["position"]]
        assert cam["position"] == v.bundle.cameras[i]["position"]
    note = v.pg.inner_text("#cam-note")
    assert ("map frame" in note) if v.bundle.mode == "map" else ("camera frame" in note)
    v.pg.click('#tabs button[data-tab="controls"]')
    assert v.errors == []


def test_go_to_camera_and_reset_view(view: View, tmp_path: Path) -> None:
    v = view
    v.pg.click("#reset-view")
    v.settle()
    home = viewer_camera(v)
    index = len(v.bundle.cameras) // 2
    assert_at_camera(v, index, go_to(v, index))
    out = Path(os.environ.get("OH_MY_SLAM_VIEWER_SHOTS") or tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    v.pg.screenshot(path=str(out / f"{v.bundle.mode}-go-to-camera-desktop.png"))
    # Reset view still restores the default view (and field of view)
    v.pg.click("#reset-view")
    v.settle()
    back = viewer_camera(v)
    assert back["fov"] == 55
    np.testing.assert_allclose(back["pos"], home["pos"], atol=1e-6)
    # phone width: the list sits under the view; going to a camera fits the new aspect
    v.pg.set_viewport_size({"width": 390, "height": 844})
    v.settle()
    assert_at_camera(v, index, go_to(v, index))
    v.pg.screenshot(path=str(out / f"{v.bundle.mode}-go-to-camera-phone.png"))
    # looking through the camera: its highlight and its frustum are faded out, not drawn across
    # the view
    assert v.js("() => window.__viewerFrustumFade(window.__viewer.cameraIndex)") == 0
    assert v.js("() => window.__viewerGroups.cameras.children"
                ".find(c => c.name === 'selected-camera').visible") is False
    v.pg.keyboard.press("Escape")
    assert v.js("() => window.__viewerGroups.cameras.children.length") == 1  # highlight gone
    v.pg.set_viewport_size({"width": 1280, "height": 800})
    v.pg.click("#reset-view")
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
    bundle = ViewBundle(mode="map", title="ring", scene=scene, source=source, catalog=[],
                        stats={"objects": 0, "frames": len(poses)})
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
    # "]" steps to the next camera; the selected row is kept in view
    v.pg.keyboard.press("]")
    v.pg.wait_for_function("() => !window.__viewer.flying", timeout=10000)
    v.settle()
    assert_at_camera(v, 91, viewer_camera(v))
    row = v.pg.locator('#cameras tr[data-index="91"]').bounding_box()
    box = v.pg.locator(wrap).bounding_box()
    assert row and box and box["y"] <= row["y"] and row["y"] + row["height"] <= box["y"] + box["height"]
    assert "selected" in (v.pg.get_attribute('#cameras tr[data-index="91"]', "class") or "")
    # the filter narrows the list; stepping wraps around the visible rows
    v.pg.fill("#cam-filter", "f00009")
    assert v.pg.locator("#cameras tbody tr[data-index]:visible").count() == 10
    v.pg.click("#cam-next")
    v.pg.wait_for_function("() => !window.__viewer.flying", timeout=10000)
    v.settle()
    assert v.js("() => window.__viewer.cameraIndex") == 92
    v.pg.click("#reset-view")
    v.settle()
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


def fades(v: View) -> list[float]:
    n = len(v.bundle.cameras)
    return v.js(f"() => [...Array({n}).keys()].map(i => window.__viewerFrustumFade(i))")


def test_frustums_fade_only_next_to_the_viewpoint(view: View) -> None:
    v = view
    v.pg.click("#reset-view")
    v.settle()
    if v.bundle.mode == "image":
        # the default view starts just behind the photo: its frustum does not cross the photo
        assert fades(v) == [0]
    else:
        assert min(fades(v)) == 1  # the overview draws every camera
        # looking through a camera, its neighbours fade too; far cameras stay drawn
        go_to(v, 0)
        cams = np.array([c["position"] for c in v.bundle.cameras])
        d = np.linalg.norm(cams - cams[0], axis=1)
        depth = v.js("() => window.__viewer.frustumDepth")
        f = fades(v)
        assert f[0] == 0
        assert all(a < 0.5 for a, x in zip(f, d, strict=True) if x < depth)
        assert all(a == 1 for a, x in zip(f, d, strict=True) if x > 6 * depth)
        v.pg.keyboard.press("Escape")
        v.pg.click("#reset-view")
        v.pg.click('#tabs button[data-tab="controls"]')
        v.settle()
    assert v.errors == []


def test_map_overview_looks_down_on_the_scene_and_its_cameras(map_view: View) -> None:
    v = map_view
    v.pg.click("#reset-view")
    v.settle()
    cam = viewer_camera(v)
    assert np.degrees(np.arcsin(-cam["dir"][2])) == pytest.approx(60, abs=0.5)  # bird's-eye
    inside = v.js("""() => window.__viewer.meta.cameras.every(f => {
      const p = new window.__viewerCamera.position.constructor(f.T[0][3], f.T[1][3], f.T[2][3])
        .applyMatrix4(window.__viewerGroups.points.parent.matrixWorld).project(window.__viewerCamera);
      return Math.abs(p.x) < 1 && Math.abs(p.y) < 1 && p.z < 1;
    })""")
    assert inside  # every camera centre is in view
    assert v.errors == []


def label_boxes(v: View) -> list[dict[str, Any]]:
    return v.js("""() => {
      const host = document.querySelector('#canvas-host').getBoundingClientRect();
      const cam = window.__viewerCamera;
      return window.__viewer.objects.map(o => {
        const p = o.anchor.clone().project(cam);
        const inView = p.z < 1 && p.z > -1 && Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1;
        const r = o.div.getBoundingClientRect();
        return {id: o.id, inView, shown: !o.div.hidden, mode: o.mode, text: o.div.innerText,
                inside: r.left >= host.left - 0.5 && r.right <= host.right + 0.5
                        && r.top >= host.top - 0.5 && r.bottom <= host.bottom + 0.5};
      });
    }""")


def test_every_box_in_view_is_labelled(view: View) -> None:
    """Spec §2.5 "labelled OBBs": each box whose top is in view shows its id tag inside the view
    (these scenes have room for every tag, so no "+N" chip); names are added where they fit, and
    always for the selected box."""
    v = view
    for w, h in ((1280, 800), (390, 844)):
        v.pg.set_viewport_size({"width": w, "height": h})
        v.pg.click("#reset-view")
        v.settle()
        boxes = label_boxes(v)
        assert boxes and any(b["inView"] for b in boxes)
        for b in boxes:
            assert b["shown"] == b["inView"], b
            if b["shown"]:
                assert b["inside"] and b["text"].split()[0] == str(b["id"]), b
        assert any(b["mode"] == "full" for b in boxes)
        assert v.js("() => window.__viewer.chips.length") == 0
    v.pg.set_viewport_size({"width": 1280, "height": 800})
    v.settle()
    # "id tags only" keeps every tag; the selected box still shows its name
    v.pg.select_option("#display-labels", "ids")
    oid = v.bundle.catalog[0]["id"]
    v.js(f"() => document.querySelector('.obj-label[data-id=\"{oid}\"]').click()")
    v.pg.click("#reset-view")
    v.settle()
    boxes = label_boxes(v)
    assert all(b["mode"] == ("full" if b["id"] == oid else "compact") for b in boxes if b["shown"])
    assert v.js("() => window.__viewer.selected") == oid
    v.pg.keyboard.press("Escape")
    v.pg.select_option("#display-labels", "fit")
    v.pg.click('#tabs button[data-tab="controls"]')
    v.settle()
    assert v.errors == []


DENSE_LABELS = ("cup", "chair", "dining table", "bottle", "potted plant", "person", "plate", "lamp")


@pytest.fixture(scope="module")
def dense_view(browser: Any) -> Iterator[View]:
    """120 small boxes crowded in the middle of a 6 x 6 m floor: more id tags than fit near their
    boxes at desktop and at phone width."""
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
    bundle = ViewBundle(mode="map", title="dense", scene=scene, source=source, catalog=[],
                        stats={"objects": len(objects), "frames": 0})
    with running(bundle) as url:
        v = View(browser, bundle, url)
        yield v
        v.pg.close()


def label_layout(v: View) -> dict[str, Any]:
    """Page-pixel rectangles of every visible id tag and "+N" chip, the header's, and each box's
    anchor (the top face's centre) with how it is labelled."""
    return v.js("""() => {
      const rect = (e) => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
      const cam = window.__viewerCamera, host = rect(document.querySelector('#canvas-host'));
      const chips = window.__viewer.chips;
      return {
        host, header: ['#info', '#actions'].map((s) => rect(document.querySelector(s))),
        boxes: window.__viewer.objects.map((o) => {
          const p = o.anchor.clone().project(cam);
          return {id: o.id, inView: p.z < 1 && p.z > -1 && Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1,
                  anchor: [host[0] + (p.x + 1) / 2 * (host[2] - host[0]),
                           host[1] + (1 - p.y) / 2 * (host[3] - host[1])],
                  tag: o.div.hidden ? null : rect(o.div), text: o.div.innerText,
                  chip: o.mode === 'cluster' ? chips.indexOf(o.chip) : null};
        }),
        chips: chips.map((c) => ({rect: rect(c.button), hidden: c.div.hidden, text: c.button.innerText,
                                  ids: c.members.map((o) => o.id)})),
      };
    }""")


def intersects(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# Pan the view (camera and orbit target, parallel to the image plane) so that the top-most box
# anchor on screen lands just inside the header card's bottom-right corner: the top of the crowd
# is then under the header.
UNDER_THE_HEADER = """() => {
  const c = window.__viewerCamera, t = window.__viewerControls.target, V = c.position.constructor;
  const host = document.querySelector('#canvas-host').getBoundingClientRect();
  const info = document.querySelector('#info').getBoundingClientRect();
  let top = null;
  for (const o of window.__viewer.objects) {
    const p = o.anchor.clone().project(c);
    if (!top || p.y > top.p.y) top = {o, p};
  }
  const sx = (top.p.x + 1) / 2 * host.width, sy = (1 - top.p.y) / 2 * host.height;
  const tx = info.right - host.left - 30, ty = info.bottom - host.top - 10;
  const right = new V().setFromMatrixColumn(c.matrixWorld, 0);
  const up = new V().setFromMatrixColumn(c.matrixWorld, 1);
  const fwd = new V().setFromMatrixColumn(c.matrixWorld, 2).negate();
  const z = top.o.anchor.clone().sub(c.position).dot(fwd);
  const f = host.height / (2 * Math.tan(c.fov * Math.PI / 360));
  const move = right.multiplyScalar(-(tx - sx) * z / f).add(up.multiplyScalar((ty - sy) * z / f));
  c.position.add(move);
  t.add(move);
}"""


def test_dense_labels_never_overlap(dense_view: View) -> None:
    """With more boxes than room for their tags (120 boxes, desktop and phone): no two visible
    tags or chips intersect, none intersects the header, every box whose top is in view is either
    tagged or listed by a visible "+N" chip, and a chip lists its boxes, each selecting its box."""
    v = dense_view
    for w, h in ((1440, 900), (390, 844)):
        v.pg.set_viewport_size({"width": w, "height": h})
        v.pg.click("#reset-view")
        v.settle()
        v.js(UNDER_THE_HEADER)
        v.settle()
        lay = label_layout(v)
        boxes = lay["boxes"]
        in_view = [b for b in boxes if b["inView"]]
        assert len(boxes) == 120 and len(in_view) >= 100, (w, len(in_view))
        # the header covers some boxes' anchors: their tags went elsewhere
        assert any(intersects([*b["anchor"], *b["anchor"]], hd) for b in in_view for hd in lay["header"])
        chips = [c for c in lay["chips"] if not c["hidden"]]
        assert len(chips) == len(lay["chips"]) > 0, w
        for b in boxes:
            if not b["inView"]:
                assert b["tag"] is None and b["chip"] is None, b
            elif b["tag"] is not None:
                assert b["chip"] is None and b["text"].split()[0] == str(b["id"]), b
            else:
                assert b["chip"] is not None and b["id"] in lay["chips"][b["chip"]]["ids"], b
        for c in chips:
            assert len(c["ids"]) >= 2 and c["text"] == f"+{len(c['ids'])}", c
        rects = [b["tag"] for b in boxes if b["tag"] is not None] + [c["rect"] for c in chips]
        host = lay["host"]
        for i, a in enumerate(rects):
            assert host[0] - 0.5 <= a[0] and a[2] <= host[2] + 0.5, a
            assert host[1] - 0.5 <= a[1] and a[3] <= host[3] + 0.5, a
            assert not any(intersects(a, hd) for hd in lay["header"]), (w, a)
            for b in rects[i + 1:]:
                assert not intersects(a, b), (w, a, b)
    # a chip lists its boxes; peeking at an entry thickens its box, clicking it selects the box
    k = max(range(len(chips)), key=lambda i: len(chips[i]["ids"]))
    ids = sorted(chips[k]["ids"])
    v.pg.locator(".obj-cluster:not([hidden]) .chip").nth(k).click()
    assert v.pg.is_visible("#cluster-pop")
    entries = v.pg.locator("#cluster-pop button")
    assert [int(entries.nth(i).get_attribute("data-id") or 0) for i in range(entries.count())] == ids
    assert entries.nth(0).inner_text().split()[0] == str(ids[0])
    pop = v.pg.locator("#cluster-pop").bounding_box()
    assert pop and not any(intersects([pop["x"], pop["y"], pop["x"] + pop["width"],
                                       pop["y"] + pop["height"]], hd) for hd in lay["header"])
    entries.nth(0).hover()
    assert v.js("() => window.__viewer.peek") == ids[0]
    assert v.js(f"() => window.__viewer.objects.find(o => o.id === {ids[0]}).line.material.linewidth") > 2
    entries.nth(0).click()
    assert v.pg.is_hidden("#cluster-pop")
    assert v.js("() => window.__viewer.selected") == ids[0]
    v.settle()
    assert v.js(f"() => window.__viewer.objects.find(o => o.id === {ids[0]}).mode") == "full"
    v.pg.keyboard.press("Escape")
    v.pg.set_viewport_size({"width": 1280, "height": 800})
    v.settle()
    assert v.errors == []


def test_boxes_around_the_viewpoint_fade_and_come_back_exact(view: View) -> None:
    v = view
    o = v.bundle.catalog[0]["id"]
    v.js(f"""() => {{
      const o = window.__viewer.objects.find(x => x.id === {o});
      const c = o.center.clone().applyMatrix4(window.__viewerGroups.points.parent.matrixWorld);
      window.__viewerCamera.position.copy(c);
      window.__viewerControls.target.copy(c.clone().add(new c.constructor(0.3, 0.3, 0)));
    }}""")
    v.settle()
    m = v.js(f"() => {{ const m = window.__viewer.objects.find(x => x.id === {o}).line.material;"
             " return [m.opacity, m.transparent]; }")
    assert m[0] == pytest.approx(0.2) and m[1] is True  # inside the box: its edges fade
    v.pg.click("#reset-view")
    v.settle()
    assert v.js("() => window.__viewer.objects.every(o => o.line.material.opacity === 1"
                " && !o.line.material.transparent)")  # opaque again: exact colours
    assert v.errors == []


def test_segmented_image_enlarges(image_view: View) -> None:
    v = image_view
    v.pg.click("#tab-image-btn")
    thumb = v.pg.locator("#segmented").bounding_box()
    v.pg.click("#segmented-open")
    assert v.pg.is_visible("#lightbox")
    # fitted to the window (small images are scaled up, pixels kept sharp)
    big = v.js("""() => { const i = document.querySelector('#lightbox img');
      const r = i.getBoundingClientRect(), s = Math.min(r.width / i.naturalWidth, r.height / i.naturalHeight);
      return {width: i.naturalWidth * s}; }""")
    assert thumb and big["width"] > 2 * thumb["width"]
    v.pg.click("#lightbox img")  # actual pixels
    natural = v.js("() => document.querySelector('#lightbox img').naturalWidth")
    shown = v.pg.locator("#lightbox img").bounding_box()
    assert shown and shown["width"] == pytest.approx(natural, abs=1)
    v.pg.keyboard.press("Escape")
    assert v.pg.is_hidden("#lightbox")
    v.pg.click('#tabs button[data-tab="controls"]')
    assert v.errors == []


def test_segmentation_layer_and_color_segment_are_told_apart(view: View) -> None:
    v = view
    v.pg.click('#tabs button[data-tab="controls"]')
    assert v.pg.inner_text('[data-layer="segments"]').startswith("Segmentation overlay")
    assert "over the cloud" in v.pg.inner_text("#layer-note-segments")
    assert v.pg.inner_text('#attr-color option[value="segment"]') == "segment · object colours"
    v.pg.select_option("#attr-color", "segment")
    v.settle()
    assert "color=segment" in v.pg.inner_text("#layer-note-segments")
    assert not v.pg.is_checked("#layer-segments")  # still its own, independent layer
    restore_defaults(v)
    assert "over the cloud" in v.pg.inner_text("#layer-note-segments")
    assert v.errors == []
