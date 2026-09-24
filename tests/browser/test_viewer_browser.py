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
        v.pg.screenshot(path=str(out / f"{v.bundle.mode}-{name}.png"))
    v.pg.set_viewport_size({"width": 1280, "height": 800})
    v.settle()
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
