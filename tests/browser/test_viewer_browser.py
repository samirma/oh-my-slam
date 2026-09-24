"""Viewer in a real browser (``-m browser``; Microsoft Edge/Chromium through Playwright):
toggles affect only their layer, exact colours on screen, one frustum per keyframe, catalogue ↔
3D selection, no console errors, map unchanged by viewing (AC17-AC19)."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping.api import update
from oh_my_slam.mapping.store import full_tree_hash
from oh_my_slam.viewer.bundle import map_bundle
from oh_my_slam.viewer.server import serve, url_of
from tests.fakes.client import FakeClient
from tests.synth.mapping import add_frames, mapping_room, ring

pytestmark = [pytest.mark.browser,
              pytest.mark.skipif(shutil.which("colmap") is None, reason="needs colmap")]

CHANNELS = ("msedge", "chrome")
LAYERS = ("points", "segments", "cameras", "labels", "obbs")


@pytest.fixture(scope="module")
def synth_map(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("viewmap")
    client = FakeClient()
    add_frames(client, mapping_room(), ring(12), base / "in", "v", seed=11)
    update(base / "map", [base / "in"], client=client, progress=lambda m: None)
    return base / "map"


@pytest.fixture(scope="module")
def page(synth_map: Path) -> Iterator[tuple[Any, list[str], Path, str]]:
    sync_api = pytest.importorskip("playwright.sync_api")
    before = full_tree_hash(synth_map)
    bundle = map_bundle(synth_map)
    httpd = serve(bundle, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = None
        for ch in CHANNELS:
            try:
                browser = p.chromium.launch(channel=ch, headless=True)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("no Chromium-based browser (Edge/Chrome) available")
        pg = browser.new_page(viewport={"width": 1280, "height": 800})
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(url_of(httpd))
        pg.wait_for_function("window.__viewer && window.__viewer.ready === true", timeout=120000)
        yield pg, errors, synth_map, before
        browser.close()
    httpd.shutdown()
    httpd.server_close()


def visibility(pg: Any) -> dict[str, bool]:
    return pg.evaluate("""() => {
      const g = window.__viewerGroups;
      return {points: g.points.visible, segments: g.segments.visible,
              cameras: g.cameras.visible, labels: g.labels.visible, obbs: g.obbs.visible};
    }""")


def test_each_toggle_affects_only_its_layer(page: Any) -> None:
    pg, errors, _, _ = page
    for layer in LAYERS:
        before = visibility(pg)
        pg.click(f'[data-layer="{layer}"] input')
        after = visibility(pg)
        changed = {k for k in LAYERS if before[k] != after[k]}
        assert changed == {layer}, (layer, changed)
        pg.click(f'[data-layer="{layer}"] input')  # restore
        assert visibility(pg) == before
    assert errors == []


def _set_only(pg: Any, on: set[str]) -> None:
    vis = visibility(pg)
    for layer in LAYERS:
        if vis[layer] != (layer in on):
            pg.click(f'[data-layer="{layer}"] input')
    pg.evaluate("() => { window.__viewer.display.background = '#000000'; }")
    pg.evaluate("() => { const s = window.__viewerGroups.points.parent.parent; "
                "s.background.setRGB(0, 0, 0); }")
    pg.wait_for_timeout(400)


def _pixels(pg: Any, tmp: Path) -> np.ndarray:
    from PIL import Image

    path = tmp / "shot.png"
    pg.locator("#canvas-host canvas").first.screenshot(path=str(path))
    return np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)


def test_exact_obb_and_segment_colours(page: Any, tmp_path: Path) -> None:
    pg, errors, _, _ = page
    objs = pg.evaluate("() => window.__viewer.objects.map(o => [o.id, o.hex])")
    assert objs
    _set_only(pg, {"obbs"})
    px = {tuple(p) for p in _pixels(pg, tmp_path)}
    for oid, hx in objs:
        rgb = tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))
        assert rgb in px, (oid, hx)  # OBB edge pixels in exactly the object's colour
    _set_only(pg, {"segments"})
    px = {tuple(p) for p in _pixels(pg, tmp_path)}
    assert (128, 128, 128) in px  # unsegmented points exactly mid-grey
    assert any(tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5)) in px for _, hx in objs)
    _set_only(pg, {"points", "cameras", "labels", "obbs"})
    assert errors == []


def test_one_frustum_per_keyframe_and_selection(page: Any) -> None:
    pg, errors, synth_map, _ = page
    n_frames = pg.evaluate("() => window.__viewer.meta.frustums.length")
    verts = pg.evaluate("() => window.__viewerGroups.cameras.children[0]"
                        ".geometry.attributes.position.count")
    assert verts == n_frames * 16  # 8 segments (2 vertices each) per frustum
    from oh_my_slam.mapping.store import MapReader

    assert n_frames == len(MapReader(synth_map).frames)
    first = pg.locator("#catalogue tbody tr").first
    pg.click('#tabs button[data-tab="catalogue"]')
    oid = int(first.get_attribute("data-id"))
    first.click()
    assert pg.evaluate("() => window.__viewer.selected") == oid
    assert "selected" in (first.get_attribute("class") or "")
    width = pg.evaluate(f"() => window.__viewer.objects.find(o => o.id === {oid})"
                        ".line.material.linewidth")
    assert width > 2
    stats = pg.inner_text("#stats")
    assert "points" in stats and "objects" in stats and "frames" in stats
    pg.keyboard.press("r")
    assert errors == []


def test_map_unchanged_after_viewing(page: Any) -> None:
    _, _, synth_map, before = page
    assert full_tree_hash(synth_map) == before
