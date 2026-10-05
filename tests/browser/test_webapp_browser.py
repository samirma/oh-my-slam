"""The server.sh web application in a real browser (``-m browser``; http_server.md "Evaluation → UI"):
an image job, map creation and update, the viewer, the 3D scene viewer with a PLY and a JSON opened
together, downloads, cancellation, and an automatic accessibility check of every page; plus
selection sync, stable URLs, the registry-driven rendering and the layout down to tablet width.

The service is the real Starlette app and job runner (``tests.browser.webapp``) over a scratch
workspace, with the stub inference server (``tests.fakes.stub_server``); the inference server down
is ``test_webapp_down.py``."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.commands import spec
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.web import operations as web_ops
from tests.browser.test_viewer_modules import cloud as module_cloud
from tests.browser.test_viewer_modules import located, scene_doc
from tests.browser.webapp import axe_violations, running_service
from tests.fakes import slow_command
from tests.fakes.stub_server import start_stub_server
from tests.unit.test_view_cli import minimal_map, sh
from tests.unit.test_web_api import jpeg

pytestmark = [pytest.mark.browser]

REPO = Path(__file__).resolve().parents[2]
FRAMES = sorted((REPO / "examples" / "ainex-captures").glob("*.jpg"))
needs_colmap = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")
JOB_TIMEOUT_MS = 180_000


@pytest.fixture(scope="module")
def stub() -> Iterator[None]:
    start_stub_server()
    try:
        yield
    finally:
        sh("start_inference_server.sh", "--stop")


@pytest.fixture(scope="module")
def app(stub: None, tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, str]]:
    with pytest.MonkeyPatch.context() as mp:  # job subprocesses import tests.fakes
        mp.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
        with running_service(tmp_path_factory.mktemp("web") / "ws") as (service, url):
            yield service, url


class Tab:
    """A browser page on the web application, with the console errors it logged."""

    def __init__(self, browser: Any, base: str, width: int = 1280, height: int = 900,
                 scheme: str = "light") -> None:
        self.base, self.errors = base, []
        self.ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme,
                                       accept_downloads=True)
        self.pg = self.ctx.new_page()
        self.pg.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.pg.on("pageerror", lambda e: self.errors.append(str(e)))
        self.pg.on("response", lambda r: self.errors.append(f"{r.status} {r.request.method} {r.url}")
                   if r.status >= 400 else None)

    def go(self, route: str) -> Any:
        self.pg.goto(self.base + route)
        self.pg.wait_for_selector("body[data-ready=true]", timeout=30000)
        return self.pg

    def js(self, expr: str, arg: Any = None) -> Any:
        return self.pg.evaluate(expr, arg)

    def a11y(self) -> None:
        bad = axe_violations(self.pg)
        assert not bad, "\n".join(bad)

    def close(self) -> None:
        self.ctx.close()


@pytest.fixture
def tab(browser: Any, app: tuple[Any, str]) -> Iterator[Tab]:
    t = Tab(browser, app[1])
    yield t
    t.close()


def wait_job(pg: Any, state: str = "succeeded", timeout: int = JOB_TIMEOUT_MS) -> str:
    pg.wait_for_selector(f"section.job[data-state={state}]", timeout=timeout)
    return pg.get_attribute("section.job", "data-job")


def frame_viewer(pg: Any, expr: str) -> Any:
    """``expr`` (with ``v``, the embedded viewer's Viewer) in the embedded viewer page."""
    return pg.evaluate(f"() => {{ const v = document.querySelector('iframe.viewer-frame').contentWindow.__viewerApp; return {expr}; }}")


def assert_badges_visible(pg: Any, scope: str) -> None:
    """Every object badge (colour swatch, id, label) under ``scope`` is visible inside its cell or
    caption: its id and label are laid out within it, not lifted out of the flow."""
    bad = pg.evaluate("""(scope) => {
      const out = [];
      const badges = [...document.querySelectorAll(`${scope} .obj-badge`)];
      if (!badges.length) return ['no object badge under ' + scope];
      for (const b of badges) {
        const box = (b.closest('td, th, p, figcaption, li') || b.parentElement).getBoundingClientRect();
        for (const part of b.querySelectorAll('.obj-id, .obj-name')) {
          const r = part.getBoundingClientRect(), s = getComputedStyle(part);
          const inside = r.width > 0 && r.height > 0 && r.left >= box.left - 1 && r.right <= box.right + 1
            && r.top >= box.top - 1 && r.bottom <= box.bottom + 1;
          if (!inside || s.position === 'absolute' || s.visibility === 'hidden' || !part.textContent.trim()) {
            out.push(`${part.className} "${part.textContent}" outside its cell`);
          }
        }
      }
      return out;
    }""", scope)
    assert not bad, bad


# ---------------------------------------------------------------------------------- the image job


@pytest.fixture(scope="module")
def image_job(browser: Any, app: tuple[Any, str], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """segment.sh -i through the Image page: drop zone with preview, generated form (a field
    flagged before submission, a field shown once it applies), live progress, then the result."""
    service, url = app
    image = jpeg(tmp_path_factory.mktemp("img") / "photo.jpg")
    t = Tab(browser, url)
    try:
        pg = t.go("#/image")
        pg.check("#mode-segment-image")
        assert "op=segment-image" in pg.url
        pg.set_input_files("#image-file", str(image))
        pg.wait_for_selector("img.preview:not([hidden])")
        pg.wait_for_function("() => document.querySelector('.file-state').textContent.includes('uploaded')")
        attrs = pg.locator(".field[data-param=attrs]")
        assert attrs.is_hidden()  # -p applies to -f ply or -d only
        pg.check(".field[data-param=artifacts] input[type=checkbox]")
        attrs.wait_for(state="visible")
        # an invalid value is flagged next to its field, in the command's words, before submission
        stride = attrs.locator("input[id$='-stride']")
        stride.fill("0")
        err = attrs.locator(".field-error")
        err.wait_for(state="visible")
        assert "stride" in err.inner_text()
        assert attrs.locator("input[id$='-stride']").get_attribute("aria-invalid") == "true"
        stride.fill("")
        pg.wait_for_function("() => !document.querySelector('.field[data-param=attrs] .field-error').textContent")
        pg.wait_for_function("() => document.querySelector('.command')?.textContent.includes('-d=artifacts')")
        assert "consequence" in pg.inner_html("form.image-form")
        pg.click("form.image-form button[type=submit]")
        pg.wait_for_selector("section.job", timeout=10000)
        jid = wait_job(pg)
        assert f"#/image/{jid}" in pg.url
        t.a11y()
        assert t.errors == []
        return {"id": jid, "url": pg.url, "image": image}
    finally:
        t.close()


def test_image_job_result(image_job: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    """The result: the rendered segmented image and catalogue, the embedded viewer, a download
    for the result and every file; the page returns in the same state after a reload."""
    service, _ = app
    pg = tab.go("#" + image_job["url"].split("#", 1)[1])
    wait_job(pg)
    pg.wait_for_selector("figure.segmented[data-ready=true]")
    rows = pg.locator("table.data tbody tr[data-id]")
    rows.first.wait_for()
    assert rows.count() == 2
    pg.wait_for_selector(".viewer-box[data-ready=true]", timeout=60000)
    job = service.runner.get(image_job["id"])
    names = sorted(p.name for p in (service.workspace.job_dir(job.id) / "out").rglob("*") if p.is_file())
    links = pg.locator("ul.downloads a.download")
    assert links.count() == len(names)
    for n in names:
        assert pg.locator(f"ul.downloads a.download[download='{n}']").count() == 1, n
    # a reload returns in the same state
    pg.reload()
    pg.wait_for_selector("body[data-ready=true]")
    wait_job(pg)
    pg.wait_for_selector("figure.segmented[data-ready=true]")
    assert tab.errors == []


def test_downloads_are_the_jobs_files(image_job: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    service, _ = app
    pg = tab.go(f"#/jobs/{image_job['id']}")
    wait_job(pg)
    pg.wait_for_selector("ul.downloads a.download")
    out = service.workspace.job_dir(image_job["id"]) / "out"
    for name in ("result.json", "segmented.png", "catalog.csv"):
        with pg.expect_download() as d:
            pg.click(f"ul.downloads a.download[download='{name}']")
        got = Path(d.value.path()).read_bytes()
        assert got == next(out.rglob(name)).read_bytes(), name
    assert tab.errors == []


def test_selection_is_shared_by_table_image_and_viewer(image_job: dict[str, Any], tab: Tab) -> None:
    pg = tab.go(f"#/jobs/{image_job['id']}")
    wait_job(pg)
    pg.wait_for_selector("figure.segmented[data-ready=true]")
    pg.wait_for_selector(".viewer-box[data-ready=true]", timeout=60000)
    # a table row (keyboard) → the viewer's box and the image region
    row = pg.locator("table.data tbody tr[data-id='2']")
    row.focus()
    pg.keyboard.press("Enter")
    assert row.get_attribute("aria-selected") == "true"
    assert frame_viewer(pg, "v.selected") == 2
    assert pg.get_attribute("figure.segmented", "data-selected") == "2"
    assert "2" in pg.inner_text(".seg-caption")
    # an image region → the table and the viewer
    pg.evaluate("""() => {
      const c = document.querySelector('.seg-canvas'), r = c.getBoundingClientRect();
      // object 1 of the stub segmenter covers x in [w/8, 3w/8], y in [h/4, 3h/4]
      c.dispatchEvent(new MouseEvent('click', {clientX: r.left + r.width / 4, clientY: r.top + r.height / 2, bubbles: true}));
    }""")
    assert pg.get_attribute("table.data tbody tr[data-id='1']", "aria-selected") == "true"
    assert frame_viewer(pg, "v.selected") == 1
    # a box in the viewer → the table and the image
    frame_viewer(pg, "v.select(2)")
    assert pg.get_attribute("table.data tbody tr[data-id='2']", "aria-selected") == "true"
    assert pg.get_attribute("figure.segmented", "data-selected") == "2"
    # a click on a box in the viewer's canvas selects it
    hit = pg.evaluate("""async () => {
      const f = document.querySelector('iframe.viewer-frame'), w = f.contentWindow, v = w.__viewerApp;
      v.select(null);
      const { pickObject } = await import('/static/js/pick.js');
      const host = v.host, W = host.clientWidth, H = host.clientHeight;
      for (let y = 5; y < H; y += 8) for (let x = 5; x < W; x += 8) {
        const id = pickObject(v, x, y);
        if (id === 1) {
          const c = v.renderer.domElement, r = c.getBoundingClientRect();
          const o = {clientX: r.left + x, clientY: r.top + y, bubbles: true, pointerId: 1};
          c.dispatchEvent(new w.PointerEvent('pointerdown', o));
          c.dispatchEvent(new w.PointerEvent('pointerup', o));
          return v.selected;
        }
      }
      return 'no box on screen';
    }""")
    assert hit == 1
    assert pg.get_attribute("table.data tbody tr[data-id='1']", "aria-selected") == "true"
    # every object's id and label stay visible in their cells and in the caption
    assert_badges_visible(pg, "table.data")
    assert_badges_visible(pg, ".seg-caption")
    # the selection is in the URL: a reload comes back with it everywhere
    assert "sel=1" in pg.url
    pg.reload()
    pg.wait_for_selector("body[data-ready=true]")
    pg.wait_for_selector(".viewer-box[data-ready=true]", timeout=60000)
    pg.wait_for_selector("figure.segmented[data-ready=true]")
    assert pg.get_attribute("table.data tbody tr[data-id='1']", "aria-selected") == "true"
    assert pg.get_attribute("figure.segmented", "data-selected") == "1"
    assert frame_viewer(pg, "v.selected") == 1
    assert tab.errors == []


def test_viewer_error_is_shown(image_job: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    """A ?viewer=true job whose viewer step failed or was cancelled still shows its result, and
    says why there is no viewer (the job record's viewer_error)."""
    service, _ = app
    job = service.runner.get(image_job["id"])
    saved = (job.viewer, job.viewer_error)
    try:
        for code, words in (("server_unavailable", "could not be prepared"), ("cancelled", "was cancelled")):
            job.viewer, job.viewer_error = None, {"code": code, "message": "view.sh: error: no server", "exit_code": 3}
            pg = tab.go(f"#/jobs/{job.id}")
            pg.reload()  # the same URL: load the changed record again
            pg.wait_for_selector("body[data-ready=true]")
            wait_job(pg)
            note = pg.locator(".job-result .notice.error")
            note.wait_for()
            assert words in note.inner_text()
            assert pg.locator("iframe.viewer-frame").count() == 0
            assert pg.locator("ul.downloads a.download").count() >= 1
    finally:
        job.viewer, job.viewer_error = saved


# ---------------------------------------------------------------------------- maps (mapping, COLMAP)


@pytest.fixture(scope="module")
def mapped(browser: Any, app: tuple[Any, str], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Map creation and update through the guided flow (stub server, COLMAP)."""
    if shutil.which("colmap") is None:
        pytest.skip("needs Homebrew colmap")
    service, url = app
    src = tmp_path_factory.mktemp("frames")
    files = [shutil.copy(f, src / f.name) for f in FRAMES[:3]]
    t = Tab(browser, url)
    try:
        pg = t.go("#/maps/new")
        pg.set_input_files(".field[data-param=inputs] input[type=file]", [str(files[1]), str(files[0])])
        pg.wait_for_function("() => document.querySelectorAll('.field[data-param=inputs] li.ready').length === 2")
        order = pg.locator(".field[data-param=inputs] ol.files li .file-name").all_inner_texts()
        assert order == [Path(files[1]).name, Path(files[0]).name]
        pg.click(f".field[data-param=inputs] button[aria-label='Move {Path(files[1]).name} later']")  # reorder
        order = pg.locator(".field[data-param=inputs] ol.files li .file-name").all_inner_texts()
        assert order == [Path(files[0]).name, Path(files[1]).name]
        pg.fill(".field[data-param=map] input", "room")
        pg.wait_for_function("() => document.querySelector('[data-testid=flow-command]').textContent.includes('room')")
        cmd = pg.inner_text("[data-testid=flow-command]")
        assert cmd.index(Path(files[0]).name) < cmd.index(Path(files[1]).name)  # the order is sent
        t.a11y()
        pg.click("form.flow button[type=submit]")
        pg.wait_for_selector("dialog#confirm[open]")
        assert "leaves the map exactly as it was" in pg.inner_text("dialog#confirm")
        pg.click("#confirm-yes")
        pg.wait_for_url("**#/jobs/*")
        created = wait_job(pg, timeout=300_000)
        # the update: one more frame, the map fixed
        pg = t.go("#/maps/room/update")
        assert pg.locator(".field[data-param=map]").count() == 0
        pg.set_input_files(".field[data-param=inputs] input[type=file]", [str(files[2])])
        pg.wait_for_function("() => document.querySelectorAll('.field[data-param=inputs] li.ready').length === 1")
        pg.wait_for_function("() => document.querySelector('[data-testid=flow-command]').textContent.includes('room')")
        pg.click("form.flow button[type=submit]")
        pg.click("#confirm-yes")
        pg.wait_for_url("**#/jobs/*")
        pg.wait_for_function(f"() => !location.hash.endsWith('{created}')")
        updated = wait_job(pg, timeout=300_000)
        assert t.errors == []
        return {"name": "room", "jobs": [created, updated]}
    finally:
        t.close()


def test_map_cards_and_page(mapped: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    pg = tab.go("#/maps")
    card = pg.locator("li.map-card[data-map=room]")
    card.wait_for()
    assert card.locator("img").get_attribute("alt") == "A keyframe of room"
    pg.wait_for_function("() => document.querySelector('li.map-card img').naturalWidth > 0")
    assert "frames" in card.inner_text()
    pg.fill("#map-filter", "nothing-like-this")
    assert pg.locator("li.map-card").count() == 0
    pg.fill("#map-filter", "roo")
    assert pg.locator("li.map-card").count() == 1
    assert "filter=roo" in pg.url
    tab.a11y()
    card.locator("a.card-link").click()
    pg.wait_for_url("**#/maps/room")
    hist = pg.locator("[data-testid=history] > li")
    hist.first.wait_for()
    assert hist.count() == 2  # created, then updated
    # each update record as the mapper wrote it, its per-stage timings included
    meta = json.loads((app[0].workspace.maps / "room" / "map.json").read_text())
    stages = meta["updates"][-1]["timings"]["stages_s"]
    text = hist.nth(1).inner_text()
    assert "timings total s" in text and all(f"timings stages s {k.replace('_', ' ')}" in text for k in stages)
    pg.locator("[data-testid=objects] tbody tr").first.wait_for()
    assert_badges_visible(pg, "[data-testid=objects]")
    # every export the commands offer for a map: one form per operation that takes a map
    exports = pg.locator("details.export").evaluate_all("els => els.map(e => e.dataset.op)")
    ops = [o for o in web_ops.operations().values()
           if any(p.kind is spec.Kind.MAP for p in o.options) and not o.writes_map() and not o.browser]
    assert sorted(exports) == sorted(o.id for o in ops)
    tab.a11y()
    assert tab.errors == []


def test_map_viewer_embedded_and_full_screen(mapped: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    pg = tab.go("#/maps/room")
    pg.wait_for_selector(".viewer-box[data-ready=true]", timeout=60000)
    rows = pg.locator("[data-testid=objects] tbody tr[data-id]")
    rows.first.wait_for()
    oid = int(rows.first.get_attribute("data-id"))
    rows.first.click()
    assert frame_viewer(pg, "v.selected") == oid
    frame_viewer(pg, "v.select(null)")
    assert pg.locator("[data-testid=objects] tr[aria-selected=true]").count() == 0
    full = pg.get_attribute(".viewer-box a", "href")
    assert full == "/viewer/map/room/"
    pg2 = tab.ctx.new_page()
    pg2.goto(tab.base.rstrip("/") + full)
    pg2.wait_for_selector('body[data-rendered="true"]', timeout=60000)
    assert pg2.evaluate("() => window.__viewer.meta.mode") == "map"
    pg2.close()
    # an export runs as a job and links to it
    pg.click("details.export[data-op=segment-map] summary")
    pg.check("details.export[data-op=segment-map] .field[data-param=artifacts] input[type=checkbox]")
    pg.wait_for_function("() => document.querySelector('details.export[data-op=segment-map] .command')?.textContent.includes('-d=')")
    pg.click("details.export[data-op=segment-map] button[type=submit]")
    link = pg.locator("details.export[data-op=segment-map] .notice a")
    link.wait_for()
    link.click()
    wait_job(pg)
    pg.wait_for_selector("figure.segmented[data-ready=true]")
    assert tab.errors == []


# ----------------------------------------------------------------------------- the 3D scene viewer


def scene_files(folder: Path) -> tuple[Path, Path, dict[str, Any]]:
    """A map scene JSON (two objects, three keyframes, two located cameras) and a PLY of the same
    map with labels and a located camera in its header."""
    from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope
    from oh_my_slam.mapping.locate import pose_comment

    doc = scene_doc()
    assert validation_errors(doc) == []
    xyz, rgb, labels = module_cloud()
    comments = [f"attributes {CloudAttrs(color='rgb', label=True).describe(CloudScope.MAP)}",
                pose_comment(located(0, (1.0, -2.0, 1.5)))]
    ply = folder / "map.ply"
    ply.write_bytes(ply_bytes(PointCloud(xyz, rgb, labels), comments=comments))
    js = folder / "scene.json"
    js.write_text(json.dumps(doc))
    return ply, js, doc


def test_scene_viewer_with_a_ply_and_a_json(tab: Tab, tmp_path: Path) -> None:
    ply, js, doc = scene_files(tmp_path)
    pg = tab.go("#/scene")
    requests: list[str] = []
    pg.on("request", lambda r: requests.append(f"{r.method} {r.url}"))
    pg.set_input_files("#scene-files", [str(ply), str(js)])
    pg.wait_for_selector("[data-testid=scene-view][data-built]")
    assert pg.get_attribute("[data-testid=scene-view]", "data-loaded") == "true"
    assert not [r for r in requests if r.startswith("POST")]  # read in the browser, never uploaded
    assert pg.locator("[data-testid=sources] li").count() == 2
    v = "window.__sceneViewer"
    assert pg.get_attribute("[data-testid=scene-view]", "data-upright") == "false"  # map coordinates
    assert tab.js(f"() => {v}.root.matrix.equals(new {v}.root.matrix.constructor())") is True
    assert tab.js(f"() => {v}.cloud.count") == 5000
    assert tab.js(f"() => {v}.objects.length") == 2
    # one toggle per layer, each naming its file; the layers on are in the URL
    layers = pg.locator("#scene-layers [data-layer]")
    assert layers.count() == 5
    assert "map.ply" in pg.inner_text("#scene-layers [data-layer=points]")
    assert "scene.json" in pg.inner_text("#scene-layers [data-layer=obbs]")
    pg.uncheck("#layer-obbs")
    assert tab.js(f"() => {v}.groups.obbs.visible") is False
    assert "layers=" in pg.url and "obbs" not in pg.url.split("layers=")[1].split("&")[0]
    pg.check("#layer-obbs")
    pg.check("#layer-segments")  # the PLY's labels in the JSON's object colours
    assert tab.js(f"() => {v}.groups.segments.children.length") == 1
    # OBBs in the object colours, labelled with id and label
    tags = tab.js("() => [...document.querySelectorAll('.scene-view .obj-label')].map(d => [d.dataset.id, d.querySelector('.tag').style.background])")
    assert {t[0] for t in tags} >= {"1", "2"}
    # cameras: the JSON's 3 keyframes and 2 located, the PLY header's located one; coordinates, go-to
    cams = pg.locator("[data-testid=scene-cameras] tbody tr")
    assert cams.count() == 6
    assert pg.locator("[data-testid=scene-cameras] tbody tr[data-located]").count() == 3
    before = tab.js(f"() => {v}.camera.position.toArray()")
    pg.click("[data-testid=scene-cameras] tbody tr:nth-child(4) button.goto")
    after = tab.js(f"() => {v}.camera.position.toArray()")
    x = float(pg.inner_text("[data-testid=scene-cameras] tbody tr:nth-child(4) td:nth-child(2)"))
    assert after != before and after[0] == pytest.approx(x, abs=1e-3)
    # the objects list is linked to the boxes, its ids and labels visible
    assert_badges_visible(pg, "[data-testid=objects]")
    pg.click("[data-testid=objects] tbody tr[data-id='2']")
    assert tab.js(f"() => {v}.selected") == 2 and "sel=2" in pg.url
    tab.js(f"() => {v}.select(1)")
    assert pg.get_attribute("[data-testid=objects] tbody tr[data-id='1']", "aria-selected") == "true"
    # the point-cloud controls the file allows: its colours or none
    assert pg.locator("#scene-cloud [data-attr]").evaluate_all("els => els.map(e => e.dataset.attr)") == ["color"]
    pg.select_option("#attr-color", "none")
    assert tab.js(f"() => !!{v}.groups.points.children[0].material.defines.HAS_COLOR") is False
    assert "color=none" in pg.url
    tab.a11y()
    assert tab.errors == []


def test_scene_viewer_refuses_invalid_files(tab: Tab, tmp_path: Path) -> None:
    pg = tab.go("#/scene")
    bad_json = tmp_path / "bad.json"
    doc = scene_doc()
    doc["openlabel"]["objects"]["1"]["object_data"]["cuboid"][0]["val"] = [0, 0, 0]
    doc["openlabel"]["extra"] = True
    bad_json.write_text(json.dumps(doc))
    assert validation_errors(doc)
    bad_ply = tmp_path / "bad.ply"
    bad_ply.write_bytes(b"ply\nformat binary_big_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n")
    huge = tmp_path / "huge.ply"  # a header only: refused before the body is read
    huge.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 16000001\nproperty float x\n"
                     b"property float y\nproperty float z\nend_header\n")
    not_json = tmp_path / "broken.json"
    not_json.write_text("{")
    many = tmp_path / "many.json"  # a frame range of a billion frames is not enumerated
    d = scene_doc()
    d["openlabel"]["frame_intervals"] = [{"frame_start": 0, "frame_end": 10**9}]
    many.write_text(json.dumps(d))
    pg.set_input_files("#scene-files", [str(bad_json), str(bad_ply), str(huge), str(not_json), str(many)])
    errors = pg.locator("[data-testid=scene-errors] .notice")
    errors.nth(4).wait_for(timeout=10000)
    text = pg.inner_text("[data-testid=scene-errors]")
    assert "bad.json was refused" in text and "scene schema" in text
    assert "Additional properties are not allowed ('extra' was unexpected)" in text
    assert "val must be 10 numbers" in text
    assert "bad.ply was refused" in text and "binary_big_endian" in text
    assert "huge.ply was refused" in text and "16,000,001 points" in text and "16,000,000" in text
    assert "Maps" in text  # how to view it instead
    assert "broken.json was refused" in text and "not JSON" in text
    assert "many.json was refused" in text and "frame_intervals do not match the frame keys" in text
    assert pg.get_attribute("[data-testid=scene-view]", "data-loaded") != "true"


def _all_paths(node: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _all_paths(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _all_paths(v, (*path, i))


def _mutations(doc: dict[str, Any], rng: random.Random, n: int) -> list[dict[str, Any]]:
    """``n`` single mutations of ``doc``, spread over its sections: a key deleted or renamed, a
    value of another type, an extra key, a list made longer, shorter or empty."""
    others = [7, 2.5, "text", True, None, [], {}, [1, 2], {"zz": 1}]
    candidates: list[tuple[tuple[Any, ...], str]] = []
    for path, node in _all_paths(doc):
        if path:
            candidates += [(path, "delete"), (path, "retype")]
        if isinstance(node, dict):
            candidates.append((path, "extra"))
            if node:
                candidates.append((path, "rename"))
        if isinstance(node, list):
            candidates += [(path, "longer"), (path, "empty")]
            if len(node) > 1:
                candidates.append((path, "shorter"))
    by_section: dict[Any, list[tuple[tuple[Any, ...], str]]] = {}
    for c in candidates:
        by_section.setdefault(c[0][1] if len(c[0]) > 1 else "", []).append(c)
    picked: list[tuple[tuple[Any, ...], str]] = []
    while len(picked) < n and any(by_section.values()):
        for items in by_section.values():
            if items:
                picked.append(items.pop(rng.randrange(len(items))))
    out = []
    for path, kind in picked[:n]:
        d = json.loads(json.dumps(doc))
        parent = d
        for k in path[:-1]:
            parent = parent[k]
        node = parent if not path else parent[path[-1]]
        if kind == "delete":
            del parent[path[-1]]
        elif kind == "retype":
            parent[path[-1]] = rng.choice([o for o in others if type(o) is not type(node)])
        elif kind == "extra":
            node["zz_extra"] = 1
        elif kind == "rename":
            key = rng.choice(list(node))
            node[f"{key} bad!"] = node.pop(key)
        elif kind == "longer":
            node.append(json.loads(json.dumps(node[-1])) if node else 0)
        elif kind == "shorter":
            node.pop()
        else:
            node.clear()
        out.append(d)
    return out


def _python_problems(doc: Any) -> tuple[list[str], list[str] | None]:
    from oh_my_slam.schema.validate import _validator, extra_errors

    paths = sorted({"/".join(str(p) for p in e.absolute_path) for e in _validator().iter_errors(doc)})
    try:
        extra = extra_errors(doc)
    except Exception:  # a shape the checks beyond the schema cannot read (the schema says why)
        extra = None
    return paths, extra


def test_browser_schema_validation_agrees_with_the_commands(tab: Tab, app: tuple[Any, str]) -> None:
    """The in-browser OpenLABEL check (the app's draft-07 validator + the checks of
    schema/validate.py) agrees with the commands' (jsonschema + extra_errors) on every scene
    document the fixtures and this module's jobs produce, and on a few hundred mutations of them:
    the same validity, the same schema error paths, the same extra messages."""
    from tests.unit.test_openlabel import map_example, single_image_example

    service, _ = app
    docs = [scene_doc(), single_image_example(), map_example()]
    for f in sorted(service.workspace.jobs.glob("*/out/**/*.json")):
        d = json.loads(f.read_text())
        if isinstance(d, dict) and "openlabel" in d:
            docs.append(d)
    assert len(docs) >= 5  # the image job's result and segmentation.json at least
    for d in docs:
        assert validation_errors(d) == [], validation_errors(d)
    rng = random.Random(7)
    variants = list(docs)
    for d in docs:
        variants += _mutations(d, rng, 400 // len(docs) + 1)
    assert len(variants) >= 400
    pg = tab.go("#/scene")
    got = pg.evaluate("""async (docs) => {
      const { sceneProblems } = await import('/static/js/scene/openlabel.js');
      const out = [];
      for (const d of docs) {
        const p = await sceneProblems(d);
        out.push({paths: [...new Set(p.schema.map(e => e.path))].sort(), extra: p.extra});
      }
      return out;
    }""", variants)
    invalid = 0
    for d, js in zip(variants, got, strict=True):
        paths, extra = _python_problems(d)
        assert js["paths"] == sorted(paths), (js["paths"], paths)
        if not paths and extra is not None:  # the schema holds: the checks beyond it agree
            assert js["extra"] == extra, (js["extra"], extra)
        elif not paths:  # a shape the schema allows but validate.py cannot read (it raises,
            # e.g. a frame interval that is a list): the browser refuses it with a reason
            assert js["extra"], d
        invalid += bool(paths or extra)
    assert invalid >= len(variants) // 2  # most mutations break the document; the rest stay valid


def test_scene_viewer_opens_job_files_on_a_stable_url(image_job: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    """A job's result and file opened in the 3D scene viewer from the job page; a single image's
    camera frame is shown upright by the viewer's own transform; the URL holds the sources, the
    layers, the point-cloud controls and the selection, so a reload returns in the same state."""
    from oh_my_slam.viewer.bundle import display_transform

    service, _ = app
    pg = tab.go(f"#/jobs/{image_job['id']}")
    wait_job(pg)
    pg.locator("ul.downloads a.scene-link").first.wait_for()
    hrefs = pg.locator("ul.downloads a.scene-link").evaluate_all("els => els.map(e => e.getAttribute('href'))")
    href = next(h for h in hrefs if "ply=" in h and "json=" in h)
    pg.goto(tab.base + href)
    pg.wait_for_selector("[data-testid=scene-view][data-built]")
    assert pg.locator("[data-testid=sources] li").count() == 2
    v = "window.__sceneViewer"
    assert tab.js(f"() => {v}.objects.length") == 2
    # upright: the transform view.sh -i uses, with the scene's estimated up direction
    result = json.loads((service.workspace.job_dir(image_job["id"]) / "out" / "result.json").read_text())
    up = result["openlabel"]["metadata"].get("gravity", {}).get("up_cam")
    expected = np.array(display_transform(True, up))
    got = np.array(tab.js(f"() => {v}.root.matrix.elements")).reshape(4, 4).T
    assert pg.get_attribute("[data-testid=scene-view]", "data-upright") == "true"
    assert np.allclose(got, expected, atol=1e-6)
    pg.uncheck("#layer-labels")
    pg.select_option("#attr-color", "none")
    pg.click("[data-testid=objects] tbody tr[data-id='2']")
    url = pg.url
    pg.reload()
    pg.wait_for_selector("[data-testid=scene-view][data-built]")
    assert pg.url == url
    assert tab.js(f"() => {v}.objects.length") == 2
    assert tab.js(f"() => {v}.layers.labels") is False and pg.is_checked("#layer-labels") is False
    assert pg.input_value("#attr-color") == "none"
    assert tab.js(f"() => {v}.selected") == 2
    assert tab.errors == []


def test_a_job_ply_above_the_budget_is_drawn_from_the_services_selection(
        image_job: dict[str, Any], tab: Tab, app: tuple[Any, str], tmp_path: Path) -> None:
    """A job's PLY over the display budget: its header is read alone (a Range request), then the
    service's budgeted cloud is drawn (/api/jobs/<id>/display-cloud), the header's located cameras
    kept. The header is made to claim 16,000,001 points; the service's own thinning is tested with
    a small budget in tests/unit/test_web_display.py."""
    service, _ = app
    ply, _js, _doc = scene_files(tmp_path)
    out = service.workspace.job_dir(image_job["id"]) / "out"
    shutil.copy(ply, out / "big.ply")
    url = f"/api/jobs/{image_job['id']}/files/big.ply"
    pg = tab.go("#/scene")
    seen: list[str] = []
    fake = (b"ply\nformat binary_little_endian 1.0\nelement vertex 16000001\nproperty float x\n"
            b"property float y\nproperty float z\nend_header\n")

    def handle(route: Any) -> None:
        r = route.request
        seen.append(f"{r.headers.get('range', '-')} {r.url}")
        if "range" in r.headers:
            route.fulfill(status=206, body=fake, headers={"Content-Type": "application/octet-stream"})
        else:
            route.continue_()

    pg.route(f"**{url}", handle)
    pg.on("request", lambda r: seen.append(f"req {r.url}") if "display-cloud" in r.url else None)
    pg.goto(tab.base + f"#/scene?ply={url}")
    pg.wait_for_selector("[data-testid=scene-view][data-built]")
    assert [s for s in seen if s.startswith("bytes=0-")], seen  # the header alone first
    assert not [s for s in seen if s.startswith("- ")], seen  # the file itself never downloaded
    assert [s for s in seen if "display-cloud?file=big.ply" in s], seen
    assert tab.js("() => window.__sceneViewer.cloud.total") == 5000
    assert pg.locator("[data-testid=scene-cameras] tbody tr[data-located]").count() == 1
    (out / "big.ply").unlink()
    assert tab.errors == []


# ------------------------------------------------------------------------- jobs: cancel, re-submit


@pytest.fixture
def slow_registry(app: tuple[Any, str]) -> Iterator[Any]:
    """The registry with new modes — slow.sh -m MAP (writing note.md into -d), nap.sh -i IMAGE
    (an option of a kind never seen, a rule with its own message) — and a new option on
    segment.sh -m (--shade): no change to the web application."""
    service, _ = app
    seg = spec.SEGMENT.command()
    extra = spec.Option("--shade", "shade", spec.Kind.ENUM, "a new option of the export", default="dark",
                        choices=("dark", "light"), modes=("map",))
    seg2 = dataclasses.replace(seg, options=(*seg.options, extra))
    programs = (*[p for p in spec.PROGRAMS if p is not spec.SEGMENT],
                dataclasses.replace(spec.SEGMENT, commands=(seg2,)), slow_command.map_registry_program(),
                slow_command.image_registry_program())
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(spec, "PROGRAMS", programs)
        mp.setattr(web_ops.Operation, "module", property(
            lambda op: slow_command.MODULE if op.program.prog in ("slow.sh", "nap.sh")
            else f"oh_my_slam.cli.{op.program.prog.removesuffix('.sh')}"))
        service.ops = web_ops.operations()
        minimal_map(service.workspace.maps / "empty")
        yield service
    service.ops = web_ops.operations()


def test_a_registry_change_reaches_the_ui(slow_registry: Any, image_job: dict[str, Any], tab: Tab) -> None:
    """A new option is a new field, a new mode a new form (on the page its inputs belong to), a new
    kind of option still a field, a new output file a new download, a new error a new message."""
    pg = tab.go("#/maps/empty")
    card = pg.locator("details.export[data-op=slow]")
    card.wait_for()
    card.locator("summary").click()
    pg.click("details.export[data-op=segment-map] summary")
    shade = pg.locator("details.export[data-op=segment-map] .field[data-param=shade] select")
    shade.wait_for()
    assert shade.input_value() == "dark" and "a new option of the export" in pg.inner_text(
        "details.export[data-op=segment-map] .field[data-param=shade]")
    shade.select_option("light")
    pg.wait_for_function("() => document.querySelector('details.export[data-op=segment-map] .command')?.textContent.includes('--shade=light')")
    card.locator(".field[data-param=seconds] input").fill("0.2")
    card.locator(".field[data-param=folder] input[type=checkbox]").check()
    pg.wait_for_function("() => document.querySelector('details.export[data-op=slow] .command')?.textContent.includes('-d=folder')")
    card.locator("button[type=submit]").click()
    card.locator(".notice a").click()
    wait_job(pg, timeout=60000)
    pg.wait_for_selector("ul.downloads a.download[download='note.md']")
    # a new single-image mode is on the Image page, with a field for an option of a new kind
    pg = tab.go("#/image")
    pg.check("#mode-nap")
    mood = pg.locator(".field[data-param=mood] input")
    mood.wait_for()
    assert "how the nap feels" in pg.inner_text(".field[data-param=mood]")
    mood.fill("calm")
    pg.set_input_files("#image-file", str(image_job["image"]))
    pg.wait_for_function("() => document.querySelector('.file-state').textContent.includes('uploaded')")
    # its rule's message, next to its field, before submission
    pg.fill(".field[data-param=seconds] input", "500")
    err = pg.locator(".field[data-param=seconds] .field-error")
    err.wait_for(state="visible")
    assert err.inner_text() == slow_command.NAP_LIMIT
    pg.fill(".field[data-param=seconds] input", "0.1")
    pg.fill(".field[data-param=code] input", "2")
    pg.wait_for_function("() => document.querySelector('.command')?.textContent.includes('--mood=calm')")
    pg.click("form.image-form button[type=submit]")
    wait_job(pg, "failed", 60000)
    assert "asked to fail" in pg.inner_text("section.job .notice.error")
    assert tab.errors == []


def test_cancel_states_the_consequence_then_cancels(slow_registry: Any, tab: Tab, app: tuple[Any, str]) -> None:
    service, _ = app
    pg = tab.go("#/maps/empty")
    card = pg.locator("details.export[data-op=slow]")
    card.locator("summary").click()
    card.locator(".field[data-param=seconds] input").fill("60")
    pg.wait_for_function("() => document.querySelector('details.export[data-op=slow] .command')?.textContent.includes('60')")
    card.locator("button[type=submit]").click()
    link = card.locator(".notice a")
    link.wait_for()
    jid = link.inner_text().split()[-1]
    pg.goto(tab.base + "#/jobs")
    row = pg.locator(f"tr[data-job='{jid}']")
    pg.wait_for_selector(f"tr[data-job='{jid}'][data-state=running]")
    row.locator("progress").wait_for()
    row.locator("button[data-action=cancel]").click()
    dlg = pg.locator("dialog#confirm[open]")
    dlg.wait_for()
    assert "Ctrl-C" in dlg.inner_text() and "no result" in dlg.inner_text()
    tab.a11y()
    pg.click("#confirm-no")  # changing one's mind keeps it running
    assert service.runner.get(jid).state == "running"
    # the focus returns to the button that opened the dialog
    assert pg.evaluate("() => document.activeElement.dataset.action") == "cancel"
    row.locator("button[data-action=cancel]").click()
    pg.click("#confirm-yes")
    pg.wait_for_selector(f"tr[data-job='{jid}'][data-state=cancelled]", timeout=60000)
    assert service.runner.get(jid).state == "cancelled"
    # re-submit with the same options
    pg.locator(f"tr[data-job='{jid}'] button[data-action=resubmit]").click()
    pg.wait_for_url("**#/jobs/*")
    pg.wait_for_selector("section.job[data-state=running]", timeout=30000)
    new = pg.get_attribute("section.job", "data-job")
    assert new != jid and service.runner.get(new).params == service.runner.get(jid).params
    pg.click("section.job button[data-action=cancel]")
    pg.click("#confirm-yes")
    wait_job(pg, "cancelled", 60000)
    assert tab.errors == []


def test_resubmit_asks_again_for_a_discarded_upload(image_job: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    service, _ = app
    pg = tab.go(f"#/jobs/{image_job['id']}")
    wait_job(pg)
    pg.click("section.job button[data-action=resubmit]")
    dlg = pg.locator("dialog#confirm[open]")
    dlg.wait_for()
    assert "deleted when it ended" in dlg.inner_text() and "photo.jpg" in dlg.inner_text()
    assert "choose this file again" in dlg.inner_text()
    pg.set_input_files("dialog#confirm input[type=file]", str(image_job["image"]))
    pg.wait_for_function("() => document.querySelectorAll('dialog#confirm li.ready').length === 1")
    pg.click("#confirm-yes")
    pg.wait_for_function(f"() => location.hash.startsWith('#/jobs/') && !location.hash.endsWith('{image_job['id']}')")
    new = wait_job(pg)
    old_p, new_p = service.runner.get(image_job["id"]).params, service.runner.get(new).params
    assert {k: v for k, v in new_p.items() if k != "image"} == {k: v for k, v in old_p.items() if k != "image"}
    assert new_p["image"] != old_p["image"]
    assert tab.errors == []


def test_resubmit_keeps_the_order_of_ordered_inputs(mapped: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    """The map creation's two uploaded frames, chosen again in the other order, are re-submitted in
    their previous order."""
    service, _ = app
    old = service.runner.get(mapped["jobs"][0])
    names = [Path(v).name for v in old.params["inputs"]]
    assert len(names) == 2
    pg = tab.go(f"#/jobs/{old.id}")
    wait_job(pg)
    pg.click("section.job button[data-action=resubmit]")
    pg.locator("dialog#confirm[open]").wait_for()
    assert pg.locator("dialog#confirm ol.files li .file-name").all_inner_texts() == names
    files = {f.name: f for f in FRAMES}
    pg.set_input_files("dialog#confirm input[type=file]", [str(files[n]) for n in reversed(names)])
    pg.wait_for_function("() => document.querySelectorAll('dialog#confirm li.ready').length === 2")
    assert pg.locator("dialog#confirm ol.files li .file-name").all_inner_texts() == names
    pg.click("#confirm-yes")
    pg.wait_for_function(f"() => location.hash.startsWith('#/jobs/') && !location.hash.endsWith('{old.id}')")
    pg.wait_for_selector("section.job[data-job]")
    new = service.runner.get(pg.get_attribute("section.job", "data-job"))
    assert [Path(v).name for v in new.params["inputs"]] == names
    if new.state in ("queued", "running"):
        service.runner.cancel(new.id)
        service.runner.wait(new.id, 120)
    assert tab.errors == []


def test_a_broken_event_stream_reads_the_job_list_again(image_job: dict[str, Any], tab: Tab) -> None:
    pg = tab.go("#/jobs")
    pg.wait_for_selector(f"tr[data-job='{image_job['id']}']")
    back = pg.evaluate("""async (id) => {
      const { store } = window.__app;
      store.jobs.delete(id);  // an event missed while the stream was down
      store.events.dispatchEvent(new Event('error'));
      store.events.dispatchEvent(new Event('open'));
      for (let i = 0; i < 100 && !store.jobs.has(id); i++) await new Promise(r => setTimeout(r, 50));
      return store.jobs.has(id);
    }""", image_job["id"])
    assert back


# ------------------------------------------------------------------- every page: a11y, layout, URLs


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_every_page_is_accessible_and_fits_a_tablet(image_job: dict[str, Any], browser: Any, app: tuple[Any, str],
                                                    scheme: str) -> None:
    """axe-core (every impact of the WCAG 2.0/2.1 A and AA rules) on every page, its embedded viewer
    included, in both themes, at desktop and tablet width; no page scrolls sideways."""
    service, url = app
    if not (service.workspace.maps / "plain").exists():
        minimal_map(service.workspace.maps / "plain")
    routes = ["#/image", "#/maps", "#/maps/new", "#/maps/plain", "#/maps/plain/update", "#/jobs",
              f"#/jobs/{image_job['id']}", "#/scene"]
    for width in (1280, 768):
        t = Tab(browser, url, width=width, height=1000, scheme=scheme)
        try:
            for r in routes:
                pg = t.go(r)
                pg.wait_for_timeout(300)
                if r.startswith("#/jobs/"):
                    wait_job(pg)
                if pg.locator("iframe.viewer-frame").count():
                    pg.wait_for_selector(".viewer-box[data-ready=true]", timeout=60000)
                bad = axe_violations(pg)
                assert not bad, f"{r} at {width}px ({scheme}):\n" + "\n".join(bad)
                over = t.js("() => document.documentElement.scrollWidth - window.innerWidth")
                assert over <= 0, f"{r} scrolls sideways by {over}px at {width}px"
            assert t.errors == []
        finally:
            t.close()


def test_pages_respond_within_100_ms_and_keep_their_url(tab: Tab, app: tuple[Any, str]) -> None:
    """Navigation renders at once (no request in the way): the page's own timing of a route, from
    the hash change to the rendered page, has a median under 100 ms over 5 navigations per page —
    measured in the page, so a loaded machine's delays around the test's calls do not count. The
    new page's heading gets the focus and is announced; each page's URL brings it back."""
    pg = tab.go("#/image")
    for page, heading in (("maps", "Maps"), ("jobs", "Jobs"), ("scene", "3D scene viewer"), ("image", "Image")):
        times = []
        for _ in range(5):
            pg.evaluate("() => { location.hash = '#/nowhere'; }")
            pg.wait_for_function("() => window.__app.lastRoute?.where === 'nowhere'")
            pg.click(f"#nav a[data-page={page}]")
            pg.wait_for_function("(p) => window.__app.lastRoute?.where === p", arg=page)
            assert pg.inner_text("main h1") == heading
            times.append(pg.evaluate("() => window.__app.lastRoute.end - window.__app.lastRoute.start"))
        assert sorted(times)[2] < 100, (page, times)
        assert pg.get_attribute(f"#nav a[data-page={page}]", "aria-current") == "page"
        assert pg.evaluate("() => document.activeElement.tagName") == "H1"
        assert pg.evaluate("""() => getComputedStyle(document.querySelector('main h1')).scrollMarginTop
            !== '0px' && document.querySelector('main h1').getBoundingClientRect().top
            >= document.getElementById('topbar').getBoundingClientRect().bottom""")  # not under the bar
        assert pg.inner_text("#announce") == f"{heading} page"
        url = pg.url
        pg.reload()
        pg.wait_for_selector("body[data-ready=true]")
        assert pg.inner_text("main h1") == heading and pg.url == url
    assert tab.errors == []


def test_top_bar(tab: Tab, app: tuple[Any, str]) -> None:
    service, _ = app
    pg = tab.go("#/image")
    pg.wait_for_function("() => document.getElementById('workspace').textContent === 'ws'")
    pg.wait_for_function("() => document.body.dataset.inference === 'up'")
    assert "ready" in pg.inner_text("#inference")
    c = service.runner.counts()
    assert pg.inner_text("#job-counts") == f"{c['queued']} queued · {c['running']} running"


def test_keyboard_reaches_every_control_with_visible_focus(tab: Tab) -> None:
    pg = tab.go("#/image")
    seen = []
    for _ in range(40):
        pg.keyboard.press("Tab")
        info = pg.evaluate("""() => { const e = document.activeElement;
          const s = getComputedStyle(e); return [e.tagName, e.textContent.trim().slice(0, 30) || e.id, s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) >= 2]; }""")
        if info[0] == "BODY":  # past the last control: focus left the page
            break
        seen.append(info[1])
        assert info[2], f"no visible focus on {info}"
    assert "Choose an image…" in seen and any(s.startswith("Run ") for s in seen)

