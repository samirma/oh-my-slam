"""The server.sh web application in a real browser (``-m browser``; http_server.md "Evaluation → UI"):
a single-image request, a point-cloud result drawn in the page (on the Image page and on a map's
page), an image result (the depth image, the segmented image), map creation and update, locating
images in a map, the download of a result, interruption (and a request waiting for its turn), and
an automatic
accessibility check of every page; plus the registry-driven forms, stable URLs, responsiveness, the
keyboard and the layout down to tablet width in both themes.

The service is the real Starlette app and request runner (``tests.browser.webapp``) over a scratch
workspace, with the stub inference server (``tests.fakes.stub_server``); the inference server down
is ``test_webapp_down.py``."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from oh_my_slam.commands import spec
from oh_my_slam.web import operations as web_ops
from tests.browser.webapp import Tab, axe_violations, operation_response, running_service
from tests.fakes import slow_command
from tests.fakes.stub_server import start_stub_server
from tests.unit.test_view_cli import minimal_map, sh
from tests.unit.test_web_api import jpeg

pytestmark = [pytest.mark.browser]

REPO = Path(__file__).resolve().parents[2]
FRAMES = sorted((REPO / "examples" / "ainex-captures").glob("*.jpg"))


@pytest.fixture(scope="module")
def stub() -> Iterator[None]:
    start_stub_server()
    try:
        yield
    finally:
        sh("start_inference_server.sh", "--stop")


@pytest.fixture(scope="module")
def app(stub: None, tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, str]]:
    with pytest.MonkeyPatch.context() as mp:  # the commands' subprocesses import tests.fakes
        mp.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
        with running_service(tmp_path_factory.mktemp("web") / "ws") as (service, url):
            minimal_map(service.workspace.maps / "empty")
            yield service, url


@pytest.fixture
def tab(browser: Any, app: tuple[Any, str]) -> Iterator[Tab]:
    t = Tab(browser, app[1])
    yield t
    t.close()


@pytest.fixture(scope="module")
def image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return jpeg(tmp_path_factory.mktemp("img") / "photo.jpg")


def uploads(service: Any) -> list[str]:
    """The file names of the service's uploads."""
    return sorted(f.name for d in service.workspace.uploads.iterdir() for f in d.iterdir())


def scene_objects(doc: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(id, label, colour, score) of each object of a scene description, as the page lists them."""
    out = []
    for oid, o in sorted(doc["openlabel"].get("objects", {}).items(), key=lambda kv: int(kv[0])):
        data = o.get("object_data", {})
        hexes = [t["val"] for t in data.get("text", []) if t["name"] == "color_hex"]
        scores = [n["val"] for n in data.get("num", []) if n["name"] == "score"]
        out.append((f"#{oid}", o["type"], hexes[0] if hexes else "—",
                    f"{scores[0]:.2f}" if scores else "—"))
    return out


def listed_objects(pg: Any) -> list[tuple[str, str, str, str]]:
    return [tuple(r) for r in pg.locator("[data-testid=objects] tbody tr[data-id]").evaluate_all(
        "rows => rows.map(r => [...r.children].map(c => c.textContent.trim()))")]


def wait_ready(pg: Any, param: str, n: int = 1) -> None:
    """Wait until ``n`` files of the path field ``param`` are uploaded (its text if they are not)."""
    field = f".field[data-param={param}]"
    try:
        pg.wait_for_function(f"() => document.querySelectorAll('{field} li.ready').length === {n}",
                             timeout=30000)
    except Exception as exc:
        raise AssertionError(pg.inner_text(field) if pg.locator(field).count() else pg.inner_text("main")) from exc


def wait_route(pg: Any, path: str) -> None:
    """Wait until the page has rendered the route ``#/<path>`` (whatever its query)."""
    pg.wait_for_function("(p) => location.hash.split('?')[0] === '#/' + p && window.__app.lastRoute?.where === p",
                         arg=path)


def check_download(pg: Any, body: bytes) -> str:
    """The download is the response body byte for byte; returns its file name."""
    with pg.expect_download() as d:
        pg.click("[data-testid=download]")
    assert Path(d.value.path()).read_bytes() == body
    return d.value.suggested_filename


# ------------------------------------------------------------------------- a single-image request


def test_single_image_request(tab: Tab, app: tuple[Any, str], image: Path) -> None:
    """reconstruct.sh then segment.sh -i through the Image page: drop zone with preview, generated
    form (a field shown once it applies, a value flagged with the command's message before
    submission), the request running, then the scene's objects listed, the result in full, its
    stage timings and its download."""
    service, _ = app
    pg = tab.go("#/image?op=reconstruct")
    pg.wait_for_timeout(400)  # the first validation: nothing flagged before the user acts
    assert pg.locator(".field-error:not(:empty)").count() == 0
    pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
    pg.wait_for_selector("img.preview:not([hidden])")
    pg.wait_for_selector(".field[data-param=image] li.ready")
    # -p applies to -f ply only: hidden until then, and again for the depth image
    attrs = pg.locator(".field[data-param=attrs]")
    assert attrs.is_hidden()
    pg.select_option(".field[data-param=format] select", "ply")
    attrs.wait_for(state="visible")
    pg.select_option(".field[data-param=format] select", "depth")
    attrs.wait_for(state="hidden")
    # segment.sh -i, the image kept
    pg.check("#op-segment")
    assert "op=segment" in pg.url
    pg.wait_for_selector(".field[data-param=image] li.ready")
    assert pg.locator(".field[data-param=attrs]").count() == 0  # segment.sh has no -p
    # each format says what it gives (its output's title)
    assert pg.locator(".field[data-param=format] select option").all_inner_texts() == [
        "json — the scene description (default)", "png — the segmented image"]
    # an invalid value is flagged next to its field, in the command's words, before submission
    score = pg.locator(".field[data-param=min_score] input")
    assert "default 0.5" in pg.inner_text(".field[data-param=min_score] .help")  # its default and help
    score.fill("abc")
    err = pg.locator(".field[data-param=min_score] .field-error")
    err.wait_for(state="visible")
    assert err.inner_text() == "--min-score must be a number, got 'abc'"
    assert score.get_attribute("aria-invalid") == "true"
    score.fill("")
    pg.wait_for_function("() => !document.querySelector('.field[data-param=min_score] .field-error').textContent")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.startsWith('segment.sh -i uploads/')")
    with operation_response(pg, "segment") as answer:
        pg.click("button[data-action=run]")
        pg.wait_for_selector("[data-testid=request]", timeout=5000)
        assert pg.get_attribute("[data-testid=request]", "data-state") in ("sending", "waiting", "running", "done")
    tab.wait_request()
    body = answer.body
    # the scene's objects: label, id, colour and score, as the result says
    doc = json.loads(body)
    assert listed_objects(pg) == scene_objects(doc) and len(scene_objects(doc)) == 2
    # the result in full: formatted, and exactly as received
    shown = pg.inner_text("[data-testid=result-text]")
    assert json.loads(shown) == doc
    pg.check("[data-testid=result] .check-row input")
    assert pg.text_content("[data-testid=result-text]") == body.decode()
    # the command's stages, from Server-Timing
    stages = [s.split(";")[0].strip() for s in answer.headers["server-timing"].split(",")]
    pg.click("[data-testid=result] .stages-box summary")
    shown_stages = pg.locator("[data-testid=stages] tbody th code").all_inner_texts()
    assert shown_stages == [s for s in stages if s != "total"]
    assert check_download(pg, body) == "segment-photo.json"
    # the upload was consumed; the page has uploaded the image again for the next run
    pg.wait_for_selector(".field[data-param=image] li.ready")
    assert uploads(service) == ["photo.jpg"]
    tab.a11y()
    assert tab.errors == []


# ------------------------------------------------------------------ a point-cloud result in 3D

BACKGROUND = (0x15, 0x17, 0x1C)  # the viewer's (lib/viewer.js), as view.sh draws it
CLOUD = "[data-testid=cloud]"
CANVAS = "[data-testid=cloud-canvas]"


def cloud_image(pg: Any) -> np.ndarray:
    """The 3D view's WebGL canvas as last drawn (RGB)."""
    from PIL import Image

    url = pg.eval_on_selector(f"{CANVAS} canvas", "c => c.toDataURL('image/png')")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))


def cloud_state(pg: Any) -> dict[str, Any]:
    """The 3D view's viewpoint: camera position, the point it turns around, frames drawn."""
    return pg.eval_on_selector(CLOUD, """e => { const v = e.cloudView.viewer;
      return {pos: v.camera.position.toArray(), target: v.controls.target.toArray(), frames: v.frames,
              shown: v.groups.points.children[0].geometry.attributes.position.count}; }""")


def settle(pg: Any) -> dict[str, Any]:
    """Wait until the view stops moving (orbit damping) and nothing is drawn for 300 ms."""
    last = cloud_state(pg)
    for _ in range(100):
        pg.wait_for_timeout(300)
        now = cloud_state(pg)
        if now["frames"] == last["frames"]:
            return now
        last = now
    raise AssertionError("the 3D view keeps drawing")


def distance(st: dict[str, Any]) -> float:
    return float(np.linalg.norm(np.subtract(st["pos"], st["target"])))


def turned(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Degrees the viewpoint turned around the vertical (z) axis from ``b`` to ``a``."""
    da, db = np.subtract(a["pos"], a["target"]), np.subtract(b["pos"], b["target"])
    d = np.degrees(np.arctan2(da[1], da[0]) - np.arctan2(db[1], db[0]))
    return float(abs((d + 180) % 360 - 180))


def ply_body(pg: Any, op: str) -> bytes:
    """Run the page's operation ``op`` with ``-f ply``; the response body once the 3D view is
    drawn."""
    pg.select_option(".field[data-param=format] select", "ply")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-f ply')")
    with operation_response(pg, op) as answer:
        pg.click("button[data-action=run]")
    pg.wait_for_selector("[data-testid=request][data-state=done]", timeout=180_000)
    pg.wait_for_selector(f"{CLOUD}[data-state=drawn]", timeout=60_000)
    return answer.body


def check_drawn(pg: Any, body: bytes) -> None:
    """The 3D view of a PLY result: its point count is the file's (all drawn, within the budget),
    the canvas shows the points in the colours the file carries (exactly, as view.sh writes them),
    and its text alternative states the count and what each point carries."""
    from oh_my_slam.core.ply import parse_header, parse_ply

    n = parse_header(body).count
    cloud = parse_ply(body)
    assert n > 0 and cloud.rgb is not None
    assert pg.inner_text(f"{CLOUD} [data-fact=Points]") == f"{n:,}"
    assert cloud_state(pg)["shown"] == n
    carries = pg.inner_text(f"{CLOUD} [data-fact='Each point carries']")
    assert carries.startswith("position") and "colour (color=" in carries
    label = pg.get_attribute(CANVAS, "aria-label")
    assert f"{n:,} points" in label and "colour" in label
    img = cloud_image(pg)
    drawn = np.any(img != BACKGROUND, axis=2)
    assert drawn.mean() > 0.01, f"{drawn.mean():.4f} of the canvas drawn"
    colours = {tuple(c) for c in np.unique(cloud.rgb, axis=0).tolist()}
    exact = {tuple(c) for c in np.unique(img[drawn], axis=0).tolist()} & colours
    assert len(exact) >= min(10, max(1, len(colours) - 1)), f"{len(exact)} of the file's colours on screen"


def test_a_point_cloud_result_is_drawn_in_3d(tab: Tab, image: Path) -> None:
    """reconstruct -f ply on the Image page: the PLY in the response is read in the page (no other
    request) and drawn by the viewer's rendering with the colours it carries; buttons, the keyboard,
    a drag and the wheel rotate, pan and zoom it; its header is shown, the download is the body
    byte for byte, and the page passes axe."""
    pg = tab.go("#/image?op=reconstruct")
    assert pg.is_checked("#op-reconstruct")
    pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
    pg.wait_for_selector(".field[data-param=image] li.ready")
    requests: list[str] = []
    pg.on("request", lambda r: requests.append(f"{r.method} {r.url}"))
    body = ply_body(pg, "reconstruct")
    check_drawn(pg, body)
    # read from the bytes the page held: no request but the operation's and the viewer's modules
    # (besides the health polls and the image uploaded again for the next run)
    others = [r for r in requests if not any(p in r for p in (
        "/api/ops/reconstruct", "/static/viewer/", "/api/health", "/api/uploads"))]
    assert others == [], others
    assert pg.inner_text(f"{CLOUD} [data-fact=Frame]").endswith("shown upright, as view.sh shows an image (level camera)")
    header = body[:body.index(b"end_header\n") + len(b"end_header\n")].decode("latin1")
    assert pg.text_content("[data-testid=result-text]") == header
    first = settle(pg)
    # the buttons
    pg.click("[data-move='Rotate left']")
    st = settle(pg)
    assert turned(st, first) > 10 and distance(st) == pytest.approx(distance(first), rel=1e-3)
    before = cloud_image(pg)
    pg.click("[data-move='Zoom in']")
    st2 = settle(pg)
    assert distance(st2) == pytest.approx(distance(st) / 1.25, rel=1e-3)
    assert not np.array_equal(cloud_image(pg), before)
    pg.click("[data-move='Pan left']")
    st3 = settle(pg)
    assert np.linalg.norm(np.subtract(st3["target"], st2["target"])) > 1e-3
    pg.click("[data-move='Reset view']")
    st = settle(pg)
    np.testing.assert_allclose(st["pos"], first["pos"], atol=1e-4)
    # the keyboard: the view is focusable, arrows rotate, + and - zoom, 0 resets
    pg.focus(CANVAS)
    pg.keyboard.press("ArrowRight")
    st = settle(pg)
    assert turned(st, first) > 10
    pg.keyboard.press("-")
    st2 = settle(pg)
    assert distance(st2) == pytest.approx(distance(st) * 1.25, rel=1e-3)
    pg.keyboard.press("Shift+ArrowUp")
    st3 = settle(pg)
    assert st3["target"][2] > st2["target"][2]  # the viewpoint moved up (z is up)
    pg.keyboard.press("0")
    np.testing.assert_allclose(settle(pg)["pos"], first["pos"], atol=1e-4)
    # Tab goes from the view to its buttons, with a visible focus
    pg.keyboard.press("Tab")
    focused = pg.evaluate("""() => { const e = document.activeElement, s = getComputedStyle(e);
      return [e.getAttribute('aria-label'), s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) >= 2]; }""")
    assert focused == ["Rotate left", True]
    # the pointer: a drag rotates, the wheel zooms
    box = pg.locator(CANVAS).bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    pg.mouse.move(x, y)
    pg.mouse.down()
    for i in range(1, 11):
        pg.mouse.move(x + 8 * i, y)
    pg.mouse.up()
    st = settle(pg)
    assert turned(st, first) > 10
    pg.mouse.wheel(0, -400)
    assert distance(settle(pg)) < distance(st) * 0.95
    # an idle view draws nothing (it would keep the GPU busy next to the inference server)
    frames = cloud_state(pg)["frames"]
    pg.wait_for_timeout(1000)
    assert cloud_state(pg)["frames"] == frames
    assert check_download(pg, body) == "reconstruct-photo.ply"
    tab.a11y()
    # a new result replaces the view: the previous one is disposed (its canvas and WebGL context)
    pg.wait_for_selector(".field[data-param=image] li.ready")
    pg.eval_on_selector(CLOUD, "e => { e.dataset.old = 'true'; window.__oldView = e.cloudView; }")
    pg.click("button[data-action=run]")
    pg.wait_for_selector(f"{CLOUD}[data-state=drawn]:not([data-old])", timeout=180_000)
    assert pg.locator(f"{CANVAS} canvas").count() == 1
    assert pg.evaluate("() => window.__oldView.viewer.renderer.getContext().isContextLost()")
    old = pg.evaluate("() => window.__oldView.frames")
    pg.click("[data-move='Zoom in']")  # moves the new view; the old one draws no more
    pg.wait_for_timeout(300)
    assert pg.evaluate("() => window.__oldView.frames") == old
    assert tab.errors == []


def test_a_large_point_cloud_respects_the_display_budget(browser: Any, app: tuple[Any, str], image: Path) -> None:
    """Above the viewer's display budget (here a test's 1,000 points) that many points, evenly spaced
    in the file, are drawn and the page says so; the download stays complete."""
    t = Tab(browser, app[1])
    try:
        t.ctx.add_init_script("window.__cloudBudget = 1000")
        pg = t.go("#/image?op=reconstruct")
        pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
        pg.wait_for_selector(".field[data-param=image] li.ready")
        body = ply_body(pg, "reconstruct")
        from oh_my_slam.core.ply import parse_header

        n = parse_header(body).count
        assert n > 1000
        assert cloud_state(pg)["shown"] == 1000
        assert pg.inner_text(f"{CLOUD} [data-fact=Points]") == f"1,000 drawn of {n:,}"
        note = pg.inner_text(f"{CLOUD} .cloud-note")
        assert f"Showing 1,000 of {n:,} points: evenly spaced in the file's order, read in this page" in note
        assert "PLY outputs and the map stay complete" in note
        assert check_download(pg, body) == "reconstruct-photo.ply"
        t.a11y()
        assert t.errors == []
    finally:
        t.close()


def test_a_point_cloud_that_does_not_parse(tab: Tab, image: Path) -> None:
    """A PLY that cannot be read (here the service's answer cut short) says why in the page; its
    download stays available, byte for byte as received."""
    pg = tab.go("#/image?op=reconstruct")
    pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
    pg.wait_for_selector(".field[data-param=image] li.ready")
    pg.select_option(".field[data-param=format] select", "ply")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-f ply')")
    cut: dict[str, Any] = {}

    def truncate(route: Any) -> None:
        if route.request.method != "POST":
            route.continue_()
            return
        res = route.fetch(timeout=0)
        body = res.body()
        cut["n"] = int(body.split(b"element vertex ")[1].split(b"\n")[0])
        cut["body"] = body[:body.index(b"end_header\n") + len(b"end_header\n") + 100]
        route.fulfill(status=res.status, headers=res.headers, body=cut["body"])

    pg.route("**/api/ops/reconstruct", truncate)
    pg.click("button[data-action=run]")
    tab.wait_request()
    pg.wait_for_selector(f"{CLOUD}[data-state=error]", timeout=60_000)
    msg = pg.inner_text(f"{CLOUD} .notice.error .notice-body")
    assert msg.startswith("This point cloud cannot be drawn: ")
    assert f"the PLY body is too short for {cut['n']} vertices" in msg and "download above" in msg
    assert pg.locator(f"{CLOUD} figure").is_hidden() and pg.locator(f"{CLOUD} [data-testid=cloud-tools]").is_hidden()
    assert check_download(pg, cut["body"]) == "reconstruct-photo.ply"
    tab.a11y()
    assert tab.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("width", [1280, 768])
def test_a_point_cloud_result_in_both_themes_down_to_tablet_width(browser: Any, app: tuple[Any, str], image: Path,
                                                                  scheme: str, width: int) -> None:
    """reconstruct -f ply -p color=segment: the objects' colours exactly, unsegmented points grey."""
    t = Tab(browser, app[1], width=width, height=1000, scheme=scheme)
    try:
        pg = t.go("#/image?op=reconstruct")
        pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
        pg.wait_for_selector(".field[data-param=image] li.ready")
        pg.select_option(".field[data-param=format] select", "ply")
        pg.select_option(".field[data-param=attrs] .attr[data-attr=color] select", "segment")
        pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-p color=segment')")
        body = ply_body(pg, "reconstruct")
        check_drawn(pg, body)
        assert "color=segment" in pg.inner_text(f"{CLOUD} [data-fact='Each point carries']")
        bad = axe_violations(pg)
        assert not bad, f"a point-cloud result at {width}px ({scheme}):\n" + "\n".join(bad)
        over = t.js("() => document.documentElement.scrollWidth - window.innerWidth")
        assert over <= 0, f"a point-cloud result scrolls sideways by {over}px at {width}px"
        assert t.errors == []
    finally:
        t.close()


# --------------------------------------------------------------------------------- an image result


def test_an_image_result_is_offered_byte_for_byte(tab: Tab, image: Path) -> None:
    """A PNG result — the segmented image of segment.sh -f png, the depth image of reconstruct.sh
    -f depth — is the response body, in its media type, shown in the page as received and offered
    as a download byte for byte."""
    pg = tab.go("#/image?op=segment")
    pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
    pg.wait_for_selector(".field[data-param=image] li.ready")
    for op, fmt, name in (("segment", "png", "segment-photo.png"),
                          ("reconstruct", "depth", "reconstruct-photo.png")):
        pg.check(f"#op-{op}")
        pg.wait_for_selector(".field[data-param=image] li.ready")
        pg.select_option(".field[data-param=format] select", fmt)
        pg.wait_for_function(f"() => document.querySelector('[data-testid=command]').textContent.includes('-f {fmt}')")
        with operation_response(pg, op) as answer:
            pg.click("button[data-action=run]")
        tab.wait_request()
        assert answer.headers["content-type"] == "image/png"
        assert answer.body.startswith(b"\x89PNG\r\n\x1a\n")
        assert pg.get_attribute("[data-testid=result]", "data-format") == "png"
        assert "image/png" in pg.inner_text("[data-testid=result] .result-head")
        pg.wait_for_function("() => { const i = document.querySelector("
                             "'[data-testid=result-image] img'); return i && i.complete "
                             "&& i.naturalWidth > 0; }")
        assert check_download(pg, answer.body) == name
        tab.a11y()
    assert tab.errors == []


# ----------------------------------------------------------------------- maps: create, update, locate


@pytest.fixture(scope="module")
def mapped(browser: Any, app: tuple[Any, str], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Map creation and update through the guided flow (stub server, COLMAP)."""
    if shutil.which("colmap") is None:
        pytest.skip("needs Homebrew colmap")
    service, url = app
    src = tmp_path_factory.mktemp("frames")
    files = [Path(shutil.copy(f, src / f.name)) for f in FRAMES[:3]]
    t = Tab(browser, url)
    try:
        pg = t.go("#/maps")
        pg.click("[data-action=new-map]")
        wait_route(pg, "maps/new")
        field = ".field[data-param=inputs]"
        pg.set_input_files(f"{field} input[type=file]", [str(files[1]), str(files[0])])
        wait_ready(pg, "inputs", 2)
        order = pg.locator(f"{field} ol.files li .file-name").all_inner_texts()
        assert order == [files[1].name, files[0].name]
        pg.click(f"{field} button[aria-label='Move {files[1].name} later']")  # reorder
        order = pg.locator(f"{field} ol.files li .file-name").all_inner_texts()
        assert order == [files[0].name, files[1].name]
        pg.fill(".field[data-param=map] input", "room")
        pg.wait_for_selector(".map-note:has-text('A new map room will be created')")
        pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('room')")
        cmd = pg.inner_text("[data-testid=command]")
        assert cmd.index(files[0].name) < cmd.index(files[1].name)  # the order is sent
        t.a11y()
        pg.click("button[data-action=run]")
        dlg = pg.locator("dialog#confirm[open]")
        dlg.wait_for()
        assert "leaves the map exactly as it was" in dlg.inner_text() and "many minutes" in dlg.inner_text()
        t.a11y()
        pg.click("#confirm-no")  # not now: nothing runs
        assert pg.locator("[data-testid=request]").count() == 0
        pg.click("button[data-action=run]")
        with operation_response(pg, "mapper-update") as created:
            pg.click("#confirm-yes")
        t.wait_request()
        assert json.loads(created.body)["openlabel"]
        pg.click("[data-action=open-map]")
        wait_route(pg, "maps/room")
        # the update: one more frame, the map fixed
        pg.click("[data-action=update-map]")
        wait_route(pg, "maps/room/update")
        assert pg.locator(".field[data-param=map]").count() == 0
        pg.set_input_files(f"{field} input[type=file]", [str(files[2])])
        wait_ready(pg, "inputs")
        pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('room')")
        pg.click("button[data-action=run]")
        with operation_response(pg, "mapper-update") as updated:
            pg.click("#confirm-yes")
        t.wait_request()
        assert check_download(pg, updated.body) == "mapper-update-room.json"
        assert pg.locator(f"{field} li").count() == 0  # these inputs are in the map now
        assert not {f.name for f in files} & set(uploads(service))  # consumed, not renewed
        assert t.errors == []
        return {"name": "room", "frames": files}
    finally:
        t.close()


def test_map_cards_filter_and_page(mapped: dict[str, Any], tab: Tab, app: tuple[Any, str]) -> None:
    service, _ = app
    pg = tab.go("#/maps?filter=&junk=1")
    card = pg.locator("li.map-card[data-map=room]")
    card.wait_for()
    assert "Frames" in card.inner_text() and "Updates" in card.inner_text()
    assert "junk" not in pg.url  # a key the page does not keep is dropped
    # in plain words what a map's page gives, and a link straight to its bird's-eye image
    lead = pg.inner_text("[data-testid=maps-lead]")
    assert "the bird's-eye image of the map" in lead and ".sh" not in lead
    link = card.locator("[data-action=image-link]")
    assert link.inner_text() == "Bird's-eye image of the map"
    assert link.get_attribute("href") == "#/maps/room?op=mapper-locate&format=png&run=1"
    pg.fill("#map-filter", "nothing-like-this")
    assert pg.locator("li.map-card").count() == 0
    pg.fill("#map-filter", "roo")
    assert pg.locator("li.map-card").count() == 1 and "filter=roo" in pg.url
    tab.a11y()
    card.locator("a.card-link").click()
    wait_route(pg, "maps/room")
    # the summary and the update history with timings, from map.json
    meta = json.loads((service.workspace.maps / "room" / "map.json").read_text())
    rows = pg.locator("[data-testid=history] tbody tr[data-update]")
    rows.first.wait_for()
    assert rows.count() == len(meta["updates"]) == 2
    pg.click("[data-testid=history] tr.update-detail >> nth=1 >> summary")
    stages = pg.locator("[data-testid=history] tr.update-detail >> nth=1 >> [data-testid=stages] tbody th code")
    assert stages.all_inner_texts() == list(meta["updates"][-1]["timings"]["stages_s"])
    # the operations that take a map without writing it, nothing else: one, said without a choice
    (op,) = [o for o in web_ops.operations().values()
             if o.options and any(p.kind is spec.Kind.MAP for p in o.options) and not o.writes_map()]
    assert pg.get_attribute("[data-testid=operations]", "data-op") == op.id
    assert pg.locator("[data-testid=operations] input[type=radio]").count() == 0
    # above the viewer, saying what it gives: every format of its result
    sections = pg.locator("main > section h2").all_inner_texts()
    assert sections.index("Run on this map") < sections.index("Viewer") < sections.index("Update history")
    produces = pg.inner_text("[data-testid=produces]")
    assert "the bird's-eye image of the map (PNG)" in produces and "by its format" in produces
    # the history counts the detections each update tracked, confirmed or not — under a name
    # of its own, never the summary's Objects (those the map lists) or candidates
    head = pg.locator("[data-testid=history] thead th").all_inner_texts()
    assert "Tracked" in head and "Objects" not in head and "Candidates" not in head
    title = pg.get_attribute("[data-testid=history] thead th:has-text('Tracked')", "title")
    assert "confirmed or not" in title and "summary's Objects" in title
    last = meta["updates"][-1]["objects"]
    row = pg.locator("[data-testid=history] tbody tr[data-update]").last.locator("td").all_inner_texts()
    assert row[-2].split("\n") == [f"{last['confirmed']} confirmed", f"+ {last['unconfirmed']} unconfirmed"]
    tab.a11y()
    # a card's link: the map's page runs its bird's-eye image at once and shows it, the focus on
    # it, and the URL no longer asks to run (a reload or a bookmark does not run it unasked)
    pg = tab.go("#/maps")
    pg.click("li.map-card[data-map=room] [data-action=image-link]")
    wait_route(pg, "maps/room")
    pg.wait_for_selector("[data-testid=result][data-format=png] [data-testid=result-image] img", timeout=180_000)
    pg.wait_for_function("() => document.activeElement.id === 'result-h'")
    assert pg.input_value(".field[data-param=format] select") == "png"
    assert "run=" not in pg.url and "format=png" in pg.url
    pg.reload()
    pg.wait_for_selector("body[data-ready=true]")
    pg.wait_for_timeout(800)
    assert pg.locator("[data-testid=request]").count() == 0  # not run again
    assert tab.errors == []


def test_locating_images_in_a_map(mapped: dict[str, Any], tab: Tab, image: Path) -> None:
    pg = tab.go("#/maps/room?op=mapper-locate")
    assert pg.get_attribute("[data-testid=operations]", "data-op") == "mapper-locate"
    # an image the map cannot place: left out of the result, said in a warning naming the file
    # chosen (and its number among the inputs), not its upload's path
    pg.set_input_files(".field[data-param=inputs] input[type=file]", [str(image), str(FRAMES[1])])
    wait_ready(pg, "inputs", 2)
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-m maps/room')")
    with operation_response(pg, "mapper-locate") as answer:
        pg.click("button[data-action=run]")
    tab.wait_request()
    body = answer.body
    doc = json.loads(body)
    assert doc["openlabel"]
    assert listed_objects(pg) == scene_objects(doc)
    # the result is in view, its heading focused
    assert pg.evaluate("() => document.activeElement.id") == "result-h"
    warned = json.loads(answer.headers["x-oms-warnings"])
    assert [w["inputs"][0]["index"] for w in warned] == [0]
    warning = pg.inner_text("[data-testid=warnings]")
    assert f"{image.name} (input 1): not located" in warning and "uploads/" not in warning
    # -t single: the located camera alone, said to be located, named by the file chosen
    cams = pg.locator("[data-testid=cameras] tbody tr").evaluate_all(
        "trs => trs.map(t => [t.dataset.camera, t.dataset.kind, t.firstElementChild.textContent,"
        " t.lastElementChild.textContent])")
    # one numbering: the input's, from 1, with the camera's key (as the warning and the
    # bird's-eye image's tags number them)
    assert cams == [["located_1", "located", "input 2 (located_1)", FRAMES[1].name]]
    title = pg.get_attribute("[data-testid=cameras] tbody td.cam-image span", "title")
    assert title.endswith(f"/{FRAMES[1].name}") and "/uploads/" in title  # the path, as a tooltip
    # no objects: the result's scope lists none, and the other scope does
    assert "scope is single" in pg.inner_text("[data-testid=no-objects]")
    assert check_download(pg, body) == "mapper-locate-room.json"
    tab.a11y()
    # the page returns on its URL with the same operation chosen
    pg.reload()
    pg.wait_for_selector("body[data-ready=true]")
    assert pg.get_attribute("[data-testid=operations]", "data-op") == "mapper-locate"
    assert tab.errors == []


def test_a_point_cloud_result_on_a_maps_page(mapped: dict[str, Any], tab: Tab) -> None:
    """mapper.sh locate -f ply -p color=segment on a map's page: the map points the located camera
    sees in their objects' colours (unsegmented points grey), drawn in 3D as they are (map frame,
    z up)."""
    pg = tab.go("#/maps/room?op=mapper-locate")
    pg.set_input_files(".field[data-param=inputs] input[type=file]", str(FRAMES[1]))
    pg.wait_for_selector(".field[data-param=inputs] li.ready")
    pg.select_option(".field[data-param=format] select", "ply")
    pg.select_option(".field[data-param=attrs] .attr[data-attr=color] select", "segment")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-p color=segment')")
    body = ply_body(pg, "mapper-locate")
    check_drawn(pg, body)
    assert "color=segment" in pg.inner_text(f"{CLOUD} [data-fact='Each point carries']")
    assert "z up" in pg.inner_text(f"{CLOUD} [data-fact=Frame]") and "upright" not in pg.inner_text(f"{CLOUD} [data-fact=Frame]")
    first = settle(pg)
    pg.click("[data-move='Rotate right']")
    assert turned(settle(pg), first) > 10
    pg.click("[data-move='Zoom out']")
    assert distance(settle(pg)) > distance(first) * 1.2
    assert check_download(pg, body) == "mapper-locate-room-color-segment.ply"  # its variant
    # the camera its header names is drawn at its pose, in its kind's colour and labelled, and
    # said in the caption with the file chosen
    assert pg.inner_text(f"{CLOUD} [data-fact=Camera]") == \
        f"input 1 (located_0: {FRAMES[1].name}), drawn at its pose"
    assert pg.evaluate(f"document.querySelector('{CLOUD}').cloudView.viewer.cameras.length") == 1
    assert pg.evaluate(f"document.querySelector('{CLOUD}').cloudView.viewer.groups.cameras.children"
                       ".map(c => c.name)") == ["frustums-located"]
    pg.wait_for_selector(f"{CLOUD} .obj-label[data-group=cameras]:not([hidden])")
    # labelled by the input it placed, as the tables and the bird's-eye image number it
    assert pg.inner_text(f"{CLOUD} .obj-label[data-group=cameras] .tag") == "input 1"
    assert "located_0" in pg.get_attribute(f"{CLOUD} .obj-label[data-group=cameras]", "title")
    assert pg.get_attribute(f"{CLOUD} .labels", "aria-label") == "Camera labels"
    tab.a11y()
    assert tab.errors == []


def test_locating_without_images_shows_every_format(mapped: dict[str, Any], tab: Tab) -> None:
    """With no image chosen, locate runs on the map alone: its last known camera. Each format of
    the result is shown: the scene description with its objects and cameras listed, the point
    cloud in 3D with the camera its header names, the bird's-eye image as received (its layers a
    field of the PNG format only)."""
    pg = tab.go("#/maps/room?op=mapper-locate")
    assert pg.locator(".field[data-param=inputs] li").count() == 0  # nothing chosen
    pg.select_option(".field[data-param=mode] select", "full")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-t full')")
    command = pg.inner_text("[data-testid=command]")
    assert "-m maps/room" in command and " -i " not in command
    assert pg.locator(".field[data-param=layers]").is_hidden()  # -l: the PNG's only
    with operation_response(pg, "mapper-locate") as answer:
        pg.click("button[data-action=run]")
    tab.wait_request()
    doc = json.loads(answer.body)
    assert listed_objects(pg) == scene_objects(doc)
    rows = pg.locator("[data-testid=cameras] tbody tr")
    kinds = rows.evaluate_all("trs => trs.map(t => [t.dataset.camera, t.dataset.kind, t.children[1].textContent])")
    frames = doc["openlabel"]["frames"].values()
    (last,) = [f["frame_properties"] for f in frames if f["frame_properties"].get("last_known")]
    keyframes = sorted(f["frame_properties"]["keyframe"] for f in frames
                       if not f["frame_properties"].get("last_known"))
    assert kinds == [["last_known", "last_known", "Last known camera"],
                     *(([k, "frame", "Map frame"]) for k in keyframes)]
    x, y, z = last["transforms"]["last_known_to_map"]["transform_src_to_dst"]["translation"]
    assert rows.first.locator("td.num").all_inner_texts() == [f"{v:.3f}" for v in (x, y, z)]
    assert check_download(pg, answer.body) == "mapper-locate-room-full.json"
    tab.a11y()
    # the point cloud: the map cloud, the last known camera drawn at its pose
    body = ply_body(pg, "mapper-locate")
    assert b"\ncomment last_known {" in body
    assert pg.inner_text(f"{CLOUD} [data-fact=Camera]") == \
        "last_known (the last known camera), drawn at its pose"
    assert pg.evaluate(f"document.querySelector('{CLOUD}').cloudView.viewer.cameras[0].name") == "last_known"
    assert "and 1 camera" in pg.get_attribute(CANVAS, "aria-label")
    tab.a11y()
    # the bird's-eye image, of the OBB layer alone: its format says what it gives
    png = pg.locator(".field[data-param=format] option[value=png]").inner_text()
    assert png == "png — the bird's-eye image of the map"
    assert "Scope" in pg.inner_text(".field[data-param=mode] label")
    # a scope's choice in a few words, which the closed select shows whole
    assert pg.locator(".field[data-param=mode] option[value=single]").inner_text() == \
        "single — only these cameras (default)"
    assert pg.eval_on_selector(".field[data-param=mode] select", "s => s.scrollWidth <= s.clientWidth")
    pg.select_option(".field[data-param=format] select", "png")
    pg.wait_for_selector(".field[data-param=layers]:not([hidden])")
    pg.select_option(".field[data-param=layers] select", "obb")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-l obb')")
    with operation_response(pg, "mapper-locate") as answer:
        pg.click("button[data-action=run]")
    pg.wait_for_selector("[data-testid=result][data-format=png]", timeout=180_000)  # this run's
    assert answer.headers["content-type"] == "image/png"
    assert answer.body.startswith(b"\x89PNG\r\n\x1a\n")
    pg.wait_for_function("() => { const i = document.querySelector('[data-testid=result-image] img');"
                         " return i && i.complete && i.naturalWidth > 0; }")
    from PIL import Image

    with Image.open(io.BytesIO(answer.body)) as img:
        size, text = img.size, dict(img.text)  # type: ignore[attr-defined]
    assert pg.evaluate("document.querySelector('[data-testid=result-image] img').naturalWidth") == size[0]
    assert text["layers"] == "obb" and "f000000" in json.loads(text["cameras"])  # -t full
    assert check_download(pg, answer.body) == "mapper-locate-room-full-obb.png"
    # what its own metadata says: description (its text alternative too), scale, layers, legend,
    # the cameras drawn, and a link to it at full size
    alt = pg.get_attribute("[data-testid=result-image] img", "alt")
    assert alt.startswith(f"The bird's-eye image of the map, as received, {size[0]} × {size[1]} px. ")
    assert text["Description"] in alt
    assert pg.inner_text("[data-testid=image-facts] [data-fact=Layers]") == "obb"
    assert pg.inner_text("[data-testid=image-facts] [data-fact=Scale]").startswith("1 px = ")
    assert pg.locator("[data-testid=scale-bar]").is_visible()
    marks = pg.locator("[data-testid=legend] li").evaluate_all("ls => ls.map(l => l.dataset.mark)")
    assert marks == [m["mark"] for m in json.loads(text["legend"])] and "keyframe" in marks
    drawn = pg.locator("[data-testid=result-image] ~ details [data-testid=cameras] tbody tr")
    assert drawn.count() == len(json.loads(text["cameras"]))
    full = pg.get_attribute("[data-testid=full-size]", "href")
    assert full.startswith("blob:") and pg.get_attribute("[data-testid=full-size]", "target") == "_blank"
    tab.a11y()
    # the form's values are in the page's URL: reloaded, the same bird's-eye view is one Run away
    assert "format=png" in pg.url and "layers=obb" in pg.url and "mode=full" in pg.url
    pg.reload()
    pg.wait_for_selector("body[data-ready=true]")
    assert pg.input_value(".field[data-param=format] select") == "png"
    assert pg.input_value(".field[data-param=layers] select") == "obb"
    assert pg.input_value(".field[data-param=mode] select") == "full"
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-l obb')")
    assert tab.errors == []


def test_a_missing_maps_page_offers_no_action_on_it(tab: Tab) -> None:
    """A map that is not there: its page says so and points to creating one, with nothing to run
    on it, update, view or remove."""
    pg = tab.go("#/maps/nope")
    pg.wait_for_selector("[data-testid=remove]", state="detached")
    for gone in ("[data-testid=run-on-map]", "[data-action=update-map]", "[data-testid=viewer]",
                 "[data-action=run]"):
        assert pg.locator(gone).count() == 0, gone
    assert pg.get_attribute("main [data-action=new-map]", "href") == "#/maps/new"
    pg.wait_for_timeout(600)  # its form's first check would have come by now
    assert "?" not in pg.url  # no operation kept for a map that is not there
    tab.a11y()
    # its update flow: the same, never "the new inputs extend map nope"
    pg = tab.go("#/maps/nope/update?format=ply")
    pg.wait_for_selector("main .notice.error")
    assert pg.locator("section.step").count() == 0 and pg.locator("[data-action=run]").count() == 0
    assert "extend map nope" not in pg.inner_text("main")
    assert pg.get_attribute("main [data-action=new-map]", "href") == "#/maps/new"
    assert "format=" not in pg.url
    tab.a11y()
    assert [e for e in tab.errors if "/api/maps/nope" not in e and "404" not in e] == []


# what the update records of the workspace's real maps hold (ainex-captures, test), for their
# history's width: their dates, counts, registration, objects and times
REAL_UPDATES = {
    "ainex-captures": {"id": 1, "at": 1792592810.312225, "inputs": [f"/in/{i:04d}.jpg" for i in range(79)],
                       "kind": "images", "fps": None, "frames_added": [f"f{i:06d}" for i in range(79)],
                       "frames_rejected": [], "sfm": "multiview",
                       "objects": {"total": 44, "confirmed": 33, "unconfirmed": 11},
                       "timings": {"stages_s": {"setup": 1.0, "inference": 100.0}, "total_s": 160.386}},
    "test": {"id": 1, "at": 1792617707.6158788, "inputs": [f"/in/{i:04d}.jpg" for i in range(36)],
             "kind": "images", "fps": None, "frames_added": [f"f{i:06d}" for i in range(36)],
             "frames_rejected": [], "sfm": "multiview",
             "objects": {"total": 58, "confirmed": 30, "unconfirmed": 28},
             "timings": {"stages_s": {"setup": 1.0, "inference": 60.0}, "total_s": 88.197}},
}


def test_the_update_history_fits_a_tablet(tab: Tab, app: tuple[Any, str]) -> None:
    """The history of maps like the workspace's real ones never scrolls sideways at 768 px, with
    a page scrollbar or without one (the page's width less a classic scrollbar's 15 px): its date
    and time take two lines."""
    service, _ = app
    for name, update in REAL_UPDATES.items():
        root = minimal_map(service.workspace.maps / name)
        meta = json.loads((root / "map.json").read_text())
        (root / "map.json").write_text(json.dumps({**meta, "updates": [update]}))
    for width in (768, 753):
        for scrollbar in ("scroll", "hidden"):
            pg = tab.go("#/maps")
            pg.set_viewport_size({"width": width, "height": 1024})
            for name in REAL_UPDATES:
                pg = tab.go(f"#/maps/{name}")
                pg.wait_for_selector("[data-testid=history] tbody tr[data-update]")
                pg.evaluate("(o) => { document.documentElement.style.overflowY = o; }", scrollbar)
                over = pg.eval_on_selector("[data-testid=history]",
                                           "t => t.parentElement.scrollWidth - t.parentElement.clientWidth")
                assert over <= 0, f"{name}'s history scrolls sideways by {over}px at {width}px ({scrollbar})"
                # with room to spare: its narrowest layout fits with 20 px left (a wider font)
                least, room = pg.eval_on_selector("[data-testid=history]", """t => {
                    t.style.width = 'min-content'; const w = t.getBoundingClientRect().width; t.style.width = '';
                    return [w, t.parentElement.clientWidth]; }""")
                assert least <= room - 20, (name, width, scrollbar, least, room)
                assert pg.eval_on_selector("[data-testid=history] td.when", "td => td.innerText.split('\\n').length") == 2
                assert t_page_fits(pg), f"{name} scrolls sideways at {width}px"
    pg.set_viewport_size({"width": 1280, "height": 900})
    tab.a11y()
    assert tab.errors == []


def t_page_fits(pg: Any) -> bool:
    return pg.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")


def test_a_run_asked_by_the_url_that_may_not_run_unasked_shows_where_to_run_it(tab: Tab) -> None:
    """``run=1`` with a request that does not pass its check: nothing runs; the Run button is
    focused, in view, and the URL no longer asks to run."""
    pg = tab.go("#/maps/empty?op=mapper-locate&format=ply&attrs=voxel%3D-1&run=1")
    pg.wait_for_function("() => document.activeElement?.dataset.action === 'run'")
    assert pg.locator("[data-testid=request]").count() == 0
    assert pg.eval_on_selector("button[data-action=run]", "b => { const r = b.getBoundingClientRect();"
                               " return r.top >= 0 && r.bottom <= window.innerHeight; }")
    pg.wait_for_function("() => !location.hash.includes('run=')")
    assert tab.errors == []


def test_a_pages_url_keeps_its_form_and_nothing_else(tab: Tab) -> None:
    """A map's page, the Image page and the map flows keep their forms' values in the URL (never
    the files chosen), drop the keys they do not keep, and bring the form back on a reload; the
    operations are described in the web application's words (nothing of the command line's -o)."""
    pg = tab.go("#/maps/empty?op=mapper-locate&inputs=foo.jpg&format=png&nonsense=1")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-f png')")
    pg.wait_for_function("() => !location.hash.includes('inputs=') && !location.hash.includes('nonsense')")
    assert "format=png" in pg.url and "op=mapper-locate" in pg.url
    pg = tab.go("#/maps/new?format=ply&map=room2&inputs=x.jpg")
    assert pg.input_value(".field[data-param=format] select") == "ply"
    assert pg.input_value(".field[data-param=map] input") == "room2"
    pg.wait_for_function("() => !location.hash.includes('inputs=')")
    pg.select_option(".field[data-param=format] select", "json")
    pg.wait_for_function("() => !location.hash.includes('format=')")  # the default: no key
    assert "map=room2" in pg.url and "op=" not in pg.url
    pg = tab.go("#/image")
    text = pg.inner_text("[data-testid=operations]")
    assert "-o" not in text and "stdout" not in text and "artefacts" not in text
    tab.a11y()
    assert tab.errors == []


def test_a_request_that_ends_leaves_the_focus_where_the_user_put_it(extended: Any, tab: Tab) -> None:
    """The end of a request moves the focus to its result only while the focus is still on its
    Run or Interrupt button (or nowhere): a user who went on typing keeps the focus and the
    scroll, and hears that it ended."""
    service = extended
    pg = tab.go("#/maps/empty?op=slow")
    start_slow(pg, 2)
    field = ".field[data-param=seconds] input"
    pg.focus(field)
    scrolled = pg.evaluate("() => window.scrollY")
    tab.wait_request("done", 60000)
    assert pg.evaluate("() => document.activeElement.closest('.field')?.dataset.param") == "seconds"
    assert pg.evaluate("() => window.scrollY") == scrolled
    assert pg.inner_text("#announce") == "Done: the result is below."
    wait_idle(service)
    # left on the Run button, the focus goes to the result
    pg.click("button[data-action=run]")
    tab.wait_request("done", 60000)
    pg.wait_for_function("() => document.activeElement.id === 'result-h'")
    assert tab.errors == []


# ----------------------------------------------------- interruption, waiting for its turn, registry


def test_a_maps_viewer_and_its_removal(tab: Tab, app: tuple[Any, str]) -> None:
    """A map's page starts the map's viewer — a link to its page while it runs (view.sh's own page,
    served on 127.0.0.1 only, which the page says) — and stops it, and removes the map after a
    confirmation that states it is deleted for good (the safe choice is focused; Escape keeps the
    map); the page then says it is gone. By keyboard, with no accessibility violation."""
    service, _ = app
    root = minimal_map(service.workspace.maps / "doomed")
    pg = tab.go("#/maps/doomed")
    wait_route(pg, "maps/doomed")
    status, button = pg.locator("[data-testid=viewer-status]"), pg.locator("[data-action=viewer]")
    link = pg.locator("[data-testid=viewer-link]")
    pg.wait_for_function("() => document.querySelector('[data-testid=viewer-status]').textContent "
                         "=== 'The viewer is stopped.'")
    assert button.inner_text() == "Start the viewer" and link.is_hidden()
    button.focus()
    pg.keyboard.press("Enter")
    link.wait_for(state="visible")
    url = link.get_attribute("href")
    assert url == service.viewers.state("doomed")["url"] and url.startswith("http://127.0.0.1:")
    assert "served on the machine that runs the service only" in status.inner_text()
    assert button.inner_text() == "Stop the viewer"
    assert pg.evaluate("() => document.activeElement.dataset.action") == "viewer"  # focus kept
    tab.a11y()
    with tab.ctx.expect_page() as opened:
        link.click()
    viewer = opened.value
    viewer.wait_for_selector('body[data-rendered="true"]', timeout=60000)
    assert viewer.url == url and viewer.title() == "doomed — oh-my-slam"  # view.sh's page
    viewer.close()
    button.focus()
    pg.keyboard.press("Enter")  # stop it
    link.wait_for(state="hidden")
    assert status.inner_text() == "The viewer is stopped."
    assert service.viewers.state("doomed") == {"running": False, "url": None}
    pg.keyboard.press("Enter")  # running again: the removal says it stops first
    link.wait_for(state="visible")
    remove = pg.locator("[data-action=remove-map]")
    remove.focus()
    pg.keyboard.press("Enter")
    dialog = pg.locator("#confirm")
    dialog.wait_for(state="visible")
    text = dialog.inner_text()
    assert "Remove map doomed for good?" in text and "Its viewer is stopped first." in text
    assert "is deleted from this workspace for good" in text
    assert pg.evaluate("() => document.activeElement.id") == "confirm-no"  # the safe choice
    assert pg.inner_text("#confirm-no") == "Keep the map"
    tab.a11y()
    pg.keyboard.press("Escape")
    dialog.wait_for(state="hidden")
    assert (root / "map.json").is_file() and service.viewers.state("doomed")["running"]
    remove.click()
    pg.click("#confirm-yes")
    pg.wait_for_selector("h1:has-text('Map doomed removed')")
    assert pg.evaluate("() => document.activeElement.tagName") == "H1"
    assert "was deleted from this workspace for good" in pg.inner_text("main")
    assert not root.exists() and not service.viewers.state("doomed")["running"]
    tab.a11y()
    pg.click("text=See the maps of this workspace.")
    wait_route(pg, "maps")
    pg.wait_for_selector("li.map-card[data-map=empty]")
    assert pg.locator("li.map-card[data-map=doomed]").count() == 0
    assert tab.errors == []


def test_a_refused_removal_says_why(tab: Tab, app: tuple[Any, str]) -> None:
    """A removal the service refuses (here: an update holds the map's lock) leaves the map and
    its page, with the service's message next to the button."""
    from oh_my_slam.mapping import store

    service, _ = app
    root = minimal_map(service.workspace.maps / "held")
    fd = store.lock_update(root)
    try:
        pg = tab.go("#/maps/held")
        wait_route(pg, "maps/held")
        pg.click("[data-action=remove-map]")
        pg.click("#confirm-yes")
        problem = pg.locator("[data-testid=remove] .notice.error .notice-body")
        problem.wait_for()
        assert problem.inner_text().startswith("Map held was not removed: another update is "
                                               "running on")
        assert problem.inner_text().endswith("; remove map held once that update ended")
        assert (root / "map.json").is_file() and pg.locator("h1").inner_text() == "Map held"
        tab.a11y()
    finally:
        os.close(fd)
    assert [e for e in tab.errors if "409" not in e] == []  # the refusal itself


@pytest.fixture
def extended(app: tuple[Any, str]) -> Iterator[Any]:
    """The registry with new modes — slow.sh -m MAP (reads a map, as long as asked), nap.sh -i
    IMAGE (needs the inference server, an option of a kind never seen, a rule with its own
    message) — and a new option on mapper.sh locate (--shade): no change to the web application."""
    service, _ = app
    loc = spec.MAPPER.command("locate")
    extra = spec.Option("--shade", "shade", spec.Kind.ENUM, "a new option of the export", default="dark",
                        choices=("dark", "light"))
    mapper = dataclasses.replace(spec.MAPPER, commands=(
        spec.MAPPER.command("update"), dataclasses.replace(loc, options=(*loc.options, extra))))
    programs = (*[p if p is not spec.MAPPER else mapper for p in spec.PROGRAMS],
                slow_command.map_registry_program(), slow_command.image_registry_program("required"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(spec, "PROGRAMS", programs)
        mp.setattr(web_ops.Operation, "module", property(
            lambda op: slow_command.MODULE if op.program.prog in ("slow.sh", "nap.sh")
            else f"oh_my_slam.cli.{op.program.prog.removesuffix('.sh')}"))
        service.ops = web_ops.operations()
        yield service
    service.ops = web_ops.operations()


def start_slow(pg: Any, seconds: float) -> None:
    """Run slow.sh -m on map empty for ``seconds`` from its page; returns once it runs."""
    if "#/maps/empty" not in pg.url:
        pg.goto(pg.url.split("#")[0] + "#/maps/empty?op=slow")
        pg.wait_for_selector("body[data-ready=true]")
    pg.fill(".field[data-param=seconds] input", str(seconds))
    pg.wait_for_function(f"() => document.querySelector('[data-testid=command]').textContent.includes('--seconds {seconds:g}')")
    pg.click("button[data-action=run]")
    pg.wait_for_selector("[data-testid=request][data-state=running]", timeout=30000)


def wait_idle(service: Any) -> None:
    import time

    deadline = time.monotonic() + 60
    while service.runner.counts() != {"running": 0, "waiting": 0}:
        assert time.monotonic() < deadline, service.runner.counts()
        time.sleep(0.1)


def test_interrupting_asks_first_and_stops_the_command(extended: Any, tab: Tab) -> None:
    service = extended
    pg = tab.go("#/maps/empty?op=slow")
    start_slow(pg, 60)
    assert "leaving or reloading the page interrupts it" in pg.inner_text("[data-testid=request]")
    elapsed = pg.inner_text("[data-testid=elapsed]")
    pg.wait_for_function("(e) => document.querySelector('[data-testid=elapsed]').textContent !== e", arg=elapsed)
    # the top bar says a request runs
    pg.wait_for_function("() => document.getElementById('requests').dataset.running === '1'")
    # the Interrupt button states the consequence; keeping it changes nothing
    pg.click("[data-action=interrupt]")
    dlg = pg.locator("dialog#confirm[open]")
    dlg.wait_for()
    assert "as Ctrl-C would" in dlg.inner_text() and "no result" in dlg.inner_text()
    tab.a11y()
    pg.click("#confirm-no")
    assert service.runner.counts()["running"] == 1
    assert pg.evaluate("() => document.activeElement.dataset.action") == "interrupt"  # focus back
    pg.click("[data-action=interrupt]")
    pg.click("#confirm-yes")
    tab.wait_request("interrupted", 30000)
    assert "Interrupted" in pg.inner_text("[data-testid=request] .notice")
    wait_idle(service)
    # leaving the page asks first; staying keeps it running, leaving interrupts it
    start_slow(pg, 60)
    pg.click("#nav a[data-page=image]")
    dlg.wait_for()
    assert "Leave this page" in dlg.inner_text()
    pg.click("#confirm-no")
    assert "#/maps/empty" in pg.url and pg.get_attribute("[data-testid=request]", "data-state") == "running"
    pg.click("#nav a[data-page=image]")
    pg.click("#confirm-yes")
    wait_route(pg, "image")
    pg.wait_for_selector("h1:has-text('Image')")
    wait_idle(service)
    # reloading asks too (the browser's own question): dismissed, it keeps running; accepted, it stops
    pg = tab.go("#/maps/empty?op=slow")
    start_slow(pg, 60)
    asked: list[str] = []

    def answer(d: Any) -> None:
        asked.append(d.type)
        if len(asked) == 1:
            d.dismiss()
        else:
            d.accept()

    pg.on("dialog", answer)
    pg.evaluate("() => { setTimeout(() => location.reload(), 0); }")
    pg.wait_for_timeout(1000)
    assert asked == ["beforeunload"] and service.runner.counts()["running"] == 1
    pg.evaluate("() => { setTimeout(() => location.reload(), 0); }")
    pg.wait_for_selector("body[data-ready=true]")
    assert asked == ["beforeunload", "beforeunload"]
    wait_idle(service)
    assert tab.errors == []


def test_a_request_waiting_for_its_turn_says_so(extended: Any, tab: Tab, image: Path) -> None:
    """Requests that use the inference server run one at a time: a page whose request waits behind
    another one shows it waiting, then running, and for how long."""
    service = extended
    url = service.url
    up = httpx.post(f"{url}api/uploads?name=first.jpg", content=image.read_bytes(),
                    headers={"content-type": "image/jpeg"}).json()
    first: dict[str, Any] = {}
    holder = threading.Thread(target=lambda: first.update(r=httpx.post(
        f"{url}api/ops/nap", json={"image": up["path"], "seconds": 4}, timeout=60)))
    holder.start()
    try:
        pg = tab.go("#/image?op=nap")
        pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
        pg.wait_for_selector(".field[data-param=image] li.ready")
        pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.startsWith('nap.sh')")
        assert service.runner.counts()["running"] == 1
        pg.click("button[data-action=run]")
        tab.wait_request("waiting", 10000)
        assert "Waiting for its turn" in pg.inner_text("[data-testid=request] .state")
        pg.wait_for_function("() => document.getElementById('requests').dataset.waiting === '1'")
        tab.a11y()
        tab.wait_request("done", 60000)
        assert "waiting for its turn" in pg.inner_text("[data-testid=request] .waited")
    finally:
        holder.join(60)
    assert first["r"].status_code == 200
    assert tab.errors == []


def test_a_registry_change_reaches_the_ui(extended: Any, tab: Tab, image: Path) -> None:
    """A new option is a new field, a new mode a new form on the page its inputs belong to, a new
    kind of option still a field, a new rule's message is flagged before submission, and a new
    error is shown in the command's words."""
    pg = tab.go("#/maps/empty?op=mapper-locate")
    pg.set_input_files(".field[data-param=inputs] input[type=file]", str(image))
    pg.wait_for_selector(".field[data-param=inputs] li.ready")
    shade = pg.locator(".field[data-param=shade] select")
    shade.wait_for()
    assert shade.input_value() == "dark" and "a new option of the export" in pg.inner_text(".field[data-param=shade]")
    shade.select_option("light")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('--shade light')")
    assert pg.locator("#op-slow").count() == 1  # the new mode over a map is on the map's page
    assert pg.is_checked("#op-mapper-locate")  # two operations on a map: a choice between them
    pg = tab.go("#/image")
    pg.check("#op-nap")
    mood = pg.locator(".field[data-param=mood] input")
    mood.wait_for()
    assert "how the nap feels" in pg.inner_text(".field[data-param=mood]")
    mood.fill("calm")
    pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
    pg.wait_for_selector(".field[data-param=image] li.ready")
    pg.fill(".field[data-param=seconds] input", "500")
    err = pg.locator(".field[data-param=seconds] .field-error")
    err.wait_for(state="visible")
    assert err.inner_text() == slow_command.NAP_LIMIT
    pg.fill(".field[data-param=seconds] input", "0.1")
    pg.fill(".field[data-param=code] input", "2")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('--mood calm')")
    pg.click("button[data-action=run]")
    tab.wait_request("failed", 60000)
    text = pg.inner_text("[data-testid=request] .notice.error")
    assert "asked to fail" in text and "usage" in text
    tab.errors = [e for e in tab.errors if "/api/ops/nap" not in e and "status of 400" not in e]  # its 400
    assert tab.errors == []


# ---------------------------------------------------------- every page: a11y, layout, URLs, keyboard


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_every_page_is_accessible_and_fits_a_tablet(browser: Any, app: tuple[Any, str], image: Path,
                                                    scheme: str) -> None:
    """axe-core (every impact of the WCAG 2.0/2.1 A and AA rules) on every page, a result
    included, in both themes, at desktop and tablet width; no page scrolls sideways."""
    service, url = app
    routes = ["#/image", "#/maps", "#/maps/new", "#/maps/empty", "#/maps/empty/update", "#/nowhere"]
    for width in (1280, 768):
        t = Tab(browser, url, width=width, height=1000, scheme=scheme)
        try:
            for r in routes:
                pg = t.go(r)
                pg.wait_for_timeout(300)
                bad = axe_violations(pg)
                assert not bad, f"{r} at {width}px ({scheme}):\n" + "\n".join(bad)
                over = t.js("() => document.documentElement.scrollWidth - window.innerWidth")
                assert over <= 0, f"{r} scrolls sideways by {over}px at {width}px"
            # a page with a result and a flagged field
            pg = t.go("#/image?op=segment")
            pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
            pg.wait_for_selector(".field[data-param=image] li.ready")
            pg.click("button[data-action=run]")
            t.wait_request()
            pg.click("[data-testid=result] .stages-box summary")
            pg.fill(".field[data-param=min_score] input", "x")
            pg.wait_for_selector(".field[data-param=min_score] .field-error:not(:empty)")
            bad = axe_violations(pg)
            assert not bad, f"a result at {width}px ({scheme}):\n" + "\n".join(bad)
            over = t.js("() => document.documentElement.scrollWidth - window.innerWidth")
            assert over <= 0, f"a result scrolls sideways by {over}px at {width}px"
            t.errors = [e for e in t.errors if "/api/ops/" not in e]
            assert t.errors == []
        finally:
            t.close()


def test_pages_respond_within_100_ms_and_keep_their_url(tab: Tab) -> None:
    """Navigation renders at once (no request in the way): the page's own timing of a route, from
    the hash change to the rendered page, has a median under 100 ms over 5 navigations per page —
    measured in the page, so a loaded machine's delays around the test's calls do not count. The
    new page's heading gets the focus and is announced; each page's URL brings it back."""
    pg = tab.go("#/image")
    for page, heading in (("maps", "Maps"), ("image", "Image")):
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
        assert pg.inner_text("#announce") == f"{heading} page"
        url = pg.url
        pg.reload()
        pg.wait_for_selector("body[data-ready=true]")
        assert pg.inner_text("main h1") == heading and pg.url == url
    # a map's page, the flows and a filter have their own URL
    for route, heading in (("#/maps/empty", "Map empty"), ("#/maps/new", "New map"),
                           ("#/maps/empty/update", "Update map empty")):
        pg = tab.go(route)
        assert pg.inner_text("main h1") == heading
    pg = tab.go("#/maps?filter=emp")
    assert pg.input_value("#map-filter") == "emp"
    assert tab.errors == []


def test_top_bar(tab: Tab, app: tuple[Any, str]) -> None:
    service, _ = app
    pg = tab.go("#/image")
    pg.wait_for_function(f"() => document.getElementById('workspace').textContent === '{service.workspace.root.name}'")
    pg.wait_for_function("() => document.body.dataset.inference === 'up'")
    assert "ready" in pg.inner_text("#inference")
    assert pg.inner_text("#requests") == "none"
    assert pg.locator("[data-testid=start-command]").count() == 0


def test_keyboard_reaches_every_control_with_visible_focus(tab: Tab, image: Path) -> None:
    pg = tab.go("#/image")
    pg.set_input_files(".field[data-param=image] input[type=file]", str(image))
    pg.wait_for_selector(".field[data-param=image] li.ready")
    seen = []
    for _ in range(40):
        pg.keyboard.press("Tab")
        info = pg.evaluate("""() => { const e = document.activeElement;
          const s = getComputedStyle(e); return [e.tagName, e.getAttribute('aria-label') || e.textContent.trim().slice(0, 30) || e.id,
            s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) >= 2]; }""")
        if info[0] == "BODY":  # past the last control: focus left the page
            break
        seen.append(info[1])
        assert info[2], f"no visible focus on {info}"
    assert "Choose a file…" in seen and "Remove photo.jpg" in seen and any(s.startswith("Run ") for s in seen)
    # the skip link moves to the page without leaving it
    pg.focus("[data-skip]")
    pg.keyboard.press("Enter")
    assert "#/image" in pg.url and pg.evaluate("() => document.activeElement.id") == "main"
