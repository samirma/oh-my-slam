"""The ``server.sh`` evaluation (http_server.md "Evaluation"), the paths the whole-section test does
not take, offline: variants from synthetic operation entries (attributes, applicability, a choice
listed twice), flags on the command line, the HTTP client against a local loopback server, the
browser tests' run (a stand-in for pytest), the workspace, start-up without a status, parity cases
that differ, fail or cannot be verified, the latency of a map's page, and the render time and
axe-core check with fake Playwright objects."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from oh_my_slam.tools.evaluate import service as sv
from oh_my_slam.tools.evaluate.runner import Runner
from oh_my_slam.tools.evaluate.service import Case, Inputs, Reply, ServiceEvaluation
from oh_my_slam.tools.evaluate.suite import Evaluation
from oh_my_slam.tools.evaluate.viewer import BrowserProbe
from tests.unit.test_cov_evaluate_runtime import closed_port
from tests.unit.test_evaluate_runner import fake_repo, script

# -- the cases, from synthetic operation entries ---------------------------------------------------


def param(name: str, kind: str, flag: str | None = None, **kw: Any) -> dict[str, Any]:
    return {"name": name, "flag": flag or f"--{name}", "kind": kind, "required": False,
            "default": None, "choices": None, "multiple": False, "applies": [], **kw}


def operation(*params: dict[str, Any], outputs: list[dict[str, Any]] | None = None
              ) -> dict[str, Any]:
    return {"id": "segment.sh -i", "prog": "segment.sh", "command": None, "mode": "image",
            "parameters": [param("image", "image", "-i", required=True), *params],
            "outputs": outputs or [{"via": "stdout", "format": "json"}]}


def test_variants_follow_attributes_choices_and_applicability() -> None:
    attrs = param("attrs", "attrs", "-p", attributes=[
        {"key": "stride", "schema": {"type": "integer"}, "default": "1"},
        {"key": "normals", "schema": {"type": "enum", "choices": ["off"]}, "default": "off"},
        {"key": "color", "schema": {"type": "enum", "choices": ["rgb", "segment"]},
         "default": "rgb"}])
    plain = param("plain", "attrs", "-q", attributes=[{"key": "stride",
                                                      "schema": {"type": "integer"}}])
    fmt = param("format", "enum", "-f", default="json", choices=["json", "ply", "ply"])
    level = param("level", "enum", "-l", default="low", choices=["low", "high"],
                  applies=[{"option": "level", "is": "given"}])
    never = [{"option": "missing", "in": ["x"]}, {"option": "image", "is": "video"}]
    quality = param("quality", "enum", "-Q", default="fast", choices=["fast", "best"],
                    applies=never)
    op = operation(attrs, plain, fmt, level, quality, outputs=[
        {"via": "stdout", "format": "png", "when": [{"option": "level", "is": "given"}]},
        {"via": "stdout", "format": "json"}, {"via": "-d", "format": "folder"}])
    done, skipped = sv._variants(op)
    # the first enumerated attribute with another choice; a choice listed twice runs once
    assert [name for name, _ in done] == ["default", "attrs=color=segment", "format=ply",
                                          "level=high"]
    assert skipped == [f"quality=best: no reference input meets {never}"]
    cases, uncovered = sv.parity_cases({"operations": [op]}, Inputs(image="inputs/a.jpg"))
    assert uncovered == {}
    assert [(c.variant, c.result_format) for c in cases] == [
        ("default", "json"), ("attrs=color=segment", "json"), ("format=ply", "json"),
        ("level=high", "png")]  # the output whose condition holds
    assert cases[3].params == {"image": "inputs/a.jpg", "level": "high"}


def test_flags_and_dash_paths_on_the_command_line() -> None:
    op = {"command": "update", "parameters": [
        param("verbose", "flag", "-v"), param("inputs", "images", "-i", multiple=True),
        param("map", "map", "-m"), param("quiet", "flag", "-s")]}
    assert sv.shell_argv(op, {"verbose": True, "quiet": False, "inputs": ["-x.jpg", "b.jpg"],
                              "map": "/w/m"}) == ["update", "-v", "-i", "./-x.jpg", "b.jpg",
                                                  "-m=/w/m"]


# -- HTTP --------------------------------------------------------------------------------------------


@contextmanager
def loopback() -> Iterator[str]:
    """A local HTTP server on a free port: /api/health answers, other GETs are 404, a POST
    echoes its body and content type."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def reply(self, status: int, body: bytes, headers: tuple[tuple[str, str], ...] = ()
                  ) -> None:
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/health":
                self.reply(200, b'{"status": "ok"}', (("Server-Timing", "total;dur=1.0"),))
            else:
                self.reply(404, b'{"error": "not found"}')

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.reply(200, json.dumps({"type": self.headers["Content-Type"],
                                        "body": body}).encode())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_api_client_answers_errors_as_replies() -> None:
    with loopback() as base:
        api = sv.Api(base)
        assert api.base == base + "/"
        ok = api.get("/api/health")
        assert ok.status == 200 and ok.json() == {"status": "ok"} and ok.seconds >= 0
        assert ok.headers["Server-Timing"] == "total;dur=1.0"
        missing = api.get("api/maps/none")
        assert missing.status == 404 and missing.json() == {"error": "not found"}
        echo = api.request("POST", "api/ops/reconstruct", {"image": "inputs/a.jpg"})
        assert echo.json() == {"type": "application/json", "body": {"image": "inputs/a.jpg"}}
    down = sv.Api(f"http://127.0.0.1:{closed_port()}/").get("api/health")
    assert down.status == 0 and b"refused" in down.body.lower()


# -- the browser tests -------------------------------------------------------------------------------


def ui_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests" / "browser").mkdir(parents=True)
    for name in ("test_webapp_browser.py", "test_webapp_down.py", "test_viewer_browser.py"):
        (repo / "tests" / "browser" / name).write_text("")
    return repo


def test_the_browser_tests_run_isolated_from_the_real_server(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = ui_repo(tmp_path)
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        seen.update(argv=argv, **kw)
        junit = next(a for a in argv if a.startswith("--junitxml=")).split("=", 1)[1]
        Path(junit).write_text('<testsuite><testcase classname="w" name="a"/>'
                               '<testcase classname="w" name="b"><failure/></testcase>'
                               "</testsuite>")
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setenv("OH_MY_SLAM_TEST_REAL_SERVER", "1")
    monkeypatch.setattr(sv.subprocess, "run", fake_run)
    res = sv.run_ui_tests(tmp_path / "ui", repo=repo)
    files = ["tests/browser/test_webapp_browser.py", "tests/browser/test_webapp_down.py"]
    assert (res.run, res.failed, res.error) == (2, 1, None)
    assert res.detail == {"files": files, "tests": 2, "skipped": 0, "failed": ["w::b"]}
    assert seen["argv"][:5] == [sys.executable, "-m", "pytest", "-m", "browser"]
    assert seen["argv"][-2:] == files and seen["cwd"] == repo
    assert "OH_MY_SLAM_TEST_REAL_SERVER" not in seen["env"]
    assert "OH_MY_SLAM_RUNTIME_DIR" not in seen["env"]  # the tests' own stub server
    assert seen["timeout"] == sv.UI_TIMEOUT_S and (tmp_path / "ui" / "pytest.log").is_file()

    def too_slow(argv: list[str], **kw: Any) -> None:
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(sv.subprocess, "run", too_slow)
    res = sv.run_ui_tests(tmp_path / "ui2", repo=repo)
    assert (res.run, res.failed) == (None, None)
    assert res.error == "the browser tests timed out after 3600 s"


# -- the evaluation, piece by piece ------------------------------------------------------------------


def evaluation(tmp_path: Path, **bodies: str) -> Evaluation:
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    return Evaluation(out, Runner(out, fake_repo(tmp_path, **bodies)), BrowserProbe(None))


class FakeApi:
    """The service's API as the evaluation uses it: GET replies by path, one POST reply."""

    def __init__(self, gets: dict[str, Reply] | None = None, post: Reply | None = None) -> None:
        self.base = "http://127.0.0.1:9/"
        self.gets, self.post = gets or {}, post
        self.asked: list[str] = []

    def get(self, path: str) -> Reply:
        self.asked.append(path)
        return self.gets.get(path, Reply(200, b"{}", 0.002))

    def request(self, method: str, path: str, body: Any = None,
                timeout: float | None = None) -> Reply:
        assert self.post is not None and method == "POST"
        return self.post


def service(ev: Evaluation, api: FakeApi | None = None, env: dict[str, str] | None = None,
            inputs_from: dict[str, Any] | None = None) -> ServiceEvaluation:
    se = ServiceEvaluation(ev, inputs_from or {}, env or {})
    se.api = api  # type: ignore[assignment]
    return se


def test_the_workspace_holds_copies_of_the_reference_inputs(tmp_path: Path) -> None:
    image = tmp_path / "b.jpg"
    image.write_bytes(b"B")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "map.json").write_text("{}")
    (ref / "latest").symlink_to("map.json")
    se = service(evaluation(tmp_path), inputs_from={
        "image": tmp_path / "missing.jpg", "images": [image, tmp_path / "gone.jpg"],
        "sequence": [image], "map": ref})
    inputs = se.workspace()
    assert inputs == Inputs(None, ("inputs/b.jpg",), ("inputs/b.jpg",), "reference")
    assert (se.ws / "inputs" / "b.jpg").read_bytes() == b"B"
    assert (se.ws / "maps" / "reference" / "latest").is_symlink()
    se = service(evaluation(tmp_path / "2"), inputs_from={"map": tmp_path / "no-map"})
    assert se.workspace() == Inputs()


QUIET_SERVER = r'''import http.server, signal, sys
signal.signal(signal.SIGINT, signal.default_int_handler)
if "--status" in sys.argv:
    print("server.sh: error: no status", file=sys.stderr)
    sys.exit(1)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers()
        self.wfile.write(b"{}")
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
print(f"server.sh: listening on http://127.0.0.1:{srv.server_address[1]}/", file=sys.stderr,
      flush=True)
try:
    srv.serve_forever(0.05)
except KeyboardInterrupt:
    pass
'''


def test_a_service_that_starts_without_a_status(tmp_path: Path) -> None:
    ev = evaluation(tmp_path)
    script(ev.runner.repo, "server.sh", QUIET_SERVER, shebang=f"#!{sys.executable}")
    se = service(ev)
    se.workspace()
    live = se.start()
    assert live is not None
    try:
        m = ev.metrics.items
        assert m["server_sh.start_s"].detail == {"health_status": 200}
        assert m["server_sh.resident_mb"].value is not None
        assert m["server_sh.resident_mb"].value > 0
        assert "health" not in se.details  # --status failed: nothing to keep
    finally:
        rec = se.stop(live)
    assert rec.ok and [r.tag for r in ev.runner.records] == ["server_sh_status", "server_sh"]
    assert ev.contracts.checks[("stdout", "server_sh")] == {"server_sh_status": [],
                                                           "server_sh": []}


MAPPER = """n=$(cat "$COUNTER" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$COUNTER"
[ -n "$FAIL" ] && { echo "mapper.sh: error: boom" >&2; exit 1; }
for a in "$@"; do case "$a" in -m=*) mkdir -p "${a#-m=}"; echo "{}" > "${a#-m=}/map.json";; esac
done
if [ -n "$VARY" ]; then printf '{"openlabel": {"run": %d}}' $n; else printf '{"openlabel": {}}'; fi
"""
UPDATE = {"id": "mapper.sh update", "prog": "mapper.sh", "command": "update", "mode": None,
          "parameters": [param("inputs", "images_or_video", "-i", required=True, multiple=True),
                         param("map", "map", "-m", required=True)]}


def mapping_case() -> Case:
    return Case("mapper-update", "mapper.sh update", "default",
                {"inputs": ["inputs/a.jpg"], "map": "parity-1"}, "map", "json")


@pytest.mark.parametrize(("env", "reply", "status", "why"), [
    ({}, Reply(200, b'{"openlabel": {}}', 0.5), "identical", None),
    ({"VARY": "1"}, Reply(200, b'{"openlabel": {"run": 3}}', 0.5), "unverifiable",
     "the command's own two runs differ: result: 25 vs 25 bytes, first difference at byte 22"),
    ({"FAIL": "1"}, Reply(200, b"{}", 0.5), "mismatch",
     "parity_01_shell failed (exit 1): mapper.sh: error: boom"),
    ({}, Reply(500, b'{"error": "internal"}', 0.5), "mismatch",
     "the request failed: HTTP 500: b'{\"error\": \"internal\"}'"),
])
def test_a_parity_case_and_its_control(tmp_path: Path, env: dict[str, str], reply: Reply,
                                       status: str, why: str | None) -> None:
    ev = evaluation(tmp_path, mapper=MAPPER)
    se = service(ev, FakeApi(post=reply), {"COUNTER": str(tmp_path / "count"), **env})
    se.workspace()
    row = se.run_case(UPDATE, mapping_case(), 1)
    assert row["status"] == status and row.get("why") == why
    assert row["case"] == "mapper.sh update [default]" and row["stages"] == []
    # each shell run created the map at the same path; it was set aside for the next one
    assert not (se.ws / "maps" / "parity-1").exists()
    aside = se.root / "shell_maps"
    made = sorted(p.name for p in aside.iterdir()) if aside.exists() else []
    assert made == ([] if env.get("FAIL") else ["parity_01_shell", "parity_01_shell_again"])
    if status == "unverifiable":
        assert row["request"] == ["result: 25 vs 25 bytes, first difference at byte 22"]
    if status == "identical":
        assert set(se.overheads["mapper-update"]) == {"request_s", "shell_s", "overhead_s"}
    args = ev.runner.records[0].argv[1:]
    assert args == ["update", "-i", str(se.ws / "inputs" / "a.jpg"),
                    f"-m={se.ws / 'maps' / 'parity-1'}"]


def test_parity_needs_the_services_description(tmp_path: Path) -> None:
    ev = evaluation(tmp_path)
    service(ev, FakeApi({"api/openapi.json": Reply(503, b"", 0.01)})).parity(Inputs())
    ids = [f"server_sh.parity.{k}" for k in sv.PARITY_METRICS]
    assert [ev.metrics.items[k].error for k in ids] == ["/api/openapi.json answered HTTP 503"] * 3
    ev = evaluation(tmp_path / "2")
    stray = json.dumps({"paths": {"/api/ops/stray": {"post": {}}}}).encode()
    se = service(ev, FakeApi({"api/openapi.json": Reply(200, stray, 0.01)}))
    se.parity(Inputs())
    m = ev.metrics.items
    assert m["server_sh.parity.mismatched"].value == 1
    assert m["server_sh.parity.mismatched"].detail["mismatched"] == [
        {"case": "/api/ops/stray", "why": "not an operation of commands.spec"}]
    covered = m["server_sh.parity.operations_covered_fraction"]
    assert covered.value is None and covered.error == "the service describes no operation"


def test_the_latency_of_reads_includes_the_maps_page(tmp_path: Path) -> None:
    ev = evaluation(tmp_path)
    api = FakeApi({"api/maps/reference": Reply(404, b"", 0.004)})
    service(ev, api).latency(Inputs(map="reference"))
    m = ev.metrics.items
    detail = m["server_sh.read_latency_median_ms"].detail
    assert detail["requests"] == sv.LATENCY_REPS * 5
    assert detail["failed"] == ["/api/maps/reference: HTTP 404"]
    assert detail["per_path_median_ms"]["/api/maps/reference"] == 4.0
    assert m["server_sh.read_latency_median_ms"].value == 2.0
    assert m["server_sh.read_latency_busy_p95_ms"].error == (
        "no read was made while an operation's request ran")
    assert m["server_sh.request_overhead_median_s"].error == (
        "no request could be compared with its command")


# -- the browser checks ------------------------------------------------------------------------------


class AppPage:
    """The web application in a fake browser page: renders, routes, and carries violations."""

    def __init__(self, violations: dict[str, list[str]] | None = None,
                 fail: str | None = None) -> None:
        self.violations, self.fail = violations or {}, fail
        self.url = ""
        self.calls: list[tuple[Any, ...]] = []
        self.axe = False
        self.closed = False

    def goto(self, url: str, timeout: float) -> None:
        if self.fail == "goto":
            raise RuntimeError("Timeout 60000ms exceeded.\nCall log: …")
        self.url = url
        self.calls.append(("goto", url))

    def wait_for_function(self, js: str, arg: Any = None, timeout: float = 0) -> None:
        self.calls.append(("wait", js, arg))

    def add_script_tag(self, path: str) -> None:
        self.calls.append(("axe", path))
        self.axe = True

    def evaluate(self, js: str) -> Any:
        if js == "() => !!window.axe":
            return self.axe
        if self.fail == "axe":
            raise RuntimeError("axe is not defined\n    at eval")
        return self.violations.get(self.url, [])

    def close(self) -> None:
        self.closed = True


class AppBrowser:
    def __init__(self, page: AppPage) -> None:
        self.page = page

    def new_page(self, viewport: dict[str, int]) -> AppPage:
        return self.page


def test_render_time_and_accessibility_of_the_web_application(tmp_path: Path) -> None:
    base = "http://127.0.0.1:9/"
    urls = sv.app_urls(base, "reference")
    page = AppPage({urls[1]: ["color-contrast (serious): Elements must meet contrast"]})

    @contextmanager
    def launch() -> Iterator[AppBrowser]:
        yield AppBrowser(page)

    ev = evaluation(tmp_path)
    ev.probe = BrowserProbe(launch)
    service(ev, FakeApi()).browser_checks(Inputs(map="reference"))
    m = ev.metrics.items
    assert m["server_sh.app_render_s"].value is not None and m["server_sh.app_render_s"].value >= 0
    a11y = m["server_sh.ui.a11y_violations"]
    assert a11y.value == 1 and list(a11y.detail["pages"]) == urls
    assert page.calls.count(("axe", str(sv.AXE))) == 1  # injected once, then kept
    routed = [c[2] for c in page.calls if c[0] == "wait" and c[1] == sv.ROUTED_JS]
    assert routed == ["image", "maps", "maps/new", "maps/reference", "maps/reference/update"]
    assert page.closed


def test_a_browser_that_does_not_start_fails_both_checks(tmp_path: Path) -> None:
    def launch() -> Any:
        raise RuntimeError("no Edge/Chrome for Playwright")

    ev = evaluation(tmp_path)
    ev.probe = BrowserProbe(launch)
    service(ev, FakeApi()).browser_checks(Inputs())
    for k in ("server_sh.app_render_s", "server_sh.ui.a11y_violations"):
        assert ev.metrics.items[k].error == "browser: no Edge/Chrome for Playwright"


def test_a_page_that_never_renders_or_fails_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    page = AppPage(fail="goto")
    assert ServiceEvaluation.render(AppBrowser(page), "http://x/") == (
        None, "the web application did not signal it had rendered: Timeout 60000ms exceeded.")
    assert page.closed
    page = AppPage(fail="axe")
    assert ServiceEvaluation.a11y(AppBrowser(page), ["http://x/"]) == (
        None, "axe-core: axe is not defined")
    assert page.closed and page.calls[1] == ("wait", sv.READY_JS, None)
    assert page.calls[2] == ("wait", sv.ROUTED_JS, "")  # a page without a hash route
    monkeypatch.setattr(sv, "AXE", sv.REPO / "tests" / "browser" / "vendor" / "none.js")
    assert ServiceEvaluation.a11y(AppBrowser(AppPage()), ["http://x/"]) == (
        None, "axe-core is not vendored at tests/browser/vendor/none.js")
