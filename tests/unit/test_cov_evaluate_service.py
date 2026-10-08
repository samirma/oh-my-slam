"""The ``server.sh`` evaluation (http_server.md "Evaluation"), the paths the whole-section test does
not take, offline: variants from synthetic operation entries (attributes, applicability, a choice
listed twice), flags on the command line, the HTTP client against a local loopback server, the
browser tests' run (a stand-in for pytest), the workspace, start-up without a status, a service
that does not start (each metric recorded once, the browser tests run all the same), parity cases
that differ, fail or cannot be verified, the latency of a map's page, and the render time and
axe-core check with fake Playwright objects."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from oh_my_slam.commands import spec
from oh_my_slam.tools.evaluate import service as sv
from oh_my_slam.tools.evaluate.runner import Runner
from oh_my_slam.tools.evaluate.service import Case, Inputs, Reply, ServiceEvaluation
from oh_my_slam.tools.evaluate.suite import Evaluation
from oh_my_slam.tools.evaluate.viewer import BrowserProbe
from tests.unit.test_cov_evaluate_runtime import closed_port
from tests.unit.test_evaluate_runner import fake_repo, script
from tests.unit.test_evaluate_service import served

# -- the cases, from synthetic operations ------------------------------------------------------------


def param(name: str, kind: str, flag: str | None = None, **kw: Any) -> dict[str, Any]:
    return {"name": name, "flag": flag or f"--{name}", "kind": kind, "required": False,
            "default": None, "choices": None, "multiple": False, "applies": [], **kw}


def synthetic(monkeypatch: pytest.MonkeyPatch, options: tuple[spec.Option, ...],
              entries: list[dict[str, Any]], outputs: tuple[spec.Output, ...]) -> sv.Op:
    """An operation of a program added to the registry: its definitions (``options``,
    ``outputs``) and its ``x-oms`` entry (``entries``: its parameters as the API offers them)."""
    image = spec.Option("-i", "image", spec.Kind.IMAGE, "image", required=True, type=Path)
    mode = spec.Mode(None, None, (), "never", "", (), outputs)
    cmd = spec.Command("probe.sh", None, "probe", (image, *options), (mode,))
    monkeypatch.setattr(spec, "PROGRAMS", (*spec.PROGRAMS, spec.Program("probe.sh", "", (cmd,))))
    entry = {"id": "probe.sh", "prog": "probe.sh", "command": None, "mode": None,
             "parameters": [param("image", "image", "-i", required=True), *entries]}
    return sv.Op("probe", entry, cmd, mode)


def test_variants_follow_attributes_choices_and_applicability(
        monkeypatch: pytest.MonkeyPatch) -> None:
    attrs = param("attrs", "attrs", "-p", attributes=[
        {"key": "stride", "schema": {"type": "integer"}, "default": "1"},
        {"key": "normals", "schema": {"type": "enum", "choices": ["off"]}, "default": "off"},
        {"key": "color", "schema": {"type": "enum", "choices": ["rgb", "segment"]},
         "default": "rgb"}])
    plain = param("plain", "attrs", "-q", attributes=[{"key": "stride",
                                                      "schema": {"type": "integer"}}])
    fmt = param("format", "enum", "-f", default="json", choices=["json", "ply", "ply"])
    level = param("level", "enum", "-l", default="low", choices=["low", "high"])
    quality = param("quality", "enum", "-Q", default="fast", choices=["fast", "best"])
    shape = param("shape", "enum", "-s", default="box", choices=["box", "ball"])
    never = (spec.When("missing", ("x",)), spec.When("image", video=True))
    given = spec.When("level")
    options = (
        spec.Option("-p", "attrs", spec.Kind.ATTRS, "attrs"),
        spec.Option("-q", "plain", spec.Kind.ATTRS, "plain"),
        spec.Option("-f", "format", spec.Kind.ENUM, "fmt", default="json",
                    choices=("json", "ply")),
        spec.Option("-l", "level", spec.Kind.ENUM, "level", default="low",
                    choices=("low", "high"), applies=(given,)),
        spec.Option("-Q", "quality", spec.Kind.ENUM, "quality", default="fast",
                    choices=("fast", "best"), applies=never),
        # applies with a format the parser refuses: setting it cannot make it apply
        spec.Option("-s", "shape", spec.Kind.ENUM, "shape", default="box",
                    choices=("box", "ball"), applies=(spec.When("format", ("glb",)),)),
    )
    outputs = (spec.Output("result", "stdout", "png", "", (spec.When("format", ("ply",)),)),
               spec.Output("result", "stdout", "json", ""),
               spec.Output("folder", "-d", "map", ""))
    op = synthetic(monkeypatch, options, [attrs, plain, fmt, level, quality, shape], outputs)
    done, skipped = sv._variants(op, {"image": "inputs/a.jpg"})
    # the first enumerated attribute with another choice; a choice listed twice runs once
    assert [name for name, _ in done] == ["default", "attrs=color=segment", "format=ply",
                                          "level=high"]
    assert skipped == [
        f"quality=best: no reference input meets {[w.describe() for w in never]}",
        "shape=ball: no reference input meets [{'option': 'format', 'in': ['glb']}]"]
    cases, uncovered = sv.parity_cases([op], [Inputs("a", image="inputs/a.jpg")])
    assert uncovered == {}
    assert [(c.variant, c.result_format) for c in cases] == [
        ("default", "json"), ("attrs=color=segment", "json"), ("format=ply", "png"),
        ("level=high", "json")]  # the output whose condition holds
    assert cases[3].params == {"image": "inputs/a.jpg", "level": "high"}
    assert sv._inapplicable(op, {"image": "inputs/a.jpg", "format": "glb"}, ["shape"]) is None


def test_the_command_line_is_the_registrys(tmp_path: Path) -> None:
    """The shell side's command line is ``commands.spec.argv_of``'s, as the service's: an option
    given its default that the registry omits then (``-fps``) is not passed."""
    update = next(o for o in served() if o.label == "mapper.sh update")
    se = service(evaluation(tmp_path))
    se.ws = tmp_path / "ws"
    case = Case(update.id, update.label, "default",
                {"inputs": ["inputs/v.mp4"], "map": "m", "fps": spec.DEFAULT_FPS}, "map")
    assert se.argv(update, case) == ["update", "-i", str(se.ws / "inputs" / "v.mp4"),
                                     f"-m={se.ws / 'maps' / 'm'}"]
    case.params["fps"] = 1.0
    assert se.argv(update, case)[-1] == "-fps=1.0"


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
            inputs_from: list[dict[str, Any]] | None = None) -> ServiceEvaluation:
    se = ServiceEvaluation(ev, inputs_from or [], env or {})
    se.api = api  # type: ignore[assignment]
    return se


def test_the_workspace_holds_copies_of_the_reference_inputs(tmp_path: Path) -> None:
    image = tmp_path / "b.jpg"
    image.write_bytes(b"B")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "map.json").write_text("{}")
    (ref / "latest").symlink_to("map.json")
    ev = evaluation(tmp_path)
    ws = sv.workspace_root(ev.out)
    video = sv.place(tmp_path / "b.jpg", sv.input_folder(ws, "street2.mp4"))  # the suite's
    assert video == ws / "inputs" / "street2" / "b.jpg" and video.stat().st_nlink == 2
    ran = sv.Ran(("update",), None)  # type: ignore[arg-type]
    se = service(ev, inputs_from=[
        {"name": "ainex-captures", "image": tmp_path / "missing.jpg",
         "images": [image, tmp_path / "gone.jpg"], "sequence": [image], "map": ref},
        {"name": "restaurant.jpg", "image": image},
        {"name": "street2.mp4", "sequence": [video], "ran": ran, "writes": "street2"}])
    ainex, restaurant, street2 = se.workspace()
    assert se.ws == ws
    assert ainex == Inputs("ainex-captures", None, ("inputs/ainex-captures/b.jpg",),
                           ("inputs/ainex-captures/b.jpg",), "ainex-captures")
    assert (se.ws / "inputs" / "ainex-captures" / "b.jpg").read_bytes() == b"B"
    assert (se.ws / "maps" / "ainex-captures" / "latest").is_symlink()
    assert restaurant == Inputs("restaurant.jpg", "inputs/restaurant/b.jpg")
    # a file already in the workspace stays where the suite's run used it
    assert street2 == Inputs("street2.mp4", sequence=("inputs/street2/b.jpg",), ran=ran,
                             writes="street2")
    se = service(evaluation(tmp_path / "2"), inputs_from=[{"map": tmp_path / "no-map"}])
    assert se.workspace() == [Inputs()]
    again = sv.place(image, sv.input_folder(ws, "street2.mp4"))  # placed again: replaced
    assert again == video and again.read_bytes() == b"B"
    assert sv.place(video, video.parent) == video and video.read_bytes() == b"B"  # onto itself


def test_a_file_system_without_hard_links_gets_a_copy(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    def no_link(src: Any, dst: Any) -> None:
        raise OSError("cross-device link")

    monkeypatch.setattr(sv.os, "link", no_link)
    (tmp_path / "v.mp4").write_bytes(b"V")
    placed = sv.place(tmp_path / "v.mp4", tmp_path / "ws" / "inputs" / "v")
    assert placed.read_bytes() == b"V" and placed.stat().st_nlink == 1


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


def test_a_service_that_does_not_start_records_each_metric_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """server.sh exits without listening: every metric that needs the service fails once, with
    the reason; the browser tests start a service of their own, so they still run and their
    metrics hold their outcome (spec §5: one pass/fail result per metric)."""
    ev = evaluation(tmp_path)  # the fake server.sh fails at once
    recorded: Counter[str] = Counter()
    add = ev.metrics.add

    def counted(mid: str, *args: Any, **kw: Any) -> None:
        recorded[mid] += 1
        add(mid, *args, **kw)

    monkeypatch.setattr(ev.metrics, "add", counted)
    ui = sv.UiOutcome(17, 1, None, {"failed": ["test_webapp.py::test_upload"]})
    ServiceEvaluation(ev, [], {}, ui=lambda folder: ui).run()
    assert sorted(recorded) == sorted(sv.metric_ids()) and set(recorded.values()) == {1}
    assert "recorded twice" not in capsys.readouterr().err
    m = ev.metrics.items
    tests = ("server_sh.ui.tests_failed", "server_sh.ui.tests_run")
    assert [(m[k].value, m[k].error) for k in tests] == [(1.0, None), (17.0, None)]
    assert m["server_sh.ui.tests_run"].detail == ui.detail
    why = ev.details["server_sh"]["start_error"]
    assert why.startswith("server_sh failed (no 'listening on' line on stderr)")
    assert {k for k in sv.metric_ids() if m[k].error == why} == set(sv.metric_ids()) - set(tests)
    assert "errors" not in ev.details  # a command that failed, handled: no evaluator error


MAPPER = """n=$(cat "$COUNTER" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$COUNTER"
[ -n "$FAIL" ] && { echo "mapper.sh: error: boom" >&2; exit 1; }
for a in "$@"; do case "$a" in -m=*) mkdir -p "${a#-m=}"; echo "{}" > "${a#-m=}/map.json";; esac
done
if [ -n "$VARY" ]; then printf '{"openlabel": {"run": %d}}' $n; else printf '{"openlabel": {}}'; fi
"""
def update_op() -> sv.Op:
    return next(o for o in served() if o.label == "mapper.sh update")


def mapping_case(ran: sv.Ran | None = None) -> Case:
    return Case("mapper-update", "mapper.sh update", "default",
                {"inputs": ["inputs/a.jpg"], "map": "parity-1"}, "map", "json", "ainex", ran)


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
    row = se.run_case(update_op(), mapping_case(), 1)
    assert row["status"] == status and row.get("why") == why
    assert row["case"] == "mapper.sh update [default] on ainex" and row["stages"] == []
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
    # an answered request's body keeps the contracts of every command; a failed one has none
    requests = ev.contracts.checks.get(("stdout", "server_sh"), {})
    assert ("parity_01_request" in requests) is (reply.status == 200)


@pytest.mark.parametrize(("env", "same", "reply", "status", "why"), [
    ({}, True, Reply(200, b'{"openlabel": {}}', 0.5), "identical", None),
    ({}, True, Reply(200, b'{"openlabel": {"x": 1}}', 0.5), "mismatch",
     "result: 17 vs 23 bytes, first difference at byte 15 (no control run: the suite's run is "
     "the command's result)"),
    ({"FAIL": "1"}, True, None, "mismatch",
     "mapper_street2 failed (exit 1): mapper.sh: error: boom"),
    ({}, False, None, "mismatch", "the suite's run mapper_street2 is not the case's command line"),
])
def test_a_case_the_suite_already_ran_is_compared_with_that_run(
        tmp_path: Path, env: dict[str, str], same: bool, reply: Reply | None, status: str,
        why: str) -> None:
    """street2.mp4: its section's run (through the proxy, into the workspace) is the command's
    result; it is not run again, nor a control; its map is set aside for the request to make it
    again at the same path; without a result to compare with, no request is sent."""
    ev = evaluation(tmp_path, mapper=MAPPER)
    se = service(ev, FakeApi(post=reply), {"COUNTER": str(tmp_path / "count")})
    se.workspace()
    op = update_op()
    argv = se.argv(op, mapping_case())
    rec = ev.run("mapper_street2", "mapper_street2", "mapper.sh", *argv, env=env)
    ran = sv.Ran(tuple(argv if same else argv[:-1]), rec)
    row = se.run_case(op, mapping_case(ran), 1)
    assert row["status"] == status and (row.get("why") or "").startswith(why or "")
    assert [r.tag for r in ev.runner.records] == ["mapper_street2"]  # nothing ran again
    assert se.overheads == {}  # no control run to compare the request's time with
    made = se.root / "shell_maps" / "parity_01_shell"
    assert made.is_dir() is (not env) and not (se.ws / "maps" / "parity-1").exists()


def test_a_response_body_keeps_the_contracts_of_every_command(tmp_path: Path) -> None:
    """spec §5 "Contracts, for every command": the response is the command's result, so it is
    one payload of its format and, as a scene, valid OpenLABEL in the colours of its ids."""
    from oh_my_slam.segmentation.colors import color_for_id, color_hex_for_id

    ev = evaluation(tmp_path)
    se = service(ev)
    good = list(color_for_id(3))
    wrong = [1, 2, 3] if good != [1, 2, 3] else [3, 2, 1]
    scene = {"openlabel": {"metadata": {"schema_version": "1.0.0"}, "objects": {
        "3": {"name": "cup 3", "type": "cup", "object_data": {
            "vec": [{"name": "color", "val": wrong}],
            "text": [{"name": "color_hex", "val": color_hex_for_id(3)}]}}}}}
    se.response_contracts(mapping_case(), "parity_01_request", json.dumps(scene).encode())
    se.response_contracts(mapping_case(), "parity_02_request", b"model loaded\n{}")
    png = Case("reconstruct", "reconstruct.sh", "format=depth", {}, None, "png")
    se.response_contracts(png, "parity_03_request", b"")
    checks = ev.contracts.checks
    stdout = checks[("stdout", "server_sh")]
    assert stdout["parity_01_request"] == []
    assert stdout["parity_02_request"][0].startswith("does not start with a JSON object")
    assert stdout["parity_03_request"] == ["expected one PNG payload, got nothing"]
    openlabel = checks[("openlabel", "server_sh")]
    assert set(openlabel) == {"parity_01_request", "parity_02_request"}
    assert openlabel["parity_02_request"][0].startswith("not JSON")
    (colour,) = checks[("colour", "server_sh")]["parity_01_request"]
    assert colour == f"id 3: color {tuple(wrong)} != {good}"
    assert set(checks[("colour", "server_sh")]) == {"parity_01_request"}


def test_parity_needs_the_services_description(tmp_path: Path) -> None:
    ev = evaluation(tmp_path)
    service(ev, FakeApi({"api/openapi.json": Reply(503, b"", 0.01)})).parity([])
    ids = [f"server_sh.parity.{k}" for k in sv.PARITY_METRICS]
    assert [ev.metrics.items[k].error for k in ids] == ["/api/openapi.json answered HTTP 503"] * 3
    ev = evaluation(tmp_path / "2")
    stray = json.dumps({"paths": {"/api/ops/stray": {"post": {}}}}).encode()
    se = service(ev, FakeApi({"api/openapi.json": Reply(200, stray, 0.01)}))
    se.parity([])
    m = ev.metrics.items
    assert m["server_sh.parity.mismatched"].value == 1
    assert m["server_sh.parity.mismatched"].detail["mismatched"] == [
        {"case": "/api/ops/stray", "why": "not an operation of commands.spec"}]
    covered = m["server_sh.parity.operations_covered_fraction"]
    assert covered.value is None and covered.error == "the service describes no operation"


def test_the_latency_of_reads_includes_the_maps_page(tmp_path: Path) -> None:
    ev = evaluation(tmp_path)
    api = FakeApi({"api/maps/reference": Reply(404, b"", 0.004)})
    service(ev, api).latency("reference")
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
    service(ev, FakeApi()).browser_checks("reference")
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
    service(ev, FakeApi()).browser_checks(None)
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
