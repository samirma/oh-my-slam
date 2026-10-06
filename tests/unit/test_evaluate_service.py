"""The ``server.sh`` evaluation (http_server.md "Evaluation") offline, with fakes: the parity cases
enumerated from the service's description of the commands (a new option or mode is covered with no
evaluator change), the command lines, the record-or-replay inference proxy, the browser-test
report, and the whole section against a fake ``server.sh`` whose operations answer within their
request (performance, parity and UI metrics)."""

from __future__ import annotations

import dataclasses
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import httpx
import pytest

from oh_my_slam.commands import spec
from oh_my_slam.core.images import png_bytes
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.tools.evaluate import proxy as px
from oh_my_slam.tools.evaluate import service as sv
from oh_my_slam.tools.evaluate.metrics import load_targets
from oh_my_slam.tools.evaluate.runner import Runner
from oh_my_slam.tools.evaluate.suite import EXAMPLES, Evaluation
from oh_my_slam.tools.evaluate.viewer import BrowserProbe
from tests.unit.test_evaluate_runner import fake_repo, script
from tests.unit.test_view_cli import minimal_map

INPUTS = sv.Inputs(image="inputs/a.jpg", images=("inputs/b.jpg",),
                   sequence=("inputs/c.jpg", "inputs/d.jpg"), map="reference")


def served() -> dict[str, Any]:
    """The operations as the service's /api/openapi.json describes them."""
    from oh_my_slam.web import openapi, operations

    got, problems = sv.operations_of(openapi.document(operations.operations()))
    assert problems == []
    return got


def cases_by_op(describe: dict[str, Any]) -> dict[str, list[sv.Case]]:
    cases, uncovered = sv.parity_cases(describe, INPUTS)
    assert uncovered == {}
    out: dict[str, list[sv.Case]] = {}
    for c in cases:
        out.setdefault(c.op, []).append(c)
    return out


# -- the cases, from the definitions ---------------------------------------------------------------


def test_every_operation_of_the_service_gets_its_cases() -> None:
    by_op = cases_by_op(served())
    assert set(by_op) == {"reconstruct", "mapper-update", "mapper-locate",
                          "segment-image"}  # view.sh stays a command only
    variants = {op: [c.variant for c in cs] for op, cs in by_op.items()}
    assert variants["reconstruct"] == ["default", "format=depth", "format=ply",
                                       "attrs=color=segment"]
    assert variants["mapper-update"] == ["default", "format=ply", "attrs=color=segment",
                                         "mode=single"]
    assert variants["mapper-locate"] == ["default", "format=ply", "attrs=color=segment",
                                         "mode=full"]
    assert variants["segment-image"] == ["default", "format=png"]
    # inputs by kind; -p applies with -f ply only, so its case asks for -f ply
    attrs = by_op["reconstruct"][3]
    assert attrs.params == {"image": "inputs/a.jpg", "attrs": "color=segment", "format": "ply"}
    assert attrs.result_format == "ply" and by_op["reconstruct"][0].result_format == "json"
    # the images are PNG results: the depth image and the segmented image
    assert by_op["reconstruct"][1].result_format == "png"
    assert [c.result_format for c in by_op["segment-image"]] == ["json", "png"]
    assert by_op["mapper-locate"][0].params == {"inputs": ["inputs/b.jpg"], "map": "reference"}
    # a mapping case creates a map of its own, whose name it carries
    update = by_op["mapper-update"]
    assert {c.params["map"] for c in update} == {f"parity-{k}" for k in range(1, 5)}
    assert all(c.writes_map == "map" for c in update)
    assert by_op["mapper-locate"][0].writes_map is None
    # no case names where the command writes (-o, -d): the response is the result
    assert not any({"output", "artifacts"} & set(c.params) for cs in by_op.values() for c in cs)


def test_a_new_option_or_mode_is_covered_without_evaluator_changes(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry is the evaluator's only list: a choice added to an option, and a new mode,
    become cases."""
    rec = spec.RECONSTRUCT.commands[0]
    fmt = rec.option("format")
    options = tuple(dataclasses.replace(fmt, choices=("json", "ply", "glb"))
                    if o is fmt else o for o in rec.options)
    extra = spec.Option("-q", "quality", spec.Kind.ENUM, "quality", default="fast",
                        choices=("fast", "best"))
    rec2 = dataclasses.replace(rec, options=(*options, extra))
    monkeypatch.setattr(spec, "PROGRAMS", (spec.Program(spec.RECONSTRUCT.prog,
                                                        spec.RECONSTRUCT.description, (rec2,)),
                                           *spec.PROGRAMS[1:]))
    # through the service's OpenAPI document, as the evaluator reads it
    variants = [c.variant for c in cases_by_op(served())["reconstruct"]]
    assert variants == ["default", "format=ply", "format=glb", "attrs=color=segment",
                        "quality=best"]  # the choices given above replace json, depth, ply
    # an operation whose input kind has no reference input is reported, not dropped silently
    cases, uncovered = sv.parity_cases(served(), sv.Inputs(image="inputs/a.jpg"))
    assert {c.op for c in cases} == {"reconstruct", "segment-image"}
    assert "mapper.sh update" in uncovered and "inputs" in uncovered["mapper.sh update"]


def test_the_operations_are_read_from_the_openapi_document() -> None:
    """The service's own /api/openapi.json carries each operation's registry entry, as the API
    offers it, under x-oms: the evaluator's operations are exactly the service's — every mode of
    the programs it offers, without the options that only choose where the command writes."""
    from oh_my_slam.web import openapi, operations

    ops = operations.operations()
    by_label = {op.label: op for op in ops.values()}
    expected = [by_label[d["id"]].entry(d) for d in spec.describe()["operations"]
                if d["id"] in by_label]
    got, problems = sv.operations_of(openapi.document(ops))
    assert problems == [] and got == {"operations": expected}
    assert {o["prog"] for o in got["operations"]} == {p.prog for p in spec.PROGRAMS if p.service}
    assert not [p for o in got["operations"] for p in o["parameters"]
                if p["kind"] in ("file_out", "folder_out")]
    doc = openapi.document(ops)
    doc["paths"]["/api/ops/stray"] = {"post": {}}  # an operation path without its entry
    assert sv.operations_of(doc)[1] == ["/api/ops/stray"]


def test_the_command_line_of_a_case_is_the_services(tmp_path: Path) -> None:
    """The shell side runs the command line the service runs for the same request: the same
    options, the same absolute workspace paths."""
    from oh_my_slam.web import operations
    from oh_my_slam.web.workspace import Workspace

    ws = Workspace(tmp_path / "ws")
    ws.create()
    for rel in ("inputs/a.jpg", "inputs/b.jpg", "inputs/c.jpg", "inputs/d.jpg"):
        (ws.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws.root / rel).write_bytes(b"x")
    minimal_map(ws.maps / "reference")
    service_ops = operations.operations()
    ops = {o["id"]: o for o in served()["operations"]}
    ev = sv.ServiceEvaluation.__new__(sv.ServiceEvaluation)
    ev.ws = ws.root
    for op in ops.values():
        for case in cases_by_op({"operations": [op]})[sv.operation_id(op)]:
            service_op = service_ops[case.op]
            prep = operations.prepare(service_op, case.params, ws)
            assert prep.problems == [], (case, prep.problems)
            assert sv.shell_argv(op, ev.absolute(op, case)) == prep.argv, op["id"]
    assert sv.operation_id(ops["mapper.sh locate"]) == "mapper-locate"


def test_the_accessibility_check_visits_absolute_page_urls() -> None:
    base = "http://127.0.0.1:5000/"
    assert sv.app_urls(base, "reference") == [
        "http://127.0.0.1:5000/#/image", "http://127.0.0.1:5000/#/maps",
        "http://127.0.0.1:5000/#/maps/new", "http://127.0.0.1:5000/#/maps/reference",
        "http://127.0.0.1:5000/#/maps/reference/update"]
    assert sv.app_urls(base, None) == [
        "http://127.0.0.1:5000/#/image", "http://127.0.0.1:5000/#/maps",
        "http://127.0.0.1:5000/#/maps/new"]


def test_differences_name_what_differs() -> None:
    a = sv.Outcome(True, result=b"abcdef")
    assert sv.differences(a, a) == []
    b = sv.Outcome(True, result=b"abcXef")
    assert sv.differences(a, b) == ["result: 6 vs 6 bytes, first difference at byte 3"]
    assert sv.timing_stages("connect;dur=12.3, inference;dur=830.1, total;dur=842.4") == [
        "connect", "inference", "total"]
    assert sv.timing_stages(None) == []


def test_the_browser_tests_report_is_read(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text("""<testsuites><testsuite>
<testcase classname="t" name="a"/><testcase classname="t" name="b"><failure/></testcase>
<testcase classname="t" name="c"><skipped/></testcase><testcase classname="t" name="d"><error/>
</testcase></testsuite></testsuites>""")
    res = sv.parse_junit(junit, ["x.py"])
    assert (res.run, res.failed) == (3, 2) and res.detail["failed"] == ["t::b", "t::d"]
    assert sv.parse_junit(tmp_path / "none.xml", []).error is not None
    assert "no web application" in (sv.run_ui_tests(tmp_path, repo=tmp_path).error or "")


# -- the inference proxy ---------------------------------------------------------------------------


class Upstream:
    """A fake inference server on a Unix socket: every POST answers a fresh number and writes a
    file into the request's out_dir."""

    def __init__(self, folder: Path) -> None:
        self.calls = 0
        self.socket = folder / "up.sock"
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def address_string(self) -> str:
                return "up"

            def log_message(self, *a: Any) -> None:
                pass

            def do_GET(self) -> None:
                self.reply(200, {"status": "ready"})

            def do_POST(self) -> None:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                up.calls += 1
                out = Path(req["out_dir"]) / f"depth{up.calls}.npy"
                out.write_bytes(f"depth {up.calls}".encode())
                self.reply(200, {"n": up.calls, "depth_path": str(out)})

            def reply(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = px._Server(str(self.socket), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_the_proxy_records_once_and_replays_identical_requests() -> None:
    root = Path(tempfile.mkdtemp(prefix="oms-t-", dir="/tmp"))
    up = Upstream(root)
    try:
        with px.InferenceProxy(root / "rt", up.socket) as proxy:
            client = httpx.Client(transport=httpx.HTTPTransport(uds=str(proxy.socket)),
                                  base_url="http://x")
            img_a, img_b = root / "a.jpg", root / "b.jpg"
            img_a.write_bytes(b"A")
            img_b.write_bytes(b"B")

            def ask(image: Path, out: str) -> dict[str, Any]:
                (root / out).mkdir(exist_ok=True)
                r = client.post("/v1/geometry", json={"image_path": str(image),
                                                       "out_dir": str(root / out), "side": 512})
                assert r.status_code == 200
                body: dict[str, Any] = r.json()
                return body

            first = ask(img_a, "o1")
            again = ask(img_a, "o2")  # same image (another path is fine), same fields: replayed
            copy = root / "a_copy.jpg"
            copy.write_bytes(b"A")
            third = ask(copy, "o3")
            other = ask(img_b, "o4")  # another image: forwarded
            assert first["n"] == again["n"] == third["n"] == 1 and other["n"] == 2
            assert up.calls == 2
            assert Path(again["depth_path"]).parent == root / "o2"  # restored into its out_dir
            assert Path(again["depth_path"]).read_bytes() == b"depth 1"
            assert client.get("/health").json() == {"status": "ready"}  # forwarded
            assert proxy.stats() == {"recorded": 2, "replayed": 2, "forwarded": 3}
            assert proxy.env() == {"OH_MY_SLAM_RUNTIME_DIR": str(root / "rt")}
            client.close()
    finally:
        up.close()


def test_request_keys_follow_contents_not_paths(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    key = px.request_key
    assert key("/r", {"image_path": str(a), "out_dir": "x"}) == \
        key("/r", {"image_path": str(b), "out_dir": "y"})
    assert key("/r", {"image_path": str(a), "k": 1}) != key("/r", {"image_path": str(a), "k": 2})
    assert key("/r", {"image_paths": [str(a)]}) != key("/s", {"image_paths": [str(a)]})
    b.write_bytes(b"other")
    assert key("/r", {"image_path": str(a)}) != key("/r", {"image_path": str(b)})


# -- the section, against a fake server.sh ------------------------------------------------------------

FAKE_SERVER = r'''import json, os, signal, sys, time, http.server, threading
from pathlib import Path
signal.signal(signal.SIGINT, lambda *a: os._exit(0))
args = sys.argv[1:]
if "--status" in args:
    print(json.dumps({"status": "ok"}))
    sys.exit(0)
DESCRIBE = json.loads(Path(os.environ["FAKE_DESCRIBE"]).read_text())
PAYLOADS = Path(os.environ["FAKE_PAYLOADS"])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, status, body, ctype="application/json", headers=()):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status); self.send_header("Content-Type", ctype)
        for k, v in headers: self.send_header(k, v)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        p = self.path
        if p == "/api/openapi.json": return self.reply(200, {"paths": {
            "/api/ops/reconstruct": {"post": {"x-oms": DESCRIBE["operations"][0]}},
            "/api/ops/reconstruct/validate": {"post": {}}, "/api/health": {"get": {}}}})
        return self.reply(200, b"<html></html>" if p == "/" else {}, "text/html" if p == "/" else "application/json")
    def do_POST(self):
        assert self.headers["Content-Type"] == "application/json"
        params = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(0.4)  # the command runs within the request
        fmt = params.get("format", "json")
        data = (PAYLOADS / f"{fmt}.out").read_bytes()
        if fmt == "ply" and os.environ.get("FAKE_DIFFER"): data += b"x"
        self.reply(200, data, "application/json" if fmt == "json" else "application/octet-stream",
                   [("Server-Timing", "inference;dur=300.0, export;dur=1.0, total;dur=301.0")])
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
print(f"server.sh: listening on http://0.0.0.0:{srv.server_address[1]}/", file=sys.stderr, flush=True)
srv.serve_forever()
'''


def fake_service(tmp_path: Path, differ: bool) -> Evaluation:
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    (payloads / "json.out").write_bytes(b'{"openlabel": {"metadata": {"schema_version": "1.0.0"}}}')
    (payloads / "ply.out").write_bytes(ply_bytes(PointCloud(
        __import__("numpy").zeros((2, 3), "float32"))))
    (payloads / "depth.out").write_bytes(png_bytes(__import__("numpy").zeros((2, 3), "uint16")))
    rec = next(o for o in served()["operations"] if o["id"] == "reconstruct.sh")
    rec = {**rec, "parameters": [p for p in rec["parameters"] if p["name"] != "attrs"]}
    (tmp_path / "describe.json").write_text(json.dumps({"operations": [rec]}))
    repo = fake_repo(tmp_path, reconstruct=f'''case "$*" in *-f=ply*) cat "{payloads}/ply.out";;
*-f=depth*) cat "{payloads}/depth.out";; *) cat "{payloads}/json.out";; esac''')
    script(repo, "server.sh", FAKE_SERVER, shebang=f"#!{__import__('sys').executable}")
    out = tmp_path / "out"
    runner = Runner(out, repo)
    runner.env.update(FAKE_DESCRIBE=str(tmp_path / "describe.json"), FAKE_PAYLOADS=str(payloads),
                      **({"FAKE_DIFFER": "1"} if differ else {}))
    return Evaluation(out, runner, BrowserProbe(None), examples=EXAMPLES)


@pytest.mark.parametrize("differ", [False, True])
def test_the_section_measures_compares_and_judges(tmp_path: Path, differ: bool) -> None:
    ev = fake_service(tmp_path, differ)
    sv.ServiceEvaluation(ev, {"image": EXAMPLES / "restaurant.jpg"}, {},
                         ui=lambda folder: sv.UiOutcome(23, 0)).run()
    m = ev.metrics.items
    ev.metrics.judge(load_targets(EXAMPLES / "targets.json"), None)
    assert set(sv.metric_ids()) <= set(m)
    assert m["server_sh.start_s"].value is not None and m["server_sh.start_s"].value < 30
    assert m["server_sh.resident_mb"].value is not None and m["server_sh.resident_mb"].value > 0
    assert m["server_sh.read_latency_median_ms"].value is not None
    assert m["server_sh.read_latency_p95_ms"].passed is True
    assert m["server_sh.request_overhead_median_s"].value is not None
    assert set(m["server_sh.request_overhead_median_s"].detail) == {"reconstruct"}
    # read while the requests ran (the fake answers after 0.4 s)
    assert m["server_sh.read_latency_busy_p95_ms"].value is not None
    # no browser in this test: the render time and the axe check fail with the reason
    assert "no browser" in (m["server_sh.app_render_s"].error or "")
    assert m["server_sh.ui.tests_failed"].passed and m["server_sh.ui.tests_run"].passed
    assert m["server_sh.parity.operations_covered_fraction"].value == 1.0
    cases = ev.details["server_sh"]["parity"]["cases"]
    assert [c["case"] for c in cases] == ["reconstruct.sh [default]",
                                          "reconstruct.sh [format=depth]",
                                          "reconstruct.sh [format=ply]"]
    assert cases[0]["stages"] == ["inference", "export", "total"]  # from Server-Timing
    mism = m["server_sh.parity.mismatched"]
    if differ:
        assert mism.value == 1 and mism.passed is False
        assert "first difference at byte" in mism.detail["mismatched"][0]["why"]
    else:
        assert mism.value == 0 and mism.passed and m["server_sh.parity.unverifiable"].value == 0
    tags = [r.tag for r in ev.runner.records]
    assert tags[0] == "server_sh" or "server_sh" in tags
    assert {"parity_01_shell", "parity_01_shell_again", "parity_02_shell"} <= set(tags)
    # the depth image's shell runs wrote one PNG to stdout, as the contract wants
    assert ev.contracts.checks[("stdout", "reconstruct")]["parity_02_shell"] == []
    # server.sh kept stdout empty while serving, and --status gave one JSON document
    assert ev.contracts.checks[("stdout", "server_sh")] == {"server_sh": [], "server_sh_status": []}
