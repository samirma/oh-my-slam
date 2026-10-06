"""``server.sh`` on the reference inputs (http_server.md "Evaluation"): performance, parity with the
commands, and the web application's UI.

The service runs over a scratch workspace (``<out>/server_sh/ws``) holding copies of the reference
inputs and of the reference map, with the inference proxy (``proxy``) as its inference server, and
so do the shell runs it is compared with. There are no jobs: an operation runs within its own HTTP
request, and the response is its result.

**Performance** (``server_sh.*``): start-up time (process start → the ``listening on`` line),
resident memory once listening (the process tree, idle), time from opening the web application
to its having rendered (``body[data-ready=true]``; a page without scripts: its ``load``), latency
of read-only requests (median and p95 idle, after the parity requests; p95 of the reads made
while an operation's request runs), and the overhead of an operation's request over the same
command run from the shell (median over the operations of the request's wall time — sent to
answered — minus the wall time of the same command with the same, replayed, inference).

**Parity** (``server_sh.parity.*``): the operations come from the service's own description of
the commands (``/api/openapi.json``: each operation's ``x-oms`` entry, ``commands.spec.describe()``
as the API offers it) — nothing here names a command or an option. Each operation gets a default
case (its required parameters on the reference inputs) and one case per non-default value of each
choice and per point-cloud attribute (a non-default choice of its first enumerated attribute),
with the options a case needs to apply set as the definitions say (``applies``). Each case runs
the command from the shell twice (the first run records the inference, the second replays it) and
then as a request, and compares the response body with the command's stdout byte for byte. A
difference is a mismatch, unless the command's own two runs differ too (``unverifiable``: the
command is not byte-reproducible). A case that maps writes the map at the same workspace path
each time, so the map's path in the result is the same; inputs are the same absolute workspace
paths on both sides.

**UI** (``server_sh.ui.*``): the web application's browser tests (``tests/browser/test_webapp*.py``,
``-m browser``: the main flows, the inference server down, and an axe-core check of each page),
and the vendored axe-core run on the live service's pages over the reference data.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oh_my_slam.tools.evaluate.memory import tree_rss_mb
from oh_my_slam.tools.evaluate.runner import REPO, Live, RunRecord, RunSpec

if TYPE_CHECKING:
    from oh_my_slam.tools.evaluate.suite import Evaluation

ENTRY = "server.sh"
GROUP = "server_sh"
PREFIX = "server_sh"
LISTEN = re.compile(r"^server\.sh: listening on http://[^:/\s]+:(\d+)/$", re.MULTILINE)
POLL_S = 0.05
BUSY_POLL_S = 0.1  # reads made while an operation's request runs
REQUEST_TIMEOUT_S = 1800.0  # one operation's request (a mapping one takes minutes)
SERVICE_TIMEOUT_S = 7200.0  # the service's whole run: every parity case, latency, browser
START_TIMEOUT_S = 120.0
LATENCY_REPS = 10
READS = ("", "api/health", "api/openapi.json", "api/maps")  # read-only requests (and a map's)
APP_PAGES = ("#/image", "#/maps", "#/maps/new", "#/maps/{map}", "#/maps/{map}/update")
READY_JS = ("() => document.readyState === 'complete' && document.body !== null && "
            "(document.body.dataset.ready === 'true' || "
            "document.querySelector('script') === null)")
ROUTED_JS = ("(where) => !window.__app || !window.__app.lastRoute || "
             "window.__app.lastRoute.where === where")  # the app's own route timing
RENDER_TIMEOUT_MS = 60_000
AXE = REPO / "tests" / "browser" / "vendor" / "axe-core" / "axe.min.js"
UI_TESTS = "tests/browser/test_webapp*.py"
UI_TIMEOUT_S = 3600.0
PERF_METRICS = ("start_s", "resident_mb", "app_render_s", "read_latency_median_ms",
                "read_latency_p95_ms", "read_latency_busy_p95_ms", "request_overhead_median_s")
PARITY_METRICS = ("mismatched", "unverifiable", "operations_covered_fraction")
UI_METRICS = ("tests_failed", "tests_run", "a11y_violations")


def app_urls(base: str, map_name: str | None) -> list[str]:
    """The web application's pages checked by axe-core, as absolute URLs (a map's pages only when
    there is a map to show)."""
    pages = [p for p in APP_PAGES if "{map}" not in p or map_name]
    return [base.rstrip("/") + "/" + p.format(map=map_name or "") for p in pages]


def metric_ids() -> list[str]:
    return [*(f"{PREFIX}.{k}" for k in PERF_METRICS),
            *(f"{PREFIX}.parity.{k}" for k in PARITY_METRICS),
            *(f"{PREFIX}.ui.{k}" for k in UI_METRICS)]


# ------------------------------------------------------------------------------------------------
# the cases, from the commands' definitions


@dataclass(frozen=True)
class Inputs:
    """Reference inputs inside the workspace (paths relative to it; ``map``: a map's name)."""

    image: str | None = None  # one image (Kind "image")
    images: tuple[str, ...] = ()  # images to locate (Kind "images")
    sequence: tuple[str, ...] = ()  # images to map (Kind "images_or_video")
    map: str | None = None  # an existing map (read-only modes)

    def of(self, kind: str) -> Any:
        return {"image": self.image, "images": list(self.images) or None,
                "images_or_video": list(self.sequence) or None, "map": self.map}.get(kind)


@dataclass
class Case:
    op: str  # the API operation id (``segment-image``)
    label: str  # the command as typed (``segment.sh -i``)
    variant: str
    params: dict[str, Any]  # API parameters: workspace paths and map names
    writes_map: str | None = None  # the parameter naming the map the case creates
    result_format: str = "json"  # the format of the result (the response body / stdout)


def operations_of(openapi: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The operations of the service's ``/api/openapi.json``: each ``POST /api/ops/<id>`` carries
    its ``commands.spec.describe()`` entry as the API offers it under ``x-oms``. Returns
    ``{"operations": [...]}`` (the shape of ``describe()``) and the operation paths whose entry is
    missing or whose id does not match their path."""
    ops, problems = [], []
    for path, item in (openapi.get("paths") or {}).items():  # the registry's order
        if not path.startswith("/api/ops/") or path.endswith("/validate"):
            continue
        entry = ((item or {}).get("post") or {}).get("x-oms")
        if not isinstance(entry, dict) or "parameters" not in entry \
                or f"/api/ops/{operation_id(entry)}" != path:
            problems.append(path)
            continue
        ops.append(entry)
    return {"operations": ops}, problems


def operation_id(op: dict[str, Any]) -> str:
    """The service's URL id of an operation (``web.operations.Operation.id``)."""
    parts = [str(op["prog"]).removesuffix(".sh"), op.get("command"), op.get("mode")]
    return "-".join(p for p in parts if p)


def _api_params(op: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {p["name"]: p for p in op["parameters"]}


def _attrs_value(p: dict[str, Any]) -> str | None:
    """A non-default choice of the first enumerated attribute (``color=segment``)."""
    for a in p.get("attributes") or []:
        schema = a.get("schema") or {}
        if schema.get("type") == "enum":
            other = [c for c in schema.get("choices") or [] if c != a.get("default")]
            if other:
                return f"{a['key']}={other[0]}"
    return None


def _holds(when: dict[str, Any], given: dict[str, Any], params: dict[str, dict[str, Any]]) -> bool:
    opt = when["option"]
    if "in" in when:
        value = given.get(opt, (params.get(opt) or {}).get("default"))
        return value in when["in"]
    if when.get("is") == "given":
        return given.get(opt) is not None
    return False  # "video": the parity inputs are images


def _satisfy(applies: list[dict[str, Any]], given: dict[str, Any],
             params: dict[str, dict[str, Any]]) -> bool:
    """Set the first condition of ``applies`` that a case can meet; False if none."""
    for w in applies:
        opt = params.get(w["option"])
        if opt is None:
            continue
        if w.get("in"):
            given[w["option"]] = w["in"][0]
            return True
    return False


def _variants(op: dict[str, Any]) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    params = _api_params(op)
    out: list[tuple[str, dict[str, Any]]] = [("default", {})]
    for p in params.values():
        if p["kind"] == "enum":
            out += [(f"{p['name']}={c}", {p["name"]: c}) for c in p.get("choices") or []
                    if c != p.get("default")]
        elif p["kind"] == "attrs" and (v := _attrs_value(p)) is not None:
            out.append((f"{p['name']}={v}", {p["name"]: v}))
    done, skipped, seen = [], [], set()
    for name, given in out:
        for pname in list(given):
            applies = params[pname].get("applies") or []
            if applies and not any(_holds(w, given, params) for w in applies) \
                    and not _satisfy(applies, given, params):
                skipped.append(f"{name}: no reference input meets {applies}")
                break
        else:
            key = json.dumps(given, sort_keys=True)
            if key not in seen:
                seen.add(key)
                done.append((name, given))
    return done, skipped


def _result_format(op: dict[str, Any], given: dict[str, Any]) -> str:
    """The format of the operation's result for these parameters: its stdout output whose
    condition holds (the first one when none says)."""
    params = _api_params(op)
    results = [o for o in op.get("outputs") or [] if o.get("via") == "stdout"]
    for o in results:
        if not o.get("when") or any(_holds(w, given, params) for w in o["when"]):
            return str(o["format"])
    return str(results[0]["format"]) if results else "json"


def parity_cases(describe: dict[str, Any], inputs: Inputs
                 ) -> tuple[list[Case], dict[str, str]]:
    """Every case of every operation of ``describe`` (the service's operations, as
    :func:`operations_of` reads them), and the operations that have none, with the reason."""
    cases: list[Case] = []
    uncovered: dict[str, str] = {}
    fresh = 0
    for op in describe.get("operations", []):
        oid = operation_id(op)
        params = _api_params(op)
        written = {o.get("via") for o in op.get("outputs") or []}
        map_out = next((n for n, p in params.items()
                        if p["kind"] == "map" and p["flag"] in written), None)
        required = [n for n, p in params.items() if p.get("required")]
        missing = [n for n in required if n != map_out and inputs.of(params[n]["kind"]) is None]
        if missing:
            uncovered[op["id"]] = (f"no reference input for {', '.join(missing)} "
                                   f"({', '.join(params[n]['kind'] for n in missing)})")
            continue
        variants, _ = _variants(op)
        for name, given in variants:
            case_params = {n: inputs.of(params[n]["kind"]) for n in required}
            if map_out is not None:
                fresh += 1
                case_params[map_out] = f"parity-{fresh}"
            case_params.update(given)
            cases.append(Case(oid, op["id"], name, case_params, map_out,
                              _result_format(op, case_params)))
    return cases, uncovered


def shell_argv(op: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """The command line of an operation for parameters (paths already absolute), as the service
    builds it (``spec.argv_of``)."""
    argv = [op["command"]] if op.get("command") else []
    for p in op["parameters"]:
        flag, name = p["flag"], p["name"]
        v = params.get(name)
        if v is None:
            continue
        if p["kind"] == "flag":
            argv += [flag] if v else []
            continue
        values = list(v) if isinstance(v, list | tuple) else [v]
        if p.get("multiple"):
            argv += [flag, *(f"./{x}" if str(x).startswith("-") else str(x) for x in values)]
        else:
            argv += [f"{flag}={x}" for x in values]
    return argv


# ------------------------------------------------------------------------------------------------
# HTTP


@dataclass
class Reply:
    status: int
    body: bytes
    seconds: float
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.body)


class Api:
    """The service's API at ``base`` (``http://127.0.0.1:<port>/``), as a same-origin client:
    loopback ``Host``, no foreign ``Origin``, JSON bodies (the service's request guard)."""

    def __init__(self, base: str, timeout: float = 120.0) -> None:
        self.base, self.timeout = base.rstrip("/") + "/", timeout

    def request(self, method: str, path: str, body: Any = None,
                timeout: float | None = None) -> Reply:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path.lstrip("/"), data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return Reply(r.status, r.read(), time.perf_counter() - t0, dict(r.headers))
        except urllib.error.HTTPError as exc:
            return Reply(exc.code, exc.read(), time.perf_counter() - t0, dict(exc.headers))
        except (urllib.error.URLError, OSError) as exc:
            return Reply(0, str(exc).encode(), time.perf_counter() - t0)

    def get(self, path: str) -> Reply:
        return self.request("GET", path)


# ------------------------------------------------------------------------------------------------
# outcomes


@dataclass
class Outcome:
    """What one side of a case produced: the result bytes (the command's stdout, or the response
    body), and for a request the stages its ``Server-Timing`` header named."""

    ok: bool
    error: str | None = None
    result: bytes | None = None
    wall_s: float | None = None
    stages: list[str] = field(default_factory=list)


def differences(a: Outcome, b: Outcome) -> list[str]:
    if a.result == b.result:
        return []
    return [f"result: {len(a.result or b'')} vs {len(b.result or b'')} bytes"
            + ("" if a.result is None or b.result is None
               else f", first difference at byte {_first_diff(a.result, b.result)}")]


def timing_stages(header: str | None) -> list[str]:
    """The stage names of a ``Server-Timing`` header value (``name;dur=<ms>, …``)."""
    return [part.split(";")[0].strip() for part in (header or "").split(",") if part.strip()]


def _first_diff(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    diff = np.flatnonzero(np.frombuffer(a[:n], np.uint8) != np.frombuffer(b[:n], np.uint8))
    return int(diff[0]) if len(diff) else n


@dataclass
class UiOutcome:
    run: int | None  # browser tests that ran (passed or failed)
    failed: int | None
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def run_ui_tests(out: Path, repo: Path = REPO) -> UiOutcome:
    """The web application's browser tests (``UI_TESTS``) with pytest, ``-m browser``, against
    the stub inference server in an isolated runtime folder (never the real one)."""
    files = sorted(repo.glob(UI_TESTS))
    if not files:
        return UiOutcome(None, None, f"no web application browser tests ({UI_TESTS})")
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    env = {k: v for k, v in os.environ.items()
           if k not in ("OH_MY_SLAM_TEST_REAL_SERVER", "OH_MY_SLAM_RUNTIME_DIR")}
    argv = [sys.executable, "-m", "pytest", "-m", "browser", "-q", "-p", "no:cacheprovider",
            f"--junitxml={junit}", *(str(f.relative_to(repo)) for f in files)]
    with (out / "pytest.log").open("wb") as log:
        try:
            subprocess.run(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=UI_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            return UiOutcome(None, None, f"the browser tests timed out after {UI_TIMEOUT_S:.0f} s")
    return parse_junit(junit, [str(f.relative_to(repo)) for f in files])


def parse_junit(path: Path, files: list[str]) -> UiOutcome:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return UiOutcome(None, None, f"no test report: {exc}")
    cases = list(root.iter("testcase"))
    failed, skipped = [], 0
    for c in cases:
        if c.find("skipped") is not None:
            skipped += 1
        elif c.find("failure") is not None or c.find("error") is not None:
            failed.append(f"{c.get('classname')}::{c.get('name')}")
    run = len(cases) - skipped
    return UiOutcome(run, len(failed), None, {"files": files, "tests": len(cases),
                                              "skipped": skipped, "failed": failed[:20]})


# ------------------------------------------------------------------------------------------------
# the evaluation


class ServiceEvaluation:
    """``server.sh`` evaluated through an :class:`Evaluation` (its runner, contracts, metrics)."""

    def __init__(self, ev: Evaluation, inputs_from: dict[str, Any], env: dict[str, str],
                 ui: Callable[[Path], UiOutcome] = run_ui_tests) -> None:
        self.ev, self.env, self.ui = ev, env, ui
        self.root = ev.out / "server_sh"
        self.ws = self.root / "ws"
        self.inputs_from = inputs_from
        self.details: dict[str, Any] = {}
        self.busy_ms: list[float] = []
        self.overheads: dict[str, dict[str, float]] = {}
        self.api: Api | None = None

    # -- set-up ------------------------------------------------------------------------------------

    def workspace(self) -> Inputs:
        """Copy the reference inputs (and map) into the workspace; the cases' inputs."""
        src = self.inputs_from
        (self.ws / "inputs").mkdir(parents=True, exist_ok=True)
        (self.ws / "maps").mkdir(parents=True, exist_ok=True)
        self.ws = self.ws.resolve()

        def put(path: Path | None) -> str | None:
            if path is None or not Path(path).is_file():
                return None
            target = self.ws / "inputs" / Path(path).name
            shutil.copyfile(path, target)
            return str(target.relative_to(self.ws))

        image = put(src.get("image"))
        images = tuple(x for p in src.get("images", []) if (x := put(p)) is not None)
        sequence = tuple(x for p in src.get("sequence", []) if (x := put(p)) is not None)
        name = None
        ref = src.get("map")
        if ref is not None and Path(ref).is_dir():
            name = "reference"
            shutil.copytree(ref, self.ws / "maps" / name, symlinks=True)
        return Inputs(image, images, sequence, name)

    def spec(self, tag: str, *args: str | Path, stdout: str = "json",
             ok_exit: tuple[int, ...] = (0,), timeout_s: float = 3600.0) -> RunSpec:
        return RunSpec(tag, GROUP, ENTRY, tuple(str(a) for a in args), stdout, None, None,
                       ok_exit, timeout_s, tuple(self.env.items()))

    def start(self) -> Live | None:
        m = self.ev.metrics
        live = self.ev.runner.start(self.spec("server_sh", "--data", self.ws, "--no-browser",
                                              stdout="empty", ok_exit=(0, 130, -2),
                                              timeout_s=SERVICE_TIMEOUT_S))
        deadline = live.t0 + START_TIMEOUT_S
        port = None
        while live.proc is not None and live.poll() is None and time.perf_counter() < deadline:
            found = LISTEN.search(live.stderr_text())
            if found:
                port = int(found.group(1))
                break
            time.sleep(POLL_S)
        if port is None:
            live.stop()
            rec = live.finish(error="no 'listening on' line on stderr")
            self.ev.check_payloads(rec)
            self.details["start_error"] = rec.failure()
            return None
        started = time.perf_counter() - live.t0
        self.api = Api(f"http://127.0.0.1:{port}/")
        health = self.api.get("api/health")
        m.add(f"{PREFIX}.start_s", started, {"health_status": health.status})
        pid = live.proc.pid if live.proc is not None else None
        m.add(f"{PREFIX}.resident_mb", tree_rss_mb(pid) if pid else None, {"pid": pid},
              error="the service's process is gone")
        status = self.ev.run("server_sh_status", GROUP, ENTRY, "--data", self.ws, "--status",
                             env=self.env)
        if status.ok:
            self.details["health"] = json.loads(status.stdout_bytes())
        return live

    def stop(self, live: Live) -> RunRecord:
        live.stop()
        rec = live.finish()
        self.ev.check_payloads(rec)
        return rec

    # -- requests ----------------------------------------------------------------------------------

    def request_outcome(self, case: Case) -> Outcome:
        """Run ``case`` as a request (it answers when the command ended), reading the service
        while it runs: the read-only latency under load."""
        assert self.api is not None
        api = self.api
        done = threading.Event()
        busy: list[float] = []

        def reads() -> None:
            while not done.wait(BUSY_POLL_S):
                busy.append(api.get("api/health").seconds * 1000)

        watcher = threading.Thread(target=reads, daemon=True)
        watcher.start()
        try:
            r = api.request("POST", f"api/ops/{case.op}", case.params, timeout=REQUEST_TIMEOUT_S)
        finally:
            done.set()
            watcher.join()
        self.busy_ms += busy
        if r.status != 200:
            return Outcome(False, f"HTTP {r.status}: {r.body[:300]!r}", wall_s=r.seconds)
        timing = next((v for k, v in r.headers.items() if k.lower() == "server-timing"), None)
        return Outcome(True, result=r.body, wall_s=r.seconds, stages=timing_stages(timing))

    # -- the shell side ------------------------------------------------------------------------------

    def absolute(self, op: dict[str, Any], case: Case) -> dict[str, Any]:
        """The case's parameters with the absolute workspace paths the service gives the
        command."""
        params = _api_params(op)
        out: dict[str, Any] = {}
        for name, v in case.params.items():
            kind = params[name]["kind"]
            if kind == "map":
                out[name] = str(self.ws / "maps" / v)
            elif kind in ("image", "images", "images_or_video"):
                out[name] = [str(self.ws / x) for x in v] if isinstance(v, list) \
                    else str(self.ws / v)
            else:
                out[name] = v
        return out

    def shell_outcome(self, op: dict[str, Any], case: Case, tag: str) -> tuple[Outcome, RunRecord]:
        argv = shell_argv(op, self.absolute(op, case))
        rec = self.ev.run(tag, "parity", op["prog"], *argv, stdout=case.result_format,
                          env=self.env)
        if case.result_format == "json":
            self.ev.scene(rec)
        return Outcome(rec.ok, None if rec.ok else rec.failure(), rec.stdout_bytes(),
                       wall_s=rec.wall_s), rec

    def _set_map_aside(self, case: Case, tag: str) -> None:
        """Move the map a shell run created out of the workspace, so the next run (the same
        path) creates it again."""
        if case.writes_map is None:
            return
        made = self.ws / "maps" / case.params[case.writes_map]
        if made.exists():
            aside = self.root / "shell_maps" / tag
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(made), str(aside))

    def run_case(self, op: dict[str, Any], case: Case, n: int) -> dict[str, Any]:
        row: dict[str, Any] = {"case": f"{case.label} [{case.variant}]", "op": case.op,
                               "params": case.params}
        first, _ = self.shell_outcome(op, case, f"parity_{n:02d}_shell")
        self._set_map_aside(case, f"parity_{n:02d}_shell")
        # the control: the same command again, with the same (replayed) inference
        second, _ = self.shell_outcome(op, case, f"parity_{n:02d}_shell_again")
        self._set_map_aside(case, f"parity_{n:02d}_shell_again")
        req = self.request_outcome(case)
        row["stages"] = req.stages
        if not first.ok or not req.ok:
            row.update(status="mismatch", why=first.error if not first.ok else
                       f"the request failed: {req.error}")
            return row
        diffs = differences(first, req)
        if not diffs:
            row["status"] = "identical"
        elif second.ok and differences(first, second):
            row.update(status="unverifiable", why="the command's own two runs differ: "
                       + "; ".join(differences(first, second)[:3]), request=diffs[:5])
        else:
            row.update(status="mismatch", why="; ".join(diffs[:5]))
        if second.ok and req.wall_s is not None and second.wall_s is not None \
                and case.op not in self.overheads:
            self.overheads[case.op] = {"request_s": round(req.wall_s, 3),
                                       "shell_s": round(second.wall_s, 3),
                                       "overhead_s": round(req.wall_s - second.wall_s, 3)}
        return row

    def parity(self, inputs: Inputs) -> None:
        m = self.ev.metrics
        ids = [f"{PREFIX}.parity.{k}" for k in PARITY_METRICS]
        assert self.api is not None
        reply = self.api.get("api/openapi.json")
        if reply.status != 200:
            m.fail(ids, f"/api/openapi.json answered HTTP {reply.status}")
            return
        describe, problems = operations_of(reply.json())
        cases, uncovered = parity_cases(describe, inputs)
        ops = {o["id"]: o for o in describe.get("operations", [])}
        rows: list[dict[str, Any]] = [{"case": p, "status": "mismatch",
                                       "why": "not an operation of commands.spec"}
                                      for p in problems]
        for n, case in enumerate(cases, start=1):
            rows.append(self.run_case(ops[case.label], case, n))
        self.details["parity"] = {"cases": rows, "uncovered": uncovered}
        ran = {r["case"].split(" [")[0] for r in rows}
        bad = [r for r in rows if r["status"] == "mismatch"]
        unver = [r for r in rows if r["status"] == "unverifiable"]
        m.add(ids[0], len(bad), {"cases": len(rows), "mismatched": [
            {"case": r["case"], "why": r.get("why")} for r in bad]})
        m.add(ids[1], len(unver), {"unverifiable": [
            {"case": r["case"], "why": r.get("why")} for r in unver]})
        m.add(ids[2], len(ran) / len(ops) if ops else None,
              {"operations": sorted(ops), "uncovered": uncovered},
              error="the service describes no operation")

    # -- performance and UI ------------------------------------------------------------------------------

    def latency(self, inputs: Inputs) -> None:
        assert self.api is not None
        paths = list(READS)
        if inputs.map:
            paths.append(f"api/maps/{inputs.map}")
        samples: dict[str, list[float]] = {}
        failed = []
        for _ in range(LATENCY_REPS):
            for p in paths:
                r = self.api.get(p)
                if r.status != 200:
                    failed.append(f"/{p}: HTTP {r.status}")
                samples.setdefault("/" + p, []).append(r.seconds * 1000)
        every = [v for vs in samples.values() for v in vs]
        detail = {"per_path_median_ms": {k: round(float(np.median(v)), 2)
                                         for k, v in samples.items()},
                  "requests": len(every), "failed": sorted(set(failed))[:10]}
        m = self.ev.metrics
        m.add(f"{PREFIX}.read_latency_median_ms", float(np.median(every)), detail)
        m.add(f"{PREFIX}.read_latency_p95_ms", float(np.percentile(every, 95)), detail)
        m.add(f"{PREFIX}.read_latency_busy_p95_ms",
              float(np.percentile(self.busy_ms, 95)) if self.busy_ms else None,
              {"requests": len(self.busy_ms)},
              error="no read was made while an operation's request ran")
        over = self.overheads
        m.add(f"{PREFIX}.request_overhead_median_s",
              float(np.median([v["overhead_s"] for v in over.values()])) if over else None,
              over, error="no request could be compared with its command")

    def browser_checks(self, inputs: Inputs) -> None:
        """Time until the web application has rendered, and axe-core on its pages."""
        m = self.ev.metrics
        ids = (f"{PREFIX}.app_render_s", f"{PREFIX}.ui.a11y_violations")
        assert self.api is not None
        launch = self.ev.probe.launch
        if launch is None:
            m.fail(ids, "no browser available")
            return
        with ExitStack() as stack:
            try:
                browser = stack.enter_context(launch())
            except Exception as exc:
                m.fail(ids, f"browser: {exc}")
                return
            render, error = self.render(browser, self.api.base)
            m.add(ids[0], render, error=error)
            pages = app_urls(self.api.base, inputs.map)
            found, error = self.a11y(browser, pages)
            m.add(ids[1], None if found is None else sum(len(v) for v in found.values()),
                  {"pages": found}, error=error)

    @staticmethod
    def render(browser: Any, base: str) -> tuple[float | None, str | None]:
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        try:
            t0 = time.perf_counter()
            page.goto(base, timeout=RENDER_TIMEOUT_MS)
            page.wait_for_function(READY_JS, timeout=RENDER_TIMEOUT_MS)
            return time.perf_counter() - t0, None
        except Exception as exc:
            return None, f"the web application did not signal it had rendered: " \
                         f"{str(exc).splitlines()[0]}"
        finally:
            page.close()

    @staticmethod
    def a11y(browser: Any, urls: list[str]) -> tuple[dict[str, list[str]] | None, str | None]:
        if not AXE.is_file():
            return None, f"axe-core is not vendored at {AXE.relative_to(REPO)}"
        found: dict[str, list[str]] = {}
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            for url in urls:
                page.goto(url, timeout=RENDER_TIMEOUT_MS)
                page.wait_for_function(READY_JS, timeout=RENDER_TIMEOUT_MS)
                # a hash route renders in place: wait for the app to have routed to this page
                where = url.split("#/", 1)[1].split("?")[0] if "#/" in url else ""
                page.wait_for_function(ROUTED_JS, arg=where, timeout=RENDER_TIMEOUT_MS)
                if not page.evaluate("() => !!window.axe"):
                    page.add_script_tag(path=str(AXE))
                found[url] = page.evaluate("""async () => {
                    const r = await axe.run(document, {runOnly: {type: 'tag',
                        values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']}});
                    return r.violations.map(v => `${v.id} (${v.impact}): ${v.help}`);
                }""")
        except Exception as exc:
            return None, f"axe-core: {str(exc).splitlines()[0]}"
        finally:
            page.close()
        return found, None

    def ui_tests(self) -> None:
        m = self.ev.metrics
        res = self.ui(self.root / "ui")
        m.add(f"{PREFIX}.ui.tests_failed", res.failed, res.detail, error=res.error)
        m.add(f"{PREFIX}.ui.tests_run", res.run, res.detail, error=res.error)

    # -- all ---------------------------------------------------------------------------------------

    def run(self) -> None:
        m = self.ev.metrics
        with m.expect(*metric_ids()):
            inputs = self.workspace()
            live = self.start()
            if live is None:
                m.fail(metric_ids(), self.details.get("start_error", "server.sh did not start"))
            else:
                try:
                    with m.expect(*(f"{PREFIX}.parity.{k}" for k in PARITY_METRICS)):
                        self.parity(inputs)
                    with m.expect(*(f"{PREFIX}.{k}" for k in PERF_METRICS[3:])):
                        self.latency(inputs)
                    with m.expect(f"{PREFIX}.app_render_s", f"{PREFIX}.ui.a11y_violations"):
                        self.browser_checks(inputs)
                finally:
                    self.details["stop"] = self.stop(live).to_dict()
            self.ui_tests()
        self.ev.details["server_sh"] = self.details

