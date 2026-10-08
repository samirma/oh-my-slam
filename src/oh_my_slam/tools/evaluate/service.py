"""``server.sh`` on the reference inputs (http_server.md "Evaluation"): performance, parity with the
commands, and the web application's UI.

The service runs over a scratch workspace (``<out>/server_sh/ws``) holding copies of the reference
inputs and of their maps, with the inference proxy (``proxy``) as its inference server, and so do
the shell runs it is compared with. There are no jobs: an operation runs within its own HTTP
request, and the response is its result.

**Performance** (``server_sh.*``): start-up time (process start → the ``listening on`` line),
resident memory once listening (the process tree, idle), time from opening the web application
to its having rendered (``body[data-ready=true]``; a page without scripts: its ``load``), latency
of read-only requests (median and p95 idle, after the parity requests; p95 of the reads made
while an operation's request runs), and the overhead of an operation's request over the same
command run from the shell (median over the operations of the request's wall time — sent to
answered — minus the wall time of the same command with the same, replayed, inference).

**Parity** (``server_sh.parity.*``), for every mode of the commands and every reference input: the
operations come from the service's own description of the commands (``/api/openapi.json``: each
operation's ``x-oms`` entry, ``commands.spec.describe()`` as the API offers it), matched to their
definitions in ``commands.spec``, which give each case's command line (``spec.argv_of``, as the
service builds it), applicability (``When.holds``), result format (``spec.result_format``) and the
map it writes (``spec.writes_map``) — nothing here names a command or an option. The reference
inputs (``Inputs``, in the suite's order: restaurant.jpg, each example sequence with its
one-update map, street2.mp4) each give a value to every kind of path parameter they can. Each
operation runs on every reference input that has a value for each of its required parameters:
its default case there, and on the first such input one case per non-default value of each choice
and per point-cloud attribute (a non-default choice of its first enumerated attribute), with the
options a case needs to apply set as the definitions say. Each case runs the command from the
shell twice (the first run records the inference, the second replays it) and then as a request,
and compares the response body with the command's stdout byte for byte. A difference is a
mismatch, unless the command's own two runs differ too (``unverifiable``: the command is not
byte-reproducible). street2.mp4 is not mapped again: the suite's own ``mapper.sh update`` of it,
made through the recording proxy into the workspace (``Ran``), is the command's result its
request is compared with, without a control run. A case that maps writes the map at the same
workspace path each time, so the map's path in the result is the same; inputs are the same
absolute workspace paths on both sides.

**Contracts** (``contract.*.server_sh``): every response body is the command's result, so it keeps
the contracts of every command: exactly one payload of the result's format (stdout purity), and a
JSON result is a valid OpenLABEL scene whose object colours are those of their ids.

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
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oh_my_slam.commands import spec
from oh_my_slam.core.errors import OhMySlamError
from oh_my_slam.tools.evaluate.contracts import (
    parse_scene,
    payload_problems,
    scene_colour_problems,
)
from oh_my_slam.tools.evaluate.memory import tree_rss_mb
from oh_my_slam.tools.evaluate.runner import REPO, Live, RunRecord, RunSpec
from oh_my_slam.tools.evaluate.scene import doc_objects

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
UI_TEST_METRICS = ("tests_failed", "tests_run")  # the browser tests: they start their own service
UI_METRICS = (*UI_TEST_METRICS, "a11y_violations")


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


def workspace_root(out: Path) -> Path:
    """The service's scratch workspace under the evaluation's output folder, resolved: the paths
    the service gives the commands, which a run compared with a request must use too."""
    return (Path(out) / GROUP / "ws").resolve()


def input_folder(ws: Path, name: str) -> Path:
    """Where the files of the reference input ``name`` go in the workspace."""
    return Path(ws) / "inputs" / (Path(name).stem or "input")


def place(path: Path, folder: Path) -> Path:
    """``path`` put into ``folder`` (a hard link where the file system allows it, else a copy:
    a long video is not copied); the placed file."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(path).name
    if target.exists() and target.samefile(path):
        return target  # already there
    target.unlink(missing_ok=True)
    try:
        os.link(path, target)
    except OSError:
        shutil.copyfile(path, target)
    return target


@dataclass(frozen=True)
class Ran:
    """A command run the suite already made on a reference input (``argv``: after the program, as
    ``commands.spec.argv_of`` gives it), compared with instead of being run again: street2.mp4 is
    mapped once, by its own section, through the recording inference proxy."""

    argv: tuple[str, ...]
    record: RunRecord


@dataclass(frozen=True)
class Inputs:
    """One reference input inside the workspace (paths relative to it; ``map``: a map's name): the
    value it gives each kind of path parameter. An input the suite already ran (``ran``, which
    wrote the map ``writes``) has only that case."""

    name: str = ""  # the reference input (restaurant.jpg, ainex-captures, ..., street2.mp4)
    image: str | None = None  # one image (Kind "image")
    images: tuple[str, ...] = ()  # images to locate (Kind "images")
    sequence: tuple[str, ...] = ()  # images, or one video, to map (Kind "images_or_video")
    map: str | None = None  # an existing map (read-only modes)
    ran: Ran | None = None
    writes: str | None = None

    def of(self, kind: str) -> Any:
        return {"image": self.image, "images": list(self.images) or None,
                "images_or_video": list(self.sequence) or None, "map": self.map}.get(kind)


@dataclass
class Case:
    op: str  # the API operation id (``mapper-update``)
    label: str  # the command as typed (``mapper.sh update``)
    variant: str
    params: dict[str, Any]  # API parameters: workspace paths and map names
    writes_map: str | None = None  # the parameter naming the map the case creates
    result_format: str = "json"  # the format of the result (the response body / stdout)
    input: str = ""  # the reference input (``Inputs.name``)
    ran: Ran | None = None  # the run the suite already made of it (``Inputs.ran``)

    @property
    def name(self) -> str:
        return f"{self.label} [{self.variant}]" + (f" on {self.input}" if self.input else "")


@dataclass(frozen=True)
class Op:
    """An operation the service offers: its URL id, its ``x-oms`` entry (its
    ``commands.spec.describe()`` entry as the API offers it: the parameters) and its definitions
    (``commands.spec``), from which its command lines, applicability and result formats come."""

    id: str
    entry: dict[str, Any]
    command: spec.Command
    mode: spec.Mode

    @property
    def label(self) -> str:
        return self.command.label(self.mode)

    @property
    def prog(self) -> str:
        return spec.program_of(self.command).prog

    def params(self) -> dict[str, dict[str, Any]]:
        return {p["name"]: p for p in self.entry["parameters"]}


def operations_of(openapi: dict[str, Any]) -> tuple[list[Op], list[str]]:
    """The operations of the service's ``/api/openapi.json``: each ``POST /api/ops/<id>`` with its
    ``x-oms`` entry, matched to the mode of ``commands.spec`` whose operation id it is
    (``spec.operation_id``), in the service's order; and the operation paths whose entry is
    missing or names another mode."""
    modes = {spec.operation_id(p, c, m): (c, m) for p, c, m in spec.operations()}
    ops, problems = [], []
    for path, item in (openapi.get("paths") or {}).items():
        if not path.startswith("/api/ops/") or path.endswith("/validate"):
            continue
        oid = path.removeprefix("/api/ops/")
        entry = ((item or {}).get("post") or {}).get("x-oms")
        found = modes.get(oid)
        if found is None or not isinstance(entry, dict) or "parameters" not in entry \
                or entry.get("id") != found[0].label(found[1]):
            problems.append(path)
            continue
        ops.append(Op(oid, entry, *found))
    return ops, problems


def _attrs_value(p: dict[str, Any]) -> str | None:
    """A non-default choice of the first enumerated attribute (``color=segment``)."""
    for a in p.get("attributes") or []:
        schema = a.get("schema") or {}
        if schema.get("type") == "enum":
            other = [c for c in schema.get("choices") or [] if c != a.get("default")]
            if other:
                return f"{a['key']}={other[0]}"
    return None


def _inapplicable(op: Op, params: dict[str, Any], names: list[str]) -> list[str] | None:
    """The parameters of ``names`` whose option does not apply to ``params``: none of its
    conditions (``Option.applies``) holds (``When.holds``) on the arguments the command's own
    parser makes of them; None when they do not parse."""
    try:
        args = spec.parse(op.command, op.mode, params)
    except OhMySlamError:
        return None
    return [n for n in names if (o := op.command.option(n)).applies
            and not any(w.holds(args) for w in o.applies)]


def _applied(op: Op, base: dict[str, Any], given: dict[str, Any]) -> str | None:
    """Make every parameter of ``given`` (a variant) apply on ``base`` (a reference input's
    parameters): one that does not gets the first of its conditions a case can meet set — another
    parameter's value. Returns why it cannot, else None."""
    params = op.params()
    for name in list(given):
        applies = op.command.option(name).applies
        if not applies or _inapplicable(op, {**base, **given}, [name]) == []:
            continue
        settable = next((w for w in applies if w.option in params and w.values), None)
        if settable is not None:
            given[settable.option] = settable.values[0]
        if settable is None or _inapplicable(op, {**base, **given}, [name]) != []:
            return f"no reference input meets {[w.describe() for w in applies]}"
    return None


def _variants(op: Op, base: dict[str, Any]) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    """The default case and one per non-default value of each choice and per point-cloud
    attribute, made applicable on ``base``; and the variants that cannot be, with the reason."""
    out: list[tuple[str, dict[str, Any]]] = [("default", {})]
    for p in op.params().values():
        if p["kind"] == "enum":
            out += [(f"{p['name']}={c}", {p["name"]: c}) for c in p.get("choices") or []
                    if c != p.get("default")]
        elif p["kind"] == "attrs" and (v := _attrs_value(p)) is not None:
            out.append((f"{p['name']}={v}", {p["name"]: v}))
    done, skipped, seen = [], [], set()
    for name, given in out:
        why = _applied(op, base, given)
        if why is not None:
            skipped.append(f"{name}: {why}")
            continue
        key = json.dumps(given, sort_keys=True)
        if key not in seen:
            seen.add(key)
            done.append((name, given))
    return done, skipped


def parity_cases(ops: Sequence[Op], inputs: Sequence[Inputs]
                 ) -> tuple[list[Case], dict[str, str]]:
    """The cases of every operation of ``ops`` on the reference inputs (in order), and the
    operations no reference input can run, with the reason. An operation runs on every input that
    has a value for each of its required parameters: its default case there, and on the first
    such input also one case per variant (``_variants``); an input the suite already ran gets only
    the case that run is (``Inputs.ran``). A case that maps creates a map of its own."""
    cases: list[Case] = []
    uncovered: dict[str, str] = {}
    fresh = 0
    for op in ops:
        params = op.params()
        out = spec.writes_map(op.command, op.mode)
        map_out = out.name if out is not None and out.name in params else None
        required = [n for n, p in params.items() if p.get("required") and n != map_out]
        lacking: list[list[str]] = []
        first = True
        for inp in inputs:
            missing = [n for n in required if inp.of(params[n]["kind"]) is None]
            if missing:
                lacking.append(missing)
                continue
            base = {n: inp.of(params[n]["kind"]) for n in required}
            todo: list[tuple[str, dict[str, Any]]] = [("default", {})]
            if inp.ran is None and first:
                parsed = {**base, **({map_out: "parity"} if map_out is not None else {})}
                todo, first = _variants(op, parsed)[0], False
            for name, given in todo:
                case_params = dict(base)
                if map_out is not None and inp.ran is not None and inp.writes:
                    case_params[map_out] = inp.writes
                elif map_out is not None:
                    fresh += 1
                    case_params[map_out] = f"parity-{fresh}"
                case_params.update(given)
                fmt = spec.result_format(op.command, op.mode, case_params) or "json"
                cases.append(Case(op.id, op.label, name, case_params, map_out, fmt, inp.name,
                                  inp.ran))
        if not any(c.op == op.id for c in cases):
            least = min(lacking, key=len) if lacking else []
            uncovered[op.label] = "no reference input" + (
                f" for {', '.join(least)} ({', '.join(params[n]['kind'] for n in least)})"
                if least else "")
    return cases, uncovered


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

    def __init__(self, ev: Evaluation, inputs_from: Sequence[dict[str, Any]],
                 env: dict[str, str], ui: Callable[[Path], UiOutcome] = run_ui_tests) -> None:
        self.ev, self.env, self.ui = ev, env, ui
        self.root = ev.out / GROUP
        self.ws = self.root / "ws"
        self.inputs_from = inputs_from
        self.details: dict[str, Any] = {}
        self.busy_ms: list[float] = []
        self.overheads: dict[str, dict[str, float]] = {}
        self.api: Api | None = None

    # -- set-up ------------------------------------------------------------------------------------

    def workspace(self) -> list[Inputs]:
        """Put the reference inputs into the workspace (``input_folder``; a file the suite
        already put there stays) and copy their maps (named after the input); the cases' inputs,
        in order."""
        (self.ws / "inputs").mkdir(parents=True, exist_ok=True)
        (self.ws / "maps").mkdir(parents=True, exist_ok=True)
        self.ws = self.ws.resolve()
        out = []
        for src in self.inputs_from:
            name = str(src.get("name") or "")
            folder = input_folder(self.ws, name)
            image = self._put(src.get("image"), folder)
            images = tuple(x for p in src.get("images", []) if (x := self._put(p, folder)))
            sequence = tuple(x for p in src.get("sequence", []) if (x := self._put(p, folder)))
            map_name = None
            ref = src.get("map")
            if ref is not None and Path(ref).is_dir():
                map_name = folder.name
                shutil.copytree(ref, self.ws / "maps" / map_name, symlinks=True)
            out.append(Inputs(name, image, images, sequence, map_name, src.get("ran"),
                              src.get("writes")))
        return out

    def _put(self, path: Path | None, folder: Path) -> str | None:
        """``path`` copied into ``folder`` of the workspace (a file already in the workspace, as
        the suite put street2.mp4 there, stays); its workspace path, None if it is no file."""
        if path is None or not Path(path).is_file():
            return None
        real = Path(path).resolve()
        if self.ws in real.parents:
            return str(real.relative_to(self.ws))
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, folder / real.name)
        return str((folder / real.name).relative_to(self.ws))

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

    def absolute(self, op: Op, case: Case) -> dict[str, Any]:
        """The case's parameters with the absolute workspace paths the service gives the
        command."""
        params = op.params()
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

    def argv(self, op: Op, case: Case) -> list[str]:
        """The case's command line (after the program), as the service builds it for the same
        request: ``commands.spec.argv_of`` on the absolute workspace paths."""
        return spec.argv_of(op.command, op.mode, self.absolute(op, case))

    def shell_outcome(self, op: Op, case: Case, tag: str) -> tuple[Outcome, RunRecord]:
        rec = self.ev.run(tag, "parity", op.prog, *self.argv(op, case),
                          stdout=case.result_format, env=self.env)
        if case.result_format == "json":
            self.ev.scene(rec)
        return Outcome(rec.ok, None if rec.ok else rec.failure(), rec.stdout_bytes(),
                       wall_s=rec.wall_s), rec

    def ran_outcome(self, op: Op, case: Case) -> Outcome:
        """The run the suite already made of ``case`` (``Case.ran``; its contracts were checked
        then), when it is the case's command line."""
        assert case.ran is not None
        rec, argv = case.ran.record, self.argv(op, case)
        if list(case.ran.argv) != argv:
            return Outcome(False, f"the suite's run {rec.tag} is not the case's command line "
                                  f"({' '.join(case.ran.argv)} != {' '.join(argv)})")
        return Outcome(rec.ok, None if rec.ok else rec.failure(), rec.stdout_bytes(),
                       wall_s=rec.wall_s)

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

    def run_case(self, op: Op, case: Case, n: int) -> dict[str, Any]:
        row: dict[str, Any] = {"case": case.name, "op": case.op, "params": case.params}
        second: Outcome | None = None
        if case.ran is not None:  # the suite's run is the command's result: no control run
            first = self.ran_outcome(op, case)
            self._set_map_aside(case, f"parity_{n:02d}_shell")
            if not first.ok:  # nothing to compare a request with
                row.update(status="mismatch", why=first.error)
                return row
        else:
            first, _ = self.shell_outcome(op, case, f"parity_{n:02d}_shell")
            self._set_map_aside(case, f"parity_{n:02d}_shell")
            # the control: the same command again, with the same (replayed) inference
            second, _ = self.shell_outcome(op, case, f"parity_{n:02d}_shell_again")
            self._set_map_aside(case, f"parity_{n:02d}_shell_again")
        req = self.request_outcome(case)
        row["stages"] = req.stages
        if req.ok:
            self.response_contracts(case, f"parity_{n:02d}_request", req.result or b"")
        if not first.ok or not req.ok:
            row.update(status="mismatch", why=first.error if not first.ok else
                       f"the request failed: {req.error}")
            return row
        diffs = differences(first, req)
        if not diffs:
            row["status"] = "identical"
        elif second is not None and second.ok and differences(first, second):
            row.update(status="unverifiable", why="the command's own two runs differ: "
                       + "; ".join(differences(first, second)[:3]), request=diffs[:5])
        else:
            row.update(status="mismatch", why="; ".join(diffs[:5]) + (
                " (no control run: the suite's run is the command's result)"
                if second is None else ""))
        if second is not None and second.ok and req.wall_s is not None \
                and second.wall_s is not None and case.op not in self.overheads:
            self.overheads[case.op] = {"request_s": round(req.wall_s, 3),
                                       "shell_s": round(second.wall_s, 3),
                                       "overhead_s": round(req.wall_s - second.wall_s, 3)}
        return row

    def response_contracts(self, case: Case, tag: str, body: bytes) -> None:
        """The contracts of every command on a response body (the command's result): one payload
        of the result's format, and a scene that is valid OpenLABEL with its ids' colours."""
        log = self.ev.contracts
        log.check("stdout", "server_sh", tag, payload_problems(body, case.result_format))
        if case.result_format != "json":
            return
        doc, problems = parse_scene(body)
        log.check("openlabel", "server_sh", tag, problems)
        if doc is not None:
            log.check("colour", "server_sh", tag, scene_colour_problems(doc_objects(doc)))

    def parity(self, inputs: Sequence[Inputs]) -> None:
        m = self.ev.metrics
        ids = [f"{PREFIX}.parity.{k}" for k in PARITY_METRICS]
        assert self.api is not None
        reply = self.api.get("api/openapi.json")
        if reply.status != 200:
            m.fail(ids, f"/api/openapi.json answered HTTP {reply.status}")
            return
        found, problems = operations_of(reply.json())
        cases, uncovered = parity_cases(found, inputs)
        ops = {o.id: o for o in found}
        rows: list[dict[str, Any]] = [{"case": p, "status": "mismatch",
                                       "why": "not an operation of commands.spec"}
                                      for p in problems]
        for n, case in enumerate(cases, start=1):
            rows.append(self.run_case(ops[case.op], case, n))
        by_input = {i.name: sorted({c.label for c in cases if c.input == i.name})
                    for i in inputs}
        self.details["parity"] = {"cases": rows, "uncovered": uncovered, "inputs": by_input}
        bad = [r for r in rows if r["status"] == "mismatch"]
        unver = [r for r in rows if r["status"] == "unverifiable"]
        m.add(ids[0], len(bad), {"cases": len(rows), "mismatched": [
            {"case": r["case"], "why": r.get("why")} for r in bad]})
        m.add(ids[1], len(unver), {"unverifiable": [
            {"case": r["case"], "why": r.get("why")} for r in unver]})
        m.add(ids[2], len({c.op for c in cases}) / len(ops) if ops else None,
              {"operations": sorted(o.label for o in found), "uncovered": uncovered},
              error="the service describes no operation")

    # -- performance and UI ------------------------------------------------------------------------------

    def latency(self, map_name: str | None) -> None:
        """The read-only requests' latency (``READS``, and the page of the map ``map_name``)."""
        assert self.api is not None
        paths = list(READS)
        if map_name:
            paths.append(f"api/maps/{map_name}")
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

    def browser_checks(self, map_name: str | None) -> None:
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
            pages = app_urls(self.api.base, map_name)
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
        """The web application's browser tests (``UI_TEST_METRICS``). They start a service of
        their own, with the stub inference server, so they run whether or not the evaluated
        service started."""
        m = self.ev.metrics
        res = self.ui(self.root / "ui")
        m.add(f"{PREFIX}.ui.tests_failed", res.failed, res.detail, error=res.error)
        m.add(f"{PREFIX}.ui.tests_run", res.run, res.detail, error=res.error)

    # -- all ---------------------------------------------------------------------------------------

    def run(self) -> None:
        """Every metric of ``metric_ids`` is recorded once: a service that does not start fails
        each metric that needs it, with the reason, and the browser tests run in any case."""
        m = self.ev.metrics
        own = {f"{PREFIX}.ui.{k}" for k in UI_TEST_METRICS}  # ui_tests records them
        with m.expect(*metric_ids()):
            inputs = self.workspace()
            map_name = next((i.map for i in inputs if i.map), None)
            live = self.start()
            if live is None:
                m.fail([k for k in metric_ids() if k not in own],
                       self.details.get("start_error", "server.sh did not start"))
            else:
                try:
                    with m.expect(*(f"{PREFIX}.parity.{k}" for k in PARITY_METRICS)):
                        self.parity(inputs)
                    with m.expect(*(f"{PREFIX}.{k}" for k in PERF_METRICS[3:])):
                        self.latency(map_name)
                    with m.expect(f"{PREFIX}.app_render_s", f"{PREFIX}.ui.a11y_violations"):
                        self.browser_checks(map_name)
                finally:
                    self.details["stop"] = self.stop(live).to_dict()
            self.ui_tests()
        self.ev.details["server_sh"] = self.details

