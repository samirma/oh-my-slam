"""The agent skill ``SKILL.md`` (spec §2.7, ``agent_skill.md``): the committed file is the
generator's output; its description states every operation, what it produces and when the skill
can be used, within the Agent Skills limit; it names every operation and route of the ``server.sh``
API with what it does, pointing to ``/api/openapi.json`` instead of repeating it; it offers no
operation, parameter, value, route or local script the service does not support, and runs no
script; every limit and number it states is the code's, and follows the code when it changes; a
registry change reaches it with no hand edit; and its snippet is POSIX ``sh`` that finds a running
``server.sh`` in the spec's order."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from jsonschema import Draft202012Validator

from oh_my_slam.commands import entry_points, spec
from oh_my_slam.core import constants
from oh_my_slam.core.errors import (
    API_MEANING,
    HTTP_STATUS,
    MEANING,
    SERVER_HINT,
    ExitCode,
    ServerUnavailableError,
    ServiceNotRunningError,
    error_code,
)
from oh_my_slam.schema import openlabel
from oh_my_slam.web import app as web_app
from oh_my_slam.web import main as web_main
from oh_my_slam.web import openapi, skill
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.operations import Operation
from oh_my_slam.web.runner import Runner
from oh_my_slam.web.workspace import Workspace
from tests.fakes import slow_command
from tests.unit.test_web_server_sh import LINE, server, start

REPO = Path(__file__).resolve().parents[2]
REGENERATE = "regenerate it: uv run python -m oh_my_slam.web.skill"
API_ROUTES = {"GET /api/health", "GET /api/openapi.json", "POST /api/ops/{op}",  # spec §2.6 "API"
              "POST /api/ops/{op}/validate", "POST /api/uploads", "PUT /api/uploads",
              "DELETE /api/uploads/{id}", "GET /api/maps", "GET /api/maps/{name}"}
SH_BLOCK = re.compile(r"^ *```sh\n(.*?)^ *```$", re.S | re.M)
CODE_SPAN = re.compile(r"`([^`\n]+)`")
FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
SCRIPT = re.compile(r"[\w-]+\.sh\b")
# what the skill may tell the user to run (agent_skill.md): the start of either server, and
# server.sh --status for the service's URL (with the start's --port and --data)
USER_COMMANDS = {"./start_inference_server.sh", "start_inference_server.sh", "./server.sh",
                 "server.sh", "server.sh --status", "server.sh --port <n>"}


def committed() -> str:
    return (REPO / "SKILL.md").read_text("utf-8")


def part(text: str, title: str) -> str:
    """The ``## title`` section of ``text``."""
    return text.split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]


def description(text: str) -> str:
    return str(yaml.safe_load(text.split("---\n", 2)[1])["description"])


def snippet(text: str) -> str:
    """The first ``sh <<'EOF'`` … ``EOF`` snippet of ``text``."""
    m = re.search(r"^sh <<'EOF'\n(.*?)^EOF$", text, re.S | re.M)
    assert m, "no snippet"
    return m.group(1)


def flat(text: str) -> str:
    return " ".join(text.split())


def ops() -> dict[str, Operation]:
    return skill.operations()


def document() -> dict[str, Any]:
    return openapi.document(ops())


def api_routes(tmp_path: Path) -> list[str]:
    """``METHOD /path`` of every route of the real app (HEAD aside), path parameters as in the
    OpenAPI document."""
    ws = Workspace(tmp_path / "data")
    ws.create()
    service = Service(ws, Runner(ws), url="http://0.0.0.0:0/")
    app = inspect.getclosurevars(create_app(service)).nonlocals["app"]  # inside the guard
    service.runner.shutdown()
    return sorted(f"{m} " + re.sub(r"\{(\w+):\w+\}", r"{\1}", r.path)
                  for r in app.routes for m in r.methods - {"HEAD"})


def curl_flags() -> set[str]:
    """Every option this machine's curl has."""
    out = subprocess.run(["curl", "--help", "all"], capture_output=True, text=True, timeout=30)
    return set(FLAG.findall(out.stdout))


def flags_of(words: list[str]) -> set[str]:
    """The options among ``words``, combined short ones apart (``-sS``: ``-s``, ``-S``)."""
    out = set()
    for w in words:
        if re.fullmatch(r"-[A-Za-z]{2,}", w):
            out |= {f"-{c}" for c in w[1:]}
        elif re.fullmatch(r"--?[A-Za-z][\w-]*", w):
            out.add(w)
    return out


def schema(doc: dict[str, Any], name: str) -> Draft202012Validator:
    """A validator of the document's schema ``name`` (its references resolved in the document)."""
    return Draft202012Validator({**doc, "$ref": f"#/components/schemas/{name}"})


def replaced(program: spec.Program, command: str | None, **changes: Any) -> spec.Program:
    """``program`` with ``changes`` applied to its command ``command``."""
    cmds = tuple(dataclasses.replace(c, **changes) if c.name == command else c
                 for c in program.commands)
    return dataclasses.replace(program, commands=cmds)


def with_programs(monkeypatch: pytest.MonkeyPatch, *programs: spec.Program) -> None:
    """The registry with ``programs`` in place of those of the same name (or added)."""
    names = {p.prog for p in programs}
    kept = [p for p in spec.PROGRAMS if p.prog not in names]
    monkeypatch.setattr(spec, "PROGRAMS", (*kept, *programs))


# -- the file and its description ----------------------------------------------------------------


def test_the_committed_skill_is_the_generated_one() -> None:
    assert committed() == skill.render(), f"SKILL.md is out of date; {REGENERATE}"
    assert skill.DEFAULT_PATH == REPO / "SKILL.md"


def test_front_matter() -> None:
    _, front, body = committed().split("---\n", 2)
    meta = yaml.safe_load(front)
    assert meta["name"] == "oh-my-slam" and set(meta) == {"name", "description"}
    assert 0 < len(meta["description"]) <= skill.DESCRIPTION_MAX == 1024  # the Agent Skills limit
    assert "/api/openapi.json" in body and "**the running service's document wins**" in flat(body)


def test_the_description_states_every_capability_and_when_to_use_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §2.7 "Description": every operation with what it takes and produces, the need for a
    server.sh reachable from the agent's machine, and the operations that need the inference
    server."""
    text = description(committed())
    assert "server.sh reachable from this machine" in text and "no checkout" in text
    assert "with sh and curl" in text
    clauses = text.split("Operations: ", 1)[1].split(". ", 1)[0].split("; ")
    by_id = {c.split(": ", 1)[0]: c for c in clauses}
    assert list(by_id) == list(ops())
    for op in ops().values():
        clause = by_id[op.id]
        for o in skill.inputs(op):
            assert skill.takes(o) in clause, (op.id, o.name)
        for out in skill.results(op):
            assert skill.FORMATS[out.format] in clause.split(" → ")[-1], (op.id, out.format)
    (writer,) = [op.id for op in ops().values() if op.writes_map() is not None]
    assert f"{writer} writes a map; the others are read-only" in text
    assert {o.format for op in ops().values() for o in skill.results(op)} == {"json", "png", "ply"}
    assert "JSON = OpenLABEL scene" in text and "PLY = point cloud" in text
    assert "PNG = depth image or segmented image" in text  # the names of the PNG results

    def ids(inference: str) -> list[str]:
        return [op.id for op in ops().values() if op.mode.inference == inference]

    assert ("Inference server needed by " + skill.listing(ids("required")) + ", and by "
            f"mapper-locate for maps over {constants.UPDATE_EXHAUSTIVE_MAX} keyframes; the other "
            "routes work without it.") in text
    assert set(SCRIPT.findall(text)) == {entry_points.WEB_SERVICE.prog}
    assert "job" not in text.lower() and "view" not in text.lower()
    monkeypatch.setattr(skill, "DESCRIPTION_MAX", 100)  # one that would not fit is refused
    with pytest.raises(ValueError, match="Agent Skills: at most 100"):
        skill.render()


# -- the operations and routes ----------------------------------------------------------------------


def table(text: str) -> dict[str, list[str]]:
    """The operations' table: operation → its cells."""
    rows = {}
    for line in part(text, "Operations and routes").splitlines():
        if line.startswith("| `"):
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
            rows[cells[0].strip("`")] = cells
    return rows


def test_every_operation_and_route_is_named_with_what_it_does(tmp_path: Path) -> None:
    """Spec §2.7 "Points to the API description": every route (the app's, exactly the spec's)
    and every operation by name, with what it takes and gives, whether it writes a map and whether
    it needs the inference server — and none of the operations' parameters' descriptions, which
    /api/openapi.json gives."""
    text = committed()
    routes = api_routes(tmp_path)
    assert {r for r in routes if r.split(" ", 1)[1].startswith("/api/")} == API_ROUTES
    section = part(text, "Operations and routes")
    missing = [r for r in API_ROUTES if f"`{r}`" not in section]
    assert not missing, f"routes missing from SKILL.md ({REGENERATE}): {missing}"
    rows = table(text)
    assert list(rows) == list(ops()), REGENERATE
    for op in ops().values():
        _, route, takes, writes, inference = rows[op.id]
        assert route == f"`POST /api/ops/{op.id}`"
        assert set(re.findall(r"`(\w+)`", takes)) <= {o.name for o in op.options}
        assert all(f"`{o.name}`" in takes for o in skill.inputs(op))
        assert all(skill.FORMATS[o.format] in takes for o in skill.results(op))
        assert (writes == "read-only") == (op.writes_map() is None), op.id
        assert inference == {"required": "needed", "never": "not needed"}.get(
            op.mode.inference, f"only for {skill.condition(op.mode)}")
    for p in spec.PROGRAMS:  # the document's job: no parameter's description is repeated
        for c in p.commands:
            for o in c.options:  # (a few words, such as "map folder", may occur anywhere)
                assert len(o.help) < 20 or o.help not in text, (c.label(c.modes[0]), o.flag)
    assert "| Parameter" not in text and "| Option" not in text
    assert ("`$BASE/api/openapi.json` describes every operation, parameter, result and error"
            in text)


def test_the_order_of_calls_the_errors_and_the_rules() -> None:
    """Spec §2.7: the typical order of calls (raw uploads with curl -T, -F refused; validate;
    run with no client timeout, saved with -o; Server-Timing), the error shape and the service's
    own refusals, the rules, and a ready-to-run curl example for each operation."""
    text = committed()
    flow = part(text, "A request, call by call")
    steps = [flow.index(s) for s in ("**Check the service", "**Upload", "**Validate",
                                     "**Run it and wait", "**Read the stage timings", "**Report")]
    assert steps == sorted(steps)
    assert "curl -T <file>" in flow and "curl -F" in flow and "with no client timeout" in flow
    assert "--max-time" not in flow and " -o result" in flow and "byte for byte" in flow
    assert "grep -i '^server-timing:' headers.txt" in flow
    examples = SH_BLOCK.findall(part(text, "Examples"))[0].splitlines()
    for op in ops().values():
        runs = [line for line in examples if f'"$BASE/api/ops/{op.id}"' in line]
        assert len(runs) == 1 and " -o " in runs[0], op.id
        assert any(line.startswith(f"# {op.id}: ") for line in examples)
        i = examples.index(runs[0])
        uploads = [o for o in skill.inputs(op) if o.kind in spec.ACCEPTS]
        assert all(line.startswith("curl -sS -T ") for line in examples[i - len(uploads):i])
    errors = part(text, "Errors")
    assert '{"error": {"code": "<code>", "message": "<the command\'s own message>"' in errors
    rules = flat(part(text, "Rules"))
    for rule in ("The inference server may be down.",
                 "Never start or stop `start_inference_server.sh` or `server.sh`",
                 "tell the user how to start it", "Never change a map except through",
                 "Ask the user first** before updating an existing map",
                 "before starting a long mapping request",
                 "Inputs are uploads or paths inside the workspace",
                 "An upload is consumed by the one request it is given to",
                 "Wait for an answer with no client timeout", "never a local script",
                 "leave it to the user"):
        assert rule in rules, rule


def test_the_skill_offers_nothing_the_service_does_not_support(tmp_path: Path) -> None:
    """No route, operation, parameter or value the service does not have — every curl command is
    checked against the app's routes and the operations' parameters, and uses only curl's own
    options — and no local script: it runs none, documents none of their options, and names only
    what the user runs (starting a server, server.sh --status)."""
    text = committed()
    routes = [r.split(" ", 1) for r in api_routes(tmp_path)]
    patterns = [(m, re.compile("^" + re.sub(r"\{\w+\}", "[^/]+", path) + "$"))
                for m, path in routes]

    def route(method: str, path: str) -> bool:
        return any(m == method and pat.match(path) for m, pat in patterns)

    for method, path in re.findall(r"`(GET|POST|PUT|PATCH|DELETE) (/[^`\s]*)", text):
        assert route(method, path.split("?")[0]), (method, path)
    curl = curl_flags()
    blocks = SH_BLOCK.findall(text)
    curls = [line.strip() for b in blocks for line in b.splitlines()
             if line.strip().startswith("curl ") and "$BASE" in line]
    curls += [s for s in CODE_SPAN.findall(text) if "curl " in s and "$BASE" in s]
    assert len(curls) >= 3 + 2 * len(ops())
    for line in curls:
        words = shlex.split(line[line.index("curl "):])
        assert flags_of(words) <= curl, line
        method = words[words.index("-X") + 1] if "-X" in words else (
            "PUT" if "-T" in words else "GET")
        path, _, query = next(w for w in words if w.startswith("$BASE/")).removeprefix(
            "$BASE").partition("?")
        assert route(method, path), line
        if path == "/api/uploads":
            assert re.fullmatch(r"name=[\w.-]+", query), line
        if path.startswith("/api/ops/"):
            op = ops()[path.split("/")[3]]
            body = json.loads(words[words.index("-d") + 1])
            names = {o.name: o for o in op.options}
            assert set(body) <= set(names), line
            for k, v in body.items():
                assert names[k].choices is None or v in names[k].choices
    # flags named in passing are curl's, or those of what the user runs (server.sh)
    service_flags = {o.flag for o in entry_points.WEB_SERVICE.command().options
                     if o.name in ("status", "port", "data")}
    prose = SH_BLOCK.sub("", text)
    named = flags_of(FLAG.findall(prose))
    assert named <= curl | service_flags, named - curl - service_flags
    assert not re.search(r"(?<![\w-])-\w+=", text)  # no command line of a command
    # scripts: only the two servers, named for what the user runs; none is run
    commands = {p.prog for p in spec.PROGRAMS}
    servers = {entry_points.INFERENCE_SERVER.prog, entry_points.WEB_SERVICE.prog}
    assert commands | servers == {f.name for f in REPO.glob("*.sh")}
    assert set(SCRIPT.findall(text)) == servers
    for span in CODE_SPAN.findall(text):
        if SCRIPT.search(span) and not span.startswith(f"{entry_points.WEB_SERVICE.prog}: "):
            assert span in USER_COMMANDS, span
    assert {f.split()[1] for f in USER_COMMANDS if " " in f} <= service_flags
    for block in blocks:  # comments and messages aside, a block runs curl and sh tools only
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', block)
        code = re.sub(r"(?m)(^|\s)#.*$", "", code)
        assert not SCRIPT.search(code), code
    for gone in ("/api/jobs", "cancel", "$REPO", "OMS_REPO", "oh-my-slam/repo", "uv sync",
                 "view.sh", "examples/", "--no-browser", "--stop"):
        assert gone not in text, gone


# -- the errors -----------------------------------------------------------------------------------


def test_every_error_the_service_answers_with_is_described() -> None:
    """The exit-code table (each code, its HTTP status and meaning) and the service's own
    refusals, as the OpenAPI document exports them from the code."""
    text = committed()
    errors = part(text, "Errors")
    rows = re.findall(r"^\| (\d+) \| `(\w+)` \| (\d+) \| (.*) \|$", errors, re.M)
    assert rows == [(str(int(c)), error_code(c), str(HTTP_STATUS[c]), API_MEANING[c])
                    for c in ExitCode]
    # an API client's meanings: no script option, no Ctrl-C; 130 is the service's stop
    for c, meaning in API_MEANING.items():
        assert "--status" not in meaning and "Ctrl-C" not in meaning, c
        assert meaning == MEANING[c] or c in (ExitCode.SERVER_UNAVAILABLE, ExitCode.INTERRUPTED)
    assert "the service stopped" in API_MEANING[ExitCode.INTERRUPTED]
    refused = re.findall(r"^\| `(\w+)` \| (\d+) \| (.*) \|$", errors, re.M)
    table_ = web_app.refusals()
    assert refused == [(r.code, str(r.http_status), r.when) for r in table_.values()]
    assert skill.refusals(document()) == {r.code: dataclasses.asdict(r) for r in table_.values()}


# -- the limits and numbers it states ----------------------------------------------------------------


def numbers_free(text: str, *allowed: str) -> str:
    """``text`` without its placeholders, network addresses, "3D" and the ``allowed`` patterns."""
    for pattern in (r"\{\{\w+\}\}", r"0\\?\.0\\?\.0\\?\.0", r"127\\?\.0\\?\.0\\?\.1", r"\b3D\b",
                    *allowed):
        text = re.sub(pattern, "", text)
    return text


def test_the_template_states_no_number_of_its_own() -> None:
    """Every number of SKILL.md comes from the code: the template holds none (but network
    addresses, a list's numbering and the snippet's shell syntax)."""
    for name in ("INTRO", "RULES", "OPERATIONS", "FIND", "WORKFLOW", "EXAMPLES", "ERRORS"):
        rest = numbers_free(getattr(skill, name), r"(?m)^\d\. ")
        assert not re.search(r"\d", rest), (name, re.findall(r".{20}\d.{20}", rest))
    rest = numbers_free(skill.ADDRESS_SNIPPET, r"exit [01]\b", r"2>/dev/null", r">&2", r"\$1\b",
                        r"\\1", r"0-9", r"# [123]\. ")
    assert not re.search(r"\d", rest), re.findall(r".{20}\d.{20}", rest)


def test_every_limit_the_skill_states_is_the_codes() -> None:
    text = committed()
    errors = part(text, "Errors")
    assert f"| `too_large` | 413 | an upload over {web_app.MAX_UPLOAD_BYTES / 2**30:g} GiB |" \
        in errors
    assert f"less than {web_app.MIN_FREE_BYTES / 2**30:g} GiB free" in errors
    assert document()["x-oms"]["limits"] == {"max_upload_bytes": web_app.MAX_UPLOAD_BYTES,
                                             "min_free_bytes": web_app.MIN_FREE_BYTES}
    probe = re.search(r"--max-time (\d+) ", snippet(part(text, "Find the service")))
    assert probe and int(probe.group(1)) == skill.PROBE_TIMEOUT_S
    assert f"within {skill.PROBE_TIMEOUT_S} s" in flat(part(text, "Find the service"))
    assert set(skill.CONDITION_WORDS) == set(spec.CONDITIONS)  # every condition has its words
    for op in ops().values():
        for k, v in (op.mode.inference_condition or {}).items():
            assert k != "map_keyframes_greater_than" or v == constants.UPDATE_EXHAUSTIVE_MAX
            assert f"over {v} keyframes" in table(text)[op.id][4]
    rules = flat(part(text, "Rules"))
    down = ExitCode.SERVER_UNAVAILABLE
    assert f"fail with HTTP {HTTP_STATUS[down]} `{error_code(down)}`" in rules
    assert f"the command that starts it, `{web_app.START_COMMAND}`" in rules
    assert ("offer what works without it: the workspace's maps (`GET /api/maps`, "
            "`GET /api/maps/{name}`); `mapper-locate` except for maps over "
            f"{constants.UPDATE_EXHAUSTIVE_MAX} keyframes.") in rules
    (writer,) = [op for op in ops().values() if op.writes_map() is not None]
    stages = [str(s) for s in writer.mode.stages]
    assert f"runs {len(stages)} stages, from `{stages[0]}` to `{stages[-1]}`" in rules
    refused = web_app.refusals()
    assert f"answers {refused['not_found'].http_status} `not_found` for a new one" in rules
    outside = ExitCode.USAGE
    assert f"which are refused ({HTTP_STATUS[outside]} `{error_code(outside)}`)" in rules
    find = flat(part(text, "Find the service"))
    assert f"answers {refused['forbidden'].http_status} `forbidden`" in find
    assert f"curl exit {' or '.join(map(str, skill.CURL_EXITS))}" in find
    flow = flat(part(text, "A request, call by call"))
    assert f"is refused with {refused['unsupported_media_type'].http_status}" in flow


def test_a_changed_limit_changes_the_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """The limits are read from the code when the file is rendered: changing one changes it."""
    before = skill.render()
    monkeypatch.setattr(web_app, "MAX_UPLOAD_BYTES", 3 << 30)
    monkeypatch.setattr(skill, "PROBE_TIMEOUT_S", 5)
    update, locate = spec.MAPPER.command("update"), spec.MAPPER.command("locate")
    (locating,) = locate.modes
    (updating,) = update.modes
    mapper = replaced(replaced(
        spec.MAPPER, "update", modes=(dataclasses.replace(updating, stages=updating.stages[:-1]),)),
        "locate", modes=(dataclasses.replace(locating, inference_condition={
            "map_keyframes_greater_than": 300}),))
    with_programs(monkeypatch, mapper)
    text = skill.render()
    assert text != before
    assert "an upload over 3 GiB" in text and "an upload over 8 GiB" not in text
    assert "--max-time 5 " in text and "within 5 s" in text and "--max-time 3 " not in text
    assert "maps over 300 keyframes" in text and "over 150" not in text
    assert f"runs {len(updating.stages) - 1} stages" in flat(text)


# -- the samples ----------------------------------------------------------------------------------------


def test_the_samples_follow_the_code(tmp_path: Path) -> None:
    """The lines, records and layouts the skill shows are the code's and the document's."""
    text = committed()
    doc = document()
    assert LINE.match("server.sh: listening on http://0.0.0.0:8123/")
    assert "`server.sh: listening on http://0.0.0.0:<port>/`" in text
    # what the user runs: each server's start, and the service's status
    assert skill.servers() == [entry_points.INFERENCE_SERVER, entry_points.WEB_SERVICE]
    assert skill.start_command(entry_points.INFERENCE_SERVER) == web_app.START_COMMAND
    # the inference server's start command is one definition: its errors and the health name it
    assert web_app.START_COMMAND == constants.START_INFERENCE_SERVER \
        == constants.CHECKOUT + entry_points.INFERENCE_SERVER.prog
    assert SERVER_HINT.endswith(web_app.START_COMMAND)
    assert str(ServerUnavailableError()).endswith(SERVER_HINT)
    with pytest.raises(ServiceNotRunningError,  # what server.sh --status says when it is down
                       match=f"start it with {skill.start_command(entry_points.WEB_SERVICE)} "
                             "--data"):
        web_main.status(Workspace(tmp_path / "none"))
    assert skill.status_command() == "server.sh --status"
    assert {skill.service_flag("port"), skill.service_flag("data")} == {"--port", "--data"}
    # the health fields the workflow names, and the inference server's states
    health = doc["components"]["schemas"]["Health"]["properties"]
    assert {"data", "url", "requests"} <= set(health["service"]["properties"])
    assert set(health["service"]["properties"]["requests"]["properties"]) == {"running",
                                                                              "waiting"}
    inference = health["inference"]["properties"]
    assert "start_command" in inference
    assert skill.statuses() == ", ".join(f"`{s}`" for s in inference["status"]["enum"])
    for name, sample in (("Upload", r"→ \d+ `(\{.*?\})`: its `path`"),
                         ("Validation", r"→ `(\{\"valid\".*?\})`")):
        m = re.search(sample, text)
        assert m, name
        schema(doc, name).validate(json.loads(m.group(1).replace("[…]", "[]")))
    m = re.search(r"answers (\d+) `(\{\"error\".*?\})`; a", flat(part(text, "Errors")))
    assert m and int(m.group(1)) == HTTP_STATUS[ExitCode.USAGE]
    schema(doc, "Error").validate(json.loads(m.group(2).replace("[…]", "[]")))
    m = re.search(r"e\.g\. (\d+) `(\{.*?\})`\.", flat(part(text, "Errors")))
    assert m and int(m.group(1)) == HTTP_STATUS[ExitCode.NOT_REGISTERED]
    schema(doc, "Error").validate(json.loads(m.group(2)))
    ex = skill.example(ops())
    assert skill.server_timing([str(s) for s in ex.mode.stages]) in text
    cub = openlabel.cuboid_val(np.array([1.0, 2.0, 3.0]), np.eye(3), np.array([4.0, 5.0, 6.0]))
    named = dict(zip(skill.CUBOID, cub, strict=True))
    assert (named["x"], named["y"], named["z"], named["qw"], named["sx"], named["sz"]) == (
        1, 2, 3, 1, 4, 6)
    assert f"`val` is `{','.join(skill.CUBOID)}`" in text
    # a PNG result is named by the outputs in that format, with the parameter value that gives each
    seg = next(op.id for op in ops().values() if op.program is spec.SEGMENT)
    assert ("PNG is the depth image (`reconstruct` with `\"format\":\"depth\"`) or the segmented "
            f"image (`{seg}` with `\"format\":\"png\"`)") in text


def test_the_generator_renders_every_kind_of_input_and_result() -> None:
    """Kinds the registry may use that no operation does yet: a required choice, a result for a
    video only or for a parameter given, a result in a format with no gloss, an operation with no
    input file or none on one image."""
    pick = spec.Option("--pick", "pick", spec.Kind.ENUM, "a choice", required=True,
                       choices=("a", "b"))
    count = spec.Option("--count", "count", spec.Kind.NUMBER, "how many", required=True,
                        default=3)
    (mode,) = spec.RECONSTRUCT.commands[0].modes
    png = spec.Output("result", "stdout", "png", "an image")
    shot = dataclasses.replace(mode, rules=(), attrs_scope=None, outputs=(png,))
    cmd = spec.Command("pick.sh", None, "pick", (pick, count), (shot,))
    op = Operation(spec.Program("pick.sh", "pick", (cmd,)), cmd, shot)
    assert skill.sample_params(op) == {"pick": "a", "count": 3}
    assert skill.takes_gives(op) == "PNG (an image)" and skill.gives(op) == "PNG (an image)"
    assert skill.formats_text({"pick": op}) == "Results: PNG is an image (`pick`)."
    assert skill.example({"pick": op}) is op
    assert skill.when_text((spec.When("inputs", video=True),)) == "for one video in `inputs`"
    assert skill.when_text((spec.When("map"), spec.When("mode", ("full", "single")))) == \
        'with `map` or with `"mode":"full"` or with `"mode":"single"`'
    locate = spec.MAPPER.command("locate")
    assert skill.takes(locate.option("map")) == skill.EXISTING_MAP
    assert skill.sample_file(spec.Option("-x", "x", spec.Kind.MAP, "a map")) == "photo.jpg"
    assert skill.title(png) == "an image" and skill.title(spec.Output(
        "result", "stdout", "png", "the depth image: 16 bits")) == "the depth image"


def test_lists_read_as_sentences() -> None:
    assert [skill.listing(list(items)) for items in ("", "a", "ab", "abc")] == [
        "", "a", "a and b", "a, b and c"]
    assert skill.listing(["a", "b"], "or") == "a or b"


def test_the_inference_need_reads_right_whatever_the_modes_need(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The description and the rules follow the modes' inference need: with no mode that needs
    it at times, none is named so; one that never needs it works without it."""
    locate = spec.MAPPER.command("locate")
    (mode,) = locate.modes
    with_programs(monkeypatch, replaced(spec.MAPPER, "locate", modes=(dataclasses.replace(
        mode, inference="never", inference_condition=None),)))
    text = skill.render()
    assert "; mapper-locate and the other routes work without it." in description(text)
    assert ("offer what works without it: the workspace's maps (`GET /api/maps`, "
            "`GET /api/maps/{name}`); `mapper-locate`.") in flat(part(text, "Rules"))
    assert table(text)["mapper-locate"][4] == "not needed"


def test_main_writes_the_skill(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "SKILL.md"
    assert skill.main([str(out)]) == 0 and out.read_text("utf-8") == committed()
    assert skill.main(["-h"]) == 0 and skill.main(["a", "b"]) == 2
    assert "usage: python -m oh_my_slam.web.skill" in capsys.readouterr().err
    res = subprocess.run([sys.executable, "-m", "oh_my_slam.web.skill", str(out)],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0 and res.stderr == f"wrote {out}\n"


# -- a registry change -----------------------------------------------------------------------------


def test_a_registry_change_reaches_the_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new input of an operation and a whole new command change the skill with no edit to the
    generator: its row, example and description; a new optional parameter is the document's."""
    locate = spec.MAPPER.command("locate")
    ref = spec.Option("--ref", "ref", spec.Kind.IMAGE, "a reference image", required=True)
    radius = spec.Option("--radius", "radius", spec.Kind.NUMBER, "search radius in metres",
                         default=1.5, minimum=0, type=float)
    mapper = replaced(spec.MAPPER, "locate", options=(*locate.options, ref, radius))
    with_programs(monkeypatch, mapper, slow_command.registry_program())  # type: ignore[arg-type]
    text = skill.render()
    assert text != committed()
    rows = table(text)
    assert "`ref` (one image)" in rows["mapper-locate"][2]
    assert rows["slow"] == ["`slow`", "`POST /api/ops/slow`", "JSON", "**writes the map** `map`",
                            "not needed"]
    assert '"$BASE/api/ops/slow"' in part(text, "Examples")
    front = description(text)
    assert "; slow: JSON." in front and "mapper-update and slow write a map" in front
    assert "; slow and the other routes work without it." in front
    props = document()["paths"]["/api/ops/mapper-locate"]["post"]["requestBody"]["content"][
        "application/json"]["schema"]["properties"]
    assert props["radius"]["minimum"] == 0 and "radius" not in text


# -- the snippet ---------------------------------------------------------------------------------------

SHELLS = [s for s in ("sh", "dash") if shutil.which(s)]


@pytest.mark.parametrize("shell", SHELLS)
def test_address_snippet_is_posix_and_finds_the_service_in_order(tmp_path: Path,
                                                                  shell: str) -> None:
    """User URL → cached URL while it answers → server.json of the workspace (0.0.0.0 →
    127.0.0.1) → ask the user; the URL found is cached, and nothing is scanned."""
    code = snippet(part(committed(), "Find the service"))
    assert subprocess.run([shell, "-n"], input=code, text=True).returncode == 0
    home, data = tmp_path / "home", tmp_path / "home" / "ws"
    cache = tmp_path / "cache" / "oh-my-slam" / "server_url"
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "XDG_CACHE_HOME": str(tmp_path / "cache")}

    def run(**extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([shell], input=code, env={**env, **extra}, capture_output=True,
                              text=True, timeout=60)

    none = run()
    assert none.returncode == 1 and none.stdout == "" and not cache.exists()
    assert "Ask the user" in none.stderr and "server.sh --status" in none.stderr
    proc, url = start(data)
    try:
        base = url.rstrip("/")
        port = base.rsplit(":", 1)[1]
        assert '"url":"http://0.0.0.0:' in (data / "server.json").read_text()
        found = run(OMS_DATA=str(data))  # 3. the workspace's server.json
        assert (found.returncode, found.stdout) == (0, base + "\n"), found.stderr
        assert cache.read_text() == base + "\n"
        assert run().stdout == base + "\n"  # 2. the cache, while it answers
        cache.write_text("http://127.0.0.1:9\n")  # a stale cache: searched again
        assert run(OMS_DATA="~/ws").stdout == base + "\n"
        assert cache.read_text() == base + "\n"
        # 1. the user's URL wins (its path dropped, 0.0.0.0 is this machine), and must answer
        assert run(OMS_URL=f" http://0.0.0.0:{port}/#/maps ").stdout == base + "\n"
        assert run(OMS_URL=f"localhost:{port}").stdout == f"http://localhost:{port}\n"
        dead = run(OMS_URL="http://127.0.0.1:9")
        assert dead.returncode == 1 and "check the URL with the user" in dead.stderr
        assert run(OMS_URL=f"http://127.0.0.1:{port}$(touch {tmp_path}/x)").returncode == 1
        assert not (tmp_path / "x").exists()
    finally:
        server("--data", str(data), "--stop")
        proc.wait(60)
    gone = run(OMS_DATA=str(data))  # the cached URL stopped answering and server.json is gone
    assert gone.returncode == 1 and "Ask the user" in gone.stderr
