"""The agent skill ``SKILL.md`` (spec §2.7, ``agent_skill.md``): the committed file is the
generator's output; its description states every capability within the Agent Skills limit; it
covers every script and mode (where it is, its options with their defaults and allowed values, what
it does, a command the script's own parser accepts, a sample result, the error shape) and every API
route and operation, pointing to ``/api/openapi.json`` instead of repeating it; it offers no script,
mode, option, value, route or operation the project does not have; every limit and number it states
is the code's, and follows the code when it changes; a registry change reaches it with no hand edit;
and its two snippets are POSIX ``sh`` that find the checkout and a running ``server.sh`` in the
spec's order."""

from __future__ import annotations

import dataclasses
import glob
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
from starlette.testclient import TestClient

from oh_my_slam.cli import view as cli_view
from oh_my_slam.client.protocol import Health
from oh_my_slam.commands import entry_points, spec
from oh_my_slam.commands.parser import RaisingParser
from oh_my_slam.core import constants, process
from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.errors import HTTP_STATUS, MEANING, ExitCode, OhMySlamError, error_code
from oh_my_slam.schema import openlabel
from oh_my_slam.web import app as web_app
from oh_my_slam.web import skill
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.operations import Operation
from oh_my_slam.web.runner import STOPPING, Outcome, Runner
from oh_my_slam.web.workspace import Workspace
from tests.fakes import slow_command
from tests.unit.test_web_server_sh import LINE, server, start

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "src" / "oh_my_slam" / "web"
REGENERATE = "regenerate it: uv run python -m oh_my_slam.web.skill"
API_ROUTES = {"GET /api/health", "GET /api/openapi.json", "POST /api/ops/{op}",  # spec §2.6 "API"
              "POST /api/ops/{op}/validate", "POST /api/uploads", "DELETE /api/uploads/{id}",
              "GET /api/maps", "GET /api/maps/{name}"}
SH_BLOCK = re.compile(r"^ *```sh\n(.*?)^ *```$", re.S | re.M)
CODE_SPAN = re.compile(r"`([^`\n]+)`")
INVOCATION = re.compile(r'"\$REPO/([\w.-]+)"([^`\n;]*)')
FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")


def committed() -> str:
    return (REPO / "SKILL.md").read_text("utf-8")


def part(text: str, title: str) -> str:
    """The ``## title`` section of ``text``."""
    return text.split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]


def subsections(text: str) -> dict[str, str]:
    """The ``### `` sections of ``text`` by title."""
    return {title: body for title, _, body in
            (chunk.partition("\n") for chunk in text.split("\n### ")[1:])}


def script_sections(text: str) -> dict[str, str]:
    """Each script mode's section, by the mode's label."""
    return {t.strip("`"): b for t, b in subsections(part(text, "Scripts")).items()
            if t.startswith("`")}


def snippet(text: str) -> str:
    """The first ``sh <<'EOF'`` … ``EOF`` snippet of ``text``."""
    m = re.search(r"^sh <<'EOF'\n(.*?)^EOF$", text, re.S | re.M)
    assert m, "no snippet"
    return m.group(1)


def invocations(text: str) -> list[tuple[str, list[str]]]:
    """Every ``"$REPO/<script>" args…`` of ``text``: the script and its arguments."""
    return [(m.group(1), shlex.split(m.group(2))) for m in INVOCATION.finditer(text)]


def table_rows(section: str) -> dict[str, list[str]]:
    """The option table of a script section: flag → its cells."""
    rows = {}
    for line in section.splitlines():
        if line.startswith("| `"):
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
            rows[cells[0].split("`")[1].split()[0]] = cells
    return rows


def expected_default(m: spec.Mode, o: spec.Option) -> str:
    if o.kind is spec.Kind.ATTRS:
        return f"`{CloudAttrs().describe(m.scope())}`"
    if o.default is None or o.kind is spec.Kind.FLAG or o.name == m.selector:
        return ""
    return f"`{o.default:g}`" if isinstance(o.default, float) else f"`{o.default}`"


def by_label() -> dict[str, skill.Script]:
    return {c.label(m): (p, c, m) for p, c, m in skill.covered()}


def api_routes(tmp_path: Path) -> list[str]:
    """``METHOD /path`` of every route of the real app (HEAD aside), path parameters as in the
    OpenAPI document."""
    ws = Workspace(tmp_path / "data")
    ws.create()
    service = Service(ws, Runner(ws), url="http://0.0.0.0:0/")
    app = inspect.getclosurevars(create_app(service)).nonlocals["app"]  # inside the guard
    return sorted(f"{m} " + re.sub(r"\{(\w+):\w+\}", r"{\1}", r.path)
                  for r in app.routes for m in r.methods - {"HEAD"})


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
    assert "/api/openapi.json" in body and "**the running service's document wins**" in body


def test_the_description_states_every_capability_and_when_to_use_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §2.7 "Description": every script mode and API operation with what it produces, the
    need for the checkout (scripts) or a reachable server.sh (API), and the modes that need the
    inference server."""
    text = yaml.safe_load(committed().split("---\n", 2)[1])["description"]
    scripts = text.split("Scripts, ", 1)[1].split(". API, ", 1)[0]
    clauses = {c.split()[0].rstrip(":"): c for c in scripts.split(": ", 1)[1].split("; ")}
    assert set(clauses) == {p.prog for p in entry_points.scripts()}
    for p, c, m in skill.covered():
        clause = clauses[p.prog]
        for word in (c.name, c.option(m.selector).flag if m.selector else None):
            assert word is None or re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", clause)
        for o in m.outputs:  # each format it writes on stdout, a map it writes, a viewer
            if o.via in ("stdout", "browser") or o.format == "map":
                assert skill.FORMATS.get(o.format, o.format) in clause, (c.label(m), o.format)
    for op in skill.operations().values():
        assert re.search(rf"(?<![\w-]){re.escape(op.id)}(?![\w-])", text), op.id
    assert "checkout on this machine" in text and "server.sh reachable from this machine" in text
    stdout = {o.format for _, _, m in skill.covered() for o in m.outputs if o.via == "stdout"}
    assert stdout == {"json", "png", "ply"}  # each result format is glossed
    assert "JSON = OpenLABEL scene" in text and "PLY = point cloud" in text
    assert "PNG = depth image or segmented image" in text  # the names of the PNG results

    def labels(inference: str) -> list[str]:
        return [c.label(m) for p, c, m in skill.covered()
                if skill.is_command(p) and m.inference == inference]

    need = text.split("Inference server needed by ", 1)[1].split(". ", 1)[0]
    required, rest = need.split(", and by ", 1)
    sometimes, without = rest.split("; ", 1)
    assert required == skill.listing(labels("required"))
    assert sometimes == skill.listing([f"{c.label(m)} for {skill.condition(m)}"
                                       for _, c, m in skill.covered()
                                       if m.inference == "conditional"])
    assert sometimes == f"mapper.sh locate for maps over {constants.UPDATE_EXHAUSTIVE_MAX} keyframes"
    assert without == f"{skill.listing(labels('never'))} works without it"  # persisted maps
    assert labels("never") == ["view.sh -m"]
    assert "job" not in text.lower()
    monkeypatch.setattr(skill, "DESCRIPTION_MAX", 100)  # one that would not fit is refused
    with pytest.raises(ValueError, match="Agent Skills: at most 100"):
        skill.render()


# -- the scripts ------------------------------------------------------------------------------------


def test_every_script_and_mode_is_covered() -> None:
    """Spec §2.7 "Covers every script": each mode of each script (server.sh: --status only) with
    where it is, its options (defaults, allowed values), what it does, a ready-to-run command its
    own parser accepts for that mode, a sample result and its exit statuses."""
    sections = script_sections(committed())
    service_lifecycle = {c.label(m) for c in entry_points.WEB_SERVICE.commands for m in c.modes
                         if m.selector != "status"}
    every = {c.label(m) for p in entry_points.scripts() for c in p.commands for m in c.modes}
    assert set(sections) == every - service_lifecycle, f"{REGENERATE}"
    assert {p.prog for p in entry_points.scripts()} == {f.name for f in REPO.glob("*.sh")}
    for p, c, m in skill.covered():
        s = sections[c.label(m)]
        where = f'`"$REPO/{p.prog}"' + (f" {c.name}" if c.name else "") + "`: "
        assert s.lstrip().startswith(where), s[:80]
        rows = table_rows(s)
        assert set(rows) == {o.flag for o in skill.options_of(c, m)}, c.label(m)
        assert ("No options." in s) == (not rows)
        for o in skill.options_of(c, m):
            flag, value, default, meaning = rows[o.flag]
            assert o.help.replace("|", "\\|") in meaning and default == expected_default(m, o)
            assert ("(required)" in flag) == (o.required or o.name == m.selector)
            assert all(f"`{choice}`" in value for choice in o.choices or ())
        lead = s.split("\n\n", 1)[0]
        if m.lifecycle is not None:
            assert f"**{m.lifecycle.capitalize()} a server:**" in lead
            assert "never run it yourself" in lead
        else:
            assert ("**Needs the inference server**" in lead) == (m.inference != "never")
            assert ("**Writes the map**" in lead) == (skill.map_written(c, m) is not None)
            assert ("**Runs until interrupted**" in lead) == (skill.browser(m) is not None)
        (block,) = SH_BLOCK.findall(s)
        ((prog, argv),) = invocations(block)
        args = spec.build_parser(p, RaisingParser).parse_args(argv)
        assert prog == p.prog and getattr(args, "command", None) == c.name
        assert c.mode_of(args) is m, c.label(m)
        assert "\n→ " in s
        exits = re.search(r"Exit statuses on failure: (.*)", s)
        assert exits, c.label(m)
        for e in spec.errors_of(m):
            assert f"{e['exit_code']} (`{e['code']}`)" in exits.group(1)


def test_the_skill_offers_nothing_the_project_does_not_have(tmp_path: Path) -> None:
    """No script, subcommand, mode, option, value, route, operation or parameter that the
    project does not have: every command line is read by the script's own parser or checked
    against the app's routes and the operations' parameters."""
    text = committed()
    progs = {p.prog: p for p in entry_points.scripts()}
    covered = skill.covered()
    assert set(re.findall(r"[\w-]+\.sh\b", text)) <= set(progs)
    for block in SH_BLOCK.findall(text):  # ready-to-run: the script's own parser takes them
        for prog, argv in invocations(block):
            p = progs[prog]
            assert os.access(REPO / prog, os.X_OK)
            args = spec.build_parser(p, RaisingParser).parse_args(argv)
            cmd = p.command(getattr(args, "command", None))
            mode = cmd.mode_of(args)
            assert any(p is q and cmd is d and mode is n for q, d, n in covered), argv
            assert {t.split("=")[0] for t in argv if t.startswith("-")} <= {
                o.flag for o in cmd.mode_options(mode)}
    for span in CODE_SPAN.findall(text):  # named in passing: subcommands and options exist
        for prog, argv in invocations(span):
            p = progs[prog]
            cmd = p.command(argv[0]) if p.subcommands else p.commands[0]
            assert {t for t in argv if t.startswith("-")} <= {o.flag for o in cmd.options}
    labels = by_label()
    for label, s in script_sections(text).items():  # a script's section names its options only
        _, c, _ = labels[label]
        assert set(FLAG.findall(s)) <= {o.flag for o in c.options}, (label, FLAG.findall(s))

    routes = [r.split(" ", 1) for r in api_routes(tmp_path)]
    patterns = [(m, re.compile("^" + re.sub(r"\{\w+\}", "[^/]+", path) + "$"))
                for m, path in routes]

    def route(method: str, path: str) -> bool:
        return any(m == method and pat.match(path) for m, pat in patterns)

    for method, path in re.findall(r"`(GET|POST|PUT|PATCH|DELETE) (/[^`\s]*)", text):
        assert route(method, path.split("?")[0]), (method, path)
    ops = skill.operations()
    curls = [line for b in SH_BLOCK.findall(text) for line in b.splitlines()
             if line.strip().startswith("curl ") and "$BASE" in line]
    curls += [s for s in CODE_SPAN.findall(text) if "curl " in s and "$BASE" in s]
    assert len(curls) >= 3 + len(ops)
    for line in curls:
        words = shlex.split(line[line.index("curl "):])
        method = words[words.index("-X") + 1] if "-X" in words else "GET"
        path = next(w for w in words if w.startswith("$BASE/")).removeprefix("$BASE").split("?")[0]
        assert route(method, path), line
        if path.startswith("/api/ops/"):
            op = ops[path.split("/")[3]]
            body = json.loads(words[words.index("-d") + 1])
            names = {o.name: o for o in op.options}
            assert set(body) <= set(names), line
            for k, v in body.items():
                assert names[k].choices is None or v in names[k].choices
    table = re.findall(r"^\| `([\w-]+)` \| `([^`]+)` \|", part(text, "API"), re.M)
    assert table == [(op.id, op.label) for op in ops.values()]
    for gone in ("/api/jobs", "cancel", "resubmit", "/viewer", "oh-my-slam-api"):
        assert gone not in text, gone


# -- the API ------------------------------------------------------------------------------------------


def test_every_api_route_is_covered_and_the_document_is_not_repeated(tmp_path: Path) -> None:
    """Spec §2.7 "Points to the API description": every route (the app's, exactly the spec's),
    the operations, how to find the service, the order of calls, the error shape and curl
    examples — and none of the operations' parameters, which /api/openapi.json describes."""
    text = committed()
    api = part(text, "API")
    routes = api_routes(tmp_path)
    assert {r for r in routes if r.split(" ", 1)[1].startswith("/api/")} == API_ROUTES
    missing = [r for r in API_ROUTES if f"`{r}`" not in api]
    assert not missing, f"routes missing from SKILL.md ({REGENERATE}): {missing}"
    assert "`$BASE/api/openapi.json` describes every operation, parameter, result and error" in api
    for op in skill.operations().values():
        for o in op.options:
            assert o.help not in api, (op.id, o.flag)  # the document says it, not the skill
    assert "| Parameter" not in api and "| Option" not in api
    flow = subsections(api)["A request, call by call"]
    steps = [flow.index(s) for s in ("**Check the service", "**Upload", "**Validate",
                                     "**Run it and wait", "**Read the stage timings")]
    assert steps == sorted(steps)
    assert "with no client timeout" in flow and "-X POST -T " in flow and "curl -F" in flow
    assert "--max-time" not in flow and " -o result" in flow
    assert "grep -i '^server-timing:' headers.txt" in flow
    errors = part(text, "Errors")
    assert '{"error": {"code": "<code>", "message": "<the command\'s own message>"' in errors
    rules = part(text, "Rules")
    for rule in ("Never start or stop `server.sh` or the inference server",
                 "Never write into a map's folder", "Ask the user first",
                 "API inputs are uploads or paths inside the workspace",
                 "An\n  upload is consumed by the one request it is given to",
                 "with no client timeout", "Offer only what exists"):
        assert rule in rules, rule


def test_every_code_the_service_answers_with_is_described() -> None:
    """The service's own refusals (``_error``, ``RunError``) are in SKILL.md, besides the
    commands' exit codes, which come from the registry."""
    from oh_my_slam.web import runner

    app, run = (WEB / "app.py").read_text("utf-8"), (WEB / "runner.py").read_text("utf-8")
    codes = {c for m in re.finditer(r'_error\(\w+, "(\w+)"(?: if [^"]* else "(\w+)")?', app)
             for c in m.groups() if c}
    codes |= set(re.findall(r'RunError\(\d+,.*?, "(\w+)"\)', app + run, re.S))
    codes |= {runner.STOPPING}  # a request the service interrupted as it stopped
    assert {"forbidden", "unsupported_media_type", "too_large", "upload_in_use",
            "stopping"} <= codes
    text = committed()
    assert not [c for c in codes if f"`{c}`" not in text], codes
    assert set(skill.service_errors()) == codes - {error_code(c) for c in ExitCode} | {"not_found"}
    assert all(f"| {int(c)} | `{error_code(c)}` | {HTTP_STATUS[c]} |" in text for c in ExitCode)


def test_the_service_refusals_have_the_apps_statuses(tmp_path: Path) -> None:
    """The HTTP status the skill states for each of the service's own refusals is the one the app
    answers with."""
    ws = Workspace(tmp_path / "data")
    ws.create()
    service = Service(ws, Runner(ws), url="http://0.0.0.0:0/", extra_hosts={"testserver"},
                      max_upload_bytes=10)
    got: dict[str, int] = {}
    octet = {"content-type": "application/octet-stream"}
    with TestClient(create_app(service)) as client:
        for r in (client.get("/api/maps/nope"),
                  client.get("/api/maps", headers={"host": "elsewhere.example"}),
                  client.post("/api/ops/reconstruct/validate", content=b"{}",
                              headers={"content-type": "text/plain"}),
                  client.post("/api/uploads?name=a.jpg", content=b"x" * 100, headers=octet)):
            got[r.json()["error"]["code"]] = r.status_code
        service.max_upload_bytes, service.min_free_bytes = 1 << 40, 1 << 62
        r = client.post("/api/uploads?name=a.jpg", content=b"x", headers=octet)
        got[r.json()["error"]["code"]] = r.status_code
    service.runner.shutdown()
    app, run = (WEB / "app.py").read_text("utf-8"), (WEB / "runner.py").read_text("utf-8")
    m = re.search(r'_error\((\d+), "upload_in_use"', app)
    assert m and f"RunError({m.group(1)}," in run
    got["upload_in_use"] = int(m.group(1))
    status, body = web_app.failure(Outcome(130, None, STOPPING))
    assert body["error"]["code"] == STOPPING and f"_error({status}, STOPPING" in app
    got[STOPPING] = status
    assert got == {k: s for k, (s, _) in skill.service_errors().items()}


# -- the limits and numbers it states ----------------------------------------------------------------


def numbers_free(text: str, *allowed: str) -> str:
    """``text`` without its placeholders, network addresses, "3D" and the ``allowed`` patterns."""
    for pattern in (r"\{\{\w+\}\}", r"0\\?\.0\\?\.0\\?\.0", r"127\\?\.0\\?\.0\\?\.1", r"\b3D\b",
                    *allowed):
        text = re.sub(pattern, "", text)
    return text


def test_the_template_states_no_number_of_its_own() -> None:
    """Every number of SKILL.md comes from the code: the template holds none (but network
    addresses, a list's numbering and the snippets' shell syntax)."""
    for name in ("INTRO", "RULES", "CHECKOUT", "API", "FIND", "WORKFLOW", "ERRORS"):
        rest = numbers_free(getattr(skill, name), r"(?m)^\d\. ")
        assert not re.search(r"\d", rest), (name, re.findall(r".{20}\d.{20}", rest))
    for name in ("REPO_SNIPPET", "ADDRESS_SNIPPET"):
        rest = numbers_free(getattr(skill, name), r"exit [01]\b", r"return 1\b", r"2>/dev/null",
                            r">&2", r"\$1\b", r"\\1", r"0-9", r"# [123]\. ")
        assert not re.search(r"\d", rest), (name, re.findall(r".{20}\d.{20}", rest))


def test_every_limit_the_skill_states_is_the_codes() -> None:
    text = committed()
    sections = script_sections(text)
    stated = {}
    for _, c, m in skill.covered():  # each option's default (the table test checks them all)
        for o in skill.options_of(c, m):
            stated[o.name] = table_rows(sections[c.label(m)])[o.flag][2]
    assert stated["fps"] == f"`{constants.DEFAULT_FPS:g}`"
    assert stated["min_score"] == f"`{constants.DEFAULT_MIN_SCORE:g}`"
    assert stated["data"] == f"`{constants.DEFAULT_DATA}`"
    assert set(skill.CONDITION_WORDS) == set(spec.CONDITIONS)  # every condition has its words
    for _, c, m in skill.covered():
        for k, v in (m.inference_condition or {}).items():
            assert k != "map_keyframes_greater_than" or v == constants.UPDATE_EXHAUSTIVE_MAX
            assert skill.CONDITION_WORDS[k].format(v) in sections[c.label(m)]
            assert f"over {v} keyframes" in skill.condition(m)
    errors = part(text, "Errors")
    assert f"| `too_large` | 413 | an upload over {web_app.MAX_UPLOAD_BYTES / 2**30:g} GiB |" \
        in errors
    assert f"less than {web_app.MIN_FREE_BYTES / 2**30:g} GiB free" in errors
    rows = re.findall(r"^\| (\d+) \| `(\w+)` \| (\d+) \| (.*) \|$", errors, re.M)
    assert rows == [(str(int(c)), error_code(c), str(HTTP_STATUS[c]), MEANING[c]) for c in ExitCode]
    probe = re.search(r"--max-time (\d+) ", snippet(part(text, "API")))
    assert probe and int(probe.group(1)) == skill.PROBE_TIMEOUT_S
    assert f"answers there within {skill.PROBE_TIMEOUT_S} s" in part(text, "API")
    rules = " ".join(part(text, "Rules").split())
    down = ExitCode.SERVER_UNAVAILABLE
    assert f"fail with exit {int(down)}," in rules
    assert f"with HTTP {HTTP_STATUS[down]} `{error_code(down)}`" in rules
    # what works without it (persisted maps): the modes that never need it, and those that need
    # it only at times
    assert ("offer what works without it: the script `view.sh -m`; `mapper.sh locate` and "
            "`mapper-locate` except for maps over "
            f"{constants.UPDATE_EXHAUSTIVE_MAX} keyframes.") in rules
    (writer,) = [m for p, c, m in skill.covered() if skill.map_written(c, m) is not None]
    stages = [str(s) for s in writer.stages]
    assert f"run {len(stages)} stages, from `{stages[0]}` to `{stages[-1]}`" in rules


def test_a_changed_limit_changes_the_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """The limits are read from the code when the file is rendered: changing one changes it."""
    before = skill.render()
    monkeypatch.setattr(web_app, "MAX_UPLOAD_BYTES", 3 << 30)
    monkeypatch.setattr(skill, "PROBE_TIMEOUT_S", 5)
    update, locate = spec.MAPPER.command("update"), spec.MAPPER.command("locate")
    fps = dataclasses.replace(update.option("fps"), default=3.0)
    (locating,) = locate.modes
    (updating,) = update.modes
    mapper = replaced(replaced(spec.MAPPER, "update", options=tuple(
        fps if o.name == "fps" else o for o in update.options),
        modes=(dataclasses.replace(updating, stages=updating.stages[:-1]),)), "locate",
        modes=(dataclasses.replace(locating, inference_condition={
            "map_keyframes_greater_than": 300}),))
    with_programs(monkeypatch, mapper)
    text = skill.render()
    assert text != before
    assert "an upload over 3 GiB" in text and "an upload over 8 GiB" not in text
    assert "--max-time 5 " in text and "within 5 s" in text and "--max-time 3 " not in text
    rows = table_rows(script_sections(text)["mapper.sh update"])
    assert rows["-fps"][2] == "`3`"
    assert "maps over 300 keyframes" in text and "over 150" not in text
    assert f"run {len(updating.stages) - 1} stages" in text


# -- the samples ----------------------------------------------------------------------------------------


def test_the_samples_follow_the_code(tmp_path: Path) -> None:
    """The lines, records and layouts the skill shows are the code's."""
    text = committed()
    port = "8123"
    assert cli_view.URL_LINE.match(skill.listening(spec.VIEW).replace("<port>", port))
    assert f"`{skill.listening(spec.VIEW)}`" in text
    assert LINE.match(f"server.sh: listening on http://0.0.0.0:{port}/")
    assert "`server.sh: listening on http://0.0.0.0:<port>/`" in text
    assert set(json.loads(skill.inference_health_sample().replace("{…}", "{}"))) == set(
        Health.model_fields)
    ws = Workspace(tmp_path / "data")
    ws.create()
    service = Service(ws, Runner(ws), url="http://0.0.0.0:0/",
                      inference_health=lambda: {"status": "down"})
    real = service.health({"running": 0, "waiting": 0}, [])
    sample = skill.service_health({"status": "ready"})
    assert set(sample) == set(real) and set(sample["service"]) == set(real["service"])
    service.runner.shutdown()
    cub = openlabel.cuboid_val(np.array([1.0, 2.0, 3.0]), np.eye(3), np.array([4.0, 5.0, 6.0]))
    named = dict(zip(skill.CUBOID, cub, strict=True))
    assert (named["x"], named["y"], named["z"], named["qw"], named["sx"], named["sz"]) == (
        1, 2, 3, 1, 4, 6)
    assert f"`val` is `{','.join(skill.CUBOID)}`" in text
    # a PNG result is named by the outputs in that format, with the option that gives each
    assert ("PNG is the depth image (`reconstruct.sh` with `-f depth`) or the segmented image "
            "(`segment.sh -i` with `-f png`)") in text
    assert "(JSON, PNG, PLY)" in part(text, "Scripts")  # what the scripts write on stdout
    for block in SH_BLOCK.findall(text):  # the sample inputs are the checkout's own
        for _, argv in invocations(block):
            for arg in argv:
                if arg.startswith("$REPO/"):
                    assert glob.glob(str(REPO / arg.removeprefix("$REPO/"))), arg
    run_main = inspect.getsource(process.run_main)
    assert 'f"{prog}: error: {exc}"' in run_main and 'f"{prog}: interrupted"' in run_main
    assert "uv sync" in (REPO / "scripts" / "_common.sh").read_text()


def test_the_generator_renders_every_kind_of_option_and_output() -> None:
    """Kinds the registry may use that no command does yet: an option of an unknown kind or a
    required choice, an output for a video only, a PLY or another format as the result."""
    mood = spec.Option("--mood", "mood", slow_command.MOOD, "how it feels")  # type: ignore[arg-type]
    pick = spec.Option("--pick", "pick", spec.Kind.ENUM, "a choice", required=True,
                       choices=("a", "b"))
    count = spec.Option("--count", "count", spec.Kind.NUMBER, "how many", required=True)
    assert skill.value_text(mood) == slow_command.MOOD
    assert skill.sample_words(pick) == ["a"] and skill.sample_words(count) == ['"$COUNT"']
    update = spec.MAPPER.command("update")
    assert skill.when_text(update, (spec.When("inputs", video=True),)) == "for one video `-i`"
    assert skill.when_text(update, (spec.When("map"), spec.When("mode", ("full", "single")))) \
        == "with `-m` or with `-t full` or with `-t single`"
    (mode,) = spec.RECONSTRUCT.commands[0].modes
    ply = spec.Output("result", "stdout", "ply", "a cloud")
    assert "`format binary_little_endian 1.0`, `comment attributes color=rgb," in \
        skill.output_sample(spec.RECONSTRUCT, mode, ply)
    bare = dataclasses.replace(mode, attrs_scope=None)
    assert "comment attributes" not in skill.output_sample(spec.RECONSTRUCT, bare, ply)
    png = spec.Output("result", "stdout", "png", "an image")
    assert skill.output_sample(spec.RECONSTRUCT, mode, png) == "image/png bytes"
    cmd = spec.Command("pick.sh", None, "pick", (pick,), (dataclasses.replace(
        mode, rules=(), attrs_scope=None),))
    op = Operation(spec.Program("pick.sh", "pick", (cmd,)), cmd, cmd.modes[0])
    assert skill.sample_params(op) == {"pick": "a"}
    shot = dataclasses.replace(mode, outputs=(png,))  # a result in a format with no gloss
    assert skill.formats_text([(spec.RECONSTRUCT, spec.RECONSTRUCT.commands[0], shot)]) == \
        "Results: PNG is an image (`reconstruct.sh`)."
    assert skill.title(png) == "an image" and skill.title(spec.Output(
        "result", "stdout", "png", "the depth image: 16 bits")) == "the depth image"


def test_lists_read_as_sentences() -> None:
    assert [skill.listing(list(items)) for items in ("", "a", "ab", "abc")] == [
        "", "a", "a and b", "a, b and c"]
    assert skill.listed("script", ["`a`"]) == "the script `a`"
    assert skill.listed("operation", ["`a`", "`b`"]) == "the operations `a` and `b`"


def test_the_inference_need_reads_right_whatever_the_modes_need(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The description and the rules follow the modes' inference need: with no mode that needs
    it at times, none is named so; several modes that never need it are listed together."""
    locate = spec.MAPPER.command("locate")
    (mode,) = locate.modes
    mapper = replaced(spec.MAPPER, "locate", modes=(dataclasses.replace(
        mode, inference="never", inference_condition=None),))
    monkeypatch.setattr(spec, "PROGRAMS", tuple(mapper if p is spec.MAPPER else p
                                                for p in spec.PROGRAMS))  # in place
    text = skill.render()
    front = yaml.safe_load(text.split("---\n", 2)[1])["description"]
    assert ("Inference server needed by reconstruct.sh, mapper.sh update, segment.sh -i "
            "and view.sh -i; mapper.sh locate and view.sh -m work without it.") in front
    assert "offer what works without it: the scripts `mapper.sh locate` and `view.sh -m`; " \
        "the operation `mapper-locate`." in " ".join(part(text, "Rules").split())


def test_main_writes_the_skill(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "SKILL.md"
    assert skill.main([str(out)]) == 0 and out.read_text("utf-8") == committed()
    assert skill.main(["-h"]) == 0 and skill.main(["a", "b"]) == 2
    assert "usage: python -m oh_my_slam.web.skill" in capsys.readouterr().err
    res = subprocess.run([sys.executable, "-m", "oh_my_slam.web.skill", str(out)],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0 and res.stderr == f"wrote {out}\n"


# -- a registry change -----------------------------------------------------------------------------


class ShadeError(OhMySlamError):
    exit_code = ExitCode.MAP_LOCKED


def test_a_registry_change_reaches_the_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """An option added to a mode and one removed, a whole new command, and a new server mode
    change the skill with no edit to the generator."""
    locate = spec.MAPPER.command("locate")
    radius = spec.Option("--radius", "radius", spec.Kind.NUMBER, "search radius in metres",
                         default=1.5, minimum=0, type=float)
    (mode,) = locate.modes
    rule = spec.Rule("radius", ("radius",), "--radius suits the map", lambda ctx: None,
                     (ShadeError,))
    mapper = replaced(spec.MAPPER, "locate", options=(*(o for o in locate.options
                                                         if o.name != "mode"), radius),
                      modes=(dataclasses.replace(mode, rules=(*mode.rules, rule)),))
    with_programs(monkeypatch, mapper, slow_command.registry_program())  # type: ignore[arg-type]
    server = entry_points.INFERENCE_SERVER.commands[0]
    restart = spec.Option("--restart", "restart", spec.Kind.FLAG, "restart the server",
                          default=False, group="action")
    restarting = spec.Mode("restart", "restart", (), "never", "restarts the inference server",
                           (), (), lifecycle="starts")
    monkeypatch.setattr(entry_points, "INFERENCE_SERVER", replaced(
        entry_points.INFERENCE_SERVER, None, options=(*server.options, restart),
        modes=(*server.modes, restarting)))

    text = skill.render()
    assert text != committed()
    sections = script_sections(text)
    rows = table_rows(sections["mapper.sh locate"])
    assert rows["--radius"] == ["`--radius RADIUS`", "number ≥ 0", "`1.5`",
                                "search radius in metres"]
    assert "-t" not in rows and "`map_locked`" in sections["mapper.sh locate"]
    assert "6 (`map_locked`)" in sections["mapper.sh locate"]
    assert "slow.sh" in sections and "`\"$REPO/slow.sh\"`: Sleep." in sections["slow.sh"]
    assert "| `slow` | `slow.sh` |" in text
    front = text.split("---\n", 2)[1]
    assert "slow.sh → JSON + map" in front and ", slow run those modes" in front
    assert "; view.sh -m and slow.sh work without it." in front  # a new mode without inference
    assert "**Starts a server:** restarts the inference server" in \
        sections["start_inference_server.sh --restart"]
    assert "`start_inference_server.sh --restart`" in part(text, "Rules")


# -- the snippets ------------------------------------------------------------------------------------

SHELLS = [s for s in ("sh", "dash") if shutil.which(s)]


@pytest.mark.parametrize("shell", SHELLS)
def test_checkout_snippet_is_posix_and_finds_the_checkout_in_order(tmp_path: Path,
                                                                    shell: str) -> None:
    """User path → cached path while it holds the scripts → ask the user; the path found is
    cached."""
    code = snippet(part(committed(), "Scripts"))
    assert subprocess.run([shell, "-n"], input=code, text=True).returncode == 0
    home = tmp_path / "home"
    cache = tmp_path / "cache" / "oh-my-slam" / "repo"
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "XDG_CACHE_HOME": str(tmp_path / "cache")}

    def run(**extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([shell], input=code, env={**env, **extra}, capture_output=True,
                              text=True, timeout=60, cwd=tmp_path)

    none = run()
    assert none.returncode == 1 and none.stdout == "" and not cache.exists()
    assert "Ask the user" in none.stderr and "OMS_REPO=<path>" in none.stderr
    checkout = home / "oh my slam"  # a path with a space works
    checkout.mkdir(parents=True)
    for p in entry_points.scripts()[:-1]:
        (checkout / p.prog).touch()
    partial = run(OMS_REPO=str(checkout))  # not every script: not the checkout
    assert partial.returncode == 1 and "check the path with the user" in partial.stderr
    (checkout / entry_points.scripts()[-1].prog).touch()
    found = run(OMS_REPO="~/oh my slam/")  # 1. the user's path (~ expanded, no trailing /)
    assert (found.returncode, found.stdout) == (0, f"{checkout}\n"), found.stderr
    assert cache.read_text() == f"{checkout}\n"
    assert run().stdout == f"{checkout}\n"  # 2. the cache, while it holds the scripts
    assert run(OMS_REPO="home/oh my slam").stdout == f"{checkout}\n"  # relative to $PWD
    assert run(OMS_REPO=str(REPO)).stdout == f"{REPO}\n"  # the real checkout
    assert run(OMS_REPO=f"{tmp_path}/it's").returncode == 1  # a quote: never pasted
    (checkout / "view.sh").unlink()  # the cached path no longer holds the scripts: ask
    cache.write_text(f"{checkout}\n")
    gone = run()
    assert gone.returncode == 1 and "Ask the user" in gone.stderr


@pytest.mark.parametrize("shell", SHELLS)
def test_address_snippet_is_posix_and_finds_the_service_in_order(tmp_path: Path,
                                                                  shell: str) -> None:
    """User URL → cached URL while it answers → server.json of the workspace (0.0.0.0 →
    127.0.0.1) → ask the user; the URL found is cached, and nothing is scanned."""
    code = snippet(part(committed(), "API"))
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
