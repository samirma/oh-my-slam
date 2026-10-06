"""The agent skill ``SKILL.md`` at the repository root (spec §2.7, ``agent_skill.md``): one
self-contained file that lets an AI agent use every feature through both kinds of entry point — the
scripts of the checkout with ``sh``, and the ``server.sh`` API with ``curl``.

It is generated, never hand-edited, from the same shared definitions as the commands and
``/api/openapi.json``:

* every script and mode (``commands.entry_points.scripts()``: the commands of ``commands.spec`` and
  the two servers' own entry points) with its options, defaults, allowed values, outputs, inference
  need, effect and exit statuses. The modes that start or stop a server are covered for the
  inference server only, whose start command the agent reports (``app.START_COMMAND``): ``server.sh``
  is covered for ``--status`` alone;
* for the API, the service's ``/api/openapi.json`` is the source of every operation, parameter,
  result and error, so the API part adds only what that document cannot say — finding the service,
  the order of calls, the error shape, the rules and a few ``curl`` examples;
* the samples come from the code: the commands' parsers and their refusals, the inference server's
  error, the health records, the upload record (``workspace.Upload``), the OpenLABEL builders, a
  failed command's answer (``app.failure``) and the ``Server-Timing`` of a timing record
  (``app.timing_header``). Limits are read from their modules when the file is rendered.

What concerns no single command (finding the checkout and the service, the request workflow, the
rules) is the template below. ``tests/unit/test_agent_skill.py`` fails when the committed file
differs from this output, misses a script mode, option or API route, offers one the project does
not support, or states a limit that differs from the code.

Regenerate: ``uv run python -m oh_my_slam.web.skill [PATH]`` (default ``SKILL.md`` at the
repository root)."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, get_args

import numpy as np

from oh_my_slam.client.protocol import Health, ServerStatus
from oh_my_slam.commands import entry_points, spec
from oh_my_slam.commands.parser import ParameterError, RaisingParser
from oh_my_slam.commands.spec import Command, Kind, Mode, Option, Output, Program
from oh_my_slam.core import constants, ply
from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.errors import (
    HTTP_STATUS,
    MEANING,
    ExitCode,
    ServerUnavailableError,
    error_code,
)
from oh_my_slam.schema import openlabel
from oh_my_slam.web import app, openapi, workspace
from oh_my_slam.web import main as service
from oh_my_slam.web.operations import (
    RESULT,
    Operation,
    error_body,
    operations,
    result_format,
)
from oh_my_slam.web.runner import STOPPING, Outcome

Json = dict[str, Any]
Script = tuple[Program, Command, Mode]
NAME = "oh-my-slam"
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "SKILL.md"
CACHE_DIR = "${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam"
REPO_CACHE = f"{CACHE_DIR}/repo"  # the checkout's path
URL_CACHE = f"{CACHE_DIR}/server_url"  # the service's base URL
DESCRIPTION_MAX = 1024  # the Agent Skills limit
PROBE_TIMEOUT_S = 3  # the address snippet: a service answers /api/health within this
CREATED = 1791281700.0  # the samples' clock
JSON_POST = "-H 'Content-Type: application/json'"
HEADERS = "headers.txt"  # where the workflow saves a response's headers (-D)
VIEWER_HOST = "127.0.0.1"  # view.sh binds the loopback address
CURL_EXITS = (7, 28)  # curl's exit statuses for "could not connect" and "timed out"
HOME_DATA = constants.DEFAULT_DATA.replace("~", "$HOME", 1)  # the default workspace, in sh

# The scripts' sample inputs by option kind: the checkout's reference inputs (examples/) and a map
# of the default workspace (interchangeable with server.sh's), so a sample command runs as it is
# once $REPO is set. Another kind gets a shell variable named after the option.
SAMPLE_IN: dict[Kind, str] = {
    Kind.IMAGE: '"$REPO/examples/restaurant.jpg"',
    Kind.IMAGES: '"$REPO"/examples/office_sequence/*.jpg',
    Kind.IMAGES_OR_VIDEO: '"$REPO"/examples/office_sequence/*.jpg',
    Kind.MAP: f'"{HOME_DATA}/{workspace.MAPS}/office"',
}
# The description's words for the inputs and what a mode produces (another is named as it is).
INPUTS = {Kind.IMAGE: "image", Kind.IMAGES: "images", Kind.IMAGES_OR_VIDEO: "images/video",
          Kind.MAP: "map"}
FORMATS = {"json": "JSON", "ply": "PLY", "png": "PNG", "map": "map", "html": "web viewer"}
# The description's gloss of a result format; another is glossed by the names of the results in
# it (``title``: "PNG = depth image or segmented image").
GLOSS = {"json": "OpenLABEL scene (labelled objects, oriented bounding boxes)",
         "ply": "point cloud"}
# Each inference condition of the registry (``spec.CONDITIONS``) in words, its value for ``{}``.
CONDITION_WORDS = {"map_keyframes_greater_than": "maps over {} keyframes"}
# The 10 values of an OBB cuboid as schema.openlabel writes them (cuboid_val: quaternion scalar
# last); tests/unit/test_agent_skill.py checks them against the builder.
CUBOID = ("x", "y", "z", "qx", "qy", "qz", "qw", "sx", "sy", "sz")


def gib(n: int) -> str:
    return f"{n / 2**30:g} GiB"


def num(x: Any) -> str:
    """A default or bound as the commands' help writes it (``2`` for 2.0)."""
    return f"{x:g}" if isinstance(x, int | float) and not isinstance(x, bool) else str(x)


def fill(text: str, **values: str) -> str:
    """``text`` with each ``{{key}}`` replaced (JSON and shell keep their single braces)."""
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    assert "{{" not in text, text[text.index("{{"):][:40]
    return text


def compact(obj: Any) -> str:
    """JSON as the service writes it."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def code(text: str) -> str:
    return f"`{text}`"


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def sh(*lines: str) -> str:
    return "```sh\n" + "\n".join(lines) + "\n```"


def sentence(text: str) -> str:
    """``text`` with a capital first letter and no final full stop."""
    return text[:1].upper() + text[1:].rstrip(".")


def statuses() -> str:
    return ", ".join(code(s) for s in ("down", *get_args(ServerStatus)))


def listing(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return " and ".join([", ".join(items[:-1]), items[-1]] if len(items) > 2 else items)


def listed(noun: str, items: list[str]) -> str:
    """``the script a``, ``the scripts a and b``."""
    return f"the {noun}{'s' if len(items) > 1 else ''} {listing(items)}"


def condition(mode: Mode) -> str:
    """When a "conditional" mode needs the inference server, in words (any of its conditions)."""
    return " or ".join(CONDITION_WORDS[k].format(v)
                       for k, v in (mode.inference_condition or {}).items())


def title(out: Output) -> str:
    """An output's name: its text up to the first colon (``the depth image: one 16-bit …``)."""
    return out.text.split(":", 1)[0]


# -- the scripts: everything here comes from the commands' and the servers' definitions -----------


def reported(p: Program) -> bool:
    """Whether the agent reports ``p``'s start command to the user: the inference server's, which
    its errors and the service's health name."""
    return PurePosixPath(app.START_COMMAND).name == p.prog


def covered() -> list[Script]:
    """Every script mode the skill covers (spec §2.7 "Covers every script"): all, but the modes
    that start or stop a server only for the server whose start command the agent reports."""
    return [(p, c, m) for p in entry_points.scripts() for c in p.commands for m in c.modes
            if m.lifecycle is None or reported(p)]


def label(c: Command, m: Mode) -> str:
    return c.label(m)


def options_of(c: Command, m: Mode) -> list[Option]:
    """The mode's options as the skill lists them: a flag that selects the mode is in its label."""
    return [o for o in c.mode_options(m) if not (o.name == m.selector and o.kind is Kind.FLAG)]


def shown_flag(o: Option) -> str:
    if o.kind in (Kind.FLAG, Kind.ENUM):
        return o.flag
    return f"{o.flag} {o.metavar or o.name.upper()}" + ("…" if o.multiple else "")


def value_text(o: Option) -> str:
    """What an option's value is, with its allowed values."""
    if o.kind is Kind.FLAG:
        return "flag"
    if o.kind is Kind.ENUM:
        return ", ".join(code(c) for c in o.choices or ())
    if o.kind is Kind.NUMBER:
        what = "integer" if o.type is int else "finite number" if o.finite else "number"
        return " ".join([what] + ([f"≥ {num(o.minimum)}"] if o.minimum is not None else [])
                        + ([f"> {num(o.exclusive_minimum)}"] if o.exclusive_minimum is not None
                           else []))
    if o.kind is Kind.ATTRS:
        return "`key=value,…`" + (", repeatable" if o.repeatable else "")
    if o.kind is Kind.MAP:
        return "map folder: " + ("an existing map" if o.must_exist else
                                 "a map, or a new or empty folder")
    if o.kind is Kind.FILE_OUT:
        return "file"
    if o.kind is Kind.FOLDER_OUT:
        return "folder"
    if o.kind in spec.ACCEPTS:
        return ("files" if o.multiple else "file") + (", in order" if o.ordered else "") + ": " \
            + " ".join(sorted(spec.ACCEPTS[o.kind]))
    return str(o.kind)


def default_text(m: Mode, o: Option) -> str:
    if o.kind is Kind.ATTRS:  # the keys that apply to the mode's cloud, with their defaults
        return code(CloudAttrs().describe(m.scope()))
    if o.default is None or o.kind is Kind.FLAG or o.name == m.selector:
        return ""
    return code(num(o.default))


def options_table(c: Command, m: Mode) -> list[str]:
    opts = options_of(c, m)
    if not opts:
        return ["No options."]
    rows = ["| Option | Value | Default | Meaning |", "|---|---|---|---|"]
    for o in opts:
        required = " (required)" if o.required or o.name == m.selector else ""
        meaning = o.help + (f" ({o.applies_text})" if o.applies_text else "")
        rows.append(f"| {code(shown_flag(o))}{required} | {cell(value_text(o))} "
                    f"| {cell(default_text(m, o))} | {cell(meaning)} |")
    return rows


def has(c: Command, m: Mode, kind: Kind) -> Option | None:
    return next((o for o in c.mode_options(m) if o.kind is kind), None)


def map_written(c: Command, m: Mode) -> Option | None:
    """The map option a mode writes through (an output in the map format written ``via`` it)."""
    flags = {o.via for o in m.outputs if o.format == "map"}
    return next((o for o in c.mode_options(m) if o.kind is Kind.MAP and o.flag in flags), None)


def browser(m: Mode) -> Output | None:
    return next((o for o in m.outputs if o.via == "browser"), None)


def effect_text(c: Command, m: Mode) -> str:
    """What a mode changes: a map, a file, nothing; or it runs until interrupted."""
    if m.lifecycle is not None:
        return "The user runs it: never run it yourself; give the user the command."
    writes, page = map_written(c, m), browser(m)
    if writes is not None:
        return (f"**Writes the map** `{writes.flag}`: ask the user first (see "
                "[Rules](#rules)).")
    if page is not None:
        return ("**Runs until interrupted** (Ctrl-C): start it in the background and give the "
                "user the URL it prints.")
    names = [code(o.flag) for o in written(c, m)]
    return "Read-only" + (f"; it writes only to {' and '.join(names)}." if names else ".")


def written(c: Command, m: Mode) -> list[Option]:
    """The options naming where a mode writes its outputs: ``-o`` (the result) and a folder that
    outputs are written to (``-d``)."""
    vias = {o.via for o in m.outputs}
    return [o for o in c.mode_options(m)
            if (o.kind is Kind.FILE_OUT and RESULT in vias)
            or (o.kind is Kind.FOLDER_OUT and o.flag in vias)]


def does(c: Command, m: Mode) -> str:
    """What a mode does: the server it starts or stops, or its need of the inference server."""
    text = m.inference_text
    if m.lifecycle is not None:
        lead = f"**{m.lifecycle.capitalize()} a server:** {text}."
    elif m.inference == "required":
        lead = f"**Needs the inference server** ({text})."
    elif m.inference == "conditional":
        lead = f"**Needs the inference server** {text} ({condition(m)})."
    else:
        lead = f"Works without the inference server ({text})."
    return f"{lead} {effect_text(c, m)}"


def when_text(c: Command, whens: tuple[spec.When, ...]) -> str:
    def one(w: spec.When) -> str:
        flag = c.option(w.option).flag
        if w.video:
            return f"for one video `{flag}`"
        if w.values:
            return " or ".join(f"with `{flag} {v}`" for v in w.values)
        return f"with `{flag}`"

    return " or ".join(one(w) for w in whens)


def results_text(c: Command, m: Mode) -> str:
    """Where a mode's outputs go, each with its condition."""
    flags = {o.flag: o for o in c.mode_options(m)}
    parts = []
    stdout = [o for o in m.outputs if o.via == RESULT]
    if stdout:
        where = "on stdout, or in the `-o` file" if has(c, m, Kind.FILE_OUT) else "on stdout"
        parts.append(f"Result {where}: " + "; ".join(
            (f"{when_text(c, o.when)}, " if o.when else "") + o.text for o in stdout) + ".")
    for o in m.outputs:
        option = flags.get(o.via)
        if option is not None and o.format == "map":
            parts.append(f"The map `{o.via}`: {o.text}.")
    files = [o for o in m.outputs if o.via in flags and flags[o.via].kind is Kind.FOLDER_OUT]
    if files:
        folder = flags[files[0].via]
        parts.append(f"With `{shown_flag(folder)}` it also writes there: " + "; ".join(
            f"`{o.name}`, {o.text}" for o in files) + ".")
    page = browser(m)
    if page is not None:
        parts.append(f"It serves {page.text}.")
    return " ".join(parts)


def sample_words(o: Option) -> list[str]:
    if o.kind in SAMPLE_IN:
        return [SAMPLE_IN[o.kind]]
    if o.choices:
        return [o.choices[0]]
    return [f'"${o.name.upper()}"']


def parsed(p: Program, words: list[str]) -> argparse.Namespace:
    """``words`` (shell words) parsed by the script's own parser."""
    return spec.build_parser(p, RaisingParser).parse_args(shlex.split(" ".join(words)))


def result_of(m: Mode, args: argparse.Namespace) -> Output | None:
    """The mode's stdout output for these arguments (its condition holds)."""
    return next((o for o in m.outputs if o.via == RESULT
                 and (not o.when or any(w.holds(args) for w in o.when))), None)


def sample_command(p: Program, c: Command, m: Mode) -> tuple[list[str], Output | None, str | None]:
    """A ready-to-run command's words after the script (its required options with the sample
    inputs, and ``-o`` when it has one), the result it gives and the file that holds it."""
    words = [c.name] if c.name else []
    for o in c.mode_options(m):
        if o.name == m.selector and o.kind is Kind.FLAG:
            words.append(o.flag)
        elif o.required or o.name == m.selector:
            words += [o.flag, *sample_words(o)]
    result = result_of(m, parsed(p, words))
    out, saved = has(c, m, Kind.FILE_OUT), None
    if out is not None and result is not None:
        saved = "result" + spec.suffix_of(result.format)
        words += [out.flag, f'"$PWD/{saved}"']
        parsed(p, words)  # the whole command parses
    return words, result, saved


def scene_sample(full: bool = False) -> str:
    """An OpenLABEL scene as the commands write it, cut short: its metadata and one object built
    by the shared builders (``schema.openlabel``); ``full``: the object's whole entry."""
    obj = openlabel.object_entry(
        "chair 1", "chair", "camera",
        openlabel.cuboid(np.array([0.41, 0.18, 2.35]), np.eye(3), np.array([0.48, 0.51, 0.92]),
                         "camera"),
        nums=[openlabel.num("score", 0.91)])
    if not full:
        obj = {k: obj[k] for k in ("name", "type")}
    doc = {"openlabel": {"metadata": {"schema_version": openlabel.SCHEMA_VERSION, "…": 0},
                         "objects": {"1": {**obj, "…": 0}, "…": 0}, "…": 0}}
    return compact(doc).replace(',"…":0', ",…")


def inference_health_sample() -> str:
    """``start_inference_server.sh --status``: the inference server's ``/health`` record."""
    h = Health(status="ready", device="mps", precision="fp16", pid=4321, uptime_s=812.5)
    return compact(h.model_dump()).replace('"models":{}', '"models":{…}').replace(
        '"versions":{}', '"versions":{…}')


def service_health(inference: Json) -> Json:
    """The service's health record (``GET /api/health``, ``server.sh --status``), sample values."""
    return {"status": "ok", "service": {
        "version": "<version>", "url": "http://0.0.0.0:52026/", "pid": 4321,
        "workspace": PurePosixPath(constants.DEFAULT_DATA).name,
        "data": constants.DEFAULT_DATA.replace("~", "/Users/<user>", 1),
        "started_at": CREATED, "requests": {"running": 0, "waiting": 0}, "in_progress": []},
        "inference": inference}


def service_health_sample() -> str:
    return compact(service_health({"status": "ready", "health": {"…": 0},
                                   "start_command": None})).replace('{"…":0}', "{…}")


def json_sample(p: Program) -> str:
    """A JSON result: a server's health for a server's ``--status``, else a scene."""
    return {entry_points.INFERENCE_SERVER.prog: inference_health_sample,
            entry_points.WEB_SERVICE.prog: service_health_sample}.get(p.prog, scene_sample)()


def output_sample(p: Program, m: Mode, out: Output) -> str:
    if out.format == "json":
        return code(json_sample(p))
    if out.format == "ply":
        attrs = f", `comment attributes {CloudAttrs().describe(m.scope())}`" \
            if m.attrs_scope is not None else ""
        return f"`ply`, `format {ply.BINARY}`{attrs}, …, `end_header`, then the points"
    return f"{spec.media_type(out.format)} bytes"


def listening(p: Program) -> str:
    return f"{p.prog}: listening on http://{VIEWER_HOST}:<port>/"


def sample_result(p: Program, m: Mode, result: Output | None, saved: str | None) -> str:
    if result is not None:
        where = f"`{saved}` (stdout stays empty)" if saved else "stdout"
        return f"→ {where}: {output_sample(p, m, result)}"
    if browser(m) is not None:
        return (f"→ stderr: `{listening(p)}`, and the browser opens on it (not with "
                "`--no-browser`); nothing on stdout")
    return "→ nothing on stdout; it says on stderr what it did"


def failure_sample(p: Program, c: Command, m: Mode) -> str | None:
    """A one-line failure of the mode, from the code: the inference server's error for a mode that
    needs it, else the script's parser refusing a missing option."""
    if m.inference != "never":
        return f"{p.prog}: error: {ServerUnavailableError()}"
    words = [c.name] if c.name else []
    if m.selector is not None and c.option(m.selector).kind is Kind.FLAG:
        words.append(c.option(m.selector).flag)
    try:
        parsed(p, words)
    except ParameterError as exc:
        return f"{p.prog}: error: {exc}"
    return None


def exit_line(p: Program, c: Command, m: Mode) -> str:
    codes = ", ".join(f"{e['exit_code']} (`{e['code']}`)" for e in spec.errors_of(m))
    sample = failure_sample(p, c, m)
    return (f"Exit statuses on failure: {codes}, or another of [Errors](#errors)"
            + (f"; e.g. `{sample}`" if sample else "") + ".")


def script_section(p: Program, c: Command, m: Mode) -> list[str]:
    words, result, saved = sample_command(p, c, m)
    where = f'"$REPO/{p.prog}"' + (f" {c.name}" if c.name else "")
    text = results_text(c, m)
    return [f"### `{label(c, m)}`", "",
            f"`{where}`: {sentence(c.help)}. {does(c, m)}", "",
            *options_table(c, m), "", *([text, ""] if text else []),
            sh(" ".join([f'"$REPO/{p.prog}"', *words])), "",
            sample_result(p, m, result, saved), "", exit_line(p, c, m), ""]


# -- the API: what /api/openapi.json cannot say -----------------------------------------------------


def service_errors() -> dict[str, tuple[int, str]]:
    """The service's own refusals (no command's): code → (HTTP status, when). Their shape is that
    of every error; tests/unit/test_agent_skill.py checks that each code oh_my_slam.web answers
    with is here or in the commands' exit-code table."""
    return {
        "not_found": (404, "no such map, upload or operation"),
        "forbidden": (403, "a `Host` that does not name the service's machine, or a foreign "
                           "`Origin`"),
        "unsupported_media_type": (415, "a POST whose body is not `application/json`, or an "
                                        "upload sent as a form (`curl -F`)"),
        "too_large": (413, f"an upload over {gib(app.MAX_UPLOAD_BYTES)}"),
        "insufficient_storage": (413, "an upload that would leave less than "
                                      f"{gib(app.MIN_FREE_BYTES)} free on the workspace's disk"),
        "upload_in_use": (409, "an upload that another request in progress was given"),
        STOPPING: (503, "a request while the service stops, or one whose command it interrupted "
                        "when it stopped"),
    }


def sample_file(op: Operation, o: Option) -> str:
    """A sample upload of a suffix the option accepts (a video where it may be one)."""
    accepts = sorted(spec.ACCEPTS.get(o.kind, ())) or [".jpg"]
    stem, prefer = {Kind.IMAGES_OR_VIDEO: ("video", ".mp4"),
                    Kind.IMAGES: ("query", ".jpg")}.get(o.kind, ("photo", ".jpg"))
    return stem + (prefer if prefer in accepts else accepts[0])


def sample_params(op: Operation) -> Json:
    """The required parameters of an operation with placeholder values."""
    out: Json = {}
    for o in op.options:
        if not (o.required or o.name == op.mode.selector):
            continue
        if o.kind is Kind.MAP:
            out[o.name] = "<map>"
        elif o.kind in spec.ACCEPTS:
            path = f"{workspace.UPLOADS}/<upload>/{sample_file(op, o)}"
            out[o.name] = [path] if o.multiple else path
        else:
            out[o.name] = o.choices[0] if o.choices else o.default
    return out


def command_of(op: Operation, params: Json) -> list[str]:
    """The command line a request runs, as validate shows it (a map as ``maps/<name>``)."""
    shown = {k: (f"{workspace.MAPS}/{v}" if op.command.option(k).kind is Kind.MAP else v)
             for k, v in params.items()}
    return [op.program.prog, *spec.argv_of(op.command, op.mode, shown)]


def result_file(op: Operation, params: Json) -> str:
    return "result" + spec.suffix_of(result_format(op, params) or "")


def validated(op: Operation, params: Json) -> str:
    return compact({"valid": True, "command": command_of(op, params),
                    "inference": op.mode.inference == "required", "problems": [],
                    "by_parameter": {}})


def refusal(op: Operation) -> str:
    """What a request of ``op`` with an empty body answers: the refusal of the command's own
    parser, with its status (``problems`` repeats every problem, shortened here)."""
    status, body = error_body(spec.dry_run(op.command, op.mode, {}))
    err = ",".join(f"{json.dumps(k)}:" + ("[…]" if k == "problems" else compact(v))
                   for k, v in body["error"].items())
    return f"{status} `{{\"error\":{{{err}}}}}`"


def server_timing(stages: list[str]) -> str:
    """The ``Server-Timing`` header of a run that took these stages (sample figures)."""
    return app.timing_header({"stages_s": dict.fromkeys(stages, 0.5),
                              "total_s": 0.5 * len(stages)})


def failed(code_: ExitCode) -> str:
    """The answer to a request whose command ended with ``code_`` (``app.failure``)."""
    status, body = app.failure(Outcome(int(code_), "<the command's message>"))
    return f"{status} `{compact(body)}`"


def run_curl(op: Operation, params: Json) -> str:
    return (f"curl -sS -X POST -o {result_file(op, params)} -D {HEADERS} "
            f"-w '%{{http_code}}\\n' \"$BASE/api/ops/{op.id}\" {JSON_POST} -d '{compact(params)}'")


def inference_cell(op: Operation) -> str:
    m = op.mode
    need = {"required": "needed", "never": "not needed"}.get(
        m.inference, f"only for {condition(m)}")
    return need + ("; **writes the map**" if op.writes_map() is not None else "")


def routes(doc: Json) -> list[str]:
    """Every route of the API, with what it is for: the operations' two, then the service's own
    from the OpenAPI document (and the document itself)."""
    lines = ["* `POST /api/ops/{op}/validate`: check a request with the command's own checks "
             "(and the inference server's, when it needs it); nothing runs",
             "* `POST /api/ops/{op}`: run an operation within the request and answer when its "
             "command ends",
             "* `GET /api/openapi.json`: the description of every operation, parameter, result "
             "and error"]
    for path, item in doc["paths"].items():
        if path.startswith("/api/ops/"):
            continue
        for method, op in item.items():
            query = [p["name"] for p in op.get("parameters", []) if p["in"] == "query"]
            lines.append(f"* `{method.upper()} {path}`"
                         + (f" (query `{'`, `'.join(query)}`)" if query else "")
                         + f": {op.get('summary', '')}")
    return lines


def success(doc: Json, path: str, method: str) -> str:
    """The success status the OpenAPI document gives a route."""
    return next(s for s in doc["paths"][path][method]["responses"] if s.startswith("2"))


def example(ops: dict[str, Operation]) -> Operation:
    """The workflow's example: an operation on one image (else the first)."""
    return next((op for op in ops.values() if any(o.kind is Kind.IMAGE for o in op.options)),
                next(iter(ops.values())))


# -- the description ----------------------------------------------------------------------------------


def produces(c: Command, m: Mode) -> str:
    """A mode's inputs and what it produces, in the description's words."""
    ins = [INPUTS.get(o.kind, str(o.kind)) for o in c.mode_options(m)
           if (o.required or o.name == m.selector) and o.kind in INPUTS]
    out = list(dict.fromkeys(FORMATS.get(o.format, o.format) for o in m.outputs
                             if o.via in (RESULT, "browser") or o.format == "map"))
    files = [o for o in m.outputs if o.via in {x.flag for x in c.mode_options(m)
                                                if x.kind is Kind.FOLDER_OUT}]
    products = "/".join(o for o in out if o != "map") + (" + map" if "map" in out else "") \
        + (" + files" if files else "")
    return (": " + " + ".join(ins) if ins else "") + (f" → {products}" if products else "")


def mode_word(c: Command, m: Mode) -> str:
    """A mode as the description names it after its script: subcommand and selector, or the
    name of a server's mode that has no selector (its start)."""
    selector = c.option(m.selector).flag if m.selector else (m.name if m.lifecycle else None)
    return " ".join(x for x in (c.name, selector) if x)


def is_command(p: Program) -> bool:
    """One of the commands (``spec.PROGRAMS``), not a server's own entry point."""
    return any(p is q for q in spec.PROGRAMS)


def grouped(scripts: list[Script]) -> list[list[Script]]:
    """The script modes, by script."""
    by_prog: dict[str, list[Script]] = {}
    for s in scripts:
        by_prog.setdefault(s[0].prog, []).append(s)
    return list(by_prog.values())


def description(scripts: list[Script], ops: dict[str, Operation]) -> str:
    """What the skill can do and when to use it (spec §2.7 "Description"): every script mode and
    API operation with what it produces, the conditions (the checkout, a reachable server.sh) and
    the modes that need the inference server, within the Agent Skills limit."""
    def clause(group: list[Script]) -> str:
        body = ", ".join(mode_word(c, m) + produces(c, m) for _, c, m in group)
        return group[0][0].prog + ("" if body.startswith((":", " ")) else " ") + body

    def gloss(fmt: str) -> str:
        names = dict.fromkeys(title(o).removeprefix("the ") for _, _, m in scripts
                              for o in m.outputs if o.via == RESULT and o.format == fmt)
        return f"{FORMATS.get(fmt, fmt)} = {GLOSS.get(fmt) or ' or '.join(names)}"

    def commands(inference: str) -> list[Script]:
        return [s for s in scripts if is_command(s[0]) and s[2].inference == inference]

    need = [label(c, m) for _, c, m in commands("required")]
    sometimes = [f"{label(c, m)} for {condition(m)}" for _, c, m in commands("conditional")]
    never = [label(c, m) for _, c, m in commands("never")]
    text = ("oh-my-slam monocular RGB 3D mapping on a Mac. Scripts, with its checkout on this "
            "machine: " + "; ".join(clause(g) for g in grouped(scripts))
            + ". API, with curl and a server.sh reachable from this machine (LAN too): "
            "operations " + ", ".join(ops) + " run those modes; also uploads, validation, maps, "
            "health. " + ", ".join(map(gloss, result_formats(scripts)))
            + ". Inference server needed by " + listing(need)
            + (f", and by {listing(sometimes)}" if sometimes else "")
            + (f"; {listing(never)} work{'s' if len(never) == 1 else ''} without it"
               if never else "")
            + ". Use it to reconstruct, map, locate, segment or view images/video in 3D.")
    if len(text) > DESCRIPTION_MAX:
        raise ValueError(f"the description has {len(text)} characters (Agent Skills: at most "
                         f"{DESCRIPTION_MAX}); shorten the template or the words of "
                         "skill.INPUTS / FORMATS / GLOSS / CONDITION_WORDS")
    return text


def front_matter(scripts: list[Script], ops: dict[str, Operation]) -> str:
    """The skill's name and description (a YAML double-quoted scalar: JSON is valid YAML)."""
    return (f"---\nname: {NAME}\ndescription: "
            f"{json.dumps(description(scripts, ops), ensure_ascii=False)}\n---\n")


# -- the template ------------------------------------------------------------------------------------

INTRO = """# oh-my-slam

oh-my-slam turns RGB images and video into 3D on a Mac: scene descriptions of labelled objects
with oriented bounding boxes, point clouds, and persistent maps in which images can be located.
Two kinds of entry point give the same results, byte for byte:

* **[Scripts](#scripts)**: the shell scripts at the root of the oh-my-slam checkout, on the Mac
  that holds it.
* **[API](#api)**: `server.sh`, a web service that runs every mode of {{service_progs}} within an
  HTTP request, for this Mac or any machine on the LAN, with `curl`.

Use the scripts when the checkout is on this machine, else the API, and follow the
[Rules](#rules). This file is generated from the project's own definitions (`uv run python -m
oh_my_slam.web.skill` in the checkout): offer only what it, or the running service's
`/api/openapi.json`, lists.

| Script | Modes | What it does |
|---|---|---|
{{table}}

{{formats}}
"""

RULES = """## Rules

* **The inference server may be down.** Then the scripts {{need_scripts}} fail with exit
  {{down_exit}}, and the operations {{need_ops}} with HTTP {{down_http}} `{{down_code}}`, each with
  a message that names the start command `{{start}}` (`GET /api/health` gives it as
  `inference.start_command`). Report the message and that command to the user instead of
  retrying, and offer what works without it: {{without}}.
* **Never start or stop `server.sh` or the inference server**, and never kill their processes:
  the user runs {{lifecycle}}. Only {{status}} are yours to run.
* **Never write into a map's folder** (`<data>/{{maps}}/<name>/`, or any folder a script takes as a
  map): maps change only through {{writers}}.
* **Ask the user first** before updating an existing map, and before starting a long mapping
  request: {{writers}} run {{writer_stages}} and can take many minutes. Say that an update changes
  the map for good.
* **API inputs are uploads or paths inside the workspace** (relative to it, such as
  `{{uploads}}/<upload>/photo.jpg`), never paths outside it, which are refused ({{outside}}). **An
  upload is consumed by the one request it is given to**, whatever its outcome, so repeating a
  request means uploading again; validating does not consume it.
* **Wait for an API answer with no client timeout** (no `--max-time`, no `-m`): disconnecting
  interrupts the command as Ctrl-C would, and an interrupted map update leaves the map as it was.
  Requests that use the inference server run one at a time, in arrival order: send them one by
  one.
* **Results are the commands' own bytes:** save them with `-o` (script or `curl`) and never
  rewrite them; report a failure's message as it is. Don't repeat a refused or failed request
  unchanged.
* **Offer only what exists:** the scripts, modes, options and values below, and the operations,
  parameters and routes of `/api/openapi.json`.
"""

CHECKOUT = """## Scripts

### Find the checkout

The scripts are at the root of the oh-my-slam checkout. This snippet prints the checkout's
absolute path and caches it in `{{cache}}`. It tries, in order: the path the user gives
(`OMS_REPO`); the cached path, if it still holds the scripts. Otherwise it fails: ask the user for
the path of the checkout (the folder that holds `reconstruct.sh`) and run it again with the first
line changed to `OMS_REPO=<path> sh <<'EOF'`.

```sh
sh <<'EOF'
{{snippet}}EOF
```

Your shell keeps no variables between commands, so begin **every** command with
`REPO='<that path>';` and call the scripts by their absolute path, from any directory, with
absolute paths for every file and folder, e.g. `REPO='/Users/me/oh-my-slam'; {{example}}`. The
checkout needs `uv sync` once (a script whose Python environment is missing says so: ask the user
to run it there), and the scripts that use the inference server need it running (see
[Rules](#rules)).

Every script writes at most one result on stdout ({{stdout}}), or with `-o` to that file instead;
everything else goes to stderr: progress, and on failure what went wrong, with the exit status of
[Errors](#errors).
"""

REPO_SNIPPET = r"""cache="{{cache}}"
holds() {  # the checkout: the scripts at its root
  for s in {{scripts}}; do
    [ -f "$1/$s" ] || return 1
  done
}
clean() {  # an absolute path, ~ expanded, without a trailing /; none with ' or a line break
  p=$1
  case $p in "~") p=$HOME ;; "~/"*) p=$HOME/${p#"~/"} ;; esac
  case $p in /*) ;; ?*) p=$PWD/$p ;; esac
  case $p in */) p=${p%/} ;; esac
  case $p in *"'"*|*"
"*) p= ;; esac
  printf '%s' "$p"
}
found() {  # cache the path, print it and stop
  { mkdir -p "${cache%/*}" && printf '%s\n' "$1" >"$cache"; } 2>/dev/null
  printf '%s\n' "$1"
  exit 0
}
if [ -n "${OMS_REPO:-}" ]; then  # 1. the path the user gives
  r=$(clean "$OMS_REPO")
  [ -n "$r" ] && holds "$r" && found "$r"
  echo "oh-my-slam: $OMS_REPO does not hold the oh-my-slam scripts; check the path with the user" >&2
  exit 1
fi
r=$(clean "$(cat "$cache" 2>/dev/null)")  # 2. the cached path, while it holds the scripts
[ -n "$r" ] && holds "$r" && found "$r"
echo "oh-my-slam: no checkout found (tried $cache). Ask the user for the path of the" \
  "oh-my-slam checkout (the folder that holds reconstruct.sh), then run this again with" \
  "OMS_REPO=<path>." >&2
exit 1
"""

API = """## API

`server.sh` is the web service: it runs every mode of {{progs}} as an operation within one HTTP
request. There are no jobs: the service answers when the command ends, with its result, byte for
byte, or its error. It keeps maps and uploads in its workspace (`{{data}}/` unless it was started
with `--data <folder>`), keeps no results, and binds `0.0.0.0`, so it serves this Mac and any
machine on the LAN; you need only `sh` and `curl`.

**`$BASE/api/openapi.json` describes every operation, parameter, result and error.** An
operation's parameters are its command's options, with the same names, values, defaults and
checks (`-o` and `-d` are none: the response is the result), and its whole definition is under
`x-oms`: outputs, checks, errors, stages and inference need. Read it before a request; where it
and this file disagree (an older or newer service), **the running service's document wins**.

| Operation | Runs | Inference server |
|---|---|---|
{{ops}}

The routes:

{{routes}}
"""

FIND = """### Find the service

`server.sh` binds a free port unless it was started with `--port <n>`, so its URL changes from run
to run. This snippet prints the base URL and caches it in `{{cache}}`. It tries, in order: the URL
the user gives (`OMS_URL`); the cached URL, if `/api/health` answers there within {{probe}} s; on
the Mac that runs the service, the URL the service records in `{{state}}` in its workspace
(`{{data}}/`, or the `--data` folder the user names, as `OMS_DATA`), with `0.0.0.0` replaced by
`127.0.0.1`. Otherwise it fails: ask the user for the URL that `server.sh` printed when it started
(`server.sh: listening on http://0.0.0.0:<port>/`; from another machine, the Mac's address or host
name in place of `0.0.0.0`) or that `server.sh --status` reports (`service.url`), and run it again
with the first line changed to `OMS_URL=<url> sh <<'EOF'` (or `OMS_DATA=<folder> sh <<'EOF'`).
There is no network or port scan; `server.sh --port <n>` keeps the URL stable for other machines.

```sh
sh <<'EOF'
{{snippet}}EOF
```

Begin **every** command with `BASE=<that URL>;` (a `;`, not a `BASE=... curl` prefix), e.g.
`BASE=http://127.0.0.1:<port>; curl -sS "$BASE/api/health"`. Use the URL as found: the service
answers {{forbidden}} `forbidden` to a `Host` that does not name its machine. `GET /api/health`
names the service's workspace (`service.data`): check it is the one the user means. If a request
later fails to connect (curl exit {{curl_exits}}), run the snippet again: it searches again only
when the cached URL stops answering.
"""

# The address snippet (spec §2.7 "Finds the project"): POSIX sh and curl only, no scan. It prints
# the base URL (scheme, host and port) and caches it; a URL is used only when it is made of URL
# characters, since the agent pastes it into its commands.
ADDRESS_SNIPPET = r"""cache="{{cache}}"
probe() {  # an oh-my-slam service answers /api/health there within {{probe}} s
  curl -fsS --max-time {{probe}} "$1/api/health" 2>/dev/null | grep -q '"inference"'
}
clean() {  # scheme://host:port of a URL (its path dropped); 0.0.0.0 is this machine
  u=$(printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  case $u in *://*) ;; ?*) u="http://$u" ;; esac
  u=$(printf '%s' "$u" | sed -e 's|^\([A-Za-z]*://[^/?#]*\).*|\1|' \
    -e 's|://0\.0\.0\.0:|://127.0.0.1:|' -e 's|://0\.0\.0\.0$|://127.0.0.1|')
  case $u in http://*|https://*) ;; *) u= ;; esac
  case $u in *[!]A-Za-z0-9.:/_[-]*) u= ;; esac
  printf '%s' "$u"
}
found() {  # cache the URL, print it and stop
  { mkdir -p "${cache%/*}" && printf '%s\n' "$1" >"$cache"; } 2>/dev/null
  printf '%s\n' "$1"
  exit 0
}
if [ -n "${OMS_URL:-}" ]; then  # 1. the URL the user gives
  u=$(clean "$OMS_URL")
  [ -n "$u" ] && probe "$u" && found "$u"
  echo "oh-my-slam: no oh-my-slam service answers at $OMS_URL; check the URL with the user" >&2
  exit 1
fi
u=$(clean "$(cat "$cache" 2>/dev/null)")  # 2. the cached URL, while it answers
[ -n "$u" ] && probe "$u" && found "$u"
data=${OMS_DATA:-{{data}}}  # 3. on the service's machine: its {{state}}
case $data in "~") data=$HOME ;; "~/"*) data=$HOME/${data#"~/"} ;; esac
u=$(clean "$(sed -n 's/.*"url": *"\([^"]*\)".*/\1/p' "$data/{{state}}" 2>/dev/null)")
[ -n "$u" ] && probe "$u" && found "$u"
echo "oh-my-slam: no service found (tried $cache and $data/{{state}})." \
  "Ask the user for the URL that server.sh printed on start or that server.sh --status" \
  "reports, then run this again with OMS_URL=<url>." >&2
exit 1
"""

WORKFLOW = """### A request, call by call

The example is `{{op}}` (`{{label}}`); every operation works the same way.

1. **Check the service** and the inference server: `curl -sS "$BASE/api/health"`.
   `inference.status` is one of {{statuses}}; an operation that needs the inference server is
   accepted while it is `ready` or `loading`. `service.requests` counts the requests `running` and
   those `waiting` for their turn.
2. **Upload each input file**, or name a path inside the workspace instead. The body is the raw
   file, sent with `-T` (a form, `curl -F`, is refused with {{form}}); `name` is the file name the
   command sees: keep its suffix and use only letters, digits, `.`, `_` and `-`. A parameter that
   takes several files (a JSON list) takes one upload per file, in the order the command reads
   them.
   ```sh
   {{upload_curl}}
   ```
   → {{created}} `{{upload}}`: its `path` is the parameter's value.
3. **Validate**: the command's own checks, and the inference server's when the request needs it;
   nothing runs and no upload is consumed. `command` is the command line it will run; with
   `"valid":false`, `problems` and `by_parameter` say what to change.
   ```sh
   curl -sS -X POST "$BASE/api/ops/{{op}}/validate" {{json}} -d '{{body}}'
   ```
   → `{{validated}}`
4. **Run it and wait**, with no client timeout: nothing comes back until the command ends (a
   mapping request can take many minutes, and a request that uses the inference server may first
   wait for its turn). Save the body with `-o`, the headers with `-D`; `-w` prints the status.
   ```sh
   {{run_curl}}
   ```
   → {{ok}}. Any other status means the file holds an error (see [Errors](#errors)).
5. **Read the stage timings** in the `Server-Timing` header: each of the command's own stages
   under its own name, then `total`, in milliseconds: `grep -i '^server-timing:' {{headers}}` →
   `server-timing: {{timing}}`.
6. **Report** the result file, or the error's `message` as it is. A map's summary and update
   history are at `GET /api/maps/<name>`; the browser application at `$BASE/` runs the same
   operations.
"""

ERRORS = """## Errors

A script that fails says why on stderr, the commands in one line `<script>: error: <message>`
(`<script>: interrupted` on Ctrl-C), and exits with a status of the table. An operation that fails
answers the command's message and the status's code, with the HTTP status that one generic rule
gives that code (the table):

```json
{"error": {"code": "<code>", "message": "<the command's own message>", "http_status": <status>}}
```

A refused request (and validate's `problems`) adds `rule`, `parameters` (those it concerns),
`exit_code`, `problems` (every problem) and `by_parameter`. For example, `POST /api/ops/{{op}}`
with the body `{}` answers {{refusal}}; a request whose command ran and failed answers its
`exit_code` and message, e.g. {{failed}}.

| Exit | Code | HTTP | Meaning |
|---|---|---|---|
{{codes}}

The service's own refusals have the same shape:

| Code | HTTP | When |
|---|---|---|
{{service}}
"""


def result_formats(scripts: list[Script]) -> list[str]:
    """The formats the scripts write on stdout (or ``-o``), in the order they first appear."""
    return list(dict.fromkeys(o.format for _, _, m in scripts for o in m.outputs
                              if o.via == RESULT))


def formats_text(scripts: list[Script]) -> str:
    """What the results are: the formats the scripts write on stdout; one without a gloss of its
    own is named by its results, each with the mode and the option that give it."""
    gloss = {"json": f"JSON is an ASAM OpenLABEL {openlabel.SCHEMA_VERSION} scene description (or "
                     "a server's health, for `--status`): each object has a label (`type`), a "
                     "score, a colour that is the same in every output, and an oriented bounding "
                     f"box `cuboid` whose `val` is `{','.join(CUBOID)}` (metres, quaternion "
                     f"scalar last), e.g. `{scene_sample(full=True)}`",
             "ply": "PLY is a point cloud in metres whose header records its attributes (`-p`)"}

    def named(fmt: str) -> str:
        return f"{FORMATS.get(fmt, fmt)} is " + " or ".join(
            f"{title(o)} ({code(label(c, m))}" + (f" {when_text(c, o.when)}" if o.when else "")
            + ")" for _, c, m in scripts for o in m.outputs if o.via == RESULT and o.format == fmt)

    return "Results: " + "; ".join(gloss.get(f) or named(f) for f in result_formats(scripts)) + "."


def render(ops: dict[str, Operation] | None = None) -> str:
    """The whole ``SKILL.md``."""
    ops = operations() if ops is None else ops
    doc = openapi.document(ops)
    scripts = covered()

    table = [f"| `{g[0][0].prog}` | " + ", ".join(code(label(c, m)) for _, c, m in g)
             + f" | {cell(g[0][0].description)} |" for g in grouped(scripts)]
    service_progs = list(dict.fromkeys(op.program.prog for op in ops.values()))

    def labels(pick: Callable[[Program, Mode], bool]) -> list[str]:
        return [code(label(c, m)) for p, c, m in scripts if pick(p, m)]

    def op_ids(pick: Callable[[Operation], bool]) -> list[str]:
        return [code(op.id) for op in ops.values() if pick(op)]

    need_scripts = labels(lambda p, m: is_command(p) and m.inference == "required")
    need_ops = op_ids(lambda op: op.mode.inference == "required")
    never_scripts = labels(lambda p, m: is_command(p) and m.inference == "never")
    never_ops = op_ids(lambda op: op.mode.inference == "never")
    without = ([listed("script", never_scripts)] if never_scripts else []) \
        + ([listed("operation", never_ops)] if never_ops else [])
    for p, c, m in scripts:
        if is_command(p) and m.inference == "conditional":
            without.append(listing([code(label(c, m)), *(code(op.id) for op in ops.values()
                                                         if op.mode is m)])
                           + f" except for {condition(m)}")
    writers_s = [(p, c, m) for p, c, m in scripts if map_written(c, m) is not None]
    writers = [code(label(c, m)) for _, c, m in writers_s] + op_ids(
        lambda op: op.writes_map() is not None)
    stages = [str(s) for s in writers_s[0][2].stages] if writers_s else []
    ex = example(ops)
    params = sample_params(ex)
    upload = workspace.Upload("<upload>", "photo.jpg",
                              Path(f"/<data>/{workspace.UPLOADS}/<upload>/photo.jpg"),
                              2481152).describe(Path("/<data>"))
    ex_stages = [str(s) for s in ex.mode.stages]
    media = spec.media_type(result_format(ex, params) or "json")
    outside = workspace.OutsideWorkspaceError.exit_code
    down = ExitCode.SERVER_UNAVAILABLE
    refusals = service_errors()
    first = next(s for s in scripts if is_command(s[0]))  # the checkout's example command
    examples = []
    for op in ops.values():
        if op is ex:
            continue
        note = " (writes the map: ask the user first)" if op.writes_map() is not None else ""
        examples += [f"# {op.label}{note}", run_curl(op, sample_params(op))]

    parts = [
        front_matter(scripts, ops),
        fill(INTRO, service_progs=listing([code(p) for p in service_progs]),
             table="\n".join(table), formats=formats_text(scripts)),
        fill(RULES, need_scripts=listing(need_scripts) or "none",
             need_ops=listing(need_ops) or "none", start=app.START_COMMAND,
             down_exit=str(int(down)), down_http=str(HTTP_STATUS[down]), down_code=error_code(down),
             without="; ".join(without) or "nothing",
             lifecycle=", ".join(labels(lambda p, m: m.lifecycle is not None)) + " and "
             + code(entry_points.WEB_SERVICE.prog),
             status=" and ".join(code(label(c, m)) for p in (entry_points.INFERENCE_SERVER,
                                                             entry_points.WEB_SERVICE)
                                 for c in p.commands for m in c.modes
                                 if m.lifecycle is None),
             maps=workspace.MAPS, writers=" and ".join(writers) or "the mapping operation",
             writer_stages=(f"{len(stages)} stages, from `{stages[0]}` to `{stages[-1]}`,"
                            if stages else "their stages"),
             uploads=workspace.UPLOADS, outside=f"{HTTP_STATUS[outside]} `{error_code(outside)}`"),
        fill(CHECKOUT, cache=REPO_CACHE, example=" ".join(
            [f'"$REPO/{first[0].prog}"', *sample_command(*first)[0]]), snippet=fill(
            REPO_SNIPPET, cache=REPO_CACHE,
            scripts=" ".join(p.prog for p in entry_points.scripts())),
            stdout=", ".join(FORMATS.get(f, f) for f in result_formats(scripts))),
        *("\n".join(script_section(p, c, m)) for p, c, m in scripts),
        fill(API, progs=listing([code(p) for p in service_progs]),
             data=constants.DEFAULT_DATA, ops="\n".join(
                 f"| `{op.id}` | `{op.label}` | {inference_cell(op)} |" for op in ops.values()),
             routes="\n".join(routes(doc))),
        fill(FIND, cache=URL_CACHE, probe=num(PROBE_TIMEOUT_S), state=service.STATE,
             data=constants.DEFAULT_DATA, forbidden=str(refusals["forbidden"][0]),
             curl_exits=" or ".join(map(str, CURL_EXITS)), snippet=fill(
                 ADDRESS_SNIPPET, cache=URL_CACHE, probe=num(PROBE_TIMEOUT_S),
                 state=service.STATE, data=HOME_DATA)),
        fill(WORKFLOW, op=ex.id, label=ex.label, statuses=statuses(), json=JSON_POST,
             form=str(refusals["unsupported_media_type"][0]),
             created=success(doc, "/api/uploads", "post"),
             upload_curl=f"curl -sS -X POST -T {upload['name']} -H 'Content-Type: "
                         f"application/octet-stream' \"$BASE/api/uploads?name={upload['name']}\"",
             upload=compact(upload), body=compact(params), validated=validated(ex, params),
             run_curl=run_curl(ex, params),
             ok=(f"{success(doc, f'/api/ops/{ex.id}', 'post')}, `Content-Type: {media}`, "
                 f"`Server-Timing: {server_timing(ex_stages)}`; `{result_file(ex, params)}` "
                 "holds the result, byte for byte"),
             headers=HEADERS, timing=server_timing(ex_stages)),
        "### Examples\n\n" + sh(*examples) + "\n",
        fill(ERRORS, op=ex.id, refusal=refusal(ex), failed=failed(ExitCode.NOT_REGISTERED),
             codes="\n".join(f"| {int(c)} | `{error_code(c)}` | {HTTP_STATUS[c]} "
                             f"| {cell(MEANING[c])} |" for c in ExitCode),
             service="\n".join(f"| `{k}` | {status} | {cell(when)} |"
                               for k, (status, when) in refusals.items())),
    ]
    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1 or args[:1] in (["-h"], ["--help"]):
        print("usage: python -m oh_my_slam.web.skill [PATH]   (default: SKILL.md at the "
              "repository root)", file=sys.stderr)
        return 0 if args[:1] in (["-h"], ["--help"]) else 2
    path = Path(args[0]) if args else DEFAULT_PATH
    path.write_text(render(), encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
