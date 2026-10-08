"""The agent skill ``SKILL.md`` at the repository root (spec §2.7, ``agent_skill.md``): one
self-contained file that lets an AI agent use every operation of the ``server.sh`` API with ``sh``
and ``curl``, from any machine that reaches the service and with no checkout. The API is its only
entry point: it runs no local script and documents none, and names one only to tell the user what
to run (starting a server, or ``server.sh --status`` for the service's URL).

It is generated, never hand-edited, from the same shared definitions as ``/api/openapi.json``:

* every operation (``web.operations``: one per mode of the commands the service offers) with its
  route, what it takes and gives, whether it writes a map and whether it needs the inference
  server, and every other route from the OpenAPI document (``web.openapi``). That document stays
  the source of every parameter, result and error, so the skill adds only what it cannot say —
  finding the service, the order of calls, the error shape, the rules and a ``curl`` example per
  operation;
* what the user runs, from the entry points' definitions (``commands.entry_points``): the servers
  the agent never starts or stops, how the user starts each (as typed in the checkout: the
  inference server's ``core.constants.START_INFERENCE_SERVER``, which its errors and the
  service's health name) and ``server.sh --status``;
* the errors from the same document: the commands' exit codes (``x-oms.exit_codes``, with their
  meanings to an API client from ``core.errors.API_MEANING``) and the service's own refusals with
  the limits they state (``x-oms.refusals``, from ``web.app.refusals``);
* the samples come from the code: the upload record (``workspace.Upload``), a refused request
  (``operations.error_body``), a failed command's answer (``app.failure``), the ``Server-Timing``
  of a timing record (``app.timing_header``) and the OpenLABEL builders. Limits are read from
  their modules when the file is rendered.

What concerns no single operation (finding the service, the order of calls, the rules) is the
template below. ``tests/unit/test_agent_skill.py`` fails when the committed file differs from this
output, misses an operation or route, offers an operation, parameter, endpoint or script the
service does not support, or states a limit that differs from the code.

Regenerate: ``uv run python -m oh_my_slam.web.skill [PATH]`` (default ``SKILL.md`` at the
repository root)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, get_args

import numpy as np

from oh_my_slam.client.protocol import ServerStatus
from oh_my_slam.commands import entry_points, spec
from oh_my_slam.commands.spec import Kind, Mode, Option, Output, Program, When
from oh_my_slam.core import constants
from oh_my_slam.core.errors import API_MEANING, HTTP_STATUS, ExitCode, error_code
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
from oh_my_slam.web.runner import Outcome

Json = dict[str, Any]
NAME = "oh-my-slam"
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "SKILL.md"
URL_CACHE = "${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/server_url"  # the service's base URL
DESCRIPTION_MAX = 1024  # the Agent Skills limit
PROBE_TIMEOUT_S = 3  # the address snippet: a service answers /api/health within this
JSON_POST = "-H 'Content-Type: application/json'"
HEADERS = "headers.txt"  # where the workflow saves a response's headers (-D)
CURL_EXITS = (7, 28)  # curl's exit statuses for "could not connect" and "timed out"
HOME_DATA = constants.DEFAULT_DATA.replace("~", "$HOME", 1)  # the default workspace, in sh

# What an operation takes, by input kind (a map the operation only reads must exist).
TAKES = {Kind.IMAGE: "one image", Kind.IMAGES: "images", Kind.IMAGES_OR_VIDEO: "images or one video",
         Kind.MAP: "a new or existing map"}
EXISTING_MAP = "an existing map"
FORMATS = {"json": "JSON", "ply": "PLY", "png": "PNG"}
# The description's gloss of a result format; another is glossed by the names of the results in
# it (``title``: "PNG = depth image or segmented image").
GLOSS = {"json": "OpenLABEL scene (labelled objects with oriented bounding boxes; for a map, "
                 "camera poses)",
         "ply": "point cloud"}
# Each inference condition of the registry (``spec.CONDITIONS``) in words, its value for ``{}``.
CONDITION_WORDS = {"map_keyframes_greater_than": "maps over {} keyframes"}
# The 10 values of an OBB cuboid as schema.openlabel writes them (cuboid_val: quaternion scalar
# last); tests/unit/test_agent_skill.py checks them against the builder.
CUBOID = ("x", "y", "z", "qx", "qy", "qz", "qw", "sx", "sy", "sz")


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


def listing(items: list[str], conj: str = "and") -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return f" {conj} ".join([", ".join(items[:-1]), items[-1]] if len(items) > 2 else items)


def statuses() -> str:
    return ", ".join(code(s) for s in ("down", *get_args(ServerStatus)))


def condition(mode: Mode) -> str:
    """When a "conditional" mode needs the inference server, in words (any of its conditions)."""
    return " or ".join(CONDITION_WORDS[k].format(v)
                       for k, v in (mode.inference_condition or {}).items())


def title(out: Output) -> str:
    """An output's name: its text up to the first colon (``the depth image: one 16-bit …``)."""
    return out.text.split(":", 1)[0]


# -- what the user runs: the entry points' definitions -----------------------------------------------


def servers() -> list[Program]:
    """The long-lived servers (the entry points with a mode that starts or stops one): the user
    starts and stops them, never the agent."""
    return [p for p in entry_points.scripts()
            if any(m.lifecycle is not None for c in p.commands for m in c.modes)]


def start_command(p: Program) -> str:
    """How the user starts the server ``p``: its start mode as typed in the checkout, as the
    inference server's start command is (``core.constants.START_INFERENCE_SERVER``), which its
    errors and the service's health name."""
    (label,) = [c.label(m) for c in p.commands for m in c.modes if m.lifecycle == "starts"]
    return constants.CHECKOUT + label


def status_command() -> str:
    """``server.sh --status``: the service's mode that neither starts nor stops it, which reports
    its URL (the user runs it)."""
    c = entry_points.WEB_SERVICE.command()
    (m,) = [m for m in c.modes if m.lifecycle is None]
    return c.label(m)


def service_flag(name: str) -> str:
    """The flag of a ``server.sh`` option the user may give (``--port``, ``--data``)."""
    return entry_points.WEB_SERVICE.command().option(name).flag


# -- the operations: everything here comes from the operations' definitions ---------------------------


def inputs(op: Operation) -> list[Option]:
    """The inputs an operation must be given: its required files and map."""
    return [o for o in op.options if o.kind in TAKES and (o.required or o.name == op.mode.selector)]


def takes(o: Option) -> str:
    return EXISTING_MAP if o.kind is Kind.MAP and o.must_exist else TAKES[o.kind]


def results(op: Operation) -> list[Output]:
    """The results an operation answers with (its command's stdout)."""
    return [o for o in op.mode.outputs if o.via == RESULT]


def result_name(o: Output) -> str:
    """A result by its format; one without a gloss of its own with its name (``PNG (depth
    image)``)."""
    fmt = FORMATS.get(o.format, o.format)
    return fmt if o.format in GLOSS else f"{fmt} ({title(o).removeprefix('the ')})"


def gives(op: Operation) -> str:
    """What an operation answers: its results and the parameters that pick one."""
    outs = results(op)
    by = list(dict.fromkeys(w.option for o in outs for w in o.when))
    return listing([result_name(o) for o in outs], "or") + (
        f", by {listing([code(b) for b in by])}" if by else "")


def takes_gives(op: Operation) -> str:
    ins = " + ".join(f"`{o.name}` ({takes(o)}" + (", in order" if o.ordered else "") + ")"
                     for o in inputs(op))
    return f"{ins} → {gives(op)}" if ins else gives(op)


def map_cell(op: Operation) -> str:
    writes = op.writes_map()
    return f"**writes the map** `{writes.name}`" if writes is not None else "read-only"


def inference_cell(op: Operation) -> str:
    return {"required": "needed", "never": "not needed"}.get(
        op.mode.inference, f"only for {condition(op.mode)}")


def writers(ops: dict[str, Operation]) -> list[str]:
    """The operations that write a map."""
    return [code(op.id) for op in ops.values() if op.writes_map() is not None]


def routes(doc: Json) -> list[str]:
    """The service's own routes from the OpenAPI document, with what each is for."""
    lines = []
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


def sample_file(o: Option) -> str:
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
            path = f"{workspace.UPLOADS}/<upload>/{sample_file(o)}"
            out[o.name] = [path] if o.multiple else path
        else:
            out[o.name] = o.choices[0] if o.choices else o.default
    return out


def upload_curl(name: str) -> str:
    """An upload as ``curl -T`` sends it: the raw file as the body of a ``PUT``."""
    return f"curl -sS -T {name} \"$BASE/api/uploads?name={name}\""


def run_curl(op: Operation, params: Json) -> str:
    return (f"curl -sS -X POST -o {result_file(op, params)} -D {HEADERS} "
            f"-w '%{{http_code}}\\n' \"$BASE/api/ops/{op.id}\" {JSON_POST} -d '{compact(params)}'")


def result_file(op: Operation, params: Json) -> str:
    return "result" + spec.suffix_of(result_format(op, params) or "")


def examples(ops: dict[str, Operation]) -> list[str]:
    """A ready-to-run example per operation: its uploads, then the request."""
    lines = []
    for op in ops.values():
        lines.append(f"# {op.id}: {map_cell(op).replace('*', '')}; inference server "
                     f"{inference_cell(op)}"
                     + ("; ask the user first" if op.writes_map() is not None else ""))
        lines += [upload_curl(sample_file(o)) for o in inputs(op) if o.kind in spec.ACCEPTS]
        lines.append(run_curl(op, sample_params(op)))
    return lines


def validated(op: Operation, params: Json) -> str:
    """Validate's answer to a valid request (its ``command`` elided)."""
    return compact({"valid": True, "command": [], "inference": op.mode.inference == "required",
                    "problems": [], "by_parameter": {}}).replace('"command":[]', '"command":[…]')


def refusal(op: Operation) -> str:
    """What a request of ``op`` with an empty body answers: the refusal of the command's own
    parser, with its status (the messages, the command's words, elided)."""
    status, body = error_body(spec.dry_run(op.command, op.mode, {}))
    err = {**body["error"], "message": "…", "problems": [],
           "by_parameter": {k: ["…"] for k in body["error"]["by_parameter"]}}
    return f"{status} `{compact({'error': err}).replace('"problems":[]', '"problems":[…]')}`"


def server_timing(stages: list[str]) -> str:
    """The ``Server-Timing`` header of a run that took these stages (sample figures)."""
    return app.timing_header({"stages_s": dict.fromkeys(stages, 0.5),
                              "total_s": 0.5 * len(stages)})


def failed(code_: ExitCode) -> str:
    """The answer to a request whose command ended with ``code_`` (``app.failure``)."""
    status, body = app.failure(Outcome(int(code_), "<the command's message>"))
    return f"{status} `{compact(body)}`"


def refusals(doc: Json) -> dict[str, Json]:
    """The service's own refusals (no command's), by code, as its OpenAPI document exports them
    (``x-oms.refusals``, from ``web.app.refusals``)."""
    return {r["code"]: r for r in doc["x-oms"]["refusals"]}


# -- the results --------------------------------------------------------------------------------------


def result_formats(ops: dict[str, Operation]) -> list[str]:
    """The formats the operations answer with, in the order they first appear."""
    return list(dict.fromkeys(o.format for op in ops.values() for o in results(op)))


def when_text(whens: tuple[When, ...]) -> str:
    """When a result is the answer, by the request's parameters."""
    def one(w: When) -> str:
        if w.video:
            return f"for one video in `{w.option}`"
        if w.values:
            return " or ".join(f"with `{compact({w.option: v})[1:-1]}`" for v in w.values)
        return f"with `{w.option}`"

    return " or ".join(one(w) for w in whens)


def scene_sample() -> str:
    """An OpenLABEL scene as the commands write it, cut short: its metadata and one object built
    by the shared builders (``schema.openlabel``)."""
    obj = openlabel.object_entry(
        "chair 1", "chair", "camera",
        openlabel.cuboid(np.array([0.41, 0.18, 2.35]), np.eye(3), np.array([0.48, 0.51, 0.92]),
                         "camera"),
        nums=[openlabel.num("score", 0.91)])
    doc = {"openlabel": {"metadata": {"schema_version": openlabel.SCHEMA_VERSION, "…": 0},
                         "objects": {"1": {**obj, "…": 0}, "…": 0}, "…": 0}}
    return compact(doc).replace(',"…":0', ",…")


def formats_text(ops: dict[str, Operation]) -> str:
    """What the results are: each format the operations answer with; one without a gloss of its
    own is named by its results, each with the operation and the parameter value that give it."""
    attrs = list(dict.fromkeys(code(o.name) for op in ops.values() for o in op.options
                               if o.kind is Kind.ATTRS))
    gloss = {"json": f"JSON is an ASAM OpenLABEL {openlabel.SCHEMA_VERSION} scene description: "
                     "each object has a label (`type`), a score, a colour that is the same in "
                     "every result, and an oriented bounding box `cuboid` whose `val` is "
                     f"`{','.join(CUBOID)}` (metres, quaternion scalar last), e.g. "
                     f"`{scene_sample()}`; a map's scene also holds its camera poses",
             "ply": "PLY is a point cloud in metres whose header records its attributes"
                    + (f" ({listing(attrs)})" if attrs else "")}

    def named(fmt: str) -> str:
        return f"{FORMATS.get(fmt, fmt)} is " + " or ".join(
            f"{title(o)} ({code(op.id)}" + (f" {when_text(o.when)}" if o.when else "") + ")"
            for op in ops.values() for o in results(op) if o.format == fmt)

    return "Results: " + "; ".join(gloss.get(f) or named(f) for f in result_formats(ops)) + "."


# -- the description ----------------------------------------------------------------------------------


def description(ops: dict[str, Operation]) -> str:
    """What the skill can do and when to use it (spec §2.7 "Description"): every operation with
    what it takes and produces, the need for a reachable server.sh and the operations that need
    the inference server, within the Agent Skills limit."""
    def clause(op: Operation) -> str:
        ins = " + ".join(takes(o) for o in inputs(op))
        outs = "/".join(dict.fromkeys(FORMATS.get(o.format, o.format) for o in results(op)))
        return f"{op.id}: " + (f"{ins} → " if ins else "") + outs

    def gloss(fmt: str) -> str:
        names = dict.fromkeys(title(o).removeprefix("the ") for op in ops.values()
                              for o in results(op) if o.format == fmt)
        return f"{FORMATS.get(fmt, fmt)} = {GLOSS.get(fmt) or ' or '.join(names)}"

    def ids(inference: str) -> list[str]:
        return [op.id for op in ops.values() if op.mode.inference == inference]

    web = entry_points.WEB_SERVICE.prog
    writing = [op.id for op in ops.values() if op.writes_map() is not None]
    sometimes = [f"{op.id} for {condition(op.mode)}" for op in ops.values()
                 if op.mode.inference == "conditional"]
    text = (f"oh-my-slam monocular RGB 3D mapping on a Mac, through the HTTP API of its {web} web "
            "service with sh and curl, from this machine or another on the LAN; needs a "
            f"{web} reachable from this machine, no checkout. Operations: "
            + "; ".join(clause(op) for op in ops.values()) + ". "
            + f"{listing(writing)} write{'s' if len(writing) == 1 else ''} a map; the others "
              "are read-only. Also uploads, validation, the workspace's maps, health. "
            + ", ".join(gloss(f) for f in result_formats(ops))
            + ". Inference server needed by " + listing(ids("required"))
            + (f", and by {listing(sometimes)}" if sometimes else "")
            + "; " + listing([*ids("never"), "the other routes"]) + " work without it. Use it "
            "when the user asks for any of these on images or video, or about the maps.")
    if len(text) > DESCRIPTION_MAX:
        raise ValueError(f"the description has {len(text)} characters (Agent Skills: at most "
                         f"{DESCRIPTION_MAX}); shorten the template or the words of "
                         "skill.TAKES / FORMATS / GLOSS / CONDITION_WORDS")
    return text


def front_matter(ops: dict[str, Operation]) -> str:
    """The skill's name and description (a YAML double-quoted scalar: JSON is valid YAML)."""
    return (f"---\nname: {NAME}\ndescription: "
            f"{json.dumps(description(ops), ensure_ascii=False)}\n---\n")


# -- the template ------------------------------------------------------------------------------------

INTRO = """# oh-my-slam

oh-my-slam turns RGB images and video into 3D on a Mac: scene descriptions of labelled objects
with oriented bounding boxes, point clouds, and persistent maps in which images can be located.
This skill uses it through one entry point, the HTTP API of `{{service}}`, its web service. You
need only `sh` and `curl`, on the Mac that runs the service or on any machine on the LAN (the
service binds `0.0.0.0`), and no checkout of oh-my-slam.

**`$BASE/api/openapi.json` describes every operation, parameter, result and error.** Read it before
a request; where it and this file disagree (an older or newer service), **the running service's
document wins**. It lists each operation's parameters with their values, defaults and checks, and
gives its whole definition under `x-oms`: outputs, checks, errors, stages and inference need. This
file adds only what that document cannot say: the operations and routes, how to find the service,
the order of calls, an example per operation, the errors, and the [Rules](#rules), which you
follow.
"""

RULES = """## Rules

* **The inference server may be down.** Then the operations {{need_ops}} fail with HTTP
  {{down_http}} `{{down_code}}`, with a message that names the command that starts it, `{{start}}`,
  as `GET /api/health` does (`inference.start_command`). Report the message and that command to
  the user instead of retrying, and offer what works without it: {{without}}.
* **Never start or stop {{servers}}**, and never kill their processes. When one is not running,
  tell the user how to start it: `{{start}}` for the inference server, and `{{service_start}}` in
  the oh-my-slam checkout on the Mac for the service (with `{{port}} <n>` for a URL that stays the
  same, `{{data_flag}} <folder>` for another workspace).
* **Never change a map except through {{writers}}**, the mapping operation: no other request
  writes one, and nothing else may touch a map's folder.
* **Ask the user first** before updating an existing map (`GET /api/maps/<name>` answers
  {{not_found}} `not_found` for a new one), and before starting a long mapping request:
  {{writers}} runs {{writer_stages}} and can take many minutes. Say that an update changes the map
  for good.
* **Inputs are uploads or paths inside the workspace** (relative to it, such as
  `{{uploads}}/<upload>/photo.jpg`; a map by its name), never paths outside it, which are refused
  ({{outside}}). **An upload is consumed by the one request it is given to**, whatever its
  outcome, so repeating a request means uploading again; validating does not consume it.
* **Wait for an answer with no client timeout** (no `--max-time`, no `-m`): disconnecting
  interrupts the command as Ctrl-C would, and an interrupted map update leaves the map as it was.
  Requests that use the inference server run one at a time, in arrival order: send them one by
  one.
* **Results are the commands' own bytes:** save them with `curl -o` and never rewrite them; report
  a failure's message as it is. Don't repeat a refused or failed request unchanged.
* **Offer only what the service supports:** the operations, parameters, values and routes of
  `/api/openapi.json`, and never a local script. Viewing an image or a map in 3D is no operation
  of the API: leave it to the user.
"""

OPERATIONS = """## Operations and routes

Each operation runs one of oh-my-slam's commands within one HTTP request. There are no jobs: the
service answers when the command ends, with its result, byte for byte, or its error. It keeps maps
and uploads in its workspace (`{{data}}/` unless the service was started with
`{{data_flag}} <folder>`) and keeps no results.

| Operation | Route | Takes → gives | Map | Inference server |
|---|---|---|---|---|
{{ops}}

{{formats}}

The routes:

* `POST /api/ops/{op}`: run the operation `{op}` within the request and answer when its command
  ends
* `POST /api/ops/{op}/validate`: check a request with the command's own checks, and the inference
  server's when the request needs it; nothing runs and no upload is consumed
* `GET /api/openapi.json`: the description of every operation, parameter, result and error
{{routes}}

Only the operations need the inference server, and a map changes only through {{writers}}. The
browser application at `$BASE/` runs the same operations, for the user.
"""

FIND = """## Find the service

`{{service}}` binds a free port unless it was started with `{{port}} <n>`, so its URL changes from
run to run. This snippet prints the base URL and caches it in `{{cache}}`. It tries, in order: the
URL the user gives (`OMS_URL`); the cached URL, if `/api/health` answers there within {{probe}} s;
on the Mac that runs the service, the URL the service records in `{{state}}` in its workspace
(`{{data}}/`, or the `{{data_flag}}` folder the user names, as `OMS_DATA`), with `0.0.0.0` replaced
by `127.0.0.1`. Otherwise it fails: ask the user for the URL that `{{service}}` printed when it
started (`{{service}}: listening on http://0.0.0.0:<port>/`; from another machine, the Mac's address
or host name in place of `0.0.0.0`) or that `{{status}}` reports (`service.url`), and run it again
with the first line changed to `OMS_URL=<url> sh <<'EOF'` (or `OMS_DATA=<folder> sh <<'EOF'`).
There is no network or port scan; `{{service}} {{port}} <n>` keeps the URL stable for other
machines.

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

# The address snippet (spec §2.7 "Finds the service"): POSIX sh and curl only, no scan. It prints
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
  "Ask the user for the URL that {{service}} printed on start or that {{status}}" \
  "reports, then run this again with OMS_URL=<url>." >&2
exit 1
"""

WORKFLOW = """## A request, call by call

The example is `{{op}}`; every operation works the same way.

1. **Check the service** and the inference server: `curl -sS "$BASE/api/health"`.
   `inference.status` is one of {{statuses}}; an operation that needs the inference server is
   accepted while it is `ready` or `loading`, and otherwise `inference.start_command` names the
   command that starts it (see [Rules](#rules)). `service.requests` counts the requests `running`
   and those `waiting` for their turn.
2. **Upload each input file**, or name a path inside the workspace instead. The body is the raw
   file, sent with `curl -T <file>` (a form, `curl -F`, is refused with {{form}}); `name` is the
   file name the command sees: keep its suffix and use only letters, digits, `.`, `_` and `-`. A
   parameter that takes several files (a JSON list) takes one upload per file, in the order the
   command reads them.
   ```sh
   {{upload_curl}}
   ```
   → {{created}} `{{upload}}`: its `path` is the parameter's value.
3. **Validate**: the command's own checks, and the inference server's when the request needs it;
   nothing runs and no upload is consumed. The body is a JSON object of parameters, sent as
   `application/json` (`{{json}} -d '…'`, or `--json '…'` where the machine's curl has it; a
   plain `-d` is refused with {{form}}). With `"valid":false`, `problems` and `by_parameter` say
   what to change.
   ```sh
   curl -sS -X POST "$BASE/api/ops/{{op}}/validate" {{json}} -d '{{body}}'
   ```
   → `{{validated}}`
4. **Run it and wait**, with no client timeout: nothing comes back until the command ends (a
   mapping request can take many minutes, and a request that uses the inference server may first
   wait for its turn). Before {{writers}}, ask the user first (see [Rules](#rules)). Save the body
   with `-o`, the headers with `-D`; `-w` prints the status.
   ```sh
   {{run_curl}}
   ```
   → {{ok}}. Any other status means the file holds an error (see [Errors](#errors)).
5. **Read the stage timings** in the `Server-Timing` header: each of the command's own stages
   under its own name, then `total`, in milliseconds: `grep -i '^server-timing:' {{headers}}` →
   `server-timing: {{timing}}`.
6. **Report** the result file, or the error's `message` as it is. A map's summary and update
   history are at `GET /api/maps/<name>`.
"""

EXAMPLES = """## Examples

One per operation, each after `BASE=<that URL>;`. `<upload>` is the `id` that the upload above
it answered (its `path` is the value to give), and `<map>` the name of a map of the workspace.

{{examples}}
"""

ERRORS = """## Errors

An operation that fails answers the command's own message and the machine-readable code of its
exit status, with the HTTP status that one generic rule gives that code (the table):

```json
{"error": {"code": "<code>", "message": "<the command's own message>", "http_status": <status>}}
```

A refused request (and validate's `problems`) adds `rule`, `parameters` (the parameters it
concerns), `exit_code`, `problems` (every problem) and `by_parameter`. A message is the command's
own, so it may name a parameter by the command's flag: the document gives each parameter's flag as
its `x-oms.flag`. For example, `POST /api/ops/{{op}}` with the body `{}` answers {{refusal}}; a
request whose command ran and failed answers its `exit_code` and message, e.g. {{failed}}.

| `exit_code` | `code` | HTTP | Meaning |
|---|---|---|---|
{{codes}}

The service's own refusals have the same shape:

| `code` | HTTP | When |
|---|---|---|
{{service}}
"""


def render(ops: dict[str, Operation] | None = None) -> str:
    """The whole ``SKILL.md``."""
    ops = operations() if ops is None else ops
    doc = openapi.document(ops)
    web = entry_points.WEB_SERVICE
    ex = example(ops)
    params = sample_params(ex)
    upload = workspace.Upload("<upload>", "photo.jpg",
                              Path(f"/<data>/{workspace.UPLOADS}/<upload>/photo.jpg"),
                              2481152).describe(Path("/<data>"))
    ex_stages = [str(s) for s in ex.mode.stages]
    media = spec.media_type(result_format(ex, params) or "json")
    outside = workspace.OutsideWorkspaceError.exit_code
    down = ExitCode.SERVER_UNAVAILABLE
    refused = {k: str(r["http_status"]) for k, r in refusals(doc).items()}
    writing = [op for op in ops.values() if op.writes_map() is not None]
    stages = [str(s) for s in writing[0].mode.stages] if writing else []
    without = ["the workspace's maps (`GET /api/maps`, `GET /api/maps/{name}`)"]
    without += [f"{code(op.id)}" for op in ops.values() if op.mode.inference == "never"]
    without += [f"{code(op.id)} except for {condition(op.mode)}" for op in ops.values()
                if op.mode.inference == "conditional"]
    data_flag, port = service_flag("data"), service_flag("port")

    parts = [
        front_matter(ops),
        fill(INTRO, service=web.prog),
        fill(RULES, need_ops=listing([code(op.id) for op in ops.values()
                                      if op.mode.inference == "required"]) or "none",
             down_http=str(HTTP_STATUS[down]), down_code=error_code(down),
             start=app.START_COMMAND, without=listing(without, ";").replace(" ; ", "; "),
             servers=listing([code(p.prog) for p in servers()], "or"),
             service_start=start_command(web), port=port, data_flag=data_flag,
             writers=listing(writers(ops)) or "the mapping operation",
             not_found=refused["not_found"],
             writer_stages=(f"{len(stages)} stages, from `{stages[0]}` to `{stages[-1]}`,"
                            if stages else "its stages"),
             uploads=workspace.UPLOADS, outside=f"{HTTP_STATUS[outside]} `{error_code(outside)}`"),
        fill(OPERATIONS, data=constants.DEFAULT_DATA, data_flag=data_flag, ops="\n".join(
            f"| `{op.id}` | `POST /api/ops/{op.id}` | {cell(takes_gives(op))} | {map_cell(op)} "
            f"| {inference_cell(op)} |" for op in ops.values()),
             formats=formats_text(ops), routes="\n".join(routes(doc)),
             writers=listing(writers(ops)) or "no operation"),
        fill(FIND, service=web.prog, port=port, cache=URL_CACHE, probe=f"{PROBE_TIMEOUT_S:g}",
             state=service.STATE, data=constants.DEFAULT_DATA, data_flag=data_flag,
             status=status_command(), forbidden=refused["forbidden"],
             curl_exits=" or ".join(map(str, CURL_EXITS)), snippet=fill(
                 ADDRESS_SNIPPET, cache=URL_CACHE, probe=f"{PROBE_TIMEOUT_S:g}",
                 state=service.STATE, data=HOME_DATA, service=web.prog,
                 status=status_command())),
        fill(WORKFLOW, op=ex.id, statuses=statuses(), json=JSON_POST,
             form=refused["unsupported_media_type"],
             created=success(doc, "/api/uploads", "put"),
             upload_curl=upload_curl(upload["name"]), upload=compact(upload), body=compact(params),
             validated=validated(ex, params), run_curl=run_curl(ex, params),
             writers=listing(writers(ops)) or "a request that writes a map",
             ok=(f"{success(doc, f'/api/ops/{ex.id}', 'post')}, `Content-Type: {media}`, "
                 f"`Server-Timing: {server_timing(ex_stages)}`; `{result_file(ex, params)}` "
                 "holds the result, byte for byte"),
             headers=HEADERS, timing=server_timing(ex_stages)),
        fill(EXAMPLES, examples=sh(*examples(ops))),
        fill(ERRORS, op=ex.id, refusal=refusal(ex), failed=failed(ExitCode.NOT_REGISTERED),
             codes="\n".join(f"| {c['exit_code']} | `{c['code']}` | {c['http_status']} "
                             f"| {cell(API_MEANING[ExitCode(c['exit_code'])])} |"
                             for c in doc["x-oms"]["exit_codes"]),
             service="\n".join(f"| `{r['code']}` | {r['http_status']} | {cell(r['when'])} |"
                               for r in refusals(doc).values())),
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
