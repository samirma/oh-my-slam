"""The agent skill ``SKILL.md`` at the repository root (spec §2.7): one self-contained file that lets
an AI agent use the ``server.sh`` API with ``sh`` and ``curl`` alone.

Like ``/api/openapi.json`` it is generated, never hand-edited: the operations, their parameters,
defaults, allowed values, checks, results, stages and error codes come from ``openapi.document``
(the commands' definitions, ``spec.describe()``, as ``Operation.entry`` offers them), and the
samples from the service's own code — the upload record (``workspace.Upload``), the command line
and the refusal the commands' parser gives (``spec.argv_of``, ``spec.dry_run``,
``operations.error_body``), a failed command's error (``app.failure``) and the ``Server-Timing``
header of a timing record (``app.timing_header``) — so a command that gains, changes or loses an
option, an output or an error changes the skill with no hand edit. What concerns no single command
(finding the service, the request workflow, safety, the service's own endpoints and refusals) is
the template below; an endpoint added to the OpenAPI document without a template entry still gets
a generic one. ``tests/unit/test_agent_skill.py`` fails when the committed file differs from this
output or misses a route of the app.

Regenerate: ``uv run python -m oh_my_slam.web.skill [PATH]`` (default ``SKILL.md`` at the
repository root)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

from oh_my_slam.client.protocol import ServerStatus
from oh_my_slam.commands import spec
from oh_my_slam.commands.spec import Kind
from oh_my_slam.core.errors import ExitCode
from oh_my_slam.web import openapi
from oh_my_slam.web.app import (
    MAX_UPLOAD_BYTES,
    MIN_FREE_BYTES,
    START_COMMAND,
    failure,
    timing_header,
)
from oh_my_slam.web.operations import (
    PATH_IN,
    RESULT,
    Operation,
    error_body,
    operations,
    result_format,
)
from oh_my_slam.web.runner import Outcome
from oh_my_slam.web.workspace import UPLOADS, Upload

Json = dict[str, Any]
NAME = "oh-my-slam-api"
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "SKILL.md"
CACHE = "${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam-api/server_url"
CREATED = 1791281700.0  # the samples' clock
SAMPLE_UPLOAD = Upload("<upload>", "photo.jpg", Path(f"/<data>/{UPLOADS}/<upload>/photo.jpg"),
                       2481152)
JSON_POST = "-H 'Content-Type: application/json'"
HEADERS = "headers.txt"  # where the workflow saves a response's headers (-D)


def gib(n: int) -> str:
    return f"{n / 2**30:g} GiB"


# The service's own refusals (no command's): code → (HTTP status, when). Their shape is that of
# every error; tests/unit/test_agent_skill.py checks that each code oh_my_slam.web answers with
# is here or in the commands' exit-code table.
SERVICE_ERRORS: dict[str, tuple[int, str]] = {
    "not_found": (404, "no such map, upload or operation"),
    "forbidden": (403, "a `Host` that does not name the service's machine, or a foreign `Origin`"),
    "unsupported_media_type": (415, "a POST whose body is not `application/json`, or an upload "
                                    "sent as a form (`curl -F`)"),
    "too_large": (413, f"an upload over {gib(MAX_UPLOAD_BYTES)}"),
    "insufficient_storage": (413, "an upload that would leave less than "
                                  f"{gib(MIN_FREE_BYTES)} free on the workspace's disk"),
    "upload_in_use": (409, "an upload that another request in progress was given (run it, or "
                           "delete it, with an upload of its own)"),
    "stopping": (503, "a request while the service stops, or one whose command it interrupted "
                      "when it stopped"),
}

# The service's own features (every route but an operation's run), as the description names
# them: routes → words. tests/unit/test_agent_skill.py checks that these routes are exactly the
# app's, so an endpoint that is added or removed needs its words changed here.
FEATURES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("POST /api/ops/{op}/validate",), "validate requests"),
    (("POST /api/uploads", "DELETE /api/uploads/{id}"), "upload and discard inputs"),
    (("GET /api/maps", "GET /api/maps/{name}"), "list maps with their summaries and update "
                                                "history"),
    (("GET /api/health",), "service and inference-server health"),
    (("GET /api/openapi.json",), "the OpenAPI document"),
)
# The description's words for the operations' output formats (another format is named as is).
FORMATS = {"json": "JSON", "ply": "PLY", "map": "map"}
DESCRIPTION_MAX = 1024  # the Agent Skills limit

# The address snippet (spec §2.7 "Server address"): POSIX sh and curl only, no scan. It prints
# the base URL (scheme, host and port) and caches it; a URL is used only when it is made of URL
# characters, since the agent pastes it into its commands.
ADDRESS_SNIPPET = r"""cache="${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam-api/server_url"
probe() {  # an oh-my-slam service answers /api/health there within 3 s
  curl -fsS --max-time 3 "$1/api/health" 2>/dev/null | grep -q '"inference"'
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
  echo "oh-my-slam-api: no oh-my-slam service answers at $OMS_URL; check the URL with the user" >&2
  exit 1
fi
u=$(clean "$(cat "$cache" 2>/dev/null)")  # 2. the cached URL, while it answers
[ -n "$u" ] && probe "$u" && found "$u"
data=${OMS_DATA:-$HOME/oh-my-slam-data}  # 3. on the service's machine: its server.json
case $data in "~") data=$HOME ;; "~/"*) data=$HOME/${data#"~/"} ;; esac
u=$(clean "$(sed -n 's/.*"url": *"\([^"]*\)".*/\1/p' "$data/server.json" 2>/dev/null)")
[ -n "$u" ] && probe "$u" && found "$u"
echo "oh-my-slam-api: no service found (tried $cache and $data/server.json)." \
  "Ask the user for the URL that server.sh printed on start or that server.sh --status" \
  "reports, then run this again with OMS_URL=<url>." >&2
exit 1
"""


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


def chain(names: list[str]) -> str:
    return " → ".join(code(s) for s in names) or "none"


def statuses() -> str:
    return ", ".join(code(s) for s in ("down", *get_args(ServerStatus)))


# -- operations: everything here comes from the commands' definitions ------------------------------


def entry(doc: Json, op: Operation) -> Json:
    """The operation's ``spec.describe()`` entry as the API offers it (the OpenAPI document's)."""
    return doc["paths"][f"/api/ops/{op.id}"]["post"]["x-oms"]


def sample_file(p: Json) -> str:
    """A sample input file of a suffix the parameter accepts (a video where it may be one)."""
    accepts = p["accepts"] or [".jpg"]
    stem, prefer = {str(Kind.IMAGES_OR_VIDEO): ("video", ".mp4"),
                    str(Kind.IMAGES): ("query", ".jpg")}.get(p["kind"], ("photo", ".jpg"))
    return stem + (prefer if prefer in accepts else accepts[0])


def sample_value(p: Json) -> Any:
    """A placeholder value for a required parameter."""
    if p["kind"] == str(Kind.MAP):
        return "<map>"
    if p["kind"] in {str(k) for k in PATH_IN}:
        path = f"{UPLOADS}/<upload>/{sample_file(p)}"
        return [path] if p["multiple"] else path
    return p["choices"][0] if p["choices"] else p["default"]


def sample_params(d: Json) -> Json:
    """The required parameters of an operation with placeholder values."""
    return {p["name"]: sample_value(p) for p in d["parameters"] if p["required"]}


def command_of(op: Operation, params: Json) -> list[str]:
    """The command line a request runs, as validate shows it (a map as ``maps/<name>``)."""
    shown = {k: (f"maps/{v}" if op.command.option(k).kind is Kind.MAP else v)
             for k, v in params.items()}
    return [op.program.prog, *spec.argv_of(op.command, op.mode, shown)]


def result_file(op: Operation, params: Json) -> str:
    """The file name the examples save a result in: ``result`` and its format's suffix."""
    return "result" + spec.suffix_of(result_format(op, params) or "")


def validated(op: Operation, d: Json, params: Json) -> str:
    return compact({"valid": True, "command": command_of(op, params),
                    "inference": d["inference"] == "required", "problems": [],
                    "by_parameter": {}})


def refusal(op: Operation) -> str | None:
    """What a request of ``op`` with an empty body answers: the refusal of the command's own
    parser, with its status (``problems`` repeats every problem, shortened here)."""
    problems = spec.dry_run(op.command, op.mode, {})
    if not problems:
        return None
    status, body = error_body(problems)
    err = ",".join(f"{json.dumps(k)}:" + ("[…]" if k == "problems" else compact(v))
                   for k, v in body["error"].items())
    return f"{status} `{{\"error\":{{{err}}}}}`"


def server_timing(stages: list[str]) -> str:
    """The ``Server-Timing`` header of a run that took these stages (sample figures)."""
    return timing_header({"stages_s": dict.fromkeys(stages, 0.5), "total_s": 0.5 * len(stages)})


def failed(code_: ExitCode) -> str:
    """The answer to a request whose command ended with ``code_``, from the service's own
    ``app.failure`` (the command's message is a placeholder)."""
    status, body = failure(Outcome(int(code_), "<the command's message>"))
    return f"{status} `{compact(body)}`"


def ok(op: Operation, params: Json, d: Json) -> str:
    """What a successful run answers: its status, media type and timings, and the saved file."""
    media = next((o["media_type"] for o in d["outputs"] if o["via"] == RESULT
                  and o["format"] == result_format(op, params)), "application/octet-stream")
    return (f"200, `Content-Type: {media}`, `Server-Timing: {server_timing(d['stages'])}`; "
            f"`{result_file(op, params)}` holds the result")


def when_text(w: Json) -> str:
    if w.get("is") == "video":
        return f"{code(w['option'])} is one video ({' '.join(w['suffixes'])})"
    if w.get("is") == "given":
        return f"{code(w['option'])} is given"
    return f"{code(w['option'])} is " + " or ".join(code(v) for v in w["in"])


def bounds(s: Json) -> list[str]:
    out = []
    if s.get("minimum") is not None:
        out.append(f"≥ {s['minimum']:g}")
    if s.get("exclusive_minimum") is not None:
        out.append(f"> {s['exclusive_minimum']:g}")
    if s.get("less_than"):
        out.append(f"< {code(s['less_than'])}")
    if s.get("finite") is False:
        out.append("or `inf`")
    return out


def param_type(p: Json) -> str:
    """What a parameter's value is, in the API's terms."""
    kind = p["kind"]
    if kind == str(Kind.FLAG):
        return "`true` or `false`"
    if kind == str(Kind.NUMBER):
        return " ".join([("finite " if p["finite"] else "") + "number",
                         *bounds({**p, "finite": None})])
    if kind == str(Kind.ENUM):
        return "one of " + ", ".join(code(c) for c in p["choices"])
    if kind == str(Kind.ATTRS):
        return "`key=value,…` (keys below)" + (", or a list of them" if p["repeatable"] else "")
    if kind == str(Kind.MAP):
        return "map name, `<name>` or `maps/<name>`: " + (
            "an existing map" if p["must_exist"] else "an existing map, or a new name to create")
    what = "list of workspace paths" if p["multiple"] else "workspace path"
    return what + (", in order (the order matters)" if p["ordered"] else "") + (
        ": " + " ".join(p["accepts"]) if p["accepts"] else "")


def param_table(d: Json) -> list[str]:
    rows = ["| Parameter | Flag | Value | Default | Meaning |", "|---|---|---|---|---|"]
    for p in d["parameters"]:
        default = p["default"]
        shown_default = "" if default is None else code(
            default if isinstance(default, str) else json.dumps(default))
        meaning = p["help"] + (f" ({p['applies_text']})" if p["applies_text"] else "")
        rows.append(f"| {code(p['name'])}{' (required)' if p['required'] else ''} "
                    f"| {code(p['flag'])} | {cell(param_type(p))} | {cell(shown_default)} "
                    f"| {cell(meaning)} |")
    return rows


def attrs_table(d: Json) -> list[str]:
    out: list[str] = []
    for p in d["parameters"]:
        if "attributes" not in p:
            continue
        out += ["", f"Keys of {code(p['name'])} (keys not given keep their default):", "",
                "| Key | Values | Default | Effect |", "|---|---|---|---|"]
        for a in p["attributes"]:
            s = a["schema"]
            values = " \\| ".join(code(c) for c in s["choices"]) if s["type"] == "enum" \
                else " ".join([s["type"], *bounds(s)])
            out.append(f"| {code(a['key'])} | {values} | {code(a['default'])} "
                       f"| {cell(a['effect'])} |")
    return out


def outputs_text(op: Operation, d: Json) -> list[str]:
    by_flag = {o.flag: o.name for o in op.options}
    out = []
    for o in d["outputs"]:
        cond = " when " + " or ".join(when_text(w) for w in o["when"]) if o["when"] else ""
        if o["via"] == RESULT:
            out.append(f"* The response{cond} ({o['media_type']}): {o['text']}.")
        elif o["format"] == "map":
            out.append(f"* The map {code(by_flag.get(o['via'], o['via']))} of the workspace: "
                       f"{o['text']} (`GET /api/maps/<map>`).")
        else:  # an output a future mode writes through one of its parameters
            out.append(f"* {code(o['name'])}{cond} ({o['media_type']}), written through "
                       f"{code(by_flag.get(o['via'], o['via']))}: {o['text']}.")
    return out


def condition(d: Json) -> str:
    return ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in (d["inference_condition"] or {}).items())


def inference_text(d: Json) -> str:
    if d["inference"] == "never":
        return f"**Works without the inference server** ({d['inference_text']}), at once."
    if d["inference"] == "conditional":
        return (f"**Needs the inference server** {d['inference_text']} ({condition(d)}): then it "
                "waits for its turn.")
    return f"**Needs the inference server** ({d['inference_text']}): it waits for its turn."


def effect_text(op: Operation) -> str:
    writes = op.writes_map()
    if writes is not None:
        what = next(o.text for o in op.mode.outputs if o.via == writes.flag)
        return (f"**Writes the map** named by {code(writes.name)} ({what}): ask the user first "
                "(see Safety).")
    return "Read-only: the result is the response; nothing is kept."


def run_curl(op: Operation, params: Json) -> str:
    return (f"curl -sS -X POST -o {result_file(op, params)} -D {HEADERS} "
            f"-w '%{{http_code}}\\n' \"$BASE/api/ops/{op.id}\" {JSON_POST} -d '{compact(params)}'")


def operation_section(doc: Json, op: Operation) -> list[str]:
    d = entry(doc, op)
    params = sample_params(d)
    errors = ", ".join(f"{code(e['code'])} ({e['http_status']})" for e in d["errors"])
    refused = refusal(op)
    lines = [f"### `{op.id}` — `{op.label}`", "",
             f"`POST /api/ops/{op.id}` · `POST /api/ops/{op.id}/validate`. "
             f"{d['description'][:1].upper()}{d['description'][1:].rstrip('.')}. "
             f"{inference_text(d)} {effect_text(op)}", "",
             *param_table(d), *attrs_table(d), "", "Answers with:", "", *outputs_text(op, d), "",
             f"Stages (`Server-Timing`): {chain(d['stages'])}. Checks: "
             + "; ".join(r["text"] for r in d["rules"]) + f". Refused with: {errors}; a run can "
             "end with any code of the table in Errors.", "",
             sh(f"curl -sS -X POST \"$BASE/api/ops/{op.id}/validate\" {JSON_POST} "
                f"-d '{compact(params)}'", run_curl(op, params)), "",
             f"→ validate: `{validated(op, d, params)}`", "",
             f"→ run: {ok(op, params, d)} (see Request workflow).", ""]
    if refused:
        lines += [f"→ refused, e.g. the body `{{}}`: {refused}", ""]
    return lines


# -- the template ------------------------------------------------------------------------------------


def description(ops: dict[str, Operation]) -> str:
    """What the skill can do and when to use it (spec §2.7 "Description"): every operation with
    what it produces, the service's other features, and the conditions (a running service; the
    operations that need the inference server). Within the Agent Skills limit: when more
    commands would not fit, the subcommands' help and then the formats are left out."""
    def made(op: Operation) -> str:
        return "/".join(dict.fromkeys(FORMATS.get(o["format"], o["format"])
                                      for o in op.entry(described[op.label])["outputs"]))

    def clause(p: spec.Program, helps: bool, formats: bool) -> str:
        """The command's operations, each run of those that produce the same formats followed
        by them."""
        mine = [op for op in ops.values() if op.program is p]
        parts = []
        for i, op in enumerate(mine):
            parts.append(op.id + (f" ({op.command.help})" if helps and op.command.name else ""))
            if formats and (i + 1 == len(mine) or made(mine[i + 1]) != made(op)):
                parts[-1] += f" → {made(op)}"
        return f"{p.prog} {p.description.rstrip('.')}: {', '.join(parts)}"

    def ids(inference: str) -> str:
        return ", ".join(op.id for op in ops.values() if op.mode.inference == inference)

    described = {d["id"]: d for d in spec.describe()["operations"]}
    never, sometimes = ids("never"), ids("conditional")
    inference = (f"Inference server needed except for {never}" if never
                 else "Every operation needs the inference server") + (
        f" ({sometimes}: at times)." if sometimes else ".")
    for helps, formats in ((True, True), (False, True), (False, False)):
        programs = "; ".join(clause(p, helps, formats) for p in spec.PROGRAMS
                             if any(op.program is p for op in ops.values()))
        text = ("Use the oh-my-slam web service (server.sh) with sh and curl, from its Mac or the "
                "LAN. Operations, each answered when its command ends with the command's own "
                f"result and stage timings (JSON = OpenLABEL scene): {programs}. Also: "
                + "; ".join(words for _, words in FEATURES) + ". Use when the user asks for any "
                f"of these and a server.sh is running (its URL, else ask). {inference}")
        if len(text) <= DESCRIPTION_MAX:
            return text
    raise ValueError(f"the description has {len(text)} characters (Agent Skills: at most "
                     f"{DESCRIPTION_MAX}); shorten FEATURES or the commands' descriptions")


def front_matter(ops: dict[str, Operation]) -> str:
    """The skill's name and description (a YAML double-quoted scalar: JSON is valid YAML)."""
    return (f"---\nname: {NAME}\ndescription: {json.dumps(description(ops), ensure_ascii=False)}"
            "\n---\n")


INTRO = """# oh-my-slam API

`server.sh` is the oh-my-slam web service: a long-lived HTTP service on a Mac that runs every
mode of {{progs}}. There are no jobs: each operation runs within its own HTTP request, and the
service answers when the command ends, with the command's own result. It keeps maps and uploads
in a workspace (`~/oh-my-slam-data/` unless it was started with `--data <folder>`), keeps no
results, and serves a browser application at its root URL. It binds `0.0.0.0`, so it is reachable
from the Mac itself and from any Linux or macOS machine on the LAN; you need only `sh` and `curl`.
Responses are JSON unless noted.

This file is generated from the service's own definitions (`uv run python -m
oh_my_slam.web.skill` in the repository). The running service describes itself at
`$BASE/api/openapi.json`, with each operation's full definition under `x-oms`: where that
document and this file disagree (an older or newer service), **the service's document wins**.

| Command | Operations | What it does |
|---|---|---|
{{programs}}
"""


FIND = """## Find the service

`server.sh` binds a free port unless it was started with `--port <n>`, so its URL changes from run
to run. Run this snippet once: it prints the base URL on stdout and caches it in
`{{cache}}`.

It tries, in order: the URL the user gives (`OMS_URL`); the cached URL, if `/api/health` answers
there within 3 s; on the Mac that runs the service, the URL the service records in `server.json`
in its workspace (`~/oh-my-slam-data/`, or the `--data` folder the user names, as `OMS_DATA`),
with `0.0.0.0` replaced by `127.0.0.1`. Otherwise it fails: then ask the user for the URL that
`server.sh` printed when it started (`server.sh: listening on http://0.0.0.0:<port>/`; from
another machine, the Mac's address or host name in place of `0.0.0.0`) or that
`server.sh --status` reports (`service.url`). There is no network or port scan, since the port is
not known in advance; `server.sh --port <n>` keeps the URL stable for other machines.

Change the first line to `OMS_URL=<url> sh <<'EOF'` when the user gives a URL, or to
`OMS_DATA=<folder> sh <<'EOF'` when they name the service's `--data` folder.

```sh
sh <<'EOF'
{{snippet}}EOF
```

Your shell keeps no variables between commands, so begin **every** command with
`BASE=<that URL>;` (a `;`, not a `BASE=... curl` prefix), e.g.
`BASE=http://127.0.0.1:52026; curl -sS "$BASE/api/health"`. All examples below use
`"$BASE/..."`. Use the URL as found: the service answers 403 `forbidden` to a `Host` that does
not name its machine. `GET /api/health` names the service's workspace (`service.data`): check it
is the one the user means. If a request later fails to connect (curl exit 7 or 28), run the
snippet again: it searches again only when the cached URL stops answering.
"""

SAFETY = """## Safety

* **Inference server down.** These need the inference server, a separate process on the Mac:
  {{inference_ops}}. When it is down they are refused with 503 `server_unavailable`, and
  `GET /api/health` says so in `inference` (`status`, `message`, `start_command`). Report the
  message and the start command it gives (`{{start}}`) to the user instead of retrying, and offer
  what still works: {{no_inference}}, and every read-only endpoint (health, maps).
* **Never start or stop `server.sh` or the inference server**, and never kill their processes:
  the user runs them.
* **Never write into a map's folder** (`<data>/maps/<name>/`, not even through the shell on the
  Mac): maps change only through {{writers}}.
* **Ask the user first** before updating an existing map ({{writers}} with the name of a map that
  `GET /api/maps/<name>` finds) and before starting a long mapping request ({{writers}} on new
  inputs runs {{writer_stages}} and can take many minutes). Say what follows when you ask: an
  update changes the map for good.
* **Wait for the answer with no client timeout** (no `--max-time`, no `-m`): disconnecting
  interrupts the command, as Ctrl-C would — there is no result, and an interrupted map update
  leaves the map as it was. Requests that use the inference server run one at a time, in arrival
  order, each waiting for its turn with its connection open: send them one by one, not in bulk.
* **Inputs are uploads or paths inside the workspace** (relative to it, such as
  `uploads/<upload>/photo.jpg`), never paths outside it, which are refused (400). A file on your
  machine reaches the service only as an upload.
* **An upload is consumed by the one request it is given to** and deleted when that request ends,
  whatever its outcome (refused, failed, interrupted or done), so repeating a request means
  uploading again. Validating does not consume it. Delete an upload you will not use.
* **Results are the command's own bytes:** save them with `-o`, never re-serialise or edit them.
  Report a failed request's `error.message` as it is: it is the command's message.
* Don't repeat a refused or failed request unchanged.
"""

ERRORS = """## Errors

Every error is

```json
{"error": {"code": "<code>", "message": "<the command's own message>", "http_status": <status>}}
```

A refused request (and validate's `problems`) adds `rule`, `parameters` (those it concerns),
`exit_code`, `problems` (every problem) and `by_parameter` (the messages per parameter). For
example, `POST /api/ops/{{op}}` with the body `{}` answers {{refusal}}. A request whose command ran
and failed answers its `exit_code` with the command's message, e.g. {{failed}}. The code is the
name of the command's exit status, mapped to an HTTP status by one rule (input errors 4xx,
inference server unavailable 503, internal 500):

| Code | Exit status | HTTP |
|---|---|---|
{{codes}}

The service's own refusals have the same shape:

| Code | HTTP | When |
|---|---|---|
{{service}}
"""

WORKFLOW = """## Request workflow

Every operation runs within its own request: the service answers once the command has ended,
with its result. The example is `{{op}}` (`{{label}}`); every operation in Operations works the
same way.

1. **Check the service** and the inference server.
   ```sh
   curl -sS "$BASE/api/health"
   ```
   `inference.status` is one of {{statuses}}. An operation that needs inference is accepted
   while it is `ready` or `loading`; otherwise see Safety. `service.requests` counts the requests
   `running` and those `waiting` for their turn at the inference server.

2. **Upload each input file**, or skip this and name a path inside the workspace. The request
   body is the raw file, streamed with `-T` (not a form: `curl -F` is refused with 415). `name` is
   the file name the command sees: keep its suffix (it tells the type) and use only letters,
   digits, `.`, `_` and `-`.
   ```sh
   {{upload_curl}}
   ```
   → 201 `{{upload}}`

   The `path` is the parameter's value. A parameter that takes several files (a JSON list) takes
   one upload per file, listed in the order the command should read them. The request you give
   an upload to consumes it (see Safety).

3. **Validate** the request: the command's own checks, and the inference server's when the
   request needs it; nothing runs, and no upload is consumed.
   ```sh
   curl -sS -X POST "$BASE/api/ops/{{op}}/validate" {{json}} -d '{{body}}'
   ```
   → `{{validated}}`

   With `"valid":false`, `problems` and `by_parameter` say what to change, in the command's
   words. `command` is the command line the request will run.

4. **Run it and wait for the answer**, with no client timeout. Nothing comes back until the
   command ends: a mapping request can take many minutes, and a request that uses the inference
   server may first wait for its turn. Disconnecting interrupts the command (see Safety). Save
   the body with `-o` and the headers with `-D`; `-w` prints the HTTP status.
   ```sh
   {{run_curl}}
   ```
   → {{ok}}: byte-identical to what the command writes to stdout, so never rewrite it. Any other
   status means the file holds the error of Errors instead.

5. **Read the stage timings** from the `Server-Timing` header: each of the command's own stages
   (here {{stages}}) under its own name, then `total`, in milliseconds.
   ```sh
   grep -i '^server-timing:' {{headers}}
   ```
   → `server-timing: {{timing}}`

6. **Report** the result file to the user, or the error's `message` as it is. A map's summary and
   its update history are at `GET /api/maps/<name>`, and the browser application at `$BASE/`
   runs the same operations.
"""

OPERATIONS = """## Operations

Each operation is one mode of a command, with one parameter per option: the same names, values,
defaults and checks. The options that only choose where the command writes (`-o`, `-d`) are no
parameters: the response is the result, and the files a command writes to a folder are not
offered.

### `POST /api/ops/{op}` · `POST /api/ops/{op}/validate`

`POST /api/ops/<op>` runs the operation and answers when its command ends (200, the result, with
`Server-Timing`); `POST /api/ops/<op>/validate` runs the same checks and nothing else (200,
`{valid, command, inference, problems, by_parameter}`). The body is a JSON object of parameters,
sent with `Content-Type: application/json`; a parameter left out takes its default. Path
parameters name an upload (`uploads/<upload>/<file>`) or another path inside the workspace; a map
is `<name>` or `maps/<name>`. Placeholders such as `<upload>` and `<map>` stand for values the API
returns or the user names. An operation or parameter this file does not list, or one it lists that
the service refuses as unknown: read `$BASE/api/openapi.json`, which wins.
"""


def endpoint(method: str, path: str, effect: str, text: str, curl: str | None = None,
             sample: str | None = None, also: tuple[str, ...] = ()) -> Json:
    """One endpoint of the template. ``sample``: a JSON answer (code), else text (a line break
    makes it a block)."""
    return {"method": method, "path": path, "effect": effect, "text": text, "curl": curl,
            "sample": sample, "also": also}


def fixed_endpoints() -> list[Json]:
    """The service's own endpoints (not per command), with samples from its own records and
    constants."""
    health = {"status": "ok", "service": {
        "version": "<version>", "url": "http://0.0.0.0:52026/", "pid": 4321,
        "workspace": "oh-my-slam-data", "data": "/Users/<user>/oh-my-slam-data",
        "started_at": CREATED, "requests": {"running": 1, "waiting": 0},
        "in_progress": [{"operation": "segment-image",
                         "command": ["segment.sh", "-i=uploads/9f2c4e1a7b3d5f60/photo.jpg"],
                         "state": "running", "arrived_at": CREATED + 2.0,
                         "started_at": CREATED + 2.0}]},
        "inference": {"status": "down", "message": "<why>", "start_command": START_COMMAND}}
    return [
        endpoint("GET", "/api/health", "Read-only.",
                 "The service (version, URL, workspace, the requests `running` and `waiting` "
                 "for their turn, and `in_progress`: each one's operation, command line, state "
                 "and times, in arrival order) and the inference server: `inference.status` is one of "
                 f"{statuses()}; `start_command` is the command that starts it when it is not "
                 "`ready` (report it, never run it), and `health` its own health when it answers.",
                 'curl -sS "$BASE/api/health"', compact(health)),
        endpoint("GET", "/api/openapi.json", "Read-only.",
                 "The running service's OpenAPI 3.1 document: every operation with its parameters, "
                 "and its whole definition under `x-oms` (parameters, outputs, rules, errors, "
                 "stages, inference need). It wins where it disagrees with this file.",
                 'curl -sS -o openapi.json "$BASE/api/openapi.json"',
                 '{"openapi":"3.1.0","info":{…},"paths":{"/api/ops/<op>":{"post":{…,"x-oms":{…}}}'
                 ',…},"components":{…},"x-oms":{"exit_codes":[…],"stages":[…]}}'),
        endpoint("POST", "/api/uploads",
                 "Stores a file in the workspace until the request it is given to ends.",
                 "The request body is the file itself (`-T`), with its own media type or "
                 "`application/octet-stream`; a form (`curl -F`, multipart) is refused with 415, "
                 "since a cross-site web page could send one. At most "
                 f"{gib(MAX_UPLOAD_BYTES)}, and it must leave {gib(MIN_FREE_BYTES)} free (413). "
                 "Answers 201; use its `path` as an input parameter of one request.",
                 "curl -sS -X POST -T photo.jpg -H 'Content-Type: application/octet-stream' "
                 '"$BASE/api/uploads?name=photo.jpg"',
                 compact(SAMPLE_UPLOAD.describe(Path("/<data>")))),
        endpoint("DELETE", "/api/uploads/{id}", "Deletes an upload.",
                 "Only one that no request in progress was given (409 `upload_in_use` "
                 "otherwise: it goes when that request ends); 404 `not_found` for no such upload.",
                 'curl -sS -X DELETE -w \'%{http_code}\\n\' "$BASE/api/uploads/<upload>"',
                 "204, no body"),
        endpoint("GET", "/api/maps", "Read-only.",
                 "Every map of the workspace with summary figures from its own metadata: each "
                 "scalar of its `map.json`, the size of each of its lists (`<key>_count`), "
                 "`frames`, `objects` and `last_update`.",
                 'curl -sS "$BASE/api/maps"',
                 '[{"name":"<map>","path":"maps/<map>",…,"frames":302,"objects":143,'
                 '"last_update":{"id":2,"at":1790931612.9,"kind":"video","frames_added":120,'
                 '"total_s":512.4}},…]'),
        endpoint("GET", "/api/maps/{name}", "Read-only.",
                 "One map's summary plus `meta`, its whole `map.json`, whose `updates[]` hold each "
                 "update's record with its `timings`. 404 `not_found` when there is no such map.",
                 'curl -sS "$BASE/api/maps/<map>"',
                 '{"name":"<map>","path":"maps/<map>",…,"meta":{…,"updates":[{…,"timings":{…}},'
                 '…],…}}'),
    ]


GROUPS: tuple[tuple[str, Callable[[str], bool]], ...] = (  # the rest: "Service"
    ("Uploads", lambda p: p.startswith("/api/uploads")),
    ("Maps (read-only)", lambda p: p.startswith("/api/maps")),
)


def endpoints_section(doc: Json) -> list[str]:
    """Every endpoint of the OpenAPI document besides the operations (and the document itself),
    from its template entry, or a generic one built from the document."""
    template = {f"{e['method']} {e['path']}": e for e in fixed_endpoints()}
    found: dict[str, Json] = {}
    for path, item in doc["paths"].items():
        if path.startswith("/api/ops/"):
            continue
        for method, op in item.items():
            key = f"{method.upper()} {path}"
            e = template.get(key) or endpoint(
                method.upper(), path, "Read-only." if method == "get" else "Changes state.",
                op.get("summary", "") + ".", "curl -sS" + ("" if method == "get" else
                                                          f" -X {method.upper()}") + ' "$BASE'
                + path.replace("{", "<").replace("}", ">") + '"')
            found[key] = {**e, "parameters": op.get("parameters", [])}
    found["GET /api/openapi.json"] = {**template["GET /api/openapi.json"], "parameters": []}
    lines = ["## Endpoints", "",
             "The service's own endpoints, besides the operations. In paths, `{id}` is an upload's "
             "id and `{name}` a map's name. They answer an error of Errors when they fail, except "
             "a path or method the service does not have, which answers plain text (404 "
             "`Not Found`, 405 `Method Not Allowed`)."]
    group = {key: next((t for t, member in GROUPS if member(e["path"])), "Service")
             for key, e in found.items()}
    for title in ("Service", *(t for t, _ in GROUPS)):
        lines += ["", f"### {title}"]
        for key, e in found.items():
            if group[key] != title:
                continue
            query = [f"`{p['name']}`" + ("" if p.get("required") else " (optional)")
                     + (f": {p['description']}" if p.get("description") else "")
                     for p in e["parameters"] if p["in"] == "query"]
            lines += ["", "#### " + " · ".join(code(h) for h in (key, *e["also"])), "",
                      f"{e['effect']} {e['text']}" + (" Query: " + "; ".join(query) + "."
                                                       if query else "")]
            if e["curl"]:
                lines += ["", sh(e["curl"])]
            sample = e["sample"]
            if sample and "\n" in sample:
                lines += ["", "```text\n" + sample + "\n```"]
            elif sample and sample.startswith(("{", "[")):
                lines += ["", f"→ `{sample}`"]
            elif sample:
                lines += ["", f"→ {sample}"]
    return lines


def example(ops: dict[str, Operation]) -> Operation:
    """The workflow's example: an operation on one image (else the first)."""
    return next((op for op in ops.values() if any(o.kind is Kind.IMAGE for o in op.options)),
                next(iter(ops.values())))


def render(ops: dict[str, Operation] | None = None) -> str:
    """The whole ``SKILL.md``."""
    ops = operations() if ops is None else ops
    doc = openapi.document(ops)
    entries = {op.id: entry(doc, op) for op in ops.values()}

    def ids(pick: Callable[[Operation], bool]) -> str:
        return ", ".join(code(op.id) for op in ops.values() if pick(op))

    def of(p: spec.Program) -> Callable[[Operation], bool]:
        return lambda op: op.program is p

    offered = [p for p in spec.PROGRAMS if any(op.program is p for op in ops.values())]
    programs = [f"| `{p.prog}` | {ids(of(p))} | {p.description} |" for p in offered]
    progs = [code(p.prog) for p in offered]
    writers = [op for op in ops.values() if op.writes_map() is not None]
    stages = entries[writers[0].id]["stages"] if writers else []
    writer_stages = f"{len(stages)} stages, from `{stages[0]}` to `{stages[-1]}`," if stages \
        else "its stages"
    no_inference = [ids(lambda op: entries[op.id]["inference"] == "never")] + [
        f"{code(op.id)} (it needs the server {entries[op.id]['inference_text']}: "
        f"{condition(entries[op.id])})" for op in ops.values()
        if entries[op.id]["inference"] == "conditional"]
    ex = example(ops)
    d = entries[ex.id]
    params = sample_params(d)
    upload = SAMPLE_UPLOAD.describe(Path("/<data>"))
    parts = [
        front_matter(ops),
        fill(INTRO, progs=", ".join(progs[:-1]) + (" and " if len(progs) > 1 else "") + progs[-1],
             programs="\n".join(programs)),
        fill(FIND, cache=CACHE, snippet=ADDRESS_SNIPPET),
        fill(SAFETY, inference_ops=ids(lambda op: entries[op.id]["inference"] == "required"),
             start=START_COMMAND, no_inference=", ".join(filter(None, no_inference)) or "nothing",
             writers=ids(lambda op: op in writers) or "the mapping operation",
             writer_stages=writer_stages),
        fill(ERRORS, op=ex.id, refusal=refusal(ex) or "the command's refusal",
             failed=failed(ExitCode.NOT_REGISTERED), codes="\n".join(
                 f"| `{c['code']}` | {c['exit_code']} | {c['http_status']} |"
                 for c in doc["x-oms"]["exit_codes"]),
             service="\n".join(f"| `{k}` | {status} | {cell(when)} |"
                               for k, (status, when) in SERVICE_ERRORS.items())),
        fill(WORKFLOW, op=ex.id, label=ex.label, statuses=statuses(), json=JSON_POST,
             upload_curl=f"curl -sS -X POST -T {upload['name']} -H 'Content-Type: "
                         f"application/octet-stream' \"$BASE/api/uploads?name={upload['name']}\"",
             upload=compact(upload), body=compact(params), validated=validated(ex, d, params),
             run_curl=run_curl(ex, params), ok=ok(ex, params, d), stages=chain(d["stages"]),
             headers=HEADERS, timing=server_timing(d["stages"])),
        OPERATIONS,
        *("\n".join(operation_section(doc, op)) for op in ops.values()),
        "\n".join(endpoints_section(doc)),
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
