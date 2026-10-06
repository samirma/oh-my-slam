"""The agent skill ``SKILL.md`` at the repository root (spec §2.7): one self-contained file that lets
an AI agent use the ``server.sh`` API with ``sh`` and ``curl`` alone.

Like ``/api/openapi.json`` it is generated, never hand-edited: the operations, their parameters,
defaults, allowed values, checks, result files, stages and error codes come from
``openapi.document`` (the commands' definitions, ``spec.describe()``), and the samples from the
service's own code — the job and upload records (``jobs.Job``, ``workspace.Upload``), the command
line and the refusal the commands' parser gives (``spec.argv_of``, ``spec.dry_run``,
``operations.error_body``), the timing record and its summary line (``core.timing``) — so a
command that gains, changes or loses an option, an output or an error changes the skill with no
hand edit. What concerns no single command (finding the service, the job workflow, safety, the
service's own endpoints and refusals) is the template below; an endpoint added to the OpenAPI
document without a template entry still gets a generic one. ``tests/unit/test_agent_skill.py``
fails when the committed file differs from this output or misses a route of the app.

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
from oh_my_slam.core import timing
from oh_my_slam.web import openapi
from oh_my_slam.web.app import MAX_UPLOAD_BYTES, MIN_FREE_BYTES, START_COMMAND
from oh_my_slam.web.jobs import Job
from oh_my_slam.web.operations import (
    OUT_DIR,
    PATH_IN,
    Operation,
    _result_format,
    error_body,
    operations,
)
from oh_my_slam.web.workspace import UPLOADS, Upload

Json = dict[str, Any]
NAME = "oh-my-slam-api"
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "SKILL.md"
CACHE = "${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam-api/server_url"
CREATED = 1791281700.0  # the samples' clock
FOLDER = "art"  # the samples' folder name (-d)
SAMPLE_UPLOAD = Upload("<upload>", "photo.jpg", Path(f"/<data>/{UPLOADS}/<upload>/photo.jpg"),
                       2481152)
JSON_POST = "-H 'Content-Type: application/json'"


def gib(n: int) -> str:
    return f"{n / 2**30:g} GiB"


# The service's own refusals (no command's): code → (HTTP status, when). Their shape is that of
# every error; tests/unit/test_agent_skill.py checks that each code oh_my_slam.web answers with
# is here or in the commands' exit-code table.
SERVICE_ERRORS: dict[str, tuple[int, str]] = {
    "not_found": (404, "no such map, job, upload, file or operation, or a job without that result"),
    "forbidden": (403, "a `Host` that does not name the service's machine, or a foreign `Origin`"),
    "unsupported_media_type": (415, "a POST whose body is not `application/json`, or an upload "
                                    "sent as a form (`curl -F`)"),
    "too_large": (413, f"an upload over {gib(MAX_UPLOAD_BYTES)}"),
    "insufficient_storage": (413, "an upload that would leave less than "
                                  f"{gib(MIN_FREE_BYTES)} free on the workspace's disk"),
    "upload_in_use": (409, "an upload that is already the input of a queued or running job"),
    "not_cancellable": (409, "cancelling a job that has ended"),
    "gone": (410, "re-submitting a job whose operation the commands no longer offer"),
    "stopping": (503, "a submission while the service shuts down"),
}

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


def brief(obj: Json, *keys: str) -> str:
    """``obj`` as compact JSON with only ``keys`` (each must exist: a renamed field fails here)."""
    return "{" + ",".join(f"{json.dumps(k)}:{compact(obj[k])}" for k in keys) + ",…}"


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
    """The operation's ``spec.describe()`` entry, as the OpenAPI document carries it."""
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


def sample_params(op: Operation, d: Json, folder: bool = False) -> Json:
    """The required parameters of ``op`` with placeholder values (and its folder when
    ``folder``)."""
    params = {p["name"]: sample_value(p) for p in d["parameters"]
              if p["required"] and p["service"]}
    if folder:
        params.update({o.name: FOLDER for o in op.options if o.kind is Kind.FOLDER_OUT})
    return params


def shown(op: Operation, params: Json) -> tuple[Json, str | None]:
    """The parameters as a job records them (a map as ``maps/<name>``, the result's default name)
    and the result's file name."""
    out = {k: (f"maps/{v}" if op.command.option(k).kind is Kind.MAP else v)
           for k, v in params.items()}
    file_out = next((o for o in op.options if o.kind is Kind.FILE_OUT), None)
    if file_out is None:
        return out, None
    out[file_out.name] = "result" + spec.suffix_of(_result_format(op, params) or "")
    return out, out[file_out.name]


def command_of(op: Operation, params: Json) -> list[str]:
    return [op.program.prog, *spec.argv_of(op.command, op.mode, shown(op, params)[0])]


def validated(op: Operation, d: Json, params: Json) -> str:
    return compact({"valid": True, "command": command_of(op, params),
                    "inference": d["inference"] == "required", "problems": [],
                    "by_parameter": {}})


def refusal(op: Operation) -> str | None:
    """What a submission of ``op`` with an empty body answers: the refusal of the command's own
    parser, with its status (``problems`` repeats every problem, shortened here)."""
    problems = spec.dry_run(op.command, op.mode, {})
    if not problems:
        return None
    status, body = error_body(problems)
    err = ",".join(f"{json.dumps(k)}:" + ("[…]" if k == "problems" else compact(v))
                   for k, v in body["error"].items())
    return f"{status} `{{\"error\":{{{err}}}}}`"


def sample_job(op: Operation, d: Json, params: Json, **state: Any) -> Json:
    """The job a submission of ``params`` answers, from the service's own ``Job`` record."""
    recorded, result = shown(op, params)
    writes = op.writes_map()
    job = Job(id="<job>", operation=op.id, label=op.label, params=params,
              command=command_of(op, params), steps=[], inference=d["inference"] != "never",
              conditional=d["inference"] == "conditional",
              uploads=["<upload>"] if f"{UPLOADS}/<upload>/" in json.dumps(params) else [],
              writes=f"<data>/{recorded[writes.name]}" if writes else None, result_name=result,
              result_format=_result_format(op, params) if result else None, created_at=CREATED,
              **state)
    return job.public()


def succeeded(d: Json) -> Json:
    return {"state": "succeeded", "started_at": CREATED + 1, "ended_at": CREATED + 5,
            "stage": d["stages"][-1] if d["stages"] else None, "exit_code": 0,
            "stages": [{"stage": s, "seconds": 0.5} for s in d["stages"]]}


def files_of(op: Operation, d: Json) -> list[Json]:
    """The files a job of ``op`` writes with its sample parameters (and folder): its result and
    its folder outputs."""
    folders = {o.flag for o in op.options if o.kind is Kind.FOLDER_OUT}
    params = sample_params(op, d, folder=True)
    _, result = shown(op, params)
    media = {f"{FOLDER}/{o['name']}": o["media_type"] for o in d["outputs"] if o["via"] in folders}
    if result:
        media[result] = spec.media_type(_result_format(op, params) or "json")
    return [{"path": n, "size": 1024, "media_type": media[n], "url": f"/api/jobs/<job>/files/{n}"}
            for n in sorted(media)]


def timings_of(label: str, stages: list[str]) -> tuple[str, str]:
    """A job's timing record and its log's summary line, as the command's ``core.timing`` writes
    them (sample figures)."""
    fields = list(timing.Timings(sample_every=None).to_dict())
    figures: Json = {"total_s": 3.0, "stages_s": dict.fromkeys(stages, 0.5), "server": {},
                     "peak_rss_mb": {"self": 1234.0, "children": 0.0}}
    record = "{" + ",".join([f'"command":{compact(label)}', "…", *(
        f"{json.dumps(k)}:" + (compact(figures[k]) if k in ("total_s", "stages_s") else "…")
        for k in fields)]) + "}"
    return record, timing.summary_line(figures)


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


def param_type(p: Json, d: Json) -> str:
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
    if kind == str(Kind.FILE_OUT):
        names = sorted({"result" + spec.suffix_of(o["format"]) for o in d["outputs"]
                        if o["name"] == "result"})
        return (f"plain file name in the job's `{OUT_DIR}/` (default "
                + " or ".join(code(n) for n in names) + ", by the result's format)")
    if kind == str(Kind.FOLDER_OUT):
        return f"plain folder name in the job's `{OUT_DIR}/` (none: no files)"
    what = "list of workspace paths" if p["multiple"] else "workspace path"
    return what + (", in order (the order matters)" if p["ordered"] else "") + (
        ": " + " ".join(p["accepts"]) if p["accepts"] else "")


def param_table(d: Json) -> list[str]:
    rows = ["| Parameter | Flag | Value | Default | Meaning |", "|---|---|---|---|---|"]
    for p in d["parameters"]:
        if not p["service"]:
            continue
        default = p["default"]
        shown_default = "" if default is None else code(
            default if isinstance(default, str) else json.dumps(default))
        meaning = p["help"] + (f" ({p['applies_text']})" if p["applies_text"] else "")
        rows.append(f"| {code(p['name'])}{' (required)' if p['required'] else ''} "
                    f"| {code(p['flag'])} | {cell(param_type(p, d))} | {cell(shown_default)} "
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
        if o["via"] == "stdout":
            out.append(f"* The result{cond} (`GET /api/jobs/<job>/result`, {o['media_type']}): "
                       f"{o['text']}.")
        elif o["via"] == "browser":
            out.append("* The viewer page: the finished job's `viewer` "
                       "(`$BASE/viewer/job/<job>/`); give it to the user.")
        elif o["format"] == "map":
            out.append(f"* The map {code(by_flag.get(o['via'], o['via']))} of the workspace: "
                       f"{o['text']} (`GET /api/maps/<map>`).")
        else:
            folder = by_flag.get(o["via"], o["via"])
            regions = (" Each pixel is painted in its object's colour, so the object under a "
                       "pixel is the one of that colour.") if o["object_regions"] else ""
            out.append(f"* `<{folder}>/{o['name']}`{cond} (`GET /api/jobs/<job>/files/<{folder}>/"
                       f"{o['name']}`, {o['media_type']}): {o['text']}.{regions}")
    return out


def condition(d: Json) -> str:
    return ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in (d["inference_condition"] or {}).items())


def inference_text(d: Json) -> str:
    if d["inference"] == "never":
        return f"**Works without the inference server** ({d['inference_text']})."
    if d["inference"] == "conditional":
        return f"**Needs the inference server** {d['inference_text']} ({condition(d)})."
    return f"**Needs the inference server** ({d['inference_text']})."


def effect_text(op: Operation) -> str:
    writes = op.writes_map()
    if writes is not None:
        return (f"**Writes the map** named by {code(writes.name)} (creates or extends it): ask the "
                "user first (see Safety).")
    if op.browser:
        return "Read-only: saves a viewer in the job's own folder."
    return "Read-only: writes only the job's own files."


def operation_section(doc: Json, op: Operation) -> list[str]:
    post = doc["paths"][f"/api/ops/{op.id}"]["post"]
    d = post["x-oms"]
    params = sample_params(op, d)
    body = compact(params)
    viewer = [p for p in post["parameters"] if p["name"] == "viewer"]
    errors = ", ".join(f"{code(e['code'])} ({e['http_status']})" for e in d["errors"])
    refused = refusal(op)
    lines = [f"### `{op.id}` — `{op.label}`", "",
             f"`POST /api/ops/{op.id}` · `POST /api/ops/{op.id}/validate`. "
             f"{d['description'][:1].upper()}{d['description'][1:].rstrip('.')}. "
             f"{inference_text(d)} {effect_text(op)}", "",
             *param_table(d), *attrs_table(d), "", "Produces:", "", *outputs_text(op, d), "",
             f"Stages: {chain(d['stages'])}. Checks: "
             + "; ".join(r["text"] for r in d["rules"]) + f". Refused with: {errors}; a job can "
             "end with any code of the table in Errors.", ""]
    if viewer:
        lines += [f"`POST /api/ops/{op.id}?viewer=true`: {viewer[0]['description']}.", ""]
    lines += [sh(f"curl -sS -X POST \"$BASE/api/ops/{op.id}/validate\" {JSON_POST} -d '{body}'",
                 f"curl -sS -X POST \"$BASE/api/ops/{op.id}\" {JSON_POST} -d '{body}'"), "",
              f"→ validate: `{validated(op, d, params)}`", "",
              f"→ submit: 202, `Location: /api/jobs/<job>`, the job "
              f"`{brief(sample_job(op, d, params), 'id', 'operation', 'label')}` "
              "(see Job workflow).", ""]
    if refused:
        lines += [f"→ refused, e.g. the body `{{}}`: {refused}", ""]
    return lines


# -- the template ------------------------------------------------------------------------------------


def front_matter(ops: dict[str, Operation]) -> str:
    """The skill's name and description (a YAML double-quoted scalar: JSON is valid YAML)."""
    programs = "; ".join(
        f"{p.prog} ({', '.join(op.id for op in ops.values() if op.program is p)}): "
        f"{p.description.rstrip('.')}" for p in spec.PROGRAMS
        if any(op.program is p for op in ops.values()))
    description = (
        "Use the oh-my-slam web service (server.sh) through its HTTP API with sh and curl only, "
        "from the Mac that runs it or another machine on the LAN. Every mode of its commands is "
        f"an operation run as a job. {programs}. Use when the user asks to run these through "
        "the service, or to list, inspect or download its maps, jobs and results.")
    return f"---\nname: {NAME}\ndescription: {json.dumps(description, ensure_ascii=False)}\n---\n"


INTRO = """# oh-my-slam API

`server.sh` is the oh-my-slam web service: a long-lived HTTP service on a Mac that runs every
mode of the oh-my-slam commands as a job, keeps maps, uploads and results in a workspace
(`~/oh-my-slam-data/` unless it was started with `--data <folder>`), and serves a browser
application at its root URL. It binds `0.0.0.0`, so it is reachable from the Mac itself and from
any Linux or macOS machine on the LAN; you need only `sh` and `curl`. Responses are JSON unless
noted.

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
  what still works: {{no_inference}}, and every read-only endpoint (maps, jobs, results, viewers).
* **Never start or stop `server.sh` or the inference server**, and never kill their processes:
  the user runs them.
* **Never write into a map's folder** (`<data>/maps/<name>/`, not even through the shell on the
  Mac): maps change only through {{writers}}. Map files are download-only.
* **Ask the user first** before updating an existing map ({{writers}} with the name of a map that
  `GET /api/maps/<name>` finds), before starting a long mapping job ({{writers}} on new inputs
  runs {{writer_stages}} and can take many minutes), and before cancelling a job you did not
  submit. Say what follows when you ask: an update changes the map for good; a cancelled job
  leaves no result.
* **Inputs are uploads or paths inside the workspace** (relative to it, such as
  `uploads/<upload>/photo.jpg`), never paths outside it, which are refused (400). A file on your
  machine reaches the service only as an upload.
* **An upload is consumed by one job** and deleted when that job ends, whatever its state, so
  submitting again (also `resubmit`) means uploading again. Delete an upload you will not submit.
* **Results are the command's own bytes:** save them with `-o`, never re-serialise or edit them.
  Report a failed job's `error.message` as it is: it is the command's message.
* Don't repeat a refused or failed request unchanged. Jobs that use the inference server run one
  at a time: submit them one by one, not in bulk.
"""

ERRORS = """## Errors

Every error is

```json
{"error": {"code": "<code>", "message": "<the command's own message>", "http_status": <status>}}
```

A refused operation (and validate's `problems`) adds `rule`, `parameters` (those it concerns),
`exit_code`, `problems` (every problem) and `by_parameter` (the messages per parameter). For
example, `POST /api/ops/{{op}}` with the body `{}` answers {{refusal}}. A job that failed or was
cancelled carries `code`, `exit_code`, `http_status` and `message` in its `error`. The code is the
name of the command's exit status, mapped to an HTTP status by one rule (input errors 4xx,
inference server unavailable 503, internal 500):

| Code | Exit status | HTTP | Job state |
|---|---|---|---|
{{codes}}

The service's own refusals have the same shape:

| Code | HTTP | When |
|---|---|---|
{{service}}
"""

WORKFLOW = """## Job workflow

Every operation runs as a job. The example is `{{op}}` (`{{label}}`); every operation in
Operations works the same way.

1. **Check the service** and the inference server.
   ```sh
   curl -sS "$BASE/api/health"
   ```
   `inference.status` is one of {{statuses}}. An operation that needs inference is accepted
   while it is `ready` or `loading`; otherwise see Safety.

2. **Upload each input file**, or skip this and name a path inside the workspace. The request
   body is the raw file, streamed with `-T` (not a form: `curl -F` is refused with 415). `name` is
   the file name the command sees: keep its suffix (it tells the type) and use only letters,
   digits, `.`, `_` and `-`.
   ```sh
   {{upload_curl}}
   ```
   → 201 `{{upload}}`

   The `path` is the parameter's value. A parameter that takes several files (a JSON list) takes
   one upload per file, listed in the order the command should read them.

3. **Validate** the request: the command's own checks; nothing is queued.
   ```sh
   curl -sS -X POST "$BASE/api/ops/{{op}}/validate" {{json}} -d '{{body}}'
   ```
   → `{{validated}}`

   With `"valid":false`, `problems` and `by_parameter` say what to change, in the command's
   words. `command` is the command line the job will run.

4. **Submit** the same body.
   ```sh
   curl -sS -X POST "$BASE/api/ops/{{op}}" {{json}} -d '{{body}}'
   ```
   → 202, `Location: /api/jobs/<job>`, the job:
   `{{queued}}`

   Keep its `id`. A refused submission answers the error of Errors and queues nothing.

5. **Follow the job** until its `state` is `succeeded`, `failed` or `cancelled`, by its event
   stream, which ends with the job (each `event: job` carries the whole job on its `data:` line,
   written with spaces: `"state": "running"`):
   ```sh
   curl -sS -N "$BASE/api/jobs/<job>/events"
   ```
   or by polling it every few seconds:
   ```sh
   BASE=<url>; JOB=<job>; while :; do s=$(curl -sS "$BASE/api/jobs/$JOB" | sed -n 's/.*"state":"\\([a-z]*\\)".*/\\1/p'); echo "$s"; case $s in queued|running) sleep 5 ;; *) break ;; esac; done
   ```
   While it runs, `stage` is the command's own timing stage (here {{stages}}) and `progress` is
   `{"stage","done","total"}` where the command knows its size. Jobs that use the inference
   server run one at a time in submission order, so a job may stay `queued` for a while.

6. **Download the result and each file** with `-o`. They are byte-identical to what the command
   writes, so never rewrite them. `-w` prints the HTTP status: anything but 200 means the file
   holds the error instead.
   ```sh
   curl -sS -o {{result}} -w '%{http_code}\\n' "$BASE/api/jobs/<job>/result"
   curl -sS "$BASE/api/jobs/<job>/files"
   ```
   → `{{files}}`

   Every file, under a folder named after the job:
   ```sh
   BASE=<url>; JOB=<job>; curl -sS "$BASE/api/jobs/$JOB/files" | tr '{}' '\\n\\n' | sed -n 's/.*"path":"\\([^"]*\\)".*/\\1/p' | while read -r p; do curl -sS --create-dirs -o "$JOB/$p" "$BASE/api/jobs/$JOB/files/$p"; done
   ```

7. **Report.** The finished job: `{{succeeded}}`. Its `stage` names and `stages` seconds are the
   command's own; `GET /api/jobs/<job>/timings` has the command's whole timing record and
   `GET /api/jobs/<job>/log` everything it printed. A `failed` job's `error` holds the command's
   message and code; a `cancelled` one has `error.code` `interrupted`.

{{viewers}}A map's viewer is always at `$BASE/viewer/map/<name>/`, and the browser application at
`$BASE/`: give the user those URLs to look at results.
"""

OPERATIONS = """## Operations

Each operation is one mode of a command, with one parameter per option: the same names, values,
defaults and checks.

### `POST /api/ops/{op}` · `POST /api/ops/{op}/validate`

`POST /api/ops/<op>` submits a job (202, the job); `POST /api/ops/<op>/validate` runs the same
checks and queues nothing (200, `{valid, command, inference, problems, by_parameter}`). The body
is a JSON object of parameters, sent with `Content-Type: application/json`; a parameter left out
takes its default. Path parameters name an upload (`uploads/<upload>/<file>`) or another path
inside the workspace; a map is `<name>` or `maps/<name>`; `output` and folder parameters are
plain names in the job's own folder, never paths. Placeholders such as `<upload>`, `<map>` and
`<job>` stand for values the API returns or the user names. An operation or parameter this file
does not list, or one it lists that the service refuses as unknown: read
`$BASE/api/openapi.json`, which wins.
"""

DOWNLOAD = ("200 and the file itself, saved as it is by `-o`; an error answers the JSON of Errors "
            "instead (`-w '%{http_code}'` shows which)")


def endpoint(method: str, path: str, effect: str, text: str, curl: str | None = None,
             sample: str | None = None, also: tuple[str, ...] = ()) -> Json:
    """One endpoint of the template. ``sample``: a JSON answer (code), else text (a line break
    makes it a block)."""
    return {"method": method, "path": path, "effect": effect, "text": text, "curl": curl,
            "sample": sample, "also": also}


def fixed_endpoints(ex: Operation, d: Json) -> list[Json]:
    """The service's own endpoints (not per command), with samples from its own records and
    constants and from the example operation ``ex``."""
    from oh_my_slam.viewer.bundle import display_transform

    params = sample_params(ex, d, folder=True)
    queued = sample_job(ex, d, params)
    done = sample_job(ex, d, params, **succeeded(d))
    again = {k: v for k, v in params.items() if f"{UPLOADS}/<upload>/" in json.dumps(v)}
    record, line = timings_of(ex.label, d["stages"])
    health = {"status": "ok", "service": {
        "version": "<version>", "url": "http://0.0.0.0:52026/", "pid": 4321,
        "workspace": "oh-my-slam-data", "data": "/Users/<user>/oh-my-slam-data",
        "started_at": CREATED, "jobs": {"queued": 0, "running": 0}},
        "inference": {"status": "down", "message": "<why>", "start_command": START_COMMAND}}
    event = "retry: 1000\n\nevent: job\nid: 3\ndata: {\"id\": \"<job>\", …, \"state\": " \
        "\"running\", \"stage\": \"" + (d["stages"] or ["<stage>"])[0] + "\", …}\n\n…"
    first = files_of(ex, d)[0]["path"]
    return [
        endpoint("GET", "/api/health", "Read-only.",
                 "The service (version, URL, workspace, queued and running jobs) and the inference "
                 f"server: `inference.status` is one of {statuses()}; `start_command` is the "
                 "command that starts it when it is not `ready` (report it, never run it), and "
                 "`health` its own health when it answers.",
                 'curl -sS "$BASE/api/health"', compact(health)),
        endpoint("GET", "/api/openapi.json", "Read-only.",
                 "The running service's OpenAPI 3.1 document: every operation with its parameters, "
                 "and its whole definition under `x-oms` (parameters, outputs, rules, errors, "
                 "stages, inference need). It wins where it disagrees with this file.",
                 'curl -sS -o openapi.json "$BASE/api/openapi.json"',
                 '{"openapi":"3.1.0","info":{…},"paths":{"/api/ops/<op>":{"post":{…,"x-oms":{…}}}'
                 ',…},"components":{…},"x-oms":{"exit_codes":[…],"stages":[…]}}'),
        endpoint("POST", "/api/uploads", "Stores a file in the workspace until its job ends.",
                 "The request body is the file itself (`-T`), with its own media type or "
                 "`application/octet-stream`; a form (`curl -F`, multipart) is refused with 415, "
                 "since a cross-site web page could send one. At most "
                 f"{gib(MAX_UPLOAD_BYTES)}, and it must leave {gib(MIN_FREE_BYTES)} free (413). "
                 "Answers 201; use its `path` as an input parameter.",
                 "curl -sS -X POST -T photo.jpg -H 'Content-Type: application/octet-stream' "
                 '"$BASE/api/uploads?name=photo.jpg"',
                 compact(SAMPLE_UPLOAD.describe(Path("/<data>")))),
        endpoint("DELETE", "/api/uploads/{id}", "Deletes an upload.",
                 "Only one that no queued or running job uses (409 `upload_in_use` otherwise: "
                 "cancel the job instead); 404 `not_found` for no such upload.",
                 'curl -sS -X DELETE -w \'%{http_code}\\n\' "$BASE/api/uploads/<upload>"',
                 "204, no body"),
        endpoint("GET", "/api/maps", "Read-only.",
                 "Every map of the workspace with summary figures from its own metadata: each "
                 "scalar of its `map.json`, the size of each of its lists (`<key>_count`), "
                 "`frames`, `objects`, `last_update` and `thumbnail`, a keyframe image of the map "
                 "(a path for `GET /api/maps/{name}/files/{path}`).",
                 'curl -sS "$BASE/api/maps"',
                 '[{"name":"<map>","path":"maps/<map>",…,"frames":302,"objects":143,'
                 '"last_update":{"id":2,"at":1790931612.9,"kind":"video","frames_added":120,'
                 '"total_s":512.4},"thumbnail":"<keyframe image>"},…]'),
        endpoint("GET", "/api/maps/{name}", "Read-only.",
                 "One map's summary plus `meta`, its whole `map.json`, whose `updates[]` hold each "
                 "update's record with its `timings`. 404 `not_found` when there is no such map.",
                 'curl -sS "$BASE/api/maps/<map>"',
                 '{"name":"<map>","path":"maps/<map>",…,"meta":{…,"updates":[{…,"timings":{…}},'
                 '…],…}}'),
        endpoint("GET", "/api/maps/{name}/files/{path}", "Read-only download.",
                 "A file of the map, such as `map.json` or the summary's `thumbnail`; hidden "
                 "entries are never served (404).",
                 "curl -sS -o map.json -w '%{http_code}\\n' "
                 '"$BASE/api/maps/<map>/files/map.json"', DOWNLOAD),
        endpoint("GET", "/api/jobs", "Read-only.",
                 "Every job, oldest first; the list survives a service restart.",
                 'curl -sS "$BASE/api/jobs"',
                 f"[{brief(done, 'id', 'operation', 'label', 'state')},…]"),
        endpoint("GET", "/api/jobs/events", "Read-only.",
                 "Server-sent events of every job's changes. It never ends by itself: bound it "
                 "with `--max-time` (curl then exits 28).",
                 'curl -sS -N --max-time 60 "$BASE/api/jobs/events"', event),
        endpoint("GET", "/api/jobs/{id}", "Read-only.",
                 "One job; poll it to follow the job. 404 `not_found` for no such job.",
                 'curl -sS "$BASE/api/jobs/<job>"',
                 brief(done, "id", "operation", "state", "stage", "progress", "stages", "error",
                       "result")),
        endpoint("GET", "/api/jobs/{id}/events", "Read-only.",
                 "Server-sent events of one job; the stream ends when the job ends, its last event "
                 "with the final state.",
                 'curl -sS -N "$BASE/api/jobs/<job>/events"', event),
        endpoint("POST", "/api/jobs/{id}/cancel",
                 "Stops a job: ask the user first unless you submitted it.",
                 "The effect of Ctrl-C on the command: a queued job is dropped, a running one is "
                 "interrupted (a cancelled map update leaves the map as it was), and there is no "
                 "result. 409 `not_cancellable` once the job has ended.",
                 f'curl -sS -X POST {JSON_POST} "$BASE/api/jobs/<job>/cancel"',
                 brief({**queued, "state": "running", "cancel_requested": True}, "id", "state",
                       "cancel_requested")),
        endpoint("POST", "/api/jobs/{id}/resubmit",
                 "Submits a new job (it needs the inference server when its operation does).",
                 "The same operation and parameters again; the body (a JSON object, `{}` for "
                 "none) replaces some. The old job's uploads were deleted when it ended: upload "
                 "the files again and pass their new paths. Answers 202 and the new job, or the "
                 "refusal of a submission; 410 `gone` when the operation no longer exists.",
                 f"curl -sS -X POST \"$BASE/api/jobs/<job>/resubmit\" {JSON_POST} "
                 f"-d '{compact(again)}'",
                 brief({**queued, "resubmitted_from": "<old job>"}, "id", "operation", "state",
                       "resubmitted_from")),
        endpoint("GET", "/api/jobs/{id}/result", "Read-only download.",
                 "The result, byte for byte the command's stdout or `-o` file. 404 `not_found` "
                 "until the job has succeeded, and for a viewer operation, which has none.",
                 f"curl -sS -o {queued['result_name']} -w '%{{http_code}}\\n' "
                 '"$BASE/api/jobs/<job>/result"', DOWNLOAD),
        endpoint("GET", "/api/jobs/{id}/files", "Read-only.",
                 f"Every file the job wrote in its `{OUT_DIR}/` folder.",
                 'curl -sS "$BASE/api/jobs/<job>/files"', compact(files_of(ex, d))),
        endpoint("GET", "/api/jobs/{id}/files/{path}", "Read-only download.",
                 "One file the job wrote, by its `path` in the list.",
                 f"curl -sS --create-dirs -o \"<job>/{first}\" -w '%{{http_code}}\\n' "
                 f'"$BASE/api/jobs/<job>/files/{first}"', DOWNLOAD),
        endpoint("GET", "/api/jobs/{id}/log", "Read-only.",
                 "Everything the command printed on stderr (text/plain), its `timings:` line "
                 "included.", 'curl -sS "$BASE/api/jobs/<job>/log"',
                 f"[oh-my-slam] …\n[oh-my-slam] {line}"),
        endpoint("GET", "/api/jobs/{id}/timings", "Read-only.",
                 "The command's timing record (per-stage seconds and memory, inference requests, "
                 "counts); `null` until it is written.",
                 'curl -sS "$BASE/api/jobs/<job>/timings"', record),
        endpoint("GET", "/api/maps/{name}/viewer/{path}", "Read-only.",
                 "The map's viewer: give the user `$BASE/viewer/map/<map>/`, the same viewer at "
                 "its page URL (`/api/maps/{name}/viewer` redirects to `…/viewer/`). Its data is "
                 "the viewer's own: `api/meta`, `api/scene` (the scene JSON), `api/catalog` and "
                 "`api/cloud?<attributes>` (binary).",
                 'curl -sS "$BASE/api/maps/<map>/viewer/api/meta"',
                 '{"mode": "map", "title": "<map>", …}', also=("GET /api/maps/{name}/viewer",)),
        endpoint("GET", "/api/jobs/{id}/viewer/{path}", "Read-only.",
                 "The viewer a job saved (its `viewer` field): give the user "
                 "`$BASE/viewer/job/<job>/` (`/api/jobs/{id}/viewer` redirects to `…/viewer/`).",
                 'curl -sS "$BASE/api/jobs/<job>/viewer/api/meta"', '{"mode": "image", …}',
                 also=("GET /api/jobs/{id}/viewer",)),
        endpoint("GET", "/viewer/map/{name}/{path}", "Read-only.",
                 "The page URL of a map's viewer, for the user's browser.",
                 "curl -sS -o /dev/null -w '%{http_code}\\n' \"$BASE/viewer/map/<map>/\"",
                 "200, the viewer's HTML page", also=("GET /viewer/map/{name}",)),
        endpoint("GET", "/viewer/job/{id}/{path}", "Read-only.",
                 "The page URL of a job's saved viewer, for the user's browser.",
                 "curl -sS -o /dev/null -w '%{http_code}\\n' \"$BASE/viewer/job/<job>/\"",
                 "200, the viewer's HTML page", also=("GET /viewer/job/{id}",)),
        endpoint("GET", "/api/jobs/{id}/display-cloud", "Read-only.",
                 "For the browser's 3D scene viewer, not for saving: a PLY file of the job as the "
                 "viewer draws it, the viewer's binary cloud document within its display budget. "
                 "Download the PLY itself from `result` or `files`.",
                 "curl -sS -o cloud.bin -w '%{http_code}\\n' "
                 '"$BASE/api/jobs/<job>/display-cloud?file=<path>"',
                 "200, the binary cloud document (application/octet-stream); 400 `usage` for a "
                 "file that is not a PLY the viewer can draw"),
        endpoint("GET", "/api/display-transform", "Read-only.",
                 "The viewer's display transform of a scene: identity in map coordinates, the "
                 "upright transform for a single image's camera frame. 400 `usage` for a bad "
                 "`up`.",
                 'curl -sS "$BASE/api/display-transform?camera=false"',
                 compact({"camera_frame": False, "display_transform": display_transform(False)})),
    ]


GROUPS: tuple[tuple[str, Callable[[str], bool]], ...] = (  # the rest: "Service"
    ("Uploads", lambda p: p.startswith("/api/uploads")),
    ("Maps (read-only)", lambda p: p.startswith("/api/maps") and "/viewer" not in p),
    ("Jobs", lambda p: p.startswith("/api/jobs") and "/viewer" not in p and "display" not in p),
    ("Viewer", lambda p: "/viewer" in p or "display" in p),
)


def endpoints_section(doc: Json, ex: Operation, d: Json) -> list[str]:
    """Every endpoint of the OpenAPI document besides the operations (and the document itself),
    from its template entry, or a generic one built from the document."""
    template = {f"{e['method']} {e['path']}": e for e in fixed_endpoints(ex, d)}
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
             "The service's own endpoints, besides the operations. In paths, `{id}` is a job's or "
             "an upload's id, `{name}` a map's name and `{path}` a file's path. Any of them "
             "answers an error of Errors when it fails."]
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
    """The workflow's example: an operation on one image with a result and a folder of files
    (else the one closest to that)."""
    def score(op: Operation) -> int:
        kinds = {o.kind for o in op.options}
        return 4 * (Kind.IMAGE in kinds) + 2 * (Kind.FILE_OUT in kinds) + (Kind.FOLDER_OUT in kinds)
    return max(ops.values(), key=score)


def render(ops: dict[str, Operation] | None = None) -> str:
    """The whole ``SKILL.md``."""
    ops = operations() if ops is None else ops
    doc = openapi.document(ops)
    entries = {op.id: entry(doc, op) for op in ops.values()}

    def ids(pick: Callable[[Operation], bool]) -> str:
        return ", ".join(code(op.id) for op in ops.values() if pick(op))

    def of(p: spec.Program) -> Callable[[Operation], bool]:
        return lambda op: op.program is p

    programs = [f"| `{p.prog}` | {ids(of(p))} | {p.description} |"
                for p in spec.PROGRAMS if any(op.program is p for op in ops.values())]
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
    params = sample_params(ex, d, folder=True)
    browser = ids(lambda op: op.browser)
    single = ids(lambda op: op.viewer_input is not None and not op.browser)
    viewers = (f"A viewer operation ({browser}) has no result file: its finished job's `viewer` is "
               "the page (`$BASE/viewer/job/<job>/`). " if browser else "") + (
        f"{single} also save the viewer of their image with `?viewer=true`: the job's `viewer`, "
        "or, when only that step failed (the result stands), `viewer_error` with its `code`, "
        "`exit_code`, `http_status` and `message` (code `cancelled` when the job was cancelled "
        "during that step). " if single else "")
    upload = SAMPLE_UPLOAD.describe(Path("/<data>"))
    parts = [
        front_matter(ops),
        fill(INTRO, programs="\n".join(programs)),
        fill(FIND, cache=CACHE, snippet=ADDRESS_SNIPPET),
        fill(SAFETY, inference_ops=ids(lambda op: entries[op.id]["inference"] == "required"),
             start=START_COMMAND, no_inference=", ".join(filter(None, no_inference)) or "nothing",
             writers=ids(lambda op: op in writers) or "the mapping operation",
             writer_stages=writer_stages),
        fill(ERRORS, op=ex.id, refusal=refusal(ex) or "the command's refusal", codes="\n".join(
            f"| `{c['code']}` | {c['exit_code']} | {c['http_status']} | `{c['job_state']}` |"
            for c in doc["x-oms"]["exit_codes"]),
             service="\n".join(f"| `{k}` | {status} | {cell(when)} |"
                               for k, (status, when) in SERVICE_ERRORS.items())),
        fill(WORKFLOW, op=ex.id, label=ex.label, statuses=statuses(), json=JSON_POST,
             upload_curl=f"curl -sS -X POST -T {upload['name']} -H 'Content-Type: "
                         f"application/octet-stream' \"$BASE/api/uploads?name={upload['name']}\"",
             upload=compact(upload), body=compact(params), validated=validated(ex, d, params),
             queued=compact(sample_job(ex, d, params)), stages=chain(d["stages"]),
             result=shown(ex, params)[1] or "result", files=compact(files_of(ex, d)),
             succeeded=brief(sample_job(ex, d, params, **succeeded(d)), "id", "state", "stage",
                             "stages", "exit_code", "error", "result"),
             viewers=viewers),
        OPERATIONS,
        *("\n".join(operation_section(doc, op)) for op in ops.values()),
        "\n".join(endpoints_section(doc, ex, d)),
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
