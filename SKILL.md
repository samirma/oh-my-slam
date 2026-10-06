---
name: oh-my-slam-api
description: "Use the oh-my-slam web service (server.sh) with sh and curl, from its Mac or any machine on the LAN. Jobs (JSON = OpenLABEL scene): reconstruct.sh Single-image reconstruction (stdout or -o file): reconstruct → JSON/PLY; mapper.sh Multi-frame mapping (persistent map): mapper-update (create or extend a map), mapper-locate (camera pose of images in an existing map (read-only)) → JSON/PLY/map; segment.sh Instance segmentation → JSON + OBBs, artefacts: segment-image, segment-map → JSON/PLY/PNG/CSV/Markdown; view.sh Browser visualisation of an image or a map: view-image, view-map → viewer page. Also: upload inputs; list maps, download map files; follow, cancel and resubmit jobs, get their log, stage timings, results and files; open 3D viewers of maps and jobs; service and inference-server health; the OpenAPI document. Use when the user asks for any of these and a server.sh is running (its URL, else ask). Inference server needed except for segment-map, view-map (mapper-locate: at times)."
---

# oh-my-slam API

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
| `reconstruct.sh` | `reconstruct` | Single-image reconstruction (stdout or -o file). |
| `mapper.sh` | `mapper-update`, `mapper-locate` | Multi-frame mapping (persistent map). |
| `segment.sh` | `segment-image`, `segment-map` | Instance segmentation → JSON + OBBs, artefacts. |
| `view.sh` | `view-image`, `view-map` | Browser visualisation of an image or a map. |

## Find the service

`server.sh` binds a free port unless it was started with `--port <n>`, so its URL changes from run
to run. Run this snippet once: it prints the base URL on stdout and caches it in
`${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam-api/server_url`.

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
cache="${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam-api/server_url"
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
EOF
```

Your shell keeps no variables between commands, so begin **every** command with
`BASE=<that URL>;` (a `;`, not a `BASE=... curl` prefix), e.g.
`BASE=http://127.0.0.1:52026; curl -sS "$BASE/api/health"`. All examples below use
`"$BASE/..."`. Use the URL as found: the service answers 403 `forbidden` to a `Host` that does
not name its machine. `GET /api/health` names the service's workspace (`service.data`): check it
is the one the user means. If a request later fails to connect (curl exit 7 or 28), run the
snippet again: it searches again only when the cached URL stops answering.

## Safety

* **Inference server down.** These need the inference server, a separate process on the Mac:
  `reconstruct`, `mapper-update`, `segment-image`, `view-image`. When it is down they are refused with 503 `server_unavailable`, and
  `GET /api/health` says so in `inference` (`status`, `message`, `start_command`). Report the
  message and the start command it gives (`./start_inference_server.sh`) to the user instead of retrying, and offer
  what still works: `segment-map`, `view-map`, `mapper-locate` (it needs the server only for retrieval in maps of more keyframes than are matched exhaustively: map keyframes greater than 150), and every read-only endpoint (maps, jobs, results, viewers).
* **Never start or stop `server.sh` or the inference server**, and never kill their processes:
  the user runs them.
* **Never write into a map's folder** (`<data>/maps/<name>/`, not even through the shell on the
  Mac): maps change only through `mapper-update`. Map files are download-only.
* **Ask the user first** before updating an existing map (`mapper-update` with the name of a map that
  `GET /api/maps/<name>` finds), before starting a long mapping job (`mapper-update` on new inputs
  runs 15 stages, from `setup` to `commit`, and can take many minutes), and before cancelling a job you did not
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

## Errors

Every error is

```json
{"error": {"code": "<code>", "message": "<the command's own message>", "http_status": <status>}}
```

A refused operation (and validate's `problems`) adds `rule`, `parameters` (those it concerns),
`exit_code`, `problems` (every problem) and `by_parameter` (the messages per parameter). For
example, `POST /api/ops/segment-image` with the body `{}` answers 400 `{"error":{"rule":"arguments","parameters":["image","map"],"message":"one of the arguments -i -m is required","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["one of the arguments -i -m is required"]}}}`. A job that failed or was
cancelled carries `code`, `exit_code`, `http_status` and `message` in its `error`. The code is the
name of the command's exit status, mapped to an HTTP status by one rule (input errors 4xx,
inference server unavailable 503, internal 500):

| Code | Exit status | HTTP | Job state |
|---|---|---|---|
| `ok` | 0 | 200 | `succeeded` |
| `internal` | 1 | 500 | `failed` |
| `usage` | 2 | 400 | `failed` |
| `server_unavailable` | 3 | 503 | `failed` |
| `not_a_map` | 4 | 422 | `failed` |
| `not_registered` | 5 | 422 | `failed` |
| `map_locked` | 6 | 409 | `failed` |
| `interrupted` | 130 | 499 | `cancelled` |

The service's own refusals have the same shape:

| Code | HTTP | When |
|---|---|---|
| `not_found` | 404 | no such map, job, upload, file or operation, or a job without that result |
| `forbidden` | 403 | a `Host` that does not name the service's machine, or a foreign `Origin` |
| `unsupported_media_type` | 415 | a POST whose body is not `application/json`, or an upload sent as a form (`curl -F`) |
| `too_large` | 413 | an upload over 8 GiB |
| `insufficient_storage` | 413 | an upload that would leave less than 1 GiB free on the workspace's disk |
| `upload_in_use` | 409 | an upload that is already the input of a queued or running job |
| `not_cancellable` | 409 | cancelling a job that has ended |
| `gone` | 410 | re-submitting a job whose operation the commands no longer offer |
| `stopping` | 503 | a submission while the service shuts down |

## Job workflow

Every operation runs as a job. The example is `segment-image` (`segment.sh -i`); every operation in
Operations works the same way.

1. **Check the service** and the inference server.
   ```sh
   curl -sS "$BASE/api/health"
   ```
   `inference.status` is one of `down`, `loading`, `ready`, `error`, `stopping`. An operation that needs inference is accepted
   while it is `ready` or `loading`; otherwise see Safety.

2. **Upload each input file**, or skip this and name a path inside the workspace. The request
   body is the raw file, streamed with `-T` (not a form: `curl -F` is refused with 415). `name` is
   the file name the command sees: keep its suffix (it tells the type) and use only letters,
   digits, `.`, `_` and `-`.
   ```sh
   curl -sS -X POST -T photo.jpg -H 'Content-Type: application/octet-stream' "$BASE/api/uploads?name=photo.jpg"
   ```
   → 201 `{"id":"<upload>","name":"photo.jpg","size":2481152,"path":"uploads/<upload>/photo.jpg"}`

   The `path` is the parameter's value. A parameter that takes several files (a JSON list) takes
   one upload per file, listed in the order the command should read them.

3. **Validate** the request: the command's own checks; nothing is queued.
   ```sh
   curl -sS -X POST "$BASE/api/ops/segment-image/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg","artifacts":"art"}'
   ```
   → `{"valid":true,"command":["segment.sh","-i=uploads/<upload>/photo.jpg","-o=result.json","-d=art"],"inference":true,"problems":[],"by_parameter":{}}`

   With `"valid":false`, `problems` and `by_parameter` say what to change, in the command's
   words. `command` is the command line the job will run.

4. **Submit** the same body.
   ```sh
   curl -sS -X POST "$BASE/api/ops/segment-image" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg","artifacts":"art"}'
   ```
   → 202, `Location: /api/jobs/<job>`, the job:
   `{"id":"<job>","operation":"segment-image","label":"segment.sh -i","params":{"image":"uploads/<upload>/photo.jpg","artifacts":"art"},"command":["segment.sh","-i=uploads/<upload>/photo.jpg","-o=result.json","-d=art"],"inference":true,"conditional":false,"uploads":["<upload>"],"writes":null,"result_name":"result.json","result_format":"json","saves_viewer":false,"state":"queued","created_at":1791281700.0,"started_at":null,"ended_at":null,"stage":null,"progress":null,"stages":[],"counts":{},"exit_code":null,"error":null,"viewer":null,"viewer_error":null,"viewer_progress":null,"log_tail":[],"resubmitted_from":null,"cancel_requested":false,"result":null}`

   Keep its `id`. A refused submission answers the error of Errors and queues nothing.

5. **Follow the job** until its `state` is `succeeded`, `failed` or `cancelled`, by its event
   stream, which ends with the job (each `event: job` carries the whole job on its `data:` line,
   written with spaces: `"state": "running"`):
   ```sh
   curl -sS -N "$BASE/api/jobs/<job>/events"
   ```
   or by polling it every few seconds:
   ```sh
   BASE=<url>; JOB=<job>; while :; do s=$(curl -sS "$BASE/api/jobs/$JOB" | sed -n 's/.*"state":"\([a-z]*\)".*/\1/p'); echo "$s"; case $s in queued|running) sleep 5 ;; *) break ;; esac; done
   ```
   While it runs, `stage` is the command's own timing stage (here `connect` → `inference` → `segment` → `export` → `artifacts` → `write`) and `progress` is
   `{"stage","done","total"}` where the command knows its size. Jobs that use the inference
   server run one at a time in submission order, so a job may stay `queued` for a while.

6. **Download the result and each file** with `-o`. They are byte-identical to what the command
   writes, so never rewrite them. `-w` prints the HTTP status: anything but 200 means the file
   holds the error instead.
   ```sh
   curl -sS -o result.json -w '%{http_code}\n' "$BASE/api/jobs/<job>/result"
   curl -sS "$BASE/api/jobs/<job>/files"
   ```
   → `[{"path":"art/catalog.csv","size":1024,"media_type":"text/csv","url":"/api/jobs/<job>/files/art/catalog.csv"},{"path":"art/catalog.md","size":1024,"media_type":"text/markdown","url":"/api/jobs/<job>/files/art/catalog.md"},{"path":"art/segmentation.json","size":1024,"media_type":"application/json","url":"/api/jobs/<job>/files/art/segmentation.json"},{"path":"art/segmented.png","size":1024,"media_type":"image/png","url":"/api/jobs/<job>/files/art/segmented.png"},{"path":"art/segments.ply","size":1024,"media_type":"application/octet-stream","url":"/api/jobs/<job>/files/art/segments.ply"},{"path":"result.json","size":1024,"media_type":"application/json","url":"/api/jobs/<job>/files/result.json"}]`

   Every file, under a folder named after the job:
   ```sh
   BASE=<url>; JOB=<job>; curl -sS "$BASE/api/jobs/$JOB/files" | tr '{}' '\n\n' | sed -n 's/.*"path":"\([^"]*\)".*/\1/p' | while read -r p; do curl -sS --create-dirs -o "$JOB/$p" "$BASE/api/jobs/$JOB/files/$p"; done
   ```

7. **Report.** The finished job: `{"id":"<job>","state":"succeeded","stage":"write","stages":[{"stage":"connect","seconds":0.5},{"stage":"inference","seconds":0.5},{"stage":"segment","seconds":0.5},{"stage":"export","seconds":0.5},{"stage":"artifacts","seconds":0.5},{"stage":"write","seconds":0.5}],"exit_code":0,"error":null,"result":{"name":"result.json","format":"json","url":"/api/jobs/<job>/result"},…}`. Its `stage` names and `stages` seconds are the
   command's own; `GET /api/jobs/<job>/timings` has the command's whole timing record and
   `GET /api/jobs/<job>/log` everything it printed. A `failed` job's `error` holds the command's
   message and code; a `cancelled` one has `error.code` `interrupted`.

A viewer operation (`view-image`, `view-map`) has no result file: its finished job's `viewer` is the page (`$BASE/viewer/job/<job>/`). `reconstruct`, `segment-image` also save the viewer of their image with `?viewer=true`: the job's `viewer`, or, when only that step failed (the result stands), `viewer_error` with its `code`, `exit_code`, `http_status` and `message` (code `cancelled` when the job was cancelled during that step). A map's viewer is always at `$BASE/viewer/map/<name>/`, and the browser application at
`$BASE/`: give the user those URLs to look at results.

## Operations

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

### `reconstruct` — `reconstruct.sh`

`POST /api/ops/reconstruct` · `POST /api/ops/reconstruct/validate`. Single-image reconstruction (stdout or -o file). **Needs the inference server** (reconstructs the image with the inference server). Read-only: writes only the job's own files.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `image` (required) | `-i` | workspace path: .bmp .jpeg .jpg .png .tif .tiff .webp |  | input RGB image |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
| `output` | `-o` | plain file name in the job's `out/` (default `result.json` or `result.ply`, by the result's format) |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `attrs` | `-p` | `key=value,…` (keys below), or a list of them | `color=rgb,stride=1,min-depth=0,max-depth=inf,edge=0.04,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=rgb\|segment\|height\|none [rgb], stride=N [1], min-depth=METRES [0], max-depth=METRES\|inf [inf], edge=JUMP [0.04], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; requires -f ply (only with -f ply) |

Keys of `attrs` (keys not given keep their default):

| Key | Values | Default | Effect |
|---|---|---|---|
| `color` | `rgb` \| `segment` \| `height` \| `none` | `rgb` | per-point colour: image colour, object colour, height ramp, or no colour |
| `stride` | integer ≥ 1 | `1` | keep every n-th pixel along each image axis |
| `min-depth` | number ≥ 0 < `max-depth` | `0` | drop pixels closer than this depth |
| `max-depth` | number > 0 or `inf` | `inf` | drop pixels farther than this depth |
| `edge` | number ≥ 0 | `0.04` | drop flying pixels on depth discontinuities (0 disables) |
| `voxel` | number ≥ 0 | `0` | keep one point per voxel of this size (0 = off; colours are not averaged) |
| `normals` | `on` \| `off` | `off` | add nx ny nz float properties |
| `label` | `on` \| `off` | `off` | add an int label property (object id, 0 = unsegmented) |
| `encoding` | `binary` \| `ascii` | `binary` | binary_little_endian 1.0 or ASCII PLY |

Produces:

* The result when `format` is `json` (`GET /api/jobs/<job>/result`, application/json): the OpenLABEL 1.0.0 scene description (spec §3): objects, labels, scores, colours and OBBs in the camera frame.
* The result when `format` is `ply` (`GET /api/jobs/<job>/result`, application/octet-stream): the point cloud (camera frame, metres) shaped by -p.

Stages: `connect` → `inference` → `segment` → `export` → `write`. Checks: -p sets point-cloud attributes, which only the PLY output has: use -f ply; every key and value is valid for the command (spec §2.2); the -o file can be written (it is not a folder), checked before any work; the -i image exists. Refused with: `usage` (400), `server_unavailable` (503); a job can end with any code of the table in Errors.

`POST /api/ops/reconstruct?viewer=true`: also save the viewer of the request's image: one more step of the same job, which replays the command's recorded inference (no second pass; only what the command did not ask, e.g. segmentation for reconstruct -f ply, goes to the server).

```sh
curl -sS -X POST "$BASE/api/ops/reconstruct/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
curl -sS -X POST "$BASE/api/ops/reconstruct" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
```

→ validate: `{"valid":true,"command":["reconstruct.sh","-i=uploads/<upload>/photo.jpg","-o=result.json"],"inference":true,"problems":[],"by_parameter":{}}`

→ submit: 202, `Location: /api/jobs/<job>`, the job `{"id":"<job>","operation":"reconstruct","label":"reconstruct.sh",…}` (see Job workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["image"],"message":"the following arguments are required: -i","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["the following arguments are required: -i"]}}}`

### `mapper-update` — `mapper.sh update`

`POST /api/ops/mapper-update` · `POST /api/ops/mapper-update/validate`. Create or extend a map. **Needs the inference server** (infers depth and objects of every new keyframe). **Writes the map** named by `map` (creates or extends it): ask the user first (see Safety).

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `inputs` (required) | `-i` | list of workspace paths, in order (the order matters): .avi .bmp .jpeg .jpg .m4v .mkv .mov .mp4 .png .tif .tiff .webm .webp |  | image files, or exactly one video |
| `map` (required) | `-m` | map name, `<name>` or `maps/<name>`: an existing map, or a new name to create |  | map folder |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
| `output` | `-o` | plain file name in the job's `out/` (default `result.json` or `result.ply`, by the result's format) |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `attrs` | `-p` | `key=value,…` (keys below), or a list of them | `color=rgb,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=rgb\|segment\|height\|none [rgb], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; requires -f ply (only with -f ply) |
| `mode` | `-t` | one of `full`, `single` | `full` | full = whole map with all keyframe poses; single = new input only (default: full) |
| `fps` | `-fps` | number > 0 | `2.0` | video frames per second to sample (default: 2; ignored for images) (video input only; ignored for images) |

Keys of `attrs` (keys not given keep their default):

| Key | Values | Default | Effect |
|---|---|---|---|
| `color` | `rgb` \| `segment` \| `height` \| `none` | `rgb` | per-point colour: image colour, object colour, height ramp, or no colour |
| `voxel` | number ≥ 0 | `0` | keep one point per voxel of this size (0 = off; colours are not averaged) |
| `normals` | `on` \| `off` | `off` | add nx ny nz float properties |
| `label` | `on` \| `off` | `off` | add an int label property (object id, 0 = unsegmented) |
| `encoding` | `binary` \| `ascii` | `binary` | binary_little_endian 1.0 or ASCII PLY |

Produces:

* The result when `format` is `json` (`GET /api/jobs/<job>/result`, application/json): the OpenLABEL 1.0.0 scene description (spec §3) of the whole map (-t full) or of the new input (-t single), map coordinates.
* The result when `format` is `ply` (`GET /api/jobs/<job>/result`, application/octet-stream): the map cloud (-t full) or the new frames' points (-t single).
* The map `map` of the workspace: the map folder, created or extended (`GET /api/maps/<map>`).

Stages: `setup` → `ingest` → `inference` → `sfm` → `features_matching` → `pose_refinement` → `focal_rerun` → `map_frame` → `depth_alignment` → `persist_frames` → `validity` → `objects` → `cloud` → `export` → `commit`. Checks: -p sets point-cloud attributes, which only the PLY output has: use -f ply; every key and value is valid for the command (spec §2.2); the -o file can be written (it is not a folder), checked before any work; -fps must be positive for a video; for images it is ignored with a warning; -i names existing image files, in order, or exactly one video; -m is a map, an empty folder or a new one; any other folder is refused and left untouched. Refused with: `usage` (400), `server_unavailable` (503), `not_a_map` (422); a job can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/mapper-update/validate" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/video.mp4"],"map":"<map>"}'
curl -sS -X POST "$BASE/api/ops/mapper-update" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/video.mp4"],"map":"<map>"}'
```

→ validate: `{"valid":true,"command":["mapper.sh","update","-i","uploads/<upload>/video.mp4","-m=maps/<map>","-o=result.json"],"inference":true,"problems":[],"by_parameter":{}}`

→ submit: 202, `Location: /api/jobs/<job>`, the job `{"id":"<job>","operation":"mapper-update","label":"mapper.sh update",…}` (see Job workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["inputs","map"],"message":"the following arguments are required: -i, -m","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"inputs":["the following arguments are required: -i, -m"]}}}`

### `mapper-locate` — `mapper.sh locate`

`POST /api/ops/mapper-locate` · `POST /api/ops/mapper-locate/validate`. Camera pose of images in an existing map (read-only). **Needs the inference server** only for retrieval in maps of more keyframes than are matched exhaustively (map keyframes greater than 150). Read-only: writes only the job's own files.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `inputs` (required) | `-i` | list of workspace paths: .bmp .jpeg .jpg .png .tif .tiff .webp |  | one or more image files (a video is refused) |
| `map` (required) | `-m` | map name, `<name>` or `maps/<name>`: an existing map |  | existing map folder |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
| `output` | `-o` | plain file name in the job's `out/` (default `result.json` or `result.ply`, by the result's format) |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `attrs` | `-p` | `key=value,…` (keys below), or a list of them | `color=rgb,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=rgb\|segment\|height\|none [rgb], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; requires -f ply (only with -f ply) |
| `mode` | `-t` | one of `full`, `single` | `single` | single = the located camera poses only; full = the whole map plus the located poses (default: single) |

Keys of `attrs` (keys not given keep their default):

| Key | Values | Default | Effect |
|---|---|---|---|
| `color` | `rgb` \| `segment` \| `height` \| `none` | `rgb` | per-point colour: image colour, object colour, height ramp, or no colour |
| `voxel` | number ≥ 0 | `0` | keep one point per voxel of this size (0 = off; colours are not averaged) |
| `normals` | `on` \| `off` | `off` | add nx ny nz float properties |
| `label` | `on` \| `off` | `off` | add an int label property (object id, 0 = unsegmented) |
| `encoding` | `binary` \| `ascii` | `binary` | binary_little_endian 1.0 or ASCII PLY |

Produces:

* The result when `format` is `json` (`GET /api/jobs/<job>/result`, application/json): the OpenLABEL 1.0.0 scene description (spec §3): the located camera poses (-t single), or the whole map plus them (-t full).
* The result when `format` is `ply` (`GET /api/jobs/<job>/result`, application/octet-stream): the map points visible from the located cameras (-t single) or the whole map cloud (-t full); the located poses in the header.

Stages: `setup` → `features_matching` → `pose` → `export`. Checks: -p sets point-cloud attributes, which only the PLY output has: use -f ply; every key and value is valid for the command (spec §2.2); -i names image files; a video is refused; -m is an existing map; a missing or empty folder is not created, a non-empty folder that is not a map is refused; -o is not inside the map folder, which locate never writes; the -o file can be written (it is not a folder), checked before any work. Refused with: `usage` (400), `server_unavailable` (503), `not_a_map` (422); a job can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/mapper-locate/validate" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/query.jpg"],"map":"<map>"}'
curl -sS -X POST "$BASE/api/ops/mapper-locate" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/query.jpg"],"map":"<map>"}'
```

→ validate: `{"valid":true,"command":["mapper.sh","locate","-i","uploads/<upload>/query.jpg","-m=maps/<map>","-o=result.json"],"inference":false,"problems":[],"by_parameter":{}}`

→ submit: 202, `Location: /api/jobs/<job>`, the job `{"id":"<job>","operation":"mapper-locate","label":"mapper.sh locate",…}` (see Job workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["inputs","map"],"message":"the following arguments are required: -i, -m","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"inputs":["the following arguments are required: -i, -m"]}}}`

### `segment-image` — `segment.sh -i`

`POST /api/ops/segment-image` · `POST /api/ops/segment-image/validate`. Instance segmentation → JSON + OBBs, artefacts. **Needs the inference server** (segments the image with the inference server). Read-only: writes only the job's own files.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `image` (required) | `-i` | workspace path: .bmp .jpeg .jpg .png .tif .tiff .webp |  | input RGB image |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
| `output` | `-o` | plain file name in the job's `out/` (default `result.json` or `result.ply`, by the result's format) |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `attrs` | `-p` | `key=value,…` (keys below), or a list of them | `color=segment,stride=1,min-depth=0,max-depth=inf,edge=0.04,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=segment (fixed) [segment], stride=N [1], min-depth=METRES [0], max-depth=METRES\|inf [inf], edge=JUMP [0.04], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; shapes the -f ply output and segments.ply, so it needs -f ply or -d; with -m the pixel-level keys (stride, min-depth, max-depth, edge) are refused (only with -f ply or -d) |
| `artifacts` | `-d` | plain folder name in the job's `out/` (none: no files) |  | also write segmentation.json, segmented.png, catalog.csv, catalog.md and segments.ply into FOLDER |
| `min_score` | `--min-score` | finite number | `0.5` | drop detections below this score (default 0.5; -i only) (-i only) |

Keys of `attrs` (keys not given keep their default):

| Key | Values | Default | Effect |
|---|---|---|---|
| `color` | `segment` | `segment` | per-point colour: image colour, object colour, height ramp, or no colour |
| `stride` | integer ≥ 1 | `1` | keep every n-th pixel along each image axis |
| `min-depth` | number ≥ 0 < `max-depth` | `0` | drop pixels closer than this depth |
| `max-depth` | number > 0 or `inf` | `inf` | drop pixels farther than this depth |
| `edge` | number ≥ 0 | `0.04` | drop flying pixels on depth discontinuities (0 disables) |
| `voxel` | number ≥ 0 | `0` | keep one point per voxel of this size (0 = off; colours are not averaged) |
| `normals` | `on` \| `off` | `off` | add nx ny nz float properties |
| `label` | `on` \| `off` | `off` | add an int label property (object id, 0 = unsegmented) |
| `encoding` | `binary` \| `ascii` | `binary` | binary_little_endian 1.0 or ASCII PLY |

Produces:

* The result when `format` is `json` (`GET /api/jobs/<job>/result`, application/json): the OpenLABEL 1.0.0 scene description (spec §3) (camera frame).
* The result when `format` is `ply` (`GET /api/jobs/<job>/result`, application/octet-stream): the object-coloured point cloud.
* `<artifacts>/segmentation.json` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/segmentation.json`, application/json): the OpenLABEL 1.0.0 scene description (spec §3), identical to -f json.
* `<artifacts>/segmented.png` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/segmented.png`, image/png): the image (for a map, keyframes) with each instance mask painted in its object's colour. Each pixel is painted in its object's colour, so the object under a pixel is the one of that colour.
* `<artifacts>/catalog.csv` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/catalog.csv`, text/csv): one row per object.
* `<artifacts>/catalog.md` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/catalog.md`, text/markdown): the catalogue as a table by descending volume.
* `<artifacts>/segments.ply` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/segments.ply`, application/octet-stream): the object-coloured cloud, identical to -f ply.

Stages: `connect` → `inference` → `segment` → `export` → `artifacts` → `write`. Checks: --min-score is a finite number; -p sets point-cloud attributes, which shape the PLY output: use -f ply or -d <folder>; every key and value is valid for the command (spec §2.2); the -o file can be written (it is not a folder), checked before any work; the -d folder can be created and written, checked before any work; the -i image exists. Refused with: `usage` (400), `server_unavailable` (503); a job can end with any code of the table in Errors.

`POST /api/ops/segment-image?viewer=true`: also save the viewer of the request's image: one more step of the same job, which replays the command's recorded inference (no second pass; only what the command did not ask, e.g. segmentation for reconstruct -f ply, goes to the server).

```sh
curl -sS -X POST "$BASE/api/ops/segment-image/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
curl -sS -X POST "$BASE/api/ops/segment-image" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
```

→ validate: `{"valid":true,"command":["segment.sh","-i=uploads/<upload>/photo.jpg","-o=result.json"],"inference":true,"problems":[],"by_parameter":{}}`

→ submit: 202, `Location: /api/jobs/<job>`, the job `{"id":"<job>","operation":"segment-image","label":"segment.sh -i",…}` (see Job workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["image","map"],"message":"one of the arguments -i -m is required","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["one of the arguments -i -m is required"]}}}`

### `segment-map` — `segment.sh -m`

`POST /api/ops/segment-map` · `POST /api/ops/segment-map/validate`. Instance segmentation → JSON + OBBs, artefacts. **Works without the inference server** (exports the map's persistent objects without inference). Read-only: writes only the job's own files.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `map` (required) | `-m` | map name, `<name>` or `maps/<name>`: an existing map |  | existing map folder (read-only) |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
| `output` | `-o` | plain file name in the job's `out/` (default `result.json` or `result.ply`, by the result's format) |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `attrs` | `-p` | `key=value,…` (keys below), or a list of them | `color=segment,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=segment (fixed) [segment], stride=N [1], min-depth=METRES [0], max-depth=METRES\|inf [inf], edge=JUMP [0.04], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; shapes the -f ply output and segments.ply, so it needs -f ply or -d; with -m the pixel-level keys (stride, min-depth, max-depth, edge) are refused (only with -f ply or -d) |
| `artifacts` | `-d` | plain folder name in the job's `out/` (none: no files) |  | also write segmentation.json, segmented.png, catalog.csv, catalog.md and segments.ply into FOLDER |

Keys of `attrs` (keys not given keep their default):

| Key | Values | Default | Effect |
|---|---|---|---|
| `color` | `segment` | `segment` | per-point colour: image colour, object colour, height ramp, or no colour |
| `voxel` | number ≥ 0 | `0` | keep one point per voxel of this size (0 = off; colours are not averaged) |
| `normals` | `on` \| `off` | `off` | add nx ny nz float properties |
| `label` | `on` \| `off` | `off` | add an int label property (object id, 0 = unsegmented) |
| `encoding` | `binary` \| `ascii` | `binary` | binary_little_endian 1.0 or ASCII PLY |

Produces:

* The result when `format` is `json` (`GET /api/jobs/<job>/result`, application/json): the OpenLABEL 1.0.0 scene description (spec §3) of the map's objects (map coordinates).
* The result when `format` is `ply` (`GET /api/jobs/<job>/result`, application/octet-stream): the object-coloured map cloud.
* `<artifacts>/segmentation.json` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/segmentation.json`, application/json): the OpenLABEL 1.0.0 scene description (spec §3), identical to -f json.
* `<artifacts>/segmented.png` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/segmented.png`, image/png): the image (for a map, keyframes) with each instance mask painted in its object's colour. Each pixel is painted in its object's colour, so the object under a pixel is the one of that colour.
* `<artifacts>/catalog.csv` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/catalog.csv`, text/csv): one row per object.
* `<artifacts>/catalog.md` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/catalog.md`, text/markdown): the catalogue as a table by descending volume.
* `<artifacts>/segments.ply` when `artifacts` is given (`GET /api/jobs/<job>/files/<artifacts>/segments.ply`, application/octet-stream): the object-coloured cloud, identical to -f ply.

Stages: `export` → `artifacts` → `write`. Checks: --min-score applies to -i only; -p sets point-cloud attributes, which shape the PLY output: use -f ply or -d <folder>; every key and value is valid for the command (spec §2.2); the -o file can be written (it is not a folder), checked before any work; the -d folder can be created and written, checked before any work; -m is a map (it is opened read-only). Refused with: `usage` (400), `not_a_map` (422); a job can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/segment-map/validate" -H 'Content-Type: application/json' -d '{"map":"<map>"}'
curl -sS -X POST "$BASE/api/ops/segment-map" -H 'Content-Type: application/json' -d '{"map":"<map>"}'
```

→ validate: `{"valid":true,"command":["segment.sh","-m=maps/<map>","-o=result.json"],"inference":false,"problems":[],"by_parameter":{}}`

→ submit: 202, `Location: /api/jobs/<job>`, the job `{"id":"<job>","operation":"segment-map","label":"segment.sh -m",…}` (see Job workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["image","map"],"message":"one of the arguments -i -m is required","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["one of the arguments -i -m is required"]}}}`

### `view-image` — `view.sh -i`

`POST /api/ops/view-image` · `POST /api/ops/view-image/validate`. Browser visualisation of an image or a map. **Needs the inference server** (reconstructs and segments the image with the inference server). Read-only: saves a viewer in the job's own folder.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `image` (required) | `-i` | workspace path: .bmp .jpeg .jpg .png .tif .tiff .webp |  | RGB image to reconstruct and segment |

Produces:

* The viewer page: the finished job's `viewer` (`$BASE/viewer/job/<job>/`); give it to the user.

Stages: `connect` → `inference`. Checks: the -i image exists. Refused with: `usage` (400), `server_unavailable` (503); a job can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/view-image/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
curl -sS -X POST "$BASE/api/ops/view-image" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
```

→ validate: `{"valid":true,"command":["view.sh","-i=uploads/<upload>/photo.jpg"],"inference":true,"problems":[],"by_parameter":{}}`

→ submit: 202, `Location: /api/jobs/<job>`, the job `{"id":"<job>","operation":"view-image","label":"view.sh -i",…}` (see Job workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["image","map"],"message":"one of the arguments -i -m is required","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["one of the arguments -i -m is required"]}}}`

### `view-map` — `view.sh -m`

`POST /api/ops/view-map` · `POST /api/ops/view-map/validate`. Browser visualisation of an image or a map. **Works without the inference server** (opens the persisted map read-only). Read-only: saves a viewer in the job's own folder.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `map` (required) | `-m` | map name, `<name>` or `maps/<name>`: an existing map |  | map folder (opened read-only) |

Produces:

* The viewer page: the finished job's `viewer` (`$BASE/viewer/job/<job>/`); give it to the user.

Stages: none. Checks: -m is a map (it is opened read-only). Refused with: `usage` (400), `not_a_map` (422); a job can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/view-map/validate" -H 'Content-Type: application/json' -d '{"map":"<map>"}'
curl -sS -X POST "$BASE/api/ops/view-map" -H 'Content-Type: application/json' -d '{"map":"<map>"}'
```

→ validate: `{"valid":true,"command":["view.sh","-m=maps/<map>"],"inference":false,"problems":[],"by_parameter":{}}`

→ submit: 202, `Location: /api/jobs/<job>`, the job `{"id":"<job>","operation":"view-map","label":"view.sh -m",…}` (see Job workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["image","map"],"message":"one of the arguments -i -m is required","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["one of the arguments -i -m is required"]}}}`

## Endpoints

The service's own endpoints, besides the operations. In paths, `{id}` is a job's or an upload's id, `{name}` a map's name and `{path}` a file's path. Any of them answers an error of Errors when it fails.

### Service

#### `GET /api/health`

Read-only. The service (version, URL, workspace, queued and running jobs) and the inference server: `inference.status` is one of `down`, `loading`, `ready`, `error`, `stopping`; `start_command` is the command that starts it when it is not `ready` (report it, never run it), and `health` its own health when it answers.

```sh
curl -sS "$BASE/api/health"
```

→ `{"status":"ok","service":{"version":"<version>","url":"http://0.0.0.0:52026/","pid":4321,"workspace":"oh-my-slam-data","data":"/Users/<user>/oh-my-slam-data","started_at":1791281700.0,"jobs":{"queued":0,"running":0}},"inference":{"status":"down","message":"<why>","start_command":"./start_inference_server.sh"}}`

#### `GET /api/openapi.json`

Read-only. The running service's OpenAPI 3.1 document: every operation with its parameters, and its whole definition under `x-oms` (parameters, outputs, rules, errors, stages, inference need). It wins where it disagrees with this file.

```sh
curl -sS -o openapi.json "$BASE/api/openapi.json"
```

→ `{"openapi":"3.1.0","info":{…},"paths":{"/api/ops/<op>":{"post":{…,"x-oms":{…}}},…},"components":{…},"x-oms":{"exit_codes":[…],"stages":[…]}}`

### Uploads

#### `POST /api/uploads`

Stores a file in the workspace until its job ends. The request body is the file itself (`-T`), with its own media type or `application/octet-stream`; a form (`curl -F`, multipart) is refused with 415, since a cross-site web page could send one. At most 8 GiB, and it must leave 1 GiB free (413). Answers 201; use its `path` as an input parameter. Query: `name`: the file name (its suffix tells its type).

```sh
curl -sS -X POST -T photo.jpg -H 'Content-Type: application/octet-stream' "$BASE/api/uploads?name=photo.jpg"
```

→ `{"id":"<upload>","name":"photo.jpg","size":2481152,"path":"uploads/<upload>/photo.jpg"}`

#### `DELETE /api/uploads/{id}`

Deletes an upload. Only one that no queued or running job uses (409 `upload_in_use` otherwise: cancel the job instead); 404 `not_found` for no such upload.

```sh
curl -sS -X DELETE -w '%{http_code}\n' "$BASE/api/uploads/<upload>"
```

→ 204, no body

### Maps (read-only)

#### `GET /api/maps`

Read-only. Every map of the workspace with summary figures from its own metadata: each scalar of its `map.json`, the size of each of its lists (`<key>_count`), `frames`, `objects`, `last_update` and `thumbnail`, a keyframe image of the map (a path for `GET /api/maps/{name}/files/{path}`).

```sh
curl -sS "$BASE/api/maps"
```

→ `[{"name":"<map>","path":"maps/<map>",…,"frames":302,"objects":143,"last_update":{"id":2,"at":1790931612.9,"kind":"video","frames_added":120,"total_s":512.4},"thumbnail":"<keyframe image>"},…]`

#### `GET /api/maps/{name}`

Read-only. One map's summary plus `meta`, its whole `map.json`, whose `updates[]` hold each update's record with its `timings`. 404 `not_found` when there is no such map.

```sh
curl -sS "$BASE/api/maps/<map>"
```

→ `{"name":"<map>","path":"maps/<map>",…,"meta":{…,"updates":[{…,"timings":{…}},…],…}}`

#### `GET /api/maps/{name}/files/{path}`

Read-only download. A file of the map, such as `map.json` or the summary's `thumbnail`; hidden entries are never served (404).

```sh
curl -sS -o map.json -w '%{http_code}\n' "$BASE/api/maps/<map>/files/map.json"
```

→ 200 and the file itself, saved as it is by `-o`; an error answers the JSON of Errors instead (`-w '%{http_code}'` shows which)

### Jobs

#### `GET /api/jobs`

Read-only. Every job, oldest first; the list survives a service restart.

```sh
curl -sS "$BASE/api/jobs"
```

→ `[{"id":"<job>","operation":"segment-image","label":"segment.sh -i","state":"succeeded",…},…]`

#### `GET /api/jobs/events`

Read-only. Server-sent events of every job's changes. It never ends by itself: bound it with `--max-time` (curl then exits 28).

```sh
curl -sS -N --max-time 60 "$BASE/api/jobs/events"
```

```text
retry: 1000

event: job
id: 3
data: {"id": "<job>", …, "state": "running", "stage": "connect", …}

…
```

#### `GET /api/jobs/{id}`

Read-only. One job; poll it to follow the job. 404 `not_found` for no such job.

```sh
curl -sS "$BASE/api/jobs/<job>"
```

→ `{"id":"<job>","operation":"segment-image","state":"succeeded","stage":"write","progress":null,"stages":[{"stage":"connect","seconds":0.5},{"stage":"inference","seconds":0.5},{"stage":"segment","seconds":0.5},{"stage":"export","seconds":0.5},{"stage":"artifacts","seconds":0.5},{"stage":"write","seconds":0.5}],"error":null,"result":{"name":"result.json","format":"json","url":"/api/jobs/<job>/result"},…}`

#### `GET /api/jobs/{id}/events`

Read-only. Server-sent events of one job; the stream ends when the job ends, its last event with the final state.

```sh
curl -sS -N "$BASE/api/jobs/<job>/events"
```

```text
retry: 1000

event: job
id: 3
data: {"id": "<job>", …, "state": "running", "stage": "connect", …}

…
```

#### `POST /api/jobs/{id}/cancel`

Stops a job: ask the user first unless you submitted it. The effect of Ctrl-C on the command: a queued job is dropped, a running one is interrupted (a cancelled map update leaves the map as it was), and there is no result. 409 `not_cancellable` once the job has ended.

```sh
curl -sS -X POST -H 'Content-Type: application/json' "$BASE/api/jobs/<job>/cancel"
```

→ `{"id":"<job>","state":"running","cancel_requested":true,…}`

#### `POST /api/jobs/{id}/resubmit`

Submits a new job (it needs the inference server when its operation does). The same operation and parameters again; the body (a JSON object, `{}` for none) replaces some. The old job's uploads were deleted when it ended: upload the files again and pass their new paths. Answers 202 and the new job, or the refusal of a submission; 410 `gone` when the operation no longer exists.

```sh
curl -sS -X POST "$BASE/api/jobs/<job>/resubmit" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
```

→ `{"id":"<job>","operation":"segment-image","state":"queued","resubmitted_from":"<old job>",…}`

#### `GET /api/jobs/{id}/result`

Read-only download. The result, byte for byte the command's stdout or `-o` file. 404 `not_found` until the job has succeeded, and for a viewer operation, which has none.

```sh
curl -sS -o result.json -w '%{http_code}\n' "$BASE/api/jobs/<job>/result"
```

→ 200 and the file itself, saved as it is by `-o`; an error answers the JSON of Errors instead (`-w '%{http_code}'` shows which)

#### `GET /api/jobs/{id}/files`

Read-only. Every file the job wrote in its `out/` folder.

```sh
curl -sS "$BASE/api/jobs/<job>/files"
```

→ `[{"path":"art/catalog.csv","size":1024,"media_type":"text/csv","url":"/api/jobs/<job>/files/art/catalog.csv"},{"path":"art/catalog.md","size":1024,"media_type":"text/markdown","url":"/api/jobs/<job>/files/art/catalog.md"},{"path":"art/segmentation.json","size":1024,"media_type":"application/json","url":"/api/jobs/<job>/files/art/segmentation.json"},{"path":"art/segmented.png","size":1024,"media_type":"image/png","url":"/api/jobs/<job>/files/art/segmented.png"},{"path":"art/segments.ply","size":1024,"media_type":"application/octet-stream","url":"/api/jobs/<job>/files/art/segments.ply"},{"path":"result.json","size":1024,"media_type":"application/json","url":"/api/jobs/<job>/files/result.json"}]`

#### `GET /api/jobs/{id}/files/{path}`

Read-only download. One file the job wrote, by its `path` in the list.

```sh
curl -sS --create-dirs -o "<job>/art/catalog.csv" -w '%{http_code}\n' "$BASE/api/jobs/<job>/files/art/catalog.csv"
```

→ 200 and the file itself, saved as it is by `-o`; an error answers the JSON of Errors instead (`-w '%{http_code}'` shows which)

#### `GET /api/jobs/{id}/log`

Read-only. Everything the command printed on stderr (text/plain), its `timings:` line included.

```sh
curl -sS "$BASE/api/jobs/<job>/log"
```

```text
[oh-my-slam] …
[oh-my-slam] timings: total 3.0 s | connect 0.50, inference 0.50, segment 0.50, export 0.50, artifacts 0.50, write 0.50 | peak RSS 1234 MB
```

#### `GET /api/jobs/{id}/timings`

Read-only. The command's timing record (per-stage seconds and memory, inference requests, counts); `null` until it is written.

```sh
curl -sS "$BASE/api/jobs/<job>/timings"
```

→ `{"command":"segment.sh -i",…,"total_s":3.0,"stages_s":{"connect":0.5,"inference":0.5,"segment":0.5,"export":0.5,"artifacts":0.5,"write":0.5},"parts":…,"server":…,"counts":…,"peak_rss_mb":…,"stages_peak_rss_mb":…,"t0_unix":…,"stage_windows":…}`

### Viewer

#### `GET /api/maps/{name}/viewer/{path}` · `GET /api/maps/{name}/viewer`

Read-only. The map's viewer: give the user `$BASE/viewer/map/<map>/`, the same viewer at its page URL (`/api/maps/{name}/viewer` redirects to `…/viewer/`). Its data is the viewer's own: `api/meta`, `api/scene` (the scene JSON), `api/catalog` and `api/cloud?<attributes>` (binary).

```sh
curl -sS "$BASE/api/maps/<map>/viewer/api/meta"
```

→ `{"mode": "map", "title": "<map>", …}`

#### `GET /api/jobs/{id}/viewer/{path}` · `GET /api/jobs/{id}/viewer`

Read-only. The viewer a job saved (its `viewer` field): give the user `$BASE/viewer/job/<job>/` (`/api/jobs/{id}/viewer` redirects to `…/viewer/`).

```sh
curl -sS "$BASE/api/jobs/<job>/viewer/api/meta"
```

→ `{"mode": "image", …}`

#### `GET /viewer/map/{name}/{path}` · `GET /viewer/map/{name}`

Read-only. The page URL of a map's viewer, for the user's browser.

```sh
curl -sS -o /dev/null -w '%{http_code}\n' "$BASE/viewer/map/<map>/"
```

→ 200, the viewer's HTML page

#### `GET /viewer/job/{id}/{path}` · `GET /viewer/job/{id}`

Read-only. The page URL of a job's saved viewer, for the user's browser.

```sh
curl -sS -o /dev/null -w '%{http_code}\n' "$BASE/viewer/job/<job>/"
```

→ 200, the viewer's HTML page

#### `GET /api/jobs/{id}/display-cloud`

Read-only. For the browser's 3D scene viewer, not for saving: a PLY file of the job as the viewer draws it, the viewer's binary cloud document within its display budget. Download the PLY itself from `result` or `files`. Query: `file` (optional).

```sh
curl -sS -o cloud.bin -w '%{http_code}\n' "$BASE/api/jobs/<job>/display-cloud?file=<path>"
```

→ 200, the binary cloud document (application/octet-stream); 400 `usage` for a file that is not a PLY the viewer can draw

#### `GET /api/display-transform`

Read-only. The viewer's display transform of a scene: identity in map coordinates, the upright transform for a single image's camera frame. 400 `usage` for a bad `up`. Query: `camera` (optional): the scene is in a camera frame; `cs_types` (optional): a scene JSON's coordinate-system types, comma-separated: a camera frame when none is a scene_cs; `comment` (optional): a PLY's header comments (they name its frame); `up` (optional): x,y,z: the estimated up direction in the camera frame.

```sh
curl -sS "$BASE/api/display-transform?camera=false"
```

→ `{"camera_frame":false,"display_transform":[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]}`
