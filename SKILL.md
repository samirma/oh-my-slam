---
name: oh-my-slam-api
description: "Use the oh-my-slam web service (server.sh) with sh and curl, from its Mac or the LAN. Operations, each answered when its command ends with the command's own result and stage timings (JSON = OpenLABEL scene): reconstruct.sh Single-image reconstruction (stdout or -o file): reconstruct → JSON/PLY; mapper.sh Multi-frame mapping (persistent map): mapper-update (create or extend a map) → JSON/PLY/map, mapper-locate (camera pose of images in an existing map (read-only)) → JSON/PLY; segment.sh Instance segmentation → JSON + OBBs, artefacts: segment-image, segment-map → JSON/PLY. Also: validate requests; upload and discard inputs; list maps with their summaries and update history; service and inference-server health; the OpenAPI document. Use when the user asks for any of these and a server.sh is running (its URL, else ask). Inference server needed except for segment-map (mapper-locate: at times)."
---

# oh-my-slam API

`server.sh` is the oh-my-slam web service: a long-lived HTTP service on a Mac that runs every
mode of `reconstruct.sh`, `mapper.sh` and `segment.sh`. There are no jobs: each operation runs within its own HTTP request, and the
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
| `reconstruct.sh` | `reconstruct` | Single-image reconstruction (stdout or -o file). |
| `mapper.sh` | `mapper-update`, `mapper-locate` | Multi-frame mapping (persistent map). |
| `segment.sh` | `segment-image`, `segment-map` | Instance segmentation → JSON + OBBs, artefacts. |

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
  `reconstruct`, `mapper-update`, `segment-image`. When it is down they are refused with 503 `server_unavailable`, and
  `GET /api/health` says so in `inference` (`status`, `message`, `start_command`). Report the
  message and the start command it gives (`./start_inference_server.sh`) to the user instead of retrying, and offer
  what still works: `segment-map`, `mapper-locate` (it needs the server only for retrieval in maps of more keyframes than are matched exhaustively: map keyframes greater than 150), and every read-only endpoint (health, maps).
* **Never start or stop `server.sh` or the inference server**, and never kill their processes:
  the user runs them.
* **Never write into a map's folder** (`<data>/maps/<name>/`, not even through the shell on the
  Mac): maps change only through `mapper-update`.
* **Ask the user first** before updating an existing map (`mapper-update` with the name of a map that
  `GET /api/maps/<name>` finds) and before starting a long mapping request (`mapper-update` on new
  inputs runs 15 stages, from `setup` to `commit`, and can take many minutes). Say what follows when you ask: an
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

## Errors

Every error is

```json
{"error": {"code": "<code>", "message": "<the command's own message>", "http_status": <status>}}
```

A refused request (and validate's `problems`) adds `rule`, `parameters` (those it concerns),
`exit_code`, `problems` (every problem) and `by_parameter` (the messages per parameter). For
example, `POST /api/ops/reconstruct` with the body `{}` answers 400 `{"error":{"rule":"arguments","parameters":["image"],"message":"the following arguments are required: -i","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["the following arguments are required: -i"]}}}`. A request whose command ran
and failed answers its `exit_code` with the command's message, e.g. 422 `{"error":{"code":"not_registered","exit_code":5,"message":"<the command's message>","http_status":422}}`. The code is the
name of the command's exit status, mapped to an HTTP status by one rule (input errors 4xx,
inference server unavailable 503, internal 500):

| Code | Exit status | HTTP |
|---|---|---|
| `ok` | 0 | 200 |
| `internal` | 1 | 500 |
| `usage` | 2 | 400 |
| `server_unavailable` | 3 | 503 |
| `not_a_map` | 4 | 422 |
| `not_registered` | 5 | 422 |
| `map_locked` | 6 | 409 |
| `interrupted` | 130 | 499 |

The service's own refusals have the same shape:

| Code | HTTP | When |
|---|---|---|
| `not_found` | 404 | no such map, upload or operation |
| `forbidden` | 403 | a `Host` that does not name the service's machine, or a foreign `Origin` |
| `unsupported_media_type` | 415 | a POST whose body is not `application/json`, or an upload sent as a form (`curl -F`) |
| `too_large` | 413 | an upload over 8 GiB |
| `insufficient_storage` | 413 | an upload that would leave less than 1 GiB free on the workspace's disk |
| `upload_in_use` | 409 | an upload that another request in progress was given (run it, or delete it, with an upload of its own) |
| `stopping` | 503 | a request while the service stops, or one whose command it interrupted when it stopped |

## Request workflow

Every operation runs within its own request: the service answers once the command has ended,
with its result. The example is `reconstruct` (`reconstruct.sh`); every operation in Operations works the
same way.

1. **Check the service** and the inference server.
   ```sh
   curl -sS "$BASE/api/health"
   ```
   `inference.status` is one of `down`, `loading`, `ready`, `error`, `stopping`. An operation that needs inference is accepted
   while it is `ready` or `loading`; otherwise see Safety. `service.requests` counts the requests
   `running` and those `waiting` for their turn at the inference server.

2. **Upload each input file**, or skip this and name a path inside the workspace. The request
   body is the raw file, streamed with `-T` (not a form: `curl -F` is refused with 415). `name` is
   the file name the command sees: keep its suffix (it tells the type) and use only letters,
   digits, `.`, `_` and `-`.
   ```sh
   curl -sS -X POST -T photo.jpg -H 'Content-Type: application/octet-stream' "$BASE/api/uploads?name=photo.jpg"
   ```
   → 201 `{"id":"<upload>","name":"photo.jpg","size":2481152,"path":"uploads/<upload>/photo.jpg"}`

   The `path` is the parameter's value. A parameter that takes several files (a JSON list) takes
   one upload per file, listed in the order the command should read them. The request you give
   an upload to consumes it (see Safety).

3. **Validate** the request: the command's own checks, and the inference server's when the
   request needs it; nothing runs, and no upload is consumed.
   ```sh
   curl -sS -X POST "$BASE/api/ops/reconstruct/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
   ```
   → `{"valid":true,"command":["reconstruct.sh","-i=uploads/<upload>/photo.jpg"],"inference":true,"problems":[],"by_parameter":{}}`

   With `"valid":false`, `problems` and `by_parameter` say what to change, in the command's
   words. `command` is the command line the request will run.

4. **Run it and wait for the answer**, with no client timeout. Nothing comes back until the
   command ends: a mapping request can take many minutes, and a request that uses the inference
   server may first wait for its turn. Disconnecting interrupts the command (see Safety). Save
   the body with `-o` and the headers with `-D`; `-w` prints the HTTP status.
   ```sh
   curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/reconstruct" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
   ```
   → 200, `Content-Type: application/json`, `Server-Timing: connect;dur=500.0, inference;dur=500.0, segment;dur=500.0, export;dur=500.0, write;dur=500.0, total;dur=2500.0`; `result.json` holds the result: byte-identical to what the command writes to stdout, so never rewrite it. Any other
   status means the file holds the error of Errors instead.

5. **Read the stage timings** from the `Server-Timing` header: each of the command's own stages
   (here `connect` → `inference` → `segment` → `export` → `write`) under its own name, then `total`, in milliseconds.
   ```sh
   grep -i '^server-timing:' headers.txt
   ```
   → `server-timing: connect;dur=500.0, inference;dur=500.0, segment;dur=500.0, export;dur=500.0, write;dur=500.0, total;dur=2500.0`

6. **Report** the result file to the user, or the error's `message` as it is. A map's summary and
   its update history are at `GET /api/maps/<name>`, and the browser application at `$BASE/`
   runs the same operations.

## Operations

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

### `reconstruct` — `reconstruct.sh`

`POST /api/ops/reconstruct` · `POST /api/ops/reconstruct/validate`. Single-image reconstruction (stdout or -o file). **Needs the inference server** (reconstructs the image with the inference server): it waits for its turn. Read-only: the result is the response; nothing is kept.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `image` (required) | `-i` | workspace path: .bmp .jpeg .jpg .png .tif .tiff .webp |  | input RGB image |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
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

Answers with:

* The response when `format` is `json` (application/json): the OpenLABEL 1.0.0 scene description (spec §3): objects, labels, scores, colours and OBBs in the camera frame.
* The response when `format` is `ply` (application/octet-stream): the point cloud (camera frame, metres) shaped by -p.

Stages (`Server-Timing`): `connect` → `inference` → `segment` → `export` → `write`. Checks: -p sets point-cloud attributes, which only the PLY output has: use -f ply; every key and value is valid for the command (spec §2.2); the -i image exists. Refused with: `usage` (400), `server_unavailable` (503); a run can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/reconstruct/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/reconstruct" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
```

→ validate: `{"valid":true,"command":["reconstruct.sh","-i=uploads/<upload>/photo.jpg"],"inference":true,"problems":[],"by_parameter":{}}`

→ run: 200, `Content-Type: application/json`, `Server-Timing: connect;dur=500.0, inference;dur=500.0, segment;dur=500.0, export;dur=500.0, write;dur=500.0, total;dur=2500.0`; `result.json` holds the result (see Request workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["image"],"message":"the following arguments are required: -i","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["the following arguments are required: -i"]}}}`

### `mapper-update` — `mapper.sh update`

`POST /api/ops/mapper-update` · `POST /api/ops/mapper-update/validate`. Create or extend a map. **Needs the inference server** (infers depth and objects of every new keyframe): it waits for its turn. **Writes the map** named by `map` (the map folder, created or extended): ask the user first (see Safety).

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `inputs` (required) | `-i` | list of workspace paths, in order (the order matters): .avi .bmp .jpeg .jpg .m4v .mkv .mov .mp4 .png .tif .tiff .webm .webp |  | image files, or exactly one video |
| `map` (required) | `-m` | map name, `<name>` or `maps/<name>`: an existing map, or a new name to create |  | map folder |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
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

Answers with:

* The response when `format` is `json` (application/json): the OpenLABEL 1.0.0 scene description (spec §3) of the whole map (-t full) or of the new input (-t single), map coordinates.
* The response when `format` is `ply` (application/octet-stream): the map cloud (-t full) or the new frames' points (-t single).
* The map `map` of the workspace: the map folder, created or extended (`GET /api/maps/<map>`).

Stages (`Server-Timing`): `setup` → `ingest` → `inference` → `sfm` → `features_matching` → `pose_refinement` → `focal_rerun` → `map_frame` → `depth_alignment` → `persist_frames` → `validity` → `objects` → `cloud` → `export` → `commit`. Checks: -p sets point-cloud attributes, which only the PLY output has: use -f ply; every key and value is valid for the command (spec §2.2); -fps must be positive for a video; for images it is ignored with a warning; -i names existing image files, in order, or exactly one video; -m is a map, an empty folder or a new one; any other folder is refused and left untouched. Refused with: `usage` (400), `server_unavailable` (503), `not_a_map` (422); a run can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/mapper-update/validate" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/video.mp4"],"map":"<map>"}'
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/mapper-update" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/video.mp4"],"map":"<map>"}'
```

→ validate: `{"valid":true,"command":["mapper.sh","update","-i","uploads/<upload>/video.mp4","-m=maps/<map>"],"inference":true,"problems":[],"by_parameter":{}}`

→ run: 200, `Content-Type: application/json`, `Server-Timing: setup;dur=500.0, ingest;dur=500.0, inference;dur=500.0, sfm;dur=500.0, features_matching;dur=500.0, pose_refinement;dur=500.0, focal_rerun;dur=500.0, map_frame;dur=500.0, depth_alignment;dur=500.0, persist_frames;dur=500.0, validity;dur=500.0, objects;dur=500.0, cloud;dur=500.0, export;dur=500.0, commit;dur=500.0, total;dur=7500.0`; `result.json` holds the result (see Request workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["inputs","map"],"message":"the following arguments are required: -i, -m","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"inputs":["the following arguments are required: -i, -m"]}}}`

### `mapper-locate` — `mapper.sh locate`

`POST /api/ops/mapper-locate` · `POST /api/ops/mapper-locate/validate`. Camera pose of images in an existing map (read-only). **Needs the inference server** only for retrieval in maps of more keyframes than are matched exhaustively (map keyframes greater than 150): then it waits for its turn. Read-only: the result is the response; nothing is kept.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `inputs` (required) | `-i` | list of workspace paths: .bmp .jpeg .jpg .png .tif .tiff .webp |  | one or more image files (a video is refused) |
| `map` (required) | `-m` | map name, `<name>` or `maps/<name>`: an existing map |  | existing map folder |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
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

Answers with:

* The response when `format` is `json` (application/json): the OpenLABEL 1.0.0 scene description (spec §3): the located camera poses (-t single), or the whole map plus them (-t full).
* The response when `format` is `ply` (application/octet-stream): the map points visible from the located cameras (-t single) or the whole map cloud (-t full); the located poses in the header.

Stages (`Server-Timing`): `setup` → `features_matching` → `pose` → `export`. Checks: -p sets point-cloud attributes, which only the PLY output has: use -f ply; every key and value is valid for the command (spec §2.2); -i names image files; a video is refused; -m is an existing map; a missing or empty folder is not created, a non-empty folder that is not a map is refused. Refused with: `usage` (400), `server_unavailable` (503), `not_a_map` (422); a run can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/mapper-locate/validate" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/query.jpg"],"map":"<map>"}'
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/mapper-locate" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/query.jpg"],"map":"<map>"}'
```

→ validate: `{"valid":true,"command":["mapper.sh","locate","-i","uploads/<upload>/query.jpg","-m=maps/<map>"],"inference":false,"problems":[],"by_parameter":{}}`

→ run: 200, `Content-Type: application/json`, `Server-Timing: setup;dur=500.0, features_matching;dur=500.0, pose;dur=500.0, export;dur=500.0, total;dur=2000.0`; `result.json` holds the result (see Request workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["inputs","map"],"message":"the following arguments are required: -i, -m","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"inputs":["the following arguments are required: -i, -m"]}}}`

### `segment-image` — `segment.sh -i`

`POST /api/ops/segment-image` · `POST /api/ops/segment-image/validate`. Instance segmentation → JSON + OBBs, artefacts. **Needs the inference server** (segments the image with the inference server): it waits for its turn. Read-only: the result is the response; nothing is kept.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `image` (required) | `-i` | workspace path: .bmp .jpeg .jpg .png .tif .tiff .webp |  | input RGB image |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
| `attrs` | `-p` | `key=value,…` (keys below), or a list of them | `color=segment,stride=1,min-depth=0,max-depth=inf,edge=0.04,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=segment (fixed) [segment], stride=N [1], min-depth=METRES [0], max-depth=METRES\|inf [inf], edge=JUMP [0.04], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; shapes the -f ply output and segments.ply, so it needs -f ply or -d; with -m the pixel-level keys (stride, min-depth, max-depth, edge) are refused (only with -f ply or -d) |
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

Answers with:

* The response when `format` is `json` (application/json): the OpenLABEL 1.0.0 scene description (spec §3) (camera frame).
* The response when `format` is `ply` (application/octet-stream): the object-coloured point cloud.

Stages (`Server-Timing`): `connect` → `inference` → `segment` → `export` → `artifacts` → `write`. Checks: --min-score is a finite number; -p sets point-cloud attributes, which shape the PLY output: use -f ply or -d <folder>; every key and value is valid for the command (spec §2.2); the -i image exists. Refused with: `usage` (400), `server_unavailable` (503); a run can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/segment-image/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/segment-image" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
```

→ validate: `{"valid":true,"command":["segment.sh","-i=uploads/<upload>/photo.jpg"],"inference":true,"problems":[],"by_parameter":{}}`

→ run: 200, `Content-Type: application/json`, `Server-Timing: connect;dur=500.0, inference;dur=500.0, segment;dur=500.0, export;dur=500.0, artifacts;dur=500.0, write;dur=500.0, total;dur=3000.0`; `result.json` holds the result (see Request workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["image","map"],"message":"one of the arguments -i -m is required","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["one of the arguments -i -m is required"]}}}`

### `segment-map` — `segment.sh -m`

`POST /api/ops/segment-map` · `POST /api/ops/segment-map/validate`. Instance segmentation → JSON + OBBs, artefacts. **Works without the inference server** (exports the map's persistent objects without inference), at once. Read-only: the result is the response; nothing is kept.

| Parameter | Flag | Value | Default | Meaning |
|---|---|---|---|---|
| `map` (required) | `-m` | map name, `<name>` or `maps/<name>`: an existing map |  | existing map folder (read-only) |
| `format` | `-f` | one of `json`, `ply` | `json` | output format (default: json) |
| `attrs` | `-p` | `key=value,…` (keys below), or a list of them | `color=segment,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=segment (fixed) [segment], stride=N [1], min-depth=METRES [0], max-depth=METRES\|inf [inf], edge=JUMP [0.04], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; shapes the -f ply output and segments.ply, so it needs -f ply or -d; with -m the pixel-level keys (stride, min-depth, max-depth, edge) are refused (only with -f ply or -d) |

Keys of `attrs` (keys not given keep their default):

| Key | Values | Default | Effect |
|---|---|---|---|
| `color` | `segment` | `segment` | per-point colour: image colour, object colour, height ramp, or no colour |
| `voxel` | number ≥ 0 | `0` | keep one point per voxel of this size (0 = off; colours are not averaged) |
| `normals` | `on` \| `off` | `off` | add nx ny nz float properties |
| `label` | `on` \| `off` | `off` | add an int label property (object id, 0 = unsegmented) |
| `encoding` | `binary` \| `ascii` | `binary` | binary_little_endian 1.0 or ASCII PLY |

Answers with:

* The response when `format` is `json` (application/json): the OpenLABEL 1.0.0 scene description (spec §3) of the map's objects (map coordinates).
* The response when `format` is `ply` (application/octet-stream): the object-coloured map cloud.

Stages (`Server-Timing`): `export` → `artifacts` → `write`. Checks: -p sets point-cloud attributes, which shape the PLY output: use -f ply or -d <folder>; every key and value is valid for the command (spec §2.2); -m is a map (it is opened read-only). Refused with: `usage` (400), `not_a_map` (422); a run can end with any code of the table in Errors.

```sh
curl -sS -X POST "$BASE/api/ops/segment-map/validate" -H 'Content-Type: application/json' -d '{"map":"<map>"}'
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/segment-map" -H 'Content-Type: application/json' -d '{"map":"<map>"}'
```

→ validate: `{"valid":true,"command":["segment.sh","-m=maps/<map>"],"inference":false,"problems":[],"by_parameter":{}}`

→ run: 200, `Content-Type: application/json`, `Server-Timing: export;dur=500.0, artifacts;dur=500.0, write;dur=500.0, total;dur=1500.0`; `result.json` holds the result (see Request workflow).

→ refused, e.g. the body `{}`: 400 `{"error":{"rule":"arguments","parameters":["image","map"],"message":"one of the arguments -i -m is required","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["one of the arguments -i -m is required"]}}}`

## Endpoints

The service's own endpoints, besides the operations. In paths, `{id}` is an upload's id and `{name}` a map's name. They answer an error of Errors when they fail, except a path or method the service does not have, which answers plain text (404 `Not Found`, 405 `Method Not Allowed`).

### Service

#### `GET /api/health`

Read-only. The service (version, URL, workspace, and the requests `running` and `waiting` for their turn) and the inference server: `inference.status` is one of `down`, `loading`, `ready`, `error`, `stopping`; `start_command` is the command that starts it when it is not `ready` (report it, never run it), and `health` its own health when it answers.

```sh
curl -sS "$BASE/api/health"
```

→ `{"status":"ok","service":{"version":"<version>","url":"http://0.0.0.0:52026/","pid":4321,"workspace":"oh-my-slam-data","data":"/Users/<user>/oh-my-slam-data","started_at":1791281700.0,"requests":{"running":1,"waiting":0}},"inference":{"status":"down","message":"<why>","start_command":"./start_inference_server.sh"}}`

#### `GET /api/openapi.json`

Read-only. The running service's OpenAPI 3.1 document: every operation with its parameters, and its whole definition under `x-oms` (parameters, outputs, rules, errors, stages, inference need). It wins where it disagrees with this file.

```sh
curl -sS -o openapi.json "$BASE/api/openapi.json"
```

→ `{"openapi":"3.1.0","info":{…},"paths":{"/api/ops/<op>":{"post":{…,"x-oms":{…}}},…},"components":{…},"x-oms":{"exit_codes":[…],"stages":[…]}}`

### Uploads

#### `POST /api/uploads`

Stores a file in the workspace until the request it is given to ends. The request body is the file itself (`-T`), with its own media type or `application/octet-stream`; a form (`curl -F`, multipart) is refused with 415, since a cross-site web page could send one. At most 8 GiB, and it must leave 1 GiB free (413). Answers 201; use its `path` as an input parameter of one request. Query: `name`: the file name (its suffix tells its type).

```sh
curl -sS -X POST -T photo.jpg -H 'Content-Type: application/octet-stream' "$BASE/api/uploads?name=photo.jpg"
```

→ `{"id":"<upload>","name":"photo.jpg","size":2481152,"path":"uploads/<upload>/photo.jpg"}`

#### `DELETE /api/uploads/{id}`

Deletes an upload. Only one that no request in progress was given (409 `upload_in_use` otherwise: it goes when that request ends); 404 `not_found` for no such upload.

```sh
curl -sS -X DELETE -w '%{http_code}\n' "$BASE/api/uploads/<upload>"
```

→ 204, no body

### Maps (read-only)

#### `GET /api/maps`

Read-only. Every map of the workspace with summary figures from its own metadata: each scalar of its `map.json`, the size of each of its lists (`<key>_count`), `frames`, `objects` and `last_update`.

```sh
curl -sS "$BASE/api/maps"
```

→ `[{"name":"<map>","path":"maps/<map>",…,"frames":302,"objects":143,"last_update":{"id":2,"at":1790931612.9,"kind":"video","frames_added":120,"total_s":512.4}},…]`

#### `GET /api/maps/{name}`

Read-only. One map's summary plus `meta`, its whole `map.json`, whose `updates[]` hold each update's record with its `timings`. 404 `not_found` when there is no such map.

```sh
curl -sS "$BASE/api/maps/<map>"
```

→ `{"name":"<map>","path":"maps/<map>",…,"meta":{…,"updates":[{…,"timings":{…}},…],…}}`
