---
name: oh-my-slam
description: "oh-my-slam monocular RGB 3D mapping on a Mac. Scripts, with its checkout on this machine: start_inference_server.sh start, --status → JSON, --stop; reconstruct.sh: image → JSON/PNG/PLY; mapper.sh update: images/video + map → JSON/PLY + map, locate: images + map → JSON/PLY; segment.sh -i: image → JSON/PNG + files; view.sh -i: image → web viewer, -m: map → web viewer; server.sh --status → JSON. API, with curl and a server.sh reachable from this machine (LAN too): operations reconstruct, mapper-update, mapper-locate, segment-image run those modes; also uploads, validation, maps, health. JSON = OpenLABEL scene (labelled objects, oriented bounding boxes), PNG = depth image or segmented image, PLY = point cloud. Inference server needed by reconstruct.sh, mapper.sh update, segment.sh -i and view.sh -i, and by mapper.sh locate for maps over 150 keyframes; view.sh -m works without it. Use it to reconstruct, map, locate, segment or view images/video in 3D."
---

# oh-my-slam

oh-my-slam turns RGB images and video into 3D on a Mac: scene descriptions of labelled objects
with oriented bounding boxes, point clouds, and persistent maps in which images can be located.
Two kinds of entry point give the same results, byte for byte:

* **[Scripts](#scripts)**: the shell scripts at the root of the oh-my-slam checkout, on the Mac
  that holds it.
* **[API](#api)**: `server.sh`, a web service that runs every mode of `reconstruct.sh`, `mapper.sh` and `segment.sh` within an
  HTTP request, for this Mac or any machine on the LAN, with `curl`.

Use the scripts when the checkout is on this machine, else the API, and follow the
[Rules](#rules). This file is generated from the project's own definitions (`uv run python -m
oh_my_slam.web.skill` in the checkout): offer only what it, or the running service's
`/api/openapi.json`, lists.

| Script | Modes | What it does |
|---|---|---|
| `start_inference_server.sh` | `start_inference_server.sh`, `start_inference_server.sh --status`, `start_inference_server.sh --stop` | Start (idempotent), stop or query the inference server. |
| `reconstruct.sh` | `reconstruct.sh` | Single-image reconstruction (stdout or -o file). |
| `mapper.sh` | `mapper.sh update`, `mapper.sh locate` | Multi-frame mapping (persistent map). |
| `segment.sh` | `segment.sh -i` | Instance segmentation → JSON + OBBs or segmented image, artefacts. |
| `view.sh` | `view.sh -i`, `view.sh -m` | Browser visualisation of an image or a map. |
| `server.sh` | `server.sh --status` | Local web service: the commands as an HTTP API and a browser application. |

Results: JSON is an ASAM OpenLABEL 1.0.0 scene description (or a server's health, for `--status`): each object has a label (`type`), a score, a colour that is the same in every output, and an oriented bounding box `cuboid` whose `val` is `x,y,z,qx,qy,qz,qw,sx,sy,sz` (metres, quaternion scalar last), e.g. `{"openlabel":{"metadata":{"schema_version":"1.0.0",…},"objects":{"1":{"name":"chair 1","type":"chair","ontology_uid":"0","coordinate_system":"camera","object_data":{"cuboid":[{"name":"obb","val":[0.41,0.18,2.35,0.0,0.0,0.0,1.0,0.48,0.51,0.92],"coordinate_system":"camera"}],"num":[{"name":"score","val":0.91}]},…},…},…}}`; PNG is the depth image (`reconstruct.sh` with `-f depth`) or the segmented image (`segment.sh -i` with `-f png`); PLY is a point cloud in metres whose header records its attributes (`-p`).

## Rules

* **The inference server may be down.** Then the scripts `reconstruct.sh`, `mapper.sh update`, `segment.sh -i` and `view.sh -i` fail with exit
  3, and the operations `reconstruct`, `mapper-update` and `segment-image` with HTTP 503 `server_unavailable`, each with
  a message that names the start command `./start_inference_server.sh` (`GET /api/health` gives it as
  `inference.start_command`). Report the message and that command to the user instead of
  retrying, and offer what works without it: the script `view.sh -m`; `mapper.sh locate` and `mapper-locate` except for maps over 150 keyframes.
* **Never start or stop `server.sh` or the inference server**, and never kill their processes:
  the user runs `start_inference_server.sh`, `start_inference_server.sh --stop` and `server.sh`. Only `start_inference_server.sh --status` and `server.sh --status` are yours to run.
* **Never write into a map's folder** (`<data>/maps/<name>/`, or any folder a script takes as a
  map): maps change only through `mapper.sh update` and `mapper-update`.
* **Ask the user first** before updating an existing map, and before starting a long mapping
  request: `mapper.sh update` and `mapper-update` run 15 stages, from `setup` to `commit`, and can take many minutes. Say that an update changes
  the map for good.
* **API inputs are uploads or paths inside the workspace** (relative to it, such as
  `uploads/<upload>/photo.jpg`), never paths outside it, which are refused (400 `usage`). **An
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

## Scripts

### Find the checkout

The scripts are at the root of the oh-my-slam checkout. This snippet prints the checkout's
absolute path and caches it in `${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/repo`. It tries, in order: the path the user gives
(`OMS_REPO`); the cached path, if it still holds the scripts. Otherwise it fails: ask the user for
the path of the checkout (the folder that holds `reconstruct.sh`) and run it again with the first
line changed to `OMS_REPO=<path> sh <<'EOF'`.

```sh
sh <<'EOF'
cache="${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/repo"
holds() {  # the checkout: the scripts at its root
  for s in start_inference_server.sh reconstruct.sh mapper.sh segment.sh view.sh server.sh; do
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
EOF
```

Your shell keeps no variables between commands, so begin **every** command with
`REPO='<that path>';` and call the scripts by their absolute path, from any directory, with
absolute paths for every file and folder, e.g. `REPO='/Users/me/oh-my-slam'; "$REPO/reconstruct.sh" -i "$REPO/examples/restaurant.jpg" -o "$PWD/result.json"`. The
checkout needs `uv sync` once (a script whose Python environment is missing says so: ask the user
to run it there), and the scripts that use the inference server need it running (see
[Rules](#rules)).

Every script writes at most one result on stdout (JSON, PNG, PLY), or with `-o` to that file instead;
everything else goes to stderr: progress, and on failure what went wrong, with the exit status of
[Errors](#errors).

### `start_inference_server.sh`

`"$REPO/start_inference_server.sh"`: Start (idempotent), stop or query the inference server. **Starts a server:** starts the inference server in the background and returns once its models are loaded; a server that already runs is left as it is. The user runs it: never run it yourself; give the user the command.

No options.

```sh
"$REPO/start_inference_server.sh"
```

→ nothing on stdout; it says on stderr what it did

Exit statuses on failure: 2 (`usage`), or another of [Errors](#errors).

### `start_inference_server.sh --status`

`"$REPO/start_inference_server.sh"`: Start (idempotent), stop or query the inference server. **Needs the inference server** (prints the inference server's health; exit 3 when it is not running). Read-only.

No options.

Result on stdout: the inference server's health: its status (loading, ready, error, stopping), device, precision and each model's load state.

```sh
"$REPO/start_inference_server.sh" --status
```

→ stdout: `{"status":"ready","models":{…},"device":"mps","precision":"fp16","versions":{…},"pid":4321,"protocol":1,"queue_depth":0,"queue_limit":8,"uptime_s":812.5,"detail":null}`

Exit statuses on failure: 2 (`usage`), 3 (`server_unavailable`), or another of [Errors](#errors); e.g. `start_inference_server.sh: error: inference server is not running — start it with ./start_inference_server.sh`.

### `start_inference_server.sh --stop`

`"$REPO/start_inference_server.sh"`: Start (idempotent), stop or query the inference server. **Stops a server:** stops the inference server; its socket and state file are removed. The user runs it: never run it yourself; give the user the command.

No options.

```sh
"$REPO/start_inference_server.sh" --stop
```

→ nothing on stdout; it says on stderr what it did

Exit statuses on failure: 2 (`usage`), or another of [Errors](#errors).

### `reconstruct.sh`

`"$REPO/reconstruct.sh"`: Single-image reconstruction (stdout or -o file). **Needs the inference server** (reconstructs the image with the inference server). Read-only; it writes only to `-o`.

| Option | Value | Default | Meaning |
|---|---|---|---|
| `-i IMAGE` (required) | file: .bmp .jpeg .jpg .png .tif .tiff .webp |  | input RGB image |
| `-f` | `json`, `depth`, `ply` | `json` | output format: json = the scene description, depth = the depth image (16-bit PNG), ply = the point cloud (default: json) |
| `-o FILE` | file |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `-p ATTRS` | `key=value,…`, repeatable | `color=rgb,stride=1,min-depth=0,max-depth=inf,edge=0.04,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=rgb\|segment\|height\|none [rgb], stride=N [1], min-depth=METRES [0], max-depth=METRES\|inf [inf], edge=JUMP [0.04], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; requires -f ply (only with -f ply) |

Result on stdout, or in the `-o` file: with `-f json`, the OpenLABEL 1.0.0 scene description (spec §3): objects, labels, scores, colours and OBBs in the camera frame; with `-f depth`, the depth image: one 16-bit single-channel PNG of the input's pixel size, each pixel the metric depth along the optical axis in units of 1/256 m, 0 where the model gives no valid depth; with `-f ply`, the point cloud (camera frame, metres) shaped by -p.

```sh
"$REPO/reconstruct.sh" -i "$REPO/examples/restaurant.jpg" -o "$PWD/result.json"
```

→ `result.json` (stdout stays empty): `{"openlabel":{"metadata":{"schema_version":"1.0.0",…},"objects":{"1":{"name":"chair 1","type":"chair",…},…},…}}`

Exit statuses on failure: 2 (`usage`), 3 (`server_unavailable`), or another of [Errors](#errors); e.g. `reconstruct.sh: error: inference server is not running — start it with ./start_inference_server.sh`.

### `mapper.sh update`

`"$REPO/mapper.sh" update`: Create or extend a map. **Needs the inference server** (infers depth and objects of every new keyframe). **Writes the map** `-m`: ask the user first (see [Rules](#rules)).

| Option | Value | Default | Meaning |
|---|---|---|---|
| `-i INPUTS…` (required) | files, in order: .avi .bmp .jpeg .jpg .m4v .mkv .mov .mp4 .png .tif .tiff .webm .webp |  | image files, or exactly one video |
| `-m MAP` (required) | map folder: a map, or a new or empty folder |  | map folder |
| `-f` | `json`, `ply` | `json` | output format (default: json) |
| `-o FILE` | file |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `-p ATTRS` | `key=value,…`, repeatable | `color=rgb,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=rgb\|segment\|height\|none [rgb], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; requires -f ply (only with -f ply) |
| `-t` | `full`, `single` | `full` | full = whole map with all keyframe poses; single = new input only (default: full) |
| `-fps FPS` | number > 0 | `2` | video frames per second to sample (default: 2; ignored for images) (video input only; ignored for images) |

Result on stdout, or in the `-o` file: with `-f json`, the OpenLABEL 1.0.0 scene description (spec §3) of the whole map (-t full) or of the new input (-t single), map coordinates; with `-f ply`, the map cloud (-t full) or the new frames' points (-t single). The map `-m`: the map folder, created or extended.

```sh
"$REPO/mapper.sh" update -i "$REPO"/examples/office_sequence/*.jpg -m "$HOME/oh-my-slam-data/maps/office" -o "$PWD/result.json"
```

→ `result.json` (stdout stays empty): `{"openlabel":{"metadata":{"schema_version":"1.0.0",…},"objects":{"1":{"name":"chair 1","type":"chair",…},…},…}}`

Exit statuses on failure: 2 (`usage`), 3 (`server_unavailable`), 4 (`not_a_map`), or another of [Errors](#errors); e.g. `mapper.sh: error: inference server is not running — start it with ./start_inference_server.sh`.

### `mapper.sh locate`

`"$REPO/mapper.sh" locate`: Camera pose of images in an existing map (read-only). **Needs the inference server** only for retrieval in maps of more keyframes than are matched exhaustively (maps over 150 keyframes). Read-only; it writes only to `-o`.

| Option | Value | Default | Meaning |
|---|---|---|---|
| `-i INPUTS…` (required) | files: .bmp .jpeg .jpg .png .tif .tiff .webp |  | one or more image files (a video is refused) |
| `-m MAP` (required) | map folder: an existing map |  | existing map folder |
| `-f` | `json`, `ply` | `json` | output format (default: json) |
| `-o FILE` | file |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `-p ATTRS` | `key=value,…`, repeatable | `color=rgb,voxel=0,normals=off,label=off,encoding=binary` | point-cloud attributes key=value[,key=value...], defaults in brackets: color=rgb\|segment\|height\|none [rgb], voxel=METRES [0], normals=on\|off [off], label=on\|off [off], encoding=binary\|ascii [binary]; requires -f ply (only with -f ply) |
| `-t` | `full`, `single` | `single` | single = the located camera poses only; full = the whole map plus the located poses (default: single) |

Result on stdout, or in the `-o` file: with `-f json`, the OpenLABEL 1.0.0 scene description (spec §3): the located camera poses (-t single), or the whole map plus them (-t full); with `-f ply`, the map points visible from the located cameras (-t single) or the whole map cloud (-t full); the located poses in the header.

```sh
"$REPO/mapper.sh" locate -i "$REPO"/examples/office_sequence/*.jpg -m "$HOME/oh-my-slam-data/maps/office" -o "$PWD/result.json"
```

→ `result.json` (stdout stays empty): `{"openlabel":{"metadata":{"schema_version":"1.0.0",…},"objects":{"1":{"name":"chair 1","type":"chair",…},…},…}}`

Exit statuses on failure: 2 (`usage`), 3 (`server_unavailable`), 4 (`not_a_map`), or another of [Errors](#errors); e.g. `mapper.sh: error: inference server is not running — start it with ./start_inference_server.sh`.

### `segment.sh -i`

`"$REPO/segment.sh"`: Instance segmentation → JSON + OBBs or segmented image, artefacts. **Needs the inference server** (segments the image with the inference server). Read-only; it writes only to `-o` and `-d`.

| Option | Value | Default | Meaning |
|---|---|---|---|
| `-i IMAGE` (required) | file: .bmp .jpeg .jpg .png .tif .tiff .webp |  | input RGB image |
| `-f` | `json`, `png` | `json` | output format: json = the scene description, png = the segmented image (default: json) |
| `-o FILE` | file |  | write the result to FILE instead of stdout (stdout then stays empty) |
| `-d FOLDER` | folder |  | also write segmentation.json, segmented.png, catalog.csv and catalog.md into FOLDER |
| `--min-score MIN_SCORE` | finite number | `0.5` | drop detections below this score (default 0.5) |

Result on stdout, or in the `-o` file: with `-f json`, the OpenLABEL 1.0.0 scene description (spec §3) (camera frame); with `-f png`, the segmented image: the input image dimmed, each instance mask painted opaque in its object's colour. With `-d FOLDER` it also writes there: `segmentation.json`, the OpenLABEL 1.0.0 scene description (spec §3), identical to -f json; `segmented.png`, the segmented image, identical to -f png; `catalog.csv`, one row per object; `catalog.md`, the catalogue as a table by descending volume.

```sh
"$REPO/segment.sh" -i "$REPO/examples/restaurant.jpg" -o "$PWD/result.json"
```

→ `result.json` (stdout stays empty): `{"openlabel":{"metadata":{"schema_version":"1.0.0",…},"objects":{"1":{"name":"chair 1","type":"chair",…},…},…}}`

Exit statuses on failure: 2 (`usage`), 3 (`server_unavailable`), or another of [Errors](#errors); e.g. `segment.sh: error: inference server is not running — start it with ./start_inference_server.sh`.

### `view.sh -i`

`"$REPO/view.sh"`: Browser visualisation of an image or a map. **Needs the inference server** (reconstructs and segments the image with the inference server). **Runs until interrupted** (Ctrl-C): start it in the background and give the user the URL it prints.

| Option | Value | Default | Meaning |
|---|---|---|---|
| `-i IMAGE` (required) | file: .bmp .jpeg .jpg .png .tif .tiff .webp |  | RGB image to reconstruct and segment |
| `--no-browser` | flag |  | do not open a browser |

It serves the viewer page (URL on stderr).

```sh
"$REPO/view.sh" -i "$REPO/examples/restaurant.jpg"
```

→ stderr: `view.sh: listening on http://127.0.0.1:<port>/`, and the browser opens on it (not with `--no-browser`); nothing on stdout

Exit statuses on failure: 2 (`usage`), 3 (`server_unavailable`), or another of [Errors](#errors); e.g. `view.sh: error: inference server is not running — start it with ./start_inference_server.sh`.

### `view.sh -m`

`"$REPO/view.sh"`: Browser visualisation of an image or a map. Works without the inference server (opens the persisted map read-only). **Runs until interrupted** (Ctrl-C): start it in the background and give the user the URL it prints.

| Option | Value | Default | Meaning |
|---|---|---|---|
| `-m MAP` (required) | map folder: an existing map |  | map folder (opened read-only) |
| `--no-browser` | flag |  | do not open a browser |

It serves the viewer page (URL on stderr).

```sh
"$REPO/view.sh" -m "$HOME/oh-my-slam-data/maps/office"
```

→ stderr: `view.sh: listening on http://127.0.0.1:<port>/`, and the browser opens on it (not with `--no-browser`); nothing on stdout

Exit statuses on failure: 2 (`usage`), 4 (`not_a_map`), or another of [Errors](#errors); e.g. `view.sh: error: one of the arguments -i -m is required`.

### `server.sh --status`

`"$REPO/server.sh"`: Local web service: the commands as an HTTP API and a browser application. Works without the inference server (prints the running service's health; exit 3 when no service runs for the workspace). Read-only.

| Option | Value | Default | Meaning |
|---|---|---|---|
| `--data DATA` | folder | `~/oh-my-slam-data` | workspace folder for maps and uploads (default: ~/oh-my-slam-data/) |

Result on stdout: the service's health: its URL, workspace, the requests in progress and the inference server's state.

```sh
"$REPO/server.sh" --status
```

→ stdout: `{"status":"ok","service":{"version":"<version>","url":"http://0.0.0.0:52026/","pid":4321,"workspace":"oh-my-slam-data","data":"/Users/<user>/oh-my-slam-data","started_at":1791281700.0,"requests":{"running":0,"waiting":0},"in_progress":[]},"inference":{"status":"ready","health":{…},"start_command":null}}`

Exit statuses on failure: 2 (`usage`), 3 (`server_unavailable`), or another of [Errors](#errors).

## API

`server.sh` is the web service: it runs every mode of `reconstruct.sh`, `mapper.sh` and `segment.sh` as an operation within one HTTP
request. There are no jobs: the service answers when the command ends, with its result, byte for
byte, or its error. It keeps maps and uploads in its workspace (`~/oh-my-slam-data/` unless it was started
with `--data <folder>`), keeps no results, and binds `0.0.0.0`, so it serves this Mac and any
machine on the LAN; you need only `sh` and `curl`.

**`$BASE/api/openapi.json` describes every operation, parameter, result and error.** An
operation's parameters are its command's options, with the same names, values, defaults and
checks (`-o` and `-d` are none: the response is the result), and its whole definition is under
`x-oms`: outputs, checks, errors, stages and inference need. Read it before a request; where it
and this file disagree (an older or newer service), **the running service's document wins**.

| Operation | Runs | Inference server |
|---|---|---|
| `reconstruct` | `reconstruct.sh` | needed |
| `mapper-update` | `mapper.sh update` | needed; **writes the map** |
| `mapper-locate` | `mapper.sh locate` | only for maps over 150 keyframes |
| `segment-image` | `segment.sh -i` | needed |

The routes:

* `POST /api/ops/{op}/validate`: check a request with the command's own checks (and the inference server's, when it needs it); nothing runs
* `POST /api/ops/{op}`: run an operation within the request and answer when its command ends
* `GET /api/openapi.json`: the description of every operation, parameter, result and error
* `GET /api/health`: service and inference-server health, and the requests running or waiting for their turn
* `POST /api/uploads` (query `name`): upload one input file (raw request body)
* `DELETE /api/uploads/{id}`: discard an upload no request was given
* `GET /api/maps`: the workspace's maps with their summaries
* `GET /api/maps/{name}`: a map's summary and metadata (map.json)

### Find the service

`server.sh` binds a free port unless it was started with `--port <n>`, so its URL changes from run
to run. This snippet prints the base URL and caches it in `${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/server_url`. It tries, in order: the URL
the user gives (`OMS_URL`); the cached URL, if `/api/health` answers there within 3 s; on
the Mac that runs the service, the URL the service records in `server.json` in its workspace
(`~/oh-my-slam-data/`, or the `--data` folder the user names, as `OMS_DATA`), with `0.0.0.0` replaced by
`127.0.0.1`. Otherwise it fails: ask the user for the URL that `server.sh` printed when it started
(`server.sh: listening on http://0.0.0.0:<port>/`; from another machine, the Mac's address or host
name in place of `0.0.0.0`) or that `server.sh --status` reports (`service.url`), and run it again
with the first line changed to `OMS_URL=<url> sh <<'EOF'` (or `OMS_DATA=<folder> sh <<'EOF'`).
There is no network or port scan; `server.sh --port <n>` keeps the URL stable for other machines.

```sh
sh <<'EOF'
cache="${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/server_url"
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
  echo "oh-my-slam: no oh-my-slam service answers at $OMS_URL; check the URL with the user" >&2
  exit 1
fi
u=$(clean "$(cat "$cache" 2>/dev/null)")  # 2. the cached URL, while it answers
[ -n "$u" ] && probe "$u" && found "$u"
data=${OMS_DATA:-$HOME/oh-my-slam-data}  # 3. on the service's machine: its server.json
case $data in "~") data=$HOME ;; "~/"*) data=$HOME/${data#"~/"} ;; esac
u=$(clean "$(sed -n 's/.*"url": *"\([^"]*\)".*/\1/p' "$data/server.json" 2>/dev/null)")
[ -n "$u" ] && probe "$u" && found "$u"
echo "oh-my-slam: no service found (tried $cache and $data/server.json)." \
  "Ask the user for the URL that server.sh printed on start or that server.sh --status" \
  "reports, then run this again with OMS_URL=<url>." >&2
exit 1
EOF
```

Begin **every** command with `BASE=<that URL>;` (a `;`, not a `BASE=... curl` prefix), e.g.
`BASE=http://127.0.0.1:<port>; curl -sS "$BASE/api/health"`. Use the URL as found: the service
answers 403 `forbidden` to a `Host` that does not name its machine. `GET /api/health`
names the service's workspace (`service.data`): check it is the one the user means. If a request
later fails to connect (curl exit 7 or 28), run the snippet again: it searches again only
when the cached URL stops answering.

### A request, call by call

The example is `reconstruct` (`reconstruct.sh`); every operation works the same way.

1. **Check the service** and the inference server: `curl -sS "$BASE/api/health"`.
   `inference.status` is one of `down`, `loading`, `ready`, `error`, `stopping`; an operation that needs the inference server is
   accepted while it is `ready` or `loading`. `service.requests` counts the requests `running` and
   those `waiting` for their turn.
2. **Upload each input file**, or name a path inside the workspace instead. The body is the raw
   file, sent with `-T` (a form, `curl -F`, is refused with 415); `name` is the file name the
   command sees: keep its suffix and use only letters, digits, `.`, `_` and `-`. A parameter that
   takes several files (a JSON list) takes one upload per file, in the order the command reads
   them.
   ```sh
   curl -sS -X POST -T photo.jpg -H 'Content-Type: application/octet-stream' "$BASE/api/uploads?name=photo.jpg"
   ```
   → 201 `{"id":"<upload>","name":"photo.jpg","size":2481152,"path":"uploads/<upload>/photo.jpg"}`: its `path` is the parameter's value.
3. **Validate**: the command's own checks, and the inference server's when the request needs it;
   nothing runs and no upload is consumed. `command` is the command line it will run; with
   `"valid":false`, `problems` and `by_parameter` say what to change.
   ```sh
   curl -sS -X POST "$BASE/api/ops/reconstruct/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
   ```
   → `{"valid":true,"command":["reconstruct.sh","-i=uploads/<upload>/photo.jpg"],"inference":true,"problems":[],"by_parameter":{}}`
4. **Run it and wait**, with no client timeout: nothing comes back until the command ends (a
   mapping request can take many minutes, and a request that uses the inference server may first
   wait for its turn). Save the body with `-o`, the headers with `-D`; `-w` prints the status.
   ```sh
   curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/reconstruct" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
   ```
   → 200, `Content-Type: application/json`, `Server-Timing: connect;dur=500.0, inference;dur=500.0, segment;dur=500.0, export;dur=500.0, write;dur=500.0, total;dur=2500.0`; `result.json` holds the result, byte for byte. Any other status means the file holds an error (see [Errors](#errors)).
5. **Read the stage timings** in the `Server-Timing` header: each of the command's own stages
   under its own name, then `total`, in milliseconds: `grep -i '^server-timing:' headers.txt` →
   `server-timing: connect;dur=500.0, inference;dur=500.0, segment;dur=500.0, export;dur=500.0, write;dur=500.0, total;dur=2500.0`.
6. **Report** the result file, or the error's `message` as it is. A map's summary and update
   history are at `GET /api/maps/<name>`; the browser application at `$BASE/` runs the same
   operations.

### Examples

```sh
# mapper.sh update (writes the map: ask the user first)
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/mapper-update" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/video.mp4"],"map":"<map>"}'
# mapper.sh locate
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/mapper-locate" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/query.jpg"],"map":"<map>"}'
# segment.sh -i
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/segment-image" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
```

## Errors

A script that fails says why on stderr, the commands in one line `<script>: error: <message>`
(`<script>: interrupted` on Ctrl-C), and exits with a status of the table. An operation that fails
answers the command's message and the status's code, with the HTTP status that one generic rule
gives that code (the table):

```json
{"error": {"code": "<code>", "message": "<the command's own message>", "http_status": <status>}}
```

A refused request (and validate's `problems`) adds `rule`, `parameters` (those it concerns),
`exit_code`, `problems` (every problem) and `by_parameter`. For example, `POST /api/ops/reconstruct`
with the body `{}` answers 400 `{"error":{"rule":"arguments","parameters":["image"],"message":"the following arguments are required: -i","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["the following arguments are required: -i"]}}}`; a request whose command ran and failed answers its
`exit_code` and message, e.g. 422 `{"error":{"code":"not_registered","exit_code":5,"message":"<the command's message>","http_status":422}}`.

| Exit | Code | HTTP | Meaning |
|---|---|---|---|
| 0 | `ok` | 200 | success |
| 1 | `internal` | 500 | internal error (also: inference failed, or COLMAP is missing or the wrong version) |
| 2 | `usage` | 400 | usage or input error: a bad option or value, a missing or unsupported input file |
| 3 | `server_unavailable` | 503 | a server it needs does not answer: the inference server (not running, or its models failed to load), or for --status the server it queries |
| 4 | `not_a_map` | 422 | the map folder is not a map (and not empty, for an update) |
| 5 | `not_registered` | 422 | nothing could be placed in the map (no overlap); the map is unchanged |
| 6 | `map_locked` | 409 | another update holds the map |
| 130 | `interrupted` | 499 | interrupted (Ctrl-C, or a server.sh request whose client left) |

The service's own refusals have the same shape:

| Code | HTTP | When |
|---|---|---|
| `not_found` | 404 | no such map, upload or operation |
| `forbidden` | 403 | a `Host` that does not name the service's machine, or a foreign `Origin` |
| `unsupported_media_type` | 415 | a POST whose body is not `application/json`, or an upload sent as a form (`curl -F`) |
| `too_large` | 413 | an upload over 8 GiB |
| `insufficient_storage` | 413 | an upload that would leave less than 1 GiB free on the workspace's disk |
| `upload_in_use` | 409 | an upload that another request in progress was given |
| `stopping` | 503 | a request while the service stops, or one whose command it interrupted when it stopped |
