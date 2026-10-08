---
name: oh-my-slam
description: "oh-my-slam monocular RGB 3D mapping on a Mac, through the HTTP API of its server.sh web service with sh and curl, from this machine or another on the LAN; needs a server.sh reachable from this machine, no checkout. Operations: reconstruct: one image → JSON/PNG/PLY; mapper-update: images or one video + a new or existing map → JSON/PLY; mapper-locate: images + an existing map → JSON/PLY; segment: one image → JSON/PNG. mapper-update writes a map; the others are read-only. Also uploads, validation, the workspace's maps, health. JSON = OpenLABEL scene (labelled objects with oriented bounding boxes; for a map, camera poses), PNG = depth image or segmented image, PLY = point cloud. Inference server needed by reconstruct, mapper-update and segment, and by mapper-locate for maps over 150 keyframes; the other routes work without it. Use it when the user asks for any of these on images or video, or about the maps."
---

# oh-my-slam

oh-my-slam turns RGB images and video into 3D on a Mac: scene descriptions of labelled objects
with oriented bounding boxes, point clouds, and persistent maps in which images can be located.
This skill uses it through one entry point, the HTTP API of `server.sh`, its web service. You
need only `sh` and `curl`, on the Mac that runs the service or on any machine on the LAN (the
service binds `0.0.0.0`), and no checkout of oh-my-slam.

**`$BASE/api/openapi.json` describes every operation, parameter, result and error.** Read it before
a request; where it and this file disagree (an older or newer service), **the running service's
document wins**. It lists each operation's parameters with their values, defaults and checks, and
gives its whole definition under `x-oms`: outputs, checks, errors, stages and inference need. This
file adds only what that document cannot say: the operations and routes, how to find the service,
the order of calls, an example per operation, the errors, and the [Rules](#rules), which you
follow.

## Rules

* **The inference server may be down.** Then the operations `reconstruct`, `mapper-update` and `segment` fail with HTTP
  503 `server_unavailable`, with a message that names the command that starts it, `./start_inference_server.sh`,
  as `GET /api/health` does (`inference.start_command`). Report the message and that command to
  the user instead of retrying, and offer what works without it: the workspace's maps (`GET /api/maps`, `GET /api/maps/{name}`); `mapper-locate` except for maps over 150 keyframes.
* **Never start or stop `start_inference_server.sh` or `server.sh`**, and never kill their processes. When one is not running,
  tell the user how to start it: `./start_inference_server.sh` for the inference server, and `./server.sh` in
  the oh-my-slam checkout on the Mac for the service (with `--port <n>` for a URL that stays the
  same, `--data <folder>` for another workspace).
* **Never change a map except through `mapper-update`**, the mapping operation: no other request
  writes one, and nothing else may touch a map's folder.
* **Ask the user first** before updating an existing map (`GET /api/maps/<name>` answers
  404 `not_found` for a new one), and before starting a long mapping request:
  `mapper-update` runs 15 stages, from `setup` to `commit`, and can take many minutes. Say that an update changes the map
  for good.
* **Inputs are uploads or paths inside the workspace** (relative to it, such as
  `uploads/<upload>/photo.jpg`; a map by its name), never paths outside it, which are refused
  (400 `usage`). **An upload is consumed by the one request it is given to**, whatever its
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

## Operations and routes

Each operation runs one of oh-my-slam's commands within one HTTP request. There are no jobs: the
service answers when the command ends, with its result, byte for byte, or its error. It keeps maps
and uploads in its workspace (`~/oh-my-slam-data/` unless the service was started with
`--data <folder>`) and keeps no results.

| Operation | Route | Takes → gives | Map | Inference server |
|---|---|---|---|---|
| `reconstruct` | `POST /api/ops/reconstruct` | `image` (one image) → JSON, PNG (depth image) or PLY, by `format` | read-only | needed |
| `mapper-update` | `POST /api/ops/mapper-update` | `inputs` (images or one video, in order) + `map` (a new or existing map) → JSON or PLY, by `format` | **writes the map** `map` | needed |
| `mapper-locate` | `POST /api/ops/mapper-locate` | `inputs` (images) + `map` (an existing map) → JSON or PLY, by `format` | read-only | only for maps over 150 keyframes |
| `segment` | `POST /api/ops/segment` | `image` (one image) → JSON or PNG (segmented image), by `format` | read-only | needed |

Results: JSON is an ASAM OpenLABEL 1.0.0 scene description: each object has a label (`type`), a score, a colour that is the same in every result, and an oriented bounding box `cuboid` whose `val` is `x,y,z,qx,qy,qz,qw,sx,sy,sz` (metres, quaternion scalar last), e.g. `{"openlabel":{"metadata":{"schema_version":"1.0.0",…},"objects":{"1":{"name":"chair 1","type":"chair","ontology_uid":"0","coordinate_system":"camera","object_data":{"cuboid":[{"name":"obb","val":[0.41,0.18,2.35,0.0,0.0,0.0,1.0,0.48,0.51,0.92],"coordinate_system":"camera"}],"num":[{"name":"score","val":0.91}]},…},…},…}}`; a map's scene also holds its camera poses; PNG is the depth image (`reconstruct` with `"format":"depth"`) or the segmented image (`segment` with `"format":"png"`); PLY is a point cloud in metres whose header records its attributes (`attrs`).

The routes:

* `POST /api/ops/{op}`: run the operation `{op}` within the request and answer when its command
  ends
* `POST /api/ops/{op}/validate`: check a request with the command's own checks, and the inference
  server's when the request needs it; nothing runs and no upload is consumed
* `GET /api/openapi.json`: the description of every operation, parameter, result and error
* `GET /api/health`: service and inference-server health, and the requests running or waiting for their turn
* `PUT /api/uploads` (query `name`): upload one input file: the raw file as the body (curl -T <file>)
* `POST /api/uploads` (query `name`): upload one input file: the raw file as the body, with its own media type
* `DELETE /api/uploads/{id}`: discard an upload no request was given
* `GET /api/maps`: the workspace's maps with their summaries
* `GET /api/maps/{name}`: a map's summary and metadata (map.json)

Only the operations need the inference server, and a map changes only through `mapper-update`. The
browser application at `$BASE/` runs the same operations, for the user.

## Find the service

`server.sh` binds a free port unless it was started with `--port <n>`, so its URL changes from
run to run. This snippet prints the base URL and caches it in `${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/server_url`. It tries, in order: the
URL the user gives (`OMS_URL`); the cached URL, if `/api/health` answers there within 3 s;
on the Mac that runs the service, the URL the service records in `server.json` in its workspace
(`~/oh-my-slam-data/`, or the `--data` folder the user names, as `OMS_DATA`), with `0.0.0.0` replaced
by `127.0.0.1`. Otherwise it fails: ask the user for the URL that `server.sh` printed when it
started (`server.sh: listening on http://0.0.0.0:<port>/`; from another machine, the Mac's address
or host name in place of `0.0.0.0`) or that `server.sh --status` reports (`service.url`), and run it again
with the first line changed to `OMS_URL=<url> sh <<'EOF'` (or `OMS_DATA=<folder> sh <<'EOF'`).
There is no network or port scan; `server.sh --port <n>` keeps the URL stable for other
machines.

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

## A request, call by call

The example is `reconstruct`; every operation works the same way.

1. **Check the service** and the inference server: `curl -sS "$BASE/api/health"`.
   `inference.status` is one of `down`, `loading`, `ready`, `error`, `stopping`; an operation that needs the inference server is
   accepted while it is `ready` or `loading`, and otherwise `inference.start_command` names the
   command that starts it (see [Rules](#rules)). `service.requests` counts the requests `running`
   and those `waiting` for their turn.
2. **Upload each input file**, or name a path inside the workspace instead. The body is the raw
   file, sent with `curl -T <file>` (a form, `curl -F`, is refused with 415); `name` is the
   file name the command sees: keep its suffix and use only letters, digits, `.`, `_` and `-`. A
   parameter that takes several files (a JSON list) takes one upload per file, in the order the
   command reads them.
   ```sh
   curl -sS -T photo.jpg "$BASE/api/uploads?name=photo.jpg"
   ```
   → 201 `{"id":"<upload>","name":"photo.jpg","size":2481152,"path":"uploads/<upload>/photo.jpg"}`: its `path` is the parameter's value.
3. **Validate**: the command's own checks, and the inference server's when the request needs it;
   nothing runs and no upload is consumed. The body is a JSON object of parameters, sent as
   `application/json` (`-H 'Content-Type: application/json' -d '…'`, or `--json '…'` where the machine's curl has it; a
   plain `-d` is refused with 415). With `"valid":false`, `problems` and `by_parameter` say
   what to change.
   ```sh
   curl -sS -X POST "$BASE/api/ops/reconstruct/validate" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
   ```
   → `{"valid":true,"command":[…],"inference":true,"problems":[],"by_parameter":{}}`
4. **Run it and wait**, with no client timeout: nothing comes back until the command ends (a
   mapping request can take many minutes, and a request that uses the inference server may first
   wait for its turn). Before `mapper-update`, ask the user first (see [Rules](#rules)). Save the body
   with `-o`, the headers with `-D`; `-w` prints the status.
   ```sh
   curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/reconstruct" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
   ```
   → 200, `Content-Type: application/json`, `Server-Timing: connect;dur=500.0, inference;dur=500.0, segment;dur=500.0, export;dur=500.0, write;dur=500.0, total;dur=2500.0`; `result.json` holds the result, byte for byte. Any other status means the file holds an error (see [Errors](#errors)).
5. **Read the stage timings** in the `Server-Timing` header: each of the command's own stages
   under its own name, then `total`, in milliseconds: `grep -i '^server-timing:' headers.txt` →
   `server-timing: connect;dur=500.0, inference;dur=500.0, segment;dur=500.0, export;dur=500.0, write;dur=500.0, total;dur=2500.0`.
6. **Report** the result file, or the error's `message` as it is. A map's summary and update
   history are at `GET /api/maps/<name>`.

## Examples

One per operation, each after `BASE=<that URL>;`. `<upload>` is the `id` that the upload above
it answered (its `path` is the value to give), and `<map>` the name of a map of the workspace.

```sh
# reconstruct: read-only; inference server needed
curl -sS -T photo.jpg "$BASE/api/uploads?name=photo.jpg"
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/reconstruct" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
# mapper-update: writes the map `map`; inference server needed; ask the user first
curl -sS -T video.mp4 "$BASE/api/uploads?name=video.mp4"
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/mapper-update" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/video.mp4"],"map":"<map>"}'
# mapper-locate: read-only; inference server only for maps over 150 keyframes
curl -sS -T query.jpg "$BASE/api/uploads?name=query.jpg"
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/mapper-locate" -H 'Content-Type: application/json' -d '{"inputs":["uploads/<upload>/query.jpg"],"map":"<map>"}'
# segment: read-only; inference server needed
curl -sS -T photo.jpg "$BASE/api/uploads?name=photo.jpg"
curl -sS -X POST -o result.json -D headers.txt -w '%{http_code}\n' "$BASE/api/ops/segment" -H 'Content-Type: application/json' -d '{"image":"uploads/<upload>/photo.jpg"}'
```

## Errors

An operation that fails answers the command's own message and the machine-readable code of its
exit status, with the HTTP status that one generic rule gives that code (the table):

```json
{"error": {"code": "<code>", "message": "<the command's own message>", "http_status": <status>}}
```

A refused request (and validate's `problems`) adds `rule`, `parameters` (the parameters it
concerns), `exit_code`, `problems` (every problem) and `by_parameter`. A message is the command's
own, so it may name a parameter by the command's flag: the document gives each parameter's flag as
its `x-oms.flag`. For example, `POST /api/ops/reconstruct` with the body `{}` answers 400 `{"error":{"rule":"arguments","parameters":["image"],"message":"…","code":"usage","exit_code":2,"http_status":400,"problems":[…],"by_parameter":{"image":["…"]}}}`; a
request whose command ran and failed answers its `exit_code` and message, e.g. 422 `{"error":{"code":"not_registered","exit_code":5,"message":"<the command's message>","http_status":422}}`.

| `exit_code` | `code` | HTTP | Meaning |
|---|---|---|---|
| 0 | `ok` | 200 | success |
| 1 | `internal` | 500 | internal error (also: inference failed, or COLMAP is missing or the wrong version) |
| 2 | `usage` | 400 | usage or input error: a bad option or value, a missing or unsupported input file |
| 3 | `server_unavailable` | 503 | the inference server does not answer (not running, or its models failed to load) |
| 4 | `not_a_map` | 422 | the map folder is not a map (and not empty, for an update) |
| 5 | `not_registered` | 422 | nothing could be placed in the map (no overlap); the map is unchanged |
| 6 | `map_locked` | 409 | another update holds the map |
| 130 | `interrupted` | 499 | interrupted: the service stopped while the request waited or ran; send it again once the service runs |

The service's own refusals have the same shape:

| `code` | HTTP | When |
|---|---|---|
| `not_found` | 404 | no such route, map, upload or operation |
| `method_not_allowed` | 405 | a method the route does not take |
| `forbidden` | 403 | a `Host` that does not name the service's machine, or a foreign `Origin` |
| `unsupported_media_type` | 415 | a `POST` whose body is not `application/json` (an operation) or that has no type of its own (an upload), or an upload sent as a form (`curl -F`) |
| `too_large` | 413 | an upload over 8 GiB |
| `insufficient_storage` | 413 | an upload that would leave less than 1 GiB free on the workspace's disk |
| `upload_in_use` | 409 | an upload that another request in progress was given |
| `stopping` | 503 | a request that arrives while the service stops |
