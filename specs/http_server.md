# 2.6 Web service — `server.sh`

Part of the [high-level specification](../high_level_spec.md) (§2 Components).

```sh
server.sh [--port <n>] [--data <folder>] [--no-browser]
server.sh --status | --stop
```

Starts a long-lived local HTTP service that exposes `reconstruct.sh` ([§2.2](reconstruct.md)),
`mapper.sh` ([§2.3](mapper.md)) and `segment.sh` ([§2.4](segment.md)) through an HTTP API,
and serves a simple browser application built on that API. These three commands are the
API's whole scope: `view.sh` ([§2.5](view.md)) and `start_inference_server.sh`
([§2.1](start_inference_server.md)) remain commands only. A user who never opens a terminal
after `server.sh` must be able to run every mode of the three commands and get its result.

* `--port <n>` — port to bind on `0.0.0.0`, **default `0`** (a free port chosen by the OS).
* `--data <folder>` — the workspace that holds maps and uploads, **default
  `~/oh-my-slam-data/`** (see [Workspace](#workspace)).
* `--no-browser` — do not open the default browser once the service is listening.
* `--status` — print the service's health JSON on stdout; fail if it is not running.
* `--stop` — stop a running service, interrupting the requests in progress as described in
  [Requests](#requests).

Once the service accepts connections, stderr carries exactly one line
`server.sh: listening on http://0.0.0.0:<port>/`. Apart from `--status`, `server.sh` writes
nothing to stdout. Ctrl-C and SIGTERM are its normal stop. One service runs per workspace: a
second `server.sh` on the same `--data` reports the running one's URL and exits.

`server.sh` is a client of the inference server ([§2.1](start_inference_server.md)), as the
commands are, and never loads a model itself. It runs while the inference server is down;
then only the operations that the commands would refuse for that reason fail.

## Single source of truth

This specification deliberately does not restate the commands. Their options, defaults,
validation rules, output formats and error conditions are defined only in
[§2.2](reconstruct.md)–[§2.4](segment.md), and the service and its web application must
**derive** them from there — from the same shared definitions the commands use — rather than
keep their own list. When a command gains, changes or loses an option, an output or an error,
the API and the web application reflect it with no change to this specification and no
hand-written change to the web service.

Concretely:

* **Operations.** Each mode of the three commands forms one API operation with one parameter
  per option, with the same names, values, defaults and validation. The options that only
  choose where the command writes (`-o`, `-d`) are not parameters: the result is the
  response itself, and the files a command writes to a folder are not offered. The set of
  operations and parameters is generated from the commands' own option definitions.
* **Results.** For the same inputs and options, a successful response's body is
  **byte-identical** to what the command writes to stdout or to `-o`. Every contract the
  commands guarantee — the scene format
  ([§3](../high_level_spec.md#3-scene-description-json-returned-by-the-tools)), the colour
  contract, read-only maps, the map-update semantics — therefore holds for the service
  without being restated here.
* **Errors.** Every error a command can report reaches the client with the command's message
  and a machine-readable code derived from its exit status, mapped to an HTTP status by one
  generic rule (input errors → 4xx, inference server unavailable → 503, internal → 500). No
  per-error list is kept here.
* **Timings.** A response carries the command's per-stage timings, under the command's own
  stage names, in a standard `Server-Timing` header.

## Ownership

`server.sh` owns only the HTTP layer, the order in which requests run, and the web
application's shell. It calls the same Python entry points of the shared package that the
commands call — never the shell scripts and never a re-implementation — so it duplicates none
of the logic owned by reconstruction, mapping or segmentation, and it never imports the
inference stack into its own process. The package's ownership contracts are extended to the
web-service module.

## Workspace

* Maps live in `<data>/maps/<name>/` in exactly the format `mapper.sh` writes, so a map is
  interchangeable between the service and the commands. A request names a map by its name.
* Inputs reach the service as uploads, stored under `<data>/uploads/`, or as paths inside the
  workspace; paths that resolve outside it are refused.
* Uploads are transient. An upload is consumed by the one request it is given to and is
  deleted as soon as that request ends, whatever its outcome; anything the command keeps of
  its input (for example in a map) is kept by the command, not by the service. An
  interrupted upload is deleted at once, and an upload that no request has consumed is
  deleted when the service stops or starts.
* The service keeps no results: a result exists only in its response. It changes a map only
  through the mapping operation and never deletes one. Operations the commands perform
  without the inference server also work without it here.

## Requests

There are no jobs: an operation runs within its own HTTP request.

* A request is first validated with the command's own validation; an invalid request is
  refused at once and runs nothing. Each operation also has a validation endpoint that runs
  the same checks and nothing else.
* A valid request runs the operation and answers when it ends: with the result on success,
  with the command's error otherwise. The service never times out a running request. A
  mapping request can take many minutes, and the client keeps its connection open until the
  answer.
* Requests that use the inference server run one at a time, in arrival order; a request
  waits for its turn with its connection open. All other requests, including the operations
  the commands perform without the inference server, are answered at once, even while one
  runs. Two requests never write the same map at once.
* A client that disconnects interrupts its request, whether it is waiting or running, with
  the effect of interrupting the command: a map update leaves the map as it was. Stopping
  the service interrupts every request in progress the same way.

## Operations

The API offers one operation per mode of the three commands, listed here by command. Their
parameters, results and errors are the command's own (see
[Single source of truth](#single-source-of-truth)); a mode that a command gains becomes an
operation in the same way.

### `reconstruct.sh` ([§2.2](reconstruct.md))

* **Reconstruct an image.** One image in; the response is the scene description (JSON), the depth image (PNG) or
  the point cloud (PLY), as the command's format option selects. Needs the inference server.

### `mapper.sh` ([§2.3](mapper.md))

* **Update a map.** Creates or extends the named map in the workspace from images or a
  video, in the order the request gives them; the response is the command's result. Needs
  the inference server, and is usually the longest request. It is the only operation that
  writes a map.
* **Locate in a map.** Finds the camera poses of images in an existing map, without changing
  it; the response is the command's result. Needs the inference server only when the command
  does.

### `segment.sh` ([§2.4](segment.md))

* **Segment an image.** One image in; the response is the scene description (JSON) or the
  segmented image (PNG). Needs the inference server.

The artefacts that `segment.sh -d` writes to a folder are not offered by the API.

## API

All endpoints are under `/api/` and are described by an OpenAPI document at
`/api/openapi.json`, generated from the operation definitions above. Besides the operations,
each with its validation endpoint, the API offers only: service and inference-server health;
uploads (create and discard); and the workspace's maps, as a list and one by one, with a
summary taken from each map's own metadata, so a client can name a map. There are no job,
download or viewer endpoints.

Because the service listens on the LAN, it refuses (with its own error, listed in the OpenAPI
document) a request whose `Host` does not name this machine, a state-changing request from a
foreign `Origin` or with a content type a browser could send cross-site without asking, and an
upload too large for its cap or for the workspace's free disk space. The caps are recorded in
`README.md`.

## Web application

The web application is served at `/` by the same process and uses only the public API, so
everything it does can also be scripted. Its forms and result pages are **rendered from the
API description**, not written per command: a new option appears as a new field, a new error
as a new message. It has no map or scene viewer. A point-cloud result is still drawn in the
page (see Structure).

### Structure

* **Top bar** — workspace name, inference-server status (with the command to start it when it
  is down), and whether a request is running or waiting for its turn.
* **Image** — one image in, any operation that takes a single image. A drop zone with a
  preview, the generated option form, then the running request and, on success, the result.
  A scene description (JSON) is shown in full, with its objects listed (label, `id`, colour,
  score). A point cloud (PLY) is loaded in the page and drawn in 3D with the colours and
  attributes it carries, and the user can rotate, pan and zoom it. The drawing reuses the
  [§2.5](view.md) viewer's own rendering of clouds, so a cloud looks the same as in `view.sh`.
  Every result can be downloaded byte for byte.
* **Maps** — the workspace's maps as cards (name and summary figures from the map's
  metadata), with a filter. Creating or updating a map is a guided flow over the mapping
  operation's options, with the input order visible and editable. A map's page shows its
  summary and its update history with timings, from the map's metadata, and runs the
  operations that take a map (locating images in it), with the same result
  display and download.

Every page has a stable URL, so a map's page can be bookmarked, reloaded or shared on this
machine.

### UX requirements

* **Responsive feedback.** Every action responds visibly within 100 ms. While a request runs,
  the page shows whether it is running or waiting for its turn, and for how long; the rest of
  the interface stays usable.
* **Interruption.** A running request belongs to its page: cancelling it, or leaving or
  reloading the page, interrupts it (see [Requests](#requests)), so the page asks first and
  states the consequence.
* **Forms.** Each field shows its default and help text taken from the command's option
  definition. Fields that do not apply to the current choices are hidden. Invalid values are
  flagged next to the field, with the command's message, before submission.
* **Actionable errors.** Messages say what happened and what to do, in the commands' words.
  When the inference server is down, the actions that need it are disabled with an explanation
  and the rest stays available.
* **Confirmation.** Starting a long operation, such as a map update, states the consequence
  first.
* **Accessibility.** Keyboard-operable throughout, visible focus, labelled controls, WCAG 2.1
  AA contrast. Object colours always appear with the object's label or `id`, so colour is
  never the only cue.
* **Layout and theme.** Usable from desktop down to tablet width. Light and dark themes follow
  the system preference; object colours are those of the colour contract in both.
* **Self-contained.** All scripts, styles and fonts are served by the service; nothing is
  loaded from a CDN.

## Evaluation

The benchmark evaluators of [§5](../high_level_spec.md#5-benchmark-evaluators) cover `server.sh`
on the same reference inputs as the commands:

* **Performance** — start-up time and resident memory, time until the web application has
  rendered, latency of read-only requests, and the overhead of an operation's request over
  the same command run from the shell.
* **Parity** — for every mode of the three commands and every reference input, the response
  body is byte-identical to the command's result. The check enumerates the operations from
  the commands' definitions, so a new option or mode is covered without changing the
  evaluator.
* **UI** — browser tests run the main flows (single-image request, a point-cloud result drawn
  in the page, map creation and update, locating images in a map, download,
  interruption, inference server down) and check accessibility automatically.
