# 2.6 Web service — `server.sh`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

```sh
server.sh [--port <n>] [--data <folder>] [--no-browser]
server.sh --status | --stop
```

Starts a long-lived local HTTP service that exposes every feature of `reconstruct.sh`
([§2.2](reconstruct.md)), `mapper.sh` ([§2.3](mapper.md)), `segment.sh` ([§2.4](segment.md))
and `view.sh` ([§2.5](view.md)) through an HTTP API, and serves a browser application built on
that API. A user who never opens a terminal after `server.sh` must be able to do everything
those commands can do.

* `--port <n>` — port to bind on `0.0.0.0`, **default `0`** (a free port chosen by the OS).
* `--data <folder>` — the workspace that holds maps, uploads and results, **default
  `~/oh-my-slam-data/`** (see [Workspace](#workspace)).
* `--no-browser` — do not open the default browser once the service is listening.
* `--status` — print the service's health JSON on stdout; fail if it is not running.
* `--stop` — stop a running service, cancelling in-flight jobs as described in [Jobs](#jobs).

Once the service accepts connections, stderr carries exactly one line
`server.sh: listening on http://0.0.0.0:<port>/`. Apart from `--status`, `server.sh` writes
nothing to stdout. Ctrl-C and SIGTERM are its normal stop. One service runs per workspace: a
second `server.sh` on the same `--data` reports the running one's URL and exits.

`server.sh` is a client of the inference server ([§2.1](start_inference_server.md)), as the
commands are, and never loads a model itself. It runs while the inference server is down;
then only the operations that the commands would refuse for that reason fail.

## Single source of truth

This specification deliberately does not restate the commands. Their options, defaults,
validation rules, output formats, artefacts, error conditions and viewer controls are defined
only in [§2.2](reconstruct.md)–[§2.5](view.md), and the service and its web application must
**derive** them from there — from the same shared definitions the commands use — rather than
keep their own list. When a command gains, changes or loses an option, an output, an artefact
or an error, the API and the web application reflect it with no change to this specification
and no hand-written change to the web service.

Concretely:

* **Operations.** Each command's modes and options form one API operation with one parameter
  per option, with the same names, values, defaults and validation. The set of operations and
  parameters is generated from the commands' own option definitions.
* **Results.** For the same inputs and options, a job's result is **byte-identical** to what
  the command writes to stdout or to `-o`, and any files the command writes to a folder are
  byte-identical too. Every contract the commands guarantee — the scene format
  ([§3](high_level_spec.md#3-scene-description-json-returned-by-the-tools)), the colour
  contract, read-only maps, the map-update semantics — therefore holds for the service
  without being restated here.
* **Errors.** Every error a command can report reaches the client with the command's message
  and a machine-readable code derived from its exit status, mapped to an HTTP status by one
  generic rule (input errors → 4xx, inference server unavailable → 503, internal → 500). No
  per-error list is kept here.
* **Timings and logs.** Each job records what the command records — per-stage timings, the
  lines it would print to stderr — in the same form.
* **Viewer.** The web application embeds the [§2.5](view.md) viewer itself, so its layers,
  controls and camera features are those of `view.sh` at all times.

## Ownership

`server.sh` owns only the HTTP layer, the job runner and the web application's shell. It
calls the same Python entry points of the shared package that the commands call — never the
shell scripts and never a re-implementation — so it duplicates none of the logic owned by
reconstruction, mapping, segmentation or the viewer, and it never imports the inference stack
into its own process. The package's ownership contracts are extended to the web-service
module.

## Workspace

* Maps live in `<data>/maps/<name>/` in exactly the format `mapper.sh` writes, so a map is
  interchangeable between the service and the commands.
* Inputs reach the service as uploads, stored under `<data>/uploads/`, or as paths inside the
  workspace; paths that resolve outside it are refused.
* Uploads are transient. An upload is consumed by the one job it is submitted to and is
  deleted as soon as that job ends, whatever its state (`succeeded`, `failed` or
  `cancelled`); anything the command keeps of its input (for example in a map) is kept by the
  command, not by the service. An interrupted upload is deleted at once, and an upload that no
  job has consumed is deleted when the service stops or starts, so `<data>/uploads/` holds only
  the inputs of queued and running jobs.
* Each job's result and files are kept under `<data>/jobs/<id>/`, so they can be downloaded
  again after a page reload or a service restart. The job list survives a restart.
* The service changes a map only through the mapping operation and never deletes one.
  Operations the commands perform without the inference server also work without it here.

## Jobs

Every operation that needs the inference server, or can take more than a moment, runs as a
job.

* Submitting validates the request synchronously, with the command's own validation, and
  returns the job id or the validation error; nothing is queued for an invalid request.
* A job is `queued`, `running`, `succeeded`, `failed` or `cancelled`. While running it reports
  the stage it is in — the stage names are the command's own timing stages — and progress
  where the command knows its size.
* Progress is pushed to clients as server-sent events and is also available by polling.
* A job can be cancelled; its effect is that of interrupting the command (a cancelled map
  update leaves the map as it was).
* Jobs that use the inference server run one at a time, in submission order. Read-only
  requests are served while a job runs. Two jobs never write the same map at once.

## API

All endpoints are under `/api/` and are described by an OpenAPI document at
`/api/openapi.json`, generated from the operation definitions above. Besides one operation per
command mode, the API offers: service and inference-server health; uploads; the list of maps
with a summary taken from each map's own metadata; jobs (list, inspect, progress stream,
cancel, download of the result and of each produced file); and the viewer's data, served by
the viewer's own code.

## Web application

The web application is served at `/` by the same process and uses only the public API, so
everything it does can also be scripted. Its forms, result pages and downloads are **rendered
from the API description**, not written per command: a new option appears as a new field, a
new output file as a new download, a new error as a new message.

### Structure

* **Top bar** — workspace name, inference-server status (with the command to start it when it
  is down), and the number of queued and running jobs.
* **Image** — one image in, any command mode that takes a single image. A drop zone with a
  preview, the generated option form, then the job's live progress and, on success, the
  result: rendered images and tables the command produces, the embedded viewer, and a
  download for every result and file.
* **Maps** — the workspace's maps as cards (name, a keyframe thumbnail, and summary figures
  from the map's metadata), with a filter. Creating or updating a map is a guided flow over
  the mapping operation's options, with the input order visible and editable. A map's page
  shows the embedded viewer, the map's objects, its update history with timings, and every
  export the commands offer for a map.
* **Jobs** — every job with its kind, inputs, state, times and per-stage timings; running jobs
  show progress, failed ones their message; jobs can be cancelled or re-submitted with the
  same options (an uploaded input, already discarded, is asked for again).
* **Viewer** — the [§2.5](view.md) viewer, embedded in result and map pages and also available
  full-screen on its own URL.
* **3D scene viewer** — opens, alone or together in one view, any point cloud (PLY) and any
  scene description (JSON, [§3](high_level_spec.md#3-scene-description-json-returned-by-the-tools))
  produced by the commands: a job's result or file, or a file the user opens from disk.
  * The point cloud is drawn with the attributes its PLY carries (colour, normals, label),
    with the viewer's point-cloud controls for what the file allows.
  * The scene JSON is drawn in 3D: each object's OBB from its `cuboid`, in the object's
    colour and with its label and `id`; every camera pose it contains, with its position
    coordinates and the option to move the viewpoint to it; and camera poses written in a
    PLY header (such as `mapper.sh locate`'s) are shown the same way. Located cameras are
    distinguishable from a map's own frames.
  * A PLY and a JSON opened together share map coordinates, so the OBBs and cameras sit on
    the cloud; each has its own layer toggle, and a list of the JSON's objects is linked to
    their boxes.
  * The rendering reuses the [§2.5](view.md) viewer's own drawing of clouds, OBBs, labels and
    cameras, so both look and behave the same. A file opened from disk is read in the
    browser and never uploaded; a file that is not a valid PLY or a document that does not
    validate against the scene schema is refused with a message saying why.

Selecting an object anywhere (a table row, an image region, a box in the viewer) highlights it
everywhere else on the page. Every page has a stable URL, so a map, a job or a view can be
bookmarked, reloaded or shared on this machine and returns in the same state.

### UX requirements

* **Responsive feedback.** Every action responds visibly within 100 ms. Nothing blocks the
  interface: long work runs as jobs, and the user can leave a page and come back to it.
* **Forms.** Each field shows its default and help text taken from the command's option
  definition. Fields that do not apply to the current choices are hidden. Invalid values are
  flagged next to the field, with the command's message, before submission.
* **Actionable errors.** Messages say what happened and what to do, in the commands' words.
  When the inference server is down, the actions that need it are disabled with an explanation
  and the rest stays available.
* **Confirmation.** Cancelling a job, and starting a long operation, state the consequence
  first.
* **Accessibility.** Keyboard-operable throughout, visible focus, labelled controls, WCAG 2.1
  AA contrast. Object colours always appear with the object's label or `id`, so colour is
  never the only cue.
* **Layout and theme.** Usable from desktop down to tablet width; the viewer adapts to the
  space it is given. Light and dark themes follow the system preference; object colours are
  those of the colour contract in both.
* **Self-contained.** All scripts, styles and fonts are served by the service; nothing is
  loaded from a CDN.

## Evaluation

The benchmark evaluators of [§5](high_level_spec.md#5-benchmark-evaluators) cover `server.sh`
on the same reference inputs as the commands:

* **Performance** — start-up time and resident memory, time until the web application has
  rendered, latency of read-only requests, and the overhead of a job over the same command
  run from the shell.
* **Parity** — for every command mode and reference input, results and files are
  byte-identical to the command's. The check enumerates the operations from the commands'
  definitions, so a new option or mode is covered without changing the evaluator.
* **UI** — browser tests run the main flows (image job, map creation and update, viewer, 3D scene viewer
  with a PLY and a JSON opened together, downloads, cancellation, inference server down) and check accessibility automatically.
