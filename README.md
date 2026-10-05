# oh-my-slam

Monocular RGB mapping on an Apple-silicon Mac. It reconstructs a single image as a metric point
cloud, builds and updates persistent maps from photos or video, and describes the scene as
labelled objects with oriented bounding boxes (OBBs) in ASAM OpenLABEL 1.0.0. Input is RGB only:
depth, intrinsics and gravity come from models. `specs/` holds the requirements;
this file records the implementation decisions (defaults, conventions, exit codes, the
OpenLABEL mapping and the colour palette).

| Entry point | What it does |
|---|---|
| `start_inference_server.sh [--status\|--stop]` | Starts the resident model server, or stops or queries it. |
| `reconstruct.sh -i IMAGE [-f json\|ply] [-o FILE] [-p ATTRS]` | One image → OpenLABEL scene (default) or point cloud, in the camera frame. |
| `mapper.sh update -i IMAGES\|FOLDERS\|VIDEO -m MAP [-f json\|ply] [-o FILE] [-p ATTRS] [-t full\|single] [-fps N]` | Creates or extends a persistent map. |
| `mapper.sh locate -i IMAGES -m MAP [-f json\|ply] [-o FILE] [-p ATTRS] [-t full\|single]` | Camera pose of each image in an existing map, which stays untouched. |
| `segment.sh -i IMAGE [-f json\|ply] [-o FILE] [-d DIR] [-p ATTRS] [--min-score S]` | Objects of one image: OBBs, colours, and with `-d` five artefact files. |
| `segment.sh -m MAP [-f json\|ply] [-o FILE] [-d DIR] [-p ATTRS]` | The persistent objects of a map, read-only and without the server. |
| `view.sh -i IMAGE \| -m MAP [--no-browser]` | Local browser viewer. `-m` needs no server. |
| `server.sh [--port N] [--data DIR] [--no-browser]` / `--status` / `--stop` | Local web service: every command mode as an HTTP API job, maps and results in a workspace. |

## Install

```sh
brew install colmap            # COLMAP 4.2.x CLI (feature extraction and matching)
uv sync                        # Python 3.12 environment in .venv, dev tools included
./start_inference_server.sh    # the first start downloads the model weights
```

The `pycolmap` wheel in `.venv` is 4.2.x, and the `colmap` CLI must be 4.2.x too: `mapper.sh`
checks both before it maps and exits 1 with a hint otherwise. Each entry script is a thin wrapper
that runs `.venv/bin/python -m oh_my_slam.cli.<command>` and exits 2 if `.venv` is missing. The
scripts never call each other.

Model weights are downloaded on the first server start:

* MoGe-2 ViT-L normal (`Ruicheng/moge-2-vitl-normal`) and MapAnything
  (`facebook/map-anything-apache`) go to the Hugging Face cache.
* GeoCalib (pinhole weights from its GitHub release) goes to the torch hub cache.
* YOLOE-26x-seg and its MobileCLIP2-B text encoder go to `~/Library/Caches/oh-my-slam/weights`.

`THIRD_PARTY_LICENSES.md` lists every component and its licence.

## Quick start

```sh
./start_inference_server.sh --status                 # health JSON on stdout; exit 3 if not running
./reconstruct.sh -i photo.jpg > scene.json
./reconstruct.sh -i photo.jpg -f ply -p color=segment,voxel=0.01,normals=on -o objects.ply
./segment.sh -i photo.jpg -d out/ --min-score 0.6
./mapper.sh update -i walk.mp4 -m maps/home > map.json
./mapper.sh update -i more_photos/ -m maps/home -t single -f ply -o new_part.ply
./mapper.sh locate -i where_am_i.jpg -m maps/home > pose.json
./segment.sh -m maps/home -d out_map/
./view.sh -m maps/home
./start_inference_server.sh --stop
```

## Commands

### `start_inference_server.sh`

With no option, the command starts the server in the background and waits until the models are
loaded, for at most 20 minutes (the first start downloads the weights). If a server is already
running, the command reports it and exits 0.

| Option | Effect |
|---|---|
| `--status` | Prints `/health` as JSON on stdout: status (`loading`, `ready`, `error` when a model failed to load, or `stopping`), device, precision, and each model's load state. Exits 3 if the server is not running. |
| `--stop` | Stops the server and removes its socket and state file. |

The server log is `~/Library/Caches/oh-my-slam/server.log`. [Inference server](#inference-server)
describes what runs inside it.

### `reconstruct.sh`

`reconstruct.sh -i IMAGE [-f json|ply] [-o FILE] [-p ATTRS]`

* `-f json` (the default) returns the scene of the image in its camera frame. It has the same
  objects, ids and colours as `segment.sh -i` with default options.
* `-f ply` returns the point cloud, shaped by `-p`. It runs only the inference that the
  attributes need: segmentation for `color=segment` or `label=on`, and gravity for
  `color=height`.
* `-p` requires `-f ply`.

### `mapper.sh update`

`mapper.sh update -i INPUTS… -m MAP [-f json|ply] [-o FILE] [-p ATTRS] [-t full|single] [-fps N]`

Only `-i` and `-m` are required. With no other option, the command writes the whole map as JSON
to stdout.

* **`-i`** takes image files or exactly one video (spec §2.3); a folder is refused (pass its
  files, e.g. `dir/*.jpg`, which the shell sorts by name):
  * Images keep the order given.
  * Image types: jpg, jpeg, png, bmp, tif, tiff, webp.
  * Video types: mp4, mov, m4v, avi, mkv, webm.
* **`-m`** is the map folder:
  * A folder that is missing or empty becomes a new map. A folder holding nothing but a
    `.DS_Store` or this tool's own `.lock` / `.staging` leftovers is empty. Any other entry makes
    it non-empty, hidden ones too: a folder with only a `.git` is refused.
  * A map is extended.
  * Any other folder is refused and left untouched (exit 4).
* **`-t`** sets the scope of the result (default `full`):
  * `full` returns the whole map: every object (see *Update semantics* for when a detection is an
    object of the map) and every keyframe pose. With `-f ply` it returns the whole map cloud.
  * `single` returns only the keyframes added by this update and the objects they observe. With
    `-f ply` it returns the points those keyframes see.
* **`-fps`** applies to video only (default `2`; it must be positive). The command keeps the
  sharpest frame in each 1/fps time slot. For images, `-fps` is ignored with a warning, whatever
  its value.
* **`-p`** requires `-f ply`. Pixel-level keys are refused, because a map's points are already
  3D.

The result is always in map coordinates. Every option is checked before the server is contacted.

### `mapper.sh locate`

`mapper.sh locate -i IMAGES… -m MAP [-f json|ply] [-o FILE] [-p ATTRS] [-t full|single]`

Only `-i` and `-m` are required. With no other option, the command writes the located camera
poses as JSON to stdout. The map is read-only: nothing in its folder changes (no `.lock`, no
`.staging/`), and the images are not added.

* **`-i`** takes one or more image files (the types of `update`). A video is refused (exit 2), and
  so is a folder.
* **`-m`** must be an existing map. A missing or empty folder is an input error (exit 2) and is not
  created; any other folder that is not a map exits 4.
* **`-t`** sets the scope of the result (default `single`):
  * `single` returns the located camera of each image, with no objects. With `-f ply` it returns
    the map points visible from the located cameras: inside a camera's frustum and not hidden
    behind nearer map points (a coarse z-buffer, 96 cells on the long side; the image has no depth
    of its own).
  * `full` returns the map exactly as `update -t full` returns it, plus the located cameras. With
    `-f ply` it returns the whole map cloud.
* **`-o`** may not point inside the map folder (exit 2).
* **`-p`** requires `-f ply`, with the map-scope keys of `update`.

An image the map cannot localise is named on stderr with the reason (too few matches with map
points, no consistent pose, or a pose that contradicts its matches); the others are still
returned. If no image is located, the command exits 2.

*How an image is located.* The map's COLMAP database is cloned (`cp -c`) into a scratch folder;
the images get their SIFT features there (a camera the map already has when the image size
matches and no EXIF focal says otherwise, as for `update`) and are matched against **every
keyframe** of a map of at most `UPDATE_EXHAUSTIVE_MAX` (150) keyframes, with no inference
server — the bound `update` uses. In a larger map they are matched against the
`RETRIEVAL_TOP_K` (30) keyframes most similar by the retrieval descriptor, which the inference
server computes for the image: that case needs the server and exits 3 when it is down. Each
verified match to a keyframe keypoint with a triangulated point in `sfm/model` gives a 2D–3D
correspondence; a keyframe without a model (a one-keyframe map has no `sfm/`) gives its stored
metric depth at the keypoint instead. COLMAP's LO-RANSAC absolute pose with refinement solves
them; the focal length is estimated too unless the image shares a map camera. A pose needs 15
inliers, and its median epipolar distance to its verified matches, under the stored keyframe
poses, must stay within 0.25° (the check the map's own keyframes pass).

*A concurrent update.* `locate` reads one state of the map. If an update commits while it runs
(`map.json` or the commit marker changes, or a file it read disappears), it opens the map again
and starts over, up to 3 times; after that it exits 2 with "retry once the update is done".

*Representation of a located camera.* Image `k` (its position in `-i`, from 0) gets its own
`sensor_cs` and camera stream `located_<k>` (its intrinsics, `uri` the image path), so it is never
confused with the map's `camera_<id>`. Its frame is keyed past every keyframe index the map has
used (`next_frame_index + k`) and holds `timestamp` (the key, as for a keyframe), `located:
true`, `image`, `inliers`, the stream `uri` and
the transform `located_<k>_to_map`, the camera-to-map pose in the form of a keyframe's
`camera_<id>_to_map`. In a PLY the header carries one comment per input image after the frame and
attribute lines, in the JSON result's representation: `located_<k> {"image": …, "located": true,
"transform_src_to_dst": {"quaternion": [qx, qy, qz, qw], "translation": [x, y, z]},
"stream_properties": {"intrinsics_pinhole": {…}, "intrinsics_source": …}}` — the frame's
`located_<k>_to_map` transform and the stream's properties — or `{"image": …, "located": false}`
for an image that could not be located. The `-t single` document's metadata holds the map's base fields
(`tool` `"mapper"`, `map_frame`) and `scope: "single"`; the `map` coordinate system lists the
`located_<k>` children.

### `segment.sh`

`segment.sh -i IMAGE [-f json|ply] [-o FILE] [-d DIR] [-p ATTRS] [--min-score S]`
`segment.sh -m MAP [-f json|ply] [-o FILE] [-d DIR] [-p ATTRS]`

Exactly one of `-i` and `-m` is required.

* **`-i`** segments the image, lifts each instance into 3D, and fits an upright OBB.
* **`-m`** exports the map's persistent objects with their ids and colours. It runs no inference,
  needs no server, and never writes to the map.
* **`-d DIR`** also writes five artefacts into `DIR` (below). Without `-o` and `-d`, the command
  writes no file.
* **`-p`** shapes both the `-f ply` output and `segments.ply`, so it needs `-f ply` or `-d`.
  `color` is fixed to `segment`, and any other value is refused. With `-m`, the pixel-level keys
  are refused.
* **`--min-score S`** applies to `-i` only (default `0.5`). Any number is accepted, as the spec
  sets no bound; below 0.05, the lowest score the detector is asked for, it keeps what 0.05
  keeps, and above 1 it keeps no object.

`--min-score` only adds or removes objects. The objects kept at two thresholds have the same id,
colour, mask, points and box, for these reasons:

* Below 0.5 the detector is asked for every detection scoring above the floor of 0.05; from
  0.5 up, for those scoring 0.5 or more. The detections between the two floors have no effect
  on an object scoring 0.5 or more: they never claim pixels from it, never suppress it, and rank
  after it in ids.
* Overlapping masks are resolved among all of those detections before the threshold applies.
  Every pixel goes to at most one detection:
  * Detections scoring 0.5 or more claim pixels first, smallest mask first. A nested object
    therefore keeps its pixels, such as a plate on a table or a person on a sofa.
  * Detections scoring below 0.5 get only the pixels that no detection of 0.5 or more covers.
* Only then are the objects below `S` dropped.
* Ids are numbered 1…N in a fixed priority order: score descending, then mask area descending,
  then label, then box. A higher threshold therefore drops objects only from the end.

The same image with the same options gives the same ids and colours on every run.

| `-d` artefact | Contents |
|---|---|
| `segmentation.json` | The same bytes as `-f json`. |
| `segmented.png` | The image dimmed to 35 %, with each object's exclusive mask painted opaque in exactly its colour (no blending). For a map: a contact sheet of up to 6 keyframes, chosen so that every object appears where possible. |
| `catalog.csv` | `id,label,score,color_hex,width_m,height_m,depth_m,volume_m3,center_x,center_y,center_z,pixel_count,point_count`, ordered by id. |
| `catalog.md` | The same rows with a colour swatch, ordered by descending volume. |
| `segments.ply` | The same bytes as `-f ply`: points coloured by object, unsegmented points `#808080`. |

### `view.sh`

`view.sh -i IMAGE | -m MAP [--no-browser]`

The viewer binds `127.0.0.1` on a free port (port 0: a fixed port may be held by another
process). It opens the default browser unless `--no-browser` is given, and serves until Ctrl-C.
Nothing is written to stdout. Once the server accepts connections, stderr carries exactly one line of this form:

```
view.sh: listening on http://127.0.0.1:<port>/
```

The two modes differ:

* **`-i`** reconstructs and segments the image once, through the inference server. It shows the
  cloud, the segmented image, the catalogue, the labelled OBBs and the camera at its estimated
  pose. The display rotates the camera frame so that the estimated up direction is +z.
* **`-m`** opens the map read-only, without the server. It shows the map's complete cloud, every
  keyframe camera and the labelled OBBs.

The first view looks down 60° on the whole scene and its cameras, so that the walls of a room
hide little of its floor, objects and camera cluster.

The page has up to four tabs:

* **Controls**
  * *Layers* has one independent switch each for the point cloud, the segmentation, the camera
    poses, the labels and the oriented boxes. The segmentation layer draws only the points of
    each object, in its colour, over the cloud. It is not the `color=segment` attribute, which
    recolours the whole cloud (unsegmented points grey).
  * *Point cloud* has live controls for the point-cloud attributes that affect the display. For
    an image these are `color`, `stride`, `min-depth`, `max-depth`, `edge`, `voxel` and
    `normals`. For a map they are `color`, `voxel` and `normals`. With `normals=on` the points
    are shaded by their normals, except with `color=segment`, whose object colours and
    unsegmented grey are always drawn exactly (§2.4 colour contract). An invalid combination is reported under the controls and the
    last good cloud stays.
* **Catalogue** lists the image's objects, largest first (`-i` only, as §2.5 asks).
* **Cameras** lists every displayed camera's centre (x, y, z in metres, in the scene frame), with
  a *Go to* button that moves the viewpoint to that camera, looking where it looked.
* **Image** shows the segmented image (`-i` only).

Every box whose top is in view is labelled next to its top face: its id on a tag in the object's
colour, then its name, on a dark plate so that it reads over any cloud. No label ever covers
another or leaves the view. Larger boxes on screen are labelled first; a tag that would cover
another moves to a free place on rings farther out, and a name is added only where it covers
nothing. Where boxes are so crowded that a tag finds no free place, that box shows no tag; its
id and label are listed under the Labels layer ("No room for: …", the first 20 and how many more;
not a live region, so moving the view announces nothing), and its tag appears once the view is
zoomed in.

The viewer contains no geometry, segmentation or colour logic of its own:

* Every displayed cloud is derived on request from data already in memory, by the same code
  that writes PLY files. A control therefore never re-runs inference.
* **Display budget (§2.5).** The page draws every point of a cloud of at most 16,000,000 points
  (`DISPLAY_POINT_BUDGET` in `viewer/bundle.py`). Above that it draws a voxel-grid selection: the
  first point (in derivation order) of each occupied voxel, for the smallest voxel edge whose grid
  has at most 16,000,000 occupied voxels. The search (`core.geometry.budget_voxel_grid`) counts
  the occupied voxels of one edge per step. Its steps are secants of logit(count / points) against
  log(edge), aimed just past the budget until both sides are counted, then regula falsi inside
  the counted bracket. It stops when the bracket is within 2 %: the returned edge was counted with
  at most 16,000,000 voxels, and an edge less than 2 % finer was counted with more. It never
  returns more than the budget. The count is not strictly monotonic at that scale, so an edge
  in between may still fit. Points
  with a non-finite coordinate are never selected. Points at no more distinct places than the
  budget keep one point per place.
  Points are selected, never averaged, so each keeps its own position, colour, normal and object
  id, and the segmentation layer draws the same subset. Under the point-cloud controls the page
  then says "Showing X of Y points: one per voxel of E edge".
  The selection belongs to the shared derivation (`segmentation.cloud.derive_thinned`). It
  computes normals only for the selected points. It keeps the edge found for each set of
  position attributes, and the latest selection's indices, so a colour or normals change reuses
  them. PLY outputs and the map are never thinned.
* **Cost of the budget.** Measured on the 16.15-million-point `street2` map on the M4 Max:
  * The edge search plus the selection takes 2.8 s (6 counts). The selection keeps 15,999,725
    points, one per 0.670 mm voxel; 0.658 mm was counted and does not fit.
  * A later request with the same position attributes costs 0.2 to 0.7 s (colours included). A
    known edge without its indices costs 0.8 s.
  * `view.sh -m` starts this work in a background thread when it opens a map above the budget.
    The page's first `/api/cloud` therefore completes 3.2 s after the map was opened, not 3.2 s
    after the request.
  * In headless Edge (ANGLE Metal) the view orbits at 55 frames per second.
* `encoding` and `label` concern PLY files only and have no control.

The page draws a frame only when something visible changes: the viewpoint, a layer, a control,
the window size. An idle page therefore leaves the GPU to the inference server; with
the 8.9-million-point living-room map open, it draws no frames in 10 s, where it used to draw 600.

The page sets `<body data-rendered="true">` after its first frame with the cloud has rendered.
The evaluator waits for this attribute.

The viewer is built to be reused by another server (the `server.sh` web application embeds it and
draws PLY and scene files with it):

* **Server side.** `viewer.routes.ViewerRoutes(bundle).handle(method, path, query)` answers every
  route (`/`, `/static/…`, `/api/meta`, `/api/scene`, `/api/catalog`, `/api/segmented.png`,
  `/api/cloud`) as a framework-neutral `Response(status, headers, body pieces)`. `view.sh`'s
  stdlib server (`viewer/server.py`) is a thin adapter over it, and another server mounts the same
  object under a prefix of its own. The page uses relative URLs only, so it works at `/` and under
  any prefix ending in `/`.
* **Browser side.** `static/app.js` is `view.sh`'s page. It composes ES modules in `static/lib/`:
  * `data.js`: `DataSource(base)`, the routes under a base URL, and the cloud structure every
    drawing function takes.
  * `viewer.js`: the `Viewer` class. It holds the renderer, the layers, on-demand drawing,
    framing and *Go to*. Its `select(id)` / `onSelect(cb)` API highlights an object, so a host
    page can highlight it in its other views. `view.sh` itself has no selection UI.
  * `cloud.js`: the points and segmentation materials.
  * `obbs.js`: objects and boxes from OpenLABEL cuboids.
  * `cameras.js`: cameras from a scene document (`sceneCameras`, mirroring
    `bundle.scene_cameras`, `located` frames included) or from a `mapper.sh locate` PLY header
    (`plyCameras`), frustums, and the camera table. Located cameras are drawn dashed, in their own
    colour, and labelled "located", so colour is never the only cue.
  * `labels.js`: the non-overlapping label layout.
  * `layers.js` and `controls.js`: the layer and attribute controls, and the display-budget notice.
  * `ply.js`: an in-browser PLY reader (ASCII and binary little-endian; x y z, normals, colour,
    label; the header comments).

### `server.sh`

```sh
./server.sh                          # workspace ~/oh-my-slam-data/, a free port, opens the browser
./server.sh --port 8765 --data ~/ws --no-browser
./server.sh --status                 # health JSON on stdout; exit 3 if no service runs for --data
./server.sh --stop                   # cancels queued jobs, interrupts running ones, then exits
```

A long-lived HTTP service (spec §2.6) in `oh_my_slam.web`, on Starlette under uvicorn.

* **Options.** `--port` binds `0.0.0.0` (default `0`: the OS picks a free port). `--data` is the
  workspace (default `~/oh-my-slam-data/`). `--no-browser` skips opening
  `http://127.0.0.1:<port>/`: the browser gets the loopback address, and the stderr line names the
  bound address. `--status` and `--stop` take only `--data`; any `--port` (even `0`) or
  `--no-browser` with them is a usage error.
* **Output.** Once accepting connections, stderr carries exactly one line
  `server.sh: listening on http://0.0.0.0:<port>/`. Nothing else is printed while it runs: job
  output goes to each job's log. stdout is empty except for `--status`.
* **Stopping.** Ctrl-C, SIGTERM and `--stop` are the same normal stop (exit 0): queued jobs are
  cancelled, and running ones are interrupted and waited for, up to 120 s before they are killed.
  * A second Ctrl-C or SIGTERM during the stop SIGKILLs every job's process group, records those
    jobs as `cancelled` (with no process group left), and exits at once (exit 130).
  * `--stop` sends that second signal itself after 180 s, then SIGKILL after 10 s more.
  * Each running step's `job.json` records its process group, the leader's start time, and the job
    id, which is also in the leader's environment (`OH_MY_SLAM_JOB`).
  * If the service itself was killed, the next start SIGKILLs a group still alive only when its
    leader matches both the start time and the job id, so a recycled process id is never
    signalled. The job is then marked `cancelled`.
  * There is no lifeline in the children: the commands would have to watch for one.
* **One service per workspace.** `<data>/server.lock` (flock) and `<data>/server.json` (pid, URL,
  port). A second `server.sh` on the same `--data` prints `already running … at <url>` on stderr,
  opens the browser on it unless `--no-browser`, and exits 0. `--status` without a running service
  exits 3, like `start_inference_server.sh --status`.
* **Inference.** The service is a client of the inference server and never loads a model, torch or
  Open3D (import-linter contracts, plus a test that checks `sys.modules`). It runs while the
  inference server is down.
  * A request that will use inference is answered with the commands' own exit-3 message as a 503
    when the server is down or its models failed to load. A server that is still loading is
    accepted, because the command waits for it.
  * "Will use inference" is decided by `commands.spec.needs_inference`, which evaluates the mode's
    inference condition, at submission and again when a queued job may start. `mapper locate` on
    a map of at most `UPDATE_EXHAUSTIVE_MAX` (150) keyframes needs none, so it runs without the
    server and outside the inference queue.
  * `segment -m` and `view -m` also run without the server.
* **Request guard** (CSRF and DNS rebinding; the 0.0.0.0 binding is required by the spec):
  * every request's `Host` must name this machine: localhost, its host name, or one of its
    addresses. Anything else gets 403. The addresses are re-read on an unknown name, at most every
    30 s, for example after the machine joins another network.
  * a state-changing request must carry no `Origin`, or this service's own: the request's `Host`,
    or one of the machine's names on the service's port, over `http`. Scheme, host and port all
    count; anything else gets 403.
  * a JSON request's body must be `application/json`. An upload may carry any media type except
    the CORS-safelisted `text/plain`, `application/x-www-form-urlencoded` and
    `multipart/form-data` (415). A cross-site form therefore cannot submit, upload or cancel.

**Workspace.**

| Path | Contents |
|---|---|
| `maps/<name>/` | Maps, exactly as `mapper.sh` writes them. An API map parameter is `<name>` or `maps/<name>`, and maps live nowhere else. The service never deletes a map and changes one only through `mapper-update`. |
| `uploads/<id>/<file>` | Raw-body uploads (`POST /api/uploads?name=<file>`, `application/octet-stream`). A job refers to one by its path `uploads/<id>/<file>`. Each upload belongs to at most one queued or running job and is deleted when that job ends, whatever its state. An interrupted upload is deleted at once, and every upload is deleted at start and stop. An upload may hold at most 8 GiB (long phone videos fit) and must leave 1 GiB free on the workspace's disk; otherwise it gets 413. |
| `jobs/<id>/` | `job.json` (the record, which survives restarts), `progress.jsonl` (`OH_MY_SLAM_PROGRESS`), `timings.json` (`OH_MY_SLAM_TIMINGS`), `stderr.log` (every line the command printed), `stdout`, `out/` (everything the command wrote), `viewer/` (a saved viewer) and `display/` (its PLY files as the 3D scene viewer draws them). |

Any other workspace path is accepted as an input, relative to the workspace or absolute. A path
that resolves outside the workspace (through `..`, `~` or a symlink), or that goes through a
hidden entry (an upload still arriving, a map's `.staging`), is refused as an input error (400) on
that parameter.

**Single source of truth.** Every operation, parameter, default, validation rule, output and
error comes from `oh_my_slam.commands.spec`. Nothing in `oh_my_slam.web` names a command or an
option.

* There is one operation per command mode: `reconstruct`, `mapper-update`, `mapper-locate`,
  `segment-image`, `segment-map`, `view-image` and `view-map` (program, subcommand and mode
  joined). Each has one parameter per option, under the option's `dest` name.
* A request is a JSON object of parameters. Generic rules apply by option kind:
  * path inputs are workspace paths.
  * `output` (`-o`) and `artifacts` (`-d`) are plain names inside the job's `out/`. The result
    always goes through `-o` and defaults to `result.<ext>` for the result's format. A `-d`
    folder named like the result file is refused.
* The command's own parser and rules (`spec.dry_run`) then check the request synchronously.
  Problems come back per parameter with the command's messages, and the HTTP status of the first
  one follows the generic rule of `core.errors.HTTP_STATUS`: input errors 400 (`not_a_map` and
  `not_registered` 422, `map_locked` 409), inference server down 503, internal 500. Nothing is
  queued for an invalid request. `POST /api/ops/<op>/validate` runs the same checks without
  queuing anything.
* `/api/openapi.json` is generated from `spec.describe()`. Each operation and parameter carries
  its whole registry entry under `x-oms`. `/api/operations` returns `spec.describe()` itself.

**Jobs.**

* **Process.** A job is a list of steps. Each step is a Python entry point run as a subprocess in
  its own process group, with:
  * `OH_MY_SLAM_PROGRESS` and `OH_MY_SLAM_TIMINGS` pointing into the job folder;
  * a default SIGINT, so a cancel works even when the service was started with SIGINT ignored.
    Every entry point's `run_main` installs Python's `default_int_handler`, and so do `serve()` and
    the runner when they find SIGINT ignored, so that exec hands the children a default SIGINT;
  * no recording or replay variable inherited from the service's environment.
  * The command step is `python -m oh_my_slam.cli.<command> <argv>`, the same module the shell
    script execs.
  * A viewer step is `python -m oh_my_slam.cli.view_save <dir> <prog> <argv>`, given the
    command's own command line (see Viewer).
* **States.** States are `queued`, `running`, `succeeded`, `failed` and `cancelled`
  (`core.errors.job_state`). A failure carries the command's `<prog>: error:` message, the code
  of its exit status (`usage`, `server_unavailable`, …) and that code's HTTP status.
* **Progress.** The current stage is the command's own timing stage, with `done`/`total` where the
  command reports it, plus per-stage seconds and counts. Changes are pushed as server-sent events
  (`/api/jobs/events` for every job, `/api/jobs/<id>/events` for one until it ends) and are also
  available by polling.
* **Order.**
  * Jobs that use the inference server run one at a time, in submission order.
  * The other jobs start at once, up to half the CPU count together, and wait for a slot beyond
    that.
  * Two jobs never write the same map at once.
* **Cancel.** Cancelling sends SIGINT to the job's process group, the same as Ctrl-C. An
  interrupted map update does not commit, so the map is left as it was.
  * A command that ignores SIGINT gets SIGTERM after 30 s and SIGKILL after 30 s more. Its atomic
    commit still keeps the map whole.
  * A queued job is simply dropped.
  * After a restart, a job that was queued or running shows as `cancelled`.
* **Results.** The result is the command's `-o` file, and `-d` artefacts are its files in `out/`,
  so both are byte-identical to a direct run (tested against the CLI with the stub server).
  Download them with `/api/jobs/<id>/result` and `/api/jobs/<id>/files/<path>`.
* **Re-submit.** `POST /api/jobs/<id>/resubmit` runs the same parameters again, with the viewer
  step if the original had one. Its body may replace some parameters, for example a new upload in
  place of one that was discarded.

**Viewer.** The viewer's data is always served in-process by the viewer's own routes
(`viewer.routes.ViewerRoutes`). No viewer process stays alive.

* **Maps.** `/api/maps/<name>/viewer/…`, with the page URL `/viewer/map/<name>/`, serves the map's
  read-only bundle (`viewer.bundle.map_bundle`). It is rebuilt when `map.json` changes, and the 2
  most recent bundles are kept.
* **`view-image` jobs.** The job runs the viewer step on view.sh's own command line:
  * it parses that line with `spec.build_parser` and `spec.validate`;
  * it builds the bundle view.sh would serve, with the same `cli.view.make_bundle` and
    `viewer.bundle.bundle_of`;
  * it saves that bundle in `jobs/<id>/viewer/` (`save_bundle`).

  A new view.sh option reaches it with no web change. View options that mean nothing to the
  service (`Option.service = False`: `--no-browser`) are not API parameters.
  * The saved bundle holds the scene, the catalogue, the segmented image, the display transform,
    and the cloud source with its depth grid, validity, colours, labels, intrinsics and up
    direction. The live controls re-derive every cloud from that source with the shared code,
    with no inference.
  * `/api/jobs/<id>/viewer/…` (page `/viewer/job/<id>/`) serves it with `load_bundle`. The 2 most
    recent bundles are kept, and the viewer returns unchanged after a reload or a restart.
* **`view-map` jobs.** The viewer step saves a reference to the map, and the viewer is that map's
  viewer.
* **The Image page's viewer.** A single-image submission can ask for its viewer with
  `?viewer=true` (`POST /api/ops/reconstruct?viewer=true`, `segment-image?viewer=true`).
  * The viewer step is then one more step of the same job, given the command's own command line
    (its `--min-score` included). It runs before the job ends and its upload is deleted.
  * So there is one upload, one job, the command's byte-identical result, and a saved viewer.
    Uploads stay transient, and an upload still has exactly one consumer.
  * There is no second inference pass for what the command already asked. The command step records
    every inference response (`OH_MY_SLAM_INFERENCE_RECORD=jobs/<id>/inference/`,
    `client.replay`, with the depth and validity files). The viewer step replays or forwards them
    (`OH_MY_SLAM_INFERENCE_REPLAY`).
    * Responses are matched by route and request, path fields aside.
    * A request the recording does not hold goes to the server. For example, `reconstruct -f ply`
      asks no segmentation.
    * The recording is deleted once the viewer step ends.
  * As a result, the viewer's objects, ids and colours are exactly the result's, and selecting an
    object highlights the same object everywhere.
  * That viewer step is optional. If it fails, for instance because it needs the server and the
    server is down, the job still `succeeded`, its result stays downloadable, and the failure is
    the job's `viewer_error`, with the message and code (for example `server_unavailable`).
    Cancelling during that step does the same, with `viewer_error.code == "cancelled"`. The OpenAPI
    `Job` schema describes both fields.

**API overview.**

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service and inference health. When the inference server is down, the response includes `start_command`. |
| `POST /api/ops/<op>[?viewer=true]` | Submit a job. |
| `POST /api/ops/<op>/validate` | Check a request without queuing it. |
| `GET\|POST /api/uploads` | List or create uploads. |
| `DELETE /api/uploads/<id>` | Discard an unconsumed upload. |
| `GET /api/maps` | List maps, with summaries from `map.json` and the frame/object records. |
| `GET /api/maps/<name>` | One map's summary and its full `map.json`. |
| `GET /api/maps/<name>/files/<path>` | A file of the map, read-only. Hidden entries are never served. |
| `GET /api/maps/<name>/viewer/<path>` | The map's viewer. |
| `GET /api/jobs` | List jobs. |
| `GET /api/jobs/<id>` | Inspect one job. |
| `/api/jobs/<id>/events` | Server-sent progress events. |
| `POST /api/jobs/<id>/cancel` | Cancel a job. |
| `POST /api/jobs/<id>/resubmit` | Run the job again. |
| `/api/jobs/<id>/result` | Download the result. |
| `/api/jobs/<id>/files[/<path>]` | List or download the files the job wrote. |
| `/api/jobs/<id>/log` | The job's stderr log. |
| `/api/jobs/<id>/timings` | The job's timings. |
| `/api/jobs/<id>/viewer/<path>` | The job's saved viewer. |
| `/api/jobs/<id>/display-cloud[?file=<path>]` | A job's PLY as the viewer draws it, within the display budget (3D scene viewer). |
| `GET /api/display-transform` | The viewer's display transform of a scene (`cs_types`, `comment`, `camera`, `up`). |

**Web application** (spec §2.6 "Web application"). `/` serves a browser application in
`web/static/`: plain ES modules with no build step, served by the service itself (nothing from a
CDN). It talks only to the public API above, so everything it does can be scripted. `/static/…`
serves its files, `/static/viewer/…` the viewer's modules and vendored three.js (which it reuses),
and `/static/openlabel_json_schema.json` the vendored scene schema.

* **Rendered from the API description.** The operations, their parameters and their outputs are
  read from `/api/openapi.json` (each one's `x-oms` registry entry). Nothing in the app names a
  command or an option (a unit test checks this). A new option becomes a new field, a new mode a
  new form (on the page its inputs belong to), an option of a kind the app does not know a text
  field, a new output file a new download, and a new error a new message; a browser test adds all
  of these to the registry and finds them in the pages. Output entries say how to render a file
  (`object_regions`: an image painted in the objects' colours), and a video condition carries the
  suffixes of a video.
* **Forms** have one field per parameter, chosen by its kind:
  * path inputs get a drop zone and file picker that upload at once, or take a workspace path;
    ordered inputs (`mapper.sh update -i`) are numbered and can be reordered;
  * a map is chosen from the workspace's maps (the mapping mode also takes a new name);
  * `-o` is a name in the job's folder; `-d` is a checkbox plus a folder name;
  * `-p` gets one control per attribute of the mode;
  * enums, numbers and flags get the matching control.

  Each field shows its flag, help and default. Fields whose `applies` condition fails are hidden
  and not sent. Each change is checked by `POST /api/ops/<op>/validate`, and each message appears
  next to the field it names (`by_parameter`), with the command line the job will run.
* **Pages.** Each has a stable hash URL, so a reload or a shared link returns to the same state
  (the selected object is `?sel=<id>`). A new page moves the focus to its heading and is announced:
  * `#/image` (`?op=` picks the mode, `/<job>` shows its job): a mode that takes one image, a
    drop zone with a preview, the form, then the job's progress and result.
  * `#/maps`, `#/maps/new`, `#/maps/<name>`, `#/maps/<name>/update`. A map's page has the
    embedded viewer, the objects (from the map viewer's `api/scene`), the update history (each
    record of `map.json → updates[]` with every figure it holds, per-stage timings included), and
    one form per operation that takes a map.
  * `#/jobs` and `#/jobs/<id>`: the job list and a single job.
  * `#/scene?ply=<url>&json=<url>`: the 3D scene viewer (also `layers=`, `color=`, `normals=`).
* **Top bar.** Workspace name, inference-server status (with `start_command` when it is down),
  and the number of queued and running jobs, kept current by `/api/jobs/events`.
* **Results.** Every file a job wrote can be downloaded. Images are drawn; in one the API marks
  with `object_regions` (`segmented.png`), the object under a pixel is the one whose colour that
  pixel has (colour contract). CSV files are
  tables, and a scene JSON's objects are listed. A `.ply` or scene `.json` opens in the 3D scene
  viewer. A single-image job is submitted with `?viewer=true` where the operation offers it, and
  its `viewer_error` (including `cancelled`) is shown above the result, which still stands.
* **The embedded viewer** is view.sh's own page in an iframe at its stable URL
  (`/viewer/map/<name>/`, `/viewer/job/<id>/`), which also opens it full screen. Selecting an
  object highlights it everywhere on the page: in the tables, in the image regions, and on a box
  in the viewer (the viewer's `select` / `onSelect`; a click on a box picks the smallest box on
  screen under the pointer).
* **Jobs.** A cancel first says what it does: the command is interrupted as Ctrl-C would, there is
  no result, a map update leaves the map as it was, and the uploads are deleted. Re-submit runs
  the same options again, and asks for the files again when the inputs were uploads (deleted when
  the job ended): they are listed in their previous order, and each file chosen takes its place. Starting a map creation or update states its consequence in a confirmation.
* **Inference server down.** Actions of a mode that always needs the server are disabled, with
  the reason and the start command. A conditional need (`mapper.sh locate` on a large map) is
  decided by the service's own check when the form is validated. Everything else stays available.
* **3D scene viewer.** It opens a PLY, a scene JSON, or both, from a job (by URL) or from disk.
  Files from disk are read in the browser and never uploaded. Both files are drawn in the same map
  coordinates by the viewer's own modules (`Viewer`, `parsePly`, `sceneObjects`, `sceneCameras`,
  `plyCameras`, layers, labels, the camera table with *Go to*). Each layer toggle names its file.
  A scene in a single image's camera frame is shown upright with the viewer's own transform, from
  `GET /api/display-transform` (view.sh -i's `upright_transform`, with the scene's estimated up
  direction). The service decides the frame by one rule (`viewer.bundle.is_camera_frame`): a JSON's
  coordinate-system types (`cs_types=`) with no `scene_cs`, or a PLY whose header names that frame
  (`comment=`). Files are parsed and validated in
  a Web Worker, and a PLY's header is read first (the first bytes of a disk file, a `Range`
  request for a job's).
  * The point-cloud controls offer what the file allows: its colours or none, and shading by its
    normals. The segmentation layer is available when the PLY has labels and the JSON has colours.
  * A file that is not a PLY the viewer can draw is refused with the parser's reason.
  * A JSON that fails the vendored OpenLABEL schema is refused with the reasons. The app checks it
    with its own draft-07 validator (`js/scene/jsonschema.js`, kept in agreement with `jsonschema`
    by a browser test), plus the checks of `schema/validate.py`.
  * A PLY from disk with more than 16,000,000 points is refused, because the browser would draw
    it whole above the display budget (§2.5); the refusal says to view it as a map or as a job's
    file. A job's PLY above the budget is drawn from `GET /api/jobs/<id>/display-cloud[?file=]`:
    the viewer's cloud document of the file, thinned by the shared voxel-grid selection
    (`segmentation.cloud.display_selection`, each kept point with exactly its values), with the
    file's header comments (its located cameras). A binary file is read through a memory map (its
    positions, then the kept points only). The document is built once per file version, while
    other requests wait for that build, and is kept as a file in the job's folder
    (`jobs/<id>/display/`), never in memory.
* **Accessibility and layout.**
  * Everything is reachable from the keyboard, focus is always visible, and every control has a
    label. Object colours always appear with their id or label.
  * Light and dark themes follow the system and meet WCAG 2.1 AA contrast. Object colours are the
    colour contract's in both themes.
  * Pages work from desktop down to tablet width (768 px).
  * The browser tests run the vendored axe-core (`tests/browser/vendor/`) on every page, its
    embedded viewer included, in both themes and at both widths, and fail on any violation of the
    WCAG 2.0/2.1 A and AA rules.

## Point-cloud attributes

`-p key=value[,key=value…]` works with every PLY output: `reconstruct.sh -f ply`,
`mapper.sh -f ply`, and `segment.sh` with `-f ply` or `-d`. The `view.sh` controls use the same
attributes. `-p` may be repeated. Keys that are not given keep their defaults. An unknown key, a
duplicate key, a bad value, or `min-depth` ≥ `max-depth` fails with exit 2 before any inference
runs. The definition lives once, in `core/cloud_attrs.py`.

| Key | Values | Default | Scope | Effect |
|---|---|---|---|---|
| `color` | `rgb`, `segment`, `height`, `none` | `rgb` (`segment`, fixed, in `segment.sh`) | image, map | Image colour; object colour (unsegmented `#808080`); viridis ramp along up (1st–99th percentile; the estimated gravity for an image, +z for a map); or no colour properties. |
| `stride` | integer ≥ 1 | `1` | image only | Keep every n-th pixel along each image axis. |
| `min-depth` | metres ≥ 0 | `0` | image only | Drop pixels closer than this. |
| `max-depth` | metres > 0, or `inf` | `inf` | image only | Drop pixels farther than this. |
| `edge` | relative depth jump ≥ 0 | `0.04` | image only | Drop flying pixels on depth discontinuities; `0` disables the filter. |
| `voxel` | metres ≥ 0 | `0` (off) | image, map | Keep the first point (in pixel or storage order) of each voxel. Colours are never averaged. |
| `normals` | `on`, `off` | `off` | image, map | Add `nx ny nz`. For an image they come from the depth grid. For a map they come from each point's 16 nearest neighbours in the whole map, are oriented towards the keyframe cameras, and are computed only for the emitted points. |
| `label` | `on`, `off` | `off` | image, map | Add `int label`: the object id, `0` = unsegmented. |
| `encoding` | `binary`, `ascii` | `binary` | image, map | `binary_little_endian 1.0` or `ascii 1.0`. |

The pixel-level keys (`stride`, depth range, `edge`) act before unprojection. A map is refused
them. Attributes shape only the emitted cloud; objects, ids, OBBs and colours never depend on
them.

**PLY layout.** The vertex properties come in this order. Each group is present only when
enabled:

| Group | Properties | Present when |
|---|---|---|
| position | `float x y z` | always |
| normals | `float nx ny nz` | `normals=on` |
| colour | `uchar red green blue` | unless `color=none` |
| label | `int label` | `label=on` |

The header has two `comment` lines:

* The frame line:
  * `oh-my-slam camera frame (OpenCV axes: x right, y down, z forward), metres` for an image.
  * `oh-my-slam map frame (z up), metres` for a map.
* `attributes key=value,…`: the effective attributes, with defaults included. For a map, only
  the keys that apply to maps are listed.

The same source and attributes always give byte-identical files.

## Output contract and exit codes

* **stdout** carries at most one payload: one JSON document (compact, newline-terminated) or one
  PLY, and no banners or progress. The commands redirect fd 1 to stderr before any library runs,
  so output from native code (COLMAP, Open3D) lands on stderr too.
* **`-o FILE`** (`reconstruct.sh`, `mapper.sh`, `segment.sh`) writes that payload atomically to
  `FILE`, creating parent folders, and leaves stdout empty. A `-o` target (and a `segment.sh -d`
  folder) that is a folder, or where nothing can be written, is a usage error (exit 2) raised
  before the server is contacted or any work starts.
* **stderr** gets everything human-facing. Log lines start with `[oh-my-slam]`, and errors look
  like `<command>: error: …`.
* **Timing summary.** `reconstruct.sh`, `segment.sh`, `mapper.sh update` and `mapper.sh locate`
  each log a one-line `timings:` summary.
* **stdout of the other commands.** `view.sh` writes nothing to stdout. `--status` writes the
  health JSON. The help (`-h`) of every command goes to stderr.

| Exit | Meaning |
|---|---|
| 0 | Success. A consumer closing the pipe early (`\| head`) also exits 0. |
| 1 | Internal error. Also used when the server stays busy after retries, when inference fails, or when COLMAP is missing or the wrong version. |
| 2 | Usage or input error: bad option, bad `-p`, missing or unsupported input file, `.venv` missing; for `mapper.sh locate` also a missing or empty map folder, an `-o` inside the map, or no image located. |
| 3 | The inference server is not running, or its models failed to load. The message says what to run (`./start_inference_server.sh`, or a restart after fixing the cause the log names). |
| 4 | `-m` folder is not empty and not a map (`mapper.sh`), or is not a map (`segment.sh -m`, `view.sh -m`). |
| 5 | Nothing could be registered, for example because the new images do not overlap the map. The map is unchanged. |
| 6 | Another `mapper.sh update` holds the map lock. |
| 130 | Interrupted (Ctrl-C; a cancelled job). `view.sh` treats Ctrl-C and SIGTERM as its normal stop and exits 0, also when it was started as a shell background job. |

For the web service (spec §2.6), `core/errors.py` maps each exit code to a machine-readable code
(its lower-case name, e.g. `not_a_map`), an HTTP status by one rule (`HTTP_STATUS`: 2 → 400, 4
and 5 → 422, 6 → 409, 3 → 503, 130 → 499, 1 and anything else → 500) and a job state
(`job_state()`: 0 → `succeeded`; 130, or a process stopped by SIGINT or SIGTERM → `cancelled`;
any other → `failed`).

## Coordinate conventions

* **Units** are metres everywhere.
* **Single image (the scene frame)** is the image's camera frame, with OpenCV axes: x right,
  y down, z forward. The camera sits at the origin with the identity pose. `reconstruct.sh`,
  `segment.sh -i` and `view.sh -i` report points, boxes and the camera in this frame.
* **Map frame:**
  * The axes are x forward, y left, z up. The frame is gravity-aligned.
  * The origin is the camera centre of the first keyframe (in capture order) that the first
    update posed.
  * x is that camera's viewing direction projected onto the floor, and y = z × x.
  * Up is the confidence-weighted mean of the per-keyframe gravity estimates (GeoCalib refined by
    the floor plane). The map is then levelled so that the floor normal of the whole map is +z
    (corrections up to 10°).
  * A map created from a single image uses that image's pose and gravity.
  * Later updates are expressed in the same frame.
* **Poses** are camera-to-parent: the camera's axes and centre in the scene or map frame.
* **OBBs** are upright: the box z axis is the estimated up, and only the yaw is fitted. The box
  axes are x = width (the longer horizontal side), y = depth, z = height. The yaw lies in
  (−90°, 90°]. The fit keeps the least-area yaw over the convex-hull edge directions, with robust
  2nd–98th percentile extents. For floor-standing classes, a box whose visible bottom floats up to
  0.8 m above the detected floor is extended down to it (person 1.2 m, dog and cat 0.5 m, horse
  1.0 m).

## Scene description (OpenLABEL 1.0.0)

Every JSON output is `{"openlabel": {…}}` and validates against the vendored, sha256-pinned
OpenLABEL 1.0.0 JSON schema (`src/oh_my_slam/schema/`). The schema allows no root key besides
`openlabel`, so the schema URL goes in `metadata.schema_url`. The code also checks what the
schema does not: stream intrinsics, the top-level `frame_intervals`, the cuboid shape and unit
quaternions.

| Field | Single image | Map |
|---|---|---|
| `metadata` | `schema_version` `"1.0.0"`, `schema_url` `https://openlabel.asam.net/V1-0-0/schema/openlabel_json_schema.json`, `name` (file name), `annotator` `"oh-my-slam <version>"`, `tagged_file`, `tool` (`reconstruct`, `segment` or `view`), `intrinsics_source` (`exif` or `model`), `depth_grid`, `gravity` (`up_cam`, source, uncertainties, floor height) | The same base fields, with `tool` `"mapper"` (`"segment"` in the output of `segment.sh -m`), plus `map_frame`, `scale` (SfM → metric), `update_count`, `keyframes`, `floor_z`, and `scope: "single"` for `-t single` |
| `ontologies` | `"0"`: `uri` `https://www.lvisdataset.org/`, `boundary_list` (vocabulary ∪ labels found), `boundary_mode` `include` | the same |
| `coordinate_systems` | `camera`: `sensor_cs`, root | `map`: `scene_cs`, root, `axes` `"x-forward,y-left,z-up"`, `gravity_aligned`, `units` `"m"`; one `camera_<id>` `sensor_cs` child per COLMAP camera |
| `streams` | `camera`: `intrinsics_pinhole` (`width_px`, `height_px`, 3×4 `camera_matrix`, zero `distortion_coeffs`), `intrinsics_source`, `uri` | `camera_<id>`, the same fields (full-resolution intrinsics; source `colmap` once SfM has refined them) |
| `frames` | `"0"`: `timestamp` 0, stream `uri` | one per keyframe, keyed by keyframe index. `timestamp` is the index (capture times are never read), plus stream `uri` (`frames/fNNNNNN.jpg`), `keyframe`, `pose_source`, `update_id`, `low_confidence`, and a transform `camera_<id>_to_map` (see below) |
| `frame_intervals` | closed intervals over the frame keys | the same |
| `objects` | keyed by id (below), `coordinate_system` `camera` | the same, `coordinate_system` `map` |

Each keyframe's transform is `{src: camera_<id>, dst: map, transform_src_to_dst: {quaternion:
[qx, qy, qz, qw], translation: [x, y, z]}}`. It is the camera-to-map pose, with the quaternion
scalar last.

Each object is keyed by its id as a string and has these fields:

| Field | Value |
|---|---|
| `name` | `"<label> <id>"` |
| `type` | the label |
| `ontology_uid` | `"0"` |
| `object_data.cuboid[0]` | `name` `"obb"`, `coordinate_system`, `val` = `[x, y, z, qx, qy, qz, qw, sx, sy, sz]` (centre, box-to-parent rotation with the scalar last and `qw ≥ 0`, size = width, depth, height), and `attributes.num` `width_m`, `depth_m`, `height_m`, `volume_m3` |
| `object_data.num` | `score`, `pixel_count`, `point_count`, `observations` |
| `object_data.text` | `color_hex` |
| `object_data.vec` | `color` `[r, g, b]`; for a map object detected under several labels, also `detected_as` (every label, most evidence first) |
| `frame_intervals` | the frames that detected the object |

For a map object, `score` is the mean of its three best detection scores, and `point_count` is
the number of map-cloud points attributed to it (its points in `segments.ply`), as for a single
image, where it counts the object's lifted points.

## Colour contract and palette

An object's colour is a pure function of its id (`segmentation/colors.py`). The same sRGB triple
appears in all of these:

* the JSON `color` / `color_hex`
* the mask pixels of `segmented.png`
* `catalog.csv` and `catalog.md`
* `segments.ply`, and every PLY or viewer cloud with `color=segment`
* the OBB rendered by `view.sh`

Masks are opaque, and each pixel and point belongs to at most one object.

Every object colour reads on the viewer's dark surfaces and on the dimmed `segmented.png`:

* WCAG contrast of at least 3:1 (the WCAG 1.4.11 minimum for graphical objects) against the
  viewer's panel `#1d2027`, and therefore against its darker canvas `#15171c`. That also makes
  every object colour brighter than any pixel of the dimmed photo (35 % of white).
* OKLab lightness at most 0.93 (no near-white), OKLab chroma at least 0.08, and an OKLab distance
  of at least 0.11 from the unsegmented mid-grey.

Ids 1–19 use this palette. Ten colours come from Sasha Trubetskoy's list of distinct colours; the
entries that fail the rules above (navy, maroon, purple, beige and the palest pastels) are
replaced. Any two differ by at least 0.095 ΔE_OK, about five just-noticeable differences:

| id | colour | id | colour | id | colour | id | colour |
|---:|---|---:|---|---:|---|---:|---|
| 1 | `#e6194b` | 6 | `#9c4dff` | 11 | `#1fb5a3` | 16 | `#ffc49b` |
| 2 | `#3cb44b` | 7 | `#42d4f4` | 12 | `#dcbeff` | 17 | `#5a9cff` |
| 3 | `#ffe119` | 8 | `#f032e6` | 13 | `#9a6324` | 18 | `#b4339c` |
| 4 | `#4363d8` | 9 | `#a8f04a` | 14 | `#8ef0c0` | 19 | `#c9a227` |
| 5 | `#f58231` | 10 | `#ff9ec7` | 15 | `#808000` | | |

Higher ids cycle by hue, and vary lightness and chroma as they go:

* Id `19 + n + 1` aims at the OKLCh hue `n × 137.508°` (the golden angle).
* The candidates are the 8-bit colours within ±30° of that hue, on six lightness tiers
  (0.62–0.91) and three chroma levels (0.24, 0.15 and 0.10, clipped to the sRGB gamut), that meet
  the rules above.
* The id takes the candidate farthest in OKLab from every earlier id, with the last 10 ids counted
  1.5 times as close, so that neighbouring ids differ most.
* A colour that would repeat an earlier id after 8-bit rounding is nudged to the nearest unused
  triple. Every id therefore gets a distinct colour.

For ids 1–120, any two colours differ by at least 0.045 ΔE_OK and consecutive ids by at least
0.15 (`tests/unit/test_colors.py`).

Mid-grey `#808080` (128, 128, 128) is reserved for unsegmented points, and the palette never
produces it. The `color=height` ramp is viridis.

## Maps

### Folder layout

| Path | Contents |
|---|---|
| `map.json` | Format version, map frame, scale, next frame and object ids, `floor_z`, and the update history. Each update records its inputs, frames added and rejected, SfM method, notes, object summary and per-stage timings. Written last. |
| `frames/fNNNNNN.jpg` | Keyframe images (upright JPEG copies). |
| `frames.json` | Per keyframe: source, camera id, intrinsics, `T_map_cam`, depth grid, pose source, update id, depth scale, `low_confidence`, and registration stats. |
| `per_frame/fNNNNNN/` | `depth.npy` (float16 aligned metric depth), `valid.png` (validity after latest wins), `instances.json` (RLE masks, labels, scores, object ids) and `descriptor.npy` (retrieval descriptor). |
| `sfm/database.db`, `sfm/model/` | The COLMAP database, and the COLMAP model in map coordinates. |
| `cloud.ply`, `cloud_objects.npy` | The map cloud, and the object id of each of its points. |
| `objects.json`, `objects/points_NNNNNN.npy` | Object state (evidence, strikes, merges) and each object's canonical points. |
| `.lock`, `.staging/` | The update lock, and the staging area of an update in progress. |

An update writes everything into `.staging/`, then records the list of staged files (the commit
point), moves them into place, and writes `map.json` last. An update killed before the commit
point leaves the map untouched. One killed after it is completed by the next update. Readers
(`segment.sh -m`, `view.sh -m`) never lock or write. Do not edit files in `.staging/`. A reader
that starts while an update applies its commit can read a mix of old and new files: it reads a
file that vanishes in between once more, but it has no snapshot of the whole map, and a rebuild,
which replaces most files, widens that window.

### Update semantics

* **The order of addition is "latest".** A later update wins over an earlier one, and within one
  update a later keyframe wins over an earlier one, in input order (the order of the `-i` images;
  a video's frame order). A sequence mapped in one update or split across several in the same
  order gives the same objects, labels and boxes, within the evaluator's stability targets (spec
  §2.3 and §5; not to the centimetre: the threaded global mapper is not deterministic, and two
  runs of one update differ too); only ids may differ, where an earlier update had already
  published one, since a published id persists. An object is judged by the keyframes added
  after its last detection wherever the updates are cut, and a map of photos with a keyframe SfM
  did not pose is rebuilt with the next update's photos (see *Rebuilding a weakly posed map*
  under Mapping, step 4): `office_sequence` in one update, 6 + 7 and 4 + 4 + 5 photos gives the
  same 8 objects and labels, box centres 1-2 cm apart (median box IoU 0.63-0.73 against the
  one-update map; a second one-update run: 0.81-0.93), 7 of the 8 ids, and no cup, its place
  drawn from the photos that see it empty. Where nothing changed, the
  keyframes of an update agree and their order does not matter:
  * Object association groups all of the update's instances at once, strongest agreement first.
  * The cloud's colours and object ids do not depend on keyframe order either.
  * Order affects keyframe names, the numbering of new objects (by their earliest keyframe), and
    every place that changed (see *Later wins*).
* **Later wins.** "Latest" is the order of addition. A later update invalidates, per pixel, the
  parts of older keyframes that it contradicts: free space seen behind an old point, or a new
  surface in front of an old ray. It also removes, or gives a strike to, objects that it sees
  through. Keyframes of the same update invalidate each other's pixels only where an object
  changed during the update (below). Capture timestamps are never read.
  * **Which keyframes judge an object.** Every object of the map (every confirmed one: a
    candidate that later keyframes see through is one whose depth nothing confirmed, and it stays
    a candidate) is judged by the update's keyframes added after its last detection — within the
    update as across updates, whether the camera looked away or not, from any viewpoint and
    distance. A keyframe judges when its pose error, seen from the object's distance, is at most
    15 % of the object's extent (2° is 2.5 cm at 0.7 m): an SfM pose with 100 points whose
    reprojection error (as an angle: pixels over the focal length) is below that, or a pose
    refined with feature matches whose residual is; it is not low confidence, has at least 80 % of the object's
    samples in its image (the image border counts), at least half of them unoccluded, on at
    least 20 distinct depth pixels, and agrees with the object's detections about its
    surroundings. Monocular depth of two keyframes disagrees by a smooth factor (tens of percent
    between viewpoints metres apart) and a video's loop can be misaligned by decimetres, so the
    surface in a band around the object's mask in the detecting keyframe nearest by viewpoint
    (half the mask's size wide: its support, the wall behind it) is lifted into the map and
    compared with the judging keyframe's depth: where it sees the same surface (within a factor
    1.3), their median ratio must lie within 25 % of 1 and half of the band within the depth
    noise, max(5 cm, 8 % of the depth), of it once divided out. The keyframe's depth is divided
    by that ratio. A keyframe that detected something at the object's place does not judge it:
    another object, at most twice its size (not its support), whose mask holds half the
    object's samples, or 20 of them when a keyframe detected that object at the object's own
    place (the object under another label, which a keyframe across the room places a little
    apart: a bag seen as a handbag), unless a keyframe detected it beside the object.
    Only places within the fused depth are judged (the depth to which the map's fusion draws a
    keyframe, 2.5 times the map's median depth, at most 30 m, see Integration): an object is
    judged only when a keyframe that detected it has it within its fused depth, and only by
    keyframes that have most of its samples within theirs. Beyond it the map draws no surface,
    and monocular depth places a car 40-100 m down a street metres off: on `street.mp4` such
    distant cars were "removed" or "moved" by later keyframes whose depth looked past the box
    beside them.
  * **Removing an object.** A sample is seen through when the keyframe sees farther than it by a
    margin: max(0.25 m, 15 % of the depth), but for a small object at most 15 % of its own
    extent (at least the depth noise): a cup on a windowsill is seen through by 5-20 cm. The
    keyframe's verdict is the share of the samples it sees through beyond what it would see
    through anyway: the share the object's own detecting keyframes see through (its 3 largest
    detections: monocular depth sees between a ladder's rungs and a plant's leaves, and places a
    small object differently from one keyframe to the next), or the share of the object's
    surroundings the judging keyframe sees through, where nothing changed, whichever is larger.
    A sample where the keyframe sees the object's support at its foot (the surface within the
    margin lies in the object's bottom quarter, at most the margin above its base) tells
    nothing: the keyframe sees the table or the windowsill there whether the object stands on
    it or not. The share leaves such samples out; they count as seen in place only for whether
    the keyframe sees the object in place. In `office_sequence` mapped in two updates the cup's
    four detections are placed 5-15 cm apart (the first update's four photos have their
    matches mostly on the trees behind the window, see Known limitations), so its detecting
    keyframes see through half of its samples themselves, and the sill at its foot held 15 % of
    them within the margin in the two photos that see its place empty: both saw it gone by
    0.63-0.70, and it only got a strike.
    The latest keyframe wins: a keyframe that sees the object in place (at most 40 % through)
    outweighs every earlier one, so only the keyframes added after the last such one count. The
    update removes the object when most of those see through at least 60 % of it: three
    keyframes remove it; two remove an object detected in at least 4 keyframes when both see at
    least 70 % through from well-supported poses (absence, like presence, is confirmed by two
    keyframes); else the object gets a strike, and a second strike from a later update removes
    it. An update that sees it in place clears its strikes; an update counts once, however many
    of its keyframes judge.
  * **The removed object's pixels go too.** Its detection masks (grown by 2 px) are invalidated
    in the keyframes that detected it (`valid.png`), so the fused map cloud loses the object's
    points together with its record.
  * **Its place shows the latest observation.** The keyframes that saw through the object (its
    witnesses) are then the only ones that see the surface it stood on or hid, often fewer than
    the 3 keyframes a surface usually needs (the cup is removed by 2 photos). The map records
    the place (`objects.json → vacated`: the retired masks, the object's box and the witnesses)
    and, in it, draws what the witnesses see however few keyframes see it, takes colour and object
    id from the witnesses (not the cup's shadow in the older photos), and drops points that the
    witnesses see through and no later update sees (what is left of the object). The witnesses
    are one observation of the place. A witness that sees a nearer surface by more than twice the
    depth noise (another object in front) says nothing about a point; per point, the median over
    the others of how far beyond it they see decides whether they see it (within the visibility
    tolerance, drawn) or through it (dropped, unless one of them sees it). Their monocular depth
    of the empty place disagrees by a few percent: in `office_sequence` mapped in three updates
    (4 + 4 + 5 photos) the two photos that see the cup gone placed the sill 3.5 cm in front of
    and 3.2 cm behind the surface the other photos agree on, and when either witness alone could
    carve that surface, the one that saw farther left a hole of about 8 × 10 cm where the cup
    stood. A place hidden from a witness by more than half (a vase in front of the cup) is not
    judged by it: when only one keyframe judges, the object gets a strike and stays. The
    place is its detecting keyframes' retired pixels at or behind the object (up to the depth
    noise, max(5 cm, 8 % of the depth), in front), and its box grown by max(5 cm, 3 % of its
    viewing distance). A later update that sees the place again is fused like any other.
  * **A place that changed within one update.** The order of addition is the only sign of
    "latest", so an update whose last keyframes contradict its first ones ends with the latter:
    mapping `examples/office_sequence` in one update leaves the map without the cup, as mapping
    it in two does. An object the update detected is judged, by the rules above, by the
    update's keyframes added after its last detection. An object the update first detected where
    its earlier keyframes saw free space (judged by the same rules, with three keyframes at
    least) arrived: the pixels with which those keyframes saw through it are retired, so that its own
    keyframes draw it however few they are. The ids of the update's detections stay counted, so
    the other objects keep theirs.
  * **An object that moved.** Objects are re-identified by label, size and colour: an object
    detected only by keyframes added after every detection of another of a compatible label, of
    comparable size (scales within a factor 1.5) and colour (median CIELab of its detections'
    pixels: chroma within 10, lightness within 25), standing apart from it, is that object moved
    there when the places were judged empty in their turn: the old place by at least one
    keyframe added after its last detection, the new place by at least one keyframe added before
    the first detection there (keyframes that judge as above, from at most 1.5 times the
    farthest distance the object was detected from — their silence counts only where the
    detector would have seen it at 2/3 of its smallest detected size — and without a detection
    of something of its size there that its own keyframes did not detect beside it), and at one
    of the two places the verdicts show the change as they show a removal (three keyframes, or
    two for an established object at its old place). Monocular depth cannot tell a thin object from the
    wall right behind it (a paper-towel roll on a shelf at 2.5 m), so the detector's silence at
    one place counts, but only with the depth's evidence at the other. The object keeps its id at
    its new place, with the box and points of its latest detections; its old place is vacated
    like a removed object's (its old detections name no object any more, and the keyframes that
    saw it empty draw it), and the pixels with which earlier keyframes saw through its new place
    are retired, so its latest keyframes draw it there however few they are. In 10 photos of a
    living room mapped in one update, a paper-towel roll stands on a shelf in photos 6–7 and on
    the dining table in photos 8–9: the map shows it once, on the table, with the id it got on
    the shelf. A cup of another colour that
    appears elsewhere is another object, and a second cup beside one that stays is a second one.
* **Persistent identity.** Each object keeps one id and one colour for the life of the map, and
  its OBB is refitted from all accumulated evidence:
  * New ids come from a counter and are never reused.
  * Duplicates are merged, and a merge keeps the lower id. Objects of compatible labels merge
    when their points or boxes overlap. Objects of different labels merge when the detector's
    label flickered between keyframes: no keyframe detected both, their sizes are comparable
    (not a part of the other or an item resting on it), and at least half of either one's
    points lie on the other's surface. The merged object takes the label with the most
    evidence and lists the others in `detected_as`.
  * **A part named on its own** joins its object. The detector can name an object in some
    keyframes and only a part of it in others: in `office_sequence` a solar figurine is a
    "figurine" in two photos and its top alone a "bottle opener" in two others. The smaller of two
    objects of different labels (less than half the other's size, but at least a quarter: a
    handle on a door or a faucet in front of a window is an item of its own) that no keyframe
    detected together merges into the larger when each is detected reliably in at least 2 keyframes, at
    least 90 % of its points lie inside the larger's box (grown by 2 cm) and 80 % on the larger's
    own surface (within max(1 cm, 2 % of the viewing distance) of its points), every keyframe that
    detected either had at least 80 % of the other in its image and mostly unoccluded, the
    larger's keyframes saw the part from at most 1.5 times the distance it was detected from, and
    in each of them that sees at least 10 of the part's points (at least 2 must) 80 % of those lie
    on its detection of the larger object or in free space where it sees past a thin part, more
    of them on it than free. An item resting on or in a larger object (a cup on a
    table, a book in a bookcase) is usually detected together with it by some keyframe, lies
    beside the larger object's masks in most of its keyframes, or is out of view in the other's
    keyframes, and stays separate. The part's label votes count in proportion to its size, so the
    object keeps the whole's label.
  * **Copies placed by inconsistent depth** are merged too. Keyframes whose monocular depth
    disagrees (locally, even after the global depth adjustment of Refinement) can place an
    object twice along the same viewing rays. Objects of compatible labels that no keyframe
    detected together merge when, for most of the 3 pairs of their keyframes nearest by
    viewpoint, the two keyframes' depths disagree by at least 5 % (measured on the surfaces both
    see) and the detections coincide once that ratio is removed.
  * **Objects seen twice** are merged. Keyframes that see an object from different sides of a
    room place it at depths a few percent apart, and each set puts its copy on its own viewing
    rays: in the `livingroom.mp4` map a lamp seen along x from one side and along y from another
    was two objects 0.2 m apart, and a door handle seen from 1.7 m and from 4.5 m two objects
    0.12 m apart. The detector's label can also differ between the sides (a robot-vacuum dock
    seen as a toilet from one side and as a dryer from the other). Two objects that no keyframe
    detected together, each detected reliably in at least 2 keyframes, of compatible labels or of
    comparable size, merge when both of these hold:
    * The keyframes of each saw the other's place as the object they detected. The other's points
      that such a keyframe sees (unoccluded, in view) lie on its detection, grown by the depth
      noise at the object's distance (max(5 cm, 3 % of the distance)), or in free space, where
      the keyframe sees farther. At most a fifth of them may lie beside its detection on another
      surface, where the keyframe saw something else (two lamps side by side, each detected from
      one side only, stay two).
    * Each object, moved along its own viewing rays by a depth factor within ±15 %, lies on the
      other's detections for at least 40 % of the points those keyframes see, and the two moved
      copies stand in one place: along both horizontal axes, the middle of each lies within the
      other's extent grown by the depth noise (a metal rack seen in front of a sideboard, but
      standing beside its end, stays apart).
  * **Horizontal surfaces** (tables, desks, counters, beds, rugs) are often split by the
    detector around the items that rest on them: a counter top around a book becomes a left and
    a right "desk". Two detections of one keyframe with compatible surface labels whose masks
    touch with continuous depth (at least 50 contact pixels, depth within 3 %) are one
    detection. Pieces of one surface seen from different keyframes (a desk from one side, a bed
    or a rug from another) merge whatever surface labels they carry, when no keyframe detected
    both and either they share surface — at least 100 of their 1 cm points within 5 cm
    horizontally and 10 cm vertically, at median heights within 10 cm — or the map's fused
    surface joins them: one connected, horizontal patch of it (local normal within ~30° of
    vertical) reaches at least half of each piece's points, at median heights within 15 cm and
    at most 0.3 m apart. The second test catches pieces that monocular depth of a close surface
    placed apart (the kitchen island's corner, 0.5 m below the camera, placed 10 cm lower by the
    keyframes that close the loop). At floor height it is not used: the floor joins anything.
  * **Every object of the map is a confirmed one.** A detection is only a candidate until it is
    confirmed, and a candidate is not an object of the map: `-t full` and `segment.sh -m` list
    every object, that is every confirmed candidate (`-t single` those that the new frames
    observe), and the scene JSON carries no confirmed flag, since every object in it is. A
    candidate is confirmed once it is detected with a reliable mask (not mostly in the
    image-border band, where the object is cut off and monocular depth is unreliable) in at
    least 2 keyframes, or in 1 when no other keyframe of the map had it in view (occlusion is
    ignored, so a detection whose depth puts it inside another surface cannot confirm itself).
    It must also be visibly drawn in the map cloud, so that it appears in `segments.ply` and
    every `color=segment` cloud in its colour: it needs at least 5 % of the cells of its box's
    largest face at the map's sampling at its nearest detection, and at least 10. A cell is a
    cloud voxel, or the footprint of one depth-grid pixel where that is coarser: a car seen from
    40 m at the nearest in a 1080p video (grid focal 660 px) has one depth pixel per 6 cm, 9
    voxels of 2 cm, and its detections cannot give it more points than its masks have pixels. A
    keyframe's vote for an object counts only for cloud points inside the object's attribution
    gate, its box grown by the depth noise at its viewing distance (max(5 cm, 3 % of the
    distance)): detection masks are drawn generously (a "carpet" mask over a counter top and the
    floor beyond it), and the object's coloured points must coincide with its box. The vote
    needs a third of the keyframes that see a point, so an object detected in fewer of them wins
    only part of its surface (a refrigerator detected in 7 of the ~15 keyframes that see its
    front) or none of it (a light switch on a wall, a dishwasher detected in 2 of ~13). Each
    confirmed object therefore also takes the unlabelled cloud points nearest its own lifted
    points (the 4 nearest to each, within max(3 cm, 2 % of its viewing distance), inside its
    gate; a point two objects pick goes to the nearer), provided a keyframe that detected it
    fused it: an object detected only beyond the fused depth (a car 40–100 m down a street) has
    no surface of its own in the cloud, and the points near its far-placed lifted points are
    other surfaces. An object stays short when its surface did not survive the fusion (other
    keyframes look at it without seeing it, see *Geometry*) or when its detections are all beyond the
    fused depth and do not meet the cloud; it is then still a candidate.
  * Unconfirmed candidates are kept in the map's state, so that a later update can still confirm
    them; they are never emitted, in any format or scope.
  * The OBB is fitted to the detections that agree with each other: monocular depth of a small
    object can vary by tens of percent between keyframes, and the union of such detections is a
    streak along the viewing rays. With 3 or more detections, the box covers those whose bounds
    overlap (allowing for depth noise) the detection most others agree with.
  * Only the best evidence shapes the OBB. Once 2 confidently placed keyframes detected an
    object, the detections of low-confidence keyframes no longer shape its box: an uncertain pose
    moves the whole detection, and in a split `office_sequence` map it placed the wallet onto the
    Raspberry Pi case beside it. Of the rest, detections mostly in the image-border band are left
    out once the reliable ones are at least 2 and the majority. The other detections still count
    for the label and for confirmation. Each object records which kind of detection each of its
    points came from, so the box is fitted to the points of the detections that shape it.
  * A mask that ran onto the surface an object stands on or rises from is trimmed before the
    detection is used. The mask points in the object's bottom band (a quarter of its height) that
    lie outside the plan footprint of its points above that band are removed from the mask and the
    points: a strip of the windowsill in front of a window, the floor in front of a chair. This
    happens only when the object is mostly above that band and the strip is at most a quarter of
    its points, so a keyboard or a laptop's base is left whole. Heights decide, not surface
    normals, because monocular depth often bends such a strip into the object's own surface.
  * The OBB covers the **observed surface** only (see Known limitations): no class-typical size
    is assumed.
* **Geometry.** The map's geometry is a point cloud: the surface of a TSDF fusion of the aligned
  depth maps. Each point takes its colour and object id from the latest update that sees it. No
  mesh is produced. A surface is drawn where 3 keyframes updated it in the TSDF, or, where
  fewer keyframes have it in view, where all of them updated it: the laptop and the right half
  of the monitor that only photos 0 and 12 of `office_sequence` show, or a backpack that only
  photo 7 shows. A keyframe has a point in view when the point lies within its fused depth and
  projects onto a pixel it fuses (valid, not a depth edge, within that depth), whatever that
  pixel shows. Speckle that other keyframes look at without seeing it (a blob one keyframe
  places in front of a wall that the others see) still needs 3 of them, and so does the second
  zero crossing that 1 or 2 keyframes leave a centimetre or two beside a surface many see (the
  rim of the TSDF band). What only one or two
  keyframes see is drawn only from keyframes whose depth scale was measured with a spread
  (IQR / median of the depth ratios) of at most 0.1: nothing else checks a keyframe there, and a
  keyframe that fits the map worse (a stretch of `livingroom.mp4` scaled with spreads of
  0.14–0.27) would draw a surface where it is not. Low-confidence keyframes are not fused.

## How it works

### Inference server

`src/oh_my_slam/server` is a FastAPI app on a Unix socket, `~/Library/Caches/oh-my-slam/srv.sock`
(mode 0600). It falls back to a short `/tmp` path if the socket path is too long. `/health`
answers immediately, even while the models load. It loads four models:

| Model | Role |
|---|---|
| MoGe-2 ViT-L normal | Metric point map, depth and validity, intrinsics, plus a DINOv2 class-token descriptor for retrieval. Its normal head is not loaded: nothing reads its normals, and the other outputs are the same without it. |
| GeoCalib (pinhole) | Gravity direction with uncertainty. |
| YOLOE-26x-seg with the MobileCLIP2-B text encoder | Open-vocabulary instance masks over `segmentation/data/default_labels.txt`, a curated list of LVIS and COCO nouns. |
| MapAnything (Apache-2.0 checkpoint) | Metric multi-view poses, used as a fallback. |

The device is MPS when available, else the CPU, with a per-process MPS memory cap of 70 %. MPS is not thread-safe, so every model call runs on one GPU worker
thread. The queue holds 8 jobs; beyond that the server answers HTTP 503 and the client retries.
torch lives only in the server process. The commands never import it, because loading torch and
Open3D in one process aborts on a duplicate libomp.

### Single image

These steps serve `reconstruct.sh`, `segment.sh -i` and `view.sh -i`:

1. **Intrinsics:** EXIF focal length, else MoGe's estimate.
2. **Depth:** MoGe-2 metric depth on a grid whose long side is at most 1024 px. Point colours
   are the resized pixels.
3. **Gravity:** GeoCalib, refined by a RANSAC floor plane within 5°.
4. **Detection:** YOLOE detections above the floor (0.05 below the default threshold, 0.5 from
   it up), in two passes over the image: a fine pass on the image at 1024 px, and a coarse pass
   on the image at 768 px, which the detector upsamples to its 1024 px input. The coarse pass
   finds large plain objects that fine detail misleads, such as a dark monitor showing a menu
   bar, and contributes only detections covering at least 5 % of the image. It is skipped for
   images of 768 px or less. A detection overlapping a higher-scoring one of the other pass
   (mask IoU > 0.5) is the same object and is dropped. What the detector is shown never depends
   on the depth grid: the masks are resampled onto it, so the mapper's 768 px keyframes get the
   same detections as `segment.sh -i`. Background labels (wall, floor, ceiling, …) are prompted
   but never reported. Masks under 64 px are dropped, and duplicates across labels are removed
   (mask IoU > 0.7, higher priority wins).
5. **Exclusive masks and lifting:** the claim order above gives every pixel at most one owner,
   and the masks are lifted to 3D without depth-edge pixels.
6. **OBBs:** each object gets an upright OBB, then ids and colours are assigned. A box of a
   floor-standing class is extended down to the floor when its visible bottom floats at most
   the class's gap above it (0.8 m for furniture and doors, 1.2 m for a person, 0.15 m for
   classes that also stand on furniture or hang on walls, such as plants and shelves). The
   visible part must span at least a fifth of the grounded height, except for classes seen
   mostly from the top (tables, desks, counters, beds), and boxes under 10 cm are never
   grounded. A door behind a kitchen island, or cut off by the bottom of every keyframe that
   detected it, therefore still reaches the floor.

Geometry and detection requests run concurrently on two connections. The server reads each
request's image from disk and downscales it to the long side its model reads (768 or 1024 px,
640 for gravity, 1024 for multi-view poses) on its one device thread, where decoding and
resizing a 12 MP photo cost more than some of the models. The client therefore decodes each
image once and sends every request the image already at that size, as an uncompressed BMP
that the server decodes in about a millisecond and does not resize: the model reads the same
pixels, so its results are unchanged.

### Mapping (`mapper.sh update`)

1. **Lock and stage.** Resolve the inputs into keyframes.
2. **Per-keyframe inference.** Each keyframe gets depth (768 px grid), gravity, a descriptor and
   detections at the default threshold (the single-image detection, masks on the 768 px grid).
   Two keyframes are processed at a time, each as soon as ingest has written it, so the video
   is decoded while the first keyframes' inference runs. For a new map whose keyframes have
   one size, the SIFT features of step 3 are extracted on the CPU while the server's GPU runs
   this inference; their camera gets its prior focal length (the median of the keyframes'
   estimates) once inference is done, which leaves the database as extracting afterwards does.
3. **Features and matching.** The Homebrew `colmap` CLI extracts and matches SIFT features.
   SIFT doubles each photo for its finest scale (COLMAP's default), but not the keyframes of a
   video of at least 1600 px: for those it was most of the COLMAP time and memory (the lv walk:
   36 s and 8.7 GB doubled, 10 s and 2.4 GB not). Pairs are chosen as follows:
   * Photos: every pair up to 200 images.
   * Otherwise: sequential neighbours plus descriptor retrieval, with loop-closure candidates
     for video.
   * Updates: new keyframes are matched against the whole map up to 150 keyframes, and against
     retrieved keyframes beyond that.

   **Weak links of a video** are matched again with LightGlue on the same SIFT keypoints
   (COLMAP's `SIFT_LIGHTGLUE`, on the CPU). A cut between two consecutive keyframes is weak
   when at most one verified pair of keyframes at most 12 apart spans it (loop closures do not
   count): what lies beyond hangs on one link. Every listed pair that spans a weak cut, is at
   most 6 keyframes apart and was not verified is matched again; SIFT's result stays where it
   verified more. On the office walk the white wardrobe doors and the door
   (`f000101`–`f000109`, 67–1300 SIFT keypoints each) had no verified pair to the rest at one
   end and 0–1 at the other. The global mapper left these nine keyframes a reconstruction of
   their own in 4 of 5 builds, and the fifth joined them on one 15-inlier pair, 5 % off in
   depth against their neighbours. LightGlue verifies 15–16 of the 41–42 pairs across those cuts
   (15–67 inliers; SIFT found 0–22 raw matches), and every build poses all 124 keyframes in one
   reconstruction, the nine 1.6–1.8 % off in depth against their neighbours. LightGlue costs
   about 1 s per pair and CPU thread (COLMAP's CoreML provider cannot compile its dynamic shapes
   and was 6× slower), so it runs on these pairs only: 7 s on the office walk, nothing on a walk
   without a weak cut (livingroom). The first use downloads COLMAP's `sift-lightglue.onnx`
   (46 MB) into `~/.cache/colmap`; if that fails, the pairs keep SIFT's matches. `map.json`
   (`updates[].notes.weak_links`) records the cuts and the pairs. Photos keep SIFT: on a 12 MP
   photo's ~5000 SIFT keypoints LightGlue took 4.6 s per pair, and the learned features cost
   more than they gave on the photo sets (ALIKED+LightGlue: features and matching +28 s on
   office, +14 s on hallway, where 1 of 3 maps broke; LoMa-B: 8 min of extraction for the 13
   office photos).
4. **Poses (pycolmap 4.2):**
   * **New map:** global mapping (GLOMAP) runs first. If it places fewer than 60 % of the
     keyframes, incremental mapping runs instead. Rotation-dominant input goes to MapAnything
     multi-view poses followed by triangulation and bundle adjustment. Input counts as
     rotation-dominant when more than half of the verified pairs are panoramic, or when the
     baseline-to-depth ratio is below 0.02, or when the SfM points fix no metric scale (no
     keyframe has 50 well-triangulated points: a few photos panned from one spot, whose SfM
     units are arbitrary — the camera 8 cm from the last one placed 3 m away). MapAnything runs
     in chunks of 24, anchored on up to 4 already-posed keyframes, and is also the fallback when
     SfM fails.
   * **Weakly linked parts of a new map** (typical of a video walk through several rooms): a
     stretch that hangs on the rest by one weak link — a doorway crossed with a few dozen
     matches, or one keyframe shared with the rest — has no scale of its own for the global
     mapper, which returns it shrunk onto one point or at an arbitrary scale depending on its
     random start, and its orientation rests on that link alone. After SfM:
     1. An SfM pose needs 20 triangulated points. Keyframes with fewer are not accepted as posed
        (a stretch shrunk onto a point triangulates nothing); incremental mapping with the others
        fixed may register them again. After step 2, a pose that contradicts the keyframe's own
        verified matches (median symmetric epipolar distance above 0.25°; correct poses stay
        within about 0.1°) is not accepted either: on the 6-photo office map the global mapper
        settled one keyframe on 67 matches to one neighbour against 931 to two others. Nor is a
        pose whose SfM points the keyframe's own depth contradicts: its depth ratio over the
        model's metric scale lies outside 0.8–1.25 for photos (the depth alignment would leave
        it unfused, and no neighbour covers its view) or 0.5–2 for video (it would be left out).
        Two photos taken walking down a hallway, a forward motion the global mapper cannot
        resolve, were placed 1.3–3.9 m along it and 0.5–1.6 m apart in height from run to run,
        with ratios of 1.3–4. Such keyframes are joined like unplaced ones (step 3). So are
        photos that hang on the rest through one keyframe: at most two photos whose triangulated
        points that keyframe sees and no other (a link is 15 shared points). Nothing in the
        reconstruction fixes how far from that keyframe they are, and they are too few for step 2
        to judge by their depth. The two hallway photos are such a pair: over 100 seeds of the
        global mapper on one set of matches their depth ratios ranged over 0.015–400, and in 6
        seeds the depth check kept one or both placements (the first photo 0.13–0.92 m from the
        room's first photo instead of 1.46 m). Joined like unplaced ones, they were at 1.46 m in
        all 100. Keyframes of a video are not judged this way. The multi-view refinement itself
        can still go wrong there: in 1 of 10 full runs the global mapper left both photos
        without support, and their refined poses put the second 2.6 m off and the first 0.4 m
        too high, against their depth (both low confidence).
     2. Each keyframe's SfM scale is measured against its MoGe depth (median depth ratio at its
        triangulated points), and its tilt against its GeoCalib gravity (1–3° on correctly posed
        keyframes). Co-visible keyframes whose ratios agree within 15 % and gravity within 10°
        form blocks. A block that hangs on the rest by a bridge or through one viewpoint and
        disagrees with the largest block by more than 12 % in scale or 5° in gravity is scaled and
        levelled about the keyframe it hangs on (the one with the most verified matches to it);
        what hangs on it follows.
     3. Keyframes still unplaced that verified matches connect to the posed ones get MapAnything
        poses anchored on the posed keyframes next to them in capture order (video) or on the most
        similar ones (photos). A secondary reconstruction that shares 3 or more keyframes with the
        main one is merged through the similarity of the shared poses instead.
     4. Realigned and multi-view keyframes are refined with every verified match and the depth,
        the rest fixed, and realigned blocks are levelled with their gravity again (the weak link
        can pull their tilt away).
     5. Guards: a keyframe placed within a tenth of the median step of another keyframe's centre
        while looking elsewhere (optical axes more than 3° apart) is left out, unless its own
        points or matches support the pose or the pair's matches are a pure rotation; a re-placed
        keyframe whose gravity is still more than 25° off is left out too.

     Keyframes with no verified match to the rest are left out and listed. `map.json`
     (`updates[].notes.sfm_unsupported`, `sfm_join`) records what was re-placed and how.
   * **Update:** photos of the same size whose EXIF focal length matches an existing camera's
     prior (the same device and zoom) share that camera, and so its refined focal length;
     without EXIF, frames of the same size share it as before. Incremental mapping continues
     the stored model only (no further models), with the map's keyframes and cameras fixed and
     a fixed random seed, and the result is mapped back onto the map frame. A fixed keyframe
     that COLMAP dropped and registered again elsewhere (a weakly supported stored pose) is left
     out of that similarity and keeps its stored pose; a result that moved most of them is
     discarded. New cameras keep the focal length the extension refined. Keyframes it cannot
     place, whose depth contradicts the pose, or whose pose contradicts their verified matches
     (as above) are posed by anchored MapAnything and refined with the matches and depth.
   * **Maps with fewer than 3 keyframes:** SfM is re-run over all keyframes and aligned to the
     stored poses.
   * **Rebuilding a weakly posed map.** An update holds the stored keyframes fixed, so a pose SfM
     did not give would be frozen for good: the four window photos of `office_sequence` mapped
     first are posed by the multi-view fallback (no SfM scale, matches mostly on the trees behind
     the glass, see Known limitations), and the 4 + 4 + 5 map kept boxes up to 3 times longer
     near the camera, two extra objects and one window fewer than the one-update map.
     * *When.* A map of photos (every update) with at most 60 keyframes, one of which SfM did not
       pose (refined by feature matches, or holding fewer than 15 SfM points), is mapped again
       with the new photos, as one update of them all. Not when the last update already rebuilt
       it and left it rotation-dominant (a head turning in place: more photos from the same spot
       give no parallax, and it would be rebuilt every time at a cost growing with the map);
       `updates[].notes.restart_skipped` says why. A rebuild that would leave out any stored
       keyframe is abandoned in the same update and the map is extended instead
       (`notes.restart_abandoned`); one that succeeds is recorded in `notes.restarted`.
     * *What it keeps.* The stored keyframes are mapped from what the map holds of them: their
       images, depth and validity (pixels earlier updates retired stay retired, so an object
       they removed does not come back), intrinsics, gravity, descriptors and detections — no
       inference runs again — and the SfM database keeps their features and matches. Each keeps
       the update that added it: latest wins judges the rebuild's verdicts update by update (an
       object's strikes as those updates gave them), and the places of removed objects carry
       over.
     * *Ids.* An id the map published stays with its object. Each detection records the id it
       was first published with (`instances.json → first_id`, never rewritten), and a rebuilt
       object takes the ids of the stored detections it owns that were the first to carry
       them (an id's founding detection): the published ones before a candidate's, then the
       lowest. A published id whose founding detection the rebuild groups with an object that
       keeps another published id goes instead to a rebuilt object that holds none, has a
       compatible label and stands where the map last published it (their boxes overlap, or
       their centres lie within the attribution gate, max(5 cm, 3 % of the viewing distance)).
       Only when no such object exists does it resolve to the object that took its detection,
       provisionally (`objects.json → rebuild_merged`): each rebuild decides again, so the id
       returns to its founder when a later rebuild separates the objects; a map that stops
       being rebuilt keeps its last provisional merges. The map's other merges (`merged_into`)
       are permanent, and when the extension merges two objects the published id stays before a
       lower candidate's. A published id that no object takes is listed in the update's
       `objects.removed`, and one whose object is kept only as a candidate in
       `objects.unpublished`; none vanishes unreported. Objects first seen by the new photos are
       numbered on from the map's count, as an extension numbers them. So ids can differ from
       the one-update map's where an earlier update had published one (spec §2.3). In
       `office_sequence` split 4 + 4 + 5 the four window photos publish two windows, 9 and 22
       (the one-update map numbers the second 21). Before the geometric rule, the next update's
       rebuild could leave the second window's ids out of place for one update: in one run it
       grouped both windows into one object (22 resolved to 9 meanwhile); in another it grouped
       22's founding detection with window 9 and published the rest of the second window as 26.
       Both runs ended with 22 on the second window (26 resolving to 22) after the third update.
       With the rule, in three runs: 22 stayed on the second window after every update in one;
       in another the second update's rebuild kept the second window as a candidate only (22
       listed `unpublished`); in the third it mapped both windows as one object, so no object
       stood apart for 22 and it resolved to 9 provisionally. After the third update 22 was on
       the second window in all three.
     * *Cost.* The rebuild re-poses and re-fuses every keyframe: office 4 + 4 + 5, updates 2 and 3
       took 18 s and 17 s (extending: 17 s each); ainex 40 + 39, update 2 took 127 s
       (extending: 84 s; the whole sequence in one update: 158 s).
     Maps posed by SfM, videos and larger maps are extended.
   * If no new keyframe overlaps the map, the command exits 5 and the map is unchanged.
5. **Refinement.**
   1. Geometry is re-run for keyframes whose COLMAP focal length differs by more than 3 %.
      MoGe-2's network does not depend on the focal length, which enters only its
      post-processing (the point map's depth shift is solved for that focal length, then depth
      and intrinsics follow). The first pass therefore asks the server to keep the network
      output of each keyframe without EXIF (the point map before MoGe's output remap and the
      binary mask: 2.2 MiB at 768×432), and the re-run sends the pixels the first pass read and
      gets that output re-solved with the COLMAP focal length: the depth, validity and
      intrinsics a second forward pass gives, bit for bit, in 7 ms instead of 180 ms (on the
      office video, 106 keyframes in 1.3 s instead of 18–20 s). The server keeps at most 1 GiB
      of such outputs, oldest dropped first; it hands each out once, and drops those older than
      30 minutes at its next geometry request. A re-run that finds none runs the network.
   2. For a new map, the metric scale is the median ratio between MoGe depth and SfM depth, and
      the map frame is defined.
   3. Each keyframe's depth is aligned to the map. The scale comes from its SfM points, or
      densely from overlapping keyframes. A scale outside 0.5–2 rejects the keyframe. A scale
      outside 0.8–1.25 marks it low-confidence, and such keyframes are excluded from fusion.
      Dense alignment goes keyframe by keyframe, so its scale drifts along a long sequence, and
      one scale per keyframe cannot reconcile a near field (a counter top 0.3–0.7 m away) that
      disagrees with the neighbours' by 10–30 % while the far walls agree. The depth of all the
      map's keyframes is therefore adjusted together: each keyframe is paired with every
      keyframe (up to its 50 nearest by viewpoint) whose optical axis differs by less than 45°,
      sequence neighbours and loop closures alike. Each pair measures the median log ratio of
      the two depths on the surfaces both see, in both directions and in 6 bins of depth, and
      one correction per keyframe — a scale and a near/far tilt about its median depth,
      `log d' = log d + a + b (log d − log median)` — is solved from all of them at once
      (robust least squares; the tilt has a prior of 0 and is at most ±0.3, and no pixel's depth
      changes by more than a factor 1.5; sparse-scaled keyframes and the map's first keyframe
      hold the metric scale but are tilted like the others; low-confidence keyframes follow the
      others). The map's stored
      keyframes take part too, so a later update that closes a loop spreads the correction over
      the whole loop, as one update with the whole sequence would: their depth is rewritten and
      their objects move with them. `frames.json` records each keyframe's `depth_scale` (at its
      median depth) and `stats.depth_exponent` (1 + b).
   4. For a new map, the map is levelled with the floor plane.
6. **Integration.** The update applies latest wins, updates the objects (the fused surface
   tells pieces of one horizontal surface: it is fused only in the boxes where they ask for it),
   fuses the cloud once (Open3D TSDF; after the objects this update removed retired their
   pixels), gives the cloud's points their object ids, exports the scene and commits. Each
   keyframe is fused up to 2.5 times the map's median depth (at most 30 m) as it placed that
   depth before its near/far tilt: the tilt moves surfaces, not which of them the keyframe
   contributes (outdoors, where it deepens the far field by 15–35 %, a fixed cut dropped the
   views that make facades 10–20 m from the path reach 3 keyframes). Every step of the fusion is
   local — a voxel depends only on the keyframes, a surface point on its two voxels, the
   few-views and removed-place tests on the point — so a map of more than 300 000 voxel blocks
   (a street walk: millions) is fused in slabs of about as many blocks, one at a time, with the
   same points and the memory of a slab. The per-keyframe projections of the cloud's points
   (few views, attribution, removed places) cull the points in 25 cm cells outside the
   keyframe's frustum and run on up to 8 threads over parts of the map. Monocular depth is
   least reliable near the image border, and the TSDF (a 4 cm band) keeps each disagreeing
   placement of a surface as a layer of its own: on `ainex-captures` the keyframes that saw the
   wall beside door 1 in their outer 15 % placed it up to 6 % nearer or farther than those that
   saw it near their centre, and the wall came out 12–15 cm thick, doubled below the light
   switch. Before fusion, each keyframe's border (its outer 15 % on each side) therefore takes
   the depth of the keyframes that see the same surface near their centre: every 4th border
   pixel is projected into its 8 nearest keyframes by viewpoint, and where it lands in one's
   central part on the same surface (within 10 %), its cell of 4×4 pixels is scaled by their
   mean depth ratio. Border pixels no keyframe sees centrally keep their depth.
   Then the keyframes are made to agree everywhere, as multi-view stereo fuses its depth maps.
   Neighbouring keyframes still disagree by ~3 % (7–10 % outliers), and on `office.mp4` the
   desk legs came out fat, doubled or tripled and the monitors as slabs with offset copies.
   The copies come from keyframes of other passes 40–55° away: the sequence neighbours agree
   with each other. Every 2nd pixel is therefore projected into every keyframe whose optical
   axis lies within 60° (up to the 96 nearest by viewpoint). A keyframe that sees the same
   surface there (within 3 %) gives its depth along the pixel's ray, and the pixel's 2×2 cell
   takes the median of these depths and its own. A keyframe whose depth there lies more than 3 %
   beyond the pixel sees through it. Where at least 2 keyframes see through a pixel, more than
   see it, the cell is left out of the fusion: these free-space violations are the offset
   copies. A keyframe of an older update never counts as seeing through a newer one's pixel,
   because what it saw through may have been placed there since (latest wins). Most of the gain
   comes from leaving those pixels out (about 10 % of the office's pixels). The median alone
   merges the layers within 3 % (a third fewer cloud points). On the office walk, rendered from
   8 keyframes against their own depth, leg pixels within max(4 cm, 3 %) went from 0.73–0.75 to
   0.85, and the ring around them where the map stands in front of the keyframe went from
   0.39–0.42 to 0.27. For the monitors these went from 0.63 to 0.79 and from 0.40 to 0.22. The
   price is up to 1 point more of pixels where the map lies behind the keyframe (the nearer of
   two disagreeing placements is the one left out). It takes about 2 s for 124 keyframes, at
   every fusion setup, and the fusion of fewer points saves about as much. Both the consensus
   and the border correction change the depth in memory only: the stored depth is unchanged.

### Ownership

import-linter contracts in `pyproject.toml` and `tests/unit/test_ownership.py` enforce ownership
at the Python-module level:

* `reconstruction` owns depth, intrinsics, gravity, point generation and fusion. It never imports
  segmentation or mapping.
* `segmentation` owns detection, exclusive masks, lifting, OBB fitting, colours, the catalogue,
  the artefacts, and the derivation of every emitted cloud (`segmentation/cloud.py`).
* `mapping` owns inputs, SfM, the map frame, identity and the store. It reaches the server only
  through `reconstruction` and `segmentation`.
* `viewer` only serves data.
* `web` (`server.sh`) owns the HTTP layer and the job runner. It runs the commands' entry points
  as subprocesses and never imports torch, Open3D or the inference server's internals.
* `server` owns the models and nothing else.
* The layers run `cli` | `web` (independent siblings; `web` is the coming web service) >
  `commands` > `tools` > `viewer` > `mapping` > `segmentation` > `reconstruction` > `client` >
  `schema` > `core`.
* `commands/spec.py` is the commands' single source of truth (spec §2.6): every mode, option
  (flag, kind, choices, default, accepted files, bounds, help, applicability), validation rule,
  output and timing stage of `reconstruct.sh`, `mapper.sh update` / `locate`, `segment.sh -i` /
  `-m` and `view.sh -i` / `-m` is declared there once. Each command builds its argparse parser
  (`spec.build_parser`) and runs its checks (`spec.validate`: each rule's pure check, then its
  preparation such as creating `-d`, in order, before any work) from it. For the web service,
  `spec.parse` turns API parameters into the command's arguments through the same parser (an
  argument error names its parameters), `spec.dry_run` reports argparse's and every rule's
  problems per parameter without touching the filesystem, and
  `spec.describe()` exports everything as JSON-serialisable data. Stage names are
  `core.timing.Stage`; shared defaults and input suffixes are in `core/constants.py`.
* The web service runs each job as a subprocess (`python -m oh_my_slam.cli.<command>`) with
  `OH_MY_SLAM_PROGRESS` set, never in its own process.

## Benchmark evaluator

```sh
uv run python -m oh_my_slam.tools.evaluate [--out DIR] [--targets PATH] [--baseline PATH] \
                                           [--street2 PATH] [--set-baseline]
uv run python -m oh_my_slam.tools.evaluate --rejudge RESULT_DIR [--targets PATH] [--baseline PATH]
```

A single command benchmarks every entry point on `examples/`, strictly one command at a time
(spec §5):

* the server's cold start and resident memory (it stops and restarts the server)
* `reconstruct.sh` (JSON and PLY), `segment.sh -i -d` and `view.sh -i` on `restaurant.jpg`
* `segment.sh -i` on each of the 79 `ainex-captures` frames
* `mapper.sh update` on the sequence, once in one update and once split across 3 updates; between
  the split map's first and second update, `mapper.sh locate` of the second update's captures
  (held out of the map so far)
* `segment.sh -m -d` and `view.sh -m` on the one-update map, and `segment.sh -m -f ply -o` on the
  split map
* `mapper.sh locate` on the one-update map (the reference map): `-t single` JSON, `-t full` JSON
  and `-f ply -o`; the map folder, hidden entries (`.staging/`, `.lock`) included, must not change
* `mapper.sh update` on `office_sequence` (13 images; a cup on the window sill is gone in the last
  ones): the whole sequence in one update, and split as its annotation says (4+4+5 and 6+7)
* `mapper.sh update` on `street2.mp4` (150 s of street video) at the default `-fps`. It lives
  outside the repository, by default in `~/oh-my-slam-data/loop/inputs/street2.mp4`; `--street2`
  names another place. Without it the street2 metric fails and says where it looked.
* `server.sh` over a scratch workspace with copies of the reference inputs and map (see below)

### The `ainex-captures` file names

The spec no longer spells out the capture names, which encode the **commanded** head motion as
`NNN_<motion>_<tilt>.jpg` (`tools/evaluate/names.py`; a name that does not follow it is an error):

* `NNN` is the capture order.
* `<motion>` is the commanded yaw relative to frame 001, positive to the left: `bootstrap` (0°),
  `bootstrap_side1` / `bootstrap_side2` (0°, after a small sideways step), `bootstrap_leftYYY` and
  `bootstrap_leftYYY_side` (+YYY°), `left_YYY` (+YYY°), `right_to_YYY` (+YYY°, turning back
  towards 0°) and `right_YYY` (−YYY°).
* `<tilt>` is `level`, or `up` / `down` relative to the `level` frame of the same motion.

The same-heading pairs are 001/053 (053 returns to 001's heading) and 026/078 (`left_210` meets
`right_150`, which closes the 360° loop).

### Metrics

| Group | Measures |
|---|---|
| `perf.*` | Per image and per mapping update: end-to-end wall time, client and server peak memory, and `view.sh` time to the rendered page; for a map built in several updates also its slowest update (`per_update_wall_s`). Groups: the single-image commands, one `ainex-captures` frame (median), the two `ainex` maps, `segment.sh -m`, `view.sh -m`, `mapper.sh locate`, the office maps (one update, and the slowest update of the splits) and street2. |
| `perf.<group>.stage.<stage>.*` | Per stage, every stage the commands record: seconds (`.s`: median over the frames of a per-frame group, else the slowest run, so per image and per mapping update), client (`.client_peak_mb`) and server (`.server_peak_gb`) peak memory while it ran. |
| `pose.*` | Yaw against the headings in the capture names, pitch direction of `up`/`down` frames, registered fraction, and same-heading pairs, for both `ainex` maps; `pose.locate.*` the same for the held-out captures `mapper.sh locate` placed (located fraction, yaw error relative to the map's capture 001; the detail compares each with the pose the map gives it once added); `pose.street2.registered_fraction` the share of the video's sampled frames the map registered. |
| `map.*` | Frame agreement of the same-heading pairs and of every overlapping keyframe pair (optical axes < 45° apart, any distance in capture order: median and p90 over the pairs, share of pairs above 10 %, worst pair; the detail splits sequence neighbours, ≤ 10 keyframes apart, from loop closures); and the stability of ids, labels and OBBs between the one-update and the split map. Ids and boxes are compared on a label-aware pairing, labels on a label-blind one. `matched_fraction` is the share of the one-update map's objects the split map has: the split map may keep an object an earlier update published that no later image contradicts (`mapper.md`), so its extra objects are listed in the detail (`extra_published`, id and label) and not counted against it. As `mapper.md` allows, an id may differ from the one-update map's where an earlier update of the split map had published it: `id_agreement` counts such a pair as agreeing (its detail keeps the strict share, `same_id`). |
| `map_update.*` | Map update on `office_sequence`. For the one-update map and for each split (`split_4_4_5`, `split_6_7`) after its last update: the share of the annotated absent objects (the cup) the map no longer has (`absent_fraction`), and `hole_fraction`, the share of the cells of the cup's annotated place where the map shows no surface (the map cloud projected with the map's own poses into the images that showed the cup: a cell with no point, or whose nearest point lies more than 25 % behind the surface around it, is a hole). The control (`before_present_fraction`): the split map after an update of exactly the images that show the cup has it. Per split: the ids and labels of the unchanged objects from the first update to the last, after aligning the two updates by their common captures (`stability.label_agreement`, `stability.id_agreement`; a rebuild may re-gauge the frame and OBBs are refined, so box figures are detail only); `ids_persistent_fraction`, every id an update published for an unchanged object is, in every later update, still an object of a compatible label at the same place; and the split against the one-update map (`vs_one_update.*`, ids as in `map.*`). A remnant is a map object with a compatible label whose box, projected with the map's own poses into the images that showed the object, covers its annotated region. What changed, and the splits, are annotated in `examples/ground_truth/office_sequence.json` (`kind: "map_update"`). |
| `seg.*` | Detections per frame. |
| `seg.map_consistency.*` | Per-frame detections compared with the map's objects. The map is built from the same detector, so these measure consistency, not accuracy. |
| `contract.*` | Colour contract, OpenLABEL validity (and `mapper.sh locate`'s located cameras marked as such), stdout purity (including `server.sh`, and `locate -f ply`'s one pose line per input image), artefacts, same objects (`locate -t full` gives the map as `update -t full` does), and read-only maps. The colour contract covers the viewer's OBBs and its `color=segment` cloud (`/api/cloud`). |
| `server_sh.*` | `server.sh` (http_server.md "Evaluation"); see below. |
| `gt.*` | Accuracy against ground truth, when annotations exist. |

### `server.sh`

The evaluator starts `server.sh --data <out>/server_sh/ws --no-browser` (port 0), with copies of
the reference inputs and of the one-update `ainex` map in that workspace.

* **Performance:** start-up time (to the `listening on` line), resident memory of its process tree
  once listening, time from opening the web application to `body[data-ready=true]`, latency of
  read-only requests (median and p95 when idle; p95 of the requests made while a job runs), and the
  median overhead of a job over the same command run from the shell.
* **Parity:** the operations come from the service's own `/api/openapi.json` (each job
  operation's `x-oms` entry, the commands' `commands.spec.describe()`), so a new mode or option is covered without changing the evaluator.
  Each operation gets a default case on the reference inputs and one case per non-default choice,
  per artefact folder and per point-cloud attribute. Each case runs the command from the shell
  twice and then as a job, and the job's result and every file of its artefact folder must be
  byte-identical to the command's; for `view.sh`, what its viewer serves (`VIEWER_ROUTES`). The
  models are not bit-reproducible from one request to the next, so the service and the shell
  runs share an **inference proxy** (`tools/evaluate/proxy.py`): it forwards the first request of
  a kind to the inference server and answers identical requests (same route, fields and input
  file contents) with the recorded response. A difference is a mismatch, unless the command's
  own two runs differ too (`unverifiable`).
* **UI:** the web application's browser tests (`tests/browser/test_webapp*.py`, `-m browser`,
  with the stub inference server in an isolated runtime folder), and the vendored axe-core
  (WCAG 2.0/2.1 A and AA) on the live service's pages over the reference data.

### Targets, baseline and results

Per-stage memory comes from two sources. Each command records its stages (`core.timing`), and
the peak resident set of its own process during each stage, sampled every 50 ms. It writes these
to `OH_MY_SLAM_TIMINGS`, and a map update also keeps them in `map.json → updates[].timings`. The
server reports no memory of its own, so the evaluator samples two figures every 0.2 s: the
command's process tree, which includes COLMAP, and the server's physical footprint. It
attributes each sample to the stage whose time window it falls in.

Targets are data in `examples/targets.json`, and every metric has one. Each target is an
`op`/`value` pair with regression tolerances, and a target can be edited without changing code.
A key with `*` is a pattern that targets every metric it matches without a target of its own (the
most specific pattern wins): the per-stage metrics and the office splits, whose names are data,
are targeted that way. The file's `rationale` explains each group of targets. Each metric's
`measured` value is the reference run the targets were derived from, and the evaluator ignores
it. Ground-truth files dropped into `examples/ground_truth/` are picked up without code changes
(see its `README.md`). `--rejudge RESULT_DIR` runs nothing: it judges a stored run again with the
current targets and baseline and rewrites its `result.json` and `summary.md`.

Each run is compared with the stored baseline `~/oh-my-slam-data/evaluations/baseline.json`,
and regressions are flagged. Without a baseline the report says `baseline missing — not
compared`, and `summary.regressions` in `result.json` is `null`. For each failed or regressed
metric, the summary's `why` column gives the error, the value against the target, or the change
from the baseline and the tolerance it exceeded.

`--set-baseline` stores the run, once it is judged, as the baseline later runs are compared with.

Results go outside the repository, to `~/oh-my-slam-data/evaluations/<UTC>/` by default:

* `result.json` (machine-readable)
* `summary.md` (human-readable)
* `runs/` (stdout, stderr and timings per command)
* `outputs/`
* `maps/`
* `server_sh/` (the service's workspace, the shell side of the parity cases, the browser tests'
  report)

The command prints the path of `summary.md` on stdout. It exits 0 when every metric passes, 1
when one fails, and 2 on a usage error. It needs the model weights. It also needs Microsoft Edge
or Google Chrome for Playwright's page timing. Run it on an otherwise idle machine, because
concurrent GPU work invalidates timings. `OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m eval`
runs it end to end as a test.

## Known limitations

* **Files opened from disk in the future 3D scene viewer are not budgeted yet.** `static/lib/ply.js`
  draws every point of a PLY read in the browser. Applying the §2.5 display budget there would
  mean porting the selection, which segmentation's derivation owns, to JavaScript. This will be
  decided when `server.sh`'s scene viewer is built.
* **Labels come from an open-vocabulary detector (YOLOE with text prompts) and inherit its
  confusions.** On the rendered `ainex-captures` kitchen, prompting each label alone on the
  mislabelled instances shows that the detector itself prefers the wrong label. A kettle scores
  0.63 as a teapot and 0.10 as a kettle. The counter scores 0.57 as a desk and 0.03 as a
  counter. The light switch scores 0.57 as a power outlet and 0.32 as a light switch. The toaster
  scores 0.77 as a tissue box and 0.05 as a toaster, and the coffee maker scores 0.64 as a water
  dispenser and 0.17 as a coffee maker. A horse picture on a book cover is detected as a person.
  The vocabulary, prompt form and NMS settings do not cause these errors, so they are not
  patched per scene. One gap was in the vocabulary: common produce was missing, so a tomato or a
  watermelon took the nearest label (apple). Tomato, watermelon and a few other LVIS produce
  nouns are now included.
* **Label flicker.** When the detector names one object differently from keyframe to keyframe,
  the map keeps one object with the best-supported label and lists the others in `detected_as`.
* **Monocular depth of small, distant objects** varies between keyframes, so their map boxes
  are only as consistent as the agreeing detections (see Update semantics).
* **Boxes cover the observed surface, not the whole object.** An OBB is fitted to the points the
  keyframes saw. An object seen only from the front has the depth of its visible surface: in
  the `ainex-captures` map the refrigerator against the wall is about 0.8 m wide and 1.65 m
  tall but only 0.1–0.15 m deep, because only its front was seen. The fit does not invent the hidden part from a class-typical size,
  which would be wrong for any object that is not typical. The only extension is floor
  grounding, where the visible bottom floats just above the detected floor (see Coordinate
  conventions).
* **Monocular depth disagrees locally.** Before the global depth adjustment (Refinement), the
  keyframes that close the 360° loop of `ainex-captures` disagreed with those that opened it by
  15–21 %, and the near field of a keyframe (the kitchen island 0.3–0.7 m below the camera, seen
  at a grazing angle) by 10–30 % with its neighbours' while their far walls agreed. A scale and
  a near/far tilt per keyframe remove most of it: over the ~775 overlapping keyframe pairs of
  that map (optical axes < 45° apart, sequence neighbours and loop closures alike), the median
  pair disagrees by about 1.8 %, 90 % of the pairs by less than 3.5 %, and the worst pair by
  9–11 % (with one scale per keyframe: 1.9 %, 5.5 % and 21 %, keyframes two apart). What remains
  varies across an image in a way a tilt in depth does not model: the worst pairs are 40–45°
  apart and overlap only near their image borders, where monocular depth is least reliable (the
  bootstrap frame at left 15° against left 60°, left 60° against the return at right-to 10°).
  The tilt also means a keyframe's depth is not one scale of MoGe's: `frames.json` records
  `depth_scale` at its median depth and `stats.depth_exponent`.
* **Large horizontal surfaces at floor height can stay fragmented.** Pieces of a surface seen
  from different keyframes merge when they share surface or when a horizontal patch of the
  fused surface joins them (see Update semantics). At floor height only the first test applies,
  since the floor joins anything: two pieces of one rug that neither overlap nor touch stay two
  objects.
* **Scenery behind glass.** Monocular depth places what is seen through a window (trees, a
  street) on the pane. Where most feature matches lie there, poses that lift keypoints with
  their depth are pulled towards a camera that did not move: the first four photos of
  `office_sequence` mapped on their own have about 90 % of their matches on the trees behind
  the window, and the multi-view refinement leaves their cameras 1-6 cm apart where the
  13-photo map measures 8-25 cm (SfM of the four photos alone is no better: rotations 2.5-4.7°
  off, 15-37 points per photo). Objects near the camera, the cup and the wallet on the
  windowsill, are then placed 5-15 cm apart by each photo, and their boxes are that much
  longer. A later update rebuilds such a map with its photos (Mapping, step 4), so the whole
  sequence split as 4 + 4 + 5 or 6 + 7 photos ends with the one-update map's objects, labels
  and boxes (one window keeps a different id, see there). In a map of the
  whole sequence the global mapper's result varies from run to run for the same reason: in some
  runs the depth check (step 1 of *Weakly linked parts*) rejects photos whose SfM points are
  mostly those trees (depth ratios of 0.25-0.3 and 1.8-2), and their multi-view poses leave them
  low confidence.
* **Video through several rooms.** Where a walk crosses a doorway in a second or two,
  consecutive keyframes share only a few matches, and the global mapper can leave the stretch
  behind the doorway at any scale or tilt, or shrink it onto one point (see Poses). The mapper
  corrects the scale with the depth and the tilt with gravity, but the stretch's heading and
  position still rest on that one link, or on multi-view poses anchored on its capture-order
  neighbours, so they can be a few degrees and decimetres off. LightGlue on the weak links
  (Mapping, step 3) gives such a cut more matches, but it cannot add keypoints: a keyframe of a
  blank surface (the office walk's `f000101`, 67 SIFT keypoints) rests on four pairs of about
  20 inliers, and in 1 of 5 builds its SfM pose was 0.4 m off its neighbours. Keyframes whose
  gravity still disagrees by more than 25° afterwards are left out: on the user's
  `livingroom.mp4` at `-fps 1`, the dim corridor at 77–81 s. The global mapper starts from
  random positions, so two runs on one video can differ in such stretches.

## Development

```sh
uv run pytest -m "not models and not browser and not eval" -q         # offline suite (stub models)
uv run pytest --cov=oh_my_slam -m "not models and not browser and not eval"
OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m models                 # real models; server running
uv run pytest -m browser                                              # viewer in Edge/Chrome (Playwright)
uv run ruff check . && uv run mypy src && uv run lint-imports         # lint, types, ownership
```

The offline suite runs the shipped server process with deterministic stub models
(`tests/fakes/stub_models.py`, started by `tests/fakes/stub_server.py`). The mapping end-to-end tests are
skipped when `colmap` is not installed. The `eval` marker selects the evaluator's end-to-end run
(see [Benchmark evaluator](#benchmark-evaluator)).

Environment variables:

| Variable | Effect |
|---|---|
| `OH_MY_SLAM_TIMINGS=path.json` | Writes the full per-stage timing record of `reconstruct.sh`, `segment.sh` or `mapper.sh update` / `locate` there: stage times, per-stage peak resident set and stage time windows (the evaluator's per-stage figures, spec §5). Each map update also keeps its record in `map.json → updates[].timings`. |
| `OH_MY_SLAM_PROGRESS=path` | Appends one JSON line per live progress event of `reconstruct.sh`, `segment.sh` or `mapper.sh update` / `locate` to that path (`/dev/fd/<n>` reaches a pipe): `begin`, `stage_start` / `stage_end` (stage names of `core.timing.Stage`), `count` (sizes such as `keyframes_sampled`), `part`, `progress` (`done` of `total` items of the running stage) and `finish` (with the outcome: `ok`, `exit_code` and its `code`). stdout and the stderr text are unchanged. This is how the web service follows the jobs it runs as subprocesses. |
| `OH_MY_SLAM_RUNTIME_DIR` | Replaces `~/Library/Caches/oh-my-slam` (socket, log, state, scratch): the test suite runs its stub server there, beside a running real one. |
| `OH_MY_SLAM_TEST_REAL_SERVER=1` | Lets the `models` and `eval` tests use the running real server. |

## Troubleshooting

* **`inference server models failed to load (…)` (exit 3).** The server runs but a model did
  not load; the message names it and the log. Fix the cause, then restart the server with
  `./start_inference_server.sh --stop && ./start_inference_server.sh`.
* **`inference server is not running — start it with ./start_inference_server.sh` (exit 3).**
  Start the server. `--status` shows which models loaded, and the log is
  `~/Library/Caches/oh-my-slam/server.log`.
* **Exit 5 on an update.** The new images share no verified feature matches with the map.
* **Exit 6.** Another `mapper.sh update` is running on the same map.
* **`COLMAP CLI … and pycolmap … must both be 4.2.x`.** Run `brew upgrade colmap`.
