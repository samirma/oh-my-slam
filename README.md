# oh-my-slam

Monocular RGB mapping on an Apple-silicon Mac. It reconstructs a single image as a metric point
cloud, builds and updates persistent maps from photos or video, and describes the scene as
labelled objects with oriented bounding boxes (OBBs) in ASAM OpenLABEL 1.0.0. Input is RGB only:
depth, intrinsics and gravity come from models. `high_level_spec.md` holds the requirements;
this file records the implementation decisions (defaults, conventions, exit codes, the
OpenLABEL mapping and the colour palette).

| Entry point | What it does |
|---|---|
| `start_inference_server.sh [--foreground\|--status\|--stop] [--timeout S]` | Starts the resident model server, or stops or queries it. |
| `reconstruct.sh -i IMAGE [-f json\|ply] [-o FILE] [-p ATTRS]` | One image → OpenLABEL scene (default) or point cloud, in the camera frame. |
| `mapper.sh update -a IMAGES\|FOLDERS\|VIDEO -m MAP [-f json\|ply] [-o FILE] [-p ATTRS] -t full\|single [-fps N]` | Creates or extends a persistent map. |
| `segment.sh -i IMAGE [-f json\|ply] [-o FILE] [-d DIR] [-p ATTRS] [--min-score S]` | Objects of one image: OBBs, colours, and with `-d` five artefact files. |
| `segment.sh -m MAP [-f json\|ply] [-o FILE] [-d DIR] [-p ATTRS]` | The persistent objects of a map, read-only and without the server. |
| `view.sh -i IMAGE \| -m MAP [--port N] [--no-browser]` | Local browser viewer. `-m` needs no server. |

## Install

```sh
brew install colmap            # COLMAP 4.2.x CLI (feature extraction and matching)
uv sync                        # Python 3.12 environment in .venv, dev tools included
./scripts/install_tools.sh     # checks that `colmap` is installed and is 4.2.x
./start_inference_server.sh    # the first start downloads the model weights
```

`scripts/install_tools.sh` installs nothing. It exits 1 with a hint if `colmap` is missing or not
4.2.x, because the `pycolmap` wheel in `.venv` is 4.2.x and both must match. Each entry script is
a thin wrapper that runs `.venv/bin/python -m oh_my_slam.cli.<command>` and exits 2 if `.venv` is
missing. The scripts never call each other.

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
./mapper.sh update -a walk.mp4 -m maps/home -t full -fps 2 > map.json
./mapper.sh update -a more_photos/ -m maps/home -t single -f ply -o new_part.ply
./segment.sh -m maps/home -d out_map/
./view.sh -m maps/home
./start_inference_server.sh --stop
```

## Commands

### `start_inference_server.sh`

With no option, the command starts the server in the background and waits until the models are
loaded. The default timeout is `--timeout 1200` seconds. If a server is already running, the
command reports it and exits 0.

| Option | Effect |
|---|---|
| `--foreground` | Runs the server in this terminal. Ctrl-C stops it. |
| `--status` | Prints `/health` as JSON on stdout: status, device, precision, and each model's load state. Exits 3 if the server is not running. |
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

`mapper.sh update -a INPUTS… -m MAP [-f json|ply] [-o FILE] [-p ATTRS] -t full|single [-fps N]`

* **`-a`** takes image files, folders of images, or exactly one video:
  * Images keep the order given; the files in a folder are sorted by name.
  * Hidden and non-image files in a folder are skipped.
  * Image types: jpg, jpeg, png, bmp, tif, tiff, webp, heic, heif.
  * Video types: mp4, mov, m4v, avi, mkv, webm.
* **`-m`** is the map folder:
  * A folder that is missing, or that contains only hidden entries, becomes a new map.
  * A map is extended.
  * Any other folder is refused and left untouched (exit 4).
* **`-t`** is required:
  * `full` returns the whole map: every exported object and every keyframe pose. With `-f ply`
    it returns the whole map cloud.
  * `single` returns only the keyframes added by this update and the objects they observe. With
    `-f ply` it returns the points those keyframes see.
* **`-fps`** applies to video only (default `2`). The command keeps the sharpest frame in each
  1/fps time slot. For images, `-fps` is ignored with a warning.
* **`-p`** requires `-f ply`. Pixel-level keys are refused, because a map's points are already
  3D.

The result is always in map coordinates. Every option is checked before the server is contacted.

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
* **`--min-score S`** applies to `-i` only (default `0.5`). Allowed values are `[0.25, 1]`.

`--min-score` only adds or removes objects. The objects kept at two thresholds have the same id,
colour, mask, points and box, for these reasons:

* The detector is always asked for every detection scoring above the fixed floor of 0.25.
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

`view.sh -i IMAGE | -m MAP [--port N] [--no-browser]`

The viewer binds `127.0.0.1` on a free port (port 0) unless `--port` is given. It opens the
default browser unless `--no-browser` is given, and serves until Ctrl-C. Nothing is written to
stdout. Once the server accepts connections, stderr carries exactly one line of this form:

```
view.sh: listening on http://127.0.0.1:<port>/
```

The two modes differ:

* **`-i`** reconstructs and segments the image once, through the inference server. It shows the
  cloud, the segmented image, the catalogue, the labelled OBBs and the camera at its estimated
  pose. The display rotates the camera frame so that the estimated up direction is +z.
* **`-m`** opens the map read-only, without the server. It shows the map cloud, every keyframe
  camera and the labelled OBBs.

The page has four tabs:

* **Controls**
  * *Layers* has one independent switch each for the point cloud, the segmentation (object
    colours), the camera poses, the labels and the oriented boxes.
  * *Point cloud* has live controls for the point-cloud attributes that affect the display. For
    an image these are `color`, `stride`, `min-depth`, `max-depth`, `edge`, `voxel` and
    `normals`. For a map they are `color`, `voxel` and `normals`. The panel also shows the
    equivalent `-p …` string and a *Defaults* button.
  * *Display* sets the point size, the normals shading, the maximum number of labels and the
    background.
* **Catalogue** lists the objects, with a label filter.
* **Cameras** lists every displayed camera's centre (x, y, z in metres, in the scene frame), with
  a *Go to* button that moves the viewpoint to that camera, looking where it looked. `[` and `]`
  step through the cameras.
* **Image** shows the segmented image (`-i` only).

Other keys: `R` resets the view, and `Esc` clears the selection.

The viewer contains no geometry, segmentation or colour logic of its own:

* Every displayed cloud is derived on request from data already in memory, by the same code
  that writes PLY files. A control therefore never re-runs inference.
* Clouds above 3,000,000 points are thinned for display only, and the page says so.
* `encoding` and `label` concern PLY files only and have no control.

The page sets `<body data-rendered="true">` after its first frame with the cloud has rendered.
The evaluator waits for this attribute.

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
| `normals` | `on`, `off` | `off` | image, map | Add `nx ny nz`. For an image they come from the depth grid; for a map they are oriented towards the keyframe cameras. |
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
  `FILE`, creating parent folders, and leaves stdout empty. `-o` naming a folder is a usage
  error.
* **stderr** gets everything human-facing. Log lines start with `[oh-my-slam]`, and errors look
  like `<command>: error: …`.
* **Timing summary.** `reconstruct.sh`, `segment.sh` and `mapper.sh update` each log a
  one-line `timings:` summary.
* **stdout of the other commands.** `view.sh` writes nothing to stdout. `--status` writes the
  health JSON.

| Exit | Meaning |
|---|---|
| 0 | Success. A consumer closing the pipe early (`\| head`) also exits 0. |
| 1 | Internal error. Also used when the server stays busy after retries, when inference fails, or when COLMAP is missing or the wrong version. |
| 2 | Usage or input error: bad option, bad `-p`, missing or unsupported input file, `.venv` missing. |
| 3 | The inference server is not running. The message says to run `./start_inference_server.sh`. |
| 4 | `-m` folder is not empty and not a map (`mapper.sh`), or is not a map (`segment.sh -m`, `view.sh -m`). |
| 5 | Nothing could be registered, for example because the new images do not overlap the map. The map is unchanged. |
| 6 | Another `mapper.sh update` holds the map lock. |
| 130 | Interrupted (Ctrl-C). `view.sh` treats Ctrl-C as its normal stop and exits 0. |

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
| `object_data.vec` | `color` `[r, g, b]` |
| `object_data.boolean` | `confirmed` |
| `frame_intervals` | the frames that detected the object |

For a map object, `score` is the mean of its three best detection scores.

## Colour contract and palette

An object's colour is a pure function of its id (`segmentation/colors.py`). The same sRGB triple
appears in all of these:

* the JSON `color` / `color_hex`
* the mask pixels of `segmented.png`
* `catalog.csv` and `catalog.md`
* `segments.ply`, and every PLY or viewer cloud with `color=segment`
* the OBB rendered by `view.sh`

Masks are opaque, and each pixel and point belongs to at most one object.

Ids 1–19 use this palette, which is Sasha Trubetskoy's list of distinct colours without grey,
white and black:

| id | colour | id | colour | id | colour | id | colour |
|---:|---|---:|---|---:|---|---:|---|
| 1 | `#e6194b` | 6 | `#911eb4` | 11 | `#469990` | 16 | `#aaffc3` |
| 2 | `#3cb44b` | 7 | `#42d4f4` | 12 | `#dcbeff` | 17 | `#808000` |
| 3 | `#ffe119` | 8 | `#f032e6` | 13 | `#9a6324` | 18 | `#ffd8b1` |
| 4 | `#4363d8` | 9 | `#bfef45` | 14 | `#fffac8` | 19 | `#000075` |
| 5 | `#f58231` | 10 | `#fabed4` | 15 | `#800000` | | |

Higher ids cycle:

* Cycle `k = (id − 1) // 19` rotates the hue of the same 19 colours by `k × 0.381966` of a turn in
  HLS, keeping lightness and saturation.
* A rotated colour that collides with an earlier id after 8-bit rounding is nudged in steps of
  0.002 turn until it is unused. Every id therefore gets a distinct colour.

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
| `scene.json` | The cached full scene. |
| `.lock`, `.staging/` | The update lock, and the staging area of an update in progress. |

An update writes everything into `.staging/`, then records the list of staged files (the commit
point), moves them into place, and writes `map.json` last. An update killed before the commit
point leaves the map untouched. One killed after it is completed by the next update. Readers
(`segment.sh -m`, `view.sh -m`) never lock or write. Do not edit files in `.staging/`.

### Update semantics

* **One update is one observation.** The keyframes of one update are processed without regard to
  their order:
  * Object association groups all of the update's instances at once, strongest agreement first.
  * Latest wins and the cloud's colours and object ids do not depend on keyframe order either.
  * Order only affects keyframe names and the numbering of new objects (by their earliest
    keyframe).
* **Later wins.** "Latest" is the order of addition. A later update invalidates, per pixel, the
  parts of older keyframes that it contradicts: free space seen behind an old point, or a new
  surface in front of an old ray. It also removes, or gives a strike to, objects that it sees
  through. Keyframes of the same update never invalidate each other. Capture timestamps are
  never read.
* **Persistent identity.** Each object keeps one id and one colour for the life of the map, and
  its OBB is refitted from all accumulated evidence:
  * New ids come from a counter and are never reused.
  * A merge keeps the lower id.
  * An object is exported once it is confirmed, meaning it was detected in at least min(3, V)
    keyframes, where V is the number of keyframes that could see it.
  * Unconfirmed objects are kept, so that a later update can still confirm them.
* **Geometry.** The map's geometry is a point cloud: the surface of a TSDF fusion of the aligned
  depth maps. Each point takes its colour and object id from the latest update that sees it. No
  mesh is produced.

## How it works

### Inference server

`src/oh_my_slam/server` is a FastAPI app on a Unix socket, `~/Library/Caches/oh-my-slam/srv.sock`
(mode 0600). It falls back to a short `/tmp` path if the socket path is too long. `/health`
answers immediately, even while the models load. It loads four models:

| Model | Role |
|---|---|
| MoGe-2 ViT-L normal | Metric point map, depth and validity, intrinsics, plus a DINOv2 class-token descriptor for retrieval. |
| GeoCalib (pinhole) | Gravity direction with uncertainty. |
| YOLOE-26x-seg with the MobileCLIP2-B text encoder | Open-vocabulary instance masks over `segmentation/data/default_labels.txt`, a curated list of LVIS and COCO nouns. |
| MapAnything (Apache-2.0 checkpoint) | Metric multi-view poses, used as a fallback. |

The device is MPS when available (`OH_MY_SLAM_DEVICE=cpu|mps` overrides it), with a per-process
MPS memory cap of 70 %. MPS is not thread-safe, so every model call runs on one GPU worker
thread. The queue holds 8 jobs; beyond that the server answers HTTP 503 and the client retries.
torch lives only in the server process. The commands never import it, because loading torch and
Open3D in one process aborts on a duplicate libomp.

### Single image

These steps serve `reconstruct.sh`, `segment.sh -i` and `view.sh -i`:

1. **Intrinsics:** EXIF focal length, else MoGe's estimate.
2. **Depth:** MoGe-2 metric depth on a grid whose long side is at most 1024 px. Point colours
   are the resized pixels.
3. **Gravity:** GeoCalib, refined by a RANSAC floor plane within 5°.
4. **Detection:** YOLOE detections above the 0.25 floor. Background labels (wall, floor,
   ceiling, …) are prompted but never reported. Masks under 64 px are dropped, and duplicates
   across labels are removed (mask IoU > 0.7, higher priority wins).
5. **Exclusive masks and lifting:** the claim order above gives every pixel at most one owner,
   and the masks are lifted to 3D without depth-edge pixels.
6. **OBBs:** each object gets an upright OBB, then ids and colours are assigned.

Geometry and detection requests run concurrently on two connections.

### Mapping (`mapper.sh update`)

1. **Lock and stage.** Resolve the inputs into keyframes.
2. **Per-keyframe inference.** Each keyframe gets depth (768 px grid), gravity, a descriptor and
   detections at the default threshold. Two keyframes are processed at a time.
3. **Features and matching.** The Homebrew `colmap` CLI extracts SIFT features by default.
   ALIKED + LightGlue (ONNX/CoreML, Homebrew build only) is used with
   `OH_MY_SLAM_FEATURES=aliked`. Pairs are chosen as follows:
   * Photos: every pair up to 200 images.
   * Otherwise: sequential neighbours plus descriptor retrieval, with loop-closure candidates
     for video.
   * Updates: new keyframes are matched against the whole map up to 150 keyframes, and against
     retrieved keyframes beyond that.
4. **Poses (pycolmap 4.2):**
   * **New map:** global mapping (GLOMAP) runs first. If it places fewer than 60 % of the
     keyframes, incremental mapping runs instead. Rotation-dominant input goes to MapAnything
     multi-view poses followed by triangulation and bundle adjustment. Input counts as
     rotation-dominant when more than half of the verified pairs are panoramic, or when the
     baseline-to-depth ratio is below 0.02. MapAnything runs in chunks of 24, anchored on up to
     4 already-posed keyframes, and is also the fallback when SfM fails.
   * **Update:** incremental mapping runs with the map's keyframes fixed, and the result is
     mapped back onto the map frame. A result that moved the fixed keyframes is discarded.
     Keyframes it cannot place, or whose depth contradicts the pose, are posed by anchored
     MapAnything.
   * **Maps with fewer than 3 keyframes:** SfM is re-run over all keyframes and aligned to the
     stored poses.
   * If no new keyframe overlaps the map, the command exits 5 and the map is unchanged.
5. **Refinement.**
   1. Geometry is re-run for keyframes whose COLMAP focal length differs by more than 3 %.
   2. For a new map, the metric scale is the median ratio between MoGe depth and SfM depth, and
      the map frame is defined.
   3. Each keyframe's depth is aligned to the map. The scale comes from its SfM points, or
      densely from overlapping keyframes. A scale outside 0.5–2 rejects the keyframe. A scale
      outside 0.8–1.25 marks it low-confidence, and such keyframes are excluded from fusion.
   4. For a new map, the map is levelled with the floor plane.
6. **Integration.** The update applies latest wins, updates the objects, fuses the cloud (Open3D
   TSDF), exports the scene and commits.

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
* `server` owns the models and nothing else.
* The layers run `cli` > `tools` > `viewer` > `mapping` > `segmentation` > `reconstruction` >
  `client` > `schema` > `core`.

## Benchmark evaluator

```sh
uv run python -m oh_my_slam.tools.evaluate [--out DIR] [--targets PATH] [--baseline PATH] \
                                           [--set-baseline] [--splits N]
uv run python -m oh_my_slam.tools.evaluate --resummarise DIR|latest [--set-baseline]
```

A single command benchmarks every entry point on `examples/`, strictly one command at a time
(spec §5):

* the server's cold start and resident memory (it stops and restarts the server)
* `reconstruct.sh` (JSON and PLY), `segment.sh -i -d` and `view.sh -i` on `restaurant.jpg`
* `segment.sh -i` on each of the 79 `ainex-captures` frames
* `mapper.sh update` on the sequence, once in one update and once split across `--splits`
  updates (default and minimum 3)
* `segment.sh -m` and `view.sh -m` on both maps

Video sampling (`-fps`) is not covered, because the examples contain no video. The metric groups
are:

| Group | Measures |
|---|---|
| `perf.*` | End-to-end wall time, client and server peak memory, and `view.sh` time to the rendered page. The report also breaks each command down per stage: time, and client and server peak memory. |
| `pose.*` | Yaw against the headings in the capture names, pitch direction of `up`/`down` frames, registered fraction, and same-heading pairs. |
| `map.*` | Frame agreement over overlapping keyframes, and the stability of ids, labels and OBBs between the one-update and the split map. |
| `seg.*` | Detections per frame. |
| `seg.map_consistency.*` | Per-frame detections compared with the map's objects. The map is built from the same detector, so these measure consistency, not accuracy. |
| `contract.*` | Colour contract, OpenLABEL validity, stdout purity, artefacts, exit codes, same objects, and read-only maps. The colour contract covers the viewer's OBBs and its `color=segment` cloud (`/api/cloud`). |
| `gt.*` | Accuracy against ground truth, when annotations exist. |

Per-stage memory comes from two sources. Each command records its stages (`core.timing`), and
the peak resident set of its own process during each stage, sampled every 50 ms. It writes these
to `OH_MY_SLAM_TIMINGS`, and a map update also keeps them in `map.json → updates[].timings`. The
server reports no memory of its own, so the evaluator samples two figures every 0.2 s: the
command's process tree, which includes COLMAP, and the server's physical footprint. It
attributes each sample to the stage whose time window it falls in.

Targets are data in `examples/targets.json`. Each target is an `op`/`value` pair with regression
tolerances, and a target can be edited without changing code. The file's `rationale` explains
each group of targets. Each metric's `measured` value is the reference run the targets were
derived from, and the evaluator ignores it. Ground-truth files dropped into
`examples/ground_truth/` are picked up without code changes (see its `README.md`).

Each run is compared with the stored baseline `~/oh-my-slam-data/evaluations/baseline.json`,
and regressions are flagged. Without a baseline the report says `baseline missing — not
compared`, and `summary.regressions` in `result.json` is `null`. For each failed or regressed
metric, the summary's `why` column gives the error, the value against the target, or the change
from the baseline and the tolerance it exceeded.

To keep a finished run as the baseline, either pass `--set-baseline` to the run itself or store
it afterwards:

```sh
uv run python -m oh_my_slam.tools.evaluate --resummarise latest --set-baseline
```

`--resummarise DIR|latest` runs nothing. It judges the values of a stored run (`latest` is the
newest `<UTC>` folder) against the current targets and baseline again, and rewrites its
`result.json` and `summary.md`. Use it after editing the targets, or with `--set-baseline` to
store that run as the baseline.

Results go outside the repository, to `~/oh-my-slam-data/evaluations/<UTC>/` by default:

* `result.json` (machine-readable)
* `summary.md` (human-readable)
* `runs/` (stdout, stderr and timings per command)
* `outputs/`
* `maps/`

The command prints the path of `summary.md` on stdout. It exits 0 when every metric passes, 1
when one fails, and 2 on a usage error. It needs the model weights. It also needs Microsoft Edge
or Google Chrome for Playwright's page timing. Run it on an otherwise idle machine, because
concurrent GPU work invalidates timings. `OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m eval`
runs it end to end as a test.

## Development

```sh
uv run pytest -m "not models and not browser and not eval" -q         # offline suite (stub server)
uv run pytest --cov=oh_my_slam -m "not models and not browser and not eval"
OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m models                 # real models; server running
uv run pytest -m browser                                              # viewer in Edge/Chrome (Playwright)
uv run ruff check . && uv run mypy src && uv run lint-imports         # lint, types, ownership
```

The offline suite runs against a deterministic stub server. The mapping end-to-end tests are
skipped when `colmap` is not installed. The `eval` marker selects the evaluator's end-to-end run
(see [Benchmark evaluator](#benchmark-evaluator)).

Environment variables:

| Variable | Effect |
|---|---|
| `OH_MY_SLAM_DEVICE=cpu\|mps` | Selects the server's device. |
| `OH_MY_SLAM_FEATURES=sift\|aliked` | COLMAP features for mapping (default `sift`). |
| `OH_MY_SLAM_TIMINGS=path.json` | Writes the full per-stage timing record of `reconstruct.sh`, `segment.sh` or `mapper.sh update` there: stage times, per-stage peak resident set and stage time windows. Each map update also keeps its record in `map.json → updates[].timings`. |
| `OH_MY_SLAM_DEBUG=1` | Prints tracebacks for internal errors. |
| `OH_MY_SLAM_LOG=DEBUG` | Sets the log level. |
| `OH_MY_SLAM_RUNTIME_DIR` | Replaces `~/Library/Caches/oh-my-slam` (socket, log, state, scratch). |
| `OH_MY_SLAM_WEIGHTS_DIR` | Replaces the Ultralytics weights folder. |
| `OH_MY_SLAM_QUEUE` | Server queue length (default 8). |
| `OH_MY_SLAM_COLMAP` | The COLMAP executable. |
| `OH_MY_SLAM_GEOMETRY_FP16=0\|1` | Overrides MoGe fp16 autocast on MPS (default on). |
| `OH_MY_SLAM_MAPANYTHING_REPO` | Overrides the MapAnything checkpoint. |

## Troubleshooting

* **`inference server is not running — start it with ./start_inference_server.sh` (exit 3).**
  Start the server. `--status` shows which models loaded, and the log is
  `~/Library/Caches/oh-my-slam/server.log`.
* **Exit 5 on an update.** The new images share no verified feature matches with the map.
* **Exit 6.** Another `mapper.sh update` is running on the same map.
* **`COLMAP CLI … and pycolmap … must both be 4.2.x`.** Run `brew upgrade colmap`, then
  `./scripts/install_tools.sh`.
