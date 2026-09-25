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
* **`-m`** opens the map read-only, without the server. It shows the map's complete cloud, every
  keyframe camera and the labelled OBBs.

The first view of an image is from just behind the photo's viewpoint. The first view of a map
looks down 60° on the whole scene and all its cameras, so that the walls of a room hide little
of its floor, objects and camera cluster. *Reset view* (`R`) returns to it.

The page has four tabs:

* **Controls**
  * *Layers* has one independent switch each for the point cloud, the segmentation overlay, the
    camera poses, the labels and the oriented boxes. The segmentation overlay draws only the
    points of each object, in its colour, over the cloud. It is not the `color=segment`
    attribute, which recolours the whole cloud (unsegmented points grey); the layer's note says
    when that attribute is on as well.
  * *Point cloud* has live controls for the point-cloud attributes that affect the display. For
    an image these are `color`, `stride`, `min-depth`, `max-depth`, `edge`, `voxel` and
    `normals`. For a map they are `color`, `voxel` and `normals`. The panel also shows the
    equivalent `-p …` string and a *Defaults* button.
  * *Display* sets the point size, the normals shading, the labels (*id tags + names that fit*,
    or *id tags only*) and the background.
* **Catalogue** lists the objects, with a label filter.
* **Cameras** lists every displayed camera's centre (x, y, z in metres, in the scene frame), with
  a *Go to* button that moves the viewpoint to that camera, looking where it looked. `[` and `]`
  step through the cameras.
* **Image** shows the segmented image (`-i` only). A click enlarges it to the window; in the
  enlarged view a click switches between fit and actual pixels, and `Esc` closes it.

Other keys: `R` resets the view, and `Esc` clears the selection.

Every box whose top is in view carries a label: its id on a tag in the object's colour, next to
the box or, when that spot is taken, a little farther out with a leader line to it. No label
covers another label, the header or the help line, and labels never leave the view. Tags that
still do not fit are counted on a `+N` chip near their boxes: hovering or clicking the chip lists
them (id and name), and clicking an entry selects that box. Names are added wherever they fit,
larger boxes on screen first. The selected box always shows its name, and hovering a box or its
tag shows its id and name. Camera frustums fade out as the viewpoint comes
near them. The camera being looked through and its neighbours, for example the rest of a capture
that turns in place, therefore never draw lines across the view. Boxes that enclose the viewpoint
fade the same way, except the selected one.

The viewer contains no geometry, segmentation or colour logic of its own:

* Every displayed cloud is derived on request from data already in memory, by the same code
  that writes PLY files. A control therefore never re-runs inference.
* The page shows the complete cloud up to 12,000,000 points. In Edge on the M4 Max, a cloud of
  that size loads in about 2 s and orbits at 60 frames per second. Larger clouds are thinned for
  display only, keeping every k-th point, and the page says so.
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
| `object_data.vec` | `color` `[r, g, b]`; for a map object detected under several labels, also `detected_as` (every label, most evidence first) |
| `object_data.boolean` | `confirmed` |
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
| `scene.json` | The full scene as of the last update. `segment.sh -m` and `view.sh -m` rebuild the scene from the persisted state instead, so that object colours follow the current palette. |
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
  * Duplicates are merged, and a merge keeps the lower id. Objects of compatible labels merge
    when their points or boxes overlap. Objects of different labels merge when the detector's
    label flickered between keyframes: no keyframe detected both, their sizes are comparable
    (not a part of the other or an item resting on it), and at least half of either one's
    points lie on the other's surface. The merged object takes the label with the most
    evidence and lists the others in `detected_as`.
  * **Copies placed by inconsistent depth** are merged too. Keyframes whose monocular depth
    disagrees (locally, even after the global depth adjustment of Refinement) can place an
    object twice along the same viewing rays. Objects of compatible labels that no keyframe
    detected together merge when, for most of the 3 pairs of their keyframes nearest by
    viewpoint, the two keyframes' depths disagree by at least 5 % (measured on the surfaces both
    see) and the detections coincide once that ratio is removed.
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
  * An object is exported once it is confirmed: detected with a reliable mask (not mostly in
    the image-border band, where the object is cut off and monocular depth is unreliable) in at
    least 2 keyframes, or in 1 when no other keyframe of the map had it in view (occlusion is
    ignored, so a detection whose depth puts it inside another surface cannot confirm itself).
    It must also be visibly drawn in the map cloud, so that it appears in `segments.ply` and
    every `color=segment` cloud in its colour: it needs at least 5 % of the cloud points its
    box's largest face holds (one per cloud voxel), and at least 10. A keyframe's vote for an
    object counts only for cloud points inside the object's attribution gate, its box grown by
    the depth noise at its viewing distance (max(5 cm, 3 % of the distance)): detection masks
    are drawn generously (a "carpet" mask over a counter top and the floor beyond it), and the
    object's coloured points must coincide with its box. The vote needs a third of the
    keyframes that see a point, so an object detected in fewer of them wins only part of its
    surface (a refrigerator detected in 7 of the ~15 keyframes that see its front) or none of it
    (a light switch on a wall, a dishwasher detected in 2 of ~13). Each confirmed object
    therefore also takes the unlabelled cloud points nearest its own lifted points (the 4
    nearest to each, within max(3 cm, 2 % of its viewing distance), inside its gate; a point two
    objects pick goes to the nearer). An object stays short only when its surface did not
    survive the fusion (seen by fewer than 3 keyframes, such as a pendant lamp); it is then not
    exported.
  * Unconfirmed objects are kept, so that a later update can still confirm them.
  * The OBB is fitted to the detections that agree with each other: monocular depth of a small
    object can vary by tens of percent between keyframes, and the union of such detections is a
    streak along the viewing rays. With 3 or more detections, the box covers those whose bounds
    overlap (allowing for depth noise) the detection most others agree with.
  * The OBB covers the **observed surface** only (see Known limitations): no class-typical size
    is assumed.
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
6. **OBBs:** each object gets an upright OBB, then ids and colours are assigned. A box of a
   floor-standing class is extended down to the floor when its visible bottom floats at most
   the class's gap above it (0.8 m for furniture, 1.2 m for a person, 0.15 m for classes that
   also stand on furniture or hang on walls, such as plants and shelves). The visible part must
   span at least a fifth of the grounded height, except for classes seen mostly from the top
   (tables, desks, counters, beds), and boxes under 10 cm are never grounded.

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
   * **Weakly linked parts of a new map** (typical of a video walk through several rooms): a
     stretch that hangs on the rest by one weak link — a doorway crossed with a few dozen
     matches, or one keyframe shared with the rest — has no scale of its own for the global
     mapper, which returns it shrunk onto one point or at an arbitrary scale depending on its
     random start, and its orientation rests on that link alone. After SfM:
     1. An SfM pose needs 20 triangulated points. Keyframes with fewer are not accepted as posed
        (a stretch shrunk onto a point triangulates nothing); incremental mapping with the others
        fixed may register them again.
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
6. **Integration.** The update applies latest wins, fuses the cloud (Open3D TSDF), updates the
   objects (the fused surface tells pieces of one horizontal surface), gives the cloud's points
   their object ids, exports the scene and commits.

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
| `map.*` | Frame agreement of the same-heading pairs and of every overlapping keyframe pair (optical axes < 45° apart, any distance in capture order: median and p90 over the pairs, share of pairs above 10 %, worst pair; the detail splits sequence neighbours, ≤ 10 keyframes apart, from loop closures); near-duplicate objects (compatible labels, or both horizontal-surface labels at one height; never detected in the same keyframe; boxes within 0.3 m); the largest share of an object's detected mask points (lifted with the detecting keyframe's depth) outside its box grown by the depth noise, max(5 cm, 5 % of the depth), and, as a check of the mapper's attribution gate, of its cloud points outside that gate; and the stability of ids, labels and OBBs between the one-update and the split map. Ids and boxes are compared on a label-aware pairing, labels on a label-blind one. |
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

## Known limitations

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
* **Video through several rooms.** Where a walk crosses a doorway in a second or two,
  consecutive keyframes share only a few matches, and the global mapper can leave the stretch
  behind the doorway at any scale or tilt, or shrink it onto one point (see Poses). The mapper
  corrects the scale with the depth and the tilt with gravity, but the stretch's heading and
  position still rest on that one link, or on multi-view poses anchored on its capture-order
  neighbours, so they can be a few degrees and decimetres off. Keyframes whose gravity still
  disagrees by more than 25° afterwards are left out: on the user's `livingroom.mp4` at `-fps 1`,
  the dim corridor at 77–81 s. The global mapper starts from random positions, so two runs on one
  video can differ in such stretches.

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
