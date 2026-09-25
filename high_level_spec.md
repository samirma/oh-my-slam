# Monocular RGB Mapping — High-Level Specification

## 1. Goal

Build an application that reconstructs 3D point clouds from **RGB-only** images and
video, incrementally assembles them into a persistent map that can be inspected in a
browser, and describes that map as a set of **labelled objects with oriented bounding
boxes (OBBs)**.

The system is exposed through five shell entry points:

| Entry point | Responsibility |
| --- | --- |
| `start_inference_server.sh` | Start the depth- and segmentation-inference server and any other long-lived services required for mapping. |
| `reconstruct.sh` | Single-frame reconstruction: image → scene description (JSON + OBBs) or point cloud. |
| `mapper.sh` | Multi-frame mapping: build and update a persistent map. |
| `segment.sh` | Instance segmentation: image or map → JSON + OBBs, a colour-coded segmented image, and an object catalogue. |
| `view.sh` | Browser visualisation of either a single image reconstruction or a persisted map. |

This document states requirements only. Detailed decisions — defaults, coordinate
conventions, exit codes, the OpenLABEL field mapping, the colour palette — are recorded in
`README.md` and in the implementation plan (`~/.dev-workflow/oh-my-slam.md`), and must not
contradict this document.

## 2. Components

### 2.1 Inference server

```sh
start_inference_server.sh
```

Loads the monocular depth-estimation model, the instance-segmentation model, and any other
service or model needed for mapping, and keeps them resident, so that individual
reconstructions do not pay model start-up cost. `reconstruct.sh`, `mapper.sh update`,
`segment.sh -i` and `view.sh -i` use this server. Any operation that requires inference must
fail with a clear, actionable error if the server is not running. Operations on an already
persisted map (`segment.sh -m`, `view.sh -m`) must not require the inference server.

### 2.2 Single-frame reconstruction — `reconstruct.sh`

```sh
reconstruct.sh -i <image>                # JSON scene description (default) to stdout
reconstruct.sh -i <image> -f ply         # point cloud to stdout
reconstruct.sh -i <image> -f ply -p color=segment,voxel=0.01,normals=on
reconstruct.sh -i <image> -f ply -o cloud.ply   # point cloud to a file
```

* `-i <image>` — input RGB image.
* `-f json|ply` — output format, **default `json`**. Unless `-o` is given, the result is
  written to **stdout** so it can be piped or redirected; diagnostics go to stderr.
  * `json` — the scene description of §3: detected objects, their labels, colours and OBBs;
    the same objects, `id`s and colours as `segment.sh -i` with default options.
  * `ply` — the point cloud; per-point colour by default (see `color` below).
* `-o <file>` — write the result to this file instead of stdout; stdout then stays empty.
* `-p <key=value[,key=value…]>` — point-cloud attributes for the PLY output (table below);
  requires a PLY output. Keys that are not given keep their defaults. An unknown key or an
  out-of-range value fails with an actionable error before any inference runs.

#### Point-cloud attributes

The same `-p` attributes apply to every command that writes a PLY: `reconstruct.sh -f ply`,
`mapper.sh -f ply`, and `segment.sh` with `-f ply` or `-d` (§2.3, §2.4).

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `color` | `rgb` \| `segment` \| `height` \| `none` | `rgb` (`segment` in `segment.sh`) | Per-point colour: the image colour; the object colour of the §2.4 colour contract (unsegmented points mid-grey, as in `segments.ply`); a ramp along the up axis (estimated gravity for an image, the map's up axis for a map); or no colour properties at all. |
| `stride` | integer ≥ 1 | `1` | Keep every n-th pixel along each image axis. |
| `min-depth`, `max-depth` | metres | full range | Keep only pixels whose depth lies within the range. |
| `edge` | relative depth jump ≥ 0 | `0.04` | Drop pixels on depth discontinuities (flying pixels); `0` disables the filter. |
| `voxel` | metres ≥ 0 | `0` (off) | Keep one representative point per voxel, chosen deterministically. Colours are not averaged, so object colours stay exact. |
| `normals` | `on` \| `off` | `off` | Add `nx ny nz` float properties. |
| `label` | `on` \| `off` | `off` | Add an `int label` property holding the object `id` (`0` = unsegmented). |
| `encoding` | `binary` \| `ascii` | `binary` | `binary_little_endian 1.0` or ASCII PLY. |

* Pixel-level attributes (`stride`, depth range, `edge`) apply before unprojection, so they
  exist only for single images (`reconstruct.sh`, `segment.sh -i`); on a map (`mapper.sh`,
  `segment.sh -m`) they fail with an actionable error. `voxel` applies to the resulting 3D
  points.
* `segment.sh` always colours by object: `color` is fixed to `segment` there, and any other
  value fails. Its remaining attributes shape both stdout and `segments.ply`.
* Attributes shape only the emitted cloud. The scene description, OBBs, object `id`s and
  object colours are the same whatever `-p` says.
* The PLY header records the effective attributes, defaults included, in a `comment` line,
  so every file states how it was produced.
* The attribute set, its defaults and its validation are defined once in the shared package
  and reused by every command that writes a PLY and by the `view.sh` controls (§2.5).

### 2.3 Mapping — `mapper.sh`

```sh
mapper.sh update -i <image(s)|video> -m <map-folder>   # whole map as JSON (default) to stdout
mapper.sh update -i <image(s)|video> -m <map-folder> [-f json|ply] [-o <file>] [-p <attrs>] [-t full|single] [-fps <n>]
```

`update` creates the map if `<map-folder>` does not yet exist or is empty, and otherwise
extends the existing map with the new input. A non-empty folder that is not a map is refused
and left untouched. Only `-i` and `-m` are required; every other option has a default.

* `-i <image(s)|video>` — one or more images, or a video file.
* `-m <map-folder>` — map directory; holds the persisted map and its metadata.
* `-f json|ply` — output format, **default `json`**: the scene description of §3, or the
  point cloud, in map coordinates. `-t` selects what either format covers.
* `-t full|single` — scope of the result, **default `full`**:
  * `full` — the **entire** map: every object, and the estimated camera pose of each
    contributing frame (PLY: the whole map cloud).
  * `single` — only what the **newly added** input covers: the poses of the new frames and
    the objects observed in them (PLY: the new frames' points).
* `-o <file>` — write the result to this file instead of stdout; stdout then stays empty.
* `-p <attrs>` — point-cloud attributes for `-f ply` (§2.2).
* `-fps <n>` — for video input, the number of frames per second to sample for analysis
  (default `2`); ignored for images.

The map's geometry is a point cloud; no surface mesh is produced.

Since an image captures a specific point in time for a map section, any new image that
contradicts the current data should update the map with the latest information to keep it
current. "Latest" is the order of addition: a later update wins over an earlier one.
Frames within one update count as one observation of the scene. Capture timestamps are not
used.

Object identity is persistent: an object observed across several frames keeps one `id` and
one colour for the lifetime of the map, and its OBB is refined as evidence accumulates.

### 2.4 Segmentation — `segment.sh`

```sh
segment.sh -i <image> [-f json|ply] [-o <file>] [-d <folder>] [-p <attrs>] [--min-score <s>]
segment.sh -m <map-folder> [-f json|ply] [-o <file>] [-d <folder>] [-p <attrs>]
```

For an image, segments it into object instances, lifts each instance into 3D using the depth
from the inference server, and fits an OBB to it. For a map, exports the map's persistent
objects (their `id`s and colours kept) without running inference and without modifying the
map. Exactly one of `-i` and `-m` is required.

* `-i <image>` — input RGB image. Mutually exclusive with `-m`.
* `-m <map-folder>` — export the objects of an existing map instead of segmenting an image.
* `-f json|ply` — output format, **default `json`**.
* `-o <file>` — write the result to this file instead of stdout; stdout then stays empty.
* `-d <folder>` — also write the output artefacts listed below into this folder. Without
  `-o` and `-d`, the result goes to stdout and no files are written.
* `-p <attrs>` — point-cloud attributes for the PLY output (§2.2).
* `--min-score <s>` — `-i` only: drop detections below this confidence (default `0.5`). It
  only adds or removes objects: the objects present at two thresholds keep the same `id` and
  colour.

#### Output artefacts

With `-d <folder>`, `segment.sh` writes:

| File | Contents |
| --- | --- |
| `segmentation.json` | The scene description of §3 — objects, labels, colours and OBBs. Identical to what `-f json` outputs. |
| `segmented.png` | The input image (for a map, a selection of its keyframes) with each instance mask painted in that object's colour, over a dimmed copy of the original. |
| `catalog.csv` | One row per object: `id,label,score,color_hex,width_m,height_m,depth_m,volume_m3,center_x,center_y,center_z,pixel_count,point_count`. |
| `catalog.md` | The same catalogue as a human-readable table, ordered by descending volume. |
| `segments.ply` | Point cloud with each point coloured by its object colour (unsegmented points mid-grey). Identical to what `-f ply` outputs. |

#### Colour contract

Every artefact of a single run agrees on colour. One colour per object `id`, drawn
deterministically from a fixed, perceptually distinct palette, so that the value in
`segmentation.json` (`color` / `color_hex`), the pixels of that instance's mask in
`segmented.png`, the `color_hex` column of `catalog.csv` and the swatch of `catalog.md`, the
per-point colour in `segments.ply` and in any PLY or `view.sh` cloud coloured with
`color=segment`, and the OBB colour rendered by `view.sh` are **the same sRGB triple**. Masks
are painted opaque, and each pixel and each point belongs to at most one object, so no two
colours ever mix.

Object `id`s are positive integers (`0` means unsegmented). The mapping is a pure function
of the object `id`, so re-running against the same map yields the same colours, and the
palette cycles by hue once it is exhausted. The palette never produces the mid-grey reserved
for unsegmented points. For a single image, `id`s are assigned in a deterministic order, so
re-running the same image with the same options gives each object the same `id` and colour.

### 2.5 Visualisation — `view.sh`

```sh
view.sh -i <image>
view.sh -m <map-folder>
```

Starts a local web server for interactive browser visualisation. Exactly one input is
required: `-i` and `-m` are mutually exclusive.

* `-i <image>` — reconstruct and segment one RGB image, then show its colour point cloud,
  segmented image, object catalogue, labelled OBBs, and the camera at its estimated pose.
* `-m <map-folder>` — load an existing map without modifying it, then show its complete
  point cloud, camera poses, and labelled OBBs.

The interface must provide independent controls for the available point-cloud,
camera-pose, segmentation, label, and OBB layers. The point-cloud layer also offers live
controls for the §2.2 attributes that affect what is displayed: `color`, `stride`,
`min-depth` / `max-depth`, `edge`, `voxel` and `normals` for an image; only `color`,
`voxel` and `normals` for a map. Changing a control re-derives the cloud through the shared
point-cloud code from data already computed, and never re-runs inference. `encoding` and the
`label` property concern PLY files only and have no control.
The web interface must show the position coordinates of every displayed camera and provide an
option to move the viewer viewpoint to that camera position.

`view.sh` owns only the web server and browser UI. It consumes reconstruction, mapping, and
segmentation data through their existing implementations and must not duplicate depth
inference, point-cloud generation, map loading, segmentation, OBB fitting, object identity,
or colour assignment.

## 3. Scene description (JSON) returned by the tools

The scene JSON follows the well-known ASAM OpenLABEL 1.0.0 schema, with each object's OBB
stored as a `cuboid`. Every document names the schema URL it conforms to inside
`openlabel.metadata` (the schema allows no other root key, so a root `$schema` would fail
validation) and must validate against that schema.

## 4. Constraints

* **Input is RGB only.** No depth sensor, no stereo pair, no IMU — depth comes from
  monocular inference.
* **Python**, with dependencies and the virtual environment managed by `uv` (`.venv`), is
  preferable but not mandatory.
* Ownership is enforced between the modules of the shared package, not by the shell scripts
  calling each other. Taken literally at the shell level it would be circular:
  `reconstruct.sh -f json` needs objects, and `segment.sh` needs depth. Mapping delegates
  depth and reconstruction to the reconstruction code, and the segmentation code is the
  single owner of segmentation, OBB fitting and colour assignment. Neither reconstruction,
  mapping nor the viewer re-implements that logic.
* Shared logic lives in one common package used by all tools. Avoid duplicated and
  redundant code; follow the language's standard project conventions (typed interfaces,
  small focused modules, tests).
* stdout carries at most one JSON document or one PLY file, machine-parseable on its own — no
  banners, no progress output. Everything human-facing goes to stderr. `reconstruct.sh`,
  `mapper.sh` and `segment.sh` accept `-o <file>` to write that result to a file instead.
* It must work on the target machine, an Apple M4 Max Mac with 36 GB of unified memory,
  where performance is measured; smaller Macs are best-effort. MPS support is preferable but
  not mandatory.
* The tools must be accurate and performant, both per image and for mapping, as measured by
  the evaluators of §5 against their targets.
* The project must have comprehensive unit tests ensuring that all required components and
  their behaviour align with this specification.
* Licences are not a blocker: the project is for personal and research use, so copyleft and
  non-commercial components are acceptable. Every third-party component and its licence is
  listed in `THIRD_PARTY_LICENSES.md`.

## 5. Benchmark evaluators

The project must ship benchmark evaluators that measure the accuracy and performance of
every entry point, using the files in `examples/` as reference inputs. Video sampling
(`mapper.sh -fps`) is not covered, since the examples contain no video.

* `start_inference_server.sh` — cold-start time and resident memory; no input file needed.
* `examples/restaurant.jpg` — single-frame reference for `reconstruct.sh`, `segment.sh -i`
  and `view.sh -i`.
* `examples/ainex-captures/` — an ordered 79-frame capture sequence (640×480, rendered, no
  EXIF) of a robot head turning in place, for `mapper.sh` and, frame by frame, `segment.sh -i`;
  and — on the resulting map — for `segment.sh -m` and `view.sh -m`. File names encode the
  **commanded** head motion as `NNN_<motion>_<tilt>.jpg`:
  * `NNN` — capture order

The evaluators must report at least:

* **Performance** — per-stage and end-to-end wall time and peak memory, per image and per
  mapping update; server cold start and resident memory; `view.sh` time until the page has
  rendered.
* **Pose accuracy** — estimated camera yaw against the headings encoded in the capture file
  names, pitch direction of the `up` / `down` frames, and the fraction of frames successfully
  registered.
* **Map quality** — point-cloud consistency across overlapping frames (e.g. the same-heading
  pairs above), and stability of object `id`s, labels and OBBs when the same sequence is
  mapped in one update versus split across several.
* **Segmentation** — detections, labels and scores per frame (`restaurant.jpg` and each
  `ainex-captures` frame), compared with the map's objects for the frames that observe them.
* **Contracts**, for every command — colour contract, OpenLABEL validity, stdout purity.

Evaluators run as a single command, write a machine-readable result (JSON) and a
human-readable summary, and store results outside the repository so runs can be compared
over time. Each metric has a target and a pass/fail result, and each run is also compared
with a stored baseline run so that regressions are flagged. Targets and the baseline are
data read by the evaluators, so they change without code changes. Ground-truth annotations
added later for the example files must be picked up without changing evaluator code.
