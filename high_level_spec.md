# Monocular RGB Mapping — High-Level Specification

## 1. Goal

Build a Python application that reconstructs 3D point clouds from **RGB-only** images and
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

## 2. Components

### 2.1 Inference server

```sh
start_inference_server.sh
```

Loads the monocular depth-estimation model, the instance-segmentation model, and any other
service or model needed for mapping, and keeps them resident, so that individual
reconstructions do not pay model start-up cost. `reconstruct.sh`, `mapper.sh`, `segment.sh`
and image-mode `view.sh` use this server. Any operation that requires inference must fail
with a clear, actionable error if the server is not running. Viewing an already persisted
map must not require the inference server.

### 2.2 Single-frame reconstruction — `reconstruct.sh`

```sh
reconstruct.sh -i <image>                # JSON scene description (default) to stdout
reconstruct.sh -i <image> -f ply         # point cloud to stdout
reconstruct.sh -i <image> -f ply -p color=segment,voxel=0.01,normals=on
```

* `-i <image>` — input RGB image.
* `-f json|ply` — output format, **default `json`**. The result is written to **stdout** so
  it can be piped or redirected; diagnostics go to stderr.
  * `json` — the scene description of §3: detected objects, their labels and their OBBs.
  * `ply` — the raw point cloud, with per-point colour.
* `-p <key=value[,key=value…]>` — point-cloud attributes for the PLY output (table below);
  requires `-f ply`. Keys that are not given keep their defaults. An unknown key or an
  out-of-range value fails with an actionable error before any inference runs.

#### Point-cloud attributes

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `color` | `rgb` \| `segment` \| `height` \| `none` | `rgb` | Per-point colour: the image colour; the object colour of the §2.4 colour contract (unsegmented points mid-grey, as in `segments.ply`); a ramp along the estimated up axis; or no colour properties at all. |
| `stride` | integer ≥ 1 | `1` | Keep every n-th pixel along each image axis. |
| `min-depth`, `max-depth` | metres | full range | Keep only pixels whose depth lies within the range. |
| `edge` | relative depth jump ≥ 0 | `0.04` | Drop pixels on depth discontinuities (flying pixels); `0` disables the filter. |
| `voxel` | metres ≥ 0 | `0` (off) | Keep one representative point per voxel. Colours are not averaged, so object colours stay exact. |
| `normals` | `on` \| `off` | `off` | Add `nx ny nz` float properties. |
| `label` | `on` \| `off` | `off` | Add an `int label` property holding the object `id` (`0` = unsegmented). |
| `encoding` | `binary` \| `ascii` | `binary` | `binary_little_endian 1.0` or ASCII PLY. |

* Pixel-level attributes (`stride`, depth range, `edge`) apply before unprojection; `voxel`
  applies to the resulting 3D points.
* Attributes shape only the emitted cloud. The scene description, OBBs, object `id`s and
  colours are the same whatever `-p` says.
* The PLY header records the effective attributes, defaults included, in a `comment` line,
  so every file states how it was produced.
* The attribute set, its defaults and its validation are defined once in the shared package
  and reused by `view.sh` (§2.5).

### 2.3 Mapping — `mapper.sh`

```sh
mapper.sh update -a <image(s)|video> -m <folder> [-f json|ply] -t full|single [-fps <n>]
```

`update` creates the map if `<folder>` does not yet exist, otherwise extends the existing
map with the new input.

* `-a <image(s)|video>` — one or more images, or a video file.
* `-m <folder>` — map directory; holds the persisted map and its metadata.
* `-f json|ply` — output format, **default `json`** (the scene description of §3, covering
  the whole map in map coordinates).
* `-t full` — return the scene for the **entire** map, including the estimated camera pose
  of each contributing frame.
* `-t single` — return the scene for the **newly added** input only.
* `-fps <n>` — for video input, the number of frames per second to sample for analysis.

Since an image captures a specific point in time for a map section, any new image that
contradicts the current data should update the map with the latest information to keep it
current.

Object identity is persistent: an object observed across several frames keeps one `id` and
one colour for the lifetime of the map, and its OBB is refined as evidence accumulates.

### 2.4 Segmentation — `segment.sh`

```sh
segment.sh -i <image> [-o <folder>] [-f json|ply] [--min-score <s>] [--labels a,b,c]
segment.sh -m <map-folder> [-o <folder>] [-f json|ply]
```

Segments the input into object instances, lifts each instance into 3D using the depth from
the inference server, and fits an OBB to it.

* `-i <image>` — input RGB image. Mutually exclusive with `-m`.
* `-m <map-folder>` — segment an existing map instead of a single image.
* `-o <folder>` — write the output artefacts listed below. If omitted, only the JSON (or
  PLY) goes to stdout and no files are written.
* `-f json|ply` — stdout format, **default `json`**.
* `--min-score <s>` — drop detections below this confidence (default `0.5`).
* `--labels a,b,c` — restrict the output to these class labels.

#### Output artefacts

With `-o <folder>`, `segment.sh` writes:

| File | Contents |
| --- | --- |
| `segmentation.json` | The scene description of §3 — objects, labels, colours and OBBs. Identical to what goes to stdout. |
| `segmented.png` | The input image with each instance mask painted in that object's colour, over a dimmed copy of the original. |
| `catalog.csv` | One row per detected object: `id,label,score,color_hex,width_m,height_m,depth_m,volume_m3,center_x,center_y,center_z,pixel_count,point_count`. |
| `catalog.md` | The same catalogue as a human-readable table, ordered by descending volume. |
| `segments.ply` | Point cloud with each point coloured by its object colour (unsegmented points mid-grey). Written when `-f ply`, or always when `-o` is given. |

#### Colour contract

Every artefact of a single run agrees on colour. One colour per object `id`, drawn
deterministically from a fixed, perceptually distinct palette, so that the value in
`segmentation.json` (`color` / `color_hex`), the pixels of that instance's mask in
`segmented.png`, the swatch column of `catalog.csv` / `catalog.md`, the per-point colour in
`segments.ply`, and the OBB colour rendered by `view.sh` are **the same sRGB triple**.

The mapping is a pure function of the object `id`, so re-running against the same map
yields the same colours, and the palette cycles by hue once it is exhausted.

### 2.5 Visualisation — `view.sh`

```sh
view.sh -i <image>
view.sh -m <map-folder>
```

Starts a local web server for interactive browser visualisation. Exactly one input is
required: `-i` and `-m` are mutually exclusive.

* `-i <image>` — reconstruct and segment one RGB image, then show its colour point cloud,
  segmented image, object catalogue, and labelled OBBs, camera-pose.
* `-m <map-folder>` — load an existing map without modifying it, then show its complete
  point cloud, camera poses, and labelled OBBs.

The interface must provide independent controls for the available point-cloud,
camera-pose, segmentation, label, and OBB layers. `view.sh` owns only the web server and
browser UI. It consumes reconstruction, mapping, and segmentation data through their
existing implementations and must not duplicate depth inference, point-cloud generation,
map loading, segmentation, OBB fitting, object identity, or colour assignment.

## 3. Scene description (JSON) returned by the tools

Use a well-known JSON schema: the ASAM OpenLABEL OBB format, referenced by its schema URL.

## 4. Constraints

* **Input is RGB only.** No depth sensor, no stereo pair, no IMU — depth comes from
  monocular inference.
* **Python**, with dependencies and the virtual environment managed by `uv` (`.venv`), is
  preferable but not mandatory.
* `mapper.sh` delegates to `reconstruct.sh` whenever depth/reconstruction is needed, and
  `segment.sh` is the single owner of segmentation, OBB fitting and colour assignment;
  neither `reconstruct.sh`, `mapper.sh` nor `view.sh` re-implements that logic.
* Shared logic lives in a common Python package used by all tools. Avoid duplicated and
  redundant code; follow standard Python project conventions (typed interfaces, small
  focused modules, tests).
* JSON on stdout must be machine-parseable on its own — no banners, no progress output.
  Everything human-facing goes to stderr.
* It must work on a Mac with an M4 chip; MPS support is preferable but not mandatory.
* The tools must be accurate and performant, both per image and for mapping.
* The project must have comprehensive unit tests ensuring that all required components and
  their behaviour align with the expected plan.
* The licence shouldn't be a blocker.

## 5. Benchmark evaluators

The project must ship benchmark evaluators that measure the accuracy and performance of
every entry point, using the files in `examples/` as reference inputs:

* `examples/restaurant.jpg` — single-frame reference for `reconstruct.sh`, `segment.sh -i`
  and `view.sh -i`.
* `examples/ainex-captures/` — an ordered 79-frame capture sequence for `mapper.sh` and
  `segment.sh -m`. File names encode the capture motion
  (`NNN_<direction>_<yaw°>_<level|up|down>`), which gives the expected relative yaw and pitch
  of each frame.

The evaluators must report at least:

* **Performance** — per-stage and end-to-end wall time and peak memory, per image and per
  mapping update.
* **Pose accuracy** — estimated camera yaw and pitch against the headings encoded in the
  capture file names, and the fraction of frames successfully registered.
* **Map quality** — point-cloud consistency across overlapping frames (e.g. the loop back to
  the starting heading), and stability of object `id`s, labels and OBBs across frames that
  observe the same object.
* **Segmentation** — detections, labels and scores per frame, plus contract checks (colour
  contract, OpenLABEL validity, stdout purity).

Evaluators run as a single command, write a machine-readable result (JSON) and a
human-readable summary, and store results outside the repository so runs can be compared
over time. Ground-truth annotations added later for the example files must be picked up
without changing evaluator code.
