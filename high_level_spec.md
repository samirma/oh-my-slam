# Monocular RGB Mapping — High-Level Specification

## 1. Goal

Build a Python application that reconstructs 3D point clouds from **RGB-only** images and
video, incrementally assembles them into a persistent map that can be inspected in a
browser, and describes that map as a set of **labelled objects with oriented bounding
boxes (OBBs)**.

The system is exposed through four shell entry points:

| Entry point | Responsibility |
| --- | --- |
| `start_inference_server.sh` | Start the depth- and segmentation-inference server and any other long-lived services required for mapping. |
| `reconstruct.sh` | Single-frame reconstruction: image → scene description (JSON + OBBs) or point cloud. |
| `mapper.sh` | Multi-frame mapping: build, update and visualise a persistent map. |
| `segment.sh` | Instance segmentation: image or map → JSON + OBBs, a colour-coded segmented image, and an object catalogue. |

## 2. Components

### 2.1 Inference server

```sh
start_inference_server.sh
```

Loads the monocular depth-estimation model, the instance-segmentation model, and any other
service or model needed for mapping, and keeps them resident, so that individual
reconstructions do not pay model start-up cost. `reconstruct.sh`, `mapper.sh` and
`segment.sh` talk to this server; if it is not running they must fail with a clear,
actionable error.

### 2.2 Single-frame reconstruction — `reconstruct.sh`

```sh
reconstruct.sh -i <image>                # JSON scene description (default) to stdout
reconstruct.sh -i <image> -f ply         # point cloud to stdout
reconstruct.sh view -i <image>           # serve the reconstruction in a browser
```

* `-i <image>` — input RGB image.
* `-f json|ply` — output format, **default `json`**. The result is written to **stdout** so
  it can be piped or redirected; diagnostics go to stderr.
  * `json` — the scene description of §3: detected objects, their labels and their OBBs.
  * `ply` — the raw point cloud, with per-point colour.
* `view` — start a web server that renders the reconstructed point cloud interactively,
  with the OBBs drawn over it.

### 2.3 Mapping — `mapper.sh`

```sh
mapper.sh update -a <image(s)|video> -m <folder> [-f json|ply] -t full|single [-fps <n>]
mapper.sh view -m <folder>
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

Since an image captures a specific point in time for a map section, any new image that contradicts the current data should update the map with the latest information to keep it current.

`view` starts a web server for navigating the map. The web interface should offer options
to show:

- the point cloud of the whole map, with and without the dense map (GLB);
- object OBBs and labels overlaid.

Object identity is persistent: an object observed across several frames keeps one `id` and
one colour for the lifetime of the map, and its OBB is refined as evidence accumulates.

### 2.4 Segmentation — `segment.sh`

```sh
segment.sh -i <image> [-o <folder>] [-f json|ply] [--min-score <s>] [--labels a,b,c]
segment.sh -m <map-folder> [-o <folder>] [-f json|ply]
segment.sh view -i <image>
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
* `view` — serve the segmented image, the catalogue and the 3D OBBs in a browser.

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
`segments.ply`, and the OBB colour in the `view` server are **the same sRGB triple**.

The mapping is a pure function of the object `id`, so re-running against the same map
yields the same colours, and the palette cycles by hue once it is exhausted.

## 3. Scene description (JSON) returned by the tools

Use a well-known JSON schema: the ASAM OpenLABEL OBB format, referenced by its schema URL.

## 4. Constraints

* **Input is RGB only.** No depth sensor, no stereo pair, no IMU — depth comes from
  monocular inference.
* **Python**, with dependencies and the virtual environment managed by `uv` (`.venv`).
* `mapper.sh` delegates to `reconstruct.sh` whenever depth/reconstruction is needed, and
  `segment.sh` is the single owner of segmentation, OBB fitting and colour assignment;
  neither `reconstruct.sh` nor `mapper.sh` re-implements that logic.
* Shared logic lives in a common Python package used by all tools. Avoid duplicated and
  redundant code; follow standard Python project conventions (typed interfaces, small
  focused modules, tests).
* JSON on stdout must be machine-parseable on its own — no banners, no progress output.
  Everything human-facing goes to stderr.
* It must work on a Mac with an M4 chip.
* The tools must be accurate and performant, both per image and for mapping.
* The project must have comprehensive unit tests ensuring that all required components and
  their behaviour align with the expected plan.

