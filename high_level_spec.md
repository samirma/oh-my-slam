# Monocular RGB Mapping — High-Level Specification

## 1. Goal

Build a Python application that reconstructs 3D point clouds from **RGB-only** images and
video, and incrementally assembles them into a persistent map that can be inspected in a
browser.

The system is exposed through three shell entry points:

| Entry point | Responsibility |
| --- | --- |
| `start_inference_server.sh` | Start the depth-inference server and any other long-lived services required for mapping. |
| `reconstruct.sh` | Single-frame reconstruction: image → point cloud. |
| `mapper.sh` | Multi-frame mapping: build, update and visualise a persistent map. |

## 2. Components

### 2.1 Inference server

```sh
start_inference_server.sh
```

Loads the monocular depth-estimation model (and any other service, model needed for mapping) and
keeps it resident, so that individual reconstructions do not pay model start-up cost.
`reconstruct.sh` and `mapper.sh` talk to this server; if it is not running they must fail
with a clear, actionable error.

### 2.2 Single-frame reconstruction — `reconstruct.sh`

```sh
reconstruct.sh -i <image> -f ply        # write the point cloud to stdout
reconstruct.sh view -i <image>          # serve the point cloud in a browser
```

* `-i <image>` — input RGB image.
* `-f ply` — output format; the point cloud is written to **stdout** so it can be piped or
  redirected. Diagnostics go to stderr.
* `view` — start a web server that renders the reconstructed point cloud interactively.

### 2.3 Mapping — `mapper.sh`

```sh
mapper.sh update -a <image(s)|video> -m <folder> -f ply -t full|single [-fps <n>]
mapper.sh view -m <folder>
```

`update` creates the map if `<folder>` does not yet exist, otherwise extends the existing
map with the new input.

* `-a <image(s)|video>` — one or more images, or a video file.
* `-m <folder>` — map directory; holds the persisted map and its metadata.
* `-f ply` — output format.
* `-t full` — return the point cloud for the **entire** map, including the estimated camera
  pose of each contributing frame.
* `-t single` — return the point cloud for the **newly added** input only.
* `-fps <n>` — for video input, the number of frames per second to sample for analysis.

`view` starts a web server for navigating the point cloud of the whole map.

## 3. Constraints

* **Input is RGB only.** No depth sensor, no stereo pair, no IMU — depth comes from
  monocular inference.
* **Python**, with dependencies and the virtual environment managed by `uv` (`.venv`).
* `mapper.sh` delegates to `reconstruct.sh` whenever depth/reconstruction is needed; it must
  not re-implement that logic.
* Shared logic lives in a common Python package used by both tools. Avoid duplicated and
  redundant code; follow standard Python project conventions (typed interfaces, small
  focused modules, tests).
