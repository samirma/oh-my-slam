# oh-my-slam — High-Level Specification

## 1. Purpose and scope

oh-my-slam exists to serve one client: the **robot-guide skill**. The skill's goal states that
"at any point in time, the working map can be accessible", describing the most up-to-date map of
the scene, while a robot (a wheeled base, a humanoid or an arm, real or simulated) moves through
that scene. oh-my-slam builds and keeps that map from the robot's camera frames, and answers
the questions the skill's scripts ask about it.

**Scope test.** A feature belongs here only if a row of the client contract (§2) needs it.
Anything else is removed (§12). oh-my-slam knows nothing about any particular robot, scene or
task:
* robot knowledge reaches it only as camera pose priors and camera parameters (§4);
* footprints and height bands are parameters of the requests that need them.

The skill's scripts do all numeric work against this API, so every number the skill needs is in
a response body, with its unit, frame and uncertainty.

This document states requirements. Detailed decisions are recorded in `README.md` and must not
contradict it. Examples of such decisions: defaults, thresholds, encodings, palette.

| File | Contents |
| --- | --- |
| [specs/api.md](specs/api.md) | HTTP API: endpoints, requests, responses, errors |
| [specs/map.md](specs/map.md) | Map model: frame, versions, update jobs, quality gate, objects, grid |
| [specs/inference.md](specs/inference.md) | Inference server and its lanes |
| [specs/evaluation.md](specs/evaluation.md) | Replay evaluator and targets |
| [specs/api.schema.json](specs/api.schema.json) | JSON Schema (2020-12) of every request and response body |
| `SKILL.md` | Short API reference for the robot-guide client, generated from the API definitions |

## 2. Client contract

Each row is something the skill needs. Each row's test is in §11.

| # | The skill needs | Provided by |
| --- | --- | --- |
| C1 | One map frame that never jumps: positions it stored before an update are still valid after it. | Anchored map frame (map.md §1) |
| C2 | The map's metric scale and how well it is known, so map distances can be driven. | `scale_source`, `scale_sigma_rel` on every metric output |
| C3 | A readable, current map at any moment, including while an update runs. | Versioned snapshots; two-phase update jobs (map.md §2–§3) |
| C4 | To know, per frame sent, whether it was used, and why not. | Per-input status in the update result (api.md §4) |
| C5 | To undo an update that made the map worse, in seconds. | Revert to a kept version (api.md §5) |
| C6 | To reject an update automatically when it would damage the map. | Quality report and `accept_if` gate (map.md §4) |
| C7 | To check frames before sending them as an update. | Dry run (api.md §4) |
| C8 | Where the camera is, with an uncertainty and a warning when the answer is ambiguous. | Locate (api.md §6) |
| C9 | Where the robot can go: free, occupied and unknown floor, the frontiers, rooms and doorways. | Navigation grid (map.md §6) |
| C10 | The objects of the scene: what, where, how big, their real colour, whether they moved. | Object records (map.md §5) |
| C11 | What the camera sees now, without changing the map: objects, points, and a range scan. | Frame analysis (api.md §7) |
| C12 | Locate and frame analysis answered within seconds even while an update runs. | Inference lanes (inference.md) |
| C13 | A map that shows what is there now: a later observation wins over an earlier one. | Update semantics (map.md §3) |

## 3. Frames and units

* **Units.** Metres, degrees, seconds, everywhere. Angles in responses are degrees.
* **Camera frame.** Optical axes: x right, y down, z forward.
* **Map frame.** Gravity-aligned and right-handed: x and y horizontal, z up. Yaw is measured
  counter-clockwise from +x about +z. How the frame is fixed is in map.md §1.
* **Poses.** `T_a_b` is the 4×4 homogeneous pose of frame b in frame a, written as nested
  row-major arrays `M[row][col]`. `T_map_cam` is the pose of the camera's *optical* frame in the
  map frame, so it maps optical-frame points to map-frame points.
* **Scale.** Every response that carries map-frame metres also carries:
  * `scale_source`: one of `priors`, `depth`, `camera_height`, `monocular`;
  * `scale_sigma_rel`: the relative one-sigma uncertainty of the map's scale.

## 4. Inputs

* **Frames.** Colour images: JPEG or PNG, 8-bit RGB. Each frame names its camera and may carry:
  * a pose prior;
  * a depth image: 16-bit PNG, depth along the optical axis, scale given per camera, 0 = no
    value.
* **Cameras.** Each camera is described once per map and can be referenced by id afterwards:
  * intrinsics `fx, fy, cx, cy` in pixels, and image size;
  * distortion: the OpenCV model, or none;
  * optional ignore mask, for pixels showing the robot itself;
  * optional height of the optical centre above the floor, with a sigma: the scale anchor for
    robots without metric priors. A frame may carry its own height, with a sigma, which
    overrides the camera's: a camera on a pan-tilt head or a wrist moves with the robot's joints.
* **Pose priors.** `T_map_cam` with a trust level:
  * `metric` — an absolute pose in the map frame, with position and rotation sigmas. For
    example, odometry with a known start, or kinematics of a fixed arm.
  * `drifting` — reliable relative motion between frames of one request, with drift over time.
  * `none` — no prior. This is the case for camera-only robots.

  Odometry, depth and joint kinematics reach oh-my-slam only through priors and depth. There is
  no IMU, stereo or LiDAR input.
* There is no video input. The client samples its own frames.

## 5. Map model and lifecycle

A map is created by its first committed update. Every commit after that produces an immutable,
integer **version**.
* Reads are atomic: each answer comes from one version, which it names. A read may ask for a
  past version that is still kept.
* An update runs as an asynchronous **job** in two phases:
  * **registration** publishes the new frames with provisional poses within seconds;
  * **fusion** then fuses their depth and objects.

  Each phase that succeeds commits a version.
* An update can be checked first (dry run), gated (`accept_if`) and undone (revert).
* A map is deleted only by its delete request.

The details are in [specs/map.md](specs/map.md).

## 6. HTTP API

One long-lived service, `server.sh`, serves the whole API under `/api/`. [specs/api.md](specs/api.md)
defines it, and `/api/openapi.json` describes it.
* It listens on a fixed default port, **9091**.
* It is reachable from the LAN and from Docker containers using host networking.
* It owns no model: the inference server (§8) does.

## 7. Output schemas

Every JSON body follows [specs/api.schema.json](specs/api.schema.json), a flat, documented JSON
Schema (draft 2020-12). Each response names its schema version in `schema`. A change that
removes or renames a field, or changes its meaning, raises the major version. The skill's client
is written against this schema. OpenLABEL is no longer produced.

## 8. Concurrency and latency

* **Reads.** Reads never wait for an update. These are map summary, objects, grid, frames,
  bird's-eye image, last pose, versions and jobs.
* **Interactive work.** Locate and frame analysis run in an interactive lane that never waits
  behind an update's work. A job's registration phase also runs there.
* **Bulk work.** Fusion runs in a bulk lane.
* **Ordering.** One job writes a given map at a time. A map's further jobs queue in arrival
  order.
* **Disconnects.** A job continues when its client disconnects. Only a cancel request or a stop
  of the service ends it.
* **Latency targets** (specs/evaluation.md):
  * a frame is readable in a map version at most 2 s after its upload;
  * fusion takes at most 3 s per new frame, for maps up to 500 keyframes.

## 9. Uncertainty

Every estimate the skill acts on carries its uncertainty in the same response:
* camera poses: position and rotation sigmas;
* a located camera: also its ambiguity, the distance to the second-best hypothesis;
* object centres: a sigma, and whether each extent was seen in full or partially;
* the map scale: its source and relative sigma;
* grid cells: an observation count, with the unknown state kept distinct from free.

## 10. Constraints

* **Ownership.** Each concern has a single owner module, and none of it is reimplemented
  elsewhere:
  * depth and point-cloud generation;
  * segmentation, box fitting and colour;
  * mapping (registration, fusion, objects, grid, versions);
  * the HTTP layer.

  The ownership is enforced by import contracts.
* **Code.** Shared logic lives in one package. Any language or runtime that serves the
  targets is allowed, CUDA included. Models adapted for another runtime are third-party
  components like any other.
* **Platforms.**
  * The service and its inference must run on the target machine, an Apple M4 Max Mac with
    36 GB of unified memory, where performance is measured.
  * A CUDA host may also run them.
  * Smaller Macs are best-effort.
* **Errors.** Each error carries a typed, machine-readable code that says whether a retry can
  succeed (api.md §8). An input that cannot be used, such as a frame that cannot be located, is
  a per-input status in a successful response, never an error.
* **Tests.** Every module is covered by offline unit tests: no network, no model weights, no
  running inference server, with models replaced by test doubles. The build fails below 100 %
  line and branch coverage of the whole package, models and evaluators included.
* **Licences.** Licences are not a blocker: the project is for personal and research use.
  Every third-party component and its licence is listed in `THIRD_PARTY_LICENSES.md`.

## 11. Evaluation

A single-command evaluator replays recorded robot-guide episodes, from the base, the humanoid
and the arm, against the simulator's ground truth. It also runs on the reference captures in
`examples/` (`office_sequence/`, `ainex-captures/`, `camera/`).
* It measures each client-contract row (§2).
* Its targets are data in `examples/targets.json`.
* It writes a machine-readable result and a summary outside the repository, and compares each
  run with a stored baseline.

The details are in [specs/evaluation.md](specs/evaluation.md).

## 12. Removed features

These features are removed because the client never uses them. Across all recorded episodes the
calls were:

| Call | Count |
| --- | --- |
| update JSON | 158 |
| locate PNG | 138 |
| segment JSON | 34 |
| locate JSON | 15 |
| locate PLY | 6 |
| reconstruct PLY | 6 |
| reconstruct JSON | 5 |

Last-known camera, pixel-to-map, layers and video were never used.

* The public shell entry points `reconstruct.sh`, `segment.sh`, `mapper.sh`, `view.sh` and
  `start_inference_server.sh`. The inference server is managed by `server.sh` (inference.md).
* The rule that the HTTP API mirrors the command line byte for byte.
* The web application and the 3D viewer.
* Video input and frame sampling (`-fps`).
* PLY and PNG variants of update and locate, the bird's-eye layers option, and the "last known
  camera". Two things remain: the bird's-eye image, as a versioned read, and the last
  *recorded* pose (api.md §6).
* The map list, and `agent_skill.md`. `SKILL.md` becomes a short API reference.
* The rule to ask the user before updating an existing map.
* The outdoor and video examples (`street2.mp4`).
* OpenLABEL output. It is replaced by the schema of §7.
* The single-image colour contract across artefacts. An object's `color` stays a display colour,
  a pure function of its id. Its observed appearance is `mean_rgb` (map.md §5).
