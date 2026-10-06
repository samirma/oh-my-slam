# Monocular RGB Mapping — High-Level Specification

## 1. Goal

Build an application that reconstructs 3D point clouds from **RGB-only** images and
video, incrementally assembles them into a persistent map that can be inspected in a
browser, and describes that map as a set of **labelled objects with oriented bounding
boxes (OBBs)**.

The system is exposed through six shell entry points and an agent skill:

| Entry point | Responsibility |
| --- | --- |
| `start_inference_server.sh` | Start the depth- and segmentation-inference server and any other long-lived services required for mapping. |
| `reconstruct.sh` | Single-frame reconstruction: image → scene description (JSON + OBBs) or point cloud. |
| `mapper.sh` | Multi-frame mapping: build and update a persistent map, and locate an image's camera in it. |
| `segment.sh` | Instance segmentation: image or map → JSON + OBBs, a colour-coded segmented image, and an object catalogue. |
| `view.sh` | Browser visualisation of either a single image reconstruction or a persisted map. |
| `server.sh` | Web service: an HTTP API and a browser application giving access to every feature of `reconstruct.sh`, `mapper.sh`, `segment.sh` and `view.sh`. |
| `SKILL.md` agent skill (`oh-my-slam-api`) | Lets an AI agent use every endpoint of the `server.sh` API with `curl`, from this Mac or any machine on the LAN (§2.7). |

This document states requirements only. Detailed decisions — defaults, coordinate
conventions, exit codes, the OpenLABEL field mapping, the colour palette — are recorded in
`README.md` and in the implementation plan (`~/.dev-workflow/oh-my-slam.md`), and must not
contradict this document.

## 2. Components

Each entry point is specified in its own file:

| Section | Entry point | Specification |
| --- | --- | --- |
| §2.1 | `start_inference_server.sh` | [Inference server](start_inference_server.md) |
| §2.2 | `reconstruct.sh` | [Single-frame reconstruction](reconstruct.md) |
| §2.3 | `mapper.sh` | [Mapping](mapper.md) |
| §2.4 | `segment.sh` | [Segmentation](segment.md) |
| §2.5 | `view.sh` | [Visualisation](view.md) |
| §2.6 | `server.sh` | [Web service](http_server.md) |
| §2.7 | `SKILL.md` | [Agent skill](agent_skill.md) |

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
* `examples/office_sequence/` — an ordered 13-image sequence (file names are capture
  timestamps) of an office scene that changes during the capture: a cup visible in the
  first images is gone in the last ones. Reference input for the map-update behaviour of
  [§2.3](mapper.md): mapping the whole sequence must produce a map that reflects the latest
  observation — without the cup.

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
* **Map update** — on `examples/office_sequence/`, the final map reflects the latest
  observation: the cup visible in the first images is absent from the map after the full
  sequence is mapped, while objects that never changed keep their `id`s, labels and OBBs.
* **Segmentation** — detections, labels and scores per frame (`restaurant.jpg` and each
  `ainex-captures` frame), compared with the map's objects for the frames that observe them.
* **Contracts**, for every command — colour contract, OpenLABEL validity, stdout purity.

Evaluators run as a single command, write a machine-readable result (JSON) and a
human-readable summary, and store results outside the repository so runs can be compared
over time. Each metric has a target and a pass/fail result, and each run is also compared
with a stored baseline run so that regressions are flagged. Targets and the baseline are
data read by the evaluators, so they change without code changes. Ground-truth annotations
added later for the example files must be picked up without changing evaluator code.
