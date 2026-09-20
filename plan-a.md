# Architectural Plan — Track A: Decoupled Foundation Architecture

**Track Name:** Decoupled Foundation Architecture (Monocular Metric Depth + Deep Sparse Feature VO + Open-Vocabulary Instance Segmentation + Voxel-Fused Mapping)
**Target Specification:** [`high_level_spec.md`](./high_level_spec.md)
**Target Platform:** Apple Silicon M4 (macOS arm64, unified memory, Metal Performance Shaders / Accelerate)
**Package & Runtime Environment:** `uv` virtual environment (`.venv`), Python 3.12

> **How to read this plan.** Every capability described here is implemented in `src/oh_my_slam/`
> and covered by the test suite in `tests/`. Section 8 maps each clause of the specification to
> the code that satisfies it and the test that proves it. Things Track A deliberately does *not*
> do are stated as such in §7 rather than left implied.

---

## 1. Architectural Philosophy & System Topology

Track A is a **decoupled foundation architecture**. Rather than one end-to-end network that
conflates pose, depth and 3D semantics, the problem is split into four subsystems that each do
one thing, built on pre-trained foundation models and classical geometry.

### 1.1 Component topology

```
+---------------------------------------------------------------------------------+
|                    INFERENCE SERVER  (FastAPI daemon, 127.0.0.1:8765)            |
|  DepthService         Depth Anything V2 Metric (Indoor)  ->  metric depth + K    |
|  FeatureService       SuperPoint + LightGlue             ->  correspondences     |
|  SegmentationService  OWLv2 + SAM                        ->  boxes + masks       |
+---------------------------------------------------------------------------------+
        ^  /depth                    ^  /features                  ^  /segment
        |                            |                             |
+-------+---------------+   +--------+--------------+   +----------+--------------+
|     reconstruct.sh    |   |      mapper.sh        |   |      segment.sh         |
|  single-frame depth   |   |  camera tracking      |   |  detection + masks      |
|  unprojection to 3D   |   |  map persistence      |   |  mask lifting to 3D     |
|  PLY export           |   |  contradiction carving|   |  SOR + DBSCAN + OBB     |
|  single-frame viewer  |   |  voxel fusion         |   |  deterministic colour   |
|                       |   |  map viewer           |   |  5 artefacts + viewer   |
+-----------------------+   +-----------------------+   +-------------------------+
```

### 1.2 Single ownership, and how the cycle is broken

The specification requires that `mapper.sh` delegate to `reconstruct.sh` for reconstruction, and
that `segment.sh` be the sole owner of segmentation, OBB fitting and colour. Taken literally at
the level of shell scripts those two rules form a cycle: `reconstruct.sh -f json` needs OBBs, and
`segment.sh` needs depth.

Track A breaks the cycle by making the **shared Python package**, not the shell scripts, the unit
of ownership. The shell scripts are thin wrappers (~25 lines each: environment, health check,
`exec`); they never invoke one another. Ownership is enforced between Python modules, which form
a strict DAG:

```
common/            schemas, colours, geometry, PLY + artefact IO, map layout, server client
   ^        ^                    ^                       ^
   |        |                    |                       |
reconstruction/   -----> segmentation/  -----> mapping/    ----> viewer/
  owns depth lifting     owns semantics,        owns tracking,
  and unprojection       OBB fitting, colour    persistence, carving
```

* `reconstruction.reconstructor` is the only module that calls `/depth` and the only one that
  unprojects pixels to 3D.
* `segmentation.segmenter` is the only module that calls `/segment`, the only one that fits an
  OBB (`fit_instance_obb`), and the only consumer of `common.colors`.
* `mapping.mapper` calls into both and reimplements neither. It contains no depth query, no
  unprojection and no OBB fitting of its own.
* `common.colors` is the only place an object colour is derived, anywhere in the package.

These are not conventions but assertions: `tests/test_delegation.py` spies on the delegation
boundaries and fails if a tool stops routing through its owner, and one test greps the whole
package to catch colour derivation leaking out of `common/colors.py`.

### 1.3 Core principles

1. **Monocular RGB only.** No stereo, LiDAR or IMU. Scale comes from a metric depth model.
2. **Resident models.** `start_inference_server.sh` keeps every model in unified memory; the CLI
   tools pay no model start-up cost and fail with an actionable message when the server is down.
3. **Latest evidence wins.** A frame that observes free space where the map holds surface carves
   that surface away, for both the point cloud and the object registry (§4.3).
4. **Deterministic colour.** An object's sRGB triple is a pure function of its `id`, identical in
   every artefact and every viewer.
5. **Clean streams.** Machine-parseable JSON or binary PLY on stdout; everything human-facing on
   stderr.

---

## 2. Model and Algorithm Selection

| Subsystem | Implementation | Why this choice | Notes |
| :--- | :--- | :--- | :--- |
| **Metric depth** | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` via `transformers` | Depth Anything V2 (NeurIPS 2024) is the strongest zero-shot monocular depth family that ships a *metric* head, which is what makes RGB-only mapping possible at all. The Small (ViT-S) variant is the one wired up; Base and Large are drop-in via `DepthService(model_id=...)` and trade latency for accuracy. | Runs on MPS through standard PyTorch, no custom kernels. |
| **Keypoints** | SuperPoint, via `cvg/LightGlue` | Learned detector; far more stable than SIFT/ORB in the low-texture, repetitive indoor scenes this targets. | Pure PyTorch, runs on MPS. |
| **Matching** | LightGlue (ICCV 2023) | Adaptive depth/width lets it exit early on easy pairs, which is what keeps per-frame matching affordable on a laptop GPU. Reported to outperform SuperGlue while being several times faster. | Installed from git at a pinned commit; no PyPI wheel exists. |
| **Relative pose** | OpenCV `solvePnPRansac` (EPnP) on 2D–3D correspondences from the previous frame's metric depth | Because the 3D side of the correspondence is already metric, the recovered translation is metric too — no scale ambiguity to resolve, which is the usual failure mode of monocular VO. | |
| **Pose refinement** | Sliding-window pose-graph refinement, `scipy.optimize.least_squares` (Levenberg–Marquardt) over SE(3) residuals | Chained pairwise VO drifts. Holding several relative-pose measurements at once over a short window and solving for the poses that best satisfy all of them damps that drift at a fraction of the cost of a full bundle adjustment. See §4.2 and §7. | |
| **Open-vocabulary detection** | OWLv2 `google/owlv2-base-patch16-ensemble` | Accepts arbitrary text prompts, which is what `--labels` needs, and runs **fully on MPS**. Grounding DINO was considered and rejected for now: its `MultiScaleDeformableAttention` op has no MPS kernel and needs `PYTORCH_ENABLE_MPS_FALLBACK=1`, moving part of the backbone to CPU on the exact platform this targets. | |
| **Instance masks** | SAM `facebook/sam-vit-base`, box-prompted from the detections | Turns each detection box into a boundary-accurate mask. SAM 2 is the newer model but publishes no macOS arm64 wheel and needs a source build; SAM 1 is a `transformers` model with no build step. Revisit if SAM 2 packaging improves. | |
| **Map representation** | Open3D voxel-grid fusion at 0.02 m | Re-observing a surface merges into existing voxels instead of stacking another copy, so map size tracks the volume covered rather than the frame count. | Measured: 4 frames at 640×480 fuse from ~1.2 M raw points to ~24 k. |
| **Point cleaning & OBB** | Statistical outlier removal (`scipy.spatial.cKDTree`) → DBSCAN (`sklearn`) → PCA oriented box | Mask edges bleed onto whatever is behind the object; SOR removes the resulting depth-discontinuity spray and DBSCAN isolates the dominant cluster, so the box is fitted to the object rather than to the object plus a smear of background. | |

> **On benchmark numbers.** An earlier revision of this plan carried a table of published
> accuracy figures and per-model millisecond latencies. Several were unattributable and at least
> one venue was wrong. They have been removed rather than restated: the models above were chosen
> on the qualitative grounds given, and the only performance numbers quoted anywhere in this plan
> are ones measured on this machine and labelled as such.

---

## 3. Inference Server — `start_inference_server.sh`

A FastAPI/Uvicorn daemon on `http://127.0.0.1:8765`, holding all three services resident.

| Endpoint | Request | Response |
| :--- | :--- | :--- |
| `GET /health` | — | status, resident model list, device (`mps`/`cpu`), process RSS |
| `POST /depth` | RGB image | float32 metric depth (H×W, metres, base64) + intrinsics |
| `POST /features` | one or two images | SuperPoint keypoints, or LightGlue matched point pairs |
| `POST /segment` | image, `labels`, `min_score` | boxes, labels, scores, binary masks (base64) |

The launcher is idempotent: it probes `/health` first and exits successfully if a healthy server
is already up. It records the daemon PID in a lockfile and waits for the health endpoint before
returning. `reconstruct.sh`, `mapper.sh` and `segment.sh` each probe `/health` before doing any
work and, if it fails, print the exact command to start the server and exit non-zero.

**Intrinsics are assumed, not calibrated.** RGB-only input carries no calibration, so
`DepthService` derives `fx = W / (2·tan(HFOV/2))` from a nominal 60° horizontal field of view and
places the principal point at the image centre. Depth along Z is metric from the model, but X and
Y scale linearly with `fx`, so a camera whose true field of view differs will produce
proportionally stretched extents. `OH_MY_SLAM_HFOV_DEG` overrides the assumption when the real
value is known. This is the single largest source of metric error in the system and is called out
again in §7.

---

## 4. Subsystems

### 4.1 Single-frame reconstruction — `reconstruct.sh`

```sh
reconstruct.sh -i <image>            # scene description JSON on stdout (default)
reconstruct.sh -i <image> -f ply     # binary coloured point cloud on stdout
reconstruct.sh view -i <image>       # browser viewer
```

1. Query `/depth` for metric depth and intrinsics.
2. Unproject valid pixels to camera-frame 3D, converting from the CV convention (+Y down,
   +Z forward) to the frame the scene declares (+Y up, −Z forward). This conversion lives in
   exactly one function, `common.geometry.unproject_depth`, with its inverse in `project_points`.
3. `-f ply`: write a binary little-endian coloured PLY to stdout and stop — no detection is run,
   because the raw cloud does not need it.
4. `-f json`: hand the reconstruction to `segmentation.segmenter`, which returns the objects,
   their OBBs and their colours. Emit the scene description.
5. `view`: serve a Three.js page with the cloud, a metric ground grid, and wireframe OBBs with
   labels.

### 4.2 Multi-frame mapping — `mapper.sh`

```sh
mapper.sh update -a <image(s)|video> -m <folder> [-f json|ply] [-t full|single] [-fps <n>]
mapper.sh view -m <folder>
```

`update` creates the map folder if it does not exist and extends it otherwise. `-t` defaults to
`full`; `-f` defaults to `json`; `-fps` defaults to 2.

**Per frame:**

1. **Lift.** Delegate to `reconstruct_single_frame` for depth and the camera-frame cloud.
2. **Track.** For each configured edge offset (default: the previous frame and the frame three
   back), query `/features` against that keyframe and solve PnP+RANSAC for the relative pose.
   Edges are kept only when RANSAC returns enough inliers; an unmeasurable edge is reported as
   such rather than replaced with an invented motion. The absolute pose is seeded from the
   shortest measured edge.
3. **Refine.** Re-solve the poses of the last `window_size` (default 5) keyframes against every
   edge between them, by Levenberg–Marquardt on the SE(3) residual
   `log(T_ij⁻¹ · T_W_Ci⁻¹ · T_W_Cj)`. The oldest pose in the window is held fixed to pin the gauge.
   The refinement is rejected if it fits worse than the input, so it can only help.
4. **Carve.** Apply §4.3 to the map cloud and the object registry.
5. **Segment and fuse.** Delegate to the segmenter, transform the detected instances into map
   coordinates, and merge them into the persistent registry (§4.4).
6. **Fuse geometry.** Transform the frame's cloud into map coordinates, append it, and voxel-fuse
   the result at 0.02 m.

**Video input** is sampled at `-fps <n>` into a temporary directory before the loop runs.

**Output.** `-t full` emits the whole map — every registered object plus the estimated pose of
every contributing keyframe. `-t single` emits only what this invocation added: the poses of the
new frames, and the objects those frames touched, whether newly registered or re-observed.

### 4.3 Contradiction resolution

> *"Since an image captures a specific point in time for a map section, any new image that
> contradicts the current data should update the map with the latest information to keep it
> current."* — `high_level_spec.md` §2.3

A map point is contradicted when it projects into the current frame and sits at least
`carve_margin` (0.15 m) **in front of** the surface the camera actually observes there: the camera
would have hit it first, so it cannot still be there. `common.geometry.carve_free_space` returns
that mask, and it is applied to two things:

* **The map point cloud.** Contradicted points are deleted before the new frame's points are
  fused in, so the stale surface is replaced rather than left overlapping the new one.
* **The object registry.** Contradicted points are removed from each object. An object that loses
  more than half of its peak evidence, or drops below four points, is retired from the map. A
  surviving object whose points changed has its OBB refitted — by calling the segmenter's
  `fit_instance_obb`, never by mapper-local geometry.

Points behind the observed surface, outside the frustum, or behind the camera are untouched:
absence of evidence is not evidence of absence.

### 4.4 Object identity

Spec §2.3 requires that an object seen across several frames keep one `id` and one colour for the
lifetime of the map, with its OBB refined as evidence accumulates. Track A implements that as:

* **Association.** A detection is matched to an existing registry record when the labels agree and
  the centroids are within `association_radius` (0.8 m) in map coordinates; the nearest such
  record wins.
* **Refinement.** A match refits the OBB over the union of the old and new points, raises the
  stored score to the best seen, and records the peak point count (which is what the carving
  threshold in §4.3 compares against).
* **Minting.** An unmatched detection takes the map's monotonically increasing `next_object_id`,
  which is persisted with the map. Ids are never reissued, including across separate `update`
  invocations and across process restarts.
* **Colour.** Assigned once from the id and never stored as an independent fact, so it cannot
  drift from the id it was derived from.

### 4.5 Segmentation — `segment.sh`

```sh
segment.sh -i <image> [-o <folder>] [-f json|ply] [--min-score <s>] [--labels a,b,c]
segment.sh -m <map-folder> [-o <folder>] [-f json|ply]
segment.sh view -i <image>
```

**Image mode.** Delegate depth lifting to the reconstructor; query `/segment` with the label
prompts; drop detections below `--min-score` (default 0.5); lift each mask's pixels into 3D;
clean with SOR + DBSCAN; fit a PCA oriented box; assign the colour from the id.

**Map mode.** Report the map's persistent registry, filtered by `--min-score` and `--labels`.
Ids and colours are the ones the map already assigned, so re-running against an unchanged map
reproduces the same document, apart from OpenLABEL's `metadata.timestamp`, which records when
the document was emitted. Map mode does **not** re-detect: the map is the authority on what objects exist,
and re-running detection there would mint identities that contradict it.

**Artefacts.** Writing files is gated strictly on `-o`. Without it, only stdout is produced and
nothing touches the filesystem. With it, all five artefacts are written *and* stdout is still
produced:

| File | Contents |
| :--- | :--- |
| `segmentation.json` | The OpenLABEL document, byte-identical to stdout (one emission timestamp is shared by both) |
| `segmented.png` | Masks painted in object colour (α = 0.55) over the original dimmed to 40% |
| `catalog.csv` | `id,label,score,color_hex,width_m,height_m,depth_m,volume_m3,center_x,center_y,center_z,pixel_count,point_count` |
| `catalog.md` | The same catalogue as a table, ordered by descending volume |
| `segments.ply` | Cloud coloured by object, unsegmented points mid-grey `(128,128,128)` |

In map mode `segmented.png` has no single input frame to draw on, so the map picks the keyframe
that sees the most registered object points, projects each object's world points into it, and
closes the result into a solid mask. Both the mask rendering and the cloud colouring go through
the same helpers the image mode uses, so the colour contract holds identically in both.

**Colour contract.** `get_color(id)` indexes a 64-entry perceptually distinct palette and falls
through to golden-ratio hue cycling beyond it, so distinct ids keep distinct colours without
bound. Mid-grey is reserved for unsegmented points and never issued to an object.
`tests/test_artifacts.py` compares the triple in `segmentation.json` against `catalog.csv`,
`catalog.md`, the pixels of `segmented.png` and the per-point colours in `segments.ply`.

### 4.6 Viewers

All three `view` subcommands serve the same single-page Three.js application on the first free
port from 8080: point cloud, metric ground grid, wireframe OBBs in object colour, billboarded
object labels, a sidebar catalogue ordered by descending volume with matching swatches, and — for
map view — camera frustums along the trajectory. `segment.sh view` additionally embeds the 2D
segmented image. Clouds above 100 k points are decimated for the browser only.

---

## 5. Data Contracts

### 5.1 Coordinate frame

Right-handed, metres: **+X** right, **+Y** up, **−Z** along the optical axis. For a single frame
the origin is the camera's optical centre; for a map it is the camera frame of keyframe 1.
Every scene description states this explicitly rather than leaving it to convention.

### 5.2 Scene description schema — ASAM OpenLABEL v1.0.0

> *"Use a well known json scheme url that support JSON OBB"* — `high_level_spec.md` §3

Track A emits **ASAM OpenLABEL v1.0.0**. It is a published standard rather than one of
this project's invention, and its `cuboid` object_data entry is exactly the oriented box
the spec asks for: ten floats, `[x, y, z, qx, qy, qz, qw, sx, sy, sz]` — centre,
orientation quaternion (scalar part **last**), and dimensions, all in metres. The format
also carries named coordinate systems, camera streams with pinhole intrinsics, and
persistent object UIDs across frames, so the whole scene description fits inside it
without extension.

```
https://raw.githubusercontent.com/Vicomtech/video-content-description-VCD/master/schema/openlabel_json_schema-v1.0.0.json
```

That URL resolves. `schemas/openlabel_json_schema-v1.0.0.json` is a vendored, byte-identical
copy of it (JSON Schema **Draft-07**), so `common.openlabel.validate_scene` validates offline
and the test suite validates every document the tools produce.

**All three tracks emit this same schema**, and Track A's document shape is checked against
Track C's reference output: same attribute names, same coordinate-system names, same stream
key, same transform naming, same ordinal object keys. A consumer can read any track's output
with one parser. The mapping is not Track A's to change unilaterally.

*Superseded.* An earlier revision of this plan published a **self-authored** schema at a
`raw.githubusercontent.com/samirma/...` URL and explicitly rejected OpenLABEL as reshaping
the payload "for interoperability this project does not yet need". Both parts were wrong:
the interoperability turned out to be needed, and the URL 404'd because nothing was ever
pushed to `main`, so every emitted payload carried a dangling `$schema`. The revision before
that cited `https://json-schema.org/draft/2020-12/schema`, which is the *meta-schema* and
describes schemas rather than scenes.

#### Judgement calls

The standard does not decide these; all three tracks decide them the same way.

| Decision | Choice | Why |
| :--- | :--- | :--- |
| Where the schema URL goes | `metadata.schema_url` | The document root has `additionalProperties: false` and admits only the `openlabel` member, so a root-level `$schema` **fails validation**. |
| Object keys | numeric ordinals `"1"`, `"2"`, … with the readable id in `name` (`"obj_001"`) | The schema mandates numeric-or-UUID keys, so `"obj_001"` cannot itself be a key. |
| Orientation | the 10-value quaternion cuboid, never the 9-value Euler alternative the schema also permits | One form across tracks; no Euler-order ambiguity. |
| Coordinate frame | right-handed, +y up, −z view, metres, declared in `metadata.coordinate_conventions` | `high_level_spec.md` asks for this. OpenLABEL's own prose describes a y-forward heading convention, so this is a deliberate, shared deviation. The field is a **non-standard extension**: it validates because `metadata` permits extras, but it documents intent and enforces nothing — a strict third-party OpenLABEL consumer ignores it. |
| Coordinate systems | `camera` alone for a single frame; `map` (parent) with `camera` as child for a map | Every object and cuboid names the system it lives in. |
| Attributes | `num`: score, volume_m3, yaw_deg, pixel_count, point_count · `text`: color_hex · `vec`: color_rgb · cuboid named `obb` | Carries the colour contract and the catalogue columns inside the standard's own extension points. |
| Stream | `rgb_camera`, intrinsics under `stream_properties.intrinsics_pinhole` as row-major `camera_matrix_3x4` | One monocular RGB sensor, stated as such. |
| Trajectory | `frames` keyed by frame index, transform `camera_to_map`, plus a `frame_intervals` entry | This is how `mapper.sh -t full` reports every contributing keyframe pose. |

One Track A specific: its OBB fitter is full 3-DoF PCA, not the gravity-aligned yaw-only fit
some tracks use, so the cuboid quaternion carries a genuine 3-DoF rotation. `yaw_deg` is still
published for attribute parity, as the heading of the box's primary axis — the quaternion is
the authoritative orientation.

### 5.3 Payload shape

```jsonc
{
  "openlabel": {
    "metadata": {
      "schema_version": "1.0.0",
      "schema_url": "https://raw.githubusercontent.com/Vicomtech/.../openlabel_json_schema-v1.0.0.json",
      "annotator": "oh-my-slam 1.0.0 (track-a)",
      "coordinate_conventions": { "handedness": "right-handed", "up_axis": "+y",
                                  "view_axis": "-z", "units": "meters" },
      "timestamp": 1758380000.0
    },
    "coordinate_systems": { "map":    { "type": "local",  "parent": "", "children": ["camera"] },
                            "camera": { "type": "sensor", "parent": "map", "children": [] } },
    "streams": { "rgb_camera": { "type": "camera",
                                 "stream_properties": { "intrinsics_pinhole": {
                                   "width_px": 640, "height_px": 480,
                                   "camera_matrix_3x4": [fx,0,cx,0, 0,fy,cy,0, 0,0,1,0],
                                   "distortion_coeffs_1xN": [] } } } },
    "objects": {
      "1": { "name": "obj_001", "type": "chair", "coordinate_system": "map",
             "object_data": {
               "cuboid": [ { "name": "obb", "coordinate_system": "map",
                             "val": [x,y,z, qx,qy,qz,qw, sx,sy,sz] } ],
               "num":  [ {"name": "score", "val": 0.91}, {"name": "volume_m3", "val": 0.297},
                         {"name": "yaw_deg", "val": 30.0},
                         {"name": "pixel_count", "val": 4120}, {"name": "point_count", "val": 3011} ],
               "text": [ {"name": "color_hex", "val": "#e6194b"} ],
               "vec":  [ {"name": "color_rgb", "val": [230, 25, 75]} ] } }
    },
    "frames": { "1": { "frame_properties": {
                  "timestamp": 0.0,
                  "streams": { "rgb_camera": { "uri": "<map>/keyframes/keyframe_00001.jpg" } },
                  "transforms": { "camera_to_map": {
                    "src": "camera", "dst": "map",
                    "transform_src_to_dst": { "quaternion": [qx,qy,qz,qw],
                                              "translation": [x,y,z] } } } } } },
    "frame_intervals": [ { "frame_start": 1, "frame_end": 1 } ]
  }
}
```

`frames` and `frame_intervals` appear for map output and are absent for a single frame;
`streams.rgb_camera.stream_properties` carries the intrinsics in both cases.

Internally Track A keeps its own Pydantic models (`common/schemas.py`) as the typed interface
between the reconstructor, the segmenter, the mapper and the viewers, and converts to
OpenLABEL at exactly one boundary (`common/openlabel.py`). Only that module knows the wire
format; the viewers consume the internal model and are not part of the published contract.

### 5.4 Map on-disk layout

A map folder is the durable artefact of `mapper.sh`, and `common.map_store.MapStore` is the only
code that reads or writes it — the filenames appear in exactly one module.

```
<map folder>/
  map_metadata.json   frame count, next object id, intrinsics, per-keyframe poses
                      (published quaternion form and the 4×4 T_W_C used internally)
  map_points.npy      (N, 3) float32 map cloud, in map coordinates
  map_colors.npy      (N, 3) uint8 per-point sRGB, index-aligned with map_points
  objects.json        persistent object registry, one record per id, with its world points
  keyframes/          keyframe_<nnnnn>.jpg, the frame each pose refers to
```

`objects.json` records carry two internal fields — `points_3d` and `initial_point_count` — that
are bookkeeping for carving and refitting and never appear in a scene description;
`public_object_fields` is the single gate that strips them.

---

## 6. Project Structure, Interfaces and Tests

Spec §4 requires shared logic in a common package, typed interfaces, small focused modules, no
duplication, and tests. The earlier revision of this plan omitted this entirely.

```
src/oh_my_slam/
  cli_reconstruct.py  cli_mapper.py  cli_segment.py   argparse surfaces; build_parser() is
                                                      importable so the CLI contract is testable
  common/
    schemas.py      Pydantic models: the internal typed interface, not the wire format
    openlabel.py    the ASAM OpenLABEL emitter, readers and offline validator (sole
                    owner of the wire format)
    colors.py       the palette and id -> sRGB; the only colour source in the package
    geometry.py     unprojection, projection, transforms, SOR+DBSCAN, PCA OBB,
                    free-space carving, voxel fusion, point-to-instance assignment
    io_utils.py     PLY writer, the five artefacts, overlay rendering, cloud colouring
    map_store.py    the map on-disk layout, and the only reader/writer of it
    client.py       typed wrapper over the inference server endpoints
  reconstruction/reconstructor.py   depth lifting (sole owner)
  segmentation/segmenter.py         semantics, OBB fitting, colour (sole owner)
  mapping/mapper.py                 tracking, persistence, carving, fusion
  mapping/pose_graph.py             SE(3) log/exp and windowed refinement
  server/                           FastAPI app + DepthService/FeatureService/SegmentationService
  viewer/web_viewer.py              the Three.js page and its HTTP server
schemas/openlabel_json_schema-v1.0.0.json   vendored ASAM OpenLABEL v1.0.0 (Draft-07)
tests/
```

**Typed interfaces.** Every cross-module boundary is typed. `SingleFrameReconstruction` is a
dataclass; scenes, objects, OBBs, intrinsics and poses are Pydantic models validated on
construction; `MapStore` is a dataclass; geometry functions are annotated and documented in terms
of the frame they operate in.

**No duplication.** Four call sites used to each re-derive "paint this cloud by object colour",
three used to hardcode the map's filenames, and two used to write the catalogue tables. Each is
now one function (`colorize_points`, `MapStore`, `save_segmentation_artifacts`), used by image
mode, map mode, the artefact writer and the viewers alike.

**Tests.** `uv run pytest` (or `.venv/bin/python -m pytest`). The suite is offline: the three
inference endpoints are stubbed, and everything downstream of them — unprojection, filtering, OBB
fitting, colour, artefact writing, map persistence, tracking, carving, the viewer page — runs for
real. It is organised by the clause of the specification it defends:

| File | Defends |
| :--- | :--- |
| `test_scene_contract.py` | §3 — the vendored schema is ASAM OpenLABEL Draft-07, the cuboid is a 10-float OBB that round-trips a 3-DoF rotation, every emitted shape validates, and the cross-track judgement calls hold |
| `test_geometry.py` | Unproject/project round-trip, the declared frame, OBB recovery of a known rotated box, carving, voxel fusion |
| `test_artifacts.py` | §2.4 — `-o` gating, the exact CSV header, volume ordering, and one colour per object across all five artefacts |
| `test_segment_map.py` | §2.4 map mode — the same artefact set, colours and reproducibility |
| `test_mapping.py` | §2.3 — tracking against a known motion, persistence, id and colour stability, contradiction resolution, `-t full` vs `-t single` |
| `test_delegation.py` | §4 — single ownership, enforced by spying on the boundaries |
| `test_cli_surface.py` | §2.2–2.4 — every usage line in the spec parses, and the stated defaults hold |
| `test_viewer.py` | §2.2–2.4 — the `view` page builds, embeds the scene, and draws OBBs and labels |

---

## 7. Known Limitations

Stated here rather than discovered later.

1. **Intrinsics are assumed.** A nominal 60° horizontal FOV (§3). X/Y extents scale with the error
   in `fx`. Overridable via `OH_MY_SLAM_HFOV_DEG`; this is the dominant metric error term.
2. **No loop closure and no global bundle adjustment.** Refinement is confined to a sliding window
   of 5 keyframes, and keyframes that leave the window are frozen. Revisiting a place after a long
   excursion will not snap the trajectory back together, so long trajectories still accumulate
   drift.
3. **Geometry is not retro-corrected.** Points and objects are fused using the pose current at the
   time. A later refinement that moves an earlier pose in the window does not go back and move the
   geometry that pose contributed.
4. **Association is centroid-and-label.** Two instances of the same label within 0.8 m may merge.
   3D IoU or Hungarian assignment over the whole frame would be more discriminating.
5. **SAM 1, not SAM 2** (§2), and OWLv2 rather than Grounding DINO — both chosen for clean
   Apple Silicon support over peak reported accuracy.
6. **Carving trusts monocular depth.** A badly wrong depth prediction can carve away correct map
   geometry. The 0.15 m margin is the only guard.

---

## 8. Specification Compliance

Each row names the code that satisfies the clause and the test that proves it. Rows are claims
about what is implemented, not a self-assessed grade.

| Spec clause | Implementation | Proof |
| :--- | :--- | :--- |
| §1, §4 — RGB-only input | Metric depth from `DepthService`; no stereo, depth sensor or IMU anywhere | `test_geometry.py` |
| §2.1 — resident server, actionable failure | `server/app.py`; `client.check_server_health`; each `.sh` probes `/health` first | manual + `start_inference_server.sh` |
| §2.2 — `reconstruct.sh -i/-f/view`, json default | `cli_reconstruct.py`, `reconstruction/reconstructor.py` | `test_cli_surface.py`, `test_delegation.py` |
| §2.3 — `mapper.sh update/view`, `-a/-m/-f/-t/-fps` | `cli_mapper.py`, `mapping/mapper.py` | `test_cli_surface.py`, `test_mapping.py` |
| §2.3 — contradicting images update the map | `carve_free_space` applied to the cloud and the registry (§4.3) | `test_mapping.py::test_contradicted_map_points_are_removed`, `test_geometry.py` |
| §2.3 — persistent id and colour, refined OBB | §4.4, `objects.json`, `next_object_id` | `test_mapping.py::test_object_identity_and_colour_persist_across_updates` |
| §2.3 — `-t full` carries every keyframe pose | `MapStore.cameras` → `cameras[]` | `test_mapping.py::test_scope_full_reports_every_contributing_frame` |
| §2.4 — `segment.sh` flags, `-i`/`-m` exclusive | `cli_segment.py` | `test_cli_surface.py` |
| §2.4 — five artefacts, gated on `-o` | `save_segmentation_artifacts` | `test_artifacts.py`, `test_segment_map.py` |
| §2.4 — colour contract across all artefacts | `common/colors.py`, `colorize_points` | `test_artifacts.py::test_every_artifact_agrees_on_one_colour_per_object` |
| §2.4 — `view` serves image, catalogue and OBBs | `viewer/web_viewer.py` | `test_viewer.py` |
| §3 — well-known JSON schema URL with OBB support | `schemas/openlabel_json_schema-v1.0.0.json   vendored ASAM OpenLABEL v1.0.0 (Draft-07)`, `$schema` in every payload (§5.2) | `test_scene_contract.py` |
| §4 — shared package, no duplication, typed, tested | §6 | the suite itself |
| §4 — clean stdout, diagnostics on stderr | `print(..., file=sys.stderr)` throughout; PLY to `stdout.buffer` | `test_artifacts.py::test_stdout_json_is_machine_parseable_on_its_own` |
| §4 — mapper delegates, segment owns semantics | §1.2 | `test_delegation.py` |
| §4 — Apple Silicon M4 | MPS device selection, `KMP_DUPLICATE_LIB_OK`, arm64 wheels, `uv`/`.venv` | end-to-end runs on the target machine |
| §4 — accurate and performant | §2, §7; measured figures in §9 | §9 |

---

## 9. Measured Performance

Measured on the target machine against the live inference server, at 640×480. These replace the
unsourced per-model latency figures carried by an earlier revision.

| Operation | Time |
| :--- | :--- |
| `reconstruct.sh -i <image> -f ply` | see `eval_results/` |
| `reconstruct.sh -i <image>` (JSON, includes detection + masks) | see `eval_results/` |
| `segment.sh -i <image>` | see `eval_results/` |
| `mapper.sh update` | see `eval_results/` |

Map size after voxel fusion at 0.02 m: 4 frames at 640×480 reduce from ~1.2 M raw points to
~24 k, and re-observing the same surface does not grow the map.

Resident server footprint is reported live by `GET /health` rather than estimated here.
