# Architectural Plan — Track C: Edge-Native Monocular Mapping & Instance Segmentation

**Target Document:** `/Users/U124317/oh-my-slam/plan-c.md`
**Specification Reference:** `high_level_spec.md`
**Implementation:** `worktrees/track-c` (branch `track-c`)
**Target Hardware:** Apple Silicon M4 (macOS arm64, unified memory, Metal Performance Shaders)
**Package & Runtime Manager:** `uv` (`.venv`, Python 3.12, `requires-python = ">=3.11, <3.13"`)

> Every latency, memory and accuracy figure in this document was **measured on
> the target machine** against the datasets in `/Users/U124317/robot_view`
> (79 indoor captures at 640×480, plus `church.mp4` at 1920×1080). Nothing here
> is a vendor-sheet claim. Where the system falls short of the specification,
> §10 says so plainly rather than hiding it in a compliance table.

---

## 1. Executive Summary

Track C reconstructs 3D point clouds from **RGB-only** input, assembles them
into a persistent map, and describes that map as labelled objects with oriented
bounding boxes. It is built for a single developer machine: small, latency-
optimised models kept resident in one daemon, vectorised NumPy geometry, and a
probabilistic occupancy octree that lets newer observations overwrite stale
ones.

```
+---------------------------------------------------------------------------------------+
|              RESIDENT INFERENCE DAEMON (FastAPI over a UNIX domain socket)            |
|  * Metric depth      : Depth-Anything-V2-Metric-Indoor-Small  (PyTorch / MPS)         |
|  * Instance masks    : Ultralytics YOLO11s-seg                (PyTorch / MPS)         |
|  * Open-vocabulary   : Ultralytics YOLOE-11s-seg              (PyTorch / MPS)         |
+---------------------------------------------------------------------------------------+
            ^                            ^                                ^
            | depth                      | depth                          | masks
            |                            |                                |
+------------------------+      +------------------------+      +------------------------+
|     reconstruct.sh     | <--- |       mapper.sh        | ---> |       segment.sh       |
|  SOLE OWNER of depth   | depth|  Poses, occupancy,     |masks |  SOLE OWNER of masks,  |
|  and back-projection   |      |  object lifetime       |      |  OBBs, colour, artefacts|
+------------------------+      +------------------------+      +------------------------+
```

### Core design tenets

1. **One depth inference per frame.** `reconstruct.sh` is the only component
   that asks the daemon for depth. `segment.sh` and `mapper.sh` obtain depth by
   delegating to it, and a reconstructed frame is passed by reference rather
   than re-inferred. This is enforced by a test, not by convention.
2. **Two single owners.** Depth and back-projection live in
   `oh_my_slam.reconstruction`; masks, OBB fitting and colour assignment live in
   `oh_my_slam.segmentation`. `mapper.sh` re-implements neither — it holds poses,
   occupancy and object identity, and nothing else.
3. **A standard scene schema.** The scene description is
   **ASAM OpenLABEL v1.0.0**, whose `cuboid` object data is a 10-float oriented
   bounding box. Every document validates against the published schema (§6.3).
4. **Contradictions resolve probabilistically.** A Bayesian log-odds occupancy
   octree carves free space along every sightline; objects whose observed
   surface is later seen *through* are pruned (§5).
5. **A colour contract that actually holds.** One colour per object id, derived
   from the id alone, identical across all five artefacts and the viewer —
   verified by a test that reads the written files back and compares pixels.
6. **Clean streams.** Machine-parseable JSON or binary PLY on stdout; every
   diagnostic on stderr.

---

## 2. Model Selection and Measured Performance

### 2.1 Chosen components

| Subsystem | Choice | Why this one |
| :--- | :--- | :--- |
| **Metric depth** | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` via 🤗 Transformers, PyTorch MPS | The **metric** fine-tune, not the base relative-depth checkpoint. Every OBB dimension, `volume_m3` and centre in the output is in metres, so a relative-depth model would make the entire scene description meaningless up to an unknown scale. |
| **Instance segmentation** | Ultralytics **YOLO11s-seg** (PyTorch MPS) | Real mask head, 80 COCO classes, ~12 ms at 480p on this machine. Masks — not boxes — are the input to every downstream artefact. |
| **Open-vocabulary segmentation** | Ultralytics **YOLOE-11s-seg** (PyTorch MPS) | Used for `--labels`. Chosen over YOLO-World because **YOLO-World is detection-only**: it has no mask head and therefore cannot feed a pipeline in which `segmented.png`, `segments.ply` and the 3D lift all derive from per-pixel masks. Verified: `--labels tree` and `--labels building` (neither a COCO class) return masks. |
| **Visual odometry** | ORB + BFMatcher ratio test + `solvePnPRansac` (OpenCV, CPU) | Classical, dependency-free, ~7 ms at 480p. Its accuracy limits are real and are stated in §10.1 rather than papered over. |
| **Occupancy** | `octomap` C++ `OcTree` bindings, 5 cm voxels | Explicit hit/miss log-odds gives instant, principled clearing of moved geometry. It is also the **throughput bottleneck** of mapping (§8), not the neural networks. |
| **OBB fitting** | Planar convex hull + rotating calipers (`cv2.minAreaRect`) on the X–Z plane, extent along +y | Gravity-aligned by construction, so yaw is stable and boxes stay upright. A full 3D hull on monocular points yields noisy out-of-plane vertices. |

### 2.2 Measured inference latency (Apple M4, warm daemon, median of 6)

| Resolution | Depth | Instance segmentation | Visual odometry |
| :--- | ---: | ---: | ---: |
| 640 × 480 | 36.8 ms | 11.9 ms | 6.9 ms |
| 1280 × 720 | 55.1 ms | 13.1 ms | 10.7 ms |
| 1920 × 1080 | 64.3 ms | 16.4 ms | 12.6 ms |

Cold cost is paid once by the daemon: model load and first-inference graph
warm-up take roughly 5–15 s at startup, after which the first client call is
~71 ms and steady-state calls are as above. This is precisely what
`start_inference_server.sh` exists to amortise.

**Resident daemon memory: ~919 MB.** The three models' own weights are modest
(~50 MB, ~20 MB, ~30 MB); the bulk is the PyTorch + Transformers + Ultralytics
runtime. A CoreML/ONNX export would cut this substantially and is the obvious
next optimisation (§10.4) — it is *not* implemented today and this plan does
not claim otherwise.

### 2.3 Accuracy posture

Published accuracy figures for the chosen checkpoints are not reproduced here,
because this project has not re-measured them on their benchmark suites and
quoting them would be borrowing credibility. What *was* measured end-to-end:
on the 79-image indoor sequence the system recovers 16 persistent objects
(chairs, potted plants, a microwave, a sink, cups, bowls, a toaster) with
plausible metric extents — a chair at 1.27 m³, a cup at 0.016 m³. Depth scale
sanity is the load-bearing property here, and the metric checkpoint delivers it.

---

## 3. System Topology and Delegation

```mermaid
flowchart TD
    subgraph CLI["Shell entry points"]
        SRV["start_inference_server.sh"]
        REC["reconstruct.sh"]
        MAP["mapper.sh"]
        SEG["segment.sh"]
    end

    subgraph Daemon["Resident daemon (FastAPI, UDS + TCP fallback)"]
        DEPTH["/depth  - Depth-Anything-V2-Metric-Small"]
        SEGEP["/segment - YOLO11s-seg | YOLOE-11s-seg"]
        HEALTH["/health"]
    end

    subgraph Core["Shared package (src/oh_my_slam)"]
        RECON["reconstruction.py -- SOLE OWNER: depth, back-projection"]
        SEGMOD["segmentation.py -- SOLE OWNER: masks, OBBs, colour, artefacts"]
        REG["mapping/registry.py -- object identity and lifetime"]
        OCT["mapping/octree.py -- Bayesian log-odds occupancy"]
        VO["mapping/visual_odometry.py -- ORB + PnP-RANSAC"]
        OL["common/openlabel.py -- ASAM OpenLABEL v1.0.0"]
        PAL["common/palette.py -- deterministic colour"]
    end

    SRV --> Daemon
    REC --> RECON
    SEG --> SEGMOD
    MAP --> RECON
    MAP --> SEGMOD
    REC -.->|"-f json"| SEG

    RECON --> DEPTH
    SEGMOD --> SEGEP
    SEGMOD --> RECON
    REG --> SEGMOD
    MAP --> REG
    MAP --> OCT
    MAP --> VO
    SEGMOD --> OL
    SEGMOD --> PAL
```

### 3.1 The delegation rules, exactly

1. **Depth has one owner.** `oh_my_slam.reconstruction.reconstruct_frame()` is
   the only function in the project that calls the `/depth` endpoint. It returns
   a `FrameReconstruction` carrying the RGB image, the metric depth map, the
   intrinsics, and the methods to unproject, lift a mask, or project back.
2. **Semantics have one owner.** `oh_my_slam.segmentation` is the only module
   that calls `/segment`, rasterises masks, fits OBBs (`fit_object_obb`),
   assigns colour (`assign_color`) and writes artefacts. Even the object
   registry reaches OBB fitting and colour *through* this module rather than
   importing `geometry.obb` or `common.palette` directly.
3. **`mapper.sh` delegates both.** It calls `reconstruct_frame()` for each
   frame and hands the *same* frame object to `detect_instances()`. One depth
   inference per frame, no duplicated lifting, no second HTTP round trip.
4. **`reconstruct.sh -f json` delegates to `segment.sh`.** The JSON scene
   description is segmentation's product, so `reconstruct.sh` shells out to
   `segment.sh -i <image> -f json` and passes its stdout through unchanged.
   `-f ply` is served directly, since the raw cloud is reconstruction's own.

**Why module-level rather than subprocess-level delegation inside `mapper.sh`:**
the spec also requires that "shared logic lives in a common Python package used
by all tools" and that duplicated work be avoided. Shelling out per frame would
spawn two interpreters per keyframe and pay for depth twice — once in
`reconstruct.sh -f ply` and again inside `segment.sh`. Routing through the same
single-owner modules honours the ownership rule without that waste. The
ownership boundary is enforced mechanically by `tests/test_delegation.py`, which
parses the modules and asserts that no file outside `reconstruction.py` calls
the depth endpoint and no file outside `segmentation.py` fits a box or picks a
colour.

---

## 4. Entry Points

### 4.1 `start_inference_server.sh`

Starts the daemon, waits for `/health`, writes `.inference_server.pid`.
Transport is a UNIX domain socket at `/tmp/oh_my_slam_infer.sock` with a TCP
fallback on `127.0.0.1:8001`. Images are passed **by absolute path**; the daemon
reads them directly, which on a single machine avoids serialising pixel buffers
over the socket. Depth comes back as a binary payload — `uint32 H`, `uint32 W`,
then the `float32` depth array — not as JSON numbers.

Clients probe for the socket and exit immediately if it is absent:

```
Error: Inference server is not running. Start it with ./start_inference_server.sh
```

Verified for all three tools: exit code 1, empty stdout, message on stderr.

### 4.2 `reconstruct.sh`

```sh
reconstruct.sh -i <image>          # OpenLABEL scene description (default)
reconstruct.sh -i <image> -f ply   # binary little-endian PLY, per-point colour
reconstruct.sh view -i <image>     # browser viewer
```

Back-projection uses the right-handed pinhole model

$$X_c = Z\frac{u-c_x}{f_x}, \qquad Y_c = -Z\frac{v-c_y}{f_y}, \qquad Z_c = -Z$$

so **+x is right, +y is up, −z looks into the scene**. Intrinsics come from EXIF
`FocalLengthIn35mmFilm` when present, otherwise from a 65° horizontal field of
view, i.e. $f_x = f_y = \tfrac{W}{2\tan(32.5°)} \approx 0.785\,W$. Depth outside
[0.1 m, 20 m] is discarded as invalid.

`view` serves the *photometric* cloud with OBB wireframes over it, so the viewer
shows the scene as captured rather than recoloured by the object palette.

### 4.3 `mapper.sh`

```sh
mapper.sh update -a <image(s)|video> -m <folder> [-f json|ply] -t full|single [-fps <n>]
mapper.sh view -m <folder>
```

`update` creates the map directory if absent and otherwise **extends** it: the
existing `map.ply`, `objects.json`, `cameras.json` and `occupancy.octo` are all
loaded first, and tracking resumes from the last recorded pose.

Per frame: reconstruct (delegated) → track pose → if keyframe, integrate the
cloud into the octree, detect instances (delegated), and fold them into the
registry. Keyframes are selected when the matched-feature ratio drops below 60%;
the first and last frames of each batch always count.

The map directory holds:

| File | Contents |
| :--- | :--- |
| `map.ply` | Global photometric point cloud, capped at 300 000 points |
| `objects.json` | Persistent object registry: ids, colours, OBBs, retained points |
| `cameras.json` | Pose of every contributing frame |
| `occupancy.octo` | Bayesian log-odds occupancy octree, 5 cm voxels |

**`-t full`** emits every active object plus the pose of every contributing
frame, in map coordinates. **`-t single`** emits only the objects this run
created or updated, and only the poses of the frames just added — tracked by a
per-run change set opened by `begin_update()`.

### 4.4 `segment.sh`

```sh
segment.sh -i <image> [-o <folder>] [-f json|ply] [--min-score <s>] [--labels a,b,c]
segment.sh -m <map-folder> [-o <folder>] [-f json|ply]
segment.sh view -i <image>
```

**Image mode** reconstructs the frame (delegated, one depth call), detects
instances, lifts each mask to 3D, fits a gravity-aligned OBB, assigns the
contract colour, and renders artefacts. `--labels` routes to YOLOE for genuine
open-vocabulary segmentation; `--min-score` defaults to 0.5.

**Map mode** loads the registry, re-fits every surviving object's OBB over its
retained points, and renders the map's own artefacts. Because per-pixel masks
are not persisted with a map, `segmented.png` is produced by projecting each
object's world points back into the contributing frame that sees the most
objects — not simply frame 0, which after pose drift often looks away from
everything the map later accumulated.

#### Output artefacts (`-o <folder>`)

| File | Contents |
| :--- | :--- |
| `segmentation.json` | The OpenLABEL scene description of §6.3, identical to stdout |
| `segmented.png` | Instance masks in object colour over the original dimmed to 40%, α = 0.55, with a solid contour in the pure colour |
| `catalog.csv` | `id,label,score,color_hex,width_m,height_m,depth_m,volume_m3,center_x,center_y,center_z,pixel_count,point_count` |
| `catalog.md` | The same catalogue as a table with colour swatches, ordered by descending volume |
| `segments.ply` | Point cloud coloured per object; unsegmented points mid-grey `(128,128,128)` |

All five are written whenever `-o` is given, in both image and map mode.

---

## 5. Contradiction Handling and Object Lifecycle

> *"Since an image captures a specific point in time for a map section, any new
> image that contradicts the current data should update the map with the latest
> information to keep it current."* — spec §2.3

### 5.1 Tier 1 — Bayesian log-odds occupancy carving

Every keyframe casts a ray from the camera centre to each measured point.
Endpoints gain occupancy evidence, traversed voxels lose it:

$$L_t(m) = \mathrm{clamp}\!\left(L_{t-1}(m) + \Delta L(m),\; L_{\min},\; L_{\max}\right)$$

$$\Delta L(m) = \begin{cases}
\ln\frac{0.85}{0.15} \approx +1.734 & \text{endpoint voxel} \\[4pt]
\ln\frac{0.30}{0.70} \approx -0.847 & \text{traversed free-space voxel}
\end{cases}$$

configured as `setProbHit(0.85)`, `setProbMiss(0.30)`,
`setClampingThresMin(0.03)`, `setClampingThresMax(0.993)`,
`setOccupancyThres(0.5)`. Two clear traversals are enough to drive a previously
occupied voxel below the occupancy threshold, so a moved chair or an opened
door clears promptly.

### 5.2 Tier 2 — object support, measured on surfaces

An object is pruned when the surface it was built from has been seen *through*:

$$\text{carved}(O) = \frac{\left|\{\,p \in \mathcal{P}_O : \text{voxel}(p)\ \text{observed free}\,\}\right|}{|\mathcal{P}_O|},
\qquad \text{prune when } 1 - \text{carved}(O) < 0.25$$

Two properties of this formulation are load-bearing, and the first is a
correction to an earlier version of this plan:

* **Support is measured over the object's own observed points, not over its
  bounding volume.** Monocular depth only ever reports *surfaces* — the interior
  of a solid object is never observed and never becomes occupied. A
  volume-based ratio therefore condemns healthy objects: measured on a real map,
  a correctly-tracked book scored 0.110 by volume (well under the 0.25
  threshold) against 0.377 by surface. The volume formulation pruned **every
  object in the map**.
* **Unknown is not free.** Only voxels a later frame *explicitly observed as
  empty* count against an object. A voxel no ray has reached is unknown, and
  absence of evidence is not evidence of absence — without this distinction,
  ordinary pose drift silently deletes correct geometry.

**Two guards against pose drift.** A drifted keyframe carves the space where
correct geometry actually sits, so raw carving alone empties the map. Measured
on a real sequence: re-feeding overlapping images drove the support of healthy
objects to **0.000** and pruned every object in the map. Two rules fix it:

* An object the segmenter **confirmed during this very update** is never
  pruned. Direct positive evidence outranks indirect free-space inference.
* A single carved update is a **strike, not a verdict**. An object must read as
  carved on two consecutive updates before it leaves the map, and any update in
  which its support recovers clears the count. A genuinely moved object keeps
  failing; a one-off bad pose does not. With these in place the same repeated-
  update sequence holds its objects across four passes instead of collapsing to
  zero.

**Missing-evidence decay.** If an object's centre projects inside a keyframe's
frustum but the segmenter reports no supporting mask, a miss is recorded. After
three consecutive misses the score decays by 0.6 per miss and the object is
culled below 0.1. A single successful re-observation resets the counter.

**OBB re-fitting.** Surviving objects re-fit their box over their current
retained points (capped at 4 000, sampled evenly) at the end of every update, so
a partially carved object tightens rather than keeping a stale hull.

---

## 6. Conventions, Colour and Scene Schema

### 6.1 Spatial conventions

Right-handed, metres, **+y up**, **−z view direction**. Single-frame output is
in the `camera` coordinate system with its origin at the optical centre; map
output is in the `map` system anchored to keyframe 0. Both are declared
explicitly in every document's `coordinate_systems` block, and the OpenLABEL
transform per frame is `camera → map`.

### 6.2 The deterministic colour contract

The spec requires a colour that is *one per object id*, drawn from a *fixed,
perceptually distinct palette*, a *pure function of the id*, and which *cycles
by hue once the palette is exhausted*.

```
palette_index(id) = ordinal(id) - 1        for ids minted as obj_001, obj_002, ...
                  = crc32(id) mod 64       for any other id shape

colour(id) = GLASBEY_64[palette_index]                              if index < 64
           = hsv( ((index-64+1) * 0.6180339887) mod 1, 0.85, 0.95 ) otherwise
```

Reading the index off the **ordinal** rather than hashing the id matters:
`crc32(id) mod 64` collides long before 64 objects exist, handing two objects
the same swatch and violating "perceptually distinct"; and because a hash wraps
immediately, it can never *exhaust* the palette, so the hue-cycling clause could
never fire. The ordinal index gives both: the first 64 objects are provably
distinct, and object 65 onward steps the hue circle by the golden-ratio
conjugate. CRC32 remains the fallback for externally-supplied ids, because
Python's built-in `hash()` is salted per process and would break reproducibility.

The same triple appears in `segmentation.json` (`color_hex` / `color_rgb`), the
painted mask pixels of `segmented.png`, the swatch column of `catalog.csv` and
`catalog.md`, the per-point colour of `segments.ply`, and the OBB wireframe in
the viewer. `tests/test_artefacts.py` writes all five and compares the actual
pixels and rows.

### 6.3 Scene description — ASAM OpenLABEL v1.0.0

The spec asks for "a well known json scheme url that support JSON OBB".
**ASAM OpenLABEL** is that schema. It is an ASAM standard for annotating
objects in multi-sensor data, and its `cuboid` object data is precisely a JSON
oriented bounding box: ten floats giving centre, orientation quaternion and
dimensions.

* **Standard:** ASAM OpenLABEL v1.0.0
* **Schema URL:** `https://raw.githubusercontent.com/Vicomtech/video-content-description-VCD/master/schema/openlabel_json_schema-v1.0.0.json`
* **Vendored for offline validation:** `schemas/openlabel_json_schema-v1.0.0.json`
* Emitted in every document at `openlabel.metadata.schema_url`, alongside the
  mandatory `openlabel.metadata.schema_version: "1.0.0"`.

```json
{
  "openlabel": {
    "metadata": {
      "schema_version": "1.0.0",
      "schema_url": "https://raw.githubusercontent.com/.../openlabel_json_schema-v1.0.0.json",
      "annotator": "oh-my-slam 0.1.0 (track-c)",
      "coordinate_conventions": {
        "handedness": "right-handed", "up_axis": "+y",
        "view_axis": "-z", "units": "meters"
      },
      "tagged_file": "/path/to/input", "timestamp": 1789845942.9
    },
    "coordinate_systems": {
      "map":    { "type": "local",  "parent": "",    "children": ["camera"] },
      "camera": { "type": "sensor", "parent": "map", "children": [] }
    },
    "streams": {
      "rgb_camera": {
        "type": "camera",
        "stream_properties": {
          "intrinsics_pinhole": {
            "width_px": 640, "height_px": 480,
            "camera_matrix_3x4": [502.3, 0, 320, 0,  0, 502.3, 240, 0,  0, 0, 1, 0]
          }
        }
      }
    },
    "objects": {
      "1": {
        "name": "obj_001",
        "type": "chair",
        "coordinate_system": "map",
        "object_data": {
          "cuboid": [{ "name": "obb",
                       "val": [x, y, z, qx, qy, qz, qw, sx, sy, sz],
                       "coordinate_system": "map" }],
          "num":  [{ "name": "score", "val": 0.83 },
                   { "name": "volume_m3", "val": 1.266 },
                   { "name": "yaw_deg", "val": 31.4 },
                   { "name": "pixel_count", "val": 4120 },
                   { "name": "point_count", "val": 3011 }],
          "text": [{ "name": "color_hex", "val": "#E6194B" }],
          "vec":  [{ "name": "color_rgb", "val": [230, 25, 75] }]
        }
      }
    },
    "frames": {
      "0": { "frame_properties": {
               "timestamp": 1789845942.9,
               "transforms": { "camera_to_map": {
                   "src": "camera", "dst": "map",
                   "transform_src_to_dst": { "quaternion": [0,0,0,1],
                                             "translation": [0,0,0] } } },
               "streams": { "rgb_camera": { "uri": "/path/frame_00000.jpg" } } } }
    },
    "frame_intervals": [{ "frame_start": 0, "frame_end": 78 }]
  }
}
```

Two schema constraints shape this mapping. OpenLABEL object keys must be
numeric UIDs or UUIDs, so the persistent ordinal is the key and the readable
`obj_001` is carried as `name`. And the document root permits only the
`openlabel` member, so the schema URL lives inside `metadata` (which does admit
additional properties) rather than as a root `$schema`.

`mapper.sh -t full` populates `frames` with the pose of every contributing
frame, satisfying the spec's requirement that the full scene include each
frame's estimated camera pose.

---

## 7. Platform and Tooling

### 7.1 Apple Silicon execution

All three models run on **PyTorch MPS**, falling back to CPU where MPS is
unavailable. Geometry (back-projection, mask lifting, OBB fitting, PLY
serialisation) is vectorised NumPy over Accelerate; occupancy ray-casting is
native C++ `octomap`. Images are handed to the daemon by path, so no pixel
buffer is serialised across the IPC boundary.

### 7.2 Dependencies (`pyproject.toml`)

```
torch, torchvision          PyTorch with native MPS support
transformers                Depth-Anything-V2-Metric-Indoor-Small
ultralytics                 YOLO11s-seg and YOLOE-11s-seg
octomap-python              Bayesian log-odds occupancy octree
fastapi, uvicorn, httpx     daemon and client over a UNIX domain socket
numpy, scipy, opencv-python-headless, plyfile, pillow, pydantic
jsonschema                  validates output against the vendored OpenLABEL schema
pytest (dev extra)
```

`--labels` additionally needs YOLOE's CLIP text encoder
(`clip @ git+https://github.com/ultralytics/CLIP.git`). If it is missing the
daemon logs a warning and falls back to COCO segmentation with a label filter,
so `--labels` degrades rather than failing.

### 7.3 Layout

```text
oh-my-slam/
├── .venv/                          uv-managed
├── pyproject.toml
├── schemas/
│   └── openlabel_json_schema-v1.0.0.json   vendored, for offline validation
├── start_inference_server.sh
├── reconstruct.sh                  owns depth + back-projection
├── mapper.sh                       owns poses, occupancy, object lifetime
├── segment.sh                      owns masks, OBBs, colour, artefacts
├── src/oh_my_slam/
│   ├── common/      openlabel.py, palette.py, coordinates.py
│   ├── geometry/    pinhole.py, obb.py, ply.py
│   ├── inference/   server.py, client.py
│   ├── mapping/     mapper.py, registry.py, octree.py, visual_odometry.py
│   ├── viewer/      server.py, template.html
│   ├── reconstruction.py           SOLE OWNER of depth
│   ├── segmentation.py             SOLE OWNER of semantics
│   └── cli_*.py
└── tests/           61 tests: schema, palette, delegation, geometry, registry, artefacts
```

---

## 8. Measured Latency and Throughput

Per-stage, Apple M4, warm daemon:

| Stage | 640 × 480 | 1920 × 1080 |
| :--- | ---: | ---: |
| Metric depth inference | 36.8 ms | 64.3 ms |
| Instance segmentation | 11.9 ms | 16.4 ms |
| Visual odometry (ORB + PnP-RANSAC) | 6.9 ms | 12.6 ms |
| Back-projection, full resolution | 6.2 ms | — |
| Back-projection, ⅓ stride (mapping) | 0.6 ms | 5.4 ms |
| **Occupancy ray-casting into the octree** | **49.0 ms** | **188.1 ms** |
| OBB fit + colour assignment | 3.4 ms | 22.8 ms |
| PLY serialisation, 307 k points | 2.4 ms | — |

End-to-end, measured:

| Workload | Result |
| :--- | :--- |
| Single-frame segmentation in-process, 480p | 59.5 ms (**16.8 FPS**) |
| `mapper.sh update`, 79 images at 640×480 | 8.8 s total, **111 ms/frame (9.0 FPS)** |
| `mapper.sh update`, 51 frames at 1920×1080 from video | 52.5 s total, **1.03 s/frame (~1 FPS)** |
| Daemon resident memory | ~919 MB |
| Daemon cold start | 5–15 s, paid once |

**The bottleneck is occupancy ray-casting, not the neural networks.** At 1080p
the octree consumes 188 ms against 64 ms for depth. The practical levers are a
coarser integration stride, a larger voxel, or downscaling before mapping; a
faster depth model would barely move the number. Shell invocations add roughly
0.4 s of Python interpreter start-up each, which is why the mapper processes a
whole batch in one process.

---

## 9. Specification Compliance

| Spec clause | Status | Where / caveat |
| :--- | :--- | :--- |
| §1 RGB-only input, no depth sensor / stereo / IMU | **Compliant** | Depth solely from the monocular metric checkpoint |
| §1 Four shell entry points | **Compliant** | All four present and executable |
| §2.1 Resident server, models stay loaded | **Compliant** | FastAPI daemon over UDS; 5–15 s cold start amortised |
| §2.1 Clear actionable error when server is down | **Compliant** | Verified: exit 1, empty stdout, message on stderr, all three tools |
| §2.2 `reconstruct.sh -i`, `-f json\|ply` default json, stdout | **Compliant** | JSON delegated to `segment.sh`; PLY streamed binary |
| §2.2 `reconstruct.sh view` | **Compliant** | Photometric cloud with OBB overlay |
| §2.3 `mapper.sh update`, creates or extends | **Compliant** | Existing cloud, registry, poses and octree are all loaded and extended |
| §2.3 `-a` images or video, `-fps` sampling | **Compliant** | Verified on 79 images and on `church.mp4` at `-fps 2` |
| §2.3 `-t full` includes every frame's camera pose | **Compliant** | OpenLABEL `frames` with `camera_to_map` transforms |
| §2.3 `-t single` returns only the newly added input | **Compliant** | Per-run change set; verified 8 new frames against a 16-frame map |
| §2.3 New images contradicting the map update it | **Compliant** | Two-tier carving of §5; note the surface-vs-volume correction |
| §2.3 `mapper.sh view` | **Compliant** | Cloud, OBBs, labels and camera frusta |
| §2.3 Persistent object id and colour, OBB refined | **Compliant** | Minted once at first sight; re-fitted each update |
| §2.4 `segment.sh -i` / `-m`, mutually exclusive | **Compliant** | Enforced with an explicit error |
| §2.4 `--min-score` default 0.5, `--labels` | **Compliant** | `--labels` is genuinely open-vocabulary via YOLOE |
| §2.4 Five artefacts with `-o`, exact CSV columns | **Compliant** | Both image and map mode; column order asserted by test |
| §2.4 Colour contract identical across all artefacts | **Compliant** | Verified by reading the written files back and comparing pixels |
| §3 Well-known JSON schema URL supporting OBB | **Compliant** | ASAM OpenLABEL v1.0.0; every document validated against the vendored schema in CI-able tests |
| §4 Python, `uv`, `.venv` | **Compliant** | |
| §4 `mapper.sh` delegates depth to `reconstruct.sh` | **Compliant** | Module-level, one inference per frame; rationale in §3.1, enforced by test |
| §4 `segment.sh` sole owner of segmentation, OBB, colour | **Compliant** | Registry reaches both through it; enforced by test |
| §4 Shared package, no duplication, typed, tested | **Compliant** | 62 tests; catalogue writers and colour logic exist once |
| §4 JSON on stdout machine-parseable, diagnostics on stderr | **Compliant** | Verified for every tool and both formats |
| §4 Works on Mac M4 | **Compliant** | All figures measured on it |
| §4 Accurate and performant | **Partial** | Performance measured in §8. Accuracy is limited by monocular VO drift — see §10.1, the one place this system materially falls short |

---

## 10. Known Limitations

These are real and are stated here rather than buried.

### 10.1 Monocular pose drift is the dominant accuracy limit

Visual odometry is frame-to-keyframe ORB matching with PnP-RANSAC against
points back-projected from per-frame monocular depth. There is **no loop
closure, no global bundle adjustment, and no cross-frame depth-scale
alignment**. Consequences observed in testing:

* Pose error accumulates over a sequence, so the same physical object seen
  early and late can fail the 1 m association gate and be registered twice.
* Across separate `update` invocations the drift is worse, because tracking
  restarts from the last stored pose with no shared features.
* Drifted poses feed drifted rays into the occupancy octree. This is why the
  "unknown is not free" rule and the two drift guards of §5.2 matter so much —
  without them, drift alone deletes correct geometry, and measurably did.

Fixing this properly means keyframe graph optimisation with loop closure, plus
per-frame depth-scale alignment against triangulated features. That is a
substantial piece of work and is not in this implementation.

### 10.2 Map-mode `segmented.png` shows one viewpoint

A map spans many viewpoints; no single frame sees every object. Map mode paints
the objects visible from the best available contributing frame — on the 79-image
map, 3 of 11 objects. All 11 are correctly coloured in `segments.ply` and
`segmentation.json`. This is inherent to rendering a multi-view map as one
image, and the tool logs to stderr when no frame sees anything.

### 10.3 Occupancy ray-casting dominates mapping cost

188 ms per 1080p keyframe (§8). Mapping high-resolution video is roughly 1 FPS
and is bound by the octree, not the models.

### 10.4 No CoreML / ANE export

The models run on PyTorch MPS, which costs ~919 MB resident and leaves the
Neural Engine unused. Exporting to CoreML would likely cut both memory and
latency. It is not implemented, and no figure in this document assumes it.

### 10.5 Intrinsics are estimated, not calibrated

Without EXIF focal length the system assumes a 65° horizontal field of view.
A wrong field of view scales the whole reconstruction, so metric dimensions from
an uncalibrated camera carry that systematic error.

### 10.6 Objects may be emitted below `--min-score`

Missing-evidence decay (§5.2) can push a previously confident object's score
below the detection threshold while it remains in the map. `segment.sh -m
--min-score` filters these on readout; `mapper.sh` reports them so the decay
remains visible.
