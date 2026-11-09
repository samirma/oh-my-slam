# Experiment cases

A record of every experiment run against oh-my-slam: the stack it used, what it changed, how it
was measured and what came out of it. Before starting a new experiment, check here whether it has
been tried, and add an entry when it ends (template at the bottom).

Machine for every result below: Apple M3 Pro, 18 GB, macOS 27.0.1. All GPU work went through
`~/oh-my-slam-data/bin/heavy`, one job at a time. Nothing was committed; every worktree is under
`/Users/samir/oh-my-slam/.claude/worktrees/` and its full diff is in
`~/oh-my-slam-data/experiments/<worktree>.patch`.

## How the results were measured

- **Evaluator.** `python -m oh_my_slam.tools.evaluate`, main's targets
  (`examples/targets.json`). "Failed" counts metrics that miss their target; "regressions" are
  accuracy or contract metrics that pass on main and fail on the candidate. Control: main's run
  `~/oh-my-slam-data/evaluations/20261026T123734Z` (63 of 640 failed).
- **Map benchmark** (round 3). `~/oh-my-slam-data/experiments/bench/` (`bench_one.sh`,
  `analyze.py`). Each example is mapped twice:
  - **All at once:** one `mapper.sh update`.
  - **Image by image:** one update per image, in capture order. street2 instead uses five 30 s
    clips, one clip per update.

  Limits: 60 min all at once, 30 min per image-by-image update, and an update is stopped when swap
  passes 20 GB or free disk falls under 10 GB.
- **Reports:**
  - Round 1: https://claude.ai/artifact/8uXwXkkoquXqRs3ULLwxum
  - Rounds 2 and 3: https://claude.ai/artifact/MKR7niiQeGvgfWsjrvnYg8. It holds every map's
    screenshot and its comparison with main.

## Worktrees

All are under `/Users/samir/oh-my-slam/.claude/worktrees/`. None has a commit of its own: each worktree's
changes are uncommitted on top of its base commit, and saved as a patch in `~/oh-my-slam-data/experiments/`.

| Worktree | Branch | Base | Patch | Case |
|---|---|---|---|---|
| (repository root) | `main` | `d4df2c0` + uncommitted diff | `main-uncommitted.patch` | 0 |
| `mapanything-only` | `worktree-mapanything-only` | `08f8977` | `mapanything-only-uncommitted.patch` | 0b |
| `exp-runtime` | `exp/runtime` | `d4df2c0` + main's diff | `exp-runtime.patch` | 1 |
| `exp-locate` | `exp/locate` | `d4df2c0` + main's diff | `exp-locate.patch` (`exp-locate.own.patch`: own changes only) | 2 |
| `exp-models` | `exp/models` | `d4df2c0` + main's diff | `exp-models.patch` (no own code changes) | 3 |
| `exp-cpp` | `exp/cpp` | `d4df2c0` | `exp-cpp.patch` | 4 |
| `exp-rust` | `exp/rust` | `d4df2c0` | `exp-rust.patch` | 5 |
| `exp-swift` | `exp/swift` | `d4df2c0` | `exp-swift.patch` | 6 |
| `exp-mlx-mapanything` | `exp/mlx-mapanything` | `08f8977` | `exp-mlx-mapanything.patch` | 7 |
| `exp-mapanything-latest` | `exp/mapanything-latest` | `08f8977` | `exp-mapanything-latest.patch` | 8 |
| `exp-moge3` | `exp/moge3` | `08f8977` | `exp-moge3.patch`, `exp-moge3.moge3-impl.patch` | 9 |

To run a worktree's server without colliding with main's, set
`OH_MY_SLAM_RUNTIME_DIR=~/oh-my-slam-data/runtime/<worktree>`.

## Overall verdict (2026-11-08)

- **Without Python or COLMAP: exp-cpp.** It has the fewest failed metrics of any case (24 of 625),
  is among the fastest in both modes, and agrees best with main among the native ports.
- **When maps grow one image per update: exp-rust.** It is the only case that refused no image, but
  it has the weakest agreement with main and duplicate objects image by image.
- **If Python is allowed: exp-mlx-mapanything** has the fewest regressions (2). It needs an unmerged
  `mlx-vlm` fork, and image by image it refuses 32 of 36 camera images.
- **Not merged anywhere.** main is unchanged and `baseline.json` was not re-recorded.

## Index

| # | Case | Stack | Runtime has Python / COLMAP | Verdict |
|---|---|---|---|---|
| 0 | main (control) | Python 3.12, PyTorch MPS, COLMAP 4.2 SIFT + pycolmap, MoGe-2 ViT-L, GeoCalib, YOLOE-26x + MobileCLIP | yes / yes | baseline |
| 0b | mapanything-only | Python, PyTorch MPS, MapAnything places keyframes (no COLMAP), MoGe-2, GeoCalib, YOLOE-26x | yes / no | source of `baseline.json`; reference algorithm for round 2 |
| 1 | exp-runtime | main's stack; faster model loading and CPU stages | yes / yes | **selected in round 1** (not merged) |
| 2 | exp-locate | main's stack; locate fallback and bird's-eye fix | yes / yes | keep, combine with 1 |
| 3 | exp-models | main's stack; MoGe-2 ViT-B, MetricAnything, YOLOE-26l | yes / yes | rejected |
| 4 | exp-cpp | C++, OpenCV 4.12, ONNX Runtime (CoreML EP) + Core ML direct | no / no | **best all at once**; recommended without Python |
| 5 | exp-rust | Rust, `ort` 2.0 rc15 (CoreML EP) + Core ML via `objc2`, AVFoundation | no / no | fastest; **best image by image** |
| 6 | exp-swift | Swift, Core ML, Accelerate/Vision, no third-party packages | no / no | slowest native port, server stalls |
| 7 | exp-mlx-mapanything | Python; MapAnything on MLX (mlx 0.32, pinned `mlx-vlm` fork) | yes / no | most accurate in the evaluator; camera fails image by image |
| 8 | exp-mapanything-latest | Python, PyTorch MPS, MapAnything; mapanything-only brought to the latest spec | yes / no | agrees best with main all at once |
| 9 | exp-moge3 | MoGe-3 ViT-L (MLX port and patched PyTorch) in the depth slot | yes / — | rejected |
| 10 | exp-romav2 | main's stack + RoMa v2 (DINOv3 ViT-L inside) dense locate fallback; LoFTR tried first | yes / yes | superseded by 12: same 12 of 12, but +30 s and needs the server |
| 11 | exp-maskcluster | main's stack; MaskClustering-style view-consensus re-association every update (+ MoGe DINOv2 tokens, dropped) | yes / yes | rejected: no gain, objects stage 4× slower, 7 tests broken |
| 12 | exp-locate-motion | main's stack + exp-locate + chained rotation-only locate through same-call anchors (turning maps only) | yes / yes | **keep**: camera locate 8 → 12 of 12, no guessed poses |
| 13 | exp-export-thresholds | main's stack; confirmation judged in order of addition (rule A), export without cloud points (variant B) | yes / yes | rejected: exposes cloud-point and floor-height order dependence; found the gap is geometry |
| 14 | exp-eval-splits | evaluator: even/odd interleaved camera locate (+ old split as `camera_locate_unseen`), one-image-per-update group for office/camera/ainex | yes / yes | **keep**: ready to propose for main; main fails IoU 0.41 (office) and locate 0.75 |
| 15 | exp-da3-depth | main's stack; Depth Anything 3 (large 1.1 any-view + nested metric) in the depth slot, conditioned on poses over co-visible windows | yes / yes | **keep (evaluator candidate)**: camera pairs p90 14 → 4.5 %, scale fixed, office image-by-image IoU 0.35 → 0.43; +42 % camera time |
| 16 | exp-map-long | VGGT-Long chunking (re-implemented, numpy) + MapAnything in the pose slot; whole keyframe set replayed every update | yes / no (poses) | rejected as pose engine (camera/ainex worse, street2 1175 s); **replay protocol kept**: office image-by-image IoU 0.41 → 0.88 |
| 17 | exp-n1 | C++ (exp-cpp) + COLMAP 4.2.1 as a library + Ceres/Eigen port of main's pose pipeline (rotation-only path); Core ML models | no / library only | **keep**: every mapper probe under target (office 41 s, ainex 92 s, camera 82 s), poses within 0.1–0.35° of main; coverage 97.1/89.3 % |
| 18 | exp-sam31 | main's stack + SAM 3.1 (fp16, MPS) redrawing YOLOE-26x masks from its boxes; LLMDet-L as proposer (arm B) | yes / yes | rejected: masks +8 % edge alignment, but box stability unchanged (camera split IoU 0.28 → 0.26), segment 2.3–3.4× slower; LLMDet does not fit in 18 GB |
| 19 | exp-longstream | main's stack + LongStream (fp16, MPS, streaming) in the pose slot, whole keyframe set replayed every update; COLMAP SIFT kept for triangulation | yes / yes (matching only) | rejected as pose engine (street2 696 → 1001 s, turning-in-place yaw off by 20° median); **replay protocol confirmed**: office image-by-image IoU 0.78 |
| 20 | exp-replay-main | main's stack and engine; every update replays the whole ordered keyframe set from each keyframe's stored first inference (depth, intrinsics, detections), camera prior reset and pairs re-verified | yes / yes | **keep (candidate for main)**: image by image = one update (box IoU office 1.00, camera 0.99, ainex 1.00; ainex refusals 39 → 0); one-update maps unchanged; cost: update time grows with the map (ainex last 208 s > 189 s) |
| — | VGGT-Ω, SAM 3.1, CalibAnyView, VGG-T³, Core AI | researched online only | — | not tested (see below) |

## Round 1 (2026-10-30): three changes to main

Each experiment started from main's working copy (`d4df2c0` + uncommitted diff). The full
evaluator chain did not finish because the machine ran out of memory, so these results come from
before/after probes in each worktree.

### 1. exp-runtime: same models, faster runtime
- **Changes:**
  - MapAnything built on the `meta` device and loaded with `load_state_dict(assign=True)`, offline
    from a pinned HF revision.
  - GeoCalib's LM refinement moved off the MPS thread to a 2-thread CPU pool.
  - COLMAP SIFT thread count sized from free memory.
  - Cached depth medians, a faster `support_level.ring` and a one-pass `validity` index.
- **Results:**
  - Cold start: 39 s → 17 s.
  - MapAnything load: 27.6 s → 5.2 s.
  - Office map: 218–335 s → 64 s.
  - street2, first 30 s: 131 s → 99 s.
  - Office SIFT memory: 13.5 GB → 4.7 GB.
  - Accuracy: outputs bit-identical, or within run-to-run noise.
- **Tried and dropped:** MapAnything `minibatch_size` 4 or 8. No speed-up, and depth changed by
  up to 6.7.
- **Not tried:** MoGe-derived gravity. `validity` is still O(N²).

### 2. exp-locate: lens handling and locate
- **Changes:**
  - A rotation-only locate fallback for keyframes taken from one spot (needs ≥ 15 inliers within
    0.5°).
  - The bird's-eye image draws ID tags before the OBB outlines.
- **Results:**
  - Camera `located_fraction`: 0.25 → 0.67 (target 0.9).
  - Bird's-eye contract failures: 5 per image → 0.
- **Findings:**
  - Undistorting the query before SIFT does not help.
  - About 8 of 12 is the ceiling with SIFT on that split.
  - `baseline.json` was recorded on the dirty mapanything-only worktree, not on `08f8977`.
  - The office 6+7 split collapse, in `_join_unplaced`, is not fixed.

### 3. exp-models: other engines in the same slots
- **Depth:**
  - MoGe-2 ViT-B: 2× faster, but office AbsRel 3.71 % against 3.14 % for ViT-L.
  - MetricAnything Student: AbsRel 3.37 %, no gain.
- **Segmentation:** YOLOE-26l finds 85 objects on `restaurant.jpg` against 116 for 26x (target ≥ 100),
  with label agreement 0.91.
- **Risk found:** maps don't record which model produced their descriptors. A model swap needs that
  record and a re-index of the stored keyframes.

## Round 2 (2026-10-31): no COLMAP, no Python at runtime

Brief: `~/oh-my-slam-data/experiments/BRIEF_NATIVE.md`.
- Every candidate ports the same algorithm, the COLMAP-free pipeline of mapanything-only:
  - MapAnything places the keyframes;
  - MoGe-2 gives depth;
  - GeoCalib gives gravity;
  - YOLOE + MobileCLIP give segmentation.
- Each one is brought to the latest spec. Python is used only offline, to convert the weights, and
  as the evaluator.
- Shared converted models are in `~/oh-my-slam-data/native-models/`, with parity in
  `MANIFEST.json`. They include a MapAnything v2 package of 46 functions, adding lower-resolution
  single calls.
- **Runtime deviation (C++, Rust):** ONNX Runtime was unusable for MoGe-2 and MapAnything. One
  2-view MapAnything call took 64 s with an 84 GB peak, and each MoGe input shape cached 5.6 GB.
  Those models call Core ML directly; YOLOE stays on ONNX Runtime with the CoreML provider.

Evaluator results (`~/oh-my-slam-data/experiments/eval-exp-*`):

| Case | Failed | Regressions | Fixed vs main | Full run | Cold start | Server peak | office / camera / ainex / street2 map | Coverage (line / branch) |
|---|---|---|---|---|---|---|---|---|
| main | 63 / 640 | — | — | 62 min | 42.3 s | 15.4 GB | 180 / 179 / 301 / 842 s | 100 / 100 % |
| exp-cpp | **24 / 625** | 6 | 47 | 38 min | 16.4 s | 12.7 GB | 54 / 81 / 291 / 792 s | 99.09 / 92.79 % |
| exp-rust | 36 / 673 | 13 | 40 | 36 min | **2.1 s** | **9.6 GB** | 51 / 81 / 264 / **714 s** | 98.06 / 88.54 % |
| exp-swift | 65 / 613 | 9 | 30 | 185 min | 9.2 s | 12.4 GB | 95 / 121 / 342 s / failed | 100 / 100 % (regions) |
| exp-mlx-mapanything | 27 / 604 | **2** | 37 | 51 min | 14.8 s | 13.4 GB | 47 / 90 / 254 / 992 s | 100 / 100 % |
| exp-mapanything-latest | 32 / 604 | 3 | 35 | 57 min | 35.5 s | 11.2 GB | 44 / 83 / 300 / 1014 s | 100 / 100 % |

### 4. exp-cpp: C++ + OpenCV + ONNX Runtime / Core ML
- **Size:** about 55 k lines, one static library per ownership area. OpenCV 4.12 is built locally
  under `native/third_party`. 463 unit tests.
- **Fix round:**
  - Camera tilt error: 14.7° → 3.5°.
  - Camera object id agreement: 0.72 → 1.0.
  - Server peak: 15.1 → 12.7 GB.
  - Turning in place reproduces the reference exactly (RANSAC sampling matched to numpy).
- **Beyond the reference:** a lens search and a weaker pair rule in camera locate.
- **Open:** coverage at 99.09 / 92.79 %; split box overlap (ainex 0.36 against a target of 0.5).

### 5. exp-rust: Rust + ONNX Runtime / Core ML
- **Crates:** `ort` 2.0 rc15 (CoreML EP), `objc2-core-ml`, `objc2-av-foundation` for video,
  `jsonschema`. 9 crates, one per ownership area. 273 tests.
- **Fixed:** swapped arguments in the MoGe-to-MapAnything depth fit put ainex's down-looking frames
  about 2× too deep (worst yaw error 90° → 7.9°).
- **Open:** object stability between one update and split updates (camera id agreement 0.77);
  ainex locate 23 of 26.

### 6. exp-swift: Swift + Core ML
- **Size:** about 48 k lines, no third-party packages. 548–556 tests, 100 % lines and regions.
- **Fixed:**
  - A native SIFT crash.
  - The daemon inheriting the caller's stdout.
  - A Core ML output leak.
  - Every loaded MapAnything function staying resident, which reached 29.2 GB.
- **Open:** the server stops answering late in a full run ("server busy"), so street2 failed in the
  evaluator; camera locate places 4 of 12.

### 7. exp-mlx-mapanything: MapAnything on MLX (Python kept)
- **Stack:** `mlx==0.32.*` and the `mlx-vlm` fork `eklipse2k8/mlx-vlm@558084d`, with the
  `map-anything-fp16` weights (CC-BY-NC; an Apache variant exists).
- **Results:**
  - Cold start: 40.6 → 14.8 s.
  - ainex probe: 510 → 252 s.
  - Office poses within 0.017° of PyTorch.
- **Open:** depends on an unmerged fork.

### 8. exp-mapanything-latest: mapanything-only at the latest spec
- The Python COLMAP-free reference, updated to the current spec.
- Best map quality in the evaluator (map 34 of 42, map updates 32 of 33).

### 9. exp-moge3: MoGe-3 ViT-L in the depth slot
- Tried through the MLX port (`mlx-community/moge-3-vitl-mlx-fp32`) and patched PyTorch.
- **Rejected:**
  - Camera frame agreement was worse with and without SSR refinement.
  - Refinement was 6–7× slower per keyframe and ran out of memory under the server cap.

## Round 3 (2026-11-07): building a map all at once and image by image

Every implementation of round 2 plus main, on office (13 images), camera (36), ainex (79) and
street2, measured on an idle machine. C++ and Rust were measured twice, because the first run
rounded update times up. The last column compares each image-by-image map with the same
implementation's all-at-once map, as required by `specs/mapper.md:79-84`.

Image by image vs all at once (spec targets: paired ≥ 0.9, label ≥ 0.95, median IoU ≥ 0.5, centre
≤ 0.10 m, `unexcused_extra` 0):

| Case | All at once: office / camera / ainex / street2 | Image by image: total (median per update) | Images refused image by image | Image by image vs all at once: paired (office / camera / ainex) |
|---|---|---|---|---|
| main | 330* / 152 / 285 / 696 s | 222 s (17) / 1436 s (43) / 1350 s (18) / stopped at clip 2 | ainex 39 of 79 | **1.00** / **0.84** / 0.12 |
| exp-cpp | **42 / 59** / 279 / 718 s | **76 s (5)** / **161 s (4)** / 472 s (6) / 850 s | camera 7, ainex 2; street2 kept 117 frames | 1.00 / 0.08 / 0.29 |
| exp-rust | 40 / 80 / **250** / 726 s | 81 s (6) / 172 s (4) / **431 s (5)** / 955 s‡ | **none** | 0.90 / 0.25 / **0.66** |
| exp-swift | 78 / 127 / 367 / 1044 s | 147 s (12) / 385 s (10) / 1467 s (18) / 993 s | camera 5, ainex 2, 2 clips | 1.00 / 0.26 / 0.36 |
| exp-mlx-mapanything | 57 / 95 / 258 / 1041 s | 116 s (10) / 176 s (5) / 1048 s (13) / 748 s | **camera 32 of 36**, ainex 2, 2 clips | 1.00 / 0.04 / 0.36 |
| exp-mapanything-latest | 53 / 87 / 309 / 1062 s | 108 s (9) / 298 s (9) / 1063 s (14) / 1060 s | camera 7, ainex 2, 1 clip | 1.00 / 0.17 / 0.36 |

Footnotes:
- \* Main's office all-at-once time is from 1 Nov, on a busy machine. Its idle re-runs were stopped
  twice, once on low disk and once on swap over 20 GB: main can't build that map on an 18 GB Mac
  without heavy swapping.
- ‡ Rust's first run. Its second run's street2 clip 3 took 784 s, and clip 4 stalled in inference
  at the 30-min limit (cause not established).

Agreement of each all-at-once map with main's ("direction matched", scale-free):

| Case | office | camera | ainex |
|---|---|---|---|
| exp-cpp | 1.00 | 0.72 | 0.85 |
| exp-rust | 0.88 | 0.48 | 0.64 |
| exp-swift | 0.75 | 0.60 | 0.85 |
| exp-mlx-mapanything | 1.00 | 0.56 | 0.94 |
| exp-mapanything-latest | 1.00 | 0.60 | 1.00 |

Findings:
- **No implementation meets the one-update-equals-split rule image by image, main included.** No
  median box overlap reaches 0.5 (the best is main's office, 0.36).
- **The evaluator doesn't check this mode.** Its only splits are office `[4, 4, 5]` and `[6, 7]`
  (`examples/ground_truth/office_sequence.json`) plus one camera split, so a candidate can pass
  the evaluator and still fail image by image.
- **Every MapAnything-based case grows its office map about 13 % when built image by image** (main
  5 %, Rust 6 %), so the boxes barely overlap even where the objects match.
- **Rust's image-by-image ainex map has 62 objects against 32 all at once**, which looks like
  duplicates. `unexcused_extra` would fail.
- **Every camera map is about 2.1× main's size.** From the doorway (about 2 m), main is about 20 %
  small and C++ about 30 % large. That set turns in place, so its motion gives no scale.

## Round 4 (2026-11-08): candidate models for the failing metrics

Requested by session "cuda" for the user. Brief: `~/oh-my-slam-data/experiments/BRIEF_ROUND4.md`; downloads logged in `~/oh-my-slam-data/experiments/round4_downloads.md`. Blocked on the user: exp-da3-depth (agent launch denied by the permission classifier), SAM 3.1 and DINOv3 (gated licences), COLMAP 4.2.1 (Homebrew upgrade), package installs in worktrees.

### 10. exp-romav2: dense matching (RoMa v2) for the images SIFT cannot locate
- **Hypothesis:** dense matches raise `camera_locate.located_fraction` (target 0.9). They come
  from RoMa v2 on the server, with the keyframes chosen by retrieval, and are solved by PnP on the
  map's stored depth, alone or with exp-locate's rotation-only fallback.
- **Stack:** main's stack plus RoMa v2 (`Parskatt/RoMaV2` @95c9968, package `romav2` 2.0.1).
  - Checkpoint `romav2.0.1.pt` (1.02 GiB, GitHub release v2.0.1). It embeds DINOv3 ViT-L/16
    under the DINOv3 License, which the user accepted on 2026-11-08.
  - It runs on the server's single MPS thread, setting `base` (640 x 640, one direction), 2000
    sampled matches per pair.
  - Python and COLMAP at runtime: yes.
- **Started from:** `d4df2c0` + main's diff.
  - Final patch: `~/oh-my-slam-data/experiments/exp-romav2.patch`. Own changes only, against
    main's working copy: `exp-romav2.own.patch`.
  - Both include exp-locate's own patch (rotation-only fallback, bird's-eye tag fix).
  - The first attempt with LoFTR is `exp-romav2.dense.patch`.
- **Apple support checked:**
  - RoMa v2's fused correlation kernel is optional and Linux-only. Its pure-PyTorch
    `grid_sample` path runs on MPS.
  - RoMa refuses to run unless float32 matmul precision is "highest", and another model's library
    lowers it in the server. The adapter raises it around RoMa's own calls only.
  - The `precise` setting (800 + 1280 px, bidirectional) took about 200 s for one pair and then
    failed with NaN probabilities in `sample` on MPS, so it was dropped.
  - MegaLoc (MIT, 915 MB) was not needed: retrieval uses the map's stored MoGe descriptors to pick
    the top 4 keyframes.
- **Changes:**
  - server: a `/v1/match` route and the adapter `server/models/match_romav2.py`, protocol version
    2;
  - client and gateway: `client.match` and `reconstruction/matching.py`;
  - `mapping/locate.py`, a second pass for the images SIFT plus the turn leave unlocated:
    - matches with the `DENSE_TOP_K = 4` keyframes most like the image;
    - overlap ≥ 0.5, verified by a MAGSAC fundamental matrix with at least 15 per pair;
    - depth points only, then the same PnP and epipolar gate;
    - then the turn, counting inliers within `DENSE_TURN_INLIER_DEG = 1.0°`;
  - a new stage `dense_matching`;
  - `mapper-locate` with images now needs the inference server for `server.sh` (condition
    `images_given`), with SKILL.md regenerated;
  - tests for all of this, including a fake dense matcher built from ground truth;
  - `romav2` pinned in pyproject and the lock, and listed in `THIRD_PARTY_LICENSES.md` with the
    DINOv3 row.
- **Measured with:**
  - `probes/variants.sh` (via `scripts/variant_locate.py`, in-process, with fallbacks switched)
    and `scripts/locmetrics2.py`, which computes the evaluator's `pose.camera_locate.*` for either
    split;
  - offline: `romav2_probe.py` against all 12 keyframes, with `dense_eval.py` and `turn_eval.py`.
- **Results.** Located of 12, yaw error median / max, locate wall time:

  | Variant | Block split (map 009–020, locate 021–032) | Interleaved (map even 010–032, locate odd 009–031) |
  |---|---|---|
  | SIFT (main) | 3 (0.25), 2.06 / 2.23°, 4.5 s | 9 (0.75), 0.35 / 1.30°, 5.6 s |
  | SIFT + turn (exp-locate) | 8 (0.67), 1.23 / 2.23°, 4.4 s | **12 (1.0)**, 0.35 / 1.30°, 5.1 s |
  | RoMa v2 alone | 4 (0.33), 2.15 / 3.63°, 61 s (dense 53 s, 9 images) | 11 (0.92), 0.44 / 1.30°, 27 s |
  | **RoMa v2 + turn at 1.0°** | **12 (1.0)**, 1.51 / 2.32°, 36.5 s (dense 28.9 s, 4 images) | 12 (1.0), 0.35 / 1.30°, 6.9 s (dense not called) |
  | RoMa v2 + turn at 0.5° | 10 (0.83), 1.58 / 3.03°, 37.6 s | 12 (1.0), 5.4 s |
  | LoFTR + turn (first attempt) | 8 (0.67), 16 s | — |

  - Server: 7.58 GiB in use at start (main 6.66, so +0.92 GiB). Cold start 45–46 s (main
    38–41 s).
  - After dense matching the driver allocation is 12.4 of the 13.3 GiB cap. The footprint peaked
    at 14 GB on this 18 GB Mac. Allocated memory stayed at 7.58 GiB, so the server is not marked
    degraded.
  - RoMa v2 takes 1.2 s per pair on MPS.
- **Findings:**
  - RoMa v2 does what LoFTR could not: it finds hundreds to thousands of matches between p07/p08
    (the desk side) and the map, through the near-field pole and the wall edge.
  - Its PnP poses still fail the 0.25° epipolar gate there; only the rotation-only placement
    turns them into poses.
  - At 0.5° the dense rays' inlier share is 7–31 %; at 1° it is 16–68 %. The lens is off the pan
    axis and the matches cover the near field, so the extra parallax needs the wider angle.
  - p07-mid passes at 27 % against the 25 % floor: fragile.
  - On the interleaved split, which the evaluator will use, exp-locate's fallback alone already
    gives 12 of 12, so the dense pass is never called there and costs nothing.
- **Verdict: keep, as a fallback behind exp-locate's turn.**
  - It lifts the block split from 8 to 12 of 12 and is neutral on the interleaved one.
  - Costs: about 30 s per locate that needs it, +0.9 GiB resident and a cache near the MPS cap
    afterwards, +5 s cold start, a 1 GB checkpoint under the DINOv3 License.
  - It changes `server.sh`'s contract: `mapper-locate` with images needs the inference server.
    The browser tests were not run.
- **Open:**
  - The p07-mid margin (27 % vs 25 %).
  - Free the MPS cache after a dense job, since the driver sits near the cap.
  - Dense matching for mapping (office): not tried.
  - The turn threshold of 1° was chosen on this one split.
- **Orchestrator's verdict (overrides the agent's "keep"): superseded by exp-locate-motion (12).** That one reaches the same 12 of 12 on the block split with SIFT and same-call anchors, +1 s, no inference server and no change to server.sh's contract. RoMa v2 adds about 30 s and 0.9 GiB per dense locate and makes every locate with images need the server. Keep RoMa v2 in mind only if a split appears where neither SIFT nor same-call anchors place an image.

### 11. exp-maskcluster: one global view-consensus association per update (MaskClustering) + DINOv2 mask cue
- Stack: main's (Python 3.12, PyTorch MPS, COLMAP 4.2, MoGe-2 ViT-L, GeoCalib, YOLOE-26x); Python / COLMAP at runtime: yes / yes. No new model, no download: the DINOv2 cue used MoGe-2's own DINOv2 ViT-L backbone (fine-tuned by MoGe), last-layer patch tokens.
- Started from: `d4df2c0` + main's diff; patch: `~/oh-my-slam-data/experiments/exp-maskcluster.patch` (own changes only: `exp-maskcluster.own.patch`; the version with the DINOv2 cue: `exp-maskcluster/with_dinov2_cue.own.patch`)
- Hypothesis: objects drift between one update and image-by-image updates because they are associated incrementally; one global association of all keyframes' current YOLOE masks on every update (MaskClustering, Yan et al., CVPR 2024, arXiv:2401.07745; `PKU-EPIC/MaskClustering`), with DINOv2 mask features as an appearance cue, makes the two maps agree.
- Changes (`src/oh_my_slam/mapping/`; ownership unchanged, lint-imports 11/11 kept):
  - `objects.py`: `view_consensus_sets` / `observer_thresholds` / `_consensus_groups`: MaskClustering's view consensus (tau_vis 0.3, tau_contain 0.8, tau_rate 0.9; observer thresholds from the 95th..0th percentile, as `get_observer_num_thresholds`), keeping main's constraints (compatible labels, no two instances of one keyframe, published ids apart). `_group` uses it whenever the map holds no object (always, with the next item).
  - `api.py` `_reassociate`: every update that extends a map re-forms all objects from every keyframe's stored detections (as a rebuild does, without re-placing keyframes), with the committed validity of the stored keyframes; `objects.id_holders`: each previous id goes to the new object that holds most of its detections (membership overlap; ties to the earliest detection), so ids and colours stay a function of the id.
  - DINOv2 cue (tested offline, then removed): per-detection descriptors pooled under each mask, a veto below / a rescue above a cosine threshold.
- Measured with: probes in `~/oh-my-slam-data/experiments/exp-maskcluster/{before,after}` (`probe.sh`, same commands on a copy of main's code and of this worktree), CPU-only objects-stage replays of main's bench maps (`replay.py`, `replay/`), DINOv2 descriptors from `extract_features.py` (probe only).
- Results (before = copy of main's code, after = this experiment; same probe, same machine, idle except for these):

  | Probe | before | after |
  |---|---|---|
  | office image by image vs one update (13 updates) | paired 1.00, label 1.00, IoU 0.352, centre 0.043 m, 3 extra | paired 1.00, label 1.00, IoU 0.357 (0.308 against its own one-update objects), centre 0.043 m, 4 extra |
  | office split [4,4,5] | IoU 0.758, centre 0.013 m, 2 extra, id 0.875 | IoU 0.736 (0.684), 2 extra, id 0.875 |
  | office split [6,7] | IoU 0.636, 1 extra, id 0.875 | IoU 0.636 (0.55), 1 extra, id 0.875 |
  | camera split 12/12/12 vs one update | id 0.60, label 1.00, IoU 0.283, 4 extra | id 0.50, label 0.77, IoU 0.092, 2 extra |
  | camera image by image (36 updates) | all 36 accepted; paired 0.84, IoU 0.123, 9 extra, 1454 s | **image 16 refused** (placement: depth links contradict; rebuild abandoned), series stopped |
  | objects stage, ainex one update | 33.7 s (total 282 s) | **140.1 s** (total 372 s) |
  | objects stage, street2 first 30 s | 14.9 s (total 127 s) | 18.0 s (total 135 s): slower |

  - Main's office one-update map cannot be built on this 18 GB machine (killed at 20 GB of swap, as in round 3); its office comparisons use the bench map `bench/main/office_sequence_once` and a CPU replay of its objects stage with this code ("own one-update objects").
  - Objects-stage replays of main's bench maps (CPU, `replay.py`) reproduce main's measured numbers with main's grouping (office 3 extra, camera 12 against 11 measured). Geometric view consensus: office 5 extra, camera 11. With the published-id lock lifted, camera falls to 6 extra but office stays at 4.
  - DINOv2 cue: MoGe-2's DINOv2 tokens pooled per mask separate objects poorly. Same object against a different object of a compatible label: raw cosine medians 0.91 vs 0.87 (office) and 0.69 vs 0.55 (camera); centred by the image's mean token, 0.76 vs 0.70 and 0.35 vs 0.13, with heavy overlap. A veto and a rescue swept over 5 settings changed nothing on camera and made office worse at best, so it was not plumbed through the server.
  - The extras are not decided by grouping. In the one-update map their counterparts are either confirmed but not exported (too few cloud points), split by the detector, or never confirmed, while the image-by-image copy was published early and stays published (spec). Most of the IoU loss is on small objects (cups, phones, keys) whose centres agree within 1–10 cm: depth and pose, not association.
- Gates: the experiment's code passed ruff, mypy and lint-imports. Unit tests: 9 new tests pass; 7 existing tests failed (merge keeps the lower id, split ids, moved-object ids, an untouched cabinet's box IoU 0.49 after re-association, the rebuild published-id test, a vase box IoU 0.86 < 0.9, the e2e update sequence): real semantic regressions, not fixed since the probes showed no gain. The full suite with coverage was not run on the experiment's code. The worktree is reverted to main's exact state (diff identical to main's working copy; ruff, mypy and lint-imports pass).
- Verdict: **rejected**. Re-associating all keyframes each update with view consensus does not make image by image agree with one update. Office is flat, the camera splits are worse (id 0.60 → 0.50, label 1.00 → 0.77, IoU 0.28 → 0.09), camera image by image refused an image main accepted, and the objects stage is 4.2× slower on ainex. The incremental-association premise is largely refuted: main already rebuilds camera on 34 of 36 image-by-image updates, re-associating every detection, and still disagrees.
- Open: the disagreement comes from (1) export thresholds (cloud points) against published-stays-published, (2) per-keyframe depth and pose differences on small objects, and (3) detector splits. Fix those, not the grouping. If DINOv2 is tried again, use a vanilla DINOv2 or a CLIP-style region feature rather than MoGe's fine-tuned tokens. The camera image-16 refusal is not explained: run-to-run placement variance, or the different pixel retirement of re-associated stored keyframes.

### 12. exp-locate-motion: place unlocated images through the images of the same call that the map located (turning-in-place prior)
- **Hypothesis:** in a `locate` call with several images of a camera turning in place, those the
  map locates can place the rest. The map never saw the held-out p07/p08 desk side, but the
  neighbouring pan and tilt steps did. A rotation-only fit through those steps gives the yaw from
  image evidence alone, without guessing from the input order or the pan pattern.
- **Stack:** main's stack (Python, COLMAP 4.2 SIFT + pycolmap). No model, no new dependency.
  Python / COLMAP at runtime: yes / yes.
- **Started from:** `d4df2c0` + main's diff + exp-locate's own patch (`exp-locate.own.patch`,
  rotation-only fallback + bird's-eye tag order). Patch:
  `~/oh-my-slam-data/experiments/exp-locate-motion.patch` (includes exp-locate's own changes).
- **Changes:**
  - `mapping/locate.py`:
    - `_query_matches(between_queries=True)` reads the verified query-query matches.
    - `anchor_pairs` pairs each unlocated image of a map camera with each located image of the
      same camera. Those pairs get a second COLMAP `matches_importer` pass, run only when such
      pairs exist and `map_turns_in_place` holds: most keyframes are multi-view, or `one_spot`
      holds over all keyframes.
    - `located_by_neighbours`: for each anchor (an image located against the map, never a chained
      one), `turn_ransac` must keep ≥ 15 matches that are ≥ 50 % of the pair's matches, within
      0.5°. Each anchor then gives one rotation. The largest set within 5° of one of them must
      hold more than half of the inliers. The image gets their weighted chordal mean
      (`rotation_mean`) at their centre. Otherwise it stays unlocated with its reason.
    - `turn_ransac` is batched (all 500 two-match Kabsch fits at once). Its results are
      bit-identical, and it is 1.4–3.5× faster.
  - `README.md`: a paragraph on images placed through each other.
  - `tests/unit/test_mapping_locate.py`:
    - analytic tests: placement, another spot, lens, focal or unlocated anchors, disagreeing
      anchors, `anchor_pairs`, `rotation_mean`, `map_turns_in_place`;
    - COLMAP tests: one held-out view of a rotation-dominant map hidden from the map is turned
      from the others, and a walk map never chains.
- **Measured with:** probes in `~/oh-my-slam-data/experiments/exp-locate-motion/{probes,scripts,out}`.
  - Maps:
    - the current split uses exp-romav2's `maps/camera_p1_base` (009–020, read-only);
    - ainex uses exp-locate's `maps/ainex_p1_new` (001–027);
    - the interleaved splits use new maps in `maps/camera_{even,pan}` (one update each).
  - `locmetrics.py` computes the evaluator's `pose.camera_locate.*`.
  - `locmetrics_split.py` computes the same for any split. Its reference capture is the first
    mapped one, and its headings come from the evaluator's one-update camera map (run
    20261026T123734Z).
  - A/B: exp-locate's `locate.py` swapped in, in the same worktree.
- **Results:**

  | case | exp-locate (before) | this (after) |
  |---|---|---|
  | camera, evaluator split (map 009–020, locate 021–032) | 8/12 (0.67), yaw 1.23 / 2.23° | **12/12 (1.0), yaw 1.09–1.14 / 2.23°** |
  | camera, locate 021–044 on the same map (stress) | — | 24/24, yaw 1.06 / 2.23° |
  | camera, even/odd interleave (map 010,012…032, locate 009,011…031) | 12/12, 0.35 / 1.30° | 12/12, 0.35 / 1.30° (no chain needed) |
  | camera, pan interleave (map p03,p05,p07, locate p04,p06,p08) | 10/12, 9.09 / 22.15° | 11/12, 6.09 / 22.15° |
  | ainex held-out (map 001–027, locate 028–053) | 26/26, 2.32 / 6.96° | 26/26, 2.32 / 6.96° (unchanged) |

  - Wall time of the camera split locate, 3 alternating runs before vs after (before the
    `turn_ransac` batching): 5.39–5.55 s → 6.24–6.62 s (+1.0 s). Single runs of the final code
    took 5.5–7.1 s, but the machine was shared and the noise was ±1.5 s; the final A/B was
    stopped behind the machine lock. The `features_matching` stage is +0.4 s (44 more pairs) and
    `pose` +0.3–0.6 s. Peak RSS is
    unchanged (~180 MB), and the inference server is not used.
  - The four chained captures (p07-low, p07-mid, p08-low, p08-floor) are 0.8–1.6° in yaw from
    the split map. 4–7 of 5–8 anchors agree for each, with 490–1470 inliers.
  - The pan interleave's errors come from its map, not from locate. Mapping p03/p05/p07 alone
    (two-step pan gaps) compresses the map's own yaw: p07 lies at 82° where the one-update map
    puts 94° (−11 to −13°). The held-out p06/p08 are located consistently with that wrong map.
    The chain added p06-low (4.2°) and correctly rejected p06-floor (1 of 3 anchors agree).
- **Finding / spec:**
  - The 0.9 target **is** reachable on the current split without guessing. Every chained pose
    rests on ≥ 15 verified matches with images that the map located from the same spot.
  - The first version (without the map gate) accepted a false pose on a synthetic walk map: a
    rotation alone explains 50–65 % of a walk's neighbouring matches, so the share alone cannot
    tell a turn from a walk. `map_turns_in_place` fixes it, and a COLMAP test covers it.
  - Spec risk: `mapper.md` says an image "the map cannot localise (not enough overlap with it)"
    is reported. Here an image with little overlap with the map is localised through the other
    inputs. Each image's result now depends on which other images are in the call. Both are not
    forbidden, but the spec should say it.
  - Split proposal (the user already decided to interleave): the even/odd interleave of 009–032
    is a good split. Every held-out capture has two mapped tilts of its pan position, main +
    exp-locate already scores 12/12 at 0.35 / 1.30°, and the 0.9 target stays meaningful. Do
    **not** interleave by pan position: it tests the mapper's yaw over two-step gaps
    (11–13° off), not locate. To keep the "unseen side" case, keep the current split as a second
    metric.
- **Verdict:** keep, with exp-locate.
- **Open:**
  - When the map itself is wrong, anchors inherit its error (pan interleave).
  - A joint rotation averaging over all queries could also refine the anchors.

### 13. exp-export-thresholds: confirmation and export rules consistent with "published stays published"
- Stack: main's (Python 3.12, PyTorch MPS, COLMAP 4.2, MoGe-2 ViT-L, GeoCalib, YOLOE-26x); Python / COLMAP at runtime: yes / yes. No new model, no download.
- Started from: `d4df2c0` + main's diff; patch: `~/oh-my-slam-data/experiments/exp-export-thresholds.patch` (no own changes: reverted; main's diff only). The attempt itself: `~/oh-my-slam-data/experiments/exp-export-thresholds/order_of_addition.own.patch`.
- Hypothesis (from exp-maskcluster): the extra objects of a split or image-by-image map come from export rules that depend on how evidence is grouped into updates. A map built one image per update confirms a one-detection object as soon as its keyframe arrives, because no other keyframe has it in view yet (`views_in_frustum` <= 1). It publishes it, and the object stays published. One update sees the later keyframes too and leaves the same object a candidate. Judging the same rule in order of addition should make every build order publish the same objects.
- Findings before any change:
  - **The probes' "extras" were not unexcused.** exp-maskcluster's `cmp.py` (and `bench/analyze.py`) call `stability_metrics` without `published`. With what the earlier updates published, as the evaluator passes it (`cmp2.py`), main's real maps have **0 unexcused extras** on office image by image, office [4,4,5] and office [6,7], and on camera image by image. All the extras are excused (published early: office cup, car, comb; camera 9 objects).
  - **What does fail in the evaluator** is `map.camera_stability.unexcused_extra` = 3 in `20261026T123734Z`: dryer 1, doorway 8 and person 58. Of these:
    - dryer 1 was confirmed with one view, under the cloud-point minimum at update 1 (54 < 85), and exported later.
    - doorway 8 is a grouping difference (once: doorway 4 holds frames 2, 3, 10 and 13; split: 4 holds frame 2 and 8 holds the rest).
    - person 58 is confirmed in both maps (2 reliable views), but has 0 cloud points in one update against 105–129 in the split.
  - **With the same geometry, grouping is order-free.** A new CPU replay (`replay_seq.py`) feeds the stored keyframes of the one-update maps (poses, depth, validity, detections) to main's objects stage regrouped as [4,4,5], [6,7], 12/12/12 or one per update. The office splits then give IoU 1.00, centre 0.000 m and id 1.00, and the camera splits the same. **The real IoU loss (0.35 office, 0.12–0.28 camera) is therefore entirely per-keyframe pose/depth, not the objects stage.** Only the extras differ.
- Change tried (`mapping/objects.py`): confirmation judged in order of addition.
  - New `MapObject.views_at_first`: the keyframes, in index order (= order of addition), that detected the object or have it in view, counted up to its first reliable detection (`viewing_frames`, `views_to_first_detection`). `confirm` takes one reliable detection when that count is <= 1, i.e. when the first keyframe that could see the place detected the object. This is exactly what image by image already does.
  - New test `test_a_single_detection_confirms_as_it_would_have_in_order_of_addition`. All 114 tests of the 6 object test files passed unchanged.
  - A variant B, by monkeypatch only (`variant_views.py`): a confirmed object with >= 2 reliable keyframes is exported whatever its cloud points.
- Measured with: probes in `~/oh-my-slam-data/experiments/exp-export-thresholds/`:
  - `replay_seq.py` (CPU replay of the objects stage for main's code, update by update, with optional rebuilds; it reproduces the real office one-update map at IoU 0.744, same ids);
  - `replays.sh` / `cmp_all.sh` (`r/{before,after,afterB}`);
  - `cmp2.py` (`stability_metrics` with the earlier updates' `-t full` scenes as published);
  - `probe.sh` (a copy of exp-maskcluster's, runtime `exp-export-thresholds`) for the real office runs in `real_after/`, under heavy + guard;
  - `cmp_real.sh`.
- Results:

  | Probe (unexcused extras; matched; IoU) | before (main) | after (rule A) | A + B |
  |---|---|---|---|
  | replay, same geometry, office [4,4,5] / [6,7] / one per update | 0 / 0 / 0; matched 1.0 | 0 / **1 (car)** / 0; matched **0.9** (shelf) | same as A |
  | replay, same geometry, camera 12/12/12 / one per update | 2 (person, basket) / 0; 1.0 | 2 / 0; 1.0 | **1** (basket) / 0 |
  | replay, own geometry, office image by image | 0; 1.0; IoU 0.306 | 0; 0.9; IoU 0.433 | same as A |
  | replay, own geometry, camera image by image (34 rebuilds) | 0; 0.80; IoU 0.067 | 0; 0.81; IoU 0.051 | 0; **0.76**; IoU 0.042 |
  | real office image by image (13 updates, 9–26 s each) vs its own one-update objects | 0 (3 excused); 1.0; IoU 0.352, centre 0.043 m | 0 (1 excused); 1.0; IoU 0.368, centre 0.042 m | — |
  | real office [4,4,5] | 0; 1.0; IoU 0.758; id 1.0 | 0; **0.9**; IoU 0.689; id 1.0 | — |
  | real office [6,7] | 0; 1.0; IoU 0.636; id 0.875 | **1 (car)**; 1.0; IoU 0.561; id 0.90 | — |
  | camera id agreement (replay splits) | 1.00 | 1.00 | 1.00 |

  - Rule A makes comb 30 (office) agree in every build order, but it exposes two other order dependences:
    - **Cloud points.** Car 20 is now confirmed in every order. It has 0 cloud points in one update against 31–41 in the splits, so [6,7] exports it in its last update, unexcused.
    - **The floor.** Shelf 39 (frame 6) is exported by one update. In every split, its detection is dropped as the floor named as an object, because `floor_z` is fixed by the first update of a map.
  - B removes camera person 55, but exports objects with no cloud points (camera one update 25 → 29 objects) and lowers camera image-by-image matching (0.80 → 0.76).
  - On the real office [6,7] split, rule A adds 1 unexcused extra, which the evaluator would count as a regression (`map_update.split_6_7.vs_one_update.unexcused_extra`).
- Gates: with the change, ruff, mypy and lint-imports passed, and so did the 6 object test files (114 + 1 new). After the revert, the worktree equals main's working copy: ruff, mypy and lint-imports pass (11 contracts kept). The full suite with coverage was not run, because no code change is kept.
- Verdict: **rejected** (reverted). Judging confirmation in order of addition is principled and costs nothing. But on its own it moves the disagreement to the cloud-point threshold and the first update's floor. It adds an unexcused extra on office [6,7] and matches less on [4,4,5], and it does not fix camera.
- Open:
  - (1) The cloud-point minimum is the order-dependent rule now: an object's map-cloud votes differ between one update and a split (car 0 vs 31–41, person 0 vs 105–129). Pixels retired by within-update input order differ from those retired across updates. Make `apply_input_order` and `apply_latest_wins` retire the same pixels, or count an object's cloud support per keyframe instead of from the final vote.
  - (2) `floor_z` is fixed by a map's first update (`update_objects`: it is set only when the map has no old frames). Splits then drop detections as "the floor" that one update keeps; recompute it from all keyframes on every update.
  - (3) Rule A is worth re-trying once (1) and (2) are fixed.
  - (4) The office and camera IoU gap is all geometry: replays with the same keyframes give IoU 1.0.
  - (5) `bench/analyze.py` and exp-maskcluster's `cmp.py` overstate unexcused extras (no `published`); use `cmp2.py`.

### 14. exp-eval-splits: the evaluator checks the two splits the user asked for (interleaved camera locate, one image per update)
- Stack: main's (Python 3.12, PyTorch MPS, COLMAP 4.2 SIFT + pycolmap, MoGe-2, GeoCalib, YOLOE-26x); Python / COLMAP at runtime: yes / yes. Evaluator-only change: no mapper, model or dependency change.
- Started from: `d4df2c0` + main's diff; patch: `~/oh-my-slam-data/experiments/exp-eval-splits.patch`
- Changes (user decisions 2026-11-08):
  - `tools/evaluate/suite.py`:
    - `pose.camera_locate.*` is now the even/odd interleave (`interleaved`): of 009–032 (the split map's first two updates) the even captures 010…032 are mapped in one update (`maps/camera_interleave`, run `mapper_camera_interleave`) and the odd 009…031 located there (`camera_locate_interleaved`, read-only contract). The reference is the first mapped capture (010). Headings and "vs mapped" come from the one-update camera map, carried into the interleave map's frame by `split_alignment`. Target unchanged (0.9).
    - The old split (locate 021–032 after the first update) is kept as `pose.camera_locate_unseen.*`, with the same targets. ainex's `pose.locate.*` is unchanged (`CaptureSequence.interleave`, `.unseen`).
    - A new section 5 maps `office_sequence`, `camera` and `ainex-captures` one image per update (`mapper.sh update -t full`, capture order, `ok_exit` 0/5, 30 min per update, a timeout stops the series). It runs after the office maps and is judged against each sequence's one-update map (`single_views`). `--skip-image-by-image` (`Evaluation.image_by_image`) skips it; its metrics are then not expected, and the summary says it was skipped.
  - `tools/evaluate/imagebyimage.py` (new) records `image_by_image.<office|camera|ainex>.*`:
    - `vs_one_update.*` (`mapquality.stability_metrics`, every earlier update's `-t full` objects as `Published`);
    - `refused_images` (exit 5; an update that fails otherwise leaves it without a value);
    - `per_update_median_s` (accepted + refused updates) and `last_update_s`.
  - `__main__.py`, `__init__.py`: the `--skip-image-by-image` flag. `report.py`: an "One image per update" section, a per-series table, and the two camera locate tables. `locate.py`: docstring.
  - `examples/targets.json`:
    - `image_by_image.*` targets: `vs_one_update` = `map.stability` (paired ≥ 0.9, unexcused 0, label ≥ 0.95, id ≥ 0.8, centre ≤ 0.10 m, extent ≤ 0.2, IoU ≥ 0.5).
    - `refused_images` ≤ floor(5 % of images), as the `registered_fraction` ≥ 0.95 targets: office 0, camera 1, ainex 3.
    - median and last update ≤ the sequence's slowest split-update target: office 51 s, camera 85 s, ainex 140 s (tol_rel 0.35).
    - Also `pose.camera_locate_unseen.*` and updated rationale.
  - `high_level_spec.md` §5 (pose accuracy: locate on interleaved and unseen captures; map quality: one image per update, refusals, time per update, skippable); `README.md` (benchmark list, CLI, `pose.*` and new `image_by_image.*` rows).
  - Tests: `tests/unit/test_evaluate_imagebyimage.py` (new, 13 tests), plus `test_cov_evaluate_suite.py`, `test_cov_evaluate_report.py` and `test_evaluate_runner.py` adjusted.
- Measured with: `~/oh-my-slam-data/experiments/exp-eval-splits/probe.py` (the evaluator's own new methods, real commands, under `heavy` + `guard.py`). The reference one-update maps are main's evaluator run `20261026T123734Z` (`maps/office`, `maps/camera_single`). Output is in `run1/probe.json`.
- Results (main's mapper code, i.e. what the new metrics say about main):
  - office one image per update: 13/13 accepted, 0 refused, median 17.0 s, last 21.4 s, total 224 s (round 3: 222 s, median 17). vs one update: paired 1.00, unexcused 0, label 1.00, id 1.00, centre 0.037 m, extent 0.03, **IoU 0.41 (fails 0.5)**.
  - camera interleaved locate: **9/12 located (0.75, fails 0.9)**; yaw error median 0.45°, max 1.47°. Unlocated: 009 floor, 021 p06 high, 023 p06 low. exp-locate-motion reports 12/12 for main + exp-locate. 54 s (map 12 images + locate).
  - Contracts on these runs: stdout, OpenLABEL, colour and readonly all pass.
- Added evaluator time: about 50 min with the image-by-image group. Round 3 main took office 222 s, camera 1436 s and ainex 1350 s, and this probe gives office 224 s. The interleaved split adds about 1 min. `--skip-image-by-image` brings the run back to about +1 min.
- Gates:
  - `pytest -m "not models and not browser and not eval" --cov=oh_my_slam`: 1694 passed, 1 failed, 24 errors in 26 min. Line and branch coverage is 100.00 % (19404 statements, 4970 branches).
    - The failure is the known one (`test_cov_server_process::test_a_model_failure_answers_500…`).
    - The 24 errors are `tests/unit/test_js_units.py`: `bun` is not installed on this machine, and no JS changed.
  - ruff, mypy (117 files) and lint-imports (11/11) all pass.
- Patches:
  - `~/oh-my-slam-data/experiments/exp-eval-splits.patch`: the full worktree diff against `d4df2c0`, main's diff included.
  - `exp-eval-splits.own.patch`: these changes only.
- Verdict: keep (evaluator change requested by the user). Not run: the full evaluator (orchestrator), camera and ainex image by image.
- Open:
  - `CLAUDE.md`'s evaluator command line does not list `--skip-image-by-image` (not edited by this agent).
  - The time targets are rules, not measurements: re-derive them from the first full run.
  - `baseline.json` has no value for the new metrics: they show as "not compared" until it is re-recorded.

### 15. exp-da3-depth: Depth Anything 3 in the depth slot, conditioned on the mapper's poses over co-visible windows
- **Stack:** main's stack with Depth Anything 3 (DA3) in place of MoGe-2 in the depth slot. Code:
  `ByteDance-Seed/Depth-Anything-3@3d835ec` (Apache-2.0), installed with uv overrides that drop
  numpy<2, xformers, evo, e3nn, moviepy and pre-commit, plus `addict`, which DA3 imports but does
  not declare. Weights:
  - default (`OH_MY_SLAM_DA3_ANYVIEW=large`): the any-view network of `DA3-LARGE-1.1` (0.35 B
    parameters; the HF card says Apache-2.0, the README table CC-BY-NC-4.0) with the metric
    network taken from `DA3NESTED-GIANT-LARGE-1.1`;
  - `giant`: all of `DA3NESTED-GIANT-LARGE-1.1` (1.40 B, CC-BY-NC-4.0).

  Both backbones run in fp16 on the server's MPS thread, with no CUDA-only piece: DA3's xformers
  SwiGLU falls back to its pure-PyTorch layer, its attention uses SDPA, and no model_editor
  request was needed. Python / COLMAP at runtime: yes / yes.
- **Started from:** `d4df2c0` + main's diff; patch
  `~/oh-my-slam-data/experiments/exp-da3-depth.patch`.
- **Hypothesis:** DA3 conditioned on the known poses over windows of co-visible keyframes gives
  more consistent multi-view metric depth than per-frame MoGe-2. That would fix camera frame
  agreement (pairs p90 ≤ 5 %; main 14 %) and the camera-set scale (doorway about 2 m; main
  1.68 m).
- **Changes:**
  - `server/models/geometry_da3.py` (new) replaces MoGe-2 as the `geometry` adapter.
    - It builds DA3's nested network (any-view + metric) from its configs, without DA3's
      `api.py`, whose export modules import moviepy, trimesh, pycolmap and evo.
    - Weights load with `torch.nn.init` stubbed out and `assign=True`.
    - The metric scale is DA3's least-squares fit, computed deterministically (DA3 subsamples
      at random) and with the caller's focal length where known.
    - The `keep_forward` focal re-solve rescales the kept result by the focal ratio.
    - It frees the MPS cache after a posed window.
  - `/v1/posed_depth` (protocol 2): `PosedDepthRequest/Response`, `client.posed_depth`, and the
    gateway `reconstruction.api.posed_depths`.
  - `mapping/api.py`:
    - New stage `posed_depth`, after the focal re-run and before the map frame. The placed
      keyframes are inferred again in windows of at most 6 co-visible keyframes (≥ 15 shared
      SfM points, else optical axes < 60°), conditioned on the model's poses and intrinsics at
      729 tokens per view. The map's undistorted keyframes of the same grid fill a window as
      context.
    - `_posed_placements`: an update's keyframes placed by anchored multi-view (or their own
      map) are first inferred on that placement, with the map's keyframes of their verified
      pairs as context, before `_depth_contradicted_placements` judges them.
    - A rebuild's stored keyframes are never inferred again (`_fresh`): they keep the map's
      aligned depth with retired pixels. Before this fix, office updates 4 and 10 (rebuilds)
      revived retired pixels, which cost the each-vs-once IoU 0.43 → 0.31.
  - Descriptor record:
    - `map.json → descriptor_model`, where an absent field means MoGe-2 (legacy).
    - An update whose server descriptor model differs re-indexes every stored keyframe's
      descriptor.
    - `mapper.sh locate` matches every keyframe when the models differ.
    - The DA3 descriptor is the metric network's class token (1024-d). On ainex, P@5 within 30°
      was 0.942, against 0.932 for MoGe.
  - Spec and docs: `Stage.POSED_DEPTH` is declared for `mapper.sh update`; `SKILL.md` was
    regenerated (16 stages); README and THIRD_PARTY_LICENSES were updated.
  - Tests: `test_cov_geometry_da3.py` runs the adapter with real CPU torch in a subprocess,
    around a stand-in `depth_anything_3` package. `test_mapping_posed_depth.py` covers the
    windows, posed placements, descriptor model and locate. The fakes have `posed_depth`.
- **Measured with:**
  - Probes in `~/oh-my-slam-data/experiments/exp-da3-depth/`:
    - `probes/depth_probe_da3.py` with exp-models' `depth_analyze.py`;
    - `probe.sh` (exp-maskcluster's, with this worktree's runtime);
    - `probes/map_probe.py`: the evaluator's `agreement_metrics` / `stability_metrics`, the
      doorway OBB, and box IoU by object size;
    - the ablation `probes/mapper_noposed.py`;
    - `probes/mem_probe.py`.
  - "before" is main's code: exp-maskcluster's `before/` maps, built with the same `probe.sh`
    on this machine. Main's office one-update map is `bench/main/office_sequence_once` (main
    cannot build it here).
- **Results:**

  | Probe | main (MoGe-2) | DA3 giant, no posed stage | DA3 giant + posed | DA3 large + posed (default) |
  |---|---|---|---|---|
  | camera one update, pairs median / p90 / >10 % / max (%) | 3.26 / **14.0** / 16.8 / 30.0 | 2.85 / 6.79 / 0 / 8.68 | 1.62 / **3.30** / 0 / 7.32 | 2.44 / **4.54** / 0 / 8.15 |
  | camera tilt pairs median / p90 / >10 % / max (%) | 3.52 / 9.76 / 11.1 / 23.7 | 2.83 / 6.26 / 0 / 7.83 | 1.63 / 3.38 / 0 / 7.32 | 2.99 / 4.01 / 1.85 / 20.6 |
  | camera keyframe pairs \|log ratio\| p90, before → after the scale adjustment | 0.214 → 0.077 | 0.189 → 0.075 | 0.094 → 0.041 | 0.149 → 0.060 |
  | camera doorway OBB height (about 2 m) | 1.68 m | 2.31 m | 2.25 m | 3.16 m (box merged; see below) |
  | camera keyboard / laptop / chair longest side | 0.17 / 0.17 / 0.55 m | 0.36 / 0.34 / 0.77 m | 0.37 / 0.34 / 0.75 m | 0.43 / 0.42 / 0.82 m |
  | camera floor below the first camera | 0.68 m | 1.08 m | 1.04 m | 1.23 m |
  | camera one update wall time (inference / focal re-run / posed) | 163 s (56 / 28 / –) | 281 s | 342 s (118 / 80 / 31) | 232 s (77 / 48 / 18) |
  | geometry compute per call (36 keyframes × 2 passes) | 0.41 s | – | 1.95 s | 0.89 s |
  | office AbsRel vs filtered SfM points (single / posed window) | 2.17 % | 2.51 % / – | – / 2.22 % (window 7) | 2.95 % / 2.20 % (window 6) |
  | office per-image log-scale spread (single / posed) | 0.047 | 0.083 / – | – / 0.032 | 0.073 / 0.035 |
  | depth time per keyframe, probe (single 1400 tok / posed 729 tok) | 0.33 s | 1.39 s / – | – / 0.79 s | 0.78 s / 0.41 s |
  | server memory at rest (allocated, MPS) | 6.66 GiB | 8.79 GiB | 8.79 GiB | 6.96 GiB |
  | cold start (models loaded) | 31–33 s | 33–40 s (54 s on a cold page cache) | same | 34–37 s |

  Office, image by image (13 updates) against one update (large, final code; main: exp-maskcluster `before/office_each` against `bench/main/office_sequence_once`):

  | Office | main | DA3 large + posed, final code (`after_large4`) | same, before the rebuild fix (`after_large3`) |
  |---|---|---|---|
  | one update, pairs median / p90 / max (%) | 2.39 / 5.67 / 6.93 | **1.89 / 3.79** / 6.70 | same map |
  | images refused image by image | 0 of 13 | 0 of 13 | 0 of 13 (4 of 13 before `_posed_placements`) |
  | image-by-image total (13 updates) | 262 s | 295 s | 304 s |
  | each vs once: paired / label / id | 1.00 / 1.00 / 0.875 | 1.00 / 1.00 / **1.00** | 1.00 / 1.00 / 0.857 |
  | each vs once: median box IoU, all | 0.352 | **0.432** | 0.309 |
  | small objects (longest side < 0.30 m): n, median IoU, median centre offset | 4, 0.161, 0.061 m | 4, **0.416**, 0.032 m | 4, 0.310, 0.032 m |
  | larger objects: n, median IoU | 4, 0.573 | 3, **0.646** | 3, 0.309 |
  | centre offset median, all | 0.043 m | **0.029 m** | 0.053 m |
  | `unexcused_extra` | 3 | 3 (car, tree, comb) | 5 |
  | one-update office map wall time | killed at 20 GB swap here (bench map from 1 Nov) | killed at 20 GB swap ×3; 71 s with COLMAP on 2 threads (`probes/mapper_lowmem.py`, probe harness only) |

  - **Pose conditioning is what brings camera agreement under 5 %.** DA3 alone, with no posed
    stage, removes the gross pairs (over 10 %: 16.8 → 0) and fixes the scale, but leaves p90 at
    6.8 %. The posed windows halve it (3.3 % giant, 4.5 % large).
  - **The camera-set scale comes from DA3's metric network,** not from the conditioning. On
    every DA3 map, desks, keyboards and laptops come out at their real size, about 2× main's:
    main's camera map is about 2× too small (round 3 saw "camera maps 2.1× main's" in every
    port). The doorway box is a noisy proxy: main's 1.68 m doorway sits in a map where a
    keyboard is 0.17 m. The large variant's 3.16 m doorway box absorbed more than the door.
    Its objects are about 15–20 % larger than the giant's.
  - **Giant on this 18 GB Mac:**
    - Its 1.7 GB of extra weights leave MapAnything too little memory.
    - Office image by image: update 4 (a rebuild through the multi-view fallback) ran out of
      memory on 4 views, and the 2-view chunks then ran 1126 s with the GPU at 98 %. I stopped
      it (main: 21 s).
    - The one-update office map was stopped at 20 GB swap, as main's is.
  - **Large:**
    - At rest it is +0.3 GiB over main.
    - MapAnything 5 views at 1024 run fine next to it once the cache of a posed window
      (+3–4 GiB in the driver) is freed (`mem_probe.log`). Without the free, office image by
      image failed at image 10 with a 5-view anchored chunk out of memory.
  - **Image by image (large):**
    - Without `_posed_placements`, images 7–10 of 13 were refused ("placement the depth of
      their links contradicts"). A single-view DA3 keyframe disagrees with stored keyframes
      whose depth was inferred on their poses (main refused none).
    - With it (and the cache freed after posed windows), all 13 were accepted.
- **Gates (final code):**
  - `uv run pytest -m "not models and not browser and not eval" -q --cov=oh_my_slam`: 1696
    passed, 1 failed, 24 errors, coverage **100.00 %** line and branch.
    - The failure is the known pre-existing
      `test_cov_server_process::test_a_model_failure_answers_500_with_its_type_and_message`.
    - The 24 errors are `tests/unit/test_js_units.py`: `bun` is not installed on this machine.
      They are environmental, and the JS is untouched.
  - ruff, mypy (117 files) and lint-imports (11 of 11 contracts kept) pass.
  - `SKILL.md` regenerated: `mapper-update` runs 16 stages.
- **Spec risks:**
  - Licences: the metric weights are CC-BY-NC-4.0 (the user allowed this), and DA3-LARGE-1.1's
    licence is inconsistent between its HF card and the README.
  - Protocol bumped to 2: a running older server is refused with the restart hint, as designed.
  - `reconstruct.sh` depth changes model:
    - The evaluator's single-image depth metrics will move.
    - DA3's validity is only "not sky", where MoGe also cut uncertain pixels, so clouds are
      denser: the camera cloud went from 0.9 M to 3.3–3.9 M points and the cloud stage from 6
      to 13 s.
  - Time: camera all at once 163 → 232 s (large). Every keyframe is inferred twice or three
    times: first pass, undistorted re-run, posed window.
  - The `posed_depth` stage is not in `examples/targets.json` (no target). The env switch
    `OH_MY_SLAM_DA3_ANYVIEW` is an experiment knob, not spec.
- **Verdict:** **keep as a candidate for the evaluator (large any-view, the default); not
  merged.**
  - For:
    - Camera frame agreement meets the pairs target: p90 14.0 → 4.5 % (giant 3.3 %).
    - The office one-update map's pairs p90 goes from 5.7 to 3.8 %.
    - The camera-set scale is fixed: objects at their real size, where main's are about half.
    - Office image by image vs one update improves. Median box IoU 0.35 → 0.43; small objects
      0.16 → 0.42, so per-keyframe depth was indeed the small-object loss (exp-maskcluster); id
      agreement 0.875 → 1.00; centre offset 4.3 → 2.9 cm; extras unchanged at 3.
  - Against:
    - The each-vs-once IoU is still below the 0.5 target (0.43). Camera image by image was not
      measured.
    - The camera tilt pairs keep one 20.6 % outlier (large).
    - Time: +42 % on camera all at once (232 against 163 s) and +13 % on office image by image
      (295 against 262 s).
    - Denser clouds.
    - The giant any-view network is clearly the better depth model, but it does not fit next to
      MapAnything on 18 GB. Re-test it on the 36 GB target machine.
- **Open / for a later DA3-for-poses experiment (MapAnything's slot):**
  - The any-view network already predicts poses: `cam_dec`, or `use_ray_pose=True` (the README
    says more accurate, slower). The nested network scales its translations to metres.
    `ref_view_strategy` reorders views when N ≥ 3 without poses.
  - Memory grows with the window: global attention every other layer from layer 13 (giant) or
    8 (large). Giant at 8 views × 1400 tokens reached 11.5 GB of driver memory on its own. Use
    the large and ≤ 6–8 views per chunk on 18 GB, and free the cache before MapAnything or
    anything else.
  - DA3-Streaming (chunked, sim(3) loop closure for long videos, street2) is CUDA-only as
    shipped: `alignment_torch.py` calls `.cuda()`, and `da3_streaming.py` uses
    `torch.cuda.get_device_capability` and `torch.cuda.amp`. It would need the model_editor
    route or a small patch.
  - Pose conditioning normalises translations by their median (clamped at 0.1). A pure rotation
    (camera set) is accepted: its depth scale then comes from the metric network alone.
  - Not done here: camera split / image-by-image probes with DA3, ainex, street2.

### 16. exp-map-long: Map-Long (VGGT-Long chunking + MapAnything) in the pose slot, every update replaying the whole ordered keyframe set
- **Stack:** main's stack (Python 3.12, PyTorch MPS, MoGe-2, GeoCalib, YOLOE-26x) with the pose slot
  replaced by Map-Long: VGGT-Long's chunked long-sequence pipeline (`DengKaiCQ/VGGT-Long` @ c160869,
  ICRA 2026) with the MapAnything backbone, on the Apache weights already in the cache
  (`facebook/map-anything-apache`, main's `/v1/multiview` adapter, unchanged). No download, no new
  dependency, no server change. COLMAP still extracts and matches SIFT (for `locate`, and the
  triangulated model); GLOMAP / incremental mapping no longer run. Python / COLMAP at runtime:
  yes / yes (SIFT only).
- **Started from:** `d4df2c0` + main's diff; patch `~/oh-my-slam-data/experiments/exp-map-long.patch`.
  The upstream clone is kept outside the worktree, `~/oh-my-slam-data/experiments/exp-map-long/VGGT-Long`
  (nothing of it is vendored; inside the worktree it broke `ruff check .`).
- **Upstream claims verified (code @ c160869):**
  - `torch.cuda.get_device_capability()` unguarded in `vggt_long.py:104` and
    `base_models/base_model.py:26`: raises `AssertionError: Torch not compiled with CUDA enabled` on
    this Mac (checked with torch 2.14.1). Device is `"cuda" if available else "cpu"` (never MPS).
  - xformers: only in the DINOv2 copy under `LoopModels/` (SALAD loop detector), behind
    `XFORMERS_AVAILABLE` with a pure-PyTorch fallback.
  - faiss-gpu: in `requirements.txt`, but loop detection uses CPU `faiss.IndexFlatIP`; GPU indices
    only in `LoopModels/utils/validation.py` (training).
  - C++ Sim(3) solver: `fastloop/solve.cpp` with `fastloop/solve_python.py` fallback (pypose +
    scipy), loop closure only.
  - Its MapAnything adapter feeds the first chunk's mean predicted K to every later chunk
    (`self.k`), and Map-Long's config uses SE(3) joins (`using_sim3: False`).
  - None of these block MPS for what was ported: the port re-implements the chunking and the
    IRLS joins in numpy on the mapper side and runs MapAnything through main's server adapter (single
    MPS thread). Loop closure (SALAD + Sim(3) pose graph) was not ported. No model_editor request.
- **Changes:**
  - `mapping/maplong.py` (new): `chunk_ranges` (VGGT-Long's chunking: chunk 16, overlap 8 — upstream
    60/30 does not fit the 13.3 GiB MPS cap), `ChunkView` (pose, K, depth, conf per keyframe and
    chunk), `weighted_sim3`/`robust_sim3` (weighted Kabsch/Umeyama + Huber IRLS, delta 0.1 m, 5 rounds,
    VGGT-Long's confidence threshold 0.1 × lower median, ≤ 200 k pairs), `align_chunks`, `poses`
    (later chunk wins a shared keyframe), `poses_within_memory` (an OOM restarts every chunk at half
    size), `depth_scale` (MoGe depth → MapAnything depth, median z ratio after reprojection).
    `OH_MY_SLAM_MAPLONG_SIM3=1` joins by Sim(3) instead (experiment switch).
  - `mapping/api.py`:
    - `POSE_ENGINE` (`OH_MY_SLAM_POSE_ENGINE`, default `maplong`; `sfm` = main; `maplong-refine`).
    - `_restart_plan`: with Map-Long every update with a stored map is a rebuild of all stored
      keyframes plus the new ones (main's `_try_rebuild` machinery: detections, published ids,
      retired pixels kept). Chunk boundaries come from the position in global frame-index order.
    - `_run_sfm` → `_maplong_model` for a new map / rebuild: chunked MapAnything (images only, no
      intrinsics, so a stored keyframe gets exactly the inputs it had), SE(3) joins, SIFT matches
      triangulated at those poses (method `multiview`, metric). `maplong-refine` also runs the
      multi-view fallback's `_refine_multiview` (+ one-centre test) and `_repose_on_aligned_depth`.
    - `_align_depths`: each keyframe's depth scaled to MapAnything's own depth of it
      (`depth_scale_method: maplong`, holding the gauge in `_adjust_depth_scales` like `sparse`).
    - `per_frame/*/depth_raw.npy` (float32, the network's unaligned depth) is stored, and a replay
      starts from it (with nothing carried), so a replayed keyframe's depth does not carry the
      previous alignment. Without it the replayed map's levelling drifted 0.3° (test).
  - `mapping/store.py` docstring; `README.md` (pose section); `THIRD_PARTY_LICENSES.md` (VGGT
    License row for the ported algorithm; MapAnything row now "every keyframe's pose").
  - Tests: `tests/unit/test_mapping_maplong.py` (12: chunking, prefix chunks ⊂ whole, Kabsch/IRLS,
    reflection, synthetic walk joined exactly, pose fallback, OOM halving, Sim(3) per-chunk scales,
    depth scale), `tests/unit/test_mapping_maplong_e2e.py` (7, real COLMAP + fake server: chunks,
    image by image = one update, OOM, unscaled keyframe, refine variant ×3); fake `multiview` writes
    depth/conf; `tests/conftest.py` runs every other test on `sfm` (env set at import + autouse
    fixture, marker `maplong`); `pyproject.toml` marker.
- **Measured with:** `~/oh-my-slam-data/experiments/exp-map-long/` — `probe_once.sh` (one update per
  example, `-t full`, guard, server `footprint` every 2 s), `probe_ibi.sh` (one image per update with
  a map clone after each), `analyze.py` (exp-eval-splits' evaluator code: `pan_pose_metrics` /
  `pose_metrics`, `agreement_metrics` / `pair_metrics`, `stability_metrics` with `Published`,
  `split_alignment`). Every run from a frozen copy (`before_wt` = main, `after_wt*` = this code),
  under `heavy`. Office one-update needs `mapper_lowmem.py` (COLMAP SIFT on 2 threads; exp-da3's
  harness) for both main and Map-Long: with the default threads both were killed at 20 GB swap
  during inference + early SIFT. The machine was shared with other agents' test suites.
- **Results (one update; "raw" = default `maplong` SE(3); total / stage.sfm in s):**

  | Probe | main | Map-Long raw | Map-Long Sim(3) | Map-Long + refine |
  |---|---|---|---|---|
  | office (13, low-mem harness) total / sfm | 56.7 / 2.6 | 71.5 / 16.0 | 70.1 / 15.5 | 72.9 / 15.9 |
  | office pairs agreement median / p90 / max % | 2.39 / 5.67 / 6.93 | **1.49 / 2.98 / 4.84** | same (1 chunk) | 3.08 / 5.92 / 7.21 |
  | camera (36) total / sfm (+pose_refinement, focal re-run) | 167 / 48 (+12, +30) | 139 / 51 (+0, +1) | 127 / 45 | 152 / 44 (+10, +26) |
  | camera tilt yaw diff median / max ° | 0.38 / 1.55 | 1.58 / 5.58 | 1.57 / 5.59 | 0.38 / 1.55 |
  | camera agreement tilt p90 / pairs p90 % | 9.8 / 14.0 | 13.3 / 13.6 | 13.3 / 13.8 | 8.5 / 12.8 |
  | camera objects / doorway height | 25 / 1.68 m | **6** / none | 7 / none | 25 / 1.72 m |
  | ainex (79) total / sfm | 304 / 112 | 254 / 127 | 299 / 130 | 301 / 125 |
  | ainex yaw err median / max ° | 2.24 / 8.87 | 4.83 / 15.0 | 5.98 / 15.9 | 2.22 / 8.89 |
  | ainex pairs p90 / same-heading p90 % | 1.55 / 4.08 | 22.2 / 27.0 | 7.1 / 23.0 | 1.55 / 4.08 |
  | street2 first 30 s (61 kf) total / sfm | 140 / 13.5 | 196 / 75.5 | 205 / 77.7 | 208 / 76.2 |
  | street2 30 s pairs median / p90 % | 6.41 / 12.7 | 4.05 / 11.2 | 4.07 / 10.2 | 4.14 / 13.0 |
  | street2 full (302 kf) total / sfm | 696–842 / 64 (bench, main) | **1175 / 421** (37 chunks; part of it beside a test suite) | — | — |

  Poses against main's on the same images (rigid fit; median / p90 / max °), raw: office rotation
  1.4 / 4.1 / 4.8, relative yaw 5.2 / 6.9 / 10.1, tilt 0.7 / 1.4 / 1.6, camera-set extent 2.07× main's;
  camera rotation 6.7 / 9.9 / 13.8, relative yaw 5.8 / 17.4 / 22.4, tilt 2.1 / 5.4 / 7.4, centres
  spread 85× main's one spot; ainex rotation 6.1 / 9.8 / 11.0, yaw 7.7 / 12.5 / 14.7; street2 30 s
  rotation 4.6 / 8.0 / 8.4, yaw 2.0 / 11.8 / 12.1, extent 0.86×. With refinement: camera 0.06 / 0.18 /
  0.53, ainex 0.01 / 0.02 / 0.04 (it converges back to main's multi-view fallback), street2 2.1 / 2.7.
  - **Join residuals** (median, 8 shared keyframes): camera 0.09–0.16 m; street2 0.06–0.80 m; ainex
    0.04–**7.9 m** (SE(3)), and with Sim(3) per-join scales of 1.15–**7.7**: MapAnything's metric
    scale is not stable from chunk to chunk on the rendered robot head turning in place, so the joins
    have nothing consistent to fit. Camera (a pan from one spot through a lens) gets its centres
    spread out (MapAnything places a turning camera's views apart), objects lose cloud support and
    only 6 of 25 are exported; the camera-set scale cannot be read (no doorway exported; the one
    desk is 1.41 m against main's 0.89 m).
  - **Server peak footprint** (phys_footprint, 2 s samples): main 9.3 / 12.3 / 14.3 / 10.2 GB
    (office / camera / ainex / street2 30 s); Map-Long 11.3 / 11.3 / 12.3 / 11.3 GB. 16 views per
    chunk ran without an OOM on every example (4:3 ainex included; main's anchored chunks of 24 hit
    OOM on ainex and fell back to 14). Swap guard never tripped except office at default SIFT threads
    (both main and Map-Long, before any pose code ran).

  **Office, one image per update (Map-Long raw) against its own one-update map, evaluator metrics:**

  | | main (exp-eval-splits run1) | Map-Long |
  |---|---|---|
  | refused | 0 of 13 | 0 of 13 |
  | matched / unexcused / label / id | 1.00 / 0 / 1.00 / 1.00 | 1.00 / 0 / 1.00 / 1.00 |
  | centre median / extent median | 0.037 m / 0.028 | **0.008 m / 0.010** |
  | OBB IoU median (target 0.5) | 0.41 | **0.884** (small objects 0.91) |
  | camera poses vs one update | — | **identical (0.00°, extent ratio 1.000)** |
  | per-update median / last / total | 17.0 / 21.4 / 224 s | 23.5 / 35.0 / 298 s |
  | per-update stage.sfm as the map grows (1→13 kf) | — | 0, 3.2, 3.3, 4.5 … 13.5, 14.8 s (≈1.1 s per stored keyframe) |
  | server peak | — | 13.0 GB |

  - Pair IoUs: tv 0.93, keyboard 0.90, key 0.69, wallet 0.92, wine glass 0.98, handbag 0.88,
    windows 0.36 / 0.73 / 0.38 (two windows split differently: image by image publishes 4 windows,
    car and comb early — excused). The **cup** is published by updates 1–11 and gone in the last two
    (latest wins), as in the one-update map, which has none: it agrees at the end.
  - Replay cost: an update re-runs MapAnything on every chunk, depth alignment, objects and cloud of
    the whole map; inference (MoGe/GeoCalib/YOLOE) runs only for the new image. Per update ≈ 8 s +
    2.1 s × keyframes on office (35 s at 13). For camera (36) about 75 s at the end; ainex (79) and
    street2 (hundreds of keyframes, video) would grow to minutes per update, and `depth_raw.npy`
    adds about 1.7 MB per keyframe on disk.
- **Gates:** ruff, mypy (117 files), lint-imports (11/11) pass. Full suite
  `pytest -m "not models and not browser and not eval" --cov=oh_my_slam`: 1700 passed, 1 failed (the known `test_cov_server_process::test_a_model_failure_answers_500…`), 24 errors (`test_js_units.py`: `bun` not installed, as on main) in 24 min; line and branch coverage 100.00 % (19491 statements, 5006 branches).
  SKILL.md unchanged (no API change; `test_agent_skill` passes).
- **Spec risks:**
  - Licence: the ported chunk/IRLS logic is a derivative of VGGT-Long, VGGT License v1 (Meta research
    licence: derivatives under the same terms, licence copy, acceptable use policy) — recorded in
    THIRD_PARTY_LICENSES.md; the user allows all licences.
  - "Each update replays everything" makes an update's cost grow with the map, and stored keyframes'
    poses change on every update (allowed by mapper.md? it already allows rebuilds; spec should say
    it). An abandoned rebuild still extends the map with main's SfM path (not Map-Long).
  - `OH_MY_SLAM_POSE_ENGINE` / `OH_MY_SLAM_MAPLONG_SIM3` are experiment switches, not spec.
  - A map written by main (no `depth_raw.npy`) is replayed from its aligned depth (with the field it
    carries), so its first Map-Long update is not exact.
- **Verdict: rejected as a replacement for the pose slot; keep the replay protocol as the finding.**
  - Raw Map-Long fails the turning sets: camera (tilt yaw max 1.55 → 5.6°, 25 → 6 objects) and ainex
    (yaw max 8.9 → 15°, pairs p90 1.6 → 22 %): MapAnything's chunk geometry for a camera turning in
    place is neither rigid nor of a stable scale, and no join (SE(3) or Sim(3)) recovers it. With
    main's refinement on top it converges back to main's multi-view fallback (0.01–0.06°) at the same
    cost: nothing gained.
  - It is slower where main uses SfM: street2 30 s 140 → 196 s (stage.sfm 13.5 → 75.5 s: 7 chunks of
    MapAnything), full street2 1175 s against 696–842 s (target 680; stage.sfm 421 s for 37 chunks,
    server peak 12.0 GB, `depth_raw.npy` +383 MB on a 1.4 GB map), office +15 s.
  - It helps where it is one chunk: office frame agreement p90 5.7 → 3.0 %.
  - The **replay protocol** is what makes image by image equal one update: office IoU 0.41 → 0.88,
    centre 3.7 → 0.8 cm, poses identical, no refusal — at +33 % update time on office. Any
    deterministic pose engine replayed this way would get the same property; main's own SfM is not
    deterministic across runs and was not tried with replay.
- **Open:**
  - Replay with main's engine (SfM/GLOMAP + multi-view fallback) on every update, to keep its
    accuracy on the turning sets and get the image-by-image agreement; cost grows with the map.
  - Map-Long with intrinsics fed to MapAnything (its own `calib` path) and with VGGT-Long's loop
    closure for camera's 360° pan: not tried.
  - Camera image by image (36 updates of replay) and ainex image by image: not run (time).

### 17. exp-n1: N1 — exp-cpp with main's own pose pipeline on COLMAP 4.2.1 as a C++ library (no Python, no MapAnything in the mapper)
- **Stack:** C++20 (exp-cpp: OpenCV 4.12, ONNX Runtime 1.31 CoreML EP + Core ML direct for MoGe-2 / GeoCalib / YOLOE), plus Homebrew COLMAP 4.2.1 (`libcolmap_*`, GLOMAP global mapper, Ceres 2.2, glog) linked read-only; Eigen switched to Homebrew's 5.0.1 (the one COLMAP is built with). Python / COLMAP at runtime: **no / yes (library, never the CLI)**.
- **Started from:** `d4df2c0` + exp-cpp's full working copy (worktree `exp-n1`, branch `exp/n1`); patch: `~/oh-my-slam-data/experiments/exp-n1.patch` (binary-safe, `git add -N` + `git diff --binary`).
- **Changes (native/):**
  - `src/mapping/sfm.{hpp,cpp}` (new): port of `mapping/sfm.py` on the COLMAP library — SIFT extraction (`CreateFeatureExtractorController`, 4096 features, doubled octave except video ≥ 1600 px, images registered first in keyframe order), pair matching + two-view verification (`CreateImagePairsFeatureMatcher`, seed 0), `verified_turns` / `two_view_stats` / `rotation_pairs` (turn residual of PLANAR_OR_PANORAMIC pairs), match graph / components, `verified_matches` (pinhole pixels of a lens camera), global mapper (`GlobalPipeline`, seeded) and incremental mapper (`IncrementalPipeline`, extension of an input model with frames fixed + `back_onto`), triangulation from poses (`RunPointTriangulatorImpl`), `extend_with_poses`, `set_distortion` / `drop_distortion` (SIMPLE_DIVISION), `vet`, `contradicted`, `epipolar_deg`, `baseline_ratio`; SIFT threads sized from free memory (12 MP doubled photos: 13 GB over all cores otherwise).
  - `src/mapping/panorama.{hpp,cpp}` (new): line-by-line port of `mapping/panorama.py` (Eigen): `refine_poses` (LM on left rotation increments and centres, graduated Cauchy 100/30/8/2.5 px, centre/focal/distortion priors, gravity tilt prior, division-model lens with its Jacobian, `depth_balance`, depth-noise whitening), `refine_turning`, `screen_pairs` + `pair_rotation` (numpy `default_rng(0).integers` reproduced bit for bit: PCG64 + Lemire 32-bit), `centre_hints`, `corner_shift`.
  - `src/mapping/sfm_place.{hpp,cpp}` (new): main's `_run_sfm` / `_extend` / `_refine_multiview` / `_one_centre` / `_repose_on_aligned_depth` / `_gravity_priors` / `_plausible_focal` / `_plane_incremental` / `_turns_through_lens` / `_complete_registration` in place of exp-cpp's MapAnything placement; the map keeps `sfm/database.db` (features + matches cached between updates; kept on a rebuild) and `sfm/model` (COLMAP model in map coordinates); early SIFT extraction of a new map while its keyframes are inferred (`_EarlyFeatures`).
  - **Deviation:** the rotation-only path's initial poses are chained through the verified pairs' own rotations (maximum spanning tree by inlier count, `pair_rotation`), not MapAnything's; the same screening and refinement then solve them. The mapper never calls MapAnything (the server still loads it lazily for nothing). `_join_unplaced` (merging other reconstructions, block rescaling, anchored MV of unregistered keyframes), `_own_map` / `_align_own_map` (video blocks), `_pan_onto_video`, `_calibrate_new_camera` (lens from matches: exp-cpp's GeoCalib lens is kept and handed to COLMAP as SIMPLE_DIVISION, the refinement holding it) and LightGlue re-matching are **not** ported.
  - `src/mapping/api.cpp`: placement → `sfm_placement`; record intrinsics = the SfM/refined camera; focal re-run (and undistorted re-run on a refined lens) with the posed camera; SfM scale not judged against (0.5, 2); dense alignment for refined keyframes; repose on the aligned depth; model persisted; exp-cpp's `one_centre` skipped (sfm_place does main's). Dead MapAnything glue removed.
  - `src/mapping/validity.cpp`: `well_registered` with main's rules for SfM (≥ 100 observations, ≤ 1.5 px) and refined poses (≥ 20 matches, ≤ 1°).
  - `src/mapping/transaction.cpp`: `sfm/` owned again (no longer legacy), `clone_for_edit`, `start_over(keep)`.
  - `src/mapping/locate_colmap.cpp` (new), `locate.cpp`, `locate_pose.cpp`: keyframe SIFT from the map's database (no re-extraction), query SIFT with COLMAP in memory, absolute pose by `EstimateAbsolutePose` + `RefineAbsolutePose`; exp-locate-motion's same-call anchors on a map turning in place (`located_by_neighbours`).
  - `cmake/Deps.cmake`, `CMakeLists.txt`: `oms_colmap` link target (mapping area only), Homebrew Eigen.
  - Tests: `tests/mapping/test_sfm.cpp`, `test_panorama.cpp` (new); `mapper_fake.hpp` renders textured views of the test room (`RoomScene::render`) so COLMAP can match them; `test_mapper_update.cpp` adapted.
  - `README.md`, `THIRD_PARTY_LICENSES.md`.
- **Measured with:** probes in `~/oh-my-slam-data/experiments/exp-n1/{probes,scripts}` (`probe_all.sh`, `map.sh`, `locate_probe.sh`, `posecmp.py`, `officecmp.py`, `videocmp.py`), all under `heavy` + `guard.py`, runtime dir `~/oh-my-slam-data/runtime/exp-n1`. References: main's evaluator maps `20261026T123734Z/maps`, exp-cpp's `eval-exp-cpp2/maps`, bench `main/street2_once`.
- **Results:** 
  Same probes, same machine (exp-n1 final binaries; exp-cpp from `eval-exp-cpp2`, main from `20261026T123734Z`):

  | Probe | exp-n1 | exp-cpp | main | target |
  |---|---|---|---|---|
  | office one update: total / features_matching / sfm | **41 s / 14.9 s / 0.86 s** | 54 s / — / 35.4 s | 180 s / 130.8 / 3.7 s | 78 / 40 / 2.6 s |
  | ainex (mapper_single): total / sfm + pose_refinement / features | **92 s / 0.10 + 5.2 s / 5.9 s** | 291 s / 221 s / — | 301 s / 129 + 36 s / 8.5 | 270 / 180 + 25 / 13 s |
  | camera one update: total / sfm + pose_ref | 82 s / 16.1 + 1.9 s | 80 s / 38.5 s | 179 s / 51 + 10.5 s | 160 s |
  | street2 first 30 s (61 keyframes): total / sfm / features | **83 s / 9.5 s / 6.0 s** | (full street2: sfm 491 s) | (round 1: 131 s total) | full street2 sfm 110 s |
  | camera split 12/12/12 per update | 21 / 25 / 20 s | — | 146 s (slowest) | 85 s |
  | office split 6+7 per update | 21 / 30 s | — | — | 51 s |
  | client peak RSS office / camera / ainex / street2 | 7.0 / 7.5 / 3.1 / 2.4 GB (SIFT during inference) | — | — | 15 / 8 / 8 / 14 GB |
  | server memory after each probe (status `allocated_gib`) | 2.6–3.2 GiB (MapAnything never called) | evaluator peak 12.7 GB | 15.4 GB | 16–20 GB |

  Pose accuracy against main's maps on the same images (rotations aligned, `posecmp.py` / `officecmp.py` / `videocmp.py`, evaluator functions for `pose.*` and `map.*`):

  | Map | rotation vs main (median / max) | centres vs main | yaw error median / max | tilt yaw diff median / max | frame agreement pairs p90 |
  |---|---|---|---|---|---|
  | office one update (13/13) | **0.0° / 0.0°** (identical SfM) | identical shape, scale ×1.035 | — | — | — |
  | office split 6+7 (13/13; rebuilt at update 2 as main does) | **0.0° / 0.0°** vs main's one-update map | identical shape, ×1.049 | — | — | — |
  | ainex (79/79) | 0.35° / 0.64° | — | 2.59 / 8.36° (main 2.24 / 8.87, exp-cpp 2.67 / 11.19) | — | 1.62 % (main 1.55, exp-cpp 2.89); same-heading p90 4.25 % (main 4.08, exp-cpp 6.33) |
  | camera one update (36/36) | 0.11° / 1.05° (exp-cpp: 10.7° / 16.8°) | — | — | 0.43 / 1.39° (main 0.38 / 1.55, exp-cpp 0.60 / 3.50) | 13.9 % (main 14.0, exp-cpp 6.28; target 5) |
  | camera split 12/12/12 (36/36) | 4.0° / 7.3° vs main's split (main's split is 0.2° from its one-update map) | — | — | 0.80 / 2.75° | 11.0 % |
  | street2 first 30 s (61/61) | 0.30° / 6.4° vs main's street2 map | 3.2 cm median, 0.37 m max; scale ×1.08 (exp-cpp: 2.6° / 7.0°, 0.39 m, ×1.30) | — | — | — |

  Camera locate (`locmetrics_split.py`, the evaluator's `held_out_metrics`):

  | Split | exp-n1 | exp-cpp | main (+ exp-locate + exp-locate-motion) |
  |---|---|---|---|
  | interleaved (map 010…032, locate 009…031) | **11/12 (0.92)**, yaw 0.43 / 1.13°, 6.3 s | — | 9/12 (0.75), 0.45 / 1.47° (main); 12/12 with exp-locate |
  | unseen (map 009–020, locate 021–032) | before anchors 5/12; **with same-call anchors 11/12 (0.92)**, yaw 2.24 / 3.56° | 4/12 (0.33) | 3/12 (main); 12/12 with exp-locate-motion |

  The one image left in each split: 021 p06-high (interleaved) is rejected by the epipolar gate (0.44° > 0.25°); 029 p08-high (unseen) has 2 map matches and no anchor agreeing.
  Locate time: 6–8 s for 12 images (features 5.9–6.5 s: COLMAP SIFT of 12 queries at full resolution, doubled; keyframes read from the database), pose 0.05–1.2 s.
  No-Python proof (`native/verify_no_python.sh`): every entry point and mode ran with the fakes (python, colmap CLI, …) never called (`fake-calls.log` empty) and no python/colmap process seen; its static link check failed only because it forbade linking COLMAP (round 2's rule) — changed to forbid Python, also through the COLMAP dylibs (all 7 binaries: 0 Python libraries).
- **Gates:**
  - Native unit tests (`native/test.sh`, LLVM coverage build, all areas): 470 tests; 468 passed and 2 failed in that run (`ServerNets.CoreMLPackageOnTheCpu`, `ErrorPathsOfTheWrappers`): the worktree copy lacked the **empty** folder `tests/data/server/tiny/tiny.mlpackage/Data/com.apple.CoreML/weights/` that Core ML requires (git cannot carry an empty folder, so the patch cannot either: create it after applying); restored, both pass (5/5 `ServerNets.*`). Mapping area: 148/148 (incl. new `Sfm.*`, `SfmMath.*`, `Panorama.*` and the reworked `MappingUpdate.*` on rendered textured views).
  - Coverage of `native/src/**`: **97.10 % lines / 89.29 % branches** (exp-cpp 99.09 / 92.79; target 100). Below target mainly in the new code: `sfm_place.cpp` 84.97 / 66.85, `locate_colmap.cpp` 42.99 / 22.73 (only real maps have a database), `api.cpp` 86.65 / 66.19, `sfm.cpp` 96.08 / 78.47, `panorama.cpp` 98.26 / 88.54. Measured with the 2 server tests failing (their lines counted as missed).
  - Python: `ruff check .`, `mypy src` (116 files), `lint-imports` (11/11 kept) pass; no Python changed. The doc-checking tests (`test_third_party_licenses.py`, `test_cli_single.py`, `test_evaluate_report.py`) 30/30 pass. The full Python suite was not run (no Python change; ~26 min under the lock).
  - `native/verify_no_python.sh`: every entry point and mode ran, no fake python/colmap called, no python/colmap process; the static link check now forbids Python (also through the COLMAP dylibs) instead of COLMAP itself; checked separately on all 7 binaries (0 Python libraries). Not re-run end to end after that edit.
- **Verdict:** **keep (stage 1 passes).** Main's own pose pipeline runs natively: office and the office 6+7 split reproduce main's poses exactly (0.0°), camera 0.11°, ainex 0.35°, street2 0.30°; the `sfm` stage drops from 221 / 35 / 491 s (exp-cpp) to 0.1 (+5.2 pose refinement) / 0.9 / 9.5 s (first 30 s of street2), every mapper probe is under its target, the server never runs MapAnything (2.6–3.2 GiB), and camera locate reaches 0.92 on both splits. Stage 2 kept exp-cpp's depth, objects, TSDF and Core ML models unchanged (no GTSAM / Open3D needed).
- **Open:**
  - Not ported from main's `_run_sfm` / `_join_unplaced`: merging other reconstructions, block rescaling (`_fix_blocks`), hanging photos, the dense test of a depth-contradicted pose, the collapse / gravity guards, `_own_map` / `_align_own_map` (a video added to a map as a block), `_pan_onto_video`, `_calibrate_new_camera` (a new camera's lens from its own matches in an update; exp-cpp's GeoCalib lens is used when taken), `_remap_small`, LightGlue re-matching of weak links and loose pairs. The full street2 (all clips, loops) and video-onto-map updates were not probed.
  - The camera split (12/12/12) ends 4.0° (median) from main's split map where main's split is 0.2° from its one-update map: the update path chains the new keyframes' rotations from the map and refines them; main anchors MapAnything there.
  - `locate_colmap.cpp` and the update / join branches of `sfm_place.cpp` need unit tests (coverage); the mapper's MapAnything placement code (`placement.cpp`) is now unused by the mapper but still compiled and tested.
  - Client peak RSS of a new camera map 7.5 GB (target 8 GB): SIFT runs during inference with threads sized from free memory (`extraction_threads`, 60 % of it).
  - The evaluator was not run (orchestrator).

### 18. exp-sam31: SAM 3.1 redraws the proposal model's masks from its boxes (arm A: YOLOE-26x proposes; arm B: LLMDet-L proposes)
- Stack: main's (Python 3.12, PyTorch 2.14 MPS, COLMAP 4.2, MoGe-2 ViT-L, GeoCalib, MapAnything, YOLOE-26x + MobileCLIP2) + SAM 3.1 (`facebookresearch/sam3@f1647e8`, weights `facebook/sam3.1@daa6319` `sam3.1_multiplex.pt`, 3.50 GB, SAM License, gated: access granted to the user's HF account) running in the inference server on MPS (fp16 encoder); arm B adds LLMDet Swin-L (`iSEE-Laboratory/llmdet_large@bec37f2`, 1.38 GB, Apache-2.0, `transformers` 5.22 MM Grounding DINO). Python / COLMAP at runtime: yes / yes.
- Started from: `d4df2c0` + main's diff; patch: `~/oh-my-slam-data/experiments/exp-sam31.patch` (reverted: main's diff only). The attempt itself: `exp-sam31.own.patch` (arm A, with its tests and licence rows; arm B's adapter is `exp-sam31/armB_seg_llmdet.py` + `armB_wiring_snapshot.diff`).
- Hypothesis: better masks at object boundaries (SAM 3.1 prompted with each proposal's box) give tighter OBBs, so more objects and steadier boxes across updates, in particular on small objects (cups, phones, keys: exp-maskcluster's IoU losses).
- Online checks (official sources):
  - SAM 3.1 (2026-03-27) is the "Object Multiplex" release: new checkpoint `sam3.1_multiplex.pt` holding a detector (`detector.*`: Perception-Encoder ViT 447 M params, text encoder 354 M, DETR encoder/decoder) and a multiplex tracker (`tracker.model.*`, with SAM-style interactive prompt encoder and mask decoder). Its README requires CUDA 12.6; no MPS path. Text prompts go through `Sam3Processor.set_text_prompt`; box prompts on the detector are *exemplars* (find all objects like this one), not per-instance masks; per-instance box masks come from the tracker's interactive SAM heads.
  - MLX port `mlx-community/sam3.1-bf16` (3.49 GB, `mlx-vlm >= 0.4.3`): text and box-guided detection on Apple silicon; not used (a second runtime next to torch MPS in the server, and the PyTorch path below worked).
  - LLMDet: Apache-2.0, merged into `transformers` 4.55; LVIS minival AP 51.1 (Swin-L), 48.3 (Swin-B), 44.7 (Swin-T), checkpoints are inference-only (no LLM). YOLOE-26x: 40.6 text-prompt AP on LVIS minival (Ultralytics docs). Claims confirmed.
- CUDA-only parts met and their fallbacks (no model_editor request needed):
  - `sam3/__init__` imports `sam3.model.edt`, which imports Triton at module level: replaced by an inert module when Triton is absent (`seg_sam31.stub_triton`). A global `triton` stub was tried first and broke `torch._dynamo` (it checks `triton.language.dtype`).
  - `sam3/perflib/fused.py`: the ViT MLP calls `aten::_addmm_activation` in bf16, not implemented on MPS: the MLPs get the unfused forward (`plain_mlps`).
  - `PositionEmbeddingSine(precompute_resolution=...)` allocates on `"cuda"`: built with `None` (the interactive neck does not use it). `TransformerDecoder` (detector) also hard-codes `device="cuda"` in its coord cache; avoided by not building the detector.
  - RoPE: `use_rope_real=True` (no complex tensors); checkpoint `freqs_cis` (complex) vs computed `freqs_cis_real/imag` are the only allowed state-dict mismatches.
  - Package-level imports also need `pkg_resources` (setuptools < 81) and `pycocotools`; `sam3`'s `numpy<2` pin is overridden (`numpy>=2.1`).
- Changes (arm A; reverted after the probes, kept in `exp-sam31.own.patch`):
  - `server/models/seg_sam31.py` (new): `Sam31BoxSegmenter` builds only the ViT trunk, the interactive neck branch and the interactive prompt encoder / mask decoder from the 3.1 checkpoint (no text encoder, no detector, no tracker memory); fp16 encoder weights on MPS; image read at 1024 px and squeezed to 1008², features cached for the last image (fine and coarse passes share them); boxes as corner-point prompts in chunks of 32; single-mask output thresholded at logit 0 on the caller's grid. A SAM mask replaces YOLOE's only if SAM says the box holds an object, after cutting it to the box widened by 10 % (≥ 4 px), and if its IoU with YOLOE's mask is ≥ 0.5 (`apply_refined`). Labels, scores, boxes and ids are YOLOE's; colours stay a function of id and masks are still unblended.
  - `server/models/seg_yoloe.py`: optional `refiner` (looked up in the registry at request time) redraws the masks after detection; detections the caller would drop for size (`SegmentRequest.min_share`; exact because an accepted SAM mask is at most 2x YOLOE's) are not redrawn.
  - `server/models/__init__.py`: SAM 3.1 registered as a required adapter (`segment_sam31`), wired into YOLOE.
  - `client/protocol.py`: `SegmentRequest.min_share`; `segmentation/detect.py` sends it (coarse pass: 5 % of the image).
  - `pyproject.toml` / `uv.lock`: `sam3` (git, pinned), `setuptools>=70,<81`, `pycocotools`, `numpy>=2.1` override; `transformers>=4.55` for arm B.
  - Tests: `tests/unit/test_seg_sam31.py` (10 tests: mask rules, Triton stand-in, unfused MLPs, loading/encoding/decoding with NumPy doubles of torch and sam3, 100 % line+branch of the module), `test_cov_server_models.py` (registry with the new adapter and the refiner lookup; YOLOE refiner hook incl. `min_share`).
- Arm B (measured, not kept): `server/models/seg_llmdet.py` (`OH_MY_SLAM_PROPOSER=llmdet`): main's 304 prompts in 4 chunks of ≤ 250 tokens, boxes ≥ 0.3 kept, cross-chunk NMS 0.7, score mapped 0.3→0.5, every mask from SAM 3.1 (box dropped if SAM finds nothing).
- Measured with (all under `heavy` + `guard.py`, runtime `~/oh-my-slam-data/runtime/exp-sam31`, scripts in `~/oh-my-slam-data/experiments/exp-sam31/`):
  - `seg_probe.sh` (cold start, `segment.sh -i -d` on restaurant.jpg x3, 3 office + 3 camera frames, server phys_footprint sampled at 0.5 s as the evaluator measures it) for before (this worktree with the SAM wiring disabled = main), arm A fp32, arm A fp16 (`after_A16`), arm B (`after_B`).
  - `probe_adapter.py` (in-process YOLOE + SAM on restaurant: per-stage times, both masks saved) and `boundary.py` (edge alignment: mean Lab gradient on mask boundary pixels / image mean; collage `restaurant_collage.png`, red YOLOE, green SAM).
  - `probe_llmdet.py` + `agree_labels.py` (LLMDet counts, time, label agreement with YOLOE's boxes at box IoU ≥ 0.5).
  - Mapper: `probe.sh` (copy of exp-maskcluster's) office_each / office_445 / office_67 / office_once and camera_once / camera_split; `cmp3.py` (= exp-export-thresholds' `cmp2.py` + per-object rows and a small-object median: cups, keys, wallets, keyboards, TVs, or OBB volume < 2 L).
- Results (before = this worktree with the SAM wiring disabled, i.e. main's code; same probes, same machine; mapper "before" = main's runs recorded in entries 11 and round 3, whose raw files were since deleted from `exp-maskcluster/before`):

  | Probe | before (main) | arm A: YOLOE + SAM 3.1 | arm B: LLMDet-L + SAM 3.1 |
  |---|---|---|---|
  | restaurant.jpg objects (`segment.sh -i`, ≥ 0.5) | 116 | 115 (114 of 115 masks redrawn, 1 kept YOLOE's) | in-process: 220 boxes ≥ 0.3 → 159 after NMS (19 ≥ 0.5, 105 ≥ 0.35); in the server: OOM, 0 |
  | label agreement with YOLOE's boxes (box IoU ≥ 0.5) | — | 1.00 (labels are YOLOE's) | 0.71 at 0.3 (79 of 115 YOLOE boxes matched), 0.74 at 0.25, 0.67 at 0.35 |
  | mask boundary edge alignment, restaurant (median over the 114 redrawn) | 1.835 | 1.975 (+8 %; better on 78 of 115); SAM/YOLOE mask IoU median 0.81, area ratio 0.945 | — |
  | segment.sh restaurant warm (s) | 3.37 / 3.23 | fp32: 9.1 / 8.1; fp16 + size filter: 7.66 / 7.50 | server: every request OOM (MPS 13.08 of 13.32 GiB); in-process LLMDet alone 12.5-13.5 s (4 chunks) + SAM 3.95 s for 220 boxes |
  | segment.sh office x3 / camera x3 (s) | 2.24-2.35 / 1.94-2.16 | fp32: 6.1-7.3 / 6.4-6.6; fp16: 5.3-5.7 / 3.9-5.1 | OOM |
  | in-process SAM on restaurant (fine pass, 115 boxes) | — | encoder 1.57 s, decoder 0.84 s (YOLOE 0.33 s) | — |
  | cold start (s) | 47.9 | 48.8 (fp32), 45.4 (fp16) | 66.9 |
  | server peak phys_footprint during seg probe (GB) | 10.24 | 13.76 (fp32), 11.88 (fp16) | 16.55 (and OOM at the MPS cap) |
  | office image by image (13 updates) | 262 s total; never stalls | fp32: update 4 **stalled**: MapAnything OOM on 4 views, 2-view retry crawled, killed at 1800 s. fp16: 270 s total (12.7-31.3 s per update) | not run (does not fit) |
  | office image by image vs main's one-update map (`bench/main/office_sequence_once`) | IoU 0.352, small objects 0.161 (tv 0.00, keyboard 0.10, key 0.00, wallet 0.22), centre 0.043 m | IoU **0.421**, small **0.300** (tv 0.27, keyboard 0.27, key 0.02, wallet 0.33), centre 0.037 m; 0 unexcused | — |
  | office [4,4,5] vs the same reference | IoU 0.758, small 0.681; 33/42/44 s | IoU 0.621, small 0.621; 50/50/49 s | — |
  | office [6,7] vs the same reference | IoU 0.636, small 0.533; 58/62 s | IoU 0.628, small 0.530; 76/89 s | — |
  | office one update | killed at 20 GB swap (150 s) | killed at 20 GB swap (163 s): COLMAP SIFT, as on main | — |
  | camera one update (36) | 152-163 s (inference stage 40.7 s) | 248 s (inference stage 139.9 s); 25 objects, same set as main's (matched 1.00, label 1.00, IoU 0.835 to main's map) | — |
  | camera split 12/12/12 vs its own one update | id 0.60, label 1.00, IoU 0.283 (entry 11) | id 0.64, label 1.00, IoU **0.259**, small objects 0.019 (cups 0-0.30, phones 0, speakers 0-0.01), 1 unexcused (doorway 8, as in the evaluator); 72/112/167 s vs 48/82/136 s | — |

  - The office comparisons are against main's one-update map built with YOLOE masks, because a one-update office map cannot be built on this 18 GB machine (main and arm A both killed at 20 GB swap). They therefore mix mask sources: arm A's splits lose IoU against a YOLOE reference they no longer share masks with (0.758 → 0.621), yet arm A's image-by-image map still agrees better with it (0.352 → 0.421). The only own-vs-own comparison, camera 12/12/12, does not improve (0.283 → 0.259; small objects 0.02).
  - Visual check (`restaurant_collage.png`): SAM follows bottle shoulders and chair legs where YOLOE's masks are blobby; on thin structures (a palm) both are poor.
- Verdict: **rejected** (reverted; worktree = main's working copy). Arm A gives visibly better masks (+8 % edge alignment) and keeps the object count (115 vs 116 ≥ 100) and labels, but box stability across updates does not follow: office image by image gains 0.07 median IoU against main's one-update map, the own-vs-own camera split loses 0.02 and its small objects stay near 0. It costs 2.3-3.4x per segmented image (restaurant 7.5 s against the 3.5 s target), +99 s of inference on a 36-image camera map, and with fp32 weights its 1.8 GB pushed MapAnything into an OOM crawl that stalled an office update; fp16 weights (0.9 GB) fixed that and keep the seg probe's peak at 11.9 GB (target 14). Arm B does not fit: LLMDet-L takes 12.5-13.5 s per image on the M3 Pro for main's 304 prompts (4 chunks of 256 tokens), its labels agree with YOLOE's on only 0.71 of matched boxes, and next to SAM 3.1 and main's models every segment request runs out of MPS memory (server peak 16.6 GB). This confirms exp-export-thresholds: the small-object box losses are per-keyframe depth and pose, not mask shape.
- Open:
  - If masks matter again (export, catalogue crops, viewer), redraw only small detections or only on `segment.sh -i`, not on every mapper keyframe; the encoder (1.6 s) dominates, so one encoding per keyframe is the floor.
  - SAM 3.1's own text grounding (detector + 354 M text encoder, ~830 M params) was not measured: with 304 prompts it is 304 decoder passes per image (prompts are separate queries), so only the labels YOLOE found could be asked; build it with `use_rope_real`, the coord-cache and position-encoding `cuda` defaults patched, and `plain_mlps`.
  - LLMDet-B/T may halve LLMDet's time, not the 4x gap; its 256-token prompt limit is the structural cost for a 304-label vocabulary.
  - The MLX port (`mlx-community/sam3.1-bf16`) would move SAM out of the torch MPS budget (separate allocator), which is what broke MapAnything; not tried.
- Gates (on the experiment's code, before the revert): ruff, mypy (117 files), lint-imports (11 kept) pass; `tests/unit/test_seg_sam31.py` + `test_cov_server_models.py` 52 passed, `seg_sam31.py` 100 % line and branch; full suite with coverage: 1692 passed, 1 failed (the known `test_a_model_failure_answers_500_with_its_type_and_message`), 24 errors (`tests/unit/test_js_units.py`: `bun` is not installed on this machine; no JS was touched), coverage 100.00 % line and branch. After the revert the worktree diff is byte-identical to `main-uncommitted-r4.patch`; ruff, mypy (116 files) and lint-imports (11 kept) pass, and `uv sync` removed sam3/pycocotools/iopath/portalocker and restored setuptools 84.

### 19. exp-longstream: LongStream (CVPR 2026) in the pose slot, the whole ordered keyframe set replayed on every update

- **Stack:** main's stack (Python 3.12, PyTorch 2.14.1 MPS, COLMAP 4.2 SIFT + pycolmap, MoGe-2, GeoCalib, YOLOE-26x, MapAnything). LongStream replaces COLMAP's global/incremental mapper as the pose source. Python / COLMAP at runtime: yes / yes. COLMAP is still used for SIFT matching and for triangulation at LongStream's poses.
  - Code: `3dagentworld/longstream@72399ad` (MIT), checked out in the worktree's `third_party/longstream`. It is not in the patch; clone it there.
  - Weights: HF `NicolasCC/LongStream@bdba22f`, `50_longstream.pt`. It is a 14.3 GB training checkpoint, which with 27 GB free would have taken the disk under the 15 GB floor. Only its `model` state dict (1412 fp32 tensors, 1.19 B parameters, 4.76 GB) was range-fetched from the torch zip into safetensors (`fetch_model.py`), at `~/oh-my-slam-data/experiments/exp-longstream/ckpt/`. It is symlinked into `~/Library/Caches/oh-my-slam/weights/longstream/`.
  - **The weights state no licence** (no licence tag or text on the HF card, checked 2026-11-08), so all rights are reserved by default. The vendored CroCo code is CC BY-NC-SA 4.0.
- **Started from:** `d4df2c0` + main's diff; patch `~/oh-my-slam-data/experiments/exp-longstream.patch`.
- **Claims checked:**
  - **Metric through a scale head: yes.** street2's first 30 s: scale within 8 % of main's MoGe-over-SfM scale (Sim(3) ratio 1.08). Full video: path 173.6 m against main's 172.7 m.
  - **No calibration: half true.** It needs no intrinsics, but the released checkpoint predicts no field of view either. Every frame gets upstream's fallback, f = max(H, W) / 2 (90° horizontal), so it gives no usable intrinsics. The mapper keeps its own focal, and the depth scale pairs pixels by image position, not through its K.
  - **Bounded KV cache, keyframe stride 8, refresh: yes.** Upstream's `streaming_refresh` clears the caches every `refresh - 1 = 3` keyframes, i.e. every 24 images.
    - Standalone, 60 frames at 518 × 280: fp32 0.97 s/frame with a 9.1 GiB driver peak (point head included); fp16 0.66 s/frame and 5.8 GiB.
    - fp16 against fp32 poses: 3 cm and 0.08° median, 7 cm and 0.21° max.
  - **SDPA with a fallback: yes.** Attention is `F.scaled_dot_product_attention`; xformers and CroCo's cuRoPE are optional, with PyTorch fallbacks.
  - **fp32: yes, upstream.**
  - **Port to MPS:** only `device: mps` was needed. No model_editor request was sent.
    - fp16 needed input-casting pre-hooks: upstream builds some float32 tensors, e.g. the DPT heads' inputs.
    - The fp32 server does not fit next to the other models on 18 GB: 6.6 GiB resting + 9.1 GiB peak against a 13.3 GiB cap. The server runs it in fp16 (`OH_MY_SLAM_LONGSTREAM_DTYPE`).
  - **batch_refresh mode:** not usable on MPS as shipped. It builds a dense float N×N attention mask over 25 frames × ~782 tokens, so SDPA's math path would need about 24 GB for the score tensor. Streaming mode was used.
- **Changes:**
  - **Server:**
    - `server/models/pose_longstream.py` (new): adapter `stream_poses`.
      - Upstream's streaming refresh, re-implemented so memory is bounded: the point head is dropped, and depth and confidence are written per image.
      - Its pose composition and image resize are checked equal to upstream's `compose_abs_from_rel` / `pose_encoding_to_extri_intri` and `load_images_for_eval`. That loader silently skips the BMPs the client sends.
      - Also: dtype switch and `cast_inputs` hooks.
    - Registered only when `OH_MY_SLAM_POSE_ENGINE=longstream` (`server/models/__init__.py`).
    - `server/app.py`: route `/v1/stream_poses`.
  - **Protocol and client:** `StreamPosesRequest/Response`, protocol 2 (`client/protocol.py`, `version.py`), and `client.stream_poses`.
  - **Reconstruction gateway:** `reconstruction/multiview.py` `run_stream_poses`, with a shared `_results`.
  - **Mapping:**
    - `mapping/longstream.py` (new): `StreamView`, and `depth_scale`, which pairs pixels by image position.
    - `mapping/api.py` (Map-Long's hooks, adapted):
      - `POSE_ENGINE` (default `longstream` here; `sfm` = main).
      - `_stream_model`: all keyframes in index order, then `triangulate_with_poses` with no bundle adjustment.
      - Depth scaled to LongStream's (`depth_scale_method: longstream`).
      - `_restart_plan` replays every stored keyframe on every update. Stored keyframes restart from `depth_raw.npy`, the unaligned depth, and no inference runs again.
      - `locate` is unchanged.
    - `store.py` docstring.
  - **Tests:**
    - `test_cov_pose_longstream.py` + `tests/fakes/longstream_runner.py`: real CPU torch in a subprocess and a stand-in package. They cover the refresh and anchor sequence, keyframe indices, compose, fp16 hooks and weight errors.
    - `test_mapping_longstream_e2e.py`: on the fake server, image by image gives the same poses as one update, and the depth scale is recovered.
    - Also: client, route and registry tests; `FakeClient.stream_poses`; a stub adapter; a `longstream` marker; the conftest engine fixture (other tests run `sfm`).
  - **Config and docs:** ruff excludes `third_party`; README (poses step); THIRD_PARTY_LICENSES row.
- **Measured with:** `~/oh-my-slam-data/experiments/exp-longstream/`.
  - `probe_ls.py` + `compare.py`: standalone fp32/fp16 against main's bench street2 map.
  - `probe_once.sh` / `probe_ibi.sh` / `batch.sh` (Map-Long's harness, own runtime). Before = the same worktree with `OH_MY_SLAM_POSE_ENGINE=sfm`, i.e. main's path; after = `longstream`, server fp16. Office runs used `LOWMEM=1` (COLMAP on 2 threads, harness only).
  - `analyze.py`: exp-eval-splits' `stability_metrics` / `agreement_metrics` / `judge_poses`, run with its venv.
  - `plaus.py`: walking speed, camera height above the ground from the cloud, and car boxes.
- **Results:**

  | Probe | before (main path) | after (LongStream fp16) |
  |---|---|---|
  | street2 first 30 s: total / sfm stage | 117.1 s / 11.8 s | 159.7 s / 47.2 s |
  | street2 30 s poses vs main (rotation median / max; relative yaw p90) | — | 1.9° / 6.1°; 6.0° |
  | street2 full 150 s: total / sfm stage (target 680 s) | 696 s / 64 s (bench, 1 Nov); evaluator 842 s | **1001 s** / 227 s (LongStream call 206 s, 0.68 s/keyframe) |
  | street2 full: path, speed | 172.7 m, 1.15 m/s | 173.6 m, 1.15 m/s |
  | street2 full: camera height above ground (median) | 1.88 m | 1.95 m |
  | street2 full: cars longer than 2.5 m (median length × height) | 4.23 × 1.22 m (16 cars) | 4.17 × 1.33 m (18 cars) |
  | street2 full vs main: ATE (Sim(3), RMSE / path); heading difference median / p90 / last | — | 1.06 %; 3.6° / 9.2° / 3.6° |
  | street2 full: end-to-start gap (an open walk: nothing to close) | 75.7 m | 64.0 m (Sim(3) scale to main 1.14: the route's shape bends) |
  | camera all at once (turning in place): total | 157.1 s | 111.3 s |
  | camera: tilt-yaw difference median / max | 0.38° / 1.55° | 1.15° / **8.71°** |
  | camera: yaw against main's, relative to the first capture (median / p90 / max) | — | **20.2° / 41.5° / 44.9°**: the pan is compressed about 22 % (p11 at 154° against 198°) |
  | camera: tilt against main's (median / max) | — | 1.96° / 4.95° |
  | camera: camera-centre spread (median distance to mean) | 1.3 cm | 15.3 cm (translation where there is none) |
  | camera pairs agreement p90 / over 10 % | 14.0 % / 16.8 % | 6.3 % / 3.8 % |
  | camera keyboard / laptop / desk longest side | 0.17 / 0.18 / 0.90 m | 0.45 / 0.64 / 1.77 m (doorway 2.74 m) |
  | office one update (LOWMEM): total / sfm stage | 62.9 s / 2.8 s | 78.5 s / 16.7 s |
  | office pairs p90; poses vs main (rotation median / max) | 5.67 % | 4.45 %; 1.69° / 2.39° |
  | **office image by image vs its own one update** (exp-eval-splits' code) | main: IoU 0.35–0.41, paired 1.0, id 0.875–1.0 (exp-maskcluster / exp-eval-splits) | **0 of 13 refused; poses identical (0.00°); paired 1.0, unexcused 0, id 1.0, centre 0.012 m, extent 0.048, IoU 0.782 (small objects 0.827); label 0.889 (one tree/window, fails 0.95)** |
  | office image by image: median / last / total | 17.0 / 21.4 / 224 s (exp-eval-splits) | 22.1 / 34.9 / 294 s. LongStream replay 2.3 s at 2 keyframes → 10.8 s at 13 (~0.83 s per stored keyframe, 518 × 378). sfm stage 3.2 → 14.6 s |
  | server at rest / peak footprint | 6.56 GiB / 9.2–12.0 GB | 8.72 GiB / 12–14 GB (street2 14 GB); no out of memory, not degraded; swap stayed under 5 GB |
  | cold start | 33–36 s | 47–52 s |

- **Gates:**
  - `pytest -m "not models and not browser and not eval" --cov=oh_my_slam` (under `heavy`): 1694 passed, 1 failed, 24 errors in 21 min 50 s. Coverage is 100.00 % line and branch (19565 statements, 5004 branches).
    - The failure is the known `test_cov_server_process::test_a_model_failure_answers_500…`.
    - The 24 errors are `test_js_units.py`: `bun` is not installed here.
  - ruff, mypy (118 files) and lint-imports (11 of 11) pass. `SKILL.md` is unchanged: no web API change, and `test_agent_skill` passes.
- **Verdict: rejected for street2 and for turning in place; the replay design is worth keeping.**
  - **street2 (the slot it was sent for):**
    - Slower: 1001 s against 696 s. The stream costs about 0.68 s per keyframe (206 s for 302 keyframes) where SfM took 64 s. SIFT matching stays for triangulation, and the objects stage was also slower here (266 against 155 s).
    - Its scale is plausible and close to main's (speed, camera height, cars), but main's already was.
    - Drift is moderate: heading p90 9° from main's SfM over 150 s, and the route bends (Sim(3) 1.14).
  - **Pure rotation (camera):** fails. It compresses the turn by about 22 %, to 45° of yaw error at the end of the pan, and invents 15 cm of translation. Its depth is metric, which fixes the camera-set scale (objects about 2.5× main's, as DA3 found) and agreement (pairs p90 14 → 6 %), but the poses are wrong.
  - **What did work: the global-rebuild protocol.** Replaying the ordered keyframe set on every update makes the image-by-image office map equal the one-update map in poses. Box IoU rose 0.35–0.41 → 0.78, the only target of image by image no implementation had met. The remaining miss is one label, from the objects stage's order rules.
    - Every update costs one stream over all keyframes (linear: +0.83 s per stored office keyframe), so long maps get dearer per update.
    - The same protocol with a faster or deterministic pose source (SfM on the whole set, or Map-Long) should give the same equality.
- **Open:**
  - The weights' missing licence.
  - fp32 does not fit next to the other models on 18 GB. Re-measure fp32 on the 36 GB target.
  - A keyframe stride matched to walking video: the model was trained on KITTI at 10 Hz, and street2's keyframes are 2 fps.
  - Running LongStream concurrently with per-keyframe inference, since it only needs the images: it would hide most of the 206 s.
  - The predicted field of view is unused (the checkpoint's FoV output is constant).
  - Camera image by image and ainex were not run.

### 20. exp-replay-main: every update replays the whole ordered keyframe set with main's own pose engine (COLMAP SIFT + GLOMAP + fallbacks), each stored keyframe as its first inference gave it
- **Stack:** main's (Python 3.12, PyTorch MPS, COLMAP 4.2.1 SIFT + pycolmap GLOMAP, MapAnything multi-view fallback, MoGe-2 ViT-L, GeoCalib, YOLOE-26x). No new model, no download, no new dependency. Python / COLMAP at runtime: yes / yes.
- **Started from:** `d4df2c0` + main's diff; patch `~/oh-my-slam-data/experiments/exp-replay-main.patch` (full worktree diff, main's diff included; own changes only: `exp-replay-main.own.patch`).
- **Hypothesis:** exp-map-long's replay protocol (16) with main's pose engine instead of Map-Long keeps main's accuracy and makes image by image = one update on office, camera and ainex.
- **What exp-map-long's replay was missing for main's engine** (found by reading main's rebuild, `_try_rebuild` / `_stored_frame`): a stored keyframe re-entered a rebuild with its *aligned* depth, its *retired* validity, its *refined* intrinsics passed as EXIF (so `_camera_prior`, `_focal_carried` and `_rerun_focal` saw a different keyframe), its trimmed instance masks as detections, and the database camera as an earlier update had refined it, with pairs verified under an earlier update's prior focal length. Each of these makes the replay differ from one update. Raw depth alone (Map-Long's `depth_raw.npy`) was not enough.
- **Changes** (`src/oh_my_slam/mapping/`):
  - `api.py`:
    - `REPLAY` (`OH_MY_SLAM_REPLAY`, default on; `0` = main's rule) and `_replays`: an update replays when the map has keyframes and every earlier update took the same kind of input. `_restart_plan(…, replay)` then returns every stored keyframe, with no `RESTART_MAX_KEYFRAMES` cap.
    - `FirstPass` (`NewFrame.first`, snapshot right after `_infer_frames`) saved once per keyframe as `per_frame/<f>/first_pass.npz` (depth float32, validity, grid RGB, descriptor, intrinsics, K_grid, gravity, EXIF intrinsics, frame meta) plus `first_dets.json` (detections with the id each was first published with). `_replayed_frame` rebuilds the stored keyframe from them (nothing carried: `field=()`, `lens=None`); first ids come from `first_dets.json`, else the last update's `instances.json` by detection position (`det`). `start_over` keeps the `first_pass.npz` files.
    - A replay's new keyframes are inferred as a new map's (`_lens_of([])`); the undistorted re-run then happens for all keyframes as in one update.
    - `_run_sfm`: in a replay the stored camera is reset to the camera a new map of these keyframes starts from (`Sfm.reset_camera`, prior = median focal estimate or EXIF), and `_sync_verification` drops every two-view geometry when that camera differs from the one recorded in `map.json → verified_camera`; `match_pairs` then re-verifies every pair from its stored matches (no matching). Features and matches are never recomputed.
    - `Rebuild.replay`: vacated places are not carried (judged again), the video pan path (`_pan_onto_video`, pan pairs) is skipped, notes say `replayed`. A replay that places stored keyframes but none of the input is refused (exit 5, map unchanged).
  - `sfm.py`: `Sfm.clear_verification`, `Sfm.reset_camera`, public `camera_params`.
  - `objects.py`: `instances.json` records each detection's position (`det`); a rebuild's detection that carries a published id is never "the floor named as an object" (see published objects below).
  - `store.py` docstring, `README.md` (*Replay* under update semantics).
  - Tests: `tests/unit/test_mapping_replay.py` (8: one image per update = one update with constant and spread focal estimates, real COLMAP + fake server; refusal; a map written without first pass; `_replays`; first-pass round trip ×2; the floor exemption); `tests/conftest.py` runs every other test with `REPLAY` off (marker `replay`, `pyproject.toml`); `test_mapping_cov_api.py` stand-in frame fields.
- **Measured with:** `~/oh-my-slam-data/experiments/exp-replay-main/`: `probe_once.sh`, `probe_ibi.sh` (one image per update, `-t full`, map.json + frames.json copied after each accepted update), `chain.sh`, `analyze.py` (exp-map-long's: exp-eval-splits' `stability_metrics` with `Published`, `split_alignment`, pose comparisons). All runs under `heavy` + `guard.py`, frozen code copies (`before_wt` = main, `after_wt` = replay, `after2_wt` = replay + floor exemption). Office one-update *and* office image by image both use `mapper_lowmem.py` (COLMAP on 2 threads) so the thread count is the same. "Before" for camera and ainex image by image is round 3's main run (`bench/main`, same main code: `main-uncommitted.patch` = `main-uncommitted-r4.patch`), compared here against this worktree's main one-update maps.
- **Results:**

  One image per update against one update (evaluator metrics; target paired ≥ 0.9, unexcused 0, label ≥ 0.95, id ≥ 0.8, centre ≤ 0.10 m, IoU ≥ 0.5):

  | | office main | office replay | camera main (r3) | camera replay | ainex main (r3) | ainex replay |
  |---|---|---|---|---|---|---|
  | refused images | 0 / 13 | 0 / 13 | 0 / 36 | 0 / 36 | **39 / 79** | **0 / 79** |
  | paired | 1.00 | 1.00 | 0.84 | 1.00 | 0.12 | 1.00 |
  | unexcused extra | 0 | 0 | 1 | 0 | 18 | 0 |
  | label / id | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.75 / 0.75 | 1.00 / 0.97 |
  | centre median | 0.044 m | **0.000 m** | 0.045 m | **0.000 m** | 0.40 m | **0.000 m** |
  | median box IoU (small objects) | 0.36 (0.16) | **1.00 (1.00)** | 0.23 (0.15) | **0.996 (0.996)**; 0.989 with the floor exemption | 0.10 (0.02) | **1.00 (1.00)** |
  | poses vs one update: rotation median / max | 0.58 / 4.47° | **0.00 / 0.00°** | 0.24 / 1.48° | **0.00 / 0.00°** | 11.0 / 175° | **0.00 / 0.00°** |
  | per update median / last / total | 17.3 / 15.3 / 225 s | 21.4 / 21.8 / 266 s | 43.4 / 71.9 / 1436 s | 44.5 / 108.4 / 1726 s | 8.4 / 81.8 / 1350 s | 86.5 / 208.0 / 7437 s |

  - Ainex id agreement 0.97: one object keeps an id an earlier update published (allowed by mapper.md).
  - Camera IoU 0.996, not 1.00: a stored keyframe keeps the update that added it, so the fusion's keyframe order (`order_key`: updates oldest first) differs from one update's content order; poses and depth scales are identical.

  Cost of a replay as the map grows (update wall time at the 1st / 25 % / 50 % / 75 % / last image):
  - office 10.7 / 19.4 / 22.8 / 23.3 / 21.8 s (main 8.2 / 17.6 / 17.6 / 18.8 / 15.3; office is EXIF, SIFT of the new photo dominates).
  - camera 7.9 / 24.0 / 47.3 / 74.7 / 108.4 s; summed stages: sfm 635 s, focal/undistorted re-run 498 s, pose refinement 137 s, cloud 108 s, objects 97 s, features+matching (incl. re-verification) 78 s.
  - ainex 7.6 / 42.0 / 84.2 / 143.3 / 208.0 s; summed: sfm 3351 s, pose refinement 946 s, objects 809 s, focal re-run 773 s, depth alignment 611 s, cloud 532 s. Last update: sfm 85 s, refinement 36 s, objects 30 s.
  - Re-verification (camera, ainex: no EXIF, so the median focal prior moves almost every update): 35 and 78 updates re-verified up to 298 and 983 stored pairs; it is a small part of features+matching. Office (EXIF): never.
  - Disk: first-pass files add ~1.2 MB per keyframe (office 60 → 79 MB, camera 65 → 93 MB, ainex 145 → 192 MB).
  - Street2 by 30 s clips: not run (time; the lock queue was long).

  One update (same probe, `before_wt` vs `after_wt`): poses identical (office, ainex 0.00° rotation/yaw/tilt, extent 1.000; camera 0.00°, extent 0.998, tilt-yaw max 1.544 vs 1.545°: the multi-view fallback's run-to-run noise); frame agreement, objects (8 / 25 / 33) and pose metrics unchanged; time 59 / 148 / 257 s → 60 / 151 / 262 s (first-pass files).

- **Published stays published?** Not strictly with the replay as first built (`after_wt`): every published id that vanished came back, but for one update:
  - camera updates 14, 16, 27: a dryer, a toilet and a cup vanished for one update and came back the next. Each coincided with the replay's floor (`floor_z`, re-estimated from all keyframes as one update does) naming their detections "the floor" (`floor_named` 4 / 7 / 1). 14 and 16 listed them in `objects.removed`, 27 not (cup 129).
  - office update 7 (tree) and 13 (cup): latest wins, as in one update (main removes the tree too).
  - ainex: none.
  - Fix kept: a rebuild's detection carrying a published id is exempt from the floor test (`objects.update_objects`). Result in `after2`: see below. An object a replay no longer supports otherwise stays published while its founding detection is in the map (a rebuilt object that takes a published id is published whatever the rebuild's evidence); it disappears only by latest wins (pixels retired, place vacated, listed in `removed`), or, without the exemption, by the floor. The final maps are unaffected either way (they equal one update).
  - With the exemption (`after2`, office and camera image by image again): camera's flickers at updates 14 and 16 are gone; one remains, cup 130 missing from update 27 only, with nothing in `removed` and `floor_named` 0 (cause not established: that update's `objects.json` was not kept; candidates are a box not fitted that update or the id resolving provisionally into another object, `rebuild_merged`). Office: only the two latest-wins removals. vs one update: office paired 1.00, unexcused 0, IoU 1.00, centre 0.000 m, poses 0.00°, 13/13, median / last / total 19.2 / 21.1 / 246 s; camera paired 1.00, unexcused 0, label/id 1.00, IoU 0.989 (small 0.99), centre 0.000 m, poses 0.00°, 36/36, 46.6 / 107.2 / 1731 s, 33 objects against 32 without the exemption (the extra one is a published object the split map keeps, excused). Ainex was not re-run with the exemption (it had no flicker; the exemption only acts where a published detection is floor-named).
- **Gates (final code):** `uv run pytest -m "not models and not browser and not eval" -q --cov=oh_my_slam` (under heavy): 1689 passed, 1 failed (the known `test_cov_server_process::test_a_model_failure_answers_500…`), 24 errors (`test_js_units.py`: `bun` not installed, as on main), coverage **100.00 %** line and branch (19378 statements, 4970 branches). ruff, mypy (116 files), lint-imports (11/11) pass. No API change: `SKILL.md` unchanged. No new component (THIRD_PARTY_LICENSES unchanged); README documents the replay.
- **Spec risks:**
  - Cost grows with the map: every update re-runs SfM, the focal/undistorted re-runs (server inference of every stored keyframe without EXIF), depth alignment, objects and cloud of the whole map. Ainex's last update takes 208 s (exp-eval-splits' target: ainex ≤ 140 s per update, tol 0.35 → 189 s fails); camera's 108 s (target 85 s, ×1.35 = 115 s passes). Total image-by-image ainex 7437 s against 1350 s.
  - Stored keyframes' poses change on every update (mapper.md allows rebuilds; it should say so for every update).
  - A replay that leaves out a stored keyframe is abandoned and the map extended (main's rule); none happened here. An image the replay cannot place is refused (exit 5).
  - Mixed input kinds (photos then a video) are not replayed (main's rules apply). Video replay (street2 by clips) is implemented but not measured.
  - A map written by main (no `first_pass.npz`) is replayed from what it holds (main's `_stored_frame`), so its first replay is not exact.
  - `OH_MY_SLAM_REPLAY` is an experiment switch, not spec.
- **Verdict: keep (the hypothesis holds).** With main's own engine, replaying the ordered keyframe set from each keyframe's first inference makes the map built one image per update the map built in one update: poses identical to 0.00° on office, camera and ainex, median box IoU 0.36 → 1.00 (office), 0.23 → 0.99 (camera), 0.10 → 1.00 (ainex), unexcused extras 0, and ainex's 39 refused images → 0. One-update maps are unchanged (main's accuracy kept). Against: cost grows with the map (ainex image by image 1350 → 7437 s, last update 82 → 208 s; camera 1436 → 1731 s, last 72 → 107 s; office +10 %), and published objects can still vanish for one update (camera cup, 1 case of 36 updates after the floor fix). The image-by-image targets of exp-eval-splits pass for agreement and refusals; ainex's per-update time target does not.
- **Open:**
  - Cost: cache what a replay recomputes identically (the focal/undistorted re-run of a keyframe for an unchanged lens; objects of untouched keyframes), or replay only every N updates and extend in between.
  - Street2 by 30 s clips (video replay) not measured.
  - Camera IoU 0.99: make the fusion order of a replay independent of the stored update ids.
  - Cup 130's one-update disappearance on camera (update 27): keep each update's `objects.json` in the probe and find the rule.

## Researched, not tested (online only, `~/oh-my-slam-data/experiments/research_late2026.md`)

| Candidate | Why not tested |
|---|---|
| VGGT-Ω (CVPR 2026) | Depth is not metric; 16.5 GB at 100 frames; gated non-commercial licence |
| VGG-T³ (CVPR 2026) | No Apple port |
| LongStream, Mamba-VGGT | CUDA-only research code; worth watching for street2 scale |
| Depth Anything 3 | No MLX or Core ML port found |
| SAM 3.1 (Core ML community port) | About 3.5 GB extra on 18 GB, cost per prompt, SAM licence |
| InfiniDepth | Best boundary F1 in MoGe-3's own table; port and licence unknown |
| CalibAnyView | No code released as of 2026-10-31 |
| Core AI (macOS 27) | Beta runtime; Ultralytics export covers YOLO26 only, not YOLOE |
| MegaLoc for retrieval | Still the place-recognition candidate; no 2026 model clearly beats it |

## Open questions and ideas not yet tried

- Image by image vs one update: on identical keyframes the objects stage gives the same map in every build order (exp-export-thresholds replay: IoU 1.00, ids 1.00). The real gap is per-keyframe pose and depth. Also make the cloud-point export count and the floor height order-independent, then retry rule A.
- Correction: exp-maskcluster's and round 3's `analyze.py` comparisons call `stability_metrics` without what earlier updates published, so their "unexcused extras" are overcounted. Counted as the evaluator does, main's office and camera image-by-image maps have 0; the evaluator's real failure is `map.camera_stability.unexcused_extra` = 3.

- camera_locate: solved on the current split by exp-locate-motion (12 of 12) through same-call anchors. Spec `mapper.md` should say that one image's locate may depend on the other images of the call. For the evaluator, an even/odd interleave of 009–032 is a fair extra split; interleaving by pan position bends the map itself.

- Make the image-by-image map match the all-at-once map (spec `mapper.md:79-84`):
  - first add a one-image-per-update split to the ground truth for office, camera and ainex, so the
    evaluator checks it;
  - then fix the scale drift and the duplicate objects. Rust is the best starting point.
- Why C++ refuses camera images image by image when Rust accepts them.
- Combine exp-runtime and exp-locate in one branch (round 1); re-record `baseline.json` after review.
- Camera-set scale (all cases).
- Export more MapAnything v2 shapes (every N from 13 up), so street2 and ainex can use single calls.
- MoGe-derived gravity in place of GeoCalib (round 1, not tried).
- A descriptor-model record in maps before any model swap.
- An MLX stack: MapAnything on MLX plus a depth model on MLX, via mlx-swift for a native port.

## Template for a new entry

```
### N. <worktree>: <one-line hypothesis>
- Stack: language, runtime, models and versions; Python / COLMAP at runtime: yes/no
- Started from: <commit + diff>; patch: ~/oh-my-slam-data/experiments/<worktree>.patch
- Changes:
- Measured with: evaluator run dir / bench dir / probes
- Results: failed / regressions / key times / image by image vs all at once
- Verdict: selected / keep / rejected, and why
- Open:
```
