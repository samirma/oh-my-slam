# Recommendations — future experiments and A/B tests

Status: 2026-09-22. The baseline is the plan in the `oh-my-slam` devtask
(`~/.dev-workflow/oh-my-slam.md`) as built. None of the items below is part of that plan; each one
needs a measured result before it is adopted.

Where the baseline stands:
- SAM 3 access is denied (U1), so segmentation runs on YOLOE alone.
- Gate G4 chose SIFT features.
- MapAnything-apache anchors rotation-dominant input.
- Textures come from a per-face best-view atlas, with OpenMVS seam levelling turned off.
- Score calibration is the identity map until the datasets are approved (U2).

Sources:
- the research reports of 2026-09-22 (online only);
- the gate measurements in the devtask's Implementation Notes;
- the validation inputs in `/Users/U124317/robot_view`.

---

## 1. Future experiments

Every experiment should plug in behind a switch (an environment variable or config key), the way
`OH_MY_SLAM_FEATURES` already does, so the baseline stays the default until the experiment has been
measured.

| # | Experiment | Where it plugs in | How it would be used | Expected benefit | Cost / risk | Prerequisite |
|---|---|---|---|---|---|---|
| E1 | **SAM3-LiteText-L** as the segmentation refiner (ungated `vil-uob/sam3-litetext-l`; `Sam3LiteTextModel` exists in the pinned transformers 5.17) | `server/models/seg_sam3.py`; switch `OH_MY_SLAM_SEG_REFINER=sam3|litetext|none` | YOLOE proposes ≤ 8 concepts, then LiteText re-segments and re-scores them. Its scores become the scores of record, because 0.5 is SAM 3's native operating point. | Unblocks gate G3 without gated weights. SA-Co/Gold cgF1 53.1 vs 54.1 for SAM 3 (arXiv 2602.12173); text encoder 3.7× faster; better masks, hence tighter OBBs | The image encoder comes from SAM 3, so treat the weights as SAM Licence (the HF card says Apache). MPS latency is unknown. | User accepts SAM Licence terms |
| E2 | **SAM 3.1 on MLX** (`mlx-community/sam3.1-bf16`, mlx-vlm) | Separate server model adapter, running on the GPU thread | Fallback for E1 if the PyTorch/MPS path misses the latency budget. Object Multiplex handles up to 16 objects per pass. | Native Metal; the model card reports about 40 ms/frame tracking on an M3 Max | Adds a second ML framework to the server. It is unverified whether MLX and torch can share the GPU safely in one process. | E1 result |
| E3 | **LoMa matcher** (built into Homebrew COLMAP 4.2, MIT) | `mapping/sfm.py`; `OH_MY_SLAM_FEATURES=loma` (LOMA_B / LOMA_B128) | Used for hard inputs: low texture, wide baselines, lighting changes between update sessions | +18.6 mAA on HardMatch and +12.4 on IMC2022 over ALIKED+LightGlue (arXiv 2604.04931). Could raise registration on months-later updates. | ALIKED was already ~25× slower than SIFT on ainex (149 s vs 4 s of matching), and LoMa is likely slower still. CoreML may fall back to the CPU. | — |
| E4 | **Pi3X** as an alternative multi-view backend | `reconstruction/multiview.py`; switch `OH_MY_SLAM_MULTIVIEW=mapanything|pi3x`. Via MapAnything's `pi3x` wrapper, if the pinned commit `3d10cf7` has it; otherwise the Pi3 PR #153 patches | The same chunked, pose-anchored path used for rotation-dominant or low-parallax input | Pi3X accepts known poses and intrinsics as input and outputs approximate metric scale. Strong relative geometry (RealEstate10K AUC@30 85.9 for π³). | Weights are CC-BY-NC; Mac support is an unmerged PR; no published metric-accuracy numbers | — |
| E5 | **Loop closure between multi-view chunks** (design reference: VGGT-Long's "Map-Long" mode) | `mapping/sfm.py` anchored-chunk path | Detect revisits by DINOv2 descriptor, estimate the SE(3) between chunks, then optimise a pose graph over the chunk transforms | Closes the ainex 026↔078 heading gap (8.3° now), and in general stops chunk drift on long rotation-dominant sequences | The ainex gap may be partly in the data: the filename angles are commanded headings, and the notes found the return half drifts about +5° | — |
| E6 | **Score calibration on real data** (`tools/calibrate_scores.py` is already implemented) | `segmentation/data/calibration_*.json` | Fit isotonic maps on 500 LVIS val images for each segmentation path | `--min-score 0.5` then means "≥ 50% precision"; needed so YOLOE and SAM 3 scores sit on one scale | Downloads COCO val + LVIS annotations (~1.2 GB) | U2 consent |
| E7 | **Periodic map consolidation** | `mapping/api.py` (new step) | Every N updates, or when the scale spread grows past a threshold, re-run `global_mapper` on all keyframes, align it to the current map frame with Sim(3), re-fuse, and re-express the OBBs. Ids must be kept. | Removes drift frozen in by `fix_existing_frames` across many updates | Map coordinates move slightly after consolidation; costs a full re-map | — |
| E8 | **Multi-view-consistent depth for fusion** (MapAnything with poses and intrinsics as input) | `reconstruction/depth.py` + `fusion.py`; switch `OH_MY_SLAM_FUSION_DEPTH=mono|mvs` | Replace per-frame MoGe depth plus scale alignment with depth predicted jointly across views | Sharper meshes and fewer doubled surfaces. MapAnything reports ScanNet rel error 3.34 with poses+K vs 22.2 without (arXiv 2509.13414). | About 0.4 s per view on MPS (G6: 24 views in 9.2 s) | — |
| E9 | **Gaussian-splat layer**: Brush trains from the COLMAP model; Spark renders it in three.js | Optional `mapping` post-step + a new viewer layer | An extra photorealistic layer next to the mesh | Much better view synthesis than a textured mesh | Outside the exact-colour rule; Spark's r186 compatibility and training time are unverified | — |
| E10 | **AnyCalib focal cross-check** (Apache; COLMAP publishes an ONNX export) | `reconstruction/api.py` intrinsics priority | For images without EXIF, e.g. ainex: compare the MoGe FOV, AnyCalib FOV and COLMAP focal, and warn or choose when they disagree | Reduces metric-scale error from a wrong FOV (≈ 2% of scale per degree at 60° FOV) | One more model | — |
| E11 | **Amodal OBBs as a second opinion** (Boxer, runs on MPS, CC BY-NC) | `segmentation/obb.py` | Compare with the floor-grounding prior (C29). Could replace it for single images where the floor is invisible. | Boxes covering the full object instead of the visible surface; on CA-1M, Boxer 0.412 vs CuTR 0.250 mAP (with depth) | Non-commercial licence; integration effort | — |
| E12 | **Texturing quality** | `reconstruction/texture.py` | Try the OpenMVS `develop` build, or mvs-texturing, to get seam levelling back (disabled because the 2.4.0 arm64 build corrupted textures). Also try a 400k-face budget. | Fewer visible seams and exposure jumps between faces | OpenMVS scales superlinearly with faces (250k faces took 32 s; 740k took over 10 min) | — |
| E13 | **Adaptive keyframing** | `core/video.py` / `mapping/ingest.py`; `OH_MY_SLAM_KEYFRAMES=fps|adaptive` | Keep a frame when estimated overlap with the last keyframe drops below about 65% (tracked features), instead of a fixed 2 fps | Fewer redundant frames on slow walks (livingroom), more frames on fast turns; faster builds at equal coverage | Needs cheap per-frame tracking during sampling | — |
| E14 | **VGGT-Ω as a pose second opinion** (watch) | `reconstruction/multiview.py` | Only if access is granted and the licence (non-commercial) is acceptable | State-of-the-art camera accuracy (ETH3D AUC@30 90.4) | Unknown scale, gated, code hard-codes `cuda` | Licence and access |

Also watch, without an experiment slot yet:
- LingBot-Map: streaming poses; not metric; Mac support unmerged.
- depth-anything.cpp: runs DA3 without xformers.
- MoGe-3: doesn't run on macOS; an MLX port exists at about 2 s/image.
- Apple SHARP: single image to a 3D Gaussian-splat scene.
- Apple Core AI's SAM 3 segmenter: Swift runtime only.

---

## 2. A/B tests

**Protocol, applies to every test**
- Warm server, runs one at a time, nothing else using the GPU.
- ≥ 5 repetitions for timings; report median and p95.
- Fixed random seeds, and the same inputs for both arms.
- Outputs go to `~/oh-my-slam-data/experiments/<UTC>-<AB id>/`. Never create a folder named
  `benchmark` or `eval_results`.
- The only difference between arms is the switch; record the git state and the switch values.
- Results go into the devtask's Implementation Notes. Adopt only if the decision rule passes and
  the offline and validation suites (`tools/validate_inputs.py`) still pass.
- Keep a fresh copy of each map per arm, because updates mutate maps.

| # | Hypothesis | A (baseline) | B (and C) | Inputs | Metrics | Decision rule |
|---|---|---|---|---|---|---|
| AB1 | A SAM 3-class refiner improves segmentation quality within budget | YOLOE-26x alone, identity calibration | YOLOE → SAM3-LiteText-L (E1); C: SAM 3.1 MLX (E2) | restaurant.jpg; 20 keyframes each from livingroom, church and ainex; LVIS 500 once U2 is approved | Precision/recall at score ≥ 0.5 (LVIS, or 40 hand-labelled images); mask boundary F-score; AC25 chair/person height pass rates; duplicate objects per map (AC28: 1 TV, 1 sofa, 1 coffee table); JSON latency p50/p95 | B wins if precision@0.5 rises ≥ 5 points **or** map duplicates fall, **and** `reconstruct.sh` JSON p50 ≤ 3.0 s. C only if B misses the latency budget. |
| AB2 | Learned matchers raise registration on hard inputs enough to pay their time | SIFT (current G4 choice) | ALIKED+LightGlue; C: LoMa-B (E3) | church.mp4, livingroom.mp4, ainex; plus a later-session update (e.g. a new clip of the same room) | Registration %; mean reprojection error; mean track length; per-frame metric-scale spread (IQR/median); features+matching time; total `update` time | Registration must be ≥ 95%. Among arms that pass: B/C win only if they register ≥ 3 points more frames, or shrink the scale spread by ≥ 20%, at ≤ 2× the total update time. Otherwise stay on SIFT. |
| AB3 | Pi3X anchors rotation-dominant input better than MapAnything | MapAnything-apache, chunked and anchored | Pi3X, same path (E4); later with E5 loop closure on both arms | ainex-captures (79), ainex split 1–40 / 41–79 | Yaw residual vs the filename angles (raw and single-offset-fitted, median and max); 026↔078 loop error; camera-centre radius; metric scale ratio vs MoGe; time and peak memory at 24 views | B wins if the median fitted yaw residual improves ≥ 30% **and** the loop error does not grow, with peak memory ≤ 20 GB. The licence is recorded, not a veto. |
| AB4 | Multi-view depth gives cleaner fused geometry | Per-frame MoGe + scale alignment | MapAnything with poses and intrinsics as input (E8) | livingroom.mp4, church.mp4 | RMS distance of floor/wall points to their fitted plane; doubled-surface count along viewing rays; AC28 dimension pass rate; mesh screenshots rated blind; seconds per keyframe | B wins if plane RMS drops ≥ 25% and doubled surfaces fall, at ≤ +1 s per keyframe |
| AB5 | Better texturing without visible seams or wrong views | Per-face best-view atlas | OpenMVS `develop` with seam levelling; C: mvs-texturing (E12) | ainex, livingroom, church maps (same mesh for all arms) | Colour jump across neighbouring faces (seam visibility); atlas sharpness (variance of Laplacian); wrong-view artefacts found in the viewer; texturing time | Win = seam metric −20% with no increase in artefacts, at ≤ 2× time. Blind side-by-side screenshots break ties. |
| AB6 | Adaptive keyframing gives the same map faster | Fixed 2 fps | Overlap-driven keyframes (E13) | livingroom.mp4 (92 s), church.mp4 (25 s) | Keyframes; registration %; cloud coverage (voxel count at 5 cm); object duplicates; AC27/AC28 checks; total `update` time | B wins if coverage stays ≥ 98% of A, all AC checks still pass, and total time drops ≥ 20% |
| AB7 | More mesh faces are worth the texturing time | 200k faces | 400k faces | livingroom, church maps | Blind geometric detail rating; distance from the fused cloud to the mesh surface; texturing time; `view.sh` first-render time | B only if the cloud-to-mesh distance improves ≥ 15%, texturing stays ≤ 420 s and first render ≤ 5 s |
| AB8 | Stronger association cues reduce duplicated and merged objects | Projected-mask IoU ≥ 0.3 with a size-scaled gate | B: IoU ≥ 0.4; C: add DINOv2 appearance cosine (weights 0.4/0.4/0.2) | livingroom map; ainex as two updates; synthetic change scenes | Objects per physical item (target 1 for TV, sofa, coffee table); wrong merges; id stability across the split update (AC26); false deletions under 5% depth + 1° pose noise | Win = fewer duplicates with zero new wrong merges and zero false deletions |
| AB9 | Latest-wins margin trades false invalidation against missed removals | τ(z) = max(0.15 m, 0.10·z) | B: max(0.10, 0.08·z); C: max(0.20, 0.12·z) | Synthetic remove/move/add scenes; living-room single-frame update (static scene); ainex re-map | % of cells falsely invalidated in static scenes; % of removed-object pixels detected; object removals on static data (must be 0) | Pick the arm with 0 false object removals and the lowest false-invalidation rate while still detecting ≥ 60% of removed-object pixels |
| AB10 | Floor grounding (C29) gives plausible single-image heights without inventing geometry | Grounding on | Off; C: Boxer amodal boxes (E11) | restaurant.jpg; 20 living-room frames; 10 church frames | % of chair/table/person heights inside their class ranges; boxes wrongly extended (support not actually on the floor, judged in the viewer); OBB volume change | Keep grounding unless C beats it by ≥ 10 points on in-range % without more wrong extensions |

---

## 3. Suggested order

1. **AB1 with E1**: unblocks the only failed gate (G3). Needs your acceptance of the SAM Licence.
2. **E6 calibration**: makes `--min-score` comparable between segmentation arms. Needs U2 consent.
3. **AB8 and AB9**: identity and latest-wins affect every map; they're cheap because they use
   existing data.
4. **AB3 (+ E5)**: the ainex rotation accuracy is the weakest validation result.
5. **AB4 and AB5**: mesh and texture quality are the most visible part of `view.sh`.
6. **AB2, AB6, AB7**: speed and robustness tuning once quality is settled.
7. E7, E9, E10, E11, E14 when there's a concrete need.
