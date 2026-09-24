# oh-my-slam

Monocular RGB mapping on a Mac (Apple silicon): single-image metric reconstruction, persistent
maps built from photos or video, and scene descriptions as labelled objects with oriented
bounding boxes (ASAM OpenLABEL 1.0.0). RGB only — depth, intrinsics and gravity come from models.

| Entry point | What it does |
|---|---|
| `./start_inference_server.sh` | Starts the resident model server (MoGe-2, GeoCalib, YOLOE-26x-seg, MapAnything). |
| `./reconstruct.sh -i IMG [-f json\|ply]` | One image → OpenLABEL scene (default) or coloured point cloud, camera frame, on stdout. |
| `./mapper.sh update -a IMGS\|FOLDERS\|VIDEO -m DIR -t full\|single [-f json\|ply] [-fps N]` | Creates or extends a persistent map. |
| `./segment.sh -i IMG [-o DIR] [-f json\|ply] [--min-score S] [--labels a,b]` / `-m MAP` | Objects, OBBs, colours; with `-o` also `segmented.png`, `catalog.csv/.md`, `segments.ply`. |
| `./view.sh -i IMG` / `-m MAP` | Local browser viewer (127.0.0.1, free port). `-m` needs no server. |

## Install

```sh
brew install colmap                 # COLMAP 4.2.x (features/matching CLI; GLOMAP inside)
uv sync                             # Python 3.12 environment in .venv (torch 2.14, pycolmap 4.2, …)
./scripts/install_tools.sh          # checks colmap 4.2.x
./start_inference_server.sh         # first start downloads ~2 GB of weights (MapAnything 4.9 GB
                                    # comes from the HF cache if present); later starts ~30 s
```

Model weights: MoGe-2 ViT-L normal (HF), GeoCalib (GitHub release, torch hub cache),
YOLOE-26x-seg + MobileCLIP2-B (Ultralytics, `~/Library/Caches/oh-my-slam/weights`),
MapAnything Apache-2.0 checkpoint (HF). See `THIRD_PARTY_LICENSES.md`.

## Usage

```sh
./start_inference_server.sh --status            # health JSON on stdout (exit 3 if not running)
./reconstruct.sh -i photo.jpg > scene.json
./reconstruct.sh -i photo.jpg -f ply > cloud.ply
./segment.sh -i photo.jpg -o out/ --labels chair,"dining table" --min-score 0.6
./mapper.sh update -a walk.mp4 -m maps/home -t full -fps 2 > map.json
./mapper.sh update -a more_photos/ -m maps/home -t single > new_part.json
./segment.sh -m maps/home -o out_map/
./view.sh -m maps/home
./start_inference_server.sh --stop
```

stdout carries exactly one JSON document or one binary PLY; progress and diagnostics go to stderr.
Exit codes: 0 ok, 1 internal error, 2 usage/input error, 3 inference server not running,
4 `-m` folder is neither empty nor a map, 5 nothing could be registered (e.g. no overlap),
6 another update holds the map lock.

### Conventions

* Camera frame: OpenCV axes (x right, y down, z forward), metres.
* Map frame: metres, z up (gravity-aligned), origin at the first keyframe's camera centre, x along
  that camera's forward direction projected on the floor.
* OBB: OpenLABEL 10-value cuboid `(x, y, z, qx, qy, qz, qw, sx, sy, sz)` (scalar-last quaternion);
  box axes x = width (longer horizontal side), y = depth, z = height (up). Floor-standing objects
  whose lower part is hidden are extended to the detected floor.
* Colours: a pure function of the object id (19-colour palette, then hue rotation); the same sRGB
  triple appears in the JSON, `segmented.png`, the catalogue, `segments.ply` and the viewer.
  Unsegmented points are #808080.
* Objects keep their id and colour for the life of the map; ids are never reused.

### Map folder

```
map.json          format version, map frame, scale, next ids, update history (written last)
frames/fNNNNNN.jpg, frames.json   keyframes: camera, K, T_map_cam, registration stats
per_frame/fNNNNNN/   depth.npy (float16, aligned metres), valid.png (latest wins),
                     instances.json (RLE masks, labels, scores, object ids), descriptor.npy
sfm/database.db, sfm/model/       COLMAP database and model (map coordinates)
cloud.ply, cloud_objects.npy      map cloud (TSDF-fused surface, latest colour wins) and object id per point
objects.json, objects/points_NNNNNN.npy, scene.json
```

Updates are staged in `.staging/` and committed atomically (map.json last); an interrupted update
leaves the previous map untouched.

## How it works

* **Server** (`src/oh_my_slam/server`): FastAPI on a Unix socket (`~/Library/Caches/oh-my-slam/srv.sock`,
  mode 0600); every model call runs on one GPU thread (MPS is not thread-safe), queue of 8 (503).
* **Single image**: EXIF focal (else MoGe's) → MoGe-2 metric depth → GeoCalib gravity refined by the
  floor plane → YOLOE instances → masks lifted to 3D → upright OBBs.
* **Mapping**: keyframes (sharpest frame per 1/fps slot for video) → per-keyframe depth, gravity,
  detections → COLMAP SIFT features + matching → GLOMAP (global) poses, incremental fallback,
  MapAnything multi-view poses for rotation-dominant input (chunked and anchored on posed views) →
  metric scale (MoGe vs SfM depth) and z-up map frame → per-keyframe depth alignment (sparse or
  dense) → latest-wins validity → object association (Hungarian on projected-mask IoU / 3D overlap),
  merging, removal on evidence of absence → map cloud (surface of a TSDF fusion); no mesh.
* **Ownership** (enforced by import-linter and `tests/unit/test_ownership.py`): reconstruction owns
  depth/intrinsics/gravity/clouds/fusion; segmentation owns instances, calibration,
  lifting, OBBs, colours, catalogue and artefacts; mapping owns inputs, SfM, map frame, identity and
  the store; the viewer only serves data. Shell scripts never call each other.

## Development

```sh
uv run pytest -m "not models and not browser and not eval" -q        # offline suite
uv run pytest --cov=oh_my_slam -m "not models and not browser and not eval"
uv run lint-imports && uv run ruff check . && uv run mypy src
OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m models                 # with the server running
uv run pytest -m browser                                              # Edge/Chrome via Playwright
uv run python -m oh_my_slam.tools.perf latency photo.jpg              # AC22 timings (one at a time)
uv run python -m oh_my_slam.tools.validate_inputs --inputs ~/robot_view   # user-input validation
uv run python -m oh_my_slam.tools.perf_report measure|renders|shots|write --out DIR  # R44 report
uv run python -m oh_my_slam.tools.cloud_quality --map maps/NAME     # cloud layering / accuracy metrics
```

Useful environment variables: `OH_MY_SLAM_DEVICE=cpu`, `OH_MY_SLAM_FEATURES=aliked`,
`OH_MY_SLAM_DEBUG=1` (tracebacks), `OH_MY_SLAM_LOG=DEBUG`,
`OH_MY_SLAM_TIMINGS=path.json` (per-stage timings of `mapper.sh update`, `reconstruct.sh` and
`segment.sh -i` written there as JSON; the one-line `timings:` summary always goes to stderr, and
each map update also keeps its record in `map.json → updates[].timings`).

## Troubleshooting

* `inference server is not running — start it with ./start_inference_server.sh` (exit 3): start it;
  `--status` shows which models loaded; the log is `~/Library/Caches/oh-my-slam/server.log`.
* Exit 5 on update: the new images do not overlap the map (no verified feature matches).
* Exit 6: another `mapper.sh update` is running on the same map.
* Evaluation/calibration datasets (TUM RGB-D, NYUv2, LVIS) are not downloaded automatically; put
  them in `~/oh-my-slam-data` and run `tools/evaluate.py` / `tools/calibrate_scores.py`.
