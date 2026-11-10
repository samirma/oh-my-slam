# Evaluation

Part of the [high-level specification](../high_level_spec.md) (§11).

## Command and outputs

Two evaluators measure the client contract. Both are client-side tools under `tools/`, outside
the service's scope: they use only the public API, and neither the service nor the package
imports or depends on them.
* **Replay evaluator.** `tools/replay_eval/` is a standalone client of the public API: stdlib,
  numpy and requests only, no import from the service's code. It runs on the machine that holds
  the episodes and the simulator truth:
  `python tools/replay_eval/replay_eval.py --server URL --episodes DIR --house PATH
  --map-prefix NAME --out FILE.json`. Accuracy against simulator truth comes only from it.
* **Reference evaluator.** `tools/evaluate/`: `python tools/evaluate/evaluate.py [--url URL]
  [--out DIR] [--baseline PATH] [--set-baseline] [--requests DIR]` runs the reference captures,
  and optionally replays request sets recorded from earlier replays (`--requests`), for
  regression and latency without truth.

Every latency target is measured from a LAN client, uploads included, against the serving
machine (high_level_spec.md §10). Candidate implementations are each served on their own port
and run through the same replay evaluator.

* **Sequential.** It runs alone, one measurement at a time.
* **Service under test.** `--url` points the reference evaluator at a running service. Without
  it, it starts a service locally on a free port.
* **Outputs.** It writes `result.json` (machine-readable: each metric's value, target, pass/fail
  and the baseline comparison) and `summary.md` to `~/oh-my-slam-data/evaluations/<UTC>/`.
* **Data, not code.** Targets live in `examples/targets.json`, the single source of targets,
  keyed by the metric ids below; a stored baseline flags regressions. Both are data: changing
  them needs no code change. This document defines the metrics and does not repeat their
  thresholds, apart from the migration appendix that `targets.json` supersedes.

## Replay of recorded episodes

`--episodes DIR` names a folder of robot-guide episodes `e<NNN>/`. Each one holds:
* `truth.jsonl`, the simulator ground truth:
  * a first `meta` line: robot, scene, `floor_z`;
  * `robot` lines: base pose over time;
  * `objects` lines;
* `agent_files/`: the run folders with frames, `views.json` (each frame's 4×4 optical pose in
  the robot's start frame), and update and locate results;
* `oms/agent_requests.jsonl`, `meta.json`, `verdict.md`;
* `robot_control/<robot>/calibration.md`;
* for ProcTHOR scenes, the house JSON (rooms with `floorPolygon`, walls, doors). Simulator world
  (x, y) is THOR (x, z), with z up.

The replay sends the episode's frames to the API in the order the agent sent them, with the
camera from its calibration. It runs it twice:
* once with the `views.json` poses as `metric` priors, for the base and the arm;
* once with `trust: none`, as for camera-only robots.

Each episode is replayed as a fresh map. `views.json` poses are in the robot's start frame *of
that run*, and for a warm run that is not the map frame of an earlier run.

It then compares the API's answers with the truth, after the single similarity transform from
the map frame to the simulator world that the map's first update fixed. Scale is judged on that
transform, not re-fitted per read.

## Metrics

Each metric's threshold is the entry of `examples/targets.json` under its id. Parameters that
are part of a definition (a distance, an angle, a cell size) stay here.

| Id | Contract | Metric |
| --- | --- | --- |
| `locate.recall` | C8 | Share of frames within 0.3 m and 15° of a keyframe that are located |
| `locate.median_pos_error` | C8 | Median position error of located frames |
| `locate.ambiguity_flagged` | C8 | Share of located frames whose pose error exceeds 0.10 m or 5° that carry a non-null `ambiguity` (a supported rival beyond 0.10 m or 5° is never silent, api.md §6) |
| `locate.nearest_view` | C8 | Share of rejected frames with `no_overlap` and a prior that carry a `nearest_view` whose turn points toward mapped structure |
| `locate.sigma_containment` | C8 | Share of located frames whose position error ≤ 3 × `sigma_pos_m` and rotation error ≤ 3 × `sigma_rot_deg` |
| `locate.sigma_tightness` | C8 | Information, no target: median over located frames of error / sigma (position and rotation reported separately); near 1 means sigmas neither over- nor under-confident |
| `update.sigma_containment` | C1, C4 | Share of non-rejected update frames (registered and `kept_at_prior`, the latter against their prior's sigmas) whose committed-pose position error ≤ 3 × its position sigma and rotation error ≤ 3 × its rotation sigma |
| `scale.error_metric` | C2 | Scale error with `metric` priors |
| `scale.error_camera_height` | C2 | Scale error with camera height only |
| `frame.shift` | C1 | Map-frame shift of stored keyframes between consecutive versions, median position and rotation, beyond the stated `gauge_shift` |
| `update.readable_latency` | C3 | Upload to frame readable in a version (`provisional_version` committed) |
| `fusion.time_per_frame` | C3 | Fusion time per new frame, maps up to 500 keyframes |
| `reads.atomic` | C3 | Share of reads during an update answered from one version |
| `jobs.eta_accuracy` | C3 | Share of jobs whose `eta_s` when queued, and dry runs whose `predicted_s`, fall within the tolerance of the measured time |
| `fusion.intermediate_interval` | C3 | Longest interval between intermediate versions while fusion lasts longer than it |
| `inputs.reason_agreement` | C4 | Share of rejected inputs whose `reason` agrees with the truth (bare wall, too close, no overlap) |
| `revert.latency` | C5 | Revert latency |
| `gate.warping_rejected` | C6 | Share of the warping updates listed in `targets.json` rejected by `accept_if` at the thresholds `targets.json` gives (`gate.accept_if`) |
| `gate.others_accepted` | C6 | Share of the other updates of the passing episodes accepted at those thresholds |
| `grid.floor_plan_agreement` | C9 | Share of observed cells agreeing with the floor plan (free inside rooms, occupied at walls) |
| `passages.door_recall` | C9 | Share of the floor plan's doors found as passages within 0.3 m |
| `passages.geometry` | C9 | Share of found doors whose `centre_m` is within the tolerance of the floor plan's door-gap centre and whose `width_m` is within the tolerance of its free width |
| `grid.floor_observed` | C9 | Share of free cells with `floor_observed` where the floor plan has floor |
| `grid.camera_cells_free` | C9 | Share of keyframe camera cells, and their near blind ring, that are `free` |
| `features.region_set` | C9 | Share of frontiers bordering a region, and of objects in or on a region, with `region` set |
| `grid.band_occupied` | C9 | Share of cells holding an obstacle between the band's `z_min` and the requested `z_max` marked `occupied` |
| `grid.floor_height` | C9 | Floor height error, against the surface the robot stands on (map.md §6): the large horizontal plane nearest the priors' origin height, else the lowest large horizontal plane below the cameras |
| `reads.latency` | C9, C10 | `GET grid` (5 cm cells) and `GET objects` on a committed version, house-sized map, each |
| `objects.support_height` | C10 | Support height error of objects on a surface (`supported_by` surface `height_m`) |
| `objects.relations` | C10 | Share of `supported_by` / `inside_of` agreeing with the truth's relations |
| `objects.centre_error`, `objects.label_agreement`, `objects.id_stability` | C10 | Object centre error, label agreement and id stability across updates |
| `analysis.range_scan_error` | C11 | Range-scan error against the floor-plan walls within the band, relative to range |
| `interactive.latency_under_load` | C12 | Locate and analyse latency while an update runs: ratio to idle latency, and absolute |
| `locate.latency_idle` | C12 | Locate while idle, per frame of the request |
| `update.later_wins` | C13 | `office_sequence/`: the cup is `struck`, and the other objects keep their ids and boxes |

## Reference captures

The reference captures in `examples/` measure the camera-only path: no priors, one camera with
EXIF or the calibration in `README.md`.
* `office_sequence/`: map update semantics.
* `ainex-captures/`: turning in place, headings from file names (README "The ainex-captures file
  names").
* `camera/`: wide-angle lens, turning in place.

Each runs in one job and split into several. The split runs include one frame per job and an
interleaved split.

## Contracts

The evaluator checks these too; the target for each is zero violations:
* schema validity of every response against `api.schema.json`;
* reads atomic per version;
* a job continuing after its client disconnects;
* `Server-Timing` present;
* bird's-eye `pixel_to_map` agreeing with the objects' footprints.

## Appendix: current thresholds

The migration source for `examples/targets.json`, which supersedes this table once it holds
these ids.

| Id | Threshold |
| --- | --- |
| `locate.recall` | ≥ 0.9 |
| `locate.median_pos_error` | ≤ max(2 cm, 1 % of the scene depth) |
| `locate.ambiguity_flagged` | ≥ 0.8 |
| `locate.nearest_view` | ≥ 0.8 |
| `locate.sigma_containment` | ≥ 0.99 |
| `locate.sigma_tightness` | information only |
| `update.sigma_containment` | ≥ 0.99 |
| `scale.error_metric` | ≤ 3 % |
| `scale.error_camera_height` | ≤ 10 % |
| `frame.shift` | ≤ 2 cm and ≤ 0.5° |
| `update.readable_latency` | ≤ 2 s |
| `fusion.time_per_frame` | ≤ 3 s |
| `reads.atomic` | 100 % |
| `jobs.eta_accuracy` | ≥ 0.8, tolerance ±30 % |
| `fusion.intermediate_interval` | ≤ 10 s |
| `inputs.reason_agreement` | ≥ 0.8 |
| `revert.latency` | ≤ 2 s |
| `gate.accept_if` | `{max_residual_m: 0.10, max_gauge_shift_m: 0.05, max_rejected_share: 0.5}` |
| `gate.warping_rejected` | All (100 %), on the warping updates of episodes e034 and e040 |
| `gate.others_accepted` | ≥ 0.9 |
| `grid.floor_plan_agreement` | ≥ 0.9 |
| `passages.door_recall` | ≥ 0.8 |
| `passages.geometry` | ≥ 0.8, tolerances 0.10 m (centre) and 0.10 m (width) |
| `grid.floor_observed` | ≥ 0.9 |
| `grid.camera_cells_free` | 100 % |
| `features.region_set` | 100 % |
| `grid.band_occupied` | ≥ 0.95 |
| `grid.floor_height` | ≤ 2 cm |
| `reads.latency` | ≤ 0.5 s |
| `objects.support_height` | ≤ 2 cm |
| `objects.relations` | ≥ 0.9 |
| `objects.centre_error`, `objects.label_agreement`, `objects.id_stability` | The round-4 values already in `targets.json` |
| `analysis.range_scan_error` | ≤ 5 % of range |
| `interactive.latency_under_load` | ≤ 2 × idle latency, and ≤ 3 s |
| `locate.latency_idle` | ≤ 2 s |
| `update.later_wins` | As before (unchanged ids and boxes; cup `struck`) |
