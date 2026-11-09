# Evaluation

Part of the [high-level specification](../high_level_spec.md) (§11).

## Command and outputs

Two evaluators measure the client contract:
* **Replay evaluator.** `tools/replay_eval/` is a standalone client of the public API: stdlib,
  numpy and requests only, no import from the service's code. It runs on the machine that holds
  the episodes and the simulator truth:
  `python tools/replay_eval/replay_eval.py --server URL --episodes DIR --house PATH
  --map-prefix NAME --out FILE.json`. Accuracy against simulator truth comes only from it.
* **Reference evaluator.** `uv run python -m oh_my_slam.tools.evaluate [--url URL] [--out DIR]
  [--baseline PATH] [--set-baseline] [--requests DIR]` runs the reference captures, and
  optionally replays request sets recorded from earlier replays (`--requests`), for
  regression and latency without truth.

Every latency target is measured from a LAN client, uploads included, against the serving
machine (high_level_spec.md §10). Candidate implementations are each served on their own port
and run through the same replay evaluator.

* **Sequential.** It runs alone, one measurement at a time.
* **Service under test.** `--url` points the reference evaluator at a running service. Without
  it, it starts a service locally on a free port.
* **Outputs.** It writes `result.json` (machine-readable: each metric's value, target, pass/fail
  and the baseline comparison) and `summary.md` to `~/oh-my-slam-data/evaluations/<UTC>/`.
* **Data, not code.** Targets live in `examples/targets.json`, and a stored baseline flags
  regressions. Both are data: changing them needs no code change.

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

## Targets

| Contract | Metric | Target |
| --- | --- | --- |
| C8 | Frames within 0.3 m and 15° of a keyframe that are located | ≥ 0.9 |
| C8 | Median position error of located frames | ≤ max(2 cm, 1 % of the scene depth) |
| C8 | `ambiguity` non-null on located frames whose pose error exceeds 0.3 m or 15° | ≥ 0.8 |
| C8 | `ambiguity` null on located frames within 0.10 m and 5° of the truth | ≥ 0.9 |
| C2 | Scale error with `metric` priors | ≤ 3 % |
| C2 | Scale error with camera height only | ≤ 10 % |
| C1 | Map-frame shift of stored keyframes between consecutive versions | ≤ 2 cm and ≤ 0.5° (median), beyond the stated `gauge_shift` |
| C3 | Upload to frame readable in a version (`provisional_version` committed) | ≤ 2 s |
| C3 | Fusion time per new frame, maps up to 500 keyframes | ≤ 3 s |
| C3 | Reads during an update answered from one version | 100 % |
| C4 | Rejected inputs whose `reason` agrees with the truth (bare wall, too close, no overlap) | ≥ 0.8 |
| C5 | Revert latency | ≤ 2 s |
| C6 | Episodes' warping updates (e034, e040) rejected by `accept_if` at the default thresholds | All |
| C9 | Grid cells agreeing with the floor plan (free inside rooms, occupied at walls), among observed cells | ≥ 0.9 |
| C9 | Doors of the floor plan found as passages within 0.3 m | ≥ 0.8 |
| C9 | Passage `centre_m` within 0.10 m of the floor plan's door-gap centre, and `width_m` within 0.10 m of its free width | ≥ 0.8 of found doors |
| C9 | Free cells with `floor_observed` where the floor plan has floor | ≥ 0.9 |
| C10 | Support height of objects on a surface (`supported_by` surface `height_m`) | ≤ 2 cm |
| C10 | `supported_by` / `inside_of` agreeing with the truth's relations | ≥ 0.9 |
| C8 | Rejected frames with `no_overlap` that carry a `nearest_view` whose turn points toward mapped structure | ≥ 0.8 |
| C3 | `eta_s` of a queued job, and `predicted_s` of a dry run, within ±30 % of the measured time | ≥ 0.8 of jobs |
| C9 | Floor height error | ≤ 2 cm |
| C10 | Object centre error, label agreement and id stability across updates | As in `targets.json`, from the round-4 values |
| C11 | Range-scan error against the floor-plan walls within the band | ≤ 5 % of range |
| C12 | Locate and analyse latency while an update runs | ≤ 2 × idle latency, and ≤ 3 s |
| C12 | Locate of one frame while idle | ≤ 2 s |
| C9, C10 | `GET grid` (5 cm cells) and `GET objects` on a committed version, house-sized map | ≤ 0.5 s each |
| C3 | Intermediate version during fusion | At least every 10 s while fusion lasts longer |
| C13 | `office_sequence/`: the cup is `struck`, and the other objects keep their ids and boxes | As before |

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
