# Evaluation

Part of the [high-level specification](../high_level_spec.md) (§11).

## Command and outputs

`uv run python -m oh_my_slam.tools.evaluate [--episodes DIR] [--url URL] [--out DIR]
[--baseline PATH] [--set-baseline]` measures every row of the client contract.

* **Sequential.** It runs alone, one measurement at a time.
* **Service under test.** `--url` points it at a running service, which may be a LAN host.
  Latency is then measured as a LAN client sees it, uploads included. Without `--url` it starts a
  service locally on a free port.
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
| C8 | Ambiguity flagged when the second-best hypothesis is within 3 sigma | Measured on frames with true symmetric matches |
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
