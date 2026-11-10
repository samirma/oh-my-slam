# Replay evaluator

`replay_eval.py` is the standalone replay evaluator of `specs/evaluation.md`. It replays recorded
robot-guide episodes against a running oh-my-slam service through the public HTTP API
(`specs/api.md`), then compares the answers with the simulator truth. The comparison uses the one
similarity transform from the map frame to the simulator world that the map's first update fixed.

It runs on the machine that holds the episodes, which is not the serving machine. It needs:

* Python 3.11 or later, with `numpy` and `requests`;
* optionally `jsonschema`, for full schema validation.

It imports nothing from `oh_my_slam`. Its only file outside this folder is
`specs/api.schema.json`, which it finds at `../../specs/api.schema.json` by default. If you copy
this folder alone, pass `--schema PATH`.

## Usage

```sh
# 1. check that the episode keys match the layout (no server needed); send the output back if not
python3 tools/replay_eval/replay_eval.py --episodes ~/robot_guide_skill.state/experiments \
    --episode e041 e017 e044 --dump-layout > layout_dump.json

# 2. replay and measure
python3 tools/replay_eval/replay_eval.py --server http://192.168.0.106:9091 \
    --episodes ~/robot_guide_skill.state/experiments --house PATH/TO/house.json \
    --map-prefix rg- --out ~/replay_eval/result-$(date -u +%Y%m%dT%H%M%SZ).json \
    [--episode e041 ...] [--priors metric|none|both]
```

| Option | Meaning |
| --- | --- |
| `--server URL` | The service under test. |
| `--episodes DIR` | A folder of `e<NNN>/` episodes. `--episode` (repeatable) picks some of them. |
| `--house PATH` | A ProcTHOR house JSON, or a folder of them named after the episodes' scenes (`<scene>.json`, `<scene with : as _>.json` or `<last token of scene>.json`). Without it, the floor-plan metrics are `not_measurable`. Scenes starting with `ithor` (the arm) never use it: their objects come from `truth.jsonl`. |
| `--map-prefix NAME` | Each replay is a fresh map `<prefix><episode>-<mode>`, deleted at the end unless `--keep` is given. A map of that name is deleted first. |
| `--priors` | `both` (the default) gives each robot its default replays: the base gets `metric` and `none`, the humanoid gets `none`, and the arm gets `metric`. `metric` or `none` forces that one replay for every robot. |
| `--out FILE.json` | The result. A summary table also goes to stdout. |
| `--dump-layout` | Prints what each episode folder holds, key by key, plus the result of loading it, then exits. |
| `--layout FILE.json` | Overrides any key or convention of the layout table (`rpl_layout.py`, deep-merged), with no code change. |
| `--targets FILE.json` | The targets file, by default `examples/targets.json` (`rpl_targets.py`): `{"metrics": {metric id: threshold}}`, keyed by the metric ids of `specs/evaluation.md`. |
| `--record-requests DIR` | Also writes every request as a request set (`DIR/<episode>-<mode>/requests.jsonl` and `files/`) that `tools/evaluate/evaluate.py --requests DIR` replays. |
| `--accept-if JSON` | The `accept_if` sent with every update when O9 is served. The default is the gate parameters of `specs/evaluation.md` ("Gate parameters"): `{"max_residual_m": 0.10, "max_gauge_shift_m": 0.05, "max_rejected_share": 0.5}`. They are measurement parameters, not targets. `{}` checks nothing. |
| `--passing e041,...` | The passing episodes whose clean updates the gate must accept (C6). Without it, `meta.json`'s `verdict`, or else `verdict.md`, decides (see below). |
| `--base-priors drifting\|metric` | The trust of the wheeled base's odometry priors in the prior replay. The default is `drifting`. |
| `--no-height` | No `height_above_floor_m` in the `none` replay; its scale is then reported as `diag.scale_error_monocular`. |
| `--no-probes` | No reads while a job runs. This drops C3 one-version reads, the busy C12 latencies and the C3 fusing cadence. |
| `--no-jsonschema` | Minimal required-keys validation, even if `jsonschema` is installed. |
| `--wait-s`, `--probe-interval-s`, `--job-timeout-s`, `--timeout-s`, `--retries`, `--retry-wait-s`, `--range-scans`, `--align-min-spread-m` | Polling, timeouts, retries of retryable errors, frame-analysis samples and the alignment spread (see below). |

The exit status is 0 when at least one replay was measured, and 1 otherwise.

## What a replay does

1. `GET /api/health` reads `version`, `capabilities` and `timings`. Metrics whose group is not
   in `capabilities` are `not_measurable`, and each one names the missing group.
2. Every frame set that has a `views.json` is replayed, in time order: the humanoid's `stamp`,
   else the file-name time, else the set folder order (a set without a time keeps its place after
   the set before it). `oms/agent_requests.jsonl` is not used: it is a log of the agent's shell
   commands, and most updates never appear in it.
   * Each set is one update job: a multipart request with a `request` JSON part plus one file
     part per frame, and the camera from the intrinsics table.
   * Every fifth frame of a set (`replay.holdout_every`) is held out of its update and located
     right after it (`POST locate`, O5).
   * Sets whose name starts with `loc` or `look` (`replay.locate_prefixes`) are located whole,
     never mapped: they are turn-in-place frames.
   * Prior replay (mode `metric`): each frame carries its `views.json` pose as a prior.
     * Base: the odometry over-counts by 5–10 %, so its priors are `drifting` by default
       (`--base-priors metric` for `metric`). Either way, `sigma_pos_m` is the larger of
       0.02 m and 10 % of the odometry distance travelled since the run's first frame. This is
       an O1 test of biased priors, reported as `diag.prior_residual_m` / `_deg`.
     * Arm: `metric`, with sigmas 0.02 m and 1°.

     A later run's poses are moved into the first run's start frame through the truth start
     poses, because a map has one frame.
   * `none` replay: `trust: none`. Floor robots also send `height_above_floor_m` = `pose[2][3]`
     of the `views.json` pose, with sigma 0.02 m.
   * With O10, a dry run (`?dry_run=1`) precedes each update, for `predicted_s`. With O9,
     every update carries `accept_if` (the skill's thresholds, or `--accept-if`).
3. Each job is polled with `GET /api/jobs/{id}?wait_s=` until it ends. Meanwhile a second thread
   reads `GET frames` every `--probe-interval-s`, and also runs a one-frame locate (O5) and
   analyse (O7) on a keyframe of the previous version. After the job, every version a probe saw
   is read again with `?version=n`, and must have the same content. The provisional and result
   versions' frames are read, as are the result version's objects (O8). One idle one-frame
   locate and analyse follow.
4. At the end come the map summary, the grid (`cell_m=0.05&z_min=0.05&z_max=1.8`, O6), the
   objects (O8), range scans on `--range-scans` keyframes with their `T_map_cam` (O7), the
   versions list, and a revert to `latest - 1` (O2). The map is then deleted.
   * If nothing was located during the replay, every sent frame that is not a keyframe is
     located, in batches of 8.

Every response is validated against `specs/api.schema.json`, as is every error body
(`ErrorBody`). A missing `Server-Timing` header is counted too.

## Skill-v2 episodes (picked automatically)

An episode is read as v2 when `agent_files/runs/<run>/frames/*.json` sidecars exist and no
`<run>/<set>/views.json` does (`rpl_episode_v2.py`). The old reader stays for the old layout.

* **Frames.** `frames/<name>.png`, `<name>.depth.png` (16-bit, mm) and a `<name>.json` sidecar
  `{name, image, depth, camera_id, width_px, height_px, stamp, ros_stamp, state_ros_stamp,
  T_body_cam, T_body_cam_source, height_above_floor_m, height_sigma_m, pose_source, pose_state,
  prior {T_map_cam, sigma_pos_m, sigma_rot_deg, trust}}`.
  * Any field may be null or missing, except `stamp` and the image file. Files starting with
    `_` (`_select.json`) are not frames.
  * Errors name the file and the key.
* **Plan.**
  * **Updates:** every job of `oms/jobs.json` (`{"jobs": [{job_id, frames, accept_if, state,
    submitted_at, measured}]}`, or a bare list), in submission order. Each carries its frames,
    their sidecar priors and depth images, their heights, and the job's own `accept_if`. The
    run's `oms/jobs.json` is read first, else the episode's.
  * **Locates:** the `loc*` frames that are in no job, located at their stamps. Frames less than
    10 s apart are grouped into one request, which carries their sidecar priors.
  * Other frames (`close*`, `probe*` outside jobs, …) were not sent, and are not sent here.
  * Each job's `measured.s` is reported as `recorded_s` next to the replayed time.
* **Intrinsics, per run and camera:**
  1. `calibrate_camera.json` → `intrinsics {fx_px, fy_px, cx_px, cy_px, width_px, height_px}`,
     when its size matches the frames;
  2. `camera_info_<camera_id>.json` → ROS CameraInfo `msg` (`K`, `width`, `height`; `D` with
     `plumb_bob` or `rational_polynomial` becomes OpenCV distortion when non-zero);
  3. the episode's `robot_control/<robot>/calibration.json` (URDF robot names, e.g.
     `yahboomcar_x3plus`, `re_bot_arm_rs`) whose values hold `camera.<sidecar camera_id>.*`.
     * The harness layout is parsed first: `{"robot_id", "values": {"camera.<id>.fx_px":
       {"value", "unit", "source", "run_id", "date", "confirmed"}, …}}`, with `fx_px`,
       `fy_px`, `cx_px`, `cy_px`, `width_px` and `height_px`.
     * Failing that, a tolerant search is tried: the id as a key or a `camera_id`/`id`/`name`
       value; `fx`… keys or a `K`/`camera_matrix` 3×3; the size from `width`/`height` or
       `resolution`.
     * A calibration of another size is skipped. One without a size is taken at the frame's
       size.

     `--dump-layout` prints each `calibration.json`'s key tree (keys and types only).
  4. `--camera '{"fx": …, "fy": …, "cx": …, "cy": …, "width": …, "height": …}'`.

  The error lists what was tried. A camera id calibrated differently in a later run is sent
  as `<id>@<run>`.
* **Truth.** The truth `cams.<camera>` position at the sidecar `stamp` (no matching needed). The
  orientation comes from the robot's link, as in v1. The run's start frame (the priors' map
  frame) is fitted by a gravity-aligned similarity of the prior positions to the truth
  positions; its scale is reported as `odometry_scale`. When it cannot be fitted, the start
  frame is the arm's `base_link` or the base pose at the first stamp.
* **Replay.**
  * The prior replay (`metric`) sends each frame's sidecar prior as recorded (trust and
    sigmas). The `none` replay sends `trust: none`.
  * Both send the sidecar heights (`--no-height` to drop them) and the depth images with
    updates (`--no-depth` to drop them).
* **`--dump-layout`** prints, per run: sidecar counts by name prefix, sidecar keys, prior trusts,
  depth count, camera ids and the intrinsics source, job count and keys, `oms/` file names, and
  the line count and keys of `log.jsonl`. Never values.

## Assumptions about the episode keys (to confirm with `--dump-layout`)

They follow the conductor's layout (`~/oh-my-slam-data/robot-guide/episode_layout.md`). Each is a
key of `LAYOUT` in `rpl_layout.py`.

* **Frames.** `agent_files/runs/<run>/<set>/*.jpg`, with one `views.json` per set. File names are
  not assumed to follow `<set>-date-time-NN`: the first `YYYYMMDD-HHMMSS` found anywhere in the
  name is a UTC time (the agents ran in Docker on UTC; `frame_time.utc_offset_h` is 0), and a name without one has no time. Frames of a
  set are ordered by `stamp`, then by file name.
* **`views.json`.** A dict keyed by file name. The base's value is `{"pose", "base": [x, y,
  yaw]}`, with `base` the odometry in the start frame. The humanoid's is `{"pose", "pan", "tilt",
  "pitch", "roll", "imu_yaw", "stamp"}`. The arm's is the bare 4x4. The pose is the full optical
  pose in the run's start frame.
* **`truth.jsonl`** (keys confirmed by the conductor).
  * `meta`: `{cameras, kind, objects, placement, robot, robot_row, t, t0}`, plus the
    simulator's hello block:
    * at `meta.sim.hello` (the arm episodes), or else at `meta.hello` (`truth.meta_hello` lists
      the paths, in order);
    * it holds `{engine, scene, floor_z, worktop: {name, z, area_m2, centroid, bounds},
      starting, rtf}`.

    `scene`, `floor_z` and `worktop.z` are all read from that block. `robot` is an id; the kind
    comes from the robots table. `placement` is kept: `"worktop"` puts the robot on the worktop.
  * `robot` lines: `{bodies, cams, kind, pos, quat, sim_time, t}`, plus `joints` on the arm.
    `t` is epoch s, `quat` is `[w, x, y, z]`, and `cams.<camera>` is the optical centre in the
    world.
  * `objects` lines: `bodies.<name>.{pos, quat}`.
* **Robots.** The table maps `meta.robot` to a kind, a camera name and intrinsics:
  * `ainex`: humanoid, `camera`, 640×480, f 270, c (319.5, 239.5);
  * `rebot_b601_rs`: arm, `camera_color_optical_frame`, 640×360, f 337.2, c (319.5, 179.5);
  * `rosmaster_x3_plus`: base, `camera_color_optical_frame`, 640×480, f 655.8, c (319.5,
    239.5). The EXIF agrees.

  A frame whose size is not the table's takes its EXIF (FocalLength × FocalPlaneX/YResolution,
  with mm as the unit when the tag is absent, and c at the image centre). Without EXIF, it takes
  the table scaled to its size, with a warning. `calibration.md` is not read.
* **Truth pose of a frame.**
  * **Start frame to world.** It depends on the robot:
    * **Base:** a 2-D similarity fit (yaw, translation and scale) of the views `base` odometry
      [x, y] to the truth base track at the frames' file-name times. The scale is reported as
      `odometry_scale`. The truth base yaw at odometry yaw 0 is fitted too: it is used when
      matching, so the body's quaternion need not point along its heading. When the frames
      span less than 0.3 m, the fallback is the truth base pose at the run's first frame (its
      yaw and its (x, y) at `floor_z`).
    * **Arm:** truth `bodies.base_link.pos`, with identity rotation (`start_link`), at the run's
      first frame. The robot line's `pos` is another body. Only when `base_link` is missing
      does the fallback refine the line's pose: first by the translation that best lays the
      views positions on the truth camera samples, then by ICP until it settles.
    * **Humanoid:** the truth base pose at the run's first frame (its yaw and its (x, y) at
      `floor_z`).
  * **Time.** The frame's time is found as follows:
    * **Humanoid:** its `stamp`.
    * **Base:** the time in its window at which the truth base track, taken relative to the
      start pose, best matches the views `base` odometry rescaled by `odometry_scale` (cost:
      hypot(position error, 0.5 m × yaw error in rad)).
    * **Arm:** the truth `cams` sample nearest the views position, in that window.

    The window is ±3 s (`frame_window_s`) around the frame's own file-name time, when its set
    names each frame with its own time (the base). Otherwise it is the set's window: from the
    set before it to the set after it, with 2 s of slack. File-name times are UTC, as is the
    epoch truth `t`. A frame with no
    time keeps the views pose composed with the start pose (`truth_source: views`), with a
    warning.
  * **Position.** `cams.<camera>` at that time: `camera` (humanoid), or
    `camera_color_optical_frame` (base and arm).
  * **Orientation.** From the link quaternion at the matched sample: `head_tilt_link`
    (humanoid), `camera_link` (base) or `gripper_end` (arm).
    * **Humanoid and base:** their links are x forward, y left, z up, so optical x = −link y,
      optical y = −link z and optical z = link x.
    * **Arm:** its wrist camera sits at a fixed, unknown mount rotation from `gripper_end`
      (`mount_fit`). That rotation is estimated once per episode as the robust mean of
      gripper_end⁻¹ · views rotation (through the `base_link` start pose), over all frames,
      then applied to every sample. The dump reports the spread of the per-frame estimates as
      `mount_rot_spread_deg`; it should be small. The arm's truth position is the simulator's
      `cams.camera_color_optical_frame`, so `views_vs_truth_m` for the arm should be a few
      millimetres, not exactly 0.
    * A kind without a link (`rotation_body`), or a frame whose sample lacks it, keeps the views
      rotation composed with the start pose.
  * **Check.** `--dump-layout` prints `views_vs_truth_m: {pos_m, rot_deg}`: the median
    difference between the views poses (via the start pose) and the truth poses. For the
    humanoid, it shows the odometry drift.
* **Pass or fail.** `meta.json` decides when it has `verdict`, as a bool, or as text read with
  the words below. Otherwise `verdict.md` is read as prose: the episode passed when the earliest
  of the pass words (`pass`, `success`, `succeeded`) or fail words (`fail`, `abort`, `timeout`,
  `timed out`, `error`) is a pass word. When neither file decides, the episode is not judged.
  `--passing` overrides all of this.
* **`--dump-layout`.** It prints only keys, counts and parsed values: never raw lines of
  `agent_requests.jsonl` (only its line count and keys) or of any other file.
* **House JSON.** Simulator world (x, y, z) = THOR (x, z, y).
  * Rooms: `floorPolygon`. Walls: `polygon`, or ids `wall|room|x0|z0|x1|z1`.
  * Doors: the `holePolygon` x range along `wall0` from its first point.
  * Objects: `position`, with `children` resting on or (for container types) inside their
    parent. Their label is the id before the first `|`.

## Metrics (`rpl_metrics.py`)

The metric ids are those of `specs/evaluation.md`; their thresholds are the entries of
`examples/targets.json` (none is in the code). Each metric is a list of samples, reduced to a
share, a median, a p90, a maximum or a count, and compared with its target; a compound metric
(`frame.shift`, `interactive.latency_under_load`, `locate.sigma_tightness`) has one value per
part and passes when every part does. The overall value pools the samples of all replays.
Statuses are `pass`, `fail`, `info` (no target), `no_samples`, `no_target` and
`not_measurable` (with `missing`). `diag.*` entries are diagnostics (information only), and
`contract.*` entries are contracts (zero violations).

| Group | Metrics |
| --- | --- |
| Always | `update.readable_latency` (upload → `provisional_version`/`result_version` seen, p90); `fusion.time_per_frame` (p90); `reads.atomic`; `inputs.reason_agreement`; `update.sigma_containment` (non-rejected inputs, `kept_at_prior` included with its prior's sigmas, within 3 sigmas of the truth after the alignment); `scale.error_metric` (metric replay), `scale.error_camera_height` (none replay with heights); `frame.shift` (`pos_m`, `rot_deg`: medians beyond `quality.gauge_shift`, fused keyframes only); `contract.inputs`, `contract.schema_valid`, `contract.server_timing`; diagnostics `diag.scale_error_monocular`, `diag.prior_residual_m` / `_deg` |
| O2 | `revert.latency`; `contract.revert` |
| O3 | `jobs.eta_accuracy` (job `eta_s`, and dry-run `predicted_s` when O10 is served, within the targets' `time_tolerance_rel`; also measured with `O3-jobs`); `fusion.intermediate_interval` (when fusion lasts longer than 10 s) |
| Always (look sweeps) | `diag.sweep_yaw_error_deg` (each look-sweep keyframe's map yaw after the gravity-aligned fit against truth, median) and `diag.sweep_yaw_span_ratio` (map over truth yaw span of each sweep, median; > 1 over-counts). A sweep is a run of consecutive `look*` frames, in time order, from one stand (within 0.25 m of its first; `sweep` in the layout). `info.sweeps` lists each sweep's median, max, spans and ratio. |
| O5 | `locate.recall`; `locate.median_pos_error` (median of error / max(`abs_m`, `scene_depth_share` × the scene depth), both from the targets); `locate.ambiguity_flagged` (located frames off by more than 0.10 m or 5° carrying an `ambiguity`); `locate.sigma_right_median` (world metres, frames within 0.05 m with `rivals` empty); `locate.sigma_containment`; `locate.sigma_tightness` (`pos`, `rot`); `locate.nearest_view` (only `no_overlap` frames sent with a prior; a null turn counts as a miss); `locate.latency_idle` (p90, per frame of each idle request); `interactive.latency_under_load` (`ratio` of busy over idle medians, `abs_s` p90 of busy locates and analyses); diagnostics `diag.locate_pos_error_m`, `diag.ambiguity_null_on_right`, `diag.ambiguity_rule`, `diag.locate_first_s` |
| O6 | `grid.camera_cells_free`, with `diag.camera_stand_free` (the camera cells alone). Both index the grid by `origin_m` (the corner of cell [0, 0]), rows = y and cols = x, with the cameras' map-frame poses from `GET frames` of the grid's version; `features.region_set`; `contract.band_as_asked`; `grid.floor_height` (against the lowest large support plane in view: the hello block's `floor_z`, or its `worktop.z` when `meta.placement` is `worktop` or the robot's kind is arm); `grid.id_stability` (a grid is read at each committed version: a region of the same area within 2 % and centroid within one cell, or a passage of the same centre and width within one cell, keeps its id); `reads.latency` (with `GET objects`). With a floor plan: `grid.floor_plan_agreement`; `passages.door_recall`; `passages.geometry` (the targets' centre and width tolerances); `grid.floor_observed` (cells with `floor_observed` 2, free by rule, are left out); `grid.band_occupied`; `grid.floor_false_obstacles` (a count: occupied cells, and present floor-standing objects lower than 0.06 m, on observed floor-plan floor more than one cell from every truth object, inside some keyframe camera's horizontal field of view — the band's frustum projected on the floor) |
| O7 | `interactive.latency_under_load` (analyses); with a floor plan, `analysis.range_scan_error` (median) |
| O8 | `reads.latency` (with `GET grid`); `objects.centre_error`; `objects.label_agreement`; `objects.id_stability`; `objects.support_height`; `objects.relations` |
| O9 | `gate.warping_rejected` and `gate.others_accepted`, per update at the gate parameters. A warping update is one whose priors slip by at least 0.3 m against the truth after its first frame (its first prior put on its truth pose, the largest offset of the later ones); a clean update is one whose priors all stay within their position sigma, or that has none. Clean updates of the passing episodes must be accepted; updates neither warping nor clean are left out. The slip of the images' geometry is not judged: it needs the service's answers. |

  * **Per-object section** (`runs.<mode>.info.objects`, no images or paths).
    * `truth[]`, one row per truth object:
      * its name, label, canonical label, world centre and size (when the truth has one);
      * the matched map object: the same label within 1 m, else the nearest within 1 m. Its
        id, label, canonical label, `detected_as`, world centre after alignment, size,
        `mean_rgb`, state, `supported_by` and `inside_of`;
      * `centre_error_m`, `label_agreed` and `support_height_error_m`.
    * `unmatched_map[]`: the map objects matched to no truth object, with label and centre.

    So a low `objects.label_agreement` can be split into matching (a label that should have agreed)
    and real misdetection. The support height is judged against the worktop (arm) or the floor
  (ProcTHOR objects without a parent). The relation is judged against the house's
  `children` nesting.
* **C11.** Beams of kind `obstacle` are compared with a ray cast against the floor-plan walls.
  `bearing_deg` is taken as a map-frame yaw from the camera position, as the call carries
  `T_map_cam`.

The C10 targets are the round-4 values in `examples/targets.json` (`objects.centre_error`
≤ 0.10 m, `objects.label_agreement` ≥ 0.95, `objects.id_stability` ≥ 0.8).

These items are not measured by the replay:
* C13 and the split runs: the reference evaluator measures them;
* the reference captures;
* the contracts "job continues after a client disconnect" and "bird's-eye `pixel_to_map`".

## Output

`--out FILE.json` holds the following:
* `health.start` and `health.end`: the server's `/api/health`, with its version, capabilities
  and timings;
* `client`: host, Python, whether `jsonschema` was used, and the total time;
* `episodes.<id>`:
  * robot, kind, scene, frames, events and warnings;
  * per replay mode, `runs.<mode>`: `metrics` (each with value, target, status, n and detail),
    `info` (the alignment and the locate tolerance), `timings_s`, `calls` (per route: n,
    median, p90, errors), `jobs` (state, versions, readable and total seconds, rejected inputs,
    quality) and `warnings`;
* `overall`: every metric over all replays.

## Sending results back

Send the `--out` JSON to the `slam` session, along with the `--dump-layout` output when a key
did not match. You can also copy it to the serving machine:

```sh
scp ~/replay_eval/result-*.json samir@192.168.0.106:~/oh-my-slam-data/evaluations/replay/
```

## Tests

The tests run offline against synthetic episodes: two box rooms joined by a door, with header-only
JPEG and PNG frames. A fake service on a free loopback port gives schema-valid answers derived
from the truth through a known similarity. Run them, with their own coverage gate (100 % line and
branch), from the repository root:

```sh
uv run pytest tests/replay_eval -q --cov=tools/replay_eval --cov-branch --cov-report=term-missing
uv run ruff check tools/replay_eval tests/replay_eval && uv run mypy tools/replay_eval
```

The `[tool.coverage.report] fail_under = 100` setting in `pyproject.toml` applies to this run too.
