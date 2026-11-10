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
| `--targets FILE.json` | Overrides targets: `{metric id: {"op": "<=" or ">=", "value": x}}`, optionally under `"targets"`. |
| `--accept-if JSON` | The `accept_if` sent with every update when O9 is served. The default is `{}`, meaning the service's default thresholds. |
| `--warping e034,e040` | The episodes whose warping update the gate must reject (C6). |
| `--no-height` | No `height_above_floor_m` in the `none` replay; its scale is then reported as `c2.scale_error_monocular`. |
| `--no-probes` | No reads while a job runs. This drops C3 one-version reads, the busy C12 latencies and the C3 fusing cadence. |
| `--no-jsonschema` | Minimal required-keys validation, even if `jsonschema` is installed. |
| `--wait-s`, `--probe-interval-s`, `--job-timeout-s`, `--timeout-s`, `--retries`, `--retry-wait-s`, `--range-scans`, `--align-min-spread-m` | Polling, timeouts, retries of retryable errors, frame-analysis samples and the alignment spread (see below). |

The exit status is 0 when at least one replay was measured, and 1 otherwise.

## What a replay does

1. `GET /api/health` reads `version`, `capabilities` and `timings`. Metrics whose group is not
   in `capabilities` are `not_measurable`, and each one names the missing group.
2. The agent's requests in `oms/agent_requests.jsonl` are replayed in order. Each update request
   becomes one update job carrying that frame set. It is a multipart request: a `request` JSON
   part plus one file part per frame, with the camera from the intrinsics table. Each locate
   request becomes one `POST locate` (O5). A set that the requests never name was not sent, and
   is not sent here either. A set sent again as an update is skipped, as the agent's retry
   (`requests.dedupe_updates`).
   * `metric` replay: each frame carries its `views.json` pose as a `metric` prior, with sigmas
     0.02 m and 1°. A later run's poses are moved into the first run's start frame through the
     truth start poses, because a map has one frame.
   * `none` replay: `trust: none`. Floor robots also send `height_above_floor_m` = `pose[2][3]`
     of the `views.json` pose, with sigma 0.02 m.
   * With O10, a dry run (`?dry_run=1`) precedes each update, for `predicted_s`. With O9,
     `accept_if` is sent.
3. Each job is polled with `GET /api/jobs/{id}?wait_s=` until it ends. Meanwhile a second thread
   reads `GET frames` every `--probe-interval-s`, and also runs a one-frame locate (O5) and
   analyse (O7) on a keyframe of the previous version. After the job, every version a probe saw
   is read again with `?version=n`, and must have the same content. The provisional and result
   versions' frames are read, as are the result version's objects (O8). One idle one-frame
   locate and analyse follow.
4. At the end come the map summary, the grid (`cell_m=0.05&z_min=0.05&z_max=1.8`, O6), the
   objects (O8), range scans on `--range-scans` keyframes with their `T_map_cam` (O7), the
   versions list, and a revert to `latest - 1` (O2). The map is then deleted.
   * If the episode has no locate request, every sent frame that is not a keyframe is located,
     in batches of 8.

Every response is validated against `specs/api.schema.json`, as is every error body
(`ErrorBody`). A missing `Server-Timing` header is counted too.

## Assumptions about the episode keys (to confirm with `--dump-layout`)

They follow the conductor's layout (`~/oh-my-slam-data/robot-guide/episode_layout.md`). Each is a
key of `LAYOUT` in `rpl_layout.py`.

* **Frames.** `agent_files/runs/<run>/<set>/<set>-YYYYMMDD-HHMMSS-NN.jpg`, with one `views.json`
  per set. The file-name time is local time at UTC+1 (`frame_name.utc_offset_h`).
* **`views.json`.** A dict keyed by file name. Each value is `{"pose": 4x4, ...}`: the base has
  `base`, and the humanoid has `stamp` (epoch s). For the arm, the value is the bare 4x4. The
  pose is the optical pose in the run's start frame.
* **`truth.jsonl`.**
  * Every line has `kind`. The first `meta` line holds `robot` and `hello.{scene, floor_z,
    worktop.z}`.
  * `robot` lines hold `t` (epoch s), `pos`, `quat` (`[w, x, y, z]`), `cams.<camera>` (the
    optical centre in the world) and `bodies`.
  * `objects` lines hold `bodies.<name>.{pos, quat}`.
* **Robots.** The table maps `meta.robot` to a kind, a camera name and intrinsics:
  * `ainex`: humanoid, `camera`, 640×480, f 270, c (319.5, 239.5);
  * `rebot_b601_rs`: arm, `camera_color_optical_frame`, 640×360, f 337.2, c (319.5, 179.5);
  * `rosmaster_x3_plus`: base, `camera_color_optical_frame`. Its fx and fy come from each
    frame's EXIF: FocalLength × FocalPlaneX/YResolution, with the unit taken as mm when the tag
    is absent. cx and cy are at the image centre.

  A frame of another size gets scaled intrinsics and a warning. `calibration.md` is not read.
* **`oms/agent_requests.jsonl`.** The keys are not known. A line is an update or a locate by the
  earliest of the words `locate`, `update` or a skip word (`segment`, `reconstruct`, ...) in its
  JSON text. It names its frames by any of these, in this order:
  * frame file names;
  * a `run/set` path;
  * a path component equal to a set name.

  A file-name list that covers only part of a set sends only those frames. `--dump-layout`
  prints the first three lines and the classification counts.
* **Truth pose of a frame.**
  * **Start frame to world.** The truth base pose at the run's first frame. For floor robots,
    this is its yaw and its (x, y) at `floor_z`. For the arm, it is the full pose.
  * **Position.** The truth `cams.<camera>` position at the frame's time:
    * the humanoid's `stamp`, interpolated;
    * otherwise the file-name time. That time has a resolution of 1 s and is shared by a whole
      set, so the frame's position is the point of the truth track nearest the views position
      (composed with the start pose), within the time window between the neighbouring sets.
    * Without any time, the views position composed with the start pose (`truth_source:
      views`, with a warning).
  * **Orientation.** The `views.json` rotation composed with the start pose, unless
    `rotation_body.<kind>` names a truth body whose quaternion is the camera's optical
    orientation. For the drifting humanoid, this orientation carries its odometry drift. Its
    positions are exact.
  * **Check.** `--dump-layout` prints `views_vs_truth_m`: the median distance between the views
    positions and the truth positions. A large value means one of these assumptions is wrong.
* **House JSON.** Simulator world (x, y, z) = THOR (x, z, y).
  * Rooms: `floorPolygon`. Walls: `polygon`, or ids `wall|room|x0|z0|x1|z1`.
  * Doors: the `holePolygon` x range along `wall0` from its first point.
  * Objects: `position`, with `children` resting on or (for container types) inside their
    parent. Their label is the id before the first `|`.

## Metrics (`rpl_metrics.py`)

Each metric is a list of samples, reduced to a share, a median, a p90 or a maximum, and compared
with its target. The overall value pools the samples of all replays. Statuses are `pass`, `fail`,
`info` (no target), `no_samples` and `not_measurable` (with `missing`).

| Group | Metrics |
| --- | --- |
| Always | `c3.readable_s` (upload → `provisional_version`/`result_version` seen, p90 ≤ 2 s); `c3.fusion_s_per_frame` (p90 ≤ 3 s); `c3.reads_one_version` (100 %); `c4.reason_agreement` (≥ 0.8); `c4.inputs_contract`; `c2.scale_error_priors` (≤ 3 %, metric replay), `c2.scale_error_height` (≤ 10 %, none replay with heights), `c2.scale_error_monocular` (info); `c1.shift_pos_m` (≤ 2 cm) and `c1.shift_rot_deg` (≤ 0.5°), as medians beyond `quality.gauge_shift`; `contract.schema_valid`; `contract.server_timing` |
| O2 | `c5.revert_s` (≤ 2 s); `c5.revert_contract` |
| O3 | `c3.eta_within_30pct` (job `eta_s`, and dry-run `predicted_s` when O10 is served); `c3.fusing_version_gap_s` (≤ 10 s when fusion lasts longer) |
| O5 | `c8.located_near_keyframe` (≥ 0.9); `c8.pos_error_over_tolerance` (median ≤ 1, where the tolerance is max(2 cm, 1 % of the scene depth)); `c8.pos_error_m` (info); `c8.ambiguity_on_wrong` (≥ 0.8); `c8.ambiguity_null_on_right` (≥ 0.9); `c8.ambiguity_rule` (api.md §6); `c8.nearest_view_turn` (≥ 0.8); `c12.locate_idle_s` (≤ 2 s); `c12.locate_busy_s` (≤ 3 s); `c12.locate_busy_ratio` (≤ 2) |
| O6 | `c9.camera_cells_free`; `c9.regions_set`; `c9.band_as_asked`; `c9.floor_height_error_m` (≤ 2 cm); `c9.grid_read_s` (≤ 0.5 s). With a floor plan: `c9.grid_agreement` (≥ 0.9); `c9.doors_found` (≥ 0.8); `c9.passage_geometry` (≥ 0.8); `c9.floor_observed` (≥ 0.9); `c9.band_occupied` (≥ 0.95) |
| O7 | `c12.analyse_busy_s` (≤ 3 s); `c12.analyse_busy_ratio` (≤ 2); with a floor plan, `c11.range_rel_error` (median ≤ 5 %) |
| O8 | `c10.objects_read_s` (≤ 0.5 s); `c10.centre_error_m`; `c10.label_agreement`; `c10.id_stability`; `c10.support_height_error_m` (≤ 2 cm); `c10.relations_agreement` (≥ 0.9) |
| O9 | `c6.warping_rejected` (every `--warping` episode has an update rejected by the gate) |

The truth-derived checks have these thresholds (`Params` in `rpl_metrics.py`):

* **Alignment.** Umeyama fit of the keyframe camera positions against the truth positions, on
  the first version whose truth positions span 0.5 m (`--align-min-spread-m`). The map frame is
  fixed by the first update, so a later version is in the same frame. Nearly collinear
  positions get a yaw-only fit, since both frames are z-up.
* **C4 truth reasons.**
  * `too_close`: the floor-plan wall along the heading is under 0.35 m away.
  * `plain`: all 9 rays across the horizontal field of view hit walls under 1.5 m away, and no
    truth object stands in front of the walls. `few_features` counts as agreeing with it.
  * `no_overlap`: no earlier frame of the replay lies within 1.5 m with its optical axis less
    than one horizontal field of view away.
  * Without a floor plan, only `no_overlap` is judged.
* **C8.**
  * Near a keyframe: within 0.3 m and 15° of another frame of the located version.
  * Scene depth: the median wall distance along the keyframes' headings, or their height above
    the worktop or the floor.
  * `nearest_view_turn`: within 45° of the truth turn.
* **C9 grid.**
  * Floor-plan classes come from a raster:
    * wall: within 7.5 cm of a solid wall;
    * not judged: 7.5–15 cm from one;
    * door gap;
    * floor inside rooms.
  * Grid cells are compared at their world positions. Agreement counts observed cells: floor
    must be free, walls occupied.
  * Doors are judged when some observed cell lies within 0.3 m of the gap centre.
  * The blind ring has the radius at which the image's bottom edge meets the floor. Floor-plan
    walls inside it may stay occupied.
* **C10.** Objects are matched to the nearest truth object within 1 m, of the same label for
  the centre error. The support height is judged against the worktop (arm) or the floor
  (ProcTHOR objects without a parent). The relation is judged against the house's
  `children` nesting.
* **C11.** Beams of kind `obstacle` are compared with a ray cast against the floor-plan walls.
  `bearing_deg` is taken as a map-frame yaw from the camera position, as the call carries
  `T_map_cam`.

The C10 targets (`c10.centre_error_m` ≤ 0.10 m, `c10.label_agreement` ≥ 0.8,
`c10.id_stability` ≥ 0.8) are placeholders until the round-4 values reach `targets.json`.
Override them with `--targets`.

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
