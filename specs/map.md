# Map model

Part of the [high-level specification](../high_level_spec.md) (§5). Field names refer to
[api.schema.json](api.schema.json).

## 1. Anchored map frame (O1)

* **When the frame is fixed.** The map frame is fixed by the first committed update and never
  changes afterwards. No re-gauging.
* **With priors** (`metric` or `drifting` on at least one frame of the first update):
  * the map frame is the priors' frame;
  * the priors must be gravity-aligned with z up;
  * the first update reports the angle between the priors' up axis and the gravity it
    estimates (`quality.prior_gravity_deg`).
* **Without priors:**
  * the origin is the optical centre of the first registered frame, in request order;
  * +x is that camera's viewing direction projected onto the horizontal plane;
  * +z is up, estimated from gravity and the floor plane.
* **Re-estimating poses.** An update may re-estimate the poses of stored frames, for example in
  a rebuild after a loop closes. Its result is then aligned back to the previous version before
  commit, by a similarity transform fitted on the frames both versions hold. The update reports:
  * that transform's size as `quality.gauge_shift` (translation in m, rotation in deg, scale
    ratio);
  * the residuals of the anchors, which are the metric priors and the previous version's poses.
* **Scale.** The scale comes from the first available of:
  1. `metric` priors (`scale_source: priors`);
  2. depth images (`depth`);
  3. camera heights above the floor, per frame or per camera (`camera_height`);
  4. monocular depth (`monocular`).

  `scale_sigma_rel` is estimated from the residuals of that source. A later update may refine
  the scale: `quality.scale_ratio_vs_prev` reports the change.

## 2. Versions (O2)

* **Versions.** Every commit creates version *n + 1*: integers from 1, never reused, immutable
  once written. A version holds:
  * the frames and their poses, each pose marked `provisional` or `fused`;
  * the cloud and its free-space fusion;
  * the objects;
  * the grid inputs;
  * the scale;
  * the metadata.
* **Reads.**
  * A read without `?version=` answers from the latest committed version at the moment the
    request arrives.
  * Each answer comes from one version and names it in `map_version`, even while later
    versions are committed.
  * A version that is no longer kept answers `version_gone`.
* **Revert.** `revert {to_version}` commits a new version whose content equals `to_version`.
  History stays linear and readers never see a version change in place.
* **Retention.** The service keeps at least the last *N* versions of each map (default 20,
  recorded in `README.md`). Older ones are deleted, except one that a running job or read is
  using. Kept versions share their unchanged content, so a version costs only what changed.

## 3. Update jobs (O3)

An update request is validated synchronously and refused at once if it is invalid. Otherwise it
becomes a job with a `job_id`. A job goes through these states:

| State | Meaning |
| --- | --- |
| `queued` | Waiting for the map's earlier jobs. |
| `registering` | Phase 1: registers every new frame against the version current when the phase starts, and computes the quality report. If the gate (§4) passes, it commits a version with the new frames at provisional poses, and the new frames are readable from then on. |
| `fusing` | Phase 2: fuses depth into the cloud and the free space, then updates the objects, refined poses and the scale. While it lasts longer than the interval in `README.md` (default 10 s), it commits an intermediate version (`kind: fusing`) at that interval, so the grid and objects grow during fusion. It commits the final version. |
| `committed` | Done. `result_version` is the final version. |
| `rejected` | The gate failed. No version was committed, and `quality` states why. |
| `cancelled` | Cancelled by `DELETE /api/jobs/{id}`, see below. |
| `failed` | Ended by an error. Whatever was committed stays as is; `error` says what happened. |

* **Cancel.**
  * A job cancelled before registration commits nothing.
  * A job cancelled during fusion keeps its provisional version as the map's latest. The
    frames stay provisional, and the next job fuses them.
* **Progress.** Each job reports `progress` (0–1) and `eta_s` for its current phase.
* **What ends a job.** A client disconnect never cancels a job. Only a cancel or a stop of the
  service ends it. A job still running when the service stops is `failed` on restart.
* **Order of addition.** A later job wins over an earlier one, and within one job a later frame
  wins over an earlier one, in request order. Capture timestamps are not used.
* **What a later observation does.**
  * A later observation of a surface or object replaces the earlier one: an object seen gone is
    `struck` (§5), and the surface behind it is refilled from the latest view.
  * An object id persists for the lifetime of the map.
* **Split updates.** Mapping one sequence in one job, or split across several in the same order,
  gives the same objects and poses within the evaluator's targets.

## 4. Quality report and gate (O9)

* **The quality report.** Every job, dry runs included, reports `quality`:
  * `scale_ratio_vs_prev` and `gauge_shift`;
  * the anchor residuals (median and max, in m and deg, against metric priors and against the
    previous version);
  * `rejected_share`: inputs rejected over inputs sent;
  * `low_confidence_share`: registered inputs whose pose sigma exceeds the threshold in
    `README.md`;
  * `rebuilt`: whether stored poses were re-estimated;
  * `prior_gravity_deg`, on a map's first update.
* **The gate.** The request option `accept_if {max_residual_m, max_gauge_shift_m,
  max_rejected_share}` is evaluated on the registration phase's report, before anything is
  committed.
  * A failing job is `rejected` with `quality.accepted: false` and the failing checks listed.
  * Without `accept_if`, the job is accepted. Hard limits in `README.md` still reject updates
    that would make the map inconsistent, for example a frame set that registers to nothing.

## 5. Objects (O8)

Each object record has:

| Field | Meaning |
| --- | --- |
| `id` | Positive integer, persistent for the map's lifetime. |
| `label`, `detected_as[]`, `score` | Label chosen, labels it was detected as, and confidence. |
| `centre_m`, `size_m`, `yaw_deg` | Upright oriented box in the map frame: z is up, x is the longer horizontal side. |
| `sigma_centre_m` | One-sigma uncertainty of the centre. |
| `extent_seen {x, y, z}` | `full` or `partial` per box axis, from the views that bound it. |
| `mean_rgb` | Observed colour: robust mean of the object's pixels in sRGB 0–255, from the views that saw it best. |
| `color` | Display colour only: a pure function of `id`. |
| `state` | `confirmed` (seen again where it was), `moved` (seen elsewhere: `moved_from_m` gives the earlier centre), or `struck` (seen gone; kept for one version, then removed). |
| `first_seen_version`, `last_seen_version` | Versions where it first and last appeared. |
| `region` | The grid region (§6) of the object, or null. It is set when its centre or footprint lies in a region or on its boundary; the region chosen is the one its footprint overlaps most. An object on a wall, such as a door or a shelf, belongs to the region it was seen from. Null only when it touches no region. |
| `supported_by` | `{kind: surface \| object, id, verified_version}` for the support surface or object it rests on, or null. |
| `inside_of` | `{id, verified_version}` when its centre lies within another object's concave interior (a bowl, a drawer, a box), or null. |

`Objects` also lists `surfaces[]`: the horizontal support planes estimated from the fused
points, such as the floor, table tops, counters and shelves. Each has `id`, `height_m`,
`polygon_m` and `area_m2`. An arm descends to and places on these heights.

* **Doorways.** A door opening is reported as a passage of the grid (§6), not as an object.
* **Floor height.** The floor height used for boxes comes from the grid's floor estimate, so
  boxes and grid agree.

## 6. Navigation grid (O6)

`GET /api/maps/{name}/grid` returns a 2-D grid built from the free space the fusion keeps:
* cells seen through are **free**;
* cells holding surfaces within the height band are **occupied**;
* everything else is **unknown**.

**Where cameras stood is free.** The cells under each keyframe's camera position are free,
whatever the camera saw, because the robot stood there. So is the ring around that position
that the camera could not see: its near blind zone on the floor, from the camera's height and
its lower field-of-view edge.

**The band applies as asked.** `occupied` uses the requested `z_min`–`z_max`, whatever its
height. A request whose `z_max` is the robot's height plus a margin marks everything the robot
could hit.

A cell under an overhang that lies wholly above the band's `z_max`, such as a table top above
the band, is free if it was seen through below the overhang, and unknown otherwise. Only surfaces
inside the band make a cell occupied. `floor_observed` (below) says whether the floor itself was
seen there.

**Parameters.**
* `cell_m`: the cell size.
* `z_min`, `z_max`: the height band in the map frame, relative to the floor estimate
  `floor_z_m` (default 0.05–1.8 m).
* Optional `footprint_radius_m`: the robot's radius.

**Header.**
* `map_version`, `origin_m` (the corner of cell [0, 0]), `cell_m`, `shape [rows (y), cols (x)]`,
  `z_band_m` and `floor_z_m`;
* `scale_source` and `scale_sigma_rel`.

**Layers.** Each layer is a little-endian array, base64, row-major:
* `state`: uint8, 0 unknown, 1 free, 2 occupied;
* `observations`: uint16, the number of frames that observed the cell;
* `height_m`: float32, the highest surface in the cell, NaN if none;
* `floor_observed`: uint8, 1 when a floor surface was observed in the cell at the floor height.
  A cell that is free but whose floor was not observed means the camera saw through it above a
  floor nobody saw: there may be a low obstacle or a step, so the planner looks first.

**Features.**
* `regions[]` — the free space split into regions at passages. Each region has `id`,
  `area_m2`, `polygon_m`, and `coverage_share`, the observed share of its floor, counting
  unknown cells enclosed by its boundary as unobserved.
* `passages[]` — narrowings between two regions that a robot can drive through. Each has:
  * `jambs_m`, the two end points of the free gap at floor level;
  * `centre_m`, the midpoint of the free gap between the jambs. It is not the midpoint of the
    door frame, nor of the open leaves' free ends;
  * `width_m`, the clear width between the jambs: the free width at the narrowest point within
    the height band, with open leaves counted as obstacles;
  * `normal_deg`, the crossing direction: perpendicular to the jamb line, from `regions[0]` to
    `regions[1]`;
  * `regions [a, b]`;
  * for kind `door`, `leaf`: null, or `{polygon_m, intrudes}`, the leaf's footprint and whether
    it intrudes into the gap;
  * `kind`:
  * `door`, when a door-labelled object or a door-shaped gap is there;
  * `opening`;
  * `narrow`, narrower than 2 × `footprint_radius_m` when that parameter is given.
* `frontiers[]` — boundaries between free and unknown cells, longer than a minimum length. Each
  has `id`, `centre_m`, `length_m` and `region`. `region` is set whenever the frontier borders a
  region, and is null only for a frontier that touches none.

**Identifiers.** Region and passage ids are stable between versions where the geometry did not
change, so the skill can refer to "the kitchen" across updates.

## 7. Points near a place

`GET /api/maps/{name}/points?min=x,y,z&max=x,y,z&voxel_m=&max_points=` returns the fused points
inside an axis-aligned box of the map frame [`Points`]:
* `xyz`, `object_id` and `rgb` as base64 arrays, as in frame analysis;
* thinned to `voxel_m` (default 0.01);
* capped at `max_points` (default 200 000), with `truncated` stating whether the cap applied.

There is no whole-map cloud read.

## 8. Bird's-eye image

`GET /api/maps/{name}/birdseye.png` draws the version seen from above:
* the cloud, the objects' footprints with their ids, and the keyframe cameras;
* the PNG's text metadata records `map_version` and `pixel_to_map`, an affine map from pixel to
  map x, y.

`GET /api/maps/{name}/birdseye` returns the same metadata as JSON.
