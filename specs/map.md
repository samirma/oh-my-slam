# Map model

Part of the [high-level specification](../high_level_spec.md) (§5). Field names refer to
[api.schema.json](api.schema.json).

## 1. Anchored map frame (O1)

* **When the frame is fixed.** The map frame is fixed by the first committed update and never
  changes afterwards. No re-gauging.
* **With priors** (`metric` or `drifting` on at least one frame of the first update). This is
  the normal case: on a cold start every robot, camera-only included, sends its first frames with
  `metric` priors, with the start pose on the floor as origin. Frames after walking without
  odometry come as `drifting` or `none`.
  * the map frame is the priors' frame, always: whenever priors are sent, the rule without
    priors below never applies;
  * the priors must be gravity-aligned with z up;
  * the first update reports the angle between the priors' up axis and the gravity it
    estimates (`quality.prior_gravity_deg`);
  * if every prior-carrying frame of the first update is `rejected`, the update is rejected:
    job state `rejected`, `quality.failed_checks` holding `prior_frames_rejected`, no version
    committed and no map created. A frame `kept_at_prior` is not rejected.
* **Without priors:**
  * the origin is the optical centre of the first registered frame, in request order;
  * +x is that camera's viewing direction projected onto the horizontal plane;
  * +z is up, estimated from gravity and the floor plane.
* **Without priors, with a camera height** (`height_above_floor_m`): z = 0 is the floor plane,
  the same convention as with priors. The origin is straight below the first registered
  frame's optical centre; +z is up, gravity-aligned; +x is from the first camera, as above.
* **Re-estimating poses.** An update may re-estimate the poses of stored frames, for example in
  a rebuild after a loop closes. Its result is then aligned back to the previous version before
  commit, by a similarity transform fitted on the frames both versions hold. The update reports:
  * that transform's size as `quality.gauge_shift` (translation in m, rotation in deg, scale
    ratio);
  * the residuals of the anchors, which are the metric priors and the previous version's poses.
* **Never re-posing.** An engine that never re-estimates stored poses has no such transform.
  Its `quality.gauge_shift` is the movement of the stored keyframes: over the previous
  version's keyframes, the median translation (m) and rotation (deg) between their pose in the
  previous version and the pose that would be committed; `scale_ratio` 1. It is about 0 when
  no stored keyframe moved. Stored `kept_at_prior` keyframes are stored keyframes: their
  movement counts here and in `previous_residuals`.
* **Residuals against the previous version.** `quality.previous_residuals` covers the earlier
  keyframes only: its `count` is at most the previous version's keyframe count. A stored
  keyframe that moves beyond tolerance shows in `previous_residuals.max_m` / `max_deg` and is
  gated by `max_residual_m` (§4).
* **New-frame disagreement.** The disagreement of the update's new frames between committed
  pose and image-only pose (the pose their images alone give against the map) is in
  `quality.max_displacement_m` and `max_displacement_deg` (§4), not in `gauge_shift`:
  * a frame whose image-only pose is unconstrained in translation (parallax, baseline over
    median scene depth, below 0.02) counts only its rotation;
  * frames `kept_at_prior` or `rejected` have no image-only pose and are left out.

  So the gates on `gauge_shift` and `max_residual_m` measure stored-map movement, and the gates
  on `max_displacement_m` / `max_displacement_deg` measure new-frame disagreement. Prior
  corrections show in `prior_disagreement_deg` and `prior_residual_*` only.
* **Priors against images.** A prior is weighed by its sigmas against the relative pose the
  images give between overlapping frames. Where two frames overlap by at least 50 % and their
  image relative rotation disagrees with their priors' relative rotation by more than 3 of the
  priors' rotation sigmas (`metric` and `drifting` priors alike), the rotation is corrected from
  the images. `quality` always reports `prior_disagreement_deg`: the largest such disagreement
  over the update's frames, 0 when none.
* **Scale.** The scale comes from the first available of:
  1. `metric` priors (`scale_source: priors`);
  2. depth images (`depth`);
  3. camera heights above the floor, per frame or per camera (`camera_height`);
  4. monocular depth (`monocular`).

  `scale_sigma_rel` is estimated from the residuals of that source. It bounds the relative error
  of any horizontal distance, in every direction. A later update may refine
  the scale: `quality.scale_ratio_vs_prev` reports the change.

## 2. Versions (O2)

* **Versions.** Every commit creates version *n + 1*: integers from 1, never reused, immutable
  once written. A version holds:
  * the frames and their poses, each pose marked `provisional` (registered, not yet fused),
    `fused`, or `kept_at_prior` (the prior, unconfirmed by the images);
  * the cloud and its free-space fusion. Occupancy, heights, the cloud and object records come
    only from image-confirmed frames; a `kept_at_prior` frame counts only for free-by-rule cells
    (§6) until it is registered. A later update may re-register a stored `kept_at_prior`
    frame: when a later job's images confirm it, its pose leaves `kept_at_prior` in that
    version (it becomes a registered pose, `provisional` then `fused`), and from then on it
    feeds occupancy, heights, the cloud and objects. There is no automatic background
    refinement;
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
* **Retention (`--keep-versions`).** The service keeps at least the last *N* versions of each
  map (default 20, recorded in `README.md`). Older ones are deleted, except one that a running
  job or read is using. Kept versions share their unchanged content, so a version costs only what changed.

## 3. Update jobs (O3)

An update request is validated synchronously and refused at once if it is invalid. Otherwise it
becomes a job with a `job_id`. A job goes through these states:

| State | Meaning |
| --- | --- |
| `queued` | Waiting for the map's earlier jobs. |
| `registering` | Phase 1: registers every new frame against the version current when the phase starts, and computes the quality report. If the gate (§4) passes, it commits a version with the new frames at provisional poses, and the new frames are readable from then on. |
| `fusing` | Phase 2: fuses depth into the cloud and the free space, then updates the objects, refined poses and the scale. While it lasts longer than the interval in `README.md` (default 10 s), it commits an intermediate version (`kind: fusing`) at that interval, so the grid and objects grow during fusion. It commits the final version. |
| `committed` | Done. `result_version` is the final version. |
| `rejected` | The gate failed, or every prior-carrying frame of a map's first update was rejected (§1). No version was committed, and `quality` states why. |
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
  wins over an earlier one, in request order. Capture timestamps are not used. An engine may
  register frames in any internal order, but the committed result equals request-order
  semantics: a later observation wins (C13).
* **What a later observation does.**
  * A later observation of a surface or object replaces the earlier one: an object seen gone is
    `struck` (§5), and the surface behind it is refilled from the latest view.
  * An object id persists for the lifetime of the map.
* **Split updates.** Mapping one sequence in one job, or split across several in the same order,
  gives the same objects and poses within the evaluator's targets.

## 4. Quality report and gate (O9)

* **The quality report.** Every job, dry runs included, reports `quality`:
  * `scale_ratio_vs_prev` and `gauge_shift` (§1);
  * `max_displacement_m` and `max_displacement_deg`: over the update's new frames that have an
    image-only pose, the largest translation and rotation between committed pose and
    image-only pose (§1), or null when there is none. A frame unconstrained in translation
    (§1) contributes to `max_displacement_deg` only;
  * `prior_disagreement_deg` (§1), 0 when no image relative rotation disagrees with the priors;
  * the anchor residuals (median and max, in m and deg, against metric priors and, over earlier
    keyframes only, against the previous version);
  * `rejected_share`: inputs rejected over inputs sent;
  * `low_confidence_share`: registered inputs whose pose sigma exceeds the threshold in
    `README.md`;
  * `rebuilt`: whether stored poses were re-estimated;
  * `prior_gravity_deg`, on a map's first update.
* **The gate.** The request option `accept_if {max_residual_m, max_gauge_shift_m,
  max_displacement_m, max_displacement_deg, max_prior_disagreement_deg, max_rejected_share}` is
  evaluated on the registration phase's report, before anything is committed:
  * `max_residual_m` gates `previous_residuals.max_m` (stored keyframes) only, so a stored
    keyframe that moved fails the update. New-frame disagreement is gated only by
    `max_displacement_*`. Prior corrections, shown in `prior_disagreement_deg` and
    `prior_residual_*`, never fail it;
  * `max_gauge_shift_m` gates the stored keyframes' median movement;
  * `max_displacement_m` and `max_displacement_deg` gate `quality.max_displacement_m` and
    `quality.max_displacement_deg`, so a single new frame that disagrees with its images fails
    the update, a frame unconstrained in translation through its rotation. A null value (no
    measurable frame) passes;
  * `max_prior_disagreement_deg` gates `quality.prior_disagreement_deg`, so a client can refuse
    an update that bends its priors. Without it there is no such gate.
  * A failing job is `rejected` with `quality.accepted: false` and the failing checks listed.
  * `accept_if: {}`, or a field left out, applies no check for that field.
  * Without `accept_if`, the job is accepted. Hard limits in `README.md` still reject updates
    that would make the map inconsistent, for example a frame set that registers to nothing.

## 5. Objects (O8)

Each object record [`MapObject`] has the fields below; frame analysis returns the subset
[`AnalysisObject`] (api.md §7). Every field is present in an objects read.

| Field | Meaning |
| --- | --- |
| `id` | Positive integer, persistent for the map's lifetime. |
| `label`, `detected_as[]`, `score` | Label chosen, labels it was detected as, and confidence. |
| `detection_votes[]` | `{label, share}`, highest share first: `share` is the number of frames that gave that label over the number of frames in which the object was in view, detected or not, so shares sum to at most 1 and may sum to less. Map objects only; frame analysis does not carry it. |
| `centre_m`, `size_m`, `yaw_deg` | Upright oriented box in the map frame: z is up, x is the longer horizontal side. |
| `sigma_centre_m` | One-sigma uncertainty of the centre. When the map's builder never recorded which views bound the box, it comes from the formula documented in `README.md`. |
| `extent_seen {x, y, z}` | `full` or `partial` per box axis, from the views that bound it. When the map's builder never recorded those views, `partial` on every axis. |
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

* **Doorways.** A door opening is a passage of the grid (§6), not an object. A detected door
  leaf is an object: label `door`, with its `state` and `region`. The passage names it in
  `leaves[].object_id`. A door object is a leaf: thin (< 0.20 m), vertical and at least 0.6 m
  tall. Detections of a doorway, a gate or an opening are not objects; they may only hint a
  passage.
* **Floor height.** Objects use the grid's floor estimate: one estimate per version, with the
  same `floor_z_m` and `floor_sigma_m`, so boxes and grid agree.
* **Floor band.** A surface within max(0.06 m, 3 × `floor_sigma_m`) of `floor_z_m` is floor
  (§6): it never makes a floor-standing object. This minimum height applies
  only above the floor estimate: an object resting on any other support surface keeps its
  detection whatever its height (a 16 mm plate or an egg on a worktop is still an object). A
  surface of `surfaces[]` is never itself raised to an object by this rule.

## 6. Navigation grid (O6)

`GET /api/maps/{name}/grid` returns a 2-D grid built from the free space the fusion keeps:
* cells seen through are **free**;
* cells holding surfaces within the height band are **occupied**, except floor;
* everything else is **unknown**.

**Where cameras stood is free.** The cells under each keyframe's camera position are free,
whatever the camera saw, because the robot stood there. So is the ring around that position
that the camera could not see: its near blind zone on the floor, from the camera's height and
its lower field-of-view edge.

**Unconfirmed frames.** A `kept_at_prior` keyframe counts only for these free-by-rule cells
(camera stand and near blind disc) until a later job's images confirm it (§2). Occupancy,
heights, the cloud and object records come only from image-confirmed frames.

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
  `floor_z_m` (default 0.05–1.8 m). `floor_z_m` is the surface the robot stands on:
  * when priors fixed the map frame, it is the large horizontal plane nearest the priors'
    origin height (z = 0). Robots put that origin where they stand: on the floor for a base or a
    humanoid, at the base link on the worktop for an arm;
  * without priors but with a camera height (`height_above_floor_m`), it is the floor plane the
    height defines (about 0, §1), refined by an observed large horizontal plane within 3 sigma
    of it;
  * with neither priors nor a camera height, it is the lowest large horizontal plane seen below
    the cameras.

  Other planes, such as the room floor seen past a worktop's edge, are support surfaces
  (`surfaces[]`), not the floor.

  **Floor band.** A surface within max(0.06 m, 3 × `floor_sigma_m`) of `floor_z_m` is floor:
  it never makes an occupied cell or defines a passage, whatever `z_min` is.
* Optional `footprint_radius_m`: the robot's radius.
* `height_cap_m`: the ceiling of the `height_m` layer, relative to `floor_z_m`. Default 2.0 m,
  or the band's `z_max` when the request gives one. Echoed in the header.

**Header.**
* `map_version`, `origin_m` (the corner of cell [0, 0]), `cell_m`, `shape [rows (y), cols (x)]`,
  `z_band_m`, `floor_z_m`, `floor_sigma_m` and `height_cap_m`. `z_band_m` and `height_cap_m`
  are relative to `floor_z_m`, as the request's `z_min` and `z_max` are. `floor_z_m` is the
  floor's absolute map-frame z, and `floor_sigma_m` the one-sigma uncertainty of its height;
* `scale_source` and `scale_sigma_rel`.

**Layers.** Each layer is a little-endian array, base64, row-major:
* `state`: uint8, 0 unknown, 1 free, 2 occupied;
* `observations`: uint16, the number of frames that observed the cell;
* `height_m`: float32, the highest surface in the cell at or below `height_cap_m`, relative to
  `floor_z_m`, NaN if none. Surfaces above the cap, such as a ceiling, are left out;
* `floor_observed`: uint8:
  * 1 — a floor surface was observed in the cell at the floor height;
  * 2 — not observed, but free by rule: a camera stood here, or the cell lies inside a camera's
    near blind disc;
  * 0 — not observed.
  A cell that is free but whose floor was not observed means the camera saw through it above a
  floor nobody saw: there may be a low obstacle or a step, so the planner looks first.

**Features.**
* `regions[]` — the free space split into regions at passages. Each region has `id`,
  `area_m2`, `polygon_m`, and `coverage_share`, the observed share of its floor, counting
  unknown cells enclosed by its boundary as unobserved.
* `passages[]` — narrowings between two regions, or between a region and a never-observed far
  side, that a robot can drive through. Each has:
  * `jambs_m`, the two end points of the free gap at floor level;
  * `centre_m`, the midpoint of the free gap between the jambs. It is not the midpoint of the
    door frame, nor of the open leaves' free ends;
  * `width_m`, the clear width between the jambs: the free width at the narrowest point within
    the height band, with open leaves counted as obstacles;
  * `normal_deg`, the crossing direction: perpendicular to the jamb line, from `regions[0]` to
    `regions[1]` (or toward the unobserved side);
  * `regions [a, b]`, or `[a, null]` for a doorway whose far side was never observed;
  * `frontier`: when `regions[1]` is null, the id of the frontier (below) on the unobserved
    side; otherwise null;
  * `leaves[]`: the door leaves found at the passage, each `{polygon_m, intrudes, object_id}`:
    the leaf's footprint, whether it intrudes into the gap, and its object id (§5). Empty when
    none;
  * `leaf`: deprecated, `leaves[0]` or null;
  * `kind`: `door` when a leaf (a door object) is found, so `leaves` is not empty; else
    `opening`. (`narrow` is no longer a kind: see below.)
  * `narrow`: true when `width_m` is below 2 × `footprint_radius_m`, false when not, null when
    that parameter is not given.
* `frontiers[]` — boundaries between free and unknown cells, longer than a minimum length. Each
  has `id`, `centre_m`, `length_m` and `region`. `region` is set whenever the frontier borders a
  region, and is null only for a frontier that touches none.

**Identifiers.** Region and passage ids are stable between versions where the geometry did not
change, so the skill can refer to "the kitchen" across updates.

## 7. Points near a place (O6)

`GET /api/maps/{name}/points?min=x,y,z&max=x,y,z&voxel_m=&max_points=` returns the fused points
inside an axis-aligned box of the map frame [`Points`]:
* `xyz`, `object_id` and `rgb` as base64 arrays, as in frame analysis;
* thinned to `voxel_m` (default 0.01);
* capped at `max_points` (default 200 000), with `truncated` stating whether the cap applied.

There is no whole-map cloud read.

## 8. Bird's-eye image (O1)

Bird's-eye belongs to step 1 of the order of work (high_level_spec.md §12, §13).

`GET /api/maps/{name}/birdseye.png` draws the version seen from above:
* the cloud, the footprints of the present objects (`confirmed` or `moved`) with their ids,
  and the keyframe cameras. `struck` objects are not drawn;
* the PNG's text metadata records `map_version` and `pixel_to_map`, an affine map from pixel to
  map x, y: (u, v) are integer column and row indices, and `pixel_to_map` maps them to the
  centre of pixel (u, v).

`GET /api/maps/{name}/birdseye` returns the same metadata as JSON, with the version's
`scale_source` and `scale_sigma_rel`.
