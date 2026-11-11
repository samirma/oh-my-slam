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
    committed and no map created. A frame `kept_at_prior` is not rejected. An engine that cannot
    tie kept frames to the priors' frame, on a first update where no prior-carrying frame
    registers, may reject the update the same way (`prior_frames_rejected`).
* **Two-view parallax.** The one parallax measure of this specification. For a pair of frames,
  it is the median, over their epipolar inliers, of the rotation-only residual divided by
  max(sin of the angle between the ray and the baseline, 0.3): the sine is floored at 0.3. It is
  measured in pixels of the image as sent. Below 1 px the pair is
  translation-unconstrained: one standpoint. Translation in front of a planar scene can read as
  a rotation: an accepted degenerate case, since priors are expected whenever odometry exists.
* **Overlap.** The one overlap measure of this specification is the 8 × 8 inlier-cell
  coverage of a reference image: the share of the reference image's 64 cells that hold at
  least one matched inlier. A check that names a reference uses that image; for a pair with no
  named reference, the overlap is the smaller of the two coverages, each image taken as the
  reference.
* **Without priors:**
  * the origin is the optical centre of the first registered frame, in canonical name order
    (§2, match graph);
  * +x is that camera's viewing direction projected onto the horizontal plane;
  * +z is up, estimated from gravity and the floor plane;
  * frames whose pairwise two-view parallax (above) is below 1 px form one standpoint group,
    registered about a common centre. The group fixes only that centre: each frame keeps its
    own image-only rotation, which the group never overrides. The common centre is the
    overlapped stored keyframe with the most pair inliers (ties: the lowest keyframe id), or,
    when they overlap none, the mean of their image-only centres; its position sigma is their
    spread;
  * **Measurable pair.** A pair of frames is measurable only with at least 100 epipolar
    inliers. Fewer counts as no measurable pair, which passes every pair limit (§4);
  * a rotation-only hypothesis takes the standpoint's position sigma, and the position test
    (api.md §4) uses it;
  * the committed relative rotations of overlapping neighbours must agree with their
    image-measured relative rotations: a hard limit (§4, `rotation_inconsistent`). With priors
    this limit does not apply; the prior-rotation correction below covers it;
  * sigmas stay honest: the containment targets (evaluation.md) apply.
* **Every engine.** Calibrated sigmas and the `rotation_inconsistent` limit (§4) apply to every
  engine, the older mapper included: every engine's sigmas mean the same thing. The older
  mapper is exempt from standpoint grouping only.
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
  movement counts here and in `previous_residuals`, except the movement of their confirmation
  when a job re-registers them (§2), which is reported only in `quality.reregistered`.
* **Residuals against the previous version.** `quality.previous_residuals` covers the earlier
  keyframes only: its `count` is at most the previous version's keyframe count. A stored
  keyframe that moves beyond tolerance shows in `previous_residuals.max_m` / `max_deg` and is
  gated by `max_residual_m` (§4).
* **New-frame displacement.** How far the service moves the update's new frames from what was
  sent is in `quality.max_displacement_m` and `max_displacement_deg` (§4), not in
  `gauge_shift`. A client setting `max_displacement_*` asks exactly that.
  * A frame with a prior: its committed pose against the prior as sent (a `drifting` prior as
    placed through the drift estimate).
  * A frame without a prior: its committed pose against its image-only pose (the pose its
    images alone give against the map). Such a frame whose image-only pose is unconstrained in
    translation (two-view parallax, above, below 1 px against every reference it registers to)
    counts only its rotation.
  * Frames `kept_at_prior` or `rejected` are left out.
  * `prior_conflict` is unchanged (api.md §4).

  So the gates on `gauge_shift` and `max_residual_m` measure stored-map movement, and the gates
  on `max_displacement_m` / `max_displacement_deg` measure new-frame displacement. Prior
  corrections show in `prior_disagreement_deg` and `prior_residual_*` only.
* **Priors against images.** A prior is weighed by its sigmas against the relative pose the
  images give between overlapping frames. Where two frames overlap (§1, Overlap) by at least
  50 % and their image relative rotation disagrees with their priors' relative rotation by more
  than 3 × hypot(σ_rot,a, σ_rot,b) of the two frames'
  prior rotation sigmas (`metric` and `drifting` priors alike), the rotation is corrected from
  the images. `quality` always reports `prior_disagreement_deg`: the largest such disagreement
  over the update's frames, 0 when none.
* **Drift estimate.** The drift estimate that places `drifting` priors is robust:
  * it is fitted over image-confirmed frames with `drifting` priors;
  * a frame whose residual exceeds 3 × sqrt(σ_prior² + σ_hyp²) is left out of the fit and is
    rejected with `prior_conflict`;
  * a first estimate, when the map has none yet, needs the support of min(3, n) frames, n the
    job's `drifting`-prior frames, and of at least 50 % of them;
  * a change of the estimate between versions larger than 3 × its own sigma needs the support
    of at least 3 frames and of at least 50 % of the job's `drifting`-prior frames. Without it
    the change is refused, the previous estimate stays, and the unsupported frames are
    rejected with `prior_conflict`.
* **Static image content.** Features whose image position stays within 2 px across at least 3
  frames of a job whose relative rotation exceeds 5° are excluded from matching. A job without
  such rotation keeps all its features.
* **Scale.** The scale comes from the first available of:
  1. `metric` priors (`scale_source: priors`);
  2. depth images (`depth`);
  3. camera heights above the floor, per frame or per camera (`camera_height`);
  4. monocular depth (`monocular`).

  `scale_sigma_rel` is estimated from the residuals of that source, from image-confirmed frames
  only (§2). It is one sigma: the relative error of any horizontal distance, in every
  direction, stays within 3 × `scale_sigma_rel`. A later update may refine the scale:
  `quality.scale_ratio_vs_prev` reports the change.

## 2. Versions (O2)

* **Versions.** Every commit creates version *n + 1*: integers from 1, never reused, immutable
  once written. A version holds:
  * the frames and their poses, each pose marked `provisional` (registered, not yet fused),
    `fused`, or `kept_at_prior` (the prior, unconfirmed by the images);
  * the cloud and its free-space fusion. Occupancy, heights (the `height_m` layer and object
    heights), the cloud, object records and all scale evidence (depth, camera height) come only
    from image-confirmed frames; a `kept_at_prior` frame counts only for free-by-rule cells
    (§6) until it is registered. Exception: a frame kept only for `too_close` with a `metric`
    prior contributes its detections and its in-range depth at the prior pose (api.md §4), but
    never scale evidence.
  * **Height gate.** When frames carry `height_above_floor_m` and `height_sigma_m`, a frame
    whose committed camera height disagrees with it by more than
    3 × sqrt(height_sigma² + σ_z²) (σ_z = `sigma_pos_m` / √3, its committed position sigma
    being 3-D RMS) contributes
    nothing to occupancy, heights or objects. Its registration status is unchanged, and
    `quality.height_inconsistent` lists it by name.
  * Only image-confirmed keyframes are registration references for new
    frames; `kept_at_prior` frames of earlier versions are not, nor are frames that end the job
    kept. Within a job, a job frame counts as an edge once it is registered (match graph,
    below).
  * **Re-registration.** When a job registers new frames that overlap a stored `kept_at_prior`
    frame (the overlap of §1 with the kept frame as the reference: at least 10 % of the kept
    frame's image covered by a new registered frame of the job), the service re-runs that
    stored frame's registration against the confirmed map. A frame kept for a failed check is
    eligible too. Only its images can confirm it, never depth; when they do not, it stays
    `kept_at_prior` and is not listed in `quality.reregistered`. If its images confirm it, its
    pose leaves `kept_at_prior` in that version (it becomes a registered pose, `provisional`
    then `fused`), and from then on it feeds occupancy, heights, the cloud and objects. The job
    lists each such frame in `quality.reregistered` (§4); the movement of that confirmation is excluded from `previous_residuals` and
    `gauge_shift`. A dry run predicts `reregistered` with the same test, without committing;
    the prediction is approximate. An engine that cannot re-register (the older mapper) reports
    `reregistered` as an empty list; this is allowed. Re-registration never rejects or merges
    keyframes, even within the duplicate limits. Nothing re-registers a frame outside a job:
    there is no automatic background refinement;
  * **Match graph.** A job's registration is order-independent and runs over the job's match
    graph:
    1. An edge joins two frames of the job, or a new frame and a confirmed stored keyframe.
       Each edge needs at least the registration minimum of inliers: the implementation
       constant in `README.md` (currently 15) behind `few_features` (api.md §6). An edge counts
       only if the frame's fitted pose then passes the prior checks (step 3). The 100-inlier
       measurable-pair threshold (§1) stays for the rotation check and the other pair
       limits.
    2. Each connected component of new frames registers as a whole. Relative poses come from
       the images. The component is placed by its matches to confirmed keyframes, by the
       priors of all its frames together, or both. A component that reaches no confirmed
       keyframe is seeded at one frame's prior. Its frames' reported sigmas cover the seed
       prior's sigma combined with each frame's fusion, and are never tighter than the seed
       prior's.
    3. The prior checks (`prior_conflict` with the combined bound, and the looseness tests,
       api.md §4) apply frame by frame to the component's fitted poses. A failing frame is
       `kept_at_prior`. The rest of the component stays registered if it is still connected
       and placed without that frame. Removals iterate to a fixed point, re-checking
       connectivity after each removal, and the final statuses do not depend on the order in
       which frames are removed.
    4. A frame is kept with `no_overlap` only if its component cannot be placed at all. A lone
       frame, with no edge at the registration minimum, cannot be placed: with a prior it is
       `kept_at_prior`, reason `no_overlap`, `detail` `no_image_pose`.
    5. `kept_at_prior` frames of earlier versions are not references, and a job frame counts
       as an edge only once it is registered. Earlier kept frames that match a registered
       component with at least the registration minimum of inliers go through re-registration
       (above, images only), so the same job can confirm them.
    6. The outcome does not depend on the order of frames in the request. Where any order is
       still needed (the origin frame, the seed frame, a duplicate's earlier frame), frames are
       taken in canonical name order. A dry run applies the same procedure and predicts the
       same statuses, except on maps built by the MapAnything engine, where it predicts with
       geometric registration and its statuses can differ (api.md §4);
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
  job or read is using. Kept versions share their unchanged content, so a version costs only
  what changed.

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
  * `max_displacement_m` and `max_displacement_deg`: the largest translation and rotation, over
    the update's new frames, between the committed pose and the prior as sent (frames with a
    prior) or the image-only pose (frames without), §1; null when there is no such frame. A
    frame without a prior that is unconstrained in translation contributes to
    `max_displacement_deg` only;
  * `prior_disagreement_deg` (§1), 0 when no image relative rotation disagrees with the priors;
  * `height_inconsistent`: the names of the frames the height gate (§2) excluded from content;
    empty when none;
  * `reregistered`: one `{name, displacement_m, displacement_deg}` per stored `kept_at_prior`
    frame this job's images confirmed (§2): how far confirmation moved the frame from its
    prior-placed pose. Empty when there are none; a dry run predicts it. This movement is not
    in `previous_residuals` or `gauge_shift`;
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
    keyframe that moved fails the update. New-frame displacement is gated only by
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
    that would make the map inconsistent. An update fails the hard limit on content only when
    it registers nothing and keeps nothing: an update whose frames are all `kept_at_prior` is
    accepted and stores them.
  * **Rotation consistency** (updates without priors only): a hard limit rejects the update,
    `failed_checks` holding `rotation_inconsistent`, when the committed relative rotations
    between overlapping neighbours disagree with their image-measured relative rotations by
    more than 3° in median over the pairs, or by more than 10° on any single pair.
    * Overlapping neighbours: pairs whose overlap (§1, Overlap) is at least 10 %.
    * Without priors: the update's frames carry no priors.
    * Image-measured: each pair's own two-view relative rotation, independent of the joint
      solve.
  * Pair limits (rotation consistency and any other limit on pairs of frames) are measured on
    registered frames only, over measurable pairs (at least 100 epipolar inliers, §1). With no
    measurable pair, every pair limit passes.

## 5. Objects (O8)

Each object record [`MapObject`] has the fields below; frame analysis returns the subset
[`AnalysisObject`] (api.md §7). Every field is present in an objects read.

| Field | Meaning |
| --- | --- |
| `id` | Positive integer, persistent for the map's lifetime. |
| `label`, `detected_as[]`, `score` | Label chosen, labels it was detected as, and confidence. |
| `detection_votes[]` | `{label, share}`, highest share first: `share` is the number of frames that gave that label over the number of image-confirmed frames whose view frustum contains the object's box centre, detected or not, plus every confirmed frame that detected the object even with the box centre outside its frustum (`kept_at_prior` frames excluded, except a frame kept only for `too_close` with a `metric` prior, which counts as a confirmed frame here, api.md §4), so shares sum to at most 1 and may sum to less. For a map built by the older mapper, shares come from that mapper's per-label vote counts over its observing frames. Map objects only; frame analysis does not carry it. |
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
* **Floor band.** A surface within max(0.06 m, 3 × `floor_sigma_m`) of `floor_z_m`, above or
  below it, is floor (§6): it never makes a floor-standing object. The minimum object height
  applies above the floor estimate only: an object resting on any other support surface keeps its
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
(camera stand and near blind disc) until a later job's images confirm it (§2). Occupancy, the
`height_m` layer, the cloud and object records come only from image-confirmed frames, except
the in-range depth and detections of a frame kept only for `too_close` with a `metric` prior
(api.md §4).

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
    height defines (about 0, §1), refined by the observed large horizontal plane nearest it,
    accepted when within 3 × that plane's own point spread; `floor_sigma_m` is that spread. The
    height used is the first registered frame's `height_above_floor_m`;
  * with neither priors nor a camera height, it is the lowest large horizontal plane seen below
    the cameras.

  Other planes, such as the room floor seen past a worktop's edge, are support surfaces
  (`surfaces[]`), not the floor.

  **Floor band.** A surface within max(0.06 m, 3 × `floor_sigma_m`) of `floor_z_m`, on either
  side of it, is floor: it never makes an occupied cell or defines a passage, whatever `z_min`
  is.
* Optional `footprint_radius_m`: the robot's radius.
* `height_cap_m`: the ceiling of the `height_m` layer, relative to `floor_z_m`. Default 2.0 m,
  or the band's `z_max` when the request gives one. Echoed in the header.

**Header.**
* `map_version`, `origin_m` (the corner of cell [0, 0]), `cell_m`, `shape [rows (y), cols (x)]`,
  `z_band_m`, `floor_z_m`, `floor_sigma_m` and `height_cap_m`. `z_band_m` and `height_cap_m`
  are relative to `floor_z_m`, as the request's `z_min` and `z_max` are. `floor_z_m` is the
  floor's absolute map-frame z, and `floor_sigma_m` the spread (standard deviation) of the floor
  points' heights, not the uncertainty of their mean;
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
