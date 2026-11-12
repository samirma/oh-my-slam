# HTTP API — `server.sh`

Part of the [high-level specification](../high_level_spec.md) (§6). Every JSON body is defined in
[api.schema.json](api.schema.json) under the `$defs` name given in brackets, e.g.
[`UpdateRequest`].

```sh
server.sh [--port <n>] [--host <addr>] [--data <folder>] [--keep-versions <n>]
server.sh --status | --stop
```

`--keep-versions` belongs to O2 (map.md §2).

## 1. Service

* **Starting.** `server.sh` binds `--host` (default `0.0.0.0`) on `--port` (default **9091**):
  * it is reachable from the LAN and from Docker containers using host networking;
  * it starts the inference server ([inference.md](inference.md)) if it is not running;
  * once it accepts connections, it writes one line to stderr:
    `server.sh: listening on http://<host>:<port>/`.
* **Data.** Maps live under `--data` (default `~/oh-my-slam-data/`). One service runs per data
  folder.
* **`--status`** prints `GET /api/health` on stdout.
* **`--stop`** stops the service and the inference server it started.
* **Request bodies.**
  * Requests that carry images are `multipart/form-data`. A part named `request` holds the
    JSON body, and every other part is a file, referenced from the JSON by its part name.
  * All other requests are `application/json`.
* **Responses.**
  * Every JSON response carries `schema` (the schema version, e.g. `"oms/2.6"`), except
    `/api/openapi.json`, which is an OpenAPI document.
  * Every response carries per-stage timings in a `Server-Timing` header.
  * No information is carried only in headers.
* **Access.** Requests from the LAN are accepted without authentication: the service is meant
  for a trusted network. Upload caps are recorded in `README.md`.

## 2. Endpoints

Health, timings (§8) and errors (§9) are base: every implementation serves them, in no group.
Each other endpoint belongs to the group named in its section, or in map.md.

| Method and path | Purpose | Body → response |
| --- | --- | --- |
| `GET /api/health` | Service, inference server and lane states | → [`Health`] |
| `GET /api/openapi.json` | This API, generated from its definitions | → OpenAPI 3.1 |
| `POST /api/maps/{name}/updates` | Start an update job (creates the map) | [`UpdateRequest`] → 202 [`Job`] |
| `POST /api/maps/{name}/updates?dry_run=1` | Check frames, write nothing | [`UpdateRequest`] → 200 [`DryRun`] |
| `GET /api/jobs/{id}` | Job state and result | → [`Job`] |
| `DELETE /api/jobs/{id}` | Cancel a job | → [`Job`] |
| `GET /api/maps/{name}` | Map summary, with the version's label vocabulary | → [`MapSummary`] |
| `GET /api/maps/{name}/versions` | Kept versions (O2) | → [`Versions`] |
| `POST /api/maps/{name}/revert` | Commit a copy of a kept version (O2) | [`RevertRequest`] → [`MapSummary`] |
| `GET /api/maps/{name}/frames` | Keyframes and their poses | → [`Frames`] |
| `GET /api/maps/{name}/objects` | Object records | → [`Objects`] |
| `GET /api/maps/{name}/grid` | Navigation grid (O6) | → [`Grid`] |
| `GET /api/maps/{name}/points` | Fused points inside a box (O6) | → [`Points`] |
| `GET /api/maps/{name}/birdseye.png` | Bird's-eye image (O1) | → `image/png` |
| `GET /api/maps/{name}/birdseye` | Its metadata (O1) | → [`Birdseye`] |
| `POST /api/maps/{name}/locate` | Locate frames in the map (O5) | [`LocateRequest`] → [`Locate`] |
| `GET /api/maps/{name}/pose` | Last recorded pose (O5) | → [`Pose`] |
| `DELETE /api/maps/{name}` | Delete the map and its versions | → 204 |
| `POST /api/frames/analyse` | Analyse one frame, no map change | [`AnalyseRequest`] → [`Analysis`] |

Every `GET` under `/api/maps/{name}`, except `/pose` and `/versions`, accepts `?version=<n>` and
names its version in `map_version`. `GET …/pose` takes no `?version`: the recorded pose lives
outside the versions, and its `map_version` is the version it was located in (§6).
`GET …/versions` takes no `?version` either: it lists every kept version (§5), so its response
has no `map_version` (its `latest_version` names the newest). A `?version` sent to either
answers 400 `invalid_request` with `details.field: "version"`.

An unknown route answers HTTP 404 and a known route with another method HTTP 405, both with the
error code `invalid_request` (§9).

## 3. Cameras and frames in requests

* **`cameras`** — a dictionary from `camera_id` to [`Camera`]. A camera already known to the map
  may be omitted; sending it again with other values is refused (`camera_conflict`). A camera
  id, once defined, stays known to the map with its values: a revert does not roll the registry
  back (map.md §2), and camera ids are never freed. A camera sent by an earlier job of the same
  map that is still queued or running counts as known, so a later job may omit it. If that
  earlier job does not commit, the later job fails with `invalid_request`.
* **`frames[]`** — each [`FrameInput`] has:
  * `name`, unique within the map: a name the map or the request already holds is rejected
    `duplicate`, whatever the image;
  * `image`, a part name;
  * `camera_id`;
  * an optional `prior` [`Prior`]. A prior without sigmas takes fixed defaults: `metric`
    0.05 m / 2°, `drifting` 0.20 m / 5°. Clients should always send sigmas. A prior may also
    give `sigma_tilt_deg` (oms/2.4), the 1-σ uncertainty of its pitch and roll, known from
    gravity, independent of `sigma_rot_deg`; when given, pitch and roll are weighted by it and
    yaw by `sigma_rot_deg` (map.md §1);
  * an optional `height_above_floor_m` with `height_sigma_m`, overriding the camera's;
  * an optional `depth`, a part name;
  * an optional `ignore_mask`, a part name, combined with the camera's mask. A pixel in the
    union of the two masks contributes nothing anywhere, on every engine, in every use (map.md §2,
    Ignored pixels).
* **`labels`** (update requests, optional, oms/2.3) — text prompts added to the map's label
  vocabulary (map.md §5): at most 64 entries, each 1–64 characters after trimming. Labels are
  compared as map.md §5 defines (case-folded, `_` as a space). An invalid label refuses the
  request with `invalid_request` (`details.field: "labels"`); it is never a frame status. The
  objects read reports the effect on each record's label as `label_scores` (oms/2.4, map.md §5).
* **Request order** is the order of addition (map.md §3).
* **`analyse`** takes one frame, with its camera inline or by id of a map's camera (`map`
  given).

## 4. Updates (O3, O4, O9, O10)

* **Starting a job.** `POST /api/maps/{name}/updates` creates the map if it does not exist. It
  answers `202` with the [`Job`] in state `queued`, and a `Location: /api/jobs/{id}` header that
  duplicates `job_id`.
* **Watching a job.** `GET /api/jobs/{id}` returns:
  * its state, `progress` and `eta_s`;
  * `base_version`, the version its registration ran against (null for a new map, and while the
    job is queued: a queued job's base is fixed when it starts, §5);
  * `provisional_version` and `result_version`;
  * `latest_version`, the newest version it has committed so far, intermediate `fusing`
    versions included, or null before any;
  * `created_at` and `ended_at`. Once registration has run, the job also carries
  `inputs[]` and `quality`.
  * `timings_s`, the seconds the job spent per stage, a map of stage name to seconds. Stages
    are additive and a consumer ignores those it does not know. `stored_pairs` (oms/2.4) is the
    time spent measuring stored-to-stored pairs for the cycle rule (map.md §1, Measurable
    pair); it is 0 when no such pair was measured, and cached pairs cost nothing.
  * An optional `?wait_s=<s>` (≤ 30) holds the answer until the state changes or the time
    passes. A job already in a terminal state (`committed`, `rejected`, `cancelled`, `failed`)
    answers at once, whatever `wait_s`.
  * Jobs are kept for at least one hour after they end.
* **Inputs.** `inputs[]` has one [`InputStatus`] per frame sent, in request order:

  | Field | Meaning |
  | --- | --- |
  | `name` | The frame's name. |
  | `status` | `registered`, `provisional` (registered, not yet fused: the O3 two-phase state), `kept_at_prior` or `rejected`. |
  | `reason` | Null for `registered` and `provisional`. Set for `rejected` and `kept_at_prior`: `no_overlap`, `few_features`, `plain`, `too_close`, `blur`, `prior_conflict`, `duplicate` or `unusable_image`. |
  | `inliers` | Number of inliers. |
  | `T_map_cam`, `sigma_pos_m`, `sigma_rot_deg` | The pose and its uncertainty. Null for a frame whose `status` is `rejected`. For the frames of a job the gate rejected (`accept_if`, map.md §4) the frames keep their own statuses, and these fields hold the pose and sigmas that would have been committed. |
  | `depth_scale` | Optional (oms/2.4). Present when the frame's monocular depth got its own scale from the floor check (map.md §2): the factor applied to its raw depth, in map metres, in place of the map's. Absent otherwise, also for a frame the floor check did not judge (`detail` `floor_not_judged`). |
  | `prior_residual_m`, `prior_residual_deg` | The residual between the committed pose and the frame's prior as placed (a `drifting` prior through the drift estimate, map.md §1); null without a prior. On a job the gate rejected, against the pose that would have been committed. Their maximum over the registered frames is `quality.max_displacement_m` / `max_displacement_deg`, frames without a prior aside: those are measured against their image-only pose (map.md §1). |
  | `min_distance_m`, `scene_distance_m` | With reason `too_close` only: the distance limit applied, and the frame's median scene depth. |
  | `duplicate_of` | With reason `duplicate` only: the name of the frame it duplicates. |
  | `detail` | For a `kept_at_prior` frame whose images did not confirm it, why: `no_image_pose` (the images give no pose), `loose_image_pose` (the image pose is too loose to confirm the prior) or `conflict_unsupported` (the image pose conflicts with the prior without enough support to reject it). For a `registered` or `provisional` monocular frame the floor check did not judge (oms/2.5, map.md §2): `floor_not_judged` (its own depth scale lay beyond the monocular scale bound: it keeps the map's scale, has no `depth_scale`, is not rejected and keeps its pose, but its content is excluded: no cloud points, occupancy, `height_m` or objects from it, only free-by-rule cells; the client can retake the view). Null otherwise, including a frame kept for a failed check. |

  **Kept at prior.** `kept_at_prior` is a frame with a prior that its images did not confirm.
  `reason` says why. Its prior residuals are 0, measured against the prior as placed (for a
  `drifting` prior, after mapping through the drift estimate). It is stored, but locate never
  uses it as a reference (§6), nor does registration of new frames: a `kept_at_prior` frame of
  an earlier version, or a frame that ends the job kept, is never a registration reference
  (map.md §2, match graph). Occupancy, heights, the
  cloud, object records and scale evidence come only from image-confirmed frames, with one
  exception below: a `kept_at_prior` frame counts only for free-by-rule cells (its camera stand
  and near blind disc, map.md §6) until a later job's images confirm it; its pose then becomes a
  registered one in that version (map.md §2), and the job reports it in
  `quality.reregistered`. A registered or provisional monocular frame with `detail`
  `floor_not_judged` is not a `kept_at_prior` frame, but its content is excluded likewise (map.md
  §2); it counts for free-by-rule cells.
  * **Too-close exception** (every engine, the older mapper included). A frame that is
    `kept_at_prior` only because it is `too_close`, and has a `metric` prior, still contributes
    at its prior pose: its detections, to object records and `detection_votes`; its depth
    points, to the cloud, occupancy and heights. On the older mapper engine its depth reaches
    occupancy and heights but not the cloud, which is that engine's own fused cloud. Depth
    points count only from a depth image, inside its valid range (`README.md`). Without a depth
    image, its detections and `detection_votes` still count at the prior pose, placed like any
    monocular detection. It never contributes scale evidence: scale stays confirmed-only.
  * Every other `kept_at_prior` frame contributes nothing until its images confirm it,
    including a `too_close` frame with a `drifting` prior and a frame kept for any other failed
    check.
  * A client tells the kept frames that carried content by their `reason`, `too_close`, which
    every kept frame carries (the schema requires a reason for `kept_at_prior`).
  * With a `metric` prior, its pose is the prior and its sigmas are the prior's.
  * With a `drifting` prior, its pose is the prior mapped into the map frame through the
    current drift estimate. Its sigmas are drift-grown: the root sum of squares of the prior's
    sigma (or the 0.20 m / 5° default, §3) and the drift estimate's own sigma at that pose,
    which grows with distance from the frames that supported the estimate (map.md §1). This
    holds on every engine.

  **Choosing a status.** Registration is order-independent and runs over the job's match graph
  (map.md §2). An edge needs the registration minimum of inliers (the `README.md` constant,
  currently 15, behind `few_features`, §6) and counts only if the frame's fitted pose passes
  the prior checks. Each connected component of new frames registers as a whole, and the
  checks below apply frame by frame to its fitted poses, iterating to a fixed point. A lone
  frame with a prior and no edge is `kept_at_prior`, `no_overlap`, `detail` `no_image_pose`.
  The outcome does not depend on frame order, and a dry run predicts the same statuses.
  * A hypothesis's support is its inlier count over the best image hypothesis's inlier count;
    it is supported at 0.25 or more (the same definition as locate, §6).
  * **Conflict with the prior.** A hypothesis conflicts with the prior when its distance from
    the prior exceeds 3 × sqrt(σ_prior² + σ_hypothesis²), tested separately for position and
    rotation. σ_hypothesis is the hypothesis's own image-only calibrated sigma, before fusion
    with the prior and with no rival inflation. For a `drifting` prior, σ_prior is the
    drift-grown sigma. The same bound applies to locate (§6).
  * `rejected` with `prior_conflict` only when the images confirm a pose that conflicts with the
    prior, and no supported hypothesis lies within that bound. The images confirm a pose when its
    image-only pose meets the measurable-pair threshold (at least 100 inliers and 10 % overlap, both
    with the same confirmed or registered reference; map.md §1, Measurable pair). A frame that fails
    a looseness test is `kept_at_prior`, `detail` `loose_image_pose`, whatever else it fails;
    otherwise a frame beyond the conflict bound that does not meet the threshold is `kept_at_prior`,
    `detail` `conflict_unsupported`.
  * A frame whose images merely fail to confirm it is `kept_at_prior`, not rejected.
  * In updates, `few_features` is only the checks' verdict: fewer keypoints than the
    registration minimum (§6); one `README.md` constant serves both (currently 15). A frame
    with a prior that passed the checks but was not placed is `kept_at_prior` with
    `no_overlap`, `detail` `no_image_pose` or `loose_image_pose`.
  * A frame with a prior whose image fails a quality check, or that is too close, is
    `kept_at_prior`. Its reason is the first failing check in the order `plain`, `blur`,
    `few_features`, `too_close`, with `min_distance_m` and `scene_distance_m` when it is
    `too_close`. It is `rejected` only when the image is unusable (`unusable_image`: it does not
    decode, or its size is not the camera's). This wins over `prior_conflict`: a frame that
    fails a check, or is too close, and whose image also strongly conflicts with its prior, is
    `kept_at_prior` with that check's reason. Depth agreement alone never confirms a frame with
    a prior that failed a check: it stays `kept_at_prior`.
  * This holds for an origin frame too: a frame of a map's first update whose image fails a
    quality check is `kept_at_prior` and still defines the map frame (map.md §1). An engine that
    cannot tie kept frames to the priors' frame, when no prior-carrying frame registers, may
    reject the first update with `prior_frames_rejected` (map.md §1).
  * `duplicate`: a name duplicate, or a pose duplicate whose image agrees. `duplicate_of` names
    the frame it duplicates. A duplicate is always `rejected`, never `kept_at_prior`.
    * A name duplicate holds a name the map or the request already holds (§3). It is always
      rejected.
    * A pose duplicate lies within 0.02 m and 2° of a stored confirmed keyframe of the same camera,
      or of another frame registered in the same job with the same camera. There is one
      pose-duplicate pass, after the registration fixed point, over all frames registered in the job
      in canonical name order; of a pair, the later-named frame is the duplicate (map.md §2, match
      graph). A stored kept keyframe re-registered in this job counts as a frame of this job for the
      pass. If a rejected duplicate served as a reference, as the drift reference, as scale evidence
      or for a re-registration, the registration fixed point runs once more without it: at most once
      per job (a duplicate found in that run is rejected without a further run); a rejected
      duplicate keeps its first `duplicate_of`; and if the later member is a stored kept keyframe
      the job re-registered, its re-registration is undone and the second run skips it. The pose
      test uses the frame's registered pose: a frame that does not register can be a duplicate only
      by name. A pose duplicate is rejected only when its image also agrees with that keyframe: at
      least 90 % of its features re-matched. No detection is run for it. Otherwise it is
      `registered`, and the later observation wins (map.md §3, C13).
    * Stored `kept_at_prior` keyframes do not count for this test. 0.02 m and 2° are defaults;
      `README.md` may raise them up to, but not reaching, 0.05 m and 5°, so a frame 0.05 m or
      5° away is never a duplicate. A view the map already has, beyond those limits, is
      `registered`.
  * `too_close` is only the camera-to-scene distance limit, with `min_distance_m` and
    `scene_distance_m`. Depth never overrides it.
  * A frame with at least 300 image feature inliers to neighbouring keyframes is never rejected
    for distance alone. Depth correspondences never count toward these 300.
  * A frame kept for a failed check is eligible for re-registration by a later job (map.md §2),
    but only its images can confirm it, never depth. When they do not, it stays
    `kept_at_prior` and is not listed in `quality.reregistered`.

  **Pose sigmas** (`sigma_pos_m`, `sigma_rot_deg`, here and in [`Frames`]) include the
  image-registration noise, calibrated so that the containment targets
  (`update.sigma_containment`, evaluation.md) are met. They are 3-D RMS (§6), and mean the same
  for every engine.

  **Drift correction** (oms/2.5). `quality.drift_correction` reports what the drift estimate did
  to the job's `drifting` priors: `{translation_m, rotation_deg, odo_scale}`, defined in map.md §4.
  A job's inputs' `prior_residual_*` are measured from the prior as placed, so with it a client
  sees how much of a frame's distance from the prior it sent is the estimate and how much the
  images. Null when the job has no `drifting` prior.

* **Gate.** `accept_if` ([`AcceptIf`]) gates the job (map.md §4). The job's `quality` (and a
  dry run's) lists in `rotation_pairs` every measurable pair the rotation check evaluated, with
  its measured, committed and difference angles, so a client can see which pairs drove
  `rotation_inconsistent` (map.md §4). `failed_checks` holds `depth_inconsistent` (oms/2.6) when
  two committed views contradict each other's sensor depth images (map.md §4, Depth
  consistency), with or without priors; a dry run predicts it. `quality.depth_pairs` (oms/2.6)
  lists every pair that limit evaluated, `{a, b, agree_ratio_ab, agree_ratio_ba, fails}`, on jobs
  and dry runs, so on a rejection a client sees which frames contradict and can drop or retake
  them.
* **Dry run.** `?dry_run=1` answers synchronously with a [`DryRun`] and writes nothing. It never
  waits behind fusion: it runs on the interactive lane, or ahead of fusion on a service with a
  single worker (inference.md). Documented limit: a job registers frames that carry depth
  images with the geometric engine, as the dry run does, and may place monocular frames and
  frames the geometric engine leaves unplaced with the MapAnything engine (map.md §2, Match
  graph). For those frames the dry run predicts with geometric registration, so their statuses
  and poses can differ from the update's.
  Per frame it gives:
  * `features`;
  * the predicted link to the map: `map_inliers`, `predicted_status` (`registered`,
    `kept_at_prior` or `rejected`), `predicted_reason`;
  * the predicted links to its neighbours in the request: `neighbour_inliers[]`;
  * the flags `plain` and `too_close`;
  * with `predicted_reason` `duplicate`: `duplicate_of`, as in `inputs[]`;
  * with `predicted_reason` `too_close`: `min_distance_m` and `scene_distance_m`, as in
    `inputs[]`.

  It also gives a predicted `quality`, without fusion (`drift_correction` included, map.md §4),
  and `scale_source` and `scale_sigma_rel`. It accepts `labels` and reports in `labels` the vocabulary it would use
  (map.md §5).

## 5. Versions and revert (O2)

The versions read, revert and `--keep-versions` are O2.

* `GET /api/maps/{name}/versions` lists the kept versions [`VersionInfo`]: `version`,
  `created_at`, `kind` (`update`, `provisional`, `fusing`, `revert`), `job_id`, `frames` and
  `objects`.
* `POST /api/maps/{name}/revert {to_version}` commits a new version equal to `to_version`, and
  answers with the new summary.
  * It is refused with `map_busy` while a job of that map is registering or fusing. While only
    queued jobs exist (none running), it proceeds.
  * A queued job then starts from the reverted version, and its `base_version` reports the
    version it actually ran on.
  * A revert reuses the target's derived reads and keeps the camera registry and the object ids
    (map.md §2), so `revert.latency` (evaluation.md) holds.
* `DELETE /api/maps/{name}` is refused with `map_busy` while a job of that map is queued,
  registering or fusing: it never cancels a job. Unlike a revert, it is refused even while only
  queued jobs exist.
* A job holds its base version, and every read holds the version it answers from, a dry run or
  a locate with `version` included; retention treats held versions as map.md §2 says.

## 6. Locate (O5)

* **Request.** `POST /api/maps/{name}/locate {frames[], record, version?}` registers each frame
  against one version without changing the map.
* **Response.** [`Locate`] gives `map_version`, `scale_source`, `scale_sigma_rel`, and one entry
  per frame:

  | Field | Meaning |
  | --- | --- |
  | `name` | The frame's name. |
  | `status` | `located` or `rejected`. |
  | `reason` | When rejected: `no_overlap`, `few_features`, `plain`, `blur` or `prior_conflict`; otherwise null. |
  | `T_map_cam`, `sigma_pos_m`, `sigma_rot_deg` | The pose and its uncertainty. The sigmas describe the chosen hypothesis only, not the spread of rivals; they include the image-registration noise, calibrated so that `locate.sigma_containment` (evaluation.md) is met. |
  | `inlier_ratio` | The share of inliers. |
  | `ambiguity` | `{pos_m, rot_deg}` of the strongest supported rival, or null; see below. |
  | `rivals` | `[{pos_m, rot_deg, support}]`, every supported rival; empty when none. |
  | `consistent_with_prior` | Null without a prior; always true for a located frame with a prior (see below). |
  | `height_consistent` | With a camera-height scale (see below): whether the located camera height agrees with the expected one within 3 sigma; null otherwise. |

* **With a prior.** The reported pose is the hypothesis nearest the prior among those with
  support at least 0.25, nearest by hypot(dp / sqrt(σp² + σh,p²), dr / sqrt(σr² + σh,r²)): the
  position and rotation distances to the prior over the combined prior and hypothesis sigmas, the
  same normalisation as the conflict bound. Only hypotheses within
  the conflict bound of §4 qualify: 3 × sqrt(σ_prior² + σ_hypothesis²), tested separately for
  position and rotation. When none does, the frame is `rejected` with `prior_conflict`, not
  located with `consistent_with_prior: false`; so a located frame with a prior always has
  `consistent_with_prior: true`. A prior without sigmas takes the fixed defaults of §3
  (`metric` 0.05 m / 2°, `drifting` 0.20 m / 5°). Clients should always send sigmas.
* **Sigma convention.** `sigma_pos_m` and `sigma_rot_deg` are 3-D RMS, for update and locate
  alike: σ² is the trace of the position or rotation covariance.
* **Locate sigmas.** A located pose's sigmas come from the images alone, with trust class
  `none`. The prior only ranks and gates (`prior_conflict`); it never changes the sigmas, so a
  client that fuses the answer with its own prior does not count the prior twice.
  * σ_rel = hypot(scale × covariance σ, floor[depth kind / none]), for position and rotation
    separately: the registration covariance sigma in map metres, and the noise floor below.
  * σ_rel is combined with the reference keyframe's map sigma only while
    `locate.sigma_containment` and `locate.sigma_tightness_bound` (evaluation.md) both hold.
    If the combined sigma is too loose (tightness below the lower bound), σ_rel is reported
    alone.
  * **Camera-height scale.** When the map's `scale_source` is `camera_height`, the reported
    position sigma is hypot(σ_rel, |z_located − z_expected|).
    * z_expected is the floor height plus the locate frame's `height_above_floor_m`, or else
      the camera's registered height in the map.
    * Without any height, `height_consistent` is null and the position sigma is σ_rel.
    * `height_consistent` is false when |z_located − z_expected| > 3 × sqrt(height_sigma² +
      σ_z²), with σ_z = σ_rel / √3 (the sigmas are 3-D RMS). height_sigma is, in this order:
      1. the locate frame's own `height_sigma_m`;
      2. else the median `height_sigma_m` that the answer's reference keyframes carried when
         they were sent (the map frames' `height_sigma_m`);
      3. else, when none of them carries one, the median of those reference keyframes'
         `sigma_pos_m` / √3, from their registered pose sigmas. Only image-confirmed
         (registered) reference keyframes count; kept references are skipped, and if none
         remain step 4 applies;
      4. else the default recorded in `README.md`.
    * Such a hypothesis ranks below any height-consistent rival. `consistent_with_prior` does
      not change.
  * Noise floors, position / rotation. These are defaults, tunable in `README.md`:

    | Depth | `metric` | `drifting` | `none` |
    | --- | --- | --- | --- |
    | with depth | 8 mm / 1.25° | 8 mm / 1.25° | 12 mm / 1.25° |
    | without depth | 0.15 m / 2.1° | 0.15 m / 2.1° | 0.15 m / 2.1° |

    Locate uses the `none` column.
* **Ambiguity.** `ambiguity` is non-null whenever a supported rival hypothesis exists.
  * A rival is a pose hypothesis, other than the reported one, farther than 0.10 m or 5° from
    it.
  * A hypothesis's support is its inlier count over the best hypothesis's inlier count (at most
    1). A rival is supported when its support is at least 0.25, whatever else its inliers
    explain.
  * `pos_m` and `rot_deg` are the distance from the best-supported rival to the reported pose.
  * `rivals` lists every supported rival with its distance and support, strongest first;
    `ambiguity` repeats its first entry. Rivals within 0.10 m and 5° of each other are one
    rival: only the strongest of them is listed.
  * So a supported rival 0.2 m away is never silent, nor is the 90° flip of a symmetric room.
    The client then asks for a second frame.
* **Reasons.**
  * `few_features`: the frame has fewer keypoints than a pose needs inliers. That minimum is an
    implementation constant recorded in `README.md` (currently 15), the registration minimum;
    the `few_features` check uses it, one constant for both. In updates `few_features` is only
    the checks' verdict (§4).
  * `no_overlap`: the frame has enough keypoints but gets no pose; see `nearest_view` below.
* **Out of the mapped area.** A frame rejected with `no_overlap` carries `nearest_view`. It is
  guaranteed when the frame has a prior. Without a prior it is allowed, with `turn_deg` null,
  when the service can estimate one; otherwise null:
  * the keyframe most similar to the frame by retrieval: its `name` and `T_map_cam`. Only
    confirmed keyframes qualify, never a `kept_at_prior` one;
  * `turn_deg`: the signed turn from the frame's prior heading toward that keyframe's view.
    Positive is counter-clockwise. It is null when the frame carries no prior, because the
    frame's own heading in the map is then unknown.

  A robot without odometry can then turn toward mapped structure.
* **Provisional frames.** Locate works against a version whose newest frames are still
  provisional, using their provisional poses.
* **Frames kept at prior.** Locate never uses a `kept_at_prior` frame as a reference.
* **Nothing located** is a `200` response with every frame `rejected`, not an error.
* **Recording.** With `record: true`, the last located frame's pose is stored as the map's
  *recorded pose*, outside the map's versions.
  * `GET /api/maps/{name}/pose` returns it with the time it was recorded, the `map_version` it
    was located in, its sigmas, and that version's `scale_source` and `scale_sigma_rel`.
  * It answers `no_pose` when nothing was recorded.

## 7. Frame analysis (O7)

* **Request.** `POST /api/frames/analyse {image, camera | camera_id + map, T_map_cam?, depth?,
  outputs[], z_band_m?, bearing_step_deg?}` analyses one frame and changes nothing.
* **Outputs.** `outputs` chooses any of:
  * `objects` — [`AnalysisObject`]: as map.md §5, plus `bbox_px`, without the map-only fields
    (`state`, `moved_from_m`, versions, `region`, `supported_by`, `inside_of`), which only
    [`MapObject`] carries;
  * `points` — the frame's 3-D points (`float32` xyz), their object ids (`uint32`, 0 = none) and
    `rgb` (`uint8`), each as a base64 array, thinned to the voxel size given;
  * `range_scan` — per bearing in the horizontal plane: `range_m` and `kind`:
    * `obstacle`: the nearest surface within `z_band_m`;
    * `floor_end`: the farthest floor seen, when no obstacle is;
    * `unseen`.
  * `depth` — a 16-bit PNG part, as the input depth.
* **Object ids.** With `map` and `T_map_cam`, a detection matched to a map object carries that
  object's persistent `id`, in `objects` and in the points' `object_id`. So the client can pick
  one map object's points out of a close view. A detection that matches no map object, and every
  detection when there is no map or no pose, gets a local id from 1 000 000 up. `object_id` 0
  means no object.
* **Range-scan bearings.**
  * With `T_map_cam`: `bearing_deg` is a direction in the map frame, counter-clockwise from map
    +x, and `range_m` is measured from the camera's position projected on the floor.
  * Without it: `bearing_deg` is counter-clockwise from the camera's viewing direction.
* **Frame of the outputs.** With `T_map_cam`, every output is in the map frame and scale-corrected
  to the map. Without it, outputs are in a gravity-aligned camera frame: the camera at the
  origin, z up, yaw 0 along the viewing direction.
* **Scale.** The response carries `scale_source` and `scale_sigma_rel`.

## 8. Expected durations

* **Timings.** `GET /api/health` reports `timings`: the median and p90 seconds of the last 50
  measurements on this machine, as run:
  * per frame: registration, fusion, locate and frame analysis;
  * per request: grid and objects reads.

  Before any measurement, they report the values recorded in `README.md` for this machine,
  marked `source: default`.
* **Job estimates.** A [`Job`] fills `eta_s` from the moment it is queued: the queue ahead plus
  its own predicted registration and fusion.
* **Dry-run estimates.** A [`DryRun`] reports `predicted_s {registration, fusion}` for its frame
  set.

So the client can gate even a run's first update on its deadline.

## 9. Errors

* **Error body.** Every error answers an [`Error`]:
  * `code`, from a fixed list;
  * `message`, which says what to do;
  * `retryable`;
  * `details`.
* **Error list.**

  | `code` | HTTP | Retryable | When |
  | --- | --- | --- | --- |
  | `invalid_request` | 400 | no | Body or parameter fails validation; `details.field` names it |
  | `invalid_request` | 404 | no | Unknown route |
  | `invalid_request` | 405 | no | Method not allowed on a known route |
  | `camera_conflict` | 409 | no | Camera redefined with other values |
  | `map_not_found`, `job_not_found` | 404 | no | Unknown name or id |
  | `version_gone` | 410 | no | Version no longer kept |
  | `no_pose` | 404 | no | No recorded pose |
  | `map_busy` | 409 | yes | Revert during a running job; delete during a queued or running job |
  | `inference_unavailable` | 503 | yes | Inference server down or restarting |
  | `out_of_memory` | 503 | yes | Device memory exhausted; the server recovers |
  | `payload_too_large` | 413 | no | Upload over the cap or the free disk |
  | `internal` | 500 | no | A bug; `details` names the failing stage |

* **Inputs are not errors.** A frame that cannot be used is never an error: it is an input
  status.
