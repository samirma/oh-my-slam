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
  * Every JSON response carries `schema` (the schema version, e.g. `"oms/2.2"`), except
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
| `GET /api/maps/{name}` | Map summary | → [`MapSummary`] |
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

Every `GET` under `/api/maps/{name}`, except `/pose`, accepts `?version=<n>` and names its
version in `map_version`. `GET …/pose` takes no `?version`: the recorded pose lives outside the
versions, and its `map_version` is the version it was located in (§6). A `?version` sent to it
answers 400 `invalid_request` with `details.field: "version"`.

An unknown route answers HTTP 404 and a known route with another method HTTP 405, both with the
error code `invalid_request` (§9).

## 3. Cameras and frames in requests

* **`cameras`** — a dictionary from `camera_id` to [`Camera`]. A camera already known to the map
  may be omitted; sending it again with other values is refused (`camera_conflict`). A camera
  sent by an earlier job of the same map that is still queued or running counts as known, so a
  later job may omit it. If that earlier job does not commit, the later job fails with
  `invalid_request`.
* **`frames[]`** — each [`FrameInput`] has:
  * `name`, unique within the map: a name the map or the request already holds is rejected
    `duplicate`, whatever the image;
  * `image`, a part name;
  * `camera_id`;
  * an optional `prior` [`Prior`]. A prior without sigmas takes fixed defaults: `metric`
    0.05 m / 2°, `drifting` 0.20 m / 5°. Clients should always send sigmas;
  * an optional `height_above_floor_m` with `height_sigma_m`, overriding the camera's;
  * an optional `depth`, a part name;
  * an optional `ignore_mask`, a part name, combined with the camera's mask.
* **Request order** is the order of addition (map.md §3).
* **`analyse`** takes one frame, with its camera inline or by id of a map's camera (`map`
  given).

## 4. Updates (O3, O4, O9, O10)

* **Starting a job.** `POST /api/maps/{name}/updates` creates the map if it does not exist. It
  answers `202` with the [`Job`] in state `queued`, and a `Location: /api/jobs/{id}` header that
  duplicates `job_id`.
* **Watching a job.** `GET /api/jobs/{id}` returns:
  * its state, `progress` and `eta_s`;
  * `provisional_version` and `result_version`;
  * `latest_version`, the newest version it has committed so far, intermediate `fusing`
    versions included, or null before any;
  * `created_at` and `ended_at`. Once registration has run, the job also carries
  `inputs[]` and `quality`.
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
  | `T_map_cam`, `sigma_pos_m`, `sigma_rot_deg` | The pose and its uncertainty; null when rejected. |
  | `prior_residual_m`, `prior_residual_deg` | The residual between the committed pose and the frame's prior; null without a prior. |
  | `min_distance_m`, `scene_distance_m` | With reason `too_close` only: the distance limit applied, and the frame's median scene depth. |
  | `duplicate_of` | With reason `duplicate` only: the name of the frame it duplicates. |

  **Kept at prior.** `kept_at_prior` is a frame with a prior that its images did not confirm.
  `reason` says why. Its prior residuals are 0, measured against the prior as placed (for a
  `drifting` prior, after mapping through the drift estimate). It is stored, but locate never
  uses it as a reference (§6), nor does registration of new frames. Occupancy, heights, the
  cloud, object records and scale evidence come only from image-confirmed frames: a `kept_at_prior` frame counts only for free-by-rule cells (its camera
  stand and near blind disc, map.md §6) until a later job's images confirm it; its pose then
  becomes a registered one in that version (map.md §2), and the job reports it in
  `quality.reregistered`.
  * With a `metric` prior, its pose is the prior and its sigmas are the prior's.
  * With a `drifting` prior, its pose is the prior mapped into the map frame through the
    current drift estimate, and its sigmas are the prior's (or the 0.20 m / 5° default, §3)
    plus the drift estimate's uncertainty.

  **Choosing a status.**
  * A hypothesis's support is its inlier count over the best image hypothesis's inlier count;
    it is supported at 0.25 or more (the same definition as locate, §6).
  * `rejected` with `prior_conflict` only when the images confirm a pose farther than 3 × the
    prior's sigmas from the prior (the prior's sigmas alone, not combined with the image's), and
    no supported hypothesis lies within 3 × the prior's sigmas of it.
  * A frame whose images merely fail to confirm it is `kept_at_prior`, not rejected.
  * A frame with a prior whose image fails a quality check, or that is too close, is
    `kept_at_prior`. Its reason is the first failing check in the order `plain`, `blur`,
    `few_features`, `too_close`, with `min_distance_m` and `scene_distance_m` when it is
    `too_close`. It is `rejected` only when the image is unusable (`unusable_image`: it does not
    decode, or its size is not the camera's).
  * This holds for an origin frame too: a frame of a map's first update whose image fails a
    quality check is `kept_at_prior` and still defines the map frame (map.md §1). An engine that
    cannot tie kept frames to the priors' frame, when no prior-carrying frame registers, may
    reject the first update with `prior_frames_rejected` (map.md §1).
  * `duplicate`: an input within 0.02 m and 2° of a stored keyframe of the same camera, or of
    an earlier `registered` frame of the same request and camera, or one whose name the map or
    the request already holds (§3). `duplicate_of` names that frame. The pose test uses the
    frame's registered pose, after registration: a frame that does not register can be a
    duplicate only by name. A duplicate is always `rejected`, never `kept_at_prior`. Stored
    `kept_at_prior` keyframes do not count for this test. 0.02 m and 2° are defaults;
    `README.md` may raise them up to, but not reaching, 0.05 m and 5°, so a frame 0.05 m or 5°
    away is never a duplicate. A view the map already has, beyond those limits, is `registered`.
  * `too_close` is only the camera-to-scene distance limit, with `min_distance_m` and
    `scene_distance_m`.
  * A frame with at least 300 inliers to neighbouring keyframes is never rejected for distance
    alone.

  **Pose sigmas are a lower bound** until O5 calibrates them against replay truth: until then
  `sigma_pos_m` and `sigma_rot_deg`, here and in [`Frames`], come from the registration's noise
  and the anchoring's residual, not from a pose covariance, and the true error can be larger.

* **Gate.** `accept_if` ([`AcceptIf`]) gates the job (map.md §4).
* **Dry run.** `?dry_run=1` answers synchronously with a [`DryRun`] and writes nothing. It never
  waits behind fusion: it runs on the interactive lane, or ahead of fusion on a service with a
  single worker (inference.md). Per frame it gives:
  * `features`;
  * the predicted link to the map: `map_inliers`, `predicted_status` (`registered`,
    `kept_at_prior` or `rejected`), `predicted_reason`;
  * the predicted links to its neighbours in the request: `neighbour_inliers[]`;
  * the flags `plain` and `too_close`;
  * with `predicted_reason` `duplicate`: `duplicate_of`, as in `inputs[]`;
  * with `predicted_reason` `too_close`: `min_distance_m` and `scene_distance_m`, as in
    `inputs[]`.

  It also gives a predicted `quality`, without fusion, and `scale_source` and
  `scale_sigma_rel`.

## 5. Versions and revert (O2)

The versions read, revert and `--keep-versions` are O2.

* `GET /api/maps/{name}/versions` lists the kept versions [`VersionInfo`]: `version`,
  `created_at`, `kind` (`update`, `provisional`, `fusing`, `revert`), `job_id`, `frames` and
  `objects`.
* `POST /api/maps/{name}/revert {to_version}` commits a new version equal to `to_version`, and
  answers with the new summary.
  * It is refused with `map_busy` while a job of that map is registering or fusing.
  * Queued jobs then start from the reverted version.

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
  | `T_map_cam`, `sigma_pos_m`, `sigma_rot_deg` | The pose and its uncertainty. The sigmas describe the chosen hypothesis only, not the spread of rivals. |
  | `inlier_ratio` | The share of inliers. |
  | `ambiguity` | `{pos_m, rot_deg}` of the strongest supported rival, or null; see below. |
  | `rivals` | `[{pos_m, rot_deg, support}]`, every supported rival; empty when none. |
  | `consistent_with_prior` | Null without a prior; always true for a located frame with a prior (see below). |

* **With a prior.** The reported pose is the hypothesis nearest the prior among those with
  support at least 0.25, nearest by hypot(dp / σp, dr / σr): the position and rotation
  distances to the prior over the prior's position and rotation sigmas. When none lies within 3 × the prior's sigmas, the frame is `rejected`
  with `prior_conflict`, not located with `consistent_with_prior: false`; so a located frame
  with a prior always has `consistent_with_prior: true`. A prior without sigmas takes the fixed
  defaults of §3 (`metric` 0.05 m / 2°, `drifting` 0.20 m / 5°). Clients should always send
  sigmas.
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
    implementation constant recorded in `README.md` (currently 15).
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
  | `map_busy` | 409 | yes | Revert or delete during a running job |
  | `inference_unavailable` | 503 | yes | Inference server down or restarting |
  | `out_of_memory` | 503 | yes | Device memory exhausted; the server recovers |
  | `payload_too_large` | 413 | no | Upload over the cap or the free disk |
  | `internal` | 500 | no | A bug; `details` names the failing stage |

* **Inputs are not errors.** A frame that cannot be used is never an error: it is an input
  status.
