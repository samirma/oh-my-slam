# HTTP API — `server.sh`

Part of the [high-level specification](../high_level_spec.md) (§6). Every JSON body is defined in
[api.schema.json](api.schema.json) under the `$defs` name given in brackets, e.g.
[`UpdateRequest`].

```sh
server.sh [--port <n>] [--host <addr>] [--data <folder>] [--keep-versions <n>]
server.sh --status | --stop
```

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
  * Every JSON response carries `schema` (the schema version, e.g. `"oms/2.1"`).
  * Every response carries per-stage timings in a `Server-Timing` header.
  * No information is carried only in headers.
* **Access.** Requests from the LAN are accepted without authentication: the service is meant
  for a trusted network. Upload caps are recorded in `README.md`.

## 2. Endpoints

| Method and path | Purpose | Body → response |
| --- | --- | --- |
| `GET /api/health` | Service, inference server and lane states | → [`Health`] |
| `GET /api/openapi.json` | This API, generated from its definitions | → OpenAPI 3.1 |
| `POST /api/maps/{name}/updates` | Start an update job (creates the map) | [`UpdateRequest`] → 202 [`Job`] |
| `POST /api/maps/{name}/updates?dry_run=1` | Check frames, write nothing | [`UpdateRequest`] → 200 [`DryRun`] |
| `GET /api/jobs/{id}` | Job state and result | → [`Job`] |
| `DELETE /api/jobs/{id}` | Cancel a job | → [`Job`] |
| `GET /api/maps/{name}` | Map summary | → [`MapSummary`] |
| `GET /api/maps/{name}/versions` | Kept versions | → [`Versions`] |
| `POST /api/maps/{name}/revert` | Commit a copy of a kept version | [`RevertRequest`] → [`MapSummary`] |
| `GET /api/maps/{name}/frames` | Keyframes and their poses | → [`Frames`] |
| `GET /api/maps/{name}/objects` | Object records | → [`Objects`] |
| `GET /api/maps/{name}/grid` | Navigation grid | → [`Grid`] |
| `GET /api/maps/{name}/points` | Fused points inside a box | → [`Points`] |
| `GET /api/maps/{name}/birdseye.png` | Bird's-eye image | → `image/png` |
| `GET /api/maps/{name}/birdseye` | Its metadata | → [`Birdseye`] |
| `POST /api/maps/{name}/locate` | Locate frames in the map | [`LocateRequest`] → [`Locate`] |
| `GET /api/maps/{name}/pose` | Last recorded pose | → [`Pose`] |
| `DELETE /api/maps/{name}` | Delete the map and its versions | → 204 |
| `POST /api/frames/analyse` | Analyse one frame, no map change | [`AnalyseRequest`] → [`Analysis`] |

Every `GET` under `/api/maps/{name}` accepts `?version=<n>` and names its version in
`map_version`.

## 3. Cameras and frames in requests

* **`cameras`** — a dictionary from `camera_id` to [`Camera`]. A camera already known to the map
  may be omitted; sending it again with other values is refused (`camera_conflict`).
* **`frames[]`** — each [`FrameInput`] has:
  * `name`, unique in the request;
  * `image`, a part name;
  * `camera_id`;
  * an optional `prior` [`Prior`];
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
    passes.
  * Jobs are kept for at least one hour after they end.
* **Inputs.** `inputs[]` has one [`InputStatus`] per frame sent, in request order:

  | Field | Meaning |
  | --- | --- |
  | `name` | The frame's name. |
  | `status` | `registered`, `provisional` (registered, not yet fused) or `rejected`. |
  | `reason` | When rejected: `no_overlap`, `few_features`, `plain`, `too_close`, `blur`, `prior_conflict` or `duplicate`; otherwise null. |
  | `inliers` | Number of inliers. |
  | `T_map_cam`, `sigma_pos_m`, `sigma_rot_deg` | The pose and its uncertainty; null when rejected. |
  | `prior_residual_m`, `prior_residual_deg` | The residual to its prior; null without a prior. |

* **Gate.** `accept_if` ([`AcceptIf`]) gates the job (map.md §4).
* **Dry run.** `?dry_run=1` answers synchronously with a [`DryRun`] and writes nothing. It runs
  on the interactive lane. Per frame it gives:
  * `features`;
  * the predicted link to the map: `map_inliers`, `predicted_status`, `predicted_reason`;
  * the predicted links to its neighbours in the request: `neighbour_inliers[]`;
  * the flags `plain` and `too_close`.

  It also gives a predicted `quality`, without fusion.

## 5. Versions and revert (O2)

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
  | `reason` | When rejected: `no_overlap`, `few_features`, `plain`, `too_close` or `blur`. |
  | `T_map_cam`, `sigma_pos_m`, `sigma_rot_deg` | The pose and its uncertainty. |
  | `inlier_ratio` | The share of inliers. |
  | `ambiguity` | `{pos_m, rot_deg}` or null; see below. |
  | `consistent_with_prior` | Bool, or null without a prior. |

* **Ambiguity.** `ambiguity` is non-null only when a rival hypothesis has comparable support.
  * The rival is the best pose hypothesis, other than the reported one, that is farther than
    0.10 m or 5° from it.
  * Its support is comparable when its inlier count is at least `ambiguity_support_ratio`
    (default 0.6, recorded in `README.md`) of the reported pose's.
  * `pos_m` and `rot_deg` are the distance from the rival pose to the reported one.
  * So a non-null `ambiguity` always means a genuinely different pose with comparable
    support, for example the 90° flip of a symmetric room. The client then asks for a second
    frame.
* **Out of the mapped area.** A frame rejected with `no_overlap` carries `nearest_view` when the
  service can estimate one, else null:
  * the keyframe most similar to the frame by retrieval: its `name` and `T_map_cam`;
  * `turn_deg`: the signed turn from the frame's guessed heading (its prior when given, else the
    nearest view's) toward that keyframe's view. Positive is counter-clockwise.

  A robot without odometry can then turn toward mapped structure.
* **Provisional frames.** Locate works against a version whose newest frames are still
  provisional, using their provisional poses.
* **Nothing located** is a `200` response with every frame `rejected`, not an error.
* **Recording.** With `record: true`, the last located frame's pose is stored as the map's
  *recorded pose*, outside the map's versions.
  * `GET /api/maps/{name}/pose` returns it with the time it was recorded, its `map_version`
    and its sigmas.
  * It answers `no_pose` when nothing was recorded.

## 7. Frame analysis (O7)

* **Request.** `POST /api/frames/analyse {image, camera | camera_id + map, T_map_cam?, depth?,
  outputs[], z_band_m?, bearing_step_deg?}` analyses one frame and changes nothing.
* **Outputs.** `outputs` chooses any of:
  * `objects` — as map.md §5, plus `bbox_px`, without the map-only fields (`state`,
    `moved_from_m`, versions);
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
