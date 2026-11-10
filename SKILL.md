---
name: oh-my-slam
description: "Short reference of the oh-my-slam HTTP API (schema oms/2.1) for the robot-guide skill's client: update jobs (and dry runs) that build an anchored map from camera frames with pose priors, per-frame status and the quality gate; the map summary, keyframes, objects and support surfaces, navigation grid, points near a place and bird's-eye image of any kept version; locating frames with uncertainty and ambiguity, the recorded pose; health and timings."
---

# oh-my-slam API

Generated from the API definitions (`python -m oh_my_slam.service.skill`); never edit it by hand. The full contract is `GET /api/openapi.json` (OpenAPI 3.1, its schemas those of `specs/api.schema.json`) and `specs/api.md`.

* Service: `server.sh` on port 9091, every route under `/api/`, reachable from the LAN.
* Every JSON body carries `schema` (`oms/2.1`); units are metres, degrees and seconds; `T_a_b` is the 4x4 row-major pose of b in a; the map frame has z up.
* Implemented groups (`GET /api/health` → `capabilities`): O1, O3-jobs, O4, O5, O6, O8, O9, O10. Refuse what is missing instead of calling it.
* Errors answer `{schema, error: {code, message, retryable, details}}`; a frame that cannot be used is an input status, never an error.

## Endpoints

| Route | Purpose | Body | Response | Query |
| --- | --- | --- | --- | --- |
| `GET /api/health` | Service, inference server and lane states |  | 200 [`Health`] |  |
| `GET /api/openapi.json` | This API (OpenAPI 3.1) |  | 200 |  |
| `POST /api/maps/{name}/updates` | Start an update job (creates the map); with dry_run=1, check the frames and write nothing | [`UpdateRequest`] (multipart: part `request` + file parts) | 202 [`Job`], 200 [`DryRun`] | `dry_run` |
| `GET /api/jobs/{id}` | Job state and result |  | 200 [`Job`] | `wait_s` |
| `GET /api/maps/{name}` | Map summary |  | 200 [`MapSummary`] | `version` |
| `GET /api/maps/{name}/frames` | Keyframes and their poses |  | 200 [`Frames`] | `version` |
| `GET /api/maps/{name}/objects` | Object records and support surfaces |  | 200 [`Objects`] | `version` |
| `GET /api/maps/{name}/grid` | Navigation grid |  | 200 [`Grid`] | `version`, `cell_m`, `z_min`, `z_max`, `footprint_radius_m` |
| `GET /api/maps/{name}/points` | Fused points inside a box |  | 200 [`Points`] | `version`, `min`, `max`, `voxel_m`, `max_points` |
| `GET /api/maps/{name}/birdseye.png` | Bird's-eye image: the cloud, the objects' footprints with their ids and the keyframe cameras seen from above; its text metadata records map_version and pixel_to_map |  | 200 `image/png` | `version` |
| `GET /api/maps/{name}/birdseye` | The bird's-eye image's metadata: size and pixel_to_map |  | 200 [`Birdseye`] | `version` |
| `POST /api/maps/{name}/locate` | Locate frames in the map | [`LocateRequest`] (multipart: part `request` + file parts) | 200 [`Locate`] |  |
| `GET /api/maps/{name}/pose` | Last recorded pose |  | 200 [`Pose`] |  |
| `DELETE /api/maps/{name}` | Delete the map and its versions |  | 204 |  |

An update answers `202` with a job (`Location: /api/jobs/{id}`); poll it with `?wait_s=30` until `state` is `committed`, `rejected` or `failed`. `inputs[]` gives each frame's `status`, `reason`, `T_map_cam` and sigmas; `quality` the residuals and the gate.

```sh
curl -s http://HOST:9091/api/maps/home/updates \
  -F 'request={"cameras":{"head":{"fx":525,"fy":525,"cx":320,"cy":240,"width":640,"height":480}},"frames":[{"name":"f0","image":"f0","camera_id":"head","prior":{"trust":"metric","T_map_cam":[[0,0,1,0],[-1,0,0,0],[0,-1,0,0.5],[0,0,0,1]],"sigma_pos_m":0.02,"sigma_rot_deg":1}}]};type=application/json' \
  -F f0=@f0.jpg
curl -s 'http://HOST:9091/api/jobs/JOB_ID?wait_s=30'
```

## Errors

| `code` | HTTP | Retryable |
| --- | --- | --- |
| `invalid_request` | 400 | no |
| `camera_conflict` | 409 | no |
| `map_not_found` | 404 | no |
| `job_not_found` | 404 | no |
| `version_gone` | 410 | no |
| `no_pose` | 404 | no |
| `map_busy` | 409 | yes |
| `inference_unavailable` | 503 | yes |
| `out_of_memory` | 503 | yes |
| `payload_too_large` | 413 | no |
| `internal` | 500 | no |
