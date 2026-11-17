# oh-my-slam — High-Level Specification

## 1. Purpose and scope

oh-my-slam exists to serve one client: the **robot-guide skill**. The skill's goal states that
"at any point in time, the working map can be accessible", describing the most up-to-date map of
the scene, while a robot (a wheeled base, a humanoid or an arm, real or simulated) moves through
that scene. oh-my-slam builds and keeps that map from the robot's camera frames, and answers
the questions the skill's scripts ask about it.

**Scope test.** A feature belongs here only if a row of the client contract (§2) needs it.
Anything else is removed (§13). oh-my-slam knows nothing about any particular robot, scene or
task:
* robot knowledge reaches it only as camera pose priors and camera parameters (§4);
* footprints and height bands are parameters of the requests that need them.

The skill's scripts do all numeric work against this API, so every number the skill needs is in
a response body, with its unit, frame and uncertainty.

**The measure of "serves the skill"** is what each kind of robot can do with no geometry done
by the language model:
* **Base:** a wheeled base, or a walking robot, gets from locate, the grid and its passages
  everything it needs to plan a route and verify that it followed it.
* **Arm:** an arm gets from objects, support surfaces, frame analysis and points everything it
  needs to place its gripper and verify the outcome.

**Pruning.** A response field that no skill script reads is a candidate for removal. The skill
publishes the fields its client reads, and the rest are pruned in a later minor version.

This document states requirements. Detailed decisions are recorded in `README.md` and must not
contradict it. Examples of such decisions: defaults, thresholds, encodings, palette.

| File | Contents |
| --- | --- |
| [specs/api.md](specs/api.md) | HTTP API: endpoints, requests, responses, errors |
| [specs/map.md](specs/map.md) | Map model: frame, versions, update jobs, quality gate, objects, grid |
| [specs/inference.md](specs/inference.md) | Inference server and its lanes |
| [specs/evaluation.md](specs/evaluation.md) | Replay evaluator and targets |
| [specs/api.schema.json](specs/api.schema.json) | JSON Schema (2020-12) of every request and response body |
| `SKILL.md` | Short API reference for the robot-guide client, generated from the API definitions |

## 2. Client contract

Each row is something the skill needs. Each row's test is in §11.

| # | The skill needs | Provided by |
| --- | --- | --- |
| C1 | One map frame that never jumps: positions it stored before an update are still valid after it. | Anchored map frame (map.md §1) |
| C2 | The map's metric scale and how well it is known, so map distances can be driven. | `scale_source`, `scale_sigma_rel` on every metric output |
| C3 | A readable, current map at any moment, including while an update runs. | Versioned snapshots; two-phase update jobs (map.md §2–§3) |
| C4 | To know, per frame sent, whether it was used, and why not. | Per-input status in the update result (api.md §4) |
| C5 | To undo an update that made the map worse, in seconds. | Revert to a kept version (api.md §5) |
| C6 | To reject an update automatically when it would damage the map. | Quality report and `accept_if` gate (map.md §4) |
| C7 | To check frames before sending them as an update. | Dry run (api.md §4) |
| C8 | Where the camera is, with an uncertainty and a warning when the answer is ambiguous. | Locate (api.md §6) |
| C9 | Where the robot can go: free, occupied and unknown floor, the frontiers, rooms and doorways. | Navigation grid (map.md §6) |
| C10 | The objects of the scene: what, where, how big, their real colour, whether they moved. | Object records (map.md §5) |
| C11 | What the camera sees now, without changing the map: objects, points, and a range scan. | Frame analysis (api.md §7) |
| C12 | Locate and frame analysis answered within seconds even while an update runs. | Inference lanes (inference.md) |
| C13 | A map that shows what is there now: a later observation wins over an earlier one. | Update semantics (map.md §3) |
| C14 | To find objects it names by text in what the camera sees now: the best match per name, with its 3-D centre (also in the camera's own frame), extent and image mask. | Frame analysis labels and matches (api.md §7) |

## 3. Frames and units

* **Units.** Metres, degrees, seconds, everywhere. Angles in responses are degrees.
* **Camera frame.** Optical axes: x right, y down, z forward.
* **Map frame.** Gravity-aligned and right-handed: x and y horizontal, z up. Yaw is measured
  counter-clockwise from +x about +z. How the frame is fixed is in map.md §1.
* **Poses.** `T_a_b` is the 4×4 homogeneous pose of frame b in frame a, written as nested
  row-major arrays `M[row][col]`. `T_map_cam` is the pose of the camera's *optical* frame in the
  map frame, so it maps optical-frame points to map-frame points.
* **Body frame** (oms/2.15). Fixed to the robot's body: z up (along gravity when the body stands
  level), x forward, y left. `T_body_cam` is the pose of the camera's optical frame in it.
* **Scale.** Every response that carries map-frame metres also carries:
  * `scale_source`: one of `priors`, `depth`, `camera_height`, `monocular`, and `baseline`
    (frame analysis with motion stereo, oms/2.11);
  * `scale_sigma_rel`: the relative one-sigma uncertainty of the map's scale.

## 4. Inputs

* **Frames.** Colour images: JPEG or PNG, 8-bit RGB. Each frame names its camera and may carry:
  * a pose prior;
  * a depth image: 16-bit PNG, depth along the optical axis, scale given per camera, 0 = no
    value.
* **Cameras.** Each camera is described once per map and can be referenced by id afterwards:
  * intrinsics `fx, fy, cx, cy` in pixels, and image size;
  * distortion: the OpenCV model, or none;
  * optional ignore mask, for pixels showing the robot itself. A frame may add its own mask,
    for example the arms in a head camera or a held object in a wrist camera. The two masks are
    combined, and a pixel in their union contributes nothing anywhere, on every engine, in every
    use: no keypoints or matches, no depth points, no occupancy, free-space or floor evidence,
    no detections or votes, no cloud points, and none to the frame quality checks, the
    `too_close` scene distance, scale evidence or the camera-height check (map.md §2);
  * optional height of the optical centre above the floor, with a sigma: the scale anchor for
    robots without metric priors. A frame may carry its own height, with a sigma, which
    overrides the camera's: a camera on a pan-tilt head or a wrist moves with the robot's joints.
* **Pose priors.** `T_map_cam` with a trust level:
  * `metric` — an absolute pose in the map frame, with position and rotation sigmas. For
    example, odometry with a known start, or kinematics of a fixed arm.
  * `drifting` — reliable relative motion between frames of one request, with drift over time.
  * `none` — no prior. This is the case for camera-only robots.

* **Body links** (oms/2.15). A frame may carry its camera's mount on the robot's body
  (`T_body_cam`, from the robot's model and its joints, per frame or fixed per camera) and the
  body's motion since an earlier named frame (yaw and translation, each with its sigma and
  source), with the image and joint-reading stamps and whether the body stood still. Composed,
  they are a metric relative camera prior: the camera's lever arm about the turn axis gives a
  turn in place a known baseline (api.md §4, Body links; map.md §1).

  Odometry, depth and joint kinematics reach oh-my-slam only through priors, body links and
  depth: an IMU's yaw, odometry and scan matching enter only as a body motion between frames.
  There is no raw IMU, stereo or LiDAR input.
* There is no video input. The client samples its own frames.

## 5. Map model and lifecycle

A map is created by its first committed update. Every commit after that produces an immutable,
integer **version**.
* Reads are atomic: each answer comes from one version, which it names. A read may ask for a
  past version that is still kept.
* An update runs as an asynchronous **job** in two phases:
  * **registration** publishes the new frames with provisional poses within seconds;
  * **fusion** then fuses their depth and objects.

  Each phase that succeeds commits a version.
* An update can be checked first (dry run), gated (`accept_if`) and undone (revert).
* A map is deleted only by its delete request.

The details are in [specs/map.md](specs/map.md).

## 6. HTTP API

One long-lived service, `server.sh`, serves the whole API under `/api/`. [specs/api.md](specs/api.md)
defines it, and `/api/openapi.json` describes it.
* It listens on a fixed default port, **9091**.
* It is reachable from the LAN and from Docker containers using host networking.
* It owns no model: the inference server (§8) does.

## 7. Output schemas

Every JSON body follows [specs/api.schema.json](specs/api.schema.json), a flat, documented JSON
Schema (draft 2020-12). Each response names its schema version in `schema`. A change that
removes or renames a field, or changes its meaning, raises the major version. The skill's client
is written against this schema. OpenLABEL is no longer produced.

**Narrowings accepted in oms/2.2.** Four changes narrow what a response may hold, yet stay in
minor 2.2, because the API has a single client, updated in step with it:
1. passage `kind`: `narrow` is deprecated and no longer produced (a separate `narrow` flag
   replaces it), and `door` requires a found leaf;
2. locate's `reason` no longer includes `too_close`;
3. health `lanes` may be a single `worker` lane instead of `interactive` and `bulk`;
4. an input status's `reason` is null for `registered` and `provisional`, and set for
   `rejected` and `kept_at_prior`.

**oms/2.3** is additive: the update request's optional `labels` (a map's label vocabulary,
map.md §5), reported on the map summary and the dry run.

**oms/2.4** is additive: the object record's optional `label_scores` (up to three label
candidates with their scores, map.md §5), and the label choice rule that prefers a client label
over a default one whose score is higher by no more than a margin (map.md §5). It also adds,
all optional or empty by default: the quality report's `floor_inconsistent` (frames whose own
depth puts the floor away from their sent height, map.md §2); the input status's `depth_scale`
(the per-frame depth scale applied to a monocular frame, map.md §2); and the prior's
`sigma_tilt_deg` (the uncertainty of pitch and roll apart from yaw, map.md §1). The sent camera
height also becomes a z measurement in registration (map.md §2), and `floor_sigma_m` a robust
standard deviation with a floor band tested on surface height (map.md §6).
It also adds the job's `timings_s` stage `stored_pairs`, the seconds spent measuring
stored-to-stored pairs from the stored keyframes' own images when the cycle rule needs a chain
for a pair that involves a stored keyframe (map.md §1).

**oms/2.5** is additive: the quality report's `drift_correction` (the drift estimate's correction
at the job's frames, map.md §4), and the explicit meaning of an input status's `T_map_cam` and
`prior_residual_*` (from the prior as placed; on a job the gate rejected, the pose that would
have been committed, api.md §4), and the input status's `detail` value `floor_not_judged` (a
registered or provisional monocular frame the floor check did not judge, map.md §2; the existing
`kept_at_prior` values are unchanged). It also changes behaviour with no change of shape: the
monocular noise floor without depth is 0.15 m (api.md §6); the sent height is measured from
z = 0 when priors fixed the map frame, weighted with the floor's own sigma, which is 0 under
that convention (map.md §2); on every map, a version keeps the previous `floor_z_m` within the
floor band (map.md §6); the drift estimate is fitted
against the priors as sent (map.md §1); and a monocular frame beyond the scale bound is not
judged by the floor check, keeps k and is not rejected for it, but its content is excluded (map.md §2).

**oms/2.6** is additive: the `failed_checks` value `depth_inconsistent` (map.md §4), a hard limit
that rejects an update, with or without priors, when two committed views contradict each other's
sensor depth images; it is predicted by a dry run, and monocular frames are not tested by it. It
also adds the quality report's `depth_pairs` (every pair that limit evaluated, with its agreement
share in each direction and whether it failed, as `rotation_pairs` does for the rotation check,
map.md §4). It also changes behaviour with no change of shape: the parallax choice that separates translational
twins also separates two depth-confirmed hypotheses whose rotations are at least 10° apart, in
pair measurement and in the registration gate, when the measured parallax is plausible for the
pair (map.md §1, Symmetric scenes, and §2, Match graph); and a whole sequence registered through a
lattice pose, which every measured pair agrees with, is left to the depth-consistency limit, so a
false lattice pair can cause a rejection (map.md §1, Cycle consistency). A frame kept at a prior its
images contradict (`kept_at_prior`, `detail` `conflict_unsupported`) contributes nothing to the map, free-by-rule
grid cells included, whatever its `reason`: also a frame kept for a failed check (such as `too_close`) whose
image-only pose conflicts with its prior, the check winning the `reason` and `detail` showing the conflict; the
too-close content exception applies only to a frame without such a conflict (map.md §2, §6; api.md §4). A frame
whose content is withheld but whose pose is trusted (excluded by the floor check or the height gate, or
`floor_not_judged`) keeps its free-by-rule cells, and a `floor_not_judged` frame's depth gives no scale evidence:
its sent camera height constrains only its own camera pose and is never combined with its own depth or floor
plane into a scale estimate (map.md §2). Such a keyframe stays a registration reference for image matching and
pose, its features used without its depth: that depth lifts no later frame, in registration or in locate (no keypoint
depth for metric lifting or alignment, no depth confirmation, no scale evidence), so a frame whose only depth
references are such keyframes is one with no depth reference (map.md §2; api.md §6).

**oms/2.7** is additive: frame analysis (api.md §7) takes the request's optional `labels` (text
prompts, as an update's) and `labels_only`, the output `masks` (each object's image mask, COCO
run-length encoded), and answers `matches` (the best object per client label, or none). Every
analysis object also carries `centre_cam_m`, its centre in the camera's optical frame. A
`T_map_cam` may be sent without a map: the outputs are then in the frame that pose is given in,
at the scale of the frame's own depth.

**oms/2.8** is additive: the quality report's `scale_check` (the job's sent camera heights
against its frames' own floor heights in their raw depth, api.md §4) and frame analysis's
`scale_check` (a sent camera height against the frame's own floor height, api.md §7), so a
client sees when a documented camera height and the depth disagree.

**oms/2.9** is additive: the grid marks a **drop**, a cell whose floor was not seen and that
holds a surface below the floor band, as occupied with `floor_observed` 3 and its negative
`height_m`, and never makes it, or a cell behind it from a camera stand, free by rule; the grid
request's `blind_disc` (`full`, `footprint`, `none`) limits the near blind disc (map.md §6). A
cell that was free by rule or free over an unseen drop may now be occupied, so a robot on a
raised surface does not plan off its edge.

**oms/2.10** is additive: in frame analysis a sent pose's up axis defines the horizontal, and a
monocular frame's floor is rotated onto it (request `sigma_tilt_deg`); `scale_check.floor_tilt_deg`
reports how far the depth's floor was tilted; and a range-scan beam whose return no supported
pixel holds is `unseen`, with each beam's `support` count (api.md §7).

**oms/2.11** is additive: frame analysis takes `frames[]`, other images of the same camera with
client-given relative poses and sigmas, for motion stereo: the service refines the rotations on
the images, triangulates, and gives the reference frame metric depth with `scale_source`
`baseline` when a pair's known translation qualifies, a `baseline_check` against a sent camera
height, a `motion` report per frame, and the `depth_sigma` output (api.md §7). `ScaleSource`
gains `baseline`.

**oms/2.12** changes behaviour with no change of shape: `monocular` `scale_sigma_rel` is a
constant measured against truth, as one sigma of the scale's logarithm (map.md §1); a located
pose's position sigma includes the reference depth's scale error, and locate forms hypotheses
from separate groups of reference keyframes, so a biased solution meets a rival (api.md §6).

**oms/2.13** guards registration, locate and the drift estimate against repeated texture, and is
additive in shape. A points fit's inliers must agree with the frame's own depth, a fit whose
off-floor inliers are fewer than the registration minimum is no pose by itself unless it only
confirms the prior (it is fitted again without the floor points), and a frame that fails either takes its usual fallback (map.md
§2, Texture-aliasing guards); locate applies the floor rule, a floor-dominated fit becoming a
rival (api.md §6). A drifting prior's position is moved by more than its own sigma only when three
independent views agree on that correction, otherwise the frame is held as a prior conflict
(map.md §1, No correction from one view). It also changes behaviour with no change of shape: a `floor_not_judged` keyframe's depth lifts a later frame's points fit for its pose only (never alignment, scale evidence, depth confirmation or locate; map.md §2), and `few_features` counts keypoints as registration extracts them (api.md §6); one view with
at least 100 off-floor inliers and no rival fit among its matches may move a drifting prior
alone (map.md §1). New optional fields: the input status's
`prior_residual_sent_m` / `prior_residual_sent_deg` (against the prior as sent) and
`inlier_floor_share`, the locate result's `prior_residual_m` / `prior_residual_deg` and
`inlier_floor_share`, and `quality.drift_correction`'s `held` and `agreeing_frames`. Ignored
pixels are never filled before a model sees the image: every model gets it unaltered and its
outputs drop them afterwards; only keypoint extraction sees them filled, and the ignored
region's border is a depth edge for a range-scan beam's support (map.md §2, Ignored pixels;
api.md §7, Unsupported beams).

**oms/2.14** is additive: a monocular keyframe's own depth scale is measured by motion stereo
against neighbour keyframes, the relative translation taken from the two priors as sent with a
sigma that covers the odometry's error over the move, its scale and a client fix between them
(map.md §1, Keyframe depth scale). A confident measurement decides the floor check of a frame
the check did not judge about k: when its floor agrees with the measured scale, the frame is
judged and gives its content at its floor-check scale (map.md §2, Floor check). It changes no
registration: lifting at the measured scale was replayed and refused. New optional fields: the
input status's and the dry run frame's `stereo_scale` `{scale, sigma_rel, neighbours, judged}`
(api.md §4).

**oms/2.15** is additive: a frame's `body` carries its camera's mount on the robot's body
(`T_body_cam`, `measured` or `commanded`, with its joint-reading stamp; a fixed mount as the
camera's `body_mount`) and the body's motion since an earlier frame it names (`T_from_body` with
yaw and translation sigmas and their sources `imu`, `odometry`, `lidar_match`,
`dead_reckoning`, or `none` for a head pan with the body still), with the image stamp and a
`standing` flag (api.md §4, Body links). Composed,
they are a metric relative camera prior between frames of one body that replaces the priors'
difference for that pair, never adds to it (map.md §1, Body links). A mount read more than
0.03 s from its exposure (0.5 s when standing) is stale and its links are unused, never an
error. A link whose frames match is a linked edge of registration: the images measure the
relative rotation, the link the metric offset, and a points fit that conflicts with the linked
pose is no placement (map.md §2, Match graph); a frame kept at its prior with a usable link to
a confirmed frame is kept at the linked pose. A link path is the baseline of the keyframe depth
scale, so a turning look measures its keyframes' scales, and link-measured scales are a vote for
the monocular depth scale k pooled with camera heights, never on a map whose scale comes from
`metric` priors or depth images (`scale_source` `baseline` when they weigh more; map.md §1,
Scale). New optional fields: the camera's
`body_mount`, the frame's `body`, the input status's and the dry run frame's `body_link`, and
`stereo_scale.link_neighbours`. Locate takes no `body`.

**oms/2.16** is additive: a depth-consistency contradiction (map.md §4) rejects frames, not
updates. A new frame whose committed view contradicts a stored keyframe's sensor depth is
`rejected` with the new reason `depth_inconsistent` and `depth_inconsistent_with` naming the
frames involved (two new frames contradicting each other are both rejected; a frame of any
depth kind registered only through rejected frames goes with them); the job registers again without them, in rounds
to a fixed point, and the rest commits when it passes every limit. The whole update is still
rejected `depth_inconsistent` when the frames so rejected (with their dependents) are more than
half of its registered frames (a lattice-registered sequence) or a contradiction remains; that last round's
decision is reported with `applied: false`. A stored keyframe contradicted by
at least three new frames that are anchored in the stored map, agree with a stored keyframe
they were not registered against, contradict no other stored keyframe, and no two of which fail
their pair with each other (an unevaluated pair counts as agreement) is marked suspect instead (`quality.suspect_keyframes`): it keeps its pose and content but is
no longer paired nor a registration or locate reference; the mark belongs to the committed
version (frames reads: `depth_suspect_version`). `quality.depth_pairs` rows gain `outcome` and
keep listing the rejected frames' pairs.

**oms/2.17** is additive: a locate frame may carry `depth`, with an update frame's conventions
(16-bit PNG of the image's size, registered, × the camera's `depth_scale_m`, 0 = no value, the
ignore masks applied; another size is `invalid_request`). Each pose hypothesis is tested against
its references' sensor depth with registration's depth confirmation, and between confirmed
hypotheses the one whose agreement falls short of the best by more than a margin is refused, as
the parallax separates translational twins in registration (api.md §6, Query depth). A refused
hypothesis is neither reported nor a rival; every hypothesis refused is the new locate reason
`depth_inconsistent`. Monocular references test nothing; the located pose and its sigmas are
unchanged. New optional field: the locate result's `depth_check` `{pose, refused}`, present only
when the frame sent depth; a frame without depth is answered as before.

**oms/2.18** is additive: one **prior scale** per map (map.md §1, Prior scale), the odometry's
error in counting travel, shared by `metric` and `drifting` priors (one client, one odometry).
It is measured on depth-image baselines (a registered frame and its main reference, both priors
of one trust: image distance over prior distance), robustly and never from one sample, decided
at each job's end and applied only when significant; under it a `metric` prior is placed with
its horizontal position scaled about the origin (the map frame is not re-gauged) and a
`drifting` prior through the drift estimate, whose odometry scale it is. Before oms/2.18 an
unspecified bias rescaled `metric` priors only. When the applied scale changes, the keyframes
whose positions rest on their priors are re-placed by the first-order response of their fusion
(their references' corrections carried through the image constraints), the cloud fused again,
in the version the job commits. `prior_residual_*` are against the prior as placed for every
status, `prior_residual_sent_*` against the prior as sent. A registered frame's committed sigmas
compose its references' own (map.md §2, Reference sigma); the prior checks, fusion weights and
gates comparing the frame with its own measurements keep the frame's own sigmas. New optional
quality fields: `prior_scale {value, sigma, applied, evidence_count}` and `replaced[]
{name, displacement_m}`. A `metric` prior keeps its trust under an applied scale (no demotion);
the scale source is then not `priors`. Odometry epochs (map.md §1): a prior may carry
`odometry_epoch`, which the client increments when it re-anchors its odometry; a newer epoch
restarts the drift estimate (`quality.drift_correction.restarted` `signalled`), no odometry
baseline straddles two epochs, and an unsignalled re-anchor (most of a job's supporting frames
near their priors as sent and beyond the bound of their priors as placed) restarts it too
(`detected`) instead of fitting a change. A region-level suspect test of depth consistency is
not part of oms/2.18 (false depth rejections are oms/2.19's). Locate places a frame's prior as an
update does (api.md §6, Prior placement): through the located version's prior scale and drift
estimate, a drifting prior with its drift-grown sigma, in the conflict bound and the rivals'
`outside_prior`; new optional `prior_residual_placed_*` beside the as-sent `prior_residual_*`.

**oms/2.19** changes behaviour only (no field; the schema stays oms/2.18): fewer false
depth-consistency rejections. A depth pair fails only on a supported contradiction (map.md §4):
below the agreement threshold, either fewer than a clear share of the counted points agree or at
least a minimum number of points contradict; a few scattered see-through points (thin
structures, depth edges, far range, a rotation within its sigma) no longer reject a correct
view, while a lattice pose's contradictions, hundreds of points, still do. A rotation-only fit
more than twice its `metric` prior's rotation sigma from it on a thin depth confirmation is too
loose to confirm the frame (`kept_at_prior`, `loose_image_pose`; map.md §2), so it no longer
commits a wrong heading that later frames register on.

## 8. Concurrency and latency

* **Reads.** Reads never wait for an update. These are map summary, objects, grid, frames,
  bird's-eye image, last pose, jobs, and versions (O2).
* **Interactive work.** Locate, frame analysis and dry runs run in an interactive lane that
  never waits behind an update's work. A job's registration phase also runs there. A service
  with a single worker instead runs them ahead of fusion (inference.md).
* **Bulk work.** Fusion runs in a bulk lane.
* **Ordering.** One job writes a given map at a time. A map's further jobs queue in arrival
  order.
* **Disconnects.** A job continues when its client disconnects. Only a cancel request or a stop
  of the service ends it.
* **Latency targets** (specs/evaluation.md; thresholds in `examples/targets.json`):
  * upload to frame readable in a map version: `update.readable_latency`;
  * fusion per new frame, for maps up to 500 keyframes: `fusion.time_per_frame`.

## 9. Uncertainty

Every estimate the skill acts on carries its uncertainty in the same response:
* camera poses: position and rotation sigmas;
* a located camera: also its ambiguity, the distance to the second-best hypothesis;
* object centres: a sigma, and whether each extent was seen in full or partially;
* the map scale: its source and relative sigma;
* grid cells: an observation count, with the unknown state kept distinct from free.

## 10. Constraints

* **Ownership.** Each concern has a single owner module, and none of it is reimplemented
  elsewhere:
  * depth and point-cloud generation;
  * segmentation, box fitting and colour;
  * mapping (registration, fusion, objects, grid, versions);
  * the HTTP layer.

  The ownership is enforced by import contracts.
* **Code.** Shared logic lives in one package. Any language or runtime that serves the
  targets is allowed, CUDA included. Models adapted for another runtime are third-party
  components like any other.
* **Platforms.**
  * **The serving machine is where targets are measured.** It is the Mac that serves the
    episodes: an Apple M3 Pro with 18 GB, at 192.168.0.106 on the user's LAN. Every target holds
    there, measured from a LAN client.
  * An Apple M4 Max with 36 GB, and CUDA hosts, must also run the service. Numbers measured on
    them are reported separately.
* **Errors.** Each error carries a typed, machine-readable code that says whether a retry can
  succeed (api.md §8). An input that cannot be used, such as a frame that cannot be located, is
  a per-input status in a successful response, never an error.
* **Tests.** Every module is covered by offline unit tests: no network, no model weights, no
  running inference server, with models replaced by test doubles. The build fails below 100 %
  line and branch coverage of the whole package, models included. The evaluators are not part
  of the package (§11).
* **Licences.** Licences are not a blocker: the project is for personal and research use.
  Every third-party component and its licence is listed in `THIRD_PARTY_LICENSES.md`.

## 11. Evaluation

The evaluators are client-side tools. They may live in the repository under `tools/`, but they
are outside the service's scope: the service never depends on them, and the package does not
contain them.

A standalone evaluator replays recorded robot-guide episodes, from the base, the humanoid and
the arm, against the simulator's ground truth.
* It is a remote client of the API, run on the machine that holds the episodes and the truth.
* It also runs on the reference captures in `examples/`:
  * `office_sequence/`;
  * `ainex-captures/`;
  * `camera/`, the only one through a distorting lens: it covers the camera model's distortion
    (§4) for C1 and C8.
* It measures each client-contract row (§2).
* Its targets are data in `examples/targets.json`, the single source of targets, keyed by the
  metric ids of evaluation.md.
* It writes a machine-readable result and a summary outside the repository, and compares each
  run with a stored baseline.

The details are in [specs/evaluation.md](specs/evaluation.md).

## 12. Order of work and capabilities

This order of work is guidance, so a robot can run on the API as early as possible. A group is
claimed when it is served and passes its tests, whatever the order:
1. O1 + O4, with the removals of §13;
2. O5 + O6: a wheeled base can then run a whole episode;
3. O7: needed by the camera-only humanoid and the arm;
4. O2 + O3;
5. O8–O10.

Until O2 + O3 land, a job commits one version per update and fusion blocks it.
`GET /api/health` lists the groups the running service implements in `capabilities` (`O1` …
`O10`), so the client can refuse what is missing instead of failing. A partly served group is
named by its part: `O3-jobs` means asynchronous jobs (202, then poll) without the provisional
phase, cancel or lanes. O10 may be claimed without lanes, provided a dry run never waits behind
fusion; the C12 latency target stays its test.

Health, timings and errors are base: every implementation serves them, in no group. The group
of every other endpoint is named in api.md and map.md; the bird's-eye reads belong to step 1
(§13).

## 13. Removed features

These features are removed because the client never uses them. Across all recorded episodes the
calls were:

| Call | Count |
| --- | --- |
| update JSON | 158 |
| locate PNG | 138 |
| segment JSON | 34 |
| locate JSON | 15 |
| locate PLY | 6 |
| reconstruct PLY | 6 |
| reconstruct JSON | 5 |

Last-known camera, pixel-to-map, layers and video were never used.

* The public shell entry points `reconstruct.sh`, `segment.sh`, `mapper.sh`, `view.sh` and
  `start_inference_server.sh`. The inference server is managed by `server.sh` (inference.md).
* The rule that the HTTP API mirrors the command line byte for byte.
* The web application and the 3D viewer.
* Video input and frame sampling (`-fps`).
* PLY and PNG variants of update and locate, the bird's-eye layers option, and the "last known
  camera". Two things remain: the bird's-eye image, as a versioned read, and the last
  *recorded* pose (api.md §6).
* The map list, and `agent_skill.md`. `SKILL.md` becomes a short API reference.
* The rule to ask the user before updating an existing map.
* The outdoor and video examples (`street2.mp4`), and the single-image reference
  `restaurant.jpg`. The episodes cover frame analysis.
* OpenLABEL output. It is replaced by the schema of §7.
* The single-image colour contract across artefacts. An object's `color` stays a display colour,
  a pure function of its id. Its observed appearance is `mean_rgb` (map.md §5).
