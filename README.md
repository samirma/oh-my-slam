# oh-my-slam

The map service of the robot-guide skill: it builds and keeps an anchored, versioned map of a
scene from a robot's camera frames, with objects and uncertainties, and serves it over HTTP
(`server.sh`). `high_level_spec.md` and `specs/` hold the requirements (API schema
`specs/api.schema.json`, oms/2.x); this file records what the spec defers to it: defaults,
thresholds, encodings, limits, the reference captures' file names, and the measured decisions
behind the mapping pipeline.

## Install

```sh
brew install colmap            # COLMAP 4.2.x CLI (feature extraction and matching)
uv sync                        # Python 3.12 environment in .venv, dev tools included
./server.sh                    # 0.0.0.0:9091; starts the inference server (the first start downloads the weights)
```

The `pycolmap` wheel in `.venv` is 4.2.x, and the `colmap` CLI must be 4.2.x too: the mapper
checks both before it maps. `server.sh` is a thin wrapper that runs
`.venv/bin/python -m oh_my_slam.service.main` and exits 2 if `.venv` is missing.

Model weights are downloaded on the first start of the inference server:

* MoGe-2 ViT-L normal (`Ruicheng/moge-2-vitl-normal`) and MapAnything
  (`facebook/map-anything-apache`, pinned to revision `00f9c245`) go to the Hugging Face cache.
  Once they are cached the server reads them without contacting the Hub.
* GeoCalib (pinhole weights from its GitHub release) goes to the torch hub cache.
* YOLOE-26x-seg and its MobileCLIP2-B text encoder go to `~/Library/Caches/oh-my-slam/weights`.

`THIRD_PARTY_LICENSES.md` lists every component and its licence.

## The service (`server.sh`, `oh_my_slam.service`)

The HTTP API of `specs/api.md` (schema oms/2.1) for the robot-guide skill. Implemented so far
(`/api/health → capabilities`): **O1** (anchored map frame and priors), **O3-jobs**
(asynchronous jobs: 202, then poll; no provisional phase, cancel or lanes), **O4** (per-input
status), **O5** (locate with uncertainty), **O6** (navigation grid, points near a place),
**O8** (object records, support surfaces and relations), **O9** (quality report and `accept_if` gate) and **O10** (dry run), with the service plumbing
they need. New maps take the *prior-first registration* (below).

```sh
./server.sh                                    # 0.0.0.0:9091, data ~/oh-my-slam-data/
./server.sh --host 0.0.0.0 --port 9091 --data ~/maps-data
./server.sh --status                           # GET /api/health on stdout; exit 3 if not running
./server.sh --stop                             # and the inference server, if this service started it
```

* **Process.** Starlette under uvicorn. It binds `--host` on `--port` (0: a free port), starts
  the inference server in the background when it is not running (`server.sh --stop` stops it
  again only if it started it), and writes `server.sh: listening on http://<host>:<port>/` on
  stderr once it accepts connections. Its log is `<data>/oms/service.log`; the inference server's
  log `~/Library/Caches/oh-my-slam/server.log` (`OH_MY_SLAM_RUNTIME_DIR` moves it). One service per data folder (`<data>/oms/service.lock`).
  `--keep-versions` is accepted; retention comes with O2 (every version is kept until then).
* **Data.** Everything lives under `<data>/oms/`, beside what the older tools keep in the same
  folder: `maps/<name>/` (`map.json`: cameras, mask digests, latest version; `masks/`;
  `versions/<n>.json`, immutable; `mapper/`, the mapper's own map of the latest version;
  `derived/`, what the grid, points and locate read per version (below); `pose.json`, the
  recorded pose),
  `jobs/<id>.json` and the scratch folders `work/` and `trash/`.
* **Endpoints.** `GET /api/health`, `GET /api/openapi.json`, `POST /api/maps/{name}/updates`,
  `GET /api/jobs/{id}` (`?wait_s` ≤ 30), `GET /api/maps/{name}` and `/frames` (`?version=`),
  `GET /api/maps/{name}/grid`, `/points` and `/objects`, `POST /api/maps/{name}/locate`,
  `GET /api/maps/{name}/pose`, `DELETE /api/maps/{name}`, and `?dry_run=1` (O10, *Dry run*).
  The other endpoints of api.md come with their groups (O2, O3, O7).
  `routes.ENDPOINTS` defines them once: the router and `/api/openapi.json` (OpenAPI 3.1, its
  `components.schemas` the `$defs` of `specs/api.schema.json`) both derive from it.
* **Bodies.** Every request body is validated against `api.schema.json` (the package copy,
  `src/oh_my_slam/service/api.schema.json`, must equal `specs/api.schema.json`; a test checks it),
  and the tests validate every response body against it. Errors are `ErrorBody`
  (`{schema, error: {code, message, retryable, details}}`); `details.field` names the failing
  field. An unknown route or method is `invalid_request` with HTTP 404 or 405.
* **Uploads.** A request body is at most 512 MiB (`app.UPLOAD_CAP_BYTES`); the data disk must
  have 3 times the upload free. Larger: `payload_too_large`.
* **Map names**: 1-64 letters, digits, `.`, `_`, `-`, starting with a letter or digit.
* **Server-Timing** on every response: the request's stages (`upload`, `validate`) and `total`.
  A job's stages are in its body, `timings_s` (`checks`, `register`, `anchor`, `commit`, `total`).
* **Lanes.** Until O3 the service has one job worker: `lanes.interactive` reports it (busy while a
  job runs, the queue, the age of the oldest job); `lanes.bulk` is always idle.

### Update jobs (minimal until O3)

`POST /api/maps/{name}/updates` validates the request synchronously (schema; unique frame names;
every part present; images decodable 8-bit JPEG/PNG of their camera's size; depth 16-bit PNGs of
that size; masks 8-bit PNGs; prior poses rigid; cameras not redefined, `camera_conflict`), refuses
with `inference_unavailable` when the inference server is not ready, records new cameras, and
answers `202` with the job (`queued`, `Location: /api/jobs/{id}`). One worker runs the jobs one at
a time, in arrival order: `queued` → `registering` → `committed`, or `rejected` (the gate or a
hard limit: nothing committed), or `failed` (`error`). A job commits one version;
`provisional_version` stays null until O3, and every committed pose is `fused`. A client
disconnect never touches a job; a job still running when the service stops is `failed` at the
next start. Jobs are kept one hour after they end (`jobs/<id>.json`).

The mapper runs in a process of its own per job (`python -m oh_my_slam.service.mapper_proc`), on
an APFS clone (`cp -c`) of the map's mapper map; the commit moves the clone into place and writes
the version. A rejected or failed job leaves the map as it was.

### Anchored map frame (O1)

* **Cameras.** The mapper models one focal length with the principal point at the image centre.
  A frame whose camera is not that already is resampled once (OpenCV `initUndistortRectifyMap`,
  no rotation) to its *canonical pinhole*: same size, `f = sqrt(fx fy)`, principal point at the
  centre, no distortion. The optical frame does not change, so the mapper's poses are the
  camera's. The canonical intrinsics are given to the mapper (`mapping.api.update(intrinsics=)`)
  in place of EXIF and the monocular focal estimate; the mapper holds them as it holds an EXIF
  focal length (refined within 1.15, see *A focal length the keyframes carry*). A canonical JPEG
  without a mask reaches the mapper byte for byte. EXIF orientation is not applied: intrinsics
  describe the stored pixels.
* **Ignore mask.** The camera's mask and the frame's own (`FrameInput.ignore_mask`) are combined.
  Their pixels take no part in the per-frame checks and are filled from their
  surroundings (`cv2.inpaint`) in the image the mapper gets, so no feature is extracted on the
  robot's body. Not yet applied to the mapper's depth and fusion (the cloud reads come with O6/O7).
* **The map frame.** The mapper keeps its own internal frame, which a rebuild may define again.
  The service frame is reached from it by one similarity `S` per version (`versions/<n>.json →
  sim3`), fitted to *anchors* (`service.anchor`):
  * first update with `metric` priors: the metric priors (their frame becomes the map frame);
    without metric but with `drifting` priors: the drifting ones; without priors: the first-camera
    rule (origin at the first registered frame in request order, +x its viewing direction on the
    horizontal plane, +z up, from the mapper's gravity-aligned frame);
  * every later update: the previous version's poses of the frames both versions hold, together
    with every metric prior of the map. The frame therefore never moves; a rebuild is aligned back.
  The fit is robust: hypotheses from single anchors (rotation and translation) and from pairs
  (scale), scored by the anchors within 3 sigmas (`K_SIGMA`), then a weighted Procrustes of centres
  (weight `1/σ_pos²`) and orientations (`1/(2σ_rot²)`) together on the inliers, refined up to three
  times. Sigmas are floored at 1 cm and 0.5°; a prior without sigmas counts 5 cm and 2°.
  The scale is fitted only when the inliers' centres spread over 5 position sigmas (a camera
  turning in place cannot fix it); otherwise it is held (first update: the next scale source;
  later: the previous version's).
* **Scale source** (`scale_source`, `scale_sigma_rel` on the job, the summary and the frames):
  1. `priors`: metric priors that fix the scale (`σ_s = RMS residual / (spread √(n−1))`); drifting
     priors on a map's first update count here too (their relative motion is metric);
  2. `depth`: depth PNGs (camera `depth_scale_m`, default 1 mm) against the mapper's aligned depth
     of the same frames: the median of the per-frame median ratios (≥ 100 common pixels);
     `σ = 1.4826 MAD(log r)/√n`, at least 0.5 %;
  3. `camera_height`: per-frame heights (else the camera's) over the mapper's camera heights above
     its floor (`floor_z`, internal heights above 5 cm); `σ` from their spread and the given
     sigmas (default 2 cm);
  4. `monocular`: the mapper's MoGe-aligned metres, `σ = 10 %`.
  A later update keeps its map's source unless metric priors (two or more inliers) fix the scale.
* **Quality** (map.md §4): `gauge_shift` is the alignment applied to the update's result in the
  previous gauge, `A = S_new ∘ S_prev⁻¹`: `translation_m` the median displacement it gives the
  stored frames, `rotation_deg` its angle, `scale_ratio` its scale (≈ 1 for an extension; a rebuild
  shows its own gauge). `scale_ratio_vs_prev` is the spread of the stored frames' new map-frame
  centres over their previous spread. `prior_residuals` (against every metric prior) and
  `previous_residuals` (stored frames against the previous version), `rebuilt` (the mapper rebuilt
  the map), `prior_gravity_deg` (first update with priors: the angle between the priors' +z and
  the mapper's gravity-aligned +z).
* **Pose priors in the mapper.** Priors enter after registration only, through `S` and the
  post-checks below: the mapper's pair selection is already exhaustive for up to 200 photos, and
  pycolmap 4.2's pose-prior bundle adjustment would need the priors in the mapper's internal
  gauge before the gauge exists. Measured below (*Live check*).

### Per-input status (O4)

`inputs[]` has one entry per frame, in request order: `registered` with `T_map_cam`,
`sigma_pos_m`, `sigma_rot_deg` and `inliers` (the mapper's observations of the frame, or its
multi-view refinement's matches), or `rejected` with a reason; `prior_residual_m/deg` whenever the
frame has a prior and was placed.

| Reason | When (`service.checks`, `service.update`) |
| --- | --- |
| `duplicate` | The name, or the exact image bytes, already in the map or earlier in the request. |
| `too_close` | Median depth < 0.30 m: from the depth PNG, or the monocular depth model for a *suspicious* frame (< 150 SIFT features or texture < 0.05) — before mapping; or the mapper's aligned depth of a registered frame. |
| `plain` | Texture < 0.01: the share of pixels with an intensity gradient above 4 grey levels per pixel (Sobel/8) at 512 px. Bare walls 0.0003-0.0024; the 128 reference frames ≥ 0.0395. |
| `blur` | Sharpness < 0.035: variance of the Laplacian at ≤ 1024 px over that at a quarter of the size. Reference frames ≥ 0.12; a Gaussian blur of σ ≥ 1.2 px < 0.035. |
| `few_features` | < 20 SIFT keypoints at ≤ 1024 px (reference frames ≥ 27) before mapping; or not placed by the mapper with < 150. |
| `no_overlap` | Not placed by the mapper (no verified link to the map), with ≥ 150 features. |
| `prior_conflict` | Placed more than 3 sigmas (`K_SIGMA`, the prior's sigmas floored at 1 cm and 0.5°) from its metric prior, or from its drifting prior after the similarity that best fits the request's drifting priors. |

Frames rejected before mapping never reach the mapper. A frame rejected after registration
(`prior_conflict`, `too_close`) makes the job map once more without it, so it is not in the
committed map (if the second registration rejects more, they stay in the mapper's map but not
in the version). Nothing registered is the hard limit: the job is `rejected`
(`failed_checks: ["hard_limit"]`). `accept_if` is applied to the report before the commit
(`max_residual_m` against both residual maxima, `max_gauge_shift_m` against
`gauge_shift.translation_m`, `max_rejected_share`). `low_confidence_share` counts registered
frames with `sigma_pos_m` > 0.10 m or the mapper's low-confidence flag.

**Pose sigmas** are a documented proxy until the bundle adjustment's covariance is read: the
registration's angular noise per observation `a` (the mean reprojection error over the focal
length, or the multi-view refinement's median match residual) over `√n` observations:
`σ_rot = a √(2/n)`, `σ_pos = d a √(8/n)` with `d` the frame's median depth (2 m without), times 3
for a low-confidence keyframe, combined with the anchoring's RMS residual, floored at 1 cm and 0.1°.
Anchored maps (below) report their own sigmas.

### Prior-first registration: camera-only frames, odometry, map consistency (impl/reg2)

* **Camera-only frames** (prior `none`) take the engine `OH_MY_SLAM_NONE_ENGINE` names:
  `mapanything` (default, `service.register_ma`, ported from the MapAnything candidate
  `cand/mapanything-priors`) or `geometric` (`mapping.anchored`, as frames with priors). The
  frames with priors are registered first; the camera-only ones after them, against the map's
  confirmed keyframes (a new camera-only map's origin is the geometric engine's: the first-camera
  rule). One MapAnything call per 4 new frames with up to 4 anchors (the most similar keyframes by
  thumbnail among the 60 most recent); anchors closer than 0.30 m leave a call without depth images
  unposed (a turn in place). Its output is aligned rigidly on the anchors and scaled by the frames'
  depth images, else by the anchors' stored depth: the map's own depth scale, whose camera-height
  vote (`ScaleState`, `floor_candidates`) each registered frame feeds as a monocular keyframe does
  — never MapAnything's metres. A frame is registered where its depth agrees with the anchors'
  (`mapping.anchored.agreement`: ≥ 10 % of its points seen, ≥ 75 % of those within 10 % for depth
  images, 15 % between MapAnything's own depths), else `no_overlap` (`few_features` under 150
  features); `too_close` under 0.30 m median depth. Out of memory halves the new frames per call,
  then keeps 2 anchors.
* **Drifting odometry.** Its sigmas grow with the odometry's *path* since the last frame the images
  confirmed (`Drift.observe`: every drifting prior seen, kept, provisional or rejected, across
  updates), not with the straight line between the two priors: a robot that drove away and came
  back has drifted with its path. Two consecutive strong conflicts whose image poses imply the same
  correction (`_same_drift`: one predicted from the other within 3 combined sigmas) re-anchor the
  odometry on the first, and both are placed again (a slip, a long unseen drive); a single one stays
  `prior_conflict`.
* **Map consistency of unconfirmed frames.** A frame kept `provisional` at its prior (no image
  pose) is checked against the depth the keyframes near it see (`AnchoredMap._map_consistency`):
  when enough of its points are seen and fewer than 75 % agree (10 % depth images, 20 % monocular
  after scaling by its median ratio to the map's), the median depth difference is its residual
  (`quality.previous_residuals`, so `max_residual_m` sees a warp), and beyond 3 position sigmas
  plus the depth tolerance it is `prior_conflict` and not fused. Registered frames report the
  residual of their committed pose to their image-only pose, as before.
* **Pose sigmas** of an anchored map's frames: the image pose's covariance (Gauss-Newton, inflated
  by its χ²) with the depth bias the covariance does not see (1 % of the references' median
  depth), combined with the prior as the estimate combines them — for a metric prior the weighted
  mean's own variance (`w² σ_img² + (1−w)² σ_prior²`, `w` from the weights the fusion used, which
  floor the image pose at 5 cm / 1°), for a drifting prior the inverse-variance combination; floored
  at 1 mm and 0.01° (no longer at 1 cm / 0.1°).
* **Bare walls with depth.** A frame with a depth image and a metric or drifting prior is not
  rejected `plain` or `few_features` by the checks (`update.DEPTH_PLACED`); when the images do not
  place it, its depth is aligned on the depth of the keyframes near its prior
  (`mapping.anchored.depth_icp`: point-to-plane Gauss-Newton, the prior a factor, so a wall fixes
  its normal and the prior the rest), and it is `registered` (`method: depth`) when its depth then
  agrees with theirs and the alignment stays within 3 sigmas of the prior. Measured on the
  ray-cast room (`tests.synth.room`, the base camera: 640×480, fx 655.8, 52° HFOV; drifting priors
  over-counting the yaw by 1° per frame, σ 5 cm / 3°): every frame of a 360° turn registers up to a
  45° step, with textured walls (max error 0.8 cm / 0.3°), bare walls and textured boxes (4.5 cm /
  1.1°) and an empty bare room (1.4 cm / 0.4°); at 48° bare walls fail (3 of 9), at 52° (no
  overlap) every engine does; 40° steps hold with 3° per frame of odometry drift.
* **A look in two halves.** A dry run reads the latest committed version and its mapper map as one
  snapshot (`Store.snapshot`, under the commit lock), so it never mixes two versions and never
  waits for a running job; the jobs run one at a time and each reads the version committed when
  it starts (`test_a_look_in_two_halves_with_a_dry_run_between`).
* **Guided matching** is vectorised (one pass over all candidate pairs) and skipped for a keyframe
  that already has 300 ratio-test matches.
* **Dry run.** Features, map placements and neighbour links run 6 frames at a time
  (`dry_run.DRY_THREADS`), each placed against the stored map without scale evidence; drifting
  priors are predicted along the request (the path grows frame by frame) and the re-anchoring rule
  applies as in the update. The update's checks run 4 frames at a time.

### Objects, surfaces and relations (O8)

`GET /api/maps/{name}/objects` [`Objects`] on an anchored map (`mapping.anchored_objects`,
called by `service.prior_first` after the engine placed and fused the update's keyframes):

* **Detections.** Every new frame is segmented (YOLOE-26x + MobileCLIP2 in the inference
  server, `segmentation.api.detect_on_grid`) while it is registered, masks on its monocular
  depth grid (640 px on the long side), and only the frames kept as keyframes are used. Each
  detection is lifted on the keyframe's fused depth into the map frame (`lift_on_grid`, without
  the support its mask bled onto, without the floor named as an object).
* **Association, incremental.** Only the new keyframes are folded in, in request order. A
  detection goes to the object it overlaps (points within the object's box grown by 5 cm, or the
  object's within its own; for compatible labels also the nearness of the centres; for other
  labels both ways at 0.7) and the box is refitted at once; else it starts an object whose id is
  never reused. `match` is what frame analysis (O7) reuses to give a detection a map id.
* **Latest wins.** A keyframe that has an object in view and does not detect it judges it on its
  depth: samples seen through by more than max(4 cm, 4 % of the depth) against samples in place
  (hidden ones and the bottom quarter, the support, tell nothing). After the object's last
  sighting, more keyframes seeing it gone than in place strike it (one alone must see 70 %
  through, from a depth image; monocular views need two): `state: struck` for that version, removed by the next update, and the cloud voxels
  in its box that the witnesses see through are carved. A new object of a compatible label,
  similar size (×1.6) and colour (60 sRGB) appearing in the same update is that object moved:
  `moved`, `moved_from_m`. An object is listed from two detections, or one that no keyframe
  contradicted by seeing its place without detecting it.
* **See-through objects.** Monocular depth looks past glass and foliage, and per detection it
  shows no sign of it (the depth inside a globe's or a window's mask matched its own points within
  1-7 % on `office_sequence`). So an object is treated as see-through by its history: a keyframe
  detected it after another keyframe's depth had seen it gone, or (with depth images) the depth
  inside its masks lies beyond its points in several views. It is then judged by its detection
  only (a keyframe that has its place in view, unhidden, at ≥ 200 px and does not detect it), and
  no keyframe judges any object on depth where one of its detections of a comparable size (≤ 4×)
  covers the object's place. On `office_sequence` (640 px in 4 + 4, 1280 px in 13 and 4 + 9) only
  the cup is struck; the glass globe, the tree behind the window and the window panes stay.
* **Fragments.** A detection cut by the image border that has half its points within an
  object's box grown by 5 cm is that object, whatever its label (an apple's edge seen as an
  orange), and votes for its label with a quarter of its score; one cut detection alone is not
  listed until another view confirms it. Round produce the detector confuses (apple, orange,
  tomato, peach, lemon, pear, onion, potato) is associated together.
* **Views from above** (`segmentation.regions`, `segmentation.api.look_closer`, called from
  `service.prior_first.fuse_objects`). A keyframe whose optical axis is within 45° of straight
  down and whose depth has a dominant support (30 % of its samples within ±1 cm of one height)
  is looked at closer, with two more detector calls:
  * **Raised areas.** Its depth finds what stands on the support: areas raised by more than
    max(8 mm, 2 × the depth's relative noise × the depth) and at most 0.4 m, holes filled (the
    inside of a plate's rim or of a mug), not on the robot (nearer than 15 cm, or a fifth of
    the outline touching that), of 300 px to 0.6 m. An area the frame's detections cover less
    than half of is shown to the detector as a crop around it (grown by a quarter of its
    size, 128 px at least, upsampled to the detector's 1024 px input). A crop detection is
    kept only when 60 % of its mask lies on the raised area, so the support's texture is never
    named, and only when the frame has no detection there yet.
  * **Label scores.** One class-wise pass (`SegmentRequest.per_label`: every label a
    candidate scores from 0.05, not only its best) gives each detection the other labels the
    detector scores it as (`Detection.alternatives`; an instance overlapping it by IoU 0.7 is
    the same candidate under another label).
* **Labels.** An object's label is the one with the most **evidence**: the scores of the
  sightings detected as it (`votes`; a cut view counts a quarter) plus the scores the detector
  gave its other sightings under it (`others`, from the label scores), summed over the views.
  Each label's evidence is weighted by how well the object fits the class
  (`segmentation.detect.label_fit`, `data/label_sizes.txt`):
  * the largest side of the box against the class's typical range (the lower bound halved
    while its sides were not all seen across), and the smallest side against a flat class's
    thickness (a mirror, an envelope, a mouse); a box that misses by a factor f counts 1/f²;
  * the relief its depth shows from above (`anchored_objects.relief`, objects up to 0.6 m
    seen by a ray within 45° of down). The top inside 40 % of the half-footprint against the
    rim from 60 %: lower by max(1 cm, a quarter of the height) is `open` (a bowl, a plate, a
    cup), higher by as much is `closed` (an apple, a mouse). A class of the other kind counts
    0.3.

  A class without a typical shape fits any box. Generic labels (`ball`, `food`, `fruit`,
  `vegetable`, `toy`, `container`) match any label and never outweigh a specific one that has a
  quarter of their evidence. `detected_as` lists the labels detected, most votes first, and the
  label chosen first when only the label scores gave it. The vocabulary has the simulated
  kitchens' items (lettuce, spatula, ladle, salt and pepper shakers, sponge, spray bottle,
  credit card, besides the produce and tableware it had).
* **Fields.** `mean_rgb`: the pixel-weighted mean of the robust mean colours (inner mask, 10-90 %
  luminance) of the 3 sightings that saw it best (uncut masks first, then largest). `color`: the
  id's display colour. `sigma_centre_m`: the sightings' pose sigma and depth noise (1 % sensor,
  5 % monocular, of the distance) over their number, plus the spread of their centres while an
  axis is only partly seen. `extent_seen`: an axis is `full` when an uncut sighting sees it across
  (|cos| ≤ 0.8 to the ray) or sightings from both ends see it; `z` when one sees top and bottom
  uncut, or the top with the box resting on a support.
* **Surfaces.** Horizontal planes of the fused cloud's up-facing points (normals within 25° of
  up, turned to the cameras: no object sides, walls or ceiling): 5 cm cells whose points concentrate (50 %
  of ±10 cm) in a ±2 cm slab at a height many cells share; connected areas of at least 0.06 m²
  with a core (no lines of wall cells), up to the highest camera. `height_m` is the median of
  their points; ids are stable (same height within 4 cm and overlapping extent).
* **Relations.** `supported_by`: the surface or object top within 5 cm of the box's bottom under
  its centre — the surface when both fit unless the object's top is nearer by 1 cm and it is no
  table-like top; of surfaces as near, the largest; a support object is at least as large as the
  object and no floor covering (a rug or carpet is listed only on a room-sized surface). The box
  as fitted then rests on its support: an object seen only from above (its height `partial`)
  finds its support up to 30 cm below its top part and its height is completed down to it. `inside_of`: the centre within the box of a
  larger container (bowl, box, basket, drawer, cup...), which also supports it.
  `verified_version`: the latest version that observed the object or its support with the
  relation holding (surfaces are measured by every update).
* **Region** is the grid's (`nav.object_regions`, the default grid), set when the read's answer
  is computed. Door-labelled objects feed the grid's passages of kind `door`; the points read
  carries each object's id in `object_id`.
* **Per version.** The records are committed with the version (`versions/<n>.objects.json`); the
  answer with regions is computed by the derivation worker after the commit and kept
  (`derived/<n>/objects.json`): a read loads it. Maps of the older mapper list their objects with
  defaults (no surfaces, colour grey, `partial` extents).

Measured on the serving machine (M3 Pro), 2026-11-10, in-process service with the shared
inference server, `examples/office_sequence` downscaled to 1280 × 960, trust `none`:

* **Latest wins.** Split 4 + 9: version 1 lists the cup (`confirmed`, on the windowsill surface,
  `mean_rgb` (159, 155, 147) for a cream mug in shade); version 2 lists it `struck` (p04, p10,
  p11, p12 see its place 97-100 % through), and the laptop, keyboard, key, wallet and glass keep
  their ids 2, 4, 5, 6, 8. In one update of 13 the cup is `struck` in version 1 as well.
* **Surfaces.** Desk top at −0.53 m (0.35 m²; the keyboard, key, laptop and speaker on it) and
  windowsill at −0.41 m (0.22 m²; the cup, wallet and glass on it), ids stable across versions;
  a lower ledge at −0.71 m.
* **Speed.** Segmentation per keyframe 0.22 s at 640 × 480 and 0.18 s at 640 × 360 (median of
  11, one pass); about 1 s at 1280 × 960 (two passes). Objects fusion of 13 keyframes 7.8-14 s
  including the detections it waits for. The objects read: 1-2 ms once the version's answer is
  kept; 0.6-2.6 s when it arrives before the derivation worker has derived the version (the read
  derives it).
* **Arm's view from above** (`e053_evidence`, a rendered kitchen worktop: an arm-mounted
  RGB-D camera at 640 × 360 looking straight down from 0.35 m, metric priors, 10 keyframes in
  3 updates). The objects were replayed offline from the map's stored keyframes (images, depth,
  poses) with the shared inference server, 2026-11-10. Label agreement counts a label the same
  as or compatible with the true one; the centre error is the horizontal distance to the true
  base position:

  | object | main build (served) | + O8 merge | + views from above, evidence, fit |
  |---|---|---|---|
  | apple | apple, 0.4 cm | apple, 0.4 cm | apple, 0.4 cm |
  | egg | egg, 0.7 cm | egg, 0.4 cm | egg, 0.4 cm |
  | plate | missed | plate, 0.5 cm | plate, 0.5 cm |
  | bowl | missed | mirror, 1.3 cm | **bowl**, 1.3 cm |
  | mug | satellite dish, 5.4 cm | computer mouse, 1.3 cm | **cup** (compatible), 1.3 cm |
  | tomato | apple, 0.6 cm | apple, 0.6 cm | apple, 0.6 cm |
  | not there | a 2 cm wire | — | — |
  | **agreement** | 2/6 = 0.33 | 3/6 = 0.50 | **5/6 = 0.83** (4/6 exact) |

  The bowl is a mirror on every frame (0.61, 0.52) and the detector's bowl score is 0.12 at
  best. The mirror does not fit an 8 cm-thick box with an open top (0.08 of evidence against the
  bowl's 0.12). The mug is a cup at 0.48 in its one uncut view; as a satellite dish (0.65) it
  is a third of a dish's least size, and as a mouse (0.68, cut by the border) it is twice a
  mouse's height and open. The tomato stays an apple: the detector scores it as an apple in
  all three views (0.85, 0.68, 0.56) and as a tomato at 0.33, 0.07 and 0.29. Its size (8.6 cm)
  and relief fit both classes, so neither prior can tell them apart. The closer looks at raised
  areas added nothing on this map: the plate's crop is a mirror off its rim (26 % on the raised
  area). The serving inference server predates `per_label`, so the label scores were emulated
  with removal rounds (6 rounds, each without the labels that won a candidate in the round
  before). That is a lower bound of the class-wise pass: a label that won another candidate is
  missing.
* **Synthetic worktop** (an arm's view from above at 640 × 360, depth noise 1 % + 3 mm): the
  worktop within 1 cm, the apple, egg, plate and mug bottoms on it within 1 cm, the tomato
  `supported_by` the plate.
* **Synthetic table** (`tests/unit/test_anchored_objects.py`, exact depth and masks): floor and
  table top within 5 mm, `supported_by`, `inside_of` (apple in basket, egg in bowl), box resting
  on the top, `mean_rgb` exact, struck and moved with ids kept.
### Locate (O5)

`POST /api/maps/{name}/locate` (multipart, as an update: a `request` part with the
`LocateRequest`, one part per image or mask) places each frame in one version without changing the
map (`mapping.relocalize`, from the version's derived keyframes, below). It needs no inference
server.

* **Checks.** A frame is first measured as in an update (`service.checks`): `plain`, `blur` or
  `few_features` before matching (`too_close` needs a depth part, which a locate frame has not).
  The camera's ignore mask and the frame's own are combined, rectified with the image to the
  camera's canonical pinhole (lens handling as in an update), and no feature is taken there.
* **Matching.** SIFT (OpenCV, ≤ 2000 keypoints at ≤ 1024 px) against every keyframe up to 24,
  else the 12 with most nearest-neighbour votes (FLANN, L2 < 230) plus up to 6 keyframes within
  max(1 m, 3 σ + 0.5 m) of the frame's prior; Lowe ratio 0.8, ≥ 8 matches per keyframe.
* **Map points.** Each keyframe keypoint's map point is triangulated with the stored keyframe
  poses where a neighbour keyframe (4 nearest with a baseline ≥ 5 % of the scene depth, optical
  axes within 60°) sees it (reprojection ≤ 2 px at 640 px, rays ≥ 2° apart); the others come from
  the keyframe's depth, rescaled to the triangulated ones (median over ≥ 20). Provisional frames
  take part with their provisional poses.
* **Pose.** RANSAC PnP (EPnP, 2000 iterations, 4 px at 640 px) on every 2-D/3-D correspondence;
  besides that pooled hypothesis, a second fit without its inliers and a fit on each of the 6
  keyframes with most correspondences alone. The reported pose is the hypothesis explaining most
  query keypoints, refined (Levenberg-Marquardt) on its inliers. Without enough correspondences
  (< 15 inliers), a frame whose matched keyframes were taken from one spot (centres within
  max(5 cm, 2 % of the depth)) is placed at their centre with the rotation that turns its rays onto
  theirs (the turn fit of `mapping.locate`, ≥ 15 inliers that are ≥ 25 % of the matches); its
  height is the floor plus the frame's `height_above_floor_m` (else the camera's) when given.
* **Sigmas.** `SIGMA_SCALE × √(fit² + spread²)`, no floor (the keyframes' own sigmas, a floored
  proxy in this base, are not added): *fit* is the Gauss-Newton covariance `s² (JᵀJ)⁻¹` of the inliers' reprojection residuals
  (`s` their RMS, ≥ 0.3 px), as the 3-D RMS of the centre (m) and of the rotation (deg); *spread*
  is the RMS distance of the other hypotheses of the same place, weighted by their squared
  relative support. Keyframes whose depth or pose disagree therefore widen the sigmas.
  `SIGMA_SCALE` is 1.0 until the replay calibrates it (see the measurements below).
* **Ambiguity** (`ambiguity_support_ratio` **0.6**). The rival is the best-supported other
  hypothesis farther than **0.10 m or 5°** that is another place: explaining mostly the same
  query keypoints by map points ≥ 0.5 m (median) from the pose's (a symmetric room's twin view), or
  mostly other keypoints while lying > 0.3 m or 15° away. `ambiguity` is set when its support is
  ≥ 0.6 of the pose's; `pos_m`, `rot_deg` are its distance. A prior, then the frame's height,
  picks between the pose and its rival when only one agrees with it. Turn fits: a second turn
  fit without the first one's inliers, > 5° away, at the same ratio.
* **`consistent_with_prior`**: the pose within 3 σ of a `metric` or `drifting` prior, the two
  sigmas combined per component (the prior's default 5 cm, 2°); null without a prior.
* **`inlier_ratio`**: query keypoints the pose explains over those with a correspondence (turn
  fits: inlier rays over matched rays).
* **`nearest_view`** (no_overlap only): the keyframe with most nearest-neighbour votes (the
  retrieval of this service: SIFT votes, no inference); `turn_deg` from the frame's prior heading
  toward where the keyframe looks (the mean of its keypoints' map points) seen from the prior's
  position, counter-clockwise positive, in (−180, 180]; null without a prior. `nearest_view` is
  null when no keyframe has a vote.
* **Recording.** `record: true` stores the last located frame of the request as
  `maps/<name>/pose.json` (outside the versions); `GET /api/maps/{name}/pose` answers it, or
  `no_pose`.

### Navigation grid and points (O6)

`GET /api/maps/{name}/grid?cell_m=&z_min=&z_max=&footprint_radius_m=&version=` (defaults 0.05 m,
0.05 m, 1.8 m; `cell_m` 0.01–1 m) and `GET /api/maps/{name}/points?min=x,y,z&max=x,y,z&voxel_m=
&max_points=` (defaults 0.01 m, 200 000; at most 2 000 000), from the version's derived data
(`mapping.navgrid`, `service.nav`).

* **Free space.** The fusion's TSDF keeps only voxels near surfaces, so the grid keeps its own
  occupancy volume: 5 cm voxels (5 cm high), per voxel the frames whose rays passed through it
  (`free`), ended in it (`hit`), and ended in it on a horizontal surface (`flat`, normal within
  25° of vertical); per column the frames that observed it. A keyframe adds its aligned depth:
  surface points from (almost) every pixel (≤ 400 000), free-space rays every few pixels (one
  voxel apart at 4 m, ≤ 40 000), each ray free up to 5 cm + 2 % short of its hit, depth edges
  skipped, ranges ≤ 6 m.
* **A band.** A voxel is a surface when `hit ≥ 0.3 (hit + free)`, unless it is floor-like (at least
  half its hits flat, within 0.10 m of the floor). A cell is `occupied` when a surface voxel's centre
  lies in `[floor + z_min, floor + z_max]`, `free` when not occupied and a ray passed through it in
  the band (or its floor was seen), else `unknown`; an overhang above `z_max` never makes a cell
  occupied. `height_m` is the highest surface or floor voxel's centre below `max(z_max, 2.0)`,
  relative to the floor; `floor_observed` 1 for a floor-like voxel in the column, 2 for a cell free
  by rule (below) whose floor was not seen, else 0. `floor_z_m` is the
  version's floor, else the lowest layer with ≥ 25 % of the most flat hits of any layer, else the
  cameras' heights, else 1 m below the lowest camera.
* **Camera stands.** The cell under every keyframe camera (and the disc of `footprint_radius_m`)
  is free whatever was seen; so are the unknown cells of its blind disc that the camera cell
  reaches without crossing an occupied cell: radius `h / tan(depression)` of the bottom row's
  centre ray at camera height `h` (0 when that ray looks < 2° below the horizon; at most 2 m).
  This is done at 5 cm for the features and again on the cells asked.
* **Resolution.** Passages, regions and frontiers are computed at the volume's 5 cm whatever
  `cell_m` is asked (the layers are then resampled), so they do not change with the cell size.
* **Passages.** Through every free cell on the medial axis of the free space with a clearance
  below 1 m, the chord toward its nearest obstacle and on to the opposite one, refined to the
  shortest within ±10°; kept when it lies in a wall — beyond both ends the obstacle goes on along
  the chord for 0.4 m, no free cell on the line, ≥ 50 % occupied (beside the line included) — and
  ≤ 2 m long; the shortest one per gap; then the narrowest free chord beside it is taken (shifted
  along its normal by up to ±0.5 m, turned by up to ±20°, both ends on obstacles, its midpoint
  between the walled chord's jambs), so that an open leaf or a deep frame counts as an obstacle. Its ends are `jambs_m`, its length `width_m`, its midpoint
  `centre_m`, `normal_deg` perpendicular to it from `regions[0]` to `regions[1]`. `kind`:
  `narrow` when narrower than `2 × footprint_radius_m`; else `door` when a door-labelled object
  of the version lies within 0.6 m of its centre, or both jambs are occupied and it is 0.55–1.3 m
  wide; else `opening`. `leaf` (doors): the occupied cells within the gap's width of the jamb line
  (off it by > 0.15 m), along it within 0.15 m of the gap, spanning ≥ 0.3 m — their rectangle;
  `intrudes` when they reach between the jambs.
* **Regions.** The chords are cut out of the free cells; each connected part of ≥ 0.25 m² is a
  region (a chord whose two sides stay connected is dropped). `polygon_m` its outer boundary,
  `coverage_share` its cells with `floor_observed` 1 over its free cells plus the unknown cells its
  boundary encloses (holes filled).
* **Frontiers.** Free cells next to an unknown cell (or the grid's edge), in 8-connected runs of
  ≥ 0.3 m; `region` the region most of its cells belong to (cells on a cut: the nearest region).
* **Ids.** `derived/ids.json` registers every region, passage and frontier answered; a new one
  takes the id of the registered one of its kind whose centre lies within 0.75 m (regions, area
  within a factor 2), 0.3 m (passages) or 0.5 m (frontiers), else the next id.
* **Objects.** `nav.object_regions` is the hook the objects read (O8) calls: the region an
  object's footprint (plus one cell) overlaps most, else the region of the cameras that saw it,
  else null.
* **Points.** The fused cloud in the map frame (with its object ids and colours) inside the box,
  one point per `voxel_m` voxel (`core.geometry.voxel_downsample_indices`), then, above
  `max_points`, the budget grid of `core.geometry.budget_voxel_indices` (`truncated: true`).

**Encodings.** Every `Binary` is little-endian, row-major, base64: `state` uint8 (0 unknown,
1 free, 2 occupied), `observations` uint16, `height_m` float32 (NaN: none), `floor_observed` uint8 (0 not seen, 1 seen, 2 free by rule),
shape `[rows (y), cols (x)]`; cell `[r, c]` covers x `origin_m[0] + c·cell_m` to `+cell_m`, y
likewise with `r`. Points: `xyz` float32 `[n, 3]`, `object_id` uint32 `[n]` (0: none), `rgb`
uint8 `[n, 3]`.

### O5 and O6 measured (2026-11-10, M3 Pro, served from the working tree, heavy lock held)

* **Locate, `examples/ainex-captures`** (640×480, fx 270, camera height 0.34 m): a map of the 64
  frames other than every fifth registered 21 of them (this base's registration); of the 15
  held-out frames 8 were located, 6 `no_overlap` (their headings, left 90°–210° and right
  90°–120°, have no keyframe in that map) and 1 `few_features`. Latency per frame over HTTP
  (upload included) median 0.3 s, max 0.55 s. The map's keyframes disagree with each other by
  10–30° about the same view (single-keyframe fits of one frame range over 40°): the located
  frames report `sigma_rot_deg` 3.8–24° and `sigma_pos_m` 1.1–4 cm from that spread, and one
  (`065_right_060_down`) a rival 34° away with 0.6 of its support (`ambiguity`). Against a map of
  all 79 frames (aligned on 17 shared keyframes, 1.3 cm rms) only one held-out frame has a
  reference: 0.8 cm, 5.5° (σ 3.5 cm, 14.7°).
* **Locate, `examples/office_sequence`** (12 MP): a map of 11 photos, the 2 held out located,
  0.5 s each; against the map of all 13 (aligned 5.5 cm rms) 4.8 cm / 0.26° and 13 cm / 1.9°,
  with σ 9.5 cm / 2.2° and 10.4 cm / 1.6° (median error/σ 0.8 in position, 0.6 in rotation; the
  reference itself is uncertain by its 5.5 cm alignment). No ambiguity.
* **Synthetic room** (`tests/synth/scene.py`, exact depth): 8 frames between 12 keyframes located
  within 0.1–0.5 cm and 0.02–0.1°, 50–90 ms each; a symmetric twin view gives `ambiguity` at 180°,
  and a prior or the camera height picks the right one (`tests/unit/test_relocalize.py`).
* **Grid.** Derivation of a version (volume, features, triangulation): 1.8–2.7 s for 11–23
  keyframes. `GET grid` 9–25 ms on these maps; on a synthetic 15 × 15 m house of nine rooms
  (108 frames at 640×480, a 340 × 340 grid at 5 cm) 0.20 s in process for layers, stands and
  features, with 11 of its 12 doors found (0.89–0.90 m; one narrowed to 0.5 m by a box beside it;
  the missed one has a box 5 cm in front of it) and two spurious passages beside boxes. Camera cells free: 23/23 (ainex) and 13/13 (office). `GET points` of a
  10 m box 35–330 ms. The two-room scene of `tests/synth/rooms.py` gives the door exactly: width
  0.90 m, jambs at y 1.50 and 2.40, normal 0°/180°, kind `door`; with an open leaf 0.85 m, leaf
  reported.

### Derived versions

What the grid, points and locate read is derived from each committed version once, on a worker
thread subscribed to the store's commits (`service.derived`; a read of a version not derived yet
derives it first): under `maps/<name>/derived/<n>/`, `volume.npz` (the previous version's volume
plus the new or moved frames' rays; all of them again when a stored frame moved), `cloud.npz`
(map frame), `locate.npz` (keyframe points) and `meta.json` (floor, objects, door positions);
`derived/features/<image sha>.npz` holds each keyframe image's SIFT once for all versions.
It uses two interfaces only: the version record (`frames[]` with `name`, `camera_id`, `internal`,
`T_map_cam`, sigmas, `sha256`; `sim3`; `floor_z_m`) and the backend's `InternalMap`
(`depth`, `image`, `cloud`, `objects` by internal frame name).

### Live check

Run on the serving machine (M3 Pro, 18 GB), 2026-11-10, service and inference server started by
`server.sh` on 0.0.0.0:9091, the 13 photos of `examples/office_sequence` (12 MP, the EXIF
intrinsics given as the camera), one job each (scripts in the session's scratchpad):

| Run | Result |
| --- | --- |
| 1. trust `none`, new map | 13/13 registered, `scale_source: monocular` (σ 10 %); 58 s wall (checks 1.5 s, mapper 56 s). |
| 2. metric priors = run 1's poses after a known similarity (s 1.3, yaw 40°, t (2, −1, 0.5)) plus noise (σ 2 cm per axis, 0.5°), sent with σ 3 cm / 1°; plus a bare-wall frame (the right part of `examples/camera/img_012`) | 13/13 registered, the wall `plain` (texture 0.0003, 0 features), `scale_source: priors`, `scale_sigma_rel` 0.029. Against the known similarity of run 1: median 1.1 cm, max 1.5 cm, 0.31° — within the noise; scale 1.319 for 1.3 (1.5 %). Prior residuals median 2.1 cm, max 3.7 cm, 0.74°. `prior_gravity_deg` 0.22. 58 s. |
| 3. main path: the first 7 photos with metric priors (as run 2, new noise), then the other 6 with `drifting` priors in an odometry frame rotated −30° and moved (5, 5, 0) | Update 1: 5/7 registered within 0.9–3.6 cm of the truth; 2 `prior_conflict` (17 and 20 cm, 3–5°: the 7 photos alone are the weakly posed window set, see *Rebuilding a weakly posed map*), mapped again without them. Update 2: the mapper rebuilt the map (`rebuilt`, `gauge_shift` 1.59 m / 1.56° / scale 0.117 for its new internal gauge), aligned back: stored frames within 0.8 cm median, 1.5 cm max of version 1; the new frames 0.3–2.1 cm from the truth, in the priors' frame; 2 drifting frames `prior_conflict` at 8–9° (their default σ of 2°). 69 s and 76 s. |

Measured per frame (`/api/health → timings`, 26 frames): registration 3.7 s, fusion 0.54 s. A job's
`eta_s` uses them; a 7-frame job took 69-76 s against 30 s predicted (the mapper's fixed costs).
The pose sigma proxy stayed at its floors (1 cm, 0.1°) on these SfM poses, which is optimistic
against the 1-3 cm the priors measured.

### Expected durations

`/api/health → timings` defaults before any measurement (`service.timings.DEFAULTS`, seconds,
measured 2026-11-10 on the M3 Pro with the prior-first path at 640x480: task_map frames with
priors and monocular depth, objects fused; ainex camera-only frames by MapAnything): registration
0.6 and fusion 1.3 per frame, MapAnything's registration 1.9, checks 0.02 per frame, job overhead
0.4; locate 2.0 per frame and grid read 0.5 (the
targets: measured values replace them as soon as the service has run them); frame analysis 1.5
and objects read 0.1 are placeholders until their groups measure them. Each kind
keeps its last 50 measurements in `<data>/oms/timings.json`; a job records its frames' mean.
A prediction adds up a job's phases, each measured as run (`Timings.predict_phases`):
`job_overhead` (per job: clone, save, commit) + `checks_per_frame` × frames sent +
`registration_per_frame` × frames the geometric engine gets +
`registration_mapanything_per_frame` × camera-only frames MapAnything gets +
`fusion_per_frame` × frames kept. `eta_s` = the remaining predicted time of the jobs ahead + the
job's own (every frame counted as kept); a dry run's `predicted_s` uses its own predicted checks
and statuses (`registration` = overhead + checks + registration, `fusion`).


## Maps

### Folder layout

| Path | Contents |
|---|---|
| `map.json` | Format version, map frame, scale, next frame and object ids, `floor_z`, and the update history. Each update records its inputs, frames added and rejected, SfM method, notes, object summary and per-stage timings. Written last. |
| `frames/fNNNNNN.jpg` | Keyframe images (upright JPEG copies). |
| `frames.json` | Per keyframe: source, camera id, intrinsics, `T_map_cam`, depth grid, pose source, update id, depth scale, `low_confidence`, gravity (`up_cam`, which a rebuild reuses), and registration and depth stats. |
| `per_frame/fNNNNNN/` | `depth.npy` (float16 aligned metric depth), `valid.png` (validity after latest wins), `instances.json` (RLE masks, labels, scores, object ids) and `descriptor.npy` (retrieval descriptor). |
| `sfm/database.db`, `sfm/model/` | The COLMAP database, and the COLMAP model in map coordinates. |
| `cloud.ply`, `cloud_objects.npy` | The map cloud, and the object id of each of its points. |
| `objects.json`, `objects/points_NNNNNN.npy`, `objects/sources_NNNNNN.npy` | Object state (evidence, strikes, merges), each object's canonical points, and the kinds of detection each point came from (which detections shape its box). |
| `.lock`, `.staging/` | The update lock, and the staging area of an update in progress. |

An update writes everything into `.staging/`, then records the list of staged files (the commit
point), moves them into place, and writes `map.json` last. An update killed before the commit
point leaves the map untouched. One killed after it is completed by the next update. Readers
(the service reads versions, not this folder; locate) never lock or write. Do not edit files in `.staging/`. A reader
that starts while an update applies its commit can read a mix of old and new files: it reads a
file that vanishes in between once more, but it has no snapshot of the whole map, and a rebuild,
which replaces most files, widens that window.

### Update semantics

* **The order of addition is "latest".** A later update wins over an earlier one, and within one
  update a later keyframe wins over an earlier one, in input order (the order of the `-i` images;
  a video's frame order). A sequence mapped in one update or split across several in the same
  order gives the same objects, labels and boxes, within the evaluator's stability targets (spec
  §2.3 and §5; not to the centimetre: the threaded global mapper is not deterministic, and two
  runs of one update differ too); only ids may differ, where an earlier update had already
  published one, since a published id persists; and the split map may keep an object that an
  earlier update published and that no later image contradicts, since published objects stay
  published (see *Every object of the map is a confirmed one*), even where the one-update map,
  judging all the evidence at once, never confirms it. An object is judged by the keyframes added
  after its last detection wherever the updates are cut, and a map of photos with a keyframe SfM
  did not pose is rebuilt with the next update's photos (see *Rebuilding a weakly posed map*
  under Mapping, step 4): `office_sequence` split 6 + 7 and 4 + 4 + 5 photos has the one-update
  map's 8 objects with their labels, box centres 1.5-1.8 cm apart (median box IoU 0.59-0.65
  against the one-update map), 7 of the 8 ids, and no cup, its place drawn from the photos that
  see it empty (2026-10-05). It also keeps what an earlier update published that the one-update
  map does not confirm: 6 + 7 a comb (30) its first update published; 4 + 4 + 5 a tree behind the
  glass (24) the four window photos published and a car (20) the second update's rebuild
  published; no later photo sees through their places. All 8 of the one-update map's objects
  pair with the split map's; of the split map's 9 and 10 objects, these 1 and 2 do not (the
  evaluator's `map.stability.matched_fraction`, over the larger count, was 0.89 and 0.80).
  `ainex-captures` split 40 + 39 has the one-update map's 33 objects, ids and labels (median IoU
  0.85, centres 1 cm apart). Where nothing changed, the keyframes of an update agree and their order does not matter:
  * Object association groups all of the update's instances at once, strongest agreement first.
  * A cloud point's object id is a vote of the keyframes that see it, whatever their order. Its
    colour is that of the finest view among the keyframes whose local colour (the mean of 5 x 5
    pixels) agrees with the latest keyframe's, within 64 of 255 per channel: where the keyframes
    agree, the finest view of all.
  * Order affects keyframe names, the numbering of new objects (by their earliest keyframe), and
    every place that changed (see *Later wins*): there a later keyframe wins over an earlier one.
* **Later wins.** "Latest" is the order of addition. A later update invalidates, per pixel, the
  parts of older keyframes that it contradicts: free space seen behind an old point, or a new
  surface in front of an old ray, by more than max(15 cm, 10 % of the depth) once the two
  keyframes' global depth factor (the median ratio of the samples within 25 % of 1: not those
  one of them sees occluded or changed) is divided out — on the old keyframe's far field (beyond
  1.6 times its median depth), by more than the two keyframes disagree there on most of it (the
  90th percentile of their far samples' depth ratios), since monocular depth of the far field
  disagrees by far more than 10 % in a static scene (see Known limitations). It also removes, or
  gives a strike to,
  objects that it sees through. Within an update a later keyframe wins over an earlier one, in
  input order: after the objects are judged (below) and before the cloud is fused, each keyframe,
  from the latest back, loses the pixels that the update's later keyframes contradict by the
  same test, the later keyframes that are well registered and not low confidence judging as one
  with what they still hold; the cloud is then drawn from the later keyframes there, so no hole
  is left. A later keyframe's colour wins where its local colour
  contradicts an earlier one's (above). Capture timestamps are never read.
  * **Which keyframes judge an object.** Every object of the map (every confirmed one: a
    candidate that later keyframes see through is one whose depth nothing confirmed, and it stays
    a candidate) is judged by the update's keyframes added after its last detection — within the
    update as across updates, whether the camera looked away or not, from any viewpoint and
    distance. A keyframe judges when its pose error, seen from the object's distance, is at most
    15 % of the object's extent (2° is 2.5 cm at 0.7 m): an SfM pose with 100 points whose
    reprojection error (as an angle: pixels over the focal length) is below that, or a pose
    refined with feature matches whose residual is; it is not low confidence, has at least 80 % of the object's
    samples in its image (the image border counts), at least half of them unoccluded, on at
    least 20 distinct depth pixels, and agrees with the object's detections about its
    surroundings. Monocular depth of two keyframes disagrees by a smooth factor (tens of percent
    between viewpoints metres apart) and a video's loop can be misaligned by decimetres, so the
    surface in a band around the object's mask in the detecting keyframe nearest by viewpoint
    (half the mask's size wide: its support, the wall behind it) is lifted into the map and
    compared with the judging keyframe's depth: where it sees the same surface (within a factor
    1.3), their median ratio must lie within 25 % of 1 and half of the band within the depth
    noise, max(5 cm, 8 % of the depth), of it once divided out. The keyframe's depth is divided
    by that ratio. A keyframe that detected something at the object's place does not judge it:
    another object, at most twice its size (not its support), whose mask holds half the
    object's samples, or 20 of them when a keyframe detected that object at the object's own
    place (the object under another label, which a keyframe across the room places a little
    apart: a bag seen as a handbag), unless a keyframe detected it beside the object.
    Only places within the fused depth are judged (the depth to which the map's fusion draws a
    keyframe, 2.5 times the map's median depth, at most 30 m, see Integration): an object is
    judged only when a keyframe that detected it has it within its fused depth, and only by
    keyframes that have most of its samples within theirs. Beyond it the map draws no surface,
    and monocular depth places a car 40-100 m down a street metres off: on `street.mp4` such
    distant cars were "removed" or "moved" by later keyframes whose depth looked past the box
    beside them.
  * **Removing an object.** A sample is seen through when the keyframe sees farther than it by a
    margin: max(0.25 m, 15 % of the depth), but for a small object at most 15 % of its own
    extent (at least the depth noise): a cup on a windowsill is seen through by 5-20 cm. The
    keyframe's verdict is the share of the samples it sees through beyond what it would see
    through anyway: the share the object's own detecting keyframes see through (its 3 largest
    detections: monocular depth sees between a ladder's rungs and a plant's leaves, and places a
    small object differently from one keyframe to the next), or the share of the object's
    surroundings the judging keyframe sees through, where nothing changed, whichever is larger.
    A sample where the keyframe sees the object's support at its foot (the surface within the
    margin lies in the object's bottom quarter, at most the margin above its base) tells
    nothing: the keyframe sees the table or the windowsill there whether the object stands on
    it or not. The share leaves such samples out; they count as seen in place only for whether
    the keyframe sees the object in place. In `office_sequence` mapped in two updates the cup's
    four detections are placed 5-15 cm apart (the first update's four photos have their
    matches mostly on the trees behind the window, see Known limitations), so its detecting
    keyframes see through half of its samples themselves, and the sill at its foot held 15 % of
    them within the margin in the two photos that see its place empty: both saw it gone by
    0.63-0.70, and it only got a strike.
    The latest keyframe wins: a keyframe that sees the object in place (at most 40 % through)
    outweighs every earlier one, so only the keyframes added after the last such one count. The
    update removes the object when most of those see through at least 60 % of it: three
    keyframes remove it; two remove an object detected in at least 4 keyframes when both see at
    least 70 % through from well-supported poses (absence, like presence, is confirmed by two
    keyframes); else the object gets a strike, and a second strike from a later update removes
    it. An update that sees it in place clears its strikes; an update counts once, however many
    of its keyframes judge.
  * **The removed object's pixels go too.** Its detection masks (grown by 2 px) are invalidated
    in the keyframes that detected it (`valid.png`), so the fused map cloud loses the object's
    points together with its record.
  * **Its place shows the latest observation.** The keyframes that saw through the object (its
    witnesses) are then the only ones that see the surface it stood on or hid, often fewer than
    the 3 keyframes a surface usually needs (the cup is removed by 2 photos). The map records
    the place (`objects.json → vacated`: the retired masks, the object's box and the witnesses)
    and, in it, draws what the witnesses see however few keyframes see it, takes colour and object
    id from the witnesses (not the cup's shadow in the older photos), and drops points that the
    witnesses see through and no later update sees (what is left of the object). The witnesses
    are one observation of the place. A witness that sees a nearer surface by more than twice the
    depth noise (another object in front) says nothing about a point; per point, the median over
    the others of how far beyond it they see decides whether they see it (within the visibility
    tolerance, drawn) or through it (dropped, unless one of them sees it). Their monocular depth
    of the empty place disagrees by a few percent: in `office_sequence` mapped in three updates
    (4 + 4 + 5 photos) the two photos that see the cup gone placed the sill 3.5 cm in front of
    and 3.2 cm behind the surface the other photos agree on, and when either witness alone could
    carve that surface, the one that saw farther left a hole of about 8 × 10 cm where the cup
    stood. A place hidden from a witness by more than half (a vase in front of the cup) is not
    judged by it: when only one keyframe judges, the object gets a strike and stays. The
    place is its detecting keyframes' retired pixels at or behind the object (up to the depth
    noise, max(5 cm, 8 % of the depth), in front), and its box grown by max(5 cm, 3 % of its
    viewing distance). A later update that sees the place again is fused like any other.
  * **A place that changed within one update.** The order of addition is the only sign of
    "latest", so an update whose last keyframes contradict its first ones ends with the latter:
    mapping `examples/office_sequence` in one update leaves the map without the cup, as mapping
    it in two does. An object the update detected is judged, by the rules above, by the
    update's keyframes added after its last detection. An object the update first detected where
    its earlier keyframes saw free space (judged by the same rules, with three keyframes at
    least) arrived: the pixels with which those keyframes saw through it are retired, so that its own
    keyframes draw it however few they are. The ids of the update's detections stay counted, so
    the other objects keep theirs.
  * **An object that moved.** Objects are re-identified by label, size and colour: an object
    detected only by keyframes added after every detection of another of a compatible label, of
    comparable size (scales within a factor 1.5) and colour (median CIELab of its detections'
    pixels: chroma within 10, lightness within 25), standing apart from it, is that object moved
    there when the places were judged empty in their turn: the old place by at least one
    keyframe added after its last detection, the new place by at least one keyframe added before
    the first detection there (keyframes that judge as above, from at most 1.5 times the
    farthest distance the object was detected from — their silence counts only where the
    detector would have seen it at 2/3 of its smallest detected size — and without a detection
    of something of its size there that its own keyframes did not detect beside it), and at one
    of the two places the verdicts show the change as they show a removal (three keyframes, or
    two for an established object at its old place). Monocular depth cannot tell a thin object from the
    wall right behind it (a paper-towel roll on a shelf at 2.5 m), so the detector's silence at
    one place counts, but only with the depth's evidence at the other. The object keeps its id at
    its new place, with the box and points of its latest detections; its old place is vacated
    like a removed object's (its old detections name no object any more, and the keyframes that
    saw it empty draw it), and the pixels with which earlier keyframes saw through its new place
    are retired, so its latest keyframes draw it there however few they are. In 10 photos of a
    living room mapped in one update, a paper-towel roll stands on a shelf in photos 6–7 and on
    the dining table in photos 8–9: the map shows it once, on the table, with the id it got on
    the shelf. A cup of another colour that
    appears elsewhere is another object, and a second cup beside one that stays is a second one.
* **Persistent identity.** Each object keeps one id and one colour for the life of the map, and
  its OBB is refitted from all accumulated evidence:
  * New ids come from a counter and are never reused.
  * Duplicates are merged, and a merge keeps the lower id. Objects of compatible labels merge
    when their points or boxes overlap. Objects of different labels merge when the detector's
    label flickered between keyframes: no keyframe detected both, one of them was detected in at
    least 2 keyframes, their sizes are comparable (not a part of the other or an item resting on
    it), and at least half of either one's points lie on the other's surface. Two single
    sightings are not enough, since the size of one sighting of a few centimetres is noise: on
    `examples/camera`'s desk a speaker and a stapler 2 cm apart measured 0.43–0.50 of each
    other's size, on either side of the comparison, and merged — two keyframes, so confirmed
    and published — in the 9 + 9 + 9 map of the earlier 27-frame capture only; a pair of headphones and a lamp measured 0.39 in
    the one-update map and 0.92–0.95 in the split ones. The cost: a wall lamp that another
    keyframe saw as a thermostat stays a candidate (the one-update map has 19 objects instead of
    20). The merged object takes the label with the most evidence and lists the others in
    `detected_as`.
  * **A part named on its own** joins its object. The detector can name an object in some
    keyframes and only a part of it in others: in `office_sequence` a solar figurine is a
    "figurine" in two photos and its top alone a "bottle opener" in two others. The smaller of two
    objects of different labels (less than half the other's size, but at least a quarter: a
    handle on a door or a faucet in front of a window is an item of its own) that no keyframe
    detected together merges into the larger when each is detected reliably in at least 2 keyframes, at
    least 90 % of its points lie inside the larger's box (grown by 2 cm) and 80 % on the larger's
    own surface (within max(1 cm, 2 % of the viewing distance) of its points), every keyframe that
    detected either had at least 80 % of the other in its image and mostly unoccluded, the
    larger's keyframes saw the part from at most 1.5 times the distance it was detected from, and
    in each of them that sees at least 10 of the part's points (at least 2 must) 80 % of those lie
    on its detection of the larger object or in free space where it sees past a thin part, more
    of them on it than free. An item resting on or in a larger object (a cup on a
    table, a book in a bookcase) is usually detected together with it by some keyframe, lies
    beside the larger object's masks in most of its keyframes, or is out of view in the other's
    keyframes, and stays separate. The part's label votes count in proportion to its size, so the
    object keeps the whole's label.
  * **Copies placed by inconsistent depth** are merged too. Keyframes whose monocular depth
    disagrees (locally, even after the global depth adjustment of Refinement) can place an
    object twice along the same viewing rays. Objects of compatible labels that no keyframe
    detected together merge when, for most of the 3 pairs of their keyframes nearest by
    viewpoint, the two keyframes' depths disagree by at least 5 % (measured on the surfaces both
    see) and the detections coincide once that ratio is removed.
  * **Objects seen twice** are merged. Keyframes that see an object from different sides of a
    room place it at depths a few percent apart, and each set puts its copy on its own viewing
    rays: in the `livingroom.mp4` map a lamp seen along x from one side and along y from another
    was two objects 0.2 m apart, and a door handle seen from 1.7 m and from 4.5 m two objects
    0.12 m apart. The detector's label can also differ between the sides (a robot-vacuum dock
    seen as a toilet from one side and as a dryer from the other). Two objects that no keyframe
    detected together, each detected reliably in at least 2 keyframes, of compatible labels or of
    comparable size, merge when both of these hold:
    * The keyframes of each saw the other's place as the object they detected. The other's points
      that such a keyframe sees (unoccluded, in view) lie on its detection, grown by the depth
      noise at the object's distance (max(5 cm, 3 % of the distance)), or in free space, where
      the keyframe sees farther. At most a fifth of them may lie beside its detection on another
      surface, where the keyframe saw something else (two lamps side by side, each detected from
      one side only, stay two).
    * Each object, moved along its own viewing rays by a depth factor within ±15 %, lies on the
      other's detections for at least 40 % of the points those keyframes see, and the two moved
      copies stand in one place: along both horizontal axes, the middle of each lies within the
      other's extent grown by the depth noise (a metal rack seen in front of a sideboard, but
      standing beside its end, stays apart).
  * **Horizontal surfaces** (tables, desks, counters, beds, rugs) are often split by the
    detector around the items that rest on them: a counter top around a book becomes a left and
    a right "desk". Two detections of one keyframe with compatible surface labels whose masks
    touch with continuous depth (at least 50 contact pixels, depth within 3 %) are one
    detection. Pieces of one surface seen from different keyframes (a desk from one side, a bed
    or a rug from another) merge whatever surface labels they carry, when no keyframe detected
    both and either they share surface — at least 100 of their 1 cm points within 5 cm
    horizontally and 10 cm vertically, at median heights within 10 cm — or the map's fused
    surface joins them: one connected, horizontal patch of it (local normal within ~30° of
    vertical) reaches at least half of each piece's points, at median heights within 15 cm and
    at most 0.3 m apart. The second test catches pieces that monocular depth of a close surface
    placed apart (the kitchen island's corner, 0.5 m below the camera, placed 10 cm lower by the
    keyframes that close the loop). At floor height it is not used: the floor joins anything.
  * **Every object of the map is a confirmed one.** A detection is only a candidate until it is
    confirmed, and a candidate is not an object of the map: `-t full` lists
    every object, that is every confirmed candidate (`-t single` those that the new frames
    observe), and the scene JSON carries no confirmed flag, since every object in it is. A
    candidate is confirmed once it is detected with a reliable mask (not mostly in the
    image-border band, where the object is cut off and monocular depth is unreliable) in at
    least 2 keyframes, or in 1 when no other keyframe of the map had it in view (occlusion is
    ignored, so a detection whose depth puts it inside another surface cannot confirm itself),
    or with any masks in at least 4 keyframes that place it alike (their bounds overlap within
    the depth noise): a detection in the band counts half. A tomato at the foot of a robot's
    downward views (the kitchen sequence `b16`), cut off in all 4 keyframes that detected it,
    is confirmed this way; one-off phantoms in the band are not.
    It must also be visibly drawn in the map cloud, so that it appears in every `color=segment`
    cloud in its colour: it needs at least 5 % of the cells of its box's
    largest face at the map's sampling at its nearest detection, and at least 10. A cell is a
    cloud voxel, or the footprint of one depth-grid pixel where that is coarser: a car seen from
    40 m at the nearest in a 1080p video (grid focal 660 px) has one depth pixel per 6 cm, 9
    voxels of 2 cm, and its detections cannot give it more points than its masks have pixels. A
    keyframe's vote for an object counts only for cloud points inside the object's attribution
    gate, its box grown by the depth noise at its viewing distance (max(5 cm, 3 % of the
    distance)): detection masks are drawn generously (a "carpet" mask over a counter top and the
    floor beyond it), and the object's coloured points must coincide with its box. The vote
    needs a third of the keyframes that see a point, so an object detected in fewer of them wins
    only part of its surface (a refrigerator detected in 7 of the ~15 keyframes that see its
    front) or none of it (a light switch on a wall, a dishwasher detected in 2 of ~13). Each
    confirmed object therefore also takes the unlabelled cloud points nearest its own lifted
    points (the 4 nearest to each, within max(3 cm, 2 % of its viewing distance), inside its
    gate; a point two objects pick goes to the nearer), provided a keyframe that detected it
    fused it: an object detected only beyond the fused depth (a car 40–100 m down a street) has
    no surface of its own in the cloud, and the points near its far-placed lifted points are
    other surfaces. An object stays short when its surface did not survive the fusion (other
    keyframes look at it without seeing it, see *Geometry*) or when its detections are all beyond the
    fused depth and do not meet the cloud; it is then still a candidate.
  * Unconfirmed candidates are kept in the map's state, so that a later update can still confirm
    them; they are never emitted, in any format or scope.
  * **Published objects stay published.** An object the map has exported (`objects.json →
    published`) stays confirmed and exported, with its id, label and colour, whatever later
    evidence says of its confirmation: keyframes that have it in view without detecting it (two
    detections are then needed where one sufficed), a cloud that draws fewer of its points, a
    floor a rebuild places higher, or a rebuild's own evidence (*Rebuilding a weakly posed map*).
    Its box is refitted as evidence accumulates. Only latest wins removes it (evidence of
    absence: *Later wins*), and a merge may join it with another object (the lower or published
    id stays; a rebuild's merges are provisional, see *Rebuilding a weakly posed map*). An object
    that moved keeps its published state at its new place. The consequence: a false detection
    lying on a real surface (the comb, 30, that the first 6 photos of `office_sequence`
    publish) is seen in place by every later view, since the surface is where its depth puts it, so once
    published it stays for the life of the map; only a later view that sees through its place
    removes it.
  * A box rests on the surface beneath it where no keyframe saw below its visible bottom: the
    object was cut off by the bottom of every image that detected it, or hidden behind its own
    front. The support is the highest level of the detecting keyframes' points on horizontal
    surfaces (normals within 30° of up) around the box's footprint (within 15 cm of it, at most
    0.3 m below the box; levels of 1 cm with their neighbours), leaving out the points inside or
    within 2 cm of another object's box (candidates too) unless that object is the one it stands
    on (its footprint holds the box's centre, its top at most 2 cm above the box's bottom): a
    tomato 5.5 cm from a plate otherwise rested on the plate's rim, 3.3 cm above the counter.
    The level must also surround the foot, as every confident keyframe that sees around it saw
    it within 25 cm of the footprint: at least 3 of its points in front of two opposite edges of
    the footprint, or in three of the four quadrants the box's axes divide that ring into. A
    ledge on one side (a partition's edge, a sill) is no support: on `examples/camera` a cup
    misplaced on a wall had rested 8.5 cm lower on a partition's edge beside it.
    The box is extended down to the support unless one of the detecting keyframes saw the space
    in between empty (30 % of the samples there
    that it has in view seen through by more than the depth noise) or the visible part is a
    sliver (under a fifth) of the extended box. On `b16` a cup cut off in all 5 of its keyframes
    had its box 6.5 cm above the counter it stands on, and a tomato 3.5 cm; both rest on it now,
    while a shelf seen with the floor beneath it keeps its box, and a power
    outlet on a wall has no support.
  * The OBB is fitted to the detections that agree with each other: monocular depth of a small
    object can vary by tens of percent between keyframes, and the union of such detections is a
    streak along the viewing rays. With 3 or more detections, the box covers those whose bounds
    overlap (allowing for depth noise) the detection most others agree with.
  * Only the best evidence shapes the OBB. Once 2 confidently placed keyframes detected an
    object, the detections of low-confidence keyframes no longer shape its box: an uncertain pose
    moves the whole detection, and in a split `office_sequence` map it placed the wallet onto the
    Raspberry Pi case beside it. Of the rest, detections mostly in the image-border band are left
    out once the reliable ones are at least 2 and the majority. The other detections still count
    for the label and for confirmation. Each object records which kind of detection each of its
    points came from, so the box is fitted to the points of the detections that shape it.
  * A mask that ran onto the surface an object stands on or rises from is trimmed before the
    detection is used. The mask points in the object's bottom band (a quarter of its height) that
    lie outside the plan footprint of its points above that band are removed from the mask and the
    points: a strip of the windowsill in front of a window, the floor in front of a chair. This
    happens only when the object is mostly above that band and the strip is at most a quarter of
    its points, so a keyboard or a laptop's base is left whole. Heights decide, not surface
    normals, because monocular depth often bends such a strip into the object's own surface.
  * The OBB covers the **observed surface** only (see Known limitations): no class-typical size
    is assumed.
* **Geometry.** The map's geometry is a point cloud: the surface of a TSDF fusion of the aligned
  depth maps. Each point takes its colour and object id from the latest update that sees it. No
  mesh is produced. A surface is drawn where 3 keyframes updated it in the TSDF, or, where
  fewer keyframes have it in view, where all of them updated it: the laptop and the right half
  of the monitor that only photos 0 and 12 of `office_sequence` show, or a backpack that only
  photo 7 shows. A keyframe has a point in view when the point lies within its fused depth and
  projects onto a pixel it fuses (valid, not a depth edge, within that depth), whatever that
  pixel shows. Speckle that other keyframes look at without seeing it (a blob one keyframe
  places in front of a wall that the others see) still needs 3 of them, and so does the second
  zero crossing that 1 or 2 keyframes leave a centimetre or two beside a surface many see (the
  rim of the TSDF band). What only one or two
  keyframes see is drawn only from keyframes whose depth scale was measured with a spread
  (IQR / median of the depth ratios) of at most 0.1: nothing else checks a keyframe there, and a
  keyframe that fits the map worse (a stretch of `livingroom.mp4` scaled with spreads of
  0.14–0.27) would draw a surface where it is not. Low-confidence keyframes are not fused.


## How the mapping works

### Inference server

`src/oh_my_slam/server` is a FastAPI app on a Unix socket, `~/Library/Caches/oh-my-slam/srv.sock`
(mode 0600). It falls back to a short `/tmp` path if the socket path is too long. `/health`
answers immediately, even while the models load. It loads four models:

| Model | Role |
|---|---|
| MoGe-2 ViT-L normal | Metric point map, depth and validity, intrinsics, plus a DINOv2 class-token descriptor for retrieval. Its normal head is not loaded: nothing reads its normals, and the other outputs are the same without it. |
| GeoCalib (pinhole) | Gravity direction with uncertainty. |
| YOLOE-26x-seg with the MobileCLIP2-B text encoder | Open-vocabulary instance masks over `segmentation/data/default_labels.txt`, a curated list of LVIS and COCO nouns. |
| MapAnything (Apache-2.0 checkpoint) | Metric multi-view poses, used as a fallback. |

The device is MPS when available, else the CPU, with a per-process MPS memory cap of 100 % of Metal's
recommended working set (13.3 GiB on the M4 Max under macOS 27; the weights take 6.6 GiB of it,
and the former 70 % left a multi-view chunk too little). After each request the GPU thread measures the
memory; after a failed one, or with the memory in use at 90 % of the cap or more, it first frees
Python's cycles and the allocator's cache. The server is `degraded` while the last request ran out of
memory or the memory in use stays at 90 % of the cap or more: requests are still accepted (the commands
warn), and `GET /api/health` names the restart. A request the device runs out of memory for
answers HTTP 500 with the error `out_of_memory`: the mapper then sends that multi-view chunk again
with half its new views (its anchors kept), down to 2, and keeps the smaller size for the rest of the
update. The memory a chunk needs grows with its views and their aspect: 4:3 robot frames failed at
13-24 views, where 16:9 ones map in chunks of 24. MPS is not thread-safe, so every model call runs on one GPU worker
thread (loading, warm-up and every request; a failed request's traceback frames are cleared there,
so the tensors they held are freed on that thread too). The queue holds 8 jobs; beyond that the
server answers HTTP 503 and the client retries. Only GeoCalib's network runs there: its
Levenberg-Marquardt refinement, which synchronised with the CPU at each of its 30 steps (about
0.3 s per image of that thread), runs on the CPU on one of two other threads (`CpuFinish`), with
CPU tensors only. MapAnything is built on the `meta` device and given its checkpoint's tensors
(no random initialisation of its ViT-G first): it loads in about 5 s instead of 28 s.
torch lives only in the server process. The commands never import it, because loading torch and
Open3D in one process aborts on a duplicate libomp.

### Mapping (an update)

1. **Lock and stage.** Resolve the inputs into keyframes.
2. **Per-keyframe inference.** Each keyframe gets depth (768 px grid), gravity, a descriptor and
   detections at the default threshold (the single-image detection, masks on the 768 px grid).
   Two keyframes are processed at a time, each as soon as ingest has written it, so the video
   is decoded while the first keyframes' inference runs. For a new map whose keyframes have
   one size, the SIFT features of step 3 are extracted on the CPU while the server's GPU runs
   this inference; their camera gets its prior focal length (the median of the keyframes'
   estimates) once inference is done, which leaves the database as extracting afterwards does.
3. **Features and matching.** The Homebrew `colmap` CLI extracts and matches SIFT features.
   SIFT doubles each photo for its finest scale (COLMAP's default), but not the keyframes of a
   video of at least 1600 px: for those it was most of the COLMAP time and memory (the lv walk:
   36 s and 8.7 GB doubled, 10 s and 2.4 GB not). COLMAP extracts one image per thread, and a
   12 MP photo read doubled takes 2.3 GB: the extraction runs on as many threads as three
   quarters of the memory available holds (at least 2; every core when they fit), which leaves
   the features unchanged. On every core beside the inference server, the 13 office photos
   swapped the machine (extraction 305 s, the update 180-335 s; at 2 threads 64 s).
   Pairs are chosen as follows:
   * Photos: every pair up to 200 images.
   * Otherwise: sequential neighbours plus descriptor retrieval, with loop-closure candidates
     for video.
   * Updates: new keyframes are matched against the whole map up to 150 keyframes, and against
     retrieved keyframes beyond that. A new video also gets its own loop-closure candidates, as a
     new map's: `room.mp4` starts in an office that white walls cut off from the rest of its
     walk, and only its later view of the office through the door joins that stretch to the map.
   * **Loosely verified pairs of a video added to a map** (with the map's keyframes, and its
     loop closures)
     that SIFT verified with fewer than 40 inliers are matched again with LightGlue on the same
     keypoints, the 300 most similar by the retrieval descriptor: the office of `room.mp4` and
     its view through the door share 16 SIFT-verified pairs of 15–21 inliers, 53 of up to 95
     with LightGlue. Photos are matched exhaustively up to 200 and are not (on the synthetic test
     rooms LightGlue verified a photo of another room against the map), nor is a new map (given
     38 more loop closures of the dark hallway, the global SfM of `room.mp4` posed six keyframes
     30–37° off their gravity).
   * **A new camera's lens.** When the new keyframes share one camera that no stored keyframe
     has and that has no lens yet, their own pairs are matched first. Multi-view poses of the
     new keyframes alone are refined with the camera's focal length (within 1.15 of an EXIF one:
     step 4) and division distortion (rotation-staged when rotations alone explain the matches within 0.5°: a camera turning in
     place). A lens found by a fit within 1° becomes the database camera, held from then on;
     the keyframes are inferred again on their undistorted images, and every pair is matched
     (again) with it. The wide-angle pan-tilt camera of `examples/camera`, added to a phone
     video's map of the same office, fits 1390 px and k = −0.513 (0.43°).

   **Weak links of a video** are matched again with LightGlue on the same SIFT keypoints
   (COLMAP's `SIFT_LIGHTGLUE`, on the CPU). A cut between two consecutive keyframes is weak
   when at most one verified pair of keyframes at most 12 apart spans it (loop closures do not
   count): what lies beyond hangs on one link. Every listed pair that spans a weak cut, is at
   most 6 keyframes apart and was not verified is matched again; SIFT's result stays where it
   verified more. On the office walk the white wardrobe doors and the door
   (`f000101`–`f000109`, 67–1300 SIFT keypoints each) had no verified pair to the rest at one
   end and 0–1 at the other. The global mapper left these nine keyframes a reconstruction of
   their own in 4 of 5 builds, and the fifth joined them on one 15-inlier pair, 5 % off in
   depth against their neighbours. LightGlue verifies 15–16 of the 41–42 pairs across those cuts
   (15–67 inliers; SIFT found 0–22 raw matches), and every build poses all 124 keyframes in one
   reconstruction, the nine 1.6–1.8 % off in depth against their neighbours. LightGlue costs
   about 1 s per pair and CPU thread (COLMAP's CoreML provider cannot compile its dynamic shapes
   and was 6× slower), so it runs on these pairs only: 7 s on the office walk, nothing on a walk
   without a weak cut (livingroom). The first use downloads COLMAP's `sift-lightglue.onnx`
   (46 MB) into `~/.cache/colmap`; if that fails, the pairs keep SIFT's matches. `map.json`
   (`updates[].notes.weak_links`) records the cuts and the pairs. Photos keep SIFT: on a 12 MP
   photo's ~5000 SIFT keypoints LightGlue took 4.6 s per pair, and the learned features cost
   more than they gave on the photo sets (ALIKED+LightGlue: features and matching +28 s on
   office, +14 s on hallway, where 1 of 3 maps broke; LoMa-B: 8 min of extraction for the 13
   office photos).
4. **Poses (pycolmap 4.2):**
   * **New map:** global mapping (GLOMAP) runs first. If it places fewer than 60 % of the
     keyframes, incremental mapping runs instead. Photos whose matches a plane holds (more than
     half of the verified pairs planar or panoramic, but not a turn: a robot arm's camera over a
     worktop) are mapped incrementally first, from seeds 0, 1, 2 in turn until no keyframe's
     verified matches contradict its pose; GLOMAP's relative poses are ambiguous on a plane. Rotation-dominant input goes to MapAnything
     multi-view poses followed by triangulation and bundle adjustment. Input counts as
     rotation-dominant when more than half of the verified pairs are panoramic (a pair COLMAP
     leaves planar or panoramic counts only when one rotation explains its matches within 1°:
     a textured worktop seen from places centimetres apart is a plane, not a turn), or when the
     baseline-to-depth ratio is below 0.02, or when the SfM points fix no metric scale (no
     keyframe has 50 well-triangulated points: a few photos panned from one spot, whose SfM
     units are arbitrary — the camera 8 cm from the last one placed 3 m away). MapAnything runs
     in chunks of 24, anchored on up to 4 already-posed keyframes, and is also the fallback when
     SfM fails.
   * **A focal length the keyframes carry stays within 1.15 of it.** When every new keyframe has
     its EXIF focal length (or the pinhole of its camera's known lens), an SfM model whose camera
     is further off it is not accepted (`updates[].notes.sfm_focal_rejected`) and incremental
     mapping with the focal length held poses the keyframes, however far the camera turns
     (`sfm_focal_held`); the multi-view refinements below hold it beyond that factor, the
     distortion alone refined. Without EXIF the prior is the keyframes' monocular estimates, and
     the factor 2 (below). A robot head camera (EXIF 270 px, its `camera_info`'s) walking 0.23 m
     straight ahead was refined to 365 px by the multi-view fallback: pure translation cannot
     tell the two apart, every epipolar line running through the focus of expansion, and the
     images located in that map came out stretched 1.4 times ahead and 2.3 times sideways. SfM's
     bundle adjustment of short walks drifted as far (394 and 441 px for that camera, 1204 px
     for one of 656 px). An EXIF focal length is not always exact: a phone's 35 mm-equivalent
     tag (a rounded integer) can be 8 % off, which SfM corrects within that factor.
   * **Refinement of multi-view poses** (`mapping.panorama.refine_turning`, then again on the
     aligned depth). Every verified match, lifted with its keyframe's monocular depth, must
     reproject onto its partner. Pairs whose matches imply a rotation far from the multi-view
     poses' are left out first (repetitive texture: 128 of 425 pairs on the kitchen sequence
     `b16`). Each depth stratum of a pair (depth ratios of 1.25) weighs alike, so the few far
     matches that carry a moving head's parallax are not outvoted by the many on one near
     surface (far points 7.1 → 2.1 px). From the second robust scale on, each residual is
     whitened along the direction its depth moves it by the depth noise measured from the
     matches (5–7 % on `b16`), so the wrong depth of near surfaces no longer pulls the rotations
     of the views that see far ones: `b16`'s two sweeps' pan axes came 1.8° → 0.5° apart.
     `map.json` records the measured noise (`notes.pose_refinement.depth_noise`).
   * **Lens distortion of a new multi-view map.** Its refinement (`mapping.panorama`) fits one
     focal length for the keyframes, and with it one radial distortion coefficient (COLMAP's
     `SIMPLE_DIVISION`). `examples/camera` is a pan-tilt camera turning in place through a 110°
     wide-angle lens whose image corners lie about 500 px from where a pinhole would see them.
     Pinhole SfM collapses its frames, fails, or explains them with a focal length off by a large
     factor (253–294 px from the global mapper, 28,800–271,000 px from the incremental one, for a
     1370 px prior). A new map's SfM model whose focal length is off the camera's prior (the
     keyframes' monocular estimates) by more than a factor of 2 (an EXIF one by more than 1.15,
     above) is not accepted
     (`updates[].notes.sfm_focal_rejected`; the office photos and street2 stay within 1.1), so
     such input takes this path. A focal length within that factor does not make the model right:
     on the camera's first 9 frames alone the global mapper fitted 1398–1515 px and spread the
     centres of the camera turning in place 3.5–8 m apart, its rotations 20–30° off. So an
     accepted SfM model is also tested against a pan from one spot: rotations alone, about one
     centre, with one shared lens (`panorama.refine_turning`, from SfM's own rotations, its
     focal length and distortion refined unless the camera's lens is known), fitted to up to 200
     verified matches per pair. When they explain the matches within 0.5° (median) through a
     lens (the one found, or a known one) that moves the image corners by at least 15 %, SfM's
     translation is the lens's distortion: the model is not accepted
     (`updates[].notes.sfm_turning_rejected`; every fit is in `sfm_turning_fit`) and the
     multi-view fallback poses the keyframes, staged for rotation. The camera's first 9 frames
     fit 0.24° with a coefficient of −0.47 (37 % at the corners). A camera that moves leaves
     degrees; the office photos, taken from about one spot, 0.49° with +0.11 (8.5 %: parallax
     the lens term absorbs; their own refinement fits no distortion); a turning pinhole (the
     rendered `ainex-captures` head) no distortion: their SfM models stand. A rebuild (below) fits the lens again from all its keyframes when they
     share one camera and the new photos have its image size (its stored camera is a pinhole
     again for it; otherwise the stored keyframes keep their `SIMPLE_DIVISION` camera as fitted): held from an update of the first 9 frames (−0.45), the
     camera's lens bent the 27-frame map (the earlier capture of `examples/camera`: 1920×888,
     tilts `down`, `mid`, `up`), its rotations 4° off the one-update map's, where fitted
     again (−0.53) they are within 0.7°. A pinhole fit here ran the focal length from about 1350 px to
     6700–9800 px and shrank the pan steps and the tilts to a fifth to a seventh (on the
     uncropped frames). With the distortion the 27 frames fit 1392 px and a coefficient of
     −0.524: tilts of +13° to +22° and −13° to −18°, pan steps of 13° and 26–28°, camera centres
     within 10 cm. A rotation-only fit of the matches with a fisheye lens model, outside the
     mapper, agrees (+21°/−17° and a 27.7° step for p08/p09). Over few views the focal length and
     the distortion trade off: on the uncropped frames the first 9 fitted 3750 px and −0.77 about
     as well as 1350 px and −0.40, and turned by half as much. A focal length more than twice or
     less than half its prior (1.15 times an EXIF one) is therefore held at the prior and the
     distortion refined alone. A
     coefficient that moves the image corners by less than 10 % of the half-diagonal is set to 0,
     so pinhole cameras stay pinhole cameras: the rendered `ainex-captures` fit 1 %, and the four
     window photos that open `office_sequence` (a phone, which corrects its images itself; trees
     behind glass) +0.06 with a focal length 1.2 times their EXIF one, 3.5 % — what few views of
     a distant scene let the focal length and the lens trade. Kept, that gave the map's camera the
     wrong focal length for good, the next updates' photos (their EXIF focal length) a camera of
     their own, and the split map two cameras its rebuild could not join (the cup then stayed).
     `examples/camera`'s lens moves its corners by 43 %. Otherwise the map's camera becomes a
     `SIMPLE_DIVISION` camera, which later extensions keep as it is; a rebuild fits it again. SfM itself keeps the pinhole model: with
     the distortion it posed the frames orbiting a shallow scene, the centres 0.4–0.8 of the
     scene depth apart and the tilts 1.5–2 times too large.
   * **Keyframes of a camera with distortion are inferred undistorted.** Monocular depth of the
     distorted image, lifted along the lens's rays afterwards, stays inconsistent (the depth was
     solved for a pinhole), so once the multi-view fit has found the lens the keyframes' depth,
     gravity and descriptor are inferred again on their undistorted image. That image is a
     pinhole of the same size and principal point whose focal length holds the whole lens
     (`1 + k r²` times the lens's, `r` the image corners' distance from the principal point over
     the focal length: 0.70 on `examples/camera`, whose undistorted image at the lens's own focal
     length cut a sixth of the width at each side and 9.5 % of the scene's solid angle). Holding
     only the lens's field of view along the image axes (0.75, the lens's corners cut) left that
     map's walls leaning 2–3° and a door frame at the image's side 2–5° from vertical; holding
     the whole lens, within 1–2°. Where it shows nothing of the lens it takes the colours of the
     lens's nearest edge — a black border made MoGe misjudge the surfaces next to it, the tilts of
     one pan position disagreeing by 22 % instead of 5 % — and its depth there is invalid. The
     detections stay those of the keyframe's own image, which the detector sees as the camera
     took it, moved onto the undistorted grid through the lens model (masks resampled, boxes from
     their outline): detecting on the undistorted image changed them (a doorway became a stool, a
     wardrobe and a chair two sinks). Its depth is lifted where segmentation lifts any mask: a
     clock in the very corner of one `examples/camera` frame lies wholly on depth edges there and
     has no points.
     An update of such a map infers its keyframes undistorted from the start. Their depth,
     validity and masks then lie on the undistorted image's grid (`K_grid`: the pinhole of
     `frames.json → K`, `Intrinsics.pinhole`, on its `grid`);
     `frames.json → K` records the lens as `k`, the pose refinement, the contradiction checks and
     locate use the feature matches where the lens's pinhole sees them (read on the
     depth grid where the undistorted image shows them), and the stored keyframe image (the
     input) is undistorted wherever its colours meet the depth. The camera's distortion is
     recorded as OpenCV's rational
     coefficients (k1, k2, p1, p2, k3, k4, k5, k6; within 0.0001 px of the lens) and the exact
     model in `intrinsics_custom`. On `examples/camera` independent SIFT matches, rotated by the
     map's poses, meet within 0.23° (median; p90 0.82°) along the rays the map lifts its depth
     with, where the pinhole's rays left 2.59° (p90 7.1°), and the desk lamp seen at the image
     centre and at its border lifts to one place instead of two 0.5 m apart. It costs one more
     inference pass of the keyframes' geometry (about 12 s for the earlier capture's 27 frames).
   * **Weakly linked parts of a new map** (typical of a video walk through several rooms): a
     stretch that hangs on the rest by one weak link — a doorway crossed with a few dozen
     matches, or one keyframe shared with the rest — has no scale of its own for the global
     mapper, which returns it shrunk onto one point or at an arbitrary scale depending on its
     random start, and its orientation rests on that link alone. After SfM:
     1. An SfM pose needs 20 triangulated points. Keyframes with fewer are not accepted as posed
        (a stretch shrunk onto a point triangulates nothing); incremental mapping with the others
        fixed may register them again. After step 2, a pose that contradicts the keyframe's own
        verified matches (median symmetric epipolar distance above 0.25°; correct poses stay
        within about 0.1°) is not accepted either: on the 6-photo office map the global mapper
        settled one keyframe on 67 matches to one neighbour against 931 to two others. Nor is a
        pose whose SfM points the keyframe's own depth contradicts: its depth ratio over the
        model's metric scale lies outside 0.8–1.25 for photos (the depth alignment would leave
        it unfused, and no neighbour covers its view) or 0.5–2 for video (it would be left out).
        Two photos taken walking down a hallway, a forward motion the global mapper cannot
        resolve, were placed 1.3–3.9 m along it and 0.5–1.6 m apart in height from run to run,
        with ratios of 1.3–4. Such keyframes are joined like unplaced ones (step 3), except in
        an incremental model of photos a plane holds together none of whose keyframes its
        matches contradict: there the depth is wrong, not the pose (a robot arm's views 0.18 m
        above a worktop, within 1.9 cm of its kinematics, at ratios of 0.58–1.54), and the
        keyframe keeps its pose and is fused at the scale of its own points. Photos that hang
        on the rest through one keyframe are joined like unplaced ones too: at most two photos
        whose triangulated points that keyframe sees and no other (a link is 15 shared points).
        Nothing in the
        reconstruction fixes how far from that keyframe they are, and they are too few for step 2
        to judge by their depth. The two hallway photos are such a pair: over 100 seeds of the
        global mapper on one set of matches their depth ratios ranged over 0.015–400, and in 6
        seeds the depth check kept one or both placements (the first photo 0.13–0.92 m from the
        room's first photo instead of 1.46 m). Joined like unplaced ones, they were at 1.46 m in
        all 100. Keyframes of a video are not judged this way. The multi-view refinement itself
        can still go wrong there: in 1 of 10 full runs the global mapper left both photos
        without support, and their refined poses put the second 2.6 m off and the first 0.4 m
        too high, against their depth (both low confidence).
     2. Each keyframe's SfM scale is measured against its MoGe depth (median depth ratio at its
        triangulated points), and its tilt against its GeoCalib gravity (1–3° on correctly posed
        keyframes). Co-visible keyframes whose ratios agree within 15 % and gravity within 10°
        form blocks. A block that hangs on the rest by a bridge or through one viewpoint and
        disagrees with the largest block by more than 12 % in scale or 5° in gravity is scaled and
        levelled about the keyframe it hangs on (the one with the most verified matches to it);
        what hangs on it follows.
     3. Keyframes still unplaced that verified matches connect to the posed ones get MapAnything
        poses anchored on the posed keyframes next to them in capture order (video) or on the most
        similar ones (photos). A secondary reconstruction that shares 3 or more keyframes with the
        main one is merged through the similarity of the shared poses instead.
     4. Realigned and multi-view keyframes are refined with every verified match and the depth,
        the rest fixed, and realigned blocks are levelled with their gravity again (the weak link
        can pull their tilt away).
     5. Guards: a keyframe placed within a tenth of the median step of another keyframe's centre
        while looking elsewhere (optical axes more than 3° apart) is left out, unless its own
        points or matches support the pose or the pair's matches are a pure rotation; a re-placed
        keyframe whose gravity is still more than 25° off is left out too.

     Keyframes with no verified match to the rest are left out and listed. `map.json`
     (`updates[].notes.sfm_unsupported`, `sfm_join`) records what was re-placed and how.
   * **Update:** photos of the same size whose EXIF focal length matches an existing camera's
     prior (the same device and zoom) share that camera, and so its refined focal length;
     without EXIF, frames of the same size and the same kind of input (video frames, or
     photos) share it as before. A video and photos are never one camera: the pan-tilt
     camera's 1920 x 1080 photos took a phone video's pinhole, and their lens was never fitted. Incremental mapping continues
     the stored model only (no further models), with the map's keyframes and cameras fixed and
     a fixed random seed, and the result is mapped back onto the map frame. A fixed keyframe
     that COLMAP dropped and registered again elsewhere (a weakly supported stored pose) is left
     out of that similarity and keeps its stored pose; a result that moved most of them is
     discarded. New cameras keep the focal length the extension refined. Keyframes it cannot
     place, whose depth contradicts the pose, or whose pose contradicts their verified matches
     (as above) are posed by anchored MapAnything and refined with the matches and depth. In
     that refinement each new keyframe's gravity estimate is a prior on its tilt (its GeoCalib
     uncertainty, at least 1°), since the map is gravity-aligned: `room.mp4`'s office, joined to
     its walk by 16 loop-closure pairs of 15–21 inliers, was otherwise refined 13–21° off its
     gravity, and the camera registered on it 17°. A new map's joined keyframes take the same
     prior, about the mean gravity of the keyframes SfM posed (it is not levelled yet): mapped
     alone, `room.mp4`'s dark hallway hung 20–38° off its gravity.
     * *A video placed as a block.* A video whose keyframes the extension leaves unregistered
       (12 or more) is first mapped on its own, as a new map of it would be, from a copy of the
       map's database (its features and matches; only its images registered), levelled with its
       gravity and in metres. That block is placed by the similarity about the vertical (yaw,
       scale, translation) that brings the points its keyframes share with stored keyframes, at
       their depths, onto each other (RANSAC; points within 8 m, agreeing within 5 cm or 8 % of
       their depth), together with the centres of its keyframes the extension did register
       (within 10 cm): at least 40 agreeing points from 3 pairs, scale within 1/3–3. The scale
       is 1 when both are metric (the far points a view down a hallway shares, whose depth is the
       least sure, shrank `livingroom.mp4` to 0.79 of its size), fitted only when the block has
       no metric scale of its own or the map's is a pan's (multi-view depth). Its
       keyframes then take the block's poses and are refined together with the matches, depth
       and gravity, the map's keyframes fixed. Anchored chunk by chunk on the stored keyframes it
       looked most like, `livingroom.mp4` added to a map of `room.mp4` (14 SIFT pairs of 15–20
       inliers between them) was shrunk to a third and stacked onto the bedroom; as a block it is
       one apartment with them (depth agreement with the room's keyframes 0.96).
     * *A camera turning in place* (a new camera of the update whose lens was fitted, or a new
       map's multi-view fallback that found a lens) is given one optical centre, the median of
       its keyframes' (of those SfM registered, when three or more), and its rotations are
       refined alone, with the depth (a pure-rotation model bends them to the parallax of the
       phone views they are matched with: 15° off their gravity); taken when one centre explains
       the matches within 1° (median). The
       pan-tilt camera's views up at a bare ceiling were otherwise placed 0.13–0.17 m from the
       others.
   * **Maps with fewer than 3 keyframes:** SfM is re-run over all keyframes and aligned to the
     stored poses.
   * **Rebuilding a weakly posed map.** An update holds the stored keyframes fixed, so a pose SfM
     did not give would be frozen for good: the four window photos of `office_sequence` mapped
     first are posed by the multi-view fallback (no SfM scale, matches mostly on the trees behind
     the glass, see Known limitations), and the 4 + 4 + 5 map kept boxes up to 3 times longer
     near the camera, two extra objects and one window fewer than the one-update map.
     * *When.* A map of photos (every update) with at most 60 keyframes, one of which SfM did not
       pose (refined by feature matches, or holding fewer than 15 SfM points), is mapped again
       with the new photos, as one update of them all. A map of at least 3 photos that SfM
       posed none of (a pan from one spot: the pan-tilt camera alone is posed by multi-view, at
       the scale of its own depth, about half of metric through its wide lens) is rebuilt with a
       new video as the video and then the pan would be mapped: the video on its own (see *A
       video placed as a block*), the pan's keyframes matched with every keyframe of the video
       and registered onto it with their lens held (incremental mapping; anchored multi-view for
       those it leaves), then given one centre. Mapped together by global SfM, the pan-tilt
       camera's keyframes bent `room.mp4`'s walk (98 keyframes realigned, the lens refitted to
       k = −0.41) or were left out. The photos and the video keep cameras of their own. Not when the last two rebuilds left the
       same keyframes weak and those are still the weak ones (more photos did not help, and a
       rebuild costs more each time), nor when the last update already rebuilt it and left it
       rotation-dominant (a head turning in place: more photos from the same spot give no
       parallax, and it would be rebuilt every time at a cost growing with the map);
       `updates[].notes.restart_skipped` says why. A rebuild that would leave out any stored
       keyframe is abandoned in the same update and the map is extended instead
       (`notes.restart_abandoned`); one that succeeds is recorded in `notes.restarted`.
     * *What it keeps.* The stored keyframes are mapped from what the map holds of them: their
       images, depth and validity (pixels earlier updates retired stay retired, so an object
       they removed does not come back), intrinsics, gravity, descriptors and detections — no
       inference runs again — and the SfM database keeps their features and matches. Each keeps
       the update that added it: latest wins judges the rebuild's verdicts update by update (an
       object's strikes as those updates gave them), and the places of removed objects carry
       over.
     * *Ids.* An id the map published stays with its object. Each detection records the id it
       was first published with (`instances.json → first_id`, never rewritten), and a rebuilt
       object takes the ids of the stored detections it owns that were the first to carry
       them (an id's founding detection): the published ones before a candidate's, then the
       lowest. A rebuilt object that takes a published id is published whatever the rebuild's
       evidence says of its confirmation (published objects stay published), and latest wins
       judges it as any object of the map. The rebuild never gives one object two published
       ids: its grouping keeps instances whose detections carry different ones apart, as it keeps
       two existing objects apart, and its merge of duplicates never joins two objects holding
       the founding detections of different ones (an object with other detections of a published
       object may still join another as its duplicate). Every published id thus keeps an object
       of its own, in the output. A candidate's id (never exported) whose founding
       detection another object took resolves to that object, provisionally
       (`objects.json → rebuild_merged`: each rebuild decides again). The map's other merges
       (`merged_into`) are permanent, and when the extension merges two objects the published id
       stays before a lower candidate's. A published id that no object takes is listed in the
       update's `objects.removed`; none vanishes unreported. A stored detection keeps its ids
       when the rebuild infers its keyframe again on another undistorted image (a lens fitted
       again: the masks move onto the new grid, the ids with them); lost there,
       `examples/camera` (its earlier 27-frame capture) split 9 + 9 + 9 renumbered every object at each rebuild and gave four
       published ids to other objects. Objects first seen by the new photos are numbered on from
       the map's count, as an extension numbers them. So ids can differ from the one-update
       map's where an earlier update had published one (spec §2.3). In `office_sequence` split
       4 + 4 + 5 the four window photos publish two windows, 9 and 22 (the one-update map
       numbers the second 21). Before the rebuild kept published ids apart, its grouping could
       put 22's founding detection with window 9, or both windows in one object: 22 then went
       to the other window's object for an update (`rebuild_merged`), and a geometric rule that
       gave it to a rebuilt object standing where the map had published it did not cover the
       second case (2026-10-05: 9 and 22 in one box for an update). Kept apart, 22 stays on the
       second window after every update.
     * *Cost.* The rebuild re-poses and re-fuses every keyframe: office 4 + 4 + 5, updates 2 and 3
       took 18 s and 17 s (extending: 17 s each); ainex 40 + 39, update 2 took 127 s
       (extending: 84 s; the whole sequence in one update: 158 s).
     Maps posed by SfM, videos and larger maps are extended.
   * If no new keyframe overlaps the map, the command exits 5 and the map is unchanged.
5. **Refinement.**
   1. Geometry is re-run for keyframes whose COLMAP focal length differs by more than 3 %
      (keyframes whose camera turned out to have distortion are inferred again undistorted
      instead, step 4).
      MoGe-2's network does not depend on the focal length, which enters only its
      post-processing (the point map's depth shift is solved for that focal length, then depth
      and intrinsics follow). The first pass therefore asks the server to keep the network
      output of each keyframe without EXIF (the point map before MoGe's output remap and the
      binary mask: 2.2 MiB at 768×432), and the re-run sends the pixels the first pass read and
      gets that output re-solved with the COLMAP focal length: the depth, validity and
      intrinsics a second forward pass gives, bit for bit, in 7 ms instead of 180 ms (on the
      office video, 106 keyframes in 1.3 s instead of 18–20 s). The server keeps at most 1 GiB
      of such outputs, oldest dropped first; it hands each out once, and drops those older than
      30 minutes at its next geometry request. A re-run that finds none runs the network.
   2. For a new map, the metric scale is the median ratio between MoGe depth and SfM depth, and
      the map frame is defined.
   3. Each keyframe's depth is aligned to the map. The scale comes from its SfM points, or
      densely from overlapping keyframes. A scale outside 0.5–2 rejects the keyframe. A scale
      outside 0.8–1.25 marks it low-confidence, and such keyframes are excluded from fusion.
      A dense fit compares the keyframe's depth with what its neighbours see along the same
      rays. A surface much nearer than the rest of a view (`examples/camera`'s ring at the lens)
      is seen by a neighbour beside where the keyframe sees it, against the wall behind, and
      monocular depth may place it anywhere: such comparisons made 25–45 % of a fit at ratios of
      7–9 (or 0.1), and the fit gave scales up to 9.3 where the walls agree at 1.0. Pixels
      nearer than half the view's median depth therefore take no part, on either side, and the
      ratio is the median of the dominant cluster (within a factor 1.5 of the densest 10 % wide
      window). A fit that fails or stays inconsistent (spread above 0.3) is never stored: the
      keyframe takes the median scale of the keyframes it was fitted against and is
      low-confidence (one on `examples/camera` where there were seven, one at ×9.3). Each
      keyframe is first fitted against keyframes before it in capture order, so where a few
      disagree with the rest the order decides which agreement a fit finds (p07_down at ×2.0
      against the p06 frames, whose view of the ring it shares, where p07_mid and the p08 frames,
      which see the same desk, agree at ×1.05); once all are aligned, the densely aligned ones are
      therefore fitted twice more against the 8 aligned keyframes nearest in viewing direction,
      on either side. `ainex-captures` keeps its scales and its frame agreement.
      Dense alignment goes keyframe by keyframe, so its scale drifts along a long sequence, and
      one scale per keyframe cannot reconcile a near field (a counter top 0.3–0.7 m away) that
      disagrees with the neighbours' by 10–30 % while the far walls agree. The depth of all the
      map's keyframes is therefore adjusted together: each keyframe is paired with every
      keyframe (up to its 50 nearest by viewpoint) whose optical axis differs by less than 45°,
      sequence neighbours and loop closures alike. Each pair measures the median log ratio of
      the two depths on the surfaces both see, in both directions, in bins of image place (the
      source keyframe's cells between the field nodes below) and of depth (up to 6 per cell),
      and one correction per keyframe — a scale, a near/far tilt about its median depth and a
      smooth field over the image, `log d' = log d + a + b (log d − log median) + F(x, y)` — is
      solved from all of them at once (robust least squares; the tilt has a prior of 0 and is
      at most ±0.3; F is bilinear between 4 × 3 nodes, each pulled to 0 and to its neighbours,
      its mean over the image 0 so that it moves no scale, and the fields summed over the
      keyframes 0 node by node, since overlapping keyframes see their shared surfaces at the
      edges of both and cannot tell every keyframe bending alike from none; no pixel's depth
      changes by more than a factor 1.5; sparse-scaled keyframes and the map's first keyframe
      hold the metric scale but are tilted and bent like the others; low-confidence keyframes
      follow the others). Monocular depth bends surfaces across the image, differently in every
      keyframe: on `examples/camera` the field brought the pairs' median disagreement from 4.3 %
      to 2.5 %, their p90 from 11.7 % to 4.7 %, the share of pairs above 10 % from 14 % to 0 and
      the worst pair from 22.5 % to 8.1 %, with the walls within 1–2° of vertical. The map's stored
      keyframes take part too, so a later update that closes a loop spreads the correction over
      the whole loop, as one update with the whole sequence would: their depth is rewritten and
      their objects move with them. The field bends the keyframes of a map the multi-view fallback
      posed (every keyframe of the adjustment: a pan from one spot, `examples/camera`,
      `ainex-captures`); a map SfM poses takes none, and a field an earlier update gave its
      keyframes is undone. There the field fitted the overlaps' clutter, and `office_sequence`
      mapped in one update and in several (4 + 4 + 5, 6 + 7) gave boxes agreeing at IoU
      0.41–0.47 (0.57–0.62 without it). `frames.json` records each keyframe's `depth_scale` (at
      its median depth), `stats.depth_exponent` (1 + b), `stats.depth_field` (the field's 4 × 3
      nodes) and `stats.depth_bend` (the field's largest factor away from 1), every adjustment's
      composed.
   4. For a new map, the map is levelled with its horizontal surfaces — before the global
      adjustment, on the depth as aligned: the adjustment bends the keyframes to agree with each
      other, it does not know where up is (on `examples/camera` its floor leaned 2.5° against the
      walls, and levelled with it, so did the walls). Up is the mean of the surface normals of the
      confidently placed keyframes (every 4th pixel, away from depth edges) within 10° of the
      current up, then within 5° of that mean, five times (`mapping.frame.horizontal_up`); a
      ceiling counts like a floor. Unlike a RANSAC plane of the floor band, which levelled the
      kitchen sequence's same keyframes up to 1.5° apart depending on their tilt before (the
      floor seen 2-3 m off at grazing angles, fanned between keyframes), it does not depend on
      the tilt before, and it weighs the counter top the camera looks at as much as its pixels.
6. **Integration.** The update applies latest wins, updates the objects (the fused surface
   tells pieces of one horizontal surface: it is fused only in the boxes where they ask for it),
   applies latest wins within the update in input order (see *Later wins*), fuses the cloud once
   (Open3D TSDF; after the objects this update removed retired their pixels), gives the cloud's
   points their object ids, exports the scene and commits. Each
   keyframe is fused up to 2.5 times the map's median depth (at most 30 m) as it placed that
   depth before its near/far tilt: the tilt moves surfaces, not which of them the keyframe
   contributes (outdoors, where it deepens the far field by 15–35 %, a fixed cut dropped the
   views that make facades 10–20 m from the path reach 3 keyframes). Every step of the fusion is
   local — a voxel depends only on the keyframes, a surface point on its two voxels, the
   few-views and removed-place tests on the point — so a map of more than 300 000 voxel blocks
   (a street walk: millions) is fused in slabs of about as many blocks, one at a time, with the
   same points and the memory of a slab. The per-keyframe projections of the cloud's points
   (few views, attribution, removed places) cull the points in 25 cm cells outside the
   keyframe's frustum and run on up to 8 threads over parts of the map. Monocular depth is
   least reliable near the image border, and the TSDF (a 4 cm band) keeps each disagreeing
   placement of a surface as a layer of its own: on `ainex-captures` the keyframes that saw the
   wall beside door 1 in their outer 15 % placed it up to 6 % nearer or farther than those that
   saw it near their centre, and the wall came out 12–15 cm thick, doubled below the light
   switch. Before fusion, each keyframe's border (its outer 15 % on each side) therefore takes
   the depth of the keyframes that see the same surface near their centre (reconstruction's
   `reconstruction/borders.py`, applied to the keyframes by mapping): every 4th border
   pixel is projected into its 8 nearest keyframes by viewpoint, and where it lands in one's
   central part on the same surface (within 10 %), its cell of 4×4 pixels is scaled by their
   mean depth ratio. Border pixels no keyframe sees centrally keep their depth.
   Then the keyframes are made to agree everywhere, as multi-view stereo fuses its depth maps.
   Neighbouring keyframes still disagree by ~3 % (7–10 % outliers), and on `office.mp4` the
   desk legs came out fat, doubled or tripled and the monitors as slabs with offset copies.
   The copies come from keyframes of other passes 40–55° away: the sequence neighbours agree
   with each other. Every 2nd pixel is therefore projected into every keyframe whose optical
   axis lies within 60° (up to the 96 nearest by viewpoint). A keyframe that sees the same
   surface there (within 3 %) gives its depth along the pixel's ray, and the pixel's 2×2 cell
   takes the median of these depths and its own. A keyframe whose depth there lies more than 3 %
   beyond the pixel sees through it. Where at least 2 keyframes see through a pixel, more than
   see it, the cell is left out of the fusion: these free-space violations are the offset
   copies. A keyframe of an older update never counts as seeing through a newer one's pixel,
   because what it saw through may have been placed there since (latest wins). Most of the gain
   comes from leaving those pixels out (about 10 % of the office's pixels). The median alone
   merges the layers within 3 % (a third fewer cloud points). On the office walk, rendered from
   8 keyframes against their own depth, leg pixels within max(4 cm, 3 %) went from 0.73–0.75 to
   0.85, and the ring around them where the map stands in front of the keyframe went from
   0.39–0.42 to 0.27. For the monitors these went from 0.63 to 0.79 and from 0.40 to 0.22. The
   price is up to 1 point more of pixels where the map lies behind the keyframe (the nearer of
   two disagreeing placements is the one left out). It takes about 2 s for 124 keyframes, at
   every fusion setup, and the fusion of fewer points saves about as much. Both the consensus
   and the border correction change the depth in memory only: the stored depth is unchanged.


### Ownership

import-linter contracts in `pyproject.toml` and `tests/unit/test_ownership.py` enforce it:

* `service` owns the HTTP layer (`server.sh`): routes, validation, jobs, versions, anchoring and
  per-input status. Nothing imports it; it reaches the models only through the mapping,
  reconstruction and segmentation APIs, and the inference server through its client and lock.
* `mapping` owns registration, fusion, objects, the store and the bird's-eye renderer. It reaches
  the inference server only through `reconstruction` and `segmentation`.
* `segmentation` owns detection, exclusive masks, lifting, box fitting and colour assignment
  (`segmentation/colors.py`). It writes no PLY.
* `reconstruction` owns depth, intrinsics, gravity, point generation and fusion and the
  derivation of point clouds (`reconstruction/cloud.py`). It never imports segmentation or
  mapping: labels and colours reach it as data.
* `server` (the inference server process) owns the models and nothing else; torch lives only
  there (loading torch and Open3D in one process aborts on a duplicate libomp).
* The layers run `service` > `mapping` > `segmentation` > `reconstruction` > `client` > `core`.

## Display colours

An object's `color` is a display colour, a pure function of its id (`segmentation/colors.py`),
not its appearance (that is `mean_rgb`). Mid-grey `#808080` is reserved for unsegmented points.
Every colour has a WCAG contrast of at least 3:1 against `#1d2027`, OKLab lightness at most 0.93,
chroma at least 0.08 and a distance of at least 0.11 from mid-grey.

Ids 1–19 use this palette. Ten colours come from Sasha Trubetskoy's list of distinct colours; the
entries that fail the rules above (navy, maroon, purple, beige and the palest pastels) are
replaced. Any two differ by at least 0.095 ΔE_OK, about five just-noticeable differences:

| id | colour | id | colour | id | colour | id | colour |
|---:|---|---:|---|---:|---|---:|---|
| 1 | `#e6194b` | 6 | `#9c4dff` | 11 | `#1fb5a3` | 16 | `#ffc49b` |
| 2 | `#3cb44b` | 7 | `#42d4f4` | 12 | `#dcbeff` | 17 | `#5a9cff` |
| 3 | `#ffe119` | 8 | `#f032e6` | 13 | `#9a6324` | 18 | `#b4339c` |
| 4 | `#4363d8` | 9 | `#a8f04a` | 14 | `#8ef0c0` | 19 | `#c9a227` |
| 5 | `#f58231` | 10 | `#ff9ec7` | 15 | `#808000` | | |

Higher ids cycle by hue, and vary lightness and chroma as they go:

* Id `19 + n + 1` aims at the OKLCh hue `n × 137.508°` (the golden angle).
* The candidates are the 8-bit colours within ±30° of that hue, on six lightness tiers
  (0.62–0.91) and three chroma levels (0.24, 0.15 and 0.10, clipped to the sRGB gamut), that meet
  the rules above.
* The id takes the candidate farthest in OKLab from every earlier id, with the last 10 ids counted
  1.5 times as close, so that neighbouring ids differ most.
* A colour that would repeat an earlier id after 8-bit rounding is nudged to the nearest unused
  triple. Every id therefore gets a distinct colour.

For ids 1–120, any two colours differ by at least 0.045 ΔE_OK and consecutive ids by at least
0.15 (`tests/unit/test_colors.py`).

Mid-grey `#808080` (128, 128, 128) is reserved for unsegmented points, and the palette never
produces it.


## Reference captures

### The `ainex-captures` file names

The spec no longer spells out the capture names, which encode the **commanded** head motion as
`NNN_<motion>_<tilt>.jpg` (`tools/evaluate/names.py`; a name that does not follow it is an error):

* `NNN` is the capture order.
* `<motion>` is the commanded yaw relative to frame 001, positive to the left: `bootstrap` (0°),
  `bootstrap_side1` / `bootstrap_side2` (0°, after a small sideways step), `bootstrap_leftYYY` and
  `bootstrap_leftYYY_side` (+YYY°), `left_YYY` (+YYY°), `right_to_YYY` (+YYY°, turning back
  towards 0°) and `right_YYY` (−YYY°).
* `<tilt>` is `level`, or `up` / `down` relative to the `level` frame of the same motion.

The same-heading pairs are 001/053 (053 returns to 001's heading) and 026/078 (`left_210` meets
`right_150`, which closes the 360° loop).

### The `camera` file names

`examples/camera/` holds 36 frames (1920×1080, no EXIF) of a real pan-tilt indoor camera turning
in place in a home office, through a wide-angle lens with visible distortion. The spec gives it
the uses of `ainex-captures`, and its names encode the
**commanded** motion as `img_NNN_pPP_<tilt>.jpg` (`tools/evaluate/names.py`, `PanCapture`):

* `NNN` is the capture order (009 to 044).
* `pPP` is the pan position (p03 to p11), numbered in pan order: the camera turns left as `PP`
  increases, by a step angle the names do not record.
* `<tilt>` is `floor`, `low`, `mid` or `high`, in increasing pitch: `floor` and `low` point below
  `mid`, `high` above it. The four tilts of one pan position are consecutive, in serpentine
  order: `floor`, `low`, `mid`, `high` at p03, `high`, `mid`, `low`, `floor` at p04, and so on.

Hidden files such as `.DS_Store` and non-images are ignored. An image of a sequence's folder whose
name does not follow the sequence's grammar ends that sequence's part of the run with an error
(`details.errors`); the rest of the run goes on.
Without an angle in the names, the pose accuracy of `camera` is the pan order and the agreement of
the tilts: a pan position's heading (the circular mean of its registered frames' yaws) must turn
left from one pan position to the next, and the four tilts of a pan position, which share its
heading, are its same-heading pairs (every two of them: their yaws, and their depths in
`map.camera_*.frame_agreement_tilt_*`). Pitch direction compares each `floor`, `low` and `high`
frame with the `mid` frame of its pan position: `floor` and `low` must point below it, `high`
above it.


## Known limitations

* **Labels come from an open-vocabulary detector (YOLOE with text prompts) and inherit its
  confusions.** On the rendered `ainex-captures` kitchen, prompting each label alone on the
  mislabelled instances shows that the detector itself prefers the wrong label. A kettle scores
  0.63 as a teapot and 0.10 as a kettle. The counter scores 0.57 as a desk and 0.03 as a
  counter. The light switch scores 0.57 as a power outlet and 0.32 as a light switch. The toaster
  scores 0.77 as a tissue box and 0.05 as a toaster, and the coffee maker scores 0.64 as a water
  dispenser and 0.17 as a coffee maker. A horse picture on a book cover is detected as a person,
  and so are the framed photos on the `examples/camera` shelf (0.50–0.80); a light switch there
  is a wall lamp (0.77, 0.25 as a light switch). The vocabulary size and NMS settings do not
  cause these errors, so they are not patched per scene. The wording of a prompt can matter: the detector scores a monitor far
  higher as `tv` than as `television` or `computer monitor`, and a mouse and a microwave higher
  under those words (Single image, step 4). `wall switch` would name the `ainex-captures` light
  switch, but it also takes the plates of the door handles there, so it is not used. One gap
  was in the vocabulary: common produce was missing, so a tomato or a watermelon took the
  nearest label (apple). Tomato, watermelon and a few other LVIS produce nouns are now included.
* **Large plain built-ins score low.** The detector scores a doorway, a white built-in wardrobe
  or a window with roller shutters seen at a steep angle at 0.1–0.47, so a single frame finds
  no object at the default `--min-score 0.5` in 5 of the 27 frames of the earlier
  `examples/camera` capture (tilts `down`, `mid`, `up`; 3 up frames facing the door and
  wardrobe, 1 facing the window, and the mid frame with the wardrobe and the open door; not
  measured yet on the current 36 frames). Input resolution (640–1920 px), lens undistortion, contrast
  enhancement, flips, tiles, 76 prompt wordings, averaged synonym prompts and the detection head
  were tried: what lifts one of them past 0.5 lowers others or the rest of the image (at 640 px
  the doorway scores 0.54 and the wardrobe 0.22, as a cabinet, instead of 0.47), and the
  vocabulary's size changes no score. Two more detection passes fused with the others (the
  image at 640 px on a 640 px input, and at 512 px upsampled to 1024, without the coarse pass's
  5 % size floor) find the doorway, a light switch and the hanging bunting, so 25 of the 27
  frames, but they also add about 20 false labels on `ainex-captures` (a person on a book cover,
  light switches and a thermostat on door-handle plates, rugs on bare floor, shelves on
  drawers), at least one on `restaurant.jpg` (a bowl on a pendant lamp) and many on
  `examples/camera`, so they are not used.
* **Label flicker.** When the detector names one object differently from keyframe to keyframe,
  the map keeps one object with the best-supported label and lists the others in `detected_as`.
* **Monocular depth of small, distant objects** varies between keyframes, so their map boxes
  are only as consistent as the agreeing detections (see Update semantics).
* **Boxes cover the observed surface, not the whole object.** An OBB is fitted to the points the
  keyframes saw. An object seen only from the front has the depth of its visible surface: in
  the `ainex-captures` map the refrigerator against the wall is about 0.8 m wide and 1.65 m
  tall but only 0.1–0.15 m deep, because only its front was seen. The fit does not invent the hidden part from a class-typical size,
  which would be wrong for any object that is not typical. The only extensions are floor
  grounding, where the visible bottom floats just above the detected floor (see Coordinate
  conventions), and resting on the surface beneath where no keyframe saw below the box (see
  Update semantics). A box is one upright cuboid: `examples/camera`'s L-shaped workstation (two desks
  butted together, detected as one desk) gets a box yawed about 50° to the walls, across the
  corner, that holds the chair in front of it, and the upper cabinets' fronts there run 5° off
  the wall behind them (monocular depth of a surface seen at a grazing angle).
* **Monocular depth disagrees locally.** Before the global depth adjustment (Refinement), the
  keyframes that close the 360° loop of `ainex-captures` disagreed with those that opened it by
  15–21 %, and the near field of a keyframe (the kitchen island 0.3–0.7 m below the camera, seen
  at a grazing angle) by 10–30 % with its neighbours' while their far walls agreed. A scale and
  a near/far tilt per keyframe remove most of it: over the ~775 overlapping keyframe pairs of
  that map (optical axes < 45° apart, sequence neighbours and loop closures alike), the median
  pair disagreed by about 1.8 %, 90 % of the pairs by less than 3.5 %, and the worst pair by
  9–11 % (with one scale per keyframe: 1.9 %, 5.5 % and 21 %, keyframes two apart). What
  remained varied across an image in a way a tilt in depth does not model: the worst pairs are
  40–45° apart and overlap only near their image borders, where monocular depth is least
  reliable. A smooth field over each keyframe's image (Refinement) takes up most of that: the
  median pair at 0.8 %, 90 % below 1.6 % and the worst at 4.1 %. On `examples/camera` it
  leaves the median pair at 2.5 %, 90 % below 4.7 % and the worst at 8.1 % (with the
  out-of-focus near field invalid, below); within a pair, what remains lies in the thin strips
  in which an up and a down frame of one pan position overlap at the lens's edge (up to 18 %
  at their p90; measured on the earlier 27-frame capture, tilts `down`, `mid`, `up`). The corrections also mean a keyframe's depth is not one scale of MoGe's:
  `frames.json` records `depth_scale` at its median depth, `stats.depth_exponent` and
  `stats.depth_bend`.
* **Large horizontal surfaces at floor height can stay fragmented.** Pieces of a surface seen
  from different keyframes merge when they share surface or when a horizontal patch of the
  fused surface joins them (see Update semantics). At floor height only the first test applies,
  since the floor joins anything: two pieces of one rug that neither overlap nor touch stay two
  objects.
* **Scenery behind glass.** Monocular depth places what is seen through a window (trees, a
  street) on the pane. Where most feature matches lie there, poses that lift keypoints with
  their depth are pulled towards a camera that did not move: the first four photos of
  `office_sequence` mapped on their own have about 90 % of their matches on the trees behind
  the window, and the multi-view refinement leaves their cameras 1-6 cm apart where the
  13-photo map measures 8-25 cm (SfM of the four photos alone is no better: rotations 2.5-4.7°
  off, 15-37 points per photo). Objects near the camera, the cup and the wallet on the
  windowsill, are then placed 5-15 cm apart by each photo, and their boxes are that much
  longer. A later update rebuilds such a map with its photos (Mapping, step 4), so the whole
  sequence split as 4 + 4 + 5 or 6 + 7 photos ends with the one-update map's objects, labels
  and boxes (one window keeps a different id, see there), and with the objects its first
  updates published that the one-update map does not have (a tree behind the glass, see
  *Update semantics*): published objects stay published. In a map of the
  whole sequence the global mapper's result varies from run to run for the same reason: in some
  runs the depth check (step 1 of *Weakly linked parts*) rejects photos whose SfM points are
  mostly those trees (depth ratios of 0.25-0.3 and 1.8-2), and their multi-view poses leave them
  low confidence.
* **Video through several rooms.** Where a walk crosses a doorway in a second or two,
  consecutive keyframes share only a few matches, and the global mapper can leave the stretch
  behind the doorway at any scale or tilt, or shrink it onto one point (see Poses). The mapper
  corrects the scale with the depth and the tilt with gravity, but the stretch's heading and
  position still rest on that one link, or on multi-view poses anchored on its capture-order
  neighbours, so they can be a few degrees and decimetres off. LightGlue on the weak links
  (Mapping, step 3) gives such a cut more matches, but it cannot add keypoints: a keyframe of a
  blank surface (the office walk's `f000101`, 67 SIFT keypoints) rests on four pairs of about
  20 inliers, and in 1 of 5 builds its SfM pose was 0.4 m off its neighbours. Keyframes whose
  gravity still disagrees by more than 25° afterwards are left out: on the user's
  `livingroom.mp4` at `-fps 1`, the dim corridor at 77–81 s. The global mapper starts from
  random positions, so two runs on one video can differ in such stretches.
* **Wide-angle lenses.** A new multi-view map's camera gets its lens distortion, and its
  keyframes are inferred undistorted (Mapping, step 4), on a pinhole of the image's size that
  holds the whole lens: the depth grid has 0.70 times the lens's resolution at the image
  centre, and the viewer's frustums and "Go to" show that pinhole's field of view (89° wide on
  `examples/camera`). The one-coefficient division model fits the feature matches (0.24°
  median residual; a free principal point moves 20 px and changes nothing) but bends long
  straight edges near the image's side (a door frame 700 px long is straight in the camera's
  image and bows 26 px at the lens's focal length in the undistorted one), and the straight
  edges of different frames prefer different coefficients (−0.04 to −0.69); a two-coefficient
  model fitted to the matches bends them as well. Lifted through the map, such edges are
  straight within a few centimetres. A map that SfM poses keeps a pinhole camera, as before.
* **Monocular depth of out-of-focus objects at the lens.** `examples/camera`'s p05–p07 frames see a
  dark ring 0.2–0.3 m from the lens, out of focus, over a third of the image. MoGe places it
  anywhere from 0.2 m to the depth of the wall behind, differently in each frame, so keyframes
  that share it disagree by 10–20 % however their depth is scaled. A keyframe's near field
  (closer than half its median depth) whose outline is out of focus — at least a quarter of its
  edges twice as wide as the image's typical edge — therefore has no depth
  (`reconstruction.defocus`: the ring's outlines have 41–77 % such edges, near surfaces in focus
  0–5 %; a sharp rendering or a uniformly blurred frame has none). It is not in the cloud, and
  the ring's two pairs no longer disagree by 13–14 %. Where MoGe places the ring as far as the
  wall (p06_mid), it is not near and stays.
* **Latest wins cannot see a change that fills a keyframe's far field.** A camera turning in
  place sees the corridor beyond `examples/camera`'s door in several keyframes, and their
  monocular depth there disagrees far beyond the 10 % latest wins allows: the keyframes panned
  beside the door place the corridor at a third of the depth the keyframe facing it gives, which
  reads as a new surface in front of it. With the 10 % margin the corridor floor kept about
  1,300 points of the 35,000 it has without latest wins, and the keyframes facing the door kept
  6–29 % of their far pixels. On a keyframe's far field (beyond 1.6 times its median depth) the
  margin is therefore at least what the two keyframes disagree there on most of it (90 % of their
  far samples): the corridor floor keeps about 25,000 points and those keyframes 90–99 % of their
  far pixels (`ainex-captures` and `office_sequence` are unchanged). A change that fills a pair's
  far field — a door closed on the corridor while the camera turns — is that disagreement
  itself, and latest wins does not see it; nearer changes (a cup taken away) keep the 10 %.
* **Absolute scale of a map without SfM depth.** A map posed by the multi-view fallback takes
  its metric scale from MoGe's depth of its first keyframe (the median over its keyframes
  agrees within 1 %). On `examples/camera` that scale is about 15–20 % small (ceiling 2.16 m,
  door top 1.7 m, desk 0.63 m), and it depends on the image MoGe sees: its depth of the raw
  frames is 0.99–1.72 times that of their undistorted images (median 1.3), frame by frame, so
  neither gives a better metric scale than the other. Proportions hold (door top over desk 2.7).


## Development

```sh
uv run pytest -m "not models" -q --cov=oh_my_slam   # offline suite (stub models); fails below 100 % line and branch
OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m models   # real models; inference server running
uv run ruff check . && uv run mypy src && uv run lint-imports
uv run python -m oh_my_slam.service.skill                # regenerate SKILL.md
```

Every COLMAP, mapping or model run goes through the machine lock (`~/oh-my-slam-data/bin/heavy`).
The offline suite runs the shipped inference server process with deterministic stub models
(`tests/fakes/stub_models.py`, started by `tests/fakes/stub_server.py`) and the service with a
mapper double (`tests/fakes/service.py`). The mapping end-to-end tests are skipped when `colmap`
is not installed.

| Variable | Effect |
|---|---|
| `OH_MY_SLAM_RUNTIME_DIR` | Replaces `~/Library/Caches/oh-my-slam` (the inference server's socket, log, state, scratch): a second service can run its own inference server. |
| `OH_MY_SLAM_INFERENCE_CMD` | Replaces the inference server's command line that `server.sh` starts (the tests' no-op). |
| `OH_MY_SLAM_TEST_REAL_SERVER=1` | Lets the `models` tests use the running real server. |
