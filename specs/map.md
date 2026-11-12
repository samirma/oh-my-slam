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
  * frames whose pairwise two-view parallax (above) is below 1 px, over measurable pairs
    (below), form one standpoint group,
    registered about a common centre. The group fixes only that centre: each frame keeps its
    own image-only rotation, which the group never overrides. The common centre is the
    overlapped stored keyframe with the most pair inliers (ties: the lowest keyframe id), or,
    when they overlap none, the mean of their image-only centres; its position sigma is their
    spread;
  * **Measurable pair.** A pair of frames is measurable only with at least 100 epipolar
    inliers and an overlap (§1, Overlap) of at least 10 %. For a frame's image-only pose against
    its references, both must hold with the same confirmed or registered reference. Otherwise
    it counts as no measurable pair, which passes every pair limit (§4). Standpoint grouping and
    the pair limits use this threshold.
    * The epipolar inliers are those of a relative pose that the pair's own depth images
      confirm when both frames carry depth; repetitive or symmetric content can make the
      largest consensus a false geometry, and a pair with no confirmed pose counts as no
      measurable pair. This depth confirmation is the same test that registration applies to
      candidate poses on every engine but the older mapper (§2, Match graph), so a lattice pose
      can be neither measured nor registered. Its agreement threshold is a constant in
      `README.md`. It never uses the depth of a `floor_not_judged` keyframe (§2, the references
      bullet): against such a reference it is the case of a reference without depth.
    * **Depth agreement.** Points of one view's depth image are mapped into the other view under
      the candidate pose. A point is a contradiction only if the other view would see through
      it: it lies in front of the surface that view measures along that ray. A point behind
      that surface (occluded) is no contradiction. A point outside the other view's image or
      depth range counts neither way. The agreement is measured over the points that count.
    * **Symmetric scenes.** When the scene itself is symmetric, depth cannot reject a lattice
      pose: both poses agree with the depths. When two distinct pose hypotheses of a pair,
      with rotations at least 10° apart (a constant in `README.md`), both pass the depth
      confirmation, the pair takes the one whose predicted two-view parallax (above) matches
      the measured parallax of the pair's matches, over their largest consensus, as for
      translational twins (below), but only when that measured parallax is plausible for the
      pair: not above the largest two-view parallax that the hypotheses predict, each from its
      own baseline and the depths, times a margin (the parallax plausibility margin, a constant
      in `README.md`). A measurement far above what the baselines and depths allow is a false
      consensus, not evidence. When the parallax cannot separate them, within its noise (a
      model in `README.md`, as below), or is implausible, the pair is ambiguous and counts as no
      measurable pair, and registration must not choose between them on that pair alone. This
      applies to measurement and, the same way, to the registration gate (§2, Match graph,
      step 2).
    * **Translational twins.** On a plain wall and floor, a pose with the true rotation and a
      shifted position can agree with both depth images as well as the true one. When two
      depth-confirmed hypotheses share the rotation (differ by less than 10°) but their
      positions are at least 0.3 m apart (a constant in `README.md`), the pair takes the one
      whose predicted two-view parallax (above) matches the measured parallax of the pair's
      matches: a turn in place predicts about 1 px, a shifted pose hundreds. Only when the
      parallax cannot separate them, within its noise (a model in `README.md`), is the pair
      ambiguous: it counts as no measurable pair, and registration must not choose a pose on
      it alone. This applies to measurement and, the same way, to the registration gate (§2,
      Match graph, step 2), alongside the rule above.
    * **Cycle consistency.** The cycle-consistency rule applies to every pair: a pair whose
      two-view rotation disagrees by more than 10° (a constant in `README.md`) with the chain
      of stronger measured pairs is not measurable. Where both frames carry depth it applies
      after the depth confirmation, because a scene can be symmetric for depth too: in an empty
      rectilinear room each view of a narrow camera sees the floor and one flat wall, and a
      pose turned by 90° and shifted agrees with both depth images exactly, while the true pose
      of a pair turned by more than the field of view shares no view. A chain composes the
      two-view rotations along a path of measurable pairs that each have strictly more inliers
      than the pair. The rule is independent of the joint solve (§4). It cannot catch a lattice
      geometry that stays consistent around a whole cycle, or a pair with no stronger chain:
      an accepted degenerate case, as above. Each frame of a measurable pair gives the pair a
      verdict: a frame with two or more measurable pairs, none of which has a chain of stronger
      pairs to be judged against, puts each of those pairs outside; a frame whose only
      measurable pair has no chain (for example either frame of a two-frame job, or a frame
      joined by one pair), or a frame with some chained and some unchained pairs, keeps its
      pairs. A pair is outside the pair limits (§4) only when **both** its frames put it
      outside; it then neither passes nor fails the rotation check, and it never rejects the
      job on its own. Otherwise the pair stays measurable and is judged against the committed
      rotations, so a sequence measured by its adjacent pairs only stays judged. Depth
      confirmation does not change this. A false lattice pair kept this way can cause a
      rejection. A whole sequence registered through a lattice pose, which every measured pair
      agrees with, is left to the depth-consistency limit (§4).
      * **Stored-to-stored pairs.** When the rule needs a chain for a pair that involves a
        stored keyframe, including a standpoint join of a new frame to stored keyframes, the job
        measures the stored-to-stored pairs that chain needs. Each is measured from the two
        stored keyframes' own images as a two-view fit, with the depth confirmation where both
        carry depth, under the same rules as any measured pair (this rule included, which ends
        because a chain uses only strictly stronger pairs). The measurement never uses the
        stored or committed poses, and the chain never falls back to committed rotations: a
        pair whose chain still cannot be measured is the degenerate case above. A measurement
        is cached per version and reused across jobs while both stored frames are unchanged.
        The seconds spent are reported as the job's `timings_s.stored_pairs` (api.md §4);
  * a rotation-only hypothesis takes the standpoint's position sigma, and the position test
    (api.md §4) uses it;
  * the committed relative rotations of overlapping neighbours must agree with their
    image-measured relative rotations: a hard limit (§4, `rotation_inconsistent`). With priors
    this limit does not apply; the prior-rotation correction below covers it;
  * sigmas stay honest: the containment targets (evaluation.md) apply.
* **Every engine.** These apply to every engine, the older mapper included: calibrated sigmas
  (every engine's sigmas mean the same thing), the `rotation_inconsistent` limit (§4), the
  robust drift estimate across versions, drift-grown sigmas for kept frames, acceptance of an
  update whose frames are all `kept_at_prior` (§4), and the too-close content exception
  (api.md §4). The older mapper is exempt from
  static-content exclusion, standpoint grouping and the match graph (§2) only. Its
  registrations come from its own internal solve and are not depth-gated (the depth
  confirmation of §1, Measurable pair, and §2, Match graph, step 2); this is a documented
  limit, which `README.md` records.
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
  than 3 × hypot(σ_rot,a, σ_rot,b) of the two frames' prior rotation sigmas (`metric` and
  `drifting` priors alike), the rotation is corrected from the images. `quality` always
  reports `prior_disagreement_deg`: the largest such disagreement over the update's frames, 0
  when none.
* **Prior rotation sigma.** A prior's rotation is weighed by `sigma_rot_deg`, for all three
  axes. When the prior also gives `sigma_tilt_deg` (oms/2.4), pitch and roll are weighted by it
  and yaw by `sigma_rot_deg`: a robot knows pitch and roll from gravity whatever its yaw
  drift. This holds on every engine, wherever the prior's rotation is used: the conflict bound
  (api.md §4), the pose refinement, the drift estimate and the rotation check above. The
  reported `sigma_rot_deg` stays 3-D RMS (api.md §4).
* **Drift estimate.** The drift estimate that places `drifting` priors is robust:
  * it is fitted over image-confirmed frames with `drifting` priors, each image-only pose weighed
    by its calibrated sigma against the prior as sent by the prior's own sigma, so a correction
    the priors' sigmas do not allow is not made;
  * a frame whose residual exceeds 3 × sqrt(σ_prior² + σ_hyp²) is left out of the fit and is
    rejected with `prior_conflict` (when its image-only pose meets the measurable-pair threshold;
    otherwise `kept_at_prior`, `conflict_unsupported`);
  * a frame can support a first estimate or a change when its image-only pose, per frame and
    across all its references, has at least 100 inliers against confirmed keyframes and an
    8 × 8 overlap of at least 10 % with at least one of them, even if it conflicts under the
    current estimate;
  * a first estimate, when the map has none yet, and a change of the estimate between versions
    larger than 3 × its own sigma, both need the same robust support: min(3, n) supporting
    frames and at least 50 % of n, where n counts every `drifting`-prior frame of the job that
    can support it, whatever its final status. One strong frame alone is never enough;
  * a refused change keeps the previous estimate, and each unsupported frame is rejected with
    `prior_conflict` (when its image-only pose meets the measurable-pair threshold; otherwise
    `kept_at_prior`, `conflict_unsupported`);
  * a refused first estimate leaves the drift at the identity: priors are used as sent, with
    their sigmas. Only frames that conflict with their priors under the identity are rejected
    with `prior_conflict`; frames consistent with their priors keep their normal status;
  * after a change is accepted, the job's frames are re-tested under the new estimate;
  * **Drift-grown sigma.** A frame placed through the drift estimate has the root sum of
    squares of its prior's sigma and the drift estimate's own sigma at that pose. The drift
    estimate's sigma includes its extrapolation away from the frames that supported it: it
    grows with the distance from the frame's prior position to the nearest supporting frame's
    prior position, at the rate recorded in `README.md`, and is never only the fit residual at
    those frames. Travel since the client's last fix is the client's part: `drifting` priors
    already carry sigmas grown with travel. This holds on every engine.
  * **Supporting frames.** The supporting frames are the inliers of the fit that produced the
    current drift estimate, plus every `drifting`-prior frame that images confirmed under that
    same estimate, the job's own registered `drifting` frames that were not fit inliers
    included. The inliers of a fit that this job changes support at once, because they define
    the estimate. The job's other confirmed `drifting` frames, the non-inliers, join the
    support only from the commit of the version that registered them, never within the job that
    registered them, so no frame loosens its own bound; this holds on every engine, also when
    the job's drift change was refused. The fit's outliers never join the support. The set has
    no count bound, and resets to the new fit's inliers whenever the estimate changes. The
    no-count-bound rule applies only to the supporting frames: odometry-scale evidence stays
    capped by a constant recorded in `README.md`.
* **Static image content.** Features whose image position stays within 2 px across at least 3
  frames of a job whose relative rotation exceeds 5° are excluded from matching. A job without
  such rotation keeps all its features.
* **Scale.** The scale comes from the first available of:
  1. `metric` priors (`scale_source: priors`);
  2. depth images (`depth`);
  3. camera heights above the floor, per frame or per camera (`camera_height`). A frame the
     floor check did not judge (`floor_not_judged`, §2) gives no such evidence from its own
     depth or floor plane: its sent height constrains only its own camera pose, and is never
     combined with that frame's depth into a scale estimate, since that depth is the
     out-of-bound part. Its height may still count where the evidence is formed from poses
     and other frames' floor;
  4. monocular depth (`monocular`).

  `scale_sigma_rel` is estimated from the residuals of that source, from image-confirmed frames
  only (§2), and never from the depth of a `floor_not_judged` keyframe, whether as the frame's
  own evidence or as a reference's for a later frame (§2). It is one sigma: the relative error of any horizontal distance, in every
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
    (§6) until it is registered, with one exception: a frame whose images show a conflict with
    its prior beyond the conflict bound (api.md §4; `detail` `conflict_unsupported`) counts for
    nothing until a later job's images confirm it: no detections, no depth, and no free-by-rule
    cells either (no camera stand, no near blind disc), because the prior is exactly what is in
    doubt and content never enters the map at a pose the images contradict. This holds whatever
    the frame's reason: also for a frame kept for a failed check (`plain`, `blur`,
    `few_features`, `too_close`) whose image pose conflicts, the failed check winning the
    reason. A conflict is evaluated only when an image-only pose exists for the frame. If a
    later job's images confirm it, it counts normally from then on (re-registration, below).
    Every other `kept_at_prior` frame (no image pose, a loose image pose, a failed check whose
    image pose shows no conflict) keeps counting for free-by-rule cells. A frame
    whose content is withheld but whose pose is trusted still counts for free-by-rule cells:
    a sensor-depth frame excluded by the floor check or the height gate, and a registered or
    provisional monocular frame the floor check did not judge (`floor_not_judged`, below), which
    gives no cloud, occupancy, heights or objects either. A `floor_not_judged` frame's depth
    gives no scale evidence, and its sent camera height constrains only its own camera pose
    (never combined with its own depth or floor plane into camera-height evidence or k); its
    pose counts like any registered frame's for everything that uses poses. Nor does its depth
    serve any later frame (oms/2.6): see the references bullet below. Exception: a frame
    kept only for `too_close` with a `metric` prior and no conflict with it (`detail` null)
    contributes its detections and its in-range depth at the prior pose (api.md §4), but never
    scale evidence.
  * **Height gate.** When frames carry `height_above_floor_m` and `height_sigma_m`, a frame
    whose committed camera height disagrees with it by more than
    3 × sqrt(height_sigma² + σ_z²) (σ_z = `sigma_pos_m` / √3, its committed position sigma
    being 3-D RMS) contributes
    nothing to occupancy, heights or objects. Its registration status is unchanged, it still
    counts for free-by-rule cells (§6: its pose is trusted), and
    `quality.height_inconsistent` lists it by name. The gate is evaluated on the frame's
    image-only pose, before the height term of the registration below, so that it still
    measures a disagreement; where the frame has no such term, on the committed pose.
  * **Sent height in registration** (oms/2.4). When a frame sends `height_above_floor_m` with
    `height_sigma_m` and the map has a floor (§6: `floor_z_m` from priors or a camera height,
    the base version's, or on a first update the floor those define), the height is a
    z measurement in the pose refinement: the camera's z is the floor + the height, weighted by
    hypot(`height_sigma_m`, the floor's own sigma). The floor is z = 0 when priors fixed the map
    frame, unless the observed plane nearest it lies beyond max(0.06 m, 3 × `floor_sigma_m`)
    (§6); else it is `floor_z_m`. Under the z = 0 convention the floor's own sigma is 0, so only
    `height_sigma_m` counts. With `floor_z_m` (the observed plane beyond the band, or no priors
    fixing the frame) it is `floor_sigma_m`, and when that plane is measured on monocular depth
    it also carries `scale_sigma_rel` × the height (root sum of squares). The
    `scale_sigma_rel` term never applies to the z = 0 convention. The term adds to the frame's
    image evidence and its prior, and its effect shows in the committed pose and sigmas, which
    stay calibrated (api.md §4).
  * **Floor check** (oms/2.4). A frame with a sent `height_above_floor_m` whose own depth,
    at its committed pose, holds a measurable floor plane (a large horizontal plane below the
    camera, the one nearest the expected height) has its own camera height h_f above that plane.
    With H the sent height and `height_sigma_m` its sigma:
    * a frame with a sensor depth image whose |h_f − H| exceeds
      3 × sqrt(height_sigma² + (σ_depth · H)² + σ_z²) contributes no content: no cloud points, no
      grid evidence (no occupancy, no free space from its depth, no heights) and no objects. It
      is a registered frame with a trusted pose and still counts for free-by-rule cells (§6).
      σ_z is the gate's; σ_depth, the relative depth sigma, is a constant in `README.md`;
    * a monocular frame gets its own depth scale, k · H / h_f, with k the map's monocular depth
      scale, in place of k for its cloud points, its grid evidence and its objects. If that
      scale differs from k by more than the monocular scale bound (a constant in `README.md`,
      the relative bound on one monocular frame's depth scale), the frame is not judged for
      registration and pose: it keeps k, is never rejected for it and is not listed in
      `quality.floor_inconsistent`. Its content is excluded all the same (oms/2.5): no cloud
      points, no grid evidence (no occupancy, no free space from its depth, no `height_m`) and
      no objects, since its depth has no scale it can be placed with. Its input status carries
      no `depth_scale` and has `detail` `floor_not_judged`, so the client knows the content of
      that view is missing and can retake it. Like any registered frame it still counts for
      free-by-rule cells (camera stand and near blind disc, §6): its pose is trusted, and its
      pose counts like any registered frame's for everything that uses poses. Its depth gives no
      scale evidence: it does not enter the monocular depth scale k. Its sent height constrains
      only its own camera pose (the height term of the registration above and the height gate):
      it is never combined with the frame's own depth or floor plane into a scale estimate,
      neither camera-height evidence (§1) nor k, because that depth is the out-of-bound part.
      The height may still count where camera-height evidence is formed from poses and other
      frames' floor. Its depth also lifts no later frame (oms/2.6): the keyframe stays a
      registration reference for image matching and pose, its features used without its depth
      (the rule is in the references bullet below).

    A frame excluded by the floor check (sensor depth) is listed in `quality.floor_inconsistent`,
    by name. A monocular frame is never listed: one whose own scale was applied reports that
    scale as `depth_scale` in its input status, and one beyond the bound is not judged: no
    `depth_scale`, `detail` `floor_not_judged`, and its content is excluded as an excluded
    frame's is. A frame without a measurable floor plane, without a sent
    height, or without a map floor is not checked. Excluded frames are still registered: their
    status, pose and sigmas are unchanged, and only their content is withheld (a monocular frame
    that was not judged is withheld the same way). Their pose is trusted, so they still count
    for free-by-rule cells (§6). The check
    comes before the floor estimate (§6), which is made from the kept frames only.
  * Only image-confirmed keyframes are registration references for new
    frames; `kept_at_prior` frames of earlier versions are not, nor are frames that end the job
    kept. Within a job, a job frame counts as an edge once it is registered (match graph,
    below).
  * **A `floor_not_judged` keyframe as a reference** (oms/2.6). An image-confirmed keyframe whose
    `detail` is `floor_not_judged` (its own depth scale lay beyond the monocular scale bound,
    above) stays a registration reference for image matching and pose: its features are used,
    without its depth. Its pose is trusted, its depth is the out-of-bound part, so that depth
    lifts no later frame:
    * no keypoint depth for metric lifting or alignment of a frame registered against it;
    * no depth confirmation (§1, Measurable pair; step 2 of the match graph below) of a
      candidate pose against it;
    * no scale evidence for any later frame: not the alignment residual, not camera-height
      evidence, not the monocular depth scale k (§1, Scale).

    A frame whose only references that carry depth are `floor_not_judged` ones is a frame with no
    depth reference: the case of a reference without depth, which §1 (Measurable pair: the
    depth confirmation applies only where both frames carry depth) and the match graph (step 2:
    only against a reference that carries depth images) already define. Its pose comes from the
    images alone, it gets no depth confirmation and no scale evidence from those references, and
    the update's scale source is chosen from the evidence that remains (§1, Scale). Locate
    follows the same rule (api.md §6).
  * **Re-registration.** A stored `kept_at_prior` frame is eligible when a new registered frame of
    the job covers at least 10 % of its image (the overlap of §1, with the kept frame as the
    reference) and matches it with at least the registration minimum of inliers (the `README.md`
    constant, currently 15). The service then re-runs that stored frame's registration against the
    confirmed map. A frame kept for a failed check is eligible too. It is confirmed by its images
    only, never by depth, and only when its fitted pose passes the prior checks (api.md §4);
    otherwise it stays `kept_at_prior` and is not listed in `quality.reregistered`. If its images
    confirm it, its pose leaves `kept_at_prior` in that version (it becomes a registered pose,
    `provisional` then `fused`), and from then on it counts normally: it feeds occupancy,
    heights, the cloud, objects and free-by-rule cells (a frame kept with `detail`
    `conflict_unsupported`, whatever its reason, counted for none of them until then). The job lists each such frame in `quality.reregistered` (§4); the movement of that
    confirmation is excluded from `previous_residuals` and `gauge_shift`. A dry run predicts
    `reregistered` with the same test, without committing; the prediction is approximate. An engine
    that cannot re-register (the older mapper) reports `reregistered` as an empty list; this is
    allowed. Re-registration never rejects or merges keyframes, even within the duplicate limits.
    Nothing re-registers a frame outside a job: there is no automatic background refinement;
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
       priors of all its frames together, or both. On every engine but the older mapper (§1,
       Every engine), a candidate pose of a frame that carries depth images, against a
       reference that carries them (not a `floor_not_judged` one, whose depth is not used:
       above), is placed only if it passes the depth confirmation of §1
       (Measurable pair); a candidate failing it is not a placement. When two distinct candidate
       poses of a frame against one reference both pass it (§1, Symmetric scenes), the pair is
       ambiguous and that reference alone places the frame in neither, unless the parallax
       separates them. Two that share the rotation but lie far apart (§1, Translational
       twins), and two whose rotations are at least 10° apart (§1, Symmetric scenes), are
       separated by their parallax first: the one whose predicted parallax matches the measured
       parallax of the pair's matches is placed, the second case only when that measured
       parallax is plausible for the pair (§1, Symmetric scenes). The pair is ambiguous only
       when the parallax cannot separate them or, for rotations at least 10° apart, is
       implausible; a prior sent for the frame still decides first, within 3 × its position
       sigma. A component that reaches
       no confirmed keyframe is seeded at one frame's prior. Its frames' reported sigmas cover
       the seed prior's sigma combined with each frame's fusion, and are never tighter than the
       seed prior's. This floor applies to every frame whose placement depends on the component,
       whatever the engine or pass (MapAnything, a retry, re-registration), until the
       component links to a confirmed keyframe.
    3. The prior checks (`prior_conflict` with the combined bound, and the looseness tests, api.md
       §4) apply frame by frame to the component's fitted poses. A frame that fails a looseness test
       is `kept_at_prior` with `detail` `loose_image_pose`, whatever else it fails. Otherwise, a
       frame beyond the conflict bound is rejected with `prior_conflict` when its image-only pose
       meets the measurable-pair threshold (at least 100 inliers and 10 % overlap, both with the
       same confirmed or registered reference, §1), and is otherwise `kept_at_prior` with `detail`
       `conflict_unsupported`. The rest of the component stays registered if it is still connected
       and placed without that frame. Removals iterate to a fixed point, re-checking connectivity
       after each removal, and the final statuses do not depend on the order in which frames are
       removed.
    4. A frame is kept with `no_overlap` only if its component cannot be placed at all. A lone
       frame, with no edge at the registration minimum, cannot be placed: with a prior it is
       `kept_at_prior`, reason `no_overlap`, `detail` `no_image_pose`.
    5. `kept_at_prior` frames of earlier versions are not references, and a job frame counts
       as an edge only once it is registered. An earlier kept frame that a registered frame of
       the component covers at least 10 % of (8 × 8 coverage of the kept frame's image) and
       matches with at least the registration minimum of inliers goes through re-registration
       (above): confirmed by images only, never by depth, and only when its fitted pose passes
       the prior checks. So the same job can confirm it.
    6. **Pose duplicates.** After the registration fixed point, one pose-duplicate pass covers
       every frame registered in the job, in canonical name order; of a pair, the later-named
       frame is the duplicate (api.md §4). A stored kept keyframe re-registered in this job
       counts as a frame of this job for the pass, not as a stored confirmed keyframe. If a
       frame the pass rejects served as a reference for other frames, as the drift reference,
       as scale evidence, or for a re-registration, the job's registration fixed point runs
       once more without it, so none of its effects remain:
       * that second run happens at most once per job; a duplicate found in it is rejected
         without a further run;
       * a frame rejected as a duplicate keeps its first `duplicate_of`, even if its partner
         ends differently in the second run;
       * if the later member of the pair is a stored kept keyframe that the job re-registered,
         its re-registration is undone (it stays kept as stored) and the second run skips
         it.
    7. The outcome does not depend on the order of frames in the request. Where any order is
       still needed (the origin frame, the seed frame, the pose-duplicate pass), frames are
       taken in canonical name order. A dry run applies the same procedure and predicts the
       same statuses, except for monocular frames and frames the geometric engine leaves
       unplaced (item 8), which a job may place with the MapAnything engine: it predicts
       them with geometric registration, so their statuses and poses, and the
       `rotation_pairs` they enter, can differ (api.md §4).
    8. **Engine order in a job.** Frames that carry depth images register with the geometric
       engine first, as a dry run does. MapAnything takes the monocular frames and the frames
       the geometric engine leaves unplaced. A MapAnything call with a single anchor has no
       anchor residual to estimate its rotation noise from, so it reports `sigma_rot_deg` no
       tighter than a rotation sigma floor, a constant in `README.md`, calibrated so that
       `update.sigma_containment` (evaluation.md) holds;
  * the objects;
  * the grid inputs;
  * the scale;
  * the label vocabulary (§5);
  * the metadata.
* **Ignored pixels.** A pixel in the union of the camera's ignore mask and the frame's own mask
  contributes nothing anywhere, on every engine, in every use. The list is examples, not the
  whole set: no keypoints or matches, no depth points, no occupancy, free-space or floor
  evidence, no detections or votes, no cloud points, and also none to the frame quality checks
  (`plain`, `blur`), the `too_close` scene distance, scale evidence (depth residuals, monocular
  depth) or the camera-height check. A detection is dropped when its share in the
  ignored region exceeds `IGNORED_DETECTION_MAX_SHARE` (`README.md`, 0.5): the share of its
  mask, or of its box when it has no mask. On every engine, a view where the object lies mostly
  in ignored pixels, by that same share, is not a view of it (§5, Ignored pixels).
* **Reads.**
  * A read without `?version=` answers from the latest committed version at the moment the
    request arrives.
  * Each answer comes from one version and names it in `map_version`, even while later
    versions are committed.
  * A version that is no longer kept answers `version_gone`.
* **Revert.** `revert {to_version}` commits a new version whose content equals `to_version`.
  History stays linear and readers never see a version change in place.
  * It does not roll back the map's camera registry: a camera id, once defined, stays known
    with its values, and camera ids are never freed. The map summary's `cameras` lists the
    registry as of that version's commit, so a revert version lists every camera defined so
    far, including those defined after `to_version`.
  * Object ids and surface ids are unique across the map's whole version history and never
    reused, even after a revert: an object or surface created later never takes an id held in a
    version that was reverted away. When a revert's target has no stored mapper map, the map
    records an id high-water mark for objects and surfaces, and the next mapper map numbers from
    it, so ids never repeat. `first_seen_version` and `last_seen_version` are version numbers of
    that history.
  * It reuses the target's derived reads (grid, cloud, locate index) instead of re-deriving
    them, so `revert.latency` (evaluation.md) holds.
* **Retention (`--keep-versions`).** The service keeps at least the last *N* versions of each
  map (default 20, recorded in `README.md`). Older ones are deleted, except a held one.
  * **Holds.** A job holds its base version, the version its registration ran against. A read
    holds the version it answers from, a dry run or a locate with `version` included, until it
    ends. A held version does not count toward *N*: a held version older than the newest *N* is
    kept in addition. It is deleted at the first commit after its last hold ends.
  * **Sharing.** Kept versions share the parts a version does not change: the mapper's map and
    the derived reads (grid, cloud, locate index). A version costs only what changed.

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
  * An object id or surface id persists for the lifetime of the map and is never reused, not
    even after a revert (§2).
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
  * `floor_inconsistent` (oms/2.4): the names of the frames the floor check (§2) excluded from
    content, by the sent height against the frame's own floor (frames with a sensor depth image
    only; a monocular frame is rescaled or not judged, never listed here, though a frame not
    judged gives no content too); empty when none. A frame may be in both lists;
  * `drift_correction` (oms/2.5): `{translation_m, rotation_deg, odo_scale}`, the drift
    estimate's correction at the job's frames, from the prior as sent to the prior as placed (§1,
    Drift estimate): the largest translation and the largest rotation, over the job's frames
    with a `drifting` prior, between a frame's prior as sent and as placed, and the estimate's
    odometry scale (map metres per odometry metre, 1 without scale evidence). The estimate is
    the one the job's frames were placed with, after any change the job accepted. On a dry run,
    the estimate it would place them with. Null when no frame of the job has a `drifting`
    prior; the identity estimate gives `{0, 0, 1}`. It tells a client how much of a frame's
    `prior_residual_*` (api.md §4) is the estimate moving the prior and how much is the images;
  * `rotation_pairs`: every measurable pair (§1) the rotation check evaluated, on dry runs as
    on jobs, each `{a, b, inliers, overlap, measured_deg, committed_deg, difference_deg}`: the
    frame names in canonical order, the pair's epipolar inliers, its 8 × 8 overlap, the
    image-measured and the committed relative rotation angles, and the angle between those two
    rotations. Empty unless the check ran on at least one measurable pair; it shows which pairs
    drove `rotation_inconsistent`;
  * `depth_pairs` (oms/2.6): every pair the depth-consistency limit (below) evaluated, on dry
    runs as on jobs, each `{a, b, agree_ratio_ab, agree_ratio_ba, fails}`: the frame names in
    canonical order; the share of points that agree among those that agree or contradict, in
    each direction (a's depth points mapped into b, then b's into a), null for a direction with
    fewer than the depth confirmation's minimum count of such points; and whether the pair
    failed the limit. Empty when the limit evaluated no pair (no sensor depth, or the older
    mapper). On a `depth_inconsistent` rejection it shows which frames contradict each other,
    so a client can drop or retake them;
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
      solve: the measurement is never seeded from, nor refined toward, the committed or the
      joint solve. It uses the pair's images, its depth images when both frames carry them,
      and the cycle-consistency rule (§1, Measurable pair), nothing else.
  * **Depth consistency** (every update whose frames carry sensor depth images, with or without
    priors: wide drifting priors do not exclude a 90° lattice pose): a hard limit rejects the
    update, `failed_checks` holding `depth_inconsistent`, when two committed views contradict
    each other's depth images. A dry run predicts it, as it does the other limits.
    * The pairs: the update's registered (image-confirmed) frames with each other and with the
      stored image-confirmed keyframes, where both carry sensor depth images and the two can
      share a view under their committed poses (the test is in `README.md`). A frame kept at
      its prior without image confirmation is not paired.
    * Each view's depth points are mapped into the other under the pair's **committed**
      relative pose (§1, Depth agreement). A point is a contradiction only when the other view
      would see through it by more than the depth tolerance of the depth confirmation (a
      constant in `README.md`) plus three times the two frames' combined committed position
      sigma.
    * A pair fails under the depth confirmation's minimum count and agreement threshold (§1,
      Measurable pair), in either direction: at least that minimum of points agree or
      contradict, and fewer than that threshold of them agree.
    * It uses the committed poses and the depth images only, never the pairs' image
      measurements, so it catches a commit that every measured pair agrees with: a sequence
      registered through a lattice pose (an empty rectilinear room with a repeating wall
      texture, a narrow camera) agrees pair by pair, but its views contradict the frames they
      were never measured against.
    * Sensor depth only: a pair with a monocular frame is not tested, as its depth scale is
      estimated, not measured.
    * A scene that is symmetric as a whole, where the committed depths agree everywhere,
      remains the accepted degenerate case.
    * Every engine but the older mapper (§1, Every engine), whose registrations are not
      depth-gated.
  * Pair limits (rotation consistency and any other limit on pairs of frames) are measured on
    registered frames only, over measurable pairs (§1: at least 100 epipolar inliers,
    depth-confirmed where both frames carry depth and consistent with the chain of stronger
    pairs, as there, and an overlap of at least 10 %). With no measurable pair, every pair limit
    passes.

## 5. Objects (O8)

Each object record [`MapObject`] has the fields below; frame analysis returns the subset
[`AnalysisObject`] (api.md §7). Every field is present in an objects read, except `label_scores`,
where its row allows omission.

| Field | Meaning |
| --- | --- |
| `id` | Positive integer, unique across the map's whole version history and never reused, even after a revert (§2). |
| `label`, `detected_as[]`, `score` | Label chosen (Label choice, below), labels it was detected as, and confidence. |
| `label_scores[]` | Optional (oms/2.4). Up to the 3 best label candidates of the record, `{label, score}`, in the order of the label choice (below): the first is `label` (whose score is always above 0, §5 Label choice), the rest follow by score, highest first. `score` is the label's `detection_votes[].share` (same count, same denominator, same ignored-pixel rule), in 0–1, so the scores sum to at most 1. Present whenever the version's label vocabulary holds client labels (§5, Label vocabulary); otherwise it may be omitted. A label with no share is no candidate: the list holds fewer than 3 entries when fewer labels got votes. Map objects only; frame analysis does not carry it. |
| `detection_votes[]` | `{label, share}`, highest share first: `share` is the number of frames that gave that label over the number of image-confirmed frames whose view frustum contains the object's box centre and that gives a view of it (Ignored pixels, below), detected or not, plus every confirmed frame that detected the object even with the box centre outside its frustum (`kept_at_prior` frames excluded, except a frame kept only for `too_close` with a `metric` prior, which counts as a confirmed frame here, api.md §4), so shares sum to at most 1 and may sum to less. For a map built by the older mapper, shares come from that mapper's per-label vote counts over its observing frames. Map objects only; frame analysis does not carry it. |
| `centre_m`, `size_m`, `yaw_deg` | Upright oriented box in the map frame: z is up, x is the longer horizontal side. |
| `sigma_centre_m` | One-sigma uncertainty of the centre. When the map's builder never recorded which views bound the box, it comes from the formula documented in `README.md`. |
| `extent_seen {x, y, z}` | `full` or `partial` per box axis, from the views that bound it. A view of the object's top seen uncut (an elevation threshold, a constant in `README.md`), together with a support plane measured under its footprint, makes `z` `full`: the support plane bounds the bottom; the support may be measured from other views. The other rules per axis are unchanged. When the map's builder never recorded those views, `partial` on every axis. |
| `mean_rgb` | Observed colour: robust mean of the object's pixels in sRGB 0–255, from the views that saw it best. |
| `color` | Display colour only: a pure function of `id`. |
| `state` | `confirmed` (seen again where it was), `moved` (seen elsewhere: `moved_from_m` gives the earlier centre), or `struck` (seen gone; kept for one version, then removed). |
| `first_seen_version`, `last_seen_version` | Real version numbers of the map's history where it first and last appeared; a revert version carries the values of `to_version`. |
| `region` | The grid region (§6) of the object, or null. It is set when its centre or footprint lies in a region or on its boundary; the region chosen is the one its footprint overlaps most. An object on a wall, such as a door or a shelf, belongs to the region it was seen from. Null only when it touches no region. |
| `supported_by` | `{kind: surface \| object, id, verified_version}` for the support surface or object it rests on, or null. |
| `inside_of` | `{id, verified_version}` when its centre lies within another object's concave interior (a bowl, a drawer, a box), or null. |

`Objects` also lists `surfaces[]`: the horizontal support planes estimated from the fused
points, such as the floor, table tops, counters and shelves. Each has `id`, `height_m`,
`polygon_m` and `area_m2`. A surface `id` is a positive integer, unique across the map's whole
version history and never reused, even after a revert (§2). An arm descends to and places on
these heights.

* **Label vocabulary.** An update request's `labels` (api.md §3) are added to the map's
  vocabulary, which is stored with the version: the union of the labels of all its updates,
  equal labels kept once, in their first form.
  Detection on the map, in fusion and in any later re-detection, prompts with the default
  vocabulary (`README.md`, `default_labels.txt`) merged with the map's vocabulary, duplicates
  removed from the prompt list. Two labels are equal when their Unicode case-folded forms are
  equal after trimming, reading `_` as a space and collapsing whitespace runs to one space. A
  label equal to a default stays in the stored vocabulary and in `MapSummary.labels`: it is
  dropped only from the prompt list. The map summary reports the version's vocabulary
  read-only, and a dry run the vocabulary it would use.
* **Label choice.** A record's `label` is the engine's own choice among the labels of its
  `detection_votes` (its evidence rule, `README.md`; it need not be the highest `share`), with
  two exceptions (oms/2.4). A label with no share (share 0 or no `detection_votes` entry) is never
  chosen: when the engine's choice has no share, the best-share label is chosen instead. And a
  **client label** is a
  label equal (by the comparison above) to a label of the map's vocabulary, whether or not it
  also equals a default; every other label is a **default label**. A client label beats a default
  label whose share exceeds the client label's by no more than `CLIENT_LABEL_MARGIN`
  (`README.md`), an absolute difference of shares. Concretely, with the best-share label `L` and
  the engine's choice `E` (replaced by `L` when `E` has no share): if `E` is a client label it is
  chosen; else the client label of the highest share among those within the margin of `L`'s
  share is chosen, if there is one (so `L` itself when `L` is a client label); else `E`. The margin is applied
  once, against the best-share label, not chained. Candidates of equal share are ordered as
  before this rule. The same rule chooses the label of a merged object (`README.md`, Persistent
  identity) from the merged evidence. `label_scores` lists the candidates in this order. The
  rule changes the label only, not `detection_votes`, `detected_as[]` or `score`, and a map with
  no client label chooses as before.
* **Ignored pixels.** Detections follow §2: masked pixels give no detection or vote. On every
  engine, a view where the object lies mostly in ignored pixels (§2) is not a view of it: the
  object's share of its projected box or mask in the ignored region exceeds
  `IGNORED_DETECTION_MAX_SHARE` (`README.md`, 0.5, the share of the dropped-detection rule).
  Such a view enters no view count, no extent evidence and no label vote share, numerator or
  denominator (`detection_votes[].share`).
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

**Ignored pixels** (§2) give no occupancy, free-space or floor evidence.

**Where cameras stood is free.** The cells under each keyframe's camera position are free,
whatever the camera saw, because the robot stood there. So is the ring around that position
that the camera could not see: its near blind zone on the floor, from the camera's height and
its lower field-of-view edge.

**Unconfirmed frames.** A `kept_at_prior` keyframe counts only for these free-by-rule cells
(camera stand and near blind disc) until a later job's images confirm it (§2), except a frame
kept at a prior its images contradict (`detail` `conflict_unsupported`, whatever its reason:
`no_overlap`, or a failed check such as `too_close` that won the reason): it counts for
nothing, so it makes no camera stand and no near blind disc and gives no `floor_observed` = 2,
because the prior is exactly what is in doubt. A conflict is evaluated only when an image-only
pose exists for the frame. When a later job's images confirm it, it counts normally from then
on. Every other `kept_at_prior` keyframe (no image pose, a loose image pose, a failed check
whose image pose shows no conflict) keeps counting for these cells.
Occupancy, the `height_m` layer, the cloud and object records come only from image-confirmed
frames, except the in-range depth and detections of a frame kept only for `too_close` with a
`metric` prior (api.md §4). A frame whose content is withheld still counts for these cells,
because it is a registered frame with a trusted pose: a sensor-depth frame excluded by the floor
check or the height gate (§2), and a registered or provisional monocular frame the floor check
did not judge (`floor_not_judged`, §2), which gives no occupancy, `height_m`, cloud or objects.

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
    origin height (z = 0), and z = 0 itself unless that plane lies beyond
    max(0.06 m, 3 × `floor_sigma_m`) from it. Robots put that origin where they stand: on the
    floor for a base or a humanoid, at the base link on the worktop for an arm;
  * without priors but with a camera height (`height_above_floor_m`), it is the floor plane the
    height defines (about 0, §1), refined by the observed large horizontal plane nearest it,
    accepted when within 3 × that plane's own point spread; `floor_sigma_m` is a robust
    standard deviation of the plane's points' heights (below). The
    height used is the first registered frame's `height_above_floor_m`;
  * with neither priors nor a camera height, it is the lowest large horizontal plane seen below
    the cameras.

  **Floor across versions** (oms/2.5). On every map, whatever fixed its frame, a version keeps
  the previous version's `floor_z_m` unless the new estimate (by whichever of the rules above
  applies) differs from it by more than max(0.06 m, 3 × the new estimate's `floor_sigma_m`).

  Other planes, such as the room floor seen past a worktop's edge, are support surfaces
  (`surfaces[]`), not the floor.

  **Floor band.** A surface within max(0.06 m, 3 × `floor_sigma_m`) of `floor_z_m`, on either
  side of it, is floor: it never makes an occupied cell or defines a passage, whatever `z_min`
  is. The band classifies by the height of the surface, the height of the points that make it,
  never by the centre of the voxel that holds them: the result does not depend on where the
  floor falls in the grid's voxels.

  **Floor sigma** (oms/2.4). `floor_sigma_m` is a robust standard deviation of the heights of
  the floor points, for example 1.4826 × the median absolute deviation, not a half-width of a
  histogram peak. The points are those of the kept frames, after the floor check (§2): a frame
  the check excluded, or did not judge (`floor_not_judged`), adds none.
* Optional `footprint_radius_m`: the robot's radius.
* `height_cap_m`: the ceiling of the `height_m` layer, relative to `floor_z_m`. Default 2.0 m,
  or the band's `z_max` when the request gives one. Echoed in the header.

**Header.**
* `map_version`, `origin_m` (the corner of cell [0, 0]), `cell_m`, `shape [rows (y), cols (x)]`,
  `z_band_m`, `floor_z_m`, `floor_sigma_m` and `height_cap_m`. `z_band_m` and `height_cap_m`
  are relative to `floor_z_m`, as the request's `z_min` and `z_max` are. `floor_z_m` is the
  floor's absolute map-frame z, and `floor_sigma_m` the robust standard deviation of the floor
  points' heights (Floor sigma above), not the uncertainty of their mean;
* `scale_source` and `scale_sigma_rel`.

**Layers.** Each layer is a little-endian array, base64, row-major:
* `state`: uint8, 0 unknown, 1 free, 2 occupied;
* `observations`: uint16, the number of frames that observed the cell;
* `height_m`: float32, the highest surface in the cell at or below `height_cap_m`, relative to
  `floor_z_m`, NaN if none. Surfaces above the cap, such as a ceiling, are left out;
* `floor_observed`: uint8:
  * 1 — a floor surface was observed in the cell at the floor height;
  * 2 — not observed, but free by rule: a camera stood here, or the cell lies inside a camera's
    near blind disc. The cameras are those of registered and provisional frames (also when their
    content is withheld, §2) and of `kept_at_prior` frames, except one kept with
    `detail` `conflict_unsupported` (whatever its reason), which makes no such cell;
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
