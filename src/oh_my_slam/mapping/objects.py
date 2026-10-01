"""Persistent objects: identity across keyframes and updates, evidence accumulation, OBB refits
(through the segmentation API), confirmation, merging and removal on evidence of absence.

Semantics (spec §2.3):

* **One update, one observation.** The order of an update's keyframes never matters — except for a
  place that changed while it was captured (see *Latest wins*). All of its
  instances are grouped at once (``_group``): strongest agreement first, never two instances of
  one keyframe in one group, and an object the map already held absorbs a group only when most of
  the group's instances match it directly. A group without such an object becomes a new object.
* **Evidence is order-free.** An object's evidence — label votes, scores, the keyframes that
  detected it, one ``Sighting`` per detection, its mean viewing distance and a canonical point set
  (``canonical_points``) — is a pure function of the instances it received, so its label and OBB
  do not depend on the order or the grouping into updates in which that evidence arrived.
* **Confirmation** is recomputed from all evidence (``confirm``): detections with a reliable mask
  (not mostly in the image-border band, where monocular depth is unreliable) in >= 2 distinct
  keyframes — or in the one keyframe that had the object in view, when no other keyframe of the
  map did (``views_in_frustum``; occlusion is ignored, so a detection whose depth puts it behind
  another surface cannot confirm itself). Unconfirmed objects are kept, never exported, so a later
  update can still confirm them.
* **Boxes** are fitted (by segmentation) to the points of the sightings that agree with each
  other (``fit_points``): monocular depth of small objects varies between keyframes, and the union
  of inconsistent sightings is a streak along the viewing rays, not the object. Only the best
  evidence shapes a box (``shaping_sightings``): the detections of confidently placed keyframes
  once there are two (a keyframe of low confidence places the object offset as a whole, onto its
  neighbour), and of these the reliable ones when they are the majority; each object records
  which kind of detection each of its points came from (``sources``). A mask that bled onto the
  support in front of an object (a window's onto the windowsill) is trimmed before anything else
  (``segmentation.api.trim_support``).
* **Keyframes re-scaled by a later update.** An update adjusts the depth of every keyframe of the
  map (``mapping.api._adjust_depth_scales``: a loop it closes spreads over the whole loop); the
  objects of the stored keyframes it corrects move with them (``rescale_objects``).
* **Latest wins.** Order of addition is the only sign of "latest", within an update as across
  updates. Every object is judged by the update's keyframes added after its last detection that
  could have detected it there and agree with its detections about its surroundings
  (``_Places``: in view and unoccluded, near enough, their depth divided by the local ratio to
  the detecting keyframes, from any viewpoint and distance); seeing through it by more than the
  depth noise, the object's size-scaled margin (``absence_tau``) and the disagreement of its own
  detections — where the support at its foot, which looks the same with or without the object,
  is no evidence (``_evidence``) — removes it or gives it a strike, and an update that re-detects it or sees it in
  place clears its strikes (``_absence``). The removed object's detection masks are invalidated
  in the keyframes that detected it (``retire_pixels``), so the map cloud loses its points too,
  and its place is drawn from the keyframes that saw through it (``Vacated``). An object first
  detected where the update's earlier keyframes saw free space arrived: their views through it
  are retired, so that its keyframes draw it however few they are.
* **Moved objects.** An object that keyframes added later detect only elsewhere — re-identified by
  label, size and colour, its old place seen empty after and its new place seen empty before
  (``_moves``) — keeps its id at its new place; its old place is vacated like a removed object's,
  and the earlier keyframes' views through its new place are retired, within one update or
  across updates.
* **Ids** come from ``next_object_id`` and are never reused. It counts the map's detections: the
  detections of an update's keyframes are numbered in keyframe order, continuing the count, and a
  new object takes the number of its first detection (earliest keyframe, then the detector's
  order). Ids therefore have gaps, and they are bookkeeping only, but they depend on nothing
  except which detections form the object — not on the unconfirmed candidates, merges or removals
  of earlier updates — so a sequence mapped in one update or split over several in order gets the
  same ids wherever it associates the same detections. A merge keeps the lower id; merged and
  removed ids disappear for good (merges are recorded in ``merged_into`` so old per-frame instance
  files still resolve).
* **Split surfaces.** A horizontal surface that runs out of the view and under the items on it (a
  counter top around a book) is often split by the detector into several instances of one kind.
  Instances of one keyframe that ``split_surface`` allows and that touch with continuous depth
  (``surface_pieces``) are one instance of that keyframe (``join_instances``); pieces seen from
  different keyframes that share or continue surface at one height are merged, whatever surface
  labels they carry (a counter top labelled desk from one side and rug from another:
  ``_one_surface``).
* **Merging** joins duplicates: objects with compatible labels that overlap, and — whatever their
  labels — objects of comparable size that occupy the same space (most of either one's points on
  the other's surface) and that no keyframe detected as two instances: the detector's label
  flickered between keyframes (a door seen as a wardrobe); and a part the detector named on its
  own in the keyframes that did not name the whole (a figurine's top as a bottle opener:
  ``_part_of``). The merged object's label is the one
  with the most evidence (score-weighted votes); the others are exported as ``detected_as``.
* **Depth-explained duplicates.** The per-keyframe depth scale drifts along a long sequence, so
  the keyframes that close a loop can place an object 10-15 % nearer or further than those that
  saw it first: two copies along the same viewing rays, too far apart for the overlap tests (a
  faucet 0.4 m from itself at 3 m). Objects of compatible labels that no keyframe detected
  together merge when their sightings agree once the depth ratio measured between their
  keyframes is removed, either keyframe's (``_depth_explained``).
* **Loop copies.** Where a weakly linked stretch of a video comes back to a place, its
  keyframes can be misaligned with the first visit's as a whole (pose, not only depth): every
  object there is mapped twice, the copies offset alike. Pairs of such copies — compatible
  labels, never detected together, seen from distant viewpoints by keyframes that disagree —
  merge when other pairs between the same keyframes are displaced alike (``_loop_copies``).
* **Objects seen twice.** An object seen from two sides of a room by keyframes that never see it
  together is placed once by each set, each copy along its own keyframes' viewing rays at their
  depth (a lamp 0.2 m from itself, a door handle seen from 1.7 m and from 4.5 m), or a label
  flickers between the sets (a dock seen as a toilet from one side and as a dryer from the
  other). Two such objects merge when each set of keyframes saw the other's place as the object
  it detected there, and never as something else beside it (``_seen_as_one``).
* **Point counts** are those of the map cloud: the points attributed to the object
  (``set_cloud_counts``; its votes inside its box grown by the depth noise, and the unlabelled
  cloud points nearest its own lifted points: ``geometry``), as the
  ``point_count`` of a single image counts its cloud's points. An object is exported only with
  ``min_cloud_points`` of them — a share of the cells of its box's largest face at the map's
  sampling at its nearest detection (a cloud voxel, or a depth pixel's footprint where coarser) —
  so every exported object is visibly drawn in the cloud (``segments.ply``, ``color=segment``) in
  its colour.
* **Boxes cover the observed surface.** A box is fitted to the points the keyframes saw: an
  object seen only from the front (a refrigerator against a wall) has the depth of its visible
  surface, not its physical depth; no class-typical size is assumed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

from oh_my_slam.core import rle
from oh_my_slam.core.geometry import depth_edge_mask, project, unproject_pixels, voxel_keys
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping.store import OBJECTS_JSON, frame_file, load_valid
from oh_my_slam.mapping.validity import (
    BORDER,
    MIN_OBSERVATIONS,
    POSE_MAX_RESIDUAL_DEG,
    View,
    keyframe_view,
    pose_supported,
    stored_view,
    tau,
    well_registered,
)
from oh_my_slam.reconstruction.depth import DepthCorrection
from oh_my_slam.reconstruction.gravity import floor_candidate_height
from oh_my_slam.segmentation.api import (
    OBB,
    LiftedInstance,
    SceneObject,
    compatible,
    fit_object_obb,
    join_instances,
    lift_detections,
    obb_iou_upright,
    split_surface,
    surface_label,
    trim_support,
)

POINT_CAP = 30000
POINT_VOXEL = 0.01
IOU_GATE = 0.3
CENTROID_GATE = 1.5
# Confirmation: reliable detections in this many distinct keyframes (when another keyframe had the
# object in view). A detection is reliable unless most of its mask lies in the image-border band
# that ``View.usable`` excludes (``BORDER``): there the object is cut off and its monocular depth
# is unreliable, which is where one-off phantoms come from (a counter's edge at the bottom of a
# downward view, labelled "carpet" and placed inside the counter).
CONFIRM_DETECTIONS = 2
BORDER_EVIDENCE = 0.5  # largest share of a reliable detection's mask in the border band
IN_VIEW_SHARE = 0.5  # a keyframe has an object in view when this share of its points projects in
VIEW_SAMPLES = 200  # points per object for the in-view test
MERGE_OVERLAP = 0.5
MERGE_BOX_IOU = 0.3  # compatible objects whose boxes overlap this much are one physical object
MERGE_CONTAINMENT = 0.6
# Objects of incompatible labels are one object when no keyframe detected both, the smaller one's
# size is at least this share of the larger's (not a part of it or an item resting on it) and
# MERGE_OVERLAP of either one's points lie on the other's surface.
MERGE_SCALE = 0.5
# Box fit: two sightings agree when their bounds overlap or are at most max(CONSENSUS_TOL_MIN,
# CONSENSUS_TOL_REL · viewing distance) apart (monocular depth noise); with at least CONSENSUS_MIN
# sightings the box is fitted to the points in the bounds of those agreeing with the best one.
CONSENSUS_MIN = 3
# ... and only the sightings that may shape it take part (``shaping_sightings``): those of
# confidently placed keyframes, and of these the reliable ones, each set once it has SHAPE_MIN.
SHAPE_MIN = 2
CONSENSUS_TOL_MIN = 0.05
CONSENSUS_TOL_REL = 0.03
CONSENSUS_MARGIN = 0.02
REMOVE_FRACTION = 0.6
REMOVE_FRACTION_FEW = 0.7
REMOVE_MIN_FRAMES = 3
FEW_MIN_INLIERS = 100
VISIBLE_SHARE = 0.5  # of an object's samples, unoccluded in a keyframe that judges it
FEW_MIN_OBSERVATIONS = 4
# "Seen through" needs the surface behind an object's point by more than a margin. The margin of a
# large object is max(ABSENCE_TAU_MIN, ABSENCE_TAU_REL · z): monocular depth of thin or large
# surfaces is unreliable. A small object cannot be seen through by that much — a cup on a windowsill
# is seen through by 5-20 cm — so the margin also scales with the object's size, at least
# max(ABSENCE_TAU_SMALL_MIN, ABSENCE_TAU_SMALL_REL · z), the depth noise of the keyframes.
ABSENCE_TAU_MIN = 0.25
ABSENCE_TAU_REL = 0.15
ABSENCE_TAU_SMALL_MIN = 0.05
ABSENCE_TAU_SMALL_REL = 0.08
ABSENCE_TAU_SIZE = 0.15
# A multi-view pose that matches leave uncertain by more than POSE_MAX_RESIDUAL_DEG still judges an
# object at least this many times the pose's lateral error at the object's distance.
ABSENCE_POSE_SHARE = 0.15
# Latest wins for objects (``_Places``). A keyframe judges an object's place when it sees it:
# PLACE_FRAMED of its samples in the image (the image-border band counts), VISIBLE_SHARE of them
# unoccluded, on at least PLACE_MIN_PIXELS distinct pixels of its depth grid (a far or tiny object
# is a few pixels, all of them depth edges); its silence there is evidence only from at most
# PLACE_RANGE times the farthest distance a keyframe detected it from (it then appears at least
# 2/3 as large as in the smallest detection); and when it agrees with the object's detections
# about the object's surroundings: the surface in a band around the object's mask (RING_WIDTH of the mask's size, beyond
# MASK_DILATE px; RING_SAMPLES points, at least RING_MIN) in the detecting keyframe nearest by
# viewpoint (the nearest PLACE_REFS are tried), within a factor RING_SAME of the judge's depth where
# it sees the same surface; their median ratio at most RING_BIAS from 1 (the depth disagreement
# the global adjustment leaves between keyframes; validity.MAX_GLOBAL_BIAS), and VISIBLE_SHARE of
# the points within the depth noise (``absence_tau``'s floor) of it once it is removed. What its
# own PLACE_REFS largest detections' keyframes see through (its transparency) is no evidence.
PLACE_FRAMED = 0.8
PLACE_RANGE = 1.5
# ... and where it sees the object's support at the object's foot — the lifted surface within the
# absence margin of a sample lies in the object's bottom band, FOOT_BAND of its height (the bottom
# band of segmentation's support trimming) — the sample is no evidence either way (``_evidence``)
FOOT_BAND = 0.25
PLACE_MIN_PIXELS = 20
PLACE_REFS = 3
RING_WIDTH = 0.5
RING_SAMPLES = 400
RING_MIN = 20
RING_SAME = 1.3
RING_BIAS = 0.25
# Moved objects (``_moves``): an object detected only by keyframes added after every detection of an
# object of a compatible label, of comparable size (scales within MOVE_SCALE) and colour (median
# CIELab of the detections' pixels in MOVE_COLOUR_VIEWS of its keyframes: chroma within
# MOVE_CHROMA, lightness within MOVE_LIGHTNESS; lighting changes lightness more than chroma),
# standing elsewhere (their boxes grown by the depth noise do not meet).
MOVE_SCALE = 1.5
MOVE_CHROMA = 10.0
MOVE_LIGHTNESS = 25.0
MOVE_COLOUR_VIEWS = 3
MAP_FLOOR_MAX_BELOW = 0.10
MASK_DILATE = 3
PAIR_NEIGHBOURS = 10  # keyframes (nearest by viewpoint) whose instances each keyframe's meet
EARLIER_VIEWS = 2  # an existing object's detecting keyframes (nearest by viewpoint) re-checked
# Split surfaces: two instances of one keyframe are pieces of one surface when at least
# SURFACE_CONTACT_PX pixels of one lie within SURFACE_CONTACT_BAND px of the other, on valid
# pixels free of depth edges, with depth continuous across the contact (median relative
# difference <= SURFACE_CONTACT_REL; separate surfaces that meet in the image differ by a depth
# edge there, > 4 %).
SURFACE_CONTACT_PX = 50
SURFACE_CONTACT_BAND = 2
SURFACE_CONTACT_REL = 0.03
# ... and two objects are pieces of one surface seen from different keyframes (``_one_surface``)
# when they share at least SURFACE_SHARED_POINTS canonical points (1 cm voxels: 100 cm² of
# surface) within SURFACE_PLAN_M horizontally and SURFACE_HEIGHT_TOL vertically — the depth noise
# of a surface seen at a grazing angle is mostly a height offset — at median heights within
# SURFACE_HEIGHT_TOL.
SURFACE_SHARED_POINTS = 100
SURFACE_PLAN_M = 0.05
SURFACE_HEIGHT_TOL = 0.1
# ... or when the map's fused surface joins them (``_Surfaces.joined``), though their own points
# neither meet nor lie at one height: monocular depth of a surface seen at close range disagrees
# between keyframes by 20-30 % (a counter top 0.5 m below the camera placed 10 cm lower and
# further by the keyframes that close a loop). The fused surface averages every keyframe that
# sees the gap between the pieces, so a horizontal patch of it that reaches both is their surface:
# points whose local normal is within ~30° of vertical (|n_z| >= BRIDGE_NORMAL_Z, from the cloud
# thinned to BRIDGE_SAMPLE voxels), joined through BRIDGE_VOXEL voxels, inside the pieces' plan
# extent grown by BRIDGE_PAD and their median heights ± BRIDGE_BAND; at least BRIDGE_SHARE of
# each piece's points near the patches (within BRIDGE_NEAR) lie on one patch. The pieces' median
# heights are at most BRIDGE_HEIGHT_TOL apart and their points at most BRIDGE_GAP_M apart in plan;
# at floor height (within BRIDGE_FLOOR_M of it) the floor would join anything, so it is not used.
BRIDGE_HEIGHT_TOL = 0.15
BRIDGE_GAP_M = 0.3
BRIDGE_PAD = 0.3
BRIDGE_BAND = 0.05
BRIDGE_SAMPLE = 0.01
BRIDGE_VOXEL = 0.02
BRIDGE_NORMAL_Z = 0.85
BRIDGE_NEAR = 0.03
BRIDGE_SHARE = 0.5
BRIDGE_MIN_POINTS = 20
BRIDGE_FLOOR_M = 0.15
# Depth-explained duplicates (``_depth_explained``): the DEPTH_PAIRS pairs of detecting keyframes
# nearest by viewpoint are compared; a pair counts when its keyframes' depths disagree by at least
# DEPTH_RATIO_MIN — beyond the alignment noise (consecutive keyframes agree within ~2 %), else the
# overlap tests are valid and decide — and the sightings agree once the ratio is removed. The
# objects' centres must lie within DEPTH_RATIO_MAX of their distance (plus their extents).
DEPTH_PAIRS = 3
DEPTH_RATIO_MIN = 0.05
DEPTH_RATIO_MAX = 0.3
RATIO_MIN_POINTS = 500  # shared surface points for a keyframe pair's depth ratio
# Loop copies (``_loop_copies``): where a video's weakly linked stretch comes back to a place, its
# keyframes can be misaligned with the first visit's by more than a depth scale — in
# livingroom.mp4 the stretch 31-52, placed through one link, puts the dining area 0.3-0.45 m off
# (along x, and 0.15-0.3 m lower) from where keyframes 54-65, which see it from the other side,
# do: ten objects on the table and the sideboard mapped twice, each pair offset alike, none
# explained by the depth ratio along one keyframe's rays. The copies are recognised as a group:
# a candidate pair's nearest confident keyframes are at least LOOP_VIEW_DIST apart by viewpoint
# (seen from different places: consecutive keyframes of a video are ~0.1 apart, the two visits
# of the dining area 1.4-3.3) and disagree in depth, and LOOP_SUPPORT other candidate pairs
# linking the same keyframes (keyframes within LOOP_LINK: one stretch, one alignment) are
# displaced alike. A single pair is never enough: two chairs side by side, or two cars parked
# in a row, are offset like a copy.
LOOP_VIEW_DIST = 1.0
LOOP_LINK = 0.6
LOOP_SUPPORT = 2
LOOP_PRECISE = 0.5  # m: a pair whose sightings all span at most this pins the offset down
LOOP_OVERLAP = 0.5  # share of the shorter bounds that must overlap along each axis
# Objects seen twice (``_seen_as_one``): keyframes that see an object from different sides place it
# at depths a few percent apart, and each copy lies on its own keyframes' viewing rays. In
# livingroom.mp4 a lamp seen along x by keyframes 62-76 and along y by keyframes 109-131 is placed
# twice 0.21 m apart (each set 2-4 % short), a bottle on the dining table 0.16 m apart, a door
# handle seen from 1.7 m and from 4.5 m 0.12 m apart: no depth ratio measured between the sets
# explains it (they share little surface), nor do the objects' points or boxes overlap. The
# SEEN_SAMPLES points of each object are projected into the detecting keyframes of the other.
# There, an unoccluded point in view lies on the detection (its mask grown by the depth noise at
# the object's distance, max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL · depth), at least MASK_DILATE
# px), or in free space (the keyframe sees farther than the point by more than that noise: the
# copy is misplaced), or beside the detection on another surface: that keyframe saw something
# else there. Each object's keyframes must judge at least SEEN_EVIDENCE of the other's points and
# see at most SEEN_APART of them beside their detections; then each object, moved along its own
# viewing rays (about its keyframes' mean camera centre) by the depth factor nearest 1 among
# SEEN_STEPS steps each way up to 1 ± SEEN_DEPTH, must lie on the other's detections with at
# least SEEN_ON of its judged points, and the two moved copies must stand in one place: the middle
# of each (of its 2-98 % bounds) within the other's bounds grown by the depth noise, along both
# horizontal axes. The faces of a lamp seen from two sides pass; a metal rack that the keyframes
# of a sideboard see in front of it, standing beside the sideboard's end, does not.
SEEN_SAMPLES = 400
SEEN_EVIDENCE = 20
SEEN_APART = 0.2
SEEN_ON = 0.4
SEEN_DEPTH = 0.15
SEEN_STEPS = 6
# Parts named on their own (``_part_of``): the detector names an object in some keyframes and only
# a part of it in others (a solar figurine detected whole in two keyframes, its top alone as a
# "bottle opener" in two others; never both in one keyframe). The part — under a label that is not
# compatible, smaller than MERGE_SCALE of the whole, so no other test applies, but at least
# PART_MIN_SCALE of it (a substantial part: a handle on a door, a faucet in front of a window, a
# switch on a wall are items of their own) — and the whole are
# each detected reliably in CONFIRM_DETECTIONS keyframes (the detector names them consistently).
# The part lies on the whole: PART_INSIDE of its points inside the whole's box grown by
# CONSENSUS_MARGIN, and PART_SURFACE of them within max(POINT_VOXEL, PART_NEAR_REL · its viewing
# distance) — the depth noise between keyframes — of the whole's own points: the whole's
# keyframes lifted that surface as the whole (a car seen through a window and placed on the glass
# lies in the box of a tree seen there by other keyframes, not on its surface). Each side's
# keyframes saw the other's place and named only what they detected there: every keyframe that
# detected one had at least PART_FRAMED of the other's points in its image and VISIBLE_SHARE of
# those it can judge (at least PART_EVIDENCE: a 2 x 1.5 x 4 cm part has ~20 points) unoccluded; the whole's keyframes saw the part from
# at most PART_RANGE times the distance its own keyframes detected it from (they could have
# detected it), and in each of them that judges at least PART_EVIDENCE of the part's points (at
# least CONFIRM_DETECTIONS do) PART_ON of those lie on its detection of the whole or in free space
# (it sees past a thin part there), most of them on it, never beside it (``_judged``): a
# thermos standing on a book lies beside the flat masks of the book in most of the book's
# keyframes, though a few masks that ran up over it put it in the book's box. An item resting on or in a larger object (a cup on a table, a book in a bookcase,
# a cushion on a sofa) is usually detected together with it by some keyframe, or out of view or
# too far away in the keyframes that detected the other: never merged. A merged part votes for
# its labels in proportion to its size (its scale over the whole's): its label names a part.
PART_INSIDE = 0.9
PART_MIN_SCALE = 0.25
PART_FRAMED = 0.8
PART_RANGE = 1.5
PART_ON = 0.8
PART_EVIDENCE = 10
PART_SURFACE = 0.8
PART_NEAR_REL = 0.02
# An object is exported only with at least ``min_cloud_points`` map-cloud points: EXPORT_MIN_SUPPORT
# of the cells of its box's largest face at the resolution its keyframes sampled it, and at least
# EXPORT_MIN_CLOUD_POINTS: every exported object is then visibly drawn in the cloud (segments.ply,
# color=segment) in its colour. A cell is a cloud voxel, or the footprint of one pixel of the
# fused depth grid at the depth of its nearest detection where that is larger: an object's points
# come from its detections (the votes of the keyframes that detected it, and the unlabelled cloud
# points nearest its own lifted points, ``geometry.support_labels``: at most a few per depth pixel
# of its masks, the densest from its nearest detection), and a car seen from 40-60 m in a 1080p
# video (grid focal 660 px) has one depth pixel per 6-9 cm, 9-20 cloud voxels of 2 cm. The
# nearest detection, not the mean viewing distance: a car driving towards the camera, seen from
# 11 m by one keyframe and from 43-52 m by others, sampled at 2 cm, must be drawn at 2 cm. A confirmed object whose detecting keyframes are
# fewer than a third of those that see its surface wins few or no points in the vote (a
# dishwasher detected in 2 of the ~13 keyframes that see its front won 1 of the ~21,700 cloud
# points in its box) and lives on its support. It still has too few when its surface did not
# survive the fusion (seen by fewer than 3 keyframes, e.g. a pendant lamp), or when its detections
# placed it where the cloud holds no surface.
EXPORT_MIN_CLOUD_POINTS = 10
EXPORT_MIN_SUPPORT = 0.05
UP = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------------------------------------
# canonical point sets


def _voxel_hash(keys: NDArray[np.int64]) -> NDArray[np.uint64]:
    """Well-mixed 64-bit hash of integer voxel keys (deterministic across runs)."""
    k = keys.astype(np.uint64)
    h = (k[:, 0] * np.uint64(0x9E3779B97F4A7C15)) ^ (k[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)) \
        ^ (k[:, 2] * np.uint64(0x165667B19E3779F9))
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return h


def canonical_points(points: NDArray[Any]) -> NDArray[np.float32]:
    """One point per ``POINT_VOXEL`` voxel (the one nearest the voxel centre) and at most
    ``POINT_CAP`` of them (the voxels with the smallest hash), sorted by voxel.

    A pure function of the point *set* that composes — ``canonical(canonical(a) ∪ b) ==
    canonical(a ∪ b)`` — so an object's points, and the OBB fitted to them, do not depend on the
    order in which its instances arrived or on how they were split into updates."""
    return canonical_sources(points, np.zeros(len(np.asarray(points).reshape(-1, 3)), np.uint8))[0]


def canonical_sources(points: NDArray[Any], sources: NDArray[Any]
                      ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """``canonical_points`` with each point's sources (``SRC_*`` bits): a kept point carries the
    union of the sources of its voxel's points, so it composes as the points do."""
    p = np.asarray(points, np.float32).reshape(-1, 3)
    src = np.asarray(sources, np.uint8).reshape(-1)
    if len(p) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8)
    p64 = p.astype(np.float64)
    keys = voxel_keys(p64, POINT_VOXEL)
    d = np.sum((p64 - (keys + 0.5) * POINT_VOXEL) ** 2, axis=1)
    order = np.lexsort((p64[:, 2], p64[:, 1], p64[:, 0], d, keys[:, 2], keys[:, 1], keys[:, 0]))
    k = keys[order]
    first = np.r_[True, np.any(k[1:] != k[:-1], axis=1)]
    sel = order[first]
    merged = np.bitwise_or.reduceat(src[order], np.flatnonzero(first)).astype(np.uint8)
    p, keys = p[sel], keys[sel]
    if len(p) > POINT_CAP:
        h = _voxel_hash(keys)
        keep = np.sort(np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0], h))[:POINT_CAP])
        p, merged = p[keep], merged[keep]
    return p, merged


# Where an object's points came from (bits of ``MapObject.sources``): a detection that shapes a box
# first (``shaping_sightings``) — a reliable one from a confidently placed keyframe —, one mostly
# in the image-border band, one from a keyframe of low confidence.
SRC_RELIABLE = 1
SRC_BORDER = 2
SRC_LOW_CONFIDENCE = 4
SRC_ANY = SRC_RELIABLE | SRC_BORDER | SRC_LOW_CONFIDENCE


def source_of(reliable: bool, confident: bool) -> int:
    """The ``SRC_*`` bit of a detection's points."""
    return SRC_LOW_CONFIDENCE if not confident else SRC_RELIABLE if reliable else SRC_BORDER


# ------------------------------------------------------------------------------------------------
# objects and instances


@dataclass(frozen=True)
class Sighting:
    """Summary of one detection of an object: its keyframe, lifted point count, the share of its
    mask in the image-border band, the centroid and robust (2-98 %) bounds of its points in map
    coordinates, and whether its keyframe was placed confidently (not ``low_confidence``: a
    reliable depth scale and a pose the matches support)."""

    frame: int
    points: int
    border: float
    centroid: tuple[float, float, float]
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    confident: bool = True

    @property
    def reliable(self) -> bool:
        return self.border <= BORDER_EVIDENCE

    def key(self) -> tuple[Any, ...]:
        return (self.frame, self.points, self.centroid, self.lo, self.hi, self.border,
                self.confident)

    def to_list(self) -> list[float]:
        return [self.frame, self.points, self.border, *self.centroid, *self.lo, *self.hi,
                float(self.confident)]

    @staticmethod
    def from_list(v: list[float]) -> Sighting:
        def t(x: list[float]) -> tuple[float, float, float]:
            return (float(x[0]), float(x[1]), float(x[2]))
        return Sighting(int(v[0]), int(v[1]), float(v[2]), t(v[3:6]), t(v[6:9]), t(v[9:12]),
                        bool(v[12]) if len(v) > 12 else True)


@dataclass
class MapObject:
    id: int
    label: str
    label_votes: dict[str, float]
    scores: list[float]
    points: NDArray[np.float32]
    obb: OBB | None = None
    frames: list[int] = field(default_factory=list)  # keyframes that detected it (sorted)
    confirmed: bool = False
    strikes: int = 0
    views_in_frustum: int = 0  # keyframes of the map that have it in view (or detected it)
    created_update: int = 0
    last_seen_update: int = 0
    obs_depth: float = 2.0  # mean camera distance of its detections
    pixel_count: int = 0
    sightings: list[Sighting] = field(default_factory=list)  # one per detection (sorted)
    cloud_points: int | None = None  # map-cloud points attributed to it (None: not yet counted)
    cloud_min: int | None = None  # the cloud points it needs to be exported (min_cloud_points)
    # SRC_* bits per point (None: not recorded — a map written before sources were, or points
    # given directly — every point counts as a source of every kind)
    sources: NDArray[np.uint8] | None = None
    _memo: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def memo(self, name: str, compute: Callable[[], Any]) -> Any:
        """``compute()``, once per state of the object's evidence: its points, sightings and
        keyframes, which are replaced, never modified in place, when evidence arrives (the merge
        tests compare every pair of objects, most of them unchanged since the last round)."""
        state = (self.points, self.sightings, self.frames)
        held = self._memo.get("")
        if held is None or any(x is not y for x, y in zip(held, state, strict=True)):
            self._memo = {"": state}
        if name not in self._memo:
            self._memo[name] = compute()
        return self._memo[name]

    @property
    def observations(self) -> int:
        """Keyframes in which the object was detected."""
        return len(self.frames)

    def reliable_frames(self) -> set[int]:
        """Keyframes with a reliable detection (``Sighting.reliable``); keyframes without a
        sighting (maps written before sightings were recorded) count as reliable."""
        seen = {s.frame for s in self.sightings}
        return {s.frame for s in self.sightings if s.reliable} | (set(self.frames) - seen)

    def _add_sightings(self, new: list[Sighting]) -> None:
        self.sightings = sorted([*self.sightings, *new], key=Sighting.key)

    @property
    def score(self) -> float:
        top = sorted(self.scores, reverse=True)[:3]
        return float(np.mean(top)) if top else 0.0

    @property
    def centroid(self) -> NDArray[np.float64]:
        c: NDArray[np.float64] = self.memo("centroid", lambda: self.points.mean(0).astype(
            np.float64) if len(self.points) else np.zeros(3))
        return c.copy()

    @property
    def extent(self) -> float:
        """Horizontal diagonal of its points (``_extent``)."""
        return float(self.memo("extent", lambda: _extent(self.points)))

    def tree(self) -> cKDTree:
        """KD-tree of its points, built once per state of its evidence (``memo``): the merge tests
        query an object against every other one near it, and again after each merge."""
        t: cKDTree = self.memo("tree", lambda: cKDTree(self.points))
        return t

    def forget_tree(self) -> None:
        """Free ``tree`` (the merge tests are done; the cloud is fused next)."""
        self._memo.pop("tree", None)

    def point_sources(self) -> NDArray[np.uint8]:
        """``sources``, every kind (``SRC_ANY``) for points whose sources were not recorded."""
        if self.sources is None or len(self.sources) != len(self.points):
            return np.full(len(self.points), SRC_ANY, np.uint8)
        return self.sources

    def add_points(self, pts: NDArray[Any], sources: NDArray[Any] | int = SRC_ANY) -> None:
        """Add points (their ``SRC_*`` bits: one for all, or one per point)."""
        new = np.asarray(pts, np.float32).reshape(-1, 3)
        src = np.broadcast_to(np.asarray(sources, np.uint8), (len(new),))
        self.points, self.sources = canonical_sources(
            np.concatenate([self.points, new]), np.concatenate([self.point_sources(), src]))

    def vote(self, label: str, score: float) -> None:
        self.label_votes[label] = self.label_votes.get(label, 0.0) + score
        self.label = max(self.label_votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        self.scores = sorted([*self.scores, float(score)], reverse=True)[:10]

    def add(self, obs: list[Observation], uid: int) -> None:
        """Evidence of instances of update ``uid`` (in any order: the result is the same)."""
        obs = sorted(obs, key=lambda ob: ob.key)
        n0 = len(self.frames)
        for ob in obs:
            self.vote(ob.label, ob.score)
        self.add_points(np.concatenate([ob.points for ob in obs]),
                        np.concatenate([np.full(len(ob.points), ob.source, np.uint8)
                                        for ob in obs]))
        self.frames = sorted(set(self.frames) | {ob.frame for ob in obs})
        self.obs_depth = float((self.obs_depth * n0 + sum(ob.depth for ob in obs))
                               / (n0 + len(obs)))
        self.pixel_count = max([self.pixel_count] + [ob.pixel_count for ob in obs])
        self.last_seen_update = max(self.last_seen_update, uid)
        self._add_sightings([ob.sighting for ob in obs])

    def absorb(self, gone: MapObject) -> None:
        """Merge another object's evidence into this one."""
        for lab, v in gone.label_votes.items():
            self.label_votes[lab] = self.label_votes.get(lab, 0.0) + v
        self.label = max(self.label_votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        self.scores = sorted(self.scores + gone.scores, reverse=True)[:10]
        self.add_points(gone.points, gone.point_sources())
        n_a, n_b = len(self.frames), len(gone.frames)
        if n_a + n_b:
            self.obs_depth = (self.obs_depth * n_a + gone.obs_depth * n_b) / (n_a + n_b)
        self.frames = sorted(set(self.frames) | set(gone.frames))
        self.views_in_frustum = max(self.views_in_frustum, gone.views_in_frustum)
        self.strikes = min(self.strikes, gone.strikes)
        self.created_update = min(self.created_update, gone.created_update)
        self.last_seen_update = max(self.last_seen_update, gone.last_seen_update)
        self.pixel_count = max(self.pixel_count, gone.pixel_count)
        self._add_sightings(gone.sightings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "label_votes": self.label_votes,
            "scores": self.scores, "score": self.score,
            "obb": None if self.obb is None else self.obb.to_dict(),
            "observations": self.observations, "frames": self.frames,
            "confirmed": self.confirmed, "strikes": self.strikes,
            "views_in_frustum": self.views_in_frustum, "created_update": self.created_update,
            "last_seen_update": self.last_seen_update, "obs_depth": self.obs_depth,
            "pixel_count": self.pixel_count, "point_count": self.point_count,
            "cloud_points": self.cloud_points, "cloud_min_points": self.cloud_min,
            "point_file": points_file(self.id),
            "sightings": [s.to_list() for s in self.sightings],
        }

    @staticmethod
    def from_dict(d: dict[str, Any], points: NDArray[np.float32]) -> MapObject:
        cloud = d.get("cloud_points")
        least = d.get("cloud_min_points")
        return MapObject(
            id=int(d["id"]), label=d["label"], label_votes=dict(d.get("label_votes", {})),
            scores=list(d.get("scores", [])), points=points,
            obb=None if d.get("obb") is None else OBB.from_dict(d["obb"]),
            frames=sorted({int(f) for f in d.get("frames", [])}),
            confirmed=bool(d.get("confirmed", False)), strikes=int(d.get("strikes", 0)),
            views_in_frustum=int(d.get("views_in_frustum", 0)),
            created_update=int(d.get("created_update", 0)),
            last_seen_update=int(d.get("last_seen_update", 0)),
            obs_depth=float(d.get("obs_depth", 2.0)), pixel_count=int(d.get("pixel_count", 0)),
            sightings=sorted((Sighting.from_list(v) for v in d.get("sightings", [])),
                             key=Sighting.key),
            cloud_points=None if cloud is None else int(cloud),
            cloud_min=None if least is None else int(least),
        )

    @property
    def point_count(self) -> int:
        """Points of the map cloud attributed to the object (as a single image's ``point_count``
        counts its cloud's points); the stored sample's size for a map written before counts
        were recorded."""
        return len(self.points) if self.cloud_points is None else self.cloud_points

    def scene_object(self) -> SceneObject:
        assert self.obb is not None
        return SceneObject(
            id=self.id, label=self.label, score=self.score, obb=self.obb,
            pixel_count=self.pixel_count, point_count=self.point_count,
            observations=self.observations, frames=list(self.frames),
            labels=(self.label, *(lab for lab, _ in sorted(self.label_votes.items(),
                                                           key=lambda kv: (-kv[1], kv[0]))
                                  if lab != self.label)),
        )


@dataclass
class Observation:
    """One lifted instance of a keyframe of this update, in map coordinates: a detection, or the
    pieces of a split surface joined (``members``: the detections it stands for)."""

    frame: int  # keyframe index
    view: View
    inst: LiftedInstance
    points: NDArray[np.float32]
    centroid: NDArray[np.float64]
    depth: float  # median camera distance
    extent: float
    members: tuple[LiftedInstance, ...] = ()
    confident: bool = True  # its keyframe was placed confidently (not ``low_confidence``)
    _tree: cKDTree | None = field(default=None, repr=False)

    @staticmethod
    def of(frame: int, view: View, inst: LiftedInstance,
           members: tuple[LiftedInstance, ...] = (), confident: bool = True) -> Observation:
        pts = np.asarray(inst.lifted.points, np.float64)
        dist = float(np.median(np.linalg.norm(pts - view.T_map_cam.t, axis=1)))
        return Observation(frame, view, inst, pts.astype(np.float32), pts.mean(0), dist,
                           _extent(pts), members or (inst,), confident)

    @property
    def sighting(self) -> Sighting:
        lo, hi = np.percentile(self.points.astype(np.float64), [2, 98], axis=0)
        c = self.centroid

        def t(x: NDArray[Any]) -> tuple[float, float, float]:
            return (float(x[0]), float(x[1]), float(x[2]))
        return Sighting(self.frame, len(self.points), border_share(self.inst.mask), t(c), t(lo),
                        t(hi), self.confident)

    @property
    def source(self) -> int:
        """The ``SRC_*`` bit of its points."""
        return source_of(border_share(self.inst.mask) <= BORDER_EVIDENCE, self.confident)

    @property
    def label(self) -> str:
        return self.inst.detection.label

    @property
    def score(self) -> float:
        return float(self.inst.detection.score)

    @property
    def pixel_count(self) -> int:
        return int(self.inst.mask.sum())

    @property
    def radius(self) -> float:
        """Distance within which another point set counts as the same surface."""
        return max(0.05, 0.02 * self.depth)

    @property
    def key(self) -> tuple[str, float, float, float, float, int]:
        """Content key: a total order that does not depend on the keyframes' order."""
        c = self.centroid
        return (self.label, float(c[0]), float(c[1]), float(c[2]), -self.score,
                -self.pixel_count)

    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree


def points_file(oid: int) -> str:
    return f"objects/points_{oid:06d}.npy"


def sources_file(oid: int) -> str:
    """The ``SRC_*`` bits of the points of ``points_file`` (absent: not recorded)."""
    return f"objects/sources_{oid:06d}.npy"


@dataclass
class Vacated:
    """The place of an object that an update removed (``retire_pixels``): the pixels retired from
    each keyframe that detected it (``masks``: keyframe name -> RLE mask; the object's detection,
    grown by ``RETIRE_DILATE``), its box and viewing distance (the box grown by the depth noise
    holds what stood by it, such as its shadow) and the keyframes that saw through it
    (``witnesses``: the latest observation of the place). The map cloud there is drawn from the
    witnesses (``geometry.fused_cloud_points``, ``geometry.attribute_points``): the older
    keyframes' views of the place are retired, so the surface behind the object is seen only by
    them, often fewer than a surface's usual views."""

    update: int
    object: int
    masks: dict[str, dict[str, Any]]
    witnesses: list[str]
    box: OBB | None = None
    obs_depth: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"update": self.update, "object": self.object,
                "masks": dict(sorted(self.masks.items())), "witnesses": sorted(self.witnesses),
                "box": None if self.box is None else self.box.to_dict(),
                "obs_depth": self.obs_depth}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Vacated:
        return Vacated(int(d["update"]), int(d["object"]), dict(d.get("masks", {})),
                       [str(n) for n in d.get("witnesses", [])],
                       None if d.get("box") is None else OBB.from_dict(d["box"]),
                       float(d.get("obs_depth", 0.0)))


@dataclass
class ObjectState:
    objects: list[MapObject]
    next_id: int
    merged_into: dict[int, int] = field(default_factory=dict)
    floor_z: float | None = None
    observed: set[int] = field(default_factory=set)  # ids observed in this update
    # keyframe index -> pixels this update invalidated in it: the masks of the objects it removed
    invalidated: dict[int, int] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    vacated: list[Vacated] = field(default_factory=list)  # places of removed objects (all updates)

    def by_id(self) -> dict[int, MapObject]:
        return {o.id: o for o in self.objects}

    def resolve(self, oid: int) -> int | None:
        seen = set()
        while oid in self.merged_into and oid not in seen:
            seen.add(oid)
            oid = self.merged_into[oid]
        return oid if oid in self.by_id() else None

    def exported(self) -> list[SceneObject]:
        """The confirmed objects with a box and enough points in the map cloud
        (``min_cloud_points``; not yet counted: maps written before cloud counts were
        recorded)."""
        return [o.scene_object() for o in sorted(self.objects, key=lambda o: o.id)
                if o.confirmed and o.obb is not None
                and (o.cloud_points is None or o.cloud_points >= (
                    EXPORT_MIN_CLOUD_POINTS if o.cloud_min is None else o.cloud_min))]


def sample_spacing(voxel: float, obs_depth: float = 0.0, focal: float = 0.0) -> float:
    """How finely the map samples a surface seen from ``obs_depth`` metres: the cloud's ``voxel``,
    or the footprint of one pixel of the fused depth grids (focal length ``focal`` px) where that
    is coarser; ``focal`` 0: the voxel."""
    return max(float(voxel), float(obs_depth) / focal) if focal > 0 else float(voxel)


def min_cloud_points(box: OBB, voxel: float, seen_from: float = 0.0, focal: float = 0.0) -> int:
    """The map-cloud points an object with ``box``, detected from ``seen_from`` metres at the
    nearest, needs to be exported: ``EXPORT_MIN_SUPPORT`` of the cells of its largest face at the
    map's sampling (``sample_spacing``: the cloud ``voxel``, or the depth pixel's footprint where
    coarser), at least ``EXPORT_MIN_CLOUD_POINTS``."""
    a, b = np.sort(np.asarray(box.size, np.float64))[1:]
    cell = sample_spacing(voxel, seen_from, focal)
    return max(EXPORT_MIN_CLOUD_POINTS, int(np.ceil(EXPORT_MIN_SUPPORT * a * b / cell ** 2)))


def load_state(current: Any, meta: dict[str, Any]) -> ObjectState:
    import json

    p = current(OBJECTS_JSON)
    if not p.exists():
        return ObjectState([], int(meta.get("next_object_id", 1)), floor_z=meta.get("floor_z"))
    d = json.loads(p.read_text())
    objs = []
    for od in d.get("objects", []):
        pp = current(points_file(int(od["id"])))
        pts = np.load(pp).astype(np.float32) if pp.exists() else np.zeros((0, 3), np.float32)
        o = MapObject.from_dict(od, pts)
        sp = current(sources_file(o.id))
        if sp.exists():
            src = np.load(sp).astype(np.uint8)
            o.sources = src if len(src) == len(pts) else None
        objs.append(o)
    return ObjectState(objs, int(d.get("next_id", meta.get("next_object_id", 1))),
                       {int(k): int(v) for k, v in d.get("merged_into", {}).items()},
                       floor_z=d.get("floor_z", meta.get("floor_z")),
                       vacated=[Vacated.from_dict(v) for v in d.get("vacated", [])])


def load_vacated(current: Any) -> list[Vacated]:
    """The places of the objects the map's updates removed (``Vacated``), as stored or staged."""
    import json

    p = current(OBJECTS_JSON)
    if not p.exists():
        return []
    return [Vacated.from_dict(v) for v in json.loads(p.read_text()).get("vacated", [])]


# ------------------------------------------------------------------------------------------------
# geometry helpers


def _visible_pixels(view: View, pts: NDArray[Any]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(u, v) pixels of ``view`` onto which the in-view, unoccluded points of ``pts`` project."""
    if len(pts) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    inside, z, d, u, v = view.lookup_pixels(pts)
    vis = inside & (z <= d + tau(z))
    return u[vis], v[vis]


def projected_mask(view: View, pts: NDArray[Any], dilate: int = MASK_DILATE) -> NDArray[np.bool_]:
    """Pixels of ``view`` covered by the visible (unoccluded) points of an object."""
    mask = np.zeros(view.depth.shape, bool)
    u, v = _visible_pixels(view, pts)
    if len(u) == 0:
        return mask
    mask[v, u] = True
    return ndimage.binary_dilation(mask, iterations=dilate) if dilate else mask


def projected_iou(view: View, mask: NDArray[np.bool_], pts: NDArray[Any],
                  dilate: int = MASK_DILATE) -> float:
    """IoU of ``mask`` with the pixels ``pts`` cover in ``view`` (``projected_mask``), computed on
    the bounding box of both (same result as on the whole image, much cheaper)."""
    u, v = _visible_pixels(view, pts)
    if len(u) == 0:
        return 0.0
    h, w = mask.shape
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    r0, r1 = int(v.min()), int(v.max())
    c0, c1 = int(u.min()), int(u.max())
    if len(rows):
        r0, r1 = min(r0, int(rows[0])), max(r1, int(rows[-1]))
        c0, c1 = min(c0, int(cols[0])), max(c1, int(cols[-1]))
    r0, r1 = max(0, r0 - dilate), min(h, r1 + dilate + 1)
    c0, c1 = max(0, c0 - dilate), min(w, c1 + dilate + 1)
    pm = np.zeros((r1 - r0, c1 - c0), bool)
    pm[v - r0, u - c0] = True
    if dilate:
        pm = ndimage.binary_dilation(pm, iterations=dilate)
    m = mask[r0:r1, c0:c1]
    union = np.logical_or(pm, m).sum()
    return float(np.logical_and(pm, m).sum() / union) if union else 0.0


def overlap_fraction(a: NDArray[Any], b: NDArray[Any], radius: float,
                     tree: cKDTree | None = None) -> float:
    """Fraction of points of ``a`` within ``radius`` of ``b`` (``tree``: a KD-tree of ``b``). Only
    the points within ``radius`` of ``b``'s bounding box are looked up: the others cannot be (the
    objects the merge tests compare are near each other, most of their points apart)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    sa = a if len(a) <= 4000 else a[np.linspace(0, len(a) - 1, 4000).astype(int)]
    t = tree or cKDTree(b)
    pad = radius * (1.0 + 1e-6)  # a point that far from the box along one axis is farther
    near = np.all((sa >= t.mins - pad) & (sa <= t.maxes + pad), axis=1)
    if not near.any():
        return 0.0
    d, _ = t.query(sa[near], k=1, distance_upper_bound=radius)
    return float(np.count_nonzero(np.isfinite(d)) / len(sa))


def _extent(pts: NDArray[Any]) -> float:
    """Horizontal diagonal of a point set (large objects are seen piecewise)."""
    if len(pts) < 2:
        return 0.0
    lo, hi = np.percentile(np.asarray(pts)[:, :2], [2, 98], axis=0)
    return float(np.linalg.norm(hi - lo))


def absence_tau(z: NDArray[Any], size: float = float("inf")) -> NDArray[Any]:
    """Margin for "seen through" an object of extent ``size`` at depth ``z``: wider than the pixel
    test (monocular depth of thin objects is less reliable than of large surfaces), but no wider
    than ``ABSENCE_TAU_SIZE`` of the object's own extent, and never below the depth noise."""
    wide = np.maximum(ABSENCE_TAU_MIN, ABSENCE_TAU_REL * z)
    narrow = np.maximum(np.maximum(ABSENCE_TAU_SMALL_MIN, ABSENCE_TAU_SMALL_REL * z),
                        ABSENCE_TAU_SIZE * size)
    return np.minimum(wide, narrow)


def border_share(mask: NDArray[np.bool_]) -> float:
    """Share of a mask's pixels in the image-border band that ``View.usable`` excludes."""
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if n == 0:
        return 0.0
    h, w = m.shape
    bh, bw = max(1, int(BORDER * h)), max(1, int(BORDER * w))
    inner = int(m[bh:h - bh, bw:w - bw].sum())
    return float((n - inner) / n)


def depth_ratio(a: View, b: View) -> float | None:
    """How much further keyframe ``b`` places the surfaces both keyframes see than ``a`` does:
    the median, over ``a``'s surface points that ``b`` sees, of ``b``'s observed depth over the
    point's depth in ``b``'s camera (the same surface only: within a factor 1.3, larger
    differences are occlusions). Monocular depth maps disagree by a largely global factor (as in
    ``validity``). None when they share fewer than ``RATIO_MIN_POINTS`` points."""
    pts, _, _ = a.grid_points()
    if len(pts) == 0:
        return None
    inside, z, d = b.lookup(pts)
    r = d[inside] / z[inside]
    r = r[(r > 1 / 1.3) & (r < 1.3)]
    return float(np.median(r)) if len(r) >= RATIO_MIN_POINTS else None


def _bbox(mask: NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def continuous_contact(a: NDArray[np.bool_], b: NDArray[np.bool_], depth: NDArray[Any],
                       good: NDArray[np.bool_]) -> bool:
    """Whether masks ``a`` and ``b`` touch along a continuous surface: at least
    ``SURFACE_CONTACT_PX`` pixels of ``b`` within ``SURFACE_CONTACT_BAND`` px of ``a`` are
    ``good`` (valid, no depth edge), and there ``b``'s depth continues ``a``'s (the median relative
    difference to ``a``'s mean depth around each contact pixel is at most
    ``SURFACE_CONTACT_REL``)."""
    ba, bb = _bbox(a), _bbox(b)
    if ba is None or bb is None:
        return False
    m = SURFACE_CONTACT_BAND + 1
    r0, r1 = max(ba[0], bb[0]) - m, min(ba[1], bb[1]) + m
    c0, c1 = max(ba[2], bb[2]) - m, min(ba[3], bb[3]) + m
    if r1 <= r0 or c1 <= c0:  # the boxes (with the band) do not meet
        return False
    h, w = a.shape
    sl = (slice(max(0, r0 - m), min(h, r1 + m)), slice(max(0, c0 - m), min(w, c1 + m)))
    A, B, G, D = a[sl], b[sl], good[sl], np.asarray(depth, np.float64)[sl]
    near = ndimage.binary_dilation(A, iterations=SURFACE_CONTACT_BAND) & B & G
    size = 2 * SURFACE_CONTACT_BAND + 1
    wa = ndimage.uniform_filter((A & G).astype(np.float64), size)
    sa = ndimage.uniform_filter(np.where(A & G, D, 0.0), size)
    ok = near & (wa > 1e-9)
    if int(ok.sum()) < SURFACE_CONTACT_PX:
        return False
    rel = D[ok] / (sa[ok] / wa[ok]) - 1.0
    return abs(float(np.median(rel))) <= SURFACE_CONTACT_REL


def surface_pieces(view: View, insts: list[LiftedInstance]) -> list[list[int]]:
    """The instances of one keyframe grouped into objects: pieces of one horizontal surface that
    the detector split around the items on it (labels allowed by ``split_surface``, touching with
    continuous depth, ``continuous_contact``) form one group; every other instance is alone.
    Groups are listed by their first instance."""
    n = len(insts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if split_surface(insts[i].detection.label, insts[j].detection.label)]
    if pairs:
        ok = view.valid & (view.depth > 0)
        good = ok & ~depth_edge_mask(np.where(ok, view.depth, 0.0))
        for i, j in pairs:
            if find(i) != find(j) and continuous_contact(insts[i].mask, insts[j].mask,
                                                         view.depth, good):
                parent[max(find(i), find(j))] = min(find(i), find(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def in_view(pts: NDArray[Any], K: Any, T_map_cam: Any) -> bool:
    """Whether a keyframe (grid intrinsics ``K`` and pose) has at least ``IN_VIEW_SHARE`` of the
    points in front of it and inside its image. Occlusion is ignored: the question is whether the
    keyframe could have detected the object, and an object whose depth is wrong would otherwise
    hide from every other keyframe."""
    if len(pts) == 0:
        return False
    pc = T_map_cam.inverse().apply(np.asarray(pts, np.float64))
    z = pc[:, 2]
    w, h = K.width, K.height
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K.fx * pc[:, 0] / z + K.cx
        v = K.fy * pc[:, 1] / z + K.cy
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return bool(inside.mean() >= IN_VIEW_SHARE)


def count_views(obj: MapObject, records: list[Any]) -> int:
    """Keyframes of the map (``records``) that detected the object or have it in view."""
    pts = obj.points
    if len(pts) > VIEW_SAMPLES:
        pts = pts[np.linspace(0, len(pts) - 1, VIEW_SAMPLES).astype(int)]
    detected = set(obj.frames)
    return sum(1 for r in records if r.index in detected or in_view(pts, r.K_grid, r.T_map_cam))


def below_floor(obj: MapObject, floor_z: float | None, margin: float = 0.3) -> bool:
    """An object entirely below the map floor is not physical (e.g. seen in a mirror or through
    a reflective screen, where monocular depth places it behind the glass)."""
    if floor_z is None or len(obj.points) < 10:
        return False
    return float(np.percentile(obj.points[:, 2], 90)) < floor_z - margin


def shaping_sightings(obj: MapObject) -> list[Sighting]:
    """The sightings that may shape an object's box: those of confidently placed keyframes when
    at least ``SHAPE_MIN`` of them detected it (a keyframe of low confidence — an unreliable depth
    scale, or a pose the matches do not support — places the object offset as a whole: a wallet
    placed onto the neighbouring item; the map cloud leaves these keyframes out too), and of
    these the reliable ones (``Sighting.reliable``) when they are at least ``SHAPE_MIN`` and a
    strict majority: a detection mostly in the image-border band is cut off and its monocular
    depth unreliable, as for confirmation. The others still count as evidence (label, keyframes,
    confirmation); they shape the box only while nothing better saw the object."""
    s = obj.sightings
    confident = [x for x in s if x.confident]
    if len(confident) >= SHAPE_MIN:
        s = confident
    reliable = [x for x in s if x.reliable]
    if len(reliable) >= SHAPE_MIN and 2 * len(reliable) > len(s):
        s = reliable
    return s


def _agreeing(s: list[Sighting], obs_depth: float) -> list[Sighting]:
    """The sightings of ``s`` that agree with the one most others agree with (ties: more points,
    then content). Two sightings agree when their bounds overlap, allowing a gap of the depth
    noise at ``obs_depth``: the partial views of a large object overlap one another, while
    sightings of a small object that monocular depth scattered along the viewing rays do not."""
    if not s:
        return []
    lo = np.array([x.lo for x in s])
    hi = np.array([x.hi for x in s])
    tol = max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * obs_depth)
    near = np.all((lo[:, None] <= hi[None] + tol) & (lo[None] <= hi[:, None] + tol), axis=2)
    best = min(range(len(s)), key=lambda i: (-int(near[i].sum()), -s[i].points, s[i].key()))
    return [x for x, ok in zip(s, near[best], strict=True) if ok]


def agreeing_sightings(obj: MapObject) -> list[Sighting]:
    """The sightings that shape the box (``shaping_sightings``) that agree with one another
    (``_agreeing``) — all of them when there are fewer than ``CONSENSUS_MIN``."""
    s = shaping_sightings(obj)
    return _agreeing(s, obj.obs_depth) if len(s) >= CONSENSUS_MIN else s


def fit_points(obj: MapObject) -> NDArray[np.float32]:
    """The points a box is fitted to (all of them unless every detection has its sighting
    recorded): the points of the kinds of detection that shape the box (``shaping_sightings``,
    ``SRC_*``), and when some of those disagree (``agreeing_sightings``), only those inside the
    bounds (plus ``CONSENSUS_MARGIN``) of the agreeing ones."""
    def compute() -> NDArray[np.float32]:
        s = obj.sightings
        if not s or {x.frame for x in s} != set(obj.frames):
            return obj.points
        shaping = shaping_sightings(obj)
        keep = agreeing_sightings(obj)
        ok = np.ones(len(obj.points), bool)
        if len(shaping) < len(s):
            allowed = 0
            for x in shaping:
                allowed |= source_of(x.reliable, x.confident)
            ok = (obj.point_sources() & allowed) != 0
        if len(keep) < len(shaping):
            lo = np.min([x.lo for x in keep], axis=0) - CONSENSUS_MARGIN
            hi = np.max([x.hi for x in keep], axis=0) + CONSENSUS_MARGIN
            ok &= np.all((obj.points >= lo) & (obj.points <= hi), axis=1)
        if ok.all() or ok.sum() < 10:
            return obj.points
        pts: NDArray[np.float32] = obj.points[ok]
        return pts
    out: NDArray[np.float32] = obj.memo("fit_points", compute)
    return out


def refit(obj: MapObject, floor_z: float | None) -> None:
    if len(obj.points) < 10:
        return
    obj.obb = fit_object_obb(fit_points(obj), obj.label, UP, floor_z)


def confirm(obj: MapObject) -> None:
    """Confirmed when reliable detections (``MapObject.reliable_frames``) come from at least
    ``CONFIRM_DETECTIONS`` keyframes, or from one when no other keyframe of the map has the object
    in view (``views_in_frustum`` <= 1: a single image, or a place only one keyframe saw)."""
    need = CONFIRM_DETECTIONS if obj.views_in_frustum > 1 else 1
    obj.confirmed = len(obj.reliable_frames()) >= need


# ------------------------------------------------------------------------------------------------
# the update


def map_floor(ctx: Any, per_frame: int | None = None) -> tuple[NDArray[np.float64], float | None]:
    """Map-frame grid points of this update's confidently placed keyframes (about ``per_frame``
    of each, all when None) and the floor height among them: the lowest well-supported
    horizontal surface. Maps accumulate stray points below the floor (see-through shelves,
    reflections, drift), so up to 10 % of the points may lie beneath it (single images: 3 %)."""
    pts = []
    placed = [nf for nf in ctx.new if nf.record is not None and nf.depth is not None]
    for nf in sorted(placed, key=lambda nf: nf.record.order_key):
        if nf.record.low_confidence:
            continue
        p, _, _ = keyframe_view(nf).grid_points()
        pts.append(p if per_frame is None else p[:: max(1, len(p) // per_frame)])
    if not pts:
        return np.zeros((0, 3)), None
    allp = np.concatenate(pts)
    return allp, floor_candidate_height(allp[:, 2], max_below=MAP_FLOOR_MAX_BELOW)


def _instances_json(frame_items: list[tuple[int, LiftedInstance]]) -> dict[str, Any]:
    return {"instances": [
        {"object_id": oid, "label": inst.detection.label, "score": inst.detection.score,
         "source": inst.detection.source, "mask": rle.encode(inst.mask)}
        for oid, inst in frame_items
    ]}


def rescale_objects(state: ObjectState, rescaled: dict[int, DepthCorrection], records: list[Any]
                    ) -> set[int]:
    """Move the stored objects with the stored keyframes whose depth this update corrected
    (``rescaled``: keyframe index -> correction; ``mapping.api._adjust_depth_scales``): each
    sighting's centroid and bounds move along their viewing rays, about their keyframe's camera
    centre, by that keyframe's factor at their depth; the points — which do not record their
    keyframe — scale about the sightings' mean camera centre by the sightings' mean factor at
    their centroids (geometric, weighted by their points; keyframes not corrected count as 1),
    and the box is refitted. Returns the ids of the objects that moved."""
    moved: set[int] = set()
    if not rescaled:
        return moved
    poses = {r.index: r.T_map_cam for r in records}
    for o in state.objects:
        if not any(s.frame in rescaled and s.frame in poses for s in o.sightings):
            continue
        out, logs, weights, centres = [], [], [], []
        for s in o.sightings:
            T = poses.get(s.frame)
            corr = rescaled.get(s.frame) if T is not None else None
            if T is not None:
                centres.append(T.t)
                c = 1.0 if corr is None else _factor_at(corr, T, s.centroid)
                logs.append(np.log(c))
                weights.append(float(max(s.points, 1)))
            if corr is None or T is None:
                out.append(s)
                continue

            def moved_to(v: tuple[float, float, float], T: Pose = T,
                         corr: DepthCorrection = corr) -> tuple[float, float, float]:
                w = T.t + _factor_at(corr, T, v) * (np.asarray(v, np.float64) - T.t)
                return (float(w[0]), float(w[1]), float(w[2]))
            out.append(Sighting(s.frame, s.points, s.border, moved_to(s.centroid),
                                moved_to(s.lo), moved_to(s.hi), s.confident))
        w = np.asarray(weights)
        c_obj = float(np.exp(np.sum(w * np.asarray(logs)) / np.sum(w)))
        C_obj = np.sum(np.asarray(centres) * w[:, None], axis=0) / np.sum(w)
        if len(o.points):
            o.points, o.sources = canonical_sources(
                C_obj + c_obj * (o.points.astype(np.float64) - C_obj), o.point_sources())
        o.obs_depth *= c_obj
        o.sightings = sorted(out, key=Sighting.key)
        refit(o, state.floor_z)
        moved.add(o.id)
    return moved


def _factor_at(corr: DepthCorrection, T: Pose, p: Any) -> float:
    """``corr``'s depth factor at map point ``p`` seen from camera ``T`` (its z-depth there)."""
    z = float((np.asarray(p, np.float64) - T.t) @ T.R[:, 2])
    return float(corr.factor(np.array([z]))[0])


def update_objects(ctx: Any, records: list[Any], progress: Any,
                   surface: NDArray[Any] | Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]]
                   | None = None, cuts: dict[int, float] | None = None) -> ObjectState:
    """The update's objects (see the module docstring); ``surface``: the map's fused surface, or
    a function giving it within a box (``geometry.SurfaceQuery``), evidence that pieces of a
    horizontal surface are one object; ``cuts``: per keyframe index, how deep the map fuses it
    (``geometry.keyframe_depth_cuts``: places beyond are not judged, ``_Places``)."""
    tx = ctx.tx
    state = load_state(tx.current, ctx.meta)
    uid = ctx.update_id
    moved = rescale_objects(state, getattr(ctx, "rescaled", {}) or {}, records)
    _, fz = map_floor(ctx)
    if state.floor_z is None or (fz is not None and not ctx.old_frames):
        state.floor_z = fz
    ctx.meta["floor_z"] = state.floor_z

    # 1. every instance of the update, lifted into the map (without the support its mask bled
    #    onto: ``trim_support``), with the number of its detection:
    #    all detections of the update's keyframes (placed or not) in keyframe order, continuing
    #    the map's count; the pieces of a split surface are one instance (numbered by the first)
    first_new = state.next_id
    first_number: dict[int, int] = {}
    count = first_new
    for nf in sorted(ctx.new, key=lambda nf: nf.kf.index):
        first_number[nf.kf.index] = count
        count += len(nf.dets)
    new_views: dict[int, tuple[Any, View]] = {}
    per_frame: dict[int, list[int]] = {}
    obs: list[Observation] = []
    number: list[int] = []  # detection number per observation
    for nf in ctx.new:
        if nf.record is None or nf.depth is None:
            continue
        rec = nf.record
        view = keyframe_view(nf)
        new_views[rec.index] = (nf, view)
        insts = [trim_support(inst, nf.depth, nf.frame.K_grid, view.valid, rec.T_map_cam)
                 for inst in lift_detections(nf.frame, nf.dets, rec.T_map_cam, depth=nf.depth,
                                             valid=view.valid)]
        pieces = surface_pieces(view, insts)
        per_frame[rec.index] = list(range(len(obs), len(obs) + len(pieces)))
        rank = {id(det): j for j, det in enumerate(nf.dets)}
        for group in pieces:
            members = tuple(insts[k] for k in group)
            obs.append(Observation.of(rec.index, view, join_instances(list(members)), members,
                                      confident=not rec.low_confidence))
            number.append(first_number[nf.kf.index]
                          + min(rank[id(m.detection)] for m in members))
    views = _Views(ctx, {f: v for f, (_, v) in new_views.items()}, records)

    # 2. group them (order-free) and fold each group into an existing or a new object; new
    #    objects get provisional ids >= first_new (content order) until their final numbering
    owner: list[int] = [0] * len(obs)  # object id per observation
    touched: set[int] = set()
    groups = _group(obs, state.objects, _Earlier(ctx, state, views))
    existing = state.by_id()
    fresh_groups = sorted((m for oid, m in groups if oid is None),
                          key=lambda m: min(obs[i].key for i in m))
    targets: list[tuple[MapObject, list[int]]] = [
        (existing[oid], m) for oid, m in groups if oid is not None]
    for k, m in enumerate(fresh_groups):
        o = MapObject(first_new + k, obs[m[0]].label, {}, [], np.zeros((0, 3), np.float32),
                      created_update=uid)
        state.objects.append(o)
        targets.append((o, m))
    for o, m in targets:
        o.add([obs[i] for i in m], uid)
        o.strikes = 0  # re-detected: the latest observation says it is there
        touched.add(o.id)
        for i in m:
            owner[i] = o.id
    for o in state.objects:  # provisional boxes: the merge test compares boxes of new objects too
        if o.id in touched:
            refit(o, state.floor_z)

    # 3. merge duplicates (lower id kept; keepers are refitted), then count the keyframes of the
    #    whole map that have each object in view and confirm
    alias: dict[int, int] = {}
    surfaces = None if surface is None else _Surfaces(surface, state.floor_z)
    instances: dict[int, list[tuple[int, Any]]] = {}
    for i, ob in enumerate(obs):
        instances.setdefault(ob.frame, []).append((owner[i], ob.inst.mask))
    masks = _Masks(ctx, views, instances, state.merged_into, alias)
    merged = _merge(state, touched, alias, views, surfaces, masks)
    for o in state.objects:
        o.forget_tree()
        o.views_in_frustum = count_views(o, records)
        confirm(o)

    # 4. drop non-physical objects, and let the latest keyframes win (``_Places``): every object is
    #    judged by this update's keyframes added after its last detection that could have detected
    #    it there (``_absence``); an object they see elsewhere moves there first (``_moves``: it
    #    keeps the id of where it was, and that place is vacated like a removed object's); an
    #    object first detected where the update's earlier keyframes saw free space arrived, and
    #    their views through it are retired so that the latest keyframes draw it
    dropped = {o.id for o in state.objects if below_floor(o, state.floor_z)}
    places = _Places(views, masks, state.objects, cuts)
    new_idx = sorted(new_views)
    moves = [mv for mv in _moves(state, touched - dropped, places, _Colours(ctx, views, masks))
             if mv.src.id not in dropped and mv.dst.id not in dropped]

    def later(o: MapObject) -> list[Verdict]:
        return places.verdicts(o, [f for f in new_idx if f > max(o.frames)])

    src_ids = {mv.src.id for mv in moves}
    # only the map's objects: a candidate (unconfirmed) that later keyframes see through is one
    # whose single detection's depth nothing confirms, and it stays a candidate
    candidates = [o for o in state.objects
                  if o.confirmed and o.id not in dropped and o.id not in src_ids]
    witnesses: dict[int, list[int]] = {}
    removed = _absence(candidates, {o.id: later(o) for o in candidates}, witnesses)
    kept = [mv for mv in moves if mv.dst.id not in removed]
    if len(kept) < len(moves):  # moved to a place the update then saw empty: judged as before
        back = [mv.src for mv in moves if mv not in kept]
        removed += _absence(back, {o.id: later(o) for o in back}, witnesses)
    moves = kept
    arrived: dict[int, NDArray[np.bool_]] = {}

    def arrive(o: MapObject, through: list[Verdict]) -> None:
        for v in through:
            px = places.through_pixels(o, v)
            arrived[v.frame] = arrived[v.frame] | px if v.frame in arrived else px

    for mv in moves:
        witnesses[mv.src.id] = mv.departed
        arrive(mv.dst, mv.arrived)
    settled = dropped | set(removed) | {mv.dst.id for mv in moves}
    for o in state.objects:
        if o.confirmed and o.id in touched and o.id not in settled \
                and min(o.frames) in new_views:
            earlier = places.verdicts(o, [f for f in new_idx if f < min(o.frames)])
            if _judgement(o, earlier, few=False) == "gone":  # absent: 3 keyframes or more
                last = max((v.frame for v in earlier if v.in_place),
                           default=-1)
                arrive(o, [v for v in earlier if v.share >= REMOVE_FRACTION and v.frame > last])
    retired = set(removed) | {mv.src.id for mv in moves}
    state.invalidated, vacated = retire_pixels(ctx, views, masks, [o for o in state.objects
                                                                   if o.id in retired],
                                               witnesses, arrived)
    state.vacated.extend(vacated)
    moved_from = {mv.src.id: mv.dst for mv in moves}
    _forget_detections(ctx, views, masks, [mv.src for mv in moves], new_views)
    gone = dropped | retired
    for oid in sorted(gone | set(alias)):
        if oid < first_new and oid not in moved_from:
            tx.delete(points_file(oid))
            tx.delete(sources_file(oid))
    state.objects = [o for o in state.objects if o.id not in gone]

    # 5. final ids of the new objects: the number of their first detection (bookkeeping)
    def resolved(oid: int) -> int:
        while oid in alias:
            oid = alias[oid]
        return oid

    first_detection: dict[int, int] = {}
    for i, oid in enumerate(owner):
        k = resolved(oid)
        first_detection[k] = min(first_detection.get(k, number[i]), number[i])
    fresh = [o for o in state.objects if o.id >= first_new]
    rename = {o.id: first_detection[o.id] for o in fresh}
    moved_ids: list[int] = []
    for src_id, dst in sorted(moved_from.items()):
        # a moved object keeps the id of where it was; an id of its own that the map stored
        # resolves to it
        src_final = src_id if src_id < first_new else first_detection[src_id]
        if dst.id < first_new:
            state.merged_into[dst.id] = src_final
            tx.delete(points_file(dst.id))
            tx.delete(sources_file(dst.id))
        rename[dst.id] = src_final
        dst.created_update = min(dst.created_update, next(mv.src.created_update for mv in moves
                                                          if mv.src.id == src_id))
        moved_ids.append(src_final)
    for o in state.objects:
        o.id = rename.get(o.id, o.id)
    state.next_id = count
    for old, keeper in alias.items():
        if old < first_new:  # a stored id; its keeper has a lower id, so is stored too
            state.merged_into[old] = keeper
    touched = {rename.get(t, t) for t in touched if t not in alias and t not in gone}
    for v in vacated:  # the places of this update record the objects' final ids
        if v.object >= first_new:
            v.object = first_detection.get(v.object, v.object)

    def final_id(oid: int) -> int:
        oid = resolved(oid)
        if oid in moved_from:
            return 0  # detected where it no longer is
        if oid in gone:
            return 0 if oid >= first_new else oid  # a removed id resolves to nothing
        return rename.get(oid, oid)

    for f, idxs in per_frame.items():
        nf, _ = new_views[f]
        items = [(final_id(owner[i]), m) for i in idxs for m in obs[i].members]
        tx.write_json(frame_file(nf.record.name, "instances.json"), _instances_json(items))
    state.observed = touched
    for o in state.objects:
        if o.id in touched or o.id in moved:
            tx.save_npy(points_file(o.id), o.points.astype(np.float32))
            tx.save_npy(sources_file(o.id), o.point_sources())
    save_state(tx, state)
    confirmed = sum(o.confirmed for o in state.objects)
    state.summary = {
        "instances": len(obs), "touched": len(touched), "new": len(fresh), "merged": merged,
        "removed": sorted(oid for oid in removed if oid < first_new),
        "withdrawn": sum(oid >= first_new for oid in removed),
        "moved": sorted(moved_ids),
        "pixels_invalidated": sum(state.invalidated.values()),
        "vacated": len(vacated),
        "below_floor_dropped": len(dropped),
        "total": len(state.objects), "confirmed": confirmed,
        "unconfirmed": len(state.objects) - confirmed,
    }
    progress(f"objects: {confirmed} confirmed of {len(state.objects)}; "
             f"{len(removed)} removed, {len(moved_ids)} moved, {merged} merged")
    return state


RETIRE_DILATE = 2


def retire_pixels(ctx: Any, views: _Views, masks: _Masks, gone: list[MapObject],
                  witnesses: dict[int, list[int]] | None = None,
                  arrived: dict[int, NDArray[np.bool_]] | None = None
                  ) -> tuple[dict[int, int], list[Vacated]]:
    """Latest wins for the cloud: the detection masks of the objects this update removes are
    invalidated in the keyframes that detected them (``valid.png``, staged), so fusion stops
    drawing them: the object's own record and its points in the map cloud go together. Returns
    keyframe index -> number of pixels invalidated, and the removed objects' places (``Vacated``,
    with their ``witnesses``: object id -> indices of the keyframes that saw through it), where
    the map cloud is then drawn from the witnesses. ``arrived`` (keyframe index -> pixels): the
    pixels with which earlier keyframes saw through the new place of a moved object (``Move``)
    are invalidated too, so that its keyframes alone draw it there."""
    from oh_my_slam.core.images import png_bytes

    kill: dict[int, NDArray[np.bool_]] = {f: m.copy() for f, m in (arrived or {}).items()}
    places: list[Vacated] = []
    for o in sorted(gone, key=lambda o: o.id):
        retired: dict[str, dict[str, Any]] = {}
        for f in sorted(o.frames):
            view = views.get(f)
            if view is None:
                continue
            found = [m if isinstance(m, np.ndarray) else rle.decode(m)
                     for oid, m in masks.instances(f) if masks.owner(oid) == o.id]
            found = [m for m in found if m.shape == view.depth.shape]
            if found:
                m = ndimage.binary_dilation(np.logical_or.reduce(found), iterations=RETIRE_DILATE)
                kill[f] = kill.get(f, np.zeros(view.depth.shape, bool)) | m
                retired[views.records[f].name] = rle.encode(m)
        seen_through = [views.records[f].name for f in (witnesses or {}).get(o.id, [])
                        if f in views.records]
        if retired and seen_through:
            places.append(Vacated(int(ctx.update_id), o.id, retired, seen_through, o.obb,
                                  float(o.obs_depth)))
    out: dict[int, int] = {}
    for f, m in kill.items():
        view = views.get(f)
        assert view is not None
        m = m & view.valid
        if m.any():
            name = views.records[f].name
            ctx.tx.write_bytes(frame_file(name, "valid.png"),
                               png_bytes((view.valid & ~m).astype(np.uint8) * 255))
            out[f] = int(m.sum())
    return out, places


def _forget_detections(ctx: Any, views: _Views, masks: _Masks, objs: list[MapObject],
                       new_views: dict[int, Any]) -> None:
    """The stored keyframes' detections of objects that moved (``Move.src``) no longer name them
    (``instances.json``, staged: object id 0, as a removed object's): they show where the object
    no longer is, and the object keeps its id elsewhere. This update's keyframes are written with
    their final ids by ``update_objects``."""
    import json

    ids = {o.id for o in objs}
    for f in sorted({f for o in objs for f in o.frames} - set(new_views)):
        rec = views.records.get(f)
        if rec is None:
            continue
        rel = frame_file(rec.name, "instances.json")
        p = ctx.tx.current(rel)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        insts = data.get("instances", [])
        hit = False
        for x in insts:
            if masks.owner(int(x["object_id"])) in ids:
                x["object_id"] = 0
                hit = True
        if hit:
            ctx.tx.write_json(rel, data)


def save_state(tx: Any, state: ObjectState) -> None:
    tx.write_json(OBJECTS_JSON, {
        "next_id": state.next_id,
        "floor_z": state.floor_z,
        "merged_into": {str(k): v for k, v in sorted(state.merged_into.items())},
        "objects": [o.to_dict() for o in sorted(state.objects, key=lambda o: o.id)],
        "vacated": [v.to_dict() for v in state.vacated],
    })


def set_cloud_counts(tx: Any, state: ObjectState, labels: NDArray[Any],
                     voxel: float | None = None, focal: float = 0.0,
                     nearest: dict[int, float] | None = None) -> None:
    """Record each object's point count in the map cloud (``labels``: object id per cloud point)
    and, for a cloud of ``voxel`` spacing fused from depth grids of focal length ``focal`` px, the
    count it needs to be exported: ``min_cloud_points`` at the depth of its nearest detection
    (``nearest``: object id → depth, ``geometry.nearest_detections``; else its mean viewing
    distance). Store the state again."""
    ids = np.asarray(labels, np.int64).reshape(-1)
    counts = np.bincount(ids[ids > 0]) if (ids > 0).any() else np.zeros(1, np.int64)
    for o in state.objects:
        o.cloud_points = int(counts[o.id]) if o.id < len(counts) else 0
        seen_from = (nearest or {}).get(o.id, o.obs_depth)
        o.cloud_min = (None if voxel is None or o.obb is None
                       else min_cloud_points(o.obb, voxel, seen_from, focal))
    save_state(tx, state)


# ------------------------------------------------------------------------------------------------
# association


def _object_affinity(ob: Observation, o: MapObject, tree: cKDTree,
                     earlier: _Earlier | None = None) -> float:
    """How well an instance matches an existing object: max(projected-mask IoU, share of the
    instance's points on the object's); when neither reaches the gate, also the IoU of the
    object's mask in its detecting keyframes of earlier updates (``_Earlier``) with the instance's
    projected points — the other direction of ``_pair_affinity``, so an object whose depth
    differs between the updates (its points lie behind the new keyframe's surface, hidden) is
    matched as it would be within one update."""
    s = max(projected_iou(ob.view, ob.inst.mask, o.points),
            overlap_fraction(ob.points, o.points, ob.radius, tree))
    if s < IOU_GATE and earlier is not None:
        for view, mask in earlier.masks(o, ob):
            s = max(s, projected_iou(view, mask, ob.points))
    return s


class _Views:
    """The map's keyframes by index: their poses, their views (this update's from memory, earlier
    ones loaded lazily from the map) and the depth ratio of keyframe pairs (``depth_ratio``),
    cached."""

    def __init__(self, ctx: Any, new: dict[int, View], records: list[Any]) -> None:
        self.ctx = ctx
        self.records = {r.index: r for r in records}
        self.views: dict[int, View | None] = dict(new)
        self.prior: dict[int, View | None] = {}
        self.ratios: dict[tuple[int, int], float | None] = {}

    def pose(self, index: int) -> Pose | None:
        rec = self.records.get(index)
        return None if rec is None else rec.T_map_cam

    def get(self, index: int) -> View | None:
        if index not in self.views:
            rec = self.records.get(index)
            self.views[index] = (None if rec is None or self.ctx is None
                                 else stored_view(self.ctx.tx.current, rec))
        return self.views[index]

    def before(self, index: int) -> View | None:
        """The keyframe's view as the map held it before this update (a stored keyframe's
        committed validity: this update's latest wins not applied); this update's keyframes'
        as ``get``."""
        if index not in self.prior:
            rec = self.records.get(index)
            root = None if self.ctx is None else getattr(self.ctx.tx, "root", None)
            stored = {r.index for r in getattr(self.ctx, "old_frames", None) or []}
            now = self.get(index)
            if now is not None and rec is not None and root is not None and index in stored:
                valid = load_valid(lambda rel: root / rel, rec.name, now.depth)
                now = View(now.depth, valid & (now.depth > 0), now.K, now.T_map_cam)
            self.prior[index] = now
        return self.prior[index]

    def ratio(self, a: int, b: int) -> float | None:
        if (a, b) not in self.ratios:
            va, vb = self.get(a), self.get(b)
            self.ratios[(a, b)] = None if va is None or vb is None else depth_ratio(va, vb)
        return self.ratios[(a, b)]


class _Earlier:
    """Keyframes of earlier updates that detected an existing object: their stored views and the
    object's masks in them (``instances.json``), loaded lazily. ``masks`` gives the
    ``EARLIER_VIEWS`` of them nearest to an instance's viewpoint."""

    def __init__(self, ctx: Any, state: ObjectState, views: _Views) -> None:
        self.ctx = ctx
        self.state = state
        self.records = {r.index: r for r in ctx.old_frames}
        self.views = views
        self.instances: dict[int, list[dict[str, Any]]] = {}

    def _view(self, rec: Any) -> View | None:
        return self.views.get(rec.index)

    def _mask(self, rec: Any, oid: int, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
        if rec.index not in self.instances:
            import json

            p = self.ctx.tx.current(frame_file(rec.name, "instances.json"))
            self.instances[rec.index] = (json.loads(p.read_text()).get("instances", [])
                                         if p.exists() else [])
        masks = [rle.decode(x["mask"]) for x in self.instances[rec.index]
                 if self.state.resolve(int(x["object_id"])) == oid]
        masks = [m for m in masks if m.shape == shape]
        return np.logical_or.reduce(masks) if masks else None

    def masks(self, o: MapObject, ob: Observation) -> list[tuple[View, NDArray[np.bool_]]]:
        T = ob.view.T_map_cam
        scale = max(0.5, ob.depth)

        def distance(f: int) -> tuple[float, int]:
            R = self.records[f].T_map_cam
            return (float(np.linalg.norm(R.t - T.t)) / scale
                    + 1.0 - float(R.R[:, 2] @ T.R[:, 2]), f)

        out = []
        for f in sorted((f for f in o.frames if f in self.records), key=distance)[:EARLIER_VIEWS]:
            view = self._view(self.records[f])
            if view is None:
                continue
            mask = self._mask(self.records[f], o.id, view.depth.shape)
            if mask is not None:
                out.append((view, mask))
        return out


def _pair_affinity(a: Observation, b: Observation) -> float:
    """How well two instances of different keyframes agree (symmetric): the IoU of each one's
    mask with the other's projected points; when neither reaches the gate, also the share of
    either's points on the other's (a partial view whose projection is occluded)."""
    s = max(projected_iou(a.view, a.inst.mask, b.points),
            projected_iou(b.view, b.inst.mask, a.points))
    if s >= IOU_GATE:
        return s
    return max(s, overlap_fraction(a.points, b.points, a.radius, b.tree()),
               overlap_fraction(b.points, a.points, b.radius, a.tree()))


def _compatible_matrix(a: list[str], b: list[str]) -> NDArray[np.bool_]:
    """``compatible`` for every label pair (evaluated once per distinct pair)."""
    ua, ub = sorted(set(a)), sorted(set(b))
    ia, ib = {x: k for k, x in enumerate(ua)}, {x: k for k, x in enumerate(ub)}
    table = np.array([[compatible(x, y) for y in ub] for x in ua], bool).reshape(len(ua), len(ub))
    return table[np.array([ia[x] for x in a], int)[:, None], np.array([ib[y] for y in b], int)]


def _near(ca: NDArray[Any], ea: NDArray[Any], cb: NDArray[Any], eb: NDArray[Any]
          ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Index pairs (i, j) whose centroids lie within max(CENTROID_GATE, 0.6 · larger extent)."""
    if len(ca) == 0 or len(cb) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    r = max(CENTROID_GATE, 0.6 * float(max(ea.max(), eb.max())))
    lists = cKDTree(ca).query_ball_tree(cKDTree(cb), r)
    i = np.array([k for k, js in enumerate(lists) for _ in js], np.int64)
    j = np.array([x for js in lists for x in js], np.int64)
    if len(i) == 0:
        return i, j
    gate = np.maximum(CENTROID_GATE, 0.6 * np.maximum(ea[i], eb[j]))
    ok = np.linalg.norm(ca[i] - cb[j], axis=1) <= gate
    return i[ok], j[ok]


def _neighbour_frames(views: dict[int, View]) -> set[tuple[int, int]]:
    """Keyframe pairs whose instances are compared: each keyframe with its ``PAIR_NEIGHBOURS``
    nearest by viewpoint (camera distance over the scene depth, plus 1 − cos of the angle between
    the optical axes; ties by pose), so the work grows linearly with the update. Chains of
    neighbours still link an object's views across the update, and a place revisited from the
    same viewpoint is a neighbour whenever it was captured."""
    keys = sorted(views, key=lambda f: (*views[f].T_map_cam.t.tolist(),
                                        *views[f].T_map_cam.R.reshape(-1).tolist(), f))
    if len(keys) <= PAIR_NEIGHBOURS + 1:
        return {(a, b) for a in keys for b in keys if a != b}
    C = np.array([views[f].T_map_cam.t for f in keys])
    F = np.array([views[f].T_map_cam.R[:, 2] for f in keys])
    depths = [float(np.median(v.depth[v.valid])) for v in views.values() if v.valid.any()]
    scale = max(0.5, float(np.median(depths))) if depths else 2.0
    d = np.linalg.norm(C[:, None] - C[None], axis=2) / scale + (1.0 - F @ F.T)
    np.fill_diagonal(d, np.inf)
    near = np.argsort(d, axis=1, kind="stable")[:, :PAIR_NEIGHBOURS]
    out: set[tuple[int, int]] = set()
    for i, row in enumerate(near):
        for j in row.tolist():
            out.add((keys[i], keys[j]))
            out.add((keys[j], keys[i]))
    return out


def _candidate_pairs(obs: list[Observation], frames: set[tuple[int, int]] | None = None
                     ) -> list[tuple[int, int]]:
    """Instance pairs (i < j) of different keyframes (of neighbouring keyframes when ``frames``
    is given) with compatible labels and nearby centroids."""
    if len(obs) < 2:
        return []
    c = np.array([ob.centroid for ob in obs])
    e = np.array([ob.extent for ob in obs])
    i, j = _near(c, e, c, e)
    frame = np.array([ob.frame for ob in obs])
    compat = _compatible_matrix([ob.label for ob in obs], [ob.label for ob in obs])
    ok = (i < j) & (frame[i] != frame[j]) & compat[i, j]
    return sorted((a, b) for a, b in zip(i[ok].tolist(), j[ok].tolist(), strict=True)
                  if frames is None or (obs[a].frame, obs[b].frame) in frames)


def _candidate_objects(obs: list[Observation], objs: list[MapObject]
                       ) -> list[tuple[int, int]]:
    """(instance, existing object) pairs with compatible labels and the instance's centroid
    within max(CENTROID_GATE, 0.6 · the object's extent) of the object's."""
    have = [j for j, o in enumerate(objs) if len(o.points)]
    if not obs or not have:
        return []
    c = np.array([ob.centroid for ob in obs])
    co = np.array([objs[j].centroid for j in have])
    eo = np.array([objs[j].extent for j in have])
    i, k = _near(c, np.zeros(len(obs)), co, eo)
    compat = _compatible_matrix([ob.label for ob in obs], [objs[j].label for j in have])
    ok = compat[i, k]
    return sorted(zip(i[ok].tolist(), [have[x] for x in k[ok].tolist()], strict=True))


def _group(obs: list[Observation], objects: list[MapObject], earlier: _Earlier | None = None
           ) -> list[tuple[int | None, list[int]]]:
    """Group this update's instances into objects without regard to the keyframes' order.

    Edges — instance/instance and instance/existing object — are taken strongest first (ties
    by content) and join two groups unless that would put two instances of one keyframe or two
    existing objects in one group, or give an existing object a group most of whose instances do
    not match it directly (a group that grew around a new object is not absorbed through one
    weak link). Returns (existing object id or None, instance indices) per group."""
    n = len(obs)
    objs = list(objects)
    to_obj: dict[tuple[int, int], float] = {}
    # (-strength, kind, content key, content key / object id, node a, node b); kind 0 = existing
    edges: list[tuple[float, int, tuple[Any, ...], tuple[Any, ...], int, int]] = []
    for i, j in _candidate_objects(obs, objs):
        s = _object_affinity(obs[i], objs[j], objs[j].tree(), earlier)
        if s >= IOU_GATE:
            to_obj[(i, j)] = s
            edges.append((-s, 0, obs[i].key, (objs[j].id,), i, n + j))
    views = {ob.frame: ob.view for ob in obs}
    for i, j in _candidate_pairs(obs, _neighbour_frames(views)):
        s = _pair_affinity(obs[i], obs[j])
        if s >= IOU_GATE:
            ka, kb = sorted([obs[i].key, obs[j].key])
            edges.append((-s, 1, ka, kb, i, j))
    edges.sort(key=lambda e: e[:4])

    parent = list(range(n + len(objs)))
    frames: dict[int, set[int]] = {i: {ob.frame} for i, ob in enumerate(obs)}
    frames.update({n + j: set() for j in range(len(objs))})
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    members.update({n + j: [] for j in range(len(objs))})
    anchor: dict[int, int | None] = {i: None for i in range(n)}
    anchor.update({n + j: j for j in range(len(objs))})

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def absorbs(j: int, group: list[int]) -> bool:
        """Existing object ``j`` takes ``group`` only if most of its instances match it."""
        hits = sum((i, j) in to_obj for i in group)
        return 2 * hits >= len(group)

    for *_, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb or frames[ra] & frames[rb]:
            continue
        ja, jb = anchor[ra], anchor[rb]
        if ja is not None and jb is not None:
            continue
        if ja is not None and not absorbs(ja, members[rb]):
            continue
        if jb is not None and not absorbs(jb, members[ra]):
            continue
        parent[rb] = ra
        frames[ra] |= frames.pop(rb)
        members[ra] += members.pop(rb)
        anchor[ra] = ja if ja is not None else jb
        anchor.pop(rb)
    out: list[tuple[int | None, list[int]]] = []
    for r, m in members.items():
        if m:
            k = anchor[r]
            out.append((None if k is None else objs[k].id, sorted(m)))
    return out


# ------------------------------------------------------------------------------------------------
# merging


def containment(a: OBB, b: OBB, pad: float = 0.1, samples: int = 3000) -> float:
    """Fraction of the smaller box (both padded by ``pad``) inside the larger one. Thin objects
    (TVs, pictures, doors) seen from two sides give boxes offset in depth whose IoU is small."""
    pa = OBB(a.center, a.R, a.size + 2 * pad)
    pb = OBB(b.center, b.R, b.size + 2 * pad)
    small, big = (pa, pb) if pa.volume <= pb.volume else (pb, pa)
    rng = np.random.default_rng(0)
    local = rng.uniform(-0.5, 0.5, (samples, 3)) * small.size
    pts = local @ small.R.T + small.center
    return float(big.contains(pts).mean())


def _scale(o: MapObject) -> float:
    """Size of an object: the largest robust (2-98 %) extent — horizontal diagonal or height — of
    the points its box is fitted to (``fit_points``)."""
    def compute() -> float:
        pts = fit_points(o)
        if len(pts) < 2:
            return 0.0
        lo, hi = np.percentile(pts[:, 2], [2, 98])
        return max(_extent(pts), float(hi - lo))
    return float(o.memo("scale", compute))


def _reach(a: MapObject, b: MapObject) -> float:
    """How far apart the copies of one object that disagreeing keyframes placed may lie:
    ``DEPTH_RATIO_MAX`` of the farther one's viewing distance, plus their mean extent."""
    return (DEPTH_RATIO_MAX * max(a.obs_depth, b.obs_depth)
            + (a.extent + b.extent) / 2)


def _viewpoint_distance(Ta: Pose, Tb: Pose, scale: float) -> float:
    """Distance between two viewpoints: camera distance over the scene depth ``scale``, plus
    1 − cos of the angle between the optical axes (as ``_neighbour_frames``)."""
    return float(np.linalg.norm(Ta.t - Tb.t)) / scale + 1.0 - float(Ta.R[:, 2] @ Tb.R[:, 2])


def _nearest_pairs(a: MapObject, b: MapObject, views: _Views, confident: bool = False
                   ) -> list[tuple[float, Sighting, Sighting]]:
    """The ``DEPTH_PAIRS`` pairs (sighting of ``a``, sighting of ``b``) whose keyframes are nearest
    by viewpoint (ties by content), with that distance; ``confident``: only confident keyframes
    (not of low confidence: an unreliable depth scale or a pose the matches do not support) and
    sightings that fit their object (``_fits``)."""
    scale = max(0.5, min(a.obs_depth, b.obs_depth))

    def usable(o: MapObject, s: Sighting) -> Pose | None:
        rec = views.records.get(s.frame)
        if confident and ((rec is not None and getattr(rec, "low_confidence", False))
                          or not _fits(o, s)):
            return None
        return views.pose(s.frame)

    pairs: list[tuple[float, tuple[Any, ...], tuple[Any, ...], Sighting, Sighting]] = []
    for sa in a.sightings:
        Ta = usable(a, sa)
        if Ta is None:
            continue
        for sb in b.sightings:
            Tb = usable(b, sb)
            if Tb is not None:
                pairs.append((_viewpoint_distance(Ta, Tb, scale), sa.key(), sb.key(), sa, sb))
    return [(d, sa, sb) for d, _, _, sa, sb in sorted(pairs, key=lambda p: p[:3])[:DEPTH_PAIRS]]


def _fits(o: MapObject, s: Sighting) -> bool:
    """Whether a sighting spans no more than its object's box (axis-aligned extent) plus the
    depth noise along every axis: a mask that bled onto the surfaces behind (the legs of a chair
    onto the floor and the table: 0.93 m deep for a chair 0.39 m deep) places nothing."""
    if o.obb is None:
        return True
    box = np.abs(o.obb.R) @ np.asarray(o.obb.size, np.float64)
    tol = max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * o.obs_depth)
    return bool(np.all(np.subtract(s.hi, s.lo) <= box + 2 * tol))


def _overlaps(sa: Sighting, sb: Sighting, tol: float, shift: Any = 0.0) -> bool:
    """Whether ``sa``'s bounds, moved by ``shift``, overlap ``sb``'s within ``tol``."""
    lo, hi = np.asarray(sa.lo) + shift, np.asarray(sa.hi) + shift
    return bool(np.all((np.asarray(sb.lo) <= hi + tol) & (lo <= np.asarray(sb.hi) + tol)))


def _depth_explained(a: MapObject, b: MapObject, views: _Views) -> float:
    """>= 1 when two objects of compatible labels that no keyframe detected together are one
    object placed twice by keyframes whose depths disagree (a loop closed by keyframes whose
    depth scale drifted).

    The ``DEPTH_PAIRS`` pairs (sighting of ``a``, sighting of ``b``) whose keyframes are nearest by
    viewpoint are compared. A pair supports the merge when its keyframes' depths disagree by at
    least ``DEPTH_RATIO_MIN`` (``depth_ratio``) and one sighting, scaled about its keyframe's
    camera centre by the inverse of that ratio (into the other keyframe's depth), overlaps the
    other sighting within the depth-noise tolerance of the box consensus. Returns the supporting
    pairs over a strict majority of the pairs compared, the better of the two directions (either
    keyframe's depth may be the one that drifted, and the two ratios are not reciprocal: they are
    measured on what each keyframe sees). Symmetric: the order of ``a`` and ``b`` — the merge
    orders pairs by provisional ids — never matters; tested one way only, a basket placed twice
    across the loop of ``livingroom.mp4`` scored 1.5 one way and 0 the other and stayed two."""
    if not compatible(a.label, b.label) or set(a.frames) & set(b.frames):
        return 0.0
    if not a.sightings or not b.sightings or np.linalg.norm(a.centroid - b.centroid) > _reach(a, b):
        return 0.0
    pairs = _nearest_pairs(a, b, views)
    if not pairs:
        return 0.0
    tol = max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * max(a.obs_depth, b.obs_depth))

    def support(swap: bool) -> int:
        n = 0
        for _, sa, sb in pairs:
            if swap:
                sa, sb = sb, sa
            r = views.ratio(sa.frame, sb.frame)
            Tb = views.pose(sb.frame)
            if r is None or Tb is None or abs(np.log(r)) < np.log1p(DEPTH_RATIO_MIN):
                continue
            c = Tb.t
            moved = Sighting(sb.frame, sb.points, sb.border, sb.centroid,
                             _t3(c + (np.asarray(sb.lo) - c) / r),
                             _t3(c + (np.asarray(sb.hi) - c) / r))
            n += _overlaps(moved, sa, tol)
        return n

    return max(support(False), support(True)) / (len(pairs) // 2 + 1)


def _t3(x: Any) -> tuple[float, float, float]:
    return (float(x[0]), float(x[1]), float(x[2]))


@dataclass
class _Copy:
    """A candidate loop copy (``_loop_copies``): objects ``a`` and ``b``, their nearest keyframe
    pairs, the depth-noise tolerance and the offset from ``a``'s sightings to ``b``'s."""

    a: MapObject
    b: MapObject
    pairs: list[tuple[Sighting, Sighting]]
    tol: float
    offset: NDArray[np.float64]

    def explained(self, shift: NDArray[np.float64]) -> bool:
        """Whether moving ``a``'s sightings by ``shift`` lays them on ``b``'s, for a strict
        majority of the keyframe pairs: along every axis the bounds overlap, within the
        tolerance, by at least ``LOOP_OVERLAP`` of the shorter one (bounds that merely touch are
        neighbours: two pillows side by side on a sofa)."""
        ok = 0
        for sa, sb in self.pairs:
            lo_a, hi_a = np.asarray(sa.lo) + shift, np.asarray(sa.hi) + shift
            lo_b, hi_b = np.asarray(sb.lo), np.asarray(sb.hi)
            common = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b) + self.tol
            ok += bool(np.all(common >= LOOP_OVERLAP * np.minimum(hi_a - lo_a, hi_b - lo_b)))
        return 2 * ok > len(self.pairs)

    @property
    def precise(self) -> bool:
        """Whether its offset pins the displacement down: every sighting spans at most
        ``LOOP_PRECISE`` (the middle of a chair seen from behind and from the front differs by
        half its depth, and its bounds overlap any copy offset by less than its size)."""
        return all(float(np.max(np.subtract(s.hi, s.lo))) <= LOOP_PRECISE
                   for pair in self.pairs for s in pair)


def _loop_copies(objects: list[MapObject], touched: set[int], views: _Views
                 ) -> set[frozenset[int]]:
    """Pairs of objects that are one object placed twice across a loop whose keyframes are
    misaligned (``_Loops``)."""
    return _Loops(objects, touched, views).copies()


class _Loops:
    """Loop copies among ``objects``: the copies of a whole group of objects that keyframes
    misaligned across a loop placed twice are displaced alike (see ``LOOP_*``).

    A candidate pair (``candidates``): compatible labels, no keyframe detected both, each detected
    reliably in at least ``CONFIRM_DETECTIONS`` keyframes, one of them touched by this update,
    centres within ``_reach``; its ``DEPTH_PAIRS`` nearest pairs of confident keyframes are at
    least ``LOOP_VIEW_DIST`` apart by viewpoint (the copies were seen from different places,
    never from neighbouring viewpoints whose alignment the overlap tests trust) and at least one
    of them disagrees in depth by ``DEPTH_RATIO_MIN`` (``depth_ratio``, either direction: the
    keyframes are shown to be misaligned). Its offset is the mean, over those keyframe pairs, of
    the move from the middle of ``a``'s sighting bounds to the middle of ``b``'s.

    Support (``support``): another candidate that links the same keyframes (some keyframe pair of
    each within ``LOOP_LINK`` of the other's, either way round), or an object already joined
    across the loop (its sightings in keyframes linked with each side, ``spanning``), agrees when
    each one's offset lays the other's sightings on its counterpart's (``_Copy.explained``). A
    candidate that ``LOOP_SUPPORT`` agree with is a copy; an object that would be a copy of two
    others is left alone (ambiguous)."""

    def __init__(self, objects: list[MapObject], touched: set[int], views: _Views) -> None:
        self.views = views
        self.robust = [o for o in objects if len(o.points) and o.sightings
                       and len(o.reliable_frames()) >= CONFIRM_DETECTIONS]
        self.scene = max(0.5, float(np.median([o.obs_depth for o in self.robust]))) \
            if self.robust else 1.0
        self._links: dict[tuple[int, int], bool] = {}
        self.cands = self.candidates(touched)

    def candidates(self, touched: set[int]) -> list[_Copy]:
        robust, views = self.robust, self.views
        if len(robust) < 2:
            return []
        cent = np.array([o.centroid for o in robust])
        compat = _compatible_matrix([o.label for o in robust], [o.label for o in robust])
        out: list[_Copy] = []
        for i, j in zip(*np.triu_indices(len(robust), 1), strict=True):
            a, b = robust[int(i)], robust[int(j)]
            if not compat[i, j] or not (a.id in touched or b.id in touched) \
                    or set(a.frames) & set(b.frames) \
                    or np.linalg.norm(cent[i] - cent[j]) > _reach(a, b):
                continue
            near = _nearest_pairs(a, b, views, confident=True)
            if not near or near[0][0] < LOOP_VIEW_DIST:
                continue
            pairs = [(sa, sb) for _, sa, sb in near]
            if _disagree(views, pairs) is False:
                continue
            offset = np.mean([(np.add(sb.lo, sb.hi) - np.add(sa.lo, sa.hi)) / 2
                              for sa, sb in pairs], axis=0)
            tol = max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * max(a.obs_depth, b.obs_depth))
            out.append(_Copy(a, b, pairs, tol, offset))
        return out

    def distance(self, fa: int, fb: int) -> float:
        """Viewpoint distance of two keyframes (0 for one keyframe, inf when one is unknown)."""
        if fa == fb:
            return 0.0
        Ta, Tb = self.views.pose(fa), self.views.pose(fb)
        return np.inf if Ta is None or Tb is None else _viewpoint_distance(Ta, Tb, self.scene)

    def linked(self, fa: int, fb: int) -> bool:
        key = (min(fa, fb), max(fa, fb))
        if key not in self._links:
            self._links[key] = self.distance(fa, fb) <= LOOP_LINK
        return self._links[key]

    def way(self, c: _Copy, d: _Copy) -> int:
        """+1: ``d`` links the keyframes ``c`` links in the same order, -1 reversed, 0 not."""
        for flip in (False, True):
            for sa, sb in c.pairs:
                for sc, sd in d.pairs:
                    if flip:
                        sc, sd = sd, sc
                    if self.linked(sa.frame, sc.frame) and self.linked(sb.frame, sd.frame):
                        return -1 if flip else 1
        return 0

    def spanning(self, c: _Copy, x: MapObject) -> _Copy | None:
        """``x`` seen across the loop: its sightings in the keyframes nearest to (and linked
        with) those of ``c``'s two sides, as a copy of itself; None when it is not."""
        ends: list[Sighting] = []
        for side in ({sa.frame for sa, _ in c.pairs}, {sb.frame for _, sb in c.pairs}):
            best = min(((self.distance(sx.frame, f), sx.key(), sx) for sx in x.sightings
                        for f in sorted(side)), key=lambda t: t[:2], default=None)
            if best is None or best[0] > LOOP_LINK:
                return None
            ends.append(best[2])
        s1, s2 = ends
        if s1.frame == s2.frame:
            return None
        return _Copy(x, x, [(s1, s2)], max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * x.obs_depth),
                     (np.add(s2.lo, s2.hi) - np.add(s1.lo, s1.hi)) / 2)

    def support(self, c: _Copy) -> list[int]:
        """Ids of what agrees with candidate ``c``: the lower id of each agreeing candidate pair,
        and each agreeing object joined across the loop. Only precise ones (``_Copy.precise``)
        vouch for an offset: ``c`` must be explained by theirs, and they by ``c``'s when ``c`` is
        precise itself (a large object's offset is not)."""
        out: list[int] = []

        def agrees(d: _Copy, w: int) -> bool:
            return d.precise and c.explained(w * d.offset) \
                and (not c.precise or d.explained(w * c.offset))

        for d in self.cands:
            if d is c or {d.a.id, d.b.id} & {c.a.id, c.b.id}:
                continue
            w = self.way(c, d)
            if w and agrees(d, w):
                out.append(d.a.id)
        for x in self.robust:
            if x.id in (c.a.id, c.b.id):
                continue
            e = self.spanning(c, x)
            if e is not None and agrees(e, 1):
                out.append(x.id)
        return out

    def copies(self) -> set[frozenset[int]]:
        found = [c for c in self.cands if len(self.support(c)) >= LOOP_SUPPORT]
        count: dict[int, int] = {}
        for c in found:
            for oid in (c.a.id, c.b.id):
                count[oid] = count.get(oid, 0) + 1
        return {frozenset((c.a.id, c.b.id)) for c in found
                if count[c.a.id] == 1 and count[c.b.id] == 1}


def _disagree(views: _Views, pairs: list[tuple[Sighting, Sighting]]) -> bool | None:
    """Whether the keyframes of some of ``pairs`` disagree in depth by at least
    ``DEPTH_RATIO_MIN`` (``depth_ratio``, either way); None when no pair shares enough surface to
    tell (seen from opposite sides)."""
    ratios = [r for sa, sb in pairs
              for r in (views.ratio(sa.frame, sb.frame), views.ratio(sb.frame, sa.frame))
              if r is not None]
    if not ratios:
        return None
    return any(abs(np.log(r)) >= np.log1p(DEPTH_RATIO_MIN) for r in ratios)


def _surface_kinds(a: MapObject, b: MapObject) -> bool:
    """Whether two objects may be pieces of one horizontal surface by their labels: some labels
    of the two (their label or any label they were detected as) are compatible surface labels
    (``split_surface``), or both are labelled as horizontal surfaces of whatever kind
    (``surface_label``: the detector names one counter top a desk from one side and a rug from
    another; the geometry of ``_one_surface`` decides)."""
    la, lb = {a.label, *a.label_votes}, {b.label, *b.label_votes}
    if any(split_surface(x, y) for x in sorted(la) for y in sorted(lb)):
        return True
    return surface_label(a.label) and surface_label(b.label)


class _Surfaces:
    """The map's fused surface (``geometry.fuse_map``) as evidence that two pieces are one
    horizontal surface (``joined``; see ``BRIDGE_*``), with the results cached per pair of
    point sets."""

    def __init__(self, cloud: NDArray[Any] | Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]],
                 floor_z: float | None) -> None:
        """``cloud``: the surface points, or a function giving those within a box (lowest,
        highest corner; ``geometry.SurfaceQuery``: fused there only when asked)."""
        if callable(cloud):
            self.query = cloud
        else:
            pts = np.asarray(cloud, np.float64).reshape(-1, 3)
            self.query = lambda lo, hi: pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
        self.floor_z = floor_z
        self.cache: dict[tuple[Any, ...], bool] = {}

    def joined(self, a: MapObject, b: MapObject) -> bool:
        key = (a.id, b.id, len(a.points), len(b.points), a.points[:1].tobytes(),
               b.points[:1].tobytes())
        if key not in self.cache:
            self.cache[key] = self._joined(a, b)
        return self.cache[key]

    def _joined(self, a: MapObject, b: MapObject) -> bool:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        from oh_my_slam.core.geometry import voxel_downsample_indices

        ha, hb = float(np.median(a.points[:, 2])), float(np.median(b.points[:, 2]))
        if self.floor_z is not None and max(ha, hb) < self.floor_z + BRIDGE_FLOOR_M:
            return False
        gap, _ = cKDTree(a.points[:, :2]).query(b.points[:, :2], k=1,
                                                distance_upper_bound=BRIDGE_GAP_M)
        if not np.isfinite(gap).any():
            return False
        both = np.concatenate([a.points, b.points]).astype(np.float64)
        lo, hi = both[:, :2].min(0) - BRIDGE_PAD, both[:, :2].max(0) + BRIDGE_PAD
        p = np.asarray(self.query(np.r_[lo, min(ha, hb) - BRIDGE_BAND],
                                  np.r_[hi, max(ha, hb) + BRIDGE_BAND]), np.float64).reshape(-1, 3)
        if len(p) < 50:
            return False
        p = p[np.sort(voxel_downsample_indices(p, BRIDGE_SAMPLE))]
        if len(p) < 50:
            return False
        _, nn = cKDTree(p).query(p, k=10)
        q = p[nn] - p[nn].mean(axis=1, keepdims=True)
        _, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", q, q))
        flat = p[np.abs(vec[:, 2, 0]) >= BRIDGE_NORMAL_Z]
        if len(flat) < 50:
            return False
        keys = np.unique(np.floor(flat / BRIDGE_VOXEL).astype(np.int64), axis=0)
        links = cKDTree(keys).query_pairs(1.8, output_type="ndarray")  # 26-neighbourhood
        graph = coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])),
                           shape=(len(keys), len(keys)))
        _, comp = connected_components(graph, directed=False)
        centres = cKDTree((keys + 0.5) * BRIDGE_VOXEL)

        def on(o: MapObject) -> NDArray[np.int64]:
            d, j = centres.query(np.asarray(o.points, np.float64), k=1,
                                 distance_upper_bound=BRIDGE_NEAR)
            return np.asarray(comp[j[np.isfinite(d)]], np.int64)

        ca, cb = on(a), on(b)
        if len(ca) < BRIDGE_MIN_POINTS or len(cb) < BRIDGE_MIN_POINTS:
            return False
        for patch in np.intersect1d(ca, cb).tolist():
            if (ca == patch).mean() >= BRIDGE_SHARE and (cb == patch).mean() >= BRIDGE_SHARE:
                return True
        return False


def _one_surface(a: MapObject, b: MapObject, surfaces: _Surfaces | None = None) -> float:
    """>= 1 when two objects are pieces of one horizontal surface seen from different keyframes
    (a counter top around the camera, labelled desk from one side and bed or rug from another):
    their labels allow it (``_surface_kinds``), no keyframe detected both (it would have seen two
    things there), and either their median heights agree within ``SURFACE_HEIGHT_TOL`` (a rug
    under a table is a different surface) and they share surface — the points of ``a`` within
    ``SURFACE_PLAN_M`` horizontally and ``SURFACE_HEIGHT_TOL`` vertically of ``b``'s, over
    ``SURFACE_SHARED_POINTS`` — or the map's fused surface joins them (``surfaces``:
    ``_Surfaces.joined``)."""
    if not _surface_kinds(a, b):
        return 0.0
    if set(a.frames) & set(b.frames) or len(a.points) == 0 or len(b.points) == 0:
        return 0.0
    dz = abs(float(np.median(a.points[:, 2]) - np.median(b.points[:, 2])))
    shared = 0.0
    if dz <= SURFACE_HEIGHT_TOL:
        k = np.array([1.0, 1.0, SURFACE_PLAN_M / SURFACE_HEIGHT_TOL])
        d, _ = cKDTree(b.points * k).query(a.points * k, k=1,
                                           distance_upper_bound=SURFACE_PLAN_M)
        shared = float(np.isfinite(d).sum()) / SURFACE_SHARED_POINTS
    if shared < 1.0 and surfaces is not None and dz <= BRIDGE_HEIGHT_TOL \
            and surfaces.joined(a, b):
        return 1.0
    return shared


@dataclass
class _Grown:
    """An object's detections in one keyframe, grown by the depth noise (``SEEN_*``): a crop of
    the grid at (``r0``, ``c0``)."""

    r0: int
    c0: int
    mask: NDArray[np.bool_]

    def covers(self, u: NDArray[np.int64], v: NDArray[np.int64]) -> NDArray[np.bool_]:
        h, w = self.mask.shape
        r, c = v - self.r0, u - self.c0
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        out = np.zeros(len(u), bool)
        out[ok] = self.mask[r[ok], c[ok]]
        return out


class _Masks:
    """The detection masks of the objects in their keyframes, for ``_seen_as_one``: this update's
    instances (``instances``: keyframe -> [(object id, mask)], in memory) and those of the stored
    keyframes (their ``instances.json``, loaded lazily), by the object that owns them now (ids
    resolved through the map's earlier merges and this update's, ``alias``). Grown masks and test
    results are cached; ``forget`` drops those of an object whose evidence changed."""

    def __init__(self, ctx: Any, views: _Views, instances: dict[int, list[tuple[int, Any]]],
                 merged_into: dict[int, int], alias: dict[int, int]) -> None:
        self.ctx = ctx
        self.views = views
        self.frames = {f: list(v) for f, v in instances.items()}
        self.merged_into = merged_into
        self.alias = alias
        self.grown: dict[tuple[int, int], _Grown | None] = {}
        self.results: dict[tuple[Any, ...], float] = {}

    def owner(self, oid: int) -> int:
        seen: set[int] = set()
        while oid not in seen:
            seen.add(oid)
            nxt = self.alias.get(oid, self.merged_into.get(oid))
            if nxt is None:
                break
            oid = nxt
        return oid

    def instances(self, f: int) -> list[tuple[int, Any]]:
        if f not in self.frames:
            import json

            rec = self.views.records.get(f)
            out: list[tuple[int, Any]] = []
            if rec is not None and self.ctx is not None:
                p = self.ctx.tx.current(frame_file(rec.name, "instances.json"))
                if p.exists():
                    out = [(int(x["object_id"]), x["mask"])
                           for x in json.loads(p.read_text()).get("instances", [])]
            self.frames[f] = out
        return self.frames[f]

    def mask(self, o: MapObject, f: int, view: View) -> _Grown | None:
        """``o``'s detections in keyframe ``f`` grown by the depth noise at its distance there."""
        key = (o.id, f)
        if key not in self.grown:
            masks = [m if isinstance(m, np.ndarray) else rle.decode(m)
                     for oid, m in self.instances(f) if self.owner(oid) == o.id]
            masks = [m for m in masks if m.shape == view.depth.shape]
            self.grown[key] = _grow(np.logical_or.reduce(masks), view) if masks else None
        return self.grown[key]

    def forget(self, *ids: int) -> None:
        drop = set(ids)
        self.grown = {k: v for k, v in self.grown.items() if k[0] not in drop}


def _grow(mask: NDArray[np.bool_], view: View) -> _Grown | None:
    """``mask`` grown by the depth noise at the median depth under it (``SEEN_*``)."""
    box = _bbox(mask)
    if box is None:
        return None
    ok = mask & view.valid & (view.depth > 0)
    z = float(np.median(view.depth[ok])) if ok.any() else 1.0
    r = int(np.ceil(max(MASK_DILATE,
                        max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * z) * view.K.fx / z)))
    h, w = mask.shape
    r0, r1 = max(0, box[0] - r), min(h, box[1] + r)
    c0, c1 = max(0, box[2] - r), min(w, box[3] + r)
    near = ndimage.distance_transform_edt(~mask[r0:r1, c0:c1]) <= r
    return _Grown(r0, c0, near)


def _seen_samples(o: MapObject) -> NDArray[np.float64]:
    pts = np.asarray(o.points, np.float64)
    if len(pts) > SEEN_SAMPLES:
        pts = pts[np.linspace(0, len(pts) - 1, SEEN_SAMPLES).astype(int)]
    return pts


def _judged(o: MapObject, pts: NDArray[np.float64], masks: _Masks) -> NDArray[np.float64]:
    """How the keyframes that detected ``o`` saw the points ``pts`` (of another object): counts of
    (on its detections, in free space, beside its detections, judged), over the unoccluded points
    in view of each keyframe (``SEEN_*``)."""
    out = np.zeros(4)
    for f in o.frames:
        out += _judged_in(o, f, pts, masks)
    return out


def _judged_in(o: MapObject, f: int, pts: NDArray[np.float64], masks: _Masks
               ) -> NDArray[np.float64]:
    """``_judged`` for one keyframe ``f`` that detected ``o``."""
    view = masks.views.get(f)
    grown = None if view is None else masks.mask(o, f, view)
    if view is None or grown is None:
        return np.zeros(4)
    inside, z, d, u, v = view.lookup_pixels(pts)
    keep = inside & (z - d <= tau(z))  # occluded points tell nothing
    z, d, u, v = z[keep], d[keep], u[keep], v[keep]
    free = d - z > np.maximum(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * z)
    on = grown.covers(u, v) & ~free
    return np.array([on.sum(), free.sum(), (~on & ~free).sum(), len(z)], np.float64)


def _ray_origin(o: MapObject, views: _Views) -> NDArray[np.float64] | None:
    """Mean camera centre of an object's sightings (weighted by their points)."""
    centres, weights = [], []
    for s in o.sightings:
        T = views.pose(s.frame)
        if T is not None:
            centres.append(T.t)
            weights.append(float(max(s.points, 1)))
    return None if not centres else np.average(np.asarray(centres), axis=0, weights=weights)


def _placed(o: MapObject, pts: NDArray[np.float64], origin: NDArray[np.float64], masks: _Masks
            ) -> tuple[float, float] | None:
    """(depth factor, share on ``o``'s detections) of the factor nearest 1 that moves ``pts``
    along their viewing rays (about ``origin``) onto the detections of ``o`` (``SEEN_ON``)."""
    step = np.log1p(SEEN_DEPTH) / SEEN_STEPS
    for k in range(2 * SEEN_STEPS + 1):  # factors 1, 1 + x, 1 / (1 + x), ...
        factor = float(np.exp(step * ((k + 1) // 2) * (1 if k % 2 else -1)))
        c = _judged(o, origin + factor * (pts - origin), masks)
        if c[3] >= SEEN_EVIDENCE and c[0] >= SEEN_ON * c[3]:
            return factor, float(c[0] / c[3])
    return None


def _meet_in_plan(a: NDArray[np.float64], b: NDArray[np.float64], tol: float) -> bool:
    """Whether each of two point sets has its middle (of its 2-98 % bounds) within the other's
    bounds grown by ``tol``, along both horizontal axes."""
    lo_a, hi_a = np.percentile(a[:, :2], [2, 98], axis=0)
    lo_b, hi_b = np.percentile(b[:, :2], [2, 98], axis=0)
    mid_a, mid_b = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2
    return bool(np.all((mid_a >= lo_b - tol) & (mid_a <= hi_b + tol)
                       & (mid_b >= lo_a - tol) & (mid_b <= hi_a + tol)))


def _seen_as_one(a: MapObject, b: MapObject, masks: _Masks) -> float:
    """>= 1 when two objects that no keyframe detected together are one object seen twice: by
    keyframes on different sides that placed it at different depths, or under another label
    (``SEEN_*``). Both must be detected reliably in ``CONFIRM_DETECTIONS`` keyframes, their centres
    within ``_reach``. The value is the smaller on-detection share over ``SEEN_ON``; 0 when the
    keyframes saw them apart, or saw too little to tell."""
    if set(a.frames) & set(b.frames) or not a.sightings or not b.sightings \
            or min(len(a.reliable_frames()), len(b.reliable_frames())) < CONFIRM_DETECTIONS \
            or len(a.points) == 0 or len(b.points) == 0 \
            or np.linalg.norm(a.centroid - b.centroid) > _reach(a, b):
        return 0.0
    key = (a.id, b.id, len(a.points), len(b.points), tuple(a.frames), tuple(b.frames))
    if key not in masks.results:
        masks.results[key] = _seen_twice(a, b, masks)
    return masks.results[key]


def _seen_twice(a: MapObject, b: MapObject, masks: _Masks) -> float:
    """The tests of ``_seen_as_one`` once its gates are passed (``SEEN_*``)."""
    pa, pb = _seen_samples(a), _seen_samples(b)
    for c in (_judged(a, pb, masks), _judged(b, pa, masks)):
        if c[3] < SEEN_EVIDENCE or c[2] > SEEN_APART * c[3]:
            return 0.0
    ca, cb = _ray_origin(a, masks.views), _ray_origin(b, masks.views)
    if ca is None or cb is None:
        return 0.0
    on_a, on_b = _placed(a, pb, cb, masks), _placed(b, pa, ca, masks)
    if on_a is None or on_b is None:
        return 0.0
    tol = max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * max(a.obs_depth, b.obs_depth))
    if not _meet_in_plan(ca + on_b[0] * (pa - ca), cb + on_a[0] * (pb - cb), tol):
        return 0.0
    return min(on_a[1], on_b[1]) / SEEN_ON


def _part_whole(a: MapObject, b: MapObject) -> tuple[MapObject, MapObject, float] | None:
    """(part, whole, part's scale over the whole's) when one of two objects is smaller than
    ``MERGE_SCALE`` of the other (``_scale``); None otherwise."""
    sa, sb = _scale(a), _scale(b)
    if min(sa, sb) >= MERGE_SCALE * max(sa, sb) or max(sa, sb) <= 0:
        return None
    return (a, b, sa / sb) if sa < sb else (b, a, sb / sa)


def _sees(view: View, pts: NDArray[np.float64]) -> bool:
    """Whether a keyframe had ``pts`` in view: ``PART_FRAMED`` of them in its image, and of those
    it can judge (at least ``PART_EVIDENCE``) ``VISIBLE_SHARE`` unoccluded."""
    if len(pts) == 0:
        return False
    uv, zc = project(view.T_map_cam.inverse().apply(pts), view.K.K())
    h, w = view.depth.shape
    with np.errstate(invalid="ignore"):
        framed = (zc > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if framed.mean() < PART_FRAMED:
        return False
    inside, z, d = view.lookup(pts)
    z, d = z[inside], d[inside]
    return len(z) >= PART_EVIDENCE and float((z - d <= tau(z)).mean()) >= VISIBLE_SHARE


def _part_of(a: MapObject, b: MapObject, views: _Views, masks: _Masks | None = None) -> float:
    """>= 1 when the smaller of two objects of incompatible labels that no keyframe detected
    together is a part of the larger that the detector named on its own (``PART_*``); with
    ``masks`` (the objects' detections), the whole's keyframes must also have seen the part's
    place as the whole: in each, ``PART_ON`` of the part's points it judges lie on its detection
    or in free space, more on it than free (``_judged_in``). The value is the share of the part inside the whole's box over
    ``PART_INSIDE``."""
    pw = _part_whole(a, b)
    if pw is None or set(a.frames) & set(b.frames) \
            or min(len(a.reliable_frames()), len(b.reliable_frames())) < CONFIRM_DETECTIONS:
        return 0.0
    part, whole, ratio = pw
    if ratio < PART_MIN_SCALE:
        return 0.0
    own = np.asarray(fit_points(part), np.float64)
    if whole.obb is None or len(own) == 0:
        return 0.0
    inside = float(whole.obb.contains(own, CONSENSUS_MARGIN).mean())
    near = max(POINT_VOXEL, PART_NEAR_REL * part.obs_depth)
    if inside < PART_INSIDE or overlap_fraction(own, whole.points, near,
                                                whole.tree()) < PART_SURFACE:
        return 0.0
    reach = [float(np.linalg.norm(T.t - np.asarray(s.centroid)))
             for s in part.sightings if (T := views.pose(s.frame)) is not None]
    if not reach:
        return 0.0
    samples = own[np.linspace(0, len(own) - 1, min(len(own), SEEN_SAMPLES)).astype(int)]
    for o, other in ((part, whole), (whole, part)):
        pts = samples if other is part else _seen_samples(other)
        for f in o.frames:
            view = views.get(f)
            if view is None or not _sees(view, pts):
                return 0.0
            if o is whole and float(np.linalg.norm(view.T_map_cam.t - part.centroid)) \
                    > PART_RANGE * max(reach):
                return 0.0
    if masks is not None:
        judges = 0
        for f in whole.frames:
            on, free, _, judged = _judged_in(whole, f, samples, masks)
            if judged < PART_EVIDENCE:
                continue
            if on + free < PART_ON * judged or on < free:
                return 0.0
            judges += 1
        if judges < CONFIRM_DETECTIONS:
            return 0.0
    return inside / PART_INSIDE


def _merge_strength(a: MapObject, b: MapObject, views: _Views | None = None,
                    surfaces: _Surfaces | None = None,
                    loops: set[frozenset[int]] | None = None,
                    masks: _Masks | None = None, parts: _Masks | None = None) -> float:
    """>= 1 when two objects are one physical object. Compatible labels: >= 50 % of the smaller
    one's points on the other, or boxes overlapping (IoU >= 0.3, or the smaller padded box >= 60 %
    inside the other), or (with ``views``) copies placed by keyframes whose depths disagree
    (``_depth_explained``), or copies of a loop whose keyframes are misaligned (``loops``: pairs
    of ids from ``_loop_copies``; still never detected together). Incompatible labels (the
    detector's label flickered between keyframes): no keyframe detected both (it would have seen
    two things there), their sizes are comparable (``MERGE_SCALE``: neither is a part of the other
    or an item resting on it) and >= 50 % of either one's points lie on the other's surface, or
    (with ``views``) the smaller is a part of the larger that the detector named on its own
    (``_part_of``). Whatever the labels: pieces of one horizontal surface (``_one_surface``, with ``surfaces``:
    the map's fused surface); with ``masks`` (the detections of the objects in their keyframes),
    one object seen twice by keyframes that never saw it together (``_seen_as_one``; incompatible
    labels of comparable size only); ``parts``: the detections for ``_part_of``. The value
    orders the merges (strongest first)."""
    same_kind = compatible(a.label, b.label)
    if not same_kind and set(a.frames) & set(b.frames):
        return 0.0
    if np.linalg.norm(a.centroid - b.centroid) > max(CENTROID_GATE * 2, a.extent + b.extent):
        return 0.0
    surface = _one_surface(a, b, surfaces)
    if not same_kind:
        if _part_whole(a, b) is not None:
            if surface < 1.0 and views is not None:
                return max(surface, _part_of(a, b, views, parts))
            return surface
        radius = max(0.05, 0.02 * min(a.obs_depth, b.obs_depth))
        s = max(surface, overlap_fraction(a.points, b.points, radius, b.tree()) / MERGE_OVERLAP,
                overlap_fraction(b.points, a.points, radius, a.tree()) / MERGE_OVERLAP)
        return max(s, _seen_as_one(a, b, masks)) if s < 1.0 and masks is not None else s
    small, big = (a, b) if len(a.points) <= len(b.points) else (b, a)
    radius = max(0.05, 0.02 * small.obs_depth)
    s = max(surface, overlap_fraction(small.points, big.points, radius, big.tree())
            / MERGE_OVERLAP)
    if a.obb is not None and b.obb is not None:
        s = max(s, obb_iou_upright(a.obb, b.obb, samples=3000) / MERGE_BOX_IOU,
                containment(a.obb, b.obb) / MERGE_CONTAINMENT)
    if s < 1.0 and views is not None:
        s = max(s, _depth_explained(a, b, views))
    if s < 1.0 and loops and frozenset((a.id, b.id)) in loops \
            and not set(a.frames) & set(b.frames):
        s = 1.0
    if s < 1.0 and masks is not None:
        s = max(s, _seen_as_one(a, b, masks))
    return s


def _weigh_part(a: MapObject, b: MapObject, views: _Views | None, surfaces: _Surfaces | None,
                masks: _Masks | None = None) -> None:
    """Before two objects merge: when one is a part of the other named on its own (``_part_of``,
    not pieces of one surface), its label votes are scaled by its size over the whole's — its
    label names a part (``PART_*``)."""
    if views is None or compatible(a.label, b.label):
        return
    pw = _part_whole(a, b)
    if pw is None or _one_surface(a, b, surfaces) >= 1.0 \
            or _part_of(a, b, views, masks) < 1.0:
        return
    part, _, ratio = pw
    part.label_votes = {k: v * ratio for k, v in part.label_votes.items()}


def _content_key(o: MapObject) -> tuple[Any, ...]:
    return (o.label, *o.centroid.tolist(), len(o.points))


def _merge(state: ObjectState, touched: set[int], alias: dict[int, int],
           views: _Views | None = None, surfaces: _Surfaces | None = None,
           masks: _Masks | None = None) -> int:
    """Merge duplicates among the objects (at least one of each pair touched by this update),
    strongest pair first; the lower id is kept and ``alias`` maps each merged id to its keeper.
    ``views`` (the map's keyframes) enables the depth-explained test (``_depth_explained``) and
    the loop copies (``_loop_copies``: sought among the objects once no other merge is left —
    the copies of a group are recognised on whole objects, not on the pieces the other tests
    join — and merged like the rest; again until none is found), ``surfaces`` (the map's fused
    surface) the surface-continuity test (``_Surfaces``), ``masks`` (the objects' detections) the
    objects seen twice (``_seen_as_one``: tested, likewise on whole objects, once no other merge
    and no loop copy is left, then with every other test)."""
    strength: dict[tuple[int, int], float] = {}
    loops: set[frozenset[int]] = set()
    seen: _Masks | None = None

    def pairs_of(o: MapObject) -> None:
        for p in state.objects:
            if p.id == o.id or not (o.id in touched or p.id in touched):
                continue
            a, b = (o, p) if o.id < p.id else (p, o)
            s = _merge_strength(a, b, views, surfaces, loops, seen, masks)
            if s >= 1.0:
                strength[(a.id, b.id)] = s
            else:
                strength.pop((a.id, b.id), None)

    def exhausted() -> None:
        """No merge left: look for loop copies, then (once) for objects seen twice."""
        nonlocal loops, seen
        if views is not None:
            found = _loop_copies(state.objects, touched, views) - loops
            loops |= found
            for oid in sorted({x for p in found for x in p}):
                pairs_of(by[oid])
        if strength or masks is None or seen is not None:
            return
        seen = masks
        strength.update(_seen_pairs(state.objects, touched, seen))

    by = state.by_id()
    for o in sorted(state.objects, key=lambda o: o.id):
        if o.id in touched:
            pairs_of(o)
    merged = 0
    if not strength:
        exhausted()
    while strength:
        (ka, kb), _ = min(strength.items(), key=lambda kv: (
            -kv[1], sorted([_content_key(by[kv[0][0]]), _content_key(by[kv[0][1]])])))
        keep, gone = by[ka], by[kb]  # ka < kb: the lower id is kept
        _weigh_part(keep, gone, views, surfaces, masks)
        keep.absorb(gone)
        refit(keep, state.floor_z)
        alias[gone.id] = keep.id
        if masks is not None:
            masks.forget(keep.id, gone.id)
        state.objects = [o for o in state.objects if o.id != gone.id]
        del by[gone.id]
        touched.add(keep.id)
        touched.discard(gone.id)
        loops = {frozenset(keep.id if x == gone.id else x for x in p) for p in loops}
        loops = {p for p in loops if len(p) == 2}
        strength = {k: v for k, v in strength.items() if gone.id not in k and keep.id not in k}
        pairs_of(keep)
        merged += 1
        if not strength:
            exhausted()
    return merged


def _seen_pairs(objects: list[MapObject], touched: set[int], masks: _Masks
                ) -> dict[tuple[int, int], float]:
    """The pairs (ids, lower first; at least one touched by this update) of objects seen twice
    (``_seen_as_one``) under the label conditions of ``_merge_strength``, with their strength.
    The pairs are first gated all at once (centres within ``_reach`` and the merge gate, reliable
    detections)."""
    objs = sorted((o for o in objects if len(o.points) and o.sightings
                   and len(o.reliable_frames()) >= CONFIRM_DETECTIONS), key=lambda o: o.id)
    if len(objs) < 2:
        return {}
    c = np.array([o.centroid for o in objs])
    e = np.array([o.extent for o in objs])
    depth = np.array([o.obs_depth for o in objs])
    hit = np.array([o.id in touched for o in objs])
    i, j = np.triu_indices(len(objs), 1)
    near = np.linalg.norm(c[i] - c[j], axis=1) <= np.minimum(
        DEPTH_RATIO_MAX * np.maximum(depth[i], depth[j]) + (e[i] + e[j]) / 2,
        np.maximum(CENTROID_GATE * 2, e[i] + e[j]))
    out: dict[tuple[int, int], float] = {}
    scale: dict[int, float] = {}
    for x, y in zip(i[near & (hit[i] | hit[j])].tolist(), j[near & (hit[i] | hit[j])].tolist(),
                    strict=True):
        a, b = objs[x], objs[y]
        if set(a.frames) & set(b.frames):
            continue
        if not compatible(a.label, b.label):
            for o in (a, b):
                if o.id not in scale:
                    scale[o.id] = _scale(o)
            if min(scale[a.id], scale[b.id]) < MERGE_SCALE * max(scale[a.id], scale[b.id]):
                continue
        s = _seen_as_one(a, b, masks)
        if s >= 1.0:
            out[(a.id, b.id)] = s
    return out


# ------------------------------------------------------------------------------------------------
# latest wins for objects: how keyframes saw an object's place


def _judges(rec: Any, o: MapObject, size: float) -> bool:
    """Whether keyframe ``rec`` (a record) is registered well enough to judge object ``o``
    (extent ``size``): well registered (``well_registered``); or its pose error, seen from the
    object's distance, is small against the object (``ABSENCE_POSE_SHARE`` of its extent: a cup
    0.7 m away, 2° off, is 2.5 cm off): an SfM pose with ``validity.MIN_OBSERVATIONS`` points
    whose reprojection error, as an angle (error in pixels over the focal length), is below that,
    or a pose refined with feature matches (``stats`` has ``pose_matches``, whatever the map's
    ``pose_source``: a map joins SfM and multi-view keyframes) whose match residual is."""
    if well_registered(rec.stats, rec.pose_source):
        return True
    limit = float(np.degrees(ABSENCE_POSE_SHARE * size / max(o.obs_depth, 0.1)))
    if "pose_matches" not in rec.stats:
        err = rec.stats.get("reproj_error")
        fx = getattr(getattr(rec, "K", None), "fx", 0.0)
        return (rec.stats.get("observations", 0) >= MIN_OBSERVATIONS and err is not None
                and fx > 0 and float(np.degrees(float(err) / fx)) <= limit)
    return pose_supported(rec.stats, max(POSE_MAX_RESIDUAL_DEG, limit))


@dataclass(frozen=True)
class Verdict:
    """How one keyframe saw an object's place (``_Places.verdict``): the share of the object's
    judged samples it saw through, whether its pose is well supported (many SfM inliers or feature
    matches: the few-keyframes rule of ``_judgement``), and its depth ratio to the object's
    detections there (measured on the object's surroundings; its depth is divided by it).

    ``share`` leaves out the samples where the keyframe sees the object's support at its foot,
    which cannot tell whether the object is there (``_evidence``); ``held`` counts them as seen in
    place. A keyframe sees the object in place (``in_place``) by ``held``, and through it by
    ``share``: what cannot tell never removes an object, nor keeps a keyframe that saw its place
    empty from removing it."""

    frame: int
    share: float
    supported: bool
    ratio: float = 1.0
    held: float | None = None  # ``share`` with the object's foot counted as seen in place

    @property
    def in_place(self) -> bool:
        """The keyframe saw the object where it was: at most 1 - ``REMOVE_FRACTION`` of it seen
        through (``held``)."""
        return (self.share if self.held is None else self.held) <= 1.0 - REMOVE_FRACTION


@dataclass(frozen=True)
class _Footprint:
    """Detections of an object in a keyframe (``_Places._footprint``)."""

    shape: tuple[int, ...]
    box: tuple[int, int, int, int]
    row: float  # centre
    col: float


class _Places:
    """Latest wins for objects: how the map's keyframes saw the places of its objects.

    A keyframe judges an object's place (``verdict``) only where it could have detected the
    object: well registered (``_judges``), not low confidence, at least ``PLACE_FRAMED`` of the
    object's samples in its image (the image-border band counts: a place at the edge of the latest
    photo is seen), from at most ``PLACE_RANGE`` times the distance the object was detected from,
    and at least ``VISIBLE_SHARE`` of the samples unoccluded (``PLACE_MIN_SAMPLES`` at least).

    Monocular depth of two keyframes disagrees — by a largely smooth factor between keyframes far
    apart, tens of percent between viewpoints metres apart, and by pose errors where a video
    closes a loop — so the keyframe must also agree with the object's detections about its
    surroundings: the surface around the object's mask in the detecting keyframe nearest by
    viewpoint (a band ``RING_WIDTH`` of the mask's size wide: its support, the wall behind it),
    lifted into the map, must lie on the judging keyframe's depth for ``VISIBLE_SHARE`` of the
    points it sees there, once their median ratio (at most ``RING_BIAS`` from 1) is removed, within
    the depth noise (``absence_tau``'s floor). Its depth is divided by that ratio. A keyframe that
    disagrees locally (a loop misaligned by decimetres, another scale there) does not judge.

    A sample is seen through when the keyframe sees farther than it by the absence margin
    (``absence_tau``, which scales with the object's size). The verdict is the share seen through
    beyond what the object's own detecting keyframes see through (``transparency``): monocular
    depth sees between a ladder's rungs and a plant's leaves, and places a small object
    differently from one keyframe to the next; a keyframe that sees through no more of it than
    they do is no evidence."""

    def __init__(self, views: _Views, masks: _Masks, objects: list[MapObject],
                 cuts: dict[int, float] | None = None) -> None:
        """``cuts``: per keyframe index, how deep the map fuses it (``geometry.
        keyframe_depth_cuts``); without them every depth is judged."""
        self.views = views
        self.masks = masks
        self.objects = {o.id: o for o in objects}
        self.cuts = cuts
        self._fused: dict[int, bool] = {}
        self._rings: dict[tuple[int, int], NDArray[np.float64] | None] = {}
        self._noise: dict[int, tuple[float, float]] = {}
        self._edges: dict[int, NDArray[np.bool_]] = {}
        # per (object id, keyframe): the shape, bounding box and centre of its detections there
        # (``_overlap`` compares them for every object a keyframe detected near another one)
        self._footprints: dict[tuple[int, int], _Footprint | None] = {}

    def _edge(self, f: int, view: View) -> NDArray[np.bool_]:
        if f not in self._edges:
            ok = view.valid & (view.depth > 0)
            self._edges[f] = depth_edge_mask(np.where(ok, view.depth, 0.0))
        return self._edges[f]

    @staticmethod
    def samples(o: MapObject) -> tuple[NDArray[np.float64], float]:
        def compute() -> tuple[NDArray[np.float64], float]:
            pts = np.asarray(o.points, np.float64)
            if len(pts) > SEEN_SAMPLES:
                pts = pts[np.linspace(0, len(pts) - 1, SEEN_SAMPLES).astype(int)]
            return pts, float(np.linalg.norm(np.ptp(pts, axis=0))) if len(pts) else 0.0
        out: tuple[NDArray[np.float64], float] = o.memo("place_samples", compute)
        return out

    def reach(self, o: MapObject) -> float:
        """The farthest distance a keyframe detected the object from."""
        far = [float(np.linalg.norm(np.asarray(s.centroid) - T.t)) for s in o.sightings
               if (T := self.views.pose(s.frame)) is not None]
        return max(far) if far else float(o.obs_depth)

    def _detecting(self, o: MapObject, n: int) -> list[int]:
        size = {s.frame: s.points for s in o.sightings}
        return sorted(o.frames, key=lambda f: (-size.get(f, 0), f))[:n]

    def transparency(self, o: MapObject) -> tuple[float, float]:
        """The share of the object's samples its own detecting keyframes see through (by the
        margin of ``verdict``; the largest over up to ``PLACE_REFS`` of them, the largest
        detections): monocular depth sees between the rungs of a ladder or the leaves of a plant,
        and places a small object differently from one keyframe to the next. A keyframe that
        judges the object must see through it by more than that (``verdict``). As for the
        verdict's ``share`` and ``held``: without and with the samples where the keyframes see
        the object's support at its foot (``_evidence``)."""
        if o.id not in self._noise:
            pts, size = self.samples(o)
            worst = [0.0, 0.0]
            for f in self._detecting(o, PLACE_REFS):
                view = self.views.before(f)
                if view is None:
                    continue
                ok, z, d, u, v = _lookup_valid(view, pts)
                through, judged, held = _evidence(view, ok, z, d, u, v, size, self.foot(o))
                for k, n in enumerate((judged, held)):
                    if n.sum() >= PLACE_MIN_PIXELS:
                        worst[k] = max(worst[k], float(through.sum() / n.sum()))
            self._noise[o.id] = (worst[0], worst[1])
        return self._noise[o.id]

    def foot(self, o: MapObject) -> tuple[float, float]:
        """(the height of the object's foot, its height): the 2nd percentile of its samples'
        heights, and the span to their 98th percentile (``_evidence``)."""
        def compute() -> tuple[float, float]:
            pts, _ = self.samples(o)
            if not len(pts):
                return 0.0, 0.0
            lo, hi = np.percentile(pts[:, 2], [2, 98])
            return float(lo), float(hi - lo)
        out: tuple[float, float] = o.memo("place_foot", compute)
        return out

    def ring(self, o: MapObject, f: int) -> NDArray[np.float64] | None:
        """Map points of the surface around the object's detection in keyframe ``f``."""
        key = (o.id, f)
        if key not in self._rings:
            self._rings[key] = None
            view = self.views.before(f)
            masks = [] if view is None else [
                m if isinstance(m, np.ndarray) else rle.decode(m)
                for oid, m in self.masks.instances(f) if self.masks.owner(oid) == o.id]
            masks = [m for m in masks if view is not None and m.shape == view.depth.shape]
            if view is not None and masks:
                mask = np.logical_or.reduce(masks)
                width = max(MASK_DILATE + 1, int(np.ceil(RING_WIDTH * np.sqrt(mask.sum()))))
                box = _bbox(mask)
                assert box is not None
                h, w = mask.shape
                r0, r1 = max(0, box[0] - width), min(h, box[1] + width)
                c0, c1 = max(0, box[2] - width), min(w, box[3] + width)
                dist = ndimage.distance_transform_edt(~mask[r0:r1, c0:c1])
                band = (dist > MASK_DILATE) & (dist <= width)
                ok = view.valid[r0:r1, c0:c1] & (view.depth[r0:r1, c0:c1] > 0)
                ok &= ~self._edge(f, view)[r0:r1, c0:c1]
                v, u = np.nonzero(band & ok)
                if len(v) >= RING_MIN:
                    if len(v) > RING_SAMPLES:
                        k = np.linspace(0, len(v) - 1, RING_SAMPLES).astype(int)
                        v, u = v[k], u[k]
                    v, u = v + r0, u + c0
                    z = view.depth[v, u].astype(np.float64)
                    self._rings[key] = view.T_map_cam.apply(unproject_pixels(u, v, z,
                                                                             view.K.K()))
        return self._rings[key]

    def occupied(self, o: MapObject, j: int, uv: NDArray[np.float64]) -> bool:
        """Whether keyframe ``j`` detected something at the object's place (``uv``: its samples
        in the image): another object, not much larger (not its support: at most twice its size),
        whose mask (grown by ``MASK_DILATE``) holds ``VISIBLE_SHARE`` of the samples — or
        ``PLACE_MIN_PIXELS`` of them when a keyframe detected it at the object's own place
        (``_overlap``: the object itself under another label, which a keyframe across the room
        places a little apart: a bag seen as a handbag). An item a keyframe detected beside the
        object does not count. The keyframe then did not see the place empty."""
        if not len(uv):
            return False
        mine = set(o.frames)
        for oid, m in self.masks.instances(j):
            p = self.objects.get(self.masks.owner(oid))
            if p is None or p is o or _scale(p) > 2 * _scale(o):
                continue
            both = sorted(mine & set(p.frames))
            if both and not self._overlap(o, p, both[0]):
                continue  # detected beside it
            mask = m if isinstance(m, np.ndarray) else rle.decode(m)
            h, w = mask.shape
            u = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
            v = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
            r = MASK_DILATE
            r0, r1 = max(0, int(v.min()) - r), min(h, int(v.max()) + r + 1)
            c0, c1 = max(0, int(u.min()) - r), min(w, int(u.max()) + r + 1)
            crop = mask[max(0, r0 - r):r1 + r, max(0, c0 - r):c1 + r]
            if not crop.any():
                continue
            near = ndimage.binary_dilation(crop, iterations=r)
            hit = near[v - max(0, r0 - r), u - max(0, c0 - r)]
            # the object itself under another label (a keyframe detected both at one place):
            # anywhere on its place; anything else: over most of it
            if (both and int(hit.sum()) >= PLACE_MIN_PIXELS) or float(hit.mean()) >= VISIBLE_SHARE:
                return True
        return False

    def _mask_of(self, o: MapObject, f: int) -> NDArray[np.bool_] | None:
        found = [m if isinstance(m, np.ndarray) else rle.decode(m)
                 for oid, m in self.masks.instances(f) if self.masks.owner(oid) == o.id]
        return np.logical_or.reduce(found) if found else None

    def _footprint(self, o: MapObject, f: int) -> _Footprint | None:
        """The shape, bounding box and centre of ``o``'s detections in keyframe ``f`` (None: none,
        or no pixel); the masks do not change while the places are judged."""
        key = (o.id, f)
        if key not in self._footprints:
            m = self._mask_of(o, f)
            box = None if m is None else _bbox(m)
            fp = None
            if m is not None and box is not None:
                v, u = np.nonzero(m)
                fp = _Footprint(m.shape, box, float(v.mean()), float(u.mean()))
            self._footprints[key] = fp
        return self._footprints[key]

    def _overlap(self, o: MapObject, p: MapObject, f: int) -> bool:
        """Whether keyframe ``f`` detected ``o`` and ``p`` at one place: the centre of either
        mask inside the other's bounding box (grown by ``MASK_DILATE``). A keyframe's masks are
        exclusive, so one thing detected twice (a bag as a bag and as a handbag) is two masks
        side by side, each over the other's middle; an item beside another is not."""
        a, b = self._footprint(o, f), self._footprint(p, f)
        if a is None or b is None or a.shape != b.shape:
            return False
        for x, y in ((a, b), (b, a)):
            box = y.box
            if box[0] - MASK_DILATE <= x.row <= box[1] + MASK_DILATE \
                    and box[2] - MASK_DILATE <= x.col <= box[3] + MASK_DILATE:
                return True
        return False

    def local_ratio(self, o: MapObject, j: int, view: View) -> tuple[float, float] | None:
        """Keyframe ``j``'s depth ratio to the object's detections around the object, when it
        agrees with them there (see the class docstring), and the share of those surroundings it
        sees through by the object's margin once divided by that ratio: what it sees through
        where nothing changed (a pose a few centimetres off beside a depth edge); None when it
        does not agree."""
        T = view.T_map_cam
        scale = max(0.5, float(o.obs_depth))
        refs = sorted((f for f in o.frames if self.views.pose(f) is not None),
                      key=lambda f: (_viewpoint_distance(T, self.views.pose(f), scale), f))
        for f in refs[:PLACE_REFS]:
            ring = self.ring(o, f)
            if ring is None:
                continue
            ok, z, d, _, _ = _lookup_valid(view, ring)
            if ok.sum() < RING_MIN:
                continue
            r = d[ok] / z[ok]
            same = (r > 1.0 / RING_SAME) & (r < RING_SAME)
            if same.sum() < RING_MIN:
                continue
            rho = float(np.median(r[same]))
            if not (1.0 - RING_BIAS <= rho <= 1.0 + RING_BIAS):
                continue
            tol = np.maximum(ABSENCE_TAU_SMALL_MIN / z[ok], ABSENCE_TAU_SMALL_REL)
            if float((np.abs(r / rho - 1.0) <= tol).mean()) >= VISIBLE_SHARE:
                _, size = self.samples(o)
                beyond = r / rho - 1.0 > absence_tau(z[ok], size) / z[ok]
                return rho, float(beyond.mean())
        return None

    def fused(self, o: MapObject) -> bool:
        """Whether a keyframe that detected the object fused it: one of its sightings lies within
        that keyframe's fused depth (``cuts``; always, without them). Beyond it the map holds no
        surface, and monocular depth places the object too loosely (a car 40-100 m down a
        street, placed metres off) for its place to be judged."""
        if self.cuts is None:
            return True
        if o.id not in self._fused:
            ok = False
            for sg in o.sightings:
                T = self.views.pose(sg.frame)
                cut = self.cuts.get(sg.frame)
                if T is not None and cut is not None:
                    z = float((np.asarray(sg.centroid, np.float64) - T.t) @ T.R[:, 2])
                    ok = ok or 0.0 < z < cut
            self._fused[o.id] = ok
        return self._fused[o.id]

    def verdict(self, o: MapObject, j: int, detectable: bool = False) -> Verdict | None:
        """Keyframe ``j``'s verdict on the object's place (see the class docstring); with
        ``detectable``, only from where it could have detected the object too (``PLACE_RANGE``:
        its silence there is evidence). Only a keyframe that fuses the place (most of the
        object's samples within its fused depth, ``cuts``) judges an object a keyframe that
        detected it fused (``fused``): what lies beyond the fused depth, the map does not draw."""
        rec = self.views.records.get(j)
        pts, size = self.samples(o)
        if rec is None or rec.low_confidence or not len(pts) or j in o.frames \
                or not _judges(rec, o, size) or not in_view(pts, rec.K_grid, rec.T_map_cam) \
                or not self.fused(o):
            return None
        view = self.views.before(j)  # this update's latest wins may have retired the place
        if view is None:
            return None
        uv, z = project(view.T_map_cam.inverse().apply(pts), view.K.K())
        h, w = view.depth.shape
        with np.errstate(invalid="ignore"):
            framed = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if framed.mean() < PLACE_FRAMED:
            return None
        if self.cuts is not None and float(np.median(z[framed])) >= self.cuts.get(j, np.inf):
            return None
        if detectable and float(np.median(z[framed])) > PLACE_RANGE * self.reach(o):
            return None
        if self.occupied(o, j, uv[framed]):
            return None
        local = self.local_ratio(o, j, view)
        if local is None:
            return None
        ratio, around = local
        ok, zz, d, u, v = _lookup_valid(view, pts[framed])
        dn = d / ratio
        t = absence_tau(zz, size)
        visible = ok & (dn - zz >= -t)  # seen through or on: not hidden behind something nearer
        pixels = len(np.unique(v[visible] * w + u[visible]))
        if visible.sum() < VISIBLE_SHARE * int(framed.sum()) or pixels < PLACE_MIN_PIXELS:
            return None
        through, judged, held = _evidence(view, ok, zz, dn, u, v, size, self.foot(o))
        supported = (rec.stats.get("observations", FEW_MIN_INLIERS) >= FEW_MIN_INLIERS
                     or "pose_matches" in rec.stats
                     or rec.pose_source in ("identity", "multiview"))
        # what its own detections see through, and what this keyframe sees through around the
        # object where nothing changed, are no evidence
        shares = []
        for n, s0 in zip((judged, held), self.transparency(o), strict=True):
            seen = float(through.sum() / n.sum()) if n.any() else 0.0
            s0 = max(s0, around)
            shares.append(max(0.0, (seen - s0) / (1.0 - s0)) if s0 < 1.0 else 0.0)
        return Verdict(j, shares[0], supported, ratio, held=shares[1])

    def verdicts(self, o: MapObject, frames: list[int], detectable: bool = False
                 ) -> list[Verdict]:
        out = []
        for f in frames:
            v = self.verdict(o, f, detectable)
            if v is not None:
                out.append(v)
        return out

    def through_pixels(self, o: MapObject, v: Verdict) -> NDArray[np.bool_]:
        """The pixels with which keyframe ``v.frame`` saw through the object (its points): under
        its footprint (the points projected, grown by ``MASK_DILATE``), valid and farther than
        its farthest point there by the margin of ``verdict``, its depth divided by
        ``v.ratio``."""
        view = self.views.get(v.frame)
        assert view is not None
        pts = np.asarray(o.points, np.float64)
        _, size = self.samples(o)
        h, w = view.depth.shape
        uv, z = project(view.T_map_cam.inverse().apply(pts), view.K.K())
        with np.errstate(invalid="ignore"):
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        out = np.zeros((h, w), bool)
        if not ok.any():
            return out
        u = np.rint(uv[ok, 0]).astype(np.int64).clip(0, w - 1)
        v_ = np.rint(uv[ok, 1]).astype(np.int64).clip(0, h - 1)
        far = np.zeros((h, w))
        np.maximum.at(far, (v_, u), z[ok])
        far = ndimage.grey_dilation(far, size=(2 * MASK_DILATE + 1, 2 * MASK_DILATE + 1))
        t = absence_tau(far, size)
        return (far > 0) & view.valid & (view.depth / v.ratio > far + t)


def _lookup_valid(view: View, pts: NDArray[np.float64]
                  ) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64],
                             NDArray[np.int64], NDArray[np.int64]]:
    """(on a valid pixel, depth in the camera, observed depth, u, v) of each point; unlike
    ``View.lookup``, the image-border band and depth edges count."""
    uv, z = project(view.T_map_cam.inverse().apply(pts), view.K.K())
    h, w = view.depth.shape
    with np.errstate(invalid="ignore"):
        inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    u = np.zeros(len(pts), np.int64)
    v = np.zeros(len(pts), np.int64)
    u[inside] = np.rint(uv[inside, 0]).astype(np.int64).clip(0, w - 1)
    v[inside] = np.rint(uv[inside, 1]).astype(np.int64).clip(0, h - 1)
    d = np.where(inside, view.depth[v, u], 0.0).astype(np.float64)
    ok = inside & view.valid[v, u] & (d > 0)
    return ok, np.where(inside, z, 0.0), d, u, v


def _evidence(view: View, ok: NDArray[np.bool_], z: NDArray[np.float64], d: NDArray[np.float64],
              u: NDArray[np.int64], v: NDArray[np.int64], size: float,
              foot: tuple[float, float]
              ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """(seen through, judged, held) for an object's samples in a keyframe (``_lookup_valid``, with
    ``d`` the keyframe's depth there as the verdict uses it): a sample is seen through when the
    keyframe sees farther than it by the absence margin (``absence_tau``), and judged when it is
    seen through or on the keyframe's surface within that margin — except where that surface is
    the object's support at its foot: the surface the keyframe sees there, lifted into the map,
    lies in the object's bottom band (``FOOT_BAND`` of its ``height``, at most the margin, above
    its ``base``). A keyframe sees the table, the floor or the windowsill under the foot of a
    cup whether the cup stands there or not, so those samples are no evidence either way: the
    cup of ``office_sequence``, gone in the last photos, left 15 % of its samples on the
    windowsill within the margin, and counted as seen in place they kept both photos that saw
    the empty sill below a removal. ``held``: seen through or on, the foot included (a keyframe
    sees the object in place by it: the foot cannot tell, and what cannot tell never removes an
    object)."""
    base, height = foot
    t = absence_tau(z, size)
    through = ok & (d - z > t)
    on = ok & (np.abs(d - z) <= t)
    held = through | on
    if on.any():
        idx = np.flatnonzero(on)
        seen = view.T_map_cam.apply(unproject_pixels(u[idx], v[idx], d[idx], view.K.K()))
        at_foot = seen[:, 2] <= base + np.minimum(t[idx], FOOT_BAND * height)
        on[idx[at_foot]] = False
    return through, through | on, held


def _judgement(o: MapObject, verdicts: list[Verdict], few: bool = True) -> str | None:
    """What the verdicts of keyframes say about an object's place, the latest winning: a keyframe
    that sees the object in place (``Verdict.in_place``: <= 1 - ``REMOVE_FRACTION`` of its samples
    seen through, the support at its foot counted as seen in place) outweighs
    every earlier one, so only the verdicts of the keyframes added after the last such keyframe
    count ("in place" when there are none). Of those: "gone" when most see through it
    (>= ``REMOVE_FRACTION``) and at least ``REMOVE_MIN_FRAMES`` do, or (with ``few``) at least
    ``CONFIRM_DETECTIONS`` do — absence, like presence, is confirmed by two keyframes — every one
    of them >= ``REMOVE_FRACTION_FEW`` from a well-supported pose, and the object is established
    (>= ``FEW_MIN_OBSERVATIONS`` detections); "strike" when most see through it with
    less evidence; None otherwise."""
    if not verdicts:
        return None
    ordered = sorted(verdicts, key=lambda v: v.frame)
    last_in_place = max((k for k, v in enumerate(ordered) if v.in_place),
                        default=-1)
    after = ordered[last_in_place + 1:]
    if not after:
        return "in place"
    strong = [v for v in after if v.share >= REMOVE_FRACTION]
    if 2 * len(strong) <= len(after):
        return None
    established = (few and len(strong) >= CONFIRM_DETECTIONS
                   and all(v.share >= REMOVE_FRACTION_FEW and v.supported for v in strong)
                   and o.observations >= FEW_MIN_OBSERVATIONS)
    return "gone" if len(strong) >= REMOVE_MIN_FRAMES or established else "strike"


def _absence(candidates: list[MapObject], verdicts: dict[int, list[Verdict]],
             witnesses: dict[int, list[int]] | None = None) -> list[int]:
    """Objects that this update, as a whole, shows to be gone (``_judgement`` of the verdicts of
    its keyframes added after each object's last detection, ``_Places``): "gone" removes the
    object; a "strike" is remembered, and a second strike (from a later update) removes it; "in
    place" clears the strikes (latest wins). An update counts once, however many of its keyframes
    judge. ``witnesses``, when given, receives the keyframes that saw through each removed object
    (object id -> indices)."""
    removed = []
    for o in candidates:
        vs = verdicts.get(o.id, [])
        verdict = _judgement(o, vs)
        if verdict == "in place":
            o.strikes = 0
        elif verdict == "strike":
            o.strikes += 1
        if verdict == "gone" or (verdict == "strike" and o.strikes >= 2):
            removed.append(o.id)
            if witnesses is not None:
                last = max((v.frame for v in vs if v.in_place), default=-1)
                witnesses[o.id] = sorted(v.frame for v in vs
                                         if v.share >= REMOVE_FRACTION and v.frame > last)
    return removed


# ------------------------------------------------------------------------------------------------
# moved objects


@dataclass
class Move:
    """An object that the latest keyframes see elsewhere (``_moves``): ``src`` where it was, ``dst``
    where it is now; ``departed``: keyframes added after ``src``'s last detection that saw its
    place without it; ``arrived``: the verdicts of keyframes added before ``dst``'s first detection
    that saw through its place (free space then, the object now)."""

    src: MapObject
    dst: MapObject
    departed: list[int]
    arrived: list[Verdict]


class _Colours:
    """Median CIELab colour (L 0-100, a, b) of the objects' detections: the pixels of their masks
    (``_Masks``) in up to ``MOVE_COLOUR_VIEWS`` of their detecting keyframes (the largest
    detections), per keyframe, then the median over the keyframes. This update's keyframe images
    are in memory, stored ones are loaded when needed."""

    def __init__(self, ctx: Any, views: _Views, masks: _Masks) -> None:
        self.ctx = ctx
        self.views = views
        self.masks = masks
        self.rgb: dict[int, NDArray[np.uint8]] = {}
        for nf in getattr(ctx, "new", []) or []:
            if nf.record is not None and getattr(nf.frame, "rgb", None) is not None:
                self.rgb[nf.record.index] = nf.frame.rgb
        self.cache: dict[int, NDArray[np.float64] | None] = {}

    def _image(self, f: int, shape: tuple[int, int]) -> NDArray[np.uint8] | None:
        img = self.rgb.get(f)
        if img is None:
            from oh_my_slam.core.images import load_rgb

            rec = self.views.records.get(f)
            if rec is None or self.ctx is None:
                return None
            p = self.ctx.tx.current(rec.image)
            if not p.exists():
                return None
            img = load_rgb(p, max_side=max(shape))
        return img if img.shape[:2] == shape else None

    def of(self, o: MapObject) -> NDArray[np.float64] | None:
        if o.id not in self.cache:
            import cv2

            size = {s.frame: s.points for s in o.sightings}
            per_frame = []
            for f in sorted(o.frames, key=lambda f: (-size.get(f, 0), f))[:MOVE_COLOUR_VIEWS]:
                rec = self.views.records.get(f)
                if rec is None:
                    continue
                shape = (int(rec.K_grid.height), int(rec.K_grid.width))
                found = [m if isinstance(m, np.ndarray) else rle.decode(m)
                         for oid, m in self.masks.instances(f) if self.masks.owner(oid) == o.id]
                found = [m for m in found if m.shape == shape]
                img = self._image(f, shape) if found else None
                if img is None:
                    continue
                px = img[np.logical_or.reduce(found)]
                if len(px):
                    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
                    lab = lab.astype(np.float64)
                    lab[:, 0] *= 100.0 / 255.0
                    lab[:, 1:] -= 128.0
                    per_frame.append(np.median(lab, axis=0))
            self.cache[o.id] = np.median(per_frame, axis=0) if per_frame else None
        return self.cache[o.id]


def _moves(state: ObjectState, touched: set[int], places: _Places, colours: _Colours
           ) -> list[Move]:
    """Objects the latest keyframes see elsewhere: the object keeps its id, and its old place, like
    a removed object's, shows the latest views (spec §2.3: a later update wins, object identity is
    persistent).

    ``dst`` (touched by this update) and ``src`` are confirmed objects of compatible labels,
    comparable size and colour (``MOVE_*``), standing apart, every detection of ``src`` made by a
    keyframe added before every detection of ``dst`` (in the order of addition). The move needs
    the place of each seen without it in its turn (``_Places.verdict``: keyframes that could have
    detected it there, and detected nothing else of its size there: ``_Places.occupied``):
    ``src``'s by at least one keyframe added after its last detection, and ``dst``'s by at least
    one added before its first detection; and at one of the two places the keyframes' depth shows the
    change as it shows a removal (``_judgement`` "gone") — monocular depth cannot tell a thin
    object from the wall right behind it, and the detector's silence alone is no proof of absence.
    Each object moves once: the pairs whose colours agree best first."""
    objs = sorted((o for o in state.objects if o.confirmed and len(o.points) and o.frames
                   and o.obb is not None), key=lambda o: o.id)
    order = sorted(places.views.records)
    cands: list[tuple[float, int, int, MapObject, MapObject]] = []
    for b in objs:
        if b.id not in touched:
            continue
        first_b = min(b.frames)
        for a in objs:
            if a is b or max(a.frames) >= first_b or not compatible(a.label, b.label):
                continue
            sa, sb = _scale(a), _scale(b)
            if min(sa, sb) <= 0 or max(sa, sb) > MOVE_SCALE * min(sa, sb):
                continue
            assert a.obb is not None and b.obb is not None
            gap = (max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * a.obs_depth)
                   + max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * b.obs_depth))
            pa = OBB(a.obb.center, a.obb.R, a.obb.size + gap)
            if obb_iou_upright(pa, OBB(b.obb.center, b.obb.R, b.obb.size + gap)) > 0 or \
                    bool(pa.contains(b.centroid[None], 0.0)[0]):
                continue
            ca, cb = colours.of(a), colours.of(b)
            if ca is None or cb is None:
                continue
            chroma = float(np.linalg.norm(ca[1:] - cb[1:]))
            if chroma > MOVE_CHROMA or abs(float(ca[0] - cb[0])) > MOVE_LIGHTNESS:
                continue
            cands.append((chroma, a.id, b.id, a, b))
    moves: list[Move] = []
    used: set[int] = set()
    for _, _, _, a, b in sorted(cands, key=lambda c: c[:3]):
        if a.id in used or b.id in used:
            continue
        dep = places.verdicts(a, [f for f in order if f > max(a.frames)], detectable=True)
        arr = places.verdicts(b, [f for f in order if f < min(b.frames)], detectable=True)
        if not dep or not arr:
            continue
        if _judgement(a, dep) != "gone" and _judgement(b, arr, few=False) != "gone":
            continue
        last = max((v.frame for v in arr if v.in_place), default=-1)
        moves.append(Move(a, b, [v.frame for v in dep],
                          [v for v in arr if v.share >= REMOVE_FRACTION and v.frame > last]))
        used |= {a.id, b.id}
    return moves


def label_map_for(reader_instances: list[dict[str, Any]], shape: tuple[int, int],
                  state: ObjectState) -> NDArray[np.int32]:
    """Per-pixel persistent object ids of a keyframe (merged ids resolved, removed → 0)."""
    lab = np.zeros(shape, np.int32)
    for inst in reader_instances:
        oid = state.resolve(int(inst["object_id"]))
        if oid is None:
            continue
        m = rle.decode(inst["mask"])
        if m.shape == shape:
            lab[m] = oid
    return lab

