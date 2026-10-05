# Ground truth for the evaluator

`python -m oh_my_slam.tools.evaluate` reads every `*.json` file in this folder and its
subfolders. The `kind` field of each file selects the metrics it adds. Files and fields can be
added without changing any code. The targets of the extra metrics are the `gt.*` entries of
`examples/targets.json`.

These `gt.*` metrics are the evaluator's accuracy measure. Without annotations, segmentation is
measured only against the map built from the same detector (`seg.map_consistency.*`, which
measures consistency, not accuracy). Poses are measured only against the commanded headings in
the capture names, and the true headings deviate from those by several degrees. The summary
says when no annotations were found. New annotations take effect on the next run.

A file that is malformed, or that describes an image the evaluator did not run, is skipped. The
evaluator lists skipped files and the reason under `details.ground_truth.skipped` in `result.json`.

## `kind: "objects"`: the objects in one example image

```json
{
  "kind": "objects",
  "image": "restaurant.jpg",
  "objects": [
    {"label": "chair", "cuboid": [0.4, 0.9, 6.2, 0, 0, 0, 1, 0.5, 0.5, 0.9]},
    {"label": "person"}
  ]
}
```

* `image` is the image path relative to `examples/`, for example `restaurant.jpg` or
  `ainex-captures/001_bootstrap_level.jpg`.
* `label` uses the detector's vocabulary. Compatible labels, such as sofa and couch, count as the
  same label.
* `cuboid` is optional. It is an OpenLABEL 10-value cuboid `x, y, z, qx, qy, qz, qw, sx, sy, sz`
  (quaternion scalar last). It is given in metres in the image's camera frame (OpenCV axes: x right,
  y down, z forward), which is the frame `segment.sh -i` reports its boxes in.

The evaluator compares each file with that image's `segment.sh -i` output:

* When every annotated object has a cuboid, objects are paired by box overlap. A pair counts only if
  its labels are compatible.
* Otherwise, labels are paired one to one: identical labels first, then compatible ones.

Metrics:

* `gt.objects.recall` is the share of annotated objects that were found.
* `gt.objects.precision` is the share of detections that match an annotated object.
* `gt.objects.obb_iou_median` is the median 3D IoU of the paired boxes. It is reported only when
  cuboids are annotated.

## `kind: "poses"`: the true head orientation per capture

```json
{
  "kind": "poses",
  "frames": {
    "001_bootstrap_level.jpg": {"yaw_deg": -11.8, "pitch_deg": 0.0},
    "005_bootstrap_left015_up.jpg": {"yaw_deg": 3.4, "pitch_deg": 14.5}
  }
}
```

* The keys are capture file names in `examples/ainex-captures/`.
* `yaw_deg` is the heading in degrees, positive to the left, from any fixed zero.
* `pitch_deg` is the elevation in degrees above the horizon, positive upwards.
* Either value can be left out.

The evaluator compares these values with the camera poses of the map built in one update:

* `gt.poses.yaw_err_median_deg` and `gt.poses.yaw_err_max_deg` are computed after the single yaw
  offset that best aligns the two sets of headings.
* `gt.poses.pitch_err_median_deg` compares pitch directly, since the map frame is gravity-aligned.

## `kind: "map_update"`: what changed during `examples/office_sequence/`

```json
{
  "kind": "map_update",
  "sequence": "office_sequence",
  "absent": [
    {"label": "cup",
     "seen_in": {"20260929_122226.jpg": [0.29, 0.42, 0.42, 0.59],
                 "20260929_122229.jpg": [0.81, 0.43, 0.95, 0.66]}}
  ],
  "stable": ["monitor", "keyboard"],
  "splits": [[4, 4, 5], [6, 7]]
}
```

* `sequence` is the example folder, relative to `examples/`. Only `office_sequence` is evaluated.
* `absent` lists objects that were in the scene early in the sequence and are gone later. `label`
  uses the detector's vocabulary (compatible labels, such as cup and mug, count as the same).
  `seen_in` maps the images that show the object to its region in that image: `[x0, y0, x1, y1]`
  as shares of the image width and height (y down). The map's own camera poses project the map
  objects into these images, so the test does not depend on where the map puts its origin.
* `stable` is optional: labels that restrict the comparison of the objects that never changed. By
  default every object the early images observe, except the absent ones, is compared.
* The images are in name order. The **early part** of the sequence is the images up to the last one
  named in any `seen_in`.
* `splits` is optional: the ways the sequence is mapped across several updates, each the sizes of
  consecutive updates (at least two positive integers adding up to the number of images). By
  default one split: the early part, then the rest.

The evaluator builds, outside the repository, a map of the whole sequence in one `mapper.sh
update`, and one map per split (`split_<sizes>`, e.g. `split_4_4_5`), an update per part, keeping
each update's `-t full` scene. Files about the same sequence are merged. Metrics:

* `map_update.absent_fraction` and `map_update.<split>.absent_fraction` are the share of the
  `absent` objects that the whole-sequence map, and each split map after its last update, no longer
  have. A remnant is a map object with a compatible label (its own or one of its `detected_as`)
  whose box, projected into the `seen_in` images, covers at least a quarter of the region in one of
  them. If none of those images is registered in the map, the label alone decides.
* `map_update.hole_fraction` and `map_update.<split>.hole_fraction`: no hole where an absent object
  stood. The map cloud is projected, with the map's own poses, into each registered `seen_in`
  image; the region (8 x 8 cells) and a ring around it are compared: a region cell with no map
  point, or whose nearest point lies more than 25 % behind the ring's median nearest depth, is a
  hole.
* `map_update.before_present_fraction` is the same test as `absent_fraction` on the split map whose
  first update is exactly the early part, after that update. It is the control: an object that was
  never detected early cannot be seen to disappear.
* `map_update.<split>.stability.*` are the metrics of `map.stability.*` for the objects that never
  changed: the objects the first update's images observe, after the last update against after the
  first (ids, labels, OBBs; one map frame, no alignment).
* `map_update.<split>.ids_persistent_fraction`: every id an update published for an unchanged object
  is, in every later update, still an object with a compatible label whose box overlaps or nearly
  coincides with it.
* `map_update.<split>.vs_one_update.*`: the split map against the one-update map, as `map.stability.*`
  (an id may differ where an earlier update of the split had published one).

Their targets are the `map_update.*` entries of `examples/targets.json` (`map_update.split_*.…`
patterns for the splits).
