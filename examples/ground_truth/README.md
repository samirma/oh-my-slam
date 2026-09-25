# Ground truth for the evaluator

`python -m oh_my_slam.tools.evaluate` reads every `*.json` file in this folder and its
subfolders. The `kind` field of each file selects the metrics it adds. Files and fields can be
added without changing any code. The targets of the extra metrics are the `gt.*` entries of
`examples/targets.json`.

These `gt.*` metrics are the evaluator's accuracy measure. Without annotations, segmentation is
measured only against the map built from the same detector (`seg.map_consistency.*`, which
measures consistency, not accuracy). Poses are measured only against the commanded headings in
the capture names, and the true headings deviate from those by several degrees. The summary
says when no annotations were found. New annotations take effect on the next run:
`--resummarise` only judges a stored run's values again.

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
