# 2.5 Visualisation — `view.sh`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

```sh
view.sh -i <image>
view.sh -m <map-folder>
```

Starts a local web server for interactive browser visualisation. Exactly one input is
required: `-i` and `-m` are mutually exclusive.

* `-i <image>` — reconstruct and segment one RGB image, then show its colour point cloud,
  segmented image, object catalogue, labelled OBBs, and the camera at its estimated pose.
* `-m <map-folder>` — load an existing map without modifying it, then show its complete
  point cloud, camera poses, and labelled OBBs.

The interface must provide independent controls for the available point-cloud,
camera-pose, segmentation, label, and OBB layers. The point-cloud layer also offers live
controls for the [§2.2](reconstruct.md#point-cloud-attributes) attributes that affect what is displayed: `color`, `stride`,
`min-depth` / `max-depth`, `edge`, `voxel` and `normals` for an image; only `color`,
`voxel` and `normals` for a map. Changing a control re-derives the cloud through the shared
point-cloud code from data already computed, and never re-runs inference. `encoding` and the
`label` property concern PLY files only and have no control.
The web interface must show the position coordinates of every displayed camera and provide an
option to move the viewer viewpoint to that camera position.

`view.sh` owns only the web server and browser UI. It consumes reconstruction, mapping, and
segmentation data through their existing implementations and must not duplicate depth
inference, point-cloud generation, map loading, segmentation, OBB fitting, object identity,
or colour assignment.
