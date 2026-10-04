# 2.5 Visualisation — `view.sh`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

```sh
view.sh -i <image> [--no-browser]
view.sh -m <map-folder> [--no-browser]
```

Starts a local web server for interactive browser visualisation. Exactly one input is
required: `-i` and `-m` are mutually exclusive. Once the server accepts connections it opens
the default browser on its page; `--no-browser` only prints the URL on stderr.

* `-i <image>` — reconstruct and segment one RGB image, then show its colour point cloud,
  segmented image, object catalogue, labelled OBBs, and the camera at its estimated pose.
* `-m <map-folder>` — load an existing map without modifying it, then show its point cloud,
  camera poses, and labelled OBBs.

The viewer draws every point of a cloud of at most 16 000 000 points. Above that budget it
shows a voxel-grid subsample: one original point per occupied voxel, with the smallest voxel
edge that yields at most 16 000 000 points. Each shown point keeps its own position, colour,
normal and object id (no averaging), and the segmentation layer uses the same subset.
Whenever points are omitted, the point-cloud controls state "showing X of Y points" with the
voxel edge. Thinning concerns the display only: PLY outputs and the persisted map stay
complete.

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
