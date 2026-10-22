# 2.3 Mapping — `mapper.sh`

Part of the [high-level specification](../high_level_spec.md) (§2 Components).

```sh
mapper.sh update -i <image(s)|video> -m <map-folder>   # whole map as JSON (default) to stdout
mapper.sh update -i <image(s)|video> -m <map-folder> [-f json|ply] [-o <file>] [-p <attrs>] [-t full|single] [-fps <n>]
mapper.sh locate -i <image(s)> -m <map-folder>   # located camera pose(s) only, as JSON, map untouched
mapper.sh locate -m <map-folder>                 # the map's last known camera pose, as JSON
mapper.sh locate -m <map-folder> -f png          # bird's-eye image of the map, with the last known camera
mapper.sh locate [-i <image(s)>] -m <map-folder> [-f json|ply|png] [-o <file>] [-p <attrs>] [-t full|single] [-l <layers>]
```

`update` creates the map if `<map-folder>` does not yet exist or is empty, and otherwise
extends the existing map with the new input. A non-empty folder that is not a map is refused
and left untouched. Only `-i` and `-m` are required; every other option has a default.

* `-i <image(s)|video>` — one or more images, or a video file.
* `-m <map-folder>` — map directory; holds the persisted map and its metadata.
* `-f json|ply` — output format, **default `json`**: the scene description of [§3](../high_level_spec.md#3-scene-description-json-returned-by-the-tools), or the
  point cloud, in map coordinates. `-t` selects what either format covers.
* `-t full|single` — scope of the result, **default `full`**:
  * `full` — the **entire** map: every object, and the estimated camera pose of each
    contributing frame (PLY: the whole map cloud).
  * `single` — only what the **newly added** input covers: the poses of the new frames and
    the objects observed in them (PLY: the new frames' points).
* `-o <file>` — write the result to this file instead of stdout; stdout then stays empty.
* `-p <attrs>` — point-cloud attributes for `-f ply` ([§2.2](reconstruct.md#point-cloud-attributes)).
* `-fps <n>` — for video input, the number of frames per second to sample for analysis
  (default `2`); ignored for images.

`locate` registers each input image against the **entire** existing map and estimates its
camera pose: the position and orientation in map coordinates, i.e. the point of view the
image was taken from, in the same pose representation as `update`'s result. Without input
images it reports the map's **last known camera** instead. Only `-m` is required.

* `-i <image(s)>` — one or more images; a video is refused. Optional: without `-i`, the
  result's camera is the map's last known camera, i.e. the pose of the keyframe added last
  in the order of addition defined below (a later update, then a later frame in input
  order). It is marked as the last known camera, distinguishable from a located camera and,
  in a `-t full` result, from the map's other frames. `locate` never records a position, so
  locating images does not change the last known camera. Without `-i`, `locate` only reads
  the map and never needs the inference server.
* `-m <map-folder>` — an existing map; an empty or missing folder is an input error and is
  not created.
* `-f json|ply|png` — output format, **default `json`**: `json` and `ply` as for `update`;
  `png` — a bird's-eye image of the map ([below](#birds-eye-image)).
* `-t full|single` — scope of the result, **default `single`**. "The cameras" are the located
  camera of each input image, or the last known camera without `-i`:
  * `single` — the cameras' poses only, with no objects and no point cloud (PLY: the map
    points visible from the cameras; PNG: the whole map with these cameras only).
  * `full` — the **entire** map exactly as `update -t full` returns it (every object and the
    pose of each contributing frame; PLY: the whole map cloud), plus the cameras,
    distinguishable from the map's own frames (PNG: the map with the cameras and every
    keyframe's camera).
* `-l <layers>` — what a bird's-eye image draws under the cameras: `cloud` (the point cloud),
  `obb` (the objects' OBBs), or both, `cloud,obb` (**default**). It requires `-f png` and is
  refused with the other formats.
* `-o <file>` — as for `update`.
* `-p <attrs>` — point-cloud attributes for `-f ply` ([§2.2](reconstruct.md#point-cloud-attributes)).

In a PLY result the cameras are written in the file header, one line each (one per input
image, or the last known camera), since a PLY has no camera element; the vertices are map
points only. A bird's-eye image records them in its PNG text metadata, in the same
representation.

The map is **read-only** for `locate`: the images are not added, nothing in the map folder
(including `.staging/`) changes, and a concurrent reader sees no difference. An image the
map cannot localise (not enough overlap with it) is reported with an actionable message
naming the image, and only the located images are in the result; if no image can be
localised, the command fails with an input error.

The map's geometry is a point cloud; no surface mesh is produced.

Since an image captures a specific point in time for a map section, any new image that
contradicts the current data should update the map with the latest information to keep it
current. "Latest" is the order of addition: a later update wins over an earlier one, and
within one update a later frame wins over an earlier one, in input order (the order of the
`-i` images; a video's frame order). Capture timestamps are not used. Mapping a sequence
in one update or split across several in the same order gives the same objects, labels
and OBBs (within the evaluator's stability targets, §5). Two differences are allowed,
because identity persistence (below) takes precedence: an `id` may differ where an earlier
update had already published one, and the split map may keep an object that an earlier
update published and that no later image contradicts.

Example: in `examples/office_sequence/` a cup is visible in the first images and gone in the
last ones. Mapping the whole sequence, in one update or in several, must produce a map
without the cup, with no hole where it stood (the latest observation of that surface fills
it), while every object that never changed keeps its `id`, label and OBB.

Object identity is persistent: an object observed across several frames keeps one `id` and
one colour for the lifetime of the map, and its OBB is refined as evidence accumulates.

### Bird's-eye image

`-f png` draws the map seen from straight above: an orthographic projection along the map's
up axis onto its horizontal plane, with one scale for both image axes, as one 8-bit RGB PNG.
It always shows the whole map, whatever `-t` says; `-t` only chooses which cameras are drawn.
From bottom to top:

* **Point cloud** (layer `cloud`) — the map's points in their own colours (as `color=rgb`),
  each pixel showing the highest point that falls in it. A ceiling or roof, which would hide
  the floor plan, is left out; how it is recognised is recorded in `README.md`.
* **OBBs** (layer `obb`) — every object of the map, as the footprint of its OBB (the boxes are
  upright, so it is their horizontal rectangle), outlined in the object's colour and marked
  with its `id`. Outlines are drawn opaque with no anti-aliasing, so the colour contract's exact
  sRGB triple appears ([§2.4](segment.md#colour-contract)).
* **Cameras** — each camera of the result as a marker at its position, with its viewing
  direction projected onto the horizontal plane. A located camera, the last known camera and
  the map's own keyframes (`-t full`) differ in shape, not only in colour.

The image covers everything it draws. Its size, scale and axis orientation are recorded in
`README.md`. Its PNG text metadata records the mapping between pixel coordinates and the
map's horizontal coordinates, the layers drawn, and the cameras (above), so a pixel can be
turned into a map position without the JSON result.
