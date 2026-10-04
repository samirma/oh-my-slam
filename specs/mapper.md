# 2.3 Mapping — `mapper.sh`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

```sh
mapper.sh update -i <image(s)|video> -m <map-folder>   # whole map as JSON (default) to stdout
mapper.sh update -i <image(s)|video> -m <map-folder> [-f json|ply] [-o <file>] [-p <attrs>] [-t full|single] [-fps <n>]
mapper.sh locate -i <image(s)> -m <map-folder>   # located camera pose(s) only, as JSON, map untouched
mapper.sh locate -i <image(s)> -m <map-folder> [-f json|ply] [-o <file>] [-p <attrs>] [-t full|single]
```

`update` creates the map if `<map-folder>` does not yet exist or is empty, and otherwise
extends the existing map with the new input. A non-empty folder that is not a map is refused
and left untouched. Only `-i` and `-m` are required; every other option has a default.

* `-i <image(s)|video>` — one or more images, or a video file.
* `-m <map-folder>` — map directory; holds the persisted map and its metadata.
* `-f json|ply` — output format, **default `json`**: the scene description of [§3](high_level_spec.md#3-scene-description-json-returned-by-the-tools), or the
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
image was taken from, in the same pose representation as `update`'s result. Only `-i` and
`-m` are required.

* `-i <image(s)>` — one or more images; a video is refused.
* `-m <map-folder>` — an existing map; an empty or missing folder is an input error and is
  not created.
* `-f json|ply` — output format, **default `json`**, as for `update`.
* `-t full|single` — scope of the result, **default `single`**:
  * `single` — the located camera pose of each input image only, with no objects and no
    point cloud (PLY: the map points visible from the located cameras).
  * `full` — the **entire** map exactly as `update -t full` returns it (every object and the
    pose of each contributing frame; PLY: the whole map cloud), plus the located camera
    pose of each input image, distinguishable from the map's own frames.
* `-o <file>` — as for `update`.
* `-p <attrs>` — point-cloud attributes for `-f ply` ([§2.2](reconstruct.md#point-cloud-attributes)).

In a PLY result the located camera poses are written in the file header, one line per
input image, since a PLY has no camera element; the vertices are map points only.

The map is **read-only** for `locate`: the images are not added, nothing in the map folder
(including `.staging/`) changes, and a concurrent reader sees no difference. An image the
map cannot localise (not enough overlap with it) is reported with an actionable message
naming the image; if no image can be localised, the command fails with an input error.

The map's geometry is a point cloud; no surface mesh is produced.

Since an image captures a specific point in time for a map section, any new image that
contradicts the current data should update the map with the latest information to keep it
current. "Latest" is the order of addition: a later update wins over an earlier one, and
within one update a later frame wins over an earlier one, in input order (the order of the
`-i` images; a video's frame order). Capture timestamps are not used. The result is the same
whether a sequence is mapped in one update or split across several in the same order.

Example: in `examples/office_sequence/` a cup is visible in the first images and gone in the
last ones. Mapping the whole sequence, in one update or in several, must produce a map
without the cup, with no hole where it stood (the latest observation of that surface fills
it), while every object that never changed keeps its `id`, label and OBB.

Object identity is persistent: an object observed across several frames keeps one `id` and
one colour for the lifetime of the map, and its OBB is refined as evidence accumulates.
