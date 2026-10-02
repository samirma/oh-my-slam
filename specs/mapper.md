# 2.3 Mapping — `mapper.sh`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

```sh
mapper.sh update -i <image(s)|video> -m <map-folder>   # whole map as JSON (default) to stdout
mapper.sh update -i <image(s)|video> -m <map-folder> [-f json|ply] [-o <file>] [-p <attrs>] [-t full|single] [-fps <n>]
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

The map's geometry is a point cloud; no surface mesh is produced.

Since an image captures a specific point in time for a map section, any new image that
contradicts the current data should update the map with the latest information to keep it
current. "Latest" is the order of addition: a later update wins over an earlier one.
Frames within one update count as one observation of the scene. Capture timestamps are not
used.

As example is this examples/office_sequence that the first images has a cup and the last one don't, as result the expected map is to no longer have that cup there

Object identity is persistent: an object observed across several frames keeps one `id` and
one colour for the lifetime of the map, and its OBB is refined as evidence accumulates.
