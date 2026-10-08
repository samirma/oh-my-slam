# 2.4 Segmentation — `segment.sh`

Part of the [high-level specification](../high_level_spec.md) (§2 Components).

```sh
segment.sh -i <image> [-f json|png] [-o <file>] [-d <folder>] [--min-score <s>]
```

Segments an image into object instances, lifts each instance into 3D using the depth from the
inference server, and fits an OBB to it. `-i` is required.

* `-i <image>` — input RGB image.
* `-f json|png` — output format, **default `json`**.
  * `json` — the scene description of [§3](../high_level_spec.md#3-scene-description-json-returned-by-the-tools): objects, labels, colours and OBBs.
  * `png` — the segmented image: the input image at the reconstruction's working resolution
    (recorded in `README.md`) with each instance mask painted in that object's colour, over a
    dimmed copy of the original. Identical to `segmented.png` below.
* `-o <file>` — write the result to this file instead of stdout; stdout then stays empty.
* `-d <folder>` — also write the output artefacts listed below into this folder. Without
  `-o` and `-d`, the result goes to stdout and no files are written.
* `--min-score <s>` — drop detections below this confidence (default `0.5`). It
  only adds or removes objects: the objects present at two thresholds keep the same `id` and
  colour.

## Output artefacts

With `-d <folder>`, `segment.sh` writes:

| File | Contents |
| --- | --- |
| `segmentation.json` | The scene description of [§3](../high_level_spec.md#3-scene-description-json-returned-by-the-tools) — objects, labels, colours and OBBs. Identical to what `-f json` outputs. |
| `segmented.png` | The segmented image. Identical to what `-f png` outputs. |
| `catalog.csv` | One row per object: `id,label,score,color_hex,width_m,height_m,depth_m,volume_m3,center_x,center_y,center_z,pixel_count,point_count`. |
| `catalog.md` | The same catalogue as a human-readable table, ordered by descending volume. |

## Colour contract

Every artefact of a single run agrees on colour. One colour per object `id`, drawn
deterministically from a fixed, perceptually distinct palette, so that the value in
`segmentation.json` (`color` / `color_hex`), the pixels of that instance's mask in
`segmented.png`, the `color_hex` column of `catalog.csv` and the swatch of `catalog.md`, the
per-point colour in any PLY or `view.sh` cloud coloured with
`color=segment`, and the OBB colour rendered by `view.sh` are **the same sRGB triple**. Masks
are painted opaque, and each pixel and each point belongs to at most one object, so no two
colours ever mix.

Object `id`s are positive integers (`0` means unsegmented). The mapping is a pure function
of the object `id`, so re-running yields the same colours, and the
palette cycles by hue once it is exhausted. The palette never produces the mid-grey reserved
for unsegmented points. For a single image, `id`s are assigned in a deterministic order, so
re-running the same image with the same options gives each object the same `id` and colour.
