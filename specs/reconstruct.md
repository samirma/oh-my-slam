# 2.2 Single-frame reconstruction — `reconstruct.sh`

Part of the [high-level specification](../high_level_spec.md) (§2 Components).

```sh
reconstruct.sh -i <image>                # JSON scene description (default) to stdout
reconstruct.sh -i <image> -f depth       # depth image (PNG) to stdout
reconstruct.sh -i <image> -f depth -o depth.png   # depth image to a file
reconstruct.sh -i <image> -f ply         # point cloud to stdout
reconstruct.sh -i <image> -f ply -p color=height,voxel=0.01,normals=on
reconstruct.sh -i <image> -f ply -o cloud.ply   # point cloud to a file
```

* `-i <image>` — input RGB image.
* `-f json|depth|ply` — output format, **default `json`**. Unless `-o` is given, the result is
  written to **stdout** so it can be piped or redirected; diagnostics go to stderr.
  * `json` — the scene description of [§3](../high_level_spec.md#3-scene-description-json-returned-by-the-tools): detected objects, their labels, colours and OBBs.
  * `depth` — the depth image of the input: one 16-bit single-channel PNG with the pixel
    size of the input, each pixel holding the metric depth along the camera's optical axis
    and a reserved value where the model gives no valid depth. The depth scale and the
    reserved value are recorded in `README.md`. The image is the model's own depth, with no
    point-cloud attribute applied.
  * `ply` — the point cloud; per-point colour by default (see `color` below).
* `-o <file>` — write the result to this file instead of stdout; stdout then stays empty.
* `-p <key=value[,key=value…]>` — point-cloud attributes for the PLY output (table below);
  requires `-f ply` (it is refused with `-f json` and `-f depth`). Keys that are not given keep their defaults. An unknown key or an
  out-of-range value fails with an actionable error before any inference runs.

## Point-cloud attributes

Every command that writes a PLY takes the same `-p` attributes, defined here once; the spec of
each such command points to this table.

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `color` | `rgb` \| `segment` \| `height` \| `none` | `rgb` | Per-point colour: the image colour; the object colour of the scene description (unsegmented points mid-grey); a ramp along the up axis (estimated gravity for an image, the map's up axis for a map); or no colour properties at all. |
| `stride` | integer ≥ 1 | `1` | Keep every n-th pixel along each image axis. |
| `min-depth`, `max-depth` | metres | full range | Keep only pixels whose depth lies within the range. |
| `edge` | relative depth jump ≥ 0 | `0.04` | Drop pixels on depth discontinuities (flying pixels); `0` disables the filter. |
| `voxel` | metres ≥ 0 | `0` (off) | Keep one representative point per voxel, chosen deterministically. Colours are not averaged, so they stay exact. |
| `normals` | `on` \| `off` | `off` | Add `nx ny nz` float properties. |
| `label` | `on` \| `off` | `off` | Add an `int label` property holding the object `id` (`0` = unsegmented). |
| `encoding` | `binary` \| `ascii` | `binary` | `binary_little_endian 1.0` or ASCII PLY. |

* Pixel-level attributes (`stride`, depth range, `edge`) apply before unprojection, so they
  exist only for commands that write the cloud of a single image; for the cloud of a map they
  fail with an actionable error. `voxel` applies to the resulting 3D points.
* Attributes shape only the emitted cloud. Any scene description, OBBs, object `id`s and
  object colours a command reports are the same whatever `-p` says.
* The PLY header records the effective attributes, defaults included, in a `comment` line,
  so every file states how it was produced.
* The attribute set, its defaults and its validation are defined once in the shared package
  and reused by every command that writes a PLY and by the `view.sh` controls ([§2.5](view.md)).
