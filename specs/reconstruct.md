# 2.2 Single-frame reconstruction — `reconstruct.sh`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

```sh
reconstruct.sh -i <image>                # JSON scene description (default) to stdout
reconstruct.sh -i <image> -f ply         # point cloud to stdout
reconstruct.sh -i <image> -f ply -p color=segment,voxel=0.01,normals=on
reconstruct.sh -i <image> -f ply -o cloud.ply   # point cloud to a file
```

* `-i <image>` — input RGB image.
* `-f json|ply` — output format, **default `json`**. Unless `-o` is given, the result is
  written to **stdout** so it can be piped or redirected; diagnostics go to stderr.
  * `json` — the scene description of [§3](high_level_spec.md#3-scene-description-json-returned-by-the-tools): detected objects, their labels, colours and OBBs;
    the same objects, `id`s and colours as `segment.sh -i` with default options.
  * `ply` — the point cloud; per-point colour by default (see `color` below).
* `-o <file>` — write the result to this file instead of stdout; stdout then stays empty.
* `-p <key=value[,key=value…]>` — point-cloud attributes for the PLY output (table below);
  requires a PLY output. Keys that are not given keep their defaults. An unknown key or an
  out-of-range value fails with an actionable error before any inference runs.

## Point-cloud attributes

The same `-p` attributes apply to every command that writes a PLY: `reconstruct.sh -f ply`,
`mapper.sh -f ply`, and `segment.sh` with `-f ply` or `-d` ([§2.3](mapper.md), [§2.4](segment.md)).

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `color` | `rgb` \| `segment` \| `height` \| `none` | `rgb` (`segment` in `segment.sh`) | Per-point colour: the image colour; the object colour of the [§2.4](segment.md#colour-contract) colour contract (unsegmented points mid-grey, as in `segments.ply`); a ramp along the up axis (estimated gravity for an image, the map's up axis for a map); or no colour properties at all. |
| `stride` | integer ≥ 1 | `1` | Keep every n-th pixel along each image axis. |
| `min-depth`, `max-depth` | metres | full range | Keep only pixels whose depth lies within the range. |
| `edge` | relative depth jump ≥ 0 | `0.04` | Drop pixels on depth discontinuities (flying pixels); `0` disables the filter. |
| `voxel` | metres ≥ 0 | `0` (off) | Keep one representative point per voxel, chosen deterministically. Colours are not averaged, so object colours stay exact. |
| `normals` | `on` \| `off` | `off` | Add `nx ny nz` float properties. |
| `label` | `on` \| `off` | `off` | Add an `int label` property holding the object `id` (`0` = unsegmented). |
| `encoding` | `binary` \| `ascii` | `binary` | `binary_little_endian 1.0` or ASCII PLY. |

* Pixel-level attributes (`stride`, depth range, `edge`) apply before unprojection, so they
  exist only for single images (`reconstruct.sh`, `segment.sh -i`); on a map (`mapper.sh`,
  `segment.sh -m`) they fail with an actionable error. `voxel` applies to the resulting 3D
  points.
* `segment.sh` always colours by object: `color` is fixed to `segment` there, and any other
  value fails. Its remaining attributes shape both stdout and `segments.ply`.
* Attributes shape only the emitted cloud. The scene description, OBBs, object `id`s and
  object colours are the same whatever `-p` says.
* The PLY header records the effective attributes, defaults included, in a `comment` line,
  so every file states how it was produced.
* The attribute set, its defaults and its validation are defined once in the shared package
  and reused by every command that writes a PLY and by the `view.sh` controls ([§2.5](view.md)).
