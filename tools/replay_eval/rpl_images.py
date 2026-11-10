"""Image size and EXIF focal length of JPEG and PNG frames, with the standard library only."""

from __future__ import annotations

import struct

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# JPEG start-of-frame markers (every SOFn except DHT C4, JPG C8 and DAC CC)
SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
TAG_EXIF_IFD = 0x8769
TAG_FOCAL_LENGTH = 0x920A
TAG_FP_XRES = 0xA20E
TAG_FP_YRES = 0xA20F
TAG_FP_UNIT = 0xA210
# FocalPlaneResolutionUnit -> millimetres per unit
UNIT_MM = {1: 1.0, 2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}
UNIT_NAMES = {"inch": 2, "cm": 3, "mm": 4}


class ImageError(ValueError):
    """Not a JPEG or PNG this reader understands."""


def mime(data: bytes) -> str:
    if data.startswith(PNG_SIGNATURE):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    raise ImageError("not a JPEG or PNG image")


def _jpeg_segments(data: bytes) -> list[tuple[int, bytes]]:
    """(marker, payload) of each JPEG segment before the scan data."""
    out: list[tuple[int, bytes]] = []
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ImageError("corrupt JPEG: expected a marker")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        out.append((marker, data[i + 4:i + 2 + length]))
        if marker == 0xDA:  # start of scan: no more headers
            break
        i += 2 + length
    return out


def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    out: list[tuple[bytes, bytes]] = []
    i = len(PNG_SIGNATURE)
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i:i + 4])[0]
        kind = data[i + 4:i + 8]
        out.append((kind, data[i + 8:i + 8 + length]))
        if kind == b"IEND":
            break
        i += 12 + length
    return out


def image_size(data: bytes) -> tuple[int, int]:
    """(width, height) in pixels."""
    if mime(data) == "image/png":
        for kind, body in _png_chunks(data):
            if kind == b"IHDR":
                w, h = struct.unpack(">II", body[:8])
                return int(w), int(h)
        raise ImageError("PNG without IHDR")
    for marker, body in _jpeg_segments(data):
        if marker in SOF_MARKERS:
            h, w = struct.unpack(">HH", body[1:5])
            return int(w), int(h)
    raise ImageError("JPEG without a start-of-frame segment")


def _tiff_block(data: bytes) -> bytes | None:
    if mime(data) == "image/png":
        for kind, body in _png_chunks(data):
            if kind == b"eXIf":
                return body
        return None
    for marker, body in _jpeg_segments(data):
        if marker == 0xE1 and body.startswith(b"Exif\x00\x00"):
            return body[6:]
    return None


def _ifd(tiff: bytes, offset: int, order: str) -> dict[int, float]:
    """Numeric tags of one IFD (first value of each), plus the Exif IFD pointer."""
    sizes = {3: 2, 4: 4, 5: 8, 9: 4, 10: 8}
    tags: dict[int, float] = {}
    (count,) = struct.unpack(order + "H", tiff[offset:offset + 2])
    for k in range(count):
        entry = tiff[offset + 2 + 12 * k:offset + 14 + 12 * k]
        tag, kind, n = struct.unpack(order + "HHI", entry[:8])
        if kind not in sizes or n < 1:
            continue
        raw = entry[8:12]
        if sizes[kind] > 4:
            (ptr,) = struct.unpack(order + "I", raw)
            raw = tiff[ptr:ptr + 8]
        if kind == 3:
            tags[tag] = struct.unpack(order + "H", raw[:2])[0]
        elif kind in (4, 9):
            tags[tag] = struct.unpack(order + ("I" if kind == 4 else "i"), raw[:4])[0]
        else:
            num, den = struct.unpack(order + ("II" if kind == 5 else "ii"), raw[:8])
            if den != 0:
                tags[tag] = num / den
    return tags


def exif_focal_px(data: bytes, default_unit: str = "mm") -> tuple[float, float] | None:
    """(fx, fy) in pixels from FocalLength and FocalPlaneX/YResolution, or None.

    fx = FocalLength [mm] x FocalPlaneXResolution [px per unit] / mm per unit. The unit is
    FocalPlaneResolutionUnit, or ``default_unit`` when the tag is absent.
    """
    tiff = _tiff_block(data)
    if tiff is None or len(tiff) < 8:
        return None
    order = {b"II": "<", b"MM": ">"}.get(tiff[:2])
    if order is None:
        return None
    (ifd0,) = struct.unpack(order + "I", tiff[4:8])
    tags = _ifd(tiff, ifd0, order)
    if TAG_EXIF_IFD in tags:
        tags = {**tags, **_ifd(tiff, int(tags[TAG_EXIF_IFD]), order)}
    if TAG_FOCAL_LENGTH not in tags or TAG_FP_XRES not in tags:
        return None
    unit = int(tags.get(TAG_FP_UNIT, UNIT_NAMES[default_unit]))
    mm_per_unit = UNIT_MM.get(unit, 1.0)
    fx = tags[TAG_FOCAL_LENGTH] * tags[TAG_FP_XRES] / mm_per_unit
    fy = tags[TAG_FOCAL_LENGTH] * tags.get(TAG_FP_YRES, tags[TAG_FP_XRES]) / mm_per_unit
    return float(fx), float(fy)
