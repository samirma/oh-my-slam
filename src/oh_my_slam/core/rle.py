"""COCO run-length encoding of binary masks (column-major counts, compressed string form)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def encode_counts(mask: NDArray[Any]) -> list[int]:
    """Column-major run lengths, starting with a run of zeros (possibly empty)."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return []
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs = [0, *runs]
    return [int(r) for r in runs]


def decode_counts(counts: list[int], height: int, width: int) -> NDArray[np.bool_]:
    total = int(sum(counts))
    if total != height * width:
        raise ValueError(f"RLE covers {total} pixels, expected {height * width}")
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, counts)
    return flat.reshape((height, width), order="F")


def counts_to_string(counts: list[int]) -> str:
    """COCO ``rleToString``: LEB128-like 6-bit chunks with delta coding from index 3."""
    out: list[str] = []
    for i, c in enumerate(counts):
        x = int(c)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            ch = x & 0x1F
            x >>= 5
            more = (x != -1) if (ch & 0x10) else (x != 0)
            if more:
                ch |= 0x20
            out.append(chr(ch + 48))
    return "".join(out)


def string_to_counts(s: str) -> list[int]:
    counts: list[int] = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            ch = ord(s[p]) - 48
            x |= (ch & 0x1F) << (5 * k)
            more = bool(ch & 0x20)
            p += 1
            k += 1
            if not more and (ch & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def encode(mask: NDArray[Any]) -> dict[str, Any]:
    h, w = np.asarray(mask).shape
    return {"size": [int(h), int(w)], "counts": counts_to_string(encode_counts(mask))}


def decode(rle: dict[str, Any]) -> NDArray[np.bool_]:
    h, w = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = string_to_counts(counts)
    return decode_counts(list(counts), h, w)


def area(rle: dict[str, Any]) -> int:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = string_to_counts(counts)
    return int(sum(counts[1::2]))
