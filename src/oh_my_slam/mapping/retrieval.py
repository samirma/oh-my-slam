"""Image retrieval for matching: global descriptors (DINOv2 class token from the geometry model),
top-K candidate pairs, and the pair lists fed to COLMAP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def similarity(a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.float64]:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def top_k_pairs(desc_query: NDArray[Any], desc_db: NDArray[Any], k: int,
                query_ids: list[int], db_ids: list[int], min_gap: int = 0
                ) -> set[tuple[int, int]]:
    """For each query, the ``k`` most similar database items (excluding itself and items within
    ``min_gap`` of it in capture order). Pairs are returned as sorted id tuples."""
    if len(desc_query) == 0 or len(desc_db) == 0 or k <= 0:
        return set()
    sim = similarity(desc_query, desc_db)
    qa = np.asarray(query_ids)[:, None]
    da = np.asarray(db_ids)[None, :]
    sim[np.abs(qa - da) <= min_gap] = -np.inf
    pairs: set[tuple[int, int]] = set()
    kk = min(k, sim.shape[1])
    order = np.argsort(-sim, axis=1)[:, :kk]
    for qi, row in enumerate(order):
        for j in row:
            if np.isfinite(sim[qi, j]):
                a, b = query_ids[qi], db_ids[j]
                if a != b:
                    pairs.add((min(a, b), max(a, b)))
    return pairs


def sequential_pairs(ids: list[int], overlap: int) -> set[tuple[int, int]]:
    out = set()
    for i, a in enumerate(ids):
        for b in ids[i + 1: i + 1 + overlap]:
            out.add((min(a, b), max(a, b)))
    return out


def all_pairs(ids_a: list[int], ids_b: list[int] | None = None) -> set[tuple[int, int]]:
    out = set()
    if ids_b is None:
        for i, a in enumerate(ids_a):
            for b in ids_a[i + 1:]:
                out.add((min(a, b), max(a, b)))
    else:
        for a in ids_a:
            for b in ids_b:
                if a != b:
                    out.add((min(a, b), max(a, b)))
    return out


def write_pair_list(path: Path, pairs: set[tuple[int, int]], names: dict[int, str]) -> int:
    lines = [f"{names[a]} {names[b]}" for a, b in sorted(pairs)]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)
