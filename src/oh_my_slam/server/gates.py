"""Model-side gate measurements, run one at a time with the inference server stopped.

    python -m oh_my_slam.server.gates g2 IMG [IMG ...]   MPS fp16 vs CPU fp32 depth agreement
    python -m oh_my_slam.server.gates g6 IMG [IMG ...]   MapAnything views per chunk vs memory

Prints one JSON document with the measurements to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.images import load_rgb


def gate_g2(images: list[Path], num_tokens: int = 2500, max_side: int = 1024) -> dict[str, Any]:
    from oh_my_slam.server.models.geometry_moge import MoGeGeometry

    rgbs = [load_rgb(p, max_side=max_side) for p in images]
    results: dict[str, list[np.ndarray]] = {}
    timings: dict[str, list[float]] = {}
    for label, device, fp16 in (("mps_fp16", "mps", True), ("mps_fp32", "mps", False),
                                ("cpu_fp32", "cpu", False)):
        geo = MoGeGeometry()
        geo.load(device)
        geo.warmup()
        outs, ts = [], []
        for rgb in rgbs:
            t = time.perf_counter()
            out = geo.infer_array(rgb, None, num_tokens, fp16=fp16)
            ts.append(time.perf_counter() - t)
            outs.append(out["depth"])
        results[label] = outs
        timings[label] = ts
        del geo
    per_image = []
    for a, b, c in zip(results["mps_fp16"], results["cpu_fp32"], results["mps_fp32"], strict=True):
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        rel16 = float(np.median(np.abs(a[ok] - b[ok]) / b[ok]))
        ok2 = np.isfinite(c) & np.isfinite(b) & (c > 0) & (b > 0)
        rel32 = float(np.median(np.abs(c[ok2] - b[ok2]) / b[ok2]))
        per_image.append({"mps_fp16_vs_cpu_fp32": rel16, "mps_fp32_vs_cpu_fp32": rel32})
    med = float(np.median([x["mps_fp16_vs_cpu_fp32"] for x in per_image]))
    return {
        "gate": "G2",
        "images": [str(p) for p in images],
        "per_image": per_image,
        "median_rel_diff_fp16": med,
        "median_s": {k: float(np.median(v)) for k, v in timings.items()},
        "pass": med <= 0.01,
    }


def gate_g6(images: list[Path], sizes: tuple[int, ...] = (8, 12, 16, 24),
            resolution: int = 518) -> dict[str, Any]:
    import torch

    from oh_my_slam.client.protocol import MultiviewRequest
    from oh_my_slam.server.models.multiview_mapanything import MapAnythingMultiview

    mv = MapAnythingMultiview()
    mv.load("mps")
    out: list[dict[str, Any]] = []
    tmp = Path("/tmp/oms-gate-g6")
    tmp.mkdir(exist_ok=True)
    for n in sizes:
        paths = [str(images[i % len(images)]) for i in range(n)]
        torch.mps.empty_cache()
        base = torch.mps.driver_allocated_memory()
        t = time.perf_counter()
        peak = base
        try:
            mv.run(MultiviewRequest(image_paths=paths, out_dir=str(tmp), resolution=resolution))
            peak = max(peak, torch.mps.driver_allocated_memory())
            ok = True
            err = None
        except Exception as exc:  # OOM is the interesting outcome
            ok, err = False, f"{type(exc).__name__}: {exc}"[:300]
        out.append({
            "views": n, "ok": ok, "error": err, "seconds": time.perf_counter() - t,
            "driver_allocated_gb_after": peak / 1e9, "model_gb": base / 1e9,
        })
        torch.mps.empty_cache()
    return {"gate": "G6", "resolution": resolution, "runs": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oh_my_slam.server.gates")
    ap.add_argument("gate", choices=["g2", "g6"])
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--resolution", type=int, default=518)
    args = ap.parse_args(argv)
    res = gate_g2(args.images) if args.gate == "g2" else gate_g6(args.images,
                                                                 resolution=args.resolution)
    sys.stdout.write(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
