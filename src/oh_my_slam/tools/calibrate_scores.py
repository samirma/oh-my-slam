"""Fit the score calibration maps (C16): isotonic regression of "detection is correct" on the raw
score, per segmentation path, on an LVIS validation subset (downloaded only with consent, U2).

    python -m oh_my_slam.tools.calibrate_scores --lvis-json lvis_v1_val.json --images DIR \
        [--n 500] [--holdout 0.3] [--source yoloe] [--write]

A detection is correct when it overlaps a ground-truth instance of the same (normalised) category
with mask IoU >= 0.5. The fitted map is monotone piecewise-linear; ``--write`` stores it in
``segmentation/data/calibration_<source>.json``. Precision among held-out detections with a
calibrated score >= 0.5 is reported (AC14 wants >= 0.5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def isotonic(x: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pool-adjacent-violators: non-decreasing fit of y on x; returns (knots_x, knots_y)."""
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    blocks: list[list[float]] = []  # [sum_y, count, x_min, x_max]
    for xv, yv in zip(xs, ys, strict=True):
        blocks.append([yv, 1.0, xv, xv])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, lo, _ = blocks.pop(-2)
            blocks[-1][0] += s
            blocks[-1][1] += c
            blocks[-1][2] = lo
    kx, ky = [], []
    for s, c, lo, hi in blocks:
        v = s / c
        kx += [lo, hi]
        ky += [v, v]
    kx_a, ky_a = np.array(kx), np.array(ky)
    keep = np.concatenate([[True], np.diff(kx_a) > 1e-9])
    return np.concatenate([[0.0], kx_a[keep], [1.0]]), np.concatenate(
        [[ky_a[0] if len(ky_a) else 0.0], ky_a[keep], [ky_a[-1] if len(ky_a) else 1.0]])


def precision_at(scores: NDArray[Any], correct: NDArray[Any], thr: float) -> float:
    sel = scores >= thr
    return float(correct[sel].mean()) if sel.any() else float("nan")


def collect(lvis_json: Path, images: Path, n: int) -> tuple[NDArray[Any], NDArray[Any]]:
    """(raw score, correct) for every detection on ``n`` images (needs the inference server)."""
    from oh_my_slam.core import rle
    from oh_my_slam.segmentation.detect import detect, mask_iou, normalize_label

    data = json.loads(Path(lvis_json).read_text())
    cats = {c["id"]: normalize_label(c["name"].replace("_", " ")) for c in data["categories"]}
    anns: dict[int, list[dict[str, Any]]] = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    scores, correct = [], []
    for info in data["images"][:n]:
        path = images / Path(info.get("coco_url", info.get("file_name", ""))).name
        if not path.exists():
            continue
        gts = anns.get(info["id"], [])
        labels = sorted({cats[a["category_id"]] for a in gts}) or ["person"]
        dets = detect(path, labels=labels, min_score=0.0)
        for d in dets:
            h, w = d.mask.shape
            ok = False
            for a in gts:
                if cats[a["category_id"]] != d.label or not isinstance(a["segmentation"], dict):
                    continue
                m = rle.decode(a["segmentation"])
                if m.shape == (h, w) and mask_iou(m, d.mask) >= 0.5:
                    ok = True
                    break
            scores.append(d.raw_score)
            correct.append(ok)
    return np.array(scores), np.array(correct, bool)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oh_my_slam.tools.calibrate_scores")
    ap.add_argument("--lvis-json", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--source", default="yoloe")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    s, c = collect(args.lvis_json, args.images, args.n)
    rng = np.random.default_rng(0)
    test = rng.random(len(s)) < args.holdout
    kx, ky = isotonic(s[~test], c[~test].astype(float))
    cal = np.interp(s[test], kx, ky)
    report = {"detections": len(s), "holdout": int(test.sum()),
              "precision_at_0.5_calibrated": precision_at(cal, c[test], 0.5),
              "precision_at_0.5_raw": precision_at(s[test], c[test], 0.5)}
    if args.write:
        out = Path(__file__).resolve().parents[1] / "segmentation" / "data" / \
            f"calibration_{args.source}.json"
        out.write_text(json.dumps({"source": args.source, "status": "fitted",
                                   "knots_raw": kx.tolist(), "knots_calibrated": ky.tolist(),
                                   "report": report}, indent=1) + "\n")
    sys.stdout.write(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
