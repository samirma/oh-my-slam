"""Accuracy evaluation (AC6, AC9) on public datasets in ``~/oh-my-slam-data`` (downloaded only
with the user's consent, U2). Needs a running inference server.

    python -m oh_my_slam.tools.evaluate depth --nyu DIR [--n 100]
        DIR/rgb/*.png + DIR/depth/*.npy (metres); NYUv2 intrinsics. AbsRel and δ1.
    python -m oh_my_slam.tools.evaluate tum --seq DIR [--fps 2]
        TUM RGB-D sequence folder (rgb.txt, groundtruth.txt, rgb/). ATE after Sim(3) and SE(3),
        scale error, registration rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.geometry import umeyama
from oh_my_slam.core.types import Intrinsics

NYU_K = Intrinsics(518.8579, 519.4696, 325.5824, 253.7362, 640, 480, "given")


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    ok = (gt > 0.1) & (gt < 10.0) & np.isfinite(pred) & (pred > 0)
    p, g = pred[ok], gt[ok]
    ratio = np.maximum(p / g, g / p)
    return {"absrel": float(np.mean(np.abs(p - g) / g)), "delta1": float(np.mean(ratio < 1.25))}


def eval_depth(root: Path, n: int) -> dict[str, Any]:
    from oh_my_slam.reconstruction.api import reconstruct_image

    rgbs = sorted((root / "rgb").glob("*.png"))[:n]
    per = []
    for rgb in rgbs:
        gt = np.load(root / "depth" / (rgb.stem + ".npy"))
        fr = reconstruct_image(rgb, intrinsics=NYU_K, want_gravity=False)
        pred = fr.depth
        if pred.shape != gt.shape:
            import cv2

            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        per.append(depth_metrics(pred, gt))
    return {"images": len(per), "absrel": float(np.mean([m["absrel"] for m in per])),
            "delta1": float(np.mean([m["delta1"] for m in per])),
            "pass": bool(np.mean([m["absrel"] for m in per]) <= 0.10
                         and np.mean([m["delta1"] for m in per]) >= 0.90)}


def _read_tum_list(path: Path) -> list[list[str]]:
    return [ln.split() for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]


def ate(est: np.ndarray, gt: np.ndarray, with_scale: bool) -> tuple[float, float]:
    sim = umeyama(est, gt, with_scale=with_scale)
    err = np.linalg.norm(sim.apply(est) - gt, axis=1)
    return float(np.sqrt(np.mean(err**2))), sim.s


def eval_tum(seq: Path, fps: float) -> dict[str, Any]:
    from oh_my_slam.mapping.api import update
    from oh_my_slam.mapping.store import MapReader

    rgb = _read_tum_list(seq / "rgb.txt")
    gt = np.array([[float(v) for v in row[:4]] for row in _read_tum_list(seq / "groundtruth.txt")])
    step = max(1, round((len(rgb) / (float(rgb[-1][0]) - float(rgb[0][0]))) / fps))
    chosen = rgb[::step]
    with tempfile.TemporaryDirectory(prefix="oms-tum-") as tmp:
        frames = Path(tmp) / "in"
        frames.mkdir()
        stamps = {}
        for i, (ts, rel) in enumerate(chosen):
            dst = frames / f"{i:05d}.png"
            dst.symlink_to((seq / rel).resolve())
            stamps[f"{i:05d}"] = float(ts)
        res = update(Path(tmp) / "map", [frames], progress=lambda m: None)
        reader = MapReader(Path(tmp) / "map")
        est, ref = [], []
        for f in reader.frames:
            ts = stamps[Path(f.source).stem]
            j = int(np.argmin(np.abs(gt[:, 0] - ts)))
            if abs(gt[j, 0] - ts) < 0.02:
                est.append(f.T_map_cam.t)
                ref.append(gt[j, 1:4])
    est_a, ref_a = np.array(est), np.array(ref)
    ate_sim3, s = ate(est_a, ref_a, True)
    ate_se3, _ = ate(est_a, ref_a, False)
    reg = len(res.new_frames) / len(chosen)
    return {"sequence": seq.name, "keyframes": len(chosen), "registered": reg,
            "ate_sim3_m": ate_sim3, "ate_se3_m": ate_se3, "scale_error": abs(s - 1.0),
            "pass": ate_sim3 <= 0.05 and ate_se3 <= 0.10 and abs(s - 1) <= 0.10 and reg >= 0.95}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oh_my_slam.tools.evaluate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("depth")
    d.add_argument("--nyu", type=Path, required=True)
    d.add_argument("--n", type=int, default=100)
    t = sub.add_parser("tum")
    t.add_argument("--seq", type=Path, required=True)
    t.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args(argv)
    res = eval_depth(args.nyu, args.n) if args.cmd == "depth" else eval_tum(args.seq, args.fps)
    sys.stdout.write(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
