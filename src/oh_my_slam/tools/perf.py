"""Performance runs for AC22, one at a time (never run two of these concurrently).

    python -m oh_my_slam.tools.perf latency IMAGE [--n 10]   reconstruct.sh -f ply / json p50, p90
    python -m oh_my_slam.tools.perf cold-start               stop + start the server, seconds
    python -m oh_my_slam.tools.perf memory                   server physical footprint (GB)
    python -m oh_my_slam.tools.perf gates IMG...             G2 and G6 (stops the server first)

Results are printed as one JSON document on stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def _run(args: list[str], timeout: float = 900) -> tuple[subprocess.CompletedProcess[bytes], float]:
    t0 = time.perf_counter()
    res = subprocess.run(args, capture_output=True, timeout=timeout)
    return res, time.perf_counter() - t0


def latency(image: Path, n: int) -> dict[str, Any]:
    out: dict[str, Any] = {"image": str(image), "runs": n}
    for fmt in ("ply", "json"):
        _run([str(REPO / "reconstruct.sh"), "-i", str(image), "-f", fmt])  # warm-up
        times = []
        for _ in range(n):
            res, dt = _run([str(REPO / "reconstruct.sh"), "-i", str(image), "-f", fmt])
            if res.returncode != 0:
                raise RuntimeError(res.stderr.decode()[-500:])
            times.append(dt)
        times.sort()
        out[fmt] = {"p50_s": statistics.median(times), "p90_s": times[int(0.9 * (n - 1))],
                    "min_s": times[0]}
    return out


def server_pid() -> int | None:
    from oh_my_slam.core import paths

    try:
        return int(json.loads(paths.state_file().read_text())["pid"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None


def footprint_gb(pid: int) -> dict[str, float]:
    """Physical footprint (includes Metal/GPU allocations on Apple silicon) via `footprint`."""
    res = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True)
    m = re.search(r"Footprint:\s*([\d.]+)\s*([KMG]B)", res.stdout)
    peak = re.search(r"peak.*?:\s*([\d.]+)\s*([KMG]B)", res.stdout, re.IGNORECASE)
    scale = {"KB": 1e-6, "MB": 1e-3, "GB": 1.0}
    out = {}
    if m:
        out["footprint_gb"] = float(m.group(1)) * scale[m.group(2)]
    if peak:
        out["peak_gb"] = float(peak.group(1)) * scale[peak.group(2)]
    import psutil

    out["rss_gb"] = psutil.Process(pid).memory_info().rss / 1e9
    return out


def cold_start() -> dict[str, Any]:
    start = str(REPO / "start_inference_server.sh")
    _run([start, "--stop"])
    res, dt = _run([start], timeout=1800)
    return {"ok": res.returncode == 0, "cold_start_s": dt}


def gates(images: list[Path]) -> dict[str, Any]:
    start = str(REPO / "start_inference_server.sh")
    _run([start, "--stop"])
    py = str(REPO / ".venv" / "bin" / "python")
    out = {}
    for g in ("g2", "g6"):
        res, _ = _run([py, "-m", "oh_my_slam.server.gates", g, *map(str, images)], timeout=1800)
        out[g] = json.loads(res.stdout) if res.returncode == 0 else {"error": res.stderr[-500:]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oh_my_slam.tools.perf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lat = sub.add_parser("latency")
    lat.add_argument("image", type=Path)
    lat.add_argument("--n", type=int, default=10)
    sub.add_parser("cold-start")
    sub.add_parser("memory")
    gt = sub.add_parser("gates")
    gt.add_argument("images", nargs="+", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "latency":
        res: dict[str, Any] = latency(args.image, args.n)
    elif args.cmd == "cold-start":
        res = cold_start()
    elif args.cmd == "memory":
        pid = server_pid()
        res = {"pid": pid, **(footprint_gb(pid) if pid else {})}
    else:
        res = gates(args.images)
    sys.stdout.write(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
