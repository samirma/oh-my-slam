"""Development & performance report (R44), in four steps (run them one at a time):

    python -m oh_my_slam.tools.perf_report measure [--out DIR] [--only SECTIONS] [--reps 5]
    python -m oh_my_slam.tools.perf_report renders --out DIR
    python -m oh_my_slam.tools.perf_report shots --out DIR
    python -m oh_my_slam.tools.perf_report write --out DIR

``measure`` runs, strictly sequentially against the warm inference server: ``reconstruct.sh``
(JSON and ``-f ply``) and ``segment.sh -i -o`` on restaurant.jpg ``--reps`` times; a whole-map
build per map input (``mapper.sh update -t full``, ``-fps 2`` for videos) followed by
``segment.sh -m -o``; then, per map, ``--reps`` single-image updates (``-t single``), each on a
fresh APFS clone of the built map (``cp -c -R``), plus one more with ``-f ply -t full`` whose
stdout is the rendered ``-f ply`` output. Every command runs with ``OH_MY_SLAM_TIMINGS`` (per-stage
record, see ``core.timing``) while a sampler reads the server's physical footprint every 0.25 s.
Results go to ``DIR/measurements.json`` (saved after every run), stdout/stderr to ``outputs/`` and
``logs/``. ``renders`` draws the PLY outputs, ``shots`` drives ``view.sh`` in a headless browser
(Edge), ``write`` produces ``report.md`` + ``report.html``. Default ``DIR``:
``~/oh-my-slam-data/reports/<UTC>/``. User inputs are only read.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
INPUTS = Path("/Users/U124317/robot_view")
MAPS = {
    "ainex": {"input": "ainex-captures", "fps": None,
              "update": {"kind": "image", "src": "ainex-captures/040_right_to_090_level.jpg"}},
    "church": {"input": "church.mp4", "fps": 2,
               "update": {"kind": "frame", "src": "church.mp4", "t": 12.75}},
    "livingroom": {"input": "livingroom.mp4", "fps": 2,
                   "update": {"kind": "frame", "src": "livingroom.mp4", "t": 45.0}},
}
SECTIONS = ("machine", "restaurant", "ainex", "church", "livingroom")


# ------------------------------------------------------------------------------------------------
# server memory


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
        "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
        "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
        "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
        "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time",
        "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
        "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
        "ri_runnable_time")]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None


def phys_footprint(pid: int) -> tuple[float, float] | None:
    """(current, lifetime peak) physical footprint in GB — includes Metal/MPS allocations."""
    if _libproc is None:
        return None
    info = _RUsageInfoV4()
    if _libproc.proc_pid_rusage(int(pid), 4, ctypes.byref(info)) != 0:
        return None
    return info.ri_phys_footprint / 1e9, info.ri_lifetime_max_phys_footprint / 1e9


def server_pid() -> int | None:
    from oh_my_slam.core import paths

    try:
        return int(json.loads(paths.state_file().read_text())["pid"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None


class ServerMemory:
    """Samples the server's physical footprint while a command runs (peak over the run)."""

    def __init__(self, pid: int | None, every: float = 0.25) -> None:
        self.pid, self.every = pid, every
        self.start = self.peak = self.end = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            fp = phys_footprint(self.pid) if self.pid else None
            if fp:
                self.peak = max(self.peak, fp[0])
            self._stop.wait(self.every)

    def __enter__(self) -> ServerMemory:
        fp = phys_footprint(self.pid) if self.pid else None
        self.start = self.peak = fp[0] if fp else 0.0
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        fp = phys_footprint(self.pid) if self.pid else None
        self.end = fp[0] if fp else 0.0
        self.peak = max(self.peak, self.end)

    def as_dict(self) -> dict[str, float]:
        return {"start_gb": round(self.start, 2), "peak_gb": round(self.peak, 2),
                "end_gb": round(self.end, 2)}


# ------------------------------------------------------------------------------------------------
# measurement


class Measure:
    def __init__(self, out: Path, reps: int) -> None:
        self.out = out
        self.reps = reps
        self.file = out / "measurements.json"
        self.data: dict[str, Any] = (json.loads(self.file.read_text()) if self.file.exists()
                                     else {"runs": [], "maps": {}, "update_inputs": {}})
        self.pid = server_pid()

    def save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        tmp.replace(self.file)

    def drop_group(self, group: str) -> None:
        self.data["runs"] = [r for r in self.data["runs"] if r["group"] != group]

    def run(self, tag: str, group: str, args: list[str], stdout_to: Path | None = None,
            timeout: float = 3600) -> dict[str, Any]:
        for d in ("logs", "timings"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        tpath = self.out / "timings" / f"{tag}.json"
        tpath.unlink(missing_ok=True)
        env = {**os.environ, "OH_MY_SLAM_TIMINGS": str(tpath)}
        print(f"  run {tag}: {' '.join(args)}", file=sys.stderr, flush=True)
        with ServerMemory(self.pid) as mem:
            t0 = time.perf_counter()
            res = subprocess.run(args, capture_output=True, env=env, timeout=timeout,
                                 cwd=self.out)
            wall = time.perf_counter() - t0
        (self.out / "logs" / f"{tag}.stderr.txt").write_bytes(res.stderr)
        if stdout_to is not None:
            stdout_to.parent.mkdir(parents=True, exist_ok=True)
            stdout_to.write_bytes(res.stdout)
        rec = {"tag": tag, "group": group, "cmd": " ".join(args), "exit": res.returncode,
               "wall_s": round(wall, 3), "stdout_bytes": len(res.stdout),
               "stdout": str(stdout_to.relative_to(self.out)) if stdout_to else None,
               "server_mem": mem.as_dict(),
               "timings": json.loads(tpath.read_text()) if tpath.exists() else None}
        self.data["runs"].append(rec)
        self.save()
        print(f"    exit {res.returncode} in {wall:.1f} s, server peak {mem.peak:.1f} GB",
              file=sys.stderr, flush=True)
        if res.returncode != 0:
            print(res.stderr.decode(errors="replace")[-1500:], file=sys.stderr, flush=True)
        return rec


def sh(name: str) -> str:
    return str(REPO / name)


def du_bytes(path: Path) -> int:
    res = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True)
    return int(res.stdout.split()[0]) * 1024 if res.returncode == 0 else 0


def machine(m: Measure) -> None:
    from oh_my_slam.client.client import InferenceClient

    health = InferenceClient().health().model_dump()
    fp = phys_footprint(m.pid) if m.pid else None
    sysctl = {k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
              for k in ("machdep.cpu.brand_string", "hw.ncpu", "hw.memsize")}
    tools = {
        "colmap": subprocess.run(["colmap", "version"], capture_output=True,
                                 text=True).stdout.splitlines()[:1],
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
    }
    m.data["machine"] = {
        "cpu": sysctl["machdep.cpu.brand_string"], "cores": sysctl["hw.ncpu"],
        "memory_gb": round(int(sysctl["hw.memsize"] or 0) / 2**30, 1), "tools": tools,
        "server": {k: health.get(k) for k in ("status", "device", "precision", "versions",
                                              "uptime_s")},
        "server_models": {k: {"name": v["name"], "loaded": v["loaded"]}
                          for k, v in health.get("models", {}).items()},
        "server_footprint_gb_at_start": None if not fp else round(fp[0], 2),
        "server_lifetime_peak_gb_at_start": None if not fp else round(fp[1], 2),
        "measured_at": datetime.now(UTC).isoformat(),
    }
    m.save()


def restaurant(m: Measure) -> None:
    img = INPUTS / "restaurant.jpg"
    for g in ("restaurant_json", "restaurant_ply", "restaurant_segment", "restaurant_warmup"):
        m.drop_group(g)
    out = m.out / "outputs" / "restaurant"
    m.run("restaurant_warmup", "restaurant_warmup", [sh("reconstruct.sh"), "-i", str(img)])
    for k in range(1, m.reps + 1):
        m.run(f"restaurant_json_{k}", "restaurant_json", [sh("reconstruct.sh"), "-i", str(img)],
              out / "reconstruct.json")
    for k in range(1, m.reps + 1):
        m.run(f"restaurant_ply_{k}", "restaurant_ply",
              [sh("reconstruct.sh"), "-i", str(img), "-f", "ply"], out / "reconstruct.ply")
    for k in range(1, m.reps + 1):
        seg = out / "segment_o"
        shutil.rmtree(seg, ignore_errors=True)
        m.run(f"restaurant_segment_{k}", "restaurant_segment",
              [sh("segment.sh"), "-i", str(img), "-o", str(seg)], out / "segment.json")


def _update_input(m: Measure, name: str) -> Path:
    spec = MAPS[name]["update"]
    src = INPUTS / spec["src"]
    if spec["kind"] == "image":
        return src
    dst = m.out / "inputs" / f"{name}_t{spec['t']:g}s.jpg"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{spec['t']}", "-i", str(src),
                        "-frames:v", "1", "-q:v", "2", str(dst)], check=True)
    return dst


def build_map(m: Measure, name: str) -> None:
    spec = MAPS[name]
    mdir = m.out / "maps" / name
    for g in (f"{name}_build", f"{name}_segment_m", f"{name}_update", f"{name}_update_ply"):
        m.drop_group(g)
    shutil.rmtree(mdir, ignore_errors=True)
    args = [sh("mapper.sh"), "update", "-a", str(INPUTS / spec["input"]), "-m", str(mdir),
            "-t", "full"]
    if spec["fps"]:
        args += ["-fps", str(spec["fps"])]
    out = m.out / "outputs" / name
    rec = m.run(f"{name}_build", f"{name}_build", args, out / "full.json")
    if rec["exit"] != 0:
        return
    meta = json.loads((mdir / "map.json").read_text())
    frames = json.loads((mdir / "frames.json").read_text())["frames"]
    objs = json.loads((mdir / "objects.json").read_text())
    m.data["maps"][name] = {
        "dir": str(mdir.relative_to(m.out)), "input": str(INPUTS / spec["input"]),
        "fps": spec["fps"], "size_bytes": du_bytes(mdir),
        "size_breakdown": {p.name: du_bytes(p) for p in sorted(mdir.iterdir())},
        "keyframes": len(frames),
        "low_confidence": sum(bool(f.get("low_confidence")) for f in frames),
        "objects": len(json.loads((out / "full.json").read_text())["openlabel"].get(
            "objects", {})),
        "objects_stored": len(objs.get("objects", [])),
        "sfm": meta["updates"][-1].get("sfm"), "floor_z": meta.get("floor_z"),
    }
    m.save()
    seg = out / "segment_m"
    shutil.rmtree(seg, ignore_errors=True)
    m.run(f"{name}_segment_m", f"{name}_segment_m",
          [sh("segment.sh"), "-m", str(mdir), "-o", str(seg)], out / "segment_m.json")


def update_map(m: Measure, name: str) -> None:
    mdir = m.out / "maps" / name
    if not (mdir / "map.json").exists():
        print(f"  no {name} map; skipping updates", file=sys.stderr)
        return
    for g in (f"{name}_update", f"{name}_update_ply"):
        m.drop_group(g)
    img = _update_input(m, name)
    m.data["update_inputs"][name] = str(img)
    work = m.out / "work"
    work.mkdir(exist_ok=True)
    out = m.out / "outputs" / name
    for k in range(1, m.reps + 2):
        copy = work / f"{name}_update_{k}"
        shutil.rmtree(copy, ignore_errors=True)
        subprocess.run(["cp", "-c", "-R", str(mdir), str(copy)], check=True)
        if k <= m.reps:
            m.run(f"{name}_update_{k}", f"{name}_update",
                  [sh("mapper.sh"), "update", "-a", str(img), "-m", str(copy), "-t", "single"],
                  out / "update_single.json")
        else:  # the -f ply output (whole updated map cloud) for the format renderings
            m.run(f"{name}_update_ply", f"{name}_update_ply",
                  [sh("mapper.sh"), "update", "-a", str(img), "-m", str(copy), "-t", "full",
                   "-f", "ply"], out / "update_full.ply")
        if k == m.reps:
            keep = m.out / "maps" / f"{name}_updated"
            shutil.rmtree(keep, ignore_errors=True)
            copy.rename(keep)
        else:
            shutil.rmtree(copy, ignore_errors=True)


def measure(out: Path, only: list[str], reps: int) -> None:
    m = Measure(out, reps)
    if m.pid is None:
        raise SystemExit("inference server not running: ./start_inference_server.sh")
    for sec in only:
        print(f"== {sec}", file=sys.stderr, flush=True)
        if sec == "machine":
            machine(m)
        elif sec == "restaurant":
            restaurant(m)
        elif sec in MAPS:
            build_map(m, sec)
            update_map(m, sec)
        elif sec.endswith("_updates") and sec.removesuffix("_updates") in MAPS:
            update_map(m, sec.removesuffix("_updates"))
        else:
            raise SystemExit(f"unknown section {sec}")
    shutil.rmtree(out / "work", ignore_errors=True)


# ------------------------------------------------------------------------------------------------
# summaries used by the report


def runs(data: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return [r for r in data["runs"] if r["group"] == group and r["exit"] == 0]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {"min": min(values), "median": statistics.median(values), "max": max(values),
            "n": len(values)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oh_my_slam.tools.perf_report")
    ap.add_argument("step", choices=("measure", "renders", "shots", "write"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default=",".join(SECTIONS))
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = (args.out or Path.home() / "oh-my-slam-data" / "reports" / stamp).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "measure":
        measure(out, [s for s in args.only.split(",") if s], args.reps)
    else:
        from oh_my_slam.tools import perf_report_render as rr

        getattr(rr, args.step)(out)
    print(str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
