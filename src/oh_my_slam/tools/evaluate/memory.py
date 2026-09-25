"""Memory of the measured processes: the command's process tree (resident set) and the inference
server (physical footprint, which on Apple silicon includes Metal/MPS allocations).

Per stage: the server reports no memory of its own (``/health`` has none), so :class:`PeakSampler`
keeps timestamped samples of both, and :func:`stage_peaks` attributes them to the stages of the
command through the stage time windows the command records (``core.timing``: ``t0_unix``,
``stage_windows``; both processes read the same wall clock). The client's per-stage peak is the
larger of the command's own in-process peak (``stages_peak_rss_mb``, sampled every 50 ms, exact
when the stage set the process's high-water mark) and the tree samples (which add child processes
such as COLMAP)."""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from bisect import bisect_left, bisect_right
from typing import Any

import psutil

from oh_my_slam.server.lifecycle import pid_alive, read_state


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
_RUSAGE_INFO_V4 = 4


def phys_footprint_gb(pid: int) -> tuple[float, float] | None:
    """(current, lifetime peak) physical footprint of ``pid`` in GB; None where unavailable."""
    if _libproc is None:
        return None
    info = _RUsageInfoV4()
    if _libproc.proc_pid_rusage(int(pid), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return info.ri_phys_footprint / 1e9, info.ri_lifetime_max_phys_footprint / 1e9


def server_pid() -> int | None:
    """The running inference server's pid (from its state file), if it is alive."""
    state = read_state()
    pid = state.get("pid") if state else None
    return pid if isinstance(pid, int) and pid_alive(pid) else None


def tree_rss_mb(pid: int) -> float:
    """Resident set of ``pid`` and all its descendants (e.g. COLMAP), in MB."""
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0.0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            continue
    return total / 1e6


Sample = tuple[float, float, float | None]  # (wall-clock time, client tree MB, server GB)


class PeakSampler:
    """Samples, every ``every`` seconds until stopped, the resident set of a command's process
    tree and the server's physical footprint; keeps the peaks and the timestamped samples."""

    def __init__(self, pid: int, server: int | None, every: float = 0.2) -> None:
        self.pid, self.server, self.every = pid, server, every
        self.client_peak_mb = 0.0
        self.server_peak_gb: float | None = None
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self) -> None:
        t = time.time()
        tree = tree_rss_mb(self.pid)
        fp = phys_footprint_gb(self.server) if self.server is not None else None
        self.samples.append((t, tree, None if fp is None else fp[0]))
        self.client_peak_mb = max(self.client_peak_mb, tree)
        if fp is not None:
            self.server_peak_gb = max(self.server_peak_gb or 0.0, fp[0])

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.every)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def _during(samples: list[Sample], windows: list[tuple[float, float]]) -> list[Sample]:
    """The samples taken within any of ``windows``; a window shorter than the sampling period
    (no sample inside) gets the samples just before and just after it."""
    times = [s[0] for s in samples]
    out: list[Sample] = []
    for a, b in windows:
        i, j = bisect_left(times, a), bisect_right(times, b)
        out += samples[i:j] if j > i else samples[max(i - 1, 0):i] + samples[j:j + 1]
    return out


def stage_peaks(timings: dict[str, Any] | None, samples: list[Sample]
                ) -> dict[str, dict[str, float | None]] | None:
    """Per stage of one run (``timings``: its ``OH_MY_SLAM_TIMINGS`` record): seconds, the
    client's peak resident set (MB) and the server's peak footprint (GB) while it ran; None when
    the command records no timings."""
    if not timings or not timings.get("stages_s"):
        return None
    own = timings.get("stages_peak_rss_mb") or {}
    t0 = timings.get("t0_unix")
    windows: dict[str, list[tuple[float, float]]] = {}
    if isinstance(t0, int | float):
        for name, a, b in timings.get("stage_windows") or []:
            windows.setdefault(name, []).append((t0 + a, t0 + b))
    ordered = sorted(timings["stages_s"], key=lambda k: min(
        (w[0] for w in windows.get(k, [])), default=float("inf")))
    out: dict[str, dict[str, float | None]] = {}
    for name in ordered:
        seen = _during(samples, windows.get(name, []))
        client = max([float(own.get(name) or 0.0), *(s[1] for s in seen)])
        server = [s[2] for s in seen if s[2] is not None]
        out[name] = {"s": round(float(timings["stages_s"][name]), 3),
                     "client_peak_mb": round(client, 1) if client > 0 else None,
                     "server_peak_gb": round(max(server), 2) if server else None}
    return out
