"""Memory of the measured processes: the command's process tree (resident set) and the inference
server (physical footprint, which on Apple silicon includes Metal/MPS allocations)."""

from __future__ import annotations

import ctypes
import sys
import threading

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


class PeakSampler:
    """Samples, every ``every`` seconds until stopped, the resident set of a command's process
    tree and the server's physical footprint; keeps the peaks."""

    def __init__(self, pid: int, server: int | None, every: float = 0.2) -> None:
        self.pid, self.server, self.every = pid, server, every
        self.client_peak_mb = 0.0
        self.server_peak_gb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self) -> None:
        self.client_peak_mb = max(self.client_peak_mb, tree_rss_mb(self.pid))
        fp = phys_footprint_gb(self.server) if self.server is not None else None
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
