"""Replay one episode against the API as a fresh map, and record every answer and its timing."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rpl_api import Api, ApiError, Part
from rpl_episode import Episode, Event, Frame
from rpl_geometry import Arr, inv_T
from rpl_images import mime

TERMINAL = {"committed", "rejected", "cancelled", "failed"}
# the gate the skill sends (specs/evaluation.md, C6); an empty accept_if checks nothing
GATE = {"max_residual_m": 0.10, "max_gauge_shift_m": 0.05, "max_rejected_share": 0.5}
MAP = "/api/maps/{name}"


@dataclass
class Options:
    wait_s: float = 10.0  # GET /api/jobs/{id}?wait_s= (at most 30)
    probe_interval_s: float = 0.5
    probes: bool = True
    heights: bool = True
    keep: bool = False
    accept_if: dict[str, Any] = field(default_factory=lambda: dict(GATE))  # sent with O9
    grid_cell_m: float = 0.05
    z_band_m: tuple[float, float] = (0.05, 1.8)
    range_scans: int = 5
    job_timeout_s: float = 1800.0
    locate_batch: int = 8
    sigma_pos_m: float = 0.02
    sigma_rot_deg: float = 1.0
    height_sigma_m: float = 0.02


@dataclass
class JobObs:
    set_key: str
    names: list[str]
    t_start: float
    job_id: str = ""
    t_posted: float = 0.0
    eta_queued: float | None = None
    states: list[tuple[float, str]] = field(default_factory=list)
    t_fusing: float | None = None
    t_readable: float | None = None
    t_end: float | None = None
    job: dict[str, Any] = field(default_factory=dict)
    dry: dict[str, Any] | None = None
    dry_s: float | None = None
    error: str | None = None
    versions_seen: list[tuple[float, int]] = field(default_factory=list)  # job latest_version

    def observe(self, t: float, job: dict[str, Any]) -> None:
        self.job = job
        self.states.append((t, str(job["state"])))
        if job.get("latest_version") is not None:
            self.versions_seen.append((t, int(job["latest_version"])))
        if self.t_fusing is None and job["state"] == "fusing":
            self.t_fusing = t
        if self.t_readable is None and (job.get("provisional_version") is not None
                                        or job.get("result_version") is not None):
            self.t_readable = t
        if job["state"] in TERMINAL:
            self.t_end = t


@dataclass
class LocateObs:
    names: list[str]
    response: dict[str, Any]
    seconds: float
    source: str  # plan | fallback


@dataclass
class Record:
    episode: str
    mode: str
    map: str
    capabilities: list[str]
    sent: list[str] = field(default_factory=list)
    jobs: list[JobObs] = field(default_factory=list)
    frames: dict[int, dict[str, Any]] = field(default_factory=dict)
    objects: dict[int, dict[str, Any]] = field(default_factory=dict)
    locates: list[LocateObs] = field(default_factory=list)
    probe_versions: list[tuple[float, int]] = field(default_factory=list)
    consistency: list[bool] = field(default_factory=list)
    busy_locate_s: list[float] = field(default_factory=list)
    idle_locate_s: list[float] = field(default_factory=list)
    busy_analyse_s: list[float] = field(default_factory=list)
    idle_analyse_s: list[float] = field(default_factory=list)
    grid: dict[str, Any] | None = None
    grid_s: float | None = None
    grid_band: tuple[float, float] = (0.05, 1.8)
    objects_s: float | None = None
    analyses: list[tuple[str, list[list[float]], dict[str, Any]]] = field(default_factory=list)
    revert_s: float | None = None
    revert_ok: bool | None = None
    summary: dict[str, Any] | None = None
    versions: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    digests: dict[int, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def digest(frames_body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(frames_body["frames"], sort_keys=True).encode()).hexdigest()


class Replayer:
    def __init__(self, api: Api, episode: Episode, mode: str, capabilities: list[str],
                 map_name: str, options: Options) -> None:
        self.api, self.ep, self.mode, self.opt = api, episode, mode, options
        self.caps = set(capabilities)
        self.rec = Record(episode.id, mode, map_name, sorted(capabilities))
        self.by_name = {f.name: f for f in episode.all_frames()}
        self.path = MAP.format(name=map_name)
        self.anchor_run: str | None = None
        self.probe_frame: Frame | None = None

    # ---- requests -------------------------------------------------------------------------
    def prior_pose(self, f: Frame) -> Arr:
        """The views.json pose in the map frame: the first run's start frame. A later run's
        poses are moved into it through the truth start poses (README "Assumptions")."""
        assert self.anchor_run is not None
        if f.run == self.anchor_run:
            return f.pose
        sp = self.ep.start_poses
        return inv_T(sp[self.anchor_run]) @ sp[f.run] @ f.pose

    def frame_entry(self, f: Frame, part: str) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": f.name, "image": part, "camera_id": f.camera_id}
        if self.mode == "metric":
            entry["prior"] = {"trust": "metric", "T_map_cam": self.prior_pose(f).tolist(),
                              "sigma_pos_m": self.opt.sigma_pos_m,
                              "sigma_rot_deg": self.opt.sigma_rot_deg}
        else:
            entry["prior"] = {"trust": "none"}
            if self.opt.heights and f.height_m is not None:
                entry["height_above_floor_m"] = f.height_m
                entry["height_sigma_m"] = self.opt.height_sigma_m
        return entry

    def body_and_parts(self, frames: list[Frame]) -> tuple[dict[str, Any], list[Part]]:
        parts: list[Part] = []
        entries = []
        for i, f in enumerate(frames):
            data = f.path.read_bytes()
            parts.append((f"f{i}", f.file, data, mime(data)))
            entries.append(self.frame_entry(f, f"f{i}"))
        cams = {f.camera_id: self.ep.cameras[f.camera_id] for f in frames}
        return {"cameras": cams, "frames": entries}, parts

    # ---- reads ----------------------------------------------------------------------------
    def read_frames(self, version: int | None, api: Api | None = None) -> dict[str, Any]:
        params = None if version is None else {"version": version}
        body, _ = (api or self.api).call("GET", self.path + "/frames", "GET frames", "Frames",
                                         params=params)
        v = int(body["map_version"])
        with self.rec.lock:
            self.rec.frames.setdefault(v, body)
        return dict(body)

    def locate(self, frames: list[Frame], api: Api | None = None) -> tuple[dict[str, Any], float]:
        body, parts = self.body_and_parts(frames)
        body["record"] = False
        res, s = (api or self.api).call("POST", self.path + "/locate", "POST locate", "Locate",
                                        body=body, parts=parts)
        return dict(res), s

    def analyse(self, f: Frame, T_map_cam: list[list[float]] | None = None,
                api: Api | None = None) -> tuple[dict[str, Any], float]:
        data = f.path.read_bytes()
        body: dict[str, Any] = {"image": "image", "camera_id": f.camera_id, "map": self.rec.map,
                                "outputs": ["range_scan"], "z_band_m": list(self.opt.z_band_m)}
        if T_map_cam is not None:
            body["T_map_cam"] = T_map_cam
        res, s = (api or self.api).call("POST", "/api/frames/analyse", "POST analyse",
                                        "Analysis", body=body,
                                        parts=[("image", f.file, data, mime(data))])
        return dict(res), s

    # ---- probes while a job runs -------------------------------------------------------------
    def probe_loop(self, stop: threading.Event) -> None:
        api = self.api.clone()
        while not stop.wait(self.opt.probe_interval_s):
            try:
                body = self.read_frames(None, api)
                v = int(body["map_version"])
                with self.rec.lock:
                    self.rec.probe_versions.append((time.monotonic(), v))
                    self.rec.digests.setdefault(v, digest(body))
                if self.probe_frame is not None and "O5" in self.caps:
                    _, s = self.locate([self.probe_frame], api)
                    self.rec.busy_locate_s.append(s)
                if self.probe_frame is not None and "O7" in self.caps:
                    _, s = self.analyse(self.probe_frame, api=api)
                    self.rec.busy_analyse_s.append(s)
            except ApiError as e:
                if e.code != "map_not_found":
                    self.rec.warnings.append(f"probe: {e}")

    def check_consistency(self) -> None:
        """Each version a probe read is read again by number: the same content (C3)."""
        with self.rec.lock:
            pending = dict(self.rec.digests)
            self.rec.digests.clear()
        for v, d in sorted(pending.items()):
            try:
                body, _ = self.api.call("GET", self.path + "/frames", "GET frames", "Frames",
                                        params={"version": v})
            except ApiError as e:
                if e.code != "version_gone":
                    raise
                continue
            self.rec.consistency.append(int(body["map_version"]) == v and digest(body) == d)

    # ---- one update -----------------------------------------------------------------------
    def update(self, event: Event, frames: list[Frame]) -> JobObs:
        body, parts = self.body_and_parts(frames)
        if "O9" in self.caps:
            body["accept_if"] = dict(self.opt.accept_if)
        obs = JobObs(event.set_key, [f.name for f in frames], time.monotonic())
        if "O10" in self.caps:
            obs.dry, obs.dry_s = self.api.call("POST", self.path + "/updates", "POST dry run",
                                               "DryRun", params={"dry_run": 1}, body=body,
                                               parts=parts)
        stop = threading.Event()
        prober = threading.Thread(target=self.probe_loop, args=(stop,), daemon=True)
        if self.opt.probes:
            prober.start()
        job: dict[str, Any] = {}
        try:
            obs.t_start = time.monotonic()
            job, _ = self.api.call("POST", self.path + "/updates", "POST updates", "Job",
                                   body=body, parts=parts, ok=(202,))
            obs.t_posted = time.monotonic()
            obs.job_id, obs.eta_queued = str(job["job_id"]), job.get("eta_s")
            obs.observe(obs.t_posted, job)
            while job["state"] not in TERMINAL:
                if time.monotonic() - obs.t_start > self.opt.job_timeout_s:
                    job, _ = self.api.call("DELETE", f"/api/jobs/{obs.job_id}", "DELETE job",
                                           "Job")
                    obs.error = f"timed out after {self.opt.job_timeout_s:.0f} s; cancelled"
                    obs.observe(time.monotonic(), job)
                    break
                job, _ = self.api.call("GET", f"/api/jobs/{obs.job_id}", "GET job", "Job",
                                       params={"wait_s": min(30.0, self.opt.wait_s)})
                obs.observe(time.monotonic(), job)
        except ApiError as e:
            obs.error = str(e)
            self.rec.warnings.append(f"update {event.set_key}: {e}")
        finally:
            stop.set()
            if prober.is_alive():
                prober.join()
        self.rec.sent.extend(obs.names)
        for key in ("provisional_version", "result_version"):
            if job.get(key) is not None:
                self.read_frames(int(job[key]))
        self.check_consistency()
        result = job.get("result_version")
        if result is not None and "O8" in self.caps:
            objs, _ = self.api.call("GET", self.path + "/objects", "GET objects", "Objects",
                                    params={"version": result})
            self.rec.objects[int(result)] = objs
        self.idle_probes(job)
        return obs

    def idle_probes(self, job: dict[str, Any]) -> None:
        """One-frame locate and analyse while no job runs (C12 idle latency)."""
        good = [i["name"] for i in job.get("inputs") or [] if i["status"] != "rejected"]
        if not good:
            return
        self.probe_frame = self.by_name[good[-1]]
        if "O5" in self.caps:
            _, s = self.locate([self.probe_frame])
            self.rec.idle_locate_s.append(s)
        if "O7" in self.caps:
            _, s = self.analyse(self.probe_frame)
            self.rec.idle_analyse_s.append(s)

    def plan_locate(self, frames: list[Frame], source: str) -> None:
        try:
            res, s = self.locate(frames)
        except ApiError as e:
            if e.code != "map_not_found":
                raise
            self.rec.warnings.append(f"locate before the map exists: {len(frames)} frames")
            return
        self.rec.locates.append(LocateObs([f.name for f in frames], res, s, source))
        v = int(res["map_version"])
        if v not in self.rec.frames:
            self.read_frames(v)

    # ---- the whole replay -----------------------------------------------------------------
    def run(self) -> Record:
        t0 = time.monotonic()
        try:
            self.api.call("DELETE", self.path, "DELETE map", None, ok=(204,))
        except ApiError as e:
            if e.code != "map_not_found":
                raise
        for event in self.ep.events:
            frames = self.ep.frames(event)
            if not frames:
                continue
            if event.op == "update":
                if self.anchor_run is None:
                    self.anchor_run = frames[0].run
                self.rec.jobs.append(self.update(event, frames))
            elif "O5" in self.caps and self.anchor_run is not None:
                self.plan_locate(frames, "plan")
        self.rec.timings_s["replay_s"] = time.monotonic() - t0
        t1 = time.monotonic()
        if self.rec.jobs and any(j.job.get("result_version") for j in self.rec.jobs):
            self.final_reads()
        self.rec.timings_s["final_reads_s"] = time.monotonic() - t1
        if not self.opt.keep:
            try:
                self.api.call("DELETE", self.path, "DELETE map", None, ok=(204,))
            except ApiError as e:
                self.rec.warnings.append(f"delete: {e}")
        self.rec.timings_s["total_s"] = time.monotonic() - t0
        return self.rec

    def final_reads(self) -> None:
        summary, _ = self.api.call("GET", self.path, "GET map", "MapSummary")
        self.rec.summary = summary
        frames = self.read_frames(None)
        keyframes = [r for r in frames["frames"] if r["name"] in self.by_name]
        if "O5" in self.caps and not any(lo.source == "plan" for lo in self.rec.locates):
            kept = {r["name"] for r in keyframes}
            rest = [self.by_name[n] for n in self.rec.sent if n not in kept]
            for i in range(0, len(rest), self.opt.locate_batch):
                self.plan_locate(rest[i:i + self.opt.locate_batch], "fallback")
        if "O6" in self.caps:
            lo, hi = self.opt.z_band_m
            self.rec.grid_band = (lo, hi)
            self.rec.grid, self.rec.grid_s = self.api.call(
                "GET", self.path + "/grid", "GET grid", "Grid",
                params={"cell_m": self.opt.grid_cell_m, "z_min": lo, "z_max": hi})
        if "O8" in self.caps:
            objs, self.rec.objects_s = self.api.call("GET", self.path + "/objects",
                                                     "GET objects", "Objects")
            self.rec.objects[int(objs["map_version"])] = objs
        if "O7" in self.caps and keyframes and self.opt.range_scans > 0:
            pick = np.unique(np.linspace(0, len(keyframes) - 1,
                                         self.opt.range_scans).round().astype(int))
            for i in pick:
                r = keyframes[int(i)]
                res, _ = self.analyse(self.by_name[r["name"]], r["T_map_cam"])
                self.rec.analyses.append((r["name"], r["T_map_cam"], res))
        if "O2" in self.caps:
            self.rec.versions, _ = self.api.call("GET", self.path + "/versions",
                                                 "GET versions", "Versions")
            latest = int(summary["latest_version"])
            if latest >= 2:
                res, self.rec.revert_s = self.api.call(
                    "POST", self.path + "/revert", "POST revert", "MapSummary",
                    body={"to_version": latest - 1})
                self.rec.revert_ok = int(res["map_version"]) == latest + 1
