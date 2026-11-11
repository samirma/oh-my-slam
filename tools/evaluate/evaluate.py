#!/usr/bin/env python3
"""Reference evaluator of oh-my-slam (specs/evaluation.md "Reference evaluator"): runs the
reference captures of examples/ through the public HTTP API, camera-only (no priors), in one job
and split into several, and optionally replays recorded request sets (--requests), for regression
and latency without truth.

A client-side tool: Python 3.11+ standard library, numpy and requests (jsonschema when present).
It imports nothing from the service; it shares the replay evaluator's HTTP client, schema check
and targets reader (tools/replay_eval/). Accuracy against truth comes only from the replay
evaluator: the metrics the captures cannot measure are reported ``not_measured`` with the reason.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import itertools
import json
import math
import platform
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "replay_eval"))

import numpy as np
import requests
from rpl_api import Api, ApiError, Log, Part
from rpl_floorplan import poly_distance
from rpl_geometry import umeyama
from rpl_metrics import grid_id_matches, layer
from rpl_targets import DEFAULT_TARGETS, Targets, compare, judge, reduce
from rpl_validate import DEFAULT_SCHEMA, Validator

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
DEFAULT_OUT = Path.home() / "oh-my-slam-data" / "evaluations"
BASELINE = "baseline_reference.json"  # beside the runs; the legacy baseline.json is not this
TERMINAL = {"committed", "rejected", "cancelled", "failed"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SPLITS = ("single", "halves", "interleaved", "per_frame")
MATCH_M = 0.3  # objects of two versions (one map frame) are the same within this distance
PAIR_M = 0.5  # objects of two runs (aligned) are paired within this distance
LOCATE_BATCH = 8

# How each metric is reduced, and why the reference captures cannot measure the others.
REDUCE = {
    "update.readable_latency": "p90", "fusion.time_per_frame": "p90", "reads.atomic": "share",
    "jobs.eta_accuracy": "share", "reads.latency": "max", "locate.latency_idle": "p90",
    "locate.recall": "share", "grid.id_stability": "share", "features.region_set": "share",
    "grid.camera_cells_free": "share", "objects.id_stability": "share",
    "objects.label_agreement": "share", "objects.centre_error": "median",
    "revert.latency": "max",
}
PARTS = {
    "frame.shift": {"pos_m": "median", "rot_deg": "median"},
    "interactive.latency_under_load": {"ratio": "max", "abs_s": "p90"},
    "update.later_wins": {"cup_struck": "share", "ids_kept": "share",
                          "box_centre_delta_m": "median"},
}
BASIS = {
    "locate.recall": "the keyframes' own images (no truth poses)",
    "objects.centre_error": "split runs against the one-update run, aligned on shared keyframes "
                            "(no truth)",
    "objects.label_agreement": "split runs against the one-update run, paired by geometry",
    "objects.id_stability": "present objects at the same place in consecutive versions",
    "update.later_wins": "office_sequence split runs, examples/ground_truth absent objects",
}
TRUTH = "needs simulator truth: tools/replay_eval/replay_eval.py"
NOT_MEASURED = {
    "locate.median_pos_error": TRUTH, "locate.ambiguity_flagged": TRUTH,
    "locate.sigma_right_median": TRUTH, "locate.nearest_view": TRUTH + " (frames with priors)",
    "locate.sigma_containment": TRUTH, "locate.sigma_tightness": TRUTH,
    "update.sigma_containment": TRUTH, "scale.error_metric": TRUTH + " (metric priors)",
    "scale.error_camera_height": TRUTH, "inputs.reason_agreement": TRUTH,
    "scale.error_by_direction": TRUTH + " (truth positions in four directions)",
    "grid.unconfirmed_occupancy": TRUTH + " (floor plan, kept_at_prior frames' views)",
    "gate.warping_rejected": TRUTH + " (warping updates)", "gate.others_accepted": TRUTH,
    "grid.floor_plan_agreement": TRUTH + " (floor plan)",
    "passages.door_recall": TRUTH + " (floor plan)", "passages.geometry": TRUTH,
    "grid.floor_observed": TRUTH + " (floor plan)",
    "grid.floor_false_obstacles": TRUTH + " (floor plan)",
    "grid.band_occupied": TRUTH + " (floor plan)", "grid.floor_height": TRUTH,
    "objects.support_height": TRUTH, "objects.relations": TRUTH,
    "analysis.range_scan_error": TRUTH + " (floor-plan walls)",
    "fusion.intermediate_interval": "needs O3 (intermediate fusing versions)",
}


# ---- the captures -----------------------------------------------------------------------------
@dataclass
class Capture:
    """A reference capture: its folder, its camera (intrinsics when the images carry no EXIF
    focal length) and the frames in capture order."""

    key: str
    folder: Path
    fx: float | None = None  # pixels; None: from EXIF, else hfov_deg
    hfov_deg: float = 60.0
    splits_from_truth: list[list[int]] = field(default_factory=list)  # chunk sizes
    absent: list[str] = field(default_factory=list)  # labels gone by the sequence's end

    def frames(self) -> list[Path]:
        return sorted(p for p in self.folder.iterdir()
                      if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))


def default_captures(examples: Path = EXAMPLES) -> list[Capture]:
    """The three reference captures (evaluation.md): office_sequence (EXIF focal length; its
    absent objects and splits from examples/ground_truth), ainex-captures (640x480, fx 270, README
    "Locate, examples/ainex-captures") and camera (1920x1080, no EXIF: a 110° wide-angle lens,
    taken as a pinhole of that field of view; override with --cameras)."""
    office = Capture("office_sequence", examples / "office_sequence")
    gt = examples / "ground_truth" / "office_sequence.json"
    if gt.is_file():
        data = json.loads(gt.read_text())
        office.splits_from_truth = [list(map(int, s)) for s in data.get("splits", [])]
        office.absent = [str(a["label"]) for a in data.get("absent", [])]
    return [office, Capture("ainex-captures", examples / "ainex-captures", fx=270.0),
            Capture("camera", examples / "camera", hfov_deg=110.0)]


def jpeg_info(data: bytes) -> tuple[int, int, float | None]:
    """(width, height, 35 mm-equivalent focal length or None) of a JPEG, or of a PNG (no
    focal length), from its headers."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h), None
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG or PNG image")
    i, f35, size = 2, None, None
    while i + 4 <= len(data) and size is None:
        if data[i] != 0xFF:
            raise ValueError("malformed JPEG")
        marker = data[i + 1]
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + length]
        if marker == 0xE1 and seg[:6] == b"Exif\x00\x00":
            f35 = exif_focal35(seg[6:])
        elif marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", seg[1:5])
            size = (int(w), int(h))
        i += 2 + length
    if size is None:
        raise ValueError("JPEG without a frame header")
    return size[0], size[1], f35


def exif_focal35(tiff: bytes) -> float | None:
    """EXIF FocalLengthIn35mmFilm (tag 0xA405 of the Exif IFD), or None."""
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return None
    e = "<" if tiff[:2] == b"II" else ">"

    def entries(off: int) -> Iterator[tuple[int, int, int]]:
        if off + 2 > len(tiff):
            return
        n = struct.unpack(e + "H", tiff[off:off + 2])[0]
        for k in range(n):
            p = off + 2 + 12 * k
            if p + 12 > len(tiff):
                return
            tag, typ, _ = struct.unpack(e + "HHI", tiff[p:p + 8])
            val = struct.unpack(e + ("H" if typ == 3 else "I"), tiff[p + 8:p + (10 if typ == 3
                                                                                else 12)])[0]
            yield tag, typ, val

    ifd0 = struct.unpack(e + "I", tiff[4:8])[0]
    exif = next((v for t, _, v in entries(ifd0) if t == 0x8769), None)
    if exif is None:
        return None
    f35 = next((v for t, _, v in entries(exif) if t == 0xA405), None)
    return float(f35) if f35 else None


def camera_of(c: Capture, first: bytes, overrides: dict[str, Any]) -> dict[str, Any]:
    """The API camera of a capture: --cameras override, else its fx, else the EXIF 35 mm focal
    length (fx = f35 / 36 x the long side), else its field of view."""
    if c.key in overrides:
        return dict(overrides[c.key])
    w, h, f35 = jpeg_info(first)
    fx = c.fx
    if fx is None and f35:
        fx = f35 / 36.0 * max(w, h)
    if fx is None:
        fx = w / 2.0 / math.tan(math.radians(c.hfov_deg) / 2.0)
    return {"fx": fx, "fy": fx, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}


def split(n: int, how: str, chunks: list[int] | None = None) -> list[list[int]]:
    """Frame indices per job: ``single`` one job; ``halves`` two; ``per_frame`` one frame per
    job; ``interleaved`` three jobs dealt round-robin (job j holds frames j, j+3, ...), sent in
    that order; ``truth`` the chunk sizes of examples/ground_truth."""
    idx = list(range(n))
    if how == "single":
        return [idx]
    if how == "halves":
        return [idx[:n // 2], idx[n // 2:]] if n > 1 else [idx]
    if how == "per_frame":
        return [[i] for i in idx]
    if how == "interleaved":
        return [s for s in (idx[j::3] for j in range(3)) if s]
    if how == "truth":
        assert chunks is not None
        out, i = [], 0
        for k in chunks:
            out.append(idx[i:i + k])
            i += k
        return [s for s in out if s]
    raise ValueError(f"unknown split {how!r}")


# ---- measurement ------------------------------------------------------------------------------
@dataclass
class Samples:
    values: dict[str, list[float]] = field(default_factory=dict)
    contracts: dict[str, list[float]] = field(default_factory=dict)

    def add(self, key: str, v: float | bool) -> None:
        self.values.setdefault(key, []).append(float(v))

    def check(self, key: str, ok: bool) -> None:
        self.contracts.setdefault(key, []).append(float(ok))

    def merge(self, other: Samples) -> None:
        for k, v in other.values.items():
            self.values.setdefault(k, []).extend(v)
        for k, v in other.contracts.items():
            self.contracts.setdefault(k, []).extend(v)


def digest(frames_body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(frames_body["frames"], sort_keys=True).encode()).hexdigest()


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def frame_shift(a: dict[str, Any], b: dict[str, Any], gauge: dict[str, Any] | None,
                s: Samples) -> None:
    """frame.shift: each keyframe fused in version ``a`` and present in ``b``, beyond the stated
    gauge shift."""
    g = gauge or {"translation_m": 0.0, "rotation_deg": 0.0}
    after = {r["name"]: r for r in b["frames"]}
    for r in a["frames"]:
        if r.get("pose_state", "fused") != "fused" or r["name"] not in after:
            continue
        Ta, Tb = np.asarray(r["T_map_cam"]), np.asarray(after[r["name"]]["T_map_cam"])
        s.add("frame.shift#pos_m", max(0.0, float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]))
                                       - float(g["translation_m"])))
        s.add("frame.shift#rot_deg", max(0.0, angle_deg(Ta[:3, :3], Tb[:3, :3])
                                         - float(g["rotation_deg"])))


def present(objs: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in objs["objects"] if o.get("state") != "struck"]


def object_ids(a: dict[str, Any], b: dict[str, Any], s: Samples) -> None:
    """objects.id_stability: a present object of version ``a`` found at the same place with the
    same label in ``b`` keeps its id."""
    for o in present(a):
        same = [p for p in present(b) if p["label"] == o["label"] and np.linalg.norm(
            np.asarray(p["centre_m"]) - np.asarray(o["centre_m"])) <= MATCH_M]
        if same:
            s.add("objects.id_stability", any(p["id"] == o["id"] for p in same))


def region_set(grid: dict[str, Any], objs: dict[str, Any] | None, s: Samples) -> None:
    cell = float(grid["cell_m"])
    polys = [np.asarray(r["polygon_m"], dtype=float) for r in grid["regions"]]
    for fr in grid["frontiers"]:
        c = np.asarray([fr["centre_m"]], dtype=float)
        if any(float(poly_distance(c, p)[0]) <= 2 * cell for p in polys):
            s.add("features.region_set", fr["region"] is not None)
    for o in present(objs) if objs else []:
        c = np.asarray([o["centre_m"][:2]], dtype=float)
        if any(float(poly_distance(c, p)[0]) <= cell for p in polys):
            s.add("features.region_set", o["region"] is not None)


def camera_cells(grid: dict[str, Any], frames: dict[str, Any], cams: dict[str, Any],
                 s: Samples) -> None:
    """grid.camera_cells_free: each keyframe camera's cell and its near blind disc on the floor
    (from its height above floor_z_m and its lower field-of-view edge) are free."""
    state = layer(grid, "state")
    rows, cols = state.shape
    cell = float(grid["cell_m"])
    ox, oy = (float(v) for v in grid["origin_m"])
    fz = float(grid["floor_z_m"])
    gx, gy = np.meshgrid(ox + (np.arange(cols) + 0.5) * cell, oy + (np.arange(rows) + 0.5) * cell)
    for r in frames["frames"]:
        T = np.asarray(r["T_map_cam"])
        p = T[:3, 3]
        cam = cams.get(r["camera_id"])
        radius = 0.0
        if cam is not None:
            d = T[:3, :3] @ np.array([0.0, (cam["height"] - 1 - cam["cy"]) / cam["fy"], 1.0])
            if d[2] < -1e-6 and (fz - p[2]) / d[2] > 0:
                radius = float((fz - p[2]) / d[2] * math.hypot(d[0], d[1]))
        c, rr = math.floor((p[0] - ox) / cell), math.floor((p[1] - oy) / cell)
        if not (0 <= c < cols and 0 <= rr < rows):
            s.add("grid.camera_cells_free", False)
            continue
        disc = np.hypot(gx - p[0], gy - p[1]) <= max(0.0, radius - cell)
        disc[rr, c] = True
        s.add("grid.camera_cells_free", bool((state[disc] == 1).all()))


def birdseye_agrees(meta: dict[str, Any], objs: dict[str, Any]) -> bool:
    """The bird's-eye contract: every present object's centre falls inside the image through
    the inverse of ``pixel_to_map``."""
    A = np.asarray(meta["pixel_to_map"], dtype=float)
    M = np.vstack([A, [0.0, 0.0, 1.0]])
    try:
        inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return False
    for o in present(objs):
        u, v, _ = inv @ np.array([o["centre_m"][0], o["centre_m"][1], 1.0])
        if not (-0.5 <= u < meta["width"] - 0.5 and -0.5 <= v < meta["height"] - 0.5):
            return False
    return True


def align(a: dict[str, Any], b: dict[str, Any]) -> Any:
    """The similarity from run ``b``'s map frame to run ``a``'s, on their shared keyframes'
    centres, or None with fewer than three."""
    pa = {r["name"]: np.asarray(r["T_map_cam"])[:3, 3] for r in a["frames"]}
    shared = [r for r in b["frames"] if r["name"] in pa]
    if len(shared) < 3:
        return None
    src = np.array([np.asarray(r["T_map_cam"])[:3, 3] for r in shared])
    dst = np.array([pa[r["name"]] for r in shared])
    try:
        return umeyama(src, dst)[0]
    except ValueError:
        return None


def compare_runs(one: dict[str, Any], other: dict[str, Any], s: Samples) -> None:
    """objects.label_agreement and objects.centre_error of a split run against the one-update
    run: objects paired by geometry alone (nearest within PAIR_M after the alignment)."""
    sim = align(one["frames"], other["frames"])
    if sim is None or one["objects"] is None or other["objects"] is None:
        return
    ref = present(one["objects"])
    for o in present(other["objects"]):
        c = sim.point(o["centre_m"])
        d = [(float(np.linalg.norm(np.asarray(p["centre_m"]) - c)), i) for i, p in enumerate(ref)]
        if not d:
            return
        dist, i = min(d)
        if dist <= PAIR_M:
            s.add("objects.label_agreement", ref[i]["label"] == o["label"])
            s.add("objects.centre_error", dist)


def later_wins(versions: list[dict[str, Any]], absent: list[str], s: Samples) -> None:
    """update.later_wins on a split run: the absent objects present after the first job are
    struck or gone at the end; the other objects present then keep their ids and boxes."""
    if len(versions) < 2:
        return
    first, last = versions[0], versions[-1]
    final = {o["id"]: o for o in present(last)}
    for o in present(first):
        if o["label"] in absent:
            s.add("update.later_wins#cup_struck", o["id"] not in final)
            continue
        s.add("update.later_wins#ids_kept", o["id"] in final)
        if o["id"] in final:
            s.add("update.later_wins#box_centre_delta_m", float(np.linalg.norm(
                np.asarray(final[o["id"]]["centre_m"]) - np.asarray(o["centre_m"]))))


# ---- the runs ---------------------------------------------------------------------------------
@dataclass
class Options:
    wait_s: float = 0.5
    job_timeout_s: float = 3600.0
    cell_m: float = 0.05
    z_band_m: tuple[float, float] = (0.05, 1.8)
    eta_tolerance: float = 0.3


class Runner:
    """One capture split into jobs, as a fresh map."""

    def __init__(self, api: Api, caps: set[str], capture: Capture, camera: dict[str, Any],
                 how: str, map_name: str, opt: Options) -> None:
        self.api, self.caps, self.c, self.cam, self.how = api, caps, capture, camera, how
        self.path = f"/api/maps/{map_name}"
        self.opt = opt
        self.s = Samples()
        self.frames: dict[int, dict[str, Any]] = {}
        self.objects: list[dict[str, Any]] = []
        self.grids: dict[int, dict[str, Any]] = {}
        self.idle_locate: list[float] = []
        self.busy_locate: list[float] = []
        self.jobs: list[dict[str, Any]] = []
        self.probe: Path | None = None

    def body(self, paths: list[Path]) -> tuple[dict[str, Any], list[Part]]:
        parts = [(f"f{i}", p.name, p.read_bytes(),
                  "image/png" if p.suffix.lower() == ".png" else "image/jpeg")
                 for i, p in enumerate(paths)]
        return ({"cameras": {"cam": self.cam},
                 "frames": [{"name": p.name, "image": f"f{i}", "camera_id": "cam",
                             "prior": {"trust": "none"}} for i, p in enumerate(paths)]}, parts)

    def locate(self, paths: list[Path]) -> tuple[dict[str, Any], float]:
        body, parts = self.body(paths)
        body["record"] = False
        res, sec = self.api.call("POST", self.path + "/locate", "POST locate", "Locate",
                                 body=body, parts=parts)
        return dict(res), sec / len(paths)

    def read_frames(self, version: int | None) -> dict[str, Any]:
        params = None if version is None else {"version": version}
        body, _ = self.api.call("GET", self.path + "/frames", "GET frames", "Frames",
                                params=params)
        return dict(body)

    def job(self, paths: list[Path], disconnect: bool) -> dict[str, Any]:
        body, parts = self.body(paths)
        dry = None
        if "O10" in self.caps:
            dry, _ = self.api.call("POST", self.path + "/updates", "POST dry run", "DryRun",
                                   params={"dry_run": 1}, body=body, parts=parts)
        if "O5" in self.caps and self.probe is not None:
            _, sec = self.locate([self.probe])
            self.idle_locate.append(sec)
            self.s.add("locate.latency_idle", sec)
        t0 = time.monotonic()
        api = self.api
        if disconnect:  # the job must go on after its client hangs up
            api = self.api.clone()
        job, _ = api.call("POST", self.path + "/updates", "POST updates", "Job", body=body,
                          parts=parts, ok=(202,))
        if disconnect:
            api.session.close()
        t_post = time.monotonic()
        eta, job_id = job.get("eta_s"), str(job["job_id"])
        t_readable, seen = None, {}
        while job["state"] not in TERMINAL:
            if time.monotonic() - t0 > self.opt.job_timeout_s:
                break
            job, _ = self.api.call("GET", f"/api/jobs/{job_id}", "GET job", "Job",
                                   params={"wait_s": self.opt.wait_s})
            if t_readable is None and (job.get("provisional_version") or job.get(
                    "result_version")):
                t_readable = time.monotonic()
            with contextlib.suppress(ApiError):  # the map may not exist yet
                fb = self.read_frames(None)
                seen.setdefault(int(fb["map_version"]), digest(fb))
                if "O5" in self.caps and self.probe is not None and job["state"] not in TERMINAL:
                    _, sec = self.locate([self.probe])
                    self.busy_locate.append(sec)
                    self.s.add("interactive.latency_under_load#abs_s", sec)
        t_end = time.monotonic()
        if t_readable is None and job.get("result_version"):
            t_readable = t_end
        if disconnect:
            self.s.check("contract.job_survives_disconnect", job["state"] in TERMINAL - {"failed"})
        for v, d in seen.items():  # reads atomic: the same version reads the same again
            try:
                again = self.read_frames(v)
            except ApiError as e:
                if e.code != "version_gone":
                    raise
                continue
            ok = int(again["map_version"]) == v and digest(again) == d
            self.s.add("reads.atomic", ok)
            self.s.check("contract.reads_atomic", ok)
        if t_readable is not None:
            self.s.add("update.readable_latency", t_readable - t0)
        kept = [i for i in job.get("inputs") or [] if i["status"] != "rejected"]
        if job["state"] == "committed":
            self.s.add("fusion.time_per_frame", (t_end - t0) / max(1, len(kept)))
            measured = t_end - t_post
            if eta is not None and measured > 0 and self.caps & {"O3", "O3-jobs"}:
                self.s.add("jobs.eta_accuracy",
                           abs(float(eta) - measured) <= self.opt.eta_tolerance * measured)
            if dry is not None:
                pred = float(dry["predicted_s"]["registration"]) + float(
                    dry["predicted_s"]["fusion"])
                self.s.add("jobs.eta_accuracy",
                           abs(pred - (t_end - t0)) <= self.opt.eta_tolerance * (t_end - t0))
        if kept:
            self.probe = next(p for p in paths if p.name == kept[-1]["name"])
        self.jobs.append({"frames": len(paths), "job_id": job_id, "state": job["state"],
                          "result_version": job.get("result_version"),
                          "seconds": round(t_end - t0, 3),
                          "rejected": {i["name"]: i["reason"] for i in job.get("inputs") or []
                                       if i["status"] == "rejected"},
                          "quality": job.get("quality")})
        return dict(job)

    def reads(self, version: int) -> None:
        self.frames[version] = self.read_frames(version)
        objs = None
        if "O8" in self.caps:
            objs, sec = self.api.call("GET", self.path + "/objects", "GET objects", "Objects",
                                      params={"version": version})
            self.s.add("reads.latency", sec)
            self.objects.append(objs)
        if "O6" in self.caps:
            lo, hi = self.opt.z_band_m
            grid, sec = self.api.call("GET", self.path + "/grid", "GET grid", "Grid", params={
                "version": version, "cell_m": self.opt.cell_m, "z_min": lo, "z_max": hi})
            self.s.add("reads.latency", sec)
            self.grids[version] = grid
            region_set(grid, objs, self.s)
            camera_cells(grid, self.frames[version], {"cam": self.cam}, self.s)
        if "O1" in self.caps and objs is not None:
            meta, _ = self.api.call("GET", self.path + "/birdseye", "GET birdseye", "Birdseye",
                                    params={"version": version})
            self.s.check("contract.birdseye_footprints", birdseye_agrees(meta, objs))

    def run(self) -> dict[str, Any]:
        with contextlib.suppress(ApiError):
            self.api.call("DELETE", self.path, "DELETE map", None, ok=(204,))
        paths = self.c.frames()
        chunks = split(len(paths), self.how, self.c.splits_from_truth[0]
                       if self.how == "truth" else None)
        gauges: dict[int, Any] = {}
        for k, idx in enumerate(chunks):
            job = self.job([paths[i] for i in idx], disconnect=k == 0)
            v = job.get("result_version")
            if v is not None:
                gauges[int(v)] = (job.get("quality") or {}).get("gauge_shift")
                self.reads(int(v))
        versions = sorted(self.frames)
        for a, b in itertools.pairwise(versions):
            frame_shift(self.frames[a], self.frames[b], gauges.get(b), self.s)
            if a in self.grids and b in self.grids:
                for ok in grid_id_matches(self.grids[a], self.grids[b]):
                    self.s.add("grid.id_stability", ok)
        for a, b in itertools.pairwise(self.objects):
            object_ids(a, b, self.s)
        if self.idle_locate and self.busy_locate:
            self.s.add("interactive.latency_under_load#ratio",
                       float(np.median(self.busy_locate)) / float(np.median(self.idle_locate)))
        if versions and "O5" in self.caps:
            kf = {r["name"] for r in self.frames[versions[-1]]["frames"]}
            mine = [p for p in paths if p.name in kf][:LOCATE_BATCH]
            if mine:
                res, sec = self.locate(mine)
                self.s.add("locate.latency_idle", sec)
                for r in res["frames"]:
                    self.s.add("locate.recall", r["status"] == "located")
        if "O2" in self.caps and len(versions) >= 2:
            t = time.monotonic()
            res, _ = self.api.call("POST", self.path + "/revert", "POST revert", "MapSummary",
                                   body={"to_version": versions[-2]})
            self.s.add("revert.latency", time.monotonic() - t)
            self.s.check("contract.revert", int(res["map_version"]) > versions[-1])
        if self.how != "single" and self.c.absent:
            later_wins(self.objects, self.c.absent, self.s)
        return {"split": self.how, "jobs": self.jobs, "versions": versions,
                "frames": self.frames[versions[-1]] if versions else None,
                "objects": self.objects[-1] if self.objects else None}

    def finish(self, keep: bool) -> None:
        if not keep:
            with contextlib.suppress(ApiError):
                self.api.call("DELETE", self.path, "DELETE map", None, ok=(204,))


def replay_requests(api: Api, folder: Path, prefix: str, opt: Options, s: Samples
                    ) -> dict[str, Any]:
    """A recorded request set (``requests.jsonl`` and its ``files/``, written by replay_eval
    --record-requests): each request sent again in order, its maps renamed with ``prefix``. A
    job is followed to its end; recorded job reads are skipped (their ids are the recording's).
    Latencies per route, and update.readable_latency / locate.latency_idle samples."""
    lines = [json.loads(x) for x in (folder / "requests.jsonl").read_text().splitlines() if x]
    per: dict[str, list[float]] = {}
    errors = []
    for r in lines:
        if r["route"] in ("GET job", "DELETE job"):
            continue
        path = r["path"].replace("/api/maps/", f"/api/maps/{prefix}", 1)
        parts = None if r["parts"] is None else [
            (p["name"], p["file"], (folder / p["path"]).read_bytes(), p["media"])
            for p in r["parts"]]
        t0 = time.monotonic()
        try:
            body, sec = api.call(r["method"], path, r["route"], r["schema"], params=r["params"],
                                 body=r["body"], parts=parts, ok=tuple(r["ok"]))
        except ApiError as e:
            errors.append(f"{r['route']}: {e}")
            continue
        per.setdefault(r["route"], []).append(sec)
        if r["route"] == "POST updates":
            job = body
            while job["state"] not in TERMINAL and time.monotonic() - t0 < opt.job_timeout_s:
                job, _ = api.call("GET", f"/api/jobs/{job['job_id']}", "GET job", "Job",
                                  params={"wait_s": opt.wait_s})
                if job.get("result_version"):
                    s.add("update.readable_latency", time.monotonic() - t0)
        elif r["route"] == "POST locate":
            s.add("locate.latency_idle", sec / max(1, len(r["body"]["frames"])))
    return {"set": folder.name, "requests": len(lines), "errors": errors[:50],
            "routes": {k: {"n": len(v), "median_s": float(np.median(v)),
                           "p90_s": float(np.percentile(v, 90))} for k, v in per.items()}}


# ---- results ----------------------------------------------------------------------------------
def metrics(s: Samples, targets: Targets, caps: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mid in targets.metrics:
        t = targets.get(mid)
        entry: dict[str, Any] = {"target": t}
        if mid in BASIS:
            entry["basis"] = BASIS[mid]
        value: Any
        if mid in PARTS:
            value = {p: reduce(s.values.get(f"{mid}#{p}", []), how)
                     for p, how in PARTS[mid].items()}
            n = sum(len(s.values.get(f"{mid}#{p}", [])) for p in PARTS[mid])
            if not n:
                value = None
        elif mid in REDUCE:
            value = reduce(s.values.get(mid, []), REDUCE[mid])
            n = len(s.values.get(mid, []))
        else:
            out[mid] = {**entry, "value": None, "n": 0, "status": "not_measured",
                        "reason": NOT_MEASURED.get(mid, "not measured by the reference "
                                                        "evaluator")}
            continue
        status = judge(value, t)
        reason = None
        if value is None:
            status, reason = "not_measured", "no samples (a capability this service lacks: " \
                f"{', '.join(sorted(caps))} served)"
        out[mid] = {**entry, "value": value, "n": n, "status": status}
        if reason:
            out[mid]["reason"] = reason
    return out


def contracts(s: Samples, log: Log) -> dict[str, dict[str, Any]]:
    for c in log.calls:
        if c.schema is not None:
            s.check("contract.schema_valid", not c.schema_errors)
        s.check("contract.server_timing", c.server_timing)
    out = {}
    for k, v in sorted(s.contracts.items()):
        bad = int(len(v) - sum(v))
        out[k] = {"checked": len(v), "violations": bad, "status": "pass" if not bad else "fail"}
    return out


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, dict):
        return ", ".join(f"{k} {fmt(x)}" for k, x in v.items())
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def summary(result: dict[str, Any]) -> str:
    lines = [f"# Reference evaluation {result['started_at']}", "",
             f"Service {result['url']} ({result['health']['version']}), capabilities "
             f"{', '.join(result['health']['capabilities'])}.", "",
             "| Metric | Value | Target | n | Status | Baseline |", "| --- | --- | --- | --- | --- | --- |"]
    for mid, m in result["metrics"].items():
        t = m["target"] or {}
        target = "info" if t.get("info") else (
            " and ".join(f"{k} {p['op']} {p['value']}" for k, p in t["parts"].items())
            if "parts" in t else f"{t.get('op', '')} {t.get('value', '')}".strip())
        status = m["status"] + (f" ({m['reason']})" if m.get("reason") else "")
        base = m.get("baseline")
        cmp = "-" if base is None else fmt(base["baseline"]) + (
            " **regressed**" if base["regressed"] else "")
        lines.append(f"| `{mid}` | {fmt(m['value'])} | {target} | {m['n']} | {status} | {cmp} |")
    lines += ["", "| Contract | Checked | Violations |", "| --- | --- | --- |"]
    lines += [f"| {k} | {c['checked']} | {c['violations']} |"
              for k, c in result["contracts"].items()]
    if result["errors"]:
        lines += ["", "Errors:", ""] + [f"* {e}" for e in result["errors"]]
    return "\n".join(lines) + "\n"


# ---- the service --------------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def local_service(server_sh: Path, timeout_s: float = 900.0) -> Iterator[str]:
    """A service of this checkout on a free loopback port with a temporary data folder, stopped
    afterwards (``server.sh --stop`` stops the inference server only if it started it)."""
    data = Path(tempfile.mkdtemp(prefix="oms-eval-"))
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([str(server_sh), "--host", "127.0.0.1", "--port", str(port),
                             "--data", str(data)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"evaluate: {server_sh} exited with {proc.returncode}")
            try:
                if requests.get(url + "/api/health", timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise SystemExit(f"evaluate: no service at {url} after {timeout_s:.0f} s")
            time.sleep(1.0)
        yield url
    finally:
        subprocess.run([str(server_sh), "--stop", "--data", str(data)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.poll() is None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(30)
        shutil.rmtree(data, ignore_errors=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="evaluate.py", description=__doc__.splitlines()[0])
    p.add_argument("--url", help="a running service; default: start one on a free port")
    p.add_argument("--out", type=Path, help="output folder (default "
                   "~/oh-my-slam-data/evaluations/<UTC>/)")
    p.add_argument("--baseline", type=Path, help="a stored result.json to compare with "
                   "(default ~/oh-my-slam-data/evaluations/baseline_reference.json when present)")
    p.add_argument("--set-baseline", action="store_true",
                   help="store this run's result.json as the baseline")
    p.add_argument("--requests", type=Path, help="a folder of recorded request sets "
                   "(replay_eval.py --record-requests)")
    p.add_argument("--captures", default="office_sequence,ainex-captures,camera",
                   help="comma-separated captures of examples/ to run ('' for none)")
    p.add_argument("--splits", default=",".join((*SPLITS, "truth")),
                   help="single, halves, interleaved, per_frame, truth (office_sequence's "
                   "ground-truth splits)")
    p.add_argument("--cameras", type=Path, help="JSON {capture: Camera} overriding intrinsics")
    p.add_argument("--examples", type=Path, default=EXAMPLES)
    p.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    p.add_argument("--keep", action="store_true", help="keep the maps (default: delete)")
    p.add_argument("--wait-s", type=float, default=0.5)
    p.add_argument("--timeout-s", type=float, default=600.0, help="per HTTP request")
    p.add_argument("--job-timeout-s", type=float, default=3600.0)
    p.add_argument("--server-sh", type=Path, default=ROOT / "server.sh")
    return p.parse_args(argv)


def evaluate(args: argparse.Namespace, url: str) -> dict[str, Any]:
    targets = Targets.load(args.targets)
    validator = Validator(args.schema)
    log = Log()
    api = Api(url, validator, log, timeout_s=args.timeout_s)
    health, _ = api.call("GET", "/api/health", "GET health", "Health")
    caps = set(health["capabilities"])
    opt = Options(wait_s=args.wait_s, job_timeout_s=args.job_timeout_s,
                  eta_tolerance=targets.param("jobs.eta_accuracy", "time_tolerance_rel", 0.3))
    overrides = json.loads(args.cameras.read_text()) if args.cameras else {}
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    wanted = [c for c in args.captures.split(",") if c]
    splits = [h for h in args.splits.split(",") if h]
    unknown = sorted(set(splits) - {*SPLITS, "truth"})
    if unknown:
        raise SystemExit(f"evaluate: unknown split {', '.join(unknown)}")
    total, runs, errors = Samples(), {}, []
    for cap in default_captures(args.examples):
        if cap.key not in wanted:
            continue
        try:
            paths = cap.frames()
            cam = camera_of(cap, paths[0].read_bytes(), overrides)
        except (OSError, ValueError, IndexError) as e:
            errors.append(f"{cap.key}: {type(e).__name__}: {e}")
            continue
        results: dict[str, Any] = {}
        for how in splits:
            if how == "truth" and not cap.splits_from_truth:
                continue
            print(f"evaluate: {cap.key} {how} ...", file=sys.stderr, flush=True)
            r = Runner(api, caps, cap, cam, how, f"eval-{stamp}-{cap.key}-{how}", opt)
            try:
                results[how] = r.run()
            except ApiError as e:
                errors.append(f"{cap.key} {how}: {e}")
            finally:
                r.finish(args.keep)
            total.merge(r.s)
        one = results.get("single")
        for how, res in results.items():
            if one is not None and how != "single":
                compare_runs(one, res, total)
        runs[cap.key] = {"camera": cam, "splits": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("frames", "objects")}
            for k, v in results.items()}}
    request_sets = []
    if args.requests is not None:
        sets = [args.requests] if (args.requests / "requests.jsonl").is_file() else sorted(
            p for p in args.requests.iterdir() if (p / "requests.jsonl").is_file())
        for k, folder in enumerate(sets):
            request_sets.append(replay_requests(api, folder, f"req{stamp}-{k}-", opt, total))
    health_end, _ = api.call("GET", "/api/health", "GET health", "Health")
    return {"tool": "evaluate", "format": 1, "url": url, "targets": str(args.targets),
            "health": health, "health_end": health_end,
            "client": {"host": platform.node(), "python": platform.python_version(),
                       "jsonschema": validator.full},
            "metrics": metrics(total, targets, caps), "contracts": contracts(total, log),
            "runs": runs, "requests": request_sets, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = dt.datetime.now(dt.UTC)
    out = args.out or DEFAULT_OUT / started.strftime("%Y%m%dT%H%M%SZ")
    if args.url:
        result = evaluate(args, args.url.rstrip("/"))
    else:
        with local_service(args.server_sh) as url:
            result = evaluate(args, url)
    result = {"started_at": started.isoformat(), "finished_at": dt.datetime.now(
        dt.UTC).isoformat(), **result}
    baseline_path = args.baseline or DEFAULT_OUT / BASELINE
    if baseline_path.is_file():
        base = json.loads(baseline_path.read_text())
        if isinstance(base, dict) and isinstance(base.get("metrics"), dict):
            cmp = compare(result["metrics"], base["metrics"], Targets.load(args.targets))
            for mid, c in cmp.items():
                result["metrics"][mid]["baseline"] = c
            result["baseline"] = str(baseline_path)
        else:
            result["errors"].append(f"baseline {baseline_path}: not a result.json of this "
                                    "evaluator; not compared")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    (out / "summary.md").write_text(summary(result))
    if args.set_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(result, indent=1, default=str))
    print(summary(result))
    print(f"evaluate: wrote {out}", file=sys.stderr)
    failed = any(m["status"] == "fail" for m in result["metrics"].values()) or any(
        c["status"] == "fail" for c in result["contracts"].values())
    return 1 if failed or result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
