"""A fake oh-my-slam service (stdlib http.server on a free loopback port) whose answers are
schema-valid and derived from the synthetic episode's truth through a known map -> world
similarity, so the evaluator's metrics have known values."""

from __future__ import annotations

import base64
import email.parser
import email.policy
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .synth import DOOR_Y, house

ALL_CAPS = [f"O{i}" for i in range(1, 11)]
NOW = "2026-11-09T10:00:00Z"


@dataclass
class Sim:
    """world = s R map + t, R a yaw."""

    s: float = 1.0
    yaw_deg: float = 0.0
    t: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def R(self) -> np.ndarray:
        a = math.radians(self.yaw_deg)
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0],
                         [0, 0, 1.0]])

    def to_map(self, p: Any) -> np.ndarray:
        return self.R.T @ (np.asarray(p, float) - np.asarray(self.t)) / self.s

    def to_world(self, p: Any) -> np.ndarray:
        return self.s * self.R @ np.asarray(p, float) + np.asarray(self.t)

    def pose_to_map(self, T: np.ndarray) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = self.R.T @ T[:3, :3]
        out[:3, 3] = self.to_map(T[:3, 3])
        return out


@dataclass
class Config:
    truth: dict[str, np.ndarray] = field(default_factory=dict)
    metric_sim: Sim = field(default_factory=Sim)  # map frame of a metric-prior map
    none_sim: Sim = field(default_factory=lambda: Sim(1.05, 30.0, (0.5, 0.2, 0.0)))
    capabilities: list[str] = field(default_factory=lambda: list(ALL_CAPS))
    reject: dict[str, str] = field(default_factory=dict)  # frame name -> reason
    locate_reject: set[str] = field(default_factory=set)  # rejected no_overlap
    locate_offset: dict[str, float] = field(default_factory=dict)  # name -> x error (world m)
    ambiguous: set[str] = field(default_factory=set)
    gate_reject: set[str] = field(default_factory=set)  # jobs holding these frames are rejected
    durations: tuple[float, float, float] = (0.02, 0.05, 0.12)  # queued, registering, fusing
    gauge_shift: float = 0.0
    move_on_update: float = 0.0  # stored keyframes move this much (map m) at each update
    invalid_health: bool = False
    no_server_timing: bool = False
    fail_route: str | None = None  # answer 500 internal on this path suffix
    busy_once: bool = False  # the first revert answers map_busy (retryable)
    eta: float | None = 0.19
    floor_world_z: float = 0.0  # the world z of the plane the grid reports as floor_z_m


@dataclass
class Job:
    id: str
    map: str
    names: list[str]
    inputs: list[dict[str, Any]]
    created: float
    gated: bool
    state: str = "queued"
    provisional: int | None = None
    result: int | None = None
    cancelled: bool = False


@dataclass
class MapState:
    sim: Sim
    cameras: dict[str, Any] = field(default_factory=dict)
    versions: list[dict[str, Any]] = field(default_factory=list)  # each: frames list, kind, ...
    frames: dict[str, dict[str, Any]] = field(default_factory=dict)


class Fake:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.maps: dict[str, MapState] = {}
        self.pending_sim: dict[str, Sim] = {}
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.requests: list[tuple[str, str]] = []
        self.revert_calls = 0
        self.accept_ifs: list[Any] = []
        self.priors: list[Any] = []
        self.locate_priors: list[Any] = []
        self.depths: list[str] = []
        self.heights: list[float] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> Fake:
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()

    # ---- the map model ----------------------------------------------------------------------
    def commit(self, name: str, kind: str, job: Job | None) -> int:
        m = self.maps[name]
        v = len(m.versions) + 1
        m.versions.append({"version": v, "kind": kind, "job_id": job.id if job else None,
                           "frames": [dict(r) for r in m.frames.values()]})
        return v

    def advance(self, job: Job) -> None:
        """Move a job through its states by elapsed time, committing its versions."""
        if job.state in ("committed", "rejected", "cancelled", "failed"):
            return
        q, r, f = self.cfg.durations
        age = time.monotonic() - job.created
        if age >= q and job.state == "queued":
            job.state = "registering"
        if age >= q + r and job.state == "registering":
            if job.gated and any(n in self.cfg.gate_reject for n in job.names):
                job.state = "rejected"
                return
            if job.map not in self.maps:
                self.maps[job.map] = MapState(self.pending_sim.pop(job.map))
            m = self.maps[job.map]
            for row in m.frames.values():
                T = np.asarray(row["T_map_cam"])
                T[0, 3] += self.cfg.move_on_update
                row["T_map_cam"] = T.tolist()
            for i in job.inputs:
                if i["status"] != "rejected":
                    m.frames[i["name"]] = {
                        "name": i["name"], "camera_id": i["camera_id"],
                        "T_map_cam": i["T_map_cam"], "sigma_pos_m": 0.01, "sigma_rot_deg": 0.5,
                        "pose_state": "provisional", "added_version": len(m.versions) + 1}
            job.provisional = self.commit(job.map, "provisional", job)
            job.state = "fusing"
        if age >= q + r + f and job.state == "fusing":
            m = self.maps[job.map]
            for row in m.frames.values():
                row["pose_state"] = "fused"
            job.result = self.commit(job.map, "update", job)
            job.state = "committed"

    def advance_all(self) -> None:
        for j in self.jobs.values():
            self.advance(j)

    def next_change(self, job: Job) -> float:
        q, r, f = self.cfg.durations
        age = time.monotonic() - job.created
        for edge in (q, q + r, q + r + f):
            if age < edge:
                return edge - age
        return 0.0

    def quality(self, job: Job) -> dict[str, Any]:
        rejected = sum(1 for i in job.inputs if i["status"] == "rejected")
        res = {"median_m": 0.01, "max_m": 0.02, "median_deg": 0.1, "max_deg": 0.2, "count": 3}
        gated = job.gated and any(n in self.cfg.gate_reject for n in job.names)
        return {"scale_ratio_vs_prev": 1.0,
                "gauge_shift": {"translation_m": self.cfg.gauge_shift, "rotation_deg": 0.0,
                                "scale_ratio": 1.0},
                "prior_residuals": res, "previous_residuals": None,
                "rejected_share": rejected / max(1, len(job.inputs)),
                "low_confidence_share": 0.0, "rebuilt": False, "prior_gravity_deg": 0.5,
                "max_displacement_m": 0.01, "max_displacement_deg": 0.1,
                "prior_disagreement_deg": 0.0, "reregistered": [], "height_inconsistent": [],
                "rotation_pairs": [],
                "accepted": not gated, "failed_checks": ["max_residual_m"] if gated else []}

    def job_body(self, job: Job) -> dict[str, Any]:
        registered = job.state not in ("queued",)
        return {"schema": "oms/2.3", "job_id": job.id, "map": job.map, "state": job.state,
                "progress": 1.0 if job.state == "committed" else 0.5,
                "eta_s": self.cfg.eta, "created_at": NOW, "base_version": None,
                "provisional_version": job.provisional, "result_version": job.result,
                "inputs": [{k: v for k, v in i.items() if k != "camera_id"}
                           for i in job.inputs] if registered else None,
                "quality": self.quality(job) if registered else None,
                "scale_source": "priors", "scale_sigma_rel": 0.01,
                "timings_s": {"registration": 0.05}, "error": None,
                "latest_version": job.result or job.provisional,
                "ended_at": NOW if job.state in ("committed", "rejected", "cancelled") else None}

    # ---- answers ----------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        stat = {"median_s": 0.1, "p90_s": 0.2, "count": 0, "source": "default"}
        body: dict[str, Any] = {
            "schema": "oms/2.3", "version": "fake-1", "service": "ready",
            "inference": {"state": "ready", "device": "mps", "memory_used_gb": 1.0,
                          "memory_cap_gb": 10.0, "error": None},
            "lanes": {"worker": {"state": "idle", "queued": 0, "oldest_s": None}},
            "capabilities": self.cfg.capabilities,
            "timings": {k: dict(stat) for k in (
                "registration_per_frame", "fusion_per_frame", "locate_per_frame",
                "analyse_per_frame", "grid_read", "objects_read")}}
        if self.cfg.invalid_health:
            del body["lanes"]
        return body

    def version_of(self, name: str, query: dict[str, list[str]]) -> dict[str, Any]:
        m = self.maps[name]
        if "version" in query:
            v = int(query["version"][0])
            if v < 1 or v > len(m.versions):
                raise FakeError(410, "version_gone", "version gone")
            return m.versions[v - 1]
        return m.versions[-1]

    def scale(self, name: str) -> dict[str, Any]:
        return {"scale_source": "priors", "scale_sigma_rel": 0.01}

    def frames(self, name: str, q: dict[str, list[str]]) -> dict[str, Any]:
        v = self.version_of(name, q)
        return {"schema": "oms/2.3", "map": name, "map_version": v["version"],
                **self.scale(name), "frames": v["frames"]}

    def summary(self, name: str, v: dict[str, Any] | None = None) -> dict[str, Any]:
        m = self.maps[name]
        v = v or m.versions[-1]
        return {"schema": "oms/2.3", "map": name, "map_version": v["version"],
                "latest_version": len(m.versions), "created_at": NOW,
                "frames": len(v["frames"]), "provisional_frames": 0, "objects": 4,
                "cameras": m.cameras, "bounds_m": {"min": [0, 0, 0], "max": [1, 1, 1]},
                "floor_z_m": 0.0, "running_job": None, "labels": [], **self.scale(name)}

    def world_floor_map_z(self, m: MapState, x: float, y: float) -> float:
        return float(m.sim.to_map([x, y, 0.0])[2])

    def grid(self, name: str, q: dict[str, list[str]]) -> dict[str, Any]:
        m = self.maps[name]
        v = self.version_of(name, q)
        cell = float(q.get("cell_m", ["0.05"])[0])
        band = [float(q.get("z_min", ["0.05"])[0]), float(q.get("z_max", ["1.8"])[0])]
        corners = np.array([m.sim.to_map([x, y, 0]) for x in (-0.5, 7.5) for y in (-0.5, 3.5)])
        lo, hi = corners.min(axis=0)[:2], corners.max(axis=0)[:2]
        cols, rows = (np.ceil((hi - lo) / cell)).astype(int)
        state = np.zeros((rows, cols), np.uint8)
        observed = np.zeros((rows, cols), np.uint16)
        floor = np.zeros((rows, cols), np.uint8)
        for r in range(rows):
            for c in range(cols):
                wx, wy, _ = m.sim.to_world([lo[0] + (c + 0.5) * cell, lo[1] + (r + 0.5) * cell,
                                            0.0])
                if not (-0.05 <= wx <= 7.05 and -0.05 <= wy <= 3.05):
                    continue
                observed[r, c] = 3
                edge = min(abs(wx), abs(wx - 7), abs(wy), abs(wy - 3)) <= 0.08
                inner = abs(wx - 4) <= 0.08 and not DOOR_Y[0] <= wy <= DOOR_Y[1]
                state[r, c] = 2 if edge or inner else 1
                floor[r, c] = 1 if state[r, c] == 1 else 0
        door = m.sim.to_map([4.0, sum(DOOR_Y) / 2, 0.0])
        ja = m.sim.to_map([4.0, DOOR_Y[0], 0.0])
        jb = m.sim.to_map([4.0, DOOR_Y[1], 0.0])
        rooms = [[(0, 0), (4, 0), (4, 3), (0, 3)], [(4, 0), (7, 0), (7, 3), (4, 3)]]
        regions = [{"id": i + 1, "area_m2": 12.0, "coverage_share": 0.9,
                    "polygon_m": [m.sim.to_map([x, y, 0])[:2].tolist() for x, y in poly]}
                   for i, poly in enumerate(rooms)]

        def b64(a: np.ndarray) -> dict[str, Any]:
            return {"dtype": str(a.dtype), "shape": list(a.shape),
                    "data": base64.b64encode(a.astype(a.dtype.newbyteorder("<")).tobytes())
                    .decode()}

        return {"schema": "oms/2.3", "map": name, "map_version": v["version"],
                "origin_m": lo.tolist(), "cell_m": cell, "shape": [int(rows), int(cols)],
                "z_band_m": band, "floor_z_m": float(m.sim.to_map([0.0, 0.0, self.cfg.floor_world_z])[2]),
                "floor_sigma_m": 0.01, "height_cap_m": float(band[1]),
                **self.scale(name),
                "layers": {"state": b64(state), "observations": b64(observed),
                           "height_m": b64(np.full((rows, cols), np.nan, np.float32)),
                           "floor_observed": b64(floor)},
                "regions": regions,
                "passages": [{"id": 1, "centre_m": door[:2].tolist(),
                              "width_m": (DOOR_Y[1] - DOOR_Y[0]) / m.sim.s,
                              "normal_deg": 0.0, "regions": [1, 2], "kind": "opening",
                              "jambs_m": [ja[:2].tolist(), jb[:2].tolist()], "leaf": None,
                              "leaves": [], "narrow": None, "frontier": None}],
                "frontiers": [{"id": 1, "centre_m": m.sim.to_map([1.0, 1.0, 0])[:2].tolist(),
                               "length_m": 1.0, "region": 1},
                              {"id": 2, "centre_m": [1e3, 1e3], "length_m": 1.0,
                               "region": None}]}

    def objects(self, name: str, q: dict[str, list[str]]) -> dict[str, Any]:
        m = self.maps[name]
        v = self.version_of(name, q)
        floor_h = self.world_floor_map_z(m, 2.0, 2.5)
        truth = {"DiningTable": (1, [2.0, 2.5, 0.4]), "Mug": (2, [2.0, 2.5, 0.85]),
                 "Bowl": (3, [2.3, 2.5, 0.83]), "Apple": (4, [2.3, 2.5, 0.86]),
                 "Sofa": (5, [6.0, 2.5, 0.4]), "Chair": (6, [5.0, 0.5, 0.4])}
        support = {"DiningTable": {"kind": "surface", "id": 1}, "Mug": {"kind": "object", "id": 1},
                   "Bowl": {"kind": "surface", "id": 2}, "Apple": None,
                   "Sofa": {"kind": "surface", "id": 1}, "Chair": None}
        out = []
        for label, (oid, w) in truth.items():
            c = m.sim.to_map(w)
            sup = support[label]
            out.append({
                "id": oid, "label": {"Mug": "cup", "DiningTable": "dining table"}.get(label, label),
                "detected_as": [label], "score": 0.9,
                "centre_m": c.tolist(), "size_m": [0.5, 0.4, 0.3], "yaw_deg": 0.0,
                "sigma_centre_m": 0.02, "extent_seen": {"x": "full", "y": "full", "z": "partial"},
                "mean_rgb": [10, 20, 30], "color": [1, 2, 3], "state": "confirmed",
                "detection_votes": [{"label": label, "share": 1.0}],
                "moved_from_m": None, "first_seen_version": 1, "last_seen_version": v["version"],
                "region": 1 if w[0] < 4 else 2,
                "supported_by": None if sup is None else {**sup, "verified_version": 1},
                "inside_of": {"id": 3, "verified_version": 1} if label == "Apple" else None})
        out.append({**out[0], "id": 9, "state": "struck"})
        surfaces = [{"id": 1, "height_m": floor_h, "area_m2": 20.0,
                     "polygon_m": [[0, 0], [1, 0], [1, 1]]},
                    {"id": 2, "height_m": floor_h + 0.8 / m.sim.s, "area_m2": 1.0,
                     "polygon_m": [[0, 0], [1, 0], [1, 1]]}]
        return {"schema": "oms/2.3", "map": name, "map_version": v["version"],
                **self.scale(name), "objects": out, "surfaces": surfaces}

    def locate(self, name: str, req: dict[str, Any]) -> dict[str, Any]:
        self.locate_priors.extend(f.get("prior") for f in req["frames"])
        m = self.maps[name]
        v = m.versions[-1]
        out = []
        kf = v["frames"][0] if v["frames"] else None
        for f in req["frames"]:
            n = f["name"]
            T = self.cfg.truth[n]
            base = {"name": n, "T_map_cam": None, "sigma_pos_m": None, "sigma_rot_deg": None,
                    "inlier_ratio": None, "ambiguity": None, "rivals": [],
                    "consistent_with_prior": None, "height_consistent": None}
            if n in self.cfg.locate_reject:
                nv = None
                if kf is not None:
                    kt = self.cfg.truth[kf["name"]]
                    turn = (math.degrees(math.atan2(kt[1, 2], kt[0, 2]) -
                                         math.atan2(T[1, 2], T[0, 2])) + 180) % 360 - 180
                    prior = f.get("prior", {}).get("trust", "none") != "none"
                    nv = {"name": kf["name"], "T_map_cam": kf["T_map_cam"],
                          "turn_deg": turn if prior else None}
                out.append({**base, "status": "rejected", "reason": "no_overlap",
                            "nearest_view": nv})
                continue
            Tw = T.copy()
            Tw[0, 3] += self.cfg.locate_offset.get(n, 0.0)
            amb = {"pos_m": 0.5, "rot_deg": 90.0} if n in self.cfg.ambiguous else None
            out.append({**base, "status": "located", "reason": None,
                        "rivals": [] if amb is None else [{**amb, "support": 0.5}],
                        "T_map_cam": m.sim.pose_to_map(Tw).tolist(), "sigma_pos_m": 0.01,
                        "sigma_rot_deg": 0.5, "inlier_ratio": 0.8, "ambiguity": amb,
                        "consistent_with_prior": True if "prior" in f else None,
                        "height_consistent": None,
                        "nearest_view": None})
        return {"schema": "oms/2.3", "map": name, "map_version": v["version"],
                **self.scale(name), "frames": out}

    def analyse(self, req: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"schema": "oms/2.3", "scale_source": "priors",
                                "scale_sigma_rel": 0.01}
        if "T_map_cam" not in req:
            return {**body, "frame": "gravity_camera", "range_scan": []}
        m = self.maps[req["map"]]
        T = np.asarray(req["T_map_cam"])
        p = m.sim.to_world(T[:3, 3])
        beams = []
        for k in range(-2, 3):
            b_map = math.degrees(math.atan2(T[1, 2], T[0, 2])) + 10 * k
            b = math.radians(b_map + m.sim.yaw_deg)
            d = np.array([math.cos(b), math.sin(b)])
            hits = []
            for axis, edge in ((0, 0.0), (0, 7.0), (1, 0.0), (1, 3.0), (0, 4.0)):
                if abs(d[axis]) > 1e-9:
                    t = (edge - p[axis]) / d[axis]
                    y = p[1 - axis] + t * d[1 - axis]
                    door = axis == 0 and edge == 4.0 and DOOR_Y[0] <= y <= DOOR_Y[1]
                    if t > 1e-6 and not door:
                        hits.append(t)
            beams.append({"bearing_deg": b_map, "range_m": min(hits) / m.sim.s,
                          "kind": "obstacle"})
        beams.append({"bearing_deg": 0.0, "range_m": None, "kind": "unseen"})
        return {**body, "frame": "map", "range_scan": beams}

    # ---- HTTP -------------------------------------------------------------------------------
    def handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def send(self, status: int, body: Any = None) -> None:
                data = b"" if body is None else json.dumps(body).encode()
                self.send_response(status)
                if body is not None:
                    self.send_header("Content-Type", "application/json")
                if not fake.cfg.no_server_timing:
                    self.send_header("Server-Timing", "total;dur=1")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def body(self) -> tuple[dict[str, Any], dict[str, bytes]]:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                ctype = self.headers.get("Content-Type", "")
                if not ctype.startswith("multipart/"):
                    return (json.loads(raw) if raw else {}), {}
                msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                    b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
                parts: dict[str, bytes] = {}
                req: dict[str, Any] = {}
                for part in msg.iter_parts():
                    pname = part.get_param("name", header="content-disposition")
                    data = part.get_payload(decode=True)
                    if pname == "request":
                        req = json.loads(data)
                    else:
                        parts[str(pname)] = data
                return req, parts

            def handle_any(self, method: str) -> None:
                url = urlparse(self.path)
                q = parse_qs(url.query)
                path = url.path
                fake.requests.append((method, path))
                try:
                    with fake.lock:
                        fake.advance_all()
                    if fake.cfg.fail_route and path.endswith(fake.cfg.fail_route):
                        raise FakeError(500, "internal", "boom")
                    status, body = fake.route(method, path, q, self)
                    self.send(status, body)
                except FakeError as e:
                    self.send(e.status, {"schema": "oms/2.3", "error": {
                        "code": e.code, "message": e.message, "retryable": e.retryable,
                        "details": {}}})

            def do_GET(self) -> None:
                self.handle_any("GET")

            def do_POST(self) -> None:
                self.handle_any("POST")

            def do_DELETE(self) -> None:
                self.handle_any("DELETE")

        return H

    def route(self, method: str, path: str, q: dict[str, list[str]],
              h: Any) -> tuple[int, Any]:
        parts = path.strip("/").split("/")
        if path == "/api/health":
            return 200, self.health()
        if path == "/api/frames/analyse":
            req, _ = h.body()
            with self.lock:
                if req.get("map") not in self.maps:
                    raise FakeError(404, "map_not_found", "no map")
                return 200, self.analyse(req)
        if parts[:2] == ["api", "jobs"]:
            with self.lock:
                job = self.jobs.get(parts[2])
                if job is None:
                    raise FakeError(404, "job_not_found", "no job")
                if method == "DELETE":
                    job.state = "cancelled"
                    return 200, self.job_body(job)
            wait = min(30.0, float(q.get("wait_s", ["0"])[0]))
            time.sleep(min(wait, self.next_change(job)))
            with self.lock:
                self.advance(job)
                return 200, self.job_body(job)
        name = parts[2]
        tail = parts[3] if len(parts) > 3 else ""
        if tail == "updates":
            req, files = h.body()
            return self.update(name, req, files, "dry_run" in q)
        with self.lock:
            if name not in self.maps:
                raise FakeError(404, "map_not_found", f"no map {name}")
            m = self.maps[name]
            if method == "DELETE":
                del self.maps[name]
                return 204, None
            if tail == "":
                return 200, self.summary(name)
            if tail == "frames":
                return 200, self.frames(name, q)
            if tail == "grid":
                return 200, self.grid(name, q)
            if tail == "objects":
                return 200, self.objects(name, q)
            if tail == "birdseye":
                v = self.version_of(name, q)
                return 200, {"schema": "oms/2.3", "map": name, "map_version": v["version"],
                             "width": 200, "height": 200,
                             "pixel_to_map": [[0.1, 0.0, -10.0], [0.0, -0.1, 10.0]],
                             **self.scale(name)}
            if tail == "versions":
                return 200, {"schema": "oms/2.3", "map": name,
                             "latest_version": len(m.versions),
                             "versions": [{"version": v["version"], "created_at": NOW,
                                           "kind": v["kind"], "job_id": v["job_id"],
                                           "frames": len(v["frames"]), "objects": 0}
                                          for v in m.versions]}
        if tail == "revert":
            req, _ = h.body()
            with self.lock:
                self.revert_calls += 1
                if self.cfg.busy_once and self.revert_calls == 1:
                    raise FakeError(409, "map_busy", "busy", True)
                old = m.versions[int(req["to_version"]) - 1]
                m.frames = {r["name"]: dict(r) for r in old["frames"]}
                self.commit(name, "revert", None)
                return 200, self.summary(name)
        if tail == "locate":
            req, _ = h.body()
            with self.lock:
                return 200, self.locate(name, req)
        raise FakeError(404, "invalid_request", f"no route {method} {path}")

    def update(self, name: str, req: dict[str, Any], files: dict[str, bytes],
               dry: bool) -> tuple[int, Any]:
        for f in req["frames"]:
            if f["image"] not in files:
                raise FakeError(400, "invalid_request", f"no part {f['image']}")
        metric = req["frames"][0].get("prior", {}).get("trust") in ("metric", "drifting")
        with self.lock:
            sim = self.maps[name].sim if name in self.maps else (
                self.cfg.metric_sim if metric else self.cfg.none_sim)
            if name not in self.maps:
                self.pending_sim[name] = sim
            inputs = []
            for f in req["frames"]:
                n = f["name"]
                if n in self.cfg.reject:
                    inputs.append({"name": n, "status": "rejected", "reason": self.cfg.reject[n],
                                   "inliers": 3, "T_map_cam": None, "sigma_pos_m": None,
                                   "sigma_rot_deg": None, "prior_residual_m": None,
                                   "prior_residual_deg": None, "detail": None,
                                   "camera_id": f["camera_id"]})
                    if self.cfg.reject[n] == "too_close":
                        inputs[-1].update(min_distance_m=0.3, scene_distance_m=0.2)
                    continue
                inputs.append({"name": n, "status": "registered", "reason": None,
                               "inliers": 200,
                               "T_map_cam": sim.pose_to_map(self.cfg.truth[n]).tolist(),
                               "sigma_pos_m": 0.01, "sigma_rot_deg": 0.5,
                               "prior_residual_m": 0.01 if metric else None,
                               "prior_residual_deg": 0.1 if metric else None, "detail": None,
                               "camera_id": f["camera_id"]})
            if dry:
                job = Job("dry", name, [], inputs, 0.0, False)
                return 200, {"schema": "oms/2.3", "map": name, "base_version": None,
                             "frames": [{"name": i["name"], "features": 500,
                                         "map_inliers": None, "neighbour_inliers": [],
                                         "plain": False, "too_close": False,
                                         "predicted_status": "registered" if i["status"] !=
                                         "rejected" else "rejected",
                                         "predicted_reason": i["reason"],
                                         **{k: i[k] for k in ("min_distance_m",
                                                              "scene_distance_m") if k in i}}
                                        for i in inputs],
                             "quality": self.quality(job),
                             "predicted_s": {"registration": 0.07, "fusion": 0.12}, "labels": [],
                             "scale_source": "priors", "scale_sigma_rel": 0.01}
            self.accept_ifs.append(req.get("accept_if"))
            self.priors.extend(f.get("prior") for f in req["frames"])
            self.heights.extend(f["height_above_floor_m"] for f in req["frames"]
                                if "height_above_floor_m" in f)
            self.depths.extend(f["depth"] for f in req["frames"]
                               if "depth" in f and f["depth"] in files)
            if name in self.maps:
                self.maps[name].cameras.update(req.get("cameras", {}))
            job = Job(uuid.uuid4().hex[:8], name, [f["name"] for f in req["frames"]], inputs,
                      time.monotonic(), "accept_if" in req)
            self.jobs[job.id] = job
            return 202, self.job_body(job)


class FakeError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.retryable = status, code, message, retryable


__all__ = ["ALL_CAPS", "Config", "Fake", "Sim", "house"]
