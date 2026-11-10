"""Replayer paths that a well-behaved service never takes: errors, races and fallbacks."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import replay_eval
from rpl_api import ApiError, Log
from rpl_episode import Event, load_episode
from rpl_layout import load_layout
from rpl_replay import Options, Replayer

from . import synth
from .fake_server import Config, Fake, Sim

Handler = Callable[[str, str, str, dict[str, Any]], Any]


class StubApi:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.log = Log()

    def call(self, method: str, path: str, route: str, schema: str | None = None,
             **kw: Any) -> tuple[Any, float]:
        return self.handler(method, path, route, kw), 0.01

    def clone(self) -> StubApi:
        return self


def err(code: str, status: int = 400) -> ApiError:
    return ApiError(status, code, code, False)


@pytest.fixture
def ep(tmp_path: Path) -> Any:
    synth.build(tmp_path)
    return load_episode(tmp_path / "e041", load_layout(None))


def replayer(ep: Any, handler: Handler, caps: tuple[str, ...] = ("O1",), **opt: Any) -> Replayer:
    r = Replayer(StubApi(handler), ep, "none", list(caps), "m", Options(**opt))  # type: ignore
    r.anchor_run = "r1"
    return r


def frames_body(v: int, names: list[str] | None = None) -> dict[str, Any]:
    return {"map_version": v, "frames": [{"name": n, "T_map_cam": [[1, 0, 0, 0]] * 4}
                                         for n in names or []]}


def test_probe_warnings_and_consistency(ep: Any) -> None:
    stop = threading.Event()

    def handler(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        stop.set()
        raise err("internal", 500)

    r = replayer(ep, handler, probe_interval_s=0.0)
    r.probe_loop(stop)
    assert r.rec.warnings == ["probe: HTTP 500 internal: internal"]

    calls: list[int] = []

    def versions(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        v = kw["params"]["version"]
        calls.append(v)
        if v == 1:
            raise err("version_gone", 410)
        raise err("internal", 500)

    r = replayer(ep, versions)
    r.rec.digests = {1: "a", 2: "b"}
    with pytest.raises(ApiError, match="internal"):
        r.check_consistency()
    assert calls == [1, 2] and r.rec.consistency == []


def test_update_error_and_idle_probes(ep: Any) -> None:
    def handler(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        if route == "POST updates":
            raise err("invalid_request")
        return frames_body(1)

    r = replayer(ep, handler, probes=False)
    event = ep.events[0]
    obs = r.update(event, ep.frames(event))
    assert "invalid_request" in (obs.error or "") and r.rec.warnings
    r.idle_probes({"inputs": [{"name": "x", "status": "rejected"}]})
    assert r.probe_frame is None


def test_locates_before_the_map_and_new_versions(ep: Any) -> None:
    frames = ep.frames(ep.events[0])
    seen: list[str] = []

    def missing(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        raise err("map_not_found", 404)

    r = replayer(ep, missing)
    r.plan_locate(frames[:1], "plan")
    assert "locate before the map exists" in r.rec.warnings[0]

    def broken(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        raise err("internal", 500)

    with pytest.raises(ApiError):
        replayer(ep, broken).plan_locate(frames[:1], "plan")

    def located(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        seen.append(route)
        if route == "POST locate":
            return {"map_version": 3, "frames": []}
        return frames_body(3)

    r = replayer(ep, located)
    r.plan_locate(frames[:1], "plan")
    assert seen == ["POST locate", "GET frames"] and 3 in r.rec.frames
    r.plan_locate(frames[:1], "plan")
    assert seen[-1] == "POST locate"


def test_run_delete_errors_and_empty_events(ep: Any) -> None:
    def busy(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        raise err("map_busy", 409)

    with pytest.raises(ApiError, match="map_busy"):
        replayer(ep, busy).run()
    deletes: list[int] = []

    def handler(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        deletes.append(1)
        raise err("map_not_found", 404)

    ep.events = [Event("update", "r1/start", ["nothing.jpg"])]
    rec = replayer(ep, handler).run()
    assert rec.jobs == [] and len(deletes) == 2
    assert rec.warnings == ["delete: HTTP 404 map_not_found: map_not_found"]


def test_fallback_locates_and_single_version(tmp_path: Path) -> None:
    root = tmp_path / "eps"
    sets = [synth.SetSpec("r1", "start", [(0, 0, 0), (0.2, 0, 0), (0.4, 0, 0), (0.6, 0, 0)])]
    syn = synth.build(root, sets=sets)  # four frames: nothing held out, no locate set
    rej = syn.names["r1/start"][1]
    cfg = Config(truth=syn.truth, metric_sim=Sim(1.0, 0.0, tuple(syn.start["r1"][:3, 3])),
                 reject={rej: "blur"}, capabilities=["O1", "O2", "O4", "O5"],
                 durations=(0.0, 0.0, 0.0))
    with Fake(cfg) as fake:
        out = tmp_path / "r.json"
        assert replay_eval.main(["--server", fake.url, "--episodes", str(root), "--out",
                                 str(out), "--priors", "metric", "--no-probes"]) == 0
    res = json.loads(out.read_text())
    run = res["episodes"]["e041"]["runs"]["metric"]
    assert run["locates"] == 1
    assert run["metrics"]["c5.revert_s"]["status"] == "pass"
    # one job with no provisional phase commits versions 1 (provisional) and 2: revert ran
    assert res["overall"]["c8.located_near_keyframe"]["value"] == 1.0


def test_final_reads_without_a_second_version(ep: Any) -> None:
    routes: list[str] = []

    def handler(method: str, path: str, route: str, kw: dict[str, Any]) -> Any:
        routes.append(route)
        if route == "GET map":
            return {"latest_version": 1}
        if route == "GET versions":
            return {"versions": []}
        return frames_body(1)

    r = replayer(ep, handler, caps=("O2",))
    r.final_reads()
    assert "POST revert" not in routes and r.rec.revert_s is None
