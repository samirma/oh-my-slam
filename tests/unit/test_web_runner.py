"""``server.sh`` job runner (spec §2.6 "Jobs") with a stand-in command: states, inference jobs one at
a time in submission order while the others run (bounded), map writers never concurrent, progress
and events, cancellation (SIGINT, then SIGTERM and SIGKILL for a deaf command), a stop and a second
stop signal, persistence across a restart (a left-behind process group is killed), re-submission."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from oh_my_slam.web.app import create_app
from oh_my_slam.web.jobs import Job, Runner
from oh_my_slam.web.workspace import Workspace
from tests.unit.test_view_cli import minimal_map
from tests.unit.test_web_api import Svc, make_svc, slow, slow_op

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _repo_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "data")
    w.create()
    return w


@pytest.fixture
def svc(ws: Workspace) -> Iterator[Svc]:
    service = make_svc(ws)
    with TestClient(create_app(service)) as client:
        yield Svc(service, client)
    service.runner.shutdown()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for(cond: object, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:  # type: ignore[operator]
        time.sleep(0.05)
    assert cond(), "condition not reached"  # type: ignore[operator]


def test_inference_jobs_run_one_at_a_time_in_order_others_meanwhile(svc: Svc) -> None:
    runner = svc.runner
    a = runner.submit(slow_op(), {}, slow(3), runner.new_id())
    b = runner.submit(slow_op(), {}, slow(0.1), runner.new_id())
    c = runner.submit(slow_op(), {}, slow(0.1, inference=False), runner.new_id())
    # scheduling is synchronous: the states hold right after submission
    assert [runner.get(j.id).state for j in (a, b, c)] == ["running", "queued", "running"]
    assert svc.client.get("/api/health").json()["service"]["jobs"] == {"queued": 1, "running": 2}
    c = runner.wait(c.id, 60)
    assert runner.get(b.id).state == "queued"  # still behind a, though c is long done
    a, b = runner.wait(a.id, 60), runner.wait(b.id, 60)
    assert [j.state for j in (a, b, c)] == ["succeeded"] * 3
    assert b.started_at >= a.ended_at and c.started_at < a.ended_at
    assert a.stages[0]["stage"] == "setup" and a.stages[0]["seconds"] > 2.0
    assert a.progress == {"stage": "setup", "done": 10, "total": 10}
    assert a.log_tail[-1] == "slow.sh: slept 3 s"
    assert svc.client.get(f"/api/jobs/{a.id}/log").text.endswith("slept 3 s\n")
    assert svc.client.get(f"/api/jobs/{a.id}/timings").status_code == 200


def test_other_jobs_run_at_most_max_parallel_together(ws: Workspace) -> None:
    runner = Runner(ws, max_parallel=1)
    first = runner.submit(slow_op(), {}, slow(1, inference=False), runner.new_id())
    second = runner.submit(slow_op(), {}, slow(0, inference=False), runner.new_id())
    assert [runner.get(j.id).state for j in (first, second)] == ["running", "queued"]
    second = runner.wait(second.id, 60)
    assert second.started_at >= runner.get(first.id).ended_at
    runner.shutdown()


def test_two_jobs_never_write_the_same_map(svc: Svc) -> None:
    runner = svc.runner
    w1 = runner.submit(slow_op(), {}, slow(1, inference=False, writes="/m"), runner.new_id())
    w2 = runner.submit(slow_op(), {}, slow(0, inference=False, writes="/m"), runner.new_id())
    other = runner.submit(slow_op(), {}, slow(0, inference=False, writes="/n"), runner.new_id())
    assert [runner.get(j.id).state for j in (w1, w2, other)] == ["running", "queued", "running"]
    w1, w2 = runner.wait(w1.id, 60), runner.wait(w2.id, 60)
    assert w2.started_at >= w1.ended_at


def test_cancel_a_running_job_interrupts_the_command(svc: Svc) -> None:
    runner = svc.runner
    job = runner.submit(slow_op(), {}, slow(30), runner.new_id())
    wait_for(lambda: runner.get(job.id).progress is not None)  # the command is in its stage
    assert svc.client.post(f"/api/jobs/{job.id}/cancel", json={}).status_code == 200
    job = runner.wait(job.id, 30)
    assert job.state == "cancelled" and job.exit_code == 130  # the command's own Ctrl-C exit
    assert svc.client.post(f"/api/jobs/{job.id}/cancel", json={}).status_code == 409
    assert svc.client.get(f"/api/jobs/{job.id}/result").status_code == 404


def test_cancel_escalates_for_a_command_that_ignores_sigint(ws: Workspace) -> None:
    runner = Runner(ws, cancel_grace_s=0.5)
    job = runner.submit(slow_op(), {}, slow(60, "--ignore-sigint"), runner.new_id())
    wait_for(lambda: runner.get(job.id).progress is not None)
    runner.cancel(job.id)
    job = runner.wait(job.id, 30)
    assert job.state == "cancelled" and job.exit_code == -signal.SIGTERM
    runner.shutdown()


def test_progress_events_stream_until_the_job_ends(svc: Svc) -> None:
    runner = svc.runner
    job = runner.submit(slow_op(), {}, slow(0.5), runner.new_id())
    states, stages = [], set()
    with svc.client.stream("GET", f"/api/jobs/{job.id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                states.append(ev["state"])
                stages.add(ev["stage"])
    assert states[-1] == "succeeded" and "setup" in stages
    assert svc.client.get(f"/api/jobs/{job.id}").json()["state"] == "succeeded"
    assert svc.client.get("/api/jobs/missing").status_code == 404


def _orphan(jid: str | None) -> subprocess.Popen[bytes]:
    """A process group a killed service could have left: its leader carries ``jid`` (if any)."""
    env = {**os.environ, "OH_MY_SLAM_JOB": jid} if jid else dict(os.environ)
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            start_new_session=True, env=env)


def _stale(ws: Workspace, jid: str, pgid: int, ctime: float | None) -> None:
    job = Job(id=jid, operation="slow", label="slow", params={}, command=[], steps=[],
              inference=True, state="running", pgid=pgid, leader_ctime=ctime)
    (ws.jobs / jid).mkdir()
    (ws.jobs / jid / "job.json").write_text(json.dumps(dataclasses.asdict(job)))


def test_job_list_survives_a_restart(ws: Workspace) -> None:
    import psutil

    runner = Runner(ws)
    done = runner.submit(slow_op(), {"x": 1}, slow(0), runner.new_id())
    runner.wait(done.id, 30)
    runner.shutdown()
    # a job the killed service left running, its process group still alive and still its own
    mine = _orphan("20000101-000000-aaaaaa")
    wait_for(lambda: "OH_MY_SLAM_JOB" in psutil.Process(mine.pid).environ())
    _stale(ws, "20000101-000000-aaaaaa", mine.pid, psutil.Process(mine.pid).create_time())
    # recycled numbers: a group whose leader started at another time, or belongs to no job
    other_time = _orphan("20000101-000000-bbbbbb")
    _stale(ws, "20000101-000000-bbbbbb", other_time.pid,
           psutil.Process(other_time.pid).create_time() - 5.0)
    stranger = _orphan(None)
    _stale(ws, "20000101-000000-cccccc", stranger.pid, psutil.Process(stranger.pid).create_time())
    try:
        again = Runner(ws)
        again.load()
        assert mine.wait(10) == -signal.SIGKILL
        time.sleep(0.5)
        assert other_time.poll() is None and stranger.poll() is None  # never signalled
        assert again.get(done.id).state == "succeeded" and again.get(done.id).params == {"x": 1}
        for jid in ("20000101-000000-aaaaaa", "20000101-000000-bbbbbb", "20000101-000000-cccccc"):
            assert again.get(jid).state == "cancelled"
            saved = json.loads((ws.jobs / jid / "job.json").read_text())
            assert saved["state"] == "cancelled" and saved["pgid"] is None
    finally:
        for p in (mine, other_time, stranger):
            p.kill()
            p.wait()


def test_the_running_steps_process_group_is_recorded(ws: Workspace) -> None:
    runner = Runner(ws)
    job = runner.submit(slow_op(), {}, slow(30), runner.new_id())
    wait_for(lambda: runner.get(job.id).pgid is not None)
    saved = json.loads((ws.jobs / job.id / "job.json").read_text())
    saved_pgid = saved["pgid"]
    assert saved_pgid == runner.get(job.id).pgid and alive(saved_pgid)
    import psutil

    assert saved["leader_ctime"] == psutil.Process(saved["pgid"]).create_time()
    assert psutil.Process(saved["pgid"]).environ()["OH_MY_SLAM_JOB"] == job.id
    runner.kill_all()  # the second stop signal: the record is final before the process exits
    saved = json.loads((ws.jobs / job.id / "job.json").read_text())
    assert saved["state"] == "cancelled" and saved["pgid"] is None
    assert saved["leader_ctime"] is None
    wait_for(lambda: not alive(runner.get(job.id).pgid or saved_pgid))
    job = runner.wait(job.id, 30)
    assert job.state == "cancelled"


def test_resubmit_uses_the_same_parameters(svc: Svc) -> None:
    minimal_map(svc.ws.maps / "m")
    r = svc.client.post("/api/ops/segment-map", json={"map": "m", "format": "ply"})
    first = svc.runner.wait(r.json()["id"], 120)
    assert first.state == "succeeded", first.log_tail
    r = svc.client.post(f"/api/jobs/{first.id}/resubmit", json={"format": "json"})
    assert r.status_code == 202
    second = svc.runner.wait(r.json()["id"], 120)
    assert second.resubmitted_from == first.id and second.params == {"map": "m", "format": "json"}
    assert first.result_name == "result.ply" and second.result_name == "result.json"


def test_service_shutdown_cancels_running_and_queued_jobs(ws: Workspace) -> None:
    runner = Runner(ws, stop_grace_s=30)
    running = runner.submit(slow_op(), {}, slow(30), runner.new_id())
    queued = runner.submit(slow_op(), {}, slow(30), runner.new_id())
    wait_for(lambda: runner.get(running.id).pgid is not None)
    runner.shutdown()
    assert runner.get(running.id).state == "cancelled"
    assert runner.get(queued.id).state == "cancelled"


def test_a_conditional_jobs_inference_need_is_rechecked_when_it_starts(ws: Workspace) -> None:
    """Queued behind an inference job, a job whose need depends on what it reads is re-checked
    when it may start: no longer needing inference, it runs at once."""
    runner = Runner(ws, reevaluate=lambda job: False)
    first = runner.submit(slow_op(), {}, slow(2), runner.new_id())
    cond = runner.submit(slow_op(), {}, slow(0, conditional=True), runner.new_id())
    assert runner.get(cond.id).state == "running" and not runner.get(cond.id).inference
    assert runner.get(first.id).state == "running"
    runner.shutdown()
