"""``server.sh``'s two operations on a map of the workspace that are no command's mode (spec §2.6
"Operations", "API"): removing a map — deleted for good, refused while a request in progress reads
or writes it or an update holds its lock, its viewer stopped first, 404 for a name that is no map,
behind the request guard — and starting or stopping its viewer — ``view.sh -m`` run from its
shared definition, at most one per map (a second start answers its URL again), served on
127.0.0.1 only, stopped when the service stops, its state in the map list and the map's summary.
The viewers are real ``view.sh`` processes on a minimal map (each on a port of its own choosing);
stand-in scripts play a viewer that fails, never listens or ignores SIGTERM."""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import subprocess
import threading
import time
import urllib.request
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from oh_my_slam.core.errors import MapLockedError
from oh_my_slam.mapping import store
from oh_my_slam.web import viewers as web_viewers
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.operations import (
    Operation,
    Prepared,
    named_map,
    named_uploads,
    operations,
    prepare,
)
from oh_my_slam.web.runner import RunError, command_message
from oh_my_slam.web.viewers import ViewerError, Viewers
from oh_my_slam.web.workspace import NotFoundError, Workspace
from tests.fakes import slow_command
from tests.unit.test_view_cli import minimal_map
from tests.unit.test_web_api import (  # noqa: F401
    _repo_importable,
    _test_client_host,
    jpeg,
    make_svc,
    with_programs,
)
from tests.unit.test_web_requests import Call, send, until, upload


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "data")
    w.create()
    minimal_map(w.maps / "m")
    return w


@pytest.fixture
def service(ws: Workspace) -> Iterator[Service]:
    s = make_svc(ws)
    yield s
    s.runner.shutdown()
    s.viewers.stop_all()


@pytest.fixture
def client(service: Service) -> Iterator[TestClient]:
    with TestClient(create_app(service)) as c:
        yield c


def page(url: str) -> int:
    """The status of the viewer's page at ``url`` (it is view.sh's own server)."""
    with urllib.request.urlopen(url + "api/meta", timeout=10) as r:
        return int(r.status)


def proc_of(service: Service, name: str) -> subprocess.Popen[bytes]:
    return service.viewers._viewers[name].proc


def script(tmp_path: Path, body: str) -> str:
    """A stand-in for the viewer's interpreter: an executable shell script (it is given the
    viewer's arguments and ignores them)."""
    path = tmp_path / f"viewer-{len(list(tmp_path.glob('viewer-*')))}.sh"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


LISTEN = 'echo "view.sh: listening on http://127.0.0.1:1/" >&2'


# -- the viewer ---------------------------------------------------------------------------------------


def test_a_viewer_starts_once_per_map_and_stops(service: Service, client: TestClient) -> None:
    """view.sh -m on the map, served on 127.0.0.1: a second start answers the same URL (the same
    process), the map list and the map's summary say it runs, and a stop ends it (stopping one
    that does not run is no error)."""
    assert client.get("/api/maps").json()[0]["viewer"] == {"running": False, "url": None}
    r = client.post("/api/maps/m/viewer", json={})
    assert r.status_code == 200
    url = r.json()["url"]
    assert r.json() == {"running": True, "url": url} and url.startswith("http://127.0.0.1:")
    assert page(url) == 200
    proc = proc_of(service, "m")
    assert proc.args[1:3] == ["-m", web_viewers.MODULE] == ["-m", "oh_my_slam.cli.view"]
    assert proc.args[3:] == [f"-m={service.workspace.maps / 'm'}", "--no-browser"]
    again = client.post("/api/maps/m/viewer", content=b"",  # no body: the same as {}
                        headers={"content-type": "application/json"})
    assert again.json() == {"running": True, "url": url} and proc_of(service, "m") is proc
    assert client.get("/api/maps").json()[0]["viewer"] == {"running": True, "url": url}
    assert client.get("/api/maps/m").json()["viewer"] == {"running": True, "url": url}
    stopped = client.delete("/api/maps/m/viewer")
    assert stopped.status_code == 200 and stopped.json() == {"running": False, "url": None}
    assert proc.poll() == 0  # view.sh's normal stop
    assert client.delete("/api/maps/m/viewer").json() == {"running": False, "url": None}
    assert client.get("/api/maps/m").json()["viewer"] == {"running": False, "url": None}


def test_a_viewer_takes_only_an_existing_maps_name(service: Service, client: TestClient,
                                                   ws: Workspace) -> None:
    (ws.maps / "plain").mkdir()
    for name in ("nope", "plain"):
        for r in (client.post(f"/api/maps/{name}/viewer", json={}),
                  client.delete(f"/api/maps/{name}/viewer")):
            assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", name
    for r in (client.post("/api/maps/.staging/viewer", json={}),  # never a map's name
              client.delete("/api/maps/.staging/viewer"), client.get("/api/maps/.staging")):
        assert r.status_code == 400 and "a plain file or folder name" in r.json()["error"][
            "message"]
    for body in ({"map": "m"}, [], "x"):
        r = client.post("/api/maps/m/viewer", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "usage"
        assert "takes only the map's name" in r.json()["error"]["message"]
    r = client.post("/api/maps/m/viewer", content=b"{", headers={"content-type":
                                                                   "application/json"})
    assert r.status_code == 400
    assert service.viewers.state("m") == {"running": False, "url": None}


def test_a_viewer_that_fails_answers_its_own_error(service: Service, client: TestClient,
                                                   tmp_path: Path) -> None:
    """A viewer that ends before it listens answers its exit status and its own message, by the
    generic rule; one that never listens is killed after the start timeout (an internal error)."""
    service.viewers.python = script(tmp_path, 'echo "loading" >&2\n'
                                              'echo "view.sh: error: not a map: m" >&2; exit 4')
    r = client.post("/api/maps/m/viewer", json={})
    assert r.status_code == 422 and r.json()["error"] == {
        "code": "not_a_map", "exit_code": 4, "message": "not a map: m", "http_status": 422}
    service.viewers.python = script(tmp_path, f"echo $$ > {tmp_path}/pid; exec sleep 30")
    service.viewers.start_timeout_s = 1.0
    r = client.post("/api/maps/m/viewer", json={})
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal"
    assert r.json()["error"]["message"] == "the viewer of map m did not start within 1 s"
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):  # killed and reaped
        os.kill(pid, 0)
    assert service.viewers.state("m") == {"running": False, "url": None}


def test_a_viewer_that_ended_by_itself_counts_as_stopped(service: Service, ws: Workspace,
                                                         tmp_path: Path) -> None:
    service.viewers.python = script(tmp_path, f"{LISTEN}\nsleep 30")
    url = service.viewers.start("m", ws.maps / "m")
    assert service.viewers.state("m") == {"running": True, "url": url}
    proc = proc_of(service, "m")
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    assert service.viewers.state("m") == {"running": False, "url": None}
    assert service.viewers.start("m", ws.maps / "m") == url  # a new one
    assert proc_of(service, "m") is not proc
    again = proc_of(service, "m")
    os.killpg(again.pid, signal.SIGKILL)
    again.wait()
    service.viewers.stop("m")  # nothing left to end
    assert again.returncode == -signal.SIGKILL and "m" not in service.viewers._viewers


def test_a_viewer_deaf_to_sigterm_is_killed(service: Service, ws: Workspace, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_viewers, "STOP_TIMEOUT_S", 0.5)
    service.viewers.python = script(tmp_path, f"trap '' TERM\n{LISTEN}\n"
                                              "while :; do sleep 0.1; done")
    service.viewers.start("m", ws.maps / "m")
    proc = proc_of(service, "m")
    t0 = time.monotonic()
    service.viewers.stop("m")
    assert proc.returncode == -signal.SIGKILL and time.monotonic() - t0 < 5


def test_stopping_the_service_stops_every_viewer(service: Service, ws: Workspace,
                                                 tmp_path: Path) -> None:
    """Every viewer the service started is stopped (and none starts any more); a second stop
    signal kills them without waiting."""
    minimal_map(ws.maps / "n")
    service.viewers.start("m", ws.maps / "m")
    service.viewers.start("n", ws.maps / "n")
    procs = [proc_of(service, "m"), proc_of(service, "n")]
    service.viewers.stop_all()
    assert [p.returncode for p in procs] == [0, 0]
    with pytest.raises(RunError, match="the service is stopping") as e:
        service.viewers.start("m", ws.maps / "m")
    assert e.value.code == "stopping"
    with TestClient(create_app(service)) as c:
        r = c.post("/api/maps/m/viewer", json={})
    assert r.status_code == 503 and r.json()["error"]["code"] == "stopping"
    # a second stop signal: SIGKILL, waiting for none
    viewers = Viewers(ws, python=script(tmp_path, f"{LISTEN}\nsleep 30"))
    viewers.start("m", ws.maps / "m")
    viewers.start("n", ws.maps / "n")
    ended = viewers._viewers["n"].proc
    os.killpg(ended.pid, signal.SIGKILL)  # one that ended by itself is left alone
    ended.wait()
    proc = viewers._viewers["m"].proc
    viewers.kill_all()
    assert proc.wait(10) == -signal.SIGKILL and viewers.stopping


@pytest.mark.parametrize(("tail", "code", "message"), [
    (["a", "view.sh: error: no map", "b"], 4, "no map"),
    (["view.sh: internal error: ValueError: x"], 1, "ValueError: x"),
    (["mapper.sh: error: not mine", "last words"], -9, "the viewer was stopped by signal 9"),
    (["first", "last"], 2, "last"),
    ([], 3, "the viewer exited with status 3")])
def test_a_failed_viewers_message(tail: list[str], code: int, message: str) -> None:
    """A viewer's error message is a command's (``runner.command_message``, one helper for
    both): its own error line, else its last stderr line, else its exit status."""
    assert command_message("view.sh", tail, code, "viewer") == message
    assert command_message("view.sh", tail, code) == message.replace("viewer", "command")
    assert str(ViewerError(code, message)) == message and ViewerError(code, message).code == code


# -- the removal ---------------------------------------------------------------------------------------


def test_a_map_is_removed_for_good(service: Service, client: TestClient, ws: Workspace) -> None:
    """Deleted from the workspace (no trace left), its viewer stopped first; the name is then no
    map (404), and a new map may take it."""
    url = client.post("/api/maps/m/viewer", json={}).json()["url"]
    proc = proc_of(service, "m")
    assert page(url) == 200
    r = client.delete("/api/maps/m")
    assert r.status_code == 204 and r.content == b""
    assert proc.returncode == 0 and service.viewers.state("m")["running"] is False
    assert list(ws.maps.iterdir()) == [] and list(ws.removed.iterdir()) == []
    assert client.get("/api/maps").json() == []
    for r in (client.get("/api/maps/m"), client.delete("/api/maps/m")):
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    minimal_map(ws.maps / "m")
    assert client.get("/api/maps/m").status_code == 200


def test_only_a_map_can_be_removed(client: TestClient, ws: Workspace) -> None:
    (ws.maps / "plain").mkdir()
    (ws.maps / "plain" / "keep.txt").write_text("x")
    for name in ("nope", "plain"):
        r = client.delete(f"/api/maps/{name}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", name
    r = client.delete("/api/maps/.staging")
    assert r.status_code == 400 and r.json()["error"]["code"] == "usage"
    assert (ws.maps / "plain" / "keep.txt").read_text() == "x"
    assert (ws.maps / "m" / "map.json").is_file()


def test_a_linked_map_loses_only_its_link(client: TestClient, ws: Workspace) -> None:
    target = minimal_map(ws.maps / "real")
    (ws.maps / "alias").symlink_to(target)
    assert client.delete("/api/maps/alias").status_code == 204
    assert not (ws.maps / "alias").exists() and (target / "map.json").is_file()
    assert sorted(m["name"] for m in client.get("/api/maps").json()) == ["m", "real"]


def test_a_map_an_update_holds_is_not_removed(service: Service, client: TestClient,
                                               ws: Workspace) -> None:
    """An update that holds the map's lock (mapper.sh run from a terminal) keeps it: the
    commands' own error (409 ``map_locked``), before any change — the map stays as it was, and so
    does its viewer."""
    url = client.post("/api/maps/m/viewer", json={}).json()["url"]
    proc = proc_of(service, "m")
    fd = store.lock_update(ws.maps / "m")
    try:
        r = client.delete("/api/maps/m")
        assert r.status_code == 409 and r.json()["error"]["code"] == "map_locked"
        assert r.json()["error"]["message"] == (f"another update is running on {ws.maps / 'm'}; "
                                                "remove map m once that update ended")
        with pytest.raises(MapLockedError), ws.map_lock("m"):
            pass
    finally:
        os.close(fd)
    assert (ws.maps / "m" / "map.json").is_file() and not ws.removed.exists()  # nothing moved
    assert service.viewers.state("m") == {"running": True, "url": url}
    assert proc_of(service, "m") is proc and proc.poll() is None and page(url) == 200
    assert client.delete("/api/maps/m").status_code == 204
    assert proc.returncode == 0


@pytest.mark.parametrize("program", ["reader", "writer"])
def test_a_map_a_request_in_progress_uses_is_not_removed(
        service: Service, ws: Workspace, monkeypatch: pytest.MonkeyPatch, program: str) -> None:
    """Refused with the service's own error (409 ``map_in_use``) while a request in progress —
    running, or waiting for its turn — reads the map (locating images in it) or writes it (an
    update); removed once it ended."""
    fake = (slow_command.map_registry_program() if program == "reader"
            else slow_command.registry_program())
    with_programs(monkeypatch, service, fake)
    service.inference_check = lambda: None

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        a, ta = await send(app, "/api/ops/slow", {"seconds": 3, "map": "m"})
        await until(lambda: bool(service.runner.running))
        run = service.runner.running[0]
        assert (run.reads, run.writes) == ((str(ws.maps / "m"), None) if program == "reader"
                                           else (None, str(ws.maps / "m")))
        refused = await Call(app, "/api/maps/m", method="DELETE")()
        assert refused.status == 409 and refused.json()["error"] == {
            "code": "map_in_use", "http_status": 409,
            "message": "map m is used by a request in progress (slow, running); remove it once "
                       "that request ended"}
        assert (ws.maps / "m" / "map.json").is_file()
        await ta
        assert a.status == 200
        done = await Call(app, "/api/maps/m", method="DELETE")()
        assert done.status == 204 and not (ws.maps / "m").exists()

    try:
        asyncio.run(scenario(create_app(service)))
    finally:
        service.runner.shutdown()


@pytest.mark.parametrize("op", ["mapper-update", "mapper-locate"])
def test_a_map_a_request_still_being_validated_names_is_not_removed(
        service: Service, ws: Workspace, monkeypatch: pytest.MonkeyPatch, op: str) -> None:
    """From its arrival on: a request that names the map — to write it (an update) or to read it
    (locating images in it) — keeps it while it is still being validated (the command's checks,
    the inference server's), and the refused removal changes nothing, its viewer included; once
    the request ended, the map is removed. (The commands do nothing here: their interpreter is
    ``true``.)"""
    jpeg(ws.root / "a.jpg")
    service.runner.python = "/usr/bin/true"
    service.inference_check = lambda: None
    validate, gate = service.prepare, threading.Event()

    def held_in_validation(o: Operation, params: Any) -> Prepared:
        gate.wait(30)
        return validate(o, params)

    monkeypatch.setattr(service, "prepare", held_in_validation)
    url = service.viewers.start("m", ws.maps / "m")

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        a, ta = await send(app, f"/api/ops/{op}", {"inputs": ["a.jpg"], "map": "m"})
        refused = await Call(app, "/api/maps/m", method="DELETE")()
        assert refused.status == 409 and refused.json()["error"] == {
            "code": "map_in_use", "http_status": 409,
            "message": f"map m is used by a request in progress ({op}, validating); remove it "
                       "once that request ended"}
        assert service.viewers.state("m") == {"running": True, "url": url}
        assert (ws.maps / "m" / "map.json").is_file()
        gate.set()
        await ta
        assert a.status == 200, a.body
        done = await Call(app, "/api/maps/m", method="DELETE")()
        assert done.status == 204 and not (ws.maps / "m").exists()
        assert service.viewers.state("m")["running"] is False

    try:
        asyncio.run(scenario(create_app(service)))
    finally:
        gate.set()
        service.runner.shutdown()


def test_a_request_that_names_a_map_being_removed_is_refused(
        service: Service, ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """While a removal stops the map's viewer (here one that takes a second to exit, as a large
    map's may), a request that names the map — to read it or to write it — is refused at once,
    before its validation (409 ``map_in_use``), and consumes its uploads all the same; so is a
    second removal. The map is never renamed away under a request that names it, and once it is
    gone its name is free. (The commands do nothing here: their interpreter is ``true``.)"""
    jpeg(ws.root / "a.jpg")
    service.runner.python = "/usr/bin/true"
    service.inference_check = lambda: None
    stop, detach = service.viewers.stop, ws.detach_map
    at_rename: list[object] = []

    def slow_stop(name: str) -> None:
        stop(name)
        time.sleep(1.0)

    def watched_detach(name: str) -> Path:
        at_rename.append(service.runner.map_user(str(ws.maps / name)))
        return detach(name)

    monkeypatch.setattr(service.viewers, "stop", slow_stop)
    monkeypatch.setattr(ws, "detach_map", watched_detach)

    async def scenario(app: Callable[..., Awaitable[None]]) -> None:
        removal = asyncio.ensure_future(Call(app, "/api/maps/m", method="DELETE")())
        await until(lambda: str(ws.maps / "m") in service.runner.removing)
        for op in ("mapper-locate", "mapper-update"):
            given = upload(ws, "q.jpg")
            r = await Call(app, f"/api/ops/{op}", {"inputs": [given], "map": "maps/m"})()
            assert r.status == 409 and r.json()["error"] == {
                "code": "map_in_use", "http_status": 409,
                "message": "map m is being removed; send the request again once its removal "
                           "ended"}, op
            assert not (ws.root / given).parent.exists()  # consumed
            assert service.runner.in_progress() == []  # never admitted
        again = await Call(app, "/api/maps/m", method="DELETE")()
        assert again.status == 409
        assert again.json()["error"]["message"] == "map m is already being removed"
        done = await removal
        assert done.status == 204 and at_rename == [None]
        assert not service.runner.removing and not (ws.maps / "m").exists()
        new = await Call(app, "/api/ops/mapper-update", {"inputs": ["a.jpg"], "map": "m"})()
        assert new.status == 200  # the name is free: an update may create a map of that name

    try:
        asyncio.run(scenario(create_app(service)))
    finally:
        service.runner.shutdown()


def test_the_uploads_a_request_names_are_known_before_its_checks(ws: Workspace) -> None:
    """``named_uploads``: every upload among a request's file inputs, once each, whatever else
    is wrong with it (a refused request consumes them too); a map is never an upload."""
    a, b = upload(ws, "a.jpg"), upload(ws, "b.mp4")
    ida, idb = a.split("/")[1], b.split("/")[1]
    ops = operations()
    update, segment = ops["mapper-update"], ops["segment"]
    assert named_uploads(update, {"inputs": [a, "../x.jpg", 5, b, a], "map": a}, ws) == [ida, idb]
    assert named_uploads(segment, {"image": [a, b], "format": "nope"}, ws) == [ida, idb]
    assert named_uploads(segment, {"image": "photo.jpg"}, ws) == []
    for raw in (None, [], {}, {"map": a}):
        assert named_uploads(update, raw, ws) == []
    assert prepare(segment, {"image": [a, b]}, ws).uploads == [ida, idb]  # refused: one image


def test_the_map_a_request_names_is_known_before_its_checks(ws: Workspace) -> None:
    """``named_map``: the map parameter resolved as validation resolves it (``<name>`` or
    ``maps/<name>``), or nothing when the operation takes no map or the value could be none."""
    ops = operations()
    locate = ops["mapper-locate"]
    assert named_map(locate, {"map": "m"}, ws) == named_map(locate, {"map": "maps/m"}, ws) \
        == str(ws.maps / "m")
    assert named_map(ops["mapper-update"], {"map": "new"}, ws) == str(ws.maps / "new")
    for raw in ({}, {"map": 5}, {"map": ["m"]}, {"map": "../m"}, {"map": "uploads/x"}, None, []):
        assert named_map(locate, raw, ws) is None, raw
    assert named_map(ops["reconstruct"], {"map": "m", "image": "a.jpg"}, ws) is None


def test_the_removal_is_guarded_like_every_change(ws: Workspace) -> None:
    """A foreign ``Origin`` or a ``Host`` that does not name this machine is refused (403); a
    viewer's start needs a JSON body type, which a cross-site form cannot send (415)."""
    service = make_svc(ws)
    service.url = "http://0.0.0.0:8000/"
    with TestClient(create_app(service)) as c:
        for r in (c.delete("/api/maps/m", headers={"origin": "http://evil.example"}),
                  c.delete("/api/maps/m", headers={"host": "evil.example"}),
                  c.post("/api/maps/m/viewer", json={}, headers={"origin": "http://evil.example"}),
                  c.delete("/api/maps/m/viewer", headers={"origin": "http://evil.example"})):
            assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
            r = c.post("/api/maps/m/viewer", content=b"{}", headers={"content-type": ctype})
            assert r.status_code == 415, ctype
        assert service.viewers.state("m")["running"] is False
        assert (ws.maps / "m" / "map.json").is_file()
        r = c.delete("/api/maps/m", headers={"origin": "http://localhost:8000"})
        assert r.status_code == 204
    service.runner.shutdown()


def test_what_a_removal_left_is_cleared_and_readers_are_recorded(ws: Workspace) -> None:
    """A removal interrupted after its rename leaves the map under ``.removed``, which the next
    start clears; a request that only reads a map records it (``reads``), one that writes it
    ``writes``."""
    gone = ws.detach_map("m")
    assert gone.parent == ws.removed and (gone / "map.json").is_file()
    with pytest.raises(NotFoundError):
        ws.map_dir("m")
    ws.clear_removed()
    assert list(ws.removed.iterdir()) == []
    minimal_map(ws.maps / "m")
    from oh_my_slam.web.operations import operations

    (ws.root / "a.jpg").write_bytes(b"x")
    ops = operations()
    locate = prepare(ops["mapper-locate"], {"inputs": ["a.jpg"], "map": "m"}, ws)
    update = prepare(ops["mapper-update"], {"inputs": ["a.jpg"], "map": "m"}, ws)
    assert (locate.reads, locate.writes) == (str(ws.maps / "m"), None)
    assert (update.reads, update.writes) == (None, str(ws.maps / "m"))
