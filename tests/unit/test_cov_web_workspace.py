"""The web service's workspace (spec §2.6 "Workspace") and its operations' translation of a request
into the command's own command line, at their edges: paths that cannot be resolved or resolve to a
hidden entry, a map named by a path that is not directly in ``maps/``, the maps listed with the
summary figures of their own metadata (and the folders that are not maps left out), clearing what
uploads and requests left; values a parameter cannot take; a result format or an inference need
read from parameters that do not parse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oh_my_slam.commands import spec
from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.timing import Stage
from oh_my_slam.mapping import store
from oh_my_slam.web import operations as web_ops
from oh_my_slam.web.workspace import NotFoundError, OutsideWorkspaceError, Workspace, inside
from tests.fakes import slow_command
from tests.unit.test_view_cli import minimal_map


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "data")
    w.create()
    return w


# -- paths ---------------------------------------------------------------------------------------


def test_a_name_that_cannot_be_resolved_is_not_found(ws: Workspace) -> None:
    with pytest.raises(NotFoundError, match="not found"):
        inside(ws.maps, "m\0")  # an embedded NUL byte: no path at all


def test_a_name_that_resolves_to_a_hidden_entry_is_not_found(ws: Workspace) -> None:
    minimal_map(ws.maps / ".old")
    (ws.maps / "alias").symlink_to(ws.maps / ".old")
    with pytest.raises(NotFoundError):
        ws.map_summary("alias")
    assert [m["name"] for m in ws.list_maps()] == []


@pytest.mark.parametrize("value", ["", 7, None, "a\0b"])
def test_a_request_path_must_be_a_non_empty_text(ws: Workspace, value: object) -> None:
    with pytest.raises(UsageError, match="expected a path inside the workspace"):
        ws.resolve(value)  # type: ignore[arg-type]


def test_a_map_is_a_folder_directly_in_maps(ws: Workspace) -> None:
    minimal_map(ws.maps / "m")
    assert ws.map_path("m") == ws.map_path("maps/m") == (ws.maps / "m").resolve()
    for value in ("maps/m/frames", "inputs/m"):
        with pytest.raises(OutsideWorkspaceError, match="a map is a folder directly in"):
            ws.map_path(value)


def test_a_path_outside_the_workspace_is_shown_as_it_is(ws: Workspace, tmp_path: Path) -> None:
    assert ws.relative(ws.root / "inputs" / "a.jpg") == "inputs/a.jpg"
    assert ws.relative(tmp_path / "elsewhere.jpg") == str(tmp_path / "elsewhere.jpg")


def test_only_an_existing_map_has_a_folder(ws: Workspace) -> None:
    (ws.maps / "notes").mkdir()
    (ws.maps / "notes" / "todo.txt").write_text("x")
    for name in ("notes", "missing"):
        with pytest.raises(NotFoundError, match=f"no map {name}"):
            ws.map_dir(name)


# -- maps ----------------------------------------------------------------------------------------


def test_the_maps_are_listed_with_their_own_metadata(ws: Workspace) -> None:
    """Every map in ``maps/`` by name, with the summary figures of its ``map.json`` (scalars as
    they are, lists and objects counted), its frames and objects, and its last update; what is
    not a map (a hidden folder, a file, a folder of other files) is left out."""
    root = minimal_map(ws.maps / "b")
    meta = json.loads((root / store.MAP_JSON).read_text())
    meta.update({"title": "kitchen", "scale": 1.5, "up": [0, 0, 1], "ids": {"next": 3},
                 "updates": [{"id": 1, "at": "t1", "kind": "create", "frames_added": [0, 1],
                              "timings": {"total_s": 12.5}},
                             {"id": 2, "at": "t2", "kind": "extend", "frames_added": None}]})
    (root / store.MAP_JSON).write_text(json.dumps(meta))
    (root / store.OBJECTS_JSON).write_text(json.dumps({"objects": [{"id": 1}, {"id": 2}]}))
    plain = minimal_map(ws.maps / "a")
    (plain / store.OBJECTS_JSON).write_text(json.dumps("not a list"))
    (ws.maps / ".staging").mkdir()
    (ws.maps / "readme.txt").write_text("x")
    (ws.maps / "junk").mkdir()
    (ws.maps / "junk" / "notes.txt").write_text("x")

    maps = ws.list_maps()
    assert [m["name"] for m in maps] == ["a", "b"]
    a, b = maps
    assert a["path"] == "maps/a" and a["frames"] == 0 and a["objects"] is None
    assert a["update_count"] == 1 and "last_update" not in a  # no update history recorded
    assert b["title"] == "kitchen" and b["scale"] == 1.5 and b["up_count"] == 3 and b["ids_count"] == 1
    assert b["updates_count"] == 2 and b["objects"] == 2
    assert b["last_update"] == {"id": 2, "at": "t2", "kind": "extend", "frames_added": 0,
                                "total_s": None}
    assert "meta" not in b and ws.map_summary("b", full=True)["meta"] == meta


def test_clearing_deletes_files_and_folders_but_not_what_a_link_points_to(ws: Workspace,
                                                                          tmp_path: Path) -> None:
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "file.jpg").write_bytes(b"x")
    (ws.uploads / "abc").mkdir()
    (ws.uploads / "abc" / "a.jpg").write_bytes(b"x")
    (ws.uploads / ".stray.part").write_bytes(b"x")
    (ws.uploads / "link").symlink_to(kept)
    ws.clear_uploads()
    assert list(ws.uploads.iterdir()) == [] and (kept / "file.jpg").is_file()
    ws.clear_requests()  # no request folder yet: nothing to do
    assert not ws.requests.exists()


# -- the operations' translation -----------------------------------------------------------------


def _slow(monkeypatch: pytest.MonkeyPatch) -> web_ops.Operation:
    """The stand-in command ``slow.sh`` (``tests.fakes.slow_command``), added to the registry."""
    prog = slow_command.registry_program()
    monkeypatch.setattr(spec, "PROGRAMS", (*spec.PROGRAMS, prog))
    cmd = prog.commands[0]  # type: ignore[attr-defined]
    return web_ops.Operation(prog, cmd, cmd.modes[0])  # type: ignore[arg-type]


def test_a_flag_takes_true_or_false_and_a_single_value_option_one_value(
        ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    op = _slow(monkeypatch)
    prep = web_ops.prepare(op, {"ignore_sigint": "yes", "seconds": [1, 2]}, ws)
    assert {p.parameters: p.message for p in prep.problems} == {
        ("ignore_sigint",): "--ignore-sigint is a flag: give true or false",
        ("seconds",): "--seconds takes one value"}
    ok = web_ops.prepare(op, {"ignore_sigint": True, "seconds": 1}, ws)
    assert ok.problems == [] and ok.argv == ["--seconds=1", "--ignore-sigint"]


def test_the_result_format_and_inference_need_of_parameters_that_do_not_parse(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Read through the command's own parser: no format for parameters it refuses; an inference
    need as the mode declares it (only a mode that never uses the server is known not to)."""
    ops = web_ops.operations()
    bad = {"image": "a.jpg", "format": "xml"}
    assert web_ops.result_format(ops["reconstruct"], bad) is None
    assert web_ops.inference_of(ops["reconstruct"], bad) is True
    assert web_ops.inference_of(ops["mapper-locate"], {"map": "m"}) is True  # conditional
    assert web_ops.inference_of(_slow(monkeypatch), {"seconds": "soon"}) is False  # never


def test_a_mode_whose_result_does_not_go_to_stdout_for_these_parameters_has_no_format() -> None:
    fmt = spec.Option("-f", "format", spec.Kind.ENUM, "the format", default="json",
                      choices=("json", "ply"))
    mode = spec.Mode(None, None, (), "never", "needs nothing", (Stage.SETUP,), (
        spec.Output("result", "stdout", "ply", "the cloud", (spec.When("format", ("ply",)),)),))
    cmd = spec.Command("cloud.sh", None, "a cloud", (fmt,), (mode,))
    prog = spec.Program("cloud.sh", "a cloud", (cmd,))
    op = web_ops.Operation(prog, cmd, mode)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(spec, "PROGRAMS", (*spec.PROGRAMS, prog))
        assert web_ops.result_format(op, {"format": "ply"}) == "ply"
        assert web_ops.result_format(op, {}) is None  # json: written nowhere on stdout
    assert web_ops.media_of(None) == "application/octet-stream"
