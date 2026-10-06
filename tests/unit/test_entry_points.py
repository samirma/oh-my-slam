"""Every entry point's shared definitions (``commands.entry_points``): the two servers' parsers are
built from them, each server mode is selected by its flag (the start by none), the modes declare
what they start or stop and how they fail, the list of scripts is the checkout's six in the spec's
order, every exit status has a meaning, and the workspace default is one value."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from oh_my_slam.cli import server as cli_server
from oh_my_slam.commands import entry_points, spec
from oh_my_slam.commands.entry_points import INFERENCE_SERVER, WEB_SERVICE
from oh_my_slam.core import constants
from oh_my_slam.core.errors import MEANING, ExitCode, ServiceNotRunningError, UsageError
from oh_my_slam.web import main as web_main
from oh_my_slam.web import workspace

REPO = Path(__file__).resolve().parents[2]


def _options(ap: argparse.ArgumentParser) -> list[argparse.Action]:
    return [a for a in ap._actions if a.option_strings and a.dest != "help"]


def test_each_servers_parser_is_built_from_the_registry() -> None:
    for program, ap in ((INFERENCE_SERVER, cli_server.build_parser()),
                        (WEB_SERVICE, web_main.build_parser())):
        (cmd,) = program.commands
        assert ap.prog == program.prog and ap.description == program.description
        assert not program.service  # no API operation
        actions = _options(ap)
        assert [a.option_strings for a in actions] == [[o.flag] for o in cmd.options]
        assert [(a.dest, a.help) for a in actions] == [(o.name, o.help) for o in cmd.options]
    progs = (cli_server.PROG, web_main.PROG)
    assert progs == (INFERENCE_SERVER.prog, WEB_SERVICE.prog)


@pytest.mark.parametrize(("program", "argv", "mode"), [
    (INFERENCE_SERVER, [], "start"), (INFERENCE_SERVER, ["--status"], "status"),
    (INFERENCE_SERVER, ["--stop"], "stop"), (WEB_SERVICE, ["--port", "8", "--no-browser"], "serve"),
    (WEB_SERVICE, ["--data", "d", "--status"], "status"), (WEB_SERVICE, ["--stop"], "stop")])
def test_a_servers_mode_is_selected_by_its_flag(program: spec.Program, argv: list[str],
                                                mode: str) -> None:
    (cmd,) = program.commands
    assert cmd.mode_of(spec.build_parser(program).parse_args(argv)).name == mode


def test_a_commands_mode_is_still_selected_by_its_value() -> None:
    (view,) = spec.VIEW.commands
    assert view.mode_of(argparse.Namespace(image=Path("a.jpg"), map=None)).name == "image"
    assert view.mode_of(argparse.Namespace(image=None, map=Path("m"))).name == "map"
    with pytest.raises(UsageError, match="one of the arguments -i -m is required"):
        view.mode_of(argparse.Namespace(image=None, map=None))


def test_what_the_servers_modes_do_and_how_they_fail() -> None:
    def codes(m: spec.Mode) -> set[str]:
        return {e["code"] for e in spec.errors_of(m)}

    start, status, stop = INFERENCE_SERVER.commands[0].modes
    assert (start.lifecycle, status.lifecycle, stop.lifecycle) == ("starts", None, "stops")
    assert codes(start) == codes(stop) == {"usage"}
    assert codes(status) == {"usage", "server_unavailable"}  # the inference server's own error
    serve, w_status, w_stop = WEB_SERVICE.commands[0].modes
    assert (serve.lifecycle, w_status.lifecycle, w_stop.lifecycle) == ("starts", None, "stops")
    assert codes(w_status) == {"usage", "server_unavailable"}  # no service runs: exit 3
    assert ServiceNotRunningError in w_status.errors
    assert ServiceNotRunningError.exit_code is ExitCode.SERVER_UNAVAILABLE
    (cmd,) = WEB_SERVICE.commands
    assert [o.flag for o in cmd.mode_options(w_status)] == ["--data", "--status"]
    assert [o.flag for o in cmd.mode_options(serve)] == ["--port", "--data", "--no-browser"]


def test_every_entry_point_is_listed_once_in_the_specs_order() -> None:
    progs = [p.prog for p in entry_points.scripts()]
    assert progs == [INFERENCE_SERVER.prog, *(p.prog for p in spec.PROGRAMS), WEB_SERVICE.prog]
    assert len(set(progs)) == len(progs) == 6
    assert set(progs) == {f.name for f in REPO.glob("*.sh")}


def test_every_exit_status_has_a_meaning() -> None:
    assert set(MEANING) == set(ExitCode) and all(MEANING.values())


def test_the_workspace_default_is_one_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                            ) -> None:
    assert str(workspace.DEFAULT_DATA) == constants.DEFAULT_DATA
    data = WEB_SERVICE.commands[0].option("data")
    assert data.default == constants.DEFAULT_DATA and constants.DEFAULT_DATA in data.help
    monkeypatch.setenv("HOME", str(tmp_path))  # server.sh --status without --data: the default
    with pytest.raises(ServiceNotRunningError, match=str(tmp_path / "oh-my-slam-data")):
        web_main.main(["--status"])
