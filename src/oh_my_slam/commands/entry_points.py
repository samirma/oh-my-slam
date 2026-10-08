"""Every entry point at the root of the checkout, in the order of the spec's table (§2): the commands
of ``commands.spec`` and the two long-lived servers' own, ``start_inference_server.sh`` (§2.1) and
``server.sh`` (§2.6), declared here once as data with the same types (options, defaults, modes). Each
server builds its parser from here (``spec.build_parser``), and the agent skill
(``oh_my_slam.web.skill``) takes from :func:`scripts` the servers it tells the user to start.
The servers are no API operations (``Program.service`` is False) and stay out of
``spec.PROGRAMS``, the commands whose parsers, rules and stages the web service runs.

A server's modes are selected by a flag (``--status``, ``--stop``), the mode without a selector
being its start; ``Mode.lifecycle`` marks the modes that start or stop it, which the user runs."""

from __future__ import annotations

from pathlib import Path

from oh_my_slam.commands import spec
from oh_my_slam.commands.spec import Command, Kind, Mode, Option, Output, Program
from oh_my_slam.core.constants import DEFAULT_DATA, INFERENCE_SERVER_PROG
from oh_my_slam.core.errors import ExitCode, ServiceNotRunningError

_DOWN = f"exit {int(ExitCode.SERVER_UNAVAILABLE)}"  # --status of a server that does not run
_INFERENCE_HELP = "Start (idempotent), stop or query the inference server."

INFERENCE_SERVER = Program(INFERENCE_SERVER_PROG, _INFERENCE_HELP, (
    Command(INFERENCE_SERVER_PROG, None, _INFERENCE_HELP, (
        Option("--stop", "stop", Kind.FLAG, "stop the running server", default=False,
               group="action"),
        Option("--status", "status", Kind.FLAG, "print health JSON to stdout", default=False,
               group="action"),
    ), (
        Mode("start", None, (), "never",
             "starts the inference server in the background and returns once its models are "
             "loaded; a server that already runs is left as it is", (), (), lifecycle="starts"),
        Mode("status", "status", (), "required",
             f"prints the inference server's health; {_DOWN} when it is not running", (),
             (Output("result", "stdout", "json", "the inference server's health: its status "
                     "(loading, ready, error, stopping), device, precision and each model's "
                     "load state"),)),
        Mode("stop", "stop", (), "never",
             "stops the inference server; its socket and state file are removed", (), (),
             lifecycle="stops"),
    )),
), service=False)

_SERVICE_HELP = "Local web service: the commands as an HTTP API and a browser application."

WEB_SERVICE = Program("server.sh", _SERVICE_HELP, (
    Command("server.sh", None, _SERVICE_HELP, (
        Option("--port", "port", Kind.NUMBER, "port to bind on 0.0.0.0 (default: 0, a free port)",
               default=0, minimum=0, type=int, modes=("serve",)),
        Option("--data", "data", Kind.FOLDER_OUT,
               f"workspace folder for maps and uploads (default: {DEFAULT_DATA}/)",
               default=DEFAULT_DATA, type=Path),
        Option("--no-browser", "no_browser", Kind.FLAG,
               "do not open the default browser once listening", default=False,
               modes=("serve",)),
        Option("--status", "status", Kind.FLAG,
               "print the running service's health JSON on stdout", default=False,
               group="action"),
        Option("--stop", "stop", Kind.FLAG, "stop the running service", default=False,
               group="action"),
    ), (
        Mode("serve", None, (), "never",
             "serves the API and the web application until interrupted, also while the "
             "inference server is down", (),
             (Output("service", "browser", "html", "the web application and its API (URL on "
                     "stderr)"),), lifecycle="starts"),
        Mode("status", "status", (), "never",
             f"prints the running service's health; {_DOWN} when no service runs for the "
             "workspace", (),
             (Output("result", "stdout", "json", "the service's health: its URL, workspace, the "
                     "requests in progress and the inference server's state"),),
             errors=(ServiceNotRunningError,)),
        Mode("stop", "stop", (), "never",
             "stops the running service, interrupting the requests in progress", (), (),
             lifecycle="stops"),
    )),
), service=False)


def scripts() -> tuple[Program, ...]:
    """Every entry point, in the spec's order: the inference server, the commands
    (``spec.PROGRAMS``, read when called) and the web service."""
    return (INFERENCE_SERVER, *spec.PROGRAMS, WEB_SERVICE)
