"""``view.sh`` — local browser visualisation.

    view.sh -i <image>        reconstruct + segment one image (needs the inference server)
    view.sh -m <map-folder>   open a persisted map read-only (no server needed)

Binds 127.0.0.1 on a free port, opens the default browser unless --no-browser, and serves until
Ctrl-C or SIGTERM (both exit 0). Nothing is written to stdout. Once the server accepts
connections, stderr carries exactly one line of the form (``URL_LINE``)::

    view.sh: listening on http://127.0.0.1:<port>/

The page sets ``<body data-rendered="true">`` once its first frame with the point cloud has been
drawn (and keeps it), or ``data-error="<message>"`` if loading fails.
"""

from __future__ import annotations

import re
import signal
import sys
import webbrowser

from oh_my_slam.cli.common import ArgumentParser, run_main
from oh_my_slam.commands import spec
from oh_my_slam.core import timing
from oh_my_slam.core.log import claim_stdout, get_logger
from oh_my_slam.core.timing import Stage

PROGRAM = spec.VIEW
COMMAND = PROGRAM.command()
PROG = PROGRAM.prog
URL_LINE = re.compile(r"^view\.sh: listening on (http://127\.0\.0\.1:\d+/)$")
log = get_logger("oh_my_slam.cli.view")


def _stop_on_signal(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _install_stop_handlers() -> None:
    """Ctrl-C and SIGTERM are the normal stop. A process started as a shell background job
    inherits SIGINT as ignored, and Python then installs no handler of its own, so both are set
    explicitly."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _stop_on_signal)


def build_parser() -> ArgumentParser:
    return spec.build_parser(PROGRAM)


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    v = spec.validate(COMMAND, args, log.warning)  # -i exists or -m is a map, before any work
    claim_stdout()  # nothing goes to stdout; the URL is printed on stderr
    from oh_my_slam.viewer.bundle import image_bundle, map_bundle
    from oh_my_slam.viewer.server import serve, url_of

    if args.image is not None:
        from oh_my_slam.reconstruction.api import connect_server

        # timed as view.sh -i's stages (progress events only: no summary line, stderr unchanged)
        with timing.collect():
            with timing.stage(Stage.CONNECT):
                client = connect_server()  # exit 3 with the hint when the server is down
            log.info("reconstructing and segmenting %s …", args.image.name)
            with timing.stage(Stage.INFERENCE):
                bundle = image_bundle(args.image, client)
    else:
        bundle = map_bundle(args.map, reader=v.reader)  # the reader the map rule opened
    _install_stop_handlers()
    httpd = serve(bundle)
    try:
        url = url_of(httpd)
        print(f"{PROG}: listening on {url}", file=sys.stderr, flush=True)
        print(f"{PROG}: showing {bundle.mode} '{bundle.title}' (Ctrl-C to stop)", file=sys.stderr,
              flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
