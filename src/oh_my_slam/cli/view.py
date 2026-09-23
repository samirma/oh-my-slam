"""``view.sh`` — local browser visualisation.

    view.sh -i <image>        reconstruct + segment one image (needs the inference server)
    view.sh -m <map-folder>   open a persisted map read-only (no server needed)

Binds 127.0.0.1 on a free port (or --port), prints the URL on stderr, opens the default browser
unless --no-browser, and serves until Ctrl-C.
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

from oh_my_slam.cli.common import ArgumentParser, run_main
from oh_my_slam.core.errors import InputError
from oh_my_slam.core.log import claim_stdout, get_logger

PROG = "view.sh"
log = get_logger("oh_my_slam.cli.view")


def build_parser() -> ArgumentParser:
    ap = ArgumentParser(prog=PROG, description="Browser visualisation of an image or a map.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", dest="image", type=Path, help="RGB image to reconstruct and segment")
    src.add_argument("-m", dest="map", type=Path, help="map folder (opened read-only)")
    ap.add_argument("--port", type=int, default=0, help="port (default: any free port)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    claim_stdout()  # nothing goes to stdout; the URL is printed on stderr
    from oh_my_slam.viewer.bundle import image_bundle, map_bundle
    from oh_my_slam.viewer.server import serve, url_of

    if args.image is not None:
        if not args.image.is_file():
            raise InputError(f"image not found: {args.image}")
        from oh_my_slam.reconstruction.api import connect_server

        client = connect_server()  # exit 3 with the hint when the server is down
        log.info("reconstructing and segmenting %s …", args.image.name)
        bundle = image_bundle(args.image, client)
    else:
        bundle = map_bundle(args.map)
    httpd = serve(bundle, args.port)
    url = url_of(httpd)
    print(f"{PROG}: serving {bundle.mode} '{bundle.title}' at {url} (Ctrl-C to stop)",
          file=sys.stderr, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
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
