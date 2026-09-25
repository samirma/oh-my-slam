"""``mapper.sh update`` — build and update a persistent map.

    mapper.sh update -i <image(s)|folder(s)|video> -m <folder> [-f json|ply] [-o <file>]
                     [-p <attrs>] [-t full|single] [-fps <n>]

Creates the map if <folder> is missing or empty, extends it if it is a map, refuses any other
non-empty folder (exit 4). Only -i and -m are required. The result goes to stdout, or to
``-o <file>`` (stdout then stays empty): the OpenLABEL scene (json, default) of the whole map
(-t full, default, with every keyframe pose) or of the new input only (-t single); -f ply writes
the map cloud (full) or the new frames' points (single), map coordinates, shaped by the ``-p``
point-cloud attributes (map scope: the pixel-level keys are refused). -fps (default 2) samples
video input and is ignored for images. Options are validated before the server is contacted.
"""

from __future__ import annotations

from pathlib import Path

from oh_my_slam.cli.common import (
    ArgumentParser,
    add_result_options,
    attrs_help,
    cloud_attrs_arg,
    run_main,
)
from oh_my_slam.core.cloud_attrs import CloudScope
from oh_my_slam.core.log import claim_stdout, get_logger

PROG = "mapper.sh"
SCOPE = CloudScope.MAP
log = get_logger("oh_my_slam.cli.mapper")


def build_parser() -> ArgumentParser:
    ap = ArgumentParser(prog=PROG, description="Multi-frame mapping (persistent map).")
    sub = ap.add_subparsers(dest="command", required=True, parser_class=ArgumentParser)
    up = sub.add_parser("update", help="create or extend a map")
    up.add_argument("-i", dest="inputs", nargs="+", type=Path, required=True,
                    help="image files, image folders, or exactly one video")
    up.add_argument("-m", dest="map", type=Path, required=True, help="map folder")
    up.add_argument("-f", dest="format", choices=("json", "ply"), default="json",
                    help="output format (default: json)")
    add_result_options(up, attrs_help(SCOPE, "requires -f ply"))
    up.add_argument("-t", dest="mode", choices=("full", "single"), default="full",
                    help="full = whole map with all keyframe poses; single = new input only "
                         "(default: full)")
    up.add_argument("-fps", dest="fps", type=float, default=None,
                    help="video frames per second to sample (default: 2; ignored for images)")
    return ap


def main(argv: list[str]) -> int:
    from oh_my_slam.core.errors import UsageError
    from oh_my_slam.core.images import VIDEO_SUFFIXES
    from oh_my_slam.mapping.ingest import DEFAULT_FPS

    args = build_parser().parse_args(argv)
    attrs = cloud_attrs_arg(args.attrs, SCOPE, writes_ply=args.format == "ply",
                            requires="only the PLY output has: use -f ply")
    out = claim_stdout(args.output)
    is_video = len(args.inputs) == 1 and args.inputs[0].suffix.lower() in VIDEO_SUFFIXES
    fps = DEFAULT_FPS if args.fps is None else args.fps
    if args.fps is not None and not is_video:
        log.warning("-fps applies to video input only; ignored for images")
    if fps <= 0:
        raise UsageError("-fps must be positive")
    from oh_my_slam.core import timing
    from oh_my_slam.mapping.api import update

    res = update(args.map, args.inputs, fps=fps, mode=args.mode, fmt=args.format, attrs=attrs)
    out.write_bytes(res.payload)
    timing.report(res.timings, log, command="mapper.sh update", mode=args.mode,
                  format=args.format, fps=fps if is_video else None, map=str(args.map))
    return 0


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
