"""``mapper.sh update`` — build and update a persistent map; ``mapper.sh locate`` — the camera pose
of images in an existing map, which it never modifies.

    mapper.sh update -i <image(s)|folder(s)|video> -m <folder> [-f json|ply] [-o <file>]
                     [-p <attrs>] [-t full|single] [-fps <n>]

Creates the map if <folder> is missing or empty, extends it if it is a map, refuses any other
non-empty folder (exit 4). Only -i and -m are required. The result goes to stdout, or to
``-o <file>`` (stdout then stays empty): the OpenLABEL scene (json, default) of the whole map
(-t full, default, with every keyframe pose) or of the new input only (-t single); -f ply writes
the map cloud (full) or the new frames' points (single), map coordinates, shaped by the ``-p``
point-cloud attributes (map scope: the pixel-level keys are refused). -fps (default 2) samples
video input and is ignored for images. Options are validated before the server is contacted.

    mapper.sh locate -i <image(s)> -m <map> [-f json|ply] [-o <file>] [-p <attrs>] [-t full|single]

Registers each image against the whole map (``mapping.locate``). -t single (default): the located
camera poses only (PLY: the map points visible from them); -t full: the map's scene as ``update -t
full`` returns it plus the located poses (PLY: the whole map cloud). A video, a missing or empty
-m folder, and an -o inside the map are refused (exit 2); a non-empty folder that is not a map exit
4; no image located exit 2. The inference server is needed only for retrieval in maps of more than
150 keyframes (exit 3 when it is down).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oh_my_slam.cli.common import (
    ArgumentParser,
    add_result_options,
    attrs_help,
    cloud_attrs_arg,
    run_main,
)
from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope
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
    lo = sub.add_parser("locate", help="camera pose of images in an existing map (read-only)")
    lo.add_argument("-i", dest="inputs", nargs="+", type=Path, required=True,
                    help="one or more image files (a video is refused)")
    lo.add_argument("-m", dest="map", type=Path, required=True, help="existing map folder")
    lo.add_argument("-f", dest="format", choices=("json", "ply"), default="json",
                    help="output format (default: json)")
    add_result_options(lo, attrs_help(SCOPE, "requires -f ply"))
    lo.add_argument("-t", dest="mode", choices=("full", "single"), default="single",
                    help="single = the located camera poses only; full = the whole map plus the "
                         "located poses (default: single)")
    return ap


def main(argv: list[str]) -> int:
    from oh_my_slam.core.errors import UsageError
    from oh_my_slam.core.images import VIDEO_SUFFIXES
    from oh_my_slam.mapping.ingest import DEFAULT_FPS

    args = build_parser().parse_args(argv)
    attrs = cloud_attrs_arg(args.attrs, SCOPE, writes_ply=args.format == "ply",
                            requires="only the PLY output has: use -f ply")
    if args.command == "locate":
        return _locate(args, attrs)
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


def _locate(args: argparse.Namespace, attrs: CloudAttrs) -> int:
    from oh_my_slam.core import timing
    from oh_my_slam.mapping.locate import check_output, locate, open_map, resolve_images

    # inputs, map and -o are checked before -o is prepared (which creates its folder)
    resolve_images(args.inputs)
    open_map(args.map)
    check_output(args.map, args.output)
    out = claim_stdout(args.output)
    res = locate(args.map, args.inputs, mode=args.mode, fmt=args.format, attrs=attrs)
    out.write_bytes(res.payload)
    timing.report(res.timings, log, command="mapper.sh locate", mode=args.mode,
                  format=args.format, map=str(args.map))
    return 0


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
