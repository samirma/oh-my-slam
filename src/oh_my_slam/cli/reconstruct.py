"""``reconstruct.sh -i <image> [-f json|depth|ply] [-o <file>] [-p <attrs>]`` — one-image
reconstruction.

    json (default)  ASAM OpenLABEL scene: objects, labels, scores, colours, OBBs (camera frame);
                    the segmentation code's objects, as ``segment.sh -i`` gives them by default
    depth           the depth image: a 16-bit PNG of the input's pixel size, metric depth along
                    the optical axis (``reconstruction.depthimage``)
    ply             point cloud (camera frame, metres) shaped by the ``-p`` point-cloud attributes

The result goes to stdout, or to ``-o <file>`` (stdout then stays empty). ``-p`` needs ``-f ply``
and is validated before the server is contacted. Each format runs only the inference it needs:
depth alone for ``-f depth``; for ``-f ply`` segmentation (with gravity, which its OBB fits need)
with ``color=segment`` or ``label=on``, gravity alone with ``color=height``.
"""

from __future__ import annotations

import argparse
import time

from oh_my_slam.commands import spec
from oh_my_slam.commands.parser import ArgumentParser, run_main
from oh_my_slam.core import timing
from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.log import PayloadWriter, claim_stdout, get_logger, json_payload_bytes
from oh_my_slam.core.timing import Stage
from oh_my_slam.reconstruction.api import connect_server, reconstruct_image
from oh_my_slam.reconstruction.cloud import cloud_ply
from oh_my_slam.reconstruction.depthimage import depth_png
from oh_my_slam.segmentation.api import image_cloud_source, reconstruct_and_detect, segment_frame
from oh_my_slam.segmentation.scene import single_image_scene

PROGRAM = spec.RECONSTRUCT
COMMAND = PROGRAM.command()
MODE = COMMAND.modes[0]
PROG = PROGRAM.prog
SCOPE = MODE.scope()
log = get_logger("oh_my_slam.cli.reconstruct")


def build_parser() -> ArgumentParser:
    return spec.build_parser(PROGRAM)


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    v = spec.validate(COMMAND, args, log.warning)  # -p, -o and -i, before the server is contacted
    out = claim_stdout(args.output)
    with timing.collect() as tm:
        what = _run(args, v.attrs, out)
    timing.report(tm, log, command=COMMAND.label(MODE), format=args.format,
                  image=str(args.image), **what)
    return 0


def _run(args: argparse.Namespace, attrs: CloudAttrs, out: PayloadWriter) -> dict[str, int]:
    """The command itself, timed as the stages connect / inference / segment / export / write."""
    stage = timing.stage
    t0 = time.perf_counter()
    with stage(Stage.CONNECT):
        client = connect_server()
    if args.format == "depth":
        with stage(Stage.INFERENCE):
            frame = reconstruct_image(args.image, want_gravity=False, client=client)
        with stage(Stage.EXPORT):
            data = depth_png(frame)
        with stage(Stage.WRITE):
            out.write_bytes(data)
        log.info("depth image (%dx%d) in %.2f s", frame.intrinsics.width, frame.intrinsics.height,
                 time.perf_counter() - t0)
        return {"png_bytes": len(data)}
    if args.format == "ply":
        segmented = attrs.color == "segment" or attrs.label
        with stage(Stage.INFERENCE):
            if segmented:
                frame, dets = reconstruct_and_detect(args.image, client)
            else:
                frame = reconstruct_image(args.image, want_gravity=attrs.color == "height",
                                          client=client)
        seg = None
        if segmented:
            with stage(Stage.SEGMENT):
                seg = segment_frame(frame, client=client, detections=dets)
        with stage(Stage.EXPORT):
            # the reconstruction code derives the cloud; segmentation gives the point labels
            # (when it ran) and the colour assignment, as data
            data = cloud_ply(image_cloud_source(frame, seg), attrs)
        with stage(Stage.WRITE):
            out.write_bytes(data)
        log.info("PLY (%s) in %.2f s", attrs.describe(SCOPE), time.perf_counter() - t0)
        return {"ply_bytes": len(data)}
    with stage(Stage.INFERENCE):
        frame, dets = reconstruct_and_detect(args.image, client)
    with stage(Stage.SEGMENT):
        seg = segment_frame(frame, client=client, detections=dets)
    with stage(Stage.EXPORT):
        data = json_payload_bytes(single_image_scene(seg, tool="reconstruct"))
    with stage(Stage.WRITE):
        out.write_bytes(data)
    log.info("%d objects in %.2f s", len(seg.objects), time.perf_counter() - t0)
    return {"objects": len(seg.objects), "detections": len(dets)}


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
