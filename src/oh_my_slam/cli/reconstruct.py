"""``reconstruct.sh -i <image> [-f json|ply]`` — single-image reconstruction to stdout.

    json (default)  ASAM OpenLABEL scene: objects, labels, scores, colours, OBBs (camera frame)
    ply             binary PLY point cloud (xyz float32 metres, camera frame; rgb uchar)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from oh_my_slam.cli.common import ArgumentParser, run_main
from oh_my_slam.client.client import connect
from oh_my_slam.core import timing
from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.errors import InputError
from oh_my_slam.core.log import PayloadWriter, claim_stdout, get_logger, json_payload_bytes
from oh_my_slam.reconstruction.api import reconstruct_image
from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
from oh_my_slam.segmentation.cloud import cloud_ply, image_cloud_source
from oh_my_slam.segmentation.scene import single_image_scene

PROG = "reconstruct.sh"
log = get_logger("oh_my_slam.cli.reconstruct")


def build_parser() -> ArgumentParser:
    ap = ArgumentParser(prog=PROG, description="Single-image reconstruction (stdout).")
    ap.add_argument("-i", dest="image", required=True, type=Path, help="input RGB image")
    ap.add_argument("-f", dest="format", choices=("json", "ply"), default="json",
                    help="output format (default: json)")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    out = claim_stdout()
    if not args.image.is_file():
        raise InputError(f"image not found: {args.image}")
    with timing.collect() as tm:
        what = _run(args, out)
    timing.report(tm, log, command="reconstruct.sh", format=args.format,
                  image=str(args.image), **what)
    return 0


def _run(args: argparse.Namespace, out: PayloadWriter) -> dict[str, int]:
    """The command itself, timed as the stages connect / inference / segment / export / write."""
    stage = timing.stage
    t0 = time.perf_counter()
    with stage("connect"):
        client = connect()
    if args.format == "ply":
        with stage("inference"):
            frame = reconstruct_image(args.image, want_gravity=False, client=client)
        with stage("export"):
            data = cloud_ply(image_cloud_source(frame), CloudAttrs())
        with stage("write"):
            out.write_bytes(data)
        log.info("%d bytes of PLY in %.2f s", len(data), time.perf_counter() - t0)
        return {"ply_bytes": len(data)}
    with stage("inference"):
        frame, dets = reconstruct_and_detect(args.image, client)
    with stage("segment"):
        seg = segment_frame(frame, client=client, detections=dets)
    with stage("export"):
        data = json_payload_bytes(single_image_scene(seg, tool="reconstruct"))
    with stage("write"):
        out.write_bytes(data)
    log.info("%d objects in %.2f s", len(seg.objects), time.perf_counter() - t0)
    return {"objects": len(seg.objects), "detections": len(dets)}


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
