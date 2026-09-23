"""``segment.sh`` — instance segmentation of an image or of a persisted map.

    segment.sh -i <image> [-o <folder>] [-f json|ply] [--min-score <s>] [--labels a,b,c]
    segment.sh -m <map-folder> [-o <folder>] [-f json|ply]

stdout: the OpenLABEL scene (json, default) or the segment-coloured PLY. With ``-o`` the five
artefacts are written (segmentation.json identical to the JSON payload); without it nothing is
written. ``-m`` reads the map without modifying it and needs no inference server.
"""

from __future__ import annotations

import time
from pathlib import Path

from oh_my_slam.cli.common import ArgumentParser, parse_labels, run_main
from oh_my_slam.core import timing
from oh_my_slam.core.errors import InputError, UsageError
from oh_my_slam.core.log import claim_stdout, get_logger, json_payload_bytes
from oh_my_slam.core.ply import ply_bytes

PROG = "segment.sh"
log = get_logger("oh_my_slam.cli.segment")


def _score(value: str) -> float:
    try:
        s = float(value)
    except ValueError as exc:
        raise UsageError(f"--min-score must be a number, got {value!r}") from exc
    if not 0.0 <= s <= 1.0:
        raise UsageError("--min-score must be within [0, 1]")
    return s


def build_parser() -> ArgumentParser:
    ap = ArgumentParser(prog=PROG, description="Instance segmentation → JSON + OBBs, artefacts.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", dest="image", type=Path, help="input RGB image")
    src.add_argument("-m", dest="map", type=Path, help="existing map folder (read-only)")
    ap.add_argument("-o", dest="out", type=Path, help="write the five artefacts here")
    ap.add_argument("-f", dest="format", choices=("json", "ply"), default="json",
                    help="stdout format (default: json)")
    ap.add_argument("--min-score", dest="min_score", default=None,
                    help="drop detections below this calibrated score (default 0.5; -i only)")
    ap.add_argument("--labels", dest="labels", default=None,
                    help="comma-separated labels to keep (any words; -i only)")
    return ap


def _segment_image(args: object) -> tuple[bytes, bytes]:
    from oh_my_slam.client.client import connect
    from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
    from oh_my_slam.segmentation.artifacts import write_artifacts
    from oh_my_slam.segmentation.render import segmented_image
    from oh_my_slam.segmentation.scene import single_image_scene

    image: Path = args.image  # type: ignore[attr-defined]
    if not image.is_file():
        raise InputError(f"image not found: {image}")
    min_score = 0.5 if args.min_score is None else _score(args.min_score)  # type: ignore[attr-defined]
    labels = parse_labels(args.labels)  # type: ignore[attr-defined]
    stage = timing.stage
    with stage("connect"):
        client = connect()
    with stage("inference"):
        frame, dets = reconstruct_and_detect(image, client, labels=labels, min_score=min_score)
    with stage("segment"):
        seg = segment_frame(frame, client=client, detections=dets)
    with stage("export"):
        scene = json_payload_bytes(single_image_scene(seg, tool="segment"))
        cloud = seg.segments_cloud()
        ply = ply_bytes(cloud, comment="oh-my-slam segments")
    if args.out is not None:  # type: ignore[attr-defined]
        with stage("artifacts"):
            write_artifacts(args.out, scene, segmented_image(frame.rgb, seg.label_map),  # type: ignore[attr-defined]
                            seg.objects, cloud, title=f"Objects in {image.name}")
    timing.count(objects=len(seg.objects), detections=len(dets))
    log.info("%d objects", len(seg.objects))
    return scene, ply


def _segment_map(args: object) -> tuple[bytes, bytes]:
    if args.min_score is not None or args.labels is not None:  # type: ignore[attr-defined]
        raise UsageError("-m exports the map's persistent objects; --min-score and --labels "
                         "apply to -i only")
    from oh_my_slam.mapping.export import map_segment_outputs

    return map_segment_outputs(args.map, args.out)  # type: ignore[attr-defined]


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    out = claim_stdout()
    t0 = time.perf_counter()
    if args.map is not None:
        scene, ply = _segment_map(args)
        out.write_bytes(ply if args.format == "ply" else scene)
        log.info("done in %.2f s", time.perf_counter() - t0)
        return 0
    with timing.collect() as tm:
        scene, ply = _segment_image(args)
        with timing.stage("write"):
            out.write_bytes(ply if args.format == "ply" else scene)
    log.info("done in %.2f s", time.perf_counter() - t0)
    timing.report(tm, log, command="segment.sh -i", format=args.format, image=str(args.image),
                  artifacts=args.out is not None)
    return 0


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
