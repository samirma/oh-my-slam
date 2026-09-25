"""The evaluation plan (high_level_spec.md §5): every entry point on the files in ``examples/``,
strictly one command at a time.

1. ``start_inference_server.sh``: stop, cold start, resident memory (the server's initial state is
   restored at the end).
2. ``restaurant.jpg``: ``reconstruct.sh`` (JSON, PLY, ``-o`` PLY coloured by segment),
   ``segment.sh -i`` (artefacts, ``-o`` PLY) and ``view.sh -i``.
3. Every ``ainex-captures`` frame: ``segment.sh -i``.
4. ``mapper.sh update``: the sequence in one update, and split across ``splits`` updates into a
   second map (first update ``-o`` JSON, middle ones ``-t single -f ply``, last ``-t full``).
5. On the one-update map: ``segment.sh -m`` (artefacts), ``view.sh -m``; ``segment.sh -m -f ply
   -o`` on the split map.

Every output is checked against the contracts; the metrics are computed from the outputs.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.core.images import load_rgb
from oh_my_slam.core.ply import parse_ply, read_ply
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping.store import full_tree_hash
from oh_my_slam.tools.evaluate import groundtruth as gt
from oh_my_slam.tools.evaluate.contracts import (
    ContractLog,
    artifact_problems,
    catalog_csv_problems,
    catalog_md_problems,
    cloud_colour_problems,
    parse_scene,
    payload_problems,
    png_colour_problems,
    same_objects_problems,
    scene_colour_problems,
    served_cloud_problems,
    subject_of,
)
from oh_my_slam.tools.evaluate.mapquality import (
    AGREEMENT_METRICS,
    DUPLICATE_METRIC,
    OUT_OF_BOX_METRICS,
    STABILITY_METRICS,
    agreement_metrics,
    duplicate_metrics,
    out_of_box_metrics,
    split_alignment,
    stability_metrics,
)
from oh_my_slam.tools.evaluate.memory import phys_footprint_gb, server_pid
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import Capture, captures_in
from oh_my_slam.tools.evaluate.performance import perf_ids, perf_metrics
from oh_my_slam.tools.evaluate.poses import (
    POSE_METRICS,
    capture_poses,
    capture_sources,
    pose_metrics,
)
from oh_my_slam.tools.evaluate.runner import REPO, Runner, RunRecord, RunSpec
from oh_my_slam.tools.evaluate.scene import DocObject, Json, doc_objects
from oh_my_slam.tools.evaluate.segmentation import (
    MAP_CONSISTENCY,
    MAP_CONSISTENCY_METRICS,
    detection_row,
    map_consistency,
)
from oh_my_slam.tools.evaluate.viewer import BrowserProbe

EXAMPLES = REPO / "examples"
RESTAURANT = "restaurant.jpg"
SEQUENCE = "ainex-captures"
GROUND_TRUTH = "ground_truth"
SERVER = "start_inference_server.sh"
LABELLED_SEGMENTS = "color=segment,label=on"  # per-point object ids: exact colour checks
MIN_SPLITS = 3
MAPS = ("single", "split")
SERVER_METRICS = ("perf.server.cold_start_s", "perf.server.resident_gb", "perf.server.peak_gb")
SEG_METRICS = ("seg.restaurant.objects", "seg.frames.with_detections_fraction", "seg.min_score")


def expected_ids() -> list[str]:
    """Every metric a run records (ground-truth metrics come on top when annotations exist)."""
    ids = [*SERVER_METRICS, *perf_ids(), *SEG_METRICS]
    ids += [f"{MAP_CONSISTENCY}.{k}" for k in MAP_CONSISTENCY_METRICS]
    ids += [f"pose.{mp}.{k}" for mp in MAPS for k in POSE_METRICS]
    ids += [f"map.{mp}.{k}" for mp in MAPS
            for k in (*AGREEMENT_METRICS, DUPLICATE_METRIC, *OUT_OF_BOX_METRICS)]
    ids += [f"map.stability.{k}" for k in STABILITY_METRICS]
    ids += [mid for mid, *_ in ContractLog().results()]
    return [*ids, "contract.exit_codes"]


def _problems_reading(fn: Any, *args: Any) -> list[str]:
    """``fn(*args)``, with a missing or unreadable file reported as a problem."""
    try:
        return list(fn(*args))
    except (OSError, ValueError, KeyError) as exc:
        return [f"{type(exc).__name__}: {exc}"]


@dataclass
class Evaluation:
    out: Path
    runner: Runner
    probe: BrowserProbe
    examples: Path = EXAMPLES
    splits: int = MIN_SPLITS
    metrics: Metrics = field(default_factory=Metrics)
    contracts: ContractLog = field(default_factory=ContractLog)
    details: dict[str, Any] = field(default_factory=dict)
    images: dict[str, list[DocObject]] = field(default_factory=dict)  # segment.sh -i objects
    single_poses: dict[str, Pose] | None = None
    was_running: bool = False

    # -- running and checking one command ------------------------------------------------------------

    def run(self, tag: str, group: str, entry: str, *args: str | Path, stdout: str = "json",
            output: Path | None = None, ok_exit: tuple[int, ...] = (0,),
            timeout_s: float = 3600.0) -> RunRecord:
        kind = None if output is None else output.suffix.lstrip(".")
        rec = self.runner.run(RunSpec(tag, group, entry, tuple(str(a) for a in args), stdout,
                                      output, kind, ok_exit, timeout_s))
        self.check_payloads(rec)
        return rec

    def check_payloads(self, rec: RunRecord) -> None:
        """stdout purity: one payload on success, nothing on failure (and nothing with ``-o``,
        whose file then holds exactly one payload)."""
        spec = rec.spec
        succeeded = rec.error is None and rec.exit_code == 0
        problems = payload_problems(rec.stdout_bytes(), spec.stdout if succeeded else "empty")
        if succeeded and spec.output is not None and spec.output_kind is not None:
            data = spec.output.read_bytes() if spec.output.exists() else b""
            problems += [f"-o file: {p}" for p in payload_problems(data, spec.output_kind)]
        self.contracts.check("stdout", subject_of(spec.entry), rec.tag, problems)

    def result_bytes(self, rec: RunRecord) -> bytes:
        """The run's result: its ``-o`` file (empty if missing) or its stdout."""
        out = rec.spec.output
        if out is None:
            return rec.stdout_bytes()
        return out.read_bytes() if out.exists() else b""

    def scene(self, rec: RunRecord) -> Json | None:
        """The OpenLABEL result of a successful run, checked for validity and colours."""
        if not rec.ok:
            return None
        subject = subject_of(rec.spec.entry)
        doc, problems = parse_scene(self.result_bytes(rec))
        self.contracts.check("openlabel", subject, rec.tag, problems)
        if doc is None:
            return None
        self.contracts.check("colour", subject, rec.tag, scene_colour_problems(doc_objects(doc)))
        return doc

    def cloud_colours(self, rec: RunRecord, ids: set[int] | None) -> None:
        """Colour contract of a ``color=segment`` PLY result."""
        if rec.ok:
            self.contracts.check("colour", subject_of(rec.spec.entry), rec.tag, _problems_reading(
                lambda: cloud_colour_problems(parse_ply(self.result_bytes(rec)), ids)))

    def artefacts(self, rec: RunRecord, folder: Path, doc: Json, sheet: bool) -> None:
        """The five ``-d`` artefacts: exactly those files, the JSON identical to stdout, and every
        colour in them the object colours of ``doc``."""
        objs = doc_objects(doc)
        ids = {o.id for o in objs}
        self.contracts.check("artifacts", "segment", rec.tag,
                             _problems_reading(artifact_problems, folder, rec.stdout_bytes()))
        problems = _problems_reading(
            lambda: png_colour_problems(load_rgb(folder / "segmented.png"), objs, sheet))
        problems += _problems_reading(
            lambda: catalog_csv_problems((folder / "catalog.csv").read_text(), objs))
        problems += _problems_reading(
            lambda: catalog_md_problems((folder / "catalog.md").read_text(), objs))
        problems += _problems_reading(
            lambda: cloud_colour_problems(read_ply(folder / "segments.ply"), ids))
        self.contracts.check("colour", "segment", f"{rec.tag}/artefacts", problems)

    def view(self, tag: str, source: tuple[str, str, Json | None], *args: str | Path) -> None:
        """``view.sh`` with ``args``: time to the rendered page, and the scene it draws (OBB
        colours) checked against the contracts and against ``source`` = (same-objects subject,
        what the scene is compared with, that output's scene)."""
        spec = RunSpec(tag, tag, "view.sh", (*(str(a) for a in args), "--no-browser"),
                       stdout="empty", ok_exit=(0, 130), timeout_s=900.0)
        seen = self.probe.measure(self.runner, spec)
        rec = seen.record
        if seen.error is not None:
            rec.notes["render_error"] = seen.error
        self.check_payloads(rec)
        self.details[f"viewer.{tag}"] = {"url": seen.url, "render_s": seen.render_s,
                                         "error": seen.error,
                                         "console_errors": (seen.console_errors or [])[:10]}
        if seen.console_errors is not None:  # the page was opened
            self.contracts.check("console_errors", "view", tag, seen.console_errors)
        if seen.scene is None:
            return
        doc, problems = parse_scene(seen.scene)
        self.contracts.check("openlabel", "view", tag, problems)
        if doc is None:
            return
        objs = doc_objects(doc)
        self.contracts.check("colour", "view", tag, scene_colour_problems(objs))
        self.contracts.check("colour", "view", f"{tag}/cloud color=segment",
                             served_cloud_problems(seen.cloud, {o.id for o in objs}))
        subject, name, other = source
        if other is not None:
            self.contracts.check("same_objects", subject, f"view.sh vs {name}",
                                 same_objects_problems(doc_objects(other), objs,
                                                       geometry=subject == "map"))

    # -- sections --------------------------------------------------------------------------------------

    def server_start(self) -> None:
        first = self.run("server_status_initial", "server", SERVER, "--status", ok_exit=(0, 3))
        self.was_running = first.exit_code == 0
        self.run("server_stop", "server", SERVER, "--stop", stdout="empty")
        cold = self.run("server_cold_start", "server", SERVER, stdout="empty", timeout_s=1800.0)
        if cold.ok:
            pid = server_pid()
            fp = phys_footprint_gb(pid) if pid is not None else None
            self.metrics.add(SERVER_METRICS[0], cold.wall_s)
            self.metrics.add(SERVER_METRICS[1], None if fp is None else fp[0], {"pid": pid},
                             error="server footprint unavailable")
        else:
            self.metrics.fail(SERVER_METRICS[:2], cold.failure())
        status = self.run("server_status", "server", SERVER, "--status")
        if status.ok:
            self.details["server_health"] = json.loads(status.stdout_bytes())

    def server_restore(self) -> None:
        pid = server_pid()
        fp = phys_footprint_gb(pid) if pid is not None else None
        self.metrics.add(SERVER_METRICS[2], None if fp is None else fp[1], {"pid": pid},
                         error="the server is not running at the end of the evaluation")
        if self.was_running and pid is None:
            self.run("server_restart", "server", SERVER, stdout="empty", timeout_s=1800.0)
        elif not self.was_running:
            self.run("server_stop_final", "server", SERVER, "--stop", stdout="empty")

    def restaurant(self) -> None:
        img = self.examples / RESTAURANT
        outputs = self.out / "outputs"
        self.scene(self.run("reconstruct_warmup", "warmup", "reconstruct.sh", "-i", img))
        recon = self.scene(self.run("reconstruct_json", "reconstruct_json", "reconstruct.sh",
                                    "-i", img))
        self.run("reconstruct_ply", "reconstruct_ply", "reconstruct.sh", "-i", img, "-f", "ply",
                 stdout="ply")
        rec = self.run("reconstruct_segment_ply", "contracts", "reconstruct.sh", "-i", img,
                       "-f", "ply", "-p", LABELLED_SEGMENTS, "-o",
                       outputs / "reconstruct_segment.ply", stdout="empty",
                       output=outputs / "reconstruct_segment.ply")
        self.cloud_colours(rec, None if recon is None else {o.id for o in doc_objects(recon)})
        folder = outputs / "segment_image"
        rec = self.run("segment_image", "segment_image", "segment.sh", "-i", img, "-d", folder,
                       "-p", "label=on")
        seg = self.scene(rec)
        with self.metrics.expect(SEG_METRICS[0]):
            if seg is None:
                self.metrics.fail(SEG_METRICS[:1], rec.failure())
            else:
                self.artefacts(rec, folder, seg, sheet=False)
                objs = self.images[RESTAURANT] = doc_objects(seg)
                self.details["segmentation.restaurant"] = detection_row(RESTAURANT, objs)
                self.metrics.add(SEG_METRICS[0], len(objs))
        if recon is not None and seg is not None:
            self.contracts.check("same_objects", "image", "reconstruct.sh vs segment.sh -i",
                                 same_objects_problems(doc_objects(recon), doc_objects(seg),
                                                       geometry=False))
        rec = self.run("segment_image_ply", "contracts", "segment.sh", "-i", img, "-f", "ply",
                       "-o", outputs / "segment_image.ply", stdout="empty",
                       output=outputs / "segment_image.ply")
        self.cloud_colours(rec, None if seg is None else {o.id for o in doc_objects(seg)})
        self.view("view_image", ("image", "segment.sh -i", seg), "-i", img)

    def frames(self, captures: list[Capture]) -> None:
        rows: list[dict[str, Any]] = []
        for c in captures:
            rec = self.run(f"segment_frame_{c.index:03d}", "segment_frames", "segment.sh", "-i",
                           self.examples / SEQUENCE / c.name, timeout_s=600.0)
            doc = self.scene(rec)
            if doc is None:
                rows.append({"image": c.name, "error": rec.failure()})
                continue
            objs = self.images[f"{SEQUENCE}/{c.name}"] = doc_objects(doc)
            rows.append({**detection_row(c.name, objs), "wall_s": round(rec.wall_s, 3)})
        self.details["segmentation.frames"] = rows
        done = [r for r in rows if "error" not in r]
        failed = [r["error"] for r in rows if "error" in r]
        self.metrics.add(SEG_METRICS[1], sum(r["objects"] > 0 for r in done) / len(done)
                         if done else None, {"segmented": len(done), "frames": len(rows)},
                         error=f"no frame was segmented ({failed[0] if failed else 'no frames'})")

    def build_maps(self, captures: list[Capture]) -> tuple[Json | None, Json | None]:
        """The one-update map and the split map; their ``-t full`` scenes."""
        single_dir, split_dir = self.out / "maps" / "single", self.out / "maps" / "split"
        single = self.scene(self.run("mapper_single", "mapper_single", "mapper.sh", "update",
                                     "-a", self.examples / SEQUENCE, "-m", single_dir,
                                     "-t", "full"))
        split = None
        parts = np.array_split(np.arange(len(captures)), self.splits)
        for k, idx in enumerate(parts, start=1):
            tag = f"mapper_split_{k}"
            args: tuple[str | Path, ...] = (
                "update", "-a", *(self.examples / SEQUENCE / captures[i].name for i in idx),
                "-m", split_dir)
            if k == 1:
                target = self.out / "outputs" / f"{tag}.json"
                self.scene(self.run(tag, "mapper_split", "mapper.sh", *args, "-t", "full",
                                    "-o", target, stdout="empty", output=target))
            elif k < len(parts):
                self.cloud_colours(self.run(tag, "mapper_split", "mapper.sh", *args, "-t",
                                            "single", "-f", "ply", "-p", LABELLED_SEGMENTS,
                                            stdout="ply"), None)
            else:
                split = self.scene(self.run(tag, "mapper_split", "mapper.sh", *args, "-t", "full"))
        return single, split

    def map_metrics(self, captures: list[Capture], single: Json | None, split: Json | None
                    ) -> None:
        dirs = {"single": self.out / "maps" / "single", "split": self.out / "maps" / "split"}
        poses: dict[str, dict[str, Pose]] = {}
        for name, doc in (("single", single), ("split", split)):
            ids = [f"pose.{name}.{k}" for k in POSE_METRICS]
            with self.metrics.expect(*ids):
                if doc is None:
                    self.metrics.fail(ids, f"the {name} map was not built")
                else:
                    poses[name] = capture_poses(doc, dirs[name])
                    self.details[f"poses.{name}"] = pose_metrics(
                        self.metrics, f"pose.{name}", poses[name], captures)
            ids = [f"map.{name}.{k}" for k in AGREEMENT_METRICS]
            with self.metrics.expect(*ids):
                if doc is None:
                    self.metrics.fail(ids, f"the {name} map was not built")
                else:
                    self.details[f"map.{name}.pairs"] = agreement_metrics(
                        self.metrics, f"map.{name}", dirs[name], captures)
            mid = f"map.{name}.{DUPLICATE_METRIC}"
            with self.metrics.expect(mid):
                if doc is None:
                    self.metrics.fail([mid], f"the {name} map was not built")
                else:
                    self.details[f"map.{name}.duplicates"] = duplicate_metrics(
                        self.metrics, f"map.{name}", doc_objects(doc))
            ids = [f"map.{name}.{k}" for k in OUT_OF_BOX_METRICS]
            with self.metrics.expect(*ids):
                if doc is None:
                    self.metrics.fail(ids, f"the {name} map was not built")
                else:
                    self.details[f"map.{name}.out_of_box"] = out_of_box_metrics(
                        self.metrics, f"map.{name}", dirs[name], doc_objects(doc))
        self.single_poses = poses.get("single")
        ids = [f"map.stability.{k}" for k in STABILITY_METRICS]
        with self.metrics.expect(*ids):
            if single is None or split is None:
                self.metrics.fail(ids, "both maps are needed")
            else:
                self.details["map.stability"] = stability_metrics(
                    self.metrics, "map.stability", doc_objects(single), doc_objects(split),
                    split_alignment(poses["single"], poses["split"]))
        ids = [f"{MAP_CONSISTENCY}.{k}" for k in MAP_CONSISTENCY_METRICS]
        with self.metrics.expect(*ids):
            if single is None:
                self.metrics.fail(ids, "the single map was not built")
            else:
                frames = {c.name: self.images[f"{SEQUENCE}/{c.name}"] for c in captures
                          if f"{SEQUENCE}/{c.name}" in self.images}
                self.details["segmentation.map"] = map_consistency(
                    self.metrics, MAP_CONSISTENCY, frames, doc_objects(single),
                    capture_sources(single, dirs["single"]))

    def map_commands(self, single: Json | None, split: Json | None) -> None:
        single_dir, split_dir = self.out / "maps" / "single", self.out / "maps" / "split"
        before = full_tree_hash(single_dir) if single is not None else None
        folder = self.out / "outputs" / "segment_map"
        rec = self.run("segment_map", "segment_map", "segment.sh", "-m", single_dir, "-d", folder,
                       "-p", "label=on")
        doc = self.scene(rec)
        if doc is not None:
            self.artefacts(rec, folder, doc, sheet=True)
            if single is not None:
                self.contracts.check("same_objects", "map", "segment.sh -m vs mapper.sh -t full",
                                     same_objects_problems(doc_objects(single), doc_objects(doc),
                                                           geometry=True))
        target = self.out / "outputs" / "segment_map_split.ply"
        rec = self.run("segment_map_ply", "contracts", "segment.sh", "-m", split_dir, "-f", "ply",
                       "-o", target, stdout="empty", output=target)
        self.cloud_colours(rec, None if split is None else {o.id for o in doc_objects(split)})
        self.view("view_map", ("map", "mapper.sh -t full", single), "-m", single_dir)
        if before is not None:
            same = full_tree_hash(single_dir) == before
            self.contracts.check("readonly", "map", "segment.sh -m, view.sh -m",
                                 [] if same else ["the map folder changed"])

    def ground_truth(self) -> None:
        files, skipped = gt.discover(self.examples / GROUND_TRUTH)
        gt.object_metrics(self.metrics, [f for f in files if f.kind == "objects"], self.images,
                          skipped)
        gt.pose_metrics(self.metrics, [f for f in files if f.kind == "poses"], self.single_poses)
        self.details["ground_truth"] = {"files": [str(f.path) for f in files],
                                        "skipped": skipped}

    # -- all -------------------------------------------------------------------------------------------

    def section(self, name: str, fn: Any, *args: Any) -> Any:
        """``fn(*args)``; an unexpected error is logged and ends only this section (its metrics
        then fail as not computed)."""
        print(f"== {name}", file=sys.stderr, flush=True)
        try:
            return fn(*args)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self.details.setdefault("errors", []).append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    def run_all(self) -> None:
        captures = captures_in(self.examples / SEQUENCE)
        self.section("inference server", self.server_start)
        try:
            self.section("restaurant.jpg", self.restaurant)
            self.section(f"{SEQUENCE}: segment.sh -i per frame", self.frames, captures)
            maps = self.section(f"{SEQUENCE}: mapper.sh update", self.build_maps, captures)
            single, split = maps or (None, None)
            self.section("pose accuracy, map quality", self.map_metrics, captures, single, split)
            self.section("segment.sh -m, view.sh -m", self.map_commands, single, split)
        finally:
            self.section("restore the inference server", self.server_restore)
        self.section("summary", self.summarise)
        self.metrics.fail(expected_ids(), "not computed (an earlier step failed; see errors)")

    def summarise(self) -> None:
        m = self.metrics
        with m.expect(*perf_ids()):
            perf_metrics(m, self.runner.records)
        with m.expect(SEG_METRICS[2]):
            scores = [o.score for objs in self.images.values() for o in objs
                      if o.score is not None]
            m.add(SEG_METRICS[2], min(scores) if scores else None, error="no detections")
        for mid, value, detail, error in self.contracts.results():
            m.add(mid, value, detail, error)
        bad = [r.failure() for r in self.runner.records if not r.ok]
        m.add("contract.exit_codes", len(bad), {"runs": len(self.runner.records), "failed": bad})
        self.ground_truth()
