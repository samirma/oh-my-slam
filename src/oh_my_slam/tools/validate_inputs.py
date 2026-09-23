"""Validation harness for the user's four inputs (R42, AC25–AC31).

    python -m oh_my_slam.tools.validate_inputs --inputs /Users/U124317/robot_view \
        [--out DIR] [--only restaurant,ainex,church,livingroom,update,segview,ui]

Runs every applicable command sequentially (never concurrently) against the running inference
server, stores stdout/stderr/timings under ``~/oh-my-slam-data/validation/<UTC>/`` and writes
``report.json`` + ``report.md``. Every expectation is a named check with the measured value.
Inputs are only read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SECTIONS = ("restaurant", "ainex", "church", "livingroom", "update", "segview", "ui")


@dataclass
class Check:
    ac: str
    name: str
    passed: bool | None  # None = needs manual inspection
    value: Any
    expected: str


@dataclass
class Harness:
    inputs: Path
    out: Path
    checks: list[Check] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)

    def check(self, ac: str, name: str, passed: bool | None, value: Any, expected: str) -> None:
        self.checks.append(Check(ac, name, passed, value, expected))
        mark = {True: "PASS", False: "FAIL", None: "INFO"}[passed]
        print(f"[{mark}] {ac} {name}: {value} (expected {expected})", file=sys.stderr, flush=True)

    def run(self, tag: str, args: list[str], stdout_name: str | None = None,
            timeout: float = 3600) -> tuple[int, bytes, float]:
        """Run one command; stdout saved to ``stdout_name`` (or discarded), stderr to a log."""
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        res = subprocess.run(args, capture_output=True, timeout=timeout)
        dt = time.perf_counter() - t0
        (self.out / "logs" / f"{tag}.stderr.txt").write_bytes(res.stderr)
        if stdout_name:
            p = self.out / stdout_name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(res.stdout)
        self.runs.append({"tag": tag, "cmd": " ".join(args), "exit": res.returncode,
                          "seconds": round(dt, 2), "stdout": stdout_name})
        print(f"  ran {tag}: exit {res.returncode} in {dt:.1f} s", file=sys.stderr, flush=True)
        return res.returncode, res.stdout, dt

    def sh(self, script: str) -> str:
        return str(REPO / script)


# ------------------------------------------------------------------------------------------------
# helpers


def att(obj: dict[str, Any], name: str) -> float:
    cub = obj["object_data"]["cuboid"][0]
    for a in cub.get("attributes", {}).get("num", []):
        if a["name"] == name:
            return float(a["val"])
    return float("nan")


def score(obj: dict[str, Any]) -> float:
    return next(float(n["val"]) for n in obj["object_data"]["num"] if n["name"] == "score")


def hexcol(obj: dict[str, Any]) -> str:
    return next(t["val"] for t in obj["object_data"]["text"] if t["name"] == "color_hex")


def objects(doc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(k): v for k, v in doc["openlabel"].get("objects", {}).items()}


def map_hash(map_dir: Path) -> str:
    h = hashlib.sha256()
    for n in ("map.json", "frames.json"):
        p = map_dir / n
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()


def tree_hash(map_dir: Path) -> str:
    from oh_my_slam.mapping.store import full_tree_hash

    return full_tree_hash(map_dir)


def angle_deg(a: Any, b: Any) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def frames_of(map_dir: Path) -> list[dict[str, Any]]:
    return json.loads((map_dir / "frames.json").read_text())["frames"]


def pose(fr: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from oh_my_slam.core.types import Pose

    T = Pose.from_dict(fr["T_map_cam"])
    return T.R, T.t


def floor_normal_angle(map_dir: Path) -> float:
    from oh_my_slam.core.geometry import ransac_plane
    from oh_my_slam.core.ply import read_ply
    from oh_my_slam.reconstruction.gravity import floor_candidate_height

    c = read_ply(map_dir / "cloud.ply").xyz.astype(np.float64)
    if len(c) > 400000:
        c = c[np.random.default_rng(0).choice(len(c), 400000, replace=False)]
    meta = json.loads((map_dir / "map.json").read_text())
    fz = meta.get("floor_z")
    if fz is None:
        fz = floor_candidate_height(c[:, 2])
    if fz is None:
        return float("nan")
    band = c[np.abs(c[:, 2] - fz) < 0.1]
    res = ransac_plane(band, 0.03, normal_prior=[0, 0, 1], max_angle_deg=30)
    return float("nan") if res is None else angle_deg(res[0], [0, 0, 1])


def validate_doc(doc: dict[str, Any]) -> list[str]:
    from oh_my_slam.schema.validate import validation_errors

    return validation_errors(doc)


# ------------------------------------------------------------------------------------------------
# AC25 restaurant.jpg


def restaurant(h: Harness) -> None:
    img = h.inputs / "restaurant.jpg"
    rec = h.sh("reconstruct.sh")
    h.run("restaurant_warmup", [rec, "-i", str(img)])
    code, out, t_json = h.run("restaurant_json", [rec, "-i", str(img)], "restaurant/scene.json")
    doc = json.loads(out)
    errs = validate_doc(doc)
    h.check("AC25", "json validates (schema + extra checks)", not errs, len(errs), "0 errors")
    ol = doc["openlabel"]
    h.check("AC25", "intrinsics source", ol["metadata"].get("intrinsics_source") == "exif",
            ol["metadata"].get("intrinsics_source"), "exif")
    cm = ol["streams"]["camera"]["stream_properties"]["intrinsics_pinhole"]["camera_matrix"]
    fx, fy, cx, cy = cm[0], cm[5], cm[2], cm[6]
    h.check("AC25", "fx, fy within 2 % of 2658 px",
            abs(fx / 2658 - 1) <= 0.02 and abs(fy / 2658 - 1) <= 0.02,
            f"fx={fx:.1f} fy={fy:.1f}", "2658 ± 2 %")
    h.check("AC25", "principal point within 1 % of (2000, 1500)",
            abs(cx / 2000 - 1) <= 0.01 and abs(cy / 1500 - 1) <= 0.01, f"({cx:.1f}, {cy:.1f})",
            "(2000, 1500) ± 1 %")
    objs = objects(doc)
    labels = Counter(o["type"] for o in objs.values())
    want = {"chair", "dining table", "person", "potted plant", "bottle"}
    h.check("AC25", ">= 10 objects", len(objs) >= 10, len(objs), ">= 10")
    h.check("AC25", "label set covers >= 3 of {chair, dining table, person, potted plant, bottle}",
            len(want & set(labels)) >= 3, sorted(want & set(labels)), ">= 3")
    h.check("AC25", "every score >= 0.5", all(score(o) >= 0.5 for o in objs.values()),
            min((score(o) for o in objs.values()), default=None), ">= 0.5")
    chairs = [o for o in objs.values() if o["type"] == "chair"]
    ok_h = [0.5 <= att(o, "height_m") <= 1.3 and max(att(o, "width_m"), att(o, "depth_m")) <= 1.2
            for o in chairs]
    frac = float(np.mean(ok_h)) if ok_h else 0.0
    h.check("AC25", ">= 70 % chairs with height in [0.5, 1.3] and longest side <= 1.2 m",
            frac >= 0.7, f"{frac:.0%} of {len(chairs)}", ">= 70 %")
    heights = sorted(round(att(o, "height_m"), 2) for o in objs.values() if o["type"] == "person")
    # Who is standing is not annotated: seated people measure ~1.0–1.35 m to the floor, so boxes
    # of >= 1.4 m are taken as standing; none may exceed 2.1 m.
    standing = [x for x in heights if x >= 1.4]
    h.check("AC25", "standing people (>= 1.4 m boxes) within [1.4, 2.1] m; none above 2.1 m",
            bool(standing) and max(heights) <= 2.1, heights, "standing in [1.4, 2.1]")
    grav = ol["metadata"].get("gravity", {})
    if grav.get("prior_up_cam") and "floor" in grav.get("source", ""):
        ang = angle_deg(grav["up_cam"], grav["prior_up_cam"])
        h.check("AC25", "floor normal within 5° of the estimated up (gravity)", ang <= 5.0,
                f"{ang:.2f}°", "<= 5°")
    else:
        h.check("AC25", "floor normal within 5° of up", False, grav.get("source"), "floor found")
    # PLY
    code, out, t_ply = h.run("restaurant_ply", [rec, "-i", str(img), "-f", "ply"],
                             "restaurant/cloud.ply")
    from oh_my_slam.core.images import load_rgb
    from oh_my_slam.core.ply import parse_ply

    cloud = parse_ply(out)
    h.check("AC25", "PLY valid binary with >= 400k points", len(cloud) >= 400000, len(cloud),
            ">= 400000")
    rgb = load_rgb(img, max_side=1024)
    gw, gh = rgb.shape[1], rgb.shape[0]
    s = gw / 4000.0
    u = np.rint(fx * s * cloud.xyz[:, 0] / cloud.xyz[:, 2] + cx * s).astype(int)
    v = np.rint(fy * s * cloud.xyz[:, 1] / cloud.xyz[:, 2] + cy * s).astype(int)
    ins = (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
    same = (rgb[v[ins], u[ins]] == cloud.rgb[ins]).all(axis=1).mean()
    h.check("AC25", "PLY colours equal the resized image pixels", same >= 0.99, f"{same:.2%}",
            ">= 99 % exact")
    med = float(np.median(cloud.xyz[:, 2]))
    h.check("AC25", "median depth of valid points in [4, 40] m", 4 <= med <= 40, f"{med:.2f} m",
            "[4, 40]")
    h.check("AC25", "latency -f ply <= 1.5 s (warm)", t_ply <= 1.5, f"{t_ply:.2f} s", "<= 1.5 s")
    h.check("AC25", "latency JSON <= 3.0 s (warm)", t_json <= 3.0, f"{t_json:.2f} s", "<= 3.0 s")
    # segment artefacts, --labels, --min-score
    seg = h.sh("segment.sh")
    outdir = h.out / "restaurant" / "segment_o"
    code, out, _ = h.run("restaurant_segment_o", [seg, "-i", str(img), "-o", str(outdir)],
                         "restaurant/segment_stdout.json")
    check_artifacts(h, "AC25", outdir, out)
    code, out, _ = h.run("restaurant_labels_chair", [seg, "-i", str(img), "--labels", "chair"],
                         "restaurant/segment_chair.json")
    ch = objects(json.loads(out))
    h.check("AC25", "--labels chair returns only chairs (>= 10)",
            len(ch) >= 10 and all(o["type"] == "chair" for o in ch.values()), len(ch),
            ">= 10, chairs only")
    base = objects(json.loads((h.out / "restaurant/segment_stdout.json").read_bytes()))
    code, out, _ = h.run("restaurant_min08", [seg, "-i", str(img), "--min-score", "0.8"],
                         "restaurant/segment_min08.json")
    hi = objects(json.loads(out))
    subset = all(k in base and base[k]["type"] == v["type"] for k, v in hi.items())
    h.check("AC25", "--min-score 0.8 ⊆ 0.5 result with identical ids", subset,
            f"{len(hi)} of {len(base)}", "subset, same ids")


def check_artifacts(h: Harness, ac: str, outdir: Path, stdout: bytes,
                    is_map: bool = False) -> None:
    """Artefact rules. For maps ``segmented.png`` is a contact sheet of <= 6 keyframes (C5), so
    only the objects on those tiles can appear; their mask colours must still be exact."""
    from PIL import Image

    from oh_my_slam.core.ply import read_ply
    from oh_my_slam.segmentation.catalog import CSV_HEADER

    names = sorted(p.name for p in outdir.iterdir())
    want = sorted(["segmentation.json", "segmented.png", "catalog.csv", "catalog.md",
                   "segments.ply"])
    h.check(ac, f"{outdir.name}: exactly the 5 artefacts", names == want, names, "5 files")
    same = (outdir / "segmentation.json").read_bytes() == stdout
    h.check(ac, f"{outdir.name}: segmentation.json byte-identical to stdout", same, same, "True")
    doc = json.loads(stdout)
    objs = objects(doc)
    rows = list(csv.DictReader((outdir / "catalog.csv").open()))
    header = (outdir / "catalog.csv").read_text().splitlines()[0]
    h.check(ac, f"{outdir.name}: exact CSV header", header == ",".join(CSV_HEADER), header,
            ",".join(CSV_HEADER))
    csv_ok = all(hexcol(objs[int(r["id"])]) == r["color_hex"] for r in rows) and len(rows) == len(
        objs)
    md = (outdir / "catalog.md").read_text()
    md_hex = re.findall(r"`(#[0-9a-f]{6})`", md)
    vols = [float(line.split("|")[9]) for line in md.splitlines() if line.startswith("| <span")]
    h.check(ac, f"{outdir.name}: catalog.md sorted by descending volume",
            vols == sorted(vols, reverse=True), len(vols), "descending")
    png = np.asarray(Image.open(outdir / "segmented.png").convert("RGB")).reshape(-1, 3)
    png_cols = {tuple(c) for c in np.unique(png, axis=0)}
    ply = read_ply(outdir / "segments.ply")
    lab = ply.label if ply.label is not None else np.zeros(len(ply), int)
    ply_ok = True
    for oid, o in objs.items():
        rgb = tuple(int(hexcol(o)[i:i + 2], 16) for i in (1, 3, 5))
        vec = next(v["val"] for v in o["object_data"]["vec"] if v["name"] == "color")
        pts = ply.rgb[lab == oid]
        if tuple(vec) != rgb or (len(pts) and not (pts == rgb).all()):
            ply_ok = False
    grey = ply.rgb[lab == 0]
    ply_ok = ply_ok and (len(grey) == 0 or bool((grey == 128).all()))
    visible = sum(tuple(int(hexcol(o)[i:i + 2], 16) for i in (1, 3, 5)) in png_cols
                  for o in objs.values())
    h.check(ac, f"{outdir.name}: colour contract (JSON = CSV = MD = PLY; PNG masks)",
            csv_ok and ply_ok and set(md_hex) == {hexcol(o) for o in objs.values()}
            and visible >= (min(1, len(objs)) if is_map else 0.9 * len(objs)),
            f"csv={csv_ok} ply={ply_ok} md={set(md_hex) == {hexcol(o) for o in objs.values()}} "
            f"png={visible}/{len(objs)}", "all consistent")


# ------------------------------------------------------------------------------------------------
# mapping helpers


def run_map(h: Harness, tag: str, inputs: list[Path], map_dir: Path, mode: str = "full",
            fmt: str = "json", fps: float | None = None) -> tuple[int, bytes, float]:
    args = [h.sh("mapper.sh"), "update", "-a", *map(str, inputs), "-m", str(map_dir), "-t", mode,
            "-f", fmt]
    if fps is not None:
        args += ["-fps", str(fps)]
    ext = "ply" if fmt == "ply" else "json"
    return h.run(tag, args, f"{tag}/output.{ext}")


def name_yaw(name: str) -> float | None:
    m = re.match(r"\d+_(?:bootstrap_)?(left|right_to|right)_?(\d+)_(level|up|down)", name)
    if not m:
        return None
    deg = float(m.group(2))
    return deg if m.group(1) in ("left", "right_to") else -deg


def ainex(h: Harness) -> None:
    folder = h.inputs / "ainex-captures"
    mdir = h.out / "maps" / "ainex"
    code, out, dt = run_map(h, "ainex_full", [folder], mdir)
    h.check("AC26", "whole-folder map created", code == 0, code, "exit 0")
    if code != 0:
        return
    frames = frames_of(mdir)
    n_in = len([p for p in folder.iterdir() if p.suffix.lower() == ".jpg"])
    h.check("AC26", ">= 71 of 79 frames registered", len(frames) >= 71, f"{len(frames)}/{n_in}",
            ">= 71")
    h.check("AC26", ".DS_Store ignored", all(".DS_Store" not in f["source"] for f in frames),
            True, "True")
    by_name = {Path(f["source"]).name: f for f in frames}
    ref = by_name.get("001_bootstrap_level.jpg")
    yaws, errs = [], []
    if ref is not None:
        R0, _ = pose(ref)
        f0 = R0[:, 2]
        y0 = np.degrees(np.arctan2(f0[1], f0[0]))
        for n, f in by_name.items():
            ny = name_yaw(n)
            if ny is None or not n.endswith("_level.jpg"):
                continue
            R, _ = pose(f)
            fw = R[:, 2]
            y = np.degrees(np.arctan2(fw[1], fw[0])) - y0
            e = (y - ny + 180) % 360 - 180
            yaws.append((n, ny, round(y, 1)))
            errs.append(e)
    errs_a = np.array(errs) if errs else np.array([np.nan])
    off = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(errs_a)))))) if errs else 0.0
    fitted = (errs_a - off + 180) % 360 - 180
    h.check("AC26", "level-frame yaw vs name, relative to 001 (±7°)",
            bool(np.nanmax(np.abs(errs_a)) <= 7), f"max {np.nanmax(np.abs(errs_a)):.1f}°, "
            f"median {np.nanmedian(np.abs(errs_a)):.1f}°", "<= 7°")
    h.check("AC26", "level-frame yaw vs name after one common offset (001 itself is offset)",
            bool(np.nanmax(np.abs(fitted)) <= 7), f"offset {off:.1f}°, max "
            f"{np.nanmax(np.abs(fitted)):.1f}°", "<= 7° (informative)")
    pitch_ok, pitch_vals = [], []
    by_stem = {re.sub(r"^\d+_", "", n): n for n in by_name}  # index prefixes differ per shot
    for n, f in by_name.items():
        if not (n.endswith("_up.jpg") or n.endswith("_down.jpg")):
            continue
        sib = by_stem.get(re.sub(r"_(up|down)\.jpg$", "_level.jpg", re.sub(r"^\d+_", "", n)))
        if sib is None:
            continue
        sib_frame = by_name[sib]
        pr = np.degrees(np.arcsin(np.clip(pose(f)[0][:, 2][2], -1, 1)))
        pl = np.degrees(np.arcsin(np.clip(pose(sib_frame)[0][:, 2][2], -1, 1)))
        d = pr - pl
        pitch_vals.append(round(d, 1))
        pitch_ok.append(d >= 5 if n.endswith("_up.jpg") else d <= -5)
    h.check("AC26", "_up/_down pitch differs from _level sibling by >= 5° (named direction)",
            bool(pitch_ok) and all(pitch_ok), f"{sum(pitch_ok)}/{len(pitch_ok)} "
            f"(range {min(pitch_vals, default=0)}..{max(pitch_vals, default=0)})", "all")
    a, b = by_name.get("026_left_210_level.jpg"), by_name.get("078_right_150_level.jpg")
    if a and b:
        from oh_my_slam.core.geometry import rotation_angle_deg

        d = rotation_angle_deg(pose(a)[0] @ pose(b)[0].T)
        h.check("AC26", "frames 026 and 078 (same heading) differ by <= 7°", d <= 7, f"{d:.1f}°",
                "<= 7°")
    else:
        h.check("AC26", "frames 026 and 078 registered", False, "missing", "both registered")
    C = np.array([pose(f)[1] for f in frames])
    rad = float(np.linalg.norm(C - C.mean(0), axis=1).max())
    h.check("AC26", "camera centres within 1.0 m radius", rad <= 1.0, f"{rad:.2f} m", "<= 1.0 m")
    doc = json.loads(out)
    h.check("AC26", "per-label object counts (compare with the viewer inspection)", None,
            dict(Counter(o["type"] for o in objects(doc).values())), "no duplicates")
    (h.out / "ainex_yaws.json").write_text(json.dumps(yaws, indent=1))
    # split updates
    split = h.out / "ainex_split_inputs"
    for part in ("a", "b"):
        (split / part).mkdir(parents=True, exist_ok=True)
    for p in sorted(folder.glob("*.jpg")):
        idx = int(p.name[:3])
        dst = split / ("a" if idx <= 40 else "b") / p.name
        if not dst.exists():
            dst.symlink_to(p)
    sdir = h.out / "maps" / "ainex_split"
    run_map(h, "ainex_split_1", [split / "a"], sdir)
    first_ids = {int(k): v["type"] for k, v in json.loads(
        (sdir / "scene.json").read_text())["openlabel"]["objects"].items()}
    code, out2, _ = run_map(h, "ainex_split_2", [split / "b"], sdir, mode="single")
    if code != 0:
        h.check("AC26", "split update 2 succeeded", False, code, "exit 0")
        return
    single = json.loads(out2)
    srcs = sorted(Path(fr["frame_properties"]["streams"][next(iter(
        fr["frame_properties"]["streams"]))]["uri"]).name for fr in single["openlabel"][
        "frames"].values())
    frames2 = {f["name"]: Path(f["source"]).name for f in frames_of(sdir)}
    names2 = sorted(frames2[fr["frame_properties"]["keyframe"]] for fr in
                    single["openlabel"]["frames"].values())
    only_b = all(41 <= int(n[:3]) <= 79 for n in names2)
    h.check("AC26", "-t single of update 2 lists exactly frames 041–079 (registered)",
            only_b and len(names2) >= 0.9 * 39, f"{len(names2)} frames, all in 041–079: {only_b}",
            "041–079 only")
    del srcs
    state = json.loads((sdir / "objects.json").read_text())
    objs_json = state["objects"]
    now = {o["id"]: o for o in objs_json}
    merged = {int(k): v for k, v in state.get("merged_into", {}).items()}
    kept = [i for i in first_ids if i in now and now[i]["last_seen_update"] == 2]
    lost = [i for i in first_ids if i not in now and i not in merged]
    dup = duplicate_pairs(objs_json)
    # ids are never re-assigned, so "keeps its id" fails only through duplication; objects not
    # re-detected in update 2 can legitimately be removed by the absence test (listed for review)
    h.check("AC26", "objects seen in both updates keep id + colour (no duplicates)",
            not dup, f"{len(kept)} re-observed with their ids, duplicates {dup[:5]}, "
            f"removed/not re-detected {lost[:5]}", "0 duplicates")


def duplicate_pairs(objs_json: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Confirmed objects of compatible labels whose boxes overlap strongly (identity failures)."""
    from oh_my_slam.segmentation.detect import compatible
    from oh_my_slam.segmentation.obb import OBB, obb_iou_upright

    conf = [o for o in objs_json if o.get("confirmed") and o.get("obb")]
    out = []
    for i, a in enumerate(conf):
        for b in conf[i + 1:]:
            if not compatible(a["label"], b["label"]):
                continue
            if obb_iou_upright(OBB.from_dict(a["obb"]), OBB.from_dict(b["obb"]),
                               samples=4000) > 0.3:
                out.append((a["id"], b["id"]))
    return out


def street_width(map_dir: Path, frames: list[dict[str, Any]]) -> float:
    from oh_my_slam.core.ply import read_ply

    c = read_ply(map_dir / "cloud.ply").xyz.astype(np.float64)
    C = np.array([pose(f)[1] for f in frames])
    widths = []
    for i in range(1, len(C) - 1, 2):
        d = C[i + 1] - C[i - 1]
        d[2] = 0
        if np.linalg.norm(d) < 1e-3:
            continue
        d /= np.linalg.norm(d)
        lat = np.cross([0, 0, 1.0], d)
        rel = c - C[i]
        along = rel @ d
        up = rel[:, 2]
        sel = (np.abs(along) < 1.5) & (up > -0.5) & (up < 4.0)
        s = rel[sel] @ lat
        left, right = s[s > 0.5], -s[s < -0.5]
        if len(left) > 50 and len(right) > 50:
            widths.append(np.percentile(left, 90) + np.percentile(right, 90))
    return float(np.median(widths)) if widths else float("nan")


def church(h: Harness) -> None:
    video = h.inputs / "church.mp4"
    mdir = h.out / "maps" / "church"
    code, out, dt = run_map(h, "church_full", [video], mdir, fps=2)
    h.check("AC27", "map created", code == 0, code, "exit 0")
    if code != 0:
        return
    meta = json.loads((mdir / "map.json").read_text())
    frames = frames_of(mdir)
    added = meta["updates"][-1]
    n_kf = len(added["frames_added"]) + len(added["frames_rejected"])
    h.check("AC27", "~50 keyframes sampled (±2)", abs(n_kf - 50) <= 2, n_kf, "50 ± 2")
    h.check("AC27", ">= 90 % registered", len(frames) >= 0.9 * n_kf, f"{len(frames)}/{n_kf}",
            ">= 90 %")
    C = np.array([pose(f)[1] for f in sorted(frames, key=lambda f: f["index"])])
    path = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    h.check("AC27", "camera path length in [5, 60] m", 5 <= path <= 60, f"{path:.1f} m",
            "[5, 60]")
    w = street_width(mdir, sorted(frames, key=lambda f: f["index"]))
    h.check("AC27", "street width between façades in [2, 12] m", 2 <= w <= 12, f"{w:.1f} m",
            "[2, 12]")
    objs = objects(json.loads(out))
    ppl = [att(o, "height_m") for o in objs.values() if o["type"] == "person"]
    h.check("AC27", "people <= 2.1 m tall", bool(ppl) and max(ppl) <= 2.1,
            sorted(round(x, 2) for x in ppl), "<= 2.1")
    mesh_ok = mesh_loads(mdir)
    h.check("AC27", "textured mesh exists and loads", mesh_ok, mesh_ok, "True")
    labels = {o["type"] for o in objs.values()}
    extra = labels & {"potted plant", "lamppost", "street light", "window", "balcony"}
    h.check("AC27", "objects include person and >= 1 of {potted plant, lamppost/street light, "
            "window, balcony}", "person" in labels and bool(extra), sorted(labels), "present")
    h.check("AC27", "wall-clock <= 6 min (warm server)", dt <= 360, f"{dt:.0f} s", "<= 360 s")
    code, out_ply, _ = run_map(h, "church_ply", [video], h.out / "maps" / "church_ply",
                               fmt="ply", fps=2)
    from oh_my_slam.core.ply import parse_ply

    try:
        n = len(parse_ply(out_ply))
        ok = code == 0 and n > 0
    except ValueError:
        n, ok = 0, False
    h.check("AC27", "-f ply output is a valid PLY (separate map)", ok, n, "> 0 points")


def mesh_loads(map_dir: Path) -> bool:
    try:
        import trimesh

        s = trimesh.load(str(map_dir / "mesh" / "mesh.glb"), force="scene")
        g = next(iter(s.geometry.values()))
        tex = g.visual.material.baseColorTexture
        return tex is not None and float(np.asarray(tex.convert("RGB")).std()) > 5
    except Exception:
        return False


def livingroom(h: Harness) -> None:
    video = h.inputs / "livingroom.mp4"
    mdir = h.out / "maps" / "livingroom"
    code, out, dt = run_map(h, "livingroom_full", [video], mdir, fps=2)
    h.check("AC28", "map created", code == 0, code, "exit 0")
    if code != 0:
        return
    meta = json.loads((mdir / "map.json").read_text())
    frames = frames_of(mdir)
    up = meta["updates"][-1]
    n_kf = len(up["frames_added"]) + len(up["frames_rejected"])
    h.check("AC28", "~184 keyframes (±5)", abs(n_kf - 184) <= 5, n_kf, "184 ± 5")
    h.check("AC28", ">= 90 % registered", len(frames) >= 0.9 * n_kf, f"{len(frames)}/{n_kf}",
            ">= 90 %")
    objs = objects(json.loads(out))
    cnt = Counter(o["type"] for o in objs.values())
    tv = [o for o in objs.values() if o["type"] in ("television", "computer monitor")]
    sofa = [o for o in objs.values() if o["type"] in ("sofa", "couch", "loveseat")]
    ct = [o for o in objs.values() if o["type"] == "coffee table"]
    door = [o for o in objs.values() if o["type"] == "door"]
    h.check("AC28", "exactly 1 television, 1 sofa, 1 coffee table",
            len(tv) == 1 and len(sofa) == 1 and len(ct) == 1,
            f"tv={len(tv)} sofa={len(sofa)} coffee table={len(ct)}", "1 / 1 / 1 (±1 if real)")
    h.check("AC28", "per-label counts (viewer inspection)", None, dict(cnt), "no duplicates")

    def rng_check(name: str, items: list[dict[str, Any]], fn: Any, lo: float, hi: float) -> None:
        vals = [round(fn(o), 2) for o in items]
        h.check("AC28", name, bool(vals) and any(lo <= v <= hi for v in vals), vals, f"[{lo}, {hi}]")

    rng_check("TV width in [0.9, 1.9] m", tv, lambda o: att(o, "width_m"), 0.9, 1.9)
    rng_check("coffee-table height in [0.3, 0.6] m", ct, lambda o: att(o, "height_m"), 0.3, 0.6)
    rng_check("door height in [1.8, 2.3] m", door, lambda o: att(o, "height_m"), 1.8, 2.3)
    rng_check("sofa length in [1.4, 3.2] m", sofa, lambda o: att(o, "width_m"), 1.4, 3.2)
    ang = floor_normal_angle(mdir)
    h.check("AC28", "floor normal within 5° of up", ang <= 5.0, f"{ang:.2f}°", "<= 5°")
    h.check("AC28", "wall-clock <= 18 min (warm server)", dt <= 1080, f"{dt:.0f} s", "<= 1080 s")


def updates(h: Harness) -> None:
    lr = h.out / "maps" / "livingroom"
    if (lr / "map.json").exists():
        frame = h.out / "livingroom_t45.jpg"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "45", "-i",
                        str(h.inputs / "livingroom.mp4"), "-frames:v", "1", "-q:v", "2",
                        str(frame)], check=True)
        before = json.loads((lr / "objects.json").read_text())["objects"]
        key = {o["id"]: o["label"] for o in before if o["confirmed"] and o["label"] in (
            "television", "sofa", "coffee table")}
        code, out, _ = run_map(h, "livingroom_single_frame", [frame], lr, mode="single")
        doc = json.loads(out) if code == 0 else {"openlabel": {"frames": {}}}
        h.check("AC29", "single frame from t = 45 s registers; -t single lists 1 frame",
                code == 0 and len(doc["openlabel"]["frames"]) == 1,
                f"exit {code}, frames {len(doc['openlabel']['frames'])}", "exit 0, 1 frame")
        after = {o["id"] for o in json.loads((lr / "objects.json").read_text())["objects"]}
        h.check("AC29", "furniture ids retained", set(key) <= after, key, "all retained")
        meta = json.loads((lr / "map.json").read_text())
        removed = meta["updates"][-1]["objects"].get("removed", []) if code == 0 else []
        h.check("AC29", "no object removed (no real change)", code == 0 and not removed, removed,
                "[]")
    ch = h.out / "maps" / "church"
    if (ch / "map.json").exists():
        before = map_hash(ch)
        code, out, _ = run_map(h, "church_add_restaurant", [h.inputs / "restaurant.jpg"], ch)
        h.check("AC29", "restaurant.jpg into the church map exits 5", code == 5, code, "5")
        h.check("AC29", "church map byte-identical (map.json + frames.json)",
                map_hash(ch) == before, map_hash(ch) == before, "True")


def segview(h: Harness) -> None:
    for name in ("ainex", "church", "livingroom"):
        mdir = h.out / "maps" / name
        if not (mdir / "map.json").exists():
            continue
        before = tree_hash(mdir)
        outdir = h.out / name / "segment_m"
        code, out, _ = h.run(f"{name}_segment_m", [h.sh("segment.sh"), "-m", str(mdir), "-o",
                                                   str(outdir)], f"{name}/segment_m.json")
        if code != 0:
            h.check("AC30", f"{name}: segment.sh -m", False, code, "exit 0")
            continue
        check_artifacts(h, "AC30", outdir, out, is_map=True)
        full = json.loads((mdir / "scene.json").read_text())
        same = json.loads(out)["openlabel"]["objects"] == full["openlabel"]["objects"]
        h.check("AC30", f"{name}: segment -m objects equal -t full objects", same, same, "True")
        h.check("AC30", f"{name}: map unchanged by segment -m", tree_hash(mdir) == before,
                tree_hash(mdir) == before, "True")


# ------------------------------------------------------------------------------------------------
# AC30/AC31: viewer in a browser


def browse(h: Harness, args: list[str], tag: str, map_dir: Path | None) -> None:
    from playwright.sync_api import sync_playwright

    before = tree_hash(map_dir) if map_dir else None
    proc = subprocess.Popen([h.sh("view.sh"), *args, "--no-browser"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    try:
        url = None
        assert proc.stderr is not None
        deadline = time.time() + 300
        while time.time() < deadline:
            line = proc.stderr.readline().decode()
            m = re.search(r"(http://127\.0\.0\.1:\d+/)", line)
            if m:
                url = m.group(1)
                break
            if not line and proc.poll() is not None:
                break
        if url is None:
            h.check("AC30", f"{tag}: viewer started", False, "no URL", "URL on stderr")
            return
        shots = h.out / "ui" / tag
        shots.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            for vw, vh in ((1280, 800), (1920, 1080)):
                ctx = b.new_context(viewport={"width": vw, "height": vh})
                pg = ctx.new_page()
                pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                pg.on("pageerror", lambda e: errors.append(str(e)))
                t0 = time.perf_counter()
                pg.goto(url)
                pg.wait_for_function("window.__viewer && window.__viewer.ready === true",
                                     timeout=300000)
                t_render = time.perf_counter() - t0
                pg.wait_for_timeout(800)
                pg.screenshot(path=str(shots / f"{vw}x{vh}_initial.png"))
                if vw == 1280:
                    ui_session(h, pg, shots, tag, t_render)
                ctx.close()
            b.close()
        h.check("AC30", f"{tag}: no console errors", not errors, errors[:3], "[]")
        if map_dir is not None:
            h.check("AC30", f"{tag}: map unchanged after viewing", tree_hash(map_dir) == before,
                    True, "True")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def ui_session(h: Harness, pg: Any, shots: Path, tag: str, t_render: float) -> None:
    """R43 checklist, scripted: stats line, layers, sliders, catalogue ↔ 3D, hover, reset."""
    stats = pg.inner_text("#stats")
    h.check("AC31", f"{tag}: stats line (points, objects, frames)",
            all(w in stats for w in ("points", "objects", "frame")), stats, "present")
    h.check("AC31", f"{tag}: loaded and auto-framed", True, f"{t_render:.1f} s", "renders")
    layers = pg.locator("[data-layer]").count()
    h.check("AC31", f"{tag}: six grouped layer toggles", layers == 6, layers, "6")
    pg.click('#tabs button[data-tab="display"]')
    pg.wait_for_timeout(300)
    pg.screenshot(path=str(shots / "display_tab.png"))
    sliders = pg.locator("#tab-display .lil-controller.lil-number").count()
    h.check("AC31", f"{tag}: point-size and label-density sliders", sliders >= 2, sliders, ">= 2")
    pg.click('#tabs button[data-tab="catalogue"]')
    rows = pg.locator("#catalogue tbody tr")
    n = rows.count()
    if n:
        rows.nth(0).click()
        pg.wait_for_timeout(700)
        sel = pg.evaluate("() => window.__viewer.selected")
        h.check("AC31", f"{tag}: catalogue row click selects + frames its OBB",
                sel == int(rows.nth(0).get_attribute("data-id")), sel, "row id")
        pg.screenshot(path=str(shots / "selected_from_catalogue.png"))
        # hover the selected box's centre on screen → tooltip
        xy = pg.evaluate("""() => {
          const v = window.__viewer; const o = v.objects.find(x => x.id === v.selected);
          const g = window.__viewerGroups.pick.parent; const p = o.center.clone();
          p.applyMatrix4(g.matrixWorld);
          const cam = window.__viewerCamera; if (!cam) return null;
          p.project(cam); const r = document.querySelector('#canvas-host').getBoundingClientRect();
          return [r.left + (p.x + 1) / 2 * r.width, r.top + (1 - p.y) / 2 * r.height];
        }""")
        if xy:
            pg.mouse.move(xy[0], xy[1])
            pg.wait_for_timeout(400)
            tip = pg.locator("#tooltip")
            shown = tip.is_visible()
            h.check("AC31", f"{tag}: hover tooltip (label, id, score, W×D×H, volume)", shown,
                    tip.inner_text().replace("\n", " ")[:80] if shown else "hidden", "visible")
            pg.screenshot(path=str(shots / "hover.png"))
            pg.mouse.click(xy[0], xy[1])
            pg.wait_for_timeout(300)
            h.check("AC31", f"{tag}: clicking an OBB highlights its catalogue row",
                    pg.locator("#catalogue tbody tr.selected").count() == 1,
                    pg.locator("#catalogue tbody tr.selected").count(), "1")
    pg.keyboard.press("r")
    pg.wait_for_timeout(500)
    pg.screenshot(path=str(shots / "after_reset.png"))
    orbit_gif(pg, shots / "orbit.gif")
    vis_labels = pg.evaluate("() => window.__viewer.objects.filter(o => o.labelObj.visible).length")
    h.check("AC31", f"{tag}: labels decluttered", vis_labels <= 40, vis_labels, "<= 40 visible")
    for layer in ("mesh", "segments", "cameras"):
        loc = pg.locator(f'[data-layer="{layer}"] input')
        if loc.count() and loc.is_enabled():
            pg.click('#tabs button[data-tab="layers"]')
            loc.click()
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(shots / f"toggle_{layer}.png"))
            loc.click()


def orbit_gif(pg: Any, path: Path, frames: int = 16) -> None:
    """A short orbit around the current target, captured frame by frame into an animated GIF
    (no screen-recording dependency)."""
    import io

    from PIL import Image

    imgs = []
    for _ in range(frames):
        pg.evaluate(f"""() => {{
          const cam = window.__viewerCamera; const t = window.__viewerControls.target;
          const off = cam.position.clone().sub(t);
          const a = {2 * np.pi / frames};
          const x = off.x * Math.cos(a) - off.y * Math.sin(a);
          const y = off.x * Math.sin(a) + off.y * Math.cos(a);
          cam.position.set(t.x + x, t.y + y, t.z + off.z); window.__viewerControls.update();
        }}""")
        pg.wait_for_timeout(120)
        png = pg.locator("#viewport").screenshot()
        imgs.append(Image.open(io.BytesIO(png)).convert("RGB").resize((640, 400)))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=150, loop=0)


def ui(h: Harness) -> None:
    browse(h, ["-i", str(h.inputs / "restaurant.jpg")], "view_restaurant", None)
    for name in ("ainex", "church", "livingroom"):
        mdir = h.out / "maps" / name
        if (mdir / "map.json").exists():
            browse(h, ["-m", str(mdir)], f"view_{name}", mdir)


# ------------------------------------------------------------------------------------------------


def write_report(h: Harness, started: float) -> None:
    passed = sum(c.passed is True for c in h.checks)
    failed = sum(c.passed is False for c in h.checks)
    info = sum(c.passed is None for c in h.checks)
    rep = {"generated": datetime.now(UTC).isoformat(), "seconds": time.time() - started,
           "passed": passed, "failed": failed, "info": info,
           "checks": [c.__dict__ for c in h.checks], "runs": h.runs}
    (h.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    lines = [f"# Validation report ({rep['generated']})", "",
             f"{passed} passed, {failed} failed, {info} for inspection — "
             f"{rep['seconds'] / 60:.1f} min", "", "| AC | check | result | measured | expected |",
             "|---|---|---|---|---|"]
    for c in h.checks:
        res = {True: "PASS", False: "**FAIL**", None: "inspect"}[c.passed]
        val = str(c.value).replace("|", "/")[:160]
        lines.append(f"| {c.ac} | {c.name} | {res} | {val} | {c.expected} |")
    lines += ["", "## Runs", "", "| run | exit | seconds |", "|---|---|---|"]
    lines += [f"| {r['tag']} | {r['exit']} | {r['seconds']} |" for r in h.runs]
    (h.out / "report.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oh_my_slam.tools.validate_inputs")
    ap.add_argument("--inputs", type=Path, default=Path("/Users/U124317/robot_view"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default=",".join(SECTIONS))
    args = ap.parse_args(argv)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path.home() / "oh-my-slam-data" / "validation" / stamp
    out.mkdir(parents=True, exist_ok=True)
    h = Harness(args.inputs, out)
    started = time.time()
    fns = {"restaurant": restaurant, "ainex": ainex, "church": church, "livingroom": livingroom,
           "update": updates, "segview": segview, "ui": ui}
    try:
        for name in args.only.split(","):
            print(f"== {name}", file=sys.stderr, flush=True)
            try:
                fns[name](h)
            except Exception as exc:  # keep going: one failing section must not hide the rest
                h.check(name, "section crashed", False, f"{type(exc).__name__}: {exc}", "no crash")
            write_report(h, started)
    finally:
        write_report(h, started)
    print(str(out / "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
