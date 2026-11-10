# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repository

Implementation of `high_level_spec.md` (repository root) and `specs/` (`api.md`, `map.md`,
`inference.md`, `evaluation.md`, `api.schema.json`). Package `src/oh_my_slam` (uv, Python 3.12);
`README.md` records the decisions the spec defers to it.

Commands:

- Environment: `uv sync`; external tool: `brew install colmap` (4.2.x, as pycolmap).
- Service: `./server.sh [--host 0.0.0.0] [--port 9091] [--data DIR]`, `./server.sh --status|--stop`.
  It starts the inference server when it is not running (log
  `~/Library/Caches/oh-my-slam/server.log`, or under `OH_MY_SLAM_RUNTIME_DIR`); its own log is
  `<data>/oms/service.log`.
- Offline tests: `uv run pytest -m "not models" -q --cov=oh_my_slam` (fails below 100 % line and
  branch). Real models: `OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m models` with the inference
  server running.
- Lint/types/ownership: `uv run ruff check . && uv run mypy src && uv run lint-imports`.
- `SKILL.md`: `uv run python -m oh_my_slam.service.skill` regenerates it after any API change;
  `tests/unit/test_service_skill.py` fails while it is stale.
- Mapping features are SIFT (Homebrew `colmap` CLI); poses through pycolmap.

Never import torch/open3d in the same process (duplicate libomp aborts): torch lives only in the
inference server process. Never edit files in `.staging/` of a map.

## Spec invariants that span several sections

- **One client.** oh-my-slam exists to serve the robot-guide skill; a feature belongs only if a row
  of the client contract (high_level_spec.md §2) needs it. Robot knowledge reaches it only as
  camera pose priors and camera parameters.
- **One entry point.** `server.sh` (`oh_my_slam.service`) serves the whole API under `/api/` on
  port 9091, reachable from the LAN; it owns no model and manages the inference server.
- **API and schema.** Every JSON body follows `specs/api.schema.json` (oms/2.x; minors are
  additive; a package copy at `src/oh_my_slam/service/api.schema.json` must equal it). Requests are
  validated against it; tests check every response against it. `/api/openapi.json` and `SKILL.md`
  are generated from the same endpoint definitions (`service/routes.py`). Errors are typed
  (`code`, `retryable`); an unusable frame is an input status, never an error.
- **Capabilities.** `GET /api/health → capabilities` lists the groups really implemented (now O1,
  O4); the order of work is high_level_spec.md §12.
- **Map frame and versions.** The map frame is fixed by a map's first update (the metric priors'
  frame when given — the main path: every robot starts with metric priors — else the drifting
  priors', else the first-camera rule) and never re-gauged: each update fits the similarity from
  the mapper's internal frame onto the previous version's poses and the metric priors. Every
  commit is an immutable integer version; reads name the version they answer from. Every metric
  output carries `scale_source` and `scale_sigma_rel`.
- **Jobs.** An update is an asynchronous job; a client disconnect never cancels it. Until O2 + O3
  a job commits one version and fusion blocks it.
- **Ownership** (import-linter in `pyproject.toml`, `tests/unit/test_ownership.py`): `service`
  (HTTP) > `mapping` (registration, fusion, objects, store, bird's-eye) > `segmentation`
  (detection, boxes, colours) > `reconstruction` (depth, point generation, cloud derivation) >
  `client` > `core`; `server` owns the models only. Nothing imports `service`; reconstruction never
  imports segmentation or mapping.
- **Units and frames.** Metres, degrees, seconds; map frame gravity-aligned with z up; camera
  optical frame x right, y down, z forward; `T_a_b` is the pose of b in a.

## Machine and tooling facts (verified locally)

- Apple M3 Pro, 18 GiB unified memory, macOS 27.0.1, no CUDA (`sysctl`, `sw_vers`, 2026-10-26).
  The spec's target machine is still the M4 Max with 36 GB; evaluations run on this one.
- **COLMAP:**
  - Homebrew `colmap` 4.2.0 includes GLOMAP as `global_mapper`, and ALIKED/LightGlue feature
    extraction and matching through ONNX Runtime with the CoreML provider.
  - The **PyPI pycolmap 4.2 wheel is built without ONNX**. It lists the ALIKED/LightGlue enums but
    fails when they are used. The mapper extracts and matches SIFT through the Homebrew CLI and
    maps through pycolmap; keep both on the same 4.2.x version.
  - `patch_match_stereo` (dense MVS) requires CUDA, so it is unavailable on this machine.
- **PyTorch MPS:**
  - Two threads touching MPS in one process abort the whole process (pytorch#197805; its fix was
    reverted). Route *all* device work — load, `.to()`, forward, readback — through a single thread.
  - torch 2.13 produces corrupted attention output on macOS 26; use 2.14+.

## Working rules for this repo

- **Never bind a fixed port.** Orphaned processes from earlier work hold ports such as 8080, 8081
  and 8088.
- Run benchmarks one at a time; concurrent GPU work invalidates timings. Every colmap, mapper, inference or
  `-m models` command goes through one machine lock: `~/oh-my-slam-data/bin/heavy <cmd>` (Python `fcntl.flock` wrapper; macOS has no `flock`).
- Research subagents verify claims online only: no installs, weight downloads or local benchmarks.
  This machine is the user's daily driver.
- Only commit when the user requeres to