"""Benchmark evaluator of every entry point on ``examples/`` (high_level_spec.md §5).

    uv run python -m oh_my_slam.tools.evaluate [--out DIR] [--set-baseline] [--baseline PATH]
                                               [--targets PATH] [--splits N]

Runs, strictly one at a time, ``start_inference_server.sh`` (cold start, resident memory),
``reconstruct.sh`` / ``segment.sh -i`` / ``view.sh -i`` on ``restaurant.jpg``, ``segment.sh -i`` on
every ``ainex-captures`` frame, ``mapper.sh update`` on the sequence in one update and split across
``N`` updates, and ``segment.sh -m`` / ``view.sh -m`` on the resulting maps. Metrics: performance,
pose accuracy, map quality, segmentation, contracts (and ground truth when annotations exist under
``examples/ground_truth/``). Each has a target in ``examples/targets.json`` (data) and is compared
with the stored baseline run (``~/oh-my-slam-data/evaluations/baseline.json``).

Output (default ``~/oh-my-slam-data/evaluations/<UTC>/``, never inside the repository):
``result.json``, ``summary.md``, ``runs/`` (stdout, stderr, timings per command), ``outputs/``
(``-o`` / ``-d`` results) and ``maps/`` (the two maps). Progress goes to stderr; stdout gets the
path of ``summary.md``. Exit 0 when every metric passes, 1 when one fails, 2 on a usage error.

Modules: ``names`` (capture-name grammar), ``runner`` / ``memory`` (running and measuring
commands), ``suite`` (the plan), ``contracts``, ``performance``, ``poses``, ``mapquality``,
``segmentation``, ``groundtruth`` (metrics), ``viewer`` (browser timing), ``metrics`` (targets and
baseline), ``report``.
"""
