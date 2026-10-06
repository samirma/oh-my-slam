"""Benchmark evaluator of every entry point on ``examples/`` (specs/high_level_spec.md §5).

    uv run python -m oh_my_slam.tools.evaluate [--out DIR] [--set-baseline] [--baseline PATH]
                                               [--targets PATH] [--street2 PATH]

Runs, strictly one at a time, ``start_inference_server.sh`` (cold start, resident memory),
``reconstruct.sh`` / ``segment.sh -i`` / ``view.sh -i`` on ``restaurant.jpg``, ``segment.sh -i`` on
every ``ainex-captures`` frame, ``mapper.sh update`` on the sequence in one update and split across
3 updates, ``mapper.sh locate`` (held-out captures, and the reference map), ``segment.sh -m`` /
``view.sh -m`` on the resulting maps, ``mapper.sh update`` on ``office_sequence`` (one update and
its annotated splits, else its two halves) and on the street2 video, and ``server.sh``
(performance, parity with the commands, UI). Metrics: performance (end to end and per stage: time,
client and server peak memory), pose accuracy, map quality, segmentation (and its consistency with
the map), contracts, the web service, and — when annotations exist under
``examples/ground_truth/`` (optional; format in ``groundtruth``) — ground truth and the map
update. Each has a target in ``examples/targets.json`` (data) and is
compared with the stored baseline run (``~/oh-my-slam-data/evaluations/baseline.json``); without a
baseline the report says so instead of counting regressions.

Output (default ``~/oh-my-slam-data/evaluations/<UTC>/``, never inside the repository):
``result.json``, ``summary.md``, ``runs/`` (stdout, stderr, timings per command), ``outputs/``
(``-o`` / ``-d`` results) and ``maps/`` (the two maps). Progress goes to stderr; stdout gets the
path of ``summary.md``. Exit 0 when every metric passes, 1 when one fails, 2 on a usage error.
``--set-baseline`` stores the run as the baseline later runs are compared with.

Modules: ``names`` (capture-name grammar), ``runner`` / ``memory`` (running and measuring
commands), ``suite`` (the plan), ``contracts``, ``performance``, ``poses``, ``locate``,
``mapquality``, ``mapupdate``, ``segmentation``, ``groundtruth`` (metrics), ``viewer`` (browser
timing), ``service`` (``server.sh``) with ``proxy`` (record-or-replay inference), ``metrics``
(targets and baseline), ``report``.
"""
