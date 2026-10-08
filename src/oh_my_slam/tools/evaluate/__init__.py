"""Benchmark evaluator of every entry point on ``examples/`` (high_level_spec.md §5).

    uv run python -m oh_my_slam.tools.evaluate [--out DIR] [--set-baseline] [--baseline PATH]
                                               [--targets PATH] [--street2 PATH]

Runs, strictly one at a time, ``start_inference_server.sh`` (cold start, resident memory),
``reconstruct.sh`` / ``segment.sh -i`` / ``view.sh -i`` on ``restaurant.jpg``; on each capture
sequence (``ainex-captures``, ``camera``) ``segment.sh -i`` on every frame, ``mapper.sh update``
on the sequence in one update and split across 3 updates and ``view.sh -m`` on the one-update map;
``mapper.sh locate`` (held-out ``ainex-captures`` captures, and the reference map: the one-update
``ainex-captures`` map), ``mapper.sh update`` on ``office_sequence`` (one update and its annotated
splits) and on the street2 video, and ``server.sh`` (performance, parity with the commands, UI).
Metrics: performance (end to end and per stage: time, client and server peak memory), pose
accuracy, map quality, map update, segmentation, contracts, the web service, and ground truth when
annotations exist under ``examples/ground_truth/``. Each has a target in ``examples/targets.json``
(data) and is compared with the stored baseline run
(``~/oh-my-slam-data/evaluations/baseline.json``); without a baseline the report says so instead
of counting regressions, and it lists the metrics the baseline has no value for (not compared).

Output (default ``~/oh-my-slam-data/evaluations/<UTC>/``, never inside the repository):
``result.json``, ``summary.md``, ``runs/`` (stdout, stderr, timings per command), ``outputs/``
(``-o`` / ``-d`` results) and ``maps/`` (the maps it builds). Progress and the path of
``summary.md`` go to stderr; stdout stays empty. Exit 0 when every metric passes, 1 when one
fails, 2 on a usage error.
``--set-baseline`` stores the run as the baseline later runs are compared with.

Modules: ``names`` (capture-name grammars), ``runner`` / ``memory`` (running and measuring
commands), ``suite`` (the plan), ``contracts``, ``performance``, ``poses``, ``locate``,
``mapquality``, ``mapupdate``, ``segmentation``, ``groundtruth`` (metrics), ``viewer`` (browser
timing), ``service`` (``server.sh``) with ``proxy`` (record-or-replay inference), ``metrics``
(targets and baseline), ``report``.
"""
