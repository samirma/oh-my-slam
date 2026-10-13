"""``server.sh`` — the web service of spec §2.6: an HTTP API over every mode of the commands the
registry offers to it (``reconstruct.sh``, ``mapper.sh``, ``segment.sh``), derived from the
commands' shared definitions (``oh_my_slam.commands.spec``), and two operations on a map of the
workspace (removing it; starting or stopping its ``view.sh`` viewer). There are no jobs: an
operation runs within its own request, as the command's own Python entry point in a subprocess,
and its response is the command's result.

* ``workspace`` — ``<data>/maps``, ``<data>/uploads``; paths confined to it; a map's removal.
* ``operations`` — one operation per command mode, its request turned into the command line.
* ``openapi`` — the OpenAPI document generated from ``spec.describe()``.
* ``runner`` — the order requests run in, their subprocesses, interruption.
* ``viewers`` — the maps' ``view.sh -m`` viewers, at most one per map.
* ``app`` — the HTTP routes (Starlette); ``main`` — the ``server.sh`` command line.
* ``skill`` — the generator of the agent skill ``SKILL.md`` (spec §2.7).

This package never loads a model, torch or Open3D: it is a client of the inference server, as the
commands are (import-linter contracts in ``pyproject.toml``)."""
