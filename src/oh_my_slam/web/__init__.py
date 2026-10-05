"""``server.sh`` — the web service of spec §2.6: an HTTP API over every command mode, derived from
the commands' shared definitions (``oh_my_slam.commands.spec``), a job runner that runs each job as
the command's own Python entry point in a subprocess, and the workspace that holds maps, uploads
and job results.

* ``workspace`` — ``<data>/maps``, ``<data>/uploads``, ``<data>/jobs``; paths confined to it.
* ``operations`` — one operation per command mode, its request turned into the command line.
* ``openapi`` — the OpenAPI document generated from ``spec.describe()``.
* ``jobs`` — queue, subprocesses, progress, cancellation, persistence.
* ``app`` — the HTTP routes (Starlette); ``main`` — the ``server.sh`` command line.

This package never loads a model, torch or Open3D: it is a client of the inference server, as the
commands are (import-linter contracts in ``pyproject.toml``)."""
