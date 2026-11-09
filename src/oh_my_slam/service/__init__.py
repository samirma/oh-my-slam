"""``server.sh``: the oh-my-slam HTTP service for the robot-guide skill (``specs/api.md``).

One long-lived Starlette process serves ``/api/``. It owns no model: mapping runs through the
mapping, reconstruction and segmentation APIs, which reach the inference server. Every JSON body
follows ``api.schema.json`` (a copy of ``specs/api.schema.json``, schema version ``oms/2.0``).

Modules:

* ``schema`` — the JSON Schema and its validators;
* ``errors`` — the typed errors of api.md §8;
* ``routes`` — the endpoint definitions, from which ``app`` routes and ``openapi`` documents;
* ``app`` — the Starlette application;
* ``store`` — maps, cameras and versions on disk;
* ``jobs`` — update jobs and their worker;
* ``update`` — an update job: per-frame checks, registration, anchoring, commit;
* ``anchor`` — similarity transforms and the robust fit that anchors the map frame;
* ``cameras`` — the requests' cameras, rectified to the pinhole the mapper sees;
* ``checks`` — per-frame image measurements behind the rejection reasons;
* ``backend`` — the mapper and depth model behind an update (``mapper_proc`` runs the mapper);
* ``inference`` — the inference server's state, started and stopped with the service;
* ``main`` — the command line.
"""
