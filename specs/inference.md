# Inference server and lanes

Part of the [high-level specification](../high_level_spec.md) (§8, §10).

* **What it serves.** The inference server keeps every model resident:
  * monocular depth;
  * gravity and intrinsics;
  * segmentation;
  * multi-view registration fallbacks;
  * whatever else mapping needs.

  So no request pays a model's start-up cost. `server.sh` starts and stops it, and talks to it
  over a local socket. No other process loads a model.
* **Lanes.**
  * The **interactive** lane serves locate, frame analysis, dry runs and the registration phase
    of jobs.
  * The **bulk** lane serves fusion.
  * Bulk work is split into steps short enough that an interactive request waits at most one
    step. Each step runs one frame's inference, so a step is bounded by that frame's model time.
  * On a device that cannot run two models at once, such as the single MPS queue of Apple
    silicon, the lanes are priority queues: the interactive lane always runs first.
  * On a CUDA host they may run concurrently.
* **Health.** `GET /api/health` [`Health`] reports:
  * the service state, and the inference server state (`ready`, `starting`, `degraded`, `down`);
  * device memory in use against its cap;
  * per lane: `idle` / `busy`, queue length, and the age of the oldest item.

  When the inference server is down, an operation that needs it answers
  `inference_unavailable`. Reads of committed versions never need it.
* **Restart.** A device out-of-memory error answers `out_of_memory` for that request. The server
  frees and recovers without a manual restart, and reports `degraded` until it has.
