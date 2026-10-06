# 2.1 Inference server — `start_inference_server.sh`

Part of the [high-level specification](../high_level_spec.md) (§2 Components).

```sh
start_inference_server.sh
start_inference_server.sh --status | --stop
```

Loads the monocular depth-estimation model, the instance-segmentation model, and any other
service or model needed for mapping, and keeps them resident, so that individual
reconstructions do not pay model start-up cost. `reconstruct.sh`, `mapper.sh update`,
`segment.sh` and `view.sh -i` use this server. Any operation that requires inference must
fail with a clear, actionable error if the server is not running. Operations on an already
persisted map (`view.sh -m`) must not require the inference server.

* `--status` — print the server's health JSON on stdout; fail with the server-unavailable
  error if it is not running.
* `--stop` — stop a running server.
