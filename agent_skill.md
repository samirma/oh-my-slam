# 2.7 Agent skill — `SKILL.md`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

The project ships one Agent Skill so that an AI agent can use every operation of the
`server.sh` network API ([§2.6](specs/http_server.md)) without help. The API is the skill's
only entry point. The local scripts are out of its scope: the skill never runs
`reconstruct.sh`, `mapper.sh`, `segment.sh`, `view.sh`, `start_inference_server.sh` or
`server.sh`, nor documents their options. It names a script only to tell the user what to run
(starting the inference server or `server.sh`, or `server.sh --status` for the service's URL),
and what only a script does (viewing an image or a map in `view.sh`) is left to the user.

* **Location and name.** `SKILL.md` at the repository root, with front matter `name: oh-my-slam`
  and a `description`.
* **Self-contained.** One file that needs only standard tools: `sh` and `curl`, the HTTP client
  through which the skill calls the API. Installing it means copying it to
  `<skills dir>/oh-my-slam/SKILL.md` on any Linux or macOS machine that reaches the service:
  the Mac that runs `server.sh`, or any machine on the LAN (the service binds `0.0.0.0`). It
  needs no checkout.
* **Description.** The front matter's `description` is what an agent reads to decide whether to
  use the skill, so it states every capability, what each produces, and when the skill can be
  used: the requests it serves, the need for a `server.sh` reachable from the agent's machine,
  and which operations need the inference server. It stays within the Agent Skills limit of
  1024 characters.
* **Finds the service.** `server.sh` binds a free port by default, so its URL changes from run to
  run. A shell snippet resolves the base URL in order: a URL the user gives; the cached URL, if
  `/api/health` answers there within a few seconds; on the machine running the service, the URL
  the service records in its workspace (`server.json` in `~/oh-my-slam-data/`, or in the
  `--data` folder the user names), with `0.0.0.0` replaced by `127.0.0.1`; and otherwise it
  asks the user for the URL that `server.sh` printed on start or that `server.sh --status`
  reports. The URL it finds is cached in `${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/server_url`,
  and the search runs again only when the cached URL stops answering. There is no subnet or
  port scan; `server.sh --port <n>` keeps the URL stable for other machines.
* **Points to the API description.** The skill gives the URL of the service's
  `/api/openapi.json`, which the agent reads to discover every operation, its parameters,
  results and errors; the running service's document wins over anything the skill says. The
  skill therefore stays concise and does not repeat that document. It adds only what the
  document cannot say:
  * how to find the service (above);
  * every operation and route by name, with what it does: read-only or writes a map, and
    whether it needs the inference server;
  * the order of calls in a typical request: upload the inputs (`curl -T`, the raw file as the
    body; the service refuses `-F` forms) or name paths inside the workspace, validate, run,
    and save the result with `-o`, byte for byte; stage timings are in the `Server-Timing`
    header. A mapping request can take many minutes and inference requests run one at a time,
    so the agent keeps the connection open with no client timeout: disconnecting interrupts
    the command;
  * the error shape (the command's message and machine-readable code, with the HTTP status of
    the generic rule) and the service's own refusals;
  * the project's rules (below) and a ready-to-run `curl` example for each operation.
* **Carries the project's rules.** Limits, lifetimes, gated actions and what the agent must ask
  the user first:
  * When the inference server is down, the operations that need it fail (HTTP 503): the agent
    reports the start command that the message and `/api/health` name instead of retrying, and
    still offers what works without it (the workspace's maps, and the operations that do not
    need it).
  * The agent never starts or stops `server.sh` or the inference server; when either is not
    running, it tells the user how to start it. It never changes a map except through the
    mapping operation.
  * It asks the user before updating an existing map and before starting a long mapping
    request.
  * API inputs are uploads or paths inside the workspace, never paths outside it; an upload is
    consumed by the one request it is given to, so repeating a request means uploading again.
  * It never offers an operation, parameter, value or endpoint that the service does not
    support, nor any local script.
* **Stays in step with the project.** `SKILL.md` is generated, never hand-edited, from the same
  shared definitions as `/api/openapi.json`, so an operation, parameter, output, error or limit
  that is added, changed or removed changes the skill. A test fails when the committed
  `SKILL.md` differs from the generated one, when an API operation or route is missing from it,
  when it offers an operation, parameter, endpoint or script that the service does not
  support, or when a limit it states differs from the value in the code.
