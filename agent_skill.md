# 2.7 Agent skill — `SKILL.md`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

The project ships one Agent Skill so that an AI agent can use every feature without help,
through both kinds of entry point the project offers: the local scripts and the `server.sh`
network API ([§2.6](specs/http_server.md)).

* **Location and name.** `SKILL.md` at the repository root, with front matter `name: oh-my-slam`
  and a `description`.
* **Self-contained.** One file that needs only the project's own entry points and standard tools
  (`sh`, and `curl` for the API). Installing it means copying it to
  `<skills dir>/oh-my-slam/SKILL.md`: on the Mac that holds the checkout, where both the scripts
  and the API are usable, or on any Linux or macOS machine on the LAN, where only the API is
  (the service binds `0.0.0.0`).
* **Description.** The front matter's `description` is what an agent reads to decide whether to
  use the skill, so it states every capability, what each produces, and when the skill can be
  used: the requests it serves, the need for the checkout (scripts) or for a `server.sh`
  reachable from the agent's machine (API), and which operations need the inference server.
  It stays within the Agent Skills limit of 1024 characters.
* **Finds the project.**
  * *Scripts.* The entry points are the shell scripts at the root of the checkout. A shell
    snippet resolves the checkout in order: a path the user gives; the cached path, if it still
    holds the scripts; and otherwise it asks the user. The path it finds is cached in
    `${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/repo`. The skill says how to run the scripts
    (from any directory, with absolute paths) and what they need first (`uv sync` once; the
    inference server for the operations that use it).
  * *API.* `server.sh` binds a free port by default, so its URL changes from run to run. A shell
    snippet resolves the base URL in order: a URL the user gives; the cached URL, if
    `/api/health` answers there within a few seconds; on the machine running the service, the
    URL the service records in its workspace (`server.json` in `~/oh-my-slam-data/`, or in the
    `--data` folder the user names), with `0.0.0.0` replaced by `127.0.0.1`; and otherwise it
    asks the user for the URL that `server.sh` printed on start or that `server.sh --status`
    reports. The URL it finds is cached in `${XDG_CACHE_HOME:-$HOME/.cache}/oh-my-slam/server_url`,
    and the search runs again only when the cached URL stops answering. There is no subnet or
    port scan; `server.sh --port <n>` keeps the URL stable for other machines.
* **Covers every script.** For each script and mode — `reconstruct.sh`, `mapper.sh update`,
  `mapper.sh locate`, `segment.sh`, `view.sh`, `start_inference_server.sh` and `server.sh`
  (`--status` only) — the skill gives where it is, its parameters (with their defaults and
  allowed values), what it does (read-only, writes a file, writes a map, needs the inference
  server, runs until interrupted), a ready-to-run command, a sample result and the error shape
  (the exit status and its meaning, and the one-line message on stderr).
* **Points to the API description.** For the API the skill gives the URL of the service's
  `/api/openapi.json`, which the agent reads to discover every operation, its parameters,
  results and errors; the running service's document wins over anything the skill says. The
  API part therefore stays concise and does not repeat that document. It adds only what the
  document cannot say:
  * how to find the service (above);
  * the order of calls in a typical request: upload the inputs (`curl -T`, the raw file as the
    body; the service refuses `-F` forms) or name paths inside the workspace, validate, run,
    and save the result with `-o`, byte for byte; stage timings are in the `Server-Timing`
    header. A mapping request can take many minutes and inference requests run one at a time,
    so the agent keeps the connection open with no client timeout: disconnecting interrupts
    the command;
  * the error shape (the command's message and machine-readable code, with the HTTP status of
    the generic rule);
  * the project's rules (below) and a few ready-to-run `curl` examples.
* **Carries the project's rules.** Limits, lifetimes, gated actions and what the agent must ask
  the user first:
  * When the inference server is down, the operations that need it fail (exit status 3, or
    HTTP 503): the agent reports the start command instead of retrying, and still offers what
    works without it (persisted maps).
  * The agent never starts or stops `server.sh` or the inference server and never writes into
    a map's folder (maps change only through `mapper.sh update` or the mapping operation).
  * It asks the user before updating an existing map and before starting a long mapping
    request.
  * API inputs are uploads or paths inside the workspace, never paths outside it; an upload is
    consumed by the one request it is given to, so repeating a request means uploading again.
  * It never offers an action, mode, option or endpoint that the project does not support.
* **Stays in step with the project.** `SKILL.md` is generated, never hand-edited, from the same
  shared definitions as the commands and `/api/openapi.json`, so a script, mode, option,
  output, error or limit that is added, changed or removed changes the skill. A test fails when
  the committed `SKILL.md` differs from the generated one, when a script mode, option or API
  route is missing from it, when it offers an action, mode, option or endpoint the project does
  not support, or when a limit it states differs from the value in the code.
