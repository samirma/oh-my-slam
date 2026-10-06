# 2.7 Agent skill — `SKILL.md`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

* The repository root holds an Agent Skill, `SKILL.md` (front matter `name: oh-my-slam-api`
  and a `description`), that lets an AI agent use the [§2.6](http_server.md) API with `curl`
  alone. It is one self-contained file needing only `sh` and `curl`; importing it means
  copying it to `<skills dir>/oh-my-slam-api/SKILL.md` on the Mac that runs `server.sh` or on
  any Linux or macOS machine on the LAN (the service binds `0.0.0.0`).
* **Description.** The front matter's `description` is what an agent reads to decide whether
  to use the skill, so it states all of the skill's capabilities and when it can be used. The
  capabilities are every operation, with what it produces, and every other feature of the API.
  When it can be used covers the requests it serves, the need for a `server.sh` reachable from
  the agent's machine, and which operations need the inference server. It is generated like
  the rest of the file, so a capability that is added, changed or removed changes it, and it
  stays within the Agent Skills limit of 1024 characters.
* **Generated, not hand-written.** Like the API, the skill follows the
  [single source of truth](http_server.md#single-source-of-truth): its operations,
  parameters, defaults, allowed values, results and error codes are generated from the
  same shared definitions as `/api/openapi.json`, so a command that gains, changes or loses an
  option, an output or an error changes the skill without a hand-written edit. A test fails
  when the committed `SKILL.md` differs from the generated one or when an API route is
  missing from it. The skill tells the agent that a running service's `/api/openapi.json`
  wins where the two disagree (an older or newer service).
* **It describes every endpoint:** each route under `/api/`, with its method, parameters or
  body, what it does (read-only, writes a map, needs the inference server), a ready-to-run
  `curl` command (`-T` for uploads, sending the raw file as the request body, since the service
  refuses `-F` forms; `-o` for results), a sample
  response, and the API's error shape (the command's message and machine-readable code, with
  the HTTP status of the generic rule).
* **Request workflow.** The skill walks the agent through a whole request: upload the inputs
  (or name paths inside the workspace), validate the request, then run the operation and
  wait for its answer. A mapping request can take many minutes and the service runs
  inference requests one at a time, so the agent keeps the connection open with no client
  timeout: disconnecting interrupts the command. Results are byte-identical to the
  command's output, so the agent saves them with `-o` and never rewrites them; stage names
  and timings are read from the response's `Server-Timing` header.
* **Server address.** `server.sh` binds a free port by default, so the URL changes from run to
  run. A shell snippet in the skill resolves the service's base URL and caches it in
  `${XDG_CACHE_HOME:-~/.cache}/oh-my-slam-api/server_url`, outside the repository. It uses, in
  order: a URL the user gives; the cached URL, if `/api/health` answers there within a few
  seconds; on the machine running the service, the URL the service records in its workspace
  (`server.json` in `~/oh-my-slam-data/`, or in the `--data` folder the user names), with
  `0.0.0.0` replaced by `127.0.0.1`; and finally asking the user for the URL that `server.sh`
  printed on start or that `server.sh --status` reports. There is no subnet or port scan, since
  the port is not known in advance; `server.sh --port <n>` keeps the URL stable for other
  machines. The URL it finds is written to the cache, and the search runs again only when the
  cached URL stops answering.
* **Safety.** The skill carries the project's rules. When the inference server is down,
  inference-requiring operations answer 503: the agent reports the start command from the
  health response instead of retrying, and still offers what works without it (persisted
  maps). The agent never starts or stops `server.sh` or the inference server, and never writes
  into a map's folder: maps change only through the mapping operation. It asks the user before
  updating an existing map and before starting a long mapping request. Inputs are uploads or
  paths inside the workspace, never paths outside it, and an upload is consumed by one
  request, so repeating a request means uploading again.
