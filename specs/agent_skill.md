# 2.7 Agent skill — `SKILL.md`

Part of the [high-level specification](high_level_spec.md) (§2 Components).

* The repository root holds an Agent Skill, `SKILL.md` (front matter `name: oh-my-slam-api`
  and a `description`), that lets an AI agent use the [§2.6](http_server.md) API with `curl`
  alone. It is one self-contained file needing only `sh` and `curl`; importing it means
  copying it to `<skills dir>/oh-my-slam-api/SKILL.md` on the Mac that runs `server.sh` or on
  any Linux or macOS machine on the LAN (the service binds `0.0.0.0`).
* **Generated, not hand-written.** Like the API, the skill follows the
  [single source of truth](http_server.md#single-source-of-truth): its operations,
  parameters, defaults, allowed values, result files and error codes are generated from the
  same shared definitions as `/api/openapi.json`, so a command that gains, changes or loses an
  option, an output or an error changes the skill without a hand-written edit. A test fails
  when the committed `SKILL.md` differs from the generated one or when an API route is
  missing from it. The skill tells the agent that a running service's `/api/openapi.json`
  wins where the two disagree (an older or newer service).
* **It describes every endpoint:** each route under `/api/`, with its method, parameters or
  body, what it does (read-only, writes a map, needs the inference server), a ready-to-run
  `curl` command (`-F` for uploads, `-N` for the event streams, `-o` for results and files), a
  sample response, and the API's error shape (the command's message and machine-readable
  code, with the HTTP status of the generic rule).
* **Job workflow.** The skill walks the agent through a whole job: upload the inputs (or name
  paths inside the workspace), validate and submit the operation, follow the job by polling
  or by its event stream until it ends, then download the result and each produced file.
  Results are byte-identical to the command's output, so the agent saves them with `-o` and
  never rewrites them; stage names and timings are read from the job.
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
  updating an existing map, before starting a long mapping job, and before cancelling a job
  it did not submit. Inputs are uploads or paths inside the workspace, never paths outside it,
  and an upload is consumed by one job, so re-submitting means uploading again.
