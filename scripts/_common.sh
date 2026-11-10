#!/usr/bin/env bash
# Used by server.sh: locate the repository, check the uv environment and exec a Python module
# (a full oh_my_slam.* module path).
set -euo pipefail

OMS_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OMS_PY="${OMS_ROOT}/.venv/bin/python"

oms_exec() {
  local module="$1"
  shift
  if [[ ! -x "${OMS_PY}" ]]; then
    echo "oh-my-slam: Python environment missing — run 'uv sync' in ${OMS_ROOT}" >&2
    exit 2
  fi
  export PYTHONUNBUFFERED=1
  # Let Python cache bytecode (__pycache__, git-ignored): with PYTHONDONTWRITEBYTECODE inherited
  # (set by some agent/IDE shells) a checkout whose caches were never written recompiles every
  # module it imports.
  unset PYTHONDONTWRITEBYTECODE
  exec "${OMS_PY}" -m "${module}" "$@"
}
