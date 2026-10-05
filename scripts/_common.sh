#!/usr/bin/env bash
# Shared by the six entry points: locate the repository, check the uv environment and exec the
# Python command-line module (oh_my_slam.cli.<name>, or a full oh_my_slam.* module path). The entry scripts never call each other (delegation happens at the
# Python API level, see README "Ownership").
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
  # module it imports — about +0.5 s and +40 MB on every command.
  unset PYTHONDONTWRITEBYTECODE
  if [[ "${module}" != oh_my_slam.* ]]; then
    module="oh_my_slam.cli.${module}"
  fi
  exec "${OMS_PY}" -m "${module}" "$@"
}
