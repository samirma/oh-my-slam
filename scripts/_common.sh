#!/usr/bin/env bash
# Shared by the five entry points: locate the repository, check the uv environment and exec the
# Python command-line module. The entry scripts never call each other (delegation happens at the
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
  exec "${OMS_PY}" -m "oh_my_slam.cli.${module}" "$@"
}
