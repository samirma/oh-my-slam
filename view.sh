#!/usr/bin/env bash
# Thin wrapper: see oh_my_slam.cli.view.
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"
oms_exec view "$@"
