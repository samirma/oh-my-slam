#!/usr/bin/env bash
# Checks the external tool oh-my-slam uses outside Python:
#   * Homebrew COLMAP 4.2.x (ALIKED/LightGlue feature extraction + matching through ONNX/CoreML)
set -euo pipefail

say() { echo "install_tools: $*" >&2; }

# --- COLMAP -----------------------------------------------------------------------------------
if ! command -v colmap >/dev/null 2>&1; then
  say "colmap not found — install it with: brew install colmap"
  exit 1
fi
colmap_version="$(colmap version 2>&1 | head -1)"
if [[ "${colmap_version}" != *"COLMAP 4.2."* ]]; then
  say "colmap must be 4.2.x (found: ${colmap_version}); pycolmap in .venv is 4.2.x"
  exit 1
fi
say "ok: ${colmap_version}"
