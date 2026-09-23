#!/usr/bin/env bash
# Installs / checks the external tools oh-my-slam uses outside Python:
#   * Homebrew COLMAP 4.2.x (ALIKED/LightGlue feature extraction + matching through ONNX/CoreML)
#   * OpenMVS 2.4.0 macOS arm64 binaries (TextureMesh), sha256-pinned; optional — the mapper
#     falls back to its NumPy atlas texturing when it is missing.
set -euo pipefail

OPENMVS_VERSION="2.4.0"
OPENMVS_URL="https://github.com/cdcseacave/openMVS/releases/download/v${OPENMVS_VERSION}/OpenMVS_macOS_arm64.zip"
OPENMVS_SHA256="3d4c616c97031b1ab6e2eecb0ddd5614fb99513c0782a32c9350602faf38799b"
TOOLS_DIR="${OH_MY_SLAM_TOOLS_DIR:-${HOME}/.local/share/oh-my-slam/tools}"
OPENMVS_DIR="${TOOLS_DIR}/openmvs-${OPENMVS_VERSION}"

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

# --- OpenMVS ----------------------------------------------------------------------------------
if [[ -x "${OPENMVS_DIR}/TextureMesh" ]]; then
  say "ok: OpenMVS ${OPENMVS_VERSION} at ${OPENMVS_DIR}"
  exit 0
fi
mkdir -p "${TOOLS_DIR}"
tmp_zip="$(mktemp -t openmvs).zip"
trap 'rm -f "${tmp_zip}"' EXIT
say "downloading ${OPENMVS_URL}"
curl -fsSL -o "${tmp_zip}" "${OPENMVS_URL}"
actual="$(shasum -a 256 "${tmp_zip}" | awk '{print $1}')"
if [[ "${actual}" != "${OPENMVS_SHA256}" ]]; then
  say "sha256 mismatch for OpenMVS zip: expected ${OPENMVS_SHA256}, got ${actual}"
  exit 1
fi
staging="$(mktemp -d "${TOOLS_DIR}/.openmvs.XXXXXX")"
unzip -q "${tmp_zip}" -d "${staging}"
# The archive may contain a top-level folder; locate TextureMesh and flatten to OPENMVS_DIR.
texture_mesh="$(find "${staging}" -type f -name TextureMesh | head -1)"
if [[ -z "${texture_mesh}" ]]; then
  say "TextureMesh not found inside the archive"
  rm -rf "${staging}"
  exit 1
fi
rm -rf "${OPENMVS_DIR}"
mv "$(dirname "${texture_mesh}")" "${OPENMVS_DIR}"
rm -rf "${staging}"
xattr -dr com.apple.quarantine "${OPENMVS_DIR}" 2>/dev/null || true
chmod +x "${OPENMVS_DIR}"/* 2>/dev/null || true
# OpenMVS writes a <Tool>-<id>.log into its working folder (default: the current directory), so
# probe it with a throw-away working folder instead of the caller's directory.
probe_dir="$(mktemp -d -t openmvs-probe)"
probe_rc=0
(cd "${probe_dir}" && "${OPENMVS_DIR}/TextureMesh" --help -w "${probe_dir}" >/dev/null 2>&1) \
  || probe_rc=$?
rm -rf "${probe_dir}"
if [[ ${probe_rc} -le 1 ]]; then
  say "ok: OpenMVS ${OPENMVS_VERSION} installed at ${OPENMVS_DIR}"
else
  say "OpenMVS installed but TextureMesh does not run (Gatekeeper?); atlas texturing will be used"
fi
