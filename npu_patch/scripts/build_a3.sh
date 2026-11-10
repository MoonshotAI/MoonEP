#!/usr/bin/env bash
# One-command Ascend 910_93 (A3) build + install for MoonEP.
#
# Thin wrapper: the build itself is implemented once in build_base.sh, which
# selects bisheng arch=a2 (non-SIMT, 2201) for -soc_type Ascend910_93. See that
# script for the full step list and the shared switches (BUILD_DIR, FORCE_SHMEM,
# SHMEM_HOME_PATH, ASCEND_MOONEP_RUNTIME_INCLUDE_DIR).
#
# Usage:
#   bash npu_patch/scripts/build_a3.sh [CANN_SET_ENV]
#   FORCE_SHMEM=1 bash npu_patch/scripts/build_a3.sh [CANN_SET_ENV]
#
# A3 notes:
#   - Non-SIMT (2201): the UDMA/URMA kernels and tests (test_npu/urma/*) are
#     A5-only and are excluded from this build; do not run --use_urma here.
#   - Default build dir: build_ascend_a3.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/build_base.sh" -soc_type Ascend910_93 "$@"
