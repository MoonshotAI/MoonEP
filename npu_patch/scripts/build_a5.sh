#!/usr/bin/env bash
# One-command Ascend 950 (A5) build + install for MoonEP.
#
# Thin wrapper: the build itself is implemented once in build_base.sh, which
# selects bisheng arch=a5 for -soc_type Ascend950. See that script for the full
# step list and the shared switches (BUILD_DIR, FORCE_SHMEM, SHMEM_HOME_PATH,
# ASCEND_MOONEP_RUNTIME_INCLUDE_DIR).
#
# Usage:
#   bash npu_patch/scripts/build_a5.sh [CANN_SET_ENV]
#   FORCE_SHMEM=1 bash npu_patch/scripts/build_a5.sh [CANN_SET_ENV]
#
# A5 notes:
#   - Only product with SIMT: the udma/urma kernels (test_npu/urma/*) are
#     runnable only here; every other SoC must not run --use_urma.
#   - Default build dir: build_ascend_a5.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/build_base.sh" -soc_type Ascend950 "$@"