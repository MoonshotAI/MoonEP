#!/usr/bin/env bash
# MoonEP Ascend (NPU) base build -- the single implementation of the build logic.
#
# Per-SoC wrapper scripts only pass -soc_type in; partial builds are not supported:
#   bash npu_patch/scripts/build_a5.sh [CANN_SET_ENV]        # Ascend950 / A5
#   bash npu_patch/scripts/build_a3.sh [CANN_SET_ENV]        # Ascend910_93 / A3
#   bash npu_patch/scripts/build_base.sh -soc_type Ascend950 [CANN_SET_ENV]
#
# Steps:
#   1. (auto)source CANN set_env.sh when ASCEND_HOME_PATH is unset
#   2. git submodule update --init --recursive (cann/shmem, catlass)
#   3. build cann/shmem for the SoC -> npu_patch/3rdparty/shmem/install
#   4. cmake -S npu_patch/native -B <build dir> -> ascend_moonep_kernels
#   5. stage the kernel + shmem .so into npu_patch/lib
#   6. pip install -e . (compiles the npu_patch._C NPU extension)
#
# Environment:
#   BUILD_DIR                          cmake build dir (default: build_ascend_<a2|a3|a5>)
#   SHMEM_HOME_PATH                    reuse a prebuilt cann/shmem install
#   FORCE_SHMEM=1                      rebuild cann/shmem even when an install exists
#   ASCEND_MOONEP_RUNTIME_INCLUDE_DIR  override the CANN runtime header directory
#
# NOTE: the kernel .so and the npu_patch._C host extension are linked, so all
# three steps (kernel lib, shmem lib, pip install) must run; no single step suffices.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NATIVE_DIR="${ROOT_DIR}/npu_patch/native"
LIB_TARGET="${ROOT_DIR}/npu_patch/lib"
SHMEM_SOURCE_DIR="${ROOT_DIR}/npu_patch/3rdparty/shmem"

usage() {
    cat <<'EOF'
Usage: bash npu_patch/scripts/build_base.sh -soc_type <SoC> [CANN_SET_ENV]

  -soc_type {Ascend910B|Ascend910_93|Ascend950}   target SoC (typically Ascend950)
  CANN_SET_ENV            optional path to the CANN set_env.sh; when omitted it
                          is auto-discovered under /usr/local/Ascend/*/set_env.sh

Examples:
  bash npu_patch/scripts/build_base.sh -soc_type Ascend950
  bash npu_patch/scripts/build_base.sh -soc_type Ascend910_93
  bash npu_patch/scripts/build_a5.sh /usr/local/Ascend/ascend-toolkit/set_env.sh
  FORCE_SHMEM=1 BUILD_DIR=/tmp/b_a5 bash npu_patch/scripts/build_a5.sh
EOF
}

SOC_TYPE=""
CANN_ENV=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -soc_type|--soc_type)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] -soc_type requires a value, for example: -soc_type Ascend950"
                usage
                exit 1
            fi
            SOC_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
        *)
            if [[ -n "${CANN_ENV}" ]]; then
                echo "[ERROR] Only one positional argument (CANN set_env.sh) is accepted."
                usage
                exit 1
            fi
            CANN_ENV="$1"
            shift
            ;;
    esac
done

case "${SOC_TYPE}" in
    Ascend910B)
        ARCH=a2
        BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build_ascend_a2"}
        ;;
    Ascend910_93)
        # A3 uses the same non-SIMT Bisheng backend as A2; the product
        # distinction is carried by the explicit ASCEND_MOONEP_ARCH_A3 macro.
        ARCH=a2
        BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build_ascend_a3"}
        ;;
    Ascend950)
        ARCH=a5
        BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build_ascend_a5"}
        ;;
    "")
        echo "[ERROR] -soc_type is required."
        usage
        exit 1
        ;;
    *)
        echo "[ERROR] Unsupported -soc_type: ${SOC_TYPE}"
        echo "        Supported values: Ascend910B, Ascend910_93, Ascend950"
        exit 1
        ;;
esac

# 1. CANN environment.
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ -z "${CANN_ENV}" ]]; then
        CANN_ENV=$(ls -d /usr/local/Ascend/*/set_env.sh 2>/dev/null | head -n1 || true)
    fi
    if [[ -n "${CANN_ENV}" && -f "${CANN_ENV}" ]]; then
        echo "[INFO] sourcing ${CANN_ENV}"
        # shellcheck disable=SC1090
        source "${CANN_ENV}"
    fi
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Pass the CANN set_env.sh as an argument."
    exit 1
fi
echo "[INFO] ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
echo "[INFO] SOC_TYPE: ${SOC_TYPE}"
echo "[INFO] ASCEND_MOONEP_BISHENG_ARCH: ${ARCH}"
echo "[INFO] BUILD_DIR: ${BUILD_DIR}"

# CANN products lay out runtime headers differently; this is only a non-fatal
# precheck -- the hard failure is left to cmake (same path and message) to avoid
# false aborts on A5.
RUNTIME_INCLUDE_DIR="${ASCEND_MOONEP_RUNTIME_INCLUDE_DIR:-${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime/runtime}"
if [[ ! -f "${RUNTIME_INCLUDE_DIR}/rt.h" ]]; then
    echo "[WARN] CANN runtime header not found: ${RUNTIME_INCLUDE_DIR}/rt.h"
    echo "[WARN] Set ASCEND_MOONEP_RUNTIME_INCLUDE_DIR if CANN uses a different layout."
fi

# 2. Dependency trees (git submodules).
if [[ -d "${ROOT_DIR}/.git" ]]; then
    echo "[INFO] git submodule update --init --recursive"
    git -C "${ROOT_DIR}" submodule update --init --recursive || \
        echo "[WARN] submodule update failed; continue if 3rdparty is already present."
fi

# 3. cann/shmem (one install shared by all SoCs; rebuild with FORCE_SHMEM=1 when switching SoC).
SHMEM_HOME_PATH="${SHMEM_HOME_PATH:-${SHMEM_SOURCE_DIR}/install}"
if [[ ! -f "${SHMEM_SOURCE_DIR}/scripts/build.sh" ]]; then
    echo "[ERROR] cann/shmem submodule is not initialized."
    echo "        Run: git submodule update --init --recursive"
    exit 1
fi
if [[ "${FORCE_SHMEM:-0}" == "1" || ! -f "${SHMEM_HOME_PATH}/set_env.sh" ]]; then
    echo "[INFO] Building cann/shmem (${SHMEM_SOURCE_DIR}) for ${SOC_TYPE}..."
    pushd "${SHMEM_SOURCE_DIR}" >/dev/null
    bash scripts/build.sh -soc_type "${SOC_TYPE}"
    popd >/dev/null
    if [[ ! -f "${SHMEM_HOME_PATH}/set_env.sh" ]]; then
        echo "[ERROR] SHMEM build finished but ${SHMEM_HOME_PATH}/set_env.sh was not generated."
        exit 1
    fi
else
    echo "[INFO] Reusing SHMEM installation: ${SHMEM_HOME_PATH}"
fi
echo "[INFO] SHMEM_HOME_PATH: ${SHMEM_HOME_PATH}"

# 4/5. Configure + build the kernel library.
echo "[INFO] cmake -S ${NATIVE_DIR} -B ${BUILD_DIR} (${SOC_TYPE})"
# Each SoC gets its own build dir (the cmake cache pins arch/SIMT; reusing it
# across SoCs fails silently). Also remove the legacy shared build_ascend dir.
rm -rf "${BUILD_DIR}" "${ROOT_DIR}/build_ascend"
CMAKE_ARGS=(
    -S "${NATIVE_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng"
    -DASCEND_MOONEP_SOC_TYPE="${SOC_TYPE}"
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}"
)
if [[ -n "${ASCEND_MOONEP_RUNTIME_INCLUDE_DIR:-}" ]]; then
    CMAKE_ARGS+=(-DASCEND_MOONEP_RUNTIME_INCLUDE_DIR="${ASCEND_MOONEP_RUNTIME_INCLUDE_DIR}")
fi
cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --target ascend_moonep_kernels --parallel

mkdir -p "${LIB_TARGET}"
cp "${BUILD_DIR}/lib/libascend_moonep_kernels.so" "${LIB_TARGET}/"
find "${SHMEM_HOME_PATH}/shmem/lib" \
    -maxdepth 1 \( -type f -o -type l \) \( -name "*.so" -o -name "*.so.*" \) \
    -exec cp -P {} "${LIB_TARGET}/" \;

# 6. Compile + install the npu_patch._C extension.
echo "[INFO] pip install -e npu_patch (NPU backend)"
# Force the NPU build path; otherwise setup.py falls back to the CUDA/GPU branch
# when torch_npu is not importable during the pip step.
export MOONEP_BUILD_BACKEND=npu
export ASCEND_MOONEP_KERNEL_LIB_DIR="${LIB_TARGET}"
export SHMEM_HOME_PATH
# Export backend and UDMA support separately for the host extension build.
# Only the current A5 build ships UDMA kernels.
if [[ "${ARCH}" == "a5" ]]; then
    export ASCEND_MOONEP_SIMT=1
    export ASCEND_MOONEP_UDMA=1
else
    export ASCEND_MOONEP_SIMT=0
    export ASCEND_MOONEP_UDMA=0
fi
# Product architecture for acl_shared_buffer.cpp's VMM export macro.
export ASCEND_MOONEP_PRODUCT="${SOC_TYPE}"
pushd "${ROOT_DIR}/npu_patch" >/dev/null
python -m pip install -e . --no-build-isolation --no-deps
popd >/dev/null

echo "[INFO] Build completed."
echo "[INFO] Kernel library: ${LIB_TARGET}/libascend_moonep_kernels.so"