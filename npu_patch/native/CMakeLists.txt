cmake_minimum_required(VERSION 3.16)

# CANN environment.
if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first.")
    endif()
endif()

set(CCEC "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" CACHE FILEPATH "Bisheng compiler")
if(NOT EXISTS "${CCEC}")
    message(FATAL_ERROR "Bisheng compiler not found: ${CCEC}")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${CCEC}" CACHE FILEPATH "CXX compiler")
endif()

project(ascend_moonep LANGUAGES CXX)

get_filename_component(ASCEND_MOONEP_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
if(NOT ASCEND_MOONEP_CXX_COMPILER_NAME MATCHES "bisheng")
    message(FATAL_ERROR
        "ascend_moonep kernels must be configured with Bisheng, but CMAKE_CXX_COMPILER="
        "${CMAKE_CXX_COMPILER}. Remove the existing build directory or pass "
        "-DCMAKE_CXX_COMPILER=${CCEC}.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ASCEND_MOONEP_SOC_TYPE "Ascend950" CACHE STRING "Target SoC: Ascend910B, Ascend910_93, or Ascend950")
set_property(CACHE ASCEND_MOONEP_SOC_TYPE PROPERTY STRINGS Ascend910B Ascend910_93 Ascend950)
if(ASCEND_MOONEP_SOC_TYPE STREQUAL "Ascend910B")
    set(ASCEND_MOONEP_BISHENG_ARCH a2)
    set(ASCEND_MOONEP_PRODUCT_DEFINITION ASCEND_MOONEP_ARCH_A2=1)
    set(ASCEND_MOONEP_NPU_DEFINITION ASCEND_MOONEP_ARCH_2201=1)
    set(ASCEND_MOONEP_CATLASS_ARCH 2201)
    set(ASCEND_MOONEP_SIMT OFF)
    set(ASCEND_MOONEP_UDMA OFF)
elseif(ASCEND_MOONEP_SOC_TYPE STREQUAL "Ascend910_93")
    set(ASCEND_MOONEP_BISHENG_ARCH a2)
    set(ASCEND_MOONEP_PRODUCT_DEFINITION ASCEND_MOONEP_ARCH_A3=1)
    set(ASCEND_MOONEP_NPU_DEFINITION ASCEND_MOONEP_ARCH_2201=1)
    set(ASCEND_MOONEP_CATLASS_ARCH 2201)
    set(ASCEND_MOONEP_SIMT OFF)
    set(ASCEND_MOONEP_UDMA OFF)
elseif(ASCEND_MOONEP_SOC_TYPE STREQUAL "Ascend950")
    set(ASCEND_MOONEP_BISHENG_ARCH a5)
    set(ASCEND_MOONEP_PRODUCT_DEFINITION ASCEND_MOONEP_ARCH_A5=1)
    set(ASCEND_MOONEP_NPU_DEFINITION ASCEND_MOONEP_ARCH_3510=1)
    set(ASCEND_MOONEP_CATLASS_ARCH 3510)
    set(ASCEND_MOONEP_SIMT ON)
    set(ASCEND_MOONEP_UDMA ON)
else()
    message(FATAL_ERROR
        "ASCEND_MOONEP_SOC_TYPE must be one of: Ascend910B, Ascend910_93, Ascend950; "
        "got '${ASCEND_MOONEP_SOC_TYPE}'")
endif()

set(ASCEND_MOONEP_ARCH_DEFINITIONS
    ${ASCEND_MOONEP_PRODUCT_DEFINITION}
    ${ASCEND_MOONEP_NPU_DEFINITION})
# Expose UDMA kernel support to the host wrappers.
if(ASCEND_MOONEP_UDMA)
    list(APPEND ASCEND_MOONEP_ARCH_DEFINITIONS ASCEND_MOONEP_UDMA=1)
endif()

# Compile UDMA kernels only when UDMA support is enabled.
if(ASCEND_MOONEP_UDMA)
    set(ASCEND_MOONEP_UDMA_KERNEL_SOURCES
        kernels/moonep_combine_udma.cpp
        kernels/moonep_dispatch_udma.cpp
        kernels/moonep_grad_reduce_udma.cpp
        kernels/prefetch_udma.cpp)
else()
    set(ASCEND_MOONEP_UDMA_KERNEL_SOURCES "")
endif()

# The planning kernel has one backend per instruction-set family, selected at
# build time; the two backends export the same planning_kernel_do symbol, so
# exactly one may be compiled into a target.
if(ASCEND_MOONEP_SIMT)
    set(ASCEND_MOONEP_PLANNING_KERNEL_SOURCE kernels/planning_simt.cpp)
else()
    set(ASCEND_MOONEP_PLANNING_KERNEL_SOURCE kernels/planning_aicore.cpp)
endif()

message(STATUS
    "ascend-moonep target: soc=${ASCEND_MOONEP_SOC_TYPE}, "
    "bisheng_arch=${ASCEND_MOONEP_BISHENG_ARCH}, npu_arch=${ASCEND_MOONEP_CATLASS_ARCH}, "
    "simt=${ASCEND_MOONEP_SIMT}")

# SHMEM is managed as a submodule and built into 3rdparty/shmem/install by default.
set(ASCEND_MOONEP_SHMEM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../3rdparty/shmem" CACHE PATH "SHMEM source directory")
execute_process(
    COMMAND git -C "${ASCEND_MOONEP_SHMEM_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE ASCEND_MOONEP_SHMEM_GIT_RC
    OUTPUT_VARIABLE ASCEND_MOONEP_SHMEM_ACTUAL_SHA
    ERROR_VARIABLE ASCEND_MOONEP_SHMEM_GIT_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ASCEND_MOONEP_SHMEM_GIT_RC EQUAL 0)
    message(FATAL_ERROR
        "Cannot read the cann/shmem submodule revision: "
        "${ASCEND_MOONEP_SHMEM_GIT_ERROR}. Run git submodule update --init --recursive.")
endif()
if(NOT DEFINED SHMEM_HOME_PATH)
    if(DEFINED ENV{SHMEM_HOME_PATH})
        set(SHMEM_HOME_PATH "$ENV{SHMEM_HOME_PATH}")
    else()
        set(SHMEM_HOME_PATH "${ASCEND_MOONEP_SHMEM_SOURCE_DIR}/install")
    endif()
endif()

if(NOT EXISTS "${SHMEM_HOME_PATH}/shmem/include/shmem.h")
    message(FATAL_ERROR
        "SHMEM installation not found under ${SHMEM_HOME_PATH}. "
        "Run the matching npu_patch/scripts/build_<soc>.sh first.")
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include("${CMAKE_SOURCE_DIR}/cmake/AscendMoonEPKernel.cmake")

# All device kernels and their host launch wrappers are linked into one library.
# The UDMA/URMA kernels require the SIMT backend (3510 / A5); they must be
# excluded from non-SIMT (A2/A3, 2201) builds since they have no arch guard.
ascend_moonep_add_kernel_library(
    ascend_moonep_kernels
    SOURCES
        kernels/barrier_bench.cpp
        kernels/moonep_combine.cpp
        kernels/moonep_dispatch.cpp
        kernels/moonep_grad_reduce.cpp
        ${ASCEND_MOONEP_PLANNING_KERNEL_SOURCE}
        kernels/prefetch.cpp
        ${ASCEND_MOONEP_UDMA_KERNEL_SOURCES}
    INCLUDES
        include
        kernels
        ../3rdparty/catlass/include
        ../3rdparty/catlass
    COMPILE_DEFINITIONS
        CATLASS_ARCH=${ASCEND_MOONEP_CATLASS_ARCH}
        ${ASCEND_MOONEP_ARCH_DEFINITIONS}
)

message(STATUS "ascend-moonep kernel library: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libascend_moonep_kernels.so")
message(STATUS "SHMEM installation: ${SHMEM_HOME_PATH}")
