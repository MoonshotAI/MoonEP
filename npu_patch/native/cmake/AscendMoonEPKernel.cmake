include(CMakeParseArguments)

set(ASCEND_MOONEP_XASC_TIMESTAMP 20260327)
set(ASCEND_MOONEP_BISHENG_TIMESTAMP 0)
set(ASCEND_MOONEP_USE_XASC FALSE)
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -v
    OUTPUT_VARIABLE ASCEND_MOONEP_BISHENG_STDOUT
    ERROR_VARIABLE ASCEND_MOONEP_BISHENG_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
set(ASCEND_MOONEP_BISHENG_VERSION_OUTPUT
    "${ASCEND_MOONEP_BISHENG_STDOUT} ${ASCEND_MOONEP_BISHENG_STDERR}")
string(REGEX MATCH
    "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])"
    ASCEND_MOONEP_BISHENG_DATE
    "${ASCEND_MOONEP_BISHENG_VERSION_OUTPUT}")
if(ASCEND_MOONEP_BISHENG_DATE)
    string(REGEX REPLACE
        "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])"
        "\\1\\2\\3"
        ASCEND_MOONEP_BISHENG_TIMESTAMP
        "${ASCEND_MOONEP_BISHENG_DATE}")
    if(ASCEND_MOONEP_BISHENG_TIMESTAMP GREATER_EQUAL ASCEND_MOONEP_XASC_TIMESTAMP)
        set(ASCEND_MOONEP_USE_XASC TRUE)
    endif()
else()
    message(WARNING
        "Cannot extract a build date from '${CMAKE_CXX_COMPILER} -v'; "
        "falling back to legacy -xcce mode")
endif()
if(ASCEND_MOONEP_USE_XASC)
    set(ASCEND_MOONEP_DRIVER_MODE -xasc)
else()
    set(ASCEND_MOONEP_DRIVER_MODE -xcce)
endif()
message(STATUS
    "Bisheng build date=${ASCEND_MOONEP_BISHENG_TIMESTAMP}; "
    "AscendC driver=${ASCEND_MOONEP_DRIVER_MODE}")

function(ascend_moonep_add_kernel_library TARGET)
    set(OPTIONS)
    set(ONE_VALUE_ARGS)
    set(MULTI_VALUE_ARGS SOURCES INCLUDES COMPILE_DEFINITIONS)
    cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "${TARGET}: SOURCES must not be empty")
    endif()

    set(FFTS_COMPILE_FLAG)
    set(FFTS_COMPILE_DEFINITION)
    if(ASCEND_MOONEP_BISHENG_ARCH STREQUAL "a5")
        set(CCE_AICORE_ARCH_BASE dav-c310)
        set(ASCEND_NPU_ARCH dav-3510)
    elseif(ASCEND_MOONEP_BISHENG_ARCH STREQUAL "a2")
        set(CCE_AICORE_ARCH_BASE dav-c220)
        set(ASCEND_NPU_ARCH dav-2201)
        set(FFTS_COMPILE_DEFINITION FFTS_CONFIG=1)
        if(ASCEND_MOONEP_USE_XASC)
            set(FFTS_COMPILE_FLAG --cce-auto-set-ffts=false)
        endif()
    else()
        message(FATAL_ERROR "Unsupported ASCEND_MOONEP_BISHENG_ARCH=${ASCEND_MOONEP_BISHENG_ARCH}")
    endif()

    if(ASCEND_MOONEP_USE_XASC)
        set(ASCENDC_DRIVER_COMPILE_OPTIONS
            -xasc
            "--npu-arch=${ASCEND_NPU_ARCH}"
            --cce-auto-infer-kernel-type=false
            ${FFTS_COMPILE_FLAG}
        )
        set(ASCENDC_DRIVER_LINK_OPTIONS)
    else()
        set(ASCENDC_DRIVER_COMPILE_OPTIONS
            -xcce
            "--cce-aicore-arch=${CCE_AICORE_ARCH_BASE}"
            "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
            "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
            "SHELL:-mllvm -cce-aicore-record-overflow=true"
            "SHELL:-mllvm -cce-aicore-addr-transform"
            "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
        )
        set(ASCENDC_DRIVER_LINK_OPTIONS
            "--cce-aicore-arch=${CCE_AICORE_ARCH_BASE}"
        )
    endif()

    set(KERNEL_SOURCES)
    foreach(SOURCE IN LISTS ARG_SOURCES)
        if(IS_ABSOLUTE "${SOURCE}")
            list(APPEND KERNEL_SOURCES "${SOURCE}")
        else()
            get_filename_component(ABS_SOURCE "${SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
            list(APPEND KERNEL_SOURCES "${ABS_SOURCE}")
        endif()
    endforeach()

    set(KERNEL_INCLUDE_DIRS)
    foreach(INCLUDE_DIR IN LISTS ARG_INCLUDES)
        if(NOT IS_ABSOLUTE "${INCLUDE_DIR}")
            get_filename_component(INCLUDE_DIR "${INCLUDE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        endif()
        list(APPEND KERNEL_INCLUDE_DIRS "${INCLUDE_DIR}")
    endforeach()

    set(COMPILE_DEF_FLAGS)
    foreach(DEF IN LISTS ARG_COMPILE_DEFINITIONS)
        list(APPEND COMPILE_DEF_FLAGS "-D${DEF}")
    endforeach()

    get_filename_component(ASCEND_HOME_REAL_PATH "${ASCEND_HOME_PATH}" REALPATH)
    get_filename_component(ASCEND_BISHENG_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(ASCEND_BISHENG_COMPILER_DIR "${ASCEND_BISHENG_BIN_DIR}" DIRECTORY)
    get_filename_component(ASCEND_BISHENG_TOOLS_DIR "${ASCEND_BISHENG_COMPILER_DIR}" DIRECTORY)
    get_filename_component(ASCEND_BISHENG_ROOT_DIR "${ASCEND_BISHENG_TOOLS_DIR}" DIRECTORY)
    set(ASCEND_CANN_ROOT_CANDIDATES
        "${ASCEND_HOME_PATH}"
        "${ASCEND_HOME_REAL_PATH}"
        "${ASCEND_BISHENG_ROOT_DIR}"
    )
    set(ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES)
    if(DEFINED ASCEND_MOONEP_RUNTIME_INCLUDE_DIR)
        list(APPEND ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES "${ASCEND_MOONEP_RUNTIME_INCLUDE_DIR}")
    elseif(DEFINED ENV{ASCEND_MOONEP_RUNTIME_INCLUDE_DIR})
        list(APPEND ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES "$ENV{ASCEND_MOONEP_RUNTIME_INCLUDE_DIR}")
    endif()
    if(DEFINED ENV{ASCEND_DRIVER_PATH})
        list(APPEND ASCEND_CANN_ROOT_CANDIDATES "$ENV{ASCEND_DRIVER_PATH}")
    endif()
    if(EXISTS "/usr/local/Ascend")
        file(GLOB ASCEND_INSTALLED_CANN_ROOTS
            LIST_DIRECTORIES TRUE
            "/usr/local/Ascend/cann-*"
        )
        list(APPEND ASCEND_CANN_ROOT_CANDIDATES ${ASCEND_INSTALLED_CANN_ROOTS})
        if(EXISTS "/usr/local/Ascend/driver")
            list(APPEND ASCEND_CANN_ROOT_CANDIDATES "/usr/local/Ascend/driver")
        endif()
        file(GLOB_RECURSE ASCEND_INSTALLED_RT_HEADERS
            LIST_DIRECTORIES FALSE
            "/usr/local/Ascend/*/rt.h"
        )
        foreach(ASCEND_INSTALLED_RT_HEADER IN LISTS ASCEND_INSTALLED_RT_HEADERS)
            get_filename_component(ASCEND_INSTALLED_RT_INCLUDE_DIR "${ASCEND_INSTALLED_RT_HEADER}" DIRECTORY)
            list(APPEND ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES "${ASCEND_INSTALLED_RT_INCLUDE_DIR}")
        endforeach()
        if(UNIX)
            execute_process(
                COMMAND find /usr/local/Ascend -name rt.h -type f
                OUTPUT_VARIABLE ASCEND_FIND_RT_HEADERS
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(ASCEND_FIND_RT_HEADERS)
                string(REPLACE "\n" ";" ASCEND_FIND_RT_HEADERS "${ASCEND_FIND_RT_HEADERS}")
                foreach(ASCEND_FIND_RT_HEADER IN LISTS ASCEND_FIND_RT_HEADERS)
                    get_filename_component(ASCEND_FIND_RT_INCLUDE_DIR "${ASCEND_FIND_RT_HEADER}" DIRECTORY)
                    list(APPEND ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES "${ASCEND_FIND_RT_INCLUDE_DIR}")
                endforeach()
            endif()
        endif()
    endif()
    list(REMOVE_DUPLICATES ASCEND_CANN_ROOT_CANDIDATES)

    set(ASCEND_CANN_INCLUDE_CANDIDATES)
    set(ASCEND_CANN_LINK_CANDIDATES)
    foreach(ASCEND_CANN_ROOT IN LISTS ASCEND_CANN_ROOT_CANDIDATES)
        list(APPEND ASCEND_CANN_INCLUDE_CANDIDATES
            "${ASCEND_CANN_ROOT}/include"
            "${ASCEND_CANN_ROOT}/include/aclnn"
            "${ASCEND_CANN_ROOT}/include/experiment/runtime"
            "${ASCEND_CANN_ROOT}/include/experiment/msprof"
            "${ASCEND_CANN_ROOT}/pkg_inc"
            "${ASCEND_CANN_ROOT}/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/runtime/include"
            "${ASCEND_CANN_ROOT}/include/runtime"
            "${ASCEND_CANN_ROOT}/include/acl/runtime"
            "${ASCEND_CANN_ROOT}/aarch64-linux/include"
            "${ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc"
            "${ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/aarch64-linux/runtime/include"
            "${ASCEND_CANN_ROOT}/aarch64-linux/include/runtime"
            "${ASCEND_CANN_ROOT}/x86_64-linux/include"
            "${ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc"
            "${ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/x86_64-linux/runtime/include"
            "${ASCEND_CANN_ROOT}/x86_64-linux/include/runtime"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/runtime/include"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/runtime"
            "${ASCEND_CANN_ROOT}/compiler/tikcpp"
            "${ASCEND_CANN_ROOT}/compiler/tikcpp/tikcfw"
            "${ASCEND_CANN_ROOT}/compiler/tikcpp/tikcfw/impl"
            "${ASCEND_CANN_ROOT}/compiler/tikcpp/tikcfw/interface"
            "${ASCEND_CANN_ROOT}/aarch64-linux/tikcpp"
            "${ASCEND_CANN_ROOT}/aarch64-linux/tikcpp/tikcfw"
            "${ASCEND_CANN_ROOT}/aarch64-linux/tikcpp/tikcfw/impl"
            "${ASCEND_CANN_ROOT}/aarch64-linux/tikcpp/tikcfw/interface"
            "${ASCEND_CANN_ROOT}/x86_64-linux/tikcpp"
            "${ASCEND_CANN_ROOT}/x86_64-linux/tikcpp/tikcfw"
            "${ASCEND_CANN_ROOT}/x86_64-linux/tikcpp/tikcfw/impl"
            "${ASCEND_CANN_ROOT}/x86_64-linux/tikcpp/tikcfw/interface"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/impl"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/interface"
        )
        list(APPEND ASCEND_CANN_LINK_CANDIDATES
            "${ASCEND_CANN_ROOT}/lib64"
            "${ASCEND_CANN_ROOT}/aarch64-linux/lib64"
            "${ASCEND_CANN_ROOT}/x86_64-linux/lib64"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
        )
    endforeach()
    list(APPEND ASCEND_CANN_INCLUDE_CANDIDATES ${ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES})
    list(REMOVE_DUPLICATES ASCEND_CANN_INCLUDE_CANDIDATES)
    list(REMOVE_DUPLICATES ASCEND_CANN_LINK_CANDIDATES)

    find_path(ASCEND_ACL_INCLUDE_DIR
        NAMES acl/acl.h
        HINTS ${ASCEND_CANN_INCLUDE_CANDIDATES}
        NO_DEFAULT_PATH
    )
    if(NOT ASCEND_ACL_INCLUDE_DIR)
        message(FATAL_ERROR
            "Cannot find acl/acl.h under ASCEND_HOME_PATH=${ASCEND_HOME_PATH}. "
            "The include path must be the directory that contains the acl/ subdirectory, "
            "for example <cann>/include, not <cann>/include/acl.")
    endif()
    message(STATUS "ACL include root: ${ASCEND_ACL_INCLUDE_DIR}")

    find_path(ASCEND_RUNTIME_INCLUDE_DIR
        NAMES rt.h
        HINTS ${ASCEND_CANN_INCLUDE_CANDIDATES}
        NO_DEFAULT_PATH
    )
    if(NOT ASCEND_RUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR
            "Cannot find rt.h under ASCEND_HOME_PATH=${ASCEND_HOME_PATH}. "
            "CANN 8.x Bisheng runtime headers include rt.h directly, so the "
            "directory that contains rt.h must be on the compiler include path. "
            "Find it with `find /usr/local/Ascend -name rt.h` and pass "
            "`-DASCEND_MOONEP_RUNTIME_INCLUDE_DIR=<directory-containing-rt.h>` "
            "if it is installed in a non-standard location.")
    endif()
    message(STATUS "Runtime include root: ${ASCEND_RUNTIME_INCLUDE_DIR}")

    add_library(${TARGET} SHARED ${KERNEL_SOURCES})
    set_target_properties(${TARGET}
        PROPERTIES
            UNITY_BUILD ON
            UNITY_BUILD_BATCH_SIZE 16 # split into multiple .so if the operator count grows later
    )
    target_compile_options(${TARGET}
        PRIVATE
            -O2
            ${ASCENDC_DRIVER_COMPILE_OPTIONS}
            -Xhost-start -ftrapv -Xhost-end
            ${COMPILE_DEF_FLAGS}
    )

    target_compile_definitions(${TARGET}
        PRIVATE
            L2_CACHE_HINT
            ${FFTS_COMPILE_DEFINITION}
    )
    target_include_directories(${TARGET}
        PRIVATE
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/kernels"
            "${ASCEND_ACL_INCLUDE_DIR}"
            "${ASCEND_RUNTIME_INCLUDE_DIR}"
            "${CMAKE_SOURCE_DIR}/../3rdparty/catlass/include"
            "${CMAKE_SOURCE_DIR}/../3rdparty/catlass/examples/common"
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
            "${SHMEM_HOME_PATH}/shmem/src/device"
            "${SHMEM_HOME_PATH}/shmem/src/device_simt"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device"
            ${ASCEND_CANN_INCLUDE_CANDIDATES}
            ${KERNEL_INCLUDE_DIRS}
    )

    target_link_directories(${TARGET}
        PRIVATE
            ${ASCEND_CANN_LINK_CANDIDATES}
            "${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "${SHMEM_HOME_PATH}/shmem/lib"
    )
    target_link_options(${TARGET}
        PRIVATE
            --cce-fatobj-link
            ${ASCENDC_DRIVER_LINK_OPTIONS}
            "SHELL:-Wl,-rpath,\\$ORIGIN"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/lib64"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/shmem/lib"
    )
    target_link_libraries(${TARGET}
        PRIVATE
            -Wl,--no-as-needed
            runtime
            ascendcl
            shmem
            graph_base
            tiling_api
            platform
            register
            dl
            -Wl,--as-needed
            stdc++
            m
    )

    set(${TARGET}_LIBRARY "$<TARGET_FILE:${TARGET}>" PARENT_SCOPE)
endfunction()
