cmake_minimum_required(VERSION 3.18)

# ─── ROCm/HIP toggle ──────────────────────────────────────────────────────────
# Set -DDFLASH27B_USE_HIP=ON to build for AMD ROCm (Strix Halo / gfx1151).
# Default: OFF (original NVIDIA CUDA path).
option(DFLASH27B_USE_HIP "Build for AMD ROCm/HIP instead of NVIDIA CUDA" OFF)

if(NOT DFLASH27B_USE_HIP)
    # CUDA path: capture user arch override before project() resets CMAKE_CUDA_ARCHITECTURES
    set(DFLASH27B_USER_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    project(dflash27b LANGUAGES C CXX CUDA)
else()
    project(dflash27b LANGUAGES C CXX)
    # HIP language support: either via hipcc CXX or explicit enable.
    # The ggml-hip subdirectory calls enable_language(HIP) itself; we call it
    # here first so the parent project can set LANGUAGE HIP on our own sources.
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES gfx1151)
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
    find_package(HIP REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_INSTALL_RPATH "$ORIGIN/deps/llama.cpp/ggml/src;$ORIGIN/deps/llama.cpp/ggml/src/ggml-cuda;$ORIGIN/../deps/llama.cpp/ggml/src;$ORIGIN/../deps/llama.cpp/ggml/src/ggml-cuda")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ld.lld (ROCm's LLVM linker) requires -fPIC for all translation units on x86_64.
# GNU ld is more permissive. Set globally for HIP builds to avoid relocation errors.
if(DFLASH27B_USE_HIP)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# ─── ggml (from the llama.cpp submodule) ─────────────────────────────────────

if(DFLASH27B_USE_HIP)
    set(GGML_HIP          ON  CACHE BOOL "" FORCE)
    set(GGML_CUDA         OFF CACHE BOOL "" FORCE)
else()
    set(GGML_CUDA         ON  CACHE BOOL "" FORCE)
endif()
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(GGML_METAL        OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN       OFF CACHE BOOL "" FORCE)
set(GGML_BLAS         OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL       OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "deps/llama.cpp submodule missing. Run: "
        "git submodule update --init --recursive")
endif()

option(DFLASH27B_FA_ALL_QUANTS "Compile ggml-cuda fattn kernels for all KV-quant pairs" ON)
set(GGML_CUDA_FA_ALL_QUANTS ${DFLASH27B_FA_ALL_QUANTS} CACHE BOOL "" FORCE)

# ─── Architecture / feature resolution ────────────────────────────────────────

if(DFLASH27B_USE_HIP)
    # HIP: arch list is gfx strings, not SM integers.
    set(_dflash27b_archs "${CMAKE_HIP_ARCHITECTURES}")

    # gfx1151 (RDNA3.5) supports BF16 + WMMA-equivalent ops, analogous to sm_80+.
    # This gate controls whether flashprefill_kernels (WMMA path) is compiled.
    # Phase 1: leave OFF (uses ggml q8 fallback). Phase 2: set ON after rocWMMA port.
    option(DFLASH27B_HIP_SM80_EQUIV
        "Enable flashprefill WMMA kernel path for HIP (requires rocWMMA port — Phase 2)"
        OFF)

    # Blackwell consumer workaround is NVIDIA-specific, skip.
    set(_dflash_is_consumer_blackwell OFF)

else()
    # Original CUDA arch resolution ────────────────────────────────────────────
    if(DFLASH27B_USER_CUDA_ARCHITECTURES)
        set(_dflash27b_archs "${DFLASH27B_USER_CUDA_ARCHITECTURES}")
    else()
        set(_dflash27b_archs "75;86")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND _dflash27b_archs "120")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
            list(APPEND _dflash27b_archs "110")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
            list(APPEND _dflash27b_archs "121")
        endif()
    endif()

    # Consumer Blackwell workaround
    option(DFLASH27B_USE_BLACKWELL_CONSUMER_FIX
        "Enable ggml consumer-Blackwell workaround" OFF)
    if(DFLASH27B_USE_BLACKWELL_CONSUMER_FIX)
        set(_dflash_is_consumer_blackwell ON)
    endif()
    if(NOT DEFINED _dflash_is_consumer_blackwell)
        set(_dflash_is_consumer_blackwell OFF)
        foreach(_arch IN LISTS _dflash27b_archs)
            string(REGEX REPLACE "[^0-9]" "" _dflash_arch_num "${_arch}")
            if(_dflash_arch_num MATCHES "^12[0-9]$")
                set(_dflash_is_consumer_blackwell ON)
                break()
            endif()
        endforeach()
    endif()
endif()

if(_dflash_is_consumer_blackwell)
    set(GGML_CUDA_BLACKWELL_CONSUMER ON CACHE BOOL
        "Skip sm_12X→sm_12Xa for consumer Blackwell (no FP4)" FORCE)
endif()

add_subdirectory(deps/llama.cpp/ggml EXCLUDE_FROM_ALL)

# Inject hip_compat/ into ggml-hip's include path so that CUDA-header references
# in the luce-dflash custom ops (gated_delta_net.cu, fattn-chunked.cu) resolve.
if(DFLASH27B_USE_HIP AND TARGET ggml-hip)
    target_include_directories(ggml-hip PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
endif()

if(NOT DFLASH27B_USE_HIP)
    find_package(CUDAToolkit REQUIRED)
endif()

# ─── dflash27b static library ──────────────────────────────────────────────

add_library(dflash27b STATIC
    src/errors.cpp
    src/gguf_target_loader.cpp
    src/gguf_draft_loader.cpp
    src/safetensors_draft.cpp
    src/qwen35_target_graph.cpp
    src/qwen3_dflash_graph.cpp
    src/qwen3_drafter.cpp
    src/qwen3_0p6b_loader.cpp
    src/qwen3_0p6b_graph.cpp
    src/flashprefill_q8.cpp
    src/kv_cache.cpp
    src/kv_quant.cpp
    src/delta_net_chunked.cpp
)
# ld.lld (ROCm's LLVM linker) requires -fPIC on x86_64 for static libs linked into HIP executables
if(DFLASH27B_USE_HIP)
    set_target_properties(dflash27b PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(dflash27b PRIVATE DFLASH27B_USE_HIP=1)
endif()

if(DFLASH27B_USE_HIP)
    # HIP f16 conversion kernel (manually hipified from f16_convert.cu)
    target_sources(dflash27b PRIVATE src/f16_convert.hip.cu)
    set_source_files_properties(src/f16_convert.hip.cu PROPERTIES LANGUAGE HIP)
else()
    target_sources(dflash27b PRIVATE src/f16_convert.cu)
    set_target_properties(dflash27b PROPERTIES CUDA_ARCHITECTURES "${_dflash27b_archs}")
endif()

# ─── FlashPrefill custom kernels ───────────────────────────────────────────

if(DFLASH27B_USE_HIP)
    if(DFLASH27B_HIP_SM80_EQUIV)
        # Phase 2: rocWMMA port of flashprefill_kernels (requires manual port of kernel 4).
        # flashprefill_kernels.cu must be hipified and fragment layouts updated for gfx1151.
        # See docs/GOTCHAS.md "WMMA Fragment Register Layout" section.
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/flashprefill_kernels.hip.cu")
            target_sources(dflash27b PRIVATE
                src/flashprefill_kernels.hip.cu
                src/flashprefill_select.cpp
                src/flashprefill.cpp)
            set_source_files_properties(src/flashprefill_kernels.hip.cu PROPERTIES LANGUAGE HIP)
            target_compile_definitions(dflash27b PRIVATE DFLASH27B_HAVE_FLASHPREFILL=1 DFLASH27B_MIN_SM=80)
            # rocWMMA is header-only; locate it under the ROCm prefix
            find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp
                HINTS /opt/rocm/include ${HIP_INCLUDE_DIRS}
                NO_DEFAULT_PATH)
            if(ROCWMMA_INCLUDE_DIR)
                target_include_directories(dflash27b PRIVATE ${ROCWMMA_INCLUDE_DIR})
            else()
                message(WARNING
                    "rocwmma/rocwmma.hpp not found under /opt/rocm/include — "
                    "flashprefill HIP kernels may fail to compile. "
                    "Install rocwmma: sudo apt install rocwmma (or dnf install rocwmma)")
            endif()
            message(STATUS "dflash: HIP Phase 2 — custom flashprefill kernels enabled (rocWMMA)")
        else()
            message(WARNING
                "DFLASH27B_HIP_SM80_EQUIV=ON but src/flashprefill_kernels.hip.cu not found. "
                "Falling back to ggml q8 path. See docs/PORTING_PLAN.md Phase 2.")
        endif()
    else()
        # Phase 1: use ggml flash_attn_ext (q8 path) for drafter scoring.
        # No custom WMMA kernels needed. TTFT improvement is still significant.
        message(STATUS "dflash: HIP Phase 1 — using ggml q8 fallback for flashprefill (no rocWMMA)")
    endif()
else()
    # CUDA path: FlashPrefill WMMA kernels need sm_80+ (BF16 WMMA).
    list(GET _dflash27b_archs 0 _dflash27b_min_sm)
    string(REGEX REPLACE "[^0-9]" "" _dflash27b_min_sm "${_dflash27b_min_sm}")
    target_compile_definitions(dflash27b PRIVATE DFLASH27B_MIN_SM=${_dflash27b_min_sm})
    if(_dflash27b_min_sm GREATER_EQUAL 80)
        target_sources(dflash27b PRIVATE
            src/flashprefill_kernels.cu
            src/flashprefill_select.cpp
            src/flashprefill.cpp)
        target_compile_definitions(dflash27b PRIVATE DFLASH27B_HAVE_FLASHPREFILL=1)
    endif()
endif()

# ─── BSA (Block-Sparse Attention) ──────────────────────────────────────────

if(NOT DEFINED DFLASH27B_ENABLE_BSA)
    set(DFLASH27B_ENABLE_BSA ON)
endif()

if(DFLASH27B_USE_HIP)
    # HIP Phase 3: BSA backed by our rocWMMA kernel (no CUTLASS required).
    # bsa_launcher_hip.cu provides the same extern "C" symbol as bsa_launcher.cu
    # by forwarding to launch_sparse_flash_forward_bf16.
    if(DFLASH27B_ENABLE_BSA AND DFLASH27B_HIP_SM80_EQUIV)
        target_sources(dflash27b PRIVATE src/bsa_launcher_hip.cu)
        set_source_files_properties(src/bsa_launcher_hip.cu PROPERTIES LANGUAGE HIP)
        message(STATUS "dflash: HIP Phase 3 — BSA launcher enabled (rocWMMA wrapper)")
    elseif(DFLASH27B_ENABLE_BSA AND NOT DFLASH27B_HIP_SM80_EQUIV)
        message(STATUS "BSA on HIP requires DFLASH27B_HIP_SM80_EQUIV (rocWMMA); disabling.")
        set(DFLASH27B_ENABLE_BSA OFF)
    endif()
else()
    # CUDA-only BSA guards (SM check + submodule check)
    if(DFLASH27B_ENABLE_BSA)
        foreach(_arch IN LISTS _dflash27b_archs)
            if(_arch LESS 80)
                message(WARNING "DFLASH27B_ENABLE_BSA=ON but arch ${_arch} < 80; disabling.")
                set(DFLASH27B_ENABLE_BSA OFF)
                break()
            endif()
        endforeach()
    endif()
    if(DFLASH27B_ENABLE_BSA AND NOT EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include/cutlass/numeric_types.h")
        message(WARNING "BSA submodule missing — disabling. Run git submodule update --init --recursive.")
        set(DFLASH27B_ENABLE_BSA OFF)
    endif()
    if(DFLASH27B_ENABLE_BSA)
        target_sources(dflash27b PRIVATE src/bsa_fwd_inst.cu src/bsa_launcher.cu)
    endif()
endif()

# ─── Include directories ────────────────────────────────────────────────────

target_include_directories(dflash27b
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(DFLASH27B_USE_HIP)
    # hip_compat/ provides transparent CUDA → HIP header shims.
    # C++ sources that include <cuda_runtime.h> will pick up the HIP version.
    target_include_directories(dflash27b PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
else()
    target_include_directories(dflash27b PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(DFLASH27B_ENABLE_BSA)
    if(DFLASH27B_USE_HIP)
        # HIP BSA: only DFLASH27B_HAVE_BSA=1 needed; no CUTLASS headers.
        target_compile_definitions(dflash27b PRIVATE DFLASH27B_HAVE_BSA=1)
    else()
        target_include_directories(dflash27b PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/bsa_stubs
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include
            ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/block_sparse_attn/src)
        target_compile_options(dflash27b PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_compile_definitions(dflash27b PRIVATE
            FLASHATTENTION_DISABLE_DROPOUT FLASH_NAMESPACE=flash DFLASH27B_HAVE_BSA=1)
    endif()
endif()

# ─── Link libraries ─────────────────────────────────────────────────────────

if(DFLASH27B_USE_HIP)
    set(_dflash_ggml_gpu_backend ggml-hip)
else()
    set(_dflash_ggml_gpu_backend ggml-cuda)
endif()

target_link_libraries(dflash27b
    PUBLIC
        ggml
        ${_dflash_ggml_gpu_backend}
        ggml-base
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     target_compile_options(dflash27b PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-function>
     )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     target_compile_options(dflash27b PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
     )
endif()

# ─── PFlash phase-split harness ──────────────────────────────────────────────

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/pflash_daemon.cpp")
    add_executable(pflash_daemon test/pflash_daemon.cpp)
    target_include_directories(pflash_daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(DFLASH27B_USE_HIP)
        target_include_directories(pflash_daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
    endif()
    target_link_libraries(pflash_daemon PRIVATE dflash27b ggml ${_dflash_ggml_gpu_backend})
endif()

# ─── Tests ─────────────────────────────────────────────────────────────────

option(DFLASH27B_TESTS "Build numerics tests" ON)
if(DFLASH27B_TESTS)
    # Helper macro: create test exe with common HIP/CUDA compat setup
    macro(dflash_add_test _name _src)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
            add_executable(${_name} ${_src})
            target_include_directories(${_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
            if(DFLASH27B_USE_HIP)
                target_include_directories(${_name} PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
            endif()
            target_link_libraries(${_name} PRIVATE dflash27b ggml ${_dflash_ggml_gpu_backend})
        endif()
    endmacro()

    dflash_add_test(test_kv_quant       test/test_kv_quant.cpp)
    dflash_add_test(test_draft_vs_reference test/test_draft_vs_reference.cpp)
    dflash_add_test(smoke_load_draft    test/smoke_load_draft.cpp)
    dflash_add_test(spike_thin_copy     test/spike_thin_copy.cpp)
    dflash_add_test(smoke_draft_graph   test/smoke_draft_graph.cpp)
    dflash_add_test(smoke_qwen3_0p6b_forward test/smoke_qwen3_0p6b_forward.cpp)
    dflash_add_test(test_vs_oracle      test/test_vs_oracle.cpp)
    dflash_add_test(smoke_load_target   test/smoke_load_target.cpp)
    dflash_add_test(smoke_target_forward test/smoke_target_forward.cpp)
    dflash_add_test(test_generate       test/test_generate.cpp)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_flashprefill_kernels.cpp")
        add_executable(test_flashprefill_kernels test/test_flashprefill_kernels.cpp)
        target_include_directories(test_flashprefill_kernels PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
        if(DFLASH27B_USE_HIP)
            target_include_directories(test_flashprefill_kernels PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
        else()
            set_target_properties(test_flashprefill_kernels PROPERTIES
                CUDA_ARCHITECTURES "${_dflash27b_archs}")
        endif()
        target_link_libraries(test_flashprefill_kernels PRIVATE dflash27b)
        if(DFLASH27B_USE_HIP)
            # hip::host exports ROCm include dirs without injecting --offload-arch into g++ CXX compile
            target_link_libraries(test_flashprefill_kernels PRIVATE ggml ${_dflash_ggml_gpu_backend} hip::host)
        else()
            target_link_libraries(test_flashprefill_kernels PRIVATE CUDA::cudart)
        endif()
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_score.cpp")
        add_executable(bench_score test/bench_score.cpp)
        target_include_directories(bench_score PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        if(DFLASH27B_USE_HIP)
            target_include_directories(bench_score PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
            target_link_libraries(bench_score PRIVATE dflash27b ggml ${_dflash_ggml_gpu_backend} hip::host)
        else()
            target_link_libraries(bench_score PRIVATE dflash27b CUDA::cudart)
        endif()
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_dflash.cpp")
        add_executable(test_dflash test/test_dflash.cpp)
        target_include_directories(test_dflash PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        if(DFLASH27B_USE_HIP)
            target_include_directories(test_dflash PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
            target_link_libraries(test_dflash PRIVATE
                dflash27b ggml ${_dflash_ggml_gpu_backend} hip::host)
        else()
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(test_dflash PRIVATE dflash27b CUDA::cudart)
        endif()
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(test_dflash PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()
