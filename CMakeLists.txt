cmake_minimum_required(VERSION 3.18)
project(rmbg VERSION 0.1.0 LANGUAGES C CXX)

set(RMBG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RMBG_BUILD_CLI    "Build rmbg-cli"           ON)
option(RMBG_BUILD_TESTS    "Build tests"              ON)
option(RMBG_SHARED         "Build shared library"     OFF)
option(RMBG_GGML_CUDA      "ggml CUDA backend"        OFF)
option(RMBG_GGML_VULKAN    "ggml Vulkan backend"      OFF)
option(RMBG_GGML_METAL     "ggml Metal backend"       OFF)

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_CUDA   ${RMBG_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_VULKAN ${RMBG_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_METAL  ${RMBG_GGML_METAL}  CACHE BOOL "" FORCE)

if(NOT EXISTS "${RMBG_ROOT}/third_party/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml submodule missing. Run:\n"
        "  git submodule update --init --recursive third_party/ggml")
endif()

# ---------------------------------------------------------------------------
# Apply RMBG-specific ggml patches automatically.
# The ggml submodule is kept pristine (tracked only by its pinned commit);
# all RMBG custom ops (deform-im2col, swin qkv/windows, ...) live in
# third_party/ggml-rmbg.patch and are applied here so that any machine
# building this repo gets identical sources.
# ---------------------------------------------------------------------------
set(GGML_PATCH "${RMBG_ROOT}/third_party/ggml-rmbg.patch")
if(EXISTS "${GGML_PATCH}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env git apply --check "${GGML_PATCH}"
        WORKING_DIRECTORY "${RMBG_ROOT}/third_party/ggml"
        RESULT_VARIABLE GGML_PATCH_FWD
        ERROR_QUIET
    )
    if(GGML_PATCH_FWD EQUAL 0)
        # Not yet applied -> apply it.
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env git apply "${GGML_PATCH}"
            WORKING_DIRECTORY "${RMBG_ROOT}/third_party/ggml"
            RESULT_VARIABLE GGML_PATCH_APPLY
        )
        if(NOT GGML_PATCH_APPLY EQUAL 0)
            message(FATAL_ERROR "Failed to apply ggml patch: ${GGML_PATCH}")
        endif()
        message(STATUS "Applied ggml patch: ${GGML_PATCH}")
    else()
        # Forward apply failed -> check whether it is already applied.
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env git apply --check --reverse "${GGML_PATCH}"
            WORKING_DIRECTORY "${RMBG_ROOT}/third_party/ggml"
            RESULT_VARIABLE GGML_PATCH_REV
            ERROR_QUIET
        )
        if(GGML_PATCH_REV EQUAL 0)
            message(STATUS "ggml patch already applied: ${GGML_PATCH}")
        else()
            message(FATAL_ERROR
                "ggml source tree is neither clean nor patched. "
                "Reset it to the pinned commit first:\n"
                "  cd third_party/ggml && git checkout -- . && git clean -fd")
        endif()
    endif()
endif()
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

set(RMBG_SOURCES
    src/backend.cpp
    src/model_loader.cpp
    src/preprocess.cpp
    src/engine.cpp
    src/rmbg_capi.cpp
    src/swin_backbone.cpp
    src/swin_graph.cpp
    src/rmbg_graph.cpp
    src/nn_ops.cpp
    src/birefnet_decoder.cpp
    src/birefnet_encoder.cpp
)
if(RMBG_SHARED)
    add_library(rmbg SHARED ${RMBG_SOURCES})
else()
    add_library(rmbg STATIC ${RMBG_SOURCES})
endif()
target_include_directories(rmbg PUBLIC include PRIVATE third_party/ggml/examples)
target_link_libraries(rmbg PUBLIC ggml)

if(RMBG_BUILD_CLI)
    add_executable(rmbg-cli examples/rmbg-cli.cpp)
    target_link_libraries(rmbg-cli PRIVATE rmbg)
endif()

if(RMBG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
