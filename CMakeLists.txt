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
# Materialize the pinned ggml commit in the build tree and apply all RMBG
# changes there. This keeps the submodule immutable while making the patch a
# real configure input: editing it automatically rebuilds the patched tree.
# ---------------------------------------------------------------------------
set(GGML_PATCH "${RMBG_ROOT}/third_party/ggml-rmbg.patch")
if(NOT EXISTS "${GGML_PATCH}")
    message(FATAL_ERROR "Required ggml patch is missing: ${GGML_PATCH}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GGML_PATCH}")
find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${RMBG_ROOT}/third_party/ggml" rev-parse HEAD
    OUTPUT_VARIABLE GGML_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GGML_REV_PARSE_RESULT
)
if(NOT GGML_REV_PARSE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to read the pinned ggml submodule commit")
endif()
file(SHA256 "${GGML_PATCH}" GGML_PATCH_SHA256)
set(GGML_PATCH_PIPELINE_VERSION 5)
string(SUBSTRING "${GGML_COMMIT}" 0 12 GGML_COMMIT_SHORT)
string(SUBSTRING "${GGML_PATCH_SHA256}" 0 12 GGML_PATCH_SHORT)
set(GGML_PATCH_ID "${GGML_COMMIT_SHORT}-${GGML_PATCH_SHORT}")
set(GGML_PATCHED_SOURCE "${CMAKE_BINARY_DIR}/_deps/ggml-rmbg-src-${GGML_PATCH_ID}")
set(GGML_PATCHED_BINARY "${CMAKE_BINARY_DIR}/third_party/ggml-rmbg-${GGML_PATCH_ID}")
set(GGML_PATCH_STAMP "${GGML_PATCHED_SOURCE}/.rmbg-patch-stamp")
set(GGML_EXPECTED_STAMP
    "${GGML_PATCH_PIPELINE_VERSION}\n${GGML_COMMIT}\n${GGML_PATCH_SHA256}\n")
set(GGML_CURRENT_STAMP "")
if(EXISTS "${GGML_PATCH_STAMP}")
    file(READ "${GGML_PATCH_STAMP}" GGML_CURRENT_STAMP)
endif()
if(NOT GGML_CURRENT_STAMP STREQUAL GGML_EXPECTED_STAMP)
    file(REMOVE_RECURSE "${GGML_PATCHED_SOURCE}")
    file(MAKE_DIRECTORY "${GGML_PATCHED_SOURCE}")
    set(GGML_ARCHIVE "${CMAKE_BINARY_DIR}/_deps/ggml-${GGML_COMMIT}.tar")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${RMBG_ROOT}/third_party/ggml"
                archive --format=tar --output "${GGML_ARCHIVE}" HEAD
        RESULT_VARIABLE GGML_ARCHIVE_RESULT
    )
    if(GGML_ARCHIVE_RESULT EQUAL 0)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E tar xf "${GGML_ARCHIVE}"
            WORKING_DIRECTORY "${GGML_PATCHED_SOURCE}"
            RESULT_VARIABLE GGML_EXTRACT_RESULT
        )
    endif()
    file(REMOVE "${GGML_ARCHIVE}")
    if(NOT GGML_ARCHIVE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to materialize ggml commit ${GGML_COMMIT}")
    endif()
    if(NOT GGML_EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ggml commit ${GGML_COMMIT}")
    endif()
    execute_process(
        # The build tree is inside the parent repository. Prevent git from
        # treating this directory as a path prefix and silently skipping every
        # patch entry that lies outside that prefix.
        COMMAND "${CMAKE_COMMAND}" -E env
                "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
                "${GIT_EXECUTABLE}" apply --check "${GGML_PATCH}"
        WORKING_DIRECTORY "${GGML_PATCHED_SOURCE}"
        RESULT_VARIABLE GGML_PATCH_CHECK_RESULT
    )
    if(NOT GGML_PATCH_CHECK_RESULT EQUAL 0)
        message(FATAL_ERROR
            "ggml patch does not apply to pinned commit ${GGML_COMMIT}: ${GGML_PATCH}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
                "${GIT_EXECUTABLE}" apply "${GGML_PATCH}"
        WORKING_DIRECTORY "${GGML_PATCHED_SOURCE}"
        RESULT_VARIABLE GGML_PATCH_APPLY_RESULT
    )
    if(NOT GGML_PATCH_APPLY_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to apply ggml patch: ${GGML_PATCH}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
                "${GIT_EXECUTABLE}" apply --check --reverse "${GGML_PATCH}"
        WORKING_DIRECTORY "${GGML_PATCHED_SOURCE}"
        RESULT_VARIABLE GGML_PATCH_VERIFY_RESULT
    )
    if(NOT GGML_PATCH_VERIFY_RESULT EQUAL 0 OR
       NOT EXISTS "${GGML_PATCHED_SOURCE}/src/ggml-cuda/rmbg-deform-im2col.cu")
        message(FATAL_ERROR "ggml patch post-apply verification failed: ${GGML_PATCH}")
    endif()
    file(WRITE "${GGML_PATCH_STAMP}" "${GGML_EXPECTED_STAMP}")
    message(STATUS "Materialized ggml ${GGML_COMMIT} with patch ${GGML_PATCH_SHA256}")
endif()
set(GGML_BUILD_COMMIT "${GGML_COMMIT_SHORT}+rmbg.${GGML_PATCH_SHORT}")
add_subdirectory("${GGML_PATCHED_SOURCE}" "${GGML_PATCHED_BINARY}" EXCLUDE_FROM_ALL)

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
target_include_directories(rmbg PUBLIC include PRIVATE "${GGML_PATCHED_SOURCE}/examples")
target_link_libraries(rmbg PUBLIC ggml)

if(RMBG_BUILD_CLI)
    add_executable(rmbg-cli examples/rmbg-cli.cpp)
    target_link_libraries(rmbg-cli PRIVATE rmbg)
endif()

if(RMBG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
