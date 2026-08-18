add_executable(test_swin_block0 test_swin_block0.cpp)
target_include_directories(test_swin_block0 PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_swin_block0 PRIVATE rmbg ggml)
add_test(NAME swin_block0 COMMAND test_swin_block0)

add_executable(test_swin_stage0 test_swin_stage0.cpp)
target_include_directories(test_swin_stage0 PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_swin_stage0 PRIVATE rmbg ggml)
add_test(NAME swin_stage0 COMMAND test_swin_stage0)

add_executable(test_swin_stage01 test_swin_stage01.cpp)
target_include_directories(test_swin_stage01 PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_swin_stage01 PRIVATE rmbg ggml)
add_test(NAME swin_stage01 COMMAND test_swin_stage01)

add_executable(test_swin_stage2 test_swin_stage2.cpp)
target_include_directories(test_swin_stage2 PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_swin_stage2 PRIVATE rmbg ggml)
add_test(NAME swin_stage2 COMMAND test_swin_stage2)

add_executable(test_swin_stage3 test_swin_stage3.cpp)
target_include_directories(test_swin_stage3 PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_swin_stage3 PRIVATE rmbg ggml)
add_test(NAME swin_stage3 COMMAND test_swin_stage3)

add_executable(test_decoder_squeeze test_decoder_squeeze.cpp)
target_include_directories(test_decoder_squeeze PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_decoder_squeeze PRIVATE rmbg ggml)
add_test(NAME decoder_squeeze COMMAND test_decoder_squeeze)

add_executable(test_decoder_squeeze_stages test_decoder_squeeze_stages.cpp)
target_include_directories(test_decoder_squeeze_stages PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_decoder_squeeze_stages PRIVATE rmbg ggml)
add_test(NAME decoder_squeeze_stages COMMAND test_decoder_squeeze_stages)

add_executable(test_encoder_4scale test_encoder_4scale.cpp)
target_include_directories(test_encoder_4scale PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_encoder_4scale PRIVATE rmbg ggml)
add_test(NAME encoder_4scale COMMAND test_encoder_4scale)

add_executable(test_forward_alpha test_forward_alpha.cpp)
target_include_directories(test_forward_alpha PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_forward_alpha PRIVATE rmbg ggml)
add_test(NAME forward_alpha COMMAND test_forward_alpha)

add_executable(test_swin_graph_pe test_swin_graph_pe.cpp)
target_include_directories(test_swin_graph_pe PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_swin_graph_pe PRIVATE rmbg ggml)
add_test(NAME swin_graph_pe COMMAND test_swin_graph_pe)

add_executable(test_swin_graph_full test_swin_graph_full.cpp)
target_include_directories(test_swin_graph_full PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_swin_graph_full PRIVATE rmbg ggml)
add_test(NAME swin_graph_full COMMAND test_swin_graph_full)

add_executable(test_rmbg_graph test_rmbg_graph.cpp)
target_include_directories(test_rmbg_graph PRIVATE ${RMBG_ROOT}/include ${RMBG_ROOT}/tests)
target_link_libraries(test_rmbg_graph PRIVATE rmbg ggml)
add_test(NAME rmbg_graph COMMAND test_rmbg_graph)

# Unit tests run from the build tree. Make their fixture and development-model
# paths independent of that working directory.
set_tests_properties(
    swin_block0 swin_stage0 swin_stage01 swin_stage2 swin_stage3
    decoder_squeeze decoder_squeeze_stages encoder_4scale forward_alpha swin_graph_pe
    PROPERTIES ENVIRONMENT "RMBG_ROOT=${RMBG_ROOT}")
if(RMBG_GGML_CUDA)
    set_tests_properties(swin_graph_full rmbg_graph PROPERTIES
        ENVIRONMENT "RMBG_ROOT=${RMBG_ROOT};RMBG_DEVICE=gpu;NVIDIA_TF32_OVERRIDE=0"
        TIMEOUT 180)
elseif(RMBG_GGML_VULKAN)
    # Intermediate Swin fixtures use tight layer-local thresholds, so retain
    # the scalar diagnostic profile there. The end-to-end alpha fixture is the
    # acceptance gate for the production optimized profile.
    set_tests_properties(swin_graph_full PROPERTIES
        ENVIRONMENT "RMBG_ROOT=${RMBG_ROOT};RMBG_DEVICE=gpu;RMBG_VULKAN_MODE=strict;GGML_VK_DISABLE_F16=1;GGML_VK_DISABLE_COOPMAT=1;GGML_VK_DISABLE_COOPMAT2=1;GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1"
        TIMEOUT 180)
    set_tests_properties(rmbg_graph PROPERTIES
        ENVIRONMENT "RMBG_ROOT=${RMBG_ROOT};RMBG_DEVICE=gpu"
        TIMEOUT 180)
else()
    set_tests_properties(swin_graph_full rmbg_graph PROPERTIES
        ENVIRONMENT "RMBG_ROOT=${RMBG_ROOT};RMBG_DEVICE=cpu"
        TIMEOUT 300)
endif()

add_executable(test_ggml_backends test_ggml_backends.cpp)
target_include_directories(test_ggml_backends PRIVATE ${RMBG_ROOT}/include)
target_link_libraries(test_ggml_backends PRIVATE ggml)
# Not in CTest — informational only.
