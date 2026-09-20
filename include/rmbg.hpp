#pragma once

#include <string>
#include <vector>
#include "ggml-backend.h"

namespace rmbg {

struct Config {
    int input_size = 1024;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3]  = {0.229f, 0.224f, 0.225f};
    std::string backbone = "swin_v1_l";
};

class RmbgDeviceGraph;

// Explicit backend/graph tuning options. These replace the former
// RMBG_VULKAN_MODE / RMBG_STRICT_MATH / RMBG_VK_DIRECT_CONV /
// RMBG_VK_SCALAR_DIRECT_CONV environment switches; defaults reproduce the
// previous "optimized" production profile. Remaining environment knobs are
// research-only deep switches (RMBG_VK_DEFORM_PROJECT, RMBG_VK_QKV_LAYOUT,
// RMBG_VK_FLASH_ATTN) and upstream ggml variables (GGML_VK_*, NVIDIA_TF32_OVERRIDE).
struct BackendOptions {
    // Vulkan math profile:
    //   Optimized  (default): parity-safe measured profile — fp16 device
    //              shaders disabled, coopmat kept for f16/quantized matmuls,
    //              exact fp32 f32-matmul routing, scalar direct convolutions.
    //   Strict:    pure fp32 everywhere (disables coopmat + fp16 device
    //              shaders; slowest, used for bit-stable debugging).
    //   UnsafeFast: everything on coopmat/tensor-core incl. f32 staging
    //              through fp16 — fast, NOT parity-safe.
    enum class VulkanMode { Optimized, Strict, UnsafeFast };
    VulkanMode vulkan_mode = VulkanMode::Optimized;   // was RMBG_VULKAN_MODE

    bool strict_math = false;        // was RMBG_STRICT_MATH: CUDA — disable TF32 (pure fp32 GEMMs)
    bool direct_conv = true;         // was RMBG_VK_DIRECT_CONV: build Vulkan convs via ggml_conv_2d_direct
    bool scalar_direct_conv = true;  // was RMBG_VK_SCALAR_DIRECT_CONV: force scalar conv2d pipelines
    bool f32_matmul_exact = true;    // ggml_backend_vk_set_f32_matmul_exact
    bool cuda_f16_gemm = false;      // was RMBG_CUDA_F16_GEMM: experimental f16 GEMM (Swin MLP, CUDA)
    int  cuda_f16_min_stage = 2;     // was RMBG_CUDA_F16_MIN_STAGE
    bool cuda_nn_gemm = false;       // was RMBG_CUDA_NN_GEMM: pre-transposed NN Swin GEMMs (CUDA)
};

struct Model {
    Model() = default;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    Config cfg;
    ggml_backend_t backend = nullptr;
    RmbgDeviceGraph * graph = nullptr;
    std::string backend_name;
    bool graph_ready = false;
};

// Low-level callers that create a ggml backend themselves must call this before
// ggml_backend_init*. The high-level load_gguf API already does so.
void configure_backend_profile(const char * device, const BackendOptions & options);
inline void configure_backend_profile(const char * device) {
    configure_backend_profile(device, BackendOptions{});
}

bool load_gguf(const char * path, const char * device, const BackendOptions & options,
               Model & out, std::string & err);
inline bool load_gguf(const char * path, const char * device, Model & out, std::string & err) {
    return load_gguf(path, device, BackendOptions{}, out, err);
}
inline bool load_gguf(const char * path, Model & out, std::string & err) {
    return load_gguf(path, "auto", out, err);
}
void free_model(Model & m);
bool remove_background(Model & m,
                       const void * image_bytes, int image_len,
                       std::vector<uint8_t> & out_png,
                       std::string & err);

} // namespace rmbg
