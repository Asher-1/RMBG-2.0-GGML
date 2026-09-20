#include "rmbg.hpp"
#include "rmbg_graph.hpp"
#include "swin_backbone.hpp"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>

namespace rmbg {

static std::string lower(std::string value);

static void set_env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

static void clear_env(const char * key) {
#ifdef _WIN32
    _putenv_s(key, "");
#else
    unsetenv(key);
#endif
}

// Explicit routing hooks provided by the patched ggml-vulkan backend.
// Weak-declared so CPU-only/CUDA-only builds still link; a null symbol is a no-op.
#if defined(__GNUC__) && !defined(_WIN32)
extern "C" void ggml_backend_vk_set_f32_matmul_exact(bool exact) __attribute__((weak));
extern "C" void ggml_backend_vk_set_scalar_direct_conv(bool scalar) __attribute__((weak));
static void set_f32_matmul_exact(bool exact) {
    if (ggml_backend_vk_set_f32_matmul_exact) {
        ggml_backend_vk_set_f32_matmul_exact(exact);
    }
}
static void set_scalar_direct_conv(bool scalar) {
    if (ggml_backend_vk_set_scalar_direct_conv) {
        ggml_backend_vk_set_scalar_direct_conv(scalar);
    }
}
#else
static void set_f32_matmul_exact(bool) {}
static void set_scalar_direct_conv(bool) {}
#endif

static void configure_vulkan_math(const BackendOptions & options) {
    // The GGML_VK_* device-shader toggles below are read from the environment
    // inside ggml at Vulkan device creation (before any explicit API exists),
    // so they must stay env-based. Everything RMBG controls is explicit.
    if (options.vulkan_mode == BackendOptions::VulkanMode::Strict) {
        set_env("GGML_VK_DISABLE_F16", "1");
        set_env("GGML_VK_DISABLE_COOPMAT", "1");
        set_env("GGML_VK_DISABLE_COOPMAT2", "1");
        set_env("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1");
        set_scalar_direct_conv(false);
        set_f32_matmul_exact(true);
        return;
    }
    if (options.vulkan_mode == BackendOptions::VulkanMode::UnsafeFast) {
        clear_env("GGML_VK_DISABLE_F16");
        clear_env("GGML_VK_DISABLE_COOPMAT");
        clear_env("GGML_VK_DISABLE_COOPMAT2");
        clear_env("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT");
        set_scalar_direct_conv(false);
        // Opt-out: f32 matmuls may take the coopmat/tensor-core path.
        set_f32_matmul_exact(false);
        return;
    }

    // Optimized (default): F32 accumulation stays enabled for sensitive work —
    // exact fp32 f32-matmul routing + scalar direct convolutions, coopmat kept
    // for f16/quantized matmuls.
    set_env("GGML_VK_DISABLE_F16", "1");
    clear_env("GGML_VK_DISABLE_COOPMAT");
    set_env("GGML_VK_DISABLE_COOPMAT2", "1");
    set_env("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1");
    set_scalar_direct_conv(options.scalar_direct_conv);
    set_f32_matmul_exact(options.f32_matmul_exact);
}

void configure_backend_profile(const char * device, const BackendOptions & options) {
    const std::string requested = lower(device ? device : "auto");
    const bool generic_gpu = requested == "gpu";
    if (options.strict_math &&
        (requested == "auto" || generic_gpu || requested.rfind("cuda", 0) == 0)) {
        // Bit-stable FP32 GEMMs (TF32 off). The default keeps cuBLAS TF32
        // enabled; its measured alpha error remains below 1.4e-3.
        set_env("NVIDIA_TF32_OVERRIDE", "0");
    }
    if (requested == "auto" || generic_gpu || requested.rfind("vulkan", 0) == 0) {
        configure_vulkan_math(options);
    }
}

static std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return value;
}

static ggml_backend_t pick_backend(const char * device, const BackendOptions & options) {
    configure_backend_profile(device, options);
    if (!device || !device[0] || std::strcmp(device, "auto") == 0) {
        ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (b) return b;
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (std::strcmp(device, "cpu") == 0)
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    std::string needle = lower(device);
    needle.erase(std::remove(needle.begin(), needle.end(), ':'), needle.end());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string name = lower(ggml_backend_dev_name(dev));
        const std::string desc = lower(ggml_backend_dev_description(dev));
        if (name.find(needle) != std::string::npos || desc.find(needle) != std::string::npos)
            return ggml_backend_dev_init(dev, nullptr);
    }
    return nullptr;
}

bool load_gguf(const char * path, const char * device, const BackendOptions & options,
               Model & out, std::string & err) {
    if (!path || !path[0]) { err = "empty path"; return false; }
    free_model(out);

    WeightMap weights;
    if (!weights.load_gguf(path, err)) return false;
    if (!weights.get_f32("sq0_conv_in_weight")) {
        const std::filesystem::path decoder =
            std::filesystem::path(path).parent_path() / "decoder_alpha_f16.gguf";
        if (!std::filesystem::exists(decoder)) {
            err = "GGUF has no decoder weights; use a unified GGUF or place " +
                  decoder.string() + " beside the encoder";
            return false;
        }
        if (!weights.merge_gguf(decoder.string().c_str(), err)) return false;
    }

    // Metadata is optional for split compatibility files.
    ggml_context * meta = nullptr;
    struct gguf_init_params params = { .no_alloc = true, .ctx = &meta };
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx) {
        const int k_size = gguf_find_key(ctx, "rmbg.input_size");
        if (k_size >= 0) out.cfg.input_size = (int) gguf_get_val_u32(ctx, k_size);
        const int k_mean = gguf_find_key(ctx, "rmbg.img.mean");
        if (k_mean >= 0) {
            const float * mean = (const float *) gguf_get_arr_data(ctx, k_mean);
            for (int i = 0; i < 3; ++i) out.cfg.mean[i] = mean[i];
        }
        const int k_std = gguf_find_key(ctx, "rmbg.img.std");
        if (k_std >= 0) {
            const float * st = (const float *) gguf_get_arr_data(ctx, k_std);
            for (int i = 0; i < 3; ++i) out.cfg.std[i] = st[i];
        }
        gguf_free(ctx);
        ggml_free(meta);
    }

    out.backend = pick_backend(device, options);
    if (!out.backend) { err = std::string("requested ggml backend unavailable: ") +
                              (device ? device : "auto"); return false; }
    out.backend_name = ggml_backend_name(out.backend);
    std::unique_ptr<RmbgDeviceGraph> graph(new RmbgDeviceGraph);
    if (!graph->init(out.backend, weights, out.cfg.input_size, options, err)) {
        graph.reset();
        ggml_backend_free(out.backend);
        out.backend = nullptr;
        return false;
    }
    out.graph = graph.release();
    out.graph_ready = true;
    return true;
}

void free_model(Model & m) {
    delete m.graph;
    m.graph = nullptr;
    if (m.backend) { ggml_backend_free(m.backend); m.backend = nullptr; }
    m.backend_name.clear();
    m.graph_ready = false;
}

} // namespace rmbg
