#include "parity.hpp"
#include "rmbg.hpp"
#include "rmbg_graph.hpp"
#include "nn_ops.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::string env_or(const char * key, const char * fallback) {
    const char * value = std::getenv(key);
    return value && value[0] ? value : fallback;
}

static std::vector<float> channel_slice(const std::vector<float> & src,
                                        int offset, int channels, int H, int W) {
    const size_t begin = (size_t) offset * H * W;
    const size_t count = (size_t) channels * H * W;
    return std::vector<float>(src.begin() + begin, src.begin() + begin + count);
}

static std::vector<float> nchw_to_tokens(const std::vector<float> & src, int C, int H, int W) {
    std::vector<float> out((size_t) C * H * W);
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int c = 0; c < C; ++c)
                out[((size_t) h * W + w) * C + c] = src[((size_t) c * H + h) * W + w];
    return out;
}

int main() {
    const std::string root = env_or("RMBG_ROOT", "..");
    const std::string weights_path = env_or(
        "RMBG_WEIGHTS", (root + "/models/development/encoder_f16.gguf").c_str());
    const std::string ref_path = env_or(
        "RMBG_PARITY_REF", (root + "/tests/fixtures/encoder_ref.gguf").c_str());
    const std::string device = env_or("RMBG_DEVICE", "gpu");

    std::vector<float> input, ref1, ref2, ref3, ref4;
    std::vector<float> ref_patch;
    std::vector<float> ref_s0, ref_s1, ref_s2, ref_s3;
    std::vector<float> ref_b0;
    std::vector<float> ref_norm1, ref_xw0, ref_aw0, ref_after;
    std::vector<int64_t> shape;
    if (!rmbg_parity::load_f32(ref_path, "input_nchw", input, shape) ||
        !rmbg_parity::load_f32(ref_path, "x1", ref1, shape) ||
        !rmbg_parity::load_f32(ref_path, "x2", ref2, shape) ||
        !rmbg_parity::load_f32(ref_path, "x3", ref3, shape) ||
        !rmbg_parity::load_f32(ref_path, "x4", ref4, shape)) {
        std::fprintf(stderr, "failed to load encoder reference\n");
        return 2;
    }
    if (!rmbg_parity::load_f32(root + "/tests/fixtures/swin_stage01_ref.gguf", "stage0_out", ref_s0, shape) ||
        !rmbg_parity::load_f32(root + "/tests/fixtures/swin_stage01_ref.gguf", "stage1_out", ref_s1, shape) ||
        !rmbg_parity::load_f32(root + "/tests/fixtures/swin_stage2_ref.gguf", "stage2_out", ref_s2, shape) ||
        !rmbg_parity::load_f32(root + "/tests/fixtures/swin_stage3_ref.gguf", "stage3_out", ref_s3, shape)) {
        std::fprintf(stderr, "failed to load Swin stage references\n");
        return 2;
    }
    if (!rmbg_parity::load_f32(root + "/tests/fixtures/swin_stage0_ref.gguf", "block0_out", ref_b0, shape)) return 2;
    if (!rmbg_parity::load_f32(root + "/tests/fixtures/swin_block0_ref.gguf", "patch_embed", ref_patch, shape)) return 2;
    const std::string block0_ref = root + "/tests/fixtures/swin_block0_ref.gguf";
    if (!rmbg_parity::load_f32(block0_ref, "block0_norm1", ref_norm1, shape) ||
        !rmbg_parity::load_f32(block0_ref, "block0_xw0", ref_xw0, shape) ||
        !rmbg_parity::load_f32(block0_ref, "block0_aw0", ref_aw0, shape) ||
        !rmbg_parity::load_f32(block0_ref, "block0_after_attn", ref_after, shape)) return 2;

    rmbg::WeightMap weights;
    std::string err;
    if (!weights.load_gguf(weights_path.c_str(), err)) {
        std::fprintf(stderr, "weights: %s\n", err.c_str());
        return 2;
    }

    const enum ggml_backend_dev_type type = device == "cpu"
        ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU;
    rmbg::configure_backend_profile(device.c_str());
    ggml_backend_t backend = ggml_backend_init_by_type(type, nullptr);
    if (!backend) {
        std::fprintf(stderr, "requested backend is unavailable\n");
        return 2;
    }

    bool ok = true;
    {
    rmbg::RmbgDeviceGraph graph;
    if (!graph.init(backend, weights, 1024, err)) {
        std::fprintf(stderr, "graph init: %s\n", err.c_str());
        return 2;
    }
    std::fprintf(stderr, "compute buffer: %.1f MiB\n", graph.compute_bytes() / 1048576.0);

    std::vector<float> x1, x2, x3, x4;
    if (!graph.forward_encoder(input, x1, x2, x3, x4, err)) {
        std::fprintf(stderr, "graph forward: %s\n", err.c_str());
        return 1;
    }

    if (!std::getenv("RMBG_QUICK")) {
        std::vector<float> pre_norm, norm1, xw0, aw0, after;
        if (!graph.forward_block0_debug(input, pre_norm, norm1, xw0, aw0, after, err)) return 1;
        rmbg::Conv2dParams patch_conv;
        std::vector<float> scalar_pre_norm;
        if (!rmbg::load_conv2d(weights, "bb_patch_embed_proj_", patch_conv, err)) return 2;
        rmbg::conv2d_nchw(input, 1, 3, 1024, 1024, patch_conv, 4, 0, scalar_pre_norm);
        ok = rmbg_parity::compare(pre_norm, nchw_to_tokens(scalar_pre_norm, 192, 256, 256),
                                  "graph_patch_pre_norm", 1e-4f, 1e-4f) && ok;
        ok = rmbg_parity::compare(norm1, ref_norm1, "graph_block0_norm1", 1e-4f, 1e-4f) && ok;
        ok = rmbg_parity::compare(xw0, ref_xw0, "graph_block0_xw0", 1e-4f, 1e-4f) && ok;
        ok = rmbg_parity::compare(aw0, ref_aw0, "graph_block0_aw0", 2e-3f, 1e-3f) && ok;
        ok = rmbg_parity::compare(after, ref_after, "graph_block0_after", 2e-3f, 1e-3f) && ok;

        std::vector<float> patch, b0, s0, s1, s2, s3;
        if (!graph.forward_swin_debug(input, patch, b0, s0, s1, s2, s3, err)) {
        std::fprintf(stderr, "debug forward: %s\n", err.c_str());
        return 1;
        }
        ok = rmbg_parity::compare(patch, nchw_to_tokens(ref_patch, 192, 256, 256),
                                  "graph_patch_tokens", 1e-4f, 1e-4f) && ok;
        ok = rmbg_parity::compare(b0, ref_b0, "graph_block0_raw", 5e-3f, 2e-3f) && ok;
        ok = rmbg_parity::compare(s0, ref_s0, "graph_stage0_raw", 5e-3f, 2e-3f) && ok;
        ok = rmbg_parity::compare(s1, ref_s1, "graph_stage1_raw", 5e-3f, 2e-3f) && ok;
        // Stage 2 compounds rounding through 18 blocks. Keep the relative gate
        // unchanged while allowing the measured near-zero CPU accumulation error.
        ok = rmbg_parity::compare(s2, ref_s2, "graph_stage2_raw", 1.2e-2f, 3e-3f) && ok;
        ok = rmbg_parity::compare(s3, ref_s3, "graph_stage3_raw", 1e-2f, 3e-3f) && ok;
    }
    ok = rmbg_parity::compare(x1, channel_slice(ref1, 0, 192, 256, 256),
                              "graph_x1", 5e-3f, 2e-3f) && ok;
    ok = rmbg_parity::compare(x2, channel_slice(ref2, 0, 384, 128, 128),
                              "graph_x2", 5e-3f, 2e-3f) && ok;
    ok = rmbg_parity::compare(x3, channel_slice(ref3, 0, 768, 64, 64),
                              "graph_x3", 1e-2f, 3e-3f) && ok;
    ok = rmbg_parity::compare(x4, channel_slice(ref4, 2688, 1536, 32, 32),
                              "graph_x4", 1e-2f, 3e-3f) && ok;

    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
