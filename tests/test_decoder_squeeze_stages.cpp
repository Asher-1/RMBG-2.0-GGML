#include "nn_ops.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static std::string env_or(const char * k, const char * def) {
    const char * v = std::getenv(k);
    return v && v[0] ? v : def;
}

static bool run_conv_relu(const std::vector<float> & x4, int C, int H, int W,
                          const rmbg::WeightMap & wm, std::vector<float> & out,
                          std::string & err) {
    rmbg::Conv2dParams cin;
    rmbg::BatchNorm2dParams bn_in;
    if (!rmbg::load_conv2d(wm, "sq0_conv_in_", cin, err)) return false;
    if (!rmbg::load_batch_norm2d(wm, "sq0_bn_in_", bn_in, err)) return false;
    std::vector<float> x;
    rmbg::conv2d_nchw(x4, 1, C, H, W, cin, 1, 1, x);
    rmbg::batch_norm2d_nchw(x, 1, cin.oc, H, W, bn_in, out);
    rmbg::relu_inplace(out);
    return true;
}

int main() {
    const std::string root = env_or("RMBG_ROOT", "..");
    const std::string weights = env_or(
        "RMBG_WEIGHTS", (root + "/models/development/squeeze_f16.gguf").c_str());
    const std::string ref_gguf = env_or(
        "RMBG_PARITY_REF", (root + "/tests/fixtures/squeeze_stages.gguf").c_str());

    std::vector<float> x4_in, ref_conv, ref_att;
    std::vector<int64_t> sh;
    if (!rmbg_parity::load_f32(root + "/tests/fixtures/squeeze_ref.gguf", "x4_in", x4_in, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "conv_relu", ref_conv, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "dec_att_out", ref_att, sh)) return 2;

    rmbg::WeightMap wm;
    std::string err;
    if (!wm.load_gguf(weights.c_str(), err)) {
        std::fprintf(stderr, "weights: %s\n", err.c_str());
        return 2;
    }

    std::vector<float> conv_relu, att;
    if (!run_conv_relu(x4_in, 5760, 32, 32, wm, conv_relu, err)) {
        std::fprintf(stderr, "conv_relu: %s\n", err.c_str());
        return 1;
    }
    bool ok = rmbg_parity::compare(conv_relu, ref_conv, "conv_relu", 1e-3f, 1e-3f);

    struct Branch { const char * ref; const char * prefix; };
    const Branch branches[] = {
        {"aspp_d0", "sq0_dec_att_aspp_deforms_0_"},
        {"aspp_d1", "sq0_dec_att_aspp_deforms_1_"},
        {"aspp_d2", "sq0_dec_att_aspp_deforms_2_"},
    };
    for (const Branch & b : branches) {
        std::vector<float> ref_b, got_b;
        if (!rmbg_parity::load_f32(ref_gguf, b.ref, ref_b, sh)) continue;
        if (!rmbg::aspp_deform_branch_forward(conv_relu, 1, 64, 32, 32, wm, b.prefix, got_b, err)) {
            std::fprintf(stderr, "%s: %s\n", b.ref, err.c_str());
            return 1;
        }
        ok = rmbg_parity::compare(got_b, ref_b, b.ref, 5e-2f, 1e-2f) && ok;
    }

    {
        std::vector<float> ref_gap, gap;
        if (rmbg_parity::load_f32(ref_gguf, "gap_up", ref_gap, sh)) {
            const int C = 64, H = 32, W = 32, N = 1;
            std::vector<float> pooled((size_t) N * C, 0.f);
            for (int c = 0; c < C; ++c) {
                double acc = 0.0;
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        acc += conv_relu[(((size_t) c * H + h) * W + w)];
                pooled[c] = (float) (acc / (H * W));
            }
            rmbg::Conv2dParams gconv;
            rmbg::BatchNorm2dParams gbn;
            if (!rmbg::load_conv2d(wm, "sq0_dec_att_global_avg_pool_1_", gconv, err) ||
                !rmbg::load_batch_norm2d(wm, "sq0_dec_att_global_avg_pool_2_", gbn, err)) {
                std::fprintf(stderr, "gap weights: %s\n", err.c_str());
                return 1;
            }
            std::vector<float> g1;
            rmbg::conv2d_nchw(pooled, N, C, 1, 1, gconv, 1, 0, g1);
            rmbg::batch_norm2d_nchw(g1, N, gconv.oc, 1, 1, gbn, gap);
            rmbg::relu_inplace(gap);
            std::vector<float> gap_up;
            rmbg::bilinear_resize_nchw(gap, N, gconv.oc, 1, 1, H, W, gap_up);
            gap = std::move(gap_up);
            ok = rmbg_parity::compare(gap, ref_gap, "gap_up", 5e-2f, 1e-2f) && ok;
        }
    }

    if (!rmbg::aspp_deformable_forward(conv_relu, 1, 64, 32, 32, wm, "sq0_dec_att_", att, err)) {
        std::fprintf(stderr, "dec_att: %s\n", err.c_str());
        return 1;
    }
    ok = rmbg_parity::compare(att, ref_att, "dec_att_out", 5e-2f, 1e-2f) && ok;

    return ok ? 0 : 1;
}
