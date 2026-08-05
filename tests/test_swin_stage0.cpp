#include "swin_backbone.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static std::string env_or(const char * k, const char * def) {
    const char * v = std::getenv(k);
    return v && v[0] ? v : def;
}

static std::vector<float> pe_to_blc(const std::vector<float> & pe,
                                    const std::vector<int64_t> & sh) {
    const int64_t C = sh[sh.size() - 3];
    const int64_t H = sh[sh.size() - 2];
    const int64_t W = sh[sh.size() - 1];
    std::vector<float> tok((size_t) H * W * C);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w)
                tok[((size_t) h * W + w) * C + c] = pe[((size_t) c * H + h) * W + w];
    return tok;
}

int main() {
    const std::string root = env_or("RMBG_ROOT", "..");
    const std::string weights = env_or("RMBG_WEIGHTS", (root + "/models/development/swin_stage0_f16.gguf").c_str());
    const std::string ref_gguf = env_or("RMBG_PARITY_REF", (root + "/tests/fixtures/swin_stage0_ref.gguf").c_str());

    std::vector<float> input, ref_pe, ref_b0, ref_s0;
    std::vector<int64_t> sh, pe_sh;
    if (!rmbg_parity::load_f32(ref_gguf, "input_nchw", input, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "patch_embed", ref_pe, pe_sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "block0_out", ref_b0, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "stage0_out", ref_s0, sh)) return 2;

    rmbg::WeightMap wm;
    std::string err;
    if (!wm.load_gguf(weights.c_str(), err)) {
        std::fprintf(stderr, "weights: %s\n", err.c_str());
        return 2;
    }

    rmbg::SwinBackboneForward fwd;
    if (!fwd.init(wm, err)) {
        std::fprintf(stderr, "init: %s\n", err.c_str());
        return 2;
    }

    std::vector<float> got_pe, got_s0;
    if (!fwd.forward_stage0(input, 1024, 1024, got_pe, got_s0, err)) {
        std::fprintf(stderr, "forward_stage0: %s\n", err.c_str());
        return 1;
    }

    bool ok_pe = rmbg_parity::compare(got_pe, ref_pe, "patch_embed", 1e-3f, 1e-3f);
    bool ok_s0 = rmbg_parity::compare(got_s0, ref_s0, "stage0_out", 1e-3f, 1e-3f);

    if (pe_sh.size() >= 3) {
        const int64_t H = pe_sh[pe_sh.size() - 2], W = pe_sh[pe_sh.size() - 1];
        auto tok = pe_to_blc(ref_pe, pe_sh);
        std::vector<float> got_b0;
        if (fwd.forward_block(tok, (int) H, (int) W, 0, got_b0, err))
            rmbg_parity::compare(got_b0, ref_b0, "block0_out", 1e-3f, 1e-3f);
        std::vector<float> got_b1;
        if (fwd.forward_block(got_b0, (int) H, (int) W, 1, got_b1, err))
            rmbg_parity::compare(got_b1, ref_s0, "block1_out", 1e-3f, 1e-3f);
    }

    return (ok_pe && ok_s0) ? 0 : 1;
}
