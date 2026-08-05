#include "swin_backbone.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static std::string env_or(const char * k, const char * def) {
    const char * v = std::getenv(k);
    return v && v[0] ? v : def;
}

int main() {
    const std::string root = env_or("RMBG_ROOT", "..");
    const std::string weights = env_or("RMBG_WEIGHTS", (root + "/models/development/swin_stage01_f16.gguf").c_str());
    const std::string ref_gguf = env_or("RMBG_PARITY_REF", (root + "/tests/fixtures/swin_stage01_ref.gguf").c_str());

    std::vector<float> input, ref_pe, ref_s0, ref_merge, ref_s1;
    std::vector<int64_t> sh;
    if (!rmbg_parity::load_f32(ref_gguf, "input_nchw", input, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "patch_embed", ref_pe, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "stage0_out", ref_s0, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "merge_out", ref_merge, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "stage1_out", ref_s1, sh)) return 2;

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
    if (!fwd.has_stage1()) {
        std::fprintf(stderr, "init: stage1 weights missing\n");
        return 2;
    }

    std::vector<float> got_pe, got_s0, got_merge, got_s1;
    if (!fwd.forward_stage01(input, 1024, 1024, got_pe, got_s0, got_merge, got_s1, err)) {
        std::fprintf(stderr, "forward_stage01: %s\n", err.c_str());
        return 1;
    }

    bool ok_pe = rmbg_parity::compare(got_pe, ref_pe, "patch_embed", 1e-3f, 1e-3f);
    bool ok_s0 = rmbg_parity::compare(got_s0, ref_s0, "stage0_out", 1e-3f, 1e-3f);
    bool ok_m = rmbg_parity::compare(got_merge, ref_merge, "merge_out", 1e-3f, 1e-3f);
    bool ok_s1 = rmbg_parity::compare(got_s1, ref_s1, "stage1_out", 1e-3f, 1e-3f);

    return (ok_pe && ok_s0 && ok_m && ok_s1) ? 0 : 1;
}
