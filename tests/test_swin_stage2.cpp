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
    const std::string weights = env_or("RMBG_WEIGHTS", (root + "/models/development/swin_stage2_f16.gguf").c_str());
    const std::string ref_gguf = env_or("RMBG_PARITY_REF", (root + "/tests/fixtures/swin_stage2_ref.gguf").c_str());

    std::vector<float> input, ref_s2;
    std::vector<int64_t> sh;
    if (!rmbg_parity::load_f32(ref_gguf, "input_nchw", input, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "stage2_out", ref_s2, sh)) return 2;

    rmbg::WeightMap wm;
    std::string err;
    if (!wm.load_gguf(weights.c_str(), err)) {
        std::fprintf(stderr, "weights: %s\n", err.c_str());
        return 2;
    }

    rmbg::SwinBackboneForward fwd;
    if (!fwd.init(wm, err)) {
        std::fprintf(stderr, "init: %s (%d blocks, need 22)\n", err.c_str(), fwd.num_blocks());
        return 2;
    }
    if (!fwd.has_stage2()) {
        std::fprintf(stderr, "init: stage2 incomplete (%d blocks)\n", fwd.num_blocks());
        return 2;
    }

    std::vector<float> got_pe, got_s2;
    std::fprintf(stderr, "running stage2 forward (18 blocks @ 64x64)...\n");
    if (!fwd.forward_stage2(input, 1024, 1024, got_pe, got_s2, err)) {
        std::fprintf(stderr, "forward_stage2: %s\n", err.c_str());
        return 1;
    }

    // f16 weights + 18 blocks @ dim768: max drift ~0.05 vs f32 PyTorch ref; mean ~2e-5.
    bool ok = rmbg_parity::compare(got_s2, ref_s2, "stage2_out", 5e-2f, 1e-3f);
    return ok ? 0 : 1;
}
