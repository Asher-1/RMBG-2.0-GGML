#include "nn_ops.hpp"
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
    const std::string weights = env_or(
        "RMBG_WEIGHTS", (root + "/models/development/squeeze_f16.gguf").c_str());
    const std::string ref_gguf = env_or(
        "RMBG_PARITY_REF", (root + "/tests/fixtures/squeeze_ref.gguf").c_str());

    std::vector<float> x4_in, ref_out;
    std::vector<int64_t> sh_in, sh_out;
    if (!rmbg_parity::load_f32(ref_gguf, "x4_in", x4_in, sh_in)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "squeeze_out", ref_out, sh_out)) return 2;
    if (sh_in.size() != 3 && sh_in.size() != 4) {
        std::fprintf(stderr, "bad x4_in shape rank %zu\n", sh_in.size());
        return 2;
    }
    const int N = (sh_in.size() == 4) ? (int) sh_in[0] : 1;
    const int C = (int) sh_in[sh_in.size() - 3];
    const int H = (int) sh_in[sh_in.size() - 2];
    const int W = (int) sh_in[sh_in.size() - 1];

    rmbg::WeightMap wm;
    std::string err;
    if (!wm.load_gguf(weights.c_str(), err)) {
        std::fprintf(stderr, "weights: %s\n", err.c_str());
        return 2;
    }

    std::vector<float> got;
    std::fprintf(stderr, "running squeeze_module (%d x %dx%d) ...\n", C, H, W);
    if (!rmbg::squeeze_module_forward(x4_in, N, C, H, W, wm, got, err)) {
        std::fprintf(stderr, "squeeze_module_forward: %s\n", err.c_str());
        return 1;
    }

    bool ok = rmbg_parity::compare(got, ref_out, "squeeze_out", 5e-2f, 1e-2f);
    return ok ? 0 : 1;
}
