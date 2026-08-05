#include "birefnet_decoder.hpp"
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
        "RMBG_WEIGHTS", (root + "/models/development/encoder_f16.gguf").c_str());
    const std::string ref_gguf = env_or(
        "RMBG_PARITY_REF", (root + "/tests/fixtures/encoder_ref.gguf").c_str());

    std::vector<float> input, ref1, ref2, ref3, ref4;
    std::vector<int64_t> sh;
    if (!rmbg_parity::load_f32(ref_gguf, "input_nchw", input, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "x1", ref1, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "x2", ref2, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "x3", ref3, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "x4", ref4, sh)) return 2;

    rmbg::WeightMap wm;
    std::string err;
    if (!wm.load_gguf(weights.c_str(), err)) {
        std::fprintf(stderr, "weights: %s\n", err.c_str());
        return 2;
    }

    rmbg::SwinBackboneForward bb;
    if (!bb.init(wm, err)) {
        std::fprintf(stderr, "bb init: %s\n", err.c_str());
        return 2;
    }

    rmbg::Encoder4ScaleOutput got;
    std::fprintf(stderr, "running forward_encoder_4scale (1024x1024) ...\n");
    if (!rmbg::forward_encoder_4scale(input, 1024, 1024, bb, got, err)) {
        std::fprintf(stderr, "forward_encoder_4scale: %s\n", err.c_str());
        return 1;
    }

    bool ok = true;
    ok = rmbg_parity::compare(got.x1, ref1, "x1", 5e-2f, 1e-2f) && ok;
    ok = rmbg_parity::compare(got.x2, ref2, "x2", 5e-2f, 1e-2f) && ok;
    ok = rmbg_parity::compare(got.x3, ref3, "x3", 5e-2f, 1e-2f) && ok;
    ok = rmbg_parity::compare(got.x4, ref4, "x4", 5e-2f, 1e-2f) && ok;
    return ok ? 0 : 1;
}
