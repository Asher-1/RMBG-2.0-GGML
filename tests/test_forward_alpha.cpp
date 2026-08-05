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
    const std::string enc_weights = env_or(
        "RMBG_ENC_WEIGHTS", (root + "/models/development/encoder_f16.gguf").c_str());
    const std::string dec_weights = env_or(
        "RMBG_DEC_WEIGHTS", (root + "/models/development/decoder_alpha_f16.gguf").c_str());
    const std::string ref_gguf = env_or(
        "RMBG_PARITY_REF", (root + "/tests/fixtures/alpha_ref.gguf").c_str());

    std::vector<float> input, ref_alpha;
    std::vector<int64_t> sh;
    if (!rmbg_parity::load_f32(ref_gguf, "input_nchw", input, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "alpha_logits", ref_alpha, sh)) return 2;

    rmbg::WeightMap wm;
    std::string err;
    if (!wm.load_gguf(enc_weights.c_str(), err)) {
        std::fprintf(stderr, "encoder weights: %s\n", err.c_str());
        return 2;
    }
    if (!wm.merge_gguf(dec_weights.c_str(), err)) {
        std::fprintf(stderr, "decoder weights: %s\n", err.c_str());
        return 2;
    }

    rmbg::SwinBackboneForward bb;
    if (!bb.init(wm, err)) {
        std::fprintf(stderr, "bb init: %s\n", err.c_str());
        return 2;
    }

    std::vector<float> got;
    std::fprintf(stderr, "running forward_alpha (1024x1024) ...\n");
    if (!rmbg::forward_alpha(input, 1024, 1024, bb, wm, got, err)) {
        std::fprintf(stderr, "forward_alpha: %s\n", err.c_str());
        return 1;
    }

    bool ok = rmbg_parity::compare(got, ref_alpha, "alpha_logits", 5e-2f, 1e-2f);
    return ok ? 0 : 1;
}
