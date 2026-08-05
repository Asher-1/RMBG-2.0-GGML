#include "swin_graph.hpp"
#include "swin_backbone.hpp"
#include "parity.hpp"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <string>

static std::string env_or(const char * k, const char * def) {
    const char * v = std::getenv(k);
    return v && v[0] ? v : def;
}

int main() {
    const std::string root = env_or("RMBG_ROOT", "..");
    const std::string weights = env_or("RMBG_WEIGHTS", (root + "/models/development/swin_block0_f16.gguf").c_str());
    const std::string ref_gguf = env_or("RMBG_PARITY_REF", (root + "/tests/fixtures/swin_block0_ref.gguf").c_str());

    std::vector<float> input, ref_pe;
    std::vector<int64_t> sh;
    if (!rmbg_parity::load_f32(ref_gguf, "input_nchw", input, sh)) return 2;
    if (!rmbg_parity::load_f32(ref_gguf, "patch_embed", ref_pe, sh)) return 2;

    rmbg::WeightMap wm;
    std::string err;
    if (!wm.load_gguf(weights.c_str(), err)) {
        std::fprintf(stderr, "weights: %s\n", err.c_str());
        return 2;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "no ggml backend\n");
        return 2;
    }

    rmbg::SwinPatchEmbedGraph graph;
    if (!graph.init(backend, wm, err)) {
        std::fprintf(stderr, "graph init: %s\n", err.c_str());
        ggml_backend_free(backend);
        return 2;
    }

    std::vector<float> got_pe, tok;
    if (!graph.forward(input, 1024, 1024, got_pe, tok, err)) {
        std::fprintf(stderr, "graph forward: %s\n", err.c_str());
        graph.free();
        ggml_backend_free(backend);
        return 1;
    }

    rmbg::SwinBackboneForward cpu;
    if (!cpu.init(wm, err)) {
        std::fprintf(stderr, "cpu init: %s\n", err.c_str());
        graph.free();
        ggml_backend_free(backend);
        return 2;
    }
    std::vector<float> cpu_pe, cpu_tok;
    if (!cpu.forward_patch_embed(input, 1024, 1024, cpu_pe, cpu_tok, err)) {
        std::fprintf(stderr, "cpu forward: %s\n", err.c_str());
        graph.free();
        ggml_backend_free(backend);
        return 1;
    }

    bool ok_g = rmbg_parity::compare(got_pe, ref_pe, "graph_patch_embed", 1e-3f, 1e-3f);
    bool ok_c = rmbg_parity::compare(cpu_pe, ref_pe, "cpu_patch_embed", 1e-3f, 1e-3f);
    rmbg_parity::compare(got_pe, cpu_pe, "graph_vs_cpu", 1e-4f, 1e-4f);

    graph.free();
    ggml_backend_free(backend);
    return (ok_g && ok_c) ? 0 : 1;
}
