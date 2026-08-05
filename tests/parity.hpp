#pragma once
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace rmbg_parity {

inline bool load_f32(const std::string & path, const char * name,
                     std::vector<float> & out, std::vector<int64_t> & shape) {
    ggml_context * ctx = nullptr;
    gguf_init_params p{ false, &ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) return false;
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) { gguf_free(g); ggml_free(ctx); return false; }
    shape.clear();
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i) shape.push_back(t->ne[i]);
    size_t n = (size_t) ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), ggml_get_data_f32(t), n * sizeof(float));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

inline bool compare(const std::vector<float> & got, const std::vector<float> & ref,
                    const char * label, float atol, float rtol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu\n", label, got.size(), ref.size());
        return false;
    }
    if (got.empty()) return false;
    double maxabs = 0.0, sumabs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double) got[i] - (double) ref[i]);
        sumabs += d;
        if (d > maxabs) { maxabs = d; worst = i; }
    }
    bool ok = true;
    for (size_t i = 0; i < got.size() && ok; ++i) {
        double tol = (double) atol + (double) rtol * std::fabs((double) ref[i]);
        if (std::fabs((double) got[i] - (double) ref[i]) > tol) ok = false;
    }
    std::fprintf(stderr,
        "[%s] n=%zu max|d|=%.3e mean|d|=%.3e (worst@%zu got=%.5f ref=%.5f) -> %s\n",
        label, got.size(), maxabs, sumabs / got.size(), worst,
        got[worst], ref[worst], ok ? "OK" : "FAIL");
    return ok;
}

} // namespace rmbg_parity
