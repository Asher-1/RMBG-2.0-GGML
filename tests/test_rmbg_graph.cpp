#include "parity.hpp"
#include "rmbg.hpp"
#include "rmbg_graph.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static std::string env_or(const char * key, const char * fallback) {
    const char * value = std::getenv(key);
    return value && value[0] ? value : fallback;
}

int main() {
    const std::string root = env_or("RMBG_ROOT", "..");
    const std::string encoder = env_or(
        "RMBG_ENC_WEIGHTS", (root + "/models/development/encoder_f16.gguf").c_str());
    const std::string decoder = env_or(
        "RMBG_DEC_WEIGHTS", (root + "/models/development/decoder_alpha_f16.gguf").c_str());
    const std::string unified = env_or("RMBG_WEIGHTS", "");
    const std::string reference = env_or(
        "RMBG_PARITY_REF", (root + "/tests/fixtures/alpha_ref.gguf").c_str());
    const std::string device = env_or("RMBG_DEVICE", "gpu");

    std::vector<float> input, ref;
    std::vector<int64_t> shape;
    if (!rmbg_parity::load_f32(reference, "input_nchw", input, shape) ||
        !rmbg_parity::load_f32(reference, "alpha_logits", ref, shape)) return 2;
    for (float & value : ref) value = 1.f / (1.f + std::exp(-value));

    rmbg::WeightMap weights;
    std::string err;
    if (!weights.load_gguf((unified.empty() ? encoder : unified).c_str(), err) ||
        (unified.empty() && !weights.merge_gguf(decoder.c_str(), err))) {
        std::fprintf(stderr, "weights: %s\n", err.c_str());
        return 2;
    }
    const enum ggml_backend_dev_type type = device == "cpu"
        ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU;
    rmbg::configure_backend_profile(device.c_str());
    ggml_backend_t backend = ggml_backend_init_by_type(type, nullptr);
    if (!backend) { std::fprintf(stderr, "backend unavailable\n"); return 2; }
    std::string backend_name = ggml_backend_name(backend);
    std::transform(backend_name.begin(), backend_name.end(), backend_name.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    if (device != "gpu" && backend_name.find(device) == std::string::npos) {
        std::fprintf(stderr, "requested backend '%s', initialized '%s'\n",
                     device.c_str(), ggml_backend_name(backend));
        ggml_backend_free(backend);
        return 2;
    }
    std::fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));
    if (device == "cpu") {
        const int fallback_threads = std::max(1u, std::thread::hardware_concurrency());
        const int threads = std::max(1, std::atoi(
            env_or("RMBG_CPU_THREADS", std::to_string(fallback_threads).c_str()).c_str()));
        ggml_backend_cpu_set_n_threads(backend, threads);
        std::fprintf(stderr, "cpu threads: %d\n", threads);
    }

    bool ok = false;
    {
        rmbg::RmbgDeviceGraph graph;
        if (!graph.init(backend, weights, 1024, err)) {
            std::fprintf(stderr, "graph init: %s\n", err.c_str());
            return 2;
        }
        std::fprintf(stderr, "compute buffer: %.1f MiB\n", graph.compute_bytes() / 1048576.0);
        std::vector<float> got;
        if (!graph.forward(input, got, err)) {
            std::fprintf(stderr, "graph forward: %s\n", err.c_str());
            return 1;
        }
        size_t over_2e3 = 0, over_1e2 = 0, over_5e2 = 0;
        double mse = 0.0;
        size_t worst = 0;
        float worst_diff = 0.f;
        for (size_t i = 0; i < got.size(); ++i) {
            const float d = std::fabs(got[i] - ref[i]);
            mse += (double) d * d;
            over_2e3 += d > 2e-3f;
            over_1e2 += d > 1e-2f;
            over_5e2 += d > 5e-2f;
            if (d > worst_diff) { worst_diff = d; worst = i; }
        }
        std::fprintf(stderr,
            "error distribution: rmse=%.3e >2e-3=%zu >1e-2=%zu >5e-2=%zu\n",
            std::sqrt(mse / got.size()), over_2e3, over_1e2, over_5e2);
        const int wy = (int) (worst / 1024), wx = (int) (worst % 1024);
        std::fprintf(stderr, "worst neighborhood centered at y=%d x=%d:\n", wy, wx);
        for (int y = std::max(0, wy - 2); y <= std::min(1023, wy + 2); ++y) {
            for (int x = std::max(0, wx - 2); x <= std::min(1023, wx + 2); ++x) {
                const size_t i = (size_t) y * 1024 + x;
                std::fprintf(stderr, " (%d,%d %.4f/%.4f)", y, x, got[i], ref[i]);
            }
            std::fputc('\n', stderr);
        }
        ok = rmbg_parity::compare(got, ref, "graph_alpha", 2e-3f, 2e-3f);
        const int iterations = std::max(0, std::atoi(env_or("RMBG_BENCH_ITERS", "0").c_str()));
        if (iterations > 0) {
            const int warmup = std::max(0, std::atoi(env_or("RMBG_BENCH_WARMUP", "2").c_str()));
            for (int i = 0; i < warmup; ++i) {
                if (!graph.forward(input, got, err)) {
                    std::fprintf(stderr, "benchmark warmup: %s\n", err.c_str());
                    return 1;
                }
            }
            double total_ms = 0.0, best_ms = 1e100;
            std::vector<double> samples;
            samples.reserve(iterations);
            for (int i = 0; i < iterations; ++i) {
                const auto begin = std::chrono::steady_clock::now();
                if (!graph.forward(input, got, err)) {
                    std::fprintf(stderr, "benchmark forward: %s\n", err.c_str());
                    return 1;
                }
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                total_ms += ms;
                best_ms = std::min(best_ms, ms);
                samples.push_back(ms);
            }
            std::fprintf(stderr,
                         "steady-state: warmup=%d iterations=%d mean=%.2f ms best=%.2f ms\n",
                         warmup, iterations, total_ms / iterations, best_ms);
            std::fprintf(stderr, "samples-ms:");
            for (const double sample : samples) std::fprintf(stderr, " %.6f", sample);
            std::fputc('\n', stderr);
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
