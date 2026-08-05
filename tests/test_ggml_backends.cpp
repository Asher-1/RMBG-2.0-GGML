// test_ggml_backends.cpp — 验证 ggml CUDA/Vulkan 后端可用性 + matmul benchmark
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

static bool test_backend(const char * backend_name, int N, int warmup, int runs) {
    fprintf(stderr, "\n=== 测试后端: %s (matmul %dx%d) ===\n", backend_name, N, N);

    ggml_backend_t backend = nullptr;
    if (std::strcmp(backend_name, "CPU") == 0) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else {
        backend = ggml_backend_init_by_name(backend_name, nullptr);
    }
    if (!backend) {
        fprintf(stderr, "  [X] 后端 %s 不可用\n", backend_name);
        return false;
    }

    const char * actual_name = ggml_backend_name(backend);
    fprintf(stderr, "  [OK] 后端已初始化: %s\n", actual_name);

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    fprintf(stderr, "  设备内存: free=%zu MB, total=%zu MB\n",
            free_mem / (1024*1024), total_mem / (1024*1024));

    // static graph: 创建 context (no_alloc=true)
    size_t ctx_size = (size_t)N * N * sizeof(float) * 6 + 1024 * 1024 * 8;
    ggml_init_params ip = { .mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "  ggml_init failed\n");
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    // 构建 graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    // 分配 tensors 到后端 (需要 no_alloc=true context)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "  alloc_ctx_tensors failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // 填充数据
    std::vector<float> ad(N * N), bd(N * N);
    for (int i = 0; i < N * N; i++) {
        ad[i] = (float)(i % 7) * 0.1f - 0.3f;
        bd[i] = (float)(i % 11) * 0.05f - 0.25f;
    }
    ggml_backend_tensor_set(a, ad.data(), 0, N * N * sizeof(float));
    ggml_backend_tensor_set(b, bd.data(), 0, N * N * sizeof(float));

    // 分配 compute buffer
    size_t compute_size = ggml_graph_size(gf) * 256 + 1024 * 1024 * 4;
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t compute_buf = ggml_backend_buft_alloc_buffer(buft, compute_size);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, gf);
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;

    fprintf(stderr, "  matmul %dx%d: %.2f ms/iter (%d runs)\n", N, N, ms, runs);

    // 清理
    ggml_backend_buffer_free(compute_buf);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return true;
}

int main() {
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "GGML 后端可用性测试\n");
    fprintf(stderr, "========================================\n");

    // 列出所有注册后端
    fprintf(stderr, "\n--- 注册后端 ---\n");
    size_t nreg = ggml_backend_reg_count();
    for (size_t i = 0; i < nreg; i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) continue;
        fprintf(stderr, "  [%zu] %s (devices: %zu)\n", i,
                ggml_backend_reg_name(reg),
                ggml_backend_reg_dev_count(reg));
        for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); j++) {
            ggml_backend_dev_t d = ggml_backend_reg_dev_get(reg, j);
            if (!d) continue;
            fprintf(stderr, "       [%zu] %s - %s\n", j,
                    ggml_backend_dev_name(d),
                    ggml_backend_dev_description(d));
        }
    }

    // 测试各后端
    const int N = 512;
    const int warmup = 3;
    const int runs = 10;

    test_backend("CPU", N, warmup, runs);
    test_backend("CUDA0", N, warmup, runs);
    test_backend("Vulkan0", N, warmup, runs);

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "测试完成\n");
    fprintf(stderr, "========================================\n");
    return 0;
}
