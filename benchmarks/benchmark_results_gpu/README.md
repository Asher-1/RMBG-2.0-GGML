# RMBG-2.0 GGML 多后端综合评测报告

**评测日期**: 2026-08-03  
**评测环境**: Linux 20.04, x86_64  
**GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)  
**Driver**: 550.144.03 (CUDA 12.4)  
**CUDA Toolkit**: 11.8  
**Vulkan SDK**: 1.4.350.0  
**CPU**: AMD Ryzen 9 5950X 16-Core  

---

## 1. 系统环境与后端可用性

### 1.1 硬件配置

| 组件 | 规格 |
|------|------|
| CPU | AMD Ryzen 9 5950X 16-Core |
| GPU | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| 内存 | 64 GB |
| Driver | NVIDIA 550.144.03 |
| CUDA | 12.4 (driver), 11.8 (toolkit) |
| Vulkan | 1.4.350 (SDK), 1.3.277 (driver ICD) |

### 1.2 GGML 后端可用性验证

通过 `test_ggml_backends` 程序验证所有 ggml 后端：

| 后端 | 状态 | 设备名称 | 设备内存 |
|------|------|----------|----------|
| CPU | ✅ 可用 | AMD Ryzen 9 5950X | 64203 MB |
| CUDA | ✅ 可用 | NVIDIA GeForce RTX 3060 | 12015 MB (free: 10599 MB) |
| Vulkan | ✅ 可用 | NVIDIA GeForce RTX 3060 (NVIDIA) | 12534 MB (free: 10875 MB) |

**结论**: 三个 ggml 后端全部可用。

---

## 2. GGML 基础算子 Benchmark (512×512 matmul)

使用 `ggml_mul_mat` 在 512×512 矩阵上测试各后端推理内核性能：

| 后端 | 推理时间 (ms/iter) | 相对 CPU 加速比 |
|------|---------------------|-----------------|
| **CPU** | 6.96 - 7.45 | 1.0x |
| **CUDA** | **0.05** | **~149x** |
| **Vulkan** | **0.09** | **~77x** |

### 分析

- **CUDA 后端** 在简单 matmul 上达到 0.05ms/iter，是 CPU 的 **149 倍**
- **Vulkan 后端** 达到 0.09ms/iter，是 CPU 的 **77 倍**
- CUDA 比 Vulkan 快约 1.8 倍（在此简单算子上）

---

## 3. 精度对比 (C++ vs PyTorch)

### 3.1 组件级 Parity 测试

所有测试使用 PyTorch 导出的参考数据 (GGUF 格式) 与 C++ 纯数值实现对比：

| 组件 | 元素数量 | Max Diff | Mean Diff | 状态 |
|------|----------|----------|-----------|------|
| patch_embed | 12,582,912 | 6.691e-06 | 1.973e-07 | ✅ PASS |
| swin_block0 | 12,582,912 | 2.885e-05 | 1.164e-06 | ✅ PASS |
| swin_stage0 | 12,582,912 | 5.686e-05 | 2.049e-06 | ✅ PASS |
| swin_stage01 | 6,291,456 | 1.371e-04 | 4.352e-06 | ✅ PASS |
| decoder_squeeze | 3,145,728 | 9.537e-06 | 6.155e-07 | ✅ PASS |
| encoder_4scale (x1) | 25,165,824 | 3.920e-04 | 1.987e-06 | ✅ PASS |
| encoder_4scale (x2) | 12,582,912 | 1.237e-04 | 2.059e-06 | ✅ PASS |
| encoder_4scale (x3) | 6,291,456 | 9.332e-04 | 1.395e-06 | ✅ PASS |
| encoder_4scale (x4) | 5,898,240 | 2.232e-04 | 1.273e-06 | ✅ PASS |
| **forward_alpha (端到端)** | **1,048,576** | **5.264e-04** | **5.259e-06** | ✅ PASS |

### 3.2 精度结论

- **所有组件测试通过**，最大差异 < 1e-3
- 端到端 forward_alpha 最大差异仅 5.264e-04
- C++ 实现与 PyTorch 参考实现 **数值精度一致**
- 差异来源：浮点运算顺序、GELU erf 实现、LayerNorm 数值稳定性

---

## 4. 推理速度对比

### 4.1 PyTorch CPU 推理

| 模型 | 输入尺寸 | 设备 | 推理时间 |
|------|----------|------|----------|
| BiRefNet (fp32) | 1024×1024 | CPU | 14,761 - 15,816 ms |

### 4.2 C++ 纯数值实现 (CPU)

| 测试 | 推理时间 | 说明 |
|------|----------|------|
| test_swin_block0 | ~20 s | patch_embed + 1 block |
| test_swin_stage0 | ~80 s | patch_embed + 2 blocks |
| test_decoder_squeeze | ~19 s | squeeze_module |
| test_encoder_4scale | ~660 s (11 min) | 完整 encoder |
| test_forward_alpha | ~720 s (12 min) | 完整 encoder + decoder |

### 4.3 GGML 基础算子性能

| 后端 | 512×512 matmul | 相对加速 |
|------|----------------|----------|
| CPU | 7.45 ms | 1.0x |
| CUDA | 0.05 ms | 149x |
| Vulkan | 0.09 ms | 77x |

### 4.4 PyTorch CUDA 推理

**当前状态**: PyTorch 2.11.0+cu130 需要更新的 NVIDIA 驱动（当前 550.144.03 不支持 CUDA 13.0）。
PyTorch CUDA 推理在此系统上不可用。

**注意**: ggml CUDA 后端使用 CUDA toolkit 11.8 (driver API)，与当前驱动兼容，正常工作。

---

## 5. 效果图对比

### 5.1 PyTorch CPU 效果图

PyTorch BiRefNet 模型成功去除背景：
- `../benchmark_results/comparison_t4.png`
- `../benchmark_results/comparison_collage5.png`

### 5.2 C++/GGML 效果图

**当前状态**: 由于 GGML 推理图尚未完成，无法生成端到端 C++/GGML 效果图。

根据精度测试结论，C++ 实现与 PyTorch 数值差异极小 (max_diff < 1e-3)，预期完成推理图后视觉效果与 PyTorch 基本一致。

---

## 6. 综合对比总结

### 6.1 各后端状态矩阵

| 维度 | CPU | CUDA | Vulkan | PyTorch CPU | PyTorch CUDA |
|------|-----|------|--------|-------------|--------------|
| **后端可用** | ✅ | ✅ | ✅ | ✅ | ❌ (驱动版本) |
| **精度验证** | ✅ | ⚠️ 需推理图 | ⚠️ 需推理图 | ✅ (参考) | - |
| **基础算子** | ✅ 7.45ms | ✅ 0.05ms | ✅ 0.09ms | - | - |
| **端到端推理** | ⚠️ 纯C++ 12min | ❌ 推理图WIP | ❌ 推理图WIP | ✅ 15s | ❌ |

### 6.2 关键发现

| 发现 | 详情 |
|------|------|
| 1. ggml CUDA/Vulkan 后端完全可用 | RTX 3060 上 CUDA 和 Vulkan 均正常初始化并运行 |
| 2. CUDA 基础算子比 CPU 快 149x | 512×512 matmul: 0.05ms vs 7.45ms |
| 3. Vulkan 基础算子比 CPU 快 77x | 512×512 matmul: 0.09ms vs 7.45ms |
| 4. C++ 精度与 PyTorch 一致 | 端到端 max_diff = 5.264e-04 |
| 5. 推理图尚未完成 | 端到端 GPU 推理需要完成 ggml graph |

### 6.3 预期端到端性能

完成 GGML 推理图后，基于基础算子 benchmark 推算：

| 后端 | 预期推理时间 | 相对 PyTorch CPU 加速 |
|------|-------------|----------------------|
| CPU (ggml) | ~2-5 s | 3-8x |
| CUDA | ~0.1-0.5 s | **30-150x** |
| Vulkan | ~0.2-1.0 s | **15-75x** |
| PyTorch CPU | ~15 s | 1.0x |

---

## 7. 构建命令参考

```bash
# CPU 构建
cmake -B build -DRMBG_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# CUDA 构建 (RTX 3060 = sm_86)
cmake -B build-cuda -DRMBG_BUILD_TESTS=ON -DRMBG_GGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j$(nproc)

# Vulkan 构建
cmake -B build-vulkan -DRMBG_BUILD_TESTS=ON -DRMBG_GGML_VULKAN=ON \
      -DVulkan_INCLUDE_DIR=/home/ludahai/VulkanSDK/1.4.350.0/x86_64/include \
      -DVulkan_LIBRARY=/usr/lib/x86_64-linux-gnu/libvulkan.so \
      -DVulkan_GLSLC_EXECUTABLE=/usr/local/bin/glslc \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j$(nproc)

# 运行后端测试
build-cuda/tests/test_ggml_backends
build-vulkan/tests/test_ggml_backends

# 运行精度测试
RMBG_ROOT=$(pwd) build/tests/test_forward_alpha
```

---

**报告生成时间**: 2026-08-03  
**评测工具**: test_ggml_backends, benchmark_quick.py, parity tests
