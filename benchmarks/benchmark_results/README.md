# RMBG-2.0 GGML 综合评测报告

**评测日期**: 2026-08-03  
**评测环境**: Linux 20.04, CPU (x86_64)  
**项目版本**: RMBG-2.0-GGML (commit: current)

---

## 1. 评测概述

本次评测针对 RMBG-2.0 GGML C++ 推理实现与 PyTorch 参考实现进行全面对比，涵盖：

- **精度对比**: 各组件数值精度与 PyTorch 参考实现的差异
- **推理速度**: CPU 后端推理时间对比
- **效果图对比**: 实际图片背景去除效果可视化对比

### 1.1 项目当前状态

根据 `docs/PORTING.md` 和项目代码分析：

| 组件 | 状态 | 说明 |
|------|------|------|
| GGUF 转换 + 元数据 | ✅ 完成 | `convert_rmbg_to_gguf.py`, `rmbg-cli info` |
| PyTorch Bridge (过渡方案) | ✅ 完成 | `scripts/rmbg_pytorch_bridge.py` |
| Swin-L CPU Forward (24 blocks) | ✅ 完成 | 所有 parity tests 通过 |
| BiRefNet Decoder (ASPPDeformable) | ✅ 完成 | `forward_alpha` 测试通过 |
| GGML 推理图 | ❌ 未完成 | `model_loader.cpp` 返回错误 |
| CUDA/Vulkan 后端 | ❌ 未完成 | 依赖 GGML 推理图 |

**重要说明**: 当前 C++ 实现使用纯 C++ 数值计算（非 ggml graph），端到端推理图尚未完成。

---

## 2. 精度对比 (C++ vs PyTorch)

### 2.1 组件级精度测试

所有 parity 测试使用 PyTorch 导出的参考数据（GGUF 格式）与 C++ 实现进行对比：

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

### 2.2 精度分析

**结论**: C++ 实现与 PyTorch 参考实现精度一致

- **最大差异**: 5.264e-04 (端到端 forward_alpha)
- **平均差异**: 5.259e-06
- **容差标准**: atol=5e-2, rtol=1e-2 (测试通过)

差异来源分析：
1. **浮点运算顺序**: C++ 和 PyTorch 的矩阵乘法实现可能导致微小差异
2. **GELU 激活函数**: C++ 使用精确的 `erf` 形式
3. **LayerNorm**: 数值稳定性处理略有不同

所有差异在可接受范围内（< 1e-3），不影响最终视觉效果。

---

## 3. 推理速度对比

### 3.1 CPU 后端推理时间

| 实现 | 输入尺寸 | 推理时间 | 备注 |
|------|----------|----------|------|
| **C++ (纯标量实现)** | 1024×1024 | ~11 分钟 | 未使用 ggml graph 优化 |
| **PyTorch (CPU)** | 1024×1024 | ~15 秒 | 使用 Torch 优化 |
| **加速比** | - | **~44x** | PyTorch 更快 |

### 3.2 速度分析

**当前状态**: C++ 实现显著慢于 PyTorch

**原因分析**:
1. **未使用 ggml graph**: 当前 C++ 实现是纯标量计算，未利用 ggml 的图优化
2. **未使用 SIMD 指令**: 纯 C++ 实现未使用 AVX/NEON 等向量指令
3. **未使用多线程**: 当前实现是单线程的
4. **内存访问模式**: 未优化的内存布局导致缓存命中率低

**预期改进**:
- 完成 ggml graph 后，可利用 ggml 的优化内核
- 支持 CPU 多线程、SIMD 指令
- 支持 GPU 后端 (CUDA/Vulkan) 进一步加速

### 3.3 GPU 后端状态

| 后端 | 状态 | 说明 |
|------|------|------|
| CUDA | ❌ 不可用 | 当前系统 NVIDIA 驱动版本过旧 |
| Vulkan | ❌ 不可用 | 当前系统 GLIBC 版本问题 |
| Metal | ❌ 未测试 | 仅 macOS 支持 |

**注意**: GPU 后端需要完成 GGML 推理图后才能测试。

---

## 4. 效果图对比

### 4.1 测试图片

使用项目自带的示例图片进行测试：
- `t4.png`: 单物体场景
- `collage5.png`: 多物体拼贴图

### 4.2 PyTorch 推理效果

PyTorch BiRefNet 模型成功去除背景，生成高质量的 alpha matte。

效果图已保存至：
- `comparison_t4.png`
- `comparison_collage5.png`

### 4.3 C++ 效果图

**当前状态**: 由于 GGML 推理图尚未完成，无法生成端到端 C++ 效果图。

但根据精度测试结果，C++ 实现与 PyTorch 数值差异极小（max_diff < 1e-3），预期视觉效果与 PyTorch 基本一致。

---

## 5. 总结与建议

### 5.1 主要发现

| 维度 | 结论 |
|------|------|
| **精度** | ✅ C++ 实现与 PyTorch 精度一致 (max_diff < 1e-3) |
| **速度 (CPU)** | ⚠️ 纯 C++ 实现目前慢于 PyTorch (~44x) |
| **GPU 支持** | ❌ CUDA/Vulkan 后端需要完成 GGML 推理图 |
| **项目状态** | 🚧 核心算法已完成，推理图待完成 |

### 5.2 已完成工作

1. ✅ GGUF 模型格式转换工具
2. ✅ Swin-L Backbone 完整实现 (24 blocks)
3. ✅ BiRefNet Decoder 实现 (ASPPDeformable)
4. ✅ 端到端 forward_alpha 精度验证通过
5. ✅ 完整的 parity 测试套件

### 5.3 待完成工作

1. ❌ **GGML 推理图**: 将纯 C++ 实现转换为 ggml graph
2. ❌ **GPU 后端**: 启用 CUDA/Vulkan 后端
3. ❌ **性能优化**: 利用 ggml 的优化内核和多线程
4. ❌ **端到端推理**: 完成从图片输入到 alpha mask 输出的完整流程

### 5.4 预期改进

完成 GGML 推理图后，预期可获得：

| 后端 | 预期加速比 | 说明 |
|------|------------|------|
| CPU (ggml) | 5-10x | 利用 SIMD、多线程优化 |
| CUDA | 20-50x | GPU 加速 |
| Vulkan | 10-30x | 跨平台 GPU 加速 |

---

## 6. 附录

### 6.1 测试命令

```bash
# 构建项目
cmake -B build -DRMBG_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 运行 parity 测试
RMBG_ROOT=$(pwd) build/tests/test_forward_alpha

# 运行 benchmark
python3 scripts/benchmark_quick.py --output_dir benchmarks/benchmark_results --device cpu
```

### 6.2 文件清单

- `benchmark_report.json`: 评测数据
- `comparison_t4.png`: 效果图对比
- `comparison_collage5.png`: 效果图对比
- `scripts/benchmark_quick.py`: 评测脚本

### 6.3 参考资料

- [docs/PORTING.md](docs/PORTING.md): 项目移植计划
- [tests/parity.hpp](tests/parity.hpp): 精度对比工具
- [scripts/rmbg_pytorch_bridge.py](scripts/rmbg_pytorch_bridge.py): PyTorch 参考实现

---

**报告生成时间**: 2026-08-03  
**评测工具版本**: benchmark_quick.py v1.0
