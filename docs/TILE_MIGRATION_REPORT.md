# CUDA / ROCm 执行策略迁移至 Metal：实施与验证

日期：2026-09-17。设备 Apple M3 Pro，macOS 15.6.1，Python 3.14.2 / Torch 2.14.0。旧版基线 `9ac670e`，旧扩展已保存。固定 CUDA 参考为 `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`；ROCm 使用此前保存的同输入实测结果，本次未重新运行远端 GPU，也没有 NVIDIA 同输入实测。

实现归档于提交 `316f4c6`。2026-09-18 文档校对确认报告所列 6 个实现文件哈希仍与当前源码一致；此校对没有重新测量本报告的性能和全场景数值。

**协作 tile 前向/反向、分层扫描、稳定 radix sort、浮点原子加法、临时资源复用和当前 MPS stream 调度均已实现。19 项原生测试与 45 项 Python 测试通过；七组真实场景前向与旧版逐元素一致。严格跨后端数值验收仍未全部通过，不能称为 CUDA/ROCm 全量数值等价。**

完整数值、源码与扩展哈希、旧版/新版重复实验见 [migration-results.json](analysis/tile-2026-09-17/migration-results.json)。原始图像、状态、梯度及日志保存在本机 `output/migration-20260917/`，大文件不提交到 Git。新 Metal PNG、官方公开 CUDA 图与 ROCm/CPU 参照的路径见 [ARTIFACTS.md](ARTIFACTS.md)。

## 已完成的代码迁移

| 环节 | 当前实现 | 与 CUDA / ROCm 的对应关系 |
| --- | --- | --- |
| tile 执行 | 每个 16×16 threadgroup 对应一个软件 tile | 对应原版 16×16 block |
| 前向 | 协作加载 256 个候选的中心、conic/opacity、RGB；逐像素顺序混合 | 对应 shared memory 批处理；仍按深度前后顺序 |
| 提前结束 | 每个像素保留 done；整组完成判断后统一退出 | 对应原版整块 done 统计；越界和已完成线程仍参与 barrier |
| 反向 | 逆序批次共享加载，按 last contributor 恢复透射率并计算梯度 | 保留原版逆序遍历、限幅替代梯度和参数映射 |
| 梯度累加 | device `atomic_float` 加法；MSL 3.1，关闭 fast math | 保持每像素原子加法，未引入浮点 SIMD 归约 |
| 前缀和 | 256 元素块内扫描 → 递归块总和扫描 → uniform add | 对应设备级 inclusive scan；溢出保护用饱和整数加法 |
| 排序 | 每轮 4 位的稳定 LSD radix；深度 32 位 + tile 有效位 | 对应稳定的 tile/depth key 排序；相同键保持高斯输入顺序 |
| 临时空间 | 原生 renderer 池；Torch 每 stream 独立池 | 复用 scan/sort workspace；saved state 独立持有 |
| Torch 调度 | 在当前 MPS stream 的 serial queue 和 command buffer 内编码 | 继承调用方队列顺序；取消跨独立 queue 的强制等待 |

原生与 Torch 共享 [primitives.h](../src/metal/primitives.h) 的扫描、排序与 dispatch 编排；两种前向输出共享 `render_tile`。原生保留 RGBA texture，Torch 保留 CHW Tensor；Python API、SH 0–3、RGB/covariance 预计算路径、scale/rotation、markVisible 和一阶 autograd 接口不变。

保留 alpha 上限 0.99、跳过阈值 1/255、透射率阈值 0.0001，以及“触发透射率终止的候选不参与混合”的原版规则。last contributor 仍表示最后接受候选在 tile 列表内的 1-based 位置，不是接受数量。

稳定 scatter 的位置由 bucket 的跨块前缀、前面 SIMD 的计数及当前 SIMD 的排他前缀构成；没有用无序全局 atomic ticket 安排同键位置。扫描、radix 和 tile 都 dispatch 完整线程组，尾块通过有效掩码处理。

Metal 仍使用自己的 16 字节实例记录与内部状态布局，没有直接链接 CUB/hipCUB，也没有使用 Apple render pass 的 tile shader。相同的是算法与并行组织，不是二进制 ABI、SIMD 宽度或底层指令。

## 性能与资源

同一 CPU snapshot，559,263 个高斯；每个尺寸预热 3 次，测量 5 次取中位数。计时在 GPU 工作前后执行 MPS synchronize，不包含图像/梯度拷回 CPU；dense backward 使用完全相同的像素权重。旧扩展和新扩展在同一台 Mac 上运行。

| 场景 | 旧前向 | 新前向 | 前向加速 | 旧反向 | 新反向 | 反向加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 980×545 | 209.15 ms | 26.08 ms | 8.02× | 230.99 ms | 17.63 ms | 13.10× |
| 1959×1090 | 495.26 ms | 67.14 ms | 7.38× | 894.31 ms | 42.03 ms | 21.28× |

旧版全局 scan 为 20 轮；新版对该点数为三级扫描、5 次 dispatch。旧 bitonic 为 253 / 276 轮全列表操作；新 radix 为 11 / 12 个 4-bit 轮次，每轮包含 histogram、分层计数扫描和 scatter，**轮次数不等于 kernel 次数**。实例数组不再补齐为 2 的幂，但 radix 需要第二份记录及直方图，不能据此声称总峰值内存下降。

迁移中单独保留了 CAS 版本测量：协作 tile 的同时写入使 CAS 竞争增大，980 场景 backward 一度为约 518 ms；切换原生浮点原子加法后降至约 18 ms。因此，性能收益不能只归因于共享内存读取。

当前 warm benchmark 的 sampled driver allocation 为约 1.16 / 1.25 GB；三个 saved-state Tensor 加 radii 的逻辑字节数为 133,701,792 / 231,245,920。前者含缓存与其他已分配资源，**不是实际 GPU 峰值或单帧独占量**。尚未取得逐 kernel GPU 时间线或系统级峰值内存测量。

原生 C++ 另验证了真实模型的可接收子集：456,979 个高斯、2,941,149 个实例，输出与同子集的 Torch RGB/covariance 路径逐元素一致。中位 wall time 42.79 ms、GPU command time 21.15 ms，请求的活动资源 163,132,268 字节。它先剔除原视角不可见点，再剔除 108 个 float32 协方差不满足原生 `Scene` 正定检查的点；完整 ID 清单在 JSON 中。这不是完整模型性能对比；没有修改协方差、放宽原生契约或改动上面七组完整模型测试。

## 正确性与运行时检查

| 验证 | 结果 |
| --- | --- |
| CMake / CTest | 19 passed，0 failed，0 skipped |
| PyTorch / pytest | 45 passed，包含有限差分、源码 CPU oracle、20 步优化 |
| Metal API Validation + GPU Validation | 19 项 CTest、45 项 pytest 全部通过 |
| 安装后的 C++ find_package consumer | 构建、链接和 GPU 执行成功 |
| 独立目录安装的 Python wheel | 实际执行 SH3 + scale/rotation 前向与反向成功 |
| 七组真实场景 vs 旧 Metal | 图像、radii、counts、ID、ranges、last 和有效浮点状态全部逐元素一致 |
| 用同一 Metal 几何重新做 CPU 稳定排序 | 七组的 IDs / ranges 均零差异 |
| 七组 sparse backward | 对旧 Metal、ROCm、CPU 及同状态 CPU 反向均通过原门槛 |
| dense backward / autograd | 少量敏感点超限，详见下一节，未标为全部通过 |

状态对照使用语义解码器：剔除点的无效 scratch 规范化，预计算输入按调用方值恢复；不是不同版本原始 byte buffer 的逐字节 ABI 比较。

新增覆盖包括 65,537 个输入的多级扫描、跨许多块的重复深度键、池从大场景切到小场景和空场景、769 个候选的三批次前向/逆序反向、非整 tile 边缘和局部提前结束。MPS 测试覆盖在前后排队的 Tensor 运算、非连续/未对齐临时输入、反向前保留 12 张图、debug 同步和四个 Python 调用线程。

运行时检查发现并修复了原生空场景的问题：虽然不会读取候选，绑定为 `Projected*` 的 buffer 仍须至少容纳一个 64 字节元素，不能仅分配 16 字节占位空间。检查层启用证据见 [validation-enabled.log](analysis/tile-2026-09-17/validation-enabled.log)。

MPS 正常路径只在 Forward 读实例数/错误码时等待；`debug=True` 额外等待 Forward 和 Backward 完成。saved state 始终独立，只有短期 workspace 复用。completion handler 不析构 Python Tensor，避免 GIL 与 GPU 等待互相阻塞。实现使用当前 Torch 的内部 stream/storage API，其他 Torch 版本应重新编译验证；尚未在其他型号 Mac 或真正多个 MPS stream 上实测。

## 严格数值门槛与尚未通过的项目

保持图像绝对误差门槛 `2e-4`；每个梯度元素使用绝对容差 `5e-4` 加参考值绝对值的 `0.002` 倍。未扩大阈值，也未忽略少数元素后宣称通过。

| 完整场景 / 输入路径 | 图像对 CPU 最大绝对误差 | 超限通道元素数 | 与旧 Metal 图像 |
| --- | ---: | ---: | --- |
| 00001，980，SH+scale | 0.00261343 | 172 / 1,602,300 | 完全一致 |
| 00072，980，SH+scale | 0.00303003 | 210 / 1,602,300 | 完全一致 |
| 00187，980，SH+scale | 0.00251946 | 93 / 1,602,300 | 完全一致 |
| 00001，1959，SH+scale | 0.00292134 | 747 / 6,405,930 | 完全一致 |
| 00001，980，SH+covariance | 0.00261343 | 177 / 1,602,300 | 完全一致 |
| 00001，980，RGB+scale | 0.00261343 | 172 / 1,602,300 | 完全一致 |
| 00001，980，RGB+covariance | 0.00261343 | 177 / 1,602,300 | 完全一致 |

这些图像差异完全继承旧版。前向计算表达式及 exp 等数学函数的浮点差异，在 alpha / 透射率阈值附近仍可能改变分支；tile 迁移没有改动这些数学策略。对同一 Metal 几何的 CPU 合成也有少量阈值超限，排序复查为零差异，不能把这些超限归因于 radix 排序或共享批次遗漏。

dense 梯度的误差按案例、参数及高斯 ID 完整保存在 JSON 中。原子浮点加法次序会变；三维协方差导数在敏感高斯上会放大很小的累加差异。以 ID 246787 为例，其主视角投影 conic 特征值之比约 46,031，意味着不同方向的响应尺度差别很大，不能只看图像接近就断言梯度也逐元素接近。

为区分迁移错误与原有不确定性，旧扩展、新扩展各在**同一份 Forward 缓冲**上重复 dense backward 5 次；下表是协方差梯度的最大重复极差，即五次最大值减最小值。它是观测范围，不是误差上界或新的验收容差。

| 案例 | 旧版最大重复极差 | 新版最大重复极差 |
| --- | ---: | ---: |
| 00001，980，SH+scale | 0.0175173 | 0.00873876 |
| 00001，1959，SH+scale | 0.0648627 | 0.0565550 |
| 00001，980，SH+covariance | 0.0189912 | 0.0156677 |
| 00001，980，RGB+scale | 0.0398226 | 0.0128835 |
| 00001，980，RGB+covariance | 0.0269710 | 0.0142851 |

新旧重复实验都在 ID 246787 上超限；旧版 00187 视角还在 ID 279445 上超限。新版某些尺度分量也超限，因此不能概括为每个参数的波动都改善。最终一轮 autograd 与直接 backward 比较中，1959 SH+scale 有 1 个 scale 元素超限，SH+covariance 有 6 个 covariance 元素超限，RGB+covariance 有 4 个 covariance 元素超限，均对应 ID 246787。这些检查保留为失败，不以再次运行偶然通过覆盖它们。

结论是：目前没有从排序、批次、边缘线程、生命周期或梯度公式对照中发现新的迁移逻辑错误；已有的跨后端阈值差异与敏感梯度不确定性尚未消除。减少这类误差需要单独研究数学求值顺序或累加策略，可能影响与原 CUDA/ROCm 原子累加方案的一致性。本次没有悄悄改成确定性归约或修改梯度公式。

## 复现

从仓库根目录运行；先完成 editable 安装，并按[开发文档](DEVELOPMENT.md#原源码对照依赖)准备 pytest 所需的相邻 CUDA 源码与 GLM。下面第一组测试不需要大模型资产：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDGR_BUILD_TORCH=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j 8
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest tests/python -q

MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ctest --test-dir build --output-on-failure
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 .venv/bin/python -m pytest tests/python -q
```

真实场景资产为之前 ROCm 全量验证保存的 `inputs.pt`、`inputs.json`、`precomputed_00001_980.pt` 和七组 `metal/rocm/cpu` 子目录；每个子目录包含图像、状态、dense/sparse 梯度及 provenance report。它们是本机已保存的显式测试资产，不随 Git 分发，公开模型下载脚本也不会生成这些跨后端参照。运行库不依赖相邻仓库。以下命令需要这些资产及给定的固定源码/GLM 路径；脚本先核对 snapshot / 输入路径 / 预计算数据哈希，再比较数值。

```bash
.venv/bin/python tools/benchmark_scene.py \
  --snapshot ../diff-gaussian-rasterization-rocm/output/full-scene-20260907/inputs.pt \
  --output output/benchmark.json

.venv/bin/python tools/validate_tile_migration.py \
  --references ../diff-gaussian-rasterization-rocm/output/full-scene-20260907 \
  --upstream ../diff-gaussian-rasterization-rocm/upstream \
  --glm ../diff-gaussian-rasterization-rocm/third_party/glm \
  --export-native --output output/tile-validation

.venv/bin/python tools/audit_gradient_repeats.py \
  --references ../diff-gaussian-rasterization-rocm/output/full-scene-20260907 \
  --output output/tile-repeats

build/dgr-native-benchmark output/tile-validation/native-scene.bin \
  output/tile-validation/native-expected.bin
```

完整验证脚本会保存失败细节；前向旧版对照或 autograd 对照失败时返回非零。跨 CPU 的严格门槛另有 `strict_cpu_passed` 字段，当前为 false，因此进程返回 0 也不代表跨后端全等价。重复实验保留实测范围，不自动放宽门槛。

原生 benchmark 的两个二进制文件仅用于本次测试运输，不是公共资产格式；导出同时保存筛选说明。旧扩展备份位于本机 `output/migration-20260917/baseline-extension.so`；旧版计时及重复范围已经纳入公开 JSON，无需依赖该二进制来阅读报告。

实现依据：Apple 的 [线程组与内存同步规则](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)、[Metal compute 与浮点原子操作](https://developer.apple.com/videos/play/wwdc2022/10159/)，以及与本机 Torch git version 对应的 [MPSStream](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/mps/MPSStream.mm) / [MPSAllocator](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/mps/MPSAllocator.mm) 源码。
