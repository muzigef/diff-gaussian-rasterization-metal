# 验证记录

[返回文档导航](README.md)

最近完整验证日期为 **2026-09-17**，对应 tile 迁移实现 `316f4c6`。2026-09-18 文档校对确认实现哈希与报告一致，并重新枚举测试清单；未把这次文档检查记为全套 GPU 重跑。

**19 项 CTest、45 项 pytest 执行通过，启用 Metal API/GPU Validation 后也全部通过。七组真实场景前向与旧 Metal 完全一致；严格 CPU/ROCm 图像、部分 dense 梯度及 autograd 对照仍未全部通过。** 功能测试通过不等于“完美复刻 CUDA”已经验收。

## 环境与证据

设备 Apple M3 Pro（arm64）、macOS 15.6.1、Xcode 26.0.1；原生 C++17、Torch 扩展 C++20、MSL 3.1。Python 3.14.2、Torch 2.14.0、NumPy 2.5.2。实际执行 Metal/MPS 内核，没有用 CPU fallback 代替 GPU 验证。

| 证据 | 内容与范围 |
| --- | --- |
| [TILE_MIGRATION_REPORT.md](TILE_MIGRATION_REPORT.md) | 当前迁移结果、七组场景、性能方法与未通过项 |
| [migration-results.json](analysis/tile-2026-09-17/migration-results.json) | 每项数值、输入/源码/扩展哈希、旧新重复反向实验 |
| [metal-validation.log](analysis/tile-2026-09-17/metal-validation.log) | 检查层下 19 项 CTest 和 45 项 pytest 结果 |
| [validation-enabled.log](analysis/tile-2026-09-17/validation-enabled.log) | API/GPU Validation 启用及空场景执行记录 |
| [consumer.log](analysis/tile-2026-09-17/consumer.log)、[wheel-consumer.log](analysis/tile-2026-09-17/wheel-consumer.log) | 安装后 C++ 与 Python wheel 消费验证 |
| [upstream.json](upstream.json) | 固定 CUDA 版本与早期实际源码哈希 |

固定 CUDA 参考为 `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`。原源码 Float32 oracle 在 CPU 上执行；ROCm 对照来自 2026-09-07 保存的同输入 GPU 结果，本次迁移未重新运行远端 GPU。**没有同输入 NVIDIA GPU 逐阶段/梯度实测。** 官方公开 render 不能替代该基准。

## 当前测试覆盖

### 原生：19 项通过，0 失败，0 跳过

- 5 项 CPU core：单 Gaussian、深度顺序、空输入、数据校验与资源限制。
- 13 项 Metal：单点/重叠、空输入/全部剔除、alpha cap/skip/early stop、tile 边界、同深度稳定性、扫描规模、相机、随机场景、限制、frame 生命周期、移动语义、分层排序和多批次 tile。
- 1 项 JSON I/O：schema、形状和非法类型拒绝。

新增 `metal.hierarchical_sort` 覆盖 65,537 个输入、三级扫描、跨块重复键及资源池缩小/空输入；`metal.tile_batches` 覆盖 769 个候选、非整 tile 边缘和提前结束。整数中间量要求一致，包括 offsets、半径、tile 矩形、排序 ID、ranges、最后贡献位置。

### Python：45 项通过，0 失败，0 跳过

| 文件 | 数量 | 主要覆盖 |
| --- | ---: | --- |
| [test_compatibility.py](../tests/python/test_compatibility.py) | 25 | RGB/SH 0–3、两种形状输入、Forward/Backward、原版梯度约定、有限差分、20 步优化、输入布局、可见性/空输入/退化、MPS 顺序与并发调用 |
| [test_upstream_source.py](../tests/python/test_upstream_source.py) | 19 | 16 组固定 CUDA 数学体 CPU 对照、有限负 determinant、2 组多批次逆序合成与边缘像素 |
| [test_public_sample.py](../tests/python/test_public_sample.py) | 1 | PLY 零/极小四元数激活 |

MPS 生命周期覆盖临时切片、前后排队的 Tensor 运算、12 个保留的 autograd graph、debug 开关及 4 个 Python 调用线程。并发调用测试不代表已经验证真正多个 MPS stream。

全部源码对照需要相邻 CUDA 源码和 GLM；缺失时有测试跳过。依赖准备与 `-rs` 检查见[开发文档](DEVELOPMENT.md#原源码对照依赖)。测试收集数量不等于测试执行通过数量。

### 构建与消费

Release 构建实际编译并执行 Metal shader。安装后的 C++ 库通过独立 `find_package(DGR)` 消费方构建、链接和 GPU 执行；Python wheel 安装到独立目录后实际执行 SH3 + scale/rotation 前向和反向。原生库不依赖 PyTorch，可微入口通过可选 Torch 扩展提供。

## 数值标准

这些是不同层级的现有标准，不能把小场景的相对容差或平均误差代替真实场景逐元素门槛。

| 检查 | 标准 |
| --- | --- |
| 原生投影中心 | 绝对误差不超过 2e-4 |
| 原生深度 | 2e-5 × max(1, 参考绝对值) |
| 原生 conic / cov2D | 5e-5 × max(1, 参考绝对值) |
| 原生图像 | 最大绝对误差 2e-4，且 RMSE 不超过 2e-5 |
| 原生最终透射率 | 绝对误差不超过 2e-5 |
| Python 小场景图像主对照 | atol=2e-4、rtol=2e-4；部分专项对照更严格 |
| Python / 真实场景梯度主对照 | 每元素绝对误差不超过 5e-4 加参考绝对值的 0.002 倍 |
| 七组真实场景图像 | 每通道元素绝对误差不超过 2e-4，不叠加相对容差 |
| 迁移前后 Metal 前向及有效状态 | 逐元素一致；状态使用语义解码，不要求私有 byte buffer ABI 一致 |

`atol` 是绝对容差，`rtol` 是随参考值大小增加的相对容差。梯度表示损失对某个参数微小变化的敏感程度；每个参数位置都单独检查，不能只看梯度最大值或均值。

代表元素的有限差分使用 epsilon=1e-3，门槛为 0.015 加数值梯度绝对值的 1%，避开离散分支。它是独立交叉验证，不是所有元素和分支的穷尽证明，也不替代原版 alpha cap / scale modifier 特殊梯度约定。

## 七组真实场景：当前仍有未通过项

559,263 个 Gaussian，三个 980 宽视角、一个 1959 宽视角，以及主视角的 SH+covariance、RGB+scale、RGB+covariance 路径。所有后端使用已保存的同一份 CPU Tensor snapshot；输入与参照哈希在验证前检查。

| 检查 | 结果 |
| --- | --- |
| 新 Metal 对旧 Metal 图像、整数及有效浮点状态 | 7/7 逐元素一致 |
| 相同 Metal 几何下的 CPU 稳定排序 | 7/7 的 IDs/ranges 零差异 |
| 稀疏像素损失的全部参数梯度 | 7/7 对旧 Metal、ROCm、CPU 及同状态 CPU 反向通过 |
| 对 CPU 的严格图像门槛 | 7/7 存在超限；最大绝对误差约 0.00252–0.00303 |
| dense 梯度 | 少量高敏感参数仍超限，逐后端记录于 JSON |
| autograd 与直接 backward | 最终记录 4/7 通过，3/7 有敏感 scale/covariance 元素超限 |
| 汇总 `strict_cpu_passed` / `autograd_passed` | 均为 false |

主视角 980×545 对 CPU 的最大误差为 0.00261343，172 / 1,602,300 个颜色通道超出 2e-4；详细七组数值见[迁移报告](TILE_MIGRATION_REPORT.md#严格数值门槛与尚未通过的项目)。这些图像差异继承旧 Metal，不能归因于本次 radix 或协作批次改变。

新旧版本在相同 Forward 状态上各重复 dense backward 5 次，都观察到敏感协方差梯度的超限波动；原子累加顺序和高条件数会放大差异。重复实验不构成允许放宽容差的理由，也不把一次偶然通过覆盖为最终结论。

复验命令、资产要求与脚本退出码含义见[迁移报告](TILE_MIGRATION_REPORT.md#复现)。图像、浮点数组和参照位置见 [ARTIFACTS.md](ARTIFACTS.md)。

## 历史证据：2026-09-05

以下保留早期测量的来源，不能用来表示当前性能或当前测试总数。

| 记录 | 当时结果 |
| --- | --- |
| 原生 / Python 测试 | 17 项 CTest、40 项 pytest 通过；当前已扩展为 19/45 |
| `fixtures/overlap.json` | 65×49，2 个 Gaussian，18 个实例；最大图像误差 1.7289263e-7、RMSE 9.3102165e-9 |
| train 7000、00001、980×545 | 559,263 个 Gaussian，457,087 个可见；前向/反向成功 |
| 64 个固定网格像素 vs 独立 CPU double | 最大绝对误差 2.949212e-6、RMSE 4.981214e-7 |
| 对官方公开 render / GT | RMSE 分别为 0.06355186 / 0.08829261，未认证同 checkpoint 等价 |
| 整图 vs 原源码 Host Float32 | RMSE 9.37461e-6；最大误差 0.00261343、172 个通道超限 |

原始记录见 [public-sample.json](public-sample.json)、[difference-audit.json](difference-audit.json)，解释与边界修复见 [DIFFERENCE_ANALYSIS.md](DIFFERENCE_ANALYSIS.md)。旧记录的冷启动时间包含不同工作量，不能与迁移报告的预热中位数直接比较。

## 尚未完成的验收

- 同输入 NVIDIA GPU 的逐阶段、图像和全部梯度对照；当前严格 CPU/ROCm 数值失败也仍待解决。
- 所有阈值邻域、退化输入和全部 SH 系数有限差分的穷尽覆盖。
- 实际 3DGS 工程完整训练、densification 与收敛质量。
- 多设备生产吞吐、长期运行、逐 kernel GPU 时间线和实际峰值内存；Metal API/GPU Validation 已执行，GPU capture/profile 尚未完成。
- macOS 最低版本、其他 Torch 版本、其他 Apple GPU、iOS，以及真正多个 MPS stream。
