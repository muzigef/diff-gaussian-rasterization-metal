# 验证记录

日期：2026-09-05。设备：Apple M3 Pro（arm64），Xcode 26.0.1。原生 C++17，Torch 扩展 C++20；Python 3.14.2、Torch 2.14.0、NumPy 2.5.2。实际执行 Metal/MPS 内核，没有用 CPU fallback 代替 GPU 验证。

## 原生测试

`ctest --test-dir build --output-on-failure`：**17 项通过，0 失败，0 跳过**。

- CPU 手算：单 Gaussian、深度顺序、空输入、非法数据与资源限制。
- Metal：单点/重叠、空输入黑图、全部剔除时背景、alpha cap/skip/early stop。
- tile 边界、非整 tile 尺寸、非对角 covariance、相机平移旋转与非对称 FOV。
- 同深度稳定排序，1、2、255、256、257、513 点 scan/sort，固定种子场景。
- 实例/分配/投影限制，frame 生命周期、移动语义。
- JSON schema、形状、布尔值冒充数字等输入拒绝。

整数中间量要求一致，包括 offsets、半径、tile 矩形、排序 ID、ranges、最后贡献位置。

| 浮点项目 | 门槛 |
| --- | --- |
| 投影中心绝对误差 | 2e-4 |
| conic / cov2D | 5e-5 × max(1, 参考绝对值) |
| 图像最大绝对误差 | 2e-4 |
| 图像 RMSE | 2e-5 |
| 最终透射率绝对误差 | 2e-5 |

`fixtures/overlap.json` 的 GPU 示例：65×49、2 个 Gaussian、18 个 tile 实例；图像最大绝对误差 **1.7289263e-7**，RMSE **9.3102165e-9**。小场景冷启动耗时不作为性能结论。

## Python / 梯度测试

`.venv/bin/python -m pytest tests/python -q`：**40 项通过**。独立 CPU double PyTorch reference 位于 `tests/python/reference.py`，不调用 Metal backward；新增原源码 Float32 oracle 位于 `tests/upstream_oracle/`。

| 覆盖 | 内容 |
| --- | --- |
| 10 组 Forward/Backward | RGB 与 SH 0–3 × covariance 与 scale/rotation，逐类梯度对照 |
| 预计算路径 | SH 转 RGB、scale/rotation 转 covariance；非单位 scale modifier 原版梯度约定 |
| 可见性/异常 | near 阈值、离屏点、prefiltered 错误和 debug Forward 快照 |
| 空输入 | 预计算 RGB/covariance 的黑图和空梯度 |
| SH clamp | 负颜色通道的 SH 梯度为零 |
| 输入布局/状态 | 非连续输入、不对齐 storage_offset、两次 Forward 后一起反传 |
| 相机/跨 tile | 两组非单位 view/combined projection、6 点多 tile 梯度，prefiltered 正常路径 |
| 优化 | 20 步 Adam 拟合颜色，最终 loss 小于初值 25% |
| 有限差分 | means3D、opacity、SH、scale、rotation 的代表元素 |
| alpha cap | 原版限幅处仍反传的替代梯度 |
| 原源码直接对照 | 16 组随机 Forward/Backward，SH 0–3 与四种输入组合 |
| 退化和输入工具 | 零 determinant 单点剔除、有限负 determinant 原版分支、零/极小 quaternion 的 PLY 激活 |

图像 atol/rtol=2e-4；梯度 atol=5e-4、rtol=2e-3。有限差分 epsilon=1e-3，门槛为 0.015 + 数值梯度绝对值的 1%，避开离散分支。它是独立交叉验证，不是所有元素和分支的穷尽证明。

## 构建与消费

- Release CMake 构建通过，并实际编译执行 Metal shader。
- `pip install -e . --no-build-isolation --no-deps` 成功；在工程外 /tmp 导入已安装扩展，MPS 可用。
- `cmake --install` 后独立 `tests/consumer` 通过 find_package 构建、链接并在 GPU 上运行。
- 原生库不依赖 PyTorch；可微接口通过可选 Torch 扩展提供。

## 官方数据对照

来源为 [官方项目](https://github.com/graphdeco-inria/gaussian-splatting) 的 [预训练模型](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) 和 [评估图片](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip)。

`tools/fetch_public_sample.py --images` 提取 train 场景选定 checkpoint、相机、配置及第一张公开测试图像。ZIP CRC 校验后在 `output/public/train/provenance.json` 和 `image_provenance.json` 记录 URL、成员路径、字节数、SHA-256。分段数据缓存于 .downloads，可重试。

`tools/render_public_sample.py` 读取原 PLY，执行模型层的 opacity sigmoid、scale exp、quaternion normalization，再调用 degree 3 的 Metal rasterizer。由相机 JSON 的 camera-to-world rotation/position 构造 view 与 combined projection，输出 PNG 和同名 JSON 指标。

公开图片没有逐内核缓存或 backward 梯度。官方仓库提醒预训练模型与论文结果存在版本差异，因此 render/GT 误差必须结合 checkpoint、相机和分辨率解释，不能直接认证 CUDA 后端逐元素等价。

### 本机实际结果

官方 train、iteration 7000、camera img_name=00001，559,263 个 Gaussian，980×545，与公开图片相同分辨率。公开 ZIP 使用照片名 00001、00009 等，不是当前 render.py 的枚举序号；工具以包内路径和 camera img_name 对齐。

| 检查 | 结果 |
| --- | --- |
| 可见 Gaussian | 457,087 |
| Forward / Backward | 成功；means3D、means2D、SH、opacity、scale、rotation 梯度均有限且有非零元素 |
| 64 个固定网格像素对 CPU double | 最大绝对误差 2.949212e-6，RMSE 4.981214e-7，通过 2e-4 门槛 |
| 对公开 render | RMSE 0.06355186，PSNR 23.93743 dB |
| 对公开 GT | RMSE 0.08829261，PSNR 21.08151 dB |

CPU 抽样包含模型全部 Gaussian 的候选选择与合成，只抽样输出像素。该检查证明这些同输入像素符合独立数学参考，不能推及未检查的像素、梯度或所有输入。公开 render 比较存在明显差异，**不作为等价通过**；代码/训练版本差异的贡献尚未逐项归因。

单次冷样本 Forward 约 0.295 秒、Backward 约 0.268 秒，包括当前绑定同步开销；未预热或重复统计，不是吞吐基准。

可追溯记录保存在 [public-sample.json](public-sample.json)，本地导出为 `output/public/train/metal_00001_980.png`，完整抽样数据在 `cpu_pixel_check.json`。

## 深入审计补充

完整记录见 [DIFFERENCE_ANALYSIS.md](DIFFERENCE_ANALYSIS.md)。原 CUDA 数学体在 Host Float32 上完整渲染同一场景，Metal 对其 RMSE=9.37461e-6；最大绝对误差 0.00261343，172 个通道超过 2e-4 门槛，**全图数值等价未通过**。相同深度输入的独立排序与 Metal 全部一致；最大 12 个像素离群点已追踪到 alpha cutoff 分支。

真实模型上另检查 36 个固定像素 loss 的全部参数梯度，与原源码解析反向结果在既定门槛内。Host oracle 不是 NVIDIA GPU 执行，不能替代 CUDA golden。

已复现并修复零 determinant 错误中断整帧，以及 PLY quaternion normalization 缺少 epsilon；二者都未影响当前官方 train 场景。详见 [difference-audit.json](difference-audit.json)。

## 尚未验证

- 固定 CUDA 提交、同一输入的逐阶段 Float32 数据与全部梯度直接对照。
- 所有阈值邻域、退化输入、全部 SH 系数有限差分及大型随机梯度压力测试。
- 实际 3DGS 工程完整训练、densification 与收敛质量。
- 生产吞吐、长期运行、峰值进程内存、GPU sanitizer/capture。
- macOS 最低版本、其他 Torch 版本、其他 Apple GPU 或 iOS。

现有测试通过不等于“完美复刻”已经验收。
