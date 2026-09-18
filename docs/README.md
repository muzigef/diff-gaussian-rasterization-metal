# 文档导航

本工程把固定版本 `diff-gaussian-rasterization` 的 CUDA 光栅化后端迁移到 Apple Silicon Metal，长期采用 C++ / Objective-C++ + MSL。当前提供原生 C++ 前向接口，以及保留原 Python API 的 PyTorch MPS 可微接口。

首次运行请从[工程 README](../README.md)开始。**最近文档核对：2026-09-18，实现基线 `316f4c6`；最近完整 GPU 验证：2026-09-17。** 使用与架构文档描述当前实现；历史分析保留原日期和测量值，以免把旧实验当成新结果。

## 按任务阅读

| 你要做什么 | 阅读文档 |
| --- | --- |
| 安装、渲染火车示例、切换视角 | [工程 README](../README.md) |
| 理解模块边界、GPU 流水线和状态保存 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 阅读迁移后的实现、性能实测和数值验收结论 | [TILE_MIGRATION_REPORT.md](TILE_MIGRATION_REPORT.md) |
| 对比迁移前 Metal 与 CUDA / ROCm 的组织及迁移路线 | [TILE_MIGRATION_ANALYSIS.md](TILE_MIGRATION_ANALYSIS.md) |
| 在 Python 中调用渲染、读取梯度、接入训练 | [PYTHON_API.md](PYTHON_API.md) |
| 在 C++ 应用中链接库、获取图像或 Metal texture | [CPP_API.md](CPP_API.md) |
| 准备 PLY、相机、协方差和 JSON 场景 | [DATA_FORMATS.md](DATA_FORMATS.md) |
| 修改代码、选择验证命令、定位构建和渲染问题 | [DEVELOPMENT.md](DEVELOPMENT.md) |
| 核对固定 CUDA 版本的分支和梯度约定 | [MIGRATION.md](MIGRATION.md) |
| 确认哪些内容已经验证、哪些尚未通过 | [VALIDATION.md](VALIDATION.md) |
| 找到新 Metal 图、官方 CUDA 公开图、ROCm/CPU 参照与报告 | [ARTIFACTS.md](ARTIFACTS.md) |
| 理解官方参考图与 Metal 图的差异（2026-09-05 历史审计） | [DIFFERENCE_ANALYSIS.md](DIFFERENCE_ANALYSIS.md) |

## 接口能力边界

| 入口 | 输入 | 输出 / 用途 |
| --- | --- | --- |
| C++ `MetalRasterizer::render` | `Scene`，预计算 RGB 和三维协方差 | GPU frame、RGB 回读；原生前向渲染 |
| Python `GaussianRasterizer` | MPS Tensor，SH 或 RGB、scale/rotation 或协方差 | 图像、半径，以及一阶反向传播 |
| `rasterizer-demo` | schemaVersion 1 的 JSON | 小场景 PPM 与 CPU 参考对照 |
| `render_public_sample.py` | 3DGS PLY 和相机 JSON | 真实场景 PNG、可选反向传播检查 |
| `dgr-native-benchmark` | 验证工具导出的原生子集二进制 | C++ 前向性能与同子集图像对照；测试工具，非公共模型格式 |

上层照片处理、完整 3DGS 训练、增密和裁剪不由本库实现。当前也没有可直接使用的原生公共 C++ Backward API；完整可微入口见 Python 文档。

## 证据文件

| 文件 | 内容 |
| --- | --- |
| [upstream.json](upstream.json) | 参考提交及实际使用源码的 SHA-256 |
| [public-sample.json](public-sample.json) | 2026-09-05 官方场景渲染、梯度检查和像素抽样记录 |
| [difference-audit.json](difference-audit.json) | 2026-09-05 原源码 Host 对照、差异归因与修复证据 |
| [migration-results.json](analysis/tile-2026-09-17/migration-results.json) | 2026-09-17 七组场景、性能、重复反向实验、未通过项及源码哈希 |
| [metal-validation.log](analysis/tile-2026-09-17/metal-validation.log) | 开启 Metal 检查层后的 CTest/pytest 结果 |
| [consumer.log](analysis/tile-2026-09-17/consumer.log)、[wheel-consumer.log](analysis/tile-2026-09-17/wheel-consumer.log) | 安装后的原生库与 Python wheel 消费验证 |

当前已有官方公开 render、真实照片 GT、同输入 CPU reference，以及 2026-09-07 保存的同输入 ROCm GPU 结果；迁移验证重用这些 ROCm 结果，未重新运行远端 GPU。**尚无严格配对的 NVIDIA GPU 实测基准，不能据此宣称 CUDA 全量等价**。各类图像及本地资产是否随 Git 分发，见[产物说明](ARTIFACTS.md)。

## 本次核对范围与维护

已核对工程 README、工程约定和本目录全部 Markdown：公共头文件、Python wrapper、Torch 绑定、MSL、CMake/setup、测试和工具参数。迁移报告所列 6 个实现文件的 SHA-256 与当前源码一致；CTest 枚举为 19 项，pytest 收集为 45 项。

本次检查 127 个本地链接、6 个指向历史 Git 内容的链接和 15 个产物路径，均有效；8 个工具的 `--help` 正常。实际运行文档中的 Python 最小前向/反向、安装后 C++ 接入、JSON 渲染与浮点图导出 PNG 示例，均成功。文档校对不把历史测试改记为当日完整 GPU 重跑。

`TILE_MIGRATION_ANALYSIS.md` 描述 `9ac670e` 的迁移前方案，`DIFFERENCE_ANALYSIS.md` 描述 2026-09-05 的实验；当前状态以 `ARCHITECTURE.md`、`VALIDATION.md` 和 `TILE_MIGRATION_REPORT.md` 为准。机器生成的 JSON、日志和上游许可证保留原始内容。

修改接口时同步更新对应文档；修改兼容行为时同步更新迁移契约与验证记录。新增实验应保存日期、代码与输入哈希、设备、通过/跳过/失败数量，避免覆盖历史证据。
