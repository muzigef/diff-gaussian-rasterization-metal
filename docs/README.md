# 文档导航

本工程把固定版本 `diff-gaussian-rasterization` 的 CUDA 光栅化后端迁移到 Apple Silicon Metal，长期采用 C++ / Objective-C++ + MSL。当前提供原生 C++ 前向接口，以及保留原 Python API 的 PyTorch MPS 可微接口。

首次运行请从[工程 README](../README.md)开始。以下文档按当前代码编写，功能实现、已执行验证与待完成工作分别说明。

## 按任务阅读

| 你要做什么 | 阅读文档 |
| --- | --- |
| 安装、渲染火车示例、切换视角 | [工程 README](../README.md) |
| 理解模块边界、GPU 流水线和状态保存 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 在 Python 中调用渲染、读取梯度、接入训练 | [PYTHON_API.md](PYTHON_API.md) |
| 在 C++ 应用中链接库、获取图像或 Metal texture | [CPP_API.md](CPP_API.md) |
| 准备 PLY、相机、协方差和 JSON 场景 | [DATA_FORMATS.md](DATA_FORMATS.md) |
| 修改代码、选择验证命令、定位构建和渲染问题 | [DEVELOPMENT.md](DEVELOPMENT.md) |
| 核对固定 CUDA 版本的分支和梯度约定 | [MIGRATION.md](MIGRATION.md) |
| 确认哪些内容已经验证、哪些尚未通过 | [VALIDATION.md](VALIDATION.md) |
| 理解官方参考图与 Metal 图的差异 | [DIFFERENCE_ANALYSIS.md](DIFFERENCE_ANALYSIS.md) |

## 接口能力边界

| 入口 | 输入 | 输出 / 用途 |
| --- | --- | --- |
| C++ `MetalRasterizer::render` | `Scene`，预计算 RGB 和三维协方差 | GPU frame、RGB 回读；原生前向渲染 |
| Python `GaussianRasterizer` | MPS Tensor，SH 或 RGB、scale/rotation 或协方差 | 图像、半径，以及一阶反向传播 |
| `rasterizer-demo` | schemaVersion 1 的 JSON | 小场景 PPM 与 CPU 参考对照 |
| `render_public_sample.py` | 3DGS PLY 和相机 JSON | 真实场景 PNG、可选反向传播检查 |

上层照片处理、完整 3DGS 训练、增密和裁剪不由本库实现。当前也没有可直接使用的原生公共 C++ Backward API；完整可微入口见 Python 文档。

## 证据文件

| 文件 | 内容 |
| --- | --- |
| [upstream.json](upstream.json) | 参考提交及实际使用源码的 SHA-256 |
| [public-sample.json](public-sample.json) | 已执行的官方场景渲染、梯度检查和像素抽样记录 |
| [difference-audit.json](difference-audit.json) | 原源码 Host 对照、差异归因与修复证据 |

公开 render、真实照片 GT、同输入 CPU reference、同输入 NVIDIA GPU 输出是不同的证据。当前已有前三类；**尚无严格配对的 NVIDIA GPU 实测基准，不能据此宣称 CUDA 全量等价**。具体数值与边界以验证文档为准。

文档依据：工程公共头文件、Python wrapper、Torch 绑定、MSL、测试和工具脚本。修改接口时同步更新对应文档；修改兼容行为时同步更新迁移契约与验证记录。
