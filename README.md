# diff-gaussian-rasterization-metal

将本地 `diff-gaussian-rasterization` 的 CUDA 后端迁移到 Apple Silicon Metal。长期架构已改为 **C++ + Objective-C++ + Metal Shading Language**，使用 CMake；原 Swift 工程已移除。

已实现固定上游版本的 Forward、Backward、SH 0–3 阶、scale/rotation 与预计算 covariance、预计算 RGB、`markVisible` 和 PyTorch MPS custom autograd。原 Python 类名、参数顺序及 `(color, radii)` 返回接口保留。目前通过本机 GPU 和独立 CPU 数学参考测试；**尚不能宣称与 CUDA 在所有输入上完全等价，也未完成性能优化**。

最新[差异审计](docs/DIFFERENCE_ANALYSIS.md)直接执行原 CUDA 数学体：整图对 Metal 的 RMSE 为 9.37e-6，仍有少量阈值像素超限；另修复了退化投影与 PLY 四元数边界行为。当前 17 项原生测试、40 项 Python 测试通过。

兼容目标是提交 [59f5f77e](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d) 对应的本地源码，实际文件哈希见 [docs/upstream.json](docs/upstream.json)。上游后续新增的 depth/antialiasing 接口不属于这个固定版本。

详细说明从 [docs 文档导航](docs/README.md)开始：包括[架构](docs/ARCHITECTURE.md)、[Python API](docs/PYTHON_API.md)、[C++ API](docs/CPP_API.md)、[数据格式](docs/DATA_FORMATS.md)和[开发排错](docs/DEVELOPMENT.md)。下面先介绍可直接运行的使用步骤。

## 使用步骤：先在当前 Mac 上跑起来

本工程提供 **3DGS 渲染与梯度计算库**。推荐先渲染已有模型，再验证反向传播，最后接入自己的训练程序。Python 负责组织数据和调用 autograd，底层计算仍由 C++ / Objective-C++ + Metal 完成。

当前这台 Mac 已有 `.venv`、编译好的扩展和官方 train 示例模型，可直接运行以下步骤。其他机器或新检出的工程需要先按后文“原 Python 接口 / 可微训练”安装环境，再按“官方真实场景”下载数据。

### 1. 进入工程并渲染示例

```sh
cd /Users/liqing93/code/diff-gaussian-rasterization-metal
.venv/bin/python tools/render_public_sample.py --width 980
open output/public/train/metal_00001_980.png
```

其他位置的工程请替换 `cd` 路径；本文后续命令均在工程根目录执行，无需先激活虚拟环境。

脚本读取 `output/public/train/point_cloud.ply` 和 `cameras.json`，用 Metal 渲染默认相机，输出 980×545 的火车场景。PNG 旁边的同名 JSON 记录图像尺寸、可见 Gaussian 数量、运行时间和可用的参考图对比结果。

### 2. 切换视角或调整分辨率

```sh
.venv/bin/python tools/render_public_sample.py --camera 1 --width 980
```

`--camera` 是 `cameras.json` 数组中从 0 开始的索引，并非图片文件名。输出文件名为 `metal_<img_name>_<实际宽度>.png`，具体路径会打印到终端。`--width` 指定最大宽度，按原始相机比例缩放，不放大超过原始尺寸。

### 3. 验证反向传播

```sh
.venv/bin/python tools/render_public_sample.py --width 980 --backward
```

脚本将输入设为可求梯度，执行渲染，以图像各通道数值的平方均值作为示例损失，再调用 `backward()`，检查梯度是否存在且数值有限，并记录非零元素数。**这一步是反向传播验证，不会更新模型参数，也不是完整训练。**

### 4. 使用自己的模型

准备一个目录，放入相互匹配的 `point_cloud.ply` 和 `cameras.json`，然后指定该目录。例如，已将文件放入 `output/my-scene/` 后运行：

```sh
.venv/bin/python tools/render_public_sample.py --directory output/my-scene --camera 0 --width 980
```

当前示例读取器面向原版 3DGS 的二进制小端 PLY，需要位置、SH 颜色系数、opacity、scale 和 rotation 等字段，不能直接读取只有 XYZ/RGB 的普通点云。相机 JSON 是数组，每个相机包含 `img_name`、`width`、`height`、`fx`、`fy`、`position` 和 `rotation`；其中位置和旋转描述 camera-to-world，脚本内部转换为 view 矩阵。当前脚本使用 3 阶 SH 和黑色背景，其他设置可参照下面的调用模板调整。

完整模板见 [tools/render_public_sample.py](tools/render_public_sample.py)：`load_ply()` 负责读取和激活模型参数，`settings_for()` 构造相机，`main()` 演示渲染、图片导出和反向传播。

如果目标是“从一组照片训练出自己的 3DGS 场景”，还需要上层训练程序提供相机数据、损失函数、优化器以及 Gaussian 的增密与裁剪。本仓库负责其中的可微光栅化环节；上层程序中的 CUDA 调用也需要适配到 MPS。

## 原生 C++ 构建

需要 Apple Silicon Mac、完整 Xcode、CMake 3.24+。原生库使用 C++17；本机验证 Apple M3 Pro、Xcode 26.0.1。CMake 最低部署目标为 macOS 14，但未在最低版本系统实测。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
./build/rasterizer-demo fixtures/overlap.json output/overlap.ppm
```

纯 C++ 公共头文件不依赖 Objective-C 或 PyTorch。`MetalRasterizer::render(Scene)` 是预计算 RGB/covariance 的便捷 Forward 入口，返回持有 GPU texture 的 `MetalFrame`；完整可微接口见下面的 Python 包。JSON 解析只属于 example 层。

```cpp
#include <dgr/metal_rasterizer.h>

dgr::Scene scene;
scene.camera = dgr::Camera::perspective(64, 64);
dgr::Gaussian gaussian;
gaussian.mean = {0, 0, 2};
gaussian.color = {1, 0, 0};
scene.gaussians.push_back(gaussian);
dgr::MetalRasterizer renderer;
auto frame = renderer.render(scene);
auto rgb = frame.read_rgb(); // 显式回读图像。
```

安装和下游 C++ 消费：

```sh
cmake --install build --prefix /tmp/dgr-install
cmake -S tests/consumer -B output/consumer-build -DCMAKE_PREFIX_PATH=/tmp/dgr-install
cmake --build output/consumer-build
./output/consumer-build/consumer
```

下游通过 `find_package(DGR CONFIG REQUIRED)` 和 `target_link_libraries(app PRIVATE dgr::metal)` 接入。非 Apple 平台可用 `-DDGR_BUILD_METAL=OFF` 构建 CPU reference。

## 原 Python 接口 / 可微训练

首次安装时执行以下命令；已经配置好的当前 Mac 可直接使用前面的示例。与原生构建一样，需要完整 Xcode 和可在 PATH 中找到的 CMake 3.24+。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install torch numpy pytest pillow setuptools
.venv/bin/python -m pip install -e . --no-build-isolation --no-deps
.venv/bin/python -m pytest tests/python -q
```

需要 `cmake` 在 PATH 中。本机测试 Python 3.14.2、PyTorch 2.14.0；可选 Torch 扩展使用 C++20。依赖声明允许 Torch 2.5+，但其内部 MPS storage/stream 接口尚未逐版本验证，其他版本需重新编译并执行测试。

下面是接入方式示意，`settings` 和模型 Tensor 的准备过程可参考上面的完整模板。需要优化的参数必须启用 `requires_grad`；`target` 应为与输出同尺寸的 `[3, H, W]` 图像 Tensor，并使用 `float32` 和 `mps`。

```python
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# settings 字段与原版相同；所有非空浮点 Tensor 使用 float32、device="mps"。
rasterizer = GaussianRasterizer(raster_settings=settings)
color, radii = rasterizer(
    means3D=means3D, means2D=means2D, opacities=opacities,
    shs=shs, scales=scales, rotations=rotations,
)
loss = (color - target).square().mean()
loss.backward()
```

颜色输入选择 `shs` 或 `colors_precomp`；形状输入选择 `scales + rotations` 或 `cov3D_precomp`。背景与相机 settings 不求梯度。`means2D` 保留原版收集屏幕位置梯度的约定，不参与 Forward 位置输入。调用方需将原训练工程中的 `.cuda()`、CUDA event 等调用迁移到 MPS；本包不负责上层完整训练工程的设备适配。

## 官方真实场景

[官方 gaussian-splatting 仓库](https://github.com/graphdeco-inria/gaussian-splatting) 提供预训练模型、相机和评估图片。本工具通过 HTTP Range 提取单个 train 场景，无需下载整个 14 GB 压缩包，并校验 ZIP CRC、记录 SHA-256。

```sh
.venv/bin/python tools/fetch_public_sample.py --images
.venv/bin/python tools/render_public_sample.py --width 980 --backward
.venv/bin/python tools/check_public_pixels.py
```

本机已运行官方 train 7000-step 模型的 559,263 个 Gaussian：980×545 Forward/Backward 成功，64 个固定像素与独立 CPU double 的最大误差为 2.95e-6。记录见 [docs/public-sample.json](docs/public-sample.json)。

文件位于 `output/public/train/`。公开评估图片与预训练模型不能假定来自完全相同的 checkpoint/代码版本；缩放也引入滤波差异。工具报告图像误差，不把它当作逐内核或梯度 CUDA golden。下载的模型和图片不纳入源码仓库。

## 修改代码后的构建与验证

修改 `shaders/` 或 `bindings/torch/` 后，重新编译 Python 扩展并运行 Python 测试：

```sh
.venv/bin/python -m pip install -e . --no-build-isolation --no-deps
.venv/bin/python -m pytest tests/python -q
```

修改 C++ 核心、原生 Metal 桥接或共享 shader 后，构建原生目标并运行 CTest。共享 shader 同时影响两个入口，因此也要执行上面的 Python 扩展重编译与测试：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

Python 扩展与原生库使用各自的构建产物，仅构建 `build/` 中的默认原生目标不会更新已安装的 `_C` 扩展。单纯修改 Python 包装层时，editable 安装会直接使用源码，但仍应运行相关 Python 测试。

## 代码导航

| 路径 | 用途 |
| --- | --- |
| `include/dgr/`、`src/core/` | C++ 数据契约、校验、独立 double reference |
| `src/metal/metal_rasterizer.mm` | 原生 GPU 资源、调度、frame 生命周期 |
| `shaders/rasterizer.metal` | 预处理、scan、分桶、排序、ranges、Forward |
| `shaders/training.metal` | SH、covariance、CHW Forward、解析 Backward、可见性 |
| `bindings/torch/` | C++/Objective-C++ MPS Tensor 互操作与扩展入口 |
| `python/diff_gaussian_rasterization/` | 保持原 API 的 autograd wrapper |
| `tests/`、`tests/python/` | 原生 GPU、CPU 对照、梯度与优化测试 |
| [docs/MIGRATION.md](docs/MIGRATION.md) | 兼容语义、资源与架构约定、未完成工作 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 已执行的测试及证据边界 |

当前使用多 pass scan 和 bitonic sort，并同步等待 GPU、每次分配资源；未据此宣称 C++ 比 Swift 更快，也未宣称接近 CUDA 吞吐。

## 来源与许可证

数学与接口改写自 [Graphdeco 原仓库](https://github.com/graphdeco-inria/diff-gaussian-rasterization)，保留版权和 [LICENSE.md](LICENSE.md)。原许可适用于非商业研究和评估，本工程没有另行授予 MIT/Apache 许可。原参考仓库及其已有未提交修改保持不动。
