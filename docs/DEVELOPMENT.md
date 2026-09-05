# 开发、验证与排错

[返回文档导航](README.md)

## 工作目录和环境

本页命令均从工程根目录执行。保留相邻 `../diff-gaussian-rasterization` 参考仓库及用户已有修改，新实现只在本工程修改。固定参考与实际文件哈希见 [upstream.json](upstream.json)。

先确认当前使用的环境和包路径：

```sh
xcode-select -p
cmake --version
.venv/bin/python - <<'PY'
import sys
import torch
import diff_gaussian_rasterization
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("MPS:", torch.backends.mps.is_available())
print("Package:", diff_gaussian_rasterization.__file__)
PY
```

原生构建要求 Apple Silicon、完整 Xcode 和 CMake 3.24+。本机验证过的组合见 [VALIDATION.md](VALIDATION.md)，不把依赖声明的最低版本等同于已经逐版本验证。Torch 绑定访问内部 MPS storage/stream API，升级 Python 或 Torch 后要重新编译扩展并验证。

## 按改动选择构建和检查

| 改动 | 必要操作 |
| --- | --- |
| `src/core/`、公共头文件 | 原生构建和相关 CTest；公共 API 改动另验证下游消费方 |
| `src/metal/` | 原生构建和 Metal CTest |
| `shaders/` | 原生构建/CTest，以及 Torch 扩展重编译/Python 测试 |
| `bindings/torch/` | 重编译 `_C`，运行 Python 测试 |
| Python wrapper | editable 安装直接生效，运行相关 Python 测试 |
| 数据读取或相机工具 | 对应工具回归，并检查真实输入的转换和输出 |
| 文档 | 核对源码、链接、命令参数和可运行示例；没有代码变化时无需重复全套 GPU 测试 |

原生路径：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

Torch 路径：

```sh
.venv/bin/python -m pip install -e . --no-build-isolation --no-deps
.venv/bin/python -m pytest tests/python -q
```

两条路径的编译产物独立。默认原生配置 `DGR_BUILD_TORCH=OFF`，因此只执行 `cmake --build build` 不会更新 pip 安装的 `_C`。Shader 经 CMake 嵌入，修改 `.metal` 文件也必须重编译；已运行的 Python 进程需要退出后重新启动，才能加载新的扩展和 context。

独立消费方检查：

```sh
cmake --install build --prefix "$PWD/output/install"
cmake -S tests/consumer -B output/consumer-build -DCMAKE_PREFIX_PATH="$PWD/output/install"
cmake --build output/consumer-build -j 2
./output/consumer-build/consumer
```

## CMake 选项

| 选项 | 默认值 / 用途 |
| --- | --- |
| `DGR_BUILD_METAL` | Apple 上默认 ON；OFF 时只构建 CPU core/reference |
| `DGR_BUILD_EXAMPLES` | ON，构建原生 JSON/PPM demo |
| `DGR_BUILD_TORCH` | OFF，启用时要求 Metal；pip 构建自动启用 |
| `BUILD_TESTING` | CTest 默认 ON，pip 构建设为 OFF |
| `Python3_EXECUTABLE` | 显式选定 Torch 扩展使用的 Python |
| `DGR_PYTHON_OUTPUT_DIR` | `_C` 产物目录，pip 构建自动设置 |
| `CMAKE_OSX_DEPLOYMENT_TARGET` | 未指定时设为 14.0；最低系统版本尚未实测 |

仅验证 CPU reference 可使用独立目录，避免覆盖 Metal 配置：

```sh
cmake -S . -B build-cpu -DDGR_BUILD_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 2
ctest --test-dir build-cpu --output-on-failure
```

## 测试分别证明什么

| 测试层 | 主要覆盖 | 边界 |
| --- | --- | --- |
| 原生 CTest | 场景校验、投影、排序、合成、资源和 frame 生命周期 | 原生 RGB/covariance 入口 |
| `test_compatibility.py` | 四类输入模式、SH、梯度、有限差分、优化、Tensor 布局和状态 | 小场景，独立 CPU 数学参考 |
| `test_upstream_source.py` | 抽取固定 CUDA 数学体后的 Float32 Forward/Backward 对照 | CPU 执行，不是 CUDA GPU |
| `test_public_sample.py` | PLY 激活边界回归 | 输入工具，不证明整个渲染器 |
| 公开场景审计 | 大模型、多阶段浮点和图像差异 | 参考图版本及数值阈值仍需区别分析 |

原源码 oracle 需要相邻参考仓库和 `third_party/glm/glm/glm.hpp`。缺少 GLM 时对应测试会跳过；MPS 不可用时 GPU Python 测试也会跳过。查看最终 skip 记录，不能把“无失败”直接写成“GPU 全部通过”。

已记录的通过数量和容差以 [VALIDATION.md](VALIDATION.md) 为准。修复逻辑错误应加入能够触发原问题的回归；不要为了通过测试而放宽阈值，或把旧版特殊梯度改成另一套数学行为。

## 真实场景与深入审计

有模型和公开图片后可以执行：

```sh
.venv/bin/python tools/render_public_sample.py --width 980 --backward
.venv/bin/python tools/check_public_pixels.py
.venv/bin/python tools/audit_public_difference.py
.venv/bin/python tools/audit_asset_inputs.py
```

若数据缺失，先按[工程 README](../README.md)下载。`--backward` 是示例损失下的梯度检查，不更新参数；`check_public_pixels.py` 只抽样像素。完整差异脚本构建原源码 Host oracle，保存 `output/analysis/images.npz` 和 JSON，再执行排序、输入路径、图像和稀疏像素损失的梯度对照。

输入审计脚本使用固定 `output/public/train` 和 `output/analysis/images.npz`，所以应在默认路径的完整差异审计之后运行。若给完整审计指定其他 `--output`，输入审计不会自动改用该路径。完整审计会处理全模型、多次渲染并保存浮点图，需要比小测试更多时间和内存。

## 常见问题

| 症状 / 错误 | 定位与处理 |
| --- | --- |
| 导入失败、找不到 `_C` 或出现 CUDA 包路径 | 用环境检查命令确认解释器和包来源，再用该解释器执行 editable 编译安装 |
| Torch 升级后编译失败或符号缺失 | 核对实际 Python/Torch 和 CMake 配置，重新构建；内部 MPS API 的跨版本兼容尚未逐一验证 |
| `must be an MPS float32 tensor` | 检查实际使用的模型、背景、view、projection、campos 的 device/dtype；可选模式用 `None` |
| `invalid shape` / `opacity size mismatch` | 按 [Python API](PYTHON_API.md) 检查点数和维度，opacity 使用 `[P,1]` |
| `Point filtered although prefiltered is set` | 输入包含 view Z 不大于 0.2 的点；检查坐标和 prefiltered 的实际含义 |
| `invalid projected covariance/coordinates` | 检查相机、数据有限性、scale 激活、协方差顺序与极端投影；零 determinant 已定义为单点跳过 |
| 黑图或只有背景 | 检查空模型、相机方向、组合投影、near 剔除、opacity；`markVisible` 不能完整判断离屏 |
| 图像方向、位置或比例错误 | 检查 world/camera 坐标、矩阵转置、proj 是否包含 view、FOV 和分辨率是否对应 |
| 天空等区域与官方图不同 | 先验证相机/模型来源，再做同输入原源码对照；公开图差异本身不能直接判定 Metal 错误 |
| `.grad` 为 None | 确认启用梯度、没有在 `no_grad()` 中渲染；中间激活 Tensor 要用 `retain_grad()` 观察，优化器应持有原始参数 |
| tile 数量超限或内存分配失败 | 检查大 scale、异常相机或大量覆盖实例；区分原生限制与训练入口，定位后再决定减少尺寸/规模或调整原生预算 |
| 修改 shader 后结果没变 | 确认重编译的是实际使用的产物，并重启 Python 进程 |

定位时先保存模型/相机来源、参数、环境、错误文本和最小复现。可使用 `debug=True` 的异常参数快照或原生 `debug_snapshot()` 分阶段比较；两者用途不同，见对应 API 文档。

## 性能与结果记录

先运行预热，再重复相同规模和输入，分别记录端到端耗时、纯 GPU command 时间、显式回读时间、点数、tile 实例数和内存。MPS 外围计时应在测量边界同步；当前绑定自身也同步等待，记录结果时应说明这一点。

单次冷启动、不同分辨率或包含不同回读工作量的时间不能直接比较。当前 scan/sort 和分配方案尚待优化，没有经过生产吞吐或长期运行验收。修改时保留小场景正确性与真实模型诊断两类证据，后续计划见 [MIGRATION.md](MIGRATION.md)。
