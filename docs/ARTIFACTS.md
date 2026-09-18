# 渲染图、参照与验证产物

[返回文档导航](README.md)

以下路径均相对 **Metal 工程根目录**。2026-09-18 已核对本机文件；`output/` 被 Git 忽略，因此新克隆仓库不会自带这些图片、大模型或跨后端参照。随仓库分发的是工具和 `docs/` 下的汇总证据。

## 迁移后的 Metal 图

本次迁移验证的根目录是 `output/migration-20260917/real-scene/`，使用保存的同一份输入 snapshot 渲染。各目录内的 `render.png` 是对应 `image.npy` 的 8-bit 显示副本。

| 视角 / 输入模式 | PNG 路径 |
| --- | --- |
| 主视角 00001，980×545，SH+scale | `output/migration-20260917/real-scene/00001_980/render.png` |
| 主视角 00001，1959×1090，SH+scale | `output/migration-20260917/real-scene/00001_1959/render.png` |
| 视角 00072，980×545，SH+scale | `output/migration-20260917/real-scene/00072_980/render.png` |
| 视角 00187，980×545，SH+scale | `output/migration-20260917/real-scene/00187_980/render.png` |
| 主视角 SH+covariance | `output/migration-20260917/real-scene/variants/sh_cov/00001_980/render.png` |
| 主视角 RGB+scale | `output/migration-20260917/real-scene/variants/rgb_scale/00001_980/render.png` |
| 主视角 RGB+covariance | `output/migration-20260917/real-scene/variants/rgb_cov/00001_980/render.png` |

在当前 Mac 打开主视角与官方公开参照：

```sh
open output/migration-20260917/real-scene/00001_980/render.png
open output/public/train/renders.png
```

每个案例目录还包含 Float32 `[3,H,W]` 的 `image.npy`、语义解码状态 `state.npz`、`gradients_dense.npz`、`gradients_sparse.npz` 和 `report.json`。七组汇总为 `output/migration-20260917/real-scene/summary.json`；整合性能和重复实验后的可提交报告为 [migration-results.json](analysis/tile-2026-09-17/migration-results.json)。

## “CUDA 对比图”是哪一张

| 参照 | 本机路径 | 能证明什么 |
| --- | --- | --- |
| 官方公开 CUDA 渲染图 | `output/public/train/renders.png` | 官方发布的视觉参照，未确认与下载模型属于同一 checkpoint/代码版本 |
| 对应真实照片 | `output/public/train/gt.png` | 图像重建质量参照；不是光栅化后端输出 |
| 同输入 ROCm GPU 图 | `../diff-gaussian-rasterization-rocm/output/full-scene-20260907/00001_980/rocm/render.ppm` | 2026-09-07 实测的配对后端输出；迁移验证重用其浮点数据 |
| 同输入 CUDA 数学体 CPU 图 | `../diff-gaussian-rasterization-rocm/output/full-scene-20260907/00001_980/cpu/render.ppm` | CPU Host Float32 oracle，未在 NVIDIA GPU 执行 |
| 迁移前 Metal 图 | `../diff-gaussian-rasterization-rocm/output/full-scene-20260907/00001_980/metal/render.ppm` | 旧 Metal 基线；七组新旧浮点图逐元素一致 |

ROCm/CPU/旧 Metal 各目录的 `image.npy` 才是严格数值检查使用的浮点图；PPM 仅供查看。**当前没有固定 CUDA 版本、同一输入的 NVIDIA GPU 实测图与梯度基准。** 不要将 CPU oracle 或 ROCm 图标记成 NVIDIA 实测。

官方图的来源记录为 `output/public/train/image_provenance.json`：评估包中的成员 `train/test/ours_7000/renders/00001.png`，SHA-256 为 `151a9e03bb8c95175db1598c5c0c899ca72355f32c2097cea425b29d202f60a7`。下载脚本保存 URL、ZIP 成员、字节数和哈希，可用于确认图像来源；历史差异解释见 [DIFFERENCE_ANALYSIS.md](DIFFERENCE_ANALYSIS.md)。

## 重新生成与导出

日常渲染工具从 PLY 和相机重新准备输入，会直接生成 PNG 和同名 JSON：

```sh
.venv/bin/python tools/fetch_public_sample.py --images
.venv/bin/python tools/render_public_sample.py --width 980 --backward
open output/public/train/metal_00001_980.png
```

该输出是本次运行的示例图；七组迁移实验使用归档 Tensor snapshot，不能把重新读取 PLY 的渲染自动当成相同输入的复验。完整复验步骤与资产要求见[迁移报告](TILE_MIGRATION_REPORT.md#复现)。

`validate_tile_migration.py` 输出 `.npy`、`.npz` 和 `.json`，**不会自动生成 PNG**。上述已保存的 `render.png` 是额外导出的；如果换到新的 `--output` 目录，可从其中一个案例的浮点图导出：

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

source = Path('output/migration-20260917/real-scene/00001_980/image.npy')
color = np.load(source)
assert color.ndim == 3 and color.shape[0] == 3
assert np.isfinite(color).all()
pixels = np.round(np.clip(color.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
Image.fromarray(pixels).save(source.with_name('render.png'))
PY
```

将 `source` 改为需要导出的路径。PNG 的 clamp、8-bit 量化会丢失数值信息；图像误差、阈值像素与梯度验收仍使用浮点数组和报告，不根据肉眼或 PNG 判断通过。
