# Python / PyTorch MPS 接口

[返回文档导航](README.md)

## 安装和导入

先按[工程 README](../README.md)安装本机扩展，使用工程的 `.venv/bin/python`。发行包名是 `diff-gaussian-rasterization-metal`，导入名保持原版的 `diff_gaussian_rasterization`，避免在同一环境混装 CUDA 包。

```python
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
```

此入口面向 MPS float32，一次调用处理一个相机和一组 Gaussian，没有 batch 维。所有实际使用的浮点 Tensor 都应在 `mps` 上；未使用的可选参数传 `None`，由 wrapper 处理空 sentinel。非连续 Tensor 可用，但可能产生 GPU 拷贝。

## 相机与渲染设置

`GaussianRasterizationSettings` 是保留原 12 个字段的 NamedTuple。Tensor 字段建议使用下列形状，绑定对相机 Tensor 检查的是元素数。相机和背景等 settings 不提供梯度。

| 字段 | 类型 / 形状 | 含义 |
| --- | --- | --- |
| `image_height` / `image_width` | int | 各在 1–8192 内；实际尺寸仍受内存限制 |
| `tanfovx` / `tanfovy` | 正的有限 float | 半水平/垂直视场角的正切，不是角度值 |
| `bg` | float32 `[3]` | 背景 RGB |
| `scale_modifier` | 有限 float | scale 路径的全局缩放；通常为 1，预计算 covariance 应由调用方准备好 |
| `viewmatrix` | float32 `[4,4]` | world-to-view，按原版列存储约定传入 |
| `projmatrix` | float32 `[4,4]` | **已包含 view 的** world-to-clip |
| `sh_degree` | int 0–3 | 实际求值的 SH 阶数；RGB 路径也要求合法值 |
| `campos` | float32 `[3]` | 相机在世界坐标中的位置，用于 SH 视角方向 |
| `prefiltered` | bool | 为 true 时，出现 near 剔除的点会报错；不是抗锯齿开关 |
| `debug` | bool | Forward/Backward 出错时保存 CPU 参数快照 |

矩阵构造、正 Z 约定和 PLY 相机转换见[数据格式](DATA_FORMATS.md)。已有上层相机若已经转置为原版格式，不要再转置一次。

## 模型输入与返回值

```python
color, radii = rasterizer(
    means3D=means3D,
    means2D=means2D,
    opacities=opacities,
    shs=shs,                       # 或 colors_precomp
    scales=scales,                 # 或 cov3D_precomp
    rotations=rotations,
)
```

这段是参数示意；可直接运行的完整例子见下一节。下表的 `P` 表示 Gaussian 数量，`M` 表示每点 SH 系数数量。

| 参数 | 形状 | 数据含义 |
| --- | --- | --- |
| `means3D` | `[P,3]` | 世界坐标中心 |
| `means2D` | `[P,3]` | dummy 输入，通常全零；只用于接收屏幕位置梯度 |
| `opacities` | `[P,1]` | sigmoid 之后的不透明度；为保证 Backward 形状匹配，使用二维列 Tensor |
| `shs` | `[P,M,3]` | SH 系数；0/1/2/3 阶分别至少需要 1/4/9/16 项 |
| `colors_precomp` | `[P,3]` | 已准备好的 RGB |
| `scales` | `[P,3]` | exp 之后的轴向标准差，不是方差或 log-scale |
| `rotations` | `[P,4]` | 四元数，顺序 w、x、y、z；内核不自动归一化 |
| `cov3D_precomp` | `[P,6]` | 世界空间协方差，顺序 xx、xy、xz、yy、yz、zz |

`shs` 与 `colors_precomp` 必须且只能提供一种。`scales + rotations` 与 `cov3D_precomp` 也必须且只能提供一种，不要同时传入两种模式的数据。

返回的 `color` 是 MPS float32 `[3,H,W]`，可求梯度；`radii` 是 MPS int32 `[P]`，表示像素半径，不提供可优化的连续梯度。输出不自动限制在 0–1；clamp 和图像编码由导出端处理。

`means2D` 的数值不改变 Forward；它的梯度保留原版视口尺度约定，第三维为零。不要把它当作覆盖三维投影结果的二维中心输入，也不要用对它做有限差分来验证该替代梯度。

## 无需下载模型的最小 Forward / Backward

将下面内容保存为工程根目录的 `minimal_mps.py`，运行 `.venv/bin/python minimal_mps.py`。它只打印结果，不写模型或图片。

```python
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

assert torch.backends.mps.is_available(), "需要可用的 MPS 设备"

def tensor(value, grad=False):
    return torch.tensor(value, device="mps", dtype=torch.float32, requires_grad=grad)

view = torch.eye(4, device="mps", dtype=torch.float32)
# 按普通行列方式构造投影，再转为原版要求的存储顺序。
projection = tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
])
settings = GaussianRasterizationSettings(
    image_height=64, image_width=64,
    tanfovx=1.0, tanfovy=1.0,
    bg=tensor([0.03, 0.04, 0.08]), scale_modifier=1.0,
    viewmatrix=view.T.contiguous(),
    projmatrix=(projection @ view).T.contiguous(),
    sh_degree=0, campos=tensor([0, 0, 0]),
    prefiltered=False, debug=False,
)
data = {
    "means3D": tensor([[0.1, 0.0, 2.0]], grad=True),
    "means2D": tensor([[0.0, 0.0, 0.0]], grad=True),
    "opacities": tensor([[0.7]], grad=True),
    "colors_precomp": tensor([[0.9, 0.2, 0.1]], grad=True),
    "scales": tensor([[0.2, 0.15, 0.1]], grad=True),
    "rotations": tensor([[1.0, 0.0, 0.0, 0.0]], grad=True),
}
rasterizer = GaussianRasterizer(settings)
color, radii = rasterizer(**data)
loss = color.square().mean()  # 仅演示反向传播，不是重建任务的完整损失。
loss.backward()
assert color.shape == (3, 64, 64)
assert torch.isfinite(color).all().item()
assert radii[0].item() > 0
print("image:", tuple(color.shape), "radii:", radii.cpu().tolist())
for name, value in data.items():
    assert value.grad is not None and torch.isfinite(value.grad).all().item(), name
    print(name, "max |gradient|:", value.grad.abs().max().item())
print("near-visible:", rasterizer.markVisible(data["means3D"]).cpu().tolist())
```

示例相机位于原点、朝正 Z，投影采用与原生 `Camera::perspective` 一致的简化设置。真实相机请使用完整相机数据。某些参数在这个对称小场景中梯度为零是可能的，不应把“所有梯度都非零”作为接口要求。

## 接入训练时的约定

需要优化的叶子 Tensor 或 `torch.nn.Parameter` 启用 `requires_grad=True`。若通过 exp、sigmoid 或 normalize 得到输入，梯度继续传回对应原始参数；中间 Tensor 的 `.grad` 默认不保留，需要检查它时调用 `retain_grad()`。`means2D` 通常单独建立为可求梯度的全零 Tensor。

训练循环由调用方执行：清空梯度、激活模型参数、渲染、与目标图计算损失、`backward()`、优化器 `step()`。目标图应使用 `[3,H,W]`、MPS float32，与输出采用一致的颜色处理。原版特殊梯度与 Forward 有限差分并非处处相等，见[迁移契约](MIGRATION.md)。

本接口提供一阶解析 Backward，未提供可用的二阶 autograd 链路；不要把 `create_graph=True` 当作二阶支持。相机、背景、半径、离散排序和增密决策也不在本接口求梯度范围内。

## 可见性、空输入与调试

`rasterizer.markVisible(positions)` 返回 MPS bool `[P]`。固定版本实际上只使用 view Z 的 near 判定：Z 大于 0.2 才通过，离屏点也可能返回 true。因此它不是“会影响当前图像”的完整可见性掩码；`radii > 0` 可反映通过投影并覆盖 tile 的点，但也不保证最终对像素有贡献。

空模型 `P=0` 返回黑图；非空但全部被剔除时返回背景色。`prefiltered=True` 不会跳过校验：若点触发 near 剔除，会抛异常。

`debug=True` 的异常快照写入当前工作目录的 `snapshot_fw.dump` 或 `snapshot_bw.dump`，可能覆盖同名文件并包含完整输入。正常渲染不会因此生成快照，也不会启用 GPU capture。

底层 `_C` 接口供 wrapper 使用。Forward 返回实例数、color、radii、geometry、binning、image；Backward 返回 means2D、RGB、opacity、means3D、covariance、SH、scale、rotation 的梯度。其确切签名见 [api.h](../bindings/torch/api.h)，状态 buffer 不应手工构造或跨后端复用。
