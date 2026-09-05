# 模型、相机与输出格式

[返回文档导航](README.md)

## 先区分两条数据路径

| 路径 | 文件 | 用途 |
| --- | --- | --- |
| 原生 CLI 示例 | 一个 schemaVersion 1 JSON | 小场景，固定 RGB 和预计算 covariance |
| Python 真实场景工具 | `point_cloud.ply` + `cameras.json` | 加载原版 3DGS 模型，转换后调用 MPS 接口 |

这些文件都不是 shader 内存的直接镜像。当前没有从原生 JSON 自动导出训练 checkpoint，C++ renderer 本身也不读取 PLY。

## 原生 JSON 场景

下面是可运行的最小场景；保存为 `output/custom.json` 后，在工程根目录执行 `./build/rasterizer-demo output/custom.json output/custom.ppm`。目录和原生程序需事先准备好。

```json
{
  "schemaVersion": 1,
  "camera": {
    "width": 64,
    "height": 64,
    "viewMatrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
    "projectionMatrix": [1,0,0,0, 0,1,0,0, 0,0,1,1, 0,0,0,0],
    "tanFovX": 1,
    "tanFovY": 1,
    "background": [0.03,0.04,0.08]
  },
  "gaussians": [
    {
      "mean": [0.1,0,2],
      "covariance": [0.04,0,0,0.0225,0,0.01],
      "color": [0.9,0.2,0.1],
      "opacity": 0.7
    }
  ]
}
```

示例中的字段必须存在，数组长度必须匹配，数值有限；布尔值不能冒充数字。`schemaVersion` 当前只能为 1。读取器会执行原生场景校验和默认资源限制。完整的双 Gaussian 例子见 [overlap.json](../fixtures/overlap.json)，解析实现见 [scene_io.mm](../examples/scene_io.mm)。

`covariance` 按 xx、xy、xz、yy、yz、zz 存储对称三维协方差。协方差描述 Gaussian 在不同方向的空间扩散；对角项是方差，也就是标准差的平方。因此 scale 为 0.2 时，对应轴的方差为 0.04。非对角项表达方向之间的关联，不应直接丢弃。

## 相机和矩阵约定

世界坐标转到相机空间后，正 Z 指向前方。固定版本的 near 判定使用 view Z 大于 0.2；公开样例投影矩阵构造中的 near/far 数值不替代这条剔除规则。

两条接口都要求组合投影已经包含 view：

$$
p_{\mathrm{clip}} = P\,V\,p_{\mathrm{world}}
$$

读作“裁剪空间位置等于投影矩阵乘视图矩阵，再乘世界空间位置”。这里位置向量是按顺序排列的四个数，最后一项补 1；矩阵是描述坐标变换的数值表。计算从右向左：先由视图矩阵 $V$ 转到相机空间，再由投影矩阵 $P$ 转到裁剪空间。代码中的 `viewmatrix` 对应 $V$，`projmatrix` 对应已经相乘的 $PV$。

原生 `Matrix4` 和 JSON 的 16 元素矩阵按列存储。Python 若先按普通数学行列构造矩阵，应传入 `view.T.contiguous()` 与 `(projection @ view).T.contiguous()`；这是调整内存顺序，不是交换坐标变换的先后。沿用原版已转置 Tensor 时不要重复转置。

## Python 工具的 cameras.json

文件根节点为数组，下面的例子包含一个相机：

```json
[
  {
    "img_name": "custom_00000",
    "width": 640,
    "height": 480,
    "fx": 320.0,
    "fy": 240.0,
    "position": [0,0,0],
    "rotation": [[1,0,0],[0,1,0],[0,0,1]]
  }
]
```

这里的 `rotation` 是普通三行三列数组，与世界坐标位置 `position` 一起描述 camera-to-world。它不是前面原生 JSON 的 16 元素列存储矩阵；`settings_for()` 将其求逆得到 view。模型和相机必须采用同一个世界坐标系。

`fx`、`fy` 是以原始图片像素为单位的焦距。工具据此计算 FOV：

$$
\tan(\mathrm{FOV}_x/2)=\frac{W}{2f_x},
\qquad
\tan(\mathrm{FOV}_y/2)=\frac{H}{2f_y}
$$

左边读作“半视场角的正切”。右边横线上的分子 $W$ 或 $H$ 是原始图片宽或高；横线下的分母是对应焦距的两倍。用分子除以分母得到设置中的 `tanfovx`、`tanfovy`。工具缩小输出尺寸时保持这两个值，所以视场保持不变。

当前 helper 假定主点居中，无镜头畸变字段；`--width` 按比例缩小尺寸，不裁剪画面。`--camera` 选择数组索引，输出名使用被选相机的 `img_name`。当前公开包的图片名对应原始照片名，不应按当前上游渲染脚本的枚举命名猜测对应关系。

## 3DGS PLY 参数

[load_ply()](../tools/render_public_sample.py) 支持 `binary_little_endian`，vertex 属性使用标量 `float`。普通 XYZ 点云、ASCII PLY、列表属性和其他字段类型不属于这个读取器的支持范围。

| PLY 字段 | 转为接口输入时的处理 |
| --- | --- |
| `x`、`y`、`z` | 组合为 `[P,3]` 世界坐标 |
| `f_dc_0` 到 `f_dc_2` | 组合为 `[P,1,3]` 的 0 阶 SH 系数 |
| `f_rest_*` | 按数字后缀排序，先按 RGB 通道分组，再转为 `[P,M-1,3]`，与 DC 拼接 |
| `opacity` | 原始 logit 经 sigmoid 变为 `[P,1]` 不透明度 |
| `scale_0` 到 `scale_2` | 原始 log-scale 经 exp 变为 `[P,3]` 标准差 |
| `rot_0` 到 `rot_3` | w、x、y、z，按向量长度归一化，分母下限为 `1e-12` |

三阶 SH 模型有 3 个 DC 字段和 45 个 `f_rest_*` 字段，最终为 `[P,16,3]`。SH 系数不是最终 RGB，不能对系数直接做显示颜色 clamp。运行时根据相机方向求值，加原版颜色偏移并保留下限截断语义。

激活只应做一次。直接调用 Python rasterizer 时要传入已激活的 scale/opacity；MSL 不会替你执行 exp/sigmoid，也不会归一化 quaternion。相反，`load_ply()` 已经做了这些转换，调用方不应重复处理。训练时通常在 PyTorch 中从原始可优化参数计算激活，让梯度继续传回原始参数。

当前公共样例脚本固定 3 阶 SH、黑色背景和 scale modifier 1。低阶模型或其他设置需要调整 helper/调用方；库本身支持的模式比这个文件工具更广。工具只加载模型，不保存优化器状态或写回训练 checkpoint。

## 输出、参考图和来源记录

| 文件或对象 | 含义 |
| --- | --- |
| 原生 `read_rgb()` | Float32、逐像素 RGB 交错排列，不做显示变换 |
| Python `color` | MPS Float32 `[3,H,W]`，通道在前 |
| `.ppm` / `metal_<img_name>_<width>.png` | 导出时 clamp 到 0–1，再编码为 8-bit RGB，不额外做 gamma 变换 |
| 同名渲染 `.json` | 渲染配置、时间、可见数量、可选图像误差及梯度检查 |
| `renders.png` | 下载的官方公开渲染图，不是本机运行 CUDA 得到的图 |
| `gt.png` | 官方对应视角的真实照片 |
| `provenance.json` / `image_provenance.json` | 下载 URL、ZIP 成员路径、字节数和 SHA-256 |
| 审计 `images.npz` | 保留 Float32 的 Metal、原源码 Host、公开图和 GT 数组 |

8-bit 导出会丢失小数精度，逐阶段审计应使用浮点数组。官方模型与公开图不能仅凭场景名和迭代标签就认定为完全配对；原因和实际误差见[差异分析](DIFFERENCE_ANALYSIS.md)。

模型、图像、下载分段缓存和审计大文件保存在 `output/` 下并被 Git 忽略。可复现证据需要保留来源记录、相机索引、设置、模型哈希和代码版本；不要以内部 geometry/binning/image byte buffer 代替模型数据。
