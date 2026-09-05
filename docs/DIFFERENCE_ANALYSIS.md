# 公开图像差异与逻辑兼容性审计

日期：2026-09-05；设备：Apple M3 Pro；固定光栅化源码提交：59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d。

## 判断

本次定位并修复了两处边界行为错误：Metal 将零行列式升级为整次渲染异常；PLY 工具的四元数归一化缺少上游 epsilon。这两处都没有影响当前 train 样本的图像。

公开 render 的主要差异高度指向**预训练模型、训练/渲染版本与论文评估图片不构成严格成对参照**。这是结合官方说明和对照实验的推断，不是已经拿到原论文 checkpoint 后完成的证明。没有发现足以解释整图 RMSE 0.06355 的 SH 布局、相机方向、排序算法或梯度翻译错误。

同时，移植还存在可测量的浮点与阈值分叉：整图有 172 个颜色通道超过原定 2e-4 绝对误差门槛。不能因为平均误差小而宣布数值等价通过。

## 比较方法：直接执行原源码数学

上一轮仅有独立 double reference 的 64 个像素，可能漏掉稀少离群点，也无法完全排除两份手写实现共有的错误。本次新增 `tests/upstream_oracle/`：

1. 从本地原 CUDA 的 forward.cu、backward.cu 直接抽取每 Gaussian 数学函数和逐像素合成循环体。
2. 保留原 GLM、常数、公式及判断条件，将 CUDA 线程调度和共享内存读取替换为 CPU 调度及同索引直接读取。
3. 以 Float32 编译执行，完整渲染 980×545；另用同一份 Metal saved state 比较原源码反向公式。
4. 用独立稳定排序检查 Metal 产生的全部 2,949,986 个 tile 实例。

原 CUDA 仓库未改动。测试保存源码 SHA-256、生成的 C++ 文件与编译命令。

**这个 oracle 是原源码在 CPU 上的执行，不是 NVIDIA GPU 执行。** CPU 编译禁用 fast math 和乘加收缩，GPU 指令、数学函数及原子累加顺序仍可能不同，因此报告不将它称作 CUDA golden。

## 整图差异的量级

同一 PLY、同一相机、同一分辨率，559,263 个 Gaussian，图像共 1,602,300 个颜色通道：

| 比较 | RMSE | 最大绝对误差 | 超过 2e-4 的通道数 |
| --- | ---: | ---: | ---: |
| Metal vs 原源码 Host 完整流水线 | 0.00000937461 | 0.00261343 | 172 |
| Metal vs 原源码 Host，仅比较像素合成 | 0.000000862245 | 0.000991911 | 2 |
| Metal vs 公开 render | 0.06355186 | 0.7590125 | 1,594,242 |
| 原源码 Host vs 公开 render | 0.06355184 | 0.7590105 | 1,594,245 |

RMSE 是把各通道误差平方、取平均、再开平方，表示整体误差大小；越小越接近。上表最有力的证据是：**换成原源码 Host 执行后，对公开图片的误差几乎没有变化**。Metal 对原源码的整体误差约比对公开图的误差小 6,779 倍。这个比值比较误差量级，不等于对各原因作了可相加的百分比归因。

官方明确区分“论文评估图片”和“发布代码重新生成的预训练模型”，并说明发布代码清理/修复后评估指标会不同。[官方 Evaluation 说明](https://github.com/graphdeco-inria/gaussian-splatting#evaluation)

当前 Metal 对 GT 的 RMSE 为 0.08829261，公开 render 对 GT 为 0.08961516。二者质量处于相近量级，但“谁更接近 GT”不能用来证明后端等价。

## 输入和显示因素的排查

| 假设 | 检查结果 | 判断 |
| --- | --- | --- |
| SH 系数排列错误 | 对照上游 PLY 的 [P,3,15] → [P,15,3] 约定，26,844,624 个系数逐元素一致 | 未发现布局错误 |
| scale / opacity / rotation 激活错误 | 与 Torch exp、sigmoid、normalize 对照，输入最大差异均为 1.19e-7；整图差异 RMSE 9.96e-7 | 不是主要差异来源 |
| view/projection 转置或相机方向错误 | 由 JSON 的 camera-to-world 求逆；与上游 Float32 矩阵构造比较，combined projection 最大差异 1.19e-7；相机位置最大差异 4.77e-7 | 未发现方向或矩阵次序错误 |
| 半像素/主点错位 | 对 x/y 各测试 -1、-0.5、0、0.5、1 像素，共 25 组 | 最优仍为原来的零偏移 |
| 小幅 FOV / 尺寸换算错误 | tan(FOV/2) 比例 0.999、0.9995、1、1.0005、1.001 | 最优为 1 |
| SH 阶数不对 | 0、1、2、3 阶对公开图 RMSE 为 0.07074、0.06879、0.06457、0.06355 | 3 阶最接近 |
| 应先全分辨率渲染再缩小 | 1959×1090 渲染后 area/bilinear/bicubic 缩到 980×545 | 最小 RMSE 约 0.06402，没有消除差异 |
| Gamma 处理遗漏 | gamma 2.2 / 1/2.2 后 RMSE 为 0.21079 / 0.22844；拟合最优 gamma=1.025 时仍为 0.06348 | 不支持标准 Gamma 错误假设 |
| 全局亮度/颜色偏移 | 每通道拟合 gain+bias 后 RMSE 0.06160 | 只能解释一小部分 |
| PNG 量化 | 当前 float → 8 bit → float 的 RMSE 0.001132，最大差异 0.001961 | 远小于公开图差异 |
| fast math 没有关闭 | 本机读取属性确认 fastMathEnabled=NO 已对应 Safe + Precise；显式重复设置后整图逐元素不变 | 排除该配置问题 |

对应上游输入语义可核查 [GaussianModel 的 PLY/激活实现](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py)、[相机 JSON](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/camera_utils.py) 和 [投影矩阵](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/graphics_utils.py)。这些主分支文件用于核查输入约定；本次光栅化数学始终固定为本地 2023 版本。

公开 ZIP 的文件名为 00001、00009 等原始照片名；本次使用 camera img_name=00001。它与当前 render.py 的枚举输出命名不同，已根据实际包内目录对齐，未将第二个相机误当第一个。

## 少量像素为何差得更明显

半径、tile 矩形、每点实例数全部一致。SH 颜色最大误差 3.58e-7，三维 covariance 最大误差 8.94e-8；主要连续误差发生在投影、二维 covariance 求逆和数学函数执行。

这些很小的变化遇到硬阈值时会产生离散变化。以最大误差像素 (491,483)、Gaussian ID 422740 为例：

| 量 | 原源码 Host 几何 | Metal 几何 + 相同 Host 像素公式 |
| --- | ---: | ---: |
| alpha | 0.003921566531 | 0.003921632189 |
| 判断结果 | 跳过 | 累计 |

判断门槛为：

$$
\alpha_{\min}=\frac{1}{255}\approx 0.003921568859
$$

读作“最小 alpha 等于一除以二百五十五”；分子是 1，分母是 255。alpha 表示当前 Gaussian 在这个像素处的合成权重，由 opacity 与空间衰减共同决定。代码 `alpha < 1.0f / 255.0f` 会跳过较小值，两边虽然只差约 0.000000066，却落在门槛两侧，因此颜色会增加或减少一次完整贡献。

追踪的最大 12 个误差像素全部出现上述 alpha 分支差异。这一追踪在两侧使用同一个 Host 像素公式，隔离了预处理结果的影响；不是直接读取 GPU exp 的内部值。只隔离像素合成时仍有 2 个通道超限，浮点函数/算术的阈值敏感性尚需 NVIDIA golden 进一步鉴定。

排序也有类似现象：使用各自计算的 Float32 深度时，2,949,986 个实例中有 428 个列表位置不同；**将同一份 Metal 深度交给独立稳定排序，ID 与 ranges 均完全一致**。排序器没有表现出稳定性错误，差异来自极近深度的数值次序。该因素的整图影响 RMSE 为 2.15e-6、最大差异 0.0006604。

不能通过扩大门槛或改变 alpha cutoff 来掩盖这些差异。对位级一致性的后续工作需要匹配 CUDA 的编译选项与设备数值行为。

## 已修复的逻辑问题

### 1. 投影退化时错误地终止整次渲染

二维 covariance 是记录高斯在屏幕上铺开方向和范围的 2×2 数字表，也称矩阵。其行列式为：

$$
d=ac-b^2
$$

读作“a 乘 c，减去 b 的平方”。a、c 是矩阵对角线元素，b 是对称的非对角元素；代码对应 `xx`、`yy`、`xy`，d 对应 `det`。后续求逆需要除以 d；当它为零时，该 Gaussian 不能按当前公式求逆。Float32 对非常大且很薄的分布也可能把它舍入到零。

原 CUDA 的 `det == 0` 分支直接返回，保留该点 radius=0。旧 Metal 把所有 `det <= 0` 当作 error=1，C++ 随后抛异常，连其他有效 Gaussian 也无法渲染。

修复后恢复原版单点剔除，并取消额外的有限负 determinant 拒绝条件；仍保留非有限值和资源限制检查。实际 GPU 的精确二次幂输入复现：

| 版本 | error | radius | tile count |
| --- | ---: | ---: | ---: |
| 修复前 shader | 1 | 0 | 0 |
| 修复后 shader | 0 | 0 | 0 |

回归测试还验证：同一批次中的正常点继续渲染，被跳过点的全部梯度为零。有限负 determinant 另与原源码分支对照；原生 C++ Scene 入口仍要求物理有效的正定 covariance，不改变这层输入契约。

### 2. PLY 四元数归一化缺少 epsilon

上游模型层采用：

$$
q_{\mathrm{out}}=\frac{q}{\max(\lVert q\rVert,10^{-12})}
$$

读作“把 q 除以 q 的长度与十的负十二次方中较大的数”。分子 q 是表示旋转的四个数，分母是至少为 1e-12 的长度。这样零向量不会除以零，极小向量也遵守上游的缩放约定。旧 PLY 工具直接除以长度，零四元数会变成 NaN，极小四元数也会得到不同结果。

现已在工具层补齐这一约定，零和 1e-14 四元数回归测试由失败变为通过。**未在 Metal 内核内新增归一化**，内核仍保留原 CUDA 对传入 quaternion 的处理方式。

当前官方模型的原始四元数最小长度约 0.3978，未触发这个问题。退化处理修复前后，当前 train 场景图像逐元素相同；两项修复都不是公开图像误差的归因解释。

## Backward 与验证范围

新增 16 组原源码 Forward/Backward 测试，覆盖 SH 0–3、RGB/SH 与 covariance/scale-rotation 四种组合、多点随机输入和非单位 scale modifier。使用相同 saved state，原源码的 8 类梯度输出均在既定门槛内。

真实场景另用 36 个固定像素的带正负权重 loss，检查全部模型参数位置上的梯度。最大绝对差异：

| 梯度 | 最大绝对差异 |
| --- | ---: |
| means2D | 2.38e-7 |
| 预计算 RGB | 5.70e-9 |
| opacity | 8.38e-9 |
| means3D | 2.32e-6 |
| covariance | 1.06e-4 |
| SH | 2.79e-9 |
| scales | 4.28e-6 |
| rotations | 6.88e-7 |

这些测试检验代码翻译与上游梯度约定，包括 alpha cap 的替代梯度和 scale modifier 的原版约定，不宣称上游所有梯度都等同于处处可微的解析数学。真实场景检查使用稀疏像素 loss，未穷尽每个输出像素的所有偏导。

最终本机验证：**17 项 CTest + 40 项 pytest 全部通过**。全图误差门槛超限仍单独记录，未纳入“完全等价通过”的声明。

## 复现与证据

```sh
cmake --build build -j 2
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest tests/python -q
.venv/bin/python tools/audit_public_difference.py
.venv/bin/python tools/audit_asset_inputs.py
```

原源码 oracle 需要相邻的 diff-gaussian-rasterization checkout 及其 GLM 子模块。本机无需 CUDA 或 NVIDIA GPU。

汇总数据见 [difference-audit.json](difference-audit.json)。原始输出位于 `output/analysis/after_fix/`，其中包含整图 Float32 数组、逐阶段误差、离群点 trace、生成 C++ 与源文件哈希。输入实验位于 `output/analysis/asset_input_audit.json`。

后续取得同一 checkpoint、相机和输入的 CUDA GPU 输出后，应优先对照这里已经识别出的阈值像素、近等深度排序和高条件数 covariance，再扩展到端到端训练。当前证据支持“主路径翻译正确、两处边界错误已修复、公开样本未严格配对”，不支持“所有输入完全等价”。
