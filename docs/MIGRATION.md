# CUDA → Metal 迁移契约

## 目标与架构

完整迁移固定版本的可微光栅化功能，只替换 GPU 后端。用户已明确长期使用 C++ + Metal；Swift Package、Swift 源码和测试已移除。

参考本地 `../../diff-gaussian-rasterization/learning-docs/用3DGS训练顶尖工程师思维.md` 的第六步/答案六，以及副本 19.2、21.4、21.5。手机 viewer 方向不缩减本工程的训练目标。

```text
Python 原 API → custom autograd → C++ extension / MPS Tensor
                                         ↓
                                Objective-C++ Metal 调度
                                         ↓
                      MSL Forward + saved state + analytic Backward

C++ Scene 便捷入口 → Objective-C++ Metal 调度 → 同一组 Forward 内核
C++ / Python CPU reference → 独立正确性对照
```

Objective-C++ 是私有 Apple API 实现层，公共原生接口是标准 C++。独立库为 C++17，Torch 扩展为 C++20。MSL 由 CMake 嵌入库，在设备上编译，关闭 fast math；安装后无需原源码路径。

## 固定上游与接口

基准提交 `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`；实际使用本地 working tree 的文件哈希见 `upstream.json`。兼容 `GaussianRasterizationSettings` 原 12 个字段，`GaussianRasterizer.forward`、`markVisible`、`rasterize_gaussians` 及三个 `_C` 入口的参数顺序。

图像为 MPS float32 `[3,H,W]`，半径为 MPS int32 `[P]`，可见性为 MPS bool `[P]`。几何、分桶、图像缓存仍以 byte Tensor 保存，但 **Metal 缓存不与 CUDA 二进制布局互通**，只能交给同后端 backward。

| 能力 | 当前实现 |
| --- | --- |
| 颜色 | 预计算 RGB 或 SH degree 0–3，保留负值 clamp 和梯度掩码 |
| Covariance | 上三角 6 元素，或 scale/rotation 计算 |
| Forward | 投影、tile 覆盖、scan、重复实例、稳定排序、alpha 合成 |
| Backward | RGB、opacity、屏幕中心/conic、三维中心/covariance、SH、scale、rotation |
| Autograd | 每次 Forward 保存独立状态，可重复 Forward 后反传 |
| 可见性 | 原版近裁剪语义 |
| prefiltered | 为 true 时触发 near 剔除则报错；正常输入继续 |
| debug | 异常时保存原版参数快照 |
| settings 梯度 | 与原版一致，不提供相机/背景等 settings 梯度 |
| depth / antialiasing | 固定版本没有这些字段，不属于本次目标 |

## 数值与分支语义

| 上游位置 | Metal 行为 |
| --- | --- |
| in_frustum | view Z <= float32 的 0.2 时剔除，不额外进行横向 clip-space 裁剪 |
| ndc2Pix | 保留半像素偏移 |
| computeCov2D | FOV clamp 系数 1.3；二维协方差对角线加 0.3 |
| computeCov3D | 保留乘法顺序，内核不归一化 quaternion |
| det == 0 | 跳过当前 Gaussian，不使整个渲染失败；有限负 determinant 保持原 CUDA 控制流 |
| 半径 | 判别量下限 0.1、3 倍标准差、ceil |
| getRect | 浮点转整数向零截断；右/下边界排他 |
| 排序 | tile、正深度 Float32 bits；同深度按输入 ID 稳定排序 |
| ranges | 半开区间，空 tile 为 [0,0] |
| alpha | 上限 0.99；小于 1/255 跳过；下一透射率小于 0.0001 时停止且不累计当前候选 |
| n_contrib | 最后接受候选的一基位置，包括前面跳过的候选 |
| 空场景 | P=0 黑图；P>0 全部剔除时为背景，匹配原 Python bridge |
| mean2D 梯度 | 原归一化视口尺度，第三维为零；Forward 忽略 dummy 输入 |
| alpha cap 梯度 | 即使 alpha 被限幅，仍沿 opacity × Gaussian 权重反传 |
| scale modifier 梯度 | 保留原 CUDA 对 scale 梯度缺少末端 modifier 因子的约定 |
| cov2D 梯度 | 保留分母平方加 1e-7 等原版稳定化细节 |

这些梯度约定并不处处等于 Forward 的有限差分，因此分别通过源码公式、可微 CPU reference 和避开分支边界的有限差分验证。不能以“数学上更合理”为由改变原版行为。

## Tensor 与生命周期

所有非空训练输入 Tensor 必须在 MPS 上且为 float32。支持非连续输入，必要时在 GPU 上转连续；不满足 16 字节对齐的连续切片另行 clone，避免 float4 绑定错位。空可选 Tensor 可以保留原 wrapper 的 CPU sentinel。

访问 MPS buffer 前同步当前 PyTorch MPS stream，然后通过自己的串行 Metal queue 调度，等待完成后返回。图像、Gaussian 与梯度保留在 MPS buffer；Forward 回读错误码和实例总数两个标量。debug 快照和显式导出会回读数据。

使用 Torch 内部 storage/stream API；本机验证 Torch 2.14.0，其他版本需重新编译测试。原生 renderer 实例需串行使用，可移动不可复制。MetalFrame 可复制共享资源，renderer 销毁后 frame 仍有效；借用 texture 不能释放或修改。保留多个 frame 或 autograd graph 会累积内存。

## 原生 JSON 便捷入口

`fixtures/overlap.json` 使用 schemaVersion 1，仅表达固定 RGB/预计算 covariance 路径，不是完整训练资产格式或 GPU 内存镜像。

| 字段 | 契约 |
| --- | --- |
| mean | 世界空间中心，3 个有限数 |
| covariance | 世界空间上三角 xx、xy、xz、yy、yz、zz，必须正定 |
| color | 线性 RGB，不自动 clamp，不做 SH 烘焙 |
| opacity | 已激活值，范围 0–1 |
| viewMatrix | 按列存储 world → view，正 Z 为前方 |
| projectionMatrix | 按列存储，包含 view 的 world → clip |
| tanFovX/Y | 半视场角正切；正且有限；与矩阵一致由调用方保证 |
| texture | RGBA32Float；RGB 合成背景，alpha 恒为 1 |

内部每 Gaussian 为 13 个 Float32，投影状态跨度 64 字节并有静态断言；训练缓存另行对齐，都不是公开 ABI。

原生入口默认限制：10 万 Gaussian、1,048,576 个 tile 实例、4,194,304 个像素、单次 Metal 分配预算 256 MiB，可通过 RenderLimits 调整。这不是进程峰值预算，不包含 CPU 数组、驱动或历史 frame。

训练入口不沿用 demo 的数量/预算限制；宽高各最多 8192，实例最多 0x3fffffff，实际受 GPU 内存限制。非法或过大的投影会拒绝，原生入口另检查有限值、正定性和 opacity；畸形输入的错误行为不模仿 CUDA 未定义行为。

## 后续验收

| 项目 | 当前状态 | 后续工作 |
| --- | --- | --- |
| C++ 架构 | CMake、公共头、私有 ObjC++ 桥、安装导出完成 | 更多工具链验证 |
| Forward / Backward / API | 所列路径已实现并通过小型 GPU 对照 | 扩大随机和阈值边界覆盖 |
| 官方数据 | 定点下载、PLY/相机加载、渲染和指标工具 | 精确确认 checkpoint 与图片对应 |
| CUDA golden | 尚无同输入逐阶段/梯度输出 | 获取可追溯 golden，在本机持续对照 |
| scan | 多 pass Hillis–Steele，饱和计数 | 分块 scan |
| sort | 补齐至 2 的幂的 bitonic，稳定 ID | radix sort、scratch 复用 |
| render | 每像素顺序读取候选 | threadgroup memory 批量加载 |
| Backward 累加 | uint CAS 的 Float32 原子累加 | 归约减少竞争，验证误差 |
| 调度 | 同步等待、每次独立分配 | 资源池、异步 stream、峰值内存统计 |
| 上层训练 | 20 步 Adam 小型优化测试 | 完整 3DGS 训练与 densification 适配 |

只有这台 Mac 不阻止移植和本地验证。公开图像可以验证真实场景，但不能证明每项反向梯度正确。跨 GPU 浮点及原子累加顺序不保证逐位一致。功能覆盖与 CUDA 数值等价分别验收。
