# 原生 C++ 接入

[返回文档导航](README.md)

原生库不依赖 Python、PyTorch 或 Swift。公共接口是标准 C++17，Apple API 由内部 Objective-C++ 实现。当前 `Scene` 接口提供固定 RGB / 预计算 covariance 的 Forward；SH、scale/rotation 和完整 Backward 的使用入口见 [Python API](PYTHON_API.md)。

## 构建与安装

在工程根目录执行：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
cmake --install build --prefix "$PWD/output/install"
```

安装产物包含头文件、静态库、DGR CMake 配置和许可证。`dgr::metal` 传递链接 `dgr::core` 及 Apple framework，无需下游手工列出这些依赖。

在自己的应用目录创建 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.24)
project(MyDGRApp LANGUAGES CXX)
find_package(DGR CONFIG REQUIRED)
add_executable(my-renderer main.cpp)
target_link_libraries(my-renderer PRIVATE dgr::metal)
```

同目录的 `main.cpp`：

```cpp
#include <dgr/metal_rasterizer.h>
#include <iostream>

int main() {
    try {
        dgr::Scene scene;
        scene.camera = dgr::Camera::perspective(64, 64);
        dgr::Gaussian gaussian;
        gaussian.mean = {0.1f, 0.0f, 2.0f};
        gaussian.covariance = {0.04f, 0, 0, 0.0225f, 0, 0.01f};
        gaussian.color = {0.9f, 0.2f, 0.1f};
        gaussian.opacity = 0.7f;
        scene.gaussians.push_back(gaussian);

        dgr::MetalRasterizer renderer;
        auto frame = renderer.render(scene);
        auto rgb = frame.read_rgb();
        if (rgb.size() != 64 * 64 * 3) return 2;
        std::cout << renderer.device_name() << '\n'
                  << frame.width() << "x" << frame.height() << '\n'
                  << "tile instances: " << frame.stats().instances << '\n';
        return 0;
    } catch (const dgr::Error &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

从该应用目录配置构建，替换安装前缀为实际路径：

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/Users/liqing93/code/diff-gaussian-rasterization-metal/output/install
cmake --build build -j 2
./build/my-renderer
```

仓库也有独立的[消费方示例](../tests/consumer/)，可验证安装后的 `find_package` 和链接。

## Scene 与相机

类型定义见 [scene.h](../include/dgr/scene.h)。`Scene` 包含 schema version、一个相机和 Gaussian 数组；默认 schema 为 1。原生 Gaussian 的颜色、不透明度和协方差都必须已经准备好，渲染器不读取 PLY 或执行模型参数激活。

`Camera::perspective(width, height, tan_fov_x, tan_fov_y, background)` 创建原点相机，默认两个 FOV 正切为 1、背景为黑色。若修改 `view_matrix`，必须同时更新包含 view 的 `projection_matrix`。矩阵、协方差顺序与 JSON 格式见 [DATA_FORMATS.md](DATA_FORMATS.md)。

`render()` 先校验数据，再提交 Metal 工作，并等待完成后返回。数值必须有限，opacity 在 0–1，三维 covariance 必须正定；这里的校验比为兼容原版而保留的 Tensor 入口更严格。相机矩阵与 FOV 是否相互一致由调用方保证。

## MetalFrame 与回读

| 方法 | 结果与开销 |
| --- | --- |
| `width()` / `height()` | 图像尺寸 |
| `read_rgb()` | 显式回读为 `std::vector<float>`，逐像素 RGB 交错排列，共 width × height × 3 个值 |
| `native_texture_handle()` | 借用的 `id<MTLTexture>`，以 `void*` 穿过公共 C++ 接口 |
| `debug_snapshot()` | 回读投影、offsets、排序、ranges、透射率和贡献位置，用于诊断 |
| `stats()` | 实例数、请求的 Metal 分配字节数、GPU command 时间 |

图像 texture 为 `RGBA32Float`，RGB 已合成背景，alpha 恒为 1；alpha 通道不是 Gaussian 透明度或剩余透射率。`read_rgb()` 不做 gamma、clamp 或 8-bit 编码。`debug_snapshot()` 的 `last_contributors` 是最后接受候选的一基位置，包含前面被跳过候选的位置，不是接受数量。

Objective-C++ 调用方可以在持有 frame 时桥接 texture：

```objective-c++
id<MTLTexture> texture = (__bridge id<MTLTexture>)frame.native_texture_handle();
```

这是借用句柄，不应 release 或修改 texture；异步消费者完成前，应继续持有一份 `MetalFrame`。renderer 可移动不可复制，同一实例串行使用。frame 可复制并共享 GPU 资源，renderer 销毁后 frame 仍有效；不要继续使用已经移走内容的对象。

## 资源限制与错误

`MetalRasterizer(RenderLimits)` 可配置每次渲染限制，默认值如下。

| 字段 | 默认值 |
| --- | ---: |
| `max_gaussians` | 100,000 |
| `max_instances` | 1,048,576 |
| `max_pixels` | 4,194,304 |
| `max_working_bytes` | 256 MiB |

三个数量上限必须为正且不超过 16,777,216；内存预算必须为正。宽高各不超过 8192，同时仍受像素数限制。预算统计本次请求的 Metal 分配，不包括 CPU 数组、驱动开销和此前保留的 frame，也不能代替进程峰值内存统计。Python 入口不沿用这些默认数量限制。

异常为 `dgr::Error`，可通过 `code()` 分类：

| ErrorCode | 常见含义 |
| --- | --- |
| `invalid_input` | schema、形状、数值、协方差、投影错误或使用已移走内容的对象 |
| `unavailable` | 缺少所需的 Apple Silicon Metal 设备 |
| `resource_limit` | 数量、预算或设备分配上限 |
| `gpu` | shader/pipeline 编译、command 创建或执行失败 |

`stats().gpu_seconds` 仅统计本次 command 执行时间，不含初始化、CPU 打包和显式回读。不要将它与端到端时间混为一谈。性能测量方法见[开发文档](DEVELOPMENT.md)。
