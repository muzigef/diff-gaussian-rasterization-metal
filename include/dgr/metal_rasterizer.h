#pragma once
#include "dgr/scene.h"
#include <memory>

namespace dgr {

struct FrameStats {
    std::size_t instances = 0;
    std::size_t allocated_bytes = 0;
    double gpu_seconds = 0; // Command execution only; excludes initialization/CPU/readback.
};

class MetalRasterizer;

// Copying a frame shares ownership. GPU resources survive the renderer that created them.
class MetalFrame {
  public:
    ~MetalFrame();
    MetalFrame(const MetalFrame &);
    MetalFrame &operator=(const MetalFrame &);
    MetalFrame(MetalFrame &&) noexcept;
    MetalFrame &operator=(MetalFrame &&) noexcept;

    int width() const;
    int height() const;
    FrameStats stats() const;
    std::vector<float> read_rgb() const; // Explicit image readback for export/tests.
    DebugSnapshot debug_snapshot() const;

    // Borrowed id<MTLTexture>, RGBA32Float, linear RGB with background, alpha=1.
    // Valid while any copy of this frame lives. Do not release or mutate it.
    // For Objective-C++ clients: (__bridge id<MTLTexture>)frame.native_texture_handle().
    void *native_texture_handle() const;

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit MetalFrame(std::shared_ptr<Impl> impl);
    const Impl &checked() const;
    friend class MetalRasterizer;
};

// Standard C++ public API: no Objective-C, Foundation, simd, or Swift headers.
// One instance must be used serially. Moved-from objects throw invalid_input on use.
class MetalRasterizer {
  public:
    explicit MetalRasterizer(RenderLimits limits = {});
    ~MetalRasterizer();
    MetalRasterizer(MetalRasterizer &&) noexcept;
    MetalRasterizer &operator=(MetalRasterizer &&) noexcept;
    MetalRasterizer(const MetalRasterizer &) = delete;
    MetalRasterizer &operator=(const MetalRasterizer &) = delete;

    std::string device_name() const;
    MetalFrame render(const Scene &scene); // Native Forward convenience API; autograd uses the Torch binding.

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    const Impl &checked() const;
};

} // namespace dgr
