#include "../../src/metal/primitives.h"
#include "api.h"
#include "shader_source.h"
#include <ATen/mps/MPSStream.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace dgr::torch_binding {
namespace {
using View = metal_detail::BufferView;
using metal_detail::sub;
struct alignas(16) Params {
    std::array<uint32_t, 4> dimensions;
    std::array<uint32_t, 4> modes;
    std::array<float, 4> optics;
};
static_assert(sizeof(Params) == 48);
size_t align16(size_t x) {
    return (x + 15) & ~size_t(15);
}
struct Geometry {
    size_t packed = 0, projected, clamped, cov, camera, size;
    explicit Geometry(size_t p)
        : projected(align16(p * 52)), clamped(projected + p * 64), cov(align16(clamped + p * 3)),
          camera(align16(cov + p * 24)), size(camera + 45 * 4) {}
};
struct Image {
    size_t ranges = 0, transmittance, contributors, size;
    Image(size_t w, size_t h)
        : transmittance(align16(((w + 15) / 16) * ((h + 15) / 16) * 8)),
          contributors(transmittance + w * h * 4), size(contributors + w * h * 4) {}
};
struct Context {
    id<MTLDevice> device;
    id<MTLBuffer> dummy;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    id<MTLLibrary> library;
    std::mutex mutex;
    std::unordered_map<at::mps::MPSStream *, metal_detail::Scratch> scratch;
    struct AsyncErrors {
        std::mutex mutex;
        std::string message;
    };
    std::shared_ptr<AsyncErrors> errors = std::make_shared<AsyncErrors>();
    void check_errors() {
        std::lock_guard<std::mutex> lock(errors->mutex);
        TORCH_CHECK(errors->message.empty(), "Metal asynchronous execution: ", errors->message);
    }
    Context() {
        device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device && device.hasUnifiedMemory, "Apple Silicon Metal device required");
        dummy = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        TORCH_CHECK(dummy, "Metal allocation failed");
        std::memset(dummy.contents, 0, 16);
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        options.languageVersion = MTLLanguageVersion3_1;
        NSError *error = nil;
        library = [device newLibraryWithSource:[NSString stringWithUTF8String:detail::shader_source]
                                       options:options
                                         error:&error];
        TORCH_CHECK(library, "Metal shader compilation failed: ", error.localizedDescription.UTF8String);
        pipelines = [NSMutableDictionary new];
    }
    template <class A>
    void encode(NSString *name, id<MTLCommandBuffer> command, std::initializer_list<View> buffers,
                const A &args, size_t threads) {
        if (!threads)
            return;
        id<MTLComputePipelineState> pipeline = pipelines[name];
        if (!pipeline) {
            NSError *error = nil;
            id<MTLFunction> f = [library newFunctionWithName:name];
            TORCH_CHECK(f, "Missing Metal kernel");
            pipeline = [device newComputePipelineStateWithFunction:f error:&error];
            TORCH_CHECK(pipeline, "Metal pipeline: ", error.localizedDescription.UTF8String);
            TORCH_CHECK(pipeline.maxTotalThreadsPerThreadgroup >= 256 && pipeline.threadExecutionWidth >= 8 &&
                            256 % pipeline.threadExecutionWidth == 0,
                        "Metal kernel requires a 256-thread group");
            pipelines[name] = pipeline;
        }
        id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
        TORCH_CHECK(e, "Metal encoder allocation failed");
        e.label = name;
        [e setComputePipelineState:pipeline];
        NSUInteger i = 0;
        for (auto v : buffers)
            [e setBuffer:v.buffer offset:v.offset atIndex:i++];
        [e setBytes:&args length:sizeof(args) atIndex:i];
        metal_detail::dispatch(e, pipeline, name, &args, threads);
        [e endEncoding];
    }
    template <class Encode> void execute(bool wait, Encode &&encode) {
        check_errors();
        auto *stream = at::mps::getCurrentMPSStream();
        // Use PyTorch's own stream: its allocator permits in-flight storage reuse
        // only in that stream's order. The command buffer retains Metal resources.
        // Do not destroy Python-owned Tensor objects in a Metal completion handler:
        // their final decref can need the GIL while a Python caller waits for the GPU.
        auto status = errors;
        at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
          stream->endKernelCoalescing();
          id<MTLCommandBuffer> command = stream->commandBuffer();
          encode(command);
          [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status == MTLCommandBufferStatusError) {
                std::lock_guard<std::mutex> lock(status->mutex);
                status->message = completed.error.localizedDescription.UTF8String ?: "unknown GPU error";
            }
          }];
          stream->synchronize(wait ? at::mps::SyncType::COMMIT_AND_WAIT : at::mps::SyncType::COMMIT);
          if (wait)
              TORCH_CHECK(command.status == MTLCommandBufferStatusCompleted,
                          "Metal execution: ", command.error.localizedDescription.UTF8String);
        });
        check_errors();
    }
};
Context &context() {
    static Context value;
    return value;
}
void check_float(const Tensor &t, const char *name) {
    TORCH_CHECK(t.device().is_mps() && t.scalar_type() == at::kFloat, name, " must be an MPS float32 tensor");
}
void check_size(const Tensor &t, std::initializer_list<int64_t> shape, const char *name) {
    TORCH_CHECK(t.sizes() == at::IntArrayRef(shape), name, " has invalid shape: ", t.sizes());
}
struct Inputs {
    std::vector<Tensor> held;
    View get(const Tensor &t, bool optional = false) {
        if (optional && t.numel() == 0)
            return {context().dummy, 0};
        TORCH_CHECK(t.device().is_mps(), "all nonempty tensors must be on MPS");
        held.push_back(t.contiguous());
        if ((held.back().storage_offset() * held.back().element_size()) % 16)
            held.back() = held.back().clone();
        const auto &c = held.back();
        return {(__bridge id<MTLBuffer>)c.storage().data(),
                static_cast<NSUInteger>(c.storage_offset() * c.element_size())};
    }
};
Tensor bytes(size_t count, const Tensor &like) {
    return at::zeros({static_cast<int64_t>(count)}, like.options().dtype(at::kByte));
}
Params params(const Tensor &means, int w, int h, const Tensor &sh, int degree, const Tensor &colors,
              const Tensor &scales, double tx, double ty, double modifier, bool prefiltered) {
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means3D must have dimensions (num_points,3)");
    check_float(means, "means3D");
    TORCH_CHECK(w > 0 && h > 0 && w <= 8192 && h <= 8192, "image dimensions must be in 1...8192");
    TORCH_CHECK(means.size(0) <= std::numeric_limits<int>::max(), "too many Gaussians");
    TORCH_CHECK(std::isfinite(tx) && std::isfinite(ty) && tx > 0 && ty > 0 && std::isfinite(modifier),
                "invalid FOV/scale modifier");
    TORCH_CHECK(degree >= 0 && degree <= 3, "supported SH degree is 0...3");
    const bool use_sh = colors.numel() == 0, use_scales = scales.numel() != 0;
    const auto count = means.size(0);
    uint32_t m = 0;
    if (count) {
        if (use_sh) {
            check_float(sh, "sh");
            TORCH_CHECK(sh.dim() == 3 && sh.size(0) == count && sh.size(2) == 3 &&
                            sh.size(1) >= (degree + 1) * (degree + 1),
                        "SH shape must be [P,M,3] with sufficient coefficients");
            m = sh.size(1);
        } else {
            check_float(colors, "colors");
            check_size(colors, {count, 3}, "colors");
        }
        if (use_scales) {
            check_float(scales, "scales");
            check_size(scales, {count, 3}, "scales");
        }
    }
    return {{uint32_t(count), uint32_t(w), uint32_t(h), m},
            {uint32_t(degree), uint32_t(use_sh), uint32_t(use_scales), uint32_t(prefiltered)},
            {float(tx), float(ty), float(modifier), 0}};
}
void check_common(const Tensor &bg, const Tensor &view, const Tensor &proj, const Tensor &campos) {
    for (const auto &t : {bg, view, proj, campos})
        check_float(t, "camera/background");
    TORCH_CHECK(bg.numel() == 3 && view.numel() == 16 && proj.numel() == 16 && campos.numel() == 3,
                "invalid camera tensor sizes");
}
void check_geometry_inputs(const Params &p, const Tensor &rotations, const Tensor &cov) {
    if (p.modes[2]) {
        check_float(rotations, "rotations");
        check_size(rotations, {p.dimensions[0], 4}, "rotations");
    } else {
        check_float(cov, "covariance");
        check_size(cov, {p.dimensions[0], 6}, "covariance");
    }
}
} // namespace

ForwardResult forward(const Tensor &bg, const Tensor &means, const Tensor &colors, const Tensor &opacity,
                      const Tensor &scales, const Tensor &rotations, double modifier, const Tensor &cov,
                      const Tensor &view, const Tensor &proj, double tanx, double tany, int height, int width,
                      const Tensor &sh, int degree, const Tensor &campos, bool prefiltered, bool debug) {
    @autoreleasepool {
        auto p = params(means, width, height, sh, degree, colors, scales, tanx, tany, modifier, prefiltered);
        const uint32_t count = p.dimensions[0];
        auto output = at::zeros({3, height, width}, means.options());
        auto radii = at::zeros({count}, means.options().dtype(at::kInt));
        if (!count)
            return {0,
                    output,
                    radii,
                    bytes(0, means),
                    bytes(0, means),
                    bytes(0, means)}; // Upstream returns black for P=0.
        check_common(bg, view, proj, campos);
        check_geometry_inputs(p, rotations, cov);
        check_float(opacity, "opacity");
        TORCH_CHECK(opacity.numel() == count, "opacity size mismatch");
        auto &ctx = context();
        std::lock_guard<std::mutex> lock(ctx.mutex);
        Geometry gl(count);
        Image il(width, height);
        auto geometry = bytes(gl.size, means), image = bytes(il.size, means);
        auto offsets = at::zeros({count}, means.options().dtype(at::kInt));
        auto &scratch = ctx.scratch[at::mps::getCurrentMPSStream()];
        scratch.reset();
        struct Trim {
            metal_detail::Scratch &s;
            ~Trim() {
                s.trim();
            }
        } trim{scratch};
        auto temporary = [&](size_t bytes) -> View {
            TORCH_CHECK(bytes <= ctx.device.maxBufferLength, "scratch buffer exceeds device limit");
            auto b = scratch.get(ctx.device, bytes);
            TORCH_CHECK(b, "Metal scratch allocation failed");
            return b;
        };
        auto scan_work = temporary(metal_detail::scan_bytes(count));
        auto error = at::zeros({1}, means.options().dtype(at::kInt));
        Inputs input;
        auto vm = input.get(means), vc = input.get(colors, true), vo = input.get(opacity),
             vs = input.get(scales, true), vr = input.get(rotations, true), vv = input.get(cov, true),
             vsh = input.get(sh, true);
        auto vview = input.get(view), vproj = input.get(proj), vbg = input.get(bg), vpos = input.get(campos);
        auto geo = input.get(geometry), img = input.get(image), off = input.get(offsets),
             err = input.get(error), rad = input.get(radii), out = input.get(output);
        constexpr uint32_t max_instances = 0x3fffffff;
        ctx.execute(true, [&](id<MTLCommandBuffer> first) {
            ctx.encode(@"training_camera", first, {vview, vproj, vbg, vpos, sub(geo, gl.camera)}, p, 1);
            ctx.encode(@"training_prepare", first,
                       {vm, vc, vo, vs, vr, vv, vsh, sub(geo, gl.camera), geo, sub(geo, gl.clamped),
                        sub(geo, gl.cov), err},
                       p, count);
            ctx.encode(@"preprocess", first, {geo, sub(geo, gl.camera), sub(geo, gl.projected), off, err},
                       std::array<uint32_t, 4>{count, uint32_t(width), uint32_t(height), max_instances},
                       count);
            ctx.encode(@"training_radii", first, {sub(geo, gl.projected), rad},
                       std::array<uint32_t, 4>{count, 0, 0, 0}, count);
            auto encode_first = [&](NSString *name, std::initializer_list<View> buffers,
                                    std::array<uint32_t, 4> args,
                                    size_t threads) { ctx.encode(name, first, buffers, args, threads); };
            metal_detail::scan(off, scan_work, count, max_instances + 1, encode_first);
        });
        // Only scalars cross to the CPU; Gaussian inputs and image/gradient tensors stay in MPS buffers.
        int err_code = 0;
        TORCH_CHECK(err.buffer.contents, "MPS error buffer is not CPU accessible");
        std::memcpy(&err_code, static_cast<const char *>(err.buffer.contents) + err.offset, 4);
        TORCH_CHECK(err_code == 0, err_code == 2 ? "Point filtered although prefiltered is set"
                                                 : "invalid projected covariance/coordinates");
        uint32_t instances = 0;
        auto *off_data = static_cast<const char *>(off.buffer.contents) + off.offset;
        TORCH_CHECK(off_data, "MPS count buffer is not CPU accessible");
        std::memcpy(&instances, off_data + (count - 1) * 4, 4);
        TORCH_CHECK(instances <= max_instances, "tile instance count exceeds supported range");
        auto binning = bytes(std::max<size_t>(1, instances) * 16, means);
        auto records = input.get(binning);
        ctx.execute(debug, [&](id<MTLCommandBuffer> second) {
            if (instances) {
                auto other = temporary(size_t(instances) * 16);
                auto groups = metal_detail::blocks(instances);
                auto histogram = temporary(size_t(groups) * 16 * 4);
                auto radix_work = temporary(metal_detail::scan_bytes(groups * 16));
                auto passes = metal_detail::radix_passes(((width + 15) / 16) * ((height + 15) / 16));
                auto initial = (passes & 1) ? other : records, next = (passes & 1) ? records : other;
                ctx.encode(@"duplicate", second, {sub(geo, gl.projected), off, initial},
                           std::array<uint32_t, 4>{count, uint32_t((width + 15) / 16), 0, 0}, count);
                auto encode_second = [&](NSString *name, std::initializer_list<View> buffers,
                                         std::array<uint32_t, 4> args, size_t threads) {
                    ctx.encode(name, second, buffers, args, threads);
                };
                metal_detail::radix_sort(initial, next, histogram, radix_work, instances, passes,
                                         encode_second);
                ctx.encode(@"identify_ranges", second, {records, img},
                           std::array<uint32_t, 4>{instances, 0, 0, 0}, instances);
            }
            ctx.encode(
                @"training_render", second,
                {geo, sub(geo, gl.projected), records, img, sub(img, il.transmittance),
                 sub(img, il.contributors), sub(geo, gl.camera), out},
                std::array<uint32_t, 4>{uint32_t(width), uint32_t(height), uint32_t((width + 15) / 16), 0},
                size_t(width) * height);
        });
        return {int(instances), output, radii, geometry, binning, image};
    }
}

BackwardResult backward(const Tensor &bg, const Tensor &means, const Tensor &radii, const Tensor &colors,
                        const Tensor &scales, const Tensor &rotations, double modifier, const Tensor &cov,
                        const Tensor &view, const Tensor &proj, double tanx, double tany, const Tensor &grad,
                        const Tensor &sh, int degree, const Tensor &campos, const Tensor &geometry,
                        int instances, const Tensor &binning, const Tensor &image, bool debug) {
    @autoreleasepool {
        TORCH_CHECK(grad.dim() == 3 && grad.size(0) == 3, "grad_out must have shape [3,H,W]");
        check_float(grad, "grad_out");
        int h = grad.size(1), w = grad.size(2);
        auto p = params(means, w, h, sh, degree, colors, scales, tanx, tany, modifier, false);
        uint32_t count = p.dimensions[0];
        auto options = means.options();
        auto gm2 = at::zeros({count, 3}, options), gc = at::zeros({count, 3}, options),
             go = at::zeros({count, 1}, options), gm3 = at::zeros({count, 3}, options);
        auto gv = at::zeros({count, 6}, options), gsh = at::zeros({count, p.dimensions[3], 3}, options),
             gs = at::zeros({count, 3}, options), gr = at::zeros({count, 4}, options),
             gconic = at::zeros({count, 4}, options);
        if (!count)
            return {gm2, gc, go, gm3, gv, gsh, gs, gr};
        check_common(bg, view, proj, campos);
        check_geometry_inputs(p, rotations, cov);
        TORCH_CHECK(radii.scalar_type() == at::kInt && radii.numel() == count, "invalid radii");
        Geometry gl(count);
        Image il(w, h);
        TORCH_CHECK(geometry.scalar_type() == at::kByte && geometry.numel() == gl.size &&
                        image.scalar_type() == at::kByte && image.numel() == il.size,
                    "invalid saved buffers");
        TORCH_CHECK(instances >= 0 && binning.scalar_type() == at::kByte &&
                        binning.numel() >= int64_t(instances) * 16,
                    "invalid saved binning");
        auto &ctx = context();
        std::lock_guard<std::mutex> lock(ctx.mutex);
        Inputs input;
        auto geo = input.get(geometry), img = input.get(image), records = input.get(binning),
             outgrad = input.get(grad);
        auto m2 = input.get(gm2), col = input.get(gc), op = input.get(go), m3 = input.get(gm3),
             v = input.get(gv), shg = input.get(gsh, true), sc = input.get(gs), rot = input.get(gr),
             conic = input.get(gconic);
        auto vm = input.get(means), rad = input.get(radii), vsh = input.get(sh, true),
             vs = input.get(scales, true), vr = input.get(rotations, true);
        ctx.execute(debug, [&](id<MTLCommandBuffer> command) {
            ctx.encode(@"training_backward_render", command,
                       {geo, sub(geo, gl.projected), records, img, sub(img, il.transmittance),
                        sub(img, il.contributors), sub(geo, gl.camera), outgrad, m2, conic, op, col},
                       std::array<uint32_t, 4>{uint32_t(w), uint32_t(h), uint32_t((w + 15) / 16), 0},
                       size_t(w) * h);
            ctx.encode(@"training_backward_preprocess", command,
                       {vm, rad, vsh, sub(geo, gl.clamped), vs, vr, sub(geo, gl.cov), sub(geo, gl.camera), m2,
                        conic, m3, col, v, shg, sc, rot},
                       p, count);
        });
        return {gm2, gc, go, gm3, gv, gsh, gs, gr};
    }
}
Tensor visible(const Tensor &means, const Tensor &view, const Tensor &proj) {
    @autoreleasepool {
        check_float(means, "means3D");
        TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "positions must be [P,3]");
        check_float(view, "viewmatrix");
        check_float(proj, "projmatrix");
        TORCH_CHECK(view.numel() == 16 && proj.numel() == 16, "matrices must have 16 values");
        auto out = at::zeros({means.size(0)}, means.options().dtype(at::kBool));
        if (!means.size(0))
            return out;
        auto &ctx = context();
        std::lock_guard<std::mutex> lock(ctx.mutex);
        Inputs input;
        auto vm = input.get(means), vv = input.get(view), vo = input.get(out);
        ctx.execute(false, [&](id<MTLCommandBuffer> command) {
            ctx.encode(@"training_visible", command, {vm, vv, vo},
                       std::array<uint32_t, 4>{uint32_t(means.size(0)), 0, 0, 0}, means.size(0));
        });
        return out;
    }
}
} // namespace dgr::torch_binding
