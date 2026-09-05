#include "dgr/metal_rasterizer.h"
#include "layout.h"
#include "shader_source.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <utility>

namespace dgr {
namespace {
std::string describe(NSError *error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "unknown Metal error";
}

// ARC owns every id member; autoreleased command objects are bounded by each public call's pool.
struct Allocator {
    id<MTLDevice> device;
    std::size_t limit;
    std::size_t allocated = 0;

    void reserve(std::size_t size) {
        if (size > limit - allocated)
            throw Error(ErrorCode::resource_limit,
                        "Metal allocation budget exceeds " + std::to_string(limit) + " bytes");
        allocated += size;
    }

    id<MTLBuffer> buffer(std::size_t size) {
        size = std::max<std::size_t>(16, size);
        reserve(size);
        if (size > device.maxBufferLength)
            throw Error(ErrorCode::resource_limit, "buffer exceeds device limit");
        id<MTLBuffer> b = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!b)
            throw Error(ErrorCode::resource_limit, "Metal buffer allocation failed");
        std::memset(b.contents, 0, size);
        return b;
    }
};

id<MTLCommandBuffer> command_buffer(id<MTLCommandQueue> queue) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command)
        throw Error(ErrorCode::gpu, "command buffer allocation failed");
    return command;
}

void complete(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw Error(ErrorCode::gpu, describe(command.error));
}

template <class T> std::vector<T> read_buffer(id<MTLBuffer> buffer, std::size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::vector<T> result(count);
    if (count)
        std::memcpy(result.data(), buffer.contents, count * sizeof(T));
    return result;
}
} // namespace

struct MetalFrame::Impl {
    id<MTLTexture> texture;
    id<MTLBuffer> projected, offsets, records, ranges, final_t, contributors;
    std::size_t count = 0, tile_count = 0;
    FrameStats stats;
};

struct MetalRasterizer::Impl {
    RenderLimits limits;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;

    explicit Impl(RenderLimits l) : limits(l) {
        validate_limits(limits);
        device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory || ![device supportsFamily:MTLGPUFamilyApple1])
            throw Error(ErrorCode::unavailable, "an Apple Silicon Metal device is required");
        queue = [device newCommandQueue];
        if (!queue)
            throw Error(ErrorCode::gpu, "command queue allocation failed");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:detail::shader_source];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library)
            throw Error(ErrorCode::gpu, "shader compilation: " + describe(error));
        auto *built = [NSMutableDictionary<NSString *, id<MTLComputePipelineState>> dictionary];
        for (NSString *name in @[
                 @"preprocess", @"scan_step", @"initialize_records", @"duplicate", @"bitonic_step",
                 @"identify_ranges", @"render"
             ]) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (!function)
                throw Error(ErrorCode::gpu, "missing kernel: " + std::string(name.UTF8String));
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                         error:&error];
            if (!pipeline)
                throw Error(ErrorCode::gpu, "pipeline compilation: " + describe(error));
            built[name] = pipeline;
        }
        pipelines = [built copy];
    }

    void encode(NSString *name, id<MTLCommandBuffer> command, std::initializer_list<id<MTLBuffer>> buffers,
                std::array<std::uint32_t, 4> args, std::size_t threads, id<MTLTexture> texture = nil) const {
        if (!threads)
            return;
        id<MTLComputePipelineState> pipeline = pipelines[name];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!pipeline || !encoder)
            throw Error(ErrorCode::gpu, "compute encoder allocation failed");
        encoder.label = name;
        [encoder setComputePipelineState:pipeline];
        NSUInteger index = 0;
        for (id<MTLBuffer> buffer : buffers)
            [encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:args.data() length:sizeof(args) atIndex:index];
        if (texture)
            [encoder setTexture:texture atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                      std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1,
                                      1)];
        // Tracked resources and separate serial encoders order dependent scan/sort passes.
        [encoder endEncoding];
    }
};

MetalRasterizer::MetalRasterizer(RenderLimits limits) {
    @autoreleasepool {
        impl_ = std::make_unique<Impl>(limits);
    }
}
MetalRasterizer::~MetalRasterizer() = default;
MetalRasterizer::MetalRasterizer(MetalRasterizer &&) noexcept = default;
MetalRasterizer &MetalRasterizer::operator=(MetalRasterizer &&) noexcept = default;

const MetalRasterizer::Impl &MetalRasterizer::checked() const {
    if (!impl_)
        throw Error(ErrorCode::invalid_input, "use of moved-from MetalRasterizer");
    return *impl_;
}

std::string MetalRasterizer::device_name() const {
    @autoreleasepool {
        return std::string(checked().device.name.UTF8String);
    }
}

MetalFrame MetalRasterizer::render(const Scene &scene) {
    @autoreleasepool {
        const auto &m = checked();
        validate_scene(scene, m.limits);
        const auto &c = scene.camera;
        const auto count = static_cast<std::uint32_t>(scene.gaussians.size());
        const auto width = static_cast<std::uint32_t>(c.width), height = static_cast<std::uint32_t>(c.height);
        const auto tiles_x = (width + 15) / 16, tiles_y = (height + 15) / 16;
        const auto pixels = static_cast<std::size_t>(width) * height;
        const auto max_instances = static_cast<std::uint32_t>(m.limits.max_instances);
        Allocator alloc{m.device, m.limits.max_working_bytes};
        id<MTLBuffer> input = alloc.buffer(static_cast<std::size_t>(count) * 13 * sizeof(float));
        // Pack directly into shared memory, without temporary per-Gaussian arrays or a second copy.
        auto *packed = static_cast<float *>(input.contents);
        for (const auto &g : scene.gaussians) {
            packed = std::copy(g.mean.begin(), g.mean.end(), packed);
            packed = std::copy(g.covariance.begin(), g.covariance.end(), packed);
            packed = std::copy(g.color.begin(), g.color.end(), packed);
            *packed++ = g.opacity;
        }
        id<MTLBuffer> camera = alloc.buffer(41 * sizeof(float));
        auto *camera_data = static_cast<float *>(camera.contents);
        std::copy(c.view_matrix.begin(), c.view_matrix.end(), camera_data);
        std::copy(c.projection_matrix.begin(), c.projection_matrix.end(), camera_data + 16);
        const std::array<float, 9> params{
            float(width),   float(height), width / (2 * c.tan_fov_x), height / (2 * c.tan_fov_y),
            c.tan_fov_x,    c.tan_fov_y,   c.background[0],           c.background[1],
            c.background[2]};
        std::copy(params.begin(), params.end(), camera_data + 32);
        id<MTLBuffer> projected = alloc.buffer(static_cast<std::size_t>(count) * sizeof(detail::Projected));
        id<MTLBuffer> offsets = alloc.buffer(static_cast<std::size_t>(count) * 4);
        id<MTLBuffer> scratch = alloc.buffer(static_cast<std::size_t>(count) * 4);
        id<MTLBuffer> error = alloc.buffer(4);
        id<MTLCommandBuffer> first = command_buffer(m.queue);
        m.encode(@"preprocess", first, {input, camera, projected, offsets, error},
                 {count, width, height, max_instances}, count);
        for (std::uint32_t stride = 1; stride < count; stride *= 2) {
            m.encode(@"scan_step", first, {offsets, scratch}, {count, stride, max_instances, 0}, count);
            id<MTLBuffer> temp = offsets;
            offsets = scratch;
            scratch = temp;
        }
        complete(first);
        std::uint32_t error_code = 0, instances = 0;
        std::memcpy(&error_code, error.contents, sizeof(error_code));
        if (error_code)
            throw Error(ErrorCode::invalid_input, "invalid projected covariance/coordinates");
        // The scalar readback remains explicit until a bounded asynchronous allocator is implemented.
        if (count)
            std::memcpy(&instances, static_cast<const char *>(offsets.contents) + (count - 1) * 4, 4);
        if (instances > max_instances)
            throw Error(ErrorCode::resource_limit, "tile instance limit exceeded");
        std::uint32_t padded = 1;
        while (padded < instances)
            padded *= 2;
        id<MTLBuffer> records = alloc.buffer(static_cast<std::size_t>(padded) * sizeof(detail::Record));
        id<MTLBuffer> ranges =
            alloc.buffer(static_cast<std::size_t>(tiles_x) * tiles_y * sizeof(detail::Range));
        id<MTLBuffer> final_t = alloc.buffer(pixels * 4), contributors = alloc.buffer(pixels * 4);
        auto *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                              width:width
                                                                             height:height
                                                                          mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        alloc.reserve([m.device heapTextureSizeAndAlignWithDescriptor:descriptor].size);
        id<MTLTexture> texture = [m.device newTextureWithDescriptor:descriptor];
        if (!texture)
            throw Error(ErrorCode::resource_limit, "output texture allocation failed");
        id<MTLCommandBuffer> second = command_buffer(m.queue);
        m.encode(@"initialize_records", second, {records}, {padded, 0, 0, 0}, padded);
        if (instances) {
            m.encode(@"duplicate", second, {projected, offsets, records}, {count, tiles_x, 0, 0}, count);
            for (std::uint32_t k = 2; k <= padded; k *= 2)
                for (std::uint32_t j = k / 2; j > 0; j /= 2)
                    m.encode(@"bitonic_step", second, {records}, {padded, j, k, 0}, padded);
            m.encode(@"identify_ranges", second, {records, ranges}, {instances, 0, 0, 0}, instances);
        }
        m.encode(@"render", second, {input, projected, records, ranges, final_t, contributors, camera},
                 {width, height, tiles_x, count}, pixels, texture);
        complete(second);
        auto result = std::make_shared<MetalFrame::Impl>();
        result->texture = texture;
        result->projected = projected;
        result->offsets = offsets;
        result->records = records;
        result->ranges = ranges;
        result->final_t = final_t;
        result->contributors = contributors;
        result->count = count;
        result->tile_count = tiles_x * tiles_y;
        result->stats = {instances, alloc.allocated,
                         std::max(0.0, first.GPUEndTime - first.GPUStartTime) +
                             std::max(0.0, second.GPUEndTime - second.GPUStartTime)};
        return MetalFrame(std::move(result));
    }
}

MetalFrame::MetalFrame(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalFrame::~MetalFrame() = default;
MetalFrame::MetalFrame(const MetalFrame &) = default;
MetalFrame &MetalFrame::operator=(const MetalFrame &) = default;
MetalFrame::MetalFrame(MetalFrame &&) noexcept = default;
MetalFrame &MetalFrame::operator=(MetalFrame &&) noexcept = default;
const MetalFrame::Impl &MetalFrame::checked() const {
    if (!impl_)
        throw Error(ErrorCode::invalid_input, "use of moved-from MetalFrame");
    return *impl_;
}
int MetalFrame::width() const {
    return static_cast<int>(checked().texture.width);
}
int MetalFrame::height() const {
    return static_cast<int>(checked().texture.height);
}
FrameStats MetalFrame::stats() const {
    return checked().stats;
}
void *MetalFrame::native_texture_handle() const {
    return (__bridge void *)checked().texture;
}

std::vector<float> MetalFrame::read_rgb() const {
    @autoreleasepool {
        const auto &f = checked();
        const std::size_t pixels = f.texture.width * f.texture.height;
        std::vector<float> rgba(pixels * 4), rgb(pixels * 3);
        [f.texture getBytes:rgba.data()
                bytesPerRow:f.texture.width * 16
                 fromRegion:MTLRegionMake2D(0, 0, f.texture.width, f.texture.height)
                mipmapLevel:0];
        for (std::size_t i = 0; i < pixels; ++i)
            std::copy_n(rgba.data() + i * 4, 3, rgb.data() + i * 3);
        return rgb;
    }
}

DebugSnapshot MetalFrame::debug_snapshot() const {
    @autoreleasepool {
        const auto &f = checked();
        DebugSnapshot d;
        for (const auto &p : read_buffer<detail::Projected>(f.projected, f.count)) {
            const auto &v = p.center_depth_radius;
            d.projections.push_back({{v[0], v[1]},
                                     v[2],
                                     int(v[3]),
                                     {p.conic_opacity[0], p.conic_opacity[1], p.conic_opacity[2]},
                                     {p.covariance_2d[0], p.covariance_2d[1], p.covariance_2d[2]},
                                     {int(p.rect[0]), int(p.rect[1]), int(p.rect[2]), int(p.rect[3])}});
        }
        d.offsets = read_buffer<std::uint32_t>(f.offsets, f.count);
        for (const auto &record : read_buffer<detail::Record>(f.records, f.stats.instances)) {
            d.sorted_tile_ids.push_back(record[0]);
            d.sorted_gaussian_ids.push_back(record[2]);
        }
        d.ranges = read_buffer<detail::Range>(f.ranges, f.tile_count);
        const std::size_t pixels = f.texture.width * f.texture.height;
        const auto transmittance = read_buffer<float>(f.final_t, pixels);
        d.final_transmittance.assign(transmittance.begin(), transmittance.end());
        d.last_contributors = read_buffer<std::uint32_t>(f.contributors, pixels);
        return d;
    }
}
} // namespace dgr
