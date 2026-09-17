#pragma once
// Shared by the native and PyTorch bridges. Saved frame buffers never enter this pool.
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace dgr::metal_detail {
struct BufferView {
    id<MTLBuffer> buffer;
    NSUInteger offset = 0;
    BufferView(id<MTLBuffer> b, NSUInteger o = 0) : buffer(b), offset(o) {}
};
inline BufferView sub(BufferView view, size_t offset) {
    view.offset += offset;
    return view;
}
inline size_t aligned(size_t bytes) {
    return (bytes + 15) & ~size_t(15);
}
inline uint32_t blocks(uint32_t count) {
    return (count + 255) / 256;
}
inline size_t scan_bytes(uint32_t count) {
    size_t size = 0;
    do {
        count = blocks(count);
        size += aligned(size_t(count) * 4);
    } while (count > 1);
    return std::max<size_t>(16, size);
}
// An in-place, saturated inclusive scan. Each level reads its entire local block
// before writing; separate encoders order levels and uniform-add passes.
template <class Encode>
void scan(BufferView values, BufferView workspace, uint32_t count, uint32_t cap, Encode &&encode) {
    if (!count)
        return;
    const auto groups = blocks(count);
    encode(@"scan_blocks", {values, workspace}, {count, cap, 0, 0}, count);
    if (groups > 1) {
        scan(workspace, sub(workspace, aligned(size_t(groups) * 4)), groups, cap, encode);
        encode(@"scan_add", {values, workspace}, {count, cap, 0, 0}, count);
    }
}
inline uint32_t radix_passes(uint32_t tile_count) {
    uint32_t bits = 0;
    for (uint32_t top = tile_count - 1; top; top >>= 1)
        ++bits;
    return 8 + (bits + 3) / 4; // 32 depth bits followed by the significant tile bits.
}
// Stable LSD radix sort of (tile ID, positive float depth bits). Duplicate emits
// Gaussian IDs in ascending order, so stability reproduces CUB/hipCUB tie ordering.
template <class Encode>
void radix_sort(BufferView input, BufferView other, BufferView histogram, BufferView workspace,
                uint32_t count, uint32_t passes, Encode &&encode) {
    const auto groups = blocks(count);
    for (uint32_t pass = 0; pass < passes; ++pass) {
        encode(@"radix_histogram", {input, histogram}, {count, groups, pass, 0}, count);
        scan(histogram, workspace, groups * 16, 0xffffffffu, encode);
        encode(@"radix_scatter", {input, other, histogram}, {count, groups, pass, 0}, count);
        std::swap(input, other);
    }
}
// Whole groups are mandatory for kernels containing barriers, including a partial
// last block and edge tiles. Linear, barrier-free kernels can use dispatchThreads.
inline void dispatch(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pipeline, NSString *name,
                     const void *args, size_t threads) {
    const bool tile = [name isEqualToString:@"render"] || [name isEqualToString:@"training_render"] ||
                      [name isEqualToString:@"training_backward_render"];
    const bool block = [name isEqualToString:@"scan_blocks"] || [name hasPrefix:@"radix_"];
    if (tile) {
        const auto *dims = static_cast<const uint32_t *>(args);
        [e dispatchThreadgroups:MTLSizeMake((dims[0] + 15) / 16, (dims[1] + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    } else if (block) {
        [e dispatchThreadgroups:MTLSizeMake((threads + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [e dispatchThreads:MTLSizeMake(threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                      std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1,
                                      1)];
    }
}
// Slots are reused only on an ordered queue (or after completion). Exact-size
// replacement bounds retained memory by the most recent frame, not a high-water
// mark from an unrelated scene. The command buffer retains replaced resources.
struct Scratch {
    std::vector<id<MTLBuffer>> slots;
    size_t cursor = 0;
    void reset() {
        cursor = 0;
    }
    id<MTLBuffer> get(id<MTLDevice> device, size_t size) {
        size = std::max<size_t>(16, size);
        if (cursor == slots.size())
            slots.push_back(nil);
        auto &buffer = slots[cursor++];
        if (!buffer || buffer.length != size)
            buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        return buffer;
    }
    size_t unused_bytes() const {
        size_t total = 0;
        for (size_t i = cursor; i < slots.size(); ++i)
            total += slots[i].length;
        return total;
    }
    void trim() {
        slots.resize(cursor);
    }
};
} // namespace dgr::metal_detail
