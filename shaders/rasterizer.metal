/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * Adaptation: Metal forward kernels, precomputed covariance/RGB, scan and bitonic sorting.
 */
#include <metal_stdlib>
using namespace metal;

struct Projected {
    float4 centerDepthRadius;
    float4 conicOpacity;
    float4 covariance2D;
    uint4 rect;
};

float transform(device const float *m, float3 p, uint row) {
    return m[row] * p.x + m[row + 4] * p.y + m[row + 8] * p.z + m[row + 12];
}

kernel void preprocess(device const float *input [[buffer(0)]], device const float *camera [[buffer(1)]],
                       device Projected *output [[buffer(2)]], device uint *counts [[buffer(3)]],
                       device atomic_uint *error [[buffer(4)]], constant uint4 &dims [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= dims.x)
        return;
    output[id] = Projected{};
    counts[id] = 0;
    device const float *g = input + id * 13;
    float3 mean(g[0], g[1], g[2]);
    float3 t(transform(camera, mean, 0), transform(camera, mean, 1), transform(camera, mean, 2));
    if (t.z <= 0.2f)
        return;
    t.x = clamp(t.x / t.z, -1.3f * camera[36], 1.3f * camera[36]) * t.z;
    t.y = clamp(t.y / t.z, -1.3f * camera[37], 1.3f * camera[37]) * t.z;
    float3 jx(camera[34] / t.z, 0, -camera[34] * t.x / (t.z * t.z));
    float3 jy(0, camera[35] / t.z, -camera[35] * t.y / (t.z * t.z));
    float3 a, b;
    for (uint k = 0; k < 3; k++) {
        float3 col(camera[k * 4], camera[k * 4 + 1], camera[k * 4 + 2]);
        a[k] = dot(jx, col);
        b[k] = dot(jy, col);
    }
    float3x3 sigma(float3(g[3], g[4], g[5]), float3(g[4], g[6], g[7]), float3(g[5], g[7], g[8]));
    float xx = dot(a, sigma * a) + 0.3f, xy = dot(a, sigma * b), yy = dot(b, sigma * b) + 0.3f;
    float det = xx * yy - xy * xy;
    // Upstream drops only this Gaussian when the projected covariance is singular.
    // Valid, extremely anisotropic scales can reach zero here after float32 rounding.
    if (det == 0.0f)
        return;
    float mid = 0.5f * (xx + yy);
    float radius = ceil(3.0f * sqrt(mid + sqrt(max(0.1f, mid * mid - det))));
    float w = transform(camera + 16, mean, 3) + 0.0000001f;
    float2 center = ((float2(transform(camera + 16, mean, 0), transform(camera + 16, mean, 1)) / w + 1.0f) *
                         float2(dims.y, dims.z) -
                     1.0f) *
                    0.5f;
    float3 conic(yy / det, -xy / det, xx / det);
    if (!isfinite(det) || !all(isfinite(conic)) || !all(isfinite(center)) || !isfinite(t.z) ||
        !isfinite(radius) || radius >= 1e9f || any(abs(center) >= 1e9f)) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    int2 grid = int2((dims.y + 15) / 16, (dims.z + 15) / 16);
    uint2 lo = uint2(clamp(int2((center - radius) / 16.0f), int2(0), grid));
    uint2 hi = uint2(clamp(int2((center + radius + 15.0f) / 16.0f), int2(0), grid));
    uint count = (hi.x - lo.x) * (hi.y - lo.y);
    if (count == 0)
        return;
    output[id].centerDepthRadius = float4(center, t.z, radius);
    output[id].conicOpacity = float4(conic, g[12]);
    output[id].covariance2D = float4(xx, xy, yy, 0);
    output[id].rect = uint4(lo, hi);
    counts[id] = min(count, dims.w + 1);
}

// Hierarchical inclusive scan, with saturated addition before any uint overflow.
// Every dispatched group has 256 lanes, including the final partial block.
uint capped_add(uint a, uint b, uint cap) {
    return a + min(b, cap - a);
}
kernel void scan_blocks(device uint *values [[buffer(0)]], device uint *sums [[buffer(1)]],
                        constant uint4 &args [[buffer(2)]], uint lane [[thread_index_in_threadgroup]],
                        uint group [[threadgroup_position_in_grid]]) {
    threadgroup uint shared[256];
    uint id = group * 256 + lane;
    shared[lane] = id < args.x ? min(values[id], args.y) : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        uint value = shared[lane];
        if (lane >= stride)
            value = capped_add(value, shared[lane - stride], args.y);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (id < args.x)
        values[id] = shared[lane];
    if (lane == 255)
        sums[group] = shared[255];
}
kernel void scan_add(device uint *values [[buffer(0)]], device const uint *sums [[buffer(1)]],
                     constant uint4 &args [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < args.x && id >= 256)
        values[id] = capped_add(values[id], sums[id / 256 - 1], args.y);
}
uint radix_digit(uint4 record, uint pass) {
    return pass < 8 ? (record.y >> (pass * 4)) & 15 : (record.x >> ((pass - 8) * 4)) & 15;
}
kernel void radix_histogram(device const uint4 *records [[buffer(0)]], device uint *hist [[buffer(1)]],
                            constant uint4 &args [[buffer(2)]], uint lane [[thread_index_in_threadgroup]],
                            uint group [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint counts[16];
    if (lane < 16)
        atomic_store_explicit(counts + lane, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint id = group * 256 + lane;
    if (id < args.x)
        atomic_fetch_add_explicit(counts + radix_digit(records[id], args.z), 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16)
        hist[lane * args.y + group] = atomic_load_explicit(counts + lane, memory_order_relaxed);
}
kernel void radix_scatter(device const uint4 *input [[buffer(0)]], device uint4 *output [[buffer(1)]],
                          device const uint *prefix [[buffer(2)]], constant uint4 &args [[buffer(3)]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]],
                          uint simd_lane [[thread_index_in_simdgroup]],
                          uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint simd_width [[threads_per_simdgroup]]) {
    // The host checks SIMD width >= 8 and that it divides the 256-thread group.
    threadgroup uint counts[16 * 32];
    uint id = group * 256 + lane, groups = 256 / simd_width;
    bool valid = id < args.x;
    uint4 record = valid ? input[id] : uint4(0);
    uint digit = radix_digit(record, args.z), rank = 0;
    for (uint bucket = 0; bucket < 16; ++bucket) {
        uint match = uint(valid && digit == bucket);
        uint preceding = simd_prefix_exclusive_sum(match), total = simd_sum(match);
        if (digit == bucket)
            rank = preceding;
        if (simd_lane == 0)
            counts[bucket * groups + simd_group] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < simd_group; ++i)
        rank += counts[digit * groups + i];
    if (valid) {
        uint bucket_block = digit * args.y + group;
        uint base = bucket_block ? prefix[bucket_block - 1] : 0;
        output[base + rank] = record;
    }
}

kernel void duplicate(device const Projected *projected [[buffer(0)]],
                      device const uint *offsets [[buffer(1)]], device uint4 *records [[buffer(2)]],
                      constant uint4 &args [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id >= args.x)
        return;
    uint offset = id == 0 ? 0 : offsets[id - 1];
    Projected p = projected[id];
    for (uint y = p.rect.y; y < p.rect.w; y++) {
        for (uint x = p.rect.x; x < p.rect.z; x++) {
            records[offset++] = uint4(y * args.y + x, as_type<uint>(p.centerDepthRadius.z), id, 0);
        }
    }
}

kernel void identify_ranges(device const uint4 *records [[buffer(0)]], device uint *ranges [[buffer(1)]],
                            constant uint4 &args [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= args.x)
        return;
    uint tile = records[id].x;
    // Scalar stores avoid overlapping vector writes when two threads set a tile's endpoints.
    if (id == 0 || records[id - 1].x != tile)
        ranges[tile * 2] = id;
    if (id + 1 == args.x || records[id + 1].x != tile)
        ranges[tile * 2 + 1] = id + 1;
}

// One 16x16 threadgroup cooperatively loads consecutive candidates, just as
// CUDA/ROCm load a batch into shared memory. Finished/invalid pixels keep loading
// and attending barriers until the entire tile is done.
struct TileBatch {
    float2 xy[256];
    float4 conic[256];
    packed_float3 rgb[256];
    uint ids[256];
    uint active[256];
};
struct PixelResult {
    float3 rgb;
    float t;
    uint last;
};
bool tile_done(bool done, threadgroup uint *active, uint lane, uint simd_lane, uint simd_group,
               uint simd_width) {
    uint count = simd_sum(uint(!done));
    if (simd_lane == 0)
        active[simd_group] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = 0;
    for (uint i = 0; i < 256 / simd_width; ++i)
        total += active[i];
    return total == 0;
}
void load_candidate(threadgroup TileBatch &batch, uint lane, uint id, device const Projected *projected,
                    device const float *packed) {
    batch.ids[lane] = id;
    batch.xy[lane] = projected[id].centerDepthRadius.xy;
    batch.conic[lane] = projected[id].conicOpacity;
    batch.rgb[lane] = packed_float3(packed[id * 13 + 9], packed[id * 13 + 10], packed[id * 13 + 11]);
}
PixelResult render_tile(device const float *packed, device const Projected *projected,
                        device const uint4 *records, uint2 range, uint2 xy, bool inside,
                        threadgroup TileBatch &batch, uint lane, uint simd_lane, uint simd_group,
                        uint simd_width) {
    PixelResult result = {float3(0), 1.0f, 0};
    bool done = !inside;
    for (uint base = range.x; base < range.y; base += 256) {
        if (tile_done(done, batch.active, lane, simd_lane, simd_group, simd_width))
            break;
        uint index = base + lane;
        if (index < range.y)
            load_candidate(batch, lane, records[index].z, projected, packed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; !done && j < min(256u, range.y - base); ++j) {
            float2 d = batch.xy[j] - float2(xy);
            float4 co = batch.conic[j];
            float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if (power > 0)
                continue;
            float alpha = min(0.99f, co.w * exp(power));
            if (alpha < 1.0f / 255.0f)
                continue;
            float next = result.t * (1.0f - alpha);
            if (next < 0.0001f) {
                done = true;
                continue;
            }
            result.rgb += float3(batch.rgb[j]) * alpha * result.t;
            result.t = next;
            result.last = base + j - range.x + 1;
        }
        // The next tile_done barrier also protects this batch from being overwritten.
    }
    return result;
}
kernel void render(device const float *input [[buffer(0)]], device const Projected *projected [[buffer(1)]],
                   device const uint4 *records [[buffer(2)]], device const uint2 *ranges [[buffer(3)]],
                   device float *finalT [[buffer(4)]], device uint *lastContributor [[buffer(5)]],
                   device const float *camera [[buffer(6)]], constant uint4 &args [[buffer(7)]],
                   texture2d<float, access::write> color [[texture(0)]],
                   uint2 tile [[threadgroup_position_in_grid]],
                   uint2 local [[thread_position_in_threadgroup]], uint lane [[thread_index_in_threadgroup]],
                   uint simd_lane [[thread_index_in_simdgroup]],
                   uint simd_group [[simdgroup_index_in_threadgroup]],
                   uint simd_width [[threads_per_simdgroup]]) {
    threadgroup TileBatch batch;
    uint2 xy = tile * 16 + local;
    bool inside = xy.x < args.x && xy.y < args.y;
    PixelResult r = render_tile(input, projected, records, ranges[tile.y * args.z + tile.x], xy, inside,
                                batch, lane, simd_lane, simd_group, simd_width);
    if (!inside)
        return;
    uint pixel = xy.y * args.x + xy.x;
    finalT[pixel] = r.t;
    lastContributor[pixel] = r.last;
    r.rgb += r.t * float3(camera[38], camera[39], camera[40]);
    if (args.w == 0)
        r.rgb = float3(0);
    color.write(float4(r.rgb, 1.0f), xy);
}
