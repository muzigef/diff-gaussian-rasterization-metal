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

// Inclusive Hillis-Steele scan. Saturation reports overflow before allocating the instance list.
kernel void scan_step(device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]],
                      constant uint4 &args [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= args.x)
        return;
    output[id] = min(args.z + 1, input[id] + (id >= args.y ? input[id - args.y] : 0));
}

kernel void initialize_records(device uint4 *records [[buffer(0)]], constant uint4 &args [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    if (id < args.x)
        records[id] = uint4(0xffffffffu);
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

bool less_record(uint4 a, uint4 b) {
    return a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
}

// Explicit Gaussian ID tie-break reproduces stable input order for equal tile/depth keys.
kernel void bitonic_step(device uint4 *records [[buffer(0)]], constant uint4 &args [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    uint partner = id ^ args.y;
    if (id >= args.x || partner <= id)
        return;
    uint4 a = records[id], b = records[partner];
    bool ascending = (id & args.z) == 0;
    if (ascending ? less_record(b, a) : less_record(a, b)) {
        records[id] = b;
        records[partner] = a;
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

kernel void render(device const float *input [[buffer(0)]], device const Projected *projected [[buffer(1)]],
                   device const uint4 *records [[buffer(2)]], device const uint2 *ranges [[buffer(3)]],
                   device float *finalT [[buffer(4)]], device uint *lastContributor [[buffer(5)]],
                   device const float *camera [[buffer(6)]], constant uint4 &args [[buffer(7)]],
                   texture2d<float, access::write> color [[texture(0)]],
                   uint pixel [[thread_position_in_grid]]) {
    if (pixel >= args.x * args.y)
        return;
    uint2 xy(pixel % args.x, pixel / args.x);
    uint2 range = ranges[(xy.y / 16) * args.z + xy.x / 16];
    float t = 1.0f;
    float3 rgb(0);
    uint last = 0;
    for (uint index = range.x; index < range.y; index++) {
        uint id = records[index].z;
        Projected p = projected[id];
        float2 d = p.centerDepthRadius.xy - float2(xy);
        float4 co = p.conicOpacity;
        float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
        if (power > 0)
            continue;
        float alpha = min(0.99f, co.w * exp(power));
        if (alpha < 1.0f / 255.0f)
            continue;
        float next = t * (1.0f - alpha);
        if (next < 0.0001f)
            break;
        rgb += float3(input[id * 13 + 9], input[id * 13 + 10], input[id * 13 + 11]) * alpha * t;
        t = next;
        last = index - range.x + 1;
    }
    finalT[pixel] = t;
    lastContributor[pixel] = last;
    rgb += t * float3(camera[38], camera[39], camera[40]);
    if (args.w == 0)
        rgb = float3(0); // Match the original Python bridge's P=0 black image.
    color.write(float4(rgb, 1.0f), xy);
}
