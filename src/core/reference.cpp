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
 * Adaptation: independent scalar C++ Float64 reference for precomputed covariance/RGB.
 */
#include "dgr/reference.h"
#include <algorithm>
#include <cmath>

namespace dgr {
namespace {
ProjectionSnapshot project(const Gaussian &g, const Camera &c) {
    auto transform = [&](const Matrix4 &m, int row) {
        return double(m[row]) * g.mean[0] + double(m[row + 4]) * g.mean[1] + double(m[row + 8]) * g.mean[2] +
               m[row + 12];
    };
    const double z = transform(c.view_matrix, 2);
    if (z <= double(0.2f))
        return {};
    const double tx = std::clamp(transform(c.view_matrix, 0) / z, -1.3 * c.tan_fov_x, 1.3 * c.tan_fov_x) * z;
    const double ty = std::clamp(transform(c.view_matrix, 1) / z, -1.3 * c.tan_fov_y, 1.3 * c.tan_fov_y) * z;
    const double fx = c.width / (2.0 * c.tan_fov_x), fy = c.height / (2.0 * c.tan_fov_y);
    const std::array<double, 3> jx{fx / z, 0, -fx * tx / (z * z)}, jy{0, fy / z, -fy * ty / (z * z)};
    auto world_row = [&](const std::array<double, 3> &j) {
        std::array<double, 3> result{};
        for (int col = 0; col < 3; ++col)
            for (int row = 0; row < 3; ++row)
                result[col] += j[row] * c.view_matrix[col * 4 + row];
        return result;
    };
    const auto a = world_row(jx), b = world_row(jy);
    const auto &v = g.covariance;
    const double sigma[3][3] = {{v[0], v[1], v[2]}, {v[1], v[3], v[4]}, {v[2], v[4], v[5]}};
    auto quadratic = [&](const auto &left, const auto &right) {
        double sum = 0;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                sum += left[row] * sigma[row][col] * right[col];
        return sum;
    };
    const double xx = quadratic(a, a) + 0.3, xy = quadratic(a, b), yy = quadratic(b, b) + 0.3;
    const double det = xx * yy - xy * xy;
    if (!(det > 0) || !std::isfinite(det))
        throw Error(ErrorCode::invalid_input, "invalid projected covariance");
    const double mid = 0.5 * (xx + yy);
    const double radius = std::ceil(3 * std::sqrt(mid + std::sqrt(std::max(0.1, mid * mid - det))));
    const double w = transform(c.projection_matrix, 3) + 0.0000001;
    const double px = ((transform(c.projection_matrix, 0) / w + 1) * c.width - 1) * 0.5;
    const double py = ((transform(c.projection_matrix, 1) / w + 1) * c.height - 1) * 0.5;
    for (double x : {px, py, z, radius, yy / det, -xy / det, xx / det})
        if (!std::isfinite(x))
            throw Error(ErrorCode::invalid_input, "non-finite projection");
    if (radius >= 1e9 || std::abs(px) >= 1e9 || std::abs(py) >= 1e9)
        throw Error(ErrorCode::invalid_input, "projection exceeds supported coordinate range");
    const int r = static_cast<int>(radius), tiles_x = (c.width + 15) / 16, tiles_y = (c.height + 15) / 16;
    // Truncate toward zero, as upstream getRect does. This is not floor division.
    const std::array<int, 4> rect{
        std::clamp(int((px - r) / 16), 0, tiles_x), std::clamp(int((py - r) / 16), 0, tiles_y),
        std::clamp(int((px + r + 15) / 16), 0, tiles_x), std::clamp(int((py + r + 15) / 16), 0, tiles_y)};
    if ((rect[2] - rect[0]) * (rect[3] - rect[1]) == 0)
        return {};
    return {{px, py}, z, r, {yy / det, -xy / det, xx / det}, {xx, xy, yy}, rect};
}
} // namespace

ReferenceFrame render_reference(const Scene &s, const RenderLimits &limits) {
    validate_scene(s, limits);
    const auto &c = s.camera;
    ReferenceFrame result;
    result.width = c.width;
    result.height = c.height;
    auto &d = result.debug;
    d.projections.reserve(s.gaussians.size());
    d.offsets.reserve(s.gaussians.size());
    const int tiles_x = (c.width + 15) / 16, tiles_y = (c.height + 15) / 16;
    std::vector<std::vector<std::uint32_t>> lists(tiles_x * tiles_y);
    std::size_t count = 0;
    for (std::size_t id = 0; id < s.gaussians.size(); ++id) {
        const auto p = project(s.gaussians[id], c);
        d.projections.push_back(p);
        count += p.tiles_touched();
        if (count > limits.max_instances)
            throw Error(ErrorCode::resource_limit, "tile instances exceed limit");
        d.offsets.push_back(static_cast<std::uint32_t>(count));
        for (int y = p.rect[1]; y < p.rect[3]; ++y)
            for (int x = p.rect[0]; x < p.rect[2]; ++x)
                lists[y * tiles_x + x].push_back(static_cast<std::uint32_t>(id));
    }
    // Independent per-tile sort rather than the GPU's global key sort.
    for (std::size_t tile = 0; tile < lists.size(); ++tile) {
        auto &list = lists[tile];
        std::sort(list.begin(), list.end(), [&](auto a, auto b) {
            const float za = float(d.projections[a].depth), zb = float(d.projections[b].depth);
            return za == zb ? a < b : za < zb;
        });
        const auto start = static_cast<std::uint32_t>(d.sorted_gaussian_ids.size());
        d.sorted_gaussian_ids.insert(d.sorted_gaussian_ids.end(), list.begin(), list.end());
        d.sorted_tile_ids.insert(d.sorted_tile_ids.end(), list.size(), static_cast<std::uint32_t>(tile));
        d.ranges.push_back(list.empty()
                               ? std::array<std::uint32_t, 2>{0, 0}
                               : std::array<std::uint32_t, 2>{
                                     start, static_cast<std::uint32_t>(d.sorted_gaussian_ids.size())});
    }
    const auto pixels = static_cast<std::size_t>(c.width) * c.height;
    result.rgb.resize(pixels * 3);
    d.final_transmittance.resize(pixels, 1);
    d.last_contributors.resize(pixels);
    for (int y = 0; y < c.height; ++y)
        for (int x = 0; x < c.width; ++x) {
            const auto pixel = static_cast<std::size_t>(y) * c.width + x;
            double t = 1;
            const auto &list = lists[(y / 16) * tiles_x + x / 16];
            for (std::size_t pos = 0; pos < list.size(); ++pos) {
                const auto id = list[pos];
                const auto &p = d.projections[id];
                const auto &g = s.gaussians[id];
                const double dx = p.center[0] - x, dy = p.center[1] - y;
                const double power =
                    -0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) - p.conic[1] * dx * dy;
                if (power > 0)
                    continue;
                const double alpha = std::min(0.99, double(g.opacity) * std::exp(power));
                if (alpha < 1.0 / 255.0)
                    continue;
                const double next = t * (1 - alpha);
                if (next < 0.0001)
                    break; // Reject the threshold-crossing candidate, matching CUDA.
                for (int ch = 0; ch < 3; ++ch)
                    result.rgb[pixel * 3 + ch] += g.color[ch] * alpha * t;
                t = next;
                d.last_contributors[pixel] = static_cast<std::uint32_t>(pos + 1);
            }
            d.final_transmittance[pixel] = t;
            if (!s.gaussians.empty())
                for (int ch = 0; ch < 3; ++ch)
                    result.rgb[pixel * 3 + ch] += t * c.background[ch];
        }
    return result;
}
} // namespace dgr
