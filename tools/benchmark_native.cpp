// Test/benchmark transport only, not a public asset ABI. See TILE_MIGRATION_REPORT.md.
#include "dgr/metal_rasterizer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

template <class T> void read(std::ifstream &file, T &value) {
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file)
        throw std::runtime_error("truncated benchmark input");
}
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::runtime_error("usage: dgr-native-benchmark <scene.bin> <expected-rgb-f32.bin>");
        std::ifstream input(argv[1], std::ios::binary), expected_file(argv[2], std::ios::binary);
        uint32_t count, width, height;
        read(input, count);
        read(input, width);
        read(input, height);
        dgr::Scene scene;
        auto &c = scene.camera;
        c.width = width;
        c.height = height;
        read(input, c.view_matrix);
        read(input, c.projection_matrix);
        read(input, c.tan_fov_x);
        read(input, c.tan_fov_y);
        read(input, c.background);
        scene.gaussians.resize(count);
        for (auto &g : scene.gaussians) {
            read(input, g.mean);
            read(input, g.covariance);
            read(input, g.color);
            read(input, g.opacity);
        }
        std::vector<float> expected(size_t(width) * height * 3);
        expected_file.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float));
        if (!expected_file)
            throw std::runtime_error("truncated expected image");
        dgr::RenderLimits limits;
        limits.max_gaussians = std::max<size_t>(count, 1);
        limits.max_instances = 16 * 1024 * 1024;
        limits.max_pixels = size_t(width) * height;
        limits.max_working_bytes = 1024ull * 1024 * 1024;
        dgr::MetalRasterizer renderer(limits);
        std::vector<double> wall, gpu;
        dgr::FrameStats stats;
        double max_error = 0;
        for (int i = 0; i < 8; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto frame = renderer.render(scene);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            stats = frame.stats();
            if (i >= 3) {
                wall.push_back(seconds);
                gpu.push_back(stats.gpu_seconds);
            }
            if (i == 7) {
                const auto actual = frame.read_rgb();
                for (size_t j = 0; j < actual.size(); ++j) {
                    if (!std::isfinite(actual[j]))
                        throw std::runtime_error("nonfinite output");
                    max_error = std::max(max_error, std::abs(double(actual[j]) - expected[j]));
                }
            }
        }
        std::sort(wall.begin(), wall.end());
        std::sort(gpu.begin(), gpu.end());
        std::cout << "{\"gaussians\":" << count << ",\"instances\":" << stats.instances
                  << ",\"requested_active_bytes\":" << stats.allocated_bytes
                  << ",\"warmups\":3,\"repetitions\":5,\"wall_median_seconds\":" << wall[2]
                  << ",\"gpu_median_seconds\":" << gpu[2] << ",\"max_abs\":" << max_error << "}\n";
        return max_error == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
