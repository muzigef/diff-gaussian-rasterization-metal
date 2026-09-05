#include "dgr/metal_rasterizer.h"
#include "dgr/reference.h"
#include "scene_io.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw dgr::Error(dgr::ErrorCode::invalid_input,
                             "usage: rasterizer-demo <scene.json> <output.ppm>");
        const auto scene = dgr::example::load_scene(argv[1]);
        dgr::MetalRasterizer renderer;
        const auto start = std::chrono::steady_clock::now();
        const auto frame = renderer.render(scene);
        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto rgb = frame.read_rgb();
        const auto reference = dgr::render_reference(scene);
        double max_error = 0, squared = 0;
        for (std::size_t i = 0; i < rgb.size(); ++i) {
            if (!std::isfinite(rgb[i]))
                throw dgr::Error(dgr::ErrorCode::gpu, "non-finite output");
            const double error = std::abs(rgb[i] - reference.rgb[i]);
            max_error = std::max(max_error, error);
            squared += error * error;
        }
        const double rmse = std::sqrt(squared / rgb.size());
        if (max_error > 2e-4 || rmse > 2e-5)
            throw dgr::Error(dgr::ErrorCode::gpu, "CPU reference mismatch");
        dgr::example::write_ppm(argv[2], frame.width(), frame.height(), rgb);
        const auto stats = frame.stats();
        std::cout << "Device: " << renderer.device_name() << '\n'
                  << "Gaussians: " << scene.gaussians.size() << "; tile instances: " << stats.instances
                  << '\n'
                  << std::setprecision(8) << "CPU reference: max abs " << max_error << "; RMSE " << rmse
                  << '\n'
                  << std::fixed << std::setprecision(3) << "GPU command time: " << stats.gpu_seconds * 1000
                  << " ms; render wall: " << wall_ms << " ms (single cold sample)\n"
                  << "Requested Metal allocations: " << stats.allocated_bytes << " bytes\n"
                  << "Written: " << argv[2] << " (linear RGB, clamped to 8-bit)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
