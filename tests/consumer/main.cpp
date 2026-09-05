#include <dgr/metal_rasterizer.h>
#include <dgr/reference.h>
#include <iostream>
int main() {
    dgr::Scene scene;
    scene.camera = dgr::Camera::perspective(1, 1, 1, 1, {1, 1, 1});
    dgr::MetalRasterizer renderer;
    auto frame = renderer.render(scene);
    if (frame.read_rgb() != std::vector<float>({0, 0, 0})) return 1;
    std::cout << renderer.device_name() << ": installed C++ library works\n";
}
