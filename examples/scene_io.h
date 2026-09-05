#pragma once
#include "dgr/scene.h"
#include <filesystem>
#include <string_view>

namespace dgr::example {
Scene parse_scene_json(std::string_view json);
Scene load_scene(const std::filesystem::path &path, std::size_t max_bytes = 64 * 1024 * 1024);
void write_ppm(const std::filesystem::path &path, int width, int height, const std::vector<float> &rgb);
} // namespace dgr::example
