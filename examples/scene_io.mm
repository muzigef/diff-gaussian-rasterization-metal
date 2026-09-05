#include "scene_io.h"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace dgr::example {
namespace {
[[noreturn]] void invalid(const char *field) {
    throw Error(ErrorCode::invalid_input, std::string("invalid JSON field: ") + field);
}
NSDictionary *object(id value, const char *field) {
    if (![value isKindOfClass:[NSDictionary class]])
        invalid(field);
    return (NSDictionary *)value;
}
double number(id value, const char *field) {
    if (![value isKindOfClass:[NSNumber class]] ||
        CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
        invalid(field);
    const double x = [(NSNumber *)value doubleValue];
    if (!std::isfinite(x))
        invalid(field);
    return x;
}
int integer(id value, const char *field) {
    const double x = number(value, field);
    if (x != std::trunc(x) || x < std::numeric_limits<int>::min() || x > std::numeric_limits<int>::max())
        invalid(field);
    return static_cast<int>(x);
}
float scalar(id value, const char *field) {
    const double x = number(value, field);
    if (std::abs(x) > std::numeric_limits<float>::max())
        invalid(field);
    return static_cast<float>(x);
}
template <std::size_t N> std::array<float, N> floats(id value, const char *field) {
    if (![value isKindOfClass:[NSArray class]] || [(NSArray *)value count] != N)
        invalid(field);
    std::array<float, N> result{};
    for (std::size_t i = 0; i < N; ++i)
        result[i] = scalar([(NSArray *)value objectAtIndex:i], field);
    return result;
}
} // namespace

Scene parse_scene_json(std::string_view json) {
    @autoreleasepool {
        NSData *data = [NSData dataWithBytes:json.data() length:json.size()];
        NSError *error = nil;
        id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (!root)
            throw Error(ErrorCode::invalid_input,
                        "JSON parse failed: " + std::string(error.localizedDescription.UTF8String));
        NSDictionary *document = object(root, "root");
        Scene scene;
        scene.schema_version = integer(document[@"schemaVersion"], "schemaVersion");
        NSDictionary *camera = object(document[@"camera"], "camera");
        auto &c = scene.camera;
        c.width = integer(camera[@"width"], "width");
        c.height = integer(camera[@"height"], "height");
        c.view_matrix = floats<16>(camera[@"viewMatrix"], "viewMatrix");
        c.projection_matrix = floats<16>(camera[@"projectionMatrix"], "projectionMatrix");
        c.tan_fov_x = scalar(camera[@"tanFovX"], "tanFovX");
        c.tan_fov_y = scalar(camera[@"tanFovY"], "tanFovY");
        c.background = floats<3>(camera[@"background"], "background");
        id points = document[@"gaussians"];
        if (![points isKindOfClass:[NSArray class]])
            invalid("gaussians");
        if ([(NSArray *)points count] > RenderLimits {}.max_gaussians)
            throw Error(ErrorCode::resource_limit, "JSON Gaussian count exceeds default limit");
        scene.gaussians.reserve([(NSArray *)points count]);
        for (id value in (NSArray *)points) {
            NSDictionary *g = object(value, "Gaussian");
            scene.gaussians.push_back({floats<3>(g[@"mean"], "mean"),
                                       floats<6>(g[@"covariance"], "covariance"),
                                       floats<3>(g[@"color"], "color"), scalar(g[@"opacity"], "opacity")});
        }
        validate_scene(scene);
        return scene;
    }
}

Scene load_scene(const std::filesystem::path &path, std::size_t max_bytes) {
    if (!max_bytes)
        throw Error(ErrorCode::invalid_input, "max_bytes must be positive");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw Error(ErrorCode::invalid_input, "cannot open scene: " + path.string());
    const auto length = file.tellg();
    if (length < 0)
        throw Error(ErrorCode::invalid_input, "cannot determine scene length");
    if (static_cast<std::uintmax_t>(length) > max_bytes)
        throw Error(ErrorCode::resource_limit, "scene file exceeds byte limit");
    std::string json(static_cast<std::size_t>(length), '\0');
    file.seekg(0);
    if (!file.read(json.data(), static_cast<std::streamsize>(json.size())) ||
        file.peek() != std::char_traits<char>::eof())
        throw Error(ErrorCode::invalid_input, "scene file changed or could not be read fully");
    return parse_scene_json(json);
}

void write_ppm(const std::filesystem::path &path, int width, int height, const std::vector<float> &rgb) {
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192 ||
        rgb.size() != std::size_t(width) * height * 3)
        throw Error(ErrorCode::invalid_input, "invalid PPM dimensions/data length");
    std::vector<unsigned char> bytes(rgb.size());
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        if (!std::isfinite(rgb[i]))
            throw Error(ErrorCode::invalid_input, "non-finite PPM color");
        bytes[i] = static_cast<unsigned char>(std::lround(std::clamp(rgb[i], 0.0f, 1.0f) * 255));
    }
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << width << ' ' << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();
    if (!file)
        throw Error(ErrorCode::invalid_input, "failed to write PPM: " + path.string());
}
} // namespace dgr::example
