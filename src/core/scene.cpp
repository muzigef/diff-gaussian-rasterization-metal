#include "dgr/scene.h"
#include <algorithm>
#include <cmath>

namespace dgr {
namespace {
template <class T> bool finite(const T &values) {
    return std::all_of(values.begin(), values.end(), [](auto x) { return std::isfinite(x); });
}
} // namespace

Camera Camera::perspective(int w, int h, float tx, float ty, Float3 bg) {
    if (!std::isfinite(tx) || !std::isfinite(ty) || tx <= 0 || ty <= 0)
        throw Error(ErrorCode::invalid_input, "FOV tangents must be finite and positive");
    Camera c;
    c.width = w;
    c.height = h;
    c.tan_fov_x = tx;
    c.tan_fov_y = ty;
    c.background = bg;
    c.projection_matrix[0] = 1 / tx;
    c.projection_matrix[5] = 1 / ty;
    return c;
}

void validate_limits(const RenderLimits &l) {
    constexpr std::size_t cap = 16'777'216;
    if (!l.max_gaussians || l.max_gaussians > cap || !l.max_instances || l.max_instances > cap ||
        !l.max_pixels || l.max_pixels > cap || !l.max_working_bytes)
        throw Error(ErrorCode::invalid_input, "invalid render limits");
}

void validate_scene(const Scene &s, const RenderLimits &l) {
    validate_limits(l);
    if (s.schema_version != 1)
        throw Error(ErrorCode::invalid_input, "unsupported scene schema version");
    const auto &c = s.camera;
    if (c.width <= 0 || c.height <= 0 || c.width > 8192 || c.height > 8192)
        throw Error(ErrorCode::invalid_input, "width and height must be in 1...8192");
    if (static_cast<std::size_t>(c.width) * c.height > l.max_pixels || s.gaussians.size() > l.max_gaussians)
        throw Error(ErrorCode::resource_limit, "pixel or Gaussian count exceeds limits");
    if (!finite(c.view_matrix) || !finite(c.projection_matrix) || !finite(c.background) ||
        !std::isfinite(c.tan_fov_x) || !std::isfinite(c.tan_fov_y) || c.tan_fov_x <= 0 || c.tan_fov_y <= 0 ||
        !std::isfinite(c.width / (2 * c.tan_fov_x)) || !std::isfinite(c.height / (2 * c.tan_fov_y)))
        throw Error(ErrorCode::invalid_input, "invalid camera matrices/background/FOV");
    for (std::size_t i = 0; i < s.gaussians.size(); ++i) {
        const auto &g = s.gaussians[i];
        const auto context = "Gaussian " + std::to_string(i) + ": ";
        if (!finite(g.mean) || !finite(g.covariance) || !finite(g.color) || !std::isfinite(g.opacity) ||
            g.opacity < 0 || g.opacity > 1)
            throw Error(ErrorCode::invalid_input, context + "expected finite values and opacity in [0,1]");
        std::array<double, 6> v{};
        std::copy(g.covariance.begin(), g.covariance.end(), v.begin());
        const double minor = v[0] * v[3] - v[1] * v[1];
        const double det = v[0] * (v[3] * v[5] - v[4] * v[4]) - v[1] * (v[1] * v[5] - v[4] * v[2]) +
                           v[2] * (v[1] * v[4] - v[3] * v[2]);
        if (!(v[0] > 0 && minor > 0 && det > 0))
            throw Error(ErrorCode::invalid_input, context + "covariance must be positive definite");
    }
}
} // namespace dgr
