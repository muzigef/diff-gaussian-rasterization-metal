#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dgr {

enum class ErrorCode { invalid_input, unavailable, resource_limit, gpu };

class Error : public std::runtime_error {
  public:
    Error(ErrorCode code, const std::string &message) : std::runtime_error(message), code_(code) {}
    ErrorCode code() const noexcept {
        return code_;
    }

  private:
    ErrorCode code_;
};

using Float3 = std::array<float, 3>;
using Matrix4 = std::array<float, 16>;

struct Gaussian {
    Float3 mean{};
    // World-space positive definite covariance: xx, xy, xz, yy, yz, zz.
    std::array<float, 6> covariance{0.03f, 0, 0, 0.03f, 0, 0.03f};
    Float3 color{};    // Linear RGB; not SH coefficients.
    float opacity = 1; // Activated opacity, not a logit.
};

struct Camera {
    int width = 1;
    int height = 1;
    Matrix4 view_matrix{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    // Column-major combined world-to-clip matrix, INCLUDING the view transform.
    Matrix4 projection_matrix{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0};
    float tan_fov_x = 1;
    float tan_fov_y = 1;
    Float3 background{};

    static Camera perspective(int width, int height, float tan_fov_x = 1, float tan_fov_y = 1,
                              Float3 background = {});
};

struct Scene {
    int schema_version = 1;
    Camera camera;
    std::vector<Gaussian> gaussians;
};

struct RenderLimits {
    std::size_t max_gaussians = 100'000;
    std::size_t max_instances = 1'048'576;
    std::size_t max_pixels = 4'194'304;
    // Metal allocations per render; excludes CPU arrays, driver overhead, retained frames.
    std::size_t max_working_bytes = 256 * 1024 * 1024;
};

void validate_limits(const RenderLimits &limits);
void validate_scene(const Scene &scene, const RenderLimits &limits = {});

// Snapshot types describe semantic values, not a binary GPU or asset ABI.
struct ProjectionSnapshot {
    std::array<double, 2> center{};
    double depth = 0;
    int radius = 0;
    std::array<double, 3> conic{};
    std::array<double, 3> covariance_2d{};
    std::array<int, 4> rect{}; // min x/y, exclusive max x/y, in 16x16 software tiles.
    std::size_t tiles_touched() const noexcept {
        return static_cast<std::size_t>(rect[2] - rect[0]) * (rect[3] - rect[1]);
    }
};

struct DebugSnapshot {
    std::vector<ProjectionSnapshot> projections;
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> sorted_gaussian_ids;
    std::vector<std::uint32_t> sorted_tile_ids;
    std::vector<std::array<std::uint32_t, 2>> ranges;
    std::vector<double> final_transmittance;
    // One-based position of last accepted candidate, NOT the count of accepted candidates.
    std::vector<std::uint32_t> last_contributors;
};

} // namespace dgr
