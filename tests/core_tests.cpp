#include "dgr/reference.h"
#include "support.h"
#include <limits>
int main(int argc, char **argv) {
    using dgr::ErrorCode;
    return run_test(
        argc, argv,
        {{"single",
          [] {
              const auto f = dgr::render_reference(scene({gaussian()}));
              const int i = (16 * 33 + 16) * 3;
              near(f.rgb[i], 0.55, 1e-8);
              near(f.rgb[i + 1], 0.1, 1e-8);
              near(f.rgb[i + 2], 0.15, 1e-8);
              near(f.debug.projections[0].covariance_2d[0], 2.341875, 1e-6);
          }},
         {"depth_order",
          [] {
              const auto f = dgr::render_reference(scene({gaussian({0, 0, 3}, {0, 0, 1}), gaussian()}));
              const int i = (16 * 33 + 16) * 3;
              near(f.rgb[i], 0.525, 1e-8);
              near(f.rgb[i + 1], 0.05, 1e-8);
              near(f.rgb[i + 2], 0.325, 1e-8);
          }},
         {"empty",
          [] {
              const auto f = dgr::render_reference(scene({}, 1, 1));
              CHECK(f.rgb == std::vector<double>({0, 0, 0}));
              CHECK(f.debug.final_transmittance == std::vector<double>{1});
          }},
         {"validation",
          [] {
              auto s = scene({gaussian()});
              s.camera.view_matrix[0] = std::numeric_limits<float>::quiet_NaN();
              throws_code([&] { dgr::render_reference(s); }, ErrorCode::invalid_input);
              s = scene({gaussian()});
              s.gaussians[0].covariance = {1, 2, 0, 1, 0, 1};
              throws_code([&] { dgr::render_reference(s); }, ErrorCode::invalid_input);
              s = scene({});
              s.camera.width = 0;
              throws_code([&] { dgr::render_reference(s); }, ErrorCode::invalid_input);
              s = scene({});
              s.schema_version = 2;
              throws_code([&] { dgr::render_reference(s); }, ErrorCode::invalid_input);
              throws_code([] { dgr::Camera::perspective(1, 1, 0, 1); }, ErrorCode::invalid_input);
              throws_code([] { dgr::render_reference(scene({gaussian({0, 0, 2}, {1, 0, 0}, -1)})); },
                          ErrorCode::invalid_input);
          }},
         {"reference_limits", [] {
              dgr::RenderLimits l;
              l.max_pixels = 1;
              throws_code([&] { dgr::render_reference(scene({}), l); }, ErrorCode::resource_limit);
              l = {};
              l.max_instances = 2;
              throws_code([&] { dgr::render_reference(scene({gaussian()}), l); }, ErrorCode::resource_limit);
          }}});
}
