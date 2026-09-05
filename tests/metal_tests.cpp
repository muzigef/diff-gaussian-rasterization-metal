#include "dgr/metal_rasterizer.h"
#include "dgr/reference.h"
#include "support.h"
#include <algorithm>
#include <limits>
#include <optional>
dgr::MetalRasterizer &renderer() {
    static dgr::MetalRasterizer r;
    return r;
}
dgr::MetalFrame compare(const dgr::Scene &s) {
    const auto c = dgr::render_reference(s);
    auto f = renderer().render(s);
    const auto d = f.debug_snapshot();
    const auto rgb = f.read_rgb();
    CHECK(d.offsets == c.debug.offsets);
    CHECK(d.sorted_gaussian_ids == c.debug.sorted_gaussian_ids);
    CHECK(d.sorted_tile_ids == c.debug.sorted_tile_ids);
    CHECK(d.ranges == c.debug.ranges);
    CHECK(d.last_contributors == c.debug.last_contributors);
    CHECK(d.projections.size() == c.debug.projections.size());
    for (size_t i = 0; i < d.projections.size(); i++) {
        const auto &a = d.projections[i];
        const auto &b = c.debug.projections[i];
        CHECK(a.radius == b.radius);
        CHECK(a.rect == b.rect);
        for (int k = 0; k < 2; k++)
            near(a.center[k], b.center[k], 2e-4);
        near(a.depth, b.depth, 2e-5 * std::max(1., std::abs(b.depth)));
        for (int k = 0; k < 3; k++) {
            near(a.conic[k], b.conic[k], 5e-5 * std::max(1., std::abs(b.conic[k])));
            near(a.covariance_2d[k], b.covariance_2d[k], 5e-5 * std::max(1., std::abs(b.covariance_2d[k])));
        }
    }
    CHECK(rgb.size() == c.rgb.size());
    double squared = 0;
    for (size_t i = 0; i < rgb.size(); i++) {
        near(rgb[i], c.rgb[i], 2e-4);
        double e = rgb[i] - c.rgb[i];
        squared += e * e;
    }
    CHECK(std::sqrt(squared / rgb.size()) <= 2e-5);
    CHECK(d.final_transmittance.size() == c.debug.final_transmittance.size());
    for (size_t i = 0; i < d.final_transmittance.size(); i++)
        near(d.final_transmittance[i], c.debug.final_transmittance[i], 2e-5);
    for (auto r : d.ranges) {
        CHECK(r[0] <= r[1]);
        CHECK(r[1] <= f.stats().instances);
    }
    CHECK(f.stats().instances == c.debug.sorted_gaussian_ids.size());
    CHECK(f.native_texture_handle());
    return f;
}
int main(int argc, char **argv) {
    using dgr::ErrorCode;
    return run_test(
        argc, argv,
        {{"single_overlap",
          [] {
              compare(scene({gaussian()}));
              compare(scene({gaussian({0, 0, 3}, {0, 0, 1}), gaussian()}));
          }},
         {"empty_culled",
          [] {
              compare(scene({}, 1, 1));
              compare(scene({}));
              CHECK(compare(scene({gaussian({0, 0, .2f}), gaussian({0, 0, -2}), gaussian({100, 100, 2})}))
                        .stats()
                        .instances == 0);
          }},
         {"alpha",
          [] {
              CHECK(compare(scene({gaussian({0, 0, 2}, {1, 0, 0}, .001f)}))
                        .debug_snapshot()
                        .last_contributors[16 * 33 + 16] == 0);
              near(compare(scene({gaussian({0, 0, 2}, {1, 0, 0}, 1)}))
                       .debug_snapshot()
                       .final_transmittance[16 * 33 + 16],
                   .01, 1e-6);
              std::vector<dgr::Gaussian> p;
              for (int i = 0; i < 8; i++)
                  p.push_back(gaussian({0, 0, 2 + i * .1f}, {1, 0, 0}, .95f));
              auto d = compare(scene(p)).debug_snapshot();
              CHECK(d.last_contributors[16 * 33 + 16] == 3);
              near(d.final_transmittance[16 * 33 + 16], .000125, 1e-8);
          }},
         {"tile_boundaries",
          [] {
              compare(scene({gaussian(),
                             gaussian({-1.95f, 0, 2}, {0, 1, 0}, .5f, {.8f, .15f, .02f, .3f, .01f, .1f}),
                             gaussian({2, -1, 2}, {0, 0, 1})},
                            33, 35));
          }},
         {"equal_depth",
          [] {
              std::vector<dgr::Gaussian> p;
              for (int i = 0; i < 40; i++)
                  p.push_back(gaussian({0, 0, 2}, {i / 40.f, .1f, .5f}, .05f));
              auto s = scene(p);
              auto a = compare(s);
              CHECK(a.read_rgb() == renderer().render(s).read_rgb());
              auto d = a.debug_snapshot();
              for (auto r : d.ranges)
                  if (r[1] > r[0]) {
                      CHECK(r[1] - r[0] == 40);
                      for (unsigned i = 0; i < 40; i++)
                          CHECK(d.sorted_gaussian_ids[r[0] + i] == i);
                  }
          }},
         {"scan_sizes",
          [] {
              for (int n : {1, 2, 255, 256, 257, 513}) {
                  std::vector<dgr::Gaussian> p;
                  for (int i = 0; i < n; i++)
                      p.push_back(gaussian({0, 0, 1 + float((i * 31) % n) / n}, {1, 0, 0}, .001f));
                  compare(scene(p, 17, 19));
              }
          }},
         {"camera",
          [] {
              auto s =
                  scene({gaussian({.1f, -.2f, 2}, {1, 0, 0}, .5f, {.07f, .012f, .009f, .04f, .006f, .02f})},
                        49, 31);
              float a = .27f;
              s.camera.view_matrix = {std::cos(a), 0, -std::sin(a), 0, 0,    1,    0,   0,
                                      std::sin(a), 0, std::cos(a),  0, -.2f, .15f, .3f, 1};
              auto p = dgr::Camera::perspective(49, 31, .8f, .7f);
              s.camera.tan_fov_x = .8f;
              s.camera.tan_fov_y = .7f;
              s.camera.projection_matrix = {};
              for (int col = 0; col < 4; col++)
                  for (int row = 0; row < 4; row++)
                      for (int k = 0; k < 4; k++)
                          s.camera.projection_matrix[col * 4 + row] +=
                              p.projection_matrix[k * 4 + row] * s.camera.view_matrix[col * 4 + k];
              compare(s);
          }},
         {"seeded",
          [] {
              uint64_t state = 20260905;
              auto random = [&] {
                  state = state * 6364136223846793005ULL + 1;
                  return float((state >> 32) & 0xffffff) / 0x1000000;
              };
              std::vector<dgr::Gaussian> p;
              for (int i = 0; i < 80; i++) {
                  dgr::Float3 m{(random() - .5f) * 6, (random() - .5f) * 4, .3f + random() * 5};
                  dgr::Float3 c{random(), random(), random()};
                  float o = .05f + random() * .7f;
                  std::array<float, 6> v{.01f + random() * .08f, 0, 0,
                                         .01f + random() * .08f, 0, .01f + random() * .08f};
                  p.push_back(gaussian(m, c, o, v));
              }
              compare(scene(p, 65, 49));
          }},
         {"limits",
          [] {
              (void)renderer();
              dgr::RenderLimits l;
              l.max_instances = 2;
              dgr::MetalRasterizer r(l);
              throws_code([&] { r.render(scene({gaussian()})); }, ErrorCode::resource_limit);
              throws_code([&] { r.render(scene(std::vector<dgr::Gaussian>(513, gaussian()))); },
                          ErrorCode::resource_limit);
              l = {};
              l.max_working_bytes = 32;
              dgr::MetalRasterizer low(l);
              throws_code([&] { low.render(scene({gaussian()})); }, ErrorCode::resource_limit);
              auto s = scene({gaussian()});
              s.camera.projection_matrix[12] = std::numeric_limits<float>::max();
              throws_code([&] { renderer().render(s); }, ErrorCode::invalid_input);
          }},
         {"lifetime",
          [] {
              std::optional<dgr::MetalFrame> saved;
              std::vector<float> expected;
              {
                  dgr::MetalRasterizer r;
                  auto f = r.render(scene({gaussian()}));
                  saved = f;
                  expected = f.read_rgb();
                  r.render(scene({}));
              }
              CHECK(saved->read_rgb() == expected);
              CHECK(saved->native_texture_handle());
          }},
         {"moves",
          [] {
              dgr::MetalRasterizer a;
              dgr::MetalRasterizer b(std::move(a));
              throws_code([&] { a.render(scene({})); }, ErrorCode::invalid_input);
              auto f = b.render(scene({gaussian()}));
              auto expected = f.read_rgb();
              auto g = std::move(f);
              throws_code([&] { f.read_rgb(); }, ErrorCode::invalid_input);
              CHECK(g.read_rgb() == expected);
          }}},
        true);
}
