#pragma once
#include <ATen/ATen.h>
#include <tuple>
namespace dgr::torch_binding {
using at::Tensor;
using ForwardResult = std::tuple<int, Tensor, Tensor, Tensor, Tensor, Tensor>;
using BackwardResult = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
ForwardResult forward(const Tensor &bg, const Tensor &means, const Tensor &colors, const Tensor &opacity,
                      const Tensor &scales, const Tensor &rotations, double modifier, const Tensor &cov,
                      const Tensor &view, const Tensor &proj, double tanx, double tany, int height, int width,
                      const Tensor &sh, int degree, const Tensor &campos, bool prefiltered, bool debug);
BackwardResult backward(const Tensor &bg, const Tensor &means, const Tensor &radii, const Tensor &colors,
                        const Tensor &scales, const Tensor &rotations, double modifier, const Tensor &cov,
                        const Tensor &view, const Tensor &proj, double tanx, double tany, const Tensor &grad,
                        const Tensor &sh, int degree, const Tensor &campos, const Tensor &geometry,
                        int instances, const Tensor &binning, const Tensor &image, bool debug);
Tensor visible(const Tensor &means, const Tensor &view, const Tensor &proj);
} // namespace dgr::torch_binding
