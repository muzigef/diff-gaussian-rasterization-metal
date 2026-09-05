#include "api.h"
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
PYBIND11_MODULE(_C, m) {
    m.def("rasterize_gaussians", &dgr::torch_binding::forward);
    m.def("rasterize_gaussians_backward", &dgr::torch_binding::backward);
    m.def("mark_visible", &dgr::torch_binding::visible);
}
