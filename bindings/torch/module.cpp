#include "api.h"
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
PYBIND11_MODULE(_C, m) {
    m.def("rasterize_gaussians", &dgr::torch_binding::forward,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("rasterize_gaussians_backward", &dgr::torch_binding::backward,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("mark_visible", &dgr::torch_binding::visible, pybind11::call_guard<pybind11::gil_scoped_release>());
}
