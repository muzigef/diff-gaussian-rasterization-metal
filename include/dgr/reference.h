#pragma once
#include "dgr/scene.h"

namespace dgr {

struct ReferenceFrame {
    int width = 0;
    int height = 0;
    std::vector<double> rgb; // Interleaved linear RGB, no display transform.
    DebugSnapshot debug;
};

// Small-scene, scalar Float64 correctness oracle; no Apple/GPU dependency.
ReferenceFrame render_reference(const Scene &scene, const RenderLimits &limits = {});

} // namespace dgr
