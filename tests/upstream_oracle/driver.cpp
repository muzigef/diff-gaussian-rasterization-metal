// Test-only host scheduling around directly extracted Inria CUDA mathematical bodies.
extern "C" void oracle_preprocess(int P, int D, int M, int W, int H, float tx, float ty, float modifier,
                                  const float *means, const float *scales, const float *rotations,
                                  const float *opacity, const float *sh, const float *colors,
                                  const float *cov, const float *view, const float *proj, const float *camera,
                                  int *radii, float *centers, float *depths, float *covout, float *rgb,
                                  float *conic, uint32_t *counts, bool *clamped, uint32_t *rects,
                                  float *cov2d) {
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    for (int i = 0; i < P; ++i) {
        oracle_index = i;
        upstream_forward::preprocessCUDA<3>(P, D, M, means, (const glm::vec3 *)scales, modifier,
                                            (const glm::vec4 *)rotations, opacity, sh, clamped, cov, colors,
                                            view, proj, (const glm::vec3 *)camera, W, H, tx, ty, W / (2 * tx),
                                            H / (2 * ty), radii, (float2 *)centers, depths, covout, rgb,
                                            (float4 *)conic, grid, counts, false);
        if (radii[i] > 0) {
            uint2 lo, hi;
            getRect(((float2 *)centers)[i], radii[i], lo, hi, grid);
            rects[4 * i] = lo.x;
            rects[4 * i + 1] = lo.y;
            rects[4 * i + 2] = hi.x;
            rects[4 * i + 3] = hi.y;
            auto v = upstream_forward::computeCov2D(((float3 *)means)[i], W / (2 * tx), H / (2 * ty), tx, ty,
                                                    (cov ? cov : covout) + i * 6, view);
            cov2d[3 * i] = v.x;
            cov2d[3 * i + 1] = v.y;
            cov2d[3 * i + 2] = v.z;
        }
    }
}

extern "C" int64_t oracle_bin(int P, int W, int H, const float *centers, const int *radii,
                              const float *depths, uint32_t *ids, uint32_t *ranges, uint64_t capacity) {
    struct Record {
        uint64_t key;
        uint32_t id;
    };
    std::vector<Record> records;
    records.reserve(capacity);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    for (int i = 0; i < P; i++)
        if (radii[i] > 0) {
            uint2 lo, hi;
            getRect(((float2 *)centers)[i], radii[i], lo, hi, grid);
            uint32_t bits;
            std::memcpy(&bits, depths + i, 4);
            for (uint32_t y = lo.y; y < hi.y; y++)
                for (uint32_t x = lo.x; x < hi.x; x++)
                    records.push_back({(uint64_t(y * grid.x + x) << 32) | bits, uint32_t(i)});
        }
    if (records.size() > capacity)
        return -1;
    std::stable_sort(records.begin(), records.end(), [](auto a, auto b) { return a.key < b.key; });
    std::memset(ranges, 0, size_t(grid.x) * grid.y * 8);
    for (size_t i = 0; i < records.size(); i++) {
        ids[i] = records[i].id;
        uint32_t tile = records[i].key >> 32;
        if (i == 0 || (records[i - 1].key >> 32) != tile)
            ranges[2 * tile] = i;
        if (i + 1 == records.size() || (records[i + 1].key >> 32) != tile)
            ranges[2 * tile + 1] = i + 1;
    }
    return records.size();
}

extern "C" void oracle_render(int W, int H, const uint32_t *ids, const uint32_t *raw_ranges,
                              const float *raw_centers, const float *features, const float *raw_conic,
                              const float *bg_color, float *final_T, uint32_t *n_contrib, float *out_color) {
    constexpr int CHANNELS = 3;
    const auto *point_list = ids;
    const auto *ranges = (const uint2 *)raw_ranges;
    const auto *points_xy_image = (const float2 *)raw_centers;
    const auto *conic_opacity = (const float4 *)raw_conic;
    auto worker = [&](int rank) {
        for (int pix_id = rank; pix_id < W * H; pix_id += 8) {
            float2 pixf(float(pix_id % W), float(pix_id / W));
            auto range = ranges[(pix_id / W / 16) * ((W + 15) / 16) + (pix_id % W / 16)];
            float T = 1.0f, C[CHANNELS] = {0};
            bool done = false;
            uint32_t contributor = 0, last_contributor = 0;
            for (uint32_t index = range.x; !done && index < range.y; index++) {
                /* FORWARD_PIXEL_BODY */
            }
            final_T[pix_id] = T;
            n_contrib[pix_id] = last_contributor;
            for (int ch = 0; ch < CHANNELS; ch++)
                out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
        }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; i++)
        threads.emplace_back(worker, i);
    for (auto &thread : threads)
        thread.join();
}

extern "C" void oracle_backward_render(int W, int H, const uint32_t *point_list, const uint32_t *raw_ranges,
                                       const float *raw_centers, const float *colors, const float *raw_conic,
                                       const float *bg_color, const float *final_Ts,
                                       const uint32_t *n_contrib, const float *dL_dpixels, float *raw_mean2d,
                                       float *raw_gradconic, float *dL_dopacity, float *dL_dcolors) {
    constexpr int C = 3;
    const auto *ranges = (const uint2 *)raw_ranges;
    const auto *points_xy_image = (const float2 *)raw_centers;
    const auto *conic_opacity = (const float4 *)raw_conic;
    auto *dL_dmean2D = (float3 *)raw_mean2d;
    auto *dL_dconic2D = (float4 *)raw_gradconic;
    for (int pix_id = 0; pix_id < W * H; pix_id++) {
        float2 pixf(float(pix_id % W), float(pix_id / W));
        auto range = ranges[(pix_id / W / 16) * ((W + 15) / 16) + (pix_id % W / 16)];
        const float T_final = final_Ts[pix_id];
        float T = T_final;
        uint32_t contributor = range.y - range.x;
        const int last_contributor = n_contrib[pix_id];
        float accum_rec[C] = {0}, dL_dpixel[C], last_alpha = 0, last_color[C] = {0};
        const float ddelx_dx = 0.5 * W, ddely_dy = 0.5 * H;
        for (int ch = 0; ch < C; ch++)
            dL_dpixel[ch] = dL_dpixels[ch * W * H + pix_id];
        // Sparse-loss audit: exactly zero upstream pixel gradients contribute nothing for finite inputs.
        if (dL_dpixel[0] == 0 && dL_dpixel[1] == 0 && dL_dpixel[2] == 0)
            continue;
        for (int64_t index = int64_t(range.y) - 1; index >= range.x; index--) {
            /* BACKWARD_PIXEL_BODY */
        }
    }
}

extern "C" void oracle_backward_preprocess(int P, int D, int M, int W, int H, float tx, float ty,
                                           float modifier, const float *means, const int *radii,
                                           const float *cov, const float *sh, const float *scales,
                                           const float *rotations, const bool *clamped, const float *view,
                                           const float *proj, const float *camera, const float *gm2,
                                           const float *gconic, float *gcolor, float *gm3, float *gcov,
                                           float *gsh, float *gscale, float *grot) {
    for (int i = 0; i < P; i++) {
        oracle_index = i;
        upstream_backward::computeCov2DCUDA(P, (const float3 *)means, radii, cov, W / (2 * tx), H / (2 * ty),
                                            tx, ty, view, gconic, (float3 *)gm3, gcov);
        upstream_backward::preprocessCUDA<3>(
            P, D, M, (const float3 *)means, radii, sh, clamped, (const glm::vec3 *)scales,
            (const glm::vec4 *)rotations, modifier, proj, (const glm::vec3 *)camera, (const float3 *)gm2,
            (glm::vec3 *)gm3, gcolor, gcov, gsh, (glm::vec3 *)gscale, (glm::vec4 *)grot);
    }
}

extern "C" void oracle_trace(int x, int y, int W, const uint32_t *point_list, const uint32_t *raw_ranges,
                             const float *raw_centers, const float *raw_conic, float *trace) {
    const auto *ranges = (const uint2 *)raw_ranges;
    const auto *points = (const float2 *)raw_centers;
    const auto *conics = (const float4 *)raw_conic;
    auto range = ranges[(y / 16) * ((W + 15) / 16) + x / 16];
    float T = 1;
    bool done = false;
    for (uint32_t i = range.x; i < range.y; i++) {
        uint32_t id = point_list[i];
        auto xy = points[id];
        auto co = conics[id];
        float2 d = {xy.x - x, xy.y - y};
        float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
        float alpha = min(0.99f, co.w * exp(power));
        float next = T * (1 - alpha);
        int action = 0;
        if (!done) {
            if (power > 0)
                action = 1;
            else if (alpha < 1.0f / 255.0f)
                action = 2;
            else if (next < 0.0001f) {
                action = 3;
                done = true;
            } else
                action = 4;
        }
        float *row = trace + (i - range.x) * 6;
        row[0] = id;
        row[1] = power;
        row[2] = alpha;
        row[3] = T;
        row[4] = next;
        row[5] = action;
        if (action == 4)
            T = next;
    }
}
