#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace dgr::detail {
struct alignas(16) Projected {
    std::array<float, 4> center_depth_radius;
    std::array<float, 4> conic_opacity;
    std::array<float, 4> covariance_2d;
    std::array<std::uint32_t, 4> rect;
};
using Record = std::array<std::uint32_t, 4>;
using Range = std::array<std::uint32_t, 2>;
static_assert(sizeof(float) == 4 && sizeof(std::uint32_t) == 4);
static_assert(std::is_trivially_copyable_v<Projected> && std::is_standard_layout_v<Projected>);
static_assert(sizeof(Projected) == 64 && alignof(Projected) == 16);
static_assert(offsetof(Projected, conic_opacity) == 16);
static_assert(offsetof(Projected, covariance_2d) == 32);
static_assert(offsetof(Projected, rect) == 48);
static_assert(sizeof(Record) == 16 && sizeof(Range) == 8);
} // namespace dgr::detail
