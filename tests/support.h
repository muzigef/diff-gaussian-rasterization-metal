#pragma once
#include "dgr/scene.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#define CHECK(...)                                                                                           \
    do {                                                                                                     \
        if (!(__VA_ARGS__))                                                                                  \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +                \
                                     ": " #__VA_ARGS__);                                                     \
    } while (false)
inline void near(double a, double b, double tolerance) {
    CHECK(std::isfinite(a));
    CHECK(std::isfinite(b));
    CHECK(std::abs(a - b) <= tolerance);
}
template <class F> void throws_code(F f, dgr::ErrorCode code) {
    try {
        f();
    } catch (const dgr::Error &e) {
        CHECK(e.code() == code);
        return;
    }
    throw std::runtime_error("expected error");
}
inline dgr::Gaussian gaussian(dgr::Float3 m = {0, 0, 2}, dgr::Float3 c = {1, 0, 0}, float o = 0.5f,
                              std::array<float, 6> v = {0.03f, 0, 0, 0.03f, 0, 0.03f}) {
    return {m, v, c, o};
}
inline dgr::Scene scene(std::vector<dgr::Gaussian> g, int w = 33, int h = 33) {
    return {1, dgr::Camera::perspective(w, h, 1, 1, {0.1f, 0.2f, 0.3f}), std::move(g)};
}
inline int run_test(int argc, char **argv, const std::map<std::string, std::function<void()>> &tests,
                    bool skip = false) {
    try {
        CHECK(argc == 2);
        CHECK(tests.count(argv[1]));
        tests.at(argv[1])();
        std::cout << "PASS " << argv[1] << '\n';
        return 0;
    } catch (const dgr::Error &e) {
        if (skip && e.code() == dgr::ErrorCode::unavailable) {
            std::cout << "SKIP " << e.what() << '\n';
            return 77;
        }
        std::cerr << e.what() << '\n';
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
