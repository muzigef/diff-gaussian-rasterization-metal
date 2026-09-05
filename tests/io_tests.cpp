#include "scene_io.h"
#include "support.h"
#include <fstream>
#include <iterator>
int main(int argc, char **argv) {
    try {
        CHECK(argc == 2);
        auto s = dgr::example::load_scene(argv[1]);
        CHECK(s.gaussians.size() == 2);
        CHECK(s.camera.width == 65);
        std::ifstream f(argv[1]);
        const std::string json{std::istreambuf_iterator<char>(f), {}};
        auto reject = [&](std::string from, std::string to) {
            auto j = json;
            auto p = j.find(from);
            CHECK(p != std::string::npos);
            j.replace(p, from.size(), to);
            throws_code([&] { dgr::example::parse_scene_json(j); }, dgr::ErrorCode::invalid_input);
        };
        reject("\"schemaVersion\": 1", "\"schemaVersion\": 2");
        reject("\"width\": 65", "\"width\": 65.5");
        reject("\"width\": 65", "\"width\": true");
        reject("\"width\": 65", "\"width\": \"65\"");
        reject("\"mean\": [0.2,0.0,3]", "\"mean\": [0.2,0.0]");
        reject("\"opacity\": 0.85", "\"opacity\": 1e100");
        throws_code([] { dgr::example::parse_scene_json("{}"); }, dgr::ErrorCode::invalid_input);
        throws_code([&] { dgr::example::load_scene(argv[1], 4); }, dgr::ErrorCode::resource_limit);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
