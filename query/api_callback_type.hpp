#pragma once

#include <functional>
#include <nlohmann/json.hpp>

namespace util::query {
    using ApiCallbackType
        = std::function<nlohmann::json(const nlohmann::json&)>;
}
