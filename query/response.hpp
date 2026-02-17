#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace util::query {

    inline nlohmann::json request_success()
    {
        nlohmann::json response;
        response["status"] = 200;
        return response;
    }

    inline nlohmann::json request_error_malformed_query()
    {
        nlohmann::json response;
        response["status"] = 400;
        response["error-message"]
            = "Incomplete or syntactically-inaccurate request";
        return response;
    }

    inline nlohmann::json request_error_bad_argument()
    {
        nlohmann::json response;
        response["status"] = 404;
        response["error-message"] = "Bad argument";
        return response;
    }

    inline nlohmann::json request_error_internal()
    {
        nlohmann::json response;
        response["pages"] = std::vector<nlohmann::json>();
        response["status"] = 500;
        response["error-message"] = "Internal server error";
        return response;
    }

    inline nlohmann::json request_error_not_implemented(
        const std::string& error_message = std::string())
    {
        // Generate & return JSON response
        nlohmann::json response;
        response["status"] = 501;
        response["error-message"]
            = error_message.empty() ? "Method not implemented" : error_message;
        return response;
    }
} // namespace util::query
