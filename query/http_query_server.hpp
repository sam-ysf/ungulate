#pragma once

#include "api_callback_type.hpp"
#include "config.hpp"
#include "cpp-httplib/httplib.h"
#include "query/query_server.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace util::query {

    class HttpQueryServer : public QueryServer,
                            public QueryServerImpl<HttpQueryServer> {
    public:
        HttpQueryServer(
            const std::string& config_dir,
            const Config& config,
            const std::unordered_map<std::string, ApiCallbackType>& sinks);

        //! @brief Enters run loop
        void listen() override
        {
            server_.listen_after_bind();
        }

        //! @brief Stops run loop
        void stop() override
        {
            server_.stop();
        }

        bool bind(int port) override
        {
            return server_.bind_to_port("0.0.0.0", port);
        }

        bool bind(int port, int /* queue_len */) override
        {
            return server_.bind_to_port("0.0.0.0", port);
        }
    private:
        httplib::Server server_;
    };
} // namespace util::query
