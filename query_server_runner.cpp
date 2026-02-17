#include "query_server_runner.hpp"
#include "config.hpp"
#include "database/sql.hpp"
#include "file/misc_utils.hpp"
#include "log/log.hpp"
#include "query/api_callback_type.hpp"
#include "query/http_query_server.hpp"
#include "query/query_server.hpp"
#include "query/query_sink.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace {

    std::shared_ptr<util::query::QueryServer> generate_query_server(
        const std::string& config_dir,
        const util::Config& config,
        const std::unordered_map<std::string, util::query::ApiCallbackType>&
            sinks)
    {
        // Sanity check
        const auto protocol = config["network-protocol"];
        if (protocol != "http") {
            return nullptr;
        }

        const auto port = config["query-port"];
        if (!port) {
            return nullptr;
        }

        auto server = std::make_shared<util::query::HttpQueryServer>(
            config_dir, config, sinks);

        server->bind(std::stoi(port.value()));

        LOG_INF(
            "query server (%s) bound to port %s",
            protocol->c_str(),
            port->c_str());
        return server;
    }
} // namespace

std::shared_ptr<util::query::QueryServerRunner> util::query::QueryServerRunner::
    initialize(
        const std::shared_ptr<util::file::FileIndexRunner>& indexer,
        const std::string& config_dir,
        const util::Config& config,
        const util::file::SqlDatabase::Params& database_params)
{
    auto status = std::make_shared<util::file::UpdateNotifierSink>();
    auto query_sink
        = std::make_shared<QuerySink>(indexer, config, status, database_params);

    std::unordered_map<std::string, ApiCallbackType> sinks;

    sinks["get-model-list"] = [query_sink](const nlohmann::json& j) {
        return query_sink->get_model_list(j);
    };

    sinks["get-file-list"] = [query_sink](const nlohmann::json& j) {
        return query_sink->get_file_list(j);
    };

    sinks["get-file"] = [query_sink](const nlohmann::json& j) {
        return query_sink->get_file(j);
    };

    sinks["get-file-original"] = [query_sink](const nlohmann::json& j) {
        return query_sink->get_file_path(j);
    };

    sinks["delete-files"] = [query_sink](const nlohmann::json& j) {
        return query_sink->delete_files(j);
    };

    sinks["parse-file"] = [query_sink](const nlohmann::json& j) {
        return query_sink->parse_file(j);
    };

    sinks["calc-file-embeddings"] = [query_sink](const nlohmann::json& j) {
        return query_sink->calc_file_embeddings(j);
    };

    sinks["get-file-embeddings"] = [query_sink](const nlohmann::json& j) {
        return query_sink->get_file_embeddings(j);
    };

    sinks["pause-files"] = [query_sink](const nlohmann::json& j) {
        return query_sink->pause_files(j);
    };

    sinks["search"] = [query_sink](const nlohmann::json& j) {
        return query_sink->search(j);
    };

    // Server type
    std::shared_ptr<QueryServer> query_server
        = generate_query_server(config_dir, config, sinks);

    if (!query_server) {
        return nullptr;
    }

    return std::make_shared<QueryServerRunner>(query_server, query_sink);
}

util::query::QueryServerRunner::QueryServerRunner(
    const std::shared_ptr<QueryServer>& query_server,
    const std::shared_ptr<QuerySink>& sink)
    : sink_(sink)
    , query_server_(query_server)
{}

void util::query::QueryServerRunner::run() const
{
    LOG_INF("query server listening");

    query_server_->listen();
}

void util::query::QueryServerRunner::stop() const
{
    query_server_->stop();
}
