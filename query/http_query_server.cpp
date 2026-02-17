#include "query/http_query_server.hpp"
#include "cpp-httplib/httplib.h"
#include "file/misc_utils.hpp"
#include <string>

util::query::HttpQueryServer::HttpQueryServer(
    const std::string& config_dir,
    const Config& config,
    const std::unordered_map<std::string, ApiCallbackType>& sinks)
    : QueryServerImpl<HttpQueryServer>(config_dir, config, sinks)
{
    // Handle CORS Preflight (OPTIONS) requests globally
    server_.Options(
        R"(/(.*))",
        [](const httplib::Request& /* req */, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Requested-With");
        // Cache preflight for 24 hours
        res.set_header("Access-Control-Max-Age", "86400");
        res.status = 200;
    });

    server_.set_file_request_handler(
        [](const httplib::Request& /* req */, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
    });

    server_.Post(
        "/net/get",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_get_net_config(req, res);
    });

    server_.Post(
        "/net/set",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_set_net_config(req, res);
    });

    // Handles model metadata request
    server_.Post(
        "/model/list",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_get_model_list(req, res);
    });

    server_.Post(
        "/model/enable",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_enable_model(req, res);
    });

    // Handles file metadata request
    server_.Post(
        "/file/list",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_get_file_list(req, res);
    });

    // Handles file download
    server_.Post(
        "/file/get",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_get_file(req, res);
    });

    // Handles file download
    server_.Post(
        "/file/get/original",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_get_file_original(req, res);
    });

    // Handles file delete
    server_.Post(
        "/file/delete",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_delete_files(req, res);
    });

    // Handles new file parse
    server_.Post(
        "/file/new",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_new_file(req, res);
    });

    // Handles file embeddings calculation
    server_.Post(
        "/file/embeddings",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_calc_file_embeddings(req, res);
    });

    // Handles file embeddings download
    server_.Post(
        "/file/embeddings/download",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_get_file_embeddings(req, res);
    });

    // Handles file parse stop
    server_.Post(
        "/file/pause",
        [this](const httplib::Request& req, httplib::Response& res) {
        post_pause_files(req, res);
    });

    // Handles keyword search
    server_.Post(
        "/search", [this](const httplib::Request& req, httplib::Response& res) {
        post_search(req, res);
    });
}
