#pragma once

#include "config.hpp"
#include "cpp-httplib/httplib.h"
#include "llm/file_utils.hpp"
#include "query/api_callback_type.hpp"
#include "query/response.hpp"
#include <mutex>
#include <nlohmann/json.hpp>
#include <utility>

namespace util::query {
    /*!*/
    class QueryServer {
    public:
        virtual ~QueryServer() = default;

        virtual void listen() = 0;

        virtual void stop() = 0;

        /*! @brief Creates socket and listens on port
         */
        virtual bool bind(int port) = 0;

        /*! @brief Creates socket and listens on port
         */
        virtual bool bind(int port, int queue_len) = 0;
    };

    template <typename DerivedType>
    class QueryServerImpl {
    public:
        virtual ~QueryServerImpl() = default;
        QueryServerImpl(
            std::string config_dir,
            Config config,
            const std::unordered_map<std::string, ApiCallbackType>& sinks);

        static void enable_cors(httplib::Response& res);

        void post_get_net_config(
            const httplib::Request&,
            httplib::Response& res) const;

        void post_set_net_config(
            const httplib::Request&,
            httplib::Response& res) const;

        void post_get_model_list(
            const httplib::Request&,
            httplib::Response& res) const;

        void post_enable_model(const httplib::Request&, httplib::Response& res);

        void post_get_file_list(const httplib::Request&, httplib::Response& res)
            const;

        void post_get_file(const httplib::Request&, httplib::Response& res)
            const;

        void post_get_file_original(
            const httplib::Request&,
            httplib::Response& res) const;

        void post_delete_files(const httplib::Request&, httplib::Response& res)
            const;

        void post_new_file(const httplib::Request&, httplib::Response& res)
            const;

        void post_calc_file_embeddings(
            const httplib::Request&,
            httplib::Response& res) const;

        void post_get_file_embeddings(
            const httplib::Request&,
            httplib::Response& res) const;

        void post_pause_files(const httplib::Request&, httplib::Response& res)
            const;

        void post_search(const httplib::Request&, httplib::Response& res) const;
    private:
        const std::string config_dir_;
        Config config_;

        mutable std::mutex api_lock_;
        std::unordered_set<std::string> pending_models_to_update_;

        const std::unordered_map<std::string, ApiCallbackType> sinks_;
    };

    template <typename DerivedType>
    QueryServerImpl<DerivedType>::QueryServerImpl(
        std::string config_dir,
        Config config,
        const std::unordered_map<std::string, ApiCallbackType>& sinks)
        : config_dir_(std::move(config_dir))
        , config_(std::move(config))
        , sinks_(sinks)
    {}

    namespace detail {

        inline std::string decode(const std::string& value)
        {
            std::string decoded_value;

            bool in_escape = false;
            for (char ch: value) {
                if (in_escape) {
                    in_escape = false;

                    switch (ch) {
                        case 't':
                        {
                            decoded_value.push_back('\t');
                            break;
                        }

                        case 'n':
                        {
                            decoded_value.push_back('\n');
                            break;
                        }

                        case 'r':
                        {
                            decoded_value.push_back('\r');
                            break;
                        }

                        case 'b':
                        {
                            decoded_value.push_back('\b');
                            break;
                        }

                        case 'f':
                        {
                            decoded_value.push_back('\f');
                            break;
                        }

                        default:
                        {
                            decoded_value.push_back(ch);
                            break;
                        }
                    }

                    continue;
                }

                if (ch == '\\') {
                    in_escape = true;
                    continue;
                }

                decoded_value.push_back(ch);
            }

            if (!decoded_value.empty() && decoded_value[0] == '\"'
                && decoded_value[decoded_value.size() - 1] == '\"') {
                auto size = static_cast<std::int64_t>(decoded_value.size());
                decoded_value = std::string(
                    decoded_value.begin() + 1,
                    decoded_value.begin() + (size - 1));
            }

            return decoded_value;
        }
    } // namespace detail

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::enable_cors(httplib::Response& res)
    {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Allow", "*");
        res.set_header(
            "Access-Control-Allow-Headers",
            "X-Requested-With, Content-Type, Accept, Origin, Authorization");
        res.set_header("Access-Control-Allow-Methods", "*");
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_get_net_config(
        const httplib::Request& /* req */,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        std::string query_port = config_["query-port"].value_or(std::string());
        std::string network_protocol
            = config_["network-protocol"].value_or(std::string());

        nlohmann::json j;
        j["query-port"] = query_port;
        j["network-protocol"] = network_protocol;

        res.set_content(j.dump(), "text/json");
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_set_net_config(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        using enum util::ConfigReturn;

        enable_cors(res);

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        switch (maybe_save_net_config(body, config_, config_dir_)) {
            case kSuccess:
            {
                res.set_content(request_success().dump(), "text/json");
                break;
            }

            case kBadArgument:
            {
                res.set_content(
                    request_error_bad_argument().dump(), "text/json");
                break;
            }

            case kNoArgument:
            {
                res.set_content(
                    request_error_malformed_query().dump(), "text/json");
                break;
            }

            case kSaveError:
            {
                res.set_content(request_error_internal().dump(), "text/json");
                break;
            }
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_get_model_list(
        const httplib::Request& /* req */,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("get-model-list");
        // Sanity check
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(nlohmann::json());
            ret["updated"] = std::vector<std::string>(
                pending_models_to_update_.begin(),
                pending_models_to_update_.end());
            res.set_content(ret.dump(), "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_enable_model(
        const httplib::Request& req,
        httplib::Response& res)
    {
        using enum util::ConfigReturn;

        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        if (!body.contains("model")) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json model = body["model"];
        if (!model.contains("model") || !model.contains("type")) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        std::string model_key = model["model"];
        std::string model_type = model["type"];

        util::ConfigReturn ret;
        if (model_type == "ocr") {
            ret = maybe_enable_ocr_model(model_key, &config_, config_dir_);
        } else if (model_type == "embedding") {
            bool model_enable = model.contains("enable")
                                && static_cast<bool>(model["enable"]);
            ret = maybe_enable_embedding_model(
                model_key, model_enable, &config_, config_dir_);
        } else {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        switch (ret) {
            case kSuccess:
            {
                pending_models_to_update_.insert(model_key);
                res.set_content(request_success().dump(), "text/json");
                break;
            }

            case kBadArgument:
            {
                res.set_content(
                    request_error_bad_argument().dump(), "text/json");
                break;
            }

            case kNoArgument:
            {
                res.set_content(
                    request_error_malformed_query().dump(), "text/json");
                break;
            }

            case kSaveError:
            {
                res.set_content(request_error_internal().dump(), "text/json");
                break;
            }
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_get_file_list(
        const httplib::Request& /* req */,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("get-file-list");
        // Sanity check
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(nlohmann::json());
            res.set_content(ret.dump(), "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_get_file(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("get-file");
        // Sanity check
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(body);
            const std::string response_body = ret.dump(
                -1, 0, false, nlohmann::detail::error_handler_t::ignore);
            res.set_content(response_body, "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_get_file_original(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("get-file-original");
        // Sanity check
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(body);
            if (ret.contains("path")) {
                std::string content
                    = util::file::fs_read_binary_file(ret["path"]);
                res.set_content(content, "application/pdf");
            } else {
                res.set_content(ret.dump(), "text/json");
            }
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_delete_files(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("delete-files");
        // Sanity check
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(body);
            res.set_content(ret.dump(), "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_new_file(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr_sink = sinks_.find("parse-file");
        // Sanity check
        if (itr_sink == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        const httplib::FormFiles& files = req.form.files;
        if (files.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json output = {};
            for (const auto& [key, value]: files) {
                nlohmann::json input;
                input["name"] = value.filename;
                input["content"] = value.content;
                output["name"] = itr_sink->second(input);
            }
            res.set_content(output.dump(), "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_calc_file_embeddings(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("calc-file-embeddings");
        // Sanity check
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(body);
            res.set_content(ret.dump(), "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_get_file_embeddings(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("get-file-embeddings");
        // Sanity check
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(body);
            const std::string response_body = ret.dump(
                -1, 0, false, nlohmann::detail::error_handler_t::ignore);
            res.set_content(response_body, "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_pause_files(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("pause-files");
        // Sanity check
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(body);
            res.set_content(ret.dump(), "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }

    template <typename DerivedType>
    void QueryServerImpl<DerivedType>::post_search(
        const httplib::Request& req,
        httplib::Response& res) const
    {
        std::scoped_lock<std::mutex> lock(api_lock_);

        enable_cors(res);

        auto itr = sinks_.find("search");
        if (itr == sinks_.end()) {
            res.set_content(
                request_error_not_implemented().dump(), "text/json");
            return;
        }

        if (req.body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        std::string decoded_body = detail::decode(req.body);
        if (decoded_body.empty()) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(decoded_body);
        } catch (std::exception&) {
            res.set_content(
                request_error_malformed_query().dump(), "text/json");
            return;
        }

        try {
            nlohmann::json ret = itr->second(body);
            res.set_content(ret.dump(), "text/json");
        } catch (std::exception&) {
            res.set_content(request_error_internal().dump(), "text/json");
        }
    }
} // namespace util::query
