#include "config.hpp"
#include "llm/configs.hpp"
#include "llm/file_utils.hpp"
#include "log/log.hpp"
#include <algorithm>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <llama.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace {
    /*! Helper, config load
     */
    template <typename T>
    inline std::optional<T> maybe_load_field(
        const std::string& key,
        const nlohmann::json& source);

    /*! Helper, config load
     */
    template <>
    inline std::optional<bool> maybe_load_field<bool>(
        const std::string& key,
        const nlohmann::json& source)
    {
        bool value = false;
        auto j = source.find(key);
        if (j == source.end()) {
            return std::nullopt;
        }

        if (const auto& v = j.value(); v.is_boolean()) {
            value = static_cast<bool>(v);
        }

        return value;
    }

    /*! Helper, config load
     */
    template <>
    inline std::optional<std::string> maybe_load_field<std::string>(
        const std::string& key,
        const nlohmann::json& source)
    {
        std::string value;
        auto j = source.find(key);
        if (j == source.end()) {
            return std::nullopt;
        }

        if (const auto& v = j.value(); v.is_string()) {
            value = static_cast<std::string>(v);
        }

        return value;
    }

    /*! Helper, config load
     */
    template <>
    inline std::optional<std::vector<std::string>> maybe_load_field<
        std::vector<std::string>>(
        const std::string& key,
        const nlohmann::json& source)
    {
        std::vector<std::string> value;

        const auto j = source.find(key);
        if (j == source.end() || !j->is_array()) {
            return std::nullopt;
        }

        const auto& vec = j.value();
        for (auto itr = vec.begin(); itr != vec.end(); ++itr) {
            value.push_back(itr.value());
        }

        return value;
    }

    /*! Helper, config load
     */
    template <>
    inline std::optional<std::unordered_map<std::string, std::string>>
    maybe_load_field<std::unordered_map<std::string, std::string>>(
        const std::string& key,
        const nlohmann::json& source)
    {
        const auto j = source.find(key);
        if (j == source.end()) {
            return std::nullopt;
        }

        return static_cast<std::unordered_map<std::string, std::string>>(
            j.value());
    }

    /*! Helper, config load
     */
    template <>
    inline std::optional<std::vector<util::file::OcrModelConfig>>
    maybe_load_field<std::vector<util::file::OcrModelConfig>>(
        const std::string& key,
        const nlohmann::json& source)
    {
        const auto j = source.find(key);
        if (j == source.end()) {
            return std::nullopt;
        }

        std::vector<util::file::OcrModelConfig> value;

        const auto& j_vec = j.value();
        for (const auto& j_vec_i: j_vec) {
            std::optional<std::string> prompt
                = maybe_load_field<std::string>("prompt", j_vec_i);
            std::optional<std::string> gguf
                = maybe_load_field<std::string>("gguf", j_vec_i);
            std::optional<std::string> mmproj
                = maybe_load_field<std::string>("mmproj", j_vec_i);
            std::optional<std::string> description
                = maybe_load_field<std::string>("description", j_vec_i);

            std::optional<std::unordered_map<std::string, std::string>> params
                = maybe_load_field<
                    std::unordered_map<std::string, std::string>>(
                    "params", j_vec_i);

            std::optional<bool> enabled
                = maybe_load_field<bool>("enabled", j_vec_i);

            std::optional<bool> clear_cache
                = maybe_load_field<bool>("clear-cache", j_vec_i);

            std::optional<bool> verbose_parse_result
                = maybe_load_field<bool>("verbose-parse-result", j_vec_i);

            if (!prompt || !gguf || !mmproj) {
                return std::nullopt;
            }

            LOG_INF("calculating %s checksum", gguf.value().c_str());

            std::string hash = util::file::fs_calculate_hash(gguf.value());

            value.push_back(
                util::file::OcrModelConfig{
                    .gguf = gguf.value(),
                    .gguf_hash = hash,
                    .mmproj = mmproj.value(),
                    .prompt = prompt.value(),
                    .description = description.value_or(std::string()),
                    .enabled = enabled.value_or(false),
                    .clear_cache = clear_cache.value_or(false),
                    .verbose_parse_result
                    = verbose_parse_result.value_or(false),
                    .params = params.value_or(
                        std::unordered_map<std::string, std::string>())});
        }

        return value;
    }

    /*! Helper, config load
     */
    template <>
    inline std::optional<std::vector<util::file::PostprocessingConfig>>
    maybe_load_field<std::vector<util::file::PostprocessingConfig>>(
        const std::string& key,
        const nlohmann::json& source)
    {
        std::vector<util::file::PostprocessingConfig> value;

        const auto j = source.find(key);
        if (j == source.end() || !j->is_array()) {
            return std::nullopt;
        }

        const auto& j_vec = j.value();
        for (const auto& j_vec_i: j_vec) {
            std::optional<std::string> gguf
                = maybe_load_field<std::string>("gguf", j_vec_i);
            std::optional<std::string> prompt
                = maybe_load_field<std::string>("prompt", j_vec_i);
            std::optional<std::string> description
                = maybe_load_field<std::string>("description", j_vec);

            std::optional<bool> enabled
                = maybe_load_field<bool>("enabled", j_vec_i);

            std::optional<bool> clear_cache
                = maybe_load_field<bool>("clear-cache", j_vec);

            std::optional<std::unordered_map<std::string, std::string>> params
                = maybe_load_field<
                    std::unordered_map<std::string, std::string>>(
                    "params", j_vec);

            if (!gguf || !prompt) {
                continue;
            }

            LOG_INF("calculating %s checksum", gguf.value().c_str());

            std::string hash = util::file::fs_calculate_hash(gguf.value());

            util::file::PostprocessingConfig v
                = {.gguf = gguf.value(),
                   .gguf_hash = hash,
                   .prompt = prompt.value(),
                   .description = description.value_or(std::string()),
                   .enabled = enabled.value_or(false),
                   .clear_cache = clear_cache.value_or(false),
                   .params = params.value_or(
                       std::unordered_map<std::string, std::string>())};

            value.push_back(std::move(v));
        }

        return value;
    }

    /*! Helper, config load
     */
    template <>
    inline std::optional<std::vector<util::file::EmbeddingConfig>>
    maybe_load_field<std::vector<util::file::EmbeddingConfig>>(
        const std::string& key,
        const nlohmann::json& source)
    {
        std::vector<util::file::EmbeddingConfig> value;

        const auto j = source.find(key);
        if (j == source.end() || !j->is_array()) {
            return std::nullopt;
        }

        const auto& j_vec = j.value();
        for (const auto& j_vec_i: j_vec) {
            std::optional<std::string> gguf
                = maybe_load_field<std::string>("gguf", j_vec_i);
            std::optional<bool> embeddings_only
                = maybe_load_field<bool>("embeddings-only", j_vec_i);
            std::optional<std::string> description
                = maybe_load_field<std::string>("description", j_vec_i);

            std::optional<bool> enabled
                = maybe_load_field<bool>("enabled", j_vec_i);

            std::optional<bool> clear_cache
                = maybe_load_field<bool>("clear-cache", j_vec_i);

            std::optional<std::vector<std::string>> expressions
                = maybe_load_field<std::vector<std::string>>("chunks", j_vec_i);

            std::optional<std::unordered_map<std::string, std::string>> params
                = maybe_load_field<
                    std::unordered_map<std::string, std::string>>(
                    "params", j_vec_i);

            if (!gguf || !expressions) {
                continue;
            }

            LOG_INF("calculating %s checksum", gguf.value().c_str());

            std::string hash = util::file::fs_calculate_hash(gguf.value());

            util::file::EmbeddingConfig v
                = {.gguf = gguf.value(),
                   .gguf_hash = hash,
                   .embeddings_only = embeddings_only.value_or(true),
                   .description = description.value_or(std::string()),
                   .enabled = enabled.value_or(false),
                   .chunks = expressions.value(),
                   .clear_cache = clear_cache.value_or(false),
                   .params = params.value_or(
                       std::unordered_map<std::string, std::string>())};

            value.push_back(std::move(v));
        }

        return value;
    }
} // namespace

namespace {
    const std::string kDefaultDashboardPort = "6767";
    const std::string kDefaultDbName = "db.sqlite";
    const std::string kDefaultNetworkProtocol = "http";
    const std::string kDefaultQueryPort = "6868";

    inline bool save_config(
        const std::filesystem::path& path,
        const std::unordered_map<std::string, std::string>& config)
    {
        nlohmann::json json_config;

        for (const auto& [key, value]: config) {
            json_config[key] = value;
        }

        std::ofstream ofs(path);
        if (!ofs.is_open()) {
            return false;
        }

        ofs << json_config.dump(2);
        ofs.close();

        return true;
    }
} // namespace

namespace {

    inline nlohmann::json load_config(
        const std::filesystem::path& dir,
        const std::string& filename)
    {
        std::ifstream ifs(dir / filename);
        if (!ifs.is_open()) {
            return nlohmann::json();
        }

        nlohmann::json config = nlohmann::json::parse(ifs);
        ifs.close();

        return config;
    };

    inline util::Config load_database_config(
        const std::filesystem::path& config_dir)
    {
        const nlohmann::json j = load_config(config_dir, "db.json");

        util::Config config;
        // Default opt.
        config.global_params["db-name"] = kDefaultDbName;

        if (auto db_name = maybe_load_field<std::string>("db-name", j)) {
            config.global_params["db-name"] = db_name.value();
        } else {
            // If no existing configuration exists, save defaults to new file
            save_config(config_dir / "db.json", config.global_params);
        }

        config.global_params["db-path"]
            = config_dir / config.global_params["db-name"];
        return config;
    }

    // Network protocol params
    inline util::Config load_net_config(const std::filesystem::path& config_dir)
    {
        const nlohmann::json j = load_config(config_dir, "net.json");

        util::Config config;
        // Default opt.
        config.global_params["dashboard-port"] = kDefaultDashboardPort;
        // Default opt.
        config.global_params["network-protocol"] = kDefaultNetworkProtocol;
        // Default opt.
        config.global_params["query-port"] = kDefaultQueryPort;

        auto dashboard_port
            = maybe_load_field<std::string>("dashboard-port", j);
        if (dashboard_port && !dashboard_port->empty()) {
            config.global_params["dashboard-port"] = dashboard_port.value();
        }

        auto network_protocol
            = maybe_load_field<std::string>("network-protocol", j);
        if (network_protocol && !network_protocol->empty()) {
            config.global_params["network-protocol"] = network_protocol.value();
        }

        auto query_port = maybe_load_field<std::string>("query-port", j);
        if (query_port && !query_port->empty()) {
            config.global_params["query-port"] = query_port.value();
        }

        // If no existing configuration exists, save defaults to new file
        if ((!dashboard_port || dashboard_port->empty())
            || (!network_protocol || network_protocol->empty())
            || (!query_port || query_port->empty())) {
            save_config(config_dir / "net.json", config.global_params);
        }

        return config;
    }

    inline util::Config load_model_config(
        const std::filesystem::path& config_dir)
    {
        const nlohmann::json j = load_config(config_dir, "model.json");

        util::Config config;

        auto models = maybe_load_field<std::vector<util::file::OcrModelConfig>>(
            "ocr", j);
        if (!models) {
            return config;
        }

        config.ocr_model_configs = std::move(models.value());
        if (std::ranges::none_of(
                config.ocr_model_configs,
                [](const util::file::OcrModelConfig& value) {
            return value.enabled;
        })) {
            config.ocr_model_configs.front().enabled = true;
        }

        if (auto prompt_configs
            = maybe_load_field<std::vector<util::file::PostprocessingConfig>>(
                "postprocessing", j)) {
            config.post_processing_configs = std::move(prompt_configs.value());
        }

        if (auto embedding_configs
            = maybe_load_field<std::vector<util::file::EmbeddingConfig>>(
                "embeddings", j)) {
            config.embedding_configs = std::move(embedding_configs.value());
        }

        return config;
    }
} // namespace

util::Config util::load_config(const std::string& dir_path)
{
    // Create configuration directory (if it does not already exist)
    std::filesystem::path config_dir(dir_path);
    if (!std::filesystem::exists(config_dir)) {
        std::filesystem::create_directories(config_dir);
    }

    Config net = load_net_config(config_dir);
    Config database = load_database_config(config_dir);
    Config model = load_model_config(config_dir);

    return net + database + model;
}

std::optional<std::string> util::Config::operator[](
    const std::string& key) const
{
    if (global_params.contains(key))
        return global_params.at(key);
    return std::nullopt;
}

bool util::Config::has_param(const std::string& key) const
{
    return global_params.contains(key);
}

util::Config& util::Config::operator+=(const Config& rhs)
{
    global_params.insert(rhs.global_params.begin(), rhs.global_params.end());

    ocr_model_configs.insert(
        ocr_model_configs.end(),
        rhs.ocr_model_configs.begin(),
        rhs.ocr_model_configs.end());

    post_processing_configs.insert(
        post_processing_configs.end(),
        rhs.post_processing_configs.begin(),
        rhs.post_processing_configs.end());

    embedding_configs.insert(
        embedding_configs.end(),
        rhs.embedding_configs.begin(),
        rhs.embedding_configs.end());

    return *this;
}

nlohmann::json util::Config::to_json() const
{
    nlohmann::json value;

    value["global"] = global_params;
    value["ocr"] = {};
    value["embeddings"] = {};

    for (const auto& config: ocr_model_configs) {
        value["ocr"].push_back(config.to_json());
    }

    for (const auto& config: embedding_configs) {
        value["embeddings"].push_back(config.to_json());
    }

    return value;
}

util::ConfigReturn util::maybe_save_net_config(
    const nlohmann::json& json_config,
    const Config& old_config,
    const std::string& dir_path)
{
    using enum ConfigReturn;

    if (!json_config.contains("network-protocol")) {
        return kNoArgument;
    }

    if (!json_config.contains("query-port")) {
        return kNoArgument;
    }

    if (static_cast<std::string>(json_config["network-protocol"])
            == old_config["network-protocol"]
        && static_cast<std::string>(json_config["query-port"])
               == old_config["query-port"]) {
        return kSuccess;
    }

    const auto path_root = std::filesystem::path(dir_path) / "net.json";

    const std::unordered_map<std::string, std::string> params
        = {{"network-protocol", json_config["network-protocol"]},
           {"query-port", json_config["query-port"]}};

    return ::save_config(path_root, params) ? kSuccess : kSaveError;
}

util::ConfigReturn util::maybe_save_model_config(
    const nlohmann::json& json_config,
    const std::string& dir_path)
{
    if (const auto ocr_config
        = maybe_load_field<std::vector<file::OcrModelConfig>>(
            "ocr", json_config);
        !ocr_config) {
        return ConfigReturn::kNoArgument;
    }

    std::vector<file::EmbeddingConfig> embeddings_configs;
    std::vector<file::PostprocessingConfig> post_processing_configs;

    if (const auto ret = maybe_load_field<std::vector<file::EmbeddingConfig>>(
            "embeddings", json_config)) {
        embeddings_configs = ret.value();
    }

    if (const auto ret
        = maybe_load_field<std::vector<file::PostprocessingConfig>>(
            "postprocessing", json_config)) {
        post_processing_configs = ret.value();
    }

    nlohmann::json model_json_config;
    model_json_config["ocr"] = json_config["ocr"];

    if (!embeddings_configs.empty()) {
        model_json_config["embeddings"] = {};
        for (const auto& config: embeddings_configs) {
            model_json_config["embeddings"].push_back(config.to_json());
        }
    }

    if (!post_processing_configs.empty()) {
        for (const auto& config: post_processing_configs) {
            model_json_config["postprocessing"].push_back(config.to_json());
        }
    }

    const auto path_root = std::filesystem::path(dir_path) / "model.json";
    std::ofstream ofs(path_root, std::ifstream::trunc);
    if (!ofs.is_open()) {
        return ConfigReturn::kSuccess;
    }

    ofs << model_json_config.dump(2);
    ofs.close();

    return ConfigReturn::kSuccess;
}

util::ConfigReturn util::maybe_enable_ocr_model(
    const std::string& model_key,
    util::Config* old_config,
    const std::string& dir_path)
{
    util::Config config = *old_config;

    if (std::ranges::none_of(
            config.ocr_model_configs,
            [&model_key](const file::OcrModelConfig& element) {
        return element.gguf_hash == model_key;
    })) {
        return ConfigReturn::kBadArgument;
    }

    for (auto& value: config.ocr_model_configs) {
        value.enabled = (value.gguf_hash == model_key);
    }

    ConfigReturn ret = maybe_save_model_config(config.to_json(), dir_path);
    if (ret == ConfigReturn::kSuccess) {
        *old_config = std::move(config);
    }

    return ret;
}

util::ConfigReturn util::maybe_enable_embedding_model(
    const std::string& model_key,
    bool enable,
    Config* old_config,
    const std::string& dir_path)
{
    util::Config config = *old_config;

    auto itr = std::ranges::find_if(
        config.embedding_configs,
        [&model_key](const file::EmbeddingConfig& element) {
        return element.gguf_hash == model_key;
    });

    if (itr == config.embedding_configs.end()) {
        return ConfigReturn::kBadArgument;
    }

    itr->enabled = enable;

    ConfigReturn ret = maybe_save_model_config(config.to_json(), dir_path);
    if (ret == ConfigReturn::kSuccess) {
        *old_config = std::move(config);
    }

    return ret;
}
