#pragma once

#include "llm/configs.hpp"
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace util {

    using Env = std::unordered_map<std::string, std::string>;

    struct Config {
        std::unordered_map<std::string, std::string> global_params;

        std::vector<file::OcrModelConfig> ocr_model_configs;
        std::vector<file::PostprocessingConfig> post_processing_configs;
        std::vector<file::EmbeddingConfig> embedding_configs;

        bool has_param(const std::string& key) const;
        std::optional<std::string> operator[](const std::string& key) const;

        nlohmann::json to_json() const;

        Config& operator+=(const Config& rhs);
        friend Config operator+(const Config& lhs, const Config& rhs)
        {
            Config config;
            config += lhs;
            config += rhs;

            return config;
        }
    };

    enum class ConfigReturn : std::uint8_t {
        kSuccess,
        kBadArgument,
        kNoArgument,
        kSaveError,
    };

    ConfigReturn maybe_save_net_config(
        const nlohmann::json& json_config,
        const Config& old_config,
        const std::string& dir_path);

    ConfigReturn maybe_save_model_config(
        const nlohmann::json& json_config,
        const std::string& dir_path);

    ConfigReturn maybe_enable_ocr_model(
        const std::string& model_key,
        Config* old_config /* in/out */,
        const std::string& dir_path);

    ConfigReturn maybe_enable_embedding_model(
        const std::string& model_key,
        bool enable,
        Config* old_config /* in/out */,
        const std::string& dir_path);

    //! @param path
    //!     /path/to/conf/directory
    Config load_config(const std::string& path);
} // namespace util
