#pragma once

#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace util::file {

    struct OcrModelConfig {
        std::string gguf;
        std::string gguf_hash;
        std::string mmproj;
        std::string prompt;
        std::string description;
        bool enabled = false;

        bool clear_cache = false;
        bool verbose_parse_result = false;

        std::unordered_map<std::string, std::string> params;

        bool operator==(const OcrModelConfig& rhs) const = default;

        bool is_ok() const
        {
            return !prompt.empty() && !gguf.empty() && !gguf_hash.empty()
                   && !mmproj.empty();
        }

        bool is_not_ok() const
        {
            return !is_ok();
        }

        nlohmann::json to_json() const
        {
            nlohmann::json value;
            value["gguf"] = gguf;
            value["mmproj"] = mmproj;
            value["prompt"] = prompt;
            value["description"] = description;
            value["clear-cache"] = clear_cache;
            value["verbose-parse-result"] = verbose_parse_result;
            value["params"] = params;
            value["enabled"] = enabled;
            return value;
        }
    };

    struct PostprocessingConfig {
        std::string gguf;
        std::string gguf_hash;
        std::string prompt;
        std::string description;
        bool enabled = false;

        bool clear_cache = false;

        std::unordered_map<std::string, std::string> params;

        bool operator==(const PostprocessingConfig& rhs) const = default;

        bool is_ok() const
        {
            return !prompt.empty() && !gguf.empty() && !gguf_hash.empty();
        }

        bool is_not_ok() const
        {
            return !is_ok();
        }

        nlohmann::json to_json() const
        {
            nlohmann::json value;
            value["gguf"] = gguf;
            value["prompt"] = prompt;
            value["description"] = description;
            value["clear-cache"] = clear_cache;
            value["params"] = params;
            value["enabled"] = enabled;
            return value;
        }
    };

    struct EmbeddingConfig {
        std::string gguf;
        std::string gguf_hash;
        bool embeddings_only = false;
        std::string description;
        bool enabled = false;

        std::vector<std::string> chunks;

        bool clear_cache = false;

        std::unordered_map<std::string, std::string> params;

        bool operator==(const EmbeddingConfig& rhs) const = default;

        bool is_ok() const
        {
            return !gguf.empty() && !gguf_hash.empty();
        }

        bool is_not_ok() const
        {
            return !is_ok();
        }

        nlohmann::json to_json() const
        {
            nlohmann::json value;
            value["gguf"] = gguf;
            value["embeddings-only"] = embeddings_only;
            value["description"] = description;
            value["enabled"] = enabled;
            value["chunks"] = chunks;
            value["clear-cache"] = clear_cache;
            value["params"] = params;
            return value;
        }
    };

    template <typename ModelType>
    inline bool have_bad_model(const ModelType& configs)
    {
        return std::ranges::any_of(configs, [](const auto& config) {
            return config.is_not_ok();
        });
    };
} // namespace util::file
