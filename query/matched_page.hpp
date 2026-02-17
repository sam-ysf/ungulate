#pragma once

#include "file/embeddings.hpp"
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

namespace util::query {

    struct MatchedPage {
        //! Global-scope unique id
        std::int64_t file = 0;
        //! Parent file name
        std::string filename;
        //! Source OCR model
        std::string model;
        //! Source OCR prompt
        std::string prompt;
        //! File-scope unique id
        std::int64_t id_in_file = 0;
        //! Count of matched phrases
        std::vector<std::string> snippets;
        //! Distance from target query
        double rank = 0;
        //! Surround highlighted text
        std::pair<std::string, std::string> highlight_tags;

        MatchedPage() = default;
        explicit MatchedPage(const file::EmbeddingsResult& result)
            : file(result.file)
            , filename(result.filename)
            , model(result.model)
            , id_in_file(result.page_in_file)
            , rank(result.distance)
        {}
    };

    struct MatchedSnippet {
        //! Global-scope unique id
        std::int64_t file = 0;
        //! Source OCR model
        std::string model;
        //! Source OCR prompt
        std::string prompt;
        //! Parent file name
        std::string filename;
        //! File-scope unique id
        std::int64_t page_in_file = 0;
        //! Text and surrounding
        std::string snippet;
        //! Distance from target query
        double rank = 0;
        //! Surround highlighted text
        std::pair<std::string, std::string> highlight_tags;
    };

    inline nlohmann::json to_json(const MatchedPage& value)
    {
        nlohmann::json j;
        j["file-uuid"] = value.file;
        j["file-name"] = value.filename;
        j["model"] = value.model;
        j["page-id-in-file"] = value.id_in_file;
        j["phrase-count"] = value.snippets.size();
        j["snippets"] = value.snippets;
        j["rank"] = value.rank;
        j["highlight-tags"] = value.highlight_tags;
        return j;
    }

    inline nlohmann::json to_json(const MatchedSnippet& value)
    {
        nlohmann::json j;
        j["file-uuid"] = value.file;
        j["file-name"] = value.filename;
        j["model"] = value.model;
        j["page-id-in-file"] = value.page_in_file;
        j["snippet"] = value.snippet;
        j["rank"] = value.rank;
        j["highlight-tags"] = value.highlight_tags;
        return j;
    }
} // namespace util::query
