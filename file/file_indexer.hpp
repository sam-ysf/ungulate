#pragma once

#include "database/sql.hpp"
#include "file/dictionaries.hpp"
#include "file/file_parser.hpp"
#include "file/indices.hpp"
#include "file/search_query_token.hpp"
#include "llm/chat_extractor.hpp"
#include "llm/document_extractor.hpp"
#include "llm/embedding_extractor.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace util::file {

    //! @class FileIndexer
    /*!*/
    class FileIndexer {
    public:
        //! Factory
        static std::shared_ptr<FileIndexer> initialize(
            const SqlDatabase::Params& database_params,
            const std::string& config_dir);

        //! Ctor.
        FileIndexer(
            SqlDatabase::Params database_params,
            FilesDictionaryType&& files,
            const std::string& config_dir);

        // Public API
        void delete_files(const std::vector<std::int64_t>& files);

        // Public API
        std::vector<File> get_file_list(bool include_in_progress) const;

        // Public API
        std::vector<std::int64_t> get_file_uuid_list(
            const std::vector<SearchQueryToken>& file_patterns,
            bool include_in_progress) const;

        // Public API
        std::vector<std::string> get_file(std::int64_t file) const;

        // Public API
        std::optional<std::string> get_file_path(std::int64_t file) const;

        // Public API
        std::vector<util::file::EmbeddingLookupResult> get_file_embeddings(
            std::int64_t file) const;

        // Public API
        std::vector<EmbeddingsResult> search_embeddings(
            const EmbeddingExtractor& extractor,
            const std::vector<SearchQueryToken>& query,
            const SqlDatabase& database,
            std::size_t limit) const;

        // Public API
        LookupResults search(
            const std::vector<SearchQueryToken>& query,
            const SqlDatabase& database,
            const std::vector<std::int64_t>& blacklisted_files
            = std::vector<std::int64_t>(),
            const std::vector<std::int64_t>& whitelisted_files
            = std::vector<std::int64_t>(),
            bool highlight = false) const;

        // Parsing
        void file_commit(const File& file);

        // Parsing
        std::optional<File> try_init_parse(
            const std::string& model,
            const std::string& prompt,
            const std::string& filename,
            const std::string& content);

        // Parsing
        bool try_extract(
            DocumentExtractor& extractor,
            const File& file_target,
            const std::shared_ptr<FileUpdateNotifier>& status);

        // Parsing
        void try_extract(
            ChatExtractor& extractor,
            std::int64_t uuid,
            const std::shared_ptr<FileUpdateNotifier>& status) const;

        // Parsing
        void try_extract(
            EmbeddingExtractor& extractor,
            std::int64_t uuid,
            const std::shared_ptr<FileUpdateNotifier>& status);
    private:
        /*! Helper
         */
        bool try_extract_impl(
            DocumentExtractor& extractor,
            File file_target,
            const FileParser& parser,
            const std::shared_ptr<FileUpdateNotifier>& status);

        mutable std::mutex lock_;

        SqlDatabase::Params database_params_;

        std::unordered_map<std::string, std::shared_ptr<FileParser>> parsers_;

        std::string config_dir_;

        Dictionaries dictionaries_;
    };
} // namespace util::file
