#pragma once

#include "config.hpp"
#include "database/sql.hpp"
#include "file/search_query_token.hpp"
#include "file_index_runner.hpp"
#include "query/matched_page.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace util::query {

    //! @class QuerySink
    /*! Processes query requests and returns search results.
     */
    class QuerySink {
    public:
        //! @param indexer
        //!      File indexer
        QuerySink(
            std::shared_ptr<util::file::FileIndexRunner> indexer,
            Config config,
            std::shared_ptr<util::file::UpdateNotifierSink> status,
            util::file::SqlDatabase::Params database_params);

        //! @param j
        //!      Query json
        nlohmann::json get_model_list(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json get_file_list(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json get_file(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json get_file_path(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json delete_files(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json parse_file(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json calc_file_embeddings(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json get_file_embeddings(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json pause_files(const nlohmann::json& j) const;

        //! @param j
        //!      Query json
        nlohmann::json search(const nlohmann::json& j) const;

        //! @brief Performs direct keyword search
        //!
        void keyword_match(
            const std::vector<util::file::SearchQuery>& queries,
            const util::file::SqlDatabase& database,
            std::vector<MatchedPage>* matched_pages /* out */,
            std::vector<MatchedSnippet>* matched_snippets_result /* out */,
            const std::vector<std::int64_t>& blacklisted_files
            = std::vector<std::int64_t>(),
            const std::vector<std::int64_t>& whitelisted_files
            = std::vector<std::int64_t>(),
            bool highlight = false) const;

        void keyword_match_impl(
            const std::vector<util::file::SearchQueryToken>& query,
            const util::file::SqlDatabase& database,
            std::vector<MatchedPage>* matched_pages_result /* out */,
            std::vector<MatchedSnippet>* matched_snippets_result /* out */,
            const std::vector<std::int64_t>& blacklisted_files
            = std::vector<std::int64_t>(),
            const std::vector<std::int64_t>& whitelisted_files
            = std::vector<std::int64_t>(),
            bool highlight = false) const;

        //! @brief Performs vector embeddings search
        //!
        void stochastic_match(
            const std::vector<util::file::SearchQuery>& queries,
            const util::file::SqlDatabase& database,
            std::size_t limit,
            std::vector<MatchedPage>* matched_pages /* out */,
            const std::vector<std::int64_t>& blacklisted_files
            = std::vector<std::int64_t>(),
            const std::vector<std::int64_t>& whitelisted_files
            = std::vector<std::int64_t>()) const;

        void stochastic_match_pages(
            const std::vector<util::file::SearchQueryToken>& query,
            const util::file::SqlDatabase& database,
            std::size_t limit,
            std::vector<MatchedPage>* matched_pages_result /* out */,
            const std::vector<std::int64_t>& blacklisted_files
            = std::vector<std::int64_t>(),
            const std::vector<std::int64_t>& whitelisted_files
            = std::vector<std::int64_t>()) const;
    private:
        std::shared_ptr<util::file::FileIndexRunner> index_runner_;

        const Config config_;

        std::shared_ptr<util::file::UpdateNotifierSink> status_;

        // New database connections are created on-demand, so store params
        // instead of persistent instance
        util::file::SqlDatabase::Params database_params_;

        mutable std::mutex lock_;
    };
} // namespace util::query
