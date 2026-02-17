#pragma once

#include "database/sql.hpp"
#include "file/indices.hpp"
#include "file/search_query_token.hpp"
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace util::file {

    //! @struct FilePAgeSnippet
    /*! Database lookup return value
     */
    struct FilePageSnippet {
        Page page;
        double rank = 0;
        std::string snippet;
    };

    //! @struct PagesLookupResult
    /*! Serialized to JSON response
     */
    struct PageLookupResults {
        struct Value {
            Page page;
            std::string snippet;
        };

        File file;
        std::vector<Value> values;
    };

    //! @struct SnippetsLookupResults
    /*! Serialized to JSON response
     */
    struct SnippetLookupResults {
        File file;
        FilePageSnippet value;
    };

    struct LookupResults {
        std::vector<PageLookupResults> page_lookup_results;
        std::vector<SnippetLookupResults> snippet_lookup_results;
        std::pair<std::string, std::string> highlight_tags;
    };

    struct EmbeddingLookupResult {
        std::string model;
        std::int64_t page = 0;
        std::vector<float> embeddings;
    };

    //! @class Dictionaries
    class Dictionaries {
    public:
        /*! Ctor.
         */
        Dictionaries() = default;

        /*! Ctor.
         */
        Dictionaries(FilesDictionaryType&& files, std::string config_dir)
            : uuids_to_files_(std::move(files))
            , config_dir_(std::move(config_dir))
        {}

        /*! @param file_stub to be added to in-progress queue
         */
        std::optional<File> file_try_to_add_or_fail(
            const File& file_stub,
            const std::string& content,
            SqlDatabase& database);

        /*! @param file to be added to dictionary of indexed files
         */
        void file_commit(const File& file);

        /*! @param files to be removed permanently
         */
        void delete_files(
            const std::vector<std::int64_t>& files,
            SqlDatabase& database);

        /*! @param patterns regex pattern to match against file names
         *! @return list of matching indexed files
         */
        std::vector<File> get_file_list(
            const std::regex& pattern,
            bool include_in_progress) const;

        /*! @return list of all indexed files
         */
        std::vector<File> get_file_list(bool include_in_progress) const;

        /*! @param patterns plaintext or regex patterns to match against file
         *!                 names
         *! @return list of matching indexed files
         */
        std::vector<std::int64_t> get_file_uuid_list(
            const std::vector<SearchQueryToken>& patterns,
            bool include_in_progress) const;

        LookupResults lookup(
            const std::vector<SearchQueryToken>& query,
            const SqlDatabase& database,
            const std::vector<std::int64_t>& blacklisted_files,
            const std::vector<std::int64_t>& whitelisted_files,
            bool highlight) const;

        std::optional<File> get_file(std::int64_t uuid) const;

        std::optional<File> get_file(const std::string& filename) const;

        /*! @param file handle
         */
        std::vector<std::string> get_file(
            std::int64_t file,
            SqlDatabase& database) const;

        /*! @param file handle
         */
        std::optional<std::string> get_file_path(
            std::int64_t file,
            SqlDatabase& database) const;

        /*! @param file handle
         */
        std::vector<EmbeddingLookupResult> get_file_embeddings(
            std::int64_t file,
            SqlDatabase& database) const;
    private:
        /*! Helper */
        std::optional<File> get_file_impl(std::int64_t uuid) const;

        /*! Helper */
        std::optional<File> get_file_impl(const std::string& filename) const;

        /*! Helper */
        std::vector<File> get_file_list_impl(bool include_in_progress) const;

        /*! Helper */
        std::vector<File> get_file_list_impl(
            const std::regex& pattern,
            bool include_in_progress) const;

        /*! Helper */
        std::vector<std::int64_t> get_file_uuids_impl(
            const std::vector<SearchQueryToken>& file_patterns,
            bool include_in_progress) const;

        /*! Helper */
        std::optional<util::file::File> file_try_to_add(
            const File& file_stub,
            const std::string& content,
            SqlDatabase& database);

        mutable std::mutex lock_;

        /*! Incomplete files, currently being indexed
         */
        FilesDictionaryType uuids_to_files_in_progress_;

        /*! Parsed files
         */
        FilesDictionaryType uuids_to_files_;

        /*! Parent to work dir
         */
        std::string config_dir_;
    };
} // namespace util::file
