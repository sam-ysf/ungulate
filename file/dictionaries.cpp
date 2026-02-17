#include "file/dictionaries.hpp"
#include "database/sql.hpp"
#include "file/indices.hpp"
#include "file/misc_utils.hpp"
#include "file/search_query_token.hpp"
#include "llm/string_utils.hpp"
#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>

namespace {

    inline bool is_same_file(const std::string& file1, const std::string& file2)
    {
        if (std::filesystem::file_size(file1)
            != std::filesystem::file_size(file2)) {
            return false;
        }

        std::ifstream ifs1(file1, std::ifstream::binary);
        std::ifstream ifs2(file2, std::ifstream::binary);

        if (!ifs1.is_open() || !ifs2.is_open()) {
            throw std::runtime_error("file read error");
        }

        auto itr1 = std::istreambuf_iterator<char>(ifs1);
        auto itr2 = std::istreambuf_iterator<char>(ifs2);

        while (itr1 != std::istreambuf_iterator<char>()
               && itr2 != std::istreambuf_iterator<char>()) {
            if (*itr1++ != *itr2++) {
                return false;
            }
        }

        return itr1 == std::istreambuf_iterator<char>()
               && itr2 == std::istreambuf_iterator<char>();
    }
} // namespace

std::optional<util::file::File> util::file::Dictionaries::
    file_try_to_add_or_fail(
        const File& file_stub,
        const std::string& content,
        SqlDatabase& database)
{
    std::scoped_lock<std::mutex> lock(lock_);

    return file_try_to_add(file_stub, content, database);
}

void util::file::Dictionaries::file_commit(const File& file)
{
    std::scoped_lock<std::mutex> lock(lock_);

    if (const auto itr = uuids_to_files_in_progress_.find(file.uuid);
        itr != uuids_to_files_in_progress_.end()) {
        uuids_to_files_in_progress_.erase(itr);
    }

    uuids_to_files_[file.uuid] = file;
}

void util::file::Dictionaries::delete_files(
    const std::vector<std::int64_t>& files,
    SqlDatabase& database)
{
    std::scoped_lock<std::mutex> lock(lock_);

    std::vector<File> deletable_files;
    for (std::int64_t file: files) {
        if (const auto itr = uuids_to_files_.find(file);
            itr != uuids_to_files_.end()) {
            deletable_files.push_back(itr->second);
        }

        if (const auto itr = uuids_to_files_in_progress_.find(file);
            itr != uuids_to_files_in_progress_.end()) {
            uuids_to_files_in_progress_.erase(itr);
        }
    }

    remove(deletable_files, database);

    for (std::int64_t file: files) {
        auto itr = uuids_to_files_.find(file);
        if (itr != uuids_to_files_.end()) {
            uuids_to_files_.erase(itr);
        }
    }
}

namespace {

    std::vector<util::file::FilePageSnippet> match(
        const std::vector<std::string>& query,
        const util::file::SqlDatabase& database,
        const std::pair<std::string, std::string>& highlight)
    {
        std::string query_str = util::file::string_join(query, " and ");

        const auto& [highlight_start, highlight_end] = highlight;

        const std::string sql = std::format(
            R"(select
                file,
                page_in_file,
                highlight(documents, 2, '{}', '{}'),
                bm25(documents)
            from documents where text match ? order by bm25(documents))",
            highlight_start,
            highlight_end);

        std::shared_ptr<util::file::Read> statement
            = database.create_statement<util::file::Read>(sql.c_str());
        statement->bind(query_str, 1);

        std::vector<util::file::FilePageSnippet> rows;
        while (true) {
            std::int64_t file = 0;
            std::int64_t id_in_file = 0;
            std::string text;
            double rank = 0;

            if (!statement->read_row(file, id_in_file, text, rank)) {
                break;
            }

            util::file::Page page{.file = file, .id_in_file = id_in_file};
            rows.push_back(
                {.page = page, .rank = rank, .snippet = std::move(text)});
        }

        return rows;
    }

    std::vector<util::file::FilePageSnippet> match_text(
        const std::vector<std::string>& query,
        const util::file::SqlDatabase& database,
        const std::pair<std::string, std::string>& highlight,
        const std::vector<std::int64_t>& blacklisted_files,
        const std::vector<std::int64_t>& whitelisted_files)
    {
        std::vector<util::file::FilePageSnippet> results
            = match(query, database, highlight);

        // Maybe exclude non-whitelisted files
        // If specified, this option overrides blacklisted files
        if (!whitelisted_files.empty()) {
            std::vector<util::file::FilePageSnippet> output;
            for (const util::file::FilePageSnippet& result: results) {
                std::int64_t file = result.page.file;
                auto itr = std::ranges::find(whitelisted_files, file);
                if (itr != whitelisted_files.end()) {
                    output.push_back(result);
                }
            }

            results = std::move(output);
        }

        // Maybe exclude blacklisted files
        // If whitelist specified, this is ignored
        else if (!blacklisted_files.empty()) {
            std::vector<util::file::FilePageSnippet> output;
            for (const util::file::FilePageSnippet& result: results) {
                std::int64_t file = result.page.file;
                auto itr = std::ranges::find(blacklisted_files, file);
                if (itr == blacklisted_files.end()) {
                    output.push_back(result);
                }
            }

            results = std::move(output);
        }

        return results;
    }
} // namespace

std::vector<util::file::File> util::file::Dictionaries::get_file_list(
    bool include_in_progress) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return get_file_list_impl(include_in_progress);
}

std::vector<util::file::File> util::file::Dictionaries::get_file_list(
    const std::regex& patterns,
    bool include_in_progress) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return get_file_list_impl(patterns, include_in_progress);
}

std::vector<std::int64_t> util::file::Dictionaries::get_file_uuid_list(
    const std::vector<SearchQueryToken>& file_patterns,
    bool include_in_progress) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return get_file_uuids_impl(file_patterns, include_in_progress);
}

namespace {

    inline std::optional<std::pair<std::string, std::string>>
    generate_unique_tag_impl(
        std::mt19937& gen,
        std::uniform_int_distribution<>& dis,
        const util::file::SqlDatabase& database)
    {
        std::string body;
        for (std::size_t i = 0; i < 8; ++i) {
            body += std::to_string(dis(gen));
        }

        static const char* sql
            = R"(select snippet(documents, 2, '', '', 1) from documents where documents match ? or ?)";
        std::shared_ptr<util::file::Read> statement
            = database.create_statement<util::file::Read>(sql);

        std::string tag1 = std::format("<{}>", body);
        std::string tag2 = std::format("</{}>", body);
        statement->bind(tag1, 1);
        statement->bind(tag2, 1);

        if (std::string test_value; statement->read_row(test_value)) {
            return std::nullopt;
        }

        return std::make_pair(tag1, tag2);
    }

    inline std::pair<std::string, std::string> generate_unique_tag(
        const util::file::SqlDatabase& database)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution dis(0, 256);

        std::pair<std::string, std::string> value;

        for (;;) {
            if (const auto tags = generate_unique_tag_impl(gen, dis, database);
                tags) {
                value = tags.value();
                break;
            }
        }

        return value;
    }
} // namespace

util::file::LookupResults util::file::Dictionaries::lookup(
    const std::vector<SearchQueryToken>& query,
    const SqlDatabase& database,
    const std::vector<std::int64_t>& blacklisted_files,
    const std::vector<std::int64_t>& whitelisted_files,
    bool highlight) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    std::vector<std::string> query_words;
    query_words.reserve(query.size());
    for (const auto& q: query) {
        query_words.push_back(q.value);
    }

    LookupResults lookup_results;
    if (highlight) {
        lookup_results.highlight_tags = generate_unique_tag(database);
    }

    std::vector<FilePageSnippet> file_pages = match_text(
        query_words,
        database,
        lookup_results.highlight_tags,
        blacklisted_files,
        whitelisted_files);

    using PageLookupResultsType
        = std::unordered_map<std::int64_t, PageLookupResults>;

    PageLookupResultsType page_results;

    for (const FilePageSnippet& value: file_pages) {
        auto itr_file = uuids_to_files_.find(value.page.file);
        if (itr_file == uuids_to_files_.end()) {
            continue;
        }

        const File& file = itr_file->second;

        PageLookupResults& page_result = page_results[value.page.file];
        page_result.file = file;
        page_result.values.push_back(
            {.page = value.page, .snippet = value.snippet});

        FilePageSnippet file_page_snippet = {
            .page = value.page, .rank = value.rank, .snippet = value.snippet};
        lookup_results.snippet_lookup_results.push_back(
            SnippetLookupResults({.file = file, .value = file_page_snippet}));
    }

    lookup_results.page_lookup_results.reserve(page_results.size());
    for (auto& [key, value]: page_results) {
        lookup_results.page_lookup_results.push_back(std::move(value));
    }

    return lookup_results;
}

std::optional<util::file::File> util::file::Dictionaries::get_file(
    std::int64_t uuid) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return get_file_impl(uuid);
}

std::optional<util::file::File> util::file::Dictionaries::get_file(
    const std::string& filename) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return get_file_impl(filename);
}

std::vector<std::string> util::file::Dictionaries::get_file(
    std::int64_t file,
    SqlDatabase& database) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    if (!uuids_to_files_.contains(file)) {
        return std::vector<std::string>();
    }

    const std::string sql = R"(select text from documents where file = ?)";

    std::shared_ptr<util::file::Read> statement
        = database.create_statement<util::file::Read>(sql.c_str());
    statement->bind(file, 1);

    std::vector<std::string> content;
    while (true) {
        std::string text;
        if (!statement->read_row(text)) {
            break;
        }

        content.push_back(std::move(text));
    }

    return content;
}

std::optional<std::string> util::file::Dictionaries::get_file_path(
    std::int64_t file,
    SqlDatabase& database) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    if (!uuids_to_files_.contains(file)) {
        return std::nullopt;
    }

    const std::string sql = R"(select path from files where id = ?)";

    std::shared_ptr<util::file::Read> statement
        = database.create_statement<util::file::Read>(sql.c_str());
    statement->bind(file, 1);

    std::string path;
    if (!statement->read_row(path)) {
        return std::nullopt;
    }

    return path;
}

std::vector<util::file::EmbeddingLookupResult> util::file::Dictionaries::
    get_file_embeddings(std::int64_t file, SqlDatabase& database) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    if (!uuids_to_files_.contains(file)) {
        return std::vector<EmbeddingLookupResult>();
    }

    std::vector<std::string> tables = get_embeddings_tables(database);

    std::vector<EmbeddingLookupResult> results;

    for (const std::string& table: tables) {
        std::string sql = std::format(
            R"(select model, page_in_file, vector from '{}' where file = {})",
            table,
            file);
        std::shared_ptr<util::file::Read> statement
            = database.create_statement<util::file::Read>(sql.c_str());

        while (true) {
            EmbeddingLookupResult result;
            if (!statement->read_row(
                    result.model, result.page, result.embeddings)) {
                break;
            }

            results.push_back(std::move(result));
        }
    }

    return results;
}

std::optional<util::file::File> util::file::Dictionaries::get_file_impl(
    std::int64_t uuid) const
{
    auto itr = uuids_to_files_.find(uuid);
    if (itr == uuids_to_files_.end()) {
        return std::nullopt;
    }

    return std::optional<File>(itr->second);
}

std::optional<util::file::File> util::file::Dictionaries::get_file_impl(
    const std::string& filename) const
{
    for (const auto& [key, value]: uuids_to_files_) {
        if (value.filename == filename) {
            return value;
        }
    }

    return std::nullopt;
}

std::vector<util::file::File> util::file::Dictionaries::get_file_list_impl(
    bool include_in_progress) const
{
    std::vector<File> files;
    for (const auto& [key, value]: uuids_to_files_) {
        files.push_back(value);
    }

    if (include_in_progress) {
        for (const auto& [key, value]: uuids_to_files_in_progress_) {
            files.push_back(value);
        }
    }

    return files;
}

std::vector<util::file::File> util::file::Dictionaries::get_file_list_impl(
    const std::regex& patterns,
    bool include_in_progress) const
{
    std::vector<File> files;
    for (const auto& [key, value]: uuids_to_files_) {
        const std::string& filename = value.filename;
        if (std::regex_match(filename, patterns)) {
            files.push_back(value);
        }
    }

    if (include_in_progress) {
        for (const auto& [key, value]: uuids_to_files_in_progress_) {
            const std::string& filename = value.filename;
            if (std::regex_match(filename, patterns)) {
                files.push_back(value);
            }
        }
    }

    return files;
}

std::vector<std::int64_t> util::file::Dictionaries::get_file_uuids_impl(
    const std::vector<SearchQueryToken>& file_patterns,
    bool include_in_progress) const
{
    std::vector<std::int64_t> value;
    for (const SearchQueryToken& token: file_patterns) {
        switch (token.format_type) {
            case TokenFormatType::kPlaintext:
            {
                if (std::optional<File> ptr = get_file_impl(token.value)) {
                    value.push_back(ptr->uuid);
                }
                break;
            }

            case TokenFormatType::kRegex:
            {
                std::vector<File> files = get_file_list(
                    std::regex(token.value), include_in_progress);
                for (const auto& ptr: files)
                    value.push_back(ptr.uuid);
                break;
            }
        }
    }
    return value;
}

std::optional<util::file::File> util::file::Dictionaries::file_try_to_add(
    const File& file_stub,
    const std::string& content,
    SqlDatabase& database)
{
    const auto add_impl = [this, &database](File file) {
        // Save to database
        save(file, database);

        uuids_to_files_in_progress_[file.uuid] = file;
        return file;
    };

    const auto restart_impl = [this](const File& file) {
        // File should be resumed
        uuids_to_files_in_progress_[file.uuid] = file;
        return file;
    };

    const auto is_file_in_progress_impl = [this](const File& file) {
        auto itr = std::ranges::find_if(
            uuids_to_files_in_progress_, [&file](const auto& entry) {
            const auto& [key, value] = entry;
            return file.hash == value.hash && file.model == value.model;
        });

        if (itr != uuids_to_files_in_progress_.end()) {
            const File& existing_file = itr->second;
            if (is_same_file(file.path, existing_file.path)) {
                return true; // File currently in progress
            }
        }

        return false;
    };

    const auto rehash_file_impl = [&file_stub, &content](std::size_t counter) {
        File file_stub_copy = file_stub;

        std::filesystem::path filename(file_stub.filename);
        filename = std::format(
            "{}-{}{}",
            filename.stem().string(),
            counter,
            filename.extension().string());
        std::string hash = std::to_string(
            std::hash<std::string>{}(content + filename.string()));
        file_stub_copy.filename = filename;
        file_stub_copy.hash = hash;

        return file_stub_copy;
    };

    File file_stub_copy = file_stub;
    std::size_t counter_max = std::numeric_limits<std::size_t>::max();

    for (std::size_t counter = 0; ++counter < counter_max;) {
        // Return if file already in progress
        if (is_file_in_progress_impl(file_stub_copy)) {
            return std::nullopt;
        }

        // Check to see if file already indexed
        auto itr = std::ranges::find_if(
            uuids_to_files_, [&file_stub_copy](const auto& entry) {
            const auto& [key, value] = entry;
            return file_stub_copy.hash == value.hash
                   && file_stub_copy.model == value.model;
        });

        // If file not yet indexed, start new parse
        if (itr == uuids_to_files_.end()) {
            return add_impl(file_stub_copy);
        }

        // If file matches paused file, resume parse
        if (const File& existing_file = itr->second;
            is_same_file(file_stub_copy.path, existing_file.path)) {
            File file = std::move(itr->second);
            uuids_to_files_.erase(itr);
            return restart_impl(file);
        }

        // Have hash collision
        // Rehash index and try to re-add
        file_stub_copy = rehash_file_impl(counter);
    }

    return std::nullopt;
}
