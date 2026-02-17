#include "file/file_indexer.hpp"
#include "database/sql.hpp"
#include "file/dictionaries.hpp"
#include "file/document.hpp"
#include "file/file_parser.hpp"
#include "file/indices.hpp"
#include "file/misc_utils.hpp"
#include "file/search_query_token.hpp"
#include "llm/configs.hpp"
#include "llm/document_extractor.hpp"
#include "llm/embedding_extractor.hpp"
#include "log/log.hpp"
#include "pdf_parser.hpp"
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <source_location>
#include <sys/types.h>
#include <unordered_map>

std::shared_ptr<util::file::FileIndexer> util::file::FileIndexer::initialize(
    const SqlDatabase::Params& database_params,
    const std::string& config_dir)
{
    LOG_INF("loading file index...");

    try {
        SqlDatabase database(database_params);

        util::file::FilesDictionaryType dictionary
            = util::file::get_all_files(database);

        std::vector<File> files;
        for (const auto& [key, value]: dictionary) {
            files.push_back(value);
            LOG_INF("loading document %s", value.filename.c_str());
        }

        remove_not(files, database);

        return std::make_shared<util::file::FileIndexer>(
            database_params, std::move(dictionary), config_dir);

    } catch (const std::exception& e) {
        const char* path = database_params.path.c_str();
        LOG_ERR(
            "%s: error loading %s: %s",
            std::source_location::current().function_name(),
            path,
            e.what());
        return nullptr;
    }
}

util::file::FileIndexer::FileIndexer(
    SqlDatabase::Params database_params,
    FilesDictionaryType&& files,
    const std::string& config_dir)
    : database_params_(std::move(database_params))
    , config_dir_(config_dir)
    , dictionaries_(std::move(files), config_dir)
{
    auto pdf_parser = std::make_shared<PdfParser>();
    parsers_["%PDF-1.0"] = pdf_parser;
    parsers_["%PDF-1.1"] = pdf_parser;
    parsers_["%PDF-1.2"] = pdf_parser;
    parsers_["%PDF-1.3"] = pdf_parser;
    parsers_["%PDF-1.4"] = pdf_parser;
    parsers_["%PDF-1.5"] = pdf_parser;
    parsers_["%PDF-1.6"] = pdf_parser;
    parsers_["%PDF-1.7"] = pdf_parser;
    parsers_["%PDF-2.0"] = pdf_parser;
}

void util::file::FileIndexer::delete_files(
    const std::vector<std::int64_t>& files)
{
    std::scoped_lock<std::mutex> lock(lock_);

    SqlDatabase database(database_params_);
    dictionaries_.delete_files(files, database);
}

std::vector<util::file::File> util::file::FileIndexer::get_file_list(
    bool include_in_progress) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return dictionaries_.get_file_list(include_in_progress);
}

std::vector<std::int64_t> util::file::FileIndexer::get_file_uuid_list(
    const std::vector<util::file::SearchQueryToken>& file_patterns,
    bool include_in_progress) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return dictionaries_.get_file_uuid_list(file_patterns, include_in_progress);
}

std::vector<std::string> util::file::FileIndexer::get_file(
    std::int64_t file) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    SqlDatabase database(database_params_);
    return dictionaries_.get_file(file, database);
}

std::optional<std::string> util::file::FileIndexer::get_file_path(
    std::int64_t file) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    SqlDatabase database(database_params_);
    return dictionaries_.get_file_path(file, database);
}

std::vector<util::file::EmbeddingLookupResult> util::file::FileIndexer::
    get_file_embeddings(std::int64_t file) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    SqlDatabase database(database_params_);
    return dictionaries_.get_file_embeddings(file, database);
}

std::vector<util::file::EmbeddingsResult> util::file::FileIndexer::
    search_embeddings(
        const util::file::EmbeddingExtractor& extractor,
        const std::vector<util::file::SearchQueryToken>& query,
        const SqlDatabase& database,
        std::size_t limit) const
{
    std::string input;
    for (const auto& token: query) {
        input.insert(input.end(), (token.value).begin(), (token.value).end());
        input.push_back(' ');
    }

    // Remove trailer
    if (input.size() != 0) {
        input.pop_back();
    }

    std::vector<util::file::Embeddings> embeddings = extractor.extract({input});
    if (embeddings.empty()) {
        return std::vector<util::file::EmbeddingsResult>();
    }

    const std::string& model = extractor.get_config().gguf_hash;
    std::vector<util::file::EmbeddingsResult> embeddings_results
        = file::calc_vector_distances(
            embeddings.front().embeddings, database, model, limit);
    return embeddings_results;
}

util::file::LookupResults util::file::FileIndexer::search(
    const std::vector<util::file::SearchQueryToken>& query,
    const SqlDatabase& database,
    const std::vector<std::int64_t>& blacklisted_files,
    const std::vector<std::int64_t>& whitelisted_files,
    bool highlight) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return dictionaries_.lookup(
        query, database, blacklisted_files, whitelisted_files, highlight);
}

void util::file::FileIndexer::file_commit(const File& file)
{
    std::scoped_lock<std::mutex> lock(lock_);

    dictionaries_.file_commit(file);
}

std::optional<util::file::File> util::file::FileIndexer::try_init_parse(
    const std::string& model,
    const std::string& prompt,
    const std::string& filename,
    const std::string& content)
{
    std::scoped_lock<std::mutex> lock(lock_);

    std::optional<std::string> workdir
        = util::file::maybe_create_files_dir(config_dir_);
    if (!workdir) {
        return std::nullopt;
    }

    std::string hash
        = std::to_string(std::hash<std::string>{}(content + filename));

    std::string path = std::filesystem::path(workdir.value()) / hash;
    if (!util::file::fs_write_file(path, content)) {
        return std::nullopt;
    }

    std::string magic = fs_read_file_magic_string(path);

    // Should be parsable
    if (!parsers_.contains(magic)) {
        return std::nullopt;
    }

    File file_stub;
    file_stub.model = model;
    file_stub.prompt = prompt;
    file_stub.path = path;
    file_stub.filename = filename;
    file_stub.hash = hash;
    file_stub.magic = magic;

    // Attempt to start indexing, fail if file already parsed or
    // is in-progress
    // Reload progress if file is already queued from previous invocation
    SqlDatabase database(database_params_);
    return dictionaries_.file_try_to_add_or_fail(file_stub, content, database);
}

bool util::file::FileIndexer::try_extract(
    DocumentExtractor& extractor,
    const File& file_target,
    const std::shared_ptr<FileUpdateNotifier>& status)
{
    std::shared_ptr<FileParser> parser;

    {
        std::scoped_lock<std::mutex> lock(lock_);

        auto itr = parsers_.find(file_target.magic);
        if (itr == parsers_.end()) {
            return false;
        }

        parser = itr->second;
    }

    return try_extract_impl(extractor, file_target, *parser, status);
}

void util::file::FileIndexer::try_extract(
    util::file::ChatExtractor& extractor,
    std::int64_t uuid,
    const std::shared_ptr<FileUpdateNotifier>& status) const
{
    std::optional<File> file = dictionaries_.get_file(uuid);

    if (!file) {
        return;
    }

    std::string prompt = extractor.get_config().prompt;

    SqlDatabase database(database_params_);
    std::vector<Document> documents = get_documents(*file, database);
    for (const Document& document: documents) {
        if (!status->is_still_set()) {
            break;
        }

        extractor.extract(prompt, document);
    }

    LOG_INF("indexed prompt %s", prompt.c_str());
}

namespace {

    inline void remove_embeddings(
        std::int64_t file,
        const std::string& table,
        util::file::SqlDatabase& database)
    {
        std::string sql = std::format("delete from '{}' where file = ?", table);

        std::shared_ptr<util::file::Update> statement
            = database.create_statement<util::file::Update>(sql.c_str());
        statement->bind(file, 1);

        database.execute(*statement);
    }

    inline std::int32_t document_to_embeddings(
        const util::file::EmbeddingExtractor& extractor,
        const util::file::Document& document,
        const std::vector<std::string>& prompts,
        util::file::SqlDatabase& database)
    {
        std::vector<util::file::Embeddings> embeddings
            = extractor.extract(prompts);
        if (embeddings.empty()) {
            return 0;
        }

        for (util::file::Embeddings& value: embeddings) {
            value.file = document.file;
            value.filename = document.filename;
            value.page_in_file = document.page_in_file;
            util::file::save(value, database);
        }

        return static_cast<std::int32_t>(embeddings.size());
    }

    inline std::vector<std::string> extract_snippets(
        const std::vector<std::string>& expressions,
        const util::file::Document& document)
    {
        std::vector<std::string> tokens;
        for (const std::string& expression: expressions) {
            const std::regex r{expression};
            auto itr = std::sregex_iterator(
                document.text.begin(), document.text.end(), r);
            for (; itr != std::sregex_iterator(); ++itr) {
                std::string value = itr->str();
                tokens.push_back(std::move(value));
            }
        }

        return tokens;
    }
} // namespace

void util::file::FileIndexer::try_extract(
    util::file::EmbeddingExtractor& extractor,
    std::int64_t uuid,
    const std::shared_ptr<FileUpdateNotifier>& status)
{
    std::optional<File> file = dictionaries_.get_file(uuid);
    if (!file) {
        return;
    }

    std::int32_t n_embeddings
        = extractor.get_llama_params().cparams.n_embeddings;
    std::string key = extractor.get_config().gguf_hash;
    std::string table_name = std::format("vec_embeddings_{}", key);

    SqlDatabase database(database_params_);

    // If table for this model doesn't exists, create it
    maybe_create_embeddings_table(key, n_embeddings, database);

    // Clear existing embeddings
    remove_embeddings(uuid, table_name, database);

    file->n_embeddings[key] = 0;

    std::vector<Document> documents = get_documents(*file, database);
    for (const Document& document: documents) {
        if (!status->is_still_set()) {
            break;
        }

        std::vector<std::string> chunks = extractor.get_config().chunks;
        std::vector<std::string> items = extract_snippets(chunks, document);
        file->n_embeddings[key]
            += document_to_embeddings(extractor, document, items, database);
        status->try_set(file.value());
    }

    dictionaries_.file_commit(file.value());

    LOG_INF(
        "indexed %d vector embeddings for %s",
        file->n_embeddings[key],
        file->path.c_str());
}

bool util::file::FileIndexer::try_extract_impl(
    util::file::DocumentExtractor& extractor,
    File file,
    const FileParser& parser,
    const std::shared_ptr<FileUpdateNotifier>& status)
{
    // Parse...
    LOG_INF("parsing %s", file.path.c_str());

    bool success = false;

    try {
        success = parser.parse(extractor, &file, database_params_, status);
    } catch (const std::exception& e) {
        // Parsing error
        LOG_ERR(
            "%s: %s",
            std::source_location::current().function_name(),
            e.what());
    }

    dictionaries_.file_commit(file);

    if (success) {
        LOG_INF(
            "indexed %s (%d/%d pages)",
            file.path.c_str(),
            file.n_pages_indexed,
            file.n_pages);
    } else {
        LOG_ERR(
            "%s: failed to index %s",
            std::source_location::current().function_name(),
            file.path.c_str());
    }

    return success;
}
