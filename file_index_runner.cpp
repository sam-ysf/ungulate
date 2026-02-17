#include "file_index_runner.hpp"
#include "database/sql.hpp"
#include "file/embeddings.hpp"
#include "file/file_indexer.hpp"
#include "file/file_parse_update_sink.hpp"
#include "file/indices.hpp"
#include "file/models.hpp"
#include "llm/arg.hpp"
#include "llm/base_utils.hpp"
#include "llm/chat_extractor.hpp"
#include "llm/configs.hpp"
#include "llm/document_extractor.hpp"
#include "llm/embedding_extractor.hpp"
#include "llm/llama_base.hpp"
#include "log/log.hpp"
#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <openssl/md5.h>
#include <optional>
#include <queue>
#include <source_location>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace util::file {

    namespace {

        template <typename Type>
        std::shared_ptr<Type> try_pull(
            std::queue<std::shared_ptr<Type>>& container)
        {
            if (container.empty()) {
                return nullptr;
            }

            std::shared_ptr<Type> value = container.front();
            container.pop();
            return value;
        }

        struct GgufSource {
            std::shared_ptr<LlamaBase> base;
            std::string gguf_hash;
            std::unordered_map<std::string, std::string> params;
        };

        std::shared_ptr<DocumentExtractor> initialize_document_extractor(
            const OcrModelConfig& config,
            std::vector<GgufSource>* ggufs)
        {
            const auto itr = std::ranges::find_if(
                *ggufs, [&config](const GgufSource& entry) {
                return entry.gguf_hash == config.gguf_hash
                       && entry.params == config.params;
            });

            // Use existing gguf if already loaded
            if (itr != ggufs->end()) {
                return std::make_shared<DocumentExtractor>(itr->base, config);
            }

            std::optional<llm_util_params> params
                = generate_llm_util_params(config.params)
                      .value_or(llm_util_params());
            params->mparams.source.gguf_hash = config.gguf_hash;
            params->mparams.source.gguf = config.gguf;

            // init llm backend for ocr
            auto base = std::make_shared<LlamaBase>(params.value());
            ggufs->push_back(
                {.base = base,
                 .gguf_hash = config.gguf_hash,
                 .params = config.params});
            return std::make_shared<DocumentExtractor>(base, config);
        }

        std::vector<std::shared_ptr<ChatExtractor>> initialize_postprocessors(
            const std::vector<PostprocessingConfig>& postprocessing_configs,
            std::vector<GgufSource>* ggufs)
        {
            std::vector<std::shared_ptr<util::file::ChatExtractor>> extractors;

            for (const auto& config: postprocessing_configs) {
                if (!config.enabled) {
                    continue;
                }

                const auto itr = std::ranges::find_if(
                    *ggufs, [&config](const GgufSource& entry) {
                    return entry.gguf_hash == config.gguf_hash
                           && entry.params == config.params;
                });

                // Use existing gguf if already loaded
                if (itr != ggufs->end()) {
                    extractors.push_back(
                        std::make_shared<util::file::ChatExtractor>(
                            itr->base, config));
                    continue;
                }

                std::optional<llm_util_params> params
                    = generate_llm_util_params(config.params);
                params = params.value_or(llm_util_params());
                params->mparams.source.gguf_hash = config.gguf_hash;
                params->mparams.source.gguf = config.gguf;

                auto base = std::make_shared<LlamaBase>(params.value());
                extractors.push_back(
                    std::make_shared<util::file::ChatExtractor>(base, config));
                ggufs->push_back(
                    {.base = base,
                     .gguf_hash = config.gguf_hash,
                     .params = config.params});
            }

            return extractors;
        }

        std::vector<std::shared_ptr<EmbeddingExtractor>>
        initialize_embedding_extractors(
            const std::vector<EmbeddingConfig>& embedding_configs,
            std::vector<GgufSource>* ggufs)
        {
            std::vector<std::shared_ptr<EmbeddingExtractor>> extractors;

            for (const auto& config: embedding_configs) {
                if (!config.enabled) {
                    continue;
                }

                const auto itr = std::ranges::find_if(
                    *ggufs, [&config](const GgufSource& entry) {
                    return entry.gguf_hash == config.gguf_hash
                           && entry.params == config.params;
                });

                // Use existing gguf if already loaded
                if (itr != ggufs->end()) {
                    extractors.push_back(
                        std::make_shared<util::file::EmbeddingExtractor>(
                            itr->base, config));
                    continue;
                }

                std::optional<llm_util_params> params
                    = generate_llm_util_params(config.params);
                params = params.value_or(llm_util_params());
                params->mparams.source.gguf_hash = config.gguf_hash;
                params->mparams.source.gguf = config.gguf;

                auto base = std::make_shared<LlamaBase>(params.value());
                extractors.push_back(
                    std::make_shared<util::file::EmbeddingExtractor>(
                        base, config));

                ggufs->push_back(
                    {.base = base,
                     .gguf_hash = config.gguf_hash,
                     .params = config.params});
            }

            return extractors;
        }
    } // namespace
} // namespace util::file

std::shared_ptr<util::file::FileIndexRunner> util::file::FileIndexRunner::
    initialize(
        const OcrModelConfig& model_config,
        const std::vector<PostprocessingConfig>& postprocessing_configs,
        const std::vector<EmbeddingConfig>& embedding_configs,
        const util::file::SqlDatabase::Params& database_params,
        const std::string& config_dir)
{
    if (model_config.is_not_ok() || have_bad_model(postprocessing_configs)
        || have_bad_model(embedding_configs)) {
        return nullptr;
    }

    try {
        std::shared_ptr<util::file::FileIndexer> file_indexer
            = util::file::FileIndexer::initialize(database_params, config_dir);
        if (file_indexer == nullptr) {
            return nullptr;
        }

        std::vector<GgufSource> ggufs;

        // init backends for document ocr
        std::shared_ptr<util::file::DocumentExtractor> document_extractor
            = initialize_document_extractor(model_config, &ggufs);

        if (!document_extractor) {
            return nullptr;
        }

        // init backends for documetn postprocessing
        std::vector<std::shared_ptr<util::file::ChatExtractor>> chat_extractors
            = initialize_postprocessors(postprocessing_configs, &ggufs);

        // init backends for vector embedding extraction
        std::vector<std::shared_ptr<util::file::EmbeddingExtractor>>
            embedding_extractors
            = initialize_embedding_extractors(embedding_configs, &ggufs);

        return std::make_shared<util::file::FileIndexRunner>(
            file_indexer,
            document_extractor,
            chat_extractors,
            embedding_extractors,
            database_params);

    } catch (std::runtime_error&) {
        return nullptr;
    }
}

util::file::FileIndexRunner::~FileIndexRunner()
{
    stop();
}

util::file::FileIndexRunner::FileIndexRunner(
    const std::shared_ptr<util::file::FileIndexer>& file_indexer,
    const std::shared_ptr<DocumentExtractor>& document_extractor,
    const std::vector<std::shared_ptr<ChatExtractor>>& chat_extractors,
    const std::vector<std::shared_ptr<EmbeddingExtractor>>&
        embedding_extractors,
    util::file::SqlDatabase::Params database_params)
    : indexer_(file_indexer)
    , document_extractor_(document_extractor)
    , postprocessors_(chat_extractors)
    , embedding_extractors_(embedding_extractors)
    , database_params_(std::move(database_params))
    , running_(true)
{
    worker_ = std::make_shared<std::jthread>([this] {
        while (true) {
            cv_.wait();

            {
                std::scoped_lock<std::mutex> lock(api_lock_);
                if (!running_) {
                    queued_requests_
                        = std::queue<std::shared_ptr<FileRequest>>();
                    break;
                }
            }

            run_impl();
        }
    });
}

bool util::file::FileIndexRunner::has_embedding_extractor() const
{
    return !embedding_extractors_.empty();
}

void util::file::FileIndexRunner::run_parse(
    const std::string& filename,
    const std::string& content,
    const std::shared_ptr<UpdateNotifierSink>& status)
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    if (!running_) {
        return;
    }

    OcrModelConfig config = document_extractor_->get_config();

    // Add file to queue and maybe notify worker thread
    if (auto queued_file = indexer_->try_init_parse(
            config.gguf_hash, config.prompt, filename, content)) {
        auto notifier = std::make_shared<ParseUpdateNotifier>(
            queued_file.value(), status);
        queued_requests_.push(
            std::make_shared<FileRequest>(FileRequest{
                .type = RequestType::kParseDocument,
                .file = std::move(queued_file.value()),
                .notifier = notifier}));
    }

    cv_.notify_parse();
}

bool util::file::FileIndexRunner::run_embedding_calc(
    std::int64_t uuid,
    const std::shared_ptr<UpdateNotifierSink>& status)
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    if (!running_) {
        return false;
    }

    // Ensure file existence
    std::vector<util::file::File> files = indexer_->get_file_list(false);
    const auto itr
        = std::ranges::find_if(files, [uuid](const util::file::File& file) {
        return file.uuid == uuid;
    });

    if (itr == files.end()) {
        return false;
    }

    // Add file to queue and maybe notify worker thread
    auto notifier = std::make_shared<EmbeddingUpdateNotifier>(*itr, status);

    queued_requests_.push(
        std::make_shared<FileRequest>(FileRequest{
            .type = RequestType::kCalcEmbeddings,
            .file = std::move(*itr),
            .notifier = notifier}));

    cv_.notify_parse();
    return true;
}

void util::file::FileIndexRunner::stop()
{
    {
        std::scoped_lock<std::mutex> lock(api_lock_);
        if (!running_) {
            return;
        }

        running_ = false;

        for (const auto& request: dispatched_requests_) {
            request->notifier->unset();
        }

        dispatched_requests_.clear();

        while (!queued_requests_.empty()) {
            auto request = try_pull(queued_requests_);
            if (request) {
                request->notifier->unset();
            }
        }

        cv_.stop();
    }

    if (worker_->joinable()) {
        worker_->join();
        worker_.reset();
    }
}

void util::file::FileIndexRunner::delete_files(
    const std::vector<std::int64_t>& files)
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    // Cancel any queued or in-flight work for the target files.
    for (int64_t file: files) {
        for (const auto& request: dispatched_requests_) {
            if (request->notifier->is(file)) {
                request->notifier->unset();
            }
        }

        std::queue<std::shared_ptr<FileRequest>> remaining;
        while (!queued_requests_.empty()) {
            auto request = try_pull(queued_requests_);
            if (!request) {
                break;
            }

            if (request->notifier->is(file)) {
                request->notifier->unset();
            } else {
                remaining.push(request);
            }
        }
        queued_requests_ = std::move(remaining);
    }

    indexer_->delete_files(files);
}

void util::file::FileIndexRunner::pause_files(
    const std::vector<std::int64_t>& files)
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    for (int64_t file: files) {
        for (const auto& request: dispatched_requests_) {
            if (request->notifier->is(file)) {
                request->notifier->unset();
            }
        }

        std::queue<std::shared_ptr<FileRequest>> remaining;
        while (!queued_requests_.empty()) {
            auto request = try_pull(queued_requests_);
            if (!request) {
                break;
            }

            if (request->notifier->is(file)) {
                request->notifier->unset();
            } else {
                remaining.push(request);
            }
        }
        queued_requests_ = std::move(remaining);
    }
}

std::unordered_map<std::string, std::string> util::file::FileIndexRunner::
    get_model_list() const
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    static const char* sql = "select model, description from models";

    util::file::SqlDatabase database(database_params_);

    std::shared_ptr<util::file::Read> statement
        = database.create_statement<util::file::Read>(sql);

    std::unordered_map<std::string, std::string> models;
    while (true) {
        std::string model;
        std::string description;

        if (!statement->read_row(model, description)) {
            break;
        }

        models.try_emplace(model, description);
    }

    return models;
}

std::vector<util::file::File> util::file::FileIndexRunner::get_file_list(
    bool include_in_progress) const
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    return indexer_->get_file_list(include_in_progress);
}

std::vector<std::int64_t> util::file::FileIndexRunner::get_file_uuid_list(
    const std::vector<util::file::SearchQueryToken>& file_patterns,
    bool include_in_progress) const
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    return indexer_->get_file_uuid_list(file_patterns, include_in_progress);
}

std::vector<std::string> util::file::FileIndexRunner::get_file(
    std::int64_t file)
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    return indexer_->get_file(file);
}

std::optional<std::string> util::file::FileIndexRunner::get_file_path(
    std::int64_t file)
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    return indexer_->get_file_path(file);
}

std::vector<util::file::EmbeddingLookupResult> util::file::FileIndexRunner::
    get_file_embeddings(std::int64_t file)
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    return indexer_->get_file_embeddings(file);
}

std::vector<util::file::EmbeddingsResult> util::file::FileIndexRunner::
    search_embeddings(
        const std::vector<util::file::SearchQueryToken>& query,
        const SqlDatabase& database,
        std::size_t limit)
{
    std::scoped_lock lock(llama_lock_, api_lock_);

    std::vector<util::file::EmbeddingsResult> accumulator;

    for (const auto& embedding_extractor: embedding_extractors_) {
        try {
            embedding_extractor->initialize();

            auto result = indexer_->search_embeddings(
                *embedding_extractor, query, database, limit);
            accumulator.insert(accumulator.end(), result.begin(), result.end());

            // Tear down to avoid overflowing GPU memory
            // FUTURE allow user to specify keep-alive
            embedding_extractor->teardown();
        }

        catch (const std::exception& e) {
            LOG_ERR(
                "%s: %s",
                std::source_location::current().function_name(),
                e.what());
            embedding_extractor->teardown();
        }
    }

    return accumulator;
}

util::file::LookupResults util::file::FileIndexRunner::search(
    const std::vector<util::file::SearchQueryToken>& query,
    const SqlDatabase& database,
    const std::vector<std::int64_t>& blacklisted_files,
    const std::vector<std::int64_t>& whitelisted_files,
    bool highlight) const
{
    std::scoped_lock<std::mutex> lock(api_lock_);

    return indexer_->search(
        query, database, blacklisted_files, whitelisted_files, highlight);
}

void util::file::FileIndexRunner::run_impl()
{
    while (true) {
        std::shared_ptr<FileRequest> target;

        std::scoped_lock<std::mutex> llock(llama_lock_);

        {
            std::scoped_lock lock(api_lock_);

            if (!running_) {
                break;
            }

            target = try_pull(queued_requests_);
            if (!target) {
                cv_.stop_parse();
                break;
            }

            dispatched_requests_.push_back(target);
        }

        switch (target->type) {
            case RequestType::kParseDocument:
            {
                run_document_extractor(target->file, target->notifier);
                target->notifier->unset_erase();
                break;
            }

            case RequestType::kCalcEmbeddings:
            {
                run_embedding_extractors(target->file, target->notifier);
                target->notifier->unset_erase();
                break;
            }
        }

        {
            std::scoped_lock<std::mutex> lock(api_lock_);

            auto itr = std::ranges::find(dispatched_requests_, target);
            if (itr != dispatched_requests_.end()) {
                dispatched_requests_.erase(itr);
            }
        }
    }
}

namespace {

    template <typename ExtractorType>
    inline void maybe_save_model(
        const ExtractorType& extractor,
        const util::file::SqlDatabase::Params& database_params)
    {
        util::file::SqlDatabase database(database_params);

        std::string model = extractor.to_identifier();
        std::string description = extractor.get_config().description;

        util::file::maybe_save_model(model, description, database);
    }
} // namespace

void util::file::FileIndexRunner::run_document_extractor(
    const File& file_target,
    const std::shared_ptr<FileUpdateNotifier>& updater) const
{
    if (!document_extractor_) {
        return;
    }

    // FUTURE
    // Move model registration to initial config
    ::maybe_save_model(*document_extractor_, database_params_);

    try {
        document_extractor_->initialize();

        indexer_->try_extract(*document_extractor_, file_target, updater);

        // Tear down to avoid overflowing GPU memory
        // FUTURE allow user to specify
        document_extractor_->teardown();
    }

    catch (const std::exception& e) {
        updater->unset();
        indexer_->file_commit(file_target);
        document_extractor_->teardown();
        LOG_ERR(
            "%s: %s",
            std::source_location::current().function_name(),
            e.what());
    }
}

[[noreturn]] void util::file::FileIndexRunner::run_postprocessors(
    const File&,
    const std::shared_ptr<FileUpdateNotifier>&) const
{
    // FUTURE
    // implement postprocessing
    throw std::logic_error(
        std::format(
            "{}: Not yet impelemented",
            std::source_location::current().function_name()));
}

void util::file::FileIndexRunner::run_embedding_extractors(
    const File& file_target,
    const std::shared_ptr<FileUpdateNotifier>& updater) const
{
    for (const auto& embedding_extractor: embedding_extractors_) {
        if (!updater->is_still_set()) {
            break;
        }

        // FUTURE
        // Move model registration to initial config
        ::maybe_save_model(*embedding_extractor, database_params_);

        try {
            embedding_extractor->initialize();

            indexer_->try_extract(
                *embedding_extractor, file_target.uuid, updater);

            // Tear down to avoid overflowing GPU memory
            // FUTURE allow user to specify
            embedding_extractor->teardown();
        } catch (const std::runtime_error&) {
            embedding_extractor->teardown();
            throw;
        }
    }
}
