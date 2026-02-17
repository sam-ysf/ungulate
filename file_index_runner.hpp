#pragma once

#include "database/sql.hpp"
#include "file/file_indexer.hpp"
#include "file/file_parse_update_sink.hpp"
#include "file/request_type.hpp"
#include "llm/chat_extractor.hpp"
#include "llm/configs.hpp"
#include "llm/document_extractor.hpp"
#include "llm/embedding_extractor.hpp"
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace util::file {

    struct FileRequest {
        RequestType type = RequestType::kParseDocument;

        File file;
        std::shared_ptr<FileUpdateNotifier> notifier;
    };

    struct RunnerState {
        void notify_parse()
        {
            std::scoped_lock lock(lock_);
            if (!parsing_) {
                parsing_ = true;
                waiting_for_notify_ = true;
                cv_.notify_all();
            }
        }

        void stop_parse()
        {
            std::scoped_lock lock(lock_);
            parsing_ = false;
        }

        void stop()
        {
            std::scoped_lock lock(lock_);
            parsing_ = false;

            waiting_for_notify_ = true;
            cv_.notify_all();
        }

        void wait()
        {
            std::unique_lock lock(lock_);
            cv_.wait(lock, [this] {
                return waiting_for_notify_;
            });

            waiting_for_notify_ = false;
        }
    private:
        bool parsing_ = false;
        bool waiting_for_notify_ = false;
        std::mutex lock_;
        std::condition_variable cv_;
    };

    class FileIndexRunner {
    public:
        static std::shared_ptr<FileIndexRunner> initialize(
            const OcrModelConfig& model_config,
            const std::vector<PostprocessingConfig>& postprocessing_configs,
            const std::vector<EmbeddingConfig>& embedding_configs,
            const util::file::SqlDatabase::Params& database_params,
            const std::string& config_dir);

        ~FileIndexRunner();
        FileIndexRunner(
            const std::shared_ptr<util::file::FileIndexer>& file_indexer,
            const std::shared_ptr<DocumentExtractor>& document_extractor,
            const std::vector<std::shared_ptr<ChatExtractor>>& chat_extractors,
            const std::vector<std::shared_ptr<EmbeddingExtractor>>&
                embedding_extractors,
            util::file::SqlDatabase::Params database_params);

        bool has_embedding_extractor() const;

        // Public API
        void run_parse(
            const std::string& filename,
            const std::string& content,
            const std::shared_ptr<UpdateNotifierSink>& status);

        // Public API
        bool run_embedding_calc(
            std::int64_t uuid,
            const std::shared_ptr<UpdateNotifierSink>& status);

        // Public API
        void stop();

        // Public API
        void delete_files(const std::vector<std::int64_t>& files);

        // Public API
        void pause_files(const std::vector<std::int64_t>& files);

        // Public API
        std::unordered_map<std::string, std::string> get_model_list() const;

        // Public API
        std::vector<util::file::File> get_file_list(
            bool include_in_progress) const;

        // Public API
        std::vector<std::int64_t> get_file_uuid_list(
            const std::vector<util::file::SearchQueryToken>& file_patterns,
            bool include_in_progress) const;

        // Public API
        std::vector<std::string> get_file(int64_t file);

        // Public API
        std::optional<std::string> get_file_path(int64_t file);

        // Public API
        std::vector<util::file::EmbeddingLookupResult> get_file_embeddings(
            int64_t file);

        // Public API
        std::vector<EmbeddingsResult> search_embeddings(
            const std::vector<util::file::SearchQueryToken>& query,
            const SqlDatabase& database,
            std::size_t limit);

        // Public API
        LookupResults search(
            const std::vector<util::file::SearchQueryToken>& query,
            const SqlDatabase& database,
            const std::vector<std::int64_t>& blacklisted_files
            = std::vector<std::int64_t>(),
            const std::vector<std::int64_t>& whitelisted_files
            = std::vector<std::int64_t>(),
            bool highlight = false) const;
    private:
        void run_impl();

        void run_document_extractor(
            const File& file_target,
            const std::shared_ptr<FileUpdateNotifier>& updater) const;

        [[noreturn]] void run_postprocessors(
            const File& file_target,
            const std::shared_ptr<FileUpdateNotifier>& updater) const;

        void run_embedding_extractors(
            const File& file_target,
            const std::shared_ptr<FileUpdateNotifier>& updater) const;

        std::queue<std::shared_ptr<FileRequest>> queued_requests_;
        std::vector<std::shared_ptr<FileRequest>> dispatched_requests_;

        mutable std::mutex api_lock_;
        mutable std::mutex llama_lock_;

        std::shared_ptr<util::file::FileIndexer> indexer_;

        std::shared_ptr<DocumentExtractor> document_extractor_;
        std::vector<std::shared_ptr<ChatExtractor>> postprocessors_;
        std::vector<std::shared_ptr<EmbeddingExtractor>> embedding_extractors_;

        util::file::SqlDatabase::Params database_params_;

        bool running_ = false;
        RunnerState cv_;
        std::shared_ptr<std::jthread> worker_;
    };
} // namespace util::file
