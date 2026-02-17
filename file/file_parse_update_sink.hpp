#pragma once

#include "file/indices.hpp"
#include "file/request_type.hpp"
#include <mutex>
#include <unordered_map>

namespace util::file {
    //! @class UpdateNotifier
    /*! Reader/writer
     */
    class UpdateNotifierSink {
    public:
        struct FileRequest {
            File file;
            bool running = false;
            RequestType type;
        };

        std::vector<FileRequest> get_all() const
        {
            std::scoped_lock<std::mutex> lock(lock_);

            std::vector<FileRequest> status;
            status.reserve(running_.size());

            for (const auto& [key, value]: running_) {
                status.push_back(value);
            }

            for (const auto& [key, value]: stopped_) {
                status.push_back(value);
            }

            return status;
        }

        bool is_still_set(std::int64_t uuid) const
        {
            std::scoped_lock<std::mutex> lock(lock_);

            return running_.contains(uuid);
        }

        void set(File file, RequestType type)
        {
            std::scoped_lock<std::mutex> lock(lock_);

            running_[file.uuid]
                = {.file = std::move(file), .running = true, .type = type};
        }

        void unset(std::int64_t uuid)
        {
            std::scoped_lock<std::mutex> lock(lock_);

            auto itr = running_.find(uuid);
            if (itr != running_.end()) {
                auto [key, request] = *itr;
                request.running = false;

                stopped_.try_emplace(key, request);
                running_.erase(itr);
            }
        }

        void unset_erase(std::int64_t uuid)
        {
            std::scoped_lock<std::mutex> lock(lock_);

            if (auto itr = running_.find(uuid); itr != running_.end()) {
                running_.erase(itr);
                return;
            }

            if (auto itr = stopped_.find(uuid); itr != stopped_.end()) {
                stopped_.erase(itr);
            }
        }
    private:
        mutable std::mutex lock_;
        std::unordered_map<std::int64_t, FileRequest> running_;
        std::unordered_map<std::int64_t, FileRequest> stopped_;
    };

    class FileUpdateNotifier {
    public:
        virtual ~FileUpdateNotifier() = default;

        virtual bool is(std::int64_t key) = 0;

        virtual bool is_still_set() = 0;

        virtual bool set(File file) = 0;

        virtual bool try_set(File file) = 0;

        virtual void unset() = 0;

        virtual void unset_erase() = 0;
    };

    class ParseUpdateNotifier : public FileUpdateNotifier {
    public:
        ~ParseUpdateNotifier() override = default;

        ParseUpdateNotifier(
            File file,
            const std::shared_ptr<UpdateNotifierSink>& sink)
            : key_(file.uuid)
            , sink_(sink)
        {
            sink_->set(std::move(file), RequestType::kParseDocument);
        }

        bool is(std::int64_t key) override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            return key_ == key;
        }

        bool is_still_set() override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            return sink_->is_still_set(key_);
        }

        bool set(File file) override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            sink_->set(std::move(file), RequestType::kParseDocument);
            return true;
        }

        bool try_set(File file) override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            if (!sink_->is_still_set(file.uuid)) {
                return false;
            }

            sink_->set(std::move(file), RequestType::kParseDocument);
            return true;
        }

        void unset() override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            sink_->unset(key_);
        }

        void unset_erase() override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            sink_->unset_erase(key_);
        }
    private:
        mutable std::mutex lock_;

        std::int64_t key_;
        std::shared_ptr<UpdateNotifierSink> sink_;
    };

    class EmbeddingUpdateNotifier : public FileUpdateNotifier {
    public:
        ~EmbeddingUpdateNotifier() override = default;

        EmbeddingUpdateNotifier(
            File file,
            const std::shared_ptr<UpdateNotifierSink>& sink)
            : key_(file.uuid)
            , sink_(sink)
        {
            sink_->set(std::move(file), RequestType::kCalcEmbeddings);
        }

        bool is(std::int64_t key) override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            return key_ == key;
        }

        bool is_still_set() override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            return sink_->is_still_set(key_);
        }

        bool set(File file) override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            sink_->set(std::move(file), RequestType::kCalcEmbeddings);
            return true;
        }

        bool try_set(File file) override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            if (!sink_->is_still_set(file.uuid)) {
                return false;
            }

            sink_->set(std::move(file), RequestType::kCalcEmbeddings);
            return true;
        }

        void unset() override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            sink_->unset(key_);
        }

        void unset_erase() override
        {
            std::scoped_lock<std::mutex> lock(lock_);

            sink_->unset_erase(key_);
        }
    private:
        mutable std::mutex lock_;

        std::int64_t key_;
        std::shared_ptr<UpdateNotifierSink> sink_;
    };
} // namespace util::file
