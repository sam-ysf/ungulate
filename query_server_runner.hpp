#pragma once

#include "config.hpp"
#include "database/sql.hpp"
#include "file_index_runner.hpp"
#include "query/query_server.hpp"
#include "query/query_sink.hpp"
#include <memory>

namespace util::query {

    class QueryServerRunner {
    public:
        static std::shared_ptr<QueryServerRunner> initialize(
            const std::shared_ptr<util::file::FileIndexRunner>& indexer,
            const std::string& config_dir,
            const util::Config& config,
            const util::file::SqlDatabase::Params& database_params);

        QueryServerRunner(
            const std::shared_ptr<QueryServer>& query_server,
            const std::shared_ptr<QuerySink>& sink);

        void run() const;
        void stop() const;
    private:
        mutable std::mutex lock_;
        // Resource fetcher backend
        std::shared_ptr<QuerySink> sink_;
        // Network backend
        std::shared_ptr<QueryServer> query_server_;
    };
} // namespace util::query
