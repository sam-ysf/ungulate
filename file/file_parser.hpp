#pragma once

#include "file/file_parse_update_sink.hpp"
#include "file/indices.hpp"
#include "llm/document_extractor.hpp"
#include <memory>
#include <optional>

namespace util::file {

    //! @class FileParser
    class FileParser {
    public:
        virtual ~FileParser() = default;

        virtual bool parse(
            util::file::DocumentExtractor& extractor,
            File* file /* in/out */,
            const SqlDatabase::Params& database_params,
            const std::shared_ptr<FileUpdateNotifier>& status) const
            = 0;
    };
} // namespace util::file
