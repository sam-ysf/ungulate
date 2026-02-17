#pragma once

#include "database/sql.hpp"
#include "file/file_parser.hpp"
#include "llm/document_extractor.hpp"
#include <memory>
#include <mutex>

namespace util::file {

    //! @class PdfParser
    /*! Parses a pdf file and encapsulates parse result.
     */
    class PdfParser : public FileParser {
    public:
        //! @Override
        bool parse(
            util::file::DocumentExtractor& extractor,
            File* file /* in/out */,
            const SqlDatabase::Params& database_params,
            const std::shared_ptr<FileUpdateNotifier>& status) const override;
    private:
        mutable std::mutex lock_;
    };
} // namespace util::file
