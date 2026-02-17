#pragma once

#include "database/sql.hpp"
#include "file/indices.hpp"
#include <string>
#include <vector>

namespace util::file {

    struct Document {
        std::int64_t file = 0;
        std::string filename;
        std::int64_t page_in_file = 0;
        std::string text;
    };

    std::vector<util::file::Document> get_documents(
        const File& file,
        SqlDatabase& database);

    //! Saves raw and formatted document to database
    void save_document(
        const std::string& text,
        std::int64_t file,
        std::int64_t page_in_file,
        SqlDatabase& database);

    void rebuild_document_index(SqlDatabase& database);
} // namespace util::file
