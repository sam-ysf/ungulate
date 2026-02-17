#pragma once

#include "database/sql.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace util::file {

    struct File {
        /*! Global-scope unique index
         */
        std::int64_t uuid = 0;

        std::string path;

        std::string model;

        std::string prompt;

        std::string filename;

        /*! Source checksum
         */
        std::string hash;

        std::string magic;

        std::int32_t n_pages = 0;

        std::int32_t n_pages_indexed = 0;

        std::unordered_map<std::string, std::string> metadata;

        std::unordered_map<std::string, std::int32_t> n_embeddings;
    };

    struct Page {
        /*! Parent file uuid
         */
        std::int64_t file = 0;

        /*! File-scope unique index
         */
        std::int64_t id_in_file = 0;

        bool operator==(const Page& rhs) const = default;

        struct Key {
            std::size_t operator()(const Page& page) const
            {
                std::size_t h1 = std::hash<std::int64_t>{}(page.file);
                std::size_t h2 = std::hash<std::int64_t>{}(page.id_in_file);
                return h1 ^ h2;
            }
        };
    };

    using FilesDictionaryType
        = std::unordered_map<std::int64_t, util::file::File>;

    //! @brief Saves to database
    void save(File& value, SqlDatabase& database);

    //! @brief Removes files from database
    void remove(const std::vector<File>& files, SqlDatabase& database);

    //! Should be called on startup, in case server is interrupted in the middle
    //! of a file delete operation
    void remove_not(const std::vector<File>& files, SqlDatabase& database);

    //! @brief Gets from database
    FilesDictionaryType get_all_files(const SqlDatabase& database);

    //! @brief Gets from database
    std::vector<std::string> get_embeddings_tables(
        const util::file::SqlDatabase& database);
} // namespace util::file
