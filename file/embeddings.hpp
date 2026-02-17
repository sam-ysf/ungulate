#pragma once

#include "database/sql.hpp"
#include <cstdint>
#include <vector>

namespace util::file {

    struct Embeddings {
        /*! Global-scope unique index
         */
        std::int64_t uuid = 0;

        /*! Parent file
         */
        std::int64_t file = 0;

        /*! Parent file
         */
        std::string filename;

        /*! Parent page
         */
        std::int64_t page_in_file = 0;

        /*! Training text label
         */
        std::string classifier_label;

        /*! Source model
         */
        std::string model;

        /*! Vector values
         */
        std::vector<float> embeddings;
    };

    void maybe_create_embeddings_table(
        const std::string& model,
        std::int32_t n_embeddings,
        util::file::SqlDatabase& database);

    //! @brief
    //!      Saves to database
    std::int64_t save(const Embeddings& value, SqlDatabase& database);

    //! @struct EmbeddingsResult
    /*! Query result
     */
    struct EmbeddingsResult {
        /*! Parent file
         */
        std::int64_t file = 0;

        /*! Parent file
         */
        std::string filename;

        /*! Parent model
         */
        std::string model;

        /*! Parent page
         */
        std::int64_t page_in_file = 0;

        /*! Vector distance from the query
         */
        double distance = 0;
    };

    std::vector<EmbeddingsResult> calc_vector_distances(
        const std::vector<float>& v,
        const SqlDatabase& database,
        const std::string& model,
        std::size_t limit);
} // namespace util::file
