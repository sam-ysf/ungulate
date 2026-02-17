#include "file/embeddings.hpp"
#include "database/sql.hpp"
#include <cstdint>
#include <format>

void util::file::maybe_create_embeddings_table(
    const std::string& model,
    std::int32_t n_embeddings,
    util::file::SqlDatabase& database)
{
    std::string table_name = std::format("vec_embeddings_{}", model);

    std::string sql = std::format(
        R"(create virtual table if not exists '{}' using vec0(
                file integer,
                filename text,
                model text,
                page_in_file integer,
                vector float[{}]))",
        table_name,
        n_embeddings);

    database.execute(sql.c_str());
}

std::int64_t util::file::save(const Embeddings& value, SqlDatabase& database)
{
    const auto vector_to_string = [](const std::vector<float>& value) {
        std::string str;
        for (float v: value) {
            str += std::format("{},", v);
        }

        if (!str.empty()) {
            str.pop_back();
        }

        return str;
    };

    const std::string table_name
        = std::format("vec_embeddings_{}", value.model);
    const std::string embeddings = vector_to_string(value.embeddings);

    std::string sql = std::format(
        R"(insert or ignore into '{}' (file, filename, model, page_in_file, vector)
                values(?, ?, ?, ?, vec_f32('[{}]')))",
        table_name,
        embeddings);

    std::shared_ptr<util::file::Insertion> statement
        = database.create_statement<util::file::Insertion>(sql.c_str());
    statement->bind(value.file, 1);
    statement->bind(value.filename, 2);
    statement->bind(value.model, 3);
    statement->bind(value.page_in_file, 4);

    database.execute(*statement);

    return statement->get_row_id();
}

std::vector<util::file::EmbeddingsResult> util::file::calc_vector_distances(
    const std::vector<float>& value,
    const SqlDatabase& database,
    const std::string& model,
    std::size_t limit)
{
    if (value.empty()) {
        return std::vector<EmbeddingsResult>();
    }

    const auto to_string = [](const std::vector<float>& value) {
        // Sanity check
        if (value.empty()) {
            return std::string();
        }

        std::string str;
        for (float v: value)
            str += std::format("{},", v);
        str.pop_back();
        return str;
    };

    const std::string table_name = std::format("vec_embeddings_{}", model);

    const std::string vector_value = to_string(value);

    std::string sql = std::format(
        "select file, filename, page_in_file, distance from '{}' where "
        "model = '{}' and vector match '[{}]' limit {}",
        table_name,
        model,
        vector_value,
        limit);

    auto statement = database.create_statement<util::file::Read>(sql.c_str());

    std::vector<EmbeddingsResult> near_vectors;

    for (;;) {
        std::int64_t file = 0;
        std::string filename;
        std::int64_t page_in_file = 0;

        double distance = 0;

        if (!statement->read_row(file, filename, page_in_file, distance)) {
            break;
        }

        near_vectors.push_back(
            EmbeddingsResult{
                .file = file,
                .filename = filename,
                .model = model,
                .page_in_file = page_in_file,
                .distance = distance});
    }

    return near_vectors;
}
