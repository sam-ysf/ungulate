#include "file/indices.hpp"
#include "database/sql.hpp"
#include <format>
#include <unordered_map>

namespace {

    inline void try_save(
        util::file::File& value,
        util::file::SqlDatabase& database)
    {
        static const char* sql =
            R"(insert or ignore into files(
                model,
                prompt,
                filename,
                path,
                hash,
                magic,
                n_pages
            ) values(?, ?, ?, ?, ?, ?, ?))";

        // Save file entry if it doesn't already exist
        std::shared_ptr<util::file::Insertion> statement
            = database.create_statement<util::file::Insertion>(sql);
        statement->bind(value.model, 1);
        statement->bind(value.prompt, 2);
        statement->bind(value.filename, 3);
        statement->bind(value.path, 4);
        statement->bind(value.hash, 5);
        statement->bind(value.magic, 6);
        statement->bind(value.n_pages, 7);

        database.execute(*statement);

        value.uuid = statement->get_row_id();
    }

    inline void try_save_metadata(
        const util::file::File& file,
        util::file::SqlDatabase& database)
    {
        static const char* sql = R"(insert or ignore into file_metadata(
                                        file,
                                        key,
                                        value
                                    ) values(?, ?, ?))";

        if (file.metadata.empty()) {
            return;
        }

        for (const auto& [key, value]: file.metadata) {
            if (value.empty()) {
                continue;
            }

            std::shared_ptr<util::file::Insertion> statement
                = database.create_statement<util::file::Insertion>(sql);

            statement->bind(file.uuid, 1);
            statement->bind(key, 2);
            statement->bind(value, 3);

            database.execute(*statement);
        }
    }

    void update(
        const util::file::File& value,
        util::file::SqlDatabase& database)
    {
        static const char* sql = R"(update files set
                model = ?,
                prompt = ?,
                filename = ?,
                path = ?,
                hash = ?,
                magic = ?,
                n_pages = ?
            where id = ?)";

        std::shared_ptr<util::file::Insertion> statement
            = database.create_statement<util::file::Insertion>(sql);
        statement->bind(value.model, 1);
        statement->bind(value.prompt, 2);
        statement->bind(value.filename, 3);
        statement->bind(value.path, 4);
        statement->bind(value.hash, 5);
        statement->bind(value.magic, 6);
        statement->bind(value.n_pages, 7);
        statement->bind(value.uuid, 8);

        database.execute(*statement);
    }

} // namespace

void util::file::save(File& value, SqlDatabase& database)
{
    if (value.uuid > 0)
        update(value, database);
    else
        try_save(value, database);

    try_save_metadata(value, database);
}

namespace {

    inline std::string generate_database_placeholders(std::size_t size)
    {
        if (size == 0) {
            return std::string();
        }

        std::string placeholders = "?";
        for (std::size_t n = 0; ++n < size;) {
            placeholders.push_back(',');
            placeholders.push_back('?');
        }

        return placeholders;
    }

    inline void remove_embeddings_from_table(
        const std::vector<util::file::File>& files,
        const std::string& table,
        const std::string& placeholders,
        util::file::SqlDatabase& database)
    {
        std::string sql_delete_files = std::format(
            "delete from '{}' where file in ({})", table, placeholders);
        std::shared_ptr<util::file::Update> statement
            = database.create_statement<util::file::Update>(
                sql_delete_files.c_str());

        std::int32_t i = 0;
        for (const util::file::File& file: files) {
            ++i;
            statement->bind(file.uuid, i);
        }

        database.execute(*statement);
    }

    inline void remove_embeddings(
        const std::vector<util::file::File>& files,
        const std::string& placeholders,
        util::file::SqlDatabase& database)
    {
        std::vector<std::string> tables = get_embeddings_tables(database);

        for (const std::string& table: tables) {
            remove_embeddings_from_table(files, table, placeholders, database);
        }
    }
} // namespace

void util::file::remove(
    const std::vector<File>& files,
    util::file::SqlDatabase& database)
{
    const std::string placeholders
        = generate_database_placeholders(files.size());

    auto initialize_statement
        = [&](const std::string& table, const std::string& col) {
        std::string sql = std::format(
            "delete from '{}' where {} in ({})", table, col, placeholders);
        auto statement = database.create_statement<Update>(sql.c_str());

        std::int32_t i = 0;
        for (const File& file: files) {
            ++i;
            statement->bind(file.uuid, i);
        }

        return statement;
    };

    remove_embeddings(files, placeholders, database);

    std::shared_ptr<Update> statement1 = initialize_statement("files", "id");
    database.execute(*statement1);
    statement1.reset();

    std::shared_ptr<Update> statement2
        = initialize_statement("documents", "file");
    database.execute(*statement2);
    statement2.reset();

    std::shared_ptr<Update> statement3
        = initialize_statement("file_metadata", "file");
    database.execute(*statement3);
    statement3.reset();
}

namespace {

    void remove_not_embeddings_from_table(
        const std::vector<util::file::File>& files,
        const std::string& table,
        const std::string& placeholders,
        util::file::SqlDatabase& database)
    {
        std::string sql_delete_files = std::format(
            "delete from '{}' where file not in ({})", table, placeholders);
        std::shared_ptr<util::file::Update> statement
            = database.create_statement<util::file::Update>(
                sql_delete_files.c_str());

        std::int32_t i = 0;
        for (const auto& file: files) {
            ++i;
            statement->bind(file.uuid, i);
        }

        database.execute(*statement);
    }

    void remove_not_embeddings(
        const std::vector<util::file::File>& files,
        const std::string& placeholders,
        util::file::SqlDatabase& database)
    {
        std::string sql_select_tables = R"(select name from sqlite_master where
        name like 'vec_embeddings_%' and rootpage = 0 and type = 'table')";
        std::shared_ptr<util::file::Read> statement_select_tables
            = database.create_statement<util::file::Read>(
                sql_select_tables.c_str());

        std::vector<std::string> tables;
        while (true) {
            std::string table;
            if (!statement_select_tables->read_row(table)) {
                break;
            }
            tables.push_back(table);
        }

        statement_select_tables.reset();

        for (const auto& table: tables) {
            remove_not_embeddings_from_table(
                files, table, placeholders, database);
        }
    }
} // namespace

void util::file::remove_not(
    const std::vector<File>& files,
    util::file::SqlDatabase& database)
{
    const std::string placeholders
        = generate_database_placeholders(files.size());

    auto initialize_statement
        = [&](const std::string& table, const std::string& col) {
        const std::string sql = "delete from " + table + " where " + col
                                + " not in (" + placeholders + ")";
        auto statement
            = database.create_statement<util::file::Update>(sql.c_str());

        std::int32_t i = 0;
        for (const File& file: files) {
            ++i;
            statement->bind(file.uuid, i);
        }

        return statement;
    };

    remove_not_embeddings(files, placeholders, database);

    std::shared_ptr<Update> statement1 = initialize_statement("files", "id");
    database.execute(*statement1);
    statement1.reset();

    std::shared_ptr<Update> statement2
        = initialize_statement("documents", "file");
    database.execute(*statement2);
    statement2.reset();

    std::shared_ptr<Update> statement3
        = initialize_statement("file_metadata", "file");
    database.execute(*statement3);
    statement3.reset();
}

namespace {

    util::file::FilesDictionaryType get_files(
        const util::file::SqlDatabase& database)
    {
        static const char* sql = R"(
            select
                files.id,
                files.model,
                files.prompt,
                files.filename,
                files.path,
                files.hash,
                files.magic,
                files.n_pages,

                count(documents.page_in_file)

            from files

            left join documents
                on files.id = documents.file

            group by files.id
        )";

        auto statement = database.create_statement<util::file::Read>(sql);

        // Fetch data...
        util::file::FilesDictionaryType values;

        for (;;) {
            util::file::File file;

            // Fetch data...
            if (!statement->read_row(
                    file.uuid,
                    file.model,
                    file.prompt,
                    file.filename,
                    file.path,
                    file.hash,
                    file.magic,
                    file.n_pages,
                    file.n_pages_indexed)) {
                break;
            }

            values[file.uuid] = file;
        }

        return values;
    }

    void get_metadata(
        util::file::FilesDictionaryType& files /* out */,
        const util::file::SqlDatabase& database)
    {
        static const char* sql = R"(
            select
                files.id,
                file_metadata.key,
                file_metadata.value

            from files

            inner join file_metadata
                on files.id = file_metadata.file
        )";

        auto select_pages = database.create_statement<util::file::Read>(sql);

        for (;;) {
            std::int64_t file = 0;

            std::string metadata_key;
            std::string metadata_value;

            if (!select_pages->read_row(file, metadata_key, metadata_value)) {
                break;
            }

            auto itr = files.find(file);
            if (itr != files.end()) {
                itr->second.metadata[metadata_key] = metadata_value;
            }
        }
    }

    void get_file_embeddings(
        util::file::FilesDictionaryType& files /* out */,
        const util::file::SqlDatabase& database)
    {
        const std::vector<std::string> tables = get_embeddings_tables(database);

        for (const std::string& table: tables) {
            std::string sql = std::format(
                R"(select file, model, count(vector) from '{}' group by file, model)",
                table);
            auto statement
                = database.create_statement<util::file::Read>(sql.c_str());

            while (true) {
                std::string file;
                std::string model;
                std::int32_t n_embeddings = 0;

                if (!statement->read_row(file, model, n_embeddings)) {
                    break;
                }

                auto itr = files.find(std::stoi(file));
                if (itr != files.end()) {
                    itr->second.n_embeddings[model] = n_embeddings;
                }
            }
        }
    }
} // namespace

util::file::FilesDictionaryType util::file::get_all_files(
    const SqlDatabase& database)
{
    util::file::FilesDictionaryType files = get_files(database);
    get_metadata(files, database);
    get_file_embeddings(files, database);

    return files;
}

std::vector<std::string> util::file::get_embeddings_tables(
    const util::file::SqlDatabase& database)
{
    std::string sql_select_tables = R"(select name from sqlite_master where
        name like 'vec_embeddings_%' and rootpage = 0 and type = 'table')";
    std::shared_ptr<util::file::Read> statement_select_tables
        = database.create_statement<util::file::Read>(
            sql_select_tables.c_str());

    std::vector<std::string> tables;

    while (true) {
        std::string table;
        if (!statement_select_tables->read_row(table)) {
            break;
        }

        tables.push_back(std::move(table));
    }

    return tables;
}
