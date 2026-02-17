#include "file/document.hpp"
#include "llm/string_utils.hpp"
#include "misc_utils.hpp"
#include <cstdlib>
#include <cstring>

std::vector<util::file::Document> util::file::get_documents(
    const File& file,
    SqlDatabase& database)
{
    const std::string sql
        = R"(select page_in_file, text from documents where file = ?)";
    std::shared_ptr<util::file::Read> statement
        = database.create_statement<util::file::Read>(sql.c_str());
    statement->bind(file.uuid, 1);

    std::vector<util::file::Document> output;

    while (true) {
        std::int64_t page;
        std::string text;

        if (!statement->read_row(page, text)) {
            break;
        }

        output.push_back(
            {.file = file.uuid,
             .filename = file.filename,
             .page_in_file = page,
             .text = std::move(text)});
    }

    return output;
}

void util::file::save_document(
    const std::string& text,
    std::int64_t file,
    std::int64_t page_in_file,
    SqlDatabase& database)
{
    const std::string sql(R"(
        insert into documents(
                file,
                page_in_file,
                text
            ) values(?, ?, ?))");

    std::shared_ptr<util::file::Insertion> statement
        = database.create_statement<util::file::Insertion>(sql.c_str());
    statement->bind(file, 1);
    statement->bind(page_in_file, 2);
    statement->bind(text, 3);

    std::vector<std::string> items = string_split(text, '\n');

    database.execute(*statement);
}

void util::file::rebuild_document_index(SqlDatabase& database)
{
    const std::string sql(
        R"(insert into documents(text) values('rebuild'))");

    std::shared_ptr<util::file::Insertion> statement
        = database.create_statement<util::file::Insertion>(sql.c_str());
    database.execute(*statement);
}