#include "models.hpp"

void util::file::maybe_save_model(
    const std::string& model,
    const std::string& description,
    util::file::SqlDatabase& database)
{
    static const char* sql =
        R"(insert into models(
                model,
                description
            ) values(?, ?)
                on conflict(model) do update set description = ?)";

    std::shared_ptr<util::file::Insertion> statement
        = database.create_statement<util::file::Insertion>(sql);
    statement->bind(model, 1);
    statement->bind(description, 2);
    statement->bind(description, 3);

    database.execute(*statement);
}
