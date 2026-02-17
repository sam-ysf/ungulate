#pragma once

#include "database/sql.hpp"
#include <string>

namespace util::file {

    void maybe_save_model(
        const std::string& model,
        const std::string& description,
        util::file::SqlDatabase& database);
}