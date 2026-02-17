#include "sql.hpp"
#include <memory>
#include <mutex>
#include <sqlite3.h>
#include <stdexcept>
#include <string>

namespace {
    /*! Sqlite API
     */
    inline void except [[noreturn]] (sqlite3* database)
    {
        std::string what(sqlite3_errmsg(database));
        throw std::runtime_error(what);
    }

    inline void maybe_finalize(sqlite3_stmt*& stmt)
    {
        if (stmt) {
            sqlite3_finalize(stmt);
            stmt = nullptr;
        }
    }

    /*! Sqlite API
     *! Binds value to sql statement
     */
    inline void bind(sqlite3_stmt* stmt, int index, const std::string& value)
    {
        sqlite3_bind_text(
            stmt,
            index,
            value.c_str(),
            static_cast<int>(value.size()),
            SQLITE_STATIC);
    }

    /*! Sqlite API
     *! Binds value to sql statement
     */
    template <typename Type>
    inline void bind(
        sqlite3_stmt* stmt,
        int index,
        const std::vector<Type>& value)
    {
        sqlite3_bind_blob(
            stmt,
            index,
            value.data(),
            static_cast<int>(value.size() * sizeof(Type)),
            SQLITE_STATIC);
    }

    /*! Sqlite API
     *! Binds value to sql statement
     */
    inline void bind(sqlite3_stmt* stmt, int index, std::int32_t value)
    {
        sqlite3_bind_int(stmt, index, value);
    }

    /*! Sqlite API
     *! Binds value to sql statement
     */
    inline void bind(sqlite3_stmt* stmt, int index, std::int64_t value)
    {
        sqlite3_bind_int64(stmt, index, value);
    }
} // namespace

void util::file::Statement::bind(
    sqlite3_stmt* stmt,
    std::int32_t value,
    int index)
{
    ::bind(stmt, index, value);
}

void util::file::Statement::bind(
    sqlite3_stmt* stmt,
    std::int64_t value,
    int index)
{
    ::bind(stmt, index, value);
}

void util::file::Statement::bind(
    sqlite3_stmt* stmt,
    const std::vector<char>& value,
    int index)
{
    ::bind(stmt, index, value);
}

void util::file::Statement::bind(
    sqlite3_stmt* stmt,
    const std::vector<float>& value,
    int index)
{
    ::bind(stmt, index, value);
}

void util::file::Statement::bind(
    sqlite3_stmt* stmt,
    const std::string& value,
    int index)
{
    ::bind(stmt, index, value);
}

util::file::Read::~Read()
{
    finalize();
}

int util::file::Read::get_column_count() const
{
    return sqlite3_column_count(stmt_);
}

void util::file::Read::finalize()
{
    maybe_finalize(stmt_);
    // End transaction
    if (execution_lock_) {
        execution_lock_->unlock();
        execution_lock_.reset();
    }
}

void util::file::Read::bind(std::int32_t value, int index)
{
    ::bind(stmt_, index, value);
}

void util::file::Read::bind(std::int64_t value, int index)
{
    ::bind(stmt_, index, value);
}

void util::file::Read::bind(const std::string& value, int index)
{
    ::bind(stmt_, index, value);
}

void util::file::Read::bind(const std::vector<char>& value, int index)
{
    ::bind(stmt_, index, value);
}

void util::file::Read::bind(const std::vector<float>& value, int index)
{
    ::bind(stmt_, index, value);
}

util::file::Insertion::~Insertion()
{
    maybe_finalize(stmt_);

    // End transaction
    if (execution_lock_) {
        execution_lock_->unlock();
    }
}

void util::file::Insertion::execute(sqlite3* database)
{
    if (sqlite3_step(stmt_) != SQLITE_DONE) {
        maybe_finalize(stmt_);
        if (execution_lock_) {
            execution_lock_->unlock();
            execution_lock_.reset();
        }
        except(database);
    } else {
        maybe_finalize(stmt_);
        if (execution_lock_) {
            execution_lock_->unlock();
            execution_lock_.reset();
        }
        row_id_ = sqlite3_last_insert_rowid(database);
    }
}

util::file::Update::~Update()
{
    if (stmt_) {
        sqlite3_finalize(stmt_);
        stmt_ = nullptr;
    }

    // End transaction
    if (execution_lock_) {
        execution_lock_->unlock();
        execution_lock_.reset();
    }
}

//! @param database
//!     parent database
void util::file::Update::execute(sqlite3* database)
{
    if (sqlite3_step(stmt_) != SQLITE_DONE) {
        maybe_finalize(stmt_);
        if (execution_lock_) {
            execution_lock_->unlock();
            execution_lock_.reset();
        }
        except(database);
    } else {
        maybe_finalize(stmt_);
        if (execution_lock_) {
            execution_lock_->unlock();
            execution_lock_.reset();
        }
    }
}

template <>
double util::file::Read::read_next<double>()
{
    int col = col_index_;
    ++col_index_;
    return sqlite3_column_double(stmt_, col);
}

template <>
std::int32_t util::file::Read::read_next<std::int32_t>()
{
    int col = col_index_;
    ++col_index_;
    return sqlite3_column_int(stmt_, col);
}

template <>
std::int64_t util::file::Read::read_next<std::int64_t>()
{
    int col = col_index_;
    ++col_index_;
    return sqlite3_column_int64(stmt_, col);
}

template <>
std::string util::file::Read::read_next<std::string>()
{
    std::string str;
    int col = col_index_;
    ++col_index_;
    if (const auto* text
        = reinterpret_cast<const char*>(sqlite3_column_text(stmt_, col));
        text != nullptr) {
        str = std::string(text);
    }
    return str;
}

template <>
std::vector<char> util::file::Read::read_next<std::vector<char>>()
{
    int col = col_index_;
    ++col_index_;
    const void* blob = sqlite3_column_blob(stmt_, col);
    int size = sqlite3_column_bytes(stmt_, col);

    if (blob == nullptr) {
        return std::vector<char>();
    }

    const auto* beg = reinterpret_cast<const char*>(blob);
    const auto* end = reinterpret_cast<const char*>(blob) + size;

    return std::vector<char>(beg, end);
}

template <>
std::vector<float> util::file::Read::read_next<std::vector<float>>()
{
    int col = col_index_;
    ++col_index_;

    const void* blob = sqlite3_column_blob(stmt_, col);

    if (blob == nullptr) {
        return std::vector<float>();
    }

    const auto size
        = static_cast<std::size_t>(sqlite3_column_bytes(stmt_, col));

    // Just in case, if size not multiple of float, have data corruption
    if (size % sizeof(float)) {
        return std::vector<float>();
    }

    const auto* beg = static_cast<const float*>(blob);
    const auto* end = static_cast<const float*>(blob) + (size / sizeof(float));

    return std::vector<float>(beg, end);
}

bool util::file::Read::try_read()
{
    // Reset column counter
    col_index_ = 0;

    bool ret = sqlite3_step(stmt_) == SQLITE_ROW;
    if (!ret) {
        finalize();
    }

    return ret;
}

/*! Dtor.
 */
util::file::SqlDatabase::~SqlDatabase()
{
    std::scoped_lock<std::mutex> lock(execution_lock_);

    sqlite3_close(database_);
}

/*! Ctor.
 */
util::file::SqlDatabase::SqlDatabase(const Params& params)
{
    std::scoped_lock<std::mutex> lock(execution_lock_);

    // Open database
    if ((sqlite3_open_v2(
            params.path.c_str(),
            &database_,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
            nullptr))
        != SQLITE_OK) {
        except(database_);
    }

    if (int res = 0;
        sqlite3_db_config(
            database_, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, &res)
        != SQLITE_OK) {
        except(database_);
    }

    for (const auto& extension: params.extensions) {
        char* message = nullptr;
        int ret = sqlite3_load_extension(
            database_, extension.c_str(), nullptr, &message);

        std::string what;
        if (message) {
            what = std::string(message);
            sqlite3_free(message);
        }

        if (ret != SQLITE_OK) {
            throw std::runtime_error(what);
        }
    }

    // Load database schema
    if (sqlite3_exec(
            database_, (params.schema).c_str(), nullptr, nullptr, nullptr)
        != SQLITE_OK) {
        except(database_);
    }
}

void util::file::SqlDatabase::execute(Statement& statement)
{
    if (!database_) {
        throw std::string("Database not initialized");
    }

    statement.execute(database_);
}

std::shared_ptr<util::file::Insertion> util::file::SqlDatabase::
    create_insertion_statement(const char* sql)
{
    // Begin atomic transaction
    auto lock = std::make_unique<std::unique_lock<std::mutex>>(execution_lock_);

    sqlite3_stmt* stmt = nullptr;
    if (int ret = sqlite3_prepare_v2(database_, sql, -1, &stmt, nullptr);
        ret != SQLITE_OK) {
        except(database_);
    }

    return std::make_shared<Insertion>(stmt, std::move(lock));
}

std::shared_ptr<util::file::Update> util::file::SqlDatabase::
    create_update_statement(const char* sql)
{
    // Begin atomic transaction
    auto lock = std::make_unique<std::unique_lock<std::mutex>>(execution_lock_);

    sqlite3_stmt* stmt = nullptr;
    if (int ret = sqlite3_prepare_v2(database_, sql, -1, &stmt, nullptr);
        ret != SQLITE_OK) {
        except(database_);
    }

    return std::make_shared<Update>(stmt, std::move(lock));
}

std::shared_ptr<util::file::Read> util::file::SqlDatabase::
    create_read_statement(const char* sql) const
{
    // Begin atomic transaction
    auto lock = std::make_unique<std::unique_lock<std::mutex>>(execution_lock_);

    sqlite3_stmt* stmt = nullptr;
    if (int ret = sqlite3_prepare_v2(database_, sql, -1, &stmt, nullptr);
        ret != SQLITE_OK) {
        except(database_);
    }

    return std::make_shared<Read>(stmt, std::move(lock));
}

void util::file::SqlDatabase::execute(const char* sql)
{
    std::scoped_lock lock(execution_lock_);

    sqlite3_stmt* stmt = nullptr;
    if (int ret = sqlite3_prepare_v2(database_, sql, -1, &stmt, nullptr);
        ret != SQLITE_OK) {
        except(database_);
    }

    if (stmt != nullptr) {
        const int ret = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        if (ret != SQLITE_DONE && ret != SQLITE_ROW) {
            except(database_);
        }
    }
}

std::mutex util::file::SqlDatabase::execution_lock_;