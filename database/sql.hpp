#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Fwd. decl.
struct sqlite3;
// Fwd. decl.
struct sqlite3_stmt;

namespace util::file {

    //! @class Statement
    /*! @brief
     *!     atomic database statement, base class
     */
    class Statement {
    public:
        //! @brief Dtor.
        virtual ~Statement() = default;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        virtual void bind(std::int32_t value, int index) = 0;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        virtual void bind(std::int64_t value, int index) = 0;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        virtual void bind(const std::vector<char>& value, int index) = 0;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        virtual void bind(const std::vector<float>& value, int index) = 0;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        virtual void bind(const std::string& value, int index) = 0;

        //! @param database
        //!     parent database
        virtual void execute(sqlite3* database) = 0;
    protected:
        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        static void bind(sqlite3_stmt* stmt, std::int32_t value, int index);

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        static void bind(sqlite3_stmt* stmt, std::int64_t value, int index);

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        static void bind(
            sqlite3_stmt* stmt,
            const std::vector<char>& value,
            int index);

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        static void bind(
            sqlite3_stmt* stmt,
            const std::vector<float>& value,
            int index);

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        static void bind(
            sqlite3_stmt* stmt,
            const std::string& value,
            int index);
    };

    //! @class Insert
    /*! @brief
     *!     atomic database insertion
     */
    class Insertion : public Statement {
    public:
        //! @brief Dtor.
        ~Insertion() override;

        //! Ctor.
        Insertion(
            sqlite3_stmt* stmt,
            std::unique_ptr<std::unique_lock<std::mutex>>&& execution_lock)
            : stmt_(stmt)
            , execution_lock_(std::move(execution_lock))
        {}

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(std::int32_t value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(std::int64_t value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::string& value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::vector<char>& value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::vector<float>& value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @return
        //!     id of inserted-row or 0 if insertion failed
        std::int64_t get_row_id() const
        {
            return row_id_;
        }

        //! @Override
        void execute(sqlite3* database) override;
    private:
        //! Inserted row id (on successful insertion
        std::int64_t row_id_ = 0;

        sqlite3_stmt* stmt_ = nullptr;

        //! Global access lock
        std::unique_ptr<std::unique_lock<std::mutex>> execution_lock_;
    };

    //! @class Update
    /*! @brief
     *!     atomic update statement
     */
    class Update : public Statement {
    public:
        //! @brief Dtor.
        ~Update() override;

        //! Ctor.
        Update(
            sqlite3_stmt* stmt,
            std::unique_ptr<std::unique_lock<std::mutex>>&& execution_lock)
            : stmt_(stmt)
            , execution_lock_(std::move(execution_lock))
        {}

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(std::int32_t value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(std::int64_t value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::vector<char>& value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::vector<float>& value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::string& value, int index) override
        {
            Statement::bind(stmt_, value, index);
        }

        //! @param database
        //!     parent database
        void execute(sqlite3* database) override;
    private:
        sqlite3_stmt* stmt_ = nullptr;

        //! Global access lock
        std::unique_ptr<std::unique_lock<std::mutex>> execution_lock_;
    };

    //! @class Read
    /*! @brief
     *!     atomic database read
     */
    class Read : public Statement {
    public:
        //! Dtor.
        ~Read() override;

        //! Ctor.
        Read(
            sqlite3_stmt* stmt,
            std::unique_ptr<std::unique_lock<std::mutex>>&& execution_lock)
            : stmt_(stmt)
            , execution_lock_(std::move(execution_lock))
        {}

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(std::int32_t value, int index) override;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(std::int64_t value, int index) override;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::string& value, int index) override;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::vector<char>& value, int index) override;

        //! @param value
        //!     field value
        //! @param index
        //!     field column index
        void bind(const std::vector<float>& value, int index) override;

        //! Reads all columns
        //! @param stmt [in] prepared sql statement
        //! @param args [out] column output values
        template <typename... ArgType>
        bool read_row(ArgType&... args)
        {
            if (!try_read()) {
                return false;
            }

            // Read all columns...
            read_next_column(args...);
            return true;
        }
    private:
        int get_column_count() const;

        void finalize();

        template <typename VarType>
        VarType read_next();

        /* Reads column, empty base case
         */
        void read_next_column() const
        {
            /* Base case */
        }

        /* Reads next column (DOUBLE)
         */
        template <typename... ArgType>
        void read_next_column(double& value, ArgType&... args)
        {
            value = read_next<double>();
            // Read remaining columns...
            if (col_index_ < get_column_count()) {
                read_next_column(args...);
            }
        }

        /* Reads next column (INT)
         */
        template <typename... ArgType>
        void read_next_column(std::int32_t& value, ArgType&... args)
        {
            value = read_next<std::int32_t>();
            // Read remaining columns...
            if (col_index_ < get_column_count()) {
                read_next_column(args...);
            }
        }

        /* Reads next column (INT64)
         */
        template <typename... ArgType>
        void read_next_column(std::int64_t& value, ArgType&... args)
        {
            value = read_next<std::int64_t>();
            // Read remaining columns...
            if (col_index_ < get_column_count()) {
                read_next_column(args...);
            }
        }

        /* Reads next column (TEXT)
         */
        template <typename... ArgType>
        void read_next_column(std::string& value, ArgType&... args)
        {
            value = read_next<std::string>();
            // Read remaining columns...
            if (col_index_ < get_column_count()) {
                read_next_column(args...);
            }
        }

        /* Reads next column (BLOB)
         */
        template <typename... ArgType>
        void read_next_column(std::vector<char>& value, ArgType&... args)
        {
            value = read_next<std::vector<char>>();
            // Read remaining columns...
            if (col_index_ < get_column_count()) {
                read_next_column(args...);
            }
        }

        /* Reads next column (BLOB)
         */
        template <typename... ArgType>
        void read_next_column(std::vector<float>& value, ArgType&... args)
        {
            value = read_next<std::vector<float>>();
            // Read remaining columns...
            if (col_index_ < get_column_count()) {
                read_next_column(args...);
            }
        }

        void execute(sqlite3*) override
        {
            /* Nothing to do */
        }

        bool try_read();

        //! Active column index
        int col_index_ = 0;

        //! Executable SQL statement
        sqlite3_stmt* stmt_ = nullptr;

        //! Global access lock
        std::unique_ptr<std::unique_lock<std::mutex>> execution_lock_;
    };

    template <>
    std::int64_t Read::read_next<std::int64_t>();

    //! @class SqlDatabase
    /*! Encapsulates an sqlite relational database
     */
    class SqlDatabase {
    public:
        //! @struct Params
        /*! Contains initialization params
         */
        struct Params {
            // /path/to/database
            std::string path;
            // Database sql schema
            std::string schema;
            // Extension paths
            std::vector<std::string> extensions;
        };

        //! @brief Dtor.
        ~SqlDatabase();

        //! @brief Ctor.
        explicit SqlDatabase(const Params& params);
        template <typename ValueType>
        std::shared_ptr<ValueType> create_statement(const char* sql);
        template <typename ValueType>
        std::shared_ptr<ValueType> create_statement(const char* sql) const;
        // Executes statement
        void execute(const char* sql);
        // Executes statement
        void execute(Statement& statement);

        //! Fetches row from table
        //! @param  value[in] unique row key
        //! @return fetched row entry
        template <typename ValueType, typename KeyType>
        std::shared_ptr<ValueType> get(const KeyType& key);

        //! Fetches all rows from table
        //! @param values[out] all fetched row entries
        template <typename ValueType>
        void get_all(ValueType& values /*[out*/);

        //! Fetches all rows from table
        //! @return all fetched row entries
        template <typename ValueType>
        std::vector<std::shared_ptr<ValueType>> get_all();

        //! Fetches all rows from table
        //! @return all fetched row entries
        template <typename KeyType, typename ValueType>
        KeyType get_all();
    private:
        // Helper
        std::shared_ptr<Insertion> create_insertion_statement(const char* sql);
        // Helper
        std::shared_ptr<Update> create_update_statement(const char* sql);
        // Helper
        std::shared_ptr<Read> create_read_statement(const char* sql) const;

        // Sqlite instance
        sqlite3* database_ = nullptr;

        // Global lock
        static std::mutex execution_lock_;
    };

    template <>
    std::shared_ptr<Insertion> inline SqlDatabase::create_statement<Insertion>(
        const char* sql)
    {
        return create_insertion_statement(sql);
    }

    template <>
    std::shared_ptr<Update> inline SqlDatabase::create_statement<Update>(
        const char* sql)
    {
        return create_update_statement(sql);
    }

    template <>
    std::shared_ptr<Read> inline SqlDatabase::create_statement<Read>(
        const char* sql)
    {
        return create_read_statement(sql);
    }

    template <>
    std::shared_ptr<Read> inline SqlDatabase::create_statement<Read>(
        const char* sql) const
    {
        return create_read_statement(sql);
    }
} // namespace util::file
