#include "log/log.hpp"
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <ggml.h>
#include <memory>
#include <mutex>
#include <unistd.h>
#include <unordered_map>

namespace {
    constexpr const char* kLogColDefault = "\033[0m";
    constexpr const char* kLogColBold = "\033[1m";
    constexpr const char* kLogColRed = "\033[31m";
    constexpr const char* kLogColGreen = "\033[32m";
    constexpr const char* kLogColYellow = "\033[33m";
    constexpr const char* kLogColBlue = "\033[34m";
    constexpr const char* kLogColMagenta = "\033[35m";
    constexpr const char* kLogColCyan = "\033[36m";
    constexpr const char* kLogColWhite = "\033[37m";

    // Terminal colors
    enum class log_color : std::uint8_t {
        kCommonLogColorDefault = 0,
        kCommonLogColorBold,
        kCommonLogColorRed,
        kCommonLogColorGreen,
        kCommonLogColorYellow,
        kCommonLogColorBlue,
        kCommonLogColorMagenta,
        kCommonLogColorCyan,
        kCommonLogColorWhite,
    };

    constexpr const char* kColors[]
        = {/* COMMON_LOG_COL_DEFAULT */ kLogColDefault,
           /* COMMON_LOG_COL_BOLD */ kLogColBold,
           /* COMMON_LOG_COL_RED */ kLogColRed,
           /* COMMON_LOG_COL_GREEN */ kLogColGreen,
           /* COMMON_LOG_COL_YELLOW */ kLogColYellow,
           /* COMMON_LOG_COL_BLUE */ kLogColBlue,
           /* COMMON_LOG_COL_MAGENTA */ kLogColMagenta,
           /* COMMON_LOG_COL_CYAN */ kLogColCyan,
           /* COMMON_LOG_COL_WHITE */ kLogColWhite};

    // Auto-detects if colors should be enabled based on terminal and
    // environment
    inline bool llm_util_log_should_use_colors_auto()
    {
        // Check NO_COLOR environment variable (https://no-color.org/)
        if (const char* no_color = std::getenv("NO_COLOR")) {
            if (std::strcmp(no_color, "") != 0) {
                return false;
            }
        }

        // Check TERM environment variable
        if (const char* term = std::getenv("TERM")) {
            if (std::strcmp(term, "dumb") == 0) {
                return false;
            }
        }

        // Check if stdout is connected to a terminal
        return isatty(fileno(stdout)) != 0;
    }

    //! Returns time in microseconds since start of unix era
    inline std::int64_t t_us()
    {
        static const auto kStart
            = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch());

        const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch());

        return (now - kStart).count();
    }
} // namespace

namespace util::log {

    using LogsType
        = std::unordered_map<std::string, std::unique_ptr<util::log::Log>>;

    namespace {
        // Log singletons
        struct Logs {
            static LogsType logs_;
        };

        LogsType Logs::logs_;
    } // namespace
} // namespace util::log

namespace util::log {

    //! @class Log
    /*! Prints logging messages to output
     */
    class Log {
    public:
        ~Log()
        {
            std::scoped_lock<std::mutex> lock(lock_);

            if (file_ != stdout) {
                std::fclose(file_);
            }
        }

        explicit Log(util::log::log_level base_log_level)
            : base_log_level_(base_log_level)
        {}

        bool set_file(const std::filesystem::path& path)
        {
            std::scoped_lock<std::mutex> lock(lock_);

            if (!std::filesystem::exists(path)) {
                return false;
            }

            const std::string finalpath = [&path] {
                std::string value;
                std::filesystem::path formatted = path;
                if (formatted.extension().empty())
                    formatted.replace_extension("log");

                std::size_t i = 0;
                while (std::filesystem::exists(formatted)) {
                    ++i;
                    value = std::format("{}.{}", formatted.string(), i);
                }
                return value;
            }();

            if (FILE* file = std::fopen(finalpath.c_str(), "a+"); file) {
                // Close existing open file
                if (file_ != stdout)
                    std::fclose(file_);
                file_ = file;
                return true;
            }

            return false;
        }

        void enable_color_output(const bool yes)
        {
            std::scoped_lock<std::mutex> lock(lock_);

            should_use_color_ = yes && llm_util_log_should_use_colors_auto();
        }

        void set_base_log_level(util::log::log_level level)
        {
            std::scoped_lock<std::mutex> lock(lock_);

            base_log_level_ = level;
        }

        util::log::log_level get_base_log_level() const
        {
            std::scoped_lock<std::mutex> lock(lock_);

            return base_log_level_;
        }

        void print(util::log::log_level level, const char* fmt, va_list args)
        {
            // cannot use args twice, so make a copy in case we need to
            // expand the buffer
            va_list args_copy;
            va_copy(args_copy, args);

            // Peek args list to get eventual buffer size
            const int size = [fmt, &args] {
                char ch = 0;
                return vsnprintf(&ch, 1, fmt, args);
            }();

            std::string message(static_cast<std::size_t>(size + 1), 0);

            if (const int ret
                = vsnprintf(message.data(), message.size(), fmt, args_copy);
                ret == size) {
                print(level, message);
            }

            va_end(args);
            va_end(args_copy);
        }

        void print(const util::log::log_level level, const std::string& message)
        {
            std::scoped_lock<std::mutex> lock(lock_);

            // Just in case
            if (base_log_level_ == log_level::kLogLevelNone) {
                return; // Nothing to do
            }

            if (level < base_log_level_) {
                return; // Nothing to do
            }

            if (message.empty()) {
                return; // Nothing to do
            }

            const auto get_color_string = [this](log_color color) {
                auto i = static_cast<std::uint8_t>(color);
                return should_use_color_ ? kColors[i] : "";
            };

            // Print log timestamp if first message in sequence
            if (level != log_level::kLogLevelCont) {
                // Time since program launch
                std::int64_t timestamp = t_us();
                const auto h = static_cast<int>((timestamp / 1000000) / 3600);
                const auto m = static_cast<int>((timestamp / 1000000) / 60);
                const auto s = static_cast<int>((timestamp / 1000000) % 60);
                const auto ms = static_cast<int>((timestamp / 1000) % 1000);
                if (h == 0) {
                    // [M:s:ms]
                    fprintf(
                        file_,
                        "%s[%d:%02d:%03d]%s ",
                        get_color_string(log_color::kCommonLogColorCyan),
                        m,
                        s,
                        ms,
                        get_color_string(log_color::kCommonLogColorDefault));
                } else {
                    // [H:M:s:ms]
                    fprintf(
                        file_,
                        "%s[%d:%02d:%02d:%03d]%s ",
                        get_color_string(log_color::kCommonLogColorCyan),
                        h,
                        m,
                        s,
                        ms,
                        get_color_string(log_color::kCommonLogColorDefault));
                }
            }

            switch (level) {
                case log_level::kLogLevelInfo:
                {
                    fprintf(
                        file_,
                        "%sI %s",
                        get_color_string(log_color::kCommonLogColorGreen),
                        get_color_string(log_color::kCommonLogColorDefault));
                    break;
                }

                case log_level::kLogLevelWarn:
                {
                    fprintf(
                        file_,
                        "%sW %s",
                        get_color_string(log_color::kCommonLogColorMagenta),
                        get_color_string(log_color::kCommonLogColorDefault));
                    break;
                }

                case log_level::kLogLevelError:
                {
                    fprintf(
                        file_,
                        "%sE %s",
                        get_color_string(log_color::kCommonLogColorRed),
                        get_color_string(log_color::kCommonLogColorDefault));
                    break;
                }

                case log_level::kLogLevelDebug:
                {
                    fprintf(
                        file_,
                        "%sD %s",
                        get_color_string(log_color::kCommonLogColorYellow),
                        get_color_string(log_color::kCommonLogColorDefault));
                    break;
                }

                default:
                {
                    break;
                }
            }

            fprintf(
                file_,
                "%s%s\n",
                message.c_str(),
                get_color_string(log_color::kCommonLogColorDefault));
            fflush(file_);
        }
    private:
        mutable std::mutex lock_;

        util::log::log_level base_log_level_ = log_level::kLogLevelInfo;

        // Output file, stdout by default unless specified
        FILE* file_ = stdout;

        // Uses colors if printing to TTY
        bool should_use_color_ = llm_util_log_should_use_colors_auto();
    };
} // namespace util::log

//
// Public API
//

void util::log::init_log(
    const std::string& key,
    const std::string& outpath,
    util::log::log_level base_log_level)
{
    if (Logs::logs_[key]) {
        return;
    }

    Logs::logs_[key] = std::make_unique<Log>(base_log_level);
    if (!outpath.empty()) {
        Logs::logs_[key]->set_file(outpath);
    }
}

util::log::Log* util::log::get_log(
    const std::string& key,
    util::log::log_level base_log_level)
{
    if (!Logs::logs_[key]) {
        util::log::init_log(key, std::string(), base_log_level);
    }

    return Logs::logs_[key].get();
}

util::log::Log* util::log::get_log_main(util::log::log_level base_log_level)
{
    return get_log("*", base_log_level);
}

util::log::log_level util::log::log_get_level(const Log* log)
{
    return log->get_base_log_level();
}

void util::log::log_set_level(Log* log, util::log::log_level level)
{
    log->set_base_log_level(level);
}

void util::log::log_enable_color_output(Log* log, bool yes)
{
    log->enable_color_output(yes);
}

void util::log::log_print(
    Log* log,
    util::log::log_level level,
    const char* fmt,
    ...)
{
    va_list args;
    va_start(args, fmt);
    log->print(level, fmt, args);
    va_end(args);
}
