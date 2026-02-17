#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))

namespace util::log {

    enum class log_level : std::uint8_t {
        kLogLevelNone = 0,
        kLogLevelDebug = 1,
        kLogLevelInfo = 2,
        kLogLevelWarn = 3,
        kLogLevelError = 4,
        kLogLevelCont = 5, // continue previous log
    };

    // Opaque type
    class Log;

    // Log API
    LOG_ATTRIBUTE_FORMAT(3, 4)
    void log_print(Log* log, log_level level, const char* fmt, ...);

    // Log API
    void init_log(
        const std::string& key,
        const std::string& path = std::string(),
        log_level base_log_level = log_level::kLogLevelInfo);

    // Log API
    Log* get_log(
        const std::string& key,
        util::log::log_level base_log_level = log_level::kLogLevelInfo);

    // Log API
    Log* get_log_main(
        util::log::log_level base_log_level = log_level::kLogLevelInfo);

    // Log API
    log_level log_get_level(const Log* log);

    // Log API
    void log_set_level(Log* log, log_level level);

    // Log API
    void log_enable_color_output(Log* log, bool yes);
} // namespace util::log

#define LOG_TMPL(level, verbosity, ...)                                        \
    do {                                                                       \
        if (util::log::log_get_level(util::log::get_log_main()) >= level) {    \
            util::log::log_print(                                              \
                util::log::get_log_main(), (level), __VA_ARGS__);              \
        }                                                                      \
    }                                                                          \
    while (0)

#define LOG(...) LOG_TMPL(util::log::log_level::kLogLevelNone, 0, __VA_ARGS__)
#define LOGV(verbosity, ...)                                                   \
    LOG_TMPL(util::log::log_level::kLogLevelNone, verbosity, __VA_ARGS__)

#define LOG_INF(...)                                                           \
    LOG_TMPL(util::log::log_level::kLogLevelInfo, 0, __VA_ARGS__)
#define LOG_WRN(...)                                                           \
    LOG_TMPL(util::log::log_level::kLogLevelWarn, 0, __VA_ARGS__)
#define LOG_ERR(...)                                                           \
    LOG_TMPL(util::log::log_level::kLogLevelError, 0, __VA_ARGS__)
#define LOG_DBG(...)                                                           \
    LOG_TMPL(                                                                  \
        util::log::log_level::kLogLevelDebug, LOG_DEFAULT_DEBUG, __VA_ARGS__)
#define LOG_CNT(...)                                                           \
    LOG_TMPL(util::log::log_level::kLogLevelCont, 0, __VA_ARGS__)

#define LOG_INFV(verbosity, ...)                                               \
    LOG_TMPL(util::log::log_level::kLogLevelInfo, verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...)                                               \
    LOG_TMPL(util::log::log_level::kLogLevelWarn, verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...)                                               \
    LOG_TMPL(util::log::log_level::kLogLevelError, verbosity, __VA_ARGS__)
#define LOG_DBGV(verbosity, ...)                                               \
    LOG_TMPL(util::log::log_level::kLogLevelDebug, verbosity, __VA_ARGS__)
#define LOG_CNTV(verbosity, ...)                                               \
    LOG_TMPL(util::log::log_level::kLogLevelCont, verbosity, __VA_ARGS__)
