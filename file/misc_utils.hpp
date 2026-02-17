#include "log/log.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <pwd.h>
#include <source_location>
#include <string>

namespace util::file {

    inline std::filesystem::path get_home_path()
    {
        std::size_t bufflen = std::max<std::size_t>(
            static_cast<std::size_t>(sysconf(_SC_GETPW_R_SIZE_MAX)),
            std::numeric_limits<std::uint16_t>::max());
        auto data = std::make_unique<char[]>(bufflen);

        passwd* check = nullptr;
        passwd passwd_out;
        if (::getpwuid_r(::getuid(), &passwd_out, data.get(), bufflen, &check)
            || (check != &passwd_out)) {
            return std::filesystem::path();
        }

        return std::filesystem::path(passwd_out.pw_dir);
    }

    inline std::optional<std::string> maybe_create_dir(
        const std::filesystem::path& workdir)
    {
        if (std::filesystem::exists(workdir)) {
            return workdir;
        }

        // Try to create working dir
        if (std::error_code err;
            !std::filesystem::create_directories({workdir}, err)) {
            LOG_ERR(
                "%s : %s",
                std::source_location::current().function_name(),
                (err.message()).c_str());
            return std::nullopt;
        }

        return workdir;
    }

    inline std::optional<std::string> maybe_create_temp_workdir()
    {
        char workdir[] = "/tmp/ungulateXXXXXX";
        if (!mkdtemp(workdir)) {
            return std::nullopt;
        }

        return std::string(workdir);
    }

    inline std::optional<std::string> maybe_create_blobs_dir(
        const std::string& config_dir)
    {
        auto dir = std::filesystem::path(config_dir);
        std::filesystem::path workdir = dir / "blobs";

        return util::file::maybe_create_dir(workdir);
    }

    inline std::optional<std::string> maybe_create_files_dir(
        const std::string& config_dir)
    {
        auto dir = std::filesystem::path(config_dir);
        std::filesystem::path workdir = dir / "files";

        return util::file::maybe_create_dir(workdir);
    }
} // namespace util::file
