#include "config.hpp"
#include "dashboard_server.hpp"
#include "database/sql.hpp"
#include "file/misc_utils.hpp"
#include "file/schema.hpp"
#include "file_index_runner.hpp"
#include "log/log.hpp"
#include "query_server_runner.hpp"
#include <csignal>
#include <filesystem>
#include <llama.h>
#include <memory>
#include <pwd.h>
#include <source_location>
#include <thread>

namespace {

    inline bool global_run_state(const bool* nextstate)
    {
        static std::atomic<bool> g_run(true);

        if (nextstate) {
            g_run.store(*nextstate);
        }

        return g_run.load();
    }

    //! @param signo
    //!     Type of signal
    void on_sigint(int signo)
    {
        // Sanity check
        if (signo == SIGINT) {
            bool nextstate = false;
            (void)global_run_state(&nextstate);
        }
    }
} // namespace

namespace {

    inline std::vector<std::string> deserialize(
        std::span<const char* const> argv)
    {
        std::vector<std::string> values;

        for (const char* entry: argv) {
            if (!entry) {
                break;
            }

            std::string arg(entry);
            std::size_t n = arg.find_first_of('=');
            std::string key = arg.substr(0, n);

            if (key.empty()) {
                continue;
            }

            if (key.starts_with("--")) {
                values.push_back(key);
            }

            else if (key.starts_with("-")) {
                std::string key_with_prefix = "-" + key;
                values.push_back(key_with_prefix);
            }

            else {
                std::string key_with_prefix = "--" + key;
                values.push_back(key_with_prefix);
            }

            if (n != std::string::npos) {
                std::string value = arg.substr(n + 1);

                if (value.empty()) {
                    continue;
                }

                values.push_back(value);
            }
        }

        return values;
    }

    // Loads command line arguments
    bool load_env(
        std::unordered_map<std::string, std::string>& env,
        std::span<const char* const> argv)
    {
        const std::filesystem::path default_config_dir
            = util::file::get_home_path() / ".config/ungulate/server";

        // SQLite
        const std::vector<std::string> sqlite_vec_paths
            = {"/usr/local/share/ungulate/plugins/vec0.so",
               "/usr/share/ungulate/plugins/vec0.so"};

        for (const std::string& path: sqlite_vec_paths) {
            if (std::filesystem::exists(path)) {
                env["db-extension-vec"] = path;
                break;
            }
        }

        // Home directory is default directory
        env["config-dir"] = default_config_dir;

        // Parse arguments
        std::vector<std::pair<std::string, std::string>> env_args;

        for (const std::string& arg: deserialize(argv)) {
            // Sanity check
            if (arg.empty()) {
                continue;
            }

            // Prints usage
            if (arg == "--help" || arg == "--h" || arg == "--usage") {
                std::fprintf(
                    stdout,
                    "usage: %s [--config-dir=</path/to/conf/dir>] [-h]\n",
                    argv[0]);
                return false;
            }

            // Parse key
            if (arg.starts_with("--")) {
                env_args.emplace_back();
                env_args.back().first = arg.substr(2);
            }

            // Parse value
            else {
                if (!env_args.empty()) {
                    (env_args.back()).second = arg;
                }
            }
        }

        for (const auto& [key, value]: env_args) {
            // Parse /path/to/config/directory
            if (key == "config-dir") {
                // Sanity check
                if (value.empty()) {
                    continue;
                }

                const std::filesystem::path dir
                    = std::filesystem::absolute(value);

                env["config-dir"] = dir;
            }

            // Parse other args
            else {
                env[key] = value;
            }
        }

        return true;
    }
} // namespace

// Entry point
int main(int argc, char** argv)
{
    // Init signal handler
    if (signal(SIGINT, on_sigint) == SIG_ERR) {
        // Print error and exit
        LOG_ERR(
            "%s: error setting SIGINT handler",
            std::source_location::current().function_name());
        return 1;
    }

    // Prepare env.
    util::Env env;
    if (!load_env(
            env, std::span(argv + 1, static_cast<std::size_t>(argc - 1)))) {
        return 1;
    }

    // Generate configuration
    // Sanity check
    if (!env.contains("config-dir")) {
        return 1;
    }

    if (!env.contains("db-extension-vec")) {
        return 1;
    }

    const util::Config config = util::load_config(env["config-dir"]);

    // Clean removed files from database
    util::file::SqlDatabase::Params database_params;
    database_params.path = config["db-path"].value_or(":memory:");
    database_params.schema = util::file::kSchema;
    database_params.extensions.push_back(env["db-extension-vec"]);

    std::shared_ptr<util::file::FileIndexRunner> file_index_runner;

    if (config.ocr_model_configs.empty()) {
        return 1;
    }

    // Use the first defined model config for global file indexing
    const auto itr = std::ranges::find_if(
        config.ocr_model_configs, [](const util::file::OcrModelConfig& value) {
        return value.enabled;
    });

    const auto& model_config = itr == config.ocr_model_configs.end()
                                   ? config.ocr_model_configs[0]
                                   : *itr;

    file_index_runner = util::file::FileIndexRunner::initialize(
        model_config,
        config.post_processing_configs,
        config.embedding_configs,
        database_params,
        env["config-dir"]);

    if (!file_index_runner.get()) {
        return 1;
    }

    auto query_server_runner = util::query::QueryServerRunner::initialize(
        file_index_runner, env["config-dir"], config, database_params);
    if (!query_server_runner.get()) {
        return 1;
    }

    auto query_server_worker
        = std::make_unique<std::jthread>([&query_server_runner]() {
        query_server_runner->run();
    });

    while (true) {
        if (!global_run_state(nullptr)) {
            break;
        }
    }

    query_server_runner->stop();
    query_server_worker->join();

    file_index_runner->stop();

    // Done
    return 0;
}
