#include "config.hpp"
#include "dashboard_server.hpp"
#include "database/sql.hpp"
#include "file/misc_utils.hpp"
#include "file/schema.hpp"
#include "file_index_runner.hpp"
#include "llm/file_utils.hpp"
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

        // Control panel
        const std::vector<std::string> dashboard_paths = {
            "/usr/local/share/ungulate/client", "/usr/share/ungulate/client"};

        for (const std::string& path: dashboard_paths) {
            if (std::filesystem::exists(path)) {
                env["dashboard-dir"] = path;
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

    // Loads control panel
    std::unique_ptr<app::DashboardServer> initialize_dashboard_server(
        const util::Config& config,
        const std::string& dashboard_dir)
    {
        if (!std::filesystem::exists(dashboard_dir)) {
            return nullptr;
        }

        const std::optional<std::string> runtime_config_dirpath
            = util::file::maybe_create_temp_workdir();
        if (!runtime_config_dirpath) {
            return nullptr;
        }

        const auto query_port = config["query-port"];
        if (!query_port) {
            return nullptr;
        }

        const std::string runtime_config = std::format(
            R"(globalThis.__RUNTIME_CONFIG__={{API_PORT:{},API_URL:"127.0.0.1"}})",
            query_port.value());
        const std::filesystem::path runtime_config_path
            = std::filesystem::path(runtime_config_dirpath.value())
              / "config.jsx";
        util::file::fs_write_file(runtime_config_path, runtime_config);

        auto dashboard_server = std::make_unique<app::DashboardServer>();

        int dashboard_port = std::stoi(config["dashboard-port"].value());
        if (!dashboard_server->bind(dashboard_port)) {
            return nullptr;
        }

        if (!dashboard_server->mount(
                dashboard_dir, runtime_config_dirpath.value())) {
            return nullptr;
        }

        LOG_INF("control panel (http) bound to port %d", dashboard_port);

        return dashboard_server;
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

    std::unique_ptr<app::DashboardServer> dashboard_server;
    if (config["dashboard-port"]) {
        if (!env.contains("dashboard-dir")) {
            return 1;
        }

        dashboard_server
            = initialize_dashboard_server(config, env["dashboard-dir"]);
        if (!dashboard_server) {
            return 1;
        }
    }

    auto query_server_worker
        = std::make_unique<std::jthread>([&query_server_runner]() {
        query_server_runner->run();
    });

    std::unique_ptr<std::jthread> dashboard_worker;
    if (config["dashboard-port"]) {
        dashboard_worker
            = std::make_unique<std::jthread>([&dashboard_server]() {
            dashboard_server->listen();
        });
    }

    while (true) {
        if (!global_run_state(nullptr)) {
            break;
        }
    }

    if (config["dashboard-port"]) {
        dashboard_server->stop();
        dashboard_worker->join();
    }

    query_server_runner->stop();
    query_server_worker->join();

    file_index_runner->stop();

    // Done
    return 0;
}
