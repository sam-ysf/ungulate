
#include "dashboard_server.hpp"
#include "cpp-httplib/httplib.h"

app::DashboardServer::DashboardServer()
    : server_(std::make_unique<httplib::Server>())
{}

void app::DashboardServer::listen()
{
    server_->listen_after_bind();
}

void app::DashboardServer::stop()
{
    server_->stop();
}

bool app::DashboardServer::bind(int port)
{
    return server_->bind_to_port("0.0.0.0", port);
}

namespace {
    // Helper
    inline bool mount_config_root(
        httplib::Server& server,
        const std::string& directory)
    {
        return server.set_mount_point("/runtime-config", directory);
    }

    // Helper
    inline bool mount_web_root(
        httplib::Server& server,
        const std::string& directory)
    {
        return server.set_mount_point("/", directory);
    }
} // namespace

bool app::DashboardServer::mount(
    const std::string& web_root,
    const std::string& config_root)
{
    return mount_config_root(*server_, config_root)
           && mount_web_root(*server_, web_root);
}