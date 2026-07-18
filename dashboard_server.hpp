#pragma once

#include <memory>
#include <string>

// Fwd. decl.
namespace httplib {
    class Server;
}

namespace app {
    //! @class DashboardServer
    /*! Control panel
     */
    class DashboardServer {
    public:
        DashboardServer();

        //! @brief Enters run loop
        void listen();

        //! @brief Stops run loop
        void stop();

        //! @brief Binds to port
        bool bind(int port);

        //! @brief Sets root filder
        bool mount(const std::string& web_root, const std::string& config_root);
    private:
        std::unique_ptr<httplib::Server> server_;
    };
} // namespace app