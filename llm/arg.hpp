#pragma once

#include "llm/base_utils.hpp"
#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace util::file {

    struct Arg {
        std::vector<std::string> args_list;
        // help texts or examples for arg values
        std::vector<std::string> value_hints;

        std::string help;

        bool is_sparam = false; // whether current arg is a sampling param

        using ArgHandlerType = std::function<
            void(llm_util_params& params, const std::string& args)>;
        int n_args = 0;

        ArgHandlerType arg_handler = [](llm_util_params&, const std::string&) {
            /* Default handler */
        };

        Arg(const std::initializer_list<std::string>& args_list,
            std::string help,
            ArgHandlerType arg_handler)
            : args_list(args_list)
            , help(std::move(help))
            , arg_handler(std::move(arg_handler))
        {}

        Arg(const std::initializer_list<std::string>& args_list,
            const std::vector<std::string>& value_hints,
            std::string help,
            ArgHandlerType arg_handler)
            : args_list(args_list)
            , value_hints(value_hints)
            , help(std::move(help))
            , n_args(static_cast<int>(value_hints.size()))
            , arg_handler(std::move(arg_handler))
        {}

        Arg& set_sparam(bool enable = true);

        nlohmann::json to_json() const;
    };

    //! Parses input arguments from CL
    std::optional<llm_util_params> generate_llm_util_params(
        const std::unordered_map<std::string, std::string>& params,
        const llm_util_params& param_hints = llm_util_params());

    nlohmann::json llm_param_usage_to_json(const std::vector<Arg>& args);

    nlohmann::json llm_builtin_chat_templates_to_json();
} // namespace util::file
