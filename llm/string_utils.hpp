#pragma once

#include <llama-cpp.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace util::file {
    //! String utility
    std::string string_escape(const std::string& value);

    //! String utility
    std::string string_join(
        const std::vector<std::string>& values,
        const std::string& separator = "");

    //! String utility
    std::string string_join(
        const std::vector<std::vector<std::string>>& values,
        const std::string& separator = "");

    //! String utility
    std::string string_replace_all(
        const std::string& value,
        const std::string& needle,
        const std::string& replacement);

    //! String utility
    std::vector<std::string> string_split(
        const std::string& value,
        char delimiter,
        bool dont_include_empty = false);

    std::string to_string(
        const std::unordered_map<std::string, std::string>& model_metadata);
} // namespace util::file
