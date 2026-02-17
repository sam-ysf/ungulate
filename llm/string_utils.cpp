#include "llm/string_utils.hpp"
#include <cstdarg>
#include <cstring>
#include <ggml.h>
#include <llama-cpp.h>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <vector>

std::string util::file::string_escape(const std::string& value)
{
    static const std::regex kSpecialChars(R"([.^$|()*+?\[\]{}\\])");
    return std::regex_replace(value, kSpecialChars, "\\$&");
}

std::string util::file::string_join(
    const std::vector<std::string>& values,
    const std::string& separator)
{
    std::ostringstream result;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::string util::file::string_join(
    const std::vector<std::vector<std::string>>& values,
    const std::string& separator)
{
    std::vector<std::string> accumulator;
    for (const auto& value: values) {
        accumulator.insert(accumulator.end(), value.begin(), value.end());
    }

    return string_join(accumulator, separator);
}

std::string util::file::string_replace_all(
    const std::string& value,
    const std::string& needle,
    const std::string& replace)
{
    if (needle.empty()) {
        return std::string();
    }

    std::string builder;
    builder.reserve(value.length());

    std::size_t last_pos = 0;
    while (true) {
        std::size_t pos = value.find(needle, last_pos);
        if (pos == std::string::npos) {
            break;
        }

        builder.append(value, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + needle.length();
    }

    builder.append(value, last_pos, std::string::npos);
    return builder;
}

std::vector<std::string> util::file::string_split(
    const std::string& value,
    char delimiter,
    bool dont_include_empty)
{
    std::vector<std::string> parts;

    std::string accumulator;
    for (char ch: value) {
        if (ch == delimiter) {
            if (dont_include_empty && accumulator.empty())
                continue;
            parts.push_back(std::move(accumulator));
        } else {
            accumulator += ch;
        }
    }

    if (!accumulator.empty()) {
        parts.push_back(accumulator);
    }

    return parts;
}
