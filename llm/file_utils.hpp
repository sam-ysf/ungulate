#pragma once

#include <string>
#include <vector>

namespace util::file {

    struct llm_util_model_source {
        // model local path
        std::string gguf;
        std::string gguf_hash;
    };

    std::string fs_calculate_hash(const std::string& path);
    std::string blob_calculate_hash(const std::string& data);

    std::string fs_read_file(const std::string& path);
    std::string fs_read_file_magic_string(const std::string& path);

    std::string fs_read_binary_file(const std::string& path);

    bool fs_write_file(const std::string& path, const std::string& body);
} // namespace util::file
