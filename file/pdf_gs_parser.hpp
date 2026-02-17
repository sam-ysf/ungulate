#pragma once

#include <string>
#include <vector>

namespace util::file {

    std::vector<std::string> pdf_pages_to_images(
        const std::string& work_dir,
        const std::string& file_path);
} // namespace util::file
