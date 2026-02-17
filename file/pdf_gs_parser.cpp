#include "file/pdf_gs_parser.hpp"
#include "log/log.hpp"
#include <filesystem>
#include <format>
#include <ghostscript/iapi.h>
#include <ghostscript/ierrors.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <source_location>

std::vector<std::string> util::file::pdf_pages_to_images(
    const std::string& work_dir,
    const std::string& file_path)
{
    void* instance = nullptr;
    if (gsapi_new_instance(&instance, nullptr) < 0) {
        LOG_ERR(
            "%s : could not create Ghostscript instance for "
            "file-to-image conversion",
            std::source_location::current().function_name());
        return std::vector<std::string>();
    }

    std::string output_files_str
        = std::format("-sOutputFile={}/%d.png", work_dir);

    std::string name = "ungulate-module-parse-pdf";
    std::string dont_pause = "-dNOPAUSE";
    std::string device = "-sDEVICE=png16m";
    std::string resolution = "-r300";
    std::string quiet = "-q";
    std::string path = file_path;

    std::vector<char*> gs_args
        = {name.data(),
           dont_pause.data(),
           device.data(),
           resolution.data(),
           output_files_str.data(),
           quiet.data(),
           path.data()};

    if (auto gs_args_size = static_cast<std::int32_t>(gs_args.size());
        gsapi_init_with_args(instance, gs_args_size, gs_args.data())) {
        LOG_ERR(
            "%s : file-to-image conversion error",
            std::source_location::current().function_name());
        gsapi_delete_instance(instance);
        return {};
    }

    gsapi_exit(instance);
    gsapi_delete_instance(instance);

    std::vector<std::string> image_paths;
    for (std::int32_t i = 0;; ++i) {
        const std::string image_path
            = std::format("{}/{}.png", work_dir, std::to_string(i + 1));
        // Break when reached last file
        if (!std::filesystem::exists(image_path)) {
            break;
        }

        // Convert to grayscale
        const cv::Mat image_to_cv = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::Mat grayscale;
        cv::cvtColor(image_to_cv, grayscale, cv::COLOR_BGR2GRAY);
        cv::imwrite(image_path, grayscale);

        image_paths.push_back(image_path);
    }

    return image_paths;
}
