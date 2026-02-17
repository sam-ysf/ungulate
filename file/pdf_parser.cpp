#include "file/pdf_parser.hpp"
#include "database/sql.hpp"
#include "file/document.hpp"
#include "file/file_parse_update_sink.hpp"
#include "file/indices.hpp"
#include "file/misc_utils.hpp"
#include "file/pdf_gs_parser.hpp"
#include "log/log.hpp"
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-global.h>
#include <poppler/cpp/poppler-page.h>
#include <source_location>
#include <string>
#include <unordered_map>

namespace {

    struct poppler_document_deleter {
        void operator()(poppler::document* document)
        {
            delete document;
        }
    };

    using poppler_document_ref
        = std::unique_ptr<poppler::document, poppler_document_deleter>;

    struct Metadata {
        std::unordered_map<std::string, std::string> kv;
        std::int32_t n_pages = 0;
    };

    inline std::string to_std_string(const poppler::ustring& str)
    {
        std::vector<char> utf8 = str.to_utf8();
        return std::string(utf8.begin(), utf8.end());
    }

    // Helper
    inline std::optional<Metadata> get_file_metadata(
        const std::string& file_path)
    {
        poppler_document_ref document(
            poppler::document::load_from_file(file_path));
        if (!document) {
            return std::nullopt;
        }

        Metadata metadata;
        metadata.n_pages = document->pages();

        std::string author = to_std_string(document->get_author());
        std::string creator = to_std_string(document->get_creator());
        std::string keywords = to_std_string(document->get_keywords());
        std::string producer = to_std_string(document->get_producer());
        std::string subject = to_std_string(document->get_subject());
        std::string title = to_std_string(document->get_title());

        if (!author.empty()) {
            metadata.kv["Author"] = author;
        }

        if (!creator.empty()) {
            metadata.kv["Creator"] = creator;
        }

        if (!keywords.empty()) {
            metadata.kv["Keywords"] = keywords;
        }

        if (!producer.empty()) {
            metadata.kv["Producer"] = producer;
        }

        if (!subject.empty()) {
            metadata.kv["Subject"] = subject;
        }

        if (!title.empty()) {
            metadata.kv["Title"] = title;
        }

        int major_version = 0;
        int minor_version = 0;
        document->get_pdf_version(&major_version, &minor_version);

        std::string major_version_str = std::to_string(major_version);
        std::string minor_version_str = std::to_string(minor_version);
        metadata.kv["Filetype"]
            = "PDF " + major_version_str + "." + minor_version_str;

        return metadata;
    }
} // namespace

namespace {

    bool image_to_document(
        util::file::DocumentExtractor& extractor,
        const util::file::File& file,
        const std::string& image_path,
        util::file::SqlDatabase& database,
        const std::shared_ptr<util::file::FileUpdateNotifier>& status)
    {
        // Pass to llm and return result
        std::string result = extractor.extract_text(image_path, status);

        if (!status->is_still_set()) {
            return false;
        }

        util::file::save_document(
            result, file.uuid, file.n_pages_indexed + 1, database);
        return true;
    }
} // namespace

bool util::file::PdfParser::parse(
    util::file::DocumentExtractor& extractor,
    File* file,
    const SqlDatabase::Params& database_params,
    const std::shared_ptr<FileUpdateNotifier>& status) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    // Sanity check
    if (!std::filesystem::exists(file->path)) {
        LOG_ERR(
            "%s : file %s does not exist",
            std::source_location::current().function_name(),
            file->path.c_str());
        return false;
    }

    // Get number of pages
    // Returns 'nullopt' if error in opening document
    auto metadata = get_file_metadata(file->path);
    if (!metadata) {
        return false;
    }

    auto workdir = maybe_create_temp_workdir();
    if (!workdir) {
        return false;
    }

    SqlDatabase database(database_params);

    file->n_pages = metadata->n_pages;
    file->metadata = metadata->kv;
    save(*file, database);

    // Save all document pages to PNG for OCR
    std::vector<std::string> image_paths
        = pdf_pages_to_images(workdir.value(), file->path);

    const auto n_pages = static_cast<std::int32_t>(image_paths.size());
    for (; file->n_pages_indexed < n_pages; ++file->n_pages_indexed) {
        const auto i = static_cast<std::size_t>(file->n_pages_indexed);

        if (!status->try_set(*file)) {
            break;
        }

        if (!image_to_document(
                extractor, *file, image_paths[i], database, status)) {
            break;
        }
    }

    std::filesystem::remove_all(workdir.value());
    return true;
}