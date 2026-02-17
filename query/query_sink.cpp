#include "query/query_sink.hpp"
#include "config.hpp"
#include "database/sql.hpp"
#include "file/dictionaries.hpp"
#include "file/embeddings.hpp"
#include "file/file_parse_update_sink.hpp"
#include "file/misc_utils.hpp"
#include "file/request_type.hpp"
#include "file/search_query_token.hpp"
#include "llm/configs.hpp"
#include "log/log.hpp"
#include "query/matched_page.hpp"
#include "query/query_deserializer.hpp"
#include "query/response.hpp"
#include <algorithm>
#include <nlohmann/detail/output/serializer.hpp>
#include <source_location>
#include <unordered_map>

namespace {
    /*! Helper
     */
    inline util::query::QueryDeserializer deserialize(const nlohmann::json& j)
    {
        util::query::QueryDeserializer qd;

        if (j.contains("query")) {
            qd.deserialize_search_query(j["query"]);
        }

        if (j.contains("search-type") && j["search-type"].is_string()) {
            const std::string type = j["search-type"];
            if (type == "keyword") {
                qd.search_type = util::query::SearchType::kKeyword;
            } else if (type == "stochastic") {
                qd.search_type = util::query::SearchType::kStochastic;
            }
        }

        if (j.contains("limit")) {
            if (const nlohmann::json& jj = j["limit"]; jj.is_number_integer()) {
                qd.limit = static_cast<std::size_t>(jj);
            } else if (jj.is_string()) {
                qd.limit = static_cast<std::size_t>(
                    std::stoi(static_cast<std::string>(jj)));
            }
        }

        if (j.contains("files")) {
            // Handle file specifier
            qd.deserialize_files(j["files"]);
        }

        if (j.contains("queries")) {
            // Handle multiple queries
            for (const nlohmann::json& jj: j["queries"]) {
                qd.deserialize_search_query(jj);
            }
        }

        if (j.contains("highlight")) {
            const nlohmann::json& j_highlight = j["highlight"];
            if (j_highlight.is_string()) {
                std::string value = j_highlight;
                qd.highlight = value == "true";
            }

            if (j_highlight.is_boolean()) {
                bool value = j_highlight;
                qd.highlight = value;
            }
        }

        return qd;
    }
} // namespace

util::query::QuerySink::QuerySink(
    std::shared_ptr<util::file::FileIndexRunner> indexer,
    Config config,
    std::shared_ptr<util::file::UpdateNotifierSink> status,
    util::file::SqlDatabase::Params database_params)
    : index_runner_(std::move(indexer))
    , config_(std::move(config))
    , status_(std::move(status))
    , database_params_(std::move(database_params))
{}

nlohmann::json util::query::QuerySink::get_model_list(
    const nlohmann::json& /* j */) const
{
    nlohmann::json result;

    for (const auto& config: config_.ocr_model_configs) {
        nlohmann::json j;
        j["model"] = config.gguf_hash;
        j["configured"] = true;
        j["description"] = config.description;
        j["enabled"] = config.enabled;
        j["type"] = "ocr";
        result["models"].push_back(j);
    }

    for (const auto& config: config_.embedding_configs) {
        nlohmann::json j;
        j["model"] = config.gguf_hash;
        j["configured"] = true;
        j["description"] = config.description;
        j["enabled"] = config.enabled;
        j["type"] = "embedding";
        result["models"].push_back(j);
    }

    std::unordered_map<std::string, std::string> models
        = index_runner_->get_model_list();
    for (const auto& [model, description]: models) {
        const bool not_in_ocr_configs = std::ranges::none_of(
            config_.ocr_model_configs,
            [&model](const file::OcrModelConfig& element) {
            return element.gguf_hash == model;
        });

        const bool not_in_embedding_configs = std::ranges::none_of(
            config_.embedding_configs,
            [&model](const file::EmbeddingConfig& element) {
            return element.gguf_hash == model;
        });

        if (not_in_ocr_configs && not_in_embedding_configs) {
            nlohmann::json j;
            j["model"] = model;
            j["description"] = description;
            j["enabled"] = false;
            j["configured"] = false;
            result["models"].push_back(j);
        }
    }

    return result;
}

nlohmann::json util::query::QuerySink::get_file_list(
    const nlohmann::json& /* j */) const
{
    auto to_json = [](const util::file::File& file) {
        nlohmann::json j;
        j["uuid"] = file.uuid;
        j["model"] = file.model;
        j["prompt"] = file.prompt;
        j["name"] = file.filename;
        j["n-pages-indexed"] = file.n_pages_indexed;
        j["n-pages"] = file.n_pages;
        j["n-embeddings"] = file.n_embeddings;
        for (const auto& [key, value]: file.metadata) {
            j["metadata"][key] = value;
        }

        return j;
    };

    auto to_json_in_progress = [&to_json](
                                   const util::file::File& file,
                                   util::file::RequestType type,
                                   bool stopping) {
        nlohmann::json j = to_json(file);
        switch (type) {
            case util::file::RequestType::kParseDocument:
            {
                j["request-type"] = "parse-document";
                break;
            }

            case util::file::RequestType::kCalcEmbeddings:
            {
                j["request-type"] = "calc-embeddings";
                break;
            }
        }

        j["stopping"] = stopping;
        return j;
    };

    nlohmann::json result;
    result["files"] = {};

    // Fetch in-progress files
    const std::vector<file::UpdateNotifierSink::FileRequest> in_progress_files
        = status_->get_all();

    for (const auto& request: in_progress_files) {
        const util::file::File& file = request.file;
        const util::file::RequestType type = request.type;
        if (request.running) {
            result["files"].push_back(to_json_in_progress(file, type, false));
        } else {
            result["files"].push_back(to_json_in_progress(file, type, true));
        }
    }

    // Fetch stopped or completed files
    const std::vector<util::file::File> parsed_files
        = index_runner_->get_file_list(true);

    for (const util::file::File& file: parsed_files) {
        const bool not_already_included = std::ranges::none_of(
            in_progress_files,
            [&file](const file::UpdateNotifierSink::FileRequest& element) {
            return element.file.uuid == file.uuid;
        });

        if (not_already_included) {
            result["files"].push_back(to_json(file));
        }
    }

    return result;
}

nlohmann::json util::query::QuerySink::get_file(const nlohmann::json& j) const
{
    if (!j.contains("file")) {
        return request_error_malformed_query();
    }

    std::int64_t file = 0;

    if (const nlohmann::json& jj = j["file"]; jj.is_number_integer()) {
        file = static_cast<std::int64_t>(jj);
    } else if (jj.is_string()) {
        file = std::stoi(static_cast<std::string>(jj));
    } else {
        return request_error_malformed_query();
    }

    // Generate & return JSON response
    std::vector<std::string> content = index_runner_->get_file(file);
    return nlohmann::json(content);
}

nlohmann::json util::query::QuerySink::get_file_path(
    const nlohmann::json& j) const
{
    if (!j.contains("file")) {
        return request_error_malformed_query();
    }

    std::int64_t file = 0;

    if (const nlohmann::json& jj = j["file"]; jj.is_number_integer()) {
        file = static_cast<std::int64_t>(jj);
    } else if (jj.is_string()) {
        file = std::stoi(static_cast<std::string>(jj));
    } else {
        return request_error_malformed_query();
    }

    // Generate & return JSON response
    std::optional<std::string> path = index_runner_->get_file_path(file);
    if (!path) {
        return request_error_bad_argument();
    }

    nlohmann::json jj;
    jj["path"] = path.value();
    return jj;
}

nlohmann::json util::query::QuerySink::delete_files(
    const nlohmann::json& j) const
{
    if (j.contains("file") && j["file"].is_array()) {
        std::vector<std::int64_t> files;
        for (const auto& jj: j["file"]) {
            if (jj.is_number_integer()) {
                files.push_back(static_cast<std::int64_t>(jj));
            } else if (jj.is_string()) {
                files.push_back(std::stoi(std::string(jj)));
            }
        }
        index_runner_->delete_files(files);
    }

    // Generate & return JSON response
    return request_success();
}

nlohmann::json util::query::QuerySink::parse_file(const nlohmann::json& j) const
{
    if (!j.contains("name") || !j.contains("content")) {
        return request_error_malformed_query();
    }

    const std::string& filename = j["name"];
    const std::string& content = j["content"];

    index_runner_->run_parse(filename, content, status_);

    return request_success();
}

nlohmann::json util::query::QuerySink::calc_file_embeddings(
    const nlohmann::json& j) const
{
    if (!index_runner_->has_embedding_extractor()) {
        std::string message = "embedding model not configured";

        const bool have_enabled_models = std::ranges::any_of(
            config_.embedding_configs, [](const file::EmbeddingConfig& config) {
            return config.enabled;
        });

        if (config_.embedding_configs.size() == 1 && !have_enabled_models) {
            message = "embedding model disabled";
        } else if (
            config_.embedding_configs.size() > 1 && !have_enabled_models) {
            message = "embedding models disabled";
        }

        return request_error_not_implemented(
            std::format("Embedding generation unavailable ({})", message));
    }

    if (!j.contains("file")) {
        return request_error_malformed_query();
    }

    std::int64_t file = 0;

    if (const nlohmann::json& jj = j["file"]; jj.is_number_integer()) {
        file = static_cast<std::int64_t>(jj);
    } else if (jj.is_string()) {
        file = std::stoi(static_cast<std::string>(jj));
    } else {
        return request_error_malformed_query();
    }

    if (index_runner_->run_embedding_calc(file, status_)) {
        return request_success();
    }

    return request_error_malformed_query();
}

nlohmann::json util::query::QuerySink::get_file_embeddings(
    const nlohmann::json& j) const
{
    if (!j.contains("file")) {
        return request_error_malformed_query();
    }

    std::int64_t file = 0;

    if (const nlohmann::json& jj = j["file"]; jj.is_number_integer()) {
        file = static_cast<std::int64_t>(jj);
    } else if (jj.is_string()) {
        file = std::stoi(static_cast<std::string>(jj));
    } else {
        return request_error_malformed_query();
    }

    // Generate & return JSON response
    std::vector<util::file::EmbeddingLookupResult> values
        = index_runner_->get_file_embeddings(file);

    if (values.empty()) {
        return std::string();
    }

    nlohmann::json output;
    for (const auto& value: values) {
        nlohmann::json jj;
        jj["model"] = value.model;
        jj["page"] = value.page;
        jj["embeddings"] = value.embeddings;
        output.push_back(std::move(jj));
    }

    return output;
}

nlohmann::json util::query::QuerySink::pause_files(
    const nlohmann::json& j) const
{
    if (j.contains("file") && j["file"].is_array()) {
        std::vector<std::int64_t> files;
        for (const auto& jj: j["file"]) {
            if (jj.is_number_integer())
                files.push_back(static_cast<std::int64_t>(jj));
            else if (jj.is_string())
                files.push_back(std::stoi(std::string(jj)));
        }
        index_runner_->pause_files(files);
    }

    // Generate & return JSON response
    return request_success();
}

namespace {

    /*! Helper
     */
    inline std::string generate_formatted_query_string(
        const std::vector<util::file::SearchQuery>& queries)
    {
        std::string value;
        for (const auto& query: queries) {
            for (const auto& token: query.tokens) {
                if (!value.empty()) {
                    value += " ";
                }

                if (token.inclusion_type
                    == util::file::TokenInclusionType::kExclude)
                    value += "(NOT)";
                value += token.value;
            }
        }

        return value;
    }

    /*! Helper
     */
    inline nlohmann::json generate_search_query_response(
        const util::query::SearchType& search_type,
        const std::vector<util::file::SearchQuery>& queries,
        const std::vector<util::query::MatchedPage>& matched_pages,
        const std::vector<util::query::MatchedSnippet>& matched_snippets)
    {
        nlohmann::json response;
        response["status"] = 200;
        response["query"] = generate_formatted_query_string(queries);
        for (const auto& page: matched_pages)
            response["pages"].push_back(to_json(page));
        for (const auto& snippet: matched_snippets)
            response["snippets"].push_back(to_json(snippet));
        switch (search_type) {
            case util::query::SearchType::kKeyword:
            {
                response["search-type"] = "keyword";
                break;
            }

            case util::query::SearchType::kStochastic:
            {
                response["search-type"] = "stochastic";
                break;
            }
        }

        return response;
    }
} // namespace

nlohmann::json util::query::QuerySink::search(const nlohmann::json& j) const
{
    try {
        const QueryDeserializer qd = deserialize(j);

        const std::vector<util::file::SearchQuery>& queries = qd.search_queries;
        if (queries.empty()) {
            return request_error_malformed_query();
        }

        util::file::SqlDatabase database(database_params_);

        std::vector<util::query::MatchedPage> matched_pages;
        std::vector<util::query::MatchedSnippet> matched_snippets;

        if (qd.search_type == SearchType::kKeyword) {
            keyword_match(
                queries,
                database,
                &matched_pages,
                &matched_snippets,
                qd.blacklisted_files,
                qd.whitelisted_files,
                qd.highlight);

            // Sort in ascending page-order
            std::ranges::sort(
                matched_pages,
                [](const MatchedPage& lhs, const MatchedPage& rhs) {
                if (lhs.snippets.size() != rhs.snippets.size())
                    return lhs.snippets.size() > rhs.snippets.size();
                return lhs.id_in_file < rhs.id_in_file;
            });

            if (0 < qd.limit && qd.limit < matched_pages.size()) {
                matched_pages.resize(qd.limit);
            }

            // Generate & return JSON response
            return generate_search_query_response(
                SearchType::kKeyword, queries, matched_pages, matched_snippets);
        }

        if (qd.search_type == SearchType::kStochastic) {
            // Ensure embeddings module configured and enabled
            if (!index_runner_->has_embedding_extractor()) {
                std::string message = "embedding model not configured";

                const bool have_enabled_models = std::ranges::any_of(
                    config_.embedding_configs,
                    [](const file::EmbeddingConfig& config) {
                    return config.enabled;
                });

                if (config_.embedding_configs.size() == 1
                    && !have_enabled_models) {
                    message = "embedding model disabled";
                } else if (
                    config_.embedding_configs.size() > 1
                    && !have_enabled_models) {
                    message = "embedding models disabled";
                }

                return request_error_not_implemented(
                    std::format(
                        "Stochastic matching unavailable ({})", message));
            }

            // FUTURE
            // configurable default limit
            std::size_t limit = qd.limit > 0 ? qd.limit : 10;

            stochastic_match(
                queries,
                database,
                limit,
                &matched_pages,
                qd.blacklisted_files,
                qd.whitelisted_files);

            // Sort in ascending page-order
            std::ranges::sort(
                matched_pages,
                [](const MatchedPage& lhs, const MatchedPage& rhs) {
                if (std::abs(lhs.rank - rhs.rank) > 1e-6)
                    return lhs.rank < rhs.rank;
                return lhs.id_in_file < rhs.id_in_file;
            });

            // Generate & return JSON response
            return generate_search_query_response(
                SearchType::kStochastic,
                queries,
                matched_pages,
                matched_snippets);
        }

        // Unknown search type
        return request_error_malformed_query();

    } catch (const nlohmann::detail::parse_error& e) {
        LOG_ERR(
            "%s: %s",
            std::source_location::current().function_name(),
            e.what());
    } catch (const std::exception& e) {
        LOG_ERR(
            "%s: %s",
            std::source_location::current().function_name(),
            e.what());
    }

    return request_error_internal();
}

void util::query::QuerySink::keyword_match(
    const std::vector<util::file::SearchQuery>& queries,
    const util::file::SqlDatabase& database,
    std::vector<MatchedPage>* matched_pages /* out */,
    std::vector<MatchedSnippet>* matched_snippets_result /* out */,
    const std::vector<std::int64_t>& blacklisted_files,
    const std::vector<std::int64_t>& whitelisted_files,
    bool highlight) const
{
    // Match query against indexed keywords directly
    for (const auto& query: queries) {
        keyword_match_impl(
            query.tokens,
            database,
            matched_pages,
            matched_snippets_result,
            blacklisted_files,
            whitelisted_files,
            highlight);
    }
}

void util::query::QuerySink::keyword_match_impl(
    const std::vector<util::file::SearchQueryToken>& query,
    const util::file::SqlDatabase& database,
    std::vector<MatchedPage>* matched_pages /* out */,
    std::vector<MatchedSnippet>* matched_snippets_result /* out */,
    const std::vector<std::int64_t>& blacklisted_files,
    const std::vector<std::int64_t>& whitelisted_files,
    bool highlight) const
{
    util::file::LookupResults results = index_runner_->search(
        query, database, blacklisted_files, whitelisted_files, highlight);

    std::unordered_map<
        util::file::Page,
        util::query::MatchedPage,
        util::file::Page::Key>
        page_ids_to_matched_pages;

    for (const util::file::PageLookupResults& result:
         results.page_lookup_results) {
        const util::file::File& file = result.file;

        for (const auto& value: result.values) {
            util::query::MatchedPage& matched_page
                = page_ids_to_matched_pages[value.page];

            matched_page.file = file.uuid;
            matched_page.model = file.model;
            matched_page.prompt = file.prompt;
            matched_page.filename = file.filename;
            matched_page.id_in_file = value.page.id_in_file;
            matched_page.snippets.push_back(value.snippet);
            matched_page.highlight_tags = results.highlight_tags;
        }
    }

    std::size_t size = matched_pages->size() + page_ids_to_matched_pages.size();
    matched_pages->reserve(size);
    for (auto& [key, value]: page_ids_to_matched_pages) {
        matched_pages->push_back(std::move(value));
    }

    for (auto& result: results.snippet_lookup_results) {
        MatchedSnippet snippet;
        snippet.filename = result.file.filename;
        snippet.model = result.file.model;
        snippet.prompt = result.file.prompt;
        snippet.file = result.file.uuid;
        snippet.page_in_file = result.value.page.id_in_file;
        snippet.snippet = std::move(result.value.snippet);
        snippet.rank = result.value.rank;
        snippet.highlight_tags = results.highlight_tags;
        matched_snippets_result->push_back(std::move(snippet));
    }
}

void util::query::QuerySink::stochastic_match(
    const std::vector<util::file::SearchQuery>& queries,
    const util::file::SqlDatabase& database,
    std::size_t limit,
    std::vector<MatchedPage>* matched_pages /* out */,
    const std::vector<std::int64_t>& blacklisted_files,
    const std::vector<std::int64_t>& whitelisted_files) const
{
    // Match query against indexed vector embeddings
    for (const auto& query: queries) {
        stochastic_match_pages(
            query.tokens,
            database,
            limit,
            matched_pages,
            blacklisted_files,
            whitelisted_files);
    }
}

namespace util::query {

    namespace {
        /*! Hash function for {file, page}
         */
        struct MatchedPageHash {
            std::size_t operator()(
                const std::tuple<std::int64_t, std::int64_t>& value) const
            {
                auto v1 = static_cast<std::size_t>(std::get<0>(value));
                auto v2 = static_cast<std::size_t>(std::get<1>(value));
                return v1 ^ ((v2 + 0x9e3779b9) + (v1 << 6) + (v2 >> 2));
            }
        };

        using MatchedPagesType = std::unordered_map<
            std::tuple<std::int64_t, std::int64_t>,
            MatchedPage,
            MatchedPageHash>;
    } // namespace
} // namespace util::query

void util::query::QuerySink::stochastic_match_pages(
    const std::vector<util::file::SearchQueryToken>& query,
    const util::file::SqlDatabase& database,
    std::size_t limit,
    std::vector<MatchedPage>* matched_pages /* out */,
    const std::vector<std::int64_t>& blacklisted_files,
    const std::vector<std::int64_t>& whitelisted_files) const
{
    std::set<std::int64_t> excluded_files(
        blacklisted_files.begin(), blacklisted_files.end());
    std::set<std::int64_t> included_files(
        whitelisted_files.begin(), whitelisted_files.end());

    /* Helper
     */
    const auto select_matched_page
        = [&included_files, &excluded_files](
              const file::EmbeddingsResult& result,
              MatchedPagesType& page_ids_to_matched_pages /* out */) {
        if (!included_files.empty()) {
            if (!included_files.contains(result.file)) {
                return;
            }
        }

        else if (excluded_files.contains(result.file)) {
            return;
        }

        const auto key = std::make_tuple(result.file, result.page_in_file);
        const auto itr = page_ids_to_matched_pages.find(key);
        if (itr == page_ids_to_matched_pages.end()) {
            page_ids_to_matched_pages[key] = util::query::MatchedPage(result);
        } else {
            auto& [k, page] = *itr;
            page.rank = std::min(page.rank, result.distance);
        }
    };

    /* Helper
     */
    const auto generate_page_results
        = [&select_matched_page](
              const std::vector<util::file::EmbeddingsResult>&
                  embeddings_results) {
        MatchedPagesType page_ids_to_matched_pages;

        for (const auto& result: embeddings_results) {
            select_matched_page(result, page_ids_to_matched_pages);
        }

        std::vector<util::query::MatchedPage> matched_pages;
        matched_pages.reserve(page_ids_to_matched_pages.size());
        for (auto& [key, value]: page_ids_to_matched_pages) {
            matched_pages.push_back(std::move(value));
        }

        return matched_pages;
    };

    const auto embeddings_results
        = index_runner_->search_embeddings(query, database, limit);

    const std::vector<MatchedPage> page_results
        = generate_page_results(embeddings_results);
    matched_pages->insert(
        matched_pages->end(), page_results.begin(), page_results.end());
}
