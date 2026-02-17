#pragma once

#include "file/search_query_token.hpp"
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <optional>

namespace util::query {

    enum class SearchType : std::uint8_t { kKeyword, kStochastic };

    //! @struct QueryDeserializer
    struct QueryDeserializer {
        SearchType search_type = SearchType::kKeyword;
        std::size_t limit = 0;

        std::vector<util::file::SearchQuery> search_queries;

        std::vector<std::int64_t> blacklisted_files;
        std::vector<std::int64_t> whitelisted_files;

        bool highlight = false;

        /// @param j
        ///     Serialized query JSON.
        void deserialize_search_query(const nlohmann::json& j)
        {
            std::optional<util::file::SearchQuery> search_query
                = parse_search_query(j);
            if (search_query == std::nullopt) {
                return;
            }

            if (std::ranges::find(search_queries, search_query.value())
                == search_queries.end()) {
                search_queries.push_back(search_query.value());
            }
        }

        /// @param j
        ///     Serialized file JSON.
        void deserialize_files(const nlohmann::json& j)
        {
            util::file::TokenInclusionType inclusion_type
                = util::file::TokenInclusionType::kInclude;

            if (!j.contains("uuids")) {
                return;
            }

            if (j.contains("type")) {
                const std::string type = j["type"];
                if (type == "whitelist") {
                    inclusion_type = util::file::TokenInclusionType::kInclude;
                } else if (type == "blacklist") {
                    inclusion_type = util::file::TokenInclusionType::kExclude;
                } else {
                    return;
                }
            }

            if (const nlohmann::json& jj = j["uuids"]; jj.is_array()) {
                for (const nlohmann::json& jjj: jj) {
                    std::int64_t uuid = 0;

                    if (jjj.is_number_integer()) {
                        uuid = static_cast<std::int64_t>(jjj);
                    } else if (jjj.is_string()) {
                        uuid = std::stoi(static_cast<std::string>(jjj));
                    }

                    switch (inclusion_type) {
                        case util::file::TokenInclusionType::kInclude:
                        {
                            whitelisted_files.push_back(uuid);
                            break;
                        }
                        case util::file::TokenInclusionType::kExclude:
                        {
                            blacklisted_files.push_back(uuid);
                            break;
                        }
                    }
                }
            }
        }

        /// @param j_query
        ///     Serialized query JSON.
        /// @param search_query [out]
        ///     Parse output.
        static std::optional<util::file::SearchQuery> parse_search_query(
            const nlohmann::json& j_token)
        {
            std::optional<util::file::SearchQuery> search_query;

            if (j_token.is_object()) {
                std::optional<util::file::SearchQueryToken> token
                    = parse_query_search_term(j_token);
                if (token != std::nullopt) {
                    search_query = util::file::SearchQuery();
                    search_query->tokens.push_back(token.value());
                }
            }

            else if (j_token.is_array()) {
                for (const auto& jj_token: j_token) {
                    std::optional<util::file::SearchQueryToken> token
                        = parse_query_search_term(jj_token);
                    if (token != std::nullopt) {
                        search_query = util::file::SearchQuery();
                        search_query->tokens.push_back(token.value());
                    }
                }
            }

            return search_query;
        }

        static std::string parse_value(const nlohmann::json& j_token)
        {
            if (j_token.contains("value")) {
                return j_token["value"];
            }

            return std::string();
        }

        static enum util::file::TokenFormatType parse_format_type(
            const nlohmann::json& j_token)
        {
            using enum util::file::TokenFormatType;

            if (j_token.contains("parse-format")) {
                std::string format = j_token["parse-format"];
                if (format == "plaintext") {
                    return kPlaintext;
                }

                if (format == "regex") {
                    return kRegex;
                }
            }

            return util::file::TokenFormatType::kPlaintext;
        }

        static enum util::file::TokenInclusionType parse_inclusion_type(
            const nlohmann::json& j_token)
        {
            using enum util::file::TokenInclusionType;

            if (j_token.contains("inclusion-type")) {
                std::string type = j_token["inclusion-type"];
                if (type == "include") {
                    return kInclude;
                }

                if (type == "exclude") {
                    return kExclude;
                }
            }

            return util::file::TokenInclusionType::kInclude;
        }

        /// @param j_token
        ///     Serialized token JSON.
        /// @param token [out]
        ///     Parse output.
        static std::optional<util::file::SearchQueryToken>
        parse_query_search_term(const nlohmann::json& j_token)
        {
            if (!j_token.contains("value")) {
                return std::nullopt;
            }

            util::file::SearchQueryToken token;
            token.format_type = parse_format_type(j_token);
            token.value = parse_value(j_token);
            token.inclusion_type = parse_inclusion_type(j_token);

            return token;
        }
    };
} // namespace util::query
