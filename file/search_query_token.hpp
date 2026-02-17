#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace util::file {

    enum class TokenFormatType : std::uint8_t {
        kPlaintext,
        kRegex,
    };

    enum class TokenInclusionType : std::uint8_t {
        kInclude,
        kExclude,
    };

    struct SearchQueryToken {
        TokenFormatType format_type = TokenFormatType::kPlaintext;
        std::string value;
        TokenInclusionType inclusion_type = TokenInclusionType::kInclude;

        SearchQueryToken() = default;
        explicit SearchQueryToken(std::string v)
            : value(std::move(v))
        {}

        bool operator==(const SearchQueryToken& rhs) const = default;
    };

    struct SearchQuery {
        std::vector<SearchQueryToken> tokens;

        bool operator==(const SearchQuery& rhs) const
        {
            if (tokens.size() != rhs.tokens.size()) {
                return false;
            }

            std::size_t i = 0;
            for (; i != tokens.size(); ++i) {
                if (tokens[i] != rhs.tokens[i])
                    return false;
            }

            return true;
        }
    };
} // namespace util::file
