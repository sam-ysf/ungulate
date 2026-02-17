#pragma once

#include <cstdint>

namespace util::file {
    //! Document processing request type
    enum class RequestType : std::uint8_t { kParseDocument, kCalcEmbeddings };
} // namespace util::file
