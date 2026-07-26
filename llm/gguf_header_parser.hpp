#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace util::file {
    class GgufHeaderParser {
    public:
        ~GgufHeaderParser();

        bool init(const std::string& gguf);

        bool parse();

        const std::unordered_map<std::string, std::string>& get_metadata()
            const;
    private:
        enum class GGUF_TYPE : std::uint8_t {
            GGUF_UINT8 = 0,
            GGUF_INT8 = 1,
            GGUF_UINT16 = 2,
            GGUF_INT16 = 3,
            GGUF_UINT32 = 4,
            GGUF_INT32 = 5,
            GGUF_FLOAT32 = 6,
            GGUF_BOOL = 7,
            GGUF_STRING = 8,
            GGUF_ARRAY = 9,
            GGUF_UINT64 = 10,
            GGUF_INT64 = 11,
            GGUF_FLOAT64 = 12
        };

        template <typename T>
        T pop()
        {
            T v = 0;
            std::memcpy(&v, &data_[head_], sizeof(T));
            head_ += sizeof(T);
            return v;
        }

        std::string pop(std::size_t n)
        {
            std::string value(n, 0);
            std::memcpy(value.data(), &data_[head_], n);
            head_ += n;
            return value;
        }

        std::size_t read_next_chunk();

        bool ensure_data_available(std::size_t needed);

        bool parse_value(GGUF_TYPE type, std::string& value /* out */);

        FILE* file_ = nullptr;

        std::size_t head_ = 0;
        std::vector<char> data_;

        std::unordered_map<std::string, std::string> metadata_;
    };
} // namespace util::file