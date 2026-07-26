#include "llm/gguf_header_parser.hpp"
#include "llm/string_utils.hpp"

util::file::GgufHeaderParser::~GgufHeaderParser()
{
    std::fclose(file_);
}

bool util::file::GgufHeaderParser::init(const std::string& gguf)
{
    if (file_) {
        std::fclose(file_);
    }

    file_ = std::fopen(gguf.c_str(), "r");
    if (file_ == nullptr) {
        return false;
    }

    head_ = 0;
    data_ = std::vector<char>();
    return true;
}

bool util::file::GgufHeaderParser::parse()
{
    if (read_next_chunk() < 24) {
        return false;
    }

    // Header
    if (const std::string magic = pop(4); magic != "GGUF") {
        return false;
    }

    // Header
    auto version = pop<std::uint32_t>();
    (void)version; // Don't need this

    // Header
    auto n_tensors = pop<std::uint64_t>();
    (void)n_tensors; // Don't need this

    // Header
    auto n_metadata = pop<std::uint64_t>();

    for (std::size_t i = 0; i < n_metadata; ++i) {
        // Boundary check
        if (!ensure_data_available(sizeof(std::uint64_t))) {
            return false;
        }

        auto key_size = pop<std::uint64_t>();

        // Boundary check
        if (!ensure_data_available(key_size)) {
            return false;
        }

        std::string key = pop(key_size);

        // Boundary check
        if (!ensure_data_available(sizeof(std::uint32_t))) {
            return false;
        }

        auto type = pop<std::uint32_t>();

        std::string value;
        if (!parse_value(GGUF_TYPE(type), value)) {
            return false;
        }

        metadata_[key] = value;
    }

    return true;
}

const std::unordered_map<std::string, std::string>& util::file::
    GgufHeaderParser::get_metadata() const
{
    return metadata_;
}

std::size_t util::file::GgufHeaderParser::read_next_chunk()
{
    char buffer[100000] = {};
    std::size_t size = std::fread(buffer, sizeof(char), sizeof(buffer), file_);
    if (size > 0) {
        data_.insert(data_.end(), buffer, buffer + size);
    }

    return size;
}

bool util::file::GgufHeaderParser::ensure_data_available(std::size_t needed)
{
    if ((data_.size() - head_) > needed) {
        return true;
    }

    return needed <= read_next_chunk();
}

bool util::file::GgufHeaderParser::parse_value(
    GGUF_TYPE type,
    std::string& value)
{
    switch (type) {
        case GGUF_TYPE::GGUF_BOOL:
        {
            if (!ensure_data_available(sizeof(std::int8_t))) {
                return false;
            }

            auto v = pop<std::int8_t>();
            value = v == 0 ? "false" : "true";

            return true;
        }

        case GGUF_TYPE::GGUF_UINT8:
        case GGUF_TYPE::GGUF_INT8:
        {
            if (!ensure_data_available(sizeof(std::int8_t))) {
                return false;
            }

            auto v = pop<std::int8_t>();
            value = std::to_string(v);

            return true;
        }

        case GGUF_TYPE::GGUF_UINT16:
        case GGUF_TYPE::GGUF_INT16:
        {
            if (!ensure_data_available(sizeof(std::int16_t))) {
                return false;
            }

            auto v = pop<std::int16_t>();
            value = std::to_string(v);

            return true;
        }

        case GGUF_TYPE::GGUF_UINT32:
        case GGUF_TYPE::GGUF_INT32:
        {
            if (!ensure_data_available(sizeof(std::int32_t))) {
                return false;
            }

            auto v = pop<std::int32_t>();
            value = std::to_string(v);

            return true;
        }

        case GGUF_TYPE::GGUF_FLOAT32:
        {
            if (!ensure_data_available(sizeof(float))) {
                return false;
            }

            auto v = pop<float>();
            value = std::to_string(v);

            return true;
        }

        case GGUF_TYPE::GGUF_STRING:
        {
            if (!ensure_data_available(sizeof(std::uint64_t))) {
                return false;
            }

            auto size = pop<std::uint64_t>();

            if (!ensure_data_available(size)) {
                return false;
            }

            value = pop(size);

            return true;
        }

        case GGUF_TYPE::GGUF_ARRAY:
        {
            if (!ensure_data_available(
                    sizeof(std::uint32_t) + sizeof(std::uint64_t))) {
                return false;
            }

            auto array_type = pop<std::uint32_t>();
            auto array_size = pop<std::uint64_t>();

            std::vector<std::string> values;
            for (; array_size > 0; --array_size) {
                std::string element;
                if (!parse_value(GGUF_TYPE(array_type), element)) {
                    return false;
                }
                values.push_back(element);
            }

            value = string_join(values, ";");

            return true;
        }

        case GGUF_TYPE::GGUF_UINT64:
        case GGUF_TYPE::GGUF_INT64:
        {
            if (!ensure_data_available(sizeof(std::uint64_t))) {
                return false;
            }

            auto v = pop<std::uint64_t>();
            value = std::to_string(v);

            return true;
        }

        case GGUF_TYPE::GGUF_FLOAT64:
        {
            if (!ensure_data_available(sizeof(double))) {
                return false;
            }

            auto v = pop<double>();
            value = std::to_string(v);

            return true;
        }
    }

    return false;
}
