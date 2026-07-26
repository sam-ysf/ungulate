#include "llm/file_utils.hpp"
#include "llm/gguf_header_parser.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <span>
#include <sstream>
#include <unordered_map>

namespace {

    // Modified base64 encode
    template <typename T>
    std::string base64_encode(const std::span<T>& value)
    {
        // All allowed base 64 characters
        static constexpr const char* kBase64Chars
            = R"(ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=)";

        auto size = static_cast<std::size_t>(
            lround(static_cast<float>(value.size()) * (4.0 / 3.0)));
        std::string out(size, 0);
        auto itr = out.begin();

        // Temp. reusable container for the three chars (octets) that are
        // encoded to a four byte array
        unsigned char octets[3] = {};

        std::size_t i = 0;
        while (i != value.size()) {
            octets[i % 3] = value[i];
            // Encode octets if next character is at the three character
            // boundary
            if ((++i % 3) == 0) {
                // Encode
                int indices[4] = {};

                indices[0] = (octets[0] >> 2) & 0x3f;
                indices[1] = (((octets[0] & 0x03) << 4) & 0xf0)
                             | ((octets[1] >> 4) & 0x0f);
                indices[2] = (((octets[1] & 0x0f) << 2) & 0xfc)
                             | ((octets[2] >> 6) & 0x03);
                indices[3] = octets[2] & 0x3f;

                *itr++ = kBase64Chars[indices[0]];
                *itr++ = kBase64Chars[indices[1]];
                *itr++ = kBase64Chars[indices[2]];
                *itr++ = kBase64Chars[indices[3]];

                // Zero-out
                std::memset(octets, 0, sizeof(octets));
            }
        }

        // Return if no trailing characters
        if ((i % 3) == 0) {
            return out;
        }

        for (; i % 3; ++i) {
            octets[i % 3] = 0;
        }

        // Encode
        const int indices[4]
            = {(octets[0] & 0xfc) >> 2,
               ((octets[0] & 0x03) << 4) | ((octets[1] & 0xf0) >> 4),
               ((octets[1] & 0x0f) << 2) | ((octets[2] & 0xc0) >> 6),
               octets[2] & 0x3f};

        *itr++ = kBase64Chars[indices[0]];
        *itr++ = kBase64Chars[indices[1]];
        *itr++ = kBase64Chars[indices[2]];
        *itr++ = kBase64Chars[indices[3]];

        return out;
    }
} // namespace

std::string util::file::blob_calculate_hash(const std::string& data)
{
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, data.data(), data.size());

    unsigned char digest[32] = {};
    unsigned int digest_len = 0;
    EVP_DigestFinal_ex(ctx, digest, &digest_len);
    EVP_MD_CTX_free(ctx);

    return base64_encode(std::span(digest, digest_len));
}

std::string util::file::fs_read_file(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        return std::string();
    }

    std::istreambuf_iterator<char> beg(file);
    std::string value(beg, std::istreambuf_iterator<char>());

    return file.close(), value;
}

std::string util::file::fs_read_file_magic_string(const std::string& path)
{
    static const std::size_t kBuffSize = 1024;
    // Read magic number
    FILE* file = std::fopen(path.c_str(), "r");
    if (file == nullptr) {
        return std::string();
    }

    std::string buff(kBuffSize, 0);
    auto itr = buff.begin();

    std::size_t size = std::fread(buff.data(), 1, kBuffSize, file);

    const auto end = buff.begin() + static_cast<std::int64_t>(size);
    for (; itr < end; ++itr) {
        if (std::isspace(*itr)) {
            break;
        }
    }

    std::fclose(file);
    return std::string(buff.begin(), itr);
}

std::string util::file::fs_read_binary_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::string();
    }

    // Move to the end to determine the file size
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg); // Move back to the beginning

    if (size <= 0) {
        file.close();
        return std::string();
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return file.close(), buffer.str();
}

bool util::file::fs_write_file(const std::string& path, const std::string& body)
{
    std::ofstream ofs(path, std::ios_base::binary);

    if (!ofs.is_open()) {
        return false;
    }

    std::ranges::copy(body, std::ostreambuf_iterator<char>(ofs));
    return true;
}
