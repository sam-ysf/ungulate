#pragma once

#include <llama-cpp.h>
#include <string>
#include <vector>

namespace util::file {
    // tokenizes a string into a vector of tokens
    // should work similar to Python's `tokenizer.encode`
    std::vector<llama_token> llm_util_tokenize(
        const struct llama_context* ctx,
        const std::string& text,
        bool add_special,
        bool parse_special = false);

    std::vector<llama_token> llm_util_tokenize(
        const struct llama_vocab* vocab,
        const std::string& text,
        bool add_special,
        bool parse_special = false);

    // tokenizes a token into a piece, optionally renders special/control tokens
    // should work similar to Python's `tokenizer.id_to_piece`
    std::string llm_util_token_to_piece(
        const struct llama_context* ctx,
        llama_token token,
        bool special = true);

    std::string llm_util_token_to_piece(
        const struct llama_vocab* vocab,
        llama_token token,
        bool special = true);

    // detokenizes a vector of tokens into a string
    // should work similar to Python's `tokenizer.decode`
    // optionally renders special/control tokens
    std::string llm_util_detokenize(
        const struct llama_context* ctx,
        const std::vector<llama_token>& tokens,
        bool special = true);

    std::string llm_util_detokenize(
        const struct llama_vocab* vocab,
        const std::vector<llama_token>& tokens,
        bool special = true);
} // namespace util::file
