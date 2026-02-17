#include "llm/vocab_utils.hpp"
#include <limits>
#include <stdexcept>

std::vector<llama_token> util::file::llm_util_tokenize(
    const llama_context* ctx,
    const std::string& text,
    bool add_special,
    bool parse_special)
{
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llm_util_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> util::file::llm_util_tokenize(
    const llama_vocab* vocab,
    const std::string& text,
    bool add_special,
    bool parse_special)
{
    // upper limit for the number of tokens
    std::vector<llama_token> result(
        add_special ? (text.size() + 2) : text.size());
    std::int32_t n_tokens = llama_tokenize(
        vocab,
        text.data(),
        static_cast<std::int32_t>(text.size()),
        result.data(),
        static_cast<std::int32_t>(result.size()),
        add_special,
        parse_special);

    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::overflow_error("Tokenization failed due to result overflow");
    }

    if (n_tokens < 0) {
        // buffer is too small to contain token output
        // resize buffer and retry
        result.resize(static_cast<std::size_t>(std::abs(n_tokens)));
        int check = llama_tokenize(
            vocab,
            text.data(),
            static_cast<std::int32_t>(text.size()),
            result.data(),
            static_cast<std::int32_t>(result.size()),
            add_special,
            parse_special);
        if (check < 0) {
            throw std::runtime_error(
                "Tokenization failed due to internal error");
        }
    } else {
        result.resize(static_cast<std::size_t>(n_tokens));
    }
    return result;
}

std::string util::file::llm_util_token_to_piece(
    const llama_context* ctx,
    llama_token token,
    bool special)
{
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llm_util_token_to_piece(vocab, token, special);
}

std::string util::file::llm_util_token_to_piece(
    const llama_vocab* vocab,
    llama_token token,
    bool special)
{
    std::string piece;
    piece.resize(
        piece.capacity()); // using string internal cache, 15 bytes + '\n'
    std::int32_t n_chars = llama_token_to_piece(
        vocab,
        token,
        piece.data(),
        static_cast<std::int32_t>(piece.size()),
        0,
        special);

    piece.resize(static_cast<std::size_t>(std::abs(n_chars)));
    if (n_chars < 0) {
        int check = llama_token_to_piece(
            vocab,
            token,
            piece.data(),
            static_cast<std::int32_t>(piece.size()),
            0,
            special);
        if (check < 0) {
            throw std::runtime_error(
                "Token mapping failed due to internal error");
        }
    }

    return piece;
}

std::string util::file::llm_util_detokenize(
    const llama_context* ctx,
    const std::vector<llama_token>& tokens,
    bool special)
{
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    return llm_util_detokenize(vocab, tokens, special);
}

std::string util::file::llm_util_detokenize(
    const llama_vocab* vocab,
    const std::vector<llama_token>& tokens,
    bool special)
{
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    std::int32_t n_chars = llama_detokenize(
        vocab,
        tokens.data(),
        static_cast<std::int32_t>(tokens.size()),
        text.data(),
        static_cast<std::int32_t>(text.size()),
        false,
        special);
    if (n_chars < 0) {
        text.resize(static_cast<std::size_t>(std::abs(n_chars)));
        int check = llama_detokenize(
            vocab,
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            text.data(),
            static_cast<int32_t>(text.size()),
            false,
            special);
        if (check < 0) {
            throw std::runtime_error(
                "Tokenization failed due to internal error");
        }
    }

    return text;
}
