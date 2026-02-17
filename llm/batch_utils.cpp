#include "llm/batch_utils.hpp"
#include <cstdint>
#include <stdexcept>

bool util::file::llm_util_batch_is_clear(const struct llama_batch& batch)
{
    return batch.n_tokens == 0;
}

void util::file::llm_util_batch_add(
    llama_batch& batch,
    llama_token id,
    llama_pos pos,
    const std::vector<llama_seq_id>& seq_ids,
    bool logits)
{
    if (!batch.seq_id[batch.n_tokens]) {
        throw std::invalid_argument("llama_batch size exceeded");
    }

    std::int32_t token_i = batch.n_tokens;
    ++batch.n_tokens;

    batch.token[token_i] = id;
    batch.pos[token_i] = pos;
    batch.n_seq_id[token_i] = static_cast<std::int32_t>(seq_ids.size());
    for (std::size_t seq_i = 0; seq_i < seq_ids.size(); ++seq_i) {
        batch.seq_id[token_i][seq_i] = seq_ids[seq_i];
    }
    batch.logits[token_i] = static_cast<std::int8_t>(logits);
}

void util::file::llm_util_batch_clear(struct llama_batch& batch)
{
    batch.n_tokens = 0;
}
