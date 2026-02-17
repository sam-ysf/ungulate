#pragma once

#include <llama-cpp.h>
#include <vector>

namespace util::file {

    bool llm_util_batch_is_clear(const struct llama_batch& batch);

    void llm_util_batch_add(
        struct llama_batch& batch,
        llama_token id,
        llama_pos pos,
        const std::vector<llama_seq_id>& seq_ids,
        bool logits);

    void llm_util_batch_clear(struct llama_batch& batch);
} // namespace util::file
