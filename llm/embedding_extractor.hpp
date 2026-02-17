#pragma once

#include "file/embeddings.hpp"
#include "llm/base_utils.hpp"
#include "llm/configs.hpp"
#include "llm/extractor_base.hpp"
#include "llm/llama_base.hpp"
#include <llama-cpp.h>
#include <llama.h>
#include <mutex>
#include <string>
#include <vector>

namespace util::file {

    class EmbeddingExtractor : public ExtractorBase {
    public:
        ~EmbeddingExtractor() override = default;

        EmbeddingExtractor(
            const std::shared_ptr<LlamaBase>& llama,
            EmbeddingConfig config);

        std::vector<Embeddings> extract(
            const std::vector<std::string>& prompts) const;

        void initialize();

        void teardown();

        bool is_same(const LlamaBase* ptr) const override;

        bool is_not_same(const LlamaBase* ptr) const override;

        bool has_same_base_object(const ExtractorBase& rhs) const override;

        EmbeddingConfig get_config() const;

        std::string to_identifier() const;

        llm_util_params get_llama_params() const;
    private:
        std::uint32_t calc_n_embd_count(
            std::uint32_t n_prompts,
            const std::vector<std::vector<std::int32_t>>& tokens) const;

        std::vector<util::file::Embeddings> calc_embeddings(
            std::uint32_t n_prompts,
            const std::vector<std::vector<std::int32_t>>& tokens) const;

        std::vector<std::vector<float>> decode_embeddings(
            const std::vector<std::vector<llama_token>>& tokens_set) const;

        std::vector<float> decode_embeddings_batches(
            const std::vector<std::vector<llama_token>>& tokens_set) const;

        //! Decodes batched prompts
        void decode_embeddings_batch(
            const llama_batch& batch,
            float* output /* out */,
            std::uint32_t n_seq,
            std::uint32_t n_embd,
            llm_util_embedding_normalize_type embd_norm) const;

        //! Generates tokens from prompt
        std::vector<llama_token> tokenize_prompt(
            const std::string& prompt) const;

        //! Generates tokens from prompt set
        std::vector<std::vector<llama_token>> tokenize_prompts(
            const std::vector<std::string>& prompts) const;

        mutable std::mutex lock_;

        //! Llama base api wrapper
        const std::shared_ptr<LlamaBase> llama_;
        //! Gguf config
        const EmbeddingConfig config_;

        //! Sequence-level representation
        enum llama_pooling_type default_pooling_type_
            = llama_pooling_type::LLAMA_POOLING_TYPE_CLS;
    };
} // namespace util::file
