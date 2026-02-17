#pragma once

#include "llm/base_utils.hpp"
#include <atomic>
#include <llama.h>

namespace util::file {

    class LlamaBase {
    public:
        //! @brief Dtor.
        ~LlamaBase();

        //! @brief Ctor.
        explicit LlamaBase(const llm_util_params& params);

        //! Reinits context and sampler
        void initialize(
            enum llama_pooling_type pooling_type,
            bool embeddings_only);

        void teardown();

        //! Clears the existing cache
        void clear_cache();

        int get_index() const;

        llm_util_params get_params() const;

        llama_context* get_context() const;

        llama_model* get_model() const;

        llm_util_sampler* get_sampler() const;
    private:
        bool maybe_initialize_ggml();

        static std::atomic<int> activation_count_;

        mutable std::mutex lock_;

        int gguf_index_ = 0;

        llm_util_params util_params_;

        llm_util_session llama_session_;
    };
} // namespace util::file
