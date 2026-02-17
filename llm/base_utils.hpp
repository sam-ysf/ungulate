#pragma once

#include "llm/cpu_utils.hpp"
#include "llm/file_utils.hpp"
#include "llm/sampling.hpp"
#include <cstdint>
#include <llama-cpp.h>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>

namespace util::file {
    // llama session lifetime
    struct llm_util_session {
        llama_model_ptr model;

        std::string model_hash;
        // FUTURE
        // Incorporate model metadata into model identifier
        std::unordered_map<std::string, std::string> model_metadata;

        llama_context_ptr context;
        llm_util_sampler_ptr sampler;

        bool is_ok() const;

        std::string to_identifier() const;
    };

    enum class llm_util_embedding_normalize_type : std::int8_t {
        kEmbeddingNormalizeNone = -1,
        kEmbeddingNormalizeMax = 0,
        kEmbeddingNormalizeTaxicab = 1,
        kEmbeddingNormalizeEuclidian = 2,
        kEmbeddingNormalizePNorm = 3,
    };

    struct llm_util_model_params {
        // offload params
        std::vector<ggml_backend_dev_t>
            devices; // devices to use for offloading

        std::int32_t n_gpu_layers
            = -1; // number of layers to store in VRAM (-1 - use default)
        std::int32_t main_gpu
            = 0; // the GPU that is used for scratch and small tensors
        llama_split_mode split_mode
            = LLAMA_SPLIT_MODE_LAYER; // how to split the model across GPUs
        // how split tensors should be distributed across GPUs
        std::vector<float> tensor_split = std::vector<float>(128, 0);

        llm_util_model_source source;

        bool use_mmap = true; // use mmap for faster loads
        bool use_mlock = false; // use mlock to keep model in memory

        bool check_tensors = false; // validate tensor data

        // disable extra buffer types (used for weight repacking)
        bool no_extra_bufts = false;

        std::vector<llama_model_kv_override> kv_overrides;
        std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

        // allocated memory for .buft member of tensor_buft_overrides
        std::shared_ptr<std::vector<std::string>> buft_overrides_alloc;

        // optional callback for model loading progress and cancellation:
        // called with a progress value between 0.0 and 1.0.
        // return false from callback to abort model loading or true to continue
        struct progress_callback {
            llama_progress_callback fn = nullptr;
            void* user_data = nullptr;
        };

        struct progress_callback progress_callback;
    };

    struct llm_util_rope_params {
        llama_rope_scaling_type rope_scaling_type
            = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
        float rope_freq_base = 0.0F; // RoPE base frequency
        float rope_freq_scale = 0.0F; // RoPE frequency scaling factor
    };

    struct llm_util_yarn_params {
        float yarn_ext_factor = -1.0F; // YaRN extrapolation mix factor
        float yarn_attn_factor = 1.0F; // YaRN magnitude scaling factor
        float yarn_beta_slow = 1.0F; // YaRN high correction dim
        float yarn_beta_fast = 32.0F; // YaRN low correction dim
        std::uint32_t yarn_orig_ctx = 0; // YaRN original context length
    };

    struct llm_util_context_params {
        // context size
        std::uint32_t n_ctx = 0;

        // logical batch size for prompt processing (must be >=32 to use BLAS)
        std::uint32_t n_batch = 4096;

        // physical batch size for prompt processing (must be >=32 to use BLAS)
        std::uint32_t n_ubatch = 2048;

        // number of parallel sequences to decode
        std::uint32_t n_parallel = 1;

        // size of embeddings vector
        std::int32_t n_embeddings = 0;

        enum llama_pooling_type pooling_type
            = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling type for embeddings
        llama_attention_type attention_type
            = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // attention type for embeddings
        llama_flash_attn_type flash_attn_type
            = LLAMA_FLASH_ATTN_TYPE_AUTO; // whether to use Flash Attention

        llm_util_rope_params rope_params;
        llm_util_yarn_params yarn_params;

        ggml_backend_sched_eval_callback cb_eval = nullptr;
        void* cb_eval_user_data = nullptr;

        // get sentence embeddings
        bool embeddings_only = false;

        // disable performance metrics
        bool no_perf = false;

        // globally disable offload host tensor operations to device
        bool no_op_offload = false;

        // use full-size SWA cache
        bool swa_full = false;
        // enable unified KV cache
        bool kv_unified = false;
        // disable KV offloading
        bool no_kv_offload = false;

        ggml_type cache_type_k = GGML_TYPE_F16; // KV cache data type for the K
        ggml_type cache_type_v = GGML_TYPE_F16; // KV cache data type for the V
    };

    struct llm_util_params {
        std::int32_t n_predict = -1; // new tokens to predict

        struct llm_util_model_params mparams;

        struct llm_util_context_params cparams;

        struct llm_util_sampling_params sampling;

        // cpu optimizations
        struct cpu_params cpuparams;

        // cpu optimizations
        struct cpu_params cpuparams_batch;

        // perform warmup run
        bool warmup = false;

        // normalisation for embeddings
        llm_util_embedding_normalize_type embeddings_normalize
            = llm_util_embedding_normalize_type::kEmbeddingNormalizeEuclidian;

        // separator of classification sequences
        std::string cls_sep = "\t";

        std::string chat_template;

        std::unordered_map<std::string, std::string> chat_template_kwargs;

        bool verbose_parse_result = false;
    };

    llama_model_params llm_util_model_params_to_llama(
        util::file::llm_util_params& params);

    llama_context_params llm_util_context_params_to_llama(
        const util::file::llm_util_params& params);
} // namespace util::file
