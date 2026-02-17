#include "llm/base_utils.hpp"
#include <nlohmann/detail/output/serializer.hpp>

bool util::file::llm_util_session::is_ok() const
{
    return model && !model_hash.empty() && !model_metadata.empty() && context
           && sampler;
}

llama_model_params util::file::llm_util_model_params_to_llama(
    util::file::llm_util_params& params)
{
    auto mparams = llama_model_default_params();

    if (!params.mparams.devices.empty()) {
        mparams.devices = params.mparams.devices.data();
    }

    if (params.mparams.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.mparams.n_gpu_layers;
    }

    mparams.main_gpu = params.mparams.main_gpu;
    mparams.split_mode = params.mparams.split_mode;
    mparams.tensor_split = params.mparams.tensor_split.data();
    mparams.use_mmap = params.mparams.use_mmap;
    mparams.use_mlock = params.mparams.use_mlock;
    mparams.check_tensors = params.mparams.check_tensors;
    mparams.use_extra_bufts = !params.mparams.no_extra_bufts;

    mparams.kv_overrides = nullptr;
    if (!params.mparams.kv_overrides.empty()) {
        // Maybe add zero-terminator
        if (auto& kv_overrides = params.mparams.kv_overrides;
            kv_overrides.back().key[0] != 0) {
            kv_overrides.emplace_back();
        }
        mparams.kv_overrides = params.mparams.kv_overrides.data();
    }

    mparams.tensor_buft_overrides = nullptr;
    if (!params.mparams.tensor_buft_overrides.empty()) {
        // Maybe add zero-terminator
        if (auto& tensor_buft_overrides = params.mparams.tensor_buft_overrides;
            tensor_buft_overrides.back().pattern != nullptr) {
            tensor_buft_overrides.push_back(llama_model_tensor_buft_override{});
        }
        mparams.tensor_buft_overrides
            = params.mparams.tensor_buft_overrides.data();
    }

    mparams.progress_callback = params.mparams.progress_callback.fn;
    mparams.progress_callback_user_data
        = params.mparams.progress_callback.user_data;

    return mparams;
}

llama_context_params util::file::llm_util_context_params_to_llama(
    const util::file::llm_util_params& params)
{
    auto cparams = llama_context_default_params();

    cpu_params target_cpuparams = params.cpuparams;
    cpu_params target_cpuparams_batch = params.cpuparams_batch;
    postprocess_cpu_params(target_cpuparams, nullptr);
    postprocess_cpu_params(target_cpuparams_batch, &target_cpuparams);

    cparams.n_ctx = params.cparams.n_ctx;
    cparams.n_seq_max = params.cparams.n_parallel;
    cparams.n_batch = params.cparams.n_batch;
    cparams.n_ubatch = params.cparams.n_ubatch;
    cparams.n_threads = target_cpuparams.n_threads;
    cparams.n_threads_batch = target_cpuparams_batch.n_threads;
    cparams.embeddings = params.cparams.embeddings_only;
    cparams.rope_scaling_type = params.cparams.rope_params.rope_scaling_type;
    cparams.rope_freq_base = params.cparams.rope_params.rope_freq_base;
    cparams.rope_freq_scale = params.cparams.rope_params.rope_freq_scale;
    cparams.yarn_ext_factor = params.cparams.yarn_params.yarn_ext_factor;
    cparams.yarn_attn_factor = params.cparams.yarn_params.yarn_attn_factor;
    cparams.yarn_beta_fast = params.cparams.yarn_params.yarn_beta_fast;
    cparams.yarn_beta_slow = params.cparams.yarn_params.yarn_beta_slow;
    cparams.yarn_orig_ctx = params.cparams.yarn_params.yarn_orig_ctx;
    cparams.pooling_type = params.cparams.pooling_type;
    cparams.attention_type = params.cparams.attention_type;
    cparams.flash_attn_type = params.cparams.flash_attn_type;
    cparams.cb_eval = params.cparams.cb_eval;
    cparams.cb_eval_user_data = params.cparams.cb_eval_user_data;
    cparams.offload_kqv = !params.cparams.no_kv_offload;
    cparams.no_perf = params.cparams.no_perf;
    cparams.op_offload = !params.cparams.no_op_offload;
    cparams.swa_full = params.cparams.swa_full;
    cparams.kv_unified = params.cparams.kv_unified;

    cparams.type_k = params.cparams.cache_type_k;
    cparams.type_v = params.cparams.cache_type_v;

    return cparams;
}
