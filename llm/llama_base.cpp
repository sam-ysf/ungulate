#include "llm/llama_base.hpp"
#include "llm/base_utils.hpp"
#include "llm/file_utils.hpp"
#include "llm/sampling.hpp"
#include "llm/vocab_utils.hpp"
#include "log/log.hpp"
#include <array>
#include <format>
#include <ggml.h>
#include <limits>
#include <llama-cpp.h>
#include <llama.h>
#include <mutex>
#include <source_location>
#include <stdexcept>
#include <unordered_map>

namespace {

    llama_model* init_model(const util::file::llm_util_params& default_params)
    {
        LOG_INF(
            "loading model: %s", default_params.mparams.source.gguf.c_str());

        auto params = default_params;
        const auto mparams = util::file::llm_util_model_params_to_llama(params);

        llama_model* model = llama_model_load_from_file(
            params.mparams.source.gguf.c_str(), mparams);
        if (model == nullptr) {
            LOG_ERR(
                "%s: failed to load model '%s', try reducing n-gpu-layers if "
                "you're running out of VRAM",
                std::source_location::current().function_name(),
                params.mparams.source.gguf.c_str());
            return nullptr;
        }

        return model;
    }

    llama_context* init_context(
        const util::file::llm_util_params& params,
        llama_model* model)
    {
        auto n_ctx_native
            = static_cast<std::uint32_t>(llama_model_n_ctx_train(model));

        std::uint32_t n_ctx = params.cparams.n_ctx;
        if (n_ctx == 0) {
            n_ctx = n_ctx_native;
        }

        auto cparams = llm_util_context_params_to_llama(params);
        cparams.n_ctx = std::min<std::uint32_t>(n_ctx, n_ctx_native);

        llama_context* context = llama_init_from_model(model, cparams);
        if (context == nullptr) {
            LOG_ERR(
                "%s: failed to create context with model '%s', try reducing "
                "n-gpu-layers if you're running out of VRAM",
                std::source_location::current().function_name(),
                params.mparams.source.gguf.c_str());
            return nullptr;
        }

        return context;
    }

    util::file::llm_util_sampler_ptr init_sampler(
        const util::file::llm_util_sampling_params& default_params,
        const llama_model* model,
        llama_context* context)
    {
        const llama_vocab* vocab = llama_model_get_vocab(model);

        auto sampling_params = default_params;

        if (sampling_params.ignore_eos
            && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
            LOG_WRN("vocab does not have an EOS token, ignoring ignore-eos");
            sampling_params.ignore_eos = false;
        }

        // initialize once
        sampling_params.logit_bias_eog = std::vector<llama_logit_bias>();
        for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
            if (llama_vocab_is_eog(vocab, i)) {
                LOG_DBG(
                    "%s: added %s logit bias = %f",
                    std::source_location::current().function_name(),
                    util::file::llm_util_token_to_piece(context, i).c_str(),
                    -std::numeric_limits<float>::infinity());
                sampling_params.logit_bias_eog.push_back(
                    {.token = i,
                     .bias = -std::numeric_limits<float>::infinity()});
            }
        }

        if (sampling_params.ignore_eos) {
            // add EOG biases to the active set of logit biases
            sampling_params.logit_bias.insert(
                sampling_params.logit_bias.end(),
                sampling_params.logit_bias_eog.begin(),
                sampling_params.logit_bias_eog.end());
        }

        if (sampling_params.penalty_last_n == -1) {
            LOG_DBG(
                "%s: setting penalty_last_n to ctx_size = %d",
                std::source_location::current().function_name(),
                llama_n_ctx(context));
            sampling_params.penalty_last_n
                = static_cast<std::int32_t>(llama_n_ctx(context));
        }

        if (sampling_params.dry_penalty_last_n == -1) {
            LOG_DBG(
                "%s: setting dry_penalty_last_n to ctx_size = %d",
                std::source_location::current().function_name(),
                llama_n_ctx(context));
            sampling_params.dry_penalty_last_n
                = static_cast<std::int32_t>(llama_n_ctx(context));
        }

        return llm_util_sampler_init(context, sampling_params);
    }

    bool check_configuration_integrity(
        const llama_model* model,
        const llama_context* context)
    {
        if (llama_pooling_type(context) != LLAMA_POOLING_TYPE_RANK) {
            return true;
        }

        const llama_vocab* vocab = llama_model_get_vocab(model);
        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            LOG_WRN(
                "%s: warning: vocab does not have a  BOS token, reranking "
                "will not work",
                std::source_location::current().function_name());
            return false;
        }

        const bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        if (const bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
            !has_sep) {
            // Try to fall back
            if (has_eos) {
                LOG_WRN(
                    "%s: warning: vocab does not have an EOS token, using SEP "
                    "token as fallback",
                    std::source_location::current().function_name());
                return true;
            }

            LOG_WRN(
                "%s: warning: vocab does not have an EOS token or SEP "
                "token, reranking will not work",
                std::source_location::current().function_name());
            return false;
        }

        return true;
    }

    util::file::llm_util_session llm_util_init_from_params(
        const util::file::llm_util_params& params)
    {
        auto get_model_metadata
            = [](llama_model* model, std::int32_t index, const auto& target) {
            std::array<char, 1024> buff = {};
            const std::int32_t size
                = target(model, index, buff.data(), buff.size());
            return size == -1 ? std::optional<std::string>()
                              : std::string(buff.data(), buff.data() + size);
        };

        llama_model_ptr model(init_model(params));
        if (model == nullptr) {
            return util::file::llm_util_session();
        }

        llama_context_ptr context(init_context(params, model.get()));
        if (context == nullptr) {
            return util::file::llm_util_session();
        }

        // Ensure model configuration
        if (!check_configuration_integrity(model.get(), context.get())) {
            return util::file::llm_util_session();
        }

        util::file::llm_util_sampler_ptr sampler(
            init_sampler(params.sampling, model.get(), context.get()));
        if (sampler == nullptr) {
            return util::file::llm_util_session();
        }

        util::file::llm_util_session iparams;

        iparams.model = std::move(model);
        iparams.model_hash = params.mparams.source.gguf_hash;

        iparams.context = std::move(context);
        iparams.sampler = std::move(sampler);

        for (std::int32_t i = 0;; ++i) {
            std::optional<std::string> key = get_model_metadata(
                iparams.model.get(), i, llama_model_meta_key_by_index);
            std::optional<std::string> value = get_model_metadata(
                iparams.model.get(), i, llama_model_meta_val_str_by_index);

            if (!key || !value) {
                break;
            }

            iparams.model_metadata[key.value()] = value.value();
        }

        return iparams;
    }

    void warmup(
        const util::file::llm_util_params& params,
        const llama_model* model,
        llama_context* context)
    {
        const llama_vocab* vocab = llama_model_get_vocab(model);
        LOG_WRN("warming up the model with an empty run - please wait ... ");

        llama_set_warmup(context, true);

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_encode(
                context,
                llama_batch_get_one(
                    tmp.data(), static_cast<std::int32_t>(tmp.size())));
            llama_token decoder_start_token_id
                = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }

        if (llama_model_has_decoder(model)) {
            auto n_tokens = static_cast<std::int32_t>(params.cparams.n_batch);
            llama_decode(
                context,
                llama_batch_get_one(
                    tmp.data(),
                    std::min<std::int32_t>(
                        static_cast<std::int32_t>(tmp.size()), n_tokens)));
        }

        llama_memory_clear(llama_get_memory(context), true);
        llama_synchronize(context);
        llama_perf_context_reset(context);
        llama_set_warmup(context, true);
    }
} // namespace

/* Llama instance counter
 */
std::atomic<int> util::file::LlamaBase::activation_count_(0);

util::file::LlamaBase::~LlamaBase()
{
    teardown();

    int n = --LlamaBase::activation_count_;
    if (n == 0) {
        llama_backend_free();
    }
}

namespace {
    /*! Helper
     */
    inline std::uint32_t get_model_context_size(
        const llama_model* model,
        std::uint32_t n_ctx)
    {
        return std::min(
            n_ctx, static_cast<std::uint32_t>(llama_model_n_ctx_train(model)));
    }

    /*! Helper
     */
    inline std::int32_t get_model_embeddings_vector_size(
        const llama_model* model)
    {
        return llama_model_n_embd(model);
    }
} // namespace

util::file::LlamaBase::LlamaBase(const llm_util_params& params)
    : util_params_(params)
{
    if (!std::filesystem::exists(params.mparams.source.gguf)) {
        throw std::invalid_argument(
            std::format("model {} not found", params.mparams.source.gguf));
    }

    if (!maybe_initialize_ggml()) {
        throw std::runtime_error("llama backend initialization error");
    }

    // If the number of prompts that would be encoded is known in advance,
    // it's more efficient to specify the parallel argument accordingly. for
    // convenience, if not specified, we fallback to unified KV cache in
    // order to support any number of prompts
    if (util_params_.cparams.n_parallel == 1) {
        LOG_INF("n_parallel == 1 -> unified KV cache is enabled");
        util_params_.cparams.kv_unified = true;
    }

    // clamp
    if (util_params_.cparams.n_batch > util_params_.cparams.n_ctx) {
        LOG_WRN(
            "%s: setting logical batch size to context window size %d",
            std::source_location::current().function_name(),
            util_params_.cparams.n_ctx);
        util_params_.cparams.n_batch = std::min(
            util_params_.cparams.n_ctx, util_params_.cparams.n_batch);
    }

    // clamp
    if (util_params_.cparams.n_ubatch > util_params_.cparams.n_ctx) {
        LOG_WRN(
            "%s: setting physical ubatch size to context window size %d",
            std::source_location::current().function_name(),
            util_params_.cparams.n_ctx);
        util_params_.cparams.n_ubatch = std::min(
            util_params_.cparams.n_ctx, util_params_.cparams.n_ubatch);
    }
}

void util::file::LlamaBase::clear_cache()
{
    std::scoped_lock<std::mutex> lock(lock_);

    const llama_context* context = llama_session_.context.get();
    llama_memory_seq_rm(llama_get_memory(context), -1, -1, -1);
}

int util::file::LlamaBase::get_index() const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return gguf_index_;
}

util::file::llm_util_params util::file::LlamaBase::get_params() const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return util_params_;
}

llama_context* util::file::LlamaBase::get_context() const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return llama_session_.context.get();
}

llama_model* util::file::LlamaBase::get_model() const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return llama_session_.model.get();
}

util::file::llm_util_sampler* util::file::LlamaBase::get_sampler() const
{
    std::scoped_lock<std::mutex> lock(lock_);

    return llama_session_.sampler.get();
}

void util::file::LlamaBase::initialize(
    enum llama_pooling_type pooling_type,
    bool embeddings_only)
{
    std::scoped_lock<std::mutex> lock(lock_);

    auto params = util_params_;
    params.cparams.pooling_type = pooling_type;
    params.cparams.embeddings_only = embeddings_only;

    // load the model
    llama_session_ = llm_util_init_from_params(params);
    if (!llama_session_.is_ok()) {
        throw std::runtime_error(
            std::format(
                "Error initializing model {}", params.mparams.source.gguf));
    }

    // set final model parameters
    const llama_model* model = llama_session_.model.get();

    params.cparams.n_ctx = get_model_context_size(model, params.cparams.n_ctx);
    params.cparams.n_embeddings = get_model_embeddings_vector_size(model);

    // Maybe do a warmup run
    if (params.warmup) {
        auto* context = llama_session_.context.get();
        warmup(params, model, context);
    }

    util_params_ = params;
}

void util::file::LlamaBase::teardown()
{
    std::scoped_lock<std::mutex> lock(lock_);

    auto& sampler = llama_session_.sampler;
    auto& context = llama_session_.context;
    auto& model = llama_session_.model;

    sampler.reset();
    context.reset();
    model.reset();
}

bool util::file::LlamaBase::maybe_initialize_ggml()
{
    gguf_index_ = LlamaBase::activation_count_++;
    if (gguf_index_ > 0) {
        LOG_INF("llama backend already initialized from a previous invocation");
        return true;
    }

    // load dynamic backends
    ggml_backend_load_all();
    // Ensure ggml correclty loaded
    if (ggml_backend_reg_count() == 0 || ggml_backend_dev_count() == 0) {
        return false;
    }

    ggml_time_init();

    llama_log_set(
        [](ggml_log_level /* level */,
           const char* text,
           void* /* user_data */) {
        util::log::log_print(
            util::log::get_log_main(),
            util::log::log_level::kLogLevelDebug,
            "%s",
            text);
    },
        nullptr);

    LOG_INF("initializing Llama backend...");
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    return true;
}
