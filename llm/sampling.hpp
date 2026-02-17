#pragma once

#include <llama-cpp.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace util::file {

    // Opaque type
    // Sampler implementation
    //
    // 1. sets logits
    // 2. applies the configured sampler chain
    // 3. applies the grammar constraints before sampling
    class llm_util_sampler;

    enum class llm_util_grammar_trigger_type : std::uint8_t {
        kTriggerTypeToken,
        kTriggerTypeWord,
        kTriggerTypePattern,
        kTriggerTypePatternFull,
    };

    enum class llm_util_sampler_type : std::uint8_t {
        kSamplerTypeNone = 0,
        kSamplerTypeDry = 1,
        kSamplerTypeTopK = 2,
        kSamplerTypeTopP = 3,
        kSamplerTypeMinP = 4,
        // kSamplerTypeTfsZ = 5,
        kSamplerTypeTypicalP = 6,
        kSamplerTypeTemperature = 7,
        kSamplerTypeXtc = 8,
        kSamplerTypeInfill = 9,
        kSamplerTypePenalties = 10,
        kSamplerTypeTopNSigma = 11,
    };

    struct llm_util_grammar_trigger {
        llm_util_grammar_trigger_type type;
        std::string value;
        llama_token token = LLAMA_TOKEN_NULL;
    };

    struct llm_util_mirostat {};

    struct llm_util_grammar {
        std::string grammar;
        // optional BNF-like grammar to constrain sampling
        bool grammar_lazy = false;
        std::vector<llm_util_grammar_trigger>
            grammar_triggers; // optional triggers (for lazy grammars)
    };

    // sampling parameters
    struct llm_util_sampling_params {
        std::uint32_t seed
            = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

        std::int32_t n_prev = 64; // number of previous tokens to remember
        std::int32_t n_probs = 0; // if greater than 0, output the probabilities
                                  // of top n_probs tokens.
        std::size_t min_keep = 0; // 0 = disabled, otherwise samplers should
                                  // return at least min_keep tokens
        std::int32_t top_k = 40; // <= 0 to use vocab size
        float top_p = 0.95F; // 1.0 = disabled
        float min_p = 0.05F; // 0.0 = disabled
        float xtc_probability = 0.00F; // 0.0 = disabled
        float xtc_threshold = 0.10F; // > 0.5 disables XTC
        float typ_p = 1.00F; // typical_p, 1.0 = disabled
        float temp = 0.00F; // <= 0.0 to sample greedily, 0.0 to not output
                            // probabilities
        float dynatemp_range = 0.00F; // 0.0 = disabled
        float dynatemp_exponent
            = 1.00F; // controls how entropy maps to temperature
                     // in dynamic temperature sampler
        std::int32_t penalty_last_n = 64; // last n tokens to penalize (0 =
                                          // disable penalty, -1 = context size)
        float penalty_repeat = 1.00F; // 1.0 = disabled
        float penalty_freq = 0.00F; // 0.0 = disabled
        float penalty_present = 0.00F; // 0.0 = disabled
        float dry_multiplier = 0.8F; // 0.0 = disabled;      DRY repetition
                                     // penalty for tokens extending repetition:
        float dry_base
            = 1.75F; // 0.0 = disabled;      multiplier * base ^ (length
                     // of sequence before token - allowed length)
        std::int32_t dry_allowed_length
            = 2; // tokens extending repetitions beyond this receive penalty
        std::int32_t dry_penalty_last_n
            = -1; // how many tokens to scan for repetitions
                  // (0 = disable penalty, -1 = context size)
        std::int32_t mirostat
            = 0; // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        float mirostat_tau = 5.00F; // target entropy
        float mirostat_eta = 0.10F; // learning rate
        float top_n_sigma = -1.00F; // -1.0 = disabled

        bool ignore_eos = false;
        bool no_perf = false; // disable performance metrics
        bool timing_per_token = false;

        std::vector<std::string> dry_sequence_breakers
            = {"\n", ":", "\"", "*"}; // default sequence breakers for DRY

        std::vector<enum llm_util_sampler_type> samplers = {
            llm_util_sampler_type::kSamplerTypePenalties,
            llm_util_sampler_type::kSamplerTypeDry,
            llm_util_sampler_type::kSamplerTypeTopNSigma,
            llm_util_sampler_type::kSamplerTypeTopK,
            llm_util_sampler_type::kSamplerTypeTypicalP,
            llm_util_sampler_type::kSamplerTypeTopP,
            llm_util_sampler_type::kSamplerTypeMinP,
            llm_util_sampler_type::kSamplerTypeXtc,
            llm_util_sampler_type::kSamplerTypeTemperature,
        };

        std::set<llama_token> preserved_tokens;

        llm_util_grammar grammar;

        std::vector<llama_logit_bias> logit_bias; // logit biases to apply
        std::vector<llama_logit_bias>
            logit_bias_eog; // pre-calculated logit biases for EOG tokens
    };

    // Public API
    // deletes callback used in smart pointer
    void inferenece_sampler_free(llm_util_sampler* sampler);

    // Public API
    // if accept_grammar is true, the token is accepted both by the sampling
    // chain and the grammar
    void llm_util_sampler_accept(llm_util_sampler* sampler, llama_token token);

    // Public API
    // samples token at the specified index
    llama_token llm_util_sampler_sample(
        llm_util_sampler* sampler,
        int token_index);

    llama_sampler* llm_util_model_get_grammar(
        const llama_model* model,
        const llm_util_sampling_params& params);

    //! @return Queried sampler type abbv.
    char llm_util_sampler_type_to_abbv(llm_util_sampler_type cnstr);

    //! @return Queried sampler type description
    std::string llm_util_sampler_type_to_str(llm_util_sampler_type cnstr);

    //! @return Queried sampler types
    std::vector<llm_util_sampler_type> llm_util_sampler_types_from_names(
        const std::vector<std::string>& names,
        bool allow_alt_names);

    //! @return Queried sampler types
    std::vector<llm_util_sampler_type> llm_util_sampler_types_from_abbvs(
        const std::string& chars);

    struct llm_util_sampler_deleter {
        void operator()(llm_util_sampler* sampler) const
        {
            inferenece_sampler_free(sampler);
        }
    };

    using llm_util_sampler_ptr
        = std::unique_ptr<llm_util_sampler, llm_util_sampler_deleter>;

    // Public API
    // factory method
    llm_util_sampler_ptr llm_util_sampler_init(
        llama_context* context,
        const llm_util_sampling_params& params);

} // namespace util::file
