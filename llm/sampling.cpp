#include "llm/sampling.hpp"
#include "llm/string_utils.hpp"
#include "log/log.hpp"
#include <cmath>
#include <source_location>
#include <stdexcept>
#include <unordered_map>

namespace util::file {

    class llm_util_sampler {
    public:
        // Dtor.
        ~llm_util_sampler();

        // Ctor.
        llm_util_sampler(
            llama_context* context,
            const llm_util_sampling_params& params);

        llm_util_sampler(const llm_util_sampler&) = delete;
        llm_util_sampler& operator=(const llm_util_sampler&) = delete;

        // Handles new token
        void accept(int token);

        // Samples next token
        llama_token sample();

        void set_logits(int token_index);
    private:
        // Helper
        bool chain_add(
            const llama_model* model,
            const llm_util_sampling_params& params);

        // Helper
        bool chain_add_mirostat(
            const llama_model* model,
            const llm_util_sampling_params& params);

        llama_context* context_ = nullptr;
        llama_sampler* chain_ = nullptr;
        llama_sampler* grammar_ = nullptr;

        using llama_token_data_buffer = std::vector<llama_token_data>;

        llama_token_data_array data_array_ = {};
        llama_token_data_buffer logits_;
    };
} // namespace util::file

util::file::llm_util_sampler::llm_util_sampler(
    llama_context* context,
    const llm_util_sampling_params& params)
    : context_(context)
{
    const llama_model* model = llama_get_model(context_);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const llama_sampler_chain_params sampler_params
        = llama_sampler_chain_default_params();

    grammar_ = llm_util_model_get_grammar(model, params);
    chain_ = llama_sampler_chain_init(sampler_params);
    llama_sampler* logit = llama_sampler_init_logit_bias(
        llama_vocab_n_tokens(vocab),
        static_cast<std::int32_t>(params.logit_bias.size()),
        params.logit_bias.data());

    llama_sampler_chain_add(chain_, logit);
    switch (params.mirostat) {
        case 0:
        {
            chain_add(model, params);
            break;
        }

        case 1:
        case 2:
        {
            chain_add_mirostat(model, params);
            break;
        }

        default:
        {
            LOG_ERR(
                "%s : unknown mirostat version",
                std::source_location::current().function_name());
            throw std::invalid_argument("unknown mirostat version");
        }
    }
}

util::file::llm_util_sampler::~llm_util_sampler()
{
    if (grammar_) {
        llama_sampler_free(grammar_);
    }

    if (chain_) {
        llama_sampler_free(chain_);
    }
}

void util::file::llm_util_sampler::accept(int token)
{
    llama_sampler_accept(grammar_, token);
    llama_sampler_accept(chain_, token);
}

llama_token util::file::llm_util_sampler::sample()
{
    llama_sampler_apply(grammar_, &data_array_);
    llama_sampler_apply(chain_, &data_array_);

    if (data_array_.selected == -1) {
        throw std::underflow_error(
            "no selected token during sampling - check "
            "your sampling configuration");
    }

    return data_array_.data[data_array_.selected].id;
}

void util::file::llm_util_sampler::set_logits(int token_index)
{
    const float* logits = llama_get_logits_ith(context_, token_index);

    const llama_model* model = llama_get_model(context_);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    auto n_vocab = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));

    logits_ = llama_token_data_buffer();
    logits_.reserve(n_vocab);

    for (std::size_t i = 0; i < n_vocab; ++i) {
        auto token_id = static_cast<llama_token>(i);
        logits_.push_back(llama_token_data{token_id, logits[token_id], 0.0F});
    }

    data_array_
        = {.data = logits_.data(),
           .size = logits_.size(),
           .selected = -1,
           .sorted = false};
}

bool util::file::llm_util_sampler::chain_add(
    const llama_model* model,
    const llm_util_sampling_params& params)
{
    if (params.mirostat != 0) {
        return false;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    using enum llm_util_sampler_type;

    for (const auto& sampler: params.samplers) {
        switch (sampler) {
            case kSamplerTypeNone:
            {
                break;
            }

            case kSamplerTypeDry:
            {
                std::vector<const char*> c_breakers;
                c_breakers.reserve(params.dry_sequence_breakers.size());
                for (const auto& str: params.dry_sequence_breakers) {
                    c_breakers.push_back(str.c_str());
                }

                llama_sampler_chain_add(
                    chain_,
                    llama_sampler_init_dry(
                        vocab,
                        llama_model_n_ctx_train(model),
                        params.dry_multiplier,
                        params.dry_base,
                        params.dry_allowed_length,
                        params.dry_penalty_last_n,
                        c_breakers.data(),
                        c_breakers.size()));
                break;
            }

            case kSamplerTypeTopK:
            {
                llama_sampler_chain_add(
                    chain_, llama_sampler_init_top_k(params.top_k));
                break;
            }

            case kSamplerTypeTopP:
            {
                llama_sampler_chain_add(
                    chain_,
                    llama_sampler_init_top_p(params.top_p, params.min_keep));
                break;
            }

            case kSamplerTypeMinP:
            {
                llama_sampler_chain_add(
                    chain_,
                    llama_sampler_init_min_p(params.min_p, params.min_keep));
                break;
            }

            case kSamplerTypeTypicalP:
            {
                llama_sampler_chain_add(
                    chain_,
                    llama_sampler_init_typical(params.typ_p, params.min_keep));
                break;
            }

            case kSamplerTypeTemperature:
            {
                llama_sampler_chain_add(
                    chain_,
                    llama_sampler_init_temp_ext(
                        params.temp,
                        params.dynatemp_range,
                        params.dynatemp_exponent));
                break;
            }

            case kSamplerTypeXtc:
            {
                llama_sampler_chain_add(
                    chain_,
                    llama_sampler_init_xtc(
                        params.xtc_probability,
                        params.xtc_threshold,
                        params.min_keep,
                        params.seed));
                break;
            }

            case kSamplerTypeInfill:
            {
                llama_sampler_chain_add(
                    chain_, llama_sampler_init_infill(vocab));
                break;
            }

            case kSamplerTypePenalties:
            {
                llama_sampler_chain_add(
                    chain_,
                    llama_sampler_init_penalties(
                        params.penalty_last_n,
                        params.penalty_repeat,
                        params.penalty_freq,
                        params.penalty_present));
                break;
            }

            case kSamplerTypeTopNSigma:
            {
                llama_sampler_chain_add(
                    chain_, llama_sampler_init_top_n_sigma(params.top_n_sigma));
                break;
            }
        }
    }

    llama_sampler_chain_add(chain_, llama_sampler_init_dist(params.seed));
    return true;
}

bool util::file::llm_util_sampler::chain_add_mirostat(
    const llama_model* model,
    const llm_util_sampling_params& params)
{
    const llama_vocab* vocab = llama_model_get_vocab(model);

    if (params.mirostat == 1) {
        llama_sampler_chain_add(chain_, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(
            chain_,
            llama_sampler_init_mirostat(
                llama_vocab_n_tokens(vocab),
                params.seed,
                params.mirostat_tau,
                params.mirostat_eta,
                100));
        return true;
    }

    if (params.mirostat == 2) {
        llama_sampler_chain_add(chain_, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(
            chain_,
            llama_sampler_init_mirostat_v2(
                params.seed, params.mirostat_tau, params.mirostat_eta));
        return true;
    }

    // unknown mirostat version
    return false;
}

util::file::llm_util_sampler_ptr util::file::llm_util_sampler_init(
    llama_context* context,
    const llm_util_sampling_params& params)
{
    return std::unique_ptr<llm_util_sampler, llm_util_sampler_deleter>(
        new llm_util_sampler(context, params));
}

void util::file::inferenece_sampler_free(llm_util_sampler* sampler)
{
    delete sampler;
}

void util::file::llm_util_sampler_accept(
    llm_util_sampler* sampler,
    llama_token token)
{
    sampler->accept(token);
}

llama_token util::file::llm_util_sampler_sample(
    llm_util_sampler* sampler,
    int token)
{
    sampler->set_logits(token);
    return sampler->sample();
}

llama_sampler* util::file::llm_util_model_get_grammar(
    const llama_model* model,
    const llm_util_sampling_params& params)
{
    const llama_vocab* vocab = llama_model_get_vocab(model);

    if (params.grammar.grammar.starts_with("%llguidance")) {
#ifdef LLAMA_USE_LLGUIDANCE
        return llama_sampler_init_llg(
            vocab, "lark", params.grammar.grammar.c_str());
#else
        throw std::runtime_error(
            "llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif
    }

    if (!params.grammar.grammar_lazy) {
        return llama_sampler_init_grammar(
            vocab, params.grammar.grammar.c_str(), "root");
    }

    std::vector<std::string> patterns_anywhere;
    std::vector<std::string> trigger_patterns;
    std::vector<llama_token> trigger_tokens;

    using enum llm_util_grammar_trigger_type;

    for (const auto& trigger: params.grammar.grammar_triggers) {
        switch (trigger.type) {
            case kTriggerTypeWord:
            {
                const auto& word = trigger.value;
                patterns_anywhere.push_back(string_escape(word));
                break;
            }

            case kTriggerTypePattern:
            {
                patterns_anywhere.push_back(trigger.value);
                break;
            }

            case kTriggerTypePatternFull:
            {
                trigger_patterns.push_back(trigger.value);
                break;
            }

            case kTriggerTypeToken:
            {
                const auto token = trigger.token;
                trigger_tokens.push_back(token);
                break;
            }
        }
    }

    if (!patterns_anywhere.empty()) {
        trigger_patterns.push_back(
            "^[\\s\\S]*?(" + string_join(patterns_anywhere, "|")
            + ")[\\s\\S]*");
    }

    std::vector<const char*> trigger_patterns_c;
    trigger_patterns_c.reserve(trigger_patterns.size());
    for (const auto& regex: trigger_patterns) {
        trigger_patterns_c.push_back(regex.c_str());
    }

    llama_sampler* grammar = llama_sampler_init_grammar_lazy_patterns(
        vocab,
        params.grammar.grammar.c_str(),
        "root",
        trigger_patterns_c.data(),
        trigger_patterns_c.size(),
        trigger_tokens.data(),
        trigger_tokens.size());
    return grammar;
}

char util::file::llm_util_sampler_type_to_abbv(llm_util_sampler_type cnstr)
{
    using enum llm_util_sampler_type;

    switch (cnstr) {
        case kSamplerTypeDry:
            return 'd';
        case kSamplerTypeTopK:
            return 'k';
        case kSamplerTypeTypicalP:
            return 'y';
        case kSamplerTypeTopP:
            return 'p';
        case kSamplerTypeTopNSigma:
            return 's';
        case kSamplerTypeMinP:
            return 'm';
        case kSamplerTypeTemperature:
            return 't';
        case kSamplerTypeXtc:
            return 'x';
        case kSamplerTypeInfill:
            return 'i';
        case kSamplerTypePenalties:
            return 'e';
        default:
            return '?';
    }
}

std::string util::file::llm_util_sampler_type_to_str(
    llm_util_sampler_type cnstr)
{
    using enum llm_util_sampler_type;

    switch (cnstr) {
        case kSamplerTypeDry:
            return "dry";
        case kSamplerTypeTopK:
            return "top_k";
        case kSamplerTypeTopP:
            return "top_p";
        case kSamplerTypeMinP:
            return "min_p";
        case kSamplerTypeTypicalP:
            return "typ_p";
        case kSamplerTypeTemperature:
            return "temperature";
        case kSamplerTypeXtc:
            return "xtc";
        case kSamplerTypeInfill:
            return "infill";
        case kSamplerTypePenalties:
            return "penalties";
        case kSamplerTypeTopNSigma:
            return "top_n_sigma";
        default:
            return "";
    }
}

std::vector<util::file::llm_util_sampler_type> util::file::
    llm_util_sampler_types_from_names(
        const std::vector<std::string>& names,
        bool allow_alt_names)
{
    using enum llm_util_sampler_type;

    std::unordered_map<std::string, llm_util_sampler_type>
        sampler_canonical_name_map{
            {"dry", kSamplerTypeDry},
            {"top_k", kSamplerTypeTopK},
            {"top_p", kSamplerTypeTopP},
            {"min_p", kSamplerTypeMinP},
            {"typ_p", kSamplerTypeTypicalP},
            {"temperature", kSamplerTypeTemperature},
            {"xtc", kSamplerTypeXtc},
            {"infill", kSamplerTypeInfill},
            {"penalties", kSamplerTypePenalties},
            {"top_n_sigma", kSamplerTypeTopNSigma},
        };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llm_util_sampler_type> sampler_alt_name_map{
        {"top-k", kSamplerTypeTopK},
        {"top-p", kSamplerTypeTopP},
        {"nucleus", kSamplerTypeTopP},
        {"min-p", kSamplerTypeMinP},
        {"typical-p", kSamplerTypeTypicalP},
        {"typical", kSamplerTypeTypicalP},
        {"typ-p", kSamplerTypeTypicalP},
        {"typ", kSamplerTypeTypicalP},
        {"temp", kSamplerTypeTemperature},
        {"top-n-sigma", kSamplerTypeTopNSigma},
    };

    std::vector<llm_util_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto& name: names) {
        auto sampler = sampler_canonical_name_map.find(name);
        if (sampler != sampler_canonical_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        if (allow_alt_names) {
            sampler = sampler_alt_name_map.find(name);
            if (sampler != sampler_alt_name_map.end()) {
                samplers.push_back(sampler->second);
                continue;
            }
        }
        LOG_WRN("unable to match sampler by name '%s'", name.c_str());
    }

    return samplers;
}

std::vector<util::file::llm_util_sampler_type> util::file::
    llm_util_sampler_types_from_abbvs(const std::string& chars)
{
    using enum llm_util_sampler_type;

    std::unordered_map<char, llm_util_sampler_type> sampler_name_map = {
        {llm_util_sampler_type_to_abbv(kSamplerTypeDry), kSamplerTypeDry},
        {llm_util_sampler_type_to_abbv(kSamplerTypeTopK), kSamplerTypeTopK},
        {llm_util_sampler_type_to_abbv(kSamplerTypeTopP), kSamplerTypeTopP},
        {llm_util_sampler_type_to_abbv(kSamplerTypeMinP), kSamplerTypeMinP},
        {llm_util_sampler_type_to_abbv(kSamplerTypeTypicalP),
         kSamplerTypeTypicalP},
        {llm_util_sampler_type_to_abbv(kSamplerTypeTemperature),
         kSamplerTypeTemperature},
        {llm_util_sampler_type_to_abbv(kSamplerTypeXtc), kSamplerTypeXtc},
        {llm_util_sampler_type_to_abbv(kSamplerTypeInfill), kSamplerTypeInfill},
        {llm_util_sampler_type_to_abbv(kSamplerTypePenalties),
         kSamplerTypePenalties},
        {llm_util_sampler_type_to_abbv(kSamplerTypeTopNSigma),
         kSamplerTypeTopNSigma},
    };

    std::vector<llm_util_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto& c: chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("unable to match sampler by char '%c'", c);
        }
    }

    return samplers;
}
