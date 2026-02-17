#include "llm/embedding_extractor.hpp"
#include "file/embeddings.hpp"
#include "llm/base_utils.hpp"
#include "llm/batch_utils.hpp"
#include "llm/cpu_utils.hpp"
#include "llm/llama_base.hpp"
#include "llm/vocab_utils.hpp"
#include "log/log.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <llama-cpp.h>
#include <llama.h>
#include <mutex>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    /* Helper
     */
    void normalize_embeddings(
        const float* inp,
        float* out,
        std::uint32_t n,
        util::file::llm_util_embedding_normalize_type norm_type)
    {
        double sum = 0.0;

        switch (norm_type) {
            case util::file::llm_util_embedding_normalize_type::
                kEmbeddingNormalizeNone:
            {
                sum = 1.0;
                break;
            }

            case util::file::llm_util_embedding_normalize_type::
                kEmbeddingNormalizeMax:
            {
                for (std::uint32_t i = 0; i < n; i++) {
                    sum = std::max<double>(sum, std::abs(inp[i]));
                }
                sum /= std::pow(2, 15); // make an int16 range
                break;
            }

            case util::file::llm_util_embedding_normalize_type::
                kEmbeddingNormalizeTaxicab:
            {
                for (std::uint32_t i = 0; i < n; i++) {
                    sum += std::abs(inp[i]);
                }
                break;
            }

            case util::file::llm_util_embedding_normalize_type::
                kEmbeddingNormalizeEuclidian:
            {
                for (std::uint32_t i = 0; i < n; i++) {
                    sum += std::pow(inp[i], 2.0);
                }
                sum = std::sqrt(sum);
                break;
            }

            case util::file::llm_util_embedding_normalize_type::
                kEmbeddingNormalizePNorm:
            {
                for (std::uint32_t i = 0; i < n; i++) {
                    sum += std::pow(std::abs(inp[i]), 3.0);
                }
                sum = std::pow(sum, 1 / 3.0);
                break;
            }
        }

        const double norm = sum > 0.0 ? (1.0 / sum) : 0.0F;

        for (std::uint32_t i = 0; i < n; i++) {
            out[i] = static_cast<float>(inp[i] * norm);
        }
    }
} // namespace

util::file::EmbeddingExtractor::EmbeddingExtractor(
    const std::shared_ptr<LlamaBase>& llama,
    EmbeddingConfig config)
    : llama_(llama)
    , config_(std::move(config))
{}

std::uint32_t util::file::EmbeddingExtractor::calc_n_embd_count(
    std::uint32_t n_prompts,
    const std::vector<std::vector<std::int32_t>>& tokens) const
{
    std::uint32_t n_embd_count = n_prompts;

    if (const enum llama_pooling_type pooling_type
        = llama_pooling_type(llama_->get_context());
        pooling_type == LLAMA_POOLING_TYPE_NONE) {
        n_embd_count = 0;
        for (std::size_t i = 0; i < n_prompts; i++) {
            n_embd_count += static_cast<std::uint32_t>(tokens[i].size());
        }
    }

    else if (
        pooling_type == LLAMA_POOLING_TYPE_RANK
        && llama_model_n_cls_out(llama_->get_model()) != n_embd_count) {
        throw std::runtime_error(
            "Number of classifier tokens does not "
            "match embedding vector size");
    }

    return n_embd_count;
}

std::vector<util::file::Embeddings> util::file::EmbeddingExtractor::
    calc_embeddings(
        std::uint32_t n_prompts,
        const std::vector<std::vector<std::int32_t>>& tokens) const
{
    // count number of embeddings
    std::uint32_t n_embd_count = calc_n_embd_count(n_prompts, tokens);

    const enum llama_pooling_type pooling_type
        = llama_pooling_type(llama_->get_context());

    std::vector<std::vector<float>> vec_embeddings = decode_embeddings(tokens);

    std::vector<util::file::Embeddings> embeddings;
    for (std::uint32_t i = 0; i < n_embd_count; ++i) {
        Embeddings value;
        value.embeddings = std::move(vec_embeddings[i]);
        value.model = config_.gguf_hash;

        if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
            const char* label = llama_model_cls_label(llama_->get_model(), i);
            if (label) {
                value.classifier_label = label;
            }
        }

        embeddings.push_back(std::move(value));
    }

    return embeddings;
}

std::vector<util::file::Embeddings> util::file::EmbeddingExtractor::extract(
    const std::vector<std::string>& prompts) const
{
    std::scoped_lock<std::mutex> lock(lock_);

    if (prompts.empty()) {
        return std::vector<util::file::Embeddings>();
    }

    llama_perf_context_print(llama_->get_context());

    if (llama_->get_model() == nullptr) {
        LOG_ERR(
            "%s: unable to load model",
            std::source_location::current().function_name());
        return std::vector<util::file::Embeddings>();
    }

    llm_util_params params = llama_->get_params();
    std::string system_info
        = get_system_info(params.cpuparams, params.cpuparams_batch);
    LOG_DBG("%s", system_info.c_str());

    // maybe clear the cache before the next run
    if (config_.clear_cache) {
        llama_->clear_cache();
    }

    // generate output
    return calc_embeddings(
        static_cast<std::uint32_t>(prompts.size()), tokenize_prompts(prompts));
}

void util::file::EmbeddingExtractor::initialize()
{
    std::scoped_lock<std::mutex> lock(lock_);

    llama_->teardown();
    llama_->initialize(default_pooling_type_, config_.embeddings_only);
}

void util::file::EmbeddingExtractor::teardown()
{
    std::scoped_lock<std::mutex> lock(lock_);

    llama_->teardown();
}

bool util::file::EmbeddingExtractor::is_same(const LlamaBase* ptr) const
{
    return ptr == llama_.get();
}

bool util::file::EmbeddingExtractor::is_not_same(const LlamaBase* ptr) const
{
    return ptr != llama_.get();
}

bool util::file::EmbeddingExtractor::has_same_base_object(
    const ExtractorBase& rhs) const
{
    return rhs.is_same(llama_.get());
}

util::file::EmbeddingConfig util::file::EmbeddingExtractor::get_config() const
{
    return config_;
}

std::string util::file::EmbeddingExtractor::to_identifier() const
{
    return config_.gguf_hash;
}

util::file::llm_util_params util::file::EmbeddingExtractor::get_llama_params()
    const
{
    return llama_->get_params();
}

namespace {
    /* Helper
     */
    inline void batch_add_seq(
        llama_batch& batch /* out */,
        const std::vector<std::int32_t>& tokens,
        std::uint32_t seq_id)
    {
        std::size_t size = tokens.size();
        for (std::size_t i = 0; i < size; ++i) {
            auto pos = static_cast<std::int32_t>(i);

            std::vector<llama_seq_id> seq_ids
                = {static_cast<llama_seq_id>(seq_id)};
            util::file::llm_util_batch_add(
                batch, tokens[i], pos, seq_ids, true);
        }
    }
} // namespace

/*! @brief Helper
 */
std::vector<std::vector<float>> util::file::EmbeddingExtractor::
    decode_embeddings(
        const std::vector<std::vector<llama_token>>& tokens_set) const
{
    const enum llama_pooling_type pooling_type
        = llama_pooling_type(llama_->get_context());

    auto n_embd
        = static_cast<std::uint32_t>(llama_model_n_embd(llama_->get_model()));

    // allocate output
    std::size_t n_embd_count = tokens_set.size();

    const std::vector<float> embeddings_buff
        = decode_embeddings_batches(tokens_set);

    std::vector<std::vector<float>> embeddings_result;

    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (std::uint32_t j = 0; j < n_embd_count; ++j) {
            embeddings_result.emplace_back();
            auto& ref = embeddings_result.back();

            for (std::uint32_t i = 0; i < n_embd; ++i) {
                ref.push_back(embeddings_buff[(n_embd * j) + i]);
            }
        }

    } else if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
        for (std::uint32_t j = 0; j < n_embd_count; ++j) {
            embeddings_result.emplace_back();
            auto& ref = embeddings_result.back();

            const uint32_t n_cls_out
                = llama_model_n_cls_out(llama_->get_model());
            for (std::uint32_t i = 0; i < n_cls_out; ++i) {
                ref.push_back(embeddings_buff[(n_embd * j) + i]);
            }
        }

    } else {
        std::uint32_t n_embd_end = n_embd;
        for (std::uint32_t j = 0; j < n_embd_count; ++j) {
            embeddings_result.emplace_back();
            auto& ref = embeddings_result.back();

            for (std::uint32_t i = 0; i < n_embd_end; ++i) {
                ref.push_back(embeddings_buff[(n_embd * j) + i]);
            }
        }
    }

    return embeddings_result;
}

/*! @brief Helper
 */
std::vector<float> util::file::EmbeddingExtractor::decode_embeddings_batches(
    const std::vector<std::vector<llama_token>>& tokens) const
{
    const enum llama_pooling_type pooling_type
        = llama_pooling_type(llama_->get_context());

    const llm_util_params& params = llama_->get_params();

    const auto n_embd
        = static_cast<std::uint32_t>(llama_model_n_embd(llama_->get_model()));

    // allocate output
    std::vector<float> embeddings_buff(tokens.size() * n_embd, 0);

    // number of embeddings already stored
    std::uint32_t e = 0;
    // number of prompts in current batch
    std::uint32_t n_seq = 0;

    const auto n_batch = static_cast<std::int32_t>(params.cparams.n_batch);

    float* batch_ptr = embeddings_buff.data();
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    for (const std::vector<llama_token>& entry: tokens) {
        // clamp current batch to n_batch tokens
        // and encode if at capacity
        if (std::size_t n
            = static_cast<std::size_t>(batch.n_tokens) + entry.size();
            n > params.cparams.n_batch) {
            float* const ptr = batch_ptr + (n_embd * e);
            decode_embeddings_batch(
                batch, ptr, n_seq, n_embd, params.embeddings_normalize);

            std::uint32_t e_increment = n_seq;
            if (pooling_type == LLAMA_POOLING_TYPE_NONE)
                e_increment = static_cast<std::uint32_t>(batch.n_tokens);
            e += e_increment;

            n_seq = 0;
            llm_util_batch_clear(batch);
        }

        // add to batch
        batch_add_seq(batch, entry, n_seq);
        ++n_seq;
    }

    // final batch
    float* const ptr = batch_ptr + (e * n_embd);
    decode_embeddings_batch(
        batch, ptr, n_seq, n_embd, params.embeddings_normalize);
    llama_batch_free(batch);

    return embeddings_buff;
}

void util::file::EmbeddingExtractor::decode_embeddings_batch(
    const llama_batch& batch,
    float* const output,
    std::uint32_t n_seq,
    std::uint32_t n_embd,
    util::file::llm_util_embedding_normalize_type embd_norm) const
{
    llama_context* const context = llama_->get_context();

    // run model
    LOG_DBG(
        "%s: n_tokens = %d, n_seq = %d",
        std::source_location::current().function_name(),
        batch.n_tokens,
        n_seq);
    if (llama_decode(context, batch) < 0) {
        LOG_ERR(
            "%s : error decoding batch",
            std::source_location::current().function_name());
        return;
    }

    const auto size = static_cast<std::uint32_t>(batch.n_tokens);
    for (std::uint32_t i = 0; i < size; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float* embd = nullptr;
        std::uint32_t embd_pos = 0;

        if (enum llama_pooling_type pooling_type = llama_pooling_type(context);
            pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(
                context, static_cast<std::int32_t>(i));
            embd_pos = i;
            if (embd == nullptr) {
                LOG_ERR(
                    "%s: failed to get token embeddings",
                    std::source_location::current().function_name());
            }

        } else {
            // try to get sequence embeddings - supported only when pooling_type
            // is not NONE
            embd = llama_get_embeddings_seq(context, batch.seq_id[i][0]);
            embd_pos = static_cast<std::uint32_t>(batch.seq_id[i][0]);
            if (embd == nullptr) {
                LOG_ERR(
                    "%s: failed to get sequence embeddings",
                    std::source_location::current().function_name());
            }
        }

        if (embd) {
            float* const ptr = output + (n_embd * embd_pos);
            normalize_embeddings(embd, ptr, n_embd, embd_norm);
        }
    }
}

/*! @brief Helper
 */
std::vector<std::int32_t> util::file::EmbeddingExtractor::tokenize_prompt(
    const std::string& prompt) const
{
    const llama_vocab* vocab = llama_model_get_vocab(llama_->get_model());

    // get added sep and eos token, if any
    std::string added_sep_token;
    if (llama_vocab_get_add_sep(vocab)) {
        added_sep_token = llama_vocab_get_text(vocab, llama_vocab_sep(vocab));
    }

    std::string added_eos_token;
    if (llama_vocab_get_add_eos(vocab)) {
        added_eos_token = llama_vocab_get_text(vocab, llama_vocab_eos(vocab));
    }

    /* Helper
     */
    const auto tokenize
        = [](const std::string& str, const std::string& separator) {
        std::vector<std::string> tokens;
        std::size_t beg = 0;
        std::size_t end = str.find(separator);

        while (end != std::string::npos) {
            tokens.push_back(str.substr(beg, end - beg));
            beg = end + separator.length();
            end = str.find(separator, beg);
        }

        tokens.push_back(str.substr(beg));
        return tokens;
    };

    /* Helper
     */
    const auto repack_with_added_special_tokens
        = [&added_eos_token,
           &added_sep_token](const std::vector<std::string>& pairs) {
        std::string prompt;

        for (std::size_t i = 0; i < pairs.size(); i++) {
            prompt += pairs[i];
            if (i != (pairs.size() - 1)) {
                if (!added_eos_token.empty()) {
                    prompt += added_eos_token;
                }
                if (!added_sep_token.empty()) {
                    prompt += added_sep_token;
                }
            }
        }

        return prompt;
    };

    const enum llama_pooling_type pooling_type
        = llama_pooling_type(llama_->get_context());

    // tokenize the prompts and trim
    std::vector<llama_token> tokens;

    // split classification pairs and insert expected separator tokens
    if (pooling_type == LLAMA_POOLING_TYPE_RANK
        && prompt.find(llama_->get_params().cls_sep) != std::string::npos) {
        const std::vector<std::string> unpacked_prompt
            = tokenize(prompt, llama_->get_params().cls_sep);
        const std::string repacked_prompt
            = repack_with_added_special_tokens(unpacked_prompt);
        tokens = llm_util_tokenize(
            llama_->get_context(), repacked_prompt, true, true);
    } else {
        tokens = llm_util_tokenize(llama_->get_context(), prompt, true, true);
    }

    if (auto tokens_size = tokens.size();
        tokens_size > (llama_->get_params()).cparams.n_batch) {
        LOG_ERR(
            "%s: number of tokens in input line (%lu) exceeds batch "
            "size (%d), increase batch size and re-run",
            std::source_location::current().function_name(),
            tokens.size(),
            (llama_->get_params()).cparams.n_batch);
        return std::vector<llama_token>();
    }

    return tokens;
}

/*! @brief Helper
 */
std::vector<std::vector<llama_token>> util::file::EmbeddingExtractor::
    tokenize_prompts(const std::vector<std::string>& prompts) const
{
    const llama_vocab* vocab = llama_model_get_vocab(llama_->get_model());

    /* Helper
     */
    const auto print_info
        = [&vocab](
              const std::vector<std::vector<std::int32_t>>& tokens_out,
              const char* func) {
        // check if the last token is SEP/EOS
        // it should be automatically added by the tokenizer when
        // 'tokenizer.ggml.add_eos_token' is set to 'true'
        for (const auto& prompt_tokens: tokens_out) {
            if (prompt_tokens.empty()
                || (prompt_tokens.back() != llama_vocab_sep(vocab)
                    && prompt_tokens.back() != llama_vocab_eos(vocab))) {
                LOG_WRN("%s: last token in the prompt is not SEP or EOS", func);
                LOG_WRN(
                    "%s: 'tokenizer.ggml.add_eos_token' should be set to "
                    "'true' in the GGUF header",
                    func);
            }
        }
    };

    /* Helper
     */
    const auto print_stats
        = [](const std::vector<std::vector<std::int32_t>>& tokens_out,
             const std::vector<std::string>& prompts,
             const char* func) {
        // tokenization stats
        for (std::uint32_t i = 0; i < tokens_out.size(); i++) {
            LOG_DBG("%s: prompt %d: '%s'", func, i, prompts[i].c_str());
            LOG_DBG(
                "%s: number of tokens in prompt = %zu",
                func,
                tokens_out[i].size());
        }
    };

    // tokenize the prompts and trim
    std::vector<std::vector<llama_token>> tokens_set;
    for (const std::string& prompt: prompts) {
        std::vector<llama_token> tokens = tokenize_prompt(prompt);
        tokens_set.push_back(std::move(tokens));
    }

    // log output
    print_info(tokens_set, std::source_location::current().function_name());
    // log output
    print_stats(
        tokens_set, prompts, std::source_location::current().function_name());

    return tokens_set;
}
