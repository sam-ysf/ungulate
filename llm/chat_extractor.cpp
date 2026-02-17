#include "llm/chat_extractor.hpp"
#include "llm/chat.hpp"
#include "llm/configs.hpp"
#include "llm/llama_base.hpp"
#include "llm/sampling.hpp"
#include "llm/vocab_utils.hpp"
#include "log/log.hpp"
#include <llama.h>
#include <mutex>
#include <source_location>
#include <string>
#include <vector>

namespace {
    // Helper, checks if we have enough space in the context to evaluate this
    // batch
    void print_context_size(
        const llama_context* context,
        const llama_batch& batch)
    {
        std::uint32_t n_ctx = llama_n_ctx(context);
        std::int32_t n_ctx_used
            = llama_memory_seq_pos_max(llama_get_memory(context), 0);
        if (static_cast<std::uint32_t>(n_ctx_used + batch.n_tokens) > n_ctx) {
            LOG_ERR(
                "%s: context size exceeded",
                std::source_location::current().function_name());
        }
    }

    // Function to apply the chat template and resize `formatted` if needed
    std::string apply_chat_template(
        const util::file::llm_util_chat_templates* templates,
        const std::vector<util::file::llm_util_chat_message>& messages,
        const bool add_generation_prompt)
    {
        util::file::llm_util_template_inputs inputs;
        inputs.messages = messages;
        inputs.add_generation_prompt = add_generation_prompt;

        auto formatted_chat
            = util::file::llm_util_chat_templates_apply(templates, inputs);
        LOG_DBG("formatted_chat.prompt: %s", formatted_chat.prompt.c_str());

        return formatted_chat.prompt;
    }
} // namespace

util::file::ChatExtractor::ChatExtractor(
    const std::shared_ptr<LlamaBase>& llama,
    PostprocessingConfig config)
    : llama_(llama)
    , config_(std::move(config))
{}

std::string util::file::ChatExtractor::extract(
    const std::string& prompt,
    const util::file::Document& document) const
{
    std::vector<llm_util_chat_message> messages;

    llm_util_chat_message message;
    message.role = "user";
    message.content = prompt + "\n\n" + document.text;
    messages.push_back(std::move(message));

    const std::string& chat_template = llama_->get_params().chat_template;
    llm_util_chat_templates_ref chat_templates
        = llm_util_chat_templates_init(llama_->get_model(), chat_template);

    return process_user_message(chat_templates.get(), messages);
}

void util::file::ChatExtractor::initialize()
{
    std::scoped_lock<std::mutex> lock(lock_);

    llama_->teardown();
    llama_->initialize(default_pooling_type_, false);
}

void util::file::ChatExtractor::teardown()
{
    std::scoped_lock<std::mutex> lock(lock_);

    llama_->teardown();
}

bool util::file::ChatExtractor::is_same(const LlamaBase* ptr) const
{
    return ptr == llama_.get();
}

bool util::file::ChatExtractor::is_not_same(const LlamaBase* ptr) const
{
    return ptr != llama_.get();
}

bool util::file::ChatExtractor::has_same_base_object(
    const ExtractorBase& rhs) const
{
    return rhs.is_same(llama_.get());
}

util::file::PostprocessingConfig util::file::ChatExtractor::get_config() const
{
    return config_;
}

std::string util::file::ChatExtractor::to_identifier() const
{
    return config_.gguf_hash;
}

util::file::llm_util_params util::file::ChatExtractor::get_llama_params() const
{
    return llama_->get_params();
}

/* Helper, evaluates a prompt and generates a response */
std::string util::file::ChatExtractor::generate_response(
    const std::string& prompt) const
{
    std::vector<llama_token> tokens = tokenize_prompt(prompt);
    if (tokens.empty()) {
        return std::string();
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(
        tokens.data(), static_cast<std::int32_t>(tokens.size()));

    std::string accumulator;

    while (true) {
        print_context_size(llama_->get_context(), batch);

        if (llama_decode(llama_->get_context(), batch)) {
            LOG_ERR(
                "%s: failed to decode",
                std::source_location::current().function_name());
            break;
        }

        // sample the next token, check is it an end of generation?
        llama_token new_token_id
            = llm_util_sampler_sample(llama_->get_sampler(), -1);

        const llama_vocab* vocab = llama_model_get_vocab(llama_->get_model());

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        std::string piece = llm_util_token_to_piece(vocab, new_token_id);
        accumulator.insert(accumulator.end(), piece.begin(), piece.end());

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    return accumulator;
}

std::string util::file::ChatExtractor::process_user_message(
    const llm_util_chat_templates* templates,
    const std::vector<llm_util_chat_message>& messages) const
{
    const std::string prompt = apply_chat_template(
        templates,
        messages,
        /* add_generation_prompt */ false);
    return generate_response(prompt);
}

// Function to tokenize the prompt
std::vector<llama_token> util::file::ChatExtractor::tokenize_prompt(
    const std::string& prompt) const
{
    const llama_vocab* vocab = llama_model_get_vocab(llama_->get_model());

    llama_memory_t llama_memory_index = llama_get_memory(llama_->get_context());
    const bool is_first = llama_memory_seq_pos_max(llama_memory_index, 0) == -1;

    std::int32_t n_tokens = [&prompt, is_first] {
        auto n_tokens_value = static_cast<std::int32_t>(prompt.size());
        if (is_first)
            n_tokens_value += 2;
        return n_tokens_value;
    }();

    std::vector<llama_token> tokens(static_cast<std::size_t>(n_tokens), 0);

    n_tokens = llama_tokenize(
        vocab,
        prompt.data(),
        static_cast<std::int32_t>(prompt.size()),
        tokens.data(),
        static_cast<std::int32_t>(tokens.size()),
        is_first,
        /* parse_special */ true);

    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        LOG_ERR(
            "%s: tokenization failed: input too large",
            std::source_location::current().function_name());
        return std::vector<llama_token>();
    }

    if (n_tokens == 0) {
        return std::vector<llama_token>();
    }

    if (n_tokens > 0) {
        tokens.resize(static_cast<std::size_t>(n_tokens), 0);
        return tokens;
    }

    // Error: Resize to the number of tokens that would have been returned and
    // try again
    tokens.resize(static_cast<std::size_t>(std::abs(n_tokens)), 0);
    if (int n_tokens_check = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<std::int32_t>(prompt.size()),
            tokens.data(),
            std::abs(n_tokens),
            is_first,
            /*parse_special */ true);
        n_tokens_check != -n_tokens) {
        LOG_ERR(
            "%s: failed to tokenize the prompt (size mismatch)",
            std::source_location::current().function_name());
        return std::vector<llama_token>();
    }

    return tokens;
}
