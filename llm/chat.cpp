#include "llm/chat.hpp"
#include "llm/string_utils.hpp"
#include "llm/vocab_utils.hpp"
#include "log/log.hpp"
#include <cstdio>
#include <exception>
#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <nlohmann/json_fwd.hpp>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

std::string util::file::llm_util_chat_format_example(
    const llm_util_chat_templates* templates,
    const std::
        unordered_map<std::string, std::string>& /* chat_template_kwargs */)
{
    llm_util_template_inputs inputs;

    auto add_simple_message = [&inputs](auto role, auto content) {
        llm_util_chat_message message;
        message.role = role;
        message.content = content;
        inputs.messages.push_back(message);
    };

    add_simple_message("system", "You are a helpful assistant");
    add_simple_message("user", "Hello");
    add_simple_message("assistant", "Hi there");
    add_simple_message("user", "How are you?");

    return llm_util_chat_templates_apply(templates, inputs).prompt;
}

util::file::llm_util_chat_templates_ref util::file::
    llm_util_chat_templates_init(
        const llama_model* model,
        const std::string& chat_template_override)
{
    std::string default_template_src = chat_template_override;
    if (default_template_src.empty()) {
        const auto* template_str = llama_model_chat_template(model, nullptr);
        if (template_str) {
            default_template_src = template_str;
        }
    }

    static const std::string kChatMLTemplateSrc
        = "{%- for message in messages -%}\n"
          "  {{- '<|im_start|>' + message.role + '\n' + message.content + "
          "'<|im_end|>\n' -}}\n"
          "{%- endfor -%}\n"
          "{%- if add_generation_prompt -%}\n"
          "  {{- '<|im_start|>assistant\n' -}}\n"
          "{%- endif -%}";

    if (default_template_src.empty() || default_template_src == "chatml") {
        default_template_src = kChatMLTemplateSrc;
    }

    if (default_template_src.find("<|channel|>") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("in message.content or")
               != std::string::npos) {
        default_template_src = string_replace_all(
            default_template_src,
            R"({%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %})",
            "{%- if false %}");
    }

    std::string token_bos;
    std::string token_eos;
    bool add_bos = false;
    bool add_eos = false;

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const auto get_token = [vocab](llama_token token) {
        if (token == LLAMA_TOKEN_NULL) {
            return std::string();
        }
        return llm_util_token_to_piece(vocab, token, true);
    };

    token_bos = get_token(llama_vocab_bos(vocab));
    token_eos = get_token(llama_vocab_eos(vocab));

    add_bos = llama_vocab_get_add_bos(vocab);
    add_eos = llama_vocab_get_add_eos(vocab);

    llm_util_chat_templates_ref templates(new llm_util_chat_templates());

    const bool has_explicit_template
        = !chat_template_override.empty() || !default_template_src.empty();
    templates->has_explicit_template = has_explicit_template;
    templates->add_bos = add_bos;
    templates->add_eos = add_eos;

    try {
        templates->template_default = std::make_unique<minja::chat_template>(
            default_template_src, token_bos, token_eos);
    } catch (const std::exception& e) {
        LOG_ERR(
            "%s: failed to parse chat template (defaulting to chatml): %s",
            std::source_location::current().function_name(),
            e.what());
        templates->template_default = std::make_unique<minja::chat_template>(
            kChatMLTemplateSrc, token_bos, token_eos);
    }

    return templates;
}

util::file::llm_util_chat_inputs util::file::llm_util_chat_templates_apply(
    const llm_util_chat_templates* templates,
    const llm_util_template_inputs& template_inputs)
{
    float alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string> contents;

    for (const auto& message: template_inputs.messages) {
        std::string content = message.content;
        for (const auto& part: message.content_parts) {
            if (part.type != "text") {
                LOG_WRN(
                    "Ignoring non-text content part: %s", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }

    for (std::size_t i = 0; i < contents.size(); ++i) {
        const auto& message = template_inputs.messages[i];
        const auto& content = contents[i];
        chat.push_back(
            {.role = message.role.c_str(), .content = content.c_str()});
        alloc_size += std::ceil(
            static_cast<float>(message.role.size() + content.size()) * 1.25F);
    }

    std::vector<char> buff(static_cast<std::size_t>(alloc_size), 0);

    // run the first time
    // could fail if alloc'd buffer is smaller than the total output length
    const std::string& source = templates->template_default->source();
    int32_t prompt_len = llama_chat_apply_template(
        source.c_str(),
        chat.data(),
        chat.size(),
        template_inputs.add_generation_prompt,
        buff.data(),
        static_cast<int>(alloc_size));

    // throw error if chat template not supported
    if (prompt_len < 0) {
        throw std::runtime_error("this custom template is not supported");
    }

    const auto buff_len = static_cast<std::size_t>(prompt_len);

    // if it turns out that our buffer is too small, we resize it and try
    // again
    if (buff_len > buff.size()) {
        buff.resize(buff_len);
        llama_chat_apply_template(
            source.c_str(),
            chat.data(),
            chat.size(),
            template_inputs.add_generation_prompt,
            buff.data(),
            static_cast<int>(buff.size()));
    }

    util::file::llm_util_chat_inputs input
        = {.prompt = std::string(buff.data(), buff_len), .grammar = {}};

    return input;
}
