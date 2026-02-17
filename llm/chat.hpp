// Chat support (incl. tool call grammar constraining & output parsing) w/
// generic & custom template handlers.

#pragma once

#include <chrono>
#include <llama-cpp.h>
#include <minja/chat-template.hpp>
#include <string>
#include <vector>

namespace util::file {
    struct llm_util_chat_templates {
        bool add_bos = false;
        bool add_eos = false;

        bool has_explicit_template = false; // Model had builtin template or
                                            // template overridde was specified.

        // always set (defaults to chatml)
        std::unique_ptr<minja::chat_template> template_default;
    };

    struct llm_util_chat_msg_content_part {
        std::string type;
        std::string text;

        bool operator==(const llm_util_chat_msg_content_part& other) const
            = default;
    };

    struct llm_util_chat_message {
        std::string role;
        std::string content;
        std::vector<llm_util_chat_msg_content_part> content_parts;
        std::string reasoning_content;

        bool empty() const
        {
            return content.empty() && content_parts.empty()
                   && reasoning_content.empty();
        }

        bool operator==(const llm_util_chat_message& other) const = default;
    };

    struct llm_util_template_inputs {
        std::vector<llm_util_chat_message> messages;
        bool add_generation_prompt = true;

        std::chrono::system_clock::time_point now
            = std::chrono::system_clock::now();
        std::unordered_map<std::string, std::string> chat_template_kwargs;
    };

    struct llm_util_chat_inputs {
        std::string prompt;
        std::string grammar;
    };

    // Check whether the template supplied via arg 'chat-template' is supported
    // or not
    bool llm_util_chat_verify_template(const std::string& tmpl);

    struct llm_util_chat_templates_deleter {
        void operator()(llm_util_chat_templates* tmpls)
        {
            delete tmpls;
        }
    };

    using llm_util_chat_templates_ref = std::
        unique_ptr<llm_util_chat_templates, llm_util_chat_templates_deleter>;

    llm_util_chat_templates_ref llm_util_chat_templates_init(
        const llama_model* model,
        const std::string& chat_template_override);

    struct llm_util_chat_inputs llm_util_chat_templates_apply(
        const llm_util_chat_templates* templates,
        const llm_util_template_inputs& inputs);

    // Returns an example of formatted chat
    std::string llm_util_chat_format_example(
        const llm_util_chat_templates* templates,
        const std::unordered_map<std::string, std::string>&
            chat_template_kwargs);
} // namespace util::file
