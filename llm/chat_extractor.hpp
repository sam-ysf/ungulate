#pragma once

#include "file/document.hpp"
#include "llm/chat.hpp"
#include "llm/configs.hpp"
#include "llm/extractor_base.hpp"
#include "llm/llama_base.hpp"
#include <string>
#include <vector>

namespace util::file {

    class ChatExtractor : public ExtractorBase {
    public:
        ~ChatExtractor() override = default;
        ChatExtractor(
            const std::shared_ptr<LlamaBase>& llama,
            PostprocessingConfig config);

        std::string extract(const std::string& prompt, const Document& document)
            const;

        void initialize();

        void teardown();

        bool is_same(const LlamaBase* ptr) const override;

        bool is_not_same(const LlamaBase* ptr) const override;

        bool has_same_base_object(const ExtractorBase& rhs) const override;

        PostprocessingConfig get_config() const;

        std::string to_identifier() const;

        llm_util_params get_llama_params() const;
    private:
        std::string generate_response(const std::string& prompt) const;

        std::string process_user_message(
            const llm_util_chat_templates* templates,
            const std::vector<llm_util_chat_message>& messages) const;

        std::vector<llama_token> tokenize_prompt(
            const std::string& prompt) const;

        mutable std::mutex lock_;

        //! Sequence-level representation
        enum llama_pooling_type default_pooling_type_
            = llama_pooling_type::LLAMA_POOLING_TYPE_CLS;

        //! Llama base api wrapper
        const std::shared_ptr<LlamaBase> llama_;
        //! Gguf config
        const PostprocessingConfig config_;
    };
} // namespace util::file
