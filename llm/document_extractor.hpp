#pragma once

#include "file/file_parse_update_sink.hpp"
#include "llm/chat.hpp"
#include "llm/configs.hpp"
#include "llm/extractor_base.hpp"
#include "llm/llama_base.hpp"
#include <llama.h>
#include <mtmd.h>
#include <string>

namespace util::file {

    class DocumentExtractor : public ExtractorBase {
    public:
        ~DocumentExtractor() override = default;

        DocumentExtractor(
            const std::shared_ptr<LlamaBase>& llama,
            OcrModelConfig config);

        std::string extract_text(
            const std::string& image_path,
            const std::shared_ptr<FileUpdateNotifier>& status);

        void initialize();

        void teardown();

        bool is_same(const LlamaBase* ptr) const override;

        bool is_not_same(const LlamaBase* ptr) const override;

        bool has_same_base_object(const ExtractorBase& rhs) const override;

        OcrModelConfig get_config() const;

        std::string to_identifier() const;

        llm_util_params get_llama_params() const;
    private:
        bool evaluate_message(
            mtmd::bitmaps& bitmaps,
            const util::file::llm_util_chat_message& message,
            bool add_bos,
            const std::shared_ptr<util::file::FileUpdateNotifier>& status);

        std::string generate_response(
            std::int32_t n_predict,
            const std::shared_ptr<util::file::FileUpdateNotifier>& status);

        mutable std::mutex lock_;

        //! Llama base api wrapper
        const std::shared_ptr<LlamaBase> llama_;
        //! Gguf config
        const OcrModelConfig config_;

        mtmd::context_ptr vision_context_;
        std::int32_t n_past_ = 0;
        std::int32_t n_batch_ = 0;
        llm_util_chat_templates_ref templates_;

        //! Sequence-level representation
        enum llama_pooling_type default_pooling_type_
            = llama_pooling_type::LLAMA_POOLING_TYPE_CLS;
    };
} // namespace util::file
