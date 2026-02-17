#include "llm/document_extractor.hpp"
#include "llm/base_utils.hpp"
#include "llm/batch_utils.hpp"
#include "llm/chat.hpp"
#include "llm/configs.hpp"
#include "llm/cpu_utils.hpp"
#include "llm/llama_base.hpp"
#include "llm/sampling.hpp"
#include "llm/vocab_utils.hpp"
#include "log/log.hpp"
#include <climits>
#include <ggml.h>
#include <llama-cpp.h>
#include <llama.h>
#include <memory>
#include <mtmd-helper.h>
#include <mtmd.h>
#include <mutex>
#include <nlohmann/detail/input/binary_reader.hpp>
#include <nlohmann/detail/output/serializer.hpp>
#include <nlohmann/json.hpp>
#include <source_location>
#include <stdexcept>
#include <vector>

namespace {

    //! @struct MtMdParams
    struct MtMdParams {
        mtmd::context_ptr vision_context;

        std::int32_t n_past = 0;
        std::int32_t n_batch = 0;

        util::file::llm_util_chat_templates_ref templates;
    };

    inline mtmd::context_ptr generate_mtmd_vision_context(
        const util::file::llm_util_params& params,
        const llama_model* model,
        const std::string& mmproj)
    {
        mtmd::context_ptr ctx_vision;

        cpu_params cpuparams = params.cpuparams;
        if (cpuparams.n_threads == -1) {
            postprocess_cpu_params(cpuparams, nullptr);
        }

        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = true;
        mparams.print_timings = true;
        mparams.n_threads = cpuparams.n_threads;
        ctx_vision.reset(mtmd_init_from_file(mmproj.c_str(), model, mparams));
        return ctx_vision;
    }

    inline MtMdParams generate_mtmd_params(
        const util::file::llm_util_params& params,
        const llama_model* model,
        const std::string& mmproj)
    {
        MtMdParams mtmd_params;
        mtmd_params.n_batch = static_cast<std::int32_t>(params.cparams.n_batch);

        if (!llama_model_chat_template(model, nullptr)
            && params.chat_template.empty()) {
            LOG_WRN(
                "model %s does not have chat template.",
                params.mparams.source.gguf.c_str());
        }

        mtmd_params.templates = util::file::llm_util_chat_templates_init(
            model, params.chat_template);

        mtmd_params.vision_context
            = generate_mtmd_vision_context(params, model, mmproj);
        return mtmd_params;
    }

    inline std::optional<mtmd::bitmap> load_image(
        const std::string& path,
        mtmd_context* ctx_vision)
    {
        mtmd::bitmap bitmap
            = mtmd_helper_bitmap_init_from_file(ctx_vision, path.c_str());
        if (!bitmap.ptr)
            return std::nullopt;
        return bitmap;
    }
} // namespace

util::file::DocumentExtractor::DocumentExtractor(
    const std::shared_ptr<LlamaBase>& llama,
    OcrModelConfig config)
    : llama_(llama)
    , config_(std::move(config))
{}

std::string util::file::DocumentExtractor::extract_text(
    const std::string& image_path,
    const std::shared_ptr<FileUpdateNotifier>& status)
{
    std::scoped_lock<std::mutex> lock(lock_);

    const auto load_bitmap
        = [&image_path](
              mtmd_context* vision_context, mtmd::bitmaps& bitmaps /* out */) {
        std::optional<mtmd::bitmap> bitmap
            = load_image(image_path, vision_context);
        if (!bitmap) {
            return 0LU;
        }

        bitmaps.entries.push_back(std::move(bitmap.value()));
        return bitmaps.entries.size();
    };

    if (!std::filesystem::exists(image_path)) {
        throw std::invalid_argument(
            std::format("Image file {} does not exist", image_path));
    }

    llm_util_chat_message message;
    message.role = "user";
    message.content = [this] {
        nlohmann::ordered_json prompt_j;
        prompt_j["type"] = "text";
        prompt_j["text"] = config_.prompt;

        std::string prompt_str = prompt_j.dump(
            -1, 0, false, nlohmann::detail::error_handler_t::ignore);
        if (prompt_str.find(mtmd_default_marker()) == std::string::npos) {
            prompt_str += mtmd_default_marker();
        }

        return prompt_str;
    }();

    if (!vision_context_) {
        LOG_ERR(
            "%s: failed to load vision model from %s",
            std::source_location::current().function_name(),
            config_.mmproj.c_str());
        throw std::invalid_argument(
            std::format("Failed to load vision model {}", config_.mmproj));
    }

    if (!mtmd_support_vision(vision_context_.get())) {
        throw std::invalid_argument(
            std::format(
                "Vision model {} does not support vision input",
                config_.mmproj));
    }

    mtmd::bitmaps bitmaps;
    if (load_bitmap(vision_context_.get(), bitmaps) == 0) {
        throw std::invalid_argument(
            std::format("Image file {} could not be initialized", image_path));
    }

    // clear the cache before the next run
    if (config_.clear_cache) {
        llama_->clear_cache();
    }

    if (!evaluate_message(bitmaps, message, true, status)) {
        return std::string();
    }

    int n_predict = llama_->get_params().n_predict;
    if (n_predict < 0) {
        n_predict = INT_MAX;
    }

    std::string result = generate_response(n_predict, status);
    llama_perf_context_print(llama_->get_context());

    return result;
}

void util::file::DocumentExtractor::initialize()
{
    std::scoped_lock<std::mutex> lock(lock_);

    llama_->teardown();
    llama_->initialize(default_pooling_type_, false);

    MtMdParams mtmd = generate_mtmd_params(
        llama_->get_params(), llama_->get_model(), config_.mmproj);
    vision_context_ = std::move(mtmd.vision_context);
    n_past_ = mtmd.n_past;
    n_batch_ = mtmd.n_batch;
    templates_ = std::move(mtmd.templates);
}

void util::file::DocumentExtractor::teardown()
{
    std::scoped_lock<std::mutex> lock(lock_);

    vision_context_ = mtmd::context_ptr();
    templates_ = llm_util_chat_templates_ref();

    llama_->teardown();
}

bool util::file::DocumentExtractor::is_same(const LlamaBase* ptr) const
{
    return ptr == llama_.get();
}

bool util::file::DocumentExtractor::is_not_same(const LlamaBase* ptr) const
{
    return ptr != llama_.get();
}

bool util::file::DocumentExtractor::has_same_base_object(
    const ExtractorBase& rhs) const
{
    return rhs.is_same(llama_.get());
}

util::file::OcrModelConfig util::file::DocumentExtractor::get_config() const
{
    return config_;
}

std::string util::file::DocumentExtractor::to_identifier() const
{
    return config_.gguf_hash;
}

util::file::llm_util_params util::file::DocumentExtractor::get_llama_params()
    const
{
    return llama_->get_params();
}

bool util::file::DocumentExtractor::evaluate_message(
    mtmd::bitmaps& bitmaps,
    const util::file::llm_util_chat_message& message,
    bool add_bos,
    const std::shared_ptr<util::file::FileUpdateNotifier>& status)
{
    llama_context* context = llama_->get_context();

    util::file::llm_util_template_inputs tmpl_inputs;
    tmpl_inputs.messages = {message};
    tmpl_inputs.add_generation_prompt = true;

    auto formatted_chat
        = llm_util_chat_templates_apply(templates_.get(), tmpl_inputs);
    LOG_DBG("formatted_chat.prompt: %s", formatted_chat.prompt.c_str());

    // Interrupt check
    if (!status->is_still_set()) {
        return false;
    }

    mtmd_input_text text;
    text.text = formatted_chat.prompt.c_str();
    text.add_special = add_bos;
    text.parse_special = true;

    mtmd::input_chunks chunks = mtmd_input_chunks_init();
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    if (int32_t res = mtmd_tokenize(
            vision_context_.get(),
            chunks.ptr.get(), // output
            &text, // text
            bitmaps_c_ptr.data(),
            bitmaps_c_ptr.size());
        res != 0) {
        LOG_ERR(
            "%s: unable to tokenize prompt, res = %d",
            std::source_location::current().function_name(),
            res);
        throw std::runtime_error("unable to tokenize prompt");
    }

    // Interrupt check
    if (!status->is_still_set()) {
        return false;
    }

    llama_pos new_n_past = 0;
    if (mtmd_helper_eval_chunks(
            vision_context_.get(),
            context,
            chunks.ptr.get(),
            n_past_,
            0,
            n_batch_,
            true,
            &new_n_past)) {
        LOG_ERR(
            "%s: unable to evaluate prompt",
            std::source_location::current().function_name());
        throw std::runtime_error("unable to evaluate prompt");
    }

    n_past_ = new_n_past;
    return true;
}

namespace {

    std::string eval_token(
        const util::file::LlamaBase& llama,
        llama_batch& batch,
        std::int32_t n_past)
    {
        llama_context* context = llama.get_context();
        const llama_model* model = llama.get_model();
        util::file::llm_util_sampler* sampler = llama.get_sampler();

        const llama_vocab* vocab = llama_model_get_vocab(model);

        llama_token token_id = util::file::llm_util_sampler_sample(sampler, -1);
        util::file::llm_util_sampler_accept(sampler, token_id);

        if (llama_vocab_is_eog(vocab, token_id)) {
            return std::string(); // end of generation
        }

        // eval the token
        util::file::llm_util_batch_clear(batch);

        util::file::llm_util_batch_add(batch, token_id, n_past, {0}, true);
        if (llama_decode(context, batch)) {
            LOG_ERR(
                "%s: failed to decode token",
                std::source_location::current().function_name());
            return std::string();
        }

        std::string piece
            = util::file::llm_util_token_to_piece(context, token_id);
        return piece;
    }
} // namespace

std::string util::file::DocumentExtractor::generate_response(
    std::int32_t n_predict,
    const std::shared_ptr<util::file::FileUpdateNotifier>& status)
{
    llama_batch batch
        = llama_batch_init(1, 0, 1); // batch for next token generation

    llm_util_params params = llama_->get_params();

    std::string accumulator;
    for (std::int32_t i = 0; i < n_predict && status->is_still_set(); ++i) {
        std::string piece = eval_token(*llama_, batch, n_past_);
        if (piece.empty()) {
            break;
        }

        if (config_.verbose_parse_result) {
            fprintf(stdout, "%s", piece.c_str());
            fflush(stdout);
        }

        accumulator.insert(accumulator.end(), piece.begin(), piece.end());
        ++n_past_;
    }

    llama_batch_free(batch);
    return accumulator;
}
