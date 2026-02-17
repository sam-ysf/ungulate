#pragma once

#include "llm/llama_base.hpp"

namespace util ::file {

    class ExtractorBase {
    public:
        virtual ~ExtractorBase() = default;

        virtual bool is_same(const LlamaBase* ptr) const = 0;

        virtual bool is_not_same(const LlamaBase* ptr) const = 0;

        virtual bool has_same_base_object(const ExtractorBase& rhs) const = 0;
    };
} // namespace util::file
