#include "llm/arg.hpp"
#include "llm/base_utils.hpp"
#include "llm/cpu_utils.hpp"
#include "llm/sampling.hpp"
#include "llm/string_utils.hpp"
#include "log/log.hpp"
#include <algorithm>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstring>
#include <format>
#include <ggml-backend.h>
#include <list>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <regex>
#include <source_location>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace util::file {

    Arg& Arg::set_sparam(bool enable)
    {
        is_sparam = enable;
        return *this;
    }

    nlohmann::json Arg::to_json() const
    {
        if (args_list.empty()) {
            return nlohmann::json();
        }

        nlohmann::json j;
        j["args"] = args_list;

        if (!value_hints.empty()) {
            j["value-hints"] = value_hints;
        }

        if (!help.empty()) {
            j["help"] = help;
        }

        return j;
    }

    namespace {

        constexpr std::vector<ggml_type> kv_cache_types()
        {
            return std::vector<ggml_type>{
                GGML_TYPE_F32,
                GGML_TYPE_F16,
                GGML_TYPE_BF16,
                GGML_TYPE_Q8_0,
                GGML_TYPE_Q4_0,
                GGML_TYPE_Q4_1,
                GGML_TYPE_IQ4_NL,
                GGML_TYPE_Q5_0,
                GGML_TYPE_Q5_1,
            };
        }

        inline bool is_truthy(const std::string& value)
        {
            return value == "on" || value == "enabled" || value == "1";
        }

        inline bool is_falsey(const std::string& value)
        {
            return value == "off" || value == "disabled" || value == "0";
        }

        inline bool is_autoy(const std::string& value)
        {
            return value == "auto" || value == "-1";
        }

        inline ggml_type kv_cache_type_from_str(const std::string& s)
        {
            for (const ggml_type& type: kv_cache_types()) {
                if (ggml_type_name(type) == s) {
                    return type;
                }
            }
            throw std::runtime_error("unsupported cache type: " + s);
        }

        inline std::string get_all_kv_cache_types()
        {
            const std::string delimeter = ", ";

            std::ostringstream message;
            for (const auto& type: kv_cache_types()) {
                message << ggml_type_name(type) << delimeter;
            }

            std::string str = message.str();

            // Sanity check
            if (str.empty()) {
                return std::string();
            }

            return str.substr(0, str.size() - delimeter.size());
        }

        inline std::vector<ggml_backend_dev_t> parse_device_list(
            const std::string& value)
        {
            std::vector<ggml_backend_dev_t> devices;
            auto dev_names = util::file::string_split(value, ',');
            if (dev_names.empty()) {
                throw std::invalid_argument("no devices specified");
            }
            if (dev_names.size() == 1 && dev_names[0] == "none") {
                devices.push_back(nullptr);
            } else {
                for (const auto& device: dev_names) {
                    auto* dev = ggml_backend_dev_by_name(device.c_str());
                    if (!dev
                        || ggml_backend_dev_type(dev)
                               != GGML_BACKEND_DEVICE_TYPE_GPU) {
                        throw std::invalid_argument(
                            std::format(R"(invalid device: {})", device));
                    }
                    devices.push_back(dev);
                }
                devices.push_back(nullptr);
            }
            return devices;
        }

        // FUTURE
        // Parse from Json config directly instead
        inline bool llm_util_string_parse_kv_override(
            const std::string& phrase,
            std::vector<llama_model_kv_override>& overrides)
        {
            llama_model_kv_override kvo = {};

            std::optional<std::string> key = [&phrase] {
                std::size_t size = phrase.find('=');
                if (size > sizeof(kvo.key)) {
                    return std::optional<std::string>();
                }

                return std::optional(phrase.substr(0, size));
            }();

            if (!key) {
                LOG_ERR(
                    R"(%s: malformed KV override '%s')",
                    std::source_location::current().function_name(),
                    phrase.c_str());
                return false;
            }

            std::optional<std::string> type = [&phrase, &key] {
                std::size_t start = key->size() + 1;
                std::size_t end = phrase.find(':', start);
                std::size_t size = end - start;

                if ((end == std::string::npos) || size == 0) {
                    return std::optional<std::string>();
                }

                return std::optional(phrase.substr(start, size));
            }();

            if (!type) {
                LOG_ERR(
                    R"(%s: malformed KV override '%s')",
                    std::source_location::current().function_name(),
                    phrase.c_str());
                return false;
            }

            std::string value = [&phrase, &key, &type] {
                std::size_t start = (key->size() + 1) + (type->size() + 1);
                return phrase.substr(start);
            }();

            std::memcpy(kvo.key, key->data(), key->size());

            static const std::string kBoolStr = "bool";
            static const std::string kIntStr = "int";
            static const std::string kFloatStr = "float";
            static const std::string kStrStr = "str";

            if (key == kIntStr) {
                kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
                kvo.val_i64 = std::stol(value);
            }

            else if (key == kFloatStr) {
                kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
                kvo.val_f64 = std::stof(value);
            }

            else if (key == kBoolStr) {
                kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;

                std::string value_nocase;
                value_nocase.reserve(value.size());
                for (char ch: value) {
                    value_nocase.push_back(static_cast<char>(std::tolower(ch)));
                }

                if (value == "true")
                    kvo.val_bool = true;
                else if (value == "false")
                    kvo.val_bool = false;
                else {
                    LOG_ERR(
                        R"(%s: invalid boolean value for KV override '%s')",
                        std::source_location::current().function_name(),
                        phrase.c_str());
                    return false;
                }
            }

            else if (key == kStrStr) {
                kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                if (value.size() > 127) {
                    LOG_ERR(
                        "%s: malformed KV override '%s', value cannot "
                        "exceed "
                        "127 chars",
                        std::source_location::current().function_name(),
                        phrase.c_str());
                    return false;
                }

                std::memcpy(kvo.val_str, value.data(), value.size());
            }

            else {
                LOG_ERR(
                    R"(%s: invalid type for KV override '%s')",
                    std::source_location::current().function_name(),
                    phrase.c_str());
                return false;
            }

            overrides.emplace_back(std::move(kvo));
            return true;
        }

        // Helper function to parse tensor buffer override strings
        inline void parse_tensor_buffer_overrides(
            const std::string& value,
            llm_util_model_params& mparams)
        {
            std::vector<llama_model_tensor_buft_override>& overrides
                = mparams.tensor_buft_overrides;
            std::shared_ptr<std::vector<std::string>>& buft_overrides_alloc
                = mparams.buft_overrides_alloc;
            if (!buft_overrides_alloc) {
                buft_overrides_alloc
                    = std::make_shared<std::vector<std::string>>();
            }

            std::map<std::string, ggml_backend_buffer_type_t> buft_list;
            for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto* dev = ggml_backend_dev_get(i);
                auto* buft = ggml_backend_dev_buffer_type(dev);
                if (buft) {
                    const std::string name = ggml_backend_buft_name(buft);
                    buft_list[name] = buft;
                }
            }

            for (const auto& token: util::file::string_split(value, ',')) {
                std::string::size_type pos = token.find('=');
                if (pos == std::string::npos) {
                    throw std::invalid_argument("invalid value");
                }
                std::string tensor_name = token.substr(0, pos);
                std::string buffer_type = token.substr(pos + 1);

                if (!buft_list.contains(buffer_type)) {
                    throw std::invalid_argument("unknown buffer type");
                }

                buft_overrides_alloc->push_back(tensor_name);
                overrides.push_back(
                    {.pattern = buft_overrides_alloc->back().c_str(),
                     .buft = buft_list.at(buffer_type)});
            }
        }

        struct llm_util_params_context {
            std::vector<Arg> options;
            explicit llm_util_params_context(
                const llm_util_params& params) noexcept(false);
        };

        llm_util_params_context::llm_util_params_context(
            const llm_util_params& params) noexcept(false)
        {
            const std::string sampler_type_chars = [&params] {
                std::string value;
                for (const llm_util_sampler_type& sampler:
                     params.sampling.samplers)
                    value += llm_util_sampler_type_to_abbv(sampler);
                return value;
            }();

            const std::string sampler_type_names = [&params] {
                std::string value;
                for (std::size_t i = 0; i != params.sampling.samplers.size();
                     ++i) {
                    const llm_util_sampler_type& sampler
                        = params.sampling.samplers[i];
                    if (i != 0)
                        value += ";";
                    value += llm_util_sampler_type_to_abbv(sampler);
                }
                return value;
            }();

            const auto add_opt
                = [this](
                      const std::initializer_list<std::string>& args_list,
                      const std::string& help,
                      const Arg::ArgHandlerType& arg_handler) -> Arg& {
                Arg arg(args_list, {}, help, arg_handler);
                options.push_back(std::move(arg));
                return options.back();
            };

            const auto add_opt_with_hint
                = [this](
                      const std::initializer_list<std::string>& args_list,
                      const std::vector<std::string>& value_hints,
                      const std::string& help,
                      const Arg::ArgHandlerType& arg_handler) -> Arg& {
                Arg arg(args_list, value_hints, help, arg_handler);
                options.push_back(std::move(arg));
                return options.back();
            };

            using ArgCallbackType
                = std::function<void(llm_util_params&, const std::string&)>;
            std::unordered_map<std::string, ArgCallbackType> callbacks;

            callbacks["threads-batch"]
                = [](llm_util_params& params, const std::string& value) {
                int n_threads = std::stoi(value);
                params.cpuparams_batch.n_threads = n_threads;
                if (params.cpuparams_batch.n_threads <= 0) {
                    params.cpuparams_batch.n_threads
                        = static_cast<std::int32_t>(
                            std::thread::hardware_concurrency());
                }
            };

            callbacks["cpu-mask"]
                = [](llm_util_params& params, const std::string& mask) {
                params.cpuparams.mask_valid = true;
                if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                    throw std::invalid_argument("invalid cpumask");
                }
            };

            callbacks["cpu-range"]
                = [](llm_util_params& params, const std::string& range) {
                params.cpuparams.mask_valid = true;
                if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                    throw std::invalid_argument("invalid range");
                }
            };

            callbacks["cpu-strict"]
                = [](llm_util_params& params, const std::string& value) {
                params.cpuparams.strict_cpu = std::stoul(value);
            };

            callbacks["prio"]
                = [](llm_util_params& params, const std::string& value) {
                int prio = std::stoi(value);
                if (prio < GGML_SCHED_PRIO_LOW
                    || prio > GGML_SCHED_PRIO_REALTIME)
                    throw std::invalid_argument("invalid value");
                params.cpuparams.priority = ggml_sched_priority(prio);
            };

            callbacks["poll"]
                = [](llm_util_params& params, const std::string& value) {
                std::int32_t poll = std::stoi(value);
                if (poll > 0) {
                    params.cpuparams.poll = static_cast<std::uint32_t>(poll);
                }
            };

            callbacks["cpu-mask-batch"]
                = [](llm_util_params& params, const std::string& mask) {
                params.cpuparams_batch.mask_valid = true;
                if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                    throw std::invalid_argument("invalid cpumask");
                }
            };

            callbacks["cpu-range-batch"]
                = [](llm_util_params& params, const std::string& range) {
                params.cpuparams_batch.mask_valid = true;
                if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                    throw std::invalid_argument("invalid range");
                }
            };

            callbacks["prio-batch"]
                = [](llm_util_params& params, const std::string& value) {
                const char* str = value.c_str();
                int prio = std::stoi(str);

                if (prio < GGML_SCHED_PRIO_LOW
                    || prio > GGML_SCHED_PRIO_REALTIME)
                    throw std::invalid_argument("invalid value");
                params.cpuparams_batch.priority = ggml_sched_priority(prio);
            };

            callbacks["cpu-strict-batch"]
                = [](llm_util_params& params, const std::string& value) {
                const char* str = value.c_str();
                params.cpuparams_batch.strict_cpu = std::stoi(str) != 0;
            };

            callbacks["poll-batch"]
                = [](llm_util_params& params, const std::string& value) {
                const char* str = value.c_str();
                int poll = std::stoi(str);
                if (poll < 0 || poll > 100)
                    throw std::invalid_argument("invalid value");
                params.cpuparams_batch.poll = static_cast<std::uint32_t>(poll);
            };

            callbacks["predict"]
                = [](llm_util_params& params, const std::string& value) {
                params.n_predict = std::stoi(value);
            };

            callbacks["ctx-size"]
                = [](llm_util_params& params, const std::string& value) {
                std::int32_t n_ctx = std::stoi(value);
                if (n_ctx > 0) {
                    params.cparams.n_ctx = static_cast<std::uint32_t>(n_ctx);
                }
            };

            callbacks["batch-size"]
                = [](llm_util_params& params, const std::string& value) {
                std::int32_t n_batch = std::stoi(value);
                if (n_batch > 0) {
                    params.cparams.n_batch
                        = static_cast<std::uint32_t>(n_batch);
                }
            };

            callbacks["ubatch-size"]
                = [](llm_util_params& params, const std::string& value) {
                std::int32_t n_ubatch = std::stoi(value);
                if (n_ubatch > 0) {
                    params.cparams.n_ubatch
                        = static_cast<std::uint32_t>(n_ubatch);
                }
            };

            add_opt_with_hint(
                {"predict"},
                {"N"},
                std::format(
                    R"(number of tokens to predict (default: {}, -1 = infinity))",
                    params.n_predict),
                callbacks.at("predict"));

            add_opt_with_hint(
                {"ctx-size"},
                {"N"},
                std::format(
                    R"(size of the prompt context (default: {}, 0 = loaded from model))",
                    params.cparams.n_ctx),
                callbacks["ctx-size"]);

            add_opt_with_hint(
                {"batch-size"},
                {"N"},
                std::format(
                    R"(logical maximum batch size (default: {}))",
                    params.cparams.n_batch),
                callbacks.at("batch-size"));

            add_opt_with_hint(
                {"ubatch-size"},
                {"N"},
                std::format(
                    R"(physical maximum batch size (default: {}))",
                    params.cparams.n_ubatch),
                callbacks.at("ubatch-size"));

            add_opt_with_hint(
                {"parallel"},
                {"N"},
                std::format(
                    R"(number of parallel sequences to decode (default: {}))",
                    params.cparams.n_parallel),
                [](llm_util_params& params, const std::string& value) {
                std::int32_t n = std::stoi(value);
                if (n > 0) {
                    params.cparams.n_parallel = static_cast<std::uint32_t>(n);
                }
            });

            add_opt(
                {"warmup"},
                R"(skip warming up the model with an empty run)",
                [](llm_util_params& params, const std::string& value) {
                params.warmup = (value == "true");
            });

            add_opt_with_hint(
                {"threads"},
                {"N"},
                std::format(
                    R"(number of threads to use during generation (default: {}))",
                    params.cpuparams.n_threads),
                [](llm_util_params& params, const std::string& value) {
                int n_threads = std::stoi(value);
                params.cpuparams.n_threads = n_threads;
                if (n_threads <= 0) {
                    params.cpuparams.n_threads = static_cast<std::int32_t>(
                        std::thread::hardware_concurrency());
                }
            });

            add_opt_with_hint(
                {"threads-batch"},
                {"N"},
                R"(number of threads to use during batch and prompt processing (default: same as threads))",
                callbacks.at("threads-batch"));

            add_opt_with_hint(
                {"cpu-mask"},
                {"N"},
                R"(CPU affinity mask: arbitrarily long hex. Complements cpu-range)",
                callbacks.at("cpu-mask"));

            add_opt_with_hint(
                {"cpu-range"},
                {"lo-hi"},
                R"(range of CPUs for affinity. Complements cpu-mask)",
                callbacks["cpu-range"]);

            add_opt_with_hint(
                {"cpu-strict"},
                {"<0|1>"},
                std::format(
                    R"(use strict CPU placement (default: {}))",
                    static_cast<std::int32_t>(params.cpuparams.strict_cpu)),
                callbacks.at("cpu-strict"));

            add_opt_with_hint(
                {"prio"},
                {"N"},
                std::format(
                    R"(set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: {}))",
                    static_cast<std::int32_t>(params.cpuparams.priority)),
                callbacks.at("prio"));

            add_opt_with_hint(
                {"poll"},
                {"<0...100>"},
                std::format(
                    R"(use polling level to wait for work (0 - no polling, default: {}))",
                    params.cpuparams.poll),
                callbacks.at("poll"));

            add_opt_with_hint(
                {"cpu-mask-batch"},
                {"N"},
                R"(CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch)",
                callbacks.at("cpu-mask-batch"));

            add_opt_with_hint(
                {"cpu-range-batch"},
                {"lo-hi"},
                R"(ranges of CPUs for affinity. Complements cpu-mask-batch)",
                callbacks.at("cpu-range-batch"));

            add_opt_with_hint(
                {"cpu-strict-batch"},
                {"<0|1>"},
                R"(use strict CPU placement (default: same as cpu-strict))",
                callbacks["cpu-strict-batch"]);

            add_opt_with_hint(
                {"prio-batch"},
                {"N"},
                std::format(
                    R"(set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: {}))",
                    static_cast<std::int32_t>(params.cpuparams_batch.priority)),
                callbacks["prio-batch"]);

            add_opt_with_hint(
                {"poll-batch"},
                {"<0|1>"},
                R"(use polling to wait for work (default: same as poll))",
                callbacks["poll-batch"]);

            add_opt(
                {"lookup-cache-static"},
                R"(path to static lookup cache to use for lookup decoding (not updated by generation))",
                callbacks["lookup-cache-static"]);

            add_opt(
                {"lookup-cache-dynamic"},
                R"(path to dynamic lookup cache to use for lookup decoding (updated by generation))",
                callbacks["lookup-cache-dynamic"]);

            add_opt_with_hint(
                {"flash-attn"},
                {"[on|off|auto]"},
                std::format(
                    R"(set Flash Attention use ('on', 'off', or 'auto', default: '{}'))",
                    llama_flash_attn_type_name(params.cparams.flash_attn_type)),
                [](llm_util_params& params, const std::string& value) {
                if (is_truthy(value)) {
                    params.cparams.flash_attn_type
                        = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                } else if (is_falsey(value)) {
                    params.cparams.flash_attn_type
                        = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                } else if (is_autoy(value)) {
                    params.cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
                } else {
                    throw std::runtime_error(
                        std::format(
                            R"(error: unknown value for flash-attn: '{}')",
                            value));
                }
            });

            add_opt(
                {"samplers"},
                std::format(
                    R"(samplers that will be used for generation in the order, separated by ';' (default: {}))",
                    sampler_type_names),
                [](llm_util_params& params, const std::string& value) {
                const auto sampler_names = util::file::string_split(value, ';');
                params.sampling.samplers
                    = llm_util_sampler_types_from_names(sampler_names, true);
            }).set_sparam();

            add_opt(
                {"seed"},
                std::format(
                    R"(RNG seed (default: {}, use random seed for {}))",
                    params.sampling.seed,
                    LLAMA_DEFAULT_SEED),
                [](llm_util_params& params, const std::string& value) {
                std::int32_t n = std::stoi(value);
                if (n > 0) {
                    params.sampling.seed = static_cast<std::uint32_t>(n);
                }
            }).set_sparam();

            add_opt(
                {"sampling-seq"},
                std::format(
                    R"(simplified sequence for samplers that will be used (default: {}))",
                    sampler_type_chars),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.samplers
                    = llm_util_sampler_types_from_abbvs(value);
            }).set_sparam();

            add_opt(
                {"ignore-eos"},
                R"(ignore end of stream token and continue generating (implies logit-bias EOS-inf))",
                [](llm_util_params& params, const std::string& value) {
                params.sampling.ignore_eos = (value == "true");
            }).set_sparam();

            add_opt_with_hint(
                {"temp"},
                {"N"},
                std::format(
                    R"(temperature (default: {}))",
                    static_cast<double>(params.sampling.temp)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.temp = std::stof(value);
                params.sampling.temp = std::max(params.sampling.temp, 0.0F);
            }).set_sparam();

            add_opt_with_hint(
                {"top-k"},
                {"N"},
                std::format(
                    R"(top-k sampling (default: {}, 0 = disabled))",
                    params.sampling.top_k),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.top_k = std::stoi(value);
            }).set_sparam();

            add_opt_with_hint(
                {"top-p"},
                {"N"},
                std::format(
                    R"(top-p sampling (default: {}, 1.0 = disabled))",
                    static_cast<double>(params.sampling.top_p)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.top_p = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"min-p"},
                {"N"},
                std::format(
                    R"(min-p sampling (default: {}, 0.0 = disabled))",
                    static_cast<double>(params.sampling.min_p)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.min_p = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"top-nsigma"},
                {"N"},
                std::format(
                    R"(top-n-sigma sampling (default: {}, -1.0 = disabled))",
                    params.sampling.top_n_sigma),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.top_n_sigma = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"xtc-probability"},
                {"N"},
                std::format(
                    R"(xtc probability (default: {}, 0.0 = disabled))",
                    static_cast<double>(params.sampling.xtc_probability)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.xtc_probability = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"xtc-threshold"},
                {"N"},
                std::format(
                    R"(xtc threshold (default: {}, 1.0 = disabled))",
                    static_cast<double>(params.sampling.xtc_threshold)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.xtc_threshold = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"typical"},
                {"N"},
                std::format(
                    R"(locally typical sampling, parameter p (default: {}, 1.0 = disabled))",
                    static_cast<double>(params.sampling.typ_p)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.typ_p = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"repeat-last-n"},
                {"N"},
                std::format(
                    R"(last N tokens to consider for penalize (default: {}, 0 = disabled, -1 = ctx_size))",
                    params.sampling.penalty_last_n),
                [](llm_util_params& params, const std::string& value) {
                int penalty_last_n = std::stoi(value);
                params.sampling.penalty_last_n = penalty_last_n;
                params.sampling.n_prev = std::max(
                    params.sampling.n_prev, params.sampling.penalty_last_n);
            }).set_sparam();

            add_opt_with_hint(
                {"repeat-penalty"},
                {"N"},
                std::format(
                    R"(penalize repeat sequence of tokens (default: {}, 1.0 = disabled))",
                    static_cast<double>(params.sampling.penalty_repeat)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.penalty_repeat = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"presence-penalty"},
                {"N"},
                std::format(
                    R"(repeat alpha presence penalty (default: {}, 0.0 = disabled))",
                    static_cast<double>(params.sampling.penalty_present)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.penalty_present = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"frequency-penalty"},
                {"N"},
                std::format(
                    R"(repeat alpha frequency penalty (default: {}, 0.0 = disabled))",
                    static_cast<double>(params.sampling.penalty_freq)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.penalty_freq = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"dry-multiplier"},
                {"N"},
                std::format(
                    R"(set DRY sampling multiplier (default: {}, 0.0 = disabled))",
                    static_cast<double>(params.sampling.dry_multiplier)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.dry_multiplier = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"dry-base"},
                {"N"},
                std::format(
                    R"(set DRY sampling base value (default: {}))",
                    static_cast<double>(params.sampling.dry_base)),
                [](llm_util_params& params, const std::string& value) {
                float potential_base = std::stof(value);
                if (potential_base >= 1.0F) {
                    params.sampling.dry_base = potential_base;
                }
            }).set_sparam();

            add_opt_with_hint(
                {"dry-allowed-length"},
                {"N"},
                std::format(
                    R"(set allowed length for DRY sampling (default: {}))",
                    params.sampling.dry_allowed_length),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.dry_allowed_length = std::stoi(value);
            }).set_sparam();

            add_opt_with_hint(
                {"dry-penalty-last-n"},
                {"N"},
                std::format(
                    R"(set DRY penalty for the last N tokens (default: {}, 0 = disable, -1 = context size))",
                    params.sampling.dry_penalty_last_n),
                [](llm_util_params& params, const std::string& value) {
                int dry_penalty_last_n = std::stoi(value);
                params.sampling.dry_penalty_last_n = dry_penalty_last_n;
            }).set_sparam();

            // called by dry-sequence-breaker
            const auto get_dry_sequence_breakers = [&params]() {
                std::string breaker = "\\n";
                if (params.sampling.dry_sequence_breakers[0] == "\n") {
                    breaker = params.sampling.dry_sequence_breakers[0];
                }

                return std::accumulate(
                    std::next(params.sampling.dry_sequence_breakers.begin()),
                    params.sampling.dry_sequence_breakers.end(),
                    std::format("'{}'", breaker),
                    [](const std::string& a, const std::string& b) {
                    return std::format("{}, '{}'", a, (b == "\n" ? "\\n" : b));
                });
            };

            add_opt_with_hint(
                {"dry-sequence-breaker"},
                {"string"},
                std::format(
                    R"(add sequence breaker for DRY sampling, clearing out default breakers ({}) in the process; use "none" to not use any sequence breakers)",
                    params.sampling.dry_sequence_breakers.empty()
                        ? "none"
                        : get_dry_sequence_breakers()),
                [](llm_util_params& params, const std::string& value) {
                if (static bool defaults_cleared = false; !defaults_cleared) {
                    params.sampling.dry_sequence_breakers.clear();
                    defaults_cleared = true;
                }

                if (value == "none") {
                    params.sampling.dry_sequence_breakers.clear();
                } else {
                    params.sampling.dry_sequence_breakers.emplace_back(value);
                }
            }).set_sparam();

            add_opt_with_hint(
                {"dynatemp-range"},
                {"N"},
                std::format(
                    R"(dynamic temperature range (default: {}, 0.0 = disabled))",
                    static_cast<double>(params.sampling.dynatemp_range)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.dynatemp_range = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"dynatemp-exp"},
                {"N"},
                std::format(
                    R"(dynamic temperature exponent (default: {}))",
                    static_cast<double>(params.sampling.dynatemp_exponent)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.dynatemp_exponent = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"mirostat"},
                {"N"},
                std::format(
                    R"(use Mirostat sampling. Top K, Nucleus, and Locally Typical samplers are ignored if used. (default: {}, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0))",
                    params.sampling.mirostat),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.mirostat = std::stoi(value);
            }).set_sparam();

            add_opt_with_hint(
                {"mirostat-lr"},
                {"N"},
                std::format(
                    R"(Mirostat learning rate, parameter eta (default: {}))",
                    static_cast<double>(params.sampling.mirostat_eta)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.mirostat_eta = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"mirostat-ent"},
                {"N"},
                std::format(
                    R"(Mirostat target entropy, parameter tau (default: {}))",
                    static_cast<double>(params.sampling.mirostat_tau)),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.mirostat_tau = std::stof(value);
            }).set_sparam();

            add_opt_with_hint(
                {"logit-bias"},
                {"token-id(+/-)bias"},
                R"(modifies the likelihood of token appearing in the completion, i.e. \"logit-bias 15043+1\" to increase likelihood of token 'Hello', or \"logit-bias 15043-1\" to decrease likelihood of token 'Hello')",
                [](llm_util_params& params, const std::string& value) {
                llama_token token_id = 0;
                char sign = 0;
                std::string bias_str;

                std::stringstream ss(value);
                if (!(ss >> token_id)) {
                    throw std::invalid_argument("invalid input format");
                }

                if (!(ss >> sign)) {
                    throw std::invalid_argument("invalid input format");
                }

                if (!std::getline(ss, bias_str)) {
                    throw std::invalid_argument("invalid input format");
                }

                const float bias
                    = std::stof(bias_str) * ((sign == '-') ? -1.0F : 1.0F);
                params.sampling.logit_bias.push_back(
                    {.token = token_id, .bias = bias});
            }).set_sparam();

            add_opt_with_hint(
                {"grammar"},
                {"string"},
                std::format(
                    R"(BNF-like grammar to constrain generations (default: '{}'))",
                    params.sampling.grammar.grammar),
                [](llm_util_params& params, const std::string& value) {
                params.sampling.grammar.grammar = value;
            }).set_sparam();

            add_opt(
                {"grammar-file"},
                R"(file to read grammar from)",
                [](llm_util_params& params, const std::string& value) {
                params.sampling.grammar.grammar = fs_read_file(value);
            }).set_sparam();

            add_opt_with_hint(
                {"pooling"},
                {"{none,mean,cls,last,rank}"},
                R"(pooling type for embeddings, use model default if unspecified)",
                [](llm_util_params& params, const std::string& value) {
                if (value == "none") {
                    params.cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
                } else if (value == "mean") {
                    params.cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                } else if (value == "cls") {
                    params.cparams.pooling_type = LLAMA_POOLING_TYPE_CLS;
                } else if (value == "last") {
                    params.cparams.pooling_type = LLAMA_POOLING_TYPE_LAST;
                } else if (value == "rank") {
                    params.cparams.pooling_type = LLAMA_POOLING_TYPE_RANK;
                } else {
                    throw std::invalid_argument("invalid value");
                }
            });

            add_opt_with_hint(
                {"attention"},
                {"{causal,non-causal}"},
                R"(attention type for embeddings, use model default if unspecified)",
                [](llm_util_params& params, const std::string& value) {
                if (value == "causal") {
                    params.cparams.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
                } else if (value == "non-causal") {
                    params.cparams.attention_type
                        = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
                } else {
                    throw std::invalid_argument("invalid value");
                }
            });

            add_opt_with_hint(
                {"rope-scaling"},
                {"{none,linear,yarn}"},
                R"(RoPE frequency scaling method, defaults to linear unless specified by the model)",
                [](llm_util_params& params, const std::string& value) {
                if (value == "none") {
                    params.cparams.rope_params.rope_scaling_type
                        = LLAMA_ROPE_SCALING_TYPE_NONE;
                } else if (value == "linear") {
                    params.cparams.rope_params.rope_scaling_type
                        = LLAMA_ROPE_SCALING_TYPE_LINEAR;
                } else if (value == "yarn") {
                    params.cparams.rope_params.rope_scaling_type
                        = LLAMA_ROPE_SCALING_TYPE_YARN;
                } else {
                    throw std::invalid_argument("invalid value");
                }
            });

            add_opt_with_hint(
                {"rope-freq-base"},
                {"N"},
                R"(RoPE base frequency, used by NTK-aware scaling (default: loaded from model))",
                [](llm_util_params& params, const std::string& value) {
                params.cparams.rope_params.rope_freq_base = std::stof(value);
            });

            add_opt_with_hint(
                {"rope-scale"},
                {"N"},
                R"(RoPE context scaling factor, expands context by a factor of N)",
                [](llm_util_params& params, const std::string& value) {
                params.cparams.rope_params.rope_freq_scale
                    = 1.0F / std::stof(value);
            });

            add_opt_with_hint(
                {"rope-freq-scale"},
                {"N"},
                R"(RoPE frequency scaling factor, expands context by a factor of 1/N)",
                [](llm_util_params& params, const std::string& value) {
                params.cparams.rope_params.rope_freq_scale = std::stof(value);
            });

            add_opt_with_hint(
                {"yarn-ext-factor"},
                {"N"},
                std::format(
                    R"(YaRN: extrapolation mix factor (default: {}, 0.0 = full interpolation))",
                    static_cast<double>(
                        params.cparams.yarn_params.yarn_ext_factor)),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.yarn_params.yarn_ext_factor = std::stof(value);
            });

            add_opt_with_hint(
                {"yarn-attn-factor"},
                {"N"},
                std::format(
                    R"(YaRN: scale sqrt(t) or attention magnitude (default: {}))",
                    static_cast<double>(
                        params.cparams.yarn_params.yarn_attn_factor)),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.yarn_params.yarn_attn_factor = std::stof(value);
            });

            add_opt_with_hint(
                {"yarn-beta-slow"},
                {"N"},
                std::format(
                    R"(YaRN: high correction dim or alpha (default: {}))",
                    static_cast<double>(
                        params.cparams.yarn_params.yarn_beta_slow)),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.yarn_params.yarn_beta_slow = std::stof(value);
            });

            add_opt_with_hint(
                {"yarn-beta-fast"},
                {"N"},
                std::format(
                    R"(YaRN: low correction dim or beta (default: {}))",
                    static_cast<double>(
                        params.cparams.yarn_params.yarn_beta_fast)),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.yarn_params.yarn_beta_fast = std::stof(value);
            });

            add_opt_with_hint(
                {"yarn-orig-ctx"},
                {"N"},
                std::format(
                    R"(YaRN: original context size of model (default: {} = model training context size))",
                    params.cparams.yarn_params.yarn_orig_ctx),
                [](llm_util_params& params, const std::string& value) {
                std::int32_t n = std::stoi(value);
                if (n > 0) {
                    params.cparams.yarn_params.yarn_orig_ctx
                        = static_cast<std::uint32_t>(n);
                }
            });

            add_opt(
                {"libllama-verbose-perfomance"},
                std::format(
                    R"(disable internal libllama performance timings (default: {}))",
                    params.cparams.no_perf ? "true" : "false"),
                [](llm_util_params& params, const std::string& value) {
                bool enable = value == "true";
                params.cparams.no_perf = enable;
                params.sampling.no_perf = enable;
            });

            add_opt(
                {"op-offload"},
                std::format(
                    R"(disable offloading host tensor operations to device (default: {}))",
                    params.cparams.no_op_offload ? "true" : "false"),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.no_op_offload = (value == "true");
            });

            add_opt(
                {"swa-full"},
                std::format(
                    R"(use full-size SWA cache (default: {}))",
                    params.cparams.swa_full ? "true" : "false"),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.swa_full = (value == "true");
            });

            add_opt(
                {"kv-unified"},
                std::format(
                    R"(use single unified KV buffer for the KV cache of all sequences (default: {}))",
                    params.cparams.kv_unified ? "true" : "false"),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.kv_unified = (value == "true");
            });

            add_opt(
                {"kv-offload"},
                R"(disable KV offload)",
                [](llm_util_params& params, const std::string& value) {
                params.cparams.no_kv_offload = (value == "true");
            });

            add_opt(
                {"cache-type-k"},
                std::format(
                    R"(KV cache data type for K"
                                allowed values: {}"
                                default: {})",
                    get_all_kv_cache_types(),
                    ggml_type_name(params.cparams.cache_type_k)),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.cache_type_k = kv_cache_type_from_str(value);
            });

            add_opt(
                {"cache-type-v"},
                std::format(
                    R"(KV cache data type for K"
                                allowed values: {}"
                                default: {})",
                    get_all_kv_cache_types(),
                    ggml_type_name(params.cparams.cache_type_v)),
                [](llm_util_params& params, const std::string& value) {
                params.cparams.cache_type_v = kv_cache_type_from_str(value);
            });

            add_opt(
                {"no-mmap"},
                R"(do not memory-map model (slower load but may reduce pageouts if not using mlock))",
                [](llm_util_params& params, const std::string& value) {
                params.mparams.use_mmap = (value == "true");
            });

            add_opt(
                {"mlock"},
                R"(force system to keep model in RAM rather than swapping or compressing)",
                [](llm_util_params& params, const std::string& value) {
                params.mparams.use_mlock = (value == "true");
            });

            add_opt_with_hint(
                {"device"},
                {"<dev1,dev2,..>"},
                R"(comma-separated list of devices to use for offloading (none = don't offload))",
                [](llm_util_params& params, const std::string& value) {
                params.mparams.devices = parse_device_list(value);
            });

            add_opt(
                {"cpu-moe"},
                R"(keep all Mixture of Experts (MoE) weights in the CPU)",
                [](llm_util_params& params, const std::string& value) {
                if (value == "true") {
                    params.mparams.tensor_buft_overrides.push_back(
                        {.pattern = "\\.ffn_(up|down|gate)_exps",
                         .buft = ggml_backend_cpu_buffer_type()});
                } else {
                    params.mparams.tensor_buft_overrides.clear();
                }
            });

            add_opt_with_hint(
                {"n-cpu-moe"},
                {"N"},
                R"(keep the Mixture of Experts (MoE) weights of the first N layers in the CPU)",
                [](llm_util_params& params, const std::string& value) {
                int n_cpu_moe = std::stoi(value);
                if (n_cpu_moe < 0) {
                    throw std::invalid_argument("invalid value");
                }
                for (int i = 0; i < n_cpu_moe; ++i) {
                    // keep strings alive and avoid leaking memory by
                    // storing them in a static vector
                    static std::list<std::string> buft_overrides;
                    buft_overrides.push_back(
                        std::format(R"(blk\.{}\.ffn_(up|down|gate)_exps)", i));
                    params.mparams.tensor_buft_overrides.push_back(
                        {.pattern = buft_overrides.back().c_str(),
                         .buft = ggml_backend_cpu_buffer_type()});
                }
            });

            add_opt_with_hint(
                {"n-gpu-layers"},
                {"N"},
                std::format(
                    R"(max. number of layers to store in VRAM (default: {}))",
                    params.mparams.n_gpu_layers),
                [](llm_util_params& params, const std::string& value) {
                params.mparams.n_gpu_layers = std::stoi(value);
                if (!llama_supports_gpu_offload()) {
                    LOG_WRN(
                        "no usable GPU found, gpu-layers option will be "
                        "ignored. one possible reason is that llama.cpp "
                        "was "
                        "compiled without GPU support");
                }
            });

            add_opt_with_hint(
                {"split-mode"},
                {"{none,layer,row}"},
                R"(how to split the model across multiple GPUs, one of:"
                    - none: use one GPU only
                    - layer (default): split layers and KV across GPUs
                    - row: split rows across GPUs)",
                [](llm_util_params& params, const std::string& value) {
                if (const std::string& arg_next = value; arg_next == "none") {
                    params.mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
                } else if (arg_next == "layer") {
                    params.mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
                } else if (arg_next == "row") {
                    params.mparams.split_mode = LLAMA_SPLIT_MODE_ROW;
                } else {
                    throw std::invalid_argument("invalid value");
                }
                if (!llama_supports_gpu_offload()) {
                    LOG_WRN(
                        "llama.cpp was compiled without support for GPU "
                        "offload. Setting the split mode has no effect");
                }
            });

            const auto generate_tensor_split
                = [](const std::vector<std::string>& split_arg) {
                std::vector<float> tensor_split(128, 0);
                const std::size_t end
                    = std::min(llama_max_devices(), split_arg.size());
                for (std::size_t i = 0; i < end; ++i) {
                    tensor_split[i] = std::stof(split_arg[i]);
                }

                return tensor_split;
            };

            const auto verify_gpu_offload_support = []() {
                bool ret = llama_supports_gpu_offload();
                if (!ret) {
                    LOG_WRN(
                        "llama.cpp was compiled without support for GPU "
                        "offload. Setting a tensor split has no effect.");
                }

                return ret;
            };

            add_opt_with_hint(
                {"tensor-split"},
                {"N0,N1,N2,..."},
                R"(fraction of the model to offload to each GPU, comma-separated list of proportions (e.g. 3,1))",
                [&verify_gpu_offload_support, &generate_tensor_split](
                    llm_util_params& params, const std::string& value) {
                if (!verify_gpu_offload_support()) {
                    return;
                }

                const std::string& arg_next = value;
                // split string by , and /
                const std::regex regex{"([,/]+)"};
                std::sregex_token_iterator itr{
                    arg_next.begin(), arg_next.end(), regex, -1};
                std::vector<std::string> split_arg{itr, {}};
                if (split_arg.size() >= llama_max_devices()) {
                    throw std::invalid_argument(
                        std::format(
                            R"(got {} input configs, but system only has {} devices)",
                            static_cast<int>(split_arg.size()),
                            static_cast<int>(llama_max_devices())));
                }

                params.mparams.tensor_split = generate_tensor_split(split_arg);
            });

            add_opt_with_hint(
                {"main-gpu"},
                {"N"},
                std::format(
                    R"(the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: {}))",
                    params.mparams.main_gpu),
                [](llm_util_params& params, const std::string& value) {
                int main_gpu = std::stoi(value);
                if (main_gpu < 0) {
                    throw std::invalid_argument(
                        std::format(
                            R"(got gpu id of {}, which is invalid)", main_gpu));
                }
                params.mparams.main_gpu = main_gpu;
                if (!llama_supports_gpu_offload()) {
                    LOG_WRN(
                        "llama.cpp was compiled without support for GPU "
                        "offload. Setting the main GPU has no effect)");
                }
            });

            add_opt(
                {"check-tensors"},
                std::format(
                    R"(check model tensor data for invalid values (default: {}))",
                    params.mparams.check_tensors ? "true" : "false"),
                [](llm_util_params& params, const std::string& value) {
                params.mparams.check_tensors = (value == "true");
            });

            add_opt(
                {"no-repack"},
                R"(disable weight repacking)",
                [](llm_util_params& params, const std::string& value) {
                params.mparams.no_extra_bufts = (value == "true");
            });

            add_opt_with_hint(
                {"override-kv"},
                {"key=type:value"},
                R"(advanced option to override model metadata by key. may be specified multiple times.
                    allowed types:
                        int, float, bool, str
                    example:
                        tokenizer.ggml.add_bos_token=bool:false)",
                [](llm_util_params& params, const std::string& value) {
                if (!llm_util_string_parse_kv_override(
                        value, params.mparams.kv_overrides)) {
                    throw std::runtime_error(
                        std::format(
                            R"(error: Invalid type for KV override: {})",
                            value));
                }
            });

            add_opt_with_hint(
                {"override-tensor"},
                {"<tensor name pattern>=<buffer type>,..."},
                R"(override tensor buffer type)",
                [](llm_util_params& params, const std::string& value) {
                parse_tensor_buffer_overrides(value, params.mparams);
            });

            add_opt_with_hint(
                {"cls-separator"},
                {"string"},
                R"(separator of classification sequences (default: \t). example: "<#seq#>")",
                [](llm_util_params& params, const std::string& value) {
                params.cls_sep = value;
            });

            add_opt_with_hint(
                {"chat-template"},
                {"string"},
                std::format(R"(overrides default chat template)"),
                [](llm_util_params& params, const std::string& value) {
                params.chat_template = value;
            });

            add_opt_with_hint(
                {"chat-template-kwargs"},
                {"string"},
                std::format(
                    R"(sets additional params for the json template parser)"),
                [](llm_util_params& params, const std::string& value) {
                auto parsed = nlohmann::ordered_json::parse(value);
                for (const auto& item: parsed.items()) {
                    params.chat_template_kwargs[item.key()]
                        = item.value().dump();
                }
            });

            add_opt_with_hint(
                {"verbose-parse-result"},
                {"string"},
                std::format(R"(prints parse output to console)"),
                [](llm_util_params& params, const std::string& value) {
                params.verbose_parse_result = (value == "true");
            });

            add_opt_with_hint(
                {"verbosity"},
                {"N"},
                R"(set the verbosity level for logging (default: info), where higher verbosity levels prints all levels above it and ignores all levels below it
                    - 0: no logging
                    - 1: debug
                    - 2: info
                    - 3: warnings
                    - 4: errors)",
                [](llm_util_params&, const std::string& value) {
                std::int32_t verbosity = std::stoi(value);
                // Clamp
                if (verbosity <= 0 || verbosity > 4) {
                    verbosity = 0;
                }
                util::log::log_set_level(
                    util::log::get_log_main(), util::log::log_level(verbosity));
            });
        }

        inline llm_util_params generate_llm_util_params_impl(
            const std::unordered_map<std::string, std::string>& args,
            const llm_util_params& initial_params,
            const llm_util_params_context& ctx_arg)
        {
            llm_util_params params = initial_params;

            std::unordered_map<std::string, const Arg*> arg_to_options;
            for (const auto& opt: ctx_arg.options) {
                for (const auto& arg: opt.args_list) {
                    arg_to_options[arg] = &opt;
                }
            }

            for (const auto& [arg, hints]: args) {
                auto itr = arg_to_options.find(arg);
                if (itr == arg_to_options.end()) {
                    throw std::invalid_argument(
                        std::format(R"(error: invalid argument: {})", arg));
                }

                try {
                    const Arg* opt = itr->second;
                    opt->arg_handler(params, hints);
                } catch (std::exception& e) {
                    throw std::invalid_argument(
                        std::format(
                            R"(error while handling argument "{}": {})",
                            arg,
                            e.what()));
                }
            }

            postprocess_cpu_params(params.cpuparams, nullptr);
            postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

            if (!params.mparams.kv_overrides.empty()) {
                params.mparams.kv_overrides.emplace_back();
                params.mparams.kv_overrides.back().key[0] = 0;
            }

            if (!params.mparams.tensor_buft_overrides.empty()) {
                params.mparams.tensor_buft_overrides.push_back(
                    {.pattern = nullptr, .buft = nullptr});
            }

            return params;
        }
    } // namespace

    std::optional<llm_util_params> generate_llm_util_params(
        const std::unordered_map<std::string, std::string>& params,
        const llm_util_params& param_hints)
    {
        try {
            llm_util_params_context ctx_args(param_hints);
            return generate_llm_util_params_impl(params, param_hints, ctx_args);
        } catch (std::exception& e) {
            LOG_ERR(
                "%s: %s",
                std::source_location::current().function_name(),
                e.what());
            return std::nullopt;
        }
    }

    nlohmann::json llm_param_usage_to_json(const std::vector<Arg>& args)
    {
        nlohmann::json j;
        for (const Arg& opt: args) {
            if (opt.is_sparam) {
                j["common-params"].push_back(opt.to_json());
            } else {
                j["sampling-params"].push_back(opt.to_json());
            }
        }

        return j;
    }

    nlohmann::json llm_builtin_chat_templates_to_json()
    {
        std::vector<const char*> supported_templates;
        int32_t prompt_len = llama_chat_builtin_templates(nullptr, 0);
        supported_templates.resize(static_cast<std::size_t>(prompt_len));
        llama_chat_builtin_templates(
            supported_templates.data(), supported_templates.size());
        nlohmann::json j;
        for (const char* value: supported_templates) {
            j["templates"].push_back(std::string(value));
        }
        return j;
    }
} // namespace util::file
