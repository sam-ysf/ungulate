#pragma once

#include <array>
#include <cstdint>
#include <ggml-opt.h>
#include <string>

struct cpu_params {
    std::int32_t n_threads = -1;

    // CPU affinity mask.
    std::array<bool, GGML_MAX_N_THREADS> cpumask = {};

    // Default: any CPU
    bool mask_valid = false;

    // Scheduling prio :
    // 0 - normal
    // 1 - medium,
    // 2 - high
    // 3 - realtime
    ggml_sched_priority priority = GGML_SCHED_PRIO_NORMAL;

    // Use strict CPU placement
    bool strict_cpu = false;

    // Polling (busywait) level (0 - no polling, 100 - mostly polling)
    std::uint32_t poll = 50;
};

ggml_threadpool_params ggml_threadpool_params_from_cpu_params(
    const cpu_params& params);

//! Returns number of physical cpus on system
int32_t cpu_get_num_physical_cores();

//! Returns number of cpus on system that are useful for math.
int32_t cpu_get_num_math();

bool set_process_priority(ggml_sched_priority prio);

void postprocess_cpu_params(
    cpu_params& cpuparams,
    const cpu_params* role_model);

bool parse_cpu_mask(
    const std::string& mask,
    std::array<bool, GGML_MAX_N_THREADS>& boolmask);

bool parse_cpu_range(
    const std::string& range,
    std::array<bool, GGML_MAX_N_THREADS>& boolmask);

std::string get_system_info(
    const cpu_params& params,
    const cpu_params& params_batch);
