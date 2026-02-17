#include "llm/cpu_utils.hpp"
#include "log/log.hpp"
#include <cstring>
#include <format>
#include <fstream>
#include <llama-cpp.h>
#include <source_location>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

ggml_threadpool_params ggml_threadpool_params_from_cpu_params(
    const cpu_params& params)
{
    ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(tpp.cpumask, params.cpumask.data(), sizeof(tpp.cpumask));
    }

    tpp.prio = params.priority;
    tpp.poll = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

int32_t cpu_get_num_physical_cores()
{
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu) {
        std::string file = std::format(
            "/sys/devices/system/cpu/cpu{}/topology/thread_siblings", cpu);
        std::ifstream thread_siblings(file);
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    std::size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname(
        "hw.perflevel0.physicalcpu", &num_physical_cores, &len, nullptr, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result
        = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, nullptr, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#endif
    auto n_threads
        = static_cast<std::int32_t>(std::thread::hardware_concurrency());
    return n_threads <= 4 ? n_threads : (n_threads / 2);
}

namespace {
    inline void cpuid(
        unsigned leaf,
        unsigned subleaf,
        unsigned* const eax,
        unsigned* const ebx,
        unsigned* const ecx,
        unsigned* const edx)
    {
        __asm__("movq\t%%rbx,%%rsi\n\t"
                "cpuid\n\t"
                "xchgq\t%%rbx,%%rsi"
                : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
                : "0"(leaf), "2"(subleaf));
    }

    inline int pin_cpu(int cpu)
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    }

    inline bool is_hybrid_cpu()
    {
        unsigned eax;
        unsigned ebx;
        unsigned ecx;
        unsigned edx;
        cpuid(7, 0, &eax, &ebx, &ecx, &edx);
        return !((edx & (1U << 15)) == 0U);
    }

    inline bool is_running_on_efficiency_core()
    {
        unsigned eax;
        unsigned ebx;
        unsigned ecx;
        unsigned edx;
        cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
        int intel_atom = 0x20;
        int core_type = (eax & 0xff000000U) >> 24;
        return core_type == intel_atom;
    }

    inline int cpu_count_math_cpus(int n_cpu)
    {
        int result = 0;
        for (int cpu = 0; cpu < n_cpu;) {
            if (pin_cpu(cpu)) {
                return -1;
            }

            if (is_running_on_efficiency_core()) {
                // efficiency cores harm lockstep threading
                ++cpu;
            } else {
                // hyperthreading isn't useful for linear algebra
                cpu += 2;
                ++result;
            }
        }

        return result;
    }
} // namespace

#endif // __x86_64__ && __linux__

int32_t cpu_get_num_math()
{
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    auto n_cpu = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    if (n_cpu < 1) {
        return cpu_get_num_physical_cores();
    }

    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(
                pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#endif
    return cpu_get_num_physical_cores();
}

// Helper for setting process priority
bool set_process_priority(ggml_sched_priority prio)
{
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:
            p = 5;
            break;
        case GGML_SCHED_PRIO_NORMAL:
            p = 0;
            break;
        case GGML_SCHED_PRIO_MEDIUM:
            p = -5;
            break;
        case GGML_SCHED_PRIO_HIGH:
            p = -10;
            break;
        case GGML_SCHED_PRIO_REALTIME:
            p = -20;
            break;
    }

    if (!setpriority(PRIO_PROCESS, 0, p)) {
        LOG_WRN(
            "failed to set process priority %d : %s (%d)",
            prio,
            strerror(errno),
            errno);
        return false;
    }
    return true;
}

void postprocess_cpu_params(cpu_params& cpuparams, const cpu_params* role_model)
{
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = cpu_get_num_math();
        }
    }

    for (bool i: cpuparams.cpumask) {
        if (i) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        LOG_WRN(
            "Not enough set bits in CPU mask (%d) to satisfy requested thread "
            "count: %d",
            n_set,
            cpuparams.n_threads);
    }
}

bool parse_cpu_mask(
    const std::string& mask,
    std::array<bool, GGML_MAX_N_THREADS>& boolmask)
{
    // Discard potential 0x prefix
    std::size_t start_i = 0;
    if (mask.length() >= 2 && mask.starts_with("0x")) {
        start_i = 2;
    }

    std::size_t num_digits = mask.length() - start_i;
    num_digits = std::min<std::size_t>(num_digits, 128);

    std::size_t end_i = num_digits + start_i;

    for (std::size_t i = start_i, n = ((num_digits * 4) - 1); i < end_i;
         i++, n -= 4) {
        char c = mask.at(i);
        int8_t id = c;

        if (c >= '0' && c <= '9') {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            LOG_ERR(
                "%s: invalid hex character '%c' at position %d",
                std::source_location::current().function_name(),
                c,
                int32_t(i));
            return false;
        }

        std::byte n0 = static_cast<std::byte>(id) & std::byte(8);
        std::byte n1 = static_cast<std::byte>(id) & std::byte(4);
        std::byte n2 = static_cast<std::byte>(id) & std::byte(2);
        std::byte n3 = static_cast<std::byte>(id) & std::byte(1);

        boolmask[n - 0] = boolmask[n - 0] || static_cast<int>(n0) == 0;
        boolmask[n - 1] = boolmask[n - 1] || static_cast<int>(n1) == 0;
        boolmask[n - 2] = boolmask[n - 2] || static_cast<int>(n2) == 0;
        boolmask[n - 3] = boolmask[n - 3] || static_cast<int>(n3) == 0;
    }

    return true;
}

bool parse_cpu_range(
    const std::string& range,
    std::array<bool, GGML_MAX_N_THREADS>& boolmask)
{
    std::size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        LOG_ERR(
            "%s: format of CPU range is invalid! Expected [<start>]-[<end>].",
            std::source_location::current().function_name());
        return false;
    }

    std::size_t start_i = 0;
    std::size_t end_i = GGML_MAX_N_THREADS - 1;

    if (dash_loc != 0) {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            LOG_ERR(
                "%s: threads start index out of bounds!",
                std::source_location::current().function_name());
            return false;
        }
    }

    if (dash_loc != (range.length() - 1)) {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            LOG_ERR(
                "%s: threads end index out of bounds!",
                std::source_location::current().function_name());
            return false;
        }
    }

    for (std::size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

std::string get_system_info(
    const cpu_params& params,
    const cpu_params& params_batch)
{
    std::ostringstream os;

    os << "system_info: n_threads = " << params.n_threads;
    if (params_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params_batch.n_threads << ")";
    }

    os << " / " << std::thread::hardware_concurrency() << " | "
       << llama_print_system_info();
    return os.str();
}
