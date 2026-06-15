#include <algorithm>
#include <atomic>
#include <gmp.h>
#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr unsigned long long kC3Over24 = 10939058860032000ULL;
constexpr const char* kPiPrefixReference = "314159265358979323846264338327950288419716939937510";

struct Triple {
    mpz_t p;
    mpz_t q;
    mpz_t t;

    Triple() {
        mpz_init(p);
        mpz_init(q);
        mpz_init(t);
    }

    ~Triple() {
        mpz_clear(p);
        mpz_clear(q);
        mpz_clear(t);
    }

    Triple(const Triple&) = delete;
    Triple& operator=(const Triple&) = delete;

    Triple(Triple&& other) noexcept : Triple() {
        mpz_swap(p, other.p);
        mpz_swap(q, other.q);
        mpz_swap(t, other.t);
    }

    Triple& operator=(Triple&& other) noexcept {
        if (this != &other) {
            mpz_swap(p, other.p);
            mpz_swap(q, other.q);
            mpz_swap(t, other.t);
        }
        return *this;
    }
};

struct Options {
    std::uint64_t digits = 0;
    int threads = 1;
    std::uint64_t chunk_terms = 8192;
    std::uint64_t leaf_terms = 8;
    std::uint64_t task_terms = 0;
    std::string parallel_mode = "chunked";
    unsigned long guard_digits = 48;
    std::string output_path;
};

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                die(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--digits") {
            options.digits = std::strtoull(require_value("--digits"), nullptr, 10);
        } else if (arg == "--threads") {
            options.threads = std::max(1, std::atoi(require_value("--threads")));
        } else if (arg == "--chunk-terms") {
            options.chunk_terms = std::strtoull(require_value("--chunk-terms"), nullptr, 10);
        } else if (arg == "--leaf-terms") {
            options.leaf_terms = std::strtoull(require_value("--leaf-terms"), nullptr, 10);
        } else if (arg == "--task-terms") {
            options.task_terms = std::strtoull(require_value("--task-terms"), nullptr, 10);
        } else if (arg == "--parallel-mode") {
            options.parallel_mode = require_value("--parallel-mode");
        } else if (arg == "--guard-digits") {
            options.guard_digits = static_cast<unsigned long>(std::strtoul(require_value("--guard-digits"), nullptr, 10));
        } else if (arg == "--output") {
            options.output_path = require_value("--output");
        } else {
            die(std::string("unknown argument: ") + arg);
        }
    }

    if (options.digits == 0) {
        die("--digits must be provided and positive");
    }
    if (options.chunk_terms == 0) {
        die("--chunk-terms must be positive");
    }
    if (options.leaf_terms == 0) {
        die("--leaf-terms must be positive");
    }
    if (options.task_terms == 0) {
        options.task_terms = options.chunk_terms;
    }
    if (
        options.parallel_mode != "chunked"
        && options.parallel_mode != "tasks"
        && options.parallel_mode != "frontier"
    ) {
        die("--parallel-mode must be one of: chunked, tasks, frontier");
    }
    return options;
}

void chudnovsky_single_term(std::uint64_t a, mpz_t p, mpz_t q, mpz_t t) {
    if (a == 0) {
        mpz_set_ui(p, 1);
        mpz_set_ui(q, 1);
        mpz_set_ui(t, 13591409);
        return;
    }

    mpz_set_ui(p, 6 * a - 5);
    mpz_mul_ui(p, p, 2 * a - 1);
    mpz_mul_ui(p, p, 6 * a - 1);

    mpz_set_ui(q, static_cast<unsigned long>(a));
    mpz_mul_ui(q, q, static_cast<unsigned long>(a));
    mpz_mul_ui(q, q, static_cast<unsigned long>(a));
    mpz_mul_ui(q, q, kC3Over24);

    const unsigned long long linear = 13591409ULL + 545140134ULL * a;
    mpz_mul_ui(t, p, linear);
    if (a & 1ULL) {
        mpz_neg(t, t);
    }
}

Triple chudnovsky_bs_block(std::uint64_t a, std::uint64_t b) {
    Triple result;
    mpz_set_ui(result.p, 1);
    mpz_set_ui(result.q, 1);
    mpz_set_ui(result.t, 0);

    mpz_t p_term;
    mpz_t q_term;
    mpz_t t_term;
    mpz_inits(p_term, q_term, t_term, nullptr);

    for (std::uint64_t index = a; index < b; ++index) {
        chudnovsky_single_term(index, p_term, q_term, t_term);
        mpz_mul(result.t, result.t, q_term);
        mpz_addmul(result.t, result.p, t_term);
        mpz_mul(result.p, result.p, p_term);
        mpz_mul(result.q, result.q, q_term);
    }

    mpz_clears(p_term, q_term, t_term, nullptr);
    return result;
}

Triple chudnovsky_bs_range(std::uint64_t a, std::uint64_t b, std::uint64_t leaf_terms) {
    if (b - a <= leaf_terms) {
        return chudnovsky_bs_block(a, b);
    }

    const std::uint64_t m = (a + b) / 2;
    Triple left = chudnovsky_bs_range(a, m, leaf_terms);
    Triple right = chudnovsky_bs_range(m, b, leaf_terms);

    mpz_mul(left.t, left.t, right.q);
    mpz_addmul(left.t, left.p, right.t);
    mpz_mul(left.p, left.p, right.p);
    mpz_mul(left.q, left.q, right.q);
    return left;
}

void merge_triples_inplace(Triple* left, Triple&& right) {
    mpz_mul(left->t, left->t, right.q);
    mpz_addmul(left->t, left->p, right.t);
    mpz_mul(left->p, left->p, right.p);
    mpz_mul(left->q, left->q, right.q);
}

Triple chudnovsky_parallel(std::uint64_t terms, int threads, std::uint64_t chunk_terms, std::uint64_t leaf_terms) {
    if (terms <= chunk_terms || threads <= 1) {
        return chudnovsky_bs_range(0, terms, leaf_terms);
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (std::uint64_t start = 0; start < terms; start += chunk_terms) {
        const std::uint64_t end = std::min<std::uint64_t>(start + chunk_terms, terms);
        ranges.emplace_back(start, end);
    }

    std::vector<Triple> partials(ranges.size());
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        partials[i] = chudnovsky_bs_range(ranges[i].first, ranges[i].second, leaf_terms);
    }

    while (partials.size() > 1) {
        const std::size_t merged_size = (partials.size() + 1) / 2;
        std::vector<Triple> merged(merged_size);
        #pragma omp parallel for schedule(static) num_threads(std::min<int>(threads, static_cast<int>(merged_size)))
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(merged_size); ++i) {
            const std::size_t left_index = static_cast<std::size_t>(2 * i);
            const std::size_t right_index = left_index + 1;
            if (right_index >= partials.size()) {
                merged[static_cast<std::size_t>(i)] = std::move(partials[left_index]);
            } else {
                merge_triples_inplace(&partials[left_index], std::move(partials[right_index]));
                merged[static_cast<std::size_t>(i)] = std::move(partials[left_index]);
            }
        }
        partials = std::move(merged);
    }
    return std::move(partials.front());
}

Triple chudnovsky_task_tree(
    std::uint64_t a,
    std::uint64_t b,
    std::uint64_t leaf_terms,
    std::uint64_t task_terms
) {
    if (b - a <= leaf_terms) {
        return chudnovsky_bs_block(a, b);
    }
    if (b - a <= task_terms) {
        return chudnovsky_bs_range(a, b, leaf_terms);
    }

    const std::uint64_t m = (a + b) / 2;
    Triple left;
    Triple right;

    #pragma omp task shared(left) if ((m - a) > task_terms)
    {
        left = chudnovsky_task_tree(a, m, leaf_terms, task_terms);
    }
    #pragma omp task shared(right) if ((b - m) > task_terms)
    {
        right = chudnovsky_task_tree(m, b, leaf_terms, task_terms);
    }
    #pragma omp taskwait

    merge_triples_inplace(&left, std::move(right));
    return left;
}

Triple chudnovsky_parallel_tasks(
    std::uint64_t terms,
    int threads,
    std::uint64_t leaf_terms,
    std::uint64_t task_terms
) {
    if (threads <= 1 || terms <= task_terms) {
        return chudnovsky_bs_range(0, terms, leaf_terms);
    }

    Triple result;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single nowait
        {
            result = chudnovsky_task_tree(0, terms, leaf_terms, task_terms);
        }
    }
    return result;
}

struct FrontierSlot {
    bool occupied = false;
    Triple value;
};

void frontier_push(std::vector<FrontierSlot>* frontier, Triple&& carry) {
    std::size_t level = 0;
    while (true) {
        if (level == frontier->size()) {
            frontier->emplace_back();
        }
        FrontierSlot& slot = (*frontier)[level];
        if (!slot.occupied) {
            slot.value = std::move(carry);
            slot.occupied = true;
            break;
        }
        merge_triples_inplace(&slot.value, std::move(carry));
        carry = std::move(slot.value);
        slot.occupied = false;
        ++level;
    }
}

Triple frontier_reduce(std::vector<FrontierSlot>* frontier) {
    Triple result;
    bool has_result = false;
    for (auto it = frontier->rbegin(); it != frontier->rend(); ++it) {
        if (!it->occupied) {
            continue;
        }
        if (!has_result) {
            result = std::move(it->value);
            has_result = true;
            continue;
        }
        merge_triples_inplace(&result, std::move(it->value));
    }
    if (!has_result) {
        die("frontier_reduce requires at least one partial");
    }
    return result;
}

Triple frontier_consume_ordered(
    std::vector<Triple>* slots,
    std::vector<std::atomic<int>>* ready
) {
    std::vector<FrontierSlot> frontier;
    frontier.reserve(8);

    for (std::size_t index = 0; index < slots->size(); ++index) {
        while ((*ready)[index].load(std::memory_order_acquire) == 0) {
            #pragma omp taskyield
        }
        frontier_push(&frontier, std::move((*slots)[index]));
    }

    return frontier_reduce(&frontier);
}

Triple chudnovsky_parallel_frontier(
    std::uint64_t terms,
    int threads,
    std::uint64_t chunk_terms,
    std::uint64_t leaf_terms
) {
    if (terms <= chunk_terms || threads <= 1) {
        return chudnovsky_bs_range(0, terms, leaf_terms);
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (std::uint64_t start = 0; start < terms; start += chunk_terms) {
        const std::uint64_t end = std::min<std::uint64_t>(start + chunk_terms, terms);
        ranges.emplace_back(start, end);
    }

    std::vector<Triple> slots(ranges.size());
    std::vector<std::atomic<int>> ready(ranges.size());
    for (std::atomic<int>& flag : ready) {
        flag.store(0, std::memory_order_relaxed);
    }

    Triple result;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        {
            for (std::size_t index = 0; index < ranges.size(); ++index) {
                #pragma omp task firstprivate(index)
                {
                    slots[index] = chudnovsky_bs_range(
                        ranges[index].first,
                        ranges[index].second,
                        leaf_terms
                    );
                    ready[index].store(1, std::memory_order_release);
                }
            }
            result = frontier_consume_ordered(&slots, &ready);
            #pragma omp taskwait
        }
    }
    return result;
}

void format_prefix(mpz_t pi_digits, std::uint64_t digits, std::string* prefix_out, bool* prefix_ok_out) {
    constexpr std::size_t prefix_len = std::char_traits<char>::length(kPiPrefixReference);
    mpz_t leading;
    mpz_init(leading);

    if (digits + 1 > prefix_len) {
        mpz_t divisor;
        mpz_init(divisor);
        mpz_ui_pow_ui(divisor, 10, static_cast<unsigned long>(digits + 1 - prefix_len));
        mpz_tdiv_q(leading, pi_digits, divisor);
        mpz_clear(divisor);
    } else {
        mpz_set(leading, pi_digits);
    }

    char* raw = mpz_get_str(nullptr, 10, leading);
    *prefix_out = raw;
    *prefix_ok_out = (*prefix_out == kPiPrefixReference);
    void (*freefunc)(void*, size_t);
    mp_get_memory_functions(nullptr, nullptr, &freefunc);
    freefunc(raw, std::strlen(raw) + 1);
    mpz_clear(leading);
}

void write_decimal_file(mpz_t pi_digits, const std::string& output_path) {
    char* raw = mpz_get_str(nullptr, 10, pi_digits);
    const std::size_t raw_len = std::strlen(raw);

    FILE* out = std::fopen(output_path.c_str(), "wb");
    if (!out) {
        void (*freefunc)(void*, size_t);
        mp_get_memory_functions(nullptr, nullptr, &freefunc);
        freefunc(raw, raw_len + 1);
        die("failed to open output file");
    }

    std::fwrite(raw, 1, 1, out);
    std::fwrite(".", 1, 1, out);
    if (raw_len > 1) {
        std::fwrite(raw + 1, 1, raw_len - 1, out);
    }
    std::fwrite("\n", 1, 1, out);
    std::fclose(out);

    void (*freefunc)(void*, size_t);
    mp_get_memory_functions(nullptr, nullptr, &freefunc);
    freefunc(raw, raw_len + 1);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::uint64_t total_digits = options.digits + options.guard_digits;
        const std::uint64_t terms = total_digits / 14 + 1;

        auto start = std::chrono::steady_clock::now();
        Triple sum;
        if (options.parallel_mode == "tasks") {
            sum = chudnovsky_parallel_tasks(terms, options.threads, options.leaf_terms, options.task_terms);
        } else if (options.parallel_mode == "frontier") {
            sum = chudnovsky_parallel_frontier(terms, options.threads, options.chunk_terms, options.leaf_terms);
        } else {
            sum = chudnovsky_parallel(terms, options.threads, options.chunk_terms, options.leaf_terms);
        }

        mpz_t scale;
        mpz_t sqrt_input;
        mpz_t sqrt_term;
        mpz_t pi_scaled;
        mpz_t guard_divisor;
        mpz_t pi_digits;
        mpz_inits(scale, sqrt_input, sqrt_term, pi_scaled, guard_divisor, pi_digits, nullptr);

        mpz_ui_pow_ui(scale, 10, total_digits);
        mpz_mul(sqrt_input, scale, scale);
        mpz_mul_ui(sqrt_input, sqrt_input, 10005);
        mpz_sqrt(sqrt_term, sqrt_input);

        mpz_mul_ui(pi_scaled, sqrt_term, 426880);
        mpz_mul(pi_scaled, pi_scaled, sum.q);
        mpz_tdiv_q(pi_scaled, pi_scaled, sum.t);

        mpz_ui_pow_ui(guard_divisor, 10, options.guard_digits);
        mpz_tdiv_q(pi_digits, pi_scaled, guard_divisor);

        std::string prefix;
        bool prefix_ok = false;
        format_prefix(pi_digits, options.digits, &prefix, &prefix_ok);

        if (!options.output_path.empty()) {
            write_decimal_file(pi_digits, options.output_path);
        }

        auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        const double digits_per_second = static_cast<double>(options.digits) / seconds;

        std::cout
            << "digits=" << options.digits
            << " terms=" << terms
            << " seconds=" << seconds
            << " digits_per_second=" << digits_per_second
            << " threads=" << options.threads
            << " chunk_terms=" << options.chunk_terms
            << " leaf_terms=" << options.leaf_terms
            << " task_terms=" << options.task_terms
            << " parallel_mode=" << options.parallel_mode
            << " prefix_ok=" << (prefix_ok ? "True" : "False")
            << " prefix=" << prefix
            << "\n";

        mpz_clears(scale, sqrt_input, sqrt_term, pi_scaled, guard_divisor, pi_digits, nullptr);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
