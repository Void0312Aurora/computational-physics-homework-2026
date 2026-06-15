#include <algorithm>
#include <cmath>
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

struct Options {
    std::uint64_t digits = 0;
    int threads = 1;
    std::uint64_t chunk_terms = 8192;
    std::uint64_t leaf_terms = 8;
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
    return options;
}

mp_bitcnt_t estimate_triple_bits(std::uint64_t max_term_index, std::uint64_t range_terms) {
    const long double safe_index = static_cast<long double>(std::max<std::uint64_t>(1, max_term_index));
    const long double safe_range = static_cast<long double>(std::max<std::uint64_t>(1, range_terms));
    const long double q_term_bits = std::log2(static_cast<long double>(kC3Over24))
        + 3.0L * std::log2(6.0L * safe_index + 8.0L)
        + 40.0L;
    const long double total_bits = safe_range * q_term_bits + 256.0L;
    return static_cast<mp_bitcnt_t>(std::ceil(total_bits));
}

struct ReservedTriple {
    mpz_t p;
    mpz_t q;
    mpz_t t;

    ReservedTriple() {
        mpz_init(p);
        mpz_init(q);
        mpz_init(t);
    }

    explicit ReservedTriple(mp_bitcnt_t reserve_bits) {
        mpz_init2(p, reserve_bits);
        mpz_init2(q, reserve_bits);
        mpz_init2(t, reserve_bits);
    }

    ~ReservedTriple() {
        mpz_clear(p);
        mpz_clear(q);
        mpz_clear(t);
    }

    ReservedTriple(const ReservedTriple&) = delete;
    ReservedTriple& operator=(const ReservedTriple&) = delete;

    ReservedTriple(ReservedTriple&& other) noexcept : ReservedTriple() {
        mpz_swap(p, other.p);
        mpz_swap(q, other.q);
        mpz_swap(t, other.t);
    }

    ReservedTriple& operator=(ReservedTriple&& other) noexcept {
        if (this != &other) {
            mpz_swap(p, other.p);
            mpz_swap(q, other.q);
            mpz_swap(t, other.t);
        }
        return *this;
    }

    void reserve(mp_bitcnt_t reserve_bits) {
        mpz_realloc2(p, reserve_bits);
        mpz_realloc2(q, reserve_bits);
        mpz_realloc2(t, reserve_bits);
    }

    void steal_from(ReservedTriple* other) {
        mpz_swap(p, other->p);
        mpz_swap(q, other->q);
        mpz_swap(t, other->t);
    }
};

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

void chudnovsky_block_into(ReservedTriple* out, std::uint64_t a, std::uint64_t b) {
    mpz_set_ui(out->p, 1);
    mpz_set_ui(out->q, 1);
    mpz_set_ui(out->t, 0);

    mpz_t p_term;
    mpz_t q_term;
    mpz_t t_term;
    mpz_inits(p_term, q_term, t_term, nullptr);

    for (std::uint64_t index = a; index < b; ++index) {
        chudnovsky_single_term(index, p_term, q_term, t_term);
        mpz_mul(out->t, out->t, q_term);
        mpz_addmul(out->t, out->p, t_term);
        mpz_mul(out->p, out->p, p_term);
        mpz_mul(out->q, out->q, q_term);
    }

    mpz_clears(p_term, q_term, t_term, nullptr);
}

void merge_triples_inplace(ReservedTriple* left, ReservedTriple* right) {
    mpz_mul(left->t, left->t, right->q);
    mpz_addmul(left->t, left->p, right->t);
    mpz_mul(left->p, left->p, right->p);
    mpz_mul(left->q, left->q, right->q);
}

struct ChunkScratch {
    explicit ChunkScratch(std::uint64_t chunk_terms, std::uint64_t max_term_index) {
        std::uint64_t current_range = std::max<std::uint64_t>(1, (chunk_terms + 1) / 2);
        while (true) {
            levels.emplace_back(estimate_triple_bits(max_term_index, current_range));
            if (current_range <= 1) {
                break;
            }
            current_range = std::max<std::uint64_t>(1, (current_range + 1) / 2);
        }
    }

    ReservedTriple& at(std::size_t depth, std::uint64_t range_terms, std::uint64_t max_term_index) {
        if (depth >= levels.size()) {
            levels.emplace_back(estimate_triple_bits(max_term_index, range_terms));
        } else {
            levels[depth].reserve(estimate_triple_bits(max_term_index, range_terms));
        }
        return levels[depth];
    }

    std::vector<ReservedTriple> levels;
};

void chudnovsky_range_into(
    ReservedTriple* out,
    std::uint64_t a,
    std::uint64_t b,
    std::uint64_t leaf_terms,
    ChunkScratch* scratch,
    std::size_t depth
) {
    if (b - a <= leaf_terms) {
        chudnovsky_block_into(out, a, b);
        return;
    }

    const std::uint64_t m = (a + b) / 2;
    chudnovsky_range_into(out, a, m, leaf_terms, scratch, depth + 1);

    ReservedTriple& right = scratch->at(depth, b - m, b);
    chudnovsky_range_into(&right, m, b, leaf_terms, scratch, depth + 1);
    merge_triples_inplace(out, &right);
}

std::vector<ReservedTriple> make_reserved_level(
    std::size_t count,
    std::uint64_t span_terms,
    std::uint64_t max_term_index
) {
    std::vector<ReservedTriple> nodes;
    nodes.reserve(count);
    const mp_bitcnt_t reserve_bits = estimate_triple_bits(max_term_index, span_terms);
    for (std::size_t i = 0; i < count; ++i) {
        nodes.emplace_back(reserve_bits);
    }
    return nodes;
}

ReservedTriple chudnovsky_parallel_level_pool(
    std::uint64_t terms,
    int threads,
    std::uint64_t chunk_terms,
    std::uint64_t leaf_terms
) {
    if (terms <= chunk_terms || threads <= 1) {
        ReservedTriple result(estimate_triple_bits(terms, terms));
        ChunkScratch scratch(terms, terms);
        chudnovsky_range_into(&result, 0, terms, leaf_terms, &scratch, 0);
        return result;
    }

    const std::size_t base_count = static_cast<std::size_t>((terms + chunk_terms - 1) / chunk_terms);
    std::vector<std::vector<ReservedTriple>> levels;
    std::uint64_t span_terms = chunk_terms;
    std::size_t count = base_count;
    while (true) {
        levels.push_back(make_reserved_level(count, std::min<std::uint64_t>(terms, span_terms), terms));
        if (count == 1) {
            break;
        }
        count = (count + 1) / 2;
        span_terms = std::min<std::uint64_t>(terms, span_terms * 2);
    }

    #pragma omp parallel num_threads(threads)
    {
        ChunkScratch scratch(chunk_terms, terms);
        #pragma omp for schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(base_count); ++index) {
            const std::uint64_t start = static_cast<std::uint64_t>(index) * chunk_terms;
            const std::uint64_t end = std::min<std::uint64_t>(start + chunk_terms, terms);
            chudnovsky_range_into(
                &levels[0][static_cast<std::size_t>(index)],
                start,
                end,
                leaf_terms,
                &scratch,
                0
            );
        }
    }

    for (std::size_t level = 0; level + 1 < levels.size(); ++level) {
        const std::size_t source_count = levels[level].size();
        const std::size_t target_count = levels[level + 1].size();
        #pragma omp parallel for schedule(static) num_threads(std::min<int>(threads, static_cast<int>(target_count)))
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(target_count); ++index) {
            const std::size_t left_index = static_cast<std::size_t>(2 * index);
            const std::size_t right_index = left_index + 1;
            levels[level + 1][static_cast<std::size_t>(index)].steal_from(&levels[level][left_index]);
            if (right_index < source_count) {
                merge_triples_inplace(
                    &levels[level + 1][static_cast<std::size_t>(index)],
                    &levels[level][right_index]
                );
            }
        }
    }

    return std::move(levels.back().front());
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
        ReservedTriple sum = chudnovsky_parallel_level_pool(
            terms,
            options.threads,
            options.chunk_terms,
            options.leaf_terms
        );

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
            << " representation=levelpool"
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
