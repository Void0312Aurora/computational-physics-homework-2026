#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BenchmarkResult {
    std::uint64_t decimal_digits = 0;
    std::size_t limbs = 0;
    std::size_t bytes_per_operand = 0;
    int repeats = 0;
    double mpn_seconds = 0.0;
    double mpz_seconds = 0.0;
    double mpn_digits_per_second = 0.0;
    double mpz_digits_per_second = 0.0;
    double speedup = 0.0;
};

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

std::uint64_t parse_u64(const std::string& text, const char* flag) {
    try {
        return std::stoull(text);
    } catch (...) {
        die(std::string("invalid value for ") + flag + ": " + text);
    }
}

std::vector<std::uint64_t> parse_digits_list(const std::string& csv) {
    std::vector<std::uint64_t> digits_list;
    std::size_t start = 0;
    while (start < csv.size()) {
        std::size_t end = csv.find(',', start);
        if (end == std::string::npos) {
            end = csv.size();
        }
        const std::string token = csv.substr(start, end - start);
        if (!token.empty()) {
            digits_list.push_back(parse_u64(token, "--digits-list"));
        }
        start = end + 1;
    }
    if (digits_list.empty()) {
        die("--digits-list must not be empty");
    }
    return digits_list;
}

std::size_t digits_to_limbs(std::uint64_t digits) {
    constexpr long double kLog2_10 = 3.32192809488736234787L;
    const long double bits = std::ceil(static_cast<long double>(digits) * kLog2_10);
    return static_cast<std::size_t>(std::ceil(bits / static_cast<long double>(GMP_NUMB_BITS)));
}

struct XorShift64 {
    std::uint64_t state = 0x123456789abcdefULL;

    explicit XorShift64(std::uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

    std::uint64_t next() {
        std::uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }
};

std::vector<mp_limb_t> make_random_limbs(std::size_t limbs, std::uint64_t seed) {
    XorShift64 rng(seed);
    std::vector<mp_limb_t> values(limbs);
    const mp_limb_t mask = static_cast<mp_limb_t>(~static_cast<mp_limb_t>(0));
    for (std::size_t i = 0; i < limbs; ++i) {
        values[i] = static_cast<mp_limb_t>(rng.next()) & mask;
    }
    if (!values.empty()) {
        values.back() |= static_cast<mp_limb_t>(1)
            << (std::min<unsigned>(GMP_NUMB_BITS, 8U) - 1U);
    }
    return values;
}

template <typename Fn>
double time_average_seconds(Fn&& fn, int repeats) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        fn();
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count() / repeats;
}

BenchmarkResult run_case(std::uint64_t digits, int requested_repeats) {
    const std::size_t limbs = digits_to_limbs(digits);
    std::vector<mp_limb_t> a = make_random_limbs(limbs, digits * 17 + 1);
    std::vector<mp_limb_t> b = make_random_limbs(limbs, digits * 31 + 7);
    std::vector<mp_limb_t> product(2 * limbs);

    mpz_t az;
    mpz_t bz;
    mpz_t rz;
    mpz_init(az);
    mpz_init(bz);
    mpz_init(rz);
    mpz_import(az, limbs, -1, sizeof(mp_limb_t), 0, 0, a.data());
    mpz_import(bz, limbs, -1, sizeof(mp_limb_t), 0, 0, b.data());
    mpz_realloc2(rz, static_cast<mp_bitcnt_t>(2 * limbs * GMP_NUMB_BITS));

    mpn_mul_n(product.data(), a.data(), b.data(), limbs);
    mpz_mul(rz, az, bz);

    double probe_seconds = time_average_seconds(
        [&]() { mpn_mul_n(product.data(), a.data(), b.data(), limbs); },
        1);
    int repeats = requested_repeats;
    if (repeats <= 0) {
        repeats = static_cast<int>(std::ceil(0.35 / std::max(probe_seconds, 1e-6)));
        repeats = std::clamp(repeats, 1, 6);
    }

    const double mpn_seconds = time_average_seconds(
        [&]() { mpn_mul_n(product.data(), a.data(), b.data(), limbs); },
        repeats);
    const double mpz_seconds = time_average_seconds(
        [&]() { mpz_mul(rz, az, bz); },
        repeats);

    const double mpn_digits_per_second = static_cast<double>(digits) / mpn_seconds;
    const double mpz_digits_per_second = static_cast<double>(digits) / mpz_seconds;

    BenchmarkResult result;
    result.decimal_digits = digits;
    result.limbs = limbs;
    result.bytes_per_operand = limbs * sizeof(mp_limb_t);
    result.repeats = repeats;
    result.mpn_seconds = mpn_seconds;
    result.mpz_seconds = mpz_seconds;
    result.mpn_digits_per_second = mpn_digits_per_second;
    result.mpz_digits_per_second = mpz_digits_per_second;
    result.speedup = mpz_seconds / mpn_seconds;

    mpz_clear(az);
    mpz_clear(bz);
    mpz_clear(rz);
    return result;
}

void write_csv(const std::string& path, const std::vector<BenchmarkResult>& results) {
    std::ofstream out(path);
    if (!out) {
        die("failed to open csv output: " + path);
    }
    out << "decimal_digits,limbs,bytes_per_operand,repeats,mpn_seconds,mpn_digits_per_second,mpz_seconds,mpz_digits_per_second,speedup\n";
    out << std::setprecision(12);
    for (const BenchmarkResult& row : results) {
        out
            << row.decimal_digits << ','
            << row.limbs << ','
            << row.bytes_per_operand << ','
            << row.repeats << ','
            << row.mpn_seconds << ','
            << row.mpn_digits_per_second << ','
            << row.mpz_seconds << ','
            << row.mpz_digits_per_second << ','
            << row.speedup << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::uint64_t> digits_list = {
            1000000ULL,
            10000000ULL,
            50000000ULL,
            100000000ULL,
        };
        int repeats = 0;
        std::string csv_path = "result/project2_mpn_mul_benchmark.csv";

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* flag) -> std::string {
                if (i + 1 >= argc) {
                    die(std::string("missing value for ") + flag);
                }
                return argv[++i];
            };

            if (arg == "--digits-list") {
                digits_list = parse_digits_list(require_value("--digits-list"));
            } else if (arg == "--repeats") {
                repeats = static_cast<int>(parse_u64(require_value("--repeats"), "--repeats"));
            } else if (arg == "--csv") {
                csv_path = require_value("--csv");
            } else {
                die("unknown argument: " + arg);
            }
        }

        std::vector<BenchmarkResult> results;
        results.reserve(digits_list.size());
        std::cout << std::fixed << std::setprecision(6);
        for (std::uint64_t digits : digits_list) {
            const BenchmarkResult row = run_case(digits, repeats);
            results.push_back(row);
            std::cout
                << "digits=" << row.decimal_digits
                << " limbs=" << row.limbs
                << " repeats=" << row.repeats
                << " mpn_seconds=" << row.mpn_seconds
                << " mpn_digits_per_second=" << row.mpn_digits_per_second
                << " mpz_seconds=" << row.mpz_seconds
                << " mpz_digits_per_second=" << row.mpz_digits_per_second
                << " speedup=" << row.speedup
                << '\n';
        }
        write_csv(csv_path, results);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
