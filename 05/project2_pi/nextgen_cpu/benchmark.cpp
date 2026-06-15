#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "rns_ntt.hpp"

namespace project2::nextgen_cpu {

namespace {

struct Options {
    std::vector<std::size_t> sizes_blocks{8, 16, 32, 64, 128, 256};
    int repeats = 3;
    std::string csv_path = "result/project2_nextgen_cpu_multiply_benchmark.csv";
};

struct Row {
    std::size_t input_blocks = 0;
    std::size_t approx_decimal_digits = 0;
    int repeats = 0;
    double avg_schoolbook_seconds = 0.0;
    double avg_rns_seconds = 0.0;
    double speedup_rns_vs_schoolbook = 0.0;
    std::size_t rns_ntt_size = 0;
    std::size_t rns_modulus_count = 0;
    std::size_t rns_radix_bits = 0;
    bool exact_match = false;
};

[[noreturn]] void die(std::string_view message) {
    throw std::runtime_error(std::string(message));
}

std::vector<std::size_t> parse_sizes(std::string_view text) {
    std::vector<std::size_t> values;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token(text.substr(start, comma == std::string_view::npos ? text.size() - start : comma - start));
        if (!token.empty()) {
            values.push_back(static_cast<std::size_t>(std::stoull(token)));
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1U;
    }
    if (values.empty()) {
        die("sizes list must not be empty");
    }
    return values;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](std::string_view flag) -> std::string {
            if (i + 1 >= argc) {
                die(std::string("missing value for ") + std::string(flag));
            }
            return argv[++i];
        };

        if (arg == "--sizes-blocks") {
            options.sizes_blocks = parse_sizes(require_value("--sizes-blocks"));
        } else if (arg == "--repeats") {
            options.repeats = std::max(1, std::stoi(require_value("--repeats")));
        } else if (arg == "--csv") {
            options.csv_path = require_value("--csv");
        } else {
            die(std::string("unknown argument: ") + arg);
        }
    }
    return options;
}

std::vector<block_t> random_blocks(std::size_t size, std::mt19937_64* rng) {
    std::uniform_int_distribution<std::uint32_t> dist(0U, 0xffffffffU);
    std::vector<block_t> blocks(size, 0);
    for (std::size_t index = 0; index < size; ++index) {
        blocks[index] = dist(*rng);
    }
    if (!blocks.empty()) {
        blocks.back() |= 0x1U;
    }
    return blocks;
}

double average(const std::vector<double>& values) {
    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

Row benchmark_size(std::size_t input_blocks, int repeats, std::mt19937_64* rng) {
    std::vector<double> schoolbook_samples;
    std::vector<double> rns_samples;
    schoolbook_samples.reserve(repeats);
    rns_samples.reserve(repeats);
    RnsWorkspace workspace;

    std::size_t last_ntt_size = 0;
    std::size_t last_modulus_count = 0;
    std::size_t last_radix_bits = 0;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        PageArena arena(/*page_size_blocks=*/std::max<std::size_t>(256, input_blocks * 16U), /*reserve_pages=*/4);
        BigHandle lhs = arena.allocate(input_blocks + 2U);
        BigHandle rhs = arena.allocate(input_blocks + 2U);
        BigHandle schoolbook = arena.allocate(input_blocks * 2U + 4U);
        BigHandle rns = arena.allocate(input_blocks * 2U + 4U);

        assign_blocks(&arena, &lhs, random_blocks(input_blocks, rng));
        assign_blocks(&arena, &rhs, random_blocks(input_blocks, rng));

        const auto schoolbook_start = std::chrono::steady_clock::now();
        mul_into(&arena, &schoolbook, lhs, rhs);
        const auto schoolbook_end = std::chrono::steady_clock::now();

        const auto rns_start = std::chrono::steady_clock::now();
        const RnsMultiplyStats rns_stats = mul_into_rns_ntt(&arena, &rns, lhs, rhs, &workspace);
        const auto rns_end = std::chrono::steady_clock::now();

        if (!equals(arena, schoolbook, rns)) {
            die("schoolbook and rns_ntt results diverged");
        }

        schoolbook_samples.push_back(std::chrono::duration<double>(schoolbook_end - schoolbook_start).count());
        rns_samples.push_back(std::chrono::duration<double>(rns_end - rns_start).count());
        last_ntt_size = rns_stats.ntt_size;
        last_modulus_count = rns_stats.modulus_count;
        last_radix_bits = rns_stats.radix_bits;
    }

    const double avg_schoolbook = average(schoolbook_samples);
    const double avg_rns = average(rns_samples);
    return Row{
        .input_blocks = input_blocks,
        .approx_decimal_digits = static_cast<std::size_t>(input_blocks * 9.632959861247398),
        .repeats = repeats,
        .avg_schoolbook_seconds = avg_schoolbook,
        .avg_rns_seconds = avg_rns,
        .speedup_rns_vs_schoolbook = avg_schoolbook / avg_rns,
        .rns_ntt_size = last_ntt_size,
        .rns_modulus_count = last_modulus_count,
        .rns_radix_bits = last_radix_bits,
        .exact_match = true,
    };
}

void write_csv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        die("failed to open benchmark csv path");
    }
    out << "input_blocks,approx_decimal_digits,repeats,avg_schoolbook_seconds,avg_rns_seconds,speedup_rns_vs_schoolbook,rns_ntt_size,rns_modulus_count,rns_radix_bits,exact_match\n";
    out << std::fixed << std::setprecision(9);
    for (const Row& row : rows) {
        out
            << row.input_blocks << ','
            << row.approx_decimal_digits << ','
            << row.repeats << ','
            << row.avg_schoolbook_seconds << ','
            << row.avg_rns_seconds << ','
            << row.speedup_rns_vs_schoolbook << ','
            << row.rns_ntt_size << ','
            << row.rns_modulus_count << ','
            << row.rns_radix_bits << ','
            << (row.exact_match ? "True" : "False") << '\n';
    }
}

}  // namespace

}  // namespace project2::nextgen_cpu

int main(int argc, char** argv) {
    using namespace project2::nextgen_cpu;

    try {
        const Options options = parse_args(argc, argv);
        std::mt19937_64 rng(20260421ULL);
        std::vector<Row> rows;
        rows.reserve(options.sizes_blocks.size());
        for (std::size_t input_blocks : options.sizes_blocks) {
            rows.push_back(benchmark_size(input_blocks, options.repeats, &rng));
        }
        write_csv(options.csv_path, rows);

        for (const Row& row : rows) {
            std::cout
                << "nextgen_cpu_multiply_benchmark"
                << " input_blocks=" << row.input_blocks
                << " approx_decimal_digits=" << row.approx_decimal_digits
                << " repeats=" << row.repeats
                << " avg_schoolbook_seconds=" << row.avg_schoolbook_seconds
                << " avg_rns_seconds=" << row.avg_rns_seconds
                << " speedup_rns_vs_schoolbook=" << row.speedup_rns_vs_schoolbook
                << " rns_ntt_size=" << row.rns_ntt_size
                << " rns_modulus_count=" << row.rns_modulus_count
                << " rns_radix_bits=" << row.rns_radix_bits
                << " exact_match=" << (row.exact_match ? "True" : "False")
                << "\n";
        }
        std::cout << "wrote_csv=" << options.csv_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
