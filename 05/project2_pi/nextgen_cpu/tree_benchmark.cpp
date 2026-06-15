#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "product_tree.hpp"

namespace project2::nextgen_cpu {

namespace {

struct Workload {
    std::size_t leaf_count = 0;
    std::size_t leaf_blocks = 0;
};

struct Options {
    std::vector<Workload> workloads{
        {32, 64},
        {32, 128},
        {64, 64},
        {64, 128},
        {64, 256},
    };
    std::size_t rns_threshold_blocks = 512;
    int repeats = 3;
    std::string csv_path = "result/project2_nextgen_cpu_tree_benchmark.csv";
};

struct Row {
    std::size_t leaf_count = 0;
    std::size_t leaf_blocks = 0;
    std::size_t approx_final_blocks = 0;
    int repeats = 0;
    double avg_schoolbook_seconds = 0.0;
    double avg_adaptive_seconds = 0.0;
    double speedup_adaptive_vs_schoolbook = 0.0;
    std::size_t adaptive_rns_mul_calls = 0;
    std::size_t adaptive_schoolbook_mul_calls = 0;
    std::size_t schoolbook_peak_live_blocks = 0;
    std::size_t adaptive_peak_live_blocks = 0;
    bool exact_match = false;
};

[[noreturn]] void die(std::string_view message) {
    throw std::runtime_error(std::string(message));
}

std::vector<Workload> parse_workloads(std::string_view text) {
    std::vector<Workload> workloads;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token(text.substr(start, comma == std::string_view::npos ? text.size() - start : comma - start));
        if (!token.empty()) {
            const std::size_t x = token.find('x');
            if (x == std::string::npos) {
                die("workloads must use leaf_countxleaf_blocks format");
            }
            workloads.push_back(Workload{
                .leaf_count = static_cast<std::size_t>(std::stoull(token.substr(0, x))),
                .leaf_blocks = static_cast<std::size_t>(std::stoull(token.substr(x + 1U))),
            });
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1U;
    }
    if (workloads.empty()) {
        die("workloads list must not be empty");
    }
    return workloads;
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

        if (arg == "--workloads") {
            options.workloads = parse_workloads(require_value("--workloads"));
        } else if (arg == "--rns-threshold-blocks") {
            options.rns_threshold_blocks = static_cast<std::size_t>(std::stoull(require_value("--rns-threshold-blocks")));
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

std::vector<std::vector<block_t>> random_leaf_blocks(const Workload& workload, std::mt19937_64* rng) {
    std::uniform_int_distribution<std::uint32_t> dist(0U, 0xffffffffU);
    std::vector<std::vector<block_t>> leaves(workload.leaf_count, std::vector<block_t>(workload.leaf_blocks, 0U));
    for (std::size_t leaf_index = 0; leaf_index < workload.leaf_count; ++leaf_index) {
        for (std::size_t block_index = 0; block_index < workload.leaf_blocks; ++block_index) {
            leaves[leaf_index][block_index] = dist(*rng);
        }
        if (!leaves[leaf_index].empty()) {
            leaves[leaf_index].back() |= 0x1U;
        }
    }
    return leaves;
}

double average(const std::vector<double>& values) {
    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

Row benchmark_workload(const Workload& workload, const Options& options, std::mt19937_64* rng) {
    std::vector<double> schoolbook_samples;
    std::vector<double> adaptive_samples;
    schoolbook_samples.reserve(options.repeats);
    adaptive_samples.reserve(options.repeats);

    std::size_t last_adaptive_rns_mul_calls = 0;
    std::size_t last_adaptive_schoolbook_mul_calls = 0;
    std::size_t last_schoolbook_peak_live_blocks = 0;
    std::size_t last_adaptive_peak_live_blocks = 0;

    const ProductTreeConfig config{
        .page_size_blocks = std::max<std::size_t>(4096, workload.leaf_blocks * workload.leaf_count),
        .rns_threshold_blocks = options.rns_threshold_blocks,
    };

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        const std::vector<std::vector<block_t>> leaves = random_leaf_blocks(workload, rng);

        const auto schoolbook_start = std::chrono::steady_clock::now();
        ProductTreeResult schoolbook_result = reduce_product_tree(leaves, TreeMultiplyMode::schoolbook, config);
        const auto schoolbook_end = std::chrono::steady_clock::now();

        const auto adaptive_start = std::chrono::steady_clock::now();
        ProductTreeResult adaptive_result = reduce_product_tree(leaves, TreeMultiplyMode::adaptive_rns, config);
        const auto adaptive_end = std::chrono::steady_clock::now();

        if (!equals(
                schoolbook_result.level.arena,
                schoolbook_result.level.nodes.front(),
                adaptive_result.level.arena,
                adaptive_result.level.nodes.front()
            )) {
            die("schoolbook and adaptive product trees diverged");
        }

        schoolbook_samples.push_back(std::chrono::duration<double>(schoolbook_end - schoolbook_start).count());
        adaptive_samples.push_back(std::chrono::duration<double>(adaptive_end - adaptive_start).count());
        last_adaptive_rns_mul_calls = adaptive_result.stats.rns_mul_calls;
        last_adaptive_schoolbook_mul_calls = adaptive_result.stats.schoolbook_mul_calls;
        last_schoolbook_peak_live_blocks = schoolbook_result.stats.peak_live_blocks;
        last_adaptive_peak_live_blocks = adaptive_result.stats.peak_live_blocks;
    }

    const double avg_schoolbook = average(schoolbook_samples);
    const double avg_adaptive = average(adaptive_samples);
    return Row{
        .leaf_count = workload.leaf_count,
        .leaf_blocks = workload.leaf_blocks,
        .approx_final_blocks = workload.leaf_count * workload.leaf_blocks,
        .repeats = options.repeats,
        .avg_schoolbook_seconds = avg_schoolbook,
        .avg_adaptive_seconds = avg_adaptive,
        .speedup_adaptive_vs_schoolbook = avg_schoolbook / avg_adaptive,
        .adaptive_rns_mul_calls = last_adaptive_rns_mul_calls,
        .adaptive_schoolbook_mul_calls = last_adaptive_schoolbook_mul_calls,
        .schoolbook_peak_live_blocks = last_schoolbook_peak_live_blocks,
        .adaptive_peak_live_blocks = last_adaptive_peak_live_blocks,
        .exact_match = true,
    };
}

void write_csv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        die("failed to open tree benchmark csv path");
    }
    out << "leaf_count,leaf_blocks,approx_final_blocks,repeats,avg_schoolbook_seconds,avg_adaptive_seconds,speedup_adaptive_vs_schoolbook,adaptive_rns_mul_calls,adaptive_schoolbook_mul_calls,schoolbook_peak_live_blocks,adaptive_peak_live_blocks,exact_match\n";
    out << std::fixed << std::setprecision(9);
    for (const Row& row : rows) {
        out
            << row.leaf_count << ','
            << row.leaf_blocks << ','
            << row.approx_final_blocks << ','
            << row.repeats << ','
            << row.avg_schoolbook_seconds << ','
            << row.avg_adaptive_seconds << ','
            << row.speedup_adaptive_vs_schoolbook << ','
            << row.adaptive_rns_mul_calls << ','
            << row.adaptive_schoolbook_mul_calls << ','
            << row.schoolbook_peak_live_blocks << ','
            << row.adaptive_peak_live_blocks << ','
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
        rows.reserve(options.workloads.size());
        for (const Workload& workload : options.workloads) {
            rows.push_back(benchmark_workload(workload, options, &rng));
        }
        write_csv(options.csv_path, rows);

        for (const Row& row : rows) {
            std::cout
                << "nextgen_cpu_tree_benchmark"
                << " leaf_count=" << row.leaf_count
                << " leaf_blocks=" << row.leaf_blocks
                << " approx_final_blocks=" << row.approx_final_blocks
                << " repeats=" << row.repeats
                << " avg_schoolbook_seconds=" << row.avg_schoolbook_seconds
                << " avg_adaptive_seconds=" << row.avg_adaptive_seconds
                << " speedup_adaptive_vs_schoolbook=" << row.speedup_adaptive_vs_schoolbook
                << " adaptive_rns_mul_calls=" << row.adaptive_rns_mul_calls
                << " adaptive_schoolbook_mul_calls=" << row.adaptive_schoolbook_mul_calls
                << " schoolbook_peak_live_blocks=" << row.schoolbook_peak_live_blocks
                << " adaptive_peak_live_blocks=" << row.adaptive_peak_live_blocks
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
