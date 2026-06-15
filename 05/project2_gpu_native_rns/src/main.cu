#include "project2_gpu_native_rns/rns_runtime.cuh"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
    project2::gpu_native_rns::SmokeConfig config;
    std::string output_tag;
    std::size_t pi_digits = 1000000;
    std::size_t pi_terms = 8;
    bool pi_digits_explicit = false;
    bool pi_terms_explicit = false;
    bool lifecycle_block_smoke = false;
    bool scaled_constant_smoke = false;
    bool reciprocal_seed_smoke = false;
    bool sqrt_seed_smoke = false;
    bool division_smoke = false;
    bool correction_domain_smoke = false;
    bool pi_route_smoke = false;
    bool pi_execution_plan_smoke = false;
    bool pi_pqt_smoke = false;
    bool pi_pqt_tensor_smoke = false;
    bool pi_pqt_tensor_tree_smoke = false;
    bool pi_end_to_end_smoke = false;
    bool pi_end_to_end_benchmark = false;
};

project2::gpu_native_rns::VerifyMode parse_verify_mode(const std::string& value) {
    using project2::gpu_native_rns::VerifyMode;
    if (value == "none") {
        return VerifyMode::kNone;
    }
    if (value == "sampled") {
        return VerifyMode::kSampled;
    }
    if (value == "full") {
        return VerifyMode::kFull;
    }
    throw std::invalid_argument("unknown verify mode: " + value + " (expected none, sampled, or full)");
}

std::string require_value(int argc, char** argv, int& index, const char* option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + option);
    }
    ++index;
    return argv[index];
} 

std::filesystem::path ensure_result_dir(const std::filesystem::path& category) {
    const std::filesystem::path result_dir = std::filesystem::path("result") / category;
    std::filesystem::create_directories(result_dir);
    return result_dir;
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--value-count") {
            options.config.value_count = std::stoull(require_value(argc, argv, index, "--value-count"));
        } else if (arg == "--slot-count") {
            options.config.slot_count = std::stoull(require_value(argc, argv, index, "--slot-count"));
        } else if (arg == "--iterations") {
            options.config.measured_iterations = std::stoi(require_value(argc, argv, index, "--iterations"));
        } else if (arg == "--input-bits") {
            options.config.input_bits = std::stoi(require_value(argc, argv, index, "--input-bits"));
        } else if (arg == "--no-verify") {
            options.config.verify_mode = project2::gpu_native_rns::VerifyMode::kNone;
        } else if (arg == "--verify-mode") {
            options.config.verify_mode = parse_verify_mode(require_value(argc, argv, index, "--verify-mode"));
        } else if (arg == "--verify-samples") {
            options.config.verify_sample_count = std::stoull(require_value(argc, argv, index, "--verify-samples"));
        } else if (arg == "--output-tag") {
            options.output_tag = require_value(argc, argv, index, "--output-tag");
        } else if (arg == "--pi-digits") {
            options.pi_digits = std::stoull(require_value(argc, argv, index, "--pi-digits"));
            options.pi_digits_explicit = true;
        } else if (arg == "--pi-terms") {
            options.pi_terms = std::stoull(require_value(argc, argv, index, "--pi-terms"));
            options.pi_terms_explicit = true;
        } else if (arg == "--lifecycle-block-smoke") {
            options.lifecycle_block_smoke = true;
        } else if (arg == "--scaled-constant-smoke") {
            options.scaled_constant_smoke = true;
        } else if (arg == "--reciprocal-seed-smoke") {
            options.reciprocal_seed_smoke = true;
        } else if (arg == "--sqrt-seed-smoke") {
            options.sqrt_seed_smoke = true;
        } else if (arg == "--division-smoke") {
            options.division_smoke = true;
        } else if (arg == "--correction-domain-smoke") {
            options.correction_domain_smoke = true;
        } else if (arg == "--pi-route-smoke") {
            options.pi_route_smoke = true;
        } else if (arg == "--pi-execution-plan-smoke") {
            options.pi_execution_plan_smoke = true;
        } else if (arg == "--pi-pqt-smoke") {
            options.pi_pqt_smoke = true;
        } else if (arg == "--pi-pqt-tensor-smoke") {
            options.pi_pqt_tensor_smoke = true;
        } else if (arg == "--pi-pqt-tree-tensor-smoke") {
            options.pi_pqt_tensor_tree_smoke = true;
        } else if (arg == "--pi-end-to-end-smoke") {
            options.pi_end_to_end_smoke = true;
        } else if (arg == "--pi-end-to-end-benchmark") {
            options.pi_end_to_end_benchmark = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        namespace fs = std::filesystem;
        const CliOptions options = parse_args(argc, argv);
        if (options.lifecycle_block_smoke) {
            return project2::gpu_native_rns::lifecycle_block_smoke_test(std::cout) ? 0 : 1;
        }
        if (options.scaled_constant_smoke) {
            return project2::gpu_native_rns::scaled_constant_smoke_test(std::cout) ? 0 : 1;
        }
        if (options.reciprocal_seed_smoke) {
            return project2::gpu_native_rns::reciprocal_seed_smoke_test(std::cout) ? 0 : 1;
        }
        if (options.sqrt_seed_smoke) {
            return project2::gpu_native_rns::sqrt_seed_smoke_test(std::cout) ? 0 : 1;
        }
        if (options.division_smoke) {
            return project2::gpu_native_rns::division_smoke_test(std::cout) ? 0 : 1;
        }
        if (options.correction_domain_smoke) {
            return project2::gpu_native_rns::correction_domain_smoke_test(std::cout) ? 0 : 1;
        }
        if (options.pi_route_smoke) {
            return project2::gpu_native_rns::pi_route_smoke_test(std::cout, options.pi_digits) ? 0 : 1;
        }
        if (options.pi_execution_plan_smoke) {
            return project2::gpu_native_rns::pi_execution_plan_smoke_test(std::cout, options.pi_digits) ? 0 : 1;
        }
        if (options.pi_pqt_smoke) {
            return project2::gpu_native_rns::pi_pqt_smoke_test(std::cout, options.pi_terms) ? 0 : 1;
        }
        if (options.pi_pqt_tensor_smoke) {
            return project2::gpu_native_rns::pi_pqt_tensor_smoke_test(std::cout, options.pi_terms) ? 0 : 1;
        }
        if (options.pi_pqt_tensor_tree_smoke) {
            return project2::gpu_native_rns::pi_pqt_tensor_tree_smoke_test(std::cout, options.pi_terms) ? 0 : 1;
        }
        if (options.pi_end_to_end_smoke) {
            const std::size_t smoke_digits = options.pi_digits_explicit ? options.pi_digits : 50;
            std::size_t smoke_terms = options.pi_terms;
            if (!options.pi_terms_explicit) {
                smoke_terms = std::max(
                    smoke_terms,
                    project2::gpu_native_rns::plan_pi_route(smoke_digits).chudnovsky_terms
                );
            }
            return project2::gpu_native_rns::pi_end_to_end_smoke_test(std::cout, smoke_terms, smoke_digits) ? 0 : 1;
        }
        if (options.pi_end_to_end_benchmark) {
            const std::size_t benchmark_digits = options.pi_digits_explicit ? options.pi_digits : 400;
            std::size_t benchmark_terms = options.pi_terms;
            if (!options.pi_terms_explicit) {
                benchmark_terms = std::max(
                    benchmark_terms,
                    project2::gpu_native_rns::plan_pi_route(benchmark_digits).chudnovsky_terms
                );
            }

            const auto report = project2::gpu_native_rns::run_pi_end_to_end_benchmark(
                benchmark_terms,
                benchmark_digits,
                options.config.measured_iterations
            );
            project2::gpu_native_rns::print_pi_end_to_end_benchmark_report(std::cout, report);

            const fs::path result_dir = ensure_result_dir("end_to_end_benchmark");
            const std::string base_name =
                options.output_tag.empty()
                    ? "project2_pi_end_to_end_benchmark"
                    : "project2_pi_end_to_end_benchmark_" + options.output_tag;

            {
                std::ofstream log_file(result_dir / (base_name + ".log"));
                if (!log_file) {
                    throw std::runtime_error("failed to open pi end-to-end benchmark log output");
                }
                project2::gpu_native_rns::print_pi_end_to_end_benchmark_report(log_file, report);
            }

            project2::gpu_native_rns::write_pi_end_to_end_benchmark_csv(
                report,
                (result_dir / (base_name + ".csv")).string()
            );
            return report.ok ? 0 : 1;
        }
        const auto report = project2::gpu_native_rns::run_smoke_test(options.config);

        project2::gpu_native_rns::print_smoke_report(std::cout, report);

        const fs::path result_dir = ensure_result_dir("runtime_smoke");
        const std::string base_name =
            options.output_tag.empty()
                ? "project2_gpu_native_rns_smoke"
                : "project2_gpu_native_rns_smoke_" + options.output_tag;

        {
            std::ofstream log_file(result_dir / (base_name + ".log"));
            if (!log_file) {
                throw std::runtime_error("failed to open smoke log output");
            }
            project2::gpu_native_rns::print_smoke_report(log_file, report);
        }

        project2::gpu_native_rns::write_smoke_csv(
            report,
            (result_dir / (base_name + ".csv")).string()
        );
        return report.ok ? 0 : 1;
    } catch (const std::exception& exc) {
        std::cerr << "project2_gpu_native_rns_smoke_error: " << exc.what() << '\n';
        return 2;
    }
}
