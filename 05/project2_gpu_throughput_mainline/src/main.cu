#include "project2_gpu_throughput_mainline/runtime.cuh"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    bool print_plan = false;
    bool pointwise_add_smoke = false;
    bool pair_pack_smoke = false;
    bool persistent_level_reduce_smoke = false;
    bool batched_fft_smoke = false;
    bool residue_fft_bridge_smoke = false;
    bool grouped_residue_fft_bridge_smoke = false;
    bool grouped_level_planner_smoke = false;
    bool grouped_level_planner_multilimb_smoke = false;
    bool grouped_level_planner_exact_moduli_smoke = false;
    bool grouped_level_planner_exact_moduli_pfactor_smoke = false;
    bool grouped_level_planner_exact_moduli_pq_smoke = false;
    bool grouped_level_planner_exact_moduli_pqt_smoke = false;
    bool pi_end_to_end_smoke = false;
    bool grouped_level_planner_split_mask63_smoke = false;
    std::string output_tag;
    project2::gpu_throughput_mainline::PackedPointwiseAddConfig pointwise_add_config;
    project2::gpu_throughput_mainline::DevicePairPackConfig pair_pack_config;
    project2::gpu_throughput_mainline::PersistentLevelReduceConfig persistent_level_reduce_config;
    project2::gpu_throughput_mainline::BatchedFftBackboneConfig batched_fft_config;
    project2::gpu_throughput_mainline::ResidueFftBridgeConfig residue_fft_bridge_config;
    project2::gpu_throughput_mainline::GroupedResidueFftBridgeConfig grouped_residue_fft_bridge_config;
    project2::gpu_throughput_mainline::GroupedLevelPlannerConfig grouped_level_planner_config;
    project2::gpu_throughput_mainline::PiEndToEndSmokeConfig pi_end_to_end_config;
};

const char* require_value(int argc, char** argv, int& index, const char* flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + flag);
    }
    ++index;
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--print-plan") {
            options.print_plan = true;
        } else if (arg == "--pointwise-add-smoke") {
            options.pointwise_add_smoke = true;
        } else if (arg == "--pair-pack-smoke") {
            options.pair_pack_smoke = true;
        } else if (arg == "--persistent-level-reduce-smoke") {
            options.persistent_level_reduce_smoke = true;
        } else if (arg == "--batched-fft-smoke") {
            options.batched_fft_smoke = true;
        } else if (arg == "--residue-fft-bridge-smoke") {
            options.residue_fft_bridge_smoke = true;
        } else if (arg == "--grouped-residue-fft-bridge-smoke") {
            options.grouped_residue_fft_bridge_smoke = true;
        } else if (arg == "--grouped-level-planner-smoke") {
            options.grouped_level_planner_smoke = true;
        } else if (arg == "--grouped-level-planner-multilimb-smoke") {
            options.grouped_level_planner_multilimb_smoke = true;
        } else if (arg == "--grouped-level-planner-exact-moduli-smoke") {
            options.grouped_level_planner_exact_moduli_smoke = true;
        } else if (arg == "--grouped-level-planner-exact-moduli-pfactor-smoke") {
            options.grouped_level_planner_exact_moduli_pfactor_smoke = true;
        } else if (arg == "--grouped-level-planner-exact-moduli-pq-smoke") {
            options.grouped_level_planner_exact_moduli_pq_smoke = true;
        } else if (arg == "--grouped-level-planner-exact-moduli-pqt-smoke") {
            options.grouped_level_planner_exact_moduli_pqt_smoke = true;
        } else if (arg == "--pi-end-to-end-smoke") {
            options.pi_end_to_end_smoke = true;
        } else if (arg == "--grouped-level-planner-split-mask63-smoke") {
            options.grouped_level_planner_split_mask63_smoke = true;
        } else if (arg == "--output-tag") {
            options.output_tag = require_value(argc, argv, index, "--output-tag");
        } else if (arg == "--batch-count") {
            options.pointwise_add_config.batch_count =
                std::stoull(require_value(argc, argv, index, "--batch-count"));
        } else if (arg == "--merge-count") {
            options.pair_pack_config.merge_count =
                std::stoull(require_value(argc, argv, index, "--merge-count"));
        } else if (arg == "--node-count") {
            const auto value = std::stoull(require_value(argc, argv, index, "--node-count"));
            options.persistent_level_reduce_config.node_count = value;
            options.residue_fft_bridge_config.node_count = value;
            options.grouped_residue_fft_bridge_config.node_count = value;
            options.grouped_level_planner_config.node_count = value;
            options.pi_end_to_end_config.term_count = value;
        } else if (arg == "--term-count") {
            options.pi_end_to_end_config.term_count =
                std::stoull(require_value(argc, argv, index, "--term-count"));
        } else if (arg == "--fft-batch-count") {
            options.batched_fft_config.batch_count =
                std::stoull(require_value(argc, argv, index, "--fft-batch-count"));
        } else if (arg == "--fft-length") {
            options.batched_fft_config.fft_length =
                std::stoull(require_value(argc, argv, index, "--fft-length"));
        } else if (arg == "--bridge-modulus-index") {
            options.residue_fft_bridge_config.bridge_modulus_index =
                std::stoi(require_value(argc, argv, index, "--bridge-modulus-index"));
        } else if (arg == "--packing-mask") {
            const auto value =
                static_cast<std::uint32_t>(std::stoul(require_value(argc, argv, index, "--packing-mask")));
            options.residue_fft_bridge_config.packing_mask = value;
            options.grouped_residue_fft_bridge_config.packing_mask = value;
            options.grouped_level_planner_config.packing_mask = value;
        } else if (arg == "--slot-count") {
            const auto value = std::stoull(require_value(argc, argv, index, "--slot-count"));
            options.pointwise_add_config.slot_count = value;
            options.pair_pack_config.slot_count = value;
            options.persistent_level_reduce_config.slot_count = value;
            options.residue_fft_bridge_config.slot_count = value;
            options.grouped_residue_fft_bridge_config.slot_count = value;
            options.grouped_level_planner_config.slot_count = value;
            options.pi_end_to_end_config.slot_count = value;
        } else if (arg == "--modulus-count") {
            const auto value = std::stoi(require_value(argc, argv, index, "--modulus-count"));
            options.pointwise_add_config.modulus_count = value;
            options.pair_pack_config.modulus_count = value;
            options.persistent_level_reduce_config.modulus_count = value;
            options.residue_fft_bridge_config.modulus_count = value;
            options.grouped_residue_fft_bridge_config.modulus_count = value;
            options.grouped_level_planner_config.modulus_count = value;
            options.pi_end_to_end_config.modulus_count = value;
        } else if (arg == "--force-closure-modulus-count") {
            options.pi_end_to_end_config.forced_closure_modulus_count =
                std::stoi(require_value(argc, argv, index, "--force-closure-modulus-count"));
        } else if (arg == "--target-digits") {
            options.pi_end_to_end_config.target_digits =
                std::stoull(require_value(argc, argv, index, "--target-digits"));
        } else if (arg == "--report-decimal-digits") {
            options.pi_end_to_end_config.report_decimal_digits =
                std::stoull(require_value(argc, argv, index, "--report-decimal-digits"));
        } else if (arg == "--warmup") {
            const auto value = std::stoi(require_value(argc, argv, index, "--warmup"));
            options.pointwise_add_config.warmup_iterations = value;
            options.pair_pack_config.warmup_iterations = value;
            options.persistent_level_reduce_config.warmup_iterations = value;
            options.batched_fft_config.warmup_iterations = value;
            options.residue_fft_bridge_config.warmup_iterations = value;
            options.grouped_residue_fft_bridge_config.warmup_iterations = value;
            options.grouped_level_planner_config.warmup_iterations = value;
            options.pi_end_to_end_config.warmup_iterations = value;
        } else if (arg == "--iterations") {
            const auto value = std::stoi(require_value(argc, argv, index, "--iterations"));
            options.pointwise_add_config.measured_iterations = value;
            options.pair_pack_config.measured_iterations = value;
            options.persistent_level_reduce_config.measured_iterations = value;
            options.batched_fft_config.measured_iterations = value;
            options.residue_fft_bridge_config.measured_iterations = value;
            options.grouped_residue_fft_bridge_config.measured_iterations = value;
            options.grouped_level_planner_config.measured_iterations = value;
            options.pi_end_to_end_config.measured_iterations = value;
        } else if (arg == "--verification-samples") {
            const auto value = std::stoull(require_value(argc, argv, index, "--verification-samples"));
            options.pointwise_add_config.verification_sample_count = value;
            options.pair_pack_config.verification_sample_count = value;
            options.persistent_level_reduce_config.verification_sample_count = value;
            options.batched_fft_config.verification_sample_count = value;
            options.residue_fft_bridge_config.verification_sample_count = value;
            options.grouped_residue_fft_bridge_config.verification_sample_count = value;
            options.grouped_level_planner_config.verification_sample_count = value;
        } else if (arg == "--help") {
            std::cout
                << "Usage: project2_gpu_throughput_mainline [--print-plan] [--pointwise-add-smoke]\n"
                << "       [--pair-pack-smoke] [--persistent-level-reduce-smoke] [--batched-fft-smoke]\n"
                << "       [--residue-fft-bridge-smoke] [--grouped-residue-fft-bridge-smoke]\n"
                << "       [--grouped-level-planner-smoke] [--grouped-level-planner-multilimb-smoke]\n"
                << "       [--grouped-level-planner-exact-moduli-smoke]\n"
                << "       [--grouped-level-planner-exact-moduli-pfactor-smoke]\n"
                << "       [--grouped-level-planner-exact-moduli-pq-smoke]\n"
                << "       [--grouped-level-planner-exact-moduli-pqt-smoke]\n"
                << "       [--pi-end-to-end-smoke]\n"
                << "       [--grouped-level-planner-split-mask63-smoke]\n"
                << "       [--batch-count N] [--merge-count N] [--node-count N] [--term-count N] [--fft-batch-count N]\n"
                << "       [--fft-length N] [--bridge-modulus-index N] [--packing-mask N]\n"
                << "       [--slot-count N] [--modulus-count N] [--force-closure-modulus-count N] [--target-digits N]\n"
                << "       [--report-decimal-digits N]\n"
                << "       [--warmup N] [--iterations N] [--verification-samples N]\n"
                << "       [--output-tag TAG]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (
        !options.print_plan &&
        !options.pointwise_add_smoke &&
        !options.pair_pack_smoke &&
        !options.persistent_level_reduce_smoke &&
        !options.batched_fft_smoke &&
        !options.residue_fft_bridge_smoke &&
        !options.grouped_residue_fft_bridge_smoke &&
        !options.grouped_level_planner_smoke &&
        !options.grouped_level_planner_multilimb_smoke &&
        !options.grouped_level_planner_exact_moduli_smoke &&
        !options.grouped_level_planner_exact_moduli_pfactor_smoke &&
        !options.grouped_level_planner_exact_moduli_pq_smoke &&
        !options.grouped_level_planner_exact_moduli_pqt_smoke &&
        !options.pi_end_to_end_smoke &&
        !options.grouped_level_planner_split_mask63_smoke
    ) {
        options.print_plan = true;
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const fs::path result_dir = fs::path("result");
        fs::create_directories(result_dir);
        int exit_code = 0;

        if (options.print_plan) {
            const auto report = project2::gpu_throughput_mainline::plan_throughput_mainline();
            project2::gpu_throughput_mainline::print_throughput_plan_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(result_dir / ("project2_gpu_throughput_plan_" + options.output_tag + ".log"));
                if (!out) {
                    throw std::runtime_error("failed to open throughput plan log output");
                }
                project2::gpu_throughput_mainline::print_throughput_plan_report(out, report);
            }
        }

        if (options.pointwise_add_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_packed_pointwise_add_smoke(options.pointwise_add_config);
            project2::gpu_throughput_mainline::print_packed_pointwise_add_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir / ("project2_gpu_throughput_pointwise_add_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open packed pointwise-add log output");
                }
                project2::gpu_throughput_mainline::print_packed_pointwise_add_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.pair_pack_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_device_pair_pack_smoke(options.pair_pack_config);
            project2::gpu_throughput_mainline::print_device_pair_pack_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir / ("project2_gpu_throughput_pair_pack_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open pair-pack log output");
                }
                project2::gpu_throughput_mainline::print_device_pair_pack_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.persistent_level_reduce_smoke) {
            const auto report = project2::gpu_throughput_mainline::run_persistent_level_reduce_smoke(
                options.persistent_level_reduce_config
            );
            project2::gpu_throughput_mainline::print_persistent_level_reduce_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir / ("project2_gpu_throughput_persistent_reduce_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open persistent-reduce log output");
                }
                project2::gpu_throughput_mainline::print_persistent_level_reduce_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.batched_fft_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_batched_fft_backbone_smoke(options.batched_fft_config);
            project2::gpu_throughput_mainline::print_batched_fft_backbone_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir / ("project2_gpu_throughput_batched_fft_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open batched-fft log output");
                }
                project2::gpu_throughput_mainline::print_batched_fft_backbone_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.residue_fft_bridge_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_residue_fft_bridge_smoke(options.residue_fft_bridge_config);
            project2::gpu_throughput_mainline::print_residue_fft_bridge_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir / ("project2_gpu_throughput_residue_fft_bridge_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open residue-fft-bridge log output");
                }
                project2::gpu_throughput_mainline::print_residue_fft_bridge_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.grouped_residue_fft_bridge_smoke) {
            const auto report = project2::gpu_throughput_mainline::run_grouped_residue_fft_bridge_smoke(
                options.grouped_residue_fft_bridge_config
            );
            project2::gpu_throughput_mainline::print_grouped_residue_fft_bridge_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir / ("project2_gpu_throughput_grouped_residue_fft_bridge_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open grouped-residue-fft-bridge log output");
                }
                project2::gpu_throughput_mainline::print_grouped_residue_fft_bridge_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.grouped_level_planner_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_grouped_level_planner_smoke(options.grouped_level_planner_config);
            project2::gpu_throughput_mainline::print_grouped_level_planner_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir / ("project2_gpu_throughput_grouped_level_planner_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open grouped-level-planner log output");
                }
                project2::gpu_throughput_mainline::print_grouped_level_planner_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.grouped_level_planner_multilimb_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_grouped_level_planner_multilimb_smoke(
                    options.grouped_level_planner_config
                );
            project2::gpu_throughput_mainline::print_grouped_level_planner_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir /
                    ("project2_gpu_throughput_grouped_level_planner_multilimb_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open grouped-level-planner-multilimb log output");
                }
                project2::gpu_throughput_mainline::print_grouped_level_planner_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.grouped_level_planner_exact_moduli_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_grouped_level_planner_exact_moduli_smoke(
                    options.grouped_level_planner_config
                );
            project2::gpu_throughput_mainline::print_grouped_level_planner_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir /
                    ("project2_gpu_throughput_grouped_level_planner_exact_moduli_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open grouped-level-planner-exact-moduli log output");
                }
                project2::gpu_throughput_mainline::print_grouped_level_planner_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.grouped_level_planner_exact_moduli_pfactor_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_grouped_level_planner_exact_moduli_pfactor_smoke(
                    options.grouped_level_planner_config
                );
            project2::gpu_throughput_mainline::print_grouped_level_planner_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir /
                    ("project2_gpu_throughput_grouped_level_planner_exact_moduli_pfactor_" + options.output_tag +
                     ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open grouped-level-planner-exact-moduli-pfactor log output");
                }
                project2::gpu_throughput_mainline::print_grouped_level_planner_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.grouped_level_planner_exact_moduli_pq_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_grouped_level_planner_exact_moduli_pq_smoke(
                    options.grouped_level_planner_config
                );
            project2::gpu_throughput_mainline::print_grouped_level_planner_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir /
                    ("project2_gpu_throughput_grouped_level_planner_exact_moduli_pq_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open grouped-level-planner-exact-moduli-pq log output");
                }
                project2::gpu_throughput_mainline::print_grouped_level_planner_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.grouped_level_planner_exact_moduli_pqt_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_grouped_level_planner_exact_moduli_pqt_smoke(
                    options.grouped_level_planner_config
                );
            project2::gpu_throughput_mainline::print_grouped_level_planner_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir /
                    ("project2_gpu_throughput_grouped_level_planner_exact_moduli_pqt_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open grouped-level-planner-exact-moduli-pqt log output");
                }
                project2::gpu_throughput_mainline::print_grouped_level_planner_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.pi_end_to_end_smoke) {
            const auto report =
                project2::gpu_throughput_mainline::run_pi_end_to_end_smoke(options.pi_end_to_end_config);
            project2::gpu_throughput_mainline::print_pi_end_to_end_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(result_dir / ("project2_gpu_throughput_pi_end_to_end_" + options.output_tag + ".log"));
                if (!out) {
                    throw std::runtime_error("failed to open pi-end-to-end log output");
                }
                project2::gpu_throughput_mainline::print_pi_end_to_end_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        if (options.grouped_level_planner_split_mask63_smoke) {
            const auto report = project2::gpu_throughput_mainline::run_grouped_level_planner_split_mask63_smoke(
                options.grouped_level_planner_config
            );
            project2::gpu_throughput_mainline::print_grouped_level_planner_report(std::cout, report);
            if (!options.output_tag.empty()) {
                std::ofstream out(
                    result_dir /
                    ("project2_gpu_throughput_grouped_level_planner_split_mask63_" + options.output_tag + ".log")
                );
                if (!out) {
                    throw std::runtime_error("failed to open grouped-level-planner-split-mask63 log output");
                }
                project2::gpu_throughput_mainline::print_grouped_level_planner_report(out, report);
            }
            if (!report.ok) {
                exit_code = 1;
            }
        }

        return exit_code;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
