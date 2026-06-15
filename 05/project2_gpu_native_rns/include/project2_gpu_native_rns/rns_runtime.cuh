#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

namespace project2::gpu_native_rns {

enum class VerifyMode {
    kNone,
    kSampled,
    kFull,
};

struct ModulusConfig {
    static constexpr int kDefaultModulusCount = 3;
    static constexpr int kIntermediateModulusCount = 7;
    static constexpr int kModulusCount = 10;
    static constexpr std::array<std::uint32_t, kModulusCount> kModuli = {
        2013265921u,
        1811939329u,
        469762049u,
        2113929217u,
        1711276033u,
        1224736769u,
        1107296257u,
        2130706433u,
        998244353u,
        754974721u,
    };
};

struct DeviceRnsTensor {
    std::size_t value_count = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    std::uint32_t* d_moduli = nullptr;
    std::uint32_t* d_residues = nullptr;
    std::int8_t* d_signs = nullptr;
    std::uint32_t* d_logical_slots = nullptr;
    std::uint32_t* d_scale_bits = nullptr;
    std::uint32_t* d_levels = nullptr;
};

struct HostRnsTensor {
    std::size_t value_count = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    std::vector<std::uint32_t> moduli;
    std::vector<std::uint32_t> residues;
    std::vector<std::int8_t> signs;
    std::vector<std::uint32_t> logical_slots;
    std::vector<std::uint32_t> scale_bits;
    std::vector<std::uint32_t> levels;
};

struct SmokeConfig {
    std::size_t value_count = 256;
    std::size_t slot_count = 4;
    int measured_iterations = 5;
    int input_bits = 0;
    VerifyMode verify_mode = VerifyMode::kFull;
    std::size_t verify_sample_count = 64;
};

struct SmokeReport {
    std::string layout;
    std::string convolution_algorithm;
    std::size_t modulus_count = 0;
    std::size_t value_count = 0;
    std::size_t slot_count = 0;
    std::size_t convolution_ntt_size = 0;
    int input_bits = 0;
    bool verification_enabled = true;
    std::string input_staging_mode;
    double one_time_input_upload_ms = 0.0;
    std::string verification_mode;
    std::size_t verification_sample_count = 0;
    std::size_t checked_scalar_slots = 0;
    std::size_t checked_convolution_coefficients = 0;
    std::size_t validated_scalar_slots = 0;
    std::size_t validated_convolution_coefficients = 0;
    int measured_iterations = 0;
    double cold_kernel_ms = 0.0;
    double cold_download_reconstruct_ms = 0.0;
    double cold_end_to_end_ms = 0.0;
    double avg_encode_ms = 0.0;
    double avg_pointwise_ms = 0.0;
    double avg_convolution_ms = 0.0;
    double avg_kernel_ms = 0.0;
    double avg_download_reconstruct_ms = 0.0;
    double avg_end_to_end_ms = 0.0;
    double avg_scalar_slots_per_second_e2e = 0.0;
    double avg_convolution_coefficients_per_second_kernel = 0.0;
    double avg_pipeline_residue_values_per_second_kernel = 0.0;
    double avg_download_over_kernel_ratio = 0.0;
    bool ok = false;
};

struct PiRouteReport {
    std::size_t target_digits = 0;
    std::size_t guard_digits = 0;
    std::size_t working_digits = 0;
    std::size_t working_bits = 0;
    std::size_t chudnovsky_terms = 0;
    std::size_t binary_split_leaf_count = 0;
    std::size_t binary_split_internal_count = 0;
    std::size_t binary_split_depth = 0;
    std::size_t final_nonmultiply_steps = 0;
    std::size_t estimated_decimal_output_chars = 0;
    std::string chosen_route;
    std::string preferred_multiply_backbone;
    std::string final_reciprocal_strategy;
    std::string final_sqrt_strategy;
    std::string rejected_route;
    std::string rejection_reason;
    std::string route_rationale;
    bool ok = false;
};

struct PiExecutionLevelReport {
    std::size_t level_index = 0;
    std::size_t node_count = 0;
    std::size_t max_terms_per_node = 0;
    std::size_t estimated_integer_bits = 0;
    std::size_t estimated_slot_count = 0;
    std::size_t estimated_ntt_size = 0;
    std::size_t safe_limb_bits = 0;
};

struct PiExecutionPlanReport {
    std::size_t target_digits = 0;
    std::size_t guard_digits = 0;
    std::size_t working_digits = 0;
    std::size_t working_bits = 0;
    std::size_t chudnovsky_terms = 0;
    std::size_t leaf_terms_per_task = 0;
    std::size_t leaf_task_count = 0;
    std::size_t merge_level_count = 0;
    std::size_t target_parallel_leaf_tasks = 0;
    std::size_t modulus_dynamic_range_bits = 0;
    std::size_t chosen_limb_bits = 0;
    std::size_t peak_slot_count = 0;
    std::size_t peak_ntt_size = 0;
    std::size_t root_safe_limb_bits = 0;
    std::string chosen_route;
    std::string execution_model;
    std::string peak_bottleneck;
    std::string plan_rationale;
    std::vector<PiExecutionLevelReport> levels;
    bool ok = false;
};

struct PiPqtNodeReport {
    std::size_t node_index = 0;
    std::size_t begin_term = 0;
    std::size_t end_term = 0;
    std::size_t depth = 0;
    std::size_t left_child = std::numeric_limits<std::size_t>::max();
    std::size_t right_child = std::numeric_limits<std::size_t>::max();
    bool is_leaf = false;
    bool t_negative = false;
    std::size_t p_bits = 0;
    std::size_t q_bits = 0;
    std::size_t t_bits = 0;
    std::size_t slot_count = 0;
    std::size_t ntt_size = 0;
};

struct PiPqtTreeReport {
    std::size_t term_count = 0;
    std::size_t node_count = 0;
    std::size_t leaf_count = 0;
    std::size_t merge_count = 0;
    std::size_t max_depth = 0;
    std::size_t chosen_limb_bits = 0;
    std::size_t root_p_bits = 0;
    std::size_t root_q_bits = 0;
    std::size_t root_t_bits = 0;
    std::size_t root_slot_count = 0;
    std::size_t root_ntt_size = 0;
    bool root_t_negative = false;
    std::string recurrence;
    std::string merge_formula;
    std::vector<std::size_t> depth_node_counts;
    std::vector<PiPqtNodeReport> nodes;
    bool ok = false;
};

struct PiPqtTensorReport {
    std::size_t term_count = 0;
    std::size_t merge_begin_term = 0;
    std::size_t merge_end_term = 0;
    std::size_t left_begin_term = 0;
    std::size_t left_end_term = 0;
    std::size_t right_begin_term = 0;
    std::size_t right_end_term = 0;
    std::size_t chosen_limb_bits = 0;
    std::size_t p_left_slot_count = 0;
    std::size_t p_right_slot_count = 0;
    std::size_t q_left_slot_count = 0;
    std::size_t q_right_slot_count = 0;
    std::size_t t_left_slot_count = 0;
    std::size_t t_right_slot_count = 0;
    std::size_t p_output_slot_count = 0;
    std::size_t q_output_slot_count = 0;
    std::size_t t_output_slot_count = 0;
    std::size_t p_output_ntt_size = 0;
    std::size_t q_output_ntt_size = 0;
    std::size_t t_output_ntt_size = 0;
    std::size_t modulus_count = 0;
    bool uses_signed_residue_encoding = false;
    bool metadata_ok = false;
    bool p_match = false;
    bool q_match = false;
    bool t_match = false;
    bool ok = false;
};

struct PiPqtTensorTreeReport {
    std::size_t term_count = 0;
    std::size_t node_count = 0;
    std::size_t leaf_count = 0;
    std::size_t merge_count = 0;
    std::size_t root_begin_term = 0;
    std::size_t root_end_term = 0;
    std::size_t chosen_limb_bits = 0;
    std::size_t peak_p_slot_count = 0;
    std::size_t peak_q_slot_count = 0;
    std::size_t peak_t_slot_count = 0;
    std::size_t peak_ntt_size = 0;
    std::size_t modulus_count = 0;
    std::size_t root_p_slot_count = 0;
    std::size_t root_q_slot_count = 0;
    std::size_t root_t_slot_count = 0;
    bool uses_signed_residue_encoding = false;
    bool metadata_ok = false;
    bool p_match = false;
    bool q_match = false;
    bool t_match = false;
    bool ok = false;
};

struct PiEndToEndReport {
    std::size_t term_count = 0;
    std::size_t target_digits = 0;
    std::size_t working_digits = 0;
    std::size_t required_terms = 0;
    std::size_t reference_prefix_digits_checked = 0;
    bool term_count_sufficient = false;
    std::size_t chosen_limb_bits = 0;
    std::size_t modulus_count = 0;
    std::size_t peak_ntt_size = 0;
    std::size_t root_q_bits = 0;
    std::size_t root_t_bits = 0;
    std::string closure_mode;
    std::string pi_decimal;
    bool prefix_match = false;
    bool ok = false;
};

struct PiBenchmarkReport {
    std::size_t term_count = 0;
    std::size_t target_digits = 0;
    std::size_t working_digits = 0;
    std::size_t required_terms = 0;
    std::size_t reference_prefix_digits_checked = 0;
    bool term_count_sufficient = false;
    std::size_t chosen_limb_bits = 0;
    std::size_t modulus_count = 0;
    std::size_t peak_ntt_size = 0;
    std::size_t root_q_bits = 0;
    std::size_t root_t_bits = 0;
    int measured_iterations = 0;
    std::string closure_mode;
    std::string tree_validation_mode;
    std::string pi_decimal;
    double cold_tree_execution_ms = 0.0;
    double cold_host_closure_ms = 0.0;
    double cold_end_to_end_ms = 0.0;
    double avg_tree_execution_ms = 0.0;
    double avg_host_closure_ms = 0.0;
    double avg_end_to_end_ms = 0.0;
    double avg_digits_per_second_tree_stage = 0.0;
    double avg_digits_per_second_closure_stage = 0.0;
    double avg_digits_per_second_e2e = 0.0;
    bool prefix_match = false;
    bool ok = false;
};

DeviceRnsTensor allocate_device_tensor(
    std::size_t value_count,
    std::size_t slot_count,
    int modulus_count = ModulusConfig::kDefaultModulusCount
);
DeviceRnsTensor allocate_polynomial_block_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    int modulus_count = ModulusConfig::kDefaultModulusCount
);
DeviceRnsTensor make_scaled_constant_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    std::uint64_t coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t scale_bits,
    int modulus_count = ModulusConfig::kDefaultModulusCount
);
DeviceRnsTensor make_reciprocal_seed_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    std::uint64_t denominator_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t denominator_scale_bits,
    std::uint32_t target_product_scale_bits,
    int modulus_count = ModulusConfig::kDefaultModulusCount
);
DeviceRnsTensor make_sqrt_seed_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    std::uint64_t radicand_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t radicand_scale_bits,
    std::uint32_t target_sqrt_scale_bits,
    int modulus_count = ModulusConfig::kDefaultModulusCount
);
DeviceRnsTensor make_division_quotient_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    std::uint64_t numerator_coefficient,
    std::uint64_t denominator_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t numerator_scale_bits,
    std::uint32_t denominator_scale_bits,
    std::uint32_t target_product_scale_bits,
    int modulus_count = ModulusConfig::kDefaultModulusCount
);
void free_device_tensor(DeviceRnsTensor& tensor);

void encode_u64(DeviceRnsTensor& tensor, const std::vector<std::uint64_t>& values);
void encode_polynomial_blocks(DeviceRnsTensor& tensor, const std::vector<std::uint64_t>& coefficients);
void encode_scaled_constant_blocks(
    DeviceRnsTensor& tensor,
    std::uint64_t coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t scale_bits
);
void encode_reciprocal_seed_blocks(
    DeviceRnsTensor& tensor,
    std::uint64_t denominator_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t denominator_scale_bits,
    std::uint32_t target_product_scale_bits
);
void encode_sqrt_seed_blocks(
    DeviceRnsTensor& tensor,
    std::uint64_t radicand_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t radicand_scale_bits,
    std::uint32_t target_sqrt_scale_bits
);
void encode_division_quotient_blocks(
    DeviceRnsTensor& tensor,
    std::uint64_t numerator_coefficient,
    std::uint64_t denominator_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t numerator_scale_bits,
    std::uint32_t denominator_scale_bits,
    std::uint32_t target_product_scale_bits
);
void set_uniform_tensor_metadata(
    DeviceRnsTensor& tensor,
    std::int8_t sign,
    std::uint32_t logical_slots,
    std::uint32_t scale_bits,
    std::uint32_t level
);
void drop_moduli_prefix(const DeviceRnsTensor& src, DeviceRnsTensor& dst, std::uint32_t scale_bits_delta = 0);

void pointwise_add(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out);
void pointwise_sub(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out);
void pointwise_mul(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out);
void pairwise_convolution(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out);
void add_polynomial_blocks(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out);
void sub_polynomial_blocks(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out);
void multiply_polynomial_blocks(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out);
void convolve_polynomial_blocks(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out);
void compute_residual_correction(
    const DeviceRnsTensor& target,
    const DeviceRnsTensor& approximate,
    DeviceRnsTensor& residual,
    DeviceRnsTensor& corrected
);

HostRnsTensor download_tensor(const DeviceRnsTensor& tensor);
std::vector<std::uint64_t> reconstruct_scalars(const HostRnsTensor& tensor);
std::string describe_layout(const DeviceRnsTensor& tensor);
bool lifecycle_block_smoke_test(std::ostream& out);
bool scaled_constant_smoke_test(std::ostream& out);
bool reciprocal_seed_smoke_test(std::ostream& out);
bool sqrt_seed_smoke_test(std::ostream& out);
bool division_smoke_test(std::ostream& out);
bool correction_domain_smoke_test(std::ostream& out);
PiRouteReport plan_pi_route(std::size_t target_digits);
void print_pi_route_report(std::ostream& out, const PiRouteReport& report);
bool pi_route_smoke_test(std::ostream& out, std::size_t target_digits);
PiExecutionPlanReport plan_pi_execution(std::size_t target_digits);
void print_pi_execution_plan_report(std::ostream& out, const PiExecutionPlanReport& report);
bool pi_execution_plan_smoke_test(std::ostream& out, std::size_t target_digits);
PiPqtTreeReport plan_pi_pqt_tree(std::size_t term_count);
void print_pi_pqt_tree_report(std::ostream& out, const PiPqtTreeReport& report);
bool pi_pqt_smoke_test(std::ostream& out, std::size_t term_count);
PiPqtTensorReport run_pi_pqt_tensor_smoke(std::size_t term_count);
void print_pi_pqt_tensor_report(std::ostream& out, const PiPqtTensorReport& report);
bool pi_pqt_tensor_smoke_test(std::ostream& out, std::size_t term_count);
PiPqtTensorTreeReport run_pi_pqt_tensor_tree_smoke(std::size_t term_count);
void print_pi_pqt_tensor_tree_report(std::ostream& out, const PiPqtTensorTreeReport& report);
bool pi_pqt_tensor_tree_smoke_test(std::ostream& out, std::size_t term_count);
PiEndToEndReport run_pi_end_to_end_smoke(std::size_t term_count, std::size_t target_digits);
void print_pi_end_to_end_report(std::ostream& out, const PiEndToEndReport& report);
bool pi_end_to_end_smoke_test(std::ostream& out, std::size_t term_count, std::size_t target_digits);
PiBenchmarkReport run_pi_end_to_end_benchmark(
    std::size_t term_count,
    std::size_t target_digits,
    int measured_iterations
);
void print_pi_end_to_end_benchmark_report(std::ostream& out, const PiBenchmarkReport& report);
void write_pi_end_to_end_benchmark_csv(const PiBenchmarkReport& report, const std::string& path);
bool pi_end_to_end_benchmark_test(
    std::ostream& out,
    std::size_t term_count,
    std::size_t target_digits,
    int measured_iterations
);

SmokeReport run_smoke_test(const SmokeConfig& config);
SmokeReport run_smoke_test();
void print_smoke_report(std::ostream& out, const SmokeReport& report);
void write_smoke_csv(const SmokeReport& report, const std::string& path);
bool smoke_test(std::ostream& out);

}  // namespace project2::gpu_native_rns
