#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace project2::gpu_throughput_mainline {

enum class GroupedPlannerSourceMode {
    SyntheticModulusResidues = 0,
    ChudnovskyPFactorLeaves = 1,
};

struct ThroughputPlanReport {
    std::string workspace_name;
    std::string priority_order;
    std::string primary_layout;
    std::string primary_batch_axis;
    std::string fft_backbone_target;
    std::string merge_scheduler_target;
    std::string current_stage;
    std::string current_benchmark;
    std::string next_blocker;
    bool ok = false;
};

struct PackedPointwiseAddConfig {
    std::size_t batch_count = 4096;
    std::size_t slot_count = 4096;
    int modulus_count = 8;
    int warmup_iterations = 2;
    int measured_iterations = 20;
    std::size_t verification_sample_count = 8;
};

struct PackedPointwiseAddReport {
    std::string layout;
    std::string operation;
    std::size_t batch_count = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
    std::size_t coefficient_count = 0;
    std::size_t residue_value_count = 0;
    double cold_kernel_ms = 0.0;
    double avg_kernel_ms = 0.0;
    double residue_values_per_second = 0.0;
    double coefficients_per_second = 0.0;
    double device_bytes_per_second = 0.0;
    std::size_t verification_sample_count = 0;
    std::size_t verified_values = 0;
    bool ok = false;
};

struct DevicePairPackConfig {
    std::size_t merge_count = 4096;
    std::size_t slot_count = 4096;
    int modulus_count = 8;
    int warmup_iterations = 2;
    int measured_iterations = 20;
    std::size_t verification_sample_count = 8;
};

struct DevicePairPackReport {
    std::string source_layout;
    std::string packed_layout;
    std::string operation;
    std::size_t merge_count = 0;
    std::size_t node_count = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
    std::size_t residue_value_count = 0;
    double cold_pack_ms = 0.0;
    double avg_pack_ms = 0.0;
    double cold_unpack_ms = 0.0;
    double avg_unpack_ms = 0.0;
    double cold_roundtrip_ms = 0.0;
    double avg_roundtrip_ms = 0.0;
    double effective_residue_values_per_second = 0.0;
    double pack_device_bytes_per_second = 0.0;
    double unpack_device_bytes_per_second = 0.0;
    double roundtrip_device_bytes_per_second = 0.0;
    std::size_t verification_sample_count = 0;
    std::size_t verified_values = 0;
    bool ok = false;
};

struct PersistentLevelReduceConfig {
    std::size_t node_count = 4096;
    std::size_t slot_count = 4096;
    int modulus_count = 8;
    int warmup_iterations = 2;
    int measured_iterations = 20;
    std::size_t verification_sample_count = 8;
};

struct PersistentLevelReduceReport {
    std::string source_layout;
    std::string packed_layout;
    std::string parent_layout;
    std::string operation;
    std::size_t node_count = 0;
    std::size_t final_node_count = 0;
    int reduced_levels = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
    std::size_t total_level_input_values = 0;
    std::size_t total_parent_output_values = 0;
    double cold_pipeline_ms = 0.0;
    double avg_pipeline_ms = 0.0;
    double effective_input_values_per_second = 0.0;
    double effective_parent_values_per_second = 0.0;
    double pipeline_device_bytes_per_second = 0.0;
    std::size_t verification_sample_count = 0;
    std::size_t verified_values = 0;
    bool ok = false;
};

struct BatchedFftBackboneConfig {
    std::size_t batch_count = 2048;
    std::size_t fft_length = 4096;
    int warmup_iterations = 2;
    int measured_iterations = 20;
    std::size_t verification_sample_count = 8;
};

struct BatchedFftBackboneReport {
    std::string input_layout;
    std::string spectrum_layout;
    std::string operation;
    std::size_t batch_count = 0;
    std::size_t fft_length = 0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
    std::size_t complex_value_count = 0;
    double plan_build_ms = 0.0;
    double cold_pipeline_ms = 0.0;
    double avg_pipeline_ms = 0.0;
    double transformed_complex_values_per_second = 0.0;
    double output_complex_values_per_second = 0.0;
    double logical_pipeline_bytes_per_second = 0.0;
    std::size_t verification_sample_count = 0;
    std::size_t verified_values = 0;
    bool ok = false;
};

struct ResidueFftBridgeConfig {
    std::size_t node_count = 4096;
    std::size_t slot_count = 4096;
    int modulus_count = 8;
    int bridge_modulus_index = 0;
    std::uint32_t packing_mask = 63;
    int warmup_iterations = 2;
    int measured_iterations = 20;
    std::size_t verification_sample_count = 8;
};

struct ResidueFftBridgeReport {
    std::string source_layout;
    std::string fft_input_layout;
    std::string spectrum_layout;
    std::string operation;
    std::size_t node_count = 0;
    std::size_t merge_count = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    int bridge_modulus_index = 0;
    std::uint32_t packing_mask = 0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
    std::size_t residue_values_packed = 0;
    std::size_t complex_value_count = 0;
    double plan_build_ms = 0.0;
    double cold_pack_ms = 0.0;
    double avg_pack_ms = 0.0;
    double cold_bridge_ms = 0.0;
    double avg_bridge_ms = 0.0;
    double packed_residue_values_per_second = 0.0;
    double transformed_complex_values_per_second = 0.0;
    double logical_bridge_bytes_per_second = 0.0;
    std::size_t verification_sample_count = 0;
    std::size_t verified_values = 0;
    bool ok = false;
};

struct GroupedResidueFftBridgeConfig {
    std::size_t node_count = 4096;
    std::size_t slot_count = 4096;
    int modulus_count = 8;
    std::uint32_t packing_mask = 63;
    int warmup_iterations = 2;
    int measured_iterations = 20;
    std::size_t verification_sample_count = 8;
};

struct GroupedResidueFftBridgeReport {
    std::string source_layout;
    std::string fft_input_layout;
    std::string spectrum_layout;
    std::string operation;
    std::size_t node_count = 0;
    std::size_t merge_count = 0;
    std::size_t fft_batch_count = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    std::uint32_t packing_mask = 0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
    std::size_t residue_values_packed = 0;
    std::size_t complex_value_count = 0;
    double plan_build_ms = 0.0;
    double cold_pack_ms = 0.0;
    double avg_pack_ms = 0.0;
    double cold_bridge_ms = 0.0;
    double avg_bridge_ms = 0.0;
    double packed_residue_values_per_second = 0.0;
    double transformed_complex_values_per_second = 0.0;
    double logical_bridge_bytes_per_second = 0.0;
    std::size_t verification_sample_count = 0;
    std::size_t verified_values = 0;
    bool ok = false;
};

struct GroupedLevelPlannerConfig {
    std::size_t node_count = 4096;
    std::size_t slot_count = 4096;
    int modulus_count = 8;
    std::uint32_t packing_mask = 63;
    int warmup_iterations = 2;
    int measured_iterations = 20;
    std::size_t verification_sample_count = 8;
    GroupedPlannerSourceMode source_mode = GroupedPlannerSourceMode::SyntheticModulusResidues;
};

struct GroupedLevelPlannerReport {
    std::string source_layout;
    std::string fft_input_layout;
    std::string parent_layout;
    std::string operation;
    std::size_t node_count = 0;
    std::size_t final_node_count = 0;
    int level_count = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    std::uint32_t packing_mask = 0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
    std::size_t total_fft_batch_count = 0;
    std::size_t total_residue_values_packed = 0;
    std::size_t total_complex_value_count = 0;
    double plan_build_ms = 0.0;
    double cold_pipeline_ms = 0.0;
    double avg_pipeline_ms = 0.0;
    double packed_residue_values_per_second = 0.0;
    double transformed_complex_values_per_second = 0.0;
    double logical_pipeline_bytes_per_second = 0.0;
    std::size_t verification_sample_count = 0;
    std::size_t verified_values = 0;
    int split_limb_count = 0;
    int split_pass_count = 0;
    std::size_t verification_mismatch_count = 0;
    double max_projection_real_error = 0.0;
    std::size_t first_mismatch_modulus = 0;
    std::size_t first_mismatch_merge = 0;
    std::size_t first_mismatch_output = 0;
    std::uint32_t first_mismatch_expected = 0;
    std::uint32_t first_mismatch_observed = 0;
    bool ok = false;
};

struct PiEndToEndSmokeConfig {
    std::size_t term_count = 16;
    std::size_t slot_count = 256;
    int modulus_count = 10;
    int forced_closure_modulus_count = 0;
    std::size_t target_digits = 50;
    std::size_t report_decimal_digits = 4096;
    int warmup_iterations = 1;
    int measured_iterations = 3;
};

struct PiEndToEndSmokeReport {
    std::size_t term_count = 0;
    std::size_t target_digits = 0;
    std::size_t working_digits = 0;
    std::size_t required_terms = 0;
    std::size_t slot_count = 0;
    int modulus_count = 0;
    int effective_closure_modulus_count = 0;
    std::size_t required_closure_half_range_bits = 0;
    std::size_t crt_product_bits = 0;
    std::size_t crt_half_range_bits = 0;
    std::size_t closure_modulus_headroom_bits = 0;
    double planner_plan_build_ms = 0.0;
    double planner_cold_pipeline_ms = 0.0;
    double planner_avg_pipeline_ms = 0.0;
    double planner_packed_residue_values_per_second = 0.0;
    double cuda_runtime_init_ms = 0.0;
    double closure_setup_ms = 0.0;
    double closure_warmup_total_ms = 0.0;
    double closure_measured_total_ms = 0.0;
    double closure_wall_ms = 0.0;
    double root_rebuild_ms = 0.0;
    double exact_root_reference_ms = 0.0;
    double final_host_tail_ms = 0.0;
    double total_smoke_ms = 0.0;
    double steady_state_pi_result_ms = 0.0;
    double steady_state_pi_digits_per_second = 0.0;
    double cold_process_pi_digits_per_second = 0.0;
    std::size_t root_p_bits = 0;
    std::size_t root_q_bits = 0;
    std::size_t root_t_bits = 0;
    std::size_t reference_prefix_digits_checked = 0;
    std::size_t reported_decimal_digits = 0;
    std::string closure_mode;
    std::string pi_decimal;
    bool term_count_sufficient = false;
    bool root_reconstruction_match = false;
    bool prefix_match = false;
    bool pi_decimal_truncated = false;
    bool ok = false;
};

ThroughputPlanReport plan_throughput_mainline();
void print_throughput_plan_report(std::ostream& out, const ThroughputPlanReport& report);

PackedPointwiseAddReport run_packed_pointwise_add_smoke(const PackedPointwiseAddConfig& config);
void print_packed_pointwise_add_report(std::ostream& out, const PackedPointwiseAddReport& report);

DevicePairPackReport run_device_pair_pack_smoke(const DevicePairPackConfig& config);
void print_device_pair_pack_report(std::ostream& out, const DevicePairPackReport& report);

PersistentLevelReduceReport run_persistent_level_reduce_smoke(const PersistentLevelReduceConfig& config);
void print_persistent_level_reduce_report(std::ostream& out, const PersistentLevelReduceReport& report);

BatchedFftBackboneReport run_batched_fft_backbone_smoke(const BatchedFftBackboneConfig& config);
void print_batched_fft_backbone_report(std::ostream& out, const BatchedFftBackboneReport& report);

ResidueFftBridgeReport run_residue_fft_bridge_smoke(const ResidueFftBridgeConfig& config);
void print_residue_fft_bridge_report(std::ostream& out, const ResidueFftBridgeReport& report);

GroupedResidueFftBridgeReport run_grouped_residue_fft_bridge_smoke(const GroupedResidueFftBridgeConfig& config);
void print_grouped_residue_fft_bridge_report(std::ostream& out, const GroupedResidueFftBridgeReport& report);

GroupedLevelPlannerReport run_grouped_level_planner_smoke(const GroupedLevelPlannerConfig& config);
GroupedLevelPlannerReport run_grouped_level_planner_multilimb_smoke(const GroupedLevelPlannerConfig& config);
GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_smoke(const GroupedLevelPlannerConfig& config);
GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_pfactor_smoke(const GroupedLevelPlannerConfig& config);
GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_pq_smoke(const GroupedLevelPlannerConfig& config);
GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_pqt_smoke(const GroupedLevelPlannerConfig& config);
GroupedLevelPlannerReport run_grouped_level_planner_split_mask63_smoke(const GroupedLevelPlannerConfig& config);
void print_grouped_level_planner_report(std::ostream& out, const GroupedLevelPlannerReport& report);

PiEndToEndSmokeReport run_pi_end_to_end_smoke(const PiEndToEndSmokeConfig& config);
void print_pi_end_to_end_report(std::ostream& out, const PiEndToEndSmokeReport& report);

}  // namespace project2::gpu_throughput_mainline
