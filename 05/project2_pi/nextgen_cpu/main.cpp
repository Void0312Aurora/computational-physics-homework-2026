#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

#include "canonical_ops.hpp"
#include "product_tree.hpp"
#include "rns_ntt.hpp"

namespace project2::nextgen_cpu {

namespace {

std::uint64_t checked_to_u64(const PageArena& arena, const BigHandle& handle) {
    return to_u64(arena, handle);
}

}  // namespace

}  // namespace project2::nextgen_cpu

int main() {
    using namespace project2::nextgen_cpu;

    try {
        PageArena arena(/*page_size_blocks=*/64, /*reserve_pages=*/2);

        const auto rewind_marker = arena.marker();
        BigHandle rewind_probe = arena.allocate(24);
        arena.zero_fill(rewind_probe);
        arena.rewind(rewind_marker);
        const std::size_t blocks_after_rewind = arena.blocks_in_use();

        TripleSlot left{
            .p = arena.allocate(4),
            .q = arena.allocate(4),
            .t = arena.allocate(4),
        };
        TripleSlot right{
            .p = arena.allocate(4),
            .q = arena.allocate(4),
            .t = arena.allocate(4),
        };
        TripleSlot merged{
            .p = arena.allocate(8),
            .q = arena.allocate(8),
            .t = arena.allocate(8),
        };
        BigHandle temp_left = arena.allocate(8);
        BigHandle temp_right = arena.allocate(8);
        BigHandle rns_lhs = arena.allocate(8);
        BigHandle rns_rhs = arena.allocate(8);
        BigHandle schoolbook_product = arena.allocate(16);
        BigHandle rns_product = arena.allocate(16);

        constexpr std::uint64_t left_p = 1'234'567ULL;
        constexpr std::uint64_t left_q = 7'654'321ULL;
        constexpr std::uint64_t left_t = 3'141'592ULL;
        constexpr std::uint64_t right_p = 2'345'678ULL;
        constexpr std::uint64_t right_q = 8'765'432ULL;
        constexpr std::uint64_t right_t = 2'718'281ULL;

        assign_u64(&arena, &left.p, left_p);
        assign_u64(&arena, &left.q, left_q);
        assign_u64(&arena, &left.t, left_t);
        assign_u64(&arena, &right.p, right_p);
        assign_u64(&arena, &right.q, right_q);
        assign_u64(&arena, &right.t, right_t);

        mul_into(&arena, &merged.p, left.p, right.p);
        mul_into(&arena, &merged.q, left.q, right.q);
        affine_t_into(&arena, &merged.t, &temp_left, &temp_right, left.t, right.q, left.p, right.t);

        const std::uint64_t merged_p = checked_to_u64(arena, merged.p);
        const std::uint64_t merged_q = checked_to_u64(arena, merged.q);
        const std::uint64_t merged_t = checked_to_u64(arena, merged.t);

        const std::uint64_t expected_p = left_p * right_p;
        const std::uint64_t expected_q = left_q * right_q;
        const std::uint64_t expected_t = left_t * right_q + left_p * right_t;

        assign_blocks(
            &arena,
            &rns_lhs,
            std::vector<block_t>{
                0x89abcdefU,
                0x01234567U,
                0x0fedcba9U,
            }
        );
        assign_blocks(
            &arena,
            &rns_rhs,
            std::vector<block_t>{
                0x13579bdfU,
                0x2468ace0U,
                0x11223344U,
            }
        );
        mul_into(&arena, &schoolbook_product, rns_lhs, rns_rhs);
        const RnsMultiplyStats rns_stats = mul_into_rns_ntt(&arena, &rns_product, rns_lhs, rns_rhs);

        const std::vector<std::vector<block_t>> tree_leaves{
            {0x13579bdfU, 0x2468ace0U},
            {0x89abcdefU, 0x01234567U},
            {0x0fedcba9U, 0x55667788U},
            {0x11223344U, 0xaabbccddU},
        };
        const ProductTreeConfig tree_config{
            .page_size_blocks = 64U,
            .rns_threshold_blocks = 2U,
        };
        ProductTreeResult tree_schoolbook = reduce_product_tree(tree_leaves, TreeMultiplyMode::schoolbook, tree_config);
        ProductTreeResult tree_adaptive = reduce_product_tree(tree_leaves, TreeMultiplyMode::adaptive_rns, tree_config);

        const bool ok =
            merged_p == expected_p
            && merged_q == expected_q
            && merged_t == expected_t
            && blocks_after_rewind == 0
            && left.p.domain == Domain::canonical
            && merged.t.domain == Domain::canonical
            && equals(arena, schoolbook_product, rns_product)
            && equals(
                tree_schoolbook.level.arena,
                tree_schoolbook.level.nodes.front(),
                tree_adaptive.level.arena,
                tree_adaptive.level.nodes.front()
            )
            && tree_adaptive.stats.rns_mul_calls != 0;

        std::cout
            << "nextgen_cpu_smoke_status=" << (ok ? "ok" : "mismatch")
            << " engine=page_block_phase2"
            << " runtime_uses_gmp=False"
            << " runtime_uses_mpn=False"
            << " page_size_blocks=" << arena.page_size_blocks()
            << " pages_in_use=" << arena.pages_in_use()
            << " blocks_in_use=" << arena.blocks_in_use()
            << " blocks_after_rewind=" << blocks_after_rewind
            << " merged_p=" << merged_p
            << " merged_q=" << merged_q
            << " merged_t=" << merged_t
            << " rns_ntt_size=" << rns_stats.ntt_size
            << " rns_modulus_count=" << rns_stats.modulus_count
            << " rns_radix_bits=" << rns_stats.radix_bits
            << " rns_input_digits=" << rns_stats.input_digits
            << " rns_output_digits=" << rns_stats.output_digits
            << " rns_output_blocks=" << rns_product.used_blocks
            << " tree_levels=" << tree_adaptive.stats.levels
            << " tree_rns_mul_calls=" << tree_adaptive.stats.rns_mul_calls
            << " tree_peak_live_blocks=" << tree_adaptive.stats.peak_live_blocks
            << " domain=" << domain_name(merged.t.domain)
            << "\n";

        return ok ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
