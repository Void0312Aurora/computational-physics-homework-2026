#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "canonical_ops.hpp"
#include "rns_ntt.hpp"

namespace project2::nextgen_cpu {

enum class TreeMultiplyMode : std::uint8_t {
    schoolbook = 0,
    adaptive_rns = 1,
};

struct ProductTreeConfig {
    std::size_t page_size_blocks = 4096;
    std::size_t rns_threshold_blocks = 1024;
};

struct ProductTreeStats {
    std::size_t leaf_count = 0;
    std::size_t levels = 0;
    std::size_t peak_live_blocks = 0;
    std::size_t peak_live_pages = 0;
    std::size_t schoolbook_mul_calls = 0;
    std::size_t rns_mul_calls = 0;
    std::size_t copy_passthroughs = 0;
    std::size_t final_blocks = 0;
};

struct ProductTreeLevel {
    explicit ProductTreeLevel(std::size_t page_size_blocks, std::size_t reserve_pages)
        : arena(page_size_blocks, reserve_pages) {}

    ProductTreeLevel(ProductTreeLevel&&) = default;
    ProductTreeLevel& operator=(ProductTreeLevel&&) = default;
    ProductTreeLevel(const ProductTreeLevel&) = delete;
    ProductTreeLevel& operator=(const ProductTreeLevel&) = delete;

    PageArena arena;
    std::vector<BigHandle> nodes;
};

struct ProductTreeResult {
    ProductTreeLevel level;
    ProductTreeStats stats;
};

inline std::size_t ceil_div(std::size_t numerator, std::size_t denominator) {
    return (numerator + denominator - 1U) / denominator;
}

inline std::size_t level_capacity_estimate_blocks(const std::vector<BigHandle>& nodes) {
    std::size_t total_blocks = 0;
    for (std::size_t index = 0; index < nodes.size(); index += 2U) {
        if (index + 1U < nodes.size()) {
            total_blocks += nodes[index].used_blocks + nodes[index + 1U].used_blocks;
        } else {
            total_blocks += std::max<std::size_t>(nodes[index].used_blocks, 1U);
        }
    }
    return std::max<std::size_t>(total_blocks, 1U);
}

inline ProductTreeLevel make_leaf_level(
    const std::vector<std::vector<block_t>>& leaf_blocks,
    const ProductTreeConfig& config
) {
    std::size_t total_blocks = 0;
    for (const auto& leaf : leaf_blocks) {
        total_blocks += std::max<std::size_t>(leaf.size(), 1U);
    }
    const std::size_t reserve_pages = std::max<std::size_t>(
        1U,
        ceil_div(total_blocks, std::max<std::size_t>(config.page_size_blocks, 1U))
    );
    ProductTreeLevel level(std::max<std::size_t>(config.page_size_blocks, 1U), reserve_pages);
    level.nodes.reserve(leaf_blocks.size());
    for (const auto& leaf : leaf_blocks) {
        BigHandle handle = level.arena.allocate(std::max<std::size_t>(leaf.size(), 1U));
        assign_blocks(&level.arena, &handle, leaf);
        level.nodes.push_back(handle);
    }
    return level;
}

inline void update_live_peak(
    ProductTreeStats* stats,
    const ProductTreeLevel& current_level,
    const ProductTreeLevel& next_level
) {
    stats->peak_live_blocks = std::max(
        stats->peak_live_blocks,
        current_level.arena.blocks_in_use() + next_level.arena.blocks_in_use()
    );
    stats->peak_live_pages = std::max(
        stats->peak_live_pages,
        current_level.arena.pages_in_use() + next_level.arena.pages_in_use()
    );
}

inline bool should_use_rns_tree(
    const BigHandle& left,
    const BigHandle& right,
    TreeMultiplyMode mode,
    const ProductTreeConfig& config
) {
    if (mode != TreeMultiplyMode::adaptive_rns) {
        return false;
    }
    return std::max(left.used_blocks, right.used_blocks) >= config.rns_threshold_blocks;
}

inline ProductTreeResult reduce_product_tree(
    const std::vector<std::vector<block_t>>& leaf_blocks,
    TreeMultiplyMode mode,
    const ProductTreeConfig& config
) {
    if (leaf_blocks.empty()) {
        throw std::runtime_error("product tree requires at least one leaf");
    }

    ProductTreeLevel current_level = make_leaf_level(leaf_blocks, config);
    ProductTreeStats stats{
        .leaf_count = leaf_blocks.size(),
        .levels = 0,
        .peak_live_blocks = current_level.arena.blocks_in_use(),
        .peak_live_pages = current_level.arena.pages_in_use(),
    };
    RnsWorkspace workspace;

    while (current_level.nodes.size() > 1U) {
        const std::size_t next_level_blocks = level_capacity_estimate_blocks(current_level.nodes);
        const std::size_t reserve_pages = std::max<std::size_t>(
            1U,
            ceil_div(next_level_blocks, std::max<std::size_t>(config.page_size_blocks, 1U))
        );
        ProductTreeLevel next_level(std::max<std::size_t>(config.page_size_blocks, 1U), reserve_pages);
        next_level.nodes.reserve((current_level.nodes.size() + 1U) / 2U);

        for (std::size_t index = 0; index < current_level.nodes.size(); index += 2U) {
            if (index + 1U < current_level.nodes.size()) {
                const BigHandle& left = current_level.nodes[index];
                const BigHandle& right = current_level.nodes[index + 1U];
                BigHandle out = next_level.arena.allocate(std::max<std::size_t>(left.used_blocks + right.used_blocks, 1U));
                if (should_use_rns_tree(left, right, mode, config)) {
                    mul_into_rns_ntt(
                        &next_level.arena,
                        &out,
                        current_level.arena,
                        left,
                        current_level.arena,
                        right,
                        &workspace
                    );
                    ++stats.rns_mul_calls;
                } else {
                    mul_into(
                        &next_level.arena,
                        &out,
                        current_level.arena,
                        left,
                        current_level.arena,
                        right
                    );
                    ++stats.schoolbook_mul_calls;
                }
                next_level.nodes.push_back(out);
            } else {
                const BigHandle& src = current_level.nodes[index];
                BigHandle out = next_level.arena.allocate(std::max<std::size_t>(src.used_blocks, 1U));
                copy_into(&next_level.arena, &out, current_level.arena, src);
                next_level.nodes.push_back(out);
                ++stats.copy_passthroughs;
            }
            update_live_peak(&stats, current_level, next_level);
        }

        current_level = std::move(next_level);
        ++stats.levels;
        stats.peak_live_blocks = std::max(stats.peak_live_blocks, current_level.arena.blocks_in_use());
        stats.peak_live_pages = std::max(stats.peak_live_pages, current_level.arena.pages_in_use());
    }

    stats.final_blocks = current_level.nodes.front().used_blocks;
    return ProductTreeResult{
        .level = std::move(current_level),
        .stats = stats,
    };
}

}  // namespace project2::nextgen_cpu
