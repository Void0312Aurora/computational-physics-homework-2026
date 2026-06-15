#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "page_arena.hpp"

namespace project2::nextgen_cpu {

inline void require_canonical_non_negative(const BigHandle& handle) {
    if (handle.domain != Domain::canonical) {
        throw std::runtime_error("canonical ops require canonical domain handles");
    }
    if (handle.sign == Sign::negative) {
        throw std::runtime_error("phase-1 canonical ops do not support negative values");
    }
}

inline void trim(PageArena* arena, BigHandle* handle) {
    require_canonical_non_negative(*handle);
    block_t* blocks = arena->data(*handle);
    while (handle->used_blocks > 0 && blocks[handle->used_blocks - 1] == 0) {
        --handle->used_blocks;
    }
    handle->sign = (handle->used_blocks == 0) ? Sign::zero : Sign::positive;
}

inline void clear(PageArena* arena, BigHandle* handle) {
    arena->zero_fill(*handle);
    handle->used_blocks = 0;
    handle->sign = Sign::zero;
}

inline void assign_u64(PageArena* arena, BigHandle* handle, std::uint64_t value) {
    require_canonical_non_negative(*handle);
    clear(arena, handle);
    block_t* blocks = arena->data(*handle);
    if (value == 0) {
        return;
    }
    if (handle->capacity_blocks < 2 && (value >> 32) != 0) {
        throw std::runtime_error("BigHandle capacity is too small for assign_u64");
    }
    blocks[0] = static_cast<block_t>(value & 0xffffffffULL);
    handle->used_blocks = 1;
    if ((value >> 32) != 0) {
        blocks[1] = static_cast<block_t>(value >> 32);
        handle->used_blocks = 2;
    }
    handle->sign = Sign::positive;
}

inline void assign_blocks(PageArena* arena, BigHandle* handle, const std::vector<block_t>& blocks_in_least_significant_first) {
    require_canonical_non_negative(*handle);
    clear(arena, handle);
    if (blocks_in_least_significant_first.size() > handle->capacity_blocks) {
        throw std::runtime_error("BigHandle capacity is too small for assign_blocks");
    }
    block_t* blocks = arena->data(*handle);
    for (std::size_t index = 0; index < blocks_in_least_significant_first.size(); ++index) {
        blocks[index] = blocks_in_least_significant_first[index];
    }
    handle->used_blocks = blocks_in_least_significant_first.size();
    trim(arena, handle);
}

inline std::uint64_t to_u64(const PageArena& arena, const BigHandle& handle) {
    require_canonical_non_negative(handle);
    if (handle.used_blocks > 2) {
        throw std::runtime_error("BigHandle does not fit into uint64_t");
    }
    const block_t* blocks = arena.data(handle);
    std::uint64_t value = 0;
    if (handle.used_blocks >= 1) {
        value = blocks[0];
    }
    if (handle.used_blocks == 2) {
        value |= static_cast<std::uint64_t>(blocks[1]) << 32;
    }
    return value;
}

inline bool equals(
    const PageArena& left_arena,
    const BigHandle& left,
    const PageArena& right_arena,
    const BigHandle& right
) {
    require_canonical_non_negative(left);
    require_canonical_non_negative(right);
    if (left.sign != right.sign || left.used_blocks != right.used_blocks) {
        return false;
    }
    const block_t* lhs = left_arena.data(left);
    const block_t* rhs = right_arena.data(right);
    for (std::size_t index = 0; index < left.used_blocks; ++index) {
        if (lhs[index] != rhs[index]) {
            return false;
        }
    }
    return true;
}

inline bool equals(const PageArena& arena, const BigHandle& left, const BigHandle& right) {
    return equals(arena, left, arena, right);
}

inline void copy_into(PageArena* dst_arena, BigHandle* dst, const PageArena& src_arena, const BigHandle& src) {
    require_canonical_non_negative(*dst);
    require_canonical_non_negative(src);
    if (dst->capacity_blocks < std::max<std::size_t>(src.used_blocks, 1U)) {
        throw std::runtime_error("copy_into destination capacity is too small");
    }

    clear(dst_arena, dst);
    const block_t* src_blocks = src_arena.data(src);
    block_t* dst_blocks = dst_arena->data(*dst);
    for (std::size_t index = 0; index < src.used_blocks; ++index) {
        dst_blocks[index] = src_blocks[index];
    }
    dst->used_blocks = src.used_blocks;
    dst->sign = src.sign;
}

inline void add_into(
    PageArena* dst_arena,
    BigHandle* dst,
    const PageArena& left_arena,
    const BigHandle& left,
    const PageArena& right_arena,
    const BigHandle& right
) {
    require_canonical_non_negative(*dst);
    require_canonical_non_negative(left);
    require_canonical_non_negative(right);

    const std::size_t output_blocks = std::max(left.used_blocks, right.used_blocks) + 1;
    if (dst->capacity_blocks < output_blocks) {
        throw std::runtime_error("add_into destination capacity is too small");
    }

    clear(dst_arena, dst);
    block_t* out = dst_arena->data(*dst);
    const block_t* lhs = left_arena.data(left);
    const block_t* rhs = right_arena.data(right);

    wide_block_t carry = 0;
    for (std::size_t index = 0; index < output_blocks - 1; ++index) {
        const wide_block_t lhs_value = (index < left.used_blocks) ? lhs[index] : 0;
        const wide_block_t rhs_value = (index < right.used_blocks) ? rhs[index] : 0;
        const wide_block_t sum = lhs_value + rhs_value + carry;
        out[index] = static_cast<block_t>(sum & 0xffffffffULL);
        carry = sum >> 32;
    }
    out[output_blocks - 1] = static_cast<block_t>(carry);
    dst->used_blocks = output_blocks;
    trim(dst_arena, dst);
}

inline void add_into(PageArena* arena, BigHandle* dst, const BigHandle& left, const BigHandle& right) {
    add_into(arena, dst, *arena, left, *arena, right);
}

inline void mul_into(
    PageArena* dst_arena,
    BigHandle* dst,
    const PageArena& left_arena,
    const BigHandle& left,
    const PageArena& right_arena,
    const BigHandle& right
) {
    require_canonical_non_negative(*dst);
    require_canonical_non_negative(left);
    require_canonical_non_negative(right);

    if (left.used_blocks == 0 || right.used_blocks == 0) {
        clear(dst_arena, dst);
        return;
    }

    const std::size_t output_blocks = left.used_blocks + right.used_blocks;
    if (dst->capacity_blocks < output_blocks) {
        throw std::runtime_error("mul_into destination capacity is too small");
    }

    clear(dst_arena, dst);
    block_t* out = dst_arena->data(*dst);
    const block_t* lhs = left_arena.data(left);
    const block_t* rhs = right_arena.data(right);

    for (std::size_t i = 0; i < left.used_blocks; ++i) {
        unsigned __int128 carry = 0;
        for (std::size_t j = 0; j < right.used_blocks; ++j) {
            const std::size_t out_index = i + j;
            const unsigned __int128 sum =
                static_cast<unsigned __int128>(out[out_index])
                + static_cast<unsigned __int128>(lhs[i]) * rhs[j]
                + carry;
            out[out_index] = static_cast<block_t>(sum & 0xffffffffULL);
            carry = sum >> 32;
        }
        std::size_t out_index = i + right.used_blocks;
        while (carry != 0) {
            if (out_index >= dst->capacity_blocks) {
                throw std::runtime_error("mul_into carry spilled past destination capacity");
            }
            const unsigned __int128 sum = static_cast<unsigned __int128>(out[out_index]) + carry;
            out[out_index] = static_cast<block_t>(sum & 0xffffffffULL);
            carry = sum >> 32;
            ++out_index;
        }
    }

    dst->used_blocks = output_blocks;
    trim(dst_arena, dst);
}

inline void mul_into(PageArena* arena, BigHandle* dst, const BigHandle& left, const BigHandle& right) {
    mul_into(arena, dst, *arena, left, *arena, right);
}

inline void affine_t_into(
    PageArena* arena,
    BigHandle* dst,
    BigHandle* temp_left,
    BigHandle* temp_right,
    const BigHandle& t_left,
    const BigHandle& q_right,
    const BigHandle& p_left,
    const BigHandle& t_right
) {
    mul_into(arena, temp_left, t_left, q_right);
    mul_into(arena, temp_right, p_left, t_right);
    add_into(arena, dst, *temp_left, *temp_right);
}

}  // namespace project2::nextgen_cpu
