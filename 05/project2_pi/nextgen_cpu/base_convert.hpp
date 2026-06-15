#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "canonical_ops.hpp"

namespace project2::nextgen_cpu {

inline std::size_t canonical_digit_count(const BigHandle& handle) {
    require_canonical_non_negative(handle);
    return std::max<std::size_t>(handle.used_blocks, 1U);
}

inline void fill_residue_buffer_from_canonical(
    const PageArena& arena,
    const BigHandle& handle,
    std::uint32_t modulus,
    std::uint32_t* dst,
    std::size_t ntt_size
) {
    require_canonical_non_negative(handle);
    if (handle.used_blocks > ntt_size) {
        throw std::runtime_error("canonical value does not fit into requested residue buffer");
    }

    const block_t* blocks = arena.data(handle);
    for (std::size_t index = 0; index < handle.used_blocks; ++index) {
        dst[index] = static_cast<std::uint32_t>(blocks[index] % modulus);
    }
    std::fill(dst + handle.used_blocks, dst + ntt_size, 0U);
}

}  // namespace project2::nextgen_cpu
