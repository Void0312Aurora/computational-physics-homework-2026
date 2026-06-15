#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace project2::nextgen_cpu {

using block_t = std::uint32_t;
using wide_block_t = std::uint64_t;
constexpr wide_block_t kBlockBase = wide_block_t{1} << 32;

enum class Domain : std::uint8_t {
    canonical = 0,
    rns = 1,
};

enum class Sign : std::int8_t {
    negative = -1,
    zero = 0,
    positive = 1,
};

[[nodiscard]] inline constexpr std::string_view domain_name(Domain domain) {
    switch (domain) {
        case Domain::canonical:
            return "canonical";
        case Domain::rns:
            return "rns";
    }
    return "unknown";
}

struct BigHandle {
    std::size_t page_id = 0;
    std::size_t offset_blocks = 0;
    std::size_t used_blocks = 0;
    std::size_t capacity_blocks = 0;
    Sign sign = Sign::zero;
    Domain domain = Domain::canonical;

    [[nodiscard]] bool is_allocated() const {
        return capacity_blocks != 0;
    }
};

struct TripleSlot {
    BigHandle p;
    BigHandle q;
    BigHandle t;
};

}  // namespace project2::nextgen_cpu
