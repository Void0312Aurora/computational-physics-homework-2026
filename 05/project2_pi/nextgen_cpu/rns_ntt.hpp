#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "base_convert.hpp"

namespace project2::nextgen_cpu {

struct NttPrime {
    std::uint32_t mod;
    std::uint32_t primitive_root;
};

struct RnsMultiplyStats {
    std::size_t input_digits = 0;
    std::size_t output_digits = 0;
    std::size_t ntt_size = 0;
    std::size_t modulus_count = 0;
    std::size_t radix_bits = 32;
};

constexpr std::array<NttPrime, 3> kNttPrimes{{
    {998244353U, 3U},
    {1004535809U, 3U},
    {469762049U, 3U},
}};

inline std::uint32_t mod_pow(std::uint32_t base, std::uint64_t exp, std::uint32_t mod) {
    std::uint64_t result = 1;
    std::uint64_t value = base;
    while (exp != 0) {
        if ((exp & 1U) != 0) {
            result = (result * value) % mod;
        }
        value = (value * value) % mod;
        exp >>= 1U;
    }
    return static_cast<std::uint32_t>(result);
}

inline std::uint32_t mod_inv(std::uint32_t value, std::uint32_t mod) {
    return mod_pow(value, mod - 2U, mod);
}

struct CrtThreeContext {
    std::uint64_t m0 = kNttPrimes[0].mod;
    std::uint64_t m1 = kNttPrimes[1].mod;
    std::uint64_t m2 = kNttPrimes[2].mod;
    unsigned __int128 m0m1 = static_cast<unsigned __int128>(m0) * m1;
    std::uint64_t inv_m0_mod_m1 = mod_inv(static_cast<std::uint32_t>(m0 % m1), static_cast<std::uint32_t>(m1));
    std::uint64_t inv_m0m1_mod_m2 = mod_inv(
        static_cast<std::uint32_t>(m0m1 % m2),
        static_cast<std::uint32_t>(m2)
    );
};

struct NttPrimePlan {
    NttPrime prime{};
    std::vector<std::uint32_t> stage_roots_forward;
    std::vector<std::uint32_t> stage_roots_inverse;
    std::uint32_t inv_n = 0;
};

struct RnsPlan {
    std::size_t ntt_size = 0;
    std::vector<std::uint32_t> bit_reversed;
    std::array<NttPrimePlan, kNttPrimes.size()> prime_plans{};
};

struct RnsWorkspace {
    std::size_t ntt_size = 0;
    std::array<std::vector<std::uint32_t>, kNttPrimes.size()> lhs_buffers;
    std::array<std::vector<std::uint32_t>, kNttPrimes.size()> rhs_buffers;

    void ensure(std::size_t requested_ntt_size) {
        ntt_size = requested_ntt_size;
        for (std::size_t modulus_index = 0; modulus_index < kNttPrimes.size(); ++modulus_index) {
            lhs_buffers[modulus_index].resize(requested_ntt_size);
            rhs_buffers[modulus_index].resize(requested_ntt_size);
        }
    }
};

inline const CrtThreeContext& crt_three_context() {
    static const CrtThreeContext context;
    return context;
}

inline std::vector<std::uint32_t> build_bit_reversal(std::size_t ntt_size) {
    if (!std::has_single_bit(ntt_size)) {
        throw std::runtime_error("NTT size must be a power of two");
    }
    const unsigned log_ntt_size = std::countr_zero(ntt_size);
    std::vector<std::uint32_t> bit_reversed(ntt_size, 0U);
    for (std::size_t index = 0; index < ntt_size; ++index) {
        std::uint32_t reversed = 0U;
        for (unsigned bit = 0; bit < log_ntt_size; ++bit) {
            reversed = (reversed << 1U) | ((index >> bit) & 1U);
        }
        bit_reversed[index] = reversed;
    }
    return bit_reversed;
}

inline NttPrimePlan build_prime_plan(const NttPrime& prime, std::size_t ntt_size) {
    NttPrimePlan plan;
    plan.prime = prime;
    plan.inv_n = mod_inv(static_cast<std::uint32_t>(ntt_size), prime.mod);
    for (std::size_t len = 2; len <= ntt_size; len <<= 1U) {
        const std::uint32_t root = mod_pow(
            prime.primitive_root,
            (prime.mod - 1U) / static_cast<std::uint32_t>(len),
            prime.mod
        );
        plan.stage_roots_forward.push_back(root);
        plan.stage_roots_inverse.push_back(mod_inv(root, prime.mod));
    }
    return plan;
}

inline RnsPlan build_rns_plan(std::size_t ntt_size) {
    RnsPlan plan;
    plan.ntt_size = ntt_size;
    plan.bit_reversed = build_bit_reversal(ntt_size);
    for (std::size_t modulus_index = 0; modulus_index < kNttPrimes.size(); ++modulus_index) {
        plan.prime_plans[modulus_index] = build_prime_plan(kNttPrimes[modulus_index], ntt_size);
    }
    return plan;
}

inline const RnsPlan& get_rns_plan(std::size_t ntt_size) {
    static std::mutex plan_mutex;
    static std::map<std::size_t, RnsPlan> plan_cache;

    std::lock_guard<std::mutex> lock(plan_mutex);
    auto [it, inserted] = plan_cache.try_emplace(ntt_size);
    if (inserted) {
        it->second = build_rns_plan(ntt_size);
    }
    return it->second;
}

inline void ntt(
    std::vector<std::uint32_t>* values,
    bool invert,
    const RnsPlan& plan,
    const NttPrimePlan& prime_plan
) {
    if (values->size() != plan.ntt_size) {
        throw std::runtime_error("NTT buffer size does not match cached plan");
    }

    std::uint32_t* data = values->data();
    for (std::size_t index = 0; index < plan.ntt_size; ++index) {
        const std::size_t reversed = plan.bit_reversed[index];
        if (index < reversed) {
            std::swap(data[index], data[reversed]);
        }
    }

    const std::vector<std::uint32_t>& stage_roots =
        invert ? prime_plan.stage_roots_inverse : prime_plan.stage_roots_forward;
    for (std::size_t len = 2, stage = 0; len <= plan.ntt_size; len <<= 1U, ++stage) {
        const std::uint32_t wlen = stage_roots[stage];
        for (std::size_t i = 0; i < plan.ntt_size; i += len) {
            std::uint64_t w = 1;
            const std::size_t half = len >> 1U;
            for (std::size_t j = 0; j < half; ++j) {
                const std::uint32_t u = data[i + j];
                const std::uint32_t v = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(data[i + j + half]) * w) % prime_plan.prime.mod
                );
                data[i + j] = u + v < prime_plan.prime.mod ? u + v : u + v - prime_plan.prime.mod;
                data[i + j + half] = u >= v ? u - v : u + prime_plan.prime.mod - v;
                w = (w * wlen) % prime_plan.prime.mod;
            }
        }
    }

    if (invert) {
        for (std::size_t index = 0; index < plan.ntt_size; ++index) {
            data[index] = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(data[index]) * prime_plan.inv_n) % prime_plan.prime.mod
            );
        }
    }
}

inline void convolution_mod(
    const PageArena& left_arena,
    const BigHandle& left,
    const PageArena& right_arena,
    const BigHandle& right,
    const RnsPlan& plan,
    const NttPrimePlan& prime_plan,
    std::vector<std::uint32_t>* lhs,
    std::vector<std::uint32_t>* rhs
) {
    fill_residue_buffer_from_canonical(left_arena, left, prime_plan.prime.mod, lhs->data(), plan.ntt_size);
    fill_residue_buffer_from_canonical(right_arena, right, prime_plan.prime.mod, rhs->data(), plan.ntt_size);

    ntt(lhs, false, plan, prime_plan);
    ntt(rhs, false, plan, prime_plan);
    for (std::size_t index = 0; index < plan.ntt_size; ++index) {
        (*lhs)[index] = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>((*lhs)[index]) * (*rhs)[index]) % prime_plan.prime.mod
        );
    }
    ntt(lhs, true, plan, prime_plan);
}

inline unsigned __int128 crt_three(
    std::uint32_t r0,
    std::uint32_t r1,
    std::uint32_t r2,
    const CrtThreeContext& context
) {
    unsigned __int128 x = r0;
    const std::uint64_t x_mod_m1 = static_cast<std::uint64_t>(x % context.m1);
    const std::uint64_t t1 =
        ((static_cast<std::uint64_t>(r1) + context.m1 - x_mod_m1) % context.m1 * context.inv_m0_mod_m1) % context.m1;
    x += static_cast<unsigned __int128>(context.m0) * t1;

    const std::uint64_t x_mod_m2 = static_cast<std::uint64_t>(x % context.m2);
    const std::uint64_t t2 =
        ((static_cast<std::uint64_t>(r2) + context.m2 - x_mod_m2) % context.m2 * context.inv_m0m1_mod_m2) % context.m2;
    x += context.m0m1 * t2;
    return x;
}

inline void assign_from_residues(
    PageArena* arena,
    BigHandle* dst,
    const std::array<std::vector<std::uint32_t>, kNttPrimes.size()>& residues,
    std::size_t convolution_size
) {
    clear(arena, dst);
    if (dst->capacity_blocks < convolution_size + 1U) {
        throw std::runtime_error("RNS destination capacity is too small");
    }

    const CrtThreeContext& crt_context = crt_three_context();
    block_t* out = arena->data(*dst);
    std::size_t out_index = 0;
    unsigned __int128 carry = 0;
    for (std::size_t index = 0; index < convolution_size; ++index) {
        const unsigned __int128 total =
            crt_three(residues[0][index], residues[1][index], residues[2][index], crt_context) + carry;
        out[out_index++] = static_cast<block_t>(total & 0xffffffffULL);
        carry = total >> 32U;
    }
    while (carry != 0) {
        if (out_index >= dst->capacity_blocks) {
            throw std::runtime_error("RNS normalization spilled past destination capacity");
        }
        out[out_index++] = static_cast<block_t>(carry & 0xffffffffULL);
        carry >>= 32U;
    }
    dst->used_blocks = out_index;
    trim(arena, dst);
}

inline RnsMultiplyStats mul_into_rns_ntt(
    PageArena* dst_arena,
    BigHandle* dst,
    const PageArena& left_arena,
    const BigHandle& left,
    const PageArena& right_arena,
    const BigHandle& right,
    RnsWorkspace* workspace
) {
    require_canonical_non_negative(*dst);
    require_canonical_non_negative(left);
    require_canonical_non_negative(right);

    const std::size_t input_digits = std::max(canonical_digit_count(left), canonical_digit_count(right));
    if (left.used_blocks == 0 || right.used_blocks == 0) {
        clear(dst_arena, dst);
        return RnsMultiplyStats{
            .input_digits = input_digits,
            .output_digits = 1U,
            .ntt_size = 1U,
            .modulus_count = kNttPrimes.size(),
            .radix_bits = 32U,
        };
    }

    const std::size_t convolution_size = left.used_blocks + right.used_blocks - 1U;
    std::size_t ntt_size = 1U;
    while (ntt_size < convolution_size) {
        ntt_size <<= 1U;
    }

    if (workspace == nullptr) {
        throw std::runtime_error("mul_into_rns_ntt requires a workspace");
    }
    workspace->ensure(ntt_size);

    const RnsPlan& plan = get_rns_plan(ntt_size);
    for (std::size_t modulus_index = 0; modulus_index < kNttPrimes.size(); ++modulus_index) {
        convolution_mod(
            left_arena,
            left,
            right_arena,
            right,
            plan,
            plan.prime_plans[modulus_index],
            &workspace->lhs_buffers[modulus_index],
            &workspace->rhs_buffers[modulus_index]
        );
    }

    assign_from_residues(dst_arena, dst, workspace->lhs_buffers, convolution_size);
    return RnsMultiplyStats{
        .input_digits = input_digits,
        .output_digits = dst->used_blocks,
        .ntt_size = ntt_size,
        .modulus_count = kNttPrimes.size(),
        .radix_bits = 32U,
    };
}

inline RnsMultiplyStats mul_into_rns_ntt(
    PageArena* arena,
    BigHandle* dst,
    const BigHandle& left,
    const BigHandle& right,
    RnsWorkspace* workspace
) {
    return mul_into_rns_ntt(arena, dst, *arena, left, *arena, right, workspace);
}

inline RnsMultiplyStats mul_into_rns_ntt(PageArena* arena, BigHandle* dst, const BigHandle& left, const BigHandle& right) {
    RnsWorkspace workspace;
    return mul_into_rns_ntt(arena, dst, *arena, left, *arena, right, &workspace);
}

}  // namespace project2::nextgen_cpu
