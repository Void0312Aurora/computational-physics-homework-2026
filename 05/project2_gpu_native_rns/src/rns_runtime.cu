#include "project2_gpu_native_rns/rns_runtime.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace project2::gpu_native_rns {

void encode_signed_polynomial_blocks(
    DeviceRnsTensor& tensor,
    const std::vector<long long>& coefficients,
    std::uint32_t logical_slots
);

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::size_t kMaxSharedNttSize = 4096;
constexpr std::size_t kMaxConstantTwiddleCount = 4096;
constexpr std::size_t kMaxCachedStageLength = kMaxConstantTwiddleCount << 1u;
constexpr int kMaxCachedStageLog2 = 13;
constexpr std::size_t kGlobalFusedStageTileSize = 1024;
constexpr std::size_t kGlobalMidFusedStageTileSize = 4096;
constexpr std::uint64_t kChudnovskyA = 13591409ull;
constexpr std::uint64_t kChudnovskyB = 545140134ull;
constexpr std::uint64_t kChudnovskyC3Over24 = 10939058860032000ull;
constexpr std::size_t kPiTensorLimbBits = 16;

bool validate_metadata_all_equal(
    const std::vector<std::uint32_t>& values,
    std::uint32_t expected
);
bool validate_signs_equal(
    const std::vector<std::int8_t>& values,
    const std::vector<std::int8_t>& expected
);

std::vector<std::uint64_t> distinct_prime_factors(std::uint64_t value) {
    std::vector<std::uint64_t> factors;
    for (std::uint64_t divisor = 2; divisor * divisor <= value; ++divisor) {
        if (value % divisor != 0) {
            continue;
        }
        factors.push_back(divisor);
        while (value % divisor == 0) {
            value /= divisor;
        }
    }
    if (value > 1) {
        factors.push_back(value);
    }
    return factors;
}

struct HostBigInt {
    std::vector<std::uint32_t> limbs;
    int sign = 0;

    HostBigInt() = default;
    HostBigInt(long long value) {
        assign(value);
    }

    void assign(long long value) {
        limbs.clear();
        if (value == 0) {
            sign = 0;
            return;
        }
        sign = value < 0 ? -1 : 1;
        unsigned long long magnitude =
            value < 0 ? static_cast<unsigned long long>(-(value + 1)) + 1ull : static_cast<unsigned long long>(value);
        while (magnitude != 0) {
            limbs.push_back(static_cast<std::uint32_t>(magnitude & 0xffffffffull));
            magnitude >>= 32u;
        }
    }

    bool is_zero() const {
        return sign == 0;
    }

    void normalize() {
        while (!limbs.empty() && limbs.back() == 0u) {
            limbs.pop_back();
        }
        if (limbs.empty()) {
            sign = 0;
        }
    }
};

std::size_t bit_width_cpp_int(const HostBigInt& value);

int abs_compare(const HostBigInt& lhs, const HostBigInt& rhs) {
    if (lhs.limbs.size() != rhs.limbs.size()) {
        return lhs.limbs.size() < rhs.limbs.size() ? -1 : 1;
    }
    for (std::size_t index = lhs.limbs.size(); index > 0; --index) {
        const std::uint32_t lhs_limb = lhs.limbs[index - 1];
        const std::uint32_t rhs_limb = rhs.limbs[index - 1];
        if (lhs_limb != rhs_limb) {
            return lhs_limb < rhs_limb ? -1 : 1;
        }
    }
    return 0;
}

HostBigInt add_abs(const HostBigInt& lhs, const HostBigInt& rhs) {
    HostBigInt out;
    out.sign = 1;
    const std::size_t limb_count = std::max(lhs.limbs.size(), rhs.limbs.size());
    out.limbs.resize(limb_count, 0u);
    std::uint64_t carry = 0;
    for (std::size_t index = 0; index < limb_count; ++index) {
        const std::uint64_t lhs_limb = index < lhs.limbs.size() ? lhs.limbs[index] : 0u;
        const std::uint64_t rhs_limb = index < rhs.limbs.size() ? rhs.limbs[index] : 0u;
        const std::uint64_t total = lhs_limb + rhs_limb + carry;
        out.limbs[index] = static_cast<std::uint32_t>(total & 0xffffffffull);
        carry = total >> 32u;
    }
    if (carry != 0) {
        out.limbs.push_back(static_cast<std::uint32_t>(carry));
    }
    out.normalize();
    return out;
}

HostBigInt sub_abs(const HostBigInt& lhs, const HostBigInt& rhs) {
    HostBigInt out;
    out.sign = 1;
    out.limbs.resize(lhs.limbs.size(), 0u);
    std::int64_t borrow = 0;
    for (std::size_t index = 0; index < lhs.limbs.size(); ++index) {
        const std::int64_t lhs_limb = static_cast<std::int64_t>(lhs.limbs[index]);
        const std::int64_t rhs_limb = index < rhs.limbs.size() ? static_cast<std::int64_t>(rhs.limbs[index]) : 0;
        std::int64_t diff = lhs_limb - rhs_limb - borrow;
        if (diff < 0) {
            diff += static_cast<std::int64_t>(1ull << 32u);
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.limbs[index] = static_cast<std::uint32_t>(diff);
    }
    out.normalize();
    return out;
}

HostBigInt operator-(HostBigInt value) {
    if (!value.is_zero()) {
        value.sign = -value.sign;
    }
    return value;
}

HostBigInt operator+(const HostBigInt& lhs, const HostBigInt& rhs) {
    if (lhs.is_zero()) {
        return rhs;
    }
    if (rhs.is_zero()) {
        return lhs;
    }
    if (lhs.sign == rhs.sign) {
        HostBigInt out = add_abs(lhs, rhs);
        out.sign = lhs.sign;
        return out;
    }
    const int compare = abs_compare(lhs, rhs);
    if (compare == 0) {
        return HostBigInt{};
    }
    if (compare > 0) {
        HostBigInt out = sub_abs(lhs, rhs);
        out.sign = lhs.sign;
        return out;
    }
    HostBigInt out = sub_abs(rhs, lhs);
    out.sign = rhs.sign;
    return out;
}

HostBigInt operator-(const HostBigInt& lhs, const HostBigInt& rhs) {
    return lhs + (-rhs);
}

HostBigInt operator*(const HostBigInt& lhs, const HostBigInt& rhs) {
    if (lhs.is_zero() || rhs.is_zero()) {
        return HostBigInt{};
    }
    HostBigInt out;
    out.sign = lhs.sign * rhs.sign;
    out.limbs.assign(lhs.limbs.size() + rhs.limbs.size(), 0u);
    for (std::size_t lhs_index = 0; lhs_index < lhs.limbs.size(); ++lhs_index) {
        unsigned __int128 carry = 0;
        for (std::size_t rhs_index = 0; rhs_index < rhs.limbs.size(); ++rhs_index) {
            const std::size_t out_index = lhs_index + rhs_index;
            const unsigned __int128 current =
                static_cast<unsigned __int128>(out.limbs[out_index]) +
                static_cast<unsigned __int128>(lhs.limbs[lhs_index]) * static_cast<unsigned __int128>(rhs.limbs[rhs_index]) +
                carry;
            out.limbs[out_index] = static_cast<std::uint32_t>(current & 0xffffffffu);
            carry = current >> 32u;
        }
        std::size_t out_index = lhs_index + rhs.limbs.size();
        while (carry != 0) {
            const unsigned __int128 current = static_cast<unsigned __int128>(out.limbs[out_index]) + carry;
            out.limbs[out_index] = static_cast<std::uint32_t>(current & 0xffffffffu);
            carry = current >> 32u;
            ++out_index;
        }
    }
    out.normalize();
    return out;
}

bool operator==(const HostBigInt& lhs, const HostBigInt& rhs) {
    return lhs.sign == rhs.sign && lhs.limbs == rhs.limbs;
}

bool operator<(const HostBigInt& lhs, int rhs) {
    return lhs.sign < 0 && rhs == 0;
}

HostBigInt abs_value(HostBigInt value) {
    if (!value.is_zero()) {
        value.sign = 1;
    }
    return value;
}

bool test_bit(const HostBigInt& value, std::size_t bit_index) {
    const std::size_t limb_index = bit_index / 32u;
    if (limb_index >= value.limbs.size()) {
        return false;
    }
    return ((value.limbs[limb_index] >> (bit_index % 32u)) & 1u) != 0u;
}

void set_bit(HostBigInt& value, std::size_t bit_index) {
    const std::size_t limb_index = bit_index / 32u;
    if (value.limbs.size() <= limb_index) {
        value.limbs.resize(limb_index + 1, 0u);
    }
    value.limbs[limb_index] |= static_cast<std::uint32_t>(1u << (bit_index % 32u));
    value.sign = 1;
}

void add_u32_inplace(HostBigInt& value, std::uint32_t addend) {
    if (addend == 0u) {
        return;
    }
    if (value.is_zero()) {
        value.sign = 1;
        value.limbs.push_back(addend);
        return;
    }

    std::uint64_t carry = addend;
    for (std::size_t index = 0; index < value.limbs.size() && carry != 0; ++index) {
        const std::uint64_t total = static_cast<std::uint64_t>(value.limbs[index]) + carry;
        value.limbs[index] = static_cast<std::uint32_t>(total & 0xffffffffull);
        carry = total >> 32u;
    }
    if (carry != 0) {
        value.limbs.push_back(static_cast<std::uint32_t>(carry));
    }
    value.sign = 1;
}

void shift_left_one_inplace(HostBigInt& value) {
    if (value.is_zero()) {
        return;
    }

    std::uint64_t carry = 0;
    for (std::size_t index = 0; index < value.limbs.size(); ++index) {
        const std::uint64_t shifted = (static_cast<std::uint64_t>(value.limbs[index]) << 1u) | carry;
        value.limbs[index] = static_cast<std::uint32_t>(shifted & 0xffffffffull);
        carry = shifted >> 32u;
    }
    if (carry != 0) {
        value.limbs.push_back(static_cast<std::uint32_t>(carry));
    }
}

void shift_right_one_inplace(HostBigInt& value) {
    if (value.is_zero()) {
        return;
    }

    std::uint32_t carry = 0u;
    for (std::size_t index = value.limbs.size(); index > 0; --index) {
        const std::uint32_t next_carry = static_cast<std::uint32_t>(value.limbs[index - 1] & 1u);
        value.limbs[index - 1] = (value.limbs[index - 1] >> 1u) | (carry << 31u);
        carry = next_carry;
    }
    value.normalize();
}

std::pair<HostBigInt, HostBigInt> div_mod_abs(const HostBigInt& dividend, const HostBigInt& divisor) {
    if (divisor.is_zero()) {
        throw std::invalid_argument("div_mod_abs requires divisor != 0");
    }
    if (dividend.is_zero()) {
        return {HostBigInt{}, HostBigInt{}};
    }
    if (abs_compare(dividend, divisor) < 0) {
        return {HostBigInt{}, dividend};
    }

    HostBigInt quotient;
    HostBigInt remainder;
    quotient.sign = 1;

    const std::size_t bit_count = bit_width_cpp_int(dividend);
    for (std::size_t bit_index = bit_count; bit_index > 0; --bit_index) {
        shift_left_one_inplace(remainder);
        if (test_bit(dividend, bit_index - 1)) {
            add_u32_inplace(remainder, 1u);
        }
        if (abs_compare(remainder, divisor) >= 0) {
            remainder = sub_abs(remainder, divisor);
            set_bit(quotient, bit_index - 1);
        }
    }
    quotient.normalize();
    remainder.normalize();
    return {quotient, remainder};
}

std::pair<HostBigInt, std::uint32_t> div_mod_u32_abs(const HostBigInt& dividend, std::uint32_t divisor) {
    if (divisor == 0u) {
        throw std::invalid_argument("div_mod_u32_abs requires divisor != 0");
    }
    if (dividend.is_zero()) {
        return {HostBigInt{}, 0u};
    }

    HostBigInt quotient;
    quotient.sign = 1;
    quotient.limbs.assign(dividend.limbs.size(), 0u);
    std::uint64_t remainder = 0;
    for (std::size_t index = dividend.limbs.size(); index > 0; --index) {
        const std::uint64_t current = (remainder << 32u) | dividend.limbs[index - 1];
        quotient.limbs[index - 1] = static_cast<std::uint32_t>(current / divisor);
        remainder = current % divisor;
    }
    quotient.normalize();
    return {quotient, static_cast<std::uint32_t>(remainder)};
}

HostBigInt pow10_host_bigint(std::size_t exponent) {
    HostBigInt out = 1;
    for (std::size_t index = 0; index < exponent; ++index) {
        out = out * 10;
    }
    return out;
}

HostBigInt integer_sqrt_host_bigint(const HostBigInt& value) {
    if (value < 0) {
        throw std::invalid_argument("integer_sqrt_host_bigint requires value >= 0");
    }
    if (value.is_zero()) {
        return value;
    }

    HostBigInt current;
    current.sign = 1;
    set_bit(current, (bit_width_cpp_int(value) + 1u) / 2u);
    while (true) {
        const auto [quotient, _] = div_mod_abs(value, current);
        (void)_;
        HostBigInt next = current + quotient;
        shift_right_one_inplace(next);
        if (abs_compare(next, current) >= 0) {
            return current;
        }
        current = std::move(next);
    }
}

std::string host_bigint_to_decimal_string(const HostBigInt& value) {
    if (value.is_zero()) {
        return "0";
    }

    HostBigInt current = abs_value(value);
    std::vector<std::uint32_t> chunks;
    while (!current.is_zero()) {
        auto [quotient, remainder] = div_mod_u32_abs(current, 1000000000u);
        chunks.push_back(remainder);
        current = std::move(quotient);
    }

    std::ostringstream out;
    if (value.sign < 0) {
        out << '-';
    }
    out << chunks.back();
    for (std::size_t index = chunks.size() - 1; index > 0; --index) {
        out << std::setw(9) << std::setfill('0') << chunks[index - 1];
    }
    return out.str();
}

std::string format_scaled_decimal(const HostBigInt& scaled_value, std::size_t fractional_digits) {
    std::string digits = host_bigint_to_decimal_string(scaled_value);
    if (!digits.empty() && digits.front() == '-') {
        throw std::invalid_argument("format_scaled_decimal expects a non-negative scaled value");
    }
    const std::size_t required_digits = fractional_digits + 1;
    if (digits.size() < required_digits) {
        digits.insert(digits.begin(), required_digits - digits.size(), '0');
    }
    return digits.substr(0, 1) + "." + digits.substr(1);
}

const std::string& pi_reference_digits() {
    static const std::string digits =
        "3141592653589793238462643383279502884197169399375105820974944592"
        "3078164062862089986280348253421170679";
    return digits;
}

__constant__ std::uint32_t g_stage_roots[ModulusConfig::kModulusCount];
__constant__ std::uint32_t g_inverse_scales[ModulusConfig::kModulusCount];

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_same_shape(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, const DeviceRnsTensor& out) {
    const bool ok =
        lhs.value_count == rhs.value_count &&
        lhs.value_count == out.value_count &&
        lhs.slot_count == rhs.slot_count &&
        lhs.slot_count == out.slot_count &&
        lhs.modulus_count == rhs.modulus_count &&
        lhs.modulus_count == out.modulus_count;
    if (!ok) {
        throw std::invalid_argument("RNS tensors must have identical shapes");
    }
}

void check_convolution_shape(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, const DeviceRnsTensor& out) {
    const bool ok =
        lhs.value_count == rhs.value_count &&
        lhs.value_count == out.value_count &&
        lhs.modulus_count == rhs.modulus_count &&
        lhs.modulus_count == out.modulus_count &&
        out.slot_count == lhs.slot_count + rhs.slot_count - 1;
    if (!ok) {
        throw std::invalid_argument("Convolution requires shared value/modulus counts and out.slot_count = lhs.slot_count + rhs.slot_count - 1");
    }
}

std::size_t scalar_count(const DeviceRnsTensor& tensor) {
    return tensor.value_count * tensor.slot_count;
}

std::size_t residue_count(const DeviceRnsTensor& tensor) {
    return scalar_count(tensor) * static_cast<std::size_t>(tensor.modulus_count);
}

std::size_t next_power_of_two(std::size_t value) {
    std::size_t power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

int block_count(std::size_t element_count) {
    const std::size_t blocks = (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return static_cast<int>(std::max<std::size_t>(1, blocks));
}

int ceil_log2_size(std::size_t value) {
    if (value <= 1) {
        return 0;
    }
    int bits = 0;
    std::size_t power = 1;
    while (power < value) {
        power <<= 1;
        ++bits;
    }
    return bits;
}

int max_safe_input_bits(std::size_t slot_count) {
    const int accumulation_bits = ceil_log2_size(slot_count);
    return std::max(1, (62 - accumulation_bits) / 2);
}

int resolve_input_bits(const SmokeConfig& config) {
    const int max_bits = max_safe_input_bits(config.slot_count);
    if (config.input_bits == 0) {
        return max_bits;
    }
    if (config.input_bits < 1 || config.input_bits > max_bits) {
        throw std::invalid_argument("input_bits must be in [1, max_safe_input_bits(slot_count)] for exact uint64 validation");
    }
    return config.input_bits;
}

std::size_t resolve_convolution_ntt_size(std::size_t out_slot_count) {
    const std::size_t ntt_size = next_power_of_two(out_slot_count);
    if (ntt_size > (1ull << 26)) {
        throw std::invalid_argument("NTT size exceeds the 2^26 power-of-two support of the current RNS moduli");
    }
    return ntt_size;
}

bool use_shared_ntt(std::size_t ntt_size) {
    return ntt_size <= kMaxSharedNttSize;
}

const char* convolution_algorithm_name(std::size_t ntt_size) {
    return use_shared_ntt(ntt_size) ? "ntt_shared" : "ntt_global";
}

const char* verify_mode_name(VerifyMode mode) {
    switch (mode) {
        case VerifyMode::kNone:
            return "none";
        case VerifyMode::kSampled:
            return "sampled";
        case VerifyMode::kFull:
            return "full";
    }
    return "unknown";
}

std::vector<std::uint32_t> build_spread_sample_indices(std::size_t total_count, std::size_t requested_count) {
    if (requested_count == 0 || total_count == 0) {
        return {};
    }
    const std::size_t actual_count = std::min(total_count, requested_count);
    std::vector<std::uint32_t> indices(actual_count);
    if (actual_count == total_count) {
        for (std::size_t index = 0; index < actual_count; ++index) {
            indices[index] = static_cast<std::uint32_t>(index);
        }
        return indices;
    }
    if (actual_count == 1) {
        indices[0] = static_cast<std::uint32_t>(total_count / 2);
        return indices;
    }
    for (std::size_t index = 0; index < actual_count; ++index) {
        const std::size_t spread =
            (index * (total_count - 1)) / (actual_count - 1);
        indices[index] = static_cast<std::uint32_t>(spread);
    }
    return indices;
}

struct SampleValidationPlan {
    std::vector<std::uint32_t> pointwise_indices;
    std::vector<std::uint32_t> convolution_indices;
};

std::uint64_t mask_for_bits(int input_bits) {
    if (input_bits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (1ull << input_bits) - 1ull;
}

std::uint32_t pow_mod_host(
    std::uint32_t base,
    std::uint64_t exponent,
    std::uint32_t modulus
) {
    std::uint64_t result = 1;
    std::uint64_t value = base % modulus;
    while (exponent != 0) {
        if ((exponent & 1u) != 0u) {
            result = (result * value) % modulus;
        }
        value = (value * value) % modulus;
        exponent >>= 1u;
    }
    return static_cast<std::uint32_t>(result);
}

int modulus_two_adicity(std::uint32_t modulus) {
    return __builtin_ctz(modulus - 1u);
}

std::uint32_t primitive_root_for_modulus(std::uint32_t modulus) {
    const std::uint64_t phi = static_cast<std::uint64_t>(modulus) - 1u;
    const std::vector<std::uint64_t> factors = distinct_prime_factors(phi);
    for (std::uint32_t candidate = 2; candidate < modulus; ++candidate) {
        bool ok = true;
        for (std::uint64_t factor : factors) {
            if (pow_mod_host(candidate, phi / factor, modulus) == 1u) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return candidate;
        }
    }
    throw std::runtime_error("failed to find primitive root for modulus");
}

const std::array<std::uint32_t, ModulusConfig::kModulusCount>& primitive_roots() {
    static const std::array<std::uint32_t, ModulusConfig::kModulusCount> roots = []() {
        std::array<std::uint32_t, ModulusConfig::kModulusCount> out{};
        for (int modulus_index = 0; modulus_index < ModulusConfig::kModulusCount; ++modulus_index) {
            out[modulus_index] = primitive_root_for_modulus(ModulusConfig::kModuli[modulus_index]);
        }
        return out;
    }();
    return roots;
}

void ensure_ntt_support_for_modulus_count(int modulus_count, std::size_t ntt_size) {
    const int required_log2 = ceil_log2_size(ntt_size);
    for (int modulus_index = 0; modulus_index < modulus_count; ++modulus_index) {
        if (modulus_two_adicity(ModulusConfig::kModuli[modulus_index]) < required_log2) {
            throw std::invalid_argument("active modulus set does not support requested NTT size");
        }
    }
}

void upload_stage_roots(std::size_t length, bool inverse) {
    std::array<std::uint32_t, ModulusConfig::kModulusCount> roots{};
    for (int modulus_index = 0; modulus_index < ModulusConfig::kModulusCount; ++modulus_index) {
        const std::uint32_t modulus = ModulusConfig::kModuli[modulus_index];
        std::uint32_t root = pow_mod_host(
            primitive_roots()[modulus_index],
            static_cast<std::uint64_t>(modulus - 1u) / length,
            modulus
        );
        if (inverse) {
            root = pow_mod_host(root, modulus - 2u, modulus);
        }
        roots[modulus_index] = root;
    }
    check_cuda(
        cudaMemcpyToSymbol(g_stage_roots, roots.data(), sizeof(std::uint32_t) * roots.size()),
        "cudaMemcpyToSymbol g_stage_roots"
    );
}

struct StageTwiddleCache {
    std::uint32_t* forward = nullptr;
    std::uint32_t* inverse = nullptr;
    std::array<std::size_t, kMaxCachedStageLog2 + 1> offsets{};
    std::size_t entries_per_direction = 0;
    bool ready = false;
};

StageTwiddleCache& stage_twiddle_cache() {
    static StageTwiddleCache cache;
    return cache;
}

void ensure_stage_twiddle_cache() {
    auto& cache = stage_twiddle_cache();
    if (cache.ready) {
        return;
    }

    std::size_t total_entries = 0;
    for (int stage_log2 = 1; stage_log2 <= kMaxCachedStageLog2; ++stage_log2) {
        cache.offsets[stage_log2] = total_entries;
        total_entries +=
            static_cast<std::size_t>(ModulusConfig::kModulusCount) * (static_cast<std::size_t>(1) << (stage_log2 - 1));
    }

    std::vector<std::uint32_t> forward(total_entries);
    std::vector<std::uint32_t> inverse(total_entries);
    for (int stage_log2 = 1; stage_log2 <= kMaxCachedStageLog2; ++stage_log2) {
        const std::size_t length = static_cast<std::size_t>(1) << stage_log2;
        const std::size_t half = length >> 1u;
        const std::size_t stage_base = cache.offsets[stage_log2];
        for (int modulus_index = 0; modulus_index < ModulusConfig::kModulusCount; ++modulus_index) {
            const std::uint32_t modulus = ModulusConfig::kModuli[modulus_index];
            const std::uint32_t forward_root = pow_mod_host(
                primitive_roots()[modulus_index],
                static_cast<std::uint64_t>(modulus - 1u) / length,
                modulus
            );
            const std::uint32_t inverse_root = pow_mod_host(forward_root, modulus - 2u, modulus);

            std::uint32_t forward_twiddle = 1u;
            std::uint32_t inverse_twiddle = 1u;
            const std::size_t modulus_base = stage_base + static_cast<std::size_t>(modulus_index) * half;
            for (std::size_t index = 0; index < half; ++index) {
                forward[modulus_base + index] = forward_twiddle;
                inverse[modulus_base + index] = inverse_twiddle;
                forward_twiddle = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(forward_twiddle) * forward_root) % modulus
                );
                inverse_twiddle = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(inverse_twiddle) * inverse_root) % modulus
                );
            }
        }
    }

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&cache.forward), sizeof(std::uint32_t) * total_entries),
        "cudaMalloc stage_twiddle_cache forward"
    );
    try {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&cache.inverse), sizeof(std::uint32_t) * total_entries),
            "cudaMalloc stage_twiddle_cache inverse"
        );
    } catch (...) {
        cudaFree(cache.forward);
        cache.forward = nullptr;
        throw;
    }

    check_cuda(
        cudaMemcpy(cache.forward, forward.data(), sizeof(std::uint32_t) * total_entries, cudaMemcpyHostToDevice),
        "cudaMemcpy stage_twiddle_cache forward"
    );
    check_cuda(
        cudaMemcpy(cache.inverse, inverse.data(), sizeof(std::uint32_t) * total_entries, cudaMemcpyHostToDevice),
        "cudaMemcpy stage_twiddle_cache inverse"
    );

    cache.entries_per_direction = total_entries;
    cache.ready = true;
}

void upload_inverse_scales(std::size_t ntt_size) {
    static std::size_t cached_ntt_size = 0;
    static bool cache_valid = false;
    if (cache_valid && cached_ntt_size == ntt_size) {
        return;
    }

    std::array<std::uint32_t, ModulusConfig::kModulusCount> scales{};
    for (int modulus_index = 0; modulus_index < ModulusConfig::kModulusCount; ++modulus_index) {
        const std::uint32_t modulus = ModulusConfig::kModuli[modulus_index];
        scales[modulus_index] = pow_mod_host(static_cast<std::uint32_t>(ntt_size % modulus), modulus - 2u, modulus);
    }
    check_cuda(
        cudaMemcpyToSymbol(g_inverse_scales, scales.data(), sizeof(std::uint32_t) * scales.size()),
        "cudaMemcpyToSymbol g_inverse_scales"
    );
    cached_ntt_size = ntt_size;
    cache_valid = true;
}

__global__ void encode_u64_kernel(
    const std::uint64_t* values,
    const std::uint32_t* moduli,
    std::uint32_t* residues,
    std::size_t scalar_count,
    int modulus_count
) {
    const std::size_t total = scalar_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / scalar_count;
        const std::size_t scalar_index = index % scalar_count;
        residues[index] = static_cast<std::uint32_t>(values[scalar_index] % moduli[modulus_index]);
    }
}

__global__ void scatter_scaled_constant_kernel(
    const std::uint32_t* moduli,
    std::uint32_t* residues,
    std::size_t value_count,
    std::size_t slot_count,
    std::size_t coefficient_slot,
    std::uint64_t coefficient,
    int modulus_count
) {
    const std::size_t total = value_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / value_count;
        const std::size_t value_index = index % value_count;
        const std::size_t offset =
            (modulus_index * value_count + value_index) * slot_count + coefficient_slot;
        residues[offset] = static_cast<std::uint32_t>(coefficient % moduli[modulus_index]);
    }
}

__global__ void init_metadata_kernel(
    const std::uint64_t* values,
    std::int8_t* signs,
    std::uint32_t* logical_slots,
    std::uint32_t* scale_bits,
    std::uint32_t* levels,
    std::size_t value_count,
    std::size_t slot_count,
    std::uint32_t modulus_count
) {
    for (std::size_t value_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         value_index < value_count;
         value_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        std::int8_t sign = 0;
        for (std::size_t slot = 0; slot < slot_count; ++slot) {
            if (values[value_index * slot_count + slot] != 0) {
                sign = 1;
                break;
            }
        }
        signs[value_index] = sign;
        logical_slots[value_index] = static_cast<std::uint32_t>(slot_count);
        scale_bits[value_index] = 0;
        levels[value_index] = modulus_count;
    }
}

__global__ void encode_u64_dual_kernel(
    const std::uint64_t* lhs_values,
    const std::uint64_t* rhs_values,
    const std::uint32_t* moduli,
    std::uint32_t* lhs_residues,
    std::uint32_t* rhs_residues,
    std::size_t scalar_count,
    int modulus_count
) {
    const std::size_t total = scalar_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / scalar_count;
        const std::size_t scalar_index = index % scalar_count;
        const std::uint32_t modulus = moduli[modulus_index];
        lhs_residues[index] = static_cast<std::uint32_t>(lhs_values[scalar_index] % modulus);
        rhs_residues[index] = static_cast<std::uint32_t>(rhs_values[scalar_index] % modulus);
    }
}

__global__ void init_metadata_dual_kernel(
    const std::uint64_t* lhs_values,
    const std::uint64_t* rhs_values,
    std::int8_t* lhs_signs,
    std::uint32_t* lhs_logical_slots,
    std::uint32_t* lhs_scale_bits,
    std::uint32_t* lhs_levels,
    std::int8_t* rhs_signs,
    std::uint32_t* rhs_logical_slots,
    std::uint32_t* rhs_scale_bits,
    std::uint32_t* rhs_levels,
    std::size_t value_count,
    std::size_t slot_count,
    std::uint32_t modulus_count
) {
    for (std::size_t value_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         value_index < value_count;
         value_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        std::int8_t lhs_sign = 0;
        std::int8_t rhs_sign = 0;
        for (std::size_t slot = 0; slot < slot_count; ++slot) {
            const std::size_t offset = value_index * slot_count + slot;
            if (lhs_values[offset] != 0) {
                lhs_sign = 1;
            }
            if (rhs_values[offset] != 0) {
                rhs_sign = 1;
            }
            if (lhs_sign != 0 && rhs_sign != 0) {
                break;
            }
        }
        lhs_signs[value_index] = lhs_sign;
        lhs_logical_slots[value_index] = static_cast<std::uint32_t>(slot_count);
        lhs_scale_bits[value_index] = 0;
        lhs_levels[value_index] = modulus_count;
        rhs_signs[value_index] = rhs_sign;
        rhs_logical_slots[value_index] = static_cast<std::uint32_t>(slot_count);
        rhs_scale_bits[value_index] = 0;
        rhs_levels[value_index] = modulus_count;
    }
}

__global__ void gather_residue_samples_kernel(
    const std::uint32_t* src,
    const std::uint32_t* sample_indices,
    std::uint32_t* out,
    std::size_t scalar_count,
    std::size_t sample_count,
    int modulus_count
) {
    const std::size_t total = sample_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / sample_count;
        const std::size_t sample_index = index % sample_count;
        const std::size_t scalar_index = sample_indices[sample_index];
        out[index] = src[modulus_index * scalar_count + scalar_index];
    }
}

__global__ void gather_residue_samples_triplet_kernel(
    const std::uint32_t* src0,
    const std::uint32_t* src1,
    const std::uint32_t* src2,
    const std::uint32_t* sample_indices,
    std::uint32_t* out,
    std::size_t scalar_count,
    std::size_t sample_count,
    int modulus_count
) {
    const std::size_t per_tensor = sample_count * static_cast<std::size_t>(modulus_count);
    const std::size_t total = per_tensor * 3u;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t tensor_index = index / per_tensor;
        const std::size_t tensor_offset = index % per_tensor;
        const std::size_t modulus_index = tensor_offset / sample_count;
        const std::size_t sample_index = tensor_offset % sample_count;
        const std::size_t scalar_index = sample_indices[sample_index];
        const std::uint32_t* src =
            tensor_index == 0 ? src0 : (tensor_index == 1 ? src1 : src2);
        out[index] = src[modulus_index * scalar_count + scalar_index];
    }
}

__global__ void pointwise_add_kernel(
    const std::uint32_t* lhs,
    const std::uint32_t* rhs,
    const std::uint32_t* moduli,
    std::uint32_t* out,
    std::size_t scalar_count,
    int modulus_count
) {
    const std::size_t total = scalar_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / scalar_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::uint64_t sum = static_cast<std::uint64_t>(lhs[index]) + rhs[index];
        out[index] = static_cast<std::uint32_t>(sum >= modulus ? sum - modulus : sum);
    }
}

__global__ void pointwise_sub_kernel(
    const std::uint32_t* lhs,
    const std::uint32_t* rhs,
    const std::uint32_t* moduli,
    std::uint32_t* out,
    std::size_t scalar_count,
    int modulus_count
) {
    const std::size_t total = scalar_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / scalar_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::uint32_t lhs_value = lhs[index];
        const std::uint32_t rhs_value = rhs[index];
        out[index] = lhs_value >= rhs_value ? lhs_value - rhs_value : modulus - (rhs_value - lhs_value);
    }
}

__global__ void pointwise_mul_kernel(
    const std::uint32_t* lhs,
    const std::uint32_t* rhs,
    const std::uint32_t* moduli,
    std::uint32_t* out,
    std::size_t scalar_count,
    int modulus_count
) {
    const std::size_t total = scalar_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / scalar_count;
        const std::uint64_t modulus = moduli[modulus_index];
        const std::uint64_t product = static_cast<std::uint64_t>(lhs[index]) * rhs[index];
        out[index] = static_cast<std::uint32_t>(product % modulus);
    }
}

__device__ std::uint32_t primitive_root_for_modulus_index(std::size_t modulus_index) {
    switch (modulus_index) {
        case 0:
            return 31u;
        case 1:
            return 13u;
        case 2:
            return 3u;
        case 3:
            return 5u;
        case 4:
            return 29u;
        case 5:
            return 3u;
        case 6:
            return 10u;
        case 7:
            return 3u;
        case 8:
            return 3u;
        case 9:
            return 11u;
        default:
            return 1u;
    }
}

__device__ std::uint32_t mul_mod_device(
    std::uint32_t lhs,
    std::uint32_t rhs,
    std::uint32_t modulus
) {
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(lhs) * rhs) % modulus);
}

__device__ std::uint32_t pow_mod_device(
    std::uint32_t base,
    std::uint64_t exponent,
    std::uint32_t modulus
) {
    std::uint64_t result = 1;
    std::uint64_t value = base % modulus;
    while (exponent != 0) {
        if ((exponent & 1u) != 0u) {
            result = (result * value) % modulus;
        }
        value = (value * value) % modulus;
        exponent >>= 1u;
    }
    return static_cast<std::uint32_t>(result);
}

__device__ std::size_t bit_reverse_index(std::size_t index, int log_size) {
    if (log_size == 0) {
        return 0;
    }
    return static_cast<std::size_t>(__brev(static_cast<unsigned int>(index)) >> (32 - log_size));
}

__host__ __device__ constexpr std::size_t stage_twiddle_base_from_half(std::size_t half) {
    return static_cast<std::size_t>(ModulusConfig::kModulusCount) * (half - 1u);
}

__device__ void ntt_in_shared(
    std::uint32_t* data,
    std::size_t size,
    int log_size,
    std::uint32_t modulus,
    std::uint32_t primitive_root,
    bool inverse
) {
    for (std::size_t index = threadIdx.x; index < size; index += blockDim.x) {
        const std::size_t reversed = bit_reverse_index(index, log_size);
        if (index < reversed) {
            const std::uint32_t tmp = data[index];
            data[index] = data[reversed];
            data[reversed] = tmp;
        }
    }
    __syncthreads();

    for (std::size_t length = 2; length <= size; length <<= 1u) {
        const std::size_t half = length >> 1u;
        std::uint32_t root = pow_mod_device(
            primitive_root,
            static_cast<std::uint64_t>(modulus - 1u) / length,
            modulus
        );
        if (inverse) {
            root = pow_mod_device(root, modulus - 2u, modulus);
        }

        const std::size_t butterfly_count = size >> 1u;
        for (std::size_t butterfly = threadIdx.x; butterfly < butterfly_count; butterfly += blockDim.x) {
            const std::size_t block = butterfly / half;
            const std::size_t j = butterfly % half;
            const std::size_t base = block * length;
            const std::uint32_t twiddle = pow_mod_device(root, j, modulus);
            const std::uint32_t even = data[base + j];
            const std::uint32_t odd = mul_mod_device(data[base + j + half], twiddle, modulus);
            const std::uint32_t sum = even + odd;
            data[base + j] = sum >= modulus ? sum - modulus : sum;
            data[base + j + half] = even >= odd ? even - odd : modulus - (odd - even);
        }
        __syncthreads();
    }

    if (inverse) {
        const std::uint32_t inv_size = pow_mod_device(static_cast<std::uint32_t>(size % modulus), modulus - 2u, modulus);
        for (std::size_t index = threadIdx.x; index < size; index += blockDim.x) {
            data[index] = mul_mod_device(data[index], inv_size, modulus);
        }
        __syncthreads();
    }
}

__global__ void ntt_convolution_kernel(
    const std::uint32_t* lhs,
    const std::uint32_t* rhs,
    const std::uint32_t* moduli,
    std::uint32_t* out,
    std::size_t value_count,
    std::size_t lhs_slot_count,
    std::size_t rhs_slot_count,
    std::size_t out_slot_count,
    std::size_t ntt_size,
    int log_ntt_size
) {
    extern __shared__ std::uint32_t shared[];
    std::uint32_t* lhs_shared = shared;
    std::uint32_t* rhs_shared = shared + ntt_size;

    const std::size_t transform_index = static_cast<std::size_t>(blockIdx.x);
    const std::size_t modulus_index = transform_index / value_count;
    const std::size_t value_index = transform_index % value_count;
    const std::uint32_t modulus = moduli[modulus_index];
    const std::uint32_t primitive_root = primitive_root_for_modulus_index(modulus_index);

    const std::size_t lhs_offset = (modulus_index * value_count + value_index) * lhs_slot_count;
    const std::size_t rhs_offset = (modulus_index * value_count + value_index) * rhs_slot_count;
    const std::size_t out_offset = (modulus_index * value_count + value_index) * out_slot_count;

    for (std::size_t slot = threadIdx.x; slot < ntt_size; slot += blockDim.x) {
        lhs_shared[slot] = slot < lhs_slot_count ? lhs[lhs_offset + slot] : 0u;
        rhs_shared[slot] = slot < rhs_slot_count ? rhs[rhs_offset + slot] : 0u;
    }
    __syncthreads();

    if (ntt_size == 1) {
        if (threadIdx.x == 0) {
            out[out_offset] = mul_mod_device(lhs_shared[0], rhs_shared[0], modulus);
        }
        return;
    }

    ntt_in_shared(lhs_shared, ntt_size, log_ntt_size, modulus, primitive_root, false);
    ntt_in_shared(rhs_shared, ntt_size, log_ntt_size, modulus, primitive_root, false);

    for (std::size_t slot = threadIdx.x; slot < ntt_size; slot += blockDim.x) {
        lhs_shared[slot] = mul_mod_device(lhs_shared[slot], rhs_shared[slot], modulus);
    }
    __syncthreads();

    ntt_in_shared(lhs_shared, ntt_size, log_ntt_size, modulus, primitive_root, true);

    for (std::size_t slot = threadIdx.x; slot < out_slot_count; slot += blockDim.x) {
        out[out_offset + slot] = lhs_shared[slot];
    }
}

__global__ void pad_and_bit_reverse_dual_kernel(
    const std::uint32_t* lhs_src,
    const std::uint32_t* rhs_src,
    std::uint32_t* lhs_dst,
    std::uint32_t* rhs_dst,
    std::size_t value_count,
    std::size_t lhs_slot_count,
    std::size_t rhs_slot_count,
    std::size_t ntt_size,
    int log_ntt_size,
    std::size_t transform_count
) {
    const std::size_t total = transform_count * ntt_size;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t transform_index = index / ntt_size;
        const std::size_t slot = index % ntt_size;
        const std::size_t reversed = bit_reverse_index(slot, log_ntt_size);
        const std::size_t modulus_index = transform_index / value_count;
        const std::size_t value_index = transform_index % value_count;
        const std::size_t lhs_src_offset = (modulus_index * value_count + value_index) * lhs_slot_count;
        const std::size_t rhs_src_offset = (modulus_index * value_count + value_index) * rhs_slot_count;
        lhs_dst[index] = reversed < lhs_slot_count ? lhs_src[lhs_src_offset + reversed] : 0u;
        rhs_dst[index] = reversed < rhs_slot_count ? rhs_src[rhs_src_offset + reversed] : 0u;
    }
}

__global__ void ntt_stages_dual_shared_kernel(
    std::uint32_t* lhs,
    std::uint32_t* rhs,
    const std::uint32_t* moduli,
    const std::uint32_t* stage_twiddles,
    std::size_t value_count,
    std::size_t ntt_size,
    std::size_t stage_start_length,
    std::size_t tile_size,
    std::size_t transform_count
) {
    extern __shared__ std::uint32_t shared[];
    std::uint32_t* lhs_shared = shared;
    std::uint32_t* rhs_shared = shared + tile_size;

    const std::size_t tiles_per_transform = ntt_size / tile_size;
    const std::size_t block_index = static_cast<std::size_t>(blockIdx.x);
    const std::size_t transform_index = block_index / tiles_per_transform;
    const std::size_t tile_index = block_index % tiles_per_transform;
    if (transform_index >= transform_count) {
        return;
    }

    const std::size_t modulus_index = transform_index / value_count;
    const std::size_t tile_offset = transform_index * ntt_size + tile_index * tile_size;
    const std::uint32_t modulus = moduli[modulus_index];

    for (std::size_t offset = threadIdx.x; offset < tile_size; offset += blockDim.x) {
        lhs_shared[offset] = lhs[tile_offset + offset];
        rhs_shared[offset] = rhs[tile_offset + offset];
    }
    __syncthreads();

    for (std::size_t length = stage_start_length; length <= tile_size; length <<= 1u) {
        const std::size_t half = length >> 1u;
        const std::size_t butterfly_count = tile_size >> 1u;
        const std::size_t stage_base = stage_twiddle_base_from_half(half);
        const std::size_t modulus_base = stage_base + modulus_index * half;

        for (std::size_t butterfly = threadIdx.x; butterfly < butterfly_count; butterfly += blockDim.x) {
            const std::size_t block = butterfly / half;
            const std::size_t j = butterfly % half;
            const std::size_t base = block * length;
            const std::uint32_t twiddle = stage_twiddles[modulus_base + j];

            const std::uint32_t lhs_even = lhs_shared[base + j];
            const std::uint32_t lhs_odd = mul_mod_device(lhs_shared[base + j + half], twiddle, modulus);
            const std::uint32_t lhs_sum = lhs_even + lhs_odd;
            lhs_shared[base + j] = lhs_sum >= modulus ? lhs_sum - modulus : lhs_sum;
            lhs_shared[base + j + half] =
                lhs_even >= lhs_odd ? lhs_even - lhs_odd : modulus - (lhs_odd - lhs_even);

            const std::uint32_t rhs_even = rhs_shared[base + j];
            const std::uint32_t rhs_odd = mul_mod_device(rhs_shared[base + j + half], twiddle, modulus);
            const std::uint32_t rhs_sum = rhs_even + rhs_odd;
            rhs_shared[base + j] = rhs_sum >= modulus ? rhs_sum - modulus : rhs_sum;
            rhs_shared[base + j + half] =
                rhs_even >= rhs_odd ? rhs_even - rhs_odd : modulus - (rhs_odd - rhs_even);
        }
        __syncthreads();
    }

    for (std::size_t offset = threadIdx.x; offset < tile_size; offset += blockDim.x) {
        lhs[tile_offset + offset] = lhs_shared[offset];
        rhs[tile_offset + offset] = rhs_shared[offset];
    }
}

__global__ void ntt_stage_kernel(
    std::uint32_t* data,
    const std::uint32_t* moduli,
    const std::uint32_t* stage_twiddles,
    std::size_t value_count,
    std::size_t ntt_size,
    std::size_t length,
    std::size_t stage_twiddle_base,
    std::size_t transform_count,
    bool use_cached_twiddles
) {
    const std::size_t half = length >> 1u;
    const std::size_t butterflies_per_transform = ntt_size >> 1u;
    const std::size_t total = transform_count * butterflies_per_transform;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t transform_index = index / butterflies_per_transform;
        const std::size_t butterfly = index % butterflies_per_transform;
        const std::size_t block = butterfly / half;
        const std::size_t j = butterfly % half;
        const std::size_t modulus_index = transform_index / value_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::uint32_t root = g_stage_roots[modulus_index];
        const std::size_t transform_offset = transform_index * ntt_size;
        const std::size_t base = transform_offset + block * length;
        const std::uint32_t twiddle =
            use_cached_twiddles
                ? stage_twiddles[stage_twiddle_base + modulus_index * half + j]
                : pow_mod_device(root, j, modulus);
        const std::uint32_t even = data[base + j];
        const std::uint32_t odd = mul_mod_device(data[base + j + half], twiddle, modulus);
        const std::uint32_t sum = even + odd;
        data[base + j] = sum >= modulus ? sum - modulus : sum;
        data[base + j + half] = even >= odd ? even - odd : modulus - (odd - even);
    }
}

__global__ void ntt_stage_dual_kernel(
    std::uint32_t* lhs,
    std::uint32_t* rhs,
    const std::uint32_t* moduli,
    const std::uint32_t* stage_twiddles,
    std::size_t value_count,
    std::size_t ntt_size,
    std::size_t length,
    std::size_t stage_twiddle_base,
    std::size_t transform_count,
    bool use_cached_twiddles
) {
    const std::size_t half = length >> 1u;
    const std::size_t butterflies_per_transform = ntt_size >> 1u;
    const std::size_t total = transform_count * butterflies_per_transform;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t transform_index = index / butterflies_per_transform;
        const std::size_t butterfly = index % butterflies_per_transform;
        const std::size_t block = butterfly / half;
        const std::size_t j = butterfly % half;
        const std::size_t modulus_index = transform_index / value_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::uint32_t root = g_stage_roots[modulus_index];
        const std::uint32_t twiddle =
            use_cached_twiddles
                ? stage_twiddles[stage_twiddle_base + modulus_index * half + j]
                : pow_mod_device(root, j, modulus);
        const std::size_t transform_offset = transform_index * ntt_size;
        const std::size_t base = transform_offset + block * length;

        const std::uint32_t lhs_even = lhs[base + j];
        const std::uint32_t lhs_odd = mul_mod_device(lhs[base + j + half], twiddle, modulus);
        const std::uint32_t lhs_sum = lhs_even + lhs_odd;
        lhs[base + j] = lhs_sum >= modulus ? lhs_sum - modulus : lhs_sum;
        lhs[base + j + half] = lhs_even >= lhs_odd ? lhs_even - lhs_odd : modulus - (lhs_odd - lhs_even);

        const std::uint32_t rhs_even = rhs[base + j];
        const std::uint32_t rhs_odd = mul_mod_device(rhs[base + j + half], twiddle, modulus);
        const std::uint32_t rhs_sum = rhs_even + rhs_odd;
        rhs[base + j] = rhs_sum >= modulus ? rhs_sum - modulus : rhs_sum;
        rhs[base + j + half] = rhs_even >= rhs_odd ? rhs_even - rhs_odd : modulus - (rhs_odd - rhs_even);
    }
}

__global__ void pointwise_multiply_and_bit_reverse_kernel(
    std::uint32_t* lhs,
    const std::uint32_t* rhs,
    const std::uint32_t* moduli,
    std::size_t value_count,
    std::size_t ntt_size,
    int log_ntt_size,
    std::size_t transform_count
) {
    const std::size_t total = transform_count * ntt_size;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t transform_index = index / ntt_size;
        const std::size_t slot = index % ntt_size;
        const std::size_t reversed = bit_reverse_index(slot, log_ntt_size);
        if (slot > reversed) {
            continue;
        }
        const std::size_t modulus_index = transform_index / value_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::size_t transform_offset = transform_index * ntt_size;
        const std::size_t lhs_index = transform_offset + slot;
        if (slot == reversed) {
            lhs[lhs_index] = mul_mod_device(lhs[lhs_index], rhs[lhs_index], modulus);
            continue;
        }

        const std::size_t reversed_index = transform_offset + reversed;
        const std::uint32_t lhs_slot = lhs[lhs_index];
        const std::uint32_t rhs_slot = rhs[lhs_index];
        const std::uint32_t lhs_reversed = lhs[reversed_index];
        const std::uint32_t rhs_reversed = rhs[reversed_index];
        lhs[lhs_index] = mul_mod_device(lhs_reversed, rhs_reversed, modulus);
        lhs[reversed_index] = mul_mod_device(lhs_slot, rhs_slot, modulus);
    }
}

__global__ void ntt_final_stage_dual_pointwise_bit_reverse_kernel(
    const std::uint32_t* lhs,
    const std::uint32_t* rhs,
    const std::uint32_t* moduli,
    const std::uint32_t* stage_twiddles,
    std::uint32_t* out,
    std::size_t value_count,
    std::size_t ntt_size,
    int log_ntt_size,
    std::size_t transform_count
) {
    const std::size_t half = ntt_size >> 1u;
    const std::size_t total = transform_count * half;
    const std::size_t stage_base = stage_twiddle_base_from_half(half);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t transform_index = index / half;
        const std::size_t j = index % half;
        const std::size_t modulus_index = transform_index / value_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::size_t transform_offset = transform_index * ntt_size;
        const std::uint32_t twiddle = stage_twiddles[stage_base + modulus_index * half + j];

        const std::uint32_t lhs_even = lhs[transform_offset + j];
        const std::uint32_t lhs_odd = mul_mod_device(lhs[transform_offset + j + half], twiddle, modulus);
        const std::uint32_t lhs_sum = lhs_even + lhs_odd;
        const std::uint32_t lhs_out0 = lhs_sum >= modulus ? lhs_sum - modulus : lhs_sum;
        const std::uint32_t lhs_out1 =
            lhs_even >= lhs_odd ? lhs_even - lhs_odd : modulus - (lhs_odd - lhs_even);

        const std::uint32_t rhs_even = rhs[transform_offset + j];
        const std::uint32_t rhs_odd = mul_mod_device(rhs[transform_offset + j + half], twiddle, modulus);
        const std::uint32_t rhs_sum = rhs_even + rhs_odd;
        const std::uint32_t rhs_out0 = rhs_sum >= modulus ? rhs_sum - modulus : rhs_sum;
        const std::uint32_t rhs_out1 =
            rhs_even >= rhs_odd ? rhs_even - rhs_odd : modulus - (rhs_odd - rhs_even);

        const std::size_t dst0 = transform_offset + bit_reverse_index(j, log_ntt_size);
        const std::size_t dst1 = transform_offset + bit_reverse_index(j + half, log_ntt_size);
        out[dst0] = mul_mod_device(lhs_out0, rhs_out0, modulus);
        out[dst1] = mul_mod_device(lhs_out1, rhs_out1, modulus);
    }
}

__global__ void ntt_stages_shared_kernel(
    std::uint32_t* data,
    const std::uint32_t* moduli,
    const std::uint32_t* stage_twiddles,
    std::size_t value_count,
    std::size_t ntt_size,
    std::size_t stage_start_length,
    std::size_t tile_size,
    std::size_t transform_count
) {
    extern __shared__ std::uint32_t shared[];

    const std::size_t tiles_per_transform = ntt_size / tile_size;
    const std::size_t block_index = static_cast<std::size_t>(blockIdx.x);
    const std::size_t transform_index = block_index / tiles_per_transform;
    const std::size_t tile_index = block_index % tiles_per_transform;
    if (transform_index >= transform_count) {
        return;
    }

    const std::size_t modulus_index = transform_index / value_count;
    const std::size_t tile_offset = transform_index * ntt_size + tile_index * tile_size;
    const std::uint32_t modulus = moduli[modulus_index];

    for (std::size_t offset = threadIdx.x; offset < tile_size; offset += blockDim.x) {
        shared[offset] = data[tile_offset + offset];
    }
    __syncthreads();

    for (std::size_t length = stage_start_length; length <= tile_size; length <<= 1u) {
        const std::size_t half = length >> 1u;
        const std::size_t butterfly_count = tile_size >> 1u;
        const std::size_t stage_base = stage_twiddle_base_from_half(half);
        const std::size_t modulus_base = stage_base + modulus_index * half;

        for (std::size_t butterfly = threadIdx.x; butterfly < butterfly_count; butterfly += blockDim.x) {
            const std::size_t block = butterfly / half;
            const std::size_t j = butterfly % half;
            const std::size_t base = block * length;
            const std::uint32_t twiddle = stage_twiddles[modulus_base + j];
            const std::uint32_t even = shared[base + j];
            const std::uint32_t odd = mul_mod_device(shared[base + j + half], twiddle, modulus);
            const std::uint32_t sum = even + odd;
            shared[base + j] = sum >= modulus ? sum - modulus : sum;
            shared[base + j + half] = even >= odd ? even - odd : modulus - (odd - even);
        }
        __syncthreads();
    }

    for (std::size_t offset = threadIdx.x; offset < tile_size; offset += blockDim.x) {
        data[tile_offset + offset] = shared[offset];
    }
}

__global__ void ntt_final_stage_scale_out_kernel(
    const std::uint32_t* src,
    const std::uint32_t* moduli,
    const std::uint32_t* stage_twiddles,
    std::uint32_t* out,
    std::size_t value_count,
    std::size_t out_slot_count,
    std::size_t ntt_size,
    std::size_t transform_count
) {
    const std::size_t half = ntt_size >> 1u;
    const std::size_t total = transform_count * half;
    const std::size_t stage_base = stage_twiddle_base_from_half(half);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t transform_index = index / half;
        const std::size_t j = index % half;
        const std::size_t modulus_index = transform_index / value_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::size_t transform_offset = transform_index * ntt_size;
        const std::size_t out_offset = transform_index * out_slot_count;
        const std::uint32_t twiddle = stage_twiddles[stage_base + modulus_index * half + j];

        const std::uint32_t even = src[transform_offset + j];
        const std::uint32_t odd = mul_mod_device(src[transform_offset + j + half], twiddle, modulus);
        const std::uint32_t sum = even + odd;
        const std::uint32_t even_out = sum >= modulus ? sum - modulus : sum;
        const std::uint32_t odd_out = even >= odd ? even - odd : modulus - (odd - even);

        if (j < out_slot_count) {
            out[out_offset + j] = mul_mod_device(even_out, g_inverse_scales[modulus_index], modulus);
        }
        const std::size_t odd_slot = j + half;
        if (odd_slot < out_slot_count) {
            out[out_offset + odd_slot] = mul_mod_device(odd_out, g_inverse_scales[modulus_index], modulus);
        }
    }
}

__global__ void scale_and_copy_out_prefix_kernel(
    const std::uint32_t* src,
    const std::uint32_t* moduli,
    std::uint32_t* out,
    std::size_t value_count,
    std::size_t out_slot_count,
    std::size_t ntt_size,
    std::size_t transform_count
) {
    const std::size_t total = transform_count * out_slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t transform_index = index / out_slot_count;
        const std::size_t modulus_index = transform_index / value_count;
        const std::size_t slot = index % out_slot_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::uint32_t scaled = mul_mod_device(
            src[transform_index * ntt_size + slot],
            g_inverse_scales[modulus_index],
            modulus
        );
        out[index] = scaled;
    }
}

__global__ void merge_metadata_kernel(
    const std::int8_t* lhs_signs,
    const std::int8_t* rhs_signs,
    const std::uint32_t* lhs_logical_slots,
    const std::uint32_t* rhs_logical_slots,
    const std::uint32_t* lhs_scale_bits,
    const std::uint32_t* rhs_scale_bits,
    const std::uint32_t* lhs_levels,
    const std::uint32_t* rhs_levels,
    std::int8_t* out_signs,
    std::uint32_t* out_logical_slots,
    std::uint32_t* out_scale_bits,
    std::uint32_t* out_levels,
    std::size_t value_count
) {
    for (std::size_t value_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         value_index < value_count;
         value_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        out_signs[value_index] = (lhs_signs[value_index] != 0 || rhs_signs[value_index] != 0) ? 1 : 0;
        out_logical_slots[value_index] =
            lhs_logical_slots[value_index] >= rhs_logical_slots[value_index]
                ? lhs_logical_slots[value_index]
                : rhs_logical_slots[value_index];
        out_scale_bits[value_index] =
            lhs_scale_bits[value_index] >= rhs_scale_bits[value_index]
                ? lhs_scale_bits[value_index]
                : rhs_scale_bits[value_index];
        out_levels[value_index] =
            lhs_levels[value_index] <= rhs_levels[value_index] ? lhs_levels[value_index] : rhs_levels[value_index];
    }
}

__global__ void multiply_metadata_kernel(
    const std::int8_t* lhs_signs,
    const std::int8_t* rhs_signs,
    const std::uint32_t* lhs_logical_slots,
    const std::uint32_t* rhs_logical_slots,
    const std::uint32_t* lhs_scale_bits,
    const std::uint32_t* rhs_scale_bits,
    const std::uint32_t* lhs_levels,
    const std::uint32_t* rhs_levels,
    std::int8_t* out_signs,
    std::uint32_t* out_logical_slots,
    std::uint32_t* out_scale_bits,
    std::uint32_t* out_levels,
    std::size_t value_count
) {
    for (std::size_t value_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         value_index < value_count;
         value_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        out_signs[value_index] = (lhs_signs[value_index] != 0 && rhs_signs[value_index] != 0) ? 1 : 0;
        out_logical_slots[value_index] =
            lhs_logical_slots[value_index] >= rhs_logical_slots[value_index]
                ? lhs_logical_slots[value_index]
                : rhs_logical_slots[value_index];
        out_scale_bits[value_index] = lhs_scale_bits[value_index] + rhs_scale_bits[value_index];
        out_levels[value_index] =
            lhs_levels[value_index] <= rhs_levels[value_index] ? lhs_levels[value_index] : rhs_levels[value_index];
    }
}

__global__ void convolution_metadata_kernel(
    const std::int8_t* lhs_signs,
    const std::int8_t* rhs_signs,
    const std::uint32_t* lhs_logical_slots,
    const std::uint32_t* rhs_logical_slots,
    const std::uint32_t* lhs_scale_bits,
    const std::uint32_t* rhs_scale_bits,
    const std::uint32_t* lhs_levels,
    const std::uint32_t* rhs_levels,
    std::int8_t* out_signs,
    std::uint32_t* out_logical_slots,
    std::uint32_t* out_scale_bits,
    std::uint32_t* out_levels,
    std::size_t value_count
) {
    for (std::size_t value_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         value_index < value_count;
         value_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        out_signs[value_index] = (lhs_signs[value_index] != 0 && rhs_signs[value_index] != 0) ? 1 : 0;
        out_logical_slots[value_index] = lhs_logical_slots[value_index] + rhs_logical_slots[value_index] - 1;
        out_scale_bits[value_index] = lhs_scale_bits[value_index] + rhs_scale_bits[value_index];
        out_levels[value_index] =
            lhs_levels[value_index] <= rhs_levels[value_index] ? lhs_levels[value_index] : rhs_levels[value_index];
    }
}

__global__ void set_uniform_metadata_kernel(
    std::int8_t* signs,
    std::uint32_t* logical_slots,
    std::uint32_t* scale_bits,
    std::uint32_t* levels,
    std::size_t value_count,
    std::int8_t sign,
    std::uint32_t logical_slot_count,
    std::uint32_t scale_bit_count,
    std::uint32_t level
) {
    for (std::size_t value_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         value_index < value_count;
         value_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        signs[value_index] = sign;
        logical_slots[value_index] = logical_slot_count;
        scale_bits[value_index] = scale_bit_count;
        levels[value_index] = level;
    }
}

__global__ void copy_metadata_with_level_drop_kernel(
    const std::int8_t* src_signs,
    const std::uint32_t* src_logical_slots,
    const std::uint32_t* src_scale_bits,
    const std::uint32_t* src_levels,
    std::int8_t* dst_signs,
    std::uint32_t* dst_logical_slots,
    std::uint32_t* dst_scale_bits,
    std::uint32_t* dst_levels,
    std::size_t value_count,
    std::uint32_t dst_slot_count,
    std::uint32_t target_level,
    std::uint32_t scale_bits_delta
) {
    for (std::size_t value_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         value_index < value_count;
         value_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        dst_signs[value_index] = src_signs[value_index];
        const std::uint32_t logical = src_logical_slots[value_index];
        dst_logical_slots[value_index] = logical <= dst_slot_count ? logical : dst_slot_count;
        dst_scale_bits[value_index] = src_scale_bits[value_index] + scale_bits_delta;
        const std::uint32_t src_level = src_levels[value_index];
        dst_levels[value_index] = src_level <= target_level ? src_level : target_level;
    }
}

__global__ void copy_residues_with_zero_suffix_kernel(
    const std::uint32_t* src_residues,
    std::uint32_t* dst_residues,
    std::size_t value_count,
    std::size_t src_slot_count,
    std::size_t dst_slot_count,
    int modulus_count
) {
    const std::size_t total_count =
        static_cast<std::size_t>(modulus_count) * value_count * dst_slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_count;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % dst_slot_count;
        const std::size_t value_index = (index / dst_slot_count) % value_count;
        const std::size_t modulus_index = index / (dst_slot_count * value_count);
        if (slot_index < src_slot_count) {
            const std::size_t src_index =
                (modulus_index * value_count + value_index) * src_slot_count + slot_index;
            dst_residues[index] = src_residues[src_index];
        } else {
            dst_residues[index] = 0u;
        }
    }
}

std::uint64_t mod_inverse(std::uint64_t a, std::uint64_t modulus) {
    std::int64_t t = 0;
    std::int64_t new_t = 1;
    std::int64_t r = static_cast<std::int64_t>(modulus);
    std::int64_t new_r = static_cast<std::int64_t>(a % modulus);
    while (new_r != 0) {
        const std::int64_t quotient = r / new_r;
        const std::int64_t next_t = t - quotient * new_t;
        const std::int64_t next_r = r - quotient * new_r;
        t = new_t;
        new_t = next_t;
        r = new_r;
        new_r = next_r;
    }
    if (r != 1) {
        throw std::runtime_error("mod_inverse requires coprime inputs");
    }
    if (t < 0) {
        t += static_cast<std::int64_t>(modulus);
    }
    return static_cast<std::uint64_t>(t);
}

std::uint64_t reconstruct_scalar(
    const HostRnsTensor& tensor,
    std::size_t value_index,
    std::size_t slot_index
) {
    unsigned __int128 x = 0;
    unsigned __int128 product = 1;
    for (int modulus_index = 0; modulus_index < tensor.modulus_count; ++modulus_index) {
        const auto modulus = static_cast<std::uint64_t>(tensor.moduli[modulus_index]);
        const std::size_t residue_offset =
            (static_cast<std::size_t>(modulus_index) * tensor.value_count + value_index) * tensor.slot_count + slot_index;
        const auto residue = static_cast<std::uint64_t>(tensor.residues[residue_offset]);
        if (modulus_index == 0) {
            x = residue;
            product = modulus;
            continue;
        }
        const std::uint64_t x_mod = static_cast<std::uint64_t>(x % modulus);
        const std::uint64_t delta = (residue + modulus - x_mod) % modulus;
        const std::uint64_t inverse = mod_inverse(static_cast<std::uint64_t>(product % modulus), modulus);
        const std::uint64_t mixed = static_cast<std::uint64_t>((static_cast<unsigned __int128>(delta) * inverse) % modulus);
        x += product * mixed;
        product *= modulus;
    }
    if (x > static_cast<unsigned __int128>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error("reconstructed value does not fit into uint64");
    }
    return static_cast<std::uint64_t>(x);
}

std::uint64_t reconstruct_scalar_from_residue_slice(
    const std::uint32_t* residues,
    std::size_t modulus_count,
    std::size_t sample_count,
    std::size_t sample_index
) {
    unsigned __int128 x = 0;
    unsigned __int128 product = 1;
    for (std::size_t modulus_index = 0; modulus_index < modulus_count; ++modulus_index) {
        const auto modulus = static_cast<std::uint64_t>(ModulusConfig::kModuli[modulus_index]);
        const auto residue = static_cast<std::uint64_t>(residues[modulus_index * sample_count + sample_index]);
        if (modulus_index == 0) {
            x = residue;
            product = modulus;
            continue;
        }
        const std::uint64_t x_mod = static_cast<std::uint64_t>(x % modulus);
        const std::uint64_t delta = (residue + modulus - x_mod) % modulus;
        const std::uint64_t inverse = mod_inverse(static_cast<std::uint64_t>(product % modulus), modulus);
        const std::uint64_t mixed = static_cast<std::uint64_t>((static_cast<unsigned __int128>(delta) * inverse) % modulus);
        x += product * mixed;
        product *= modulus;
    }
    if (x > static_cast<unsigned __int128>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error("reconstructed sampled value does not fit into uint64");
    }
    return static_cast<std::uint64_t>(x);
}

std::uint64_t bigint_mod_u64(const HostBigInt& value, std::uint64_t modulus) {
    std::uint64_t remainder = 0;
    for (std::size_t limb_index = value.limbs.size(); limb_index > 0; --limb_index) {
        remainder = static_cast<std::uint64_t>(
            ((static_cast<unsigned __int128>(remainder) << 32u) + value.limbs[limb_index - 1]) % modulus
        );
    }
    return remainder;
}

HostBigInt reconstruct_scalar_bigint(
    const HostRnsTensor& tensor,
    std::size_t value_index,
    std::size_t slot_index
) {
    HostBigInt x;
    HostBigInt product = 1;
    for (int modulus_index = 0; modulus_index < tensor.modulus_count; ++modulus_index) {
        const auto modulus = static_cast<std::uint64_t>(tensor.moduli[modulus_index]);
        const std::size_t residue_offset =
            (static_cast<std::size_t>(modulus_index) * tensor.value_count + value_index) * tensor.slot_count + slot_index;
        const auto residue = static_cast<std::uint64_t>(tensor.residues[residue_offset]);
        if (modulus_index == 0) {
            x = static_cast<long long>(residue);
            product = static_cast<long long>(modulus);
            continue;
        }
        const std::uint64_t x_mod = bigint_mod_u64(x, modulus);
        const std::uint64_t delta = (residue + modulus - x_mod) % modulus;
        const std::uint64_t inverse = mod_inverse(bigint_mod_u64(product, modulus), modulus);
        const std::uint64_t mixed = static_cast<std::uint64_t>((static_cast<unsigned __int128>(delta) * inverse) % modulus);
        x = x + product * static_cast<long long>(mixed);
        product = product * static_cast<long long>(modulus);
    }
    return x;
}

HostBigInt modulus_product_bigint(const HostRnsTensor& tensor) {
    HostBigInt product = 1;
    for (int modulus_index = 0; modulus_index < tensor.modulus_count; ++modulus_index) {
        product = product * static_cast<long long>(tensor.moduli[modulus_index]);
    }
    return product;
}

HostBigInt center_lift_residue_bigint(const HostBigInt& residue, const HostBigInt& modulus_product) {
    if (abs_compare(residue * 2, modulus_product) > 0) {
        return residue - modulus_product;
    }
    return residue;
}

std::vector<HostBigInt> reconstruct_centered_coefficients(const HostRnsTensor& tensor) {
    std::vector<HostBigInt> coefficients(tensor.value_count * tensor.slot_count);
    const HostBigInt product = modulus_product_bigint(tensor);
    for (std::size_t value_index = 0; value_index < tensor.value_count; ++value_index) {
        for (std::size_t slot_index = 0; slot_index < tensor.slot_count; ++slot_index) {
            const HostBigInt residue = reconstruct_scalar_bigint(tensor, value_index, slot_index);
            coefficients[value_index * tensor.slot_count + slot_index] = center_lift_residue_bigint(residue, product);
        }
    }
    return coefficients;
}

void launch_binary_metadata_merge(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    merge_metadata_kernel<<<block_count(lhs.value_count), kThreadsPerBlock>>>(
        lhs.d_signs,
        rhs.d_signs,
        lhs.d_logical_slots,
        rhs.d_logical_slots,
        lhs.d_scale_bits,
        rhs.d_scale_bits,
        lhs.d_levels,
        rhs.d_levels,
        out.d_signs,
        out.d_logical_slots,
        out.d_scale_bits,
        out.d_levels,
        lhs.value_count
    );
    check_cuda(cudaGetLastError(), "merge_metadata_kernel launch");
}

void launch_pointwise_multiply_metadata_merge(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    multiply_metadata_kernel<<<block_count(lhs.value_count), kThreadsPerBlock>>>(
        lhs.d_signs,
        rhs.d_signs,
        lhs.d_logical_slots,
        rhs.d_logical_slots,
        lhs.d_scale_bits,
        rhs.d_scale_bits,
        lhs.d_levels,
        rhs.d_levels,
        out.d_signs,
        out.d_logical_slots,
        out.d_scale_bits,
        out.d_levels,
        lhs.value_count
    );
    check_cuda(cudaGetLastError(), "multiply_metadata_kernel launch");
}

void launch_convolution_metadata_merge(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    convolution_metadata_kernel<<<block_count(lhs.value_count), kThreadsPerBlock>>>(
        lhs.d_signs,
        rhs.d_signs,
        lhs.d_logical_slots,
        rhs.d_logical_slots,
        lhs.d_scale_bits,
        rhs.d_scale_bits,
        lhs.d_levels,
        rhs.d_levels,
        out.d_signs,
        out.d_logical_slots,
        out.d_scale_bits,
        out.d_levels,
        lhs.value_count
    );
    check_cuda(cudaGetLastError(), "convolution_metadata_kernel launch");
}

struct PipelineTiming {
    double encode_ms = 0.0;
    double pointwise_ms = 0.0;
    double convolution_ms = 0.0;
    double kernel_ms = 0.0;
};

struct SampleGatherWorkspace {
    std::uint32_t* d_indices = nullptr;
    std::uint32_t* d_residues = nullptr;
    std::size_t index_capacity = 0;
    std::size_t residue_capacity = 0;
};

SampleGatherWorkspace& sample_gather_workspace() {
    static SampleGatherWorkspace workspace;
    return workspace;
}

void ensure_sample_gather_workspace(std::size_t sample_count) {
    auto& workspace = sample_gather_workspace();
    if (workspace.index_capacity < sample_count) {
        if (workspace.d_indices != nullptr) {
            check_cuda(cudaFree(workspace.d_indices), "cudaFree sample gather indices");
            workspace.d_indices = nullptr;
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&workspace.d_indices), sizeof(std::uint32_t) * sample_count),
            "cudaMalloc sample gather indices"
        );
        workspace.index_capacity = sample_count;
    }

    const std::size_t residue_capacity =
        static_cast<std::size_t>(ModulusConfig::kModulusCount) * sample_count;
    if (workspace.residue_capacity < residue_capacity) {
        if (workspace.d_residues != nullptr) {
            check_cuda(cudaFree(workspace.d_residues), "cudaFree sample gather residues");
            workspace.d_residues = nullptr;
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&workspace.d_residues), sizeof(std::uint32_t) * residue_capacity),
            "cudaMalloc sample gather residues"
        );
        workspace.residue_capacity = residue_capacity;
    }
}

void ensure_sample_gather_workspace_triplet(std::size_t sample_count) {
    ensure_sample_gather_workspace(sample_count);
    auto& workspace = sample_gather_workspace();
    const std::size_t residue_capacity =
        static_cast<std::size_t>(ModulusConfig::kModulusCount) * sample_count * 3u;
    if (workspace.residue_capacity < residue_capacity) {
        if (workspace.d_residues != nullptr) {
            check_cuda(cudaFree(workspace.d_residues), "cudaFree sample gather residues");
            workspace.d_residues = nullptr;
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&workspace.d_residues), sizeof(std::uint32_t) * residue_capacity),
            "cudaMalloc sample gather residues triplet"
        );
        workspace.residue_capacity = residue_capacity;
    }
}

struct EncodeWorkspace {
    std::uint64_t* lhs_values = nullptr;
    std::uint64_t* rhs_values = nullptr;
    std::size_t capacity = 0;
};

EncodeWorkspace& encode_workspace() {
    static EncodeWorkspace workspace;
    return workspace;
}

void ensure_encode_workspace(std::size_t scalar_capacity) {
    auto& workspace = encode_workspace();
    if (workspace.capacity >= scalar_capacity) {
        return;
    }

    if (workspace.lhs_values != nullptr) {
        check_cuda(cudaFree(workspace.lhs_values), "cudaFree encode workspace lhs_values");
        workspace.lhs_values = nullptr;
    }
    if (workspace.rhs_values != nullptr) {
        check_cuda(cudaFree(workspace.rhs_values), "cudaFree encode workspace rhs_values");
        workspace.rhs_values = nullptr;
    }

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&workspace.lhs_values), sizeof(std::uint64_t) * scalar_capacity),
        "cudaMalloc encode workspace lhs_values"
    );
    try {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&workspace.rhs_values), sizeof(std::uint64_t) * scalar_capacity),
            "cudaMalloc encode workspace rhs_values"
        );
    } catch (...) {
        cudaFree(workspace.lhs_values);
        workspace.lhs_values = nullptr;
        workspace.capacity = 0;
        throw;
    }
    workspace.capacity = scalar_capacity;
}

void check_encode_pair_shapes(
    const DeviceRnsTensor& lhs_tensor,
    const DeviceRnsTensor& rhs_tensor
) {
    if (lhs_tensor.value_count != rhs_tensor.value_count ||
        lhs_tensor.slot_count != rhs_tensor.slot_count ||
        lhs_tensor.modulus_count != rhs_tensor.modulus_count) {
        throw std::invalid_argument("encode_u64_pair requires matching tensor shapes");
    }
}

void upload_u64_pair_to_encode_workspace(
    const std::vector<std::uint64_t>& lhs_values,
    const std::vector<std::uint64_t>& rhs_values
) {
    if (lhs_values.size() != rhs_values.size()) {
        throw std::invalid_argument("upload_u64_pair_to_encode_workspace requires matching host vector sizes");
    }

    const std::size_t scalar_capacity = lhs_values.size();
    ensure_encode_workspace(scalar_capacity);
    auto& workspace = encode_workspace();

    check_cuda(
        cudaMemcpy(
            workspace.lhs_values,
            lhs_values.data(),
            sizeof(std::uint64_t) * scalar_capacity,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy lhs values"
    );
    check_cuda(
        cudaMemcpy(
            workspace.rhs_values,
            rhs_values.data(),
            sizeof(std::uint64_t) * scalar_capacity,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy rhs values"
    );
}

void encode_u64_pair_from_workspace(
    DeviceRnsTensor& lhs_tensor,
    DeviceRnsTensor& rhs_tensor
) {
    check_encode_pair_shapes(lhs_tensor, rhs_tensor);
    const std::size_t scalar_capacity = scalar_count(lhs_tensor);
    auto& workspace = encode_workspace();
    if (workspace.capacity < scalar_capacity || workspace.lhs_values == nullptr || workspace.rhs_values == nullptr) {
        throw std::runtime_error("encode workspace is not prepared for encode_u64_pair_from_workspace");
    }

    encode_u64_dual_kernel<<<block_count(residue_count(lhs_tensor)), kThreadsPerBlock>>>(
        workspace.lhs_values,
        workspace.rhs_values,
        lhs_tensor.d_moduli,
        lhs_tensor.d_residues,
        rhs_tensor.d_residues,
        scalar_capacity,
        lhs_tensor.modulus_count
    );
    check_cuda(cudaGetLastError(), "encode_u64_dual_kernel launch");

    init_metadata_dual_kernel<<<block_count(lhs_tensor.value_count), kThreadsPerBlock>>>(
        workspace.lhs_values,
        workspace.rhs_values,
        lhs_tensor.d_signs,
        lhs_tensor.d_logical_slots,
        lhs_tensor.d_scale_bits,
        lhs_tensor.d_levels,
        rhs_tensor.d_signs,
        rhs_tensor.d_logical_slots,
        rhs_tensor.d_scale_bits,
        rhs_tensor.d_levels,
        lhs_tensor.value_count,
        lhs_tensor.slot_count,
        static_cast<std::uint32_t>(lhs_tensor.modulus_count)
    );
    check_cuda(cudaGetLastError(), "init_metadata_dual_kernel launch");
}

struct GlobalNttWorkspace {
    std::uint32_t* lhs = nullptr;
    std::uint32_t* rhs = nullptr;
    std::uint32_t* scratch = nullptr;
    std::size_t capacity = 0;
};

GlobalNttWorkspace& global_ntt_workspace() {
    static GlobalNttWorkspace workspace;
    return workspace;
}

void ensure_global_ntt_workspace(std::size_t total_ntt_values) {
    auto& workspace = global_ntt_workspace();
    if (workspace.capacity >= total_ntt_values &&
        workspace.lhs != nullptr &&
        workspace.rhs != nullptr &&
        workspace.scratch != nullptr) {
        return;
    }

    if (workspace.lhs != nullptr) {
        check_cuda(cudaFree(workspace.lhs), "cudaFree workspace lhs");
        workspace.lhs = nullptr;
    }
    if (workspace.rhs != nullptr) {
        check_cuda(cudaFree(workspace.rhs), "cudaFree workspace rhs");
        workspace.rhs = nullptr;
    }
    if (workspace.scratch != nullptr) {
        check_cuda(cudaFree(workspace.scratch), "cudaFree workspace scratch");
        workspace.scratch = nullptr;
    }

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&workspace.lhs), sizeof(std::uint32_t) * total_ntt_values),
        "cudaMalloc workspace lhs"
    );
    try {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&workspace.rhs), sizeof(std::uint32_t) * total_ntt_values),
            "cudaMalloc workspace rhs"
        );
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&workspace.scratch), sizeof(std::uint32_t) * total_ntt_values),
            "cudaMalloc workspace scratch"
        );
    } catch (...) {
        cudaFree(workspace.lhs);
        cudaFree(workspace.rhs);
        workspace.lhs = nullptr;
        workspace.rhs = nullptr;
        workspace.capacity = 0;
        throw;
    }
    workspace.capacity = total_ntt_values;
}

PipelineTiming execute_pipeline(
    DeviceRnsTensor& lhs,
    DeviceRnsTensor& rhs,
    DeviceRnsTensor& sum,
    DeviceRnsTensor& diff,
    DeviceRnsTensor& prod,
    DeviceRnsTensor& conv
) {
    cudaEvent_t start = nullptr;
    cudaEvent_t after_encode = nullptr;
    cudaEvent_t after_pointwise = nullptr;
    cudaEvent_t after_convolution = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&after_encode), "cudaEventCreate after_encode");
    check_cuda(cudaEventCreate(&after_pointwise), "cudaEventCreate after_pointwise");
    check_cuda(cudaEventCreate(&after_convolution), "cudaEventCreate after_convolution");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    encode_u64_pair_from_workspace(lhs, rhs);
    check_cuda(cudaEventRecord(after_encode), "cudaEventRecord after_encode");

    pointwise_add(lhs, rhs, sum);
    pointwise_sub(lhs, rhs, diff);
    pointwise_mul(lhs, rhs, prod);
    check_cuda(cudaEventRecord(after_pointwise), "cudaEventRecord after_pointwise");

    pairwise_convolution(lhs, rhs, conv);
    check_cuda(cudaEventRecord(after_convolution), "cudaEventRecord after_convolution");
    check_cuda(cudaEventSynchronize(after_convolution), "cudaEventSynchronize after_convolution");

    float encode_ms = 0.0f;
    float pointwise_ms = 0.0f;
    float convolution_ms = 0.0f;
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&encode_ms, start, after_encode), "cudaEventElapsedTime encode");
    check_cuda(
        cudaEventElapsedTime(&pointwise_ms, after_encode, after_pointwise),
        "cudaEventElapsedTime pointwise"
    );
    check_cuda(
        cudaEventElapsedTime(&convolution_ms, after_pointwise, after_convolution),
        "cudaEventElapsedTime convolution"
    );
    check_cuda(cudaEventElapsedTime(&kernel_ms, start, after_convolution), "cudaEventElapsedTime kernel");

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(after_encode), "cudaEventDestroy after_encode");
    check_cuda(cudaEventDestroy(after_pointwise), "cudaEventDestroy after_pointwise");
    check_cuda(cudaEventDestroy(after_convolution), "cudaEventDestroy after_convolution");

    PipelineTiming timing;
    timing.encode_ms = static_cast<double>(encode_ms);
    timing.pointwise_ms = static_cast<double>(pointwise_ms);
    timing.convolution_ms = static_cast<double>(convolution_ms);
    timing.kernel_ms = static_cast<double>(kernel_ms);
    return timing;
}

void staged_global_ntt_convolution(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    const std::size_t ntt_size = resolve_convolution_ntt_size(out.slot_count);
    const int log_ntt_size = ceil_log2_size(ntt_size);
    const std::size_t transform_count = static_cast<std::size_t>(lhs.modulus_count) * lhs.value_count;
    const std::size_t total_ntt_values = transform_count * ntt_size;
    const std::size_t fused_tile_size = std::min(kGlobalFusedStageTileSize, ntt_size);
    const std::size_t mid_fused_tile_size = std::min(kGlobalMidFusedStageTileSize, ntt_size);
    const bool use_fused_initial_stages =
        fused_tile_size >= 2 &&
        fused_tile_size < ntt_size &&
        (ntt_size % fused_tile_size == 0);
    const bool use_fused_mid_stages =
        use_fused_initial_stages &&
        mid_fused_tile_size > fused_tile_size &&
        mid_fused_tile_size < ntt_size &&
        (ntt_size % mid_fused_tile_size == 0);
    ensure_global_ntt_workspace(total_ntt_values);
    ensure_stage_twiddle_cache();

    auto& workspace = global_ntt_workspace();
    const auto& twiddle_cache = stage_twiddle_cache();
    std::uint32_t* d_lhs_ntt = workspace.lhs;
    std::uint32_t* d_rhs_ntt = workspace.rhs;
    std::uint32_t* d_scratch_ntt = workspace.scratch;

    pad_and_bit_reverse_dual_kernel<<<block_count(total_ntt_values), kThreadsPerBlock>>>(
            lhs.d_residues,
            rhs.d_residues,
            d_lhs_ntt,
            d_rhs_ntt,
            lhs.value_count,
            lhs.slot_count,
            rhs.slot_count,
            ntt_size,
            log_ntt_size,
            transform_count
        );
    check_cuda(cudaGetLastError(), "pad_and_bit_reverse_dual_kernel launch");

    std::size_t forward_start_length = 2;
    if (use_fused_initial_stages) {
        const std::size_t tile_block_count = transform_count * (ntt_size / fused_tile_size);
        const std::size_t shared_bytes = sizeof(std::uint32_t) * fused_tile_size * 2;
        ntt_stages_dual_shared_kernel<<<static_cast<int>(tile_block_count), kThreadsPerBlock, shared_bytes>>>(
            d_lhs_ntt,
            d_rhs_ntt,
            lhs.d_moduli,
            twiddle_cache.forward,
            lhs.value_count,
            ntt_size,
            2,
            fused_tile_size,
            transform_count
        );
        check_cuda(cudaGetLastError(), "ntt_stages_dual_shared_kernel initial launch");
        forward_start_length = fused_tile_size << 1u;
    }
    if (use_fused_mid_stages && forward_start_length <= mid_fused_tile_size) {
        const std::size_t tile_block_count = transform_count * (ntt_size / mid_fused_tile_size);
        const std::size_t shared_bytes = sizeof(std::uint32_t) * mid_fused_tile_size * 2;
        ntt_stages_dual_shared_kernel<<<static_cast<int>(tile_block_count), kThreadsPerBlock, shared_bytes>>>(
            d_lhs_ntt,
            d_rhs_ntt,
            lhs.d_moduli,
            twiddle_cache.forward,
            lhs.value_count,
            ntt_size,
            forward_start_length,
            mid_fused_tile_size,
            transform_count
        );
        check_cuda(cudaGetLastError(), "ntt_stages_dual_shared_kernel mid launch");
        forward_start_length = mid_fused_tile_size << 1u;
    }

    const bool use_fused_forward_final_stage =
        forward_start_length == ntt_size &&
        ntt_size <= kMaxCachedStageLength;

    for (std::size_t length = forward_start_length; length <= ntt_size; length <<= 1u) {
        if (use_fused_forward_final_stage && length == ntt_size) {
            break;
        }
        const bool use_cached_twiddles = length <= kMaxCachedStageLength;
        const std::uint32_t* stage_twiddles = use_cached_twiddles ? twiddle_cache.forward : nullptr;
        const std::size_t stage_twiddle_base =
            use_cached_twiddles ? twiddle_cache.offsets[ceil_log2_size(length)] : 0;
        if (!use_cached_twiddles) {
            upload_stage_roots(length, false);
        }
        ntt_stage_dual_kernel<<<block_count(transform_count * (ntt_size >> 1u)), kThreadsPerBlock>>>(
            d_lhs_ntt,
            d_rhs_ntt,
            lhs.d_moduli,
            stage_twiddles,
            lhs.value_count,
            ntt_size,
            length,
            stage_twiddle_base,
            transform_count,
            use_cached_twiddles
        );
        check_cuda(cudaGetLastError(), "ntt_stage_dual_kernel forward launch");
    }

    if (use_fused_forward_final_stage) {
        ntt_final_stage_dual_pointwise_bit_reverse_kernel<<<
            block_count(transform_count * (ntt_size >> 1u)),
            kThreadsPerBlock
        >>>(
            d_lhs_ntt,
            d_rhs_ntt,
            lhs.d_moduli,
            twiddle_cache.forward,
            d_scratch_ntt,
            lhs.value_count,
            ntt_size,
            log_ntt_size,
            transform_count
        );
        check_cuda(cudaGetLastError(), "ntt_final_stage_dual_pointwise_bit_reverse_kernel launch");
        d_lhs_ntt = d_scratch_ntt;
    } else {
        pointwise_multiply_and_bit_reverse_kernel<<<block_count(total_ntt_values), kThreadsPerBlock>>>(
            d_lhs_ntt,
            d_rhs_ntt,
            lhs.d_moduli,
            lhs.value_count,
            ntt_size,
            log_ntt_size,
            transform_count
        );
        check_cuda(cudaGetLastError(), "pointwise_multiply_and_bit_reverse_kernel launch");
    }

    std::size_t inverse_start_length = 2;
    if (use_fused_initial_stages) {
        const std::size_t tile_block_count = transform_count * (ntt_size / fused_tile_size);
        const std::size_t shared_bytes = sizeof(std::uint32_t) * fused_tile_size;
        ntt_stages_shared_kernel<<<static_cast<int>(tile_block_count), kThreadsPerBlock, shared_bytes>>>(
            d_lhs_ntt,
            lhs.d_moduli,
            twiddle_cache.inverse,
            lhs.value_count,
            ntt_size,
            2,
            fused_tile_size,
            transform_count
        );
        check_cuda(cudaGetLastError(), "ntt_stages_shared_kernel initial launch");
        inverse_start_length = fused_tile_size << 1u;
    }
    if (use_fused_mid_stages && inverse_start_length <= mid_fused_tile_size) {
        const std::size_t tile_block_count = transform_count * (ntt_size / mid_fused_tile_size);
        const std::size_t shared_bytes = sizeof(std::uint32_t) * mid_fused_tile_size;
        ntt_stages_shared_kernel<<<static_cast<int>(tile_block_count), kThreadsPerBlock, shared_bytes>>>(
            d_lhs_ntt,
            lhs.d_moduli,
            twiddle_cache.inverse,
            lhs.value_count,
            ntt_size,
            inverse_start_length,
            mid_fused_tile_size,
            transform_count
        );
        check_cuda(cudaGetLastError(), "ntt_stages_shared_kernel mid launch");
        inverse_start_length = mid_fused_tile_size << 1u;
    }

    const bool use_fused_inverse_final_stage =
        inverse_start_length == ntt_size &&
        ntt_size <= kMaxCachedStageLength;

    for (std::size_t length = inverse_start_length; length <= ntt_size; length <<= 1u) {
        if (use_fused_inverse_final_stage && length == ntt_size) {
            break;
        }
        const bool use_cached_twiddles = length <= kMaxCachedStageLength;
        const std::uint32_t* stage_twiddles = use_cached_twiddles ? twiddle_cache.inverse : nullptr;
        const std::size_t stage_twiddle_base =
            use_cached_twiddles ? twiddle_cache.offsets[ceil_log2_size(length)] : 0;
        if (!use_cached_twiddles) {
            upload_stage_roots(length, true);
        }
        ntt_stage_kernel<<<block_count(transform_count * (ntt_size >> 1u)), kThreadsPerBlock>>>(
            d_lhs_ntt,
            lhs.d_moduli,
            stage_twiddles,
            lhs.value_count,
            ntt_size,
            length,
            stage_twiddle_base,
            transform_count,
            use_cached_twiddles
        );
        check_cuda(cudaGetLastError(), "ntt_stage_kernel inverse launch");
    }

    upload_inverse_scales(ntt_size);
    if (use_fused_inverse_final_stage) {
        ntt_final_stage_scale_out_kernel<<<block_count(transform_count * (ntt_size >> 1u)), kThreadsPerBlock>>>(
            d_lhs_ntt,
            lhs.d_moduli,
            twiddle_cache.inverse,
            out.d_residues,
            lhs.value_count,
            out.slot_count,
            ntt_size,
            transform_count
        );
        check_cuda(cudaGetLastError(), "ntt_final_stage_scale_out_kernel launch");
    } else {
        scale_and_copy_out_prefix_kernel<<<block_count(transform_count * out.slot_count), kThreadsPerBlock>>>(
            d_lhs_ntt,
            lhs.d_moduli,
            out.d_residues,
            lhs.value_count,
            out.slot_count,
            ntt_size,
            transform_count
        );
        check_cuda(cudaGetLastError(), "scale_and_copy_out_prefix_kernel launch");
    }
}

bool validate_results(
    const std::vector<std::uint64_t>& lhs_values,
    const std::vector<std::uint64_t>& rhs_values,
    const std::vector<std::uint64_t>& sum_host,
    const std::vector<std::uint64_t>& diff_host,
    const std::vector<std::uint64_t>& prod_host,
    const std::vector<std::uint64_t>& conv_host,
    std::size_t value_count,
    std::size_t slot_count
) {
    for (std::size_t index = 0; index < lhs_values.size(); ++index) {
        const std::uint64_t expected_sum = lhs_values[index] + rhs_values[index];
        const std::uint64_t expected_diff = lhs_values[index] - rhs_values[index];
        const std::uint64_t expected_prod = lhs_values[index] * rhs_values[index];
        if (sum_host[index] != expected_sum || diff_host[index] != expected_diff || prod_host[index] != expected_prod) {
            return false;
        }
    }

    for (std::size_t value_index = 0; value_index < value_count; ++value_index) {
        for (std::size_t out_slot = 0; out_slot < 2 * slot_count - 1; ++out_slot) {
            std::uint64_t expected = 0;
            const std::size_t lhs_begin = out_slot >= slot_count - 1 ? out_slot - (slot_count - 1) : 0;
            const std::size_t lhs_end = std::min<std::size_t>(slot_count - 1, out_slot);
            for (std::size_t lhs_slot = lhs_begin; lhs_slot <= lhs_end; ++lhs_slot) {
                const std::size_t rhs_slot = out_slot - lhs_slot;
                expected += lhs_values[value_index * slot_count + lhs_slot] * rhs_values[value_index * slot_count + rhs_slot];
            }
            const std::size_t conv_index = value_index * (2 * slot_count - 1) + out_slot;
            if (conv_host[conv_index] != expected) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::uint32_t> gather_sampled_residues(
    const DeviceRnsTensor& tensor,
    const std::vector<std::uint32_t>& sample_indices
) {
    if (sample_indices.empty()) {
        return {};
    }

    const std::size_t total_scalars = scalar_count(tensor);
    for (const std::uint32_t sample_index : sample_indices) {
        if (static_cast<std::size_t>(sample_index) >= total_scalars) {
            throw std::out_of_range("sample index exceeds tensor scalar count");
        }
    }

    ensure_sample_gather_workspace(sample_indices.size());
    auto& workspace = sample_gather_workspace();
    check_cuda(
        cudaMemcpy(
            workspace.d_indices,
            sample_indices.data(),
            sizeof(std::uint32_t) * sample_indices.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy sample gather indices"
    );

    gather_residue_samples_kernel<<<block_count(sample_indices.size() * static_cast<std::size_t>(tensor.modulus_count)), kThreadsPerBlock>>>(
        tensor.d_residues,
        workspace.d_indices,
        workspace.d_residues,
        total_scalars,
        sample_indices.size(),
        tensor.modulus_count
    );
    check_cuda(cudaGetLastError(), "gather_residue_samples_kernel launch");

    std::vector<std::uint32_t> host_residues(sample_indices.size() * static_cast<std::size_t>(tensor.modulus_count));
    check_cuda(
        cudaMemcpy(
            host_residues.data(),
            workspace.d_residues,
            sizeof(std::uint32_t) * host_residues.size(),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy sampled residues"
    );
    return host_residues;
}

std::array<std::vector<std::uint32_t>, 3> gather_sampled_residues_triplet(
    const DeviceRnsTensor& tensor0,
    const DeviceRnsTensor& tensor1,
    const DeviceRnsTensor& tensor2,
    const std::vector<std::uint32_t>& sample_indices
) {
    if (sample_indices.empty()) {
        return {};
    }
    if (tensor0.value_count != tensor1.value_count ||
        tensor0.value_count != tensor2.value_count ||
        tensor0.slot_count != tensor1.slot_count ||
        tensor0.slot_count != tensor2.slot_count ||
        tensor0.modulus_count != tensor1.modulus_count ||
        tensor0.modulus_count != tensor2.modulus_count) {
        throw std::invalid_argument("triplet sampled gather requires matching tensor shapes");
    }

    const std::size_t total_scalars = scalar_count(tensor0);
    for (const std::uint32_t sample_index : sample_indices) {
        if (static_cast<std::size_t>(sample_index) >= total_scalars) {
            throw std::out_of_range("sample index exceeds tensor scalar count");
        }
    }

    ensure_sample_gather_workspace_triplet(sample_indices.size());
    auto& workspace = sample_gather_workspace();
    check_cuda(
        cudaMemcpy(
            workspace.d_indices,
            sample_indices.data(),
            sizeof(std::uint32_t) * sample_indices.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy sample gather triplet indices"
    );

    const std::size_t per_tensor =
        sample_indices.size() * static_cast<std::size_t>(tensor0.modulus_count);
    gather_residue_samples_triplet_kernel<<<block_count(per_tensor * 3u), kThreadsPerBlock>>>(
        tensor0.d_residues,
        tensor1.d_residues,
        tensor2.d_residues,
        workspace.d_indices,
        workspace.d_residues,
        total_scalars,
        sample_indices.size(),
        tensor0.modulus_count
    );
    check_cuda(cudaGetLastError(), "gather_residue_samples_triplet_kernel launch");

    std::vector<std::uint32_t> host_residues(per_tensor * 3u);
    check_cuda(
        cudaMemcpy(
            host_residues.data(),
            workspace.d_residues,
            sizeof(std::uint32_t) * host_residues.size(),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy sampled residues triplet"
    );

    std::array<std::vector<std::uint32_t>, 3> out;
    for (std::size_t tensor_index = 0; tensor_index < 3; ++tensor_index) {
        const auto begin = host_residues.begin() + static_cast<std::ptrdiff_t>(tensor_index * per_tensor);
        const auto end = begin + static_cast<std::ptrdiff_t>(per_tensor);
        out[tensor_index] = std::vector<std::uint32_t>(begin, end);
    }
    return out;
}

std::vector<std::uint64_t> reconstruct_sampled_scalars(
    const std::vector<std::uint32_t>& sampled_residues,
    std::size_t sample_count
) {
    if (sample_count == 0 || sampled_residues.size() % sample_count != 0) {
        throw std::invalid_argument("sampled residue buffer shape does not match sample_count");
    }
    const std::size_t modulus_count = sampled_residues.size() / sample_count;

    std::vector<std::uint64_t> reconstructed(sample_count);
    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        reconstructed[sample_index] = reconstruct_scalar_from_residue_slice(
            sampled_residues.data(),
            modulus_count,
            sample_count,
            sample_index
        );
    }
    return reconstructed;
}

bool validate_sampled_pointwise_results(
    const std::vector<std::uint64_t>& lhs_values,
    const std::vector<std::uint64_t>& rhs_values,
    const std::vector<std::uint32_t>& sample_indices,
    const std::vector<std::uint64_t>& sum_samples,
    const std::vector<std::uint64_t>& diff_samples,
    const std::vector<std::uint64_t>& prod_samples
) {
    if (sample_indices.size() != sum_samples.size() ||
        sample_indices.size() != diff_samples.size() ||
        sample_indices.size() != prod_samples.size()) {
        throw std::invalid_argument("sampled pointwise buffers do not match sample index count");
    }

    for (std::size_t sample_offset = 0; sample_offset < sample_indices.size(); ++sample_offset) {
        const std::size_t scalar_index = static_cast<std::size_t>(sample_indices[sample_offset]);
        const std::uint64_t lhs = lhs_values[scalar_index];
        const std::uint64_t rhs = rhs_values[scalar_index];
        const std::uint64_t expected_sum = lhs + rhs;
        const std::uint64_t expected_diff = lhs - rhs;
        const std::uint64_t expected_prod = lhs * rhs;
        if (sum_samples[sample_offset] != expected_sum ||
            diff_samples[sample_offset] != expected_diff ||
            prod_samples[sample_offset] != expected_prod) {
            return false;
        }
    }
    return true;
}

bool validate_sampled_convolution_results(
    const std::vector<std::uint64_t>& lhs_values,
    const std::vector<std::uint64_t>& rhs_values,
    const std::vector<std::uint32_t>& sample_indices,
    const std::vector<std::uint64_t>& conv_samples,
    std::size_t value_count,
    std::size_t slot_count
) {
    if (sample_indices.size() != conv_samples.size()) {
        throw std::invalid_argument("sampled convolution buffer does not match sample index count");
    }

    const std::size_t conv_slot_count = 2 * slot_count - 1;
    for (std::size_t sample_offset = 0; sample_offset < sample_indices.size(); ++sample_offset) {
        const std::size_t flat_index = static_cast<std::size_t>(sample_indices[sample_offset]);
        const std::size_t value_index = flat_index / conv_slot_count;
        const std::size_t out_slot = flat_index % conv_slot_count;
        if (value_index >= value_count) {
            throw std::out_of_range("sampled convolution index exceeds value count");
        }

        std::uint64_t expected = 0;
        const std::size_t lhs_begin = out_slot >= slot_count - 1 ? out_slot - (slot_count - 1) : 0;
        const std::size_t lhs_end = std::min<std::size_t>(slot_count - 1, out_slot);
        for (std::size_t lhs_slot = lhs_begin; lhs_slot <= lhs_end; ++lhs_slot) {
            const std::size_t rhs_slot = out_slot - lhs_slot;
            expected += lhs_values[value_index * slot_count + lhs_slot] * rhs_values[value_index * slot_count + rhs_slot];
        }
        if (conv_samples[sample_offset] != expected) {
            return false;
        }
    }
    return true;
}

double measure_full_download_and_reconstruct_ms(
    const DeviceRnsTensor& sum,
    const DeviceRnsTensor& diff,
    const DeviceRnsTensor& prod,
    const DeviceRnsTensor& conv,
    const std::vector<std::uint64_t>& lhs_values,
    const std::vector<std::uint64_t>& rhs_values,
    std::size_t value_count,
    std::size_t slot_count,
    bool& ok
) {
    const auto start = std::chrono::steady_clock::now();
    const auto sum_host = reconstruct_scalars(download_tensor(sum));
    const auto diff_host = reconstruct_scalars(download_tensor(diff));
    const auto prod_host = reconstruct_scalars(download_tensor(prod));
    const auto conv_host = reconstruct_scalars(download_tensor(conv));
    const auto end = std::chrono::steady_clock::now();

    ok = validate_results(lhs_values, rhs_values, sum_host, diff_host, prod_host, conv_host, value_count, slot_count);
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double measure_sampled_download_and_reconstruct_ms(
    const DeviceRnsTensor& sum,
    const DeviceRnsTensor& diff,
    const DeviceRnsTensor& prod,
    const DeviceRnsTensor& conv,
    const std::vector<std::uint64_t>& lhs_values,
    const std::vector<std::uint64_t>& rhs_values,
    const SampleValidationPlan& plan,
    std::size_t value_count,
    std::size_t slot_count,
    bool& ok
) {
    const auto start = std::chrono::steady_clock::now();

    std::vector<std::uint64_t> sum_samples;
    std::vector<std::uint64_t> diff_samples;
    std::vector<std::uint64_t> prod_samples;
    if (!plan.pointwise_indices.empty()) {
        const auto pointwise_residues = gather_sampled_residues_triplet(
            sum,
            diff,
            prod,
            plan.pointwise_indices
        );
        sum_samples = reconstruct_sampled_scalars(pointwise_residues[0], plan.pointwise_indices.size());
        diff_samples = reconstruct_sampled_scalars(pointwise_residues[1], plan.pointwise_indices.size());
        prod_samples = reconstruct_sampled_scalars(pointwise_residues[2], plan.pointwise_indices.size());
    }

    std::vector<std::uint64_t> conv_samples;
    if (!plan.convolution_indices.empty()) {
        conv_samples = reconstruct_sampled_scalars(
            gather_sampled_residues(conv, plan.convolution_indices),
            plan.convolution_indices.size()
        );
    }

    const auto end = std::chrono::steady_clock::now();

    const bool pointwise_ok = validate_sampled_pointwise_results(
        lhs_values,
        rhs_values,
        plan.pointwise_indices,
        sum_samples,
        diff_samples,
        prod_samples
    );
    const bool convolution_ok = validate_sampled_convolution_results(
        lhs_values,
        rhs_values,
        plan.convolution_indices,
        conv_samples,
        value_count,
        slot_count
    );
    ok = pointwise_ok && convolution_ok;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool validate_metadata_all_equal(
    const std::vector<std::uint32_t>& values,
    std::uint32_t expected
) {
    return std::all_of(values.begin(), values.end(), [expected](std::uint32_t value) { return value == expected; });
}

bool validate_signs_equal(
    const std::vector<std::int8_t>& values,
    const std::vector<std::int8_t>& expected
) {
    return values == expected;
}

std::uint64_t reciprocal_seed_coefficient_u64(
    std::uint64_t denominator_coefficient,
    std::uint32_t target_product_scale_bits
) {
    if (denominator_coefficient == 0) {
        throw std::invalid_argument("reciprocal seed requires denominator_coefficient > 0");
    }
    if (target_product_scale_bits >= 64) {
        throw std::invalid_argument("reciprocal seed prototype currently requires target_product_scale_bits < 64");
    }
    const unsigned __int128 scaled_one = static_cast<unsigned __int128>(1) << target_product_scale_bits;
    const std::uint64_t seed = static_cast<std::uint64_t>(scaled_one / denominator_coefficient);
    if (seed == 0) {
        throw std::invalid_argument("reciprocal seed underflowed to zero; increase target_product_scale_bits");
    }
    return seed;
}

std::uint64_t exact_scaled_coefficient_u64(
    std::uint64_t coefficient,
    std::uint32_t source_scale_bits,
    std::uint32_t target_scale_bits
) {
    if (target_scale_bits < source_scale_bits) {
        throw std::invalid_argument("target_scale_bits must be >= source_scale_bits");
    }
    const std::uint32_t shift = target_scale_bits - source_scale_bits;
    const unsigned __int128 scaled = static_cast<unsigned __int128>(coefficient) << shift;
    if (scaled > static_cast<unsigned __int128>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::invalid_argument("scaled coefficient does not fit into uint64 for the current prototype");
    }
    return static_cast<std::uint64_t>(scaled);
}

std::uint64_t sqrt_seed_coefficient_u64(
    std::uint64_t radicand_coefficient,
    std::uint32_t radicand_scale_bits,
    std::uint32_t target_sqrt_scale_bits
) {
    if (radicand_coefficient == 0) {
        return 0;
    }
    if ((radicand_scale_bits & 1u) != 0u) {
        throw std::invalid_argument("sqrt seed prototype currently requires even radicand_scale_bits");
    }
    const std::uint32_t radicand_half_scale_bits = radicand_scale_bits / 2u;
    if (target_sqrt_scale_bits < radicand_half_scale_bits) {
        throw std::invalid_argument("target_sqrt_scale_bits must be >= radicand_scale_bits / 2");
    }

    const std::uint32_t compare_scale_bits = 2u * target_sqrt_scale_bits;
    const std::uint64_t target_scaled_coefficient = exact_scaled_coefficient_u64(
        radicand_coefficient,
        radicand_scale_bits,
        compare_scale_bits
    );

    long double scaled = std::sqrt(static_cast<long double>(radicand_coefficient));
    scaled = std::ldexp(scaled, static_cast<int>(target_sqrt_scale_bits - radicand_half_scale_bits));
    std::uint64_t seed = static_cast<std::uint64_t>(std::floor(scaled));

    const auto square_u128 = [](std::uint64_t value) {
        return static_cast<unsigned __int128>(value) * static_cast<unsigned __int128>(value);
    };
    const unsigned __int128 target = static_cast<unsigned __int128>(target_scaled_coefficient);

    while (seed > 0 && square_u128(seed) > target) {
        --seed;
    }
    while (seed < std::numeric_limits<std::uint64_t>::max() && square_u128(seed + 1u) <= target) {
        ++seed;
    }
    return seed;
}

std::uint64_t division_quotient_coefficient_u64(
    std::uint64_t numerator_coefficient,
    std::uint64_t denominator_coefficient,
    std::uint32_t numerator_scale_bits,
    std::uint32_t target_product_scale_bits
) {
    if (denominator_coefficient == 0) {
        throw std::invalid_argument("division prototype requires denominator_coefficient > 0");
    }
    const std::uint64_t scaled_numerator = exact_scaled_coefficient_u64(
        numerator_coefficient,
        numerator_scale_bits,
        target_product_scale_bits
    );
    return scaled_numerator / denominator_coefficient;
}

}  // namespace

DeviceRnsTensor allocate_device_tensor(std::size_t value_count, std::size_t slot_count, int modulus_count) {
    if (value_count == 0 || slot_count == 0) {
        throw std::invalid_argument("value_count and slot_count must be positive");
    }
    if (modulus_count < 1 || modulus_count > ModulusConfig::kModulusCount) {
        throw std::invalid_argument("modulus_count must be in [1, ModulusConfig::kModulusCount]");
    }

    DeviceRnsTensor tensor;
    tensor.value_count = value_count;
    tensor.slot_count = slot_count;
    tensor.modulus_count = modulus_count;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&tensor.d_moduli), sizeof(std::uint32_t) * tensor.modulus_count), "cudaMalloc moduli");
    check_cuda(
        cudaMemcpy(
            tensor.d_moduli,
            ModulusConfig::kModuli.data(),
            sizeof(std::uint32_t) * tensor.modulus_count,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy moduli"
    );
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&tensor.d_residues), sizeof(std::uint32_t) * residue_count(tensor)),
        "cudaMalloc residues"
    );
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&tensor.d_signs), sizeof(std::int8_t) * tensor.value_count), "cudaMalloc signs");
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&tensor.d_logical_slots), sizeof(std::uint32_t) * tensor.value_count),
        "cudaMalloc logical_slots"
    );
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&tensor.d_scale_bits), sizeof(std::uint32_t) * tensor.value_count),
        "cudaMalloc scale_bits"
    );
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&tensor.d_levels), sizeof(std::uint32_t) * tensor.value_count), "cudaMalloc levels");
    return tensor;
}

DeviceRnsTensor allocate_polynomial_block_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    int modulus_count
) {
    return allocate_device_tensor(block_count, coefficient_count, modulus_count);
}

DeviceRnsTensor make_scaled_constant_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    std::uint64_t coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t scale_bits,
    int modulus_count
) {
    DeviceRnsTensor tensor = allocate_polynomial_block_tensor(block_count, coefficient_count, modulus_count);
    try {
        encode_scaled_constant_blocks(
            tensor,
            coefficient,
            coefficient_slot,
            logical_slots,
            scale_bits
        );
        return tensor;
    } catch (...) {
        free_device_tensor(tensor);
        throw;
    }
}

DeviceRnsTensor make_reciprocal_seed_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    std::uint64_t denominator_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t denominator_scale_bits,
    std::uint32_t target_product_scale_bits,
    int modulus_count
) {
    DeviceRnsTensor tensor = allocate_polynomial_block_tensor(block_count, coefficient_count, modulus_count);
    try {
        encode_reciprocal_seed_blocks(
            tensor,
            denominator_coefficient,
            coefficient_slot,
            logical_slots,
            denominator_scale_bits,
            target_product_scale_bits
        );
        return tensor;
    } catch (...) {
        free_device_tensor(tensor);
        throw;
    }
}

DeviceRnsTensor make_sqrt_seed_tensor(
    std::size_t block_count,
    std::size_t coefficient_count,
    std::uint64_t radicand_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t radicand_scale_bits,
    std::uint32_t target_sqrt_scale_bits,
    int modulus_count
) {
    DeviceRnsTensor tensor = allocate_polynomial_block_tensor(block_count, coefficient_count, modulus_count);
    try {
        encode_sqrt_seed_blocks(
            tensor,
            radicand_coefficient,
            coefficient_slot,
            logical_slots,
            radicand_scale_bits,
            target_sqrt_scale_bits
        );
        return tensor;
    } catch (...) {
        free_device_tensor(tensor);
        throw;
    }
}

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
    int modulus_count
) {
    DeviceRnsTensor tensor = allocate_polynomial_block_tensor(block_count, coefficient_count, modulus_count);
    try {
        encode_division_quotient_blocks(
            tensor,
            numerator_coefficient,
            denominator_coefficient,
            coefficient_slot,
            logical_slots,
            numerator_scale_bits,
            denominator_scale_bits,
            target_product_scale_bits
        );
        return tensor;
    } catch (...) {
        free_device_tensor(tensor);
        throw;
    }
}

void free_device_tensor(DeviceRnsTensor& tensor) {
    cudaFree(tensor.d_moduli);
    cudaFree(tensor.d_residues);
    cudaFree(tensor.d_signs);
    cudaFree(tensor.d_logical_slots);
    cudaFree(tensor.d_scale_bits);
    cudaFree(tensor.d_levels);
    tensor = {};
}

DeviceRnsTensor pad_polynomial_tensor_suffix_zeros(const DeviceRnsTensor& src, std::size_t dst_slot_count) {
    if (dst_slot_count < src.slot_count) {
        throw std::invalid_argument("pad_polynomial_tensor_suffix_zeros requires dst_slot_count >= src.slot_count");
    }

    DeviceRnsTensor dst = allocate_polynomial_block_tensor(src.value_count, dst_slot_count, src.modulus_count);
    try {
        check_cuda(
            cudaMemcpy(
                dst.d_moduli,
                src.d_moduli,
                sizeof(std::uint32_t) * static_cast<std::size_t>(src.modulus_count),
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy padded tensor moduli"
        );
        copy_residues_with_zero_suffix_kernel<<<block_count(residue_count(dst)), kThreadsPerBlock>>>(
            src.d_residues,
            dst.d_residues,
            src.value_count,
            src.slot_count,
            dst.slot_count,
            src.modulus_count
        );
        check_cuda(cudaGetLastError(), "copy_residues_with_zero_suffix_kernel launch");
        check_cuda(
            cudaMemcpy(
                dst.d_signs,
                src.d_signs,
                sizeof(std::int8_t) * src.value_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy padded tensor signs"
        );
        check_cuda(
            cudaMemcpy(
                dst.d_logical_slots,
                src.d_logical_slots,
                sizeof(std::uint32_t) * src.value_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy padded tensor logical_slots"
        );
        check_cuda(
            cudaMemcpy(
                dst.d_scale_bits,
                src.d_scale_bits,
                sizeof(std::uint32_t) * src.value_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy padded tensor scale_bits"
        );
        check_cuda(
            cudaMemcpy(
                dst.d_levels,
                src.d_levels,
                sizeof(std::uint32_t) * src.value_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy padded tensor levels"
        );
        return dst;
    } catch (...) {
        free_device_tensor(dst);
        throw;
    }
}

void encode_u64(DeviceRnsTensor& tensor, const std::vector<std::uint64_t>& values) {
    if (values.size() != scalar_count(tensor)) {
        throw std::invalid_argument("encode_u64 value count does not match tensor shape");
    }

    std::uint64_t* d_values = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_values), sizeof(std::uint64_t) * values.size()), "cudaMalloc d_values");
    check_cuda(
        cudaMemcpy(d_values, values.data(), sizeof(std::uint64_t) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy values"
    );

    encode_u64_kernel<<<block_count(residue_count(tensor)), kThreadsPerBlock>>>(
        d_values,
        tensor.d_moduli,
        tensor.d_residues,
        scalar_count(tensor),
        tensor.modulus_count
    );
    check_cuda(cudaGetLastError(), "encode_u64_kernel launch");
    init_metadata_kernel<<<block_count(tensor.value_count), kThreadsPerBlock>>>(
        d_values,
        tensor.d_signs,
        tensor.d_logical_slots,
        tensor.d_scale_bits,
        tensor.d_levels,
        tensor.value_count,
        tensor.slot_count,
        static_cast<std::uint32_t>(tensor.modulus_count)
    );
    check_cuda(cudaGetLastError(), "init_metadata_kernel launch");
    check_cuda(cudaFree(d_values), "cudaFree d_values");
}

void encode_polynomial_blocks(DeviceRnsTensor& tensor, const std::vector<std::uint64_t>& coefficients) {
    encode_u64(tensor, coefficients);
}

void encode_signed_polynomial_blocks(
    DeviceRnsTensor& tensor,
    const std::vector<long long>& coefficients,
    std::uint32_t logical_slots
) {
    if (coefficients.size() != scalar_count(tensor)) {
        throw std::invalid_argument("encode_signed_polynomial_blocks value count does not match tensor shape");
    }
    if (logical_slots == 0 || logical_slots > tensor.slot_count) {
        throw std::invalid_argument("encode_signed_polynomial_blocks logical_slots must be in [1, tensor.slot_count]");
    }

    std::vector<std::uint32_t> residues(residue_count(tensor), 0u);
    bool any_nonzero = false;
    for (int modulus_index = 0; modulus_index < tensor.modulus_count; ++modulus_index) {
        const long long modulus = static_cast<long long>(ModulusConfig::kModuli[modulus_index]);
        for (std::size_t value_index = 0; value_index < tensor.value_count; ++value_index) {
            for (std::size_t slot_index = 0; slot_index < tensor.slot_count; ++slot_index) {
                const std::size_t scalar_index = value_index * tensor.slot_count + slot_index;
                const long long coefficient = coefficients[scalar_index];
                any_nonzero = any_nonzero || coefficient != 0;
                long long residue = coefficient % modulus;
                if (residue < 0) {
                    residue += modulus;
                }
                const std::size_t residue_index =
                    (static_cast<std::size_t>(modulus_index) * tensor.value_count + value_index) * tensor.slot_count +
                    slot_index;
                residues[residue_index] = static_cast<std::uint32_t>(residue);
            }
        }
    }

    check_cuda(
        cudaMemcpy(
            tensor.d_residues,
            residues.data(),
            sizeof(std::uint32_t) * residues.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy signed polynomial residues"
    );
    set_uniform_tensor_metadata(
        tensor,
        any_nonzero ? 1 : 0,
        logical_slots,
        0,
        static_cast<std::uint32_t>(tensor.modulus_count)
    );
}

void encode_scaled_constant_blocks(
    DeviceRnsTensor& tensor,
    std::uint64_t coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t scale_bits
) {
    if (logical_slots == 0 || logical_slots > tensor.slot_count) {
        throw std::invalid_argument("logical_slots must be in [1, tensor.slot_count]");
    }
    if (coefficient_slot >= logical_slots) {
        throw std::invalid_argument("coefficient_slot must be smaller than logical_slots");
    }

    check_cuda(
        cudaMemset(tensor.d_residues, 0, sizeof(std::uint32_t) * residue_count(tensor)),
        "cudaMemset scaled_constant residues"
    );
    scatter_scaled_constant_kernel<<<
        block_count(tensor.value_count * static_cast<std::size_t>(tensor.modulus_count)),
        kThreadsPerBlock
    >>>(
        tensor.d_moduli,
        tensor.d_residues,
        tensor.value_count,
        tensor.slot_count,
        coefficient_slot,
        coefficient,
        tensor.modulus_count
    );
    check_cuda(cudaGetLastError(), "scatter_scaled_constant_kernel launch");
    set_uniform_metadata_kernel<<<block_count(tensor.value_count), kThreadsPerBlock>>>(
        tensor.d_signs,
        tensor.d_logical_slots,
        tensor.d_scale_bits,
        tensor.d_levels,
        tensor.value_count,
        coefficient == 0 ? 0 : 1,
        logical_slots,
        scale_bits,
        static_cast<std::uint32_t>(tensor.modulus_count)
    );
    check_cuda(cudaGetLastError(), "set_uniform_metadata_kernel scaled constant launch");
}

void encode_reciprocal_seed_blocks(
    DeviceRnsTensor& tensor,
    std::uint64_t denominator_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t denominator_scale_bits,
    std::uint32_t target_product_scale_bits
) {
    if (target_product_scale_bits < denominator_scale_bits) {
        throw std::invalid_argument("target_product_scale_bits must be >= denominator_scale_bits");
    }
    const std::uint64_t seed_coefficient = reciprocal_seed_coefficient_u64(
        denominator_coefficient,
        target_product_scale_bits
    );
    encode_scaled_constant_blocks(
        tensor,
        seed_coefficient,
        coefficient_slot,
        logical_slots,
        target_product_scale_bits - denominator_scale_bits
    );
}

void encode_sqrt_seed_blocks(
    DeviceRnsTensor& tensor,
    std::uint64_t radicand_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t radicand_scale_bits,
    std::uint32_t target_sqrt_scale_bits
) {
    const std::uint64_t seed_coefficient = sqrt_seed_coefficient_u64(
        radicand_coefficient,
        radicand_scale_bits,
        target_sqrt_scale_bits
    );
    encode_scaled_constant_blocks(
        tensor,
        seed_coefficient,
        coefficient_slot,
        logical_slots,
        target_sqrt_scale_bits
    );
}

void encode_division_quotient_blocks(
    DeviceRnsTensor& tensor,
    std::uint64_t numerator_coefficient,
    std::uint64_t denominator_coefficient,
    std::size_t coefficient_slot,
    std::uint32_t logical_slots,
    std::uint32_t numerator_scale_bits,
    std::uint32_t denominator_scale_bits,
    std::uint32_t target_product_scale_bits
) {
    if (target_product_scale_bits < denominator_scale_bits) {
        throw std::invalid_argument("target_product_scale_bits must be >= denominator_scale_bits");
    }
    const std::uint64_t quotient_coefficient = division_quotient_coefficient_u64(
        numerator_coefficient,
        denominator_coefficient,
        numerator_scale_bits,
        target_product_scale_bits
    );
    encode_scaled_constant_blocks(
        tensor,
        quotient_coefficient,
        coefficient_slot,
        logical_slots,
        target_product_scale_bits - denominator_scale_bits
    );
}

void set_uniform_tensor_metadata(
    DeviceRnsTensor& tensor,
    std::int8_t sign,
    std::uint32_t logical_slots,
    std::uint32_t scale_bits,
    std::uint32_t level
) {
    if (logical_slots > tensor.slot_count) {
        throw std::invalid_argument("logical_slots cannot exceed tensor.slot_count");
    }
    if (level == 0 || level > static_cast<std::uint32_t>(tensor.modulus_count)) {
        throw std::invalid_argument("level must be in [1, tensor.modulus_count]");
    }
    set_uniform_metadata_kernel<<<block_count(tensor.value_count), kThreadsPerBlock>>>(
        tensor.d_signs,
        tensor.d_logical_slots,
        tensor.d_scale_bits,
        tensor.d_levels,
        tensor.value_count,
        sign,
        logical_slots,
        scale_bits,
        level
    );
    check_cuda(cudaGetLastError(), "set_uniform_metadata_kernel launch");
}

void drop_moduli_prefix(const DeviceRnsTensor& src, DeviceRnsTensor& dst, std::uint32_t scale_bits_delta) {
    if (src.value_count != dst.value_count || src.slot_count != dst.slot_count) {
        throw std::invalid_argument("drop_moduli_prefix requires matching value_count and slot_count");
    }
    if (dst.modulus_count < 1 || dst.modulus_count > src.modulus_count) {
        throw std::invalid_argument("drop_moduli_prefix requires 1 <= dst.modulus_count <= src.modulus_count");
    }

    const std::size_t dst_modulus_count = static_cast<std::size_t>(dst.modulus_count);
    const std::size_t residues_to_copy = dst_modulus_count * scalar_count(src);
    check_cuda(
        cudaMemcpy(dst.d_moduli, src.d_moduli, sizeof(std::uint32_t) * dst_modulus_count, cudaMemcpyDeviceToDevice),
        "cudaMemcpy drop_moduli_prefix moduli"
    );
    check_cuda(
        cudaMemcpy(dst.d_residues, src.d_residues, sizeof(std::uint32_t) * residues_to_copy, cudaMemcpyDeviceToDevice),
        "cudaMemcpy drop_moduli_prefix residues"
    );
    copy_metadata_with_level_drop_kernel<<<block_count(src.value_count), kThreadsPerBlock>>>(
        src.d_signs,
        src.d_logical_slots,
        src.d_scale_bits,
        src.d_levels,
        dst.d_signs,
        dst.d_logical_slots,
        dst.d_scale_bits,
        dst.d_levels,
        src.value_count,
        static_cast<std::uint32_t>(dst.slot_count),
        static_cast<std::uint32_t>(dst.modulus_count),
        scale_bits_delta
    );
    check_cuda(cudaGetLastError(), "copy_metadata_with_level_drop_kernel launch");
}

void pointwise_add(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    check_same_shape(lhs, rhs, out);
    pointwise_add_kernel<<<block_count(residue_count(lhs)), kThreadsPerBlock>>>(
        lhs.d_residues,
        rhs.d_residues,
        lhs.d_moduli,
        out.d_residues,
        scalar_count(lhs),
        lhs.modulus_count
    );
    check_cuda(cudaGetLastError(), "pointwise_add_kernel launch");
    launch_binary_metadata_merge(lhs, rhs, out);
}

void pointwise_add_raw(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    check_same_shape(lhs, rhs, out);
    pointwise_add_kernel<<<block_count(residue_count(lhs)), kThreadsPerBlock>>>(
        lhs.d_residues,
        rhs.d_residues,
        lhs.d_moduli,
        out.d_residues,
        scalar_count(lhs),
        lhs.modulus_count
    );
    check_cuda(cudaGetLastError(), "pointwise_add_kernel raw launch");
}

void pointwise_sub(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    check_same_shape(lhs, rhs, out);
    pointwise_sub_kernel<<<block_count(residue_count(lhs)), kThreadsPerBlock>>>(
        lhs.d_residues,
        rhs.d_residues,
        lhs.d_moduli,
        out.d_residues,
        scalar_count(lhs),
        lhs.modulus_count
    );
    check_cuda(cudaGetLastError(), "pointwise_sub_kernel launch");
    launch_binary_metadata_merge(lhs, rhs, out);
}

void pointwise_mul(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    check_same_shape(lhs, rhs, out);
    pointwise_mul_kernel<<<block_count(residue_count(lhs)), kThreadsPerBlock>>>(
        lhs.d_residues,
        rhs.d_residues,
        lhs.d_moduli,
        out.d_residues,
        scalar_count(lhs),
        lhs.modulus_count
    );
    check_cuda(cudaGetLastError(), "pointwise_mul_kernel launch");
    launch_pointwise_multiply_metadata_merge(lhs, rhs, out);
}

void pairwise_convolution(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    check_convolution_shape(lhs, rhs, out);
    const std::size_t ntt_size = resolve_convolution_ntt_size(out.slot_count);
    ensure_ntt_support_for_modulus_count(lhs.modulus_count, ntt_size);
    if (use_shared_ntt(ntt_size)) {
        const int log_ntt_size = ceil_log2_size(ntt_size);
        const int threads = static_cast<int>(std::min<std::size_t>(kThreadsPerBlock, std::max<std::size_t>(1, ntt_size)));
        const std::size_t transform_count = static_cast<std::size_t>(lhs.modulus_count) * lhs.value_count;
        const std::size_t shared_bytes = sizeof(std::uint32_t) * 2 * ntt_size;

        ntt_convolution_kernel<<<static_cast<int>(transform_count), threads, shared_bytes>>>(
            lhs.d_residues,
            rhs.d_residues,
            lhs.d_moduli,
            out.d_residues,
            lhs.value_count,
            lhs.slot_count,
            rhs.slot_count,
            out.slot_count,
            ntt_size,
            log_ntt_size
        );
        check_cuda(cudaGetLastError(), "ntt_convolution_kernel launch");
    } else {
        staged_global_ntt_convolution(lhs, rhs, out);
    }
    launch_convolution_metadata_merge(lhs, rhs, out);
}

void pairwise_convolution_raw(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    check_convolution_shape(lhs, rhs, out);
    const std::size_t ntt_size = resolve_convolution_ntt_size(out.slot_count);
    ensure_ntt_support_for_modulus_count(lhs.modulus_count, ntt_size);
    if (use_shared_ntt(ntt_size)) {
        const int log_ntt_size = ceil_log2_size(ntt_size);
        const int threads = static_cast<int>(std::min<std::size_t>(kThreadsPerBlock, std::max<std::size_t>(1, ntt_size)));
        const std::size_t transform_count = static_cast<std::size_t>(lhs.modulus_count) * lhs.value_count;
        const std::size_t shared_bytes = sizeof(std::uint32_t) * 2 * ntt_size;

        ntt_convolution_kernel<<<static_cast<int>(transform_count), threads, shared_bytes>>>(
            lhs.d_residues,
            rhs.d_residues,
            lhs.d_moduli,
            out.d_residues,
            lhs.value_count,
            lhs.slot_count,
            rhs.slot_count,
            out.slot_count,
            ntt_size,
            log_ntt_size
        );
        check_cuda(cudaGetLastError(), "ntt_convolution_kernel raw launch");
    } else {
        staged_global_ntt_convolution(lhs, rhs, out);
    }
}

void add_polynomial_blocks(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    pointwise_add(lhs, rhs, out);
}

void add_polynomial_blocks_raw(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    pointwise_add_raw(lhs, rhs, out);
}

void sub_polynomial_blocks(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    pointwise_sub(lhs, rhs, out);
}

void multiply_polynomial_blocks(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    pointwise_mul(lhs, rhs, out);
}

void convolve_polynomial_blocks(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    pairwise_convolution(lhs, rhs, out);
}

void convolve_polynomial_blocks_raw(const DeviceRnsTensor& lhs, const DeviceRnsTensor& rhs, DeviceRnsTensor& out) {
    pairwise_convolution_raw(lhs, rhs, out);
}

void compute_residual_correction(
    const DeviceRnsTensor& target,
    const DeviceRnsTensor& approximate,
    DeviceRnsTensor& residual,
    DeviceRnsTensor& corrected
) {
    check_same_shape(target, approximate, residual);
    check_same_shape(target, approximate, corrected);
    sub_polynomial_blocks(target, approximate, residual);
    add_polynomial_blocks(approximate, residual, corrected);
}

HostRnsTensor download_tensor(const DeviceRnsTensor& tensor) {
    HostRnsTensor host;
    host.value_count = tensor.value_count;
    host.slot_count = tensor.slot_count;
    host.modulus_count = tensor.modulus_count;
    host.moduli.resize(tensor.modulus_count);
    host.residues.resize(residue_count(tensor));
    host.signs.resize(tensor.value_count);
    host.logical_slots.resize(tensor.value_count);
    host.scale_bits.resize(tensor.value_count);
    host.levels.resize(tensor.value_count);

    check_cuda(
        cudaMemcpy(host.moduli.data(), tensor.d_moduli, sizeof(std::uint32_t) * tensor.modulus_count, cudaMemcpyDeviceToHost),
        "cudaMemcpy moduli download"
    );
    check_cuda(
        cudaMemcpy(host.residues.data(), tensor.d_residues, sizeof(std::uint32_t) * residue_count(tensor), cudaMemcpyDeviceToHost),
        "cudaMemcpy residues download"
    );
    check_cuda(
        cudaMemcpy(host.signs.data(), tensor.d_signs, sizeof(std::int8_t) * tensor.value_count, cudaMemcpyDeviceToHost),
        "cudaMemcpy signs download"
    );
    check_cuda(
        cudaMemcpy(
            host.logical_slots.data(),
            tensor.d_logical_slots,
            sizeof(std::uint32_t) * tensor.value_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy logical_slots download"
    );
    check_cuda(
        cudaMemcpy(
            host.scale_bits.data(),
            tensor.d_scale_bits,
            sizeof(std::uint32_t) * tensor.value_count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy scale_bits download"
    );
    check_cuda(
        cudaMemcpy(host.levels.data(), tensor.d_levels, sizeof(std::uint32_t) * tensor.value_count, cudaMemcpyDeviceToHost),
        "cudaMemcpy levels download"
    );
    return host;
}

std::vector<std::uint64_t> reconstruct_scalars(const HostRnsTensor& tensor) {
    std::vector<std::uint64_t> out(tensor.value_count * tensor.slot_count);
    for (std::size_t value_index = 0; value_index < tensor.value_count; ++value_index) {
        for (std::size_t slot_index = 0; slot_index < tensor.slot_count; ++slot_index) {
            out[value_index * tensor.slot_count + slot_index] = reconstruct_scalar(tensor, value_index, slot_index);
        }
    }
    return out;
}

std::string describe_layout(const DeviceRnsTensor& tensor) {
    std::ostringstream builder;
    builder
        << "RNS(native-gpu) layout: residues[modulus][value][slot], modulus_count="
        << tensor.modulus_count
        << ", value_count="
        << tensor.value_count
        << ", slot_count="
        << tensor.slot_count
        << ", metadata={sign, logical_slots, scale_bits, level}";
    return builder.str();
}

bool lifecycle_block_smoke_test(std::ostream& out) {
    constexpr std::size_t kBlockCount = 2;
    constexpr std::size_t kCoefficientCount = 8;
    constexpr int kTargetModulusCount = 2;
    constexpr std::uint32_t kLhsScaleBits = 7;
    constexpr std::uint32_t kRhsScaleBits = 11;

    const std::vector<std::uint64_t> lhs_values = {
        3, 1, 4, 1, 5, 9, 2, 6,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    const std::vector<std::uint64_t> rhs_values = {
        5, 8, 9, 7, 9, 3, 2, 3,
        6, 5, 3, 5, 8, 9, 7, 9,
    };

    DeviceRnsTensor lhs_full;
    DeviceRnsTensor rhs_full;
    DeviceRnsTensor lhs_level;
    DeviceRnsTensor rhs_level;
    DeviceRnsTensor sum;
    DeviceRnsTensor prod;
    DeviceRnsTensor conv;

    const auto cleanup = [&]() {
        if (conv.d_moduli != nullptr) {
            free_device_tensor(conv);
        }
        if (prod.d_moduli != nullptr) {
            free_device_tensor(prod);
        }
        if (sum.d_moduli != nullptr) {
            free_device_tensor(sum);
        }
        if (rhs_level.d_moduli != nullptr) {
            free_device_tensor(rhs_level);
        }
        if (lhs_level.d_moduli != nullptr) {
            free_device_tensor(lhs_level);
        }
        if (rhs_full.d_moduli != nullptr) {
            free_device_tensor(rhs_full);
        }
        if (lhs_full.d_moduli != nullptr) {
            free_device_tensor(lhs_full);
        }
    };

    try {
        lhs_full = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount);
        rhs_full = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount);
        encode_polynomial_blocks(lhs_full, lhs_values);
        encode_polynomial_blocks(rhs_full, rhs_values);

        lhs_level = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        rhs_level = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        drop_moduli_prefix(lhs_full, lhs_level, kLhsScaleBits);
        drop_moduli_prefix(rhs_full, rhs_level, kRhsScaleBits);

        sum = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        prod = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        conv = allocate_polynomial_block_tensor(kBlockCount, 2 * kCoefficientCount - 1, kTargetModulusCount);

        add_polynomial_blocks(lhs_level, rhs_level, sum);
        multiply_polynomial_blocks(lhs_level, rhs_level, prod);
        convolve_polynomial_blocks(lhs_level, rhs_level, conv);

        const auto lhs_host = download_tensor(lhs_level);
        const auto rhs_host = download_tensor(rhs_level);
        const auto sum_host = download_tensor(sum);
        const auto prod_host = download_tensor(prod);
        const auto conv_host = download_tensor(conv);

        const auto sum_scalars = reconstruct_scalars(sum_host);
        const auto prod_scalars = reconstruct_scalars(prod_host);
        const auto conv_scalars = reconstruct_scalars(conv_host);

        bool ok = true;
        ok = ok && lhs_host.modulus_count == kTargetModulusCount;
        ok = ok && rhs_host.modulus_count == kTargetModulusCount;
        ok = ok && validate_metadata_all_equal(lhs_host.logical_slots, static_cast<std::uint32_t>(kCoefficientCount));
        ok = ok && validate_metadata_all_equal(rhs_host.logical_slots, static_cast<std::uint32_t>(kCoefficientCount));
        ok = ok && validate_metadata_all_equal(lhs_host.scale_bits, kLhsScaleBits);
        ok = ok && validate_metadata_all_equal(rhs_host.scale_bits, kRhsScaleBits);
        ok = ok && validate_metadata_all_equal(lhs_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_metadata_all_equal(rhs_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(lhs_host.signs, {1, 0});
        ok = ok && validate_signs_equal(rhs_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(sum_host.logical_slots, static_cast<std::uint32_t>(kCoefficientCount));
        ok = ok && validate_metadata_all_equal(sum_host.scale_bits, kRhsScaleBits);
        ok = ok && validate_metadata_all_equal(sum_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(sum_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(prod_host.logical_slots, static_cast<std::uint32_t>(kCoefficientCount));
        ok = ok && validate_metadata_all_equal(prod_host.scale_bits, kLhsScaleBits + kRhsScaleBits);
        ok = ok && validate_metadata_all_equal(prod_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(prod_host.signs, {1, 0});

        ok = ok && validate_metadata_all_equal(conv_host.logical_slots, static_cast<std::uint32_t>(2 * kCoefficientCount - 1));
        ok = ok && validate_metadata_all_equal(conv_host.scale_bits, kLhsScaleBits + kRhsScaleBits);
        ok = ok && validate_metadata_all_equal(conv_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(conv_host.signs, {1, 0});

        for (std::size_t index = 0; index < lhs_values.size(); ++index) {
            ok = ok && sum_scalars[index] == lhs_values[index] + rhs_values[index];
            ok = ok && prod_scalars[index] == lhs_values[index] * rhs_values[index];
        }
        for (std::size_t block = 0; block < kBlockCount; ++block) {
            for (std::size_t out_slot = 0; out_slot < 2 * kCoefficientCount - 1; ++out_slot) {
                std::uint64_t expected = 0;
                const std::size_t lhs_begin =
                    out_slot >= kCoefficientCount - 1 ? out_slot - (kCoefficientCount - 1) : 0;
                const std::size_t lhs_end = std::min<std::size_t>(kCoefficientCount - 1, out_slot);
                for (std::size_t lhs_slot = lhs_begin; lhs_slot <= lhs_end; ++lhs_slot) {
                    const std::size_t rhs_slot = out_slot - lhs_slot;
                    expected +=
                        lhs_values[block * kCoefficientCount + lhs_slot] *
                        rhs_values[block * kCoefficientCount + rhs_slot];
                }
                ok = ok && conv_scalars[block * (2 * kCoefficientCount - 1) + out_slot] == expected;
            }
        }

        out << "lifecycle_block_smoke_status=" << (ok ? "ok" : "failed") << '\n';
        out << "block_count=" << kBlockCount << '\n';
        out << "coefficient_count=" << kCoefficientCount << '\n';
        out << "active_modulus_count=" << kTargetModulusCount << '\n';
        out << "lhs_scale_bits=" << kLhsScaleBits << '\n';
        out << "rhs_scale_bits=" << kRhsScaleBits << '\n';
        out << "product_scale_bits=" << (kLhsScaleBits + kRhsScaleBits) << '\n';
        out << "convolution_logical_slots=" << (2 * kCoefficientCount - 1) << '\n';

        cleanup();
        return ok;
    } catch (...) {
        cleanup();
        throw;
    }
}

bool scaled_constant_smoke_test(std::ostream& out) {
    constexpr std::size_t kBlockCount = 2;
    constexpr std::size_t kCoefficientCount = 8;
    constexpr std::size_t kCoefficientSlot = 1;
    constexpr std::uint32_t kLogicalSlots = 4;
    constexpr std::uint64_t kFullCoefficient = 9;
    constexpr std::uint64_t kLevelCoefficient = 7;
    constexpr std::uint32_t kFullScaleBits = 6;
    constexpr std::uint32_t kDropScaleBitsDelta = 3;
    constexpr std::uint32_t kDroppedScaleBits = kFullScaleBits + kDropScaleBitsDelta;
    constexpr std::uint32_t kLevelScaleBits = 11;
    constexpr int kTargetModulusCount = 2;

    DeviceRnsTensor full_constant;
    DeviceRnsTensor dropped_constant;
    DeviceRnsTensor level_constant;
    DeviceRnsTensor sum;
    DeviceRnsTensor prod;
    DeviceRnsTensor conv;

    const auto cleanup = [&]() {
        if (conv.d_moduli != nullptr) {
            free_device_tensor(conv);
        }
        if (prod.d_moduli != nullptr) {
            free_device_tensor(prod);
        }
        if (sum.d_moduli != nullptr) {
            free_device_tensor(sum);
        }
        if (level_constant.d_moduli != nullptr) {
            free_device_tensor(level_constant);
        }
        if (dropped_constant.d_moduli != nullptr) {
            free_device_tensor(dropped_constant);
        }
        if (full_constant.d_moduli != nullptr) {
            free_device_tensor(full_constant);
        }
    };

    const auto expected_monomial = [](
        std::uint64_t coefficient,
        std::size_t slot_count,
        std::size_t coefficient_slot
    ) {
        std::vector<std::uint64_t> values(kBlockCount * slot_count, 0);
        for (std::size_t block = 0; block < kBlockCount; ++block) {
            values[block * slot_count + coefficient_slot] = coefficient;
        }
        return values;
    };

    try {
        full_constant = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            kFullCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kFullScaleBits
        );
        dropped_constant = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        drop_moduli_prefix(full_constant, dropped_constant, kDropScaleBitsDelta);
        level_constant = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            kLevelCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kLevelScaleBits,
            kTargetModulusCount
        );

        sum = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        prod = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        conv = allocate_polynomial_block_tensor(kBlockCount, 2 * kCoefficientCount - 1, kTargetModulusCount);

        add_polynomial_blocks(dropped_constant, level_constant, sum);
        multiply_polynomial_blocks(dropped_constant, level_constant, prod);
        convolve_polynomial_blocks(dropped_constant, level_constant, conv);

        const auto full_host = download_tensor(full_constant);
        const auto dropped_host = download_tensor(dropped_constant);
        const auto level_host = download_tensor(level_constant);
        const auto sum_host = download_tensor(sum);
        const auto prod_host = download_tensor(prod);
        const auto conv_host = download_tensor(conv);

        const auto full_scalars = reconstruct_scalars(full_host);
        const auto dropped_scalars = reconstruct_scalars(dropped_host);
        const auto level_scalars = reconstruct_scalars(level_host);
        const auto sum_scalars = reconstruct_scalars(sum_host);
        const auto prod_scalars = reconstruct_scalars(prod_host);
        const auto conv_scalars = reconstruct_scalars(conv_host);

        bool ok = true;
        ok = ok && full_host.modulus_count == ModulusConfig::kDefaultModulusCount;
        ok = ok && dropped_host.modulus_count == kTargetModulusCount;
        ok = ok && level_host.modulus_count == kTargetModulusCount;

        ok = ok && validate_metadata_all_equal(full_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(full_host.scale_bits, kFullScaleBits);
        ok = ok && validate_metadata_all_equal(
            full_host.levels,
            static_cast<std::uint32_t>(ModulusConfig::kDefaultModulusCount)
        );
        ok = ok && validate_signs_equal(full_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(dropped_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(dropped_host.scale_bits, kDroppedScaleBits);
        ok = ok && validate_metadata_all_equal(dropped_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(dropped_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(level_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(level_host.scale_bits, kLevelScaleBits);
        ok = ok && validate_metadata_all_equal(level_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(level_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(sum_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(sum_host.scale_bits, kLevelScaleBits);
        ok = ok && validate_metadata_all_equal(sum_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(sum_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(prod_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(prod_host.scale_bits, kDroppedScaleBits + kLevelScaleBits);
        ok = ok && validate_metadata_all_equal(prod_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(prod_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(conv_host.logical_slots, 2 * kLogicalSlots - 1);
        ok = ok && validate_metadata_all_equal(conv_host.scale_bits, kDroppedScaleBits + kLevelScaleBits);
        ok = ok && validate_metadata_all_equal(conv_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(conv_host.signs, {1, 1});

        ok = ok && full_scalars == expected_monomial(kFullCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && dropped_scalars == expected_monomial(kFullCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && level_scalars == expected_monomial(kLevelCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && sum_scalars == expected_monomial(kFullCoefficient + kLevelCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && prod_scalars == expected_monomial(kFullCoefficient * kLevelCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && conv_scalars == expected_monomial(
            kFullCoefficient * kLevelCoefficient,
            2 * kCoefficientCount - 1,
            2 * kCoefficientSlot
        );

        out << "scaled_constant_smoke_status=" << (ok ? "ok" : "failed") << '\n';
        out << "block_count=" << kBlockCount << '\n';
        out << "coefficient_count=" << kCoefficientCount << '\n';
        out << "constant_slot=" << kCoefficientSlot << '\n';
        out << "logical_slots=" << kLogicalSlots << '\n';
        out << "full_modulus_count=" << ModulusConfig::kDefaultModulusCount << '\n';
        out << "active_modulus_count=" << kTargetModulusCount << '\n';
        out << "full_scale_bits=" << kFullScaleBits << '\n';
        out << "dropped_scale_bits=" << kDroppedScaleBits << '\n';
        out << "level_scale_bits=" << kLevelScaleBits << '\n';
        out << "sum_scale_bits=" << kLevelScaleBits << '\n';
        out << "product_scale_bits=" << (kDroppedScaleBits + kLevelScaleBits) << '\n';
        out << "sum_coefficient=" << (kFullCoefficient + kLevelCoefficient) << '\n';
        out << "product_coefficient=" << (kFullCoefficient * kLevelCoefficient) << '\n';
        out << "convolution_logical_slots=" << (2 * kLogicalSlots - 1) << '\n';

        cleanup();
        return ok;
    } catch (...) {
        cleanup();
        throw;
    }
}

bool reciprocal_seed_smoke_test(std::ostream& out) {
    constexpr std::size_t kBlockCount = 2;
    constexpr std::size_t kCoefficientCount = 8;
    constexpr std::size_t kCoefficientSlot = 1;
    constexpr std::uint32_t kLogicalSlots = 4;
    constexpr std::uint64_t kDenominatorCoefficient = 13;
    constexpr std::uint32_t kFullDenominatorScaleBits = 5;
    constexpr std::uint32_t kDropScaleBitsDelta = 2;
    constexpr std::uint32_t kDroppedDenominatorScaleBits = kFullDenominatorScaleBits + kDropScaleBitsDelta;
    constexpr std::uint32_t kTargetProductScaleBits = 24;
    constexpr int kTargetModulusCount = 2;

    const std::uint64_t reciprocal_seed_coefficient =
        reciprocal_seed_coefficient_u64(kDenominatorCoefficient, kTargetProductScaleBits);
    const std::uint64_t ideal_scaled_one_coefficient = 1ull << kTargetProductScaleBits;
    const std::uint64_t product_coefficient = kDenominatorCoefficient * reciprocal_seed_coefficient;
    const std::uint64_t error_coefficient = ideal_scaled_one_coefficient - product_coefficient;
    const std::uint32_t reciprocal_scale_bits = kTargetProductScaleBits - kDroppedDenominatorScaleBits;

    DeviceRnsTensor denominator_full;
    DeviceRnsTensor denominator_level;
    DeviceRnsTensor reciprocal_seed;
    DeviceRnsTensor product;
    DeviceRnsTensor ideal;
    DeviceRnsTensor error;
    DeviceRnsTensor corrected;

    const auto cleanup = [&]() {
        if (corrected.d_moduli != nullptr) {
            free_device_tensor(corrected);
        }
        if (error.d_moduli != nullptr) {
            free_device_tensor(error);
        }
        if (ideal.d_moduli != nullptr) {
            free_device_tensor(ideal);
        }
        if (product.d_moduli != nullptr) {
            free_device_tensor(product);
        }
        if (reciprocal_seed.d_moduli != nullptr) {
            free_device_tensor(reciprocal_seed);
        }
        if (denominator_level.d_moduli != nullptr) {
            free_device_tensor(denominator_level);
        }
        if (denominator_full.d_moduli != nullptr) {
            free_device_tensor(denominator_full);
        }
    };

    const auto expected_monomial = [](
        std::uint64_t coefficient,
        std::size_t slot_count,
        std::size_t coefficient_slot
    ) {
        std::vector<std::uint64_t> values(kBlockCount * slot_count, 0);
        for (std::size_t block = 0; block < kBlockCount; ++block) {
            values[block * slot_count + coefficient_slot] = coefficient;
        }
        return values;
    };

    try {
        denominator_full = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            kDenominatorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kFullDenominatorScaleBits
        );
        denominator_level = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        drop_moduli_prefix(denominator_full, denominator_level, kDropScaleBitsDelta);

        reciprocal_seed = make_reciprocal_seed_tensor(
            kBlockCount,
            kCoefficientCount,
            kDenominatorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kDroppedDenominatorScaleBits,
            kTargetProductScaleBits,
            kTargetModulusCount
        );
        product = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        ideal = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            ideal_scaled_one_coefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kTargetProductScaleBits,
            kTargetModulusCount
        );
        error = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        corrected = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);

        multiply_polynomial_blocks(denominator_level, reciprocal_seed, product);
        compute_residual_correction(ideal, product, error, corrected);

        const auto denominator_host = download_tensor(denominator_level);
        const auto reciprocal_host = download_tensor(reciprocal_seed);
        const auto product_host = download_tensor(product);
        const auto ideal_host = download_tensor(ideal);
        const auto error_host = download_tensor(error);
        const auto corrected_host = download_tensor(corrected);

        const auto denominator_scalars = reconstruct_scalars(denominator_host);
        const auto reciprocal_scalars = reconstruct_scalars(reciprocal_host);
        const auto product_scalars = reconstruct_scalars(product_host);
        const auto ideal_scalars = reconstruct_scalars(ideal_host);
        const auto error_scalars = reconstruct_scalars(error_host);
        const auto corrected_scalars = reconstruct_scalars(corrected_host);

        bool ok = true;
        ok = ok && denominator_host.modulus_count == kTargetModulusCount;
        ok = ok && reciprocal_host.modulus_count == kTargetModulusCount;
        ok = ok && product_host.modulus_count == kTargetModulusCount;

        ok = ok && validate_metadata_all_equal(denominator_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(denominator_host.scale_bits, kDroppedDenominatorScaleBits);
        ok = ok && validate_metadata_all_equal(denominator_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(denominator_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(reciprocal_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(reciprocal_host.scale_bits, reciprocal_scale_bits);
        ok = ok && validate_metadata_all_equal(reciprocal_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(reciprocal_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(product_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(product_host.scale_bits, kTargetProductScaleBits);
        ok = ok && validate_metadata_all_equal(product_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(product_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(ideal_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(ideal_host.scale_bits, kTargetProductScaleBits);
        ok = ok && validate_metadata_all_equal(ideal_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(ideal_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(error_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(error_host.scale_bits, kTargetProductScaleBits);
        ok = ok && validate_metadata_all_equal(error_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(error_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(corrected_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(corrected_host.scale_bits, kTargetProductScaleBits);
        ok = ok && validate_metadata_all_equal(corrected_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(corrected_host.signs, {1, 1});

        ok = ok && denominator_scalars == expected_monomial(kDenominatorCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && reciprocal_scalars == expected_monomial(reciprocal_seed_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && product_scalars == expected_monomial(product_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && ideal_scalars == expected_monomial(ideal_scaled_one_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && error_scalars == expected_monomial(error_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && corrected_scalars == expected_monomial(ideal_scaled_one_coefficient, kCoefficientCount, kCoefficientSlot);

        out << "reciprocal_seed_smoke_status=" << (ok ? "ok" : "failed") << '\n';
        out << "block_count=" << kBlockCount << '\n';
        out << "coefficient_count=" << kCoefficientCount << '\n';
        out << "constant_slot=" << kCoefficientSlot << '\n';
        out << "logical_slots=" << kLogicalSlots << '\n';
        out << "full_modulus_count=" << ModulusConfig::kDefaultModulusCount << '\n';
        out << "active_modulus_count=" << kTargetModulusCount << '\n';
        out << "denominator_coefficient=" << kDenominatorCoefficient << '\n';
        out << "reciprocal_seed_coefficient=" << reciprocal_seed_coefficient << '\n';
        out << "product_coefficient=" << product_coefficient << '\n';
        out << "ideal_scaled_one_coefficient=" << ideal_scaled_one_coefficient << '\n';
        out << "error_coefficient=" << error_coefficient << '\n';
        out << "denominator_scale_bits=" << kDroppedDenominatorScaleBits << '\n';
        out << "reciprocal_scale_bits=" << reciprocal_scale_bits << '\n';
        out << "target_product_scale_bits=" << kTargetProductScaleBits << '\n';

        cleanup();
        return ok;
    } catch (...) {
        cleanup();
        throw;
    }
}

bool sqrt_seed_smoke_test(std::ostream& out) {
    constexpr std::size_t kBlockCount = 2;
    constexpr std::size_t kCoefficientCount = 8;
    constexpr std::size_t kCoefficientSlot = 1;
    constexpr std::uint32_t kLogicalSlots = 4;
    constexpr std::uint64_t kRadicandCoefficient = 10005;
    constexpr std::uint32_t kFullRadicandScaleBits = 0;
    constexpr std::uint32_t kDropScaleBitsDelta = 0;
    constexpr std::uint32_t kDroppedRadicandScaleBits = kFullRadicandScaleBits + kDropScaleBitsDelta;
    constexpr std::uint32_t kTargetSqrtScaleBits = 20;
    constexpr int kTargetModulusCount = 2;

    const std::uint64_t sqrt_seed_coefficient = sqrt_seed_coefficient_u64(
        kRadicandCoefficient,
        kDroppedRadicandScaleBits,
        kTargetSqrtScaleBits
    );
    const std::uint64_t square_coefficient = sqrt_seed_coefficient * sqrt_seed_coefficient;
    const std::uint32_t kTargetSquareScaleBits = 2u * kTargetSqrtScaleBits;
    const std::uint64_t target_scaled_radicand_coefficient = exact_scaled_coefficient_u64(
        kRadicandCoefficient,
        kDroppedRadicandScaleBits,
        kTargetSquareScaleBits
    );
    const std::uint64_t error_coefficient = target_scaled_radicand_coefficient - square_coefficient;

    DeviceRnsTensor radicand_full;
    DeviceRnsTensor radicand_level;
    DeviceRnsTensor sqrt_seed;
    DeviceRnsTensor square;
    DeviceRnsTensor target;
    DeviceRnsTensor error;
    DeviceRnsTensor corrected;

    const auto cleanup = [&]() {
        if (corrected.d_moduli != nullptr) {
            free_device_tensor(corrected);
        }
        if (error.d_moduli != nullptr) {
            free_device_tensor(error);
        }
        if (target.d_moduli != nullptr) {
            free_device_tensor(target);
        }
        if (square.d_moduli != nullptr) {
            free_device_tensor(square);
        }
        if (sqrt_seed.d_moduli != nullptr) {
            free_device_tensor(sqrt_seed);
        }
        if (radicand_level.d_moduli != nullptr) {
            free_device_tensor(radicand_level);
        }
        if (radicand_full.d_moduli != nullptr) {
            free_device_tensor(radicand_full);
        }
    };

    const auto expected_monomial = [](
        std::uint64_t coefficient,
        std::size_t slot_count,
        std::size_t coefficient_slot
    ) {
        std::vector<std::uint64_t> values(kBlockCount * slot_count, 0);
        for (std::size_t block = 0; block < kBlockCount; ++block) {
            values[block * slot_count + coefficient_slot] = coefficient;
        }
        return values;
    };

    try {
        radicand_full = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            kRadicandCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kFullRadicandScaleBits
        );
        radicand_level = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        drop_moduli_prefix(radicand_full, radicand_level, kDropScaleBitsDelta);

        sqrt_seed = make_sqrt_seed_tensor(
            kBlockCount,
            kCoefficientCount,
            kRadicandCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kDroppedRadicandScaleBits,
            kTargetSqrtScaleBits,
            kTargetModulusCount
        );
        square = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        target = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            target_scaled_radicand_coefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kTargetSquareScaleBits,
            kTargetModulusCount
        );
        error = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        corrected = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);

        multiply_polynomial_blocks(sqrt_seed, sqrt_seed, square);
        compute_residual_correction(target, square, error, corrected);

        const auto radicand_host = download_tensor(radicand_level);
        const auto sqrt_seed_host = download_tensor(sqrt_seed);
        const auto square_host = download_tensor(square);
        const auto target_host = download_tensor(target);
        const auto error_host = download_tensor(error);
        const auto corrected_host = download_tensor(corrected);

        const auto radicand_scalars = reconstruct_scalars(radicand_host);
        const auto sqrt_seed_scalars = reconstruct_scalars(sqrt_seed_host);
        const auto square_scalars = reconstruct_scalars(square_host);
        const auto target_scalars = reconstruct_scalars(target_host);
        const auto error_scalars = reconstruct_scalars(error_host);
        const auto corrected_scalars = reconstruct_scalars(corrected_host);

        bool ok = true;
        ok = ok && radicand_host.modulus_count == kTargetModulusCount;
        ok = ok && sqrt_seed_host.modulus_count == kTargetModulusCount;
        ok = ok && square_host.modulus_count == kTargetModulusCount;

        ok = ok && validate_metadata_all_equal(radicand_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(radicand_host.scale_bits, kDroppedRadicandScaleBits);
        ok = ok && validate_metadata_all_equal(radicand_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(radicand_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(sqrt_seed_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(sqrt_seed_host.scale_bits, kTargetSqrtScaleBits);
        ok = ok && validate_metadata_all_equal(sqrt_seed_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(sqrt_seed_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(square_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(square_host.scale_bits, kTargetSquareScaleBits);
        ok = ok && validate_metadata_all_equal(square_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(square_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(target_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(target_host.scale_bits, kTargetSquareScaleBits);
        ok = ok && validate_metadata_all_equal(target_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(target_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(error_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(error_host.scale_bits, kTargetSquareScaleBits);
        ok = ok && validate_metadata_all_equal(error_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(error_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(corrected_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(corrected_host.scale_bits, kTargetSquareScaleBits);
        ok = ok && validate_metadata_all_equal(corrected_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(corrected_host.signs, {1, 1});

        ok = ok && radicand_scalars == expected_monomial(kRadicandCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && sqrt_seed_scalars == expected_monomial(sqrt_seed_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && square_scalars == expected_monomial(square_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && target_scalars == expected_monomial(target_scaled_radicand_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && error_scalars == expected_monomial(error_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && corrected_scalars == expected_monomial(target_scaled_radicand_coefficient, kCoefficientCount, kCoefficientSlot);

        out << "sqrt_seed_smoke_status=" << (ok ? "ok" : "failed") << '\n';
        out << "block_count=" << kBlockCount << '\n';
        out << "coefficient_count=" << kCoefficientCount << '\n';
        out << "constant_slot=" << kCoefficientSlot << '\n';
        out << "logical_slots=" << kLogicalSlots << '\n';
        out << "full_modulus_count=" << ModulusConfig::kDefaultModulusCount << '\n';
        out << "active_modulus_count=" << kTargetModulusCount << '\n';
        out << "radicand_coefficient=" << kRadicandCoefficient << '\n';
        out << "sqrt_seed_coefficient=" << sqrt_seed_coefficient << '\n';
        out << "square_coefficient=" << square_coefficient << '\n';
        out << "target_scaled_radicand_coefficient=" << target_scaled_radicand_coefficient << '\n';
        out << "error_coefficient=" << error_coefficient << '\n';
        out << "radicand_scale_bits=" << kDroppedRadicandScaleBits << '\n';
        out << "sqrt_scale_bits=" << kTargetSqrtScaleBits << '\n';
        out << "target_square_scale_bits=" << kTargetSquareScaleBits << '\n';

        cleanup();
        return ok;
    } catch (...) {
        cleanup();
        throw;
    }
}

bool division_smoke_test(std::ostream& out) {
    constexpr std::size_t kBlockCount = 2;
    constexpr std::size_t kCoefficientCount = 8;
    constexpr std::size_t kCoefficientSlot = 1;
    constexpr std::uint32_t kLogicalSlots = 4;
    constexpr std::uint64_t kNumeratorCoefficient = 355;
    constexpr std::uint64_t kDenominatorCoefficient = 113;
    constexpr std::uint32_t kFullNumeratorScaleBits = 4;
    constexpr std::uint32_t kFullDenominatorScaleBits = 2;
    constexpr std::uint32_t kDropScaleBitsDelta = 1;
    constexpr std::uint32_t kNumeratorScaleBits = kFullNumeratorScaleBits + kDropScaleBitsDelta;
    constexpr std::uint32_t kDenominatorScaleBits = kFullDenominatorScaleBits + kDropScaleBitsDelta;
    constexpr std::uint32_t kTargetProductScaleBits = 24;
    constexpr int kTargetModulusCount = 2;

    const std::uint64_t quotient_coefficient = division_quotient_coefficient_u64(
        kNumeratorCoefficient,
        kDenominatorCoefficient,
        kNumeratorScaleBits,
        kTargetProductScaleBits
    );
    const std::uint32_t quotient_scale_bits = kTargetProductScaleBits - kDenominatorScaleBits;
    const std::uint64_t scaled_numerator_coefficient = exact_scaled_coefficient_u64(
        kNumeratorCoefficient,
        kNumeratorScaleBits,
        kTargetProductScaleBits
    );
    const std::uint64_t product_coefficient = kDenominatorCoefficient * quotient_coefficient;
    const std::uint64_t remainder_coefficient = scaled_numerator_coefficient - product_coefficient;

    DeviceRnsTensor numerator_full;
    DeviceRnsTensor denominator_full;
    DeviceRnsTensor numerator_level;
    DeviceRnsTensor denominator_level;
    DeviceRnsTensor quotient;
    DeviceRnsTensor product;
    DeviceRnsTensor target;
    DeviceRnsTensor remainder;
    DeviceRnsTensor corrected;

    const auto cleanup = [&]() {
        if (corrected.d_moduli != nullptr) {
            free_device_tensor(corrected);
        }
        if (remainder.d_moduli != nullptr) {
            free_device_tensor(remainder);
        }
        if (target.d_moduli != nullptr) {
            free_device_tensor(target);
        }
        if (product.d_moduli != nullptr) {
            free_device_tensor(product);
        }
        if (quotient.d_moduli != nullptr) {
            free_device_tensor(quotient);
        }
        if (denominator_level.d_moduli != nullptr) {
            free_device_tensor(denominator_level);
        }
        if (numerator_level.d_moduli != nullptr) {
            free_device_tensor(numerator_level);
        }
        if (denominator_full.d_moduli != nullptr) {
            free_device_tensor(denominator_full);
        }
        if (numerator_full.d_moduli != nullptr) {
            free_device_tensor(numerator_full);
        }
    };

    const auto expected_monomial = [](
        std::uint64_t coefficient,
        std::size_t slot_count,
        std::size_t coefficient_slot
    ) {
        std::vector<std::uint64_t> values(kBlockCount * slot_count, 0);
        for (std::size_t block = 0; block < kBlockCount; ++block) {
            values[block * slot_count + coefficient_slot] = coefficient;
        }
        return values;
    };

    try {
        numerator_full = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            kNumeratorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kFullNumeratorScaleBits
        );
        denominator_full = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            kDenominatorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kFullDenominatorScaleBits
        );
        numerator_level = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        denominator_level = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        drop_moduli_prefix(numerator_full, numerator_level, kDropScaleBitsDelta);
        drop_moduli_prefix(denominator_full, denominator_level, kDropScaleBitsDelta);

        quotient = make_division_quotient_tensor(
            kBlockCount,
            kCoefficientCount,
            kNumeratorCoefficient,
            kDenominatorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kNumeratorScaleBits,
            kDenominatorScaleBits,
            kTargetProductScaleBits,
            kTargetModulusCount
        );
        product = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        target = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            scaled_numerator_coefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kTargetProductScaleBits,
            kTargetModulusCount
        );
        remainder = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);
        corrected = allocate_polynomial_block_tensor(kBlockCount, kCoefficientCount, kTargetModulusCount);

        multiply_polynomial_blocks(denominator_level, quotient, product);
        compute_residual_correction(target, product, remainder, corrected);

        const auto numerator_host = download_tensor(numerator_level);
        const auto denominator_host = download_tensor(denominator_level);
        const auto quotient_host = download_tensor(quotient);
        const auto product_host = download_tensor(product);
        const auto target_host = download_tensor(target);
        const auto remainder_host = download_tensor(remainder);
        const auto corrected_host = download_tensor(corrected);

        const auto numerator_scalars = reconstruct_scalars(numerator_host);
        const auto denominator_scalars = reconstruct_scalars(denominator_host);
        const auto quotient_scalars = reconstruct_scalars(quotient_host);
        const auto product_scalars = reconstruct_scalars(product_host);
        const auto target_scalars = reconstruct_scalars(target_host);
        const auto remainder_scalars = reconstruct_scalars(remainder_host);
        const auto corrected_scalars = reconstruct_scalars(corrected_host);

        bool ok = true;
        ok = ok && numerator_host.modulus_count == kTargetModulusCount;
        ok = ok && denominator_host.modulus_count == kTargetModulusCount;
        ok = ok && quotient_host.modulus_count == kTargetModulusCount;

        ok = ok && validate_metadata_all_equal(numerator_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(numerator_host.scale_bits, kNumeratorScaleBits);
        ok = ok && validate_metadata_all_equal(numerator_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(numerator_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(denominator_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(denominator_host.scale_bits, kDenominatorScaleBits);
        ok = ok && validate_metadata_all_equal(denominator_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(denominator_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(quotient_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(quotient_host.scale_bits, quotient_scale_bits);
        ok = ok && validate_metadata_all_equal(quotient_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(quotient_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(product_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(product_host.scale_bits, kTargetProductScaleBits);
        ok = ok && validate_metadata_all_equal(product_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(product_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(target_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(target_host.scale_bits, kTargetProductScaleBits);
        ok = ok && validate_metadata_all_equal(target_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(target_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(remainder_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(remainder_host.scale_bits, kTargetProductScaleBits);
        ok = ok && validate_metadata_all_equal(remainder_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(remainder_host.signs, {1, 1});

        ok = ok && validate_metadata_all_equal(corrected_host.logical_slots, kLogicalSlots);
        ok = ok && validate_metadata_all_equal(corrected_host.scale_bits, kTargetProductScaleBits);
        ok = ok && validate_metadata_all_equal(corrected_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_signs_equal(corrected_host.signs, {1, 1});

        ok = ok && numerator_scalars == expected_monomial(kNumeratorCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && denominator_scalars == expected_monomial(kDenominatorCoefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && quotient_scalars == expected_monomial(quotient_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && product_scalars == expected_monomial(product_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && target_scalars == expected_monomial(scaled_numerator_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && remainder_scalars == expected_monomial(remainder_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && corrected_scalars == expected_monomial(scaled_numerator_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && remainder_coefficient < kDenominatorCoefficient;

        out << "division_smoke_status=" << (ok ? "ok" : "failed") << '\n';
        out << "block_count=" << kBlockCount << '\n';
        out << "coefficient_count=" << kCoefficientCount << '\n';
        out << "constant_slot=" << kCoefficientSlot << '\n';
        out << "logical_slots=" << kLogicalSlots << '\n';
        out << "full_modulus_count=" << ModulusConfig::kDefaultModulusCount << '\n';
        out << "active_modulus_count=" << kTargetModulusCount << '\n';
        out << "numerator_coefficient=" << kNumeratorCoefficient << '\n';
        out << "denominator_coefficient=" << kDenominatorCoefficient << '\n';
        out << "quotient_coefficient=" << quotient_coefficient << '\n';
        out << "product_coefficient=" << product_coefficient << '\n';
        out << "scaled_numerator_coefficient=" << scaled_numerator_coefficient << '\n';
        out << "remainder_coefficient=" << remainder_coefficient << '\n';
        out << "numerator_scale_bits=" << kNumeratorScaleBits << '\n';
        out << "denominator_scale_bits=" << kDenominatorScaleBits << '\n';
        out << "quotient_scale_bits=" << quotient_scale_bits << '\n';
        out << "target_product_scale_bits=" << kTargetProductScaleBits << '\n';

        cleanup();
        return ok;
    } catch (...) {
        cleanup();
        throw;
    }
}

bool correction_domain_smoke_test(std::ostream& out) {
    constexpr std::size_t kBlockCount = 2;
    constexpr std::size_t kCoefficientCount = 8;
    constexpr std::size_t kCoefficientSlot = 1;
    constexpr std::uint32_t kLogicalSlots = 4;
    constexpr int kTargetModulusCount = 2;

    constexpr std::uint64_t kReciprocalDenominatorCoefficient = 13;
    constexpr std::uint32_t kReciprocalFullDenominatorScaleBits = 5;
    constexpr std::uint32_t kReciprocalDropScaleBitsDelta = 2;
    constexpr std::uint32_t kReciprocalDenominatorScaleBits =
        kReciprocalFullDenominatorScaleBits + kReciprocalDropScaleBitsDelta;
    constexpr std::uint32_t kReciprocalTargetProductScaleBits = 24;

    constexpr std::uint64_t kSqrtRadicandCoefficient = 10005;
    constexpr std::uint32_t kSqrtRadicandScaleBits = 0;
    constexpr std::uint32_t kSqrtTargetScaleBits = 20;

    constexpr std::uint64_t kDivisionNumeratorCoefficient = 355;
    constexpr std::uint64_t kDivisionDenominatorCoefficient = 113;
    constexpr std::uint32_t kDivisionFullNumeratorScaleBits = 4;
    constexpr std::uint32_t kDivisionFullDenominatorScaleBits = 2;
    constexpr std::uint32_t kDivisionDropScaleBitsDelta = 1;
    constexpr std::uint32_t kDivisionNumeratorScaleBits =
        kDivisionFullNumeratorScaleBits + kDivisionDropScaleBitsDelta;
    constexpr std::uint32_t kDivisionDenominatorScaleBits =
        kDivisionFullDenominatorScaleBits + kDivisionDropScaleBitsDelta;
    constexpr std::uint32_t kDivisionTargetProductScaleBits = 24;

    const std::uint64_t reciprocal_seed_coefficient = reciprocal_seed_coefficient_u64(
        kReciprocalDenominatorCoefficient,
        kReciprocalTargetProductScaleBits
    );
    const std::uint64_t reciprocal_product_coefficient =
        kReciprocalDenominatorCoefficient * reciprocal_seed_coefficient;
    const std::uint64_t reciprocal_target_coefficient = 1ull << kReciprocalTargetProductScaleBits;
    const std::uint64_t reciprocal_residual_coefficient =
        reciprocal_target_coefficient - reciprocal_product_coefficient;

    const std::uint64_t sqrt_seed_coefficient = sqrt_seed_coefficient_u64(
        kSqrtRadicandCoefficient,
        kSqrtRadicandScaleBits,
        kSqrtTargetScaleBits
    );
    const std::uint64_t sqrt_square_coefficient = sqrt_seed_coefficient * sqrt_seed_coefficient;
    const std::uint32_t sqrt_target_square_scale_bits = 2u * kSqrtTargetScaleBits;
    const std::uint64_t sqrt_target_coefficient = exact_scaled_coefficient_u64(
        kSqrtRadicandCoefficient,
        kSqrtRadicandScaleBits,
        sqrt_target_square_scale_bits
    );
    const std::uint64_t sqrt_residual_coefficient = sqrt_target_coefficient - sqrt_square_coefficient;

    const std::uint64_t division_quotient_coefficient = division_quotient_coefficient_u64(
        kDivisionNumeratorCoefficient,
        kDivisionDenominatorCoefficient,
        kDivisionNumeratorScaleBits,
        kDivisionTargetProductScaleBits
    );
    const std::uint64_t division_product_coefficient =
        kDivisionDenominatorCoefficient * division_quotient_coefficient;
    const std::uint64_t division_target_coefficient = exact_scaled_coefficient_u64(
        kDivisionNumeratorCoefficient,
        kDivisionNumeratorScaleBits,
        kDivisionTargetProductScaleBits
    );
    const std::uint64_t division_residual_coefficient =
        division_target_coefficient - division_product_coefficient;

    DeviceRnsTensor reciprocal_denominator_full;
    DeviceRnsTensor reciprocal_denominator_level;
    DeviceRnsTensor reciprocal_seed;
    DeviceRnsTensor reciprocal_product;
    DeviceRnsTensor reciprocal_target;
    DeviceRnsTensor reciprocal_residual;
    DeviceRnsTensor reciprocal_corrected;

    DeviceRnsTensor sqrt_seed;
    DeviceRnsTensor sqrt_square;
    DeviceRnsTensor sqrt_target;
    DeviceRnsTensor sqrt_residual;
    DeviceRnsTensor sqrt_corrected;

    DeviceRnsTensor division_denominator_full;
    DeviceRnsTensor division_denominator_level;
    DeviceRnsTensor division_quotient;
    DeviceRnsTensor division_product;
    DeviceRnsTensor division_target;
    DeviceRnsTensor division_residual;
    DeviceRnsTensor division_corrected;

    const auto cleanup = [&]() {
        DeviceRnsTensor* tensors[] = {
            &division_corrected,
            &division_residual,
            &division_target,
            &division_product,
            &division_quotient,
            &division_denominator_level,
            &division_denominator_full,
            &sqrt_corrected,
            &sqrt_residual,
            &sqrt_target,
            &sqrt_square,
            &sqrt_seed,
            &reciprocal_corrected,
            &reciprocal_residual,
            &reciprocal_target,
            &reciprocal_product,
            &reciprocal_seed,
            &reciprocal_denominator_level,
            &reciprocal_denominator_full,
        };
        for (DeviceRnsTensor* tensor : tensors) {
            if (tensor->d_moduli != nullptr) {
                free_device_tensor(*tensor);
            }
        }
    };

    const auto expected_monomial = [](
        std::uint64_t coefficient,
        std::size_t slot_count,
        std::size_t coefficient_slot
    ) {
        std::vector<std::uint64_t> values(kBlockCount * slot_count, 0);
        for (std::size_t block = 0; block < kBlockCount; ++block) {
            values[block * slot_count + coefficient_slot] = coefficient;
        }
        return values;
    };

    const auto validate_same_domain = [](
        const HostRnsTensor& target_host,
        const HostRnsTensor& residual_host,
        const HostRnsTensor& corrected_host,
        std::uint32_t expected_logical_slots,
        std::uint32_t expected_scale_bits
    ) {
        bool ok = true;
        ok = ok && target_host.modulus_count == kTargetModulusCount;
        ok = ok && residual_host.modulus_count == kTargetModulusCount;
        ok = ok && corrected_host.modulus_count == kTargetModulusCount;
        ok = ok && target_host.moduli == residual_host.moduli;
        ok = ok && target_host.moduli == corrected_host.moduli;

        ok = ok && validate_metadata_all_equal(target_host.logical_slots, expected_logical_slots);
        ok = ok && validate_metadata_all_equal(residual_host.logical_slots, expected_logical_slots);
        ok = ok && validate_metadata_all_equal(corrected_host.logical_slots, expected_logical_slots);
        ok = ok && target_host.logical_slots == residual_host.logical_slots;
        ok = ok && target_host.logical_slots == corrected_host.logical_slots;

        ok = ok && validate_metadata_all_equal(target_host.scale_bits, expected_scale_bits);
        ok = ok && validate_metadata_all_equal(residual_host.scale_bits, expected_scale_bits);
        ok = ok && validate_metadata_all_equal(corrected_host.scale_bits, expected_scale_bits);
        ok = ok && target_host.scale_bits == residual_host.scale_bits;
        ok = ok && target_host.scale_bits == corrected_host.scale_bits;

        ok = ok && validate_metadata_all_equal(target_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_metadata_all_equal(residual_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && validate_metadata_all_equal(corrected_host.levels, static_cast<std::uint32_t>(kTargetModulusCount));
        ok = ok && target_host.levels == residual_host.levels;
        ok = ok && target_host.levels == corrected_host.levels;

        ok = ok && validate_signs_equal(target_host.signs, {1, 1});
        ok = ok && validate_signs_equal(residual_host.signs, {1, 1});
        ok = ok && validate_signs_equal(corrected_host.signs, {1, 1});
        ok = ok && target_host.signs == residual_host.signs;
        ok = ok && target_host.signs == corrected_host.signs;
        return ok;
    };

    try {
        reciprocal_denominator_full = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            kReciprocalDenominatorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kReciprocalFullDenominatorScaleBits
        );
        reciprocal_denominator_level = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        drop_moduli_prefix(
            reciprocal_denominator_full,
            reciprocal_denominator_level,
            kReciprocalDropScaleBitsDelta
        );
        reciprocal_seed = make_reciprocal_seed_tensor(
            kBlockCount,
            kCoefficientCount,
            kReciprocalDenominatorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kReciprocalDenominatorScaleBits,
            kReciprocalTargetProductScaleBits,
            kTargetModulusCount
        );
        reciprocal_product = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        reciprocal_target = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            reciprocal_target_coefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kReciprocalTargetProductScaleBits,
            kTargetModulusCount
        );
        reciprocal_residual = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        reciprocal_corrected = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );

        sqrt_seed = make_sqrt_seed_tensor(
            kBlockCount,
            kCoefficientCount,
            kSqrtRadicandCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kSqrtRadicandScaleBits,
            kSqrtTargetScaleBits,
            kTargetModulusCount
        );
        sqrt_square = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        sqrt_target = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            sqrt_target_coefficient,
            kCoefficientSlot,
            kLogicalSlots,
            sqrt_target_square_scale_bits,
            kTargetModulusCount
        );
        sqrt_residual = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        sqrt_corrected = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );

        division_denominator_full = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            kDivisionDenominatorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kDivisionFullDenominatorScaleBits
        );
        division_denominator_level = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        drop_moduli_prefix(
            division_denominator_full,
            division_denominator_level,
            kDivisionDropScaleBitsDelta
        );
        division_quotient = make_division_quotient_tensor(
            kBlockCount,
            kCoefficientCount,
            kDivisionNumeratorCoefficient,
            kDivisionDenominatorCoefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kDivisionNumeratorScaleBits,
            kDivisionDenominatorScaleBits,
            kDivisionTargetProductScaleBits,
            kTargetModulusCount
        );
        division_product = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        division_target = make_scaled_constant_tensor(
            kBlockCount,
            kCoefficientCount,
            division_target_coefficient,
            kCoefficientSlot,
            kLogicalSlots,
            kDivisionTargetProductScaleBits,
            kTargetModulusCount
        );
        division_residual = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );
        division_corrected = allocate_polynomial_block_tensor(
            kBlockCount,
            kCoefficientCount,
            kTargetModulusCount
        );

        multiply_polynomial_blocks(reciprocal_denominator_level, reciprocal_seed, reciprocal_product);
        compute_residual_correction(
            reciprocal_target,
            reciprocal_product,
            reciprocal_residual,
            reciprocal_corrected
        );

        multiply_polynomial_blocks(sqrt_seed, sqrt_seed, sqrt_square);
        compute_residual_correction(
            sqrt_target,
            sqrt_square,
            sqrt_residual,
            sqrt_corrected
        );

        multiply_polynomial_blocks(division_denominator_level, division_quotient, division_product);
        compute_residual_correction(
            division_target,
            division_product,
            division_residual,
            division_corrected
        );

        const auto reciprocal_target_host = download_tensor(reciprocal_target);
        const auto reciprocal_residual_host = download_tensor(reciprocal_residual);
        const auto reciprocal_corrected_host = download_tensor(reciprocal_corrected);
        const auto sqrt_target_host = download_tensor(sqrt_target);
        const auto sqrt_residual_host = download_tensor(sqrt_residual);
        const auto sqrt_corrected_host = download_tensor(sqrt_corrected);
        const auto division_target_host = download_tensor(division_target);
        const auto division_residual_host = download_tensor(division_residual);
        const auto division_corrected_host = download_tensor(division_corrected);

        const auto reciprocal_target_scalars = reconstruct_scalars(reciprocal_target_host);
        const auto reciprocal_residual_scalars = reconstruct_scalars(reciprocal_residual_host);
        const auto reciprocal_corrected_scalars = reconstruct_scalars(reciprocal_corrected_host);
        const auto sqrt_target_scalars = reconstruct_scalars(sqrt_target_host);
        const auto sqrt_residual_scalars = reconstruct_scalars(sqrt_residual_host);
        const auto sqrt_corrected_scalars = reconstruct_scalars(sqrt_corrected_host);
        const auto division_target_scalars = reconstruct_scalars(division_target_host);
        const auto division_residual_scalars = reconstruct_scalars(division_residual_host);
        const auto division_corrected_scalars = reconstruct_scalars(division_corrected_host);

        bool ok = true;
        ok = ok && validate_same_domain(
            reciprocal_target_host,
            reciprocal_residual_host,
            reciprocal_corrected_host,
            kLogicalSlots,
            kReciprocalTargetProductScaleBits
        );
        ok = ok && reciprocal_target_scalars ==
            expected_monomial(reciprocal_target_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && reciprocal_residual_scalars ==
            expected_monomial(reciprocal_residual_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && reciprocal_corrected_scalars ==
            expected_monomial(reciprocal_target_coefficient, kCoefficientCount, kCoefficientSlot);

        ok = ok && validate_same_domain(
            sqrt_target_host,
            sqrt_residual_host,
            sqrt_corrected_host,
            kLogicalSlots,
            sqrt_target_square_scale_bits
        );
        ok = ok && sqrt_target_scalars ==
            expected_monomial(sqrt_target_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && sqrt_residual_scalars ==
            expected_monomial(sqrt_residual_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && sqrt_corrected_scalars ==
            expected_monomial(sqrt_target_coefficient, kCoefficientCount, kCoefficientSlot);

        ok = ok && validate_same_domain(
            division_target_host,
            division_residual_host,
            division_corrected_host,
            kLogicalSlots,
            kDivisionTargetProductScaleBits
        );
        ok = ok && division_target_scalars ==
            expected_monomial(division_target_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && division_residual_scalars ==
            expected_monomial(division_residual_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && division_corrected_scalars ==
            expected_monomial(division_target_coefficient, kCoefficientCount, kCoefficientSlot);
        ok = ok && division_residual_coefficient < kDivisionDenominatorCoefficient;

        out << "correction_domain_smoke_status=" << (ok ? "ok" : "failed") << '\n';
        out << "shared_correction_api=compute_residual_correction" << '\n';
        out << "block_count=" << kBlockCount << '\n';
        out << "coefficient_count=" << kCoefficientCount << '\n';
        out << "constant_slot=" << kCoefficientSlot << '\n';
        out << "logical_slots=" << kLogicalSlots << '\n';
        out << "active_modulus_count=" << kTargetModulusCount << '\n';
        out << "reciprocal_target_scale_bits=" << kReciprocalTargetProductScaleBits << '\n';
        out << "reciprocal_residual_coefficient=" << reciprocal_residual_coefficient << '\n';
        out << "sqrt_target_scale_bits=" << sqrt_target_square_scale_bits << '\n';
        out << "sqrt_residual_coefficient=" << sqrt_residual_coefficient << '\n';
        out << "division_target_scale_bits=" << kDivisionTargetProductScaleBits << '\n';
        out << "division_residual_coefficient=" << division_residual_coefficient << '\n';

        cleanup();
        return ok;
    } catch (...) {
        cleanup();
        throw;
    }
}

PiRouteReport plan_pi_route(std::size_t target_digits) {
    if (target_digits == 0) {
        throw std::invalid_argument("pi route planning requires target_digits > 0");
    }

    constexpr long double kLog2Of10 = 3.32192809488736234787L;
    constexpr long double kChudnovskyDigitsPerTerm = 14.1816474627254776555L;

    const auto add_checked = [](std::size_t lhs, std::size_t rhs, const char* what) {
        if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
            throw std::overflow_error(std::string(what) + " overflowed size_t");
        }
        return lhs + rhs;
    };
    const auto decimal_length = [](std::size_t value) {
        std::size_t digits = 1;
        while (value >= 10) {
            value /= 10;
            ++digits;
        }
        return digits;
    };
    const auto ceil_to_size_t = [](long double value, const char* what) {
        if (!std::isfinite(value) || value < 0.0L) {
            throw std::invalid_argument(std::string(what) + " must be finite and non-negative");
        }
        const long double rounded = std::ceil(value);
        const long double max_value = static_cast<long double>(std::numeric_limits<std::size_t>::max());
        if (rounded > max_value) {
            throw std::overflow_error(std::string(what) + " overflowed size_t");
        }
        return static_cast<std::size_t>(rounded);
    };

    PiRouteReport report;
    report.target_digits = target_digits;
    report.guard_digits = std::max<std::size_t>(32, decimal_length(target_digits) + 16);
    report.working_digits = add_checked(report.target_digits, report.guard_digits, "working_digits");
    report.working_bits = ceil_to_size_t(
        static_cast<long double>(report.working_digits) * kLog2Of10,
        "working_bits"
    );
    report.chudnovsky_terms = std::max<std::size_t>(
        1,
        ceil_to_size_t(
            static_cast<long double>(report.working_digits) / kChudnovskyDigitsPerTerm,
            "chudnovsky_terms"
        )
    );
    report.binary_split_leaf_count = report.chudnovsky_terms;
    report.binary_split_internal_count =
        report.binary_split_leaf_count > 0 ? report.binary_split_leaf_count - 1 : 0;
    report.binary_split_depth = static_cast<std::size_t>(ceil_log2_size(report.binary_split_leaf_count));
    report.final_nonmultiply_steps = 2;
    report.estimated_decimal_output_chars = add_checked(report.target_digits, 2, "estimated_decimal_output_chars");
    report.chosen_route = "chudnovsky_binary_splitting";
    report.preferred_multiply_backbone = "exact_multiply_tree_on_native_rns_ntt";
    report.final_reciprocal_strategy = "single_final_division_after_binary_split";
    report.final_sqrt_strategy = "single_final_sqrt_10005";
    report.rejected_route = "agm_like_constant_algorithm";
    report.rejection_reason = "full_precision_division_and_sqrt_iterations_remain_prototype_level_on_native_rns";
    report.route_rationale =
        "current_native_rns_path_is_multiply_strong_and_binary_splitting_pushes_most_work_into_exact_multiply";
    report.ok = true;
    return report;
}

void print_pi_route_report(std::ostream& out, const PiRouteReport& report) {
    out << "pi_route_smoke_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "target_digits=" << report.target_digits << '\n';
    out << "guard_digits=" << report.guard_digits << '\n';
    out << "working_digits=" << report.working_digits << '\n';
    out << "working_bits=" << report.working_bits << '\n';
    out << "chosen_route=" << report.chosen_route << '\n';
    out << "preferred_multiply_backbone=" << report.preferred_multiply_backbone << '\n';
    out << "chudnovsky_terms=" << report.chudnovsky_terms << '\n';
    out << "binary_split_leaf_count=" << report.binary_split_leaf_count << '\n';
    out << "binary_split_internal_count=" << report.binary_split_internal_count << '\n';
    out << "binary_split_depth=" << report.binary_split_depth << '\n';
    out << "final_nonmultiply_steps=" << report.final_nonmultiply_steps << '\n';
    out << "final_reciprocal_strategy=" << report.final_reciprocal_strategy << '\n';
    out << "final_sqrt_strategy=" << report.final_sqrt_strategy << '\n';
    out << "rejected_route=" << report.rejected_route << '\n';
    out << "rejection_reason=" << report.rejection_reason << '\n';
    out << "route_rationale=" << report.route_rationale << '\n';
    out << "estimated_decimal_output_chars=" << report.estimated_decimal_output_chars << '\n';
}

bool pi_route_smoke_test(std::ostream& out, std::size_t target_digits) {
    const PiRouteReport report = plan_pi_route(target_digits);
    print_pi_route_report(out, report);
    return report.ok;
}

PiExecutionPlanReport plan_pi_execution(std::size_t target_digits) {
    const PiRouteReport route = plan_pi_route(target_digits);

    const auto ceil_div_size = [](std::size_t numerator, std::size_t denominator) {
        if (denominator == 0) {
            throw std::invalid_argument("ceil_div_size requires denominator > 0");
        }
        return numerator / denominator + (numerator % denominator == 0 ? 0 : 1);
    };
    const auto estimate_level_bits = [&](std::size_t max_terms_per_node) {
        if (route.chudnovsky_terms == 0) {
            throw std::invalid_argument("route.chudnovsky_terms must be positive");
        }
        const unsigned __int128 scaled =
            static_cast<unsigned __int128>(route.working_bits) * static_cast<unsigned __int128>(max_terms_per_node);
        const std::size_t proportional_bits =
            static_cast<std::size_t>(scaled / static_cast<unsigned __int128>(route.chudnovsky_terms)) +
            (scaled % static_cast<unsigned __int128>(route.chudnovsky_terms) == 0 ? 0 : 1);
        return std::max<std::size_t>(32, std::min(route.working_bits, proportional_bits));
    };
    const auto safe_limb_bits_for_slot_count = [&](std::size_t slot_count, std::size_t dynamic_range_bits) {
        if (slot_count == 0) {
            throw std::invalid_argument("safe_limb_bits_for_slot_count requires slot_count > 0");
        }
        const std::size_t accumulation_bits = static_cast<std::size_t>(ceil_log2_size(slot_count));
        if (dynamic_range_bits <= accumulation_bits + 2) {
            return std::size_t{1};
        }
        return (dynamic_range_bits - accumulation_bits) / 2;
    };

    PiExecutionPlanReport report;
    report.target_digits = route.target_digits;
    report.guard_digits = route.guard_digits;
    report.working_digits = route.working_digits;
    report.working_bits = route.working_bits;
    report.chudnovsky_terms = route.chudnovsky_terms;
    report.target_parallel_leaf_tasks = 4096;
    report.leaf_terms_per_task = next_power_of_two(
        std::max<std::size_t>(1, ceil_div_size(report.chudnovsky_terms, report.target_parallel_leaf_tasks))
    );
    report.leaf_task_count = ceil_div_size(report.chudnovsky_terms, report.leaf_terms_per_task);
    report.merge_level_count = static_cast<std::size_t>(ceil_log2_size(report.leaf_task_count)) + 1;

    HostBigInt modulus_product = 1;
    for (std::uint32_t modulus : ModulusConfig::kModuli) {
        modulus_product = modulus_product * static_cast<long long>(modulus);
    }
    report.modulus_dynamic_range_bits = bit_width_cpp_int(modulus_product);
    report.chosen_limb_bits = 32;
    report.chosen_route = route.chosen_route;
    report.execution_model = "batched_bottom_up_binary_split_merge_tree";
    report.peak_bottleneck = "root_level_exact_ntt_multiply_and_final_nonmultiply_closure";
    report.plan_rationale =
        "plan_leaf_batches_to_keep_thousands_of_parallel_tasks_then_merge_up_to_a_single_root_on_native_rns_ntt";

    report.levels.reserve(report.merge_level_count);
    for (std::size_t level_index = 0; level_index < report.merge_level_count; ++level_index) {
        const std::size_t node_divisor = std::size_t{1} << level_index;
        const std::size_t node_count = ceil_div_size(report.leaf_task_count, node_divisor);
        const std::size_t max_terms_per_node = std::min(
            report.chudnovsky_terms,
            report.leaf_terms_per_task * node_divisor
        );
        const std::size_t estimated_integer_bits = estimate_level_bits(max_terms_per_node);
        const std::size_t estimated_slot_count =
            ceil_div_size(estimated_integer_bits, report.chosen_limb_bits);
        const std::size_t estimated_ntt_size = resolve_convolution_ntt_size(2 * estimated_slot_count - 1);
        const std::size_t safe_limb_bits =
            safe_limb_bits_for_slot_count(estimated_slot_count, report.modulus_dynamic_range_bits);

        report.levels.push_back(PiExecutionLevelReport{
            .level_index = level_index,
            .node_count = node_count,
            .max_terms_per_node = max_terms_per_node,
            .estimated_integer_bits = estimated_integer_bits,
            .estimated_slot_count = estimated_slot_count,
            .estimated_ntt_size = estimated_ntt_size,
            .safe_limb_bits = safe_limb_bits,
        });
    }

    if (report.levels.empty()) {
        throw std::runtime_error("pi execution plan unexpectedly produced no levels");
    }

    const PiExecutionLevelReport& root = report.levels.back();
    report.peak_slot_count = root.estimated_slot_count;
    report.peak_ntt_size = root.estimated_ntt_size;
    report.root_safe_limb_bits = root.safe_limb_bits;
    report.ok = route.ok && report.root_safe_limb_bits >= report.chosen_limb_bits;
    return report;
}

void print_pi_execution_plan_report(std::ostream& out, const PiExecutionPlanReport& report) {
    out << "pi_execution_plan_smoke_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "target_digits=" << report.target_digits << '\n';
    out << "guard_digits=" << report.guard_digits << '\n';
    out << "working_digits=" << report.working_digits << '\n';
    out << "working_bits=" << report.working_bits << '\n';
    out << "chosen_route=" << report.chosen_route << '\n';
    out << "execution_model=" << report.execution_model << '\n';
    out << "chudnovsky_terms=" << report.chudnovsky_terms << '\n';
    out << "leaf_terms_per_task=" << report.leaf_terms_per_task << '\n';
    out << "leaf_task_count=" << report.leaf_task_count << '\n';
    out << "merge_level_count=" << report.merge_level_count << '\n';
    out << "target_parallel_leaf_tasks=" << report.target_parallel_leaf_tasks << '\n';
    out << "modulus_dynamic_range_bits=" << report.modulus_dynamic_range_bits << '\n';
    out << "chosen_limb_bits=" << report.chosen_limb_bits << '\n';
    out << "peak_slot_count=" << report.peak_slot_count << '\n';
    out << "peak_ntt_size=" << report.peak_ntt_size << '\n';
    out << "root_safe_limb_bits=" << report.root_safe_limb_bits << '\n';
    out << "peak_bottleneck=" << report.peak_bottleneck << '\n';
    out << "plan_rationale=" << report.plan_rationale << '\n';
    for (const PiExecutionLevelReport& level : report.levels) {
        out << "level_" << level.level_index << "_node_count=" << level.node_count << '\n';
        out << "level_" << level.level_index << "_max_terms_per_node=" << level.max_terms_per_node << '\n';
        out << "level_" << level.level_index << "_estimated_integer_bits=" << level.estimated_integer_bits << '\n';
        out << "level_" << level.level_index << "_estimated_slot_count=" << level.estimated_slot_count << '\n';
        out << "level_" << level.level_index << "_estimated_ntt_size=" << level.estimated_ntt_size << '\n';
        out << "level_" << level.level_index << "_safe_limb_bits=" << level.safe_limb_bits << '\n';
    }
}

bool pi_execution_plan_smoke_test(std::ostream& out, std::size_t target_digits) {
    const PiExecutionPlanReport report = plan_pi_execution(target_digits);
    print_pi_execution_plan_report(out, report);
    return report.ok;
}

namespace {

struct ChudnovskyExactNode {
    std::size_t begin_term = 0;
    std::size_t end_term = 0;
    std::size_t depth = 0;
    std::size_t left_child = std::numeric_limits<std::size_t>::max();
    std::size_t right_child = std::numeric_limits<std::size_t>::max();
    bool is_leaf = false;
    HostBigInt P;
    HostBigInt Q;
    HostBigInt T;
};

std::size_t bit_width_cpp_int(const HostBigInt& value) {
    if (value.is_zero()) {
        return 1;
    }
    const std::uint32_t top_limb = value.limbs.back();
    return 32u * (value.limbs.size() - 1u) + static_cast<std::size_t>(32 - __builtin_clz(top_limb));
}

std::vector<long long> bigint_to_signed_base_coefficients(const HostBigInt& value, std::size_t limb_bits) {
    if (limb_bits == 0 || limb_bits > 16 || (32u % limb_bits) != 0u) {
        throw std::invalid_argument("bigint_to_signed_base_coefficients requires limb_bits dividing 32 and <= 16");
    }
    if (value.is_zero()) {
        return {0};
    }

    const std::uint32_t mask =
        limb_bits == 32 ? 0xffffffffu : static_cast<std::uint32_t>((1ull << limb_bits) - 1ull);
    const std::size_t chunks_per_limb = 32u / limb_bits;
    std::vector<long long> coefficients;
    coefficients.reserve(value.limbs.size() * chunks_per_limb);
    for (std::uint32_t limb : value.limbs) {
        for (std::size_t chunk_index = 0; chunk_index < chunks_per_limb; ++chunk_index) {
            coefficients.push_back(
                static_cast<long long>((limb >> (chunk_index * limb_bits)) & mask)
            );
        }
    }
    while (coefficients.size() > 1 && coefficients.back() == 0) {
        coefficients.pop_back();
    }
    if (value.sign < 0) {
        for (long long& coefficient : coefficients) {
            coefficient = -coefficient;
        }
    }
    return coefficients;
}

HostBigInt rebuild_bigint_from_centered_coefficients(
    const std::vector<HostBigInt>& coefficients,
    std::size_t limb_bits
) {
    if (limb_bits == 0 || limb_bits > 16 || (32u % limb_bits) != 0u) {
        throw std::invalid_argument("rebuild_bigint_from_centered_coefficients requires limb_bits dividing 32 and <= 16");
    }

    const HostBigInt base = static_cast<long long>(1ull << limb_bits);
    HostBigInt value;
    for (std::size_t index = coefficients.size(); index > 0; --index) {
        value = value * base + coefficients[index - 1];
    }
    return value;
}

ChudnovskyExactNode make_chudnovsky_leaf(std::size_t term_index, std::size_t depth) {
    ChudnovskyExactNode node;
    node.begin_term = term_index;
    node.end_term = term_index + 1;
    node.depth = depth;
    node.is_leaf = true;
    if (term_index == 0) {
        node.P = 1;
        node.Q = 1;
        node.T = static_cast<long long>(kChudnovskyA);
        return node;
    }

    const HostBigInt k = static_cast<long long>(term_index);
    node.P = (6 * k - 5) * (2 * k - 1) * (6 * k - 1);
    node.Q = k * k * k * static_cast<long long>(kChudnovskyC3Over24);
    node.T = node.P * (static_cast<long long>(kChudnovskyA) + static_cast<long long>(kChudnovskyB) * k);
    if ((term_index & 1u) != 0u) {
        node.T = -node.T;
    }
    return node;
}

std::size_t build_chudnovsky_exact_tree(
    std::size_t begin_term,
    std::size_t end_term,
    std::size_t depth,
    std::vector<ChudnovskyExactNode>& nodes
) {
    if (end_term <= begin_term) {
        throw std::invalid_argument("build_chudnovsky_exact_tree requires begin_term < end_term");
    }
    if (end_term - begin_term == 1) {
        nodes.push_back(make_chudnovsky_leaf(begin_term, depth));
        return nodes.size() - 1;
    }

    const std::size_t mid_term = begin_term + (end_term - begin_term) / 2;
    const std::size_t left_index = build_chudnovsky_exact_tree(begin_term, mid_term, depth + 1, nodes);
    const std::size_t right_index = build_chudnovsky_exact_tree(mid_term, end_term, depth + 1, nodes);

    ChudnovskyExactNode node;
    node.begin_term = begin_term;
    node.end_term = end_term;
    node.depth = depth;
    node.left_child = left_index;
    node.right_child = right_index;
    node.is_leaf = false;
    node.P = nodes[left_index].P * nodes[right_index].P;
    node.Q = nodes[left_index].Q * nodes[right_index].Q;
    node.T = nodes[left_index].T * nodes[right_index].Q + nodes[left_index].P * nodes[right_index].T;
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
}

std::size_t find_first_leaf_merge_node(const std::vector<ChudnovskyExactNode>& nodes) {
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        const ChudnovskyExactNode& node = nodes[node_index];
        if (node.is_leaf) {
            continue;
        }
        if (node.left_child >= nodes.size() || node.right_child >= nodes.size()) {
            throw std::runtime_error("invalid child index in Chudnovsky exact tree");
        }
        if (nodes[node.left_child].is_leaf && nodes[node.right_child].is_leaf) {
            return node_index;
        }
    }
    throw std::runtime_error("expected at least one leaf-merge node in Chudnovsky exact tree");
}

int choose_pi_tensor_modulus_count(std::size_t term_count) {
    if (term_count <= 8) {
        return ModulusConfig::kDefaultModulusCount;
    }
    if (term_count <= 47) {
        return ModulusConfig::kIntermediateModulusCount;
    }
    return ModulusConfig::kModulusCount;
}

std::size_t choose_pi_tensor_limb_bits(std::size_t term_count) {
    if (term_count <= 4) {
        return kPiTensorLimbBits;
    }
    if (term_count <= 8) {
        return 8u;
    }
    if (term_count <= 16) {
        return 4u;
    }
    if (term_count <= 32) {
        return 2u;
    }
    if (term_count <= 47) {
        return 1u;
    }
    if (term_count <= 64) {
        return 1u;
    }
    throw std::invalid_argument("pi P/Q/T tensor tree smoke currently limits term_count to 64 on the current exact path");
}

struct PiTensorTriple {
    DeviceRnsTensor P;
    DeviceRnsTensor Q;
    DeviceRnsTensor T;
    std::uint32_t p_logical_slots = 0;
    std::uint32_t q_logical_slots = 0;
    std::uint32_t t_logical_slots = 0;
};

struct PiTensorTreeExecutionOptions {
    bool validate_all_metadata = true;
};

struct PiTensorTreeExecutionResult {
    PiPqtTensorTreeReport report;
    HostBigInt root_P;
    HostBigInt root_Q;
    HostBigInt root_T;
};

struct PooledTensorEntry {
    DeviceRnsTensor tensor;
    bool in_use = false;
};

struct PiTensorMemoryPool {
    std::vector<PooledTensorEntry> entries;
};

struct PiTensorTreeExecutionWorkspace {
    std::size_t term_count = 0;
    std::size_t root_index = 0;
    std::size_t max_depth = 0;
    std::size_t chosen_limb_bits = 0;
    int modulus_count = 0;
    std::vector<ChudnovskyExactNode> exact_nodes;
    std::vector<std::vector<std::size_t>> nodes_by_depth;
    std::vector<PiTensorTriple> tensor_nodes;
    std::vector<bool> tensor_live;
    std::vector<bool> tensor_owned;
    std::vector<PiTensorTriple> leaf_cache;
    std::vector<bool> leaf_cached;
    PiTensorMemoryPool pool;
};

struct PiMergeShapeKey {
    std::size_t p_left_slots = 0;
    std::size_t p_right_slots = 0;
    std::size_t q_left_slots = 0;
    std::size_t q_right_slots = 0;
    std::size_t t_left_slots = 0;
    std::size_t t_right_slots = 0;

    bool operator==(const PiMergeShapeKey& other) const = default;
};

struct PiMergeShapeGroup {
    PiMergeShapeKey key;
    std::vector<std::size_t> node_indices;
};

DeviceRnsTensor acquire_pooled_polynomial_block_tensor(
    PiTensorMemoryPool& pool,
    std::size_t block_count,
    std::size_t coefficient_count,
    int modulus_count
) {
    for (auto& entry : pool.entries) {
        if (entry.in_use) {
            continue;
        }
        if (entry.tensor.value_count != block_count ||
            entry.tensor.slot_count != coefficient_count ||
            entry.tensor.modulus_count != modulus_count) {
            continue;
        }
        entry.in_use = true;
        return entry.tensor;
    }

    PooledTensorEntry entry;
    entry.tensor = allocate_polynomial_block_tensor(block_count, coefficient_count, modulus_count);
    entry.in_use = true;
    pool.entries.push_back(std::move(entry));
    return pool.entries.back().tensor;
}

void release_pooled_tensor(PiTensorMemoryPool& pool, DeviceRnsTensor& tensor) {
    if (tensor.d_residues == nullptr) {
        tensor = {};
        return;
    }

    for (auto& entry : pool.entries) {
        if (entry.tensor.d_residues != tensor.d_residues) {
            continue;
        }
        if (!entry.in_use) {
            throw std::runtime_error("attempted to release an idle pooled tensor");
        }
        entry.in_use = false;
        tensor = {};
        return;
    }

    throw std::runtime_error("attempted to release a tensor that is not owned by the pool");
}

void destroy_pi_tensor_memory_pool(PiTensorMemoryPool& pool) {
    for (auto& entry : pool.entries) {
        if (entry.tensor.d_moduli != nullptr) {
            free_device_tensor(entry.tensor);
        }
        entry.in_use = false;
    }
    pool.entries.clear();
}

void release_pooled_pi_tensor_triple(PiTensorMemoryPool& pool, PiTensorTriple& triple) {
    if (triple.T.d_residues != nullptr) {
        release_pooled_tensor(pool, triple.T);
    }
    if (triple.Q.d_residues != nullptr) {
        release_pooled_tensor(pool, triple.Q);
    }
    if (triple.P.d_residues != nullptr) {
        release_pooled_tensor(pool, triple.P);
    }
    triple = {};
}

PiTensorTriple make_pi_leaf_tensor_triple_pooled(
    PiTensorMemoryPool& pool,
    const ChudnovskyExactNode& node,
    std::size_t limb_bits,
    int modulus_count
);

PiTensorTreeExecutionWorkspace& pi_tensor_tree_execution_workspace() {
    static PiTensorTreeExecutionWorkspace workspace;
    return workspace;
}

void release_workspace_live_tensors(PiTensorTreeExecutionWorkspace& workspace) {
    for (std::size_t node_index = 0; node_index < workspace.tensor_nodes.size(); ++node_index) {
        if (!workspace.tensor_live[node_index] || !workspace.tensor_owned[node_index]) {
            continue;
        }
        release_pooled_pi_tensor_triple(workspace.pool, workspace.tensor_nodes[node_index]);
        workspace.tensor_live[node_index] = false;
        workspace.tensor_owned[node_index] = false;
    }
}

void release_workspace_leaf_cache(PiTensorTreeExecutionWorkspace& workspace) {
    for (std::size_t node_index = 0; node_index < workspace.leaf_cache.size(); ++node_index) {
        if (!workspace.leaf_cached[node_index]) {
            continue;
        }
        release_pooled_pi_tensor_triple(workspace.pool, workspace.leaf_cache[node_index]);
        workspace.leaf_cached[node_index] = false;
    }
}

void reset_pi_tensor_tree_execution_workspace(PiTensorTreeExecutionWorkspace& workspace) {
    release_workspace_live_tensors(workspace);
    release_workspace_leaf_cache(workspace);
    destroy_pi_tensor_memory_pool(workspace.pool);
    workspace = {};
}

PiTensorTreeExecutionWorkspace& prepare_pi_tensor_tree_execution_workspace(std::size_t term_count) {
    PiTensorTreeExecutionWorkspace& workspace = pi_tensor_tree_execution_workspace();
    const std::size_t chosen_limb_bits = choose_pi_tensor_limb_bits(term_count);
    const int modulus_count = choose_pi_tensor_modulus_count(term_count);
    if (workspace.term_count == term_count &&
        workspace.chosen_limb_bits == chosen_limb_bits &&
        workspace.modulus_count == modulus_count &&
        !workspace.exact_nodes.empty()) {
        release_workspace_live_tensors(workspace);
        return workspace;
    }

    reset_pi_tensor_tree_execution_workspace(workspace);
    workspace.term_count = term_count;
    workspace.chosen_limb_bits = chosen_limb_bits;
    workspace.modulus_count = modulus_count;
    workspace.exact_nodes.reserve(2 * term_count - 1);
    workspace.root_index = build_chudnovsky_exact_tree(0, term_count, 0, workspace.exact_nodes);
    for (const ChudnovskyExactNode& node : workspace.exact_nodes) {
        workspace.max_depth = std::max(workspace.max_depth, node.depth);
    }
    workspace.nodes_by_depth.resize(workspace.max_depth + 1);
    for (std::size_t node_index = 0; node_index < workspace.exact_nodes.size(); ++node_index) {
        workspace.nodes_by_depth[workspace.exact_nodes[node_index].depth].push_back(node_index);
    }
    workspace.tensor_nodes.resize(workspace.exact_nodes.size());
    workspace.tensor_live.assign(workspace.exact_nodes.size(), false);
    workspace.tensor_owned.assign(workspace.exact_nodes.size(), false);
    workspace.leaf_cache.resize(workspace.exact_nodes.size());
    workspace.leaf_cached.assign(workspace.exact_nodes.size(), false);
    return workspace;
}

void ensure_pi_tensor_tree_leaf_cache(PiTensorTreeExecutionWorkspace& workspace) {
    for (std::size_t node_index = 0; node_index < workspace.exact_nodes.size(); ++node_index) {
        const ChudnovskyExactNode& node = workspace.exact_nodes[node_index];
        if (!node.is_leaf || workspace.leaf_cached[node_index]) {
            continue;
        }
        workspace.leaf_cache[node_index] = make_pi_leaf_tensor_triple_pooled(
            workspace.pool,
            node,
            workspace.chosen_limb_bits,
            workspace.modulus_count
        );
        workspace.leaf_cached[node_index] = true;
    }
}

PiMergeShapeKey make_pi_merge_shape_key(const PiTensorTriple& left, const PiTensorTriple& right) {
    return PiMergeShapeKey{
        .p_left_slots = left.p_logical_slots,
        .p_right_slots = right.p_logical_slots,
        .q_left_slots = left.q_logical_slots,
        .q_right_slots = right.q_logical_slots,
        .t_left_slots = left.t_logical_slots,
        .t_right_slots = right.t_logical_slots,
    };
}

std::vector<PiMergeShapeGroup> build_pi_merge_shape_groups(
    const std::vector<std::size_t>& node_indices,
    const std::vector<ChudnovskyExactNode>& exact_nodes,
    const std::vector<PiTensorTriple>& tensor_nodes
) {
    std::vector<PiMergeShapeGroup> groups;
    for (std::size_t node_index : node_indices) {
        const ChudnovskyExactNode& node = exact_nodes[node_index];
        const PiMergeShapeKey key = make_pi_merge_shape_key(
            tensor_nodes[node.left_child],
            tensor_nodes[node.right_child]
        );
        bool inserted = false;
        for (auto& group : groups) {
            if (!(group.key == key)) {
                continue;
            }
            group.node_indices.push_back(node_index);
            inserted = true;
            break;
        }
        if (!inserted) {
            groups.push_back(PiMergeShapeGroup{
                .key = key,
                .node_indices = {node_index},
            });
        }
    }
    return groups;
}

void copy_single_value_tensor_into_batch(
    const DeviceRnsTensor& src,
    std::size_t batch_index,
    DeviceRnsTensor& dst
) {
    if (src.value_count != 1 || src.modulus_count != dst.modulus_count || batch_index >= dst.value_count) {
        throw std::invalid_argument("copy_single_value_tensor_into_batch requires src.value_count == 1 and matching modulus_count");
    }
    if (src.slot_count != dst.slot_count) {
        throw std::invalid_argument("copy_single_value_tensor_into_batch requires matching slot_count");
    }

    for (int modulus_index = 0; modulus_index < src.modulus_count; ++modulus_index) {
        const std::size_t src_offset = static_cast<std::size_t>(modulus_index) * src.slot_count;
        const std::size_t dst_offset =
            (static_cast<std::size_t>(modulus_index) * dst.value_count + batch_index) * dst.slot_count;
        check_cuda(
            cudaMemcpy(
                dst.d_residues + dst_offset,
                src.d_residues + src_offset,
                sizeof(std::uint32_t) * src.slot_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy copy_single_value_tensor_into_batch"
        );
    }
}

void copy_single_value_tensor_prefix_into_batch(
    const DeviceRnsTensor& src,
    std::size_t batch_index,
    DeviceRnsTensor& dst
) {
    if (src.value_count != 1 || src.modulus_count != dst.modulus_count || batch_index >= dst.value_count) {
        throw std::invalid_argument("copy_single_value_tensor_prefix_into_batch requires src.value_count == 1 and matching modulus_count");
    }
    if (src.slot_count > dst.slot_count) {
        throw std::invalid_argument("copy_single_value_tensor_prefix_into_batch requires src.slot_count <= dst.slot_count");
    }

    for (int modulus_index = 0; modulus_index < src.modulus_count; ++modulus_index) {
        const std::size_t src_offset = static_cast<std::size_t>(modulus_index) * src.slot_count;
        const std::size_t dst_offset =
            (static_cast<std::size_t>(modulus_index) * dst.value_count + batch_index) * dst.slot_count;
        check_cuda(
            cudaMemcpy(
                dst.d_residues + dst_offset,
                src.d_residues + src_offset,
                sizeof(std::uint32_t) * src.slot_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy copy_single_value_tensor_prefix_into_batch"
        );
    }
}

void copy_batch_value_into_single_tensor(
    const DeviceRnsTensor& src,
    std::size_t batch_index,
    DeviceRnsTensor& dst
) {
    if (dst.value_count != 1 || src.modulus_count != dst.modulus_count || batch_index >= src.value_count) {
        throw std::invalid_argument("copy_batch_value_into_single_tensor requires dst.value_count == 1 and matching modulus_count");
    }
    if (src.slot_count != dst.slot_count) {
        throw std::invalid_argument("copy_batch_value_into_single_tensor requires matching slot_count");
    }

    for (int modulus_index = 0; modulus_index < src.modulus_count; ++modulus_index) {
        const std::size_t src_offset =
            (static_cast<std::size_t>(modulus_index) * src.value_count + batch_index) * src.slot_count;
        const std::size_t dst_offset = static_cast<std::size_t>(modulus_index) * dst.slot_count;
        check_cuda(
            cudaMemcpy(
                dst.d_residues + dst_offset,
                src.d_residues + src_offset,
                sizeof(std::uint32_t) * dst.slot_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy copy_batch_value_into_single_tensor"
        );
    }
}

PiTensorTriple make_pi_leaf_tensor_triple_pooled(
    PiTensorMemoryPool& pool,
    const ChudnovskyExactNode& node,
    std::size_t limb_bits,
    int modulus_count
) {
    if (!node.is_leaf) {
        throw std::invalid_argument("make_pi_leaf_tensor_triple_pooled requires a leaf node");
    }

    const std::vector<long long> p_coeffs = bigint_to_signed_base_coefficients(node.P, limb_bits);
    const std::vector<long long> q_coeffs = bigint_to_signed_base_coefficients(node.Q, limb_bits);
    const std::vector<long long> t_coeffs = bigint_to_signed_base_coefficients(node.T, limb_bits);

    PiTensorTriple triple;
    try {
        triple.P = acquire_pooled_polynomial_block_tensor(pool, 1, p_coeffs.size(), modulus_count);
        triple.Q = acquire_pooled_polynomial_block_tensor(pool, 1, q_coeffs.size(), modulus_count);
        triple.T = acquire_pooled_polynomial_block_tensor(pool, 1, t_coeffs.size(), modulus_count);
        encode_signed_polynomial_blocks(triple.P, p_coeffs, static_cast<std::uint32_t>(p_coeffs.size()));
        encode_signed_polynomial_blocks(triple.Q, q_coeffs, static_cast<std::uint32_t>(q_coeffs.size()));
        encode_signed_polynomial_blocks(triple.T, t_coeffs, static_cast<std::uint32_t>(t_coeffs.size()));
        triple.p_logical_slots = static_cast<std::uint32_t>(p_coeffs.size());
        triple.q_logical_slots = static_cast<std::uint32_t>(q_coeffs.size());
        triple.t_logical_slots = static_cast<std::uint32_t>(t_coeffs.size());
        return triple;
    } catch (...) {
        release_pooled_pi_tensor_triple(pool, triple);
        throw;
    }
}

DeviceRnsTensor pad_polynomial_tensor_suffix_zeros_pooled(
    PiTensorMemoryPool& pool,
    const DeviceRnsTensor& src,
    std::size_t dst_slot_count
) {
    if (dst_slot_count < src.slot_count) {
        throw std::invalid_argument("pad_polynomial_tensor_suffix_zeros_pooled requires dst_slot_count >= src.slot_count");
    }

    DeviceRnsTensor dst = acquire_pooled_polynomial_block_tensor(
        pool,
        src.value_count,
        dst_slot_count,
        src.modulus_count
    );
    try {
        check_cuda(
            cudaMemcpy(
                dst.d_moduli,
                src.d_moduli,
                sizeof(std::uint32_t) * static_cast<std::size_t>(src.modulus_count),
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy pooled padded tensor moduli"
        );
        copy_residues_with_zero_suffix_kernel<<<block_count(residue_count(dst)), kThreadsPerBlock>>>(
            src.d_residues,
            dst.d_residues,
            src.value_count,
            src.slot_count,
            dst.slot_count,
            src.modulus_count
        );
        check_cuda(cudaGetLastError(), "copy_residues_with_zero_suffix_kernel pooled launch");
        check_cuda(
            cudaMemcpy(
                dst.d_signs,
                src.d_signs,
                sizeof(std::int8_t) * src.value_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy pooled padded tensor signs"
        );
        check_cuda(
            cudaMemcpy(
                dst.d_logical_slots,
                src.d_logical_slots,
                sizeof(std::uint32_t) * src.value_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy pooled padded tensor logical_slots"
        );
        check_cuda(
            cudaMemcpy(
                dst.d_scale_bits,
                src.d_scale_bits,
                sizeof(std::uint32_t) * src.value_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy pooled padded tensor scale_bits"
        );
        check_cuda(
            cudaMemcpy(
                dst.d_levels,
                src.d_levels,
                sizeof(std::uint32_t) * src.value_count,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy pooled padded tensor levels"
        );
        return dst;
    } catch (...) {
        release_pooled_tensor(pool, dst);
        throw;
    }
}

PiTensorTriple merge_pi_tensor_triples_pooled(
    PiTensorMemoryPool& pool,
    const PiTensorTriple& left,
    const PiTensorTriple& right,
    bool populate_metadata
) {
    if (left.P.modulus_count != right.P.modulus_count ||
        left.Q.modulus_count != right.Q.modulus_count ||
        left.T.modulus_count != right.T.modulus_count ||
        left.P.modulus_count != left.Q.modulus_count ||
        left.P.modulus_count != left.T.modulus_count) {
        throw std::invalid_argument("merge_pi_tensor_triples_pooled requires a uniform modulus_count across both operands");
    }

    const int modulus_count = left.P.modulus_count;
    const std::size_t p_out_slots = left.p_logical_slots + right.p_logical_slots - 1;
    const std::size_t q_out_slots = left.q_logical_slots + right.q_logical_slots - 1;
    const std::size_t t_lq_slots = left.t_logical_slots + right.q_logical_slots - 1;
    const std::size_t t_pr_slots = left.p_logical_slots + right.t_logical_slots - 1;
    const std::size_t t_out_slots = std::max(t_lq_slots, t_pr_slots);

    PiTensorTriple out;
    DeviceRnsTensor padded_q_right;
    DeviceRnsTensor padded_t_right;
    DeviceRnsTensor t_lq_tensor;
    DeviceRnsTensor t_pr_tensor;
    bool padded_q_right_owned = false;
    bool padded_t_right_owned = false;

    const auto cleanup = [&]() {
        if (t_pr_tensor.d_residues != nullptr) {
            release_pooled_tensor(pool, t_pr_tensor);
        }
        if (t_lq_tensor.d_residues != nullptr) {
            release_pooled_tensor(pool, t_lq_tensor);
        }
        if (padded_t_right_owned && padded_t_right.d_residues != nullptr) {
            release_pooled_tensor(pool, padded_t_right);
        }
        if (padded_q_right_owned && padded_q_right.d_residues != nullptr) {
            release_pooled_tensor(pool, padded_q_right);
        }
    };

    try {
        out.P = acquire_pooled_polynomial_block_tensor(pool, 1, p_out_slots, modulus_count);
        out.Q = acquire_pooled_polynomial_block_tensor(pool, 1, q_out_slots, modulus_count);
        out.T = acquire_pooled_polynomial_block_tensor(pool, 1, t_out_slots, modulus_count);
        t_lq_tensor = acquire_pooled_polynomial_block_tensor(pool, 1, t_out_slots, modulus_count);
        t_pr_tensor = acquire_pooled_polynomial_block_tensor(pool, 1, t_out_slots, modulus_count);
        const std::size_t padded_q_slots = t_out_slots + 1 - left.T.slot_count;
        const std::size_t padded_t_slots = t_out_slots + 1 - left.P.slot_count;
        if (right.Q.slot_count == padded_q_slots) {
            padded_q_right = right.Q;
        } else {
            padded_q_right = pad_polynomial_tensor_suffix_zeros_pooled(pool, right.Q, padded_q_slots);
            padded_q_right_owned = true;
        }
        if (right.T.slot_count == padded_t_slots) {
            padded_t_right = right.T;
        } else {
            padded_t_right = pad_polynomial_tensor_suffix_zeros_pooled(pool, right.T, padded_t_slots);
            padded_t_right_owned = true;
        }

        convolve_polynomial_blocks_raw(left.P, right.P, out.P);
        convolve_polynomial_blocks_raw(left.Q, right.Q, out.Q);
        convolve_polynomial_blocks_raw(left.T, padded_q_right, t_lq_tensor);
        convolve_polynomial_blocks_raw(left.P, padded_t_right, t_pr_tensor);
        add_polynomial_blocks_raw(t_lq_tensor, t_pr_tensor, out.T);

        out.p_logical_slots = static_cast<std::uint32_t>(p_out_slots);
        out.q_logical_slots = static_cast<std::uint32_t>(q_out_slots);
        out.t_logical_slots = static_cast<std::uint32_t>(t_out_slots);
        if (populate_metadata) {
            set_uniform_tensor_metadata(out.P, 1, out.p_logical_slots, 0u, static_cast<std::uint32_t>(modulus_count));
            set_uniform_tensor_metadata(out.Q, 1, out.q_logical_slots, 0u, static_cast<std::uint32_t>(modulus_count));
            set_uniform_tensor_metadata(out.T, 1, out.t_logical_slots, 0u, static_cast<std::uint32_t>(modulus_count));
        }
        cleanup();
        return out;
    } catch (...) {
        cleanup();
        release_pooled_pi_tensor_triple(pool, out);
        throw;
    }
}

void execute_pi_merge_shape_batch_pooled(
    PiTensorMemoryPool& pool,
    const std::vector<std::size_t>& node_indices,
    const std::vector<ChudnovskyExactNode>& exact_nodes,
    std::vector<PiTensorTriple>& tensor_nodes,
    std::vector<bool>& tensor_live,
    std::vector<bool>& tensor_owned
) {
    if (node_indices.empty()) {
        return;
    }

    const ChudnovskyExactNode& first_node = exact_nodes[node_indices.front()];
    const PiTensorTriple& first_left = tensor_nodes[first_node.left_child];
    const PiTensorTriple& first_right = tensor_nodes[first_node.right_child];
    const int modulus_count = first_left.P.modulus_count;
    const std::size_t batch_size = node_indices.size();

    const std::size_t p_out_slots = first_left.p_logical_slots + first_right.p_logical_slots - 1;
    const std::size_t q_out_slots = first_left.q_logical_slots + first_right.q_logical_slots - 1;
    const std::size_t t_out_slots = std::max(
        first_left.t_logical_slots + first_right.q_logical_slots - 1,
        first_left.p_logical_slots + first_right.t_logical_slots - 1
    );
    const std::size_t padded_q_slots = t_out_slots + 1 - first_left.T.slot_count;
    const std::size_t padded_t_slots = t_out_slots + 1 - first_left.P.slot_count;

    DeviceRnsTensor left_p_batch;
    DeviceRnsTensor right_p_batch;
    DeviceRnsTensor left_q_batch;
    DeviceRnsTensor right_q_batch;
    DeviceRnsTensor left_t_batch;
    DeviceRnsTensor padded_q_right_batch;
    DeviceRnsTensor padded_t_right_batch;
    DeviceRnsTensor out_p_batch;
    DeviceRnsTensor out_q_batch;
    DeviceRnsTensor t_lq_batch;
    DeviceRnsTensor t_pr_batch;
    DeviceRnsTensor out_t_batch;

    const auto release_temp = [&](DeviceRnsTensor& tensor) {
        if (tensor.d_residues != nullptr) {
            release_pooled_tensor(pool, tensor);
        }
    };
    const auto cleanup = [&]() {
        release_temp(out_t_batch);
        release_temp(t_pr_batch);
        release_temp(t_lq_batch);
        release_temp(out_q_batch);
        release_temp(out_p_batch);
        release_temp(padded_t_right_batch);
        release_temp(padded_q_right_batch);
        release_temp(left_t_batch);
        release_temp(right_q_batch);
        release_temp(left_q_batch);
        release_temp(right_p_batch);
        release_temp(left_p_batch);
    };

    try {
        left_p_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, first_left.P.slot_count, modulus_count);
        right_p_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, first_right.P.slot_count, modulus_count);
        left_q_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, first_left.Q.slot_count, modulus_count);
        right_q_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, first_right.Q.slot_count, modulus_count);
        left_t_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, first_left.T.slot_count, modulus_count);
        padded_q_right_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, padded_q_slots, modulus_count);
        padded_t_right_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, padded_t_slots, modulus_count);
        out_p_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, p_out_slots, modulus_count);
        out_q_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, q_out_slots, modulus_count);
        t_lq_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, t_out_slots, modulus_count);
        t_pr_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, t_out_slots, modulus_count);
        out_t_batch = acquire_pooled_polynomial_block_tensor(pool, batch_size, t_out_slots, modulus_count);

        check_cuda(
            cudaMemset(padded_q_right_batch.d_residues, 0, sizeof(std::uint32_t) * residue_count(padded_q_right_batch)),
            "cudaMemset padded_q_right_batch residues"
        );
        check_cuda(
            cudaMemset(padded_t_right_batch.d_residues, 0, sizeof(std::uint32_t) * residue_count(padded_t_right_batch)),
            "cudaMemset padded_t_right_batch residues"
        );

        for (std::size_t batch_index = 0; batch_index < node_indices.size(); ++batch_index) {
            const ChudnovskyExactNode& node = exact_nodes[node_indices[batch_index]];
            const PiTensorTriple& left = tensor_nodes[node.left_child];
            const PiTensorTriple& right = tensor_nodes[node.right_child];
            copy_single_value_tensor_into_batch(left.P, batch_index, left_p_batch);
            copy_single_value_tensor_into_batch(right.P, batch_index, right_p_batch);
            copy_single_value_tensor_into_batch(left.Q, batch_index, left_q_batch);
            copy_single_value_tensor_into_batch(right.Q, batch_index, right_q_batch);
            copy_single_value_tensor_into_batch(left.T, batch_index, left_t_batch);
            copy_single_value_tensor_prefix_into_batch(right.Q, batch_index, padded_q_right_batch);
            copy_single_value_tensor_prefix_into_batch(right.T, batch_index, padded_t_right_batch);
        }

        convolve_polynomial_blocks_raw(left_p_batch, right_p_batch, out_p_batch);
        convolve_polynomial_blocks_raw(left_q_batch, right_q_batch, out_q_batch);
        convolve_polynomial_blocks_raw(left_t_batch, padded_q_right_batch, t_lq_batch);
        convolve_polynomial_blocks_raw(left_p_batch, padded_t_right_batch, t_pr_batch);
        add_polynomial_blocks_raw(t_lq_batch, t_pr_batch, out_t_batch);

        for (std::size_t batch_index = 0; batch_index < node_indices.size(); ++batch_index) {
            const std::size_t node_index = node_indices[batch_index];
            const ChudnovskyExactNode& node = exact_nodes[node_index];
            const PiTensorTriple& left = tensor_nodes[node.left_child];
            const PiTensorTriple& right = tensor_nodes[node.right_child];

            PiTensorTriple out;
            out.P = acquire_pooled_polynomial_block_tensor(pool, 1, p_out_slots, modulus_count);
            out.Q = acquire_pooled_polynomial_block_tensor(pool, 1, q_out_slots, modulus_count);
            out.T = acquire_pooled_polynomial_block_tensor(pool, 1, t_out_slots, modulus_count);

            try {
                copy_batch_value_into_single_tensor(out_p_batch, batch_index, out.P);
                copy_batch_value_into_single_tensor(out_q_batch, batch_index, out.Q);
                copy_batch_value_into_single_tensor(out_t_batch, batch_index, out.T);
                out.p_logical_slots = static_cast<std::uint32_t>(left.p_logical_slots + right.p_logical_slots - 1);
                out.q_logical_slots = static_cast<std::uint32_t>(left.q_logical_slots + right.q_logical_slots - 1);
                out.t_logical_slots = static_cast<std::uint32_t>(std::max(
                    left.t_logical_slots + right.q_logical_slots - 1,
                    left.p_logical_slots + right.t_logical_slots - 1
                ));
                tensor_nodes[node_index] = out;
                tensor_live[node_index] = true;
                tensor_owned[node_index] = true;
            } catch (...) {
                release_pooled_pi_tensor_triple(pool, out);
                throw;
            }
        }

        for (std::size_t node_index : node_indices) {
            const ChudnovskyExactNode& node = exact_nodes[node_index];
            if (tensor_owned[node.left_child]) {
                release_pooled_pi_tensor_triple(pool, tensor_nodes[node.left_child]);
            }
            if (tensor_owned[node.right_child]) {
                release_pooled_pi_tensor_triple(pool, tensor_nodes[node.right_child]);
            }
            tensor_live[node.left_child] = false;
            tensor_live[node.right_child] = false;
            tensor_owned[node.left_child] = false;
            tensor_owned[node.right_child] = false;
        }

        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

bool validate_pi_tensor_metadata(
    const PiTensorTriple& triple,
    std::uint32_t expected_level,
    bool& metadata_ok
) {
    const HostRnsTensor p_host = download_tensor(triple.P);
    const HostRnsTensor q_host = download_tensor(triple.Q);
    const HostRnsTensor t_host = download_tensor(triple.T);

    const bool ok =
        p_host.modulus_count == triple.P.modulus_count &&
        q_host.modulus_count == triple.Q.modulus_count &&
        t_host.modulus_count == triple.T.modulus_count &&
        validate_metadata_all_equal(p_host.logical_slots, triple.p_logical_slots) &&
        validate_metadata_all_equal(q_host.logical_slots, triple.q_logical_slots) &&
        validate_metadata_all_equal(t_host.logical_slots, triple.t_logical_slots) &&
        validate_metadata_all_equal(p_host.scale_bits, 0u) &&
        validate_metadata_all_equal(q_host.scale_bits, 0u) &&
        validate_metadata_all_equal(t_host.scale_bits, 0u) &&
        validate_metadata_all_equal(p_host.levels, expected_level) &&
        validate_metadata_all_equal(q_host.levels, expected_level) &&
        validate_metadata_all_equal(t_host.levels, expected_level) &&
        validate_signs_equal(p_host.signs, {1}) &&
        validate_signs_equal(q_host.signs, {1}) &&
        validate_signs_equal(t_host.signs, {1});
    metadata_ok = metadata_ok && ok;
    return ok;
}

}  // namespace

PiPqtTreeReport plan_pi_pqt_tree(std::size_t term_count) {
    if (term_count == 0) {
        throw std::invalid_argument("pi P/Q/T smoke requires term_count > 0");
    }
    if (term_count > 64) {
        throw std::invalid_argument("pi P/Q/T smoke currently limits term_count to 64 for exact host-side validation");
    }

    const auto ceil_div_size = [](std::size_t numerator, std::size_t denominator) {
        if (denominator == 0) {
            throw std::invalid_argument("ceil_div_size requires denominator > 0");
        }
        return numerator / denominator + (numerator % denominator == 0 ? 0 : 1);
    };

    std::vector<ChudnovskyExactNode> exact_nodes;
    exact_nodes.reserve(2 * term_count - 1);
    const std::size_t root_index = build_chudnovsky_exact_tree(0, term_count, 0, exact_nodes);

    PiPqtTreeReport report;
    report.term_count = term_count;
    report.node_count = exact_nodes.size();
    report.leaf_count = term_count;
    report.merge_count = term_count > 0 ? term_count - 1 : 0;
    report.chosen_limb_bits = 32;
    report.recurrence = "leaf(k>0): P=(6k-5)(2k-1)(6k-1), Q=k^3*C3/24, T=P*(A+Bk)*(-1)^k; leaf(0): P=Q=1,T=A";
    report.merge_formula = "P=P_left*P_right, Q=Q_left*Q_right, T=T_left*Q_right + P_left*T_right";

    for (const ChudnovskyExactNode& node : exact_nodes) {
        report.max_depth = std::max(report.max_depth, node.depth);
    }
    report.depth_node_counts.assign(report.max_depth + 1, 0);
    report.nodes.reserve(exact_nodes.size());

    bool ok = true;
    for (std::size_t node_index = 0; node_index < exact_nodes.size(); ++node_index) {
        const ChudnovskyExactNode& exact = exact_nodes[node_index];
        ++report.depth_node_counts[exact.depth];

        const std::size_t p_bits = bit_width_cpp_int(exact.P);
        const std::size_t q_bits = bit_width_cpp_int(exact.Q);
        const std::size_t t_bits = bit_width_cpp_int(exact.T);
        const std::size_t slot_count = ceil_div_size(
            std::max({p_bits, q_bits, t_bits}),
            report.chosen_limb_bits
        );
        const std::size_t ntt_size = resolve_convolution_ntt_size(2 * slot_count - 1);

        report.nodes.push_back(PiPqtNodeReport{
            .node_index = node_index,
            .begin_term = exact.begin_term,
            .end_term = exact.end_term,
            .depth = exact.depth,
            .left_child = exact.left_child,
            .right_child = exact.right_child,
            .is_leaf = exact.is_leaf,
            .t_negative = exact.T < 0,
            .p_bits = p_bits,
            .q_bits = q_bits,
            .t_bits = t_bits,
            .slot_count = slot_count,
            .ntt_size = ntt_size,
        });

        ok = ok && exact.begin_term < exact.end_term;
        ok = ok && slot_count > 0;
        ok = ok && ntt_size > 0;
        if (exact.is_leaf) {
            ok = ok && exact.end_term - exact.begin_term == 1;
        } else {
            ok = ok && exact.left_child < node_index;
            ok = ok && exact.right_child < node_index;
            ok = ok && exact_nodes[exact.left_child].end_term == exact_nodes[exact.right_child].begin_term;
        }
    }

    const PiPqtNodeReport& root = report.nodes[root_index];
    report.root_p_bits = root.p_bits;
    report.root_q_bits = root.q_bits;
    report.root_t_bits = root.t_bits;
    report.root_slot_count = root.slot_count;
    report.root_ntt_size = root.ntt_size;
    report.root_t_negative = root.t_negative;

    ok = ok && root.begin_term == 0;
    ok = ok && root.end_term == term_count;
    ok = ok && report.node_count == 2 * report.leaf_count - 1;
    ok = ok && report.depth_node_counts.front() == 1;
    ok = ok && report.depth_node_counts.back() > 0;
    report.ok = ok;
    return report;
}

void print_pi_pqt_tree_report(std::ostream& out, const PiPqtTreeReport& report) {
    out << "pi_pqt_smoke_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "term_count=" << report.term_count << '\n';
    out << "node_count=" << report.node_count << '\n';
    out << "leaf_count=" << report.leaf_count << '\n';
    out << "merge_count=" << report.merge_count << '\n';
    out << "max_depth=" << report.max_depth << '\n';
    out << "chosen_limb_bits=" << report.chosen_limb_bits << '\n';
    out << "root_p_bits=" << report.root_p_bits << '\n';
    out << "root_q_bits=" << report.root_q_bits << '\n';
    out << "root_t_bits=" << report.root_t_bits << '\n';
    out << "root_slot_count=" << report.root_slot_count << '\n';
    out << "root_ntt_size=" << report.root_ntt_size << '\n';
    out << "root_t_negative=" << (report.root_t_negative ? 1 : 0) << '\n';
    out << "recurrence=" << report.recurrence << '\n';
    out << "merge_formula=" << report.merge_formula << '\n';
    for (std::size_t depth = 0; depth < report.depth_node_counts.size(); ++depth) {
        out << "depth_" << depth << "_node_count=" << report.depth_node_counts[depth] << '\n';
    }
    if (!report.nodes.empty()) {
        const PiPqtNodeReport* leftmost_leaf = nullptr;
        const PiPqtNodeReport* rightmost_leaf = nullptr;
        for (const PiPqtNodeReport& node : report.nodes) {
            if (!node.is_leaf) {
                continue;
            }
            if (leftmost_leaf == nullptr || node.begin_term < leftmost_leaf->begin_term) {
                leftmost_leaf = &node;
            }
            if (rightmost_leaf == nullptr || node.end_term > rightmost_leaf->end_term) {
                rightmost_leaf = &node;
            }
        }
        if (leftmost_leaf != nullptr && rightmost_leaf != nullptr) {
            out << "leftmost_leaf_interval=" << leftmost_leaf->begin_term << ":" << leftmost_leaf->end_term << '\n';
            out << "rightmost_leaf_interval=" << rightmost_leaf->begin_term << ":" << rightmost_leaf->end_term << '\n';
        }
    }
}

bool pi_pqt_smoke_test(std::ostream& out, std::size_t term_count) {
    const PiPqtTreeReport report = plan_pi_pqt_tree(term_count);
    print_pi_pqt_tree_report(out, report);
    return report.ok;
}

PiPqtTensorReport run_pi_pqt_tensor_smoke(std::size_t term_count) {
    if (term_count < 2) {
        throw std::invalid_argument("pi P/Q/T tensor smoke requires term_count >= 2");
    }
    if (term_count > 64) {
        throw std::invalid_argument("pi P/Q/T tensor smoke currently limits term_count to 64");
    }

    std::vector<ChudnovskyExactNode> exact_nodes;
    exact_nodes.reserve(2 * term_count - 1);
    build_chudnovsky_exact_tree(0, term_count, 0, exact_nodes);
    const std::size_t merge_node_index = find_first_leaf_merge_node(exact_nodes);
    const ChudnovskyExactNode& merge = exact_nodes[merge_node_index];
    const ChudnovskyExactNode& left = exact_nodes[merge.left_child];
    const ChudnovskyExactNode& right = exact_nodes[merge.right_child];

    const std::vector<long long> p_left_coeffs = bigint_to_signed_base_coefficients(left.P, kPiTensorLimbBits);
    const std::vector<long long> p_right_coeffs = bigint_to_signed_base_coefficients(right.P, kPiTensorLimbBits);
    const std::vector<long long> q_left_coeffs = bigint_to_signed_base_coefficients(left.Q, kPiTensorLimbBits);
    const std::vector<long long> q_right_coeffs = bigint_to_signed_base_coefficients(right.Q, kPiTensorLimbBits);
    const std::vector<long long> t_left_coeffs = bigint_to_signed_base_coefficients(left.T, kPiTensorLimbBits);
    const std::vector<long long> t_right_coeffs = bigint_to_signed_base_coefficients(right.T, kPiTensorLimbBits);
    const std::size_t t_lq_logical_slots = t_left_coeffs.size() + q_right_coeffs.size() - 1;
    const std::size_t t_pr_logical_slots = p_left_coeffs.size() + t_right_coeffs.size() - 1;
    const int tensor_modulus_count = choose_pi_tensor_modulus_count(term_count);

    DeviceRnsTensor p_left_tensor;
    DeviceRnsTensor p_right_tensor;
    DeviceRnsTensor q_left_tensor;
    DeviceRnsTensor q_right_tensor;
    DeviceRnsTensor q_right_t_tensor;
    DeviceRnsTensor t_left_tensor;
    DeviceRnsTensor t_right_tensor;
    DeviceRnsTensor t_right_t_tensor;
    DeviceRnsTensor p_out_tensor;
    DeviceRnsTensor q_out_tensor;
    DeviceRnsTensor t_lq_tensor;
    DeviceRnsTensor t_pr_tensor;
    DeviceRnsTensor t_out_tensor;

    const auto cleanup = [&]() {
        if (t_out_tensor.d_moduli != nullptr) {
            free_device_tensor(t_out_tensor);
        }
        if (t_pr_tensor.d_moduli != nullptr) {
            free_device_tensor(t_pr_tensor);
        }
        if (t_lq_tensor.d_moduli != nullptr) {
            free_device_tensor(t_lq_tensor);
        }
        if (q_out_tensor.d_moduli != nullptr) {
            free_device_tensor(q_out_tensor);
        }
        if (p_out_tensor.d_moduli != nullptr) {
            free_device_tensor(p_out_tensor);
        }
        if (t_right_tensor.d_moduli != nullptr) {
            free_device_tensor(t_right_tensor);
        }
        if (t_left_tensor.d_moduli != nullptr) {
            free_device_tensor(t_left_tensor);
        }
        if (t_right_t_tensor.d_moduli != nullptr) {
            free_device_tensor(t_right_t_tensor);
        }
        if (q_right_t_tensor.d_moduli != nullptr) {
            free_device_tensor(q_right_t_tensor);
        }
        if (q_right_tensor.d_moduli != nullptr) {
            free_device_tensor(q_right_tensor);
        }
        if (q_left_tensor.d_moduli != nullptr) {
            free_device_tensor(q_left_tensor);
        }
        if (p_right_tensor.d_moduli != nullptr) {
            free_device_tensor(p_right_tensor);
        }
        if (p_left_tensor.d_moduli != nullptr) {
            free_device_tensor(p_left_tensor);
        }
    };

    PiPqtTensorReport report;
    report.term_count = term_count;
    report.merge_begin_term = merge.begin_term;
    report.merge_end_term = merge.end_term;
    report.left_begin_term = left.begin_term;
    report.left_end_term = left.end_term;
    report.right_begin_term = right.begin_term;
    report.right_end_term = right.end_term;
    report.chosen_limb_bits = kPiTensorLimbBits;
    report.p_left_slot_count = p_left_coeffs.size();
    report.p_right_slot_count = p_right_coeffs.size();
    report.q_left_slot_count = q_left_coeffs.size();
    report.q_right_slot_count = q_right_coeffs.size();
    report.t_left_slot_count = t_left_coeffs.size();
    report.t_right_slot_count = t_right_coeffs.size();
    report.p_output_slot_count = p_left_coeffs.size() + p_right_coeffs.size() - 1;
    report.q_output_slot_count = q_left_coeffs.size() + q_right_coeffs.size() - 1;
    report.t_output_slot_count = std::max(t_lq_logical_slots, t_pr_logical_slots);
    report.p_output_ntt_size = resolve_convolution_ntt_size(report.p_output_slot_count);
    report.q_output_ntt_size = resolve_convolution_ntt_size(report.q_output_slot_count);
    report.t_output_ntt_size = resolve_convolution_ntt_size(report.t_output_slot_count);
    report.modulus_count = static_cast<std::size_t>(tensor_modulus_count);
    report.uses_signed_residue_encoding = true;

    const std::vector<long long> q_right_t_coeffs = [&]() {
        std::vector<long long> padded(report.t_output_slot_count + 1 - t_left_coeffs.size(), 0);
        std::copy(q_right_coeffs.begin(), q_right_coeffs.end(), padded.begin());
        return padded;
    }();
    const std::vector<long long> t_right_t_coeffs = [&]() {
        std::vector<long long> padded(report.t_output_slot_count + 1 - p_left_coeffs.size(), 0);
        std::copy(t_right_coeffs.begin(), t_right_coeffs.end(), padded.begin());
        return padded;
    }();

    try {
        p_left_tensor = allocate_polynomial_block_tensor(1, p_left_coeffs.size(), tensor_modulus_count);
        p_right_tensor = allocate_polynomial_block_tensor(1, p_right_coeffs.size(), tensor_modulus_count);
        q_left_tensor = allocate_polynomial_block_tensor(1, q_left_coeffs.size(), tensor_modulus_count);
        q_right_tensor = allocate_polynomial_block_tensor(1, q_right_coeffs.size(), tensor_modulus_count);
        q_right_t_tensor = allocate_polynomial_block_tensor(
            1,
            report.t_output_slot_count + 1 - t_left_coeffs.size(),
            tensor_modulus_count
        );
        t_left_tensor = allocate_polynomial_block_tensor(1, t_left_coeffs.size(), tensor_modulus_count);
        t_right_tensor = allocate_polynomial_block_tensor(1, t_right_coeffs.size(), tensor_modulus_count);
        t_right_t_tensor = allocate_polynomial_block_tensor(
            1,
            report.t_output_slot_count + 1 - p_left_coeffs.size(),
            tensor_modulus_count
        );
        p_out_tensor = allocate_polynomial_block_tensor(1, report.p_output_slot_count, tensor_modulus_count);
        q_out_tensor = allocate_polynomial_block_tensor(1, report.q_output_slot_count, tensor_modulus_count);
        t_lq_tensor = allocate_polynomial_block_tensor(1, report.t_output_slot_count, tensor_modulus_count);
        t_pr_tensor = allocate_polynomial_block_tensor(1, report.t_output_slot_count, tensor_modulus_count);
        t_out_tensor = allocate_polynomial_block_tensor(1, report.t_output_slot_count, tensor_modulus_count);

        encode_signed_polynomial_blocks(
            p_left_tensor,
            p_left_coeffs,
            static_cast<std::uint32_t>(p_left_coeffs.size())
        );
        encode_signed_polynomial_blocks(
            p_right_tensor,
            p_right_coeffs,
            static_cast<std::uint32_t>(p_right_coeffs.size())
        );
        encode_signed_polynomial_blocks(
            q_left_tensor,
            q_left_coeffs,
            static_cast<std::uint32_t>(q_left_coeffs.size())
        );
        encode_signed_polynomial_blocks(
            q_right_tensor,
            q_right_coeffs,
            static_cast<std::uint32_t>(q_right_coeffs.size())
        );
        encode_signed_polynomial_blocks(
            q_right_t_tensor,
            q_right_t_coeffs,
            static_cast<std::uint32_t>(q_right_coeffs.size())
        );
        encode_signed_polynomial_blocks(
            t_left_tensor,
            t_left_coeffs,
            static_cast<std::uint32_t>(t_left_coeffs.size())
        );
        encode_signed_polynomial_blocks(
            t_right_tensor,
            t_right_coeffs,
            static_cast<std::uint32_t>(t_right_coeffs.size())
        );
        encode_signed_polynomial_blocks(
            t_right_t_tensor,
            t_right_t_coeffs,
            static_cast<std::uint32_t>(t_right_coeffs.size())
        );

        convolve_polynomial_blocks(p_left_tensor, p_right_tensor, p_out_tensor);
        convolve_polynomial_blocks(q_left_tensor, q_right_tensor, q_out_tensor);
        convolve_polynomial_blocks(t_left_tensor, q_right_t_tensor, t_lq_tensor);
        convolve_polynomial_blocks(p_left_tensor, t_right_t_tensor, t_pr_tensor);
        add_polynomial_blocks(t_lq_tensor, t_pr_tensor, t_out_tensor);

        const HostRnsTensor p_out_host = download_tensor(p_out_tensor);
        const HostRnsTensor q_out_host = download_tensor(q_out_tensor);
        const HostRnsTensor t_lq_host = download_tensor(t_lq_tensor);
        const HostRnsTensor t_pr_host = download_tensor(t_pr_tensor);
        const HostRnsTensor t_out_host = download_tensor(t_out_tensor);

        const std::vector<HostBigInt> p_out_coeffs = reconstruct_centered_coefficients(p_out_host);
        const std::vector<HostBigInt> q_out_coeffs = reconstruct_centered_coefficients(q_out_host);
        const std::vector<HostBigInt> t_lq_coeffs = reconstruct_centered_coefficients(t_lq_host);
        const std::vector<HostBigInt> t_pr_coeffs = reconstruct_centered_coefficients(t_pr_host);
        const std::vector<HostBigInt> t_out_coeffs = reconstruct_centered_coefficients(t_out_host);

        const HostBigInt p_out_value = rebuild_bigint_from_centered_coefficients(p_out_coeffs, kPiTensorLimbBits);
        const HostBigInt q_out_value = rebuild_bigint_from_centered_coefficients(q_out_coeffs, kPiTensorLimbBits);
        const HostBigInt t_lq_value = rebuild_bigint_from_centered_coefficients(t_lq_coeffs, kPiTensorLimbBits);
        const HostBigInt t_pr_value = rebuild_bigint_from_centered_coefficients(t_pr_coeffs, kPiTensorLimbBits);
        const HostBigInt t_out_value = rebuild_bigint_from_centered_coefficients(t_out_coeffs, kPiTensorLimbBits);

        const HostBigInt expected_t_lq = left.T * right.Q;
        const HostBigInt expected_t_pr = left.P * right.T;

        report.metadata_ok =
            p_out_host.modulus_count == tensor_modulus_count &&
            q_out_host.modulus_count == tensor_modulus_count &&
            t_lq_host.modulus_count == tensor_modulus_count &&
            t_pr_host.modulus_count == tensor_modulus_count &&
            t_out_host.modulus_count == tensor_modulus_count &&
            validate_metadata_all_equal(p_out_host.logical_slots, static_cast<std::uint32_t>(report.p_output_slot_count)) &&
            validate_metadata_all_equal(q_out_host.logical_slots, static_cast<std::uint32_t>(report.q_output_slot_count)) &&
            validate_metadata_all_equal(t_lq_host.logical_slots, static_cast<std::uint32_t>(t_lq_logical_slots)) &&
            validate_metadata_all_equal(t_pr_host.logical_slots, static_cast<std::uint32_t>(t_pr_logical_slots)) &&
            validate_metadata_all_equal(t_out_host.logical_slots, static_cast<std::uint32_t>(report.t_output_slot_count)) &&
            validate_metadata_all_equal(p_out_host.scale_bits, 0u) &&
            validate_metadata_all_equal(q_out_host.scale_bits, 0u) &&
            validate_metadata_all_equal(t_lq_host.scale_bits, 0u) &&
            validate_metadata_all_equal(t_pr_host.scale_bits, 0u) &&
            validate_metadata_all_equal(t_out_host.scale_bits, 0u) &&
            validate_metadata_all_equal(
                p_out_host.levels,
                static_cast<std::uint32_t>(report.modulus_count)
            ) &&
            validate_metadata_all_equal(
                q_out_host.levels,
                static_cast<std::uint32_t>(report.modulus_count)
            ) &&
            validate_metadata_all_equal(
                t_lq_host.levels,
                static_cast<std::uint32_t>(report.modulus_count)
            ) &&
            validate_metadata_all_equal(
                t_pr_host.levels,
                static_cast<std::uint32_t>(report.modulus_count)
            ) &&
            validate_metadata_all_equal(
                t_out_host.levels,
                static_cast<std::uint32_t>(report.modulus_count)
            ) &&
            validate_signs_equal(p_out_host.signs, {1}) &&
            validate_signs_equal(q_out_host.signs, {1}) &&
            validate_signs_equal(t_lq_host.signs, {1}) &&
            validate_signs_equal(t_pr_host.signs, {1}) &&
            validate_signs_equal(t_out_host.signs, {1});

        report.p_match = p_out_value == merge.P;
        report.q_match = q_out_value == merge.Q;
        report.t_match = t_lq_value == expected_t_lq && t_pr_value == expected_t_pr && t_out_value == merge.T;
        report.ok = report.metadata_ok && report.p_match && report.q_match && report.t_match;
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    return report;
}

void print_pi_pqt_tensor_report(std::ostream& out, const PiPqtTensorReport& report) {
    out << "pi_pqt_tensor_smoke_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "term_count=" << report.term_count << '\n';
    out << "merge_interval=" << report.merge_begin_term << ":" << report.merge_end_term << '\n';
    out << "left_interval=" << report.left_begin_term << ":" << report.left_end_term << '\n';
    out << "right_interval=" << report.right_begin_term << ":" << report.right_end_term << '\n';
    out << "chosen_limb_bits=" << report.chosen_limb_bits << '\n';
    out << "uses_signed_residue_encoding=" << (report.uses_signed_residue_encoding ? 1 : 0) << '\n';
    out << "p_left_slot_count=" << report.p_left_slot_count << '\n';
    out << "p_right_slot_count=" << report.p_right_slot_count << '\n';
    out << "q_left_slot_count=" << report.q_left_slot_count << '\n';
    out << "q_right_slot_count=" << report.q_right_slot_count << '\n';
    out << "t_left_slot_count=" << report.t_left_slot_count << '\n';
    out << "t_right_slot_count=" << report.t_right_slot_count << '\n';
    out << "p_output_slot_count=" << report.p_output_slot_count << '\n';
    out << "q_output_slot_count=" << report.q_output_slot_count << '\n';
    out << "t_output_slot_count=" << report.t_output_slot_count << '\n';
    out << "p_output_ntt_size=" << report.p_output_ntt_size << '\n';
    out << "q_output_ntt_size=" << report.q_output_ntt_size << '\n';
    out << "t_output_ntt_size=" << report.t_output_ntt_size << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "metadata_ok=" << (report.metadata_ok ? 1 : 0) << '\n';
    out << "p_match=" << (report.p_match ? 1 : 0) << '\n';
    out << "q_match=" << (report.q_match ? 1 : 0) << '\n';
    out << "t_match=" << (report.t_match ? 1 : 0) << '\n';
}

bool pi_pqt_tensor_smoke_test(std::ostream& out, std::size_t term_count) {
    const PiPqtTensorReport report = run_pi_pqt_tensor_smoke(term_count);
    print_pi_pqt_tensor_report(out, report);
    return report.ok;
}

PiTensorTreeExecutionResult run_pi_pqt_tensor_tree_execution(
    std::size_t term_count,
    const PiTensorTreeExecutionOptions& options
) {
    if (term_count < 2) {
        throw std::invalid_argument("pi P/Q/T tensor tree smoke requires term_count >= 2");
    }

    PiTensorTreeExecutionWorkspace& workspace = prepare_pi_tensor_tree_execution_workspace(term_count);
    ensure_pi_tensor_tree_leaf_cache(workspace);
    const std::vector<ChudnovskyExactNode>& exact_nodes = workspace.exact_nodes;
    std::vector<PiTensorTriple>& tensor_nodes = workspace.tensor_nodes;
    std::vector<bool>& tensor_live = workspace.tensor_live;
    std::vector<bool>& tensor_owned = workspace.tensor_owned;
    PiTensorMemoryPool& tensor_pool = workspace.pool;
    const std::size_t root_index = workspace.root_index;
    const std::size_t chosen_limb_bits = workspace.chosen_limb_bits;
    const int tensor_modulus_count = workspace.modulus_count;
    const auto cleanup_all = [&]() {
        release_workspace_live_tensors(workspace);
    };

    PiTensorTreeExecutionResult execution;
    PiPqtTensorTreeReport& report = execution.report;
    report.term_count = term_count;
    report.node_count = exact_nodes.size();
    report.leaf_count = term_count;
    report.merge_count = term_count - 1;
    report.root_begin_term = exact_nodes[root_index].begin_term;
    report.root_end_term = exact_nodes[root_index].end_term;
    report.chosen_limb_bits = chosen_limb_bits;
    report.modulus_count = static_cast<std::size_t>(tensor_modulus_count);
    report.uses_signed_residue_encoding = true;

    bool metadata_ok = true;
    try {
        for (std::size_t depth_offset = 0; depth_offset < workspace.nodes_by_depth.size(); ++depth_offset) {
            const std::size_t depth = workspace.max_depth - depth_offset;
            std::vector<std::size_t> internal_nodes;
            for (std::size_t node_index : workspace.nodes_by_depth[depth]) {
                const ChudnovskyExactNode& node = exact_nodes[node_index];
                if (node.is_leaf) {
                    tensor_nodes[node_index] = workspace.leaf_cache[node_index];
                    tensor_live[node_index] = true;
                    tensor_owned[node_index] = false;

                    report.peak_p_slot_count = std::max<std::size_t>(
                        report.peak_p_slot_count,
                        tensor_nodes[node_index].p_logical_slots
                    );
                    report.peak_q_slot_count = std::max<std::size_t>(
                        report.peak_q_slot_count,
                        tensor_nodes[node_index].q_logical_slots
                    );
                    report.peak_t_slot_count = std::max<std::size_t>(
                        report.peak_t_slot_count,
                        tensor_nodes[node_index].t_logical_slots
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].p_logical_slots)
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].q_logical_slots)
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].t_logical_slots)
                    );
                    continue;
                }

                if (options.validate_all_metadata) {
                    tensor_nodes[node_index] = merge_pi_tensor_triples_pooled(
                        tensor_pool,
                        tensor_nodes[node.left_child],
                        tensor_nodes[node.right_child],
                        true
                    );
                    tensor_live[node_index] = true;
                    tensor_owned[node_index] = true;
                    if (tensor_owned[node.left_child]) {
                        release_pooled_pi_tensor_triple(tensor_pool, tensor_nodes[node.left_child]);
                    }
                    if (tensor_owned[node.right_child]) {
                        release_pooled_pi_tensor_triple(tensor_pool, tensor_nodes[node.right_child]);
                    }
                    tensor_live[node.left_child] = false;
                    tensor_live[node.right_child] = false;
                    tensor_owned[node.left_child] = false;
                    tensor_owned[node.right_child] = false;

                    validate_pi_tensor_metadata(
                        tensor_nodes[node_index],
                        static_cast<std::uint32_t>(report.modulus_count),
                        metadata_ok
                    );

                    report.peak_p_slot_count = std::max<std::size_t>(
                        report.peak_p_slot_count,
                        tensor_nodes[node_index].p_logical_slots
                    );
                    report.peak_q_slot_count = std::max<std::size_t>(
                        report.peak_q_slot_count,
                        tensor_nodes[node_index].q_logical_slots
                    );
                    report.peak_t_slot_count = std::max<std::size_t>(
                        report.peak_t_slot_count,
                        tensor_nodes[node_index].t_logical_slots
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].p_logical_slots)
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].q_logical_slots)
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].t_logical_slots)
                    );
                } else {
                    internal_nodes.push_back(node_index);
                }
            }

            if (!options.validate_all_metadata && !internal_nodes.empty()) {
                const std::vector<PiMergeShapeGroup> groups =
                    build_pi_merge_shape_groups(internal_nodes, exact_nodes, tensor_nodes);
                for (const PiMergeShapeGroup& group : groups) {
                    if (group.node_indices.size() == 1) {
                        const std::size_t node_index = group.node_indices.front();
                        const ChudnovskyExactNode& node = exact_nodes[node_index];
                        tensor_nodes[node_index] = merge_pi_tensor_triples_pooled(
                            tensor_pool,
                            tensor_nodes[node.left_child],
                            tensor_nodes[node.right_child],
                            false
                        );
                        tensor_live[node_index] = true;
                        tensor_owned[node_index] = true;
                        if (tensor_owned[node.left_child]) {
                            release_pooled_pi_tensor_triple(tensor_pool, tensor_nodes[node.left_child]);
                        }
                        if (tensor_owned[node.right_child]) {
                            release_pooled_pi_tensor_triple(tensor_pool, tensor_nodes[node.right_child]);
                        }
                        tensor_live[node.left_child] = false;
                        tensor_live[node.right_child] = false;
                        tensor_owned[node.left_child] = false;
                        tensor_owned[node.right_child] = false;
                    } else {
                        execute_pi_merge_shape_batch_pooled(
                            tensor_pool,
                            group.node_indices,
                            exact_nodes,
                            tensor_nodes,
                            tensor_live,
                            tensor_owned
                        );
                    }
                }

                for (std::size_t node_index : internal_nodes) {
                    report.peak_p_slot_count = std::max<std::size_t>(
                        report.peak_p_slot_count,
                        tensor_nodes[node_index].p_logical_slots
                    );
                    report.peak_q_slot_count = std::max<std::size_t>(
                        report.peak_q_slot_count,
                        tensor_nodes[node_index].q_logical_slots
                    );
                    report.peak_t_slot_count = std::max<std::size_t>(
                        report.peak_t_slot_count,
                        tensor_nodes[node_index].t_logical_slots
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].p_logical_slots)
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].q_logical_slots)
                    );
                    report.peak_ntt_size = std::max<std::size_t>(
                        report.peak_ntt_size,
                        resolve_convolution_ntt_size(tensor_nodes[node_index].t_logical_slots)
                    );
                }
            }
        }

        const PiTensorTriple& root = tensor_nodes[root_index];
        report.root_p_slot_count = root.p_logical_slots;
        report.root_q_slot_count = root.q_logical_slots;
        report.root_t_slot_count = root.t_logical_slots;

        const HostRnsTensor root_p_host = download_tensor(root.P);
        const HostRnsTensor root_q_host = download_tensor(root.Q);
        const HostRnsTensor root_t_host = download_tensor(root.T);

        const HostBigInt root_p_value = rebuild_bigint_from_centered_coefficients(
            reconstruct_centered_coefficients(root_p_host),
            chosen_limb_bits
        );
        const HostBigInt root_q_value = rebuild_bigint_from_centered_coefficients(
            reconstruct_centered_coefficients(root_q_host),
            chosen_limb_bits
        );
        const HostBigInt root_t_value = rebuild_bigint_from_centered_coefficients(
            reconstruct_centered_coefficients(root_t_host),
            chosen_limb_bits
        );

        const ChudnovskyExactNode& exact_root = exact_nodes[root_index];
        execution.root_P = root_p_value;
        execution.root_Q = root_q_value;
        execution.root_T = root_t_value;
        report.metadata_ok = options.validate_all_metadata ? metadata_ok : true;
        report.p_match = execution.root_P == exact_root.P;
        report.q_match = execution.root_Q == exact_root.Q;
        report.t_match = execution.root_T == exact_root.T;
        report.ok = report.metadata_ok && report.p_match && report.q_match && report.t_match;
        cleanup_all();
    } catch (...) {
        cleanup_all();
        throw;
    }
    return execution;
}

PiTensorTreeExecutionResult run_pi_pqt_tensor_tree_execution(std::size_t term_count) {
    return run_pi_pqt_tensor_tree_execution(term_count, PiTensorTreeExecutionOptions{});
}

PiPqtTensorTreeReport run_pi_pqt_tensor_tree_smoke(std::size_t term_count) {
    return run_pi_pqt_tensor_tree_execution(term_count).report;
}

void print_pi_pqt_tensor_tree_report(std::ostream& out, const PiPqtTensorTreeReport& report) {
    out << "pi_pqt_tensor_tree_smoke_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "term_count=" << report.term_count << '\n';
    out << "node_count=" << report.node_count << '\n';
    out << "leaf_count=" << report.leaf_count << '\n';
    out << "merge_count=" << report.merge_count << '\n';
    out << "root_interval=" << report.root_begin_term << ":" << report.root_end_term << '\n';
    out << "chosen_limb_bits=" << report.chosen_limb_bits << '\n';
    out << "uses_signed_residue_encoding=" << (report.uses_signed_residue_encoding ? 1 : 0) << '\n';
    out << "peak_p_slot_count=" << report.peak_p_slot_count << '\n';
    out << "peak_q_slot_count=" << report.peak_q_slot_count << '\n';
    out << "peak_t_slot_count=" << report.peak_t_slot_count << '\n';
    out << "peak_ntt_size=" << report.peak_ntt_size << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "root_p_slot_count=" << report.root_p_slot_count << '\n';
    out << "root_q_slot_count=" << report.root_q_slot_count << '\n';
    out << "root_t_slot_count=" << report.root_t_slot_count << '\n';
    out << "metadata_ok=" << (report.metadata_ok ? 1 : 0) << '\n';
    out << "p_match=" << (report.p_match ? 1 : 0) << '\n';
    out << "q_match=" << (report.q_match ? 1 : 0) << '\n';
    out << "t_match=" << (report.t_match ? 1 : 0) << '\n';
}

bool pi_pqt_tensor_tree_smoke_test(std::ostream& out, std::size_t term_count) {
    const PiPqtTensorTreeReport report = run_pi_pqt_tensor_tree_smoke(term_count);
    print_pi_pqt_tensor_tree_report(out, report);
    return report.ok;
}

PiEndToEndReport finalize_pi_end_to_end_report(
    std::size_t term_count,
    std::size_t target_digits,
    const PiRouteReport& route,
    const PiTensorTreeExecutionResult& execution
) {
    PiEndToEndReport report;
    report.term_count = term_count;
    report.target_digits = target_digits;
    report.working_digits = route.working_digits;
    report.required_terms = route.chudnovsky_terms;
    report.reference_prefix_digits_checked =
        std::min<std::size_t>(target_digits, pi_reference_digits().size() - 1u);
    report.term_count_sufficient = term_count >= report.required_terms;
    report.chosen_limb_bits = execution.report.chosen_limb_bits;
    report.modulus_count = execution.report.modulus_count;
    report.peak_ntt_size = execution.report.peak_ntt_size;
    report.root_q_bits = bit_width_cpp_int(execution.root_Q);
    report.root_t_bits = bit_width_cpp_int(execution.root_T);
    report.closure_mode = "host_exact_bigint_isqrt_division_after_gpu_tensor_tree";

    if (!execution.report.ok || !report.term_count_sufficient) {
        report.ok = false;
        return report;
    }

    const HostBigInt q = execution.root_Q;
    const HostBigInt t = execution.root_T;
    if (t < 0 || t.is_zero()) {
        throw std::runtime_error("pi end-to-end smoke requires a positive root T");
    }

    const HostBigInt scale = pow10_host_bigint(target_digits);
    const HostBigInt sqrt_scaled = integer_sqrt_host_bigint(HostBigInt{10005} * scale * scale);
    const HostBigInt pi_numerator = HostBigInt{426880} * sqrt_scaled * q;
    const auto [pi_scaled, pi_remainder] = div_mod_abs(pi_numerator, t);
    const std::string scaled_digits = host_bigint_to_decimal_string(pi_scaled);
    const std::string& reference = pi_reference_digits();

    report.pi_decimal = format_scaled_decimal(pi_scaled, target_digits);
    const std::size_t checked_digit_count = report.reference_prefix_digits_checked + 1u;
    report.prefix_match =
        scaled_digits.size() >= checked_digit_count &&
        scaled_digits.compare(0, checked_digit_count, reference, 0, checked_digit_count) == 0 &&
        abs_compare(pi_remainder, t) < 0;
    report.ok = execution.report.ok && report.term_count_sufficient && report.prefix_match;
    return report;
}

PiEndToEndReport run_pi_end_to_end_smoke(std::size_t term_count, std::size_t target_digits) {
    if (target_digits == 0) {
        throw std::invalid_argument("pi end-to-end smoke requires target_digits > 0");
    }
    if (target_digits + 1 > pi_reference_digits().size()) {
        throw std::invalid_argument("pi end-to-end smoke currently limits target_digits to the embedded reference prefix");
    }

    const PiRouteReport route = plan_pi_route(target_digits);
    const PiTensorTreeExecutionResult execution = run_pi_pqt_tensor_tree_execution(term_count);
    return finalize_pi_end_to_end_report(term_count, target_digits, route, execution);
}

void print_pi_end_to_end_report(std::ostream& out, const PiEndToEndReport& report) {
    out << "pi_end_to_end_smoke_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "term_count=" << report.term_count << '\n';
    out << "target_digits=" << report.target_digits << '\n';
    out << "working_digits=" << report.working_digits << '\n';
    out << "required_terms=" << report.required_terms << '\n';
    out << "reference_prefix_digits_checked=" << report.reference_prefix_digits_checked << '\n';
    out << "term_count_sufficient=" << (report.term_count_sufficient ? 1 : 0) << '\n';
    out << "chosen_limb_bits=" << report.chosen_limb_bits << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "peak_ntt_size=" << report.peak_ntt_size << '\n';
    out << "root_q_bits=" << report.root_q_bits << '\n';
    out << "root_t_bits=" << report.root_t_bits << '\n';
    out << "closure_mode=" << report.closure_mode << '\n';
    out << "pi_decimal=" << report.pi_decimal << '\n';
    out << "prefix_match=" << (report.prefix_match ? 1 : 0) << '\n';
}

bool pi_end_to_end_smoke_test(std::ostream& out, std::size_t term_count, std::size_t target_digits) {
    const PiEndToEndReport report = run_pi_end_to_end_smoke(term_count, target_digits);
    print_pi_end_to_end_report(out, report);
    return report.ok;
}

PiBenchmarkReport run_pi_end_to_end_benchmark(
    std::size_t term_count,
    std::size_t target_digits,
    int measured_iterations
) {
    if (target_digits == 0) {
        throw std::invalid_argument("pi end-to-end benchmark requires target_digits > 0");
    }
    if (measured_iterations <= 0) {
        throw std::invalid_argument("pi end-to-end benchmark requires measured_iterations > 0");
    }

    const PiRouteReport route = plan_pi_route(target_digits);
    PiBenchmarkReport report;
    report.term_count = term_count;
    report.target_digits = target_digits;
    report.working_digits = route.working_digits;
    report.required_terms = route.chudnovsky_terms;
    report.measured_iterations = measured_iterations;
    report.tree_validation_mode = "skip_intermediate_metadata_downloads_keep_root_exact_match";

    const auto cold_start = std::chrono::steady_clock::now();
    const auto cold_tree_start = cold_start;
    const PiTensorTreeExecutionResult cold_execution = run_pi_pqt_tensor_tree_execution(
        term_count,
        PiTensorTreeExecutionOptions{.validate_all_metadata = false}
    );
    const auto cold_tree_end = std::chrono::steady_clock::now();
    const PiEndToEndReport cold_report =
        finalize_pi_end_to_end_report(term_count, target_digits, route, cold_execution);
    const auto cold_end = std::chrono::steady_clock::now();

    report.term_count_sufficient = cold_report.term_count_sufficient;
    report.reference_prefix_digits_checked = cold_report.reference_prefix_digits_checked;
    report.chosen_limb_bits = cold_report.chosen_limb_bits;
    report.modulus_count = cold_report.modulus_count;
    report.peak_ntt_size = cold_report.peak_ntt_size;
    report.root_q_bits = cold_report.root_q_bits;
    report.root_t_bits = cold_report.root_t_bits;
    report.closure_mode = cold_report.closure_mode;
    report.pi_decimal = cold_report.pi_decimal;
    report.cold_tree_execution_ms =
        std::chrono::duration<double, std::milli>(cold_tree_end - cold_tree_start).count();
    report.cold_host_closure_ms =
        std::chrono::duration<double, std::milli>(cold_end - cold_tree_end).count();
    report.cold_end_to_end_ms =
        std::chrono::duration<double, std::milli>(cold_end - cold_start).count();
    report.prefix_match = cold_report.prefix_match;
    report.ok = cold_report.ok;

    if (!cold_report.ok) {
        return report;
    }

    double tree_sum_ms = 0.0;
    double closure_sum_ms = 0.0;
    double end_to_end_sum_ms = 0.0;
    PiEndToEndReport last_report = cold_report;
    bool all_ok = cold_report.ok;

    for (int iteration = 0; iteration < measured_iterations; ++iteration) {
        const auto iteration_start = std::chrono::steady_clock::now();
        const auto tree_start = iteration_start;
        const PiTensorTreeExecutionResult execution = run_pi_pqt_tensor_tree_execution(
            term_count,
            PiTensorTreeExecutionOptions{.validate_all_metadata = false}
        );
        const auto tree_end = std::chrono::steady_clock::now();
        last_report = finalize_pi_end_to_end_report(term_count, target_digits, route, execution);
        const auto iteration_end = std::chrono::steady_clock::now();

        tree_sum_ms += std::chrono::duration<double, std::milli>(tree_end - tree_start).count();
        closure_sum_ms += std::chrono::duration<double, std::milli>(iteration_end - tree_end).count();
        end_to_end_sum_ms += std::chrono::duration<double, std::milli>(iteration_end - iteration_start).count();
        all_ok = all_ok && last_report.ok;
    }

    report.term_count_sufficient = last_report.term_count_sufficient;
    report.reference_prefix_digits_checked = last_report.reference_prefix_digits_checked;
    report.chosen_limb_bits = last_report.chosen_limb_bits;
    report.modulus_count = last_report.modulus_count;
    report.peak_ntt_size = last_report.peak_ntt_size;
    report.root_q_bits = last_report.root_q_bits;
    report.root_t_bits = last_report.root_t_bits;
    report.closure_mode = last_report.closure_mode;
    report.pi_decimal = last_report.pi_decimal;
    report.avg_tree_execution_ms = tree_sum_ms / static_cast<double>(measured_iterations);
    report.avg_host_closure_ms = closure_sum_ms / static_cast<double>(measured_iterations);
    report.avg_end_to_end_ms = end_to_end_sum_ms / static_cast<double>(measured_iterations);
    report.avg_digits_per_second_tree_stage =
        report.avg_tree_execution_ms > 0.0
            ? static_cast<double>(report.target_digits) * 1000.0 / report.avg_tree_execution_ms
            : 0.0;
    report.avg_digits_per_second_closure_stage =
        report.avg_host_closure_ms > 0.0
            ? static_cast<double>(report.target_digits) * 1000.0 / report.avg_host_closure_ms
            : 0.0;
    report.avg_digits_per_second_e2e =
        report.avg_end_to_end_ms > 0.0
            ? static_cast<double>(report.target_digits) * 1000.0 / report.avg_end_to_end_ms
            : 0.0;
    report.prefix_match = last_report.prefix_match;
    report.ok = all_ok;
    return report;
}

void print_pi_end_to_end_benchmark_report(std::ostream& out, const PiBenchmarkReport& report) {
    out << "pi_end_to_end_benchmark_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "term_count=" << report.term_count << '\n';
    out << "target_digits=" << report.target_digits << '\n';
    out << "working_digits=" << report.working_digits << '\n';
    out << "required_terms=" << report.required_terms << '\n';
    out << "reference_prefix_digits_checked=" << report.reference_prefix_digits_checked << '\n';
    out << "term_count_sufficient=" << (report.term_count_sufficient ? 1 : 0) << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "chosen_limb_bits=" << report.chosen_limb_bits << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "peak_ntt_size=" << report.peak_ntt_size << '\n';
    out << "root_q_bits=" << report.root_q_bits << '\n';
    out << "root_t_bits=" << report.root_t_bits << '\n';
    out << "closure_mode=" << report.closure_mode << '\n';
    out << "tree_validation_mode=" << report.tree_validation_mode << '\n';
    out << "cold_tree_execution_ms=" << report.cold_tree_execution_ms << '\n';
    out << "cold_host_closure_ms=" << report.cold_host_closure_ms << '\n';
    out << "cold_end_to_end_ms=" << report.cold_end_to_end_ms << '\n';
    out << "avg_tree_execution_ms=" << report.avg_tree_execution_ms << '\n';
    out << "avg_host_closure_ms=" << report.avg_host_closure_ms << '\n';
    out << "avg_end_to_end_ms=" << report.avg_end_to_end_ms << '\n';
    out << "avg_digits_per_second_tree_stage=" << report.avg_digits_per_second_tree_stage << '\n';
    out << "avg_digits_per_second_closure_stage=" << report.avg_digits_per_second_closure_stage << '\n';
    out << "avg_digits_per_second_e2e=" << report.avg_digits_per_second_e2e << '\n';
    out << "pi_decimal=" << report.pi_decimal << '\n';
    out << "prefix_match=" << (report.prefix_match ? 1 : 0) << '\n';
}

void write_pi_end_to_end_benchmark_csv(const PiBenchmarkReport& report, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open pi end-to-end benchmark csv output");
    }

    out << "term_count,target_digits,working_digits,required_terms,"
           "reference_prefix_digits_checked,term_count_sufficient,measured_iterations,"
           "chosen_limb_bits,modulus_count,peak_ntt_size,root_q_bits,root_t_bits,"
           "closure_mode,tree_validation_mode,cold_tree_execution_ms,cold_host_closure_ms,cold_end_to_end_ms,"
           "avg_tree_execution_ms,avg_host_closure_ms,avg_end_to_end_ms,"
           "avg_digits_per_second_tree_stage,avg_digits_per_second_closure_stage,"
           "avg_digits_per_second_e2e,prefix_match,status\n";
    out << report.term_count << ','
        << report.target_digits << ','
        << report.working_digits << ','
        << report.required_terms << ','
        << report.reference_prefix_digits_checked << ','
        << (report.term_count_sufficient ? 1 : 0) << ','
        << report.measured_iterations << ','
        << report.chosen_limb_bits << ','
        << report.modulus_count << ','
        << report.peak_ntt_size << ','
        << report.root_q_bits << ','
        << report.root_t_bits << ','
        << report.closure_mode << ','
        << report.tree_validation_mode << ','
        << report.cold_tree_execution_ms << ','
        << report.cold_host_closure_ms << ','
        << report.cold_end_to_end_ms << ','
        << report.avg_tree_execution_ms << ','
        << report.avg_host_closure_ms << ','
        << report.avg_end_to_end_ms << ','
        << report.avg_digits_per_second_tree_stage << ','
        << report.avg_digits_per_second_closure_stage << ','
        << report.avg_digits_per_second_e2e << ','
        << (report.prefix_match ? 1 : 0) << ','
        << (report.ok ? "ok" : "failed") << '\n';
}

bool pi_end_to_end_benchmark_test(
    std::ostream& out,
    std::size_t term_count,
    std::size_t target_digits,
    int measured_iterations
) {
    const PiBenchmarkReport report =
        run_pi_end_to_end_benchmark(term_count, target_digits, measured_iterations);
    print_pi_end_to_end_benchmark_report(out, report);
    return report.ok;
}

SmokeReport run_smoke_test(const SmokeConfig& config) {
    if (config.value_count == 0 || config.slot_count == 0) {
        throw std::invalid_argument("value_count and slot_count must be positive");
    }
    if (config.measured_iterations <= 0) {
        throw std::invalid_argument("measured_iterations must be positive");
    }

    const int input_bits = resolve_input_bits(config);
    const std::size_t convolution_ntt_size = resolve_convolution_ntt_size(2 * config.slot_count - 1);
    const std::uint64_t input_mask = mask_for_bits(input_bits);
    const bool verification_enabled = config.verify_mode != VerifyMode::kNone;
    const std::size_t pointwise_scalar_count = config.value_count * config.slot_count;
    const std::size_t convolution_scalar_count = config.value_count * (2 * config.slot_count - 1);

    SampleValidationPlan sample_plan;
    if (config.verify_mode == VerifyMode::kSampled) {
        sample_plan.pointwise_indices = build_spread_sample_indices(pointwise_scalar_count, config.verify_sample_count);
        sample_plan.convolution_indices = build_spread_sample_indices(convolution_scalar_count, config.verify_sample_count);
    }

    std::vector<std::uint64_t> lhs_values(config.value_count * config.slot_count);
    std::vector<std::uint64_t> rhs_values(config.value_count * config.slot_count);

    std::uint64_t seed = 0x1234567812345678ull;
    for (std::size_t index = 0; index < lhs_values.size(); ++index) {
        seed = seed * 6364136223846793005ull + 1ull;
        lhs_values[index] = (seed >> 16) & input_mask;
        seed = seed * 6364136223846793005ull + 1ull;
        rhs_values[index] = ((seed >> 16) & input_mask) % (lhs_values[index] + 1);
    }

    DeviceRnsTensor lhs = allocate_device_tensor(config.value_count, config.slot_count);
    DeviceRnsTensor rhs = allocate_device_tensor(config.value_count, config.slot_count);
    DeviceRnsTensor sum = allocate_device_tensor(config.value_count, config.slot_count);
    DeviceRnsTensor diff = allocate_device_tensor(config.value_count, config.slot_count);
    DeviceRnsTensor prod = allocate_device_tensor(config.value_count, config.slot_count);
    DeviceRnsTensor conv = allocate_device_tensor(config.value_count, 2 * config.slot_count - 1);

    SmokeReport report;
    report.layout = describe_layout(lhs);
    report.convolution_algorithm = convolution_algorithm_name(convolution_ntt_size);
    report.modulus_count = static_cast<std::size_t>(lhs.modulus_count);
    report.value_count = config.value_count;
    report.slot_count = config.slot_count;
    report.convolution_ntt_size = convolution_ntt_size;
    report.input_bits = input_bits;
    report.verification_enabled = verification_enabled;
    report.input_staging_mode = "device_reuse";
    report.verification_mode = verify_mode_name(config.verify_mode);
    report.verification_sample_count = config.verify_mode == VerifyMode::kSampled ? config.verify_sample_count : 0;
    report.validated_scalar_slots = pointwise_scalar_count;
    report.validated_convolution_coefficients = convolution_scalar_count;
    report.checked_scalar_slots =
        verification_enabled
            ? (config.verify_mode == VerifyMode::kFull ? pointwise_scalar_count : sample_plan.pointwise_indices.size())
            : 0;
    report.checked_convolution_coefficients =
        verification_enabled
            ? (config.verify_mode == VerifyMode::kFull ? convolution_scalar_count : sample_plan.convolution_indices.size())
            : 0;
    report.measured_iterations = config.measured_iterations;

    const auto cold_start = std::chrono::steady_clock::now();
    const auto input_upload_start = std::chrono::steady_clock::now();
    upload_u64_pair_to_encode_workspace(lhs_values, rhs_values);
    const auto input_upload_end = std::chrono::steady_clock::now();
    report.one_time_input_upload_ms = std::chrono::duration<double, std::milli>(input_upload_end - input_upload_start).count();
    const PipelineTiming cold_timing = execute_pipeline(lhs, rhs, sum, diff, prod, conv);
    bool cold_ok = true;
    double cold_download_ms = 0.0;
    if (verification_enabled) {
        if (config.verify_mode == VerifyMode::kFull) {
            cold_download_ms = measure_full_download_and_reconstruct_ms(
                sum,
                diff,
                prod,
                conv,
                lhs_values,
                rhs_values,
                config.value_count,
                config.slot_count,
                cold_ok
            );
        } else {
            cold_download_ms = measure_sampled_download_and_reconstruct_ms(
                sum,
                diff,
                prod,
                conv,
                lhs_values,
                rhs_values,
                sample_plan,
                config.value_count,
                config.slot_count,
                cold_ok
            );
        }
    }
    const auto cold_end = std::chrono::steady_clock::now();
    report.cold_kernel_ms = cold_timing.kernel_ms;
    report.cold_download_reconstruct_ms = cold_download_ms;
    report.cold_end_to_end_ms = std::chrono::duration<double, std::milli>(cold_end - cold_start).count();
    report.ok = cold_ok;

    double encode_sum_ms = 0.0;
    double pointwise_sum_ms = 0.0;
    double convolution_sum_ms = 0.0;
    double kernel_sum_ms = 0.0;
    double download_sum_ms = 0.0;
    double end_to_end_sum_ms = 0.0;

    for (int iteration = 0; iteration < report.measured_iterations; ++iteration) {
        const auto iteration_start = std::chrono::steady_clock::now();
        const PipelineTiming timing = execute_pipeline(lhs, rhs, sum, diff, prod, conv);
        bool iteration_ok = true;
        double download_ms = 0.0;
        if (verification_enabled) {
            if (config.verify_mode == VerifyMode::kFull) {
                download_ms = measure_full_download_and_reconstruct_ms(
                    sum,
                    diff,
                    prod,
                    conv,
                    lhs_values,
                    rhs_values,
                    config.value_count,
                    config.slot_count,
                    iteration_ok
                );
            } else {
                download_ms = measure_sampled_download_and_reconstruct_ms(
                    sum,
                    diff,
                    prod,
                    conv,
                    lhs_values,
                    rhs_values,
                    sample_plan,
                    config.value_count,
                    config.slot_count,
                    iteration_ok
                );
            }
        }
        const auto iteration_end = std::chrono::steady_clock::now();

        encode_sum_ms += timing.encode_ms;
        pointwise_sum_ms += timing.pointwise_ms;
        convolution_sum_ms += timing.convolution_ms;
        kernel_sum_ms += timing.kernel_ms;
        download_sum_ms += download_ms;
        end_to_end_sum_ms += std::chrono::duration<double, std::milli>(iteration_end - iteration_start).count();
        report.ok = report.ok && iteration_ok;
    }

    report.avg_encode_ms = encode_sum_ms / static_cast<double>(report.measured_iterations);
    report.avg_pointwise_ms = pointwise_sum_ms / static_cast<double>(report.measured_iterations);
    report.avg_convolution_ms = convolution_sum_ms / static_cast<double>(report.measured_iterations);
    report.avg_kernel_ms = kernel_sum_ms / static_cast<double>(report.measured_iterations);
    report.avg_download_reconstruct_ms = download_sum_ms / static_cast<double>(report.measured_iterations);
    report.avg_end_to_end_ms = end_to_end_sum_ms / static_cast<double>(report.measured_iterations);
    report.avg_scalar_slots_per_second_e2e =
        report.avg_end_to_end_ms > 0.0
            ? static_cast<double>(report.validated_scalar_slots) * 1000.0 / report.avg_end_to_end_ms
            : 0.0;
    report.avg_convolution_coefficients_per_second_kernel =
        report.avg_convolution_ms > 0.0
            ? static_cast<double>(report.validated_convolution_coefficients) * 1000.0 / report.avg_convolution_ms
            : 0.0;
    const double pipeline_residue_values =
        static_cast<double>(report.modulus_count) *
        static_cast<double>(5 * report.validated_scalar_slots + report.validated_convolution_coefficients);
    report.avg_pipeline_residue_values_per_second_kernel =
        report.avg_kernel_ms > 0.0 ? pipeline_residue_values * 1000.0 / report.avg_kernel_ms : 0.0;
    report.avg_download_over_kernel_ratio =
        report.avg_kernel_ms > 0.0 ? report.avg_download_reconstruct_ms / report.avg_kernel_ms : 0.0;

    free_device_tensor(lhs);
    free_device_tensor(rhs);
    free_device_tensor(sum);
    free_device_tensor(diff);
    free_device_tensor(prod);
    free_device_tensor(conv);
    return report;
}

SmokeReport run_smoke_test() {
    return run_smoke_test(SmokeConfig{});
}

void print_smoke_report(std::ostream& out, const SmokeReport& report) {
    out << std::fixed << std::setprecision(6);
    out << report.layout << '\n';
    out << "convolution_algorithm=" << report.convolution_algorithm << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "value_count=" << report.value_count << '\n';
    out << "slot_count=" << report.slot_count << '\n';
    out << "convolution_ntt_size=" << report.convolution_ntt_size << '\n';
    out << "input_bits=" << report.input_bits << '\n';
    out << "verification_enabled=" << (report.verification_enabled ? 1 : 0) << '\n';
    out << "input_staging_mode=" << report.input_staging_mode << '\n';
    out << "one_time_input_upload_ms=" << report.one_time_input_upload_ms << '\n';
    out << "verification_mode=" << report.verification_mode << '\n';
    out << "verification_sample_count=" << report.verification_sample_count << '\n';
    out << "checked_scalar_slots=" << report.checked_scalar_slots << '\n';
    out << "checked_convolution_coefficients=" << report.checked_convolution_coefficients << '\n';
    out << "validated_scalar_slots=" << report.validated_scalar_slots << '\n';
    out << "validated_convolution_coefficients=" << report.validated_convolution_coefficients << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "cold_kernel_ms=" << report.cold_kernel_ms << '\n';
    out << "cold_download_reconstruct_ms=" << report.cold_download_reconstruct_ms << '\n';
    out << "cold_end_to_end_ms=" << report.cold_end_to_end_ms << '\n';
    out << "avg_encode_ms=" << report.avg_encode_ms << '\n';
    out << "avg_pointwise_ms=" << report.avg_pointwise_ms << '\n';
    out << "avg_convolution_ms=" << report.avg_convolution_ms << '\n';
    out << "avg_kernel_ms=" << report.avg_kernel_ms << '\n';
    out << "avg_download_reconstruct_ms=" << report.avg_download_reconstruct_ms << '\n';
    out << "avg_end_to_end_ms=" << report.avg_end_to_end_ms << '\n';
    out << "avg_scalar_slots_per_second_e2e=" << report.avg_scalar_slots_per_second_e2e << '\n';
    out << "avg_convolution_coefficients_per_second_kernel=" << report.avg_convolution_coefficients_per_second_kernel
        << '\n';
    out << "avg_pipeline_residue_values_per_second_kernel="
        << report.avg_pipeline_residue_values_per_second_kernel << '\n';
    out << "avg_download_over_kernel_ratio=" << report.avg_download_over_kernel_ratio << '\n';
    out << "status=" << (report.ok ? "ok" : "failed") << '\n';
}

void write_smoke_csv(const SmokeReport& report, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open smoke csv output: " + path);
    }
    out << std::fixed << std::setprecision(6);
    out << "layout,convolution_algorithm,modulus_count,value_count,slot_count,convolution_ntt_size,input_bits,verification_enabled,"
           "input_staging_mode,one_time_input_upload_ms,"
           "verification_mode,verification_sample_count,checked_scalar_slots,checked_convolution_coefficients,"
           "validated_scalar_slots,validated_convolution_coefficients,measured_iterations,"
           "cold_kernel_ms,cold_download_reconstruct_ms,cold_end_to_end_ms,"
           "avg_encode_ms,avg_pointwise_ms,avg_convolution_ms,avg_kernel_ms,"
           "avg_download_reconstruct_ms,avg_end_to_end_ms,"
           "avg_scalar_slots_per_second_e2e,avg_convolution_coefficients_per_second_kernel,"
           "avg_pipeline_residue_values_per_second_kernel,avg_download_over_kernel_ratio,status\n";
    out << '"' << report.layout << '"' << ','
        << report.convolution_algorithm << ','
        << report.modulus_count << ','
        << report.value_count << ','
        << report.slot_count << ','
        << report.convolution_ntt_size << ','
        << report.input_bits << ','
        << (report.verification_enabled ? 1 : 0) << ','
        << report.input_staging_mode << ','
        << report.one_time_input_upload_ms << ','
        << report.verification_mode << ','
        << report.verification_sample_count << ','
        << report.checked_scalar_slots << ','
        << report.checked_convolution_coefficients << ','
        << report.validated_scalar_slots << ','
        << report.validated_convolution_coefficients << ','
        << report.measured_iterations << ','
        << report.cold_kernel_ms << ','
        << report.cold_download_reconstruct_ms << ','
        << report.cold_end_to_end_ms << ','
        << report.avg_encode_ms << ','
        << report.avg_pointwise_ms << ','
        << report.avg_convolution_ms << ','
        << report.avg_kernel_ms << ','
        << report.avg_download_reconstruct_ms << ','
        << report.avg_end_to_end_ms << ','
        << report.avg_scalar_slots_per_second_e2e << ','
        << report.avg_convolution_coefficients_per_second_kernel << ','
        << report.avg_pipeline_residue_values_per_second_kernel << ','
        << report.avg_download_over_kernel_ratio << ','
        << (report.ok ? "ok" : "failed") << '\n';
}

bool smoke_test(std::ostream& out) {
    const SmokeReport report = run_smoke_test();
    print_smoke_report(out, report);
    return report.ok;
}

}  // namespace project2::gpu_native_rns
