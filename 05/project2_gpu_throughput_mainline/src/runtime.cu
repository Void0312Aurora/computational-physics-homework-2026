#include "project2_gpu_throughput_mainline/runtime.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace project2::gpu_throughput_mainline {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kCoefficientBaseBits = 16;
constexpr std::uint32_t kCoefficientBaseMask = 0xffffu;
constexpr std::uint64_t kChudnovskyA = 13591409ull;
constexpr std::uint64_t kChudnovskyB = 545140134ull;
constexpr std::uint64_t kChudnovskyC3Over24 = 10939058860032000ull;
constexpr std::array<std::uint32_t, 40> kDefaultModuli = {
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
    2147483647u,
    2147483629u,
    2147483587u,
    2147483579u,
    2147483563u,
    2147483549u,
    2147483543u,
    2147483497u,
    2147483489u,
    2147483477u,
    2147483423u,
    2147483399u,
    2147483353u,
    2147483323u,
    2147483269u,
    2147483249u,
    2147483237u,
    2147483179u,
    2147483171u,
    2147483137u,
    2147483123u,
    2147483077u,
    2147483069u,
    2147483059u,
    2147483053u,
    2147483033u,
    2147483029u,
    2147482951u,
    2147482949u,
    2147482943u,
};

struct SplitLimbDescriptor {
    int shift = 0;
    int bit_count = 0;
    std::uint32_t mask = 0;
};

struct SplitPassDescriptor {
    int lhs_shift = 0;
    std::uint32_t lhs_mask = 0;
    int rhs_shift = 0;
    std::uint32_t rhs_mask = 0;
    int accumulation_shift = 0;
};

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
std::pair<HostBigInt, std::uint32_t> div_mod_u32_abs(const HostBigInt& dividend, std::uint32_t divisor);

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

[[maybe_unused]] bool test_bit(const HostBigInt& value, std::size_t bit_index) {
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

[[maybe_unused]] void add_u32_inplace(HostBigInt& value, std::uint32_t addend) {
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

void mul_u32_inplace(HostBigInt& value, std::uint32_t factor) {
    if (value.is_zero() || factor == 1u) {
        return;
    }
    if (factor == 0u) {
        value = HostBigInt{};
        return;
    }

    unsigned __int128 carry = 0;
    for (std::size_t index = 0; index < value.limbs.size(); ++index) {
        const unsigned __int128 current =
            static_cast<unsigned __int128>(value.limbs[index]) * static_cast<unsigned __int128>(factor) + carry;
        value.limbs[index] = static_cast<std::uint32_t>(current & 0xffffffffu);
        carry = current >> 32u;
    }
    while (carry != 0) {
        value.limbs.push_back(static_cast<std::uint32_t>(carry & 0xffffffffu));
        carry >>= 32u;
    }
}

std::vector<std::uint32_t> shift_left_bits_copy(const std::vector<std::uint32_t>& limbs, unsigned shift_bits) {
    if (shift_bits == 0 || limbs.empty()) {
        return limbs;
    }

    std::vector<std::uint32_t> out(limbs.size(), 0u);
    std::uint64_t carry = 0u;
    for (std::size_t index = 0; index < limbs.size(); ++index) {
        const std::uint64_t current = (static_cast<std::uint64_t>(limbs[index]) << shift_bits) | carry;
        out[index] = static_cast<std::uint32_t>(current & 0xffffffffull);
        carry = current >> 32u;
    }
    if (carry != 0u) {
        out.push_back(static_cast<std::uint32_t>(carry));
    }
    return out;
}

std::vector<std::uint32_t> shift_right_bits_copy(const std::vector<std::uint32_t>& limbs, unsigned shift_bits) {
    if (shift_bits == 0 || limbs.empty()) {
        return limbs;
    }

    std::vector<std::uint32_t> out(limbs.size(), 0u);
    const std::uint32_t low_mask = static_cast<std::uint32_t>((std::uint64_t{1} << shift_bits) - 1ull);
    std::uint32_t carry = 0u;
    for (std::size_t index = limbs.size(); index > 0; --index) {
        const std::uint32_t limb = limbs[index - 1];
        out[index - 1] = (limb >> shift_bits) | (carry << (32u - shift_bits));
        carry = limb & low_mask;
    }
    while (!out.empty() && out.back() == 0u) {
        out.pop_back();
    }
    return out;
}

[[maybe_unused]] void shift_left_one_inplace(HostBigInt& value) {
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
    if (divisor.limbs.size() == 1u) {
        const auto [quotient, remainder] = div_mod_u32_abs(dividend, divisor.limbs[0]);
        return {quotient, HostBigInt(static_cast<long long>(remainder))};
    }

    constexpr std::uint64_t kBase = 1ull << 32u;
    const unsigned normalization_shift = static_cast<unsigned>(__builtin_clz(divisor.limbs.back()));
    const std::vector<std::uint32_t> normalized_divisor =
        shift_left_bits_copy(divisor.limbs, normalization_shift);
    std::vector<std::uint32_t> normalized_dividend =
        shift_left_bits_copy(dividend.limbs, normalization_shift);
    normalized_dividend.push_back(0u);

    const std::size_t divisor_limb_count = normalized_divisor.size();
    const std::size_t quotient_limb_count = normalized_dividend.size() - divisor_limb_count;

    HostBigInt quotient;
    quotient.sign = 1;
    quotient.limbs.assign(quotient_limb_count, 0u);

    for (std::size_t q_index = quotient_limb_count; q_index > 0; --q_index) {
        const std::size_t j = q_index - 1u;
        const unsigned __int128 numerator =
            (static_cast<unsigned __int128>(normalized_dividend[j + divisor_limb_count]) << 32u) |
            static_cast<unsigned __int128>(normalized_dividend[j + divisor_limb_count - 1u]);
        std::uint64_t quotient_hat =
            static_cast<std::uint64_t>(numerator / normalized_divisor[divisor_limb_count - 1u]);
        std::uint64_t remainder_hat =
            static_cast<std::uint64_t>(numerator % normalized_divisor[divisor_limb_count - 1u]);

        if (quotient_hat >= kBase) {
            quotient_hat = kBase - 1u;
        }
        while (divisor_limb_count > 1u) {
            const unsigned __int128 lhs =
                static_cast<unsigned __int128>(quotient_hat) *
                static_cast<unsigned __int128>(normalized_divisor[divisor_limb_count - 2u]);
            const unsigned __int128 rhs =
                (static_cast<unsigned __int128>(remainder_hat) << 32u) |
                static_cast<unsigned __int128>(normalized_dividend[j + divisor_limb_count - 2u]);
            if (lhs <= rhs) {
                break;
            }
            --quotient_hat;
            remainder_hat += normalized_divisor[divisor_limb_count - 1u];
            if (remainder_hat >= kBase) {
                break;
            }
        }

        std::uint64_t carry = 0u;
        std::uint64_t borrow = 0u;
        for (std::size_t divisor_index = 0; divisor_index < divisor_limb_count; ++divisor_index) {
            const unsigned __int128 product =
                static_cast<unsigned __int128>(quotient_hat) *
                static_cast<unsigned __int128>(normalized_divisor[divisor_index]) +
                static_cast<unsigned __int128>(carry);
            carry = static_cast<std::uint64_t>(product >> 32u);
            const std::uint64_t product_low = static_cast<std::uint32_t>(product & 0xffffffffu);
            const std::uint64_t subtrahend = product_low + borrow;
            const std::uint64_t current = normalized_dividend[j + divisor_index];
            if (current < subtrahend) {
                normalized_dividend[j + divisor_index] =
                    static_cast<std::uint32_t>(current + kBase - subtrahend);
                borrow = 1u;
            } else {
                normalized_dividend[j + divisor_index] =
                    static_cast<std::uint32_t>(current - subtrahend);
                borrow = 0u;
            }
        }

        const std::uint64_t high_subtrahend = carry + borrow;
        bool underflow = false;
        if (static_cast<std::uint64_t>(normalized_dividend[j + divisor_limb_count]) < high_subtrahend) {
            normalized_dividend[j + divisor_limb_count] = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(normalized_dividend[j + divisor_limb_count]) + kBase - high_subtrahend
            );
            underflow = true;
        } else {
            normalized_dividend[j + divisor_limb_count] = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(normalized_dividend[j + divisor_limb_count]) - high_subtrahend
            );
        }

        if (underflow) {
            --quotient_hat;
            std::uint64_t add_carry = 0u;
            for (std::size_t divisor_index = 0; divisor_index < divisor_limb_count; ++divisor_index) {
                const std::uint64_t sum =
                    static_cast<std::uint64_t>(normalized_dividend[j + divisor_index]) +
                    static_cast<std::uint64_t>(normalized_divisor[divisor_index]) +
                    add_carry;
                normalized_dividend[j + divisor_index] = static_cast<std::uint32_t>(sum & 0xffffffffull);
                add_carry = sum >> 32u;
            }
            normalized_dividend[j + divisor_limb_count] = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(normalized_dividend[j + divisor_limb_count]) + add_carry
            );
        }

        quotient.limbs[j] = static_cast<std::uint32_t>(quotient_hat);
    }

    quotient.normalize();

    HostBigInt remainder;
    remainder.sign = 1;
    remainder.limbs.assign(
        normalized_dividend.begin(),
        normalized_dividend.begin() + static_cast<std::ptrdiff_t>(divisor_limb_count)
    );
    remainder.limbs = shift_right_bits_copy(remainder.limbs, normalization_shift);
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

std::size_t bit_width_cpp_int(const HostBigInt& value) {
    if (value.is_zero()) {
        return 0;
    }
    const std::uint32_t top_limb = value.limbs.back();
    return 32u * (value.limbs.size() - 1u) + static_cast<std::size_t>(32 - __builtin_clz(top_limb));
}

const std::string& pi_reference_digits() {
    static const std::string digits =
        "31415926535897932384626433832795028841971693993751058209749445923078164062862089"
        "98628034825342117067982148086513282306647093844609550582231725359408128481117450"
        "28410270193852110555964462294895493038196442881097566593344612847564823378678316"
        "52712019091456485669234603486104543266482133936072602491412737245870066063155881"
        "74881520920962829254091715364367892590360011330530548820466521384146951941511609"
        "43305727036575959195309218611738193261179310511854807446237996274956735188575272"
        "48912279381830119491298336733624406566430860213949463952247371907021798609437027"
        "70539217176293176752384674818467669405132000568127145263560827785771342757789609"
        "17363717872146844090122495343014654958537105079227968925892354201995611212902196"
        "08640344181598136297747713099605187072113499999983729780499510597317328160963185"
        "95024459455346908302642522308253344685035261931188171010003137838752886587533208"
        "38142061717766914730359825349042875546873115956286388235378759375195778185778053"
        "21712268066130019278766111959092164201989380952572010654858632788659361533818279"
        "68230301952035301852968995773622599413891249721775283479131515574857242454150695"
        "95082953311686172785588907509838175463746493931925506040092770167113900984882401"
        "28583616035637076601047101819429555961989467678374494482553797747268471040475346"
        "46208046684259069491293313677028989152104752162056966024058038150193511253382430"
        "03558764024749647326391419927260426992279678235478163600934172164121992458631503"
        "02861829745557067498385054945885869269956909272107975093029553211653449872027559"
        "60236480665499119881834797753566369807426542527862551818417574672890977772793800"
        "08164706001614524919217321721477235014144197356854816136115735255213347574184946"
        "84385233239073941433345477624168625189835694855620992192221842725502542568876717"
        "90494601653466804988627232791786085784383827967976681454100953883786360950680064"
        "22512520511739298489608412848862694560424196528502221066118630674427862203919494"
        "50471237137869609563643719172874677646575739624138908658326459958133904780275900"
        "99465764078951269468398352595709825822620522489407726719478268482601476990902640"
        "13639443745530506820349625245174939965143142980919065925093722169646151570985838"
        "74105978859597729754989301617539284681382686838689427741559918559252459539594310"
        "49972524680845987273644695848653836736222626099124608051243884390451244136549762"
        "78079771569143599770012961608944169486855584840635342207222582848864815845602850"
        "60168427394522674676788952521385225499546667278239864565961163548862305774564980"
        "355936345681743241125";
    return digits;
}

std::size_t decimal_length_size_t(std::size_t value) {
    std::size_t digits = 1;
    while (value >= 10) {
        value /= 10;
        ++digits;
    }
    return digits;
}

std::size_t required_working_digits_for_pi(std::size_t target_digits) {
    if (target_digits == 0) {
        throw std::invalid_argument("required_working_digits_for_pi requires target_digits > 0");
    }
    return target_digits + std::max<std::size_t>(32, decimal_length_size_t(target_digits) + 16);
}

std::size_t required_chudnovsky_terms_for_pi(std::size_t target_digits) {
    constexpr long double kChudnovskyDigitsPerTerm = 14.1816474627254776555L;
    const std::size_t working_digits = required_working_digits_for_pi(target_digits);
    return std::max<std::size_t>(
        1,
        static_cast<std::size_t>(
            std::ceil(static_cast<long double>(working_digits) / kChudnovskyDigitsPerTerm)
        )
    );
}

HostBigInt pow10_host_bigint(std::size_t exponent) {
    HostBigInt value = 1;
    while (exponent >= 9u) {
        mul_u32_inplace(value, 1000000000u);
        exponent -= 9u;
    }
    static constexpr std::array<std::uint32_t, 9> kSmallPowersOfTen = {
        1u,
        10u,
        100u,
        1000u,
        10000u,
        100000u,
        1000000u,
        10000000u,
        100000000u
    };
    if (exponent > 0u) {
        mul_u32_inplace(value, kSmallPowersOfTen[exponent]);
    }
    return value;
}

HostBigInt divide_by_power_of_ten_host_bigint(const HostBigInt& value, std::size_t exponent) {
    HostBigInt quotient = value;
    while (exponent >= 9u) {
        quotient = div_mod_u32_abs(quotient, 1000000000u).first;
        exponent -= 9u;
    }
    static constexpr std::array<std::uint32_t, 9> kSmallPowersOfTen = {
        1u,
        10u,
        100u,
        1000u,
        10000u,
        100000u,
        1000000u,
        10000000u,
        100000000u
    };
    if (exponent > 0u) {
        quotient = div_mod_u32_abs(quotient, kSmallPowersOfTen[exponent]).first;
    }
    return quotient;
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
        current = next;
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

std::string format_scaled_decimal_prefix_from_digits(
    const std::string& digits,
    std::size_t fractional_digits,
    std::size_t reported_fractional_digits,
    bool& truncated
) {
    if (!digits.empty() && digits.front() == '-') {
        throw std::invalid_argument("format_scaled_decimal_prefix_from_digits expects a non-negative scaled value");
    }

    const std::size_t kept_fractional_digits = std::min(fractional_digits, reported_fractional_digits);
    truncated = kept_fractional_digits < fractional_digits;

    std::string formatted;
    formatted.reserve(2u + kept_fractional_digits + (truncated ? 3u : 0u));
    formatted.push_back(digits.front());
    if (fractional_digits > 0u) {
        formatted.push_back('.');
        if (kept_fractional_digits > 0u) {
            formatted.append(digits.data() + 1, kept_fractional_digits);
        }
    }
    if (truncated) {
        formatted += "...";
    }
    return formatted;
}

std::uint32_t mod_inverse_u32(std::uint32_t value, std::uint32_t modulus) {
    long long t = 0;
    long long new_t = 1;
    long long r = static_cast<long long>(modulus);
    long long new_r = static_cast<long long>(value % modulus);
    while (new_r != 0) {
        const long long quotient = r / new_r;
        const long long next_t = t - quotient * new_t;
        t = new_t;
        new_t = next_t;
        const long long next_r = r - quotient * new_r;
        r = new_r;
        new_r = next_r;
    }
    if (r != 1) {
        throw std::runtime_error("mod_inverse_u32 requires coprime inputs");
    }
    if (t < 0) {
        t += static_cast<long long>(modulus);
    }
    return static_cast<std::uint32_t>(t);
}

__device__ __forceinline__ std::uint32_t mod_inverse_u32_device(std::uint32_t value, std::uint32_t modulus) {
    long long t = 0;
    long long new_t = 1;
    long long r = static_cast<long long>(modulus);
    long long new_r = static_cast<long long>(value % modulus);
    while (new_r != 0) {
        const long long quotient = r / new_r;
        const long long next_t = t - quotient * new_t;
        t = new_t;
        new_t = next_t;
        const long long next_r = r - quotient * new_r;
        r = new_r;
        new_r = next_r;
    }
    if (t < 0) {
        t += static_cast<long long>(modulus);
    }
    return static_cast<std::uint32_t>(t);
}

std::uint32_t mod_u32_abs(const HostBigInt& value, std::uint32_t modulus) {
    return div_mod_u32_abs(abs_value(value), modulus).second;
}

[[maybe_unused]] HostBigInt centered_crt_from_residues(
    const std::vector<std::uint32_t>& residues,
    const std::vector<std::uint32_t>& moduli
) {
    if (residues.size() != moduli.size() || residues.empty()) {
        throw std::invalid_argument("centered_crt_from_residues requires equal non-empty residue/modulus vectors");
    }

    HostBigInt value = 0;
    HostBigInt modulus_product = 1;
    for (std::size_t index = 0; index < residues.size(); ++index) {
        const std::uint32_t modulus = moduli[index];
        const std::uint32_t residue = residues[index] % modulus;
        const std::uint32_t current_mod = mod_u32_abs(value, modulus);
        const std::uint32_t product_mod = mod_u32_abs(modulus_product, modulus);
        const std::uint32_t inverse = mod_inverse_u32(product_mod, modulus);
        const std::uint32_t delta =
            residue >= current_mod ? residue - current_mod : residue + modulus - current_mod;
        const std::uint32_t step =
            static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(delta) * static_cast<std::uint64_t>(inverse)) % modulus
            );
        value = value + modulus_product * static_cast<long long>(step);
        modulus_product = modulus_product * static_cast<long long>(modulus);
    }

    HostBigInt half_modulus_product = modulus_product;
    shift_right_one_inplace(half_modulus_product);
    if (abs_compare(value, half_modulus_product) > 0) {
        value = value - modulus_product;
    }
    return value;
}

std::size_t crt_product_bit_width(const std::vector<std::uint32_t>& moduli) {
    if (moduli.empty()) {
        return 0;
    }
    HostBigInt product = 1;
    for (const std::uint32_t modulus : moduli) {
        product = product * static_cast<long long>(modulus);
    }
    return bit_width_cpp_int(product);
}

std::size_t bit_width_u128(unsigned __int128 value) {
    std::size_t bits = 0;
    while (value != 0) {
        value >>= 1u;
        ++bits;
    }
    return bits;
}

void signed_div_mod_u32_euclidean(
    const HostBigInt& value,
    std::uint32_t divisor,
    HostBigInt& quotient,
    std::uint32_t& remainder
) {
    const auto [quotient_abs, remainder_abs] = div_mod_u32_abs(abs_value(value), divisor);
    if (value.sign >= 0) {
        quotient = quotient_abs;
        remainder = remainder_abs;
        return;
    }

    if (remainder_abs == 0u) {
        quotient = -quotient_abs;
        remainder = 0u;
        return;
    }

    quotient = -(quotient_abs + HostBigInt{1});
    remainder = divisor - remainder_abs;
}

std::vector<std::int32_t> extract_centered_base_digits_host_bigint(
    const HostBigInt& value,
    std::size_t digit_count,
    std::size_t limb_bits
) {
    if (limb_bits == 0 || limb_bits >= 31) {
        throw std::invalid_argument("extract_centered_base_digits_host_bigint requires limb_bits in [1, 30]");
    }

    const std::uint32_t base = std::uint32_t{1} << limb_bits;
    std::vector<std::int32_t> digits(digit_count, 0);
    HostBigInt current = value;
    for (std::size_t index = 0; index < digit_count && !current.is_zero(); ++index) {
        HostBigInt quotient;
        std::uint32_t remainder = 0u;
        signed_div_mod_u32_euclidean(current, base, quotient, remainder);

        std::int32_t digit = static_cast<std::int32_t>(remainder);
        if (remainder > base / 2u) {
            digit -= static_cast<std::int32_t>(base);
            quotient = quotient + HostBigInt{1};
        }

        digits[index] = digit;
        current = std::move(quotient);
    }

    if (!current.is_zero()) {
        throw std::runtime_error("extract_centered_base_digits_host_bigint exceeded digit_count");
    }

    return digits;
}

__host__ __device__ std::uint32_t balanced_digit_to_residue(std::int32_t digit, std::uint32_t modulus) {
    const long long reduced = static_cast<long long>(digit) % static_cast<long long>(modulus);
    return static_cast<std::uint32_t>(reduced >= 0 ? reduced : reduced + static_cast<long long>(modulus));
}

using BalancedDigitNodes = std::vector<std::vector<std::int32_t>>;

std::array<BalancedDigitNodes, 3> make_chudnovsky_leaf_balanced_digits(
    std::size_t term_count,
    std::size_t slot_count
);

void pack_balanced_digit_nodes_to_residues(
    const BalancedDigitNodes& nodes,
    const std::vector<std::uint32_t>& moduli,
    std::size_t slot_count,
    std::vector<std::uint32_t>& out
) {
    const std::size_t node_count = nodes.size();
    out.assign(moduli.size() * node_count * slot_count, 0u);
    for (std::size_t modulus_index = 0; modulus_index < moduli.size(); ++modulus_index) {
        const std::uint32_t modulus = moduli[modulus_index];
        for (std::size_t node_index = 0; node_index < node_count; ++node_index) {
            for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
                out[modulus_index * node_count * slot_count + node_index * slot_count + slot_index] =
                    balanced_digit_to_residue(nodes[node_index][slot_index], modulus);
            }
        }
    }
}

__int128 centered_crt_i128_from_residues(
    const std::uint32_t* residues,
    const std::vector<std::uint32_t>& moduli
) {
    if (moduli.empty()) {
        throw std::invalid_argument("centered_crt_i128_from_residues requires non-empty moduli");
    }

    __int128 value = 0;
    __int128 modulus_product = 1;
    for (std::size_t index = 0; index < moduli.size(); ++index) {
        const std::uint32_t modulus = moduli[index];
        const std::uint32_t residue = residues[index] % modulus;
        const std::uint32_t current_mod =
            static_cast<std::uint32_t>((value % static_cast<__int128>(modulus) + modulus) % modulus);
        const std::uint32_t product_mod =
            static_cast<std::uint32_t>(modulus_product % static_cast<__int128>(modulus));
        const std::uint32_t inverse = mod_inverse_u32(product_mod, modulus);
        const std::uint32_t delta =
            residue >= current_mod ? residue - current_mod : residue + modulus - current_mod;
        const std::uint32_t step =
            static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(delta) * static_cast<std::uint64_t>(inverse)) % modulus
            );
        value += modulus_product * static_cast<__int128>(step);
        modulus_product *= static_cast<__int128>(modulus);
    }

    const __int128 half_modulus_product = modulus_product / 2;
    if (value > half_modulus_product) {
        value -= modulus_product;
    }
    return value;
}

std::size_t balanced_level_coefficient_bound_bits(std::size_t slot_count, std::size_t limb_bits) {
    if (limb_bits == 0 || limb_bits >= 31) {
        throw std::invalid_argument("balanced_level_coefficient_bound_bits requires limb_bits in [1, 30]");
    }

    const unsigned __int128 digit_bound = static_cast<unsigned __int128>(std::uint32_t{1} << (limb_bits - 1u));
    const unsigned __int128 coefficient_bound = 2u * static_cast<unsigned __int128>(slot_count) * digit_bound * digit_bound;
    return bit_width_u128(coefficient_bound);
}

std::vector<std::uint32_t> select_effective_balanced_closure_moduli(
    int requested_modulus_count,
    std::size_t slot_count,
    int forced_effective_modulus_count = 0
) {
    if (requested_modulus_count < 1 || requested_modulus_count > static_cast<int>(kDefaultModuli.size())) {
        throw std::invalid_argument(
            "select_effective_balanced_closure_moduli requires modulus_count in [1, " +
            std::to_string(kDefaultModuli.size()) + "]"
        );
    }

    const std::size_t required_half_range_bits =
        balanced_level_coefficient_bound_bits(slot_count, static_cast<std::size_t>(kCoefficientBaseBits));
    std::vector<std::uint32_t> selected_moduli;
    unsigned __int128 modulus_product = 1;
    for (int modulus_index = 0; modulus_index < requested_modulus_count; ++modulus_index) {
        const std::uint32_t modulus = kDefaultModuli[static_cast<std::size_t>(modulus_index)];
        selected_moduli.push_back(modulus);
        modulus_product *= static_cast<unsigned __int128>(modulus);
        const std::size_t half_range_bits = bit_width_u128(modulus_product / 2u);
        if (half_range_bits > required_half_range_bits) {
            if (forced_effective_modulus_count == 0) {
                return selected_moduli;
            }
            if (forced_effective_modulus_count < static_cast<int>(selected_moduli.size())) {
                throw std::invalid_argument(
                    "forced_closure_modulus_count is smaller than the minimum exact closure modulus window"
                );
            }
            if (forced_effective_modulus_count > requested_modulus_count) {
                throw std::invalid_argument(
                    "forced_closure_modulus_count cannot exceed modulus_count"
                );
            }
            return std::vector<std::uint32_t>(
                kDefaultModuli.begin(),
                kDefaultModuli.begin() + forced_effective_modulus_count
            );
        }
    }

    throw std::runtime_error(
        "requested modulus_count is insufficient for balanced per-level exact closure reconstruction"
    );
}

std::vector<std::int32_t> normalize_centered_coefficients_i128_to_digits(
    const std::vector<__int128>& coefficients,
    std::size_t limb_bits
) {
    if (limb_bits == 0 || limb_bits >= 31) {
        throw std::invalid_argument("normalize_centered_coefficients_i128_to_digits requires limb_bits in [1, 30]");
    }

    const __int128 base = static_cast<__int128>(std::uint32_t{1} << limb_bits);
    const __int128 half_base = base / 2;
    std::vector<std::int32_t> digits(coefficients.size(), 0);
    __int128 carry = 0;
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        const __int128 total = coefficients[index] + carry;
        __int128 remainder = total % base;
        if (remainder < 0) {
            remainder += base;
        }
        __int128 next_carry = (total - remainder) / base;
        if (remainder > half_base) {
            remainder -= base;
            next_carry += 1;
        }
        digits[index] = static_cast<std::int32_t>(remainder);
        carry = next_carry;
    }

    if (carry != 0) {
        throw std::runtime_error("normalize_centered_coefficients_i128_to_digits exceeded digit_count");
    }
    return digits;
}

HostBigInt rebuild_bigint_from_balanced_digits_i32(
    const std::vector<std::int32_t>& digits,
    std::size_t limb_bits
) {
    if (limb_bits == 0 || limb_bits > 16 || (32u % limb_bits) != 0u) {
        throw std::invalid_argument("rebuild_bigint_from_balanced_digits_i32 requires limb_bits dividing 32 and <= 16");
    }

    const HostBigInt base = static_cast<long long>(1ull << limb_bits);
    HostBigInt value;
    for (std::size_t index = digits.size(); index > 0; --index) {
        value = value * base + HostBigInt(static_cast<long long>(digits[index - 1]));
    }
    return value;
}

[[maybe_unused]] HostBigInt rebuild_bigint_from_centered_coefficients_host_bigint(
    const std::vector<HostBigInt>& coefficients,
    std::size_t limb_bits
) {
    if (limb_bits == 0 || limb_bits > 16 || (32u % limb_bits) != 0u) {
        throw std::invalid_argument(
            "rebuild_bigint_from_centered_coefficients_host_bigint requires limb_bits dividing 32 and <= 16"
        );
    }

    const HostBigInt base = static_cast<long long>(1ull << limb_bits);
    HostBigInt value;
    for (std::size_t index = coefficients.size(); index > 0; --index) {
        value = value * base + coefficients[index - 1];
    }
    return value;
}

struct ExactPqtHostBigInt {
    HostBigInt p;
    HostBigInt q;
    HostBigInt t;
};

ExactPqtHostBigInt make_chudnovsky_leaf_host_bigint(std::size_t term_index) {
    ExactPqtHostBigInt node;
    if (term_index == 0) {
        node.p = 1;
        node.q = 1;
        node.t = static_cast<long long>(kChudnovskyA);
        return node;
    }

    const HostBigInt k = static_cast<long long>(term_index);
    node.p = (6 * k - 5) * (2 * k - 1) * (6 * k - 1);
    node.q = k * k * k * static_cast<long long>(kChudnovskyC3Over24);
    node.t = node.p * (static_cast<long long>(kChudnovskyA) + static_cast<long long>(kChudnovskyB) * k);
    if ((term_index & 1u) != 0u) {
        node.t = -node.t;
    }
    return node;
}

ExactPqtHostBigInt build_chudnovsky_exact_root_host_bigint(std::size_t term_count) {
    if (term_count == 0 || (term_count & (term_count - 1u)) != 0u) {
        throw std::invalid_argument("build_chudnovsky_exact_root_host_bigint requires power-of-two term_count > 0");
    }

    std::vector<ExactPqtHostBigInt> current;
    current.reserve(term_count);
    for (std::size_t term_index = 0; term_index < term_count; ++term_index) {
        current.push_back(make_chudnovsky_leaf_host_bigint(term_index));
    }

    while (current.size() > 1) {
        std::vector<ExactPqtHostBigInt> next;
        next.reserve(current.size() / 2u);
        for (std::size_t index = 0; index < current.size(); index += 2u) {
            ExactPqtHostBigInt merged;
            merged.p = current[index].p * current[index + 1u].p;
            merged.q = current[index].q * current[index + 1u].q;
            merged.t = current[index].t * current[index + 1u].q + current[index].p * current[index + 1u].t;
            next.push_back(std::move(merged));
        }
        current = std::move(next);
    }

    return current.front();
}

std::array<BalancedDigitNodes, 3> make_chudnovsky_leaf_balanced_digits(
    std::size_t term_count,
    std::size_t slot_count
) {
    std::array<BalancedDigitNodes, 3> leaves;
    for (auto& stream_nodes : leaves) {
        stream_nodes.assign(term_count, std::vector<std::int32_t>(slot_count, 0));
    }

    for (std::size_t term_index = 0; term_index < term_count; ++term_index) {
        const ExactPqtHostBigInt leaf = make_chudnovsky_leaf_host_bigint(term_index);
        leaves[0][term_index] = extract_centered_base_digits_host_bigint(
            leaf.p,
            slot_count,
            static_cast<std::size_t>(kCoefficientBaseBits)
        );
        leaves[1][term_index] = extract_centered_base_digits_host_bigint(
            leaf.q,
            slot_count,
            static_cast<std::size_t>(kCoefficientBaseBits)
        );
        leaves[2][term_index] = extract_centered_base_digits_host_bigint(
            leaf.t,
            slot_count,
            static_cast<std::size_t>(kCoefficientBaseBits)
        );
    }

    return leaves;
}

std::size_t block_count(std::size_t total) {
    if (total == 0) {
        return 1;
    }
    return (total + static_cast<std::size_t>(kThreadsPerBlock) - 1) / static_cast<std::size_t>(kThreadsPerBlock);
}

void check_cuda(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream out;
    out << what << ": " << cudaGetErrorString(status);
    throw std::runtime_error(out.str());
}

void check_cufft(cufftResult status, const char* what) {
    if (status == CUFFT_SUCCESS) {
        return;
    }
    std::ostringstream out;
    out << what << ": cuFFT error code " << static_cast<int>(status);
    throw std::runtime_error(out.str());
}

int normalized_modulus_count(int requested) {
    if (requested < 1 || requested > static_cast<int>(kDefaultModuli.size())) {
        throw std::invalid_argument(
            "modulus_count must be in [1, " + std::to_string(kDefaultModuli.size()) + "]"
        );
    }
    return requested;
}

bool is_power_of_two(std::size_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

bool is_contiguous_low_bit_mask(std::uint32_t value) {
    return value != 0u && (value & (value + 1u)) == 0u;
}

int low_bit_mask_width(std::uint32_t value) {
    int width = 0;
    while (value != 0u) {
        value >>= 1u;
        width += 1;
    }
    return width;
}

std::uint32_t low_bit_mask_for_width(int bit_count) {
    if (bit_count <= 0 || bit_count > 32) {
        throw std::invalid_argument("low_bit_mask_for_width bit_count must be in [1, 32]");
    }
    if (bit_count == 32) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    return (std::uint32_t{1} << bit_count) - std::uint32_t{1};
}

std::vector<SplitLimbDescriptor> build_split_limb_descriptors(std::uint32_t packing_mask, int max_limb_bits) {
    if (!is_contiguous_low_bit_mask(packing_mask)) {
        throw std::invalid_argument("multilimb planner requires packing_mask to be a contiguous low-bit mask");
    }
    if (max_limb_bits <= 0 || max_limb_bits > 16) {
        throw std::invalid_argument("max_limb_bits must be in [1, 16]");
    }
    const int active_bits = low_bit_mask_width(packing_mask);
    std::vector<SplitLimbDescriptor> limbs;
    for (int shift = 0; shift < active_bits;) {
        const int bit_count = std::min(max_limb_bits, active_bits - shift);
        limbs.push_back(SplitLimbDescriptor{
            .shift = shift,
            .bit_count = bit_count,
            .mask = low_bit_mask_for_width(bit_count),
        });
        shift += bit_count;
    }
    return limbs;
}

int select_multilimb_max_limb_bits(int active_bits, std::size_t slot_count, bool use_fp64) {
    if (use_fp64) {
        return std::min(active_bits, 16);
    }
    if (active_bits <= 16) {
        return 4;
    }
    constexpr std::uint64_t kSafePerPassMagnitude = 1ull << 16;
    for (int bit_count = 4; bit_count >= 1; --bit_count) {
        const std::uint64_t limb_max = (1ull << bit_count) - 1ull;
        const std::uint64_t per_pass_peak = static_cast<std::uint64_t>(slot_count) * limb_max * limb_max;
        if (per_pass_peak <= kSafePerPassMagnitude) {
            return bit_count;
        }
    }
    return 1;
}

std::vector<SplitPassDescriptor> build_split_pass_descriptors(
    const std::vector<SplitLimbDescriptor>& limbs,
    int active_bits
) {
    std::vector<SplitPassDescriptor> passes;
    for (const auto& lhs_limb : limbs) {
        for (const auto& rhs_limb : limbs) {
            const int accumulation_shift = lhs_limb.shift + rhs_limb.shift;
            if (accumulation_shift >= active_bits) {
                continue;
            }
            passes.push_back(SplitPassDescriptor{
                .lhs_shift = lhs_limb.shift,
                .lhs_mask = lhs_limb.mask,
                .rhs_shift = rhs_limb.shift,
                .rhs_mask = rhs_limb.mask,
                .accumulation_shift = accumulation_shift,
            });
        }
    }
    return passes;
}

__host__ __device__ std::uint64_t chudnovsky_pfactor_abs_value(std::size_t leaf_index) {
    const std::uint64_t k = static_cast<std::uint64_t>(leaf_index) + 1ull;
    return (6ull * k - 5ull) * (2ull * k - 1ull) * (6ull * k - 1ull);
}

__host__ __device__ std::uint64_t chudnovsky_exact_pfactor_abs_value(std::size_t leaf_index) {
    if (leaf_index == 0) {
        return 1ull;
    }
    const std::uint64_t k = static_cast<std::uint64_t>(leaf_index);
    return (6ull * k - 5ull) * (2ull * k - 1ull) * (6ull * k - 1ull);
}

__host__ __device__ std::uint32_t unsigned_u128_digit_base16(unsigned __int128 value, std::size_t slot_index) {
    for (std::size_t shift_count = 0; shift_count < slot_index; ++shift_count) {
        value >>= kCoefficientBaseBits;
    }
    return static_cast<std::uint32_t>(value & static_cast<unsigned __int128>(kCoefficientBaseMask));
}

__host__ __device__ unsigned __int128 chudnovsky_tfactor_abs_value(std::size_t leaf_index) {
    if (leaf_index == 0) {
        return static_cast<unsigned __int128>(kChudnovskyA);
    }

    const unsigned __int128 k = static_cast<unsigned __int128>(static_cast<std::uint64_t>(leaf_index));
    const unsigned __int128 pfactor = static_cast<unsigned __int128>(chudnovsky_exact_pfactor_abs_value(leaf_index));
    return pfactor *
           (static_cast<unsigned __int128>(kChudnovskyA) + static_cast<unsigned __int128>(kChudnovskyB) * k);
}

__host__ __device__ bool chudnovsky_tfactor_is_negative(std::size_t leaf_index) {
    return leaf_index != 0 && (leaf_index & 1ull) != 0ull;
}

__host__ __device__ std::uint32_t chudnovsky_exact_pfactor_digit_base16(std::size_t slot_index, std::size_t leaf_index) {
    constexpr std::size_t kDigitCapacity = sizeof(std::uint64_t) * 8u / static_cast<std::size_t>(kCoefficientBaseBits);
    if (slot_index >= kDigitCapacity) {
        return 0u;
    }
    const std::uint64_t factor = chudnovsky_exact_pfactor_abs_value(leaf_index);
    return static_cast<std::uint32_t>(
        (factor >> (slot_index * static_cast<std::size_t>(kCoefficientBaseBits))) &
        static_cast<std::uint64_t>(kCoefficientBaseMask)
    );
}

__host__ __device__ std::uint32_t chudnovsky_qfactor_digit_base16(std::size_t slot_index, std::size_t leaf_index) {
    if (leaf_index == 0) {
        return slot_index == 0 ? 1u : 0u;
    }

    constexpr std::size_t kConstantDigitCount = 4;
    const std::uint64_t k = static_cast<std::uint64_t>(leaf_index);
    const std::uint64_t scalar = k * k * k;
    std::uint64_t carry = 0;
    for (std::size_t digit_index = 0; digit_index < kConstantDigitCount; ++digit_index) {
        const std::uint32_t constant_digit =
            digit_index == 0 ? 32768u : (digit_index == 1 ? 7559u : (digit_index == 2 ? 56580u : 38u));
        const std::uint64_t value = static_cast<std::uint64_t>(constant_digit) * scalar + carry;
        if (slot_index == digit_index) {
            return static_cast<std::uint32_t>(value & static_cast<std::uint64_t>(kCoefficientBaseMask));
        }
        carry = value >> kCoefficientBaseBits;
    }
    if (slot_index == kConstantDigitCount) {
        return static_cast<std::uint32_t>(carry & static_cast<std::uint64_t>(kCoefficientBaseMask));
    }
    carry >>= kCoefficientBaseBits;
    if (slot_index == kConstantDigitCount + 1ull) {
        return static_cast<std::uint32_t>(carry & static_cast<std::uint64_t>(kCoefficientBaseMask));
    }
    return 0u;
}

__host__ __device__ std::uint32_t chudnovsky_tfactor_digit_base16(
    std::size_t slot_index,
    std::size_t leaf_index,
    std::uint32_t modulus
) {
    const std::uint32_t digit = unsigned_u128_digit_base16(chudnovsky_tfactor_abs_value(leaf_index), slot_index);
    if (!chudnovsky_tfactor_is_negative(leaf_index) || digit == 0u) {
        return digit;
    }
    return modulus - digit;
}

__global__ void initialize_batched_residues_kernel(
    std::uint32_t* lhs,
    std::uint32_t* rhs,
    const std::uint32_t* moduli,
    std::size_t coefficient_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = coefficient_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / coefficient_count;
        const std::size_t coefficient_index = index % coefficient_count;
        const std::size_t batch_index = coefficient_index / slot_count;
        const std::size_t slot_index = coefficient_index % slot_count;
        const std::uint32_t modulus = moduli[modulus_index];

        const std::uint64_t lhs_seed =
            (static_cast<std::uint64_t>(slot_index) + 1ull) * 1315423911ull +
            (static_cast<std::uint64_t>(batch_index) + 3ull) * 2654435761ull +
            (static_cast<std::uint64_t>(modulus_index) + 11ull) * 97ull;
        const std::uint64_t rhs_seed =
            (static_cast<std::uint64_t>(slot_index) + 5ull) * 2246822519ull +
            (static_cast<std::uint64_t>(batch_index) + 7ull) * 3266489917ull +
            (static_cast<std::uint64_t>(modulus_index) + 13ull) * 131ull;

        lhs[index] = static_cast<std::uint32_t>(lhs_seed % modulus);
        rhs[index] = static_cast<std::uint32_t>(rhs_seed % modulus);
    }
}

__global__ void initialize_modulus_major_residues_kernel(
    std::uint32_t* level_values,
    const std::uint32_t* moduli,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * node_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t node_index = reduced % node_count;
        const std::size_t modulus_index = reduced / node_count;
        const std::uint32_t modulus = moduli[modulus_index];

        const std::uint64_t seed =
            (static_cast<std::uint64_t>(slot_index) + 1ull) * 6364136223846793005ull +
            (static_cast<std::uint64_t>(node_index) + 3ull) * 1442695040888963407ull +
            (static_cast<std::uint64_t>(modulus_index) + 11ull) * 3202034522624059733ull;
        level_values[index] = static_cast<std::uint32_t>(seed % modulus);
    }
}

__global__ void initialize_chudnovsky_pfactor_leaves_kernel(
    std::uint32_t* level_values,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * node_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t node_index = reduced % node_count;
        const std::uint64_t factor = chudnovsky_pfactor_abs_value(node_index);
        const std::uint32_t digit =
            slot_index < (sizeof(std::uint64_t) * 8u / static_cast<std::size_t>(kCoefficientBaseBits))
                ? static_cast<std::uint32_t>((factor >> (slot_index * static_cast<std::size_t>(kCoefficientBaseBits))) &
                                             static_cast<std::uint64_t>(kCoefficientBaseMask))
                : 0u;
        level_values[index] = digit;
    }
}

__global__ void initialize_chudnovsky_exact_pfactor_leaves_kernel(
    std::uint32_t* level_values,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * node_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t node_index = reduced % node_count;
        level_values[index] = chudnovsky_exact_pfactor_digit_base16(slot_index, node_index);
    }
}

__global__ void initialize_chudnovsky_qfactor_leaves_kernel(
    std::uint32_t* level_values,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * node_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t node_index = reduced % node_count;
        level_values[index] = chudnovsky_qfactor_digit_base16(slot_index, node_index);
    }
}

__global__ void initialize_chudnovsky_tfactor_leaves_kernel(
    std::uint32_t* level_values,
    const std::uint32_t* moduli,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * node_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t node_index = reduced % node_count;
        const std::size_t modulus_index = reduced / node_count;
        level_values[index] = chudnovsky_tfactor_digit_base16(slot_index, node_index, moduli[modulus_index]);
    }
}

__global__ void pack_balanced_digit_nodes_to_residues_kernel(
    const std::int32_t* digits,
    const std::uint32_t* moduli,
    std::uint32_t* residues,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * node_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t node_index = reduced % node_count;
        const std::size_t modulus_index = reduced / node_count;
        residues[index] = balanced_digit_to_residue(
            digits[node_index * slot_count + slot_index],
            moduli[modulus_index]
        );
    }
}

__global__ void normalize_two_moduli_residues_to_balanced_digits_kernel(
    const std::uint32_t* residues,
    const std::uint32_t* moduli,
    std::int32_t* digits,
    std::size_t node_count,
    std::size_t slot_count,
    std::uint32_t inverse_m0_mod_m1,
    std::uint64_t modulus_product,
    int* overflow_flag
) {
    const std::size_t node_index = static_cast<std::size_t>(blockIdx.x);
    if (node_index >= node_count || threadIdx.x != 0) {
        return;
    }

    const std::uint32_t modulus0 = moduli[0];
    const std::uint32_t modulus1 = moduli[1];
    constexpr long long kBase = static_cast<long long>(1u << kCoefficientBaseBits);
    constexpr long long kHalfBase = kBase / 2ll;
    const long long half_modulus_product = static_cast<long long>(modulus_product / 2ull);

    long long carry = 0;
    for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
        const std::size_t offset = node_index * slot_count + slot_index;
        const std::uint32_t residue0 = residues[offset];
        const std::uint32_t residue1 = residues[node_count * slot_count + offset];
        const std::uint32_t current_mod = residue0 % modulus1;
        const std::uint32_t delta =
            residue1 >= current_mod ? residue1 - current_mod : residue1 + modulus1 - current_mod;
        const std::uint32_t step = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(delta) * static_cast<std::uint64_t>(inverse_m0_mod_m1)) % modulus1
        );

        long long coefficient =
            static_cast<long long>(residue0) + static_cast<long long>(step) * static_cast<long long>(modulus0);
        if (coefficient > half_modulus_product) {
            coefficient -= static_cast<long long>(modulus_product);
        }

        const long long total = coefficient + carry;
        long long remainder = total % kBase;
        if (remainder < 0) {
            remainder += kBase;
        }
        long long next_carry = (total - remainder) / kBase;
        if (remainder > kHalfBase) {
            remainder -= kBase;
            next_carry += 1;
        }

        digits[offset] = static_cast<std::int32_t>(remainder);
        carry = next_carry;
    }

    if (carry != 0) {
        atomicExch(overflow_flag, 1);
    }
}

__device__ __forceinline__ __int128 reconstruct_centered_small_moduli_coefficient_device(
    const std::uint32_t* residues,
    const std::uint32_t* moduli,
    std::size_t residue_stride,
    int modulus_count
) {
    unsigned __int128 value = residues[0];
    unsigned __int128 modulus_product = static_cast<unsigned __int128>(moduli[0]);
    for (int modulus_index = 1; modulus_index < modulus_count; ++modulus_index) {
        const std::uint32_t modulus = moduli[modulus_index];
        const std::uint32_t current_mod = static_cast<std::uint32_t>(value % modulus);
        const std::uint32_t residue = residues[static_cast<std::size_t>(modulus_index) * residue_stride];
        const std::uint32_t inverse =
            mod_inverse_u32_device(static_cast<std::uint32_t>(modulus_product % modulus), modulus);
        const std::uint32_t delta =
            residue >= current_mod ? residue - current_mod : residue + modulus - current_mod;
        const std::uint32_t step = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(delta) * static_cast<std::uint64_t>(inverse)) % modulus
        );
        value += modulus_product * static_cast<unsigned __int128>(step);
        modulus_product *= static_cast<unsigned __int128>(modulus);
    }

    __int128 centered = static_cast<__int128>(value);
    const __int128 half_modulus_product = static_cast<__int128>(modulus_product / 2u);
    if (centered > half_modulus_product) {
        centered -= static_cast<__int128>(modulus_product);
    }
    return centered;
}

__global__ void normalize_small_moduli_residues_to_balanced_digits_kernel(
    const std::uint32_t* residues,
    const std::uint32_t* moduli,
    std::int32_t* digits,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count,
    int* overflow_flag
) {
    const std::size_t node_index = static_cast<std::size_t>(blockIdx.x);
    if (node_index >= node_count || threadIdx.x != 0) {
        return;
    }

    constexpr __int128 kBase = static_cast<__int128>(1u << kCoefficientBaseBits);
    constexpr __int128 kHalfBase = kBase / 2;
    const std::size_t residue_stride = node_count * slot_count;

    __int128 carry = 0;
    for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
        const std::size_t offset = node_index * slot_count + slot_index;
        const __int128 coefficient = reconstruct_centered_small_moduli_coefficient_device(
            residues + offset,
            moduli,
            residue_stride,
            modulus_count
        );
        const __int128 total = coefficient + carry;
        __int128 remainder = total % kBase;
        if (remainder < 0) {
            remainder += kBase;
        }
        __int128 next_carry = (total - remainder) / kBase;
        if (remainder > kHalfBase) {
            remainder -= kBase;
            next_carry += 1;
        }

        digits[offset] = static_cast<std::int32_t>(remainder);
        carry = next_carry;
    }

    if (carry != 0) {
        atomicExch(overflow_flag, 1);
    }
}

__global__ void packed_pointwise_add_mod_kernel(
    const std::uint32_t* lhs,
    const std::uint32_t* rhs,
    const std::uint32_t* moduli,
    std::uint32_t* out,
    std::size_t coefficient_count,
    int modulus_count
) {
    const std::size_t total = coefficient_count * static_cast<std::size_t>(modulus_count);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t modulus_index = index / coefficient_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::uint32_t sum = lhs[index] + rhs[index];
        out[index] = sum >= modulus ? sum - modulus : sum;
    }
}

__global__ void pack_adjacent_pairs_modulus_major_kernel(
    const std::uint32_t* level_values,
    std::uint32_t* packed_pairs,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t merge_count = node_count / 2ull;
    const std::size_t total =
        static_cast<std::size_t>(modulus_count) * merge_count * 2ull * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t child_index = reduced % 2ull;
        const std::size_t reduced_2 = reduced / 2ull;
        const std::size_t merge_index = reduced_2 % merge_count;
        const std::size_t modulus_index = reduced_2 / merge_count;
        const std::size_t node_index = merge_index * 2ull + child_index;
        const std::size_t source_index =
            ((modulus_index * node_count) + node_index) * slot_count + slot_index;
        packed_pairs[index] = level_values[source_index];
    }
}

__global__ void reduce_packed_pairs_add_mod_kernel(
    const std::uint32_t* packed_pairs,
    const std::uint32_t* moduli,
    std::uint32_t* parent_values,
    std::size_t merge_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::size_t packed_base =
            (((modulus_index * merge_count) + merge_index) * 2ull) * slot_count + slot_index;
        const std::uint32_t lhs = packed_pairs[packed_base];
        const std::uint32_t rhs = packed_pairs[packed_base + slot_count];
        const std::uint32_t sum = lhs + rhs;
        parent_values[index] = sum >= modulus ? sum - modulus : sum;
    }
}

__global__ void initialize_fft_inputs_kernel(
    cufftComplex* lhs,
    cufftComplex* rhs,
    std::size_t batch_count,
    std::size_t fft_length
) {
    const std::size_t total = batch_count * fft_length;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t sample_index = index % fft_length;
        const std::size_t batch_index = index / fft_length;

        const float lhs_value = static_cast<float>(
            static_cast<int>((sample_index * 13ull + batch_index * 7ull + 1ull) % 17ull) - 8
        );
        const float rhs_value = static_cast<float>(
            static_cast<int>((sample_index * 5ull + batch_index * 11ull + 3ull) % 19ull) - 9
        );

        lhs[index].x = lhs_value;
        lhs[index].y = 0.0f;
        rhs[index].x = rhs_value;
        rhs[index].y = 0.0f;
    }
}

__global__ void pack_modulus_major_pairs_to_fft_inputs_kernel(
    const std::uint32_t* level_values,
    cufftComplex* lhs,
    cufftComplex* rhs,
    std::size_t node_count,
    std::size_t slot_count,
    std::size_t bridge_modulus_index,
    std::uint32_t packing_mask
) {
    const std::size_t merge_count = node_count / 2ull;
    const std::size_t total = merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t merge_index = index / slot_count;
        const std::size_t lhs_node = merge_index * 2ull;
        const std::size_t rhs_node = lhs_node + 1ull;
        const std::size_t lhs_source =
            ((bridge_modulus_index * node_count) + lhs_node) * slot_count + slot_index;
        const std::size_t rhs_source =
            ((bridge_modulus_index * node_count) + rhs_node) * slot_count + slot_index;

        lhs[index].x = static_cast<float>(level_values[lhs_source] & packing_mask);
        lhs[index].y = 0.0f;
        rhs[index].x = static_cast<float>(level_values[rhs_source] & packing_mask);
        rhs[index].y = 0.0f;
    }
}

__global__ void pack_grouped_modulus_major_pairs_to_fft_inputs_kernel(
    const std::uint32_t* level_values,
    cufftComplex* lhs,
    cufftComplex* rhs,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count,
    std::uint32_t packing_mask
) {
    const std::size_t merge_count = node_count / 2ull;
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::size_t lhs_node = merge_index * 2ull;
        const std::size_t rhs_node = lhs_node + 1ull;
        const std::size_t lhs_source =
            ((modulus_index * node_count) + lhs_node) * slot_count + slot_index;
        const std::size_t rhs_source =
            ((modulus_index * node_count) + rhs_node) * slot_count + slot_index;

        lhs[index].x = static_cast<float>(level_values[lhs_source] & packing_mask);
        lhs[index].y = 0.0f;
        rhs[index].x = static_cast<float>(level_values[rhs_source] & packing_mask);
        rhs[index].y = 0.0f;
    }
}

__global__ void pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel(
    const std::uint32_t* level_values,
    cufftComplex* lhs,
    cufftComplex* rhs,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count,
    int lhs_shift,
    std::uint32_t lhs_mask,
    int rhs_shift,
    std::uint32_t rhs_mask
) {
    const std::size_t merge_count = node_count / 2ull;
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::size_t lhs_node = merge_index * 2ull;
        const std::size_t rhs_node = lhs_node + 1ull;
        const std::size_t lhs_source =
            ((modulus_index * node_count) + lhs_node) * slot_count + slot_index;
        const std::size_t rhs_source =
            ((modulus_index * node_count) + rhs_node) * slot_count + slot_index;

        lhs[index].x = static_cast<float>((level_values[lhs_source] >> lhs_shift) & lhs_mask);
        lhs[index].y = 0.0f;
        rhs[index].x = static_cast<float>((level_values[rhs_source] >> rhs_shift) & rhs_mask);
        rhs[index].y = 0.0f;
    }
}

__global__ void pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel_fp64(
    const std::uint32_t* level_values,
    cufftDoubleComplex* lhs,
    cufftDoubleComplex* rhs,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count,
    int lhs_shift,
    std::uint32_t lhs_mask,
    int rhs_shift,
    std::uint32_t rhs_mask
) {
    const std::size_t merge_count = node_count / 2ull;
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::size_t lhs_node = merge_index * 2ull;
        const std::size_t rhs_node = lhs_node + 1ull;
        const std::size_t lhs_source =
            ((modulus_index * node_count) + lhs_node) * slot_count + slot_index;
        const std::size_t rhs_source =
            ((modulus_index * node_count) + rhs_node) * slot_count + slot_index;

        lhs[index].x = static_cast<double>((level_values[lhs_source] >> lhs_shift) & lhs_mask);
        lhs[index].y = 0.0;
        rhs[index].x = static_cast<double>((level_values[rhs_source] >> rhs_shift) & rhs_mask);
        rhs[index].y = 0.0;
    }
}

__global__ void pack_grouped_modulus_major_cross_pairs_to_fft_limb_inputs_kernel_fp64(
    const std::uint32_t* lhs_level_values,
    const std::uint32_t* rhs_level_values,
    cufftDoubleComplex* lhs,
    cufftDoubleComplex* rhs,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count,
    int lhs_shift,
    std::uint32_t lhs_mask,
    int rhs_shift,
    std::uint32_t rhs_mask
) {
    const std::size_t merge_count = node_count / 2ull;
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::size_t lhs_node = merge_index * 2ull;
        const std::size_t rhs_node = lhs_node + 1ull;
        const std::size_t lhs_source =
            ((modulus_index * node_count) + lhs_node) * slot_count + slot_index;
        const std::size_t rhs_source =
            ((modulus_index * node_count) + rhs_node) * slot_count + slot_index;

        lhs[index].x = static_cast<double>((lhs_level_values[lhs_source] >> lhs_shift) & lhs_mask);
        lhs[index].y = 0.0;
        rhs[index].x = static_cast<double>((rhs_level_values[rhs_source] >> rhs_shift) & rhs_mask);
        rhs[index].y = 0.0;
    }
}

__global__ void complex_pointwise_multiply_kernel(
    const cufftComplex* lhs,
    const cufftComplex* rhs,
    cufftComplex* out,
    std::size_t total
) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const cufftComplex a = lhs[index];
        const cufftComplex b = rhs[index];
        out[index].x = a.x * b.x - a.y * b.y;
        out[index].y = a.x * b.y + a.y * b.x;
    }
}

__global__ void complex_pointwise_multiply_kernel_fp64(
    const cufftDoubleComplex* lhs,
    const cufftDoubleComplex* rhs,
    cufftDoubleComplex* out,
    std::size_t total
) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const cufftDoubleComplex a = lhs[index];
        const cufftDoubleComplex b = rhs[index];
        out[index].x = a.x * b.x - a.y * b.y;
        out[index].y = a.x * b.y + a.y * b.x;
    }
}

__global__ void scale_complex_kernel(cufftComplex* values, float scale, std::size_t total) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        values[index].x *= scale;
        values[index].y *= scale;
    }
}

__global__ void scale_complex_kernel_fp64(cufftDoubleComplex* values, double scale, std::size_t total) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        values[index].x *= scale;
        values[index].y *= scale;
    }
}

__global__ void project_grouped_fft_output_to_level_values_kernel(
    const cufftComplex* values,
    std::uint32_t* next_level_values,
    std::size_t merge_count,
    std::size_t slot_count,
    int modulus_count,
    std::uint32_t packing_mask
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::size_t batch_index = modulus_index * merge_count + merge_index;
        const float real_value = values[batch_index * slot_count + slot_index].x;
        const long long rounded = llroundf(real_value);
        next_level_values[index] = static_cast<std::uint32_t>(rounded) & packing_mask;
    }
}

__global__ void accumulate_grouped_fft_output_to_level_values_kernel_fp64(
    const cufftDoubleComplex* values,
    std::uint32_t* next_level_values,
    std::size_t merge_count,
    std::size_t slot_count,
    int modulus_count,
    int accumulation_shift,
    std::uint32_t packing_mask
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::size_t batch_index = modulus_index * merge_count + merge_index;
        const double real_value = values[batch_index * slot_count + slot_index].x;
        const long long rounded = llround(real_value);
        const unsigned long long contribution =
            rounded <= 0
                ? 0ull
                : ((static_cast<unsigned long long>(rounded) << accumulation_shift) &
                   static_cast<unsigned long long>(packing_mask));
        next_level_values[index] =
            (next_level_values[index] + static_cast<std::uint32_t>(contribution)) & packing_mask;
    }
}

__global__ void accumulate_grouped_fft_output_to_level_values_mod_kernel_fp64(
    const cufftDoubleComplex* values,
    std::uint32_t* next_level_values,
    const std::uint32_t* moduli,
    const std::uint32_t* pass_weights_mod,
    std::size_t merge_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::size_t batch_index = modulus_index * merge_count + merge_index;
        const std::uint32_t modulus = moduli[modulus_index];
        const std::uint32_t weight = pass_weights_mod[modulus_index];
        const double real_value = values[batch_index * slot_count + slot_index].x;
        const long long rounded = llround(real_value);
        long long reduced_value = rounded % static_cast<long long>(modulus);
        if (reduced_value < 0) {
            reduced_value += static_cast<long long>(modulus);
        }
        const std::uint64_t contribution =
            (static_cast<std::uint64_t>(reduced_value) * static_cast<std::uint64_t>(weight)) %
            static_cast<std::uint64_t>(modulus);
        const std::uint64_t next =
            static_cast<std::uint64_t>(next_level_values[index]) + contribution;
        next_level_values[index] = static_cast<std::uint32_t>(next % static_cast<std::uint64_t>(modulus));
    }
}

__global__ void accumulate_grouped_fft_output_to_level_values_kernel(
    const cufftComplex* values,
    std::uint32_t* next_level_values,
    std::size_t merge_count,
    std::size_t slot_count,
    int modulus_count,
    int accumulation_shift,
    std::uint32_t packing_mask
) {
    const std::size_t total = static_cast<std::size_t>(modulus_count) * merge_count * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t merge_index = reduced % merge_count;
        const std::size_t modulus_index = reduced / merge_count;
        const std::size_t batch_index = modulus_index * merge_count + merge_index;
        const float real_value = values[batch_index * slot_count + slot_index].x;
        const long long rounded = llroundf(real_value);
        const unsigned long long contribution =
            rounded <= 0
                ? 0ull
                : ((static_cast<unsigned long long>(rounded) << accumulation_shift) &
                   static_cast<unsigned long long>(packing_mask));
        next_level_values[index] =
            (next_level_values[index] + static_cast<std::uint32_t>(contribution)) & packing_mask;
    }
}

__global__ void initialize_node_major_residues_kernel(
    std::uint32_t* nodes,
    const std::uint32_t* moduli,
    std::size_t node_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total = node_count * static_cast<std::size_t>(modulus_count) * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t modulus_index = reduced % static_cast<std::size_t>(modulus_count);
        const std::size_t node_index = reduced / static_cast<std::size_t>(modulus_count);
        const std::uint32_t modulus = moduli[modulus_index];

        const std::uint64_t seed =
            (static_cast<std::uint64_t>(slot_index) + 1ull) * 14029467366897019727ull +
            (static_cast<std::uint64_t>(node_index) + 5ull) * 11400714819323198485ull +
            (static_cast<std::uint64_t>(modulus_index) + 17ull) * 7046029254386353131ull;
        nodes[index] = static_cast<std::uint32_t>(seed % modulus);
    }
}

__global__ void pack_node_major_pairs_kernel(
    const std::uint32_t* node_major,
    std::uint32_t* packed_pairs,
    std::size_t merge_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total =
        static_cast<std::size_t>(modulus_count) * merge_count * 2ull * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t child_index = reduced % 2ull;
        const std::size_t reduced_2 = reduced / 2ull;
        const std::size_t merge_index = reduced_2 % merge_count;
        const std::size_t modulus_index = reduced_2 / merge_count;
        const std::size_t node_index = merge_index * 2ull + child_index;
        const std::size_t source_index =
            ((node_index * static_cast<std::size_t>(modulus_count)) + modulus_index) * slot_count + slot_index;
        packed_pairs[index] = node_major[source_index];
    }
}

__global__ void unpack_node_major_pairs_kernel(
    const std::uint32_t* packed_pairs,
    std::uint32_t* node_major,
    std::size_t merge_count,
    std::size_t slot_count,
    int modulus_count
) {
    const std::size_t total =
        static_cast<std::size_t>(modulus_count) * merge_count * 2ull * slot_count;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t slot_index = index % slot_count;
        const std::size_t reduced = index / slot_count;
        const std::size_t child_index = reduced % 2ull;
        const std::size_t reduced_2 = reduced / 2ull;
        const std::size_t merge_index = reduced_2 % merge_count;
        const std::size_t modulus_index = reduced_2 / merge_count;
        const std::size_t node_index = merge_index * 2ull + child_index;
        const std::size_t target_index =
            ((node_index * static_cast<std::size_t>(modulus_count)) + modulus_index) * slot_count + slot_index;
        node_major[target_index] = packed_pairs[index];
    }
}

std::uint32_t expected_lhs_value(std::size_t slot_index, std::size_t batch_index, std::size_t modulus_index) {
    const std::uint32_t modulus = kDefaultModuli[modulus_index];
    const std::uint64_t seed =
        (static_cast<std::uint64_t>(slot_index) + 1ull) * 1315423911ull +
        (static_cast<std::uint64_t>(batch_index) + 3ull) * 2654435761ull +
        (static_cast<std::uint64_t>(modulus_index) + 11ull) * 97ull;
    return static_cast<std::uint32_t>(seed % modulus);
}

std::uint32_t expected_modulus_major_value(std::size_t slot_index, std::size_t node_index, std::size_t modulus_index) {
    const std::uint32_t modulus = kDefaultModuli[modulus_index];
    const std::uint64_t seed =
        (static_cast<std::uint64_t>(slot_index) + 1ull) * 6364136223846793005ull +
        (static_cast<std::uint64_t>(node_index) + 3ull) * 1442695040888963407ull +
        (static_cast<std::uint64_t>(modulus_index) + 11ull) * 3202034522624059733ull;
    return static_cast<std::uint32_t>(seed % modulus);
}

std::uint32_t expected_chudnovsky_pfactor_digit(std::size_t slot_index, std::size_t node_index) {
    constexpr std::size_t kDigitCapacity = sizeof(std::uint64_t) * 8u / static_cast<std::size_t>(kCoefficientBaseBits);
    if (slot_index >= kDigitCapacity) {
        return 0u;
    }
    const std::uint64_t factor = chudnovsky_pfactor_abs_value(node_index);
    return static_cast<std::uint32_t>(
        (factor >> (slot_index * static_cast<std::size_t>(kCoefficientBaseBits))) &
        static_cast<std::uint64_t>(kCoefficientBaseMask)
    );
}

std::uint32_t expected_chudnovsky_qfactor_digit(std::size_t slot_index, std::size_t node_index) {
    return chudnovsky_qfactor_digit_base16(slot_index, node_index);
}

std::uint32_t expected_chudnovsky_exact_pfactor_digit(std::size_t slot_index, std::size_t node_index) {
    return chudnovsky_exact_pfactor_digit_base16(slot_index, node_index);
}

std::uint32_t expected_chudnovsky_tfactor_digit(
    std::size_t slot_index,
    std::size_t node_index,
    std::size_t modulus_index
) {
    return chudnovsky_tfactor_digit_base16(slot_index, node_index, kDefaultModuli[modulus_index]);
}

std::uint32_t evaluate_coefficient_vector_mod(
    const std::uint32_t* coefficients,
    std::size_t slot_count,
    std::uint32_t modulus
) {
    const std::uint64_t base_mod = (std::uint64_t{1} << kCoefficientBaseBits) % static_cast<std::uint64_t>(modulus);
    std::uint64_t accum = 0;
    for (std::size_t reverse_index = slot_count; reverse_index > 0; --reverse_index) {
        accum = (accum * base_mod + static_cast<std::uint64_t>(coefficients[reverse_index - 1])) %
                static_cast<std::uint64_t>(modulus);
    }
    return static_cast<std::uint32_t>(accum);
}

std::uint32_t expected_chudnovsky_pfactor_product_mod(std::size_t node_count, std::uint32_t modulus) {
    std::uint64_t product = 1u % static_cast<std::uint64_t>(modulus);
    for (std::size_t node_index = 0; node_index < node_count; ++node_index) {
        product = (product * (chudnovsky_pfactor_abs_value(node_index) % static_cast<std::uint64_t>(modulus))) %
                  static_cast<std::uint64_t>(modulus);
    }
    return static_cast<std::uint32_t>(product);
}

std::uint32_t expected_chudnovsky_qfactor_product_mod(std::size_t node_count, std::uint32_t modulus) {
    std::uint64_t product = 1u % static_cast<std::uint64_t>(modulus);
    const std::uint64_t constant_mod = kChudnovskyC3Over24 % static_cast<std::uint64_t>(modulus);
    for (std::size_t node_index = 1; node_index < node_count; ++node_index) {
        const std::uint64_t k = static_cast<std::uint64_t>(node_index);
        const std::uint64_t k_mod = k % static_cast<std::uint64_t>(modulus);
        const std::uint64_t cubic_mod =
            (k_mod * k_mod % static_cast<std::uint64_t>(modulus)) * k_mod % static_cast<std::uint64_t>(modulus);
        const std::uint64_t q_mod = cubic_mod * constant_mod % static_cast<std::uint64_t>(modulus);
        product = product * q_mod % static_cast<std::uint64_t>(modulus);
    }
    return static_cast<std::uint32_t>(product);
}

std::size_t coefficient_digit_length_base16(std::uint64_t value) {
    std::size_t length = 0;
    do {
        value >>= kCoefficientBaseBits;
        length += 1;
    } while (value != 0u);
    return length;
}

std::size_t chudnovsky_qfactor_digit_length_base16(std::size_t leaf_index) {
    if (leaf_index == 0) {
        return 1;
    }

    const std::uint64_t k = static_cast<std::uint64_t>(leaf_index);
    const unsigned __int128 value =
        static_cast<unsigned __int128>(k) * static_cast<unsigned __int128>(k) *
        static_cast<unsigned __int128>(k) * static_cast<unsigned __int128>(kChudnovskyC3Over24);
    std::size_t length = 0;
    unsigned __int128 reduced = value;
    do {
        reduced >>= kCoefficientBaseBits;
        length += 1;
    } while (reduced != 0u);
    return length;
}

std::size_t coefficient_digit_length_base16_u128(unsigned __int128 value) {
    std::size_t length = 0;
    unsigned __int128 reduced = value;
    do {
        reduced >>= kCoefficientBaseBits;
        length += 1;
    } while (reduced != 0u);
    return length;
}

std::uint32_t mod_reduce_u128(unsigned __int128 value, std::uint32_t modulus) {
    return static_cast<std::uint32_t>(value % static_cast<unsigned __int128>(modulus));
}

std::uint32_t expected_chudnovsky_tfactor_leaf_mod(std::size_t node_index, std::uint32_t modulus) {
    const std::uint32_t abs_mod = mod_reduce_u128(chudnovsky_tfactor_abs_value(node_index), modulus);
    if (!chudnovsky_tfactor_is_negative(node_index) || abs_mod == 0u) {
        return abs_mod;
    }
    return modulus - abs_mod;
}

struct ExactChudnovskyPqtRootMod {
    std::uint32_t p = 0;
    std::uint32_t q = 0;
    std::uint32_t t = 0;
};

ExactChudnovskyPqtRootMod expected_chudnovsky_exact_pqt_root_mod(std::size_t node_count, std::uint32_t modulus) {
    std::vector<std::uint32_t> p(node_count, 0u);
    std::vector<std::uint32_t> q(node_count, 0u);
    std::vector<std::uint32_t> t(node_count, 0u);
    for (std::size_t node_index = 0; node_index < node_count; ++node_index) {
        p[node_index] = chudnovsky_exact_pfactor_abs_value(node_index) % modulus;
        q[node_index] =
            node_index == 0
                ? 1u % modulus
                : mod_reduce_u128(
                      static_cast<unsigned __int128>(static_cast<std::uint64_t>(node_index)) *
                          static_cast<unsigned __int128>(static_cast<std::uint64_t>(node_index)) *
                          static_cast<unsigned __int128>(static_cast<std::uint64_t>(node_index)) *
                          static_cast<unsigned __int128>(kChudnovskyC3Over24),
                      modulus
                  );
        t[node_index] = expected_chudnovsky_tfactor_leaf_mod(node_index, modulus);
    }

    std::size_t current_nodes = node_count;
    while (current_nodes > 1) {
        const std::size_t merge_count = current_nodes / 2ull;
        for (std::size_t merge_index = 0; merge_index < merge_count; ++merge_index) {
            const std::size_t lhs = merge_index * 2ull;
            const std::size_t rhs = lhs + 1ull;
            const std::uint64_t p_parent =
                static_cast<std::uint64_t>(p[lhs]) * static_cast<std::uint64_t>(p[rhs]) %
                static_cast<std::uint64_t>(modulus);
            const std::uint64_t q_parent =
                static_cast<std::uint64_t>(q[lhs]) * static_cast<std::uint64_t>(q[rhs]) %
                static_cast<std::uint64_t>(modulus);
            const std::uint64_t t_parent =
                (static_cast<std::uint64_t>(t[lhs]) * static_cast<std::uint64_t>(q[rhs]) +
                 static_cast<std::uint64_t>(p[lhs]) * static_cast<std::uint64_t>(t[rhs])) %
                static_cast<std::uint64_t>(modulus);
            p[merge_index] = static_cast<std::uint32_t>(p_parent);
            q[merge_index] = static_cast<std::uint32_t>(q_parent);
            t[merge_index] = static_cast<std::uint32_t>(t_parent);
        }
        current_nodes = merge_count;
    }

    return ExactChudnovskyPqtRootMod{
        .p = p[0],
        .q = q[0],
        .t = t[0],
    };
}

float expected_fft_lhs_value(std::size_t sample_index, std::size_t batch_index) {
    return static_cast<float>(
        static_cast<int>((sample_index * 13ull + batch_index * 7ull + 1ull) % 17ull) - 8
    );
}

float expected_fft_rhs_value(std::size_t sample_index, std::size_t batch_index) {
    return static_cast<float>(
        static_cast<int>((sample_index * 5ull + batch_index * 11ull + 3ull) % 19ull) - 9
    );
}

float expected_bridge_residue_value(
    std::size_t slot_index,
    std::size_t node_index,
    std::size_t modulus_index,
    std::uint32_t packing_mask
) {
    return static_cast<float>(
        expected_modulus_major_value(slot_index, node_index, modulus_index) & packing_mask
    );
}

std::uint32_t expected_bridge_residue_value_u32(
    std::size_t slot_index,
    std::size_t node_index,
    std::size_t modulus_index,
    std::uint32_t packing_mask
) {
    return expected_modulus_major_value(slot_index, node_index, modulus_index) & packing_mask;
}

std::uint32_t expected_rhs_value(std::size_t slot_index, std::size_t batch_index, std::size_t modulus_index) {
    const std::uint32_t modulus = kDefaultModuli[modulus_index];
    const std::uint64_t seed =
        (static_cast<std::uint64_t>(slot_index) + 5ull) * 2246822519ull +
        (static_cast<std::uint64_t>(batch_index) + 7ull) * 3266489917ull +
        (static_cast<std::uint64_t>(modulus_index) + 13ull) * 131ull;
    return static_cast<std::uint32_t>(seed % modulus);
}

std::uint32_t expected_node_major_value(std::size_t slot_index, std::size_t node_index, std::size_t modulus_index) {
    const std::uint32_t modulus = kDefaultModuli[modulus_index];
    const std::uint64_t seed =
        (static_cast<std::uint64_t>(slot_index) + 1ull) * 14029467366897019727ull +
        (static_cast<std::uint64_t>(node_index) + 5ull) * 11400714819323198485ull +
        (static_cast<std::uint64_t>(modulus_index) + 17ull) * 7046029254386353131ull;
    return static_cast<std::uint32_t>(seed % modulus);
}

}  // namespace

ThroughputPlanReport plan_throughput_mainline() {
    ThroughputPlanReport report;
    report.workspace_name = "project2_gpu_throughput_mainline";
    report.priority_order = "throughput_first_device_residency_first_batching_first_semantics_later";
    report.primary_layout = "residues[modulus][batch][slot]";
    report.primary_batch_axis = "same_shape_merge_group";
    report.fft_backbone_target = "cufft_or_equivalent_batched_large_fft";
    report.merge_scheduler_target = "persistent_level_batched_binary_split_merge_tree";
    report.current_stage = "T4_frozen_research_snapshot_not_active_performance_mainline";
    report.current_benchmark =
        "frozen_after_exact_grouped_scheduler_and_staged_closure_proof_of_feasibility_through_5000_digit_smokes";
    report.next_blocker =
        "route_frozen_due_large_gap_to_stronger_full_pi_routes_resume_only_with_architectural_restart";
    report.ok = true;
    return report;
}

void print_throughput_plan_report(std::ostream& out, const ThroughputPlanReport& report) {
    out << "throughput_plan_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "workspace_name=" << report.workspace_name << '\n';
    out << "priority_order=" << report.priority_order << '\n';
    out << "primary_layout=" << report.primary_layout << '\n';
    out << "primary_batch_axis=" << report.primary_batch_axis << '\n';
    out << "fft_backbone_target=" << report.fft_backbone_target << '\n';
    out << "merge_scheduler_target=" << report.merge_scheduler_target << '\n';
    out << "current_stage=" << report.current_stage << '\n';
    out << "current_benchmark=" << report.current_benchmark << '\n';
    out << "next_blocker=" << report.next_blocker << '\n';
}

PackedPointwiseAddReport run_packed_pointwise_add_smoke(const PackedPointwiseAddConfig& config) {
    if (config.batch_count == 0 || config.slot_count == 0) {
        throw std::invalid_argument("batch_count and slot_count must be positive");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    const std::size_t coefficient_count = config.batch_count * config.slot_count;
    const std::size_t residue_value_count = coefficient_count * static_cast<std::size_t>(modulus_count);
    const std::size_t residue_bytes = sizeof(std::uint32_t) * residue_value_count;

    std::uint32_t* d_lhs = nullptr;
    std::uint32_t* d_rhs = nullptr;
    std::uint32_t* d_out = nullptr;
    std::uint32_t* d_moduli = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto cleanup = [&]() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_moduli);
        cudaFree(d_out);
        cudaFree(d_rhs);
        cudaFree(d_lhs);
    };

    PackedPointwiseAddReport report;
    report.layout = "residues[modulus][batch][slot]";
    report.operation = "packed_pointwise_add_mod_prime_residues";
    report.batch_count = config.batch_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.coefficient_count = coefficient_count;
    report.residue_value_count = residue_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_lhs), residue_bytes), "cudaMalloc d_lhs");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_rhs), residue_bytes), "cudaMalloc d_rhs");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), residue_bytes), "cudaMalloc d_out");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc d_moduli"
        );
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy d_moduli"
        );
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

        initialize_batched_residues_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
            d_lhs,
            d_rhs,
            d_moduli,
            coefficient_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_batched_residues_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after init");

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            packed_pointwise_add_mod_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
                d_lhs,
                d_rhs,
                d_moduli,
                d_out,
                coefficient_count,
                modulus_count
            );
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

        check_cuda(cudaEventRecord(start), "cudaEventRecord cold start");
        packed_pointwise_add_mod_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
            d_lhs,
            d_rhs,
            d_moduli,
            d_out,
            coefficient_count,
            modulus_count
        );
        check_cuda(cudaEventRecord(stop), "cudaEventRecord cold stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize cold stop");
        float cold_elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&cold_elapsed_ms, start, stop), "cudaEventElapsedTime cold");
        report.cold_kernel_ms = static_cast<double>(cold_elapsed_ms);

        double total_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            float elapsed_ms = 0.0f;
            check_cuda(cudaEventRecord(start), "cudaEventRecord start");
            packed_pointwise_add_mod_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
                d_lhs,
                d_rhs,
                d_moduli,
                d_out,
                coefficient_count,
                modulus_count
            );
            check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
            total_ms += static_cast<double>(elapsed_ms);
        }

        report.avg_kernel_ms = total_ms / static_cast<double>(config.measured_iterations);
        if (report.avg_kernel_ms > 0.0) {
            report.residue_values_per_second =
                static_cast<double>(residue_value_count) * 1000.0 / report.avg_kernel_ms;
            report.coefficients_per_second =
                static_cast<double>(coefficient_count) * 1000.0 / report.avg_kernel_ms;
            report.device_bytes_per_second =
                static_cast<double>(residue_bytes * 3ull) * 1000.0 / report.avg_kernel_ms;
        }

        std::vector<std::uint32_t> samples(report.verification_sample_count);
        bool verification_ok = true;
        for (int modulus_index = 0; modulus_index < modulus_count; ++modulus_index) {
            const std::size_t offset = static_cast<std::size_t>(modulus_index) * coefficient_count;
            check_cuda(
                cudaMemcpy(
                    samples.data(),
                    d_out + offset,
                    sizeof(std::uint32_t) * report.verification_sample_count,
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy verification samples"
            );
            for (std::size_t sample_index = 0; sample_index < report.verification_sample_count; ++sample_index) {
                const std::uint32_t modulus = kDefaultModuli[modulus_index];
                const std::uint32_t lhs_value = expected_lhs_value(sample_index, 0, static_cast<std::size_t>(modulus_index));
                const std::uint32_t rhs_value = expected_rhs_value(sample_index, 0, static_cast<std::size_t>(modulus_index));
                const std::uint32_t expected = lhs_value + rhs_value >= modulus ? lhs_value + rhs_value - modulus : lhs_value + rhs_value;
                verification_ok = verification_ok && samples[sample_index] == expected;
                report.verified_values += 1;
            }
        }

        report.ok = verification_ok;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

void print_packed_pointwise_add_report(std::ostream& out, const PackedPointwiseAddReport& report) {
    out << "packed_pointwise_add_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "layout=" << report.layout << '\n';
    out << "operation=" << report.operation << '\n';
    out << "batch_count=" << report.batch_count << '\n';
    out << "slot_count=" << report.slot_count << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "warmup_iterations=" << report.warmup_iterations << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "coefficient_count=" << report.coefficient_count << '\n';
    out << "residue_value_count=" << report.residue_value_count << '\n';
    out << "cold_kernel_ms=" << report.cold_kernel_ms << '\n';
    out << "avg_kernel_ms=" << report.avg_kernel_ms << '\n';
    out << "residue_values_per_second=" << report.residue_values_per_second << '\n';
    out << "coefficients_per_second=" << report.coefficients_per_second << '\n';
    out << "device_bytes_per_second=" << report.device_bytes_per_second << '\n';
    out << "verification_sample_count=" << report.verification_sample_count << '\n';
    out << "verified_values=" << report.verified_values << '\n';
}

DevicePairPackReport run_device_pair_pack_smoke(const DevicePairPackConfig& config) {
    if (config.merge_count == 0 || config.slot_count == 0) {
        throw std::invalid_argument("merge_count and slot_count must be positive");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    const std::size_t node_count = config.merge_count * 2ull;
    const std::size_t residue_value_count =
        node_count * static_cast<std::size_t>(modulus_count) * config.slot_count;
    const std::size_t residue_bytes = sizeof(std::uint32_t) * residue_value_count;
    const std::size_t sample_node_count = std::min<std::size_t>(4, node_count);

    std::uint32_t* d_source = nullptr;
    std::uint32_t* d_packed = nullptr;
    std::uint32_t* d_reconstructed = nullptr;
    std::uint32_t* d_moduli = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto cleanup = [&]() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_moduli);
        cudaFree(d_reconstructed);
        cudaFree(d_packed);
        cudaFree(d_source);
    };

    DevicePairPackReport report;
    report.source_layout = "nodes[node][modulus][slot]";
    report.packed_layout = "packed[modulus][merge][child][slot]";
    report.operation = "device_pair_pack_unpack_same_shape_merge_group";
    report.merge_count = config.merge_count;
    report.node_count = node_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.residue_value_count = residue_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_source), residue_bytes), "cudaMalloc d_source");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_packed), residue_bytes), "cudaMalloc d_packed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_reconstructed), residue_bytes), "cudaMalloc d_reconstructed");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc d_moduli"
        );
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy d_moduli"
        );
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

        initialize_node_major_residues_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
            d_source,
            d_moduli,
            node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_node_major_residues_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after init");

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            pack_node_major_pairs_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
                d_source,
                d_packed,
                config.merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "pack_node_major_pairs_kernel warmup launch");
            unpack_node_major_pairs_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
                d_packed,
                d_reconstructed,
                config.merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "unpack_node_major_pairs_kernel warmup launch");
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

        check_cuda(cudaEventRecord(start), "cudaEventRecord cold pack start");
        pack_node_major_pairs_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
            d_source,
            d_packed,
            config.merge_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "pack_node_major_pairs_kernel cold launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord cold pack stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize cold pack stop");
        float cold_pack_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&cold_pack_ms, start, stop), "cudaEventElapsedTime cold pack");
        report.cold_pack_ms = static_cast<double>(cold_pack_ms);

        check_cuda(cudaEventRecord(start), "cudaEventRecord cold unpack start");
        unpack_node_major_pairs_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
            d_packed,
            d_reconstructed,
            config.merge_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "unpack_node_major_pairs_kernel cold launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord cold unpack stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize cold unpack stop");
        float cold_unpack_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&cold_unpack_ms, start, stop), "cudaEventElapsedTime cold unpack");
        report.cold_unpack_ms = static_cast<double>(cold_unpack_ms);
        report.cold_roundtrip_ms = report.cold_pack_ms + report.cold_unpack_ms;

        double total_pack_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            float elapsed_ms = 0.0f;
            check_cuda(cudaEventRecord(start), "cudaEventRecord pack start");
            pack_node_major_pairs_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
                d_source,
                d_packed,
                config.merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "pack_node_major_pairs_kernel launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord pack stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize pack stop");
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime pack");
            total_pack_ms += static_cast<double>(elapsed_ms);
        }
        report.avg_pack_ms = total_pack_ms / static_cast<double>(config.measured_iterations);

        double total_unpack_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            float elapsed_ms = 0.0f;
            check_cuda(cudaEventRecord(start), "cudaEventRecord unpack start");
            unpack_node_major_pairs_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
                d_packed,
                d_reconstructed,
                config.merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "unpack_node_major_pairs_kernel launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord unpack stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize unpack stop");
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime unpack");
            total_unpack_ms += static_cast<double>(elapsed_ms);
        }
        report.avg_unpack_ms = total_unpack_ms / static_cast<double>(config.measured_iterations);

        double total_roundtrip_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            float elapsed_ms = 0.0f;
            check_cuda(cudaEventRecord(start), "cudaEventRecord roundtrip start");
            pack_node_major_pairs_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
                d_source,
                d_packed,
                config.merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "pack_node_major_pairs_kernel roundtrip launch");
            unpack_node_major_pairs_kernel<<<block_count(residue_value_count), kThreadsPerBlock>>>(
                d_packed,
                d_reconstructed,
                config.merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "unpack_node_major_pairs_kernel roundtrip launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord roundtrip stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize roundtrip stop");
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime roundtrip");
            total_roundtrip_ms += static_cast<double>(elapsed_ms);
        }
        report.avg_roundtrip_ms = total_roundtrip_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pack_ms > 0.0) {
            report.pack_device_bytes_per_second =
                static_cast<double>(residue_bytes * 2ull) * 1000.0 / report.avg_pack_ms;
        }
        if (report.avg_unpack_ms > 0.0) {
            report.unpack_device_bytes_per_second =
                static_cast<double>(residue_bytes * 2ull) * 1000.0 / report.avg_unpack_ms;
        }
        if (report.avg_roundtrip_ms > 0.0) {
            report.effective_residue_values_per_second =
                static_cast<double>(residue_value_count * 2ull) * 1000.0 / report.avg_roundtrip_ms;
            report.roundtrip_device_bytes_per_second =
                static_cast<double>(residue_bytes * 4ull) * 1000.0 / report.avg_roundtrip_ms;
        }

        std::vector<std::uint32_t> reconstructed_prefix(
            sample_node_count * static_cast<std::size_t>(modulus_count) * config.slot_count
        );
        check_cuda(
            cudaMemcpy(
                reconstructed_prefix.data(),
                d_reconstructed,
                sizeof(std::uint32_t) * reconstructed_prefix.size(),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy reconstructed verification prefix"
        );

        bool verification_ok = true;
        for (std::size_t node_index = 0; node_index < sample_node_count; ++node_index) {
            for (int modulus_index = 0; modulus_index < modulus_count; ++modulus_index) {
                for (std::size_t slot_index = 0; slot_index < report.verification_sample_count; ++slot_index) {
                    const std::size_t host_index =
                        ((node_index * static_cast<std::size_t>(modulus_count)) + static_cast<std::size_t>(modulus_index)) *
                            config.slot_count +
                        slot_index;
                    const std::uint32_t expected = expected_node_major_value(
                        slot_index,
                        node_index,
                        static_cast<std::size_t>(modulus_index)
                    );
                    verification_ok = verification_ok && reconstructed_prefix[host_index] == expected;
                    report.verified_values += 1;
                }
            }
        }

        report.ok = verification_ok;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

void print_device_pair_pack_report(std::ostream& out, const DevicePairPackReport& report) {
    out << "device_pair_pack_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "source_layout=" << report.source_layout << '\n';
    out << "packed_layout=" << report.packed_layout << '\n';
    out << "operation=" << report.operation << '\n';
    out << "merge_count=" << report.merge_count << '\n';
    out << "node_count=" << report.node_count << '\n';
    out << "slot_count=" << report.slot_count << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "warmup_iterations=" << report.warmup_iterations << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "residue_value_count=" << report.residue_value_count << '\n';
    out << "cold_pack_ms=" << report.cold_pack_ms << '\n';
    out << "avg_pack_ms=" << report.avg_pack_ms << '\n';
    out << "cold_unpack_ms=" << report.cold_unpack_ms << '\n';
    out << "avg_unpack_ms=" << report.avg_unpack_ms << '\n';
    out << "cold_roundtrip_ms=" << report.cold_roundtrip_ms << '\n';
    out << "avg_roundtrip_ms=" << report.avg_roundtrip_ms << '\n';
    out << "effective_residue_values_per_second=" << report.effective_residue_values_per_second << '\n';
    out << "pack_device_bytes_per_second=" << report.pack_device_bytes_per_second << '\n';
    out << "unpack_device_bytes_per_second=" << report.unpack_device_bytes_per_second << '\n';
    out << "roundtrip_device_bytes_per_second=" << report.roundtrip_device_bytes_per_second << '\n';
    out << "verification_sample_count=" << report.verification_sample_count << '\n';
    out << "verified_values=" << report.verified_values << '\n';
}

PersistentLevelReduceReport run_persistent_level_reduce_smoke(const PersistentLevelReduceConfig& config) {
    if (config.node_count < 2 || !is_power_of_two(config.node_count)) {
        throw std::invalid_argument("node_count must be a power of two and >= 2");
    }
    if (config.slot_count == 0) {
        throw std::invalid_argument("slot_count must be positive");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    const std::size_t level_value_count =
        static_cast<std::size_t>(modulus_count) * config.node_count * config.slot_count;
    const std::size_t level_bytes = sizeof(std::uint32_t) * level_value_count;

    std::size_t total_level_input_values = 0;
    std::size_t total_parent_output_values = 0;
    int reduced_levels = 0;
    for (std::size_t current_nodes = config.node_count; current_nodes > 1; current_nodes /= 2ull) {
        const std::size_t current_level_values =
            static_cast<std::size_t>(modulus_count) * current_nodes * config.slot_count;
        const std::size_t parent_level_values =
            static_cast<std::size_t>(modulus_count) * (current_nodes / 2ull) * config.slot_count;
        total_level_input_values += current_level_values;
        total_parent_output_values += parent_level_values;
        reduced_levels += 1;
    }
    const std::size_t total_pipeline_bytes =
        sizeof(std::uint32_t) * (total_level_input_values * 2ull + total_parent_output_values * 3ull);

    std::uint32_t* d_original = nullptr;
    std::uint32_t* d_level = nullptr;
    std::uint32_t* d_packed = nullptr;
    std::uint32_t* d_moduli = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto cleanup = [&]() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_moduli);
        cudaFree(d_packed);
        cudaFree(d_level);
        cudaFree(d_original);
    };

    PersistentLevelReduceReport report;
    report.source_layout = "levels[modulus][node][slot]";
    report.packed_layout = "packed[modulus][merge][child][slot]";
    report.parent_layout = "levels[modulus][parent][slot]";
    report.operation = "persistent_level_pack_reduce_add_tree";
    report.node_count = config.node_count;
    report.final_node_count = 1;
    report.reduced_levels = reduced_levels;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.total_level_input_values = total_level_input_values;
    report.total_parent_output_values = total_parent_output_values;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_original), level_bytes), "cudaMalloc d_original");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_level), level_bytes), "cudaMalloc d_level");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_packed), level_bytes), "cudaMalloc d_packed");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc d_moduli"
        );
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy d_moduli"
        );
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

        initialize_modulus_major_residues_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_original,
            d_moduli,
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_modulus_major_residues_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after init");

        const auto run_pipeline_once = [&](double& elapsed_ms) {
            check_cuda(cudaMemcpy(d_level, d_original, level_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy reset d_level");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after reset");

            check_cuda(cudaEventRecord(start), "cudaEventRecord pipeline start");
            for (std::size_t current_nodes = config.node_count; current_nodes > 1; current_nodes /= 2ull) {
                const std::size_t current_level_values =
                    static_cast<std::size_t>(modulus_count) * current_nodes * config.slot_count;
                const std::size_t parent_level_values =
                    static_cast<std::size_t>(modulus_count) * (current_nodes / 2ull) * config.slot_count;

                pack_adjacent_pairs_modulus_major_kernel<<<block_count(current_level_values), kThreadsPerBlock>>>(
                    d_level,
                    d_packed,
                    current_nodes,
                    config.slot_count,
                    modulus_count
                );
                check_cuda(cudaGetLastError(), "pack_adjacent_pairs_modulus_major_kernel launch");

                reduce_packed_pairs_add_mod_kernel<<<block_count(parent_level_values), kThreadsPerBlock>>>(
                    d_packed,
                    d_moduli,
                    d_level,
                    current_nodes / 2ull,
                    config.slot_count,
                    modulus_count
                );
                check_cuda(cudaGetLastError(), "reduce_packed_pairs_add_mod_kernel launch");
            }
            check_cuda(cudaEventRecord(stop), "cudaEventRecord pipeline stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize pipeline stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime pipeline");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pipeline_once(ignored_ms);
        }

        run_pipeline_once(report.cold_pipeline_ms);

        double total_pipeline_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_pipeline_once(elapsed_ms);
            total_pipeline_ms += elapsed_ms;
        }
        report.avg_pipeline_ms = total_pipeline_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pipeline_ms > 0.0) {
            report.effective_input_values_per_second =
                static_cast<double>(report.total_level_input_values) * 1000.0 / report.avg_pipeline_ms;
            report.effective_parent_values_per_second =
                static_cast<double>(report.total_parent_output_values) * 1000.0 / report.avg_pipeline_ms;
            report.pipeline_device_bytes_per_second =
                static_cast<double>(total_pipeline_bytes) * 1000.0 / report.avg_pipeline_ms;
        }

        bool verification_ok = true;
        std::vector<std::uint32_t> root_samples(report.verification_sample_count);
        for (int modulus_index = 0; modulus_index < modulus_count; ++modulus_index) {
            check_cuda(
                cudaMemcpy(
                    root_samples.data(),
                    d_level + static_cast<std::size_t>(modulus_index) * config.slot_count,
                    sizeof(std::uint32_t) * report.verification_sample_count,
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy root verification samples"
            );
            const std::uint32_t modulus = kDefaultModuli[modulus_index];
            for (std::size_t slot_index = 0; slot_index < report.verification_sample_count; ++slot_index) {
                std::uint64_t sum = 0;
                for (std::size_t node_index = 0; node_index < config.node_count; ++node_index) {
                    sum += expected_modulus_major_value(
                        slot_index,
                        node_index,
                        static_cast<std::size_t>(modulus_index)
                    );
                    sum %= modulus;
                }
                const std::uint32_t expected = static_cast<std::uint32_t>(sum);
                verification_ok = verification_ok && root_samples[slot_index] == expected;
                report.verified_values += 1;
            }
        }

        report.ok = verification_ok;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

void print_persistent_level_reduce_report(std::ostream& out, const PersistentLevelReduceReport& report) {
    out << "persistent_level_reduce_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "source_layout=" << report.source_layout << '\n';
    out << "packed_layout=" << report.packed_layout << '\n';
    out << "parent_layout=" << report.parent_layout << '\n';
    out << "operation=" << report.operation << '\n';
    out << "node_count=" << report.node_count << '\n';
    out << "final_node_count=" << report.final_node_count << '\n';
    out << "reduced_levels=" << report.reduced_levels << '\n';
    out << "slot_count=" << report.slot_count << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "warmup_iterations=" << report.warmup_iterations << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "total_level_input_values=" << report.total_level_input_values << '\n';
    out << "total_parent_output_values=" << report.total_parent_output_values << '\n';
    out << "cold_pipeline_ms=" << report.cold_pipeline_ms << '\n';
    out << "avg_pipeline_ms=" << report.avg_pipeline_ms << '\n';
    out << "effective_input_values_per_second=" << report.effective_input_values_per_second << '\n';
    out << "effective_parent_values_per_second=" << report.effective_parent_values_per_second << '\n';
    out << "pipeline_device_bytes_per_second=" << report.pipeline_device_bytes_per_second << '\n';
    out << "verification_sample_count=" << report.verification_sample_count << '\n';
    out << "verified_values=" << report.verified_values << '\n';
}

BatchedFftBackboneReport run_batched_fft_backbone_smoke(const BatchedFftBackboneConfig& config) {
    if (config.batch_count == 0) {
        throw std::invalid_argument("fft batch_count must be positive");
    }
    if (config.fft_length == 0 || !is_power_of_two(config.fft_length)) {
        throw std::invalid_argument("fft_length must be a power of two and > 0");
    }
    if (config.batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("fft batch_count exceeds cuFFT int range");
    }
    if (config.fft_length > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("fft_length exceeds cuFFT int range");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    const std::size_t complex_value_count = config.batch_count * config.fft_length;
    const std::size_t complex_bytes = sizeof(cufftComplex) * complex_value_count;
    const std::size_t logical_pipeline_bytes = sizeof(cufftComplex) * complex_value_count * 11ull;
    const std::size_t sample_batch_count = std::min<std::size_t>(2, config.batch_count);

    cufftComplex* d_original_a = nullptr;
    cufftComplex* d_original_b = nullptr;
    cufftComplex* d_work_a = nullptr;
    cufftComplex* d_work_b = nullptr;
    cufftComplex* d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cufftHandle plan = 0;

    const auto cleanup = [&]() {
        if (plan != 0) {
            cufftDestroy(plan);
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_original_b);
        cudaFree(d_original_a);
    };

    BatchedFftBackboneReport report;
    report.input_layout = "complex[batch][time_sample]";
    report.spectrum_layout = "complex[batch][frequency_bin]";
    report.operation = "batched_cufft_c2c_convolution_backbone";
    report.batch_count = config.batch_count;
    report.fft_length = config.fft_length;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.complex_value_count = complex_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.fft_length);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_original_a), complex_bytes), "cudaMalloc d_original_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_original_b), complex_bytes), "cudaMalloc d_original_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes), "cudaMalloc d_work_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes), "cudaMalloc d_work_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes), "cudaMalloc d_out");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

        initialize_fft_inputs_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
            d_original_a,
            d_original_b,
            config.batch_count,
            config.fft_length
        );
        check_cuda(cudaGetLastError(), "initialize_fft_inputs_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after fft init");

        const auto plan_begin = std::chrono::steady_clock::now();
        check_cufft(
            cufftPlan1d(
                &plan,
                static_cast<int>(config.fft_length),
                CUFFT_C2C,
                static_cast<int>(config.batch_count)
            ),
            "cufftPlan1d"
        );
        const auto plan_end = std::chrono::steady_clock::now();
        report.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto run_pipeline_once = [&](double& elapsed_ms) {
            check_cuda(cudaMemcpy(d_work_a, d_original_a, complex_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy reset d_work_a");
            check_cuda(cudaMemcpy(d_work_b, d_original_b, complex_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy reset d_work_b");

            check_cuda(cudaEventRecord(start), "cudaEventRecord fft pipeline start");
            check_cufft(cufftExecC2C(plan, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecC2C forward a");
            check_cufft(cufftExecC2C(plan, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecC2C forward b");
            complex_pointwise_multiply_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_work_a,
                d_work_b,
                d_out,
                complex_value_count
            );
            check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel launch");
            check_cufft(cufftExecC2C(plan, d_out, d_out, CUFFT_INVERSE), "cufftExecC2C inverse");
            scale_complex_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_out,
                1.0f / static_cast<float>(config.fft_length),
                complex_value_count
            );
            check_cuda(cudaGetLastError(), "scale_complex_kernel launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord fft pipeline stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize fft pipeline stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime fft pipeline");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pipeline_once(ignored_ms);
        }

        run_pipeline_once(report.cold_pipeline_ms);

        double total_pipeline_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_pipeline_once(elapsed_ms);
            total_pipeline_ms += elapsed_ms;
        }
        report.avg_pipeline_ms = total_pipeline_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pipeline_ms > 0.0) {
            report.transformed_complex_values_per_second =
                static_cast<double>(complex_value_count * 3ull) * 1000.0 / report.avg_pipeline_ms;
            report.output_complex_values_per_second =
                static_cast<double>(complex_value_count) * 1000.0 / report.avg_pipeline_ms;
            report.logical_pipeline_bytes_per_second =
                static_cast<double>(logical_pipeline_bytes) * 1000.0 / report.avg_pipeline_ms;
        }

        bool verification_ok = true;
        std::vector<cufftComplex> samples(report.verification_sample_count);
        const double tolerance_base = 0.25;
        for (std::size_t batch_index = 0; batch_index < sample_batch_count; ++batch_index) {
            check_cuda(
                cudaMemcpy(
                    samples.data(),
                    d_out + batch_index * config.fft_length,
                    sizeof(cufftComplex) * report.verification_sample_count,
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy fft verification samples"
            );
            for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                double expected_real = 0.0;
                for (std::size_t inner_index = 0; inner_index < config.fft_length; ++inner_index) {
                    const std::size_t rhs_index = (output_index + config.fft_length - inner_index) % config.fft_length;
                    expected_real += static_cast<double>(expected_fft_lhs_value(inner_index, batch_index)) *
                                     static_cast<double>(expected_fft_rhs_value(rhs_index, batch_index));
                }
                const double real_error = std::abs(static_cast<double>(samples[output_index].x) - expected_real);
                const double imag_error = std::abs(static_cast<double>(samples[output_index].y));
                const double tolerance = tolerance_base + 1.0e-3 * std::abs(expected_real);
                verification_ok = verification_ok && real_error <= tolerance && imag_error <= tolerance;
                report.verified_values += 1;
            }
        }

        report.ok = verification_ok;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

void print_batched_fft_backbone_report(std::ostream& out, const BatchedFftBackboneReport& report) {
    out << "batched_fft_backbone_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "input_layout=" << report.input_layout << '\n';
    out << "spectrum_layout=" << report.spectrum_layout << '\n';
    out << "operation=" << report.operation << '\n';
    out << "batch_count=" << report.batch_count << '\n';
    out << "fft_length=" << report.fft_length << '\n';
    out << "warmup_iterations=" << report.warmup_iterations << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "complex_value_count=" << report.complex_value_count << '\n';
    out << "plan_build_ms=" << report.plan_build_ms << '\n';
    out << "cold_pipeline_ms=" << report.cold_pipeline_ms << '\n';
    out << "avg_pipeline_ms=" << report.avg_pipeline_ms << '\n';
    out << "transformed_complex_values_per_second=" << report.transformed_complex_values_per_second << '\n';
    out << "output_complex_values_per_second=" << report.output_complex_values_per_second << '\n';
    out << "logical_pipeline_bytes_per_second=" << report.logical_pipeline_bytes_per_second << '\n';
    out << "verification_sample_count=" << report.verification_sample_count << '\n';
    out << "verified_values=" << report.verified_values << '\n';
}

ResidueFftBridgeReport run_residue_fft_bridge_smoke(const ResidueFftBridgeConfig& config) {
    if (config.node_count < 2 || (config.node_count % 2ull) != 0ull) {
        throw std::invalid_argument("bridge node_count must be even and >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("bridge slot_count must be a power of two and > 0");
    }
    if (config.node_count / 2ull > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("bridge merge_count exceeds cuFFT int range");
    }
    if (config.slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("bridge slot_count exceeds cuFFT int range");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    if (config.bridge_modulus_index < 0 || config.bridge_modulus_index >= modulus_count) {
        throw std::invalid_argument("bridge_modulus_index out of range");
    }
    if (config.packing_mask == 0 || config.packing_mask > 63u) {
        throw std::invalid_argument("packing_mask must be in [1, 63]");
    }

    const std::size_t merge_count = config.node_count / 2ull;
    const std::size_t level_value_count =
        static_cast<std::size_t>(modulus_count) * config.node_count * config.slot_count;
    const std::size_t level_bytes = sizeof(std::uint32_t) * level_value_count;
    const std::size_t complex_value_count = merge_count * config.slot_count;
    const std::size_t complex_bytes = sizeof(cufftComplex) * complex_value_count;
    const std::size_t residue_values_packed = complex_value_count * 2ull;
    const std::size_t pack_logical_bytes =
        residue_values_packed * sizeof(std::uint32_t) + complex_bytes * 2ull;
    const std::size_t bridge_logical_bytes = pack_logical_bytes + complex_bytes * 11ull;
    const std::size_t sample_merge_count = std::min<std::size_t>(2, merge_count);

    std::uint32_t* d_level = nullptr;
    std::uint32_t* d_moduli = nullptr;
    cufftComplex* d_work_a = nullptr;
    cufftComplex* d_work_b = nullptr;
    cufftComplex* d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cufftHandle plan = 0;

    const auto cleanup = [&]() {
        if (plan != 0) {
            cufftDestroy(plan);
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_moduli);
        cudaFree(d_level);
    };

    ResidueFftBridgeReport report;
    report.source_layout = "levels[modulus][node][slot]";
    report.fft_input_layout = "complex[merge_batch][slot]";
    report.spectrum_layout = "complex[merge_batch][frequency_bin]";
    report.operation = "persistent_residue_to_batched_cufft_bridge";
    report.node_count = config.node_count;
    report.merge_count = merge_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.bridge_modulus_index = config.bridge_modulus_index;
    report.packing_mask = config.packing_mask;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.residue_values_packed = residue_values_packed;
    report.complex_value_count = complex_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_level), level_bytes), "cudaMalloc d_level");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc d_moduli"
        );
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes), "cudaMalloc d_work_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes), "cudaMalloc d_work_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes), "cudaMalloc d_out");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy d_moduli"
        );

        initialize_modulus_major_residues_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_level,
            d_moduli,
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_modulus_major_residues_kernel bridge launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after bridge init");

        const auto plan_begin = std::chrono::steady_clock::now();
        check_cufft(
            cufftPlan1d(
                &plan,
                static_cast<int>(config.slot_count),
                CUFFT_C2C,
                static_cast<int>(merge_count)
            ),
            "cufftPlan1d residue bridge"
        );
        const auto plan_end = std::chrono::steady_clock::now();
        report.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto run_pack_once = [&](double& elapsed_ms) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord bridge pack start");
            pack_modulus_major_pairs_to_fft_inputs_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_level,
                d_work_a,
                d_work_b,
                config.node_count,
                config.slot_count,
                static_cast<std::size_t>(config.bridge_modulus_index),
                config.packing_mask
            );
            check_cuda(cudaGetLastError(), "pack_modulus_major_pairs_to_fft_inputs_kernel launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord bridge pack stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize bridge pack stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime bridge pack");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        const auto run_bridge_once = [&](double& elapsed_ms) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord residue bridge start");
            pack_modulus_major_pairs_to_fft_inputs_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_level,
                d_work_a,
                d_work_b,
                config.node_count,
                config.slot_count,
                static_cast<std::size_t>(config.bridge_modulus_index),
                config.packing_mask
            );
            check_cuda(cudaGetLastError(), "pack_modulus_major_pairs_to_fft_inputs_kernel bridge launch");
            check_cufft(cufftExecC2C(plan, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecC2C bridge forward a");
            check_cufft(cufftExecC2C(plan, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecC2C bridge forward b");
            complex_pointwise_multiply_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_work_a,
                d_work_b,
                d_out,
                complex_value_count
            );
            check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel bridge launch");
            check_cufft(cufftExecC2C(plan, d_out, d_out, CUFFT_INVERSE), "cufftExecC2C bridge inverse");
            scale_complex_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_out,
                1.0f / static_cast<float>(config.slot_count),
                complex_value_count
            );
            check_cuda(cudaGetLastError(), "scale_complex_kernel bridge launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord residue bridge stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize residue bridge stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime residue bridge");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pack_once(ignored_ms);
            run_bridge_once(ignored_ms);
        }

        run_pack_once(report.cold_pack_ms);
        run_bridge_once(report.cold_bridge_ms);

        double total_pack_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_pack_once(elapsed_ms);
            total_pack_ms += elapsed_ms;
        }
        report.avg_pack_ms = total_pack_ms / static_cast<double>(config.measured_iterations);

        double total_bridge_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_bridge_once(elapsed_ms);
            total_bridge_ms += elapsed_ms;
        }
        report.avg_bridge_ms = total_bridge_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pack_ms > 0.0) {
            report.packed_residue_values_per_second =
                static_cast<double>(report.residue_values_packed) * 1000.0 / report.avg_pack_ms;
        }
        if (report.avg_bridge_ms > 0.0) {
            report.transformed_complex_values_per_second =
                static_cast<double>(report.complex_value_count * 3ull) * 1000.0 / report.avg_bridge_ms;
            report.logical_bridge_bytes_per_second =
                static_cast<double>(bridge_logical_bytes) * 1000.0 / report.avg_bridge_ms;
        }

        bool verification_ok = true;
        std::vector<cufftComplex> samples(report.verification_sample_count);
        const double tolerance_base = 2.0;
        for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
            check_cuda(
                cudaMemcpy(
                    samples.data(),
                    d_out + merge_index * config.slot_count,
                    sizeof(cufftComplex) * report.verification_sample_count,
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy residue bridge verification samples"
            );
            for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                double expected_real = 0.0;
                for (std::size_t inner_index = 0; inner_index < config.slot_count; ++inner_index) {
                    const std::size_t rhs_index = (output_index + config.slot_count - inner_index) % config.slot_count;
                    expected_real += static_cast<double>(expected_bridge_residue_value(
                                         inner_index,
                                         merge_index * 2ull,
                                         static_cast<std::size_t>(config.bridge_modulus_index),
                                         config.packing_mask
                                     )) *
                                     static_cast<double>(expected_bridge_residue_value(
                                         rhs_index,
                                         merge_index * 2ull + 1ull,
                                         static_cast<std::size_t>(config.bridge_modulus_index),
                                         config.packing_mask
                                     ));
                }
                const double real_error = std::abs(static_cast<double>(samples[output_index].x) - expected_real);
                const double imag_error = std::abs(static_cast<double>(samples[output_index].y));
                const double tolerance = tolerance_base + 1.0e-4 * std::abs(expected_real);
                verification_ok = verification_ok && real_error <= tolerance && imag_error <= tolerance;
                report.verified_values += 1;
            }
        }

        report.ok = verification_ok;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

void print_residue_fft_bridge_report(std::ostream& out, const ResidueFftBridgeReport& report) {
    out << "residue_fft_bridge_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "source_layout=" << report.source_layout << '\n';
    out << "fft_input_layout=" << report.fft_input_layout << '\n';
    out << "spectrum_layout=" << report.spectrum_layout << '\n';
    out << "operation=" << report.operation << '\n';
    out << "node_count=" << report.node_count << '\n';
    out << "merge_count=" << report.merge_count << '\n';
    out << "slot_count=" << report.slot_count << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "bridge_modulus_index=" << report.bridge_modulus_index << '\n';
    out << "packing_mask=" << report.packing_mask << '\n';
    out << "warmup_iterations=" << report.warmup_iterations << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "residue_values_packed=" << report.residue_values_packed << '\n';
    out << "complex_value_count=" << report.complex_value_count << '\n';
    out << "plan_build_ms=" << report.plan_build_ms << '\n';
    out << "cold_pack_ms=" << report.cold_pack_ms << '\n';
    out << "avg_pack_ms=" << report.avg_pack_ms << '\n';
    out << "cold_bridge_ms=" << report.cold_bridge_ms << '\n';
    out << "avg_bridge_ms=" << report.avg_bridge_ms << '\n';
    out << "packed_residue_values_per_second=" << report.packed_residue_values_per_second << '\n';
    out << "transformed_complex_values_per_second=" << report.transformed_complex_values_per_second << '\n';
    out << "logical_bridge_bytes_per_second=" << report.logical_bridge_bytes_per_second << '\n';
    out << "verification_sample_count=" << report.verification_sample_count << '\n';
    out << "verified_values=" << report.verified_values << '\n';
}

GroupedResidueFftBridgeReport run_grouped_residue_fft_bridge_smoke(const GroupedResidueFftBridgeConfig& config) {
    if (config.node_count < 2 || (config.node_count % 2ull) != 0ull) {
        throw std::invalid_argument("grouped bridge node_count must be even and >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("grouped bridge slot_count must be a power of two and > 0");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    if (config.packing_mask == 0 || config.packing_mask > 63u) {
        throw std::invalid_argument("packing_mask must be in [1, 63]");
    }

    const std::size_t merge_count = config.node_count / 2ull;
    const std::size_t fft_batch_count = static_cast<std::size_t>(modulus_count) * merge_count;
    if (fft_batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped bridge fft batch_count exceeds cuFFT int range");
    }
    if (config.slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped bridge slot_count exceeds cuFFT int range");
    }

    const std::size_t level_value_count =
        static_cast<std::size_t>(modulus_count) * config.node_count * config.slot_count;
    const std::size_t level_bytes = sizeof(std::uint32_t) * level_value_count;
    const std::size_t complex_value_count = fft_batch_count * config.slot_count;
    const std::size_t complex_bytes = sizeof(cufftComplex) * complex_value_count;
    const std::size_t residue_values_packed = complex_value_count * 2ull;
    const std::size_t pack_logical_bytes =
        residue_values_packed * sizeof(std::uint32_t) + complex_bytes * 2ull;
    const std::size_t bridge_logical_bytes = pack_logical_bytes + complex_bytes * 11ull;
    const std::size_t sample_modulus_count = std::min<std::size_t>(2, static_cast<std::size_t>(modulus_count));
    const std::size_t sample_merge_count = std::min<std::size_t>(2, merge_count);

    std::uint32_t* d_level = nullptr;
    std::uint32_t* d_moduli = nullptr;
    cufftComplex* d_work_a = nullptr;
    cufftComplex* d_work_b = nullptr;
    cufftComplex* d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cufftHandle plan = 0;

    const auto cleanup = [&]() {
        if (plan != 0) {
            cufftDestroy(plan);
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_moduli);
        cudaFree(d_level);
    };

    GroupedResidueFftBridgeReport report;
    report.source_layout = "levels[modulus][node][slot]";
    report.fft_input_layout = "complex[modulus_merge_batch][slot]";
    report.spectrum_layout = "complex[modulus_merge_batch][frequency_bin]";
    report.operation = "grouped_persistent_residue_to_batched_cufft_bridge";
    report.node_count = config.node_count;
    report.merge_count = merge_count;
    report.fft_batch_count = fft_batch_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.packing_mask = config.packing_mask;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.residue_values_packed = residue_values_packed;
    report.complex_value_count = complex_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_level), level_bytes), "cudaMalloc grouped d_level");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc grouped d_moduli"
        );
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes), "cudaMalloc grouped d_work_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes), "cudaMalloc grouped d_work_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes), "cudaMalloc grouped d_out");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate grouped start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate grouped stop");
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy grouped d_moduli"
        );

        initialize_modulus_major_residues_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_level,
            d_moduli,
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_modulus_major_residues_kernel grouped launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after grouped init");

        const auto plan_begin = std::chrono::steady_clock::now();
        check_cufft(
            cufftPlan1d(
                &plan,
                static_cast<int>(config.slot_count),
                CUFFT_C2C,
                static_cast<int>(fft_batch_count)
            ),
            "cufftPlan1d grouped residue bridge"
        );
        const auto plan_end = std::chrono::steady_clock::now();
        report.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto run_pack_once = [&](double& elapsed_ms) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord grouped bridge pack start");
            pack_grouped_modulus_major_pairs_to_fft_inputs_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_level,
                d_work_a,
                d_work_b,
                config.node_count,
                config.slot_count,
                modulus_count,
                config.packing_mask
            );
            check_cuda(cudaGetLastError(), "pack_grouped_modulus_major_pairs_to_fft_inputs_kernel launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord grouped bridge pack stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize grouped bridge pack stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime grouped bridge pack");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        const auto run_bridge_once = [&](double& elapsed_ms) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord grouped residue bridge start");
            pack_grouped_modulus_major_pairs_to_fft_inputs_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_level,
                d_work_a,
                d_work_b,
                config.node_count,
                config.slot_count,
                modulus_count,
                config.packing_mask
            );
            check_cuda(cudaGetLastError(), "pack_grouped_modulus_major_pairs_to_fft_inputs_kernel bridge launch");
            check_cufft(cufftExecC2C(plan, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecC2C grouped bridge forward a");
            check_cufft(cufftExecC2C(plan, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecC2C grouped bridge forward b");
            complex_pointwise_multiply_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_work_a,
                d_work_b,
                d_out,
                complex_value_count
            );
            check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel grouped bridge launch");
            check_cufft(cufftExecC2C(plan, d_out, d_out, CUFFT_INVERSE), "cufftExecC2C grouped bridge inverse");
            scale_complex_kernel<<<block_count(complex_value_count), kThreadsPerBlock>>>(
                d_out,
                1.0f / static_cast<float>(config.slot_count),
                complex_value_count
            );
            check_cuda(cudaGetLastError(), "scale_complex_kernel grouped bridge launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord grouped residue bridge stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize grouped residue bridge stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime grouped residue bridge");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pack_once(ignored_ms);
            run_bridge_once(ignored_ms);
        }

        run_pack_once(report.cold_pack_ms);
        run_bridge_once(report.cold_bridge_ms);

        double total_pack_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_pack_once(elapsed_ms);
            total_pack_ms += elapsed_ms;
        }
        report.avg_pack_ms = total_pack_ms / static_cast<double>(config.measured_iterations);

        double total_bridge_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_bridge_once(elapsed_ms);
            total_bridge_ms += elapsed_ms;
        }
        report.avg_bridge_ms = total_bridge_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pack_ms > 0.0) {
            report.packed_residue_values_per_second =
                static_cast<double>(report.residue_values_packed) * 1000.0 / report.avg_pack_ms;
        }
        if (report.avg_bridge_ms > 0.0) {
            report.transformed_complex_values_per_second =
                static_cast<double>(report.complex_value_count * 3ull) * 1000.0 / report.avg_bridge_ms;
            report.logical_bridge_bytes_per_second =
                static_cast<double>(bridge_logical_bytes) * 1000.0 / report.avg_bridge_ms;
        }

        bool verification_ok = true;
        std::vector<cufftComplex> samples(report.verification_sample_count);
        const double tolerance_base = 2.0;
        for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
            for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                const std::size_t batch_index = modulus_index * merge_count + merge_index;
                check_cuda(
                    cudaMemcpy(
                        samples.data(),
                        d_out + batch_index * config.slot_count,
                        sizeof(cufftComplex) * report.verification_sample_count,
                        cudaMemcpyDeviceToHost
                    ),
                    "cudaMemcpy grouped residue bridge verification samples"
                );
                for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                    double expected_real = 0.0;
                    for (std::size_t inner_index = 0; inner_index < config.slot_count; ++inner_index) {
                        const std::size_t rhs_index =
                            (output_index + config.slot_count - inner_index) % config.slot_count;
                        expected_real += static_cast<double>(expected_bridge_residue_value(
                                             inner_index,
                                             merge_index * 2ull,
                                             modulus_index,
                                             config.packing_mask
                                         )) *
                                         static_cast<double>(expected_bridge_residue_value(
                                             rhs_index,
                                             merge_index * 2ull + 1ull,
                                             modulus_index,
                                             config.packing_mask
                                         ));
                    }
                    const double real_error = std::abs(static_cast<double>(samples[output_index].x) - expected_real);
                    const double imag_error = std::abs(static_cast<double>(samples[output_index].y));
                    const double tolerance = tolerance_base + 1.0e-4 * std::abs(expected_real);
                    verification_ok = verification_ok && real_error <= tolerance && imag_error <= tolerance;
                    report.verified_values += 1;
                }
            }
        }

        report.ok = verification_ok;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

void print_grouped_residue_fft_bridge_report(std::ostream& out, const GroupedResidueFftBridgeReport& report) {
    out << "grouped_residue_fft_bridge_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "source_layout=" << report.source_layout << '\n';
    out << "fft_input_layout=" << report.fft_input_layout << '\n';
    out << "spectrum_layout=" << report.spectrum_layout << '\n';
    out << "operation=" << report.operation << '\n';
    out << "node_count=" << report.node_count << '\n';
    out << "merge_count=" << report.merge_count << '\n';
    out << "fft_batch_count=" << report.fft_batch_count << '\n';
    out << "slot_count=" << report.slot_count << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "packing_mask=" << report.packing_mask << '\n';
    out << "warmup_iterations=" << report.warmup_iterations << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "residue_values_packed=" << report.residue_values_packed << '\n';
    out << "complex_value_count=" << report.complex_value_count << '\n';
    out << "plan_build_ms=" << report.plan_build_ms << '\n';
    out << "cold_pack_ms=" << report.cold_pack_ms << '\n';
    out << "avg_pack_ms=" << report.avg_pack_ms << '\n';
    out << "cold_bridge_ms=" << report.cold_bridge_ms << '\n';
    out << "avg_bridge_ms=" << report.avg_bridge_ms << '\n';
    out << "packed_residue_values_per_second=" << report.packed_residue_values_per_second << '\n';
    out << "transformed_complex_values_per_second=" << report.transformed_complex_values_per_second << '\n';
    out << "logical_bridge_bytes_per_second=" << report.logical_bridge_bytes_per_second << '\n';
    out << "verification_sample_count=" << report.verification_sample_count << '\n';
    out << "verified_values=" << report.verified_values << '\n';
}

GroupedLevelPlannerReport run_grouped_level_planner_smoke(const GroupedLevelPlannerConfig& config) {
    if (config.node_count < 2 || !is_power_of_two(config.node_count)) {
        throw std::invalid_argument("grouped level planner node_count must be a power of two and >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("grouped level planner slot_count must be a power of two and > 0");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    if (config.packing_mask == 0 || config.packing_mask > 63u) {
        throw std::invalid_argument("packing_mask must be in [1, 63]");
    }

    const std::size_t level_value_count =
        static_cast<std::size_t>(modulus_count) * config.node_count * config.slot_count;
    const std::size_t level_bytes = sizeof(std::uint32_t) * level_value_count;
    const std::size_t max_fft_batch_count =
        static_cast<std::size_t>(modulus_count) * (config.node_count / 2ull);
    if (max_fft_batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped level planner fft batch_count exceeds cuFFT int range");
    }
    if (config.slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped level planner slot_count exceeds cuFFT int range");
    }
    const std::size_t max_complex_value_count = max_fft_batch_count * config.slot_count;
    const std::size_t complex_bytes = sizeof(cufftComplex) * max_complex_value_count;

    std::size_t total_fft_batch_count = 0;
    std::size_t total_complex_value_count = 0;
    int level_count = 0;
    for (std::size_t current_nodes = config.node_count; current_nodes > 1; current_nodes /= 2ull) {
        const std::size_t batch_count = static_cast<std::size_t>(modulus_count) * (current_nodes / 2ull);
        total_fft_batch_count += batch_count;
        total_complex_value_count += batch_count * config.slot_count;
        level_count += 1;
    }
    const std::size_t total_residue_values_packed = total_complex_value_count * 2ull;
    const std::size_t logical_pipeline_bytes = total_complex_value_count * sizeof(cufftComplex) * 14ull;
    const std::size_t sample_modulus_count = std::min<std::size_t>(2, static_cast<std::size_t>(modulus_count));
    const std::size_t first_level_merge_count = config.node_count / 2ull;
    const std::size_t sample_merge_count = std::min<std::size_t>(2, first_level_merge_count);

    std::uint32_t* d_original = nullptr;
    std::uint32_t* d_level = nullptr;
    std::uint32_t* d_moduli = nullptr;
    cufftComplex* d_work_a = nullptr;
    cufftComplex* d_work_b = nullptr;
    cufftComplex* d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<cufftHandle> plans(static_cast<std::size_t>(level_count), 0);

    const auto cleanup = [&]() {
        for (cufftHandle& plan : plans) {
            if (plan != 0) {
                cufftDestroy(plan);
                plan = 0;
            }
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_moduli);
        cudaFree(d_level);
        cudaFree(d_original);
    };

    GroupedLevelPlannerReport report;
    report.source_layout = "levels[modulus][node][slot]";
    report.fft_input_layout = "complex[level_modulus_merge_batch][slot]";
    report.parent_layout = "levels[modulus][parent][slot]";
    report.operation = "grouped_level_planner_batched_cufft_tree";
    report.node_count = config.node_count;
    report.final_node_count = 1;
    report.level_count = level_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.packing_mask = config.packing_mask;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.total_fft_batch_count = total_fft_batch_count;
    report.total_residue_values_packed = total_residue_values_packed;
    report.total_complex_value_count = total_complex_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_original), level_bytes), "cudaMalloc planner d_original");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_level), level_bytes), "cudaMalloc planner d_level");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc planner d_moduli"
        );
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes), "cudaMalloc planner d_work_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes), "cudaMalloc planner d_work_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes), "cudaMalloc planner d_out");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate planner start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate planner stop");
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy planner d_moduli"
        );

        initialize_modulus_major_residues_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_original,
            d_moduli,
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_modulus_major_residues_kernel planner launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after planner init");

        const auto plan_begin = std::chrono::steady_clock::now();
        std::size_t current_nodes_for_plan = config.node_count;
        for (int level_index = 0; level_index < level_count; ++level_index) {
            const std::size_t batch_count =
                static_cast<std::size_t>(modulus_count) * (current_nodes_for_plan / 2ull);
            check_cufft(
                cufftPlan1d(
                    &plans[static_cast<std::size_t>(level_index)],
                    static_cast<int>(config.slot_count),
                    CUFFT_C2C,
                    static_cast<int>(batch_count)
                ),
                "cufftPlan1d grouped level planner"
            );
            current_nodes_for_plan /= 2ull;
        }
        const auto plan_end = std::chrono::steady_clock::now();
        report.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto execute_single_level = [&](std::size_t current_nodes, cufftHandle plan_handle) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t complex_value_count_level =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;

            pack_grouped_modulus_major_pairs_to_fft_inputs_kernel<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_level,
                d_work_a,
                d_work_b,
                current_nodes,
                config.slot_count,
                modulus_count,
                config.packing_mask
            );
            check_cuda(cudaGetLastError(), "pack_grouped_modulus_major_pairs_to_fft_inputs_kernel planner launch");
            check_cufft(cufftExecC2C(plan_handle, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecC2C planner forward a");
            check_cufft(cufftExecC2C(plan_handle, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecC2C planner forward b");
            complex_pointwise_multiply_kernel<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_work_a,
                d_work_b,
                d_out,
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel planner launch");
            check_cufft(cufftExecC2C(plan_handle, d_out, d_out, CUFFT_INVERSE), "cufftExecC2C planner inverse");
            scale_complex_kernel<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_out,
                1.0f / static_cast<float>(config.slot_count),
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "scale_complex_kernel planner launch");
            project_grouped_fft_output_to_level_values_kernel<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_out,
                d_level,
                merge_count,
                config.slot_count,
                modulus_count,
                config.packing_mask
            );
            check_cuda(cudaGetLastError(), "project_grouped_fft_output_to_level_values_kernel planner launch");
        };

        {
            check_cuda(cudaMemcpy(d_level, d_original, level_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy planner verification reset");
            execute_single_level(config.node_count, plans[0]);
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after planner verification level");

            std::vector<std::uint32_t> level_samples(sample_merge_count * config.slot_count);
            std::vector<cufftComplex> fft_samples(report.verification_sample_count);
            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                check_cuda(
                    cudaMemcpy(
                        level_samples.data(),
                        d_level + modulus_index * first_level_merge_count * config.slot_count,
                        sizeof(std::uint32_t) * level_samples.size(),
                        cudaMemcpyDeviceToHost
                    ),
                    "cudaMemcpy grouped planner verification samples"
                );
                for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                    const std::size_t batch_index = modulus_index * first_level_merge_count + merge_index;
                    check_cuda(
                        cudaMemcpy(
                            fft_samples.data(),
                            d_out + batch_index * config.slot_count,
                            sizeof(cufftComplex) * report.verification_sample_count,
                            cudaMemcpyDeviceToHost
                        ),
                        "cudaMemcpy grouped planner verification fft samples"
                    );
                    for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                        std::uint64_t expected_accum = 0;
                        double expected_real = 0.0;
                        for (std::size_t inner_index = 0; inner_index < config.slot_count; ++inner_index) {
                            const std::size_t rhs_index =
                                (output_index + config.slot_count - inner_index) % config.slot_count;
                            const auto lhs_value = static_cast<std::uint32_t>(expected_bridge_residue_value(
                                inner_index,
                                merge_index * 2ull,
                                modulus_index,
                                config.packing_mask
                            ));
                            const auto rhs_value = static_cast<std::uint32_t>(expected_bridge_residue_value(
                                rhs_index,
                                merge_index * 2ull + 1ull,
                                modulus_index,
                                config.packing_mask
                            ));
                            expected_accum += static_cast<std::uint64_t>(lhs_value) * static_cast<std::uint64_t>(rhs_value);
                            expected_real += static_cast<double>(lhs_value) * static_cast<double>(rhs_value);
                        }
                        const std::uint32_t expected =
                            static_cast<std::uint32_t>(expected_accum & static_cast<std::uint64_t>(config.packing_mask));
                        const std::size_t host_index = merge_index * config.slot_count + output_index;
                        const std::uint32_t observed = level_samples[host_index];
                        const double real_error =
                            std::abs(static_cast<double>(fft_samples[output_index].x) - expected_real);
                        report.max_projection_real_error =
                            std::max(report.max_projection_real_error, real_error);
                        if (observed != expected) {
                            report.verification_mismatch_count += 1;
                            if (report.verification_mismatch_count == 1) {
                                report.first_mismatch_modulus = modulus_index;
                                report.first_mismatch_merge = merge_index;
                                report.first_mismatch_output = output_index;
                                report.first_mismatch_expected = expected;
                                report.first_mismatch_observed = observed;
                            }
                        }
                        report.verified_values += 1;
                    }
                }
            }
        }

        const auto run_pipeline_once = [&](double& elapsed_ms) {
            check_cuda(cudaMemcpy(d_level, d_original, level_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy planner reset");
            check_cuda(cudaEventRecord(start), "cudaEventRecord planner pipeline start");

            std::size_t current_nodes = config.node_count;
            for (int level_index = 0; level_index < level_count; ++level_index) {
                execute_single_level(current_nodes, plans[static_cast<std::size_t>(level_index)]);
                current_nodes /= 2ull;
            }

            check_cuda(cudaEventRecord(stop), "cudaEventRecord planner pipeline stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize planner pipeline stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime planner pipeline");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pipeline_once(ignored_ms);
        }

        run_pipeline_once(report.cold_pipeline_ms);

        double total_pipeline_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_pipeline_once(elapsed_ms);
            total_pipeline_ms += elapsed_ms;
        }
        report.avg_pipeline_ms = total_pipeline_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pipeline_ms > 0.0) {
            report.packed_residue_values_per_second =
                static_cast<double>(report.total_residue_values_packed) * 1000.0 / report.avg_pipeline_ms;
            report.transformed_complex_values_per_second =
                static_cast<double>(report.total_complex_value_count * 3ull) * 1000.0 / report.avg_pipeline_ms;
            report.logical_pipeline_bytes_per_second =
                static_cast<double>(logical_pipeline_bytes) * 1000.0 / report.avg_pipeline_ms;
        }

        report.ok = report.verification_mismatch_count == 0;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

GroupedLevelPlannerReport run_grouped_level_planner_multilimb_smoke(const GroupedLevelPlannerConfig& config) {
    if (config.node_count < 2 || !is_power_of_two(config.node_count)) {
        throw std::invalid_argument("grouped multilimb planner node_count must be a power of two and >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("grouped multilimb planner slot_count must be a power of two and > 0");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    if (!is_contiguous_low_bit_mask(config.packing_mask)) {
        throw std::invalid_argument("grouped multilimb planner requires packing_mask to be a contiguous low-bit mask");
    }
    const int active_bits = low_bit_mask_width(config.packing_mask);
    const bool use_fp64 = active_bits > 16;
    const int max_limb_bits = select_multilimb_max_limb_bits(active_bits, config.slot_count, use_fp64);
    const auto limbs = build_split_limb_descriptors(config.packing_mask, max_limb_bits);
    const auto passes = build_split_pass_descriptors(limbs, active_bits);
    if (passes.empty()) {
        throw std::invalid_argument("grouped multilimb planner produced an empty pass schedule");
    }

    const std::size_t level_value_count =
        static_cast<std::size_t>(modulus_count) * config.node_count * config.slot_count;
    const std::size_t level_bytes = sizeof(std::uint32_t) * level_value_count;
    const std::size_t max_fft_batch_count =
        static_cast<std::size_t>(modulus_count) * (config.node_count / 2ull);
    if (max_fft_batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped multilimb planner fft batch_count exceeds cuFFT int range");
    }
    if (config.slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped multilimb planner slot_count exceeds cuFFT int range");
    }
    const std::size_t max_complex_value_count = max_fft_batch_count * config.slot_count;
    const std::size_t complex_bytes_fp32 = sizeof(cufftComplex) * max_complex_value_count;
    const std::size_t complex_bytes_fp64 = sizeof(cufftDoubleComplex) * max_complex_value_count;

    std::size_t total_fft_batch_count = 0;
    std::size_t total_complex_value_count = 0;
    int level_count = 0;
    for (std::size_t current_nodes = config.node_count; current_nodes > 1; current_nodes /= 2ull) {
        const std::size_t batch_count = static_cast<std::size_t>(modulus_count) * (current_nodes / 2ull);
        total_fft_batch_count += batch_count;
        total_complex_value_count += batch_count * config.slot_count;
        level_count += 1;
    }
    const std::size_t total_residue_values_packed = total_complex_value_count * 2ull;
    const std::size_t logical_pipeline_bytes = total_complex_value_count * sizeof(cufftComplex) * 14ull;
    const std::size_t sample_modulus_count = std::min<std::size_t>(2, static_cast<std::size_t>(modulus_count));
    const std::size_t first_level_merge_count = config.node_count / 2ull;
    const std::size_t sample_merge_count = std::min<std::size_t>(2, first_level_merge_count);
    const std::size_t sample_storage_size =
        sample_modulus_count * sample_merge_count * std::min(config.verification_sample_count, config.slot_count);
    std::uint32_t* d_original = nullptr;
    std::uint32_t* d_level_a = nullptr;
    std::uint32_t* d_level_b = nullptr;
    std::uint32_t* d_moduli = nullptr;
    cufftComplex* d_work_a = nullptr;
    cufftComplex* d_work_b = nullptr;
    cufftComplex* d_out = nullptr;
    cufftDoubleComplex* d_work_a_fp64 = nullptr;
    cufftDoubleComplex* d_work_b_fp64 = nullptr;
    cufftDoubleComplex* d_out_fp64 = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<cufftHandle> plans(static_cast<std::size_t>(level_count), 0);

    const auto cleanup = [&]() {
        for (cufftHandle& plan : plans) {
            if (plan != 0) {
                cufftDestroy(plan);
                plan = 0;
            }
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_out_fp64);
        cudaFree(d_work_b_fp64);
        cudaFree(d_work_a_fp64);
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_moduli);
        cudaFree(d_level_b);
        cudaFree(d_level_a);
        cudaFree(d_original);
    };

    GroupedLevelPlannerReport report;
    report.source_layout = "levels[modulus][node][slot]";
    report.fft_input_layout =
        "complex[level_modulus_merge_batch][slot]_multilimb_lowbits" + std::to_string(active_bits) +
        "_limb" + std::to_string(max_limb_bits) + (use_fp64 ? "_fp64" : "_fp32");
    report.parent_layout = "levels[modulus][parent][slot]";
    report.operation =
        "grouped_level_planner_multilimb_" + std::to_string(limbs.size()) + "limb_" +
        std::to_string(passes.size()) + "pass_batched_cufft_tree";
    report.node_count = config.node_count;
    report.final_node_count = 1;
    report.level_count = level_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.packing_mask = config.packing_mask;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.total_fft_batch_count = total_fft_batch_count;
    report.total_residue_values_packed = total_residue_values_packed;
    report.total_complex_value_count = total_complex_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);
    report.split_limb_count = static_cast<int>(limbs.size());
    report.split_pass_count = static_cast<int>(passes.size());

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_original), level_bytes), "cudaMalloc multilimb planner d_original");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_level_a), level_bytes), "cudaMalloc multilimb planner d_level_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_level_b), level_bytes), "cudaMalloc multilimb planner d_level_b");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc multilimb planner d_moduli"
        );
        if (use_fp64) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_work_a_fp64), complex_bytes_fp64),
                "cudaMalloc multilimb planner d_work_a_fp64"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_work_b_fp64), complex_bytes_fp64),
                "cudaMalloc multilimb planner d_work_b_fp64"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_out_fp64), complex_bytes_fp64),
                "cudaMalloc multilimb planner d_out_fp64"
            );
        } else {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes_fp32),
                "cudaMalloc multilimb planner d_work_a"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes_fp32),
                "cudaMalloc multilimb planner d_work_b"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes_fp32),
                "cudaMalloc multilimb planner d_out"
            );
        }
        check_cuda(cudaEventCreate(&start), "cudaEventCreate multilimb planner start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate multilimb planner stop");
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy multilimb planner d_moduli"
        );

        initialize_modulus_major_residues_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_original,
            d_moduli,
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_modulus_major_residues_kernel multilimb planner launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after multilimb planner init");

        const auto plan_begin = std::chrono::steady_clock::now();
        std::size_t current_nodes_for_plan = config.node_count;
        for (int level_index = 0; level_index < level_count; ++level_index) {
            const std::size_t batch_count =
                static_cast<std::size_t>(modulus_count) * (current_nodes_for_plan / 2ull);
            check_cufft(
                cufftPlan1d(
                    &plans[static_cast<std::size_t>(level_index)],
                    static_cast<int>(config.slot_count),
                    use_fp64 ? CUFFT_Z2Z : CUFFT_C2C,
                    static_cast<int>(batch_count)
                ),
                "cufftPlan1d grouped multilimb planner"
            );
            current_nodes_for_plan /= 2ull;
        }
        const auto plan_end = std::chrono::steady_clock::now();
        report.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto execute_split_pass = [&](
                                           const std::uint32_t* current_level,
                                           std::uint32_t* next_level,
                                           std::size_t current_nodes,
                                           cufftHandle plan_handle,
                                           const SplitPassDescriptor& pass
                                       ) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t complex_value_count_level =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;
            if (use_fp64) {
                pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel_fp64<<<
                    block_count(complex_value_count_level),
                    kThreadsPerBlock>>>(
                    current_level,
                    d_work_a_fp64,
                    d_work_b_fp64,
                    current_nodes,
                    config.slot_count,
                    modulus_count,
                    pass.lhs_shift,
                    pass.lhs_mask,
                    pass.rhs_shift,
                    pass.rhs_mask
                );
                check_cuda(
                    cudaGetLastError(),
                    "pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel_fp64 multilimb launch"
                );
                check_cufft(cufftExecZ2Z(plan_handle, d_work_a_fp64, d_work_a_fp64, CUFFT_FORWARD), "cufftExecZ2Z multilimb forward a");
                check_cufft(cufftExecZ2Z(plan_handle, d_work_b_fp64, d_work_b_fp64, CUFFT_FORWARD), "cufftExecZ2Z multilimb forward b");
                complex_pointwise_multiply_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                    d_work_a_fp64,
                    d_work_b_fp64,
                    d_out_fp64,
                    complex_value_count_level
                );
                check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel_fp64 multilimb launch");
                check_cufft(cufftExecZ2Z(plan_handle, d_out_fp64, d_out_fp64, CUFFT_INVERSE), "cufftExecZ2Z multilimb inverse");
                scale_complex_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                    d_out_fp64,
                    1.0 / static_cast<double>(config.slot_count),
                    complex_value_count_level
                );
                check_cuda(cudaGetLastError(), "scale_complex_kernel_fp64 multilimb launch");
                accumulate_grouped_fft_output_to_level_values_kernel_fp64<<<
                    block_count(complex_value_count_level),
                    kThreadsPerBlock>>>(
                    d_out_fp64,
                    next_level,
                    merge_count,
                    config.slot_count,
                    modulus_count,
                    pass.accumulation_shift,
                    config.packing_mask
                );
                check_cuda(
                    cudaGetLastError(),
                    "accumulate_grouped_fft_output_to_level_values_kernel_fp64 multilimb launch"
                );
            } else {
                pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel<<<
                    block_count(complex_value_count_level),
                    kThreadsPerBlock>>>(
                    current_level,
                    d_work_a,
                    d_work_b,
                    current_nodes,
                    config.slot_count,
                    modulus_count,
                    pass.lhs_shift,
                    pass.lhs_mask,
                    pass.rhs_shift,
                    pass.rhs_mask
                );
                check_cuda(cudaGetLastError(), "pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel multilimb launch");
                check_cufft(cufftExecC2C(plan_handle, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecC2C multilimb forward a");
                check_cufft(cufftExecC2C(plan_handle, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecC2C multilimb forward b");
                complex_pointwise_multiply_kernel<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                    d_work_a,
                    d_work_b,
                    d_out,
                    complex_value_count_level
                );
                check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel multilimb launch");
                check_cufft(cufftExecC2C(plan_handle, d_out, d_out, CUFFT_INVERSE), "cufftExecC2C multilimb inverse");
                scale_complex_kernel<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                    d_out,
                    1.0f / static_cast<float>(config.slot_count),
                    complex_value_count_level
                );
                check_cuda(cudaGetLastError(), "scale_complex_kernel multilimb launch");
                accumulate_grouped_fft_output_to_level_values_kernel<<<
                    block_count(complex_value_count_level),
                    kThreadsPerBlock>>>(
                    d_out,
                    next_level,
                    merge_count,
                    config.slot_count,
                    modulus_count,
                    pass.accumulation_shift,
                    config.packing_mask
                );
                check_cuda(cudaGetLastError(), "accumulate_grouped_fft_output_to_level_values_kernel multilimb launch");
            }
        };

        const auto execute_single_level = [&](
                                              const std::uint32_t* current_level,
                                              std::uint32_t* next_level,
                                              std::size_t current_nodes,
                                              cufftHandle plan_handle
                                          ) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t parent_value_count =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;
            check_cuda(
                cudaMemset(next_level, 0, sizeof(std::uint32_t) * parent_value_count),
                "cudaMemset multilimb next_level"
            );
            for (const auto& pass : passes) {
                execute_split_pass(current_level, next_level, current_nodes, plan_handle, pass);
            }
        };

        {
            check_cuda(
                cudaMemcpy(d_level_a, d_original, level_bytes, cudaMemcpyDeviceToDevice),
                "cudaMemcpy multilimb planner verification reset"
            );
            check_cuda(
                cudaMemset(
                    d_level_b,
                    0,
                    sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count) * first_level_merge_count *
                        config.slot_count
                ),
                "cudaMemset multilimb planner verification next_level"
            );

            std::vector<std::uint32_t> level_samples(sample_modulus_count * sample_merge_count * config.slot_count);
            std::vector<std::vector<double>> pass_sample_reals(
                passes.size(),
                std::vector<double>(sample_storage_size, 0.0)
            );

            const auto storage_offset = [&](std::size_t modulus_index, std::size_t merge_index) {
                return (modulus_index * sample_merge_count + merge_index) * report.verification_sample_count;
            };
            const auto level_offset = [&](std::size_t modulus_index) {
                return modulus_index * sample_merge_count * config.slot_count;
            };
            const auto copy_fft_samples = [&](std::vector<double>& storage) {
                for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                    for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                        const std::size_t batch_index = modulus_index * first_level_merge_count + merge_index;
                        if (use_fp64) {
                            std::vector<cufftDoubleComplex> host(report.verification_sample_count);
                            check_cuda(
                                cudaMemcpy(
                                    host.data(),
                                    d_out_fp64 + batch_index * config.slot_count,
                                    sizeof(cufftDoubleComplex) * report.verification_sample_count,
                                    cudaMemcpyDeviceToHost
                                ),
                                "cudaMemcpy multilimb planner fft samples fp64"
                            );
                            for (std::size_t sample_index = 0; sample_index < report.verification_sample_count; ++sample_index) {
                                storage[storage_offset(modulus_index, merge_index) + sample_index] = host[sample_index].x;
                            }
                        } else {
                            std::vector<cufftComplex> host(report.verification_sample_count);
                            check_cuda(
                                cudaMemcpy(
                                    host.data(),
                                    d_out + batch_index * config.slot_count,
                                    sizeof(cufftComplex) * report.verification_sample_count,
                                    cudaMemcpyDeviceToHost
                                ),
                                "cudaMemcpy multilimb planner fft samples fp32"
                            );
                            for (std::size_t sample_index = 0; sample_index < report.verification_sample_count; ++sample_index) {
                                storage[storage_offset(modulus_index, merge_index) + sample_index] =
                                    static_cast<double>(host[sample_index].x);
                            }
                        }
                    }
                }
            };

            for (std::size_t pass_index = 0; pass_index < passes.size(); ++pass_index) {
                execute_split_pass(d_level_a, d_level_b, config.node_count, plans[0], passes[pass_index]);
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize multilimb verification pass");
                copy_fft_samples(pass_sample_reals[pass_index]);
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                check_cuda(
                    cudaMemcpy(
                        level_samples.data() + level_offset(modulus_index),
                        d_level_b + modulus_index * first_level_merge_count * config.slot_count,
                        sizeof(std::uint32_t) * sample_merge_count * config.slot_count,
                        cudaMemcpyDeviceToHost
                    ),
                    "cudaMemcpy multilimb planner verification samples"
                );
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                    for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                        std::uint64_t expected = 0;
                        for (std::size_t pass_index = 0; pass_index < passes.size(); ++pass_index) {
                            const auto& pass = passes[pass_index];
                            std::uint64_t expected_contribution = 0;
                            for (std::size_t inner_index = 0; inner_index < config.slot_count; ++inner_index) {
                                const std::size_t rhs_index =
                                    (output_index + config.slot_count - inner_index) % config.slot_count;
                                const auto lhs_value = expected_bridge_residue_value_u32(
                                    inner_index,
                                    merge_index * 2ull,
                                    modulus_index,
                                    config.packing_mask
                                );
                                const auto rhs_value = expected_bridge_residue_value_u32(
                                    rhs_index,
                                    merge_index * 2ull + 1ull,
                                    modulus_index,
                                    config.packing_mask
                                );
                                expected_contribution +=
                                    static_cast<std::uint64_t>((lhs_value >> pass.lhs_shift) & pass.lhs_mask) *
                                    static_cast<std::uint64_t>((rhs_value >> pass.rhs_shift) & pass.rhs_mask);
                            }
                            const std::size_t fft_index = storage_offset(modulus_index, merge_index) + output_index;
                            report.max_projection_real_error = std::max(
                                report.max_projection_real_error,
                                std::abs(
                                    pass_sample_reals[pass_index][fft_index] -
                                    static_cast<double>(expected_contribution)
                                )
                            );
                            expected += expected_contribution << pass.accumulation_shift;
                        }
                        expected &= static_cast<std::uint64_t>(config.packing_mask);

                        const std::size_t host_index =
                            level_offset(modulus_index) + merge_index * config.slot_count + output_index;
                        const std::uint32_t observed = level_samples[host_index];
                        if (observed != static_cast<std::uint32_t>(expected)) {
                            report.verification_mismatch_count += 1;
                            if (report.verification_mismatch_count == 1) {
                                report.first_mismatch_modulus = modulus_index;
                                report.first_mismatch_merge = merge_index;
                                report.first_mismatch_output = output_index;
                                report.first_mismatch_expected = static_cast<std::uint32_t>(expected);
                                report.first_mismatch_observed = observed;
                            }
                        }
                        report.verified_values += 1;
                    }
                }
            }
        }

        const auto run_pipeline_once = [&](double& elapsed_ms) {
            check_cuda(
                cudaMemcpy(d_level_a, d_original, level_bytes, cudaMemcpyDeviceToDevice),
                "cudaMemcpy multilimb planner reset"
            );
            check_cuda(cudaEventRecord(start), "cudaEventRecord multilimb planner pipeline start");

            std::uint32_t* current_level = d_level_a;
            std::uint32_t* next_level = d_level_b;
            std::size_t current_nodes = config.node_count;
            for (int level_index = 0; level_index < level_count; ++level_index) {
                execute_single_level(current_level, next_level, current_nodes, plans[static_cast<std::size_t>(level_index)]);
                std::swap(current_level, next_level);
                current_nodes /= 2ull;
            }

            check_cuda(cudaEventRecord(stop), "cudaEventRecord multilimb planner pipeline stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize multilimb planner pipeline stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime multilimb planner pipeline");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pipeline_once(ignored_ms);
        }

        run_pipeline_once(report.cold_pipeline_ms);

        double total_pipeline_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_pipeline_once(elapsed_ms);
            total_pipeline_ms += elapsed_ms;
        }
        report.avg_pipeline_ms = total_pipeline_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pipeline_ms > 0.0) {
            report.packed_residue_values_per_second =
                static_cast<double>(report.total_residue_values_packed) * 1000.0 / report.avg_pipeline_ms;
            report.transformed_complex_values_per_second =
                static_cast<double>(report.total_complex_value_count * 3ull) * 1000.0 / report.avg_pipeline_ms;
            report.logical_pipeline_bytes_per_second =
                static_cast<double>(logical_pipeline_bytes) * 1000.0 / report.avg_pipeline_ms;
        }

        report.ok = report.verification_mismatch_count == 0;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_smoke(const GroupedLevelPlannerConfig& config) {
    if (config.node_count < 2 || !is_power_of_two(config.node_count)) {
        throw std::invalid_argument("grouped exact-moduli planner node_count must be a power of two and >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("grouped exact-moduli planner slot_count must be a power of two and > 0");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }
    if (config.packing_mask != std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("grouped exact-moduli planner currently requires packing_mask = 4294967295");
    }

    constexpr std::uint32_t kLimbMask = 0xffffu;
    constexpr std::size_t kPassCount = 4;
    constexpr std::array<SplitPassDescriptor, kPassCount> kPasses = {
        SplitPassDescriptor{.lhs_shift = 0, .lhs_mask = kLimbMask, .rhs_shift = 0, .rhs_mask = kLimbMask, .accumulation_shift = 0},
        SplitPassDescriptor{.lhs_shift = 0, .lhs_mask = kLimbMask, .rhs_shift = 16, .rhs_mask = kLimbMask, .accumulation_shift = 16},
        SplitPassDescriptor{.lhs_shift = 16, .lhs_mask = kLimbMask, .rhs_shift = 0, .rhs_mask = kLimbMask, .accumulation_shift = 16},
        SplitPassDescriptor{.lhs_shift = 16, .lhs_mask = kLimbMask, .rhs_shift = 16, .rhs_mask = kLimbMask, .accumulation_shift = 32},
    };
    const bool source_is_chudnovsky_pfactor =
        config.source_mode == GroupedPlannerSourceMode::ChudnovskyPFactorLeaves;

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    const std::size_t level_value_count =
        static_cast<std::size_t>(modulus_count) * config.node_count * config.slot_count;
    const std::size_t level_bytes = sizeof(std::uint32_t) * level_value_count;
    const std::size_t max_fft_batch_count =
        static_cast<std::size_t>(modulus_count) * (config.node_count / 2ull);
    if (max_fft_batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped exact-moduli planner fft batch_count exceeds cuFFT int range");
    }
    if (config.slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped exact-moduli planner slot_count exceeds cuFFT int range");
    }
    const std::size_t max_complex_value_count = max_fft_batch_count * config.slot_count;
    const std::size_t complex_bytes = sizeof(cufftDoubleComplex) * max_complex_value_count;

    std::size_t total_fft_batch_count = 0;
    std::size_t total_complex_value_count = 0;
    int level_count = 0;
    for (std::size_t current_nodes = config.node_count; current_nodes > 1; current_nodes /= 2ull) {
        const std::size_t batch_count = static_cast<std::size_t>(modulus_count) * (current_nodes / 2ull);
        total_fft_batch_count += batch_count;
        total_complex_value_count += batch_count * config.slot_count;
        level_count += 1;
    }
    const std::size_t total_residue_values_packed = total_complex_value_count * 2ull;
    const std::size_t logical_pipeline_bytes = total_complex_value_count * sizeof(cufftDoubleComplex) * 14ull;
    const std::size_t sample_modulus_count = std::min<std::size_t>(2, static_cast<std::size_t>(modulus_count));
    const std::size_t first_level_merge_count = config.node_count / 2ull;
    const std::size_t sample_merge_count = std::min<std::size_t>(2, first_level_merge_count);
    const std::size_t sample_storage_size =
        sample_modulus_count * sample_merge_count * std::min(config.verification_sample_count, config.slot_count);
    const std::size_t max_pfactor_leaf_degree =
        source_is_chudnovsky_pfactor
            ? (coefficient_digit_length_base16(chudnovsky_pfactor_abs_value(config.node_count - 1ull)) - 1ull)
            : 0ull;
    const bool can_verify_pfactor_root_without_wrap =
        !source_is_chudnovsky_pfactor || (1ull + config.node_count * max_pfactor_leaf_degree <= config.slot_count);

    std::uint32_t* d_original = nullptr;
    std::uint32_t* d_level_a = nullptr;
    std::uint32_t* d_level_b = nullptr;
    std::uint32_t* d_moduli = nullptr;
    std::uint32_t* d_pass_weights = nullptr;
    cufftDoubleComplex* d_work_a = nullptr;
    cufftDoubleComplex* d_work_b = nullptr;
    cufftDoubleComplex* d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<cufftHandle> plans(static_cast<std::size_t>(level_count), 0);

    const auto cleanup = [&]() {
        for (cufftHandle& plan : plans) {
            if (plan != 0) {
                cufftDestroy(plan);
                plan = 0;
            }
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_pass_weights);
        cudaFree(d_moduli);
        cudaFree(d_level_b);
        cudaFree(d_level_a);
        cudaFree(d_original);
    };

    GroupedLevelPlannerReport report;
    report.source_layout =
        source_is_chudnovsky_pfactor
            ? "levels[modulus][node][slot]_base2^16_chudnovsky_pfactor_leaves"
            : "levels[modulus][node][slot]";
    report.fft_input_layout = "complex[level_modulus_merge_batch][slot]_exact_moduli_2x16_fp64";
    report.parent_layout = "levels[modulus][parent][slot]";
    report.operation =
        source_is_chudnovsky_pfactor
            ? "grouped_level_planner_exact_moduli_2limb_4pass_batched_cufft_tree_chudnovsky_pfactor_leaves"
            : "grouped_level_planner_exact_moduli_2limb_4pass_batched_cufft_tree";
    report.node_count = config.node_count;
    report.final_node_count = 1;
    report.level_count = level_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.packing_mask = config.packing_mask;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.total_fft_batch_count = total_fft_batch_count;
    report.total_residue_values_packed = total_residue_values_packed;
    report.total_complex_value_count = total_complex_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);
    report.split_limb_count = 2;
    report.split_pass_count = static_cast<int>(kPassCount);

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_original), level_bytes), "cudaMalloc exact-moduli d_original");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_level_a), level_bytes), "cudaMalloc exact-moduli d_level_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_level_b), level_bytes), "cudaMalloc exact-moduli d_level_b");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc exact-moduli d_moduli"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_pass_weights),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count) * kPassCount
            ),
            "cudaMalloc exact-moduli d_pass_weights"
        );
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes), "cudaMalloc exact-moduli d_work_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes), "cudaMalloc exact-moduli d_work_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes), "cudaMalloc exact-moduli d_out");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate exact-moduli start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate exact-moduli stop");
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy exact-moduli d_moduli"
        );

        std::vector<std::uint32_t> pass_weights_host(static_cast<std::size_t>(modulus_count) * kPassCount, 0u);
        for (int modulus_index = 0; modulus_index < modulus_count; ++modulus_index) {
            const std::uint32_t modulus = kDefaultModuli[static_cast<std::size_t>(modulus_index)];
            const std::uint32_t base_mod = static_cast<std::uint32_t>((1ull << 16) % modulus);
            const std::uint32_t base_sq_mod =
                static_cast<std::uint32_t>((static_cast<std::uint64_t>(base_mod) * base_mod) % modulus);
            pass_weights_host[0 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] = 1u % modulus;
            pass_weights_host[1 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] = base_mod;
            pass_weights_host[2 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] = base_mod;
            pass_weights_host[3 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] = base_sq_mod;
        }
        check_cuda(
            cudaMemcpy(
                d_pass_weights,
                pass_weights_host.data(),
                sizeof(std::uint32_t) * pass_weights_host.size(),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy exact-moduli d_pass_weights"
        );

        if (source_is_chudnovsky_pfactor) {
            initialize_chudnovsky_pfactor_leaves_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
                d_original,
                config.node_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "initialize_chudnovsky_pfactor_leaves_kernel exact-moduli launch");
        } else {
            initialize_modulus_major_residues_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
                d_original,
                d_moduli,
                config.node_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "initialize_modulus_major_residues_kernel exact-moduli launch");
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after exact-moduli init");

        const auto plan_begin = std::chrono::steady_clock::now();
        std::size_t current_nodes_for_plan = config.node_count;
        for (int level_index = 0; level_index < level_count; ++level_index) {
            const std::size_t batch_count =
                static_cast<std::size_t>(modulus_count) * (current_nodes_for_plan / 2ull);
            check_cufft(
                cufftPlan1d(
                    &plans[static_cast<std::size_t>(level_index)],
                    static_cast<int>(config.slot_count),
                    CUFFT_Z2Z,
                    static_cast<int>(batch_count)
                ),
                "cufftPlan1d grouped exact-moduli planner"
            );
            current_nodes_for_plan /= 2ull;
        }
        const auto plan_end = std::chrono::steady_clock::now();
        report.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto execute_pass = [&](
                                      const std::uint32_t* current_level,
                                      std::uint32_t* next_level,
                                      std::size_t current_nodes,
                                      cufftHandle plan_handle,
                                      std::size_t pass_index
                                  ) {
            const auto& pass = kPasses[pass_index];
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t complex_value_count_level =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;

            pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel_fp64<<<
                block_count(complex_value_count_level),
                kThreadsPerBlock>>>(
                current_level,
                d_work_a,
                d_work_b,
                current_nodes,
                config.slot_count,
                modulus_count,
                pass.lhs_shift,
                pass.lhs_mask,
                pass.rhs_shift,
                pass.rhs_mask
            );
            check_cuda(cudaGetLastError(), "pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel_fp64 exact-moduli launch");
            check_cufft(cufftExecZ2Z(plan_handle, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecZ2Z exact-moduli forward a");
            check_cufft(cufftExecZ2Z(plan_handle, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecZ2Z exact-moduli forward b");
            complex_pointwise_multiply_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_work_a,
                d_work_b,
                d_out,
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel_fp64 exact-moduli launch");
            check_cufft(cufftExecZ2Z(plan_handle, d_out, d_out, CUFFT_INVERSE), "cufftExecZ2Z exact-moduli inverse");
            scale_complex_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_out,
                1.0 / static_cast<double>(config.slot_count),
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "scale_complex_kernel_fp64 exact-moduli launch");
            accumulate_grouped_fft_output_to_level_values_mod_kernel_fp64<<<
                block_count(complex_value_count_level),
                kThreadsPerBlock>>>(
                d_out,
                next_level,
                d_moduli,
                d_pass_weights + pass_index * static_cast<std::size_t>(modulus_count),
                merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "accumulate_grouped_fft_output_to_level_values_mod_kernel_fp64 launch");
        };

        const auto execute_single_level = [&](
                                              const std::uint32_t* current_level,
                                              std::uint32_t* next_level,
                                              std::size_t current_nodes,
                                              cufftHandle plan_handle
                                          ) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t parent_value_count =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;
            check_cuda(
                cudaMemset(next_level, 0, sizeof(std::uint32_t) * parent_value_count),
                "cudaMemset exact-moduli next_level"
            );
            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_pass(current_level, next_level, current_nodes, plan_handle, pass_index);
            }
        };

        {
            check_cuda(
                cudaMemcpy(d_level_a, d_original, level_bytes, cudaMemcpyDeviceToDevice),
                "cudaMemcpy exact-moduli verification reset"
            );
            check_cuda(
                cudaMemset(
                    d_level_b,
                    0,
                    sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count) * first_level_merge_count *
                        config.slot_count
                ),
                "cudaMemset exact-moduli verification next_level"
            );

            std::vector<std::uint32_t> level_samples(sample_modulus_count * sample_merge_count * config.slot_count);
            std::vector<std::vector<double>> pass_sample_reals(
                kPassCount,
                std::vector<double>(sample_storage_size, 0.0)
            );

            const auto storage_offset = [&](std::size_t modulus_index, std::size_t merge_index) {
                return (modulus_index * sample_merge_count + merge_index) * report.verification_sample_count;
            };
            const auto level_offset = [&](std::size_t modulus_index) {
                return modulus_index * sample_merge_count * config.slot_count;
            };
            const auto copy_fft_samples = [&](std::vector<double>& storage) {
                for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                    for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                        const std::size_t batch_index = modulus_index * first_level_merge_count + merge_index;
                        std::vector<cufftDoubleComplex> host(report.verification_sample_count);
                        check_cuda(
                            cudaMemcpy(
                                host.data(),
                                d_out + batch_index * config.slot_count,
                                sizeof(cufftDoubleComplex) * report.verification_sample_count,
                                cudaMemcpyDeviceToHost
                            ),
                            "cudaMemcpy exact-moduli fft samples"
                        );
                        for (std::size_t sample_index = 0; sample_index < report.verification_sample_count; ++sample_index) {
                            storage[storage_offset(modulus_index, merge_index) + sample_index] = host[sample_index].x;
                        }
                    }
                }
            };

            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_pass(d_level_a, d_level_b, config.node_count, plans[0], pass_index);
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize exact-moduli verification pass");
                copy_fft_samples(pass_sample_reals[pass_index]);
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                check_cuda(
                    cudaMemcpy(
                        level_samples.data() + level_offset(modulus_index),
                        d_level_b + modulus_index * first_level_merge_count * config.slot_count,
                        sizeof(std::uint32_t) * sample_merge_count * config.slot_count,
                        cudaMemcpyDeviceToHost
                    ),
                    "cudaMemcpy exact-moduli verification samples"
                );
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                const std::uint32_t modulus = kDefaultModuli[modulus_index];
                for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                    for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                        std::uint64_t expected = 0;
                        for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                            const auto& pass = kPasses[pass_index];
                            std::uint64_t expected_contribution = 0;
                            for (std::size_t inner_index = 0; inner_index < config.slot_count; ++inner_index) {
                                const std::size_t rhs_index =
                                    (output_index + config.slot_count - inner_index) % config.slot_count;
                                const auto lhs_value =
                                    source_is_chudnovsky_pfactor
                                        ? expected_chudnovsky_pfactor_digit(inner_index, merge_index * 2ull)
                                        : expected_modulus_major_value(
                                              inner_index,
                                              merge_index * 2ull,
                                              modulus_index
                                          );
                                const auto rhs_value =
                                    source_is_chudnovsky_pfactor
                                        ? expected_chudnovsky_pfactor_digit(rhs_index, merge_index * 2ull + 1ull)
                                        : expected_modulus_major_value(
                                              rhs_index,
                                              merge_index * 2ull + 1ull,
                                              modulus_index
                                          );
                                expected_contribution +=
                                    static_cast<std::uint64_t>((lhs_value >> pass.lhs_shift) & pass.lhs_mask) *
                                    static_cast<std::uint64_t>((rhs_value >> pass.rhs_shift) & pass.rhs_mask);
                            }
                            const std::size_t fft_index = storage_offset(modulus_index, merge_index) + output_index;
                            report.max_projection_real_error = std::max(
                                report.max_projection_real_error,
                                std::abs(pass_sample_reals[pass_index][fft_index] - static_cast<double>(expected_contribution))
                            );
                            const std::uint64_t pass_weight =
                                static_cast<std::uint64_t>(
                                    pass_weights_host[pass_index * static_cast<std::size_t>(modulus_count) + modulus_index]
                                );
                            expected =
                                (expected +
                                 ((expected_contribution % static_cast<std::uint64_t>(modulus)) * pass_weight) %
                                     static_cast<std::uint64_t>(modulus)) %
                                static_cast<std::uint64_t>(modulus);
                        }

                        const std::size_t host_index =
                            level_offset(modulus_index) + merge_index * config.slot_count + output_index;
                        const std::uint32_t observed = level_samples[host_index];
                        if (observed != static_cast<std::uint32_t>(expected)) {
                            report.verification_mismatch_count += 1;
                            if (report.verification_mismatch_count == 1) {
                                report.first_mismatch_modulus = modulus_index;
                                report.first_mismatch_merge = merge_index;
                                report.first_mismatch_output = output_index;
                                report.first_mismatch_expected = static_cast<std::uint32_t>(expected);
                                report.first_mismatch_observed = observed;
                            }
                        }
                        report.verified_values += 1;
                    }
                }
            }

            if (source_is_chudnovsky_pfactor && can_verify_pfactor_root_without_wrap) {
                std::uint32_t* current_level = d_level_b;
                std::uint32_t* next_level = d_level_a;
                std::size_t current_nodes = first_level_merge_count;
                for (int level_index = 1; level_index < level_count; ++level_index) {
                    execute_single_level(
                        current_level,
                        next_level,
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );
                    std::swap(current_level, next_level);
                    current_nodes /= 2ull;
                }

                std::vector<std::uint32_t> root_samples(sample_modulus_count * config.slot_count, 0u);
                for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                    check_cuda(
                        cudaMemcpy(
                            root_samples.data() + modulus_index * config.slot_count,
                            current_level + modulus_index * config.slot_count,
                            sizeof(std::uint32_t) * config.slot_count,
                            cudaMemcpyDeviceToHost
                        ),
                        "cudaMemcpy exact-moduli pfactor root samples"
                    );
                }

                for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                    const std::uint32_t modulus = kDefaultModuli[modulus_index];
                    const std::uint32_t observed_root = evaluate_coefficient_vector_mod(
                        root_samples.data() + modulus_index * config.slot_count,
                        config.slot_count,
                        modulus
                    );
                    const std::uint32_t expected_root =
                        expected_chudnovsky_pfactor_product_mod(config.node_count, modulus);
                    if (observed_root != expected_root) {
                        report.verification_mismatch_count += 1;
                        if (report.verification_mismatch_count == 1) {
                            report.first_mismatch_modulus = modulus_index;
                            report.first_mismatch_merge = 0;
                            report.first_mismatch_output = 0;
                            report.first_mismatch_expected = expected_root;
                            report.first_mismatch_observed = observed_root;
                        }
                    }
                    report.verified_values += 1;
                }
            }
        }

        const auto run_pipeline_once = [&](double& elapsed_ms) {
            check_cuda(
                cudaMemcpy(d_level_a, d_original, level_bytes, cudaMemcpyDeviceToDevice),
                "cudaMemcpy exact-moduli reset"
            );
            check_cuda(cudaEventRecord(start), "cudaEventRecord exact-moduli pipeline start");

            std::uint32_t* current_level = d_level_a;
            std::uint32_t* next_level = d_level_b;
            std::size_t current_nodes = config.node_count;
            for (int level_index = 0; level_index < level_count; ++level_index) {
                execute_single_level(current_level, next_level, current_nodes, plans[static_cast<std::size_t>(level_index)]);
                std::swap(current_level, next_level);
                current_nodes /= 2ull;
            }

            check_cuda(cudaEventRecord(stop), "cudaEventRecord exact-moduli pipeline stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize exact-moduli pipeline stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime exact-moduli pipeline");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pipeline_once(ignored_ms);
        }

        run_pipeline_once(report.cold_pipeline_ms);

        double total_pipeline_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_pipeline_once(elapsed_ms);
            total_pipeline_ms += elapsed_ms;
        }
        report.avg_pipeline_ms = total_pipeline_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pipeline_ms > 0.0) {
            report.packed_residue_values_per_second =
                static_cast<double>(report.total_residue_values_packed) * 1000.0 / report.avg_pipeline_ms;
            report.transformed_complex_values_per_second =
                static_cast<double>(report.total_complex_value_count * 3ull) * 1000.0 / report.avg_pipeline_ms;
            report.logical_pipeline_bytes_per_second =
                static_cast<double>(logical_pipeline_bytes) * 1000.0 / report.avg_pipeline_ms;
        }

        report.ok = report.verification_mismatch_count == 0;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

GroupedLevelPlannerReport run_grouped_level_planner_split_mask63_smoke(const GroupedLevelPlannerConfig& config) {
    if (config.packing_mask != 63u) {
        throw std::invalid_argument("grouped split planner currently requires packing_mask = 63");
    }
    return run_grouped_level_planner_multilimb_smoke(config);
}

GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_pfactor_smoke(const GroupedLevelPlannerConfig& config) {
    GroupedLevelPlannerConfig derived = config;
    derived.source_mode = GroupedPlannerSourceMode::ChudnovskyPFactorLeaves;
    return run_grouped_level_planner_exact_moduli_smoke(derived);
}

GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_pq_smoke(const GroupedLevelPlannerConfig& config) {
    if (config.node_count < 2 || !is_power_of_two(config.node_count)) {
        throw std::invalid_argument("grouped exact-moduli pq planner node_count must be a power of two and >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("grouped exact-moduli pq planner slot_count must be a power of two and > 0");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }
    if (config.packing_mask != std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("grouped exact-moduli pq planner currently requires packing_mask = 4294967295");
    }

    constexpr std::uint32_t kLimbMask = 0xffffu;
    constexpr std::size_t kPassCount = 4;
    constexpr std::size_t kStreamCount = 2;
    constexpr std::array<SplitPassDescriptor, kPassCount> kPasses = {
        SplitPassDescriptor{.lhs_shift = 0, .lhs_mask = kLimbMask, .rhs_shift = 0, .rhs_mask = kLimbMask, .accumulation_shift = 0},
        SplitPassDescriptor{.lhs_shift = 0, .lhs_mask = kLimbMask, .rhs_shift = 16, .rhs_mask = kLimbMask, .accumulation_shift = 16},
        SplitPassDescriptor{.lhs_shift = 16, .lhs_mask = kLimbMask, .rhs_shift = 0, .rhs_mask = kLimbMask, .accumulation_shift = 16},
        SplitPassDescriptor{.lhs_shift = 16, .lhs_mask = kLimbMask, .rhs_shift = 16, .rhs_mask = kLimbMask, .accumulation_shift = 32},
    };

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    const std::size_t level_value_count =
        static_cast<std::size_t>(modulus_count) * config.node_count * config.slot_count;
    const std::size_t level_bytes = sizeof(std::uint32_t) * level_value_count;
    const std::size_t max_fft_batch_count =
        static_cast<std::size_t>(modulus_count) * (config.node_count / 2ull);
    if (max_fft_batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped exact-moduli pq planner fft batch_count exceeds cuFFT int range");
    }
    if (config.slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped exact-moduli pq planner slot_count exceeds cuFFT int range");
    }
    const std::size_t max_complex_value_count = max_fft_batch_count * config.slot_count;
    const std::size_t complex_bytes = sizeof(cufftDoubleComplex) * max_complex_value_count;

    std::size_t total_fft_batch_count_single = 0;
    std::size_t total_complex_value_count_single = 0;
    int level_count = 0;
    for (std::size_t current_nodes = config.node_count; current_nodes > 1; current_nodes /= 2ull) {
        const std::size_t batch_count = static_cast<std::size_t>(modulus_count) * (current_nodes / 2ull);
        total_fft_batch_count_single += batch_count;
        total_complex_value_count_single += batch_count * config.slot_count;
        level_count += 1;
    }
    const std::size_t total_fft_batch_count = total_fft_batch_count_single * kStreamCount;
    const std::size_t total_complex_value_count = total_complex_value_count_single * kStreamCount;
    const std::size_t total_residue_values_packed = total_complex_value_count * 2ull;
    const std::size_t logical_pipeline_bytes =
        total_complex_value_count * sizeof(cufftDoubleComplex) * 14ull;
    const std::size_t sample_modulus_count = std::min<std::size_t>(2, static_cast<std::size_t>(modulus_count));
    const std::size_t first_level_merge_count = config.node_count / 2ull;
    const std::size_t sample_merge_count = std::min<std::size_t>(2, first_level_merge_count);
    const std::size_t sample_storage_size =
        sample_modulus_count * sample_merge_count * std::min(config.verification_sample_count, config.slot_count);
    const std::size_t max_pfactor_leaf_degree =
        coefficient_digit_length_base16(chudnovsky_pfactor_abs_value(config.node_count - 1ull)) - 1ull;
    const std::size_t max_qfactor_leaf_degree =
        chudnovsky_qfactor_digit_length_base16(config.node_count - 1ull) - 1ull;
    const std::array<bool, kStreamCount> can_verify_root_without_wrap = {
        1ull + config.node_count * max_pfactor_leaf_degree <= config.slot_count,
        1ull + config.node_count * max_qfactor_leaf_degree <= config.slot_count,
    };

    std::array<std::uint32_t*, kStreamCount> d_original = {nullptr, nullptr};
    std::array<std::uint32_t*, kStreamCount> d_level_a = {nullptr, nullptr};
    std::array<std::uint32_t*, kStreamCount> d_level_b = {nullptr, nullptr};
    std::uint32_t* d_moduli = nullptr;
    std::uint32_t* d_pass_weights = nullptr;
    cufftDoubleComplex* d_work_a = nullptr;
    cufftDoubleComplex* d_work_b = nullptr;
    cufftDoubleComplex* d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<cufftHandle> plans(static_cast<std::size_t>(level_count), 0);

    const auto cleanup = [&]() {
        for (cufftHandle& plan : plans) {
            if (plan != 0) {
                cufftDestroy(plan);
                plan = 0;
            }
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_pass_weights);
        cudaFree(d_moduli);
        for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
            cudaFree(d_level_b[stream_index]);
            cudaFree(d_level_a[stream_index]);
            cudaFree(d_original[stream_index]);
        }
    };

    GroupedLevelPlannerReport report;
    report.source_layout = "levels[stream=p|q][modulus][node][slot]_base2^16_chudnovsky_leaves";
    report.fft_input_layout = "complex[level_modulus_merge_batch][slot]_exact_moduli_2stream_pq_2x16_fp64";
    report.parent_layout = "levels[stream][modulus][parent][slot]";
    report.operation = "grouped_level_planner_exact_moduli_2stream_pq_2limb_4pass_batched_cufft_tree";
    report.node_count = config.node_count;
    report.final_node_count = 1;
    report.level_count = level_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.packing_mask = config.packing_mask;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.total_fft_batch_count = total_fft_batch_count;
    report.total_residue_values_packed = total_residue_values_packed;
    report.total_complex_value_count = total_complex_value_count;
    report.verification_sample_count = std::min(config.verification_sample_count, config.slot_count);
    report.split_limb_count = 2;
    report.split_pass_count = static_cast<int>(kPassCount);

    try {
        for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_original[stream_index]), level_bytes),
                "cudaMalloc exact-moduli pq d_original"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_level_a[stream_index]), level_bytes),
                "cudaMalloc exact-moduli pq d_level_a"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_level_b[stream_index]), level_bytes),
                "cudaMalloc exact-moduli pq d_level_b"
            );
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc exact-moduli pq d_moduli"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_pass_weights),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count) * kPassCount
            ),
            "cudaMalloc exact-moduli pq d_pass_weights"
        );
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes), "cudaMalloc exact-moduli pq d_work_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes), "cudaMalloc exact-moduli pq d_work_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes), "cudaMalloc exact-moduli pq d_out");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate exact-moduli pq start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate exact-moduli pq stop");
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy exact-moduli pq d_moduli"
        );

        std::vector<std::uint32_t> pass_weights_host(static_cast<std::size_t>(modulus_count) * kPassCount, 0u);
        for (int modulus_index = 0; modulus_index < modulus_count; ++modulus_index) {
            const std::uint32_t modulus = kDefaultModuli[static_cast<std::size_t>(modulus_index)];
            const std::uint32_t base_mod = static_cast<std::uint32_t>((1ull << 16) % modulus);
            const std::uint32_t base_sq_mod =
                static_cast<std::uint32_t>((static_cast<std::uint64_t>(base_mod) * base_mod) % modulus);
            pass_weights_host[0 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] =
                1u % modulus;
            pass_weights_host[1 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_mod;
            pass_weights_host[2 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_mod;
            pass_weights_host[3 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_sq_mod;
        }
        check_cuda(
            cudaMemcpy(
                d_pass_weights,
                pass_weights_host.data(),
                sizeof(std::uint32_t) * pass_weights_host.size(),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy exact-moduli pq d_pass_weights"
        );

        initialize_chudnovsky_pfactor_leaves_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_original[0],
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_chudnovsky_pfactor_leaves_kernel exact-moduli pq launch");
        initialize_chudnovsky_qfactor_leaves_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_original[1],
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_chudnovsky_qfactor_leaves_kernel exact-moduli pq launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after exact-moduli pq init");

        const auto plan_begin = std::chrono::steady_clock::now();
        std::size_t current_nodes_for_plan = config.node_count;
        for (int level_index = 0; level_index < level_count; ++level_index) {
            const std::size_t batch_count =
                static_cast<std::size_t>(modulus_count) * (current_nodes_for_plan / 2ull);
            check_cufft(
                cufftPlan1d(
                    &plans[static_cast<std::size_t>(level_index)],
                    static_cast<int>(config.slot_count),
                    CUFFT_Z2Z,
                    static_cast<int>(batch_count)
                ),
                "cufftPlan1d grouped exact-moduli pq planner"
            );
            current_nodes_for_plan /= 2ull;
        }
        const auto plan_end = std::chrono::steady_clock::now();
        report.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto execute_pass = [&](
                                      const std::uint32_t* current_level,
                                      std::uint32_t* next_level,
                                      std::size_t current_nodes,
                                      cufftHandle plan_handle,
                                      std::size_t pass_index
                                  ) {
            const auto& pass = kPasses[pass_index];
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t complex_value_count_level =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;

            pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel_fp64<<<
                block_count(complex_value_count_level),
                kThreadsPerBlock>>>(
                current_level,
                d_work_a,
                d_work_b,
                current_nodes,
                config.slot_count,
                modulus_count,
                pass.lhs_shift,
                pass.lhs_mask,
                pass.rhs_shift,
                pass.rhs_mask
            );
            check_cuda(cudaGetLastError(), "pack_grouped_modulus_major_pairs_to_fft_limb_inputs_kernel_fp64 exact-moduli pq launch");
            check_cufft(cufftExecZ2Z(plan_handle, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecZ2Z exact-moduli pq forward a");
            check_cufft(cufftExecZ2Z(plan_handle, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecZ2Z exact-moduli pq forward b");
            complex_pointwise_multiply_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_work_a,
                d_work_b,
                d_out,
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel_fp64 exact-moduli pq launch");
            check_cufft(cufftExecZ2Z(plan_handle, d_out, d_out, CUFFT_INVERSE), "cufftExecZ2Z exact-moduli pq inverse");
            scale_complex_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_out,
                1.0 / static_cast<double>(config.slot_count),
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "scale_complex_kernel_fp64 exact-moduli pq launch");
            accumulate_grouped_fft_output_to_level_values_mod_kernel_fp64<<<
                block_count(complex_value_count_level),
                kThreadsPerBlock>>>(
                d_out,
                next_level,
                d_moduli,
                d_pass_weights + pass_index * static_cast<std::size_t>(modulus_count),
                merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "accumulate_grouped_fft_output_to_level_values_mod_kernel_fp64 exact-moduli pq launch");
        };

        const auto execute_single_level = [&](
                                              const std::uint32_t* current_level,
                                              std::uint32_t* next_level,
                                              std::size_t current_nodes,
                                              cufftHandle plan_handle
                                          ) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t parent_value_count =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;
            check_cuda(
                cudaMemset(next_level, 0, sizeof(std::uint32_t) * parent_value_count),
                "cudaMemset exact-moduli pq next_level"
            );
            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_pass(current_level, next_level, current_nodes, plan_handle, pass_index);
            }
        };

        for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
            check_cuda(
                cudaMemcpy(d_level_a[stream_index], d_original[stream_index], level_bytes, cudaMemcpyDeviceToDevice),
                "cudaMemcpy exact-moduli pq verification reset"
            );
            check_cuda(
                cudaMemset(
                    d_level_b[stream_index],
                    0,
                    sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count) * first_level_merge_count *
                        config.slot_count
                ),
                "cudaMemset exact-moduli pq verification next_level"
            );

            std::vector<std::uint32_t> level_samples(sample_modulus_count * sample_merge_count * config.slot_count);
            std::vector<std::vector<double>> pass_sample_reals(
                kPassCount,
                std::vector<double>(sample_storage_size, 0.0)
            );

            const auto storage_offset = [&](std::size_t modulus_index, std::size_t merge_index) {
                return (modulus_index * sample_merge_count + merge_index) * report.verification_sample_count;
            };
            const auto level_offset = [&](std::size_t modulus_index) {
                return modulus_index * sample_merge_count * config.slot_count;
            };
            const auto copy_fft_samples = [&](std::vector<double>& storage) {
                for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                    for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                        const std::size_t batch_index = modulus_index * first_level_merge_count + merge_index;
                        std::vector<cufftDoubleComplex> host(report.verification_sample_count);
                        check_cuda(
                            cudaMemcpy(
                                host.data(),
                                d_out + batch_index * config.slot_count,
                                sizeof(cufftDoubleComplex) * report.verification_sample_count,
                                cudaMemcpyDeviceToHost
                            ),
                            "cudaMemcpy exact-moduli pq fft samples"
                        );
                        for (std::size_t sample_index = 0; sample_index < report.verification_sample_count; ++sample_index) {
                            storage[storage_offset(modulus_index, merge_index) + sample_index] = host[sample_index].x;
                        }
                    }
                }
            };

            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_pass(d_level_a[stream_index], d_level_b[stream_index], config.node_count, plans[0], pass_index);
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize exact-moduli pq verification pass");
                copy_fft_samples(pass_sample_reals[pass_index]);
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                check_cuda(
                    cudaMemcpy(
                        level_samples.data() + level_offset(modulus_index),
                        d_level_b[stream_index] + modulus_index * first_level_merge_count * config.slot_count,
                        sizeof(std::uint32_t) * sample_merge_count * config.slot_count,
                        cudaMemcpyDeviceToHost
                    ),
                    "cudaMemcpy exact-moduli pq verification samples"
                );
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                const std::uint32_t modulus = kDefaultModuli[modulus_index];
                for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                    for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                        std::uint64_t expected = 0;
                        for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                            const auto& pass = kPasses[pass_index];
                            std::uint64_t expected_contribution = 0;
                            for (std::size_t inner_index = 0; inner_index < config.slot_count; ++inner_index) {
                                const std::size_t rhs_index =
                                    (output_index + config.slot_count - inner_index) % config.slot_count;
                                const auto lhs_value =
                                    stream_index == 0
                                        ? expected_chudnovsky_pfactor_digit(inner_index, merge_index * 2ull)
                                        : expected_chudnovsky_qfactor_digit(inner_index, merge_index * 2ull);
                                const auto rhs_value =
                                    stream_index == 0
                                        ? expected_chudnovsky_pfactor_digit(rhs_index, merge_index * 2ull + 1ull)
                                        : expected_chudnovsky_qfactor_digit(rhs_index, merge_index * 2ull + 1ull);
                                expected_contribution +=
                                    static_cast<std::uint64_t>((lhs_value >> pass.lhs_shift) & pass.lhs_mask) *
                                    static_cast<std::uint64_t>((rhs_value >> pass.rhs_shift) & pass.rhs_mask);
                            }
                            const std::size_t fft_index = storage_offset(modulus_index, merge_index) + output_index;
                            report.max_projection_real_error = std::max(
                                report.max_projection_real_error,
                                std::abs(pass_sample_reals[pass_index][fft_index] - static_cast<double>(expected_contribution))
                            );
                            const std::uint64_t pass_weight =
                                static_cast<std::uint64_t>(
                                    pass_weights_host[pass_index * static_cast<std::size_t>(modulus_count) + modulus_index]
                                );
                            expected =
                                (expected +
                                 ((expected_contribution % static_cast<std::uint64_t>(modulus)) * pass_weight) %
                                     static_cast<std::uint64_t>(modulus)) %
                                static_cast<std::uint64_t>(modulus);
                        }

                        const std::size_t host_index =
                            level_offset(modulus_index) + merge_index * config.slot_count + output_index;
                        const std::uint32_t observed = level_samples[host_index];
                        if (observed != static_cast<std::uint32_t>(expected)) {
                            report.verification_mismatch_count += 1;
                            if (report.verification_mismatch_count == 1) {
                                report.first_mismatch_modulus = modulus_index;
                                report.first_mismatch_merge = merge_index;
                                report.first_mismatch_output = output_index;
                                report.first_mismatch_expected = static_cast<std::uint32_t>(expected);
                                report.first_mismatch_observed = observed;
                            }
                        }
                        report.verified_values += 1;
                    }
                }
            }

            if (can_verify_root_without_wrap[stream_index]) {
                std::uint32_t* current_level = d_level_b[stream_index];
                std::uint32_t* next_level = d_level_a[stream_index];
                std::size_t current_nodes = first_level_merge_count;
                for (int level_index = 1; level_index < level_count; ++level_index) {
                    execute_single_level(
                        current_level,
                        next_level,
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );
                    std::swap(current_level, next_level);
                    current_nodes /= 2ull;
                }

                std::vector<std::uint32_t> root_samples(sample_modulus_count * config.slot_count, 0u);
                for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                    check_cuda(
                        cudaMemcpy(
                            root_samples.data() + modulus_index * config.slot_count,
                            current_level + modulus_index * config.slot_count,
                            sizeof(std::uint32_t) * config.slot_count,
                            cudaMemcpyDeviceToHost
                        ),
                        "cudaMemcpy exact-moduli pq root samples"
                    );
                }

                for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                    const std::uint32_t modulus = kDefaultModuli[modulus_index];
                    const std::uint32_t observed_root = evaluate_coefficient_vector_mod(
                        root_samples.data() + modulus_index * config.slot_count,
                        config.slot_count,
                        modulus
                    );
                    const std::uint32_t expected_root =
                        stream_index == 0
                            ? expected_chudnovsky_pfactor_product_mod(config.node_count, modulus)
                            : expected_chudnovsky_qfactor_product_mod(config.node_count, modulus);
                    if (observed_root != expected_root) {
                        report.verification_mismatch_count += 1;
                        if (report.verification_mismatch_count == 1) {
                            report.first_mismatch_modulus = modulus_index;
                            report.first_mismatch_merge = 0;
                            report.first_mismatch_output = stream_index;
                            report.first_mismatch_expected = expected_root;
                            report.first_mismatch_observed = observed_root;
                        }
                    }
                    report.verified_values += 1;
                }
            }
        }

        const auto run_pipeline_once = [&](double& elapsed_ms) {
            for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                check_cuda(
                    cudaMemcpy(
                        d_level_a[stream_index],
                        d_original[stream_index],
                        level_bytes,
                        cudaMemcpyDeviceToDevice
                    ),
                    "cudaMemcpy exact-moduli pq reset"
                );
            }
            check_cuda(cudaEventRecord(start), "cudaEventRecord exact-moduli pq pipeline start");

            std::array<std::uint32_t*, kStreamCount> current_levels = d_level_a;
            std::array<std::uint32_t*, kStreamCount> next_levels = d_level_b;
            std::size_t current_nodes = config.node_count;
            for (int level_index = 0; level_index < level_count; ++level_index) {
                for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                    execute_single_level(
                        current_levels[stream_index],
                        next_levels[stream_index],
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );
                }
                std::swap(current_levels, next_levels);
                current_nodes /= 2ull;
            }

            check_cuda(cudaEventRecord(stop), "cudaEventRecord exact-moduli pq pipeline stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize exact-moduli pq pipeline stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime exact-moduli pq pipeline");
            elapsed_ms = static_cast<double>(elapsed_ms_f);
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pipeline_once(ignored_ms);
        }

        run_pipeline_once(report.cold_pipeline_ms);

        double total_pipeline_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            run_pipeline_once(elapsed_ms);
            total_pipeline_ms += elapsed_ms;
        }
        report.avg_pipeline_ms = total_pipeline_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pipeline_ms > 0.0) {
            report.packed_residue_values_per_second =
                static_cast<double>(report.total_residue_values_packed) * 1000.0 / report.avg_pipeline_ms;
            report.transformed_complex_values_per_second =
                static_cast<double>(report.total_complex_value_count * 3ull) * 1000.0 / report.avg_pipeline_ms;
            report.logical_pipeline_bytes_per_second =
                static_cast<double>(logical_pipeline_bytes) * 1000.0 / report.avg_pipeline_ms;
        }

        report.ok = report.verification_mismatch_count == 0;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

struct CapturedExactModuliPqtRoots {
    std::array<std::vector<std::uint32_t>, 3> roots;
};

GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_pqt_smoke_impl(
    const GroupedLevelPlannerConfig& config,
    CapturedExactModuliPqtRoots* captured_roots
) {
    if (config.node_count < 2 || !is_power_of_two(config.node_count)) {
        throw std::invalid_argument("grouped exact-moduli pqt planner node_count must be a power of two and >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("grouped exact-moduli pqt planner slot_count must be a power of two and > 0");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }
    if (config.packing_mask != std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("grouped exact-moduli pqt planner currently requires packing_mask = 4294967295");
    }

    constexpr std::uint32_t kLimbMask = 0xffffu;
    constexpr std::size_t kPassCount = 4;
    constexpr std::size_t kStreamCount = 3;
    constexpr std::size_t kEquivalentConvolutionStreamCount = 4;
    constexpr std::size_t kPStreamIndex = 0;
    constexpr std::size_t kQStreamIndex = 1;
    constexpr std::size_t kTStreamIndex = 2;
    constexpr std::array<SplitPassDescriptor, kPassCount> kPasses = {
        SplitPassDescriptor{.lhs_shift = 0, .lhs_mask = kLimbMask, .rhs_shift = 0, .rhs_mask = kLimbMask, .accumulation_shift = 0},
        SplitPassDescriptor{.lhs_shift = 0, .lhs_mask = kLimbMask, .rhs_shift = 16, .rhs_mask = kLimbMask, .accumulation_shift = 16},
        SplitPassDescriptor{.lhs_shift = 16, .lhs_mask = kLimbMask, .rhs_shift = 0, .rhs_mask = kLimbMask, .accumulation_shift = 16},
        SplitPassDescriptor{.lhs_shift = 16, .lhs_mask = kLimbMask, .rhs_shift = 16, .rhs_mask = kLimbMask, .accumulation_shift = 32},
    };

    const int modulus_count = normalized_modulus_count(config.modulus_count);
    const std::size_t level_value_count =
        static_cast<std::size_t>(modulus_count) * config.node_count * config.slot_count;
    const std::size_t level_bytes = sizeof(std::uint32_t) * level_value_count;
    const std::size_t max_fft_batch_count =
        static_cast<std::size_t>(modulus_count) * (config.node_count / 2ull);
    if (max_fft_batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped exact-moduli pqt planner fft batch_count exceeds cuFFT int range");
    }
    if (config.slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("grouped exact-moduli pqt planner slot_count exceeds cuFFT int range");
    }
    const std::size_t max_complex_value_count = max_fft_batch_count * config.slot_count;
    const std::size_t complex_bytes = sizeof(cufftDoubleComplex) * max_complex_value_count;

    std::size_t total_fft_batch_count_single = 0;
    std::size_t total_complex_value_count_single = 0;
    int level_count = 0;
    for (std::size_t current_nodes = config.node_count; current_nodes > 1; current_nodes /= 2ull) {
        const std::size_t batch_count = static_cast<std::size_t>(modulus_count) * (current_nodes / 2ull);
        total_fft_batch_count_single += batch_count;
        total_complex_value_count_single += batch_count * config.slot_count;
        level_count += 1;
    }
    const std::size_t total_fft_batch_count = total_fft_batch_count_single * kEquivalentConvolutionStreamCount;
    const std::size_t total_complex_value_count = total_complex_value_count_single * kEquivalentConvolutionStreamCount;
    const std::size_t total_residue_values_packed = total_complex_value_count * 2ull;
    const std::size_t logical_pipeline_bytes =
        total_complex_value_count * sizeof(cufftDoubleComplex) * 14ull;
    const std::size_t sample_modulus_count = std::min<std::size_t>(2, static_cast<std::size_t>(modulus_count));
    const std::size_t first_level_merge_count = config.node_count / 2ull;
    const std::size_t sample_merge_count = std::min<std::size_t>(2, first_level_merge_count);
    const std::size_t verified_sample_count = std::min(config.verification_sample_count, config.slot_count);
    const std::size_t sample_storage_size = sample_modulus_count * sample_merge_count * verified_sample_count;
    const std::size_t max_exact_pfactor_leaf_degree =
        coefficient_digit_length_base16(chudnovsky_exact_pfactor_abs_value(config.node_count - 1ull)) - 1ull;
    const std::size_t max_qfactor_leaf_degree =
        chudnovsky_qfactor_digit_length_base16(config.node_count - 1ull) - 1ull;
    const std::size_t max_tfactor_leaf_degree =
        coefficient_digit_length_base16_u128(chudnovsky_tfactor_abs_value(config.node_count - 1ull)) - 1ull;
    const std::size_t max_combined_leaf_degree = std::max(
        max_tfactor_leaf_degree,
        std::max(max_exact_pfactor_leaf_degree, max_qfactor_leaf_degree)
    );
    const std::array<bool, kStreamCount> can_verify_root_without_wrap = {
        1ull + config.node_count * max_exact_pfactor_leaf_degree <= config.slot_count,
        1ull + config.node_count * max_qfactor_leaf_degree <= config.slot_count,
        1ull + config.node_count * max_combined_leaf_degree <= config.slot_count,
    };

    std::array<std::uint32_t*, kStreamCount> d_original = {nullptr, nullptr, nullptr};
    std::array<std::uint32_t*, kStreamCount> d_level_a = {nullptr, nullptr, nullptr};
    std::array<std::uint32_t*, kStreamCount> d_level_b = {nullptr, nullptr, nullptr};
    std::uint32_t* d_moduli = nullptr;
    std::uint32_t* d_pass_weights = nullptr;
    cufftDoubleComplex* d_work_a = nullptr;
    cufftDoubleComplex* d_work_b = nullptr;
    cufftDoubleComplex* d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<cufftHandle> plans(static_cast<std::size_t>(level_count), 0);

    const auto cleanup = [&]() {
        for (cufftHandle& plan : plans) {
            if (plan != 0) {
                cufftDestroy(plan);
                plan = 0;
            }
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_pass_weights);
        cudaFree(d_moduli);
        for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
            cudaFree(d_level_b[stream_index]);
            cudaFree(d_level_a[stream_index]);
            cudaFree(d_original[stream_index]);
        }
    };

    GroupedLevelPlannerReport report;
    report.source_layout = "levels[stream=p|q|t][modulus][node][slot]_base2^16_exact_chudnovsky_leaves";
    report.fft_input_layout = "complex[level_modulus_merge_batch][slot]_exact_moduli_3stream_pqt_2x16_fp64";
    report.parent_layout = "levels[stream][modulus][parent][slot]";
    report.operation = "grouped_level_planner_exact_moduli_3stream_pqt_2limb_4pass_batched_cufft_tree";
    report.node_count = config.node_count;
    report.final_node_count = 1;
    report.level_count = level_count;
    report.slot_count = config.slot_count;
    report.modulus_count = modulus_count;
    report.packing_mask = config.packing_mask;
    report.warmup_iterations = config.warmup_iterations;
    report.measured_iterations = config.measured_iterations;
    report.total_fft_batch_count = total_fft_batch_count;
    report.total_residue_values_packed = total_residue_values_packed;
    report.total_complex_value_count = total_complex_value_count;
    report.verification_sample_count = verified_sample_count;
    report.split_limb_count = 2;
    report.split_pass_count = static_cast<int>(kPassCount);
    if (captured_roots != nullptr) {
        for (auto& root_stream : captured_roots->roots) {
            root_stream.clear();
        }
    }

    try {
        for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_original[stream_index]), level_bytes),
                "cudaMalloc exact-moduli pqt d_original"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_level_a[stream_index]), level_bytes),
                "cudaMalloc exact-moduli pqt d_level_a"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_level_b[stream_index]), level_bytes),
                "cudaMalloc exact-moduli pqt d_level_b"
            );
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&d_moduli), sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count)),
            "cudaMalloc exact-moduli pqt d_moduli"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_pass_weights),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count) * kPassCount
            ),
            "cudaMalloc exact-moduli pqt d_pass_weights"
        );
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes), "cudaMalloc exact-moduli pqt d_work_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes), "cudaMalloc exact-moduli pqt d_work_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes), "cudaMalloc exact-moduli pqt d_out");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate exact-moduli pqt start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate exact-moduli pqt stop");
        check_cuda(
            cudaMemcpy(
                d_moduli,
                kDefaultModuli.data(),
                sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy exact-moduli pqt d_moduli"
        );

        std::vector<std::uint32_t> pass_weights_host(static_cast<std::size_t>(modulus_count) * kPassCount, 0u);
        for (int modulus_index = 0; modulus_index < modulus_count; ++modulus_index) {
            const std::uint32_t modulus = kDefaultModuli[static_cast<std::size_t>(modulus_index)];
            const std::uint32_t base_mod = static_cast<std::uint32_t>((1ull << 16) % modulus);
            const std::uint32_t base_sq_mod =
                static_cast<std::uint32_t>((static_cast<std::uint64_t>(base_mod) * base_mod) % modulus);
            pass_weights_host[0 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] =
                1u % modulus;
            pass_weights_host[1 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_mod;
            pass_weights_host[2 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_mod;
            pass_weights_host[3 * static_cast<std::size_t>(modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_sq_mod;
        }
        check_cuda(
            cudaMemcpy(
                d_pass_weights,
                pass_weights_host.data(),
                sizeof(std::uint32_t) * pass_weights_host.size(),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy exact-moduli pqt d_pass_weights"
        );

        initialize_chudnovsky_exact_pfactor_leaves_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_original[kPStreamIndex],
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_chudnovsky_exact_pfactor_leaves_kernel exact-moduli pqt launch");
        initialize_chudnovsky_qfactor_leaves_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_original[kQStreamIndex],
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_chudnovsky_qfactor_leaves_kernel exact-moduli pqt launch");
        initialize_chudnovsky_tfactor_leaves_kernel<<<block_count(level_value_count), kThreadsPerBlock>>>(
            d_original[kTStreamIndex],
            d_moduli,
            config.node_count,
            config.slot_count,
            modulus_count
        );
        check_cuda(cudaGetLastError(), "initialize_chudnovsky_tfactor_leaves_kernel exact-moduli pqt launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after exact-moduli pqt init");

        const auto plan_begin = std::chrono::steady_clock::now();
        std::size_t current_nodes_for_plan = config.node_count;
        for (int level_index = 0; level_index < level_count; ++level_index) {
            const std::size_t batch_count =
                static_cast<std::size_t>(modulus_count) * (current_nodes_for_plan / 2ull);
            check_cufft(
                cufftPlan1d(
                    &plans[static_cast<std::size_t>(level_index)],
                    static_cast<int>(config.slot_count),
                    CUFFT_Z2Z,
                    static_cast<int>(batch_count)
                ),
                "cufftPlan1d grouped exact-moduli pqt planner"
            );
            current_nodes_for_plan /= 2ull;
        }
        const auto plan_end = std::chrono::steady_clock::now();
        report.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto execute_cross_pass = [&](
                                            const std::uint32_t* lhs_level,
                                            const std::uint32_t* rhs_level,
                                            std::uint32_t* next_level,
                                            std::size_t current_nodes,
                                            cufftHandle plan_handle,
                                            std::size_t pass_index
                                        ) {
            const auto& pass = kPasses[pass_index];
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t complex_value_count_level =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;

            pack_grouped_modulus_major_cross_pairs_to_fft_limb_inputs_kernel_fp64<<<
                block_count(complex_value_count_level),
                kThreadsPerBlock>>>(
                lhs_level,
                rhs_level,
                d_work_a,
                d_work_b,
                current_nodes,
                config.slot_count,
                modulus_count,
                pass.lhs_shift,
                pass.lhs_mask,
                pass.rhs_shift,
                pass.rhs_mask
            );
            check_cuda(cudaGetLastError(), "pack_grouped_modulus_major_cross_pairs_to_fft_limb_inputs_kernel_fp64 exact-moduli pqt launch");
            check_cufft(cufftExecZ2Z(plan_handle, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecZ2Z exact-moduli pqt forward a");
            check_cufft(cufftExecZ2Z(plan_handle, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecZ2Z exact-moduli pqt forward b");
            complex_pointwise_multiply_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_work_a,
                d_work_b,
                d_out,
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel_fp64 exact-moduli pqt launch");
            check_cufft(cufftExecZ2Z(plan_handle, d_out, d_out, CUFFT_INVERSE), "cufftExecZ2Z exact-moduli pqt inverse");
            scale_complex_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_out,
                1.0 / static_cast<double>(config.slot_count),
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "scale_complex_kernel_fp64 exact-moduli pqt launch");
            accumulate_grouped_fft_output_to_level_values_mod_kernel_fp64<<<
                block_count(complex_value_count_level),
                kThreadsPerBlock>>>(
                d_out,
                next_level,
                d_moduli,
                d_pass_weights + pass_index * static_cast<std::size_t>(modulus_count),
                merge_count,
                config.slot_count,
                modulus_count
            );
            check_cuda(cudaGetLastError(), "accumulate_grouped_fft_output_to_level_values_mod_kernel_fp64 exact-moduli pqt launch");
        };

        const auto execute_product_level = [&](
                                               const std::uint32_t* current_level,
                                               std::uint32_t* next_level,
                                               std::size_t current_nodes,
                                               cufftHandle plan_handle
                                           ) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t parent_value_count =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;
            check_cuda(
                cudaMemset(next_level, 0, sizeof(std::uint32_t) * parent_value_count),
                "cudaMemset exact-moduli pqt product next_level"
            );
            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_cross_pass(current_level, current_level, next_level, current_nodes, plan_handle, pass_index);
            }
        };

        const auto execute_t_level = [&](
                                         const std::uint32_t* current_t,
                                         const std::uint32_t* current_p,
                                         const std::uint32_t* current_q,
                                         std::uint32_t* next_t,
                                         std::size_t current_nodes,
                                         cufftHandle plan_handle
                                     ) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t parent_value_count =
                static_cast<std::size_t>(modulus_count) * merge_count * config.slot_count;
            check_cuda(
                cudaMemset(next_t, 0, sizeof(std::uint32_t) * parent_value_count),
                "cudaMemset exact-moduli pqt t next_level"
            );
            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_cross_pass(current_t, current_q, next_t, current_nodes, plan_handle, pass_index);
                execute_cross_pass(current_p, current_t, next_t, current_nodes, plan_handle, pass_index);
            }
        };

        const auto storage_offset = [&](std::size_t modulus_index, std::size_t merge_index) {
            return (modulus_index * sample_merge_count + merge_index) * report.verification_sample_count;
        };
        const auto level_offset = [&](std::size_t modulus_index) {
            return modulus_index * sample_merge_count * config.slot_count;
        };
        const auto copy_fft_samples = [&](std::vector<double>& storage) {
            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                    const std::size_t batch_index = modulus_index * first_level_merge_count + merge_index;
                    std::vector<cufftDoubleComplex> host(report.verification_sample_count);
                    check_cuda(
                        cudaMemcpy(
                            host.data(),
                            d_out + batch_index * config.slot_count,
                            sizeof(cufftDoubleComplex) * report.verification_sample_count,
                            cudaMemcpyDeviceToHost
                        ),
                        "cudaMemcpy exact-moduli pqt fft samples"
                    );
                    for (std::size_t sample_index = 0; sample_index < report.verification_sample_count; ++sample_index) {
                        storage[storage_offset(modulus_index, merge_index) + sample_index] = host[sample_index].x;
                    }
                }
            }
        };

        const auto verify_single_product_first_level = [&](
                                                           std::size_t stream_index,
                                                           auto expected_digit_fn
                                                       ) {
            check_cuda(
                cudaMemcpy(d_level_a[stream_index], d_original[stream_index], level_bytes, cudaMemcpyDeviceToDevice),
                "cudaMemcpy exact-moduli pqt verification reset"
            );
            check_cuda(
                cudaMemset(
                    d_level_b[stream_index],
                    0,
                    sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count) * first_level_merge_count *
                        config.slot_count
                ),
                "cudaMemset exact-moduli pqt verification next_level"
            );

            std::vector<std::uint32_t> level_samples(sample_modulus_count * sample_merge_count * config.slot_count, 0u);
            std::vector<std::vector<double>> pass_sample_reals(
                kPassCount,
                std::vector<double>(sample_storage_size, 0.0)
            );

            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_cross_pass(
                    d_level_a[stream_index],
                    d_level_a[stream_index],
                    d_level_b[stream_index],
                    config.node_count,
                    plans[0],
                    pass_index
                );
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize exact-moduli pqt verification pass");
                copy_fft_samples(pass_sample_reals[pass_index]);
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                check_cuda(
                    cudaMemcpy(
                        level_samples.data() + level_offset(modulus_index),
                        d_level_b[stream_index] + modulus_index * first_level_merge_count * config.slot_count,
                        sizeof(std::uint32_t) * sample_merge_count * config.slot_count,
                        cudaMemcpyDeviceToHost
                    ),
                    "cudaMemcpy exact-moduli pqt verification samples"
                );
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                const std::uint32_t modulus = kDefaultModuli[modulus_index];
                for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                    for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                        std::uint64_t expected = 0;
                        for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                            const auto& pass = kPasses[pass_index];
                            std::uint64_t expected_contribution = 0;
                            for (std::size_t inner_index = 0; inner_index < config.slot_count; ++inner_index) {
                                const std::size_t rhs_index =
                                    (output_index + config.slot_count - inner_index) % config.slot_count;
                                const auto lhs_value = expected_digit_fn(inner_index, merge_index * 2ull, modulus_index);
                                const auto rhs_value = expected_digit_fn(rhs_index, merge_index * 2ull + 1ull, modulus_index);
                                expected_contribution +=
                                    static_cast<std::uint64_t>((lhs_value >> pass.lhs_shift) & pass.lhs_mask) *
                                    static_cast<std::uint64_t>((rhs_value >> pass.rhs_shift) & pass.rhs_mask);
                            }
                            const std::size_t fft_index = storage_offset(modulus_index, merge_index) + output_index;
                            report.max_projection_real_error = std::max(
                                report.max_projection_real_error,
                                std::abs(pass_sample_reals[pass_index][fft_index] - static_cast<double>(expected_contribution))
                            );
                            const std::uint64_t pass_weight =
                                static_cast<std::uint64_t>(
                                    pass_weights_host[pass_index * static_cast<std::size_t>(modulus_count) + modulus_index]
                                );
                            expected =
                                (expected +
                                 ((expected_contribution % static_cast<std::uint64_t>(modulus)) * pass_weight) %
                                     static_cast<std::uint64_t>(modulus)) %
                                static_cast<std::uint64_t>(modulus);
                        }

                        const std::size_t host_index =
                            level_offset(modulus_index) + merge_index * config.slot_count + output_index;
                        const std::uint32_t observed = level_samples[host_index];
                        if (observed != static_cast<std::uint32_t>(expected)) {
                            report.verification_mismatch_count += 1;
                            if (report.verification_mismatch_count == 1) {
                                report.first_mismatch_modulus = modulus_index;
                                report.first_mismatch_merge = merge_index;
                                report.first_mismatch_output = stream_index;
                                report.first_mismatch_expected = static_cast<std::uint32_t>(expected);
                                report.first_mismatch_observed = observed;
                            }
                        }
                        report.verified_values += 1;
                    }
                }
            }
        };

        verify_single_product_first_level(
            kPStreamIndex,
            [&](std::size_t slot_index, std::size_t node_index, std::size_t modulus_index) {
                (void)modulus_index;
                return expected_chudnovsky_exact_pfactor_digit(slot_index, node_index);
            }
        );
        verify_single_product_first_level(
            kQStreamIndex,
            [&](std::size_t slot_index, std::size_t node_index, std::size_t modulus_index) {
                (void)modulus_index;
                return expected_chudnovsky_qfactor_digit(slot_index, node_index);
            }
        );

        {
            for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                check_cuda(
                    cudaMemcpy(d_level_a[stream_index], d_original[stream_index], level_bytes, cudaMemcpyDeviceToDevice),
                    "cudaMemcpy exact-moduli pqt t verification reset"
                );
            }
            check_cuda(
                cudaMemset(
                    d_level_b[kTStreamIndex],
                    0,
                    sizeof(std::uint32_t) * static_cast<std::size_t>(modulus_count) * first_level_merge_count *
                        config.slot_count
                ),
                "cudaMemset exact-moduli pqt t verification next_level"
            );

            std::vector<std::uint32_t> level_samples(sample_modulus_count * sample_merge_count * config.slot_count, 0u);
            std::vector<std::vector<double>> tq_pass_sample_reals(
                kPassCount,
                std::vector<double>(sample_storage_size, 0.0)
            );
            std::vector<std::vector<double>> pt_pass_sample_reals(
                kPassCount,
                std::vector<double>(sample_storage_size, 0.0)
            );

            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_cross_pass(
                    d_level_a[kTStreamIndex],
                    d_level_a[kQStreamIndex],
                    d_level_b[kTStreamIndex],
                    config.node_count,
                    plans[0],
                    pass_index
                );
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize exact-moduli pqt tq pass");
                copy_fft_samples(tq_pass_sample_reals[pass_index]);

                execute_cross_pass(
                    d_level_a[kPStreamIndex],
                    d_level_a[kTStreamIndex],
                    d_level_b[kTStreamIndex],
                    config.node_count,
                    plans[0],
                    pass_index
                );
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize exact-moduli pqt pt pass");
                copy_fft_samples(pt_pass_sample_reals[pass_index]);
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                check_cuda(
                    cudaMemcpy(
                        level_samples.data() + level_offset(modulus_index),
                        d_level_b[kTStreamIndex] + modulus_index * first_level_merge_count * config.slot_count,
                        sizeof(std::uint32_t) * sample_merge_count * config.slot_count,
                        cudaMemcpyDeviceToHost
                    ),
                    "cudaMemcpy exact-moduli pqt t verification samples"
                );
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                const std::uint32_t modulus = kDefaultModuli[modulus_index];
                for (std::size_t merge_index = 0; merge_index < sample_merge_count; ++merge_index) {
                    for (std::size_t output_index = 0; output_index < report.verification_sample_count; ++output_index) {
                        std::uint64_t expected = 0;
                        for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                            const auto& pass = kPasses[pass_index];
                            std::uint64_t tq_expected_contribution = 0;
                            std::uint64_t pt_expected_contribution = 0;
                            for (std::size_t inner_index = 0; inner_index < config.slot_count; ++inner_index) {
                                const std::size_t rhs_index =
                                    (output_index + config.slot_count - inner_index) % config.slot_count;
                                const auto t_lhs =
                                    expected_chudnovsky_tfactor_digit(inner_index, merge_index * 2ull, modulus_index);
                                const auto q_rhs =
                                    expected_chudnovsky_qfactor_digit(rhs_index, merge_index * 2ull + 1ull);
                                const auto p_lhs =
                                    expected_chudnovsky_exact_pfactor_digit(inner_index, merge_index * 2ull);
                                const auto t_rhs =
                                    expected_chudnovsky_tfactor_digit(rhs_index, merge_index * 2ull + 1ull, modulus_index);
                                tq_expected_contribution +=
                                    static_cast<std::uint64_t>((t_lhs >> pass.lhs_shift) & pass.lhs_mask) *
                                    static_cast<std::uint64_t>((q_rhs >> pass.rhs_shift) & pass.rhs_mask);
                                pt_expected_contribution +=
                                    static_cast<std::uint64_t>((p_lhs >> pass.lhs_shift) & pass.lhs_mask) *
                                    static_cast<std::uint64_t>((t_rhs >> pass.rhs_shift) & pass.rhs_mask);
                            }
                            const std::size_t fft_index = storage_offset(modulus_index, merge_index) + output_index;
                            report.max_projection_real_error = std::max(
                                report.max_projection_real_error,
                                std::abs(tq_pass_sample_reals[pass_index][fft_index] - static_cast<double>(tq_expected_contribution))
                            );
                            report.max_projection_real_error = std::max(
                                report.max_projection_real_error,
                                std::abs(pt_pass_sample_reals[pass_index][fft_index] - static_cast<double>(pt_expected_contribution))
                            );
                            const std::uint64_t pass_weight =
                                static_cast<std::uint64_t>(
                                    pass_weights_host[pass_index * static_cast<std::size_t>(modulus_count) + modulus_index]
                                );
                            expected =
                                (expected +
                                 ((tq_expected_contribution % static_cast<std::uint64_t>(modulus)) * pass_weight) %
                                     static_cast<std::uint64_t>(modulus)) %
                                static_cast<std::uint64_t>(modulus);
                            expected =
                                (expected +
                                 ((pt_expected_contribution % static_cast<std::uint64_t>(modulus)) * pass_weight) %
                                     static_cast<std::uint64_t>(modulus)) %
                                static_cast<std::uint64_t>(modulus);
                        }

                        const std::size_t host_index =
                            level_offset(modulus_index) + merge_index * config.slot_count + output_index;
                        const std::uint32_t observed = level_samples[host_index];
                        if (observed != static_cast<std::uint32_t>(expected)) {
                            report.verification_mismatch_count += 1;
                            if (report.verification_mismatch_count == 1) {
                                report.first_mismatch_modulus = modulus_index;
                                report.first_mismatch_merge = merge_index;
                                report.first_mismatch_output = kTStreamIndex;
                                report.first_mismatch_expected = static_cast<std::uint32_t>(expected);
                                report.first_mismatch_observed = observed;
                            }
                        }
                        report.verified_values += 1;
                    }
                }
            }
        }

        if (can_verify_root_without_wrap[kPStreamIndex] || can_verify_root_without_wrap[kQStreamIndex] ||
            can_verify_root_without_wrap[kTStreamIndex]) {
            for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                check_cuda(
                    cudaMemcpy(d_level_a[stream_index], d_original[stream_index], level_bytes, cudaMemcpyDeviceToDevice),
                    "cudaMemcpy exact-moduli pqt root reset"
                );
            }

            std::array<std::uint32_t*, kStreamCount> current_levels = d_level_a;
            std::array<std::uint32_t*, kStreamCount> next_levels = d_level_b;
            std::size_t current_nodes = config.node_count;
            for (int level_index = 0; level_index < level_count; ++level_index) {
                execute_product_level(
                    current_levels[kPStreamIndex],
                    next_levels[kPStreamIndex],
                    current_nodes,
                    plans[static_cast<std::size_t>(level_index)]
                );
                execute_product_level(
                    current_levels[kQStreamIndex],
                    next_levels[kQStreamIndex],
                    current_nodes,
                    plans[static_cast<std::size_t>(level_index)]
                );
                execute_t_level(
                    current_levels[kTStreamIndex],
                    current_levels[kPStreamIndex],
                    current_levels[kQStreamIndex],
                    next_levels[kTStreamIndex],
                    current_nodes,
                    plans[static_cast<std::size_t>(level_index)]
                );
                std::swap(current_levels, next_levels);
                current_nodes /= 2ull;
            }

            for (std::size_t modulus_index = 0; modulus_index < sample_modulus_count; ++modulus_index) {
                const std::uint32_t modulus = kDefaultModuli[modulus_index];
                const auto expected_root = expected_chudnovsky_exact_pqt_root_mod(config.node_count, modulus);
                const std::array<std::uint32_t, kStreamCount> expected_values = {
                    expected_root.p,
                    expected_root.q,
                    expected_root.t,
                };
                for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                    if (!can_verify_root_without_wrap[stream_index]) {
                        continue;
                    }
                    std::vector<std::uint32_t> root_samples(config.slot_count, 0u);
                    check_cuda(
                        cudaMemcpy(
                            root_samples.data(),
                            current_levels[stream_index] + modulus_index * config.slot_count,
                            sizeof(std::uint32_t) * config.slot_count,
                            cudaMemcpyDeviceToHost
                        ),
                        "cudaMemcpy exact-moduli pqt root samples"
                    );
                    const std::uint32_t observed_root = evaluate_coefficient_vector_mod(
                        root_samples.data(),
                        config.slot_count,
                        modulus
                    );
                    if (observed_root != expected_values[stream_index]) {
                        report.verification_mismatch_count += 1;
                        if (report.verification_mismatch_count == 1) {
                            report.first_mismatch_modulus = modulus_index;
                            report.first_mismatch_merge = 0;
                            report.first_mismatch_output = stream_index;
                            report.first_mismatch_expected = expected_values[stream_index];
                            report.first_mismatch_observed = observed_root;
                        }
                    }
                    report.verified_values += 1;
                }
            }
        }

        const std::size_t root_value_count = static_cast<std::size_t>(modulus_count) * config.slot_count;

        const auto run_pipeline_once = [&](double& elapsed_ms, bool capture_root_values) {
            for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                check_cuda(
                    cudaMemcpy(
                        d_level_a[stream_index],
                        d_original[stream_index],
                        level_bytes,
                        cudaMemcpyDeviceToDevice
                    ),
                    "cudaMemcpy exact-moduli pqt reset"
                );
            }
            check_cuda(cudaEventRecord(start), "cudaEventRecord exact-moduli pqt pipeline start");

            std::array<std::uint32_t*, kStreamCount> current_levels = d_level_a;
            std::array<std::uint32_t*, kStreamCount> next_levels = d_level_b;
            std::size_t current_nodes = config.node_count;
            for (int level_index = 0; level_index < level_count; ++level_index) {
                execute_product_level(
                    current_levels[kPStreamIndex],
                    next_levels[kPStreamIndex],
                    current_nodes,
                    plans[static_cast<std::size_t>(level_index)]
                );
                execute_product_level(
                    current_levels[kQStreamIndex],
                    next_levels[kQStreamIndex],
                    current_nodes,
                    plans[static_cast<std::size_t>(level_index)]
                );
                execute_t_level(
                    current_levels[kTStreamIndex],
                    current_levels[kPStreamIndex],
                    current_levels[kQStreamIndex],
                    next_levels[kTStreamIndex],
                    current_nodes,
                    plans[static_cast<std::size_t>(level_index)]
                );
                std::swap(current_levels, next_levels);
                current_nodes /= 2ull;
            }

            check_cuda(cudaEventRecord(stop), "cudaEventRecord exact-moduli pqt pipeline stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize exact-moduli pqt pipeline stop");
            float elapsed_ms_f = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms_f, start, stop), "cudaEventElapsedTime exact-moduli pqt pipeline");
            elapsed_ms = static_cast<double>(elapsed_ms_f);

            if (capture_root_values && captured_roots != nullptr) {
                for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                    captured_roots->roots[stream_index].resize(root_value_count, 0u);
                    check_cuda(
                        cudaMemcpy(
                            captured_roots->roots[stream_index].data(),
                            current_levels[stream_index],
                            sizeof(std::uint32_t) * root_value_count,
                            cudaMemcpyDeviceToHost
                        ),
                        "cudaMemcpy exact-moduli pqt captured root values"
                    );
                }
            }
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pipeline_once(ignored_ms, false);
        }

        run_pipeline_once(report.cold_pipeline_ms, false);

        double total_pipeline_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            const bool capture_this_run =
                captured_roots != nullptr && iteration + 1 == config.measured_iterations;
            run_pipeline_once(elapsed_ms, capture_this_run);
            total_pipeline_ms += elapsed_ms;
        }
        report.avg_pipeline_ms = total_pipeline_ms / static_cast<double>(config.measured_iterations);

        if (report.avg_pipeline_ms > 0.0) {
            report.packed_residue_values_per_second =
                static_cast<double>(report.total_residue_values_packed) * 1000.0 / report.avg_pipeline_ms;
            report.transformed_complex_values_per_second =
                static_cast<double>(report.total_complex_value_count * 3ull) * 1000.0 / report.avg_pipeline_ms;
            report.logical_pipeline_bytes_per_second =
                static_cast<double>(logical_pipeline_bytes) * 1000.0 / report.avg_pipeline_ms;
        }

        report.ok = report.verification_mismatch_count == 0;
        cleanup();
        return report;
    } catch (...) {
        cleanup();
        throw;
    }
}

GroupedLevelPlannerReport run_grouped_level_planner_exact_moduli_pqt_smoke(const GroupedLevelPlannerConfig& config) {
    return run_grouped_level_planner_exact_moduli_pqt_smoke_impl(config, nullptr);
}

CapturedExactModuliPqtRoots capture_exact_moduli_pqt_roots(const GroupedLevelPlannerConfig& config) {
    GroupedLevelPlannerConfig capture_config = config;
    capture_config.warmup_iterations = 0;
    capture_config.measured_iterations = 1;

    CapturedExactModuliPqtRoots captured;
    (void)run_grouped_level_planner_exact_moduli_pqt_smoke_impl(capture_config, &captured);
    return captured;
}

struct LevelwiseBalancedClosureResult {
    int effective_modulus_count = 0;
    std::size_t required_half_range_bits = 0;
    std::size_t crt_product_bits = 0;
    std::size_t closure_modulus_headroom_bits = 0;
    double cuda_runtime_init_ms = 0.0;
    double setup_ms = 0.0;
    double warmup_total_ms = 0.0;
    double plan_build_ms = 0.0;
    double cold_pipeline_ms = 0.0;
    double avg_pipeline_ms = 0.0;
    double measured_total_ms = 0.0;
    double packed_residue_values_per_second = 0.0;
    double closure_wall_ms = 0.0;
    double root_rebuild_ms = 0.0;
    std::string closure_mode;
    HostBigInt root_p;
    HostBigInt root_q;
    HostBigInt root_t;
    bool ok = false;
};

LevelwiseBalancedClosureResult run_levelwise_balanced_exact_moduli_pqt_closure(
    const PiEndToEndSmokeConfig& config
) {
    if (config.term_count < 2 || !is_power_of_two(config.term_count)) {
        throw std::invalid_argument("levelwise balanced exact-moduli pqt closure requires power-of-two term_count >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("levelwise balanced exact-moduli pqt closure requires power-of-two slot_count > 0");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and measured_iterations must be > 0");
    }

    constexpr std::uint32_t kLimbMask = 0xffffu;
    constexpr std::size_t kPassCount = 4;
    constexpr std::size_t kStreamCount = 3;
    constexpr std::size_t kEquivalentConvolutionStreamCount = 4;
    constexpr std::size_t kPStreamIndex = 0;
    constexpr std::size_t kQStreamIndex = 1;
    constexpr std::size_t kTStreamIndex = 2;
    constexpr std::array<SplitPassDescriptor, kPassCount> kPasses = {
        SplitPassDescriptor{.lhs_shift = 0, .lhs_mask = kLimbMask, .rhs_shift = 0, .rhs_mask = kLimbMask, .accumulation_shift = 0},
        SplitPassDescriptor{.lhs_shift = 0, .lhs_mask = kLimbMask, .rhs_shift = 16, .rhs_mask = kLimbMask, .accumulation_shift = 16},
        SplitPassDescriptor{.lhs_shift = 16, .lhs_mask = kLimbMask, .rhs_shift = 0, .rhs_mask = kLimbMask, .accumulation_shift = 16},
        SplitPassDescriptor{.lhs_shift = 16, .lhs_mask = kLimbMask, .rhs_shift = 16, .rhs_mask = kLimbMask, .accumulation_shift = 32},
    };

    const std::size_t required_half_range_bits =
        balanced_level_coefficient_bound_bits(config.slot_count, static_cast<std::size_t>(kCoefficientBaseBits));
    const std::vector<std::uint32_t> selected_moduli = select_effective_balanced_closure_moduli(
        config.modulus_count,
        config.slot_count,
        config.forced_closure_modulus_count
    );
    const int effective_modulus_count = static_cast<int>(selected_moduli.size());
    const std::size_t level_value_count_max =
        static_cast<std::size_t>(effective_modulus_count) * config.term_count * config.slot_count;
    const std::size_t level_bytes_max = sizeof(std::uint32_t) * level_value_count_max;
    const std::size_t max_fft_batch_count =
        static_cast<std::size_t>(effective_modulus_count) * (config.term_count / 2ull);
    if (max_fft_batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("levelwise balanced exact-moduli pqt closure fft batch_count exceeds cuFFT int range");
    }
    if (config.slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("levelwise balanced exact-moduli pqt closure slot_count exceeds cuFFT int range");
    }
    const std::size_t max_complex_value_count = max_fft_batch_count * config.slot_count;
    const std::size_t complex_bytes = sizeof(cufftDoubleComplex) * max_complex_value_count;

    std::size_t total_residue_values_packed = 0;
    int level_count = 0;
    for (std::size_t current_nodes = config.term_count; current_nodes > 1; current_nodes /= 2ull) {
        const std::size_t batch_count =
            static_cast<std::size_t>(effective_modulus_count) * (current_nodes / 2ull);
        total_residue_values_packed += batch_count * config.slot_count * 2ull * kEquivalentConvolutionStreamCount;
        level_count += 1;
    }

    std::array<std::uint32_t*, kStreamCount> d_current = {nullptr, nullptr, nullptr};
    std::array<std::uint32_t*, kStreamCount> d_next = {nullptr, nullptr, nullptr};
    std::array<std::int32_t*, kStreamCount> d_leaf_digits = {nullptr, nullptr, nullptr};
    std::array<std::int32_t*, kStreamCount> d_digits_current = {nullptr, nullptr, nullptr};
    std::array<std::int32_t*, kStreamCount> d_digits_next = {nullptr, nullptr, nullptr};
    std::uint32_t* d_moduli = nullptr;
    std::uint32_t* d_pass_weights = nullptr;
    cufftDoubleComplex* d_work_a = nullptr;
    cufftDoubleComplex* d_work_b = nullptr;
    cufftDoubleComplex* d_out = nullptr;
    int* d_overflow_flag = nullptr;
    std::vector<cufftHandle> plans(static_cast<std::size_t>(level_count), 0);

    const auto cleanup = [&]() {
        for (cufftHandle& plan : plans) {
            if (plan != 0) {
                cufftDestroy(plan);
                plan = 0;
            }
        }
        cudaFree(d_out);
        cudaFree(d_work_b);
        cudaFree(d_work_a);
        cudaFree(d_pass_weights);
        cudaFree(d_moduli);
        cudaFree(d_overflow_flag);
        for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
            cudaFree(d_digits_next[stream_index]);
            cudaFree(d_digits_current[stream_index]);
            cudaFree(d_leaf_digits[stream_index]);
            cudaFree(d_next[stream_index]);
            cudaFree(d_current[stream_index]);
        }
    };

    LevelwiseBalancedClosureResult result;
    result.effective_modulus_count = effective_modulus_count;
    result.required_half_range_bits = required_half_range_bits;
    result.crt_product_bits = crt_product_bit_width(selected_moduli);
    result.closure_modulus_headroom_bits =
        result.crt_product_bits > 0 && result.crt_product_bits - 1u > result.required_half_range_bits
            ? (result.crt_product_bits - 1u) - result.required_half_range_bits
            : 0u;
    if (effective_modulus_count == 2) {
        result.closure_mode =
            "gpu_levelwise_exact_moduli_with_device_two_moduli_balanced_normalization_and_final_host_isqrt_division";
    } else if (effective_modulus_count >= 1 && effective_modulus_count <= 4) {
        result.closure_mode =
            "gpu_levelwise_exact_moduli_with_device_small_moduli_balanced_normalization_and_final_host_isqrt_division";
    } else {
        result.closure_mode =
            "gpu_levelwise_exact_moduli_with_host_balanced_normalization_and_final_host_isqrt_division";
    }

    const bool use_two_moduli_device_normalization = effective_modulus_count == 2;
    const bool use_small_moduli_device_normalization =
        effective_modulus_count >= 1 && effective_modulus_count <= 4;
    const std::uint32_t inverse_m0_mod_m1 =
        use_two_moduli_device_normalization
            ? mod_inverse_u32(selected_moduli[0] % selected_moduli[1], selected_moduli[1])
            : 0u;
    const std::uint64_t two_moduli_product =
        use_two_moduli_device_normalization
            ? static_cast<std::uint64_t>(selected_moduli[0]) * static_cast<std::uint64_t>(selected_moduli[1])
            : 0ull;

    try {
        const auto runtime_init_begin = std::chrono::steady_clock::now();
        check_cuda(cudaFree(nullptr), "cuda runtime initialization prewarm");
        const auto runtime_init_end = std::chrono::steady_clock::now();
        result.cuda_runtime_init_ms =
            std::chrono::duration<double, std::milli>(runtime_init_end - runtime_init_begin).count();
        const auto closure_wall_begin = std::chrono::steady_clock::now();
        const auto setup_begin = closure_wall_begin;
        for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_current[stream_index]), level_bytes_max),
                "cudaMalloc levelwise balanced pqt d_current"
            );
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&d_next[stream_index]), level_bytes_max),
                "cudaMalloc levelwise balanced pqt d_next"
            );
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&d_leaf_digits[stream_index]),
                    sizeof(std::int32_t) * config.term_count * config.slot_count
                ),
                "cudaMalloc levelwise balanced pqt d_leaf_digits"
            );
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&d_digits_current[stream_index]),
                    sizeof(std::int32_t) * config.term_count * config.slot_count
                ),
                "cudaMalloc levelwise balanced pqt d_digits_current"
            );
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&d_digits_next[stream_index]),
                    sizeof(std::int32_t) * config.term_count * config.slot_count
                ),
                "cudaMalloc levelwise balanced pqt d_digits_next"
            );
        }
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_moduli),
                sizeof(std::uint32_t) * static_cast<std::size_t>(effective_modulus_count)
            ),
            "cudaMalloc levelwise balanced pqt d_moduli"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_pass_weights),
                sizeof(std::uint32_t) * static_cast<std::size_t>(effective_modulus_count) * kPassCount
            ),
            "cudaMalloc levelwise balanced pqt d_pass_weights"
        );
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_a), complex_bytes), "cudaMalloc levelwise balanced pqt d_work_a");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_b), complex_bytes), "cudaMalloc levelwise balanced pqt d_work_b");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), complex_bytes), "cudaMalloc levelwise balanced pqt d_out");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_overflow_flag), sizeof(int)), "cudaMalloc levelwise balanced pqt d_overflow_flag");
        check_cuda(
            cudaMemcpy(
                d_moduli,
                selected_moduli.data(),
                sizeof(std::uint32_t) * selected_moduli.size(),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy levelwise balanced pqt d_moduli"
        );

        std::vector<std::uint32_t> pass_weights_host(
            static_cast<std::size_t>(effective_modulus_count) * kPassCount,
            0u
        );
        for (int modulus_index = 0; modulus_index < effective_modulus_count; ++modulus_index) {
            const std::uint32_t modulus = selected_moduli[static_cast<std::size_t>(modulus_index)];
            const std::uint32_t base_mod = static_cast<std::uint32_t>((1ull << 16) % modulus);
            const std::uint32_t base_sq_mod =
                static_cast<std::uint32_t>((static_cast<std::uint64_t>(base_mod) * base_mod) % modulus);
            pass_weights_host[0 * static_cast<std::size_t>(effective_modulus_count) + static_cast<std::size_t>(modulus_index)] =
                1u % modulus;
            pass_weights_host[1 * static_cast<std::size_t>(effective_modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_mod;
            pass_weights_host[2 * static_cast<std::size_t>(effective_modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_mod;
            pass_weights_host[3 * static_cast<std::size_t>(effective_modulus_count) + static_cast<std::size_t>(modulus_index)] =
                base_sq_mod;
        }
        check_cuda(
            cudaMemcpy(
                d_pass_weights,
                pass_weights_host.data(),
                sizeof(std::uint32_t) * pass_weights_host.size(),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy levelwise balanced pqt d_pass_weights"
        );

        const auto plan_begin = std::chrono::steady_clock::now();
        std::size_t current_nodes_for_plan = config.term_count;
        for (int level_index = 0; level_index < level_count; ++level_index) {
            const std::size_t batch_count =
                static_cast<std::size_t>(effective_modulus_count) * (current_nodes_for_plan / 2ull);
            check_cufft(
                cufftPlan1d(
                    &plans[static_cast<std::size_t>(level_index)],
                    static_cast<int>(config.slot_count),
                    CUFFT_Z2Z,
                    static_cast<int>(batch_count)
                ),
                "cufftPlan1d levelwise balanced pqt planner"
            );
            current_nodes_for_plan /= 2ull;
        }
        const auto plan_end = std::chrono::steady_clock::now();
        result.plan_build_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();

        const auto execute_cross_pass = [&](
                                            const std::uint32_t* lhs_level,
                                            const std::uint32_t* rhs_level,
                                            std::uint32_t* next_level,
                                            std::size_t current_nodes,
                                            cufftHandle plan_handle,
                                            std::size_t pass_index
                                        ) {
            const auto& pass = kPasses[pass_index];
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t complex_value_count_level =
                static_cast<std::size_t>(effective_modulus_count) * merge_count * config.slot_count;

            pack_grouped_modulus_major_cross_pairs_to_fft_limb_inputs_kernel_fp64<<<
                block_count(complex_value_count_level),
                kThreadsPerBlock>>>(
                lhs_level,
                rhs_level,
                d_work_a,
                d_work_b,
                current_nodes,
                config.slot_count,
                effective_modulus_count,
                pass.lhs_shift,
                pass.lhs_mask,
                pass.rhs_shift,
                pass.rhs_mask
            );
            check_cuda(cudaGetLastError(), "pack_grouped_modulus_major_cross_pairs levelwise balanced pqt launch");
            check_cufft(cufftExecZ2Z(plan_handle, d_work_a, d_work_a, CUFFT_FORWARD), "cufftExecZ2Z levelwise balanced pqt forward a");
            check_cufft(cufftExecZ2Z(plan_handle, d_work_b, d_work_b, CUFFT_FORWARD), "cufftExecZ2Z levelwise balanced pqt forward b");
            complex_pointwise_multiply_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_work_a,
                d_work_b,
                d_out,
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "complex_pointwise_multiply_kernel_fp64 levelwise balanced pqt launch");
            check_cufft(cufftExecZ2Z(plan_handle, d_out, d_out, CUFFT_INVERSE), "cufftExecZ2Z levelwise balanced pqt inverse");
            scale_complex_kernel_fp64<<<block_count(complex_value_count_level), kThreadsPerBlock>>>(
                d_out,
                1.0 / static_cast<double>(config.slot_count),
                complex_value_count_level
            );
            check_cuda(cudaGetLastError(), "scale_complex_kernel_fp64 levelwise balanced pqt launch");
            accumulate_grouped_fft_output_to_level_values_mod_kernel_fp64<<<
                block_count(complex_value_count_level),
                kThreadsPerBlock>>>(
                d_out,
                next_level,
                d_moduli,
                d_pass_weights + pass_index * static_cast<std::size_t>(effective_modulus_count),
                merge_count,
                config.slot_count,
                effective_modulus_count
            );
            check_cuda(cudaGetLastError(), "accumulate_grouped_fft_output_to_level_values_mod levelwise balanced pqt launch");
        };

        const auto execute_product_level = [&](
                                               const std::uint32_t* current_level,
                                               std::uint32_t* next_level,
                                               std::size_t current_nodes,
                                               cufftHandle plan_handle
                                           ) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t parent_value_count =
                static_cast<std::size_t>(effective_modulus_count) * merge_count * config.slot_count;
            check_cuda(
                cudaMemset(next_level, 0, sizeof(std::uint32_t) * parent_value_count),
                "cudaMemset levelwise balanced pqt product next_level"
            );
            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_cross_pass(current_level, current_level, next_level, current_nodes, plan_handle, pass_index);
            }
        };

        const auto execute_t_level = [&](
                                         const std::uint32_t* current_t,
                                         const std::uint32_t* current_p,
                                         const std::uint32_t* current_q,
                                         std::uint32_t* next_t,
                                         std::size_t current_nodes,
                                         cufftHandle plan_handle
                                     ) {
            const std::size_t merge_count = current_nodes / 2ull;
            const std::size_t parent_value_count =
                static_cast<std::size_t>(effective_modulus_count) * merge_count * config.slot_count;
            check_cuda(
                cudaMemset(next_t, 0, sizeof(std::uint32_t) * parent_value_count),
                "cudaMemset levelwise balanced pqt t next_level"
            );
            for (std::size_t pass_index = 0; pass_index < kPassCount; ++pass_index) {
                execute_cross_pass(current_t, current_q, next_t, current_nodes, plan_handle, pass_index);
                execute_cross_pass(current_p, current_t, next_t, current_nodes, plan_handle, pass_index);
            }
        };

        std::array<BalancedDigitNodes, 3> leaf_digits =
            make_chudnovsky_leaf_balanced_digits(config.term_count, config.slot_count);
        std::array<std::vector<std::int32_t>, kStreamCount> host_leaf_digits_flat;
        for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
            host_leaf_digits_flat[stream_index].assign(config.term_count * config.slot_count, 0);
            for (std::size_t node_index = 0; node_index < config.term_count; ++node_index) {
                std::copy(
                    leaf_digits[stream_index][node_index].begin(),
                    leaf_digits[stream_index][node_index].end(),
                    host_leaf_digits_flat[stream_index].begin() + node_index * config.slot_count
                );
            }
            check_cuda(
                cudaMemcpy(
                    d_leaf_digits[stream_index],
                    host_leaf_digits_flat[stream_index].data(),
                    sizeof(std::int32_t) * host_leaf_digits_flat[stream_index].size(),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy levelwise balanced pqt leaf digits"
            );
        }
        std::array<std::vector<std::uint32_t>, kStreamCount> host_current_residues;
        std::array<std::vector<std::uint32_t>, kStreamCount> host_next_residues;
        std::array<std::vector<std::int32_t>, kStreamCount> captured_root_digits;
        const auto setup_end = std::chrono::steady_clock::now();
        result.setup_ms = std::chrono::duration<double, std::milli>(setup_end - setup_begin).count();

        const auto run_pipeline_once = [&](double& elapsed_ms, bool capture_roots_now) {
            const auto pipeline_begin = std::chrono::steady_clock::now();

            if (use_small_moduli_device_normalization) {
                std::array<std::int32_t*, kStreamCount> current_digit_levels = d_leaf_digits;
                std::array<std::int32_t*, kStreamCount> next_digit_levels = d_digits_current;
                std::array<std::int32_t*, kStreamCount> alternate_digit_levels = d_digits_next;
                std::size_t current_nodes = config.term_count;
                for (int level_index = 0; level_index < level_count; ++level_index) {
                    const std::size_t current_digit_count = current_nodes * config.slot_count;
                    for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                        pack_balanced_digit_nodes_to_residues_kernel<<<
                            block_count(static_cast<std::size_t>(effective_modulus_count) * current_digit_count),
                            kThreadsPerBlock>>>(
                            current_digit_levels[stream_index],
                            d_moduli,
                            d_current[stream_index],
                            current_nodes,
                            config.slot_count,
                            effective_modulus_count
                        );
                        check_cuda(cudaGetLastError(), "pack_balanced_digit_nodes_to_residues_kernel launch");
                    }

                    execute_product_level(
                        d_current[kPStreamIndex],
                        d_next[kPStreamIndex],
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );
                    execute_product_level(
                        d_current[kQStreamIndex],
                        d_next[kQStreamIndex],
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );
                    execute_t_level(
                        d_current[kTStreamIndex],
                        d_current[kPStreamIndex],
                        d_current[kQStreamIndex],
                        d_next[kTStreamIndex],
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );

                    const std::size_t merge_count = current_nodes / 2ull;
                    check_cuda(cudaMemset(d_overflow_flag, 0, sizeof(int)), "cudaMemset levelwise balanced pqt overflow flag");
                    for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                        if (use_two_moduli_device_normalization) {
                            normalize_two_moduli_residues_to_balanced_digits_kernel<<<
                                static_cast<unsigned int>(merge_count),
                                1>>>(
                                d_next[stream_index],
                                d_moduli,
                                next_digit_levels[stream_index],
                                merge_count,
                                config.slot_count,
                                inverse_m0_mod_m1,
                                two_moduli_product,
                                d_overflow_flag
                            );
                            check_cuda(cudaGetLastError(), "normalize_two_moduli_residues_to_balanced_digits_kernel launch");
                        } else {
                            normalize_small_moduli_residues_to_balanced_digits_kernel<<<
                                static_cast<unsigned int>(merge_count),
                                1>>>(
                                d_next[stream_index],
                                d_moduli,
                                next_digit_levels[stream_index],
                                merge_count,
                                config.slot_count,
                                effective_modulus_count,
                                d_overflow_flag
                            );
                            check_cuda(cudaGetLastError(), "normalize_small_moduli_residues_to_balanced_digits_kernel launch");
                        }
                    }
                    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after device balanced normalization");

                    int overflow_flag = 0;
                    check_cuda(
                        cudaMemcpy(&overflow_flag, d_overflow_flag, sizeof(int), cudaMemcpyDeviceToHost),
                        "cudaMemcpy levelwise balanced pqt overflow flag"
                    );
                    if (overflow_flag != 0) {
                        throw std::runtime_error("device balanced normalization exceeded digit_count");
                    }

                    if (level_index == 0) {
                        current_digit_levels = next_digit_levels;
                        next_digit_levels = alternate_digit_levels;
                    } else {
                        std::swap(current_digit_levels, next_digit_levels);
                    }
                    current_nodes = merge_count;
                }

                if (capture_roots_now) {
                    for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                        captured_root_digits[stream_index].assign(config.slot_count, 0);
                        check_cuda(
                            cudaMemcpy(
                                captured_root_digits[stream_index].data(),
                                current_digit_levels[stream_index],
                                sizeof(std::int32_t) * config.slot_count,
                                cudaMemcpyDeviceToHost
                            ),
                            "cudaMemcpy levelwise balanced pqt captured root digits"
                        );
                    }
                }
            } else {
                std::array<BalancedDigitNodes, kStreamCount> current_digits = leaf_digits;
                std::array<BalancedDigitNodes, kStreamCount> next_digits;
                std::size_t current_nodes = config.term_count;
                for (int level_index = 0; level_index < level_count; ++level_index) {
                    const std::size_t current_value_count =
                        static_cast<std::size_t>(effective_modulus_count) * current_nodes * config.slot_count;
                    const std::size_t current_level_bytes = sizeof(std::uint32_t) * current_value_count;
                    for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                        pack_balanced_digit_nodes_to_residues(
                            current_digits[stream_index],
                            selected_moduli,
                            config.slot_count,
                            host_current_residues[stream_index]
                        );
                        check_cuda(
                            cudaMemcpy(
                                d_current[stream_index],
                                host_current_residues[stream_index].data(),
                                current_level_bytes,
                                cudaMemcpyHostToDevice
                            ),
                            "cudaMemcpy levelwise balanced pqt current level"
                        );
                    }

                    execute_product_level(
                        d_current[kPStreamIndex],
                        d_next[kPStreamIndex],
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );
                    execute_product_level(
                        d_current[kQStreamIndex],
                        d_next[kQStreamIndex],
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );
                    execute_t_level(
                        d_current[kTStreamIndex],
                        d_current[kPStreamIndex],
                        d_current[kQStreamIndex],
                        d_next[kTStreamIndex],
                        current_nodes,
                        plans[static_cast<std::size_t>(level_index)]
                    );
                    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after levelwise balanced pqt level");

                    const std::size_t merge_count = current_nodes / 2ull;
                    const std::size_t next_value_count =
                        static_cast<std::size_t>(effective_modulus_count) * merge_count * config.slot_count;
                    const std::size_t next_level_bytes = sizeof(std::uint32_t) * next_value_count;
                    for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                        host_next_residues[stream_index].resize(next_value_count, 0u);
                        check_cuda(
                            cudaMemcpy(
                                host_next_residues[stream_index].data(),
                                d_next[stream_index],
                                next_level_bytes,
                                cudaMemcpyDeviceToHost
                            ),
                            "cudaMemcpy levelwise balanced pqt next level"
                        );
                    }

                    for (std::size_t stream_index = 0; stream_index < kStreamCount; ++stream_index) {
                        next_digits[stream_index].assign(
                            merge_count,
                            std::vector<std::int32_t>(config.slot_count, 0)
                        );
                        std::vector<__int128> coefficients(config.slot_count, 0);
                        std::vector<std::uint32_t> residue_scratch(selected_moduli.size(), 0u);
                        for (std::size_t parent_index = 0; parent_index < merge_count; ++parent_index) {
                            for (std::size_t slot_index = 0; slot_index < config.slot_count; ++slot_index) {
                                for (std::size_t modulus_index = 0; modulus_index < selected_moduli.size(); ++modulus_index) {
                                    residue_scratch[modulus_index] =
                                        host_next_residues[stream_index]
                                            [modulus_index * merge_count * config.slot_count +
                                             parent_index * config.slot_count + slot_index];
                                }
                                coefficients[slot_index] =
                                    centered_crt_i128_from_residues(residue_scratch.data(), selected_moduli);
                            }

                            next_digits[stream_index][parent_index] = normalize_centered_coefficients_i128_to_digits(
                                coefficients,
                                static_cast<std::size_t>(kCoefficientBaseBits)
                            );
                            if (capture_roots_now && merge_count == 1u) {
                                captured_root_digits[stream_index] = next_digits[stream_index][parent_index];
                            }
                        }
                    }

                    current_digits = std::move(next_digits);
                    current_nodes /= 2ull;
                }
            }

            const auto pipeline_end = std::chrono::steady_clock::now();
            elapsed_ms = std::chrono::duration<double, std::milli>(pipeline_end - pipeline_begin).count();
        };

        for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            double ignored_ms = 0.0;
            run_pipeline_once(ignored_ms, false);
            result.warmup_total_ms += ignored_ms;
        }

        run_pipeline_once(result.cold_pipeline_ms, false);

        double total_pipeline_ms = 0.0;
        for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
            double elapsed_ms = 0.0;
            const bool capture_roots_now = iteration + 1 == config.measured_iterations;
            run_pipeline_once(elapsed_ms, capture_roots_now);
            total_pipeline_ms += elapsed_ms;
        }
        result.measured_total_ms = total_pipeline_ms;
        result.avg_pipeline_ms = total_pipeline_ms / static_cast<double>(config.measured_iterations);
        if (result.avg_pipeline_ms > 0.0) {
            result.packed_residue_values_per_second =
                static_cast<double>(total_residue_values_packed) * 1000.0 / result.avg_pipeline_ms;
        }

        const auto root_rebuild_begin = std::chrono::steady_clock::now();
        result.root_p = rebuild_bigint_from_balanced_digits_i32(
            captured_root_digits[kPStreamIndex],
            static_cast<std::size_t>(kCoefficientBaseBits)
        );
        result.root_q = rebuild_bigint_from_balanced_digits_i32(
            captured_root_digits[kQStreamIndex],
            static_cast<std::size_t>(kCoefficientBaseBits)
        );
        result.root_t = rebuild_bigint_from_balanced_digits_i32(
            captured_root_digits[kTStreamIndex],
            static_cast<std::size_t>(kCoefficientBaseBits)
        );
        const auto root_rebuild_end = std::chrono::steady_clock::now();
        result.root_rebuild_ms = std::chrono::duration<double, std::milli>(root_rebuild_end - root_rebuild_begin).count();
        result.closure_wall_ms =
            std::chrono::duration<double, std::milli>(root_rebuild_end - closure_wall_begin).count();
        result.ok = true;
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

void print_grouped_level_planner_report(std::ostream& out, const GroupedLevelPlannerReport& report) {
    out << "grouped_level_planner_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "source_layout=" << report.source_layout << '\n';
    out << "fft_input_layout=" << report.fft_input_layout << '\n';
    out << "parent_layout=" << report.parent_layout << '\n';
    out << "operation=" << report.operation << '\n';
    out << "node_count=" << report.node_count << '\n';
    out << "final_node_count=" << report.final_node_count << '\n';
    out << "level_count=" << report.level_count << '\n';
    out << "slot_count=" << report.slot_count << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "packing_mask=" << report.packing_mask << '\n';
    out << "warmup_iterations=" << report.warmup_iterations << '\n';
    out << "measured_iterations=" << report.measured_iterations << '\n';
    out << "total_fft_batch_count=" << report.total_fft_batch_count << '\n';
    out << "total_residue_values_packed=" << report.total_residue_values_packed << '\n';
    out << "total_complex_value_count=" << report.total_complex_value_count << '\n';
    out << "plan_build_ms=" << report.plan_build_ms << '\n';
    out << "cold_pipeline_ms=" << report.cold_pipeline_ms << '\n';
    out << "avg_pipeline_ms=" << report.avg_pipeline_ms << '\n';
    out << "packed_residue_values_per_second=" << report.packed_residue_values_per_second << '\n';
    out << "transformed_complex_values_per_second=" << report.transformed_complex_values_per_second << '\n';
    out << "logical_pipeline_bytes_per_second=" << report.logical_pipeline_bytes_per_second << '\n';
    out << "verification_sample_count=" << report.verification_sample_count << '\n';
    out << "verified_values=" << report.verified_values << '\n';
    out << "split_limb_count=" << report.split_limb_count << '\n';
    out << "split_pass_count=" << report.split_pass_count << '\n';
    out << "verification_mismatch_count=" << report.verification_mismatch_count << '\n';
    out << "max_projection_real_error=" << report.max_projection_real_error << '\n';
    if (report.verification_mismatch_count > 0) {
        out << "first_mismatch_modulus=" << report.first_mismatch_modulus << '\n';
        out << "first_mismatch_merge=" << report.first_mismatch_merge << '\n';
        out << "first_mismatch_output=" << report.first_mismatch_output << '\n';
        out << "first_mismatch_expected=" << report.first_mismatch_expected << '\n';
        out << "first_mismatch_observed=" << report.first_mismatch_observed << '\n';
    }
}

PiEndToEndSmokeReport run_pi_end_to_end_smoke(const PiEndToEndSmokeConfig& config) {
    if (config.term_count < 2 || !is_power_of_two(config.term_count)) {
        throw std::invalid_argument("pi end-to-end smoke requires power-of-two term_count >= 2");
    }
    if (config.slot_count == 0 || !is_power_of_two(config.slot_count)) {
        throw std::invalid_argument("pi end-to-end smoke requires power-of-two slot_count > 0");
    }
    if (config.modulus_count < 1 || config.modulus_count > static_cast<int>(kDefaultModuli.size())) {
        throw std::invalid_argument(
            "pi end-to-end smoke requires modulus_count in [1, " + std::to_string(kDefaultModuli.size()) + "]"
        );
    }
    if (config.target_digits == 0) {
        throw std::invalid_argument("pi end-to-end smoke requires target_digits > 0");
    }

    const std::size_t max_exact_pfactor_leaf_degree =
        coefficient_digit_length_base16(chudnovsky_exact_pfactor_abs_value(config.term_count - 1ull)) - 1ull;
    const std::size_t max_qfactor_leaf_degree =
        chudnovsky_qfactor_digit_length_base16(config.term_count - 1ull) - 1ull;
    const std::size_t max_tfactor_leaf_degree =
        coefficient_digit_length_base16_u128(chudnovsky_tfactor_abs_value(config.term_count - 1ull)) - 1ull;
    const std::size_t max_combined_leaf_degree = std::max(
        max_tfactor_leaf_degree,
        std::max(max_exact_pfactor_leaf_degree, max_qfactor_leaf_degree)
    );
    if (1ull + config.term_count * max_combined_leaf_degree > config.slot_count) {
        throw std::invalid_argument("pi end-to-end smoke requires no-wrap support: increase slot_count or reduce term_count");
    }

    PiEndToEndSmokeReport report;
    const auto total_smoke_begin = std::chrono::steady_clock::now();
    report.term_count = config.term_count;
    report.target_digits = config.target_digits;
    report.working_digits = required_working_digits_for_pi(config.target_digits);
    report.required_terms = required_chudnovsky_terms_for_pi(config.target_digits);
    report.slot_count = config.slot_count;
    report.modulus_count = config.modulus_count;
    report.reference_prefix_digits_checked =
        std::min<std::size_t>(config.target_digits, pi_reference_digits().size() - 1u);
    report.term_count_sufficient = config.term_count >= report.required_terms;
    const LevelwiseBalancedClosureResult closure_result =
        run_levelwise_balanced_exact_moduli_pqt_closure(config);
    report.closure_mode = closure_result.closure_mode;
    report.planner_plan_build_ms = closure_result.plan_build_ms;
    report.planner_cold_pipeline_ms = closure_result.cold_pipeline_ms;
    report.planner_avg_pipeline_ms = closure_result.avg_pipeline_ms;
    report.planner_packed_residue_values_per_second = closure_result.packed_residue_values_per_second;
    report.cuda_runtime_init_ms = closure_result.cuda_runtime_init_ms;
    report.closure_setup_ms = closure_result.setup_ms;
    report.closure_warmup_total_ms = closure_result.warmup_total_ms;
    report.closure_measured_total_ms = closure_result.measured_total_ms;
    report.closure_wall_ms = closure_result.closure_wall_ms;
    report.effective_closure_modulus_count = closure_result.effective_modulus_count;
    report.required_closure_half_range_bits = closure_result.required_half_range_bits;
    report.crt_product_bits = closure_result.crt_product_bits;
    report.crt_half_range_bits = report.crt_product_bits > 0 ? report.crt_product_bits - 1u : 0u;
    report.closure_modulus_headroom_bits = closure_result.closure_modulus_headroom_bits;
    report.root_rebuild_ms = closure_result.root_rebuild_ms;
    if (!closure_result.ok) {
        report.ok = false;
        return report;
    }

    const HostBigInt& root_p = closure_result.root_p;
    const HostBigInt& root_q = closure_result.root_q;
    const HostBigInt& root_t = closure_result.root_t;
    report.root_p_bits = bit_width_cpp_int(root_p);
    report.root_q_bits = bit_width_cpp_int(root_q);
    report.root_t_bits = bit_width_cpp_int(root_t);

    const auto exact_root_reference_begin = std::chrono::steady_clock::now();
    const ExactPqtHostBigInt exact_root = build_chudnovsky_exact_root_host_bigint(config.term_count);
    report.root_reconstruction_match =
        root_p == exact_root.p && root_q == exact_root.q && root_t == exact_root.t;
    const auto exact_root_reference_end = std::chrono::steady_clock::now();
    report.exact_root_reference_ms =
        std::chrono::duration<double, std::milli>(exact_root_reference_end - exact_root_reference_begin).count();
    if (!report.root_reconstruction_match) {
        report.ok = false;
        return report;
    }
    if (root_t < 0 || root_t.is_zero()) {
        throw std::runtime_error("pi end-to-end smoke requires reconstructed root_t > 0");
    }

    const auto final_host_tail_begin = std::chrono::steady_clock::now();
    const HostBigInt working_scale = pow10_host_bigint(report.working_digits);
    const HostBigInt sqrt_scaled = integer_sqrt_host_bigint(HostBigInt{10005} * working_scale * working_scale);
    const HostBigInt pi_numerator = HostBigInt{426880} * sqrt_scaled * root_q;
    const auto [pi_scaled_working, pi_remainder] = div_mod_abs(pi_numerator, root_t);
    HostBigInt pi_scaled = pi_scaled_working;
    if (report.working_digits > config.target_digits) {
        pi_scaled = divide_by_power_of_ten_host_bigint(
            pi_scaled_working,
            report.working_digits - config.target_digits
        );
    }
    const std::string& reference = pi_reference_digits();
    std::string scaled_digits = host_bigint_to_decimal_string(pi_scaled);
    const std::size_t required_digits = config.target_digits + 1u;
    if (scaled_digits.size() < required_digits) {
        scaled_digits.insert(scaled_digits.begin(), required_digits - scaled_digits.size(), '0');
    }
    report.reported_decimal_digits = std::min(config.target_digits, config.report_decimal_digits);
    report.pi_decimal = format_scaled_decimal_prefix_from_digits(
        scaled_digits,
        config.target_digits,
        report.reported_decimal_digits,
        report.pi_decimal_truncated
    );
    const std::size_t checked_digit_count = report.reference_prefix_digits_checked + 1u;
    report.prefix_match =
        scaled_digits.size() >= checked_digit_count &&
        scaled_digits.compare(0, checked_digit_count, reference, 0, checked_digit_count) == 0 &&
        abs_compare(pi_remainder, root_t) < 0;
    const auto final_host_tail_end = std::chrono::steady_clock::now();
    report.final_host_tail_ms =
        std::chrono::duration<double, std::milli>(final_host_tail_end - final_host_tail_begin).count();
    report.ok = report.term_count_sufficient && report.root_reconstruction_match && report.prefix_match;
    const auto total_smoke_end = std::chrono::steady_clock::now();
    report.total_smoke_ms = std::chrono::duration<double, std::milli>(total_smoke_end - total_smoke_begin).count();
    const double measured_pipeline_ms_per_result =
        config.measured_iterations > 0
            ? report.closure_measured_total_ms / static_cast<double>(config.measured_iterations)
            : 0.0;
    report.steady_state_pi_result_ms =
        measured_pipeline_ms_per_result + report.root_rebuild_ms + report.final_host_tail_ms;
    if (report.steady_state_pi_result_ms > 0.0) {
        report.steady_state_pi_digits_per_second =
            static_cast<double>(report.target_digits) * 1000.0 / report.steady_state_pi_result_ms;
    }
    if (report.total_smoke_ms > 0.0) {
        report.cold_process_pi_digits_per_second =
            static_cast<double>(report.target_digits) * 1000.0 / report.total_smoke_ms;
    }
    return report;
}

void print_pi_end_to_end_report(std::ostream& out, const PiEndToEndSmokeReport& report) {
    out << "pi_end_to_end_smoke_status=" << (report.ok ? "ok" : "failed") << '\n';
    out << "term_count=" << report.term_count << '\n';
    out << "target_digits=" << report.target_digits << '\n';
    out << "working_digits=" << report.working_digits << '\n';
    out << "required_terms=" << report.required_terms << '\n';
    out << "slot_count=" << report.slot_count << '\n';
    out << "modulus_count=" << report.modulus_count << '\n';
    out << "effective_closure_modulus_count=" << report.effective_closure_modulus_count << '\n';
    out << "required_closure_half_range_bits=" << report.required_closure_half_range_bits << '\n';
    out << "crt_product_bits=" << report.crt_product_bits << '\n';
    out << "crt_half_range_bits=" << report.crt_half_range_bits << '\n';
    out << "closure_modulus_headroom_bits=" << report.closure_modulus_headroom_bits << '\n';
    out << "planner_plan_build_ms=" << report.planner_plan_build_ms << '\n';
    out << "planner_cold_pipeline_ms=" << report.planner_cold_pipeline_ms << '\n';
    out << "planner_avg_pipeline_ms=" << report.planner_avg_pipeline_ms << '\n';
    out << "planner_packed_residue_values_per_second=" << report.planner_packed_residue_values_per_second << '\n';
    out << "cuda_runtime_init_ms=" << report.cuda_runtime_init_ms << '\n';
    out << "closure_setup_ms=" << report.closure_setup_ms << '\n';
    out << "closure_warmup_total_ms=" << report.closure_warmup_total_ms << '\n';
    out << "closure_measured_total_ms=" << report.closure_measured_total_ms << '\n';
    out << "closure_wall_ms=" << report.closure_wall_ms << '\n';
    out << "root_rebuild_ms=" << report.root_rebuild_ms << '\n';
    out << "exact_root_reference_ms=" << report.exact_root_reference_ms << '\n';
    out << "final_host_tail_ms=" << report.final_host_tail_ms << '\n';
    out << "total_smoke_ms=" << report.total_smoke_ms << '\n';
    out << "steady_state_pi_result_ms=" << report.steady_state_pi_result_ms << '\n';
    out << "steady_state_pi_digits_per_second=" << report.steady_state_pi_digits_per_second << '\n';
    out << "cold_process_pi_digits_per_second=" << report.cold_process_pi_digits_per_second << '\n';
    out << "root_p_bits=" << report.root_p_bits << '\n';
    out << "root_q_bits=" << report.root_q_bits << '\n';
    out << "root_t_bits=" << report.root_t_bits << '\n';
    out << "reference_prefix_digits_checked=" << report.reference_prefix_digits_checked << '\n';
    out << "reported_decimal_digits=" << report.reported_decimal_digits << '\n';
    out << "closure_mode=" << report.closure_mode << '\n';
    out << "term_count_sufficient=" << (report.term_count_sufficient ? 1 : 0) << '\n';
    out << "root_reconstruction_match=" << (report.root_reconstruction_match ? 1 : 0) << '\n';
    out << "prefix_match=" << (report.prefix_match ? 1 : 0) << '\n';
    out << "pi_decimal_truncated=" << (report.pi_decimal_truncated ? 1 : 0) << '\n';
    out << "pi_decimal=" << report.pi_decimal << '\n';
}

}  // namespace project2::gpu_throughput_mainline
