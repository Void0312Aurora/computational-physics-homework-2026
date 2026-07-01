#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <immintrin.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#define MAX_DIMENSION 64
#define MAX_BATCH_PREFIXES 32
#define COMMON_LCM_Q345 60
#define THRESHOLD_UNITS ((2 * COMMON_LCM_Q345) * (2 * COMMON_LCM_Q345))
#define SATURATED_OUTSIDE_UNITS (THRESHOLD_UNITS + 1)
#define STORAGE_U16 0
#define STORAGE_RANK8 1
#define STORAGE_BITSET 2
#define AXIS_ORDER_ASC 0
#define AXIS_ORDER_DESC 1

static double monotonic_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static unsigned long long parse_ull(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        exit(2);
    }
    return value;
}

static unsigned long long parse_positive_ull(const char *text, const char *name) {
    unsigned long long value = parse_ull(text, name);
    if (value == 0ULL) {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        exit(2);
    }
    return value;
}

static unsigned long long checked_mul_ull(unsigned long long a, unsigned long long b) {
    if (b != 0ULL && a > ULLONG_MAX / b) {
        fprintf(stderr, "Point count overflows unsigned long long\n");
        exit(2);
    }
    return a * b;
}

static int axis_unit_square(unsigned int q, unsigned int digit) {
    unsigned int numerator = (2U * digit + 1U) * ((unsigned int)COMMON_LCM_Q345 / q);
    return (int)(numerator * numerator);
}

static void append_axes(uint8_t q, int count, int *axis, uint8_t *bases) {
    for (int i = 0; i < count; ++i, ++(*axis)) {
        bases[*axis] = q;
    }
}

static void build_axes(
    int q2_count,
    int q3_count,
    int q4_count,
    int q5_count,
    int q6_count,
    int axis_order,
    uint8_t *bases,
    int units[MAX_DIMENSION][7]
) {
    int axis = 0;
    if (axis_order == AXIS_ORDER_DESC) {
        append_axes(6U, q6_count, &axis, bases);
        append_axes(5U, q5_count, &axis, bases);
        append_axes(4U, q4_count, &axis, bases);
        append_axes(3U, q3_count, &axis, bases);
        append_axes(2U, q2_count, &axis, bases);
    } else {
        append_axes(2U, q2_count, &axis, bases);
        append_axes(3U, q3_count, &axis, bases);
        append_axes(4U, q4_count, &axis, bases);
        append_axes(5U, q5_count, &axis, bases);
        append_axes(6U, q6_count, &axis, bases);
    }

    int dimension = q2_count + q3_count + q4_count + q5_count + q6_count;
    for (axis = 0; axis < dimension; ++axis) {
        unsigned int q = (unsigned int)bases[axis];
        for (unsigned int digit = 0U; digit < q; ++digit) {
            units[axis][digit] = axis_unit_square(q, digit);
        }
    }
}

static unsigned long long compute_points(const uint8_t *bases, int start_axis, int axis_count) {
    unsigned long long total = 1ULL;
    for (int i = 0; i < axis_count; ++i) {
        total = checked_mul_ull(total, (unsigned long long)bases[start_axis + i]);
    }
    return total;
}

static void *allocate_aligned(unsigned long long count, size_t value_size, const char *name) {
    if (value_size == 0U || count > (unsigned long long)(SIZE_MAX / value_size)) {
        fprintf(stderr, "%s table is too large for size_t\n", name);
        exit(2);
    }
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64U, (size_t)count * value_size) != 0) {
        fprintf(stderr, "%s table allocation failed\n", name);
        exit(2);
    }
    return ptr;
}

static int initialize_index(
    unsigned long long start,
    int start_axis,
    int axis_count,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    uint8_t *index
) {
    unsigned long long work = start;
    int radius_units = 0;
    for (int i = 0; i < axis_count; ++i) {
        int axis = start_axis + i;
        uint8_t digit = (uint8_t)(work % (unsigned long long)bases[axis]);
        index[i] = digit;
        radius_units += units[axis][digit];
        work /= (unsigned long long)bases[axis];
    }
    return radius_units;
}

static void increment_index(
    int start_axis,
    int axis_count,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    uint8_t *index,
    int *radius_units
) {
    for (int i = 0; i < axis_count; ++i) {
        int axis = start_axis + i;
        uint8_t old_index = index[i];
        uint8_t new_index = (uint8_t)(old_index + 1U);
        if (new_index < bases[axis]) {
            index[i] = new_index;
            *radius_units += units[axis][new_index] - units[axis][old_index];
            break;
        }
        index[i] = 0U;
        *radius_units += units[axis][0] - units[axis][old_index];
    }
}

static void build_tail_sums_u16(
    uint16_t *tail_sums,
    int tail_start_axis,
    int tail_dimension,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    unsigned long long tail_points
) {
    uint8_t index[MAX_DIMENSION];
    int radius_units = initialize_index(0ULL, tail_start_axis, tail_dimension, bases, units, index);

    for (unsigned long long point = 0ULL; point < tail_points; ++point) {
        tail_sums[point] = (uint16_t)(radius_units <= THRESHOLD_UNITS ? radius_units : SATURATED_OUTSIDE_UNITS);
        if (point + 1ULL == tail_points) {
            break;
        }
        increment_index(tail_start_axis, tail_dimension, bases, units, index, &radius_units);
    }
}

static int compare_int_ascending(const void *left, const void *right) {
    int a = *(const int *)left;
    int b = *(const int *)right;
    return (a > b) - (a < b);
}

static uint8_t *encode_tail_ranks_u8(
    const uint16_t *tail_sums,
    unsigned long long tail_points,
    int *threshold_to_rank,
    int *unique_value_count
) {
    unsigned char seen[SATURATED_OUTSIDE_UNITS + 1];
    int values[SATURATED_OUTSIDE_UNITS + 1];
    int value_to_rank[SATURATED_OUTSIDE_UNITS + 1];
    memset(seen, 0, sizeof(seen));
    for (int i = 0; i <= SATURATED_OUTSIDE_UNITS; ++i) {
        value_to_rank[i] = -1;
    }

    int count = 0;
    for (unsigned long long i = 0ULL; i < tail_points; ++i) {
        int value = (int)tail_sums[i];
        if (!seen[value]) {
            seen[value] = 1U;
            values[count++] = value;
        }
    }
    qsort(values, (size_t)count, sizeof(values[0]), compare_int_ascending);
    if (count > 255) {
        fprintf(stderr, "rank8 storage requires <=255 unique tail values, got %d\n", count);
        exit(2);
    }
    for (int i = 0; i < count; ++i) {
        value_to_rank[values[i]] = i;
    }

    int next = 0;
    for (int threshold = 0; threshold <= THRESHOLD_UNITS; ++threshold) {
        while (next < count && values[next] <= threshold) {
            ++next;
        }
        threshold_to_rank[threshold] = next;
    }

    uint8_t *tail_ranks = (uint8_t *)allocate_aligned(tail_points, sizeof(uint8_t), "rank8 tail");
    for (unsigned long long i = 0ULL; i < tail_points; ++i) {
        tail_ranks[i] = (uint8_t)value_to_rank[(int)tail_sums[i]];
    }

    *unique_value_count = count;
    return tail_ranks;
}

static uint64_t *build_rank_bitsets(
    const uint8_t *tail_ranks,
    unsigned long long tail_points,
    int unique_value_count,
    unsigned long long *tail_word_count
) {
    unsigned long long words = (tail_points + 63ULL) >> 6U;
    unsigned long long bitset_count = (unsigned long long)unique_value_count + 1ULL;
    if (words != 0ULL && bitset_count > ULLONG_MAX / words) {
        fprintf(stderr, "Bitset word count overflows unsigned long long\n");
        exit(2);
    }
    unsigned long long total_words = bitset_count * words;
    uint64_t *bitsets = (uint64_t *)allocate_aligned(total_words, sizeof(uint64_t), "rank bitset");
    memset(bitsets, 0, (size_t)total_words * sizeof(uint64_t));

    for (int rank_bound = 1; rank_bound <= unique_value_count; ++rank_bound) {
        uint64_t *current = bitsets + (unsigned long long)rank_bound * words;
        const uint64_t *previous = bitsets + (unsigned long long)(rank_bound - 1) * words;
        memcpy(current, previous, (size_t)words * sizeof(uint64_t));
        uint8_t rank_to_add = (uint8_t)(rank_bound - 1);
        for (unsigned long long point = 0ULL; point < tail_points; ++point) {
            if (tail_ranks[point] == rank_to_add) {
                current[point >> 6U] |= 1ULL << (point & 63ULL);
            }
        }
    }

    *tail_word_count = words;
    return bitsets;
}

static inline short bound_i16_from_remaining(int remaining_units) {
    int bound_units = remaining_units + 1;
    if (bound_units < 0) {
        bound_units = 0;
    } else if (bound_units > INT16_MAX) {
        bound_units = INT16_MAX;
    }
    return (short)bound_units;
}

static inline uint8_t rank_bound_from_remaining(const int *threshold_to_rank, int remaining_units) {
    if (remaining_units < 0) {
        return 0U;
    }
    if (remaining_units > THRESHOLD_UNITS) {
        remaining_units = THRESHOLD_UNITS;
    }
    return (uint8_t)threshold_to_rank[remaining_units];
}

static inline unsigned long long popcount_words_u64(const uint64_t *words, unsigned long long word_count) {
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i zero = _mm256_setzero_si256();
    __m256i total = _mm256_setzero_si256();
    unsigned long long i = 0ULL;
    unsigned long long vector_end = word_count & ~3ULL;
    for (; i < vector_end; i += 4ULL) {
        __m256i values = _mm256_loadu_si256((const __m256i *)(const void *)(words + i));
        __m256i low = _mm256_and_si256(values, low_mask);
        __m256i high = _mm256_and_si256(_mm256_srli_epi16(values, 4), low_mask);
        __m256i low_counts = _mm256_shuffle_epi8(lookup, low);
        __m256i high_counts = _mm256_shuffle_epi8(lookup, high);
        __m256i byte_counts = _mm256_add_epi8(low_counts, high_counts);
        total = _mm256_add_epi64(total, _mm256_sad_epu8(byte_counts, zero));
    }
    uint64_t lanes[4];
    _mm256_storeu_si256((__m256i *)(void *)lanes, total);
    unsigned long long count = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < word_count; ++i) {
        count += (unsigned long long)__builtin_popcountll(words[i]);
    }
    return count;
}

static inline void count_tail_batch_bitset(
    const uint64_t *rank_bitsets,
    unsigned long long tail_word_count,
    const int *remaining_units,
    const int *threshold_to_rank,
    int batch_size,
    unsigned long long *counts
) {
    for (int b = 0; b < batch_size; ++b) {
        uint8_t rank_bound = rank_bound_from_remaining(threshold_to_rank, remaining_units[b]);
        const uint64_t *words = rank_bitsets + (unsigned long long)rank_bound * tail_word_count;
        counts[b] = popcount_words_u64(words, tail_word_count);
    }
}

#define COUNT_U16_LANE(B) do { \
    __m256i mask_values = _mm256_cmpgt_epi16(bounds[(B)], values); \
    unsigned int mask = (unsigned int)_mm256_movemask_epi8(mask_values); \
    counts[(B)] += (unsigned long long)(__builtin_popcount(mask) >> 1); \
} while (0)

static inline void count_tail_batch_u16(
    const uint16_t *tail_sums,
    unsigned long long tail_points,
    const int *remaining_units,
    int batch_size,
    unsigned long long *counts
) {
    __m256i bounds[MAX_BATCH_PREFIXES];
    for (int b = 0; b < batch_size; ++b) {
        counts[b] = 0ULL;
        bounds[b] = _mm256_set1_epi16(bound_i16_from_remaining(remaining_units[b]));
    }

    unsigned long long i = 0ULL;
    unsigned long long vector_end = tail_points & ~15ULL;
    for (; i < vector_end; i += 16ULL) {
        __m256i values = _mm256_load_si256((const __m256i *)(const void *)(tail_sums + i));
        switch (batch_size) {
            case 32:
                COUNT_U16_LANE(31);
                COUNT_U16_LANE(30);
                COUNT_U16_LANE(29);
                COUNT_U16_LANE(28);
                COUNT_U16_LANE(27);
                COUNT_U16_LANE(26);
                COUNT_U16_LANE(25);
                COUNT_U16_LANE(24);
                COUNT_U16_LANE(23);
                COUNT_U16_LANE(22);
                COUNT_U16_LANE(21);
                COUNT_U16_LANE(20);
                COUNT_U16_LANE(19);
                COUNT_U16_LANE(18);
                COUNT_U16_LANE(17);
                COUNT_U16_LANE(16);
                /* fall through */
            case 16:
                COUNT_U16_LANE(15);
                COUNT_U16_LANE(14);
                COUNT_U16_LANE(13);
                COUNT_U16_LANE(12);
                COUNT_U16_LANE(11);
                COUNT_U16_LANE(10);
                COUNT_U16_LANE(9);
                COUNT_U16_LANE(8);
                /* fall through */
            case 8:
                COUNT_U16_LANE(7);
                COUNT_U16_LANE(6);
                COUNT_U16_LANE(5);
                COUNT_U16_LANE(4);
                /* fall through */
            case 4:
                COUNT_U16_LANE(3);
                COUNT_U16_LANE(2);
                /* fall through */
            case 2:
                COUNT_U16_LANE(1);
                /* fall through */
            default:
                COUNT_U16_LANE(0);
                break;
        }
    }

    for (; i < tail_points; ++i) {
        int value = (int)tail_sums[i];
        for (int b = 0; b < batch_size; ++b) {
            counts[b] += value <= remaining_units[b] ? 1ULL : 0ULL;
        }
    }
}

#undef COUNT_U16_LANE

#define COUNT_RANK8_LANE(B) do { \
    __m256i biased_bound = biased_bounds[(B)]; \
    __m256i mask_values = _mm256_cmpgt_epi8(biased_bound, biased_values); \
    unsigned int mask = (unsigned int)_mm256_movemask_epi8(mask_values); \
    counts[(B)] += (unsigned long long)__builtin_popcount(mask); \
} while (0)

#define COUNT_RANK8_LANE4(B) do { \
    __m256i biased_bound = biased_bounds[(B)]; \
    __m256i mask0 = _mm256_cmpgt_epi8(biased_bound, biased_values0); \
    __m256i mask1 = _mm256_cmpgt_epi8(biased_bound, biased_values1); \
    __m256i mask2 = _mm256_cmpgt_epi8(biased_bound, biased_values2); \
    __m256i mask3 = _mm256_cmpgt_epi8(biased_bound, biased_values3); \
    counts[(B)] += \
        (unsigned long long)__builtin_popcount((unsigned int)_mm256_movemask_epi8(mask0)) + \
        (unsigned long long)__builtin_popcount((unsigned int)_mm256_movemask_epi8(mask1)) + \
        (unsigned long long)__builtin_popcount((unsigned int)_mm256_movemask_epi8(mask2)) + \
        (unsigned long long)__builtin_popcount((unsigned int)_mm256_movemask_epi8(mask3)); \
} while (0)

static inline void count_tail_batch_rank8(
    const uint8_t *tail_ranks,
    unsigned long long tail_points,
    const int *remaining_units,
    const int *threshold_to_rank,
    int batch_size,
    unsigned long long *counts
) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    __m256i biased_bounds[MAX_BATCH_PREFIXES];
    uint8_t rank_bounds[MAX_BATCH_PREFIXES];
    for (int b = 0; b < batch_size; ++b) {
        counts[b] = 0ULL;
        rank_bounds[b] = rank_bound_from_remaining(threshold_to_rank, remaining_units[b]);
        biased_bounds[b] = _mm256_set1_epi8((char)(rank_bounds[b] ^ 0x80U));
    }

    unsigned long long i = 0ULL;
    if (batch_size == 32) {
        unsigned long long vector4_end = tail_points & ~127ULL;
        for (; i < vector4_end; i += 128ULL) {
            __m256i raw_values0 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i));
            __m256i raw_values1 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 32ULL));
            __m256i raw_values2 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 64ULL));
            __m256i raw_values3 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 96ULL));
            __m256i biased_values0 = _mm256_xor_si256(raw_values0, bias);
            __m256i biased_values1 = _mm256_xor_si256(raw_values1, bias);
            __m256i biased_values2 = _mm256_xor_si256(raw_values2, bias);
            __m256i biased_values3 = _mm256_xor_si256(raw_values3, bias);
            COUNT_RANK8_LANE4(0);
            COUNT_RANK8_LANE4(1);
            COUNT_RANK8_LANE4(2);
            COUNT_RANK8_LANE4(3);
            COUNT_RANK8_LANE4(4);
            COUNT_RANK8_LANE4(5);
            COUNT_RANK8_LANE4(6);
            COUNT_RANK8_LANE4(7);
            COUNT_RANK8_LANE4(8);
            COUNT_RANK8_LANE4(9);
            COUNT_RANK8_LANE4(10);
            COUNT_RANK8_LANE4(11);
            COUNT_RANK8_LANE4(12);
            COUNT_RANK8_LANE4(13);
            COUNT_RANK8_LANE4(14);
            COUNT_RANK8_LANE4(15);
            COUNT_RANK8_LANE4(16);
            COUNT_RANK8_LANE4(17);
            COUNT_RANK8_LANE4(18);
            COUNT_RANK8_LANE4(19);
            COUNT_RANK8_LANE4(20);
            COUNT_RANK8_LANE4(21);
            COUNT_RANK8_LANE4(22);
            COUNT_RANK8_LANE4(23);
            COUNT_RANK8_LANE4(24);
            COUNT_RANK8_LANE4(25);
            COUNT_RANK8_LANE4(26);
            COUNT_RANK8_LANE4(27);
            COUNT_RANK8_LANE4(28);
            COUNT_RANK8_LANE4(29);
            COUNT_RANK8_LANE4(30);
            COUNT_RANK8_LANE4(31);
        }
    } else if (batch_size == 16) {
        unsigned long long vector4_end = tail_points & ~127ULL;
        for (; i < vector4_end; i += 128ULL) {
            __m256i raw_values0 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i));
            __m256i raw_values1 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 32ULL));
            __m256i raw_values2 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 64ULL));
            __m256i raw_values3 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 96ULL));
            __m256i biased_values0 = _mm256_xor_si256(raw_values0, bias);
            __m256i biased_values1 = _mm256_xor_si256(raw_values1, bias);
            __m256i biased_values2 = _mm256_xor_si256(raw_values2, bias);
            __m256i biased_values3 = _mm256_xor_si256(raw_values3, bias);
            COUNT_RANK8_LANE4(0);
            COUNT_RANK8_LANE4(1);
            COUNT_RANK8_LANE4(2);
            COUNT_RANK8_LANE4(3);
            COUNT_RANK8_LANE4(4);
            COUNT_RANK8_LANE4(5);
            COUNT_RANK8_LANE4(6);
            COUNT_RANK8_LANE4(7);
            COUNT_RANK8_LANE4(8);
            COUNT_RANK8_LANE4(9);
            COUNT_RANK8_LANE4(10);
            COUNT_RANK8_LANE4(11);
            COUNT_RANK8_LANE4(12);
            COUNT_RANK8_LANE4(13);
            COUNT_RANK8_LANE4(14);
            COUNT_RANK8_LANE4(15);
        }
    } else if (batch_size == 8) {
        unsigned long long vector4_end = tail_points & ~127ULL;
        for (; i < vector4_end; i += 128ULL) {
            __m256i raw_values0 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i));
            __m256i raw_values1 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 32ULL));
            __m256i raw_values2 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 64ULL));
            __m256i raw_values3 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 96ULL));
            __m256i biased_values0 = _mm256_xor_si256(raw_values0, bias);
            __m256i biased_values1 = _mm256_xor_si256(raw_values1, bias);
            __m256i biased_values2 = _mm256_xor_si256(raw_values2, bias);
            __m256i biased_values3 = _mm256_xor_si256(raw_values3, bias);
            COUNT_RANK8_LANE4(0);
            COUNT_RANK8_LANE4(1);
            COUNT_RANK8_LANE4(2);
            COUNT_RANK8_LANE4(3);
            COUNT_RANK8_LANE4(4);
            COUNT_RANK8_LANE4(5);
            COUNT_RANK8_LANE4(6);
            COUNT_RANK8_LANE4(7);
        }
    } else if (batch_size == 4) {
        unsigned long long vector4_end = tail_points & ~127ULL;
        for (; i < vector4_end; i += 128ULL) {
            __m256i raw_values0 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i));
            __m256i raw_values1 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 32ULL));
            __m256i raw_values2 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 64ULL));
            __m256i raw_values3 = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i + 96ULL));
            __m256i biased_values0 = _mm256_xor_si256(raw_values0, bias);
            __m256i biased_values1 = _mm256_xor_si256(raw_values1, bias);
            __m256i biased_values2 = _mm256_xor_si256(raw_values2, bias);
            __m256i biased_values3 = _mm256_xor_si256(raw_values3, bias);
            COUNT_RANK8_LANE4(0);
            COUNT_RANK8_LANE4(1);
            COUNT_RANK8_LANE4(2);
            COUNT_RANK8_LANE4(3);
        }
    }

    unsigned long long vector_end = tail_points & ~31ULL;
    for (; i < vector_end; i += 32ULL) {
        __m256i raw_values = _mm256_load_si256((const __m256i *)(const void *)(tail_ranks + i));
        __m256i biased_values = _mm256_xor_si256(raw_values, bias);
        switch (batch_size) {
            case 32:
                COUNT_RANK8_LANE(31);
                COUNT_RANK8_LANE(30);
                COUNT_RANK8_LANE(29);
                COUNT_RANK8_LANE(28);
                COUNT_RANK8_LANE(27);
                COUNT_RANK8_LANE(26);
                COUNT_RANK8_LANE(25);
                COUNT_RANK8_LANE(24);
                COUNT_RANK8_LANE(23);
                COUNT_RANK8_LANE(22);
                COUNT_RANK8_LANE(21);
                COUNT_RANK8_LANE(20);
                COUNT_RANK8_LANE(19);
                COUNT_RANK8_LANE(18);
                COUNT_RANK8_LANE(17);
                COUNT_RANK8_LANE(16);
                /* fall through */
            case 16:
                COUNT_RANK8_LANE(15);
                COUNT_RANK8_LANE(14);
                COUNT_RANK8_LANE(13);
                COUNT_RANK8_LANE(12);
                COUNT_RANK8_LANE(11);
                COUNT_RANK8_LANE(10);
                COUNT_RANK8_LANE(9);
                COUNT_RANK8_LANE(8);
                /* fall through */
            case 8:
                COUNT_RANK8_LANE(7);
                COUNT_RANK8_LANE(6);
                COUNT_RANK8_LANE(5);
                COUNT_RANK8_LANE(4);
                /* fall through */
            case 4:
                COUNT_RANK8_LANE(3);
                COUNT_RANK8_LANE(2);
                /* fall through */
            case 2:
                COUNT_RANK8_LANE(1);
                /* fall through */
            default:
                COUNT_RANK8_LANE(0);
                break;
        }
    }

    for (; i < tail_points; ++i) {
        uint8_t value = tail_ranks[i];
        for (int b = 0; b < batch_size; ++b) {
            counts[b] += value < rank_bounds[b] ? 1ULL : 0ULL;
        }
    }
}

#undef COUNT_RANK8_LANE
#undef COUNT_RANK8_LANE4

static unsigned long long count_prefix_range_batch(
    unsigned long long start,
    unsigned long long end,
    int prefix_dimension,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    const void *tail_values,
    unsigned long long tail_points,
    unsigned long long tail_word_count,
    int batch_prefixes,
    int storage_mode,
    const int *threshold_to_rank,
    uint8_t *prefix_index
) {
    int prefix_radius = initialize_index(start, 0, prefix_dimension, bases, units, prefix_index);
    unsigned long long inside_points = 0ULL;
    unsigned long long prefix = start;

    while (prefix < end) {
        int remaining_units[MAX_BATCH_PREFIXES];
        unsigned long long counts[MAX_BATCH_PREFIXES];
        int batch_size = 0;
        for (; batch_size < batch_prefixes && prefix + (unsigned long long)batch_size < end; ++batch_size) {
            unsigned long long current_prefix = prefix + (unsigned long long)batch_size;
            remaining_units[batch_size] = THRESHOLD_UNITS - prefix_radius;
            if (current_prefix + 1ULL < end) {
                increment_index(0, prefix_dimension, bases, units, prefix_index, &prefix_radius);
            }
        }

        if (storage_mode == STORAGE_BITSET) {
            count_tail_batch_bitset(
                (const uint64_t *)tail_values,
                tail_word_count,
                remaining_units,
                threshold_to_rank,
                batch_size,
                counts
            );
        } else if (storage_mode == STORAGE_RANK8) {
            count_tail_batch_rank8(
                (const uint8_t *)tail_values,
                tail_points,
                remaining_units,
                threshold_to_rank,
                batch_size,
                counts
            );
        } else {
            count_tail_batch_u16(
                (const uint16_t *)tail_values,
                tail_points,
                remaining_units,
                batch_size,
                counts
            );
        }
        for (int b = 0; b < batch_size; ++b) {
            inside_points += counts[b];
        }
        prefix += (unsigned long long)batch_size;
    }
    return inside_points;
}

static long double estimate_scale(int q2_count, int q3_count, int q4_count, int q5_count, int q6_count) {
    long double scale = 1.0L;
    for (int i = 0; i < q2_count; ++i) {
        scale *= 1.0L;
    }
    for (int i = 0; i < q3_count; ++i) {
        scale *= 2.0L / 3.0L;
    }
    for (int i = 0; i < q4_count; ++i) {
        scale *= 0.5L;
    }
    for (int i = 0; i < q5_count; ++i) {
        scale *= 2.0L / 5.0L;
    }
    for (int i = 0; i < q6_count; ++i) {
        scale *= 1.0L / 3.0L;
    }
    return scale;
}

static const char *storage_mode_name(int storage_mode) {
    if (storage_mode == STORAGE_BITSET) {
        return "bitset";
    }
    return storage_mode == STORAGE_RANK8 ? "rank8" : "u16";
}

static const char *axis_order_name(int axis_order) {
    return axis_order == AXIS_ORDER_DESC ? "desc" : "asc";
}

int main(int argc, char **argv) {
    if (argc < 7 || argc > 12) {
        fprintf(
            stderr,
            "Usage: %s <q2_axis_count> <q3_axis_count> <q4_axis_count> <q5_axis_count> <q6_axis_count> <tail_dimension> [threads] [prefix_chunk_points] [batch_prefixes:1|2|4|8|16|32] [storage_mode:0 u16|1 rank8|2 bitset] [axis_order:0 asc|1 desc]\n",
            argv[0]
        );
        return 2;
    }

    unsigned long long q2_ull = parse_ull(argv[1], "q2_axis_count");
    unsigned long long q3_ull = parse_ull(argv[2], "q3_axis_count");
    unsigned long long q4_ull = parse_ull(argv[3], "q4_axis_count");
    unsigned long long q5_ull = parse_ull(argv[4], "q5_axis_count");
    unsigned long long q6_ull = parse_ull(argv[5], "q6_axis_count");
    unsigned long long tail_dimension_ull = parse_positive_ull(argv[6], "tail_dimension");
    unsigned long long dimension_ull = q2_ull + q3_ull + q4_ull + q5_ull + q6_ull;
    if (dimension_ull == 0ULL || dimension_ull > MAX_DIMENSION || tail_dimension_ull > dimension_ull) {
        fprintf(stderr, "Invalid mixed dimension or tail dimension\n");
        return 2;
    }
    if (
        q2_ull > (unsigned long long)INT_MAX ||
        q3_ull > (unsigned long long)INT_MAX ||
        q4_ull > (unsigned long long)INT_MAX ||
        q5_ull > (unsigned long long)INT_MAX ||
        q6_ull > (unsigned long long)INT_MAX
    ) {
        fprintf(stderr, "Axis count too large\n");
        return 2;
    }

    if (argc >= 8) {
        unsigned long long threads = parse_positive_ull(argv[7], "threads");
        if (threads > (unsigned long long)INT_MAX) {
            fprintf(stderr, "Thread count too large\n");
            return 2;
        }
        omp_set_num_threads((int)threads);
    }

    unsigned long long prefix_chunk_points = 32ULL;
    if (argc >= 9) {
        prefix_chunk_points = parse_positive_ull(argv[8], "prefix_chunk_points");
    }

    int batch_prefixes = 8;
    if (argc >= 10) {
        unsigned long long parsed = parse_positive_ull(argv[9], "batch_prefixes");
        if (!(parsed == 1ULL || parsed == 2ULL || parsed == 4ULL || parsed == 8ULL || parsed == 16ULL || parsed == 32ULL)) {
            fprintf(stderr, "batch_prefixes must be 1, 2, 4, 8, 16, or 32\n");
            return 2;
        }
        batch_prefixes = (int)parsed;
    }

    int storage_mode = STORAGE_RANK8;
    if (argc >= 11) {
        unsigned long long parsed = parse_ull(argv[10], "storage_mode");
        if (!(
            parsed == (unsigned long long)STORAGE_U16 ||
            parsed == (unsigned long long)STORAGE_RANK8 ||
            parsed == (unsigned long long)STORAGE_BITSET
        )) {
            fprintf(stderr, "storage_mode must be 0, 1, or 2\n");
            return 2;
        }
        storage_mode = (int)parsed;
    }

    int axis_order = AXIS_ORDER_ASC;
    if (argc >= 12) {
        unsigned long long parsed = parse_ull(argv[11], "axis_order");
        if (!(parsed == (unsigned long long)AXIS_ORDER_ASC || parsed == (unsigned long long)AXIS_ORDER_DESC)) {
            fprintf(stderr, "axis_order must be 0 asc or 1 desc\n");
            return 2;
        }
        axis_order = (int)parsed;
    }

    int q2_count = (int)q2_ull;
    int q3_count = (int)q3_ull;
    int q4_count = (int)q4_ull;
    int q5_count = (int)q5_ull;
    int q6_count = (int)q6_ull;
    int dimension = (int)dimension_ull;
    int tail_dimension = (int)tail_dimension_ull;
    int prefix_dimension = dimension - tail_dimension;
    uint8_t bases[MAX_DIMENSION];
    int units[MAX_DIMENSION][7] = {{0}};
    build_axes(q2_count, q3_count, q4_count, q5_count, q6_count, axis_order, bases, units);

    unsigned long long prefix_points = compute_points(bases, 0, prefix_dimension);
    unsigned long long tail_points = compute_points(bases, prefix_dimension, tail_dimension);
    unsigned long long total_points = checked_mul_ull(prefix_points, tail_points);
    unsigned long long chunk_count = (prefix_points + prefix_chunk_points - 1ULL) / prefix_chunk_points;
    if (chunk_count > (unsigned long long)LLONG_MAX) {
        fprintf(stderr, "Chunk count too large for OpenMP loop\n");
        return 2;
    }

    int threshold_to_rank[THRESHOLD_UNITS + 1];
    for (int i = 0; i <= THRESHOLD_UNITS; ++i) {
        threshold_to_rank[i] = 0;
    }
    int unique_tail_values = 0;
    void *tail_values = NULL;
    unsigned long long tail_bytes = 0ULL;
    unsigned long long tail_word_count = 0ULL;

    double total_start = monotonic_seconds();
    uint16_t *tail_sums_u16 = (uint16_t *)allocate_aligned(tail_points, sizeof(uint16_t), "u16 tail");
    build_tail_sums_u16(tail_sums_u16, prefix_dimension, tail_dimension, bases, units, tail_points);
    if (storage_mode == STORAGE_RANK8 || storage_mode == STORAGE_BITSET) {
        uint8_t *tail_ranks = encode_tail_ranks_u8(tail_sums_u16, tail_points, threshold_to_rank, &unique_tail_values);
        free(tail_sums_u16);
        if (storage_mode == STORAGE_BITSET) {
            tail_values = build_rank_bitsets(tail_ranks, tail_points, unique_tail_values, &tail_word_count);
            free(tail_ranks);
            tail_bytes =
                ((unsigned long long)unique_tail_values + 1ULL) *
                tail_word_count *
                (unsigned long long)sizeof(uint64_t);
        } else {
            tail_values = tail_ranks;
            tail_word_count = 0ULL;
            tail_bytes = tail_points * (unsigned long long)sizeof(uint8_t);
        }
    } else {
        tail_values = tail_sums_u16;
        unique_tail_values = 0;
        tail_word_count = 0ULL;
        tail_bytes = tail_points * (unsigned long long)sizeof(uint16_t);
    }
    double build_runtime_s = monotonic_seconds() - total_start;

    unsigned long long inside_points = 0ULL;
    double count_start = monotonic_seconds();

#pragma omp parallel reduction(+ : inside_points)
    {
        uint8_t prefix_index[MAX_DIMENSION];
#pragma omp for schedule(static)
        for (long long chunk = 0LL; chunk < (long long)chunk_count; ++chunk) {
            unsigned long long range_start = (unsigned long long)chunk * prefix_chunk_points;
            unsigned long long range_end = range_start + prefix_chunk_points;
            if (range_end > prefix_points || range_end < range_start) {
                range_end = prefix_points;
            }
            inside_points += count_prefix_range_batch(
                range_start,
                range_end,
                prefix_dimension,
                bases,
                units,
                tail_values,
                tail_points,
                tail_word_count,
                batch_prefixes,
                storage_mode,
                threshold_to_rank,
                prefix_index
            );
        }
    }

    double count_runtime_s = monotonic_seconds() - count_start;
    double total_runtime_s = monotonic_seconds() - total_start;
    long double estimate = (long double)inside_points * estimate_scale(q2_count, q3_count, q4_count, q5_count, q6_count);
    double total_points_per_s = total_runtime_s > 0.0 ? (double)total_points / total_runtime_s : INFINITY;
    double count_points_per_s = count_runtime_s > 0.0 ? (double)total_points / count_runtime_s : INFINITY;

    printf(
        "dimension,q2_axis_count,q3_axis_count,q4_axis_count,q5_axis_count,q6_axis_count,prefix_dimension,tail_dimension,"
        "prefix_points,tail_points,total_points,inside_points,estimate,total_runtime_s,"
        "build_runtime_s,count_runtime_s,total_points_per_s,count_points_per_s,threads,"
        "prefix_chunk_points,tail_bytes,tail_word_count,simd_lanes,batch_prefixes,storage_mode,axis_order,unique_tail_values,"
        "common_lcm,threshold_units,mode\n"
    );
    printf(
        "%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%.18Le,%.9f,%.9f,%.9f,%.9e,%.9e,%d,%llu,%llu,%llu,%d,%d,%s,%s,%d,%d,%d,mixed_q26_tiled_batch_avx2\n",
        dimension,
        q2_count,
        q3_count,
        q4_count,
        q5_count,
        q6_count,
        prefix_dimension,
        tail_dimension,
        prefix_points,
        tail_points,
        total_points,
        inside_points,
        estimate,
        total_runtime_s,
        build_runtime_s,
        count_runtime_s,
        total_points_per_s,
        count_points_per_s,
        omp_get_max_threads(),
        prefix_chunk_points,
        tail_bytes,
        tail_word_count,
        storage_mode == STORAGE_BITSET ? 64 : (storage_mode == STORAGE_RANK8 ? 32 : 16),
        batch_prefixes,
        storage_mode_name(storage_mode),
        axis_order_name(axis_order),
        unique_tail_values,
        COMMON_LCM_Q345,
        THRESHOLD_UNITS
    );

    free(tail_values);
    return 0;
}
