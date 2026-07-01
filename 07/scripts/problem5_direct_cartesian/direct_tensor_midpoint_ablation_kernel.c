#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_DIMENSION 64
#define MAX_BATCH 16
#define COMMON_LCM_Q345 60
#define THRESHOLD_UNITS ((2 * COMMON_LCM_Q345) * (2 * COMMON_LCM_Q345))
#define SATURATED_OUTSIDE_UNITS (THRESHOLD_UNITS + 1)

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

static unsigned long long checked_mul_ull(unsigned long long a, unsigned long long b) {
    if (b != 0ULL && a > ULLONG_MAX / b) {
        fprintf(stderr, "Point count overflows unsigned long long\n");
        exit(2);
    }
    return a * b;
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

static int axis_unit_square(unsigned int q, unsigned int digit) {
    unsigned int numerator = (2U * digit + 1U) * ((unsigned int)COMMON_LCM_Q345 / q);
    return (int)(numerator * numerator);
}

static double axis_midpoint(unsigned int q, unsigned int digit) {
    return (double)(2U * digit + 1U) / (2.0 * (double)q);
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
    uint8_t *bases,
    int units[MAX_DIMENSION][7],
    double values[MAX_DIMENSION][7]
) {
    int axis = 0;
    append_axes(2U, q2_count, &axis, bases);
    append_axes(3U, q3_count, &axis, bases);
    append_axes(4U, q4_count, &axis, bases);
    append_axes(5U, q5_count, &axis, bases);
    append_axes(6U, q6_count, &axis, bases);

    int dimension = q2_count + q3_count + q4_count + q5_count + q6_count;
    for (axis = 0; axis < dimension; ++axis) {
        unsigned int q = (unsigned int)bases[axis];
        for (unsigned int digit = 0U; digit < q; ++digit) {
            units[axis][digit] = axis_unit_square(q, digit);
            values[axis][digit] = axis_midpoint(q, digit);
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

static unsigned long long run_float_fullsum(
    int dimension,
    const uint8_t *bases,
    double values[MAX_DIMENSION][7],
    unsigned long long total_points
) {
    unsigned long long inside = 0ULL;
    for (unsigned long long point = 0ULL; point < total_points; ++point) {
        unsigned long long work = point;
        double radius = 0.0;
        for (int axis = 0; axis < dimension; ++axis) {
            uint8_t digit = (uint8_t)(work % (unsigned long long)bases[axis]);
            double x = values[axis][digit];
            radius += x * x;
            work /= (unsigned long long)bases[axis];
        }
        inside += radius <= 1.0 + 1.0e-15 ? 1ULL : 0ULL;
    }
    return inside;
}

static unsigned long long run_integer_incremental(
    int dimension,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    unsigned long long total_points
) {
    uint8_t index[MAX_DIMENSION];
    int radius_units = initialize_index(0ULL, 0, dimension, bases, units, index);
    unsigned long long inside = 0ULL;
    for (unsigned long long point = 0ULL; point < total_points; ++point) {
        inside += radius_units <= THRESHOLD_UNITS ? 1ULL : 0ULL;
        if (point + 1ULL == total_points) {
            break;
        }
        increment_index(0, dimension, bases, units, index, &radius_units);
    }
    return inside;
}

static unsigned long long run_signed_integer_incremental(
    int dimension,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    unsigned long long orthant_points
) {
    if (dimension >= 63) {
        fprintf(stderr, "Signed ablation dimension is too large\n");
        exit(2);
    }
    unsigned long long sign_count = 1ULL << (unsigned int)dimension;
    unsigned long long inside = 0ULL;
    for (unsigned long long sign = 0ULL; sign < sign_count; ++sign) {
        (void)sign;
        inside += run_integer_incremental(dimension, bases, units, orthant_points);
    }
    return inside;
}

static unsigned long long run_prefix_tail_recompute(
    int prefix_dimension,
    int tail_dimension,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    unsigned long long prefix_points,
    unsigned long long tail_points
) {
    uint8_t prefix_index[MAX_DIMENSION];
    int prefix_radius = initialize_index(0ULL, 0, prefix_dimension, bases, units, prefix_index);
    unsigned long long inside = 0ULL;
    for (unsigned long long prefix = 0ULL; prefix < prefix_points; ++prefix) {
        int remaining = THRESHOLD_UNITS - prefix_radius;
        uint8_t tail_index[MAX_DIMENSION];
        int tail_radius = initialize_index(0ULL, prefix_dimension, tail_dimension, bases, units, tail_index);
        for (unsigned long long tail = 0ULL; tail < tail_points; ++tail) {
            inside += tail_radius <= remaining ? 1ULL : 0ULL;
            if (tail + 1ULL == tail_points) {
                break;
            }
            increment_index(prefix_dimension, tail_dimension, bases, units, tail_index, &tail_radius);
        }
        if (prefix + 1ULL == prefix_points) {
            break;
        }
        increment_index(0, prefix_dimension, bases, units, prefix_index, &prefix_radius);
    }
    return inside;
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

static uint8_t rank_bound_from_remaining(const int *threshold_to_rank, int remaining_units) {
    if (remaining_units < 0) {
        return 0U;
    }
    if (remaining_units > THRESHOLD_UNITS) {
        remaining_units = THRESHOLD_UNITS;
    }
    return (uint8_t)threshold_to_rank[remaining_units];
}

static unsigned long long run_prefix_tail_table_u16_scalar(
    int prefix_dimension,
    int tail_dimension,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    unsigned long long prefix_points,
    unsigned long long tail_points
) {
    uint16_t *tail_sums = (uint16_t *)allocate_aligned(tail_points, sizeof(uint16_t), "u16 tail");
    build_tail_sums_u16(tail_sums, prefix_dimension, tail_dimension, bases, units, tail_points);

    uint8_t prefix_index[MAX_DIMENSION];
    int prefix_radius = initialize_index(0ULL, 0, prefix_dimension, bases, units, prefix_index);
    unsigned long long inside = 0ULL;
    for (unsigned long long prefix = 0ULL; prefix < prefix_points; ++prefix) {
        int remaining = THRESHOLD_UNITS - prefix_radius;
        for (unsigned long long tail = 0ULL; tail < tail_points; ++tail) {
            inside += (int)tail_sums[tail] <= remaining ? 1ULL : 0ULL;
        }
        if (prefix + 1ULL == prefix_points) {
            break;
        }
        increment_index(0, prefix_dimension, bases, units, prefix_index, &prefix_radius);
    }

    free(tail_sums);
    return inside;
}

static unsigned long long run_prefix_tail_rank8_scalar_batch(
    int prefix_dimension,
    int tail_dimension,
    const uint8_t *bases,
    int units[MAX_DIMENSION][7],
    unsigned long long prefix_points,
    unsigned long long tail_points,
    int batch_prefixes
) {
    uint16_t *tail_sums = (uint16_t *)allocate_aligned(tail_points, sizeof(uint16_t), "u16 tail");
    build_tail_sums_u16(tail_sums, prefix_dimension, tail_dimension, bases, units, tail_points);
    int threshold_to_rank[THRESHOLD_UNITS + 1];
    int unique_value_count = 0;
    uint8_t *tail_ranks = encode_tail_ranks_u8(tail_sums, tail_points, threshold_to_rank, &unique_value_count);
    (void)unique_value_count;
    free(tail_sums);

    uint8_t prefix_index[MAX_DIMENSION];
    int prefix_radius = initialize_index(0ULL, 0, prefix_dimension, bases, units, prefix_index);
    unsigned long long inside = 0ULL;
    unsigned long long prefix = 0ULL;
    while (prefix < prefix_points) {
        uint8_t bounds[MAX_BATCH];
        unsigned long long counts[MAX_BATCH] = {0ULL};
        int batch_size = 0;
        for (; batch_size < batch_prefixes && prefix + (unsigned long long)batch_size < prefix_points; ++batch_size) {
            bounds[batch_size] = rank_bound_from_remaining(threshold_to_rank, THRESHOLD_UNITS - prefix_radius);
            if (prefix + (unsigned long long)batch_size + 1ULL < prefix_points) {
                increment_index(0, prefix_dimension, bases, units, prefix_index, &prefix_radius);
            }
        }
        for (unsigned long long tail = 0ULL; tail < tail_points; ++tail) {
            uint8_t rank = tail_ranks[tail];
            for (int b = 0; b < batch_size; ++b) {
                counts[b] += rank < bounds[b] ? 1ULL : 0ULL;
            }
        }
        for (int b = 0; b < batch_size; ++b) {
            inside += counts[b];
        }
        prefix += (unsigned long long)batch_size;
    }

    free(tail_ranks);
    return inside;
}

static const char *variant_name(int variant) {
    switch (variant) {
        case 0:
            return "float_fullsum";
        case 1:
            return "integer_incremental";
        case 2:
            return "tail_recompute";
        case 3:
            return "tail_table_u16_scalar";
        case 4:
            return "rank8_scalar_batch1";
        case 5:
            return "rank8_scalar_batch16";
        case 6:
            return "signed_integer_incremental";
        default:
            return "unknown";
    }
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(
            stderr,
            "Usage: %s <variant:0..6> <q2_count> <q3_count> <q4_count> <q5_count> <q6_count> <tail_dimension>\n",
            argv[0]
        );
        return 2;
    }

    int variant = (int)parse_ull(argv[1], "variant");
    int q2_count = (int)parse_ull(argv[2], "q2_count");
    int q3_count = (int)parse_ull(argv[3], "q3_count");
    int q4_count = (int)parse_ull(argv[4], "q4_count");
    int q5_count = (int)parse_ull(argv[5], "q5_count");
    int q6_count = (int)parse_ull(argv[6], "q6_count");
    int tail_dimension = (int)parse_ull(argv[7], "tail_dimension");
    int dimension = q2_count + q3_count + q4_count + q5_count + q6_count;
    if (dimension <= 0 || dimension > MAX_DIMENSION || tail_dimension <= 0 || tail_dimension > dimension || variant < 0 || variant > 6) {
        fprintf(stderr, "Invalid variant or dimensions\n");
        return 2;
    }

    uint8_t bases[MAX_DIMENSION];
    int units[MAX_DIMENSION][7] = {{0}};
    double values[MAX_DIMENSION][7] = {{0.0}};
    build_axes(q2_count, q3_count, q4_count, q5_count, q6_count, bases, units, values);

    int prefix_dimension = dimension - tail_dimension;
    unsigned long long prefix_points = compute_points(bases, 0, prefix_dimension);
    unsigned long long tail_points = compute_points(bases, prefix_dimension, tail_dimension);
    unsigned long long orthant_points = checked_mul_ull(prefix_points, tail_points);
    unsigned long long total_points = orthant_points;
    if (variant == 6) {
        if (dimension >= 63) {
            fprintf(stderr, "Signed ablation dimension is too large\n");
            return 2;
        }
        total_points = checked_mul_ull(orthant_points, 1ULL << (unsigned int)dimension);
    }

    double start = monotonic_seconds();
    unsigned long long inside = 0ULL;
    if (variant == 0) {
        inside = run_float_fullsum(dimension, bases, values, orthant_points);
    } else if (variant == 1) {
        inside = run_integer_incremental(dimension, bases, units, orthant_points);
    } else if (variant == 2) {
        inside = run_prefix_tail_recompute(prefix_dimension, tail_dimension, bases, units, prefix_points, tail_points);
    } else if (variant == 3) {
        inside = run_prefix_tail_table_u16_scalar(prefix_dimension, tail_dimension, bases, units, prefix_points, tail_points);
    } else if (variant == 4) {
        inside = run_prefix_tail_rank8_scalar_batch(prefix_dimension, tail_dimension, bases, units, prefix_points, tail_points, 1);
    } else if (variant == 5) {
        inside = run_prefix_tail_rank8_scalar_batch(prefix_dimension, tail_dimension, bases, units, prefix_points, tail_points, 16);
    } else {
        inside = run_signed_integer_incremental(dimension, bases, units, orthant_points);
    }
    double runtime = monotonic_seconds() - start;
    double points_per_s = runtime > 0.0 ? (double)total_points / runtime : INFINITY;

    printf(
        "variant,dimension,q2_axis_count,q3_axis_count,q4_axis_count,q5_axis_count,q6_axis_count,prefix_dimension,tail_dimension,prefix_points,tail_points,total_points,inside_points,runtime_s,points_per_s,mode\n"
    );
    printf(
        "%s,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%.9f,%.9e,ablation_scalar\n",
        variant_name(variant),
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
        inside,
        runtime,
        points_per_s
    );
    return 0;
}
