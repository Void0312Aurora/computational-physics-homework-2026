#include <float.h>
#include <math.h>
#include <quadmath.h>
#include <stdint.h>
#include <stdio.h>

#define GREGORY_TERMS 500000UL
#define MACHIN_TOL 1.0e-35Q
#define RECURRENCE_NMAX 40
#ifndef HW02_SELECTED_PROBLEM
#define HW02_SELECTED_PROBLEM 0
#endif

#define DEFINE_MACHINE_EPSILON(TAG, TYPE, ONE)                                       \
    static TYPE machine_epsilon_##TAG(void) {                                        \
        TYPE eps = ONE;                                                              \
        while (ONE + eps / (TYPE)2 > ONE) {                                          \
            eps /= (TYPE)2;                                                          \
        }                                                                            \
        return eps;                                                                  \
    }

DEFINE_MACHINE_EPSILON(f32, float, 1.0f)
DEFINE_MACHINE_EPSILON(f64, double, 1.0)
DEFINE_MACHINE_EPSILON(f128, __float128, 1.0Q)

static void format_quad(__float128 value, char *buffer, size_t size,
                        const char *fmt) {
    quadmath_snprintf(buffer, size, fmt, value);
}

static unsigned long long min_terms_for_10_digits(void) {
    long double threshold = 0.5L * powl(10.0L, -10.0L);
    long double n = ceill((4.0L / threshold - 1.0L) / 2.0L);
    return (unsigned long long)n;
}

static double gregory_forward_f64(unsigned long terms) {
    double sum = 0.0;
    double sign = 1.0;
    for (unsigned long k = 1; k <= terms; ++k) {
        sum += sign / (double)(2 * k - 1);
        sign = -sign;
    }
    return 4.0 * sum;
}

static double gregory_backward_f64(unsigned long terms) {
    double sum = 0.0;
    double sign = (terms % 2 == 0) ? -1.0 : 1.0;
    for (unsigned long k = terms; k >= 1; --k) {
        sum += sign / (double)(2 * k - 1);
        sign = -sign;
        if (k == 1) {
            break;
        }
    }
    return 4.0 * sum;
}

static __float128 gregory_forward_f128(unsigned long terms) {
    __float128 sum = 0.0Q;
    __float128 sign = 1.0Q;
    for (unsigned long k = 1; k <= terms; ++k) {
        sum += sign / (__float128)(2 * k - 1);
        sign = -sign;
    }
    return 4.0Q * sum;
}

static __float128 gregory_backward_f128(unsigned long terms) {
    __float128 sum = 0.0Q;
    __float128 sign = (terms % 2 == 0) ? -1.0Q : 1.0Q;
    for (unsigned long k = terms; k >= 1; --k) {
        sum += sign / (__float128)(2 * k - 1);
        sign = -sign;
        if (k == 1) {
            break;
        }
    }
    return 4.0Q * sum;
}

static __float128 atan_series_q(__float128 x, __float128 tol, int *terms_used) {
    __float128 x2 = x * x;
    __float128 power = x;
    __float128 sum = 0.0Q;
    __float128 sign = 1.0Q;

    for (int n = 1; n < 1000000; ++n) {
        __float128 term = sign * power / (__float128)(2 * n - 1);
        sum += term;
        if (fabsq(term) < tol) {
            *terms_used = n;
            return sum;
        }
        power *= x2;
        sign = -sign;
    }

    *terms_used = -1;
    return sum;
}

static const char *endianness_name(void) {
    uint32_t probe = 0x01020304u;
    const unsigned char *bytes = (const unsigned char *)&probe;
    if (bytes[0] == 0x04) {
        return "little-endian";
    }
    if (bytes[0] == 0x01) {
        return "big-endian";
    }
    return "mixed/unknown";
}

static void print_problem1(void) {
    double pi_ref = acos(-1.0);
    double forward_f64 = gregory_forward_f64(GREGORY_TERMS);
    double backward_f64 = gregory_backward_f64(GREGORY_TERMS);
    __float128 pi_ref_q = acosq(-1.0Q);
    __float128 forward_f128 = gregory_forward_f128(GREGORY_TERMS);
    __float128 backward_f128 = gregory_backward_f128(GREGORY_TERMS);
    char pi_ref_q_buf[128];
    char forward_q_buf[128];
    char backward_q_buf[128];
    char err_forward_q_buf[128];
    char err_backward_q_buf[128];

    format_quad(pi_ref_q, pi_ref_q_buf, sizeof(pi_ref_q_buf), "%.36Qg");
    format_quad(forward_f128, forward_q_buf, sizeof(forward_q_buf), "%.36Qg");
    format_quad(backward_f128, backward_q_buf, sizeof(backward_q_buf), "%.36Qg");
    format_quad(fabsq(forward_f128 - pi_ref_q), err_forward_q_buf,
                sizeof(err_forward_q_buf), "%.12Qe");
    format_quad(fabsq(backward_f128 - pi_ref_q), err_backward_q_buf,
                sizeof(err_backward_q_buf), "%.12Qe");

    puts("Problem 1 | Gregory-Leibniz series");
    printf("terms used                     : %lu\n", GREGORY_TERMS);
    printf("estimated terms for 10 digits  : %llu\n", min_terms_for_10_digits());
    printf("reference pi (double)          : %.17g\n", pi_ref);
    printf("forward sum (double)           : %.17g\n", forward_f64);
    printf("backward sum (double)          : %.17g\n", backward_f64);
    printf("abs err forward (double)       : %.12e\n", fabs(forward_f64 - pi_ref));
    printf("abs err backward (double)      : %.12e\n", fabs(backward_f64 - pi_ref));
    printf("reference pi (__float128)      : %s\n", pi_ref_q_buf);
    printf("forward sum (__float128)       : %s\n", forward_q_buf);
    printf("backward sum (__float128)      : %s\n", backward_q_buf);
    printf("abs err forward (__float128)   : %s\n", err_forward_q_buf);
    printf("abs err backward (__float128)  : %s\n", err_backward_q_buf);
    puts("note                           : order only changes rounding slightly.");
    puts("                                 truncation error dominates this series.\n");
}

static void print_problem2(void) {
    int terms_1_5 = 0;
    int terms_1_239 = 0;
    __float128 atan_1_5 = atan_series_q(1.0Q / 5.0Q, MACHIN_TOL, &terms_1_5);
    __float128 atan_1_239 =
        atan_series_q(1.0Q / 239.0Q, MACHIN_TOL, &terms_1_239);
    __float128 pi_est = 16.0Q * atan_1_5 - 4.0Q * atan_1_239;
    __float128 pi_ref = acosq(-1.0Q);
    char pi_buf[128];
    char ref_buf[128];
    char err_buf[128];

    format_quad(pi_est, pi_buf, sizeof(pi_buf), "%.36Qf");
    format_quad(pi_ref, ref_buf, sizeof(ref_buf), "%.36Qf");
    format_quad(fabsq(pi_est - pi_ref), err_buf, sizeof(err_buf), "%.12Qe");

    puts("Problem 2 | Machin formula");
    printf("terms for arctan(1/5)          : %d\n", terms_1_5);
    printf("terms for arctan(1/239)        : %d\n", terms_1_239);
    printf("pi from Machin (__float128)    : %s\n", pi_buf);
    printf("reference pi (__float128)      : %s\n", ref_buf);
    printf("absolute error                 : %s\n\n", err_buf);
}

static void print_problem3(void) {
    uint32_t probe = 0x01020304u;
    const unsigned char *bytes = (const unsigned char *)&probe;

    puts("Problem 3 | Endianness");
    printf("machine byte order             : %s\n", endianness_name());
    printf("probe bytes                    : %02x %02x %02x %02x\n\n", bytes[0],
           bytes[1], bytes[2], bytes[3]);
}

static void print_problem4(void) {
    float eps_f32 = machine_epsilon_f32();
    double eps_f64 = machine_epsilon_f64();
    __float128 eps_f128 = machine_epsilon_f128();
    char eps_q_buf[128];
    char ref_q_buf[128];

    format_quad(eps_f128, eps_q_buf, sizeof(eps_q_buf), "%.40Qe");
#ifdef FLT128_EPSILON
    format_quad(FLT128_EPSILON, ref_q_buf, sizeof(ref_q_buf), "%.40Qe");
#else
    snprintf(ref_q_buf, sizeof(ref_q_buf), "unavailable");
#endif

    puts("Problem 4 | Machine epsilon");
    printf("computed epsilon (float)       : %.12e\n", eps_f32);
    printf("computed epsilon (double)      : %.18e\n", eps_f64);
    printf("computed epsilon (__float128)  : %s\n", eps_q_buf);
    printf("reference epsilon (float)      : %.12e\n", FLT_EPSILON);
    printf("reference epsilon (double)     : %.18e\n", DBL_EPSILON);
    printf("reference epsilon (__float128) : %s\n\n", ref_q_buf);
}

static double z_next_direct_f64(int n, double z_n) {
    double a = ldexp(z_n * z_n, 2 - 2 * n);
    return exp2((double)n - 0.5) * sqrt(1.0 - sqrt(1.0 - a));
}

static double z_next_stable_f64(int n, double z_n) {
    double a = ldexp(z_n * z_n, 2 - 2 * n);
    return z_n * sqrt(2.0 / (1.0 + sqrt(1.0 - a)));
}

static void print_problem5(void) {
    const int checkpoints[] = {2, 5, 10, 20, 25, 30, 31, 32, 35, 40};
    const int checkpoint_count = (int)(sizeof(checkpoints) / sizeof(checkpoints[0]));
    double pi_ref = acos(-1.0);
    double z_direct = 2.0;
    double z_stable = 2.0;

    puts("Problem 5 | Sequence converging to pi");
    puts("n      direct z_n              direct error            stable z_n              stable error");
    for (int n = 2, cursor = 0; n <= RECURRENCE_NMAX; ++n) {
        if (cursor < checkpoint_count && n == checkpoints[cursor]) {
            printf("%-6d %-23.16g %-23.16g %-23.16g %-23.16g\n", n, z_direct,
                   z_direct - pi_ref, z_stable, z_stable - pi_ref);
            ++cursor;
        }
        if (n < RECURRENCE_NMAX) {
            z_direct = z_next_direct_f64(n, z_direct);
            z_stable = z_next_stable_f64(n, z_stable);
        }
    }
    puts("note                           : the direct formula loses significance in");
    puts("                                 1 - sqrt(1 - a), then the error is amplified.\n");
}

int main(void) {
    if (HW02_SELECTED_PROBLEM == 0) {
        puts("HW/02 numerical experiments");
        puts("===========================");
        putchar('\n');
    }

    if (HW02_SELECTED_PROBLEM == 0 || HW02_SELECTED_PROBLEM == 1) {
        print_problem1();
    }
    if (HW02_SELECTED_PROBLEM == 0 || HW02_SELECTED_PROBLEM == 2) {
        print_problem2();
    }
    if (HW02_SELECTED_PROBLEM == 0 || HW02_SELECTED_PROBLEM == 3) {
        print_problem3();
    }
    if (HW02_SELECTED_PROBLEM == 0 || HW02_SELECTED_PROBLEM == 4) {
        print_problem4();
    }
    if (HW02_SELECTED_PROBLEM == 0 || HW02_SELECTED_PROBLEM == 5) {
        print_problem5();
    }

    return 0;
}
