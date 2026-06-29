#include <math.h>
#include <quadmath.h>
#include <stdio.h>

#define GREGORY_TERMS 500000UL

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
    for (unsigned long remaining = terms; remaining > 0; --remaining) {
        unsigned long k = remaining - 1;
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (double)(2 * k + 1);
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
    for (unsigned long remaining = terms; remaining > 0; --remaining) {
        unsigned long k = remaining - 1;
        __float128 sign = (k % 2 == 0) ? 1.0Q : -1.0Q;
        sum += sign / (__float128)(2 * k + 1);
    }
    return 4.0Q * sum;
}

int main(void) {
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
    puts("                                 truncation error dominates this series.");

    return 0;
}
