#include <math.h>
#include <quadmath.h>
#include <stdio.h>

#define MACHIN_TOL 1.0e-35Q

static void format_quad(__float128 value, char *buffer, size_t size,
                        const char *fmt) {
    quadmath_snprintf(buffer, size, fmt, value);
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

int main(void) {
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
    printf("absolute error                 : %s\n", err_buf);

    return 0;
}
