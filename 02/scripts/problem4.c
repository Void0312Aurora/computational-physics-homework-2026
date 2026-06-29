#include <float.h>
#include <quadmath.h>
#include <stdio.h>

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

int main(void) {
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
    printf("reference epsilon (__float128) : %s\n", ref_q_buf);

    return 0;
}
