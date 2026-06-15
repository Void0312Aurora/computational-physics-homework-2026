#include <float.h>
#include <math.h>
#include <quadmath.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROBLEM1_CASE_COUNT 10
#define PROBLEM2_POINT_COUNT 2401
#define PROBLEM2_X_MIN 0.7
#define PROBLEM2_X_MAX 1.3
#ifndef HW03_SELECTED_PROBLEM
#define HW03_SELECTED_PROBLEM 0
#endif

typedef struct {
    const char *label;
    double value;
} b_case_t;

static const b_case_t problem1_cases[PROBLEM1_CASE_COUNT] = {
    {"10", 10.0},
    {"50", 50.0},
    {"97", 97.0},
    {"100", 100.0},
    {"1000", 1000.0},
    {"10000", 10000.0},
    {"100000", 100000.0},
    {"1000000", 1000000.0},
    {"100000000", 100000000.0},
    {"10000000000", 10000000000.0},
};

static const int poly_coeffs[11] = {1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1};

static FILE *open_output(const char *path) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        perror(path);
        exit(EXIT_FAILURE);
    }
    return fp;
}

static void format_f128(__float128 value, char *buffer, size_t size, const char *fmt) {
    quadmath_snprintf(buffer, size, fmt, value);
}

static float small_root_standard_f32(float b) {
    float r = sqrtf(b * b - 4.0f);
    return (b - r) * 0.5f;
}

static float small_root_rationalized_f32(float b) {
    float r = sqrtf(b * b - 4.0f);
    return 2.0f / (b + r);
}

static double small_root_standard_f64(double b) {
    double r = sqrt(b * b - 4.0);
    return (b - r) * 0.5;
}

static double small_root_rationalized_f64(double b) {
    double r = sqrt(b * b - 4.0);
    return 2.0 / (b + r);
}

static __float128 small_root_standard_f128(__float128 b) {
    __float128 r = sqrtq(b * b - 4.0Q);
    return (b - r) * 0.5Q;
}

static __float128 small_root_rationalized_f128(__float128 b) {
    __float128 r = sqrtq(b * b - 4.0Q);
    return 2.0Q / (b + r);
}

static float poly_direct_f32(float x) {
    float y = x - 1.0f;
    float value = 1.0f;
    for (int i = 0; i < 10; ++i) {
        value *= y;
    }
    return value;
}

static double poly_direct_f64(double x) {
    double y = x - 1.0;
    double value = 1.0;
    for (int i = 0; i < 10; ++i) {
        value *= y;
    }
    return value;
}

static __float128 poly_direct_f128(__float128 x) {
    __float128 y = x - 1.0Q;
    __float128 value = 1.0Q;
    for (int i = 0; i < 10; ++i) {
        value *= y;
    }
    return value;
}

static float poly_expanded_f32(float x) {
    float powers[11];
    powers[0] = 1.0f;
    for (int i = 1; i <= 10; ++i) {
        powers[i] = powers[i - 1] * x;
    }
    return 1.0f - 10.0f * powers[1] + 45.0f * powers[2] - 120.0f * powers[3] +
           210.0f * powers[4] - 252.0f * powers[5] + 210.0f * powers[6] -
           120.0f * powers[7] + 45.0f * powers[8] - 10.0f * powers[9] + powers[10];
}

static double poly_expanded_f64(double x) {
    double powers[11];
    powers[0] = 1.0;
    for (int i = 1; i <= 10; ++i) {
        powers[i] = powers[i - 1] * x;
    }
    return 1.0 - 10.0 * powers[1] + 45.0 * powers[2] - 120.0 * powers[3] +
           210.0 * powers[4] - 252.0 * powers[5] + 210.0 * powers[6] -
           120.0 * powers[7] + 45.0 * powers[8] - 10.0 * powers[9] + powers[10];
}

static __float128 poly_expanded_f128(__float128 x) {
    __float128 powers[11];
    powers[0] = 1.0Q;
    for (int i = 1; i <= 10; ++i) {
        powers[i] = powers[i - 1] * x;
    }
    return 1.0Q - 10.0Q * powers[1] + 45.0Q * powers[2] - 120.0Q * powers[3] +
           210.0Q * powers[4] - 252.0Q * powers[5] + 210.0Q * powers[6] -
           120.0Q * powers[7] + 45.0Q * powers[8] - 10.0Q * powers[9] + powers[10];
}

static float poly_horner_f32(float x) {
    float value = (float)poly_coeffs[0];
    for (int i = 1; i < 11; ++i) {
        value = value * x + (float)poly_coeffs[i];
    }
    return value;
}

static double poly_horner_f64(double x) {
    double value = (double)poly_coeffs[0];
    for (int i = 1; i < 11; ++i) {
        value = value * x + (double)poly_coeffs[i];
    }
    return value;
}

static __float128 poly_horner_f128(__float128 x) {
    __float128 value = (__float128)poly_coeffs[0];
    for (int i = 1; i < 11; ++i) {
        value = value * x + (__float128)poly_coeffs[i];
    }
    return value;
}

static _Float16 h_add(_Float16 a, _Float16 b) {
    return (_Float16)(a + b);
}

static _Float16 h_sub(_Float16 a, _Float16 b) {
    return (_Float16)(a - b);
}

static _Float16 h_mul(_Float16 a, _Float16 b) {
    return (_Float16)(a * b);
}

static _Float16 h_div(_Float16 a, _Float16 b) {
    return (_Float16)(a / b);
}

static uint16_t half_bits(_Float16 value) {
    uint16_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static _Float16 bits_to_half(uint16_t bits) {
    _Float16 value = 0.0;
    memcpy(&value, &bits, sizeof(bits));
    return value;
}

static _Float16 machine_epsilon_f16(void) {
    _Float16 one = (_Float16)1.0;
    _Float16 two = (_Float16)2.0;
    _Float16 eps = one;

    while (h_add(one, h_div(eps, two)) > one) {
        eps = h_div(eps, two);
    }
    return eps;
}

static _Float16 true_min_f16(void) {
    _Float16 x = (_Float16)1.0;
    _Float16 two = (_Float16)2.0;

    while (h_div(x, two) > (_Float16)0.0) {
        x = h_div(x, two);
    }
    return x;
}

static _Float16 roundoff_direct_f16(_Float16 t) {
    _Float16 one = (_Float16)1.0;
    _Float16 u = h_add(one, t);
    _Float16 v = h_mul(u, u);
    _Float16 w = h_sub(v, one);
    return h_div(w, t);
}

static _Float16 roundoff_rearranged_f16(_Float16 t) {
    return h_add((_Float16)2.0, t);
}

static float problem2_x_at_f32(int index) {
    float step = (float)((PROBLEM2_X_MAX - PROBLEM2_X_MIN) / (double)(PROBLEM2_POINT_COUNT - 1));
    return 0.7f + step * (float)index;
}

static double problem2_x_at_f64(int index) {
    double step = (PROBLEM2_X_MAX - PROBLEM2_X_MIN) / (double)(PROBLEM2_POINT_COUNT - 1);
    return PROBLEM2_X_MIN + step * (double)index;
}

static __float128 problem2_x_at_f128(int index) {
    __float128 step = (1.3Q - 0.7Q) / (__float128)(PROBLEM2_POINT_COUNT - 1);
    return 0.7Q + step * (__float128)index;
}

static void write_problem1_csv(void) {
    FILE *fp = open_output("result/problem1_roots.csv");

    fprintf(fp, "b,precision,method,root\n");
    for (int i = 0; i < PROBLEM1_CASE_COUNT; ++i) {
        float b32 = (float)problem1_cases[i].value;
        double b64 = problem1_cases[i].value;
        __float128 b128 = (__float128)problem1_cases[i].value;
        char standard_q[128];
        char rationalized_q[128];

        fprintf(fp, "%s,float,standard,%.9g\n", problem1_cases[i].label, small_root_standard_f32(b32));
        fprintf(fp, "%s,float,rationalized,%.9g\n", problem1_cases[i].label,
                small_root_rationalized_f32(b32));
        fprintf(fp, "%s,double,standard,%.17g\n", problem1_cases[i].label, small_root_standard_f64(b64));
        fprintf(fp, "%s,double,rationalized,%.17g\n", problem1_cases[i].label,
                small_root_rationalized_f64(b64));

        format_f128(small_root_standard_f128(b128), standard_q, sizeof(standard_q), "%.36Qg");
        format_f128(small_root_rationalized_f128(b128), rationalized_q, sizeof(rationalized_q), "%.36Qg");
        fprintf(fp, "%s,quad,standard,%s\n", problem1_cases[i].label, standard_q);
        fprintf(fp, "%s,quad,rationalized,%s\n", problem1_cases[i].label, rationalized_q);
    }

    fclose(fp);
}

static void write_problem2_csv(void) {
    FILE *fp = open_output("result/problem2_values.csv");

    fprintf(fp, "x,precision,method,value\n");
    for (int i = 0; i < PROBLEM2_POINT_COUNT; ++i) {
        float x32 = problem2_x_at_f32(i);
        double x64 = problem2_x_at_f64(i);
        __float128 x128 = problem2_x_at_f128(i);
        char x_q[128];
        char direct_q[128];
        char expanded_q[128];
        char horner_q[128];

        fprintf(fp, "%.9g,float,direct,%.9g\n", x32, poly_direct_f32(x32));
        fprintf(fp, "%.9g,float,expanded,%.9g\n", x32, poly_expanded_f32(x32));
        fprintf(fp, "%.9g,float,horner,%.9g\n", x32, poly_horner_f32(x32));

        fprintf(fp, "%.17g,double,direct,%.17g\n", x64, poly_direct_f64(x64));
        fprintf(fp, "%.17g,double,expanded,%.17g\n", x64, poly_expanded_f64(x64));
        fprintf(fp, "%.17g,double,horner,%.17g\n", x64, poly_horner_f64(x64));

        format_f128(x128, x_q, sizeof(x_q), "%.36Qg");
        format_f128(poly_direct_f128(x128), direct_q, sizeof(direct_q), "%.36Qg");
        format_f128(poly_expanded_f128(x128), expanded_q, sizeof(expanded_q), "%.36Qg");
        format_f128(poly_horner_f128(x128), horner_q, sizeof(horner_q), "%.36Qg");
        fprintf(fp, "%s,quad,direct,%s\n", x_q, direct_q);
        fprintf(fp, "%s,quad,expanded,%s\n", x_q, expanded_q);
        fprintf(fp, "%s,quad,horner,%s\n", x_q, horner_q);
    }

    fclose(fp);
}

static void write_problem3_csv(void) {
    FILE *metrics = open_output("result/problem3_half_metrics.csv");
    FILE *roundoff = open_output("result/problem3_roundoff.csv");
    _Float16 eps = machine_epsilon_f16();
    _Float16 min_normal = bits_to_half(0x0400u);
    _Float16 true_min = true_min_f16();
    _Float16 max_finite = bits_to_half(0x7bffu);

    fprintf(metrics, "metric,value,note\n");
    fprintf(metrics, "sizeof_half,%zu,_Float16 byte width on this machine\n", sizeof(_Float16));
    fprintf(metrics, "bits_of_one,0x%04x,bit pattern of (_Float16)1.0\n", half_bits((_Float16)1.0));
    fprintf(metrics, "machine_epsilon,%.17g,computed by repeated halving with explicit half rounding\n",
            (double)eps);
    fprintf(metrics, "min_normal,%.17g,binary16 bit pattern 0x0400\n", (double)min_normal);
    fprintf(metrics, "true_min,%.17g,smallest positive representable half value\n", (double)true_min);
    fprintf(metrics, "max_finite,%.17g,binary16 bit pattern 0x7bff\n", (double)max_finite);

    fprintf(roundoff, "t,direct,rearranged\n");
    for (int k = 1; k <= 15; ++k) {
        _Float16 t = (_Float16)ldexp(1.0, -k);
        fprintf(roundoff, "%.17g,%.17g,%.17g\n", (double)t, (double)roundoff_direct_f16(t),
                (double)roundoff_rearranged_f16(t));
    }

    fclose(metrics);
    fclose(roundoff);
}

static void print_environment_summary(void) {
    char one_tenth[128];

    format_f128(1.0Q / 10.0Q, one_tenth, sizeof(one_tenth), "%.36Qg");
    puts("HW/03 floating-point experiments");
    puts("================================");
    printf("sizeof(long double)           : %zu bytes\n", sizeof(long double));
    printf("LDBL_MANT_DIG                 : %d\n", LDBL_MANT_DIG);
    printf("__SIZEOF_FLOAT128__           : %d bytes\n", __SIZEOF_FLOAT128__);
    printf("FLT128_MANT_DIG               : %d\n", FLT128_MANT_DIG);
    printf("sizeof(_Float16)              : %zu bytes\n", sizeof(_Float16));
    printf("bit pattern of (_Float16)1.0  : 0x%04x\n", half_bits((_Float16)1.0));
    printf("sample __float128 value       : %s\n\n", one_tenth);
}

static void print_problem1_preview(void) {
    puts("Problem 1 preview | small root x2");
    puts("b           float standard   float rationalized   double standard          double rationalized");
    for (int i = 0; i < PROBLEM1_CASE_COUNT; ++i) {
        float b32 = (float)problem1_cases[i].value;
        double b64 = problem1_cases[i].value;
        printf("%-11s %-16.9g %-20.9g %-24.17g %-24.17g\n", problem1_cases[i].label,
               small_root_standard_f32(b32), small_root_rationalized_f32(b32),
               small_root_standard_f64(b64), small_root_rationalized_f64(b64));
    }
    putchar('\n');
}

static void print_problem2_preview(void) {
    double x = 1.0001;
    char expanded_q[128];
    char horner_q[128];

    format_f128(poly_expanded_f128((__float128)x), expanded_q, sizeof(expanded_q), "%.36Qg");
    format_f128(poly_horner_f128((__float128)x), horner_q, sizeof(horner_q), "%.36Qg");
    puts("Problem 2 preview | x = 1.0001");
    printf("float direct / expanded / horner  : %.9g / %.9g / %.9g\n", poly_direct_f32((float)x),
           poly_expanded_f32((float)x), poly_horner_f32((float)x));
    printf("double direct / expanded / horner : %.17g / %.17g / %.17g\n", poly_direct_f64(x),
           poly_expanded_f64(x), poly_horner_f64(x));
    printf("quad expanded / horner            : %s / %s\n\n", expanded_q, horner_q);
}

static void print_problem3_preview(void) {
    _Float16 t = (_Float16)ldexp(1.0, -12);
    puts("Problem 3 preview | half precision roundoff");
    printf("machine epsilon              : %.17g\n", (double)machine_epsilon_f16());
    printf("min normal / true min        : %.17g / %.17g\n", (double)bits_to_half(0x0400u),
           (double)true_min_f16());
    printf("example t                    : %.17g\n", (double)t);
    printf("direct ((1+t)^2-1)/t         : %.17g\n", (double)roundoff_direct_f16(t));
    printf("rearranged 2+t               : %.17g\n\n", (double)roundoff_rearranged_f16(t));
}

int main(void) {
    if (HW03_SELECTED_PROBLEM == 0 || HW03_SELECTED_PROBLEM == 1) {
        write_problem1_csv();
    }
    if (HW03_SELECTED_PROBLEM == 0 || HW03_SELECTED_PROBLEM == 2) {
        write_problem2_csv();
    }
    if (HW03_SELECTED_PROBLEM == 0 || HW03_SELECTED_PROBLEM == 3) {
        write_problem3_csv();
    }

    if (HW03_SELECTED_PROBLEM == 0) {
        print_environment_summary();
    }
    if (HW03_SELECTED_PROBLEM == 0 || HW03_SELECTED_PROBLEM == 1) {
        print_problem1_preview();
    }
    if (HW03_SELECTED_PROBLEM == 0 || HW03_SELECTED_PROBLEM == 2) {
        print_problem2_preview();
    }
    if (HW03_SELECTED_PROBLEM == 0 || HW03_SELECTED_PROBLEM == 3) {
        print_problem3_preview();
    }
    puts("Raw CSV files written to result/.");

    return 0;
}
