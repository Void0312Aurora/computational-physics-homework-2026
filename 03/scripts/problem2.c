#include <math.h>
#include <quadmath.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef HW03_RESULT_DIR
#define HW03_RESULT_DIR "result"
#endif

#define PROBLEM2_POINT_COUNT 2401
#define PROBLEM2_X_MIN 0.7
#define PROBLEM2_X_MAX 1.3
#define PROBLEM2_OUTPUT HW03_RESULT_DIR "/problem2_values.csv"

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

static void write_problem2_csv(void) {
    FILE *fp = open_output(PROBLEM2_OUTPUT);

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

static void run_problem2(void) {
    write_problem2_csv();
    print_problem2_preview();
}

int main(void) {
    run_problem2();
    puts("Raw CSV files written to " HW03_RESULT_DIR ".");
    return 0;
}
