#include <math.h>
#include <quadmath.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef HW03_RESULT_DIR
#define HW03_RESULT_DIR "result"
#endif

#define PROBLEM1_CASE_COUNT 10
#define PROBLEM1_OUTPUT HW03_RESULT_DIR "/problem1_roots.csv"

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

static void write_problem1_csv(void) {
    FILE *fp = open_output(PROBLEM1_OUTPUT);

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

static void run_problem1(void) {
    write_problem1_csv();
    print_problem1_preview();
}

int main(void) {
    run_problem1();
    puts("Raw CSV files written to " HW03_RESULT_DIR ".");
    return 0;
}
