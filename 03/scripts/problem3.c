#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef HW03_RESULT_DIR
#define HW03_RESULT_DIR "result"
#endif

#define PROBLEM3_METRICS_OUTPUT HW03_RESULT_DIR "/problem3_half_metrics.csv"
#define PROBLEM3_ROUNDOFF_OUTPUT HW03_RESULT_DIR "/problem3_roundoff.csv"

static FILE *open_output(const char *path) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        perror(path);
        exit(EXIT_FAILURE);
    }
    return fp;
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

static void write_problem3_csv(void) {
    FILE *metrics = open_output(PROBLEM3_METRICS_OUTPUT);
    FILE *roundoff = open_output(PROBLEM3_ROUNDOFF_OUTPUT);
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

static void run_problem3(void) {
    write_problem3_csv();
    print_problem3_preview();
}

int main(void) {
    run_problem3();
    puts("Raw CSV files written to " HW03_RESULT_DIR ".");
    return 0;
}
