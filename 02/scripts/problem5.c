#include <math.h>
#include <stdio.h>

#define RECURRENCE_NMAX 40

static double z_next_direct_f64(int n, double z_n) {
    double a = ldexp(z_n * z_n, 2 - 2 * n);
    return exp2((double)n - 0.5) * sqrt(1.0 - sqrt(1.0 - a));
}

static double z_next_stable_f64(int n, double z_n) {
    double a = ldexp(z_n * z_n, 2 - 2 * n);
    return z_n * sqrt(2.0 / (1.0 + sqrt(1.0 - a)));
}

int main(void) {
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
    puts("                                 1 - sqrt(1 - a), then the error is amplified.");

    return 0;
}
