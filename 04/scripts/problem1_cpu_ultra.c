#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

typedef struct {
    int64_t converged_total;
    int64_t iter_total;
    int64_t root_counts[3];
} GlobalStats;

typedef struct {
    int render_grid;
    int factor;
    int max_iter;
    double tol2;
    double late_tol2;
    double axis_min;
    double axis_step;
} RenderConfig;

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static double peak_rss_gib(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    return (double)usage.ru_maxrss / (1024.0 * 1024.0 * 1024.0);
#else
    return (double)usage.ru_maxrss * 1024.0 / (1024.0 * 1024.0 * 1024.0);
#endif
}

static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return;
    }
    mkdir(path, 0775);
}

static int write_binary_file(const char *path, const void *ptr, size_t item_size, size_t count) {
    FILE *fh = fopen(path, "wb");
    if (fh == NULL) {
        return 0;
    }
    const size_t written = fwrite(ptr, item_size, count, fh);
    fclose(fh);
    return written == count;
}

static inline int classify_root(double zr, double zi, double *min_d2) {
    const double sqrt3_half = 0.86602540378443864676;
    const double d0 = (zr - 1.0) * (zr - 1.0) + zi * zi;
    const double tmp = zr + 0.5;
    const double d1 = tmp * tmp + (zi - sqrt3_half) * (zi - sqrt3_half);
    const double d2 = tmp * tmp + (zi + sqrt3_half) * (zi + sqrt3_half);
    int root = 0;
    double best = d0;
    if (d1 < best) {
        best = d1;
        root = 1;
    }
    if (d2 < best) {
        best = d2;
        root = 2;
    }
    *min_d2 = best;
    return root;
}

static inline int mirror_root_label(int root) {
    if (root == 1) {
        return 2;
    }
    if (root == 2) {
        return 1;
    }
    return root;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --compute-grid N --render-grid M --tile-rows R --threads T "
            "[--max-iter K] [--tol eps] [--axis-min a] [--axis-max b] [--output-prefix prefix]\n",
            prog);
}

static int parse_int_arg(const char *name, int argc, char **argv, int *value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            *value = atoi(argv[i + 1]);
            return 1;
        }
    }
    return 0;
}

static int parse_double_arg(const char *name, int argc, char **argv, double *value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            *value = atof(argv[i + 1]);
            return 1;
        }
    }
    return 0;
}

static const char *parse_string_arg(const char *name, int argc, char **argv, const char *fallback) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return fallback;
}

static void render_block_upper(
    const RenderConfig *cfg,
    int render_row_start,
    int render_row_end,
    int render_col_start,
    int render_col_end,
    uint32_t *count0,
    uint32_t *count1,
    uint32_t *count2,
    uint32_t *iter_sum,
    uint16_t *conv_count,
    GlobalStats *stats
) {
    const int factor = cfg->factor;
    const int out_h = render_row_end - render_row_start;
    const int out_w = render_col_end - render_col_start;
    int64_t converged_total = 0;
    int64_t iter_total = 0;
    int64_t root0_total = 0;
    int64_t root1_total = 0;
    int64_t root2_total = 0;

    #pragma omp parallel for collapse(2) schedule(dynamic, 1) reduction(+:converged_total, iter_total, root0_total, root1_total, root2_total)
    for (int tile_row = 0; tile_row < out_h; ++tile_row) {
        for (int tile_col = 0; tile_col < out_w; ++tile_col) {
            const int idx = tile_row * out_w + tile_col;
            const int global_row0 = (render_row_start + tile_row) * factor;
            const int global_col0 = (render_col_start + tile_col) * factor;

            uint32_t local_count0 = 0;
            uint32_t local_count1 = 0;
            uint32_t local_count2 = 0;
            uint32_t local_iter_sum = 0;
            uint16_t local_conv_count = 0;

            for (int sr = 0; sr < factor; ++sr) {
                const double zi0 = cfg->axis_min + cfg->axis_step * (double)(global_row0 + sr);
                for (int sc = 0; sc < factor; ++sc) {
                    const double zr0 = cfg->axis_min + cfg->axis_step * (double)(global_col0 + sc);
                    double zr = zr0;
                    double zi = zi0;
                    int root = -1;
                    int step_hit = 0;

                    for (int step = 1; step <= cfg->max_iter; ++step) {
                        const double z2r = zr * zr - zi * zi;
                        const double z2i = 2.0 * zr * zi;
                        const double den = z2r * z2r + z2i * z2i;
                        if (den > 1.0e-14) {
                            const double invr = (1.0 / 3.0) * z2r / den;
                            const double invi = -(1.0 / 3.0) * z2i / den;
                            zr = (2.0 / 3.0) * zr + invr;
                            zi = (2.0 / 3.0) * zi + invi;
                        }
                        double min_d2;
                        const int cand = classify_root(zr, zi, &min_d2);
                        if (min_d2 < cfg->tol2) {
                            root = cand;
                            step_hit = step;
                            break;
                        }
                    }

                    if (root < 0) {
                        double min_d2;
                        const int cand = classify_root(zr, zi, &min_d2);
                        if (min_d2 < cfg->late_tol2) {
                            root = cand;
                            step_hit = cfg->max_iter;
                        }
                    }

                    if (root >= 0) {
                        local_conv_count += 1;
                        local_iter_sum += (uint32_t)step_hit;
                        converged_total += 1;
                        iter_total += step_hit;
                        if (root == 0) {
                            local_count0 += 1;
                            root0_total += 1;
                        } else if (root == 1) {
                            local_count1 += 1;
                            root1_total += 1;
                        } else {
                            local_count2 += 1;
                            root2_total += 1;
                        }
                    }
                }
            }

            count0[idx] = local_count0;
            count1[idx] = local_count1;
            count2[idx] = local_count2;
            iter_sum[idx] = local_iter_sum;
            conv_count[idx] = local_conv_count;
        }
    }

    stats->converged_total += converged_total;
    stats->iter_total += iter_total;
    stats->root_counts[0] += root0_total;
    stats->root_counts[1] += root1_total;
    stats->root_counts[2] += root2_total;
}

int main(int argc, char **argv) {
    int compute_grid = 0;
    int render_grid = 5000;
    int tile_rows = 64;
    int threads = 1;
    int max_iter = 55;
    double tol = 5.0e-7;
    double axis_min = -1.8;
    double axis_max = 1.8;
    const char *output_prefix = "problem1_cpu_ultra";
    const char *write_root_map = "";
    const char *write_iter_map = "";

    if (!parse_int_arg("--compute-grid", argc, argv, &compute_grid) ||
        !parse_int_arg("--threads", argc, argv, &threads)) {
        usage(argv[0]);
        return 1;
    }
    parse_int_arg("--render-grid", argc, argv, &render_grid);
    parse_int_arg("--tile-rows", argc, argv, &tile_rows);
    parse_int_arg("--max-iter", argc, argv, &max_iter);
    parse_double_arg("--tol", argc, argv, &tol);
    parse_double_arg("--axis-min", argc, argv, &axis_min);
    parse_double_arg("--axis-max", argc, argv, &axis_max);
    output_prefix = parse_string_arg("--output-prefix", argc, argv, output_prefix);
    write_root_map = parse_string_arg("--write-root-map", argc, argv, write_root_map);
    write_iter_map = parse_string_arg("--write-iter-map", argc, argv, write_iter_map);

    if (compute_grid <= 0 || render_grid <= 0 || tile_rows <= 0 || threads <= 0) {
        fprintf(stderr, "All integer arguments must be positive.\n");
        return 1;
    }
    if (render_grid % 2 != 0) {
        fprintf(stderr, "render-grid must be even for the symmetry-optimized CPU path.\n");
        return 1;
    }
    if (compute_grid % render_grid != 0) {
        fprintf(stderr, "compute-grid must be an integer multiple of render-grid.\n");
        return 1;
    }

    ensure_dir("result");
    ensure_dir("result/analysis");

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    const int factor = compute_grid / render_grid;
    const int half_start = render_grid / 2;
    const int upper_rows = render_grid - half_start;
    const double axis_step = (axis_max - axis_min) / (double)(compute_grid - 1);
    const int pixel_count = render_grid * render_grid;

    uint8_t *root_map = (uint8_t *)malloc((size_t)pixel_count * sizeof(uint8_t));
    float *iter_mean = (float *)malloc((size_t)pixel_count * sizeof(float));
    if (root_map == NULL || iter_mean == NULL) {
        fprintf(stderr, "Failed to allocate output buffers.\n");
        free(root_map);
        free(iter_mean);
        return 1;
    }
    memset(root_map, 255, (size_t)pixel_count * sizeof(uint8_t));
    memset(iter_mean, 0, (size_t)pixel_count * sizeof(float));

    RenderConfig cfg;
    cfg.render_grid = render_grid;
    cfg.factor = factor;
    cfg.max_iter = max_iter;
    cfg.tol2 = tol * tol;
    cfg.late_tol2 = 100.0 * tol * tol;
    cfg.axis_min = axis_min;
    cfg.axis_step = axis_step;

    GlobalStats stats = {0};
    const double start_time = wall_seconds();

    for (int render_row_start = half_start; render_row_start < render_grid; render_row_start += tile_rows) {
        int render_row_end = render_row_start + tile_rows;
        if (render_row_end > render_grid) {
            render_row_end = render_grid;
        }
        const int out_h = render_row_end - render_row_start;

        for (int render_col_start = 0; render_col_start < render_grid; render_col_start += tile_rows) {
            int render_col_end = render_col_start + tile_rows;
            if (render_col_end > render_grid) {
                render_col_end = render_grid;
            }
            const int out_w = render_col_end - render_col_start;
            const int block_pixels = out_h * out_w;

            uint32_t *count0 = (uint32_t *)calloc((size_t)block_pixels, sizeof(uint32_t));
            uint32_t *count1 = (uint32_t *)calloc((size_t)block_pixels, sizeof(uint32_t));
            uint32_t *count2 = (uint32_t *)calloc((size_t)block_pixels, sizeof(uint32_t));
            uint32_t *iter_sum = (uint32_t *)calloc((size_t)block_pixels, sizeof(uint32_t));
            uint16_t *conv_count = (uint16_t *)calloc((size_t)block_pixels, sizeof(uint16_t));
            if (count0 == NULL || count1 == NULL || count2 == NULL || iter_sum == NULL || conv_count == NULL) {
                fprintf(stderr, "Failed to allocate tile buffers.\n");
                free(count0);
                free(count1);
                free(count2);
                free(iter_sum);
                free(conv_count);
                free(root_map);
                free(iter_mean);
                return 1;
            }

            render_block_upper(
                &cfg,
                render_row_start,
                render_row_end,
                render_col_start,
                render_col_end,
                count0,
                count1,
                count2,
                iter_sum,
                conv_count,
                &stats
            );

            for (int r = 0; r < out_h; ++r) {
                for (int c = 0; c < out_w; ++c) {
                    const int local_idx = r * out_w + c;
                    const int global_r = render_row_start + r;
                    const int global_c = render_col_start + c;
                    const int global_idx = global_r * render_grid + global_c;
                    const uint32_t cc = conv_count[local_idx];
                    if (cc == 0) {
                        root_map[global_idx] = 255;
                        iter_mean[global_idx] = 0.0f;
                        continue;
                    }
                    uint8_t root = 0;
                    uint32_t best = count0[local_idx];
                    if (count1[local_idx] > best) {
                        best = count1[local_idx];
                        root = 1;
                    }
                    if (count2[local_idx] > best) {
                        root = 2;
                    }
                    root_map[global_idx] = root;
                    iter_mean[global_idx] = (float)iter_sum[local_idx] / (float)cc;
                }
            }

            free(count0);
            free(count1);
            free(count2);
            free(iter_sum);
            free(conv_count);
        }
    }

    for (int r = 0; r < half_start; ++r) {
        const int src_r = render_grid - 1 - r;
        for (int c = 0; c < render_grid; ++c) {
            const int dst_idx = r * render_grid + c;
            const int src_idx = src_r * render_grid + c;
            const uint8_t src_root = root_map[src_idx];
            if (src_root == 255) {
                root_map[dst_idx] = 255;
                iter_mean[dst_idx] = iter_mean[src_idx];
            } else {
                root_map[dst_idx] = (uint8_t)mirror_root_label((int)src_root);
                iter_mean[dst_idx] = iter_mean[src_idx];
            }
        }
    }

    stats.converged_total *= 2;
    stats.iter_total *= 2;
    stats.root_counts[0] *= 2;
    const int64_t sum12 = stats.root_counts[1] + stats.root_counts[2];
    stats.root_counts[1] = sum12;
    stats.root_counts[2] = sum12;

    const double elapsed = wall_seconds() - start_time;
    const double total_points = (double)compute_grid * (double)compute_grid;
    const double converged_fraction = (double)stats.converged_total / total_points;
    const double mean_iterations = stats.converged_total > 0
        ? (double)stats.iter_total / (double)stats.converged_total
        : 0.0;
    const double throughput_gpoints = total_points / elapsed / 1.0e9;

    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "result/analysis/%s.csv", output_prefix);
    FILE *csv = fopen(csv_path, "w");
    if (csv == NULL) {
        fprintf(stderr, "Failed to open CSV output: %s\n", csv_path);
        free(root_map);
        free(iter_mean);
        return 1;
    }
    fprintf(csv,
            "compute_grid,render_grid,factor,tile_rows,threads,max_iter,tol,elapsed_seconds,throughput_gpoints_per_s,peak_rss_gib,convergence_fraction,mean_iterations,root0_fraction,root1_fraction,root2_fraction\n");
    fprintf(csv,
            "%d,%d,%d,%d,%d,%d,%.12g,%.9f,%.9f,%.9f,%.12f,%.12f,%.12f,%.12f,%.12f\n",
            compute_grid,
            render_grid,
            factor,
            tile_rows,
            threads,
            max_iter,
            tol,
            elapsed,
            throughput_gpoints,
            peak_rss_gib(),
            converged_fraction,
            mean_iterations,
            stats.root_counts[0] / total_points,
            stats.root_counts[1] / total_points,
            stats.root_counts[2] / total_points);
    fclose(csv);

    char log_path[512];
    snprintf(log_path, sizeof(log_path), "result/analysis/%s.log", output_prefix);
    FILE *logf = fopen(log_path, "w");
    if (logf != NULL) {
        fprintf(logf, "Problem 1 CPU ultra benchmark\n");
        fprintf(logf, "compute_grid=%d render_grid=%d factor=%d tile_rows=%d threads=%d\n",
                compute_grid, render_grid, factor, tile_rows, threads);
        fprintf(logf, "elapsed_seconds=%.9f throughput_gpoints_per_s=%.9f peak_rss_gib=%.9f\n",
                elapsed, throughput_gpoints, peak_rss_gib());
        fprintf(logf, "convergence_fraction=%.12f mean_iterations=%.12f\n",
                converged_fraction, mean_iterations);
        fprintf(logf, "root_fractions=(%.12f, %.12f, %.12f)\n",
                stats.root_counts[0] / total_points,
                stats.root_counts[1] / total_points,
                stats.root_counts[2] / total_points);
        fclose(logf);
    }

    if (write_root_map[0] != '\0') {
        if (!write_binary_file(write_root_map, root_map, sizeof(uint8_t), (size_t)pixel_count)) {
            fprintf(stderr, "Failed to write root-map file: %s\n", write_root_map);
            free(root_map);
            free(iter_mean);
            return 1;
        }
    }
    if (write_iter_map[0] != '\0') {
        if (!write_binary_file(write_iter_map, iter_mean, sizeof(float), (size_t)pixel_count)) {
            fprintf(stderr, "Failed to write iter-map file: %s\n", write_iter_map);
            free(root_map);
            free(iter_mean);
            return 1;
        }
    }

    printf("elapsed_seconds=%.9f\n", elapsed);
    printf("throughput_gpoints_per_s=%.9f\n", throughput_gpoints);
    printf("peak_rss_gib=%.9f\n", peak_rss_gib());
    printf("convergence_fraction=%.12f\n", converged_fraction);
    printf("mean_iterations=%.12f\n", mean_iterations);

    free(root_map);
    free(iter_mean);
    return 0;
}
