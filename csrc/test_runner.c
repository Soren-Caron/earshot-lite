/* Batch runner for parity tests and benchmarking.
 *
 *   test_runner <features.f32> <n> <logits_out.i8> [repeat]
 *
 * features.f32: n x (in_frames * in_mels) raw log-mel, float32 little-endian.
 * Writes n x num_classes int8 logits and prints the mean microseconds per
 * inference (quantize + run), timed over `repeat` passes (default 1).
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "earshot.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s features.f32 n logits_out.i8 [repeat]\n", argv[0]);
        return 2;
    }
    const es_model *m = &earshot_model;
    const long n = atol(argv[2]);
    const int repeat = argc > 4 ? atoi(argv[4]) : 1;
    const size_t feat = (size_t)m->in_frames * m->in_mels;

    float *x = malloc(n * feat * sizeof(float)); /* harness only; the engine itself never allocates */
    int8_t *y = malloc(n * m->num_classes);
    if (!x || !y) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(x, sizeof(float), n * feat, f) != n * feat) {
        fprintf(stderr, "failed to read %ld clips from %s\n", n, argv[1]);
        return 1;
    }
    fclose(f);

    clock_t t0 = clock();
    for (int r = 0; r < repeat; r++)
        for (long i = 0; i < n; i++) earshot_predict(x + i * feat, y + i * m->num_classes);
    double us = 1e6 * (double)(clock() - t0) / CLOCKS_PER_SEC / ((double)n * repeat);

    f = fopen(argv[3], "wb");
    if (!f || fwrite(y, 1, n * m->num_classes, f) != (size_t)(n * m->num_classes)) return 1;
    fclose(f);
    printf("%.1f us/inference\n", us);
    free(x);
    free(y);
    return 0;
}
