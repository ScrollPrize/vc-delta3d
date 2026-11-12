/* Encoder ladder: size, encode time and decode time of encoder tunings.
 *
 *   ladder <decode rounds> <tuning>... -- <chunk.raw>...
 *
 * A tuning is reach,taps,stride,intercept,models,rounds (see vc_ng_tuning); a
 * preset name (delta, faster, fast, balanced, small, slower, medium, slow) runs that preset
 * as the library ships it. Chunks are 128^3 uint8. Every frame is
 * round-tripped; decodes are timed in interleaved rounds. Per chunk and tuning
 * it prints: bytes, encode ms, fastest decode ms, median decode ms.
 * Links the static library to reach the internal entry point. */
#include "../src/ng/ng_frame.h"
#include <vc_delta3d/codec.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { N = 128 * 128 * 128, MAX_TUNINGS = 64, MAX_ROUNDS = 31 };

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int ascending(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    vc_ng_tuning tuning[MAX_TUNINGS];
    static const struct { const char *name; int id; } presets[] = {{"delta", VC_DELTA}, {"faster", VC_FASTER},
        {"fast", VC_FAST}, {"balanced", VC_BALANCED}, {"small", VC_SMALL}, {"slower", VC_SLOWER},
        {"medium", VC_MEDIUM}, {"slow", VC_SLOW}};
    int preset[MAX_TUNINGS] = {0}, count = 0, arg = 2;
    if (argc < 5) return 1;
    int rounds = atoi(argv[1]);
    if (rounds < 1 || rounds > MAX_ROUNDS) return 1;
    for (; arg < argc && strcmp(argv[arg], "--"); ++arg, ++count) {
        if (count == MAX_TUNINGS) return 1;
        for (size_t k = 0; k < sizeof(presets) / sizeof(*presets); ++k)
            if (!strcmp(argv[arg], presets[k].name)) preset[count] = presets[k].id;
        if (preset[count]) continue;
        vc_ng_tuning *t = &tuning[count];
        if (sscanf(argv[arg], "%d,%d,%d,%lf,%u,%u", &t->reach, &t->taps, &t->stride, &t->intercept,
                   &t->models, &t->rounds) != 6) return 1;
    }
    uint8_t *src = malloc(N), *decoded = malloc(N), *frame[MAX_TUNINGS];
    size_t cap = 4 * (size_t)N + 131072;
    for (int i = 0; i < count; ++i) frame[i] = malloc(cap);
    for (++arg; arg < argc; ++arg) {
        FILE *f = fopen(argv[arg], "rb");
        if (!f || fread(src, 1, N, f) != N) { fprintf(stderr, "cannot read %s\n", argv[arg]); return 2; }
        fclose(f);
        size_t size[MAX_TUNINGS];
        double encode[MAX_TUNINGS], decode[MAX_TUNINGS][MAX_ROUNDS];
        for (int i = 0; i < count; ++i) {
            double start = now();
            int rc;
            if (preset[i] == VC_DELTA) {
                vc_encode_options options = vc_default_options();
                uint8_t *out = NULL;
                options.shape[0] = options.shape[1] = options.shape[2] = 128;
                options.preset = VC_DELTA;
                rc = !vc_compress(src, N, &options, &out, &size[i]) || size[i] > cap;
                if (!rc) memcpy(frame[i], out, size[i]);
                vc_free(out);
            } else {
                rc = preset[i] ? vc_ng_preset_encode(src, 128, (unsigned)preset[i], frame[i], cap, &size[i])
                               : vc_ng_tuned_encode(src, 128, &tuning[i], frame[i], cap, &size[i]);
            }
            encode[i] = now() - start;
            if (rc) { size[i] = 0; continue; }
            memset(decoded, 0xA5, N);
            if (!vc_decompress(frame[i], size[i], decoded, N) || memcmp(src, decoded, N)) {
                fprintf(stderr, "round trip failed: %s tuning %d\n", argv[arg], i);
                return 3;
            }
        }
        for (int r = 0; r < rounds; ++r)
            for (int j = 0; j < count; ++j) {
                int i = (j + r) % count;
                if (!size[i]) { decode[i][r] = 0; continue; }
                double start = now();
                vc_decompress(frame[i], size[i], decoded, N);
                decode[i][r] = now() - start;
            }
        printf("%s", argv[arg]);
        for (int i = 0; i < count; ++i) {
            qsort(decode[i], (size_t)rounds, sizeof(double), ascending);
            printf(" %zu %.1f %.3f %.3f", size[i], encode[i] * 1e3, decode[i][0] * 1e3, decode[i][rounds / 2] * 1e3);
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
