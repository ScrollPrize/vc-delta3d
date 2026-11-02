#include "rans.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Exercise lane tails, constant tables, full alphabets and short frames.
 * Compatibility against original frames is additionally checked by verify_port.py. */
int main(void)
{
    const size_t sizes[] = {0, 1, 7, 8, 9, 31, 32, 33, 257, 65536};
    uint32_t seed = 1;
    for (unsigned mode = 0; mode < 4; ++mode) {
        for (unsigned t = 0; t < sizeof(sizes)/sizeof(*sizes); ++t) {
            size_t n = sizes[t], length;
            uint8_t *src = malloc(n + 1), *out = malloc(n + 1), *frame;
            assert(src && out);
            for (size_t i = 0; i < n; ++i) {
                seed = seed * 1664525u + 1013904223u;
                src[i] = mode == 0 ? 0 : mode == 1 ? 255 :
                         mode == 2 ? (uint8_t)i : (uint8_t)(seed >> 24);
            }
            assert(vc_rans_encode(src, n, 16, &frame, &length));
            assert(length >= 16 + 512 + 64);
            for (unsigned i = 0; i < 16; ++i) assert(frame[i] == 0);
            assert(vc_rans_decode(frame + 16, length - 16, out, n));
            assert(memcmp(src, out, n) == 0);
            assert(!vc_rans_decode(frame + 16, length - 17, out, n));
            frame[16] ^= 1; /* Frequency sum becomes invalid. */
            assert(!vc_rans_decode(frame + 16, length - 16, out, n));
            free(frame); free(out); free(src);
        }
    }
    uint8_t *frame = NULL; size_t length = 1;
    assert(!vc_rans_encode(NULL, 1, 0, &frame, &length));
    assert(!vc_rans_encode((const uint8_t*)"x", SIZE_MAX, 0, &frame, &length));
    puts("rANS C roundtrip and malformed-frame tests passed");
    return 0;
}
