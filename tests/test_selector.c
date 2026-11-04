#include "ng/selector.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const size_t lengths[] = {1, 2, 257, 4096, 65537, 262144};
    const unsigned alphabets[] = {1, 2, 3, 17, 256};
    unsigned tests = 0;
    for (unsigned l = 0; l < 6; ++l) for (unsigned a = 0; a < 5; ++a) for (unsigned runs = 0; runs < 2; ++runs) {
        size_t n = lengths[l], bytes = 0; uint8_t *src = malloc(n), *out = malloc(n), *encoded = NULL;
        assert(src && out); uint32_t seed = 37;
        for (size_t i = 0; i < n; ++i) {
            seed = seed * 1664525u + 1013904223u;
            src[i] = (uint8_t)((runs ? i / 113 : seed >> 16) % alphabets[a]);
        }
        assert(vc_selector_encode(src,n,&encoded,&bytes));
        assert(vc_selector_decode(encoded,bytes,out,n) && !memcmp(src,out,n));
        uint8_t *again = NULL; size_t second;
        assert(vc_selector_encode(src,n,&again,&second) && second == bytes && !memcmp(again,encoded,bytes)); free(again);
        /* Every short truncation, plus the tail; malformed inputs never abort. */
        for (size_t cut = 0; cut < bytes && cut < 64; ++cut) assert(!vc_selector_decode(encoded,cut,out,n));
        assert(!vc_selector_decode(encoded,bytes-1,out,n));
        uint8_t old = encoded[0]; encoded[0] = 250; assert(!vc_selector_decode(encoded,bytes,out,n)); encoded[0] = old;
        /* Corruption can be valid different data; only safety is asserted. */
        for (size_t at = 1; at < bytes && at < 64; ++at) { encoded[at] ^= 0x80; (void)vc_selector_decode(encoded,bytes,out,n); encoded[at] ^= 0x80; }
        free(encoded);free(src);free(out);++tests;
    }
    const uint8_t invalid[] = {1,128,128,128,128,128,128,128,128,128,128,128}; uint8_t out[16];
    assert(!vc_selector_decode(invalid,sizeof(invalid),out,sizeof(out)));
    assert(!vc_selector_decode(NULL,0,out,sizeof(out)));
    printf("%u selector roundtrip, deterministic, truncation and corruption tests passed\n",tests);
    return 0;
}
