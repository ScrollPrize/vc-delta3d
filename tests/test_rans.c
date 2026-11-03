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
            uint16_t table[256];
            for(unsigned s=0;s<256;s++)table[s]=frame[16+2*s]|(unsigned)frame[17+2*s]<<8;
            assert(vc_rans_decode_frequencies(table,frame+528,length-528,out,n));
            assert(!memcmp(src,out,n));
            assert(!vc_rans_decode_frequencies(table,frame+528,length-529,out,n));
            table[0]^=1;
            assert(!vc_rans_decode_frequencies(table,frame+528,length-528,out,n));
            uint32_t freq[256];uint8_t *fixed=NULL;size_t fixed_len=0;
            for(unsigned s=0;s<256;s++)freq[s]=frame[16+2*s]|(unsigned)frame[17+2*s]<<8;
            assert(vc_rans_encode_frequencies(src,n,16,freq,&fixed,&fixed_len));
            assert(fixed_len==length&&!memcmp(frame,fixed,length));free(fixed);
            for(unsigned s=0;s<256;s++)freq[s]=16;
            assert(vc_rans_encode_frequencies(src,n,0,freq,&fixed,&fixed_len));
            assert(vc_rans_decode(fixed,fixed_len,out,n)&&!memcmp(src,out,n));free(fixed);
            assert(!vc_rans_decode(frame + 16, length - 17, out, n));
            frame[16] ^= 1; /* Frequency sum becomes invalid. */
            assert(!vc_rans_decode(frame + 16, length - 16, out, n));
            free(frame); free(out); free(src);
        }
    }
    uint8_t *frame = NULL; size_t length = 1;
    assert(!vc_rans_encode(NULL, 1, 0, &frame, &length));
    assert(!vc_rans_encode((const uint8_t*)"x", SIZE_MAX, 0, &frame, &length));
    uint32_t bad[256]={0};
    uint16_t empty[256]={0};
    assert(!vc_rans_decode_frequencies(empty,NULL,0,NULL,0));
    assert(!vc_rans_encode_frequencies((const uint8_t*)"x",1,0,bad,&frame,&length));
    bad[0]=4096;
    assert(!vc_rans_encode_frequencies((const uint8_t*)"x",1,0,bad,&frame,&length));
    puts("rANS C roundtrip and malformed-frame tests passed");
    return 0;
}
