/* Native preset/Delta interoperability. All arithmetic is shared with the
 * legacy codec; no separate entropy encoder or filter implementation. */
#include "residual_bridge.h"
#include "delta_filter.h"
#include "rans.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

size_t delta_rans_size_lower_bound(const uint64_t *hist, size_t n, uint16_t *out)
{
    if (!n) { memset(out, 0, 256 * sizeof(*out)); return 0; }
    uint32_t freq[256];
    vc_rans_normalize(hist, n, freq);
    double bits = 0;
    for (unsigned s = 0; s < 256; ++s) {
        out[s] = (uint16_t)freq[s];
        if (hist[s]) bits += (double)hist[s] * (12. - log2((double)freq[s]));
    }
    /* Eight final states absorb at most 256 ideal bits. Each update's
     * rounding error is <1/262144 bits. Round down to whole renorm words. */
    double lower = bits - 256. - (double)n / 262144. - 1.;
    return 64 + (lower > 0 ? (size_t)(lower / 32.) * 4 : 0);
}

unsigned delta_prepare_residuals(const uint8_t *src, uint8_t *residuals)
{
    unsigned mask = vc_delta_select_mask(src, NG_Z, NG_Y, NG_X);
    memcpy(residuals, src, NG_N);
    vc_delta_filter_u8(residuals, NG_Z, NG_Y, NG_X, mask);
    return mask;
}
void delta_reconstruct_residuals(uint8_t *residuals, unsigned mask)
{
    vc_delta_unfilter_u8(residuals, NG_Z, NG_Y, NG_X, mask);
}
void delta_apply_mask(uint8_t *v, unsigned mask)
{
    vc_delta_filter_u8(v, NG_Z, NG_Y, NG_X, mask);
}
int delta_rans_encode(const uint8_t *v, size_t n, uint8_t *dst, size_t cap, size_t *size)
{
    uint8_t *frame; size_t bytes;
    if (!dst || !size || !vc_rans_encode(v, n, 0, &frame, &bytes)) return -1;
    if (bytes > cap) { free(frame); return -1; }
    memcpy(dst, frame, bytes);
    free(frame); *size = bytes;
    return 0;
}
int delta_rans_decode(const uint8_t *v, size_t n, uint8_t *dst, size_t count)
{
    return vc_rans_decode(v, n, dst, count) ? 0 : -1;
}
int delta_rans_decode_constant(const uint8_t *v, size_t n, uint8_t *dst, size_t count)
{
    if (!v || (!dst && count)) return -1;
    if (n == 576) {
        int symbol = -1, valid = 1;
        for (int s = 0; s < 256; ++s) {
            unsigned f = v[2*s] | (unsigned)v[2*s+1] << 8;
            if (f) { if (f != 4096 || symbol >= 0) valid = 0; symbol = s; }
        }
        if (valid && symbol >= 0) {
            for (size_t k = 0; k < 8; ++k) {
                uint64_t state; memcpy(&state, v + 512 + 8*k, 8);
                if (state != (UINT64_C(1) << 31)) return -1;
            }
            if (count) memset(dst, symbol, count);
            return 0;
        }
    }
    return delta_rans_decode(v, n, dst, count);
}
