#ifndef VC_DELTA3D_CODEC_H
#define VC_DELTA3D_CODEC_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Stable numeric IDs. Retired LPC=2 and tooslow=7 are not accepted. */
typedef enum { VC_DELTA=1, VC_FAST=3, VC_MEDIUM=4, VC_SLOW=5, VC_SLOWER=6 } vc_preset;
typedef struct {
    uint32_t shape[3]; /* Z,Y,X in elements */
    uint32_t channels; /* 0: scalar frame; >=1: channel container */
    unsigned layout;   /* 0: channels first; 1: channels last */
    unsigned element_size, quantization;
    vc_preset preset;
} vc_encode_options;
typedef struct {
    size_t decoded_bytes;
    uint32_t channels;
    unsigned quantization;
    int delta_mask; /* -1 when not explicitly recorded */
    unsigned legacy_scalar;
} vc_frame_info;
/* Default: fast, uint8, lossless, scalar. Caller supplies shape. */
vc_encode_options vc_default_options(void);
/* All functions return 1 on success, 0 on invalid input/allocation failure.
 * Output buffers are caller-owned except vc_compress, whose result must be
 * released with vc_free. No exception crosses this API. */
int vc_compress(const void*, size_t, const vc_encode_options*, uint8_t**, size_t*);
void vc_free(void*);
int vc_frame_inspect(const uint8_t*, size_t, vc_frame_info*);
int vc_decompress(const uint8_t*, size_t, void*, size_t);
/* Alternate magic applies only to legacy D3D1 framing (e.g. VCZ1). */
int vc_decompress_with_magic(const uint8_t*, size_t, const uint8_t magic[4], void*, size_t);
int vc_quantize(void*, size_t, unsigned element_size, unsigned width);
#ifdef __cplusplus
}
#endif
#endif
