// C interface to the context-modelled lossless coder in
// volcodec_ng_lossless.h (Predictor::Ctx). llctx.c compiles the header once
// as C; codec.cpp calls it through these functions.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Edge length and voxel count of the only chunk shape the coder takes.
enum { vc_llctx_dim = 128 };
#define VC_LLCTX_VOXELS ((size_t)128 * 128 * 128)

// Largest stream vc_llctx_encode writes (a raw chunk plus its 12-byte header).
size_t vc_llctx_encode_bound(void);

// Encodes one 128^3 uint8 chunk (z, y, x order, x fastest) into dst. Returns
// 0 and sets *out_n on success, a negative status otherwise.
int vc_llctx_encode(const uint8_t* src, uint8_t* dst, size_t cap, size_t* out_n);

// Decodes a stream of n bytes into dst (dst_n must be at least
// VC_LLCTX_VOXELS). Returns 0 on success, a negative status on invalid input.
int vc_llctx_decode(const uint8_t* enc, size_t n, uint8_t* dst, size_t dst_n);

#ifdef __cplusplus
}
#endif
