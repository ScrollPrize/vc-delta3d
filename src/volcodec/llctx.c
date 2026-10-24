/* Compiles volcodec_ng_lossless.h once and exports the functions in llctx.h.
 * MIT License, Copyright (c) 2026 Chase Hendrick (see LICENSE next to this file). */
#include "llctx.h"

#include "volcodec_ng_lossless.h"

_Static_assert(VCN_DIM == vc_llctx_dim, "llctx.h and the codec agree on the chunk edge");
_Static_assert(VCN_VOXELS == VC_LLCTX_VOXELS, "llctx.h and the codec agree on the voxel count");

size_t vc_llctx_encode_bound(void) { return VCN_ENCODE_BOUND; }

int vc_llctx_encode(const uint8_t *src, uint8_t *dst, size_t cap, size_t *out_n) {
  return (int)vcn_encode_lossless_chunk(src, dst, cap, out_n);
}

int vc_llctx_decode(const uint8_t *enc, size_t n, uint8_t *dst, size_t dst_n) {
  return (int)vcn_decode(enc, n, dst, dst_n);
}
