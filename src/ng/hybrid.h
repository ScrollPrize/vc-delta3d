#pragma once
#include "volcodec_ng_lib.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Candidate policy: 0 NG only, 1 NG + Delta3D, 2 NG + Delta3D + LPC,
 * 3 NG + Delta3D + eight CABAC predictor/context candidates.
 * One scalar uint8 128^3 chunk. Bound unchanged from NG (raw is a candidate).
 * Streams retain their native VCNG/D3D1 magic; use hybrid_decode for either.
 * Winner: 0 NG, 1 Delta3D, 2 LPC; 3..10 CABAC (NG contexts 0..3,
 * then Delta contexts 0..3). Equal sizes prefer the existing rANS candidate.
 * CABAC uses NCAB magic, version 1; hybrid_decode reads all three formats.
 */
int volcodec_ng_hybrid_encode(const uint8_t*, void*, size_t, size_t*, int policy, int* winner);
int volcodec_ng_hybrid_decode(const void*, size_t, uint8_t*, size_t);
#ifdef __cplusplus
}
#endif
