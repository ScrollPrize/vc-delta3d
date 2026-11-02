/* volcodec_ng_lossless.h: lossless 128^3 uint8 chunk codec (volcodec-ng, format revision 2, the lossless
 * modes only, with mode 6).
 *
 * One chunk is 128^3 uint8 voxels, z-major, as a zarr v3 store with sharding_indexed keeps them: every
 * chunk decodes on its own. The encoder writes mode 6: each voxel is predicted from 40 of 82 causal
 * neighbours in the same row, the three rows above and the two previous slices, with weights fitted
 * per chunk by least squares (1/256 integers in the stream); the residual is coded with static rANS
 * tables, one per activity class of already decoded neighbours, in 32 interleaved states (details at
 * the mode 6 section below). The decoder also reads mode 5, an earlier lossless mode (12 neighbours,
 * canonical Huffman codes). The decoders are integer only, so every platform and build decodes the
 * same voxels; the encoder's fit uses double precision with floating-point contraction turned off, so
 * every build writes the same bytes.
 *
 * Stream: 12-byte header "VCNG", revision (2), mode, then 6 zero bytes; the payload follows.
 *   mode 2: raw chunk (2 MiB), written when coding would not make the chunk smaller;
 *   mode 3: constant chunk, one byte;
 *   mode 5: the earlier lossless Huffman mode (read only);
 *   mode 6: the context-modelled lossless mode described above (its payload starts with its stream
 *           version, 3; another version is refused with VCN_ERR_VERSION).
 * Modes 0, 1 and 4 of the volcodec-ng format are not part of this file: vcn_decode returns
 * VCN_ERR_MODE for them. Comments copied from the full codec that mention "mode 4" refer to its
 * predictor, which mode 5 shares. Comments in the mode 6 section that mention vcn_opts refer to the
 * full codec's options; here the encoder always writes mode 6.
 *
 * Generated from Chase Hendrick's volcodec_ng.h (sha256 13cb5d96c550ddf6...a8e0d25c) by lossless_header.py in
 * the codec source; the function bodies are verbatim copies.
 *
 * MIT License, Copyright (c) 2026 Chase Hendrick (see LICENSE next to this file). */
#ifndef VOLCODEC_NG_LOSSLESS_H
#define VOLCODEC_NG_LOSSLESS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "ng_geometry.h"

#ifndef NG_BLOCK_DIM
#define NG_BLOCK_DIM 128
#endif
#define VCN_DIM NG_BLOCK_DIM
#define VCN_VOXELS (VCN_DIM * VCN_DIM * VCN_DIM)
#define VCN_HDR 12u
#define VCN_REVISION 2u /* the revision this encoder writes; the decoder reads 1 and 2 */
/* the encoder falls back to a raw chunk, so a stream never exceeds this */
#define VCN_ENCODE_BOUND ((size_t)VCN_HDR + VCN_VOXELS)

#define VCN_MODE_RAW 2u
#define VCN_MODE_CONST 3u
#define VCN_MODE_LLHUF 5u
#define VCN_MODE_LLCTX 6u

typedef enum {
  VCN_OK = 0,
  VCN_ERR_ARG = -1,
  VCN_ERR_CAP = -2,
  VCN_ERR_CORRUPT = -3,
  VCN_ERR_NOMEM = -4,
  VCN_ERR_MODE = -5, /* a volcodec-ng chunk in a mode this file does not read (0, 1 or 4) */
  VCN_ERR_VERSION = -6 /* a mode 6 stream version this decoder does not read */
} vcn_status;

/* Encode one chunk losslessly into dst (cap bytes; VCN_ENCODE_BOUND always suffices). */
static inline vcn_status vcn_encode_lossless_chunk(const uint8_t *src, uint8_t *dst, size_t cap, size_t *out_n);
/* Decode one chunk of n bytes into dst (dst_n >= VCN_VOXELS). */
static inline vcn_status vcn_decode(const uint8_t *enc, size_t n, uint8_t *dst, size_t dst_n);
/* The chunk mode in a stream's header, or -1 when enc is not a volcodec-ng stream. */
static inline int vcn_chunk_mode(const uint8_t *enc, size_t n) {
  if (!enc || n < VCN_HDR || enc[0] != 'V' || enc[1] != 'C' || enc[2] != 'N' || enc[3] != 'G' || enc[4] < 1 ||
      enc[4] > VCN_REVISION)
    return -1;
  return enc[5];
}
static inline const char *vcn_status_string(vcn_status s) {
  switch (s) {
  case VCN_OK: return "ok";
  case VCN_ERR_ARG: return "bad argument";
  case VCN_ERR_CAP: return "output buffer too small";
  case VCN_ERR_CORRUPT: return "corrupt stream";
  case VCN_ERR_NOMEM: return "out of memory";
  case VCN_ERR_MODE: return "chunk mode not supported (only the lossless modes 2, 3, 5 and 6 are)";
  case VCN_ERR_VERSION: return "unsupported stream version (written by a newer or unknown encoder)";
  }
  return "unknown";
}

/* ======================================================================== */
/*                               implementation                             */
/* ======================================================================== */

/* No floating-point contraction in the implementation. The encoder's decisions are float, and a
 * compiler that fuses a * b + c into one multiply-add (one rounding instead of two) changes some of them,
 * so the streams would depend on the flags: GCC fuses by default in its GNU C modes whenever FMA is
 * available (-mfma, -march=native, every aarch64 target); before this region existed, a GCC -mfma or
 * -march=native build of the tests gave a different stream digest. GCC: optimize("fp-contract=off")
 * for every function defined here; Clang: FP_CONTRACT OFF, restored at the end of the header. Not covered:
 * Clang's -ffp-contract=fast (it ignores the pragma; -ffp-contract=fast-honor-pragmas keeps it), and
 * -ffast-math or -Ofast with any compiler. The decoders are integer only and unaffected. */
#if defined(__clang__)
#pragma float_control(push)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

/* Process-wide flags that are set on first use (CPU features, the decoder choice): relaxed atomic loads
 * and stores, so threads that make their first call at the same time do not race on them. */
#if defined(__GNUC__) || defined(__clang__)
#define VCN_LD(p) __atomic_load_n((p), __ATOMIC_RELAXED)
#define VCN_ST(p, v) __atomic_store_n((p), (v), __ATOMIC_RELAXED)
#else
#define VCN_LD(p) (*(p))
#define VCN_ST(p, v) (*(p) = (v))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VCN_INLINE __attribute__((always_inline)) static inline
#define VCN_MAYBE_UNUSED __attribute__((unused)) /* plain C kernels that a SIMD build may not call */
#else
#define VCN_INLINE static inline
#define VCN_MAYBE_UNUSED
#endif
/* Hot loops are plain C written for auto-vectorization; on x86-64 Linux they are compiled twice
 * (AVX2 and baseline) and picked at load time. Integer results are identical either way.
 * -DVCN_NO_CLONES builds the baseline only. */
#if !defined(VCN_NO_CLONES) && defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(__SANITIZE_ADDRESS__)
#define VCN_CLONES __attribute__((target_clones("avx2", "default")))
/* the same with BMI1 and BMI2 as well (one-micro-op variable shifts), for the bit coders of mode 5 */
#if defined(__clang__)
#define VCN_CLONES_BMI VCN_CLONES
#else
#define VCN_CLONES_BMI __attribute__((target_clones("arch=haswell", "default")))
#endif
#define VCN_KERNELS "auto-avx2"
#else
#define VCN_CLONES
#define VCN_CLONES_BMI
#define VCN_KERNELS "c"
#endif

/* ---- small helpers ---- */
VCN_INLINE int32_t vcn_clamp32(int64_t v, int32_t lo, int32_t hi) {
  return v < lo ? lo : (v > hi ? hi : (int32_t)v);
}
VCN_INLINE uint8_t vcn_u8(int32_t v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
static inline void vcn_wr16(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}
#include "ng_entropy_state.h"
/* little-endian 32-bit loads and stores (one move on little-endian targets) */
static inline void vcn_wr32(uint8_t *p, uint32_t v) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  memcpy(p, &v, 4);
#else
  for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
#endif
}
static inline uint32_t vcn_rd32(const uint8_t *p) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
#else
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
#endif
}

/* small dense least-squares solve (encoder only) */
static int vcn_solve(double *A, double *b, int n) { /* Cholesky; A n x n SPD, solves in place into b */
  for (int j = 0; j < n; j++) {
    double d = A[j * n + j];
    for (int k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (d <= 1e-12) return -1;
    d = sqrt(d);
    A[j * n + j] = d;
    for (int i = j + 1; i < n; i++) {
      double s2 = A[i * n + j];
      for (int k = 0; k < j; k++) s2 -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s2 / d;
    }
  }
  for (int i = 0; i < n; i++) {
    double s2 = b[i];
    for (int k = 0; k < i; k++) s2 -= A[i * n + k] * b[k];
    b[i] = s2 / A[i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    double s2 = b[i];
    for (int k = i + 1; k < n; k++) s2 -= A[k * n + i] * b[k];
    b[i] = s2 / A[i * n + i];
  }
  return 0;
}

/* AVX2 kernels on x86-64 with GCC or Clang, chosen at run time; the plain C kernels give the same
 * integers. */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(VCN_NO_AVX2)
#include <immintrin.h>
#define VCN_HAVE_AVX2 1
static inline int vcn_use_avx2(void) {
  static int v = -1;
  int r = VCN_LD(&v);
  if (r < 0) {
    r = __builtin_cpu_supports("avx2") ? 1 : 0;
    VCN_ST(&v, r);
  }
  return r;
}
#else
#define VCN_HAVE_AVX2 0
static inline int vcn_use_avx2(void) { return 0; }
#endif

/* ======================================================================== */
/*          predictor rows and least-squares fit (shared with mode 4)       */
/* ======================================================================== */
/* NEON on AArch64 (every AArch64 CPU has it, so there is no run-time check): the mode 6 kernels that have AVX2
 * forms also have NEON forms, which give the plain C integers (the test digest compares the builds).
 * -DVCN_NO_NEON or -DVCN_NO_SIMD builds the plain C kernels instead. */
#if defined(__aarch64__) && defined(__AARCH64EL__) && defined(__ARM_NEON) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(VCN_NO_NEON) && !defined(VCN_NO_SIMD)
#include <arm_neon.h>
#define VCN_HAVE_NEON 1
#else
#define VCN_HAVE_NEON 0
#endif
static inline const char *vcn_kernels(void) { return vcn_use_avx2() ? "avx2" : VCN_HAVE_NEON ? "neon" : "c"; }
#define VCN_LF_TAPS 12
#define VCN_LF_NCTX 8
#define VCN_LF_ST 8
#define VCN_LF_HDR (2 * VCN_LF_TAPS + 4 + 2 * (VCN_LF_NCTX - 1))

/* the six source rows of output row (z, y), padded: [N, NN, U, UN, US, UU][x + 2] */
static void vcn_lf_rows(const uint8_t *v, unsigned z, unsigned y, uint8_t pad[6][132]) {
  const size_t D = VCN_DIM, D2 = D * D;
  static const uint8_t flat[] = {
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
  const uint8_t *N = y ? v + z * D2 + (y - 1) * D : (z ? v + (z - 1) * D2 + y * D : flat);
  const uint8_t *NN = y >= 2 ? v + z * D2 + (y - 2) * D : N;
  const uint8_t *U = z ? v + (z - 1) * D2 + y * D : N;
  const uint8_t *UN = z && y ? v + (z - 1) * D2 + (y - 1) * D : U;
  const uint8_t *US = z && y + 1 < VCN_DIM ? v + (z - 1) * D2 + (y + 1) * D : U;
  const uint8_t *UU = z >= 2 ? v + (z - 2) * D2 + y * D : U;
  const uint8_t *src[6] = {N, NN, U, UN, US, UU};
  for (unsigned r = 0; r < 6; r++) {
    memcpy(pad[r] + 2, src[r], VCN_DIM);
    pad[r][0] = pad[r][1] = src[r][0];
    pad[r][130] = pad[r][131] = src[r][VCN_DIM - 1];
  }
}

VCN_INLINE void vcn_lf_taps(const uint8_t pad[6][132], int x, double *f) {
  const uint8_t *n = pad[0] + 2, *nn = pad[1] + 2, *u = pad[2] + 2, *un = pad[3] + 2, *us = pad[4] + 2, *uu = pad[5] + 2;
  f[0] = n[x], f[1] = n[x - 1], f[2] = n[x + 1], f[3] = nn[x], f[4] = n[x - 2], f[5] = n[x + 2];
  f[6] = u[x], f[7] = u[x - 1], f[8] = u[x + 1], f[9] = un[x], f[10] = us[x], f[11] = uu[x];
}

/* weights, bias and activity thresholds of modes 4 and 5, from the source chunk */
/* zs: slice step of the sample grid (4 in mode 4; 16 in mode 5: a quarter of the samples, a faster encoder) */
static void vcn_lf_fit(const uint8_t *v, unsigned zs, int32_t wq[VCN_LF_TAPS], int32_t *bias_out,
                       uint16_t thr[VCN_LF_NCTX - 1]) {
  uint8_t pad[6][132];
  /* least-squares weights on a sparse grid of rows and voxels */
  enum { NF = VCN_LF_TAPS + 1 };
  double A[NF * NF], bvec[NF];
  memset(A, 0, sizeof A), memset(bvec, 0, sizeof bvec);
  for (unsigned z = 2; z < VCN_DIM; z += zs)
    for (unsigned y = 2 + (z & 4u); y + 1 < VCN_DIM; y += 8) {
      vcn_lf_rows(v, z, y, pad);
      const uint8_t *row = v + ((size_t)z * VCN_DIM + y) * VCN_DIM;
      for (int x = (int)((y + z) & 3u); x < (int)VCN_DIM; x += 4) {
        double fd[NF];
        vcn_lf_taps(pad, x, fd);
        fd[VCN_LF_TAPS] = 1.0;
        for (int a = 0; a < NF; a++) {
          bvec[a] += fd[a] * row[x];
          for (int c = 0; c <= a; c++) A[a * NF + c] += fd[a] * fd[c];
        }
      }
    }
  for (int a = 0; a < NF; a++)
    for (int c = a + 1; c < NF; c++) A[a * NF + c] = A[c * NF + a];
  const int NT = VCN_LF_TAPS;
  const double cntv = A[NT * NF + NT] > 0 ? A[NT * NF + NT] : 1.0;
  double C[VCN_LF_TAPS * VCN_LF_TAPS], cb[VCN_LF_TAPS], mf[VCN_LF_TAPS], mt = bvec[NT] / cntv, tr = 0;
  for (int a = 0; a < NT; a++) mf[a] = A[a * NF + NT] / cntv;
  for (int a = 0; a < NT; a++) {
    cb[a] = bvec[a] / cntv - mf[a] * mt;
    for (int c = 0; c < NT; c++) C[a * NT + c] = A[a * NF + c] / cntv - mf[a] * mf[c];
    tr += C[a * NT + a];
  }
  for (int a = 0; a < NT; a++) C[a * NT + a] += 1e-7 * tr + 1e-9;
  int32_t bias = 0;
  if (vcn_solve(C, cb, NT) == 0) {
    double bb = mt;
    for (int k = 0; k < NT; k++) {
      double q = floor(cb[k] * 4096.0 + 0.5);
      wq[k] = (int32_t)(q < -32768 ? -32768 : (q > 32767 ? 32767 : q));
      bb -= wq[k] / 4096.0 * mf[k];
    }
    bb = floor(bb * 4096.0 + 0.5);
    bias = (int32_t)(bb < -(1 << 22) ? -(1 << 22) : (bb > (1 << 22) ? (1 << 22) : bb));
  } else { /* degenerate data: N + U - UN */
    static const int32_t L[VCN_LF_TAPS] = {4096, 0, 0, 0, 0, 0, 4096, 0, 0, -4096, 0, 0};
    memcpy(wq, L, sizeof L);
  }
  /* activity thresholds: quantiles on the same grid */
  {
    uint32_t hist[1531], ns = 0;
    memset(hist, 0, sizeof hist);
    for (unsigned z = 2; z < VCN_DIM; z += zs)
      for (unsigned y = 2 + (z & 2u); y + 1 < VCN_DIM; y += 4) {
        vcn_lf_rows(v, z, y, pad);
        const uint8_t *n = pad[0] + 2, *u = pad[2] + 2, *un = pad[3] + 2, *us = pad[4] + 2;
        for (int x = (int)((y + z) & 3u); x < (int)VCN_DIM; x += 4) {
          const int a = abs((int)n[x] - n[x - 1]) + abs((int)n[x] - n[x + 1]) + abs((int)u[x] - un[x]) +
                        abs((int)u[x] - us[x]) + abs((int)u[x] - u[x - 1]) + abs((int)u[x] - u[x + 1]);
          hist[a]++, ns++;
        }
      }
    uint32_t acc = 0;
    unsigned k = 0;
    for (unsigned a = 0; a < 1531 && k < VCN_LF_NCTX - 1; a++) {
      acc += hist[a];
      while (k < VCN_LF_NCTX - 1 && (uint64_t)acc * VCN_LF_NCTX >= (uint64_t)(k + 1) * ns) thr[k++] = (uint16_t)a;
    }
    while (k < VCN_LF_NCTX - 1) thr[k++] = 1530;
  }
  *bias_out = bias;
}

/* ======================================================================== */
/*                       Huffman lossless mode (mode 5)                      */
/* ======================================================================== */
/* Mode 5. The predictor, weights, bias and activity classes of mode 4 (the same row-parallel prediction
 * from the two previous rows and slices), with the residual coded by static canonical Huffman codes,
 * one per activity class, at most VCN_LH_L bits long, into VCN_LH_NS bit streams (voxel x of every row
 * goes to stream x mod VCN_LH_NS), so the decoder follows independent dependency chains. Bits are read
 * least significant first; each stream ends at a byte boundary with zero padding.
 * Payload: weights (12 x int16), bias (int32), thresholds (7 x u16), for each class the count n of
 * coded symbols (u16, 0 to 256) and n code lengths (4 bits each, low nibble first, 0 = absent), the
 * byte sizes of the first VCN_LH_NS - 1 streams (u32 each), then the streams. */
#define VCN_LH_L 11
#define VCN_LH_NS 4 /* the coder below is written for exactly 4 */
#define VCN_LH_TAB (1u << VCN_LH_L)
#define VCN_LH_RS 136 /* stride of a padded 16-bit row: voxel x at [x + 4], x = -2 .. 129 valid */

/* mode 4's prediction and context classes from rows already widened to 16 bits and padded (the same
 * integers as vcn_lf_pred; no widening in the loop). r: N, NN, U, UN, US, UU, each at its voxel 0 */
static void vcn_lf_pred16_c(const int16_t *const r[6], const int32_t *restrict wq, int32_t bias,
                            const uint16_t *restrict thr, uint8_t *restrict pred, uint8_t *restrict ctx) {
  const int16_t *n = r[0], *nn = r[1], *u = r[2], *un = r[3], *us = r[4], *uu = r[5];
  for (int x = 0; x < (int)VCN_DIM; x++) {
    const int32_t p = bias + wq[0] * n[x] + wq[1] * n[x - 1] + wq[2] * n[x + 1] + wq[3] * nn[x] + wq[4] * n[x - 2] +
                      wq[5] * n[x + 2] + wq[6] * u[x] + wq[7] * u[x - 1] + wq[8] * u[x + 1] + wq[9] * un[x] +
                      wq[10] * us[x] + wq[11] * uu[x];
    const int32_t q = (p + 2048) >> 12;
    pred[x] = (uint8_t)(q < 0 ? 0 : (q > 255 ? 255 : q));
    const int a = abs(n[x] - n[x - 1]) + abs(n[x] - n[x + 1]) + abs(u[x] - un[x]) + abs(u[x] - us[x]) +
                  abs(u[x] - u[x - 1]) + abs(u[x] - u[x + 1]);
    unsigned c = 0;
    for (unsigned k = 0; k < VCN_LF_NCTX - 1; k++) c += (unsigned)a > thr[k];
    ctx[x] = (uint8_t)c;
  }
}
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static void vcn_lf_pred16_avx2(const int16_t *const r[6], const int32_t *restrict wq,
                                                               int32_t bias, const uint16_t *restrict thr,
                                                               uint8_t *restrict pred, uint8_t *restrict ctx) {
  const int16_t *n = r[0], *nn = r[1], *u = r[2], *un = r[3], *us = r[4], *uu = r[5];
#define VCN_W2(a, b) _mm256_set1_epi32((int32_t)((uint32_t)(uint16_t)wq[a] | (uint32_t)(uint16_t)wq[b] << 16))
  const __m256i k01 = VCN_W2(0, 1), k23 = VCN_W2(2, 3), k45 = VCN_W2(4, 5), k67 = VCN_W2(6, 7), k89 = VCN_W2(8, 9),
                kab = VCN_W2(10, 11);
#undef VCN_W2
  const __m256i vb = _mm256_set1_epi32(bias + 2048);
  __m256i th[VCN_LF_NCTX - 1];
  for (unsigned k = 0; k < VCN_LF_NCTX - 1; k++) th[k] = _mm256_set1_epi16((short)thr[k]);
  for (int x = 0; x < (int)VCN_DIM; x += 16) {
#define VCN_L(p) _mm256_loadu_si256((const __m256i *)(p))
    const __m256i N = VCN_L(n + x), NW = VCN_L(n + x - 1), NE = VCN_L(n + x + 1), NN = VCN_L(nn + x),
                  NWW = VCN_L(n + x - 2), NEE = VCN_L(n + x + 2), U = VCN_L(u + x), UW = VCN_L(u + x - 1),
                  UE = VCN_L(u + x + 1), UN = VCN_L(un + x), US = VCN_L(us + x), UU = VCN_L(uu + x);
#undef VCN_L
    __m256i lo = vb, hi = vb;
#define VCN_P(A, B, K)                                                                                    \
  lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(A, B), K));                            \
  hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(A, B), K))
    VCN_P(N, NW, k01);
    VCN_P(NE, NN, k23);
    VCN_P(NWW, NEE, k45);
    VCN_P(U, UW, k67);
    VCN_P(UE, UN, k89);
    VCN_P(US, UU, kab);
#undef VCN_P
    const __m256i q = _mm256_packs_epi32(_mm256_srai_epi32(lo, 12), _mm256_srai_epi32(hi, 12));
    const __m256i q8 = _mm256_packus_epi16(q, q);
    _mm_storeu_si128((__m128i *)(pred + x), _mm256_castsi256_si128(_mm256_permute4x64_epi64(q8, 0x08)));
    __m256i a = _mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(N, NW)), _mm256_abs_epi16(_mm256_sub_epi16(N, NE)));
    a = _mm256_add_epi16(a, _mm256_abs_epi16(_mm256_sub_epi16(U, UN)));
    a = _mm256_add_epi16(a, _mm256_abs_epi16(_mm256_sub_epi16(U, US)));
    a = _mm256_add_epi16(a, _mm256_abs_epi16(_mm256_sub_epi16(U, UW)));
    a = _mm256_add_epi16(a, _mm256_abs_epi16(_mm256_sub_epi16(U, UE)));
    __m256i c = _mm256_setzero_si256();
    for (unsigned k = 0; k < VCN_LF_NCTX - 1; k++) c = _mm256_sub_epi16(c, _mm256_cmpgt_epi16(a, th[k]));
    const __m256i c8 = _mm256_packus_epi16(c, c);
    _mm_storeu_si128((__m128i *)(ctx + x), _mm256_castsi256_si128(_mm256_permute4x64_epi64(c8, 0x08)));
  }
}
#endif
static inline void vcn_lf_pred16(const int16_t *const r[6], const int32_t *restrict wq, int32_t bias,
                                 const uint16_t *restrict thr, uint8_t *restrict pred, uint8_t *restrict ctx) {
#if VCN_HAVE_AVX2
  if (vcn_use_avx2()) {
    vcn_lf_pred16_avx2(r, wq, bias, thr, pred, ctx);
    return;
  }
#endif
  vcn_lf_pred16_c(r, wq, bias, thr, pred, ctx);
}
/* widen a row of 128 voxels into a padded 16-bit row (two replicated voxels on each side) */
static inline void vcn_lf_widen(const uint8_t *src, int16_t *dst) {
  for (unsigned x = 0; x < VCN_DIM; x++) dst[x + 4] = src[x];
  dst[2] = dst[3] = src[0];
  dst[VCN_DIM + 4] = dst[VCN_DIM + 5] = src[VCN_DIM - 1];
}
/* the six source rows of output row (z, y), as vcn_lf_rows chooses them; s0, s1, s2: the padded 16-bit
 * slices z, z - 1, z - 2 (rows VCN_LH_RS apart), flat: a padded row of 128 */
static inline void vcn_lf_src16(const int16_t *s0, const int16_t *s1, const int16_t *s2, const int16_t *flat,
                                unsigned z, unsigned y, const int16_t *r[6]) {
  const size_t R = VCN_LH_RS;
  const int16_t *N = y ? s0 + (y - 1) * R : (z ? s1 + y * R : flat);
  const int16_t *U = z ? s1 + y * R : N;
  r[0] = N + 4;
  r[1] = (y >= 2 ? s0 + (y - 2) * R : N) + 4;
  r[2] = U + 4;
  r[3] = (z && y ? s1 + (y - 1) * R : U) + 4;
  r[4] = (z && y + 1 < VCN_DIM ? s1 + (y + 1) * R : U) + 4;
  r[5] = (z >= 2 ? s2 + y * R : U) + 4;
}

static inline uint64_t vcn_rd64(const uint8_t *p) {
  uint64_t x;
  memcpy(&x, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  x = __builtin_bswap64(x);
#endif
  return x;
}
static inline void vcn_wr64(uint8_t *p, uint64_t x) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  x = __builtin_bswap64(x);
#endif
  memcpy(p, &x, 8);
}

/* code lengths of a length-limited Huffman code for cnt (0 = absent); every present symbol gets a code */
static void vcn_lh_lengths(const uint32_t cnt[256], uint8_t len[256]) {
  unsigned idx[256], m = 0;
  memset(len, 0, 256);
  for (unsigned s = 0; s < 256; s++)
    if (cnt[s]) idx[m++] = s;
  if (m == 0) return;
  if (m == 1) {
    len[idx[0]] = 1;
    return;
  }
  /* sort by count ascending, then symbol (insertion sort; m <= 256, once per class) */
  for (unsigned i = 1; i < m; i++) {
    const unsigned t = idx[i];
    unsigned j = i;
    while (j > 0 && (cnt[idx[j - 1]] > cnt[t] || (cnt[idx[j - 1]] == cnt[t] && idx[j - 1] > t))) idx[j] = idx[j - 1], j--;
    idx[j] = t;
  }
  /* two-queue Huffman: leaves in idx order, internal nodes in creation order (nondecreasing weights) */
  uint64_t w[512];
  int par[512];
  for (unsigned i = 0; i < m; i++) w[i] = cnt[idx[i]];
  unsigned ql = 0, qi = m, nn = m;
  for (unsigned k = 0; k + 1 < m; k++) {
    unsigned pick[2];
    for (int c = 0; c < 2; c++) {
      if (ql < m && (qi >= nn || w[ql] <= w[qi])) pick[c] = ql++;
      else pick[c] = qi++;
    }
    w[nn] = w[pick[0]] + w[pick[1]];
    par[pick[0]] = par[pick[1]] = (int)nn;
    nn++;
  }
  unsigned depth[512];
  depth[nn - 1] = 0;
  for (int i = (int)nn - 2; i >= 0; i--) depth[i] = depth[par[i]] + 1;
  /* limit to VCN_LH_L: clamp, then lengthen the least frequent codes until the Kraft sum fits, then
   * shorten the most frequent codes while it still fits (units of 2^-L) */
  unsigned l[256];
  uint64_t K = 0;
  for (unsigned i = 0; i < m; i++) {
    l[i] = depth[i] > VCN_LH_L ? VCN_LH_L : depth[i];
    K += 1ull << (VCN_LH_L - l[i]);
  }
  while (K > VCN_LH_TAB) {
    unsigned best = m;
    for (unsigned i = 0; i < m; i++) /* least frequent first: the longest code below the limit */
      if (l[i] < VCN_LH_L && (best == m || l[i] > l[best])) best = i;
    l[best]++;
    K -= 1ull << (VCN_LH_L - l[best]);
  }
  for (unsigned i = m; i-- > 0;)
    while (l[i] > 1 && K + (1ull << (VCN_LH_L - l[i])) <= VCN_LH_TAB) K += 1ull << (VCN_LH_L - l[i]), l[i]--;
  for (unsigned i = 0; i < m; i++) len[idx[i]] = (uint8_t)l[i];
}

/* canonical codes, bit-reversed for least-significant-first reading; returns 0 if the lengths are valid */
static int vcn_lh_codes(const uint8_t len[256], uint16_t code[256]) {
  unsigned nl[VCN_LH_L + 1] = {0};
  uint64_t K = 0;
  for (unsigned s = 0; s < 256; s++) {
    if (len[s] > VCN_LH_L) return 1;
    if (len[s]) nl[len[s]]++, K += 1ull << (VCN_LH_L - len[s]);
  }
  if (K > VCN_LH_TAB) return 1;
  unsigned next[VCN_LH_L + 2], c = 0;
  for (unsigned b = 1; b <= VCN_LH_L; b++) c = (c + nl[b - 1]) << 1, next[b] = c;
  for (unsigned s = 0; s < 256; s++) {
    const unsigned b = len[s];
    if (!b) continue;
    unsigned v = next[b]++, r = 0;
    for (unsigned k = 0; k < b; k++) r = (r << 1) | ((v >> k) & 1u);
    code[s] = (uint16_t)r;
  }
  return 0;
}

#define VCN_LH_SCAP ((size_t)VCN_VOXELS / VCN_LH_NS * VCN_LH_L / 8 + 16)

VCN_CLONES_BMI static vcn_status vcn_decode_llhuf(const uint8_t *p, size_t n, uint8_t *v) {
  const uint8_t *end = p + n;
  if (n < VCN_LF_HDR) return VCN_ERR_CORRUPT;
  int32_t wq[VCN_LF_TAPS];
  for (unsigned i = 0; i < VCN_LF_TAPS; i++) wq[i] = (int32_t)(int16_t)vcn_rd16(p + 2 * i);
  p += 2 * VCN_LF_TAPS;
  const int32_t bias = (int32_t)vcn_rd32(p);
  p += 4;
  if (bias < -(1 << 22) || bias > (1 << 22)) return VCN_ERR_CORRUPT;
  uint16_t thr[VCN_LF_NCTX - 1];
  for (unsigned i = 0; i < VCN_LF_NCTX - 1; i++) thr[i] = (uint16_t)vcn_rd16(p + 2 * i);
  p += 2 * (VCN_LF_NCTX - 1);
  /* the code tables, then the last three slices widened to 16 bits and a flat row */
  uint16_t *blk = (uint16_t *)malloc(sizeof(uint16_t) * (VCN_LF_NCTX * VCN_LH_TAB + (3 * VCN_DIM + 1) * VCN_LH_RS));
  if (!blk) return VCN_ERR_NOMEM;
  uint16_t *restrict T = blk;
  for (unsigned t = 0; t < VCN_LF_NCTX; t++) {
    uint8_t len[256];
    uint16_t code[256];
    if (end - p < 2) goto bad;
    const unsigned m = vcn_rd16(p);
    p += 2;
    if (m > 256 || (size_t)(end - p) < (m + 1) / 2) goto bad;
    memset(len, 0, sizeof len);
    for (unsigned s = 0; s < m; s++) len[s] = (uint8_t)((p[s >> 1] >> (4 * (s & 1u))) & 15u);
    p += (m + 1) / 2;
    if (vcn_lh_codes(len, code)) goto bad;
    uint16_t *tt = T + (size_t)t * VCN_LH_TAB;
    for (unsigned i = 0; i < VCN_LH_TAB; i++) tt[i] = (uint16_t)(0x8000u | VCN_LH_L << 8); /* not a code */
    for (unsigned s = 0; s < 256; s++)
      if (len[s])
        for (unsigned j = code[s]; j < VCN_LH_TAB; j += 1u << len[s]) tt[j] = (uint16_t)(s | (unsigned)len[s] << 8);
  }
  if ((size_t)(end - p) < 4 * (VCN_LH_NS - 1)) goto bad;
  {
    const uint8_t *sp[VCN_LH_NS], *se[VCN_LH_NS];
    const uint8_t *q = p + 4 * (VCN_LH_NS - 1);
    for (unsigned k = 0; k < VCN_LH_NS; k++) {
      const size_t sz = k + 1 < VCN_LH_NS ? vcn_rd32(p + 4 * k) : (size_t)(end - q);
      if (sz > (size_t)(end - q)) goto bad;
      sp[k] = q, se[k] = q + sz, q += sz;
    }
    const uint8_t *p0 = sp[0], *p1 = sp[1], *p2 = sp[2], *p3 = sp[3];
    const uint8_t *e0 = se[0], *e1 = se[1], *e2 = se[2], *e3 = se[3];
    uint64_t f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    unsigned c0 = 0, c1 = 0, c2 = 0, c3 = 0, v0 = 0, v1 = 0, v2 = 0, v3 = 0, acc = 0;
    /* refill to at least 56 bits (5 codes of at most 11); near the end byte by byte, zeros past it
     * counted in v */
#define VCN_LH_FILL(k)                                                                                     \
  do {                                                                                                     \
    if (e##k - p##k >= 8) {                                                                                \
      f##k |= vcn_rd64(p##k) << c##k;                                                                      \
      p##k += (63u - c##k) >> 3;                                                                           \
      c##k |= 56u;                                                                                         \
    } else                                                                                                 \
      for (; c##k <= 56u; c##k += 8) {                                                                     \
        if (p##k < e##k) f##k |= (uint64_t)*p##k++ << c##k;                                                \
        else v##k++;                                                                                       \
      }                                                                                                    \
  } while (0)
#define VCN_LH_GET(k, x)                                                                                   \
  do {                                                                                                     \
    const unsigned e_ = T[(size_t)ctx[x] << VCN_LH_L | (unsigned)(f##k & (VCN_LH_TAB - 1u))];              \
    const unsigned b_ = (e_ >> 8) & 15u;                                                                   \
    sym[x] = (uint8_t)e_, acc |= e_, f##k >>= b_, c##k -= b_;                                              \
  } while (0)
    uint8_t pred[VCN_DIM], ctx[VCN_DIM], sym[VCN_DIM];
    /* the last three slices widened to 16 bits, and a flat row */
    int16_t *restrict ring = (int16_t *)(blk + VCN_LF_NCTX * VCN_LH_TAB), *flat = ring + 3 * VCN_DIM * VCN_LH_RS;
    for (unsigned x = 0; x < VCN_LH_RS; x++) flat[x] = 128;
    for (unsigned z = 0; z < VCN_DIM; z++)
      for (unsigned y = 0; y < VCN_DIM; y++) {
        const size_t SL = (size_t)VCN_DIM * VCN_LH_RS;
        const int16_t *r6[6];
        int16_t *s0 = ring + (z % 3) * SL;
        vcn_lf_src16(s0, ring + ((z + 2) % 3) * SL, ring + ((z + 1) % 3) * SL, flat, z, y, r6);
        vcn_lf_pred16(r6, wq, bias, thr, pred, ctx);
        for (unsigned x0 = 0; x0 < VCN_DIM; x0 += 4 * VCN_LH_NS) {
          VCN_LH_FILL(0); VCN_LH_FILL(1); VCN_LH_FILL(2); VCN_LH_FILL(3);
          for (unsigned x = x0; x < x0 + 4 * VCN_LH_NS; x += 4) {
            VCN_LH_GET(0, x); VCN_LH_GET(1, x + 1); VCN_LH_GET(2, x + 2); VCN_LH_GET(3, x + 3);
          }
        }
        uint8_t *row = v + ((size_t)z * VCN_DIM + y) * VCN_DIM;
        for (unsigned x = 0; x < VCN_DIM; x++)
          row[x] = (uint8_t)(pred[x] + ((sym[x] >> 1) ^ (uint8_t)(0u - (sym[x] & 1u))));
        vcn_lf_widen(row, s0 + y * VCN_LH_RS);
      }
#undef VCN_LH_FILL
#undef VCN_LH_GET
    const uint8_t *pe[VCN_LH_NS] = {p0, p1, p2, p3};
    const uint64_t buf[VCN_LH_NS] = {f0, f1, f2, f3};
    const unsigned cnt[VCN_LH_NS] = {c0, c1, c2, c3}, virt[VCN_LH_NS] = {v0, v1, v2, v3};
    for (unsigned k = 0; k < VCN_LH_NS; k++) sp[k] = pe[k];
    free(blk);
    if (acc & 0x8000u) return VCN_ERR_CORRUPT;
    /* every stream used up to its last byte, not past it: the bits left are the final byte's zero
     * padding (and the zeros read past the end) */
    for (unsigned k = 0; k < VCN_LH_NS; k++) {
      const int real = (int)cnt[k] - 8 * (int)virt[k];
      if (sp[k] != se[k] || real < 0 || real >= 8) return VCN_ERR_CORRUPT;
      if (cnt[k] < 64 && (buf[k] & ((1ull << cnt[k]) - 1u))) return VCN_ERR_CORRUPT;
      if (cnt[k] >= 64 && buf[k]) return VCN_ERR_CORRUPT;
    }
    return VCN_OK;
  }
bad:
  free(blk);
  return VCN_ERR_CORRUPT;
}

VCN_CLONES_BMI static vcn_status vcn_encode_llhuf(const uint8_t *v, uint8_t *out, size_t cap, size_t *out_n) {
  int32_t wq[VCN_LF_TAPS], bias;
  uint16_t thr[VCN_LF_NCTX - 1];
  vcn_lf_fit(v, 16, wq, &bias, thr);
  uint16_t *restrict ts = (uint16_t *)malloc(sizeof(uint16_t) * VCN_VOXELS);
  int16_t *ring = (int16_t *)malloc(sizeof(int16_t) * (3 * VCN_DIM + 1) * VCN_LH_RS);
  if (!ts || !ring) {
    free(ts), free(ring);
    return VCN_ERR_NOMEM;
  }
  /* counts by class << 8 | symbol in four tables (voxel x mod 4), 64 words more than their size apart
   * so that the counters of neighbouring voxels never sit at the same offset in a 4 KiB page */
  enum { CW = VCN_LF_NCTX * 256 + 64 };
  uint32_t cnt[4 * CW];
  memset(cnt, 0, sizeof cnt);
  {
    const size_t SL = (size_t)VCN_DIM * VCN_LH_RS;
    int16_t *flat = ring + 3 * SL;
    for (unsigned x = 0; x < VCN_LH_RS; x++) flat[x] = 128;
    uint8_t pred[VCN_DIM], cr[VCN_DIM];
    uint32_t *c0 = cnt, *c1 = cnt + CW, *c2 = cnt + 2 * CW, *c3 = cnt + 3 * CW;
    for (unsigned z = 0; z < VCN_DIM; z++) {
      int16_t *s0 = ring + (z % 3) * SL;
      for (unsigned y = 0; y < VCN_DIM; y++) vcn_lf_widen(v + ((size_t)z * VCN_DIM + y) * VCN_DIM, s0 + y * VCN_LH_RS);
      for (unsigned y = 0; y < VCN_DIM; y++) {
        const size_t r0 = ((size_t)z * VCN_DIM + y) * VCN_DIM;
        const int16_t *r6[6];
        vcn_lf_src16(s0, ring + ((z + 2) % 3) * SL, ring + ((z + 1) % 3) * SL, flat, z, y, r6);
        vcn_lf_pred16(r6, wq, bias, thr, pred, cr);
        uint16_t *t = ts + r0;
        for (unsigned x = 0; x < VCN_DIM; x++) { /* zigzag mod 256: (2r) ^ -(r >> 7) */
          const uint8_t r = (uint8_t)(v[r0 + x] - pred[x]);
          t[x] = (uint16_t)(cr[x] << 8 | (uint8_t)((uint8_t)(r << 1) ^ (uint8_t)(0u - (r >> 7))));
        }
        for (unsigned x = 0; x < VCN_DIM; x += 4) c0[t[x]]++, c1[t[x + 1]]++, c2[t[x + 2]]++, c3[t[x + 3]]++;
      }
    }
  }
  free(ring);
  uint8_t len[VCN_LF_NCTX][256];
  uint32_t ct[VCN_LF_NCTX * 256]; /* code | length << 16 by (class << 8 | symbol) */
  for (unsigned t = 0; t < VCN_LF_NCTX; t++) {
    uint32_t c[256];
    uint16_t code[256];
    for (unsigned s = 0; s < 256; s++) {
      const unsigned j = t * 256 + s;
      c[s] = cnt[j] + cnt[CW + j] + cnt[2 * CW + j] + cnt[3 * CW + j];
    }
    vcn_lh_lengths(c, len[t]);
    (void)vcn_lh_codes(len[t], code);
    for (unsigned s = 0; s < 256; s++) ct[t * 256 + s] = len[t][s] ? (uint32_t)code[s] | (uint32_t)len[t][s] << 16 : 0;
  }
  uint8_t hdr[VCN_LF_HDR + VCN_LF_NCTX * (2 + 128) + 4 * VCN_LH_NS];
  size_t pos = 0;
  for (unsigned i = 0; i < VCN_LF_TAPS; i++) vcn_wr16(hdr + pos, (uint32_t)(uint16_t)(int16_t)wq[i]), pos += 2;
  vcn_wr32(hdr + pos, (uint32_t)bias), pos += 4;
  for (unsigned i = 0; i < VCN_LF_NCTX - 1; i++) vcn_wr16(hdr + pos, thr[i]), pos += 2;
  for (unsigned t = 0; t < VCN_LF_NCTX; t++) {
    unsigned m = 256;
    while (m > 0 && len[t][m - 1] == 0) m--;
    vcn_wr16(hdr + pos, m), pos += 2;
    for (unsigned s = 0; s < m; s += 2)
      hdr[pos++] = (uint8_t)(len[t][s] | (s + 1 < m ? len[t][s + 1] : 0u) << 4);
  }
  /* the streams, each into its own buffer (8 bytes of slack for the 64-bit stores) */
  const size_t scap = VCN_LH_SCAP;
  uint8_t *restrict sb = (uint8_t *)malloc(scap * VCN_LH_NS);
  if (!sb) {
    free(ts);
    return VCN_ERR_NOMEM;
  }
  uint8_t *o[VCN_LH_NS];
  uint64_t bb[VCN_LH_NS];
  unsigned nb[VCN_LH_NS];
  for (unsigned k = 0; k < VCN_LH_NS; k++) o[k] = sb + k * scap, bb[k] = 0, nb[k] = 0;
  {
    /* four codes per stream (at most 44 bits) between flushes, the streams' state in registers */
    uint8_t *o0 = o[0], *o1 = o[1], *o2 = o[2], *o3 = o[3];
    uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    unsigned n0 = 0, n1 = 0, n2 = 0, n3 = 0;
#define VCN_LH_PUT(k, i)                                                                                   \
  do {                                                                                                     \
    const uint32_t e_ = ct[t[i]];                                                                          \
    b##k |= (uint64_t)(e_ & 0xFFFFu) << n##k;                                                              \
    n##k += e_ >> 16;                                                                                      \
  } while (0)
#define VCN_LH_FLUSH(k)                                                                                    \
  do {                                                                                                     \
    vcn_wr64(o##k, b##k);                                                                                  \
    o##k += n##k >> 3, b##k >>= n##k & ~7u, n##k &= 7u;                                                    \
  } while (0)
    for (size_t i0 = 0; i0 < VCN_VOXELS; i0 += 4 * VCN_LH_NS) {
      const uint16_t *t = ts + i0;
      for (unsigned j = 0; j < 16; j += 4) {
        VCN_LH_PUT(0, j); VCN_LH_PUT(1, j + 1); VCN_LH_PUT(2, j + 2); VCN_LH_PUT(3, j + 3);
      }
      VCN_LH_FLUSH(0); VCN_LH_FLUSH(1); VCN_LH_FLUSH(2); VCN_LH_FLUSH(3);
    }
#undef VCN_LH_PUT
#undef VCN_LH_FLUSH
    o[0] = o0, o[1] = o1, o[2] = o2, o[3] = o3;
    bb[0] = b0, bb[1] = b1, bb[2] = b2, bb[3] = b3;
    nb[0] = n0, nb[1] = n1, nb[2] = n2, nb[3] = n3;
  }
  free(ts);
  size_t sz[VCN_LH_NS], total = pos + 4 * (VCN_LH_NS - 1);
  for (unsigned k = 0; k < VCN_LH_NS; k++) {
    if (nb[k]) vcn_wr64(o[k], bb[k]), o[k]++;
    sz[k] = (size_t)(o[k] - (sb + k * scap));
    total += sz[k];
  }
  vcn_status st = VCN_OK;
  if (total > cap) st = VCN_ERR_CAP;
  else {
    for (unsigned k = 0; k + 1 < VCN_LH_NS; k++) vcn_wr32(hdr + pos, (uint32_t)sz[k]), pos += 4;
    memcpy(out, hdr, pos);
    uint8_t *w = out + pos;
    for (unsigned k = 0; k < VCN_LH_NS; k++) memcpy(w, sb + k * scap, sz[k]), w += sz[k];
    *out_n = total;
  }
  free(sb);
  return st;
}

/* ======================================================================== */
/*              context-modelled lossless mode (mode 6, 22 Oct 2026)        */
/* ======================================================================== */
/* Mode 6, the ratio-first lossless mode (opt in: vcn_opts.ll_fast = 3), stream version 3 (22 Oct 2026). The
 * payload's first byte is the stream version (VCN_LC_VERSION); the decoder reads version 3 only and refuses
 * any other with VCN_ERR_VERSION, so a later format change is detected instead of misread. The earlier
 * drafts (versions 1 to 3 without this byte) were never merged; such a stream starts with r0 = 3, so it reads
 * as version 3 and is then refused by the radius check (its K = 16 lands on r2). A golden stream pins
 * version 3 (tests/fixtures/m6-golden.txt).
 * Each voxel is predicted from a causal box: in-row W, WW, WWW (r0 = 3), the rows y-3..y-1 of the same
 * slice within x +-3, the 7 x 7 window of slice z-1 (r1 = 3) and the 3 x 3 window of slice z-2 (r2 = 1),
 * 82 candidate taps of which the encoder keeps 40 by greedy forward selection on the normal equations
 * (Schur complements), then refits them by least squares (exact integer sums) and rounds the weights to
 * 1/256 keeping their sum (int16 in the stream; slice 0 has its own in-slice set). The prediction is
 * computed modulo 2^16 in 1/256 steps (as vc-delta3d's LPC does), so the decoder sums 16-bit lanes with
 * 8-bit multiply-adds; the predicted value is the rounded prediction modulo 256. Outside the chunk the
 * source is padded row by row (x replicate, y from the previous slice's first row or the last row, z
 * before 0 from slice 0), so every voxel uses the same predictor. The residual's class comes from an
 * activity sum of the residual magnitudes of 10 decoded neighbours in earlier rows and slices
 * (2N + 2U + NW + NE + UW + UE + UN + US + NN + UU), cut into K = 16 classes at thresholds; each class has
 * a bias (CALIC-style bias cancellation, static per chunk) added to the prediction, and its own static
 * rANS table (12-bit, LEB128 in the stream). The residual's sign is flipped when the prediction was
 * rounded up. The encoder sets the thresholds at the quantiles of the activity of the bias-free residual
 * and the biases at the mean bias-free residual per class; the classes it codes with (version 3) use the
 * magnitudes of the coded, biased residuals, |residual| = (symbol + 1) >> 1, so the decoder has them as
 * soon as it has a row's symbols. A symbol's table depends only on earlier rows, so the decoder first
 * decodes the symbols of a group of rows (32 interleaved rANS states, x mod 32), then rebuilds the voxels. Rows go
 * on a wavefront: row (z, y) runs at step y + d z with d = max(r1, r2) + 1, and the rows of one step (up
 * to 32) are rebuilt together as 16-bit lanes; a lane's serial part is the in-row taps and the rounding.
 * Integer only: the AVX2, NEON and plain C paths give the same bytes. Payload: version, r0, r1, r2, K, table bits; two
 * weight sets (int16 per tap, uint16 bias); K-1 thresholds; K biases (int16, 1/256); K tables; 32
 * states; words. */
#define VCN_LC_VERSION 3u /* the mode 6 stream version this encoder writes and the only one its decoder reads */
#define VCN_LC_STRIDE 128 /* fit on one voxel of each row */
#define VCN_LC_NSEL 40    /* taps kept of the 82 candidates */
#define VCN_LC_D NG_X
#define VCN_LC_Y NG_Y
#define VCN_LC_Z NG_Z
#define VCN_LC_XP 4                                                 /* x pad each side */
#define VCN_LC_YP 4                                                 /* y pad each side */
#define VCN_LC_ZP 3                                                 /* z pad before */
#define VCN_LC_RP (VCN_LC_D + 2 * VCN_LC_XP)                        /* row pitch 136 */
#define VCN_LC_SP (VCN_LC_RP * (VCN_LC_Y + 2 * VCN_LC_YP))          /* slice pitch */
#define VCN_LC_NB ((size_t)VCN_LC_SP * (VCN_LC_Z + VCN_LC_ZP) + 64)         /* (+ slack for 32-byte loads) */
#define VCN_LC_IDX(z, y, x) ((size_t)((z) + VCN_LC_ZP) * VCN_LC_SP + (size_t)((y) + VCN_LC_YP) * VCN_LC_RP + (size_t)((x) + VCN_LC_XP))
#define VCN_LC_MAXT 160
#define VCN_LC_MAXF (2 * VCN_LC_MAXT + 1) /* fit-only: taps, tap×activity, activity */
#define VCN_LC_MAXK 128 /* up to four independent entropy banks of32 classes */
#define VCN_LC_TB 12            /* table bits written by the encoder */
#define VCN_LC_RL (1u << 15)    /* rANS state lower bound; states stay below 2^31 */
#define VCN_LC_NL 32            /* rows rebuilt together */
#define VCN_LC_NS 32            /* rANS states (symbol x of a row uses state x mod 32) */
typedef struct {
  int r[3];             /* radius for dz = 0, -1, -2 (-1: none) */
  int nt, nrow;         /* taps; of them the first nrow are in-row (dx = -1..-r0) */
  int dz[VCN_LC_MAXT], dy[VCN_LC_MAXT], dx[VCN_LC_MAXT];
  ptrdiff_t off[VCN_LC_MAXT];
} vcn_lc_taps;
static void vcn_lc_build(vcn_lc_taps *t, int r0, int r1, int r2) {
  t->r[0] = r0, t->r[1] = r1, t->r[2] = r2;
  int n = 0;
  for (int k = 1; k <= r0; k++) t->dz[n] = 0, t->dy[n] = 0, t->dx[n] = -k, n++;
  t->nrow = n;
  for (int dz = -2; dz <= 0; dz++) {
    int r = t->r[-dz];
    if (r < 0) continue;
    for (int dy = -r; dy <= (dz ? r : -1); dy++)
      for (int dx = -r; dx <= r; dx++) t->dz[n] = dz, t->dy[n] = dy, t->dx[n] = dx, n++;
  }
  t->nt = n;
  for (int i = 0; i < n; i++) t->off[i] = (ptrdiff_t)t->dz[i] * VCN_LC_SP + (ptrdiff_t)t->dy[i] * VCN_LC_RP + t->dx[i];
}
/* pads, row by row (valid in raster order and in the decoder's wavefront order) */
static inline void vcn_lc_row_begin(uint8_t *b, int z, int y, uint8_t flat) {
  if (y == 0) /* top pad rows of slice z := row 0 of slice z-1 with its pads, or flat */
    for (int yy = -VCN_LC_YP; yy < 0; yy++) {
      uint8_t *r = b + VCN_LC_IDX(z, yy, -VCN_LC_XP);
      if (z > 0) memcpy(r, b + VCN_LC_IDX(z - 1, 0, -VCN_LC_XP), VCN_LC_RP);
      else memset(r, flat, VCN_LC_RP);
    }
  uint8_t *r = b + VCN_LC_IDX(z, y, 0);
  memset(r - VCN_LC_XP, r[-VCN_LC_RP], VCN_LC_XP); /* left pad := the row above's first voxel */
}
static inline void vcn_lc_row_end(uint8_t *b, int z, int y) {
  uint8_t *r = b + VCN_LC_IDX(z, y, 0);
  memset(r - VCN_LC_XP, r[0], VCN_LC_XP);
  memset(r + VCN_LC_D, r[VCN_LC_D - 1], VCN_LC_XP);
  int y1 = y, y2 = y;
  if (y == VCN_LC_Y - 1) {
    for (int yy = VCN_LC_Y; yy < VCN_LC_Y + VCN_LC_YP; yy++) memcpy(b + VCN_LC_IDX(z, yy, -VCN_LC_XP), r - VCN_LC_XP, VCN_LC_RP);
    y2 = VCN_LC_Y + VCN_LC_YP - 1;
  }
  if (y == 0) y1 = -VCN_LC_YP;
  if (z == 0) /* slices before 0 := slice 0 */
    for (int k = 1; k <= VCN_LC_ZP; k++)
      for (int yy = y1; yy <= y2; yy++) memcpy(b + VCN_LC_IDX(-k, yy, -VCN_LC_XP), b + VCN_LC_IDX(0, yy, -VCN_LC_XP), VCN_LC_RP);
}
/* context activity (all but W, WW) from |e| buffer */
static void vcn_lc_apre(const uint8_t *e, size_t r0i, uint16_t *restrict a);
static void ng_apre_scalar(const uint8_t *e, size_t r0i, uint16_t *restrict a) {
  const uint8_t *restrict n = e + r0i - VCN_LC_RP, *restrict nn = e + r0i - 2 * VCN_LC_RP, *restrict u = e + r0i - VCN_LC_SP,
                          *restrict uu = e + r0i - 2 * VCN_LC_SP;
  for (int x = 0; x < VCN_LC_D; x++)
    a[x] = (uint16_t)(2 * n[x] + 2 * u[x] + n[x - 1] + n[x + 1] + u[x - 1] + u[x + 1] + u[x - VCN_LC_RP] + u[x + VCN_LC_RP] +
                      nn[x] + uu[x]);
}
static void vcn_lc_norm(const uint32_t *cnt, uint16_t *freq, int tb) {
  const uint32_t M = 1u << tb;
  uint64_t tot = 0;
  unsigned maxs = 0;
  for (int s = 0; s < 256; s++) { tot += cnt[s]; if (cnt[s] > cnt[maxs]) maxs = s; }
  memset(freq, 0, 512);
  if (!tot) { freq[0] = (uint16_t)M; return; }
  int32_t sum = 0;
  for (int s = 0; s < 256; s++)
    if (cnt[s]) {
      uint32_t f = (uint32_t)(((uint64_t)cnt[s] * M + tot / 2) / tot);
      freq[s] = (uint16_t)(f < 1 ? 1 : f), sum += freq[s];
    }
  while (sum != (int32_t)M) {
    if (sum < (int32_t)M) { freq[maxs] += (uint16_t)(M - sum); sum = M; break; }
    unsigned best = maxs;
    for (int s = 0; s < 256; s++) if (freq[s] > freq[best]) best = s;
    int32_t d = sum - (int32_t)M, room = freq[best] - 1, t = d < room ? d : room;
    if (t < 1) return; /* cannot happen with <= 256 symbols and M >= 1024 */
    freq[best] = (uint16_t)(freq[best] - t), sum -= t;
  }
}
/* integer weights q near the least-squares w: moves of one step on one weight, or up on one and down on
   another, while they lower the added error (q - w)' C (q - w) (C the centred covariance) */
static void vcn_lc_qopt(const double *C, const double *w, int32_t *q, int n) {
  double g[VCN_LC_MAXF]; /* g = C (q - w), in steps of 1/256 */
  for (int i = 0; i < n; i++) {
    double s = 0;
    for (int k = 0; k < n; k++) s += C[i * n + k] * (q[k] - w[k] * 256.0);
    g[i] = s;
  }
  for (int it = 0; it < 64; it++) {
    double best = -1e-9 * (C[0] + 1e-30);
    int bi = -1, bj = -1, bd = 0;
    for (int i = 0; i < n; i++)
      for (int d = -1; d <= 1; d += 2) {
        const double f = C[i * n + i] + 2 * d * g[i];
        if (f < best) best = f, bi = i, bj = -1, bd = d;
        for (int j = i + 1; j < n; j++) {
          const double f2 = f + C[j * n + j] - 2 * d * g[j] - 2 * C[i * n + j];
          if (f2 < best) best = f2, bi = i, bj = j, bd = d;
        }
      }
    if (bi < 0) break;
    q[bi] += bd;
    for (int k = 0; k < n; k++) g[k] += bd * C[k * n + bi];
    if (bj >= 0) {
      q[bj] -= bd;
      for (int k = 0; k < n; k++) g[k] -= bd * C[k * n + bj];
    }
  }
}
/* G[a][k] += sum over the batch of F[a][s] F[k][s], for the 4 x 4 tiles with a-tile >= k-tile (G is
   ncp x ncp, row a; entries with a >= k are valid); F is ncp columns of VCN_LC_SB int16 at pitch VCN_LC_SBP */
#define VCN_LC_SB 1024
#define VCN_LC_SBP (VCN_LC_SB + 32) /* column pitch: one cache line more, so the columns do not share cache sets */
#include "ng_fit_cache.h"
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static int64_t ng_fit_dot_avx2(const int16_t*a,const int16_t*b,size_t n){
  int64_t total=0;
  for(size_t start=0;start<n;start+=VCN_LC_SB){
    __m256i sum=_mm256_setzero_si256();
    for(size_t i=start;i<start+VCN_LC_SB;i+=16)
      sum=_mm256_add_epi32(sum,_mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)(a+i)),_mm256_loadu_si256((const __m256i*)(b+i))));
    int32_t lanes[8];_mm256_storeu_si256((__m256i*)lanes,sum);
    for(int k=0;k<8;k++)total+=lanes[k];
  }return total;
}
#elif VCN_HAVE_NEON
static int64_t ng_fit_dot_neon(const int16_t*a,const int16_t*b,size_t n){
  int64_t total=0;
  for(size_t start=0;start<n;start+=VCN_LC_SB){
    int32x4_t sum=vdupq_n_s32(0);
    for(size_t i=start;i<start+VCN_LC_SB;i+=8){
      int16x8_t x=vld1q_s16(a+i),y=vld1q_s16(b+i);
      sum=vmlal_high_s16(vmlal_s16(sum,vget_low_s16(x),vget_low_s16(y)),x,y);
    }total+=vaddvq_s32(sum);
  }return total;
}
#endif
static int64_t ng_fit_dot(const int16_t*a,const int16_t*b,size_t n,int simd){
#if VCN_HAVE_AVX2
  if(simd&&vcn_use_avx2())return ng_fit_dot_avx2(a,b,n);
#elif VCN_HAVE_NEON
  if(simd)return ng_fit_dot_neon(a,b,n);
#endif
  return ng_fit_dot_c(a,b,n);
}
static void ng_fit_store_batch(ng_fit_bank_cache*c,const int16_t*F){
  for(int k=0;k<c->ncol;k++)memcpy(c->columns+(size_t)k*c->capacity+c->stored,F+(size_t)k*VCN_LC_SBP,VCN_LC_SB*sizeof(int16_t));
  c->stored+=VCN_LC_SB;
}
VCN_MAYBE_UNUSED static void vcn_lc_gram_c(const int16_t *F, int ncp, int64_t *G) {
  for (int a = 0; a < ncp; a++)
    for (int k = 0; k <= (a | 3); k++) {
      const int16_t *restrict fa = F + (size_t)a * VCN_LC_SBP, *restrict fk = F + (size_t)k * VCN_LC_SBP;
      int32_t s = 0;
      for (int i = 0; i < VCN_LC_SB; i++) s += fa[i] * fk[i];
      G[(size_t)a * ncp + k] += s;
    }
}
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static int ng_gram_sum256(__m256i v){
  __m128i h=_mm_add_epi32(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
  h=_mm_add_epi32(h,_mm_shuffle_epi32(h,0x4e));return _mm_cvtsi128_si32(_mm_add_epi32(h,_mm_shuffle_epi32(h,0xb1)));
}
#define NG_GATTR __attribute__((target("avx2")))
#define NG_GNAME vcn_lc_gram_avx2
#define NG_GROWS 2
#define NG_GV __m256i
#define NG_GZERO _mm256_setzero_si256
#define NG_GSTEP 16
#define NG_GLOAD _mm256_loadu_si256
#define NG_GADD _mm256_add_epi32
#define NG_GMADD _mm256_madd_epi16
#define NG_GSUM ng_gram_sum256
#include "ng_gram_vector.h"
#define NG_GATTR __attribute__((target("avx512f,avx512bw")))
#define NG_GNAME ng_gram_wide
#define NG_GROWS 4
#define NG_GV __m512i
#define NG_GZERO _mm512_setzero_si512
#define NG_GSTEP 32
#define NG_GLOAD _mm512_loadu_si512
#define NG_GADD _mm512_add_epi32
#define NG_GMADD _mm512_madd_epi16
#define NG_GSUM _mm512_reduce_add_epi32
#include "ng_gram_vector.h"
static void ng_gram_dispatch(const int16_t*F,int ncp,int64_t*G){
if(!ng_runtime_only&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw"))ng_gram_wide(F,ncp,G);
  else if(vcn_use_avx2())vcn_lc_gram_avx2(F,ncp,G);else vcn_lc_gram_c(F,ncp,G);
}
#define vcn_lc_gram(F,ncp,G) ng_gram_dispatch(F,ncp,G)
#elif VCN_HAVE_NEON
/* the same 4 x 4 tiles with widening multiply-adds into int32 lanes (each lane sums 256 products of at most
   255 * 255, the tile entry at most VCN_LC_SB of them: exact, as in the C form) */
static void vcn_lc_gram_neon(const int16_t *F, int ncp, int64_t *G) {
  for (int a0 = 0; a0 < ncp; a0 += 4)
    for (int k0 = 0; k0 <= a0; k0 += 4) {
      int32x4_t acc[4][4];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = vdupq_n_s32(0);
      const int16_t *fa = F + (size_t)a0 * VCN_LC_SBP, *fk = F + (size_t)k0 * VCN_LC_SBP;
      for (int v = 0; v < VCN_LC_SB; v += 8) {
        int16x8_t A4[4], K4[4];
        for (int i = 0; i < 4; i++) A4[i] = vld1q_s16(fa + (size_t)i * VCN_LC_SBP + v), K4[i] = vld1q_s16(fk + (size_t)i * VCN_LC_SBP + v);
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++)
            acc[i][j] = vmlal_high_s16(vmlal_s16(acc[i][j], vget_low_s16(A4[i]), vget_low_s16(K4[j])), A4[i], K4[j]);
      }
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) G[(size_t)(a0 + i) * ncp + k0 + j] += vaddvq_s32(acc[i][j]);
    }
}
#define vcn_lc_gram(F, ncp, G) vcn_lc_gram_neon(F, ncp, G)
#else
#define vcn_lc_gram(F, ncp, G) vcn_lc_gram_c(F, ncp, G)
#endif
/* least squares over the candidate taps (or those of slice 0), greedy selection of nsel, weights rounded
   to 1/256 with their sum kept; returns 1 on success, else the W predictor */
static int vcn_lc_select(double *C, double *cb, double *mf, int *id, int nt, int nsel, double tr) {
    if (nsel > 0 && nsel < nt) { /* greedy forward selection by partial covariances (Schur complements) */
      double *R = (double *)malloc(sizeof(double) * nt * nt), rb[VCN_LC_MAXF];
      memcpy(R, C, sizeof(double) * nt * nt), memcpy(rb, cb, sizeof(double) * nt);
      char used[VCN_LC_MAXF] = {0};
      int ns = 0;
      while (ns < nsel) {
        int best = -1;
        double bg = -1;
        for (int a = 0; a < nt; a++)
          if (!used[a] && R[a * nt + a] > 1e-9 * (tr / nt + 1e-12)) {
            const double g = rb[a] * rb[a] / R[a * nt + a];
            if (g > bg) bg = g, best = a;
          }
        if (best < 0) break;
        used[best] = 1, ns++;
        const double piv = R[best * nt + best];
        double col[VCN_LC_MAXF];
        for (int a = 0; a < nt; a++) col[a] = R[a * nt + best];
        const double rbb = rb[best];
        for (int a = 0; a < nt; a++) {
          const double fa = col[a] / piv;
          rb[a] -= fa * rbb;
          for (int k = 0; k < nt; k++) R[a * nt + k] -= fa * col[k];
        }
      }
      /* keep selected (in index order), rebuild the reduced system */
      int m = 0, keep[VCN_LC_MAXF];
      for (int a = 0; a < nt; a++) if (used[a]) keep[m++] = a;
      double *C2 = (double *)malloc(sizeof(double) * m * m), cb2[VCN_LC_MAXF], mf2[VCN_LC_MAXF];
      int id2[VCN_LC_MAXF];
      for (int a = 0; a < m; a++) {
        cb2[a] = cb[keep[a]], mf2[a] = mf[keep[a]], id2[a] = id[keep[a]];
        for (int k = 0; k < m; k++) C2[a * m + k] = C[keep[a] * nt + keep[k]];
      }
      memcpy(C, C2, sizeof(double) * m * m), memcpy(cb, cb2, sizeof(double) * m), memcpy(mf, mf2, sizeof(double) * m);
      memcpy(id, id2, sizeof(int) * m);
      nt = m;
      free(R), free(C2);
    }
  return nt;
}
static void vcn_lc_quantize(const double *C0, const double *cb, int32_t *q, int nt) {
      /* 1/256 steps; the rounding keeps the sum of the weights (the predictor's gain) */
      double sw = 0, fr[VCN_LC_MAXF];
      long sq = 0;
      for (int k = 0; k < nt; k++) {
        double v = cb[k] * 256.0;
        v = v < -32000 ? -32000 : v > 32000 ? 32000 : v;
        sw += v, q[k] = (int32_t)floor(v + 0.5), sq += q[k], fr[k] = v - q[k];
      }
      const long tgt = (long)floor(sw + 0.5);
      while (sq != tgt) {
        const int dir = tgt > sq ? 1 : -1;
        int bk = 0;
        for (int k = 1; k < nt; k++)
          if (dir * fr[k] > dir * fr[bk]) bk = k;
        q[bk] += dir, fr[bk] -= dir, sq += dir;
      }
      vcn_lc_qopt(C0, cb, q, nt);
}
/* Weighted counterpart of the native Gram kernel. Integer weights keep exact
 * sufficient statistics; ordinary fitting still uses the original SIMD kernel. */
static void vcn_lc_gram_weighted(const int16_t *F, int ncol, int ncp, int ns, const uint8_t *weights, int64_t *G) {
  for(int a=0;a<ncol;a++)for(int k=0;k<=a;k++){
    const int16_t *fa=F+(size_t)a*VCN_LC_SBP,*fk=F+(size_t)k*VCN_LC_SBP;
    int64_t sum=0;
    for(int i=0;i<ns;i++)sum+=(int64_t)fa[i]*fk[i]*weights[i];
    G[(size_t)a*ncp+k]+=sum;
  }
}
/* feature_kind: 0 none, 1 scalar, 2 scalar plus neighbor interactions,
 * 3 centered: fit target-feature using neighbor-feature contrasts.
 * weighting: 0 ordinary, 1 low activity (255-a), 2 high activity (a). */
/* Experimental nonlinear predictor inputs, shared by fit and causal decode.
 * Slice zero deliberately substitutes N for B (no unavailable previous slice). */
static inline int ng_context_input(const uint8_t*b,size_t i,int slice0,int k){
  int w=b[i-1],n=b[i-VCN_LC_RP],back=slice0?n:b[i-VCN_LC_SP];
  int hi=w>n?w:n,lo=w<n?w:n;if(back>hi)hi=back;if(back<lo)lo=back;
  switch(k){case 0:return abs(w-n);case 1:return abs(w-back);case 2:return abs(n-back);
    case 3:return hi-lo;case 4:return hi;case 5:return lo;
    case 6:return abs(w-b[i-2]);default:return abs(n-b[i-2*VCN_LC_RP]);}
}
static inline int ng_fit_input_value(const uint8_t*b,size_t i,const vcn_lc_taps*t,int id,int kind,unsigned activity,int slice0){
  if((kind==4||kind==5)&&id>=t->nt)return id<t->nt+8?ng_context_input(b,i,slice0,id-t->nt):(int)activity;
  return id<t->nt?(int)b[i+t->off[id]]-(kind==3?(int)activity:0)
    :kind==2&&id<2*t->nt?(int)((b[i+t->off[id-t->nt]]*activity+127)/255):(int)activity;
}
static int vcn_lc_fit_features_supported(const uint8_t *b, const vcn_lc_taps *t, int zlo, int zhi, int stride, int slice_only, int nsel, int32_t *wq, int32_t *bias, const int16_t *target, const uint8_t *feature, int feature_kind, int weighting, const uint8_t *sample_mask,ng_fit_session*session,const uint8_t *external_weights,const int32_t*support) {
  VCN_LC_TICK(20);
  uint64_t sample_counts[3];
  uint8_t* selected_samples=ng_nonzero_sparse?ng_fit_nonzero_sample_mask(sample_mask,
      VCN_LC_D,VCN_LC_Y,VCN_LC_Z,VCN_LC_XP,zlo,zhi,stride,
      session&&session->uniform_sparse,sample_counts):NULL;
  if(selected_samples){sample_mask=selected_samples;stride=1;}
  int id[VCN_LC_MAXF], nt = 0;
  for (int k = 0; k < t->nt; k++)
    if ((!slice_only || t->dz[k] == 0)&&(!support||support[k])) id[nt++] = k;
  if(feature_kind==2)for(int k=0;k<t->nt;k++)if(!slice_only||t->dz[k]==0)id[nt++]=t->nt+k;
  if(feature_kind==1||feature_kind==2)id[nt++]=feature_kind==2?2*t->nt:t->nt;
  if(feature_kind==4||feature_kind==5)for(int k=0;k<8+(feature_kind==5);k++)id[nt++]=t->nt+k;
  if(!nt){
    int64_t sum=0,count=0;
    for(int z=zlo;z<zhi;z++)for(int y=0;y<VCN_LC_Y;y++)for(int x=VCN_LC_XP+ng_fit_sample_phase(y,z,stride,0);x<VCN_LC_D;x+=stride){
      size_t j=((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x;if(sample_mask&&!sample_mask[j])continue;
      unsigned w=external_weights?external_weights[j]:1;
      sum+=(int64_t)w*(target?target[j]:b[VCN_LC_IDX(z,y,x)]);count+=w;
    }
    memset(wq,0,t->nt*sizeof(*wq));*bias=count?(int32_t)(uint16_t)(int32_t)floor(256.0*sum/(count*(1.+ng_fit_intercept_weight+ng_fit_zero_weight))+.5):0;free(selected_samples);return 1;
  }
  const int n = nt + 1;
  /* normal equations in exact integers: the samples' values as int16 columns (taps, the constant 1, the
     target), batches of VCN_LC_SB samples, every column pair's dot product in int32 per batch (at most
     VCN_LC_SB * 255 * 255 < 2^31) summed into int64; the sums equal the double ones bit for bit */
  const int ncol = nt + 2, ncp = (ncol + 3) & ~3, TC = nt + 1;
  ng_fit_bank_cache* cache=session&&feature_kind==1&&!weighting&&!target&&!sample_mask&&!external_weights?session->banks+slice_only:NULL;
  const int reuse=cache&&cache->count&&cache->nt==nt&&cache->ncp==ncp;
  double *A = (double *)calloc((size_t)n * n, sizeof(double)), *bv = (double *)calloc(n, sizeof(double));
  int16_t *F = (int16_t *)malloc((size_t)ncp * VCN_LC_SBP * sizeof(int16_t));
  int64_t *G = (int64_t *)calloc((size_t)ncp * ncp, sizeof(int64_t));
  memset(wq, 0, sizeof(int32_t) * (feature_kind>=4?t->nt+8+(feature_kind==5):feature_kind==2?2*t->nt+1:t->nt+(feature_kind==1)));
  *bias = 0;
  if (!A || !bv || !F || !G) {
    free(A), free(bv), free(F), free(G);
    free(selected_samples);
    wq[0] = 256;
    return 0;
  }
  uint8_t sample_weights[VCN_LC_SB];
  long cnt = 0;
  int ns = 0;
  memset(F, 0, (size_t)ncp * VCN_LC_SBP * sizeof(int16_t));
  if(cache&&!reuse){
    ng_fit_bank_clear(cache);cache->nt=nt;cache->ncol=ncol;cache->ncp=ncp;
    cache->capacity=(((size_t)(zhi-zlo)*VCN_LC_Y*((VCN_LC_D+stride-1)/stride)+VCN_LC_SB-1)/VCN_LC_SB)*VCN_LC_SB;
    cache->columns=malloc((size_t)ncol*cache->capacity*sizeof(int16_t));
    cache->positions=malloc(cache->capacity*sizeof(size_t));
    cache->gram=malloc((size_t)ncp*ncp*sizeof(int64_t));
    if(!cache->columns||!cache->positions||!cache->gram){ng_fit_bank_clear(cache);cache=NULL;}
  }
  if(reuse){
    memcpy(G,cache->gram,(size_t)ncp*ncp*sizeof(int64_t));cnt=(long)cache->count;
    int16_t*column=cache->columns+(size_t)(nt-1)*cache->capacity;
    for(size_t i=0;i<cache->count;i++)column[i]=feature[cache->positions[i]];
    memset(column+cache->count,0,(cache->stored-cache->count)*sizeof(int16_t));
    for(int k=0;k<ncol;k++){
      int a=k>nt-1?k:nt-1,c=k<nt-1?k:nt-1;
      G[(size_t)a*ncp+c]=ng_fit_dot(column,cache->columns+(size_t)k*cache->capacity,cache->stored,session->simd);
    }
  }else{
  for (int z = zlo; z < zhi; z++)
    for (int y = 0; y < VCN_LC_Y; y++) {
      for (int x = VCN_LC_XP + ng_fit_sample_phase(y,z,stride,session&&session->uniform_sparse); x < VCN_LC_D; x += stride) {
        if(sample_mask&&!sample_mask[((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x])continue;
        const size_t i = VCN_LC_IDX(z, y, x);
        if(cache)cache->positions[cnt]=((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x;
        unsigned activity=feature_kind==4?(z==0):feature?feature[((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x]:0;
        unsigned weight=external_weights?external_weights[((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x]:weighting==1?255-activity:weighting==2?activity:1;
        if(!weight)continue;
        sample_weights[ns]=(uint8_t)weight;
        for (int k = 0; k < nt; k++) {
          int value=ng_fit_input_value(b,i,t,id[k],feature_kind,activity,z==0);
          F[(size_t)k*VCN_LC_SBP+ns]=(int16_t)value;
        }
        F[(size_t)nt * VCN_LC_SBP + ns] = 1;
        F[(size_t)TC * VCN_LC_SBP + ns] = (target ? target[((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x] : b[i])-(feature_kind==3?(int)activity:0);
        cnt++;
        if (++ns == VCN_LC_SB) {
          if(weighting||external_weights)vcn_lc_gram_weighted(F,ncol,ncp,ns,sample_weights,G);else vcn_lc_gram(F,ncp,G);
          if(cache)ng_fit_store_batch(cache,F);
          ns=0;
        }
      }
    }
  if (ns) {
    for (int k = 0; k < ncp; k++) memset(F + (size_t)k * VCN_LC_SBP + ns, 0, (size_t)(VCN_LC_SB - ns) * sizeof(int16_t));
    if(weighting||external_weights)vcn_lc_gram_weighted(F,ncol,ncp,ns,sample_weights,G);else vcn_lc_gram(F,ncp,G);
    if(cache)ng_fit_store_batch(cache,F);
  }
  if(cache){cache->count=cnt;memcpy(cache->gram,G,(size_t)ncp*ncp*sizeof(int64_t));}
  }
  for (int a = 0; a < n; a++) {
    for (int k = 0; k <= a; k++) A[(size_t)a * n + k] = (double)G[(size_t)a * ncp + k];
    bv[a] = (double)G[(size_t)TC * ncp + a];
  }
  free(F), free(G);
  VCN_LC_TICK(21);
  int ok = 0;
  if (cnt >= 4 * n) {
    /* Adding lambda*bias^2 or lambda all-zero observations changes only the
     * intercept diagonal. Eliminate that intercept analytically; keep the
     * normalization by real sample mass, so both formulations are identical. */
    double *C = (double *)malloc(sizeof(double) * nt * nt), cb[VCN_LC_MAXF], mf[VCN_LC_MAXF], cn = A[(size_t)nt*n+nt], tr = 0;
    double intercept_ratio=1.+ng_fit_intercept_weight+ng_fit_zero_weight;
    double intercept_mass=cn*intercept_ratio, mt=bv[nt]/intercept_mass;
    for (int a = 0; a < nt; a++) mf[a] = A[(size_t)nt * n + a] / intercept_mass;
    for (int a = 0; a < nt; a++) {
      cb[a] = bv[a] / cn - mf[a] * mt * intercept_ratio;
      for (int k = 0; k < nt; k++) {
        const double s = a >= k ? A[(size_t)a * n + k] : A[(size_t)k * n + a];
        C[a * nt + k] = s / cn - mf[a] * mf[k] * intercept_ratio;
      }
      tr += C[a * nt + a];
    }
    for (int a = 0; a < nt; a++) C[a * nt + a] += ng_fit_ridge_scale * (1e-6 * tr / nt + 1e-9);
    VCN_LC_TICK(22);
    if(!support)nt = vcn_lc_select(C,cb,mf,id,nt,nsel,tr);
    VCN_LC_TICK(23);
    double *C0 = (double *)malloc(sizeof(double) * nt * nt);
    if (C0) memcpy(C0, C, sizeof(double) * nt * nt);
    if (C0 && vcn_solve(C, cb, nt) == 0) {
      VCN_LC_TICK(24);
      int32_t q[VCN_LC_MAXF];
      vcn_lc_quantize(C0, cb, q, nt);
      VCN_LC_TICK(25);
      double bb = mt;
      for (int k = 0; k < nt; k++) wq[id[k]] = q[k], bb -= q[k] / 256.0 * mf[k];
      *bias = (int32_t)(uint16_t)(int32_t)floor(bb * 256.0 + 0.5);
      ok = 1;
    }
    free(C), free(C0);
  }
  if (!ok) {
    if(support)memcpy(wq,support,t->nt*sizeof(*wq));
    else wq[0] = 256; /* W */
  }
  free(A), free(bv);
  free(selected_samples);
  return ok;
}
static int vcn_lc_fit_features(const uint8_t *b,const vcn_lc_taps*t,int zlo,int zhi,int stride,int slice_only,int nsel,int32_t*wq,int32_t*bias,const int16_t*target,const uint8_t*feature,int feature_kind,int weighting,const uint8_t*mask,ng_fit_session*session,const uint8_t*weights){
  return vcn_lc_fit_features_supported(b,t,zlo,zhi,stride,slice_only,nsel,wq,bias,target,feature,feature_kind,weighting,mask,session,weights,NULL);
}
static int vcn_lc_fit_feature(const uint8_t *b, const vcn_lc_taps *t, int zlo, int zhi, int stride, int slice_only, int nsel, int32_t *wq, int32_t *bias, const int16_t *target, const uint8_t *feature) {
  return vcn_lc_fit_features(b,t,zlo,zhi,stride,slice_only,nsel,wq,bias,target,feature,feature?1:0,0,NULL,NULL,NULL);
}
static int vcn_lc_fit_target(const uint8_t *b, const vcn_lc_taps *t, int zlo, int zhi, int stride, int slice_only, int nsel, int32_t *wq, int32_t *bias, const int16_t *target) {
  return vcn_lc_fit_feature(b,t,zlo,zhi,stride,slice_only,nsel,wq,bias,target,NULL);
}
#define VCN_LC_AMAX 2560 /* activity < 14 * 129 */
typedef struct {
  vcn_lc_taps t;
  int K, tb;
  int32_t w[2][VCN_LC_MAXT], bias[2];
  int has_context;int32_t context[2][8]; /* experimental model version4 */
  uint16_t thr[VCN_LC_MAXK];
  int32_t cb[VCN_LC_MAXK]; /* class bias, 1/256 */
  uint8_t lut[VCN_LC_AMAX];
} vcn_lc_par;
static void vcn_lc_mklut(vcn_lc_par *p) {
  for (int a = 0; a < VCN_LC_AMAX; a++) {
    int c = 0;
    for (int k = 0; k < p->K - 1; k++) c += a > p->thr[k];
    p->lut[a] = (uint8_t)c;
  }
}
/* Interpolate existing Q8 class biases at class interval midpoints. Use doubled
 * activity coordinates to retain half-integer centers. Repeated thresholds
 * create empty bins, skipped here; entropy classes themselves remain hard. */
static int32_t vcn_lc_soft_bias(const vcn_lc_par *p,unsigned activity) {
  int left=-1,lc=0,last=-1;unsigned position=2*activity;
  for(int k=0;k<p->K;k++){
    unsigned lo=k?p->thr[k-1]+1:0,hi=k+1<p->K?p->thr[k]:VCN_LC_AMAX-1;
    if(k==p->K-1&&k){unsigned width=k>1?p->thr[k-1]-p->thr[k-2]:p->thr[0]+1;hi=lo+(width?width:1)-1;}
    if(hi<lo)continue;
    int center=(int)(lo+hi);last=k;
    if(position<=(unsigned)center){
      if(left<0)return p->cb[k];
      int64_t numerator=(int64_t)p->cb[left]*(center-position)+(int64_t)p->cb[k]*(position-lc);
      int64_t den=center-lc;numerator+=den/2;
      return (int32_t)(numerator>=0?numerator/den:-((-numerator+den-1)/den));
    }
    left=k;lc=center;
  }
  return last<0?0:p->cb[last];
}
/* one row's classes from the activity, and A[x] += the class bias: kc[x] = lut[activity], as uint8 */
static void vcn_lc_cls_c(const uint8_t *e, size_t ri, const vcn_lc_par *p, uint16_t *restrict A, uint8_t *restrict kc) {
  _Alignas(32) uint16_t ap[VCN_LC_D];
  vcn_lc_apre(e, ri, ap);
  for (int x = 0; x < VCN_LC_D; x++) {
    const int c = kc[x] = p->lut[ap[x]];
    int32_t bias=p->cb[c];
    if(ng_soft_classes){
      int32_t soft=vcn_lc_soft_bias(p,ap[x]);
      if(ng_soft_classes==1)bias=soft;
      else {int32_t sum=bias+soft;bias=sum>=0?(sum+1)/2:-((-sum)/2);}
    }
    A[x] = (uint16_t)(A[x] + (uint16_t)bias);
  }
}
#if VCN_HAVE_AVX2
/* Shared exact activity expression for threshold and lookup classifiers. */
#define NG_AATTR __attribute__((target("avx2"),always_inline))
#define NG_AV __m256i
#define NG_ANAME vcn_lc_activity16
#define NG_ALOAD(Q) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Q)))
#define NG_AADD _mm256_add_epi16
#define NG_ASHIFT _mm256_slli_epi16
#include "ng_activity_vector.h"
#if (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 12)
#define NG_AATTR __attribute__((target("avx512f,avx512bw"),always_inline))
#define NG_AV __m512i
#define NG_ANAME vcn_lc_activity32
#define NG_ALOAD(Q) _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(Q)))
#define NG_AADD _mm512_add_epi16
#define NG_ASHIFT _mm512_slli_epi16
#include "ng_activity_vector.h"
#endif
/* the same for K <= 16: the class as the count of thresholds below the activity, the bias by byte shuffles */
__attribute__((target("avx2"),always_inline)) static inline void vcn_lc_cls_avx2_impl(const uint8_t *e, size_t ri, const vcn_lc_par *p, uint16_t *restrict A,
                                                            uint8_t *restrict kc,const vcn_lc_par* second,uint8_t*restrict kc2,int pairbytes,uint16_t*activity) {
  const uint8_t *n = e + ri - VCN_LC_RP, *nn = e + ri - 2 * VCN_LC_RP, *u = e + ri - VCN_LC_SP, *uu = e + ri - 2 * VCN_LC_SP;
  __m256i th[VCN_LC_MAXK];
  const int nk = p->K - 1;
  for (int k = 0; k < nk; k++) th[k] = _mm256_set1_epi16((short)p->thr[k]);
  __m256i th2[VCN_LC_MAXK];
  if(second)for(int k=0;k<second->K-1;k++)th2[k]=_mm256_set1_epi16((short)second->thr[k]);
  uint8_t lo[16] = {0}, hi[16] = {0};
  for (int k = 0; k < p->K; k++) lo[k] = (uint8_t)p->cb[k], hi[k] = (uint8_t)((uint32_t)p->cb[k] >> 8);
  const __m256i tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lo)),
                thi = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)hi));
#define VCN_LC_L8(q) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(q)))
#define VCN_LC_ACT(x) vcn_lc_activity16(e,ri+(x))
  int tmax = 0;
  for (int k = 0; k < nk; k++) tmax = p->thr[k] > tmax ? p->thr[k] : tmax;
  if(pairbytes&&second)for(int k=0;k<second->K-1;k++)tmax=second->thr[k]>tmax?second->thr[k]:tmax;
  if (tmax < 255 && (!second||pairbytes)) {
    /* every threshold below 255: a > t equals min(a, 255) > t, so compare saturated bytes, 32 voxels at a
       time (unsigned order by the sign-bit flip). packus leaves the bytes in the order a[0..7], b[0..7],
       a[8..15], b[8..15]; the unpacks of the bias bytes undo it, the permute puts the classes in order */
    __m256i t8[VCN_LC_MAXK];
    const __m256i kx = _mm256_set1_epi8((char)0x80);
    for (int k = 0; k < nk; k++) t8[k] = _mm256_set1_epi8((char)(p->thr[k] ^ 0x80));
    __m256i t82[VCN_LC_MAXK];
    if(second)for(int k=0;k<second->K-1;k++)t82[k]=_mm256_set1_epi8((char)(second->thr[k]^0x80));
    for (int x = 0; x < VCN_LC_D; x += 32) {
      const __m256i a0=VCN_LC_ACT(x),a1=VCN_LC_ACT(x+16);
      if(activity){_mm256_storeu_si256((__m256i*)(activity+x),a0);_mm256_storeu_si256((__m256i*)(activity+x+16),a1);}
      const __m256i a8 = _mm256_xor_si256(_mm256_packus_epi16(a0,a1), kx);
      if(second){
        __m256i s0=_mm256_setzero_si256(),s1=s0;int k=0;
        for(;k+1<second->K-1;k+=2){s0=_mm256_sub_epi8(s0,_mm256_cmpgt_epi8(a8,t82[k]));s1=_mm256_sub_epi8(s1,_mm256_cmpgt_epi8(a8,t82[k+1]));}
        if(k<second->K-1)s0=_mm256_sub_epi8(s0,_mm256_cmpgt_epi8(a8,t82[k]));
        _mm256_storeu_si256((__m256i*)(kc2+x),_mm256_permute4x64_epi64(_mm256_add_epi8(s0,s1),0xD8));
      }
      __m256i c0 = _mm256_setzero_si256(), c1 = c0;
      int k = 0;
      for (; k + 1 < nk; k += 2)
        c0 = _mm256_sub_epi8(c0, _mm256_cmpgt_epi8(a8, t8[k])), c1 = _mm256_sub_epi8(c1, _mm256_cmpgt_epi8(a8, t8[k + 1]));
      if (k < nk) c0 = _mm256_sub_epi8(c0, _mm256_cmpgt_epi8(a8, t8[k]));
      const __m256i c = _mm256_add_epi8(c0, c1);
      const __m256i bl = _mm256_shuffle_epi8(tlo, c), bh = _mm256_shuffle_epi8(thi, c);
      _mm256_storeu_si256((__m256i *)(A + x), _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(A + x)), _mm256_unpacklo_epi8(bl, bh)));
      _mm256_storeu_si256((__m256i *)(A + x + 16),
                          _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(A + x + 16)), _mm256_unpackhi_epi8(bl, bh)));
      _mm256_storeu_si256((__m256i *)(kc + x), _mm256_permute4x64_epi64(c, 0xD8));
    }
    return;
  }
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const __m256i a = VCN_LC_ACT(x);
    if(activity)_mm256_storeu_si256((__m256i*)(activity+x),a);
    if(second){
      __m256i s0=_mm256_setzero_si256(),s1=s0;int k=0;
      for(;k+1<second->K-1;k+=2){s0=_mm256_sub_epi16(s0,_mm256_cmpgt_epi16(a,th2[k]));s1=_mm256_sub_epi16(s1,_mm256_cmpgt_epi16(a,th2[k+1]));}
      if(k<second->K-1)s0=_mm256_sub_epi16(s0,_mm256_cmpgt_epi16(a,th2[k]));
      __m256i c=_mm256_add_epi16(s0,s1);
      _mm_storeu_si128((__m128i*)(kc2+x),_mm_packus_epi16(_mm256_castsi256_si128(c),_mm256_extracti128_si256(c,1)));
    }
    __m256i c0 = _mm256_setzero_si256(), c1 = c0;
    int k = 0;
    for (; k + 1 < nk; k += 2)
      c0 = _mm256_sub_epi16(c0, _mm256_cmpgt_epi16(a, th[k])), c1 = _mm256_sub_epi16(c1, _mm256_cmpgt_epi16(a, th[k + 1]));
    if (k < nk) c0 = _mm256_sub_epi16(c0, _mm256_cmpgt_epi16(a, th[k]));
    const __m256i c = _mm256_add_epi16(c0, c1); /* classes, 0..15, in 16-bit lanes */
    const __m256i ix = _mm256_or_si256(c, _mm256_set1_epi16((short)0x8000)); /* the high bytes select zero */
    const __m256i cb = _mm256_or_si256(_mm256_shuffle_epi8(tlo, ix), _mm256_slli_epi16(_mm256_shuffle_epi8(thi, ix), 8));
    _mm256_storeu_si256((__m256i *)(A + x), _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(A + x)), cb));
    const __m128i c8 = _mm_packus_epi16(_mm256_castsi256_si128(c), _mm256_extracti128_si256(c, 1));
    _mm_storeu_si128((__m128i *)(kc + x), c8);
  }
#undef VCN_LC_ACT
#undef VCN_LC_L8
}
__attribute__((target("avx2"))) static void vcn_lc_cls_avx2(const uint8_t *e,size_t ri,const vcn_lc_par*p,uint16_t*restrict A,uint8_t*restrict kc){
  vcn_lc_cls_avx2_impl(e,ri,p,A,kc,NULL,NULL,0,NULL);
}
__attribute__((target("avx2"))) static void ng_cls_export_avx2(const uint8_t*e,size_t ri,const vcn_lc_par*p,uint16_t*A,uint8_t*kc,uint16_t*activity){
  vcn_lc_cls_avx2_impl(e,ri,p,A,kc,NULL,NULL,0,activity);
}
__attribute__((target("avx2"))) static void vcn_lc_cls_avx2_pair(const uint8_t *e,size_t ri,const vcn_lc_par*p,uint16_t*restrict A,uint8_t*restrict kc,const vcn_lc_par*second,uint8_t*restrict kc2,int pairbytes){
  if(pairbytes)vcn_lc_cls_avx2_impl(e,ri,p,A,kc,second,kc2,1,NULL);
  else vcn_lc_cls_avx2_impl(e,ri,p,A,kc,second,kc2,0,NULL);
}
#define VCN_LC_CLS(e, ri, p, A, kc) ((!ng_soft_classes && vcn_use_avx2() && (p)->K <= 16) ? vcn_lc_cls_avx2(e, ri, p, A, kc) : vcn_lc_cls_c(e, ri, p, A, kc))
#elif VCN_HAVE_NEON
/* the same for K <= 16 with NEON: the activity in 16-bit lanes, the class as the count of thresholds below it
   (saturated bytes, 16 voxels at a time, when every threshold is below 255; else 16-bit compares), the bias
   by table lookups of its low and high bytes */
static void vcn_lc_cls_neon(const uint8_t *e, size_t ri, const vcn_lc_par *p, uint16_t *restrict A, uint8_t *restrict kc) {
  const uint8_t *n = e + ri - VCN_LC_RP, *nn = e + ri - 2 * VCN_LC_RP, *u = e + ri - VCN_LC_SP, *uu = e + ri - 2 * VCN_LC_SP;
  const int nk = p->K - 1;
  uint8_t lo[16] = {0}, hi[16] = {0};
  for (int k = 0; k < p->K; k++) lo[k] = (uint8_t)p->cb[k], hi[k] = (uint8_t)((uint32_t)p->cb[k] >> 8);
  const uint8x16_t tlo = vld1q_u8(lo), thi = vld1q_u8(hi);
  int tmax = 0;
  for (int k = 0; k < nk; k++) tmax = p->thr[k] > tmax ? p->thr[k] : tmax;
  uint8x16_t t8[VCN_LC_MAXK];
  uint16x8_t t16[VCN_LC_MAXK];
  for (int k = 0; k < nk; k++) t8[k] = vdupq_n_u8((uint8_t)(p->thr[k] < 255 ? p->thr[k] : 255)), t16[k] = vdupq_n_u16(p->thr[k]);
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const uint8x16_t N = vld1q_u8(n + x), U = vld1q_u8(u + x), Nw = vld1q_u8(n + x - 1), Ne = vld1q_u8(n + x + 1),
                     Uw = vld1q_u8(u + x - 1), Ue = vld1q_u8(u + x + 1), Un = vld1q_u8(u + x - VCN_LC_RP),
                     Us = vld1q_u8(u + x + VCN_LC_RP), NN = vld1q_u8(nn + x), UU = vld1q_u8(uu + x);
    uint16x8_t al = vaddl_u8(vget_low_u8(N), vget_low_u8(U)), ah = vaddl_high_u8(N, U);
    al = vaddq_u16(al, al), ah = vaddq_u16(ah, ah);
    al = vaddq_u16(al, vaddq_u16(vaddl_u8(vget_low_u8(Nw), vget_low_u8(Ne)), vaddl_u8(vget_low_u8(Uw), vget_low_u8(Ue))));
    ah = vaddq_u16(ah, vaddq_u16(vaddl_high_u8(Nw, Ne), vaddl_high_u8(Uw, Ue)));
    al = vaddq_u16(al, vaddq_u16(vaddl_u8(vget_low_u8(Un), vget_low_u8(Us)), vaddl_u8(vget_low_u8(NN), vget_low_u8(UU))));
    ah = vaddq_u16(ah, vaddq_u16(vaddl_high_u8(Un, Us), vaddl_high_u8(NN, UU)));
    uint8x16_t c = vdupq_n_u8(0);
    if (tmax < 255) { /* a > t equals min(a, 255) > t for t < 255 */
      const uint8x16_t a8 = vcombine_u8(vqmovn_u16(al), vqmovn_u16(ah));
      for (int k = 0; k < nk; k++) c = vsubq_u8(c, vcgtq_u8(a8, t8[k]));
    } else {
      uint16x8_t cl = vdupq_n_u16(0), ch = cl;
      for (int k = 0; k < nk; k++) cl = vsubq_u16(cl, vcgtq_u16(al, t16[k])), ch = vsubq_u16(ch, vcgtq_u16(ah, t16[k]));
      c = vcombine_u8(vmovn_u16(cl), vmovn_u16(ch));
    }
    const uint8x16_t bl = vqtbl1q_u8(tlo, c), bh = vqtbl1q_u8(thi, c);
    vst1q_u16(A + x, vaddq_u16(vld1q_u16(A + x), vreinterpretq_u16_u8(vzip1q_u8(bl, bh))));
    vst1q_u16(A + x + 8, vaddq_u16(vld1q_u16(A + x + 8), vreinterpretq_u16_u8(vzip2q_u8(bl, bh))));
    vst1q_u8(kc + x, c);
  }
}
#define VCN_LC_CLS(e, ri, p, A, kc) ((!ng_soft_classes && (p)->K <= 16) ? vcn_lc_cls_neon(e, ri, p, A, kc) : vcn_lc_cls_c(e, ri, p, A, kc))
#else
#define VCN_LC_CLS(e, ri, p, A, kc) vcn_lc_cls_c(e, ri, p, A, kc)
#endif

#if VCN_HAVE_AVX2
__attribute__((target("avx512f,avx512bw,avx2")))
static void ng_apre_wide(const uint8_t*e,size_t ri,uint16_t*a){
  for(int x=0;x<VCN_LC_D;x+=32)_mm512_storeu_si512(a+x,vcn_lc_activity32(e,ri+x));
}
#endif
static void vcn_lc_apre(const uint8_t*e,size_t ri,uint16_t*restrict a){
#if VCN_HAVE_AVX2
  if(!ng_runtime_only&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw")){
    ng_apre_wide(e,ri,a);return;
  }
#endif
  ng_apre_scalar(e,ri,a);
}
/* previous-row taps: offsets and weights, and the same as pairs of signed 7-bit digits for vpmaddubsw,
   w = d0 + 128 d1 + 16384 d2 (mod 2^16), pairs ordered by the digits they need */
typedef struct {
  int np;
  int dense_r[3];
  uint16_t dense_w[VCN_LC_MAXT];
  ptrdiff_t o[VCN_LC_MAXT + 1];
  uint16_t w[VCN_LC_MAXT + 1];
  int npair, nlev[3];         /* pairs; of them nlev[0] need 1 digit, nlev[1] up to 2 */
  ptrdiff_t po[VCN_LC_MAXT / 2 + 1][2];
  int16_t pd[VCN_LC_MAXT / 2 + 1][3];
} vcn_lc_plist;
static void vcn_lc_mkplist(const vcn_lc_taps *t, const int32_t *w, vcn_lc_plist *pl) {
  if(ng_dense_pre&1){
    memcpy(pl->dense_r,t->r,sizeof(pl->dense_r));
    for(int k=0;k<t->nt;k++)pl->dense_w[k]=(uint16_t)w[k];
    return; /* no nonzero list, digit expansion, sorting, or pairing */
  }
  int n = 0;
  int d[VCN_LC_MAXT + 1][3], lev[VCN_LC_MAXT + 1];
  for (int k = t->nrow; k < t->nt; k++)
    if (w[k]||ng_dense_pre==2) pl->o[n] = t->off[k], pl->w[n] = (uint16_t)w[k], n++;
  pl->np = n;
  for (int k = 0; k < n; k++) {
    int x = (int16_t)pl->w[k];
    lev[k] = 1;
    for (int l = 0; l < 3; l++) {
      d[k][l] = ((x + 64) & 127) - 64;
      x = (x - d[k][l]) / 128;
      if (d[k][l]) lev[k] = l + 1;
    }
  }
  int ord[VCN_LC_MAXT + 1], m = 0;
  for (int L = 1; L <= 3; L++)
    for (int k = 0; k < n; k++)
      if (lev[k] == L) ord[m++] = k;
  if (n & 1) ord[n] = n, pl->o[n] = pl->o[0], d[n][0] = d[n][1] = d[n][2] = 0, lev[n] = 1; /* zero weight */
  pl->npair = (n + 1) / 2;
  pl->nlev[0] = pl->nlev[1] = 0;
  for (int i = 0; i < pl->npair; i++) {
    const int a = ord[2 * i], c = (2 * i + 1 < n) ? ord[2 * i + 1] : n;
    const int L = lev[a] > lev[c] ? lev[a] : lev[c];
    if (L <= 1) pl->nlev[0] = i + 1;
    if (L <= 2) pl->nlev[1] = i + 1;
    pl->po[i][0] = pl->o[a], pl->po[i][1] = pl->o[c];
    for (int l = 0; l < 3; l++) pl->pd[i][l] = (int16_t)(uint16_t)((uint8_t)d[a][l] | (unsigned)(uint8_t)d[c][l] << 8);
  }
}
#include "ng_dense_pre.h"
/* pre[x] = bias + the previous-row taps, modulo 2^16 */
static inline void vcn_lc_pre_span(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *restrict pre,int count){
  if(ng_dense_pre&1){ng_dense_pre_scalar(row,pl,bias,pre,count);return;}
  for(int x=0;x<count;x++)pre[x]=bias;
  for(int k=0;k<pl->np;k++){
    const uint8_t* restrict s=row+pl->o[k];uint16_t w=pl->w[k];
    for(int x=0;x<count;x++)pre[x]=(uint16_t)(pre[x]+(uint32_t)w*s[x]);
  }
}
static void vcn_lc_pre_c_impl(const uint8_t *row, const vcn_lc_plist *pl, uint16_t bias, uint16_t *restrict pre, unsigned needed) {
  for (int x = 0; x < VCN_LC_D; x++) pre[x] = bias;
  for(int x=0;x<VCN_LC_D;x+=32)if(needed&(1u<<(x>>5)))vcn_lc_pre_span(row+x,pl,bias,pre+x,32);
}
VCN_MAYBE_UNUSED static void vcn_lc_pre_c(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *pre) {
  vcn_lc_pre_c_impl(row,pl,bias,pre,~0u);
}
#if VCN_HAVE_AVX2
#define NG_PATTR __attribute__((target("avx2")))
#define NG_PNAME vcn_lc_pre_avx2_impl
#define NG_PV __m256i
#define NG_PWIDTH 32
#define NG_PSET _mm256_set1_epi16
#define NG_PZERO _mm256_setzero_si256
#define NG_PLOAD _mm256_loadu_si256
#define NG_PADD _mm256_add_epi16
#define NG_PMADD _mm256_maddubs_epi16
#define NG_PLO _mm256_unpacklo_epi8
#define NG_PHI _mm256_unpackhi_epi8
#define NG_PSHIFT _mm256_slli_epi16
#define NG_PSTORE(P,L,H) do { \
  _mm256_storeu_si256((__m256i*)(P),_mm256_permute2x128_si256(L,H,0x20)); \
  _mm256_storeu_si256((__m256i*)((P)+16),_mm256_permute2x128_si256(L,H,0x31)); \
} while(0)
#include "ng_pre_vector.h"
#if (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 12)
#define NG_HAVE_WIDE_PRE 1
#define NG_PNAME vcn_lc_pre_avx512_impl
#include "ng_pre_wide.h"
#endif
__attribute__((target("avx2"))) static void vcn_lc_pre_avx2(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *pre) {
  vcn_lc_pre_avx2_impl(row,pl,bias,pre,~0u);
}
/* Whole-weight multiplication is exact modulo 2^16, avoiding digit splitting
   when learned feature models produce large coefficients. */
__attribute__((target("avx512f,avx512bw,avx2")))
static void ng_pre_encode_multiply(const uint8_t*row,const vcn_lc_plist*pl,uint16_t bias,uint16_t*pre){
  for(int x=0;x<VCN_LC_D;x+=32){
    __m512i a=_mm512_set1_epi16((short)bias),b=_mm512_setzero_si512();
    int k=0;
    for(;k+1<pl->np;k+=2){
      __m512i v0=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row+x+pl->o[k])));
      __m512i v1=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row+x+pl->o[k+1])));
      a=_mm512_add_epi16(a,_mm512_mullo_epi16(v0,_mm512_set1_epi16((short)pl->w[k])));
      b=_mm512_add_epi16(b,_mm512_mullo_epi16(v1,_mm512_set1_epi16((short)pl->w[k+1])));
    }
    if(k<pl->np)a=_mm512_add_epi16(a,_mm512_mullo_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row+x+pl->o[k]))),_mm512_set1_epi16((short)pl->w[k])));
    _mm512_storeu_si512(pre+x,_mm512_add_epi16(a,b));
  }
}
static void ng_pre_encode(const uint8_t*row,const vcn_lc_plist*pl,uint16_t bias,uint16_t*pre){
  if(ng_dense_pre&1){ng_dense_pre_dispatch(row,pl,bias,pre,~0u);return;}
  if(!ng_runtime_only&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw")){
    ng_pre_encode_multiply(row,pl,bias,pre);return;
  }
  if(vcn_use_avx2())vcn_lc_pre_avx2(row,pl,bias,pre);else vcn_lc_pre_c(row,pl,bias,pre);
}
#define VCN_LC_PRE(r, pl, b, o) ng_pre_encode(r,pl,b,o)
#elif VCN_HAVE_NEON
/* the same with 16-bit multiply-adds by the whole weight (modulo 2^16, as the C form), 32 voxels per step */
static void vcn_lc_pre_neon_impl(const uint8_t *row, const vcn_lc_plist *pl, uint16_t bias, uint16_t *restrict pre,unsigned needed) {
  if(ng_dense_pre&1){ng_dense_pre_dispatch(row,pl,bias,pre,needed);return;}
  for (int x = 0; x < VCN_LC_D; x += 32) {
    if(!(needed & (1u<<(x>>5)))) {memset(pre+x,0,32*sizeof(*pre));continue;}
    uint16x8_t a0 = vdupq_n_u16(bias), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < pl->np; k++) {
      const uint8_t *s = row + pl->o[k] + x;
      const uint8x16_t v0 = vld1q_u8(s), v1 = vld1q_u8(s + 16);
      const uint16_t w = pl->w[k];
      a0 = vmlaq_n_u16(a0, vmovl_u8(vget_low_u8(v0)), w), a1 = vmlaq_n_u16(a1, vmovl_high_u8(v0), w);
      a2 = vmlaq_n_u16(a2, vmovl_u8(vget_low_u8(v1)), w), a3 = vmlaq_n_u16(a3, vmovl_high_u8(v1), w);
    }
    vst1q_u16(pre + x, a0), vst1q_u16(pre + x + 8, a1), vst1q_u16(pre + x + 16, a2), vst1q_u16(pre + x + 24, a3);
  }
}
static void vcn_lc_pre_neon(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *pre) {
  vcn_lc_pre_neon_impl(row,pl,bias,pre,~0u);
}
#define VCN_LC_PRE(r, pl, b, o) vcn_lc_pre_neon(r, pl, b, o)
#else
#define VCN_LC_PRE(r, pl, b, o) vcn_lc_pre_c(r, pl, b, o)
#endif
/* Only omit full SIMD spans with no NG-selected voxels. Classes still use
   every residual and the causal schedule/padding are unchanged. */
static void vcn_lc_pre_selected(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *pre,unsigned needed,int scalar) {
#if VCN_HAVE_AVX2
  if(!scalar && vcn_use_avx2()){vcn_lc_pre_avx2_impl(row,pl,bias,pre,needed);return;}
#elif VCN_HAVE_NEON
  if(!scalar){vcn_lc_pre_neon_impl(row,pl,bias,pre,needed);return;}
#endif
  vcn_lc_pre_c_impl(row,pl,bias,pre,needed);
}
/* wavefront schedule: row (z, y) runs at step y + d z; the rows of one step are independent and go in
   groups of up to 32 */
#include "ng_class_lookup.h"
#ifdef NG_HAVE_WIDE_PRE
#define NG_PNAME ng_pre_class_fused
#define NG_PFUSED 1
#include "ng_pre_wide.h"
#endif
typedef struct { int ng; uint16_t (*row)[VCN_LC_NL]; uint8_t *n; } vcn_lc_sched;
static int vcn_lc_delta(const vcn_lc_taps *t) {
  int d = t->r[1] > t->r[2] ? t->r[1] : t->r[2];
  return (d < 1 ? 1 : d) + 1;
}
static int vcn_lc_mksched(int dl, vcn_lc_sched *s) {
  s->row = (uint16_t(*)[VCN_LC_NL])malloc(sizeof *s->row * VCN_LC_Y * VCN_LC_Z);
  s->n = (uint8_t *)malloc(VCN_LC_Y * VCN_LC_Z);
  if (!s->row || !s->n) return -1;
  int g = 0;
  for (int t = 0; t <= VCN_LC_Y - 1 + dl * (VCN_LC_Z - 1); t++) {
    int k = 0;
    for (int z = 0; z < VCN_LC_Z; z++) {
      const int y = t - dl * z;
      if (y < 0) break;
      if (y >= VCN_LC_Y) continue;
      if (k == 0) s->n[g] = 0;
      s->row[g][k++] = (uint16_t)(z * VCN_LC_Y + y);
      s->n[g] = (uint8_t)k;
      if (k == VCN_LC_NL) g++, k = 0;
    }
    if (k) g++;
  }
  s->ng = g;
  return 0;
}
static int vcn_lc_read_table(const uint8_t **pp, const uint8_t *end, uint16_t *freq, uint32_t M) {
  const uint8_t *p = *pp;
  if (end - p < 2) return -1;
  unsigned n = vcn_rd16(p);
  p += 2;
  if (n == 0 || n > 256) return -1;
  uint32_t sum = 0;
  memset(freq, 0, 512);
  for (unsigned s = 0; s < n; s++) {
    uint32_t v = 0;
    unsigned sh = 0;
    for (;;) {
      if (p >= end || sh > 14) return -1;
      uint8_t by = *p++;
      v |= (uint32_t)(by & 127u) << sh;
      sh += 7;
      if (!(by & 128u)) break;
    }
    if (v > M) return -1;
    freq[s] = (uint16_t)v, sum += v;
  }
  if (sum != M) return -1;
  *pp = p;
  return 0;
}
static int vcn_lc_read_tables(const uint8_t **p,const uint8_t *end,uint32_t *T,int K,uint32_t M) {
  for(int k=0;k<K;k++) {
    uint16_t fr[256];
    if(vcn_lc_read_table(p,end,fr,M))return -1;
    if(!T)continue; /* Validate/skip metadata without constructing decode tables. */
    uint32_t cum=0,*Tk=T+(size_t)k*M;
    for(unsigned s=0;s<256;s++) {
      uint32_t f=fr[s];
      for(uint32_t j=0;j<f;j++)Tk[cum+j]=s|(f-1u)<<8|j<<20;
      cum+=f;
    }
  }
  return 0;
}
static size_t vcn_lc_write_table(uint8_t* out,const uint16_t fr[256]) {
  unsigned nn=256;while(nn>1 && fr[nn-1]==0)nn--;
  vcn_wr16(out,nn);size_t pos=2;
  for(unsigned s=0;s<nn;s++) {
    uint32_t v=fr[s];do {uint8_t by=v&127u;v>>=7;out[pos++]=(uint8_t)(by|(v?128u:0u));}while(v);
  }
  return pos;
}
/* the in-row part and the rounding, as the decoder does them: pp is the predicted value, fm 1 when the
   prediction was rounded up */
#define VCN_LC_PP(P) ((uint16_t)((P) + 128u) >> 8)
#define VCN_LC_FM(P) (((P) >> 7) & 1u)
/* phase timers for the profiler (m6speed/m6_prof.c); no code by default */
#ifndef VCN_LC_TICK
#define VCN_LC_TICK(k) ((void)0)
#endif

/* the encoder's row kernels. pass 1: P = pre + the in-row taps, |the bias-free residual|; symbols: from
   P with its class bias, the zigzag of the residual (sign flipped when P rounds up), |residual| */
static void vcn_lc_pass1_span(const uint16_t *restrict pre, const uint8_t *restrict br, const uint16_t *w, uint16_t *restrict Q,
                           uint8_t *restrict er,int count) {
  for (int x = 0; x < count; x++) {
    const uint16_t P = (uint16_t)(pre[x] + (uint16_t)(w[0] * br[x - 1]) + (uint16_t)(w[1] * br[x - 2]) + (uint16_t)(w[2] * br[x - 3]));
    const int r = (int)(int8_t)(uint8_t)(br[x] - VCN_LC_PP(P));
    Q[x] = P;
    er[x] = (uint8_t)(r < 0 ? -r : r);
  }
}
VCN_MAYBE_UNUSED static void vcn_lc_pass1_c(const uint16_t *restrict pre,const uint8_t *restrict br,const uint16_t*w,uint16_t*restrict Q,uint8_t*restrict er){vcn_lc_pass1_span(pre,br,w,Q,er,VCN_LC_D);}
#include "ng_symbol_mapping.h"
VCN_MAYBE_UNUSED static void vcn_lc_syms_enc_c(const uint16_t *restrict Q, const uint8_t *restrict vr, uint8_t *restrict sr, uint8_t *restrict er) {
  for (int x = 0; x < VCN_LC_D; x++) {
    const unsigned sy = ng_prediction_symbol(vr[x],Q[x]);
    sr[x] = (uint8_t)sy, er[x] = (uint8_t)((sy + 1) >> 1);
  }
}
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static void vcn_lc_pass1_avx2(const uint16_t *restrict pre, const uint8_t *restrict br, const uint16_t *w,
                                                              uint16_t *restrict Q, uint8_t *restrict er) {
  const __m256i w0 = _mm256_set1_epi16((short)w[0]), w1 = _mm256_set1_epi16((short)w[1]), w2 = _mm256_set1_epi16((short)w[2]),
                k128 = _mm256_set1_epi16(128);
  for (int x = 0; x < VCN_LC_D; x += 16) {
#define VCN_LC_L8(q) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(q)))
    const __m256i P = _mm256_add_epi16(_mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(pre + x)), _mm256_mullo_epi16(VCN_LC_L8(br + x - 1), w0)),
                                       _mm256_add_epi16(_mm256_mullo_epi16(VCN_LC_L8(br + x - 2), w1), _mm256_mullo_epi16(VCN_LC_L8(br + x - 3), w2)));
    const __m256i r = _mm256_sub_epi16(VCN_LC_L8(br + x), _mm256_srli_epi16(_mm256_add_epi16(P, k128), 8));
    const __m256i a = _mm256_abs_epi16(_mm256_srai_epi16(_mm256_slli_epi16(r, 8), 8));
    _mm256_storeu_si256((__m256i *)(Q + x), P);
    _mm_storeu_si128((__m128i *)(er + x), _mm_packus_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)));
  }
}
__attribute__((target("avx2"))) static void vcn_lc_syms_enc_avx2(const uint16_t *restrict Q, const uint8_t *restrict vr, uint8_t *restrict sr,
                                                                 uint8_t *restrict er) {
  const __m256i k128 = _mm256_set1_epi16(128), k1 = _mm256_set1_epi16(1);
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const __m256i P = _mm256_loadu_si256((const __m256i *)(Q + x));
    const __m256i d = _mm256_sub_epi16(VCN_LC_L8(vr + x), _mm256_srli_epi16(_mm256_add_epi16(P, k128), 8));
    const __m256i ev = _mm256_srai_epi16(_mm256_slli_epi16(d, 8), 8);
    const __m256i fm = _mm256_srai_epi16(_mm256_slli_epi16(P, 8), 15); /* -1 when P rounds up */
    const __m256i sg = _mm256_srai_epi16(_mm256_slli_epi16(_mm256_sub_epi16(_mm256_xor_si256(ev, fm), fm), 8), 8);
    const __m256i sy = _mm256_xor_si256(_mm256_slli_epi16(sg, 1), _mm256_srai_epi16(sg, 15));
    const __m256i sy8 = _mm256_and_si256(sy, _mm256_set1_epi16(255));
    const __m256i ea = _mm256_srli_epi16(_mm256_add_epi16(sy8, k1), 1);
    _mm_storeu_si128((__m128i *)(sr + x), _mm_packus_epi16(_mm256_castsi256_si128(sy8), _mm256_extracti128_si256(sy8, 1)));
    _mm_storeu_si128((__m128i *)(er + x), _mm_packus_epi16(_mm256_castsi256_si128(ea), _mm256_extracti128_si256(ea, 1)));
#undef VCN_LC_L8
  }
}
#define VCN_LC_PASS1(pre, br, w, Q, er) (vcn_use_avx2() ? vcn_lc_pass1_avx2(pre, br, w, Q, er) : vcn_lc_pass1_c(pre, br, w, Q, er))
#define VCN_LC_SYMS(Q, vr, sr, er) (vcn_use_avx2() ? vcn_lc_syms_enc_avx2(Q, vr, sr, er) : vcn_lc_syms_enc_c(Q, vr, sr, er))
#elif VCN_HAVE_NEON
static void vcn_lc_pass1_neon(const uint16_t *restrict pre, const uint8_t *restrict br, const uint16_t *w, uint16_t *restrict Q,
                              uint8_t *restrict er) {
  const uint16x8_t k128 = vdupq_n_u16(128);
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const uint8x16_t b1 = vld1q_u8(br + x - 1), b2 = vld1q_u8(br + x - 2), b3 = vld1q_u8(br + x - 3), b0 = vld1q_u8(br + x);
    uint16x8_t Pl = vld1q_u16(pre + x), Ph = vld1q_u16(pre + x + 8);
    Pl = vmlaq_n_u16(vmlaq_n_u16(vmlaq_n_u16(Pl, vmovl_u8(vget_low_u8(b1)), w[0]), vmovl_u8(vget_low_u8(b2)), w[1]), vmovl_u8(vget_low_u8(b3)), w[2]);
    Ph = vmlaq_n_u16(vmlaq_n_u16(vmlaq_n_u16(Ph, vmovl_high_u8(b1), w[0]), vmovl_high_u8(b2), w[1]), vmovl_high_u8(b3), w[2]);
    /* the residual modulo 256 as int8, then its magnitude (|-128| = 128, as the C form) */
    const uint8x16_t pp = vcombine_u8(vshrn_n_u16(vaddq_u16(Pl, k128), 8), vshrn_n_u16(vaddq_u16(Ph, k128), 8));
    vst1q_u16(Q + x, Pl), vst1q_u16(Q + x + 8, Ph);
    vst1q_u8(er + x, vreinterpretq_u8_s8(vabsq_s8(vreinterpretq_s8_u8(vsubq_u8(b0, pp)))));
  }
}
static void vcn_lc_syms_enc_neon(const uint16_t *restrict Q, const uint8_t *restrict vr, uint8_t *restrict sr, uint8_t *restrict er) {
  const uint16x8_t k128 = vdupq_n_u16(128);
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const uint16x8_t Pl = vld1q_u16(Q + x), Ph = vld1q_u16(Q + x + 8);
    const uint8x16_t pp = vcombine_u8(vshrn_n_u16(vaddq_u16(Pl, k128), 8), vshrn_n_u16(vaddq_u16(Ph, k128), 8));
    const int8x16_t ev = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(vr + x), pp));
    /* -1 where P rounds up (bit 7 of P), else 0; the sign flip and the zigzag in int8, wrapping as the C form */
    const int8x16_t fm = vshrq_n_s8(vreinterpretq_s8_u8(vcombine_u8(vmovn_u16(Pl), vmovn_u16(Ph))), 7);
    const int8x16_t sg = vsubq_s8(veorq_s8(ev, fm), fm);
    const uint8x16_t sy = vreinterpretq_u8_s8(veorq_s8(vshlq_n_s8(sg, 1), vshrq_n_s8(sg, 7)));
    vst1q_u8(sr + x, sy);
    vst1q_u8(er + x, vrshrq_n_u8(sy, 1)); /* (sy + 1) >> 1 without overflow */
  }
}
#define VCN_LC_PASS1(pre, br, w, Q, er) vcn_lc_pass1_neon(pre, br, w, Q, er)
#define VCN_LC_SYMS(Q, vr, sr, er) vcn_lc_syms_enc_neon(Q, vr, sr, er)
#else
#define VCN_LC_PASS1(pre, br, w, Q, er) vcn_lc_pass1_c(pre, br, w, Q, er)
#define VCN_LC_SYMS(Q, vr, sr, er) vcn_lc_syms_enc_c(Q, vr, sr, er)
#endif

/* encode one chunk: fit; one pass in raster order for the bias-free predictions, the activity histogram
   and the residual sums (row-parallel); thresholds and biases; a second raster pass for the classes and
   symbols; then rANS (reciprocal form) over the decoder's order */
typedef struct {
  int r0, r1, r2, K, tb, nsel, stride;
  /* Experimental encoder-only inputs; zero initialized for the ordinary codec. */
  const int16_t *fit_target;
  const uint8_t *base_prediction;
  int zero_correction;
  int single_model; /* one fitted bank, no difficulty-dependent bias */
  uint16_t *export_prediction;
  uint16_t *export_activity; /* Optional causal activity, reused by entropy-only tuning. */
  uint32_t *export_activity_histogram;
  /* Already prepared symbols/classes: share the native entropy encoder. */
  const uint8_t *import_sym, *import_cls, *import_params;
  const uint8_t *import_targets; /* diagnostic sampling; never changes source context */
  const uint16_t *import_activity;
  const uint8_t *import_class_lut;
  size_t import_params_n;
  int raster_entropy;
  const uint8_t *fit_feature;
  int feature_mode; /* 1 free feature coefficient; 3 centered contrasts */
  int32_t *export_feature_weights;
  const uint8_t *fit_mask; /* optional membership mask for model-bank regression */
  const uint8_t *fit_weights; /* optional integer entropy-surrogate weights */
  const vcn_lc_par *damp_previous; /* optional encoder-only half-step refit */
  const int32_t *damp_feature_weights;
  const vcn_lc_par *fit_support, *fixed_fit;
  const int32_t *fixed_feature_weights;
  ng_fit_session *fit_session; /* optional same-chunk independent-fit reuse */
  int32_t *context_weights; /* optional experimental 2x8 nonlinear fit outputs */
  int embed_context; /* serialize derived coefficients as model version4 */
} vcn_lc_cfg;
typedef struct { uint32_t xmax, rcp, bias; uint16_t cmpl, shift; } vcn_lc_esym;
static inline void vcn_lc_init_esym(vcn_lc_esym*E,uint32_t f,uint32_t cum,uint32_t M,int tb){
  E->xmax=((VCN_LC_RL>>tb)<<16)*f;E->cmpl=(uint16_t)(M-f);
  if(f<2){E->rcp=~0u;E->shift=32;E->bias=cum+M-1;}
  else{uint32_t sh=0;while(f>(1u<<sh))sh++;E->rcp=(uint32_t)(((1ull<<(sh+31))+f-1)/f);E->shift=(uint16_t)(sh-1+32);E->bias=cum;}
}
static inline uint32_t vcn_lc_put_esym(uint32_t xs,const vcn_lc_esym*E,uint16_t*words,size_t*nw){
  unsigned fl=xs>=E->xmax;words[*nw]=(uint16_t)xs;*nw+=fl;xs>>=fl<<4;
  uint32_t q=(uint32_t)(((uint64_t)xs*E->rcp)>>E->shift);return xs+E->bias+q*E->cmpl;
}
#if VCN_HAVE_AVX2
/* Sixteen independent states, same reverse renormalization order as scalar.
   Compact the emitted words only after all state transitions are evaluated. */
__attribute__((target("avx512f,avx512bw,avx2")))
static size_t ng_rans_encode_row_wide(const uint8_t*cr,const uint8_t*sr,int width,
    const vcn_lc_esym*table,uint32_t*S,uint16_t*words,size_t nw){
  const __m512i reverse=_mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  for(int x=width;x>0;x-=16){
    const int lane=(x-16)&(VCN_LC_NS-1);
    __m512i c=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(cr+x-16)));
    __m512i s=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(sr+x-16)));
    __m512i idx=_mm512_permutexvar_epi32(reverse,_mm512_slli_epi32(_mm512_add_epi32(_mm512_slli_epi32(c,8),s),2));
    const uint32_t*t=(const uint32_t*)table;
    __m512i xmax=_mm512_i32gather_epi32(idx,t,4),rcp=_mm512_i32gather_epi32(idx,t+1,4);
    __m512i bias=_mm512_i32gather_epi32(idx,t+2,4),cs=_mm512_i32gather_epi32(idx,t+3,4);
    __m512i old=_mm512_permutexvar_epi32(reverse,_mm512_loadu_si512(S+lane));
    __mmask16 flush=_mm512_cmpge_epu32_mask(old,xmax);
    __m512i v=_mm512_mask_srli_epi32(old,flush,old,16);
    __m512i even=_mm512_srli_epi64(_mm512_mul_epu32(v,rcp),32);
    __m512i odd=_mm512_mul_epu32(_mm512_srli_epi64(v,32),_mm512_srli_epi64(rcp,32));
    __m512i high=_mm512_mask_blend_epi32(0xaaaa,even,odd);
    __m512i q=_mm512_srlv_epi32(high,_mm512_sub_epi32(_mm512_srli_epi32(cs,16),_mm512_set1_epi32(32)));
    __m512i next=_mm512_add_epi32(_mm512_add_epi32(v,bias),_mm512_mullo_epi32(q,_mm512_and_si512(cs,_mm512_set1_epi32(65535))));
    _mm512_storeu_si512(S+lane,_mm512_permutexvar_epi32(reverse,next));
    __m512i emitted=_mm512_maskz_compress_epi32(flush,old);
    _mm256_storeu_si256((__m256i*)(words+nw),_mm512_cvtepi32_epi16(emitted));
    nw+=(unsigned)__builtin_popcount((unsigned)flush);
  }
  return nw;
}
#endif
/* Shared native model serialization, also used by fit-only experiments. */
static size_t vcn_lc_writepar(const vcn_lc_par*p,const vcn_lc_cfg*cf,uint8_t*hdr){
  size_t pos=0;hdr[pos++]=p->has_context?4:VCN_LC_VERSION;hdr[pos++]=cf->r0;hdr[pos++]=cf->r1;hdr[pos++]=cf->r2;hdr[pos++]=p->K;hdr[pos++]=p->tb;
  for(int b=0;b<2;b++){
    for(int k=0;k<p->t.nt;k++){vcn_wr16(hdr+pos,(uint16_t)(int16_t)p->w[b][k]);pos+=2;}
    vcn_wr16(hdr+pos,(uint16_t)p->bias[b]);pos+=2;
  }
  if(p->has_context)for(int b=0;b<2;b++)for(int k=0;k<8;k++){vcn_wr16(hdr+pos,(uint16_t)(int16_t)p->context[b][k]);pos+=2;}
  for(int k=0;k<p->K-1;k++){vcn_wr16(hdr+pos,p->thr[k]);pos+=2;}
  for(int k=0;k<p->K;k++){vcn_wr16(hdr+pos,(uint16_t)(int16_t)p->cb[k]);pos+=2;}
  return pos;
}
static vcn_status vcn_lc_encode_ex(const uint8_t *src, const vcn_lc_cfg *cf, uint8_t *out, size_t cap, size_t *out_n,
                                  uint8_t *export_sym, uint8_t *export_cls) {
  const size_t NV = (size_t)VCN_LC_D * VCN_LC_Y * VCN_LC_Z;
  int32_t feature_weights[2]={0,0};
  const int K = cf->K, tb = cf->tb;
  const uint32_t M = 1u << tb;
  vcn_lc_par *p = (vcn_lc_par *)calloc(1, sizeof(vcn_lc_par));
  const int imported=cf->import_sym!=NULL;
  uint8_t *b = imported?NULL:cf->fit_session&&cf->fit_session->padded?cf->fit_session->padded:(uint8_t *)malloc(VCN_LC_NB), *e = imported?NULL:(uint8_t *)malloc(VCN_LC_NB);
  if(cf->fit_session)cf->fit_session->padded=b;
  uint16_t *Pa = imported?NULL:(uint16_t *)malloc(2 * NV);
  uint32_t(*h)[VCN_LC_AMAX] = imported?NULL:(uint32_t(*)[VCN_LC_AMAX])calloc(1, sizeof *h);
  int64_t(*sa)[VCN_LC_AMAX] = imported?NULL:(int64_t(*)[VCN_LC_AMAX])calloc(1, sizeof *sa);
  uint8_t *sym = imported?NULL:export_sym?export_sym:(uint8_t *)malloc(NV);
  const int row_classes=export_sym&&!export_cls;
  uint8_t *cls = imported&&!cf->import_activity?NULL:export_cls?export_cls:(uint8_t *)malloc(row_classes?VCN_LC_D:NV);
  const uint8_t *coded_sym=imported?cf->import_sym:sym, *coded_cls=imported&&!cf->import_activity?cf->import_cls:cls;
  uint32_t(*cnt)[256] = export_sym?NULL:(uint32_t(*)[256])calloc(K, sizeof *cnt);
  vcn_lc_esym(*es)[256] = export_sym?NULL:(vcn_lc_esym(*)[256])malloc(K * sizeof *es);
  uint16_t *words = export_sym?NULL:(uint16_t *)malloc(2 * (NV + 64));
  uint8_t *hdr = (uint8_t *)malloc(9 + 2 * (2 * VCN_LC_MAXT + 4) + 4 * VCN_LC_MAXK + K * (2 + 2 * 256));
  vcn_lc_sched sc = {0, NULL, NULL};
  vcn_status st = VCN_OK;
  if (!p || (!imported && (!b || !e || !Pa || !h || !sa || !sym || !cls)) || !coded_sym || !coded_cls || (!export_sym&&(!cnt || !es || !words)) || !hdr) {
    st = VCN_ERR_NOMEM;
    goto done;
  }
  VCN_LC_TICK(-1);
  size_t pos = 0;
  vcn_lc_build(&p->t, cf->r0, cf->r1, cf->r2);
  p->K = K, p->tb = tb;
  const vcn_lc_taps *t = &p->t;
  if (cf->import_sym) {
    if(cf->import_activity){
      if(!cf->import_class_lut){st=VCN_ERR_ARG;goto done;}
      for(size_t i=0;i<NV;i++){
        unsigned a=cf->import_activity[i];if(a>=VCN_LC_AMAX){st=VCN_ERR_CORRUPT;goto done;}
        unsigned c=cf->import_class_lut[a];cls[i]=(uint8_t)c;
        if(!cf->import_targets||cf->import_targets[i])cnt[c][coded_sym[i]]++;
      }
    }else{
    for(size_t i=0;i<NV;i++) {
      if(coded_cls[i]>=K){st=VCN_ERR_CORRUPT;goto done;}
      if(!cf->import_targets||cf->import_targets[i])cnt[coded_cls[i]][coded_sym[i]]++;
    }
    }
    pos=cf->import_params_n;
    if(pos>9+2*(2*VCN_LC_MAXT+4)+4*VCN_LC_MAXK){st=VCN_ERR_CAP;goto done;}
    memcpy(hdr,cf->import_params,pos);
    goto entropy_ready;
  }
  /* the padded volume in its final state, for the fit */
  if(!cf->fit_session||!cf->fit_session->padded_ready){
  memset(b, 128, VCN_LC_NB);
  for (int z = 0; z < VCN_LC_Z; z++)
    for (int y = 0; y < VCN_LC_Y; y++) {
      vcn_lc_row_begin(b, z, y, 128);
      memcpy(b + VCN_LC_IDX(z, y, 0), src + ((size_t)z * VCN_LC_Y + y) * VCN_LC_D, VCN_LC_D);
      vcn_lc_row_end(b, z, y);
    }
  if(cf->fit_session)cf->fit_session->padded_ready=1;
  }
  VCN_LC_TICK(8);
  if(cf->fixed_fit){
    memcpy(p->w,cf->fixed_fit->w,sizeof(p->w));memcpy(p->bias,cf->fixed_fit->bias,sizeof(p->bias));
    if(cf->fit_feature)for(int bank=0;bank<2;bank++){
      feature_weights[bank]=cf->fixed_feature_weights[bank];
      if(cf->export_feature_weights)cf->export_feature_weights[bank]=feature_weights[bank];
    }
  }else if(cf->context_weights){
    for(int bank=0;bank<2;bank++){
      int32_t weights[VCN_LC_MAXF]={0};
      vcn_lc_fit_features_supported(b,t,bank?0:1,bank?1:VCN_LC_Z,bank?(cf->stride==1?1:2):cf->stride,bank,bank?cf->nsel/2:cf->nsel,
        weights,&p->bias[bank],NULL,cf->fit_feature,cf->fit_feature?5:4,0,cf->fit_mask,NULL,NULL,NULL);
      memcpy(p->w[bank],weights,t->nt*sizeof(*weights));
      memcpy(cf->context_weights+8*bank,weights+t->nt,8*sizeof(*weights));
      if(cf->fit_feature)feature_weights[bank]=weights[t->nt+8];
      if(cf->embed_context){p->has_context=1;memcpy(p->context[bank],weights+t->nt,8*sizeof(*weights));}
    }
  }else if(cf->fit_feature){
    for(int bank=0;bank<2;bank++){
      int32_t weights[VCN_LC_MAXF]={0};
      vcn_lc_fit_features_supported(b,t,bank?0:1,bank?1:VCN_LC_Z,bank?(cf->stride==1?1:2):cf->stride,bank,bank?cf->nsel/2:cf->nsel,
          weights,&p->bias[bank],NULL,cf->fit_feature,cf->feature_mode,0,cf->fit_mask,cf->fit_session,cf->fit_weights,cf->fit_support?cf->fit_support->w[bank]:NULL);
      memcpy(p->w[bank],weights,t->nt*sizeof(*weights));
      if(cf->feature_mode==1)feature_weights[bank]=weights[t->nt];
      else {int32_t sum=0;for(int k=0;k<t->nt;k++)sum+=weights[k];feature_weights[bank]=256-sum;}
      if(cf->export_feature_weights)cf->export_feature_weights[bank]=feature_weights[bank];
    }
  }else if((cf->fit_mask||cf->fit_weights||cf->fit_support)&&!cf->zero_correction){
    for(int bank=0;bank<2;bank++)
      vcn_lc_fit_features_supported(b,t,bank?0:1,bank?1:VCN_LC_Z,bank?(cf->stride==1?1:2):cf->stride,bank,bank?cf->nsel/2:cf->nsel,
          p->w[bank],&p->bias[bank],cf->fit_target,NULL,0,0,cf->fit_mask,NULL,cf->fit_weights,cf->fit_support?cf->fit_support->w[bank]:NULL);
  }else if (!cf->zero_correction) {
    vcn_lc_fit_target(b, t, cf->single_model?0:1, VCN_LC_Z, cf->stride, 0, cf->nsel, p->w[0], &p->bias[0], cf->fit_target);
    if(cf->single_model){memcpy(p->w[1],p->w[0],sizeof(p->w[0]));p->bias[1]=p->bias[0];}
    else vcn_lc_fit_target(b, t, 0, 1, cf->stride==1?1:2, 1, cf->nsel / 2, p->w[1], &p->bias[1], cf->fit_target);
  }
  if(cf->damp_previous){
    for(int bank=0;bank<2;bank++){
      for(int k=0;k<t->nt;k++)p->w[bank][k]=(int32_t)(((int64_t)p->w[bank][k]+cf->damp_previous->w[bank][k])/2);
      /* Intercepts are stored modulo65536, not as ordinary signed scalars.
         Move halfway along the shortest modular difference. */
      uint16_t delta=(uint16_t)((uint16_t)p->bias[bank]-(uint16_t)cf->damp_previous->bias[bank]);
      int32_t signed_delta=delta<32768?delta:(int32_t)delta-65536;
      p->bias[bank]=(uint16_t)((uint16_t)cf->damp_previous->bias[bank]+signed_delta/2);
      if(cf->fit_feature){
        feature_weights[bank]=(int32_t)(((int64_t)feature_weights[bank]+cf->damp_feature_weights[bank])/2);
        if(cf->export_feature_weights)cf->export_feature_weights[bank]=feature_weights[bank];
      }
    }
  }
  memset(h,0,sizeof(*h));memset(sa,0,sizeof(*sa));
  if(cnt)memset(cnt,0,(size_t)K*sizeof(*cnt));
  if(cf->export_feature_weights)for(int bank=0;bank<2;bank++)cf->export_feature_weights[bank]=feature_weights[bank];
  VCN_LC_TICK(9);
  {
    vcn_lc_plist pl[2];
    vcn_lc_mkplist(t, p->w[0], &pl[0]), vcn_lc_mkplist(t, p->w[1], &pl[1]);
    _Alignas(32) uint16_t pre[VCN_LC_D], ap[VCN_LC_D];
    if(!cf->fit_session)memset(b,128,VCN_LC_NB);
    memset(e, 0, VCN_LC_NB);
    for (int z = 0; z < VCN_LC_Z; z++) {
      const int s0 = z == 0;
      uint16_t w[3] = {0, 0, 0};
      for (int k = 0; k < t->nrow; k++) w[k] = (uint16_t)p->w[s0][k];
      for (int y = 0; y < VCN_LC_Y; y++) {
        vcn_lc_row_begin(b, z, y, 128), vcn_lc_row_begin(e, z, y, 0);
        const size_t ri = VCN_LC_IDX(z, y, 0), o = ((size_t)z * VCN_LC_Y + y) * VCN_LC_D;
        uint8_t *restrict br = b + ri, *restrict er = e + ri;
        if(!cf->fit_session)memcpy(br, src + o, VCN_LC_D);
        if(ng_screen_work&&ng_screen_size==NV){
          /* Original source padding and taps; only residual history at the
             outer boundary of the requested warm-up union is reset. */
          memset(Pa+o,0,2*VCN_LC_D);vcn_lc_apre(e,ri,ap);
          for(int x=0;x<VCN_LC_D;){
            if(!ng_screen_work[o+x]){x++;continue;}
            int begin=x;while(x<VCN_LC_D&&ng_screen_work[o+x])x++;
            vcn_lc_pre_span(br+begin,&pl[s0],(uint16_t)p->bias[s0],pre+begin,x-begin);
            if(cf->fit_feature)for(int j=begin;j<x;j++)pre[j]=(uint16_t)(pre[j]+(unsigned)(uint16_t)feature_weights[s0]*cf->fit_feature[o+j]);
            vcn_lc_pass1_span(pre+begin,br+begin,w,Pa+o+begin,er+begin,x-begin);
            for(int j=begin;j<x;j++)if(ng_screen_targets[o+j]){h[0][ap[j]]++;sa[0][ap[j]]+=(int16_t)(uint16_t)(br[j]*256u-Pa[o+j]);}
          }
          vcn_lc_row_end(b,z,y);vcn_lc_row_end(e,z,y);continue;
        }
        VCN_LC_PRE(br, &pl[s0], (uint16_t)p->bias[s0], pre);
        if(ng_simple_activity_values)memcpy(ap,ng_simple_activity_values+o,sizeof(ap));
        else vcn_lc_apre(e, ri, ap);
        uint16_t *restrict Q = Pa + o;
        // Add optional inputs before the native SIMD predictor/residual pass.
        // Only low16 bits survive; modular additions commute exactly. This
        // avoids computing residuals twice and a scalar64-bit multiply/voxel.
        if(cf->fit_feature)for(int x=0;x<VCN_LC_D;x++){
          pre[x]=(uint16_t)(pre[x]+(unsigned)(uint16_t)feature_weights[s0]*cf->fit_feature[o+x]);
        }
        if(cf->context_weights)for(int x=0;x<VCN_LC_D;x++)for(int k=0;k<8;k++)
          pre[x]=(uint16_t)(pre[x]+(int64_t)cf->context_weights[s0*8+k]*ng_context_input(b,ri+x,s0,k));
        if (cf->base_prediction) for (int x=0;x<VCN_LC_D;x++) {
          pre[x]=(uint16_t)(pre[x]+256u*cf->base_prediction[o+x]);
        }
        VCN_LC_PASS1(pre, br, w, Q, er);
        /* activity histogram and the sums of 256 v - P per activity value */
        for (int x = 0; x < VCN_LC_D; x++) {
          const int a = ap[x], q = 0;
          h[q][a]++, sa[q][a] += (int16_t)(uint16_t)(br[x] * 256u - Q[x]);
        }
        vcn_lc_row_end(b, z, y), vcn_lc_row_end(e, z, y);
      }
    }
  }
  VCN_LC_TICK(10);
  /* classes: quantiles of the bias-free activity; biases: mean of 256 v - P per class (exact integer sums,
     so the order of the sums does not matter) */
  {
    uint64_t acc = 0, calibration_count=NV;
    if(ng_screen_targets&&ng_screen_size==NV){calibration_count=0;for(int a=0;a<VCN_LC_AMAX;a++)calibration_count+=h[0][a];}
    int k = 0;
    for (int v = 0; v < VCN_LC_AMAX && k < K - 1; v++) {
      acc += h[0][v];
      while (k < K - 1 && acc * (uint64_t)K >= (uint64_t)(k + 1) * calibration_count) p->thr[k++] = (uint16_t)v;
    }
    while (k < K - 1) p->thr[k++] = VCN_LC_AMAX - 1;
    vcn_lc_mklut(p);
    int64_t sum[VCN_LC_MAXK] = {0}, n[VCN_LC_MAXK] = {0};
    for (int a = 0; a < VCN_LC_AMAX; a++) sum[p->lut[a]] += sa[0][a], n[p->lut[a]] += h[0][a];
    for (int c = 0; c < K; c++) {
      double m = n[c] ? floor((double)sum[c] / (double)n[c] + 0.5) : 0.0;
      m = m < -32768 ? -32768 : m > 32767 ? 32767 : m;
      p->cb[c] = cf->single_model?0:(int32_t)m;
    }
  }
  VCN_LC_TICK(11);
  /* symbols, in raster order: the class from the activity of the biased residual magnitudes of earlier
     rows (the decoder has them as soon as it has those rows' symbols: |e| = (symbol + 1) >> 1) */
  {
    memset(e, 0, VCN_LC_NB);
    if(cf->export_activity_histogram)memset(cf->export_activity_histogram,0,VCN_LC_AMAX*sizeof(uint32_t));
    for (int z = 0; z < VCN_LC_Z; z++)
      for (int y = 0; y < VCN_LC_Y; y++) {
        vcn_lc_row_begin(e, z, y, 0);
        const size_t ri = VCN_LC_IDX(z, y, 0), o = ((size_t)z * VCN_LC_Y + y) * VCN_LC_D;
        uint8_t *restrict er = e + ri, *restrict cr = cls + (row_classes?0:o), *restrict sr = sym + o;
        const uint8_t *restrict vr = src + o;
        uint16_t *restrict Q = Pa + o; /* += the class bias */
        if(ng_screen_work&&ng_screen_size==NV){
          uint16_t a[VCN_LC_D];vcn_lc_apre(e,ri,a);
          for(int x=0;x<VCN_LC_D;x++){
            if(!ng_screen_work[o+x]){sr[x]=0;cr[x]=0;continue;}
            cr[x]=p->lut[a[x]];Q[x]=(uint16_t)(Q[x]+p->cb[cr[x]]);
            if(cf->export_prediction)cf->export_prediction[o+x]=Q[x];
            if(cf->export_activity)cf->export_activity[o+x]=a[x];
            sr[x]=ng_prediction_symbol(vr[x],Q[x]);er[x]=(sr[x]+1u)>>1;
          }
          vcn_lc_row_end(e,z,y);continue;
        }
        if(ng_simple_activity_values){
          if(cf->export_activity)memcpy(cf->export_activity+o,ng_simple_activity_values+o,2*VCN_LC_D);
          for(int x=0;x<VCN_LC_D;x++){cr[x]=p->lut[ng_simple_activity_values[o+x]];Q[x]=(uint16_t)(Q[x]+p->cb[cr[x]]);}
        }else if(cf->export_activity && !ng_soft_classes){
#if VCN_HAVE_AVX2
          if(vcn_use_avx2() && p->K<=16)ng_cls_export_avx2(e,ri,p,Q,cr,cf->export_activity+o);
          else
#endif
          {
          uint16_t*a=cf->export_activity+o;vcn_lc_apre(e,ri,a);
          for(int x=0;x<VCN_LC_D;x++){
            cr[x]=p->lut[a[x]];Q[x]=(uint16_t)(Q[x]+p->cb[cr[x]]);
          }
          }
        }else{
          if(cf->export_activity)vcn_lc_apre(e,ri,cf->export_activity+o);
          VCN_LC_CLS(e, ri, p, Q, cr);
        }
        if(cf->export_activity_histogram)for(int x=0;x<VCN_LC_D;x++)cf->export_activity_histogram[cf->export_activity[o+x]]++;
        if (cf->export_prediction) memcpy(cf->export_prediction+o,Q,2*VCN_LC_D);
        VCN_LC_SYMS(Q, vr, sr, er);
        if (!export_sym) for (int x = 0; x < VCN_LC_D; x++) cnt[cr[x]][sr[x]]++;
        vcn_lc_row_end(e, z, y);
      }
  }
  VCN_LC_TICK(12);
  pos=vcn_lc_writepar(p,cf,hdr);
  /* Optional entropy-backend handoff: same fitted predictor and coded symbols.
     Stop before writing any rANS probability tables. */
  if (export_sym) {
    if (pos > cap) st = VCN_ERR_CAP;
    else {
      memcpy(out,hdr,pos); *out_n=pos;
      /* Caller-owned symbol/class storage is filled directly above. */
    }
    goto done;
  }
entropy_ready:
  for (int k = 0; k < K; k++) {
    uint16_t fr[256];
    vcn_lc_norm(cnt[k], fr, tb);
    pos+=vcn_lc_write_table(hdr+pos,fr);
    uint32_t cum = 0;
    for (unsigned s = 0; s < 256; s++) {
      const uint32_t f = fr[s];
      vcn_lc_esym *E = &es[k][s]; /* reciprocal form (ryg_rans), exact for states below 2^31 */
      vcn_lc_init_esym(E,f,cum,M,tb);
      cum += f;
    }
  }
  VCN_LC_TICK(13);
  if (vcn_lc_mksched(vcn_lc_delta(t), &sc)) {
    st = VCN_ERR_NOMEM;
    goto done;
  }
  if(cf->raster_entropy) {
    sc.ng=(VCN_LC_Y*VCN_LC_Z+VCN_LC_NL-1)/VCN_LC_NL;
    for(int g=0;g<sc.ng;g++) {
      int count=VCN_LC_Y*VCN_LC_Z-g*VCN_LC_NL;
      sc.n[g]=(uint8_t)(count<VCN_LC_NL?count:VCN_LC_NL);
      for(int j=0;j<sc.n[g];j++)sc.row[g][j]=(uint16_t)(g*VCN_LC_NL+j);
    }
  }
  {
    size_t nw = 0;
    uint32_t S[VCN_LC_NS];
    for (int k = 0; k < VCN_LC_NS; k++) S[k] = VCN_LC_RL;
    for (int g = sc.ng; g-- > 0;)
      for (int j = sc.n[g]; j-- > 0;) {
        const size_t o = (size_t)sc.row[g][j] * VCN_LC_D;
        const uint8_t *restrict cr = coded_cls + o, *restrict sr = coded_sym + o;
#if VCN_HAVE_AVX2
        if(!cf->import_targets && !ng_runtime_only && VCN_LC_D%16==0 && ng_use_avx512() && __builtin_cpu_supports("avx512bw")){
          nw=ng_rans_encode_row_wide(cr,sr,VCN_LC_D,&es[0][0],S,words,nw);
          continue;
        }
#endif
        for (int x = VCN_LC_D; x-- > 0;) {
          if(cf->import_targets&&!cf->import_targets[o+x])continue;
          const vcn_lc_esym *E = &es[cr[x]][sr[x]];
          S[x & (VCN_LC_NS - 1)] = vcn_lc_put_esym(S[x & (VCN_LC_NS - 1)],E,words,&nw);
        }
      }
    VCN_LC_TICK(14);
    const size_t total = pos + 4 * VCN_LC_NS + 2 * nw;
    if (total > cap) st = VCN_ERR_CAP;
    else {
      memcpy(out, hdr, pos);
      for (int k = 0; k < VCN_LC_NS; k++) vcn_wr32(out + pos + 4 * k, S[k]);
      uint8_t *o = out + pos + 4 * VCN_LC_NS;
      for (size_t j = 0; j < nw; j++) vcn_wr16(o + 2 * j, words[nw - 1 - j]);
      *out_n = total;
    }
  }
done:
  free(sc.row), free(sc.n);
  if(!cf->fit_session)free(b);
  if(!export_sym)free(sym);
  if(!export_cls)free(cls);
  free(hdr), free(words), free(es), free(cnt), free(h), free(sa), free(Pa), free(e), free(p);
  return st;
}

static vcn_status vcn_lc_encode(const uint8_t *src, const vcn_lc_cfg *cf, uint8_t *out, size_t cap, size_t *out_n) {
  return vcn_lc_encode_ex(src,cf,out,cap,out_n,NULL,NULL);
}

/* ---- decoding ----
   Per group of up to 32 rows: (1) row by row, the previous-row part of each prediction plus its class
   bias (A) and the class; (2) the group's symbols, in row order, 32 rANS states, and from them the rows'
   residual magnitudes for later rows' classes; (3) A and the symbols turned lane-major; (4) the lanes
   rebuild the voxels x by x (in-row taps, rounding, symbol); (5) the values back to the rows. */
static uint8_t vcn_lc_sym_one(vcn_lc_dec *D,unsigned c,unsigned lane) {
    const int tb=D->tb;
    uint32_t S=D->S[lane];
    const uint32_t en=D->T[(c<<tb)+(S&((1u<<tb)-1))];
    S=(((en>>8)&0xFFFu)+1u)*(S>>tb)+(en>>20);
    if(S<VCN_LC_RL) {
      if(D->end-D->p>=2)S=(S<<16)|vcn_rd16(D->p),D->p+=2;
      else S<<=16,D->over++;
    }
    D->S[lane]=S;return (uint8_t)en;
}
static void vcn_lc_syms_c(int n, const uint8_t *restrict cl, vcn_lc_dec *D, uint8_t *restrict sym) {
  for (int i = 0; i < n; i++) {
    sym[i]=vcn_lc_sym_one(D,cl[i],i&(VCN_LC_NS-1));
  }
}
/* rebuild: per x, lane j: P = A + in-row taps (mod 2^16), value = rounded P + the signed residual */
typedef struct {
  uint16_t w[3][VCN_LC_NL];  /* in-row weights per lane */
  uint16_t W[3][VCN_LC_NL];  /* the last three values per lane */
} vcn_lc_lanes;
static inline void vcn_lc_rec_c_impl(int nl, const uint16_t (*restrict A)[VCN_LC_NL], const uint8_t (*restrict SY)[VCN_LC_NL], vcn_lc_lanes *L,
                         uint8_t (*restrict V)[VCN_LC_NL],const uint8_t (*restrict DM)[VCN_LC_NL]) {
  for (int x = 0; x < VCN_LC_D; x++)
    for (int j = 0; j < nl; j++) {
      const uint16_t P = DM && DM[x][j] ? (uint16_t)(A[x][j]+(DM[x][j]==2?256u*L->W[0][j]:0)) : (uint16_t)(A[x][j] + (uint16_t)(L->w[0][j] * L->W[0][j]) + (uint16_t)(L->w[1][j] * L->W[1][j]) +
                                    (uint16_t)(L->w[2][j] * L->W[2][j]));
      const int s = SY[x][j], es = (s >> 1) ^ -(s & 1), fm = -(int)VCN_LC_FM(P);
      const int v = (VCN_LC_PP(P) + ((es ^ fm) - fm)) & 255;
      L->W[2][j] = L->W[1][j], L->W[1][j] = L->W[0][j], L->W[0][j] = (uint16_t)v;
      V[x][j] = (uint8_t)v;
    }
}
static void vcn_lc_rec_c(int nl,const uint16_t (*A)[VCN_LC_NL],const uint8_t (*SY)[VCN_LC_NL],vcn_lc_lanes* L,uint8_t (*V)[VCN_LC_NL]) {
  vcn_lc_rec_c_impl(nl,A,SY,L,V,NULL);
}
#if VCN_HAVE_AVX2
#include "ng_entropy_permutation.h"
/* eight table loads (AVX2 gathers are slow on some CPUs) */
#include "ng_gather.h"
/* the same integers, 32 symbols per step as four vectors of eight states; the lanes that need a word
   read theirs in order */
#define NG_EATTR __attribute__((target("avx2,popcnt"),always_inline))
#define NG_ENAME vcn_lc_syms_avx2_impl
#include "ng_entropy_vector.h"
#ifdef NG_HAVE_WIDE_PRE
#define NG_EATTR __attribute__((target("avx512f,avx512bw,avx512vl,popcnt"),always_inline))
#define NG_ENAME ng_syms_register_impl
#define NG_EUNROLL 1
#include "ng_entropy_vector.h"
__attribute__((target("avx512f,avx512bw,avx512vl,popcnt"))) static void ng_syms_register(int n,const uint8_t*cl,vcn_lc_dec*D,uint8_t*sym){
    ng_syms_register_impl(n,cl,D,sym,1);
}
#endif
/* Constant-specialize only the table load. Original NG keeps its existing
 * scalar-load gather; the bank decoder can use the measured hardware gather. */
__attribute__((target("avx2,popcnt"))) static void vcn_lc_syms_avx2(int n,const uint8_t*restrict cl,vcn_lc_dec*D,uint8_t*restrict sym){
  vcn_lc_syms_avx2_impl(n,cl,D,sym,0);
}
__attribute__((target("avx2,popcnt"))) static void vcn_lc_syms_gather_avx2(int n,const uint8_t*restrict cl,vcn_lc_dec*D,uint8_t*restrict sym){
  vcn_lc_syms_avx2_impl(n,cl,D,sym,1);
}
#define NG_NATTR __attribute__((target("avx2"),always_inline))
#define NG_NNAME vcn_lc_rec_avx2_impl
#define NG_NV __m256i
#define NG_NMAX 2
#define NG_NHALVES(N) ((N)>16?2:1)
#define NG_NSTEP 16
#define NG_NLOAD _mm256_loadu_si256
#define NG_NADD _mm256_add_epi16
#define NG_NMUL _mm256_mullo_epi16
#define NG_NSET _mm256_set1_epi16
#define NG_NZERO _mm256_setzero_si256
#define NG_NEQ _mm256_cmpeq_epi16
#define NG_NAND _mm256_and_si256
#define NG_NSLL _mm256_slli_epi16
#define NG_NBLEND _mm256_blendv_epi8
#define NG_NGT _mm256_cmpgt_epi16
#define NG_NXOR _mm256_xor_si256
#define NG_NSRL _mm256_srli_epi16
#define NG_NSUB _mm256_sub_epi16
#define NG_NSRA _mm256_srai_epi16
#define NG_NSTORE _mm256_storeu_si256
#define NG_NBYTES(P) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(P)))
#define NG_NOUTPUT(P,V,N) do { if((N)==1)(V)[1]=(V)[0]; _mm256_storeu_si256((__m256i*)(P),_mm256_permute4x64_epi64(_mm256_packus_epi16((V)[0],(V)[1]),0xD8)); } while(0)
#include "ng_native_recur_vector.h"
#ifdef NG_HAVE_WIDE_PRE
#define NG_NATTR __attribute__((target("avx512f,avx512bw,avx512vl"),always_inline))
#define NG_NNAME vcn_lc_rec_wide_impl
#define NG_NV __m512i
#define NG_NMAX 1
#define NG_NHALVES(N) 1
#define NG_NSTEP 32
#define NG_NADD _mm512_add_epi16
#define NG_NMUL _mm512_mullo_epi16
#define NG_NSET _mm512_set1_epi16
#define NG_NZERO _mm512_setzero_si512
#define NG_NAND _mm512_and_si512
#define NG_NSLL _mm512_slli_epi16
#define NG_NXOR _mm512_xor_si512
#define NG_NSRL _mm512_srli_epi16
#define NG_NSUB _mm512_sub_epi16
#define NG_NSRA _mm512_srai_epi16
#define NG_NLOAD _mm512_loadu_si512
#define NG_NSTORE _mm512_storeu_si512
#define NG_NEQ(A,B) _mm512_maskz_set1_epi16(_mm512_cmpeq_epi16_mask(A,B),-1)
#define NG_NGT(A,B) _mm512_maskz_set1_epi16(_mm512_cmpgt_epi16_mask(A,B),-1)
#define NG_NBLEND(A,B,M) _mm512_mask_blend_epi8(_mm512_movepi8_mask(M),A,B)
#define NG_NBYTES(P) _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(P)))
#define NG_NOUTPUT(P,V,N) _mm256_storeu_si256((__m256i*)(P),_mm512_cvtepi16_epi8((V)[0]))
#include "ng_native_recur_vector.h"
__attribute__((target("avx512f,avx512bw,avx512vl"))) static void vcn_lc_rec_wide(int nl,const uint16_t (*A)[32],const uint8_t (*SY)[32],vcn_lc_lanes*L,uint8_t(*V)[32]){vcn_lc_rec_wide_impl(nl,A,SY,L,V,NULL);}
#endif
__attribute__((target("avx2"))) static void vcn_lc_rec_avx2(int nl,const uint16_t (*A)[VCN_LC_NL],const uint8_t (*SY)[VCN_LC_NL],vcn_lc_lanes* L,uint8_t (*V)[VCN_LC_NL]) {
  vcn_lc_rec_avx2_impl(nl,A,SY,L,V,NULL);
}
__attribute__((target("avx2"))) static void vcn_lc_rec_selected_avx2(int nl,const uint16_t (*A)[VCN_LC_NL],const uint8_t (*SY)[VCN_LC_NL],vcn_lc_lanes* L,uint8_t (*V)[VCN_LC_NL],const uint8_t (*DM)[VCN_LC_NL]) {
  vcn_lc_rec_avx2_impl(nl,A,SY,L,V,DM);
}
/* transposes between the group's rows and its lanes: 8 x 8 blocks of uint16, 16 x 16 blocks of bytes */
__attribute__((target("avx2"))) static void vcn_lc_tr_a_stride(int nl, const uint16_t (*restrict R)[VCN_LC_D], uint16_t *restrict T, size_t stride) {
  for (int j0 = 0; j0 < nl; j0 += 8)
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 8) {
      __m128i a[8], t[8], u[8];
      for (int r = 0; r < 8; r++) a[r] = _mm_loadu_si128((const __m128i *)(R[j0 + r] + x0));
      for (int r = 0; r < 8; r += 2) t[r] = _mm_unpacklo_epi16(a[r], a[r + 1]), t[r + 1] = _mm_unpackhi_epi16(a[r], a[r + 1]);
      for (int r = 0; r < 8; r += 4)
        u[r] = _mm_unpacklo_epi32(t[r], t[r + 2]), u[r + 1] = _mm_unpackhi_epi32(t[r], t[r + 2]),
        u[r + 2] = _mm_unpacklo_epi32(t[r + 1], t[r + 3]), u[r + 3] = _mm_unpackhi_epi32(t[r + 1], t[r + 3]);
      for (int c = 0; c < 4; c++) {
        _mm_storeu_si128((__m128i *)(T + (x0 + 2 * c)*stride + j0), _mm_unpacklo_epi64(u[c], u[c + 4]));
        _mm_storeu_si128((__m128i *)(T + (x0 + 2 * c + 1)*stride + j0), _mm_unpackhi_epi64(u[c], u[c + 4]));
      }
    }
}
__attribute__((target("avx2"))) static void vcn_lc_tr_a(int nl, const uint16_t (*restrict R)[VCN_LC_D], uint16_t (*restrict T)[VCN_LC_NL]) {
  vcn_lc_tr_a_stride(nl,R,T[0],VCN_LC_NL);
}
__attribute__((target("avx2"))) static inline void vcn_lc_tr16(const uint8_t *src, size_t sp, __m128i *o) {
  __m128i a[16], b[16], c[16];
  for (int r = 0; r < 16; r++) a[r] = _mm_loadu_si128((const __m128i *)(src + r * sp));
  for (int i = 0; i < 8; i++) b[2 * i] = _mm_unpacklo_epi8(a[2 * i], a[2 * i + 1]), b[2 * i + 1] = _mm_unpackhi_epi8(a[2 * i], a[2 * i + 1]);
  for (int i = 0; i < 4; i++)
    c[4 * i] = _mm_unpacklo_epi16(b[4 * i], b[4 * i + 2]), c[4 * i + 1] = _mm_unpackhi_epi16(b[4 * i], b[4 * i + 2]),
    c[4 * i + 2] = _mm_unpacklo_epi16(b[4 * i + 1], b[4 * i + 3]), c[4 * i + 3] = _mm_unpackhi_epi16(b[4 * i + 1], b[4 * i + 3]);
  for (int i = 0; i < 2; i++)
    for (int k = 0; k < 4; k++)
      a[8 * i + 2 * k] = _mm_unpacklo_epi32(c[8 * i + k], c[8 * i + 4 + k]), a[8 * i + 2 * k + 1] = _mm_unpackhi_epi32(c[8 * i + k], c[8 * i + 4 + k]);
  for (int m = 0; m < 8; m++) o[2 * m] = _mm_unpacklo_epi64(a[m], a[8 + m]), o[2 * m + 1] = _mm_unpackhi_epi64(a[m], a[8 + m]);
}
__attribute__((target("avx2"))) static void vcn_lc_tr_y_width(int nl, const uint8_t (*restrict R)[VCN_LC_D], uint8_t (*restrict T)[VCN_LC_NL],int width) {
  __m128i o[16];
  for (int j0 = 0; j0 < nl; j0 += 16)
    for (int x0 = 0; x0 < width; x0 += 16) {
      vcn_lc_tr16(R[j0] + x0, VCN_LC_D, o);
      for (int c = 0; c < 16; c++) _mm_storeu_si128((__m128i *)(T[x0 + c] + j0), o[c]);
    }
}
__attribute__((target("avx2"))) static void vcn_lc_tr_y(int nl,const uint8_t R[][VCN_LC_D],uint8_t T[][VCN_LC_NL]){vcn_lc_tr_y_width(nl,R,T,VCN_LC_D);}
/* lanes back to the rows of b */
__attribute__((target("avx2"))) static void vcn_lc_tr_v(int nl, const uint8_t (*restrict T)[VCN_LC_NL], uint8_t *const *rows) {
  __m128i o[16];
  for (int j0 = 0; j0 < nl; j0 += 16) {
    const int m = nl - j0 < 16 ? nl - j0 : 16;
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 16) {
      vcn_lc_tr16(T[x0] + j0, VCN_LC_NL, o);
      for (int c = 0; c < m; c++) _mm_storeu_si128((__m128i *)(rows[j0 + c] + x0), o[c]);
    }
  }
}
#endif
#if VCN_HAVE_NEON
/* NEON forms of the decoder's kernels (the same integers as the C forms). The symbols: 32 per step as eight
   vectors of four states; table loads one lane at a time; the lanes that need a word read theirs in order
   (lane l takes word popcount(mask & ((1 << l) - 1)), by a byte table lookup) */
static const uint8_t vcn_lc_perm4[16][16] = {
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},   {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},   {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0, 1, 2, 3},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},   {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3},
    {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3},   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},   {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7},   {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7},   {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
static void vcn_lc_syms_neon(int n, const uint8_t *restrict cl, vcn_lc_dec *D, uint8_t *restrict sym) {
  const int tb = D->tb;
  const int32x4_t ptb = vdupq_n_s32(tb), ntb = vdupq_n_s32(-tb);
  const uint32x4_t kM = vdupq_n_u32((1u << tb) - 1u), kF = vdupq_n_u32(0xFFF), k1 = vdupq_n_u32(1), kL = vdupq_n_u32(VCN_LC_RL);
  static const uint32_t kb[4] = {1, 2, 4, 8};
  const uint32x4_t kbit = vld1q_u32(kb);
  uint32x4_t S[8];
  for (int h = 0; h < 8; h++) S[h] = vld1q_u32(D->S + 4 * h);
  const uint32_t *T = D->T;
  const uint8_t *p = D->p, *end = D->end;
  size_t over = D->over;
  for (int i = 0; i < n; i += 32) {
    const uint8x16_t ca = vld1q_u8(cl + i), cb = vld1q_u8(cl + i + 16);
    const uint16x8_t cw[4] = {vmovl_u8(vget_low_u8(ca)), vmovl_high_u8(ca), vmovl_u8(vget_low_u8(cb)), vmovl_high_u8(cb)};
    uint16x4_t y[8];
    for (int h = 0; h < 8; h++) {
      const uint32x4_t cv = (h & 1) ? vmovl_high_u16(cw[h >> 1]) : vmovl_u16(vget_low_u16(cw[h >> 1]));
      const uint32x4_t ix = vaddq_u32(vshlq_u32(cv, ptb), vandq_u32(S[h], kM));
      uint32x4_t en = vdupq_n_u32(0);
      en = vld1q_lane_u32(T + vgetq_lane_u32(ix, 0), en, 0);
      en = vld1q_lane_u32(T + vgetq_lane_u32(ix, 1), en, 1);
      en = vld1q_lane_u32(T + vgetq_lane_u32(ix, 2), en, 2);
      en = vld1q_lane_u32(T + vgetq_lane_u32(ix, 3), en, 3);
      uint32x4_t s2 = vmlaq_u32(vshrq_n_u32(en, 20), vaddq_u32(vandq_u32(vshrq_n_u32(en, 8), kF), k1), vshlq_u32(S[h], ntb));
      const uint32x4_t m = vcltq_u32(s2, kL);
      const unsigned bits = vaddvq_u32(vandq_u32(m, kbit));
      if (end - p >= 8) { /* no branch on the mask: an empty mask reads nothing */
        const uint32x4_t w = vmovl_u16(vreinterpret_u16_u8(vld1_u8(p)));
        const uint32x4_t wv = vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(w), vld1q_u8(vcn_lc_perm4[bits])));
        s2 = vbslq_u32(m, vorrq_u32(vshlq_n_u32(s2, 16), wv), s2);
        p += 2 * (unsigned)__builtin_popcount(bits);
      } else if (bits) {
        uint32_t sv[4];
        vst1q_u32(sv, s2);
        for (int k = 0; k < 4; k++)
          if (bits >> k & 1u) {
            if (end - p >= 2) sv[k] = (sv[k] << 16) | vcn_rd16(p), p += 2;
            else sv[k] <<= 16, over++;
          }
        s2 = vld1q_u32(sv);
      }
      S[h] = s2;
      y[h] = vmovn_u32(en);
    }
    for (int q = 0; q < 4; q++) vst1_u8(sym + i + 8 * q, vmovn_u16(vcombine_u16(y[2 * q], y[2 * q + 1])));
  }
  for (int h = 0; h < 8; h++) vst1q_u32(D->S + 4 * h, S[h]);
  D->p = p, D->over = over;
}
/* rebuild, 8 lanes per vector: NH vectors (2 when the group has at most 16 rows, else 4) */
#define VCN_LC_REC_NEON(NH)                                                                                                   \
  do {                                                                                                                         \
    uint16x8_t w0[NH], w1[NH], w2[NH], W0[NH], W1[NH], W2[NH];                                                                 \
    for (int h = 0; h < NH; h++)                                                                                               \
      w0[h] = vld1q_u16(L->w[0] + 8 * h), w1[h] = vld1q_u16(L->w[1] + 8 * h), w2[h] = vld1q_u16(L->w[2] + 8 * h),              \
      W0[h] = vld1q_u16(L->W[0] + 8 * h), W1[h] = vld1q_u16(L->W[1] + 8 * h), W2[h] = vld1q_u16(L->W[2] + 8 * h);              \
    for (int x = 0; x < VCN_LC_D; x++)                                                                                         \
      for (int h = 0; h < NH; h++) {                                                                                           \
        uint16x8_t P = vmlaq_u16(vmlaq_u16(vmlaq_u16(vld1q_u16(A[x] + 8 * h), W0[h], w0[h]), W1[h], w1[h]), W2[h], w2[h]); \
        if(DM) {                                                                                                             \
          const uint16x8_t dm=vmovl_u8(vld1_u8(DM[x]+8*h));                                                                    \
          const uint16x8_t dp=vaddq_u16(vld1q_u16(A[x]+8*h),vandq_u16(vceqq_u16(dm,vdupq_n_u16(2)),vshlq_n_u16(W0[h],8)));    \
          P=vbslq_u16(vcgtq_u16(dm,vdupq_n_u16(0)),dp,P);                                                                      \
        }                                                                                                                    \
        const uint16x8_t sy = vmovl_u8(vld1_u8(SY[x] + 8 * h));                                                                \
        const int16x8_t es =                                                                                                   \
            veorq_s16(vreinterpretq_s16_u16(vshrq_n_u16(sy, 1)), vnegq_s16(vreinterpretq_s16_u16(vandq_u16(sy, k1))));        \
        const int16x8_t fm = vshrq_n_s16(vreinterpretq_s16_u16(vshlq_n_u16(P, 8)), 15);                                      \
        const uint16x8_t vv = vandq_u16(                                                                                       \
            vaddq_u16(vshrq_n_u16(vaddq_u16(P, k128), 8), vreinterpretq_u16_s16(vsubq_s16(veorq_s16(es, fm), fm))), kB);       \
        W2[h] = W1[h], W1[h] = W0[h], W0[h] = vv;                                                                              \
        vst1_u8(V[x] + 8 * h, vmovn_u16(vv));                                                                                  \
      }                                                                                                                        \
    for (int h = 0; h < NH; h++)                                                                                               \
      vst1q_u16(L->W[0] + 8 * h, W0[h]), vst1q_u16(L->W[1] + 8 * h, W1[h]), vst1q_u16(L->W[2] + 8 * h, W2[h]);                 \
  } while (0)
static inline void vcn_lc_rec_neon_impl(int nl, const uint16_t (*restrict A)[VCN_LC_NL], const uint8_t (*restrict SY)[VCN_LC_NL], vcn_lc_lanes *L,
                            uint8_t (*restrict V)[VCN_LC_NL],const uint8_t (*restrict DM)[VCN_LC_NL]) {
  const uint16x8_t k128 = vdupq_n_u16(128), k1 = vdupq_n_u16(1), kB = vdupq_n_u16(255);
  if (nl > 16) VCN_LC_REC_NEON(4);
  else VCN_LC_REC_NEON(2);
}
static void vcn_lc_rec_neon(int nl,const uint16_t (*A)[VCN_LC_NL],const uint8_t (*SY)[VCN_LC_NL],vcn_lc_lanes* L,uint8_t (*V)[VCN_LC_NL]) {
  vcn_lc_rec_neon_impl(nl,A,SY,L,V,NULL);
}
#undef VCN_LC_REC_NEON
/* transposes between the group's rows and its lanes, 8 x 8 blocks of uint16 and 16 x 16 blocks of bytes:
   element pairs, then groups of 2, 4 (and 8) elements */
static void vcn_lc_tr_a_neon(int nl, const uint16_t (*restrict R)[VCN_LC_D], uint16_t (*restrict T)[VCN_LC_NL]) {
  for (int j0 = 0; j0 < nl; j0 += 8)
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 8) {
      uint16x8_t a[8];
      uint32x4_t b[8];
      for (int r = 0; r < 8; r++) a[r] = vld1q_u16(R[j0 + r] + x0);
      for (int i = 0; i < 4; i++) { /* b[2i + q]: rows 2i, 2i + 1 at x = q (mod 2) */
        const uint16x8x2_t t = vtrnq_u16(a[2 * i], a[2 * i + 1]);
        b[2 * i] = vreinterpretq_u32_u16(t.val[0]), b[2 * i + 1] = vreinterpretq_u32_u16(t.val[1]);
      }
      for (int i = 0; i < 2; i++) /* a[4i + q]: rows 4i..4i+3 at x = q (low half) and q + 4 (high half) */
        for (int par = 0; par < 2; par++) {
          const uint32x4x2_t t = vtrnq_u32(b[4 * i + par], b[4 * i + 2 + par]);
          a[4 * i + par] = vreinterpretq_u16_u32(t.val[0]), a[4 * i + 2 + par] = vreinterpretq_u16_u32(t.val[1]);
        }
      for (int q = 0; q < 4; q++) {
        vst1q_u16(T[x0 + q] + j0, vcombine_u16(vget_low_u16(a[q]), vget_low_u16(a[4 + q])));
        vst1q_u16(T[x0 + q + 4] + j0, vcombine_u16(vget_high_u16(a[q]), vget_high_u16(a[4 + q])));
      }
    }
}
/* o[c] = column c of the 16 x 16 block of bytes at src (row pitch sp) */
static inline void vcn_lc_tr16_neon(const uint8_t *src, size_t sp, uint8x16_t *o) {
  uint8x16_t a[16];
  uint16x8_t b[16];
  uint32x4_t c[16];
  for (int r = 0; r < 16; r++) a[r] = vld1q_u8(src + (size_t)r * sp);
  for (int i = 0; i < 8; i++) { /* b[2i + q]: rows 2i, 2i + 1 at x = q (mod 2) */
    const uint8x16x2_t t = vtrnq_u8(a[2 * i], a[2 * i + 1]);
    b[2 * i] = vreinterpretq_u16_u8(t.val[0]), b[2 * i + 1] = vreinterpretq_u16_u8(t.val[1]);
  }
  for (int i = 0; i < 4; i++) /* c[4i + q]: rows 4i..4i+3 at x = q (mod 4) */
    for (int par = 0; par < 2; par++) {
      const uint16x8x2_t t = vtrnq_u16(b[4 * i + par], b[4 * i + 2 + par]);
      c[4 * i + par] = vreinterpretq_u32_u16(t.val[0]), c[4 * i + 2 + par] = vreinterpretq_u32_u16(t.val[1]);
    }
  for (int i = 0; i < 2; i++) /* a[8i + q]: rows 8i..8i+7 at x = q (low half) and q + 8 (high half) */
    for (int q = 0; q < 4; q++) {
      const uint32x4x2_t t = vtrnq_u32(c[8 * i + q], c[8 * i + 4 + q]);
      a[8 * i + q] = vreinterpretq_u8_u32(t.val[0]), a[8 * i + 4 + q] = vreinterpretq_u8_u32(t.val[1]);
    }
  for (int q = 0; q < 8; q++)
    o[q] = vcombine_u8(vget_low_u8(a[q]), vget_low_u8(a[8 + q])), o[q + 8] = vcombine_u8(vget_high_u8(a[q]), vget_high_u8(a[8 + q]));
}
static void vcn_lc_tr_y_neon(int nl, const uint8_t (*restrict R)[VCN_LC_D], uint8_t (*restrict T)[VCN_LC_NL]) {
  uint8x16_t o[16];
  for (int j0 = 0; j0 < nl; j0 += 16)
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 16) {
      vcn_lc_tr16_neon(R[j0] + x0, VCN_LC_D, o);
      for (int c = 0; c < 16; c++) vst1q_u8(T[x0 + c] + j0, o[c]);
    }
}
/* lanes back to the rows of b */
static void vcn_lc_tr_v_neon(int nl, const uint8_t (*restrict T)[VCN_LC_NL], uint8_t *const *rows) {
  uint8x16_t o[16];
  for (int j0 = 0; j0 < nl; j0 += 16) {
    const int m = nl - j0 < 16 ? nl - j0 : 16;
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 16) {
      vcn_lc_tr16_neon(T[x0] + j0, VCN_LC_NL, o);
      for (int c = 0; c < m; c++) vst1q_u8(rows[j0 + c] + x0, o[c]);
    }
  }
}
#endif
static vcn_status vcn_lc_readpar(const uint8_t **cursor, size_t n, vcn_lc_par **result) {
  const uint8_t *p=*cursor;
  const uint8_t *end = p + n;
  if (n < 1) return VCN_ERR_CORRUPT;
  const int has_context=p[0]==4;
  if (p[0] != VCN_LC_VERSION&&!has_context) return VCN_ERR_VERSION;
  p++, n--;
  if (n < 5) return VCN_ERR_CORRUPT;
  const int r0 = p[0], r1 = (int8_t)p[1], r2 = (int8_t)p[2], K = p[3], tb = p[4];
  if (r0 < 1 || r0 > 3 || r1 < -1 || r1 > 3 || r2 < -1 || r2 > 3 || K < 2 || K > VCN_LC_MAXK || tb < 10 || tb > 12)
    return VCN_ERR_CORRUPT;
  p += 5;
  vcn_lc_par *P = (vcn_lc_par *)calloc(1, sizeof(vcn_lc_par));
  if (!P) return VCN_ERR_NOMEM;
  vcn_lc_build(&P->t, r0, r1, r2);
  P->K = K, P->tb = tb;P->has_context=has_context;
  const vcn_lc_taps *t = &P->t;
  const size_t need = 2 * (2 * (size_t)t->nt + 2) + 2 * (size_t)(K - 1) + 2 * (size_t)K + (has_context?32:0);
  if ((size_t)(end - p) < need) {
    free(P);
    return VCN_ERR_CORRUPT;
  }
  for (int s2 = 0; s2 < 2; s2++) {
    for (int k = 0; k < t->nt; k++) P->w[s2][k] = (int16_t)vcn_rd16(p), p += 2;
    P->bias[s2] = (int32_t)vcn_rd16(p), p += 2;
  }
  if(has_context)for(int b=0;b<2;b++)for(int k=0;k<8;k++){P->context[b][k]=(int16_t)vcn_rd16(p);p+=2;}
  int bad_thr = 0; /* the encoder writes thresholds below VCN_LC_AMAX; the SIMD class path needs them there */
  for (int k = 0; k < K - 1; k++) P->thr[k] = (uint16_t)vcn_rd16(p), p += 2, bad_thr |= P->thr[k] >= VCN_LC_AMAX;
  for (int k = 0; k < K; k++) P->cb[k] = (int16_t)vcn_rd16(p), p += 2;
  if (bad_thr) {
    free(P);
    return VCN_ERR_CORRUPT;
  }
  vcn_lc_mklut(P);
  *cursor=p; *result=P;
  return VCN_OK;
}
/* Delta's previous-row contribution, in the same fixed-point units as NG.
   DM=0: NG in-row taps; DM=1: no in-row tap; DM=2: +256 * left value. */
static int vcn_lc_select_pre_c(const uint8_t* b,size_t ri,int z,int y,const uint8_t* classes,
                           const uint8_t* selectors,uint16_t* A,uint8_t* DM) {
  int any=0;
  for(int x=0;x<VCN_LC_D;x++) {
    unsigned m=selectors[classes[x]];DM[x]=0;
    if(!m)continue;
    any=1;unsigned dx=(m&1) && x>0;
    if(!y)m&=~2u;if(!z)m&=~4u;
    const uint8_t* r=b+ri+x;
    int v=((m&2)?r[-VCN_LC_RP]:0)+((m&4)?r[-VCN_LC_SP]:0)-((m&6)==6?r[-VCN_LC_RP-VCN_LC_SP]:0);
    if(dx)v-=((m&2)?r[-VCN_LC_RP-1]:0)+((m&4)?r[-VCN_LC_SP-1]:0)-((m&6)==6?r[-VCN_LC_RP-VCN_LC_SP-1]:0);
    A[x]=(uint16_t)(v*256);DM[x]=(uint8_t)(dx?2:1);
  }
  return any;
}
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static int vcn_lc_select_pre_avx2(const uint8_t* b,size_t ri,int z,int y,
    const uint8_t* classes,const uint8_t* selectors,int K,uint16_t* A,uint8_t* DM) {
  uint8_t lut[16]={0};memcpy(lut,selectors,(size_t)K);
  const __m128i table=_mm_loadu_si128((const __m128i*)lut);
  const __m256i zero=_mm256_setzero_si256(),one=_mm256_set1_epi16(1),two=_mm256_set1_epi16(2),four=_mm256_set1_epi16(4);
  const __m256i valid=_mm256_set1_epi16((short)((y?2:0)|(z?4:0)));
  int any=0;
  for(int x=0;x<VCN_LC_D;x+=16) {
    const __m128i s=_mm_shuffle_epi8(table,_mm_loadu_si128((const __m128i*)(classes+x)));
    if(_mm_movemask_epi8(_mm_cmpeq_epi8(s,_mm_setzero_si128()))==65535) {
      _mm_storeu_si128((__m128i*)(DM+x),_mm_setzero_si128());continue;
    }
    any=1;
    const __m256i orig=_mm256_cvtepu8_epi16(s),m=_mm256_and_si256(orig,valid);
    const __m256i my=_mm256_cmpeq_epi16(_mm256_and_si256(m,two),two),mz=_mm256_cmpeq_epi16(_mm256_and_si256(m,four),four);
    const __m256i myz=_mm256_and_si256(my,mz);
    __m256i dx=_mm256_cmpeq_epi16(_mm256_and_si256(orig,one),one);
    if(x==0)dx=_mm256_insert_epi16(dx,0,0);
    const uint8_t* r=b+ri+x;
#define VCN_SEL_LOAD(off) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(r+(off))))
#define VCN_SEL_PRE(off) _mm256_sub_epi16(_mm256_add_epi16(_mm256_and_si256(my,VCN_SEL_LOAD(-VCN_LC_RP+(off))),_mm256_and_si256(mz,VCN_SEL_LOAD(-VCN_LC_SP+(off)))),_mm256_and_si256(myz,VCN_SEL_LOAD(-VCN_LC_RP-VCN_LC_SP+(off))))
    const __m256i v=_mm256_sub_epi16(VCN_SEL_PRE(0),_mm256_and_si256(dx,VCN_SEL_PRE(-1)));
    const __m256i isdelta=_mm256_cmpgt_epi16(orig,zero);
    const __m256i q=_mm256_blendv_epi8(_mm256_loadu_si256((const __m256i*)(A+x)),_mm256_slli_epi16(v,8),isdelta);
    _mm256_storeu_si256((__m256i*)(A+x),q);
    const __m256i kind=_mm256_and_si256(isdelta,_mm256_sub_epi16(one,dx));
    _mm_storeu_si128((__m128i*)(DM+x),_mm_packus_epi16(_mm256_castsi256_si128(kind),_mm256_extracti128_si256(kind,1)));
#undef VCN_SEL_PRE
#undef VCN_SEL_LOAD
  }
  return any;
}
#elif VCN_HAVE_NEON
static int vcn_lc_select_pre_neon(const uint8_t* b,size_t ri,int z,int y,const uint8_t* classes,
    const uint8_t* selectors,int K,uint16_t* A,uint8_t* DM) {
  uint8_t lut[16]={0};memcpy(lut,selectors,(size_t)K);const uint8x16_t table=vld1q_u8(lut);
  const uint16x8_t one=vdupq_n_u16(1),two=vdupq_n_u16(2),four=vdupq_n_u16(4),valid=vdupq_n_u16((y?2:0)|(z?4:0));
  int any=0;
  for(int x=0;x<VCN_LC_D;x+=8) {
    const uint16x8_t orig=vmovl_u8(vqtbl1_u8(table,vld1_u8(classes+x))),m=vandq_u16(orig,valid);
    if(vmaxvq_u16(orig)==0){vst1_u8(DM+x,vdup_n_u8(0));continue;}
    any=1;
    const uint16x8_t my=vceqq_u16(vandq_u16(m,two),two),mz=vceqq_u16(vandq_u16(m,four),four),myz=vandq_u16(my,mz);
    uint16x8_t dx=vceqq_u16(vandq_u16(orig,one),one);if(x==0)dx=vsetq_lane_u16(0,dx,0);
    const uint8_t* r=b+ri+x;
#define VCN_SEL_LOAD(off) vmovl_u8(vld1_u8(r+(off)))
#define VCN_SEL_PRE(off) vsubq_u16(vaddq_u16(vandq_u16(my,VCN_SEL_LOAD(-VCN_LC_RP+(off))),vandq_u16(mz,VCN_SEL_LOAD(-VCN_LC_SP+(off)))),vandq_u16(myz,VCN_SEL_LOAD(-VCN_LC_RP-VCN_LC_SP+(off))))
    const uint16x8_t v=vsubq_u16(VCN_SEL_PRE(0),vandq_u16(dx,VCN_SEL_PRE(-1))),isd=vcgtq_u16(orig,vdupq_n_u16(0));
    vst1q_u16(A+x,vbslq_u16(isd,vshlq_n_u16(v,8),vld1q_u16(A+x)));
    vst1_u8(DM+x,vmovn_u16(vandq_u16(isd,vsubq_u16(one,dx))));
#undef VCN_SEL_PRE
#undef VCN_SEL_LOAD
  }
  return any;
}
#endif
static int vcn_lc_select_pre(const uint8_t* b,size_t ri,int z,int y,const uint8_t* classes,const uint8_t* selectors,
    int K,uint16_t* A,uint8_t* DM,int scalar) {
#if VCN_HAVE_AVX2
  if(!scalar && K<=16 && vcn_use_avx2())return vcn_lc_select_pre_avx2(b,ri,z,y,classes,selectors,K,A,DM);
#elif VCN_HAVE_NEON
  if(!scalar && K<=16)return vcn_lc_select_pre_neon(b,ri,z,y,classes,selectors,K,A,DM);
#endif
  return vcn_lc_select_pre_c(b,ri,z,y,classes,selectors,A,DM);
}
static vcn_status vcn_lc_decode_core(const uint8_t *p, size_t n, uint8_t *dst,
                                  ng_symbol_reader symbol_reader, void *reader_context,
                                  const uint8_t* selectors,int force_scalar,int block_size,const uint8_t* fixed_symbols,const uint8_t* block_contexts) {
  const uint8_t *end=p+n;
  vcn_lc_par *P=NULL;
  vcn_status parse=vcn_lc_readpar(&p,n,&P);
  if(parse!=VCN_OK) return parse;
  const vcn_lc_taps *t=&P->t;
  const int K=P->K, tb=P->tb;
  const uint32_t M=1u<<tb;
  const int no_activity=fixed_symbols!=NULL || block_contexts!=NULL;
  if(selectors)for(int c=0;c<K;c++)if(selectors[c]>7){free(P);return VCN_ERR_CORRUPT;}
  if (symbol_reader && !block_size) {
    /* Raster-order backend, using the original predictor/class/reconstruction
       kernels. No encoded class array or reference to undecoded source data. */
    if (p != end) { free(P); return VCN_ERR_CORRUPT; }
    uint8_t *b=(uint8_t*)malloc(VCN_LC_NB), *e=(uint8_t*)malloc(VCN_LC_NB);
    if (!b || !e) { free(b); free(e); free(P); return VCN_ERR_NOMEM; }
    memset(b,128,VCN_LC_NB); memset(e,0,VCN_LC_NB);
    vcn_lc_plist pl[2];
    vcn_lc_mkplist(t,P->w[0],&pl[0]); vcn_lc_mkplist(t,P->w[1],&pl[1]);
    uint16_t A[VCN_LC_D][VCN_LC_NL], pre[VCN_LC_D];
    uint8_t Y[VCN_LC_D][VCN_LC_NL], V[VCN_LC_D][VCN_LC_NL], cls[VCN_LC_D];
    memset(A,0,sizeof(A));memset(Y,0,sizeof(Y));memset(V,0,sizeof(V));
    vcn_lc_lanes lanes={0};
    vcn_status status=VCN_OK;
    for (int z=0; z<VCN_LC_Z && status==VCN_OK; ++z)
      for (int y=0; y<VCN_LC_Y && status==VCN_OK; ++y) {
        const int s0=z==0; const size_t ri=VCN_LC_IDX(z,y,0), o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        vcn_lc_row_begin(b,z,y,128); vcn_lc_row_begin(e,z,y,0);
        VCN_LC_PRE(b+ri,&pl[s0],(uint16_t)P->bias[s0],pre);
        VCN_LC_CLS(e,ri,P,pre,cls);
        for (int k=0;k<3;++k) {
          lanes.w[k][0]=k<t->nrow ? (uint16_t)P->w[s0][k] : 0;
          lanes.W[k][0]=b[ri-1-k];
        }
        for (int x=0;x<VCN_LC_D;++x) {
          int symbol=symbol_reader(reader_context,o+x,cls[x]);
          if (symbol<0 || symbol>255) { status=VCN_ERR_CORRUPT; break; }
          A[x][0]=pre[x]; Y[x][0]=(uint8_t)symbol; e[ri+x]=(uint8_t)((symbol+1)>>1);
        }
        if (status!=VCN_OK) break;
        vcn_lc_rec_c(1,A,Y,&lanes,V);
        for (int x=0;x<VCN_LC_D;++x) dst[o+x]=b[ri+x]=V[x][0];
        vcn_lc_row_end(b,z,y); vcn_lc_row_end(e,z,y);
      }
    free(b); free(e); free(P); return status;
  }
  uint32_t *T = (uint32_t *)malloc(sizeof(uint32_t) * M * (size_t)K);
  uint32_t *joint_classes=NULL;
  uint8_t *b = (uint8_t *)malloc(VCN_LC_NB), *e = no_activity?NULL:(uint8_t *)malloc(VCN_LC_NB);
  /* per group: row-major A (previous-row taps plus class bias), classes and symbols; A and the symbols
     lane-major; the values lane-major */
  uint16_t(*Ar)[VCN_LC_D] = (uint16_t(*)[VCN_LC_D])calloc(VCN_LC_NL * VCN_LC_D, sizeof(uint16_t));
  uint8_t(*Kr)[VCN_LC_D] = (uint8_t(*)[VCN_LC_D])calloc(VCN_LC_NL * VCN_LC_D + 32, 1);
  uint8_t(*Yr)[VCN_LC_D] = (uint8_t(*)[VCN_LC_D])calloc(VCN_LC_NL * VCN_LC_D + 32, 1);
  uint16_t(*At)[VCN_LC_NL] = (uint16_t(*)[VCN_LC_NL])calloc(VCN_LC_NL * VCN_LC_D, sizeof(uint16_t));
  uint8_t(*Yt)[VCN_LC_NL] = (uint8_t(*)[VCN_LC_NL])calloc(VCN_LC_NL * VCN_LC_D, 1);
  uint8_t(*Vt)[VCN_LC_NL] = (uint8_t(*)[VCN_LC_NL])calloc(VCN_LC_NL * VCN_LC_D, 1);
  uint8_t(*Dr)[VCN_LC_D]=selectors?(uint8_t(*)[VCN_LC_D])calloc(VCN_LC_NL*VCN_LC_D,1):NULL;
  uint8_t(*Dt)[VCN_LC_NL]=selectors?(uint8_t(*)[VCN_LC_NL])calloc(VCN_LC_NL*VCN_LC_D,1):NULL;
  vcn_lc_lanes *L = (vcn_lc_lanes *)calloc(1, sizeof(vcn_lc_lanes));
  vcn_lc_sched sc = {0, NULL, NULL};
  vcn_status st = VCN_OK;
  if (!T || !b || (!no_activity && !e) || !Ar || !Kr || !Yr || !At || !Yt || !Vt || !L || (selectors && (!Dr || !Dt)) || vcn_lc_mksched(vcn_lc_delta(t), &sc)) {
    st = VCN_ERR_NOMEM;
    goto done;
  }
  if(!symbol_reader && !fixed_symbols && vcn_lc_read_tables(&p,end,T,K,M))st=VCN_ERR_CORRUPT;
  if (st != VCN_OK || ((symbol_reader || fixed_symbols) ? p!=end : end-p<4*VCN_LC_NS)) {
    st = VCN_ERR_CORRUPT;
    goto done;
  }
  {
    vcn_lc_dec D={0};
    if(!symbol_reader && !fixed_symbols) {
      for (int k = 0; k < VCN_LC_NS; k++) D.S[k] = vcn_rd32(p + 4 * k);
      p += 4 * VCN_LC_NS;
    }
    D.T = T, D.p = p, D.end = end, D.over = 0, D.tb = tb;
    vcn_lc_plist pl[2];
    vcn_lc_mkplist(t, P->w[0], &pl[0]), vcn_lc_mkplist(t, P->w[1], &pl[1]);
    uint16_t wr[2][3] = {{0}};
    for (int s2 = 0; s2 < 2; s2++)
      for (int k = 0; k < t->nrow; k++) wr[s2][k] = (uint16_t)P->w[s2][k];
    memset(b, 128, VCN_LC_NB);if(!no_activity)memset(e, 0, VCN_LC_NB);
#if VCN_HAVE_AVX2
    const int simd = !force_scalar && vcn_use_avx2();
    if(ng_native_fast_control&&simd&&!no_activity&&!selectors&&!fixed_symbols&&!symbol_reader){
      joint_classes=malloc(VCN_LC_AMAX*sizeof(*joint_classes));
      if(!joint_classes){st=VCN_ERR_NOMEM;goto done;}
      for(int a=0;a<VCN_LC_AMAX;a++)joint_classes[a]=(uint16_t)P->cb[P->lut[a]]|((uint32_t)P->lut[a]<<16);
    }
#elif VCN_HAVE_NEON
    const int simd = !force_scalar;
#endif
    unsigned needed_rows[(VCN_LC_Y/4)*(VCN_LC_Z/4)];
    const int block_shift=block_size==4?2:block_size==8?3:4;
    const int block_nb=VCN_LC_D>>block_shift, block_ny=VCN_LC_Y>>block_shift, block_nz=VCN_LC_Z>>block_shift;
    if(block_size)for(int r=0;r<block_ny*block_nz;r++) {
      unsigned needed=0;
      for(int x=0;x<block_nb;x++)if(!selectors[r*block_nb+x])needed|=1u<<((x*block_size)>>5);
      needed_rows[r]=needed;
    }
    VCN_LC_TICK(-1);
    for (int g = 0; g < sc.ng; g++) {
      const int nl = sc.n[g];
      int any_delta=0;
      for (int j = 0; j < nl; j++) {
        const int z = sc.row[g][j] / VCN_LC_Y, y = sc.row[g][j] % VCN_LC_Y, s0 = z == 0;
        vcn_lc_row_begin(b, z, y, 128);if(!no_activity)vcn_lc_row_begin(e, z, y, 0);
        const size_t ri = VCN_LC_IDX(z, y, 0);
        unsigned needed=block_size?needed_rows[(z>>block_shift)*block_ny+(y>>block_shift)]:(UINT32_MAX >> (32-VCN_LC_D/32));
        if(needed!=(UINT32_MAX >> (32-VCN_LC_D/32)))
          vcn_lc_pre_selected(b+ri,&pl[s0],(uint16_t)P->bias[s0],Ar[j],needed,force_scalar);
        else if(force_scalar)vcn_lc_pre_c(b+ri,&pl[s0],(uint16_t)P->bias[s0],Ar[j]);
#ifdef NG_HAVE_WIDE_PRE
        else if(ng_native_fast_control&&VCN_LC_D>=64&&VCN_LC_D%64==0&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw"))
          vcn_lc_pre_avx512_impl(b+ri,&pl[s0],(uint16_t)P->bias[s0],Ar[j],needed);
#endif
        else VCN_LC_PRE(b+ri,&pl[s0],(uint16_t)P->bias[s0],Ar[j]);
        if(block_contexts) {
          const uint8_t* ctx=block_contexts+(((z>>block_shift)*block_ny+(y>>block_shift))*block_nb);
          for(int x=0;x<VCN_LC_D;x++)Kr[j][x]=ctx[x>>block_shift]>>3;
        } else if(!fixed_symbols) {
#if VCN_HAVE_AVX2
          if(joint_classes){
#ifdef NG_HAVE_WIDE_PRE
            if(ng_use_avx512()&&__builtin_cpu_supports("avx512bw"))ng_bank_class_wide(e,ri,joint_classes,Ar[j],Kr[j]);else
#endif
            ng_bank_class_lookup(e,ri,joint_classes,Ar[j],Kr[j],1);
          }else
#endif
          if(force_scalar)vcn_lc_cls_c(e,ri,P,Ar[j],Kr[j]);
          else VCN_LC_CLS(e,ri,P,Ar[j],Kr[j]);
        }
        if(selectors) {
          if(block_size) {
            uint8_t choices[VCN_LC_D];const uint8_t identity[8]={0,1,2,3,4,5,6,7};
            const uint8_t* row_choices=selectors+(((z>>block_shift)*block_ny+(y>>block_shift))*block_nb);
            for(int x=0;x<VCN_LC_D;x++)choices[x]=row_choices[x>>block_shift];
            any_delta|=vcn_lc_select_pre(b,ri,z,y,choices,identity,8,Ar[j],Dr[j],force_scalar);
          } else any_delta|=vcn_lc_select_pre(b,ri,z,y,Kr[j],selectors,K,Ar[j],Dr[j],force_scalar);
        }
        const uint8_t *br = b + ri;
        for (int k = 0; k < 3; k++) L->w[k][j] = wr[s0][k], L->W[k][j] = br[-1 - k];
      }
      VCN_LC_TICK(0);
      if(fixed_symbols) {
        for(int j=0;j<nl;j++)memcpy(Yr[j],fixed_symbols+(size_t)sc.row[g][j]*VCN_LC_D,VCN_LC_D);
      } else if(symbol_reader) {
        for(int j=0;j<nl;j++)for(int x=0;x<VCN_LC_D;x++) {
          size_t i=(size_t)sc.row[g][j]*VCN_LC_D+x;
          int symbol=symbol_reader(reader_context,i,Kr[j][x]);
          if(symbol<0 || symbol>255){st=VCN_ERR_CORRUPT;goto done;}
          Yr[j][x]=(uint8_t)symbol;
        }
      } else {
#if VCN_HAVE_AVX2
      if (simd) {
#ifdef NG_HAVE_WIDE_PRE
        if(ng_native_fast_control&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl"))ng_syms_register(nl*VCN_LC_D,Kr[0],&D,Yr[0]);else
#endif
        if(ng_native_fast_control)vcn_lc_syms_gather_avx2(nl*VCN_LC_D,Kr[0],&D,Yr[0]);else
        vcn_lc_syms_avx2(nl * VCN_LC_D, Kr[0], &D, Yr[0]);
      }
      else
#elif VCN_HAVE_NEON
      if (simd) vcn_lc_syms_neon(nl * VCN_LC_D, Kr[0], &D, Yr[0]);
      else
#endif
        vcn_lc_syms_c(nl * VCN_LC_D, Kr[0], &D, Yr[0]);
      }
      /* the residual magnitudes for later rows' classes come straight from the symbols */
      if(!no_activity)for (int j = 0; j < nl; j++) {
        const int z = sc.row[g][j] / VCN_LC_Y, y = sc.row[g][j] % VCN_LC_Y;
        uint8_t *restrict er = e + VCN_LC_IDX(z, y, 0);
        const uint8_t *restrict yr = Yr[j];
        for (int x = 0; x < VCN_LC_D; x++) er[x] = (uint8_t)((yr[x] + 1u) >> 1);
        vcn_lc_row_end(e, z, y);
      }
      VCN_LC_TICK(1);
      /* lane-major; lanes past nl hold stale values, which only unused lanes read */
#if VCN_HAVE_AVX2
      if (simd) vcn_lc_tr_a(nl, (const uint16_t(*)[VCN_LC_D])Ar, At), vcn_lc_tr_y(nl, (const uint8_t(*)[VCN_LC_D])Yr, Yt);
      else
#elif VCN_HAVE_NEON
      if (simd) vcn_lc_tr_a_neon(nl, (const uint16_t(*)[VCN_LC_D])Ar, At), vcn_lc_tr_y_neon(nl, (const uint8_t(*)[VCN_LC_D])Yr, Yt);
      else
#endif
        for (int j = 0; j < nl; j++)
          for (int x = 0; x < VCN_LC_D; x++) At[x][j] = Ar[j][x], Yt[x][j] = Yr[j][x];
      VCN_LC_TICK(2);
      if(any_delta) {
#if VCN_HAVE_AVX2
        if(simd)vcn_lc_tr_y(nl,(const uint8_t(*)[VCN_LC_D])Dr,Dt);
        else
#elif VCN_HAVE_NEON
        if(simd)vcn_lc_tr_y_neon(nl,(const uint8_t(*)[VCN_LC_D])Dr,Dt);
        else
#endif
          for(int j=0;j<nl;j++)for(int x=0;x<VCN_LC_D;x++)Dt[x][j]=Dr[j][x];
      }
      if(any_delta) {
#if VCN_HAVE_AVX2
        if(simd)vcn_lc_rec_selected_avx2(nl,(const uint16_t(*)[VCN_LC_NL])At,(const uint8_t(*)[VCN_LC_NL])Yt,L,Vt,(const uint8_t(*)[VCN_LC_NL])Dt);
        else
#elif VCN_HAVE_NEON
        if(simd)vcn_lc_rec_neon_impl(nl,(const uint16_t(*)[VCN_LC_NL])At,(const uint8_t(*)[VCN_LC_NL])Yt,L,Vt,(const uint8_t(*)[VCN_LC_NL])Dt);
        else
#endif
          vcn_lc_rec_c_impl(nl,(const uint16_t(*)[VCN_LC_NL])At,(const uint8_t(*)[VCN_LC_NL])Yt,L,Vt,(const uint8_t(*)[VCN_LC_NL])Dt);
      } else {
#if VCN_HAVE_AVX2
      if (simd) {
#ifdef NG_HAVE_WIDE_PRE
        if(ng_native_fast_control>=2&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl"))vcn_lc_rec_wide(nl,(const uint16_t(*)[VCN_LC_NL])At,(const uint8_t(*)[VCN_LC_NL])Yt,L,Vt);else
#endif
        vcn_lc_rec_avx2(nl, (const uint16_t(*)[VCN_LC_NL])At, (const uint8_t(*)[VCN_LC_NL])Yt, L, Vt);
      }
      else
#elif VCN_HAVE_NEON
      if (simd) vcn_lc_rec_neon(nl, (const uint16_t(*)[VCN_LC_NL])At, (const uint8_t(*)[VCN_LC_NL])Yt, L, Vt);
      else
#endif
        vcn_lc_rec_c(nl, (const uint16_t(*)[VCN_LC_NL])At, (const uint8_t(*)[VCN_LC_NL])Yt, L, Vt);
      }
      VCN_LC_TICK(3);
      uint8_t *rows[VCN_LC_NL];
      for (int j = 0; j < nl; j++) {
        const int z = sc.row[g][j] / VCN_LC_Y, y = sc.row[g][j] % VCN_LC_Y;
        rows[j] = b + VCN_LC_IDX(z, y, 0);
      }
#if VCN_HAVE_AVX2
      if (simd) vcn_lc_tr_v(nl, (const uint8_t(*)[VCN_LC_NL])Vt, rows);
      else
#elif VCN_HAVE_NEON
      if (simd) vcn_lc_tr_v_neon(nl, (const uint8_t(*)[VCN_LC_NL])Vt, rows);
      else
#endif
        for (int j = 0; j < nl; j++)
          for (int x = 0; x < VCN_LC_D; x++) rows[j][x] = Vt[x][j];
      for (int j = 0; j < nl; j++) {
        const int z = sc.row[g][j] / VCN_LC_Y, y = sc.row[g][j] % VCN_LC_Y;
        memcpy(dst + ((size_t)z * VCN_LC_Y + y) * VCN_LC_D, rows[j], VCN_LC_D);
        vcn_lc_row_end(b, z, y);
      }
      VCN_LC_TICK(4);
    }
    if(!symbol_reader && !fixed_symbols) {
      if (D.over || D.p != end) st = VCN_ERR_CORRUPT;
      for (int k = 0; k < VCN_LC_NS; k++)
        if (D.S[k] != VCN_LC_RL) st = VCN_ERR_CORRUPT;
    }
  }
done:
  free(sc.row), free(sc.n), free(T), free(b), free(e), free(Ar), free(Kr), free(Yr), free(At), free(Yt), free(Vt), free(L),
      free(P),free(Dr),free(Dt),free(joint_classes);
  return st;
}

static vcn_status vcn_lc_decode_impl(const uint8_t* p,size_t n,uint8_t* dst,ng_symbol_reader reader,void* context,
    const uint8_t* selectors,int scalar,int B) {
  return vcn_lc_decode_core(p,n,dst,reader,context,selectors,scalar,B,NULL,NULL);
}
static vcn_status vcn_lc_decode_ex(const uint8_t* p,size_t n,uint8_t* dst,ng_symbol_reader reader,void* context) {
  return vcn_lc_decode_impl(p,n,dst,reader,context,NULL,0,0);
}

static vcn_status vcn_lc_decode(const uint8_t *p, size_t n, uint8_t *dst) {
  return vcn_lc_decode_ex(p,n,dst,NULL,NULL);
}

static vcn_status vcn_encode_llctx(const uint8_t *v, uint8_t *out, size_t cap, size_t *out_n) {
  const vcn_lc_cfg cf = {3, 3, 1, 16, VCN_LC_TB, VCN_LC_NSEL, VCN_LC_STRIDE};
  return vcn_lc_encode(v, &cf, out, cap, out_n);
}
static vcn_status vcn_decode_llctx(const uint8_t *p, size_t n, uint8_t *v) { return vcn_lc_decode(p, n, v); }

/* ======================================================================== */
/*                                 top level                                */
/* ======================================================================== */
static inline void vcn_header(uint8_t *h, unsigned mode) {
  h[0] = 'V', h[1] = 'C', h[2] = 'N', h[3] = 'G', h[4] = (uint8_t)VCN_REVISION, h[5] = (uint8_t)mode;
  memset(h + 6, 0, VCN_HDR - 6);
}

static inline vcn_status vcn_encode_lossless_chunk_config(const uint8_t *src, uint8_t *dst, size_t cap, size_t *out_n,
                                                         const vcn_lc_cfg *config) {
  if (!src || !dst || !out_n) return VCN_ERR_ARG;
  /* constant chunk */
  {
    size_t i = 1;
    while (i < VCN_VOXELS && src[i] == src[0]) i++;
    if (i == VCN_VOXELS) {
      if (cap < VCN_HDR + 1) return VCN_ERR_CAP;
      vcn_header(dst, VCN_MODE_CONST);
      dst[VCN_HDR] = src[0];
      *out_n = VCN_HDR + 1;
      return VCN_OK;
    }
  }
  {
    size_t n = 0;
    vcn_status st = VCN_ERR_CAP;
    (void)vcn_encode_llhuf; /* mode 5's encoder comes with its section; this file writes mode 6 */
    if (cap >= VCN_HDR) st = config ? vcn_lc_encode(src,config,dst+VCN_HDR,cap-VCN_HDR,&n)
                                  : vcn_encode_llctx(src, dst + VCN_HDR, cap - VCN_HDR, &n);
    if (st == VCN_OK && VCN_HDR + n < VCN_HDR + VCN_VOXELS) {
      vcn_header(dst, VCN_MODE_LLCTX);
      *out_n = VCN_HDR + n;
      return VCN_OK;
    }
    if (st != VCN_OK && st != VCN_ERR_CAP) return st;
  }
  /* incompressible: a raw chunk */
  if (cap < VCN_HDR + VCN_VOXELS) return VCN_ERR_CAP;
  vcn_header(dst, VCN_MODE_RAW);
  memcpy(dst + VCN_HDR, src, VCN_VOXELS);
  *out_n = VCN_HDR + VCN_VOXELS;
  return VCN_OK;
}

static inline vcn_status vcn_encode_lossless_chunk(const uint8_t *src, uint8_t *dst, size_t cap, size_t *out_n) {
  return vcn_encode_lossless_chunk_config(src,dst,cap,out_n,NULL);
}

static inline vcn_status vcn_decode(const uint8_t *enc, size_t n, uint8_t *dst, size_t dst_n) {
  if (!enc || !dst || dst_n < VCN_VOXELS) return VCN_ERR_ARG;
  const int mode = vcn_chunk_mode(enc, n);
  if (mode < 0) return VCN_ERR_CORRUPT;
  if (mode == (int)VCN_MODE_CONST) {
    if (n != VCN_HDR + 1) return VCN_ERR_CORRUPT;
    memset(dst, enc[VCN_HDR], VCN_VOXELS);
    return VCN_OK;
  }
  if (mode == (int)VCN_MODE_RAW) {
    if (n != VCN_HDR + VCN_VOXELS) return VCN_ERR_CORRUPT;
    memcpy(dst, enc + VCN_HDR, VCN_VOXELS);
    return VCN_OK;
  }
  if (mode == (int)VCN_MODE_LLHUF) {
    if (enc[6] | enc[7] | enc[8] | enc[9] | enc[10] | enc[11]) return VCN_ERR_CORRUPT;
    return vcn_decode_llhuf(enc + VCN_HDR, n - VCN_HDR, dst);
  }
  if (mode == (int)VCN_MODE_LLCTX) {
    if (enc[6] | enc[7] | enc[8] | enc[9] | enc[10] | enc[11]) return VCN_ERR_CORRUPT;
    return vcn_decode_llctx(enc + VCN_HDR, n - VCN_HDR, dst);
  }
  if (mode == 0 || mode == 1 || mode == 4) return VCN_ERR_MODE;
  return VCN_ERR_CORRUPT;
}

#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif /* VOLCODEC_NG_LOSSLESS_H */
