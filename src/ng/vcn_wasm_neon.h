/* The NEON intrinsics the NG kernels use, implemented with WebAssembly SIMD128.
 *
 * NEON and WebAssembly SIMD vectors are both 128 bits wide, so the AArch64
 * kernels translate almost one to one; ng_kernel_common.h includes this header
 * instead of <arm_neon.h> when compiling with -msimd128. The kernels then give
 * the same integers as on AArch64 (the test digest compares the builds).
 * Emscripten's own <arm_neon.h> (SIMDe) keeps 64-bit vectors as structs and
 * falls back to scalar loops for several of these; this subset decodes about
 * 15% faster in V8.
 *
 * Every vector type is a v128_t; a 64-bit NEON vector lives in the low half and
 * its high half is unspecified. Only intrinsics the library uses are defined, so
 * a kernel that starts using a new one fails to compile instead of slowing down. */
#ifndef VC_DELTA3D_WASM_NEON_H_
#define VC_DELTA3D_WASM_NEON_H_

#include <stdint.h>
#include <wasm_simd128.h>

#define NEON_INLINE static inline __attribute__((always_inline))

typedef v128_t uint8x16_t;
typedef v128_t uint16x8_t;
typedef v128_t uint32x4_t;
typedef v128_t uint64x2_t;
typedef v128_t int8x16_t;
typedef v128_t int16x8_t;
typedef v128_t int32x4_t;
// 64-bit vectors, in the low half.
typedef v128_t uint8x8_t;
typedef v128_t uint16x4_t;
typedef v128_t uint32x2_t;
typedef v128_t uint64x1_t;
typedef v128_t int8x8_t;
typedef v128_t int16x4_t;

typedef struct {
  uint8x16_t val[2];
} uint8x16x2_t;
typedef struct {
  uint16x8_t val[2];
} uint16x8x2_t;
typedef struct {
  uint32x4_t val[2];
} uint32x4x2_t;

// Loads and stores.
NEON_INLINE uint8x16_t vld1q_u8(const uint8_t* p) { return wasm_v128_load(p); }
NEON_INLINE uint16x8_t vld1q_u16(const uint16_t* p) {
  return wasm_v128_load(p);
}
NEON_INLINE uint32x4_t vld1q_u32(const uint32_t* p) {
  return wasm_v128_load(p);
}
NEON_INLINE int16x8_t vld1q_s16(const int16_t* p) { return wasm_v128_load(p); }
NEON_INLINE uint8x8_t vld1_u8(const uint8_t* p) {
  return wasm_v128_load64_zero(p);
}
#define vld1q_lane_u32(p, v, lane) wasm_v128_load32_lane((p), (v), (lane))
NEON_INLINE void vst1q_u8(uint8_t* p, uint8x16_t v) { wasm_v128_store(p, v); }
NEON_INLINE void vst1q_u16(uint16_t* p, uint16x8_t v) {
  wasm_v128_store(p, v);
}
NEON_INLINE void vst1q_u32(uint32_t* p, uint32x4_t v) {
  wasm_v128_store(p, v);
}
NEON_INLINE void vst1_u8(uint8_t* p, uint8x8_t v) {
  wasm_v128_store64_lane(p, v, 0);
}

// Lanes.
NEON_INLINE uint8x16_t vdupq_n_u8(uint8_t x) { return wasm_u8x16_splat(x); }
NEON_INLINE uint16x8_t vdupq_n_u16(uint16_t x) { return wasm_u16x8_splat(x); }
NEON_INLINE uint32x4_t vdupq_n_u32(uint32_t x) { return wasm_u32x4_splat(x); }
NEON_INLINE int32x4_t vdupq_n_s32(int32_t x) { return wasm_i32x4_splat(x); }
#define vgetq_lane_u32(v, lane) wasm_u32x4_extract_lane((v), (lane))
#define vget_lane_u64(v, lane) wasm_u64x2_extract_lane((v), (lane))

// Halves.
NEON_INLINE uint8x8_t vget_low_u8(uint8x16_t v) { return v; }
NEON_INLINE uint16x4_t vget_low_u16(uint16x8_t v) { return v; }
NEON_INLINE int8x8_t vget_low_s8(int8x16_t v) { return v; }
NEON_INLINE int16x4_t vget_low_s16(int16x8_t v) { return v; }
NEON_INLINE uint8x8_t vget_high_u8(uint8x16_t v) {
  return wasm_i64x2_shuffle(v, v, 1, 1);
}
NEON_INLINE uint16x4_t vget_high_u16(uint16x8_t v) {
  return wasm_i64x2_shuffle(v, v, 1, 1);
}
NEON_INLINE uint8x16_t vcombine_u8(uint8x8_t lo, uint8x8_t hi) {
  return wasm_i64x2_shuffle(lo, hi, 0, 2);
}
NEON_INLINE uint16x8_t vcombine_u16(uint16x4_t lo, uint16x4_t hi) {
  return wasm_i64x2_shuffle(lo, hi, 0, 2);
}

// Reinterpretation: every type is a v128_t.
#define vreinterpretq_s8_u8(v) (v)
#define vreinterpretq_u8_s8(v) (v)
#define vreinterpretq_u8_u32(v) (v)
#define vreinterpretq_u16_s16(v) (v)
#define vreinterpretq_u16_u8(v) (v)
#define vreinterpretq_u16_u32(v) (v)
#define vreinterpretq_u32_u8(v) (v)
#define vreinterpretq_u32_u16(v) (v)
#define vreinterpret_u16_u8(v) (v)
#define vreinterpret_u64_u8(v) (v)

// Bitwise.
NEON_INLINE uint8x16_t vandq_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_v128_and(a, b);
}
NEON_INLINE uint16x8_t vandq_u16(uint16x8_t a, uint16x8_t b) {
  return wasm_v128_and(a, b);
}
NEON_INLINE uint32x4_t vandq_u32(uint32x4_t a, uint32x4_t b) {
  return wasm_v128_and(a, b);
}
NEON_INLINE uint8x16_t vorrq_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_v128_or(a, b);
}
NEON_INLINE uint32x4_t vorrq_u32(uint32x4_t a, uint32x4_t b) {
  return wasm_v128_or(a, b);
}
NEON_INLINE int8x16_t veorq_s8(int8x16_t a, int8x16_t b) {
  return wasm_v128_xor(a, b);
}
// vbsl takes bits from a where the mask is set, else from b.
NEON_INLINE uint8x16_t vbslq_u8(uint8x16_t m, uint8x16_t a, uint8x16_t b) {
  return wasm_v128_bitselect(a, b, m);
}
NEON_INLINE uint16x8_t vbslq_u16(uint16x8_t m, uint16x8_t a, uint16x8_t b) {
  return wasm_v128_bitselect(a, b, m);
}
NEON_INLINE uint32x4_t vbslq_u32(uint32x4_t m, uint32x4_t a, uint32x4_t b) {
  return wasm_v128_bitselect(a, b, m);
}

// Arithmetic.
NEON_INLINE uint8x16_t vaddq_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_i8x16_add(a, b);
}
NEON_INLINE uint16x8_t vaddq_u16(uint16x8_t a, uint16x8_t b) {
  return wasm_i16x8_add(a, b);
}
NEON_INLINE uint32x4_t vaddq_u32(uint32x4_t a, uint32x4_t b) {
  return wasm_i32x4_add(a, b);
}
NEON_INLINE uint8x16_t vsubq_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_i8x16_sub(a, b);
}
NEON_INLINE int8x16_t vsubq_s8(int8x16_t a, int8x16_t b) {
  return wasm_i8x16_sub(a, b);
}
NEON_INLINE uint16x8_t vsubq_u16(uint16x8_t a, uint16x8_t b) {
  return wasm_i16x8_sub(a, b);
}
NEON_INLINE int8x16_t vabsq_s8(int8x16_t a) { return wasm_i8x16_abs(a); }
NEON_INLINE uint8x16_t vminq_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_u8x16_min(a, b);
}
NEON_INLINE uint8x16_t vmaxq_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_u8x16_max(a, b);
}
NEON_INLINE int16x8_t vminq_s16(int16x8_t a, int16x8_t b) {
  return wasm_i16x8_min(a, b);
}
NEON_INLINE int16x8_t vmaxq_s16(int16x8_t a, int16x8_t b) {
  return wasm_i16x8_max(a, b);
}
NEON_INLINE uint16x8_t vmlaq_n_u16(uint16x8_t a, uint16x8_t b, uint16_t c) {
  return wasm_i16x8_add(a, wasm_i16x8_mul(b, wasm_u16x8_splat(c)));
}
NEON_INLINE uint32x4_t vmlaq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  return wasm_i32x4_add(a, wasm_i32x4_mul(b, c));
}
NEON_INLINE int32x4_t vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return wasm_i32x4_add(a, wasm_i32x4_extmul_low_i16x8(b, c));
}
NEON_INLINE int32x4_t vmlal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  return wasm_i32x4_add(a, wasm_i32x4_extmul_high_i16x8(b, c));
}
NEON_INLINE uint32_t vaddvq_u32(uint32x4_t v) {
  v = wasm_i32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 1));
  v = wasm_i32x4_add(v, wasm_i32x4_shuffle(v, v, 1, 0, 3, 2));
  return wasm_u32x4_extract_lane(v, 0);
}
NEON_INLINE int32_t vaddvq_s32(int32x4_t v) { return (int32_t)vaddvq_u32(v); }

// Widening.
NEON_INLINE uint16x8_t vmovl_u8(uint8x8_t v) {
  return wasm_u16x8_extend_low_u8x16(v);
}
NEON_INLINE uint16x8_t vmovl_high_u8(uint8x16_t v) {
  return wasm_u16x8_extend_high_u8x16(v);
}
NEON_INLINE int16x8_t vmovl_s8(int8x8_t v) {
  return wasm_i16x8_extend_low_i8x16(v);
}
NEON_INLINE int16x8_t vmovl_high_s8(int8x16_t v) {
  return wasm_i16x8_extend_high_i8x16(v);
}
NEON_INLINE uint32x4_t vmovl_u16(uint16x4_t v) {
  return wasm_u32x4_extend_low_u16x8(v);
}
NEON_INLINE uint32x4_t vmovl_high_u16(uint16x8_t v) {
  return wasm_u32x4_extend_high_u16x8(v);
}
NEON_INLINE uint16x8_t vaddl_u8(uint8x8_t a, uint8x8_t b) {
  return wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(a),
                        wasm_u16x8_extend_low_u8x16(b));
}
NEON_INLINE uint16x8_t vaddl_high_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(a),
                        wasm_u16x8_extend_high_u8x16(b));
}
NEON_INLINE uint16x8_t vaddw_u8(uint16x8_t a, uint8x8_t b) {
  return wasm_i16x8_add(a, wasm_u16x8_extend_low_u8x16(b));
}
NEON_INLINE uint16x8_t vaddw_high_u8(uint16x8_t a, uint8x16_t b) {
  return wasm_i16x8_add(a, wasm_u16x8_extend_high_u8x16(b));
}
NEON_INLINE uint16x8_t vsubw_u8(uint16x8_t a, uint8x8_t b) {
  return wasm_i16x8_sub(a, wasm_u16x8_extend_low_u8x16(b));
}
NEON_INLINE uint16x8_t vsubw_high_u8(uint16x8_t a, uint8x16_t b) {
  return wasm_i16x8_sub(a, wasm_u16x8_extend_high_u8x16(b));
}

// Narrowing.
NEON_INLINE uint8x8_t vmovn_u16(uint16x8_t v) {
  return wasm_i8x16_shuffle(v, v, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10,
                            12, 14);
}
NEON_INLINE uint16x4_t vmovn_u32(uint32x4_t v) {
  return wasm_i16x8_shuffle(v, v, 0, 2, 4, 6, 0, 2, 4, 6);
}
// The wasm narrowing reads signed input, so clamp to 255 first.
NEON_INLINE uint8x8_t vqmovn_u16(uint16x8_t v) {
  v = wasm_u16x8_min(v, wasm_u16x8_splat(255));
  return wasm_u8x16_narrow_i16x8(v, v);
}
#define vshrn_n_u16(v, n) vmovn_u16(wasm_u16x8_shr((v), (n)))

// Shifts by an immediate.
#define vshlq_n_u8(v, n) wasm_i8x16_shl((v), (n))
#define vshlq_n_s8(v, n) wasm_i8x16_shl((v), (n))
#define vshlq_n_u32(v, n) wasm_i32x4_shl((v), (n))
#define vshrq_n_s8(v, n) wasm_i8x16_shr((v), (n))
#define vshrq_n_u32(v, n) wasm_u32x4_shr((v), (n))
// Rounding shift right: (v + 2^(n-1)) >> n without overflow, for 1 <= n <= 8.
#define vrshrq_n_u8(v, n)                    \
  wasm_i8x16_add(wasm_u8x16_shr((v), (n)), \
                 wasm_v128_and(wasm_u8x16_shr((v), (n) - 1), wasm_u8x16_splat(1)))

// Shift each lane by the signed low byte of the matching lane of `s`: left if
// positive, logical right if negative, zero if the count is 32 or more. The
// decoder only shifts by a splatted count, which takes the fast path.
NEON_INLINE uint32_t vc_wasm_neon_shl_u32(uint32_t x, int8_t s) {
  if (s >= 0) return s < 32 ? x << s : 0;
  return s > -32 ? x >> -s : 0;
}
NEON_INLINE uint32x4_t vshlq_u32(uint32x4_t v, int32x4_t s) {
  const int32_t s0 = wasm_i32x4_extract_lane(s, 0);
  if (wasm_i32x4_all_true(wasm_i32x4_eq(s, wasm_i32x4_splat(s0)))) {
    const int8_t n = (int8_t)s0;
    if (n >= 0) {
      return n < 32 ? wasm_i32x4_shl(v, n) : wasm_i32x4_splat(0);
    }
    return n > -32 ? wasm_u32x4_shr(v, -n) : wasm_i32x4_splat(0);
  }
  return wasm_u32x4_make(
      vc_wasm_neon_shl_u32(wasm_u32x4_extract_lane(v, 0),
                           (int8_t)wasm_i32x4_extract_lane(s, 0)),
      vc_wasm_neon_shl_u32(wasm_u32x4_extract_lane(v, 1),
                           (int8_t)wasm_i32x4_extract_lane(s, 1)),
      vc_wasm_neon_shl_u32(wasm_u32x4_extract_lane(v, 2),
                           (int8_t)wasm_i32x4_extract_lane(s, 2)),
      vc_wasm_neon_shl_u32(wasm_u32x4_extract_lane(v, 3),
                           (int8_t)wasm_i32x4_extract_lane(s, 3)));
}

// Comparisons.
NEON_INLINE uint8x16_t vceqq_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_i8x16_eq(a, b);
}
NEON_INLINE uint8x16_t vcgtq_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_u8x16_gt(a, b);
}
NEON_INLINE uint16x8_t vcgtq_u16(uint16x8_t a, uint16x8_t b) {
  return wasm_u16x8_gt(a, b);
}
NEON_INLINE uint32x4_t vcltq_u32(uint32x4_t a, uint32x4_t b) {
  return wasm_u32x4_lt(a, b);
}

// Permutations. Out-of-range table indices give 0, as on NEON.
NEON_INLINE uint8x16_t vqtbl1q_u8(uint8x16_t t, uint8x16_t idx) {
  return wasm_i8x16_swizzle(t, idx);
}
NEON_INLINE uint8x16_t vzip1q_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_i8x16_shuffle(a, b, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6,
                            22, 7, 23);
}
NEON_INLINE uint8x16_t vzip2q_u8(uint8x16_t a, uint8x16_t b) {
  return wasm_i8x16_shuffle(a, b, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29,
                            14, 30, 15, 31);
}
NEON_INLINE uint8x16x2_t vtrnq_u8(uint8x16_t a, uint8x16_t b) {
  uint8x16x2_t r;
  r.val[0] = wasm_i8x16_shuffle(a, b, 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26,
                                12, 28, 14, 30);
  r.val[1] = wasm_i8x16_shuffle(a, b, 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27,
                                13, 29, 15, 31);
  return r;
}
NEON_INLINE uint16x8x2_t vtrnq_u16(uint16x8_t a, uint16x8_t b) {
  uint16x8x2_t r;
  r.val[0] = wasm_i16x8_shuffle(a, b, 0, 8, 2, 10, 4, 12, 6, 14);
  r.val[1] = wasm_i16x8_shuffle(a, b, 1, 9, 3, 11, 5, 13, 7, 15);
  return r;
}
NEON_INLINE uint32x4x2_t vtrnq_u32(uint32x4_t a, uint32x4_t b) {
  uint32x4x2_t r;
  r.val[0] = wasm_i32x4_shuffle(a, b, 0, 4, 2, 6);
  r.val[1] = wasm_i32x4_shuffle(a, b, 1, 5, 3, 7);
  return r;
}

#undef NEON_INLINE

#endif  // VC_DELTA3D_WASM_NEON_H_
