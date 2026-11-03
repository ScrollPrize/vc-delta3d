#pragma once
#include <stddef.h>
#include <stdint.h>
#ifndef NG_BLOCK_DIM
#define NG_BLOCK_DIM 128
#endif
/* Private, scoped kernel geometry. Public rectangular entrypoints save/restore
 * it on every path; each worker has independent state. No shape in global codec
 * metadata and no cross-call mutable shared state. */
typedef struct { int x,y,z; } ng_geometry;
#ifdef __cplusplus
#if defined(__cpp_constinit)
#define NG_TLS_CONSTINIT constinit
#else
#define NG_TLS_CONSTINIT
#endif
extern "C" {
/* Defined with constant initializers in C. Do not emit a potentially dynamic
   C++ initialization call on every access to these scalar kernel controls. */
extern thread_local NG_TLS_CONSTINIT ng_geometry ng_current_geometry;
extern thread_local NG_TLS_CONSTINIT int ng_disable_avx512;
extern thread_local NG_TLS_CONSTINIT double ng_fit_intercept_weight;
#else
extern _Thread_local ng_geometry ng_current_geometry;
extern _Thread_local int ng_disable_avx512;
extern _Thread_local double ng_fit_intercept_weight;
#endif
int ng_geometry_valid(uint32_t x,uint32_t y,uint32_t z);
#ifdef __cplusplus
}
#endif
#ifdef NG_FIXED_GEOMETRY
#define NG_X NG_FIXED_GEOMETRY
#define NG_Y NG_FIXED_GEOMETRY
#define NG_Z NG_FIXED_GEOMETRY
#else
#define NG_X (ng_current_geometry.x)
#define NG_Y (ng_current_geometry.y)
#define NG_Z (ng_current_geometry.z)
#endif
#define NG_N ((size_t)NG_X*NG_Y*NG_Z)
/* Independent ISA ceiling preserves all
 * fixed-size specializations and AVX2 kernels. Scoped per calling thread. */
static inline int ng_use_avx512(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return !ng_disable_avx512 && __builtin_cpu_supports("avx512f");
#else
    return 0;
#endif
}
