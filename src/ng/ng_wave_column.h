#pragma once
/* Shared exact recurrence for native reconstruction and entropy integration. */
#ifdef __cplusplus
#define NG_WRESTRICT __restrict
#else
#define NG_WRESTRICT restrict
#endif
typedef struct {
    uint16_t pre[32],w0[32],w1[32],w2[32],feature[32],residual[32];
    int16_t constant[32],left[32],clamp[32],lo[32],hi[32];
} ng_bank_wave_column;
typedef struct {uint16_t pre[32],constant[32],flags[32],bounds[32],residual[32];} ng_bank_compact_column;
/* Expose the same compact predictor before entropy routing. No residual or
 * history updates: callers may select a context, then use compact_step. */
#define NG_ROUTE_PREDICT(NAME,ATTR) \
ATTR static void NAME(const ng_bank_compact_column* NG_WRESTRICT c,const uint16_t (* NG_WRESTRICT w)[32],const uint16_t (* NG_WRESTRICT prev)[32],uint16_t* NG_WRESTRICT q){ \
    for(unsigned j=0;j<32;j++){unsigned flags=c->flags[j];int16_t left=(flags&2)?prev[0][j]:0,base=(int16_t)c->constant[j]+left; \
        int16_t lo=c->bounds[j]&255,hi=c->bounds[j]>>8;if(flags&2){lo=left<lo?left:lo;hi=left>hi?left:hi;} \
        if(flags&4)base=base<lo?lo:base>hi?hi:base; \
        uint16_t dot=(uint16_t)((uint32_t)w[0][j]*prev[0][j]+(uint32_t)w[1][j]*prev[1][j]+(uint32_t)w[2][j]*prev[2][j]); \
        q[j]=(uint16_t)(c->pre[j]+((flags&1)?dot:0)+(uint32_t)((flags&1)?w[3][j]:256)*(uint8_t)base); \
    } \
}
NG_ROUTE_PREDICT(ng_route_predict_c,)
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
NG_ROUTE_PREDICT(ng_route_predict_avx2,__attribute__((target("avx2"))))
NG_ROUTE_PREDICT(ng_route_predict_avx512,__attribute__((target("avx512f,avx512bw"))))
#endif
#undef NG_ROUTE_PREDICT
#define NG_ROUTE_FINISH(NAME,ATTR) \
ATTR static void NAME(const uint16_t* NG_WRESTRICT q,const uint16_t* NG_WRESTRICT r,uint16_t (* NG_WRESTRICT prev)[32],uint8_t* NG_WRESTRICT out){ \
    for(unsigned j=0;j<32;j++){uint16_t residual=r[j];if(q[j]&128)residual=(uint16_t)-residual; \
        uint8_t v=(uint8_t)(((uint16_t)(q[j]+128u)>>8)+residual);out[j]=v; \
        prev[2][j]=prev[1][j];prev[1][j]=prev[0][j];prev[0][j]=v;} \
}
NG_ROUTE_FINISH(ng_route_finish_c,)
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
NG_ROUTE_FINISH(ng_route_finish_avx2,__attribute__((target("avx2"))))
NG_ROUTE_FINISH(ng_route_finish_avx512,__attribute__((target("avx512f,avx512bw"))))
#endif
#undef NG_ROUTE_FINISH
static inline uint16_t ng_route_full_prediction(const ng_bank_wave_column*c,const uint16_t prev[3][32],unsigned j){
    int16_t left=(int16_t)(c->left[j]*prev[0][j]),base=c->constant[j]+left,lo=c->lo[j],hi=c->hi[j];
    if(c->left[j]){lo=left<lo?left:lo;hi=left>hi?left:hi;}if(c->clamp[j])base=base<lo?lo:base>hi?hi:base;
    return (uint16_t)(c->pre[j]+(uint32_t)c->w0[j]*prev[0][j]+(uint32_t)c->w1[j]*prev[1][j]+(uint32_t)c->w2[j]*prev[2][j]+(uint32_t)c->feature[j]*(uint8_t)base);
}
#define NG_BANK_WAVE_KERNEL(name,attr) \
attr static void name(int width,const ng_bank_wave_column* NG_WRESTRICT a,uint16_t prev[3][32],uint8_t* NG_WRESTRICT result){ \
    for(int x=0;x<width;x++){ \
        const ng_bank_wave_column* NG_WRESTRICT c=a+x; \
        for(int j=0;j<32;j++){ \
            uint16_t q=ng_route_full_prediction(c,prev,j); \
            uint16_t r=c->residual[j];if(q&128)r=(uint16_t)-r; \
            uint8_t v=(uint8_t)(((uint16_t)(q+128u)>>8)+r); \
            result[x*32+j]=v;prev[2][j]=prev[1][j];prev[1][j]=prev[0][j];prev[0][j]=v; \
        } \
    } \
}
NG_BANK_WAVE_KERNEL(ng_bank_wave_c,)
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
NG_BANK_WAVE_KERNEL(ng_bank_wave_avx2,__attribute__((target("avx2"))))
NG_BANK_WAVE_KERNEL(ng_bank_wave_avx512,__attribute__((target("avx512f,avx512bw"))))
#define NG_RNAME ng_route_compact_wide_impl
#define NG_NATIVE_MASK 1
#include "ng_recur_wide.h"
__attribute__((target("avx512f,avx512bw,avx512vl"))) static void ng_route_compact_wide(const ng_bank_compact_column*c,const uint16_t weights[4][32],uint16_t prev[3][32],uint8_t*out){
    ng_route_compact_wide_impl(1,c,weights,weights[3],prev,out,0,2);
}
#define NG_RNAME ng_route_compact_symbols_impl
#define NG_NATIVE_MASK 1
#define NG_RSYMBOLS 1
#include "ng_recur_wide.h"
__attribute__((target("avx512f,avx512bw,avx512vl"))) static void ng_route_compact_symbols(const ng_bank_compact_column*c,const uint16_t weights[4][32],uint16_t prev[3][32],uint8_t*out,const uint8_t*symbols,uint8_t*errors){
    ng_route_compact_symbols_impl(1,c,weights,weights[3],prev,out,0,2,symbols,errors);
}
__attribute__((target("avx512f,avx512bw"))) static void ng_route_symbols512(const uint8_t*symbols,ng_bank_wave_column*full,ng_bank_compact_column*compact,uint8_t*errors){
    __m512i s=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)symbols));
    __m512i r=_mm512_xor_si512(_mm512_srli_epi16(s,1),_mm512_sub_epi16(_mm512_setzero_si512(),_mm512_and_si512(s,_mm512_set1_epi16(1))));
    if(compact){__m512i flags=_mm512_loadu_si512(compact->flags);flags=_mm512_or_si512(_mm512_and_si512(flags,_mm512_set1_epi16(255)),_mm512_slli_epi16(r,8));_mm512_storeu_si512(compact->flags,flags);}
    else _mm512_storeu_si512(full->residual,r);
    __m512i e=_mm512_srli_epi16(_mm512_add_epi16(s,_mm512_set1_epi16(1)),1);
    _mm256_storeu_si256((__m256i*)errors,_mm512_cvtepi16_epi8(e));
}
#endif
#undef NG_BANK_WAVE_KERNEL
#undef NG_WRESTRICT
