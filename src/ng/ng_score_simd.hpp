#pragma once
// Four independent 4x4x4 block sums per vector. A lane's additions remain in
// raster order; there are no horizontal reductions or changed rounding rules.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
__attribute__((target("avx2"),always_inline))
static inline __m256d ng_score_four_step(const uint8_t* candidate,const uint8_t* classes,const double* cost,__m256d sum){
    for(unsigned x=0;x<4;x++){
        __m128i idx=_mm_setr_epi32(
            unsigned(classes[x])*256+candidate[x],
            unsigned(classes[4+x])*256+candidate[4+x],
            unsigned(classes[8+x])*256+candidate[8+x],
            unsigned(classes[12+x])*256+candidate[12+x]);
        sum=_mm256_add_pd(sum,_mm256_i32gather_pd(cost,idx,8));
    }
    return sum;
}
__attribute__((target("avx2")))
static void ng_score_four_blocks(const uint8_t* candidate,const uint8_t* classes,
    const uint32_t* block,size_t count,const double* cost,double* sums){
    for(size_t i=0;i<count;i+=16){
        double* dst=sums+block[i];
        __m256d sum=_mm256_loadu_pd(dst);
        sum=ng_score_four_step(candidate+i,classes+i,cost,sum);
        _mm256_storeu_pd(dst,sum);
    }
}
template<unsigned Outputs=4095>
__attribute__((target("avx2")))
static unsigned ng_fixed_row_avx2(const uint8_t* src,uint8_t*const* out,size_t row,unsigned d){
    const size_t plane=size_t(d)*d;unsigned x=1;
    const __m256i zero=_mm256_setzero_si256(),mask=_mm256_set1_epi16(255);
    for(;x<d;x+=16){
        // A short final group can overlap the previous store: source samples
        // are immutable and each result is independent. Avoid a scalar tail.
        if(x+16>d)x=d-16;
        const size_t i=row+x;
#define NG_LOAD16(off) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(src+i-(off))))
        const __m256i v=NG_LOAD16(0),a=NG_LOAD16(1),b=NG_LOAD16(d),c=NG_LOAD16(plane);
        __m256i p[12];p[0]=zero;p[1]=a;p[2]=b;p[4]=c;
        p[3]=_mm256_sub_epi16(_mm256_add_epi16(a,b),NG_LOAD16(d+1));
        p[5]=_mm256_sub_epi16(_mm256_add_epi16(a,c),NG_LOAD16(plane+1));
        p[6]=_mm256_sub_epi16(_mm256_add_epi16(b,c),NG_LOAD16(plane+d));
        p[7]=_mm256_add_epi16(_mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(p[3],p[5]),p[6]),_mm256_add_epi16(_mm256_add_epi16(a,b),c)),NG_LOAD16(plane+d+1));
#undef NG_LOAD16
        p[8]=_mm256_min_epi16(_mm256_max_epi16(p[3],_mm256_min_epi16(a,b)),_mm256_max_epi16(a,b));
        p[9]=_mm256_min_epi16(_mm256_max_epi16(p[5],_mm256_min_epi16(a,c)),_mm256_max_epi16(a,c));
        p[10]=_mm256_min_epi16(_mm256_max_epi16(p[6],_mm256_min_epi16(b,c)),_mm256_max_epi16(b,c));
        p[11]=_mm256_min_epi16(_mm256_max_epi16(p[7],_mm256_min_epi16(a,_mm256_min_epi16(b,c))),_mm256_max_epi16(a,_mm256_max_epi16(b,c)));
        for(unsigned m=0;m<12;m++)if(Outputs&(1u<<m)){
            __m256i r=_mm256_and_si256(_mm256_sub_epi16(v,p[m]),mask);
            r=_mm256_and_si256(_mm256_xor_si256(_mm256_slli_epi16(r,1),_mm256_sub_epi16(zero,_mm256_srli_epi16(r,7))),mask);
            _mm_storeu_si128((__m128i*)(out[m]+i),_mm_packus_epi16(_mm256_castsi256_si128(r),_mm256_extracti128_si256(r,1)));
        }
    }
    return x;
}
#endif
