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
#endif
