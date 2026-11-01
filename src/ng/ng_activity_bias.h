#pragma once
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static void ng_activity_bias_avx2(unsigned width,uint16_t*q,const uint16_t*a,const uint32_t*table){
    for(unsigned x=0;x<width;x+=8){
        __m256i ix=_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a+x)));
        __m256i b=_mm256_i32gather_epi32((const int*)table,ix,4);
        __m128i packed=_mm_packus_epi32(_mm256_castsi256_si128(b),_mm256_extracti128_si256(b,1));
        _mm_storeu_si128((__m128i*)(q+x),_mm_add_epi16(_mm_loadu_si128((const __m128i*)(q+x)),packed));
    }
}
#endif
static void ng_activity_bias(unsigned width,uint16_t*q,const uint16_t*a,const uint32_t*table){
#if VCN_HAVE_AVX2
    if(vcn_use_avx2()){ng_activity_bias_avx2(width,q,a,table);return;}
#endif
    for(unsigned x=0;x<width;x++)q[x]=(uint16_t)(q[x]+table[a[x]]);
}
