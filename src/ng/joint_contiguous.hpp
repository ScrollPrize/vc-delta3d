#pragma once
// Explicit prediction tradeoff: no X dependency, adjacent X voxels are SIMD
// lanes. Only Y/Z/YZ Delta choices; existing modular inverse lifting unchanged.
#if JOINT_AVX2
__attribute__((target("avx2"))) static inline __m256i joint_load8(const uint8_t*p){return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));}
__attribute__((target("avx2"))) static void joint_contiguous_avx(
    const uint8_t*x,const uint8_t*y,size_t i,int d,int yy,int zz,const int*mask,JointBatch&q,int n){
    auto zero=_mm256_setzero_si256(),byte=_mm256_set1_epi32(255);
    for(int c=0;c<2;c++)for(int k=0;k<n;k+=8){
        const uint8_t*src=c?y:x;int*dst=c?q.py:q.px;
        auto a=yy?joint_load8(src+i+k-d):zero,b=zz?joint_load8(src+i+k-size_t(d)*d):zero,ab=yy&&zz?joint_load8(src+i+k-size_t(d)*d-d):zero;
        auto m=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask+k));
        auto usea=_mm256_cmpeq_epi32(_mm256_and_si256(m,_mm256_set1_epi32(2)),_mm256_set1_epi32(2));
        auto useb=_mm256_cmpeq_epi32(_mm256_and_si256(m,_mm256_set1_epi32(4)),_mm256_set1_epi32(4));
        auto v=_mm256_sub_epi32(_mm256_add_epi32(_mm256_and_si256(a,usea),_mm256_and_si256(b,useb)),_mm256_and_si256(ab,_mm256_and_si256(usea,useb)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst+k),_mm256_and_si256(v,byte));
    }
}
#endif
static void joint_contiguous_predict(const uint8_t*x,const uint8_t*y,size_t i,int d,int yy,int zz,const int*mask,JointBatch&q,int n,bool simd){
#if JOINT_AVX2
    if(simd&&joint_has_avx()&&n%8==0){joint_contiguous_avx(x,y,i,d,yy,zz,mask,q,n);return;}
#endif
    for(int k=0;k<n;k++){
        q.px[k]=delta_predict_xyz(x,i+k,mask[k],0,yy,zz,d,size_t(d)*d);
        q.py[k]=delta_predict_xyz(y,i+k,mask[k],0,yy,zz,d,size_t(d)*d);
    }
}
