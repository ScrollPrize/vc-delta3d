#pragma once
// Exact uint8 modular lifting/flip kernel, eight independent wavefront lanes.
// Included after the scalar helpers; scalar reference is shared, not duplicated.
struct JointBatch {
    alignas(32) int px[32],py[32],sx[32],sy[32],a[32],b[32],vx[32],vy[32];
};
#if JOINT_AVX2
#define J_AVX __attribute__((target("avx2")))
J_AVX static inline __m256i jcenter(__m256i x){return _mm256_sub_epi32(_mm256_and_si256(_mm256_add_epi32(x,_mm256_set1_epi32(128)),_mm256_set1_epi32(255)),_mm256_set1_epi32(128));}
J_AVX static inline __m256i jround(__m256i c,__m256i x){
    auto p=_mm256_mullo_epi32(c,x);
    auto v=_mm256_srli_epi32(_mm256_add_epi32(_mm256_abs_epi32(p),_mm256_set1_epi32(2048)),12);
    return _mm256_sign_epi32(v,p);
}
J_AVX static inline void jmod(__m256i&x,__m256i&y,__m256i a,__m256i b){
    x=jcenter(_mm256_add_epi32(x,jround(a,y)));
    y=jcenter(_mm256_add_epi32(y,jround(b,x)));
    x=jcenter(_mm256_add_epi32(x,jround(a,y)));
}
J_AVX static inline __m256i junword(__m256i s){return _mm256_xor_si256(_mm256_srli_epi32(s,1),_mm256_sub_epi32(_mm256_setzero_si256(),_mm256_and_si256(s,_mm256_set1_epi32(1))));}
template<bool Pack> J_AVX static void joint_batch_avx(JointBatch&q,int n){
    auto z=_mm256_setzero_si256(),bias=_mm256_set1_epi32(128),eight=_mm256_set1_epi32(8);
    for(int i=0;i<n;i+=8){
        auto px=_mm256_load_si256((__m256i*)(q.px+i)),py=_mm256_load_si256((__m256i*)(q.py+i));
        auto a=_mm256_load_si256((__m256i*)(q.a+i)),b=_mm256_load_si256((__m256i*)(q.b+i));
        auto rx=junword(_mm256_load_si256((__m256i*)(q.sx+i))),ry=junword(_mm256_load_si256((__m256i*)(q.sy+i)));
        auto x=_mm256_sub_epi32(px,bias),y=_mm256_sub_epi32(py,bias);jmod(x,y,a,b);
        auto fx=jcenter(_mm256_sub_epi32(bias,px)),fy=jcenter(_mm256_sub_epi32(bias,py));jmod(fx,fy,a,b);
        auto actual=jcenter(_mm256_add_epi32(y,ry)),dist=_mm256_abs_epi32(ry);
        auto flip=_mm256_and_si256(_mm256_cmpgt_epi32(dist,eight),_mm256_cmpgt_epi32(dist,_mm256_add_epi32(_mm256_abs_epi32(jcenter(_mm256_sub_epi32(actual,fy))),eight)));
        x=_mm256_blendv_epi8(x,fx,flip);x=jcenter(_mm256_add_epi32(x,rx));y=actual;
        jmod(x,y,_mm256_sub_epi32(z,a),_mm256_sub_epi32(z,b));
        x=_mm256_add_epi32(x,bias);y=_mm256_add_epi32(y,bias);
        if constexpr(Pack){
            auto ex=_mm256_abs_epi32(jcenter(_mm256_sub_epi32(x,px))),ey=_mm256_abs_epi32(jcenter(_mm256_sub_epi32(y,py)));
            x=_mm256_or_si256(x,_mm256_slli_epi32(ex,8));y=_mm256_or_si256(y,_mm256_slli_epi32(ey,8));
        }
        _mm256_store_si256((__m256i*)(q.vx+i),x);
        _mm256_store_si256((__m256i*)(q.vy+i),y);
    }
}
#undef J_AVX
#endif
static bool joint_has_avx(){
#if JOINT_AVX2
    static const bool available=__builtin_cpu_supports("avx2");return available;
#else
    return false;
#endif
}
template<bool Pack=false> static void joint_batch_reconstruct(JointBatch&q,int n,bool simd){
    int done=0;
#if JOINT_AVX2
    if(simd&&joint_has_avx()){done=n/8*8;joint_batch_avx<Pack>(q,done);}
#endif
    for(int i=done;i<n;i++){
        Lift l;l.a=q.a[i];l.b=q.b[i];auto v=lifting_values_flip({uint16_t(q.sx[i]),uint16_t(q.sy[i])},{q.px[i],q.py[i]},4,l);q.vx[i]=v[0];q.vy[i]=v[1];
        if constexpr(Pack){q.vx[i]|=std::abs(byte_center(v[0]-q.px[i]))<<8;q.vy[i]|=std::abs(byte_center(v[1]-q.py[i]))<<8;}
    }
}
