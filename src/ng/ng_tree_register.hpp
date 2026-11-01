// Register-resident unrestricted context -> mask routing -> rANS -> shared
// recurrence. Uses the same feature, entropy and recurrence math as baseline.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define NG_TREG __attribute__((target("avx512f,avx512bw,avx512vl,avx512cd,avx2,popcnt"),always_inline))
struct TreeRegisterEntropy {
    __m256i S[4];const uint8_t*p,*end;size_t over;const uint32_t*T;int tb;
    const TreeCompactEntropy*compact;
    NG_TREG TreeRegisterEntropy(ng_context_rans*c,const TreeCompactEntropy*q):p(c->d.p),end(c->d.end),over(c->d.over),T(c->d.T),tb(c->d.tb),compact(q){for(unsigned k=0;k<4;k++)S[k]=_mm256_loadu_si256((const __m256i*)(c->d.S+8*k));}
    NG_TREG void finish(ng_context_rans*c){for(unsigned k=0;k<4;k++)_mm256_storeu_si256((__m256i*)(c->d.S+8*k),S[k]);c->d.p=p;c->d.over=over;}
    NG_TREG __m256i step(const __m256i*groups,unsigned lanes=32){
        if(lanes<32){
            ng_context_rans temp{};temp.K=256;temp.d.T=T;temp.d.tb=tb;temp.d.end=end;finish(&temp);
            uint8_t ids[32],sy[32]={};alignas(32) int g[32];for(unsigned k=0;k<4;k++)_mm256_store_si256((__m256i*)(g+8*k),groups[k]);for(unsigned k=0;k<32;k++)ids[k]=g[k];
            require(ng_context_rans_read(&temp,lanes,ids,sy,0)==0);p=temp.d.p;over=temp.d.over;for(unsigned k=0;k<4;k++)S[k]=_mm256_loadu_si256((const __m256i*)(temp.d.S+k*8));return _mm256_loadu_si256((const __m256i*)sy);
        }
        const __m256i kM=_mm256_set1_epi32((1u<<tb)-1),kF=_mm256_set1_epi32(4095),k1=_mm256_set1_epi32(1),kL=_mm256_set1_epi32(1<<15),kB=_mm256_set1_epi32(255),ix=_mm256_setr_epi32(0,4,1,5,2,6,3,7);
        const __m128i tbv=_mm_cvtsi32_si128(tb);__m256i result;
#define NG_EUNROLL 1
#define NG_ECLASS(H) groups[H]
#define NG_ELOOKUP(INDEX,C,STATE) (compact?tree_compact_entry(compact,INDEX,C,STATE,tb,kM):_mm256_i32gather_epi32((const int*)T,INDEX,4))
#define NG_EOUTPUT(V) result=V
#include "ng_entropy_step.inc"
#undef NG_EUNROLL
#undef NG_ECLASS
#undef NG_ELOOKUP
#undef NG_EOUTPUT
        return result;
    }
};

#define NG_NATIVE_MASK 1
#include "ng_recur_wide_ops.h"
NG_TREG static inline void tree_register_reconstruct(const ng_bank_compact_column*c,const uint16_t weights[4][32],__m512i&p0,__m512i&p1,__m512i&p2,__m512i register_symbols,uint8_t*result,uint8_t*errors){
    const NG_RV one=NG_RSET(1),bytes=NG_RSET(255),flags=NG_RAND(NG_RLOAD(c->flags),bytes);
    const NG_RV w0=NG_RLOAD(weights[0]),w1=NG_RLOAD(weights[1]),w2=NG_RLOAD(weights[2]),f=NG_RLOAD(weights[3]);
    const NG_RMT ng=NG_REQ(NG_RAND(flags,one),one),left=NG_REQ(NG_RAND(flags,NG_RSET(2)),NG_RSET(2)),clamp=NG_REQ(NG_RAND(flags,NG_RSET(4)),NG_RSET(4));
    const NG_RV factor=NG_RBLEND(NG_RSET(256),f,ng),wx=NG_RADD(w0,f);
    const bool simple=false,xyz=false,pair=false;const unsigned trial=2,x=0,h=0;
#define NG_COMPACT_LOAD(F) NG_RLOAD(c->F)
#define NG_BASE_CONST NG_COMPACT_LOAD(constant)
#define NG_BASE_BOUNDS NG_COMPACT_LOAD(bounds)
#define NG_BASE_PRE NG_COMPACT_LOAD(pre)
#define NG_RSYMBOLS 1
#define NG_RSYMBOLSVECTOR 1
#include "ng_recur_math.inc"
#undef NG_COMPACT_LOAD
}
#include "ng_recur_undef.h"
#undef NG_NATIVE_MASK

struct TreeRegisterWave : TreeColumnMetadata {
    ng_route_wave_data*w;const entropy_tree::MaskForest&m;ng_context_rans*entropy;
    TreeRegisterEntropy re;
    __m512i p0,p1,p2;
    __m256i old_n,old_b,old_nb,current_n,current_b,current_nb;
    const uint8_t*north_base,*back_base,*nb_base;uint8_t*here_base;
    alignas(64) int features[9*32];
    unsigned column=0;
    const unsigned choice_shift=ng_choice_edge==2?1:ng_choice_edge==4?2:ng_choice_edge==8?3:ng_choice_edge==16?4:5;
    NG_TREG TreeRegisterWave(const Snapshot&s,const entropy_tree::VectorForest&v,const entropy_tree::MaskForest&mask,ng_route_wave_data*data,ng_context_rans*e,const TreeCompactEntropy*c,TreeColumnMetadata&&buffers,unsigned G):TreeColumnMetadata(std::move(buffers)),w(data),m(mask),entropy(e),re(e,c){
        prepare_tree_columns(*this,s,v,&m,w,20,G);
        for(unsigned x=0;x<w->width;x++)for(unsigned j=w->lanes;j<32;j++){routed_roots[x*32+j]=v.zero_root;routed_groups[x*32+j]=0;routed_zeros[x*32+j]=1;routed_features[x*9*32+j]=routed_features[(x*9+8)*32+j]=0;}
        p0=_mm512_loadu_si512(w->previous[0]);p1=_mm512_loadu_si512(w->previous[1]);p2=_mm512_loadu_si512(w->previous[2]);
        old_n=old_b=old_nb=_mm256_setzero_si256();
        unsigned t=w->ys[0]+w->delta*w->zs[0],depth=w->history_depth;
        here_base=w->history+(t%depth)*w->history_plane+w->zs[0]+1;
        north_base=w->history+((t+depth-1)%depth)*w->history_plane+w->zs[0]+1;
        back_base=w->history+((t+depth-w->delta)%depth)*w->history_plane+w->zs[0];
        nb_base=w->history+((t+depth-w->delta-1)%depth)*w->history_plane+w->zs[0];
    }
    NG_TREG static __m256i bytes8(__m256i a,unsigned h){
        switch(h){case 0:return _mm256_cvtepu8_epi32(_mm256_castsi256_si128(a));case 1:return _mm256_cvtepu8_epi32(_mm_srli_si128(_mm256_castsi256_si128(a),8));case 2:return _mm256_cvtepu8_epi32(_mm256_extracti128_si256(a,1));default:return _mm256_cvtepu8_epi32(_mm_srli_si128(_mm256_extracti128_si256(a,1),8));}
    }
    NG_TREG static __m256i words8(__m512i a,unsigned h){
        switch(h){case 0:return _mm256_cvtepu16_epi32(_mm512_castsi512_si128(a));case 1:return _mm256_cvtepu16_epi32(_mm512_extracti32x4_epi32(a,1));case 2:return _mm256_cvtepu16_epi32(_mm512_extracti32x4_epi32(a,2));default:return _mm256_cvtepu16_epi32(_mm512_extracti32x4_epi32(a,3));}
    }
    template<bool Fused> NG_TREG void emit(unsigned f,unsigned h,__m256i value,__m256i base,__m512i&mask){
        if constexpr(Fused){if(m.used&(1u<<f)){auto index=_mm256_add_epi32(base,_mm256_add_epi32(value,_mm256_set1_epi32(f*m.stride+256)));mask=_mm512_and_si512(mask,_mm512_i32gather_epi64(index,m.table.data(),8));}}
        else _mm256_storeu_si256((__m256i*)(features+f*32+h*8),value);
    }
    template<bool Fused> NG_TREG __m256i route(unsigned h){
        unsigned x=column,j=h*8;int*f=features;unsigned used=m.used;
        auto bank=_mm256_srli_epi32(_mm256_loadu_si256((const __m256i*)(routed_roots.data()+x*32+j)),7);
        auto base=_mm256_mullo_epi32(bank,_mm256_set1_epi32(m.bank_stride));__m512i mask=_mm512_set1_epi64(-1);
        auto my=_mm256_cmpgt_epi32(_mm256_loadu_si256((const __m256i*)(w->ys+j)),_mm256_setzero_si256());
        auto mz=_mm256_cmpgt_epi32(_mm256_loadu_si256((const __m256i*)(w->zs+j)),_mm256_setzero_si256());
        auto west=x?words8(p0,h):_mm256_set1_epi32(128),ww=x>1?words8(p1,h):_mm256_set1_epi32(128);
        auto north=_mm256_blendv_epi8(_mm256_set1_epi32(128),bytes8(current_n,h),my),back=_mm256_blendv_epi8(_mm256_set1_epi32(128),bytes8(current_b,h),mz);
        emit<Fused>(0,h,_mm256_loadu_si256((const __m256i*)(routed_features.data()+x*9*32+j)),base,mask);
        emit<Fused>(8,h,_mm256_loadu_si256((const __m256i*)(routed_features.data()+(x*9+8)*32+j)),base,mask);
        {auto w=west;
#define NG_FV __m256i
#define NG_FOP1 _mm256_set1_epi32
#define NG_FOP6 _mm256_blendv_epi8
#define NG_FOP7 _mm256_min_epi32
#define NG_FOP8 _mm256_max_epi32
#define NG_FOP9(P,V) emit<Fused>(unsigned((int*)(P)-f)/32,h,V,base,mask)
#define NG_FOP10 _mm256_sub_epi32
#define NG_FOP11 _mm256_abs_epi32
#define NG_FOP12 _mm256_add_epi32
#define NG_FOP13 _mm256_and_si256
#define NG_FOP14 _mm256_mulhi_epu16
#define NG_FNEIGHBOR(K) bytes8(K==4?old_n:K==5?old_b:K==6?current_nb:old_nb,h)
#include "ng_tree_feature_math.inc"
#undef NG_FV
#undef NG_FOP1
#undef NG_FOP6
#undef NG_FOP7
#undef NG_FOP8
#undef NG_FOP9
#undef NG_FOP10
#undef NG_FOP11
#undef NG_FOP12
#undef NG_FOP13
#undef NG_FOP14
#undef NG_FNEIGHBOR
        }
        if(!used)mask=_mm512_set1_epi64(1);
        auto leaves=_mm512_cvtepi64_epi32(_mm512_sub_epi64(_mm512_set1_epi64(63),_mm512_lzcnt_epi64(mask)));
        return _mm256_add_epi32(leaves,_mm256_loadu_si256((const __m256i*)(routed_groups.data()+x*32+j)));
    }
    template<bool Fused,int Shift=-1> NG_TREG void step(){
        unsigned x=column;const size_t stride=w->history_stride;
        current_n=_mm256_loadu_si256((const __m256i*)(north_base+x*stride));current_b=_mm256_loadu_si256((const __m256i*)(back_base+x*stride));current_nb=_mm256_loadu_si256((const __m256i*)(nb_base+x*stride));
        __m256i groups[4];
        for(unsigned h=0;h<4;h++)groups[h]=route<Fused>(h);
        __m256i sy;
        if constexpr(Fused)sy=re.step(groups,w->lanes);
        else{
            int leaves[32];uint8_t ids[32],symbols[32];m.classify512(32,features,routed_roots.data()+x*32,leaves,nullptr,true);
            context_groups(routed_groups.data()+x*32,leaves,ids);
            if(re.compact)require(context_entropy_compact(entropy,ids,symbols,re.compact)==0);else require(context_entropy_wide(entropy,ids,symbols)==0);
            sy=_mm256_loadu_si256((const __m256i*)symbols);
        }
        auto zeros=_mm256_loadu_si256((const __m256i*)(routed_zeros.data()+x*32));
        require(_mm256_testz_si256(_mm256_cmpeq_epi8(zeros,_mm256_set1_epi8(1)),sy));
        auto*c=static_cast<const ng_bank_compact_column*>(w->columns)+x;
        if constexpr(Fused)tree_register_reconstruct(c,w->block_weights[x>>(Shift<0?choice_shift:Shift)],p0,p1,p2,_mm512_cvtepu8_epi16(sy),w->values+x*32,routed_errors.data()+x*32);
        else{
            alignas(64) uint16_t previous[3][32];uint8_t symbols[32];
            _mm512_store_si512(previous[0],p0);_mm512_store_si512(previous[1],p1);_mm512_store_si512(previous[2],p2);_mm256_storeu_si256((__m256i*)symbols,sy);
            ng_route_compact_symbols(c,w->block_weights[x>>(Shift<0?choice_shift:Shift)],previous,w->values+x*32,symbols,routed_errors.data()+x*32);
            p0=_mm512_load_si512(previous[0]);p1=_mm512_load_si512(previous[1]);p2=_mm512_load_si512(previous[2]);
        }
        _mm256_storeu_si256((__m256i*)(here_base+x*stride),_mm512_cvtepi16_epi8(p0));
        old_n=current_n;old_b=current_b;old_nb=current_nb;++column;
    }
    template<bool Fused> NG_TREG void finish(){
        if constexpr(Fused){re.finish(entropy);require(!re.over);}
        _mm512_storeu_si512(w->previous[0],p0);_mm512_storeu_si512(w->previous[1],p1);_mm512_storeu_si512(w->previous[2],p2);
        for(unsigned j=0;j<w->lanes;j++)for(unsigned x=0;x<w->width;x++)w->errors[w->error_offsets[j]+x]=routed_errors[x*32+j];
    }
};
template<bool Fused> __attribute__((target("avx512f,avx512bw,avx512vl,avx512cd,avx2,popcnt")))
static int run_tree_register(const Snapshot&s,const entropy_tree::VectorForest&v,const entropy_tree::MaskForest&m,ng_route_wave_data*w,ng_context_rans*e,const TreeCompactEntropy*c,TreeColumnMetadata&buffers,unsigned G){
    TreeRegisterWave state(s,v,m,w,e,c,std::move(buffers),G);
    if(ng_choice_edge==4){for(unsigned x=0;x<w->width;x++)state.step<Fused,2>();}
    else {for(unsigned x=0;x<w->width;x++)state.step<Fused>();}
    state.finish<Fused>();buffers=std::move(static_cast<TreeColumnMetadata&>(state));return 0;
}
#undef NG_TREG
#endif
