#pragma once
/* Direct/delta-only frames need no fitted model history. */
static void ng_bank_delta_reconstruct(const uint8_t*choices,uint8_t*out){
    const unsigned edge=ng_choice_edge,nb=VCN_LC_D/edge,ny=VCN_LC_Y/edge;
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++){
        const uint8_t*row=choices+((z/edge)*ny+y/edge)*nb;
        size_t o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        for(int bx=0;bx<nb;bx++){
            unsigned base=row[bx];
            if(!base)continue;
            for(int x=bx*edge;x<(bx+1)*edge;x++)
                out[o+x]=(uint8_t)(out[o+x]+ng_directional_base_xyz(out,o+x,base,x,y,z,VCN_LC_D,(size_t)VCN_LC_D*VCN_LC_Y));
        }
    }
}

/* Lane-major bank inputs. Every lane is an independent scheduled row. The
 * inner lane loop is vectorized; x remains serial for the three causal taps.
 * All products reduce modulo 2^16, exactly like native NG. */
#include "ng_wave_column.h"

#if VCN_HAVE_AVX2
#define NG_RATTR __attribute__((target("avx2"),always_inline))
#define NG_RNAME ng_bank_compact_avx2_impl
#define NG_RV __m256i
#define NG_RHALVES 2
#define NG_RSTEP 16
#define NG_RSET _mm256_set1_epi16
#define NG_RLOAD _mm256_loadu_si256
#define NG_RSTORE _mm256_storeu_si256
#define NG_RAND _mm256_and_si256
#define NG_REQ _mm256_cmpeq_epi16
#define NG_RBLEND _mm256_blendv_epi8
#define NG_RMASK _mm256_movemask_epi8
#define NG_RADD _mm256_add_epi16
#define NG_RMUL _mm256_mullo_epi16
#define NG_RMIN _mm256_min_epi16
#define NG_RMAX _mm256_max_epi16
#define NG_RSRL _mm256_srli_epi16
#define NG_RSRA _mm256_srai_epi16
#define NG_RSLL _mm256_slli_epi16
#define NG_RSUB _mm256_sub_epi16
#define NG_RXOR _mm256_xor_si256
#define NG_ROUT(P,V) _mm_storeu_si128((__m128i*)(P),_mm_packus_epi16(_mm256_castsi256_si128(V),_mm256_extracti128_si256(V,1)))
#include "ng_recur_vector.h"
#ifdef NG_HAVE_WIDE_PRE
#define NG_RNAME ng_bank_compact_avx512_mask_impl
#include "ng_recur_wide.h"
#define NG_RNAME ng_bank_fused_base_impl
#define NG_RPACKED 1
#define NG_RFUSED 1
#include "ng_recur_wide.h"
__attribute__((target("avx512f,avx512bw,avx512vl"))) static void ng_bank_fused_base(int width,const ng_bank_compact_column*cols,const uint16_t w[3][32],const uint16_t*feature,uint16_t prev[3][32],uint8_t*result,const uint8_t flags[][32],const uint8_t raw[][32],const uint8_t*up,const uint8_t*back,const uint8_t*upback,int stride){
    ng_bank_fused_base_impl(width,cols,w,feature,prev,result,0,flags,raw,up,back,upback,stride);
}
__attribute__((target("avx512f,avx512bw,avx512vl"))) static void ng_bank_compact_wide(int width,const ng_bank_compact_column*cols,const uint16_t w[3][32],const uint16_t*feature,uint16_t prev[3][32],uint8_t*result,int homogeneous){
    if(homogeneous)ng_bank_compact_avx512_mask_impl(width,cols,w,feature,prev,result,1);
    else ng_bank_compact_avx512_mask_impl(width,cols,w,feature,prev,result,0);
}
#endif
__attribute__((target("avx2"))) static void ng_bank_compact_avx2(int width,const ng_bank_compact_column*cols,const uint16_t w[3][32],const uint16_t*feature,uint16_t prev[3][32],uint8_t*result,int homogeneous){
#ifdef NG_HAVE_WIDE_PRE
if(ng_use_avx512()&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl")){
        ng_bank_compact_wide(width,cols,w,feature,prev,result,homogeneous);return;
    }
#endif
    if(homogeneous)ng_bank_compact_avx2_impl(width,cols,w,feature,prev,result,1);
    else ng_bank_compact_avx2_impl(width,cols,w,feature,prev,result,0);
}
#endif

#if VCN_HAVE_AVX2
/* Expand the on-wire selector grid without changing the SIMD pixel width. */
__attribute__((target("avx2"),always_inline)) static inline __m128i ng_choice_pixels16(const uint8_t*b,int x,unsigned edge){
    switch(edge){
    case 2:return _mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)(b+x/2)),_mm_setr_epi8(0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7));
    case 4:{uint32_t v;memcpy(&v,b+x/4,4);return _mm_shuffle_epi8(_mm_cvtsi32_si128(v),_mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3));}
    case 8:{uint16_t v;memcpy(&v,b+x/8,2);return _mm_shuffle_epi8(_mm_cvtsi32_si128(v),_mm_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1));}
    case 16:return _mm_set1_epi8(b[x/16]);
    default:return _mm_set1_epi8(b[x/32]);
    }
}
/* Full bank metadata preparation, 16 adjacent x positions at once. Shared
 * model/feature coefficients are broadcast, choices expanded from 4³ blocks.
 * Boundary masks exactly match ng_directional_parts; all sums are int16. */
__attribute__((target("avx2"))) static void ng_block_flags(const uint8_t*choices,uint8_t*out,int y,int z,int fast){
    const __m128i masks=_mm_setr_epi8(0,1,2,3,4,5,6,7,3,5,6,7,0,0,0,0);
    const __m128i avail=_mm_set1_epi8(1|(y?2:0)|(z?4:0));
    for(int bx=0;bx<VCN_LC_D/ng_choice_edge;bx+=16){
        __m128i code;
        if(VCN_LC_D/ng_choice_edge-bx<16){uint8_t tail[16]={0};memcpy(tail,choices+bx,VCN_LC_D/ng_choice_edge-bx);code=_mm_loadu_si128((const __m128i*)tail);}
        else code=_mm_loadu_si128((const __m128i*)(choices+bx));
        __m128i ng=_mm_cmpgt_epi8(code,_mm_set1_epi8(11));
        __m128i m=_mm_sub_epi8(code,_mm_and_si128(ng,_mm_set1_epi8(12)));
        __m128i original=_mm_shuffle_epi8(masks,m),mask=_mm_and_si128(original,avail);
        __m128i left=_mm_and_si128(mask,_mm_set1_epi8(1));
        __m128i clamp=_mm_and_si128(_mm_cmpgt_epi8(m,_mm_set1_epi8(7)),_mm_cmpeq_epi8(mask,original));
        __m128i f=_mm_or_si128(_mm_and_si128(ng,_mm_set1_epi8(1)),_mm_or_si128(_mm_add_epi8(left,left),_mm_and_si128(clamp,_mm_set1_epi8(4))));
        if(fast)f=_mm_or_si128(f,_mm_and_si128(_mm_cmpeq_epi8(m,_mm_set1_epi8(1)),_mm_set1_epi8(8)));
        f=_mm_or_si128(f,_mm_slli_epi16(mask,4));
        _mm_storeu_si128((__m128i*)(out+bx),f);
    }
}
__attribute__((target("avx2"),always_inline)) static inline void ng_bank_prepare_avx2_impl(
    const uint8_t* row,const uint8_t* raw,const uint8_t* choices,const uint16_t* pre,
    int y,int z,const vcn_lc_par* p,uint16_t feature,uint16_t* const fields[11],int compact,int fast,int xyz_shortcut,unsigned edge){
    const __m256i zero=_mm256_setzero_si256(),one=_mm256_set1_epi16(1),maxbyte=_mm256_set1_epi16(255);
    const __m128i masks=_mm_setr_epi8(0,1,2,3,4,5,6,7,3,5,6,7,0,0,0,0);
    const __m256i avail=_mm256_set1_epi16(1|(y?2:0)|(z?4:0));
    const int nr=p->t.nrow;const int32_t*w=p->w[z==0];
    for(int x=0;x<VCN_LC_D;x+=16){
        __m128i c8=ng_choice_pixels16(choices,x,edge);
        if(compact&&_mm_testz_si128(c8,c8)){
            _mm256_storeu_si256((__m256i*)(fields[0]+x),zero);
            _mm256_storeu_si256((__m256i*)(fields[1]+x),zero);
            __m256i rawv=_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(raw+x)));
            _mm256_storeu_si256((__m256i*)(fields[2]+x),_mm256_slli_epi16(rawv,8));
            _mm256_storeu_si256((__m256i*)(fields[3]+x),zero);
            continue;
        }
        if(fast&&compact&&_mm_movemask_epi8(_mm_cmpeq_epi8(c8,_mm_set1_epi8(13)))==65535){
            _mm256_storeu_si256((__m256i*)(fields[0]+x),_mm256_loadu_si256((const __m256i*)(pre+x)));
            _mm256_storeu_si256((__m256i*)(fields[1]+x),zero);
            __m256i flags=_mm256_set1_epi16(11);
            if(!x)flags=_mm256_and_si256(flags,_mm256_setr_epi16(9,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11));
            __m256i rawv=_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(raw+x)));
            flags=_mm256_or_si256(flags,_mm256_slli_epi16(rawv,8));
            _mm256_storeu_si256((__m256i*)(fields[2]+x),flags);
            _mm256_storeu_si256((__m256i*)(fields[3]+x),zero);
            continue;
        }
        if(xyz_shortcut&&compact&&_mm_movemask_epi8(_mm_cmpeq_epi8(c8,_mm_set1_epi8(23)))==65535&&x&&y&&z){
#define NG_XYZ_LOAD(O) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row+x+(O))))
            __m256i b=NG_XYZ_LOAD(-VCN_LC_RP),c=NG_XYZ_LOAD(-VCN_LC_SP);
            __m256i base=_mm256_sub_epi16(_mm256_add_epi16(b,c),_mm256_add_epi16(NG_XYZ_LOAD(-VCN_LC_RP-1),NG_XYZ_LOAD(-VCN_LC_SP-1)));
            base=_mm256_add_epi16(base,_mm256_sub_epi16(NG_XYZ_LOAD(-VCN_LC_SP-VCN_LC_RP-1),NG_XYZ_LOAD(-VCN_LC_SP-VCN_LC_RP)));
            __m256i rawv=_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(raw+x)));
            __m256i flags=_mm256_set1_epi16(7);
            flags=_mm256_or_si256(flags,_mm256_slli_epi16(rawv,8));
            _mm256_storeu_si256((__m256i*)(fields[0]+x),_mm256_loadu_si256((const __m256i*)(pre+x)));
            _mm256_storeu_si256((__m256i*)(fields[1]+x),base);
            _mm256_storeu_si256((__m256i*)(fields[2]+x),flags);
            _mm256_storeu_si256((__m256i*)(fields[3]+x),_mm256_or_si256(_mm256_min_epu16(b,c),_mm256_slli_epi16(_mm256_max_epu16(b,c),8)));
#undef NG_XYZ_LOAD
            continue;
        }
        __m128i isng8=_mm_cmpgt_epi8(c8,_mm_set1_epi8(11));
        __m128i m8=_mm_sub_epi8(c8,_mm_and_si128(isng8,_mm_set1_epi8(12)));
        __m256i isng=_mm256_cvtepi8_epi16(isng8),m=_mm256_cvtepu8_epi16(m8);
        __m256i original=_mm256_cvtepu8_epi16(_mm_shuffle_epi8(masks,m8));
        __m256i mask=_mm256_and_si256(original,avail);
        if(!x)mask=_mm256_and_si256(mask,_mm256_setr_epi16(6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7));
#define NG_BANK_HAS(B) _mm256_cmpeq_epi16(_mm256_and_si256(mask,_mm256_set1_epi16(B)),_mm256_set1_epi16(B))
#define NG_BANK_LOAD(O) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row+x+(O))))
        __m256i hasy=NG_BANK_HAS(2),hasz=NG_BANK_HAS(4);
        __m256i b=_mm256_and_si256(hasy,NG_BANK_LOAD(-VCN_LC_RP));
        __m256i c=_mm256_and_si256(hasz,NG_BANK_LOAD(-VCN_LC_SP));
        __m256i base=_mm256_add_epi16(b,c);
        base=_mm256_sub_epi16(base,_mm256_and_si256(NG_BANK_HAS(3),NG_BANK_LOAD(-VCN_LC_RP-1)));
        base=_mm256_sub_epi16(base,_mm256_and_si256(NG_BANK_HAS(5),NG_BANK_LOAD(-VCN_LC_SP-1)));
        base=_mm256_sub_epi16(base,_mm256_and_si256(NG_BANK_HAS(6),NG_BANK_LOAD(-VCN_LC_SP-VCN_LC_RP)));
        base=_mm256_add_epi16(base,_mm256_and_si256(NG_BANK_HAS(7),NG_BANK_LOAD(-VCN_LC_SP-VCN_LC_RP-1)));
        __m256i lo=_mm256_min_epu16(_mm256_blendv_epi8(maxbyte,b,hasy),_mm256_blendv_epi8(maxbyte,c,hasz));
        __m256i clamp=_mm256_and_si256(one,_mm256_and_si256(_mm256_cmpgt_epi16(m,_mm256_set1_epi16(7)),_mm256_cmpeq_epi16(mask,original)));
        __m256i f=_mm256_andnot_si256(_mm256_cmpeq_epi16(m,zero),_mm256_set1_epi16(feature));
        f=_mm256_blendv_epi8(_mm256_set1_epi16(256),f,isng);
#define NG_BANK_STORE(F,V) do {_mm256_storeu_si256((__m256i*)(fields[F]+x),V);} while(0)
        NG_BANK_STORE(0,_mm256_and_si256(isng,_mm256_loadu_si256((const __m256i*)(pre+x))));
        if(compact){
            NG_BANK_STORE(1,base);
            __m256i flags=_mm256_or_si256(_mm256_and_si256(isng,one),_mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(mask,one),1),_mm256_slli_epi16(clamp,2)));
            if(fast)flags=_mm256_or_si256(flags,_mm256_and_si256(_mm256_cmpeq_epi16(m,one),_mm256_set1_epi16(8)));
            __m256i rawv=_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(raw+x)));
            flags=_mm256_or_si256(flags,_mm256_slli_epi16(rawv,8));
            NG_BANK_STORE(2,flags);
            NG_BANK_STORE(3,_mm256_or_si256(lo,_mm256_slli_epi16(_mm256_max_epu16(b,c),8)));
        }else{
        NG_BANK_STORE(1,_mm256_and_si256(isng,_mm256_set1_epi16(nr>0?w[0]:0)));
        NG_BANK_STORE(2,_mm256_and_si256(isng,_mm256_set1_epi16(nr>1?w[1]:0)));
        NG_BANK_STORE(3,_mm256_and_si256(isng,_mm256_set1_epi16(nr>2?w[2]:0)));
        NG_BANK_STORE(4,f);NG_BANK_STORE(5,_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(raw+x))));
        NG_BANK_STORE(6,base);NG_BANK_STORE(7,_mm256_and_si256(mask,one));NG_BANK_STORE(8,clamp);
        NG_BANK_STORE(9,lo);NG_BANK_STORE(10,_mm256_max_epu16(b,c));
        }
#undef NG_BANK_STORE
#undef NG_BANK_HAS
#undef NG_BANK_LOAD
    }
}
__attribute__((target("avx2"))) static void ng_bank_prepare_avx2(
    const uint8_t*row,const uint8_t*raw,const uint8_t*choices,const uint16_t*pre,int y,int z,const vcn_lc_par*p,uint16_t feature,uint16_t*const fields[11],int compact,int fast,int xyz_shortcut){
    const unsigned edge=ng_choice_edge;
    /* Keep constant instantiations for the three live preparation layouts. */
#define NG_PREP(C,X) do { if(edge==4)ng_bank_prepare_avx2_impl(row,raw,choices,pre,y,z,p,feature,fields,C,fast,X,4);else ng_bank_prepare_avx2_impl(row,raw,choices,pre,y,z,p,feature,fields,C,fast,X,edge); } while(0)
    if(!compact)NG_PREP(0,0);
    else if(xyz_shortcut)NG_PREP(1,1);
    else NG_PREP(1,0);
#undef NG_PREP
}
#endif

/* Shared entropy dispatch for whole groups and row-interleaved scheduling. */
static inline void ng_bank_read_symbols(int count,const uint8_t*classes,vcn_lc_dec*decoder,uint8_t*symbols,int optimized){
#if VCN_HAVE_AVX2
            if(vcn_use_avx2()){
#ifdef NG_HAVE_WIDE_PRE
                if(optimized&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl"))ng_syms_register(count,classes,decoder,symbols);
                else
#endif
                if(optimized)vcn_lc_syms_gather_avx2(count,classes,decoder,symbols);
                else vcn_lc_syms_avx2(count,classes,decoder,symbols);
            }else
#elif VCN_HAVE_NEON
            vcn_lc_syms_neon(count,classes,decoder,symbols);
            return; /* Do not consume the same symbols again in scalar code. */
#endif
            vcn_lc_syms_c(count,classes,decoder,symbols);
}

#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static void ng_entropy_bank_row_avx2(uint8_t*c,const uint8_t*b,unsigned K){
    const unsigned edge=ng_choice_edge;
    const __m128i lookup=_mm_setr_epi8(0,K,2*K,3*K,4*K,5*K,6*K,7*K,8*K,9*K,10*K,11*K,12*K,13*K,14*K,15*K);
    for(int x=0;x<VCN_LC_D;x+=16){
        __m128i ids=ng_choice_pixels16(b,x,edge);
        __m128i offsets=K==16?_mm_slli_epi16(ids,4):_mm_shuffle_epi8(lookup,ids);
        _mm_storeu_si128((__m128i*)(c+x),_mm_add_epi8(_mm_loadu_si128((const __m128i*)(c+x)),offsets));
    }
}
#endif
static void ng_entropy_bank_row(uint8_t*c,const uint8_t*b,unsigned K){
#if VCN_HAVE_AVX2
    if(vcn_use_avx2()){ng_entropy_bank_row_avx2(c,b,K);return;}
#endif
    for(int x=0;x<VCN_LC_D;x++)c[x]+=K*b[x/ng_choice_edge];
}
/* Output of one independent row wave; no entropy-routing callbacks. */
typedef struct {
    unsigned lanes;
    uint8_t *values, *history, *output;
    const size_t *offsets, *error_offsets;
    const int *ys, *zs;
    size_t history_plane;
    unsigned history_depth, history_stride, delta;
    void *publish_state;
} ng_published_wave;
static void ng_bank_publish_wave(ng_published_wave*w){
    ng_causal_state*h=(ng_causal_state*)w->publish_state;
    uint8_t*out=w->output,*wave_history=w->history;
    size_t history_plane=w->history_plane;unsigned history_depth=w->history_depth,history_stride=w->history_stride,dl=w->delta,nl=w->lanes;
    const int*ys=w->ys,*zs=w->zs;const size_t*offset=w->offsets,*ri=w->error_offsets;
    uint8_t(*values)[32]=(uint8_t(*)[32])w->values;
            if(wave_history){
                int t=ys[0]+dl*zs[0];
                uint8_t* dst=wave_history+(t%history_depth)*history_plane+zs[0]+1;
                /* Padding allows a full vector store. Later groups overwrite
                 * their own live lanes before any later wave reads them. */
                for(int x=0;x<VCN_LC_D;x++)memcpy(dst+(size_t)x*history_stride,values[x],32);
            }
            uint8_t* rows[32];for(int j=0;j<nl;j++)rows[j]=h->b+ri[j];
#if VCN_HAVE_AVX2
            if(vcn_use_avx2())vcn_lc_tr_v(nl,(const uint8_t(*)[32])values,rows);else
#elif VCN_HAVE_NEON
            vcn_lc_tr_v_neon(nl,(const uint8_t(*)[32])values,rows);
            if(0)
#endif
            for(int j=0;j<nl;j++)for(int x=0;x<VCN_LC_D;x++)rows[j][x]=values[x][j];
            for(int j=0;j<nl;j++){memcpy(out+offset[j],rows[j],VCN_LC_D);vcn_lc_row_end(h->b,zs[j],ys[j]);}

}
static int ng_bank_reconstruct_core(void* const* states,unsigned mode,const int32_t* weights,const uint8_t* choices,uint8_t* out,const uint8_t* entropy,size_t entropy_size,const ng_decode_context*ctx){
    const unsigned edge=ng_choice_edge;
    const unsigned edge_shift=edge==2?1:edge==4?2:edge==8?3:edge==16?4:5;
    const unsigned count=ctx->model_count;
    if(count<1||count>4)return -1;
    int compact=1,vectorprep=1;
    int full=1;
    ng_causal_state* model[4];ng_causal_state* h=NULL;int dl=2;
    for(unsigned m=0;m<count;m++){model[m]=states[m];if(model[m]){if(model[m]->b)h=model[m];int d=vcn_lc_delta(&model[m]->p->t);if(d>dl)dl=d;
        if(model[m]->p->t.nrow>3)return -1;}}
    if(!h)return -1;
    if(compact){
#if VCN_HAVE_AVX2
        if(!vcn_use_avx2())compact=vectorprep=0;
#else
        compact=vectorprep=0;
#endif
    }
    int common=count==1;
    int packed_metadata=0;
#ifdef NG_HAVE_WIDE_PRE
    packed_metadata=compact&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl");
#endif
    if(common&&mode){int first=-1;for(unsigned m=1;m<12;m++){
        if(first>=0&&(weights[2*m]!=weights[2*first]||weights[2*m+1]!=weights[2*first+1]))common=0;
        first=(int)m;
    }}
    if(full&&!common){
        // Independent fitted models use the existing block-wise wavefront
        // reconstruction below. Keep the entropy stream/schedule unchanged;
        // shared-model-only whole-row preparation is not applicable.
        full=count>1&&ctx->banks;compact=vectorprep=packed_metadata=0;
#if VCN_HAVE_AVX2
        if(full&&ctx->banks&&mode&&vcn_use_avx2())compact=1;
#endif
    }

    if(edge!=4&&!full){full=1;compact=vectorprep=packed_metadata=0;}
    uint16_t common_feature[2]={256,256};int feature_model=-1;
    int delta_only=common||compact;
    if(delta_only)for(size_t i=0;i<(size_t)((VCN_LC_D>>edge_shift))*((VCN_LC_Y>>edge_shift))*((VCN_LC_Z>>edge_shift));i++)if(choices[i]>=8){delta_only=0;break;}
    /* The direct/delta shortcut never consumes recurrence metadata. */
    if(delta_only)packed_metadata=0;
    if(vectorprep&&mode)for(unsigned m=1;m<12;m++){
        if(feature_model>=0&&(weights[2*m]!=weights[2*feature_model]||weights[2*m+1]!=weights[2*feature_model+1]))return -1;
        feature_model=m;common_feature[0]=weights[2*m];common_feature[1]=weights[2*m+1];
    }
    vcn_lc_sched sc={0};ng_bank_wave_column cols[4]={0};uint8_t values[4*32];
    vcn_lc_par* ep=NULL;vcn_lc_dec decoder={0};uint32_t* tables=NULL;uint32_t* class_table=NULL;
    uint8_t (*entropy_classes)[VCN_LC_D]=NULL,(*entropy_symbols)[VCN_LC_D]=NULL;
    uint16_t entropy_dummy[VCN_LC_D];memset(entropy_dummy,0,sizeof(entropy_dummy));
    uint16_t* activity=malloc((size_t)VCN_LC_D*32*sizeof(*activity));int rc=-1;
    int need_multi_pre=(count>1)||(edge!=4&&!common);
    uint16_t* multi_pre=need_multi_pre?calloc((count)*32*VCN_LC_D,sizeof(uint16_t)):NULL;
    if(need_multi_pre&&!multi_pre){free(activity);return -1;}
    uint16_t* shared_pre=common?calloc((size_t)VCN_LC_D*32,sizeof(*shared_pre)):NULL;
    unsigned* needed_rows=common?calloc(((VCN_LC_Y>>edge_shift))*((VCN_LC_Z>>edge_shift)),sizeof(unsigned)):NULL;
    /* A cache-line skew between field planes avoids eleven stores to the
     * same cache set on every voxel (128-wide planes otherwise alias). */
    size_t field_stride=32*VCN_LC_D+(common||compact||edge!=4?32:0);
    int nf=compact?4:11;
    uint16_t* field_storage=full?calloc(nf*field_stride,sizeof(uint16_t)):NULL;
    uint16_t (*fields[11])[VCN_LC_D];
    for(int f=0;f<11;f++)fields[f]=field_storage&&f<nf?(uint16_t(*)[VCN_LC_D])(field_storage+f*field_stride):NULL;
    uint16_t (*transposed)[32]=full?calloc(VCN_LC_D,sizeof(*transposed)):NULL;
    ng_bank_wave_column* fullcols=full?calloc(VCN_LC_D,sizeof(*fullcols)):NULL;
    ng_bank_compact_column* compactcols=compact?calloc(VCN_LC_D,sizeof(*compactcols)):NULL;
    uint8_t (*fullvalues)[32]=full?calloc(VCN_LC_D,sizeof(*fullvalues)):NULL;
    /* Scheduler waves have t=y+dl*z and consecutive z lanes. Keep only
     * the t-1/t-dl/t-dl-1 predecessors, with padding for full32-byte loads.
     * Existing raster history remains authoritative for fitted NG taps. */
    /* Cache-line-aligned column starts; retain one dummy z lane and room
     * for a complete32-byte read at the final partial group. */
    const int history_stride=((VCN_LC_Z+95)&~63),history_depth=dl+2;
    const size_t history_plane=(size_t)VCN_LC_D*history_stride;
    uint8_t* history_owner=packed_metadata?calloc(1,history_depth*history_plane+63):NULL;
    uint8_t* wave_history=history_owner?(uint8_t*)(((uintptr_t)history_owner+63)&~(uintptr_t)63):history_owner;
    if(packed_metadata&&!wave_history)goto done;
    if(!activity||(common&&!shared_pre)||(compact&&!compactcols)||(full&&(!field_storage||!transposed||!fullcols||!fullvalues))||vcn_lc_mksched(dl,&sc))goto done;

    if(common){
        if(!needed_rows)goto done;
        for(int r=0;r<((VCN_LC_Y>>edge_shift))*((VCN_LC_Z>>edge_shift));r++)for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){
            unsigned c=choices[r*((VCN_LC_D>>edge_shift))+bx];if(c>=24||(c>=12&&!model[(c-12)/12]))goto done;
            if(c>=12)needed_rows[r]|=1u<<((bx*edge)/32);
        }
    }

    /* The entrypoint requires entropy. Keep phase guards in this large loop:
     * deleting them worsens compiler-generated code (~8% measured). There is
     * no residual-only fallback; this is not a selectable decoder mode. */
    if(entropy){
        const uint8_t* cursor=entropy,*end=entropy+entropy_size;
        if(vcn_lc_readpar(&cursor,entropy_size,&ep)!=VCN_OK||vcn_lc_delta(&ep->t)!=dl)goto done;
        if(ctx->banks?(ep->K<ctx->classes||ep->K%ctx->classes):(ep->K<2||ep->K>32))goto done;
        tables=malloc((size_t)ep->K*(1u<<ep->tb)*sizeof(*tables));
        entropy_classes=calloc(32,sizeof(*entropy_classes));entropy_symbols=calloc(32,sizeof(*entropy_symbols));
        if(!tables||!entropy_classes||!entropy_symbols||vcn_lc_read_tables(&cursor,end,tables,ep->K,1u<<ep->tb))goto done;
        if(ctx->banks)ep->K=ctx->classes;
        vcn_lc_mklut(ep);
        if(cursor!=end)goto done;cursor=ctx->payload;end=cursor+ctx->payload_size;
        if(end-cursor<4*VCN_LC_NS)goto done;
        for(int k=0;k<VCN_LC_NS;k++)decoder.S[k]=vcn_rd32(cursor+4*k);
        decoder.T=tables;decoder.tb=ep->tb;decoder.p=cursor+4*VCN_LC_NS;decoder.end=end;

#if VCN_HAVE_AVX2
        if(common&&!delta_only&&vcn_use_avx2()){
            class_table=malloc(VCN_LC_AMAX*sizeof(*class_table));if(!class_table)goto done;
            for(int a=0;a<VCN_LC_AMAX;a++)class_table[a]=(uint16_t)h->p->cb[h->p->lut[a]]|((uint32_t)ep->lut[a]<<16);
        }
#endif
    }
    for(int g=0;g<sc.ng;g++){

        uint8_t packed_flags[VCN_LC_D][32],packed_raw[VCN_LC_D][32],flag_rows[32][VCN_LC_D];
        if(packed_metadata)memset(packed_flags,0,sizeof(packed_flags));
        if(packed_metadata)memset(flag_rows,0,sizeof(flag_rows));
        uint16_t prev[3][32]={{0}},row_weights[3][32]={{0}},row_feature[32]={0};int nl=sc.n[g];
        size_t ri[32],offset[32];int zs[32]={0},ys[32]={0};const uint8_t* selected[32];
        for(int j=0;j<nl;j++){
            int entropy_class_ready=0;
            int z=zs[j]=sc.row[g][j]/VCN_LC_Y,y=ys[j]=sc.row[g][j]%VCN_LC_Y;ri[j]=VCN_LC_IDX(z,y,0);offset[j]=(size_t)sc.row[g][j]*VCN_LC_D;
            if(delta_only){
                vcn_lc_row_begin(h->e,z,y,0);VCN_LC_CLS(h->e,ri[j],ep,entropy_dummy,entropy_classes[j]);
                if(ctx->banks)ng_entropy_bank_row(entropy_classes[j],ctx->banks+(((z>>edge_shift))*((VCN_LC_Y>>edge_shift))+(y>>edge_shift))*((VCN_LC_D>>edge_shift)),ctx->classes);
                continue;
            }
            selected[j]=choices+(((z>>edge_shift))*((VCN_LC_Y>>edge_shift))+(y>>edge_shift))*((VCN_LC_D>>edge_shift));
            if(packed_metadata){
#ifdef NG_HAVE_WIDE_PRE
                ng_block_flags(selected[j],flag_rows[j],y,z,ctx->frequent_x);
#endif
            }
            vcn_lc_row_begin(h->b,z,y,128);vcn_lc_row_begin(h->e,z,y,0);
            for(int k=0;k<3;k++)prev[k][j]=h->b[ri[j]-1-k];
            for(int k=0;k<3;k++)row_weights[k][j]=k<h->p->t.nrow?h->p->w[z==0][k]:0;
            row_feature[j]=common_feature[z==0];

            if(vectorprep){
                unsigned needed=0;
                if(needed_rows)needed=needed_rows[((z>>edge_shift))*((VCN_LC_Y>>edge_shift))+(y>>edge_shift)];
                else for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){unsigned c=selected[j][bx];if(c>=24||(c>=12&&!model[(c-12)/12]))goto done;if(c>=12)needed|=1u<<((bx*edge)/32);}
                uint16_t* pp=shared_pre+j*VCN_LC_D;
                if(needed){
#ifdef NG_HAVE_WIDE_PRE
                    if(class_table&&VCN_LC_D>=64&&VCN_LC_D%64==0&&needed==(UINT32_MAX>>(32-VCN_LC_D/32))&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw")){
                        ng_pre_class_fused(h->b+ri[j],&h->pl[z==0],(uint16_t)h->p->bias[z==0],pp,needed,h->e,ri[j],class_table,entropy_classes[j]);
                        entropy_class_ready=1;
                    }else
#endif
                    {
#ifdef NG_HAVE_WIDE_PRE
                    if(VCN_LC_D>=64&&VCN_LC_D%64==0&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw"))
                        vcn_lc_pre_avx512_impl(h->b+ri[j],&h->pl[z==0],(uint16_t)h->p->bias[z==0],pp,needed);
                    else
#endif
                    if(needed_rows&&needed==(UINT32_MAX>>(32-VCN_LC_D/32)))VCN_LC_PRE(h->b+ri[j],&h->pl[z==0],(uint16_t)h->p->bias[z==0],pp);
                    else vcn_lc_pre_selected(h->b+ri[j],&h->pl[z==0],(uint16_t)h->p->bias[z==0],pp,needed);
#if VCN_HAVE_AVX2
                    if(vcn_use_avx2()){
                        /* Shared-model AVX2 always has the fused lookup table. */
#ifdef NG_HAVE_WIDE_PRE
                            if(ng_use_avx512()&&__builtin_cpu_supports("avx512bw"))ng_bank_class_wide(h->e,ri[j],class_table,pp,entropy_classes[j]);
                            else
#endif
                            ng_bank_class_lookup(h->e,ri[j],class_table,pp,entropy_classes[j]);
                        entropy_class_ready=1;
                    }else
#endif
                    VCN_LC_CLS(h->e,ri[j],h->p,pp,h->classes);
                    }
                }
            }else{vcn_lc_apre(h->e,ri[j],activity+j*VCN_LC_D);
            if(common){
                uint16_t* pp=shared_pre+j*VCN_LC_D;
                VCN_LC_PRE(h->b+ri[j],&h->pl[z==0],(uint16_t)h->p->bias[z==0],pp);
                for(int x=0;x<VCN_LC_D;x++)pp[x]=(uint16_t)(pp[x]+h->p->cb[h->p->lut[activity[j*VCN_LC_D+x]]]);
            }else if(multi_pre){
                {
                unsigned used=0,needed[4]={0};for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){unsigned c=selected[j][bx];if(c>=12+12*count)goto done;if(c>=12){used|=1u<<((c-12)/12);needed[(c-12)/12]|=1u<<((bx*edge)/32);}}
                for(unsigned k=0;k<count;k++)if(used&(1u<<k)){
                    ng_causal_state*s=model[k];if(!s)goto done;
                    uint16_t*q=multi_pre+(k*32+j)*VCN_LC_D;
#ifdef NG_HAVE_WIDE_PRE
                    if(VCN_LC_D>=64&&VCN_LC_D%64==0&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw"))vcn_lc_pre_avx512_impl(h->b+ri[j],&s->pl[z==0],(uint16_t)s->p->bias[z==0],q,needed[k]);else
#endif
                    vcn_lc_pre_selected(h->b+ri[j],&s->pl[z==0],(uint16_t)s->p->bias[z==0],q,needed[k]);
                    VCN_LC_CLS(h->e,ri[j],s->p,q,s->classes);
                }
                }
            }}
            if(entropy){
                if(!entropy_class_ready)VCN_LC_CLS(h->e,ri[j],ep,entropy_dummy,entropy_classes[j]);
                if(ctx->banks)ng_entropy_bank_row(entropy_classes[j],ctx->banks+(((z>>edge_shift))*((VCN_LC_Y>>edge_shift))+(y>>edge_shift))*((VCN_LC_D>>edge_shift)),ctx->classes);
                if(common){
                    /* Every row has a multiple of32 symbols: splitting the
                     * call preserves the interleaved rANS state order. */
                    ng_bank_read_symbols(VCN_LC_D,entropy_classes[j],&decoder,entropy_symbols[j],common||compact);
                    if(decoder.over)goto done;
                }
            }
        }

#ifdef NG_HAVE_WIDE_PRE
        if(packed_metadata)vcn_lc_tr_y_width(nl,(const uint8_t(*)[VCN_LC_D])flag_rows,packed_flags,((VCN_LC_D>>edge_shift)+15)&~15);
#endif
        if(entropy){
            if(!common||delta_only)ng_bank_read_symbols(nl*VCN_LC_D,entropy_classes[0],&decoder,entropy_symbols[0],common||compact);
            if(decoder.over)goto done;
            for(int j=0;j<nl;j++){
                const uint8_t*sy=entropy_symbols[j];
                uint8_t*dst=common&&compact&&!delta_only?entropy_symbols[j]:out+offset[j];
                uint8_t*restrict er=h->e+ri[j];
                for(int x=0;x<VCN_LC_D;x++){unsigned s=sy[x];dst[x]=(uint8_t)((s>>1)^-(s&1));er[x]=(uint8_t)((s+1)>>1);}
                vcn_lc_row_end(h->e,zs[j],ys[j]);
            }
        }
        if(delta_only)continue;
        if(full){
            if(packed_metadata){
#ifdef NG_HAVE_WIDE_PRE
                vcn_lc_tr_y(nl,(const uint8_t(*)[VCN_LC_D])entropy_symbols,packed_raw);
#endif
            }
            int homogeneous=compact&&common;
            if(homogeneous)for(int j=0;j<nl&&homogeneous;j++)
                for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++)if(selected[j][bx]!=13){homogeneous=0;break;}
            int fused_base=packed_metadata;
#ifdef NG_HAVE_WIDE_PRE
            const uint8_t* neighbor_ptrs[3]={0};
            int neighbor_stride=32;
            if(fused_base&&wave_history){
                int t=ys[0]+dl*zs[0];
                neighbor_ptrs[0]=wave_history+((t+history_depth-1)%history_depth)*history_plane+zs[0]+1;
                neighbor_ptrs[1]=wave_history+((t+history_depth-dl)%history_depth)*history_plane+zs[0];
                neighbor_ptrs[2]=wave_history+((t+history_depth-dl-1)%history_depth)*history_plane+zs[0];
                neighbor_stride=history_stride;
            }

#endif
            for(int j=0;j<nl;j++){
                int z=zs[j],y=ys[j];size_t o=offset[j];
                if(homogeneous||fused_base){
                    memcpy(fields[0][j],shared_pre+j*VCN_LC_D,VCN_LC_D*sizeof(uint16_t));
                    for(int x=0;x<VCN_LC_D;x++)fields[2][j][x]=(uint16_t)entropy_symbols[j][x]<<8;
                    continue;
                }
#if VCN_HAVE_AVX2
                if(!common&&multi_pre&&mode&&vcn_use_avx2()){
                    uint8_t local[VCN_LC_D/2];unsigned used=1;
                    for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){unsigned c=selected[j][bx];local[bx]=c<12?c:12+(c-12)%12;if(c>=12)used|=1u<<((c-12)/12);}
                    if(compact){
                        uint16_t pre[VCN_LC_D],*fp[11]={0};for(int f=0;f<nf;f++)fp[f]=fields[f][j];
                        for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){unsigned c=selected[j][bx];if(c>=12)memcpy(pre+edge*bx,multi_pre+(((c-12)/12)*32+j)*VCN_LC_D+edge*bx,2*edge);else memset(pre+edge*bx,0,2*edge);}
                        ng_bank_prepare_avx2(h->b+ri[j],out+o,local,pre,y,z,h->p,0,fp,1,0,0);
                        continue;
                    }
                    uint16_t scratch[11][VCN_LC_D];uint16_t*fp[11];for(int f=0;f<11;f++)fp[f]=scratch[f];
                    for(unsigned k=0;k<count;k++)if(used&(1u<<k)){
                        ng_causal_state*s=model[k];
                        ng_bank_prepare_avx2(h->b+ri[j],out+o,local,multi_pre+(k*32+j)*VCN_LC_D,y,z,s->p,weights[24*k+2+(z==0)],fp,compact,0,0);
                        for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){unsigned c=selected[j][bx];unsigned owner=c<12?0:(c-12)/12;
                            if(owner==k){
                                for(int f=0;f<nf;f++)memcpy(fields[f][j]+edge*bx,scratch[f]+edge*bx,2*edge);
                                if(!compact){unsigned m=c>=12?c-12:0;uint16_t value=c<12?256:m%12?weights[2*m+(z==0)]:0;
                                    for(int x=edge*bx;x<edge*(bx+1);x++)fields[4][j][x]=value;}
                            }
                        }
                    }
                    continue;
                }
                if(vectorprep&&vcn_use_avx2()){
                    uint16_t* fp[11]={0};for(int f=0;f<nf;f++)fp[f]=fields[f][j];
                    ng_bank_prepare_avx2(h->b+ri[j],entropy_symbols[j],selected[j],shared_pre+j*VCN_LC_D,y,z,h->p,common_feature[z==0],fp,compact,ctx->frequent_x,1);
                    continue;
                }
#endif
                for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){
                    unsigned c=selected[j][bx];if(c>=12+12*count)goto done;
                    unsigned m=c>=12?c-12:c;ng_causal_state*s=c>=12?model[m/12]:NULL;
                    if(c>=12&&!s)goto done;
                    const int32_t*w=s?s->p->w[z==0]:NULL;int nr=s?s->p->t.nrow:0;
                    uint16_t f=s?(m%12?(mode?weights[2*m+(z==0)]:256):0):256;
                    for(int x=edge*bx;x<edge*(bx+1);x++){
                        ng_base_parts p=ng_directional_parts_xyz(out,o+x,m%12,x,y,z,VCN_LC_D,(size_t)VCN_LC_D*VCN_LC_Y);
                        fields[0][j][x]=s?(common?shared_pre[j*VCN_LC_D+x]:multi_pre?multi_pre[((m/12)*32+j)*VCN_LC_D+x]:0):0;
                        fields[1][j][x]=nr>0?w[0]:0;fields[2][j][x]=nr>1?w[1]:0;fields[3][j][x]=nr>2?w[2]:0;
                        fields[4][j][x]=f;fields[5][j][x]=out[o+x];
                        fields[6][j][x]=p.constant;fields[7][j][x]=p.left;fields[8][j][x]=p.clamp;
                        fields[9][j][x]=p.lo;fields[10][j][x]=p.hi;
                    }
                }
            }
            const size_t offsets[]={offsetof(ng_bank_wave_column,pre),offsetof(ng_bank_wave_column,w0),offsetof(ng_bank_wave_column,w1),offsetof(ng_bank_wave_column,w2),offsetof(ng_bank_wave_column,feature),offsetof(ng_bank_wave_column,residual),offsetof(ng_bank_wave_column,constant),offsetof(ng_bank_wave_column,left),offsetof(ng_bank_wave_column,clamp),offsetof(ng_bank_wave_column,lo),offsetof(ng_bank_wave_column,hi)};
            const size_t compact_offsets[]={offsetof(ng_bank_compact_column,pre),offsetof(ng_bank_compact_column,constant),offsetof(ng_bank_compact_column,flags),offsetof(ng_bank_compact_column,bounds),offsetof(ng_bank_compact_column,residual)};
            for(int f=0;f<nf;f++){
                if(packed_metadata&&f==2)continue;
                if(fused_base&&f!=0)continue;
                if(homogeneous&&f!=0&&f!=2)continue;
#if VCN_HAVE_AVX2
                if(compact&&vcn_use_avx2()){
                    vcn_lc_tr_a_stride(nl,(const uint16_t(*)[VCN_LC_D])fields[f],
                        (uint16_t*)((uint8_t*)compactcols+compact_offsets[f]),sizeof(*compactcols)/sizeof(uint16_t));
                    continue;
                }
                if(vcn_use_avx2())vcn_lc_tr_a(nl,(const uint16_t(*)[VCN_LC_D])fields[f],transposed);else
#elif VCN_HAVE_NEON
                vcn_lc_tr_a_neon(nl,(const uint16_t(*)[VCN_LC_D])fields[f],transposed);
                if(0)
#endif
                for(int j=0;j<nl;j++)for(int x=0;x<VCN_LC_D;x++)transposed[x][j]=fields[f][j][x];
                for(int x=0;x<VCN_LC_D;x++)memcpy(compact?(uint8_t*)(compactcols+x)+compact_offsets[f]:(uint8_t*)(fullcols+x)+offsets[f],transposed[x],64);
            }
#if VCN_HAVE_AVX2
             if(compact){
                if(!common){
                    for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){
                        for(int j=0;j<nl;j++){
                            unsigned c=selected[j][bx],m=c>=12?c-12:0;ng_causal_state*s=c>=12?model[m/12]:h;
                            for(unsigned k=0;k<3;k++)row_weights[k][j]=k<s->p->t.nrow?s->p->w[zs[j]==0][k]:0;
                            row_feature[j]=c>=12&&m%12?weights[2*m+(zs[j]==0)]:0;
                        }
                        if(edge==4)ng_bank_compact_avx2(4,compactcols+4*bx,row_weights,row_feature,prev,fullvalues[4*bx],0);
                        else ng_bank_compact_avx2(edge,compactcols+edge*bx,row_weights,row_feature,prev,fullvalues[edge*bx],0);
                    }
                }else
#ifdef NG_HAVE_WIDE_PRE
                if(fused_base)ng_bank_fused_base(VCN_LC_D,compactcols,row_weights,row_feature,prev,fullvalues[0],packed_flags,packed_raw,neighbor_ptrs[0],neighbor_ptrs[1],neighbor_ptrs[2],neighbor_stride);
                else
#endif
                ng_bank_compact_avx2(VCN_LC_D,compactcols,row_weights,row_feature,prev,fullvalues[0],homogeneous);
            }else
#endif
            {
#if VCN_HAVE_AVX2
            if(vcn_use_avx2())ng_bank_wave_avx2(VCN_LC_D,fullcols,prev,fullvalues[0]);else
#endif
            ng_bank_wave_c(VCN_LC_D,fullcols,prev,fullvalues[0]);
            }
            ng_published_wave published={0};published.lanes=nl;published.values=fullvalues[0];published.offsets=offset;published.ys=ys;published.zs=zs;published.error_offsets=ri;
            published.history=wave_history;published.history_plane=history_plane;published.history_stride=history_stride;published.history_depth=history_depth;published.delta=dl;published.publish_state=h;published.output=out;
            ng_bank_publish_wave(&published);
            continue;
        }
        /* Prepare and consume one block across the independent rows. The
         * complete lane metadata stays in L1 rather than expanding all x. */
        for(int bx=0;bx<(VCN_LC_D>>edge_shift);bx++){
            for(int j=0;j<nl;j++){
                int z=zs[j],y=ys[j];size_t o=offset[j];unsigned c=selected[j][bx];if(c>=12+12*count)goto done;
                unsigned m=c>=12?c-12:c;ng_causal_state*s=c>=12?model[m/12]:NULL;if(c>=12&&!s)goto done;
                const int32_t*w=s?s->p->w[z==0]:NULL;int nr=s?s->p->t.nrow:0;
                uint16_t pre[4]={0};if(s&&!common&&!multi_pre)vcn_lc_pre_span(h->b+ri[j]+4*bx,&s->pl[z==0],(uint16_t)s->p->bias[z==0],pre,4);
                for(int x=edge*bx;x<edge*(bx+1);x++){
                    ng_bank_wave_column*a=cols+x%4;ng_base_parts p=ng_directional_parts_xyz(out,o+x,m%12,x,y,z,VCN_LC_D,(size_t)VCN_LC_D*VCN_LC_Y);
                    a->constant[j]=p.constant;a->left[j]=p.left;a->clamp[j]=p.clamp;a->lo[j]=p.lo;a->hi[j]=p.hi;
                    a->pre[j]=s?(common?shared_pre[j*VCN_LC_D+x]:multi_pre?multi_pre[((m/12)*32+j)*VCN_LC_D+x]:(uint16_t)(pre[x%4]+s->p->cb[s->p->lut[activity[j*VCN_LC_D+x]]])):0;a->w0[j]=nr>0?w[0]:0;a->w1[j]=nr>1?w[1]:0;a->w2[j]=nr>2?w[2]:0;
                    a->feature[j]=s?(m%12?(mode?weights[2*m+(z==0)]:256):0):256;
                    a->residual[j]=out[o+x];
                }
            }

#if VCN_HAVE_AVX2
            if(vcn_use_avx2())ng_bank_wave_avx2(4,cols,prev,values);else
#endif
            ng_bank_wave_c(4,cols,prev,values);
            for(int j=0;j<nl;j++)for(int x=0;x<4;x++)h->b[ri[j]+4*bx+x]=out[offset[j]+4*bx+x]=values[x*32+j];
        }
        for(int j=0;j<nl;j++){
            vcn_lc_row_end(h->b,zs[j],ys[j]);

        }
    }

    if(entropy){if(decoder.over||decoder.p!=decoder.end)goto done;for(int k=0;k<VCN_LC_NS;k++)if(decoder.S[k]!=VCN_LC_RL)goto done;
}
    if(delta_only)ng_bank_delta_reconstruct(choices,out);
    rc=0;
done:free(multi_pre);free(history_owner);free(class_table);free(needed_rows);free(ep);free(tables);free(entropy_classes);free(entropy_symbols);free(sc.row);free(sc.n);free(activity);free(shared_pre);free(field_storage);free(transposed);free(fullcols);free(compactcols);free(fullvalues);return rc;
}
