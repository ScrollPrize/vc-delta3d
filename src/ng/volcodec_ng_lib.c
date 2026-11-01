/* volcodec_ng_lib.c: compiles volcodec_ng_lossless.h once and exports the C surface in
 * volcodec_ng_lib.h. MIT License, Copyright (c) 2026 Chase Hendrick (see LICENSE in this folder). */
#include "volcodec_ng_lib.h"
#include "residual_bridge.h"
#include <limits.h>
#include <string.h>

_Thread_local ng_geometry ng_current_geometry={NG_BLOCK_DIM,NG_BLOCK_DIM,NG_BLOCK_DIM};
_Thread_local unsigned ng_choice_edge=4;
_Thread_local int ng_runtime_only=0; /* private comparison control, scoped by benchmark */
_Thread_local int ng_disable_avx512=0;
int ng_set_isa_limit(int bits){
    if(bits!=0&&bits!=256)return -1;
    ng_disable_avx512=bits==256;return 0;
}
_Thread_local int ng_native_fast_control=0;
_Thread_local int ng_dense_pre=0; /* 0 selected pairs; 1 fixed-layout SIMD; 2 all-tap pairs; 3 fixed scalar */
_Thread_local int ng_fit_context_mode=0; /* scoped hybrid predictor experiment */
_Thread_local int ng_entropy_rounds=0; /* encoder experiment, default unchanged */
_Thread_local int ng_tap_swap_mode=0, ng_tap_fit_phase=0; /* bit1 screen, bit2 full */
_Thread_local double ng_tap_swap_stats[8]={0}; /* fits,samples,ranked,scored,moves,bits before/after,screen fits */
_Thread_local ng_tap_capture_fn ng_tap_capture=NULL;
_Thread_local ng_tap_override_fn ng_tap_override=NULL;
_Thread_local int ng_nonzero_sparse=1; /* preserve sparse budget over masked targets */
_Thread_local double ng_fit_ridge_scale=1.;
_Thread_local double ng_fit_intercept_weight=0.; /* relative to real sample mass */
_Thread_local double ng_fit_zero_weight=0.; /* synthetic all-zero observation mass */
_Thread_local double ng_entropy_stats[6]={0}; /* banks,samples,bits before/after,moves,trials */
_Thread_local int ng_soft_classes=0; /* experiment wrapper: 0 hard, 1 blend, 2 half blend */
_Thread_local unsigned ng_simple_activity_kind=0;
_Thread_local const uint16_t* ng_simple_activity_values=NULL;
/* Experimental bounded-history screen, never enabled by production encode. */
_Thread_local const uint8_t* ng_screen_work=NULL;
_Thread_local const uint8_t* ng_screen_targets=NULL;
_Thread_local size_t ng_screen_size=0;
size_t ng_fit_settings_signature(uint8_t*out,size_t cap){
    if(ng_simple_activity_values||ng_tap_capture||ng_tap_override)return 0;
    int modes[]={ng_runtime_only,ng_disable_avx512,ng_native_fast_control,ng_dense_pre,ng_fit_context_mode,ng_entropy_rounds,ng_nonzero_sparse,ng_soft_classes,(int)ng_simple_activity_kind,ng_tap_swap_mode,ng_tap_fit_phase};
    double weights[]={ng_fit_ridge_scale,ng_fit_intercept_weight,ng_fit_zero_weight};
    size_t n=sizeof(modes)+sizeof(weights);if(cap<n)return 0;
    memcpy(out,modes,sizeof(modes));memcpy(out+sizeof(modes),weights,sizeof(weights));return n;
}
int ng_geometry_valid(uint32_t x,uint32_t y,uint32_t z) {
    /* Bounds make padded pitches and signed SIMD offsets safe too. */
    return x>=32 && y>=32 && z>=32 && !(x%32) && !(y%32) && !(z%32) &&
           (uint64_t)y*z<=65536 && x<=1024;
}

static _Thread_local void (*ng_profile_callback)(int);
void ng_profile_tick(int k) {if(ng_profile_callback)ng_profile_callback(k);}
#define VCN_LC_TICK(k) do { if(ng_profile_callback)ng_profile_callback(k); } while(0)
#include "volcodec_ng_lossless.h"
/* Reuse the exact native solve/quantization with cached dense sufficient stats. */
int ng_tap_subset_solve(const double*C,const double*r,const double*mean,double mt,const int*ids,int n,
 const int*s,int m,int32_t*out,int32_t*bias,double*gain){
    if(n<1||n>VCN_LC_MAXF||m<1||m>n)return -1;
    double*A=(double*)malloc(sizeof(double)*2*m*m),b[VCN_LC_MAXF];int32_t q[VCN_LC_MAXF];
    if(!A)return -1;double*A0=A+(size_t)m*m;
    for(int a=0;a<m;a++){if(s[a]<0||s[a]>=n){free(A);return -1;}b[a]=r[s[a]];
      for(int k=0;k<m;k++)A[a*m+k]=C[s[a]*n+s[k]];}
    memcpy(A0,A,sizeof(double)*m*m);
    if(vcn_solve(A,b,m)){free(A);return -1;}
    *gain=0;for(int a=0;a<m;a++)*gain+=b[a]*r[s[a]];
    if(out){vcn_lc_quantize(A0,b,q,m);double bb=mt;
      for(int a=0;a<m;a++){out[ids[s[a]]]=q[a];bb-=q[a]/256.*mean[s[a]];}
      *bias=(uint16_t)(int32_t)floor(bb*256.+.5);}
    free(A);return 0;
}
/* Diagnostic export exercises the actual fitting sampler, not a test copy. */
int ng_nonzero_sample_mask(const uint8_t*mask,int d,int zlo,int zhi,int stride,
 uint8_t*out,uint64_t*stats){
    if(!mask||!out||!stats||d<8||d>1024||zlo<0||zhi>d||zlo>=zhi||stride<1)return -1;
    uint8_t*m=ng_fit_nonzero_sample_mask(mask,d,d,d,4,zlo,zhi,stride,0,stats);
    memset(out,0,(size_t)d*d*d);
    if(m){memcpy(out,m,(size_t)d*d*d);free(m);return 0;}
    for(int z=zlo;z<zhi;z++)for(int y=0;y<d;y++)for(int x=4+ng_fit_sample_phase(y,z,stride,0);x<d;x+=stride){
        size_t i=((size_t)z*d+y)*d+x;if(mask[i])out[i]=1;
    }
    return 0;
}
int ng_axis_taps(int* xyz) {
    vcn_lc_taps t;vcn_lc_build(&t,3,3,1);
    for(int k=0;k<t.nt;k++){xyz[3*k]=t.dx[k];xyz[3*k+1]=t.dy[k];xyz[3*k+2]=t.dz[k];}
    return t.nt;
}
int ng_tap_model_support(const uint8_t*src,size_t n,const int32_t*feature,uint8_t*out){
    const uint8_t*cur=src;vcn_lc_par*p=NULL;
    if(!src||!out||!feature||vcn_lc_readpar(&cur,n,&p)!=VCN_OK)return -1;
    if(cur!=src+n||p->t.nt>160){free(p);return -1;}
    memset(out,0,2*161);
    for(int b=0;b<2;b++){for(int k=0;k<p->t.nt;k++)out[b*161+k]=p->w[b][k]!=0;out[b*161+p->t.nt]=feature[b]!=0;}
    free(p);return 0;
}
int ng_axis_solve(double* covariance,double* rhs,int n,int32_t* q) {
    if(n<1 || n>VCN_LC_MAXT)return -1;
    double original[VCN_LC_MAXT*VCN_LC_MAXT];
    memcpy(original,covariance,sizeof(double)*n*n);
    if(vcn_solve(covariance,rhs,n))return -1;
    for(int k=0;k<n;k++)q[k]=(int32_t)floor(fmax(-32000.,fmin(32000.,rhs[k]*256.))+0.5);
    vcn_lc_qopt(original,rhs,q,n);
    return 0;
}
int ng_axis_solve_selected(double* C,double* cb,int n,int32_t* out) {
    if(n<1 || n>VCN_LC_MAXT)return -1;
    int id[VCN_LC_MAXT];double mf[VCN_LC_MAXT]={0},tr=0;
    for(int k=0;k<n;k++){id[k]=k;tr+=C[k*n+k];out[k]=0;}
    int m=vcn_lc_select(C,cb,mf,id,n,40,tr);
    if(!m)return 0;
    double original[VCN_LC_MAXT*VCN_LC_MAXT];int32_t q[VCN_LC_MAXT];
    memcpy(original,C,sizeof(double)*m*m);
    if(vcn_solve(C,cb,m))return -1;
    vcn_lc_quantize(original,cb,q,m);
    for(int k=0;k<m;k++)out[id[k]]=q[k];
    return 0;
}
/* Shared NG tap selection, solve and quantization for alternative traversal
 * orders. Inputs are centered, regularized covariance statistics. */
int ng_centered_fit(double*C,double*cb,const double*means,double mt,int n,int limit,int32_t*out,int32_t*bias){
    if(n<1||n>VCN_LC_MAXT||limit<1)return -1;
    int ids[VCN_LC_MAXT];double mf[VCN_LC_MAXT],tr=0;
    for(int k=0;k<n;k++){ids[k]=k;mf[k]=means[k];tr+=C[k*n+k];out[k]=0;}
    int m=limit<n?vcn_lc_select(C,cb,mf,ids,n,limit,tr):n;
    if(!m)return -1;
    double original[VCN_LC_MAXT*VCN_LC_MAXT];int32_t q[VCN_LC_MAXT];
    memcpy(original,C,sizeof(double)*m*m);
    if(vcn_solve(C,cb,m))return -1;
    vcn_lc_quantize(original,cb,q,m);
    double b=mt;for(int k=0;k<m;k++){out[ids[k]]=q[k];b-=q[k]/256.*mf[k];}
    *bias=(uint16_t)(int32_t)floor(b*256.+.5);return 0;
}
#define NG_DECLARE_FIXED(d) \
int ng_fixed##d##_encode_ex(const uint8_t*,const void*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*); \
int ng_fixed##d##_native_encode(const uint8_t*,uint8_t*,size_t,size_t*); \
int ng_fixed##d##_native_decode(const uint8_t*,size_t,uint8_t*,size_t); \
int ng_fixed##d##_activity_values(const uint8_t*,uint16_t*,int,uint32_t*); \
int ng_fixed##d##_decode_core(const uint8_t*,size_t,uint8_t*,ng_symbol_reader,void*,const uint8_t*,int,int,const uint8_t*,const uint8_t*);
NG_DECLARE_FIXED(32)
NG_DECLARE_FIXED(64)
NG_DECLARE_FIXED(128)
NG_DECLARE_FIXED(256)
typedef struct {
    int (*encode)(const uint8_t*,const void*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*);
    int (*decode)(const uint8_t*,size_t,uint8_t*,ng_symbol_reader,void*,const uint8_t*,int,int,const uint8_t*,const uint8_t*);
    int (*native_encode)(const uint8_t*,uint8_t*,size_t,size_t*);
    int (*native_decode)(const uint8_t*,size_t,uint8_t*,size_t);
    int (*activity_values)(const uint8_t*,uint16_t*,int,uint32_t*);
} ng_fixed_kernels;
#define NG_FIXED_ENTRY(d) {ng_fixed##d##_encode_ex,ng_fixed##d##_decode_core,ng_fixed##d##_native_encode,ng_fixed##d##_native_decode,ng_fixed##d##_activity_values}
static const ng_fixed_kernels ng_fixed_sets[]={NG_FIXED_ENTRY(64),NG_FIXED_ENTRY(128),NG_FIXED_ENTRY(256),NG_FIXED_ENTRY(32)};
static const ng_fixed_kernels* ng_fixed_current(void) {
    if(ng_runtime_only || NG_X!=NG_Y || NG_X!=NG_Z)return NULL;
    return NG_X==64?&ng_fixed_sets[0]:NG_X==128?&ng_fixed_sets[1]:NG_X==256?&ng_fixed_sets[2]:NG_X==32?&ng_fixed_sets[3]:NULL;
}
static vcn_status ng_encode_core(const uint8_t* src,const vcn_lc_cfg* cf,uint8_t* out,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl) {
    vcn_lc_cfg extended;int32_t context[16]={0};
    if(ng_fit_context_mode&&!cf->import_sym){extended=*cf;extended.context_weights=context;extended.embed_context=1;cf=&extended;}
    const ng_fixed_kernels* fixed=ng_fixed_current();
    return fixed ? fixed->encode(src,cf,out,cap,n,sy,cl) : vcn_lc_encode_ex(src,cf,out,cap,n,sy,cl);
}
static vcn_status ng_decode_core(const uint8_t* p,size_t n,uint8_t* dst,ng_symbol_reader reader,void* ctx,
                                const uint8_t* selectors,int scalar,int B,const uint8_t* sy,const uint8_t* contexts) {
    const ng_fixed_kernels* fixed=ng_fixed_current();
    return fixed ? fixed->decode(p,n,dst,reader,ctx,selectors,scalar,B,sy,contexts)
        : vcn_lc_decode_core(p,n,dst,reader,ctx,selectors,scalar,B,sy,contexts);
}
void ng_set_profile_callback(void (*callback)(int)) { ng_profile_callback=callback; }

_Static_assert(VCN_REVISION == 2, "volcodec_ng_lib.h documents format revision 2");
_Static_assert(VOLCODEC_NG_LIB_ERR_MODE == (int)VCN_ERR_MODE, "status codes match the header");
_Static_assert(VOLCODEC_NG_LIB_ERR_VERSION == (int)VCN_ERR_VERSION, "status codes match the header");

unsigned volcodec_ng_lib_format_revision(void) { return VCN_REVISION; }

size_t volcodec_ng_lib_encode_bound(void) { return VCN_ENCODE_BOUND; }

const char *volcodec_ng_lib_status_string(int status) { return vcn_status_string((vcn_status)status); }

const char *volcodec_ng_lib_kernels(void) { return vcn_kernels(); }

int volcodec_ng_lib_encode(const uint8_t *src_zyx, void *dst, size_t dst_cap, size_t *out_n) {
    const ng_fixed_kernels* fixed=ng_fixed_current();
    if(fixed)return fixed->native_encode(src_zyx,dst,dst_cap,out_n);
    return (int)vcn_encode_lossless_chunk(src_zyx, (uint8_t *)dst, dst_cap, out_n);
}

int volcodec_ng_lib_decode(const void *enc, size_t enc_n, uint8_t *dst_zyx, size_t dst_cap) {
    const ng_fixed_kernels* fixed=ng_fixed_current();
    if(fixed)return fixed->native_decode(enc,enc_n,dst_zyx,dst_cap);
    return (int)vcn_decode((const uint8_t *)enc, enc_n, dst_zyx, dst_cap);
}
int ng_original_fast_decode_trial(const void*enc,size_t n,uint8_t*out,size_t cap,unsigned d,int trial){
    if(!ng_geometry_valid(d,d,d))return -1;
    ng_geometry saved_geometry=ng_current_geometry;int saved=ng_native_fast_control;
    ng_current_geometry=(ng_geometry){d,d,d};ng_native_fast_control=trial;
    int result=volcodec_ng_lib_decode(enc,n,out,cap);
    ng_current_geometry=saved_geometry;ng_native_fast_control=saved;return result;
}
int ng_original_fast_decode(const void*enc,size_t n,uint8_t*out,size_t cap,unsigned d){return ng_original_fast_decode_trial(enc,n,out,cap,d,2);}

int volcodec_ng_lib_is_chunk(const void *enc, size_t enc_n) {
    return vcn_chunk_mode((const uint8_t *)enc, enc_n) >= 0;
}

int volcodec_ng_lib_chunk_mode(const void *enc, size_t enc_n) {
    return vcn_chunk_mode((const uint8_t *)enc, enc_n);
}

int ng_prepare_residuals(const uint8_t* src, uint8_t* params, size_t cap, size_t* n,
                         uint8_t* symbols, uint8_t* classes) {
    const vcn_lc_cfg cf = {3, 3, 1, 16, VCN_LC_TB, VCN_LC_NSEL, VCN_LC_STRIDE};
    return ng_encode_core(src, &cf, params, cap, n, symbols, classes);
}
static int ng_config(const int* a,vcn_lc_cfg* cf) {
    if (!a || a[0]<1 || a[0]>3 || a[1]<-1 || a[1]>3 || a[2]<-1 || a[2]>3 ||
        a[3]<2 || a[3]>32 || a[4]<2 || a[4]>160 || a[5]<1 || a[5]>128) return -1;
    *cf=(vcn_lc_cfg){a[0],a[1],a[2],a[3],VCN_LC_TB,a[4],a[5]};return 0;
}
int ng_prepare_config(const uint8_t* src,const int* a,uint8_t* params,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl) {
    vcn_lc_cfg cf;
    if(ng_config(a,&cf) || !src || !params || !n || !sy || !cl) return -1;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}
int ng_encode_config(const uint8_t* src,const int* a,uint8_t* dst,size_t cap,size_t* n) {
    vcn_lc_cfg cf;
    if(ng_config(a,&cf)) return -1;
    return vcn_encode_lossless_chunk_config(src,dst,cap,n,&cf);
}
int ng_reconstruct_residuals(const uint8_t* params, size_t n, ng_symbol_reader reader,
                              void* context, uint8_t* dst) {
    return vcn_lc_decode_ex(params,n,dst,reader,context);
}

/* Experiment target: 0 ordinary, 1 signed delta residual, 2 unsigned delta
   residual, 3 zero correction, 4 single fitted bank/no class bias.
   All fitting/prediction kernels remain shared. */
int ng_ex_prepare_activity(const uint8_t* src,const int* config,int target,unsigned mask,
    uint8_t* params,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl,uint16_t* predictions,uint16_t* activity) {
    vcn_lc_cfg cf;
    if(ng_config(config,&cf) || target<0 || target>4 || mask>7)return -1;
    cf.single_model=target==4;
    const size_t N=NG_N;
    uint8_t* base=NULL;int16_t* values=NULL;
    if(target && target!=4) {
        base=malloc(N);values=malloc(N*sizeof(*values));
        if(!base || !values){free(base);free(values);return -1;}
        for(size_t i=0;i<N;i++) {
            base[i]=delta_predict_at(src,i,mask);
            uint8_t r=(uint8_t)(src[i]-base[i]);
            values[i]=target==1?(int16_t)(r<128?r:(int)r-256):r;
        }
        cf.base_prediction=base;cf.fit_target=values;cf.zero_correction=target==3;
    }
    cf.export_prediction=predictions;
    cf.export_activity=activity;
    int rc=ng_encode_core(src,&cf,params,cap,n,sy,cl);
    free(base);free(values);return rc;
}
int ng_ex_prepare(const uint8_t* src,const int* config,int target,unsigned mask,
    uint8_t* params,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl,uint16_t* predictions) {
    return ng_ex_prepare_activity(src,config,target,mask,params,cap,n,sy,cl,predictions,NULL);
}
/* Native NG correction fitted against a caller-supplied causal base predictor.
 * Neighbor features remain original decoded source bytes, as in ng_ex_prepare. */
int ng_base_prepare(const uint8_t* src,const int* config,const uint8_t* base,
    uint8_t* params,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl,uint16_t* predictions) {
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src||!base)return -1;
    int16_t* target=malloc(NG_N*sizeof(*target));if(!target)return -1;
    for(size_t i=0;i<NG_N;i++){unsigned r=(uint8_t)(src[i]-base[i]);target[i]=r<128?r:(int)r-256;}
    cf.fit_target=target;cf.base_prediction=base;cf.export_prediction=predictions;
    int rc=ng_encode_core(src,&cf,params,cap,n,sy,cl);free(target);return rc;
}
int ng_feature_prepare_masked(const uint8_t* src,const int* config,const uint8_t* feature,const uint8_t* mask,int mode,int32_t* weights,
    uint8_t* params,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl,uint16_t* predictions){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src||!feature||!weights||(mode!=1&&mode!=3))return -1;
    cf.fit_feature=feature;cf.fit_mask=mask;cf.feature_mode=mode;cf.export_feature_weights=weights;cf.export_prediction=predictions;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}
int ng_plain_prepare_masked(const uint8_t*src,const int*config,const uint8_t*mask,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint8_t*cl,uint16_t*predictions){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src)return -1;
    cf.fit_mask=mask;cf.export_prediction=predictions;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}
int ng_context_prepare(const uint8_t*src,const int*config,const uint8_t*mask,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint8_t*cl,uint16_t*predictions){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src||!weights)return -1;
    cf.fit_mask=mask;cf.context_weights=weights;cf.export_prediction=predictions;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}
int ng_prepare_weighted(const uint8_t*src,const int*config,const uint8_t*feature,const uint8_t*mask,const uint8_t*sample_weights,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint8_t*cl,uint16_t*predictions){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src||!sample_weights||(feature&&!weights))return -1;
    cf.fit_mask=mask;cf.fit_weights=sample_weights;cf.export_prediction=predictions;
    if(feature){cf.fit_feature=feature;cf.feature_mode=1;cf.export_feature_weights=weights;}
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}
/* New==NULL: refit on old nonzero tap support. Otherwise materialize the
   old + (new-old)/2^shift model without fitting again. */
int ng_prepare_refinement(const uint8_t*src,const int*config,const uint8_t*feature,const uint8_t*mask,const uint8_t*sample_weights,
    const uint8_t*old,size_t old_n,const int32_t*old_weights,const uint8_t*next,size_t next_n,const int32_t*next_weights,unsigned shift,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint8_t*cl,uint16_t*predictions){
    vcn_lc_cfg cf;vcn_lc_par*a=NULL,*b=NULL;const uint8_t*cursor=old;
    if(ng_config(config,&cf)||!src||!old||old_n<6||shift>3||(feature&&(!old_weights||!weights)))return -1;
    if(vcn_lc_readpar(&cursor,old_n,&a)!=VCN_OK)return -1;
    if(cursor!=old+old_n||a->t.r[0]!=cf.r0||a->t.r[1]!=cf.r1||a->t.r[2]!=cf.r2){free(a);return -1;}
    cf.fit_mask=mask;cf.fit_weights=sample_weights;cf.export_prediction=predictions;
    if(feature){cf.fit_feature=feature;cf.feature_mode=1;cf.export_feature_weights=weights;}
    int32_t fw[2]={0,0};
    if(next){
      cursor=next;
      if((feature&&!next_weights)||vcn_lc_readpar(&cursor,next_n,&b)!=VCN_OK){free(a);return -1;}
      if(cursor!=next+next_n||a->t.nt!=b->t.nt||memcmp(a->t.r,b->t.r,sizeof(a->t.r))){free(a);free(b);return -1;}
      for(int bank=0;bank<2;bank++){
        for(int k=0;k<a->t.nt;k++)b->w[bank][k]=a->w[bank][k]+(int32_t)(((int64_t)b->w[bank][k]-a->w[bank][k])/(1u<<shift));
        uint16_t d=(uint16_t)(b->bias[bank]-a->bias[bank]);int32_t sd=d<32768?d:(int32_t)d-65536;
        b->bias[bank]=(uint16_t)(a->bias[bank]+sd/(int32_t)(1u<<shift));
        if(feature)fw[bank]=old_weights[bank]+(int32_t)(((int64_t)next_weights[bank]-old_weights[bank])/(1u<<shift));
      }
      cf.fixed_fit=b;cf.fixed_feature_weights=fw;
    }else cf.fit_support=a;
    int rc=ng_encode_core(src,&cf,params,cap,n,sy,cl);free(a);free(b);return rc;
}
int ng_prepare_damped(const uint8_t*src,const int*config,const uint8_t*feature,const uint8_t*mask,
    const uint8_t*old,size_t old_n,const int32_t*old_weights,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint8_t*cl,uint16_t*predictions){
    vcn_lc_cfg cf;const uint8_t*cursor=old;vcn_lc_par*p=NULL;
    if(ng_config(config,&cf)||!src||!old||old_n<6||(feature&&(!old_weights||!weights)))return -1;
    if(vcn_lc_readpar(&cursor,old_n,&p)!=VCN_OK)return -1;
    if(cursor!=old+old_n||old[1]!=cf.r0||old[2]!=cf.r1||old[3]!=cf.r2){free(p);return -1;}
    cf.fit_mask=mask;cf.export_prediction=predictions;cf.damp_previous=p;cf.damp_feature_weights=old_weights;
    if(feature){cf.fit_feature=feature;cf.feature_mode=1;cf.export_feature_weights=weights;}
    int rc=ng_encode_core(src,&cf,params,cap,n,sy,cl);free(p);return rc;
}
int ng_feature_prepare_activity(const uint8_t*src,const int*config,const uint8_t*feature,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint16_t*activity,uint32_t*histogram){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src||!feature||!weights||!activity||!histogram)return -1;
    cf.fit_feature=feature;cf.feature_mode=1;cf.export_feature_weights=weights;
    cf.export_activity=activity;cf.export_activity_histogram=histogram;
    return ng_encode_core(src,&cf,params,cap,n,sy,NULL);
}
int ng_feature_prepare(const uint8_t* src,const int* config,const uint8_t* feature,int mode,int32_t* weights,
    uint8_t* params,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl,uint16_t* predictions){
    return ng_feature_prepare_masked(src,config,feature,NULL,mode,weights,params,cap,n,sy,cl,predictions);
}
void* ng_fit_session_create(int simd){
    ng_fit_session*s=calloc(1,sizeof(*s));if(s)s->simd=simd!=0;return s;
}
void ng_fit_session_destroy(void*ptr){
    ng_fit_session*s=ptr;if(!s)return;
    for(int i=0;i<2;i++)ng_fit_bank_clear(s->banks+i);
    free(s->padded);free(s);
}
int ng_feature_prepare_cached(void*ptr,const uint8_t*src,const int*config,const uint8_t*feature,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint8_t*cl,uint16_t*predictions){
    ng_fit_session*s=ptr;vcn_lc_cfg cf;
    if(!s||ng_config(config,&cf)||!src||!feature||!weights)return -1;
    const int dims[3]={NG_X,NG_Y,NG_Z};
    if(s->source&&(s->source!=src||memcmp(s->config,config,sizeof(s->config))||memcmp(s->dims,dims,sizeof(dims))))return -1;
    s->source=src;memcpy(s->config,config,sizeof(s->config));memcpy(s->dims,dims,sizeof(dims));
    cf.fit_feature=feature;cf.feature_mode=1;cf.export_feature_weights=weights;cf.export_prediction=predictions;cf.fit_session=s;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}
/* Fit a shared correction against a spatially varying provisional Delta base.
   All regression/class fitting is the original NG implementation. */
int ng_mixed_prepare(const uint8_t* src,const int* config,const uint8_t* masks,
    uint8_t* params,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl) {
    vcn_lc_cfg cf;if(ng_config(config,&cf))return -1;
    size_t count=(size_t)NG_N;
    uint8_t* base=malloc(count);int16_t* target=malloc(count*sizeof(*target));
    if(!base || !target){free(base);free(target);return -1;}
    for(size_t i=0;i<count;i++) {
        if(masks[i]>7){free(base);free(target);return -1;}
        base[i]=masks[i]?delta_predict_at(src,i,masks[i]):0;
        uint8_t r=(uint8_t)(src[i]-base[i]);
        target[i]=masks[i]?(r<128?r:(int)r-256):src[i];
    }
    cf.fit_target=target;cf.base_prediction=base;
    int rc=ng_encode_core(src,&cf,params,cap,n,sy,cl);free(base);free(target);return rc;
}

/* Decoder for experimental integrated predictors. Classes use reconstructed
   residuals of the shared NG model, so class-table selection needs no voxel
   selectors and does not depend on the unknown current voxel. */
static int ng_predictor_pass(const uint8_t* params,size_t n,unsigned delta_mask,
    const uint8_t* selectors,int unsigned_symbols,ng_symbol_reader reader,void* ctx,uint8_t* dst,
    int selected_context,const uint8_t* src,uint8_t* symbols,uint8_t* class_ids,uint16_t* activities,
    const uint8_t* const* extra_models,const size_t* model_sizes,int model_count,
    const uint8_t* block_choices,int fixed_choice,int force_scalar) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    int joint_mode=model_count>0;
    vcn_lc_par* models[8]={p};vcn_lc_plist other_pl[8][2];
    for(int m=1;m<model_count;m++) {
        const uint8_t* cp=extra_models[m];
        if(vcn_lc_readpar(&cp,model_sizes[m],&models[m])!=VCN_OK || cp!=extra_models[m]+model_sizes[m]) {
            for(int j=0;j<=m;j++)free(models[j]);return -1;
        }
        vcn_lc_mkplist(&models[m]->t,models[m]->w[0],&other_pl[m][0]);
        vcn_lc_mkplist(&models[m]->t,models[m]->w[1],&other_pl[m][1]);
    }
    uint8_t* b=malloc(VCN_LC_NB);uint8_t* e=malloc(VCN_LC_NB);
    if(!b||!e){free(b);free(e);for(int m=1;m<model_count;m++)free(models[m]);free(p);return -1;}
    memset(b,128,VCN_LC_NB);memset(e,0,VCN_LC_NB);
    vcn_lc_plist pl[2];
    vcn_lc_mkplist(&p->t,p->w[0],&pl[0]);vcn_lc_mkplist(&p->t,p->w[1],&pl[1]);
    uint16_t pre[VCN_LC_D];uint8_t classes[VCN_LC_D];int rc=0;
    uint16_t other_pre[8][VCN_LC_D];uint8_t other_classes[8][VCN_LC_D];
    for(int z=0;z<VCN_LC_Z && !rc;z++) for(int y=0;y<VCN_LC_Y && !rc;y++) {
        int s=z==0;size_t ri=VCN_LC_IDX(z,y,0),o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        vcn_lc_row_begin(b,z,y,128);vcn_lc_row_begin(e,z,y,0);
        if(selected_context) {
            uint16_t activity[VCN_LC_D];vcn_lc_apre(e,ri,activity);
            if(activities)memcpy(activities+o,activity,sizeof(activity));
            int needs_ng=0;
            for(int x=0;x<VCN_LC_D;x++) {classes[x]=p->lut[activity[x]];needs_ng|=selectors[classes[x]]==0;}
            if(needs_ng)VCN_LC_PRE(b+ri,&pl[s],(uint16_t)p->bias[s],pre);
            else memset(pre,0,sizeof(pre));
            for(int x=0;x<VCN_LC_D;x++)pre[x]=(uint16_t)(pre[x]+p->cb[classes[x]]);
        } else {
            if(force_scalar){vcn_lc_pre_c(b+ri,&pl[s],(uint16_t)p->bias[s],pre);vcn_lc_cls_c(e,ri,p,pre,classes);}
            else {VCN_LC_PRE(b+ri,&pl[s],(uint16_t)p->bias[s],pre);VCN_LC_CLS(e,ri,p,pre,classes);}
        }
        for(int m=1;m<model_count;m++) {
            if(fixed_choice>=0 && (fixed_choice<8 || m!=(model_count==8?fixed_choice-7:1)))continue;
            if(force_scalar){vcn_lc_pre_c(b+ri,&other_pl[m][s],(uint16_t)models[m]->bias[s],other_pre[m]);vcn_lc_cls_c(e,ri,models[m],other_pre[m],other_classes[m]);}
            else {VCN_LC_PRE(b+ri,&other_pl[m][s],(uint16_t)models[m]->bias[s],other_pre[m]);VCN_LC_CLS(e,ri,models[m],other_pre[m],other_classes[m]);}
        }
        for(int x=0;x<VCN_LC_D;x++) {
            unsigned c=classes[x],sel=selectors?selectors[c]:0;
            unsigned choice=0,mi=0,dm=delta_mask;
            if(joint_mode) {
                unsigned nb=VCN_LC_D/4,ny=VCN_LC_Y/4;
                choice=fixed_choice>=0?(unsigned)fixed_choice:block_choices[((z/4)*ny+y/4)*nb+x/4];
                if(choice>14){rc=-1;break;}
                sel=choice>=1 && choice<=7?choice:0;
                if(choice>=8){dm=choice-7;mi=model_count==8?dm:model_count==2?1:0;}
                c=mi?other_classes[mi][x]:classes[x];
            }
            vcn_lc_par* active=models[mi];uint16_t q=mi?other_pre[mi][x]:pre[x];
            if(!selected_context || !sel)
                for(int k=0;k<active->t.nrow;k++) q=(uint16_t)(q+(uint16_t)active->w[s][k]*b[ri+x-1-k]);
            if(dm)q=(uint16_t)(q+256u*delta_predict_at(dst,o+x,dm));
            unsigned pred=sel?delta_predict_at(dst,o+x,sel):VCN_LC_PP(q);
            int symbol;
            if(src) {
                int r=(int8_t)(uint8_t)(src[o+x]-pred);
                if(unsigned_symbols==0 && !sel && VCN_LC_FM(q))r=(int8_t)(uint8_t)(-r);
                symbol=unsigned_symbols==1?(uint8_t)r:(r<0?-2*r-1:2*r);
                symbols[o+x]=(uint8_t)symbol;class_ids[o+x]=(uint8_t)c;
            } else symbol=reader(ctx,o+x,c);
            if(symbol<0||symbol>255){rc=-1;break;}
            int r=unsigned_symbols==1?symbol:((symbol&1)?-(symbol+1)/2:symbol/2);
            if(unsigned_symbols==0 && !sel && VCN_LC_FM(q))r=-r;
            uint8_t v=(uint8_t)(pred+r);dst[o+x]=b[ri+x]=v;
            int er=(int8_t)(uint8_t)(v-((selected_context || joint_mode)?pred:VCN_LC_PP(q)));
            e[ri+x]=(uint8_t)(er<0?-er:er);
        }
        vcn_lc_row_end(b,z,y);vcn_lc_row_end(e,z,y);
    }
    for(int m=1;m<model_count;m++)free(models[m]);free(p);free(b);free(e);return rc;
}

int ng_ex_reconstruct(const uint8_t* params,size_t n,unsigned delta_mask,const uint8_t* selectors,
    int mapping,ng_symbol_reader reader,void* ctx,uint8_t* dst) {
    return ng_predictor_pass(params,n,delta_mask,selectors,mapping,reader,ctx,dst,0,NULL,NULL,NULL,NULL,NULL,NULL,0,NULL,-1,0);
}
/* One shared causal traversal for encoding and decoding selected residuals. */
int ng_selected_pass(const uint8_t* params,size_t n,const uint8_t* selectors,int mapping,
    ng_symbol_reader reader,void* ctx,uint8_t* dst,const uint8_t* src,uint8_t* symbols,uint8_t* classes) {
    return ng_predictor_pass(params,n,0,selectors,mapping,reader,ctx,dst,1,src,symbols,classes,NULL,NULL,NULL,0,NULL,-1,0);
}

int ng_selected_observe(const uint8_t* params,size_t n,const uint8_t* selectors,const uint8_t* src,
    uint8_t* sy,uint8_t* cl,uint16_t* activities) {
    uint8_t* dst=malloc((size_t)NG_N);
    if(!dst)return -1;
    int rc=ng_predictor_pass(params,n,0,selectors,0,NULL,NULL,dst,1,src,sy,cl,activities,NULL,NULL,0,NULL,-1,0);
    free(dst);return rc;
}
int ng_joint_pass(const uint8_t* const* models,const size_t* sizes,int count,const uint8_t* choices,int fixed,
    const uint8_t* src,uint8_t* sy,uint8_t* cl,uint8_t* dst,ng_symbol_reader reader,void* ctx,int scalar) {
    if((count!=1 && count!=2 && count!=8) || fixed>14)return -1;
    return ng_predictor_pass(models[0],sizes[0],0,NULL,0,reader,ctx,dst,0,src,sy,cl,NULL,models,sizes,count,choices,fixed,scalar);
}

int ng_set_classes(uint8_t* params,size_t n,const uint16_t* thresholds,const int16_t* biases) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    size_t offset=n-2*p->K-2*(p->K-1);
    for(int k=0;k<p->K-1;k++)vcn_wr16(params+offset+2*k,thresholds[k]);
    if(biases)for(int k=0;k<p->K;k++)vcn_wr16(params+n-2*p->K+2*k,(uint16_t)biases[k]);
    free(p);return 0;
}

double ng_hist_cost_logs(const uint32_t* counts,int tb,const double* logs) {
    uint16_t freq[256];vcn_lc_norm(counts,freq,tb);
    unsigned nn=256;while(nn>1 && !freq[nn-1])nn--;
    double bits=16;
    for(unsigned s=0;s<256;s++) {
        if(s<nn){unsigned f=freq[s];do{bits+=8;f>>=7;}while(f);}
        if(counts[s])bits+=(double)counts[s]*(tb-(logs?logs[freq[s]]:log2((double)freq[s])));
    }
    return bits;
}
double ng_hist_cost(const uint32_t* counts,int tb) {return ng_hist_cost_logs(counts,tb,NULL);}

static int ng_compact_encode_order(const uint8_t* params,size_t n,const uint8_t* sy,const uint8_t* cl,
    uint8_t* out,size_t cap,size_t* size,int raster,const uint16_t* activity,const uint8_t*targets) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    vcn_lc_cfg cf={0};cf.r0=p->t.r[0];cf.r1=p->t.r[1];cf.r2=p->t.r[2];cf.K=p->K;cf.tb=p->tb;
    cf.import_sym=sy;cf.import_cls=cl;cf.import_params=params;cf.import_params_n=n;cf.raster_entropy=raster;
    cf.import_activity=activity;cf.import_class_lut=p->lut;
    cf.import_targets=targets;
    int rc=ng_encode_core(NULL,&cf,out,cap,size,NULL,NULL);free(p);return rc;
}
int ng_compact_encode(const uint8_t* p,size_t n,const uint8_t* sy,const uint8_t* cl,uint8_t* out,size_t cap,size_t* size) {
    return ng_compact_encode_order(p,n,sy,cl,out,cap,size,1,NULL,NULL);
}
int ng_wave_encode(const uint8_t* p,size_t n,const uint8_t* sy,const uint8_t* cl,uint8_t* out,size_t cap,size_t* size) {
    return ng_compact_encode_order(p,n,sy,cl,out,cap,size,0,NULL,NULL);
}
/* Experimental full-data score: exact normalized tables/header, ideal symbol
 * bits plus native state storage. No payload encoding or wavefront traversal. */
int ng_wave_estimate(const uint8_t*params,size_t n,const uint8_t*sy,const uint8_t*cl,uint8_t*out,size_t cap,size_t*hn,double*bytes){
    const uint8_t*cursor=params;vcn_lc_par*p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    unsigned K=p->K,tb=p->tb;free(p);
    if(cursor!=params+n||cap<n+(size_t)K*514)return -1;
    uint32_t(*cnt)[256]=calloc(K,sizeof(*cnt));if(!cnt)return -1;
    for(size_t i=0;i<(size_t)NG_N;i++){
        if(cl[i]>=K){free(cnt);return -1;}cnt[cl[i]][sy[i]]++;
    }
    memcpy(out,params,n);size_t pos=n;double bits=0;
    for(unsigned k=0;k<K;k++){
        uint16_t fr[256];vcn_lc_norm(cnt[k],fr,tb);
        pos+=vcn_lc_write_table(out+pos,fr);
        for(unsigned s=0;s<256;s++)if(cnt[k][s])bits+=(double)cnt[k][s]*(tb-log2((double)fr[s]));
    }
    free(cnt);*hn=pos;*bytes=4*VCN_LC_NS+bits/8.;return 0;
}
int ng_wave_encode_targets(const uint8_t*p,size_t n,const uint8_t*sy,const uint8_t*cl,const uint8_t*targets,uint8_t*out,size_t cap,size_t*size){
    return ng_compact_encode_order(p,n,sy,cl,out,cap,size,0,NULL,targets);
}
unsigned ng_rans_state_bytes(void){return 4*VCN_LC_NS;}
int ng_wave_verify_targets(const uint8_t*data,size_t n,const uint8_t*sy,const uint8_t*cl,const uint8_t*targets){
    const uint8_t*end=data+n;vcn_lc_par*p=NULL;
    if(vcn_lc_readpar(&data,n,&p)!=VCN_OK)return -1;
    unsigned K=p->K,tb=p->tb,M=1u<<tb;vcn_lc_sched sc={0,NULL,NULL};
    if(vcn_lc_mksched(vcn_lc_delta(&p->t),&sc)){free(p);return -1;}free(p);
    uint32_t*tables=malloc((size_t)K*M*sizeof(*tables));int bad=1;
    if(!tables)goto done;
    if(vcn_lc_read_tables(&data,end,tables,K,M)||end-data<4*VCN_LC_NS)goto done;
    vcn_lc_dec dec={0};dec.T=tables;dec.tb=tb;dec.end=end;
    for(int k=0;k<VCN_LC_NS;k++)dec.S[k]=vcn_rd32(data+4*k);
    dec.p=data+4*VCN_LC_NS;bad=0;
    for(int g=0;g<sc.ng;g++)for(int j=0;j<sc.n[g];j++){
        size_t o=(size_t)sc.row[g][j]*VCN_LC_D;
        for(int x=0;x<VCN_LC_D;x++)if(!targets||targets[o+x]){
            if(cl[o+x]>=K){bad=1;goto done;}
            if(vcn_lc_sym_one(&dec,cl[o+x],x&(VCN_LC_NS-1))!=sy[o+x])bad=1;
        }
    }
    bad|=dec.over||dec.p!=end;for(int k=0;k<VCN_LC_NS;k++)bad|=dec.S[k]!=VCN_LC_RL;
done:free(tables);free(sc.row);free(sc.n);return bad?-1:0;
}
int ng_wave_encode_activity(const uint8_t*p,size_t n,const uint8_t*sy,const uint16_t*a,uint8_t*out,size_t cap,size_t*size){
    if(!a)return -1;
    return ng_compact_encode_order(p,n,sy,NULL,out,cap,size,0,a,NULL);
}
/* Experimental fixed-context raster entropy: reuse the native tables, states
   and SIMD rANS, without constructing causal residual activity. */
int ng_parameter_size(const uint8_t* data,size_t n,size_t* size) {
    const uint8_t* cursor=data;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    *size=(size_t)(cursor-data);free(p);return 0;
}
int ng_bank_empty_model(uint8_t* out,size_t cap,size_t* size){
    vcn_lc_taps t;vcn_lc_build(&t,1,0,0);
    size_t need=6+4*(t.nt+1)+2*15+2*16;if(cap<need)return -1;
    memset(out,0,need);out[0]=VCN_LC_VERSION;out[1]=1;out[4]=16;out[5]=12;*size=need;return 0;
}
#include "ng_coarse_inputs.h"
/* Experiment-only: changes the appended-column fit, not the coarse Fast codec. */
static _Thread_local int ng_coarse_dense=0;
int ng_coarse_set_dense(int enabled){if(enabled!=0&&enabled!=1)return -1;ng_coarse_dense=enabled;return 0;}
/* Append input columns, but call the existing production fitter unchanged.
 * Extra planes use constant offsets solely during feature gathering; they
 * never enter the serialized native spatial-tap geometry. */
static int ng_native_columns_fit(const uint8_t*src,const uint8_t*planes,unsigned extra,int prefix,uint8_t*out,size_t cap,size_t*n,int32_t*fw){
    if(!src||!out||!n||!fw||(extra!=0&&extra!=8&&extra!=27&&extra!=54)||cap<4096)return -1;
    vcn_lc_cfg cf={3,3,1,16,VCN_LC_TB,40,ng_coarse_dense?1:128};vcn_lc_par p={0};p.K=16;p.tb=VCN_LC_TB;
    vcn_lc_build(&p.t,3,3,1);vcn_lc_taps extended=p.t;int original=p.t.nt;
    if(original+extra>VCN_LC_MAXT)return -1;
    uint8_t*b=malloc(VCN_LC_NB*(extra+1)),*mask=prefix?malloc(NG_N):NULL;
    if(!b||(prefix&&!mask)){free(b);free(mask);return -1;}
    memset(b,128,VCN_LC_NB*(extra+1));
    for(int z=0;z<NG_Z;z++)for(int y=0;y<NG_Y;y++){
        vcn_lc_row_begin(b,z,y,128);memcpy(b+VCN_LC_IDX(z,y,0),src+((size_t)z*NG_Y+y)*NG_X,NG_X);vcn_lc_row_end(b,z,y);
        for(int x=0;x<NG_X;x++){
            size_t i=((size_t)z*NG_Y+y)*NG_X+x;if(mask)mask[i]=!ng_coarse_site(x,y,z);
            for(unsigned k=0;k<extra;k++)b[(k+1)*VCN_LC_NB+VCN_LC_IDX(z,y,x)]=planes?planes[k*NG_N+i]:ng_coarse_input(src,NG_X,x,y,z,k,extra);
        }
    }
    for(unsigned k=0;k<extra;k++){int j=extended.nt++;extended.dx[j]=extended.dy[j]=extended.dz[j]=0;extended.off[j]=(k+1)*VCN_LC_NB;}
    for(int bank=0;bank<2;bank++){
        int32_t w[VCN_LC_MAXF]={0};
        vcn_lc_fit_features_supported(b,&extended,bank?0:1,bank?1:NG_Z,ng_coarse_dense?1:bank?2:128,bank,bank?20:40,w,&p.bias[bank],NULL,NULL,0,0,mask,NULL,NULL,NULL);
        memcpy(p.w[bank],w,original*sizeof(int32_t));for(unsigned k=0;k<extra;k++)fw[bank*extra+k]=w[original+k];
    }
    for(int k=0;k<15;k++)p.thr[k]=(k+1)*160;
    *n=vcn_lc_writepar(&p,&cf,out);
    /* Compare both controls against the actual public native fitting path. */
    if(!extra){
        int cfg[6]={3,3,1,16,40,ng_coarse_dense?1:128};size_t cn=0,capacity=3*NG_N+131072;
        uint8_t*control=malloc(capacity),*sy=malloc(NG_N),*cl=malloc(NG_N);
        int rc=(!control||!sy||!cl)?-1:ng_plain_prepare_masked(src,cfg,mask,control,capacity,&cn,sy,cl,NULL);
        if(!rc&&memcmp(out,control,6+4*(original+1)))rc=-1;
        free(control);free(sy);free(cl);if(rc){free(b);free(mask);return -1;}
    }
    free(b);free(mask);return 0;
}
int ng_coarse_native_fit(const uint8_t*src,unsigned extra,int prefix,uint8_t*out,size_t cap,size_t*n,int32_t*fw){return ng_native_columns_fit(src,NULL,extra,prefix,out,cap,n,fw);}
int ng_external_native_fit(const uint8_t*src,const uint8_t*planes,unsigned extra,uint8_t*out,size_t cap,size_t*n,int32_t*fw){return ng_native_columns_fit(src,planes,extra,0,out,cap,n,fw);}
int ng_entropy_header_size(const uint8_t* data,size_t n,size_t* size){
    const uint8_t* p=data;vcn_lc_par* par=NULL;if(vcn_lc_readpar(&p,n,&par)!=VCN_OK)return -1;
    int rc=vcn_lc_read_tables(&p,data+n,NULL,par->K,1u<<par->tb);
    free(par);if(rc)return -1;*size=p-data;return 0;
}
int ng_native_frame_sizes(const uint8_t* data,size_t n,size_t* header,size_t* params) {
    int mode=vcn_chunk_mode(data,n);
    if(mode<0 || n<VCN_HDR)return -1;
    *header=VCN_HDR;*params=0;
    return mode==VCN_MODE_LLCTX?ng_parameter_size(data+VCN_HDR,n-VCN_HDR,params):0;
}
int ng_known_classes_decode(const uint8_t* data,size_t n,const uint8_t* classes,uint8_t* sy,int scalar) {
    const uint8_t* end=data+n;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&data,n,&p)!=VCN_OK)return -1;
    unsigned K=p->K,tb=p->tb,M=1u<<tb;free(p);
    size_t count=(size_t)NG_N;
    // Private NDF1 API: classes were expanded from validated seven-bit tags.
    if(K!=16)return -1;
    uint32_t* tables=malloc((size_t)K*M*sizeof(*tables));if(!tables)return -1;
    if(vcn_lc_read_tables(&data,end,tables,K,M) || end-data<4*VCN_LC_NS){free(tables);return -1;}
    vcn_lc_dec d={0};d.T=tables;d.tb=tb;d.end=end;
    for(int k=0;k<VCN_LC_NS;k++)d.S[k]=vcn_rd32(data+4*k);
    d.p=data+4*VCN_LC_NS;
#if VCN_HAVE_AVX2
    if(!scalar && vcn_use_avx2())vcn_lc_syms_avx2((int)count,classes,&d,sy);
    else
#elif VCN_HAVE_NEON
    if(!scalar)vcn_lc_syms_neon((int)count,classes,&d,sy);
    else
#endif
    vcn_lc_syms_c((int)count,classes,&d,sy);
    int bad=d.over || d.p!=end;
    for(int k=0;k<VCN_LC_NS;k++)bad|=d.S[k]!=VCN_LC_RL;
    free(tables);return bad?-1:0;
}

#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static __m128i ng_load4_u16(const uint8_t* p) {
    uint32_t v;memcpy(&v,p,4);return _mm_cvtepu8_epi16(_mm_cvtsi32_si128((int)v));
}
__attribute__((target("avx2"))) static void ng_delta4(uint8_t* row,unsigned x,unsigned m,const uint8_t* sy) {
    __m128i s=ng_load4_u16(sy),zero=_mm_setzero_si128();
    __m128i r=_mm_xor_si128(_mm_srli_epi16(s,1),_mm_sub_epi16(zero,_mm_and_si128(s,_mm_set1_epi16(1))));
    const uint8_t* p=row+x;__m128i base=zero;
    if(m&2)base=_mm_add_epi16(base,ng_load4_u16(p-VCN_LC_RP));
    if(m&4)base=_mm_add_epi16(base,ng_load4_u16(p-VCN_LC_SP));
    if((m&6)==6)base=_mm_sub_epi16(base,ng_load4_u16(p-VCN_LC_RP-VCN_LC_SP));
    if(m&1) {
        int carry=0;
        if(x)carry=p[-1]-((m&2)?p[-VCN_LC_RP-1]:0)-((m&4)?p[-VCN_LC_SP-1]:0)+((m&6)==6?p[-VCN_LC_RP-VCN_LC_SP-1]:0);
        r=_mm_add_epi16(r,_mm_slli_si128(r,2));r=_mm_add_epi16(r,_mm_slli_si128(r,4));
        r=_mm_add_epi16(r,_mm_set1_epi16((short)carry));
    }
    __m128i v=_mm_and_si128(_mm_add_epi16(base,r),_mm_set1_epi16(255));
    uint32_t out=(uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(v,zero));memcpy(row+x,&out,4);
}
#endif
int ng_fast_wave_reconstruct(const uint8_t* params,size_t n,const uint8_t* choices,const uint8_t* sy,uint8_t* dst,int scalar) {
    return ng_decode_core(params,n,dst,NULL,NULL,choices,scalar,4,sy,NULL);
}
int ng_fast_context_decode(const uint8_t* payload,size_t n,const uint8_t* choices,const uint8_t* meta,uint8_t* dst,int scalar) {
    if(n<6 || payload[4]!=16)return -1;
    return ng_decode_core(payload,n,dst,NULL,NULL,choices,scalar,4,NULL,meta);
}

/* Same traversal for encoding and decoding. NG uses the original tap math,
   without causal class bias. Delta bypasses generic NG recurrence entirely. */
int ng_fast_block_pass(const uint8_t* params,size_t n,const uint8_t* choices,
    const uint8_t* src,uint8_t* sy,uint8_t* dst,int scalar) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    const unsigned nb=VCN_LC_D/4,ny=VCN_LC_Y/4,nz=VCN_LC_Z/4;
    for(unsigned i=0;i<nb*ny*nz;i++)if(choices[i]>7){free(p);return -1;}
    uint8_t* b=malloc(VCN_LC_NB);if(!b){free(p);return -1;}memset(b,128,VCN_LC_NB);
    vcn_lc_plist pl[2];for(int s=0;s<2;s++)vcn_lc_mkplist(&p->t,p->w[s],&pl[s]);
    uint16_t pre[VCN_LC_D];
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++) {
        size_t ri=VCN_LC_IDX(z,y,0),o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        uint8_t* row=b+ri;const uint8_t* c=choices+((z/4)*ny+y/4)*nb;
        vcn_lc_row_begin(b,z,y,128);
        unsigned needed=0;for(unsigned bx=0;bx<nb;bx++)if(!c[bx])needed|=1u<<(bx/8);
        if(needed)vcn_lc_pre_selected(row,&pl[z==0],(uint16_t)p->bias[z==0],pre,needed,scalar);
        for(unsigned bx=0;bx<nb;bx++) {
            unsigned choice=c[bx],m=choice;if(!y)m&=~2u;if(!z)m&=~4u;
#if VCN_HAVE_AVX2
            if(choice && !src && !scalar && vcn_use_avx2()){ng_delta4(row,bx*4,m,sy+o+bx*4);continue;}
#endif
            for(unsigned x=bx*4;x<bx*4+4;x++) {
                unsigned pred,flip=0;
                if(!choice) {
                    uint16_t q=pre[x];
                    for(int k=0;k<p->t.nrow;k++)q=(uint16_t)(q+(uint16_t)p->w[z==0][k]*row[(int)x-1-k]);
                    pred=VCN_LC_PP(q);
                    flip=VCN_LC_FM(q);
                } else {
                    int v=((m&2)?row[(int)x-VCN_LC_RP]:0)+((m&4)?row[(int)x-VCN_LC_SP]:0)-((m&6)==6?row[(int)x-VCN_LC_RP-VCN_LC_SP]:0);
                    if((m&1) && x) v+=row[x-1]-((m&2)?row[(int)x-VCN_LC_RP-1]:0)-((m&4)?row[(int)x-VCN_LC_SP-1]:0)+((m&6)==6?row[(int)x-VCN_LC_RP-VCN_LC_SP-1]:0);
                    pred=(uint8_t)v;
                }
                if(src) {
                    unsigned r=(uint8_t)(src[o+x]-pred);if(flip)r=(uint8_t)(0u-r);
                    sy[o+x]=(uint8_t)((r<<1)^(0u-(r>>7)));
                    row[x]=src[o+x];
                } else {unsigned s=sy[o+x];int r=(s&1)?-(int)((s+1)/2):(int)(s/2);row[x]=(uint8_t)(pred+(flip?-r:r));}
            }
        }
        if(dst)memcpy(dst+o,row,VCN_LC_D);vcn_lc_row_end(b,z,y);
    }
    free(b);free(p);return 0;
}
int ng_wave_decode(const uint8_t* p,size_t n,const uint8_t* selectors,uint8_t* dst,int scalar) {
    if(n<6 || p[4]<2 || p[4]>VCN_LC_MAXK)return -1;
    int any=0;for(int c=0;c<p[4];c++)any|=selectors[c];
    if(!any)selectors=NULL;
    return ng_decode_core(p,n,dst,NULL,NULL,selectors,scalar,0,NULL,NULL);
}
int ng_block_decode(const uint8_t* p,size_t n,const uint8_t* selectors,uint8_t* dst,int scalar,int B) {
    if(B!=4 && B!=8 && B!=16)return -1;
    return ng_decode_core(p,n,dst,NULL,NULL,selectors,scalar,B,NULL,NULL);
}
int ng_block_stream_decode(const uint8_t* p,size_t n,const uint8_t* selectors,uint8_t* dst,int scalar,int B,ng_symbol_reader reader,void* ctx) {
    if((B!=4 && B!=8 && B!=16) || !reader)return -1;
    return ng_decode_core(p,n,dst,reader,ctx,selectors,scalar,B,NULL,NULL);
}
static int ng_schedule_export(int delta,uint32_t*rows,uint32_t*ends,uint32_t*groups){
    vcn_lc_sched sc={0,NULL,NULL};if(vcn_lc_mksched(delta,&sc))return -1;
    int i=0;for(int g=0;g<sc.ng;g++){for(int j=0;j<sc.n[g];j++)rows[i++]=sc.row[g][j];if(ends)ends[g]=i;}
    if(groups)*groups=sc.ng;free(sc.row);free(sc.n);return i==VCN_LC_Y*VCN_LC_Z?0:-1;
}
int ng_bank_route_order(const uint8_t*const*params,const size_t*lengths,unsigned count,uint32_t*rows,uint32_t*ends,uint32_t*groups){
    int delta=2;for(unsigned m=0;m<count;m++){
        const uint8_t*cursor=params[m];vcn_lc_par*p=NULL;
        if(vcn_lc_readpar(&cursor,lengths[m],&p)!=VCN_OK)return -1;
        if(cursor!=params[m]+lengths[m]||p->t.nrow>3){free(p);return -1;}
        int d=vcn_lc_delta(&p->t);if(d>delta)delta=d;free(p);
    }
    return ng_schedule_export(delta,rows,ends,groups);
}
int ng_wave_groups(const uint8_t* params,size_t n,uint32_t* rows,uint32_t* ends,uint32_t* groups) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    int rc=ng_schedule_export(vcn_lc_delta(&p->t),rows,ends,groups);free(p);return rc;
}
int ng_wave_rows(const uint8_t* params,size_t n,uint32_t* rows) {
    return ng_wave_groups(params,n,rows,NULL,NULL);
}
size_t ng_write_freq_table(uint8_t* out,const uint16_t* freq) {return vcn_lc_write_table(out,freq);}
int ng_read_freq_table(const uint8_t* src,size_t n,uint16_t* freq,size_t* used) {
    const uint8_t* p=src;int rc=vcn_lc_read_table(&p,src+n,freq,4096);*used=p-src;return rc;
}
/* Encoder already knows source samples and fitted NG tap predictions. Only
   activity depends on previous symbols; reuse original SIMD class/symbol math. */
static int ng_selected_symbols_impl(const uint8_t* params,size_t n,const uint8_t* selectors,const uint8_t* src,
    const uint16_t* base,const uint8_t* const* delta,uint8_t* sy,uint8_t* cl,uint16_t* activities,int B,
    const uint8_t* params2,size_t n2,const uint16_t* base2,uint8_t* candidate0,uint8_t* candidate2,int clamp_id,uint8_t* clamp_candidate) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    vcn_lc_par* p2=NULL;
    if(params2){const uint8_t* cur=params2;if(vcn_lc_readpar(&cur,n2,&p2)!=VCN_OK||cur!=params2+n2){free(p);free(p2);return -1;}}
    uint8_t* e=calloc(VCN_LC_NB,1);if(!e){free(p);free(p2);return -1;}
    uint16_t Q[VCN_LC_D];
    uint16_t Q2[VCN_LC_D];uint8_t cl2[VCN_LC_D],sy2[VCN_LC_D],er2[VCN_LC_D];
    int block_shift=B==4?2:B==8?3:4,nb=VCN_LC_D>>block_shift,ny=VCN_LC_Y>>block_shift;
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++) {
        size_t ri=VCN_LC_IDX(z,y,0),o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        vcn_lc_row_begin(e,z,y,0);
        if(activities)vcn_lc_apre(e,ri,activities+o);
        memcpy(Q,base+o,sizeof(Q));VCN_LC_CLS(e,ri,p,Q,cl+o);
        if(p2){memcpy(Q2,base2+o,sizeof(Q2));VCN_LC_CLS(e,ri,p2,Q2,cl2);VCN_LC_SYMS(Q2,src+o,sy2,er2);if(candidate2)memcpy(candidate2+o,sy2,VCN_LC_D);}
        VCN_LC_SYMS(Q,src+o,sy+o,e+ri);
        if(candidate0)memcpy(candidate0+o,sy+o,VCN_LC_D);
        if(clamp_candidate){
            uint16_t qc[VCN_LC_D];uint8_t ec[VCN_LC_D];
            for(int x=0;x<VCN_LC_D;x++)qc[x]=ng_clamp_prediction(Q[x],src,o+x,x,y,z,VCN_LC_D);
            VCN_LC_SYMS(qc,src+o,clamp_candidate+o,ec);
        }
        const uint8_t* row_choices=B?selectors+(((z>>block_shift)*ny+(y>>block_shift))*nb):NULL;
        for(int x=0;x<VCN_LC_D;x++) {
            unsigned sel=B?row_choices[x>>block_shift]:selectors[cl[o+x]];
            if(sel){sy[o+x]=(clamp_candidate&&sel==(unsigned)clamp_id)?clamp_candidate[o+x]:(p2&&sel==9)?sy2[x]:delta[sel][o+x];e[ri+x]=(uint8_t)((sy[o+x]+1u)>>1);}
        }
        vcn_lc_row_end(e,z,y);
    }
    free(e);free(p);free(p2);return 0;
}
int ng_selected_symbols_fast(const uint8_t* p,size_t n,const uint8_t* s,const uint8_t* src,
    const uint16_t* base,const uint8_t* const* delta,uint8_t* sy,uint8_t* cl,uint16_t* a) {
    return ng_selected_symbols_impl(p,n,s,src,base,delta,sy,cl,a,0,NULL,0,NULL,NULL,NULL,-1,NULL);
}
/* Multiple independently fitted models, all driven by the same actual selected
 * residual history. Fixed candidates precede the model candidates. Reuses the
 * native class and symbol kernels, including fractional Q8 sign mapping. */
_Thread_local int ng_bank_selected_only=0;
#include "ng_bank_symbols.h"
#define NG_BANK_ARGS unsigned c,const uint8_t*const*p,const size_t*n,const uint16_t*const*b,const uint8_t*s,const uint8_t*x,uint8_t*const*v,uint8_t*y,uint16_t*a
int ng_fixed32_bank_symbols(NG_BANK_ARGS);
int ng_fixed64_bank_symbols(NG_BANK_ARGS);
int ng_fixed128_bank_symbols(NG_BANK_ARGS);
int ng_fixed256_bank_symbols(NG_BANK_ARGS);
int ng_bank_symbols(NG_BANK_ARGS){
    if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z){
        if(NG_X==32)return ng_fixed32_bank_symbols(c,p,n,b,s,x,v,y,a);
        if(NG_X==64)return ng_fixed64_bank_symbols(c,p,n,b,s,x,v,y,a);
        if(NG_X==128)return ng_fixed128_bank_symbols(c,p,n,b,s,x,v,y,a);
        if(NG_X==256)return ng_fixed256_bank_symbols(c,p,n,b,s,x,v,y,a);
    }
    return ng_bank_symbols_impl(c,p,n,b,s,x,v,y,a);
}
int ng_bank_selected_symbols(NG_BANK_ARGS){
    int previous=ng_bank_selected_only;ng_bank_selected_only=1;
    int result=ng_bank_symbols(c,p,n,b,s,x,v,y,a);
    ng_bank_selected_only=previous;return result;
}
#undef NG_BANK_ARGS
#define NG_SHARED_ARGS unsigned c,const uint8_t*const*p,const size_t*n,const uint16_t*const*b,const uint8_t*s,const uint8_t*x,uint8_t*const*v,uint8_t*y,uint16_t*a,const int32_t*w,uint16_t*q
int ng_fixed32_shared_symbols(NG_SHARED_ARGS);
int ng_fixed64_shared_symbols(NG_SHARED_ARGS);
int ng_fixed128_shared_symbols(NG_SHARED_ARGS);
int ng_fixed256_shared_symbols(NG_SHARED_ARGS);
int ng_bank_shared_symbols(NG_SHARED_ARGS){
    if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z){
        if(NG_X==32)return ng_fixed32_shared_symbols(c,p,n,b,s,x,v,y,a,w,q);
        if(NG_X==64)return ng_fixed64_shared_symbols(c,p,n,b,s,x,v,y,a,w,q);
        if(NG_X==128)return ng_fixed128_shared_symbols(c,p,n,b,s,x,v,y,a,w,q);
        if(NG_X==256)return ng_fixed256_shared_symbols(c,p,n,b,s,x,v,y,a,w,q);
    }
    return ng_bank_symbols_core(c,p,n,b,s,x,v,y,a,w,q);
}
#undef NG_SHARED_ARGS
int ng_block_symbols(const uint8_t* p,size_t n,const uint8_t* s,const uint8_t* src,
    const uint16_t* base,const uint8_t* const* delta,uint8_t* sy,uint8_t* cl,int B) {
    if(B!=4 && B!=8 && B!=16)return -1;
    return ng_selected_symbols_impl(p,n,s,src,base,delta,sy,cl,NULL,B,NULL,0,NULL,NULL,NULL,-1,NULL);
}
int ng_multi_symbols(const uint8_t* p,size_t n,const uint8_t* p2,size_t n2,const uint8_t* choices,const uint8_t* src,
    const uint16_t* base,const uint16_t* base2,const uint8_t* const* fixed,uint8_t* sy,uint8_t* cl,uint16_t* activity,uint8_t* candidate0,uint8_t* candidate2) {
    return ng_selected_symbols_impl(p,n,choices,src,base,fixed,sy,cl,activity,4,p2,n2,base2,candidate0,candidate2,-1,NULL);
}
int ng_multi_clamp_symbols(const uint8_t* p,size_t n,const uint8_t* p2,size_t n2,const uint8_t* choices,const uint8_t* src,
    const uint16_t* base,const uint16_t* base2,const uint8_t* const* fixed,uint8_t* sy,uint8_t* cl,uint16_t* activity,uint8_t* candidate0,uint8_t* candidate2,int clamp_id,uint8_t* cc) {
    return ng_selected_symbols_impl(p,n,choices,src,base,fixed,sy,cl,activity,4,p2,n2,base2,candidate0,candidate2,clamp_id,cc);
}
/* Incremental access to the SAME native NG row prediction and class kernels.
   Allows two channels to be reconstructed in voxel lockstep without a duplicate
   predictor/context implementation. Geometry is scoped by the caller. */
#include "ng_causal_state.h"
#include "ng_sparse_bank.h"
#define NG_SPARSE_DECLARE(d) \
void*ng_fixed##d##_sparse_create(const uint8_t*,const uint8_t*,size_t,const uint8_t*,unsigned); \
int ng_fixed##d##_sparse_fit(void*,unsigned,const uint8_t*,uint8_t*,size_t,size_t*,int32_t*); \
int ng_fixed##d##_sparse_block(void*,unsigned,unsigned,unsigned,unsigned,uint8_t*); \
int ng_fixed##d##_sparse_emit(void*,const uint8_t*,uint8_t*,uint16_t*);
NG_SPARSE_DECLARE(32) NG_SPARSE_DECLARE(64) NG_SPARSE_DECLARE(128) NG_SPARSE_DECLARE(256)
#define NG_SPARSE_DISPATCH(fn,...) \
    if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z){ \
        if(NG_X==32)return ng_fixed32_sparse_##fn(__VA_ARGS__); \
        if(NG_X==64)return ng_fixed64_sparse_##fn(__VA_ARGS__); \
        if(NG_X==128)return ng_fixed128_sparse_##fn(__VA_ARGS__); \
        if(NG_X==256)return ng_fixed256_sparse_##fn(__VA_ARGS__); }
void*ng_sparse_create(const uint8_t*s,const uint8_t*p,size_t n,const uint8_t*c,unsigned stride){NG_SPARSE_DISPATCH(create,s,p,n,c,stride);return ng_sparse_create_impl(s,p,n,c,stride);}
void ng_sparse_destroy(void*s){ng_sparse_destroy_impl(s);}
void*ng_fit_probe_create(const uint8_t*s){return ng_fit_probe_create_impl(s);}
void ng_fit_probe_destroy(void*s){ng_sparse_destroy_impl(s);}
int ng_fit_probe_samples(void*s,unsigned m,const uint32_t*p,const uint8_t*a,size_t n,unsigned budget,uint8_t*out){return ng_fit_probe_samples_impl(s,m,p,a,n,budget,out);}
int ng_tap_probe_score(void*ptr,unsigned m,const int32_t*q0,const int32_t*q1,const int32_t*bias,const uint32_t*positions,size_t n,double*out){
    ng_sparse_bank*s=ptr;if(!s||m>=12||!n||n%4)return -1;
    vcn_lc_par*p=&s->p[m];if(m)*p=s->p[0];memset(p->cb,0,sizeof(p->cb));
    for(int b=0;b<2;b++){const int32_t*q=b?q1:q0;memcpy(p->w[b],q,p->t.nt*sizeof(int32_t));
      p->bias[b]=bias[b];s->fw[m][b]=q[p->t.nt];vcn_lc_mkplist(&p->t,p->w[b],&s->pl[m][b]);}
    uint32_t hist[256]={0};uint8_t symbols[4];
    for(size_t j=0;j<n;j+=4){size_t i=positions[j];unsigned x=i%VCN_LC_D,y=i/VCN_LC_D%VCN_LC_Y,z=i/((size_t)VCN_LC_D*VCN_LC_Y);
      if(z>=VCN_LC_Z||x+4>VCN_LC_D||positions[j+3]!=i+3)return -1;
      ng_sparse_row(s,m,x,y,z,NULL,symbols);for(int k=0;k<4;k++)hist[symbols[k]]++;
    }
    *out=ng_hist_cost(hist,12);return 0;
}
int ng_sparse_fit(void*s,unsigned m,const uint8_t*c,uint8_t*p,size_t cap,size_t*n,int32_t*w){NG_SPARSE_DISPATCH(fit,s,m,c,p,cap,n,w);return ng_sparse_fit_impl(s,m,c,p,cap,n,w);}
int ng_sparse_block(void*s,unsigned m,unsigned x,unsigned y,unsigned z,uint8_t*out){NG_SPARSE_DISPATCH(block,s,m,x,y,z,out);return ng_sparse_block_impl(s,m,x,y,z,out);}
int ng_sparse_emit(void*s,const uint8_t*c,uint8_t*y,uint16_t*a){NG_SPARSE_DISPATCH(emit,s,c,y,a);return ng_sparse_emit_impl(s,c,y,a);}
#undef NG_SPARSE_DISPATCH
#define NG_DECLARE_ACTIVITY(d) \
void ng_fixed##d##_activity_begin(void*,int,int,const uint8_t**,uint8_t**); \
void ng_fixed##d##_activity_end(void*,int,int);
NG_DECLARE_ACTIVITY(32)
NG_DECLARE_ACTIVITY(64)
NG_DECLARE_ACTIVITY(128)
NG_DECLARE_ACTIVITY(256)
void ng_causal_activity_begin(void* ptr,int z,int y,const uint8_t**classes,uint8_t**errors) {
    if(!ng_runtime_only && NG_X==NG_Y && NG_X==NG_Z) {
        if(NG_X==32){ng_fixed32_activity_begin(ptr,z,y,classes,errors);return;}
        if(NG_X==64){ng_fixed64_activity_begin(ptr,z,y,classes,errors);return;}
        if(NG_X==128){ng_fixed128_activity_begin(ptr,z,y,classes,errors);return;}
        if(NG_X==256){ng_fixed256_activity_begin(ptr,z,y,classes,errors);return;}
    }
    ng_activity_row_begin_impl(ptr,z,y,classes,errors);
}
void ng_causal_activity_end(void* ptr,int z,int y) {
    if(!ng_runtime_only && NG_X==NG_Y && NG_X==NG_Z) {
        if(NG_X==32){ng_fixed32_activity_end(ptr,z,y);return;}
        if(NG_X==64){ng_fixed64_activity_end(ptr,z,y);return;}
        if(NG_X==128){ng_fixed128_activity_end(ptr,z,y);return;}
        if(NG_X==256){ng_fixed256_activity_end(ptr,z,y);return;}
    }
    ng_activity_row_end_impl(ptr,z,y);
}
void ng_causal_destroy(void* ptr) {
    ng_causal_state* s=ptr;if(!s)return;free(s->p);free(s->b);free(s->e);free(s->classes);free(s->pre);free(s->feature_activity);free(s);
}
static void* ng_causal_create_impl(const uint8_t* params,size_t n,int history) {
    ng_causal_state* s=calloc(1,sizeof(*s));if(!s)return NULL;
    const uint8_t* cursor=params;
    if(vcn_lc_readpar(&cursor,n,&s->p)!=VCN_OK || cursor!=params+n){ng_causal_destroy(s);return NULL;}
    if(history){s->b=malloc(VCN_LC_NB);s->e=calloc(VCN_LC_NB,1);}
    s->classes=malloc(VCN_LC_D);s->pre=malloc(2*VCN_LC_D);
    if((history&&(!s->b||!s->e))||!s->classes||!s->pre){ng_causal_destroy(s);return NULL;}
    if(history)memset(s->b,128,VCN_LC_NB);
    for(int k=0;k<2;k++)vcn_lc_mkplist(&s->p->t,s->p->w[k],&s->pl[k]);return s;
}
void* ng_causal_create(const uint8_t* p,size_t n){return ng_causal_create_impl(p,n,1);}
void* ng_causal_create_model(const uint8_t* p,size_t n){return ng_causal_create_impl(p,n,0);}
int ng_causal_used_taps(void*ptr,int*xyz){
    ng_causal_state*s=ptr;if(!s||!xyz)return -1;int n=0;
    for(int k=0;k<s->p->t.nt;k++)if(s->p->w[0][k]||s->p->w[1][k]){
        xyz[3*n]=s->p->t.dx[k];xyz[3*n+1]=s->p->t.dy[k];xyz[3*n+2]=s->p->t.dz[k];n++;
    }return n;
}
_Thread_local int ng_bank_scalar_control=14; /* cached shared-model SIMD; NWP1 uses integrated entropy */
_Thread_local ng_prediction_reader ng_bank_prediction_reader=NULL;
_Thread_local ng_prediction_transform ng_bank_prediction_transform=NULL;
void ng_bank_set_prediction_transform(ng_prediction_transform fn){ng_bank_prediction_transform=fn;}
_Thread_local unsigned ng_bank_trial_flags=0;
_Thread_local ng_route_context* ng_bank_routed=NULL;
_Thread_local const uint8_t* ng_bank_external_payload=NULL;
_Thread_local size_t ng_bank_external_payload_size=0;
_Thread_local unsigned ng_bank_wave_model_count=12;
_Thread_local const uint8_t* ng_bank_entropy_banks=NULL;
_Thread_local unsigned ng_bank_entropy_classes=16;
_Thread_local int ng_bank_choice_layout=0;
_Thread_local int ng_bank_multi_row_cache=1;
void ng_bank_set_multi_row_cache(int enabled){ng_bank_multi_row_cache=!!enabled;}
/* Experimental per-thread benchmark control, not encoded in the format. */
void ng_bank_set_scalar(int enabled){ng_bank_scalar_control=enabled;}
#include "ng_bank_reconstruct.h"
#include "ng_context_rans.h"
int ng_bank_reconstruct_many(void* const*s,const int32_t*w,const uint8_t*c,uint8_t*out,unsigned count){
    return ng_bank_reconstruct_scalar_count(s,1,w,c,out,count);
}
#define NG_BANK_DECLARE(d) int ng_fixed##d##_bank_reconstruct(void* const*,unsigned,const int32_t*,const uint8_t*,uint8_t*); int ng_fixed##d##_bank_wave(void* const*,unsigned,const int32_t*,const uint8_t*,uint8_t*,const uint8_t*,size_t);
NG_BANK_DECLARE(32)
NG_BANK_DECLARE(64)
NG_BANK_DECLARE(128)
NG_BANK_DECLARE(256)
int ng_bank_reconstruct(void* const*s,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out){
    if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z){
        if(NG_X==32)return ng_fixed32_bank_reconstruct(s,mode,w,c,out);
        if(NG_X==64)return ng_fixed64_bank_reconstruct(s,mode,w,c,out);
        if(NG_X==128)return ng_fixed128_bank_reconstruct(s,mode,w,c,out);
        if(NG_X==256)return ng_fixed256_bank_reconstruct(s,mode,w,c,out);
    }
    return ng_bank_reconstruct_impl(s,mode,w,c,out);
}
int ng_bank_wave(void* const*s,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,const uint8_t*e,size_t n){
    unsigned saved_flags=ng_bank_trial_flags;
    ng_bank_trial_flags=ng_bank_scalar_control==29?1:ng_bank_scalar_control==31?2:ng_bank_scalar_control==33?4:ng_bank_scalar_control==35?7:ng_bank_scalar_control==37?8:ng_bank_scalar_control==39?10:(ng_bank_scalar_control==41||ng_bank_scalar_control==14)?11:0;
    if(ng_bank_scalar_control==47)ng_bank_trial_flags=11|64;
    if(ng_bank_scalar_control==49)ng_bank_trial_flags=11|128;
    if(ng_bank_scalar_control==51)ng_bank_trial_flags=11|256;
    if(ng_bank_scalar_control==55)ng_bank_trial_flags=11|64|128;
    if(ng_bank_scalar_control==57)ng_bank_trial_flags=11|64|128|256;
    if(ng_bank_scalar_control==65)ng_bank_trial_flags=11|64|128|256|512;
    if(ng_bank_scalar_control==69)ng_bank_trial_flags=11|64|128|256|512|2048;
    if(ng_bank_scalar_control==14||ng_bank_scalar_control==109)ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|262144;
    if(ng_bank_scalar_control==111)ng_bank_trial_flags=11|64|128|256|512|2048|524288; /* scalar entropy regression */
    if(ng_bank_scalar_control==153)ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|262144|1048576|16777216;
    if(ng_bank_scalar_control==155)ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|262144|1048576|16777216|2097152;
    if(ng_bank_scalar_control==157)ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|262144|1048576|16777216|4194304;
    /* Exact compact metadata + fused preparation; no format/model changes. */
    if(ng_bank_scalar_control==14)ng_bank_trial_flags|=1048576|16777216|4194304|67108864|134217728|536870912|0x80000000u;
    if(ng_bank_scalar_control==161)ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|262144|1048576|16777216|4194304|67108864;
    if(ng_bank_scalar_control==163)ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|262144|1048576|16777216|4194304|67108864|134217728;
    if(ng_bank_scalar_control==173)ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|262144|1048576|16777216|4194304|67108864|134217728|0x80000000u;
    if(ng_bank_scalar_control==167||ng_bank_scalar_control==177){
        ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|262144|1048576|16777216|4194304|67108864|134217728|536870912;
        if(ng_bank_scalar_control==177)ng_bank_trial_flags|=0x80000000u;
    }
    int saved=ng_bank_scalar_control;ng_bank_scalar_control=saved==25?24:saved==23?22:saved==19?18:saved==17?16:saved==15?14:saved==13?12:20;int rc;
    /* Preserve generic kernels when the NG+X shortcut cannot amortize its
     * dispatch/preparation overhead. Selection depends only on saved choices. */
    if(saved==27||saved==14||ng_bank_trial_flags){
        size_t count=(size_t)(NG_X/ng_choice_edge)*(NG_Y/ng_choice_edge)*(NG_Z/ng_choice_edge),nx=0;
        for(size_t i=0;i<count;i++)nx+=c[i]==13;
        if(nx>=count/4)ng_bank_scalar_control=24;
    }
    if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z&&NG_X==32)rc=ng_fixed32_bank_wave(s,mode,w,c,out,e,n);
    else if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z&&NG_X==64)rc=ng_fixed64_bank_wave(s,mode,w,c,out,e,n);
    else if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z&&NG_X==128)rc=ng_fixed128_bank_wave(s,mode,w,c,out,e,n);
    else if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z&&NG_X==256)rc=ng_fixed256_bank_wave(s,mode,w,c,out,e,n);
    else rc=ng_bank_reconstruct_core(s,mode,w,c,out,e,n);
    ng_bank_scalar_control=saved;ng_bank_trial_flags=saved_flags;return rc;
}
int ng_bank_wave_segments(void* const*s,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,const uint8_t*header,size_t hn,const uint8_t*payload,size_t pn){
    const uint8_t* saved=ng_bank_external_payload;size_t saved_n=ng_bank_external_payload_size;
    ng_bank_external_payload=payload;ng_bank_external_payload_size=pn;
    int rc=ng_bank_wave(s,mode,w,c,out,header,hn);
    ng_bank_external_payload=saved;ng_bank_external_payload_size=saved_n;return rc;
}
int ng_bank_wave_segments_many(void* const*s,unsigned count,const int32_t*w,const uint8_t*c,uint8_t*out,const uint8_t*header,size_t hn,const uint8_t*payload,size_t pn){
    if(count<12||count>48||count%12)return -1;
    for(unsigned m=0;m<count;m++)if(s[m]!=s[(m/12)*12])return -1;
    unsigned saved=ng_bank_wave_model_count;int control=ng_bank_scalar_control;
    ng_bank_wave_model_count=count;ng_bank_scalar_control=14;
    int rc=ng_bank_wave_segments(s,1,w,c,out,header,hn,payload,pn);
    ng_bank_scalar_control=control;ng_bank_wave_model_count=saved;return rc;
}
int ng_choice_wave_decode(void* const*s,unsigned count,unsigned mode,const int32_t*w,const uint8_t*c,const uint8_t*banks,uint8_t*out,const uint8_t*header,size_t hn,const uint8_t*payload,size_t pn,unsigned K){
    if(count<12||count>48||count%12||mode>1||K<2||K>32)return -1;
    unsigned oldK=ng_bank_entropy_classes;ng_bank_entropy_classes=K;
    unsigned saved=ng_bank_wave_model_count;const uint8_t*old=ng_bank_entropy_banks;int control=ng_bank_scalar_control,layout=ng_bank_choice_layout;
    ng_bank_wave_model_count=count;ng_bank_entropy_banks=banks;ng_bank_scalar_control=14;ng_bank_choice_layout=1;
    int rc=ng_bank_wave_segments(s,mode,w,c,out,header,hn,payload,pn);
    ng_bank_wave_model_count=saved;ng_bank_entropy_banks=old;ng_bank_scalar_control=control;ng_bank_choice_layout=layout;ng_bank_entropy_classes=oldK;return rc;
}
#define NG_ROUTE_DECLARE(D) int ng_fixed##D##_bank_route(void* const*,unsigned,unsigned,const int32_t*,const uint8_t*,uint8_t*,const uint16_t*,unsigned,ng_symbol_reader,void*,uint8_t*);
NG_ROUTE_DECLARE(32) NG_ROUTE_DECLARE(64) NG_ROUTE_DECLARE(128) NG_ROUTE_DECLARE(256)
int ng_bank_route_decode(void* const*s,unsigned count,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,const uint16_t*t,unsigned K,ng_symbol_reader reader,void*ctx,uint8_t*raw){
    if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z){
#define NG_ROUTE_CASE(D) if(NG_X==D)return ng_fixed##D##_bank_route(s,count,mode,w,c,out,t,K,reader,ctx,raw);
        NG_ROUTE_CASE(32) NG_ROUTE_CASE(64) NG_ROUTE_CASE(128) NG_ROUTE_CASE(256)
#undef NG_ROUTE_CASE
    }
    return ng_bank_route_impl(s,count,mode,w,c,out,t,K,reader,ctx,raw);
}
static int ng_prediction_unused(void*ctx,size_t i,unsigned cls){(void)ctx;(void)i;(void)cls;return -1;}
int ng_bank_prediction_decode(void* const*s,unsigned count,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,const uint16_t*t,unsigned K,ng_prediction_reader reader,void*ctx){
    if(!reader)return -1;
    ng_prediction_reader old=ng_bank_prediction_reader;ng_bank_prediction_reader=reader;
    int rc=ng_bank_route_decode(s,count,mode,w,c,out,t,K,ng_prediction_unused,ctx,NULL);
    ng_bank_prediction_reader=old;return rc;
}
int ng_bank_route_wave(void* const*s,unsigned count,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,ng_route_context*ctx){
    unsigned old_count=ng_bank_wave_model_count;int old_control=ng_bank_scalar_control,old_cache=ng_bank_multi_row_cache;
    ng_route_context*old=ng_bank_routed;ng_bank_routed=ctx;ng_bank_wave_model_count=count;ng_bank_scalar_control=0;ng_bank_multi_row_cache=1;
    int rc;
    if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z&&NG_X==32)rc=ng_fixed32_bank_wave(s,mode,w,c,out,NULL,0);
    else if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z&&NG_X==64)rc=ng_fixed64_bank_wave(s,mode,w,c,out,NULL,0);
    else if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z&&NG_X==128)rc=ng_fixed128_bank_wave(s,mode,w,c,out,NULL,0);
    else if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z&&NG_X==256)rc=ng_fixed256_bank_wave(s,mode,w,c,out,NULL,0);
    else rc=ng_bank_reconstruct_core(s,mode,w,c,out,NULL,0);
    ng_bank_routed=old;ng_bank_wave_model_count=old_count;ng_bank_scalar_control=old_control;ng_bank_multi_row_cache=old_cache;return rc;
}
int ng_causal_prediction(void* ptr,size_t i,uint16_t* q,uint8_t* cls) {
    ng_causal_state* s=ptr;if(!s||i!=s->next||s->pending||i>=NG_N)return -1;
    int x=i%VCN_LC_D,y=i/VCN_LC_D%VCN_LC_Y,z=i/VCN_LC_D/VCN_LC_Y;
    size_t ri=VCN_LC_IDX(z,y,0);
    if(!x){vcn_lc_row_begin(s->b,z,y,128);vcn_lc_row_begin(s->e,z,y,0);
        VCN_LC_PRE(s->b+ri,&s->pl[z==0],(uint16_t)s->p->bias[z==0],s->pre);
        VCN_LC_CLS(s->e,ri,s->p,s->pre,s->classes);}
    uint16_t value=s->pre[x];for(int k=0;k<s->p->t.nrow;k++)value=(uint16_t)(value+(uint16_t)s->p->w[z==0][k]*s->b[ri+x-1-k]);
    if(s->p->has_context)for(int k=0;k<8;k++)value=(uint16_t)(value+(int64_t)s->p->context[z==0][k]*ng_context_input(s->b,ri+x,z==0,k));
    *q=value;*cls=s->classes[x];s->pending=1;return 0;
}
int ng_causal_add_context(void*ptr,size_t i,const int32_t*weights,uint16_t*q){
    ng_causal_state*s=ptr;if(!s||!weights||!q||i!=s->next||!s->pending||i>=NG_N)return -1;
    int x=i%VCN_LC_D,y=i/VCN_LC_D%VCN_LC_Y,z=i/VCN_LC_D/VCN_LC_Y;
    size_t at=VCN_LC_IDX(z,y,x);for(int k=0;k<8;k++)*q=(uint16_t)(*q+(int64_t)weights[8*(z==0)+k]*ng_context_input(s->b,at,z==0,k));
    return 0;
}
int ng_causal_accept(void* ptr,size_t i,uint8_t value,uint8_t predictor) {
    ng_causal_state* s=ptr;if(!s||i!=s->next||!s->pending)return -1;
    int x=i%VCN_LC_D,y=i/VCN_LC_D%VCN_LC_Y,z=i/VCN_LC_D/VCN_LC_Y;size_t ri=VCN_LC_IDX(z,y,x);
    int e=(int8_t)(uint8_t)(value-predictor);s->b[ri]=value;s->e[ri]=(uint8_t)(e<0?-e:e);
    if(x==VCN_LC_D-1){vcn_lc_row_end(s->b,z,y);vcn_lc_row_end(s->e,z,y);}
    s->pending=0;s->next++;return 0;
}
#include "ng_difficulty_model.h"
static int ng_compact_read(void* context,size_t i,unsigned c) {
    vcn_lc_dec* d=context;
    return vcn_lc_sym_one(d,c,i&(VCN_LC_NS-1));
}
int ng_compact_decode(const uint8_t* frame,size_t n,const uint8_t* selectors,uint8_t* dst) {
    const uint8_t* cursor=frame,*end=frame+n;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    size_t pn=cursor-frame;uint32_t M=1u<<p->tb;
    uint32_t* T=malloc(sizeof(uint32_t)*M*p->K);
    if(!T){free(p);return -1;}
    int rc=-1;
    if(!vcn_lc_read_tables(&cursor,end,T,p->K,M) && end-cursor>=4*VCN_LC_NS) {
        vcn_lc_dec d={0};d.T=T;d.tb=p->tb;d.end=end;
        for(int k=0;k<VCN_LC_NS;k++)d.S[k]=vcn_rd32(cursor+4*k);
        d.p=cursor+4*VCN_LC_NS;
        rc=ng_selected_pass(frame,pn,selectors,0,ng_compact_read,&d,dst,NULL,NULL,NULL);
        if(d.over || d.p!=end)rc=-1;
        for(int k=0;k<VCN_LC_NS;k++)if(d.S[k]!=VCN_LC_RL)rc=-1;
    }
    free(T);free(p);return rc;
}

/* Contexts for an already-filtered byte stream: same causal activity stencil
   as NG, but quantiles fitted on this stream rather than the pre-filter one. */
#include "ng_activity_values.h"
static int ng_activity_dispatch(const uint8_t*s,uint16_t*a,int symbols,uint32_t*h){
    const ng_fixed_kernels*f=ng_fixed_current();
    return f?f->activity_values(s,a,symbols,h):ng_activity_values_impl(s,a,symbols,h);
}
int ng_activity_values(const uint8_t* raw,uint16_t* activity) {return ng_activity_dispatch(raw,activity,0,NULL);}
int ng_activity_symbol_values(const uint8_t* symbols,uint16_t* activity) {return ng_activity_dispatch(symbols,activity,1,NULL);}
int ng_activity_symbol_histogram(const uint8_t* symbols,uint16_t* activity,uint32_t* histogram) {
    if(!symbols||!activity||!histogram)return -1;
    return ng_activity_dispatch(symbols,activity,1,histogram);
}
int ng_activity_classes(const uint8_t* raw,int K,uint16_t* thr,uint8_t* classes) {
    if(K<2||K>VCN_LC_MAXK)return -1;
    uint16_t* activity=malloc(NG_N*sizeof(uint16_t));
    if(!activity)return -1;
    if(ng_activity_values(raw,activity)){free(activity);return -1;}
    uint32_t hist[VCN_LC_AMAX]={0};
    for(size_t i=0;i<NG_N;i++)hist[activity[i]]++;
    vcn_lc_par p={0};p.K=K;uint64_t acc=0;int k=0;
    for(int a=0;a<VCN_LC_AMAX && k<K-1;a++) {
        acc+=hist[a];while(k<K-1 && acc*K>=(uint64_t)(k+1)*NG_N)p.thr[k++]=(uint16_t)a;
    }
    while(k<K-1)p.thr[k++]=VCN_LC_AMAX-1;
    vcn_lc_mklut(&p);memcpy(thr,p.thr,(K-1)*sizeof(uint16_t));
    for(size_t i=0;i<NG_N;i++)classes[i]=p.lut[activity[i]];
    free(activity);return 0;
}
#include "ng_activity_decode.h"
int ng_activity_decode(const uint16_t*thr,int K,int unmapped,ng_symbol_reader reader,void*ctx,uint8_t*raw){
    return ng_activity_decode_impl(thr,K,unmapped,reader,ctx,NULL,NULL,NULL,raw);
}
#define NG_GATHER_DECLARE(d) int ng_fixed##d##_activity_gather(const uint16_t*,int,const uint8_t*const*,const size_t*,size_t*,uint8_t*);
NG_GATHER_DECLARE(32)
NG_GATHER_DECLARE(64)
NG_GATHER_DECLARE(128)
NG_GATHER_DECLARE(256)
int ng_activity_gather(const uint16_t*t,int K,const uint8_t*const*s,const size_t*n,size_t*p,uint8_t*out){
    if(!ng_runtime_only&&NG_X==NG_Y&&NG_X==NG_Z){
        if(NG_X==32)return ng_fixed32_activity_gather(t,K,s,n,p,out);
        if(NG_X==64)return ng_fixed64_activity_gather(t,K,s,n,p,out);
        if(NG_X==128)return ng_fixed128_activity_gather(t,K,s,n,p,out);
        if(NG_X==256)return ng_fixed256_activity_gather(t,K,s,n,p,out);
    }
    return ng_activity_decode_impl(t,K,0,NULL,NULL,s,n,p,out);
}
