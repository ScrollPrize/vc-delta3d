/* Shared native C entrypoints for the retained presets.
 * MIT License, Copyright (c) 2026 Chase Hendrick; see LICENSE. */
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
_Thread_local int ng_fit_context_mode=0; /* scoped hybrid predictor experiment */
_Thread_local int ng_nonzero_sparse=1; /* preserve sparse budget over masked targets */
_Thread_local double ng_fit_intercept_weight=0.; /* relative to real sample mass */

int ng_geometry_valid(uint32_t x,uint32_t y,uint32_t z) {
    /* Bounds make padded pitches and signed SIMD offsets safe too. */
    return x>=32 && y>=32 && z>=32 && !(x%32) && !(y%32) && !(z%32) &&
           (uint64_t)y*z<=65536 && x<=1024;
}

#define VCN_LC_TICK(k) ((void)0)
#include "volcodec_ng_lossless.h"
#define NG_DECLARE_FIXED(d) \
int ng_fixed##d##_encode_ex(const uint8_t*,const void*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*); \
int ng_fixed##d##_activity_values(const uint8_t*,uint16_t*,int,uint32_t*); \
int ng_fixed##d##_decode_core(const uint8_t*,size_t,uint8_t*,ng_symbol_reader,void*,const uint8_t*,int,int,const uint8_t*,const uint8_t*);
NG_DECLARE_FIXED(32)
NG_DECLARE_FIXED(64)
NG_DECLARE_FIXED(128)
NG_DECLARE_FIXED(256)
typedef struct {
    int (*encode)(const uint8_t*,const void*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*);
    int (*decode)(const uint8_t*,size_t,uint8_t*,ng_symbol_reader,void*,const uint8_t*,int,int,const uint8_t*,const uint8_t*);
    int (*activity_values)(const uint8_t*,uint16_t*,int,uint32_t*);
} ng_fixed_kernels;
#define NG_FIXED_ENTRY(d) {ng_fixed##d##_encode_ex,ng_fixed##d##_decode_core,ng_fixed##d##_activity_values}
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


static int ng_config(const int* a,vcn_lc_cfg* cf) {
    if (!a || a[0]<1 || a[0]>3 || a[1]<-1 || a[1]>3 || a[2]<-1 || a[2]>3 ||
        a[3]<2 || a[3]>32 || a[4]<2 || a[4]>160 || a[5]<1 || a[5]>128) return -1;
    *cf=(vcn_lc_cfg){a[0],a[1],a[2],a[3],VCN_LC_TB,a[4],a[5]};return 0;
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

/* New==NULL: refit on old nonzero tap support. Otherwise materialize the
   old + (new-old)/2^shift model without fitting again. */

int ng_feature_prepare_activity(const uint8_t*src,const int*config,const uint8_t*feature,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint16_t*activity,uint32_t*histogram){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src||!feature||!weights||!activity||!histogram)return -1;
    cf.fit_feature=feature;cf.feature_mode=1;cf.export_feature_weights=weights;
    cf.export_activity=activity;cf.export_activity_histogram=histogram;
    return ng_encode_core(src,&cf,params,cap,n,sy,NULL);
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

/* Decoder for experimental integrated predictors. Classes use reconstructed
   residuals of the shared NG model, so class-table selection needs no voxel
   selectors and does not depend on the unknown current voxel. */

/* One shared causal traversal for encoding and decoding selected residuals. */

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

int ng_wave_encode(const uint8_t* p,size_t n,const uint8_t* sy,const uint8_t* cl,uint8_t* out,size_t cap,size_t* size) {
    return ng_compact_encode_order(p,n,sy,cl,out,cap,size,0,NULL,NULL);
}
/* Experimental full-data score: exact normalized tables/header, ideal symbol
 * bits plus native state storage. No payload encoding or wavefront traversal. */

/* Experimental fixed-context raster entropy: reuse the native tables, states
   and SIMD rANS, without constructing causal residual activity. */

int ng_bank_empty_model(uint8_t* out,size_t cap,size_t* size){
    vcn_lc_taps t;vcn_lc_build(&t,1,0,0);
    size_t need=6+4*(t.nt+1)+2*15+2*16;if(cap<need)return -1;
    memset(out,0,need);out[0]=VCN_LC_VERSION;out[1]=1;out[4]=16;out[5]=12;*size=need;return 0;
}

/* Append input columns, but call the existing production fitter unchanged.
 * Extra planes use constant offsets solely during feature gathering; they
 * never enter the serialized native spatial-tap geometry. */

int ng_entropy_header_size(const uint8_t* data,size_t n,size_t* size){
    const uint8_t* p=data;vcn_lc_par* par=NULL;if(vcn_lc_readpar(&p,n,&par)!=VCN_OK)return -1;
    int rc=vcn_lc_read_tables(&p,data+n,NULL,par->K,1u<<par->tb);
    free(par);if(rc)return -1;*size=p-data;return 0;
}

#if VCN_HAVE_AVX2

#endif

/* Same traversal for encoding and decoding. NG uses the original tap math,
   without causal class bias. Delta bypasses generic NG recurrence entirely. */

int ng_block_decode(const uint8_t* p,size_t n,const uint8_t* selectors,uint8_t* dst,int scalar,int B) {
    if(B!=4 && B!=8 && B!=16)return -1;
    return ng_decode_core(p,n,dst,NULL,NULL,selectors,scalar,B,NULL,NULL);
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
int ng_block_symbols(const uint8_t* p,size_t n,const uint8_t* s,const uint8_t* src,
    const uint16_t* base,const uint8_t* const* delta,uint8_t* sy,uint8_t* cl,int B) {
    if(B!=4 && B!=8 && B!=16)return -1;
    return ng_selected_symbols_impl(p,n,s,src,base,delta,sy,cl,NULL,B,NULL,0,NULL,NULL,NULL,-1,NULL);
}

/* Incremental access to the SAME native NG row prediction and class kernels.
   Allows two channels to be reconstructed in voxel lockstep without a duplicate
   predictor/context implementation. Geometry is scoped by the caller. */
#include "ng_causal_state.h"

void ng_causal_destroy(void* ptr) {
    ng_causal_state* s=ptr;if(!s)return;free(s->p);free(s->b);free(s->e);free(s->classes);free(s->pre);free(s);
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

_Thread_local int ng_bank_scalar_control=14; /* cached shared-model SIMD; NWP1 uses integrated entropy */

_Thread_local unsigned ng_bank_trial_flags=0;
_Thread_local const uint8_t* ng_bank_external_payload=NULL;
_Thread_local size_t ng_bank_external_payload_size=0;
_Thread_local unsigned ng_bank_wave_model_count=12;
_Thread_local const uint8_t* ng_bank_entropy_banks=NULL;
_Thread_local unsigned ng_bank_entropy_classes=16;
_Thread_local int ng_bank_choice_layout=0;
_Thread_local int ng_bank_multi_row_cache=1;

/* Experimental per-thread benchmark control, not encoded in the format. */

#include "ng_bank_reconstruct.h"

#define NG_BANK_DECLARE(d) int ng_fixed##d##_bank_reconstruct(void* const*,unsigned,const int32_t*,const uint8_t*,uint8_t*); int ng_fixed##d##_bank_wave(void* const*,unsigned,const int32_t*,const uint8_t*,uint8_t*,const uint8_t*,size_t);
NG_BANK_DECLARE(32)
NG_BANK_DECLARE(64)
NG_BANK_DECLARE(128)
NG_BANK_DECLARE(256)

int ng_bank_wave(void* const*s,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,const uint8_t*e,size_t n){
    unsigned saved_flags=ng_bank_trial_flags;
    /* Production integrated wavefront path: packed/fused metadata, cached
     * history, and SIMD entropy. Experimental alternatives are not retained. */
    ng_bank_trial_flags=11|64|128|256|512|2048|4096|8192|32768|65536|
        262144|1048576|16777216|4194304|67108864|134217728|536870912|0x80000000u;
    int saved=ng_bank_scalar_control;ng_bank_scalar_control=20;int rc;
    /* Preserve generic kernels when the NG+X shortcut cannot amortize its
     * dispatch/preparation overhead. Selection depends only on saved choices. */
    {
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

int ng_choice_wave_decode(void* const*s,unsigned count,unsigned mode,const int32_t*w,const uint8_t*c,const uint8_t*banks,uint8_t*out,const uint8_t*header,size_t hn,const uint8_t*payload,size_t pn,unsigned K){
    if(count<12||count>48||count%12||mode>1||K<2||K>32)return -1;
    unsigned oldK=ng_bank_entropy_classes;ng_bank_entropy_classes=K;
    unsigned saved=ng_bank_wave_model_count;const uint8_t*old=ng_bank_entropy_banks;int control=ng_bank_scalar_control,layout=ng_bank_choice_layout;
    ng_bank_wave_model_count=count;ng_bank_entropy_banks=banks;ng_bank_scalar_control=14;ng_bank_choice_layout=1;
    int rc=ng_bank_wave_segments(s,mode,w,c,out,header,hn,payload,pn);
    ng_bank_wave_model_count=saved;ng_bank_entropy_banks=old;ng_bank_scalar_control=control;ng_bank_choice_layout=layout;ng_bank_entropy_classes=oldK;return rc;
}


/* Contexts for an already-filtered byte stream: same causal activity stencil
   as NG, but quantiles fitted on this stream rather than the pre-filter one. */
#include "ng_activity_values.h"
static int ng_activity_dispatch(const uint8_t*s,uint16_t*a,int symbols,uint32_t*h){
    const ng_fixed_kernels*f=ng_fixed_current();
    return f?f->activity_values(s,a,symbols,h):ng_activity_values_impl(s,a,symbols,h);
}

int ng_activity_symbol_values(const uint8_t* symbols,uint16_t* activity) {return ng_activity_dispatch(symbols,activity,1,NULL);}
int ng_activity_symbol_histogram(const uint8_t* symbols,uint16_t* activity,uint32_t* histogram) {
    if(!symbols||!activity||!histogram)return -1;
    return ng_activity_dispatch(symbols,activity,1,histogram);
}
