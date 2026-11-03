#include "residual_bridge.h"
#include "ng_kernel_encode.h"
/* Retained encoder entrypoints; shared arithmetic, no duplicate kernels. */
#define NG_DECLARE_FIXED(d) \
int ng_fixed##d##_encode_ex(const uint8_t*,const void*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*); \
int ng_fixed##d##_activity_values(const uint8_t*,uint16_t*,int,uint32_t*);
NG_DECLARE_FIXED(32)
NG_DECLARE_FIXED(64)
NG_DECLARE_FIXED(128)
NG_DECLARE_FIXED(256)
typedef struct {
    int (*encode)(const uint8_t*,const void*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*);
    int (*activity_values)(const uint8_t*,uint16_t*,int,uint32_t*);
} ng_fixed_kernels;
#define NG_FIXED_ENTRY(d) {ng_fixed##d##_encode_ex,ng_fixed##d##_activity_values}
static const ng_fixed_kernels ng_fixed_sets[]={NG_FIXED_ENTRY(64),NG_FIXED_ENTRY(128),NG_FIXED_ENTRY(256),NG_FIXED_ENTRY(32)};
static const ng_fixed_kernels* ng_fixed_current(void) {
    if(NG_X!=NG_Y || NG_X!=NG_Z)return NULL;
    return NG_X==64?&ng_fixed_sets[0]:NG_X==128?&ng_fixed_sets[1]:NG_X==256?&ng_fixed_sets[2]:NG_X==32?&ng_fixed_sets[3]:NULL;
}
static vcn_status ng_encode_core(const uint8_t* src,const vcn_lc_cfg* cf,uint8_t* out,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl) {
    const ng_fixed_kernels* fixed=ng_fixed_current();
    return fixed ? fixed->encode(src,cf,out,cap,n,sy,cl) : vcn_lc_encode_ex(src,cf,out,cap,n,sy,cl);
}

static int ng_config(const int* a,vcn_lc_cfg* cf) {
    if (!a || a[0]<1 || a[0]>3 || a[1]<-1 || a[1]>3 || a[2]<-1 || a[2]>3 ||
        a[3]<2 || a[3]>32 || a[4]<2 || a[4]>160 || a[5]<1 || a[5]>128) return -1;
    *cf=(vcn_lc_cfg){a[0],a[1],a[2],a[3],VCN_LC_TB,a[4],a[5]};return 0;
}

int ng_feature_prepare_masked(const uint8_t* src,const int* config,const uint8_t* feature,const uint8_t* mask,int32_t* weights,
    uint8_t* params,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl,uint16_t* predictions){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src||!feature||!weights)return -1;
    cf.fit_feature=feature;cf.fit_mask=mask;cf.export_feature_weights=weights;cf.export_prediction=predictions;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}
int ng_plain_prepare_masked(const uint8_t*src,const int*config,const uint8_t*mask,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint8_t*cl,uint16_t*predictions){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src)return -1;
    cf.fit_mask=mask;cf.export_prediction=predictions;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}

int ng_feature_prepare_activity(const uint8_t*src,const int*config,const uint8_t*feature,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint16_t*activity,uint32_t*histogram){
    vcn_lc_cfg cf;if(ng_config(config,&cf)||!src||!feature||!weights||!activity||!histogram)return -1;
    cf.fit_feature=feature;cf.export_feature_weights=weights;
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
    cf.fit_feature=feature;cf.export_feature_weights=weights;cf.export_prediction=predictions;cf.fit_session=s;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
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
    uint8_t* out,size_t cap,size_t* size) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    vcn_lc_cfg cf={0};cf.r0=p->t.r[0];cf.r1=p->t.r[1];cf.r2=p->t.r[2];cf.K=p->K;cf.tb=p->tb;
    cf.import_sym=sy;cf.import_cls=cl;cf.import_params=params;cf.import_params_n=n;
    int rc=ng_encode_core(NULL,&cf,out,cap,size,NULL,NULL);free(p);return rc;
}

int ng_wave_encode(const uint8_t* p,size_t n,const uint8_t* sy,const uint8_t* cl,uint8_t* out,size_t cap,size_t* size) {
    return ng_compact_encode_order(p,n,sy,cl,out,cap,size);
}

/* Encoder already knows source samples and fitted NG tap predictions. Only
   activity depends on previous symbols; reuse original SIMD class/symbol math. */
static int ng_selected_symbols_impl(const uint8_t* params,size_t n,const uint8_t* selectors,const uint8_t* src,
    const uint16_t* base,const uint8_t* const* delta,uint8_t* sy,uint8_t* cl,int B) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    uint8_t* e=calloc(VCN_LC_NB,1);if(!e){free(p);return -1;}
    uint16_t Q[VCN_LC_D];
    int block_shift=B==4?2:B==8?3:4,nb=VCN_LC_D>>block_shift,ny=VCN_LC_Y>>block_shift;
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++) {
        size_t ri=VCN_LC_IDX(z,y,0),o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        vcn_lc_row_begin(e,z,y,0);
        memcpy(Q,base+o,sizeof(Q));VCN_LC_CLS(e,ri,p,Q,cl+o);
        VCN_LC_SYMS(Q,src+o,sy+o,e+ri);
        const uint8_t* row_choices=selectors+(((z>>block_shift)*ny+(y>>block_shift))*nb);
        for(int x=0;x<VCN_LC_D;x++) {
            unsigned sel=row_choices[x>>block_shift];
            if(sel){sy[o+x]=delta[sel][o+x];e[ri+x]=(uint8_t)((sy[o+x]+1u)>>1);}
        }
        vcn_lc_row_end(e,z,y);
    }
    free(e);free(p);return 0;
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
    if(NG_X==NG_Y&&NG_X==NG_Z){
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
    return ng_selected_symbols_impl(p,n,s,src,base,delta,sy,cl,B);
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
