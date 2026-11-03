#include "residual_bridge.h"
#include "ng_kernel_decode.h"
/* Shared model/history storage for the integrated wavefront decoder.
 * Geometry is scoped by the caller; channels decode independently. */
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


#include "ng_bank_reconstruct.h"

#define NG_BANK_DECLARE(d) int ng_fixed##d##_bank_wave(void* const*,unsigned,const int32_t*,const uint8_t*,uint8_t*,const uint8_t*,size_t,const ng_decode_context*);
NG_BANK_DECLARE(32)
NG_BANK_DECLARE(64)
NG_BANK_DECLARE(128)
NG_BANK_DECLARE(256)

int ng_choice_wave_decode(void* const*s,unsigned count,unsigned mode,const int32_t*w,const uint8_t*c,const uint8_t*banks,uint8_t*out,const uint8_t*header,size_t hn,const uint8_t*payload,size_t pn,unsigned K){
    if(count<12||count>48||count%12||mode>1||K<2||K>32||!header||!payload)return -1;
    ng_decode_context ctx={count,K,banks,payload,pn,0};
    /* Keep the measured NG+X shortcut only where its preparation amortizes. */
    size_t blocks=(size_t)(NG_X/ng_choice_edge)*(NG_Y/ng_choice_edge)*(NG_Z/ng_choice_edge),nx=0;
    for(size_t i=0;i<blocks;i++)nx+=c[i]==13;
    ctx.frequent_x=nx>=blocks/4;
    if(NG_X==NG_Y&&NG_X==NG_Z){
        switch(NG_X){
        case 32:return ng_fixed32_bank_wave(s,mode,w,c,out,header,hn,&ctx);
        case 64:return ng_fixed64_bank_wave(s,mode,w,c,out,header,hn,&ctx);
        case 128:return ng_fixed128_bank_wave(s,mode,w,c,out,header,hn,&ctx);
        case 256:return ng_fixed256_bank_wave(s,mode,w,c,out,header,hn,&ctx);
        }
    }
    return ng_bank_reconstruct_core(s,mode,w,c,out,header,hn,&ctx);
}
