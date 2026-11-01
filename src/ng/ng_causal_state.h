#pragma once
/* Shared runtime/fixed-geometry activity access. Include after the native
 * kernels: parameters and padding are exactly the original NG representation. */
typedef struct {
    vcn_lc_par* p;uint8_t *b,*e,*classes;uint16_t* pre;
    vcn_lc_plist pl[2];size_t next;int pending;
    uint16_t* feature_activity;
} ng_causal_state;

static void ng_activity_row_begin_impl(void* ptr,int z,int y,const uint8_t** classes,uint8_t** errors) {
    ng_causal_state* s=ptr;
    vcn_lc_row_begin(s->e,z,y,0);
    size_t ri=VCN_LC_IDX(z,y,0);
    /* Class selection uses errors only. The unused class-bias addition still
     * goes through the same SIMD kernel, on initialized scratch storage. */
    memset(s->pre,0,2*VCN_LC_D);
    VCN_LC_CLS(s->e,ri,s->p,s->pre,s->classes);
    *classes=s->classes;*errors=s->e+ri;
}
static void ng_activity_row_end_impl(void* ptr,int z,int y) {
    ng_causal_state* s=ptr;vcn_lc_row_end(s->e,z,y);
}
