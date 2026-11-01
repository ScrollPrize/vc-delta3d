/* Compile the SAME kernels with constant geometry, selected by CMake. No copied
 * predictor, entropy, scheduling or reconstruction implementation. */
#ifndef NG_FIXED_GEOMETRY
#error "Set NG_FIXED_GEOMETRY for this kernel instantiation"
#endif
#define NG_FIXED_NAME_IMPL(d,s) ng_fixed##d##s
#define NG_FIXED_NAME_EXPAND(d,s) NG_FIXED_NAME_IMPL(d,s)
#define NG_FIXED_NAME(s) NG_FIXED_NAME_EXPAND(NG_FIXED_GEOMETRY,s)
#include "ng_geometry.h"
#include "residual_bridge.h"
extern void ng_profile_tick(int);
#define VCN_LC_TICK(k) ng_profile_tick(k)
#include "volcodec_ng_lossless.h"
#include "ng_activity_values.h"
int NG_FIXED_NAME(_activity_values)(const uint8_t*s,uint16_t*a,int symbols,uint32_t*h){return ng_activity_values_impl(s,a,symbols,h);}
#include "ng_sparse_bank.h"
void*NG_FIXED_NAME(_sparse_create)(const uint8_t*s,const uint8_t*p,size_t n,const uint8_t*c,unsigned stride){return ng_sparse_create_impl(s,p,n,c,stride);}
void NG_FIXED_NAME(_sparse_destroy)(void*s){ng_sparse_destroy_impl(s);}
int NG_FIXED_NAME(_sparse_fit)(void*s,unsigned m,const uint8_t*c,uint8_t*p,size_t cap,size_t*n,int32_t*w){return ng_sparse_fit_impl(s,m,c,p,cap,n,w);}
int NG_FIXED_NAME(_sparse_block)(void*s,unsigned m,unsigned x,unsigned y,unsigned z,uint8_t*out){return ng_sparse_block_impl(s,m,x,y,z,out);}
int NG_FIXED_NAME(_sparse_emit)(void*s,const uint8_t*c,uint8_t*y,uint16_t*a){return ng_sparse_emit_impl(s,c,y,a);}

#include "ng_bank_symbols.h"
int NG_FIXED_NAME(_bank_symbols)(unsigned c,const uint8_t*const*p,const size_t*n,const uint16_t*const*b,const uint8_t*s,const uint8_t*x,uint8_t*const*v,uint8_t*y,uint16_t*a){
    return ng_bank_symbols_impl(c,p,n,b,s,x,v,y,a);
}
int NG_FIXED_NAME(_shared_symbols)(unsigned c,const uint8_t*const*p,const size_t*n,const uint16_t*const*b,const uint8_t*s,const uint8_t*x,uint8_t*const*v,uint8_t*y,uint16_t*a,const int32_t*w,uint16_t*q){
    return ng_bank_symbols_core(c,p,n,b,s,x,v,y,a,w,q);
}
#include "ng_causal_state.h"
#include "ng_bank_reconstruct.h"
#include "ng_activity_decode.h"
int NG_FIXED_NAME(_bank_route)(void* const*s,unsigned count,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,const uint16_t*t,unsigned K,ng_symbol_reader reader,void*ctx,uint8_t*raw){
    return ng_bank_route_impl(s,count,mode,w,c,out,t,K,reader,ctx,raw);
}
int NG_FIXED_NAME(_activity_gather)(const uint16_t*t,int K,const uint8_t*const*s,const size_t*n,size_t*p,uint8_t*out){
    return ng_activity_decode_impl(t,K,0,NULL,NULL,s,n,p,out);
}
int NG_FIXED_NAME(_bank_reconstruct)(void* const* states,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out){
    return ng_bank_reconstruct_impl(states,mode,w,c,out);
}
int NG_FIXED_NAME(_bank_wave)(void* const* states,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,const uint8_t*e,size_t n){
    return ng_bank_reconstruct_core(states,mode,w,c,out,e,n);
}

void NG_FIXED_NAME(_activity_begin)(void* ptr,int z,int y,const uint8_t**classes,uint8_t**errors) {
    ng_activity_row_begin_impl(ptr,z,y,classes,errors);
}
void NG_FIXED_NAME(_activity_end)(void* ptr,int z,int y) {
    ng_activity_row_end_impl(ptr,z,y);
}

int NG_FIXED_NAME(_native_encode)(const uint8_t* src,uint8_t* dst,size_t cap,size_t* n) {
    return vcn_encode_lossless_chunk(src,dst,cap,n);
}
int NG_FIXED_NAME(_native_decode)(const uint8_t* src,size_t n,uint8_t* dst,size_t cap) {
    return vcn_decode(src,n,dst,cap);
}

int NG_FIXED_NAME(_encode_ex)(const uint8_t* src,const void* cfg,uint8_t* out,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl) {
    return vcn_lc_encode_ex(src,(const vcn_lc_cfg*)cfg,out,cap,n,sy,cl);
}
int NG_FIXED_NAME(_decode_core)(const uint8_t* p,size_t n,uint8_t* dst,ng_symbol_reader reader,void* ctx,
                          const uint8_t* selectors,int scalar,int B,const uint8_t* sy,const uint8_t* contexts) {
    return vcn_lc_decode_core(p,n,dst,reader,ctx,selectors,scalar,B,sy,contexts);
}
