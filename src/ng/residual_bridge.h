#pragma once
#include <stddef.h>
#include <stdint.h>
#include "ng_geometry.h"
#include "ng_symbol_mapping.h"

#ifndef NG_BLOCK_DIM
#define NG_BLOCK_DIM 128
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef int (*ng_symbol_reader)(void*, size_t, unsigned);
typedef void (*ng_tap_capture_fn)(int,int,const double*,const double*,const double*,const int*,int,double);
typedef int (*ng_tap_override_fn)(int,int32_t*,int32_t*);
#ifdef __cplusplus
extern thread_local NG_TLS_CONSTINIT int ng_bank_detail_enabled;
extern thread_local NG_TLS_CONSTINIT unsigned ng_choice_edge;
extern thread_local NG_TLS_CONSTINIT int ng_disable_avx512;
extern thread_local NG_TLS_CONSTINIT int ng_native_fast_control;
extern thread_local NG_TLS_CONSTINIT unsigned ng_simple_activity_kind;
extern thread_local NG_TLS_CONSTINIT const uint16_t* ng_simple_activity_values;
extern thread_local NG_TLS_CONSTINIT const uint8_t* ng_screen_work;
extern thread_local NG_TLS_CONSTINIT const uint8_t* ng_screen_targets;
extern thread_local NG_TLS_CONSTINIT size_t ng_screen_size;
extern thread_local NG_TLS_CONSTINIT int ng_tap_swap_mode, ng_tap_fit_phase;
extern thread_local NG_TLS_CONSTINIT double ng_tap_swap_stats[8];
extern thread_local NG_TLS_CONSTINIT ng_tap_capture_fn ng_tap_capture;
extern thread_local NG_TLS_CONSTINIT ng_tap_override_fn ng_tap_override;
#else
extern _Thread_local int ng_bank_detail_enabled;
extern _Thread_local unsigned ng_choice_edge;
extern _Thread_local int ng_disable_avx512;
extern _Thread_local int ng_native_fast_control;
extern _Thread_local unsigned ng_simple_activity_kind;
extern _Thread_local const uint16_t* ng_simple_activity_values;
extern _Thread_local const uint8_t* ng_screen_work;
extern _Thread_local const uint8_t* ng_screen_targets;
extern _Thread_local size_t ng_screen_size;
extern _Thread_local int ng_tap_swap_mode, ng_tap_fit_phase;
extern _Thread_local double ng_tap_swap_stats[8];
extern _Thread_local ng_tap_capture_fn ng_tap_capture;
extern _Thread_local ng_tap_override_fn ng_tap_override;
#endif
void ng_bank_detail_mark(int);
void ng_bank_detail_get(double*);
void* ng_fit_session_create(int);
void ng_fit_session_destroy(void*);
int ng_feature_prepare_cached(void*,const uint8_t*,const int*,const uint8_t*,int32_t*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*,uint16_t*);
int ng_bank_symbols(unsigned,const uint8_t* const*,const size_t*,const uint16_t* const*,const uint8_t*,const uint8_t*,uint8_t* const*,uint8_t*,uint16_t*);
int ng_bank_selected_symbols(unsigned,const uint8_t* const*,const size_t*,const uint16_t* const*,const uint8_t*,const uint8_t*,uint8_t* const*,uint8_t*,uint16_t*);
/* Shared native NG primitives for the experimental joint-axis model. */
void ng_axis_delta_predict(const uint8_t* x,const uint8_t* y,size_t i,int d,unsigned mask,uint8_t* prediction);
void* ng_causal_create(const uint8_t* params,size_t n);
void* ng_causal_create_model(const uint8_t* params,size_t n);
int ng_context_rans_costs(const uint32_t*,unsigned,double*);
int ng_bank_wave(void* const*,unsigned,const int32_t*,const uint8_t*,uint8_t*,const uint8_t*,size_t);
void ng_causal_destroy(void* state);
int ng_difficulty_prepare(const uint8_t*,const uint8_t*,size_t,const uint16_t*,int,double,uint8_t*,size_t,size_t*,uint16_t*);
int ng_difficulty_prediction(void*,const uint8_t*,size_t,size_t,uint16_t*);
/* Internal row API for Delta-only joint frames: retains native SIMD activity,
 * callers write each error byte before ending the row. No predictor needed. */
int ng_choose_blocks(const uint8_t* const* symbols,int candidates,const uint8_t* classes,int K,uint8_t* choices);
int ng_block_costs(const uint8_t* const* symbols,int candidates,const uint8_t* classes,int K,double* costs);
/* config: r0,r1,r2,classes,tap budget,fit stride. Validated before use. */
int ng_plain_prepare_masked(const uint8_t*,const int*,const uint8_t*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*,uint16_t*);
unsigned delta_prepare_residuals(const uint8_t*, uint8_t* residuals);
void delta_reconstruct_residuals(uint8_t* residuals, unsigned mask);
void delta_apply_mask(uint8_t*, unsigned);
int delta_rans_encode(const uint8_t*,size_t,uint8_t*,size_t,size_t*);
/* Conservative byte lower bound, including the 8 states but excluding tables.
 * Uses the same normalization as the real encoder; zero for an empty stream. */
size_t delta_rans_size_lower_bound(const uint64_t*,size_t,uint16_t*);
int delta_rans_decode(const uint8_t*,size_t,uint8_t*,size_t);
int ng_ex_prepare(const uint8_t*,const int*,int,unsigned,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*,uint16_t*);
int ng_ex_prepare_activity(const uint8_t*,const int*,int,unsigned,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*,uint16_t*,uint16_t*);
int ng_feature_prepare_masked(const uint8_t*,const int*,const uint8_t*,const uint8_t*,int,int32_t*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*,uint16_t*);
int ng_feature_prepare_activity(const uint8_t*,const int*,const uint8_t*,int32_t*,uint8_t*,size_t,size_t*,uint8_t*,uint16_t*,uint32_t*);
double ng_hist_cost(const uint32_t*,int);
int ng_wave_encode(const uint8_t*,size_t,const uint8_t*,const uint8_t*,uint8_t*,size_t,size_t*);
int ng_bank_wave_segments(void* const*,unsigned,const int32_t*,const uint8_t*,uint8_t*,const uint8_t*,size_t,const uint8_t*,size_t);
int ng_choice_wave_decode(void* const*,unsigned,unsigned,const int32_t*,const uint8_t*,const uint8_t*,uint8_t*,const uint8_t*,size_t,const uint8_t*,size_t,unsigned);
int ng_block_decode(const uint8_t*,size_t,const uint8_t*,uint8_t*,int,int);
int ng_entropy_header_size(const uint8_t*,size_t,size_t*);
int ng_bank_empty_model(uint8_t*,size_t,size_t*);
size_t ng_write_freq_table(uint8_t*,const uint16_t*);
int ng_read_freq_table(const uint8_t*,size_t,uint16_t*,size_t*);
int ng_block_symbols(const uint8_t*,size_t,const uint8_t*,const uint8_t*,const uint16_t*,const uint8_t* const*,uint8_t*,uint8_t*,int);
int ng_block_semantic_prepare(const uint8_t*,int,const int*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*,uint8_t*,size_t*);
double ng_hist_cost_logs(const uint32_t*,int,const double*);
int ng_set_isa_limit(int); /* 0 automatic, 256 excludes AVX-512, per thread */
int ng_activity_symbol_values(const uint8_t*,uint16_t*);
int ng_activity_symbol_histogram(const uint8_t*,uint16_t*,uint32_t*);
typedef int (*ng_prediction_reader)(void*,size_t,unsigned,uint16_t);
typedef uint16_t (*ng_prediction_transform)(void*,size_t,uint16_t);
typedef int (*ng_spatial_reader)(void*,size_t,unsigned,unsigned,unsigned,unsigned);
typedef int (*ng_spatial_batch_reader)(void*,unsigned,const size_t*,const unsigned*,unsigned,const int*,const int*,const uint8_t*const*,uint8_t*);
typedef struct ng_route_wave_data {
    unsigned width,lanes;void*columns;uint16_t(*previous)[32];uint8_t*values;
    const size_t*offsets;const int*ys;const int*zs;
    const uint16_t*activity;const uint8_t*class_lut;
    uint8_t*history;size_t history_plane;unsigned history_stride,history_depth,delta;
    uint8_t*errors;const size_t*error_offsets;uint8_t*raw;
    const uint16_t (*block_weights)[4][32];
    void (*compact_step)(void*,const uint16_t(*)[32],uint16_t(*)[32],uint8_t*);
    void*publish_state;uint8_t*output;
    void (*publish)(struct ng_route_wave_data*);
} ng_route_wave_data;
typedef int (*ng_spatial_wave_reader)(void*,ng_route_wave_data*);
typedef struct {ng_spatial_reader read;void*ctx;const uint16_t*thresholds;unsigned K;uint8_t*raw;ng_spatial_batch_reader batch;int row_ready;ng_spatial_wave_reader wave;int compact;int reuse_activity;ng_spatial_wave_reader bulk;int(*barrier)(void*,int);int common_prediction;} ng_route_context;
int ng_context_rans_encode(const uint8_t*,const uint8_t*,const uint8_t*,size_t,unsigned,unsigned,uint16_t*,uint8_t*,size_t,size_t*);
int ng_context_rans_encode_fixed(const uint8_t*,const uint8_t*,const uint8_t*,size_t,unsigned,unsigned,uint16_t*,uint8_t*,size_t,size_t*);
void*ng_context_rans_create(const uint8_t*,size_t,unsigned,unsigned,const uint8_t*,size_t);
int ng_context_rans_read(void*,unsigned,const uint8_t*,uint8_t*,unsigned);
int ng_context_rans_read_wave(void*,unsigned,unsigned,const uint8_t*,uint8_t*);
int ng_context_rans_read_rows(void*,unsigned,const uint8_t*,uint8_t*);
int ng_context_rans_finish(void*);
void ng_context_rans_destroy(void*);
#ifdef __cplusplus
}
#endif
/* Exact causal predictor equivalent to upstream deltaFilter, including edges. */
static inline uint8_t delta_predict_xyz(const uint8_t* v,size_t i,unsigned mask,unsigned x,unsigned y,unsigned z,size_t row,size_t plane) {
    if(!x)mask&=~1u;if(!y)mask&=~2u;if(!z)mask&=~4u;
    switch(mask) {
    case 0:return 0;
    case 1:return v[i-1];
    case 2:return v[i-row];
    case 3:return (uint8_t)(v[i-1]+v[i-row]-v[i-row-1]);
    case 4:return v[i-plane];
    case 5:return (uint8_t)(v[i-1]+v[i-plane]-v[i-plane-1]);
    case 6:return (uint8_t)(v[i-row]+v[i-plane]-v[i-plane-row]);
    default:return (uint8_t)(v[i-1]+v[i-row]+v[i-plane]-v[i-row-1]-v[i-plane-1]-v[i-plane-row]+v[i-plane-row-1]);
    }
}
static inline uint8_t delta_predict_at(const uint8_t* v,size_t i,unsigned mask) {
    return delta_predict_xyz(v,i,mask,(unsigned)(i%NG_X),(unsigned)((i/NG_X)%NG_Y),(unsigned)(i/((size_t)NG_X*NG_Y)),NG_X,(size_t)NG_X*NG_Y);
}
/* Shared exact directional bases. Clamping is before modulo conversion, not
 * clamping the wrapped byte. Missing selected axes retain the legacy fallback. */
typedef struct {int constant,left,clamp,lo,hi;} ng_base_parts;
static inline ng_base_parts ng_directional_parts(const uint8_t*s,size_t i,unsigned m,unsigned x,unsigned y,unsigned z,unsigned d){
    const unsigned masks[4]={3,5,6,7};unsigned mask=m<8?m:masks[m-8];
    unsigned available=(x?1:0)|(y?2:0)|(z?4:0);
    int clamp=m>=8&&(mask&available)==mask;mask&=available;
    size_t plane=(size_t)d*d;int b=(mask&2)?s[i-d]:0,c=(mask&4)?s[i-plane]:0;
    int p=b+c,lo=255,hi=0;
    if(mask&2){if(b<lo)lo=b;if(b>hi)hi=b;}
    if(mask&4){if(c<lo)lo=c;if(c>hi)hi=c;}
    if((mask&3)==3)p-=s[i-d-1];
    if((mask&5)==5)p-=s[i-plane-1];
    if((mask&6)==6)p-=s[i-plane-d];
    if(mask==7)p+=s[i-plane-d-1];
    ng_base_parts r={p,(mask&1)!=0,clamp,lo,hi};return r;
}
static inline uint8_t ng_directional_base(const uint8_t*s,size_t i,unsigned m,unsigned x,unsigned y,unsigned z,unsigned d){
    ng_base_parts r=ng_directional_parts(s,i,m,x,y,z,d);int a=r.left?s[i-1]:0,p=r.constant+a;
    if(r.clamp){if(r.left){if(a<r.lo)r.lo=a;if(a>r.hi)r.hi=a;}p=p<r.lo?r.lo:p>r.hi?r.hi:p;}
    return (uint8_t)p;
}
/* Experimental model-independent context, always from causal source values.
 * 1: full XYZ Lorenzo (existing boundary convention); 2: rounded W/N/B mean.
 * Keep modulo-256 absolute error identical on encoder and decoder. */
static inline uint8_t ng_simple_activity_error(const uint8_t*s,size_t i,unsigned x,unsigned y,unsigned z,unsigned d,unsigned kind){
    unsigned p;
    if(kind==1)p=ng_directional_base(s,i,7,x,y,z,d);
    else {unsigned sum=0,n=0;if(x){sum+=s[i-1];n++;}if(y){sum+=s[i-d];n++;}if(z){sum+=s[i-(size_t)d*d];n++;}p=n?(sum+n/2)/n:128;}
    unsigned e=(uint8_t)(s[i]-p);return (uint8_t)(e<=128?e:256-e);
}
