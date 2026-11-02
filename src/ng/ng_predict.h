#ifndef VC_NG_PREDICT_H
#define VC_NG_PREDICT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void vc_ng_fixed_candidates(const uint8_t*,uint8_t*const*,unsigned dimension,unsigned mask);
#ifdef __cplusplus
}
#endif
#endif
