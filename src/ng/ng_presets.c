/* Public preset dispatch. Save all scoped fitter settings for nested callers. */
#include "residual_bridge.h"
int vc_ng_fast_encode(const uint8_t*,unsigned,uint8_t*,size_t,size_t*);
int vc_ng_medium_encode(const uint8_t*,unsigned,uint8_t*,size_t,size_t*);
int vc_ng_slow_encode(const uint8_t*,unsigned,unsigned,uint8_t*,size_t,size_t*);
int vc_ng_preset_encode(const uint8_t *s,unsigned d,unsigned preset,uint8_t *out,size_t cap,size_t *n)
{
    if(preset>3)return -1;
    double saved=ng_fit_intercept_weight;
    ng_fit_intercept_weight=preset>=2?100.:0.;
    int rc=preset==0?vc_ng_fast_encode(s,d,out,cap,n):preset==1?vc_ng_medium_encode(s,d,out,cap,n):
        vc_ng_slow_encode(s,d,preset-2,out,cap,n);
    ng_fit_intercept_weight=saved;return rc;
}
