#ifndef VC_NG_FRAME_H
#define VC_NG_FRAME_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { const uint8_t *data; size_t size; } vc_ng_model;
typedef struct { unsigned predictor,model,entropy; int32_t weights[2]; } vc_ng_choice;
/* Already fitted models/selected blocks: serialize native NCF1, without search.
 * Result belongs to caller (free). Zero predictor=12; no model=255. */
int vc_ng_frame_encode(unsigned dimension,unsigned edge,unsigned mode,
    const vc_ng_model*,unsigned models,const vc_ng_choice*,unsigned choices,
    const uint8_t *selectors,const uint8_t *symbols,const uint16_t *thresholds,
    unsigned classes,const uint8_t *class_ids,uint8_t **out,size_t *size);
int vc_ng_preset_decode(const uint8_t*,size_t,uint8_t*,size_t);
/* Append the ordinary zero predictor for predetermined empty blocks. */
int vc_ng_frame_encode_zeros(unsigned,const vc_ng_model*,unsigned,const vc_ng_choice*,unsigned,
    const uint8_t*,const uint8_t*,const uint16_t*,const uint8_t*,const uint8_t*,uint8_t**,size_t*);
unsigned vc_ng_frame_dimension(const uint8_t*,size_t);
#ifdef __cplusplus
}
#endif
#endif
