#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Independent rectangular uint8 leaf, contiguous X-fastest Z/Y/X layout.
 * Dimensions: multiples of32,32<=X<=1024,Y*Z<=65536. No padding.
 * Experimental NDL1 wrapper: magic, X/Y/Z uint32 LE, then baseline NDB2.
 * Every leaf is self-describing; the separate NGC1 indexed container API is
 * declared in vc_delta3d/indexed.hpp.
 * Returns0 on success,-1 on invalid input, insufficient capacity or failure.
 * Calls are thread-safe; input/output must not overlap. */
size_t ng_leaf_encode_bound(uint32_t x,uint32_t y,uint32_t z);
int ng_leaf_encode(const uint8_t* src,size_t src_n,uint32_t x,uint32_t y,uint32_t z,
                   const int* cfg,uint8_t* dst,size_t cap,size_t* written);
int ng_leaf_decode(const uint8_t* src,size_t src_n,uint8_t* dst,size_t cap,int scalar);
#ifdef __cplusplus
}
#endif
