#include "legacy.h"
#include "delta_filter.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
int main(void)
{
    const uint32_t shape[3] = {7,11,19};
    const uint8_t magic[4] = {'D','3','D','1'};
    const unsigned bins[] = {1,3,100,255};
    for (unsigned elem=1;elem<=2;++elem) for (unsigned channels=0;channels<=3;++channels)
    for (unsigned layout=0;layout<2;++layout) for (unsigned q=0;q<4;++q) {
        size_t size=(size_t)7*11*19*elem*(channels?channels:1);
        uint8_t *src=malloc(size),*expected=malloc(size),*out=malloc(size),*frame;
        assert(src && expected && out);
        for (size_t i=0;i<size;++i) src[i]=(uint8_t)(i*17+i/67);
        memcpy(expected,src,size);
        if (elem==1) vc_delta_quantize_u8(expected,size,bins[q]);
        else vc_delta_quantize_u16((uint16_t*)expected,size/2,bins[q]);
        size_t bytes; vc_legacy_info info;
        assert(vc_legacy_encode(src,size,shape,elem,bins[q],channels,layout,&frame,&bytes));
        assert(vc_legacy_probe(frame,bytes,magic,&info));
        assert(info.bytes==size && info.channels==(channels?channels:1));
        assert(vc_legacy_decode(frame,bytes,magic,out,size));
        assert(memcmp(out,expected,size)==0);
        assert(!vc_legacy_decode(frame,bytes-1,magic,out,size));
        assert(!vc_legacy_decode(frame,bytes,magic,out,size-1));
        free(frame);free(out);free(expected);free(src);
    }
    uint8_t invalid[24]={'D','3','D','1',1,1,1,2,1,0,0,0,1,0,0,0,1};
    vc_legacy_info info;
    assert(!vc_legacy_probe(invalid,sizeof(invalid),magic,&info)); /* retired LPC */
    invalid[7]=1; memset(invalid+8,255,12);
    assert(!vc_legacy_probe(invalid,sizeof(invalid),magic,&info)); /* overflow */
    puts("legacy C scalar/channel/fallback tests passed");
    return 0;
}
