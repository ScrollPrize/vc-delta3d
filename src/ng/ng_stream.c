/* Compact frequency tables around the shared legacy eight-lane rANS payload. */
#include "ng_stream.h"
#include "residual_bridge.h"
#include "rans.h"
#include <stdlib.h>
#include <string.h>
#include "header_tables.h"

/* Same bulk coder; fixed tables are decoder constants, never transmitted. */
int vc_ng_stream_fixed_encode(const uint8_t *src,size_t count,unsigned which,uint8_t **out,size_t *size)
{
    if(!out||!size)return 0;*out=NULL;*size=0;if(which>1)return 0;
    uint32_t freq[256];for(unsigned i=0;i<256;i++)freq[i]=header_freq[which][i];
    uint8_t *raw=NULL;size_t bytes=0;
    if(!vc_rans_encode_frequencies(src,count,0,freq,&raw,&bytes))return 0;
    memmove(raw,raw+512,bytes-512);*out=raw;*size=bytes-512;return 1;
}
int vc_ng_stream_fixed_decode(const uint8_t *src,size_t bytes,unsigned which,uint8_t *out,size_t count)
{
    if(which>1||bytes>SIZE_MAX-512||(!src&&bytes))return 0;
    uint8_t *raw=malloc(bytes+512);if(!raw)return 0;
    for(unsigned i=0;i<256;i++){raw[2*i]=(uint8_t)header_freq[which][i];raw[2*i+1]=(uint8_t)(header_freq[which][i]>>8);}
    if(bytes)memcpy(raw+512,src,bytes);
    int ok=vc_rans_decode(raw,bytes+512,out,count);free(raw);return ok;
}

int vc_ng_stream_encode(const uint8_t *src,size_t count,uint8_t **out,size_t *size)
{
    *out=NULL;*size=0;
    if(!count)return 1;
    uint8_t *raw=NULL;size_t bytes=0;
    if(!vc_rans_encode(src,count,0,&raw,&bytes)||bytes<512){free(raw);return 0;}
    uint16_t freq[256];uint8_t table[514];
    for(unsigned i=0;i<256;++i)freq[i]=(uint16_t)(raw[2*i]|(unsigned)raw[2*i+1]<<8);
    size_t tn=ng_write_freq_table(table,freq);
    *size=tn+bytes-512;*out=malloc(*size);
    if(!*out){free(raw);return 0;}
    memcpy(*out,table,tn);memcpy(*out+tn,raw+512,bytes-512);
    free(raw);return 1;
}

int vc_ng_stream_decode(const uint8_t *src,size_t bytes,uint8_t *out,size_t count)
{
    uint16_t freq[256];size_t used=0;
    if(ng_read_freq_table(src,bytes,freq,&used)||used>bytes||bytes-used>SIZE_MAX-512)return 0;
    size_t size=512+bytes-used;uint8_t *raw=malloc(size);if(!raw)return 0;
    for(unsigned i=0;i<256;++i){raw[2*i]=(uint8_t)freq[i];raw[2*i+1]=(uint8_t)(freq[i]>>8);}
    memcpy(raw+512,src+used,bytes-used);
    int ok=vc_rans_decode(raw,size,out,count);free(raw);return ok;
}
