#include "buffer.h"
#include "ng/header.c" /* Schema access for explicit wire-mode regression tests. */
#include "ng/residual_bridge.h"
#include <assert.h>
#include <stdio.h>

int main(void){
    const unsigned edges[]={2,4,8,16,32};
    unsigned tests=0;
    for(unsigned nm=0;nm<=4;nm++)for(unsigned e=0;e<5;e++)for(unsigned entropy=0;entropy<2;entropy++){
        vc_buffer raw={0};unsigned nc=nm+1;
        unsigned fields[7]={32,64,32,1|(edges[e]<<8),nm,nc,2};
        for(unsigned j=0;j<7;j++)assert(vc_buffer_put(&raw,fields[j],4));
        uint8_t params[4096];size_t pn=0;assert(!ng_bank_empty_model(params,sizeof(params),&pn));
        for(unsigned m=0;m<nm;m++){
            params[6]=(uint8_t)(m+1);params[7]=255;
            assert(vc_buffer_append(&raw,params,pn));
        }
        for(unsigned m=0;m<nc;m++){
            assert(vc_buffer_put(&raw,m%12,1)&&vc_buffer_put(&raw,m<nm?m:255,1)&&vc_buffer_put(&raw,0,2));
            assert(vc_buffer_put(&raw,(uint32_t)(-256*(int)m),4)&&vc_buffer_put(&raw,256*m,4));
        }
        if(entropy){
            params[6]=params[7]=0;assert(vc_buffer_append(&raw,params,pn));
            for(unsigned k=0;k<16;k++){
                uint16_t freq[256]={0};uint8_t table[514];
                freq[k]=1024;freq[255-k]=3072;
                size_t n=ng_write_freq_table(table,freq);assert(vc_buffer_append(&raw,table,n));
            }
        }
        uint8_t*z=NULL,*p=NULL;size_t zn=0,n=0;
        assert(vc_header_encode(raw.data,raw.size,&z,&zn));
        assert(vc_header_decode(z,zn,&p,&n)&&n==raw.size&&!memcmp(p,raw.data,n));free(p);
        /* Exercise the optional transmitted-table mode even though the default
         * encoder only selects fixed tables or raw labels. */
        arrays a={0};walk w={{z,zn,0,1},{0},&a,1,0};assert(schema(&w,1));
        size_t count=0;for(unsigned k=0;k<a.count;k++)count+=vc_varget(&w.input,UINT32_MAX);
        size_t mode_at=w.input.i;unsigned tag=vc_get(&w.input,1);size_t bytes=vc_varget(&w.input,UINT32_MAX);
        const uint8_t *payload=vc_take(&w.input,bytes);assert(w.input.ok);
        uint8_t *labels=malloc(count?count:1);assert(labels);
        if((tag&3)==1)assert(vc_ng_stream_fixed_decode(payload,bytes,tag>>2,labels,count));
        else {assert(bytes==count);memcpy(labels,payload,count);}
        if(count){
            uint8_t *dynamic=NULL;size_t dn=0;assert(vc_ng_stream_encode(labels,count,&dynamic,&dn));
            vc_buffer alt={0};assert(vc_buffer_append(&alt,z,mode_at)&&vc_buffer_put(&alt,(tag&4)|2,1));
            assert(vc_varput(&alt,(uint32_t)dn)&&vc_buffer_append(&alt,dynamic,dn)&&vc_buffer_append(&alt,z+w.input.i,zn-w.input.i));
            p=NULL;assert(vc_header_decode(alt.data,alt.size,&p,&n)&&n==raw.size&&!memcmp(p,raw.data,n));
            free(p);free(dynamic);free(alt.data);
        }
        free(labels);free(a.words.data);free(w.output.data);
        for(size_t cut=0;cut<zn;cut++){p=NULL;assert(!vc_header_decode(z,cut,&p,&n));free(p);}
        /* Mutated headers may be valid different headers; success must always
         * be safe and canonical enough to encode again. */
        for(size_t i=0;i<zn;i++){
            z[i]^=0x80;p=NULL;
            if(vc_header_decode(z,zn,&p,&n)){uint8_t*q=NULL;size_t qn=0;assert(vc_header_encode(p,n,&q,&qn));free(q);}
            free(p);z[i]^=0x80;
        }
        free(z);free(raw.data);tests++;
    }
    printf("%u header shapes/menus, all truncations and mutations passed\n",tests);return 0;
}
