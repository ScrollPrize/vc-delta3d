/* Retained rectangular NDB2 leaf: fixed 4^3 NG/Delta choice, compact selector
 * rANS stream and the original shared native wavefront residual payload. */
#include "buffer.h"
#include "ng_stream.h"
#include "residual_bridge.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int vc_ng_rect_encode(const uint8_t *src,unsigned x,unsigned y,unsigned z,
    uint8_t *dst,size_t cap,size_t *written)
{
    if(!src||!dst||!written||!ng_geometry_valid(x,y,z))return -1;
    ng_geometry old=ng_current_geometry;ng_current_geometry=(ng_geometry){(int)x,(int)y,(int)z};
    int soft=ng_soft_classes;ng_soft_classes=0;
    size_t n=NG_N,blocks=n/64,pn=0,en=0,sn=0;
    uint8_t params[65536],*symbols=malloc(n),*classes=malloc(n),*delta[8]={0};
    uint8_t *choices=calloc(blocks,1),*selector=NULL,*entropy=NULL;
    uint16_t *base=malloc(n*sizeof(*base));
    double *scores=calloc(blocks*8,sizeof(*scores));
    vc_buffer frame={0};int rc=-1;
    const int cfg[6]={3,3,1,16,40,128};
    if(!symbols||!classes||!choices||!base||!scores)goto done;
    if(ng_ex_prepare(src,cfg,0,0,params,sizeof(params),&pn,symbols,classes,base))goto done;
    for(size_t i=0;i<n;++i){
        size_t b=pn-32+2*classes[i];
        base[i]=(uint16_t)(base[i]-(int16_t)(params[b]|(unsigned)params[b+1]<<8));
    }
    for(unsigned m=1;m<8;++m){
        delta[m]=malloc(n);if(!delta[m])goto done;
        unsigned axis=m&(~m+1u),parent=m^axis;
        memcpy(delta[m],parent?delta[parent]:src,n);delta_apply_mask(delta[m],axis);
    }
    for(unsigned m=1;m<8;++m)for(size_t i=0;i<n;++i){
        unsigned v=delta[m][i];delta[m][i]=(uint8_t)((v<<1)^(0u-(v>>7)));
    }
    for(unsigned m=0;m<8;++m){
        const uint8_t *s=m?delta[m]:symbols;
        uint32_t histogram[16][256]={{0}};double costs[16][256];
        for(size_t i=0;i<n;++i)++histogram[classes[i]][s[i]];
        for(unsigned c=0;c<16;++c){
            double total=256;
            for(unsigned j=0;j<256;++j)total+=histogram[c][j];
            for(unsigned j=0;j<256;++j)costs[c][j]=log2(total)-log2((double)histogram[c][j]+1);
        }
        for(unsigned iz=0;iz<z;++iz)for(unsigned iy=0;iy<y;++iy){
            size_t row=((size_t)iz*y+iy)*x,br=((iz/4)*(y/4)+iy/4)*(x/4);
            for(unsigned bx=0;bx<x/4;++bx){
                size_t b=br+bx,i=row+4*bx;double score=scores[8*b+m];
                for(unsigned j=0;j<4;++j)score+=costs[classes[i+j]][s[i+j]];
                scores[8*b+m]=score;
            }
        }
    }
    for(size_t b=0;b<blocks;++b)for(unsigned m=1;m<8;++m)
        if(scores[8*b+m]<scores[8*b+choices[b]])choices[b]=(uint8_t)m;
    const uint8_t *delta_ptr[8];for(unsigned m=0;m<8;++m)delta_ptr[m]=delta[m];
    if(ng_block_symbols(params,pn,choices,src,base,delta_ptr,symbols,classes,4)||
       !vc_ng_stream_encode(choices,blocks,&selector,&sn))goto done;
    entropy=malloc(n*2+65536);if(!entropy)goto done;
    if(ng_wave_encode(params,pn,symbols,classes,entropy,n*2+65536,&en))goto done;
    const uint8_t prefix[6]={'N','D','B','2',4,9};
    if(!vc_buffer_append(&frame,prefix,6)||!vc_buffer_put(&frame,sn,4)||
       !vc_buffer_append(&frame,selector,sn)||!vc_buffer_append(&frame,entropy,en)||frame.size>cap)goto done;
    memcpy(dst,frame.data,frame.size);*written=frame.size;rc=0;
done:
    free(symbols);free(classes);free(choices);free(base);free(scores);free(selector);free(entropy);free(frame.data);
    for(unsigned m=0;m<8;++m)free(delta[m]);
    ng_current_geometry=old;ng_soft_classes=soft;return rc;
}

int vc_ng_rect_decode(const uint8_t *src,size_t n,unsigned x,unsigned y,unsigned z,
    uint8_t *dst,size_t cap)
{
    if(!src||!dst||n<10||!ng_geometry_valid(x,y,z)||cap<(size_t)x*y*z||
       memcmp(src,"NDB2",4)||src[4]!=4||src[5]!=9)return -1;
    size_t sn=(uint32_t)src[6]|(uint32_t)src[7]<<8|(uint32_t)src[8]<<16|(uint32_t)src[9]<<24;
    if(sn>n-10)return -1;
    size_t count=(size_t)(x/4)*(y/4)*(z/4);uint8_t *choices=malloc(count);if(!choices)return -1;
    int rc=-1;ng_geometry old=ng_current_geometry;
    ng_current_geometry=(ng_geometry){(int)x,(int)y,(int)z};
    if(!vc_ng_stream_decode(src+10,sn,choices,count))goto done;
    for(size_t i=0;i<count;++i)if(choices[i]>=8)goto done;
    rc=ng_block_decode(src+10+sn,n-10-sn,choices,dst,0,4);
done:
    free(choices);ng_current_geometry=old;return rc;
}
