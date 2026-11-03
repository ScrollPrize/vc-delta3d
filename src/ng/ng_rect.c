/* Rectangular encoder search; uses the SAME frame/header/selector codec as
 * cubic leaves. No NDB2 or alternative rectangular wire representation. */
#include "buffer.h"
#include "ng_frame.h"
#include "delta_filter.h"
#include "residual_bridge.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int vc_ng_rect_encode(const uint8_t *src,unsigned x,unsigned y,unsigned z,
    uint8_t *dst,size_t cap,size_t *written)
{
    if(!src||!dst||!written||!ng_geometry_valid(x,y,z))return -1;
    ng_geometry old=ng_current_geometry;ng_current_geometry=(ng_geometry){(int)x,(int)y,(int)z};
    size_t n=NG_N,blocks=n/64,pn=0;
    uint8_t params[65536],*symbols=malloc(n),*classes=malloc(n),*delta[8]={0};
    uint8_t *choices=calloc(blocks,1),*encoded=NULL;
    uint16_t *base=malloc(n*sizeof(*base));
    double *scores=calloc(blocks*8,sizeof(*scores));
    size_t bytes=0;int rc=-1;
    const int cfg[6]={3,3,1,16,40,128};
    if(!symbols||!classes||!choices||!base||!scores)goto done;
    if(ng_plain_prepare_masked(src,cfg,NULL,params,sizeof(params),&pn,symbols,classes,base))goto done;
    for(size_t i=0;i<n;++i){
        size_t b=pn-32+2*classes[i];
        base[i]=(uint16_t)(base[i]-(int16_t)(params[b]|(unsigned)params[b+1]<<8));
    }
    for(unsigned m=1;m<8;++m){
        delta[m]=malloc(n);if(!delta[m])goto done;
        unsigned axis=m&(~m+1u),parent=m^axis;
        memcpy(delta[m],parent?delta[parent]:src,n);
        vc_delta_filter_u8(delta[m],z,y,x,axis);
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
    if(ng_block_symbols(params,pn,choices,src,base,delta_ptr,symbols,classes))goto done;
    vc_ng_model model={params,pn};vc_ng_choice menu[8];uint16_t thresholds[15];
    for(unsigned m=0;m<8;m++)menu[m]=(vc_ng_choice){m,m?255:0,0,{0,0}};
    for(unsigned j=0;j<15;j++){size_t at=pn-62+2*j;thresholds[j]=(uint16_t)(params[at]|(unsigned)params[at+1]<<8);}
    if(!vc_ng_frame_encode_shape(x,y,z,4,0,&model,1,menu,8,choices,symbols,thresholds,16,classes,&encoded,&bytes)||bytes>cap)goto done;
    memcpy(dst,encoded,bytes);*written=bytes;rc=0;
done:
    free(symbols);free(classes);free(choices);free(base);free(scores);free(encoded);
    for(unsigned m=0;m<8;++m)free(delta[m]);
    ng_current_geometry=old;return rc;
}
