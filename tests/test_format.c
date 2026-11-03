/* Exercise format capabilities independently of the presets' current choices.
 * Constant fitted models make the expected prediction explicit; residuals and
 * causal activity still use the real wire mapping and entropy schedule. */
#include "ng_frame.h"
#include <vc_delta3d/codec.h>
#include <vc_delta3d/indexed.h>
#include "residual_bridge.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void run(unsigned edge,unsigned nm,unsigned banks,unsigned K,unsigned mode,unsigned X,int varied){
    const unsigned Y=32,Z=32;size_t n=(size_t)X*Y*Z,blocks=n/(edge*edge*edge);
    ng_current_geometry=(ng_geometry){X,Y,Z};ng_choice_edge=edge;
    uint8_t *src=malloc(n),*out=malloc(n),*sy=malloc(n),*cl=malloc(n),*sel=malloc(blocks);
    uint16_t *activity=malloc(n*sizeof(*activity)),thresholds[31];
    assert(src&&out&&sy&&cl&&sel&&activity);
    uint8_t parameters[4][4096];vc_ng_model models[4];vc_ng_choice menu[60];
    unsigned nc=12+12*nm;
    for(unsigned m=0;m<nm;m++){
        size_t bytes;assert(!ng_bank_empty_model(parameters[m],4096,&bytes));
        unsigned nt=(bytes-6-62)/4-1;
        for(unsigned b=0;b<2;b++){
            unsigned pos=6+2*(b*(nt+1)+nt),bias=256*(m+1)+128;
            parameters[m][pos]=bias;parameters[m][pos+1]=bias>>8;
        }
        models[m]=(vc_ng_model){parameters[m],bytes};
    }
    for(unsigned c=0;c<nc;c++)menu[c]=(vc_ng_choice){c%12,c<12?255:c/12-1,c%banks,{192,192}};
    if(varied)for(unsigned c=12;c<nc;c++){
        menu[c].weights[0]=192+3*(c%12);menu[c].weights[1]=160+5*(c%12);
    }
    /* Interleave fixed and fitted choices even when there are few big blocks. */
    for(size_t i=0;i<blocks;i++)sel[i]=(uint8_t)((13*i+12)%nc);
    for(unsigned z=0;z<Z;z++)for(unsigned y=0;y<Y;y++)for(unsigned x=0;x<X;x++){
        size_t i=((size_t)z*Y+y)*X+x;
        src[i]=(uint8_t)(x*13+y*7+z*11+(x*y)%19);
    }
    for(unsigned z=0;z<Z;z++)for(unsigned y=0;y<Y;y++)for(unsigned x=0;x<X;x++){
        size_t i=((size_t)z*Y+y)*X+x,b=((size_t)(z/edge)*(Y/edge)+y/edge)*(X/edge)+x/edge;
        vc_ng_choice*c=menu+sel[b];
        uint8_t base=ng_directional_base_xyz(src,i,c->predictor,x,y,z,X,(size_t)X*Y);
        unsigned q=c->model==255?256u*base:
            256u*(c->model+1)+128+(c->predictor?(mode?c->weights[z==0]:256)*base:0);
        q&=65535;
        uint8_t r=(uint8_t)(src[i]-((uint16_t)(q+128)>>8));
        if(q&128)r=(uint8_t)-r;
        sy[i]=(uint8_t)((r<<1)^(r&128?255:0));
    }
    assert(!ng_activity_symbol_values(sy,activity));
    for(unsigned k=0;k<K-1;k++)thresholds[k]=(uint16_t)(k*2560/K);
    for(size_t i=0;i<n;i++){cl[i]=0;for(unsigned k=0;k<K-1;k++)cl[i]+=activity[i]>thresholds[k];}
    uint8_t*frame;size_t bytes;
    assert(vc_ng_frame_encode_shape(X,Y,Z,edge,mode,models,nm,menu,nc,sel,sy,thresholds,K,cl,&frame,&bytes));
    assert(!vc_ng_preset_decode(frame,bytes,out,n));
    assert(!memcmp(src,out,n));
    if(X!=Y){
        /* Reader supports rectangular leaves independently of encoder policy. */
        size_t total=16+bytes-4;uint8_t *indexed=calloc(total,1);assert(indexed);
        memcpy(indexed,"NGC2",4);indexed[4]=16;indexed[7]=Z/32-1;
        indexed[8]=Y/32-1;indexed[9]=X/32-1;indexed[11]=8;
        for(unsigned k=0;k<4;k++)indexed[12+k]=(uint8_t)((bytes-4)>>(8*k));
        memcpy(indexed+16,frame+4,bytes-4);
        assert(vc_decompress(indexed,total,out,n)&&!memcmp(src,out,n));
        vc_chunk_index index;assert(vc_index_read(indexed,16,total,&index));
        assert(index.leaf_count==1);
        assert(vc_indexed_leaf_decode(indexed+16,bytes-4,index.leaves,out,n)&&!memcmp(src,out,n));
        vc_index_free(&index);free(indexed);
    }
    free(frame);free(activity);free(sel);free(cl);free(sy);free(out);free(src);
}
int main(void){
    const unsigned edges[]={2,4,8,16,32},classes[]={2,16,32};
    unsigned cases=0;
    for(unsigned isa=0;isa<2;isa++){
        assert(!ng_set_isa_limit(isa?256:0));
        for(unsigned e=0;e<5;e++)for(unsigned nm=0;nm<=4;nm++)
            for(unsigned b=1;b<=3;b+=2)for(unsigned k=0;k<3;k++)for(unsigned mode=0;mode<2;mode++){
                run(edges[e],nm,b,classes[k],mode,32,0);cases++;
            }
        run(2,4,3,16,1,64,0);cases++;
        run(32,4,3,32,1,128,1);cases++;
        for(unsigned e=0;e<5;e++)for(unsigned nm=1;nm<=4;nm++){
            run(edges[e],nm,1,16,1,32,1);cases++;
            run(edges[e],nm,3,16,1,64,1);cases++;
        }
    }
    assert(!ng_set_isa_limit(0));
    printf("%u generic-format cases passed\n",cases);
    return 0;
}
