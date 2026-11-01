#pragma once
/* Shared encoder/decoder feature definition. Every address is in the already
 * transmitted even-coordinate coarse grid; no fine target is consulted. */
static inline int ng_coarse_site(unsigned x,unsigned y,unsigned z){return !((x|y|z)&1u);}
static inline uint8_t ng_coarse_input(const uint8_t*src,unsigned d,unsigned x,unsigned y,unsigned z,unsigned k,unsigned count){
    unsigned c[3]={x,y,z},step=count==54&&k>=27?4:2;int q[3];if(count==54)k%=27;
    for(unsigned a=0;a<3;a++){
        if(count==8)q[a]=(int)(c[a]&~1u)+2*(int)((k>>a)&1u);
        else{unsigned digit=k%3;k/=3;q[a]=(int)(step*((c[a]+step/2)/step))+(int)step*((int)digit-1);}
        if(q[a]<0)q[a]=0;if(q[a]>(int)(d-step))q[a]=(int)(d-step);
    }
    return src[((size_t)q[2]*d+q[1])*d+q[0]];
}
