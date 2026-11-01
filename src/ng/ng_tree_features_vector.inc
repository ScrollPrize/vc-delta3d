    NG_FATTR
    void NG_FNAME(unsigned count,unsigned x,const int*ys,const int*zs,const uint8_t*const*n,int*f)const{
        const NG_FV zero=NG_FOP0(),missing=NG_FOP1(128);
        for(unsigned j=0;j<count;j+=NG_FSTEP){
            NG_FV my=NG_FOP2(NG_FOP3((const NG_FV*)(ys+j)),zero);
            NG_FV mz=NG_FOP2(NG_FOP3((const NG_FV*)(zs+j)),zero);
            NG_FV w=x?NG_FOP4(NG_FOP5((const __m128i*)(n[0]+j))):missing;
            NG_FV north=NG_FOP6(missing,NG_FOP4(NG_FOP5((const __m128i*)(n[2]+j))),my);
            NG_FV back=NG_FOP6(missing,NG_FOP4(NG_FOP5((const __m128i*)(n[3]+j))),mz);
            NG_FV ww=x>1?NG_FOP4(NG_FOP5((const __m128i*)(n[1]+j))):missing;
#define NG_FNEIGHBOR(K) NG_FOP4(NG_FOP5((const __m128i*)(n[K]+j)))
#include "ng_tree_feature_math.inc"
#undef NG_FNEIGHBOR
        }
    }
#undef NG_FATTR
#undef NG_FNAME
#undef NG_FV
#undef NG_FSTEP
#undef NG_FOP0
#undef NG_FOP1
#undef NG_FOP2
#undef NG_FOP3
#undef NG_FOP4
#undef NG_FOP5
#undef NG_FOP6
#undef NG_FOP7
#undef NG_FOP8
#undef NG_FOP9
#undef NG_FOP10
#undef NG_FOP11
#undef NG_FOP12
#undef NG_FOP13
#undef NG_FOP14
