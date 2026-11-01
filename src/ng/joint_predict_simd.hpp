#pragma once
// Resolve selector geometry once per four X positions, not once per voxel.
// The actual Delta math remains in the shared reference helper.
struct JointPreparedPredictor {
    int code[4][32]{},offset[32]{};unsigned reference[32]{};
    void prepare(int lane,int pred,int xb,int y,int z,int d,unsigned ref){
        reference[lane]=ref;
        if(pred<=7){for(int k=0;k<4;k++){int m=pred;if(xb+k==0)m&=~1;if(!y)m&=~2;if(!z)m&=~4;code[k][lane]=m;}}
        else if(pred==15){for(int k=0;k<4;k++)code[k][lane]=9;}
        else {
            static constexpr int dx[]={2,0,0,1,1,0,1},dy[]={0,2,0,1,0,1,1},dz[]={0,0,2,0,1,1,1};
            int k=pred-8;offset[lane]=dx[k]+d*dy[k]+d*d*dz[k];
            for(int j=0;j<4;j++)code[j][lane]=(xb+j>=dx[k]&&y>=dy[k]&&z>=dz[k])?8:0;
        }
    }
    JOINT_INLINE Pair predict(int lane,int xx,const uint8_t*x,const uint8_t*y,size_t i,int d)const{
        int c=code[xx&3][lane];
        if(c<=7)return {delta_predict_xyz(x,i,c,1,1,1,d,size_t(d)*d),delta_predict_xyz(y,i,c,1,1,1,d,size_t(d)*d)};
        if(c==8)return {x[i-offset[lane]],y[i-offset[lane]]};
        return {int(reference[lane]>>8),int(reference[lane]&255)};
    }
    JOINT_INLINE Pair predict_packed(int lane,int xx,const uint8_t*x,const uint8_t*y,size_t i,int d)const{
        int c=code[xx&3][lane];
        if(c>7)return predict(lane,xx,x,y,i,d);
        auto load=[&](size_t off){return uint32_t(x[i-off])|(uint32_t(y[i-off])<<16);};
        // Each lane stays in [32768-765,32768+1020], so no carry/borrow
        // crosses the 16-bit boundary. Extracting low bytes is exact modulo256.
        uint32_t v=delta_stencil(c,d,size_t(d)*d,load,0x80008000u);
        return {int(v&255),int((v>>16)&255)};
    }
};
// Exact masked byte gathers. Each four-byte load is aligned DOWN within the
// uint8 volume, including its final voxel; no padding or out-of-range reads.
struct JointPrediction {
    alignas(32) int index[7][32]{},weight[7][32]{},reference[2][32]{},medoid[32]{};
    void set(int lane,size_t i,int d,int pred,unsigned ref){
        int x=i%d,y=i/d%d,z=i/size_t(d)/d;
        medoid[lane]=pred==15;reference[0][lane]=ref>>8;reference[1][lane]=ref&255;
        for(int k=0;k<7;k++){index[k][lane]=0;weight[k][lane]=0;}
        if(pred<=7){
            int mask=pred;if(!x)mask&=~1;if(!y)mask&=~2;if(!z)mask&=~4;
            for(int k=1;k<8;k++)if((k&mask)==k){
                index[k-1][lane]=int(i)-(k&1?1:0)-(k&2?d:0)-(k&4?d*d:0);
                weight[k-1][lane]=(k==3||k==5||k==6)?-1:1;
            }
        }else if(pred<15){
            static const int dx[]={2,0,0,1,1,0,1},dy[]={0,2,0,1,0,1,1},dz[]={0,0,2,0,1,1,1};
            int k=pred-8;if(x>=dx[k]&&y>=dy[k]&&z>=dz[k]){index[0][lane]=int(i)-dx[k]-d*dy[k]-d*d*dz[k];weight[0][lane]=1;}
        }
    }
};
#if JOINT_AVX2
__attribute__((target("avx2"))) static void joint_predict_avx(const JointPrediction&p,const uint8_t*x,const uint8_t*y,JointBatch&q,int n){
    const uint8_t*src[]={x,y};int*dst[]={q.px,q.py};auto byte=_mm256_set1_epi32(255),zero=_mm256_setzero_si256();
    for(int i=0;i<n;i+=8)for(int c=0;c<2;c++){
        auto sum=zero;
        for(int k=0;k<7;k++){
            auto idx=_mm256_load_si256((const __m256i*)(p.index[k]+i)),w=_mm256_load_si256((const __m256i*)(p.weight[k]+i));
            auto aligned=_mm256_and_si256(idx,_mm256_set1_epi32(-4));
            auto active=_mm256_cmpgt_epi32(_mm256_abs_epi32(w),zero);
            auto v=_mm256_mask_i32gather_epi32(zero,(const int*)src[c],aligned,active,1);
            v=_mm256_and_si256(_mm256_srlv_epi32(v,_mm256_slli_epi32(_mm256_and_si256(idx,_mm256_set1_epi32(3)),3)),byte);
            sum=_mm256_add_epi32(sum,_mm256_sign_epi32(v,w));
        }
        auto m=_mm256_cmpgt_epi32(_mm256_load_si256((const __m256i*)(p.medoid+i)),zero);
        sum=_mm256_blendv_epi8(_mm256_and_si256(sum,byte),_mm256_load_si256((const __m256i*)(p.reference[c]+i)),m);
        _mm256_store_si256((__m256i*)(dst[c]+i),sum);
    }
}
#endif
static void joint_predict_batch(const JointPrediction&p,const uint8_t*x,const uint8_t*y,JointBatch&q,int n){
    int done=0;
#if JOINT_AVX2
    if(joint_has_avx()){done=n/8*8;joint_predict_avx(p,x,y,q,done);}
#endif
    for(int i=done;i<n;i++){
        int a=0,b=0;for(int k=0;k<7;k++)if(p.weight[k][i]){a+=p.weight[k][i]*x[p.index[k][i]];b+=p.weight[k][i]*y[p.index[k][i]];}
        q.px[i]=p.medoid[i]?p.reference[0][i]:uint8_t(a);q.py[i]=p.medoid[i]?p.reference[1][i]:uint8_t(b);
    }
}
