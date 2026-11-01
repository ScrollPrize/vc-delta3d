#include <stdio.h>
/* Experimental conditional linear fits using the native NG causal state,
 * selected taps, solver and coefficient quantization. Included after state API.
 * Model bytes: mode,count,lo,hi (u16LE), tap IDs, two Q8 coefficient banks
 * (each count+1 including the intercept feature128). */
static int ng_difficulty_features(ng_causal_state* s,size_t i,const uint8_t* ids,int count,int* f,unsigned* a) {
    if(!s||!s->pending||s->next!=i)return -1;
    int x=i%NG_X,y=i/NG_X%NG_Y,z=i/NG_X/NG_Y;
    size_t ri=VCN_LC_IDX(z,y,0);
    if(!s->feature_activity){s->feature_activity=malloc(2*NG_X);if(!s->feature_activity)return -1;}
    if(!x)vcn_lc_apre(s->e,ri,s->feature_activity);
    *a=s->feature_activity[x];
    for(int k=0;k<count;k++){
        unsigned id=vcn_rd16(ids+2*k);if(id>=(unsigned)s->p->t.nt)return -1;
        f[k]=(int)s->b[ri+x+s->p->t.off[id]]-128;
    }
    f[count]=128;return 0;
}
static unsigned ng_difficulty_mix(unsigned a,unsigned lo,unsigned hi){
    if(a<=lo)return 0;if(a>=hi)return 256;
    return ((a-lo)*256+(hi-lo)/2)/(hi-lo);
}
/* Eliminate the constant before solving, as native NG does, then recover its
 * intercept AFTER coefficient quantization at1/256 voxel precision. Treating
 * feature128 as an ordinary Q8 coefficient would quantize the bias to0.5vx. */
static int ng_difficulty_solve(double* C,double* rhs,unsigned dim,unsigned F,double ridge,int32_t* weights){
    unsigned ic=F-1,n=dim-1,ids[82];double cov[82*82],original[82*82],b[82],tr=0;
    if(C[ic*dim+ic]<=0)return -1;
    for(unsigned j=0,k=0;j<dim;j++)if(j!=ic)ids[k++]=j;
    for(unsigned j=0;j<dim;j++)for(unsigned k=0;k<j;k++)C[k*dim+j]=C[j*dim+k];
    for(unsigned j=0;j<n;j++){
        unsigned x=ids[j];b[j]=rhs[x]-C[x*dim+ic]*rhs[ic]/C[ic*dim+ic];
        for(unsigned k=0;k<n;k++){unsigned y=ids[k];cov[j*n+k]=C[x*dim+y]-C[x*dim+ic]*C[y*dim+ic]/C[ic*dim+ic];}
        tr+=cov[j*n+j];
    }
    for(unsigned j=0;j<n;j++)cov[j*n+j]+=(ids[j]>=F?ridge:1e-6)*tr/(n?n:1)+1e-9;
    memcpy(original,cov,sizeof(double)*n*n);
    if(n&&vcn_solve(cov,b,n))return -1;
    int32_t q[82];if(n)vcn_lc_quantize(original,b,q,n);
    double bias=rhs[ic];
    for(unsigned j=0;j<n;j++){weights[ids[j]]=q[j];bias-=q[j]/256.*C[ids[j]*dim+ic];}
    bias*=128./C[ic*dim+ic];weights[ic]=(int16_t)(uint16_t)(int32_t)floor(bias*256.+0.5);
    return 0;
}
int ng_difficulty_prediction(void* ptr,const uint8_t* model,size_t n,size_t i,uint16_t* out){
    if(n<8)return -1;
    unsigned mode=vcn_rd16(model),count=vcn_rd16(model+2),lo=vcn_rd16(model+4),hi=vcn_rd16(model+6);
    if(mode>=5&&mode<=8){
        ng_causal_state* s=ptr;
        if(!s||!s->pending||s->next!=i||count!=(unsigned)s->p->t.nt||n!=12+(mode>=7?4:2)*count)return -1;
        unsigned x=i%NG_X,y=i/NG_X%NG_Y,z=i/NG_X/NG_Y;size_t ri=VCN_LC_IDX(z,y,0);
        if(!s->feature_activity){s->feature_activity=malloc(2*NG_X);if(!s->feature_activity)return -1;}
        if(!x)vcn_lc_apre(s->e,ri,s->feature_activity);
        /* Full numerical activity scaled to byte range, not a class ID. */
        unsigned a=(s->feature_activity[x]+5)/10;if(a>255)a=255;
        int64_t q=(int16_t)vcn_rd16(model+8+2*count);
        for(unsigned k=0;k<count;k++)q+=(int16_t)vcn_rd16(model+8+2*k)*(int64_t)s->b[ri+x+s->p->t.off[k]];
        if(mode==8){
            const uint8_t* bank=model+10+2*count;
            int64_t high=(int16_t)vcn_rd16(bank+2*count);
            for(unsigned k=0;k<count;k++)high+=(int16_t)vcn_rd16(bank+2*k)*(int64_t)s->b[ri+x+s->p->t.off[k]];
            int64_t v=(255-a)*q+a*high+127;
            q=v>=0?v/255:-((-v+254)/255);
        }else{
            q+=(int16_t)vcn_rd16(model+10+2*count)*(int64_t)a;
            if(mode==7)for(unsigned k=0;k<count;k++)q+=(int16_t)vcn_rd16(model+12+2*count+2*k)*(int64_t)((s->b[ri+x+s->p->t.off[k]]*a+127)/255);
        }
        *out=(uint16_t)q;return 0;
    }
    if((mode<1||mode>4)||count>40||hi<=lo||hi>=2560||n!=8+2*count+4*(count+1))return -1;
    int f[41];unsigned a;if(ng_difficulty_features(ptr,i,model+8,count,f,&a))return -1;
    unsigned t=ng_difficulty_mix(a,lo,hi);const uint8_t* w=model+8+2*count;
    int64_t p=0,q=0;
    for(unsigned k=0;k<=count;k++){p+=(int16_t)vcn_rd16(w+2*k)*(int64_t)(k==count?1:f[k]);q+=(int16_t)vcn_rd16(w+2*(count+1+k))*(int64_t)((mode==2&&k==count)?1:f[k]);}
    if(mode==1)q=(int16_t)vcn_rd16(w+2*(count+1))*128LL;
    if(mode==4)q=0;
    int64_t numerator=(mode==2?(256-t)*p:256*p)+t*q;
    /* Explicit floor division: portable rounding for negative predictions. */
    numerator+=128;int64_t value=numerator>=0?numerator/256:-((-numerator+255)/256);
    *out=(uint16_t)(32768+value);return 0;
}
/* Extend the original fit directly: identical finalized padding, complete tap
 * candidate set, sampling lattice, Gram kernel, selection and quantization. */
static int ng_native_difficulty_prepare(const uint8_t* src,const uint8_t* params,size_t pn,const uint16_t* base,
        int mode,uint8_t* model,size_t cap,size_t* mn,uint16_t* prediction){
    ng_causal_state* s=ng_causal_create(params,pn);if(!s)return -1;
    unsigned nt=s->p->t.nt;size_t bytes=12+(mode>=7?4:2)*nt;
    if(2*nt+1>VCN_LC_MAXF||bytes>cap){ng_causal_destroy(s);return -1;}
    uint8_t* residual=malloc(NG_N),*feature=malloc(NG_N);uint16_t* a=malloc(2*NG_N);
    if(!residual||!feature||!a){free(residual);free(feature);free(a);ng_causal_destroy(s);return -1;}
    for(size_t i=0;i<NG_N;i++)residual[i]=(uint8_t)(src[i]-(uint8_t)((uint16_t)(base[i]+128u)>>8));
    int rc=ng_activity_values(residual,a);
    if(rc){free(residual);free(feature);free(a);ng_causal_destroy(s);return rc;}
    for(size_t i=0;i<NG_N;i++){unsigned v=(a[i]+5)/10;feature[i]=v>255?255:v;}
    free(residual);free(a);
    for(int z=0;z<NG_Z;z++)for(int y=0;y<NG_Y;y++){
        vcn_lc_row_begin(s->b,z,y,128);
        memcpy(s->b+VCN_LC_IDX(z,y,0),src+((size_t)z*NG_Y+y)*NG_X,NG_X);
        vcn_lc_row_end(s->b,z,y);
    }
    int32_t w[VCN_LC_MAXF]={0},w2[VCN_LC_MAXT]={0},bias=0,bias2=0;
    if(mode==8){
        int low=vcn_lc_fit_features(s->b,&s->p->t,0,NG_Z,128,0,40,w,&bias,NULL,feature,0,1,NULL,NULL,NULL);
        int high=vcn_lc_fit_features(s->b,&s->p->t,0,NG_Z,128,0,40,w2,&bias2,NULL,feature,0,2,NULL,NULL,NULL);
        if(!low&&high){memcpy(w,w2,nt*sizeof(*w));bias=bias2;}
        if(!high&&low){memcpy(w2,w,nt*sizeof(*w));bias2=bias;}
    }else if(mode==7)vcn_lc_fit_features(s->b,&s->p->t,0,NG_Z,128,0,40,w,&bias,NULL,feature,2,0,NULL,NULL,NULL);
    else vcn_lc_fit_feature(s->b,&s->p->t,0,NG_Z,128,0,40,w,&bias,NULL,mode==5?feature:NULL);
    /* Disabled-feature control must reproduce the original model exactly. */
    if(mode==6&&!rc&&(memcmp(w,s->p->w[0],nt*sizeof(*w))||bias!=s->p->bias[0])){fprintf(stderr,"native fit mismatch bias %d/%d\n",bias,s->p->bias[0]);rc=-1;}
    free(feature);ng_causal_destroy(s);if(rc)return rc;
    vcn_wr16(model,mode);vcn_wr16(model+2,nt);vcn_wr16(model+4,0);vcn_wr16(model+6,0);
    for(unsigned k=0;k<nt;k++)vcn_wr16(model+8+2*k,(uint16_t)w[k]);
    vcn_wr16(model+8+2*nt,(uint16_t)bias);
    if(mode==8){
        for(unsigned k=0;k<nt;k++)vcn_wr16(model+10+2*nt+2*k,(uint16_t)w2[k]);
        vcn_wr16(model+10+4*nt,(uint16_t)bias2);
    }else{
        vcn_wr16(model+10+2*nt,(uint16_t)(mode==5?w[nt]:mode==7?w[2*nt]:0));
        if(mode==7)for(unsigned k=0;k<nt;k++)vcn_wr16(model+12+2*nt+2*k,(uint16_t)w[nt+k]);
    }
    *mn=bytes;
    s=ng_causal_create(params,pn);if(!s)return -1;
    for(size_t i=0;i<NG_N&&!rc;i++){
        uint16_t q;uint8_t c;rc=ng_causal_prediction(s,i,&q,&c);
        if(!rc)rc=ng_difficulty_prediction(s,model,bytes,i,prediction+i);
        if(!rc&&mode==6&&prediction[i]!=base[i]){fprintf(stderr,"native prediction mismatch %zu: %u/%u\n",i,prediction[i],base[i]);rc=-1;}
        if(!rc)rc=ng_causal_accept(s,i,src[i],(uint8_t)((uint16_t)(prediction[i]+128u)>>8));
    }
    ng_causal_destroy(s);return rc;
}
int ng_difficulty_prepare(const uint8_t* src,const uint8_t* params,size_t pn,const uint16_t* base,
        int mode,double regularization,uint8_t* model,size_t cap,size_t* mn,uint16_t* prediction){
    if(mode>=5&&mode<=8)return ng_native_difficulty_prepare(src,params,pn,base,mode,model,cap,mn,prediction);
    if((mode<1||mode>4)||regularization<0)return -1;
    ng_causal_state* state=ng_causal_create(params,pn);if(!state)return -1;
    unsigned count=0;uint8_t ids[80];
    for(int k=0;k<state->p->t.nt;k++)if(state->p->w[0][k]){
        if(count==40){ng_causal_destroy(state);return -1;}vcn_wr16(ids+2*count++,k);
    }
    unsigned F=count+1,D=2*F;size_t bytes=8+2*count+4*F;
    if(bytes>cap){ng_causal_destroy(state);return -1;}
    uint8_t* residual=malloc(NG_N);uint16_t* activity=malloc(2*NG_N);
    if(!residual||!activity){free(residual);free(activity);ng_causal_destroy(state);return -1;}
    for(size_t i=0;i<NG_N;i++)residual[i]=(uint8_t)(src[i]-(uint8_t)((uint16_t)(base[i]+128u)>>8));
    if(ng_activity_values(residual,activity)){free(residual);free(activity);ng_causal_destroy(state);return -1;}
    unsigned hist[2560]={0};for(size_t i=0;i<NG_N;i++)hist[activity[i]]++;
    unsigned lo=0,hi=1;uint64_t sum=0;
    for(unsigned v=0;v<2560;v++){sum+=hist[v];if(sum*10<NG_N)lo=v;if(sum*10<9*NG_N)hi=v+1;}
    if(hi<=lo)hi=lo+1;if(hi>=2560){hi=2559;lo=2558;}
    free(residual);free(activity);
    double *C=calloc((size_t)D*D,sizeof(double)),*rhs=calloc(D,sizeof(double));
    double *C2=calloc((size_t)F*F,sizeof(double)),*rhs2=calloc(F,sizeof(double));
    if(!C||!rhs||!C2||!rhs2){free(C);free(rhs);free(C2);free(rhs2);ng_causal_destroy(state);return -1;}
    int rc=0;unsigned samples=0;
    for(size_t i=0;i<NG_N&&!rc;i++){
        uint16_t q;uint8_t cl;rc=ng_causal_prediction(state,i,&q,&cl);if(rc)break;
        unsigned x=i%NG_X,y=i/NG_X%NG_Y,z=i/NG_X/NG_Y;
        /* Evaluate features every voxel to keep row activity initialized. */
        int f[41];unsigned a;rc=ng_difficulty_features(state,i,ids,count,f,&a);if(rc)break;
        unsigned stride=NG_N<65536?32:128,start=VCN_LC_XP+(y*7+z*3)%stride;
        if(x>=start&&(x-start)%stride==0){
            double t=ng_difficulty_mix(a,lo,hi)/256.,target=(double)src[i]-128,features[82];
            for(unsigned k=0;k<F;k++){features[k]=f[k];features[F+k]=t*f[k];}
            if(mode==1)features[F]=128*t;
            unsigned dim=mode==1?F+1:mode==4?F:D;
            if(mode!=2){for(unsigned j=0;j<dim;j++){rhs[j]+=features[j]*target;for(unsigned k=0;k<=j;k++)C[j*dim+k]+=features[j]*features[k];}}
            else for(unsigned j=0;j<F;j++){rhs[j]+=(1-t)*f[j]*target;rhs2[j]+=t*f[j]*target;for(unsigned k=0;k<=j;k++){C[j*F+k]+=(1-t)*f[j]*f[k];C2[j*F+k]+=t*f[j]*f[k];}}
            samples++;
        }
        rc=ng_causal_accept(state,i,src[i],(uint8_t)((uint16_t)(q+128u)>>8));
    }
    ng_causal_destroy(state);int32_t weights[82]={0};
    if(!rc&&samples>=D*4){
        unsigned dim=mode==1?F+1:mode==3?D:F;
        if(mode==2&&C[(F-1)*F+F-1]<=0){
            rc=ng_difficulty_solve(C2,rhs2,F,F,regularization,weights+F);memcpy(weights,weights+F,F*sizeof(*weights));
        }else{
            rc=ng_difficulty_solve(C,rhs,dim,F,regularization,weights);
            if(!rc&&mode==2){
                if(C2[(F-1)*F+F-1]<=0)memcpy(weights+F,weights,F*sizeof(*weights));
                else rc=ng_difficulty_solve(C2,rhs2,F,F,regularization,weights+F);
            }
        }
    }else rc=-1;
    free(C);free(rhs);free(C2);free(rhs2);if(rc)return -1;
    vcn_wr16(model,mode);vcn_wr16(model+2,count);vcn_wr16(model+4,lo);vcn_wr16(model+6,hi);memcpy(model+8,ids,2*count);
    for(unsigned k=0;k<D;k++)vcn_wr16(model+8+2*count+2*k,(uint16_t)(int16_t)weights[k]);*mn=bytes;
    state=ng_causal_create(params,pn);if(!state)return -1;
    for(size_t i=0;i<NG_N&&!rc;i++){
        uint16_t q;uint8_t cl;rc=ng_causal_prediction(state,i,&q,&cl);
        if(!rc)rc=ng_difficulty_prediction(state,model,bytes,i,prediction+i);
        if(!rc)rc=ng_causal_accept(state,i,src[i],(uint8_t)((uint16_t)(q+128u)>>8));
    }
    ng_causal_destroy(state);return rc;
}
