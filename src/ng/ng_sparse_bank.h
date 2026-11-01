/* Fit/calibrate on deterministic samples; score only proposed 4³ blocks.
 * Uses the original integer Gram/NG solver, prediction and symbol mapping.
 * No full-volume candidate arrays for the additional models. */
typedef struct {
    vcn_lc_par p[12];vcn_lc_plist pl[12][2];int32_t fw[12][2];
    const uint8_t*src;const uint8_t*classes;uint8_t*b,*feature;
    ng_fit_session fit;vcn_lc_cfg cfg;
    unsigned stride;uint64_t samples;
    uint8_t*owned_classes,*probe_mask;
} ng_sparse_bank;
static void ng_sparse_destroy_impl(void*ptr){
    ng_sparse_bank*s=ptr;if(!s)return;
    for(int b=0;b<2;b++)ng_fit_bank_clear(s->fit.banks+b);
    free(s->b);free(s->feature);free(s->owned_classes);free(s->probe_mask);free(s);
}
static void*ng_sparse_create_impl(const uint8_t*src,const uint8_t*params,size_t n,const uint8_t*classes,unsigned stride){
    if(stride<16||stride>8192)return NULL;
    ng_sparse_bank*s=calloc(1,sizeof(*s));if(!s)return NULL;
    const uint8_t*cur=params;vcn_lc_par*p=NULL;
    if(vcn_lc_readpar(&cur,n,&p)!=VCN_OK||cur!=params+n){free(p);free(s);return NULL;}
    s->p[0]=*p;free(p);s->src=src;s->classes=classes;s->stride=stride;s->fit.simd=1;s->fit.uniform_sparse=1;
    s->cfg=(vcn_lc_cfg){3,3,1,16,40,128};s->cfg.tb=s->p[0].tb;
    s->b=malloc(VCN_LC_NB);s->feature=malloc((size_t)VCN_LC_D*VCN_LC_Y*VCN_LC_Z);
    if(!s->b||!s->feature){ng_sparse_destroy_impl(s);return NULL;}
    memset(s->b,128,VCN_LC_NB);
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++){
        vcn_lc_row_begin(s->b,z,y,128);
        memcpy(s->b+VCN_LC_IDX(z,y,0),src+((size_t)z*VCN_LC_Y+y)*VCN_LC_D,VCN_LC_D);
        vcn_lc_row_end(s->b,z,y);
    }
    for(int b=0;b<2;b++)vcn_lc_mkplist(&s->p[0].t,s->p[0].w[b],&s->pl[0][b]);
    return s;
}
static int ng_sparse_fit_impl(void*ptr,unsigned m,const uint8_t*choices,uint8_t*out,size_t cap,size_t*n,int32_t*fw){
    ng_sparse_bank*s=ptr;if(!s||!m||m>=12||cap<4096||choices)return -1;
    vcn_lc_par*p=&s->p[m];*p=s->p[0];
    int64_t sums[VCN_LC_MAXK]={0};uint64_t counts[VCN_LC_MAXK]={0};
    for(int bank=0;bank<2;bank++){
        int begin=bank?0:1,end=bank?1:VCN_LC_Z,stride=bank?(s->stride/64>2?s->stride/64:2):s->stride;
        for(int z=begin;z<end;z++)for(int y=0;y<VCN_LC_Y;y++)for(int x=VCN_LC_XP+ng_fit_sample_phase(y,z,stride,1);x<VCN_LC_D;x+=stride){
            size_t i=((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x;
            s->feature[i]=ng_directional_base(s->src,i,m,x,y,z,VCN_LC_D);
        }
        int32_t weights[VCN_LC_MAXF]={0};
        vcn_lc_fit_features(s->b,&p->t,begin,end,stride,bank,bank?20:40,weights,&p->bias[bank],NULL,s->feature,1,0,NULL,&s->fit,NULL);
        memcpy(p->w[bank],weights,p->t.nt*sizeof(int32_t));s->fw[m][bank]=weights[p->t.nt];fw[bank]=s->fw[m][bank];
        vcn_lc_mkplist(&p->t,p->w[bank],&s->pl[m][bank]);
        /* Class-bias calibration uses only the same sampled source positions,
         * classified by the initial plain-NG history (an explicit estimate). */
        for(int z=begin;z<end;z++)for(int y=0;y<VCN_LC_Y;y++)for(int x=VCN_LC_XP+ng_fit_sample_phase(y,z,stride,1);x<VCN_LC_D;x+=stride){
            size_t i=((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x,ri=VCN_LC_IDX(z,y,x);
            uint16_t q=(uint16_t)(p->bias[bank]+(uint32_t)(uint16_t)fw[bank]*s->feature[i]);
            for(int k=0;k<p->t.nt;k++)q=(uint16_t)(q+(uint32_t)(uint16_t)p->w[bank][k]*s->b[ri+p->t.off[k]]);
            unsigned c=s->classes[i];sums[c]+=(int16_t)(uint16_t)(256u*s->src[i]-q);counts[c]++;s->samples++;
        }
    }
    for(int c=0;c<p->K;c++){
        double v=counts[c]?floor((double)sums[c]/counts[c]+.5):0;
        p->cb[c]=(int32_t)fmax(-32768.,fmin(32767.,v));
    }
    *n=vcn_lc_writepar(p,&s->cfg,out);return 0;
}
static void ng_sparse_row(ng_sparse_bank*s,unsigned m,int x,int y,int z,const uint16_t*activity,uint8_t*out){
    vcn_lc_par*p=&s->p[m];size_t i=((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x,ri=VCN_LC_IDX(z,y,x);int bank=z==0;
    uint16_t q[4];vcn_lc_pre_span(s->b+ri,&s->pl[m][bank],(uint16_t)p->bias[bank],q,4);
    for(int j=0;j<4;j++){
        for(int k=0;k<p->t.nrow;k++){
            uint8_t v=x+j>k?s->b[ri+j-1-k]:s->b[VCN_LC_IDX(z,y-1,0)];
            q[j]=(uint16_t)(q[j]+(uint32_t)(uint16_t)p->w[bank][k]*v);
        }
        if(m)q[j]=(uint16_t)(q[j]+(uint32_t)(uint16_t)s->fw[m][bank]*ng_directional_base(s->src,i+j,m,x+j,y,z,VCN_LC_D));
        unsigned c=activity?p->lut[activity[x+j]]:s->classes[i+j];q[j]=(uint16_t)(q[j]+p->cb[c]);
        out[j]=ng_prediction_symbol(s->src[i+j],q[j]);
    }
}
static int ng_sparse_block_impl(void*ptr,unsigned m,unsigned x,unsigned y,unsigned z,uint8_t*out){
    ng_sparse_bank*s=ptr;if(!s||m>=12||x+4>VCN_LC_D||y+4>VCN_LC_Y||z+4>VCN_LC_Z)return -1;
    for(unsigned dz=0;dz<4;dz++)for(unsigned dy=0;dy<4;dy++)ng_sparse_row(s,m,x,y+dy,z+dz,NULL,out+(dz*4+dy)*4);
    return 0;
}
static int ng_sparse_emit_impl(void*ptr,const uint8_t*choices,uint8_t*symbols,uint16_t*activity){
    ng_sparse_bank*s=ptr;if(!s)return -1;uint8_t*e=calloc(VCN_LC_NB,1);if(!e)return -1;
    memset(s->b,128,VCN_LC_NB);
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++){
        size_t i=((size_t)z*VCN_LC_Y+y)*VCN_LC_D,ri=VCN_LC_IDX(z,y,0);vcn_lc_row_begin(e,z,y,0);vcn_lc_apre(e,ri,activity+i);
        vcn_lc_row_begin(s->b,z,y,128);memcpy(s->b+ri,s->src+i,VCN_LC_D);
        for(int x=0;x<VCN_LC_D;x+=4){
            unsigned c=choices[((z/4)*(VCN_LC_Y/4)+y/4)*(VCN_LC_D/4)+x/4];
            if(c>=12)ng_sparse_row(s,c-12,x,y,z,activity+i,symbols+i+x);
            else for(int j=0;j<4;j++){
                uint8_t r=(uint8_t)(s->src[i+x+j]-ng_directional_base(s->src,i+x+j,c,x+j,y,z,VCN_LC_D));
                symbols[i+x+j]=ng_prediction_symbol(r,0);
            }
        }
        for(int x=0;x<VCN_LC_D;x++)e[ri+x]=(symbols[i+x]+1u)>>1;
        vcn_lc_row_end(e,z,y);vcn_lc_row_end(s->b,z,y);
    }free(e);return 0;
}
/* Sample-only fitting probes reuse the native fitter, quantizer, padding and
 * sparse prediction kernel. No full-volume predictions or entropy coding. */
static void*ng_fit_probe_create_impl(const uint8_t*src){
    vcn_lc_par p={0};vcn_lc_build(&p.t,3,3,1);p.K=16;p.tb=12;
    vcn_lc_cfg cfg={3,3,1,16,40,128};uint8_t params[65536];size_t n=vcn_lc_writepar(&p,&cfg,params);
    ng_sparse_bank*s=ng_sparse_create_impl(src,params,n,NULL,128);if(!s)return NULL;
    size_t nv=(size_t)VCN_LC_D*VCN_LC_Y*VCN_LC_Z;
    s->owned_classes=calloc(nv,1);s->probe_mask=calloc(nv,1);s->classes=s->owned_classes;
    if(!s->owned_classes||!s->probe_mask){ng_sparse_destroy_impl(s);return NULL;}return s;
}
static int ng_fit_probe_samples_impl(void*ptr,unsigned m,const uint32_t*positions,const uint8_t*active,size_t n,unsigned budget,uint8_t*out){
    ng_sparse_bank*s=ptr;if(!s||m>=12||!positions||!active||!out||n%4||!budget)return -1;
    size_t nv=(size_t)VCN_LC_D*VCN_LC_Y*VCN_LC_Z;
    memset(s->probe_mask,0,nv);size_t total=0,k=0;
    for(size_t j=0;j<n;j++){if(positions[j]>=nv)return -1;total+=active[j]!=0;}
    for(size_t j=0;j<n;j++)if(active[j]){
        if(total<=budget||((k+1)*budget)/total!=(k*budget)/total){
            size_t i=positions[j];s->probe_mask[i]=1;
            if(m){unsigned x=i%VCN_LC_D,y=i/VCN_LC_D%VCN_LC_Y,z=i/((size_t)VCN_LC_D*VCN_LC_Y);
                s->feature[i]=ng_directional_base(s->src,i,m,x,y,z,VCN_LC_D);}
        }k++;
    }
    vcn_lc_par*p=&s->p[m];*p=s->p[0];memset(p->cb,0,sizeof(p->cb));
    for(int bank=0;bank<2;bank++){
        int32_t weights[VCN_LC_MAXF]={0};
        vcn_lc_fit_features(s->b,&p->t,bank?0:1,bank?1:VCN_LC_Z,1,bank,bank?20:40,weights,&p->bias[bank],NULL,m?s->feature:NULL,m?1:0,0,s->probe_mask,NULL,NULL);
        memcpy(p->w[bank],weights,p->t.nt*sizeof(int32_t));s->fw[m][bank]=m?weights[p->t.nt]:0;
        vcn_lc_mkplist(&p->t,p->w[bank],&s->pl[m][bank]);
    }
    for(size_t j=0;j<n;j+=4){size_t i=positions[j];unsigned x=i%VCN_LC_D,y=i/VCN_LC_D%VCN_LC_Y,z=i/((size_t)VCN_LC_D*VCN_LC_Y);
        if(x+4>VCN_LC_D||positions[j+3]!=i+3)return -1;
        ng_sparse_row(s,m,x,y,z,NULL,out+j);
    }return 0;
}
