/* Encoder-only Q8 coefficient refinement. The current bias-free NG activity
 * classes are frozen within a round. Score exact modular symbol mappings
 * with freshly normalized native rANS frequencies for every proposal.
 * Full prediction/activity passes are rebuilt between rounds by the caller. */
static double ng_entropy_score(size_t n,int K,int tb,const uint16_t*q,const uint8_t*v,
 const uint8_t*c,const uint8_t*a,const uint8_t*b,int delta,int bias,uint32_t*h){
  memset(h,0,(size_t)K*256*sizeof(*h));
  for(size_t i=0;i<n;i++){
    uint16_t p=(uint16_t)(q[i]+bias+delta*((a?(int)a[i]:0)-(b?(int)b[i]:0)));
    h[(size_t)c[i]*256+ng_prediction_symbol(v[i],p)]++;
  }
  double bits=0;uint16_t freq[256];
  for(int k=0;k<K;k++){
    const uint32_t*cnt=h+(size_t)k*256;vcn_lc_norm(cnt,freq,tb);
    for(int s=0;s<256;s++)if(cnt[s])bits+=cnt[s]*(tb-log2((double)freq[s]));
  }
  ng_entropy_stats[5]++;return bits;
}
static unsigned ng_entropy_refine(vcn_lc_par*p,const vcn_lc_cfg*cf,const uint8_t*b,
 const uint8_t*e,const uint16_t*Pa,int32_t feature_weights[2]){
  unsigned changed=0;const vcn_lc_taps*t=&p->t;
  for(int bank=0;bank<2;bank++){
    int zlo=bank?0:1,zhi=bank?1:VCN_LC_Z,stride=bank?(cf->stride==1?1:2):cf->stride;
    int uniform=cf->fit_session&&cf->fit_session->uniform_sparse;
    size_t count=0;
    for(int z=zlo;z<zhi;z++)for(int y=0;y<VCN_LC_Y;y++)
      for(int x=VCN_LC_XP+ng_fit_sample_phase(y,z,stride,uniform);x<VCN_LC_D;x+=stride){
        size_t i=((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x;if(!cf->fit_mask||cf->fit_mask[i])count++;
      }
    if(!count)continue;
    int ids[VCN_LC_MAXF],nt=0,anchor=0;int32_t*w[VCN_LC_MAXF];
    for(int k=0;k<t->nt;k++)if((!bank||!t->dz[k])&&p->w[bank][k]){
      ids[nt]=k;w[nt]=&p->w[bank][k];if(abs(*w[nt])>abs(*w[anchor]))anchor=nt;nt++;
    }
    if(cf->fit_feature){ids[nt]=t->nt;w[nt++]=&feature_weights[bank];}
    uint8_t*features=malloc(count*(size_t)(nt?nt:1)),*values=malloc(count),*classes=malloc(count);
    uint16_t*q=malloc(count*sizeof(*q));uint32_t*h=malloc((size_t)p->K*256*sizeof(*h));
    if(!features||!values||!classes||!q||!h){free(features);free(values);free(classes);free(q);free(h);continue;}
    double mean[VCN_LC_MAXF]={0};size_t row=0;
    for(int z=zlo;z<zhi;z++)for(int y=0;y<VCN_LC_Y;y++){
      uint16_t ap[VCN_LC_D];size_t ri=VCN_LC_IDX(z,y,0);vcn_lc_apre(e,ri,ap);
      for(int x=VCN_LC_XP+ng_fit_sample_phase(y,z,stride,uniform);x<VCN_LC_D;x+=stride){
        size_t i=((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x;if(cf->fit_mask&&!cf->fit_mask[i])continue;
        values[row]=b[ri+x];q[row]=Pa[i];classes[row]=p->lut[ap[x]];
        for(int k=0;k<nt;k++){
          uint8_t value=ids[k]<t->nt?b[ri+x+t->off[ids[k]]]:cf->fit_feature[i];
          features[(size_t)k*count+row]=value;mean[k]+=value;
        }
        row++;
      }
    }
    for(int k=0;k<nt;k++)mean[k]/=count;
    double best=ng_entropy_score(count,p->K,p->tb,q,values,classes,NULL,NULL,0,0,h),before=best;
    /* Coarse-to-fine integer moves, including DC-preserving exchanges with
     * the strongest spatial tap. No additional model/tap/format fields. */
    for(int level=4;level>=1;level/=4)for(int k=0;k<nt;k++){
      int best_delta=0,best_bias=0,best_partner=-1;
      for(int paired=0;paired<2;paired++){
        int partner=paired?anchor:-1;if(partner==k)continue;
        for(int sign=-1;sign<=1;sign+=2){
          int delta=sign*level;
          if(abs(*w[k]+delta)>32000||(partner>=0&&abs(*w[partner]-delta)>32000))continue;
          int bias=-(int)floor(delta*(mean[k]-(partner>=0?mean[partner]:0))+.5);
          double cost=ng_entropy_score(count,p->K,p->tb,q,values,classes,features+(size_t)k*count,
            partner>=0?features+(size_t)partner*count:NULL,delta,bias,h);
          if(cost+1e-9<best){best=cost;best_delta=delta;best_bias=bias;best_partner=partner;}
        }
      }
      if(best_delta){
        *w[k]+=best_delta;if(best_partner>=0)*w[best_partner]-=best_delta;
        p->bias[bank]=(uint16_t)(p->bias[bank]+best_bias);
        for(size_t i=0;i<count;i++)q[i]=(uint16_t)(q[i]+best_bias+best_delta*(features[(size_t)k*count+i]-(best_partner>=0?(int)features[(size_t)best_partner*count+i]:0)));
        changed++;ng_entropy_stats[4]++;
      }
    }
    for(int step=64;step>=1;step/=4){
      int chosen=0;
      for(int sign=-1;sign<=1;sign+=2){
        double cost=ng_entropy_score(count,p->K,p->tb,q,values,classes,NULL,NULL,0,sign*step,h);
        if(cost+1e-9<best){best=cost;chosen=sign*step;}
      }
      if(chosen){p->bias[bank]=(uint16_t)(p->bias[bank]+chosen);for(size_t i=0;i<count;i++)q[i]=(uint16_t)(q[i]+chosen);changed++;ng_entropy_stats[4]++;}
    }
    ng_entropy_stats[0]++;ng_entropy_stats[1]+=(double)count;ng_entropy_stats[2]+=before;ng_entropy_stats[3]+=best;
    free(features);free(values);free(classes);free(q);free(h);
  }
  return changed;
}
