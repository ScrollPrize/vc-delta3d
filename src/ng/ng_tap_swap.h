/* Opt-in encoder experiment. Keep the native tap budget, solve and Q8 rules.
 * Rank all one-for-one swaps by training covariance; test the best eight on
 * disjoint spatial samples using normalized residual histogram coding cost.
 * This proxy deliberately does not claim to reproduce adaptive class/bias
 * histories. Final frame size is measured separately, without a fallback. */
static int ng_swap_system(const double*C,const double*r,int n,const int*s,int m,
                         double*A,double*b){
  for(int a=0;a<m;a++){b[a]=r[s[a]];for(int k=0;k<m;k++)A[a*m+k]=C[s[a]*n+s[k]];}
  return vcn_solve(A,b,m);
}
static double ng_swap_score(const double*C,const double*r,const double*mean,double mt,
 int n,const int*s,int m,const int16_t*X,const uint8_t*Y,int ns,double*work){
  double *A=work,*A0=work+(size_t)m*m,b[VCN_LC_MAXF];
  int32_t q[VCN_LC_MAXF];
  for(int a=0;a<m;a++)for(int k=0;k<m;k++)A0[a*m+k]=C[s[a]*n+s[k]];
  if(ng_swap_system(C,r,n,s,m,A,b))return HUGE_VAL;
  vcn_lc_quantize(A0,b,q,m);double bias=mt;
  for(int k=0;k<m;k++)bias-=q[k]/256.*mean[s[k]];
  int32_t qb=(uint16_t)(int32_t)floor(bias*256.+.5);uint32_t hist[256]={0};
  for(int j=0;j<ns;j++){
    int32_t v=qb;for(int k=0;k<m;k++)v+=q[k]*(int32_t)X[j*n+s[k]];
    hist[ng_prediction_symbol(Y[j],(uint16_t)v)]++;
  }
  return ng_hist_cost(hist,12);
}
static int ng_tap_select(double*C,double*r,double*mean,int*id,int n,int budget,double tr,
 const uint8_t*b,const vcn_lc_taps*t,int zlo,int zhi,int stride,
 const uint8_t*fit_mask,const uint8_t*eligible,ng_fit_session*session,double mt,
 const int16_t*target,const uint8_t*feature,int kind){
  if(!(ng_tap_swap_mode&ng_tap_fit_phase)||budget<=0||budget>=n)
    return vcn_lc_select(C,r,mean,id,n,budget,tr);
  double *all=(double*)malloc(sizeof(double)*n*n), rr[VCN_LC_MAXF],mm[VCN_LC_MAXF];
  int ids[VCN_LC_MAXF],s[VCN_LC_MAXF];
  if(!all)return vcn_lc_select(C,r,mean,id,n,budget,tr);
  memcpy(all,C,sizeof(double)*n*n);memcpy(rr,r,sizeof(double)*n);
  memcpy(mm,mean,sizeof(double)*n);memcpy(ids,id,sizeof(int)*n);
  int m=vcn_lc_select(C,r,mean,id,n,budget,tr);
  for(int k=0;k<m;k++){s[k]=0;while(ids[s[k]]!=id[k])s[k]++;}
  const int cap=2048;int ns=0;
  int16_t *X=(int16_t*)malloc(sizeof(int16_t)*cap*n);uint8_t Y[2048];
  /* Keep the workspace off small worker stacks (notably macOS pthreads),
   * and reuse it for every candidate instead of allocating per swap. */
  double *work=(double*)malloc(sizeof(double)*2*m*m);
  size_t volume=(size_t)(zhi-zlo)*VCN_LC_Y*VCN_LC_D;
  /* Fixed-seed uniform reservoir over eligible non-training targets. No large
   * feature table or full-chunk predictor pass is constructed. */
  uint32_t rng=0x391af76du;size_t seen=0;
  if(X&&work)for(size_t j=0;j<volume;j++){
    size_t flat=j+(size_t)zlo*VCN_LC_Y*VCN_LC_D;
    int x=flat%VCN_LC_D,y=(flat/VCN_LC_D)%VCN_LC_Y,z=flat/((size_t)VCN_LC_D*VCN_LC_Y);
    if(x<VCN_LC_XP||(eligible&&!eligible[flat]))continue;
    int phase=ng_fit_sample_phase(y,z,stride,session&&session->uniform_sparse);
    if((!fit_mask||fit_mask[flat])&&x>=VCN_LC_XP+phase&&(x-VCN_LC_XP-phase)%stride==0)continue;
    seen++;rng=rng*1664525u+1013904223u;
    size_t slot=seen<=cap?seen-1:rng%seen;if(slot>=cap)continue;
    size_t i=VCN_LC_IDX(z,y,x);unsigned a=kind==4?z==0:feature?feature[flat]:0;
    for(int k=0;k<n;k++)X[slot*n+k]=ng_fit_input_value(b,i,t,ids[k],kind,a,z==0);
    Y[slot]=(uint8_t)((target?target[flat]:b[i])-(kind==3?a:0));
    if(ns<cap)ns++;
  }
  if(ns>=64){
    ng_tap_swap_stats[0]++;ng_tap_swap_stats[1]+=ns;ng_tap_swap_stats[7]+=ng_tap_fit_phase==1;
    double best=ng_swap_score(all,rr,mm,mt,n,s,m,X,Y,ns,work);ng_tap_swap_stats[5]+=best;
    for(int round=0;round<2;round++){
      double gain[8];int drop[8],add[8];for(int k=0;k<8;k++){gain[k]=-HUGE_VAL;drop[k]=add[k]=-1;}
      char used[VCN_LC_MAXF]={0};for(int k=0;k<m;k++)used[s[k]]=1;
      for(int a=0;a<m;a++)for(int k=0;k<n;k++)if(!used[k]){
        int old=s[a];s[a]=k;
        double coef[VCN_LC_MAXF],g=0;
        if(!ng_swap_system(all,rr,n,s,m,work,coef)){
          for(int q=0;q<m;q++)g+=rr[s[q]]*coef[q];
          for(int q=0;q<8;q++)if(g>gain[q]){
            for(int v=7;v>q;v--){gain[v]=gain[v-1];drop[v]=drop[v-1];add[v]=add[v-1];}
            gain[q]=g;drop[q]=a;add[q]=k;break;
          }
        }
        s[a]=old;ng_tap_swap_stats[2]++;
      }
      int winner=-1;
      for(int q=0;q<8&&drop[q]>=0;q++){
        int old=s[drop[q]];s[drop[q]]=add[q];
        /* Sort before scoring: exactly the final solve/quantization order. */
        int sorted[VCN_LC_MAXF];memcpy(sorted,s,m*sizeof(int));
        for(int a=1;a<m;a++){int v=sorted[a],k=a;while(k&&sorted[k-1]>v){sorted[k]=sorted[k-1];k--;}sorted[k]=v;}
        double cost=ng_swap_score(all,rr,mm,mt,n,sorted,m,X,Y,ns,work);
        if(cost<best-1e-6){best=cost;winner=q;}
        s[drop[q]]=old;ng_tap_swap_stats[3]++;
      }
      if(winner<0)break;
      s[drop[winner]]=add[winner];
      for(int a=1;a<m;a++){int v=s[a],k=a;while(k&&s[k-1]>v){s[k]=s[k-1];k--;}s[k]=v;}
      ng_tap_swap_stats[4]++;
    }
    ng_tap_swap_stats[6]+=best;
    for(int a=0;a<m;a++){r[a]=rr[s[a]];mean[a]=mm[s[a]];id[a]=ids[s[a]];
      for(int k=0;k<m;k++)C[a*m+k]=all[s[a]*n+s[k]];}
  }
  free(work);free(X);free(all);return m;
}
