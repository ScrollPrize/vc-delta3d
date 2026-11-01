/* Dynamic-context adapter around the native NG entropy primitives. */
int ng_context_rans_costs(const uint32_t*hist,unsigned groups,double*cost){
    if(!hist||!cost||!groups||groups>256)return -1;
    for(unsigned g=0;g<groups;g++){uint16_t f[256];vcn_lc_norm(hist+g*256,f,12);
        for(unsigned s=0;s<256;s++)cost[g*256+s]=f[s]?12.-log2(f[s]):16.;}
    return 0;
}
static int ng_context_rans_encode_impl(const uint8_t*sy,const uint8_t*cl,const uint8_t*lane,size_t n,unsigned K,unsigned tb,uint16_t*freq,uint8_t*out,size_t cap,size_t*used,int fixed){
    if(!sy||!cl||!lane||!freq||!out||!used||!K||K>256||tb<10||tb>12||n>UINT32_MAX)return -1;
    uint32_t*hist=calloc(K*256,sizeof(uint32_t));vcn_lc_esym*es=malloc(K*256*sizeof(*es));uint16_t*words=malloc((n+1)*sizeof(uint16_t));
    if(!hist||!es||!words){free(hist);free(es);free(words);return -1;}int rc=-1;
    for(size_t i=0;i<n;i++){if(cl[i]>=K||lane[i]>=32)goto done;hist[cl[i]*256+sy[i]]++;}
    for(unsigned k=0;k<K;k++){
        if(!fixed)vcn_lc_norm(hist+k*256,freq+k*256,tb);uint32_t cum=0;
        for(unsigned s=0;s<256;s++){uint32_t f=freq[k*256+s];vcn_lc_init_esym(es+k*256+s,f,cum,1u<<tb,tb);cum+=f;}
    }
    uint32_t S[32];for(unsigned j=0;j<32;j++)S[j]=VCN_LC_RL;size_t nw=0;
    for(size_t i=n;i-->0;)S[lane[i]]=vcn_lc_put_esym(S[lane[i]],es+cl[i]*256+sy[i],words,&nw);
    if(cap<128+2*nw)goto done;
    for(unsigned j=0;j<32;j++)vcn_wr32(out+4*j,S[j]);
    for(size_t j=0;j<nw;j++)vcn_wr16(out+128+2*j,words[nw-1-j]);
    *used=128+2*nw;rc=0;
done:free(hist);free(es);free(words);return rc;
}
int ng_context_rans_encode(const uint8_t*sy,const uint8_t*cl,const uint8_t*lane,size_t n,unsigned K,unsigned tb,uint16_t*freq,uint8_t*out,size_t cap,size_t*used){return ng_context_rans_encode_impl(sy,cl,lane,n,K,tb,freq,out,cap,used,0);}
int ng_context_rans_encode_fixed(const uint8_t*sy,const uint8_t*cl,const uint8_t*lane,size_t n,unsigned K,unsigned tb,uint16_t*freq,uint8_t*out,size_t cap,size_t*used){return ng_context_rans_encode_impl(sy,cl,lane,n,K,tb,freq,out,cap,used,1);}
void*ng_context_rans_create(const uint8_t*table,size_t tn,unsigned K,unsigned tb,const uint8_t*bytes,size_t bn){
    if(!table||!bytes||bn<128||!K||K>256||tb<10||tb>12)return NULL;
    ng_context_rans*c=calloc(1,sizeof(*c));if(!c)return NULL;
    c->table=malloc((size_t)K*(1u<<tb)*sizeof(uint32_t));if(!c->table){free(c);return NULL;}
    const uint8_t*p=table;
    if(vcn_lc_read_tables(&p,table+tn,c->table,K,1u<<tb)||p!=table+tn){free(c->table);free(c);return NULL;}
    c->K=K;c->d.T=c->table;c->d.tb=tb;c->d.p=bytes+128;c->d.end=bytes+bn;
    for(unsigned j=0;j<32;j++){c->d.S[j]=vcn_rd32(bytes+4*j);if(c->d.S[j]<VCN_LC_RL){free(c->table);free(c);return NULL;}}
    return c;
}
int ng_context_rans_read(void*ptr,unsigned n,const uint8_t*cl,uint8_t*sy,unsigned method){
    ng_context_rans*c=ptr;if(!c||n>32)return -1;
    for(unsigned j=0;j<n;j++)if(cl[j]>=c->K)return -1;
    /* Native SIMD batches are exactly32 states; boundary waves are shorter. */
    ng_bank_read_symbols(n,cl,&c->d,sy,(n<32||method==0)?524288:method==1?2048:262144|2048);
    return c->d.over?-1:0;
}
int ng_context_rans_finish(void*ptr){
    ng_context_rans*c=ptr;if(!c||c->d.over||c->d.p!=c->d.end)return -1;
    for(unsigned j=0;j<32;j++)if(c->d.S[j]!=VCN_LC_RL)return -1;return 0;
}
int ng_context_rans_read_wave(void*ptr,unsigned lanes,unsigned width,const uint8_t*groups,uint8_t*symbols){
    ng_context_rans*c=ptr;if(!c||!lanes||lanes>32)return -1;
    size_t n=(size_t)lanes*width;
    for(size_t i=0;i<n;i++)if(groups[i]>=c->K)return -1;
    if(lanes==32)ng_bank_read_symbols((int)n,groups,&c->d,symbols,2048|262144);
    else for(unsigned x=0;x<width;x++)ng_bank_read_symbols(lanes,groups+x*lanes,&c->d,symbols+x*lanes,524288);
    return c->d.over?-1:0;
}
int ng_context_rans_read_rows(void*ptr,unsigned n,const uint8_t*groups,uint8_t*symbols){
    ng_context_rans*c=ptr;if(!c||n%32)return -1;
    for(unsigned i=0;i<n;i++)if(groups[i]>=c->K)return -1;
    ng_bank_read_symbols(n,groups,&c->d,symbols,2048|262144);
    return c->d.over?-1:0;
}
void ng_context_rans_destroy(void*ptr){ng_context_rans*c=ptr;if(c){free(c->table);free(c);}}
