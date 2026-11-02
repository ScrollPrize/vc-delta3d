/* Shared candidate evaluation; specialized by the existing geometry builds. */
extern _Thread_local int ng_bank_selected_only;
static int ng_bank_symbols_core(unsigned count,const uint8_t* const* params,const size_t* sizes,
    const uint16_t* const* base,const uint8_t* choices,const uint8_t* src,
    uint8_t* const* candidates,uint8_t* symbols,uint16_t* activity,const int32_t* feature_weights,uint16_t* exported_core) {
    if(count<12||count>48||count%12)return -1;
    vcn_lc_par* p[48]={0};uint8_t* e=calloc(VCN_LC_NB,1);int rc=-1;
    if(!e)return -1;
    for(unsigned m=0;m<count;m++){const uint8_t* cur=params[m];
        if(!cur&&!sizes[m])continue;
        if(vcn_lc_readpar(&cur,sizes[m],&p[m])!=VCN_OK||cur!=params[m]+sizes[m])goto done;}
    int shared=1;
    for(unsigned m=1;m<count;m++)if(sizes[m]!=sizes[0]||memcmp(params[m],params[0],sizes[0]))shared=0;
    if(feature_weights&&!shared)goto done;
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++){
        size_t ri=VCN_LC_IDX(z,y,0),o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        vcn_lc_row_begin(e,z,y,0);vcn_lc_apre(e,ri,activity+o);
        uint8_t row_choices[VCN_LC_D];
        const unsigned edge=ng_choice_edge;
        const size_t choice_row=((z/edge)*(VCN_LC_Y/edge)+y/edge)*(VCN_LC_D/edge);
        for(unsigned x=0;x<VCN_LC_D;x+=edge)memset(row_choices+x,choices[choice_row+x/edge],edge);

        uint16_t bias[VCN_LC_D];uint8_t shared_class[VCN_LC_D];
        if(shared){memset(bias,0,sizeof(bias));VCN_LC_CLS(e,ri,p[0],bias,shared_class);}
        uint8_t needed[48]={0};
        if(ng_bank_selected_only)for(unsigned x=0;x<VCN_LC_D;x+=edge){
            unsigned m=row_choices[x];
            if(m>=12+count)goto done;
            if(m>=12)needed[m-12]=1;
        }
        for(unsigned m=0;m<count;m++){
            if(!p[m])continue;
            // A compact screened menu need not materialize unused inputs.
            if(!candidates[12+m])continue;
            if(ng_bank_selected_only&&!needed[m])continue;
            if(m&&shared&&base[m]==base[0]&&candidates[12+m]==candidates[12]&&(!ng_bank_selected_only||needed[0]))continue;
            uint16_t q[VCN_LC_D];uint8_t cl[VCN_LC_D],err[VCN_LC_D];
            if(shared){for(int x=0;x<VCN_LC_D;x++)q[x]=(uint16_t)(base[m][o+x]+bias[x]);}
            else {memcpy(q,base[m]+o,sizeof(q));VCN_LC_CLS(e,ri,p[m],q,cl);}
            if(m==0&&exported_core)memcpy(exported_core+o,q,sizeof(q));
            VCN_LC_SYMS(q,src+o,candidates[12+m]+o,err);
        }
        if(feature_weights){
            uint16_t q[VCN_LC_D];uint8_t err[VCN_LC_D];
            const uint16_t weight=(uint16_t)feature_weights[z==0?1:0];
            for(int x=0;x<VCN_LC_D;x++){
                unsigned m=row_choices[x];
                unsigned feature=m>12?ng_directional_base(src,o+x,m-12,x,y,z,VCN_LC_D):0;
                q[x]=(uint16_t)(base[0][o+x]+bias[x]+weight*feature);
            }
            VCN_LC_SYMS(q,src+o,symbols+o,err);
        }
        for(unsigned x=0;x<VCN_LC_D;x+=edge){
            unsigned m=row_choices[x];
            if(m>=12+count||!candidates[m]||(m>=12&&!p[m-12]))goto done;
            if(!feature_weights||m<12)memcpy(symbols+o+x,candidates[m]+o+x,edge);
            for(unsigned j=0;j<edge;j++)e[ri+x+j]=(symbols[o+x+j]+1u)>>1;
        }
        vcn_lc_row_end(e,z,y);
    }
    rc=0;
done:free(e);for(unsigned m=0;m<count;m++)free(p[m]);return rc;
}
static int ng_bank_symbols_impl(unsigned c,const uint8_t*const*p,const size_t*n,const uint16_t*const*b,const uint8_t*s,const uint8_t*x,uint8_t*const*v,uint8_t*y,uint16_t*a){
    return ng_bank_symbols_core(c,p,n,b,s,x,v,y,a,NULL,NULL);
}

