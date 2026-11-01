#pragma once
extern _Thread_local int ng_bank_scalar_control;
/* Shared activity decode body for callback users and contiguous decoded class
 * streams. Constant geometry removes TLS pitch/loop lookups in the hot path. */
static int ng_activity_decode_impl(const uint16_t*thr,int K,int unmapped,ng_symbol_reader reader,void*ctx,
    const uint8_t* const* streams,const size_t* counts,size_t* positions,uint8_t*raw){
    if(K<2||K>VCN_LC_MAXK)return -1;
    vcn_lc_par p={0};p.K=K;
    for(int k=0;k<K-1;k++){if(thr[k]>=VCN_LC_AMAX)return -1;p.thr[k]=thr[k];}
    vcn_lc_mklut(&p);uint8_t*e=calloc(VCN_LC_NB,1);if(!e)return -1;
    uint16_t activity[VCN_LC_D];uint8_t classes[VCN_LC_D];int rc=0;
    const int simdclass=ng_bank_scalar_control==11||ng_bank_scalar_control==12;
    if(simdclass)memset(activity,0,sizeof(activity));
    const int branchless=!unmapped&&ng_bank_scalar_control>=4;
    for(int z=0;z<VCN_LC_Z&&!rc;z++)for(int y=0;y<VCN_LC_Y&&!rc;y++){
        size_t ri=VCN_LC_IDX(z,y,0),o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        vcn_lc_row_begin(e,z,y,0);
        if(simdclass)VCN_LC_CLS(e,ri,&p,activity,classes);else vcn_lc_apre(e,ri,activity);
        for(int x=0;x<VCN_LC_D;x++){
            unsigned c=simdclass?classes[x]:p.lut[activity[x]];int s;
            if(streams){if(positions[c]>=counts[c]){rc=-1;break;}s=streams[c][positions[c]++];}
            else s=reader(ctx,o+x,c);
            if(s<0||s>255){rc=-1;break;}
            if(branchless){
                raw[o+x]=(uint8_t)((s>>1)^-(s&1));
                e[ri+x]=(uint8_t)((s+1)>>1);
            }else{
                raw[o+x]=(uint8_t)(unmapped?s:((s&1)?-(s+1)/2:s/2));
                int r=(int8_t)raw[o+x];e[ri+x]=(uint8_t)(r<0?-r:r);
            }
        }
        vcn_lc_row_end(e,z,y);
    }
    free(e);return rc;
}
