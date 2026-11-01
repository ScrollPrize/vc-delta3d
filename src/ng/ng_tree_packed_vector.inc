NG_TATTR void NG_TNAME(unsigned count,const int*features,const int*start,int*leaves)const{
    const NG_TV mask7=NG_TSET(127),mask10=NG_TSET(1023);
    alignas(64) int lane_values[NG_TSTEP];
    for(unsigned j=0;j<count;j+=NG_TSTEP){
        for(unsigned k=0;k<NG_TSTEP;k++)lane_values[k]=j+k;
        NG_TV node=NG_TLOAD(start+j),lane=NG_TLOAD(lane_values),result=NG_TSET(0);
        for(unsigned depth=0;depth<=8;depth++){
            NG_TV info=NG_TGATHER(packed.data(),node),f=NG_TAND(info,NG_TSET(15));
            NG_TV done=NG_TEQ(f,NG_TSET(15));
            NG_TV cut=NG_TAND(NG_TSRL(info,4),mask10);
            result=NG_TBLEND(result,cut,done);
            if(NG_TALL(done))break;
            f=NG_TANDNOT(done,f);
            NG_TV value=NG_TGATHER(features,NG_TADD(NG_TSLL(f,5),lane));
            NG_TV left=NG_TAND(NG_TSRL(info,14),mask7),right=NG_TAND(NG_TSRL(info,21),mask7);
            NG_TV next=NG_TOR(NG_TANDNOT(mask7,node),NG_TBLEND(left,right,NG_TGT(value,NG_TSUB(cut,NG_TSET(256)))));
            node=NG_TBLEND(next,node,done);
        }
        NG_TSTORE(leaves+j,result);
    }
}
