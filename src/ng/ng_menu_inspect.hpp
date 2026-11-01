/* Read-only diagnostic for saved NMB1 and N3A1 frames. Uses existing parsers
 * and selector decoder; no prediction, fitting, or residual re-encoding. */
extern "C" int ng_menu_inspect(const uint8_t*src,size_t n,uint64_t*out){
    try{
        require(src&&out&&n>=12);bool old=!memcmp(src,"NMB1",4);require(old||!memcmp(src,"N3A1",4));
        std::fill_n(out,96,0);Bytes f(src,src+n);packed_bank::Reader body{f};body.skip(4);
        size_t hn=body.u(4),zn=body.u(4);auto header=packed_bank::decompress(body.take(zn),hn);packed_bank::Reader h{header};
        unsigned d=h.u(4);require(ng_geometry_valid(d,d,d));size_t voxels=size_t(d)*d*d;
        out[0]=n;out[1]=hn;out[2]=zn;out[3]=n-body.p;unsigned ids[3]={};Bytes choices;
        auto model=[&](){size_t pn=h.u(4);h.skip(pn);h.skip(8);out[4]++;out[5]+=pn+8;};
        if(old){unsigned banks=h.u(4);require(banks>=1&&banks<=4);for(unsigned k=0;k<banks;k++)model();
            h.skip(30);size_t hp=h.p,bp=body.p;choices=small_bank::read_stream(h,body,voxels/64);
            out[6]=h.p-hp-12+body.p-bp;out[7]=body.p-bp;out[10]=16;
            for(auto c:choices)require(c<12+12*banks);
        }else{
            for(unsigned k=0;k<3;k++){unsigned id=h.u(4);require(id<24);ids[k]=id<12?id:12+12*k+id%12;if(id>=12)model();}
            size_t sn=h.u(4);auto sel=h.take(sn);choices.resize(voxels/64);unpack_stream(sel.data(),sel.size(),choices,true);out[6]=sn;
            for(auto&c:choices){require(c<3);c=ids[c];}h.skip(30);out[10]=48;
        }
        require(choices.size()==voxels/64);
        size_t total=0;
        for(unsigned c=0;c<out[10];c++){
            size_t count=h.u(4),tn=h.u(4),bn=h.u(4);require(count<=voxels-total);total+=count;
            h.skip(tn);body.skip(bn);out[8]+=bn;out[9]+=tn;
        }
        require(total==voxels);h.end();body.end();
        for(auto c:choices){require(c<60);out[32+c]++;out[c>=12?12:13]++;}
        for(unsigned c=0;c<60;c++)out[11]+=out[32+c]>0;
        require(n==12+zn+out[7]+out[8]);return 0;
    }catch(...){return -1;}
}
