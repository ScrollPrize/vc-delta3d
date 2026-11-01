// policy0: fixed-size screening; policy8/16/32: adaptive root with2³ leaves.
// All candidate fitting, residual coding, threshold fitting and payload writing
// stay in the production Fast/Slow engines.
namespace grid_experiment {
thread_local double screen_bytes[5]={};
thread_local unsigned chosen=4;
// Shared metadata attribution for both screening experiments. Does not change
// any encoder decision or frame. Joint-zstd attribution remains approximate.
void size_components(const Bytes&frame,size_t bytes,double scale,double*out){
    require(bytes>=12&&!memcmp(frame.data(),"NCF1",4));
    Bytes prefix(frame.begin(),frame.begin()+12);size_t p=4;
    unsigned hn=get32(prefix,p),zn=get32(prefix,p);require(12+zn<=bytes);
    auto header=packed_bank::decompress(frame.data()+12,zn,hn);
    packed_bank::Reader h{header};h.skip(8);unsigned nm=h.u(4),nc=h.u(4);h.skip(4);
    for(unsigned m=0;m<nm;m++)h.skip(h.u(4));
    h.skip(nc*(4+4*small_bank::Weight{}.size()));
    unsigned singleton=h.u(4);double selector=0;
    if(singleton==255||singleton==254){
        size_t begin=h.p;h.skip(h.u(4));auto without=header;
        without.erase(without.begin()+begin,without.begin()+h.p);
        selector=std::max(0.,double(zn)-packed_bank::compress(without).size());
    }
    out[0]=bytes*scale;out[1]=(bytes-12-zn)*scale;
    out[2]=selector*scale;out[3]=12+zn-selector;
}
}
// Read-only storage experiment: replace only the selector representation in
// an existing frame's joint-zstd header. No predictor search or residual encode.
// out: current/raw/bitpacked total bytes, corresponding compressed header
// sizes, selector count, old rANS bytes. New representations include a byte tag.
extern "C" int ng_selector_storage_trial(const uint8_t*src,size_t size,uint64_t*out){
    try{
        require(src&&out&&size>=12&&!memcmp(src,"NCF1",4));
        Bytes prefix(src,src+12);size_t p=4;
        unsigned hn=get32(prefix,p),zn=get32(prefix,p);require(zn<=size-12);
        auto header=packed_bank::decompress(src+12,zn,hn);
        packed_bank::Reader h{header};unsigned d=h.u(4),edge=choice_grid::wire_edge(h.u(4));
        choice_grid::Scope grid(edge);
        unsigned nm=h.u(4),nc=h.u(4);h.skip(4);require(nc&&nc<=60);
        for(unsigned m=0;m<nm;m++)h.skip(h.u(4));
        h.skip(nc*(4+4*small_bank::Weight{}.size()));
        unsigned singleton=h.u(4);size_t begin=h.p,old_bytes=0;
        Bytes choices;
        if(singleton==255||singleton==254){
            auto packed=h.take(h.u(4));old_bytes=packed.size();choices.resize(choice_grid::count(d));
            if(singleton==254)choices=choice_frame::unpack_ids(packed,choices.size(),nc);
            else unpack_stream(packed.data(),packed.size(),choices,true);
            for(auto id:choices)require(id<nc);
        }else require(singleton<nc);
        out[6]=choices.size();out[7]=old_bytes;
        for(unsigned variant=0;variant<3;variant++){
            Bytes changed=header;
            if(variant&&!choices.empty()){
                Bytes encoded;
                if(variant==1)encoded=choices;
                else encoded=choice_frame::pack_ids(choices,nc);
                Bytes replacement{uint8_t(variant)};put32(replacement,encoded.size());packed_bank::append(replacement,encoded);
                changed.erase(changed.begin()+begin,changed.begin()+h.p);
                changed.insert(changed.begin()+begin,replacement.begin(),replacement.end());
            }
            auto compressed=packed_bank::compress(changed);
            auto restored=packed_bank::decompress(compressed,changed.size());require(restored==changed);
            if(variant&&!choices.empty()){
                packed_bank::Reader r{restored};r.skip(begin);require(r.u(1)==variant);auto encoded=r.take(r.u(4));
                Bytes decoded(choices.size());
                if(variant==1)decoded=encoded;
                else decoded=choice_frame::unpack_ids(encoded,choices.size(),nc);
                require(decoded==choices);
                require(std::equal(restored.begin()+r.p,restored.end(),header.begin()+h.p,header.end()));
            }
            out[variant]=size-zn+compressed.size();out[3+variant]=compressed.size();
        }
        return 0;
    }catch(...){return -1;}
}
// Diagnostic fixed-size screener only. Each sample is a contiguous cube;
// no stitched neighborhoods, shared-grid selector, or altered fitting rules.
// Output per edge2/4/8/16/32: scaled total, scaled payload, scaled selector
// marginal compressed bytes, remaining metadata (counted once).
extern "C" int ng_codec_screen_fixed_samples(const uint8_t*s,unsigned d,unsigned preset,unsigned side,double*scores){
    try{
        require(s&&scores&&(preset==1||preset==3)&&(side==32||side==64)&&d>=side);
        adaptive_grid::Scope adaptive(0);
        std::fill_n(scores,20,0.);
        unsigned count=d==side?1:(side==32?8:2);
        Bytes sample(size_t(side)*side*side),frame(sample.size()*4+131072);
        double scale=double(d)*d*d/sample.size();
        for(unsigned q=0;q<count;q++){
            unsigned origin[3];
            for(unsigned axis=0;axis<3;axis++){
                unsigned half=side==32?((q>>axis)&1):q;
                origin[axis]=count==1?0:(half?(3*d/4-side/2):(d/4-side/2));
            }
            for(unsigned z=0;z<side;z++)for(unsigned y=0;y<side;y++)
                memcpy(sample.data()+(size_t(z)*side+y)*side,
                       s+(size_t(z+origin[2])*d+y+origin[1])*d+origin[0],side);
            unsigned j=0;
            for(unsigned edge:{2u,4u,8u,16u,32u}){
                choice_grid::Scope grid(edge);size_t bytes=0;
                require(ng_codec_encode(sample.data(),side,preset,frame.data(),frame.size(),&bytes)==0);
                double parts[4];grid_experiment::size_components(frame,bytes,scale,parts);
                for(unsigned k=0;k<4;k++)scores[4*j+k]+=parts[k]/count;++j;
            }
        }
        return 0;
    }catch(...){return -1;}
}
extern "C" void ng_grid_experiment_stats(double*s,uint64_t*l,unsigned*c){
    std::copy_n(grid_experiment::screen_bytes,5,s);std::copy_n(adaptive_grid::leaves,6,l);*c=grid_experiment::chosen;
}
extern "C" int ng_codec_encode_grid(const uint8_t*s,unsigned d,unsigned preset,unsigned policy,uint8_t*out,size_t cap,size_t*n){
    try{
        require(preset==1||preset==3);unsigned root=policy&255;require((policy<512||policy==512)&&(root==0||root==8||root==16||root==32));
        std::fill_n(grid_experiment::screen_bytes,5,0.);std::fill_n(adaptive_grid::leaves,6,0);
        if(root){choice_grid::Scope grid(2);adaptive_grid::Scope adaptive(root,policy&256?2:8);grid_experiment::chosen=2;
            return ng_codec_encode(s,d,preset,out,cap,n);}
        // Use the same distributed proxy as existing predictor screening.
        // Fit stride4 on32³ matches existing screen fitting, not full128³'s
        // sparse stride128. Rank complete proxy frames; no full-chunk trials.
        Bytes proxy;{choice_grid::Scope grid(4);GeometryScope geometry(d,d,d);zero_blocks::Scan zeros(s,d,true);proxy=fast_split::screen_proxy(s,d,&zeros);}
        unsigned sd=std::min(d,32u),best=4;double score=double(SIZE_MAX);
        Bytes tmp(size_t(sd)*sd*sd*4+131072);unsigned j=0;
        adaptive_grid::Scope adaptive(0);
        for(unsigned e:{2u,4u,8u,16u,32u}){
            choice_grid::Scope grid(e);size_t bytes=0;int cfg[6]={3,3,1,16,40,4};
            bool old=choice_frame::enabled;choice_frame::enabled=true;
            int rc=preset==3?ng_screen_pair_encode_impl(proxy.data(),sd,12,tmp.data(),tmp.size(),&bytes,cfg)
                :three_candidate_encode_impl(proxy.data(),sd,1,7,1,0,0,tmp.data(),tmp.size(),&bytes,0,0,0,0,false,cfg);
            choice_frame::enabled=old;require(rc==0);double value=bytes;
            if(policy==512){
                double parts[4];grid_experiment::size_components(tmp,bytes,double(d)*d*d/(size_t(sd)*sd*sd),parts);
                value=parts[1]+parts[2]+parts[3];
            }
            grid_experiment::screen_bytes[j++]=value;
            // Preserve the old tie rule in both controls: isolate score only.
            if(value<score){score=value;best=e;}
        }
        grid_experiment::chosen=best;choice_grid::Scope grid(best);return ng_codec_encode(s,d,preset,out,cap,n);
    }catch(...){return -1;}
}
