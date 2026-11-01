/* NCF1: explicit predictor/model/entropy references; native wavefront payload.
 * The search policy is deliberately not part of the decoding contract. */
namespace choice_frame {
thread_local bool enabled=false;
thread_local bool packed_selectors=true;
// Optional predictor-bank x octree-depth -> entropy-bank map. Empty means
// ordinary sharing. Keep all16 difficulty bins; at most8 banks (128 contexts).
thread_local Bytes context_map;
// Independent entropy regions on a 2-voxel grid. Predictor choices retain
// their own grid; one region may therefore contain multiple predictors.
thread_local Bytes region_banks;
void write_regions(Bytes&header,const Bytes&map,unsigned d){
    choice_grid::Scope fine(2);adaptive_grid::Scope root(32);
    auto tree=adaptive_grid::serialize(map,d);auto data=pack_stream(tree,true);
    put32(header,tree.size());put32(header,data.size());packed_bank::append(header,data);
}
Bytes read_regions(packed_bank::Reader&h,unsigned d,unsigned banks){
    unsigned count=h.u(4);require(count<=2*size_t(d/2)*(d/2)*(d/2));
    auto data=h.take(h.u(4));Bytes tree(count);unpack_stream(data.data(),data.size(),tree,true);
    choice_grid::Scope fine(2);return adaptive_grid::deserialize(tree,d,32,banks);
}
// Scoped experiment only; returned frames are size placeholders, not decodable.
thread_local unsigned estimate_dimension=0;
struct Prepared {unsigned d=0;Bytes header,params,symbols,classes,complete;};
thread_local bool retain_estimates=false;
thread_local std::map<unsigned,Prepared> prepared;
Bytes finish_entropy(Bytes header,const Bytes&entropy,size_t n,size_t hn){
    header.insert(header.end(),entropy.begin(),entropy.begin()+hn);
    auto z=packed_bank::compress(header);Bytes result={'N','C',uint8_t(adaptive_grid::root?'A':'F'),'1'};
    put32(result,header.size());put32(result,z.size());packed_bank::append(result,z);
    result.insert(result.end(),entropy.begin()+hn,entropy.begin()+n);return result;
}
// 254 identifies packed IDs in the existing jointly Zstd-compressed header.
// 255 retains the legacy rANS representation; 0..59 remain uniform IDs.
Bytes pack_ids(const Bytes&ids,unsigned count){
    require(count&&count<=60);unsigned bits=0;while((1u<<bits)<count)++bits;
    Bytes out((ids.size()*bits+7)/8,0);
    for(size_t i=0;i<ids.size();i++){require(ids[i]<count);for(unsigned b=0;b<bits;b++)out[(i*bits+b)/8]|=((ids[i]>>b)&1u)<<((i*bits+b)%8);}
    return out;
}
Bytes unpack_ids(const Bytes&packed,size_t n,unsigned count){
    require(count&&count<=60);unsigned bits=0;while((1u<<bits)<count)++bits;
    require(packed.size()==(n*bits+7)/8);Bytes out(n);
    for(size_t i=0;i<n;i++){for(unsigned b=0;b<bits;b++)out[i]|=((packed[(i*bits+b)/8]>>((i*bits+b)%8))&1u)<<b;require(out[i]<count);}
    return out;
}
struct Choice { unsigned predictor,model,entropy; small_bank::Weight weights{}; };
constexpr unsigned no_model=255;
constexpr unsigned zero_predictor=12;
// Read-only experiment observer. No effect on search or production bytes.
using Observer=void(*)(unsigned,const std::vector<Bytes>&,const std::vector<Choice>&,
    const Bytes&,const Bytes&,const std::vector<uint16_t>&,const Bytes&,unsigned,const Bytes&);
thread_local Observer observer=nullptr;
Bytes write(unsigned d,const std::vector<Bytes>&models,const std::vector<Choice>&menu,
            const Bytes&selectors,const Bytes&symbols,const std::vector<uint16_t>&thresholds,
            const Bytes&classes,unsigned mode=1,const zero_blocks::Scan*zeros=nullptr,bool size_routed=false){
    if(zeros&&zeros->any()){
        auto extended=menu;auto selected=selectors;unsigned bank=0;
        for(auto&c:menu)bank=std::max(bank,c.entropy+1);
        require(bank<4&&extended.size()<60);unsigned id=extended.size();
        extended.push_back({zero_predictor,no_model,bank,{}});
        for(size_t b=0;b<selected.size();b++)if(zeros->empty[b])selected[b]=id;
        for(size_t i=0;i<symbols.size();i++)if(!zeros->mask[i])require(symbols[i]==0);
        return write(d,models,extended,selected,symbols,thresholds,classes,mode);
    }
    if(adaptive_grid::root&&(adaptive_grid::s2_options&2)&&!size_routed){
        // Four sizes (2/4/8/16), each with independent predictor-bank x
        // difficulty contexts. Keep the native 128-context ceiling: 4/8 bins
        // per bank rather than silently overflowing the native class field.
        require(adaptive_grid::root==16&&thresholds.size()==15);
        unsigned step=(adaptive_grid::s2_options&16)?2:4;
        Bytes cl=classes;for(auto&c:cl)c/=step;
        std::vector<uint16_t>thr;for(unsigned j=step-1;j<15;j+=step)thr.push_back(thresholds[j]);
        return write(d,models,menu,selectors,symbols,thr,cl,mode,nullptr,true);
    }
    if(adaptive_grid::root&&(adaptive_grid::s2_options&8)&&!size_routed){
        require(thresholds.size()==15);unsigned step=(adaptive_grid::s2_options&16)?2:4;
        Bytes cl=classes;for(auto&c:cl)c/=step;
        std::vector<uint16_t>thr;for(unsigned j=step-1;j<15;j+=step)thr.push_back(thresholds[j]);
        return write(d,models,menu,selectors,symbols,thr,cl,mode,nullptr,true);
    }
    unsigned K=thresholds.size()+1;
    require(models.size()<=4&&!menu.empty()&&menu.size()<=60&&K>=2&&K<=32&&mode<=1);
    bool size_banks=adaptive_grid::root&&(adaptive_grid::s2_options&2)&&size_routed;
    bool mapped=!context_map.empty(),regional=!region_banks.empty();require(!mapped||(!size_banks&&adaptive_grid::root==32));
    require(!regional||(!mapped&&!size_banks&&region_banks.size()==size_t(d/2)*(d/2)*(d/2)));
    Bytes leaf_levels;if(size_banks||mapped){leaf_levels.resize(selectors.size());adaptive_grid::serialize(selectors,d,&leaf_levels);}
    unsigned banks=1;for(auto&c:menu){require(c.predictor<=zero_predictor&&(c.model==no_model||c.model<models.size())&&c.entropy<16);if(c.predictor==zero_predictor)require(c.model==no_model&&c.weights==small_bank::Weight{});banks=std::max(banks,c.entropy+1);}if(mapped){require(context_map.size()==5*banks);banks=1+*std::max_element(context_map.begin(),context_map.end());}else if(size_banks)banks*=4;require(K*banks<=128);
    if(regional){banks=1+*std::max_element(region_banks.begin(),region_banks.end());require(K*banks<=128);}
    Bytes header;put32(header,d);put32(header,choice_grid::mode_word(mode|(size_banks?2:0)|(mapped?4:0)|(regional?8:0)));put32(header,models.size());put32(header,menu.size());put32(header,banks);
    for(auto&p:models){put32(header,p.size());packed_bank::append(header,p);}
    for(auto&c:menu){header.push_back(c.predictor);header.push_back(c.model);header.push_back(c.entropy);header.push_back(0);for(auto w:c.weights)put32(header,uint32_t(w));}
    if(mapped){put32(header,context_map.size());packed_bank::append(header,context_map);}
    require(selectors.size()==N/choice_grid::volume());
    const auto&encoded_selectors=selectors;require(!encoded_selectors.empty());
    bool uniform=std::all_of(encoded_selectors.begin(),encoded_selectors.end(),[&](auto c){return c==encoded_selectors[0];});
    bool packed=packed_selectors&&!adaptive_grid::root;
    bool raw_tree=adaptive_grid::root&&(adaptive_grid::s2_options&4);
    put32(header,uniform?encoded_selectors[0]:raw_tree?253:packed?254:255);
    if(!uniform){
        if(adaptive_grid::root){auto tree=adaptive_grid::serialize(selectors,d);put32(header,adaptive_grid::root);put32(header,tree.size());auto s=raw_tree?tree:pack_stream(tree,true);put32(header,s.size());packed_bank::append(header,s);}
        else {auto s=packed?pack_ids(encoded_selectors,menu.size()):pack_stream(encoded_selectors,true);put32(header,s.size());packed_bank::append(header,s);}
    }
    if(regional)write_regions(header,region_banks,d);
    if(uniform&&menu.at(encoded_selectors[0]).predictor==zero_predictor){
        auto z=packed_bank::compress(header);Bytes result={'N','C',uint8_t(adaptive_grid::root?'A':'F'),'1'};put32(result,header.size());put32(result,z.size());packed_bank::append(result,z);
        if(estimate_dimension==d&&retain_estimates)prepared[choice_grid::edge()]={d,{},{},{},{},result};
        if(observer)observer(d,models,menu,selectors,symbols,thresholds,classes,mode,result);return result;
    }
    Bytes par;int max_delta=-1;
    for(auto id:encoded_selectors){require(id<menu.size());auto m=menu[id].model;if(m==no_model)continue;
        const auto&p=models[m];require(p.size()>=6);int delta=std::max(int(int8_t(p[2])),int(int8_t(p[3])));
        if(delta>max_delta){max_delta=delta;par=p;}
    }
    if(par.empty()){par.resize(4096);size_t n=0;require(ng_bank_empty_model(par.data(),par.size(),&n)==0);par.resize(n);}
    // Retain native tap geometry, replace entropy thresholds and zero biases.
    require(par.size()>=6&&par[4]>=2&&par[4]<=64);
    unsigned tail=4*unsigned(par[4])-2;require(par.size()>=tail);
    par.resize(par.size()-tail);par[4]=K*banks;
    for(auto t:thresholds)packed_bank::put16(par,t);
    for(unsigned i=K-1;i<K*banks-1;i++)packed_bank::put16(par,0);
    for(unsigned i=0;i<K*banks;i++)packed_bank::put16(par,0);
    Bytes cls=classes;unsigned nb=d/choice_grid::edge();
    require(classes.size()==N);
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++){
        unsigned b=((z/choice_grid::edge())*nb+y/choice_grid::edge())*nb;
        for(unsigned x=0;x<d;x+=choice_grid::edge(),b++){
            auto s=selectors[b];require(s<menu.size());unsigned bank=K*(mapped?context_map[5*menu[s].entropy+leaf_levels[b]]:size_banks?4*menu[s].entropy+leaf_levels[b]:menu[s].entropy);
            size_t i=(size_t(z)*d+y)*d+x;
            for(unsigned j=0;j<choice_grid::edge();j++){require(cls[i+j]<K);cls[i+j]+=regional?K*region_banks[((size_t(z/2)*(d/2)+y/2)*(d/2)+(x+j)/2)]:bank;}
        }
    }
    Bytes entropy(3*N+131072);size_t n=0,hn=0;
    if(estimate_dimension==d){
        double payload=0;
        require(!subset_targets::enabled());
        if(retain_estimates)prepared[choice_grid::edge()]={d,header,par,symbols,cls,{}};
        require(ng_wave_estimate(par.data(),par.size(),symbols.data(),cls.data(),entropy.data(),entropy.size(),&hn,&payload)==0);
        header.insert(header.end(),entropy.begin(),entropy.begin()+hn);
        auto z=packed_bank::compress(header);
        Bytes result={'N','C','F','1'};put32(result,header.size());put32(result,z.size());packed_bank::append(result,z);
        result.resize(result.size()+size_t(std::ceil(payload)),0);return result;
    }
    if(subset_targets::enabled()){
        require(subset_targets::active.size()==N);
        require(ng_wave_encode_targets(par.data(),par.size(),symbols.data(),cls.data(),subset_targets::active.data(),entropy.data(),entropy.size(),&n)==0);
        if(subset_targets::verify)require(ng_wave_verify_targets(entropy.data(),n,symbols.data(),cls.data(),subset_targets::active.data())==0);
        if(std::all_of(subset_targets::active.begin(),subset_targets::active.end(),[](uint8_t v){return v!=0;})){
            Bytes control(entropy.size());size_t cn=0;
            require(ng_wave_encode(par.data(),par.size(),symbols.data(),cls.data(),control.data(),control.size(),&cn)==0);
            require(cn==n&&std::equal(control.begin(),control.begin()+cn,entropy.begin()));
        }
    }else require(ng_wave_encode(par.data(),par.size(),symbols.data(),cls.data(),entropy.data(),entropy.size(),&n)==0);
    require(ng_entropy_header_size(entropy.data(),n,&hn)==0);
    auto result=finish_entropy(std::move(header),entropy,n,hn);
    if(observer)observer(d,models,menu,selectors,symbols,thresholds,classes,mode,result);return result;
}
// Unused palette entries must not disable the shared-fit SIMD path.
void normalize_unused_weights(unsigned count,const bool*assigned,int32_t*weights){
    for(unsigned k=0;k<count;k++){
        unsigned ref=12*k;for(unsigned m=1;m<12;m++)if(assigned[12*k+m]){ref=12*k+m;break;}
        for(unsigned m=1;m<12;m++)if(!assigned[12*k+m])for(unsigned j=0;j<2;j++)weights[2*(12*k+m)+j]=weights[2*ref+j];
    }
}
std::vector<Bytes> read_models(packed_bank::Reader&h,unsigned nm){
    require(nm<=4);std::vector<Bytes>params;for(unsigned k=0;k<nm;k++)params.push_back(h.take(h.u(4)));return params;
}
int decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    require(src&&out&&n>=12&&(!memcmp(src,"NCF1",4)||!memcmp(src,"NCA1",4)));bool adaptive=src[2]=='A';Bytes prefix(src,src+12);size_t p=4;size_t hn=get32(prefix,p),zn=get32(prefix,p);require(zn<=n-12&&hn<=1024*1024);
    auto header=packed_bank::decompress(src+12,zn,hn);packed_bank::Reader h{header};unsigned d=h.u(4),mode=h.u(4),nm=h.u(4),nc=h.u(4),nbanks=h.u(4);
    choice_grid::Scope grid(choice_grid::wire_edge(mode));mode&=255;bool size_banks=mode&2,mapped=mode&4,regional=mode&8;
    require(ng_geometry_valid(d,d,d)&&cap>=size_t(d)*d*d&&(mode<=5||mode==8||mode==9)&&(!(size_banks||mapped)||adaptive)&&nm<=4&&nc>=1&&nc<=60&&nbanks>=1&&nbanks<=16);mode&=1;GeometryScope scope(d,d,d);
    auto params=read_models(h,nm);
    std::vector<Choice>menu;unsigned base_banks=0;for(unsigned k=0;k<nc;k++){Choice c;c.predictor=h.u(1);c.model=h.u(1);c.entropy=h.u(1);require(h.u(1)==0);for(auto&w:c.weights)w=int32_t(packed_bank::signed_value(h.u(4),32));require(c.predictor<=zero_predictor&&(c.model==no_model||c.model<nm)&&c.entropy<(regional?16:mapped?4:nbanks));base_banks=std::max(base_banks,c.entropy+1);if(c.predictor==zero_predictor)require(c.model==no_model&&c.weights==small_bank::Weight{});menu.push_back(c);}
    Bytes mapping;if(mapped){unsigned length=h.u(4);require(length==5*base_banks);mapping=h.take(length);for(auto c:mapping)require(c<nbanks);}
    unsigned singleton=h.u(4);Bytes selectors(N/choice_grid::volume());
    if(singleton==254){require(!adaptive);auto s=h.take(h.u(4));selectors=unpack_ids(s,selectors.size(),nc);}
    else if(singleton==255||singleton==253){
        require(singleton!=253||adaptive);
        if(adaptive){unsigned r=h.u(4),count=h.u(4);require(count<=2*selectors.size());auto s=h.take(h.u(4));Bytes tree(count);if(singleton==253){require(s.size()==count);tree=std::move(s);}else unpack_stream(s.data(),s.size(),tree,true);selectors=adaptive_grid::deserialize(tree,d,r,nc);}
        else {auto s=h.take(h.u(4));unpack_stream(s.data(),s.size(),selectors,true);}
    }else{require(singleton<nc);std::fill(selectors.begin(),selectors.end(),singleton);}
    Bytes regions;
    if(regional)regions=read_regions(h,d,nbanks);
    auto entropy=h.rest();
    if(singleton<nc&&menu[singleton].predictor==zero_predictor){require(entropy.empty()&&n==12+zn);memset(out,0,N);return 0;}
    require(entropy.size()>5);
    require(entropy[4]%nbanks==0);unsigned K=entropy[4]/nbanks;
    require(K>=2&&K<=32);
    bool used[60]={};for(auto c:selectors){require(c<nc);used[c]=true;}
    unsigned remap[4]={255,255,255,255};std::vector<Bytes>live;
    for(unsigned i=0;i<menu.size();i++){
        auto&c=menu[i];if(!used[i]){c.model=no_model;continue;}
        if(c.model!=no_model){unsigned m=c.model;if(remap[m]==255){remap[m]=live.size();live.push_back(params[m]);}c.model=remap[m];}
    }
    params=std::move(live);
    if(params.empty()){Bytes empty(4096);size_t en=0;require(ng_bank_empty_model(empty.data(),empty.size(),&en)==0);empty.resize(en);params.push_back(std::move(empty));}
    using State=std::unique_ptr<void,decltype(&ng_causal_destroy)>;std::vector<State>states;void*models[48]={};int32_t weights[96]={};
    for(unsigned k=0;k<params.size();k++){auto&q=params[k];states.emplace_back(k?ng_causal_create_model(q.data(),q.size()):ng_causal_create(q.data(),q.size()),ng_causal_destroy);require(bool(states.back()));for(unsigned m=0;m<12;m++)models[12*k+m]=states.back().get();}
    bool assigned[48]={};for(auto&c:menu)if(c.model!=no_model){unsigned id=12*c.model+c.predictor;for(unsigned j=0;j<2;j++){require(!assigned[id]||weights[2*id+j]==c.weights[j]);weights[2*id+j]=c.weights[j];}assigned[id]=true;}
    // Unreferenced palette entries must not falsely disable the shared-fit
    // SIMD path. Their coefficients are never consumed by a saved selector.
    normalize_unused_weights(params.size(),assigned,weights);
    Bytes canonical(selectors.size()),banks(selectors.size()),leaf_levels;
    if(size_banks||mapped){adaptive_grid::Scope tree(mapped?32:16);leaf_levels.resize(selectors.size());adaptive_grid::serialize(selectors,d,&leaf_levels);}
    for(size_t i=0;i<selectors.size();i++){require(selectors[i]<nc);auto&c=menu[selectors[i]];canonical[i]=c.predictor==zero_predictor?0:c.model==no_model?c.predictor:12+12*c.model+c.predictor;banks[i]=regional?0:mapped?mapping[5*c.entropy+leaf_levels[i]]:size_banks?4*c.entropy+leaf_levels[i]:c.entropy;require(banks[i]<nbanks);}
    if(regional){
        unsigned old_n=d/choice_grid::edge(),ratio=choice_grid::edge()/2,n2=d/2;Bytes expanded(regions.size());
        for(unsigned z=0;z<n2;z++)for(unsigned y=0;y<n2;y++)for(unsigned x=0;x<n2;x++)expanded[adaptive_grid::index(x,y,z,n2)]=canonical[adaptive_grid::index(x/ratio,y/ratio,z/ratio,old_n)];
        choice_grid::Scope fine(2);
        return ng_choice_wave_decode(models,12*params.size(),mode,weights,expanded.data(),nbanks>1?regions.data():nullptr,out,entropy.data(),entropy.size(),src+12+zn,n-12-zn,K);
    }
    return ng_choice_wave_decode(models,12*params.size(),mode,weights,canonical.data(),nbanks>1?banks.data():nullptr,out,entropy.data(),entropy.size(),src+12+zn,n-12-zn,K);
}
Bytes empty_frame(unsigned d){return write(d,{},{{zero_predictor,no_model,0,{}}},Bytes(N/choice_grid::volume()),{},std::vector<uint16_t>(15),{});}
}
extern "C" int ng_codec_decode(const uint8_t*s,size_t n,uint8_t*out,size_t cap){try{return choice_frame::decode(s,n,out,cap);}catch(...){return -1;}}
extern "C" void ng_codec_packed_selectors(unsigned enabled){choice_frame::packed_selectors=enabled!=0;}
// Signaling-only control: identical models, partition, entropy tables and
// residual payload. Only the selector representation and joint header change.
extern "C" int ng_codec_repack_s2_signal(const uint8_t*s,size_t n,unsigned raw,uint8_t*out,size_t cap,size_t*written){try{
    require(s&&out&&written&&raw<=1&&n>=12&&(!memcmp(s,"NCF1",4)||!memcmp(s,"NCA1",4)));
    Bytes result(s,s+n);
    if(s[2]=='A'){
        Bytes prefix(s,s+12);packed_bank::Reader p{prefix};p.skip(4);unsigned hn=p.u(4),zn=p.u(4);require(zn<=n-12);
        auto header=packed_bank::decompress(s+12,zn,hn);packed_bank::Reader h{header};h.skip(4);unsigned mode=h.u(4),nm=h.u(4),nc=h.u(4);h.skip(4);require(nm<=4&&nc<=60);
        for(unsigned k=0;k<nm;k++)h.skip(h.u(4));h.skip(nc*(4+4*small_bank::Weight{}.size()));
        if(mode&4)h.skip(h.u(4));
        size_t selector_start=h.p;unsigned tag=h.u(4);
        if(tag==253||tag==255){
            unsigned root=h.u(4),count=h.u(4);require(count<=1048576);auto encoded=h.take(h.u(4));Bytes tree(count);
            if(tag==253){require(encoded.size()==count);tree=std::move(encoded);}else unpack_stream(encoded.data(),encoded.size(),tree,true);
            Bytes next(header.begin(),header.begin()+selector_start);put32(next,raw?253:255);put32(next,root);put32(next,count);
            auto body=raw?tree:pack_stream(tree,true);put32(next,body.size());packed_bank::append(next,body);packed_bank::append(next,h.rest());
            auto z=packed_bank::compress(next);result={'N','C','A','1'};put32(result,next.size());put32(result,z.size());packed_bank::append(result,z);result.insert(result.end(),s+12+zn,s+n);
        }
    }
    require(result.size()<=cap);memcpy(out,result.data(),result.size());*written=result.size();return 0;
}catch(...){return -1;}}
/* Read-only saved-choice diagnostic. Four entries per palette member:
 * predictor,model,entropy,count. Model hashes expose changes across grids. */
extern "C" int ng_codec_choice_info(const uint8_t*s,size_t n,uint64_t*out){try{
    require(s&&out&&n>=12&&!memcmp(s,"NCF1",4));std::fill_n(out,256,0);
    Bytes prefix(s,s+12);size_t pos=4;unsigned hn=get32(prefix,pos),zn=get32(prefix,pos);require(zn<=n-12);
    auto header=packed_bank::decompress(s+12,zn,hn);packed_bank::Reader h{header};
    unsigned d=h.u(4),edge=choice_grid::wire_edge(h.u(4)),nm=h.u(4),nc=h.u(4);h.skip(4);
    require(nm<=4&&nc>=1&&nc<=60);choice_grid::Scope scope(edge);
    out[0]=d;out[1]=edge;out[2]=nm;out[3]=nc;
    for(unsigned m=0;m<nm;m++){auto p=h.take(h.u(4));uint64_t hash=14695981039346656037ull;for(auto b:p)hash=(hash^b)*1099511628211ull;out[8+m]=hash;}
    for(unsigned c=0;c<nc;c++){out[16+4*c]=h.u(1);out[17+4*c]=h.u(1);out[18+4*c]=h.u(1);h.skip(1+4*small_bank::Weight{}.size());}
    unsigned single=h.u(4);Bytes ids(choice_grid::count(d));
    if(single==254){auto p=h.take(h.u(4));ids=choice_frame::unpack_ids(p,ids.size(),nc);}
    else if(single==255){auto p=h.take(h.u(4));unpack_stream(p.data(),p.size(),ids,true);}
    else {require(single<nc);std::fill(ids.begin(),ids.end(),single);}
    out[4]=ids.size();for(auto c:ids){require(c<nc);out[19+4*c]++;if(out[16+4*c]==choice_frame::zero_predictor)out[5]++;}
    return 0;
}catch(...){return -1;}}
/* Read-only byte attribution. Header is jointly compressed, so its model,
 * selector and entropy-table contributions cannot be added independently. */
extern "C" int ng_codec_inspect_sizes(const uint8_t*s,size_t n,uint64_t*out){try{
    require(s&&out&&n>=12&&!memcmp(s,"NCF1",4));Bytes f(s,s+12);packed_bank::Reader p{f};p.skip(4);
    size_t hn=p.u(4),zn=p.u(4);require(zn<=n-12&&hn<=1024*1024);
    auto header=packed_bank::decompress(s+12,zn,hn);packed_bank::Reader h{header};h.skip(8);unsigned models=h.u(4);require(models<=4);h.skip(8);
    uint64_t model_bytes=0;for(unsigned k=0;k<models;k++){size_t m=h.u(4);h.skip(m);model_bytes+=m;}
    out[0]=n;out[1]=12+zn;out[2]=n-12-zn;out[3]=hn;out[4]=model_bytes;out[5]=models;return 0;
}catch(...){return -1;}}
/* Reframe historical model-bank winners without re-fitting or re-selection. */
extern "C" int ng_codec_repack_model_bank(const uint8_t*s,size_t n,uint8_t*out,size_t cap,size_t*written){
    try{
        require(out&&written);auto f=small_bank::read_frame(s,n,cap);GeometryScope scope(f.d,f.d,f.d);
        std::vector<choice_frame::Choice>menu;for(unsigned id=0;id<12+12*f.models.size();id++){
            unsigned m=id<12?255:(id-12)/12;menu.push_back({id%12,m,0,m==255||id%12==0?small_bank::Weight{}:f.weights[m]});
        }
        for(auto&v:f.raw)v=pyramid(v);std::vector<uint16_t>a(N);require(ng_activity_symbol_values(f.raw.data(),a.data())==0);
        auto frame=choice_frame::write(f.d,f.models,menu,f.selectors,f.raw,f.thr,activity_classify(a,f.thr));
        require(frame.size()<=cap);std::copy(frame.begin(),frame.end(),out);*written=frame.size();return 0;
    }catch(...){return -1;}
}
