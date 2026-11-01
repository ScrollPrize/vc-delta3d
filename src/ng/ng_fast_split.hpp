/* Residual-energy initialization + freely reassigned model families. Reuses
 * masked fitter, bounded proposal search, wavefront entropy and SIMD decode. */
namespace fast_split {
using packed_bank::Reader;
using packed_bank::append;
thread_local double search_trace[96]={};
thread_local std::vector<Bytes> nested_frames;
// Diagnostic IDs0..11 bare,12..23 NG-input,24 plain NG.
thread_local std::vector<double> menu_scores;
// Stream recipe: fixed menu identity, fresh per-chunk fits and entropy tables.
// Empty preserves every existing preset. IDs match diagnostic menu IDs.
thread_local const std::vector<unsigned>*stream_menu=nullptr;
thread_local bool stream_onepass=false;
// Opt-in RC stream experiment. Only final entropy thresholds are reused;
// symbol counts and serialized probabilities always belong to this chunk.
thread_local const std::vector<uint16_t>*stream_thresholds=nullptr;
thread_local unsigned stream_threshold_mode=0; // 1 frozen, 2 local refinement
// Encoder-only input to the ordinary Fast path: one predictor for the chunk.
// -1 retains Fast's own screener. Never enables the nested/block-menu path.
thread_local int forced_uniform=-1;
thread_local int extra_uniform=-1;
thread_local int extra_uniform_second=-1;
thread_local unsigned last_uniform=0;
Bytes screen_proxy(const uint8_t*src,unsigned d,const zero_blocks::Scan*zeros=nullptr){
    unsigned side=std::min(d,32u);Bytes proxy(size_t(side)*side*side);
    if(d<=32){std::copy_n(src,proxy.size(),proxy.data());return proxy;}
    std::array<std::array<unsigned,3>,8> origins;unsigned nb=d/choice_grid::edge();
    std::vector<size_t>occupied;
    if(zeros&&zeros->any())for(size_t b=0;b<zeros->empty.size();b++)if(!zeros->empty[b])occupied.push_back(b);
    for(unsigned k=0;k<8;k++){
        auto&o=origins[k];o={(2*(k/4)+1)*d/4-8,(2*((k/2)%2)+1)*d/4-8,(2*(k%2)+1)*d/4-8};
        if(zeros&&zeros->any()){
            bool has_signal=false;
            // Coarse16/32 grids can straddle the16³ sample's endpoints.
            // Include every intersecting occupancy cell, not just full cells.
            for(unsigned z=o[0]/choice_grid::edge();z<(o[0]+16+choice_grid::edge()-1)/choice_grid::edge();z++)for(unsigned y=o[1]/choice_grid::edge();y<(o[1]+16+choice_grid::edge()-1)/choice_grid::edge();y++)for(unsigned x=o[2]/choice_grid::edge();x<(o[2]+16+choice_grid::edge()-1)/choice_grid::edge();x++)
                has_signal|=!zeros->empty[(size_t(z)*nb+y)*nb+x];
            if(!has_signal&&!occupied.empty()){
                size_t b=occupied[std::min(occupied.size()-1,(2*k+1)*occupied.size()/16)];
                unsigned coords[3]={unsigned(b/(nb*nb))*choice_grid::edge(),unsigned((b/nb)%nb)*choice_grid::edge(),unsigned(b%nb)*choice_grid::edge()};
                for(unsigned j=0;j<3;j++)o[j]=unsigned(std::clamp(int(coords[j])-4,0,int(d)-16));
            }
        }
    }
    for(unsigned z=0;z<32;z++)for(unsigned y=0;y<32;y++)for(unsigned x=0;x<32;x++){
        auto&o=origins[(z/16)*4+(y/16)*2+x/16];
        unsigned sz=o[0]+z%16,sy=o[1]+y%16,sx=o[2]+x%16;
        proxy[(z*32+y)*32+x]=src[(size_t(sz)*d+sy)*d+sx];
    }
    // Occupied coarse blocks can still contain empty16³ interiors. Do not
    // feed an empty proxy to the histogram/fitter for a nonempty source.
    if(std::none_of(proxy.begin(),proxy.end(),[](uint8_t v){return v!=0;})){
        const size_t total=size_t(d)*d*d;size_t i=0;while(i<total&&!src[i])++i;
        if(i<total){
            unsigned p[3]={unsigned(i/(d*d)),unsigned((i/d)%d),unsigned(i%d)};
            for(auto&v:p)v=unsigned(std::clamp(int(v)-8,0,int(d)-16));
            for(unsigned z=0;z<16;z++)for(unsigned y=0;y<16;y++)
                std::copy_n(src+(size_t(p[0]+z)*d+p[1]+y)*d+p[2],16,proxy.data()+(size_t(z)*32+y)*32);
        }
    }
    return proxy;
}
struct Samples {
    std::vector<size_t> index;
    std::vector<uint32_t> block;
    size_t blocks=0;
    explicit Samples(unsigned d){
        if(choice_grid::edge()>std::min(16u,d/2)){
            // A32³ selector cannot be represented by independent16³ tiles.
            // This branch is only used by the new coarse-grid experiments.
            blocks=choice_grid::count(d);index.resize(size_t(d)*d*d);block.resize(index.size());
            for(size_t i=0;i<index.size();i++){index[i]=i;block[i]=choice_grid::block(i,d);}return;
        }
        const unsigned side=std::min(16u,d/2),nb=side/choice_grid::edge();
        for(unsigned zp=0;zp<2;zp++)for(unsigned yp=0;yp<2;yp++)for(unsigned xp=0;xp<2;xp++){
            unsigned z0=(2*zp+1)*d/4-side/2,y0=(2*yp+1)*d/4-side/2,x0=(2*xp+1)*d/4-side/2;
            for(unsigned z=0;z<side;z++)for(unsigned y=0;y<side;y++)for(unsigned x=0;x<side;x++){
                index.push_back((size_t(z+z0)*d+y+y0)*d+x+x0);
                block.push_back(blocks+((z/choice_grid::edge())*nb+y/choice_grid::edge())*nb+x/choice_grid::edge());
            }
            blocks+=size_t(nb)*nb*nb;
        }
    }
    template<class T> std::vector<T> take(const std::vector<T>&v)const{
        std::vector<T> out;out.reserve(index.size());for(auto i:index)out.push_back(v[i]);return out;
    }
};
struct Evidence {Bytes symbols,classes;std::vector<uint16_t> thresholds;};
// Screen uniform starting assignments on eight distributed 16^3 interiors.
// Activity is computed on the full causal stream before sampling: no artificial
// tile boundaries. Each candidate gets its own thresholds and histogram.
std::vector<unsigned> screen(const std::vector<Bytes>&candidate,unsigned d,unsigned count=13,std::vector<Evidence>*cache=nullptr,unsigned K=16,const uint8_t*mask=nullptr){
    std::vector<std::pair<double,unsigned>> ranked;
    std::vector<uint16_t> activity(N);
    Samples sample(d);
    if(cache)cache->resize(count);
    for(unsigned m=0;m<count;m++){
        std::array<uint32_t,2560> activity_hist;
        require(ng_activity_symbol_histogram(candidate[m].data(),activity.data(),activity_hist.data())==0);
        if(mask){activity_hist.fill(0);for(size_t i=0;i<N;i++)if(mask[i])activity_hist[activity[i]]++;}
        Evidence evidence;
        evidence.thresholds=activity_quantiles_histogram(activity_hist,K);
        evidence.symbols=sample.take(candidate[m]);
        evidence.classes=activity_classify(sample.take(activity),evidence.thresholds);
        std::vector<Histogram>hist(K);size_t sampled=0;
        for(size_t i=0;i<sample.index.size();i++){
                if(!mask||mask[sample.index[i]]){hist[evidence.classes[i]][evidence.symbols[i]]++;sampled++;}
        }
        double cost=0;for(auto&h:hist)cost+=histogram_cost(h.data(),12,true);
        search_trace[m]=cost/sampled;ranked.emplace_back(cost,m);
        if(cache)(*cache)[m]=std::move(evidence);
    }
    std::sort(ranked.begin(),ranked.end());
    search_trace[13]=ranked[0].second;search_trace[14]=ranked[1].second;
    return {ranked[0].second,ranked[1].second};
}
void regional_reselect(unsigned d,const Bytes&symbols,const Bytes&classes,
        const std::vector<const uint8_t*>&candidate,Bytes&selected){
    const unsigned side=16,nr=d/side,nb=d/4;std::vector<uint32_t>region(N);Bytes choices(size_t(nr)*nr*nr);
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++)
        region[(size_t(z)*d+y)*d+x]=((z/side)*nr+y/side)*nr+x/side;
    for(unsigned z=0;z<nr;z++)for(unsigned y=0;y<nr;y++)for(unsigned x=0;x<nr;x++)
        choices[(z*nr+y)*nr+x]=selected[((z*4)*nb+y*4)*nb+x*4];
    activity_reselect(symbols,classes,region,candidate,choices,1);
    for(unsigned z=0;z<nb;z++)for(unsigned y=0;y<nb;y++)for(unsigned x=0;x<nb;x++)
        selected[(z*nb+y)*nb+x]=choices[((z/4)*nr+y/4)*nr+x/4];
}
struct Frame {
    unsigned d,banks;std::vector<Bytes> params;
    std::vector<small_bank::Weight> weights;Bytes choices,entropy;
    const uint8_t*payload;size_t payload_size;
};
Frame read(const uint8_t*src,size_t n){
    require(src&&n>=12&&!memcmp(src,"NFS1",4));Bytes prefix(src,src+12);Reader r{prefix};r.skip(4);size_t decoded=r.u(4),stored=r.u(4);require(stored<=n-12);
    auto meta=packed_bank::decompress(src+12,stored,decoded);Reader h{meta};Frame f;f.d=h.u(4);f.banks=h.u(4);require(ng_geometry_valid(f.d,f.d,f.d)&&f.banks>=1&&f.banks<=2);
    f.params.resize(f.banks);f.weights.resize(f.banks);
    for(unsigned k=0;k<f.banks;k++){f.params[k]=h.take(h.u(4));for(auto&w:f.weights[k])w=int32_t(packed_bank::signed_value(h.u(4),32));}
    auto selector=h.take(h.u(4));f.choices.resize(size_t(f.d/4)*(f.d/4)*(f.d/4));unpack_stream(selector.data(),selector.size(),f.choices,true);
    for(auto c:f.choices)require(c<12+12*f.banks);
    f.entropy=h.rest();f.payload=src+12+stored;f.payload_size=n-12-stored;return f;
}
Bytes frame(unsigned d,const std::vector<Bytes>&params,const std::vector<small_bank::Weight>&weights,const Bytes&sel,const Bytes&symbols,const std::vector<uint16_t>&thr,const Bytes&classes,const uint16_t*activity=nullptr,const zero_blocks::Scan*zeros=nullptr,bool separate=false){
    if(choice_frame::enabled){
        std::vector<choice_frame::Choice> menu;Bytes mapped(sel.size());
        unsigned remap[60];std::fill_n(remap,60,255);
        for(size_t i=0;i<sel.size();i++){if(separate&&zeros&&zeros->any()&&zeros->empty[i]){mapped[i]=0;continue;}unsigned id=sel[i];require(id<60);if(remap[id]==255){remap[id]=menu.size();unsigned m=id<12?255:(id-12)/12;menu.push_back({id%12,m,separate?unsigned(menu.size()):0,m==255||id%12==0?small_bank::Weight{}:weights[m]});}mapped[i]=remap[id];}
        auto cl=activity?activity_classify(std::vector<uint16_t>(activity,activity+N),thr):classes;
        return choice_frame::write(d,params,menu,mapped,symbols,thr,cl,1,zeros);
    }
    Bytes metadata;put32(metadata,d);put32(metadata,params.size());
    for(unsigned k=0;k<params.size();k++){put32(metadata,params[k].size());append(metadata,params[k]);for(auto w:weights[k])put32(metadata,uint32_t(w));}
    auto selectors=pack_stream(sel,true);put32(metadata,selectors.size());append(metadata,selectors);
    auto par=params[0];int16_t bias[16]={};require(ng_set_classes(par.data(),par.size(),thr.data(),bias)==0);
    const size_t entropy_cap=3*symbols.size()+65536;
    auto entropy=std::make_unique_for_overwrite<uint8_t[]>(entropy_cap);size_t en=0,hn=0;
    if(bank_encode_profile_enabled)ng_set_profile_callback(bank_entropy_profile_tick);
    int encoded=activity?ng_wave_encode_activity(par.data(),par.size(),symbols.data(),activity,entropy.get(),entropy_cap,&en)
        :ng_wave_encode(par.data(),par.size(),symbols.data(),classes.data(),entropy.get(),entropy_cap,&en);
    if(bank_encode_profile_enabled){bank_entropy_profile_tick(15);ng_set_profile_callback(nullptr);}
    require(encoded==0);
    require(ng_entropy_header_size(entropy.get(),en,&hn)==0);metadata.insert(metadata.end(),entropy.get(),entropy.get()+hn);
    auto z=packed_bank::compress(metadata);Bytes result={'N','F','S','1'};result.reserve(12+z.size()+en-hn);
    put32(result,metadata.size());put32(result,z.size());append(result,z);result.insert(result.end(),entropy.get()+hn,entropy.get()+en);return result;
}
}
static int ng_fast_split_encode_impl(const uint8_t*src,unsigned d,unsigned banks,unsigned rounds,unsigned sweeps,bool histogram,bool exhaustive,uint8_t*out,size_t cap,size_t*written,unsigned policy=0){
    try{
        require(src&&out&&written&&banks>=1&&banks<=2&&rounds>=1&&rounds<=3&&sweeps>=1&&sweeps<=3&&ng_geometry_valid(d,d,d));GeometryScope geometry(d,d,d);
        const unsigned count=12*banks,P=12+count,nb=d/4;const size_t blocks=size_t(nb)*nb*nb;
        const int cfg[6]={3,3,1,16,40,128};
        std::vector<Bytes> candidate(P,Bytes(N)),params(banks);std::vector<small_bank::Weight>weights(banks);
        {BankEncodeTimer timer(0);bank_fixed_candidates(src,candidate,d);}Bytes initial(65536),cl(N),membership(N),feature(N),symbols(N),fixed(N);std::vector<uint16_t> pre(N),activity(N);size_t pn=0;
        {BankEncodeTimer timer(1);require(ng_ex_prepare(src,cfg,0,0,initial.data(),initial.size(),&pn,candidate[12].data(),cl.data(),pre.data())==0);}initial.resize(pn);
        std::vector<uint32_t>block(N);std::vector<uint64_t>energy(blocks);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;block[i]=((z/4)*nb+y/4)*nb+x/4;energy[block[i]]+=(candidate[12][i]+1u)/2;
        }
        auto ordered=energy;std::nth_element(ordered.begin(),ordered.begin()+blocks/2,ordered.end());Bytes groups(blocks);
        for(size_t b=0;b<blocks;b++)groups[b]=banks==2&&(energy[b]>ordered[blocks/2]||(energy[b]==ordered[blocks/2]&&(b&1)));
        std::fill(fast_split::search_trace,fast_split::search_trace+96,0.);
        std::vector<unsigned> starts;
        {BankEncodeTimer timer(7);starts=policy?fast_split::screen(candidate,d):std::vector<unsigned>{12};}
        if(policy>=4)starts.resize(1);
        Bytes screen_symbols=policy>=4?candidate[starts[0]]:Bytes{};
        Bytes selected(blocks,12);ShortlistHistory initial_history;
        {BankEncodeTimer timer(6);
            if(policy)std::fill(selected.begin(),selected.end(),starts[0]);
            else if(histogram){std::vector<const uint8_t*> ptr(P);for(unsigned m=0;m<P;m++)ptr[m]=candidate[m<12?m:12].data();selected=choose_blocks(ptr.data(),P,cl.data(),16,4);}
            else bank_shortlist_select(src,d,3,candidate[12],cl,selected,fixed,&initial_history);
        }
        for(size_t b=0;b<blocks;b++)if(selected[b]>=12)selected[b]=12+(selected[b]-12)%12+12*groups[b];
        std::vector<uint16_t> thr(15);size_t at=initial.size()-62;for(auto&t:thr){t=initial[at]|unsigned(initial[at+1])<<8;at+=2;}
        Bytes best;const auto initial_thr=thr;
        for(auto start:starts){
        ShortlistHistory history;
        if(policy){std::fill(selected.begin(),selected.end(),start);thr=initial_thr;}
        for(unsigned round=0;round<rounds;round++){
            for(size_t i=0;i<N;i++){unsigned c=selected[block[i]],m=c<12?c:(c-12)%12;feature[i]=uint8_t(src[i]-unpyramid(candidate[m][i]));}
            std::vector<small_bank::Fit> fitted;fitted.reserve(banks);
            for(unsigned k=0;k<banks;k++){
                size_t members=0;
                for(size_t i=0;i<N;i++){unsigned c=selected[block[i]],g=round&&c>=12?(c-12)/12:groups[block[i]];membership[i]=banks==1||g==k;members+=membership[i];}
                {BankEncodeTimer timer(1);fitted.push_back(small_bank::fit(src,d,cfg,feature,members>=1024?membership.data():nullptr,candidate));}
                params[k]=fitted.back().params;weights[k]=fitted.back().weights;
            }
            std::vector<const uint8_t*> pp(count),cp(P);std::vector<size_t> sizes(count);std::vector<const uint16_t*> bp(count);std::vector<uint8_t*> wp(P);
            for(unsigned k=0;k<banks;k++)for(unsigned m=0;m<12;m++){unsigned id=12*k+m;pp[id]=params[k].data();sizes[id]=params[k].size();bp[id]=fitted[k].base[m].data();}
            for(unsigned m=0;m<P;m++){cp[m]=candidate[m].data();wp[m]=candidate[m].data();}
            auto retune=[&](){
                Bytes raw(N);for(size_t i=0;i<N;i++)raw[i]=unpyramid(symbols[i]);
                {BankEncodeTimer timer(5);
                ActivityHistogram hist(raw,activity,true);auto quant=activity_quantiles(activity,16);
                if(hist.estimate(quant)<hist.estimate(thr))thr=quant;
                auto tuned=activity_refine(raw,activity,thr,block,selected,1);if(hist.estimate(tuned)<hist.estimate(thr))thr=tuned;
                }
            };
            auto checkpoint=[&](){
                {BankEncodeTimer timer(3);require(ng_bank_symbols(count,pp.data(),sizes.data(),bp.data(),selected.data(),src,wp.data(),symbols.data(),activity.data())==0);}
                retune();
                Bytes encoded;{BankEncodeTimer timer(4);encoded=fast_split::frame(d,params,weights,selected,symbols,thr,activity_classify(activity,thr));}
                size_t size=encoded.size();if(best.empty()||size<best.size())best=std::move(encoded);
                unsigned n=unsigned(fast_split::search_trace[15]);
                if(n<40){fast_split::search_trace[16+2*n]=size;fast_split::search_trace[17+2*n]=best.size();fast_split::search_trace[15]++;}
                return size;
            };
            if(policy>=4){
                // One screen winner; fixed NG fits; exactly the requested
                // number of selection passes. No starting-frame/trial encodes.
                {BankEncodeTimer timer(3);require(ng_bank_symbols(count,pp.data(),sizes.data(),bp.data(),selected.data(),src,wp.data(),symbols.data(),activity.data())==0);}
                symbols=screen_symbols;
                require(ng_activity_symbol_values(symbols.data(),activity.data())==0);
                thr=activity_quantiles(activity,16);retune();
                for(unsigned pass=0;pass<sweeps;pass++){
                    cl=activity_classify(activity,thr);
                    {BankEncodeTimer timer(6);
                    if(banks==2)bank_shortlist_select<true,true,2>(src,d,3,candidate[12],cl,selected,fixed,&history,true,&symbols,nullptr,nullptr,cp.data());
                    else bank_shortlist_select<true,true>(src,d,3,candidate[12],cl,selected,fixed,&history,true,&symbols,nullptr,nullptr,cp.data());
                    }
                    {BankEncodeTimer timer(3);
                    auto emit=pass+1==sweeps&&(policy==4||policy==6)?ng_bank_selected_symbols:ng_bank_symbols;
                    require(emit(count,pp.data(),sizes.data(),bp.data(),selected.data(),src,wp.data(),symbols.data(),activity.data())==0);
                    }
                    retune();
                }
                {BankEncodeTimer timer(4);best=fast_split::frame(d,params,weights,selected,symbols,thr,activity_classify(activity,thr));}
                fast_split::search_trace[15]=1;fast_split::search_trace[16]=best.size();fast_split::search_trace[17]=best.size();
                fast_split::search_trace[90]=sweeps;fast_split::search_trace[91]=1;fast_split::search_trace[92]=1;
                continue;
            }
            if(policy){
                size_t accepted=checkpoint();
                if(policy==1)continue;
                for(unsigned step=0;step<sweeps;step++){
                    bool improved=false;
                    for(unsigned move=0;move<(policy==3?2u:1u);move++){
                        // Restore/recompute causal history even after a rejected move.
                        require(ng_bank_symbols(count,pp.data(),sizes.data(),bp.data(),selected.data(),src,wp.data(),symbols.data(),activity.data())==0);
                        cl=activity_classify(activity,thr);auto previous=selected;auto previous_thr=thr;
                        if(move)fast_split::regional_reselect(d,symbols,cl,cp,selected);
                        else bank_shortlist_select<true,true,2>(src,d,3,candidate[12],cl,selected,fixed,&history,true,&symbols,nullptr,nullptr,cp.data());
                        if(selected==previous)continue;
                        auto size=checkpoint();
                        if(size<accepted){accepted=size;improved=true;}else{selected=std::move(previous);thr=std::move(previous_thr);}
                    }
                    if(!improved)break;
                }
                continue;
            }
            for(unsigned sweep=0;sweep<sweeps;sweep++){
                {BankEncodeTimer timer(3);require(ng_bank_symbols(count,pp.data(),sizes.data(),bp.data(),selected.data(),src,wp.data(),symbols.data(),activity.data())==0);}
                cl=activity_classify(activity,thr);
                {BankEncodeTimer timer(6);
                if(exhaustive)activity_reselect(symbols,cl,block,cp,selected,1);
                else if(banks==2)bank_shortlist_select<true,true,2>(src,d,3,candidate[12],cl,selected,fixed,&history,true,&symbols,nullptr,nullptr,cp.data());
                else bank_shortlist_select<true,true>(src,d,3,candidate[12],cl,selected,fixed,&history,true,&symbols,nullptr,nullptr,cp.data());
                }
                checkpoint();
            }
        }
        }
        require(best.size()<=cap);memcpy(out,best.data(),best.size());*written=best.size();return 0;
    }catch(...){return -1;}
}
extern "C" int ng_fast_split_encode(const uint8_t*s,unsigned d,unsigned b,unsigned r,unsigned p,uint8_t*out,size_t cap,size_t*n){return ng_fast_split_encode_impl(s,d,b,r,p,false,false,out,cap,n);}
extern "C" int ng_fast_split_hist_encode(const uint8_t*s,unsigned d,unsigned b,unsigned r,unsigned p,uint8_t*out,size_t cap,size_t*n){return ng_fast_split_encode_impl(s,d,b,r,p,true,false,out,cap,n);}
extern "C" int ng_fast_split_full_encode(const uint8_t*s,unsigned d,unsigned b,unsigned r,unsigned p,uint8_t*out,size_t cap,size_t*n){return ng_fast_split_encode_impl(s,d,b,r,p,true,true,out,cap,n);}
// Policies6/7 match4/5 with one globally fitted family instead of two regional
// families. The initial NG fit remains necessary for the unchanged screening.
extern "C" int ng_fast_split_seeded_encode(const uint8_t*s,unsigned d,unsigned policy,unsigned sweeps,uint8_t*out,size_t cap,size_t*n){if(policy<1||policy>7)return -1;return ng_fast_split_encode_impl(s,d,policy>=6?1:2,1,sweeps,false,false,out,cap,n,policy);}
extern "C" void ng_fast_split_search_trace(double*out){if(out)std::copy_n(fast_split::search_trace,96,out);}
// Screen a genuinely small menu: blank + one/two fixed predictors, each either
// bare or assisted by its own globally fitted NG input model. No regional fits.
static int ng_screen_pair_encode_impl(const uint8_t*src,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*written,const int*config=nullptr,unsigned entropy_K=16,unsigned nested=0,bool separate=false,const std::vector<unsigned>*diagnostic_menu=nullptr,unsigned evidence_flags=0){
    struct TapPhaseScope {int old=ng_tap_fit_phase;TapPhaseScope(){ng_tap_fit_phase=1;}~TapPhaseScope(){ng_tap_fit_phase=old;}} tap_phase;
    const auto*stream_menu=fast_split::stream_menu;
    const int forced=fast_split::forced_uniform;
    const int extra=fast_split::extra_uniform;
    const int second=fast_split::extra_uniform_second;
    try{
        require(forced<0||(forced<24&&mode==12&&!nested&&!separate));
        require(extra<0||(extra<24&&forced<0&&mode==12&&!nested&&!separate));
        require(second<0||(second<24&&forced<0&&mode==12&&!nested&&!separate));
        require(src&&out&&written&&mode<=16&&nested<=3&&(!nested||mode==12)&&ng_geometry_valid(d,d,d));GeometryScope geometry(d,d,d);
        if(nested)fast_split::nested_frames.clear();
        zero_blocks::Scan zeros(src,d,choice_frame::enabled);
        if(zeros.all()){auto f=choice_frame::empty_frame(d);require(f.size()<=cap);memcpy(out,f.data(),f.size());*written=f.size();if(nested){fast_split::nested_frames.assign(nested,f);std::fill_n(fast_split::search_trace,96,0.);for(unsigned k=0;k<nested;k++)fast_split::search_trace[80+k]=fast_split::search_trace[83+k]=f.size();}return 0;}
        const unsigned threshold_method=mode>=11?mode-10:0;
        const bool fast_uniform=mode>=10;
        const bool uniform_chunk=mode>=9;
        const bool sampled=mode>=6;if(sampled)mode=uniform_chunk?4:mode-3;
        const unsigned full_d=d;const size_t full_n=N;const uint8_t*full_src=src;
        int cfg[6]={3,3,1,16,40,128};if(config)std::copy_n(config,6,cfg);
        require(entropy_K>=2&&entropy_K<=32&&(!config||fast_uniform));
        const unsigned nb=d/choice_grid::edge();const size_t blocks=size_t(nb)*nb*nb;
        std::fill(fast_split::search_trace,fast_split::search_trace+96,0.);
        Bytes proxy;std::unique_ptr<GeometryScope> screen_geometry;
        if((sampled&&d>32)||forced>=0){
            // Eight distributed 16^3 interiors packed into one native 32^3
            // screening volume. Trial prediction histories are local proxies,
            // not claimed to equal the full-volume causal history.
            BankEncodeTimer timer(7);
            if(forced<0){proxy=fast_split::screen_proxy(src,d,&zeros);src=proxy.data();}
            // A known choice needs no sample data. The common full-chunk
            // expansion below still performs exactly Fast's fit/residual path.
            d=32;screen_geometry=std::make_unique<GeometryScope>(d,d,d);
        }
        fast_split::search_trace[70]=N;fast_split::search_trace[71]=screen_geometry?0:2;
        zero_blocks::Scan screen_zeros(src,d,forced<0&&zeros.any());
        int trial_cfg[6];std::copy(cfg,cfg+6,trial_cfg);
        // The 128-wide stride is for full chunks. On 32^3 it undersamples
        // the normal equations and triggers NG's insufficient-data fallback.
        if(screen_geometry)trial_cfg[5]=4;
        std::vector<Bytes> candidate(nested?156:36);
        if(forced<0){for(unsigned m=0;m<12;m++)candidate[m].resize(N);
            BankEncodeTimer timer(0);bank_fixed_candidates(src,candidate,d);}
        std::vector<unsigned> top;
        std::vector<fast_split::Evidence> cached;
        if(forced>=0)top={unsigned(forced)%12};
        else {BankEncodeTimer timer(7);top=fast_split::screen(candidate,d,12,&cached,entropy_K,screen_zeros.fit_mask(true));}
        // Keep Fast's usual two candidates. Fit an extra NG model only when
        // the stream proposes a learned predictor absent from that shortlist.
        for(int advice:{extra,second})if(advice>=12&&std::find(top.begin(),top.end(),unsigned(advice)%12)==top.end())top.push_back(unsigned(advice)%12);
        if(stream_menu){
            require(nested&&separate&&diagnostic_menu==stream_menu);
            top.clear();for(auto id:*stream_menu)if(id>=12&&id<24)top.push_back(id-12);
            // An empty carrier is used by the existing bare-only frame helper.
            if(top.empty())top.push_back(0);
        }else if(nested)for(unsigned m=0;m<12;m++)if(std::find(top.begin(),top.end(),m)==top.end())top.push_back(m);
        if(nested&&separate&&(!stream_menu||std::find(stream_menu->begin(),stream_menu->end(),24)!=stream_menu->end()))top.push_back(0);
        unsigned banks=top.size(),P=12+12*banks;
        candidate.resize(P);
        std::vector<unsigned>plain(banks);if(nested&&separate&&(!stream_menu||std::find(stream_menu->begin(),stream_menu->end(),24)!=stream_menu->end()))plain.back()=1;
        std::vector<small_bank::Fit> fit;fit.reserve(banks);Bytes feature(N);
        std::unique_ptr<void,decltype(&ng_fit_session_destroy)> fit_cache(fast_uniform&&forced<0?ng_fit_session_create(1):nullptr,ng_fit_session_destroy);
        require(!fast_uniform||forced>=0||fit_cache!=nullptr);
        for(unsigned k=0;k<banks;k++){
            if(forced>=0){fit.emplace_back();fit.back().base.resize(12);continue;}
            {BankEncodeTimer timer(1);
            for(size_t i=0;i<feature.size();i++)feature[i]=uint8_t(src[i]-unpyramid(candidate[top[k]][i]));
            fit.push_back(plain[k]?small_bank::plain_fit(src,trial_cfg,screen_zeros.fit_mask(true)):small_bank::fit(src,d,trial_cfg,feature,screen_zeros.fit_mask(true),candidate,1u<<top[k],true,nullptr,nullptr,nullptr,screen_zeros.any()?nullptr:fit_cache.get()));}
            candidate[12+12*k+top[k]]=std::move(fit.back().initial_symbols);
        }
        fit_cache.reset();
        std::vector<const uint8_t*> pp(12*banks);std::vector<size_t> pn(12*banks);std::vector<const uint16_t*> bp(12*banks);
        std::vector<uint8_t*> wp(P);
        for(unsigned k=0;k<banks;k++)for(unsigned m=0;m<12;m++){
            pp[12*k+m]=fit[k].params.data();pn[12*k+m]=fit[k].params.size();bp[12*k+m]=fit[k].base[top[k]].data();
        }
        for(unsigned m=0;m<P;m++)wp[m]=candidate[m].empty()?nullptr:candidate[m].data();
        Bytes symbols(N);std::vector<uint16_t>activity(N);Bytes selected(blocks);
        std::vector<unsigned> options{0};
        for(unsigned k=0;k<std::min(2u,banks);k++)for(unsigned id:{top[k],12+12*k+top[k]})
            if(std::find(options.begin(),options.end(),id)==options.end())options.push_back(id);
        for(int advice:{extra,second})if(advice>=0){
            unsigned id=unsigned(advice);
            if(id>=12){auto it=std::find(top.begin(),top.end(),id%12);require(it!=top.end());id=12+12*unsigned(it-top.begin())+id%12;}
            if(std::find(options.begin(),options.end(),id)==options.end())options.push_back(id);
        }
        fast_split::Samples sample(d);
        std::vector<Bytes> sy(P),cl(P);std::vector<std::vector<uint16_t>> thresholds(P);
        std::vector<double> uniform(P);
        auto estimate=[&](const Bytes&s,const Bytes&c,const Bytes&choices,unsigned groups=1){
            const unsigned scale=unsigned(full_n/s.size());std::vector<Histogram> hist(entropy_K*groups);Histogram selectors{};
            for(size_t i=0;i<s.size();i++)if(!screen_zeros.any()||screen_zeros.mask[sample.index[i]])hist[c[i]+entropy_K*(groups>1?choices[sample.block[i]]:0)][s[i]]+=scale;
            for(auto v:choices)selectors[v]+=scale;
            double bytes=histogram_cost(selectors.data(),12,true);
            for(auto&h:hist)bytes+=histogram_cost(h.data(),12,true);
            return bytes/8.; // histogram_cost is bits; model storage is bytes.
        };
        auto all_options=options;
        if(forced>=0)all_options.clear();
        if(nested){all_options.clear();for(unsigned m=0;m<12;m++)all_options.push_back(m);for(unsigned k=0;k<banks;k++)all_options.push_back(12+12*k+top[k]);}
        for(auto id:all_options){
            std::array<uint32_t,2560> activity_hist;
            if(id>=12){BankEncodeTimer timer(3);
                require(ng_activity_symbol_histogram(candidate[id].data(),activity.data(),activity_hist.data())==0);}
            {BankEncodeTimer timer(7);
            if(id<12){
                thresholds[id]=std::move(cached[id].thresholds);sy[id]=std::move(cached[id].symbols);cl[id]=std::move(cached[id].classes);
            }else{
                if(screen_zeros.any()){activity_hist.fill(0);for(size_t i=0;i<N;i++)if(screen_zeros.mask[i])activity_hist[activity[i]]++;}
                thresholds[id]=activity_quantiles_histogram(activity_hist,entropy_K);sy[id]=sample.take(candidate[id]);
                cl[id]=activity_classify(sample.take(activity),thresholds[id]);
            }
            if(evidence_flags&2){
                require(ng_activity_symbol_values(candidate[id].data(),activity.data())==0);
                auto aa=screen_zeros.any()?screen_zeros.take(activity):activity;
                auto ss=screen_zeros.any()?screen_zeros.take(candidate[id]):candidate[id];
                thresholds[id]=activity_refine(ss,aa,activity_quantiles(aa,entropy_K),{}, {},1,true,64,true);
                cl[id]=activity_classify(sample.take(activity),thresholds[id]);
            }
            uniform[id]=estimate(sy[id],cl[id],Bytes(sample.blocks));}
        }
        std::vector<std::vector<std::array<double,256>>> independent_screen(P);
        if(evidence_flags&1)for(auto id:all_options){
            auto ss=sy[id],cc=cl[id];
            if(screen_zeros.any()){ss.clear();cc.clear();for(size_t i=0;i<sy[id].size();i++)if(screen_zeros.mask[sample.index[i]]){ss.push_back(sy[id][i]);cc.push_back(cl[id][i]);}}
            independent_screen[id]=activity_symbol_costs(ss,cc,{}, {},1);
        }
        for(unsigned j=0;j<options.size();j++){fast_split::search_trace[48+j]=options[j];fast_split::search_trace[56+j]=uniform[options[j]];}
        fast_split::search_trace[46]=options.size();
        Bytes empty_params(65536);size_t empty_size=0;
        require(ng_bank_empty_model(empty_params.data(),empty_params.size(),&empty_size)==0);empty_params.resize(empty_size);
        auto metadata_cost=[&](const std::vector<unsigned>&menu){
            std::vector<unsigned>used(banks);for(auto id:menu)if(id>=12)used[(id-12)/12]=true;
            Bytes meta;unsigned first=0;while(first<banks&&!used[first])first++;
            for(unsigned k=0;k<banks;k++)if(used[k]){packed_bank::append(meta,fit[k].params);for(auto w:fit[k].weights)put32(meta,uint32_t(w));}
            // NFS1 requires one model descriptor even for a fixed-only menu;
            // it carries the entropy geometry/history, not an enabled NG choice.
            if(first==banks)packed_bank::append(meta,empty_params);
            packed_bank::append(meta,first<banks?fit[first].params:empty_params); // entropy header parameters
            return double(packed_bank::compress(meta).size()+12);
        };
        struct Screen {double cost=std::numeric_limits<double>::infinity();std::vector<unsigned> menu;Bytes choices,symbols,classes;std::vector<uint16_t>thr;unsigned seed=0;};
        fast_split::menu_scores.clear();
        auto record_menu=[&](const std::vector<unsigned>&menu,double cost){
            if(!nested||!separate)return;
            fast_split::menu_scores.push_back(menu.size());fast_split::menu_scores.push_back(cost);
            for(unsigned i=0;i<3;i++){
                unsigned id=i<menu.size()?menu[i]:255;
                if(id!=255&&id>=12){unsigned k=(id-12)/12;id=plain[k]?24:12+top[k];}
                fast_split::menu_scores.push_back(id);
            }
        };
        Screen best_single,best_pair;unsigned trial=0;
        auto score_menu=[&](std::vector<unsigned> menu){
            std::vector<unsigned> unique;for(auto id:menu)if(std::find(unique.begin(),unique.end(),id)==unique.end())unique.push_back(id);menu=std::move(unique);
            unsigned seed=0;for(unsigned i=1;i<menu.size();i++)if(uniform[menu[i]]<uniform[menu[seed]])seed=i;
            Bytes choices(sample.blocks,seed);std::vector<const uint8_t*>cp;for(auto id:menu)cp.push_back(sy[id].data());
            const auto&classes=cl[menu[seed]];
            std::vector<Bytes>own_classes;
            std::vector<std::array<double,256>>independent_cost;
            if(evidence_flags&1)for(auto id:menu){own_classes.push_back(cl[id]);const auto&c=independent_screen[id];independent_cost.insert(independent_cost.end(),c.begin(),c.end());}
            if(separate){
                std::vector<std::array<double,256>>cost;
                for(auto id:menu){auto c=activity_symbol_costs(sy[id],classes,{}, {},1);cost.insert(cost.end(),c.begin(),c.end());}
                activity_select_costs(classes,sample.block,cp,choices,menu.size(),evidence_flags&1?independent_cost:cost,evidence_flags&1?&own_classes:nullptr);
            }else activity_reselect(sy[menu[seed]],classes,sample.block,cp,choices,1);
            Bytes mixed(sample.index.size());for(size_t i=0;i<mixed.size();i++)mixed[i]=sy[menu[choices[sample.block[i]]]][i];
            if(separate){
                if(evidence_flags&1)activity_select_costs(classes,sample.block,cp,choices,menu.size(),independent_cost,&own_classes);
                else activity_reselect(mixed,classes,sample.block,cp,choices,menu.size());
                for(size_t i=0;i<mixed.size();i++)mixed[i]=sy[menu[choices[sample.block[i]]]][i];}
            double cost=estimate(mixed,classes,choices,separate?menu.size():1)+metadata_cost(menu);
            record_menu(menu,cost);
            if(trial<16)fast_split::search_trace[24+trial]=cost;trial++;
            return Screen{cost,menu,choices,mixed,classes,thresholds[menu[seed]],menu[seed]};
        };
        auto consider=[&](std::vector<unsigned> menu,bool pair){
            auto proposal=score_menu(std::move(menu));auto&best=pair?best_pair:best_single;
            if(proposal.cost<best.cost)best=std::move(proposal);
        };
        {BankEncodeTimer timer(7);
        if(fast_uniform){
            // Uniform coding never uses block-menu trials. Keep the same
            // uniform evidence and choose below, without dead reselection.
        }else if(mode>=3){
            // Requested sequence: evaluate each predictor individually, keep
            // its better bare/NG form, then trial only best-single vs the pair.
            unsigned kept[2];double score[2];
            for(unsigned k=0;k<2;k++){
                unsigned bare=top[k],ng=12+12*k+bare;
                double a=uniform[bare]+metadata_cost({0,bare});
                double b=uniform[ng]+metadata_cost({0,ng});
                kept[k]=b<a?ng:bare;score[k]=std::min(a,b);
                fast_split::search_trace[62+k]=kept[k];
                fast_split::search_trace[64+2*k]=a;fast_split::search_trace[65+2*k]=b;
            }
            consider({0,kept[score[1]<score[0]?1:0]},false);
            consider({0,kept[0],kept[1]},true);
        }else{
        consider({0},false);
        for(unsigned k=0;k<2;k++)for(auto id:{top[k],12+12*k+top[k]})consider({0,id},false);
        for(unsigned a:{top[0],12+top[0]})for(unsigned b:{top[1],24+top[1]})consider({0,a,b},true);
        }
        }
        auto&best=(mode==0||mode==4)?best_single:(mode==2||mode==5)?best_pair:best_pair.cost<best_single.cost?best_pair:best_single;
        if(uniform_chunk&&forced<0){
            // Ablation: choose the best uniform predictor on the sample, not
            // a predictor selected by a full-chunk oracle or block search.
            unsigned chosen=options[0];double score=uniform[chosen]+metadata_cost({chosen});
            for(auto id:options){double cost=uniform[id]+metadata_cost({id});
                if(cost<score){chosen=id;score=cost;}}
            best={score,{chosen},Bytes(sample.blocks),sy[chosen],cl[chosen],thresholds[chosen],chosen};
        }
        if(forced>=0)best={0,{unsigned(forced)},Bytes(sample.blocks),{},{},{},unsigned(forced)};
        if(uniform_chunk&&!nested)fast_split::last_uniform=best.seed<12?best.seed:12+best.seed%12;
        std::vector<Screen> plans{best};
        if(nested){
          if(diagnostic_menu){
            // Explicit recipe: do not run the generic beam just to override it.
          }else if(separate){
            // A small beam over arbitrary menus: all singles, then expand
            // the best8 prefixes. No compulsory direct/NG/Fast slots.
            std::vector<Screen>beam;
            for(auto id:all_options)beam.push_back(Screen{uniform[id]+metadata_cost({id}),{id},Bytes(sample.blocks),sy[id],cl[id],thresholds[id],id});
            for(const auto&p:beam)record_menu(p.menu,p.cost);
            auto rank=[](const Screen&a,const Screen&b){return a.cost!=b.cost?a.cost<b.cost:a.menu<b.menu;};
            std::sort(beam.begin(),beam.end(),rank);plans={beam.front()};beam.resize(std::min(size_t(8),beam.size()));
            for(unsigned depth=2;depth<=nested;depth++){
                std::vector<Screen>next;std::vector<std::vector<unsigned>>seen;
                for(const auto&prefix:beam)for(auto id:all_options)if(std::find(prefix.menu.begin(),prefix.menu.end(),id)==prefix.menu.end()){
                    auto menu=prefix.menu;menu.push_back(id);std::sort(menu.begin(),menu.end());
                    if(std::find(seen.begin(),seen.end(),menu)!=seen.end())continue;
                    seen.push_back(menu);next.push_back(score_menu(std::move(menu)));
                }
                require(!next.empty());std::sort(next.begin(),next.end(),rank);plans.push_back(next.front());
                next.resize(std::min(size_t(8),next.size()));beam=std::move(next);
            }
          }else{
            // Keep Fast's exact winner, then score every extra bare/learned
            // predictor by its improvement to the CURRENT menu on the proxy.
            while(plans.back().menu.size()<nested){
                Screen next;
                for(auto id:all_options)if(std::find(plans.back().menu.begin(),plans.back().menu.end(),id)==plans.back().menu.end()){
                    auto menu=plans.back().menu;menu.push_back(id);auto p=score_menu(std::move(menu));
                    if(p.cost<next.cost)next=std::move(p);
                }
                require(!next.menu.empty());plans.push_back(std::move(next));
            }
          }
            if(diagnostic_menu){
                std::vector<unsigned>menu;
                for(auto id:*diagnostic_menu){
                    if(id<12)menu.push_back(id);
                    else {unsigned k=0;for(;k<banks;k++)if(id==24?plain[k]:!plain[k]&&top[k]==id-12)break;require(k<banks);menu.push_back(12+12*k+top[k]);}
                }
                plans={score_menu(std::move(menu))};
            }
            // Only finalist model fits are expanded to the full volume.
            std::vector<unsigned>old_banks;
            for(const auto&p:plans)for(auto id:p.menu)if(id>=12){unsigned k=(id-12)/12;if(std::find(old_banks.begin(),old_banks.end(),k)==old_banks.end())old_banks.push_back(k);}
            std::vector<small_bank::Fit>compact;std::vector<unsigned>newtop,newplain;
            for(auto k:old_banks){compact.push_back(std::move(fit[k]));newtop.push_back(top[k]);newplain.push_back(plain[k]);}
            // Fixed-only paths still use one empty carrier for the old helper.
            if(compact.empty()){compact.push_back(std::move(fit[0]));newtop.push_back(top[0]);newplain.push_back(plain[0]);}
            auto remap=[&](unsigned id){if(id<12)return id;auto at=std::find(old_banks.begin(),old_banks.end(),(id-12)/12);require(at!=old_banks.end());return unsigned(12+12*(at-old_banks.begin())+(id-12)%12);};
            std::vector<Bytes>newcandidate(12+12*compact.size());
            for(unsigned m=0;m<12;m++)newcandidate[m]=std::move(candidate[m]);
            for(auto k:old_banks){unsigned id=12+12*k+top[k];newcandidate[remap(id)]=std::move(candidate[id]);}
            for(auto&p:plans){for(auto&id:p.menu)id=remap(id);p.seed=remap(p.seed);}
            fit=std::move(compact);top=std::move(newtop);plain=std::move(newplain);candidate=std::move(newcandidate);
            banks=fit.size();P=12+12*banks;pp.resize(12*banks);pn.resize(12*banks);bp.resize(12*banks);wp.resize(P);
            best=plans.back();
        }
        fast_split::search_trace[16]=best_single.cost;fast_split::search_trace[17]=best_pair.cost;
        fast_split::search_trace[18]=best.cost;fast_split::search_trace[19]=best.menu.size();
        for(unsigned i=0;i<best.menu.size();i++)fast_split::search_trace[40+i]=best.menu[i];
        fast_split::search_trace[43]=&best==&best_pair;fast_split::search_trace[44]=trial;
        std::array<uint32_t,2560> final_hist{};bool activity_ready=false;
        if(screen_geometry){
            screen_geometry.reset();d=full_d;src=full_src;ng_tap_fit_phase=2;
            unsigned fixed_mask=zeros.any()?1u:0u;std::vector<unsigned>refit(banks);
            std::vector<unsigned>full_menu=best.menu;
            if(nested){full_menu.clear();for(const auto&p:plans)for(auto id:p.menu)if(std::find(full_menu.begin(),full_menu.end(),id)==full_menu.end())full_menu.push_back(id);}
            for(auto id:full_menu){
                if(id<12)fixed_mask|=1u<<id;
                else{unsigned k=(id-12)/12;refit[k]=true;fixed_mask|=1u<<top[k];}
            }
            for(auto&v:candidate)v.clear();
            for(unsigned m=0;m<12;m++)if(fixed_mask&(1u<<m))candidate[m].resize(N);
            {BankEncodeTimer timer(0);bank_fixed_candidates(src,candidate,d,fixed_mask);}
            for(unsigned k=0;k<banks;k++)if(refit[k]){
                BankEncodeTimer timer(1);feature.resize(N);
                for(size_t i=0;i<full_n;i++)feature[i]=uint8_t(src[i]-unpyramid(candidate[top[k]][i]));
                if(fast_uniform)activity.resize(full_n);
                bool fused=fast_uniform&&!zeros.any()&&!nested&&!shared_grid::enabled&&!subset_targets::fit_sample&&!small_bank::memo::enabled&&!(ng_tap_swap_mode&8);
                if(ng_tap_swap_mode&8)
                    fit[k]=small_bank::refine(src,d,cfg,plain[k]?nullptr:&feature,zeros.fit_mask(),nullptr,candidate,top[k],fit[k]);
                else fit[k]=plain[k]?small_bank::plain_fit(src,cfg,zeros.fit_mask()):small_bank::fit(src,d,cfg,feature,zeros.fit_mask(),candidate,fused?0:1u<<top[k],true,fused?&candidate[top[k]]:nullptr,
                    fused?activity.data():nullptr,fused?final_hist.data():nullptr);
                activity_ready=fused;
                candidate[12+12*k+top[k]]=std::move(fit[k].initial_symbols);
                fast_split::search_trace[71]++;
            }
            for(unsigned k=0;k<banks;k++)for(unsigned m=0;m<12;m++){
                pp[12*k+m]=fit[k].params.data();pn[12*k+m]=fit[k].params.size();bp[12*k+m]=fit[k].base[top[k]].data();
            }
            for(unsigned m=0;m<P;m++)wp[m]=candidate[m].empty()?nullptr:candidate[m].data();
            // Multi-model finalists may evaluate directly into symbols before
            // a seed array is copied; expand the proxy-sized scratch first.
            if(!fast_uniform||nested)symbols.resize(N);activity.resize(N);
            empty_params.resize(65536);
            require(ng_bank_empty_model(empty_params.data(),empty_params.size(),&empty_size)==0);empty_params.resize(empty_size);
        }
        if(nested){
            for(unsigned k=0;k<banks;k++)for(unsigned m=0;m<12;m++){
                pp[12*k+m]=fit[k].params.data();pn[12*k+m]=fit[k].params.size();bp[12*k+m]=fit[k].base[top[k]].data();
            }
        }
        if(shared_grid::enabled){
            require(!nested&&best.menu.size()==1);unsigned id=best.menu[0];const small_bank::Fit*f=id>=12?&fit[(id-12)/12]:nullptr;
            std::vector<shared_grid::Candidate> menu={{id%12,id,f,f?f->base[top[(id-12)/12]].data():nullptr}};
            auto frame=shared_grid::run(src,d,menu,candidate,zeros,true);
            require(frame.size()<=cap);memcpy(out,frame.data(),frame.size());*written=frame.size();return 0;
        }
        const auto uniform_candidates=nested?candidate:std::vector<Bytes>{};
        Bytes retained;unsigned stage=0;
        for(auto&plan:plans){
        best=plan;
        if(nested){candidate=uniform_candidates;for(unsigned m=0;m<P;m++)wp[m]=candidate[m].empty()?nullptr:candidate[m].data();activity_ready=false;}
        // Generate only the shortlisted learned candidates using the seed's
        // causal history, then one full selection pass using screened costs.
        std::fill(selected.begin(),selected.end(),best.seed);
        std::vector<uint8_t*> shortlist_wp=wp;
        for(unsigned id=12;id<P;id++)if(std::find(best.menu.begin(),best.menu.end(),id)==best.menu.end())shortlist_wp[id]=nullptr;
        auto evaluate_candidates=[&](bool selected_only){
            // The union of finalists may have >4 models; each individual
            // menu has <=3. Remap references, not voxel arrays, for the shared
            // native evaluator. Full fits remain cached across menus.
            std::vector<unsigned>used,remap(banks);
            for(auto id:best.menu)if(id>=12){unsigned k=(id-12)/12;if(std::find(used.begin(),used.end(),k)==used.end()){remap[k]=used.size();used.push_back(k);}}
            require(!used.empty()&&used.size()<=4);
            std::vector<const uint8_t*>par;std::vector<size_t>sz;std::vector<const uint16_t*>base;
            std::vector<uint8_t*>outputs(shortlist_wp.begin(),shortlist_wp.begin()+12);
            for(auto k:used)for(unsigned m=0;m<12;m++){par.push_back(pp[12*k+m]);sz.push_back(pn[12*k+m]);base.push_back(bp[12*k+m]);outputs.push_back(shortlist_wp[12+12*k+m]);}
            Bytes mapped=selected;for(auto&id:mapped)if(id>=12)id=12+12*remap[(id-12)/12]+(id-12)%12;
            return selected_only?ng_bank_selected_symbols(par.size(),par.data(),sz.data(),base.data(),mapped.data(),src,outputs.data(),symbols.data(),activity.data())
                :ng_bank_symbols(par.size(),par.data(),sz.data(),base.data(),mapped.data(),src,outputs.data(),symbols.data(),activity.data());
        };
        unsigned learned=0;for(auto id:best.menu)learned+=id>=12;
        {BankEncodeTimer timer(3);
        if(!learned||(learned==1&&best.seed>=12)){
            if(fast_uniform&&!zeros.any()&&!nested)symbols=std::move(candidate[best.seed]);else symbols=candidate[best.seed];
            if(!activity_ready)require((fast_uniform?ng_activity_symbol_histogram(symbols.data(),activity.data(),final_hist.data())
                :ng_activity_symbol_values(symbols.data(),activity.data()))==0);
        }else require(evaluate_candidates(false)==0);}
        std::vector<uint32_t>block(best.menu.size()>1?N:0);
        if(!block.empty())for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++)block[(size_t(z)*d+y)*d+x]=((z/choice_grid::edge())*nb+y/choice_grid::edge())*nb+x/choice_grid::edge();
        std::vector<const uint8_t*>cp;for(auto id:best.menu)cp.push_back(candidate[id].data());
        Bytes local(blocks);for(size_t b=0;b<blocks;b++)local[b]=best.choices[b%best.choices.size()];
        auto select_independent=[&](){
            std::vector<Bytes>classes;
            std::vector<std::array<double,256>>costs;
            auto aa=zeros.filtered()?zeros.take(activity):activity;
            for(auto id:best.menu){
                auto ss=zeros.filtered()?zeros.take(candidate[id]):candidate[id];
                auto t=activity_quantiles(aa,entropy_K);
                if(evidence_flags&2)t=activity_refine(ss,aa,t,{}, {},1,true,64,true);
                auto c=activity_classify(activity,t);auto ac=zeros.filtered()?zeros.take(c):c;
                auto cost=activity_symbol_costs(ss,ac,{}, {},1);
                costs.insert(costs.end(),cost.begin(),cost.end());classes.push_back(std::move(c));
            }
            activity_select_costs(classes.front(),block,cp,local,best.menu.size(),costs,&classes);
        };
        {BankEncodeTimer timer(6);
        if(best.menu.size()>1){
        if(evidence_flags&1){if(learned)require(evaluate_candidates(false)==0);select_independent();}
        else{
        unsigned groups=separate?best.menu.size():1;
        auto cost=activity_symbol_costs(best.symbols,best.classes,sample.block,best.choices,groups);
        activity_select_costs(activity_classify(activity,best.thr),block,cp,local,groups,cost);}}}
        for(size_t b=0;b<blocks;b++)selected[b]=best.menu[local[b]];
        if(zeros.any())for(size_t b=0;b<blocks;b++)if(zeros.empty[b])selected[b]=0;
        if(!std::all_of(selected.begin(),selected.end(),[&](auto id){return id==best.seed;})){
            BankEncodeTimer timer(3);
            if(!learned){
                for(size_t i=0;i<N;i++){size_t b=block.empty()?(((i/(d*d)/choice_grid::edge())*nb+(i/d%d)/choice_grid::edge())*nb+(i%d)/choice_grid::edge()):block[i];symbols[i]=candidate[selected[b]][i];}
                require(ng_activity_symbol_values(symbols.data(),activity.data())==0);
            }else require(evaluate_candidates(true)==0);
        }
        if(separate&&best.menu.size()>1&&!(stream_menu&&fast_split::stream_onepass)){
            // Match Slow's observed-distribution reselection, without refits.
            if(learned)require(evaluate_candidates(false)==0);
            if(evidence_flags&1)select_independent();
            else{
            auto active=zeros.filtered()?zeros.take(activity):activity;auto t=activity_quantiles(active,entropy_K);
            if(evidence_flags&2){auto ss=zeros.filtered()?zeros.take(symbols):symbols;auto bl=zeros.filtered()?zeros.take(block):block;t=activity_refine(ss,active,t,bl,local,best.menu.size(),true,64,true);}
            auto classes=activity_classify(activity,t);
            auto observed=zeros.filtered()?zeros.take(symbols):symbols;
            auto ac=zeros.filtered()?zeros.take(classes):classes;
            auto bl=zeros.filtered()?zeros.take(block):block;
            auto cost=activity_symbol_costs(observed,ac,bl,local,best.menu.size());
            activity_select_costs(classes,block,cp,local,best.menu.size(),cost);
            }
            for(size_t b=0;b<blocks;b++)selected[b]=zeros.any()&&zeros.empty[b]?0:best.menu[local[b]];
            if(learned)require(evaluate_candidates(true)==0);
            else {for(size_t i=0;i<N;i++)symbols[i]=candidate[selected[block[i]]][i];require(ng_activity_symbol_values(symbols.data(),activity.data())==0);}
        }
        // Freeze choices. Final observed statistics/thresholds drive the sole
        // rANS payload encode; there is no trial payload or subsequent search.
        auto thr=best.thr;std::vector<std::vector<uint16_t>> threshold_finalists;
        auto active_symbols=zeros.filtered()?zeros.take(symbols):Bytes{};
        auto active_activity=zeros.filtered()?zeros.take(activity):std::vector<uint16_t>{};
        const auto&threshold_symbols=zeros.filtered()?active_symbols:symbols;
        const auto&threshold_activity=zeros.filtered()?active_activity:activity;
        if(zeros.filtered()||nested){final_hist.fill(0);for(auto a:threshold_activity)final_hist[a]++;}
        {BankEncodeTimer timer(5);
        if(fast_uniform&&fast_split::stream_thresholds){
            thr=*fast_split::stream_thresholds;
            if(fast_split::stream_threshold_mode==2){
                auto active_block=separate&&best.menu.size()>1?(zeros.filtered()?zeros.take(block):block):std::vector<uint32_t>{};
                thr=activity_refine(threshold_symbols,threshold_activity,thr,active_block,
                    separate&&best.menu.size()>1?local:Bytes{},separate?best.menu.size():1,true,0,false);
            }
        }else if(fast_uniform){
            // Default: one observed quantile estimate. The experimental
            // refinement touches only the saved residual/activity histogram.
            thr=activity_quantiles_histogram(final_hist,entropy_K);
            if(threshold_method==1)
                thr=activity_refine(threshold_symbols,threshold_activity,thr,{}, {},1,true);
            if(threshold_method>=2){
                auto initial=thr;
                unsigned grid=threshold_method==2?64:threshold_method==4?256:128;
                auto active_block=separate&&best.menu.size()>1?(zeros.filtered()?zeros.take(block):block):std::vector<uint32_t>{};
                thr=activity_refine(threshold_symbols,threshold_activity,thr,separate&&best.menu.size()>1?active_block:std::vector<uint32_t>{},separate&&best.menu.size()>1?local:Bytes{},separate?best.menu.size():1,true,grid,true);
                if(threshold_method>=5)thr=activity_refine(threshold_symbols,threshold_activity,thr,{}, {},1,true,0);
                if(threshold_method==6){
                    threshold_finalists.push_back(initial);
                    threshold_finalists.push_back(activity_refine(threshold_symbols,threshold_activity,initial,{}, {},1,true));
                }
            }
        }else{
        ActivityHistogram hist(symbols,activity,true,true);auto quant=activity_quantiles(activity,16);
        if(hist.estimate(quant)<hist.estimate(thr))thr=quant;
        auto tuned=activity_refine(symbols,activity,thr,block,selected,1,true);
        if(hist.estimate(tuned)<hist.estimate(thr))thr=tuned;}}
        std::vector<unsigned>used(banks),mapping(banks);for(auto id:selected)if(id>=12)used[(id-12)/12]=true;
        std::vector<Bytes> params;std::vector<small_bank::Weight> weights;
        for(unsigned k=0;k<banks;k++)if(used[k]){mapping[k]=params.size();params.push_back(fit[k].params);weights.push_back(fit[k].weights);}
        fast_split::search_trace[45]=params.size();
        if(params.empty()){params.push_back(empty_params);weights.push_back({0,0});}
        for(auto&id:selected)if(id>=12)id=12+12*mapping[(id-12)/12]+(id-12)%12;
        Bytes classes;
        if(!fast_uniform){BankEncodeTimer timer(2);classes=activity_classify(activity,thr);}
        Bytes frame;{BankEncodeTimer timer(4);frame=fast_split::frame(d,params,weights,selected,symbols,thr,classes,fast_uniform?activity.data():nullptr,&zeros,separate&&best.menu.size()>1);}
        for(const auto&t:threshold_finalists)if(t!=thr){
            BankEncodeTimer timer(4);auto alternative=fast_split::frame(d,params,weights,selected,symbols,t,classes,activity.data(),&zeros);
            if(alternative.size()<frame.size()){frame=std::move(alternative);thr=t;}
        }
        if(nested){fast_split::search_trace[80+stage]=frame.size();fast_split::search_trace[86+stage]=plan.cost;fast_split::nested_frames.push_back(frame);}
        if(retained.empty()||frame.size()<retained.size())retained=std::move(frame);
        if(nested)fast_split::search_trace[83+stage]=retained.size();stage++;
        }
        require(retained.size()<=cap);memcpy(out,retained.data(),retained.size());*written=retained.size();
        fast_split::search_trace[15]=1;fast_split::search_trace[90]=1;fast_split::search_trace[91]=1;fast_split::search_trace[92]=1;
        return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"screen pair: %s\n",e.what());return -1;}
}
extern "C" int ng_screen_pair_encode(const uint8_t*s,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*n){return ng_screen_pair_encode_impl(s,d,mode,out,cap,n);}
extern "C" int ng_codec_encode_nested(const uint8_t*s,unsigned d,unsigned depth,uint8_t*out,size_t cap,size_t*n){
    if(depth<1||depth>3)return -1;
    const int cfg[6]={3,3,1,16,40,1};bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=ng_screen_pair_encode_impl(s,d,12,out,cap,n,cfg,16,depth);
    choice_frame::enabled=old;return rc;
}
extern "C" int ng_nested_frame(unsigned stage,uint8_t*out,size_t cap,size_t*n){
    if(stage>=fast_split::nested_frames.size()||!out||!n)return -1;
    const auto&f=fast_split::nested_frames[stage];if(cap<f.size())return -1;
    memcpy(out,f.data(),f.size());*n=f.size();return 0;
}
extern "C" int ng_codec_encode_fast_config(const uint8_t*s,unsigned d,const int*cfg,unsigned K,uint8_t*out,size_t cap,size_t*n){
    if(!cfg||cfg[0]<1||cfg[0]>3||cfg[1]<-1||cfg[1]>3||cfg[2]<-1||cfg[2]>3||cfg[3]<2||cfg[3]>32||cfg[4]<2||cfg[4]>160||cfg[5]<1||cfg[5]>128||K<2||K>32)return -1;
    bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=ng_screen_pair_encode_impl(s,d,12,out,cap,n,cfg,K);
    choice_frame::enabled=old;return rc;
}
extern "C" int ng_fast_split_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    try{
        require(out);auto f=fast_split::read(src,n);require(cap>=size_t(f.d)*f.d*f.d);GeometryScope scope(f.d,f.d,f.d);
        using State=std::unique_ptr<void,decltype(&ng_causal_destroy)>;std::vector<State> states;void* models[24]={};int32_t weights[48]={};
        for(unsigned k=0;k<f.banks;k++){
            states.emplace_back(k?ng_causal_create_model(f.params[k].data(),f.params[k].size()):ng_causal_create(f.params[k].data(),f.params[k].size()),ng_causal_destroy);require(bool(states.back()));
            for(unsigned m=0;m<12;m++){unsigned id=12*k+m;models[id]=states.back().get();for(unsigned b=0;b<2;b++)weights[2*id+b]=m?f.weights[k][b]:0;}
        }
        return ng_bank_wave_segments_many(models,12*f.banks,weights,f.choices.data(),out,f.entropy.data(),f.entropy.size(),f.payload,f.payload_size);
    }catch(...){return -1;}
}
extern "C" int ng_fast_split_inspect(const uint8_t*src,size_t n,uint64_t*out){
    try{require(out);auto f=fast_split::read(src,n);std::fill(out,out+40,0);out[0]=f.banks;for(auto c:f.choices){out[4+c]++;out[c<12?3:1+(c-12)/12]++;}return 0;}catch(...){return -1;}
}
