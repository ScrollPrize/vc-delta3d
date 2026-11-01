/* Three independent candidate entropy models. Experimental raster-stream frame;
 * all fitting, causal residuals, rANS and reconstruction reuse native helpers. */
namespace three_candidate {
thread_local bool screen_singleton_default=true;
// Encoder-only stream configuration; default leaves public Slow unchanged.
thread_local int forced_third=-1;
thread_local std::array<int,2> advice_third{-1,-1};
// Frozen adaptive prototypes require their historical mixed-fit incumbent.
struct LegacySlowScope {
    bool old=screen_singleton_default;
    LegacySlowScope(){screen_singleton_default=false;}
    ~LegacySlowScope(){screen_singleton_default=old;}
};
thread_local bool profile_enabled=false;
thread_local bool span_selection=false;
thread_local double profile_seconds[8]={};
struct Timer {
    unsigned phase;bool active;std::chrono::steady_clock::time_point start;
    explicit Timer(unsigned p):phase(p),active(profile_enabled){if(active)start=std::chrono::steady_clock::now();}
    ~Timer(){if(active)profile_seconds[phase]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
};
// Optional experimental cost provider. The production path remains unchanged.
// Reuses this engine's candidate evaluation and causal-history regeneration.
using Feedback=void(*)(const uint8_t*,unsigned,const std::vector<Bytes>&,
    const unsigned*,unsigned,const std::vector<uint16_t>&,const Bytes&,
    const zero_blocks::Scan&,Bytes&);
thread_local Feedback feedback=nullptr;
thread_local unsigned feedback_passes=0;
struct Stats {
    std::vector<uint16_t> thr;
    std::vector<Histogram> hist;
    std::vector<std::array<double,256>> cost;
    explicit Stats(unsigned K=16):hist(K),cost(K){}
    void finish(){
        for(unsigned c=0;c<hist.size();c++){
            double total=128;for(auto n:hist[c])total+=n;
            for(unsigned s=0;s<256;s++)cost[c][s]=-std::log2((hist[c][s]+.5)/total);
        }
    }
    void normalized_costs(){
        for(unsigned c=0;c<hist.size();c++){
            uint64_t h[256];size_t total=0;for(unsigned s=0;s<256;s++){h[s]=hist[c][s];total+=h[s];}
            uint16_t freq[256];delta_rans_size_lower_bound(h,total,freq);
            // Exact normalized cost for observed symbols; reserve one-frequency
            // cost for unseen symbols so a new block can still enter a stream.
            for(unsigned s=0;s<256;s++)cost[c][s]=12.-std::log2(double(std::max(1u,unsigned(freq[s]))));
        }
    }
};
// Exact per-search reuse of entropy fitting for identical candidate histories.
// Compare actual inputs, not only hashes or source/geometry identity.
namespace distribution_cache {
struct Entry {Bytes symbols;std::vector<uint16_t>activity;bool tune;unsigned K;Stats stats;};
thread_local bool enabled=false;
thread_local std::vector<Entry> entries;
thread_local size_t bytes=0,hits=0,misses=0;
constexpr size_t limit=128u*1024*1024;
}
Stats distribution(const Bytes&sy,const std::vector<uint16_t>&activity,bool tune=false,unsigned K=16,const zero_blocks::Scan*zeros=nullptr){
    if(zeros&&zeros->filtered())return distribution(zeros->take(sy),zeros->take(activity),tune,K);
    const bool reuse=distribution_cache::enabled&&sy.size()>32768;
    if(reuse){
        for(auto&e:distribution_cache::entries)if(e.tune==tune&&e.K==K&&e.symbols==sy&&e.activity==activity){distribution_cache::hits++;return e.stats;}
        distribution_cache::misses++;
    }
    Stats s(K);s.thr=activity_quantiles(activity,K);
    if(tune)s.thr=activity_refine(sy,activity,s.thr,{}, {},1,true);
    auto cl=activity_classify(activity,s.thr);
    for(size_t i=0;i<sy.size();i++)s.hist[cl[i]][sy[i]]++;s.finish();
    if(reuse){
        size_t bytes=sy.size()+activity.size()*2;
        if(bytes<=distribution_cache::limit){
            while(distribution_cache::bytes+bytes>distribution_cache::limit&&!distribution_cache::entries.empty()){
                auto&e=distribution_cache::entries.front();distribution_cache::bytes-=e.symbols.size()+2*e.activity.size();distribution_cache::entries.erase(distribution_cache::entries.begin());
            }
            distribution_cache::entries.push_back({sy,activity,tune,K,s});distribution_cache::bytes+=bytes;
        }
    }
    return s;
}
double score(const Stats&s){double v=0;for(auto&h:s.hist)v+=histogram_cost(h.data(),12,true);return v/8;}
// Screening-only entropy estimates on the same spatial 4³ blocks as selection.
std::vector<double> init_block_costs(const Bytes&sy,unsigned d){
    std::vector<uint16_t>a(sy.size());require(ng_activity_symbol_values(sy.data(),a.data())==0);
    auto s=distribution(sy,a);auto cl=activity_classify(a,s.thr);
    std::vector<double>cost(sy.size()/choice_grid::volume());unsigned nb=d/choice_grid::edge();
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
        size_t i=(size_t(z)*d+y)*d+x,b=(size_t(z/choice_grid::edge())*nb+y/choice_grid::edge())*nb+x/choice_grid::edge();
        cost[b]+=s.cost[cl[i]][sy[i]];
    }return cost;
}
Bytes underserved_mask(const std::vector<double>&cost,unsigned d){
    auto sorted=cost;std::sort(sorted.begin(),sorted.end());double cut=sorted[3*sorted.size()/4];
    Bytes mask(size_t(d)*d*d);unsigned nb=d/choice_grid::edge();
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++)
        mask[(size_t(z)*d+y)*d+x]=cost[(size_t(z/choice_grid::edge())*nb+y/choice_grid::edge())*nb+x/choice_grid::edge()]>=cut;
    return mask;
}
thread_local uint64_t diagnostics[12]={};
thread_local uint64_t guarded_diagnostics[12]={}; // initial/final bytes, attempt/accept pairs: choices,thresholds,refits,regions
thread_local std::vector<uint64_t> sweep_bytes;
thread_local std::vector<double> sweep_seconds;
// Common independent-stream framing for fixed and freely selected three menus.
Bytes finish_frame(Bytes header,const Bytes&symbols,const Bytes&choices,const std::vector<uint32_t>&block,
                   const std::vector<uint16_t>&thr,const Bytes&cl,const char*magic){
    Bytes raw(symbols.size());for(size_t i=0;i<raw.size();i++)raw[i]=unpyramid(symbols[i]);
    auto payload=activity_stream_payload(raw,thr,cl,&block,&choices,3);
    auto sel=pack_stream(choices,true);put32(header,sel.size());packed_bank::append(header,sel);
    for(auto t:thr)packed_bank::put16(header,t);
    Bytes body;packed_bank::Reader streams{payload};streams.skip(30);
    for(unsigned c=0;c<48;c++){
        size_t count=streams.u(4),bytes=streams.u(4);
        small_bank::stream_header(header,body,streams.take(bytes),count);
    }
    streams.end();auto packed=packed_bank::compress(header);Bytes frame(magic,magic+4);
    put32(frame,header.size());put32(frame,packed.size());packed_bank::append(frame,packed);packed_bank::append(frame,body);
    diagnostics[10]=packed.size();diagnostics[11]=48;return frame;
}
Bytes read_residuals(packed_bank::Reader&h,const Bytes&f,size_t p,unsigned d,uint8_t*out){
    auto encoded=h.take(h.u(4));Bytes choices(N/choice_grid::volume());unpack_stream(encoded.data(),encoded.size(),choices,true);for(auto c:choices)require(c<3);
    DirectionStreams streams;streams.choices=&choices;streams.K=16;streams.shift=2;streams.dimension=d;streams.separate=true;
    std::vector<uint16_t>thr(15);for(auto&t:thr)t=h.u(2);require(std::is_sorted(thr.begin(),thr.end()));
    packed_bank::Reader body{f};body.skip(p);size_t total=0;streams.pos.resize(48);
    for(unsigned c=0;c<48;c++){streams.symbols.push_back(small_bank::read_stream(h,body,N-total));total+=streams.symbols.back().size();}
    h.end();body.end();require(total==N);
    require(ng_activity_decode(thr.data(),16,0,DirectionStreams::read,&streams,out)==0);
    for(unsigned c=0;c<48;c++)require(streams.pos[c]==streams.symbols[c].size());return choices;
}
}
extern "C" void ng_three_candidate_stats(uint64_t*out){std::copy_n(three_candidate::diagnostics,12,out);}
extern "C" void ng_three_profile_enable(unsigned on){three_candidate::profile_enabled=on;std::fill_n(three_candidate::profile_seconds,8,0.);}
extern "C" void ng_three_profile_get(double*out){std::copy_n(three_candidate::profile_seconds,8,out);}
extern "C" void ng_three_span_selection(unsigned on){three_candidate::span_selection=on!=0;}
extern "C" void ng_three_distribution_cache(unsigned on){
    using namespace three_candidate::distribution_cache;
    entries.clear();bytes=hits=misses=0;enabled=on!=0;
}
extern "C" void ng_three_distribution_stats(uint64_t*out){
    using namespace three_candidate::distribution_cache;
    out[0]=hits;out[1]=misses;out[2]=bytes;
}
extern "C" void ng_three_candidate_guarded_stats(uint64_t*out){std::copy_n(three_candidate::guarded_diagnostics,12,out);}
extern "C" unsigned ng_three_candidate_sweep_stats(uint64_t*bytes,double*seconds,unsigned cap){
    unsigned n=three_candidate::sweep_bytes.size();
    for(unsigned i=0;i<std::min(n,cap);i++){bytes[i]=three_candidate::sweep_bytes[i];seconds[i]=three_candidate::sweep_seconds[i];}return n;
}
// options: 1 full-chunk initial statistics, 2 entropy-refined thresholds,
// 4 extra selection-only pass. Predictor fits and screened menu stay unchanged.
static int three_candidate_encode_impl(const uint8_t*src,unsigned d,unsigned passes,unsigned options,unsigned extra_selections,unsigned refit_rounds,unsigned search_flags,uint8_t*out,size_t cap,size_t*written,unsigned optimize=0,unsigned sweeps=0,unsigned init_flags=0,unsigned threshold_grid=0,bool threshold_feedback=false,const int*config=nullptr,unsigned entropy_K=16,unsigned singleton_trials=0){
    try{
        require(src&&out&&written&&(passes==1||passes==2)&&options<8&&search_flags<16&&extra_selections<=15&&refit_rounds<=extra_selections&&ng_geometry_valid(d,d,d));GeometryScope scope(d,d,d);
        zero_blocks::Scan zeros(src,d,choice_frame::enabled);
        if(zeros.all()){auto f=choice_frame::empty_frame(d);require(f.size()<=cap);memcpy(out,f.data(),f.size());*written=f.size();return 0;}
        using namespace three_candidate;std::fill_n(diagnostics,12,0);
        std::fill_n(guarded_diagnostics,12,0);
        sweep_bytes.clear();sweep_seconds.clear();auto sweep_start=std::chrono::steady_clock::now();
        const size_t voxels=N;unsigned nb=d/choice_grid::edge();size_t blocks=voxels/choice_grid::volume();
        int cfg[6]={3,3,1,16,40,128};unsigned third=1,singleton_seed=0;bool assisted=false;std::vector<unsigned> screen_order;
        if(config)std::copy_n(config,6,cfg);
        require(entropy_K>=2&&entropy_K<=32&&(!config||(!optimize&&!init_flags&&choice_frame::enabled)));
        std::array<Stats,3> stats;
        auto profile_start=std::chrono::steady_clock::now();
        auto proxy=fast_split::screen_proxy(src,d,&zeros);unsigned sd=std::min(d,32u);
        zero_blocks::Scan screen_zeros(proxy.data(),sd,zeros.any());
        {
            GeometryScope screen_scope(sd,sd,sd);int screen_cfg[6];std::copy_n(cfg,6,screen_cfg);screen_cfg[5]=4;
            std::vector<Bytes> fixed(12,Bytes(N));bank_fixed_candidates(proxy.data(),fixed,sd);
            std::vector<fast_split::Evidence> evidence;auto top=fast_split::screen(fixed,sd,12,&evidence,entropy_K,screen_zeros.fit_mask(true));
            for(unsigned m=0;m<12;m++)screen_order.push_back(m);
            std::sort(screen_order.begin(),screen_order.end(),[](unsigned a,unsigned b){return fast_split::search_trace[a]!=fast_split::search_trace[b]?fast_split::search_trace[a]<fast_split::search_trace[b]:a<b;});
            std::vector<uint16_t> activity(N);
            auto plain=small_bank::plain_fit(proxy.data(),screen_cfg,screen_zeros.fit_mask(true));
            require(ng_activity_symbol_values(plain.initial_symbols.data(),activity.data())==0);
            stats[0]=distribution(plain.initial_symbols,activity,false,entropy_K,&screen_zeros);
            std::array<Bytes,3> screen_symbols;screen_symbols[0]=plain.initial_symbols;screen_symbols[1]=fixed[0];
            require(ng_activity_symbol_values(fixed[0].data(),activity.data())==0);stats[1]=distribution(fixed[0],activity,false,entropy_K,&screen_zeros);
            std::vector<double>base;Bytes screen_mask;
            if(init_flags){
                base=init_block_costs(plain.initial_symbols,sd);auto direct=init_block_costs(fixed[0],sd);
                for(size_t b=0;b<base.size();b++)base[b]=std::min(base[b],direct[b]);
                std::vector<std::pair<double,unsigned>>rank;
                for(unsigned m=1;m<12;m++){auto c=init_block_costs(fixed[m],sd);double gain=0;
                    for(size_t b=0;b<base.size();b++)gain+=std::max(0.,base[b]-c[b]);rank.emplace_back(-gain,m);}
                std::sort(rank.begin(),rank.end());top={rank[0].second,rank[1].second};
                if(init_flags&2)screen_mask=underserved_mask(base,sd);
            }
            auto complement=[&](const Bytes&sy,size_t overhead){
                auto c=init_block_costs(sy,sd);double gain=0;
                for(size_t b=0;b<base.size();b++)gain+=std::max(0.,base[b]-c[b]);
                return -gain/8*voxels/N+overhead;
            };
            double best=std::numeric_limits<double>::infinity();
            std::array<bool,24>tested{};
            auto consider_third=[&](unsigned m,bool use_bare,bool use_learned){
                require(m>0&&m<12);
                if(use_bare&&!tested[m]){
                tested[m]=true;
                require(ng_activity_symbol_values(fixed[m].data(),activity.data())==0);auto bare=distribution(fixed[m],activity,false,entropy_K,&screen_zeros);
                double bare_cost=init_flags?complement(fixed[m],0):score(bare)*voxels/N;
                if(bare_cost<best){best=bare_cost;third=m;assisted=false;stats[2]=bare;screen_symbols[2]=fixed[m];}
                }
                if(!use_learned||tested[12+m])return;
                tested[12+m]=true;
                Bytes feature(N);for(size_t i=0;i<N;i++)feature[i]=uint8_t(proxy[i]-unpyramid(fixed[m][i]));
                auto fit=small_bank::fit(proxy.data(),sd,screen_cfg,feature,screen_mask.empty()?screen_zeros.fit_mask(true):screen_mask.data(),fixed,0,true);
                require(ng_activity_symbol_values(fit.initial_symbols.data(),activity.data())==0);auto learned=distribution(fit.initial_symbols,activity,false,entropy_K,&screen_zeros);
                double cost=init_flags?complement(fit.initial_symbols,fit.params.size()+8):score(learned)*voxels/N+fit.params.size()+8;
                if(cost<best){best=cost;third=m;assisted=true;stats[2]=std::move(learned);screen_symbols[2]=fit.initial_symbols;}
            };
            if(forced_third>=0)consider_third(unsigned(forced_third)%12,forced_third<12,forced_third>=12);
            else{
                for(auto m:top)if(m)consider_third(m,true,true);
                for(int id:advice_third)if(id>=0)consider_third(unsigned(id)%12,id<12,id>=12);
            }
            if(singleton_trials){
                double costs[]={score(stats[0])*voxels/N+plain.params.size(),score(stats[1])*voxels/N,best};
                singleton_seed=std::min_element(costs,costs+3)-costs;
            }
            if((options&2)&&!(options&1))for(unsigned k=0;k<3;k++){
                require(ng_activity_symbol_values(screen_symbols[k].data(),activity.data())==0);
                stats[k]=distribution(screen_symbols[k],activity,true,entropy_K,&screen_zeros);
            }
        }
        diagnostics[0]=third;diagnostics[1]=assisted;
        if(profile_enabled)profile_seconds[0]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-profile_start).count();
        profile_start=std::chrono::steady_clock::now();
        std::vector<Bytes> candidate(36);candidate[0].resize(N);candidate[third].resize(N);
        bank_fixed_candidates(src,candidate,d,(1u<<third)|1u);
        std::array<small_bank::Fit,2> fits;fits[0]=small_bank::plain_fit(src,cfg,zeros.fit_mask());
        Bytes feature(assisted?N:0);
        if(assisted){for(size_t i=0;i<N;i++)feature[i]=uint8_t(src[i]-unpyramid(candidate[third][i]));
            Bytes mask;
            if(init_flags&2){auto base=init_block_costs(fits[0].initial_symbols,d),direct=init_block_costs(candidate[0],d);
                for(size_t b=0;b<base.size();b++)base[b]=std::min(base[b],direct[b]);mask=underserved_mask(base,d);}
            fits[1]=small_bank::fit(src,d,cfg,feature,mask.empty()?zeros.fit_mask():mask.data(),candidate,1u<<third,true);}
        unsigned ids[3]={12,0,assisted?24+third:third};
        if(profile_enabled)profile_seconds[1]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-profile_start).count();
        profile_start=std::chrono::steady_clock::now();
        if(shared_grid::enabled){
            std::vector<shared_grid::Candidate> menu={{0,12,&fits[0],fits[0].base[0].data()},{0,0,nullptr,nullptr},
                {third,ids[2],assisted?&fits[1]:nullptr,assisted?fits[1].base[third].data():nullptr}};
            auto frame=shared_grid::run(src,d,menu,candidate,zeros,false);
            require(frame.size()<=cap);memcpy(out,frame.data(),frame.size());*written=frame.size();return 0;
        }
        candidate[12].resize(N);if(assisted)candidate[24+third].resize(N);
        std::vector<uint32_t>block(N);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++){
            unsigned b=((z/choice_grid::edge())*nb+y/choice_grid::edge())*nb;
            for(unsigned x=0;x<d;x+=choice_grid::edge(),b++)
                std::fill_n(block.data()+(size_t(z)*d+y)*d+x,choice_grid::edge(),b);
        }
        Bytes choices(blocks,0),canonical(blocks,12),symbols(N);std::vector<uint16_t>activity(N);
        if(profile_enabled)profile_seconds[2]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-profile_start).count();
        auto evaluate=[&](bool selected_only){
            Timer timer(3);
            const uint8_t* pp[24];size_t sizes[24];const uint16_t* bp[24];uint8_t*wp[36]={};
            for(unsigned k=0;k<24;k++){unsigned f=k/12;auto&fit=f&&assisted?fits[1]:fits[0];
                pp[k]=fit.params.data();sizes[k]=fit.params.size();bp[k]=f&&assisted?fit.base[third].data():fit.base[0].data();}
            for(auto id:ids)wp[id]=candidate[id].data();
            for(size_t b=0;b<blocks;b++)canonical[b]=zeros.any()&&zeros.empty[b]?0:ids[choices[b]];
            require((selected_only?ng_bank_selected_symbols:ng_bank_symbols)(24,pp,sizes,bp,canonical.data(),src,wp,symbols.data(),activity.data())==0);
        };
        unsigned selection_calls=0;
        auto block_costs=[&](){
            if(search_flags&1)for(auto&s:stats)s.normalized_costs();
            std::vector<std::array<double,3>>cost(blocks);
            for(unsigned k=0;k<3;k++){
                auto cl=activity_classify(activity,stats[k].thr);
                if(span_selection&&!(ng_screen_work&&ng_screen_size==N)){
                    const unsigned edge=choice_grid::edge();const auto&table=stats[k].cost;const auto&sy=candidate[ids[k]];
                    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++){
                        unsigned b=((z/edge)*nb+y/edge)*nb;size_t i=(size_t(z)*d+y)*d;
                        for(unsigned x=0;x<d;x+=edge,b++,i+=edge){
                            if(zeros.any()&&zeros.empty[b])continue;
                            double sum=cost[b][k];
                            for(unsigned j=0;j<edge;j++)sum+=table[cl[i+j]][sy[i+j]];
                            cost[b][k]=sum;
                        }
                    }
                }else for(size_t i=0;i<N;i++)if((!zeros.any()||zeros.mask[i])&&(!ng_screen_work||ng_screen_size!=N||ng_screen_work[i]))cost[block[i]][k]+=stats[k].cost[cl[i]][candidate[ids[k]][i]];
            }
            if((search_flags&2)&&selection_calls){
                uint64_t hist[256]={};for(auto c:choices)hist[c]++;
                uint16_t freq[256];delta_rans_size_lower_bound(hist,blocks,freq);
                for(auto&v:cost)for(unsigned k=0;k<3;k++)v[k]+=12.-std::log2(double(std::max(1u,unsigned(freq[k]))));
            }
            return cost;
        };
        auto select=[&](){
            Timer timer(5);
            auto cost=block_costs();
            if(adaptive_grid::root)adaptive_grid::select(d,cost,choices);
            else if((search_flags&8)&&selection_calls){
                struct Change{double gain;size_t block;unsigned choice;};std::vector<Change>changes;
                for(size_t b=0;b<blocks;b++){
                    unsigned best=std::min_element(cost[b].begin(),cost[b].end())-cost[b].begin();
                    double gain=cost[b][choices[b]]-cost[b][best];
                    if(best!=choices[b]&&gain>0)changes.push_back({gain,b,best});
                }
                std::sort(changes.begin(),changes.end(),[](const Change&a,const Change&b){return a.gain!=b.gain?a.gain>b.gain:a.block<b.block;});
                for(size_t i=0;i<(changes.size()+1)/2;i++)choices[changes[i].block]=uint8_t(changes[i].choice);
            }else for(size_t b=0;b<blocks;b++)choices[b]=std::min_element(cost[b].begin(),cost[b].end())-cost[b].begin();
            selection_calls++;
            if(zeros.any())for(size_t b=0;b<blocks;b++)if(zeros.empty[b])choices[b]=1;
        };
        auto thresholds=[&](bool final=false){
            Timer timer(6);
            auto as=zeros.filtered()?zeros.take(symbols):Bytes{};auto aa=zeros.filtered()?zeros.take(activity):std::vector<uint16_t>{};
            auto ab=zeros.filtered()?zeros.take(block):std::vector<uint32_t>{};
            const auto&sy=zeros.filtered()?as:symbols;const auto&ac=zeros.filtered()?aa:activity;const auto&bl=zeros.filtered()?ab:block;
            auto thr=activity_quantiles(ac,entropy_K);
            unsigned grid=(final||threshold_feedback)?threshold_grid:0;
            if(options&2)thr=activity_refine(sy,ac,thr,bl,choices,3,true,grid?grid:64,grid!=0);
            return thr;};
        auto refresh_at=[&](const std::vector<uint16_t>&thr){auto cl=activity_classify(activity,thr);
            for(auto&s:stats){s.hist.assign(entropy_K,{});s.cost.resize(entropy_K);s.thr=thr;}
            for(size_t i=0;i<N;i++)if((!zeros.any()||zeros.mask[i])&&(!ng_screen_targets||ng_screen_size!=N||ng_screen_targets[i]))stats[choices[block[i]]].hist[cl[i]][symbols[i]]++;
            for(auto&s:stats)s.finish();};
        auto refresh=[&](){refresh_at(thresholds());};
        evaluate(false);
        if(options&1){Timer timer(4);for(unsigned k=0;k<3;k++)stats[k]=distribution(candidate[ids[k]],activity,options&2,entropy_K,&zeros);}
        select();evaluate(true);
        for(auto c:choices)diagnostics[2+c]++;
        auto refit=[&](){
            for(unsigned k: {0u,2u})if(k==0||assisted){
                Bytes mask(N);size_t count=0;for(size_t i=0;i<N;i++){mask[i]=choices[block[i]]==k;count+=mask[i];}
                if(count>=4096){int refit_cfg[6];std::copy_n(cfg,6,refit_cfg);refit_cfg[5]=std::max(2u,std::min(128u,unsigned(count/512)));
                    if(k==0)fits[0]=small_bank::plain_fit(src,refit_cfg,mask.data(),(search_flags&4)?&fits[0]:nullptr);
                    else fits[1]=small_bank::fit(src,d,refit_cfg,feature,mask.data(),candidate,1u<<third,true,nullptr,nullptr,nullptr,nullptr,(search_flags&4)?&fits[1]:nullptr);
                    diagnostics[8+k/2]++;
                }
            }
        };
        if(passes==2){
            refit();
            evaluate(false);
            refresh();select();evaluate(true);
        }
        for(unsigned round=0;round<extra_selections;round++){
            if(round<refit_rounds)refit();
            evaluate(false);refresh();select();evaluate(true);
        }
        auto pack=[&](const std::vector<uint16_t>&thr){Timer timer(7);auto cl=activity_classify(activity,thr);
        if(choice_frame::enabled){
            std::vector<Bytes> models={fits[0].params};if(assisted)models.push_back(fits[1].params);
            std::vector<choice_frame::Choice> menu={{0,0,0,{}},{0,255,1,{}},{third,assisted?1u:255u,2,assisted?fits[1].weights:small_bank::Weight{}}};
            return choice_frame::write(d,models,menu,choices,symbols,thr,cl,1,&zeros);
        }
        Bytes header;put32(header,d);put32(header,third);put32(header,assisted);
        for(unsigned k=0;k<(assisted?2u:1u);k++){
            put32(header,fits[k].params.size());packed_bank::append(header,fits[k].params);
            for(auto w:fits[k].weights)put32(header,uint32_t(w));
        }
        return finish_frame(std::move(header),symbols,choices,block,thr,cl,"N3C1");
        };
        auto thr=thresholds(true);auto frame=pack(thr);
        if(singleton_trials){
            require(choice_frame::enabled&&!adaptive_grid::root&&!adaptive_grid::s2_root&&!optimize);
            // Reuse the existing fits; regenerate the causal symbol/activity
            // history for each singleton. Mixed-history residuals are not valid
            // substitutes for a uniform assignment.
            for(unsigned k=0;k<3;k++){
                if(singleton_trials==2&&k!=singleton_seed)continue;
                std::fill(choices.begin(),choices.end(),k);evaluate(true);
                auto sy=zeros.filtered()?zeros.take(symbols):symbols;
                auto ac=zeros.filtered()?zeros.take(activity):activity;
                auto t=activity_quantiles(ac,entropy_K);
                t=activity_refine(sy,ac,t,{}, {},1,true,64,true);
                auto cl=activity_classify(activity,t);
                std::vector<Bytes> model;
                choice_frame::Choice c{k==2?third:0,choice_frame::no_model,0,{}};
                if(k==0||(k==2&&assisted)){auto&f=fits[k==0?0:1];model.push_back(f.params);c.model=0;if(k==2)c.weights=f.weights;}
                auto trial=choice_frame::write(d,model,{c},Bytes(blocks,0),symbols,t,cl,1,&zeros);
                if(trial.size()<frame.size())frame=std::move(trial);
            }
            require(frame.size()<=cap);memcpy(out,frame.data(),frame.size());*written=frame.size();return 0;
        }
        if(adaptive_grid::s2_root){
            // Finish the unchanged Slow initialization before changing its grid.
            // Preserve voxel fit/zero eligibility; only expand the selector map.
            using namespace adaptive_grid;
            std::fill_n(s2_stats,8,0);s2_stats[0]=frame.size();
            if(exact_partition::enabled)require(frame==exact_partition::reference);
            auto old_choices=choices;auto old_empty=zeros.empty;unsigned old_nb=nb,ratio=choice_grid::edge()/2;
            choice_grid::Scope fine(2);adaptive_grid::Scope tree(s2_root);
            nb=d/2;blocks=choice_grid::count(d);choices.resize(blocks);canonical.resize(blocks);
            if(zeros.any()){zeros.empty.resize(blocks);zeros.count=0;}
            for(unsigned z=0;z<nb;z++)for(unsigned y=0;y<nb;y++)for(unsigned x=0;x<nb;x++){
                size_t b=index(x,y,z,nb),p=index(x/ratio,y/ratio,z/ratio,old_nb);
                choices[b]=old_choices[p];
                if(!old_empty.empty()){zeros.empty[b]=old_empty[p];zeros.count+=zeros.empty[b];}
            }
            for(size_t i=0;i<N;i++)block[i]=choice_grid::block(i,d);
            // The same assignment can already benefit from tree compression.
            auto initial=pack(thr);if(initial.size()<frame.size())frame=std::move(initial);
            search_flags|=1;
            if(exact_partition::enabled){
                auto ranked_costs=[&](){refresh_at(thr);return block_costs();};
                exact_partition::run(d,choices,zeros,frame,thr,evaluate,ranked_costs,thresholds,pack);
                require(frame.size()<=cap);memcpy(out,frame.data(),frame.size());*written=frame.size();return 0;
            }
            for(unsigned round=0;round<s2_rounds;round++){
                Bytes scored_choices=choices;
                if((s2_options&32)&&zeros.any())for(size_t b=0;b<blocks;b++)if(zeros.empty[b])scored_choices[b]=3;
                auto tokens=serialize(scored_choices,d);double counts[256]={};
                for(auto t:tokens)counts[t]++;
                for(unsigned k: {0u,1u,2u,3u,255u})s2_cost[k]=-std::log2((counts[k]+.5)/(tokens.size()+((s2_options&32)?2.5:2.)));
                evaluate(false);refresh_at(thr);auto cost=block_costs();
                if(zeros.any())for(size_t b=0;b<blocks;b++)if(zeros.empty[b])
                    cost[b]={1e30,0.,1e30};
                std::vector<std::vector<std::array<double,3>>> size_cost;
                if(s2_options&(2|8)){
                    Bytes labeled=choices,levels(blocks);if(zeros.any())for(size_t b=0;b<blocks;b++)if(zeros.empty[b])labeled[b]=3;
                    serialize(labeled,d,&levels);
                    unsigned bins=(s2_options&16)?8:4,step=16/bins;
                    auto cl=activity_classify(activity,thr);for(auto&c:cl)c/=step;
                    std::vector<Stats> ss(12,Stats(bins));
                    for(size_t i=0;i<N;i++)if(!zeros.any()||zeros.mask[i]){
                        unsigned b=block[i],l=(s2_options&2)?levels[b]:0;
                        ss[4*choices[b]+l].hist[cl[i]][symbols[i]]++;
                    }
                    for(unsigned k=0;k<3;k++)for(unsigned l=0;l<4;l++){
                        auto&st=ss[4*k+l];
                        for(unsigned c=0;c<bins;c++){
                            uint64_t total=0;for(auto h:st.hist[c])total+=h;
                            if(!total)for(unsigned j=0;j<step;j++)for(unsigned s=0;s<256;s++)st.hist[c][s]+=stats[k].hist[step*c+j][s];
                        }st.normalized_costs();
                    }
                    for(unsigned l=0;l<4;l++){
                        unsigned edge=2u<<l,n=d/edge;size_cost.emplace_back(size_t(n)*n*n);
                        auto&v=size_cost.back();
                        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
                            size_t i=(size_t(z)*d+y)*d+x,b=index(x/edge,y/edge,z/edge,n);
                            if(zeros.any()&&!zeros.mask[i]){v[b][0]=v[b][2]=1e30;continue;}
                            for(unsigned k=0;k<3;k++)v[b][k]+=ss[4*k+((s2_options&2)?l:0)].cost[cl[i]][candidate[ids[k]][i]];
                        }
                    }
                }
                double original_cost[256];std::copy_n(s2_cost,256,original_cost);
                // Explicit zero leaves: writer and DP must describe the same
                // topology. Zero is not a free direct-predictor sample inside
                // a larger mixed leaf: the writer emits a distinct choice.
                std::vector<std::vector<std::array<double,4>>> explicit_cost;
                if(s2_options&32){
                    std::vector<uint64_t> nonzero(blocks),zero(blocks);
                    for(size_t b=0;b<blocks;b++){zero[b]=zeros.any()&&zeros.empty[b];nonzero[b]=!zero[b];}
                    auto sums=cost;unsigned n=nb;
                    for(unsigned l=0;l<4;l++){
                        const auto&cc=size_cost.empty()?sums:size_cost[l];
                        explicit_cost.emplace_back(cc.size());auto&v=explicit_cost.back();
                        for(size_t b=0;b<cc.size();b++){
                            for(unsigned k=0;k<3;k++)v[b][k]=zero[b]?1e30:cc[b][k];
                            v[b][3]=nonzero[b]?1e30:0.;
                        }
                        if(l==3)break;
                        unsigned m=n/2;std::vector<std::array<double,3>>next(size_t(m)*m*m);
                        std::vector<uint64_t> nz(next.size()),zz(next.size());
                        for(unsigned z=0;z<m;z++)for(unsigned y=0;y<m;y++)for(unsigned x=0;x<m;x++){
                            size_t b=index(x,y,z,m);
                            for(unsigned a=0;a<8;a++){size_t c=index(2*x+(a&1),2*y+((a>>1)&1),2*z+(a>>2),n);
                                nz[b]+=nonzero[c];zz[b]+=zero[c];for(unsigned k=0;k<3;k++)next[b][k]+=sums[c][k];}
                        }sums=std::move(next);nonzero=std::move(nz);zero=std::move(zz);n=m;
                    }
                }
                // Complete-frame line search: table, selector, alignment and
                // causal-history costs are measured together, not guessed.
                const double scales[]={1.,.25,.5,2.,4.};
                for(unsigned trial=0;trial<((s2_options&1)?5u:1u);trial++){
                    for(unsigned k: {0u,1u,2u,3u,255u})s2_cost[k]=original_cost[k]*scales[trial];
                    auto prior_choices=choices;auto prior_thr=thr;
                    if(s2_options&32){
                        adaptive_grid::select(d,explicit_cost[0],choices,&explicit_cost);
                        for(auto&c:choices)if(c==3)c=1; // evaluator uses its existing zero mask
                    }else adaptive_grid::select(d,cost,choices,size_cost.empty()?nullptr:&size_cost);
                    evaluate(true);thr=thresholds(true);auto proposal=pack(thr);
                    s2_stats[2]++;
                    if(proposal.size()<frame.size()){frame=std::move(proposal);s2_stats[3]++;}
                    else {choices=std::move(prior_choices);thr=std::move(prior_thr);evaluate(true);}
                }
            }
            s2_stats[1]=frame.size();
            require(frame.size()<=cap);memcpy(out,frame.data(),frame.size());*written=frame.size();return 0;
        }
        for(unsigned round=0;feedback&&round<feedback_passes;round++){
            evaluate(false);
            auto cl=activity_classify(activity,thr);
            feedback(src,d,candidate,ids,third,thr,cl,zeros,choices);
            evaluate(true); // changed choices change the causal residual history
            thr=thresholds(true);frame=pack(thr);
        }
        guarded_diagnostics[0]=frame.size();
        auto record_sweep=[&](){sweep_bytes.push_back(frame.size());sweep_seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-sweep_start).count());};
        record_sweep();
        if(optimize){
            // Every proposal is evaluated using the complete native frame. The
            // current parameters/choices/thresholds, not just bytes, are restored
            // on rejection. Causal histories are always regenerated globally.
            auto accept=[&](unsigned kind){
                guarded_diagnostics[2+2*kind]++;
                auto old_header=diagnostics[10];
                auto proposal=pack(thr);
                if(proposal.size()<frame.size()){
                    frame=std::move(proposal);guarded_diagnostics[3+2*kind]++;return true;
                }
                diagnostics[10]=old_header;return false;
            };
            auto choice_trial=[&](const Bytes&proposal,unsigned kind){
                if(proposal==choices)return;
                auto old=choices;choices=proposal;evaluate(false);
                if(!accept(kind)){choices=std::move(old);evaluate(false);}
            };
            for(unsigned round=0;round<sweeps;round++){
                size_t before=frame.size();
                unsigned groups=(optimize&2)?4:1;
                for(unsigned g=0;g<groups;g++){
                    evaluate(false);refresh_at(thr);auto cost=block_costs();auto proposal=choices;
                    for(size_t b=0;b<blocks;b++){
                        unsigned x=b%nb,y=(b/nb)%nb,z=b/(nb*nb);
                        if((x+y+z)%groups==g)proposal[b]=std::min_element(cost[b].begin(),cost[b].end())-cost[b].begin();
                    }
                    choice_trial(proposal,0);
                }
                if(optimize&4){
                    // Two best coordinated proposals at each region size per
                    // sweep. Allow negative estimated gains: these are escape
                    // proposals, accepted only if full-frame bytes improve.
                    for(unsigned edge: {16u,32u}){
                        unsigned rb=edge/choice_grid::edge(),nr=d/edge;if(!nr)continue;
                        std::vector<uint8_t>used(size_t(nr)*nr*nr);
                        for(unsigned trial=0;trial<2;trial++){
                            refresh_at(thr);auto cost=block_costs();
                            struct Region{double gain;size_t id;unsigned mode;};
                            Region best{-std::numeric_limits<double>::infinity(),0,0};
                            std::vector<std::array<double,3>>gains(used.size());
                            std::vector<std::array<unsigned,3>>changes(used.size());
                            for(size_t b=0;b<blocks;b++){
                                unsigned x=b%nb,y=(b/nb)%nb,z=b/(nb*nb);
                                size_t r=((z/rb)*nr+y/rb)*nr+x/rb;
                                for(unsigned k=0;k<3;k++){gains[r][k]+=cost[b][choices[b]]-cost[b][k];changes[r][k]+=choices[b]!=k;}
                            }
                            for(size_t r=0;r<used.size();r++)if(!used[r])for(unsigned k=0;k<3;k++)
                                if(changes[r][k]&&gains[r][k]>best.gain)best={gains[r][k],r,k};
                            if(!std::isfinite(best.gain))break;used[best.id]=1;
                            auto proposal=choices;
                            for(size_t b=0;b<blocks;b++){
                                unsigned x=b%nb,y=(b/nb)%nb,z=b/(nb*nb);
                                if(size_t(((z/rb)*nr+y/rb)*nr+x/rb)==best.id)proposal[b]=best.mode;
                            }
                            choice_trial(proposal,3);
                        }
                    }
                }
                auto old_thr=thr;thr=thresholds();if(!accept(1))thr=std::move(old_thr);
                for(unsigned k: {0u,2u})if(k==0||assisted){
                    Bytes mask(N),weights;size_t count=0;
                    for(size_t i=0;i<N;i++){mask[i]=choices[block[i]]==k;count+=mask[i];}
                    if(count<4096&&!(optimize&32))continue;
                    if(optimize&8){
                        refresh_at(thr);auto cl=activity_classify(activity,thr);weights.resize(N);
                        // IRLS surrogate: entropy excess above the modal symbol
                        // divided by squared wrapped residual. Bounded integer
                        // weights preserve the native exact Gram accumulation.
                        double floor_cost[16];for(unsigned c=0;c<16;c++)floor_cost[c]=*std::min_element(stats[k].cost[c].begin(),stats[k].cost[c].end());
                        for(size_t i=0;i<N;i++){
                            unsigned symbol=(optimize&32)?candidate[ids[k]][i]:symbols[i];
                            double e=(symbol+1)/2;
                            double excess=std::max(.25,stats[k].cost[cl[i]][symbol]-floor_cost[cl[i]]);
                            weights[i]=uint8_t(std::clamp(std::lround(32*excess/(1+e*e)),1l,32l));
                        }
                    }
                    if(optimize&32){
                        refresh_at(thr);auto cost=block_costs();std::vector<double>membership(blocks);
                        double temperature=(optimize&512)?32.:8.;
                        for(size_t b=0;b<blocks;b++){
                            double best=*std::min_element(cost[b].begin(),cost[b].end()),sum=0;
                            for(unsigned j=0;j<3;j++)sum+=std::exp2(-(cost[b][j]-best)/temperature);
                            membership[b]=std::exp2(-(cost[b][k]-best)/temperature)/sum;
                        }
                        if(weights.empty())weights.assign(N,8);count=0;
                        for(size_t i=0;i<N;i++){weights[i]=uint8_t(std::clamp(std::lround(4*weights[i]*membership[block[i]]),0l,255l));mask[i]=weights[i]>0;count+=mask[i];}
                        if(count<4096)continue;
                    }
                    unsigned fi=k/2;auto old_fit=std::move(fits[fi]);
                    int refit_cfg[6];std::copy_n(cfg,6,refit_cfg);refit_cfg[5]=std::max(2u,std::min(128u,unsigned(count/512)));
                    if(optimize&64)fits[fi]=small_bank::refine(src,d,refit_cfg,k?&feature:nullptr,mask.data(),weights.empty()?nullptr:weights.data(),candidate,third,old_fit);
                    else if(k==0)fits[fi]=small_bank::plain_fit(src,refit_cfg,mask.data(),nullptr,weights.empty()?nullptr:weights.data());
                    else fits[fi]=small_bank::fit(src,d,refit_cfg,feature,mask.data(),candidate,1u<<third,true,nullptr,nullptr,nullptr,nullptr,nullptr,weights.empty()?nullptr:weights.data());
                    diagnostics[8+fi]++;auto start_choices=choices;
                    auto try_model=[&](){
                        evaluate(false);if(accept(2))return true;
                        if(optimize&256){refresh_at(thr);select();evaluate(false);if(accept(2))return true;}
                        choices=start_choices;evaluate(false);return false;
                    };
                    if(!try_model()){
                        bool kept=false;small_bank::Fit best_fit;Bytes best_choices;
                        if(optimize&16){
                            auto full=std::move(fits[fi]);
                            for(unsigned shift=1;shift<=3;shift++){
                                choices=start_choices;
                                fits[fi]=small_bank::refine(src,d,refit_cfg,k?&feature:nullptr,nullptr,nullptr,candidate,third,old_fit,&full,shift);
                                if(try_model()){kept=true;best_fit=std::move(fits[fi]);best_choices=choices;}
                            }
                        }
                        fits[fi]=kept?std::move(best_fit):std::move(old_fit);choices=kept?std::move(best_choices):std::move(start_choices);evaluate(false);
                    }
                }
                if((optimize&128)&&frame.size()==before){
                    refresh_at(thr);auto rcost=block_costs();std::vector<double>expense(blocks);
                    for(size_t b=0;b<blocks;b++)expense[b]=rcost[b][choices[b]];
                    auto sorted=expense;std::nth_element(sorted.begin(),sorted.begin()+3*blocks/4,sorted.end());
                    Bytes expensive(N);size_t count=0;for(size_t i=0;i<N;i++){expensive[i]=expense[block[i]]>=sorted[3*blocks/4];count+=expensive[i];}
                    int alt_cfg[6];std::copy_n(cfg,6,alt_cfg);alt_cfg[5]=std::max(2u,std::min(128u,unsigned(count/512)));
                    unsigned trials=0;
                    for(unsigned mode:screen_order)if(mode&&mode!=third){
                        if(trials++>=2)break;
                        candidate[mode].resize(N);bank_fixed_candidates(src,candidate,d,1u<<mode);
                        for(bool learned: {false,true}){
                            unsigned old_third=third;bool old_assisted=assisted;auto old_feature=std::move(feature);auto old_fit=std::move(fits[1]);auto old_choices=choices;
                            third=mode;assisted=learned;ids[2]=learned?24+mode:mode;candidate[ids[2]].resize(N);
                            if(learned){
                                feature.resize(N);for(size_t i=0;i<N;i++)feature[i]=uint8_t(src[i]-unpyramid(candidate[mode][i]));
                                fits[1]=small_bank::fit(src,d,alt_cfg,feature,expensive.data(),candidate,1u<<mode,true);
                            }
                            evaluate(false);refresh_at(thr);select();evaluate(false);
                            if(!accept(4)){
                                third=old_third;assisted=old_assisted;ids[2]=assisted?24+third:third;feature=std::move(old_feature);fits[1]=std::move(old_fit);choices=std::move(old_choices);evaluate(false);
                            }
                        }
                    }
                }
                record_sweep();if(sweeps>8&&frame.size()==before)break;
            }
        }
        guarded_diagnostics[1]=frame.size();
        for(auto c:choices)diagnostics[5+c]++;
        require(frame.size()<=cap);std::copy(frame.begin(),frame.end(),out);*written=frame.size();return 0;
    }catch(...){return -1;}
}
extern "C" int ng_codec_encode_s2(const uint8_t*s,unsigned d,unsigned root,unsigned minimum,unsigned rounds,uint8_t*out,size_t cap,size_t*n){
    if((root!=16&&root!=32)||(minimum!=2&&minimum!=4)||rounds<1||rounds>4)return -1;
    auto old=adaptive_grid::s2_root;auto oldmin=adaptive_grid::s2_min;auto oldrounds=adaptive_grid::s2_rounds;
    bool enabled=choice_frame::enabled;choice_frame::enabled=true;
    choice_grid::Scope grid(4);adaptive_grid::Scope tree(0);
    adaptive_grid::s2_root=root;adaptive_grid::s2_min=minimum;adaptive_grid::s2_rounds=rounds;
    std::fill_n(adaptive_grid::s2_stats,8,0);
    int rc=three_candidate_encode_impl(s,d,1,7,1,0,0,out,cap,n);
    choice_frame::enabled=enabled;adaptive_grid::s2_root=old;adaptive_grid::s2_min=oldmin;adaptive_grid::s2_rounds=oldrounds;return rc;
}
extern "C" void ng_codec_s2_stats(uint64_t*out){std::copy_n(adaptive_grid::s2_stats,8,out);}
// 1 exact-frame penalty line search; 2 predictor x size x4 difficulty banks;
// 4 raw tree tokens in joint Zstd header; 8 matched shared-bank control;
// 16 use8 rather than4 difficulty bins; 32 model zero leaves explicitly.
extern "C" int ng_codec_encode_s2_research(const uint8_t*s,unsigned d,unsigned flags,unsigned rounds,uint8_t*out,size_t cap,size_t*n){
    if(flags>63)return -1;
    auto old=adaptive_grid::s2_options;adaptive_grid::s2_options=flags;
    int rc=ng_codec_encode_s2(s,d,16,2,rounds,out,cap,n);
    adaptive_grid::s2_options=old;return rc;
}
// Guarded exact-byte refinement of ABC3 fixed-fit initialization.
extern "C" int ng_three_candidate_encode_init(const uint8_t*src,unsigned d,unsigned flags,uint8_t*out,size_t cap,size_t*written){
    if(flags!=0&&flags!=1&&flags!=3)return -1;
    return three_candidate_encode_impl(src,d,1,7,2,0,0,out,cap,written,0,0,flags);
}
// 1 guard, 2 four spatial groups, 4 regional escape, 8 entropy-weighted fit.
extern "C" int ng_three_candidate_encode_guarded(const uint8_t*src,unsigned d,unsigned flags,unsigned sweeps,uint8_t*out,size_t cap,size_t*written){
    if(flags<1||flags>1023||!(flags&1)||sweeps<1||sweeps>32)return -1;
    return three_candidate_encode_impl(src,d,1,7,2,0,0,out,cap,written,flags,sweeps);
}
extern "C" int ng_three_candidate_encode_options(const uint8_t*src,unsigned d,unsigned passes,unsigned options,uint8_t*out,size_t cap,size_t*written){
    return three_candidate_encode_impl(src,d,passes,options,(options&4)?1:0,0,0,out,cap,written);
}
extern "C" int ng_slow_singleton_encode(const uint8_t*s,unsigned d,uint8_t*out,size_t cap,size_t*n){
    bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=three_candidate_encode_impl(s,d,1,7,1,0,0,out,cap,n,0,0,0,0,false,nullptr,16,true);
    choice_frame::enabled=old;return rc;
}
extern "C" int ng_slow_singleton_screen_encode(const uint8_t*s,unsigned d,uint8_t*out,size_t cap,size_t*n){
    bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=three_candidate_encode_impl(s,d,1,7,1,0,0,out,cap,n,0,0,0,0,false,nullptr,16,2);
    choice_frame::enabled=old;return rc;
}
extern "C" int ng_codec_encode_slow_config(const uint8_t*s,unsigned d,const int*cfg,unsigned K,uint8_t*out,size_t cap,size_t*n){
    if(!cfg||cfg[0]<1||cfg[0]>3||cfg[1]<-1||cfg[1]>3||cfg[2]<-1||cfg[2]>3||cfg[3]<2||cfg[3]>32||cfg[4]<2||cfg[4]>160||cfg[5]<1||cfg[5]>128||K<2||K>32)return -1;
    bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=three_candidate_encode_impl(s,d,1,7,1,0,0,out,cap,n,0,0,0,0,false,cfg,K,2);
    choice_frame::enabled=old;return rc;
}
// Generic screened menu search. No named preset is specially retained.
extern "C" int ng_codec_encode_menu_evidence(const uint8_t*s,unsigned d,unsigned flags,uint8_t*out,size_t cap,size_t*n){
    if(flags>3)return -1;
    const int cfg[6]={3,3,1,16,40,1};bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=ng_screen_pair_encode_impl(s,d,12,out,cap,n,cfg,16,3,true,nullptr,flags);
    choice_frame::enabled=old;return rc;
}
extern "C" unsigned ng_menu_search_scores(double*out,unsigned cap){
    unsigned n=fast_split::menu_scores.size();if(out)std::copy_n(fast_split::menu_scores.data(),std::min(n,cap),out);return n;
}
extern "C" int ng_codec_diagnose_slow_init(const uint8_t*s,unsigned d,unsigned full_stats,uint8_t*out,size_t cap,size_t*n){
    if(full_stats>1)return -1;
    const int cfg[6]={3,3,1,16,40,1};bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=three_candidate_encode_impl(s,d,1,6|full_stats,1,0,0,out,cap,n,0,0,0,0,false,cfg,16);
    choice_frame::enabled=old;return rc;
}
// Diagnostic: compare an explicitly supplied menu through the same engine.
extern "C" int ng_codec_diagnose_menu(const uint8_t*s,unsigned d,const unsigned*ids,unsigned count,uint8_t*out,size_t cap,size_t*n){
    if(!ids||count<1||count>3)return -1;
    std::vector<unsigned>menu(ids,ids+count);for(auto id:menu)if(id>24)return -1;
    const int cfg[6]={3,3,1,16,40,1};bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=ng_screen_pair_encode_impl(s,d,12,out,cap,n,cfg,16,3,true,&menu);
    choice_frame::enabled=old;return rc;
}
extern "C" int ng_codec_encode_nested_slow(const uint8_t*s,unsigned d,unsigned depth,uint8_t*out,size_t cap,size_t*n){
    if(depth<1||depth>3)return -1;
    const int cfg[6]={3,3,1,16,40,1};bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=ng_screen_pair_encode_impl(s,d,12,out,cap,n,cfg,16,depth,true);
    choice_frame::enabled=old;return rc;
}
extern "C" int ng_three_candidate_encode_abc(const uint8_t*src,unsigned d,unsigned selections,uint8_t*out,size_t cap,size_t*written){
    if(selections<1||selections>16)return -1;
    return three_candidate_encode_impl(src,d,1,7,selections-1,0,0,out,cap,written);
}
// NCF1 threshold-only / feedback ablation, using the same bounded ABC engine.
extern "C" int ng_codec_encode_three_thresholds(const uint8_t*src,unsigned d,unsigned selections,unsigned refits,unsigned grid,unsigned feedback,uint8_t*out,size_t cap,size_t*written){
    if(selections<1||selections>16||refits>=selections||(grid!=0&&grid!=64&&grid!=256)||feedback>1)return -1;
    bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=three_candidate_encode_impl(src,d,1,7,selections-1,refits,0,out,cap,written,0,0,0,grid,feedback!=0);
    choice_frame::enabled=old;return rc;
}
extern "C" int ng_three_candidate_encode_abc_schedule(const uint8_t*src,unsigned d,unsigned selections,unsigned refit_rounds,uint8_t*out,size_t cap,size_t*written){
    if(selections<1||selections>16||refit_rounds>=selections)return -1;
    return three_candidate_encode_impl(src,d,1,7,selections-1,refit_rounds,0,out,cap,written);
}
// N=1 normalized residual costs; S=2 selector costs; M=4 half model updates;
// B=8 strongest-half block updates. No retention/acceptance safeguards.
extern "C" int ng_three_candidate_encode_search(const uint8_t*src,unsigned d,unsigned selections,unsigned refits,unsigned flags,uint8_t*out,size_t cap,size_t*written){
    if(selections<1||selections>16||refits>=selections)return -1;
    return three_candidate_encode_impl(src,d,1,7,selections-1,refits,flags,out,cap,written);
}
extern "C" int ng_three_candidate_encode(const uint8_t*src,unsigned d,unsigned passes,uint8_t*out,size_t cap,size_t*written){
    return ng_three_candidate_encode_options(src,d,passes,0,out,cap,written);
}
extern "C" int ng_three_candidate_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    try{
        require(src&&out&&n>=12&&!memcmp(src,"N3C1",4));Bytes f(src,src+n);size_t p=4;
        size_t hn=get32(f,p),zn=get32(f,p);require(zn<=n-p);
        auto header=packed_bank::decompress(src+p,zn,hn);p+=zn;packed_bank::Reader h{header};
        unsigned d=h.u(4),third=h.u(4),assisted=h.u(4);require(ng_geometry_valid(d,d,d)&&third>0&&third<12&&assisted<=1&&cap>=size_t(d)*d*d);
        GeometryScope scope(d,d,d);using State=std::unique_ptr<void,decltype(&ng_causal_destroy)>;
        std::vector<State> states;void*models[24];int32_t weights[48]={};
        for(unsigned k=0;k<(assisted?2u:1u);k++){
            auto params=h.take(h.u(4));states.emplace_back(k?ng_causal_create_model(params.data(),params.size()):ng_causal_create(params.data(),params.size()),ng_causal_destroy);require(bool(states.back()));
            int32_t w[2];for(auto&v:w)v=int32_t(packed_bank::signed_value(h.u(4),32));
            for(unsigned m=0;m<12;m++){models[12*k+m]=states.back().get();for(unsigned j=0;j<2;j++)weights[24*k+2*m+j]=m?w[j]:0;}
        }
        if(!assisted)for(unsigned m=12;m<24;m++)models[m]=models[0];
        auto choices=three_candidate::read_residuals(h,f,p,d,out);
        for(auto&c:choices)c=c==0?12:c==1?0:assisted?24+third:third;
        require(ng_bank_reconstruct_many(models,weights,choices.data(),out,24)==0);return 0;
    }catch(...){return -1;}
}
