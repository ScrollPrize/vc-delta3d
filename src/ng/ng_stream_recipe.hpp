// Slow retains direct+plain NG+one screened third choice and singleton recovery.
extern "C" int ng_slow_recipe_encode(const uint8_t*s,unsigned d,unsigned edge,int forced,int first,int second,
    uint8_t*out,size_t cap,size_t*n){
    auto valid=[](int id){return id==-1||(id>0&&id<24&&id!=12);};
    if(!valid(forced)||!valid(first)||!valid(second))return -1;
    struct Restore{int old=three_candidate::forced_third;std::array<int,2>advice=three_candidate::advice_third;
        ~Restore(){three_candidate::forced_third=old;three_candidate::advice_third=advice;}}restore;
    try{choice_grid::Scope grid(edge);three_candidate::forced_third=forced;three_candidate::advice_third={first,second};
        return ng_codec_encode(s,d,1,out,cap,n);
    }catch(...){return -1;}
}
// Exactly the public Fast encoder, with only its screening decision supplied.
// Predictor 0..11 is bare;12..23 is Fast's corresponding NG-input predictor.
extern "C" int ng_fast_recipe_encode(const uint8_t*s,unsigned d,unsigned edge,int predictor,
    uint8_t*out,size_t cap,size_t*n){
    if(predictor< -1||predictor>=24)return -1;
    struct Restore{int old=fast_split::forced_uniform;~Restore(){fast_split::forced_uniform=old;}}restore;
    try{choice_grid::Scope grid(edge);fast_split::forced_uniform=predictor;
        return ng_codec_encode(s,d,3,out,cap,n);
    }catch(...){return -1;}
}
extern "C" unsigned ng_fast_recipe_selected(){return fast_split::last_uniform;}
// Stream advice augments the ordinary screen; it never forces a winner.
extern "C" int ng_fast_guided_encode(const uint8_t*s,unsigned d,unsigned edge,int predictor,
    uint8_t*out,size_t cap,size_t*n){
    if(predictor< -1||predictor>=24)return -1;
    struct Restore{int old=fast_split::extra_uniform;~Restore(){fast_split::extra_uniform=old;}}restore;
    fast_split::extra_uniform=predictor;
    return ng_fast_recipe_encode(s,d,edge,-1,out,cap,n);
}
extern "C" int ng_fast_guided_pair_encode(const uint8_t*s,unsigned d,unsigned edge,int first,int second,
    uint8_t*out,size_t cap,size_t*n){
    if(second< -1||second>=24)return -1;
    struct Restore{int old=fast_split::extra_uniform_second;~Restore(){fast_split::extra_uniform_second=old;}}restore;
    fast_split::extra_uniform_second=second;
    return ng_fast_guided_encode(s,d,edge,first,out,cap,n);
}
// Encoder-only recipes. Frames retain all models/tables for independent decode.
namespace stream_recipe {
thread_local std::vector<std::array<unsigned,5>> last_recipes;
struct Capture {
    struct Item { Bytes frame; std::vector<unsigned> menu; unsigned edge; };
    std::vector<Item> items;
    choice_frame::Observer old;
    static thread_local Capture* active;
    Capture():old(choice_frame::observer){require(!active);active=this;choice_frame::observer=observe;}
    ~Capture(){choice_frame::observer=old;active=nullptr;}
    static void observe(unsigned,const std::vector<Bytes>&,const std::vector<choice_frame::Choice>&menu,
        const Bytes&selectors,const Bytes&,const std::vector<uint16_t>&,const Bytes&,unsigned,const Bytes&frame){
        Item item;item.frame=frame;item.edge=choice_grid::edge();
        for(unsigned j=0;j<menu.size();j++){auto&c=menu[j];if(c.predictor!=choice_frame::zero_predictor&&std::find(selectors.begin(),selectors.end(),j)!=selectors.end()){
            unsigned id=c.model==255?c.predictor:c.predictor==0&&c.weights==small_bank::Weight{}?24:12+c.predictor;
            if(std::find(item.menu.begin(),item.menu.end(),id)==item.menu.end())item.menu.push_back(id);
        }}
        active->items.push_back(std::move(item));
    }
};
thread_local Capture* Capture::active=nullptr;
struct Scope {
    bool enabled=choice_frame::enabled;double intercept=ng_fit_intercept_weight;
    const std::vector<unsigned>*old=fast_split::stream_menu;
    bool onepass=fast_split::stream_onepass;
    explicit Scope(unsigned level){choice_frame::enabled=true;ng_fit_intercept_weight=level>=2?100:0;}
    ~Scope(){choice_frame::enabled=enabled;ng_fit_intercept_weight=intercept;fast_split::stream_menu=old;fast_split::stream_onepass=onepass;}
};
void config(unsigned level,int*cfg){int c[6]={3,3,1,16,40,128};std::copy_n(c,6,cfg);
    if(level>=2)cfg[5]=1;if(level==3){cfg[2]=3;cfg[4]=160;}}
Bytes uniform(const uint8_t*src,unsigned d,unsigned level,unsigned id){
    GeometryScope geometry(d,d,d);zero_blocks::Scan zeros(src,d,true);
    if(zeros.all())return choice_frame::empty_frame(d);
    int cfg[6];config(level,cfg);unsigned pred=id==24?0:id%12;
    std::vector<Bytes>candidate(12);candidate[pred].resize(N);
    bank_fixed_candidates(src,candidate,d,1u<<pred);
    std::vector<Bytes>params;std::vector<small_bank::Weight>weights;small_bank::Fit fit;
    Bytes symbols;std::vector<uint16_t>activity(N);std::array<uint32_t,2560>hist{};
    if(id>=12){
        if(id==24){fit=small_bank::plain_fit(src,cfg,zeros.fit_mask());symbols=std::move(fit.initial_symbols);
            require(ng_activity_symbol_histogram(symbols.data(),activity.data(),hist.data())==0);
        }else{
            Bytes feature(N);for(size_t i=0;i<N;i++)feature[i]=uint8_t(src[i]-unpyramid(candidate[pred][i]));
            bool full_base=zeros.any()||small_bank::stream_fits;
            fit=small_bank::fit(src,d,cfg,feature,zeros.fit_mask(),candidate,full_base?1u<<pred:0,true,full_base?nullptr:&candidate[pred],full_base?nullptr:activity.data(),full_base?nullptr:hist.data());
            symbols=std::move(fit.initial_symbols);
            if(full_base&&!zeros.any())require(ng_activity_symbol_histogram(symbols.data(),activity.data(),hist.data())==0);
        }
        params.push_back(std::move(fit.params));weights.push_back(fit.weights);
    }else{
        symbols=std::move(candidate[pred]);require(ng_activity_symbol_histogram(symbols.data(),activity.data(),hist.data())==0);
        Bytes empty(65536);size_t n=0;require(ng_bank_empty_model(empty.data(),empty.size(),&n)==0);empty.resize(n);params.push_back(std::move(empty));weights.push_back({0,0});
    }
    Bytes selected(choice_grid::count(d),id>=12?12+pred:pred);
    // Masked zero blocks have no residual payload, but still supply history.
    if(zeros.any())for(size_t b=0;b<selected.size();b++)if(zeros.empty[b]){
        selected[b]=0;
        for(unsigned z=0;z<choice_grid::edge();z++)for(unsigned y=0;y<choice_grid::edge();y++){
            unsigned nb=d/choice_grid::edge(),e=choice_grid::edge();size_t pos=((b/(nb*nb)*e+z)*d+(b/nb%nb)*e+y)*d+(b%nb)*e;
            std::fill_n(symbols.begin()+pos,e,0);
        }
    }
    if(zeros.any()&&id>=12){
        candidate[0].resize(N);bank_fixed_candidates(src,candidate,d,1);
        const uint8_t*pp[12];size_t sz[12];const uint16_t*bp[12];uint8_t*wp[24]={};wp[0]=candidate[0].data();wp[12+pred]=symbols.data();
        for(unsigned m=0;m<12;m++){pp[m]=params[0].data();sz[m]=params[0].size();bp[m]=fit.base[pred].data();}
        require(ng_bank_selected_symbols(12,pp,sz,bp,selected.data(),src,wp,symbols.data(),activity.data())==0);
    }else if(zeros.any())require(ng_activity_symbol_histogram(symbols.data(),activity.data(),hist.data())==0);
    auto sy=zeros.filtered()?zeros.take(symbols):symbols;
    auto ac=zeros.filtered()?zeros.take(activity):activity;
    auto thr=activity_refine(sy,ac,activity_quantiles(ac,16),{}, {},1,true,64,true);
    return fast_split::frame(d,params,weights,selected,symbols,thr,{},activity.data(),&zeros);
}
}
// level0 Fast,1 Slow,2 RC1,3 RC2. Recipe words: edge,count,up to3 menu IDs.
extern "C" int ng_stream_discover(const uint8_t*s,unsigned d,unsigned level,unsigned sizes,
    unsigned*recipe,uint8_t*out,size_t cap,size_t*n){
    try{
        require(s&&recipe&&out&&n&&level<4&&sizes<2);
        stream_recipe::Scope scope(level);stream_recipe::Capture capture;
        Bytes best,buffer(cap);unsigned winning_edge=4;int cfg[6];stream_recipe::config(level,cfg);
        for(unsigned edge:{4u,8u,2u,16u,32u}){
            if(!sizes&&edge!=4)continue;choice_grid::Scope grid(edge);size_t bytes=0;int rc;
            if(level<2)rc=ng_codec_encode(s,d,level?1:3,buffer.data(),cap,&bytes);
            else rc=ng_screen_pair_encode_impl(s,d,12,buffer.data(),cap,&bytes,cfg,16,3,true,nullptr,3);
            require(rc==0);if(best.empty()||bytes<best.size()){best.assign(buffer.begin(),buffer.begin()+bytes);winning_edge=edge;}
        }
        std::fill_n(recipe,5,0);recipe[0]=winning_edge;
        for(auto&i:capture.items)if(i.frame==best){require(i.menu.size()<=3);recipe[1]=i.menu.size();std::copy(i.menu.begin(),i.menu.end(),recipe+2);break;}
        stream_recipe::last_recipes.clear();
        std::stable_sort(capture.items.begin(),capture.items.end(),[](const auto&a,const auto&b){return a.frame.size()<b.frame.size();});
        for(auto&i:capture.items)if(!i.menu.empty()&&i.menu.size()<=3){
            std::array<unsigned,5>r{i.edge,unsigned(i.menu.size()),0,0,0};std::copy(i.menu.begin(),i.menu.end(),r.begin()+2);
            if(std::find(stream_recipe::last_recipes.begin(),stream_recipe::last_recipes.end(),r)==stream_recipe::last_recipes.end())stream_recipe::last_recipes.push_back(r);
        }
        require(recipe[1]||std::all_of(s,s+size_t(d)*d*d,[](uint8_t v){return !v;}));
        std::copy(best.begin(),best.end(),out);*n=best.size();return 0;
    }catch(const std::exception&e){fprintf(stderr,"stream discovery: %s\n",e.what());return -1;}
}
extern "C" int ng_stream_recipe_encode(const uint8_t*s,unsigned d,unsigned level,const unsigned*recipe,
    uint8_t*out,size_t cap,size_t*n){
    try{
        require(s&&out&&n&&recipe&&level<16&&recipe[1]>=1&&recipe[1]<=3);
        unsigned policy=level/4;level%=4;
        std::vector<unsigned>menu(recipe+2,recipe+2+recipe[1]);for(auto id:menu)require(id<=24);
        auto unique=menu;std::sort(unique.begin(),unique.end());require(std::adjacent_find(unique.begin(),unique.end())==unique.end());
        stream_recipe::Scope scope(level);choice_grid::Scope grid(recipe[0]);
        if(menu.size()==1){auto frame=stream_recipe::uniform(s,d,level,menu[0]);require(frame.size()<=cap);std::copy(frame.begin(),frame.end(),out);*n=frame.size();return 0;}
        fast_split::stream_menu=&menu;int cfg[6];stream_recipe::config(level,cfg);
        fast_split::stream_onepass=policy>=2;
        return ng_screen_pair_encode_impl(s,d,12,out,cap,n,cfg,16,menu.size(),true,&menu,policy?0:3);
    }catch(const std::exception&e){fprintf(stderr,"stream recipe: %s\n",e.what());return -1;}
}

namespace stream_recipe {
struct Arm {
    unsigned level=0;std::array<unsigned,5> recipe{4,1,24,0,0};
    bool operator==(const Arm&b)const{return level==b.level&&recipe==b.recipe;}
};
// A search round shares identical fits only within the current input chunk.
// Retaining source-dependent fits across chunks is deliberately not enabled.
struct SearchMemo {
    bool enabled=small_bank::memo::enabled;
    size_t used=small_bank::memo::used,hits=small_bank::memo::hits,misses=small_bank::memo::misses;
    std::vector<small_bank::memo::Entry> entries;
    SearchMemo(){entries.swap(small_bank::memo::entries);small_bank::memo::enabled=true;
        small_bank::memo::used=small_bank::memo::hits=small_bank::memo::misses=0;}
    ~SearchMemo(){small_bank::memo::entries.swap(entries);small_bank::memo::enabled=enabled;
        small_bank::memo::used=used;small_bank::memo::hits=hits;small_bank::memo::misses=misses;}
};
struct State {
    unsigned profile,policy,d=0;uint64_t chunks=0,active_chunks=0,probes=0,promotions=0,rng=0x123456789abcdefULL;
    Arm incumbent,challenger;unsigned trials=0;double delta=0;
    bool initialized=false;
    small_bank::StreamFits fits;
    std::vector<Arm>pool;size_t pool_next=0;
    std::map<std::array<unsigned,12>,std::pair<double,double>> evidence;
    std::vector<Arm> round;
    std::vector<double> round_cost;
    unsigned round_samples=0;
    std::array<unsigned,2> advice{0,0};
    uint64_t next_search=0,round_start=0,search_rounds=0;
    State(unsigned p,unsigned q):profile(p),policy(q){if(q>=7){incumbent.level=p;incumbent.recipe=q>=9?std::array<unsigned,5>{4,1,q==11?1u:0u,0,0}:std::array<unsigned,5>{4,2,0,24,0};}}
    bool allowed(const Arm&a)const{
        if(policy==11)return a.level==1&&a.recipe[1]==1&&a.recipe[2]>0&&a.recipe[2]<24&&a.recipe[2]!=12;
        if(policy>=9)return a.level==0&&a.recipe[1]==1&&a.recipe[2]<24;
        if(a.level!=profile||a.recipe[1]<1||a.recipe[1]>3)return false;
        if(profile==0&&(a.recipe[1]>2||a.recipe[2]!=0))return false;
        return true;
    }
    void begin_round(){
        std::vector<std::vector<unsigned>>menus;
        auto add=[&](std::vector<unsigned> m){
            std::sort(m.begin(),m.end());m.erase(std::unique(m.begin(),m.end()),m.end());
            if(m.empty()||m.size()>3||(profile==0&&(m.size()>2||m[0]!=0)))return;
            if(std::find(menus.begin(),menus.end(),m)==menus.end())menus.push_back(std::move(m));
        };
        if(policy>=9){for(unsigned id=0;id<24;id++)if(policy!=11||(id&&id!=12))menus.push_back({id});}
        else {add({0});for(unsigned id=1;id<=24;id++)add({0,id});}
        std::vector<unsigned>current(incumbent.recipe.begin()+2,incumbent.recipe.begin()+2+incumbent.recipe[1]);if(policy<9)add(current);
        if(profile==1&&policy<9){
            for(unsigned id=0;id<=24;id++)add({id});
            for(size_t j=0;j<current.size();j++){
                auto m=current;m.erase(m.begin()+j);add(m);
                for(unsigned id=0;id<=24;id++){m=current;m[j]=id;add(m);}
            }
            if(current.size()<3)for(unsigned id=0;id<=24;id++){auto m=current;m.push_back(id);add(m);}
        }
        round.clear();
        for(auto&m:menus)for(unsigned edge:{2u,4u,8u,16u,32u}){
            Arm a;a.level=profile;a.recipe={edge,unsigned(m.size()),0,0,0};
            std::copy(m.begin(),m.end(),a.recipe.begin()+2);require(allowed(a));round.push_back(a);
        }
        round_cost.assign(round.size(),0);round_samples=0;round_start=active_chunks;
    }
    void structured_encode(const uint8_t*src,unsigned d,uint8_t*out,size_t cap,size_t*n){
        require(allowed(incumbent));
        if(round.empty()&&active_chunks>=next_search)begin_round();
        auto encode=[&](const Arm&a,uint8_t*dest,size_t*bytes){
            require(allowed(a));
            if(policy==11)require(ng_slow_recipe_encode(src,d,a.recipe[0],a.recipe[2],-1,-1,dest,cap,bytes)==0);
            else if(policy>=9)require(ng_fast_recipe_encode(src,d,a.recipe[0],a.recipe[2],dest,cap,bytes)==0);
            else require(ng_stream_recipe_encode(src,d,8+a.level,a.recipe.data(),dest,cap,bytes)==0);
        };
        if(round.empty()){encode(incumbent,out,n);return;}
        // Preserve Fast's fused fitting path: no generic fit memo in this mode.
        std::unique_ptr<SearchMemo> memo;if(policy<9)memo=std::make_unique<SearchMemo>();
        encode(incumbent,out,n);const size_t base=*n;Bytes candidate(cap);
        for(size_t i=0;i<round.size();i++){
            size_t bytes=base;
            if(!(round[i]==incumbent)){encode(round[i],candidate.data(),&bytes);++probes;
                if(bytes<*n){std::copy_n(candidate.begin(),bytes,out);*n=bytes;}}
            round_cost[i]+=std::log(double(bytes)/double(base));
        }
        if(++round_samples==((policy==7||policy==9||policy==11)?2u:4u)){
            auto best=std::min_element(round_cost.begin(),round_cost.end());
            if(*best<0){incumbent=round[size_t(best-round_cost.begin())];++promotions;}
            if(policy==11){
                advice[0]=incumbent.recipe[2];double runner=std::numeric_limits<double>::infinity();
                for(size_t i=0;i<round.size();i++)if(round[i].recipe[0]==incumbent.recipe[0]&&round[i].recipe[2]!=advice[0]&&round_cost[i]<runner){runner=round_cost[i];advice[1]=round[i].recipe[2];}
            }
            ++search_rounds;next_search=round_start<64?64:round_start<256?256:round_start+1024;
            round.clear();round_cost.clear();round_samples=0;
        }
    }
    uint64_t random(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
    Arm propose(){
        while(pool_next<pool.size()){auto a=pool[pool_next++];if(!(a==incumbent))return a;}
        Arm a=incumbent;unsigned kind=probes%5;
        if(kind==0){unsigned e=a.recipe[0];a.recipe[0]=e==2?4:e==32?16:random()%2?e*2:e/2;}
        if(kind==1){ // Fit cost belongs to the arm too, not a hidden fixed expense.
            a.level=a.level?0:profile;
            if(a.level==incumbent.level)a.recipe[0]=a.recipe[0]==4?8:4;
        }
        if(kind==2){
            if(a.recipe[1]>1){unsigned remove=random()%a.recipe[1];for(unsigned i=remove;i+1<a.recipe[1];i++)a.recipe[2+i]=a.recipe[3+i];a.recipe[1]--;a.recipe[2+a.recipe[1]]=0;}
            else a.recipe[2]=a.recipe[2]==24?0:24;
        }
        if(kind>=3){
            unsigned id=random()%25,n=a.recipe[1];
            while(std::find(a.recipe.begin()+2,a.recipe.begin()+2+n,id)!=a.recipe.begin()+2+n)id=(id+1)%25;
            if(n<3&&kind==3)a.recipe[2+a.recipe[1]++]=id;
            else a.recipe[2+random()%n]=id;
        }
        return a;
    }
};
}
extern "C" void* ng_stream_create(unsigned profile,unsigned policy){
    if(profile>3||policy>11||(policy>=7&&profile>1)||((policy==9||policy==10)&&profile!=0)||(policy==11&&profile!=1))return nullptr;
    try{return new stream_recipe::State(profile,policy);}catch(...){return nullptr;}
}
extern "C" void ng_stream_destroy(void*p){delete static_cast<stream_recipe::State*>(p);}
extern "C" void* ng_stream_clone(const void*p){try{return p?new stream_recipe::State(*static_cast<const stream_recipe::State*>(p)):nullptr;}catch(...){return nullptr;}}
extern "C" void ng_stream_stats(const void*p,uint64_t*out){
    if(!p||!out)return;auto&s=*static_cast<const stream_recipe::State*>(p);
    out[0]=s.chunks;out[1]=s.probes;out[2]=s.promotions;out[3]=s.incumbent.level;
    for(unsigned i=0;i<5;i++)out[4+i]=s.incumbent.recipe[i];
}
// Counts actual native fits and coefficient replays for profiling.
extern "C" void ng_stream_fit_stats(const void*p,uint64_t*out){
    if(!p||!out)return;auto&s=*static_cast<const stream_recipe::State*>(p);
    out[0]=s.fits.fitted;out[1]=s.fits.reused;out[2]=s.fits.models.size();
}
extern "C" void ng_stream_search_stats(const void*p,uint64_t*out){
    if(!p||!out)return;auto&s=*static_cast<const stream_recipe::State*>(p);
    out[0]=s.search_rounds;out[1]=s.round_samples;out[2]=s.round.size();out[3]=s.next_search;out[4]=s.active_chunks;
}
extern "C" void ng_stream_advice(const void*p,unsigned*out){if(p&&out){auto&s=*static_cast<const stream_recipe::State*>(p);std::copy(s.advice.begin(),s.advice.end(),out);}}
extern "C" int ng_stream_encode(void*p,const uint8_t*src,unsigned d,uint8_t*out,size_t cap,size_t*n){
    if(!p||!src||!out||!n||!ng_geometry_valid(d,d,d))return -1;
    try{
        auto&state=*static_cast<stream_recipe::State*>(p);
        require(!state.d||state.d==d);
        // Update state only after a successful encode, including error paths.
        auto s=state;s.d=d;
        struct FitScope{small_bank::StreamFits*old=small_bank::stream_fits;~FitScope(){small_bank::stream_fits=old;}} fit_scope;
        if(s.policy>=3&&s.policy<=6){s.fits.voxels=size_t(d)*d*d;s.fits.support_only=s.policy>=5;
            s.fits.refresh=s.active_chunks<8||s.active_chunks%((s.policy==3||s.policy==5)?32:128)==0;
            small_bank::stream_fits=&s.fits;}
        if(std::all_of(src,src+size_t(d)*d*d,[](uint8_t v){return !v;})){
            GeometryScope geom(d,d,d);auto f=choice_frame::empty_frame(d);require(f.size()<=cap);
            std::copy(f.begin(),f.end(),out);*n=f.size();++s.chunks;state=std::move(s);return 0;
        }
        if(s.policy>=7){s.structured_encode(src,d,out,cap,n);s.initialized=true;
            ++s.chunks;++s.active_chunks;state=std::move(s);return 0;}
        if(!s.initialized){
            require(ng_stream_discover(src,d,s.profile,1,s.incumbent.recipe.data(),out,cap,n)==0);
            s.incumbent.level=s.profile;s.initialized=true;
            for(auto&r:stream_recipe::last_recipes){stream_recipe::Arm a;a.level=s.profile;a.recipe=r;s.pool.push_back(a);}
            ++s.chunks;++s.active_chunks;state=std::move(s);return 0;
        }
        const unsigned flags=s.policy==0?0:8;
        require(ng_stream_recipe_encode(src,d,flags+s.incumbent.level,s.incumbent.recipe.data(),out,cap,n)==0);
        // Heavy early exploration, bounded maintenance thereafter. Policy2
        // explores more often. No tail freeze: normal adaptation continues.
        unsigned interval=s.active_chunks<32?1:s.active_chunks<128?4:s.active_chunks<512?16:64;
        if(s.policy==2)interval=std::max(1u,interval/4);
        if(s.active_chunks%interval==0){
            if(!s.trials)s.challenger=s.propose();
            Bytes f(cap);size_t bytes=0;
            require(ng_stream_recipe_encode(src,d,flags+s.challenger.level,s.challenger.recipe.data(),f.data(),cap,&bytes)==0);
            // Paired relative-byte score avoids comparing unrelated chunks'
            // compressibility. Average over repeated probes, not one outlier.
            s.delta+=std::log(double(bytes)/double(*n));++s.trials;++s.probes;
            if(bytes<*n){std::copy_n(f.begin(),bytes,out);*n=bytes;}
            if(s.trials==3){
                std::array<unsigned,12>key{};key[0]=s.incumbent.level;key[6]=s.challenger.level;
                std::copy(s.incumbent.recipe.begin(),s.incumbent.recipe.end(),key.begin()+1);
                std::copy(s.challenger.recipe.begin(),s.challenger.recipe.end(),key.begin()+7);
                auto&h=s.evidence[key];h.first=.75*h.first+s.trials;h.second=.75*h.second+s.delta;
                if(h.second/h.first<-.0005){s.incumbent=s.challenger;++s.promotions;}s.trials=0;s.delta=0;
            }
        }
        ++s.chunks;++s.active_chunks;state=std::move(s);return 0;
    }catch(const std::exception&e){fprintf(stderr,"stream encode: %s\n",e.what());return -1;}
}
