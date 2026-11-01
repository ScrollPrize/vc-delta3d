/* Freely selected three-candidate experiments. Native shared fitter, causal
 * evaluator, entropy streams and reconstruction; no Python codec operations. */
namespace three_free {
using three_candidate::Stats;
thread_local std::vector<std::array<unsigned,3>> menus;
thread_local std::vector<double> objectives;
struct Pool {
    const uint8_t*src;unsigned d;int cfg[6]={3,3,1,16,40,128};
    std::vector<Bytes> fixed;std::array<small_bank::Fit,12> fits;
    std::array<std::vector<double>,24> cost;std::vector<uint32_t>block;
    Pool(const uint8_t*x,unsigned dim,bool all=true):src(x),d(dim),fixed(12,Bytes(N)),block(N){
        cfg[5]=d<=32?4:128;bank_fixed_candidates(src,fixed,d);
        unsigned nb=d/4;for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++)
            block[(size_t(z)*d+y)*d+x]=(size_t(z/4)*nb+y/4)*nb+x/4;
        for(unsigned m=0;m<12;m++)cost[m]=three_candidate::init_block_costs(fixed[m],d);
        train(12,nullptr);
        if(all)for(unsigned m=1;m<12;m++)train(12+m,nullptr);
    }
    const Bytes&symbols(unsigned id)const{return id<12?fixed[id]:fits[id-12].initial_symbols;}
    double overhead(unsigned id)const{return id<12?0:double(fits[id-12].params.size()+8)*8;}
    void train(unsigned id,const Bytes*mask){
        unsigned m=id-12;int c[6];std::copy_n(cfg,6,c);
        if(mask){size_t n=std::count(mask->begin(),mask->end(),uint8_t(1));if(n<4096)return;c[5]=std::max(1u,unsigned(n/16384));}
        if(!m)fits[m]=small_bank::plain_fit(src,c,mask?mask->data():nullptr);
        else {Bytes feature(N);for(size_t i=0;i<N;i++)feature[i]=uint8_t(src[i]-unpyramid(fixed[m][i]));
            fits[m]=small_bank::fit(src,d,c,feature,mask?mask->data():nullptr,fixed,1u<<m,true);}
        cost[id]=three_candidate::init_block_costs(symbols(id),d);
    }
    double score(const std::array<unsigned,3>&ids,unsigned count=3)const{
        double v=0;for(unsigned k=0;k<count;k++)v+=overhead(ids[k]);
        for(size_t b=0;b<N/64;b++){double c=cost[ids[0]][b];for(unsigned k=1;k<count;k++)c=std::min(c,cost[ids[k]][b]);v+=c;}return v;
    }
    std::array<unsigned,3> choose(unsigned rounds,const std::array<unsigned,3>*seed=nullptr){
        std::array<unsigned,3>ids={0,0,0};
        if(seed)ids=*seed;
        else for(unsigned k=0;k<3;k++){double best=std::numeric_limits<double>::infinity();unsigned chosen=0;
            for(unsigned m=0;m<24;m++){bool duplicate=false;for(unsigned j=0;j<k;j++)duplicate|=ids[j]==m;if(duplicate)continue;
                ids[k]=m;double v=score(ids,k+1);if(v<best){best=v;chosen=m;}}
            ids[k]=chosen;
        }
        menus.push_back(ids);objectives.push_back(score(ids));
        for(unsigned round=1;round<rounds;round++){
            Bytes membership(N/64);for(size_t b=0;b<membership.size();b++){
                unsigned k=0;for(unsigned j=1;j<3;j++)if(cost[ids[j]][b]<cost[ids[k]][b])k=j;membership[b]=k;}
            for(unsigned k=0;k<3;k++){
                Bytes mask(N);size_t n=0;for(size_t i=0;i<N;i++){mask[i]=membership[block[i]]==k;n+=mask[i];}
                if(ids[k]>=12)train(ids[k],&mask);
                if(n){ // Re-estimate this slot's entropy on the blocks it serves.
                    const auto&sy=symbols(ids[k]);std::vector<uint16_t>a(N);require(ng_activity_symbol_values(sy.data(),a.data())==0);
                    auto s=three_candidate::distribution(sy,a);auto cl=activity_classify(a,s.thr);s.hist.assign(16,{});
                    for(size_t i=0;i<N;i++)if(mask[i])s.hist[cl[i]][sy[i]]++;s.finish();
                    auto&c=cost[ids[k]];std::fill(c.begin(),c.end(),0.);for(size_t i=0;i<N;i++)c[block[i]]+=s.cost[cl[i]][sy[i]];
                }
            }
            for(unsigned k=0;k<3;k++){
                unsigned chosen=ids[k];double best=score(ids);
                for(unsigned m=0;m<24;m++){bool duplicate=false;for(unsigned j=0;j<3;j++)if(j!=k)duplicate|=ids[j]==m;if(duplicate)continue;
                    auto trial=ids;trial[k]=m;double v=score(trial);if(v<best){best=v;chosen=m;}}
                ids[k]=chosen;
            }
            menus.push_back(ids);objectives.push_back(score(ids));
        }return ids;
    }
};
// Shared three-slot evaluator supports distinct inputs or repeated plain-NG
// identities with independently fitted coefficients. Legacy defaults unchanged.
Bytes encode_slots(const uint8_t*src,unsigned d,const std::vector<uint32_t>&block,const std::vector<Bytes>&fixed,
                   const std::array<unsigned,3>&menu,const std::array<const small_bank::Fit*,3>&fits,
                   Bytes*assignment=nullptr,bool selector_cost=false,bool guard=false){
    std::array<unsigned,3>ids;const uint8_t*params[36];size_t sizes[36];const uint16_t*bases[36];
    std::vector<Bytes>candidate(48);uint8_t*dest[48]={};
    for(unsigned k=0;k<3;k++){
        unsigned m=menu[k]%12;bool learned=menu[k]>=12;
        ids[k]=learned?12+12*k+m:m;
        const auto&fit=*fits[k];
        for(unsigned j=0;j<12;j++){params[12*k+j]=fit.params.data();sizes[12*k+j]=fit.params.size();bases[12*k+j]=fit.base[learned?m:0].data();}
        if(learned)candidate[ids[k]].resize(N);else candidate[ids[k]]=fixed[m];dest[ids[k]]=candidate[ids[k]].data();
    }
    Bytes choices(N/64,0),canonical(N/64),symbols(N);std::vector<uint16_t>activity(N);
    if(assignment){require(assignment->size()==choices.size());choices=*assignment;for(auto c:choices)require(c<3);}
    std::array<Stats,3>stats;
    auto evaluate=[&](bool only){for(size_t b=0;b<choices.size();b++)canonical[b]=ids[choices[b]];
        require((only?ng_bank_selected_symbols:ng_bank_symbols)(36,params,sizes,bases,canonical.data(),src,dest,symbols.data(),activity.data())==0);};
    evaluate(false);for(unsigned k=0;k<3;k++)stats[k]=three_candidate::distribution(candidate[ids[k]],activity,true);
    std::vector<uint16_t>thr;
    auto pack=[&](){Bytes header;put32(header,d);bool used[3]={};for(auto c:choices)used[c]=true;
        for(unsigned k=0;k<3;k++){auto id=menu[k];if(guard&&!used[k])id=0;put32(header,id);
            if(id>=12){auto&fit=*fits[k];put32(header,fit.params.size());packed_bank::append(header,fit.params);for(auto w:fit.weights)put32(header,uint32_t(w));}}
        return three_candidate::finish_frame(std::move(header),symbols,choices,block,thr,activity_classify(activity,thr),"N3A1");};
    Bytes best,best_choices;
    if(guard){thr=activity_quantiles(activity,16);thr=activity_refine(symbols,activity,thr,block,choices,3,true);best=pack();best_choices=choices;}
    for(unsigned pass=0;pass<3;pass++){
        if(pass)evaluate(false);
        std::vector<std::array<double,3>>cost(N/64);
        for(unsigned k=0;k<3;k++){if(selector_cost)stats[k].normalized_costs();auto cl=activity_classify(activity,stats[k].thr);for(size_t i=0;i<N;i++)cost[block[i]][k]+=stats[k].cost[cl[i]][candidate[ids[k]][i]];}
        if(selector_cost){uint64_t hist[256]={};for(auto c:choices)hist[c]++;uint16_t freq[256];delta_rans_size_lower_bound(hist,choices.size(),freq);
            for(auto&v:cost)for(unsigned k=0;k<3;k++)v[k]+=12.-std::log2(double(std::max(1u,unsigned(freq[k]))));}
        for(size_t b=0;b<choices.size();b++)choices[b]=std::min_element(cost[b].begin(),cost[b].end())-cost[b].begin();
        evaluate(true);thr=activity_quantiles(activity,16);thr=activity_refine(symbols,activity,thr,block,choices,3,true);
        auto cl=activity_classify(activity,thr);for(auto&s:stats){s.hist.assign(16,{});s.thr=thr;}
        for(size_t i=0;i<N;i++)stats[choices[block[i]]].hist[cl[i]][symbols[i]]++;for(auto&s:stats)s.finish();
        if(guard){auto frame=pack();if(frame.size()<best.size()){best=std::move(frame);best_choices=choices;}}
    }
    if(assignment)*assignment=guard?best_choices:choices;
    return guard?best:pack();
}
Bytes encode_menu(Pool&p,const std::array<unsigned,3>&menu){
    std::array<const small_bank::Fit*,3>fits;for(unsigned k=0;k<3;k++)fits[k]=menu[k]>=12?&p.fits[menu[k]%12]:&p.fits[0];
    return encode_slots(p.src,p.d,p.block,p.fixed,menu,fits);
}
}
extern "C" unsigned ng_three_free_trace(uint32_t*out,double*cost,unsigned cap){
    unsigned n=three_free::menus.size();for(unsigned i=0;i<std::min(n,cap);i++){for(unsigned k=0;k<3;k++)out[3*i+k]=three_free::menus[i][k];cost[i]=three_free::objectives[i];}return n;
}
extern "C" int ng_three_free_encode(const uint8_t*src,unsigned d,unsigned full,unsigned rounds,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&out&&written&&full<=2&&rounds>=1&&rounds<=16&&ng_geometry_valid(d,d,d));GeometryScope scope(d,d,d);
        three_free::menus.clear();three_free::objectives.clear();Bytes frame;
        std::fill_n(three_candidate::diagnostics,12,0);
        if(full==1){three_free::Pool pool(src,d);auto menu=pool.choose(rounds);frame=three_free::encode_menu(pool,menu);}
        else{
            auto proxy=fast_split::screen_proxy(src,d);unsigned sd=std::min(d,32u);std::array<unsigned,3>menu;
            {GeometryScope screen(sd,sd,sd);three_free::Pool pool(proxy.data(),sd);menu=pool.choose(full==2?1:rounds);}
            three_free::Pool pool(src,d,full==2);
            if(full==2){three_free::menus.clear();three_free::objectives.clear();menu=pool.choose(rounds,&menu);}
            else for(auto id:menu)if(id>12)pool.train(id,nullptr);
            frame=three_free::encode_menu(pool,menu);
        }
        require(frame.size()<=cap);std::copy(frame.begin(),frame.end(),out);*written=frame.size();return 0;
    }catch(...){return -1;}
}
extern "C" int ng_three_free_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    try{
        require(src&&out&&n>=12&&!memcmp(src,"N3A1",4));Bytes f(src,src+n);size_t p=4;
        size_t hn=get32(f,p),zn=get32(f,p);require(zn<=n-p);auto header=packed_bank::decompress(src+p,zn,hn);p+=zn;
        packed_bank::Reader h{header};unsigned d=h.u(4);require(ng_geometry_valid(d,d,d)&&cap>=size_t(d)*d*d);GeometryScope scope(d,d,d);
        using State=std::unique_ptr<void,decltype(&ng_causal_destroy)>;std::vector<State>states;void*models[36]={};int32_t weights[72]={};unsigned ids[3];
        for(unsigned k=0;k<3;k++){unsigned id=h.u(4);require(id<24);ids[k]=id<12?id:12+12*k+id%12;
            if(id>=12){auto params=h.take(h.u(4));states.emplace_back(states.empty()?ng_causal_create(params.data(),params.size()):ng_causal_create_model(params.data(),params.size()),ng_causal_destroy);require(bool(states.back()));
                int32_t w[2];for(auto&v:w)v=int32_t(packed_bank::signed_value(h.u(4),32));
                for(unsigned j=0;j<12;j++){models[12*k+j]=states.back().get();for(unsigned q=0;q<2;q++)weights[24*k+2*j+q]=j?w[q]:0;}}
        }
        auto choices=three_candidate::read_residuals(h,f,p,d,out);for(auto&c:choices)c=ids[c];
        require(ng_bank_reconstruct_many(models,weights,choices.data(),out,36)==0);return 0;
    }catch(...){return -1;}
}
