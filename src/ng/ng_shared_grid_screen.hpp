// Shared full-coordinate partition search. No model fitting and no trial
// residual payload encodes here: callers hand over their already fitted menu.
namespace shared_grid {
thread_local bool enabled=false;
thread_local unsigned policy=4,selected_edge=4;
thread_local double estimates[5]={};
struct Scope {bool old=enabled;unsigned p=policy;explicit Scope(unsigned q){enabled=true;policy=q;}~Scope(){enabled=old;policy=p;}};
struct Candidate {unsigned predictor,old_id;const small_bank::Fit*fit;const uint16_t*base;};
struct Plan {unsigned edge;Bytes choices,expanded;double score=0;};
Bytes run(const uint8_t*src,unsigned d,const std::vector<Candidate>&input,
          std::vector<Bytes>&candidate,const zero_blocks::Scan&reference_zeros,bool fast){
    require(choice_grid::edge()==4&&input.size()>=1&&input.size()<=3);
    const unsigned k=input.size(),zero=k,K=16,n=d/2;const size_t blocks=size_t(n)*n*n;
    choice_grid::Scope fine(2);adaptive_grid::Scope no_tree(0);
    std::fill_n(estimates,5,0.);std::fill_n(adaptive_grid::leaves,6,0);
    zero_blocks::Scan occupancy(src,d,true);
    std::vector<Bytes>models;std::vector<const small_bank::Fit*>fits;std::vector<const uint16_t*>base_by_model;
    std::vector<choice_frame::Choice>menu;std::vector<unsigned>canonical;
    if(candidate[0].empty()){candidate[0].resize(N);bank_fixed_candidates(src,candidate,d,1);}
    for(const auto&c:input){
        unsigned m=255;
        if(c.fit){m=std::find(fits.begin(),fits.end(),c.fit)-fits.begin();
            if(m==fits.size()){fits.push_back(c.fit);models.push_back(c.fit->params);base_by_model.push_back(c.base);}
        }
        menu.push_back({c.predictor,m,unsigned(menu.size()),c.fit?c.fit->weights:small_bank::Weight{}});
        canonical.push_back(m==255?c.predictor:12+12*m+c.predictor);
        candidate[c.old_id].resize(N);
    }
    menu.push_back({choice_frame::zero_predictor,255,zero,{}});canonical.push_back(0);
    Bytes symbols(N),classes,seed(blocks);std::vector<uint16_t>activity(N);
    for(unsigned z=0;z<n;z++)for(unsigned y=0;y<n;y++)for(unsigned x=0;x<n;x++){
        size_t b=adaptive_grid::index(x,y,z,n),b4=adaptive_grid::index(x/2,y/2,z/2,d/4);
        seed[b]=reference_zeros.any()&&reference_zeros.empty[b4]?zero:0;
    }
    auto evaluate=[&](const Bytes&selection,bool selected_only){
        if(models.empty()){
            for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
                size_t i=(size_t(z)*d+y)*d+x;unsigned c=selection[adaptive_grid::index(x/2,y/2,z/2,n)];
                symbols[i]=c==zero?0:candidate[input[c].old_id][i];
            }require(ng_activity_symbol_values(symbols.data(),activity.data())==0);return;
        }
        const uint8_t*pp[48]={};size_t sizes[48]={};const uint16_t*bp[48]={};uint8_t*wp[60]={};
        for(unsigned m=0;m<models.size();m++)for(unsigned j=0;j<12;j++){
            pp[12*m+j]=models[m].data();sizes[12*m+j]=models[m].size();bp[12*m+j]=base_by_model[m];
        }
        wp[0]=candidate[0].data();for(unsigned c=0;c<k;c++)wp[canonical[c]]=candidate[input[c].old_id].data();
        Bytes mapped(selection.size());for(size_t b=0;b<selection.size();b++)mapped[b]=canonical[selection[b]];
        require((selected_only?ng_bank_selected_symbols:ng_bank_symbols)(12*models.size(),pp,sizes,bp,mapped.data(),src,wp,symbols.data(),activity.data())==0);
    };
    evaluate(seed,false);
    auto thr=activity_quantiles(activity,K);classes=activity_classify(activity,thr);
    std::vector<Histogram>hist(k*K);
    for(unsigned c=0;c<k;c++)for(size_t i=0;i<N;i++)hist[c*K+classes[i]][candidate[input[c].old_id][i]]++;
    std::vector<std::array<double,256>>bits(k*K);
    for(unsigned c=0;c<k*K;c++){
        uint64_t counts[256];size_t total=0;for(unsigned s=0;s<256;s++){counts[s]=hist[c][s];total+=counts[s];}
        uint16_t freq[256];delta_rans_size_lower_bound(counts,total,freq);
        for(unsigned s=0;s<256;s++)bits[c][s]=12.-std::log2(double(std::max(1u,unsigned(freq[s]))));
    }
    using Cost=std::array<double,4>;std::vector<Cost>cost(blocks);
    for(auto&v:cost)for(unsigned c=k;c<4;c++)v[c]=1e50;
    for(size_t b=0;b<blocks;b++)if(occupancy.any()&&occupancy.empty[b])cost[b][zero]=0;
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
        size_t i=(size_t(z)*d+y)*d+x;auto&v=cost[adaptive_grid::index(x/2,y/2,z/2,n)];
        for(unsigned c=0;c<k;c++)v[c]+=bits[c*K+classes[i]][candidate[input[c].old_id][i]];
    }
    auto score=[&](Plan&p,bool tree){
        // Score selected residual histograms, not the proxy-frame ratio. Keep
        // table/model costs separate; selector bytes are small enough to code
        // exactly without running the voxel payload codec for each proposal.
        std::vector<Histogram>h(k*K);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;unsigned c=p.expanded[adaptive_grid::index(x/2,y/2,z/2,n)];
            if(c<k)h[c*K+classes[i]][candidate[input[c].old_id][i]]++;
        }
        p.score=0;for(auto&t:h)p.score+=histogram_cost(t.data(),12,true)/8.;
        bool uniform=std::all_of(p.choices.begin(),p.choices.end(),[&](auto c){return c==p.choices[0];});
        if(!uniform){Bytes tokens=p.choices;if(tree){adaptive_grid::Scope scope(policy,2);tokens=adaptive_grid::serialize(p.expanded,d);}
            p.score+=packed_bank::compress(pack_stream(tokens,true)).size();}
        // Identical fitted models/menu for every size: fixed overhead is not
        // multiplied by the64x proxy-to-chunk voxel ratio.
        for(auto&m:models)p.score+=m.size();p.score+=12+menu.size()*12;
    };
    Plan best;best.score=std::numeric_limits<double>::infinity();
    if(policy==8||policy==16||policy==32){
        best.edge=2;best.expanded.resize(blocks);
        adaptive_grid::Scope tree(policy,2);adaptive_grid::select(d,cost,best.expanded);best.choices=best.expanded;score(best,true);
    }else{
        auto sums=std::move(cost);unsigned nn=n;
        for(unsigned e=2,level=0;e<=32;e*=2,level++){
            if(policy==0||e==4){
                Plan p;p.edge=e;p.choices.resize(sums.size());p.expanded.resize(blocks);
                for(unsigned z=0;z<nn;z++)for(unsigned y=0;y<nn;y++)for(unsigned x=0;x<nn;x++){
                    size_t b=adaptive_grid::index(x,y,z,nn);auto&v=sums[b];unsigned c=std::min_element(v.begin(),v.begin()+k+1)-v.begin();
                    p.choices[b]=c;adaptive_grid::fill(p.expanded,n,x*(e/2),y*(e/2),z*(e/2),e/2,c);
                }
                score(p,false);estimates[level]=p.score;
                // Equal costs provide no reason to leave the optimized4³
                // decoder path (uniform chunks often tie exactly).
                if(p.score<best.score||(p.score==best.score&&e==4))best=std::move(p);
            }
            if(e==32||policy==4&&e==4)break;
            std::vector<Cost>next(size_t(nn/2)*(nn/2)*(nn/2));
            for(unsigned z=0;z<nn/2;z++)for(unsigned y=0;y<nn/2;y++)for(unsigned x=0;x<nn/2;x++){
                auto&v=next[adaptive_grid::index(x,y,z,nn/2)];
                for(unsigned a=0;a<8;a++){auto&t=sums[adaptive_grid::index(2*x+(a&1),2*y+((a>>1)&1),2*z+(a>>2),nn)];for(unsigned c=0;c<4;c++)v[c]+=t[c];}
            }sums=std::move(next);nn/=2;
        }
    }
    selected_edge=best.edge;evaluate(best.expanded,true);
    // One final residual/threshold computation and one payload encode.
    Bytes active_symbols;std::vector<uint16_t>active_activity;std::vector<uint32_t>active_block;
    active_symbols.reserve(N);active_activity.reserve(N);active_block.reserve(N);
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
        size_t i=(size_t(z)*d+y)*d+x,b=adaptive_grid::index(x/2,y/2,z/2,n);
        if(best.expanded[b]==zero)continue;
        active_symbols.push_back(symbols[i]);active_activity.push_back(activity[i]);active_block.push_back(b);
    }
    thr=activity_quantiles(active_activity,K);
    thr=activity_refine(active_symbols,active_activity,thr,active_block,best.expanded,k,true,64,fast);
    classes=activity_classify(activity,thr);
    choice_grid::Scope grid(best.edge);adaptive_grid::Scope tree(policy>=8?policy:0,2);
    return choice_frame::write(d,models,menu,best.choices,symbols,thr,classes);
}
}
