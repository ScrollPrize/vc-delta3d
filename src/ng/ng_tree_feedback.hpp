// Tree-aware choice feedback on the existing bounded three-choice engine.
// Fitted predictors stay fixed; residual histories and entropy trees do not.
namespace hybrid_geometry {
void tree_choice_feedback(const uint8_t*src,unsigned d,const std::vector<Bytes>&candidate,
    const unsigned*ids,unsigned third,const std::vector<uint16_t>&,
    const Bytes&classes,const zero_blocks::Scan&zeros,Bytes&choices){
    GeometryScope scope(d,d,d);
    const unsigned predictors[3]={0,0,third};
    std::vector<entropy_tree::Feature>features(N);
    std::vector<uint32_t>block(N),active;
    std::vector<double>nlogn(N+1);
    for(size_t i=1;i<=N;i++)nlogn[i]=double(i)*std::log2(double(i));
    std::array<std::vector<uint32_t>,3>assigned;
    for(size_t i=0;i<N;i++){
        block[i]=geometry_trials::block(i,d);
        if(zeros.any()&&!zeros.mask[i])continue;
        active.push_back(i);assigned[choices[block[i]]].push_back(i);
        for(unsigned f=0;f<8;f++)features[i][f]=entropy_tree::feature(src,i,d,classes[i],0,f);
    }
    std::vector<std::array<double,3>>cost(choices.size());
    uint32_t selector_hist[256]={};for(auto c:choices)selector_hist[c]++;
    double selector_cost[256];require(ng_context_rans_costs(selector_hist,1,selector_cost)==0);
    for(unsigned k=0;k<3;k++){
        for(auto i:active)features[i][8]=predictors[k];
        // An unused candidate still gets evidence, rather than an unlearnable
        // empty distribution. Otherwise fit the actually selected stream.
        const auto&fit_ids=assigned[k].empty()?active:assigned[k];
        auto tree=entropy_tree::fit(fit_ids,features,candidate[ids[k]],nlogn)[2];
        Bytes leaves(N);
        std::vector<uint32_t>hist(64*256);
        for(auto i:active)leaves[i]=tree.classify(src,i,d,classes[i],predictors[k]);
        for(auto i:fit_ids)hist[unsigned(leaves[i])*256+candidate[ids[k]][i]]++;
        std::vector<double>bits(hist.size());require(ng_context_rans_costs(hist.data(),64,bits.data())==0);
        for(auto i:active)cost[block[i]][k]+=bits[unsigned(leaves[i])*256+candidate[ids[k]][i]];
        for(auto&c:cost)c[k]+=selector_cost[k];
    }
    for(size_t b=0;b<choices.size();b++){
        if(zeros.any()&&zeros.empty[b]){choices[b]=1;continue;}
        unsigned best=choices[b];
        for(unsigned k=0;k<3;k++)if(cost[b][k]<cost[b][best])best=k;
        choices[b]=best;
    }
}
Trial*create_feedback(const uint8_t*src,unsigned d,unsigned slow,unsigned all,unsigned passes){
    require(src&&slow<=1&&all<=1&&passes<=3&&(slow||passes==0));
    GeometryScope scope(d,d,d);std::vector<Snapshot>v;Capture capture(v,d);
    struct Scope {
        three_candidate::Feedback old=three_candidate::feedback;
        unsigned rounds=three_candidate::feedback_passes;
        Scope(unsigned n){three_candidate::feedback=n?tree_choice_feedback:nullptr;three_candidate::feedback_passes=n;}
        ~Scope(){three_candidate::feedback=old;three_candidate::feedback_passes=rounds;}
    } feedback(passes);
    int cfg[6]={3,3,all?3:1,16,all?160:40,128};
    Bytes frame(3*N+131072);size_t n=0;
    auto enc=slow?ng_codec_encode_slow_config:ng_codec_encode_fast_config;
    require(enc(src,d,cfg,16,frame.data(),frame.size(),&n)==0);frame.resize(n);
    for(auto it=v.rbegin();it!=v.rend();++it)if(it->frame==frame){
        if(it->symbols.empty()){it->symbols.resize(N);it->classes.resize(N);}
        return new Trial{std::move(*it),Bytes(src,src+N)};
    }
    throw std::runtime_error("feedback frame not captured");
}
}
extern "C" void*ng_tree_feedback_create(const uint8_t*s,unsigned d,unsigned slow,unsigned all,unsigned passes){
    try{return hybrid_geometry::create_feedback(s,d,slow,all,passes);}
    catch(const std::exception&e){fprintf(stderr,"tree feedback: %s\n",e.what());return nullptr;}
}
