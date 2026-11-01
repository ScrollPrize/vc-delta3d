/* Experimental directional banks. 0=direct,1..7=Delta,8..11=clamped
 * XY/XZ/YZ/XYZ;12=plain NG,13..23=one independent NG fit per base. */
namespace {
struct BankWinner {Bytes prefix,symbols,classes;std::vector<uint16_t> thresholds;};
bool bank_single_estimated_trial(const ng_bank_search_options* options){
    return options&&options->seed!=4&&options->rounds==1&&options->passes==1&&options->thresholds!=0;
}
thread_local double bank_encode_times[8];
thread_local bool bank_encode_profile_enabled=false;
thread_local double bank_fit_times[5];
thread_local double bank_regression_times[5];
thread_local std::chrono::steady_clock::time_point bank_regression_tick;
extern "C" void ng_bank_regression_profile_get(double*out){if(out)std::copy_n(bank_regression_times,5,out);}
thread_local double bank_entropy_times[3];
thread_local std::chrono::steady_clock::time_point bank_entropy_tick;
void bank_entropy_profile_tick(int k){
    auto now=std::chrono::steady_clock::now();
    if(k>=13&&k<=15)bank_entropy_times[k-13]+=std::chrono::duration<double>(now-bank_entropy_tick).count();
    bank_entropy_tick=now;
}
extern "C" void ng_bank_entropy_profile_get(double*out){if(out)std::copy_n(bank_entropy_times,3,out);}
thread_local std::chrono::steady_clock::time_point bank_fit_tick;
void bank_fit_profile_tick(int k){
    auto now=std::chrono::steady_clock::now();
    if(k>=20&&k<=25){
        if(k>20)bank_regression_times[k-21]+=std::chrono::duration<double>(now-bank_regression_tick).count();
        bank_regression_tick=now;return;
    }
    if(k>=8&&k<=12)bank_fit_times[k-8]+=std::chrono::duration<double>(now-bank_fit_tick).count();
    bank_fit_tick=now;
}
extern "C" void ng_bank_fit_profile_get(double*out){if(out)std::copy(bank_fit_times,bank_fit_times+5,out);}
thread_local unsigned bank_search_policy=1;
thread_local unsigned bank_fit_acceleration=2;
thread_local unsigned bank_sparse_stride=0,bank_sparse_models=12;
thread_local unsigned bank_regression_stride=128;
thread_local unsigned bank_diagnostic_mode=0;
thread_local bool bank_diagnostic_enabled=false;
thread_local double bank_diagnostic[72]={}; // proposals,wins,score-minus-NG
extern "C" void ng_bank_diagnostic_reset(unsigned mode){bank_diagnostic_mode=mode;bank_diagnostic_enabled=true;std::fill(bank_diagnostic,bank_diagnostic+72,0.);}
extern "C" void ng_bank_diagnostic_get(double*out){if(out)std::copy(bank_diagnostic,bank_diagnostic+72,out);}
extern "C" void ng_bank_diagnostic_end(){bank_diagnostic_mode=0;bank_diagnostic_enabled=false;}
extern "C" void ng_bank_fit_acceleration(unsigned mode){bank_fit_acceleration=std::min(mode,2u);}
template<unsigned M,bool Independent=false>
double bank_shortlist_block(const uint8_t*src,unsigned d,unsigned x0,unsigned y0,unsigned z0,
        const Bytes&ng,const Bytes&classes,const std::vector<std::array<double,256>>&cost,
        std::array<uint8_t,64>&candidate,double score,const uint16_t*core,const int32_t*weights,const uint8_t*const* fitted,void*sparse){
    if constexpr(Independent&&M>12)if(sparse)require(ng_sparse_block(sparse,M-12,x0,y0,z0,candidate.data())==0);
    unsigned q=0;
    for(unsigned z=z0;z<z0+4;z++)for(unsigned y=y0;y<y0+4;y++)for(unsigned x=x0;x<x0+4;x++,q++){
        const size_t i=(size_t(z)*d+y)*d+x;
        uint8_t s;
        if constexpr(Independent&&M>=12)s=sparse?(M==12?ng[i]:candidate[q]):fitted[M][i];
        else if constexpr(M==12)s=ng[i];
        else if constexpr(M>12)s=ng_prediction_symbol(src[i],uint16_t(core[i]+unsigned(uint16_t(weights[z==0?1:0]))*ng_directional_base(src,i,M-12,x,y,z,d)));
        else s=pyramid(uint8_t(src[i]-ng_directional_base(src,i,M,x,y,z,d)));
        candidate[q]=s;score+=cost[classes[i]][s];
    }
    return score;
}
// One bounded proposal sweep. Costs are frozen from initial NG; selectors are
// causal in block raster order. Fixed residuals are generated only where scored.
struct ShortlistHistory {
    std::array<double,36>gain{},trials{};
    uint32_t rng=0x9e3779b9u;
    std::vector<std::array<double,256>>cost;
};
template<bool Large=false,bool Independent=false,unsigned Banks=1>
void bank_shortlist_select(const uint8_t*src,unsigned d,unsigned proposal_mode,const Bytes&ng,
        const Bytes&classes,Bytes&sel,Bytes&fixed,ShortlistHistory*history=nullptr,
        bool retain_incumbent=false,const Bytes*observed=nullptr,const uint16_t*core=nullptr,const int32_t*weights=nullptr,const uint8_t*const*fitted=nullptr,void*sparse=nullptr){
    ShortlistHistory local;auto&state=history?*history:local;
    if(state.cost.empty()||observed)state.cost=activity_symbol_costs(observed?*observed:ng,classes,{},sel,1);
    const auto&cost=state.cost;
    const unsigned nb=d/4;constexpr unsigned capacity=Large?12+12*Banks:13,draw_limit=Large?12+12*Banks:12;
    const unsigned menu=Independent&&Banks==1&&bank_sparse_models==2?14:capacity,pool=menu-1;
    auto allowed=[&](unsigned m){return !Independent||Banks!=1||bank_sparse_models!=2||m<=12||m==19;};
    std::array<size_t,capacity>counts{};
    auto&gain=state.gain;auto&trials=state.trials;auto&rng=state.rng;
    for(unsigned bz=0;bz<nb;bz++)for(unsigned by=0;by<nb;by++)for(unsigned bx=0;bx<nb;bx++){
        const size_t b=(size_t(bz)*nb+by)*nb+bx;
        std::array<unsigned,4>proposal{12,1,7,11};unsigned count=4;
        if(proposal_mode){
            count=1;
            auto add=[&](unsigned m){if(count<3&&std::find(proposal.begin(),proposal.begin()+count,m)==proposal.begin()+count)proposal[count++]=m;};
            if(retain_incumbent)add(sel[b]);
            if(bx)add(sel[b-1]);if(by)add(sel[b-nb]);if(bz)add(sel[b-size_t(nb)*nb]);
            do{
                rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
                unsigned m=rng%12;
                if(proposal_mode==1){
                    while(std::find(proposal.begin(),proposal.begin()+count,m)!=proposal.begin()+count)m=(m+1)%12;
                }else{
                    std::array<double,draw_limit>weight{};double utility_sum=0,total=0;
                    for(unsigned k=0;k<draw_limit;k++)if(k!=12&&allowed(k)){weight[k]=gain[k]/(trials[k]+8.);utility_sum+=weight[k];}
                    for(unsigned k=0;k<draw_limit;k++)if(k!=12&&allowed(k)){
                        weight[k]=proposal_mode==3&&utility_sum>0?.25/pool+.75*weight[k]/utility_sum:1./pool;
                        if(std::find(proposal.begin(),proposal.begin()+count,k)!=proposal.begin()+count)weight[k]=0;
                        total+=weight[k];
                    }
                    double draw=(double(rng)/4294967296.)*total;
                    m=0;while(m<draw_limit-1&&draw>=weight[m]){draw-=weight[m];++m;}
                }
                proposal[count++]=m;
                if(encoder_work_enabled)encoder_work[13]++;
            }while(proposal_mode>=2&&count<4);
        }
        double best=std::numeric_limits<double>::infinity(),ng_score=0;unsigned chosen=12;
        if constexpr(Large){
            if((bank_diagnostic_mode&1)&&std::find(proposal.begin(),proposal.begin()+count,13)==proposal.begin()+count)proposal[count-1]=13;
        }
        std::array<uint8_t,64>candidate{},best_symbols{};
        for(unsigned j=0;j<count;j++){
            unsigned m=proposal[j];double score=std::log2(double(b)+menu)-std::log2(double(counts[m])+1);
            // Dispatch once per proposal, not once per voxel. Compile-time M
            // removes unused directional/clamp branches without changing sums.
#define BANK_SHORTLIST_CASE(M) case M:score=bank_shortlist_block<M,Independent>(src,d,bx*4,by*4,bz*4,ng,classes,cost,candidate,score,core,weights,fitted,sparse);break;
            switch(m){BANK_SHORTLIST_CASE(0) BANK_SHORTLIST_CASE(1) BANK_SHORTLIST_CASE(2)
                BANK_SHORTLIST_CASE(3) BANK_SHORTLIST_CASE(4) BANK_SHORTLIST_CASE(5)
                BANK_SHORTLIST_CASE(6) BANK_SHORTLIST_CASE(7) BANK_SHORTLIST_CASE(8)
                BANK_SHORTLIST_CASE(9) BANK_SHORTLIST_CASE(10) BANK_SHORTLIST_CASE(11) BANK_SHORTLIST_CASE(12)
                BANK_SHORTLIST_CASE(13) BANK_SHORTLIST_CASE(14) BANK_SHORTLIST_CASE(15) BANK_SHORTLIST_CASE(16)
                BANK_SHORTLIST_CASE(17) BANK_SHORTLIST_CASE(18) BANK_SHORTLIST_CASE(19) BANK_SHORTLIST_CASE(20)
                BANK_SHORTLIST_CASE(21) BANK_SHORTLIST_CASE(22) BANK_SHORTLIST_CASE(23)
                BANK_SHORTLIST_CASE(24) BANK_SHORTLIST_CASE(25) BANK_SHORTLIST_CASE(26)
                BANK_SHORTLIST_CASE(27) BANK_SHORTLIST_CASE(28) BANK_SHORTLIST_CASE(29)
                BANK_SHORTLIST_CASE(30) BANK_SHORTLIST_CASE(31) BANK_SHORTLIST_CASE(32)
                BANK_SHORTLIST_CASE(33) BANK_SHORTLIST_CASE(34) BANK_SHORTLIST_CASE(35)}
#undef BANK_SHORTLIST_CASE
            if(m==12)ng_score=score;
            else if(proposal_mode==3){gain[m]+=std::max(0.,ng_score-score);trials[m]++;}
            if(score<best){best=score;chosen=m;best_symbols=candidate;}
            if(bank_diagnostic_enabled&&m<24){bank_diagnostic[m]++;bank_diagnostic[48+m]+=score-ng_score;}
        }
        if(bank_diagnostic_enabled&&chosen<24)bank_diagnostic[24+chosen]++;
        sel[b]=chosen;++counts[chosen];
        if(chosen<12){unsigned q=0;for(unsigned z=bz*4;z<bz*4+4;z++)for(unsigned y=by*4;y<by*4+4;y++)for(unsigned x=bx*4;x<bx*4+4;x++)fixed[(size_t(z)*d+y)*d+x]=best_symbols[q++];}
        if(encoder_work_enabled)encoder_work[7]+=count*64;
    }
    if(encoder_work_enabled)encoder_work[6]++;
}
struct BankEncodeTimer {
    unsigned phase;
    std::chrono::steady_clock::time_point start;
    explicit BankEncodeTimer(unsigned p):phase(p){if(bank_encode_profile_enabled){start=std::chrono::steady_clock::now();if(p==1)ng_set_profile_callback(bank_fit_profile_tick);}}
    ~BankEncodeTimer(){if(bank_encode_profile_enabled){bank_encode_times[phase]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();if(phase==1)ng_set_profile_callback(nullptr);}}
};
thread_local double bank_detail[6];
thread_local std::chrono::steady_clock::time_point bank_detail_tick;
thread_local double* bank_profile=nullptr;
thread_local std::chrono::steady_clock::time_point bank_tick;
void bank_mark(unsigned phase){if(bank_profile){auto now=std::chrono::steady_clock::now();bank_profile[phase]+=std::chrono::duration<double>(now-bank_tick).count();bank_tick=now;}}
template<unsigned M>
void bank_fixed_single(const uint8_t* __restrict src,uint8_t* __restrict out,unsigned d){
    bool simd=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    simd=!ng_runtime_only&&__builtin_cpu_supports("avx2");
    uint8_t* outputs[12]={};outputs[M]=out;
#endif
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++){
        const size_t row=(size_t(z)*d+y)*d;
        out[row]=pyramid(uint8_t(src[row]-ng_directional_base(src,row,M,0,y,z,d)));
        // Dispatch once per volume, with constant predictor and interior-axis
        // availability. This lets the compiler vectorize independent values;
        // the predictor definition and boundary fallback remain shared.
        unsigned begin=1;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(simd&&z&&y)begin=ng_fixed_row_avx2<1u<<M>(src,outputs,row,d);
#endif
        for(unsigned x=begin;x<d;x++){
            size_t i=row+x;
            out[i]=pyramid(uint8_t(src[i]-ng_directional_base(src,i,M,1,y,z,d)));
        }
    }
}
void bank_fixed_candidates(const uint8_t* __restrict src,std::vector<Bytes>& buffers,unsigned d,unsigned mask){
    if(ng_screen_work&&ng_screen_size==size_t(d)*d*d){
        for(size_t i=0;i<ng_screen_size;i++)if(ng_screen_work[i]){
            unsigned x=i%d,y=i/d%d,z=i/(size_t(d)*d);
            for(unsigned m=0;m<12;m++)if(mask&(1u<<m))
                buffers[m][i]=pyramid(uint8_t(src[i]-ng_directional_base(src,i,m,x,y,z,d)));
        }
        return;
    }
    if(mask!=4095){
        // Fixed selections need only the requested predictor. Share the same
        // boundary/predictor definition rather than constructing other arrays.
        for(unsigned m=0;m<12;m++)if(mask&(1u<<m)){
            switch(m){
#define NG_FIXED_SINGLE(M) case M: bank_fixed_single<M>(src,buffers[m].data(),d);break;
                NG_FIXED_SINGLE(0) NG_FIXED_SINGLE(1) NG_FIXED_SINGLE(2) NG_FIXED_SINGLE(3)
                NG_FIXED_SINGLE(4) NG_FIXED_SINGLE(5) NG_FIXED_SINGLE(6) NG_FIXED_SINGLE(7)
                NG_FIXED_SINGLE(8) NG_FIXED_SINGLE(9) NG_FIXED_SINGLE(10) NG_FIXED_SINGLE(11)
#undef NG_FIXED_SINGLE
            }
        }
        return;
    }
    // Generate the complete fixed palette with one shared neighborhood load.
    uint8_t* out[12];for(unsigned m=0;m<12;m++)out[m]=buffers[m].data();
    const size_t plane=size_t(d)*d;
    bool simd=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    simd=!ng_runtime_only&&__builtin_cpu_supports("avx2");
#endif
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++){
        const size_t row=(size_t(z)*d+y)*d;
        if(!z||!y){
            for(unsigned m=0;m<12;m++)for(unsigned x=0;x<d;x++)out[m][row+x]=pyramid(uint8_t(src[row+x]-ng_directional_base(src,row+x,m,x,y,z,d)));
            continue;
        }
        for(unsigned m=0;m<12;m++)out[m][row]=pyramid(uint8_t(src[row]-ng_directional_base(src,row,m,0,y,z,d)));
        unsigned begin=1;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(simd)begin=ng_fixed_row_avx2(src,out,row,d);
#endif
        for(unsigned x=begin;x<d;x++){
            size_t i=row+x;int v=src[i],a=src[i-1],b=src[i-d],c=src[i-plane];
            int xy=a+b-src[i-d-1],xz=a+c-src[i-plane-1],yz=b+c-src[i-plane-d];
            int xyz=xy+xz+yz-a-b-c+src[i-plane-d-1];
            out[0][i]=pyramid(uint8_t(v));out[1][i]=pyramid(uint8_t(v-a));out[2][i]=pyramid(uint8_t(v-b));out[3][i]=pyramid(uint8_t(v-xy));
            out[4][i]=pyramid(uint8_t(v-c));out[5][i]=pyramid(uint8_t(v-xz));out[6][i]=pyramid(uint8_t(v-yz));out[7][i]=pyramid(uint8_t(v-xyz));
            out[8][i]=pyramid(uint8_t(v-std::clamp(xy,std::min(a,b),std::max(a,b))));
            out[9][i]=pyramid(uint8_t(v-std::clamp(xz,std::min(a,c),std::max(a,c))));
            out[10][i]=pyramid(uint8_t(v-std::clamp(yz,std::min(b,c),std::max(b,c))));
            out[11][i]=pyramid(uint8_t(v-std::clamp(xyz,std::min(a,std::min(b,c)),std::max(a,std::max(b,c)))));
        }
    }
}
}
extern "C" { thread_local NG_TLS_CONSTINIT int ng_bank_detail_enabled=0; }
extern "C" void ng_bank_encode_profile_enable(int enabled){bank_encode_profile_enabled=enabled;encoder_work_enabled=enabled;std::fill(encoder_work,encoder_work+14,0.);std::fill(bank_encode_times,bank_encode_times+8,0.);std::fill(bank_fit_times,bank_fit_times+5,0.);std::fill_n(bank_entropy_times,3,0.);std::fill_n(bank_regression_times,5,0.);}
extern "C" void ng_bank_encode_profile_get(double*out){std::copy(bank_encode_times,bank_encode_times+8,out);}
extern "C" void ng_bank_encode_work_get(double*out){std::copy(encoder_work,encoder_work+14,out);}
extern "C" void ng_bank_encode_bound_enable(int enabled){encoder_bound_enabled=enabled!=0;}
// Experiments only: 0 retains the full search; 1/2 bound stale refit seeds.
extern "C" void ng_bank_encode_search_policy(unsigned policy){bank_search_policy=policy;}
extern "C" void ng_bank_detail_mark(int phase){
    auto now=std::chrono::steady_clock::now();
    if(phase>=0)bank_detail[phase]+=std::chrono::duration<double>(now-bank_detail_tick).count();
    bank_detail_tick=now;
}
extern "C" void ng_bank_detail_get(double* out){if(out){std::copy(bank_detail,bank_detail+4,out);out[2]+=bank_detail[4]+bank_detail[5];}}
extern "C" void ng_bank_detail_parts(double* out){if(out)std::copy(bank_detail+4,bank_detail+6,out);}
static int ng_direction_bank_encode_impl(const uint8_t*src,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*written,uint64_t*stats,unsigned effort,BankWinner* winner=nullptr,const ng_bank_search_options*options=nullptr){
    try{
        const bool independent=options&&options->seed==18;
        const bool sparse=independent&&bank_sparse_stride;
        const bool two=independent&&bank_sparse_models==2;
        require(src&&out&&written&&mode<4&&ng_geometry_valid(d,d,d));const bool shared=mode>=2&&!independent;mode&=1;GeometryScope geometry(d,d,d);
        if(options)require((shared||independent)&&mode&&options->seed<=18&&options->rounds>=1&&options->rounds<=6&&options->passes>=1&&options->passes<=6&&options->thresholds<=2);
        const bool shortlist=options&&options->seed>=8;
        const bool shortlist_refit=shortlist&&!independent&&options->seed>=14;
        const bool shortlist_again=shortlist&&!independent&&(options->seed==12||options->seed==13||options->seed>=15);
        const bool shortlist_refresh=shortlist&&!independent&&(options->seed==13||options->seed>=15);
        const bool shortlist_large=shortlist&&!independent&&options->seed>=16;
        ShortlistHistory shortlist_history;
        if(shortlist)require(options->rounds==1&&options->passes==1&&options->thresholds>=1);
        const bool joint_search=!options&&bank_search_policy>=3&&shared&&mode&&!effort;
        const bool checkpoints=joint_search&&bank_search_policy>=4;
        const bool full_seeds=options?options->seed==4:!effort&&!joint_search;
        // Derive work from the requested search, not from the full palette's capacity.
        const bool reuse_ng=options&&!shortlist&&(options->seed==1||options->seed>=5);
        const bool fixed_selection=options&&!shortlist&&options->seed!=0&&options->seed!=4&&options->rounds==1&&options->passes==1;
        const bool fixed_ng=fixed_selection&&reuse_ng;
        const unsigned fixed_mask=fixed_selection?(fixed_ng?0:1u<<(options->seed==2?11:0)):4095;
        const bool deferred_payload=winner&&!stats&&bank_single_estimated_trial(options);
        const size_t voxels=N;const int cfg[6]={3,3,1,16,40,int(independent?bank_regression_stride:128)};unsigned nb=d/4;size_t blocks=size_t(nb)*nb*nb;
        std::vector<Bytes> candidates(24),params(12);std::vector<std::vector<uint16_t>>base(12);
        for(unsigned m=0;m<(effort?13u:24u);m++)if(((independent&&!sparse)&&(!two||m<=12||m==19))||(!shortlist&&!fixed_selection)||m==12||(!shortlist&&m<12&&(fixed_mask&(1u<<m))))candidates[m].resize(voxels);
        Bytes shortlist_fixed(shortlist?N:0);std::vector<uint16_t>joint_core(shortlist_large?N:0);
        std::array<std::array<int32_t,2>,12>weights{};
        std::unique_ptr<void,decltype(&ng_fit_session_destroy)> fit_session(
            independent&&!sparse&&bank_fit_acceleration?ng_fit_session_create(bank_fit_acceleration==2):nullptr,ng_fit_session_destroy);
        if(independent&&!sparse&&bank_fit_acceleration)require(bool(fit_session));
        std::unique_ptr<void,decltype(&ng_sparse_destroy)> sparse_bank(nullptr,ng_sparse_destroy);
        Bytes feature(sparse||(options&&options->rounds==1&&!shortlist_refit&&!independent)?0:N),cl(N),initial_cl,symbols,raw(deferred_payload?0:N);
        std::vector<uint16_t>activity(N);std::vector<uint32_t>block(fixed_selection||shortlist?0:N);
        if(!sparse&&!fixed_ng&&(!shortlist||independent)){BankEncodeTimer timer(0);
        if(!fixed_selection&&!shortlist)for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;block[i]=((z/4)*nb+y/4)*nb+x/4;
        }
        bank_fixed_candidates(src,candidates,d,fixed_mask);
        }
        {BankEncodeTimer timer(1);
        for(unsigned m=0;m<12;m++){
            if(two&&m&&m!=7){params[m]=params[0];continue;}
            params[m].resize(65536);size_t pn=0;
            if(!sparse&&(!m||!shared)&&!fixed_selection)base[m].resize(voxels);
            if(!m)require(ng_ex_prepare_activity(src,cfg,0,0,params[m].data(),params[m].size(),&pn,candidates[12].data(),cl.data(),(sparse||fixed_selection)?nullptr:base[m].data(),reuse_ng?activity.data():nullptr)==0);
            else if(sparse){
                require(ng_sparse_fit(sparse_bank.get(),m,nullptr,params[m].data(),params[m].size(),&pn,weights[m].data())==0);
            }else if(shared){
                params[m]=params[0];weights[m]={0,0};
                if(!mode){base[m]=base[0];for(size_t i=0;i<voxels;i++)base[m][i]=uint16_t(base[m][i]+256u*uint8_t(src[i]-unpyramid(candidates[m][i])));}
                continue;
            }else{
                for(size_t i=0;i<voxels;i++)feature[i]=uint8_t(src[i]-unpyramid(candidates[m][i]));
                if(fit_session)require(ng_feature_prepare_cached(fit_session.get(),src,cfg,feature.data(),weights[m].data(),params[m].data(),params[m].size(),&pn,candidates[12+m].data(),cl.data(),base[m].data())==0);
                else if(mode)require(ng_feature_prepare(src,cfg,feature.data(),1,weights[m].data(),params[m].data(),params[m].size(),&pn,candidates[12+m].data(),cl.data(),base[m].data())==0);
                else require(ng_base_prepare(src,cfg,feature.data(),params[m].data(),params[m].size(),&pn,candidates[12+m].data(),cl.data(),base[m].data())==0);
            }
            params[m].resize(pn);if(!sparse&&!fixed_selection)remove_class_bias(base[m],params[m],cl,16);
            if(sparse&&!m){
                sparse_bank.reset(ng_sparse_create(src,params[0].data(),pn,cl.data(),bank_sparse_stride));require(bool(sparse_bank));
            }
            if(independent&&!sparse&&!m)initial_cl=cl;
        }
        }
        if(independent&&!sparse)cl=std::move(initial_cl);
        std::vector<const uint8_t*>ptr(24),pp(12);std::vector<uint8_t*>wptr(24);std::vector<const uint16_t*>bp(12);std::vector<size_t>pn(12);
        for(unsigned m=0;m<24;m++){ptr[m]=candidates[m].data();wptr[m]=candidates[m].data();}
        if(shortlist&&!independent)for(unsigned m=0;m<12;m++){ptr[m]=shortlist_fixed.data();wptr[m]=shortlist_fixed.data();}
        for(unsigned m=0;m<12;m++){pp[m]=two&&m&&m!=7?nullptr:params[m].data();pn[m]=pp[m]?params[m].size():0;bp[m]=base[m].data();}
        // Retain both established block assignments/entropy thresholds, without
        // whole-frame fallback. All twelve models remain charged in this trial.
        Bytes oldsel(blocks),clampsel(blocks),oldpayload;
        std::vector<uint16_t>oldthr(15),cthr(15);
        if(full_seeds){
        Bytes old(N*3+131072);size_t on=0;uint64_t unused[15];
        Bytes clampframe(N*3+131072);size_t cn=0;
        {BankEncodeTimer timer(2);require(ng_activity_refined_encode(src,d,13,clampframe.data(),clampframe.size(),&cn)==0);}clampframe.resize(cn);
        {BankEncodeTimer timer(2);require(ng_lorenzo_encode_impl(src,d,365,old.data(),old.size(),&on,unused,&clampframe)==0);}old.resize(on);
        size_t p=24;auto sn=get32(old,p);unpack_stream(old.data()+p,sn,oldsel,true);p+=sn;
        auto oldpn=get32(old,p);
        if(mode&&!shared){require(oldpn==params[11].size()&&std::equal(params[11].begin(),params[11].end(),old.begin()+p));
            size_t wp=16;for(auto w:weights[11])require(get32(old,wp)==uint32_t(w));}
        p+=oldpn;oldpayload.assign(old.begin()+p,old.end());
        for(auto&t:oldthr){t=old[p]|unsigned(old[p+1])<<8;p+=2;}
        for(auto&m:oldsel)m=m==0?23:m==8?11:m==9?0:m;
        p=8;
        if(clampframe[2]=='C'){auto m=get32(clampframe,p);std::fill(clampsel.begin(),clampsel.end(),m);require(get32(clampframe,p)==16);}
        else if(clampframe[2]=='B'){require(get32(clampframe,p)==4&&get32(clampframe,p)==16&&get32(clampframe,p)==blocks);auto n=get32(clampframe,p);unpack_stream(clampframe.data()+p,n,clampsel,true);p+=n;}
        else{require(clampframe[2]=='D');get32(clampframe,p);require(get32(clampframe,p)==1);auto n=get32(clampframe,p);unpack_stream(clampframe.data()+p,n,clampsel,true);p+=n;}
        for(auto&t:cthr){t=clampframe[p]|unsigned(clampframe[p+1])<<8;p+=2;}
        for(auto&m:clampsel)if(m==8)m=11;
        }else{
            require((shared||independent)&&mode&&effort<=3);
            size_t p=params[0].size()-62;
            for(auto&t:oldthr){t=params[0][p]|unsigned(params[0][p+1])<<8;p+=2;}
            cthr=oldthr;
            if(!independent)for(unsigned m=13;m<24;m++)ptr[m]=candidates[12].data();
            if(options&&options->seed){const unsigned choices[]={0,12,11,0,0,23,19,13};std::fill(oldsel.begin(),oldsel.end(),shortlist?12:choices[options->seed]);}
            else {BankEncodeTimer timer(7);oldsel=choose_blocks(ptr.data(),24,cl.data(),16,4);}
        }
        if(shortlist){BankEncodeTimer timer(6);
            if(independent)bank_shortlist_select<true,true>(src,d,3,candidates[12],cl,oldsel,shortlist_fixed,&shortlist_history,false,nullptr,nullptr,nullptr,ptr.data(),sparse_bank.get());
            else bank_shortlist_select(src,d,std::min(options->seed-8,3u),candidates[12],cl,oldsel,shortlist_fixed,&shortlist_history);
        }
        std::vector<Bytes>seeds{oldsel};
        if(full_seeds){seeds.push_back(clampsel);seeds.push_back(choose_blocks(ptr.data(),24,cl.data(),16,4));}
        if(joint_search&&bank_search_policy>=5){
            // Region proposals select one predictor for 8^3/16^3, then refine
            // freely at4^3. No extra signaling or spatial constraint survives.
            for(unsigned B:{8u,16u}){
                auto coarse=choose_blocks(ptr.data(),24,cl.data(),16,B);
                Bytes expanded(blocks);unsigned nc=d/B,scale=B/4;
                for(unsigned z=0;z<nb;z++)for(unsigned y=0;y<nb;y++)for(unsigned x=0;x<nb;x++)
                    expanded[(z*nb+y)*nb+x]=coarse[((z/scale)*nc+y/scale)*nc+x/scale];
                if(std::find(seeds.begin(),seeds.end(),expanded)==seeds.end())seeds.push_back(std::move(expanded));
            }
        }
        Bytes best,bestsel;size_t bestselector=0,bestpayload=0,modelbytes=0;
        double best_estimate=std::numeric_limits<double>::infinity();
        for(unsigned m=0;m<12;m++)modelbytes+=params[m].size()+4+((mode&&m)?8:0);
        // Sharing never removes a predictor choice. Refit ONE common model
        // against the selected, spatially varying base, then re-evaluate all
        // 24 choices with the existing search. Keep the best complete trial.
        auto refit_models=[&](const Bytes&selected){
            BankEncodeTimer timer(1);
            for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned bx=0;bx<nb;bx++){
                unsigned c=selected[((z/4)*nb+y/4)*nb+bx],m=c>=12?c-12:c;
                for(unsigned x=bx*4;x<bx*4+4;x++){size_t i=(size_t(z)*d+y)*d+x;feature[i]=(bank_diagnostic_mode&2)?ng_directional_base(src,i,1,x,y,z,d):m?uint8_t(src[i]-unpyramid(ptr[m][i])):0;}
            }
            Bytes fitted(65536);size_t fn=0;std::array<int32_t,2>fw{};
            if(mode)require(ng_feature_prepare(src,cfg,feature.data(),1,fw.data(),fitted.data(),fitted.size(),&fn,candidates[12].data(),cl.data(),base[0].data())==0);
            else require(ng_base_prepare(src,cfg,feature.data(),fitted.data(),fitted.size(),&fn,candidates[12].data(),cl.data(),base[0].data())==0);
            fitted.resize(fn);remove_class_bias(base[0],fitted,cl,16);
            // Only the low 16 bits survive. Unsigned 16-bit coefficients give
            // exactly the same modular product, including negative weights,
            // and avoid a 64-bit multiply plus bank branch for every voxel.
            for(unsigned bank=0;bank<2;bank++){
                const size_t begin=bank?size_t(d)*d:0,end=bank?voxels:size_t(d)*d;
                const uint16_t weight=uint16_t(mode?fw[!bank]:256);
                for(size_t i=begin;i<end;i++)base[0][i]=uint16_t(base[0][i]-unsigned(weight)*feature[i]);
            }
            for(unsigned m=0;m<12;m++){
                params[m]=fitted;if(m&&shortlist_large)weights[m]=fw;
                if(m&&!shortlist){base[m].resize(voxels);weights[m]=fw;
                    for(unsigned bank=0;bank<2;bank++){
                        const size_t begin=bank?size_t(d)*d:0,end=bank?voxels:size_t(d)*d;
                        const uint16_t weight=uint16_t(mode?fw[!bank]:256);
                        for(size_t i=begin;i<end;i++)base[m][i]=uint16_t(base[0][i]+unsigned(weight)*uint8_t(src[i]-unpyramid(candidates[m][i])));
                    }
                }
                pp[m]=params[m].data();pn[m]=params[m].size();bp[m]=base[m].data();
            }
        };
        if(shortlist_refit)refit_models(oldsel);
        for(unsigned round=0;round<(options?options->rounds:shared?(effort?effort:checkpoints?6u:3u):1u);round++){
        const Bytes round_start_selection=checkpoints?bestsel:Bytes{};
        if(round){
            refit_models(bestsel);
            if(options||effort||bank_search_policy==1||joint_search)seeds={bestsel};
            else if(bank_search_policy==2){seeds={bestsel};if(clampsel!=bestsel)seeds.push_back(clampsel);}
            else seeds.push_back(bestsel);
        }
        // Before refitting, shared-input features have zero coefficients, so
        // all twelve NG choices have exactly the same candidate symbols.
        // Keep distinct selector IDs/costs, but calculate their array once.
        for(unsigned m=0;m<12;m++){
            if(two&&m&&m!=7)continue;
            unsigned owner=shared&&mode&&round==0?0:m;
            if(!sparse&&candidates[12+owner].empty())candidates[12+owner].resize(voxels);
            bp[m]=base[owner].data();wptr[12+m]=candidates[12+owner].data();ptr[12+m]=wptr[12+m];
        }
        for(unsigned sweep=0;sweep<(shortlist_again?(options->seed==17?2u:1u):0u);sweep++){
            Bytes refreshed_classes;
            if(shortlist_refresh){
                BankEncodeTimer timer(3);symbols.resize(N);
                if(encoder_work_enabled){encoder_work[8]++;encoder_work[9]+=voxels;}
                require((shortlist_large?ng_bank_shared_symbols(12,pp.data(),pn.data(),bp.data(),oldsel.data(),src,wptr.data(),symbols.data(),activity.data(),weights[1].data(),joint_core.data()):ng_bank_symbols(12,pp.data(),pn.data(),bp.data(),oldsel.data(),src,wptr.data(),symbols.data(),activity.data()))==0);
                refreshed_classes=activity_classify(activity,oldthr);
            }
            {BankEncodeTimer timer(6);
                if(shortlist_large)bank_shortlist_select<true>(src,d,3,candidates[12],refreshed_classes,
                    oldsel,shortlist_fixed,&shortlist_history,true,&symbols,joint_core.data(),weights[1].data());
                else bank_shortlist_select(src,d,3,candidates[12],shortlist_refresh?refreshed_classes:cl,
                    oldsel,shortlist_fixed,&shortlist_history,true,shortlist_refresh?&symbols:nullptr);
            }
            seeds={oldsel};
        }
        // Exact memoization, scoped to a fixed fitted model. Revisiting a
        // selector follows the same transition, without repeated entropy work.
        std::map<Bytes,Bytes> transitions;
        const unsigned passes=options?options->passes:effort?(effort==1?1:2):checkpoints?8:3;
        for(auto sel:seeds)for(unsigned pass=0;pass<passes;pass++){
            auto found=transitions.find(sel);
            if(found!=transitions.end()){if(checkpoints)break;sel=found->second;continue;}
            const Bytes input_sel=sel;
            if(sparse){
                BankEncodeTimer timer(3);symbols.resize(N);require(ng_sparse_emit(sparse_bank.get(),sel.data(),symbols.data(),activity.data())==0);
                if(encoder_work_enabled){encoder_work[8]++;encoder_work[9]+=N;}
            }else if(ng_simple_activity_values&&independent){
                BankEncodeTimer timer(3);symbols.resize(N);
                memcpy(activity.data(),ng_simple_activity_values,2*N);
                for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
                    size_t i=(size_t(z)*d+y)*d+x;
                    unsigned m=sel[((z/4)*nb+y/4)*nb+x/4];symbols[i]=candidates[m][i];
                }
            }else if(fixed_selection&&!fixed_ng){
                BankEncodeTimer timer(3);
                symbols.swap(candidates[sel[0]]);
                require(ng_activity_symbol_values(symbols.data(),activity.data())==0);
            }else if(reuse_ng&&round==0&&pass==0){
                // Preparation already generated these exact biased symbols and their
                // causal activity. Only a changed selector/model invalidates them.
                if(fixed_ng)symbols.swap(candidates[12]);else symbols=candidates[12];
            }else{
                symbols.resize(N);
                if(encoder_work_enabled){encoder_work[8]++;encoder_work[9]+=voxels*((shared&&mode&&round==0)?1:two?2:12);}
                BankEncodeTimer timer(3);require((shortlist_large?ng_bank_shared_symbols(12,pp.data(),pn.data(),bp.data(),sel.data(),src,wptr.data(),symbols.data(),activity.data(),weights[1].data(),nullptr):ng_bank_symbols(12,pp.data(),pn.data(),bp.data(),sel.data(),src,wptr.data(),symbols.data(),activity.data()))==0);
            }
            if(!deferred_payload)for(size_t i=0;i<voxels;i++)raw[i]=unpyramid(symbols[i]);
            const Bytes& histogram_input=deferred_payload?symbols:raw;
            const auto selector=pack_stream(sel,true);
            Bytes prefix={'N','B','K','1'};put32(prefix,d);put32(prefix,mode);put32(prefix,selector.size());prefix.insert(prefix.end(),selector.begin(),selector.end());
            for(unsigned m=0;m<12;m++){put32(prefix,params[m].size());prefix.insert(prefix.end(),params[m].begin(),params[m].end());if(mode&&m)for(auto w:weights[m])put32(prefix,uint32_t(w));}
            std::map<std::vector<uint16_t>,size_t> evaluated;
            std::map<std::vector<uint64_t>,size_t> evaluated_partitions;
            ActivityHistogram trial_histogram(histogram_input,activity,checkpoints||(options&&options->thresholds==1),deferred_payload,
                !(deferred_payload&&options->thresholds==2));
            auto evaluate=[&](const std::vector<uint16_t>&thr,size_t selection_score=SIZE_MAX){
                BankEncodeTimer timer(4);
                if(deferred_payload){
                    // Exactly one estimated/quantile winner: no trial payload,
                    // size bound, or partition deduplication is needed.
                    best=prefix;bestsel=sel;
                    winner->prefix=prefix;winner->symbols=std::move(symbols);
                    winner->classes=activity_classify(activity,thr);winner->thresholds=thr;
                    return prefix.size();
                }
                auto cached=evaluated.find(thr);if(cached!=evaluated.end())return cached->second;
                bool audit=mode&&!shared&&sel==oldsel&&thr==oldthr;
                std::vector<uint64_t> partition;
                if(encoder_bound_enabled&&!audit){
                    // Thresholds crossing only empty activity bins produce
                    // identical classes/streams. Their fixed-width metadata
                    // changes bytes, not size. Preserve the first exact tie.
                    partition=trial_histogram.partition_key(thr);
                    auto same=evaluated_partitions.find(partition);
                    if(same!=evaluated_partitions.end())return same->second;
                }
                if(encoder_bound_enabled&&!audit&&selection_score!=SIZE_MAX&&!best.empty()){
                    const size_t bound=prefix.size()+trial_histogram.lower_bound(thr);
                    if(encoder_work_enabled)encoder_work[10]++;
                    if(bound>=selection_score&&bound>=best.size()){
                        if(encoder_work_enabled)encoder_work[11]++;
                        evaluated.emplace(thr,SIZE_MAX);evaluated_partitions.emplace(partition,SIZE_MAX);return SIZE_MAX;
                    }
                }
                auto classes=activity_classify(activity,thr);
                auto payload=activity_stream_payload(raw,thr,classes);
                const size_t n=prefix.size()+payload.size();
                bool win=best.empty()||n<best.size();
                if(win||audit){
                    if(audit)require(payload==oldpayload);
                    if(win){best=prefix;best.insert(best.end(),payload.begin(),payload.end());bestsel=sel;bestselector=selector.size();bestpayload=payload.size();
                        if(winner){winner->prefix=prefix;winner->symbols=symbols;winner->classes=classes;winner->thresholds=thr;}}
                }
                evaluated.emplace(thr,n);if(encoder_bound_enabled&&!audit)evaluated_partitions.emplace(partition,n);return n;
            };
            auto thr=oldthr;
            if(options&&options->thresholds==2){thr=activity_quantiles(activity,16);evaluate(thr);}
            else if(checkpoints||(options&&options->thresholds==1)){
                double estimate=trial_histogram.estimate(thr);
                for(auto t:{cthr,activity_quantiles(activity,16)}){
                    double e=trial_histogram.estimate(t);if(e<estimate){estimate=e;thr=t;}
                }
                std::vector<uint16_t> tuned;
                {BankEncodeTimer timer(5);tuned=activity_refine(histogram_input,activity,thr,block,sel,1,deferred_payload);}
                double e=trial_histogram.estimate(tuned);if(e<estimate){estimate=e;thr=tuned;}
                estimate+=prefix.size();
                // Keep a narrow uncertainty band for the final rANS state size.
                if(options||best.empty()||estimate<=best_estimate*1.0005)evaluate(thr,0);
                best_estimate=std::min(best_estimate,estimate);
            }else{
            size_t score=evaluate(thr);
            for(auto t:{cthr,activity_quantiles(activity,16)}){size_t n=evaluate(t,score);if(n<score){score=n;thr=t;}}
            std::vector<uint16_t> tuned;{BankEncodeTimer timer(5);tuned=activity_refine(raw,activity,thr,block,sel,1);}if(evaluate(tuned,pass+1==passes?0:score)<score)thr=tuned;
            }
            // The last selection was discarded at the end of the seed loop.
            if(pass+1<passes){BankEncodeTimer timer(6);activity_reselect(symbols,activity_classify(activity,thr),block,ptr,sel,1);transitions.emplace(input_sel,sel);if(checkpoints&&sel==input_sel)break;}
        }
        if(checkpoints&&round&&bestsel==round_start_selection)break;
        }
        if(deferred_payload)*written=0; // winner owns the handoff; no intermediate frame
        else {require(best.size()<=cap);memcpy(out,best.data(),best.size());*written=best.size();}
        if(stats){std::fill(stats,stats+28,0);stats[0]=modelbytes;stats[1]=bestselector;stats[2]=bestpayload;stats[3]=16;for(auto m:bestsel)stats[4+m]++;}
        return 0;
    }catch(...){return -1;}
}
extern "C" int ng_direction_bank_encode(const uint8_t*src,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*written,uint64_t*stats){
    return ng_direction_bank_encode_impl(src,d,mode,out,cap,written,stats,0);
}
static int ng_direction_bank_decode_impl(const uint8_t*src,size_t n,uint8_t*out,size_t cap,const uint8_t*payload=nullptr,size_t payload_size=0,bool reuse=false){
    try{
        require(src&&out&&n>=16&&(!memcmp(src,"NBK1",4)||!memcmp(src,"NBK2",4)||!memcmp(src,"NBK3",4)));Bytes f(src,src+n);size_t p=4;unsigned d=get32(f,p),mode=get32(f,p),sn=get32(f,p);
        require(mode<2&&ng_geometry_valid(d,d,d)&&cap>=size_t(d)*d*d&&sn<=n-p);GeometryScope geometry(d,d,d);unsigned nb=d/4;
        Bytes sel(size_t(nb)*nb*nb);unpack_stream(src+p,sn,sel,true);p+=sn;for(auto m:sel)require(m<24);
        unsigned mask=4095;if(src[3]!='1'){require(n-p>=2);mask=src[p]|unsigned(src[p+1])<<8;p+=2;require(mask<4096);}
        for(auto m:sel)if(m>=12)require(mask&(1u<<(m-12)));
        using State=std::unique_ptr<void,decltype(&ng_causal_destroy)>;std::vector<State>state;std::array<std::array<int32_t,2>,12>weights{};
        bool history=false;
        void* models[12]={0};const uint8_t*first_params=nullptr;size_t first_size=0;void*first_state=nullptr;
        for(unsigned m=0;m<12;m++){if(!(mask&(1u<<m))){state.emplace_back(nullptr,ng_causal_destroy);continue;}
            auto pn=get32(f,p);require(pn<=n-p);
            if(reuse&&payload&&first_params&&pn==first_size&&!memcmp(first_params,src+p,pn)){
                state.emplace_back(nullptr,ng_causal_destroy);models[m]=first_state;
            }else{
                state.emplace_back(history?ng_causal_create_model(src+p,pn):ng_causal_create(src+p,pn),ng_causal_destroy);require(bool(state.back()));models[m]=state.back().get();
                if(!first_params){first_params=src+p;first_size=pn;first_state=models[m];}
            }
            history=true;p+=pn;
            if(mode&&m)for(auto&w:weights[m]){auto v=get32(f,p);w=v<=INT32_MAX?int32_t(v):int32_t(int64_t(v)-4294967296LL);}}
        bank_mark(0);
        if(src[3]=='3'){
            void* states[12];for(unsigned m=0;m<12;m++)states[m]=models[m];
            if(ng_bank_detail_enabled)ng_bank_detail_mark(-1);
            bank_mark(1);unsigned wave_mode=mode;
            int rc=payload?ng_bank_wave_segments(states,wave_mode,weights[0].data(),sel.data(),out,src+p,n-p,payload,payload_size):ng_bank_wave(states,wave_mode,weights[0].data(),sel.data(),out,src+p,n-p);
            require(rc==0);bank_mark(2);return 0;
        }
        if(ng_bank_detail_enabled)ng_bank_detail_mark(-1);
        Streams streams;auto thr=activity_parse(f,p,16,1,streams);
        if(ng_bank_detail_enabled)ng_bank_detail_mark(0);
        const uint8_t* pointers[16];size_t counts[16];for(unsigned c=0;c<16;c++){pointers[c]=streams.symbols[c].data();counts[c]=streams.symbols[c].size();}
        require(ng_activity_gather(thr.data(),16,pointers,counts,streams.pos.data(),out)==0);
        for(size_t c=0;c<streams.pos.size();c++)require(streams.pos[c]==streams.symbols[c].size());
        if(ng_bank_detail_enabled)ng_bank_detail_mark(1);
        bank_mark(1);
        void* states[12];for(unsigned m=0;m<12;m++)states[m]=state[m].get();
        require(ng_bank_reconstruct(states,mode,weights[0].data(),sel.data(),out)==0);
        bank_mark(2);return 0;
    }catch(...){return -1;}
}
extern "C" int ng_direction_bank_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){return ng_direction_bank_decode_impl(src,n,out,cap);}
static int ng_direction_bank_profile_impl(const uint8_t*src,size_t n,uint8_t*out,size_t cap,double* times,const uint8_t*payload=nullptr,size_t payload_size=0,bool reuse=false){
    if(!times)return -1;std::fill(times,times+3,0.);bank_profile=times;bank_tick=std::chrono::steady_clock::now();
    std::fill(bank_detail,bank_detail+6,0.);ng_bank_detail_enabled=1;
    int rc=ng_direction_bank_decode_impl(src,n,out,cap,payload,payload_size,reuse);bank_profile=nullptr;ng_bank_detail_enabled=0;return rc;
}
extern "C" int ng_direction_bank_profile(const uint8_t*src,size_t n,uint8_t*out,size_t cap,double* times){return ng_direction_bank_profile_impl(src,n,out,cap,times);}
// Reinsert joint-Zstd's already decompressed compact tables without Python
// per-frequency objects or serializing the same frequencies a second time.
extern "C" int ng_restore_compact_tables(const uint8_t*tables,size_t tn,const uint32_t*gaps,unsigned count,
    const uint8_t*body,size_t bn,uint8_t*out,size_t cap){
    try{
        require(tables&&gaps&&body&&out&&count>0&&count<=17&&tn<=cap&&bn==cap-tn);
        size_t tp=0,bp=0,op=0;uint16_t freq[256];
        for(unsigned c=0;c<count;c++){
            size_t used=0;require(ng_read_freq_table(tables+tp,tn-tp,freq,&used)==0&&used<=tn-tp);
            require(gaps[c]<=bn-bp);memcpy(out+op,body+bp,gaps[c]);op+=gaps[c];bp+=gaps[c];
            memcpy(out+op,tables+tp,used);op+=used;tp+=used;
        }
        require(tp==tn);memcpy(out+op,body+bp,bn-bp);return 0;
    }catch(...){return -1;}
}
