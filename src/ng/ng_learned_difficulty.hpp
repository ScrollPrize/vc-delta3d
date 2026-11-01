// Coding-cost-trained integer activity correction. Same causal tap positions
// as the saved NG models; no additional source information at decode time.
namespace learned_difficulty {
constexpr unsigned max_features=15,span=24576;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
void apply_row_avx2(unsigned d,int32_t*score,const uint8_t*value,const uint8_t*north,const int32_t*a,const int32_t*b){
    for(unsigned x=0;x<d;x+=8){
        __m256i v=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(value+x)));
        __m256i n=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(north+x)));
        __m256i u=_mm256_mullo_epi32(_mm256_sub_epi32(v,_mm256_set1_epi32(128)),_mm256_loadu_si256((const __m256i*)(a+x)));
        __m256i t=_mm256_mullo_epi32(_mm256_abs_epi32(_mm256_sub_epi32(v,n)),_mm256_loadu_si256((const __m256i*)(b+x)));
        _mm256_storeu_si256((__m256i*)(score+x),_mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(score+x)),_mm256_add_epi32(u,t)));
    }
}
#endif
void apply_row(unsigned d,int32_t*score,const uint8_t*value,const uint8_t*north,const int32_t*a,const int32_t*b){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if(vector_enabled&&__builtin_cpu_supports("avx2")){apply_row_avx2(d,score,value,north,a,b);return;}
#endif
    for(unsigned x=0;x<d;x++)score[x]+=a[x]*(int(value[x])-128)+b[x]*std::abs(int(value[x])-int(north[x]));
}
struct Model {
    std::vector<std::array<int,3>> taps;
    std::vector<int32_t> weights,cuts;
    Bytes lut;
    std::vector<std::array<int32_t,6>> current_weights;
    unsigned B=1,K=16;
    unsigned F()const{return 1+2*taps.size();}
    void rebuild(){
        lut.resize(B*span);for(unsigned b=0;b<B;b++){unsigned c=0;for(int s=-8192;s<16384;s++){while(c+1<K&&s>cuts[b*(K-1)+c])c++;lut[b*span+s+8192]=c;}}
        current_weights.assign(B,{});for(unsigned t=0;t<taps.size();t++)if(!taps[t][1]&&!taps[t][2])for(unsigned b=0;b<B;b++){unsigned k=-taps[t][0]-1;require(k<3);current_weights[b][2*k]=weights[b*F()+1+2*t];current_weights[b][2*k+1]=weights[b*F()+2+2*t];}
    }
    template<class Read> void features(Read read,int16_t*out)const{
        out[0]=32;int north=read(0,-1,0);
        for(unsigned t=0;t<taps.size();t++){auto p=taps[t];int v=read(p[0],p[1],p[2]);out[1+2*t]=v-128;out[2+2*t]=std::abs(v-north);}
    }
    unsigned group(unsigned cls,unsigned bank,const int16_t*f)const{
        int score=256*int(cls);const auto*w=weights.data()+bank*F();
        for(unsigned k=0;k<F();k++)score+=w[k]*int(f[k]);
        return score_group(score,bank);
    }
    unsigned score_group(int score,unsigned bank)const{return bank*32+lut[bank*span+std::clamp(score,-8192,16383)+8192];}
};
Model fit(const Snapshot&s,const Bytes&src,unsigned kind){
    Model m;m.B=banks(s);m.K=s.thresholds.size()+1;
    std::vector<std::array<int,3>> pool;
    for(auto&p:s.models){void*state=ng_causal_create_model(p.data(),p.size());require(state);int xyz[3*512];int n=ng_causal_used_taps(state,xyz);ng_causal_destroy(state);require(n>=0&&n<=512);for(int k=0;k<n;k++)pool.push_back({xyz[3*k],xyz[3*k+1],xyz[3*k+2]});}
    std::sort(pool.begin(),pool.end());pool.erase(std::unique(pool.begin(),pool.end()),pool.end());
    std::stable_sort(pool.begin(),pool.end(),[](auto a,auto b){auto norm=[](auto p){return p[0]*p[0]+p[1]*p[1]+p[2]*p[2];};return norm(a)<norm(b);});
    unsigned rows=0,priors=0;for(auto p:pool){bool row=p[1]==0&&p[2]==0;
        if(row){if(kind!=17||p[0]>=0||p[0]<-3||rows==3)continue;rows++;}
        else {if(priors==4)continue;priors++;}m.taps.push_back(p);
    }
    // Delta/direct-only chunks need a context too. Fill missing positions from
    // the standard NG neighborhood, restricted to the minimum delta=2 schedule.
    // Do not let absence of a selected NG predictor silently remove features.
    int xyz[3*512];int nt=ng_axis_taps(xyz);require(nt>=0&&nt<=512);
    pool.clear();for(int k=0;k<nt;k++)pool.push_back({xyz[3*k],xyz[3*k+1],xyz[3*k+2]});
    std::sort(pool.begin(),pool.end());std::stable_sort(pool.begin(),pool.end(),[](auto a,auto b){return a[0]*a[0]+a[1]*a[1]+a[2]*a[2]<b[0]*b[0]+b[1]*b[1]+b[2]*b[2];});
    for(auto p:pool){if(std::find(m.taps.begin(),m.taps.end(),p)!=m.taps.end())continue;bool row=p[1]==0&&p[2]==0;
        if(row){if(kind!=17||rows==3||p[0]>=0||p[0]<-3)continue;rows++;}
        else{if(priors==4||p[1]+2*p[2]>=0)continue;priors++;}m.taps.push_back(p);
    }
    require(priors==4&&(kind!=17||rows==3));
    m.weights.assign(m.B*m.F(),0);m.cuts.resize(m.B*(m.K-1));
    for(unsigned b=0;b<m.B;b++)for(unsigned k=0;k+1<m.K;k++)m.cuts[b*(m.K-1)+k]=256*k+128;
    m.rebuild();if(kind==18)return m;
    struct Sample{int score;uint8_t bank,symbol;};std::vector<Sample>samples;
    std::vector<std::vector<int16_t>>features(m.F());
    for(auto&f:features)f.reserve(src.size());samples.reserve(src.size());
    for(size_t i=0;i<src.size();i++){
        auto&c=s.menu[s.selectors[geometry_trials::block(i,s.d)]];if(c.predictor==choice_frame::zero_predictor)continue;
        int x=i%s.d,y=i/s.d%s.d,z=i/(s.d*s.d);int16_t f[max_features];
        m.features([&](int dx,int dy,int dz){return int(geometry_trials::value(src.data(),x+dx,y+dy,z+dz,s.d));},f);
        for(unsigned k=0;k<m.F();k++)features[k].push_back(f[k]);samples.push_back({int(s.classes[i])*256,uint8_t(c.entropy),s.symbols[i]});
    }
    auto histograms=[&](){std::vector<uint32_t>hist(m.B*32*256);for(auto a:samples){unsigned c=m.lut[a.bank*span+std::clamp(a.score,-8192,16383)+8192];++hist[(a.bank*32+c)*256+a.symbol];}return hist;};
    auto objective=[&](const std::vector<uint32_t>&hist,std::vector<double>&cost){require(ng_context_rans_costs(hist.data(),m.B*32,cost.data())==0);double sum=0;for(unsigned i=0;i<hist.size();i++)sum+=hist[i]*cost[i];return sum;};
    std::vector<double>cost(m.B*32*256);double best_cost=objective(histograms(),cost);Model best=m;
    // Integer coordinate descent on actual normalized symbol costs. Every
    // active voxel participates, no sampling or magnitude-regression proxy.
    for(unsigned pass=0;pass<3;pass++){
        int deltas[5]={0,-8,-2,2,8};if(pass==2){deltas[1]=-2;deltas[2]=-1;deltas[3]=1;deltas[4]=2;}
        for(unsigned k=0;k<m.F();k++){
            double totals[4][5]={};
            std::vector<std::vector<uint32_t>> candidates(5,std::vector<uint32_t>(m.B*32*256));
            for(size_t i=0;i<samples.size();i++){auto a=samples[i];for(unsigned d=0;d<5;d++){
                int score=a.score+deltas[d]*int(features[k][i]);unsigned c=m.lut[a.bank*span+std::clamp(score,-8192,16383)+8192];++candidates[d][(a.bank*32+c)*256+a.symbol];}}
            for(unsigned d=0;d<5;d++){objective(candidates[d],cost);for(unsigned b=0;b<m.B;b++)for(unsigned i=b*32*256;i<(b+1)*32*256;i++)totals[b][d]+=candidates[d][i]*cost[i];}
            int change[4]={};for(unsigned b=0;b<m.B;b++){unsigned d=std::min_element(totals[b],totals[b]+5)-totals[b];change[b]=deltas[d];m.weights[b*m.F()+k]+=change[b];}
            for(size_t i=0;i<samples.size();i++)samples[i].score+=change[samples[i].bank]*int(features[k][i]);
        }
        // Optimize each ordered threshold with the same coding-cost objective.
        for(unsigned k=0;k+1<m.K;k++){
            double totals[4][5]={};int steps[5]={0,-64,-16,16,64};
            auto base=histograms();std::vector<std::vector<uint32_t>>candidates(5,base);
            for(auto a:samples){unsigned b=a.bank;int old=m.cuts[b*(m.K-1)+k];
                for(unsigned d=1;d<5;d++){int cut=old+steps[d];if((k&&cut<=m.cuts[b*(m.K-1)+k-1])||(k+2<m.K&&cut>=m.cuts[b*(m.K-1)+k+1]))continue;
                    if(a.score>std::min(old,cut)&&a.score<=std::max(old,cut)){++candidates[d][(b*32+(cut>old?k:k+1))*256+a.symbol];--candidates[d][(b*32+(cut>old?k+1:k))*256+a.symbol];}}}
            for(unsigned d=0;d<5;d++){objective(candidates[d],cost);for(unsigned b=0;b<m.B;b++)for(unsigned i=b*32*256;i<(b+1)*32*256;i++)totals[b][d]+=candidates[d][i]*cost[i];}
            for(unsigned b=0;b<m.B;b++){unsigned d=std::min_element(totals[b],totals[b]+5)-totals[b];m.cuts[b*(m.K-1)+k]+=steps[d];}
            m.rebuild();
        }
        m.rebuild();double value=objective(histograms(),cost);if(value<best_cost){best_cost=value;best=m;}
    }return best;
}
void write(Bytes&h,const Model&m){put32(h,m.K);put32(h,m.taps.size());for(auto p:m.taps)for(auto v:p)put32(h,uint32_t(v));for(auto v:m.weights)put32(h,uint32_t(v));for(auto v:m.cuts)put32(h,uint32_t(v));}
Model read(packed_bank::Reader&h,unsigned B,unsigned kind){Model m;m.B=B;m.K=h.u(4);require(m.K>=2&&m.K<=32);unsigned n=h.u(4);require(n<=7);m.taps.resize(n);
    for(auto&p:m.taps){for(auto&v:p){v=int32_t(packed_bank::signed_value(h.u(4),32));require(v>=-3&&v<=3);}require(p[2]<0||(p[2]==0&&p[1]<0)||(kind==17&&p[2]==0&&p[1]==0&&p[0]<0));}
    m.weights.resize(B*m.F());for(auto&w:m.weights){w=int32_t(packed_bank::signed_value(h.u(4),32));require(w>=-64&&w<=64);}
    m.cuts.resize(B*(m.K-1));for(auto&v:m.cuts){v=int32_t(packed_bank::signed_value(h.u(4),32));require(v>=-8192&&v<16384);}
    for(unsigned b=0;b<B;b++)require(std::is_sorted(m.cuts.begin()+b*(m.K-1),m.cuts.begin()+(b+1)*(m.K-1)));m.rebuild();return m;}
}
