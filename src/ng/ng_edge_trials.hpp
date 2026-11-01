// Opt-in boundary experiments. Reuse native preset fitting/selection, Q8
// prediction convention, causal state, Cholesky/quantization, rANS and Zstd.
// Scalar reference decoder for compression research, not a speed claim.
#include "ng_compact_contexts.hpp"
namespace edge_trials {
using Snapshot=hybrid_geometry::Snapshot;
constexpr int taps[12][3]={{-1,0,0},{0,-1,0},{0,0,-1},{-2,0,0},{0,-2,0},{0,0,-2},
    {-1,-1,0},{-1,0,-1},{0,-1,-1},{-2,-1,0},{-1,-2,0},{-1,-1,-1}};
unsigned face(size_t i,unsigned d){unsigned x=i%d,y=i/d%d,z=i/(size_t(d)*d);
    return x<4?1:x+4>=d?2:y<4?3:y+4>=d?4:z<2?5:0;}
struct Plan {
    unsigned d;std::vector<unsigned>rank;std::vector<std::array<int,3>>inputs;
    Plan(const Snapshot&s,bool full=false):d(s.d),rank(d*d){
        if(full){int xyz[480];int nt=ng_axis_taps(xyz);require(nt>0&&nt<160);for(int k=0;k<nt;k++)inputs.push_back({xyz[3*k],xyz[3*k+1],xyz[3*k+2]});}
        else for(auto&t:taps)inputs.push_back({t[0],t[1],t[2]});
        std::vector<const uint8_t*>p;std::vector<size_t>n;Bytes empty;
        for(auto&m:s.models){p.push_back(m.data());n.push_back(m.size());}
        if(p.empty()){empty.resize(4096);size_t used=0;require(ng_bank_empty_model(empty.data(),empty.size(),&used)==0);empty.resize(used);p.push_back(empty.data());n.push_back(empty.size());}
        std::vector<uint32_t>rows(d*d),ends(d*d);uint32_t groups=0;
        require(ng_bank_route_order(p.data(),n.data(),p.size(),rows.data(),ends.data(),&groups)==0);
        unsigned begin=0;for(unsigned g=0;g<groups;g++){for(unsigned j=begin;j<ends[g];j++)rank.at(rows[j])=g;begin=ends[g];}require(begin==d*d);
    }
    bool neighbor(size_t i,unsigned k,size_t&at)const{
        int x=i%d,y=i/d%d,z=i/(size_t(d)*d),xx=x+inputs[k][0],yy=y+inputs[k][1],zz=z+inputs[k][2];
        if(xx<0||yy<0||zz<0||xx>=int(d)||yy>=int(d)||zz>=int(d))return false;
        unsigned a=rank[zz*d+yy],b=rank[z*d+y];
        if(a>b||(a==b&&xx>=x))return false; // same wave: only earlier x, never lane order
        at=(size_t(zz)*d+yy)*d+xx;return true;
    }
};
using Weights=std::array<std::array<int32_t,160>,6>;
// Exact narrower accumulation is permitted only after a coefficient-bound
// proof for every partial sum. Arbitrary frame coefficients retain int64.
static void edge_add32(int32_t*dst,const uint8_t*src,int count,int32_t weight){for(int x=0;x<count;x++)dst[x]+=weight*(int(src[x])-128);}
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) static void edge_add32_avx2(int32_t*dst,const uint8_t*src,int count,int32_t weight){for(int x=0;x<count;x++)dst[x]+=weight*(int(src[x])-128);}
#endif
thread_local bool expanded_menu=false;
thread_local bool integrated_prediction=false; // speed experiment remains opt-in
thread_local bool lookup_routing=false;
thread_local unsigned search_strategy=0;
thread_local bool compact_frame=false;
thread_local bool edge_context=false;
thread_local unsigned context_partition=1; // retain edge/interior in combined mode
thread_local bool compact_selection=false;
thread_local bool wave_frame=false;
thread_local bool decode_profile_enabled=false;
thread_local double decode_phases[3]={};
thread_local double wave_edge_seconds=0,wave_total_seconds=0;
struct Trial {Snapshot s;Bytes source;std::array<Weights,3>weights{};bool fitted[3]={};std::array<uint64_t,10>policy_counts{};Bytes optional_frame;};
unsigned prediction_mode(unsigned mode){return mode>=24?0:mode>=22?mode-10:mode>=16?(mode<=18?mode-7:mode==21?9:0):mode>=13?mode-7:mode>=10?5:mode<=2?0:mode==3?1:mode==4?2:(mode==5||mode==7)?3:4;}
unsigned entropy_mode(unsigned mode){return mode==19||mode==21?3:mode==20?4:mode==1||mode==7||mode==8||mode==11?1:mode==2||mode==9||mode==12?2:0;}
unsigned entropy_groups(unsigned em){return em>=3?4:em==2?6:em==1?2:1;}
// Infrequently switch fixed rANS distributions using preceding decoded symbols.
// No table updates, side labels or tree traversal are needed at decode.
struct RecentEntropy {
    unsigned group=0,count=0,magnitude=0,negative=0;
    void accept(unsigned symbol,unsigned em){if(em<3)return;
        magnitude+=(symbol+1)/2;negative+=symbol&1;
        if(++count==64){group=em==3?(magnitude<64?0:magnitude<256?1:magnitude<1024?2:3):(negative<24?0:negative>40?2:1);count=magnitude=negative=0;}
    }
};
void fit(Trial&t,unsigned count){
    unsigned slot=count==0?2:count==12; if(t.fitted[slot])return;const auto&s=t.s;Plan plan(s,slot==2);if(!count)count=plan.inputs.size();unsigned n=count+1;
    std::array<std::vector<double>,6>gram,rhs;for(unsigned g=0;g<6;g++){gram[g].resize(n*n);rhs[g].resize(n);}
    uint64_t counts[6]={};
    for(size_t i=0;i<t.source.size();i++){
        unsigned g=face(i,s.d);if(!g||s.menu[s.selectors[geometry_trials::block(i,s.d)]].predictor==choice_frame::zero_predictor)continue;
        double f[160]={1};for(unsigned k=0;k<count;k++){size_t at;if(plan.neighbor(i,k,at))f[k+1]=int(t.source[at])-128;}
        double target=int(t.source[i])-128;counts[g]++;
        for(unsigned a=0;a<n;a++)if(f[a]){rhs[g][a]+=f[a]*target;for(unsigned b=0;b<=a;b++)gram[g][a*n+b]+=f[a]*f[b];}
    }
    for(unsigned g=1;g<6;g++)if(counts[g]){
        for(unsigned a=0;a<n;a++)for(unsigned b=0;b<a;b++)gram[g][b*n+a]=gram[g][a*n+b];
        double trace=0;for(unsigned k=1;k<n;k++)trace+=gram[g][k*n+k];
        gram[g][0]+=1e-6*counts[g]+1e-9;
        for(unsigned k=1;k<n;k++)gram[g][k*n+k]+=1e-6*trace/count+1e-9;
        require(ng_axis_solve(gram[g].data(),rhs[g].data(),n,t.weights[slot][g].data())==0);
    }
    t.fitted[slot]=true;
}
uint16_t edge_prediction(const uint8_t*src,size_t i,const Plan&plan,unsigned pm,const Weights&w){
    if(pm<=2){
        for(unsigned k=0;k<3;k++){size_t a,b;if(plan.neighbor(i,k,a)){
            int v=src[a];if(pm==2&&plan.neighbor(i,k+3,b))v=std::clamp(2*v-int(src[b]),0,255);
            return uint16_t(v*256);
        }}return 128*256;
    }
    unsigned count=pm==5?plan.inputs.size():pm==3?6:12,g=face(i,plan.d);int64_t q=128*256+w[g][0];
    for(unsigned k=0;k<count;k++){size_t at;if(plan.neighbor(i,k,at))q+=int64_t(w[g][k+1])*(int(src[at])-128);}
    // Conventional bounded prediction. Serialize rounded Q8 coefficients.
    return uint16_t(std::clamp<int64_t>(q,0,255*256));
}
// One shared encoder/decoder reconstruction walk, retaining original interior
// models and palette. All model histories see the actually selected predictor.
struct FastPrediction {
    const Snapshot&s;std::vector<std::unique_ptr<void,decltype(&ng_causal_destroy)>>states;
    explicit FastPrediction(const Snapshot&snapshot):s(snapshot){for(auto&m:s.models){void*p=ng_causal_create(m.data(),m.size());require(p);states.emplace_back(p,ng_causal_destroy);}}
    uint16_t predict(size_t i,const uint8_t*data){
        uint16_t q[4]={};uint8_t cls;for(unsigned m=0;m<states.size();m++)require(ng_causal_prediction(states[m].get(),i,q+m,&cls)==0);
        auto&c=s.menu[s.selectors[geometry_trials::block(i,s.d)]];
        if(c.predictor==choice_frame::zero_predictor)return 0;
        unsigned x=i%s.d,y=i/s.d%s.d,z=i/(size_t(s.d)*s.d);
        auto base=ng_directional_base(data,i,c.predictor,x,y,z,s.d);
        return c.model==choice_frame::no_model?uint16_t(base*256):uint16_t(q[c.model]+(c.predictor?int64_t(s.mode?c.weights[z==0]:256)*base:0));
    }
    void accept(size_t i,uint8_t value,uint16_t pred){for(auto&state:states)require(ng_causal_accept(state.get(),i,value,geometry_trials::predicted(pred))==0);}
};
// Reuse the production raster reconstruction and SIMD row prediction kernels.
// Only entropy routing is supplied here; do not reimplement NG arithmetic.
void native_prediction_walk(const Snapshot&s,uint8_t*out,ng_prediction_reader reader,void*ctx,ng_route_context*wave=nullptr){
    Snapshot adapted=s;if(wave&&adapted.models.empty()){Bytes p(4096);size_t n=0;require(ng_bank_empty_model(p.data(),p.size(),&n)==0);p.resize(n);adapted.models.push_back(std::move(p));}
    FastPrediction owner(adapted);void*models[48]={};int32_t weights[96]={};bool assigned[48]={};
    for(unsigned m=0;m<owner.states.size();m++)for(unsigned b=0;b<12;b++)models[12*m+b]=owner.states[m].get();
    for(auto&c:s.menu)if(c.model!=choice_frame::no_model){unsigned id=12*c.model+c.predictor;
        require(id<48);for(unsigned j=0;j<2;j++){require(!assigned[id]||weights[2*id+j]==c.weights[j]);weights[2*id+j]=c.weights[j];}assigned[id]=true;}
    choice_frame::normalize_unused_weights(owner.states.size(),assigned,weights);
    Bytes canonical(s.selectors.size());for(size_t i=0;i<canonical.size();i++){auto&c=s.menu[s.selectors[i]];
        canonical[i]=c.predictor==choice_frame::zero_predictor?0:c.model==choice_frame::no_model?c.predictor:12+12*c.model+c.predictor;}
    if(wave)require(ng_bank_route_wave(models,std::max(1u,unsigned(owner.states.size()))*12,s.mode,weights,canonical.data(),out,wave)==0);
    else require(ng_bank_prediction_decode(models,std::max(1u,unsigned(owner.states.size()))*12,s.mode,weights,canonical.data(),out,s.thresholds.data(),s.thresholds.size()+1,reader,ctx)==0);
}
std::vector<size_t>wave_positions(const Snapshot&s){
    std::vector<const uint8_t*>p;std::vector<size_t>n;Bytes empty;
    for(auto&m:s.models){p.push_back(m.data());n.push_back(m.size());}
    if(p.empty()){empty.resize(4096);size_t used=0;require(ng_bank_empty_model(empty.data(),empty.size(),&used)==0);empty.resize(used);p.push_back(empty.data());n.push_back(empty.size());}
    std::vector<uint32_t>rows(s.d*s.d),ends(s.d*s.d);uint32_t groups=0;require(ng_bank_route_order(p.data(),n.data(),p.size(),rows.data(),ends.data(),&groups)==0);
    std::vector<size_t>order;order.reserve(N);unsigned begin=0;
    for(unsigned g=0;g<groups;g++){for(unsigned x=0;x<s.d;x++)for(unsigned j=begin;j<ends[g];j++)order.push_back(size_t(rows[j])*s.d+x);begin=ends[g];}
    require(order.size()==N);return order;
}
struct PredictionCapture {
    const Snapshot&s;std::vector<uint16_t>&q;
    static int get(void*p,size_t i,unsigned cls,uint16_t q){auto&r=*static_cast<PredictionCapture*>(p);require(cls==r.s.classes[i]);r.q[i]=q;return r.s.symbols[i];}
};
// Three-bin and seven-bin conditioning share the same native prediction path.
unsigned prediction_contexts(unsigned mode){return mode==32||mode==34?5:mode==33||mode==35?9:mode==24||mode==26?3:mode==27||mode==29?7:1;}
bool zero_gate(unsigned mode){return mode==25||mode==26||mode==29||mode==34||mode==35;}
unsigned prediction_context(uint16_t q,unsigned mode){
    unsigned p=geometry_trials::predicted(q);
    if(mode==32||mode==34)return p==0?0:p<16?1:p<240?2:p<255?3:4;
    if(mode==33||mode==35)return p==0?0:p<4?1:p<16?2:p<64?3:p<192?4:p<240?5:p<252?6:p<255?7:8;
    if(mode==24||mode==26)return p<16?0:p>=240?2:1;
    if(mode==27||mode==29)return p==0?0:p<4?1:p<16?2:p<240?3:p<252?4:p<255?5:6;
    return 0;
}
unsigned conditioned_group(const Snapshot&s,size_t i,unsigned mode,unsigned cls,uint16_t q,unsigned em=0){
    unsigned bank=s.menu[s.selectors[geometry_trials::block(i,s.d)]].entropy*entropy_groups(em);
    bank+=em==1?bool(face(i,s.d)):em==2?face(i,s.d):0;
    return (bank*prediction_contexts(mode)+prediction_context(q,mode))*(s.thresholds.size()+1)+cls;
}
void walk(const Snapshot&s,const uint8_t*source,Bytes&symbols,uint8_t*out,unsigned pm,const Weights&w,std::vector<uint16_t>*predictions=nullptr){
    Plan plan(s,pm==5);FastPrediction predictor(s);
    std::vector<int16_t> errors(pm>=6&&pm<=8?size_t(s.d)*s.d*s.d:0);
    int bias_q8=0,bias_sum=0,bias_count=0,block_bias=0;
    for(size_t i=0;i<size_t(s.d)*s.d*s.d;i++){
        auto&c=s.menu[s.selectors[geometry_trials::block(i,s.d)]];bool zero=c.predictor==choice_frame::zero_predictor;
        unsigned x=i%s.d;uint16_t pred;
        const uint8_t*data=source?source:out;
        pred=predictor.predict(i,data);
        if(!zero&&pm&&pm<=5&&face(i,s.d))pred=edge_prediction(data,i,plan,pm,w);
        uint16_t uncorrected=pred;
        if(pm>=6&&pm<=8&&!zero){
            int sum=0,count=0;for(unsigned k: {1u,2u,4u,5u,8u}){size_t at;if(plan.neighbor(i,k,at)){sum+=errors[at];count++;}}
            int correction=count?sum/count:0;
            if(pm==6)correction/=2;
            if(pm!=8)correction=std::clamp(correction,-16,16);
            pred=uint16_t(pred+correction*256);
        }
        if(pm>=9){
            if(x==0&&(pm==9||pm==12))bias_q8=0;
            int correction=(pm==9||pm==12)?bias_q8/256:block_bias;
            if(!zero&&(pm<12||c.model!=choice_frame::no_model))pred=uint16_t(pred+std::clamp(correction,-8,8)*256);
        }
        uint8_t value;
        if(source){value=source[i];symbols[i]=zero?0:ng_prediction_symbol(value,pred);}
        else value=zero?0:geometry_trials::reconstruct(symbols[i],pred);
        if(out)out[i]=value;
        if(predictions)(*predictions)[i]=pred;
        if(pm>=6&&pm<=8)errors[i]=zero?0:int8_t(uint8_t(value-geometry_trials::predicted(uncorrected)));
        if(pm>=9){
            if(!zero&&(pm<12||c.model!=choice_frame::no_model)){int e=int8_t(uint8_t(value-geometry_trials::predicted(uncorrected)));e=std::clamp(e,-32,32);
                bias_q8+=(e*256-bias_q8)/16;bias_sum+=e;bias_count++;}
            unsigned length=pm==10?32:128;
            if((i+1)%length==0){block_bias=bias_count?bias_sum/bias_count:0;bias_sum=bias_count=0;}
        }
        predictor.accept(i,value,pred);
    }
}
unsigned group(const Snapshot&s,size_t i,unsigned em,unsigned cls,unsigned recent=0){
    unsigned f=face(i,s.d),g=em>=3?recent:em==1?bool(f):em==2?f:0;
    unsigned bank=s.menu[s.selectors[geometry_trials::block(i,s.d)]].entropy;
    return ((bank*entropy_groups(em)+g)*(s.thresholds.size()+1))+cls;
}
Trial*create(const uint8_t*src,unsigned d,unsigned preset){
    require(preset==1||preset==3);GeometryScope geom(d,d,d);choice_grid::Scope grid(4);
    std::vector<Snapshot>seen;hybrid_geometry::Capture capture(seen,d);Bytes frame(3*N+131072);size_t n=0;
    require(ng_codec_encode(src,d,preset,frame.data(),frame.size(),&n)==0);frame.resize(n);
    for(auto&s:seen)if(s.frame==frame){if(s.symbols.empty()){s.symbols.resize(N);s.classes.resize(N);}return new Trial{std::move(s),Bytes(src,src+N)};}
    throw std::runtime_error("edge preset capture missing");
}
void snapshot_header(Bytes&h,const Snapshot&s,unsigned mode,unsigned pm,unsigned em){
    unsigned K=s.thresholds.size()+1;packed_bank::var(h,s.d);h.push_back(mode);h.push_back(pm);h.push_back(em);h.push_back(s.mode);
    h.push_back(s.models.size());h.push_back(s.menu.size());h.push_back(K);
    for(auto&m:s.models){packed_bank::var(h,m.size());packed_bank::append(h,m);}
    for(auto&c:s.menu){h.push_back(c.predictor);h.push_back(c.model);h.push_back(c.entropy);for(auto v:c.weights)compact_contexts::signed_var(h,v);}
    bool uniform=std::all_of(s.selectors.begin(),s.selectors.end(),[&](auto c){return c==s.selectors[0];});
    h.push_back(uniform?s.selectors[0]:255);if(!uniform)packed_bank::append(h,choice_frame::pack_ids(s.selectors,s.menu.size()));
    unsigned prev=0;for(auto v:s.thresholds){require(v>=prev);packed_bank::var(h,v-prev);prev=v;}
}
struct JointMap {
    bool gate=false;unsigned groups=0,banks=0;std::vector<uint16_t>athr,pthr,ids;std::array<unsigned,2560>ai{};std::array<unsigned,256>pi{};
    void prepare(){for(unsigned a=0;a<ai.size();a++)ai[a]=std::upper_bound(athr.begin(),athr.end(),a)-athr.begin();for(unsigned p=0;p<256;p++)pi[p]=std::upper_bound(pthr.begin(),pthr.end(),p)-pthr.begin();}
    unsigned group(unsigned bank,unsigned a,uint16_t q)const{require(bank<banks&&a<ai.size());return ids[(bank*(athr.size()+1)+ai[a])*(pthr.size()+1)+pi[geometry_trials::predicted(q)]];}
    void write(Bytes&h)const{packed_bank::var(h,gate);packed_bank::var(h,groups);packed_bank::var(h,banks);packed_bank::var(h,athr.size());for(auto t:athr)packed_bank::var(h,t);packed_bank::var(h,pthr.size());for(auto t:pthr)packed_bank::var(h,t);for(auto id:ids)packed_bank::var(h,id);}
    void read(packed_bank::Reader&h){gate=h.var()!=0;groups=h.var();banks=h.var();require(groups&&groups<=2048&&banks&&banks<=16);unsigned n=h.var();require(n<128);for(unsigned k=0;k<n;k++)athr.push_back(h.var());n=h.var();require(n<256);for(unsigned k=0;k<n;k++)pthr.push_back(h.var());require(std::is_sorted(athr.begin(),athr.end())&&std::is_sorted(pthr.begin(),pthr.end()));require(athr.empty()||athr.back()<2560);require(pthr.empty()||pthr.back()<256);ids.resize(banks*(athr.size()+1)*(pthr.size()+1));for(auto&id:ids){auto v=h.var();require(v<groups);id=v;}prepare();}
};
Bytes encode(Trial&t,unsigned mode){
    require(mode<=31);const auto&s=t.s;GeometryScope geom(s.d,s.d,s.d);choice_grid::Scope grid(4);
    if(mode>=30&&!t.optional_frame.empty())return mode==31&&t.optional_frame.size()>=s.frame.size()?s.frame:t.optional_frame;
    unsigned pm=edge_context&&mode>=24?5:prediction_mode(mode),em=edge_context&&mode>=24?context_partition:entropy_mode(mode);Weights w{};
    require(!edge_context||compact_frame);
    if(pm>=3&&pm<=5){fit(t,pm==5?0:pm==3?6:12);w=t.weights[pm==5?2:pm==4];}
    std::vector<uint16_t>predictions(mode>=24?N:0);
    Bytes symbols(N);
    if(mode>=24&&!pm&&integrated_prediction){symbols=s.symbols;Bytes reconstructed(N);PredictionCapture capture{s,predictions};
        native_prediction_walk(s,reconstructed.data(),PredictionCapture::get,&capture);require(reconstructed==t.source);}
    else walk(s,t.source.data(),symbols,nullptr,pm,w,mode>=24?&predictions:nullptr);
    if(!pm)require(symbols==s.symbols); // exact selected-preset prediction control
    Bytes classes;
    if(!pm)classes=s.classes;
    else{std::vector<uint16_t>activity(N);require(ng_activity_symbol_values(symbols.data(),activity.data())==0);classes=activity_classify(activity,s.thresholds);}
    if(!pm)require(classes==s.classes);
    unsigned B=hybrid_geometry::banks(s),K=s.thresholds.size()+1,E=entropy_groups(em),G=B*K*(mode>=24?E*prediction_contexts(mode)*(zero_gate(mode)?2:1):E);
    Bytes policies,selected_headers,selected_bodies;std::vector<Bytes>selected_streams;
    if(mode>=30){
        // Fit each optional entropy representation to the same residuals.
        // Actual packed stream sizes + compressed local metadata are the cost;
        // models, predictions and residuals are computed only once.
        constexpr unsigned menu[10]={28,24,27,25,26,29,32,33,34,35};
        std::vector<std::vector<size_t>>positions(B*E*K);
        for(size_t i=0;i<N;i++)positions[group(s,i,em,classes[i])].push_back(i);
        t.policy_counts.fill(0);
        for(auto&indices:positions){
            size_t best=SIZE_MAX;unsigned winner=0;Bytes best_header,best_body;std::vector<Bytes>best_streams;
            for(unsigned c=0;c<(expanded_menu?10u:6u);c++){
                unsigned candidate=menu[c],parts=prediction_contexts(candidate)*(zero_gate(candidate)?2:1);
                if(indices.empty()&&c)continue;
                std::vector<Bytes>v(parts);
                for(size_t i:indices){unsigned g=prediction_context(predictions[i],candidate),sym=symbols[i];
                    if(zero_gate(candidate)){bool zero=t.source[i]==0;v[2*g].push_back(zero);if(!zero){unsigned zs=ng_prediction_symbol(0,predictions[i]);require(sym!=zs);v[2*g+1].push_back(sym-(sym>zs));}}
                    else v[g].push_back(sym);
                }
                Bytes ch,cb;
                if(compact_frame&&compact_selection)compact_contexts::write(ch,cb,v);
                else for(auto&part:v)small_bank::stream_header(ch,cb,pack_stream(part,true),part.size());
                Bytes cost_header=ch;cost_header.push_back(candidate);
                size_t cost=cb.size()+packed_bank::compress(cost_header).size();
                if(cost<best){best=cost;winner=c;best_header=std::move(ch);best_body=std::move(cb);if(compact_frame)best_streams=std::move(v);}
            }
            policies.push_back(menu[winner]);t.policy_counts[winner]++;
            packed_bank::append(selected_headers,best_header);packed_bank::append(selected_bodies,best_body);
            if(compact_frame)for(auto&v:best_streams)selected_streams.push_back(std::move(v));
        }
    }
    RecentEntropy recent;std::vector<Bytes>streams(G);for(size_t i=0;i<N;i++){
        if(mode>=30)break;
        if(mode>=24){unsigned g=conditioned_group(s,i,mode,classes[i],predictions[i],em);
            if(zero_gate(mode)){bool zero=t.source[i]==0;streams[2*g].push_back(zero);
                if(!zero){unsigned zs=ng_prediction_symbol(0,predictions[i]),sym=symbols[i];require(sym!=zs);streams[2*g+1].push_back(sym-(sym>zs));}}
            else streams[g].push_back(symbols[i]);
        }else{streams[group(s,i,em,classes[i],recent.group)].push_back(symbols[i]);recent.accept(symbols[i],em);}
    }
    if(wave_frame){
        require(compact_frame&&pm<=5&&em<=2);std::vector<unsigned>offsets;
        if(mode>=30){unsigned size=0;for(auto policy:policies){offsets.push_back(size);size+=prediction_contexts(policy)*(zero_gate(policy)?2:1);}selected_streams.assign(size,{});}
        else for(auto&v:streams)v.clear();
        auto&dest=mode>=30?selected_streams:streams;
        for(size_t i:wave_positions(s)){
            unsigned policy=mode,g;
            if(mode>=30){unsigned bc=group(s,i,em,classes[i]);policy=policies[bc];g=offsets[bc]+prediction_context(predictions[i],policy)*(zero_gate(policy)?2:1);}
            else if(mode>=24){g=conditioned_group(s,i,mode,classes[i],predictions[i],em);if(zero_gate(policy))g*=2;}
            else g=group(s,i,em,classes[i]);
            if(mode>=24&&zero_gate(policy)){bool zero=t.source[i]==0;dest[g].push_back(zero);if(!zero){unsigned zs=ng_prediction_symbol(0,predictions[i]);dest[g+1].push_back(symbols[i]-(symbols[i]>zs));}}
            else dest[g].push_back(symbols[i]);
        }
    }
    if(compact_frame){
        Bytes h,body;snapshot_header(h,s,mode,pm,em);
        if(pm>=3&&pm<=5){unsigned count=pm==5?Plan(s,true).inputs.size():pm==3?6:12;packed_bank::var(h,count);
            for(unsigned g=1;g<6;g++)for(unsigned k=0;k<=count;k++)compact_contexts::signed_var(h,int64_t(w[g][k])-(g>1?w[g-1][k]:0));}
        if(mode>=30)packed_bank::append(h,policies);
        compact_contexts::write(h,body,mode>=30?selected_streams:streams);
        require(h.size()<=1048576);auto z=packed_bank::compress(h);Bytes frame={'E','G','B',uint8_t(wave_frame?'3':'2')};put32(frame,h.size());put32(frame,z.size());packed_bank::append(frame,z);packed_bank::append(frame,body);
        if(mode>=30)t.optional_frame=frame;
        return mode==31&&frame.size()>=s.frame.size()?s.frame:frame;
    }
    Bytes h,body;put32(h,s.d);put32(h,mode);put32(h,s.mode);put32(h,s.models.size());put32(h,s.menu.size());put32(h,K);
    for(auto&m:s.models){put32(h,m.size());packed_bank::append(h,m);}
    for(auto&c:s.menu){put32(h,c.predictor);put32(h,c.model);put32(h,c.entropy);for(auto v:c.weights)put32(h,uint32_t(v));}
    small_bank::stream_header(h,body,pack_stream(s.selectors,true),s.selectors.size());
    for(auto v:s.thresholds)packed_bank::put16(h,v);
    if(pm>=3&&pm<=5){unsigned count=pm==5?Plan(s,true).inputs.size():pm==3?6:12;for(unsigned g=1;g<6;g++)for(unsigned k=0;k<=count;k++)put32(h,uint32_t(w[g][k]));}
    if(mode>=30){packed_bank::append(h,policies);packed_bank::append(h,selected_headers);packed_bank::append(body,selected_bodies);}
    else for(auto&v:streams)small_bank::stream_header(h,body,pack_stream(v,true),v.size());
    auto z=packed_bank::compress(h);Bytes frame={'E','G','B','1'};put32(frame,h.size());put32(frame,z.size());packed_bank::append(frame,z);packed_bank::append(frame,body);
    // The captured Fast frame is already available: this adds no second fit.
    // Expose unguarded mode30 separately so its cost model can be evaluated.
    if(mode>=30)t.optional_frame=frame;
    return mode==31&&frame.size()>=s.frame.size()?s.frame:frame;
}
struct Read {const Snapshot&s;unsigned em;std::vector<Bytes>&streams;std::vector<size_t>pos;RecentEntropy recent;
    static int get(void*p,size_t i,unsigned cls){auto&r=*static_cast<Read*>(p);unsigned g=group(r.s,i,r.em,cls,r.recent.group);if(g>=r.streams.size()||r.pos[g]>=r.streams[g].size())return -1;unsigned v=r.streams[g][r.pos[g]++];r.recent.accept(v,r.em);return v;}
};
// Activity decoding invokes this in causal raster order. Reconstruct directly
// here so the next prediction uses decoded values, never an encoder-side map.
static thread_local unsigned joint_decode_variant=0;
struct ConditionedRead {
    const Snapshot&s;unsigned mode;std::vector<Bytes>&streams;std::vector<size_t>pos;uint8_t*out;std::unique_ptr<FastPrediction> predictor;Bytes policies;std::vector<unsigned>offsets;
    struct Cursor{const uint8_t*next;const uint8_t*end;};std::vector<Cursor>cursors;
    std::vector<std::array<uint16_t,256>>routes;Bytes gated;std::vector<unsigned>bank_classes;unsigned K;
    unsigned pm=0,em=0;Weights weights{};std::unique_ptr<Plan>plan;
    const JointMap*joint=nullptr;
    std::vector<uint16_t>joint_packed;
    std::vector<unsigned>joint_ids;
    std::vector<std::array<unsigned,32>>joint_rows;
    std::array<unsigned,2560>joint_activity{};
    Bytes joint_data;
    std::vector<unsigned>joint_positions,joint_ends;
    struct EdgeRow{size_t begin=SIZE_MAX,end=0;std::vector<int64_t>pre;std::vector<int32_t>pre32;std::vector<unsigned>face,taps;};EdgeRow row;std::array<EdgeRow,32>wave_rows;bool narrow_edges=false;
    ConditionedRead(const Snapshot&snapshot,unsigned m,std::vector<Bytes>&v,uint8_t*dst):s(snapshot),mode(m),streams(v),pos(v.size()),out(dst),K(snapshot.thresholds.size()+1){if(!integrated_prediction)predictor=std::make_unique<FastPrediction>(snapshot);}
    void prepare(){
        narrow_edges=true;for(unsigned f=1;f<6;f++){uint64_t bound=uint64_t(std::abs(int64_t(128*256)+weights[f][0]));for(unsigned k=1;k<160;k++)bound+=128*uint64_t(std::abs(int64_t(weights[f][k])));if(bound>INT32_MAX)narrow_edges=false;}
        unsigned B=hybrid_geometry::banks(s)*entropy_groups(em);routes.resize(B*K);gated.resize(B*K);
        for(unsigned bc=0;bc<B*K;bc++){unsigned policy=mode>=30?policies[bc]:mode;gated[bc]=mode>=24&&zero_gate(policy);
            for(unsigned p=0;p<256;p++){unsigned g=prediction_context(uint16_t(p*256),policy);
                routes[bc][p]=mode<24?bc:mode>=30?offsets[bc]+g*(gated[bc]?2:1):((bc/K*prediction_contexts(policy)+g)*K+bc%K)*(gated[bc]?2:1);}}
        bank_classes.reserve(s.selectors.size());for(auto c:s.selectors)bank_classes.push_back(s.menu[c].entropy*entropy_groups(em)*K);
        if(!bank_classes.empty()&&std::all_of(bank_classes.begin(),bank_classes.end(),[&](unsigned c){return c==bank_classes[0];}))bank_classes.resize(1);
        for(auto&v:streams)cursors.push_back({v.data(),v.empty()?v.data():v.data()+v.size()});
        if(joint&&joint_decode_variant){
            unsigned P=joint->pthr.size()+1;
            for(unsigned a=0;a<2560;a++)joint_activity[a]=joint->ai[a]*P;
            if(joint_decode_variant==2||joint_decode_variant==4){
                // Decode-only expansion: prediction quantization, stream ID and
                // gate flag become one lookup. Wire representation is unchanged.
                unsigned count=joint->banks*(joint->athr.size()+1)*256;
                joint_packed.resize(count+1); // one safe halfword for SIMD gathers
                for(unsigned row=0;row<count/256;row++)for(unsigned p=0;p<256;p++)
                    joint_packed[row*256+p]=uint16_t((joint->ids[row*P+joint->pi[p]]<<1)|joint->gate);
            }
            if(joint_decode_variant>=3){
                joint_ids.assign(joint->ids.begin(),joint->ids.end());
                size_t total=4;for(auto&v:streams)total+=v.size();require(total<=UINT32_MAX);joint_data.reserve(total);
                for(auto&v:streams){joint_positions.push_back(joint_data.size());joint_data.insert(joint_data.end(),v.begin(),v.end());joint_ends.push_back(joint_data.size());}
                joint_data.resize(joint_data.size()+4);
            }
        }
    }
    unsigned take(unsigned g){require(g<streams.size()&&pos[g]<streams[g].size());return streams[g][pos[g]++];}
    static int symbol(void*p,size_t i,unsigned cls,uint16_t q){auto&r=*static_cast<ConditionedRead*>(p);
        if(r.mode<24)return r.take(group(r.s,i,r.em,cls));
        unsigned policy=r.mode,g,sym;
        if(r.mode>=30){unsigned bc=group(r.s,i,r.em,cls);require(bc<r.policies.size());policy=r.policies[bc];g=r.offsets[bc]+prediction_context(q,policy)*(zero_gate(policy)?2:1);}
        else{g=conditioned_group(r.s,i,r.mode,cls,q,r.em);if(zero_gate(policy))g*=2;}
        if(zero_gate(policy)){unsigned flag=r.take(g),zs=ng_prediction_symbol(0,q);require(flag<=1);
            if(flag)sym=zs;else{unsigned rank=r.take(g+1);require(rank<255);sym=rank+(rank>=zs);}}
        else sym=r.take(g);
        return sym;
    }
    // Prepare contributions from completed rows once. Keep only same-row
    // causal taps in the serial recurrence; exact int64 accumulation/clamp.
    void prepare_edge_row(EdgeRow&r,size_t i,bool wave=false){
        unsigned d=s.d,y=i/d%d,z=i/(size_t(d)*d);r.begin=i;r.end=i+d;
        auto&edge_pre=r.pre;auto&edge_face=r.face;auto&row_taps=r.taps;
        edge_pre.assign(d,0);edge_face.resize(d);row_taps.clear();
        if(narrow_edges)r.pre32.assign(d,0);
        auto add32=edge_add32;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(__builtin_cpu_supports("avx2"))add32=edge_add32_avx2;
#endif
        unsigned interior=y<4?3:y+4>=d?4:z<2?5:0;
        for(unsigned x=0;x<d;x++){unsigned f=x<4?1:x+4>=d?2:interior;edge_face[x]=f;if(f){edge_pre[x]=128*256+int64_t(weights[f][0]);if(narrow_edges)r.pre32[x]=edge_pre[x];}}
        unsigned count=pm==5?plan->inputs.size():pm==3?6:12;
        for(unsigned k=0;k<count;k++){
            auto t=plan->inputs[k];int yy=int(y)+t[1],zz=int(z)+t[2];
            if(yy<0||zz<0||yy>=int(d)||zz>=int(d))continue;
            unsigned a=plan->rank[zz*d+yy],b=plan->rank[z*d+y];if(a>b||(a==b&&t[0]>=0))continue;
            if((!t[1]&&!t[2])||(wave&&a==b)){row_taps.push_back(k);continue;}
            int lo=std::max(0,-t[0]),hi=std::min(int(d),int(d)-t[0]);
            const uint8_t*values=out+(size_t(zz)*d+yy)*d;
            // Face weights are constant on each of the three x intervals.
            for(unsigned segment=0;segment<3;segment++){
                int begin=std::max(lo,segment==0?0:segment==1?4:int(d)-4),end=std::min(hi,segment==0?4:segment==1?int(d)-4:int(d));
                if(begin>=end)continue;unsigned f=edge_face[begin];if(!f)continue;
                int64_t w=weights[f][k+1];if(!w)continue;
                if(narrow_edges)add32(r.pre32.data()+begin,values+begin+t[0],end-begin,int32_t(w));
                else for(int x=begin;x<end;x++)edge_pre[x]+=w*(int(values[x+t[0]])-128);
            }
        }
        if(narrow_edges)for(unsigned x=0;x<d;x++)edge_pre[x]=r.pre32[x];
    }
    uint16_t edge_value(const EdgeRow&r,unsigned x,uint16_t q){
        size_t i=r.begin+x;unsigned f=r.face[x];if(!f)return q;
        if(s.menu[s.selectors[geometry_trials::block(i,s.d)]].predictor==choice_frame::zero_predictor)return q;
        if(pm<3)return edge_prediction(out,i,*plan,pm,weights);
        int64_t value=r.pre[x];for(unsigned k:r.taps){auto t=plan->inputs[k];int xx=int(x)+t[0];if(xx>=0&&xx<int(s.d)){
            ptrdiff_t at=ptrdiff_t(r.begin)+xx+ptrdiff_t(t[1])*s.d+ptrdiff_t(t[2])*s.d*s.d;
            value+=int64_t(weights[f][k+1])*(int(out[at])-128);}}
        return uint16_t(std::clamp<int64_t>(value,0,255*256));
    }
    static uint16_t transform(void*p,size_t i,uint16_t q){auto&r=*static_cast<ConditionedRead*>(p);
        if(i>=r.row.end)r.prepare_edge_row(r.row,i);
        return r.edge_value(r.row,i-r.row.begin,q);
    }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    // Consume equal-context lanes in their original order. Conflict ranks give
    // each lane its own symbol; only the last lane advances the shared cursor.
    __attribute__((target("avx512f,avx512bw,avx512cd,avx512vl,avx512vpopcntdq")))
    static __m512i joint_take16(ConditionedRead&r,__m512i groups,__mmask16 active){
        auto unique=_mm512_add_epi32(_mm512_set1_epi32(r.joint_positions.size()),_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
        auto g=_mm512_mask_mov_epi32(unique,active,groups),conflict=_mm512_conflict_epi32(g);
        auto rank=_mm512_popcnt_epi32(conflict);
        auto pos=_mm512_mask_i32gather_epi32(_mm512_setzero_si512(),active,g,r.joint_positions.data(),4);
        auto end=_mm512_mask_i32gather_epi32(_mm512_setzero_si512(),active,g,r.joint_ends.data(),4);
        auto at=_mm512_add_epi32(pos,rank);
        require((_mm512_cmp_epu32_mask(at,end,_MM_CMPINT_GE)&active)==0);
        auto sym=_mm512_and_si512(_mm512_mask_i32gather_epi32(_mm512_setzero_si512(),active,at,r.joint_data.data(),1),_mm512_set1_epi32(255));
        __mmask16 last=active&~unsigned(_mm512_reduce_or_epi32(conflict));
        _mm512_mask_i32scatter_epi32(r.joint_positions.data(),last,g,_mm512_add_epi32(at,_mm512_set1_epi32(1)),4);
        return sym;
    }
    __attribute__((target("avx512f,avx512bw,avx512cd,avx512vl,avx512vpopcntdq")))
    static void joint_vector_symbols(ConditionedRead&r,const unsigned*rows,const uint16_t*q,unsigned lanes,unsigned*symbols,bool gate,bool packed){
        for(unsigned h=0;h<lanes;h+=16){
            __mmask16 active=(1u<<std::min(16u,lanes-h))-1;
            auto qq=_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(q+h)));
            auto pred=_mm512_and_si512(_mm512_srli_epi32(_mm512_add_epi32(qq,_mm512_set1_epi32(128)),8),_mm512_set1_epi32(255));
            auto row=_mm512_loadu_si512(rows+h);__m512i g;
            if(packed)g=_mm512_srli_epi32(_mm512_and_si512(_mm512_mask_i32gather_epi32(_mm512_setzero_si512(),active,_mm512_add_epi32(row,pred),r.joint_packed.data(),2),_mm512_set1_epi32(65535)),1);
            else {auto pi=_mm512_i32gather_epi32(pred,r.joint->pi.data(),4);g=_mm512_mask_i32gather_epi32(_mm512_setzero_si512(),active,_mm512_add_epi32(row,pi),r.joint_ids.data(),4);}
            if(gate)g=_mm512_slli_epi32(g,1);
            auto sy=joint_take16(r,g,active);
            if(gate){
                require((_mm512_cmp_epu32_mask(sy,_mm512_set1_epi32(1),_MM_CMPINT_GT)&active)==0);
                __mmask16 nonzero=active&_mm512_cmpeq_epi32_mask(sy,_mm512_setzero_si512());
                auto ranks=joint_take16(r,_mm512_add_epi32(g,_mm512_set1_epi32(1)),nonzero);
                require((_mm512_cmp_epu32_mask(ranks,_mm512_set1_epi32(255),_MM_CMPINT_GE)&nonzero)==0);
                alignas(64) unsigned zs[16]={};for(unsigned j=0;j<std::min(16u,lanes-h);j++)zs[j]=ng_prediction_symbol(0,q[h+j]);
                auto z=_mm512_load_si512(zs);ranks=_mm512_mask_add_epi32(ranks,_mm512_cmp_epu32_mask(ranks,z,_MM_CMPINT_GE),ranks,_mm512_set1_epi32(1));
                sy=_mm512_mask_mov_epi32(z,nonzero,ranks);
            }
            _mm512_storeu_si512(symbols+h,sy);
        }
    }
#endif
    template<bool Packed,bool Gate,bool Vector=false> static int joint_wave(ConditionedRead&r,ng_route_wave_data*w){
        const auto&m=*r.joint;unsigned P=m.pthr.size()+1,A=m.athr.size()+1;
        r.joint_rows.resize(w->width);
        for(unsigned j=0;j<w->lanes;j++){
            size_t block=(size_t(w->zs[j]/4)*(r.s.d/4)+w->ys[j]/4)*(r.s.d/4);
            for(unsigned x=0;x<w->width;x++){
                unsigned bank=(r.bank_classes.size()==1?r.bank_classes[0]:r.bank_classes[block+x/4])/r.K;
                unsigned a=w->activity[j*w->width+x];
                r.joint_rows[x][j]=Packed?(bank*A+m.ai[a])*256:bank*A*P+r.joint_activity[a];
            }
        }
        auto*columns=static_cast<ng_bank_compact_column*>(w->columns);
        for(unsigned x=0;x<w->width;x++){
            alignas(64) uint16_t q[32]={},residual[32]={};
            auto&c=columns[x];auto&weights=w->block_weights[x/4];
            if(w->compact_step){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
                if(!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw"))ng_route_predict_avx512(&c,weights,w->previous,q);
                else if(__builtin_cpu_supports("avx2"))ng_route_predict_avx2(&c,weights,w->previous,q);else
#endif
                ng_route_predict_c(&c,weights,w->previous,q);
            }else {auto*full=static_cast<ng_bank_wave_column*>(w->columns)+x;for(unsigned j=0;j<w->lanes;j++)q[j]=ng_route_full_prediction(full,w->previous,j);}
            alignas(64) unsigned symbols[32];
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            if constexpr(Vector)joint_vector_symbols(r,r.joint_rows[x].data(),q,w->lanes,symbols,Gate,Packed);
#endif
            for(unsigned j=0;j<w->lanes;j++){
                unsigned sym;
                if constexpr(Vector)sym=symbols[j];else{
                unsigned pred=geometry_trials::predicted(q[j]),g;
                if constexpr(Packed)g=r.joint_packed[r.joint_rows[x][j]+pred]>>1;
                else g=m.ids[r.joint_rows[x][j]+m.pi[pred]];
                auto&v=r.cursors[g*(Gate?2:1)];if(v.next==v.end)return -1;sym=*v.next++;
                if constexpr(Gate){
                    if(sym>1)return -1;unsigned zs=ng_prediction_symbol(0,q[j]);
                    if(sym)sym=zs;else{auto&rest=r.cursors[g*2+1];if(rest.next==rest.end)return -1;unsigned rank=*rest.next++;if(rank>=255)return -1;sym=rank+(rank>=zs);}
                }
                }
                residual[j]=uint8_t((sym>>1)^-(sym&1));w->errors[w->error_offsets[j]+x]=uint8_t((sym+1)>>1);
            }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            if(!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw"))ng_route_finish_avx512(q,residual,w->previous,w->values+x*32);
            else if(__builtin_cpu_supports("avx2"))ng_route_finish_avx2(q,residual,w->previous,w->values+x*32);else
#endif
            ng_route_finish_c(q,residual,w->previous,w->values+x*32);
        }
        // The native wave driver publishes this entire wave once. No redundant
        // scattered output writes are needed for this classifier.
        return 0;
    }
    static int wave(void*p,ng_route_wave_data*w){try{auto&r=*static_cast<ConditionedRead*>(p);
        if(r.joint&&joint_decode_variant){
            auto tick=std::chrono::steady_clock::time_point{};if(decode_profile_enabled)tick=std::chrono::steady_clock::now();
            int rc;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            if(joint_decode_variant>=3&&!ng_disable_avx512&&__builtin_cpu_supports("avx512vpopcntdq")&&__builtin_cpu_supports("avx512cd"))
                rc=joint_decode_variant==4?(r.joint->gate?joint_wave<true,true,true>(r,w):joint_wave<true,false,true>(r,w)):(r.joint->gate?joint_wave<false,true,true>(r,w):joint_wave<false,false,true>(r,w));
            else
#endif
            rc=(joint_decode_variant==2||joint_decode_variant==4)?(r.joint->gate?joint_wave<true,true>(r,w):joint_wave<true,false>(r,w)):(r.joint->gate?joint_wave<false,true>(r,w):joint_wave<false,false>(r,w));
            if(decode_profile_enabled)wave_total_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();return rc;
        }
        auto start=std::chrono::steady_clock::time_point{};if(decode_profile_enabled)start=std::chrono::steady_clock::now();
        if(r.pm)for(unsigned j=0;j<w->lanes;j++)r.prepare_edge_row(r.wave_rows[j],w->offsets[j],true);
        if(decode_profile_enabled)wave_edge_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::vector<std::array<unsigned,32>>bases(w->width);
        for(unsigned j=0;j<w->lanes;j++){
            unsigned y=w->ys[j],z=w->zs[j],interior=y<4?3:y+4>=r.s.d?4:z<2?5:0;
            size_t block=((size_t(z/4)*(r.s.d/4)+y/4)*(r.s.d/4));
            for(unsigned x=0;x<w->width;x++){
                unsigned f=x<4?1:x+4>=r.s.d?2:interior;
                bases[x][j]=(r.bank_classes.size()==1?r.bank_classes[0]:r.bank_classes[block+x/4])+(r.em==1?bool(f):r.em==2?f:0)*r.K;
            }
        }
        auto*columns=static_cast<ng_bank_compact_column*>(w->columns);
        for(unsigned x=0;x<w->width;x++){
            auto&c=columns[x];uint16_t q[32]={},residual[32]={};auto&weights=w->block_weights[x/4];
            auto*full=static_cast<ng_bank_wave_column*>(w->columns)+x;
            if(w->compact_step){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            if(!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw"))ng_route_predict_avx512(&c,weights,w->previous,q);
            else if(__builtin_cpu_supports("avx2"))ng_route_predict_avx2(&c,weights,w->previous,q);else
#endif
            ng_route_predict_c(&c,weights,w->previous,q);
            }else for(unsigned j=0;j<w->lanes;j++)q[j]=ng_route_full_prediction(full,w->previous,j);
            for(unsigned j=0;j<w->lanes;j++){
                size_t i=w->offsets[j]+x;if(r.pm)q[j]=r.edge_value(r.wave_rows[j],x,q[j]);
                unsigned a=w->activity[j*w->width+x],cls=w->class_lut[a];int sym=r.joint?r.read_joint(bases[x][j]/r.K,a,q[j]):r.read_context(bases[x][j]+cls,q[j]);if(sym<0)return -1;
                residual[j]=uint8_t((sym>>1)^-(sym&1));
                w->errors[w->error_offsets[j]+x]=uint8_t((sym+1)>>1);
            }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            if(!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw"))ng_route_finish_avx512(q,residual,w->previous,w->values+x*32);
            else if(__builtin_cpu_supports("avx2"))ng_route_finish_avx2(q,residual,w->previous,w->values+x*32);else
#endif
            ng_route_finish_c(q,residual,w->previous,w->values+x*32);
            for(unsigned j=0;j<w->lanes;j++)r.out[w->offsets[j]+x]=w->values[x*32+j];
        }if(decode_profile_enabled)wave_total_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return 0;
    }catch(...){return -1;}}
    static int lookup(void*p,size_t i,unsigned cls,uint16_t q){auto&r=*static_cast<ConditionedRead*>(p);
        unsigned bc=r.bank_classes.size()==1?r.bank_classes[0]+cls:r.bank_classes[geometry_trials::block(i,r.s.d)]+cls;
        bc+=(r.em==1?bool(face(i,r.s.d)):r.em==2?face(i,r.s.d):0)*r.K;
        return r.read_context(bc,q);
    }
    int read_context(unsigned bc,uint16_t q){
        unsigned g=routes[bc][geometry_trials::predicted(q)];auto&v=cursors[g];if(v.next==v.end)return -1;
        unsigned sym=*v.next++;if(gated[bc]){if(sym>1)return -1;unsigned zs=ng_prediction_symbol(0,q);
            if(sym)return zs;auto&rest=cursors[g+1];if(rest.next==rest.end)return -1;unsigned rank=*rest.next++;if(rank>=255)return -1;sym=rank+(rank>=zs);}
        return sym;
    }
    int read_joint(unsigned bank,unsigned a,uint16_t q){unsigned g=joint->group(bank,a,q)*(joint->gate?2:1);auto&v=cursors[g];if(v.next==v.end)return -1;unsigned sym=*v.next++;if(joint->gate){if(sym>1)return -1;unsigned zs=ng_prediction_symbol(0,q);if(sym)return zs;auto&rest=cursors[g+1];if(rest.next==rest.end)return -1;unsigned rank=*rest.next++;if(rank>=255)return -1;sym=rank+(rank>=zs);}return sym;}
    static int get(void*p,size_t i,unsigned cls){auto&r=*static_cast<ConditionedRead*>(p);uint16_t q=r.predictor->predict(i,r.out);
        auto&choice=r.s.menu[r.s.selectors[geometry_trials::block(i,r.s.d)]];
        if(r.pm&&choice.predictor!=choice_frame::zero_predictor&&face(i,r.s.d))q=edge_prediction(r.out,i,*r.plan,r.pm,r.weights);
        unsigned sym=symbol(p,i,cls,q);uint8_t value=geometry_trials::reconstruct(sym,q);
        require(choice.predictor!=choice_frame::zero_predictor||value==0);
        r.out[i]=value;r.predictor->accept(i,value,q);return sym;
    }
};
void reconstruct_frame(const Snapshot&s,unsigned mode,unsigned pm,unsigned em,const Weights&w,std::vector<Bytes>&streams,Bytes policies,std::vector<unsigned>offsets,uint8_t*out,bool wave=false,const JointMap*joint=nullptr){
    unsigned G=streams.size(),K=s.thresholds.size()+1;
    if(mode>=24||wave||(integrated_prediction&&pm<=5&&em<=2)){ConditionedRead read(s,mode,streams,out);Bytes raw(N);
        read.policies=std::move(policies);read.offsets=std::move(offsets);read.em=em;read.joint=joint;
        if(pm){require(pm>=1&&pm<=5);read.pm=pm;read.weights=w;read.plan=std::make_unique<Plan>(s,pm==5);if(!read.predictor)read.predictor=std::make_unique<FastPrediction>(s);}
        if(wave){read.prepare();ng_route_context route={};route.ctx=&read;route.thresholds=s.thresholds.data();route.K=K;route.wave=ConditionedRead::wave;route.compact=1;route.common_prediction=1;
            native_prediction_walk(s,out,nullptr,&read,&route);}
        else if(integrated_prediction){if(lookup_routing)read.prepare();
            struct Hook{Hook(bool edge){ng_bank_set_prediction_transform(edge?ConditionedRead::transform:nullptr);}~Hook(){ng_bank_set_prediction_transform(nullptr);}}hook(pm!=0);
            native_prediction_walk(s,out,lookup_routing?ConditionedRead::lookup:ConditionedRead::symbol,&read);}
        else require(ng_activity_decode(s.thresholds.data(),K,0,ConditionedRead::get,&read,raw.data())==0);
        bool vector=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        vector=joint&&joint_decode_variant>=3&&!ng_disable_avx512&&__builtin_cpu_supports("avx512vpopcntdq")&&__builtin_cpu_supports("avx512cd");
#endif
        for(unsigned g=0;g<G;g++)require(vector?read.joint_positions[g]==read.joint_ends[g]:wave||(integrated_prediction&&lookup_routing)?read.cursors[g].next==read.cursors[g].end:read.pos[g]==streams[g].size());return;
    }
    Read read{s,em,streams,std::vector<size_t>(G)};Bytes raw(N),symbols(N);
    require(ng_activity_decode(s.thresholds.data(),K,0,Read::get,&read,raw.data())==0);
    for(unsigned g=0;g<G;g++)require(read.pos[g]==streams[g].size());
    for(size_t i=0;i<N;i++)symbols[i]=pyramid(raw[i]);walk(s,nullptr,symbols,out,pm,w);
}
void decode_compact(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    auto tick=std::chrono::steady_clock::now();wave_edge_seconds=wave_total_seconds=0;
    Bytes frame(src,src+n);packed_bank::Reader r{frame};r.skip(4);auto hn=r.u(4),zn=r.u(4);auto hb=packed_bank::decompress(r.take(zn),hn);packed_bank::Reader h{hb};
    Snapshot s{};auto d=h.var();require(d<=UINT32_MAX&&ng_geometry_valid(d,d,d));s.d=d;
    GeometryScope geom(s.d,s.d,s.d);choice_grid::Scope grid(4);require(cap>=N);
    unsigned mode=h.u(1),pm=h.u(1),em=h.u(1);require(mode<=31&&pm<=12&&em<=4&&(mode<24||em<=2));s.mode=h.u(1);require(s.mode<=1);
    unsigned M=h.u(1),C=h.u(1),K=h.u(1);require(M<=4&&C&&C<=60&&K>=2&&K<=32);
    for(unsigned m=0;m<M;m++){auto size=h.var();require(size<=65536);s.models.push_back(h.take(size));}
    for(unsigned c=0;c<C;c++){choice_frame::Choice v;v.predictor=h.u(1);v.model=h.u(1);v.entropy=h.u(1);require(v.predictor<=12&&(v.model==255||v.model<M)&&v.entropy<16);
        for(auto&w:v.weights){auto x=compact_contexts::signed_var(h);require(x>=INT32_MIN&&x<=INT32_MAX);w=x;}s.menu.push_back(v);}
    unsigned selector=h.u(1);if(selector==255){unsigned bits=0;while((1u<<bits)<C)bits++;s.selectors=choice_frame::unpack_ids(h.take((N/64*bits+7)/8),N/64,C);}
    else{require(selector<C);s.selectors.assign(N/64,selector);}
    uint64_t prev=0;for(unsigned k=1;k<K;k++){auto delta=h.var();require(delta<=65535-prev);prev+=delta;s.thresholds.push_back(prev);}
    Weights w{};if(pm>=3&&pm<=5){unsigned count=h.var(),expected=pm==5?Plan(s,true).inputs.size():pm==3?6:12;require(count==expected&&count<160);
        for(unsigned g=1;g<6;g++)for(unsigned k=0;k<=count;k++){auto delta=compact_contexts::signed_var(h);require(delta>=INT64_C(-4294967295)&&delta<=INT64_C(4294967295));auto v=delta+(g>1?int64_t(w[g-1][k]):0);require(v>=INT32_MIN&&v<=INT32_MAX);w[g][k]=v;}}
    unsigned G=hybrid_geometry::banks(s)*K*entropy_groups(em)*(mode>=24?prediction_contexts(mode)*(zero_gate(mode)?2:1):1);Bytes policies;std::vector<unsigned>offsets;
    if(mode>=30){policies=h.take(hybrid_geometry::banks(s)*entropy_groups(em)*K);G=0;for(auto p:policies){require((p>=24&&p<=29)||(p>=32&&p<=35));offsets.push_back(G);G+=prediction_contexts(p)*(zero_gate(p)?2:1);}}
    if(decode_profile_enabled){auto now=std::chrono::steady_clock::now();decode_phases[0]=std::chrono::duration<double>(now-tick).count();tick=now;}
    JointMap joint;bool isjoint=src[3]=='4';if(isjoint){require(mode==0&&pm==0&&em==0);joint.read(h);require(joint.banks==hybrid_geometry::banks(s));G=joint.groups*(joint.gate?2:1);}
    bool gate=zero_gate(mode)||mode>=30||(isjoint&&joint.gate);size_t limit=N*(gate?2:1);auto streams=compact_contexts::read(h,r,G,limit);size_t total=0;for(auto&v:streams)total+=v.size();require(gate?total>=N&&total<=limit:total==N);h.end();r.end();
    if(decode_profile_enabled){auto now=std::chrono::steady_clock::now();decode_phases[1]=std::chrono::duration<double>(now-tick).count();tick=now;}
    reconstruct_frame(s,mode,pm,em,w,streams,std::move(policies),std::move(offsets),out,src[3]=='3'||isjoint,isjoint?&joint:nullptr);
    if(decode_profile_enabled)decode_phases[2]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
}
void decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    require(n>=12);if(!memcmp(src,"EGB2",4)||!memcmp(src,"EGB3",4)||!memcmp(src,"EGB4",4)){decode_compact(src,n,out,cap);return;}if(memcmp(src,"EGB1",4)){require(ng_codec_decode(src,n,out,cap)==0);return;}Bytes frame(src,src+n);packed_bank::Reader r{frame};r.skip(4);auto hn=r.u(4),zn=r.u(4);auto hb=packed_bank::decompress(r.take(zn),hn);packed_bank::Reader h{hb};
    Snapshot s{};s.d=h.u(4);unsigned mode=h.u(4);require(mode<=31&&ng_geometry_valid(s.d,s.d,s.d));GeometryScope geom(s.d,s.d,s.d);choice_grid::Scope grid(4);require(cap>=N);
    s.mode=h.u(4);unsigned M=h.u(4),C=h.u(4),K=h.u(4);require(M<=4&&C<=60&&K>=2&&K<=32);
    for(unsigned m=0;m<M;m++)s.models.push_back(h.take(h.u(4)));
    for(unsigned c=0;c<C;c++){choice_frame::Choice v;v.predictor=h.u(4);v.model=h.u(4);v.entropy=h.u(4);for(auto&w:v.weights)w=int32_t(packed_bank::signed_value(h.u(4),32));s.menu.push_back(v);}
    s.selectors=small_bank::read_stream(h,r,N/64);require(s.selectors.size()==N/64);for(auto c:s.selectors)require(c<C);
    for(unsigned k=1;k<K;k++)s.thresholds.push_back(h.u(2));
    unsigned pm=prediction_mode(mode),em=entropy_mode(mode);Weights w{};
    if(pm>=3&&pm<=5){unsigned count=pm==5?Plan(s,true).inputs.size():pm==3?6:12;for(unsigned g=1;g<6;g++)for(unsigned k=0;k<=count;k++)w[g][k]=int32_t(packed_bank::signed_value(h.u(4),32));}
    unsigned G=hybrid_geometry::banks(s)*K*(mode>=24?prediction_contexts(mode)*(zero_gate(mode)?2:1):entropy_groups(em));Bytes policies;std::vector<unsigned>offsets;
    if(mode>=30){policies=h.take(hybrid_geometry::banks(s)*K);G=0;for(auto p:policies){require((p>=24&&p<=29)||(p>=32&&p<=35));offsets.push_back(G);G+=prediction_contexts(p)*(zero_gate(p)?2:1);}}
    std::vector<Bytes>streams(G);size_t total=0,limit=N*((zero_gate(mode)||mode>=30)?2:1);
    for(auto&v:streams){v=small_bank::read_stream(h,r,limit-total);total+=v.size();}require((zero_gate(mode)||mode>=30)?total>=N&&total<=limit:total==N);h.end();r.end();
    reconstruct_frame(s,mode,pm,em,w,streams,std::move(policies),std::move(offsets),out);
}
}
extern "C" void*ng_edge_create(const uint8_t*s,unsigned d,unsigned preset){try{return edge_trials::create(s,d,preset);}catch(const std::exception&e){fprintf(stderr,"edge create: %s\n",e.what());return nullptr;}}
extern "C" void ng_edge_destroy(void*p){delete static_cast<edge_trials::Trial*>(p);}
extern "C" void ng_edge_cost_counts(void*p,uint64_t*out){auto&t=*static_cast<edge_trials::Trial*>(p);std::copy_n(t.policy_counts.begin(),6,out);}
extern "C" void ng_edge_cost_counts10(void*p,uint64_t*out){auto&t=*static_cast<edge_trials::Trial*>(p);std::copy(t.policy_counts.begin(),t.policy_counts.end(),out);}
extern "C" void ng_edge_expanded_menu(int enabled){edge_trials::expanded_menu=enabled!=0;}
extern "C" void ng_edge_integrated_prediction(int enabled){edge_trials::integrated_prediction=enabled!=0;}
extern "C" void ng_edge_lookup_routing(int enabled){edge_trials::lookup_routing=enabled!=0;}
extern "C" void ng_edge_search_strategy(unsigned strategy){edge_trials::search_strategy=strategy;}
extern "C" void ng_edge_compact_frame(int enabled){edge_trials::compact_frame=enabled!=0;}
extern "C" void ng_edge_with_context(int enabled){edge_trials::edge_context=enabled!=0;}
extern "C" void ng_edge_context_partition(unsigned value){edge_trials::context_partition=std::min(value,2u);}
extern "C" void ng_edge_compact_selection(int enabled){edge_trials::compact_selection=enabled!=0;}
extern "C" void ng_edge_wave_frame(int enabled){edge_trials::wave_frame=enabled!=0;}
extern "C" void ng_edge_wave_phases(double*out){out[0]=edge_trials::wave_edge_seconds;out[1]=edge_trials::wave_total_seconds;}
extern "C" void ng_edge_decode_profile(int enabled){edge_trials::decode_profile_enabled=enabled!=0;}
extern "C" void ng_joint_decode_variant(unsigned v){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    // VPOPCNT/CD alone do not imply BW/VL on every x86 implementation.
    if(v>=3&&v<=4&&(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512bw")||!__builtin_cpu_supports("avx512vl")||!__builtin_cpu_supports("avx512cd")||!__builtin_cpu_supports("avx512vpopcntdq")))v-=2;
#endif
    edge_trials::joint_decode_variant=v<=4?v:0;
}
extern "C" void ng_edge_decode_phases(double*out){std::copy_n(edge_trials::decode_phases,3,out);}
extern "C" int ng_compact_context_selftest(){try{
    std::vector<Bytes>input(604);
    for(unsigned i=0;i<600;i++){input[i].assign(1024,uint8_t((i%251)+1));std::fill_n(input[i].begin(),i+1,uint8_t(i%251));}
    input[600]=input[0];input[601].assign(73,0);input[602].assign(31,255);
    Bytes h,b;compact_contexts::write(h,b,input);packed_bank::Reader probe{h};require(probe.var()==input.size()&&probe.var()>256);
    packed_bank::Reader header{h},body{b};auto output=compact_contexts::read(header,body,input.size(),700000);header.end();body.end();require(input==output);
    for(auto empty: {std::vector<Bytes>{},std::vector<Bytes>(32)}){Bytes hh,bb;compact_contexts::write(hh,bb,empty);packed_bank::Reader hr{hh},br{bb};require(compact_contexts::read(hr,br,empty.size(),0)==empty);hr.end();br.end();}
    return 0;
}catch(const std::exception&e){fprintf(stderr,"compact contexts: %s\n",e.what());return -1;}}
extern "C" int ng_edge_encode(void*p,unsigned mode,uint8_t*out,size_t cap,size_t*n){try{require(p&&out&&n);auto&t=*static_cast<edge_trials::Trial*>(p);auto f=mode==255?t.s.frame:edge_trials::encode(t,mode);require(f.size()<=cap);std::copy(f.begin(),f.end(),out);*n=f.size();return 0;}catch(const std::exception&e){fprintf(stderr,"edge encode: %s\n",e.what());return -1;}}
extern "C" int ng_edge_decode(const uint8_t*s,size_t n,uint8_t*out,size_t cap){try{edge_trials::decode(s,n,out,cap);return 0;}catch(const std::exception&e){fprintf(stderr,"edge decode: %s\n",e.what());return -1;}}
