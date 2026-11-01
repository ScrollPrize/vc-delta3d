// Entropy-only experiment on an unchanged, completed Fast/Slow encode.
#include "ng_wave_column.h"
#include "ng_entropy_state.h"
namespace hybrid_geometry {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include "ng_entropy_permutation.h"
using entropy_tree::vcn_lc_gather;
#define VCN_LC_RL (1u<<15)
#define NG_EATTR __attribute__((target("avx512f,avx512bw,avx512vl,popcnt"),always_inline))
#define NG_ENAME context_entropy_wide_impl
#define NG_EUNROLL 1
#include "ng_entropy_vector.h"
#undef VCN_LC_RL
__attribute__((target("avx512f,avx512bw,avx512vl,popcnt"))) static int context_entropy_wide(void*p,const uint8_t*cl,uint8_t*symbols){
    auto*c=static_cast<ng_context_rans*>(p);context_entropy_wide_impl(32,cl,&c->d,symbols,1);return c->d.over?-1:0;
}
// Convert 32 compiled tree leaves to entropy-bank IDs without scalar staging.
// The byte stream and context assignment are unchanged.
__attribute__((target("avx512f,avx512bw,avx512vl"))) static void context_groups(const unsigned*base,const int*leaf,uint8_t*out){
    for(unsigned j=0;j<32;j+=16)
        _mm_storeu_si128((__m128i*)(out+j),_mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_loadu_si512(base+j),_mm512_loadu_si512(leaf+j))));
}
#endif
thread_local unsigned acceleration=64;
thread_local unsigned remaining_mode=4; // Measured register pipeline; other modes are opt-in trials.
thread_local uint64_t wave_batch_counts[5]={};
thread_local bool independent_wave_encode=false;
#include "ng_tree_compact_entropy.hpp"
thread_local bool specialized_profile=false;
thread_local bool profile_enabled=false;
thread_local double profile[8]={};
thread_local bool vector_enabled=true;
thread_local bool row_batch_enabled=true;
// Experimental HGT2/HGT3 decoder default. Production NCF1 dispatch is unchanged.
thread_local unsigned wave_pipeline=20;
thread_local bool bulk_rows_enabled=true;
thread_local double bulk_profile[3]={};
thread_local double wave_parts[8]={};
// Diagnostic only: replay known leaf IDs to isolate routing cost, not a codec.
thread_local unsigned replay_mode=0;
thread_local Bytes replay_leaves;
thread_local size_t replay_cursor=0;
struct Snapshot {
    unsigned d,mode;
    std::vector<Bytes> models;
    std::vector<choice_frame::Choice> menu;
    Bytes selectors,symbols,classes,frame;
    std::vector<uint16_t> thresholds;
    unsigned choice_edge=choice_grid::edge();
    Bytes regions; // optional independent spatial entropy-bank map
};
unsigned entropy_bank(const Snapshot&s,unsigned fallback,unsigned x,unsigned y,unsigned z){return s.regions.empty()?fallback:s.regions[adaptive_grid::index(x/2,y/2,z/2,s.d/2)];}
#include "ng_tree_column_metadata.hpp"
thread_local std::vector<Snapshot>*captured=nullptr;
thread_local unsigned capture_dimension=0;
void observe(unsigned d,const std::vector<Bytes>&models,const std::vector<choice_frame::Choice>&menu,
    const Bytes&sel,const Bytes&sy,const std::vector<uint16_t>&thr,const Bytes&cl,unsigned mode,const Bytes&frame){
    if(captured&&d==capture_dimension)captured->push_back({d,mode,models,menu,sel,sy,cl,frame,thr});
}
struct Capture {
    choice_frame::Observer old=choice_frame::observer;
    std::vector<Snapshot>*previous=captured;unsigned previous_d=capture_dimension;
    Capture(std::vector<Snapshot>&v,unsigned d){captured=&v;capture_dimension=d;choice_frame::observer=observe;}
    ~Capture(){choice_frame::observer=old;captured=previous;capture_dimension=previous_d;}
};
struct Trial {Snapshot s;Bytes source;std::array<entropy_tree::Forest,3>trees;unsigned feature_mask=511;Bytes cached_leaf;};
Trial*create(const uint8_t*src,unsigned d,unsigned preset,unsigned predictor=0,const std::vector<unsigned>*menu=nullptr){
    require(src&&ng_geometry_valid(d,d,d)&&preset<=1&&predictor<=4);GeometryScope scope(d,d,d);
    std::vector<Snapshot>v;Capture capture(v,d);Bytes base(3*N+131072);size_t n=0;
    int cfg[6]={3,3,1,16,40,1};
    //160 exceeds all122 raw inputs (+the optional predictor input), including
    //the first-slice budget160/2: no greedy coefficient selection in variant4.
    if(predictor){cfg[2]=3;cfg[4]=predictor==4?160:predictor==3?40:80;}
    struct ContextScope{int old=ng_fit_context_mode;ContextScope(bool on){ng_fit_context_mode=on;}~ContextScope(){ng_fit_context_mode=old;}} context(predictor==2);
    int rc;
    if(preset){bool old=choice_frame::enabled;choice_frame::enabled=true;
        struct MenuScope{const std::vector<unsigned>*old=fast_split::stream_menu;
            MenuScope(const std::vector<unsigned>*p){fast_split::stream_menu=p;}
            ~MenuScope(){fast_split::stream_menu=old;}} menus(menu);
        rc=ng_screen_pair_encode_impl(src,d,12,base.data(),base.size(),&n,cfg,16,3,true,menu,3);
        choice_frame::enabled=old;
    }else rc=ng_codec_encode_fast_config(src,d,cfg,16,base.data(),base.size(),&n);
    require(rc==0);base.resize(n);
    for(auto&s:v)if(s.frame==base){
        if(s.symbols.empty()){s.symbols.resize(N);s.classes.resize(N);}
        return new Trial{std::move(s),Bytes(src,src+N)};
    }
    throw std::runtime_error("chosen hybrid frame not captured");
}
unsigned banks(const Snapshot&s){unsigned n=1;for(auto&c:s.menu)n=std::max(n,c.entropy+1);return n;}
void tree_samples(const Trial&t,std::vector<entropy_tree::Feature>&features,std::vector<std::vector<uint32_t>>&ids){
    const auto&s=t.s;features.resize(N);ids.resize(banks(s));
    for(size_t i=0;i<N;i++){
        auto&c=s.menu.at(s.selectors.at(geometry_trials::block(i,s.d)));
        if(c.predictor==choice_frame::zero_predictor)continue;
        for(unsigned f=0;f<entropy_tree::features;f++)if(t.feature_mask&(1u<<f))features[i][f]=entropy_tree::feature(t.source.data(),i,s.d,s.classes[i],c.predictor,f);
        ids[c.entropy].push_back(i);
    }
}
void fit_trees(Trial&t){
    if(!t.trees[0].empty())return;unsigned B=banks(t.s);
    std::vector<entropy_tree::Feature>features;std::vector<std::vector<uint32_t>>ids;
    tree_samples(t,features,ids);
    std::vector<double>nlogn(N+1);for(size_t i=1;i<=N;i++)nlogn[i]=double(i)*std::log2(double(i));
    for(unsigned b=0;b<B;b++){auto trees=entropy_tree::fit(ids[b],features,t.s.symbols,nlogn,t.feature_mask);for(unsigned j=0;j<3;j++)t.trees[j].push_back(std::move(trees[j]));}
}
#include "ng_learned_difficulty.hpp"
#include "ng_entropy_alternatives.hpp"
Bytes encode(Trial&t,unsigned geometry){
    choice_grid::Scope grid(t.s.choice_edge);
    require(geometry<=18);const auto&s=t.s;if(geometry==2)return s.frame;
    const bool alternative=geometry>=10;
    const bool learned=geometry>=3;
    const bool joint=geometry>=7,wave=geometry>=6;unsigned tree_index=std::min(geometry,5u)-3;
    const bool row_entropy=geometry>=9&&geometry!=17;if(row_entropy&&!alternative)require(!(t.feature_mask&~entropy_tree::row_features));
    GeometryScope scope(s.d,s.d,s.d);unsigned K=s.thresholds.size()+1,G=alternative?(geometry>=16?32:geometry>=14?64:16):learned?(16u<<tree_index):geometry?4:1,B=banks(s);
    bool regional=!s.regions.empty();require(!regional||(geometry==7&&s.regions.size()==size_t(s.d/2)*(s.d/2)*(s.d/2)));
    unsigned E=regional?1+*std::max_element(s.regions.begin(),s.regions.end()):B;require(E*G<=256);
    entropy_alternatives::Router router;if(alternative)router=entropy_alternatives::fit(s,t.source,geometry);
    if(learned&&!alternative)fit_trees(t);
    std::vector<Bytes>streams((learned?1:K)*G*B);
    Bytes ordered_symbols,ordered_groups,ordered_lanes;
    std::vector<size_t>wave_ends;
    if(joint){ordered_symbols.reserve(N);ordered_groups.reserve(N);ordered_lanes.reserve(N);}
    auto append=[&](size_t i,unsigned lane=0){
        auto&c=s.menu.at(s.selectors.at(geometry_trials::block(i,s.d)));
        unsigned g=0;
        if(c.predictor!=choice_frame::zero_predictor){
            if(alternative)g=router.group(i,s.d,s.classes[i],K,c.entropy,t.source.data())-c.entropy*G;
            else if(learned)g=geometry==7&&!t.cached_leaf.empty()?t.cached_leaf[i]:t.trees[tree_index][c.entropy].classify(t.source.data(),i,s.d,s.classes[i],c.predictor);
            else if(geometry)g=geometry_trials::entropy_shape(t.source.data(),i%s.d,i/s.d%s.d,i/(s.d*s.d),s.d);
        }
        require(s.classes[i]<K);unsigned group=(learned?entropy_bank(s,c.entropy,i%s.d,i/s.d%s.d,i/(size_t(s.d)*s.d)):c.entropy*K+s.classes[i])*G+g;
        if(joint){ordered_symbols.push_back(s.symbols[i]);ordered_groups.push_back(group);ordered_lanes.push_back(lane);}
        else streams[group].push_back(s.symbols[i]);
    };
    if(wave){
        std::vector<const uint8_t*>p;std::vector<size_t>lengths;
        for(auto&m:s.models){p.push_back(m.data());lengths.push_back(m.size());}
        std::vector<uint32_t>rows(s.d*s.d),ends(s.d*s.d);uint32_t groups=0;
        require(ng_bank_route_order(p.data(),lengths.data(),p.size(),rows.data(),ends.data(),&groups)==0);
        unsigned start=0;for(unsigned g=0;g<groups;g++){
            if(row_entropy){for(unsigned j=start;j<ends[g];j++)for(unsigned x=0;x<s.d;x++)append(size_t(rows[j])*s.d+x,x%32);}
            else for(unsigned x=0;x<s.d;x++)for(unsigned j=start;j<ends[g];j++)append(size_t(rows[j])*s.d+x,j-start);
            start=ends[g];wave_ends.push_back(ordered_symbols.size());}
    }else for(size_t i=0;i<N;i++)append(i);
    Bytes h,body;put32(h,s.d);put32(h,choice_grid::mode_word(s.mode));put32(h,G);put32(h,K);put32(h,B);put32(h,s.models.size());put32(h,s.menu.size());
    for(auto&p:s.models){put32(h,p.size());packed_bank::append(h,p);}
    for(auto&c:s.menu){h.push_back(c.predictor);h.push_back(c.model);h.push_back(c.entropy);for(auto w:c.weights)put32(h,uint32_t(w));}
    bool uniform=std::all_of(s.selectors.begin(),s.selectors.end(),[&](auto c){return c==s.selectors[0];});
    put32(h,uniform?s.selectors[0]:255);
    if(!uniform){auto selected=pack_stream(s.selectors,true);put32(h,selected.size());packed_bank::append(h,selected);}
    for(auto v:s.thresholds)packed_bank::put16(h,v);
    if(alternative)entropy_alternatives::write(h,router);
    else if(learned)entropy_tree::write(h,t.trees[tree_index]);
    if(regional){put32(h,E);choice_frame::write_regions(h,s.regions,s.d);}
    if(joint){
        unsigned tb=geometry==8?10:12;std::vector<uint16_t>freq(E*G*256);body.resize(3*ordered_symbols.size()+128);size_t used=0;
        require(ng_context_rans_encode(ordered_symbols.data(),ordered_groups.data(),ordered_lanes.data(),ordered_symbols.size(),E*G,tb,freq.data(),body.data(),body.size(),&used)==0);body.resize(used);
        Bytes tables;uint8_t table[514];for(unsigned g=0;g<E*G;g++){size_t tn=ng_write_freq_table(table,freq.data()+g*256);tables.insert(tables.end(),table,table+tn);}
        put32(h,tb);put32(h,tables.size());packed_bank::append(h,tables);
        if(independent_wave_encode){
            require(geometry==7);body.clear();put32(h,wave_ends.size());size_t begin=0;
            for(auto end:wave_ends){Bytes stream(3*(end-begin)+128);size_t used=0;
                require(ng_context_rans_encode_fixed(ordered_symbols.data()+begin,ordered_groups.data()+begin,ordered_lanes.data()+begin,end-begin,B*G,tb,freq.data(),stream.data(),stream.size(),&used)==0);
                stream.resize(used);put32(h,used);packed_bank::append(body,stream);begin=end;
            }
        }else put32(h,body.size());
    }else for(auto&v:streams)small_bank::stream_header(h,body,pack_stream(v,true),v.size());
    auto z=packed_bank::compress(h);Bytes out={'H','G',uint8_t(alternative?'A':learned?'T':'E'),uint8_t(regional?'6':independent_wave_encode?'5':alternative?(row_entropy?'1':'2'):row_entropy?'4':joint?'3':wave?'2':'1')};put32(out,h.size());put32(out,z.size());packed_bank::append(out,z);packed_bank::append(out,body);return out;
}
#include "ng_tree_register.hpp"
#include "ng_tree_deferred.hpp"
struct Decoder : TreeColumnMetadata {
    bool independent_waves=false;
    using Entropy=std::unique_ptr<void,decltype(&ng_context_rans_destroy)>;
    Entropy wave_table_owner{nullptr,ng_context_rans_destroy};
    std::vector<Entropy>wave_entropies;
    std::vector<Bytes>wave_payloads;
    std::array<std::unique_ptr<DeferredTreeWave>,4>pending;
    unsigned pending_count=0,wave_index=0,wave_finished=0;int pending_t=-1;
    void flush_waves(){
        if(!pending_count)return;
        ++wave_batch_counts[pending_count];
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        DeferredTreeWave*q[4];for(unsigned j=0;j<pending_count;j++)q[j]=pending[j].get();
        switch(pending_count){
        case 1:tree_interleave(s,*vector,*masks,q,G,std::make_index_sequence<1>{});break;
        case 2:tree_interleave(s,*vector,*masks,q,G,std::make_index_sequence<2>{});break;
        case 3:tree_interleave(s,*vector,*masks,q,G,std::make_index_sequence<3>{});break;
        case 4:tree_interleave(s,*vector,*masks,q,G,std::make_index_sequence<4>{});break;
        }
        for(unsigned j=0;j<pending_count;j++){require(ng_context_rans_finish(q[j]->entropy.get())==0);q[j]->w.publish(&q[j]->w);++wave_finished;}
        pending_count=0;
#else
        throw std::runtime_error("unavailable interleaved kernel");
#endif
    }
    static int barrier(void*p,int t){try{auto&a=*static_cast<Decoder*>(p);if(t!=a.pending_t)a.flush_waves();a.pending_t=t;return 0;}catch(...){return -1;}}
    static int wave_independent(void*p,ng_route_wave_data*w){
        auto&a=*static_cast<Decoder*>(p);
        try{
            require(a.wave_index<a.wave_entropies.size());a.entropy.swap(a.wave_entropies[a.wave_index++]);
            bool batch=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            batch=(remaining_mode&56)&&w->compact_step&&!ng_disable_avx512&&!(a.vector->used&~511u)&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("avx512cd")&&__builtin_cpu_supports("popcnt");
#endif
            if(batch){
                auto&slot=a.pending[a.pending_count];if(!slot)slot=std::make_unique<DeferredTreeWave>();slot->capture(w,a.entropy);++a.pending_count;
                if(a.pending_count==((remaining_mode&32)?4u:(remaining_mode&16)?2u:1u))a.flush_waves();
                return 2;
            }
            a.flush_waves();int result=wave(p,w);require(!result&&ng_context_rans_finish(a.entropy.get())==0);++a.wave_finished;return 0;
        }catch(...){return -1;}
    }
    Snapshot s;unsigned G,K,B;uint8_t*out;
    bool row_entropy=false;
    entropy_alternatives::Router alternative;
    std::array<uint8_t,32> coarse_classes{};
    struct Cursor {const uint8_t*next;const uint8_t*end;};
    std::vector<Bytes>streams;std::vector<Cursor>cursors;
    entropy_tree::Forest trees;
    std::unique_ptr<void,decltype(&ng_context_rans_destroy)> entropy{nullptr,ng_context_rans_destroy};
    Bytes joint_payload;
    std::unique_ptr<TreeCompactEntropy> compact_entropy;
    Bytes row_lookup;
    Bytes bulk_groups,bulk_symbols;
    std::vector<int32_t> learned_scores;
    Bytes learned_banks,learned_north,learned_zero;
    // Shared bulk/full-context preparation. Hoist coordinates and selectors;
    // evaluate only nonzero learned coefficients, over contiguous row blocks.
    void prepare_learned(ng_route_wave_data*w){
        unsigned d=w->width,n=w->lanes;auto&m=alternative.score;
        learned_scores.resize(d*n);learned_banks.resize(d*n);learned_north.resize(d*n);learned_zero.resize(d*n);
        std::vector<int32_t> wa(d),wb(d);Bytes padded(d+6,128);
        for(unsigned j=0;j<n;j++){
            int y=w->ys[j],z=w->zs[j];size_t offset=w->offsets[j];
            auto*select=s.selectors.data()+(size_t(z/choice_grid::edge())*(d/choice_grid::edge())+y/choice_grid::edge())*(d/choice_grid::edge());
            auto*score=learned_scores.data()+j*d;auto*north=learned_north.data()+j*d;
            if(y)memcpy(north,out+offset-d,d);else memset(north,128,d);
            for(unsigned bx=0;bx<d/choice_grid::edge();bx++){
                auto&c=s.menu[select[bx]];auto*weights=m.weights.data()+c.entropy*m.F();bool zero=c.predictor==choice_frame::zero_predictor;
                memset(learned_banks.data()+j*d+bx*4,c.entropy,4);memset(learned_zero.data()+j*d+bx*4,zero,4);
                for(unsigned x=bx*4;x<bx*4+4;x++)score[x]=256*int(w->class_lut[w->activity[j*d+x]])+32*weights[0];
            }
            for(unsigned k=0;k<m.taps.size();k++){
                auto p=m.taps[k];if(!p[1]&&!p[2])continue;bool active=false;
                for(unsigned bx=0;bx<d/choice_grid::edge();bx++){
                    auto&c=s.menu[select[bx]];auto*weights=m.weights.data()+c.entropy*m.F();bool zero=c.predictor==choice_frame::zero_predictor;
                    int a=zero?0:weights[1+2*k],b=zero?0:weights[2+2*k];active|=a||b;
                    std::fill_n(wa.data()+bx*4,4,a);std::fill_n(wb.data()+bx*4,4,b);
                }
                if(!active)continue;
                int yy=y+p[1],zz=z+p[2];bool valid=yy>=0&&yy<int(d)&&zz>=0&&zz<int(d);
                if(valid)memcpy(padded.data()+3,out+(size_t(zz)*d+yy)*d,d);else memset(padded.data()+3,128,d);
                learned_difficulty::apply_row(d,score,padded.data()+3+p[0],north,wa.data(),wb.data());
            }
        }
    }
    std::array<uint8_t,511> difference_bins{};
    void prepare_row_lookup(){
        if(trees.empty())return;
        for(auto&t:trees)for(auto&n:t.nodes)if(n.feature!=255&&!(265u&(1u<<n.feature)))return;
        for(int d=-255;d<=255;d++)difference_bins[d+255]=entropy_tree::signed_bin(d);
        row_lookup.resize(s.menu.size()*K*32);
        for(unsigned m=0;m<s.menu.size();m++)for(unsigned cls=0;cls<K;cls++)for(unsigned diff=0;diff<32;diff++){
            auto&c=s.menu[m];auto&t=trees[c.entropy];unsigned node=0;
            while(t.nodes[node].feature!=255){auto&n=t.nodes[node];unsigned value=n.feature==0?cls:n.feature==3?diff:c.predictor;node=value<=n.cut?n.left:n.right;}
            row_lookup[(m*K+cls)*32+diff]=t.nodes[node].leaf;
        }
    }
    std::vector<entropy_tree::CompiledTree> compiled;
    std::unique_ptr<entropy_tree::VectorForest> vector;
    std::unique_ptr<entropy_tree::MaskForest> masks;
    unsigned x=0,y=0,z=0;
    using State=std::unique_ptr<void,decltype(&ng_causal_destroy)>;
    std::vector<State>states;
#define NG_BULK_ATTR
#define NG_BULK_NAME bulk_generic
#include "ng_tree_bulk_executor.inc"
#undef NG_BULK_ATTR
#undef NG_BULK_NAME
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define NG_BULK_ATTR __attribute__((target("avx512f,avx512bw,avx512vl,avx512cd,avx2"),flatten))
#define NG_BULK_NAME bulk_wide
#include "ng_tree_bulk_executor.inc"
#undef NG_BULK_ATTR
#undef NG_BULK_NAME
#endif
    static int bulk(void*p,ng_route_wave_data*w){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(vector_enabled&&!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("avx512cd"))return bulk_wide(p,w);
#endif
        return bulk_generic(p,w);
    }
    static int route(void*p,size_t i,unsigned cls){
        try{
            auto&a=*static_cast<Decoder*>(p);int symbol=a.symbol(i,cls);
            if(++a.x==a.s.d){a.x=0;if(++a.y==a.s.d){a.y=0;++a.z;}}
            return symbol;
        }catch(...){return -1;}
    }
    static int spatial(void*p,size_t i,unsigned cls,unsigned x,unsigned y,unsigned z){
        try{auto&a=*static_cast<Decoder*>(p);a.x=x;a.y=y;a.z=z;return a.symbol(i,cls);}catch(...){return -1;}
    }
#define NG_WAVE_ATTR
#define NG_WAVE_NAME wave_generic
#define NG_WAVE_PIPELINE wave_pipeline
#define NG_WAVE_STATIC_WIDE 0
#define NG_WAVE_STATIC_PATH 0
#include "ng_tree_wave_executor.inc"
#undef NG_WAVE_ATTR
#undef NG_WAVE_NAME
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define NG_WAVE_ATTR __attribute__((target("avx512f,avx512bw,avx512vl,avx2,popcnt"),flatten))
#define NG_WAVE_NAME wave_wide
#include "ng_tree_wave_executor.inc"
#undef NG_WAVE_ATTR
#undef NG_WAVE_NAME
#endif
#undef NG_WAVE_PIPELINE
#undef NG_WAVE_STATIC_WIDE
#undef NG_WAVE_STATIC_PATH
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define NG_WAVE_ATTR __attribute__((target("avx512f,avx512bw,avx512vl,avx512cd,avx2,popcnt"),flatten))
#define NG_WAVE_NAME wave_fixed
#define NG_WAVE_PIPELINE 18
#define NG_WAVE_STATIC_WIDE 1
#define NG_WAVE_STATIC_PATH 1
#include "ng_tree_wave_executor.inc"
#undef NG_WAVE_NAME
#undef NG_WAVE_PIPELINE
#define NG_WAVE_NAME wave_symbols
#define NG_WAVE_PIPELINE 20
#include "ng_tree_wave_executor.inc"
#undef NG_WAVE_NAME
#define NG_WAVE_NAME wave_symbols_profile
#define NG_WAVE_PROFILE 1
#include "ng_tree_wave_executor.inc"
#undef NG_WAVE_PROFILE
#undef NG_WAVE_ATTR
#undef NG_WAVE_NAME
#undef NG_WAVE_PIPELINE
#undef NG_WAVE_STATIC_WIDE
#undef NG_WAVE_STATIC_PATH
#endif
    static int wave(void*p,ng_route_wave_data*w){
        auto&decoder=*static_cast<Decoder*>(p);
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if((remaining_mode&6)&&decoder.entropy&&decoder.masks&&w->compact_step&&w->lanes==32&&!w->raw&&!ng_disable_avx512&&!(decoder.vector->used&~511u)&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("avx512cd")&&__builtin_cpu_supports("popcnt")){
            try{auto*e=static_cast<ng_context_rans*>(decoder.entropy.get());return (remaining_mode&4)?run_tree_register<true>(decoder.s,*decoder.vector,*decoder.masks,w,e,decoder.compact_entropy.get(),decoder,decoder.G):run_tree_register<false>(decoder.s,*decoder.vector,*decoder.masks,w,e,decoder.compact_entropy.get(),decoder,decoder.G);}catch(...){return -1;}
        }
        auto&a=*static_cast<Decoder*>(p);
        if(a.vector&&wave_pipeline>=18&&!profile_enabled&&w->compact_step&&w->lanes==32&&vector_enabled&&!(a.vector->used&~511u)&&!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("avx512cd")&&__builtin_cpu_supports("popcnt")){
            if(specialized_profile&&wave_pipeline>=20)return wave_symbols_profile(p,w);
            return wave_pipeline>=20?wave_symbols(p,w):wave_fixed(p,w);
        }
        if(wave_pipeline>=10&&!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("popcnt"))return wave_wide(p,w);
#endif
        return wave_generic(p,w);
    }
    static int batch(void*p,unsigned count,const size_t*indices,const unsigned*classes,unsigned x,const int*ys,const int*zs,const uint8_t*const*neighbors,uint8_t*symbols){
        try{
            auto&a=*static_cast<Decoder*>(p);unsigned d=a.s.d;
            int features[9*32]={},roots[32]={},leaves[32]={};unsigned groups[32]={},tree_ids[32]={};bool zeros[32]={};
            bool simd=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            simd=vector_enabled&&!(a.vector->used&~511u)&&__builtin_cpu_supports("avx2");
#endif
            for(unsigned j=0;j<count;j++){
                size_t i=indices[j];unsigned y=ys[j],z=zs[j];
                auto&c=a.s.menu[a.s.selectors[((size_t(z/choice_grid::edge())*(d/choice_grid::edge())+y/choice_grid::edge())*(d/choice_grid::edge()))+x/choice_grid::edge()]];
                roots[j]=a.vector->roots[c.entropy];tree_ids[j]=c.entropy;groups[j]=entropy_bank(a.s,c.entropy,x,y,z)*a.G;zeros[j]=c.predictor==choice_frame::zero_predictor;
                if(zeros[j]){roots[j]=a.vector->zero_root;continue;}
                features[j]=classes[j];features[256+j]=c.predictor;
                if(simd)continue;
                int w=x?neighbors[0][j]:128,n=y?neighbors[2][j]:128,b=z?neighbors[3][j]:128;
                features[j]=classes[j];features[32+j]=w-n;features[64+j]=w-b;features[96+j]=n-b;
                features[128+j]=std::max({w,n,b})-std::min({w,n,b});features[160+j]=std::abs(w-int(x>1?neighbors[1][j]:128));
                if(a.vector->used&(1u<<6)){
                    int base;
                    if(x&&y&&z){
                        constexpr unsigned map[]={0,0,2,4,3,5,6,7};
                        int value=int32_t(delta_stencil(7,2,4,[&](size_t offset){return uint32_t(neighbors[map[offset]][j]);}));
                        base=std::clamp(value,std::min({w,n,b}),std::max({w,n,b}));
                    }else base=ng_directional_base(a.out,i,11,x,y,z,d);
                    features[192+j]=base-(w+n+b+1)/3;
                }
                features[224+j]=w+n+b;features[256+j]=c.predictor;
            }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            if(simd){a.vector->make_features(count,x,ys,zs,neighbors,features);
                if(wave_pipeline>=3&&__builtin_cpu_supports("avx512f"))a.vector->classify_packed512(count,features,roots,leaves);
                else if(wave_pipeline>=2)a.vector->classify_packed(count,features,roots,leaves);else a.vector->classify(count,features,roots,leaves);}else
#endif
            for(unsigned j=0;j<count;j++){
                if(zeros[j]){leaves[j]=0;continue;}
                // Wavefront neighbors can still reside in the reconstruction
                // buffers, not out. Use the same gathered features as SIMD.
                if(a.vector->used&~511u){leaves[j]=a.compiled[tree_ids[j]].classify(a.out,indices[j],d,x,ys[j],zs[j],classes[j],features[256+j]);continue;}
                const auto&nodes=a.compiled[tree_ids[j]].nodes;unsigned node=0;
                for(unsigned depth=0;;depth++){
                    require(depth<=8);const auto&v=nodes[node];
                    if(v.feature==255){leaves[j]=v.leaf;break;}
                    node=features[32*v.feature+j]<=v.cut?v.left:v.right;
                }
            }
            if(a.entropy){uint8_t contexts[32];for(unsigned j=0;j<count;j++)contexts[j]=groups[j]+(zeros[j]?0:leaves[j]);require(ng_context_rans_read(a.entropy.get(),count,contexts,symbols,2)==0);
                for(unsigned j=0;j<count;j++)if(zeros[j]&&symbols[j])throw std::runtime_error("nonzero joint zero");
                return 0;
            }
            for(unsigned j=0;j<count;j++){
                unsigned g=groups[j]+(zeros[j]?0:leaves[j]);auto&cursor=a.cursors[g];
                if(cursor.next==cursor.end)throw std::runtime_error("exhausted entropy leaf");
                symbols[j]=*cursor.next++;if(zeros[j]&&symbols[j])throw std::runtime_error("nonzero zero-block symbol");
            }
            return 0;
        }catch(...){return -1;}
    }
    int symbol(size_t i,unsigned cls){
        unsigned d=s.d;
        unsigned selected=s.selectors[((size_t(z/choice_grid::edge())*(d/choice_grid::edge())+y/choice_grid::edge())*(d/choice_grid::edge()))+x/choice_grid::edge()];auto&c=s.menu[selected];
        unsigned g=0;
        if(c.predictor!=choice_frame::zero_predictor){
            if(!row_lookup.empty()){
                int north=y?out[i-d]:128,back=z?out[i-size_t(d)*d]:128;
                g=row_lookup[(selected*K+cls)*32+difference_bins[north-back+255]];
            }else if(!trees.empty())g=compiled[c.entropy].classify(out,i,d,x,y,z,cls,c.predictor);
            else if(G>1)g=geometry_trials::entropy_shape(out,x,y,z,d);
        }
        unsigned group=(trees.empty()?c.entropy*K+cls:entropy_bank(s,c.entropy,x,y,z))*G+g;
        // Bank/leaf/class bounds are validated when reading the frame/tree.
        auto&cursor=cursors[group];
        if(cursor.next==cursor.end)throw std::runtime_error("exhausted entropy leaf");
        unsigned value=*cursor.next++;
        if(c.predictor==choice_frame::zero_predictor&&value)throw std::runtime_error("nonzero zero-block symbol");
        return value;
    }
};
void decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    replay_cursor=0;if(replay_mode==1)replay_leaves.clear();
    auto tick=std::chrono::steady_clock::now();
    require(src&&out&&n>=12);
    if(!memcmp(src,"NCF1",4)){require(ng_codec_decode(src,n,out,cap)==0);return;}
    bool independent=!memcmp(src,"HGT5",4);
    bool regional=!memcmp(src,"HGT6",4);
    bool alternative=!memcmp(src,"HGA1",4)||!memcmp(src,"HGA2",4),row_entropy=!memcmp(src,"HGA1",4)||!memcmp(src,"HGT4",4),joint=regional||independent||alternative||row_entropy||!memcmp(src,"HGT3",4),wave=joint||!memcmp(src,"HGT2",4),learned=wave||!memcmp(src,"HGT1",4);require(learned||!memcmp(src,"HGE1",4));Bytes f(src,src+n);packed_bank::Reader r{f};r.skip(4);auto hn=r.u(4),zn=r.u(4);require(hn<=1024*1024);
    auto hb=packed_bank::decompress(r.take(zn),hn);packed_bank::Reader h{hb};Decoder a;a.row_entropy=row_entropy;
    a.s.d=h.u(4);a.s.mode=h.u(4);choice_grid::Scope grid(choice_grid::wire_edge(a.s.mode));a.s.mode&=255;a.G=h.u(4);a.K=h.u(4);a.B=h.u(4);unsigned nm=h.u(4),nc=h.u(4);
    require(ng_geometry_valid(a.s.d,a.s.d,a.s.d)&&cap>=size_t(a.s.d)*a.s.d*a.s.d&&a.s.mode<=1&&(learned?(a.G==16||a.G==32||a.G==64):(a.G==1||a.G==4))&&a.K>=2&&a.K<=32&&a.B>=1&&a.B<=4&&nm<=4&&nc>=1&&nc<=60);
    GeometryScope scope(a.s.d,a.s.d,a.s.d);a.out=out;
    for(unsigned m=0;m<nm;m++){auto par=h.take(h.u(4));a.states.emplace_back(m?ng_causal_create_model(par.data(),par.size()):ng_causal_create(par.data(),par.size()),ng_causal_destroy);require(bool(a.states.back()));}
    if(wave&&!nm){Bytes par(4096);size_t used=0;require(ng_bank_empty_model(par.data(),par.size(),&used)==0);a.states.emplace_back(ng_causal_create(par.data(),used),ng_causal_destroy);require(bool(a.states.back()));}
    for(unsigned c=0;c<nc;c++){
        choice_frame::Choice v;v.predictor=h.u(1);v.model=h.u(1);v.entropy=h.u(1);for(auto&w:v.weights)w=int32_t(packed_bank::signed_value(h.u(4),32));
        require(v.predictor<=12&&(v.model==255||v.model<nm)&&v.entropy<a.B);if(v.predictor==12)require(v.model==255);a.s.menu.push_back(v);
    }
    unsigned singleton=h.u(4);a.s.selectors.resize(N/choice_grid::volume());
    if(singleton==255){auto v=h.take(h.u(4));unpack_stream(v.data(),v.size(),a.s.selectors,true);}
    else {require(singleton<nc);std::fill(a.s.selectors.begin(),a.s.selectors.end(),singleton);}
    for(auto c:a.s.selectors)require(c<nc);
    std::vector<uint16_t>thr(a.K-1);for(auto&v:thr)v=h.u(2);require(std::is_sorted(thr.begin(),thr.end()));
    if(alternative){a.alternative=entropy_alternatives::read(h,a.s.d,a.B);require(a.G==(a.alternative.kind>=16?32:a.alternative.kind>=14?64:16));require(a.alternative.kind<14||a.alternative.kind>=16||a.K<=16);if(a.alternative.kind>=16)require(a.alternative.score.K==a.K);require(row_entropy==(a.alternative.kind!=17));for(unsigned c=0;c<a.K;c++)a.coarse_classes[c]=std::min(3u,c*4/a.K);}
    else if(learned){a.trees=entropy_tree::read(h,a.B,a.G);for(auto&t:a.trees)a.compiled.emplace_back(t);if(wave)a.vector=std::make_unique<entropy_tree::VectorForest>(a.trees);}
    unsigned E=a.B;if(regional){E=h.u(4);require(E&&E*a.G<=256);a.s.regions=choice_frame::read_regions(h,a.s.d,E);}
    if(row_entropy&&!alternative)require(!(a.vector->used&~entropy_tree::row_features));
    if(wave&&!alternative&&((wave_pipeline>=7&&(!(a.vector->used&~511u)||(bulk_rows_enabled&&!(a.vector->used&~entropy_tree::row_features))))||row_entropy))a.masks=std::make_unique<entropy_tree::MaskForest>(a.trees);
    a.prepare_row_lookup();
    if(profile_enabled){auto now=std::chrono::steady_clock::now();profile[0]=std::chrono::duration<double>(now-tick).count();tick=now;}
    size_t total=0;
    if(joint){unsigned tb=h.u(4);auto tables=h.take(h.u(4));
        if(independent){
            a.independent_waves=true;unsigned count=h.u(4);require(count&&count<=a.s.d*a.s.d);a.wave_payloads.reserve(count);
            for(unsigned i=0;i<count;i++){a.wave_payloads.push_back(r.take(h.u(4)));auto&payload=a.wave_payloads.back();require(payload.size()>=128);
                if(!i){a.wave_table_owner.reset(ng_context_rans_create(tables.data(),tables.size(),a.B*a.G,tb,payload.data(),payload.size()));require(bool(a.wave_table_owner));}
                auto*c=static_cast<ng_context_rans*>(std::malloc(sizeof(ng_context_rans)));require(c);*c=*static_cast<ng_context_rans*>(a.wave_table_owner.get());c->table=nullptr;c->d.p=payload.data()+128;c->d.end=payload.data()+payload.size();c->d.over=0;
                a.wave_entropies.emplace_back(c,ng_context_rans_destroy);
                for(unsigned j=0;j<32;j++){c->d.S[j]=uint32_t(payload[j*4])|(uint32_t(payload[j*4+1])<<8)|(uint32_t(payload[j*4+2])<<16)|(uint32_t(payload[j*4+3])<<24);require(c->d.S[j]>=(1u<<15));}
            }
        }else{a.joint_payload=r.take(h.u(4));a.entropy.reset(ng_context_rans_create(tables.data(),tables.size(),E*a.G,tb,a.joint_payload.data(),a.joint_payload.size()));require(bool(a.entropy));}total=N;}
    else for(unsigned g=0;g<a.G*(learned?1:a.K)*a.B;g++){a.streams.push_back(small_bank::read_stream(h,r,N-total));total+=a.streams.back().size();}
    if(joint&&(remaining_mode&1))a.compact_entropy=std::make_unique<TreeCompactEntropy>(*static_cast<ng_context_rans*>(independent?a.wave_table_owner.get():a.entropy.get()));
    require(total==N);h.end();r.end();for(auto&stream:a.streams)a.cursors.push_back({stream.data(),stream.empty()?stream.data():stream.data()+stream.size()});Bytes residual(profile_enabled?N:0);
    if(profile_enabled){auto now=std::chrono::steady_clock::now();profile[1]=std::chrono::duration<double>(now-tick).count();tick=now;}
    // Reuse native selected-model row preparation and reconstruction. The
    // callback only supplies symbols, never computes another predictor.
    void*models[48]={};int32_t weights[96]={};bool assigned[48]={};
    for(unsigned m=0;m<a.states.size();m++)for(unsigned b=0;b<12;b++)models[12*m+b]=a.states[m].get();
    for(auto&c:a.s.menu)if(c.model!=choice_frame::no_model){
        unsigned id=12*c.model+c.predictor;
        for(unsigned j=0;j<2;j++){require(!assigned[id]||weights[2*id+j]==c.weights[j]);weights[2*id+j]=c.weights[j];}assigned[id]=true;
    }
    choice_frame::normalize_unused_weights(a.states.size(),assigned,weights);
    Bytes canonical(a.s.selectors.size());
    for(size_t i=0;i<canonical.size();i++){auto&c=a.s.menu[a.s.selectors[i]];canonical[i]=c.predictor==choice_frame::zero_predictor?0:c.model==choice_frame::no_model?c.predictor:12+12*c.model+c.predictor;}
    if(wave){bool row_ready=row_batch_enabled;for(auto&t:a.trees)for(auto&n:t.nodes)if(n.feature!=255&&!(entropy_tree::row_features&(1u<<n.feature)))row_ready=false;
        if(a.alternative.kind==17)row_ready=false;
        bool bulk=!regional&&!independent&&(row_entropy||(row_ready&&bulk_rows_enabled&&wave_pipeline>=7));
        if(profile_enabled)ng_bank_detail_mark(-1);
        // Joint streams require entropy callbacks even in the unoptimized
        // pipeline: non-4 block sizes otherwise enter full-row reconstruction
        // without consuming symbols. wave_generic supports pipeline zero.
        ng_route_context context={Decoder::spatial,&a,thr.data(),a.K,profile_enabled?residual.data():nullptr,Decoder::batch,int(row_ready&&!joint),!bulk&&(wave_pipeline||joint)?(independent?Decoder::wave_independent:Decoder::wave):nullptr,wave_pipeline>=5,wave_pipeline==19,bulk?Decoder::bulk:nullptr,independent?Decoder::barrier:nullptr};require(ng_bank_route_wave(models,12*a.states.size(),a.s.mode,weights,canonical.data(),out,&context)==0);}
    else require(ng_bank_route_decode(models,std::max(1u,unsigned(a.states.size()))*12,a.s.mode,weights,canonical.data(),out,thr.data(),a.K,Decoder::route,&a,profile_enabled?residual.data():nullptr)==0);
    for(auto&cursor:a.cursors)require(cursor.next==cursor.end);
    if(independent)require(!a.pending_count&&a.wave_index==a.wave_entropies.size()&&a.wave_finished==a.wave_index);
    else if(joint)require(ng_context_rans_finish(a.entropy.get())==0);
    if(profile_enabled){
        auto now=std::chrono::steady_clock::now();profile[2]=std::chrono::duration<double>(now-tick).count();
        // Standalone routing replay: diagnostic only, not part of decode.
        // Cached complete input makes this an optimistic kernel measurement,
        // NOT a subtractable stage of the interleaved reconstruction above.
        std::vector<uint16_t>activity(N);require(ng_activity_values(residual.data(),activity.data())==0);
        auto cls=activity_classify(activity,thr);uint64_t checksum=0,active=0,decisions=0;
        tick=std::chrono::steady_clock::now();
        for(size_t i=0;i<N;i++){
            auto&c=a.s.menu[a.s.selectors[geometry_trials::block(i,a.s.d)]];
            if(c.predictor==choice_frame::zero_predictor)continue;
            if(alternative)checksum+=a.alternative.group(i,a.s.d,cls[i],a.K,c.entropy,out);
            else if(learned)checksum+=a.compiled[c.entropy].classify(out,i,a.s.d,i%a.s.d,i/a.s.d%a.s.d,i/(a.s.d*a.s.d),cls[i],c.predictor);
            else if(a.G>1)checksum+=geometry_trials::entropy_shape(out,i%a.s.d,i/a.s.d%a.s.d,i/(a.s.d*a.s.d),a.s.d);
            else checksum+=cls[i];
        }
        profile[3]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();profile[4]=double(checksum);
        for(size_t i=0;i<N;i++){
            auto&c=a.s.menu[a.s.selectors[geometry_trials::block(i,a.s.d)]];
            if(c.predictor==choice_frame::zero_predictor)continue;++active;
            if(learned&&!alternative){unsigned node=0;auto&t=a.trees[c.entropy];while(t.nodes[node].feature!=255){++decisions;auto&v=t.nodes[node];node=entropy_tree::feature(out,i,a.s.d,cls[i],c.predictor,v.feature)<=v.cut?v.left:v.right;}}
        }
        profile[5]=active?double(decisions)/active:0;profile[6]=double(active)/N;
        unsigned leaves=0;for(auto&t:a.trees)for(auto&v:t.nodes)leaves+=v.feature==255;profile[7]=leaves;
    }
}
}
extern "C" void* ng_hybrid_geometry_create(const uint8_t*s,unsigned d,unsigned preset){try{return hybrid_geometry::create(s,d,preset);}catch(const std::exception&e){fprintf(stderr,"hybrid geometry create: %s\n",e.what());return nullptr;}}
extern "C" void* ng_hybrid_predictor_create(const uint8_t*s,unsigned d,unsigned variant){try{return hybrid_geometry::create(s,d,1,variant);}catch(const std::exception&e){fprintf(stderr,"hybrid predictor create: %s\n",e.what());return nullptr;}}
// Read-only selected-block/model counts, to verify actual search integration.
extern "C" int ng_hybrid_predictor_usage(void*p,uint64_t*out){
    if(!p||!out)return -1;auto&t=*static_cast<hybrid_geometry::Trial*>(p);std::fill_n(out,6,0);
    for(auto i:t.s.selectors){auto&c=t.s.menu.at(i);out[c.predictor==choice_frame::zero_predictor?0:c.model==choice_frame::no_model?1:2]++;}
    out[3]=t.s.models.size();for(auto&m:t.s.models)out[4]+=m.size();out[5]=t.s.menu.size();return 0;
}
extern "C" void ng_hybrid_geometry_destroy(void*p){delete static_cast<hybrid_geometry::Trial*>(p);}
extern "C" int ng_entropy_tree_restrict(void*p,unsigned mask){
    if(!p||!mask||(mask&~((1u<<entropy_tree::features)-1)))return -1;
    auto&t=*static_cast<hybrid_geometry::Trial*>(p);
    if(t.feature_mask!=mask){t.feature_mask=mask;for(auto&f:t.trees)f.clear();}
    return 0;
}
extern "C" int ng_hybrid_geometry_encode(void*p,unsigned mode,uint8_t*out,size_t cap,size_t*n){try{require(p&&out&&n);auto f=hybrid_geometry::encode(*static_cast<hybrid_geometry::Trial*>(p),mode);require(f.size()<=cap);memcpy(out,f.data(),f.size());*n=f.size();return 0;}catch(...){return -1;}}
extern "C" int ng_hybrid_geometry_decode(const uint8_t*s,size_t n,uint8_t*out,size_t cap){try{hybrid_geometry::decode(s,n,out,cap);return 0;}catch(...){return -1;}}
extern "C" int ng_entropy_tree_predicates_exact(){return entropy_tree::compiled_predicates_exact()?0:-1;}
extern "C" int ng_entropy_tree_row_features_exact(const uint8_t*s,unsigned d){
    if(!s||d!=32)return -1;
    Bytes changed(s,s+size_t(d)*d*d);
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++){
        size_t start=(size_t(z)*d+y)*d;
        for(unsigned x=0;x<d;x++)changed[start+x]=s[start+x]^255;
        for(unsigned x=0;x<d;x++)for(unsigned f=9;f<entropy_tree::features;f++){
            unsigned bin=entropy_tree::feature(s,start+x,d,0,0,f);
            if(bin>=32||bin!=entropy_tree::feature(changed.data(),start+x,d,0,0,f))return -1;
            int raw=entropy_tree::row_feature(s,d,x,y,z,f);
            for(unsigned cut=0;cut<31;cut++)if((bin<=cut)!=(raw<=entropy_tree::CompiledTree::threshold(f,cut)))return -1;
        }
        memcpy(changed.data()+start,s+start,d);
    }
    return 0;
}
extern "C" void ng_entropy_tree_vector_enable(int enabled){hybrid_geometry::vector_enabled=enabled!=0;}
extern "C" void ng_entropy_tree_row_batch_enable(int enabled){hybrid_geometry::row_batch_enabled=enabled!=0;}
extern "C" void ng_entropy_tree_wave_pipeline(unsigned mode){hybrid_geometry::wave_pipeline=mode;}
extern "C" void ng_entropy_tree_acceleration(unsigned flags){hybrid_geometry::acceleration=flags;}
extern "C" void ng_entropy_tree_remaining_mode(unsigned mode){hybrid_geometry::remaining_mode=mode;}
extern "C" void ng_entropy_tree_wave_batch_counts(uint64_t*out){memcpy(out,hybrid_geometry::wave_batch_counts,sizeof(hybrid_geometry::wave_batch_counts));memset(hybrid_geometry::wave_batch_counts,0,sizeof(hybrid_geometry::wave_batch_counts));}
extern "C" int ng_entropy_tree_encode_independent(void*p,uint8_t*out,size_t cap,size_t*n){
    struct Scope{bool old=hybrid_geometry::independent_wave_encode;Scope(){hybrid_geometry::independent_wave_encode=true;}~Scope(){hybrid_geometry::independent_wave_encode=old;}}scope;
    return ng_hybrid_geometry_encode(p,7,out,cap,n);
}
extern "C" void ng_entropy_tree_specialized_profile(unsigned on){hybrid_geometry::specialized_profile=on;std::fill_n(hybrid_geometry::wave_parts,8,0.);}
extern "C" void ng_entropy_tree_bulk_rows(int enabled){hybrid_geometry::bulk_rows_enabled=enabled;}
extern "C" void ng_entropy_tree_bulk_parts(double*out){std::copy_n(hybrid_geometry::bulk_profile,3,out);}
extern "C" void ng_entropy_tree_replay(unsigned mode){hybrid_geometry::replay_mode=mode;}
extern "C" void ng_entropy_tree_wave_parts(double*out){if(out)std::copy_n(hybrid_geometry::wave_parts,8,out);}
extern "C" int ng_entropy_tree_features_exact(const uint8_t*s,unsigned d){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if(!__builtin_cpu_supports("avx2"))return 0;
    if(!s||!d||d%32)return -1;
    entropy_tree::VectorForest forest({});forest.used=1u<<6;
    for(unsigned row=0;row<d*d;row+=32)for(unsigned x=0;x<d;x++){
        uint8_t values[8][32];const uint8_t*neighbors[8];int ys[32],zs[32],f[9*32]={};
        for(unsigned k=0;k<8;k++)neighbors[k]=values[k];
        constexpr int dx[]={-1,-2,0,0,-1,-1,0,-1},dy[]={0,0,-1,0,-1,0,-1,-1},dz[]={0,0,0,-1,0,-1,-1,-1};
        for(unsigned j=0;j<32;j++){
            ys[j]=(row+j)%d;zs[j]=(row+j)/d;f[j]=j%16;f[256+j]=j%12;
            for(unsigned k=0;k<8;k++)values[k][j]=geometry_trials::value(s,int(x)+dx[k],ys[j]+dy[k],zs[j]+dz[k],d);
        }
        forest.make_features(32,x,ys,zs,neighbors,f);
        if(__builtin_cpu_supports("avx512bw")){
            int wide[9*32];memcpy(wide,f,sizeof(wide));forest.make_features512(32,x,ys,zs,neighbors,wide);
            if(memcmp(wide,f,sizeof(wide)))return -1;
        }
        for(unsigned j=0;j<32;j++)for(unsigned k=0;k<9;k++){
            int v=f[32*k+j];unsigned bin=k==1||k==2||k==3||k==6?entropy_tree::signed_bin(v):k==4||k==5?entropy_tree::magnitude(v):k==7?v/24:v;
            if(bin!=entropy_tree::feature(s,size_t(row+j)*d+x,d,j%16,j%12,k))return -1;
        }
    }
#endif
    return 0;
}
extern "C" int ng_hybrid_geometry_profile(const uint8_t*s,size_t n,uint8_t*out,size_t cap,double*stats){
    if(!stats)return -1;bool old=hybrid_geometry::profile_enabled;hybrid_geometry::profile_enabled=true;std::fill_n(hybrid_geometry::profile,8,0.);std::fill_n(hybrid_geometry::wave_parts,8,0.);std::fill_n(hybrid_geometry::bulk_profile,3,0.);
    int detail=ng_bank_detail_enabled;ng_bank_detail_enabled=1;std::fill_n(bank_detail,6,0.);
    int rc=ng_hybrid_geometry_decode(s,n,out,cap);std::copy_n(hybrid_geometry::profile,8,stats);hybrid_geometry::profile_enabled=old;ng_bank_detail_enabled=detail;return rc;
}
extern "C" void ng_entropy_tree_native_parts(double*out){std::copy_n(bank_detail,6,out);}
