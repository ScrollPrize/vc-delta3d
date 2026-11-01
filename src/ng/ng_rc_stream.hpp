// Encoder-only RC stream experiments. Frames remain independent NCF1/HGT3.
namespace rc_stream {
struct State {
    unsigned config=0,edge=4;uint64_t chunks=0;
    std::vector<unsigned>menu;std::vector<uint16_t>thresholds;
    small_bank::StreamFits fits;
    std::map<std::vector<unsigned>,entropy_tree::Tree> trees;
};
unsigned identity(const choice_frame::Choice&c){
    if(c.model==255)return c.predictor;
    return c.predictor==0&&c.weights==small_bank::Weight{0,0}?24:12+c.predictor;
}
std::vector<unsigned>menu_of(const hybrid_geometry::Snapshot&s){
    std::vector<unsigned>v;for(auto&c:s.menu)if(c.predictor!=choice_frame::zero_predictor)v.push_back(identity(c));
    std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());return v;
}
std::vector<unsigned>bank_key(const hybrid_geometry::Snapshot&s,unsigned bank){
    std::vector<unsigned>key;for(auto&c:s.menu)if(c.entropy==bank&&c.predictor!=choice_frame::zero_predictor)key.push_back(identity(c));
    std::sort(key.begin(),key.end());return key;
}
// Refine terminal split nodes without increasing leaf count or decoder work.
// Both sides retain their leaf IDs. Keep old split unless empirical entropy
// improves on the current chunk; no source or residual is reused.
void refine_trees(hybrid_geometry::Trial&t){
    std::vector<entropy_tree::Feature>x;std::vector<std::vector<uint32_t>>ids;
    hybrid_geometry::tree_samples(t,x,ids);std::vector<double>nl(N+1);
    for(size_t i=1;i<=N;i++)nl[i]=i*std::log2(double(i));
    for(unsigned b=0;b<ids.size();b++){
        auto&tree=t.trees[2][b];std::vector<std::vector<uint32_t>>at(tree.nodes.size());
        for(auto i:ids[b]){unsigned n=0;while(tree.nodes[n].feature!=255){
            auto&v=tree.nodes[n];if(tree.nodes[v.left].feature==255&&tree.nodes[v.right].feature==255){at[n].push_back(i);break;}
            n=x[i][v.feature]<=v.cut?v.left:v.right;
        }}
        for(unsigned n=0;n<at.size();n++)if(at[n].size()>=512){
            auto&v=tree.nodes[n];auto best=entropy_tree::best_split(at[n],x,t.s.symbols,nl,t.feature_mask);
            auto cost=[&](unsigned f,unsigned cut){std::array<uint32_t,256>a{},b{};unsigned na=0;
                for(auto i:at[n])if(x[i][f]<=cut){a[t.s.symbols[i]]++;na++;}else b[t.s.symbols[i]]++;
                double c=nl[na]+nl[at[n].size()-na];for(unsigned j=0;j<256;j++)c-=nl[a[j]]+nl[b[j]];return c;};
            if(best.gain>0&&cost(best.feature,best.cut)<cost(v.feature,v.cut)){v.feature=best.feature;v.cut=best.cut;}
        }
    }
}
// flags:1 menu reuse,2 threshold freeze,4 local threshold refinement,
// 8 RC1 support reuse,16 tree reuse,32 terminal-split refinement,
// 64 menu challenger. A refresh (flags=0) performs the unchanged RC search.
Bytes encode(State&s,const uint8_t*src,unsigned d,unsigned flags){
    GeometryScope geometry(d,d,d);choice_grid::Scope grid(s.edge);
    struct Scope{
        small_bank::StreamFits*fits=small_bank::stream_fits;
        const std::vector<uint16_t>*thr=fast_split::stream_thresholds;
        unsigned mode=fast_split::stream_threshold_mode;
        ~Scope(){small_bank::stream_fits=fits;fast_split::stream_thresholds=thr;fast_split::stream_threshold_mode=mode;}
    }scope;
    s.fits.voxels=N;s.fits.support_only=true;s.fits.refresh=!(flags&8);
    small_bank::stream_fits=&s.fits;
    if((flags&6)&&!s.thresholds.empty()){
        fast_split::stream_thresholds=&s.thresholds;fast_split::stream_threshold_mode=flags&4?2:1;
    }
    auto menu=(flags&1)&&!s.menu.empty()?&s.menu:nullptr;
    std::unique_ptr<hybrid_geometry::Trial>t(hybrid_geometry::create(src,d,1,s.config?4:0,menu));
    if((flags&64)&&menu){
        auto alternative=s.menu;unsigned challenger=s.chunks%25;
        if(std::find(alternative.begin(),alternative.end(),challenger)==alternative.end()){
            if(alternative.size()==3)alternative[s.chunks%3]=challenger;else alternative.push_back(challenger);
            std::sort(alternative.begin(),alternative.end());
            std::unique_ptr<hybrid_geometry::Trial>other(hybrid_geometry::create(src,d,1,s.config?4:0,&alternative));
            if(other->s.frame.size()<t->s.frame.size())t=std::move(other);
        }
    }
    if(s.config==2&&(flags&16)){
        bool complete=true;entropy_tree::Forest forest;
        for(unsigned b=0;b<hybrid_geometry::banks(t->s);b++){
            auto it=s.trees.find(bank_key(t->s,b));if(it==s.trees.end()){complete=false;break;}forest.push_back(it->second);
        }
        if(complete){for(auto&f:t->trees)f=forest;if(flags&32)refine_trees(*t);}
    }
    auto frame=hybrid_geometry::encode(*t,s.config==2?7:2);
    auto learned=menu_of(t->s);
    // An empty chunk has no evidence about the nonzero stream. Preserve its
    // learned menu, thresholds and trees rather than poisoning the next chunk.
    if(!learned.empty()){
        s.menu=std::move(learned);s.thresholds=t->s.thresholds;
        if(s.config==2)for(unsigned b=0;b<hybrid_geometry::banks(t->s);b++)s.trees[bank_key(t->s,b)]=t->trees[2][b];
    }
    ++s.chunks;return frame;
}
}
extern "C" void*ng_rc_stream_create(unsigned config,unsigned edge){try{require(config<3);choice_grid::Scope grid(edge);auto*s=new rc_stream::State;s->config=config;s->edge=edge;return s;}catch(...){return nullptr;}}
extern "C" void*ng_rc_stream_clone(void*p){try{return new rc_stream::State(*static_cast<rc_stream::State*>(p));}catch(...){return nullptr;}}
extern "C" void ng_rc_stream_destroy(void*p){delete static_cast<rc_stream::State*>(p);}
extern "C" int ng_rc_stream_encode(void*p,const uint8_t*src,unsigned d,unsigned flags,uint8_t*out,size_t cap,size_t*n){try{
    require(p&&src&&out&&n);auto frame=rc_stream::encode(*static_cast<rc_stream::State*>(p),src,d,flags);
    require(frame.size()<=cap);std::copy(frame.begin(),frame.end(),out);*n=frame.size();return 0;
}catch(...){return -1;}}
