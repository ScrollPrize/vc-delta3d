/* Experimental actual-frame tap search. No alternate codec or Python fitting. */
#include <set>
namespace tap_exact {
struct Bank {std::vector<double>C,r,mean;std::vector<int>ids,initial;double mt=0;};
using Support=std::array<std::vector<int>,2>;
struct Node {Support s;Bytes frame;};
struct Search {
    unsigned d,predictor=0,round=0;Bytes source,baseline,dense,best;
    Bank banks[2];std::vector<Node>beam;std::set<Support>seen;
    std::array<std::array<int32_t,321>,2>q{};int32_t bias[2]={};
    uint64_t trials=0,ranked=0;bool done=false;
};
thread_local Search*active=nullptr;
void capture(int bank,int stage,const double*C,const double*r,const double*mean,const int*ids,int n,double mt){
    auto&b=active->banks[bank];
    if(!stage){b.C.assign(C,C+n*n);b.r.assign(r,r+n);b.mean.assign(mean,mean+n);b.ids.assign(ids,ids+n);b.mt=mt;}
    else {b.initial.clear();for(int k=0;k<n;k++){auto p=std::find(b.ids.begin(),b.ids.end(),ids[k]);require(p!=b.ids.end());b.initial.push_back(p-b.ids.begin());}}
}
int override_fit(int bank,int32_t*w,int32_t*bias){
    auto&b=active->banks[bank];for(int id:b.ids)w[id]=active->q[bank][id];*bias=active->bias[bank];return 1;
}
struct Scope {
    Search*old=active;ng_tap_capture_fn cap=ng_tap_capture;ng_tap_override_fn fit=ng_tap_override;
    int forced=fast_split::forced_uniform;
    Scope(Search*s){active=s;fast_split::forced_uniform=s->predictor;}
    ~Scope(){active=old;ng_tap_capture=cap;ng_tap_override=fit;fast_split::forced_uniform=forced;}
};
Bytes encode(Search&s,int stride=1){
    Bytes f(s.source.size()*4+1048576);size_t n=0;int cfg[6]={3,3,1,16,40,stride};
    require(ng_codec_encode_fast_config(s.source.data(),s.d,cfg,16,f.data(),f.size(),&n)==0);f.resize(n);return f;
}
double solve(Search&s,int bank,const std::vector<int>&support,bool quantize){
    auto&b=s.banks[bank];double gain=0;
    if(quantize)s.q[bank].fill(0);
    int rc=ng_tap_subset_solve(b.C.data(),b.r.data(),b.mean.data(),b.mt,b.ids.data(),b.ids.size(),support.data(),support.size(),
        quantize?s.q[bank].data():nullptr,&s.bias[bank],&gain);
    return rc? -HUGE_VAL:gain;
}
uint32_t random(uint32_t&r){r^=r<<13;r^=r>>17;r^=r<<5;return r;}
std::vector<Support> proposals(Search&s,const Support&parent,unsigned parent_index){
    std::vector<Support>out;uint32_t rng=0x823df719u+7919*s.round+31*parent_index;
    for(int bank=0;bank<2;bank++){
        auto&keep=parent[bank];int n=s.banks[bank].ids.size();if(keep.empty()||keep.size()==size_t(n))continue;
        std::vector<int>missing;for(int k=0;k<n;k++)if(!std::binary_search(keep.begin(),keep.end(),k))missing.push_back(k);
        std::vector<std::pair<double,std::vector<int>>>rank;
        for(unsigned drop=0;drop<keep.size();drop++)for(int add:missing){
            auto v=keep;v[drop]=add;std::sort(v.begin(),v.end());
            double g=solve(s,bank,v,false);s.ranked++;rank.push_back({-g,std::move(v)});
        }
        std::sort(rank.begin(),rank.end());
        for(unsigned k=0;k<std::min(size_t(24),rank.size());k++){auto v=parent;v[bank]=rank[k].second;out.push_back(std::move(v));}
        // Eight unranked singles and sixteen two-tap moves, deterministic.
        for(int k=0;k<24;k++){
            auto v=parent;unsigned a=random(rng)%keep.size(),b=random(rng)%missing.size();v[bank][a]=missing[b];
            if(k>=8&&keep.size()>1&&missing.size()>1){unsigned a2=(a+1+random(rng)%(keep.size()-1))%keep.size();
                unsigned b2=(b+1+random(rng)%(missing.size()-1))%missing.size();v[bank][a2]=missing[b2];}
            std::sort(v[bank].begin(),v[bank].end());out.push_back(std::move(v));
        }
    }
    return out;
}
}
extern "C" void*ng_tap_exact_create(const uint8_t*src,unsigned d){
    try{
        require(src&&ng_geometry_valid(d,d,d));auto s=std::make_unique<tap_exact::Search>();s->d=d;s->source.assign(src,src+size_t(d)*d*d);
        s->baseline.resize(s->source.size()*4+1048576);size_t n=0;
        require(ng_codec_encode(src,d,3,s->baseline.data(),s->baseline.size(),&n)==0);s->baseline.resize(n);
        s->predictor=fast_split::last_uniform;s->best=s->baseline;
        if(s->predictor<12){s->dense=s->baseline;s->done=true;return s.release();}
        tap_exact::Scope scope(s.get());ng_tap_capture=tap_exact::capture;
        s->dense=tap_exact::encode(*s);ng_tap_capture=nullptr;
        if(s->banks[0].ids.empty()){s->done=true;return s.release();}
        tap_exact::Support support={s->banks[0].initial,s->banks[1].initial};
        for(int bank=0;bank<2;bank++)require(tap_exact::solve(*s,bank,support[bank],true)!=-HUGE_VAL);
        ng_tap_override=tap_exact::override_fit;
        // Mandatory exact equality: cached dense-stat refit == ordinary fit.
        require(tap_exact::encode(*s)==s->dense);
        if(s->dense.size()<s->best.size())s->best=s->dense;
        s->beam.push_back({support,s->dense});s->seen.insert(support);return s.release();
    }catch(const std::exception&e){std::fprintf(stderr,"tap exact create: %s\n",e.what());return nullptr;}
}
extern "C" int ng_tap_exact_step(void*p){
    try{
        auto&s=*static_cast<tap_exact::Search*>(p);if(s.done)return 1;
        tap_exact::Scope scope(&s);ng_tap_override=tap_exact::override_fit;ng_tap_capture=nullptr;
        auto next=s.beam;unsigned parent=0;
        for(auto&node:s.beam)for(auto&support:tap_exact::proposals(s,node.s,parent++)){
            if(!s.seen.insert(support).second)continue;
            bool valid=true;for(int b=0;b<2;b++)valid&=tap_exact::solve(s,b,support[b],true)!=-HUGE_VAL;
            if(!valid)continue;
            auto frame=tap_exact::encode(s);s.trials++;
            if(frame.size()<s.best.size())s.best=frame;
            next.push_back({std::move(support),std::move(frame)});
        }
        std::sort(next.begin(),next.end(),[](const auto&a,const auto&b){return a.frame.size()!=b.frame.size()?a.frame.size()<b.frame.size():a.s<b.s;});
        if(next.size()>3)next.resize(3);
        bool changed=next.size()!=s.beam.size();for(unsigned i=0;i<std::min(next.size(),s.beam.size());i++)changed|=next[i].s!=s.beam[i].s;
        s.beam=std::move(next);s.round++;s.done=!changed;return s.done?1:0;
    }catch(const std::exception&e){std::fprintf(stderr,"tap exact step: %s\n",e.what());return -1;}
}
extern "C" int ng_tap_exact_result(void*p,unsigned which,uint8_t*out,size_t cap,size_t*n,uint64_t*stats){
    if(!p||which>2||!out||!n||!stats)return -1;auto&s=*static_cast<tap_exact::Search*>(p);
    const auto&f=which==0?s.baseline:which==1?s.dense:s.best;if(cap<f.size())return -1;
    memcpy(out,f.data(),f.size());*n=f.size();uint64_t a[]={s.round,s.trials,s.ranked,s.done,s.predictor};std::copy_n(a,5,stats);return 0;
}
extern "C" void ng_tap_exact_destroy(void*p){delete static_cast<tap_exact::Search*>(p);}
