/* Experimental one-to-four-model alternating regression/selection. Uses the native NG
 * fitter, candidate evaluator, entropy coder and scalar reconstruction helper.
 * NMB1 is deliberately separate from the production single-model NWP1 format.
 * One-bank control uses the identical framing/search for a matched comparison. */
#include <string_view>
namespace small_bank {
using packed_bank::Reader;
using packed_bank::append;
using packed_bank::compress;
using packed_bank::decompress;
using Weight=std::array<int32_t,2>;

struct Fit {
    Bytes params;
    Weight weights;
    std::vector<std::vector<uint16_t>> base;
    Bytes initial_symbols;
};
// Optional encoder stream experiment: retain only model parameters, never
// source/residual arrays. Ordinary per-chunk fitting is unchanged when null.
struct StreamFits {size_t voxels=0;bool refresh=false,support_only=false;std::map<Bytes,Fit> models;uint64_t reused=0,fitted=0;};
thread_local StreamFits*stream_fits=nullptr;
bool stream_reuse(Fit&,const uint8_t*,const int*,const Bytes*,const uint8_t*,const std::vector<Bytes>*,unsigned);
void stream_save(const Fit&,const int*,unsigned);
// Experimental per-input search cache: exact key equality, never hash-only.
// No default behavior change, no files, no cross-chunk lifetime. This shares
// identical fits across candidate grids without sharing differing fit masks.
namespace memo {
struct Entry{size_t hash;Bytes key;Fit value;size_t bytes;};
thread_local bool enabled=false;
thread_local std::vector<Entry> entries;
thread_local size_t used=0,hits=0,misses=0;
constexpr size_t limit=size_t(256)*1024*1024;
size_t bytes(const Fit&f){size_t n=f.params.size()+f.initial_symbols.size();for(auto&v:f.base)n+=v.size()*sizeof(uint16_t);return n;}
Bytes key(unsigned kind,const uint8_t*src,const int*cfg,const uint8_t*mask,const Bytes*feature,unsigned input_mask,bool retain,const std::vector<Bytes>*fixed){
    if(!enabled)return {};
    // Halo experiments change calibration history with the warm-up mask.
    // Cache only their ordinary proxy fits, never a partially evaluated fit.
    if(ng_screen_work&&ng_screen_size==N)return {};
    uint8_t signature[128];size_t n=ng_fit_settings_signature(signature,sizeof(signature));if(!n)return {};
    Bytes k(signature,signature+n);put32(k,kind);put32(k,NG_X);put32(k,NG_Y);put32(k,NG_Z);
    for(unsigned i=0;i<6;i++)put32(k,cfg[i]);put32(k,input_mask);put32(k,retain);
    k.insert(k.end(),src,src+N);k.push_back(mask!=nullptr);if(mask)k.insert(k.end(),mask,mask+N);
    if(feature)k.insert(k.end(),feature->begin(),feature->end());
    if(fixed)for(unsigned m=0;m<12;m++)if(input_mask&(1u<<m))k.insert(k.end(),(*fixed)[m].begin(),(*fixed)[m].end());
    return k;
}
size_t hash(const Bytes&k){return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(k.data()),k.size()));}
const Fit*get(const Bytes&key){if(key.empty())return nullptr;auto h=hash(key);for(auto&e:entries)if(e.hash==h&&e.key==key){++hits;return &e.value;}++misses;return nullptr;}
void save(Bytes&&key,const Fit&fit){if(key.empty())return;size_t n=key.size()+bytes(fit);if(n>limit)return;
    while(used+n>limit&&!entries.empty()){used-=entries.front().bytes;entries.erase(entries.begin());}
    auto h=hash(key);entries.push_back({h,std::move(key),fit,n});used+=n;
}
}
Fit plain_fit(const uint8_t*src,const int*cfg,const uint8_t*mask=nullptr,const Fit*previous=nullptr,const uint8_t*sample_weights=nullptr){
    Fit reused;if(!previous&&!sample_weights&&stream_reuse(reused,src,cfg,nullptr,mask,nullptr,24))return reused;
    Bytes sampled_mask;mask=subset_targets::fit_mask(src,mask,sampled_mask);
    Bytes cache_key=previous||sample_weights?Bytes{}:memo::key(0,src,cfg,mask,nullptr,0,true,nullptr);
    if(auto*f=memo::get(cache_key))return *f;
    Fit f;f.weights={0,0};f.params.resize(65536);f.initial_symbols.resize(N);
    Bytes classes(N);f.base.resize(12);f.base[0].resize(N);size_t n=0;
    if(sample_weights)require(ng_prepare_weighted(src,cfg,nullptr,mask,sample_weights,nullptr,
        f.params.data(),f.params.size(),&n,f.initial_symbols.data(),classes.data(),f.base[0].data())==0);
    else if(previous)require(ng_prepare_damped(src,cfg,nullptr,mask,previous->params.data(),previous->params.size(),nullptr,nullptr,
        f.params.data(),f.params.size(),&n,f.initial_symbols.data(),classes.data(),f.base[0].data())==0);
    else require(ng_plain_prepare_masked(src,cfg,mask,f.params.data(),f.params.size(),&n,
        f.initial_symbols.data(),classes.data(),f.base[0].data())==0);
    f.params.resize(n);remove_class_bias(f.base[0],f.params,classes,f.params.at(4));memo::save(std::move(cache_key),f);if(!previous&&!sample_weights)stream_save(f,cfg,24);return f;
}
// Shared native fitting and feature substitution, used by both bank searches.
void feature_bases(Fit&f,const uint8_t*src,unsigned d,const Bytes&feature,const std::vector<Bytes>&fixed,unsigned input_mask,std::vector<uint16_t>&prediction,const Bytes&classes){
    if(input_mask)remove_class_bias(prediction,f.params,classes,f.params.at(4));
    f.base.resize(12);
    for(unsigned m=0;m<12;m++)if(input_mask&(1u<<m)){
        f.base[m].resize(N);
        for(size_t i=0;i<N;i++){
            uint16_t w=uint16_t(f.weights[i<size_t(d)*d]);
            uint8_t v=uint8_t(src[i]-unpyramid(fixed[m][i]));
            f.base[m][i]=uint16_t(prediction[i]-unsigned(w)*feature[i]+unsigned(w)*v);
        }
    }
}
Fit fit(const uint8_t*src,unsigned d,const int*cfg,const Bytes&feature,
        const uint8_t*mask,const std::vector<Bytes>&fixed,unsigned input_mask=4095,bool retain_symbols=false,Bytes*reuse_symbols=nullptr,uint16_t*activity=nullptr,uint32_t*histogram=nullptr,void*session=nullptr,const Fit*previous=nullptr,const uint8_t*sample_weights=nullptr){
    unsigned stream_mode=0;while(stream_mode<12&&input_mask!=(1u<<stream_mode))++stream_mode;
    Fit reused;if(!previous&&!sample_weights&&!activity&&stream_mode<12&&stream_reuse(reused,src,cfg,&feature,mask,&fixed,stream_mode))return reused;
    Bytes sampled_mask;mask=subset_targets::fit_mask(src,mask,sampled_mask);
    Bytes cache_key=previous||sample_weights||reuse_symbols||activity?Bytes{}:memo::key(session?2:1,src,cfg,mask,&feature,input_mask,retain_symbols,&fixed);
    if(auto*f=memo::get(cache_key))return *f;
    Fit f;f.params.resize(65536);size_t n=0;
    const size_t voxels=N;
    Bytes symbols=reuse_symbols?std::move(*reuse_symbols):Bytes();symbols.resize(voxels);
    Bytes classes(input_mask?voxels:0);std::vector<uint16_t>prediction(input_mask?voxels:0);
    if(sample_weights){
        require(!session&&!activity&&input_mask&&!previous);
        require(ng_prepare_weighted(src,cfg,feature.data(),mask,sample_weights,f.weights.data(),
            f.params.data(),f.params.size(),&n,symbols.data(),classes.data(),prediction.data())==0);
    }else if(previous){
        require(!session&&!activity&&input_mask);
        require(ng_prepare_damped(src,cfg,feature.data(),mask,previous->params.data(),previous->params.size(),previous->weights.data(),f.weights.data(),
            f.params.data(),f.params.size(),&n,symbols.data(),classes.data(),prediction.data())==0);
    }else if(session){
        require(!mask&&!activity);
        require(ng_feature_prepare_cached(session,src,cfg,feature.data(),f.weights.data(),f.params.data(),f.params.size(),&n,symbols.data(),classes.data(),prediction.data())==0);
    }else if(activity){
        require(!mask&&!input_mask&&histogram);
        require(ng_feature_prepare_activity(src,cfg,feature.data(),f.weights.data(),f.params.data(),f.params.size(),&n,symbols.data(),activity,histogram)==0);
    }else require(ng_feature_prepare_masked(src,cfg,feature.data(),mask,1,f.weights.data(),
        f.params.data(),f.params.size(),&n,symbols.data(),classes.data(),prediction.data())==0);
    f.params.resize(n);feature_bases(f,src,d,feature,fixed,input_mask,prediction,classes);
    if(retain_symbols)f.initial_symbols=std::move(symbols);
    memo::save(std::move(cache_key),f);if(!previous&&!sample_weights&&stream_mode<12)stream_save(f,cfg,stream_mode);
    return f;
}

Fit refine(const uint8_t*src,unsigned d,const int*cfg,const Bytes*feature,const uint8_t*mask,const uint8_t*weights,const std::vector<Bytes>&fixed,unsigned mode,const Fit&old,const Fit*next=nullptr,unsigned shift=0){
    Fit f;f.weights={0,0};f.params.resize(65536);f.initial_symbols.resize(N);Bytes classes(N);std::vector<uint16_t>prediction(N);size_t n=0;
    require(ng_prepare_refinement(src,cfg,feature?feature->data():nullptr,mask,weights,old.params.data(),old.params.size(),old.weights.data(),
        next?next->params.data():nullptr,next?next->params.size():0,next?next->weights.data():nullptr,shift,f.weights.data(),
        f.params.data(),f.params.size(),&n,f.initial_symbols.data(),classes.data(),prediction.data())==0);
    f.params.resize(n);
    if(feature)feature_bases(f,src,d,*feature,fixed,1u<<mode,prediction,classes);
    else {remove_class_bias(prediction,f.params,classes,16);f.base.resize(12);f.base[0]=std::move(prediction);}
    return f;
}
Bytes stream_key(const int*cfg,unsigned mode){
    uint8_t signature[128];size_t n=ng_fit_settings_signature(signature,sizeof(signature));
    Bytes key(signature,signature+n);for(unsigned j=0;j<6;j++)put32(key,cfg[j]);put32(key,mode);put32(key,NG_X);put32(key,NG_Y);put32(key,NG_Z);return key;
}
bool stream_reuse(Fit&out,const uint8_t*src,const int*cfg,const Bytes*feature,const uint8_t*mask,const std::vector<Bytes>*fixed,unsigned mode){
    if(!stream_fits||N>stream_fits->voxels||stream_fits->refresh)return false;
    auto it=stream_fits->models.find(stream_key(cfg,mode));if(it==stream_fits->models.end())return false;
    static const std::vector<Bytes>empty;
    // A support-only replay fits coefficients against this chunk; only the
    // nonzero tap support comes from the stream. Never reuse residual arrays.
    out=refine(src,NG_X,cfg,feature,mask,nullptr,fixed?*fixed:empty,mode,it->second,
               stream_fits->support_only?nullptr:&it->second);
    ++stream_fits->reused;return true;
}
void stream_save(const Fit&f,const int*cfg,unsigned mode){
    if(!stream_fits||N>stream_fits->voxels)return;
    Fit compact;compact.params=f.params;compact.weights=f.weights;
    stream_fits->models[stream_key(cfg,mode)]=std::move(compact);++stream_fits->fitted;
}

void stream_header(Bytes&header,Bytes&body,const Bytes&encoded,size_t count){
    put32(header,count);
    if(!count){put32(header,0);put32(header,0);return;}
    uint16_t freq[256];size_t n=0;
    require(ng_read_freq_table(encoded.data(),encoded.size(),freq,&n)==0&&n<=encoded.size());
    put32(header,n);put32(header,encoded.size()-n);
    header.insert(header.end(),encoded.begin(),encoded.begin()+n);
    body.insert(body.end(),encoded.begin()+n,encoded.end());
}
Bytes read_stream(Reader&header,Reader&body,size_t limit){
    size_t count=header.u(4),tn=header.u(4),bn=header.u(4);require(count<=limit);
    auto table=header.take(tn);auto bytes=body.take(bn);append(table,bytes);
    Bytes out(count);if(count)unpack_stream(table.data(),table.size(),out,true);
    else require(!tn&&!bn);return out;
}
Bytes frame(unsigned d,const std::vector<Bytes>&models,const std::vector<Weight>&weights,
            const Bytes&sel,const Bytes&raw,const std::vector<uint16_t>&thr,const Bytes&classes){
    Bytes header,body;put32(header,d);put32(header,models.size());
    for(unsigned k=0;k<models.size();k++){
        put32(header,models[k].size());append(header,models[k]);
        for(auto w:weights[k])put32(header,uint32_t(w));
    }
    for(auto t:thr)packed_bank::put16(header,t);
    stream_header(header,body,pack_stream(sel,true),sel.size());
    auto payload=activity_stream_payload(raw,thr,classes);Reader r{payload};r.skip(30);
    for(unsigned c=0;c<16;c++){size_t count=r.u(4),n=r.u(4);stream_header(header,body,r.take(n),count);}
    r.end();auto z=compress(header);Bytes out{'N','M','B','1'};
    put32(out,header.size());put32(out,z.size());append(out,z);append(out,body);return out;
}
}
extern "C" int ng_small_bank_encode(const uint8_t*src,unsigned d,unsigned banks,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&out&&written&&banks>=1&&banks<=4&&ng_geometry_valid(d,d,d));GeometryScope scope(d,d,d);
        const unsigned count=12*banks,P=12+count,nb=d/4;const size_t blocks=size_t(nb)*nb*nb;
        const int cfg[6]={3,3,1,16,40,128};
        std::vector<Bytes> candidate(P,Bytes(N)),models(banks,Bytes(65536));
        bank_fixed_candidates(src,candidate,d);
        Bytes classes(N),symbols(N),raw(N),feature(N),membership(N),initial_group(blocks);
        std::vector<uint16_t>prediction(N),activity(N);std::vector<uint32_t>block(N);
        std::vector<small_bank::Weight> weights(banks);size_t pn=0;
        require(ng_ex_prepare(src,cfg,0,0,models[0].data(),models[0].size(),&pn,candidate[12].data(),classes.data(),prediction.data())==0);
        models[0].resize(pn);
        std::vector<const uint8_t*>ptr(P);
        for(unsigned m=0;m<P;m++)ptr[m]=candidate[m<12?m:12].data();
        auto sel=choose_blocks(ptr.data(),P,classes.data(),16,4);
        std::vector<uint64_t>energy(blocks);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;block[i]=((z/4)*nb+y/4)*nb+x/4;
            energy[block[i]]+=(candidate[12][i]+1u)/2;
        }
        auto ordered=energy;std::nth_element(ordered.begin(),ordered.begin()+blocks/2,ordered.end());
        for(size_t b=0;b<blocks;b++)initial_group[b]=banks==2&&(energy[b]>ordered[blocks/2]||(energy[b]==ordered[blocks/2]&&(b&1)));
        if(banks>2){
            // Equal-population energy groups; stable block-index tie breaking.
            // Preserve the existing two-model initialization for comparison.
            std::vector<size_t>rank(blocks);
            for(size_t b=0;b<blocks;b++)rank[b]=b;
            std::sort(rank.begin(),rank.end(),[&](size_t a,size_t b){return energy[a]!=energy[b]?energy[a]<energy[b]:a<b;});
            for(size_t j=0;j<blocks;j++)initial_group[rank[j]]=j*banks/blocks;
        }
        std::vector<std::vector<uint16_t>>base(count,std::vector<uint16_t>(N));
        std::vector<const uint8_t*>pp(count);std::vector<size_t>sizes(count);
        std::vector<const uint16_t*>bp(count);std::vector<uint8_t*>wp(P);
        for(unsigned m=0;m<P;m++){ptr[m]=candidate[m].data();wp[m]=candidate[m].data();}
        Bytes best,bestsel=sel;
        for(unsigned round=0;round<3;round++){
            for(size_t i=0;i<N;i++){
                unsigned c=bestsel[block[i]],m=c<12?c:(c-12)%12;
                feature[i]=uint8_t(src[i]-unpyramid(candidate[m][i]));
            }
            for(unsigned k=0;k<banks;k++){
                size_t members=0;
                for(size_t i=0;i<N;i++){
                    unsigned c=bestsel[block[i]],group=round&&c>=12?(c-12)/12:initial_group[block[i]];
                    membership[i]=banks==1||group==k;members+=membership[i];
                }
                auto fitted=small_bank::fit(src,d,cfg,feature,members>=1024?membership.data():nullptr,candidate);
                models[k]=std::move(fitted.params);weights[k]=fitted.weights;
                for(unsigned m=0;m<12;m++){
                    const unsigned id=k*12+m;base[id]=std::move(fitted.base[m]);
                    pp[id]=models[k].data();sizes[id]=models[k].size();bp[id]=base[id].data();
                }
            }
            sel=bestsel;
            for(unsigned pass=0;pass<3;pass++){
                require(ng_bank_symbols(count,pp.data(),sizes.data(),bp.data(),sel.data(),src,wp.data(),symbols.data(),activity.data())==0);
                for(size_t i=0;i<N;i++)raw[i]=unpyramid(symbols[i]);
                auto thr=activity_quantiles(activity,16),tuned=activity_refine(raw,activity,thr,block,sel,1);
                ActivityHistogram hist(raw,activity,true);if(hist.estimate(tuned)<hist.estimate(thr))thr=tuned;
                classes=activity_classify(activity,thr);
                auto encoded=small_bank::frame(d,models,weights,sel,raw,thr,classes);
                if(best.empty()||encoded.size()<best.size()){best=std::move(encoded);bestsel=sel;}
                if(pass<2){auto previous=sel;activity_reselect(symbols,classes,block,ptr,sel,1);if(sel==previous)break;}
            }
        }
        require(best.size()<=cap);memcpy(out,best.data(),best.size());*written=best.size();return 0;
    }catch(...){return -1;}
}
#include "ng_incremental_bank.hpp"
namespace small_bank {
struct ParsedFrame {unsigned d;std::vector<Bytes>models;std::vector<Weight>weights;std::vector<uint16_t>thr;Bytes selectors,raw;};
ParsedFrame read_frame(const uint8_t*src,size_t n,size_t cap){
        require(src&&n>=12&&!memcmp(src,"NMB1",4));Bytes f(src,src+n);small_bank::Reader r{f};r.skip(4);
        size_t hn=r.u(4),zn=r.u(4);auto header=small_bank::decompress(r.take(zn),hn);small_bank::Reader h{header};
        unsigned d=h.u(4),banks=h.u(4);require(ng_geometry_valid(d,d,d)&&banks>=1&&banks<=4&&cap>=size_t(d)*d*d);
        GeometryScope scope(d,d,d);
        ParsedFrame result;result.d=d;result.raw.resize(N);
        for(unsigned k=0;k<banks;k++){
            result.models.push_back(h.take(h.u(4)));std::array<int32_t,2>w;
            for(auto&v:w){uint32_t bits=h.u(4);v=bits<=INT32_MAX?int32_t(bits):int32_t(int64_t(bits)-4294967296LL);}
            result.weights.push_back(w);
        }
        std::vector<uint16_t>thr(15);for(auto&t:thr)t=h.u(2);
        require(std::is_sorted(thr.begin(),thr.end()));auto sel=small_bank::read_stream(h,r,N/64);require(sel.size()==N/64);
        for(auto m:sel)require(m<12+12*banks);
        std::array<Bytes,16>streams;const uint8_t*ptr[16];size_t counts[16],pos[16]={};size_t total=0;
        for(unsigned c=0;c<16;c++){streams[c]=small_bank::read_stream(h,r,N-total);counts[c]=streams[c].size();total+=counts[c];ptr[c]=streams[c].data();}
        require(total==N);h.end();r.end();require(ng_activity_gather(thr.data(),16,ptr,counts,pos,result.raw.data())==0);
        for(unsigned c=0;c<16;c++)require(pos[c]==counts[c]);
        result.thr=std::move(thr);result.selectors=std::move(sel);return result;
}
}
extern "C" int ng_small_bank_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    try{
        require(out);auto f=small_bank::read_frame(src,n,cap);GeometryScope scope(f.d,f.d,f.d);unsigned banks=f.models.size();
        using State=std::unique_ptr<void,decltype(&ng_causal_destroy)>;std::vector<State>states;
        std::vector<small_bank::Weight>weights(12*banks);std::vector<void*>pointers(12*banks);
        for(unsigned k=0;k<banks;k++){
            auto&p=f.models[k];states.emplace_back(k?ng_causal_create_model(p.data(),p.size()):ng_causal_create(p.data(),p.size()),ng_causal_destroy);require(bool(states.back()));
            for(unsigned m=0;m<12;m++){pointers[12*k+m]=states.back().get();weights[12*k+m]=m?f.weights[k]:small_bank::Weight{0,0};}
        }
        std::copy(f.raw.begin(),f.raw.end(),out);
        require(ng_bank_reconstruct_many(pointers.data(),weights[0].data(),f.selectors.data(),out,12*banks)==0);return 0;
    }catch(...){return -1;}
}
