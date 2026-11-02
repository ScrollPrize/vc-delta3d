#include "ng_frame.h"
extern "C" int vc_ng_preset_decode(const uint8_t*,size_t,uint8_t*,size_t);
/* NCF1: explicit predictor/model/entropy references; native wavefront payload.
 * The search policy is deliberately not part of the decoding contract. */
namespace choice_frame {
thread_local bool enabled=false;
thread_local bool packed_selectors=true;
// Optional predictor-bank x octree-depth -> entropy-bank map. Empty means
// ordinary sharing. Keep all16 difficulty bins; at most8 banks (128 contexts).
thread_local Bytes context_map;
// Independent entropy regions on a 2-voxel grid. Predictor choices retain
// their own grid; one region may therefore contain multiple predictors.
thread_local Bytes region_banks;
void write_regions(Bytes&header,const Bytes&map,unsigned d){
    choice_grid::Scope fine(2);adaptive_grid::Scope root(32);
    auto tree=adaptive_grid::serialize(map,d);auto data=pack_stream(tree,true);
    put32(header,tree.size());put32(header,data.size());packed_bank::append(header,data);
}
Bytes read_regions(packed_bank::Reader&h,unsigned d,unsigned banks){
    unsigned count=h.u(4);require(count<=2*size_t(d/2)*(d/2)*(d/2));
    auto data=h.take(h.u(4));Bytes tree(count);unpack_stream(data.data(),data.size(),tree,true);
    choice_grid::Scope fine(2);return adaptive_grid::deserialize(tree,d,32,banks);
}
// Scoped experiment only; returned frames are size placeholders, not decodable.
thread_local unsigned estimate_dimension=0;
struct Prepared {unsigned d=0;Bytes header,params,symbols,classes,complete;};
thread_local bool retain_estimates=false;
thread_local std::map<unsigned,Prepared> prepared;
Bytes finish_entropy(Bytes header,const Bytes&entropy,size_t n,size_t hn){
    header.insert(header.end(),entropy.begin(),entropy.begin()+hn);
    auto z=packed_bank::compress(header);Bytes result={'N','C',uint8_t(adaptive_grid::root?'A':'F'),'1'};
    put32(result,header.size());put32(result,z.size());packed_bank::append(result,z);
    result.insert(result.end(),entropy.begin()+hn,entropy.begin()+n);return result;
}
// 254 identifies packed IDs in the existing jointly Zstd-compressed header.
// 255 retains the legacy rANS representation; 0..59 remain uniform IDs.
Bytes pack_ids(const Bytes&ids,unsigned count){
    require(count&&count<=60);unsigned bits=0;while((1u<<bits)<count)++bits;
    Bytes out((ids.size()*bits+7)/8,0);
    for(size_t i=0;i<ids.size();i++){require(ids[i]<count);for(unsigned b=0;b<bits;b++)out[(i*bits+b)/8]|=((ids[i]>>b)&1u)<<((i*bits+b)%8);}
    return out;
}
Bytes unpack_ids(const Bytes&packed,size_t n,unsigned count){
    require(count&&count<=60);unsigned bits=0;while((1u<<bits)<count)++bits;
    require(packed.size()==(n*bits+7)/8);Bytes out(n);
    for(size_t i=0;i<n;i++){for(unsigned b=0;b<bits;b++)out[i]|=((packed[(i*bits+b)/8]>>((i*bits+b)%8))&1u)<<b;require(out[i]<count);}
    return out;
}
struct Choice { unsigned predictor,model,entropy; small_bank::Weight weights{}; };
constexpr unsigned no_model=255;
constexpr unsigned zero_predictor=12;
// Read-only experiment observer. No effect on search or production bytes.
using Observer=void(*)(unsigned,const std::vector<Bytes>&,const std::vector<Choice>&,
    const Bytes&,const Bytes&,const std::vector<uint16_t>&,const Bytes&,unsigned,const Bytes&);
thread_local Observer observer=nullptr;
Bytes write(unsigned d,const std::vector<Bytes>&models,const std::vector<Choice>&menu,
            const Bytes&selectors,const Bytes&symbols,const std::vector<uint16_t>&thresholds,
            const Bytes&classes,unsigned mode=1,const zero_blocks::Scan*zeros=nullptr,bool size_routed=false){
    require(!adaptive_grid::root&&context_map.empty()&&region_banks.empty()&&!size_routed);
    require(!estimate_dimension&&!subset_targets::enabled()&&packed_selectors);
    if(zeros&&zeros->any()){
        auto extended=menu;auto selected=selectors;unsigned bank=0;
        for(auto&c:menu)bank=std::max(bank,c.entropy+1);
        require(bank<4&&extended.size()<60);unsigned id=extended.size();
        extended.push_back({zero_predictor,no_model,bank,{}});
        for(size_t b=0;b<selected.size();b++)if(zeros->empty[b])selected[b]=id;
        for(size_t i=0;i<symbols.size();i++)if(!zeros->mask[i])require(symbols[i]==0);
        return write(d,models,extended,selected,symbols,thresholds,classes,mode);
    }
    require(models.size()<=4&&!menu.empty()&&menu.size()<=60&&selectors.size()==N/choice_grid::volume());
    const bool zero=std::all_of(selectors.begin(),selectors.end(),[&](auto id){return id<menu.size()&&menu[id].predictor==zero_predictor;});
    require(zero||(symbols.size()==N&&classes.size()==N));
    vc_ng_model model[4];vc_ng_choice choices[60];
    for(size_t i=0;i<models.size();++i)model[i]={models[i].data(),models[i].size()};
    for(size_t i=0;i<menu.size();++i)choices[i]={menu[i].predictor,menu[i].model,menu[i].entropy,{menu[i].weights[0],menu[i].weights[1]}};
    uint8_t*raw=nullptr;size_t size=0;
    require(vc_ng_frame_encode(d,choice_grid::edge(),mode,model,models.size(),choices,menu.size(),
        selectors.data(),symbols.data(),thresholds.data(),thresholds.size()+1,classes.data(),&raw,&size)!=0);
    std::unique_ptr<uint8_t,decltype(&std::free)> owner(raw,&std::free);
    Bytes result(raw,raw+size);
    if(observer)observer(d,models,menu,selectors,symbols,thresholds,classes,mode,result);
    return result;
}
// Unused palette entries must not disable the shared-fit SIMD path.
void normalize_unused_weights(unsigned count,const bool*assigned,int32_t*weights){
    for(unsigned k=0;k<count;k++){
        unsigned ref=12*k;for(unsigned m=1;m<12;m++)if(assigned[12*k+m]){ref=12*k+m;break;}
        for(unsigned m=1;m<12;m++)if(!assigned[12*k+m])for(unsigned j=0;j<2;j++)weights[2*(12*k+m)+j]=weights[2*ref+j];
    }
}
std::vector<Bytes> read_models(packed_bank::Reader&h,unsigned nm){
    require(nm<=4);std::vector<Bytes>params;for(unsigned k=0;k<nm;k++)params.push_back(h.take(h.u(4)));return params;
}
int decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    return vc_ng_preset_decode(src,n,out,cap);
}
Bytes empty_frame(unsigned d){return write(d,{},{{zero_predictor,no_model,0,{}}},Bytes(N/choice_grid::volume()),{},std::vector<uint16_t>(15),{});}
}
extern "C" int ng_codec_decode(const uint8_t*s,size_t n,uint8_t*out,size_t cap){try{return choice_frame::decode(s,n,out,cap);}catch(...){return -1;}}
extern "C" void ng_codec_packed_selectors(unsigned enabled){choice_frame::packed_selectors=enabled!=0;}
// Signaling-only control: identical models, partition, entropy tables and
// residual payload. Only the selector representation and joint header change.
extern "C" int ng_codec_repack_s2_signal(const uint8_t*s,size_t n,unsigned raw,uint8_t*out,size_t cap,size_t*written){try{
    require(s&&out&&written&&raw<=1&&n>=12&&(!memcmp(s,"NCF1",4)||!memcmp(s,"NCA1",4)));
    Bytes result(s,s+n);
    if(s[2]=='A'){
        Bytes prefix(s,s+12);packed_bank::Reader p{prefix};p.skip(4);unsigned hn=p.u(4),zn=p.u(4);require(zn<=n-12);
        auto header=packed_bank::decompress(s+12,zn,hn);packed_bank::Reader h{header};h.skip(4);unsigned mode=h.u(4),nm=h.u(4),nc=h.u(4);h.skip(4);require(nm<=4&&nc<=60);
        for(unsigned k=0;k<nm;k++)h.skip(h.u(4));h.skip(nc*(4+4*small_bank::Weight{}.size()));
        if(mode&4)h.skip(h.u(4));
        size_t selector_start=h.p;unsigned tag=h.u(4);
        if(tag==253||tag==255){
            unsigned root=h.u(4),count=h.u(4);require(count<=1048576);auto encoded=h.take(h.u(4));Bytes tree(count);
            if(tag==253){require(encoded.size()==count);tree=std::move(encoded);}else unpack_stream(encoded.data(),encoded.size(),tree,true);
            Bytes next(header.begin(),header.begin()+selector_start);put32(next,raw?253:255);put32(next,root);put32(next,count);
            auto body=raw?tree:pack_stream(tree,true);put32(next,body.size());packed_bank::append(next,body);packed_bank::append(next,h.rest());
            auto z=packed_bank::compress(next);result={'N','C','A','1'};put32(result,next.size());put32(result,z.size());packed_bank::append(result,z);result.insert(result.end(),s+12+zn,s+n);
        }
    }
    require(result.size()<=cap);memcpy(out,result.data(),result.size());*written=result.size();return 0;
}catch(...){return -1;}}
/* Read-only saved-choice diagnostic. Four entries per palette member:
 * predictor,model,entropy,count. Model hashes expose changes across grids. */
extern "C" int ng_codec_choice_info(const uint8_t*s,size_t n,uint64_t*out){try{
    require(s&&out&&n>=12&&!memcmp(s,"NCF1",4));std::fill_n(out,256,0);
    Bytes prefix(s,s+12);size_t pos=4;unsigned hn=get32(prefix,pos),zn=get32(prefix,pos);require(zn<=n-12);
    auto header=packed_bank::decompress(s+12,zn,hn);packed_bank::Reader h{header};
    unsigned d=h.u(4),edge=choice_grid::wire_edge(h.u(4)),nm=h.u(4),nc=h.u(4);h.skip(4);
    require(nm<=4&&nc>=1&&nc<=60);choice_grid::Scope scope(edge);
    out[0]=d;out[1]=edge;out[2]=nm;out[3]=nc;
    for(unsigned m=0;m<nm;m++){auto p=h.take(h.u(4));uint64_t hash=14695981039346656037ull;for(auto b:p)hash=(hash^b)*1099511628211ull;out[8+m]=hash;}
    for(unsigned c=0;c<nc;c++){out[16+4*c]=h.u(1);out[17+4*c]=h.u(1);out[18+4*c]=h.u(1);h.skip(1+4*small_bank::Weight{}.size());}
    unsigned single=h.u(4);Bytes ids(choice_grid::count(d));
    if(single==254){auto p=h.take(h.u(4));ids=choice_frame::unpack_ids(p,ids.size(),nc);}
    else if(single==255){auto p=h.take(h.u(4));unpack_stream(p.data(),p.size(),ids,true);}
    else {require(single<nc);std::fill(ids.begin(),ids.end(),single);}
    out[4]=ids.size();for(auto c:ids){require(c<nc);out[19+4*c]++;if(out[16+4*c]==choice_frame::zero_predictor)out[5]++;}
    return 0;
}catch(...){return -1;}}
/* Read-only byte attribution. Header is jointly compressed, so its model,
 * selector and entropy-table contributions cannot be added independently. */
extern "C" int ng_codec_inspect_sizes(const uint8_t*s,size_t n,uint64_t*out){try{
    require(s&&out&&n>=12&&!memcmp(s,"NCF1",4));Bytes f(s,s+12);packed_bank::Reader p{f};p.skip(4);
    size_t hn=p.u(4),zn=p.u(4);require(zn<=n-12&&hn<=1024*1024);
    auto header=packed_bank::decompress(s+12,zn,hn);packed_bank::Reader h{header};h.skip(8);unsigned models=h.u(4);require(models<=4);h.skip(8);
    uint64_t model_bytes=0;for(unsigned k=0;k<models;k++){size_t m=h.u(4);h.skip(m);model_bytes+=m;}
    out[0]=n;out[1]=12+zn;out[2]=n-12-zn;out[3]=hn;out[4]=model_bytes;out[5]=models;return 0;
}catch(...){return -1;}}
/* Reframe historical model-bank winners without re-fitting or re-selection. */
extern "C" int ng_codec_repack_model_bank(const uint8_t*s,size_t n,uint8_t*out,size_t cap,size_t*written){
    try{
        require(out&&written);auto f=small_bank::read_frame(s,n,cap);GeometryScope scope(f.d,f.d,f.d);
        std::vector<choice_frame::Choice>menu;for(unsigned id=0;id<12+12*f.models.size();id++){
            unsigned m=id<12?255:(id-12)/12;menu.push_back({id%12,m,0,m==255||id%12==0?small_bank::Weight{}:f.weights[m]});
        }
        for(auto&v:f.raw)v=pyramid(v);std::vector<uint16_t>a(N);require(ng_activity_symbol_values(f.raw.data(),a.data())==0);
        auto frame=choice_frame::write(f.d,f.models,menu,f.selectors,f.raw,f.thr,activity_classify(a,f.thr));
        require(frame.size()<=cap);std::copy(frame.begin(),frame.end(),out);*written=frame.size();return 0;
    }catch(...){return -1;}
}
