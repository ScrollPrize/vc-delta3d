/* Cross-chunk support-only transfer. All models are refitted on the target. */
namespace tap_transfer {
struct Context {
    std::unique_ptr<tap_exact::Search>full;
    tap_exact::Search sparse,proxy;
    void*probe=nullptr;std::vector<uint32_t>positions;
    ~Context(){if(probe)ng_fit_probe_destroy(probe);}
};
void prepare(tap_exact::Search&s,const Bytes&source,unsigned d,unsigned predictor,int stride){
    s.d=d;s.source=source;s.predictor=predictor;
    tap_exact::Scope scope(&s);ng_tap_override=nullptr;ng_tap_capture=tap_exact::capture;
    s.dense=tap_exact::encode(s,stride);
}
tap_exact::Support support(tap_exact::Search&s,const uint8_t*mask){
    tap_exact::Support v;
    for(int b=0;b<2;b++)for(unsigned k=0;k<s.banks[b].ids.size();k++)if(mask[b*161+s.banks[b].ids[k]])v[b].push_back(k);
    return v;
}
bool fit(tap_exact::Search&s,const uint8_t*mask){
    auto v=support(s,mask);for(int b=0;b<2;b++)if(tap_exact::solve(s,b,v[b],true)==-HUGE_VAL)return false;return true;
}
}
extern "C" int ng_tap_support_from_frame(const uint8_t*src,size_t n,uint8_t*out){try{
    require(src&&out&&n>=12&&!memcmp(src,"NCF1",4));Bytes prefix(src,src+12);packed_bank::Reader p{prefix};p.skip(4);
    unsigned hn=p.u(4),zn=p.u(4);require(zn<=n-12&&hn<1048576);
    auto header=packed_bank::decompress(src+12,zn,hn);packed_bank::Reader h{header};h.skip(8);unsigned nm=h.u(4),nc=h.u(4);h.skip(4);
    auto models=choice_frame::read_models(h,nm);int selected=-1;small_bank::Weight weights{};
    for(unsigned k=0;k<nc;k++){h.u(1);unsigned model=h.u(1);h.skip(2);small_bank::Weight w;
      for(auto&v:w)v=int32_t(packed_bank::signed_value(h.u(4),32));
      if(model!=choice_frame::no_model){require(model<models.size()&&(selected<0||selected==int(model)));selected=model;weights=w;}}
    require(selected>=0);auto&model=models[selected];return ng_tap_model_support(model.data(),model.size(),weights.data(),out);
}catch(...){return -1;}}
extern "C" void*ng_tap_transfer_create(const uint8_t*src,unsigned d){try{
    auto c=std::make_unique<tap_transfer::Context>();c->full.reset(static_cast<tap_exact::Search*>(ng_tap_exact_create(src,d)));require(bool(c->full));
    require(c->full->predictor>=12);GeometryScope geometry(d,d,d);
    tap_transfer::prepare(c->sparse,c->full->source,d,c->full->predictor,128);
    require(c->sparse.dense==c->full->baseline);
    zero_blocks::Scan zeros(src,d,true);auto sample=fast_split::screen_proxy(src,d,&zeros);
    tap_transfer::prepare(c->proxy,sample,32,c->full->predictor,4);
    c->probe=ng_fit_probe_create(src);require(c->probe);
    // Distributed row groups with original neighbors, excluding known zero
    // choice blocks. Same sample positions for every candidate and budget.
    uint32_t rng=0xd4891e13;std::set<uint32_t>seen;size_t groups=size_t(d)*d*d/4;
    while(c->positions.size()<16384&&seen.size()<groups){
      uint32_t group=tap_exact::random(rng)%groups;if(!seen.insert(group).second)continue;
      size_t i=size_t(group)*4;if(zeros.any()&&!zeros.mask[i])continue;
      for(unsigned j=0;j<4;j++)c->positions.push_back(i+j);
    }
    require(c->positions.size()>=4096);return c.release();
}catch(const std::exception&e){std::fprintf(stderr,"tap transfer create: %s\n",e.what());return nullptr;}}
extern "C" void ng_tap_transfer_destroy(void*p){delete static_cast<tap_transfer::Context*>(p);}
// Modes:0 full-chunk reference;1 packed32 screen;2/3 original-context4k/16k
// sample histograms. Modes2/3 reuse sparse stats from the normal Fast fit.
// Modes4/5 use dense-fit coefficients for the same samples (diagnostic);
// mode6 provides the full-frame oracle with sparse-fit coefficients.
extern "C" int ng_tap_transfer_eval(void*p,const uint8_t*mask,unsigned mode,uint8_t*out,size_t cap,size_t*n,double*cost){try{
    require(p&&mask&&mode<=6);auto&c=*static_cast<tap_transfer::Context*>(p);
    auto&s=(mode==0||mode==4||mode==5)?*c.full:mode==1?c.proxy:c.sparse;
    require(tap_transfer::fit(s,mask));GeometryScope geometry(s.d,s.d,s.d);
    if(mode>=2&&mode<=5){require(ng_tap_probe_score(c.probe,s.predictor%12,s.q[0].data(),s.q[1].data(),s.bias,c.positions.data(),mode==2||mode==4?4096:16384,cost)==0);*n=0;return 0;}
    tap_exact::Scope scope(&s);ng_tap_capture=nullptr;ng_tap_override=tap_exact::override_fit;
    auto frame=tap_exact::encode(s,mode==1?4:1);require(frame.size()<=cap);memcpy(out,frame.data(),frame.size());*n=frame.size();*cost=frame.size();return 0;
}catch(const std::exception&e){std::fprintf(stderr,"tap transfer eval: %s\n",e.what());return -1;}}
