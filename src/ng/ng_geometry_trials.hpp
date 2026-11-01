// Geometry-prediction experiments. Native fitting/entropy reuse existing NG
// and rANS helpers; deliberately separate frame, no production dispatch change.
#include <climits>
namespace geometry_trials {
constexpr unsigned directions=5, contexts=81;
inline size_t block(size_t i,unsigned d){return choice_grid::block(i,d);}
inline uint8_t value(const uint8_t*s,int x,int y,int z,unsigned d){
    return x<0||y<0||z<0||x>=int(d)||y>=int(d)||z>=int(d)?128:s[(size_t(z)*d+y)*d+x];
}
inline int ray(const uint8_t*s,int x,int y,int z,unsigned d,unsigned dir,int*gradient=nullptr){
    static constexpr int dx[5]={0,-1,1,0,0},dy[5]={0,0,0,-1,1};
    int a=value(s,x+dx[dir],y+dy[dir],z-1,d),b=value(s,x+2*dx[dir],y+2*dy[dir],z-2,d);
    if(gradient)*gradient=std::abs(a-b);
    return std::clamp(2*a-b,0,255);
}
inline unsigned steer(const uint8_t*s,int x,int y,int z,unsigned d){
    if(z<2||x<2||y<2||x+2>=int(d)||y+2>=int(d))return 0;
    unsigned best=0;int score=INT_MAX;
    for(unsigned k=0;k<directions;k++){int a,b;ray(s,x,y,z,d,k,&a);ray(s,x,y-1,z,d,k,&b);int c=a+b;if(c<score){score=c;best=k;}}
    return best;
}
inline unsigned shape(const uint8_t*s,int x,int y,int z,unsigned d){
    int w=value(s,x-1,y,z,d),n=value(s,x,y-1,z,d),b=value(s,x,y,z-1,d);
    auto q=[](int a){return a<-4?0:a>4?2:1;};
    int ww=value(s,x-2,y,z,d);
    return q(w-n)+3*q(w-b)+9*q(n-b)+27*q(w-ww);
}
inline unsigned entropy_shape(const uint8_t*s,int x,int y,int z,unsigned d){
    int w=value(s,x-1,y,z,d),n=value(s,x,y-1,z,d),b=value(s,x,y,z-1,d);
    int range=std::max({w,n,b})-std::min({w,n,b});
    if(range<=4)return 0;
    int a=std::abs(w-n),c=std::abs(w-b),e=std::abs(n-b);
    return a<=c&&a<=e?1:c<=e?2:3;
}
struct State {
    void*p=nullptr;
    explicit State(const Bytes&params){p=ng_causal_create(params.data(),params.size());require(p!=nullptr);}
    ~State(){ng_causal_destroy(p);}
};
inline uint8_t predicted(uint16_t q){return uint8_t(uint16_t(q+128u)>>8);}
inline uint8_t reconstruct(unsigned s,uint16_t q){
    uint8_t e=unpyramid(uint8_t(s));if((q>>7)&1)e=uint8_t(-e);
    return uint8_t(predicted(q)+e);
}
// Modes:0 matched NG control;1 causal steering+NG;2 block steering+NG;
// 3 nonlinear NG correction;4 NG+geometry entropy;5 combine1+3+4;
// 6/7 coarse-to-fine linear / clamped cubic;8 causal steering without NG.
Bytes encode(const uint8_t*src,unsigned d,unsigned mode){
    require(mode<=11&&ng_geometry_valid(d,d,d));GeometryScope scope(d,d,d);
    const bool steered=mode==1||mode==2||mode==5||mode==8;
    const bool correction=mode==3||mode==5,geometry=mode==4||mode==5;
    zero_blocks::Scan zeros(src,d,true);Bytes choices(N/64,0),feature(steered?N:0);
    if(mode==2){
        std::vector<std::array<uint64_t,directions>>cost(choices.size());
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;
            for(unsigned k=0;k<directions;k++)cost[block(i,d)][k]+=std::abs(int(src[i])-ray(src,x,y,z,d,k));
        }
        for(size_t b=0;b<choices.size();b++)choices[b]=std::min_element(cost[b].begin(),cost[b].end())-cost[b].begin();
    }
    for(size_t b=0;b<choices.size();b++)if(zeros.empty[b])choices[b]=5;
    if(steered)for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
        size_t i=(size_t(z)*d+y)*d+x;unsigned k=mode==2?choices[block(i,d)]:steer(src,x,y,z,d);
        feature[i]=ray(src,x,y,z,d,k==5?0:k);
    }
    int cfg[6]={3,3,1,16,40,1};Bytes params(65536),initial(N),classes(N);std::vector<uint16_t>q(N);size_t pn=0;int32_t weights[2]={0,0};
    //9 expanded122 candidates /80 taps;10 adds8 jointly fitted nonlinear
    //inputs within that80 budget;11 expanded candidates, original40 tap budget.
    if(mode>=9){cfg[2]=3;cfg[4]=mode==11?40:80;}
    int32_t context_weights[16]={0};
    if(mode==10)require(ng_context_prepare(src,cfg,zeros.fit_mask(),context_weights,params.data(),params.size(),&pn,initial.data(),classes.data(),q.data())==0);
    else if(steered&&mode!=8)require(ng_feature_prepare_masked(src,cfg,feature.data(),zeros.fit_mask(),1,weights,params.data(),params.size(),&pn,initial.data(),classes.data(),q.data())==0);
    else if(mode!=8)require(ng_plain_prepare_masked(src,cfg,zeros.fit_mask(),params.data(),params.size(),&pn,initial.data(),classes.data(),q.data())==0);
    else {require(ng_bank_empty_model(params.data(),params.size(),&pn)==0);}
    params.resize(pn);
    std::array<int64_t,contexts>sum{};std::array<unsigned,contexts>count{};std::array<int8_t,contexts>offset{};
    State state(params);
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
        size_t i=(size_t(z)*d+y)*d+x;uint16_t p;uint8_t c;
        require(ng_causal_prediction(state.p,i,&p,&c)==0);
        if(mode==10)require(ng_causal_add_context(state.p,i,context_weights,&p)==0);
        // Independent causal traversal must match the fitted encoder's native
        // prediction/class pass, not merely round-trip its own invented stream.
        if(mode==0||mode>=9)require(p==q[i]&&c==classes[i]);
        if(steered)p=mode==8?uint16_t(unsigned(feature[i])*256):uint16_t(p+int64_t(weights[z==0])*feature[i]);
        q[i]=p;classes[i]=c;
        require(ng_causal_accept(state.p,i,src[i],predicted(p))==0);
        if(correction&&!zeros.empty[block(i,d)]){auto g=shape(src,x,y,z,d);sum[g]+=int8_t(uint8_t(src[i]-predicted(p)));count[g]++;}
    }
    if(correction)for(unsigned g=0;g<contexts;g++)offset[g]=int8_t(std::clamp<int64_t>(std::llround(double(sum[g])/(count[g]+64.)), -32,32));
    unsigned streams=geometry?64:16;std::vector<Bytes>symbols(streams);
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
        size_t i=(size_t(z)*d+y)*d+x;if(zeros.empty[block(i,d)])continue;
        uint16_t p=q[i];if(correction)p=uint16_t(p+int(offset[shape(src,x,y,z,d)])*256);
        unsigned c=classes[i]+(geometry?16*entropy_shape(src,x,y,z,d):0);
        symbols[c].push_back(ng_prediction_symbol(src[i],p));
    }
    Bytes h,body;put32(h,d);put32(h,mode);put32(h,params.size());packed_bank::append(h,params);
    for(auto w:weights)put32(h,uint32_t(w));
    if(mode==10)for(auto w:context_weights)put32(h,uint32_t(w));
    if(correction)for(auto v:offset)h.push_back(uint8_t(v));
    auto selected=pack_stream(choices,true);put32(h,selected.size());packed_bank::append(h,selected);
    for(auto&s:symbols)small_bank::stream_header(h,body,pack_stream(s,true),s.size());
    auto z=packed_bank::compress(h);Bytes out={'G','X','P','1'};put32(out,h.size());put32(out,z.size());packed_bank::append(out,z);packed_bank::append(out,body);return out;
}
void decode(const uint8_t*src,size_t bytes,uint8_t*out,size_t cap){
    require(bytes>=12&&!memcmp(src,"GXP1",4));Bytes frame(src,src+bytes);packed_bank::Reader r{frame};r.skip(4);auto hn=r.u(4),zn=r.u(4);require(hn<=1024*1024);auto hbytes=packed_bank::decompress(r.take(zn),hn);packed_bank::Reader h{hbytes};
    unsigned d=h.u(4),mode=h.u(4);require(mode<=11&&mode!=6&&mode!=7&&ng_geometry_valid(d,d,d)&&cap>=size_t(d)*d*d);GeometryScope scope(d,d,d);
    auto params=h.take(h.u(4));int32_t weights[2];for(auto&w:weights)w=int32_t(packed_bank::signed_value(h.u(4),32));
    int32_t context_weights[16]={0};if(mode==10)for(auto&w:context_weights)w=int32_t(packed_bank::signed_value(h.u(4),32));
    bool steered=mode==1||mode==2||mode==5||mode==8,correction=mode==3||mode==5,geometry=mode==4||mode==5;
    std::array<int8_t,contexts>offset{};if(correction)for(auto&v:offset)v=int8_t(packed_bank::signed_value(h.u(1),8));
    auto sel=h.take(h.u(4));Bytes choices(N/64);unpack_stream(sel.data(),sel.size(),choices,true);for(auto c:choices)require(c<=5);
    std::vector<Bytes>streams(geometry?64:16);std::vector<size_t>pos(streams.size());size_t count=0;
    for(auto&s:streams){s=small_bank::read_stream(h,r,N-count);count+=s.size();}h.end();r.end();
    State state(params);
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
        size_t i=(size_t(z)*d+y)*d+x;uint16_t p;uint8_t c;require(ng_causal_prediction(state.p,i,&p,&c)==0);
        if(mode==10)require(ng_causal_add_context(state.p,i,context_weights,&p)==0);
        if(steered){unsigned k=mode==2?choices[block(i,d)]:steer(out,x,y,z,d);auto v=ray(out,x,y,z,d,k==5?0:k);p=mode==8?uint16_t(v*256):uint16_t(p+int64_t(weights[z==0])*v);}
        uint16_t corrected=p;if(correction)corrected=uint16_t(p+int(offset[shape(out,x,y,z,d)])*256);
        if(choices[block(i,d)]==5)out[i]=0;
        else {unsigned g=c+(geometry?16*entropy_shape(out,x,y,z,d):0);require(g<streams.size()&&pos[g]<streams[g].size());out[i]=reconstruct(streams[g][pos[g]++],corrected);}
        require(ng_causal_accept(state.p,i,out[i],predicted(p))==0);
    }
    for(unsigned g=0;g<streams.size();g++)require(pos[g]==streams[g].size());
}
// Progressive interpolation schedule. Each axis pass only accesses the
// previous lattice, so its inserted sites are independent and vectorizable.
template<class Visit> void multiscale(unsigned d,Visit visit){
    for(unsigned z=0;z<d;z+=8)for(unsigned y=0;y<d;y+=8)for(unsigned x=0;x<d;x+=8)visit(x,y,z,0,0);
    for(unsigned s=4;s;s/=2){
        for(unsigned z=0;z<d;z+=2*s)for(unsigned y=0;y<d;y+=2*s)for(unsigned x=s;x<d;x+=2*s)visit(x,y,z,s,0);
        for(unsigned z=0;z<d;z+=2*s)for(unsigned y=s;y<d;y+=2*s)for(unsigned x=0;x<d;x+=s)visit(x,y,z,s,1);
        for(unsigned z=s;z<d;z+=2*s)for(unsigned y=0;y<d;y+=s)for(unsigned x=0;x<d;x+=s)visit(x,y,z,s,2);
    }
}
inline int interpolate(const uint8_t*src,unsigned d,int x,int y,int z,unsigned s,unsigned axis,bool cubic){
    int c[3]={x,y,z};int a[3]={x,y,z},b[3]={x,y,z};a[axis]-=s;b[axis]+=s;
    int l=value(src,a[0],a[1],a[2],d);if(b[axis]>=int(d))return l;
    int r=value(src,b[0],b[1],b[2],d),p=(l+r+1)/2;
    if(cubic&&c[axis]>=int(3*s)&&c[axis]+3*s<d){a[axis]-=2*s;b[axis]+=2*s;
        int ll=value(src,a[0],a[1],a[2],d),rr=value(src,b[0],b[1],b[2],d);
        p=std::clamp((9*(l+r)-ll-rr+8)/16,std::min(l,r),std::max(l,r));}
    return p;
}
Bytes coarse_encode(const uint8_t*src,unsigned d,unsigned mode){
    require(ng_geometry_valid(d,d,d));GeometryScope scope(d,d,d);Bytes residual(N);size_t k=0;
    multiscale(d,[&](unsigned x,unsigned y,unsigned z,unsigned s,unsigned axis){size_t i=(size_t(z)*d+y)*d+x;residual[k++]=s?pyramid(uint8_t(src[i]-interpolate(src,d,x,y,z,s,axis,mode==7))):src[i];});require(k==N);
    Bytes inner(3*N+131072);size_t n=0;int cfg[6]={3,3,1,16,40,1};
    require(ng_codec_encode_fast_config(residual.data(),d,cfg,16,inner.data(),inner.size(),&n)==0);inner.resize(n);
    Bytes f={'M','G','P','1'};put32(f,d);put32(f,mode);packed_bank::append(f,inner);return f;
}
void coarse_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    require(n>=12&&!memcmp(src,"MGP1",4));Bytes f(src,src+n);size_t pos=4;unsigned d=get32(f,pos),mode=get32(f,pos);require((mode==6||mode==7)&&ng_geometry_valid(d,d,d)&&cap>=size_t(d)*d*d);GeometryScope scope(d,d,d);
    Bytes residual(N);require(ng_codec_decode(src+12,n-12,residual.data(),residual.size())==0);size_t k=0;
    multiscale(d,[&](unsigned x,unsigned y,unsigned z,unsigned s,unsigned axis){size_t i=(size_t(z)*d+y)*d+x;out[i]=s?uint8_t(interpolate(out,d,x,y,z,s,axis,mode==7)+unpyramid(residual[k++])):residual[k++];});require(k==N);
}
}
extern "C" int ng_geometry_trial_encode(const uint8_t*src,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*n){
    try{require(src&&out&&n&&mode<=11);auto f=mode==6||mode==7?geometry_trials::coarse_encode(src,d,mode):geometry_trials::encode(src,d,mode);require(f.size()<=cap);memcpy(out,f.data(),f.size());*n=f.size();return 0;}catch(const std::exception&e){fprintf(stderr,"geometry trial encode: %s\n",e.what());return -1;}
}
extern "C" int ng_geometry_trial_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    try{require(src&&out&&n>=4);if(!memcmp(src,"MGP1",4))geometry_trials::coarse_decode(src,n,out,cap);else geometry_trials::decode(src,n,out,cap);return 0;}catch(const std::exception&e){fprintf(stderr,"geometry trial decode: %s\n",e.what());return -1;}
}
