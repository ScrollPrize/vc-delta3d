// Fitted multiscale NG: native NG selection/Cholesky/Q8 quantization,
// original residual mapping and activity coder; only the traversal/context
// changes. Scalar research decoder, not a production speed implementation.
namespace multiscale_fit {
struct Point {uint32_t i;uint16_t step;uint8_t axis,bank;};
struct Plan {
    unsigned d;std::vector<Point>points;std::vector<uint32_t>rank;
    std::vector<std::array<int,3>>taps;
    Plan(unsigned dim):d(dim),rank(size_t(d)*d*d,UINT32_MAX){
        require(ng_geometry_valid(d,d,d)&&!(d&(d-1)));
        points.push_back({0,0,0,0});rank[0]=0;
        for(unsigned step=d/2;step;step/=2)for(unsigned axis=0;axis<3;axis++){
            unsigned stride[3];for(unsigned a=0;a<3;a++)stride[a]=a<=axis?step:2*step;
            unsigned level=0;for(unsigned s=step;s>1;s/=2)level++;
            for(unsigned z=0;z<d;z+=stride[2])for(unsigned y=0;y<d;y+=stride[1])for(unsigned x=0;x<d;x+=stride[0]){
                unsigned c[3]={x,y,z};if(c[axis]%(2*step)!=step)continue;
                unsigned i=(z*d+y)*d+x;require(rank[i]==UINT32_MAX);rank[i]=points.size();
                points.push_back({i,uint16_t(step),uint8_t(axis),uint8_t(3*std::min(level,3u)+axis)});
            }
        }
        require(points.size()==rank.size());
        for(int z=-1;z<=1;z++)for(int y=-3;y<=3;y++)for(int x=-3;x<=3;x++)
            if(x||y||z)taps.push_back({x,y,z});
        require(taps.size()+1<=160);
    }
    // The first feature is a coarse anchor, others are differences to it.
    // This is an invertible linear feature change when taps are present;
    // unavailable boundary/future taps contribute zero difference.
    void features(const uint8_t*src,const Point&p,double*f,bool coarse_only=false)const{
        int c[3]={int(p.i%d),int(p.i/d%d),int(p.i/(d*d))};size_t pitch[3]={1,d,size_t(d)*d};
        int base=p.step?src[p.i-p.step*pitch[p.axis]]:128;f[0]=base;
        for(unsigned k=0;k<taps.size();k++){
            int q[3];bool valid=p.step;
            for(unsigned a=0;a<3;a++){q[a]=c[a]+taps[k][a]*int(p.step);valid&=q[a]>=0&&q[a]<int(d);}
            f[k+1]=0;if(!valid)continue;
            unsigned j=(q[2]*d+q[1])*d+q[0];if(rank[j]>=rank[p.i])continue;
            if(coarse_only&&(q[0]%(2*p.step)||q[1]%(2*p.step)||q[2]%(2*p.step)))continue;
            f[k+1]=int(src[j])-base;
        }
    }
};
constexpr unsigned Banks=12;
struct Stats {
    unsigned n;uint64_t count=0;std::vector<double>sum,rhs,gram;double target=0;
    Stats(unsigned m):n(m),sum(m),rhs(m),gram(m*m){}
    void add(const double*f,double t){count++;target+=t;
        for(unsigned a=0;a<n;a++){sum[a]+=f[a];rhs[a]+=f[a]*t;
            for(unsigned b=0;b<=a;b++)gram[a*n+b]+=f[a]*f[b];}
    }
};
struct Model {std::vector<int32_t>w;int32_t bias=0;};
Model fit(const Stats&s,bool full){
    Model m{std::vector<int32_t>(s.n)};m.w[0]=256;
    if(s.count<4*s.n)return m;
    std::vector<double>C(s.n*s.n),r(s.n),mean(s.n);double mt=s.target/s.count,tr=0;
    for(unsigned a=0;a<s.n;a++)mean[a]=s.sum[a]/s.count;
    for(unsigned a=0;a<s.n;a++){
        r[a]=s.rhs[a]/s.count-mean[a]*mt;
        for(unsigned b=0;b<s.n;b++)C[a*s.n+b]=s.gram[std::max(a,b)*s.n+std::min(a,b)]/s.count-mean[a]*mean[b];
        tr+=C[a*s.n+a];
    }
    for(unsigned a=0;a<s.n;a++)C[a*s.n+a]+=1e-6*tr/s.n+1e-9;
    require(ng_centered_fit(C.data(),r.data(),mean.data(),mt,s.n,full?s.n:40,m.w.data(),&m.bias)==0);return m;
}
struct Trial {
    Plan plan;Bytes source;std::vector<std::vector<Stats>>stats;
    Trial(const uint8_t*src,unsigned d):plan(d),source(src,src+size_t(d)*d*d){
        unsigned n=plan.taps.size()+1;
        for(unsigned mode=0;mode<3;mode++){stats.emplace_back();for(unsigned b=0;b<Banks;b++)stats.back().emplace_back(n);}
        double f[160],coarse[160];
        for(size_t j=1;j<plan.points.size();j++){
            auto&p=plan.points[j];plan.features(src,p,f);stats[0][p.bank].add(f,src[p.i]);
            // Deterministically spread sparse samples within each refinement
            // bank, not one compact crop. Small/coarse banks use all samples.
            if(p.step>=4||(j*0x9e3779b97f4a7c15ULL>>60)==0)stats[1][p.bank].add(f,src[p.i]);
            plan.features(src,p,coarse,true);stats[2][p.bank].add(coarse,src[p.i]);
        }
    }
};
uint16_t predict(const Model&m,const double*f){int64_t q=m.bias;for(unsigned k=0;k<m.w.size();k++)q+=int64_t(m.w[k])*int(f[k]);return uint16_t(q);}
unsigned group(const Plan&p,size_t i,unsigned cls,bool traversal){return p.points[traversal?i:p.rank[i]].axis*16+cls;}
struct Stage {
    size_t begin,end;unsigned nx,ny,nz,px,py,pz,axis;
    bool index(size_t i,size_t&j)const{
        unsigned x=i%px,y=i/px%py,z=i/(size_t(px)*py);
        if(x>=nx||y>=ny||z>=nz)return false;
        j=begin+(size_t(z)*ny+y)*nx+x;return true;
    }
};
std::vector<Stage>stages(const Plan&p){
    std::vector<Stage>v;for(size_t start=1;start<p.points.size();){
        auto&pt=p.points[start];size_t end=start+1;
        while(end<p.points.size()&&p.points[end].step==pt.step&&p.points[end].axis==pt.axis)end++;
        unsigned dims[3];for(unsigned a=0;a<3;a++)dims[a]=p.d/(pt.step*(a<pt.axis?1:2));
        require(end-start==size_t(dims[0])*dims[1]*dims[2]);
        v.push_back({start,end,dims[0],dims[1],dims[2],std::max(32u,dims[0]),std::max(32u,dims[1]),std::max(32u,dims[2]),pt.axis});start=end;
    }return v;
}
void stage_activity(const Plan&p,const Bytes&symbols,std::vector<uint16_t>&activity){
    activity[0]=0;
    for(auto&s:stages(p)){
        GeometryScope geom(s.px,s.py,s.pz);Bytes padded(size_t(s.px)*s.py*s.pz);
        for(size_t i=0;i<padded.size();i++){size_t j;if(s.index(i,j))padded[i]=symbols[j];}
        std::vector<uint16_t>a(padded.size());require(ng_activity_symbol_values(padded.data(),a.data())==0);
        for(size_t i=0;i<padded.size();i++){size_t j;if(s.index(i,j))activity[j]=a[i];}
    }
}
Bytes encode(Trial&t,unsigned mode){
    require(mode<=13);auto&p=t.plan;GeometryScope geometry(p.d,p.d,p.d);
    bool traversal=mode>=5;unsigned variant=mode%5;
    bool coarse=variant==4,full=variant==1||variant==3;unsigned which=coarse?2:variant>=2?1:0;
    std::vector<Model>models;for(auto&s:t.stats[which])models.push_back(fit(s,full));
    Bytes symbols(N);double f[160];symbols[0]=ng_prediction_symbol(t.source[0],128*256);
    for(size_t j=1;j<N;j++){auto&pt=p.points[j];p.features(t.source.data(),pt,f,coarse);symbols[traversal?j:pt.i]=ng_prediction_symbol(t.source[pt.i],predict(models[pt.bank],f));}
    std::vector<uint16_t>activity(N);
    if(mode>=10)stage_activity(p,symbols,activity);
    else require(ng_activity_symbol_values(symbols.data(),activity.data())==0);
    auto thr=activity_refine(symbols,activity,activity_quantiles(activity,16),{}, {},1,true,64,true);
    auto classes=activity_classify(activity,thr);std::vector<Bytes>streams(48);
    for(size_t j=0;j<N;j++)streams[group(p,j,classes[j],traversal)].push_back(symbols[j]);
    Bytes h,body;put32(h,p.d);put32(h,mode);put32(h,models.size());put32(h,p.taps.size()+1);
    for(auto&m:models){put32(h,uint32_t(m.bias));for(auto w:m.w)put32(h,uint32_t(w));}
    for(auto x:thr)packed_bank::put16(h,x);
    for(auto&v:streams)small_bank::stream_header(h,body,pack_stream(v,true),v.size());
    auto z=packed_bank::compress(h);Bytes frame={'M','N','G','1'};put32(frame,h.size());put32(frame,z.size());packed_bank::append(frame,z);packed_bank::append(frame,body);return frame;
}
struct Read {const Plan&p;std::vector<Bytes>&streams;bool traversal;std::vector<size_t>pos=std::vector<size_t>(48);
    static int get(void*ptr,size_t i,unsigned cls){auto&r=*static_cast<Read*>(ptr);unsigned g=group(r.p,i,cls,r.traversal);if(g>=48||r.pos[g]>=r.streams[g].size())return -1;return r.streams[g][r.pos[g]++];}
};
struct StageRead {Read&reader;const Stage&stage;Bytes&raw;
    static int get(void*ptr,size_t i,unsigned cls){auto&r=*static_cast<StageRead*>(ptr);size_t j;if(!r.stage.index(i,j))return 0;
        int sym=Read::get(&r.reader,j,cls);
        if(sym>=0)r.raw[j]=unpyramid(uint8_t(sym));return sym;}
};
void decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    require(n>=12&&!memcmp(src,"MNG1",4));Bytes frame(src,src+n);packed_bank::Reader r{frame};r.skip(4);auto hn=r.u(4),zn=r.u(4);auto hb=packed_bank::decompress(r.take(zn),hn);packed_bank::Reader h{hb};
    unsigned d=h.u(4),mode=h.u(4);require(mode<=13);Plan p(d);GeometryScope geometry(d,d,d);require(cap>=N&&h.u(4)==Banks&&h.u(4)==p.taps.size()+1);
    std::vector<Model>models;for(unsigned b=0;b<Banks;b++){Model m{std::vector<int32_t>(p.taps.size()+1)};m.bias=int32_t(packed_bank::signed_value(h.u(4),32));for(auto&w:m.w)w=int32_t(packed_bank::signed_value(h.u(4),32));models.push_back(std::move(m));}
    std::vector<uint16_t>thr(15);for(auto&x:thr)x=h.u(2);
    std::vector<Bytes>streams(48);size_t total=0;for(auto&v:streams){v=small_bank::read_stream(h,r,N-total);total+=v.size();}require(total==N);h.end();r.end();
    Read read{p,streams,mode>=5};Bytes raw(N);
    if(mode>=10){
        unsigned cls=std::lower_bound(thr.begin(),thr.end(),0)-thr.begin();int sym=Read::get(&read,0,cls);require(sym>=0);raw[0]=unpyramid(uint8_t(sym));
        for(auto&s:stages(p)){GeometryScope geom(s.px,s.py,s.pz);Bytes padded(size_t(s.px)*s.py*s.pz);StageRead reader{read,s,raw};require(ng_activity_decode(thr.data(),16,0,StageRead::get,&reader,padded.data())==0);}
    }else require(ng_activity_decode(thr.data(),16,0,Read::get,&read,raw.data())==0);
    for(unsigned g=0;g<48;g++)require(read.pos[g]==streams[g].size());
    out[0]=geometry_trials::reconstruct(pyramid(raw[0]),128*256);double f[160];
    for(size_t j=1;j<N;j++){auto&pt=p.points[j];p.features(out,pt,f,mode%5==4);out[pt.i]=geometry_trials::reconstruct(pyramid(raw[mode>=5?j:pt.i]),predict(models[pt.bank],f));}
}
}
extern "C" void*ng_multiscale_fit_create(const uint8_t*s,unsigned d){try{return new multiscale_fit::Trial(s,d);}catch(const std::exception&e){fprintf(stderr,"multiscale fit create: %s\n",e.what());return nullptr;}}
extern "C" void ng_multiscale_fit_destroy(void*p){delete static_cast<multiscale_fit::Trial*>(p);}
extern "C" int ng_multiscale_fit_encode(void*p,unsigned mode,uint8_t*out,size_t cap,size_t*n){try{auto f=multiscale_fit::encode(*static_cast<multiscale_fit::Trial*>(p),mode);require(f.size()<=cap);std::copy(f.begin(),f.end(),out);*n=f.size();return 0;}catch(const std::exception&e){fprintf(stderr,"multiscale fit encode: %s\n",e.what());return -1;}}
extern "C" int ng_multiscale_fit_decode(const uint8_t*s,size_t n,uint8_t*out,size_t cap){try{multiscale_fit::decode(s,n,out,cap);return 0;}catch(const std::exception&e){fprintf(stderr,"multiscale fit decode: %s\n",e.what());return -1;}}
