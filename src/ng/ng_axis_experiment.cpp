// Experimental lossless joint axis predictor. No production codec dispatch.
// Geometry/decoding is integer only; encoder fits reuse NG's solver and Q8 qopt.
#include "residual_bridge.h"
#include "adaptive_arithmetic.hpp"
#include <memory>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <chrono>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define JOINT_AVX2 1
#else
#define JOINT_AVX2 0
#endif

extern "C" int delta_rans_decode_constant(const uint8_t*,size_t,uint8_t*,size_t);
namespace {
#if defined(__GNUC__) || defined(__clang__)
#define JOINT_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define JOINT_INLINE __forceinline
#else
#define JOINT_INLINE inline
#endif
using Bytes=std::vector<uint8_t>;
using Vec=std::array<int,3>;
constexpr int K=16,Q=16;
JOINT_INLINE void need(bool b){if(!b)throw std::runtime_error("invalid joint axis frame");}
unsigned isqrt(unsigned x){unsigned r=0,b=1u<<30;while(b>x)b>>=2;while(b){if(x>=r+b){x-=r+b;r=(r>>1)+b;}else r>>=1;b>>=2;}return r;}
const std::array<Vec,65536>& directions(){
    static const auto lut=[](){std::array<Vec,65536> v{};
        for(int x=0;x<256;x++)for(int y=0;y<256;y++){
            int a=(x-128)*Q,b=(y-128)*Q;
            v[x*256+y]={a,b,int(isqrt(unsigned(std::max(0,127*127*Q*Q-a*a-b*b))))};
        }return v;}();return lut;
}
Vec axis(uint8_t x,uint8_t y){return directions()[unsigned(x)*256+y];}
int64_t dot(const Vec&a,const Vec&b){return int64_t(a[0])*b[0]+int64_t(a[1])*b[1]+int64_t(a[2])*b[2];}
Vec align(Vec a,const Vec&ref){if(dot(a,ref)<0)for(auto&v:a)v=-v;return a;}
JOINT_INLINE int rounded(int64_t v,int d){return int(v>=0?(v+d/2)/d:-((-v+d/2)/d));}
void put(Bytes&b,uint32_t v){for(int k=0;k<4;k++)b.push_back(uint8_t(v>>(8*k)));}
uint32_t get(const uint8_t*b,size_t n,size_t&p){need(p+4<=n);uint32_t v=0;for(int k=0;k<4;k++)v|=uint32_t(b[p++])<<(8*k);return v;}
uint8_t pyr(uint8_t r){return uint8_t((unsigned(r)<<1)^(0u-(unsigned(r)>>7)));}
uint8_t unp(uint8_t s){return uint8_t((s&1)?-(int(s)+1)/2:s/2);}
Bytes pack(const Bytes&v){
    if(v.empty())return {};
    Bytes raw(v.size()*2+4096);size_t n=0;
    need(delta_rans_encode(v.data(),v.size(),raw.data(),raw.size(),&n)==0 && n>=512);
    uint16_t freq[256];for(int k=0;k<256;k++)freq[k]=uint16_t(raw[2*k]|unsigned(raw[2*k+1])<<8);
    Bytes out(1024);out.resize(ng_write_freq_table(out.data(),freq));out.insert(out.end(),raw.begin()+512,raw.begin()+n);return out;
}
Bytes unpack(const uint8_t*v,size_t n,size_t count,bool reuse=false,uint64_t*entropy_ns=nullptr,Bytes out={},bool constant=false){
    if(!count){need(n==0);return {};}
    uint16_t freq[256];size_t used=0;need(ng_read_freq_table(v,n,freq,&used)==0 && used<=n);
    Bytes local;static thread_local Bytes scratch;Bytes&raw=reuse?scratch:local;raw.resize(512);for(int k=0;k<256;k++){raw[2*k]=uint8_t(freq[k]);raw[2*k+1]=uint8_t(freq[k]>>8);}
    raw.insert(raw.end(),v+used,v+n);out.resize(count);
    std::chrono::steady_clock::time_point stamp;if(entropy_ns)stamp=std::chrono::steady_clock::now();
    need((constant?delta_rans_decode_constant:delta_rans_decode)(raw.data(),raw.size(),out.data(),count)==0);
    if(entropy_ns)*entropy_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-stamp).count();
    return out;
}
// Grammar4: the same signed-width tags, with fixed-width remainder streams.
// LSB-first packing, zero tail padding; no entropy tables for the remainders.
Bytes pack_bits(const Bytes&v,unsigned w){
    need(w>=1&&w<=6);Bytes out((v.size()*w+7)/8,0);size_t bit=0;
    for(unsigned a:v){need(a<(1u<<w));size_t p=bit/8;unsigned shift=bit%8;
        out[p]|=uint8_t(a<<shift);if(shift+w>8)out[p+1]|=uint8_t(a>>(8-shift));bit+=w;}
    return out;
}
Bytes unpack_bits(const uint8_t*v,size_t n,size_t count,unsigned w){
    need(w>=1&&w<=6&&n==(count*w+7)/8);Bytes out(count);size_t bit=0;
    for(auto&a:out){size_t p=bit/8;unsigned shift=bit%8,r=v[p]>>shift;
        if(shift+w>8)r|=unsigned(v[p+1])<<(8-shift);a=uint8_t(r&((1u<<w)-1));bit+=w;}
    if(bit%8)need((v[n-1]>>(bit%8))==0);return out;
}
// Finite deterministic local projective charts. Each chart is a permutation of
// ALL65536 stored byte pairs: even duplicate/invalid directions remain distinct.
// Coordinates are angular-order ranks, not rounded geometric reconstructions.
struct Chart {
    Vec center;
    std::array<uint16_t,65536> forward{},inverse{};
    std::array<int16_t,65536> u{},v{};
    std::array<int,256> row_u{};
    uint16_t project(int pu,int pv)const{
        int row=int(std::lower_bound(row_u.begin(),row_u.end(),pu)-row_u.begin());
        uint16_t best=0;int64_t score=INT64_MAX;
        for(int r=std::max(0,row-2);r<=std::min(255,row+1);r++){
            auto first=inverse.begin()+r*256,last=first+256;
            auto it=std::lower_bound(first,last,pv,[&](uint16_t id,int target){return v[id]<target;});
            int idx=int(it-first);
            for(int j=std::max(0,idx-1);j<=std::min(255,idx);j++){
                auto id=first[j];int64_t a=int(u[id])-pu,b=int(v[id])-pv,s=a*a+b*b;
                if(s<score||(s==score&&id<inverse[best])){score=s;best=uint16_t(r*256+j);}
            }
        }return best;
    }
};
// Integer vectoring CORDIC, output turn/16384. Only prediction uses these
// rounded angles: exact residual inversion uses the full byte-pair permutation.
int angle(int64_t y,int64_t x){
    if(!x)return y<0?-4096:y>0?4096:0;
    constexpr int a[]={8192,4836,2555,1297,651,326,163,81,41,20,10,5,3,1,1};
    x*=65536;y*=65536;int result=0;
    for(int k=0;k<15;k++){int64_t ox=x,oy=y,div=int64_t(1)<<k;if(oy>0){x+=oy/div;y-=ox/div;result+=a[k];}else if(oy<0){x-=oy/div;y+=ox/div;result-=a[k];}}
    return rounded(result,4);
}
uint64_t isqrt64(uint64_t x){uint64_t r=0,b=uint64_t(1)<<62;while(b>x)b>>=2;while(b){if(x>=r+b){x-=r+b;r=(r>>1)+b;}else r>>=1;b>>=2;}return r;}
Vec unit(Vec v){unsigned l=unsigned(isqrt64(uint64_t(dot(v,v))));if(!l)return {0,0,1024};for(auto&x:v)x=rounded(int64_t(x)*1024,l);return v;}
Vec unit_axis(uint8_t x,uint8_t y){
    // Exact finite-domain memoization, including out-of-disk stored byte pairs.
    static const auto table=[](){std::array<Vec,65536> a{};for(unsigned i=0;i<65536;i++)a[i]=unit(axis(i>>8,i&255));return a;}();
    return table[unsigned(x)*256+y];
}
Vec cross(Vec a,Vec b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
const std::vector<Chart>& charts(){
    static const auto all=[](){std::vector<Chart> out;
        for(int z=0;z<=2;z++)for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++){
            if(!z&&(y<0||(!y&&x<=0)))continue;
            if(std::gcd(std::gcd(std::abs(x),std::abs(y)),z)!=1)continue;
            Chart c;c.center=unit({x*1024,y*1024,z*1024});
            int k=0;for(int t=1;t<3;t++)if(std::abs(c.center[t])<std::abs(c.center[k]))k=t;
            Vec base{};base[k]=1;Vec a=unit(cross(base,c.center)),b=unit(cross(c.center,a));
            struct Point {int64_t u,v;uint16_t id;};std::vector<Point> p;p.reserve(65536);
            for(unsigned id=0;id<65536;id++){
                Vec n=axis(id>>8,id&255);int64_t d=dot(n,c.center);
                // Sign at exactly90deg is deterministic and independent of input sign.
                if(d<0||(d==0&&(n[0]<0||(!n[0]&&n[1]<0)))){for(auto&v:n)v=-v;d=-d;}
                int u=angle(dot(n,a),d),v=angle(dot(n,b),d);
                c.u[id]=int16_t(u);c.v[id]=int16_t(v);p.push_back({u,v,uint16_t(id)});
            }
            std::sort(p.begin(),p.end(),[](auto&a,auto&b){return a.u!=b.u?a.u<b.u:(a.v!=b.v?a.v<b.v:a.id<b.id);});
            for(unsigned u=0;u<256;u++){
                auto first=p.begin()+u*256;
                c.row_u[u]=int(first[128].u);
                std::sort(first,first+256,[](auto&a,auto&b){return a.v!=b.v?a.v<b.v:(a.u!=b.u?a.u<b.u:a.id<b.id);});
                for(unsigned v=0;v<256;v++){unsigned code=u*256+v,id=p[code].id;c.forward[id]=uint16_t(code);c.inverse[code]=uint16_t(id);}
            }
            out.push_back(std::move(c));
        }return out;}();return all;
}
int chart_for(Vec r){
    int best=0;int64_t score=-1;const auto&all=charts();
    for(size_t k=0;k<all.size();k++){int64_t s=std::abs(dot(r,all[k].center));if(s>score){score=s;best=int(k);}}
    return best;
}
Vec mapped(uint8_t x,uint8_t y,int chart){unsigned code=unsigned(x)*256+y;return {charts()[chart].u[code],charts()[chart].v[code],chart};}
struct Model {
    int d=0,mode=0,nt=0;
    std::array<int,480> taps{};
    std::array<std::vector<int32_t>,3>w;
    Vec bias{};
    std::array<Vec,K> cb{};
    std::array<uint32_t,K-1>thr{};
    explicit Model(int dim,int m):d(dim),mode(m){nt=ng_axis_taps(taps.data());for(auto&v:w)v.resize(nt);}
    size_t count()const{return size_t(d)*d*d;}
    bool angular()const{return mode>=2;}
    Vec target(uint8_t x,uint8_t y,Vec r)const{return angular()?mapped(x,y,r[2]):align(axis(x,y),r);}
    Vec ref(const uint8_t*x,const uint8_t*y,size_t i)const{
        if(angular()){
            // Sign-invariant medoid of the three decoded face neighbors.
            size_t ids[3];Vec v[3];int n=0;
            if(i%d)ids[n++]=i-1;
            if((i/d)%d)ids[n++]=i-d;
            if(i/size_t(d)/d)ids[n++]=i-size_t(d)*d;
            if(!n)return mapped(128,128,chart_for({0,0,1024}));
            for(int k=0;k<n;k++)v[k]=unit(axis(x[ids[k]],y[ids[k]]));
            int best=0;int64_t bestscore=-1;
            for(int k=0;k<n;k++){int64_t score=0;for(int j=0;j<n;j++)score+=std::abs(dot(v[k],v[j]));if(score>bestscore){bestscore=score;best=k;}}
            size_t j=ids[best];return mapped(x[j],y[j],chart_for(v[best]));
        }
        size_t j=i;
        if(i%d)j=i-1;else if((i/d)%d)j=i-d;else if(i/size_t(d)/d)j=i-size_t(d)*d;
        else return {0,0,127*Q};
        return axis(x[j],y[j]);
    }
    void features(const uint8_t*x,const uint8_t*y,size_t i,Vec r,std::vector<Vec>&f)const{
        int xx=i%d,yy=(i/d)%d,zz=i/size_t(d)/d;
        for(int k=0;k<nt;k++){
            int a=xx+taps[3*k],b=yy+taps[3*k+1],c=zz+taps[3*k+2];Vec v=r;
            if(a>=0&&a<d&&b>=0&&b<d&&c>=0&&c<d){size_t j=(size_t(c)*d+b)*d+a;need(j<i);v=target(x[j],y[j],r);}
            for(int t=0;t<3;t++)f[k][t]=v[t]-r[t];
        }
    }
    Vec predict(Vec r,const std::vector<Vec>&f)const{
        Vec p{};for(int t=0;t<(angular()?2:3);t++){int64_t s=bias[t];for(int k=0;k<nt;k++)s+=int64_t(w[t][k])*f[k][t];p[t]=std::clamp(r[t]+rounded(s,256),-8192,8192);}if(angular())p[2]=r[2];return p;
    }
    unsigned activity(const std::vector<uint16_t>&e,size_t i)const{
        int x=i%d,y=(i/d)%d,z=i/size_t(d)/d;
        const int off[10][4]={{0,-1,0,2},{0,0,-1,2},{-1,-1,0,1},{1,-1,0,1},{-1,0,-1,1},{1,0,-1,1},{0,-1,-1,1},{0,1,-1,1},{0,-2,0,1},{0,0,-2,1}};
        unsigned a=0;for(auto&o:off){int xx=x+o[0],yy=y+o[1],zz=z+o[2];if(xx>=0&&xx<d&&yy>=0&&yy<d&&zz>=0&&zz<d)a+=o[3]*e[(size_t(zz)*d+yy)*d+xx];}return a;
    }
    int cls(unsigned a)const{return int(std::lower_bound(thr.begin(),thr.end(),a)-thr.begin());}
    uint16_t error(Vec p,Vec v)const{
        if(angular())return uint16_t(std::min(65535,(std::abs(v[0]-p[0])+std::abs(v[1]-p[1]))/(2*Q)));
        v=align(v,p);unsigned sum=0;for(int t=0;t<3;t++)sum+=std::abs(v[t]-p[t]);return uint16_t(std::min(65535u,sum/(3*Q)));
    }
    std::array<uint8_t,2> bytes(Vec p,int c)const{
        if(angular()){
            auto uv=local_bytes(p,c);unsigned id=charts()[p[2]].inverse[unsigned(uv[0])*256+uv[1]];
            return {uint8_t(id>>8),uint8_t(id)};
        }
        for(int t=0;t<3;t++)p[t]+=cb[c][t];
        if(p[2]<0)for(auto&v:p)v=-v;
        // Keep the prediction within the unit sphere; deterministic integer normalization.
        uint64_t sq=0;for(int v:p)sq+=int64_t(v)*v;
        unsigned length=isqrt(unsigned(std::min<uint64_t>(sq,0xffffffffu)));
        if(length)for(auto&v:p)v=rounded(int64_t(v)*127*Q,length);
        return {uint8_t(std::clamp(128+rounded(p[0],Q),0,255)),uint8_t(std::clamp(128+rounded(p[1],Q),0,255))};
    }
    std::array<uint8_t,2> local_bytes(Vec p,int c)const{
        unsigned code=charts()[p[2]].project(std::clamp(p[0]+cb[c][0],-4096,4096),std::clamp(p[1]+cb[c][1],-4096,4096));
        return {uint8_t(code>>8),uint8_t(code)};
    }
};
void fit(Model&m,const uint8_t*x,const uint8_t*y){
    const int n=m.nt;std::vector<Vec> f(n);
    std::array<std::vector<double>,3> gram,rhs,sum;
    Vec sy{};size_t count=0;
    for(int t=0;t<3;t++){gram[t].resize(n*n);rhs[t].resize(n);sum[t].resize(n);}
    for(size_t i=0;i<m.count();i++){
        int xx=i%m.d,yy=(i/m.d)%m.d,zz=i/size_t(m.d)/m.d;
        if(xx!=(4+(yy*7+zz*3)%m.d)%m.d)continue;
        Vec r=m.ref(x,y,i),v=m.target(x[i],y[i],r);m.features(x,y,i,r,f);count++;
        for(int t=0;t<3;t++){
            int target=v[t]-r[t];sy[t]+=target;
            for(int a=0;a<n;a++){double fa=f[a][t];sum[t][a]+=fa;rhs[t][a]+=fa*target;for(int b=0;b<=a;b++)gram[t][a*n+b]+=fa*f[b][t];}
        }
    }
    for(int t=0;t<(m.angular()?2:3);t++){
        double mt=double(sy[t])/count,tr=0;
        for(int a=0;a<n;a++){
            rhs[t][a]=rhs[t][a]/count-sum[t][a]/count*mt;
            for(int b=0;b<=a;b++){double v=gram[t][a*n+b]/count-sum[t][a]/count*sum[t][b]/count;gram[t][a*n+b]=gram[t][b*n+a]=v;}
            tr+=gram[t][a*n+a];
        }
        for(int a=0;a<n;a++)gram[t][a*n+a]+=1e-6*tr/n+1e-9;
        int failed=m.angular()?ng_axis_solve_selected(gram[t].data(),rhs[t].data(),n,m.w[t].data()):ng_axis_solve(gram[t].data(),rhs[t].data(),n,m.w[t].data());
        if(failed)std::fill(m.w[t].begin(),m.w[t].end(),0);
        double bias=mt*256;for(int a=0;a<n;a++)bias-=m.w[t][a]*sum[t][a]/count;m.bias[t]=int32_t(std::floor(bias+0.5));
    }
}
Bytes encode(const uint8_t*x,const uint8_t*y,int d,int mode,uint64_t*stats){
    Model m(d,mode);fit(m,x,y);size_t N=m.count();std::vector<Vec> pred(N),f(m.nt);std::vector<uint16_t> e(N);std::vector<unsigned> act(N);
    for(size_t i=0;i<N;i++){Vec r=m.ref(x,y,i);m.features(x,y,i,r,f);pred[i]=m.predict(r,f);act[i]=m.activity(e,i);e[i]=m.error(pred[i],m.target(x[i],y[i],pred[i]));}
    auto sorted=act;std::sort(sorted.begin(),sorted.end());for(int k=1;k<K;k++)m.thr[k-1]=sorted[(N*k+K-1)/K-1];
    std::array<std::array<int64_t,3>,K> sums{};std::array<size_t,K> counts{};
    for(size_t i=0;i<N;i++){int c=m.cls(act[i]);Vec v=m.target(x[i],y[i],pred[i]);counts[c]++;for(int t=0;t<(m.angular()?2:3);t++)sums[c][t]+=v[t]-pred[i][t];}
    for(int c=0;c<K;c++)for(int t=0;t<3;t++)m.cb[c][t]=counts[c]?std::clamp(rounded(sums[c][t],int(counts[c])),-4096,4096):0;
    std::vector<Bytes> streams(K*(mode==1?3:2));uint64_t flips=0;
    for(size_t i=0;i<N;i++){
        int c=m.cls(act[i]);auto p=m.bytes(pred[i],c);uint8_t a=x[i],b=y[i];
        if(mode==1){bool flip=dot(axis(a,b),axis(p[0],p[1]))<0;if(flip){a=uint8_t(256-a);b=uint8_t(256-b);flips++;}streams[2*K+c].push_back(flip);}
        if(mode==3){unsigned uv=charts()[pred[i][2]].forward[unsigned(a)*256+b];a=uint8_t(uv>>8);b=uint8_t(uv);p=m.local_bytes(pred[i],c);}
        streams[c].push_back(pyr(uint8_t(a-p[0])));streams[K+c].push_back(pyr(uint8_t(b-p[1])));
    }
    Bytes out{'J','A','X',uint8_t(mode>=2?'2':'1')};put(out,d);put(out,mode);put(out,m.nt);
    for(auto v:m.thr)put(out,v);
    for(int t=0;t<3;t++){put(out,uint32_t(m.bias[t]));for(auto w:m.w[t])put(out,uint32_t(w));}
    for(auto v:m.cb)for(auto b:v)put(out,uint32_t(b));
    uint64_t model=out.size()+8*streams.size(),data=0,flags=0;
    for(size_t k=0;k<streams.size();k++){auto b=pack(streams[k]);put(out,streams[k].size());put(out,b.size());out.insert(out.end(),b.begin(),b.end());if(k<2*K)data+=b.size();else flags+=b.size();}
    if(stats){stats[0]=model;stats[1]=data;stats[2]=flags;stats[3]=flips;}return out;
}
void decode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap){
    need(n>=16 && !std::memcmp(src,"JAX",3));size_t pos=4;int d=get(src,n,pos),mode=get(src,n,pos),nt=get(src,n,pos);need(src[3]==(mode>=2?'2':'1'));
    need(d>=32&&d<=256&&d%32==0&&mode>=0&&mode<=3);Model m(d,mode);need(nt==m.nt&&cap>=m.count());
    for(auto&v:m.thr)v=get(src,n,pos);need(std::is_sorted(m.thr.begin(),m.thr.end()));
    for(int t=0;t<3;t++){m.bias[t]=int32_t(get(src,n,pos));need(std::abs(int64_t(m.bias[t]))<100000000);for(auto&w:m.w[t]){w=int32_t(get(src,n,pos));need(std::abs(int64_t(w))<=65536);}}
    for(auto&v:m.cb)for(auto&b:v){b=int32_t(get(src,n,pos));need(b>=-4096&&b<=4096);}
    std::vector<Bytes> streams(K*(mode==1?3:2));std::vector<size_t> used(streams.size());
    for(auto&s:streams){size_t count=get(src,n,pos),bytes=get(src,n,pos);need(count<=m.count()&&bytes<=n-pos);s=unpack(src+pos,bytes,count);pos+=bytes;}need(pos==n);
    std::vector<Vec> f(m.nt);std::vector<uint16_t>e(m.count());
    auto read=[&](int k){need(used[k]<streams[k].size());return streams[k][used[k]++];};
    for(size_t i=0;i<m.count();i++){
        Vec r=m.ref(x,y,i);m.features(x,y,i,r,f);Vec pred=m.predict(r,f);int c=m.cls(m.activity(e,i));auto p=m.bytes(pred,c);
        if(mode==3)p=m.local_bytes(pred,c);
        uint8_t a=uint8_t(p[0]+unp(read(c))),b=uint8_t(p[1]+unp(read(K+c)));
        if(mode==1){auto flag=read(2*K+c);need(flag<=1);if(flag){a=uint8_t(256-a);b=uint8_t(256-b);}}
        if(mode==3){unsigned id=charts()[pred[2]].inverse[unsigned(a)*256+b];a=uint8_t(id>>8);b=uint8_t(id);}
        x[i]=a;y[i]=b;e[i]=m.error(pred,m.target(a,b,pred));
    }
    for(size_t k=0;k<used.size();k++)need(used[k]==streams[k].size());
}
}
extern "C" int ng_axis_encode(const uint8_t*x,const uint8_t*y,int d,int mode,uint8_t*out,size_t cap,size_t*n,uint64_t*stats){
    try{need(x&&y&&out&&n&&d>=32&&d<=256&&d%32==0&&mode>=0&&mode<=3);auto b=encode(x,y,d,mode,stats);need(b.size()<=cap);std::memcpy(out,b.data(),b.size());*n=b.size();return 0;}catch(...){return -1;}
}
extern "C" int ng_axis_decode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap){
    try{need(src&&x&&y);decode(src,n,x,y,cap);return 0;}catch(...){return -1;}
}
extern "C" int ng_axis_chart_audit(uint64_t*stats){
    try{
        need(stats);const auto&all=charts();stats[0]=all.size();stats[1]=all.size()*65536;stats[2]=stats[3]=0;
        for(auto&c:all)for(unsigned i=0;i<65536;i++)need(c.inverse[c.forward[i]]==i);
        for(auto c:all){Vec v=c.center;int a=chart_for(v);for(auto&x:v)x=-x;need(chart_for(v)==a);}
        // Fine smooth crossing at an oblique equatorial direction: negative global z
        // uses opposite stored xy, but stays near the same local angular reference.
        int chart=chart_for({800,600,0});unsigned last=0,rawlast=0;
        for(int j=-100;j<=100;j++){
            double angle=j*.002,sign=j>=0?1:-1;
            int x=int(std::round(128+127*.8*std::cos(angle)*sign)),y=int(std::round(128+127*.6*std::cos(angle)*sign));
            unsigned raw=unsigned(x)*256+y,uv=all[chart].forward[raw];
            if(j>-100){stats[2]=std::max(stats[2],uint64_t(std::abs(x-int(rawlast>>8))+std::abs(y-int(rawlast&255))));stats[3]=std::max(stats[3],uint64_t(std::abs(int(uv>>8)-int(last>>8))+std::abs(int(uv&255)-int(last&255))));}
            last=uv;rawlast=raw;
        }return 0;
    }catch(...){return -1;}
}
extern "C" void ng_axis_delta_predict(const uint8_t*x,const uint8_t*y,size_t i,int d,unsigned mask,uint8_t*out){
    unsigned xx=i%d,yy=(i/d)%d,zz=i/size_t(d)/d;
    if(!xx)mask&=~1u;if(!yy)mask&=~2u;if(!zz)mask&=~4u;
    if(!mask){out[0]=out[1]=0;return;}
    Vec refs[3];int n=0;
    for(unsigned a=1;a<=4;a<<=1)if(mask&a){size_t off=a==1?1:a==2?size_t(d):size_t(d)*d;refs[n++]=unit(axis(x[i-off],y[i-off]));}
    int best=0;int64_t score=-1;
    for(int k=0;k<n;k++){int64_t s=0;for(int j=0;j<n;j++)s+=std::abs(dot(refs[k],refs[j]));if(s>score){score=s;best=k;}}
    Vec p{};
    for(unsigned s=mask;s;s=(s-1)&mask){
        size_t off=((s&1)?1:0)+((s&2)?size_t(d):0)+((s&4)?size_t(d)*d:0);
        Vec v=align(axis(x[i-off],y[i-off]),refs[best]);int sign=(!!(s&1)+!!(s&2)+!!(s&4))&1?1:-1;
        for(int k=0;k<3;k++)p[k]+=sign*v[k];
    }
    if(p[2]<0)for(auto&v:p)v=-v;
    out[0]=uint8_t(128+rounded(p[0],Q));out[1]=uint8_t(128+rounded(p[1],Q));
}

namespace {
struct Geometry {ng_geometry old;Geometry(int d):old(ng_current_geometry){ng_current_geometry={d,d,d};}~Geometry(){ng_current_geometry=old;}};
struct Causal {void* p;explicit Causal(const Bytes&b):p(ng_causal_create(b.data(),b.size())){need(p);}~Causal(){ng_causal_destroy(p);}Causal(const Causal&)=delete;};
std::vector<int> options(int mode){
    if(mode>=100&&mode<116){std::vector<int> ids;if(mode>=108)ids.push_back(0);for(int k=1;k<=14;k++)ids.push_back(k);return ids;}
    need(mode>=0&&mode<=5);std::vector<int> ids;
    if(mode>=3)ids.push_back(0);
    if(mode==0||mode==2||mode==3||mode==4)for(int m=1;m<=7;m++)ids.push_back(m);
    if(mode==1||mode==2||mode==4||mode==5)for(int m=8;m<=14;m++)ids.push_back(m);
    return ids;
}
size_t block_index(size_t i,int d){return ((i/size_t(d)/d/4)*(d/4)+(i/d%d)/4)*(d/4)+(i%d)/4;}
bool conditional(int mode){return mode>=100&&mode<116;}
int first_channel(int mode){return conditional(mode)?(mode-100)%2:0;}
int confidence(int mode){const int thresholds[]={0,8,24,64};return thresholds[((mode-100)/2)%4];}
uint8_t conditional_delta(const uint8_t* first,const uint8_t* second,size_t i,int d,unsigned mask,int threshold){
    if(!(i%d))mask&=~1u;if(!((i/d)%d))mask&=~2u;if(!(i/size_t(d)/d))mask&=~4u;
    int p=0;
    for(unsigned s=mask;s;s=(s-1)&mask){
        size_t off=((s&1)?1:0)+((s&2)?size_t(d):0)+((s&4)?size_t(d)*d:0),j=i-off;
        int plain=std::abs(int(first[i])-int(first[j]));
        int flipped=std::abs(int(first[i])-int(uint8_t(256-first[j])));
        uint8_t v=second[j];if(plain-flipped>threshold)v=uint8_t(256-v);
        p+=((!!(s&1)+!!(s&2)+!!(s&4))&1)?int(v):-int(v);
    }return uint8_t(p);
}
uint8_t pair_prediction(const uint8_t*x,const uint8_t*y,size_t i,int d,int channel,int choice,uint16_t q,int mode){
    if(!choice)return uint8_t(uint16_t(q+128u)>>8);
    if(choice<=7)return delta_predict_at(channel?y:x,i,choice);
    if(conditional(mode)){
        int first=first_channel(mode);need(channel!=first);
        return conditional_delta(first?y:x,first?x:y,i,d,choice-7,confidence(mode));
    }
    uint8_t p[2];ng_axis_delta_predict(x,y,i,d,choice-7,p);return p[channel];
}
Bytes pair_encode(const uint8_t*x,const uint8_t*y,int d,int mode,uint64_t*stats,uint64_t*channel_bytes=nullptr){
    Geometry geometry(d);size_t N=size_t(d)*d*d;int cfg[]={3,3,1,16,40,128};const uint8_t*src[]={x,y};
    std::array<Bytes,2>params,initial_sy,initial_cl,choices;
    std::array<std::vector<Bytes>,2>cand;
    for(int c=0;c<2;c++){
        params[c].resize(65536);initial_sy[c].resize(N);initial_cl[c].resize(N);std::vector<uint16_t>q(N);size_t pn=0;
        need(ng_ex_prepare(src[c],cfg,0,0,params[c].data(),params[c].size(),&pn,initial_sy[c].data(),initial_cl[c].data(),q.data())==0);params[c].resize(pn);
        cand[c].resize(15);cand[c][0]=initial_sy[c];for(int k=1;k<15;k++)cand[c][k].resize(N);
    }
    for(int mask=1;mask<=7;mask++)for(size_t i=0;i<N;i++){
        uint8_t aligned[2];
        if(conditional(mode)){
            int first=first_channel(mode);aligned[first]=delta_predict_at(src[first],i,mask);
            aligned[1-first]=conditional_delta(src[first],src[1-first],i,d,mask,confidence(mode));
        }else ng_axis_delta_predict(x,y,i,d,mask,aligned);
        for(int c=0;c<2;c++){cand[c][mask][i]=pyr(uint8_t(src[c][i]-delta_predict_at(src[c],i,mask)));cand[c][mask+7][i]=pyr(uint8_t(src[c][i]-aligned[c]));}
    }
    auto ids=options(mode);size_t blocks=N/64;
    for(int c=0;c<2;c++){
        auto active=ids;if(conditional(mode)&&c==first_channel(mode))active.erase(std::remove_if(active.begin(),active.end(),[](int id){return id>7;}),active.end());
        std::vector<const uint8_t*> ptr;for(auto id:active)ptr.push_back(cand[c][id].data());choices[c].resize(blocks);
        need(ng_choose_blocks(ptr.data(),active.size(),initial_cl[c].data(),K,choices[c].data())==0);
        for(auto&v:choices[c])v=uint8_t(active[v]);
    }
    Causal cx(params[0]),cy(params[1]);void* states[]={cx.p,cy.p};std::array<Bytes,2>dst{Bytes(N),Bytes(N)};
    std::vector<Bytes>streams(2*K);uint64_t usage[3]={0,0,0};
    for(size_t i=0;i<N;i++)for(int step=0;step<2;step++){
        int c=step^first_channel(mode);
        uint16_t q;uint8_t cl;need(ng_causal_prediction(states[c],i,&q,&cl)==0);
        int choice=choices[c][block_index(i,d)];uint8_t p=pair_prediction(dst[0].data(),dst[1].data(),i,d,c,choice,q,mode);
        uint8_t r=uint8_t(src[c][i]-p);if(!choice&&((q>>7)&1))r=uint8_t(-r);
        streams[c*K+cl].push_back(pyr(r));dst[c][i]=src[c][i];need(ng_causal_accept(states[c],i,src[c][i],p)==0);
        usage[choice==0?0:choice<=7?1:2]++;
    }
    Bytes out{'P','D','A','1'};put(out,d);put(out,mode);uint64_t model=12+2*4+2*8+2*K*8,selectors=0,residuals=0;
    for(int c=0;c<2;c++){auto&p=params[c];put(out,p.size());out.insert(out.end(),p.begin(),p.end());model+=p.size();if(channel_bytes)channel_bytes[c]=6+4+8+K*8+p.size();}
    for(int c=0;c<2;c++){auto b=pack(choices[c]);put(out,choices[c].size());put(out,b.size());out.insert(out.end(),b.begin(),b.end());selectors+=b.size();if(channel_bytes)channel_bytes[c]+=b.size();}
    for(int k=0;k<2*K;k++){auto&s=streams[k];auto b=pack(s);put(out,s.size());put(out,b.size());out.insert(out.end(),b.begin(),b.end());residuals+=b.size();if(channel_bytes)channel_bytes[k/K]+=b.size();}
    if(stats){stats[0]=model;stats[1]=selectors;stats[2]=residuals;stats[3]=usage[0];stats[4]=usage[1];stats[5]=usage[2];}return out;
}
void pair_decode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap){
    need(n>=12&&!std::memcmp(src,"PDA1",4));size_t pos=4;int d=get(src,n,pos),mode=get(src,n,pos);need(d>=32&&d<=256&&d%32==0);auto ids=options(mode);
    size_t N=size_t(d)*d*d;need(cap>=N);Geometry geometry(d);std::array<Bytes,2>params,choices;std::vector<Bytes>streams(2*K);std::vector<size_t>used(2*K);
    for(auto&p:params){size_t len=get(src,n,pos);need(len<=n-pos&&len<=65536);p.assign(src+pos,src+pos+len);pos+=len;need(p.size()>=5&&p[4]==K);}
    for(int ch=0;ch<2;ch++){auto&c=choices[ch];size_t count=get(src,n,pos),len=get(src,n,pos);need(count==N/64&&len<=n-pos);c=unpack(src+pos,len,count);pos+=len;for(auto v:c){need(std::find(ids.begin(),ids.end(),v)!=ids.end());if(conditional(mode)&&ch==first_channel(mode))need(v<=7);}}
    size_t total=0;
    for(auto&s:streams){size_t count=get(src,n,pos),len=get(src,n,pos);need(count<=N&&len<=n-pos);s=unpack(src+pos,len,count);pos+=len;total+=count;}need(total==2*N&&pos==n);
    Causal cx(params[0]),cy(params[1]);void*states[]={cx.p,cy.p};uint8_t*dst[]={x,y};
    for(size_t i=0;i<N;i++)for(int step=0;step<2;step++){
        int c=step^first_channel(mode);
        uint16_t q;uint8_t cl;need(ng_causal_prediction(states[c],i,&q,&cl)==0);
        int choice=choices[c][block_index(i,d)];uint8_t p=pair_prediction(x,y,i,d,c,choice,q,mode);unsigned stream=c*K+cl;
        need(used[stream]<streams[stream].size());uint8_t r=unp(streams[stream][used[stream]++]);if(!choice&&((q>>7)&1))r=uint8_t(-r);
        uint8_t v=uint8_t(p+r);dst[c][i]=v;need(ng_causal_accept(states[c],i,v,p)==0);
    }
    for(size_t k=0;k<used.size();k++)need(used[k]==streams[k].size());
}
}
extern "C" int ng_pair_delta_encode(const uint8_t*x,const uint8_t*y,int d,int mode,uint8_t*out,size_t cap,size_t*n,uint64_t*stats){
    try{need(x&&y&&out&&n&&d>=32&&d<=256&&d%32==0);auto b=pair_encode(x,y,d,mode,stats);need(b.size()<=cap);std::memcpy(out,b.data(),b.size());*n=b.size();return 0;}catch(...){return -1;}
}
extern "C" int ng_pair_delta_decode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap){
    try{need(src&&x&&y);pair_decode(src,n,x,y,cap);return 0;}catch(...){return -1;}
}
extern "C" int ng_conditional_delta_encode(const uint8_t*x,const uint8_t*y,int d,int mode,uint8_t*out,size_t cap,size_t*n,uint64_t*stats){
    try{need(x&&y&&out&&n&&stats&&d>=32&&d<=256&&d%32==0);auto b=pair_encode(x,y,d,mode,stats,stats+6);need(b.size()<=cap);std::memcpy(out,b.data(),b.size());*n=b.size();return 0;}catch(...){return -1;}
}

namespace {
using Pair=std::array<int,2>;
int byte_center(int v){unsigned u=unsigned(v)&255u;return u>=128?int(u)-256:int(u);}
struct Lift {
    int a=0,b=0;
    explicit Lift(unsigned code=32896){
        int x=int(code>>8)-128,y=int(code&255)-128;
        if(x<0||(!x&&y<0)){x=-x;y=-y;}
        int r=int(isqrt(unsigned(x*x+y*y)));
        if(r){a=rounded(int64_t(y)*4096,r+x);b=rounded(-int64_t(y)*4096,r);}
    }
    Pair forward(Pair p)const{p[0]+=rounded(int64_t(a)*p[1],4096);p[1]+=rounded(int64_t(b)*p[0],4096);p[0]+=rounded(int64_t(a)*p[1],4096);return p;}
    Pair inverse(Pair p)const{p[0]-=rounded(int64_t(a)*p[1],4096);p[1]-=rounded(int64_t(b)*p[0],4096);p[0]-=rounded(int64_t(a)*p[1],4096);return p;}
    Pair modular(Pair p,bool inverse=false)const{
        int sign=inverse?-1:1;
        p[0]=byte_center(p[0]+sign*rounded(int64_t(a)*p[1],4096));
        p[1]=byte_center(p[1]+sign*rounded(int64_t(b)*p[0],4096));
        p[0]=byte_center(p[0]+sign*rounded(int64_t(a)*p[1],4096));
        return p;
    }
};
unsigned lift_reference(const uint8_t*x,const uint8_t*y,size_t i,int d,bool no_x=false){
    // All sources precede the first voxel of this4³ block in C-order.
    size_t anchor=((i/size_t(d)/d/4*4)*d+(i/d%d)/4*4)*d+(i%d)/4*4;
    size_t ids[3];int n=0;
    if(anchor%d&&!no_x)ids[n++]=anchor-1;
    if((anchor/d)%d)ids[n++]=anchor-d;
    if(anchor/size_t(d)/d)ids[n++]=anchor-size_t(d)*d;
    if(!n)return 32896;
    Vec v[3];for(int k=0;k<n;k++)v[k]=unit_axis(x[ids[k]],y[ids[k]]);
    int best=0;int64_t score=-1;
    for(int k=0;k<n;k++){int64_t s=0;for(int j=0;j<n;j++)s+=std::abs(dot(v[k],v[j]));if(s>score){score=s;best=k;}}
    size_t j=ids[best];return unsigned(x[j])*256+y[j];
}
bool folded(int kind){return kind==2||kind==3;}
bool rotated(int kind){return kind==1||kind==2||kind==4;}
int orientation(Pair p){return p[0]<0||(!p[0]&&p[1]<0);}
JOINT_INLINE Pair coordinates(Pair bytes,int kind,const Lift&l){
    bytes[0]-=128;bytes[1]-=128;return kind==4?l.modular(bytes):rotated(kind)?l.forward(bytes):bytes;
}
unsigned word(int difference,int modulus){int r=difference&(modulus-1);if(r>=modulus/2)r-=modulus;return unsigned(r<0?-2*r-1:2*r);}
int unword(unsigned w){return (w&1)?-int((w+1)/2):int(w/2);}
int wrap(int v,int modulus){v&=modulus-1;return v>=modulus/2?v-modulus:v;}
std::array<uint16_t,2> lifting_symbols(Pair value,Pair prediction,int kind,const Lift&l){
    auto a=coordinates(value,kind,l),p=coordinates(prediction,kind,l);
    if(!folded(kind)){int mod=(kind==0||kind==4)?256:512;return {uint16_t(word(a[0]-p[0],mod)),uint16_t(word(a[1]-p[1],mod))};}
    int s=orientation(a),sp=orientation(p);
    int ta=s?-a[1]:a[1],tp=sp?-p[1]:p[1];
    return {uint16_t(2*word(std::abs(a[0])-std::abs(p[0]),256)+(s^sp)),uint16_t(word(ta-tp,512))};
}
JOINT_INLINE Pair lifting_values_at(std::array<uint16_t,2> symbols,Pair p,int kind,const Lift&l){
    Pair a;
    need(symbols[0]<512&&symbols[1]<512);
    if(!folded(kind)){
        int mod=(kind==0||kind==4)?256:512;need(symbols[0]<mod&&symbols[1]<mod);
        a={wrap(p[0]+unword(symbols[0]),mod),wrap(p[1]+unword(symbols[1]),mod)};
    }else{
        int sp=orientation(p),s=sp^(symbols[0]&1),r=(std::abs(p[0])+unword(symbols[0]>>1))&255;
        int t=wrap((sp?-p[1]:p[1])+unword(symbols[1]),512);
        a={s?-r:r,s?-t:t};
    }
    if(kind==4)a=l.modular(a,true);else if(rotated(kind))a=l.inverse(a);a[0]+=128;a[1]+=128;
    need(a[0]>=0&&a[0]<=255&&a[1]>=0&&a[1]<=255);return a;
}
Pair lifting_values(std::array<uint16_t,2> symbols,Pair prediction,int kind,const Lift&l){
    return lifting_values_at(symbols,coordinates(prediction,kind,l),kind,l);
}
JOINT_INLINE Pair lifting_values_flip(std::array<uint16_t,2> symbols,Pair prediction,int kind,const Lift&l){
    if(!(symbols[0]|symbols[1]))return prediction;
    auto p=coordinates(prediction,kind,l);
    const int first_residual=unword(symbols[1]);
    // Margin8 cannot select the flipped predictor when the ordinary distance
    // is <=8. Otherwise adjust the predicted second coordinate directly:
    // adding its shift to the residual modulo256 is exactly equivalent.
    if(std::abs(first_residual)>8){
        const auto f=coordinates({(256-prediction[0])&255,(256-prediction[1])&255},kind,l);
        const int actual=wrap(p[1]+first_residual,256);
        if(std::abs(wrap(actual-f[1],256))+8<std::abs(first_residual))p[0]=f[0];
    }
    return lifting_values_at(symbols,p,kind,l);
}
#include "joint_reconstruct_simd.hpp"
#include "joint_contiguous.hpp"
std::vector<int> lifting_options(int mode){
    if(mode==82)return {2,4,6,66,68,70};
    if(mode==74||mode==76||mode==78||mode==79)return lifting_options(80);
    if(mode==75||mode==77)return lifting_options(12);
    if((mode>=70&&mode<=73)||(mode>=90&&mode<=92)||(mode>=100&&mode<=103))return lifting_options(12);
    if(mode==80||mode==81){std::vector<int>ids;for(int kind:{0,4})for(int p=1;p<16;p++)ids.push_back(kind*16+p);return ids;}
    if(mode>=50&&mode<=65)return lifting_options(12);
    if((mode>=20&&mode<=25)||(mode>=30&&mode<=35))return lifting_options(12);
    if(mode>=40&&mode<=45)return lifting_options(15);
    if(mode>=10){
        need(mode<=15);bool ng=mode>=13;int choice=(mode-10)%3;std::vector<int> ids;
        for(int kind:{0,4})if(choice==2||kind==(choice==0?0:4))for(int p=ng?0:1;p<8;p++)ids.push_back(kind*8+p);
        return ids;
    }
    need(mode>=0&&mode<=9);bool ng=mode==3||mode==4||mode==5||mode==7||mode==8;
    int kind=mode<6?mode%3:3;std::vector<int> ids;
    for(int t=0;t<4;t++)if(mode>=8||t==kind)for(int p=ng?0:1;p<8;p++)ids.push_back(t*8+p);
    return ids;
}
JOINT_INLINE Pair raw_prediction(const uint8_t*x,const uint8_t*y,size_t i,int pred,const uint16_t*q,int d){
    if(!pred)return {int(uint8_t(uint16_t(q[0]+128u)>>8)),int(uint8_t(uint16_t(q[1]+128u)>>8))};
    unsigned xx=i%d,yy=i/d%d,zz=i/size_t(d)/d;
    return {delta_predict_xyz(x,i,pred,xx,yy,zz,d,size_t(d)*d),delta_predict_xyz(y,i,pred,xx,yy,zz,d,size_t(d)*d)};
}
JOINT_INLINE Pair palette_prediction(const uint8_t*x,const uint8_t*y,size_t i,int d,int pred,const uint16_t*q,int base){
    if(base==16&&pred>=8){
        int dx=0,dy=0,dz=0;
        switch(pred){case 8:dx=2;break;case 9:dy=2;break;case 10:dz=2;break;case 11:dx=dy=1;break;case 12:dx=dz=1;break;case 13:dy=dz=1;break;case 14:dx=dy=dz=1;break;default:{unsigned r=lift_reference(x,y,i,d);return {int(r>>8),int(r&255)};}}
        if(int(i%d)<dx||int(i/d%d)<dy||int(i/size_t(d)/d)<dz)return {0,0};
        size_t j=i-dx-size_t(d)*dy-size_t(d)*d*dz;return {x[j],y[j]};
    }
    return raw_prediction(x,y,i,pred,q,d);
}
#include "joint_predict_simd.hpp"
struct JointExtras {
    int mode,first=0,contexts=1,groups=0;bool correction=false;
    std::vector<int32_t> weights;std::array<int, K> slopes{};
    explicit JointExtras(int m):mode(m){
        if(m>=70){first=1;contexts=2;}
        if(m>=50&&m<=65){first=m%2;contexts=m<=55?3:m<=57?4:m<=61?2:4;}
        if(m>=20&&m<=25){first=(m-20)%2;contexts=2<<((m-20)/2);}
        if(m>=30&&m<=35){first=(m-30)%2;correction=true;contexts=m>=34?2:m>=32?4:1;}
        if(m>=40&&m<=45){groups=(m%2)?K:1;weights.resize(groups*3);contexts=m>=44?2:m>=42?4:1;}
    }
    bool active()const{return mode>=20;}
    bool semantic()const{return (mode>=70&&mode<=79)||mode==82;}
    bool binary()const{return (mode>=72&&mode<=77)||mode==79;}
    bool semantic_seam()const{return mode==77||mode==79;}
    int semantic_classes()const{return mode==70?1:semantic_seam()?32:(mode>=73&&mode<=76)?16:4;}
    int base()const{return mode==82||mode==80||mode==81||mode==74||mode==76||mode==78||mode==79?16:8;}
    int passes()const{return mode==82||mode==90||mode==81||mode==76||mode==78||mode==79?1:mode==91||mode==75?2:0;}
    int regimes()const{return mode==102?4:(mode==100||mode==101)?2:1;}
    int nstreams()const{return mode==82?16:semantic()?binary()?4:7*semantic_classes()+12:mode>=100?K+4*regimes()*2:K*(1+contexts);}
    int bucket(unsigned s)const{
        if(contexts==1)return 0;
        if(contexts==2)return s!=0;
        if(contexts==4)return s==0?0:s<4?1:s<16?2:3;
        return s==0?0:s<2?1:s<4?2:s<8?3:s<16?4:s<32?5:s<64?6:7;
    }
    int stream(int c,int cls,unsigned first_symbol,int context=-1,int regime=0)const{
        if(c==first)return cls;
        if(mode>=100)return K+(cls/4*regimes()+regime)*2+int(first_symbol!=0);
        return K+cls*contexts+(context>=0?context:bucket(first_symbol));
    }
    int regime(const uint8_t*x,const uint8_t*y,size_t i,int d,Pair p)const{
        if(semantic_seam())return unit_axis(uint8_t(p[0]),uint8_t(p[1]))[2]<128;
        if(mode<100||mode==103)return 0;
        bool seam=unit(axis(uint8_t(p[0]),uint8_t(p[1])))[2]<128;
        if(mode==101)return seam;
        auto v=neighbors(x,y,i,d);bool boundary=false;
        for(int a=0;a<3;a++)for(int b=a+1;b<3;b++)boundary|=std::abs(dot(v[a],v[b]))<943718;
        return mode==102?int(boundary)+2*int(seam):int(boundary);
    }
    // All evidence depends only on the prediction and decoded FIRST coordinate.
    // Flip raw XY before applying the exact block transform, never negate its output.
    std::array<int,2> evidence(unsigned s,Pair prediction,int kind,const Lift&l)const{
        if(mode>=70)return JointExtras(61).evidence(s,prediction,kind,l);
        if(mode<50)return {-1,0};
        if(mode<=51)return {s==0?0:unword(s)<0?1:2,0};
        auto p=coordinates(prediction,kind,l);
        int actual=wrap(p[first]+unword(s),256);
        if(mode>=64){bool crosses=p[first]+unword(s)!=actual;return {2*int(s!=0)+int(crosses),0};}
        Pair flip{(256-prediction[0])&255,(256-prediction[1])&255};auto f=coordinates(flip,kind,l);
        int a=std::abs(wrap(actual-p[first],256)),b=std::abs(wrap(actual-f[first],256));
        int margin=(mode==52||mode==53||mode==58||mode==59)?0:8;
        int status=b+margin<a?1:a+margin<b?0:2;
        int context=mode<=55?status:mode<=57||mode>=62?(s==0?0:1+status):int(s!=0);
        int shift=mode>=58&&status==1?wrap(f[1-first]-p[1-first],256):0;
        return {context,shift};
    }
    static std::array<Vec,3> neighbors(const uint8_t*x,const uint8_t*y,size_t i,int d){
        std::array<Vec,3> v;size_t offsets[]={1,size_t(d),size_t(d)*d};bool valid[]={bool(i%d),bool((i/d)%d),bool(i/size_t(d)/d)};
        Vec ref=unit(axis(uint8_t(lift_reference(x,y,i,d)>>8),uint8_t(lift_reference(x,y,i,d))));
        for(int k=0;k<3;k++)v[k]=valid[k]?align(unit(axis(x[i-offsets[k]],y[i-offsets[k]])),ref):ref;
        return v;
    }
    void fit(const uint8_t*x,const uint8_t*y,int d,const Bytes&classes){
        if(!groups)return;std::vector<double>C(groups*9),rhs(groups*3);size_t N=size_t(d)*d*d;
        for(size_t i=0;i<N;i+=8){auto v=neighbors(x,y,i,d);Vec target=align(unit(axis(x[i],y[i])),v[0]);int g=groups==1?0:classes[i];
            for(int a=0;a<3;a++){rhs[g*3+a]+=double(dot(v[a],target));for(int b=0;b<3;b++)C[g*9+a*3+b]+=double(dot(v[a],v[b]));}}
        for(int g=0;g<groups;g++){for(int k=0;k<3;k++)C[g*9+k*3+k]+=1.;if(ng_axis_solve(C.data()+g*9,rhs.data()+g*3,3,weights.data()+g*3)!=0){weights[g*3]=256;weights[g*3+1]=weights[g*3+2]=0;}}
    }
    Pair predict(const uint8_t*x,const uint8_t*y,size_t i,int d,int pred,const uint16_t*q,int cls)const{
        if(pred||!groups)return palette_prediction(x,y,i,d,pred,q,base());
        auto v=neighbors(x,y,i,d);Vec p{};int g=groups==1?0:cls;
        for(int c=0;c<3;c++){int64_t s=0;for(int a=0;a<3;a++)s+=int64_t(weights[g*3+a])*v[a][c];p[c]=rounded(s,256);}
        p=unit(p);if(p[2]<0)for(auto&v:p)v=-v;
        return {std::clamp(128+rounded(int64_t(p[0])*127,1024),0,255),std::clamp(128+rounded(int64_t(p[1])*127,1024),0,255)};
    }
    void write(Bytes&out)const{
        if(!active())return;for(auto w:weights)put(out,uint32_t(w));if(correction)for(int s:slopes)put(out,uint32_t(s));
    }
    void read(const uint8_t*src,size_t n,size_t&pos){
        if(!active())return;for(auto&w:weights){uint32_t u=get(src,n,pos);int64_t v=u<=INT32_MAX?int64_t(u):int64_t(u)-4294967296LL;need(v>=-32000&&v<=32000);w=int32_t(v);}
        if(correction)for(auto&s:slopes){uint32_t u=get(src,n,pos);int64_t v=u<=INT32_MAX?int64_t(u):int64_t(u)-4294967296LL;need(v>=-64&&v<=64);s=int(v);}
    }
};
#include "joint_semantic.hpp"
// Scalar use of the SAME semantic grammar. Predictor fitting, activity and
// reconstruction remain the established native NG/Delta4 implementation.
Bytes scalar_semantic_encode(const uint8_t*params,size_t pn,const uint8_t*choices,const uint8_t*symbols,const uint8_t*classes,int d,int C){
    Geometry geometry(d);need(pn>=6&&params[4]==16&&(C==1||C==4||C==16));
    std::vector<uint32_t> rows(d*d);need(ng_wave_rows(params,pn,rows.data())==0);
    JointSemanticBinary model(C,1);std::array<ng_cabac::AuEncoder,4> enc;
    for(auto row:rows)for(int x=0;x<d;x++){
        size_t i=size_t(row)*d+x;need(classes[i]<16);uint8_t cls[]={classes[i],classes[i]};
        model.code(cls,enc.data(),nullptr,{symbols[i],0});
    }
    Bytes out{'N','S','C','1'};put(out,d);put(out,C);put(out,pn);out.insert(out.end(),params,params+pn);
    auto selectors=pack(Bytes(choices,choices+size_t(d)*d*d/64));put(out,selectors.size());out.insert(out.end(),selectors.begin(),selectors.end());
    for(auto&e:enc){auto b=e.finish();put(out,b.logical);put(out,b.bytes.size());out.insert(out.end(),b.bytes.begin(),b.bytes.end());}
    return out;
}
struct ScalarSemanticReader {
    JointSemanticBinary model;std::array<std::unique_ptr<ng_cabac::AuDecoder>,4> owned;ng_cabac::AuDecoder* dec[4]={};size_t used=0;
    explicit ScalarSemanticReader(int C):model(C,1){}
    static int read(void*ptr,size_t,unsigned c){
        try{auto&s=*static_cast<ScalarSemanticReader*>(ptr);need(c<16);uint8_t cls[]={uint8_t(c),uint8_t(c)};
            auto v=s.model.code(cls,nullptr,s.dec);s.used++;return v[0];
        }catch(...){return -1;}
    }
};
void scalar_semantic_decode(const uint8_t*src,size_t n,uint8_t*dst,size_t cap){
    need(n>=16&&!std::memcmp(src,"NSC1",4));size_t pos=4;int d=get(src,n,pos),C=get(src,n,pos);
    need(d>=32&&d<=256&&d%32==0&&(C==1||C==4||C==16));size_t N=size_t(d)*d*d;need(cap>=N);Geometry geometry(d);
    size_t pn=get(src,n,pos);need(pn>=6&&pn<=n-pos&&src[pos+4]==16);const uint8_t*params=src+pos;pos+=pn;
    size_t sn=get(src,n,pos);need(sn<=n-pos);auto choices=unpack(src+pos,sn,N/64);pos+=sn;for(auto v:choices)need(v<8);
    ScalarSemanticReader reader(C);
    for(int k=0;k<4;k++){
        uint32_t logical=get(src,n,pos),bytes=get(src,n,pos);need(bytes<=n-pos&&logical<=N*32);
        reader.owned[k]=std::make_unique<ng_cabac::AuDecoder>(std::span<const uint8_t>(src+pos,bytes),logical);pos+=bytes;reader.dec[k]=reader.owned[k].get();
    }
    need(pos==n);need(ng_block_stream_decode(params,pn,choices.data(),dst,0,4,ScalarSemanticReader::read,&reader)==0&&reader.used==N);
}
// Observation is optional and does not change the emitted frame.
struct JointObservation {uint8_t*symbols;uint8_t*classes;};
Bytes lifting_encode(const uint8_t*x,const uint8_t*y,int d,int mode,uint64_t*stats,JointObservation*observation=nullptr,int order=0){
    need(order==0||(order==1&&(mode==61||mode==78||mode==79||mode==82)));need(mode!=82||order==1);
    if(mode==92){uint64_t a[8],b[8];auto old=lifting_encode(x,y,d,61,a),next=lifting_encode(x,y,d,91,b);bool use=next.size()<old.size();if(stats)std::copy_n(use?b:a,8,stats);return use?next:old;}
    Geometry geometry(d);auto ids=lifting_options(mode);size_t N=size_t(d)*d*d,blocks=N/64,P=ids.size();
    JointExtras extra(mode);int parts=mode>=10?1:2,stream_count=extra.active()?extra.nstreams():2*parts*K;
    int cfg[]={3,3,1,16,40,128};const uint8_t*src[]={x,y};std::array<Bytes,2>params,sy,cl;std::array<std::vector<uint16_t>,2>q;
    for(int c=0;c<2;c++){
        params[c].resize(65536);sy[c].resize(N);cl[c].resize(N);q[c].resize(N);size_t pn=0;
        need(ng_ex_prepare(src[c],cfg,0,0,params[c].data(),params[c].size(),&pn,sy[c].data(),cl[c].data(),q[c].data())==0);params[c].resize(pn);
    }
    extra.fit(x,y,d,cl[0]);std::vector<Lift> rotations(blocks);
    for(int z=0;z<d;z+=4)for(int yv=0;yv<d;yv+=4)for(int xv=0;xv<d;xv+=4){size_t i=(size_t(z)*d+yv)*d+xv;rotations[block_index(i,d)]=Lift(lift_reference(x,y,i,d,mode==82));}
    std::vector<std::vector<Bytes>>cand(2*parts);
    for(auto&v:cand){v.resize(P);for(auto&b:v)b.resize(N);}
    for(size_t i=0;i<N;i++){
        auto&l=rotations[block_index(i,d)];uint16_t qq[]={q[0][i],q[1][i]};
        for(size_t k=0;k<P;k++){
            int id=ids[k];auto p=extra.predict(x,y,i,d,id%extra.base(),qq,cl[0][i]);auto s=lifting_symbols({x[i],y[i]},p,id/extra.base(),l);
            for(int c=0;c<2;c++)for(int h=0;h<parts;h++)cand[parts*c+h][k][i]=uint8_t(s[c]>>(8*h));
        }
    }
    std::vector<double>scores(blocks*P),part(blocks*P);
    for(int c=0;c<2*parts;c++){
        std::vector<const uint8_t*>ptr;for(auto&b:cand[c])ptr.push_back(b.data());
        need(ng_block_costs(ptr.data(),P,cl[c/parts].data(),K,part.data())==0);
        for(size_t k=0;k<scores.size();k++)scores[k]+=part[k];
    }
    Bytes choices(blocks);for(size_t b=0;b<blocks;b++){auto first=scores.begin()+b*P;choices[b]=uint8_t(ids[std::min_element(first,first+P)-first]);}
    if(extra.passes()){
        // Re-score under the selected joint residual distributions, including
        // the selector's estimated cost. Initial native classes are frozen here;
        // the final pass still evolves exact decoder classes and measures bytes.
        for(size_t i=0;i<N;i++)for(size_t k=0;k<P;k++){
            int id=ids[k];uint16_t qq[]={q[0][i],q[1][i]};auto p=extra.predict(x,y,i,d,id%extra.base(),qq,cl[0][i]);
            auto e=extra.evidence(cand[extra.first][k][i],p,id/extra.base(),rotations[block_index(i,d)]);
            auto&second=cand[1-extra.first][k][i];second=uint8_t(word(unword(second)-e[1],256));
        }
        std::array<int,256>index{};for(size_t k=0;k<P;k++)index[ids[k]]=int(k);
        for(int pass=0;pass<extra.passes();pass++){
            int scoring_streams=K*(1+extra.contexts);
            std::vector<std::array<uint32_t,256>>hist(scoring_streams);std::array<uint32_t,256>selector{};
            for(auto id:choices)selector[id]++;
            for(size_t i=0;i<N;i++){int k=index[choices[block_index(i,d)]];unsigned f=cand[extra.first][k][i];for(int c=0;c<2;c++)hist[extra.stream(c,cl[c][i],f)][cand[c][k][i]]++;}
            std::vector<std::array<double,256>>cost(scoring_streams);
            for(int s=0;s<scoring_streams;s++){double total=256;for(auto n:hist[s])total+=n;for(int v=0;v<256;v++)cost[s][v]=std::log2(total)-std::log2(1.+hist[s][v]);}
            std::fill(scores.begin(),scores.end(),0.);
            for(size_t i=0;i<N;i++){size_t b=block_index(i,d);for(size_t k=0;k<P;k++){unsigned f=cand[extra.first][k][i];for(int c=0;c<2;c++)scores[b*P+k]+=cost[extra.stream(c,cl[c][i],f)][cand[c][k][i]];}}
            for(size_t b=0;b<blocks;b++){for(size_t k=0;k<P;k++)scores[b*P+k]+=std::log2(double(blocks+P))-std::log2(1.+selector[ids[k]]);auto begin=scores.begin()+b*P;choices[b]=uint8_t(ids[std::min_element(begin,begin+P)-begin]);}
        }
    }
    // Release candidate volumes before final causal encode.
    for(auto&v:cand)v.clear();Causal cx(params[0]),cy(params[1]);void*states[]={cx.p,cy.p};std::array<Bytes,2>dst{Bytes(N),Bytes(N)};
    std::vector<Bytes>streams(stream_count);uint64_t usage[4]={0,0,0,0};std::array<Bytes,2>final_symbols{Bytes(N),Bytes(N)},final_classes{Bytes(N),Bytes(N)};Bytes evidence_context(N);
    for(size_t i=0;i<N;i++){
        uint16_t qq[2];uint8_t cls[2];for(int c=0;c<2;c++)need(ng_causal_prediction(states[c],i,qq+c,cls+c)==0);
        int id=choices[block_index(i,d)],kind=id/extra.base(),pred=id%extra.base();auto p=extra.predict(dst[0].data(),dst[1].data(),i,d,pred,qq,cls[0]);
        auto&l=rotations[block_index(i,d)];auto s=lifting_symbols({x[i],y[i]},p,kind,l);
        if(mode>=50){auto e=extra.evidence(s[extra.first],p,kind,l);evidence_context[i]=uint8_t(mode>=100||extra.semantic_seam()?extra.regime(dst[0].data(),dst[1].data(),i,d,p):e[0]);s[1-extra.first]=uint16_t(word(unword(s[1-extra.first])-e[1],256));}
        for(int c=0;c<2;c++){if(extra.active()){final_symbols[c][i]=uint8_t(s[c]);final_classes[c][i]=cls[c];}else for(int h=0;h<parts;h++)streams[(parts*c+h)*K+cls[c]].push_back(uint8_t(s[c]>>(8*h)));dst[c][i]=src[c][i];need(ng_causal_accept(states[c],i,src[c][i],uint8_t(p[c]))==0);}
        usage[0]+=pred==0;usage[1]+=rotated(kind);usage[2]+=folded(kind);usage[3]+=folded(kind)&&(s[0]&1);
    }
    if(observation)for(int c=0;c<2;c++){std::copy(final_symbols[c].begin(),final_symbols[c].end(),observation->symbols+c*N);std::copy(final_classes[c].begin(),final_classes[c].end(),observation->classes+c*N);}
    std::vector<uint32_t>counts(stream_count);JointSemantic semantic(extra.semantic_classes());JointSemanticBinary binary(extra.semantic_classes());
    std::vector<uint32_t> row_order(d*d);
    if(order)need(ng_wave_rows(params[0].data(),params[0].size(),row_order.data())==0);
    else std::iota(row_order.begin(),row_order.end(),0);
    if(extra.semantic()){
        std::array<ng_cabac::AuEncoder,4>enc;
        for(auto row:row_order)for(int xv=0;xv<d;xv++){size_t i=size_t(row)*d+xv;std::array<uint16_t,2>s{final_symbols[0][i],final_symbols[1][i]};uint8_t cls[]={final_classes[0][i],final_classes[1][i]};if(mode==82)JointSemantic::emit_tags(s,semantic.context(cls),streams);else if(extra.binary())binary.code(cls,enc.data(),nullptr,s,extra.semantic_seam()?std::max(cls[0],cls[1])+16*evidence_context[i]:-1);else semantic.emit(s,cls,streams);}
        if(extra.binary())for(int k=0;k<4;k++){auto b=enc[k].finish();counts[k]=b.logical;streams[k]=std::move(b.bytes);}
    }else if(extra.active()){
        int first=extra.first,second=1-first;
        if(extra.correction){
            int64_t xy[K]={},xx[K]={};for(size_t i=0;i<N;i++){int a=unword(final_symbols[first][i]),b=unword(final_symbols[second][i]),c=final_classes[second][i];xy[c]+=a*b;xx[c]+=a*a;}
            for(int c=0;c<K;c++)extra.slopes[c]=xx[c]?std::clamp(int(std::llround(16.*xy[c]/xx[c])),-64,64):0;
        }
        for(auto row:row_order)for(int xv=0;xv<d;xv++)for(int c=0;c<2;c++){
            size_t i=size_t(row)*d+xv;
            unsigned s=final_symbols[c][i],f=final_symbols[first][i];int cls=final_classes[c][i];
            if(c==second&&extra.correction)s=word(unword(s)-rounded(int64_t(extra.slopes[cls])*unword(f),16),256);
            streams[extra.stream(c,cls,f,mode>=50&&mode<100?evidence_context[i]:-1,mode>=100?evidence_context[i]:0)].push_back(uint8_t(s));
        }
    }
    Bytes out{'L','F','P',uint8_t(extra.active()?'3':parts==1?'2':'1')};put(out,d);put(out,mode);uint64_t model=12+8+8+stream_count*8,selectors=0,low=0,high=0;
    if(order){out[3]='4';put(out,order);model+=4;}
    if(mode==82){out[3]='5';put(out,3);model+=4;}
    size_t before=out.size();extra.write(out);model+=out.size()-before;
    for(auto&p:params){put(out,p.size());out.insert(out.end(),p.begin(),p.end());model+=p.size();}
    auto selector=pack(choices);put(out,choices.size());put(out,selector.size());out.insert(out.end(),selector.begin(),selector.end());selectors=selector.size();
    for(int k=0;k<stream_count;k++){auto b=extra.binary()?streams[k]:pack(streams[k]);put(out,extra.binary()?counts[k]:streams[k].size());put(out,b.size());out.insert(out.end(),b.begin(),b.end());if(parts==2&&(k/K)%2)high+=b.size();else low+=b.size();}
    if(stats){stats[0]=model;stats[1]=selectors;stats[2]=low;stats[3]=high;for(int k=0;k<4;k++)stats[4+k]=usage[k];}return out;
}
template<bool Profile=false,int Dim=0,int Mode=0,bool Wave=false,int Trial=0>
void lifting_decode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap,uint64_t*timings=nullptr,std::vector<Bytes>*capture=nullptr,int capture_grammar=0){
    using Clock=std::chrono::steady_clock;
    Clock::time_point stamp;
    if constexpr(Profile){std::fill(timings,timings+7,0);stamp=Clock::now();}
    auto tick=[&](int k){if constexpr(Profile){auto now=Clock::now();timings[k]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-stamp).count();stamp=now;}};
    need(n>=12&&!std::memcmp(src,"LFP",3)&&(src[3]>='1'&&src[3]<='5'));size_t pos=4;int stored_d=get(src,n,pos),stored_mode=get(src,n,pos);
    const int d=Dim?Dim:stored_d,mode=Mode?Mode:stored_mode;
    need(d==stored_d&&mode==stored_mode&&d>=32&&d<=256&&d%32==0);auto ids=lifting_options(mode);
    int order=src[3]>='4'?int(get(src,n,pos)):0;
    int grammar=src[3]=='5'?int(get(src,n,pos)):0;
    need(!grammar||(order==1&&((Mode==82&&grammar==3)||(Mode==78&&Trial>=20&&Trial<=30&&(grammar==(Trial>=22?3:Trial-19)||(grammar==4&&Trial==25))))));
    need(src[3]!='4'||(order==1&&Mode&&(mode==61||mode==78||mode==79)));
    JointExtras extra(mode);int parts=mode>=10?1:2,stream_count=grammar?(grammar==1?12:grammar==2?4:16):extra.active()?extra.nstreams():2*parts*K;need(src[3]==(grammar?'5':order?'4':extra.active()?'3':parts==1?'2':'1'));extra.read(src,n,pos);
    const int base=Mode?(Mode==61?8:16):extra.base();
    const bool semantic_mode=Mode?(Mode!=61):extra.semantic(),binary_mode=Mode?(Mode==79):extra.binary();
    size_t N=size_t(d)*d*d;need(cap>=N);Geometry geometry(d);std::array<Bytes,2>params;
    for(auto&p:params){size_t count=get(src,n,pos);need(count<=65536&&count<=n-pos);p.assign(src+pos,src+pos+count);pos+=count;need(p.size()>=5&&p[4]==K);}
    std::vector<uint32_t> row_order;
    if(order){row_order.resize(d*d);need(ng_wave_rows(params[0].data(),params[0].size(),row_order.data())==0);}
    size_t count=get(src,n,pos),bytes=get(src,n,pos);need(count==N/64&&bytes<=n-pos);auto choices=unpack(src+pos,bytes,count);pos+=bytes;
    for(int id:choices)need(std::find(ids.begin(),ids.end(),id)!=ids.end());
    static thread_local std::vector<Bytes> output_pool;
    std::vector<Bytes>streams;
    if constexpr(Trial==18||Trial==19)streams.swap(output_pool);
    streams.resize(stream_count);
    struct ReturnPool{std::vector<Bytes>&s;std::vector<Bytes>*pool;~ReturnPool(){if(pool)s.swap(*pool);}} return_pool{streams,(Trial==18||Trial==19)?&output_pool:nullptr};
    std::vector<size_t>used(stream_count);std::vector<uint32_t>counts(stream_count);size_t total=0;
    for(int k=0;k<stream_count;k++){auto&s=streams[k];count=get(src,n,pos);bytes=get(src,n,pos);need(count<=N*(extra.binary()?32:grammar==2?2:1)&&bytes<=n-pos);counts[k]=uint32_t(count);s=grammar==4&&k>=4?unpack_bits(src+pos,bytes,count,(k-4)%6+1):extra.binary()?Bytes(src+pos,src+pos+bytes):unpack(src+pos,bytes,count,Trial==1||Trial==3||Trial==12,Profile&&Trial>=10?timings+6:nullptr,std::move(s),Trial==17||Trial==19);pos+=bytes;total+=count;}need(pos==n&&(extra.semantic()||total==size_t(2*parts)*N));
    std::vector<const uint8_t*> cursors,ends;static const uint8_t empty_stream=0;
    if constexpr(Mode==82||Trial==2||Trial==3||Trial>=12){cursors.resize(stream_count);ends.resize(stream_count);for(int k=0;k<stream_count;k++){cursors[k]=streams[k].empty()?&empty_stream:streams[k].data();ends[k]=cursors[k]+streams[k].size();}}
    const int semantic_classes=Mode==78?4:Mode==79?32:extra.semantic_classes();
    JointSemantic semantic(semantic_classes);JointSemanticBinary binary(binary_mode?semantic_classes:0);std::array<std::unique_ptr<ng_cabac::AuDecoder>,4>owned;ng_cabac::AuDecoder*dec[4]={};
    if(extra.binary())for(int k=0;k<4;k++){owned[k]=std::make_unique<ng_cabac::AuDecoder>(streams[k],counts[k]);dec[k]=owned[k].get();}
    Causal cx(params[0]),cy(params[1]);void*states[]={cx.p,cy.p};uint8_t*dst[]={x,y};std::vector<Lift>rotations(N/64);std::vector<bool>ready(N/64);
    const uint8_t*row_classes[2]={};uint8_t*row_errors[2]={};
    if constexpr(Mode==82){
        JointBatch q{};alignas(32) int masks[32];
        for(auto row:row_order){int yy=row%d,zz=row/d;const uint8_t*cl[2];uint8_t*err[2];
            for(int c=0;c<2;c++)ng_causal_activity_begin(states[c],zz,yy,cl+c,err+c);
            for(int xx=0;xx<d;xx+=32){size_t i=size_t(row)*d+xx;
                for(int k=0;k<32;k++){
                    size_t b=block_index(i+k,d);int id=choices[b];
                    if(!((yy|zz|(xx+k))&3)&&id/16)rotations[b]=Lift(lift_reference(x,y,i+k,d,true));
                    masks[k]=(id%16)&((yy?2:0)|(zz?4:0));q.a[k]=id/16?rotations[b].a:0;q.b[k]=id/16?rotations[b].b:0;
                    auto s=semantic.read_tags<true>(std::max(cl[0][xx+k],cl[1][xx+k])/4,cursors.data(),ends.data());q.sx[k]=s[0];q.sy[k]=s[1];
                }
                joint_contiguous_predict(x,y,i,d,yy,zz,masks,q,32,Trial!=1);
                joint_batch_reconstruct<true>(q,32,Trial!=1);
                for(int k=0;k<32;k++){x[i+k]=uint8_t(q.vx[k]);y[i+k]=uint8_t(q.vy[k]);err[0][xx+k]=uint8_t(q.vx[k]>>8);err[1][xx+k]=uint8_t(q.vy[k]>>8);}
            }
            for(int c=0;c<2;c++)ng_causal_activity_end(states[c],zz,yy);
        }
        for(int k=0;k<stream_count;k++)need(cursors[k]==ends[k]);return;
    }
    if constexpr(Mode)for(int id:choices)need(id%base!=0);
    // Independent native wavefront rows: decode their semantics before any
    // reconstruction, then interleave X positions across rows. Activity uses
    // prior wavefronts only; residual errors have no same-row dependency.
    std::vector<uint32_t> group_ends;
    uint32_t groups=0,group=0,start_row=0,end_row=0,wave_lane=0,wave_x=0;
    std::vector<std::array<uint16_t,2>> group_symbols;
    std::array<std::vector<uint8_t*>,2> group_errors;
    if constexpr(Wave){
        static_assert(Mode==78);need(order==1);
        group_ends.resize(d*d);need(ng_wave_groups(params[0].data(),params[0].size(),row_order.data(),group_ends.data(),&groups)==0);
        group_symbols.resize(32*d);for(auto&e:group_errors)e.resize(32);
    }
    if constexpr(Trial>=29)JointSemantic::validate_tag_ranges(streams);
    else if constexpr(Trial>=27)JointSemantic::validate_tag_streams(streams);
    tick(0);
    if constexpr(Trial>=10){
        static_assert(Mode==78);need(order==1);std::vector<uint32_t> ge(d*d);uint32_t ng=0;
        need(ng_wave_groups(params[0].data(),params[0].size(),row_order.data(),ge.data(),&ng)==0);
        std::vector<std::array<uint16_t,2>> symbols(Trial>=29?0:32*d);std::array<uint8_t*,32> errors[2];
        std::vector<uint8_t> group_tags(Trial>=29?32*d:0);
        std::array<const uint8_t*,12> row_remainders[32],row_remainder_ends[32];
        std::vector<unsigned> references(N/64);JointBatch batch{};JointPrediction prediction;std::array<size_t,32> locations{};
        JointPreparedPredictor prepared;std::array<size_t,32> row_base{},block_base{};std::array<int,32> row_y{},row_z{},block_kind{};
        auto prepare_block=[&](int lane,int xx){
            size_t b=block_base[lane]+xx/4,i=row_base[lane]+xx;int id=choices[b];
            if((row_y[lane]&3)==0&&(row_z[lane]&3)==0&&(id/base||id%base==15)){
                references[b]=lift_reference(x,y,i,d);if(id/base)rotations[b]=Lift(references[b]);
            }
            prepared.prepare(lane,id%base,xx,row_y[lane],row_z[lane],d,references[b]);
            block_kind[lane]=id/base?4:0;
            batch.a[lane]=id/base?rotations[b].a:0;batch.b[lane]=id/base?rotations[b].b:0;
        };
        tick(0); // include schedule and staging-buffer setup in the profile
        uint32_t begin=0;
        for(uint32_t g=0;g<ng;g++){
            int lanes=ge[g]-begin;need(lanes>0&&lanes<=32);
            if constexpr(Trial>=24)for(int lane=0;lane<lanes;lane++){
                auto row=row_order[begin+lane];row_base[lane]=size_t(row)*d;block_base[lane]=block_index(row_base[lane],d);row_y[lane]=row%d;row_z[lane]=row/d;
            }
            for(int lane=0;lane<lanes;lane++){
                auto row=row_order[begin+lane];const uint8_t*cl[2];
                if constexpr(Profile)stamp=Clock::now();
                for(int c=0;c<2;c++)ng_causal_activity_begin(states[c],row/d,row%d,cl+c,&errors[c][lane]);
                tick(1);
                if constexpr(Trial>=29){
                    std::array<size_t,12> counts{};
                    for(int xx=0;xx<d;xx++){
                        int c=std::max(cl[0][xx],cl[1][xx])/4;need(cursors[c]!=ends[c]);
                        unsigned tags=*cursors[c]++;group_tags[xx*32+lane]=tags;
                        for(int k=0;k<2;k++){int r=JointSemantic::tag_info((tags>>(4*k))&15).remainder;if(r>=0)counts[k*6+r]++;}
                    }
                    for(int k=0;k<12;k++){
                        need(counts[k]<=size_t(ends[4+k]-cursors[4+k]));
                        row_remainders[lane][k]=cursors[4+k];cursors[4+k]+=counts[k];row_remainder_ends[lane][k]=cursors[4+k];
                    }
                    tick(2);continue;
                }
                if constexpr(Trial==26){
                    for(int xx=0;xx<d;xx++){
                        if(!(xx&3))prepare_block(lane,xx);
                        auto s=semantic.read_tags<true>(std::max(cl[0][xx],cl[1][xx])/4,cursors.data(),ends.data());
                        size_t i=row_base[lane]+xx;auto p=prepared.predict(lane,xx,x,y,i,d);
                        Lift l;l.a=batch.a[lane];l.b=batch.b[lane];auto v=lifting_values_flip(s,p,block_kind[lane],l);
                        x[i]=uint8_t(v[0]);y[i]=uint8_t(v[1]);
                        for(int c=0;c<2;c++)errors[c][lane][xx]=uint8_t(std::abs(int8_t(uint8_t(v[c]-p[c]))));
                    }
                    tick(4);for(int c=0;c<2;c++)ng_causal_activity_end(states[c],row/d,row%d);continue;
                }
                if constexpr(Trial==14||Trial==16||Trial==19)semantic.read_row(d,cl[0],cl[1],cursors.data(),ends.data(),symbols.data()+lane*d);
                else for(int xx=0;xx<d;xx++){uint8_t cc[]={cl[0][xx],cl[1][xx]};
                    if constexpr(Trial>=20&&Trial<=30){
                        int c=semantic.context(cc);auto getbyte=[&](int k){need(cursors[k]!=ends[k]);return uint16_t(*cursors[k]++);};
                        if constexpr(Trial==20){int p=getbyte(c);need(p<4);symbols[lane*d+xx]={p&1?getbyte(4+c):uint16_t(0),p&2?getbyte(8+c):uint16_t(0)};}
                        else if constexpr(Trial==21)symbols[lane*d+xx]={getbyte(c),getbyte(c)};
                        else symbols[lane*d+xx]=semantic.read_tags<Trial==23||Trial==25||Trial>=27,Trial>=27>(c,cursors.data(),ends.data());
                    }
                    else if constexpr(Trial>=12)symbols[lane*d+xx]=semantic.read_pointers(cc,cursors.data(),ends.data());
                    else symbols[lane*d+xx]=semantic.read(cc,streams,used);
                }
                if(capture)for(int xx=0;xx<d;xx++){
                    int c=std::max(cl[0][xx],cl[1][xx])/4;auto s=symbols[lane*d+xx];
                    if(capture_grammar==1){(*capture)[c].push_back((s[0]!=0)+2*(s[1]!=0));for(int k=0;k<2;k++)if(s[k])(*capture)[4+4*k+c].push_back(uint8_t(s[k]));}
                    else if(capture_grammar==2)for(int k=0;k<2;k++)(*capture)[c].push_back(uint8_t(s[k]));
                    else JointSemantic::emit_tags(s,c,*capture);
                }
                tick(2);
            }
            if constexpr(Trial==26){begin=ge[g];continue;}
            for(int xx=0;xx<d;xx++){
                if constexpr(Profile)stamp=Clock::now();
                for(int lane=0;lane<lanes;lane++){
                    if constexpr(Trial==24||Trial==25||Trial>=27){
                        if(!(xx&3))prepare_block(lane,xx);
                        size_t i=row_base[lane]+xx;locations[lane]=i;
                        auto p=Trial==28?prepared.predict_packed(lane,xx,x,y,i,d):prepared.predict(lane,xx,x,y,i,d);
                        batch.px[lane]=p[0];batch.py[lane]=p[1];
                        auto s=Trial>=29?JointSemantic::assemble_tags<true>(group_tags[xx*32+lane],row_remainders[lane].data(),nullptr):symbols[lane*d+xx];batch.sx[lane]=s[0];batch.sy[lane]=s[1];
                        continue;
                    }
                    size_t i=size_t(row_order[begin+lane])*d+xx,b=block_index(i,d);locations[lane]=i;
                    int id=choices[b];
                    if(((i|i/d|i/size_t(d)/d)&3)==0&&(id/base||id%base==15)){
                        references[b]=lift_reference(x,y,i,d);if(id/base)rotations[b]=Lift(references[b]);
                    }
                    if constexpr(Trial==15||Trial==16)prediction.set(lane,i,d,id%base,references[b]);
                    else {uint16_t qq[2]={};auto p=id%base==15?Pair{int(references[b]>>8),int(references[b]&255)}:palette_prediction(x,y,i,d,id%base,qq,base);batch.px[lane]=p[0];batch.py[lane]=p[1];}
                    batch.a[lane]=id/base?rotations[b].a:0;batch.b[lane]=id/base?rotations[b].b:0;
                    auto s=symbols[lane*d+xx];batch.sx[lane]=s[0];batch.sy[lane]=s[1];
                }
                if constexpr(Trial==15||Trial==16)joint_predict_batch(prediction,x,y,batch,lanes);
                tick(3);
                joint_batch_reconstruct<Trial==30>(batch,lanes,Trial>=11);tick(4);
                for(int lane=0;lane<lanes;lane++){
                    x[locations[lane]]=uint8_t(batch.vx[lane]);y[locations[lane]]=uint8_t(batch.vy[lane]);
                    if constexpr(Trial==30){errors[0][lane][xx]=uint8_t(batch.vx[lane]>>8);errors[1][lane][xx]=uint8_t(batch.vy[lane]>>8);}
                    else {errors[0][lane][xx]=uint8_t(std::abs(int8_t(uint8_t(batch.vx[lane]-batch.px[lane]))));
                        errors[1][lane][xx]=uint8_t(std::abs(int8_t(uint8_t(batch.vy[lane]-batch.py[lane]))));}
                }tick(5);
            }
            for(int lane=0;lane<lanes;lane++){auto row=row_order[begin+lane];for(int c=0;c<2;c++)ng_causal_activity_end(states[c],row/d,row%d);}
            if constexpr(Trial>=29)for(int lane=0;lane<lanes;lane++)need(row_remainders[lane]==row_remainder_ends[lane]);
            begin=ge[g];
        }
        if constexpr(Trial>=12){for(int k=0;k<stream_count;k++)need(cursors[k]==ends[k]);}
        else for(int k=0;k<stream_count;k++)need(used[k]==streams[k].size());
        return;
    }
    for(size_t traversal=0;traversal<N;traversal++){
        size_t i=order?size_t(row_order[traversal/d])*d+traversal%d:traversal;
        size_t lane=0;
        if constexpr(Wave){
            if(traversal==size_t(end_row)*d){
                need(group<groups);start_row=end_row;end_row=group_ends[group++];need(end_row-start_row<=32);
                wave_lane=0;wave_x=0;
                for(uint32_t r=start_row;r<end_row;r++){
                    uint32_t row=row_order[r];const uint8_t*cl[2];
                    // begin() uses one reusable class row per channel; consume
                    // both class rows before preparing the next spatial row.
                    for(int c=0;c<2;c++)ng_causal_activity_begin(states[c],row/d,row%d,cl+c,&group_errors[c][r-start_row]);
                    for(int xx=0;xx<d;xx++){uint8_t cc[]={cl[0][xx],cl[1][xx]};group_symbols[(r-start_row)*d+xx]=semantic.read(cc,streams,used);}
                }
            }
            lane=wave_lane;i=size_t(row_order[start_row+lane])*d+wave_x;
            if(++wave_lane==end_row-start_row){wave_lane=0;++wave_x;}
        }
        bool sample=false;if constexpr(Profile){sample=i%257==0;if(sample){stamp=Clock::now();timings[6]++;}}
        size_t b=block_index(i,d);
        int id=choices[b];
        if constexpr(Mode){
            if(id/base!=0&&((i|i/d|i/size_t(d)/d)&3)==0)rotations[b]=Lift(lift_reference(x,y,i,d));
        }else if(!ready[b]){rotations[b]=Lift(lift_reference(x,y,i,d));ready[b]=true;}
        if(sample)tick(1);
        uint16_t qq[2]={};uint8_t cls[2];
        if constexpr(Wave){
            for(int c=0;c<2;c++)row_errors[c]=group_errors[c][lane];
        }else if constexpr(Mode){
            if(i%d==0)for(int c=0;c<2;c++)ng_causal_activity_begin(states[c],i/size_t(d)/d,i/d%d,row_classes+c,row_errors+c);
            for(int c=0;c<2;c++)cls[c]=row_classes[c][i%d];
        }else for(int c=0;c<2;c++)need(ng_causal_prediction(states[c],i,qq+c,cls+c)==0);
        if(sample)tick(2);
        auto p=Mode?palette_prediction(x,y,i,d,id%base,qq,base):extra.predict(x,y,i,d,id%base,qq,cls[0]);std::array<uint16_t,2>sym{};
        if(sample)tick(3);
        if constexpr(Wave){sym=group_symbols[lane*d+i%d];
        }else if constexpr(Mode==61){
            int s=cls[1];need(used[s]<streams[s].size());sym[1]=streams[s][used[s]++];
            s=K+2*cls[0]+int(sym[1]!=0);need(used[s]<streams[s].size());sym[0]=streams[s][used[s]++];
        }else if(semantic_mode){
            int context=-1;
            if constexpr(Mode==79)context=std::max(cls[0],cls[1])+16*(unit_axis(uint8_t(p[0]),uint8_t(p[1]))[2]<128);
            else if(extra.semantic_seam())context=std::max(cls[0],cls[1])+16*extra.regime(x,y,i,d,p);
            if constexpr(Trial==2||Trial==3)sym=semantic.read_pointers(cls,cursors.data(),ends.data());
            else sym=binary_mode?binary.code(cls,nullptr,dec,{},context):semantic.read(cls,streams,used);
            if constexpr(!Mode){auto e=extra.evidence(sym[extra.first],p,id/extra.base(),rotations[b]);sym[1-extra.first]=uint16_t(word(unword(sym[1-extra.first])+e[1],256));}
        }else if(extra.active()){
            int regime=extra.regime(x,y,i,d,p);
            for(int step=0;step<2;step++){auto e=step&&!Mode?extra.evidence(sym[extra.first],p,id/extra.base(),rotations[b]):std::array<int,2>{-1,0};
                int c=step^extra.first,stream=extra.stream(c,cls[c],sym[extra.first],e[0],regime);need(used[stream]<streams[stream].size());unsigned s=streams[stream][used[stream]++];
                if(step&&extra.correction)s=word(unword(s)+rounded(int64_t(extra.slopes[cls[c]])*unword(sym[extra.first]),16),256);
                if(step&&mode>=50&&!Mode)s=word(unword(s)+e[1],256);sym[c]=uint16_t(s);}
        }else for(int c=0;c<2;c++)for(int h=0;h<parts;h++){int stream=(parts*c+h)*K+cls[c];need(used[stream]<streams[stream].size());unsigned v=streams[stream][used[stream]++];need(!h||v<=1);sym[c]|=uint16_t(v<<(8*h));}
        if(sample)tick(4);
        auto v=Mode?lifting_values_flip(sym,p,id/base,rotations[b]):lifting_values(sym,p,id/base,rotations[b]);for(int c=0;c<2;c++){
            dst[c][i]=uint8_t(v[c]);
            if constexpr(Mode){int e=int8_t(uint8_t(v[c]-p[c]));row_errors[c][i%d]=uint8_t(std::abs(e));}
            else need(ng_causal_accept(states[c],i,uint8_t(v[c]),uint8_t(p[c]))==0);
        }
        if constexpr(Mode)if(i%d==size_t(d-1))for(int c=0;c<2;c++)ng_causal_activity_end(states[c],i/size_t(d)/d,i/d%d);
        if(sample)tick(5);
    }
    if constexpr(Trial==2||Trial==3){for(int k=0;k<stream_count;k++)need(cursors[k]==ends[k]);}
    else if(!extra.binary())for(size_t k=0;k<used.size();k++)need(used[k]==streams[k].size());
}
template<bool Profile,int D>void lifting_trial_dispatch(const uint8_t*s,size_t n,uint8_t*x,uint8_t*y,size_t cap,int trial,uint64_t*t){
    switch(trial){
    case 1:lifting_decode<Profile,D,78,false,1>(s,n,x,y,cap,t);break;
    case 2:lifting_decode<Profile,D,78,false,2>(s,n,x,y,cap,t);break;
    case 3:lifting_decode<Profile,D,78,false,3>(s,n,x,y,cap,t);break;
    case 10:lifting_decode<Profile,D,78,false,10>(s,n,x,y,cap,t);break;
    case 11:lifting_decode<Profile,D,78,false,11>(s,n,x,y,cap,t);break;
    case 12:lifting_decode<Profile,D,78,false,12>(s,n,x,y,cap,t);break;
    case 13:lifting_decode<Profile,D,78,false,13>(s,n,x,y,cap,t);break;
    case 14:lifting_decode<Profile,D,78,false,14>(s,n,x,y,cap,t);break;
    case 15:lifting_decode<Profile,D,78,false,15>(s,n,x,y,cap,t);break;
    case 16:lifting_decode<Profile,D,78,false,16>(s,n,x,y,cap,t);break;
    case 17:lifting_decode<Profile,D,78,false,17>(s,n,x,y,cap,t);break;
    case 18:lifting_decode<Profile,D,78,false,18>(s,n,x,y,cap,t);break;
    case 19:lifting_decode<Profile,D,78,false,19>(s,n,x,y,cap,t);break;
    case 20:lifting_decode<Profile,D,78,false,20>(s,n,x,y,cap,t);break;
    case 21:lifting_decode<Profile,D,78,false,21>(s,n,x,y,cap,t);break;
    case 22:lifting_decode<Profile,D,78,false,22>(s,n,x,y,cap,t);break;
    case 23:lifting_decode<Profile,D,78,false,23>(s,n,x,y,cap,t);break;
    case 24:lifting_decode<Profile,D,78,false,24>(s,n,x,y,cap,t);break;
    case 25:lifting_decode<Profile,D,78,false,25>(s,n,x,y,cap,t);break;
    case 26:lifting_decode<Profile,D,78,false,26>(s,n,x,y,cap,t);break;
    case 27:lifting_decode<Profile,D,78,false,27>(s,n,x,y,cap,t);break;
    case 28:lifting_decode<Profile,D,78,false,28>(s,n,x,y,cap,t);break;
    case 29:lifting_decode<Profile,D,78,false,29>(s,n,x,y,cap,t);break;
    case 30:lifting_decode<Profile,D,78,false,30>(s,n,x,y,cap,t);break;
    default:need(false);
    }
}
template<bool Profile,int D>
void lifting_dispatch_mode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap,int mode,uint64_t*timings){
    switch(mode){
    case 61: lifting_decode<Profile,D,61>(src,n,x,y,cap,timings);break;
    case 78: lifting_decode<Profile,D,78>(src,n,x,y,cap,timings);break;
    case 79: lifting_decode<Profile,D,79>(src,n,x,y,cap,timings);break;
    case 82: lifting_decode<Profile,D,82>(src,n,x,y,cap,timings);break;
    default:lifting_decode<Profile>(src,n,x,y,cap,timings);
    }
}
template<bool Profile=false>
void lifting_dispatch(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap,uint64_t*timings=nullptr){
    size_t pos=4;int d=get(src,n,pos),mode=get(src,n,pos);
    switch(d){
    case 32:lifting_dispatch_mode<Profile,32>(src,n,x,y,cap,mode,timings);break;
    case 64:lifting_dispatch_mode<Profile,64>(src,n,x,y,cap,mode,timings);break;
    case 128:lifting_dispatch_mode<Profile,128>(src,n,x,y,cap,mode,timings);break;
    case 256:lifting_dispatch_mode<Profile,256>(src,n,x,y,cap,mode,timings);break;
    default:lifting_dispatch_mode<Profile,0>(src,n,x,y,cap,mode,timings);
    }
}
}
extern "C" int ng_lifting_encode(const uint8_t*x,const uint8_t*y,int d,int mode,uint8_t*out,size_t cap,size_t*n,uint64_t*stats){
    try{need(x&&y&&out&&n&&d>=32&&d<=256&&d%32==0);auto b=lifting_encode(x,y,d,mode,stats);need(b.size()<=cap);std::memcpy(out,b.data(),b.size());*n=b.size();return 0;}catch(...){return -1;}
}
extern "C" int ng_lifting_order_encode(const uint8_t*x,const uint8_t*y,int d,int mode,int order,uint8_t*out,size_t cap,size_t*n,uint64_t*stats){
    try{need(x&&y&&out&&n&&d>=32&&d<=256&&d%32==0);auto b=lifting_encode(x,y,d,mode,stats,nullptr,order);need(b.size()<=cap);std::memcpy(out,b.data(),b.size());*n=b.size();return 0;}catch(...){return -1;}
}
extern "C" int ng_scalar_semantic_encode(const uint8_t*params,size_t pn,const uint8_t*choices,const uint8_t*symbols,const uint8_t*classes,int d,int C,uint8_t*out,size_t cap,size_t*n){
    try{need(params&&choices&&symbols&&classes&&out&&n&&d>=32&&d<=256&&d%32==0);auto b=scalar_semantic_encode(params,pn,choices,symbols,classes,d,C);need(b.size()<=cap);std::memcpy(out,b.data(),b.size());*n=b.size();return 0;}catch(...){return -1;}
}
extern "C" int ng_scalar_semantic_decode(const uint8_t*src,size_t n,uint8_t*dst,size_t cap){
    try{need(src&&dst);scalar_semantic_decode(src,n,dst,cap);return 0;}catch(...){return -1;}
}
extern "C" int ng_lifting_decode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap){
    try{need(src&&x&&y);lifting_dispatch(src,n,x,y,cap);return 0;}catch(...){return -1;}
}
extern "C" int ng_lifting_contiguous_scalar_decode(const uint8_t*s,size_t n,uint8_t*x,uint8_t*y,size_t cap){
    try{need(s&&x&&y);lifting_decode<false,0,82,false,1>(s,n,x,y,cap);return 0;}catch(...){return -1;}
}
extern "C" int ng_lifting_trial_decode(const uint8_t*s,size_t n,uint8_t*x,uint8_t*y,size_t cap,int trial,uint64_t*t){
    try{need(s&&x&&y);size_t p=4;int d=get(s,n,p);need(get(s,n,p)==78);
        if(t){if(d==64)lifting_trial_dispatch<true,64>(s,n,x,y,cap,trial,t);else lifting_trial_dispatch<true,0>(s,n,x,y,cap,trial,t);}
        else{if(d==64)lifting_trial_dispatch<false,64>(s,n,x,y,cap,trial,nullptr);else lifting_trial_dispatch<false,0>(s,n,x,y,cap,trial,nullptr);}
        return 0;
    }catch(...){return -1;}
}
// Transcode entropy grammar only: retain the fitted model, every block choice,
// residual symbol, traversal and activity context of the original mode78 frame.
extern "C" int ng_lifting_simple_transcode(const uint8_t*s,size_t n,int grammar,uint8_t*out,size_t cap,size_t*size){
    try{
        need(s&&out&&size&&n>=16&&s[3]=='4'&&grammar>=1&&grammar<=4);size_t p=4;int d=get(s,n,p);need(d>=32&&d<=256&&d%32==0&&get(s,n,p)==78&&get(s,n,p)==1);
        JointExtras extra(78);extra.read(s,n,p);
        for(int k=0;k<2;k++){size_t bytes=get(s,n,p);need(bytes<=n-p);p+=bytes;}
        get(s,n,p);size_t bytes=get(s,n,p);need(bytes<=n-p);p+=bytes;
        Bytes x(size_t(d)*d*d),y(x.size());std::vector<Bytes>streams(grammar==1?12:grammar==2?4:16);
        lifting_decode<false,0,78,false,13>(s,n,x.data(),y.data(),x.size(),nullptr,&streams,grammar);
        Bytes frame(s,s+16);frame[3]='5';put(frame,grammar);frame.insert(frame.end(),s+16,s+p);
        for(size_t k=0;k<streams.size();k++){auto&v=streams[k];auto b=grammar==4&&k>=4?pack_bits(v,(k-4)%6+1):pack(v);put(frame,v.size());put(frame,b.size());frame.insert(frame.end(),b.begin(),b.end());}
        need(frame.size()<=cap);std::memcpy(out,frame.data(),frame.size());*size=frame.size();return 0;
    }catch(...){return -1;}
}
extern "C" int ng_joint_simd_audit(){
    try{uint32_t rng=512137;auto random=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
        for(int k=0;k<10000;k++){
            JointBatch q{};for(int i=0;i<32;i++){Lift l(random()&65535);q.a[i]=k%3?l.a:0;q.b[i]=k%3?l.b:0;q.px[i]=random()&255;q.py[i]=random()&255;q.sx[i]=random()&255;q.sy[i]=random()&255;}
            auto ref=q,packed=q,packed_ref=q;joint_batch_reconstruct<true>(packed_ref,32,false);joint_batch_reconstruct<true>(packed,32,true);
            for(int i=0;i<32;i++)need(packed.vx[i]==packed_ref.vx[i]&&packed.vy[i]==packed_ref.vy[i]);
            joint_batch_reconstruct(ref,32,false);joint_batch_reconstruct(q,32,true);
            for(int i=0;i<32;i++)need(q.vx[i]==ref.vx[i]&&q.vy[i]==ref.vy[i]);
        }return joint_has_avx()?1:0;
    }catch(...){return -1;}
}
extern "C" int ng_joint_tag_audit(){
    try{
        std::vector<Bytes>streams(16);
        for(int c=0;c<4;c++)for(int pair=0;pair<256;pair++){
            streams[c].push_back(pair);
            for(int k=0;k<2;k++){unsigned tag=(pair>>(4*k))&15,w=(tag+1)/2;
                if(w>1&&w<8)streams[4+k*6+w-2].push_back((pair+c)&((1u<<(w-1))-1));}
        }
        JointSemantic::validate_tag_streams(streams);JointSemantic sem(4);
        const uint8_t *a[16],*b[16],*end[16];for(int k=0;k<16;k++){a[k]=b[k]=streams[k].data();end[k]=a[k]+streams[k].size();}
        for(int c=0;c<4;c++)for(int pair=0;pair<256;pair++)need(sem.read_tags<false>(c,a,end)==sem.read_tags<true,true>(c,b,end));
        for(int k=0;k<16;k++)need(a[k]==end[k]&&b[k]==end[k]);
        auto rejects=[](const std::vector<Bytes>&v){try{JointSemantic::validate_tag_streams(v);return false;}catch(...){return true;}};
        auto bad=streams;bad[4].pop_back();need(rejects(bad));bad=streams;bad[4][0]=128;need(rejects(bad));
        return 0;
    }catch(...){return -1;}
}
extern "C" int ng_joint_predict_audit(){
    try{
        constexpr int d=32;Bytes x(d*d*d),y(x.size());uint32_t rng=94383;
        auto random=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
        for(size_t i=0;i<x.size();i++){x[i]=random();y[i]=random();}
        for(int pred=1;pred<=15;pred++)for(int k=0;k<100;k++){
            JointPrediction p;JointBatch q;std::array<Pair,32>ref{};
            int lanes=1+k%32;
            for(int lane=0;lane<lanes;lane++){
                // Include first/last faces and each one-/two-voxel boundary.
                int xx=lane<4?lane:random()%d,yy=k<4?k:random()%d,zz=random()%d;
                size_t i=(size_t(zz)*d+yy)*d+xx;unsigned r=lift_reference(x.data(),y.data(),i,d);uint16_t qq[2]={};
                p.set(lane,i,d,pred,r);ref[lane]=palette_prediction(x.data(),y.data(),i,d,pred,qq,16);
                JointPreparedPredictor prepared;prepared.prepare(lane,pred,xx&~3,yy,zz,d,r);
                need(prepared.predict(lane,xx,x.data(),y.data(),i,d)==ref[lane]);
                need(prepared.predict_packed(lane,xx,x.data(),y.data(),i,d)==ref[lane]);
            }
            joint_predict_batch(p,x.data(),y.data(),q,lanes);
            for(int lane=0;lane<lanes;lane++)need(q.px[lane]==ref[lane][0]&&q.py[lane]==ref[lane][1]);
        }
        return 0;
    }catch(...){return -1;}
}
extern "C" int ng_lifting_wave_decode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap){
    try{need(src&&x&&y);size_t pos=4;int d=get(src,n,pos);need(get(src,n,pos)==78);
        switch(d){
        case 32:lifting_decode<false,32,78,true>(src,n,x,y,cap);break;
        case 64:lifting_decode<false,64,78,true>(src,n,x,y,cap);break;
        case 128:lifting_decode<false,128,78,true>(src,n,x,y,cap);break;
        case 256:lifting_decode<false,256,78,true>(src,n,x,y,cap);break;
        default:lifting_decode<false,0,78,true>(src,n,x,y,cap);
        }return 0;
    }catch(...){return -1;}
}
extern "C" int ng_lifting_profile_decode(const uint8_t*src,size_t n,uint8_t*x,uint8_t*y,size_t cap,uint64_t*timings){
    try{need(src&&x&&y&&timings);lifting_dispatch<true>(src,n,x,y,cap,timings);return 0;}catch(...){return -1;}
}
extern "C" int ng_lifting_binary_audit(){
    try{
        uint32_t state=41721;
        auto random=[&](){state=state*1664525u+1013904223u;return state;};
        for(unsigned total=2;total<=32768;total++)for(unsigned k=0;k<8;k++){
            uint64_t range=k==0?(uint64_t(1)<<32):uint64_t(random())+1;
            uint32_t zero=1+random()%(total-1);
            need(ng_cabac::binary_quotient(range*zero,total)==range*zero/total);
        }
        ng_cabac::AuEncoder enc;std::array<ng_cabac::Binary,64> models;
        std::vector<unsigned> bits(200000),contexts(bits.size());
        for(size_t i=0;i<bits.size();i++){
            unsigned c=contexts[i]=random()%64;
            unsigned b=bits[i]=(random()>>16)%256<(i%3==0?1:i%3==1?128:255);
            models[c].code(&enc,nullptr,b);
        }
        auto bytes=enc.finish();ng_cabac::AuDecoder generic(bytes.bytes,bytes.logical),fast(bytes.bytes,bytes.logical);
        std::array<ng_cabac::Binary,64> ref{},opt{};
        for(size_t i=0;i<bits.size();i++){
            auto& m=ref[contexts[i]];unsigned total=m.c[0]+m.c[1],bit=generic.target(total)>=m.c[0];
            generic.accept(bit?m.c[0]:0,m.c[bit],total);m.update(bit);
            need(bit==bits[i]&&opt[contexts[i]].code(nullptr,&fast,0)==bit);
            need(generic.physicalReads()==fast.physicalReads());
        }
        return 0;
    }catch(...){return -1;}
}
extern "C" int ng_joint_observe(const uint8_t*x,const uint8_t*y,int d,uint8_t*symbols,uint8_t*classes){
    try{need(x&&y&&symbols&&classes&&d>=32&&d<=256&&d%32==0);JointObservation o{symbols,classes};lifting_encode(x,y,d,61,nullptr,&o);return 0;}catch(...){return -1;}
}
extern "C" int ng_lifting_audit(uint64_t*stats){
    try{need(stats);stats[0]=stats[1]=stats[2]=0;
        for(int k=0;k<64;k++){
            int x=k<32?-127+k*8:127,y=k<32?127:127-(k-32)*8;Lift l(unsigned(x+128)*256+unsigned(y+128));
            for(unsigned id=0;id<65536;id++){
                Pair raw{int(id>>8),int(id&255)},center{raw[0]-128,raw[1]-128};auto p=l.forward(center);
                need(l.inverse(p)==center);stats[0]++;stats[2]=std::max(stats[2],uint64_t(std::max(std::abs(p[0]),std::abs(p[1]))));
                auto m=l.modular(center);need(l.modular(m,true)==center);
                Pair mp{int((id*73+21)&255),int((id*151+197)&255)};
                auto ms=lifting_symbols(raw,mp,4,l);need(ms[0]<256&&ms[1]<256);
                need(lifting_values(ms,mp,4,l)==raw);
                need(std::abs(p[0])<256&&std::abs(p[1])<256);
                for(int kind=0;kind<4;kind++){Pair pred{202,51};auto s=lifting_symbols(raw,pred,kind,l);need(lifting_values(s,pred,kind,l)==raw);stats[1]++;}
                auto neg=l.forward({-center[0],-center[1]});need(neg[0]==-p[0]&&neg[1]==-p[1]);
            }
        }return 0;
    }catch(...){return -1;}
}
