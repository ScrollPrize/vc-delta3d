#include "cabac.hpp"
#include "adaptive_arithmetic.hpp"
#include "residual_bridge.h"
#include "volcodec_ng_lib.h"
#include <array>
#include <algorithm>
#include <bit>
#include <cstring>
#include <future>
#include <cstdio>

namespace {
using namespace ng_cabac;
constexpr size_t N=128*128*128, H=24;
void put(std::vector<uint8_t>& v,size_t p,uint64_t x,unsigned n) {
    for(unsigned k=0;k<n;++k) v[p+k]=uint8_t(x>>(8*k));
}
uint64_t get(std::span<const uint8_t> v,size_t p,unsigned n) {
    if(p+n>v.size()) throw std::runtime_error("CABAC truncated header");
    uint64_t x=0; for(unsigned k=0;k<n;++k) x|=uint64_t(v[p+k])<<(8*k); return x;
}
uint64_t hash(const uint8_t* src) {
    uint64_t h=1469598103934665603ull;
    for(size_t i=0;i<N;++i) { h^=src[i]; h*=1099511628211ull; } return h;
}
unsigned bucket(unsigned v) { return v==0?0:v==1?1:v<=3?2:3; }
struct Neighbors { unsigned spatial=0,agreement=0; };
Neighbors neighbors(const std::vector<uint8_t>& s,size_t p) {
    unsigned mag=0,positive=0,negative=0;
    const auto use=[&](size_t q) {
        const unsigned v=s[q]; mag=std::max(mag,(v+1)/2);
        if(v) { negative|=v&1; positive|=!(v&1); }
    };
    if(p%128) use(p-1);
    if(p/128%128) use(p-128);
    if(p/16384) use(p-16384);
    return {bucket(mag),positive+2*negative};
}
struct Grammar {
    unsigned variant,predictor,C;
    std::vector<Binary> models;
    std::vector<uint8_t> symbols;
    Grammar(unsigned v,unsigned p,unsigned classes=16):variant(v),predictor(p),C(v==0?1:(p==0?classes:4)*(v==3?4:1)),
        models(C*(15+(v>=2?16:1))),symbols(N,0) {
        for(unsigned i=0;i<C*8;++i) models[i].c[0]=3;
    }
    unsigned symbol(size_t pos,unsigned native,AuEncoder* enc,AuDecoder* dec,unsigned input=0) {
        const auto nb=neighbors(symbols,pos);
        unsigned c=variant==0?0:predictor==0?native:nb.spatial;
        if(variant==3) c=c*4+nb.spatial;
        if(c>=C) throw std::runtime_error("CABAC invalid context");
        auto bit=[&](unsigned model,unsigned b) { return models.at(model).code(enc,dec,b); };
        if(!bit(c*8,input!=0)) return symbols[pos]=0;
        unsigned mag=(input+1)/2,width=enc?std::bit_width(mag)-1:0;
        unsigned w=0;
        for(;w<7;++w) if(!bit(c*8+1+w,width>w)) break;
        if(w==7) return symbols[pos]=255; // only -128 is representable
        unsigned value=1u<<w;
        for(unsigned b=w;b>0;--b) value|=bit(C*8+c*7+b-1,(mag>>(b-1))&1)<<(b-1);
        unsigned signctx=c;
        if(variant>=2) {
            const unsigned mb=value==1?0:value<=3?1:value<=15?2:3;
            signctx=c*16+nb.agreement*4+mb;
        }
        const unsigned sign=bit(C*15+signctx,input&1);
        return symbols[pos]=uint8_t(2*value-sign);
    }
};
struct Prepared {
    std::vector<uint8_t> params,symbols,classes;
    unsigned predictor,mask=0;
};
Prepared prepare(const uint8_t* src,unsigned pred,const int* config=nullptr) {
    Prepared p; p.predictor=pred; p.symbols.resize(N);p.classes.resize(N);
    if(pred==0) {
        p.params.resize(4096);size_t n=0;
        const int status=config ? ng_prepare_config(src,config,p.params.data(),p.params.size(),&n,p.symbols.data(),p.classes.data())
            : ng_prepare_residuals(src,p.params.data(),p.params.size(),&n,p.symbols.data(),p.classes.data());
        if(status)
            throw std::runtime_error("NG residual handoff failed");
        p.params.resize(n);
    } else {
        p.mask=delta_prepare_residuals(src,p.symbols.data());
        for(auto& x:p.symbols) { const int r=x<128?int(x):int(x)-256; x=uint8_t(r<0?-2*r-1:2*r); }
    }
    return p;
}
std::vector<uint8_t> encode(const Prepared& p,unsigned variant,uint64_t checksum) {
    Grammar g(variant,p.predictor,p.predictor==0?p.params.at(4):4); AuEncoder enc;
    for(size_t i=0;i<N;++i) {
        const auto symbol=g.symbol(i,p.classes[i],&enc,nullptr,p.symbols[i]);
        if(symbol!=p.symbols[i]) throw std::runtime_error("CABAC semantic mapping mismatch");
    }
    auto body=enc.finish();
    std::vector<uint8_t> out(H+p.params.size()+body.bytes.size(),0);
    std::memcpy(out.data(),"NCAB",4);out[4]=(p.predictor==0 && p.params.at(4)!=16)?2:1;
    out[5]=p.predictor;out[6]=variant;out[7]=p.mask;
    put(out,8,p.params.size(),4);put(out,12,body.logical,4);put(out,16,checksum,8);
    std::copy(p.params.begin(),p.params.end(),out.begin()+H);
    std::copy(body.bytes.begin(),body.bytes.end(),out.begin()+H+p.params.size());return out;
}
struct Reader { Grammar& g; AuDecoder& d; };
int readSymbol(void* v,size_t pos,unsigned cls) {
    try { auto& r=*static_cast<Reader*>(v); return r.g.symbol(pos,cls,nullptr,&r.d); }
    catch(...) { return -1; } // never unwind through the C predictor
}
} // namespace

bool ng_cabac_decode(std::span<const uint8_t> bytes,uint8_t* dst) {
    try {
        if(bytes.size()<H || std::memcmp(bytes.data(),"NCAB",4) || (bytes[4]!=1 && bytes[4]!=2) || bytes[5]>1 || bytes[6]>3 || bytes[7]>7) return false;
        const auto params=get(bytes,8,4),bits=get(bytes,12,4);
        if(params>4096 || H+params+(bits+7)/8!=bytes.size()) return false;
        if(bytes[5]==0 && (params<6 || bytes[H+4]<2 || bytes[H+4]>32)) return false;
        Grammar g(bytes[6],bytes[5],bytes[5]==0?bytes[H+4]:4);AuDecoder dec(bytes.subspan(H+params),uint32_t(bits));
        if(bytes[5]==0) {
            Reader reader{g,dec};
            if(ng_reconstruct_residuals(bytes.data()+H,params,readSymbol,&reader,dst)) return false;
        } else {
            if(params) return false;
            for(size_t i=0;i<N;++i) {
                const unsigned s=g.symbol(i,0,nullptr,&dec);
                dst[i]=uint8_t(int(s>>1)^-int(s&1));
            }
            delta_reconstruct_residuals(dst,bytes[7]);
        }
        return hash(dst)==get(bytes,16,8);
    } catch(...) { return false; }
}

std::vector<std::vector<uint8_t>> ng_cabac_candidates(const uint8_t* src,bool verify) {
    const auto ng=prepare(src,0),delta=prepare(src,1);const uint64_t checksum=hash(src);
    std::vector<std::future<std::vector<uint8_t>>> jobs;
    for(unsigned id=0;id<8;++id) jobs.push_back(std::async(std::launch::async,[&,id] {
        auto out=encode(id<4?ng:delta,id%4,checksum);
        if(verify) {
            std::vector<uint8_t> decoded(N);
            if(!ng_cabac_decode(out,decoded.data()) || std::memcmp(src,decoded.data(),N))
                throw std::runtime_error("CABAC roundtrip failed candidate "+std::to_string(id));
        }
        return out;
    }));
    std::vector<std::vector<uint8_t>> out;
    for(auto& j:jobs) out.push_back(j.get());return out;
}
extern "C" int codec_cabac_matrix(const uint8_t* src,uint64_t* sizes) {
    if(!src || !sizes) return -1;
    try {
        auto all=ng_cabac_candidates(src,true);
        for(size_t i=0;i<all.size();++i) {
            sizes[i*3]=all[i].size(); sizes[i*3+1]=H+get(all[i],8,4);
            sizes[i*3+2]=all[i].size()-sizes[i*3+1];
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"CABAC matrix: %s\n",e.what());return -3; }
}

// Five complete streams: original rANS + CABAC contexts 0..3. Config is
// written via the existing predictor metadata; the selected tap subset is
// already represented by zero/nonzero fitted weights.
extern "C" int codec_ng_tuning_trial(const uint8_t* src,const int* config,uint64_t* sizes) {
    if(!src || !config || !sizes)return -1;
    try {
        const auto p=prepare(src,0,config);const auto checksum=hash(src);
        std::vector<uint8_t> decoded(N),rans(N*2+65536);size_t n=0;
        if(ng_encode_config(src,config,rans.data(),rans.size(),&n) ||
           volcodec_ng_lib_decode(rans.data(),n,decoded.data(),N) || std::memcmp(src,decoded.data(),N))
            throw std::runtime_error("tuned NG rANS roundtrip failed");
        sizes[0]=n;
        for(unsigned v=0;v<4;++v) {
            const auto payload=encode(p,v,checksum);
            if(!ng_cabac_decode(payload,decoded.data()) || std::memcmp(src,decoded.data(),N))
                throw std::runtime_error("tuned NG CABAC roundtrip failed");
            sizes[v+1]=payload.size();
        }
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"NG tuning: %s\n",e.what());return -3;}
}
