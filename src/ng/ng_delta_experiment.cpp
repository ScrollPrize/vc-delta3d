// Experimental rANS frames for NG/Delta composition. No production dispatch.
#include "residual_bridge.h"
#include "rans_frame_bridge.hpp"
#include "volcodec_ng_lib.h"
#include "ng_leaf.h"
#include "ng_packed_bank.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <memory>
#include <map>
#include <numeric>
#include "ng_score_simd.hpp"

namespace {
#define N NG_N
using Bytes=std::vector<uint8_t>;
struct SoftClassScope {
    int saved=ng_soft_classes;
    explicit SoftClassScope(int mode){ng_soft_classes=mode;}
    ~SoftClassScope(){ng_soft_classes=saved;}
};
Bytes encode_joint(const uint8_t*,const int*,int);
Bytes decode_joint(const Bytes&,int);
Bytes encode_fast(const uint8_t*,const int*,int);
Bytes decode_fast(const Bytes&,int,double* =nullptr);
thread_local double* encode_profile=nullptr;
// Inclusive entropy/payload costs and work counts, separate from the bank's
// nested phase timers. Enabled only for diagnostic calls, not timing runs.
thread_local bool encoder_work_enabled=false;
thread_local double encoder_work[14]={};
thread_local bool encoder_bound_enabled=true;
struct EncoderWorkTimer {
    unsigned slot; std::chrono::steady_clock::time_point start;
    explicit EncoderWorkTimer(unsigned s):slot(s){if(encoder_work_enabled)start=std::chrono::steady_clock::now();}
    ~EncoderWorkTimer(){if(encoder_work_enabled)encoder_work[slot]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
};
thread_local double* decode_profile=nullptr;
thread_local std::chrono::steady_clock::time_point decode_tick;
void profile_tick(int phase) {
    auto now=std::chrono::steady_clock::now();
    if(phase>=0 && phase<5)decode_profile[phase]+=std::chrono::duration<double>(now-decode_tick).count();
    decode_tick=now;
}
void require_at(bool ok,const char*file,int line) {if(!ok)throw std::runtime_error(std::string("NG/Delta experiment validation failed: ")+file+":"+std::to_string(line));}
#define require(ok) require_at((ok),__FILE__,__LINE__)
void put32(Bytes& b,uint32_t n) {for(int k=0;k<4;k++)b.push_back(uint8_t(n>>(8*k)));}
uint32_t get32(const Bytes& b,size_t& pos) {
    require(pos+4<=b.size());uint32_t n=0;
    for(int k=0;k<4;k++)n|=uint32_t(b[pos++])<<(8*k);return n;
}
uint8_t pyramid(uint8_t r) {return uint8_t((unsigned(r)<<1)^(0u-(unsigned(r)>>7)));}
uint8_t unpyramid(uint8_t s) {return uint8_t((unsigned(s)>>1)^(0u-(unsigned(s)&1u)));}
void bank_fixed_candidates(const uint8_t*,std::vector<Bytes>&,unsigned,unsigned mask=4095);
Bytes pack_span(const uint8_t* data,size_t size) {
    if(!size)return {};
    EncoderWorkTimer timer(2);
    if(encoder_work_enabled){encoder_work[0]++;encoder_work[1]+=size;}
    auto frame=delta_rans_frame(data,size);
    const auto* bytes=reinterpret_cast<const uint8_t*>(frame.data());
    return Bytes(bytes,bytes+frame.size());
}
Bytes pack(const Bytes& b) {return pack_span(b.data(),b.size());}
Bytes pack_stream_span(const uint8_t* data,size_t size,bool compact) {
    if(!size||!compact)return pack_span(data,size);
    EncoderWorkTimer timer(2);
    if(encoder_work_enabled){encoder_work[0]++;encoder_work[1]+=size;}
    auto frame=delta_rans_frame(data,size);
    const auto* raw=reinterpret_cast<const uint8_t*>(frame.data());
    require(frame.size()>=512);uint16_t freq[256];uint8_t table[514];
    for(unsigned i=0;i<256;i++)freq[i]=uint16_t(raw[2*i]|unsigned(raw[2*i+1])<<8);
    const size_t tn=ng_write_freq_table(table,freq);
    Bytes out(tn+frame.size()-512);
    std::memcpy(out.data(),table,tn);std::memcpy(out.data()+tn,raw+512,frame.size()-512);
    return out;
}
Bytes pack_stream(const Bytes& values,bool compact) {return pack_stream_span(values.data(),values.size(),compact);}
void unpack_stream(const uint8_t* data,size_t bytes,Bytes& values,bool compact) {
    if(!compact){require(delta_rans_decode(data,bytes,values.data(),values.size())==0);return;}
    uint16_t freq[256];size_t used=0;require(ng_read_freq_table(data,bytes,freq,&used)==0 && used<=bytes);
    Bytes raw(512);for(unsigned i=0;i<256;i++){raw[2*i]=uint8_t(freq[i]);raw[2*i+1]=uint8_t(freq[i]>>8);}
    raw.insert(raw.end(),data+used,data+bytes);require(delta_rans_decode(raw.data(),raw.size(),values.data(),values.size())==0);
}
struct Streams {
    std::vector<Bytes> symbols;
    std::vector<size_t> pos;
    unsigned classes=0;
    static unsigned block(size_t i) {
        const size_t d=NG_BLOCK_DIM;
        return unsigned((i%d)/(d/2)+2*((i/d%d)/(d/2))+4*((i/d/d)/(d/2)));
    }
    static int read(void* ptr,size_t i,unsigned c) {
        auto& s=*static_cast<Streams*>(ptr);
        if(s.classes)c+=s.classes*block(i);
        if(c>=s.symbols.size() || s.pos[c]>=s.symbols[c].size())return -1;
        return s.symbols[c][s.pos[c]++];
    }
};
struct Raw {
    const Bytes* data;
    static int read(void* ptr,size_t i,unsigned) {return (*static_cast<Raw*>(ptr)->data)[i];}
};
struct DirectionStreams {
    std::vector<Bytes> symbols;
    std::vector<size_t> pos;
    const Bytes* choices=nullptr;
    unsigned K=0,shift=0,family=0,dimension=NG_BLOCK_DIM;bool separate=false;
    unsigned group(size_t i,unsigned c) const {
        if(!separate && !family)return c;
        unsigned nb=dimension>>shift,x=i%dimension,y=i/dimension%dimension,z=i/(dimension*dimension);
        unsigned choice=(*choices)[((z>>shift)*nb+(y>>shift))*nb+(x>>shift)];
        return c+K*(family?(choice==0?0:choice<8?1:2):choice);
    }
    static int read(void* ptr,size_t i,unsigned c) {
        auto& s=*static_cast<DirectionStreams*>(ptr);c=s.group(i,c);
        if(c>=s.symbols.size() || s.pos[c]>=s.symbols[c].size())return -1;
        return s.symbols[c][s.pos[c]++];
    }
};
void remove_class_bias(std::vector<uint16_t>& prediction,const Bytes& params,const Bytes& classes,unsigned K) {
    const size_t count=NG_N;
    for(size_t i=0;i<count;i++) {
        size_t off=params.size()-2*K+2*classes[i];
        int16_t bias=int16_t(uint16_t(params[off]|unsigned(params[off+1])<<8));
        prediction[i]=uint16_t(prediction[i]-bias);
    }
}
using Histogram=std::array<uint32_t,256>;
// Four independent counters break dependency chains on common residuals.
// Keep only one candidate hot; shared by class and spatial-block selection.
void candidate_histogram(const uint8_t* symbols,const uint8_t* classes,unsigned K,
                         std::vector<Histogram>& split,Histogram* out,size_t stride=1) {
    const size_t count=NG_N;
    std::fill(split.begin(),split.end(),Histogram{});
    for(size_t i=0;i<count;i+=4) {
        split[classes[i]][symbols[i]]++;
        split[K+classes[i+1]][symbols[i+1]]++;
        split[2*K+classes[i+2]][symbols[i+2]]++;
        split[3*K+classes[i+3]][symbols[i+3]]++;
    }
    for(unsigned c=0;c<K;c++)for(unsigned s=0;s<256;s++)
        out[c*stride][s]=split[c][s]+split[K+c][s]+split[2*K+c][s]+split[3*K+c][s];
}
double histogram_cost(const uint32_t* counts,int tb,bool fast) {
    if(!fast)return ng_hist_cost(counts,tb);
    static const auto logs=[] {std::array<double,4097> a{};for(size_t i=1;i<a.size();i++)a[i]=std::log2(double(i));return a;}();
    return ng_hist_cost_logs(counts,tb,logs.data());
}
uint8_t candidate_symbol(const uint8_t* src,size_t i,unsigned choice,uint8_t ng) {
    return choice?pyramid(uint8_t(src[i]-delta_predict_at(src,i,choice))):ng;
}
Bytes histogram_choices(const uint8_t* src,const Bytes& sy,const Bytes& cl,unsigned K,int tb,const std::vector<Bytes>* cache=nullptr) {
    std::vector<Histogram> hist(K*8);
    if(cache) {
        // Keep only one candidate's independent counters hot at a time.
        // All eight at once exceed L1/L2 working sets and regress throughput.
        std::vector<Histogram> split(4*K);
        for(unsigned p=0;p<8;p++) {
            const uint8_t* symbols=p?(*cache)[p].data():sy.data();
            candidate_histogram(symbols,cl.data(),K,split,hist.data()+p,8);
        }
    } else for(size_t i=0;i<N;i++)for(unsigned choice=0;choice<8;choice++)
        hist[cl[i]*8+choice][candidate_symbol(src,i,choice,sy[i])]++;
    Bytes selectors(K);
    for(unsigned c=0;c<K;c++) {
        double best=std::numeric_limits<double>::infinity();
        for(unsigned p=0;p<8;p++) {
            double cost=histogram_cost(hist[c*8+p].data(),tb,cache!=nullptr);
            if(cost<best){best=cost;selectors[c]=uint8_t(p);}
        }
    }
    return selectors;
}
std::vector<uint16_t> activity_quantiles_histogram(const std::array<uint32_t,2560>&hist,unsigned K) {
    std::vector<uint16_t> thresholds;uint64_t sum=0,total=0;for(auto n:hist)total+=n;
    require(total>0);
    for(unsigned v=0;v<hist.size() && thresholds.size()+1<K;v++) {
        sum+=hist[v];while(thresholds.size()+1<K && sum*K>=(thresholds.size()+1)*total)thresholds.push_back(uint16_t(v));
    }
    return thresholds;
}
std::vector<uint16_t> activity_quantiles(const std::vector<uint16_t>& a,unsigned K) {
    std::array<uint32_t,2560> hist{};for(auto v:a){require(v<hist.size());hist[v]++;}
    return activity_quantiles_histogram(hist,K);
}
void refit_biases(const uint8_t* src,Bytes& params,const std::vector<uint16_t>& activities,
                  const std::vector<uint16_t>& base,const std::vector<uint16_t>& thresholds,Bytes& sy,Bytes& cl,bool fast=false) {
    unsigned K=unsigned(thresholds.size()+1);
    std::vector<int64_t> sums(K);std::vector<uint32_t> counts(K);std::vector<int16_t> bias(K);
    std::array<uint8_t,2560> lut{};
    if(fast)for(unsigned a=0;a<lut.size();a++)lut[a]=uint8_t(std::lower_bound(thresholds.begin(),thresholds.end(),a)-thresholds.begin());
    for(size_t i=0;i<N;i++) {
        unsigned c=fast?lut[activities[i]]:unsigned(std::lower_bound(thresholds.begin(),thresholds.end(),activities[i])-thresholds.begin());
        cl[i]=uint8_t(c);sums[c]+=int16_t(uint16_t(unsigned(src[i])*256u-base[i]));counts[c]++;
    }
    for(unsigned c=0;c<K;c++) {
        double b=counts[c]?std::floor(double(sums[c])/counts[c]+0.5):0;
        bias[c]=int16_t(std::max(-32768.0,std::min(32767.0,b)));
    }
    require(ng_set_classes(params.data(),params.size(),thresholds.data(),bias.data())==0);
    for(size_t i=0;i<N;i++) {
        uint16_t q=uint16_t(base[i]+bias[cl[i]]);
        uint8_t r=uint8_t(src[i]-uint8_t(uint16_t(q+128u)>>8));
        if((q>>7)&1)r=uint8_t(-r);
        sy[i]=pyramid(r);
    }
}
// Shared interval-partition DP. Optional empty classes retain duplicate
// thresholds, so the existing greedy partition remains representable.
template<class Cost> std::vector<uint16_t> partition_boundaries(
        const std::vector<uint16_t>&edges,unsigned K,Cost cost,bool empty=false){
    const unsigned B=edges.size(),used=empty?K:std::min(K,B);
    std::vector<double> dp((used+1)*(B+1),std::numeric_limits<double>::infinity());
    std::vector<unsigned> prev(dp.size());dp[0]=0;
    for(unsigned k=1;k<=used;k++)for(unsigned b=empty?1:k;b<=B;b++)
        for(unsigned a=empty?(k==1?0:1):k-1;a<(empty?b+1:b);a++){
            double before=dp[(k-1)*(B+1)+a];if(!std::isfinite(before))continue;
            double score=before+cost(a,b);
            if(score<dp[k*(B+1)+b]){dp[k*(B+1)+b]=score;prev[k*(B+1)+b]=a;}
        }
    std::vector<uint16_t> thresholds(K-1,2559);unsigned b=B;
    for(unsigned k=used;k>1;k--){b=prev[k*(B+1)+b];require(b>0);thresholds[k-2]=edges[b-1];}
    return thresholds;
}
// Bounded histogram DP: at most 64 quantile bins plus existing boundaries.
// Cost each interval by its best predictor and normalized native table cost.
std::vector<uint16_t> optimized_boundaries(const uint8_t* src,const Bytes& sy,
        const std::vector<uint16_t>& activities,const std::vector<uint16_t>& old,int tb,const std::vector<Bytes>* cache=nullptr) {
    auto edges=activity_quantiles(activities,64);edges.insert(edges.end(),old.begin(),old.end());edges.push_back(2559);
    std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    unsigned B=unsigned(edges.size()),K=unsigned(old.size()+1);
    std::vector<Histogram> prefix((B+1)*8);
    std::array<uint8_t,2560> bins{};
    if(cache)for(unsigned a=0;a<bins.size();a++)bins[a]=uint8_t(std::lower_bound(edges.begin(),edges.end(),a)-edges.begin());
    for(size_t i=0;i<N;i++) {
        size_t b=cache?bins[activities[i]]:std::lower_bound(edges.begin(),edges.end(),activities[i])-edges.begin();
        for(unsigned p=0;p<8;p++)prefix[(b+1)*8+p][cache && p?(*cache)[p][i]:candidate_symbol(src,i,p,sy[i])]++;
    }
    for(unsigned b=1;b<=B;b++)for(unsigned p=0;p<8;p++)for(unsigned s=0;s<256;s++)
        prefix[b*8+p][s]+=prefix[(b-1)*8+p][s];
    std::vector<double> cost((B+1)*(B+1),std::numeric_limits<double>::infinity());
    for(unsigned a=0;a<=B;a++)for(unsigned b=a;b<=B;b++)for(unsigned p=0;p<8;p++) {
        Histogram hist{};for(unsigned s=0;s<256;s++)hist[s]=prefix[b*8+p][s]-prefix[a*8+p][s];
        double& c=cost[a*(B+1)+b];c=std::min(c,histogram_cost(hist.data(),tb,cache!=nullptr));
    }
    return partition_boundaries(edges,K,[&](unsigned a,unsigned b){return cost[a*(B+1)+b];});
}
Bytes choose_blocks(const uint8_t* const* symbols,unsigned P,const uint8_t* cl,unsigned K,unsigned B,double* output_scores=nullptr) {
    std::vector<Histogram> hist(P*K),split(4*K);
    std::vector<unsigned> owner(P);
    for(unsigned p=0;p<P;p++){
        owner[p]=p;for(unsigned q=0;q<p;q++)if(symbols[p]==symbols[q]){owner[p]=q;break;}
        if(owner[p]==p)candidate_histogram(symbols[p],cl,K,split,hist.data()+p*K);
    }
    std::vector<std::array<double,256>> costs(P*K);
    for(unsigned c=0;c<P*K;c++)if(owner[c/K]==c/K){double total=256;for(auto n:hist[c])total+=n;for(unsigned s=0;s<256;s++)costs[c][s]=std::log2(total)-std::log2(double(hist[c][s])+1);}
    const unsigned nx=NG_X,ny_dim=NG_Y,nz_dim=NG_Z,nb=nx/B,ny=ny_dim/B,nz=nz_dim/B,nblocks=nb*ny*nz;
    const unsigned shift=B==4?2:B==8?3:4;
    std::vector<double> scores(size_t(nblocks)*P);
    for(unsigned p=0;p<P;p++){
        if(owner[p]!=p){for(unsigned b=0;b<nblocks;b++)scores[size_t(b)*P+p]=scores[size_t(b)*P+owner[p]];continue;}
        auto* cost=costs.data()+p*K;
        for(unsigned z=0;z<nz_dim;z++)for(unsigned y=0;y<ny_dim;y++){
            size_t row=(size_t(z)*ny_dim+y)*nx,blockrow=((z>>shift)*ny+(y>>shift))*nb;
            for(unsigned bx=0;bx<nb;bx++){double score=scores[(blockrow+bx)*P+p];size_t i=row+(bx<<shift);
                for(unsigned x=0;x<B;x++)score+=cost[cl[i+x]][symbols[p][i+x]];
                scores[(blockrow+bx)*P+p]=score;}
        }
    }
    if(output_scores)std::copy(scores.begin(),scores.end(),output_scores);
    Bytes choices(nblocks);for(unsigned b=0;b<nblocks;b++)choices[b]=uint8_t(std::min_element(scores.begin()+b*P,scores.begin()+(b+1)*P)-(scores.begin()+b*P));return choices;
}
struct BlockHandoff {Bytes params,choices,symbols,classes;};
Bytes encode(const uint8_t* src,const int* cfg,int mode,unsigned mask,bool unmapped,bool local_entropy=false,BlockHandoff* handoff=nullptr) {
    const size_t count=NG_N; const unsigned nx=NG_X,ny_dim=NG_Y,nz_dim=NG_Z;
    if(mode>=1000 && mode<=1005)return encode_fast(src,cfg,mode);
    if(mode>=900 && mode<=912)return encode_joint(src,cfg,mode);
    const bool compact_tables=mode/100>=5;
    const int block_variant=compact_tables?mode/100-4:mode/100;
    const bool new_block=block_variant>=1 && block_variant<=4 && (mode%100==4 || mode%100==8 || mode%100==16);
    const int block_size=new_block?mode%100:(mode==34 || mode==38 || mode==46)?mode-30:0;
    const bool optimized=block_size || (mode>=22 && mode<=24);
    if(block_size)mode=12;else if(optimized)mode-=10;
    auto mark_start=std::chrono::steady_clock::now();
    auto mark=[&](int phase){if(encode_profile){auto now=std::chrono::steady_clock::now();encode_profile[phase]+=std::chrono::duration<double>(now-mark_start).count();mark_start=now;}};
    require(mode>=0 && mode<=15 && mask<=7);
    require(mode<11 || (!unmapped && !local_entropy));
    require(!local_entropy || mode==0 || mode==3 || mode==4);
    Bytes input(src,src+count),params(65536),sy(count),cl(count);
    std::vector<uint16_t> predictions(count);
    if(mode==5)delta_apply_mask(input.data(),mask);
    size_t pn=0;
    int target=(mode>=1 && mode<=3)?mode:0;
    {SoftClassScope hard(0);
      require(ng_ex_prepare(input.data(),cfg,target,mask,params.data(),params.size(),&pn,
                           sy.data(),cl.data(),predictions.data())==0);}
    params.resize(pn);
    unsigned K=unsigned(cfg[3]);
    if(ng_soft_classes){
        require(block_size!=0);remove_class_bias(predictions,params,cl,K);
        Bytes ng_only(K,0);
        require(ng_selected_symbols_fast(params.data(),params.size(),ng_only.data(),input.data(),predictions.data(),nullptr,sy.data(),cl.data(),nullptr)==0);
    }
    std::vector<uint16_t> base;
    if(mode==13 || mode==14 || optimized) {
        if(optimized)base.swap(predictions);else base=predictions;
        if(!ng_soft_classes)remove_class_bias(base,params,cl,K);
    }
    mark(0);
    // Worker-local, bounded by the seven candidate volumes. No cross-thread
    // sharing; storage is released when the worker thread exits.
    static thread_local std::vector<Bytes> worker_delta_cache;
    auto& delta_cache=worker_delta_cache;const uint8_t* delta_ptr[8]={0};
    if(optimized) {
        delta_cache.resize(8);
        for(unsigned m=1;m<8;m++) {
            unsigned axis=m&(~m+1u),parent=m^axis;
            delta_cache[m]=parent?delta_cache[parent]:input;delta_apply_mask(delta_cache[m].data(),axis);
        }
        for(unsigned m=1;m<8;m++) {
            for(auto& v:delta_cache[m])v=pyramid(v);
            delta_ptr[m]=delta_cache[m].data();
        }
    }
    mark(1);
    if(block_size) {
        // Estimate each candidate using chunk-wide, initial-NG-class histograms.
        // No tiny-block histogram fit and no iterative class/choice refinement.
        const uint8_t* candidates[8];candidates[0]=sy.data();
        for(unsigned p=1;p<8;p++)candidates[p]=delta_ptr[p];
        Bytes choices=choose_blocks(candidates,8,cl.data(),K,block_size);
        const unsigned nblocks=choices.size();
        Bytes packed((nblocks*3+7)/8,0);
        for(unsigned b=0;b<nblocks;b++)for(unsigned j=0;j<3;j++)packed[(3*b+j)/8]|=((choices[b]>>j)&1)<<((3*b+j)%8);
        mark(2);
        require(ng_block_symbols(params.data(),params.size(),choices.data(),src,base.data(),delta_ptr,sy.data(),cl.data(),block_size)==0);
        if(handoff){handoff->params=params;handoff->choices=choices;handoff->symbols=sy;handoff->classes=cl;}
        mark(4);
        if(new_block) {
            bool selector_rans=block_variant!=2,streams=block_variant!=1,separate=block_variant==2 || block_variant==3;
            Bytes selector=selector_rans?pack_stream(choices,compact_tables):packed;
            Bytes frame={'N','D','B','2',uint8_t(block_size),uint8_t(selector_rans|unsigned(streams)<<1|unsigned(separate)<<2|unsigned(compact_tables)<<3)};
            put32(frame,uint32_t(selector.size()));frame.insert(frame.end(),selector.begin(),selector.end());
            if(!streams) {
                Bytes payload(count*2+65536);size_t size=0;
                require(ng_wave_encode(params.data(),params.size(),sy.data(),cl.data(),payload.data(),payload.size(),&size)==0);
                frame.insert(frame.end(),payload.begin(),payload.begin()+size);
            } else {
                put32(frame,uint32_t(params.size()));frame.insert(frame.end(),params.begin(),params.end());
                DirectionStreams s;s.K=K;s.shift=block_size==4?2:block_size==8?3:4;s.separate=separate;s.choices=&choices;
                s.symbols.resize(K*(separate?8:1));
                std::vector<uint32_t> rows(NG_BLOCK_DIM*NG_BLOCK_DIM);
                require(ng_wave_rows(params.data(),params.size(),rows.data())==0);
                for(auto row:rows)for(unsigned x=0;x<NG_BLOCK_DIM;x++) {
                    size_t i=size_t(row)*NG_BLOCK_DIM+x;s.symbols[s.group(i,cl[i])].push_back(sy[i]);
                }
                for(const auto& values:s.symbols) {
                    auto encoded=pack_stream(values,compact_tables);put32(frame,uint32_t(values.size()));put32(frame,uint32_t(encoded.size()));
                    frame.insert(frame.end(),encoded.begin(),encoded.end());
                }
            }
            mark(5);return frame;
        }
        Bytes payload(count*2+65536);size_t size=0;
        require(ng_wave_encode(params.data(),params.size(),sy.data(),cl.data(),payload.data(),payload.size(),&size)==0);
        Bytes frame={'N','D','B','1',uint8_t(block_size)};
        frame.insert(frame.end(),packed.begin(),packed.end());frame.insert(frame.end(),payload.begin(),payload.begin()+size);
        return frame;
    }
    if(mode==7)unmapped=true;
    Bytes raw(optimized?0:count);
    if(!optimized)for(size_t i=0;i<count;i++)raw[i]=uint8_t(input[i]-uint8_t(uint16_t(predictions[i]+128u)>>8));
    std::vector<uint16_t> activity_thr(K-1);
    if(mode==6 || mode==9) {
        delta_apply_mask(raw.data(),mask);
        if(mode==6){K=1;std::fill(cl.begin(),cl.end(),0);}
        else require(ng_activity_classes(raw.data(),int(K),activity_thr.data(),cl.data())==0);
        for(size_t i=0;i<count;i++)sy[i]=unmapped?raw[i]:pyramid(raw[i]);
    } else if(unmapped)sy=raw;
    if(mode==8)for(size_t i=0;i<count;i++)sy[i]=pyramid(raw[i]);
    unsigned S=local_entropy?8*K:K;
    std::vector<Bytes> encoded(S);std::vector<uint32_t> counts(S);
    Bytes selectors(K,0);
    const int choices=(mode==4 || mode==10 || mode==11)?8:1;
    for(int choice=0;choice<choices;choice++) {
        if(mode>=12)break;
        std::vector<Bytes> grouped(S);
        for(size_t i=0;i<count;i++) {
            uint8_t symbol=sy[i];
            if(choice) {
                uint8_t residual=uint8_t(input[i]-delta_predict_at(input.data(),i,choice));
                symbol=unmapped?residual:pyramid(residual);
            }
            grouped[cl[i]+(local_entropy?K*Streams::block(i):0)].push_back(symbol);
        }
        for(unsigned c=0;c<K;c++) {
            std::vector<Bytes> streams(local_entropy?8:1);size_t old=0,now=0;
            for(unsigned b=0;b<streams.size();b++) {
                streams[b]=pack(grouped[c+b*K]);now+=streams[b].size();old+=encoded[c+b*K].size();
            }
            if(choice==0 || now<old) {
                selectors[c]=uint8_t(choice);
                for(unsigned b=0;b<streams.size();b++) {
                    encoded[c+b*K]=std::move(streams[b]);counts[c+b*K]=uint32_t(grouped[c+b*K].size());
                }
            }
        }
    }
    if(mode>=12 && mode<=14)selectors=histogram_choices(input.data(),sy,cl,K,params[5],optimized?&delta_cache:nullptr);
    mark(2);
    if(mode==13 || mode==14) {
        std::vector<uint16_t> activities(count);
        if(optimized)require(ng_selected_symbols_fast(params.data(),params.size(),selectors.data(),input.data(),base.data(),delta_ptr,sy.data(),cl.data(),activities.data())==0);
        else require(ng_selected_observe(params.data(),params.size(),selectors.data(),input.data(),sy.data(),cl.data(),activities.data())==0);
        auto thresholds=activity_quantiles(activities,K);
        refit_biases(input.data(),params,activities,base,thresholds,sy,cl,optimized);
        if(mode==14) {
            thresholds=optimized_boundaries(input.data(),sy,activities,thresholds,params[5],optimized?&delta_cache:nullptr);
            refit_biases(input.data(),params,activities,base,thresholds,sy,cl,optimized);
        }
        selectors=histogram_choices(input.data(),sy,cl,K,params[5],optimized?&delta_cache:nullptr);
    }
    mark(3);
    if(mode>=10) {
        // Fix choices fitted on initial NG classes, then causally regenerate
        // classes and symbols using the errors of the selected predictors.
        if(optimized)require(ng_selected_symbols_fast(params.data(),params.size(),selectors.data(),input.data(),base.data(),delta_ptr,sy.data(),cl.data(),nullptr)==0);
        else {
            Bytes reconstructed(count);
            require(ng_selected_pass(params.data(),params.size(),selectors.data(),unmapped,
                nullptr,nullptr,reconstructed.data(),input.data(),sy.data(),cl.data())==0);
            require(reconstructed==input);
        }
        mark(4);
        if(mode>=11) {
            Bytes payload(count*2+65536);size_t size=0;
            auto coder=optimized?ng_wave_encode:ng_compact_encode;
            require(coder(params.data(),params.size(),sy.data(),cl.data(),payload.data(),payload.size(),&size)==0);
            payload.resize(size);
            Bytes frame={'N','D','C',uint8_t(optimized?'2':'1'),uint8_t(mode+(optimized?10:0)),uint8_t(K)};
            frame.insert(frame.end(),selectors.begin(),selectors.end());frame.insert(frame.end(),payload.begin(),payload.end());
            mark(5);return frame;
        } else {
            std::vector<Bytes> grouped(K);
            for(size_t i=0;i<count;i++)grouped[cl[i]].push_back(sy[i]);
            for(unsigned c=0;c<K;c++){encoded[c]=pack(grouped[c]);counts[c]=uint32_t(grouped[c].size());}
        }
    }
    Bytes frame={'N','D','E','1',uint8_t(mode),uint8_t(mask),uint8_t(unmapped),uint8_t(K)};
    if(local_entropy)frame[2]='S';
    put32(frame,uint32_t(params.size()));frame.insert(frame.end(),params.begin(),params.end());
    if(mode==9)for(auto t:activity_thr){frame.push_back(uint8_t(t));frame.push_back(uint8_t(t>>8));}
    if(mode==4 || mode==10)frame.insert(frame.end(),selectors.begin(),selectors.end());
    for(unsigned c=0;c<S;c++) {
        put32(frame,counts[c]);put32(frame,uint32_t(encoded[c].size()));
        frame.insert(frame.end(),encoded[c].begin(),encoded[c].end());
    }
    return frame;
}
Bytes decode(const Bytes& frame, double* rans_seconds=nullptr,int force_scalar=0) {
    if(frame.size()>=9 && std::memcmp(frame.data(),"NDF1",4)==0)return decode_fast(frame,force_scalar);
    if(frame.size()>=7 && std::memcmp(frame.data(),"NJB1",4)==0)return decode_joint(frame,force_scalar);
    if(frame.size()>=10 && std::memcmp(frame.data(),"NDB2",4)==0) {
        unsigned B=frame[4],flags=frame[5];require((B==4 || B==8 || B==16) && flags<=15);
        unsigned nb=NG_X/B,ny=NG_Y/B,nz=NG_Z/B,nblocks=nb*ny*nz;size_t pos=6;auto size=get32(frame,pos);
        require(size<=frame.size()-pos);Bytes choices(nblocks),dst(N);
        if(flags&1)unpack_stream(frame.data()+pos,size,choices,flags&8);
        else {
            require(size==(nblocks*3+7)/8);
            for(unsigned b=0;b<nblocks;b++)for(unsigned j=0;j<3;j++)choices[b]|=((frame[pos+(3*b+j)/8]>>((3*b+j)%8))&1)<<j;
        }
        for(auto choice:choices)require(choice<8);pos+=size;
        if(!(flags&2))require(ng_block_decode(frame.data()+pos,frame.size()-pos,choices.data(),dst.data(),force_scalar,B)==0);
        else {
            auto pn=get32(frame,pos);require(pn>=6 && pn<=frame.size()-pos);
            Bytes params(frame.begin()+pos,frame.begin()+pos+pn);pos+=pn;
            DirectionStreams s;s.K=params[4];require(s.K>=2 && s.K<=32);
            s.shift=B==4?2:B==8?3:4;s.separate=flags&4;s.choices=&choices;
            s.symbols.resize(s.K*(s.separate?8:1));s.pos.resize(s.symbols.size());size_t total=0;
            for(auto& values:s.symbols) {
                auto count=get32(frame,pos),bytes=get32(frame,pos);require(count<=N-total && bytes<=frame.size()-pos);
                values.resize(count);total+=count;
                if(count)unpack_stream(frame.data()+pos,bytes,values,flags&8);else require(bytes==0);
                pos+=bytes;
            }
            require(total==N && pos==frame.size());
            require(ng_block_stream_decode(params.data(),params.size(),choices.data(),dst.data(),force_scalar,B,DirectionStreams::read,&s)==0);
            for(size_t c=0;c<s.symbols.size();c++)require(s.pos[c]==s.symbols[c].size());
        }
        return dst;
    }
    if(frame.size()>=5 && std::memcmp(frame.data(),"NDB1",4)==0) {
        unsigned B=frame[4];require(B==4 || B==8 || B==16);
        unsigned nb=NG_BLOCK_DIM/B,nblocks=nb*nb*nb,offset=5+(nblocks*3+7)/8;
        require(frame.size()>offset);Bytes choices(nblocks),dst(N);
        for(unsigned b=0;b<nblocks;b++)for(unsigned j=0;j<3;j++)choices[b]|=((frame[5+(3*b+j)/8]>>((3*b+j)%8))&1)<<j;
        require(ng_block_decode(frame.data()+offset,frame.size()-offset,choices.data(),dst.data(),0,B)==0);
        return dst;
    }
    if(frame.size()>=6 && (std::memcmp(frame.data(),"NDC1",4)==0 || std::memcmp(frame.data(),"NDC2",4)==0)) {
        bool wave=frame[3]=='2';
        unsigned K=frame[5];require((wave?(frame[4]>=22 && frame[4]<=24):(frame[4]>=11 && frame[4]<=15)) && K>=2 && K<=32 && frame.size()>12+K);
        require(frame[6+K+4]==K);
        for(unsigned c=0;c<K;c++)require(frame[6+c]<=7);
        Bytes dst(N);
        if(wave)require(ng_wave_decode(frame.data()+6+K,frame.size()-6-K,frame.data()+6,dst.data(),0)==0);
        else require(ng_compact_decode(frame.data()+6+K,frame.size()-6-K,frame.data()+6,dst.data())==0);
        return dst;
    }
    require(frame.size()>=12 && (std::memcmp(frame.data(),"NDE1",4)==0 || std::memcmp(frame.data(),"NDS1",4)==0));
    bool local_entropy=frame[2]=='S';
    unsigned mode=frame[4],mask=frame[5],unmapped=frame[6],K=frame[7];
    require(mode<=10 && mask<=7 && unmapped<=1 && K>=1 && K<=32);
    size_t pos=8;unsigned pn=get32(frame,pos);require(pos+pn<=frame.size());
    Bytes params(frame.begin()+pos,frame.begin()+pos+pn);pos+=pn;
    require(params.size()>=6 && (mode==6 || params[4]==K));
    std::vector<uint16_t> activity_thr(K-1);
    if(mode==9) {
        require(K>=2 && pos+2*(K-1)<=frame.size());
        for(auto& t:activity_thr){t=uint16_t(frame[pos]|(frame[pos+1]<<8));pos+=2;}
    }
    Bytes selectors(K,0);
    if(mode==4 || mode==10) {
        require(pos+K<=frame.size());std::copy_n(frame.begin()+pos,K,selectors.begin());pos+=K;
        for(auto s:selectors)require(s<=7);
    }
    unsigned S=local_entropy?8*K:K;
    Streams streams;streams.classes=local_entropy?K:0;streams.symbols.resize(S);streams.pos.resize(S,0);size_t total=0;
    for(unsigned c=0;c<S;c++) {
        unsigned count=get32(frame,pos),n=get32(frame,pos);total+=count;
        require(total<=N && pos+n<=frame.size());streams.symbols[c].resize(count);
        if(count) {
            auto start=rans_seconds?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            require(delta_rans_decode(frame.data()+pos,n,streams.symbols[c].data(),count)==0);
            if(rans_seconds)*rans_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
        else require(n==0);
        pos+=n;
    }
    require(pos==frame.size() && total==N);
    Bytes dst(N);
    if(mode==6 || mode==9) {
        Bytes residual;
        if(mode==6) {
            require(K==1);residual=std::move(streams.symbols[0]);
            if(!unmapped)for(auto& v:residual)v=unpyramid(v);
        } else {
            residual.resize(N);
            require(ng_activity_decode(activity_thr.data(),int(K),unmapped,Streams::read,&streams,residual.data())==0);
            for(unsigned c=0;c<K;c++)require(streams.pos[c]==streams.symbols[c].size());
        }
        delta_reconstruct_residuals(residual.data(),mask);
        Raw r{&residual};require(ng_ex_reconstruct(params.data(),params.size(),0,nullptr,1,Raw::read,&r,dst.data())==0);
    } else {
        unsigned dm=mode>=1 && mode<=3?mask:0;
        if(mode==10)require(ng_selected_pass(params.data(),params.size(),selectors.data(),unmapped,
            Streams::read,&streams,dst.data(),nullptr,nullptr,nullptr)==0);
        else require(ng_ex_reconstruct(params.data(),params.size(),dm,mode==4?selectors.data():nullptr,
            mode==8?2:unmapped,Streams::read,&streams,dst.data())==0);
        for(unsigned c=0;c<S;c++)require(streams.pos[c]==streams.symbols[c].size());
        if(mode==5)delta_reconstruct_residuals(dst.data(),mask);
    }
    return dst;
}

void expand_fast_meta(const Bytes& meta,Bytes& choices,Bytes& classes) {
    const unsigned nb=NG_BLOCK_DIM/4;require(meta.size()==nb*nb*nb);
    choices.resize(meta.size());classes.resize(N);
    for(size_t i=0;i<meta.size();i++){require(meta[i]<128);choices[i]=meta[i]&7;}
    for(unsigned z=0;z<NG_Z;z++)for(unsigned y=0;y<NG_Y;y++) {
        const auto* row=meta.data()+((z/4)*nb+y/4)*nb;
        auto* dst=classes.data()+(size_t(z)*NG_Y+y)*NG_X;
        for(unsigned x=0;x<NG_BLOCK_DIM;x++)dst[x]=row[x/4]>>3;
    }
}
Bytes encode_fast(const uint8_t* src,const int* cfg,int mode) {
    require(cfg[3]==16);
    // Deliberately freeze existing baseline choices. This experiment targets
    // decoder design, not encoder search or a new predictor-selection policy.
    auto baseline=encode(src,cfg,504,0,false);size_t pos=6;
    auto bytes=get32(baseline,pos);require(bytes<=baseline.size()-pos);
    const unsigned nb=NG_BLOCK_DIM/4,nblocks=nb*nb*nb;
    Bytes choices(nblocks);unpack_stream(baseline.data()+pos,bytes,choices,true);pos+=bytes;
    size_t pn=0;require(ng_parameter_size(baseline.data()+pos,baseline.size()-pos,&pn)==0);
    Bytes params(baseline.begin()+pos,baseline.begin()+pos+pn),sy(N),classes,meta(nblocks);
    require(ng_fast_block_pass(params.data(),pn,choices.data(),src,sy.data(),nullptr,0)==0);
    std::vector<uint32_t> sums(nblocks);Bytes widths(nblocks);
    for(unsigned z=0;z<NG_Z;z++)for(unsigned y=0;y<NG_Y;y++)for(unsigned x=0;x<NG_BLOCK_DIM;x++) {
        size_t i=(size_t(z)*NG_Y+y)*NG_X+x,b=((z/4)*nb+y/4)*nb+x/4;
        sums[b]+=sy[i];widths[b]=std::max(widths[b],sy[i]);
    }
    auto sorted=sums;std::sort(sorted.begin(),sorted.end());std::array<uint32_t,15> thresholds;
    for(unsigned k=1;k<16;k++)thresholds[k-1]=sorted[k*nblocks/16];
    for(unsigned b=0;b<nblocks;b++) {
        unsigned c=0;
        if(!(mode&1))c=unsigned(std::lower_bound(thresholds.begin(),thresholds.end(),sums[b])-thresholds.begin());
        else {unsigned v=widths[b];while(v){c++;v>>=1;}}
        meta[b]=choices[b]|(c<<3);
    }
    expand_fast_meta(meta,choices,classes);
    auto packed=pack_stream(meta,true);Bytes payload(N*2+65536);size_t size=0;
    if(mode>=1004)require(ng_wave_encode(params.data(),pn,sy.data(),classes.data(),payload.data(),payload.size(),&size)==0);
    else require(ng_compact_encode(params.data(),pn,sy.data(),classes.data(),payload.data(),payload.size(),&size)==0);
    Bytes frame={'N','D','F','1',uint8_t(mode-1000)};put32(frame,uint32_t(packed.size()));
    frame.insert(frame.end(),packed.begin(),packed.end());frame.insert(frame.end(),payload.begin(),payload.begin()+size);
    return frame;
}
Bytes decode_fast(const Bytes& frame,int scalar,double* times) {
    auto start=std::chrono::steady_clock::now();
    require(frame.size()>=9 && frame[4]<=5);size_t pos=5;auto bytes=get32(frame,pos);
    require(bytes<=frame.size()-pos);const unsigned nb=NG_BLOCK_DIM/4;
    Bytes meta(nb*nb*nb),choices,classes,dst(N);
    unpack_stream(frame.data()+pos,bytes,meta,true);pos+=bytes;
    if(frame[4]>=4) {
        choices.resize(meta.size());for(size_t i=0;i<meta.size();i++){require(meta[i]<128);choices[i]=meta[i]&7;}
        auto metadata_end=std::chrono::steady_clock::now();
        require(ng_fast_context_decode(frame.data()+pos,frame.size()-pos,choices.data(),meta.data(),dst.data(),scalar)==0);
        if(times){times[0]=std::chrono::duration<double>(metadata_end-start).count();times[1]=0;
            times[2]=std::chrono::duration<double>(std::chrono::steady_clock::now()-metadata_end).count();}
        return dst;
    }
    Bytes sy(N);
    expand_fast_meta(meta,choices,classes);
    auto metadata_end=std::chrono::steady_clock::now();
    size_t pn=0;require(ng_parameter_size(frame.data()+pos,frame.size()-pos,&pn)==0);
    require(ng_known_classes_decode(frame.data()+pos,frame.size()-pos,classes.data(),sy.data(),scalar)==0);
    auto entropy_end=std::chrono::steady_clock::now();
    if(frame[4]>=2)require(ng_fast_wave_reconstruct(frame.data()+pos,pn,choices.data(),sy.data(),dst.data(),scalar)==0);
    else require(ng_fast_block_pass(frame.data()+pos,pn,choices.data(),nullptr,sy.data(),dst.data(),scalar)==0);
    if(times) {
        times[0]=std::chrono::duration<double>(metadata_end-start).count();
        times[1]=std::chrono::duration<double>(entropy_end-metadata_end).count();
        times[2]=std::chrono::duration<double>(std::chrono::steady_clock::now()-entropy_end).count();
    }
    return dst;
}

Bytes encode_joint(const uint8_t* src,const int* cfg,int mode) {
    const bool control=mode>=910;
    unsigned ml=control?1:(mode-900)/3,el=control?mode-910:(mode-900)%3;
    require(ml<3 && el<3 && mode!=909);
    unsigned K=cfg[3],nb=NG_BLOCK_DIM/4,blocks=nb*nb*nb;
    auto baseline=encode(src,cfg,504,0,false);size_t pos=6;auto selector_bytes=get32(baseline,pos);
    Bytes seed(blocks);unpack_stream(baseline.data()+pos,selector_bytes,seed,true);
    Bytes sy(N),cl(N),dst(N),params(65536);std::vector<uint16_t> predictions(N);size_t pn=0;
    require(ng_ex_prepare(src,cfg,0,0,params.data(),params.size(),&pn,sy.data(),cl.data(),predictions.data())==0);params.resize(pn);
    std::vector<Bytes> models;models.push_back(params);
    auto block_index=[&](size_t i){return unsigned(((i/(NG_BLOCK_DIM*NG_BLOCK_DIM)/4)*nb+(i/NG_BLOCK_DIM%NG_BLOCK_DIM)/4)*nb+(i%NG_BLOCK_DIM)/4);};
    if(!control && ml<2) {
        // Provisional directions: baseline choice, or lowest sum |Delta residual|
        // on NG-selected blocks. Fit one pooled correction, not seven averaged
        // predictions whose meaning would differ from an individual direction.
        std::vector<std::array<uint64_t,8>> costs(blocks);
        for(unsigned d=1;d<8;d++) {
            Bytes residual(src,src+N);delta_apply_mask(residual.data(),d);
            for(size_t i=0;i<N;i++){int r=int8_t(residual[i]);costs[block_index(i)][d]+=unsigned(r<0?-r:r);}
        }
        Bytes dirs=seed,masks(N);
        for(unsigned b=0;b<blocks;b++)if(!dirs[b])dirs[b]=uint8_t(std::min_element(costs[b].begin()+1,costs[b].end())-costs[b].begin());
        for(size_t i=0;i<N;i++){unsigned b=block_index(i);masks[i]=ml==0?seed[b]:dirs[b];}
        Bytes fitted(65536);size_t len=0;
        require(ng_mixed_prepare(src,cfg,masks.data(),fitted.data(),fitted.size(),&len,sy.data(),cl.data())==0);fitted.resize(len);
        if(ml==0)models[0]=std::move(fitted);else models.push_back(std::move(fitted));
    } else if(!control) {
        for(unsigned d=1;d<8;d++) {
            Bytes fitted(65536);size_t len=0;
            require(ng_ex_prepare(src,cfg,1,d,fitted.data(),fitted.size(),&len,sy.data(),cl.data(),predictions.data())==0);
            fitted.resize(len);models.push_back(std::move(fitted));
        }
    }
    const uint8_t* ptr[8];size_t sizes[8];
    for(size_t m=0;m<models.size();m++){ptr[m]=models[m].data();sizes[m]=models[m].size();}
    Bytes choices=seed;
    if(!control) {
        std::vector<std::array<double,15>> scores(blocks);
        for(unsigned candidate=0;candidate<15;candidate++) {
            require(ng_joint_pass(ptr,sizes,int(models.size()),nullptr,candidate,src,sy.data(),cl.data(),dst.data(),nullptr,nullptr,0)==0);
            require(std::memcmp(src,dst.data(),N)==0);
            std::vector<Histogram> hist(K);for(size_t i=0;i<N;i++)hist[cl[i]][sy[i]]++;
            std::vector<std::array<double,256>> cost(K);
            for(unsigned c=0;c<K;c++){double total=256;for(auto n:hist[c])total+=n;
                for(unsigned s=0;s<256;s++)cost[c][s]=std::log2(total)-std::log2(double(hist[c][s])+1);}
            for(size_t i=0;i<N;i++)scores[block_index(i)][candidate]+=cost[cl[i]][sy[i]];
        }
        for(unsigned b=0;b<blocks;b++)choices[b]=uint8_t(std::min_element(scores[b].begin(),scores[b].end())-scores[b].begin());
    }
    require(ng_joint_pass(ptr,sizes,int(models.size()),choices.data(),-1,src,sy.data(),cl.data(),dst.data(),nullptr,nullptr,0)==0);
    require(std::memcmp(src,dst.data(),N)==0);
    auto selector=pack_stream(choices,true);
    Bytes frame={'N','J','B','1',uint8_t(control?3:ml),uint8_t(el),uint8_t(models.size())};
    put32(frame,uint32_t(selector.size()));frame.insert(frame.end(),selector.begin(),selector.end());
    for(const auto& model:models){put32(frame,uint32_t(model.size()));frame.insert(frame.end(),model.begin(),model.end());}
    DirectionStreams streams;streams.K=K;streams.shift=2;streams.choices=&choices;streams.family=el==1;streams.separate=el==2;
    streams.symbols.resize(K*(el==0?1:el==1?3:15));
    for(size_t i=0;i<N;i++)streams.symbols[streams.group(i,cl[i])].push_back(sy[i]);
    for(const auto& values:streams.symbols){auto coded=pack_stream(values,true);put32(frame,uint32_t(values.size()));put32(frame,uint32_t(coded.size()));frame.insert(frame.end(),coded.begin(),coded.end());}
    return frame;
}
Bytes decode_joint(const Bytes& frame,int scalar) {
    require(frame.size()>=11 && frame[4]<=3 && frame[5]<=2);
    unsigned count=frame[6];require(count==1 || count==2 || count==8);
    size_t pos=7;auto sn=get32(frame,pos);require(sn<=frame.size()-pos);
    Bytes choices(N/64),dst(N);unpack_stream(frame.data()+pos,sn,choices,true);pos+=sn;
    for(auto c:choices)require(c<15);
    std::vector<Bytes> models(count);const uint8_t* ptr[8];size_t sizes[8];
    for(unsigned m=0;m<count;m++){auto len=get32(frame,pos);require(len>=6 && len<=frame.size()-pos);models[m]=Bytes(frame.begin()+pos,frame.begin()+pos+len);pos+=len;ptr[m]=models[m].data();sizes[m]=len;}
    DirectionStreams streams;streams.K=models[0][4];require(streams.K>=2 && streams.K<=32);
    for(auto& model:models)require(model[4]==streams.K);
    unsigned el=frame[5];streams.shift=2;streams.choices=&choices;streams.family=el==1;streams.separate=el==2;
    streams.symbols.resize(streams.K*(el==0?1:el==1?3:15));streams.pos.resize(streams.symbols.size());size_t total=0;
    for(auto& values:streams.symbols){auto n=get32(frame,pos),len=get32(frame,pos);require(n<=N-total && len<=frame.size()-pos);total+=n;values.resize(n);
        if(n)unpack_stream(frame.data()+pos,len,values,true);else require(len==0);pos+=len;}
    require(total==N && pos==frame.size());
    require(ng_joint_pass(ptr,sizes,count,choices.data(),-1,nullptr,nullptr,nullptr,dst.data(),DirectionStreams::read,&streams,scalar)==0);
    for(size_t c=0;c<streams.symbols.size();c++)require(streams.pos[c]==streams.symbols[c].size());
    return dst;
}
}

// Encode once, warm up, then measure decode only. Each sample is [total, rANS]
// seconds. Validation is outside the timed region; total includes allocations
// and destruction, while rANS includes its own table initialization.
extern "C" int ng_selector_mode_benchmark(const uint8_t* src,const int* cfg,int mode,int repeats,double* samples,uint64_t* bytes) {
    try {
        require(repeats>0);
        require(mode==4 || mode==10);
        auto frame=encode(src,cfg,mode,0,false,false);
        if(bytes)*bytes=frame.size();
        auto warm=decode(frame);
        require(std::memcmp(src,warm.data(),N)==0);
        for(int i=0;i<repeats;i++) {
            double rans=0;bool valid;
            auto start=std::chrono::steady_clock::now();
            double elapsed;
            {
                auto dst=decode(frame,&rans);
                elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                valid=std::memcmp(src,dst.data(),N)==0;
                auto cleanup=std::chrono::steady_clock::now();
                Bytes().swap(dst);
                elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-cleanup).count();
            }
            require(valid);samples[2*i]=elapsed;samples[2*i+1]=rans;
        }
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"ng_selector_decode_benchmark: %s\n",e.what());return -1;}
}

extern "C" int ng_selector_decode_benchmark(const uint8_t* src,const int* cfg,int repeats,double* samples) {
    return ng_selector_mode_benchmark(src,cfg,4,repeats,samples,nullptr);
}

namespace {
struct GeometryScope {
    ng_geometry saved=ng_current_geometry;
    GeometryScope(uint32_t x,uint32_t y,uint32_t z) {ng_current_geometry={int(x),int(y),int(z)};}
    ~GeometryScope(){ng_current_geometry=saved;}
};
}
extern "C" size_t ng_leaf_encode_bound(uint32_t x,uint32_t y,uint32_t z) {
    if(!ng_geometry_valid(x,y,z))return 0;
    return size_t(x)*y*z*3+131072;
}
namespace {
struct ActivityHistogram {
    // One scan per causal selector, reused by every threshold candidate.
    // Prefix at a+1 includes activity a, matching lower_bound classification.
    std::vector<Histogram> prefix;
    std::array<uint64_t,2561> counts{};
    ActivityHistogram(const Bytes&raw,const std::vector<uint16_t>&activity,bool force=false,bool symbols=false,bool needed=true){
        if(!needed||(!encoder_bound_enabled&&!force))return;
        prefix.resize(2561);
        for(size_t i=0;i<raw.size();i++){prefix[activity[i]+1][symbols?raw[i]:pyramid(raw[i])]++;counts[activity[i]+1]++;}
        for(unsigned a=1;a<2561;a++){
            counts[a]+=counts[a-1];
            for(unsigned s=0;s<256;s++)prefix[a][s]+=prefix[a-1][s];
        }
    }
    std::vector<uint64_t> partition_key(const std::vector<uint16_t>&thr)const{
        std::vector<uint64_t> key;key.reserve(thr.size());
        for(auto t:thr)key.push_back(counts[std::min(2560u,unsigned(t)+1)]);
        return key;
    }
    double coding_cost(const std::vector<uint16_t>&thr,bool conservative)const{
        const unsigned K=thr.size()+1;double lower=thr.size()*2+K*8;unsigned begin=0;
        for(unsigned c=0;c<K;c++){
            unsigned end=c<thr.size()?std::min(2560u,unsigned(thr[c])+1):2560;
            uint64_t hist[256],count=0;
            for(unsigned s=0;s<256;s++){hist[s]=prefix[end][s]-prefix[begin][s];count+=hist[s];}
            begin=end;if(!count)continue;
            uint16_t freq[256];uint8_t table[514];
            const size_t bound=delta_rans_size_lower_bound(hist,count,freq);
            if(conservative)lower+=bound;
            else{
                double bits=0;for(unsigned s=0;s<256;s++)if(hist[s])bits+=hist[s]*(12.-std::log2(double(freq[s])));
                lower+=64.+bits/8.; // exact normalized frequencies; final-state occupancy is estimated
            }
            lower+=ng_write_freq_table(table,freq);
        }
        return lower;
    }
    size_t lower_bound(const std::vector<uint16_t>&thr)const{return size_t(coding_cost(thr,true));}
    double estimate(const std::vector<uint16_t>&thr)const{return coding_cost(thr,false);}
};
Bytes activity_stream_payload(const Bytes&raw,const std::vector<uint16_t>&thr,const Bytes&cl,
                              const std::vector<uint32_t>*block=nullptr,const Bytes*choices=nullptr,unsigned groups=1){
    EncoderWorkTimer timer(5);
    if(encoder_work_enabled){encoder_work[3]++;encoder_work[4]+=raw.size();}
    unsigned K=thr.size()+1;const size_t voxels=raw.size();
    std::vector<size_t> counts(K*groups),offsets(K*groups+1),positions;
    for(size_t i=0;i<voxels;i++)counts[cl[i]+(groups>1?K*(*choices)[(*block)[i]]:0)]++;
    for(size_t c=0;c<counts.size();c++)offsets[c+1]=offsets[c]+counts[c];
    positions=offsets;Bytes streams(voxels);
    for(size_t i=0;i<voxels;i++)streams[positions[cl[i]+(groups>1?K*(*choices)[(*block)[i]]:0)]++]=pyramid(raw[i]);
    Bytes frame;for(auto t:thr){frame.push_back(uint8_t(t));frame.push_back(uint8_t(t>>8));}
    for(size_t c=0;c<counts.size();c++){auto b=pack_stream_span(streams.data()+offsets[c],counts[c],true);put32(frame,counts[c]);put32(frame,b.size());frame.insert(frame.end(),b.begin(),b.end());}
    return frame;
}
Bytes activity_payload(const Bytes&raw,int K){
    Bytes cl(N,0);std::vector<uint16_t>thr(K-1);
    if(K>1)require(ng_activity_classes(raw.data(),K,thr.data(),cl.data())==0);
    return activity_stream_payload(raw,thr,cl);
}
template<class T>std::vector<uint16_t> activity_parse(const Bytes&frame,size_t&p,unsigned K,unsigned groups,T&streams){
    std::vector<uint16_t>thr(K-1);
    for(auto&t:thr){require(frame.size()-p>=2);t=unsigned(frame[p])|(unsigned(frame[p+1])<<8);p+=2;}
    require(std::is_sorted(thr.begin(),thr.end()));
    streams.symbols.resize(K*groups);streams.pos.resize(K*groups,0);size_t total=0;
    for(auto&s:streams.symbols){size_t count=get32(frame,p),bytes=get32(frame,p);require(count<=N-total&&bytes<=frame.size()-p);total+=count;s.resize(count);
        if(count)unpack_stream(frame.data()+p,bytes,s,true);else require(bytes==0);p+=bytes;}
    require(p==frame.size()&&total==N);
    return thr;
}
void activity_read(const Bytes&frame,size_t&p,unsigned K,uint8_t*out){
    Streams streams;auto thr=activity_parse(frame,p,K,1,streams);
    if(K==1)for(size_t i=0;i<N;i++)out[i]=unpyramid(streams.symbols[0][i]);
    else {require(ng_activity_decode(thr.data(),K,0,Streams::read,&streams,out)==0);
        for(unsigned c=0;c<K;c++)require(streams.pos[c]==streams.symbols[c].size());}
}
Bytes activity_fixed(const Bytes&raw,int d,int mask){
    Bytes frame{'N','A','C','1'};put32(frame,d);put32(frame,mask);put32(frame,16);
    auto payload=activity_payload(raw,16);frame.insert(frame.end(),payload.begin(),payload.end());return frame;
}
void activity_reselect(const Bytes&,const Bytes&,const std::vector<uint32_t>&,
                      const std::vector<const uint8_t*>&,Bytes&,unsigned);
}
// Activity-only ablation: absolutely no ng_ex_prepare/model fitting. Same
// causal NG activity stencil and quantile thresholds, standalone rANS streams.
extern "C" int ng_activity_only_encode(const uint8_t*src,int d,int mask,int K,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&out&&written&&d>=32&&d<=256&&d%32==0&&mask>=0&&mask<=7&&(K==1||K==16));
        GeometryScope scope(d,d,d);Bytes raw(src,src+N);delta_apply_mask(raw.data(),mask);
        Bytes frame{'N','A','C','1'};put32(frame,d);put32(frame,mask);put32(frame,K);
        auto payload=activity_payload(raw,K);frame.insert(frame.end(),payload.begin(),payload.end());
        require(frame.size()<=cap);std::memcpy(out,frame.data(),frame.size());*written=frame.size();return 0;
    }catch(...){return -1;}
}
extern "C" int ng_activity_only_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    try{
        require(src&&out&&n>=16&&!std::memcmp(src,"NAC1",4));Bytes frame(src,src+n);size_t p=4;
        unsigned d=get32(frame,p),mask=get32(frame,p),K=get32(frame,p);
        require(d>=32&&d<=256&&d%32==0&&mask<=7&&(K==1||K==16)&&cap>=size_t(d)*d*d);
        GeometryScope scope(d,d,d);activity_read(frame,p,K,out);
        delta_reconstruct_residuals(out,mask);return 0;
    }catch(...){return -1;}
}
// Eight choices: direct0 plus X,Y,XY,Z,XZ,YZ,XYZ. Predictors keep causal
// neighbors across block boundaries; blocks select a direction, not a reset.
extern "C" int ng_activity_blocks_encode(const uint8_t*src,int d,int B,uint8_t*out,size_t cap,size_t*written,uint64_t*stats){
    try{
        require(src&&out&&written&&d>=32&&d<=256&&d%32==0&&(B==d||B==4||B==8||B==16));
        GeometryScope scope(d,d,d);std::array<Bytes,8>raw,sy;std::array<const uint8_t*,8>ptr{};
        Bytes best;unsigned bestmask=0;Bytes bestchoices;size_t selector_bytes=0;
        for(unsigned m=0;m<8;m++){
            raw[m]=Bytes(src,src+N);delta_apply_mask(raw[m].data(),m);
            sy[m].resize(N);for(size_t i=0;i<N;i++)sy[m][i]=pyramid(raw[m][i]);ptr[m]=sy[m].data();
            auto f=activity_fixed(raw[m],d,m);if(best.empty()||f.size()<best.size()){best=std::move(f);bestmask=m;}
        }
        unsigned nb=d/B,nblocks=nb*nb*nb;std::vector<uint32_t>block(N);
        if(B!=d){
            for(unsigned z=0;z<unsigned(d);z++)for(unsigned y=0;y<unsigned(d);y++)for(unsigned x=0;x<unsigned(d);x++)block[(size_t(z)*d+y)*d+x]=((z/B)*nb+y/B)*nb+x/B;
            Bytes cl(N);std::vector<uint16_t>thr(15);require(ng_activity_classes(raw[bestmask].data(),16,thr.data(),cl.data())==0);
            std::array<Bytes,2>seeds{Bytes(nblocks,uint8_t(bestmask)),choose_blocks(ptr.data(),8,cl.data(),16,B)};
            for(auto choices:seeds)for(int pass=0;pass<3;pass++){
                Bytes residual(N);for(size_t i=0;i<N;i++)residual[i]=raw[choices[block[i]]][i];
                auto sel=pack_stream(choices,true),payload=activity_payload(residual,16);
                Bytes f{'N','A','B','1'};put32(f,d);put32(f,B);put32(f,16);put32(f,nblocks);put32(f,sel.size());f.insert(f.end(),sel.begin(),sel.end());f.insert(f.end(),payload.begin(),payload.end());
                if(f.size()<best.size()){best=std::move(f);bestchoices=choices;selector_bytes=sel.size();}
                if(pass==2)break;
                // Re-score under the currently selected class distributions.
                // Classes are an estimate here; final classes/tables are refit
                // and actual bytes measured for EVERY candidate iteration.
                require(ng_activity_classes(residual.data(),16,thr.data(),cl.data())==0);
                for(auto& v:residual)v=pyramid(v);
                activity_reselect(residual,cl,block,std::vector<const uint8_t*>(ptr.begin(),ptr.end()),choices,1);
            }
        }
        if(stats){std::fill_n(stats,11,0);if(bestchoices.empty())stats[bestmask]=nblocks;else for(auto v:bestchoices)stats[v]++;stats[8]=nblocks;stats[9]=selector_bytes;stats[10]=!bestchoices.empty();}
        require(best.size()<=cap);std::memcpy(out,best.data(),best.size());*written=best.size();return 0;
    }catch(...){return -1;}
}
extern "C" int ng_activity_blocks_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    try{
        require(src&&out&&n>=16);if(!std::memcmp(src,"NAC1",4))return ng_activity_only_decode(src,n,out,cap);
        require(n>=24&&!std::memcmp(src,"NAB1",4));Bytes frame(src,src+n);size_t p=4;
        unsigned d=get32(frame,p),B=get32(frame,p),K=get32(frame,p);
        require(d>=32&&d<=256&&d%32==0&&(B==4||B==8||B==16)&&K==16&&cap>=size_t(d)*d*d);
        GeometryScope scope(d,d,d);unsigned nb=d/B;size_t count=get32(frame,p),bytes=get32(frame,p);
        require(count==size_t(nb)*nb*nb&&bytes<=frame.size()-p);Bytes choices(count);unpack_stream(frame.data()+p,bytes,choices,true);p+=bytes;
        for(auto m:choices)require(m<=7);activity_read(frame,p,K,out);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned bx=0;bx<nb;bx++){
            unsigned mask=choices[(z/B*nb+y/B)*nb+bx];size_t row=(size_t(z)*d+y)*d;
            for(unsigned x=bx*B;x<(bx+1)*B;x++){size_t i=row+x;out[i]=uint8_t(out[i]+delta_predict_xyz(out,i,mask,x,y,z,d,size_t(d)*d));}}
        return 0;
    }catch(...){return -1;}
}
namespace {
// One parameter-free extra predictor: 3D Lorenzo, clamped to its three
// immediate causal neighbors. Boundaries use the available Delta axes.
uint8_t clamped_delta_prediction(const uint8_t*src,size_t i,unsigned mask,unsigned x,unsigned y,unsigned z,unsigned d){
    if(mask==3||mask==5||mask==6||mask==7)return ng_directional_base(src,i,mask==3?8:mask==5?9:mask==6?10:11,x,y,z,d);
    return delta_predict_xyz(src,i,mask,x,y,z,d,size_t(d)*d);
}
uint8_t activity_prediction(const uint8_t*src,size_t i,unsigned m,unsigned x,unsigned y,unsigned z,unsigned d){
    if(m<8)return delta_predict_xyz(src,i,m,x,y,z,d,size_t(d)*d);
    return clamped_delta_prediction(src,i,7,x,y,z,d);
}
Bytes activity_classify(const std::vector<uint16_t>&a,const std::vector<uint16_t>&thr){
    std::array<uint8_t,2560>lut{};
    for(unsigned v=0;v<lut.size();v++)lut[v]=std::lower_bound(thr.begin(),thr.end(),v)-thr.begin();
    Bytes cl(a.size());for(size_t i=0;i<a.size();i++)cl[i]=lut[a[i]];return cl;
}
std::vector<uint16_t> activity_refine(const Bytes&raw,const std::vector<uint16_t>&a,
        const std::vector<uint16_t>&old,const std::vector<uint32_t>&block,const Bytes&choices,unsigned groups,bool symbols=false,
        unsigned grid=64,bool global=false){
    auto edges=grid?activity_quantiles(a,grid):std::vector<uint16_t>{};
    if(!grid)for(auto t:old)for(int v=std::max(0,int(t)-16);v<=std::min(2559,int(t)+16);v++)edges.push_back(uint16_t(v));
    edges.insert(edges.end(),old.begin(),old.end());edges.push_back(2559);
    std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    unsigned E=edges.size();std::vector<Histogram>prefix((E+1)*groups);
    std::array<unsigned,2560>bin{};for(unsigned v=0;v<bin.size();v++)bin[v]=std::lower_bound(edges.begin(),edges.end(),v)-edges.begin();
    for(size_t i=0;i<a.size();i++)prefix[(bin[a[i]]+1)*groups+(groups>1?choices[block[i]]:0)][symbols?raw[i]:pyramid(raw[i])]++;
    for(unsigned e=1;e<=E;e++)for(unsigned g=0;g<groups;g++)for(unsigned s=0;s<256;s++)prefix[e*groups+g][s]+=prefix[(e-1)*groups+g][s];
    std::vector<double>cache((E+1)*(E+1),-1);
    auto cost=[&](unsigned l,unsigned r){double&result=cache[l*(E+1)+r];if(result>=0)return result;result=0;
        for(unsigned g=0;g<groups;g++){Histogram h{};for(unsigned s=0;s<256;s++)h[s]=prefix[r*groups+g][s]-prefix[l*groups+g][s];result+=histogram_cost(h.data(),12,true);}return result;};
    if(global)return partition_boundaries(edges,old.size()+1,cost,true);
    std::vector<unsigned>cuts{0};for(auto t:old)cuts.push_back(std::lower_bound(edges.begin(),edges.end(),t)-edges.begin()+1);cuts.push_back(E);
    for(int pass=0;pass<2;pass++)for(unsigned k=1;k+1<cuts.size();k++){
        unsigned best=cuts[k];double score=cost(cuts[k-1],best)+cost(best,cuts[k+1]);
        for(unsigned v=std::max(1u,cuts[k-1]);v<=cuts[k+1];v++){double s=cost(cuts[k-1],v)+cost(v,cuts[k+1]);if(s<score){score=s;best=v;}}
        cuts[k]=best;
    }
    std::vector<uint16_t>thr;for(unsigned k=1;k+1<cuts.size();k++)thr.push_back(edges[cuts[k]-1]);return thr;
}
std::vector<std::array<double,256>> activity_symbol_costs(const Bytes& symbols,const Bytes&cl,
        const std::vector<uint32_t>&block,const Bytes&sel,unsigned groups){
    std::vector<Histogram>hist(16*groups);
    for(size_t i=0;i<symbols.size();i++)hist[cl[i]+16*(groups>1?sel[block[i]]:0)][symbols[i]]++;
    std::vector<std::array<double,256>>cost(hist.size());
    for(unsigned c=0;c<hist.size();c++){double total=256;for(auto v:hist[c])total+=v;for(unsigned v=0;v<256;v++)cost[c][v]=std::log2(total)-std::log2(hist[c][v]+1.);}
    return cost;
}
void activity_select_costs(const Bytes&cl,const std::vector<uint32_t>&block,
        const std::vector<const uint8_t*>&candidates,Bytes&sel,unsigned groups,
        const std::vector<std::array<double,256>>&cost,const std::vector<Bytes>*candidate_classes=nullptr){
    if(encoder_work_enabled)encoder_work[6]++;
    unsigned P=candidates.size();size_t blocks=sel.size();
    std::vector<unsigned>counts(P);
    for(auto m:sel)counts[m]++;
    std::vector<double>scores(blocks*P);
    // Candidate-major traversal reads each residual array sequentially. Each
    // block/candidate sum still receives voxels in exactly the original order.
    const size_t voxels=cl.size();
    bool vector_blocks=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    vector_blocks=!ng_runtime_only&&__builtin_cpu_supports("avx2")&&voxels%16==0;
    if(vector_blocks)for(size_t i=0;i<voxels;i+=16){
        for(unsigned j=0;j<16;j++)if(block[i+j]!=block[i]+j/4){vector_blocks=false;break;}
        if(!vector_blocks)break;
    }
#endif
    if(ng_screen_work&&ng_screen_size==voxels)vector_blocks=false;
    for(unsigned m=0;m<P;m++){
        const uint8_t* candidate=candidates[m];
        const auto&current_classes=candidate_classes?candidate_classes->at(m):cl;
        const auto* table=cost.data()+16*(groups>1?m:0);
        double* sums=scores.data()+m*blocks;
        if(groups==1&&!candidate_classes){
            unsigned prior=0;while(prior<m&&candidates[prior]!=candidate)++prior;
            if(prior<m){std::copy_n(scores.data()+prior*blocks,blocks,sums);continue;}
        }
        if(encoder_work_enabled)encoder_work[7]+=voxels;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(vector_blocks){ng_score_four_blocks(candidate,current_classes.data(),block.data(),voxels,table[0].data(),sums);continue;}
#endif
        // Repeated block indices form short contiguous X runs. Keep their sum
        // in a register; do not load/store it for every voxel. No reassociation.
        for(size_t i=0;i<voxels;){
            if(ng_screen_work&&ng_screen_size==voxels&&!ng_screen_work[i]){++i;continue;}
            const unsigned b=block[i];double sum=sums[b];
            do {
                if(!ng_screen_work||ng_screen_size!=voxels||ng_screen_work[i])sum+=table[current_classes[i]][candidate[i]];
                ++i;
            } while(i<voxels&&block[i]==b);
            sums[b]=sum;
        }
    }
    std::vector<double> selector_cost(P);
    for(unsigned m=0;m<P;m++)selector_cost[m]=std::log2(blocks+double(P))-std::log2(counts[m]+1.);
    for(size_t b=0;b<blocks;b++){
        unsigned best=0;double best_score=std::numeric_limits<double>::infinity();
        for(unsigned m=0;m<P;m++){
            double score=scores[m*blocks+b]+selector_cost[m];
            if(score<best_score){best_score=score;best=m;}
        }
        sel[b]=best;}
}
void activity_reselect(const Bytes& symbols,const Bytes&cl,const std::vector<uint32_t>&block,
        const std::vector<const uint8_t*>&candidates,Bytes&sel,unsigned groups){
    auto cost=activity_symbol_costs(symbols,cl,block,sel,groups);
    activity_select_costs(cl,block,candidates,sel,groups,cost);
}
}
#include "ng_lorenzo_trial.hpp"
#include "ng_direction_bank.hpp"
#include "ng_packed_bank.hpp"
#include "ng_choice_grid.hpp"
#include "ng_adaptive_grid.hpp"
// Experimental selector-grid setting; stream headers carry it for decoding.
extern "C" int ng_codec_choice_edge(unsigned edge){
    if(edge!=2&&edge!=4&&edge!=8&&edge!=16&&edge!=32)return -1;
    ng_choice_edge=edge;return 0;
}
#include "ng_zero_blocks.hpp"
#include "ng_small_bank_trial.hpp"
#include "ng_choice_frame.hpp"
#include "ng_shared_grid_screen.hpp"
#include "ng_fast_split.hpp"
#include "ng_exact_partition.hpp"
#include "ng_three_candidate.hpp"
extern "C" void ng_codec_fit_memo(unsigned enabled){
    small_bank::memo::entries.clear();small_bank::memo::used=0;
    small_bank::memo::hits=small_bank::memo::misses=0;small_bank::memo::enabled=enabled!=0;
}
extern "C" void ng_codec_fit_memo_stats(uint64_t*out){if(out){out[0]=small_bank::memo::hits;out[1]=small_bank::memo::misses;out[2]=small_bank::memo::used;}}
extern "C" int ng_codec_encode(const uint8_t*s,unsigned d,unsigned preset,uint8_t*out,size_t cap,size_t*n){
    if(preset>7)return -1;
    const bool old=choice_frame::enabled;choice_frame::enabled=true;
    int rc=preset==1?(three_candidate::screen_singleton_default?ng_slow_singleton_screen_encode(s,d,out,cap,n):ng_three_candidate_encode_options(s,d,1,7,out,cap,n)):ng_screen_pair_encode(s,d,preset>=2?9+preset:10,out,cap,n);
    choice_frame::enabled=old;return rc;
}
extern "C" int ng_codec_encode_legacy_slow(const uint8_t*s,unsigned d,uint8_t*out,size_t cap,size_t*n){
    three_candidate::LegacySlowScope legacy;return ng_codec_encode(s,d,1,out,cap,n);
}
extern "C" int ng_codec_encode_shared_grid(const uint8_t*s,unsigned d,unsigned preset,unsigned policy,uint8_t*out,size_t cap,size_t*n){
    if((preset!=1&&preset!=3)||(policy!=0&&policy!=4&&policy!=8&&policy!=16&&policy!=32))return -1;
    std::fill_n(shared_grid::estimates,5,0.);shared_grid::selected_edge=4;
    choice_grid::Scope grid(4);adaptive_grid::Scope adaptive(0);shared_grid::Scope shared(policy);
    return ng_codec_encode(s,d,preset,out,cap,n);
}
extern "C" void ng_shared_grid_stats(double*out,unsigned*edge){std::copy_n(shared_grid::estimates,5,out);*edge=shared_grid::selected_edge;}
/* Diagnostic control for matched zero-choice ablations, not another preset. */
extern "C" void ng_codec_set_zero_fit_mask(unsigned enable){zero_blocks::mask_fits=enable!=0;}
extern "C" int ng_codec_encode_zero_control(const uint8_t*s,unsigned d,unsigned preset,unsigned enable,uint8_t*out,size_t cap,size_t*n){
    if(enable>3)return -1;bool old=zero_blocks::enabled,fit=zero_blocks::mask_fits;zero_blocks::enabled=enable&1;zero_blocks::mask_fits=!(enable&2);
    int rc=ng_codec_encode(s,d,preset,out,cap,n);zero_blocks::enabled=old;zero_blocks::mask_fits=fit;return rc;
}
// Retained slow/slower presets return the selected NCF1 frame directly.
// Previously this travelled through a tree-experiment capture/replay wrapper,
// whose non-tree mode simply returned this same frame unchanged.
extern "C" int ng_rc_encode_fixed(const uint8_t*src,unsigned d,unsigned config,uint8_t*out,size_t cap,size_t*n){
    if(config>1)return -1;
    const int cfg[6]={3,3,config?3:1,16,config?160:40,1};
    const bool old=choice_frame::enabled;choice_frame::enabled=true;
    const int context=ng_fit_context_mode;ng_fit_context_mode=0;
    int rc=ng_screen_pair_encode_impl(src,d,12,out,cap,n,cfg,16,3,true,nullptr,3);
    ng_fit_context_mode=context;choice_frame::enabled=old;return rc;
}
// Stable public preset adapter; keep experimental switches out of the API.
extern "C" int vc_ng_preset_encode(const uint8_t*s,unsigned d,unsigned preset,uint8_t*out,size_t cap,size_t*n){
    if(preset>3)return -1;
    const double previous=ng_fit_intercept_weight;
    ng_fit_intercept_weight=preset>=2?100.:0.;
    choice_grid::Scope fixed(4);
    int rc=preset<2?ng_codec_encode(s,d,preset==0?3:1,out,cap,n):ng_rc_encode_fixed(s,d,preset-2,out,cap,n);
    ng_fit_intercept_weight=previous;return rc;
}
extern "C" int vc_ng_preset_decode(const uint8_t*s,size_t n,uint8_t*out,size_t cap){
    const int old=ng_native_fast_control;ng_native_fast_control=1;
    int rc=ng_codec_decode(s,n,out,cap);
    ng_native_fast_control=old;return rc;
}
extern "C" unsigned vc_ng_frame_dimension(const uint8_t*s,size_t n){
    try {
        require(s&&n>=12&&!memcmp(s,"NCF1",4));
        Bytes prefix(s,s+12);size_t p=4;auto hn=get32(prefix,p),zn=get32(prefix,p);
        require(hn>=4&&hn<=1024*1024&&zn<=n-12);
        auto header=packed_bank::decompress(s+12,zn,hn);packed_bank::Reader h{header};
        unsigned d=h.u(4);require(ng_geometry_valid(d,d,d));return d;
    } catch(...) {return 0;}
}
namespace {
Bytes encode_rect_payload(const uint8_t* src,uint32_t x,uint32_t y,uint32_t z,const int*cfg){
    require(src&&cfg&&ng_geometry_valid(x,y,z));
    GeometryScope geometry(x,y,z);return encode(src,cfg,504,0,false);
}
int decode_rect_payload(const uint8_t* src,size_t n,uint32_t x,uint32_t y,uint32_t z,uint8_t*dst,size_t cap,int scalar){
    require(src&&dst&&n>=10&&ng_geometry_valid(x,y,z)&&cap>=size_t(x)*y*z);
    require(!std::memcmp(src,"NDB2",4)&&src[4]==4&&src[5]==9);
    GeometryScope geometry(x,y,z);Bytes payload(src,src+n);
    auto out=decode(payload,nullptr,scalar);std::memcpy(dst,out.data(),out.size());return 0;
}
}
extern "C" int vc_ng_rect_encode(const uint8_t*src,unsigned x,unsigned y,unsigned z,uint8_t*dst,size_t cap,size_t*n){
    try{require(dst&&n);const int cfg[6]={3,3,1,16,40,128};auto payload=encode_rect_payload(src,x,y,z,cfg);
        require(payload.size()<=cap);std::memcpy(dst,payload.data(),payload.size());*n=payload.size();return 0;
    }catch(...){return -1;}
}
extern "C" int vc_ng_rect_decode(const uint8_t*src,size_t n,unsigned x,unsigned y,unsigned z,uint8_t*dst,size_t cap){
    try{return decode_rect_payload(src,n,x,y,z,dst,cap,0);}catch(...){return -1;}
}

extern "C" int ng_activity_refined_encode(const uint8_t*src,int d,unsigned flags,uint8_t*out,size_t cap,size_t*written){
    try{
        unsigned conditional=flags&(32|64|128|512|1024|2048|4096|8192);
        require(src&&out&&written&&d>=32&&d<=256&&d%32==0&&flags<=16383&&!((flags&16)&&(flags&1)));
        require(!conditional||((flags&16)&&(conditional==32||conditional==64||conditional==128||conditional==512||conditional==1024||conditional==2048||conditional==4096||conditional==8192)));
        GeometryScope scope(d,d,d);size_t base_size=0;
        require(ng_activity_blocks_encode(src,d,4,out,cap,&base_size,nullptr)==0);
        Bytes best(out,out+base_size);if(!flags){*written=base_size;return 0;}
        unsigned P=(flags&17)?9:8,groups=(flags&2)?P:1,nb=d/4;size_t blocks=size_t(nb)*nb*nb;
        Bytes params,conditional_params;std::vector<uint16_t>prediction;
        if(flags&16){
            const int cfg[6]={3,3,1,16,40,128};params.resize(65536);prediction.resize(N);Bytes sy(N),cl(N);size_t pn=0;
            require(ng_ex_prepare(src,cfg,4,0,params.data(),params.size(),&pn,sy.data(),cl.data(),prediction.data())==0);params.resize(pn);
            if(conditional){conditional_params.resize(2048);std::vector<uint16_t>updated(N);size_t cn=0;
                require(ng_difficulty_prepare(src,params.data(),params.size(),prediction.data(),conditional==4096?7:conditional==8192?8:conditional==1024?5:conditional==2048?6:conditional==32?1:conditional==64?2:conditional==128?3:4,
                    (flags&256)?0.1:0.001,conditional_params.data(),conditional_params.size(),&cn,updated.data())==0);
                conditional_params.resize(cn);prediction.swap(updated);
                /* The no-feature audit has identical predictions and must
                 * also emit the original frame, without test-only metadata. */
                if(conditional==2048){conditional=0;conditional_params.clear();}}
        }
        Bytes choices(blocks);size_t pos=8;
        if(best[2]=='C'){unsigned m=get32(best,pos);std::fill(choices.begin(),choices.end(),m);}
        else {pos=16;require(get32(best,pos)==blocks);auto bytes=get32(best,pos);unpack_stream(best.data()+pos,bytes,choices,true);}
        std::vector<uint32_t>block(N);std::vector<Bytes>raw(P,Bytes(N)),sy(P,Bytes(N));std::vector<const uint8_t*>ptr(P);
        std::vector<Bytes>fixed(12,Bytes(N));bank_fixed_candidates(src,fixed,d);
        for(unsigned z=0;z<unsigned(d);z++)for(unsigned y=0;y<unsigned(d);y++)for(unsigned x=0;x<unsigned(d);x++){
            size_t i=(size_t(z)*d+y)*d+x;block[i]=((z/4)*nb+y/4)*nb+x/4;}
        for(unsigned m=0;m<P;m++){
            sy[m]=std::move(fixed[m==8?11:m]);
            if(m==8&&(flags&16))for(size_t i=0;i<prediction.size();i++)sy[m][i]=pyramid(uint8_t(src[i]-uint8_t(uint16_t(prediction[i]+128u)>>8)));
            for(size_t i=0;i<sy[m].size();i++)raw[m][i]=unpyramid(sy[m][i]);
        }
        for(unsigned m=0;m<P;m++)ptr[m]=sy[m].data();
        Bytes residual(N);for(size_t i=0;i<N;i++)residual[i]=raw[choices[block[i]]][i];
        std::vector<uint16_t>a(N),thr(15);Bytes cl(N);require(ng_activity_classes(residual.data(),16,thr.data(),cl.data())==0);
        std::array<Bytes,2>seeds{choices,choose_blocks(ptr.data(),P,cl.data(),16,4)};
        for(auto sel:seeds)for(unsigned pass=0;pass<3;pass++){
            for(size_t i=0;i<N;i++)residual[i]=raw[sel[block[i]]][i];
            require(ng_activity_values(residual.data(),a.data())==0);thr=activity_quantiles(a,16);
            auto evaluate=[&](const std::vector<uint16_t>&t){
                auto c=activity_classify(a,t),payload=activity_stream_payload(residual,t,c,&block,&sel,groups),encoded_sel=pack_stream(sel,true);
                Bytes f{'N','A',uint8_t(conditional?'F':(flags&16)?'E':'D'),uint8_t(conditional?'2':'1')};put32(f,d);put32(f,P);put32(f,groups);put32(f,encoded_sel.size());f.insert(f.end(),encoded_sel.begin(),encoded_sel.end());
                if(flags&16){put32(f,params.size());f.insert(f.end(),params.begin(),params.end());}
                if(conditional){put32(f,conditional_params.size());f.insert(f.end(),conditional_params.begin(),conditional_params.end());}
                f.insert(f.end(),payload.begin(),payload.end());
                size_t size=f.size();if(size<best.size())best=std::move(f);return size;};
            auto quantile_size=evaluate(thr);
            if((flags&4)&&((flags&8)||pass==2)){
                auto tuned=activity_refine(residual,a,thr,block,sel,groups);
                if(evaluate(tuned)<quantile_size)thr=std::move(tuned);
            }
            if(pass==2)break;cl=activity_classify(a,thr);
            Bytes symbols(N);for(size_t i=0;i<N;i++)symbols[i]=pyramid(residual[i]);
            activity_reselect(symbols,cl,block,ptr,sel,groups);
        }
        require(best.size()<=cap);std::memcpy(out,best.data(),best.size());*written=best.size();return 0;
    }catch(...){return -1;}
}

extern "C" int ng_activity_refined_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    try{
        require(src&&out&&n>=20);bool conditional=!std::memcmp(src,"NAF2",4),fitted=conditional||!std::memcmp(src,"NAE1",4);if(!fitted&&std::memcmp(src,"NAD1",4))return ng_activity_blocks_decode(src,n,out,cap);
        Bytes frame(src,src+n);size_t p=4;unsigned d=get32(frame,p),P=get32(frame,p),groups=get32(frame,p),bytes=get32(frame,p);
        require(d>=32&&d<=256&&d%32==0&&(P==8||P==9)&&(groups==1||groups==P)&&cap>=size_t(d)*d*d&&bytes<=n-p);
        GeometryScope scope(d,d,d);unsigned nb=d/4;Bytes choices(size_t(nb)*nb*nb);unpack_stream(frame.data()+p,bytes,choices,true);p+=bytes;for(auto m:choices)require(m<P);
        std::unique_ptr<void,decltype(&ng_causal_destroy)>state(nullptr,ng_causal_destroy);
        if(fitted){require(P==9);auto pn=get32(frame,p);require(pn<=n-p);state.reset(ng_causal_create(frame.data()+p,pn));require(bool(state));p+=pn;}
        Bytes conditional_params;
        if(conditional){auto cn=get32(frame,p);require(cn<=n-p);conditional_params.assign(frame.begin()+p,frame.begin()+p+cn);p+=cn;}
        DirectionStreams streams;streams.choices=&choices;streams.K=16;streams.shift=2;streams.dimension=d;streams.separate=groups>1;
        auto thr=activity_parse(frame,p,16,groups,streams);
        require(ng_activity_decode(thr.data(),16,0,DirectionStreams::read,&streams,out)==0);
        for(unsigned c=0;c<streams.pos.size();c++)require(streams.pos[c]==streams.symbols[c].size());
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;unsigned m=choices[((z/4)*nb+y/4)*nb+x/4];uint16_t q=0;uint8_t cl=0;
            if(fitted)require(ng_causal_prediction(state.get(),i,&q,&cl)==0);
            uint8_t basepred=uint8_t(uint16_t(q+128u)>>8);
            if(conditional)require(ng_difficulty_prediction(state.get(),conditional_params.data(),conditional_params.size(),i,&q)==0);
            uint8_t pred=(fitted&&m==8)?uint8_t(uint16_t(q+128u)>>8):activity_prediction(out,i,m,x,y,z,d);
            out[i]=uint8_t(out[i]+pred);if(fitted)require(ng_causal_accept(state.get(),i,out[i],conditional?(conditional_params[0]>=5?uint8_t(uint16_t(q+128u)>>8):basepred):pred)==0);}
        return 0;
    }catch(...){return -1;}
}

extern "C" int ng_leaf_decode(const uint8_t* src,size_t src_n,uint8_t* dst,size_t cap,int scalar) {
    if(!src || !dst || src_n<26 || std::memcmp(src,"NDL1",4))return -1;
    try {
        Bytes header(src,src+16);size_t pos=4;auto x=get32(header,pos),y=get32(header,pos),z=get32(header,pos);
        if(!ng_geometry_valid(x,y,z) || cap<size_t(x)*y*z)return -1;
        return decode_rect_payload(src+16,src_n-16,x,y,z,dst,cap,scalar);
    }catch(...){return -1;}
}
