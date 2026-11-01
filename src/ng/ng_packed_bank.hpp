/* Complete native baseline. Python is not involved in any codec operation.
 * NBK1 -> omitted/sparse-difference Zstd models -> joint Zstd rANS tables.
 * Retains the previously tested NPK1/NBK2/RPK1 representation. */
#include <zstd.h>
#include "ng_packed_bank.h"
namespace packed_bank {
struct Reader {
    const Bytes& b;size_t p=0;
    void need(size_t n){require(p<=b.size()&&n<=b.size()-p);}
    uint64_t u(unsigned n){need(n);uint64_t v=0;for(unsigned k=0;k<n;k++)v|=uint64_t(b[p++])<<(8*k);return v;}
    uint64_t var(){uint64_t v=0;for(unsigned s=0;s<63;s+=7){auto x=u(1);v|=(x&127)<<s;if(x<128)return v;}throw std::runtime_error("varint overflow");}
    Bytes take(size_t n){need(n);Bytes r(b.begin()+p,b.begin()+p+n);p+=n;return r;}
    void skip(size_t n){need(n);p+=n;}
    Bytes rest(){return take(b.size()-p);}
    void end(){require(p==b.size());}
};
void append(Bytes& a,const Bytes& b){a.insert(a.end(),b.begin(),b.end());}
void put16(Bytes&b,unsigned v){b.push_back(v);b.push_back(v>>8);}
void var(Bytes&b,uint64_t v){while(v>=128){b.push_back((v&127)|128);v>>=7;}b.push_back(v);}
int64_t signed_value(uint64_t v,unsigned bits){return v<(uint64_t(1)<<(bits-1))?int64_t(v):int64_t(v)-(int64_t(1)<<bits);}
Bytes compress(const Bytes&b,int level=3){Bytes out(ZSTD_compressBound(b.size()));auto n=ZSTD_compress(out.data(),out.size(),b.data(),b.size(),level);require(!ZSTD_isError(n));out.resize(n);return out;}
Bytes decompress(const uint8_t*src,size_t size,size_t n){require(n<=1048576);Bytes out(n);auto got=ZSTD_decompress(out.data(),n,src,size);require(!ZSTD_isError(got)&&got==n);return out;}
Bytes decompress(const Bytes&b,size_t n){return decompress(b.data(),b.size(),n);}
struct Model {Bytes params,extra;};
using Models=std::array<Model,12>;
Bytes raw_models(const Models&m,unsigned mask){Bytes out;for(unsigned k=0;k<12;k++)if(mask&(1u<<k)){put32(out,m[k].params.size());append(out,m[k].params);append(out,m[k].extra);}return out;}
std::vector<int64_t> fields(const Model&m){
    Reader r{m.params};auto h=r.take(6);unsigned K=h[4];require(K>=2&&K<=32&&m.params.size()%2==0);
    size_t n=(m.params.size()-6)/2;require(n>2*K-1);size_t nb=n-(2*K-1);require(nb%2==0);
    std::vector<int64_t> v;for(size_t k=0;k<nb;k++)v.push_back(signed_value(r.u(2),16));
    int64_t prev=0;for(unsigned k=0;k<K-1;k++){auto t=r.u(2);v.push_back(int64_t(t)-prev);prev=t;}
    for(unsigned k=0;k<K;k++)v.push_back(signed_value(r.u(2),16));r.end();
    if(m.extra.empty()){v.push_back(0);v.push_back(0);}else{Reader e{m.extra};v.push_back(signed_value(e.u(4),32));v.push_back(signed_value(e.u(4),32));e.end();}return v;
}
Model unfields(const Bytes&h,size_t pn,const std::vector<int64_t>&v,bool extra){
    require(h.size()==6&&pn>=10&&pn<=65536&&pn%2==0);unsigned K=h[4];require(K>=2&&K<=32);
    size_t n=(pn-6)/2;require(n>2*K-1&&v.size()==n+2);size_t nb=n-(2*K-1);require(nb%2==0);
    Model m;m.params=h;
    auto signed16=[&](int64_t x){require(x>=-32768&&x<=32767);put16(m.params,uint16_t(x));};
    for(size_t k=0;k<nb;k++)signed16(v[k]);int64_t sum=0;
    for(size_t k=nb;k<nb+K-1;k++){sum+=v[k];require(sum>=0&&sum<=65535);put16(m.params,sum);}
    for(size_t k=nb+K-1;k<n;k++)signed16(v[k]);
    if(extra)for(size_t k=n;k<n+2;k++){require(v[k]>=INT32_MIN&&v[k]<=INT32_MAX);put32(m.extra,uint32_t(v[k]));}
    return m;
}
void sparse_vector(Bytes&b,const std::vector<int64_t>&v){size_t p=b.size();b.resize(p+(v.size()+7)/8,0);for(size_t k=0;k<v.size();k++)if(v[k]){b[p+k/8]|=1u<<(k%8);var(b,v[k]>=0?uint64_t(v[k])*2:uint64_t(-v[k])*2-1);}}
std::vector<int64_t> read_vector(Reader&r,size_t n){auto bits=r.take((n+7)/8);std::vector<int64_t>v(n);for(size_t k=0;k<n;k++)if(bits[k/8]&(1u<<(k%8))){auto x=r.var();require(x<=uint64_t(INT64_MAX));v[k]=(x&1)?-int64_t(x/2)-1:int64_t(x/2);}return v;}
std::vector<unsigned> order(unsigned mask,int ref){std::vector<unsigned>v;if(ref>=0){require(ref<12&&(mask&(1u<<ref)));v.push_back(ref);}for(unsigned m=0;m<12;m++)if((mask&(1u<<m))&&int(m)!=ref)v.push_back(m);return v;}
Bytes sparse_models(const Models&m,unsigned mask,int ref){
    Bytes out{uint8_t(ref<0?255:ref)};std::vector<int64_t>base;
    for(auto k:order(mask,ref)){auto values=fields(m[k]),encoded=values;
        out.insert(out.end(),m[k].params.begin(),m[k].params.begin()+6);var(out,m[k].params.size());
        if(!base.empty()){require(base.size()==encoded.size());for(size_t j=0;j<encoded.size();j++)encoded[j]-=base[j];}
        sparse_vector(out,encoded);if(int(k)==ref)base=std::move(values);
    }return out;
}
Models unsparse_models(const Bytes&data,unsigned mask,unsigned mode){
    Reader r{data};unsigned id=r.u(1);int ref=id==255?-1:int(id);Models m;std::vector<int64_t>base;
    for(auto k:order(mask,ref)){auto h=r.take(6);size_t pn=r.var();require(pn>=10&&pn<=65536&&pn%2==0);auto v=read_vector(r,(pn-6)/2+2);
        if(!base.empty()){require(base.size()==v.size());for(size_t j=0;j<v.size();j++){require(v[j]>=-INT64_C(4294967296)&&v[j]<=INT64_C(4294967296));v[j]+=base[j];}}
        m[k]=unfields(h,pn,v,mode&&k);if(int(k)==ref)base=std::move(v);
    }r.end();return m;
}
// raw_only retains just referenced models without trying compressed metadata
// representations. NWP1 compresses its complete metadata once at the end.
Bytes model_pack(const Bytes&f,bool shared=false,bool raw_only=false){
    Reader r{f};require(r.take(4)==Bytes({'N','B','K','1'}));unsigned d=r.u(4),mode=r.u(4),sn=r.u(4);require(mode<2&&ng_geometry_valid(d,d,d));
    auto selector=r.take(sn);Bytes choices(size_t(d/4)*(d/4)*(d/4));unpack_stream(selector.data(),selector.size(),choices,true);unsigned mask=0;
    for(auto c:choices){require(c<24);if(c>=12)mask|=1u<<(c-12);}
    Models models;for(unsigned m=0;m<12;m++){auto n=r.u(4);models[m].params=r.take(n);if(mode&&m)models[m].extra=r.take(8);}
    auto residual=r.rest();Bytes prefix;put32(prefix,d);put32(prefix,mode);put32(prefix,sn);append(prefix,selector);put16(prefix,mask);
    Bytes best={'N','B','K','2'};append(best,prefix);append(best,raw_models(models,mask));append(best,residual);
    if(raw_only)return best;
    for(int ref=-1;ref<12;ref++){if(ref>=0&&!(mask&(1u<<ref)))continue;auto sparse=sparse_models(models,mask,ref),z=compress(sparse);
        Bytes candidate={'N','P','K','1'};append(candidate,prefix);candidate.push_back(3);put32(candidate,sparse.size());put32(candidate,z.size());append(candidate,z);append(candidate,residual);
        if(candidate.size()<best.size())best=std::move(candidate);
    }
    if(shared&&mask){
        for(unsigned m=1;m<12;m++)require(models[m].params==models[0].params&&(!mode||models[m].extra==models[1].extra));
        Bytes data;put32(data,models[0].params.size());append(data,models[0].params);if(mode)append(data,models[1].extra);
        auto z=compress(data);Bytes candidate={'N','P','K','1'};append(candidate,prefix);candidate.push_back(4);put32(candidate,data.size());put32(candidate,z.size());append(candidate,z);append(candidate,residual);
        if(candidate.size()<best.size())best=std::move(candidate);
    }return best;
}
Bytes model_unpack(const Bytes&f){
    if(f.size()>=4&&!memcmp(f.data(),"NBK2",4))return f;
    Reader r{f};require(r.take(4)==Bytes({'N','P','K','1'}));unsigned d=r.u(4),mode=r.u(4),sn=r.u(4);require(mode<2&&ng_geometry_valid(d,d,d));auto selector=r.take(sn);
    unsigned mask=r.u(2),codec=r.u(1);require(mask<4096&&(codec==3||codec==4));size_t decoded=r.u(4);auto z=r.take(r.u(4));auto data=decompress(z,decoded);Models models;
    if(codec==3)models=unsparse_models(data,mask,mode);
    else{Reader sr{data};auto params=sr.take(sr.u(4));Bytes extra;if(mode)extra=sr.take(8);sr.end();
        for(unsigned m=0;m<12;m++)if(mask&(1u<<m)){models[m].params=params;if(mode&&m)models[m].extra=extra;}}
    Bytes out={'N','B','K','2'};put32(out,d);put32(out,mode);put32(out,sn);append(out,selector);put16(out,mask);append(out,raw_models(models,mask));append(out,r.rest());return out;
}
Bytes table_unpack(const Bytes&);
bool is_wave(const uint8_t* p,size_t n){return n>=4&&(!memcmp(p,"NWP1",4)||!memcmp(p,"NWM1",4));}
// NWM1 transforms only the selected model section. The selector and entropy
// header retain their exact bytes and are Zstd-compressed jointly as before.
Bytes metadata_restore(const Bytes&b){
    Reader r{b};require(r.take(4)==Bytes({'N','B','K','3'}));r.u(4);unsigned mode=r.u(4);require(mode<2);r.skip(r.u(4));unsigned mask=r.u(2);require(mask<4096);
    Bytes out(b.begin(),b.begin()+r.p);auto models=unsparse_models(r.take(r.var()),mask,mode);
    append(out,raw_models(models,mask));append(out,r.rest());return out;
}
Bytes metadata_unpack(const uint8_t*src,size_t stored,size_t decoded,bool transformed){
    auto b=decompress(src,stored,decoded);return transformed?metadata_restore(b):b;
}
Bytes wave_unpack(const Bytes& f){
    if(!is_wave(f.data(),f.size()))return model_unpack(table_unpack(f));
    Reader r{f};r.skip(4);auto decoded=r.u(4);auto z=r.take(r.u(4));auto result=metadata_unpack(z.data(),z.size(),decoded,!memcmp(f.data(),"NWM1",4));append(result,r.rest());return result;
}
using Range=std::pair<size_t,size_t>;
std::vector<Range> table_ranges(const Bytes&f){
    Reader r{f};auto magic=r.take(4);require(magic==Bytes({'N','P','K','1'})||magic==Bytes({'N','B','K','2'}));r.u(4);unsigned mode=r.u(4),sn=r.u(4);std::vector<Range>ranges;
    auto add=[&](size_t n){r.need(n);uint16_t freq[256];size_t used=0;require(ng_read_freq_table(f.data()+r.p,n,freq,&used)==0&&used<=n);ranges.emplace_back(r.p,r.p+used);r.skip(n);};
    add(sn);unsigned mask=r.u(2);require(mask<4096&&mode<2);
    if(magic[0]=='N'&&magic[1]=='P'){r.u(1);r.u(4);r.skip(r.u(4));}
    else for(unsigned m=0;m<12;m++)if(mask&(1u<<m)){r.skip(r.u(4));if(mode&&m)r.skip(8);}
    r.skip(30);for(unsigned c=0;c<16;c++){auto count=r.u(4),n=r.u(4);if(count)add(n);else require(n==0);}r.end();return ranges;
}
Bytes table_pack(const Bytes&f){
    auto ranges=table_ranges(f);Bytes tables,body,gaps;size_t last=0;
    for(auto[a,b]:ranges){var(gaps,a-last);body.insert(body.end(),f.begin()+last,f.begin()+a);tables.insert(tables.end(),f.begin()+a,f.begin()+b);last=b;}
    body.insert(body.end(),f.begin()+last,f.end());auto z=compress(tables);Bytes out={'R','P','K','1',3,uint8_t(ranges.size())};put32(out,tables.size());put32(out,z.size());append(out,gaps);append(out,z);append(out,body);return out.size()<f.size()?out:f;
}
Bytes table_unpack(const Bytes&f){
    if(f.size()>=4&&memcmp(f.data(),"RPK1",4))return f;
    Reader r{f};require(r.take(4)==Bytes({'R','P','K','1'}));require(r.u(1)==3);unsigned count=r.u(1);require(count>0&&count<=17);size_t decoded=r.u(4),stored=r.u(4);require(decoded<=65536);
    std::vector<uint32_t>gaps;for(unsigned c=0;c<count;c++){auto g=r.var();require(g<=UINT32_MAX);gaps.push_back(g);}
    auto tables=decompress(r.take(stored),decoded),body=r.rest();Bytes out(tables.size()+body.size());
    require(ng_restore_compact_tables(tables.data(),tables.size(),gaps.data(),count,body.data(),body.size(),out.data(),out.size())==0);return out;
}
}
/* Repack only metadata, never re-fit/reselect/re-encode residuals.
 * method0 raw,1 sparse fields,2 best reference-model delta,3 best including
 * unchanged input. Returned frame carries its own transform discriminator. */
extern "C" int ng_packed_bank_metadata_repack(const uint8_t*src,size_t n,unsigned method,int level,uint8_t*out,size_t cap,size_t*written){
    try{
        using namespace packed_bank;require(src&&out&&written&&n>=12&&!memcmp(src,"NWP1",4)&&method<=3&&level>=1&&level<=19);
        Bytes f(src,src+n);Reader fr{f};fr.skip(4);auto decoded=fr.u(4);auto z=fr.take(fr.u(4));auto metadata=decompress(z,decoded),payload=fr.rest();
        Reader r{metadata};require(r.take(4)==Bytes({'N','B','K','3'}));r.u(4);unsigned mode=r.u(4);require(mode<2);r.skip(r.u(4));unsigned mask=r.u(2);require(mask<4096);
        Bytes prefix(metadata.begin(),metadata.begin()+r.p);Models models;
        for(unsigned k=0;k<12;k++)if(mask&(1u<<k)){models[k].params=r.take(r.u(4));if(mode&&k)models[k].extra=r.take(8);}
        auto tail=r.rest();Bytes best=method==3?f:Bytes{};
        auto trial=[&](const Bytes&m,bool transformed){auto packed=compress(m,level);Bytes frame=transformed?Bytes({'N','W','M','1'}):Bytes({'N','W','P','1'});put32(frame,m.size());put32(frame,packed.size());append(frame,packed);append(frame,payload);if(best.empty()||frame.size()<best.size())best=std::move(frame);};
        if(method==0||method==3)trial(metadata,false);
        if(method>=1)for(int ref=-1;ref<12;ref++){
            if(method==1&&ref>=0)break;if(ref>=0&&!(mask&(1u<<ref)))continue;
            auto encoded=sparse_models(models,mask,ref);Bytes m=prefix;var(m,encoded.size());append(m,encoded);append(m,tail);trial(m,true);
        }
        require(best.size()<=cap);memcpy(out,best.data(),best.size());*written=best.size();return 0;
    }catch(...){return -1;}
}
// Audit API restores exactly the original uncompressed metadata and payload.
extern "C" int ng_packed_bank_metadata_unwrap(const uint8_t*src,size_t n,uint8_t*out,size_t cap,size_t*written){
    try{require(src&&out&&written&&packed_bank::is_wave(src,n));auto b=packed_bank::wave_unpack(Bytes(src,src+n));require(b.size()<=cap);memcpy(out,b.data(),b.size());*written=b.size();return 0;}catch(...){return -1;}
}
extern "C" int ng_packed_bank_repack(const uint8_t*src,size_t n,uint8_t*out,size_t cap,size_t*written){
    try{require(src&&out&&written);auto result=packed_bank::table_pack(packed_bank::model_pack(Bytes(src,src+n)));require(result.size()<=cap);memcpy(out,result.data(),result.size());*written=result.size();return 0;}catch(...){return -1;}
}
/* Reuse an already selected/fitted bank; only change its entropy layout.
 * No-model banks use a zero-weight history model. All codec work is native. */
static int ng_packed_bank_wave_repack_impl(const uint8_t*src,size_t n,uint8_t*out,size_t cap,size_t*written,const BankWinner* winner){
    EncoderWorkTimer final_timer(12);
    int stage=0;
    try{
        using namespace packed_bank;
        require(out&&written&&(winner||(src&&n>=4)));
        auto f=winner?model_pack(winner->prefix,true,true):model_unpack(table_unpack(Bytes(src,src+n)));Reader r{f};r.skip(4);unsigned d=r.u(4),mode=r.u(4);r.skip(r.u(4));size_t mask_at=r.p;unsigned mask=r.u(2);
        GeometryScope scope(d,d,d);Bytes first,feature;
        for(unsigned m=0;m<12;m++)if(mask&(1u<<m)){
            // Entropy uses one geometry/class header, while reconstruction
            // retains every selected predictor's original parameters below.
            auto par=r.take(r.u(4));if(first.empty())first=par;
            if(mode&&m)r.skip(8);
        }
        stage=1;
        if(!mask){first.resize(65536);size_t pn=0;require(ng_bank_empty_model(first.data(),first.size(),&pn)==0);first.resize(pn);}
        stage=2;size_t prefix=r.p,p=r.p;Bytes gathered_sym,gathered_cl;std::vector<uint16_t> gathered_thr;
        const Bytes& sym=winner?winner->symbols:gathered_sym;
        const Bytes& cl=winner?winner->classes:gathered_cl;
        const auto& thr=winner?winner->thresholds:gathered_thr;
        if(!winner){
        Streams streams;gathered_thr=activity_parse(f,p,16,1,streams);
        const uint8_t* ptr[16];size_t counts[16];for(int c=0;c<16;c++){ptr[c]=streams.symbols[c].data();counts[c]=streams.symbols[c].size();}
        stage=3;Bytes raw(N);gathered_sym.resize(N);gathered_cl.resize(N);require(ng_activity_gather(thr.data(),16,ptr,counts,streams.pos.data(),raw.data())==0);
        for(int c=0;c<16;c++)require(streams.pos[c]==counts[c]);
        stage=4;std::vector<uint16_t> activity(N);require(ng_activity_values(raw.data(),activity.data())==0);
        for(size_t i=0;i<N;i++){gathered_sym[i]=pyramid(raw[i]);gathered_cl[i]=std::lower_bound(thr.begin(),thr.end(),activity[i])-thr.begin();}
        }
        stage=5;int16_t bias[16]={0};require(ng_set_classes(first.data(),first.size(),thr.data(),bias)==0);
        stage=6;Bytes entropy(3*N+65536);size_t en=0;require(ng_wave_encode(first.data(),first.size(),sym.data(),cl.data(),entropy.data(),entropy.size(),&en)==0);entropy.resize(en);
        stage=7;size_t header=0;require(ng_entropy_header_size(entropy.data(),entropy.size(),&header)==0);
        Bytes metadata(f.begin(),f.begin()+prefix);metadata[3]='3';
        if(!mask){metadata[mask_at]=1;put32(metadata,first.size());append(metadata,first);}
        metadata.insert(metadata.end(),entropy.begin(),entropy.begin()+header);
        auto z=compress(metadata);Bytes result={'N','W','P','1'};put32(result,metadata.size());put32(result,z.size());append(result,z);result.insert(result.end(),entropy.begin()+header,entropy.end());
        require(result.size()<=cap);memcpy(out,result.data(),result.size());*written=result.size();return 0;
    }catch(...){return -100-stage;}
}
extern "C" int ng_packed_bank_wave_repack(const uint8_t*src,size_t n,uint8_t*out,size_t cap,size_t*written){
    return ng_packed_bank_wave_repack_impl(src,n,out,cap,written,nullptr);
}
extern "C" int ng_packed_bank_legacy_encode(const uint8_t*src,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*written){
    try{require(src&&out&&written&&ng_geometry_valid(d,d,d)&&mode<4);Bytes raw(size_t(d)*d*d*3+131072);size_t n=0;
        require(ng_direction_bank_encode(src,d,mode,raw.data(),raw.size(),&n,nullptr)==0);
        if(mode<2)return ng_packed_bank_repack(raw.data(),n,out,cap,written);
        raw.resize(n);auto result=packed_bank::table_pack(packed_bank::model_pack(raw,true));require(result.size()<=cap);memcpy(out,result.data(),result.size());*written=result.size();return 0;
    }catch(...){return -1;}
}
static int ng_packed_bank_view_impl(const uint8_t*,size_t,uint8_t*,size_t,double*,bool);
struct SimpleActivityScope {
    unsigned kind=ng_simple_activity_kind;
    const uint16_t* values=ng_simple_activity_values;
    ~SimpleActivityScope(){ng_simple_activity_kind=kind;ng_simple_activity_values=values;}
};
extern "C" int ng_packed_bank_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    if(src&&n>=4&&!memcmp(src,"NFS1",4))return ng_fast_split_decode(src,n,out,cap);
    if(src&&n>=8&&!memcmp(src,"NFA1",4)){
        unsigned kind=src[4];if(kind<1||kind>2||src[5]||src[6]||src[7]||n<20||memcmp(src+8,"NWP1",4))return -1;
        SimpleActivityScope scope;ng_simple_activity_kind=kind;ng_simple_activity_values=nullptr;
        return ng_packed_bank_view_impl(src+8,n-8,out,cap,nullptr,true);
    }
    if(src&&packed_bank::is_wave(src,n))return ng_packed_bank_view_impl(src,n,out,cap,nullptr,true);
    try{require(src&&out&&n>=4);auto native=packed_bank::wave_unpack(Bytes(src,src+n));return ng_direction_bank_decode(native.data(),native.size(),out,cap);}catch(...){return -1;}
}
extern "C" int ng_packed_bank_wave_encode(const uint8_t*src,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&out&&written&&ng_geometry_valid(d,d,d)&&(mode==2||mode==3));
        Bytes fitted(size_t(d)*d*d*3+131072);size_t n=0;BankWinner winner;
        require(ng_direction_bank_encode_impl(src,d,mode,fitted.data(),fitted.size(),&n,nullptr,0,&winner)==0);
        return ng_packed_bank_wave_repack_impl(fitted.data(),n,out,cap,written,&winner);
    }catch(...){return -1;}
}
extern "C" int ng_packed_bank_encode(const uint8_t*src,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*written){
    return mode>=2?ng_packed_bank_wave_encode(src,d,mode,out,cap,written):ng_packed_bank_legacy_encode(src,d,mode,out,cap,written);
}
extern "C" int ng_packed_bank_fast_encode(const uint8_t*src,unsigned d,unsigned effort,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&out&&written&&ng_geometry_valid(d,d,d)&&effort>=1&&effort<=3);
        Bytes raw(size_t(d)*d*d*3+131072);size_t n=0;BankWinner winner;
        require(ng_direction_bank_encode_impl(src,d,3,raw.data(),raw.size(),&n,nullptr,effort,&winner)==0);
        return ng_packed_bank_wave_repack_impl(raw.data(),n,out,cap,written,&winner);
    }catch(...){return -1;}
}
extern "C" int ng_packed_bank_search_encode(const uint8_t*src,unsigned d,const ng_bank_search_options*options,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&options&&out&&written&&ng_geometry_valid(d,d,d));
        // A sole estimated candidate hands its state directly to the writer.
        // No full-volume trial output buffer is needed in that case.
        Bytes raw(bank_single_estimated_trial(options)?1:size_t(d)*d*d*3+131072);size_t n=0;BankWinner winner;
        require(ng_direction_bank_encode_impl(src,d,3,raw.data(),raw.size(),&n,nullptr,0,&winner,options)==0);
        return ng_packed_bank_wave_repack_impl(raw.data(),n,out,cap,written,&winner);
    }catch(...){return -1;}
}
/* Native experiment: independently fitted predictors, one shared causal
 * activity field. Eight-byte wrapper explicitly records the changed context.
 * No context map is transmitted or supplied to the decoder. */
extern "C" int ng_packed_bank_fixed_activity_encode(const uint8_t*src,unsigned d,unsigned kind,const ng_bank_search_options*options,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&out&&written&&options&&options->seed==18&&options->rounds==1&&options->passes==1&&kind>=1&&kind<=2&&cap>=8&&ng_geometry_valid(d,d,d));
        GeometryScope geometry(d,d,d);SimpleActivityScope scope;
        Bytes error(N);std::vector<uint16_t> activity(N);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;error[i]=ng_simple_activity_error(src,i,x,y,z,d,kind);
        }
        require(ng_activity_values(error.data(),activity.data())==0);
        ng_simple_activity_kind=kind;ng_simple_activity_values=activity.data();size_t n=0;
        require(ng_packed_bank_search_encode(src,d,options,out+8,cap-8,&n)==0);
        memcpy(out,"NFA1",4);out[4]=kind;out[5]=out[6]=out[7]=0;*written=n+8;return 0;
    }catch(...){return -1;}
}
extern "C" int ng_packed_bank_sampled_encode(const uint8_t*src,unsigned d,unsigned models,unsigned stride,const ng_bank_search_options*options,uint8_t*out,size_t cap,size_t*written){
    if(!options||options->seed!=18||options->rounds!=1||options->passes!=1||(models!=2&&models!=12)||stride<16||stride>8192||ng_simple_activity_kind)return -1;
    struct Scope {unsigned stride=bank_sparse_stride,models=bank_sparse_models;~Scope(){bank_sparse_stride=stride;bank_sparse_models=models;}} scope;
    bank_sparse_stride=stride;bank_sparse_models=models;
    return ng_packed_bank_search_encode(src,d,options,out,cap,written);
}
extern "C" int ng_packed_bank_dense_two_encode(const uint8_t*src,unsigned d,unsigned stride,const ng_bank_search_options*options,uint8_t*out,size_t cap,size_t*written){
    if(!options||options->seed!=18||options->rounds!=1||options->passes!=1||stride<1||stride>128||ng_simple_activity_kind)return -1;
    struct Scope {unsigned sparse=bank_sparse_stride,models=bank_sparse_models,stride=bank_regression_stride;~Scope(){bank_sparse_stride=sparse;bank_sparse_models=models;bank_regression_stride=stride;}} scope;
    bank_sparse_stride=0;bank_sparse_models=2;bank_regression_stride=stride;
    return ng_packed_bank_search_encode(src,d,options,out,cap,written);
}
extern "C" int ng_packed_bank_profile(const uint8_t*src,size_t n,uint8_t*out,size_t cap,double*times){
    if(src&&times&&packed_bank::is_wave(src,n))return ng_packed_bank_view_impl(src,n,out,cap,times,true);
    try{require(src&&out&&times&&n>=4);auto start=std::chrono::steady_clock::now();auto tables=packed_bank::table_unpack(Bytes(src,src+n));auto middle=std::chrono::steady_clock::now();auto native=packed_bank::model_unpack(tables);auto end=std::chrono::steady_clock::now();
        times[0]=std::chrono::duration<double>(middle-start).count();times[1]=std::chrono::duration<double>(end-middle).count();return ng_direction_bank_profile(native.data(),native.size(),out,cap,times+2);
    }catch(...){return -1;}
}
/* Borrow the entropy payload. Only the small Zstd metadata is materialized;
 * the same validated decoder accepts disjoint metadata and payload spans. */
static int ng_packed_bank_view_impl(const uint8_t*src,size_t n,uint8_t*out,size_t cap,double*times,bool reuse=false){
    try{
        require(src&&out&&n>=12&&packed_bank::is_wave(src,n));
        auto start=std::chrono::steady_clock::now();Bytes prefix(src,src+12);size_t p=4;
        size_t decoded=get32(prefix,p),stored=get32(prefix,p);require(stored<=n-12);
        auto metadata=packed_bank::metadata_unpack(src+12,stored,decoded,!memcmp(src,"NWM1",4));
        require(metadata.size()>=4&&!memcmp(metadata.data(),"NBK3",4));
        const uint8_t*payload=src+12+stored;size_t payload_size=n-12-stored;
        if(times){times[0]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();times[1]=0;
            return ng_direction_bank_profile_impl(metadata.data(),metadata.size(),out,cap,times+2,payload,payload_size,reuse);}
        return ng_direction_bank_decode_impl(metadata.data(),metadata.size(),out,cap,payload,payload_size,reuse);
    }catch(...){return -1;}
}
extern "C" int ng_packed_bank_decode_view(const uint8_t*src,size_t n,uint8_t*out,size_t cap){return ng_packed_bank_view_impl(src,n,out,cap,nullptr);}
extern "C" int ng_packed_bank_profile_view(const uint8_t*src,size_t n,uint8_t*out,size_t cap,double*times){if(!times)return -1;return ng_packed_bank_view_impl(src,n,out,cap,times);}
extern "C" int ng_packed_bank_decode_reuse(const uint8_t*src,size_t n,uint8_t*out,size_t cap){return ng_packed_bank_view_impl(src,n,out,cap,nullptr,true);}
extern "C" int ng_packed_bank_profile_reuse(const uint8_t*src,size_t n,uint8_t*out,size_t cap,double*times){if(!times)return -1;return ng_packed_bank_view_impl(src,n,out,cap,times,true);}
extern "C" int ng_original_fast_decode(const void*,size_t,uint8_t*,size_t,unsigned);
/* Explicit optional size/speed tradeoff. The normal encoder remains unchanged.
 * NWP0 stores a dimension followed by a complete original-NG leaf; NWP1 is
 * retained when plain NG exceeds the caller's relative byte budget. */
extern "C" int ng_speed_bank_select(const uint8_t*bank,size_t bn,const uint8_t*ng,size_t nn,unsigned d,unsigned permille,uint8_t*out,size_t cap,size_t*written){
    try{
        require(bank&&ng&&out&&written&&ng_geometry_valid(d,d,d)&&permille<=100&&bn>=12&&nn>0);
        require(packed_bank::is_wave(bank,bn)&&volcodec_ng_lib_is_chunk(ng,nn));
        require(bn<=SIZE_MAX/1100&&nn<=SIZE_MAX/1000-8);
        if((nn+8)*1000<=bn*(1000+permille)){
            require(nn+8<=cap);memcpy(out,"NWP0",4);
            for(unsigned j=0;j<4;j++)out[4+j]=uint8_t(d>>(8*j));
            memcpy(out+8,ng,nn);*written=nn+8;
        }else{require(bn<=cap);memcpy(out,bank,bn);*written=bn;}
        return 0;
    }catch(...){return -1;}
}
extern "C" int ng_speed_bank_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    if(!src||!out||n<8)return -1;
    if(!memcmp(src,"NWP0",4)){
        unsigned d=0;for(unsigned j=0;j<4;j++)d|=unsigned(src[4+j])<<(8*j);
        return ng_original_fast_decode(src+8,n-8,out,cap,d);
    }
    return ng_packed_bank_decode_reuse(src,n,out,cap);
}
extern "C" int ng_speed_bank_encode(const uint8_t*src,unsigned d,unsigned permille,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&out&&written&&ng_geometry_valid(d,d,d)&&permille<=100);
        GeometryScope scope(d,d,d);Bytes bank(3*N+131072),ng(3*N+131072);size_t bn=0,nn=0;
        require(ng_packed_bank_encode(src,d,3,bank.data(),bank.size(),&bn)==0);
        require(volcodec_ng_lib_encode(src,ng.data(),ng.size(),&nn)==0);
        return ng_speed_bank_select(bank.data(),bn,ng.data(),nn,d,permille,out,cap,written);
    }catch(...){return -1;}
}
extern "C" int ng_packed_bank_inspect(const uint8_t*src,size_t n,uint64_t*info){
    try{require(src&&info);std::fill(info,info+28,0);bool wave=packed_bank::is_wave(src,n);auto f=wave?packed_bank::wave_unpack(Bytes(src,src+n)):packed_bank::table_unpack(Bytes(src,src+n));packed_bank::Reader r{f};auto magic=r.take(4);r.u(4);unsigned mode=r.u(4);auto selector=r.take(r.u(4));size_t start=r.p;unsigned mask=r.u(2);
        if(magic==Bytes({'N','P','K','1'})){r.u(1);r.u(4);r.skip(r.u(4));}
        else for(unsigned m=0;m<12;m++)if(mask&(1u<<m)){r.skip(r.u(4));if(mode&&m)r.skip(8);}
        info[0]=r.p-start;info[3]=24;
        auto native=wave?f:packed_bank::model_unpack(f);packed_bank::Reader nr{native};nr.skip(4);auto d=nr.u(4);nr.u(4);nr.skip(nr.u(4));nr.u(2);std::vector<Bytes>unique;
        for(unsigned m=0;m<12;m++)if(mask&(1u<<m)){info[1]++;auto params=nr.take(nr.u(4));if(std::find(unique.begin(),unique.end(),params)==unique.end())unique.push_back(params);if(mode&&m)nr.skip(8);}
        info[2]=unique.size();Bytes choices(size_t(d/4)*(d/4)*(d/4));unpack_stream(selector.data(),selector.size(),choices,true);for(auto c:choices){require(c<24);info[4+c]++;}return 0;
    }catch(...){return -1;}
}
