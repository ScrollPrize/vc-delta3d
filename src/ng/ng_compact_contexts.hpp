/* Shared entropy framing for independent leaf payloads. Context tables are
 * independent; rANS state sets are shared by pages of up to 256 unique tables.
 * Uses existing native normalization, compact tables and SIMD rANS routines. */
namespace compact_contexts {
using packed_bank::Reader;
using packed_bank::var;
using packed_bank::append;
void signed_var(Bytes&b,int64_t v){var(b,v<0?uint64_t(-(v+1))*2+1:uint64_t(v)*2);}
int64_t signed_var(Reader&r){auto v=r.var();return (v&1)?-int64_t(v/2)-1:int64_t(v/2);}
void write(Bytes&h,Bytes&body,const std::vector<Bytes>&streams){
    require(streams.size()<=4096);var(h,streams.size());
    std::vector<Bytes>tables;std::vector<std::array<uint16_t,256>>freqs;
    std::map<Bytes,unsigned>dictionary;std::vector<int>ids(streams.size(),-1);
    Bytes records;
    for(size_t i=0;i<streams.size();i++){
        const auto&v=streams[i];if(v.empty()){records.push_back(0);continue;}
        if(std::all_of(v.begin(),v.end(),[&](uint8_t c){return c==v[0];})){
            records.push_back(1);var(records,v.size());records.push_back(v[0]);continue;}
        uint64_t hist[256]={};for(auto c:v)hist[c]++;
        std::array<uint16_t,256>freq{};delta_rans_size_lower_bound(hist,v.size(),freq.data());
        uint8_t tmp[514];size_t n=ng_write_freq_table(tmp,freq.data());Bytes table(tmp,tmp+n);
        auto[it,inserted]=dictionary.emplace(table,tables.size());
        if(inserted){tables.push_back(table);freqs.push_back(freq);}
        ids[i]=it->second;records.push_back(2);var(records,v.size());var(records,it->second);
    }
    var(h,tables.size());for(auto&t:tables){var(h,t.size());append(h,t);}append(h,records);
    for(unsigned begin=0;begin<tables.size();begin+=256){
        unsigned count=std::min(256u,unsigned(tables.size())-begin);Bytes symbols,groups;
        for(size_t i=0;i<streams.size();i++)if(ids[i]>=int(begin)&&ids[i]<int(begin+count)){
            append(symbols,streams[i]);groups.insert(groups.end(),streams[i].size(),ids[i]-begin);}
        require(!symbols.empty());Bytes lanes(symbols.size());for(size_t i=0;i<lanes.size();i++)lanes[i]=i%32;
        std::vector<uint16_t>freq(count*256);for(unsigned j=0;j<count;j++)std::copy(freqs[begin+j].begin(),freqs[begin+j].end(),freq.begin()+j*256);
        Bytes payload(2*symbols.size()+128);size_t n=0;
        require(ng_context_rans_encode_fixed(symbols.data(),groups.data(),lanes.data(),symbols.size(),count,12,freq.data(),payload.data(),payload.size(),&n)==0);
        var(h,n);body.insert(body.end(),payload.begin(),payload.begin()+n);
    }
}
std::vector<Bytes>read(Reader&h,Reader&body,size_t expected,size_t limit){
    unsigned count=h.var();require(count==expected&&count<=4096);
    unsigned nt=h.var();require(nt<=count);std::vector<Bytes>tables;
    for(unsigned t=0;t<nt;t++){auto n=h.var();require(n<=514);auto b=h.take(n);uint16_t f[256];size_t used=0;
        require(ng_read_freq_table(b.data(),b.size(),f,&used)==0&&used==b.size());unsigned sum=0,active=0;for(auto v:f){sum+=v;active+=v!=0;}require(sum==4096&&active>=2);tables.push_back(std::move(b));}
    std::vector<Bytes>streams(count);std::vector<int>ids(count,-1);size_t total=0;
    for(unsigned i=0;i<count;i++){unsigned kind=h.u(1);require(kind<=2);if(!kind)continue;
        auto n=h.var();require(n&&n<=limit-total);total+=n;streams[i].resize(n);
        if(kind==1)std::fill(streams[i].begin(),streams[i].end(),h.u(1));
        else{auto id=h.var();require(id<nt);ids[i]=id;}}
    for(unsigned begin=0;begin<nt;begin+=256){
        unsigned k=std::min(256u,nt-begin);Bytes table,groups;
        for(unsigned j=0;j<k;j++)append(table,tables[begin+j]);
        for(unsigned i=0;i<count;i++)if(ids[i]>=int(begin)&&ids[i]<int(begin+k))groups.insert(groups.end(),streams[i].size(),ids[i]-begin);
        require(!groups.empty());auto n=h.var();require(n>=128&&n<=2*groups.size()+128);auto payload=body.take(n);
        std::unique_ptr<void,decltype(&ng_context_rans_destroy)>state(ng_context_rans_create(table.data(),table.size(),k,12,payload.data(),payload.size()),ng_context_rans_destroy);require(bool(state));
        Bytes values(groups.size());unsigned full=groups.size()/32*32;
        if(full)require(ng_context_rans_read_rows(state.get(),full,groups.data(),values.data())==0);
        if(full<groups.size())require(ng_context_rans_read(state.get(),groups.size()-full,groups.data()+full,values.data()+full,1)==0);
        require(ng_context_rans_finish(state.get())==0);size_t at=0;
        for(unsigned i=0;i<count;i++)if(ids[i]>=int(begin)&&ids[i]<int(begin+k)){std::copy_n(values.begin()+at,streams[i].size(),streams[i].begin());at+=streams[i].size();}
        require(at==values.size());
    }
    return streams;
}
}
