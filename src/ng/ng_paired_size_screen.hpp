// Incremental, distributed paired samples. A sample is exactly the larger
// block: compare it with its eight children, never a larger proxy cube.
namespace paired_size {
thread_local int sparse_halo=-1;
thread_local size_t last_work=0;
constexpr unsigned M=12,K=4,F=10;
struct State {
    const uint8_t*src;unsigned d,preset,side,count=0;
    std::vector<unsigned> order;
    std::vector<Bytes> symbols{M};Bytes classes,empty;
    std::vector<std::array<double,F>> raw;
    std::array<std::array<uint32_t,256>,M*K> hist{};
    bool fit_aware=false;void*probe=nullptr;std::vector<uint32_t> positions;
    ~State(){ng_fit_probe_destroy(probe);}
    State(const uint8_t*s,unsigned dim,unsigned p,unsigned edge,unsigned seed):src(s),d(dim),preset(p),side(edge){
        require(src&&ng_geometry_valid(d,d,d)&&(p==1||p==3)&&(edge==4||edge==8||edge==16||edge==32)&&d%edge==0);
        unsigned nr=d/side;std::array<std::vector<unsigned>,8> strata;
        for(unsigned z=0;z<nr;z++)for(unsigned y=0;y<nr;y++)for(unsigned x=0;x<nr;x++)
            strata[(x>=nr/2)+2*(y>=nr/2)+4*(z>=nr/2)].push_back((z*nr+y)*nr+x);
        uint32_t rng=seed?seed:1;auto draw=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
        for(auto&v:strata)for(size_t i=v.size();i>1;i--)std::swap(v[i-1],v[draw()%i]);
        std::array<unsigned,8> ids{0,1,2,3,4,5,6,7};
        for(unsigned i=8;i>1;i--)std::swap(ids[i-1],ids[draw()%i]);
        for(size_t j=0;order.size()<size_t(nr)*nr*nr;j++)for(auto id:ids)if(j<strata[id].size())order.push_back(strata[id][j]);
    }
    void extend(unsigned n){
        require(n>=count&&n<=order.size());unsigned nr=d/side,half=side/2;size_t V=size_t(side)*side*side;
        for(auto&v:symbols)v.resize(n*V);classes.resize(n*V);empty.resize(n*8,1);raw.resize(n);positions.resize(n*V);
        for(unsigned q=count;q<n;q++){
            unsigned r=order[q],ox=r%nr*side,oy=r/nr%nr*side,oz=r/(nr*nr)*side;
            for(unsigned z=0;z<side;z++)for(unsigned y=0;y<side;y++)for(unsigned x=0;x<side;x++){
                unsigned xx=ox+x,yy=oy+y,zz=oz+z;size_t i=(size_t(zz)*d+yy)*d+xx,j=q*V+(z*side+y)*side+x;
                positions[j]=i;
                int v=src[i],a=xx?src[i-1]:v,b=yy?src[i-d]:v,c=zz?src[i-size_t(d)*d]:v;
                unsigned cl=std::min(3,(std::abs(a-b)+std::abs(a-c)+std::abs(b-c))/16);classes[j]=cl;
                if(v)empty[q*8+(x>=half)+2*(y>=half)+4*(z>=half)]=0;
                raw[q][4]+=double(v)/255/V;raw[q][5]+=double(v==0)/V;
                raw[q][6]+=double(std::abs(v-a)+std::abs(v-b)+std::abs(v-c))/(3*255*V);
                raw[q][7]+=double(v)*v/(255.*255*V);
                for(unsigned m=0;m<M;m++){
                    auto s=pyramid(uint8_t(v-ng_directional_base(src,i,m,xx,yy,zz,d)));
                    symbols[m][j]=s;hist[m*K+cl][s]++;
                }
            }
        }count=n;
    }
    void evidence(double*out){
        require(count>=2);size_t V=size_t(side)*side*side;
        std::array<std::array<double,256>,M*K> bits;
        for(unsigned k=0;k<M*K;k++){
            double total=128;for(auto n:hist[k])total+=n;
            for(unsigned s=0;s<256;s++)bits[k][s]=std::log2(total/(hist[k][s]+.5));
        }
        using Costs=std::array<double,M>;std::vector<Costs> sums(count*8);Costs totals{};
        unsigned half=side/2;
        for(unsigned q=0;q<count;q++)for(unsigned z=0;z<side;z++)for(unsigned y=0;y<side;y++)for(unsigned x=0;x<side;x++){
            size_t i=q*V+(z*side+y)*side+x,b=q*8+(x>=half)+2*(y>=half)+4*(z>=half);
            for(unsigned m=0;m<M;m++)sums[b][m]+=bits[m*K+classes[i]][symbols[m][i]];
        }
        for(size_t b=0;b<sums.size();b++)if(!empty[b])for(unsigned m=0;m<M;m++)totals[m]+=sums[b][m];
        std::array<unsigned,M> rank;std::iota(rank.begin(),rank.end(),0);
        std::stable_sort(rank.begin(),rank.end(),[&](unsigned a,unsigned b){return totals[a]<totals[b];});
        // Fit the same native NG model on the two candidate zero masks. Keep
        // prediction on original-volume neighbors, never on stitched samples.
        std::array<std::vector<double>,2> ngcost;
        std::array<bool,2> prefer_ng{};
        if(fit_aware){
            GeometryScope scope(d,d,d);if(!probe)probe=ng_fit_probe_create(src);require(probe);
            for(unsigned level=0;level<2;level++){
                Bytes active(count*V),pred(count*V);
                for(unsigned q=0;q<count;q++){
                    bool all=true;for(unsigned k=0;k<8;k++)all&=empty[q*8+k];
                    for(unsigned z=0;z<side;z++)for(unsigned y=0;y<side;y++)for(unsigned x=0;x<side;x++){
                        size_t i=q*V+(z*side+y)*side+x,b=q*8+(x>=half)+2*(y>=half)+4*(z>=half);
                        active[i]=!(level?all:empty[b]);
                    }
                }
                require(ng_fit_probe_samples(probe,rank[0],positions.data(),active.data(),active.size(),4096,pred.data())==0);
                std::array<std::array<uint32_t,256>,K> h{};
                for(size_t i=0;i<pred.size();i++)if(active[i])h[classes[i]][pred[i]]++;
                std::array<std::array<double,256>,K> cost{};
                for(unsigned c=0;c<K;c++){double total=128;for(auto v:h[c])total+=v;for(unsigned v=0;v<256;v++)cost[c][v]=std::log2(total/(h[c][v]+.5));}
                ngcost[level].resize(count*8);double total_ng=0,total_bare=0;
                for(unsigned q=0;q<count;q++)for(unsigned z=0;z<side;z++)for(unsigned y=0;y<side;y++)for(unsigned x=0;x<side;x++){
                    size_t i=q*V+(z*side+y)*side+x,b=q*8+(x>=half)+2*(y>=half)+4*(z>=half);
                    ngcost[level][b]+=cost[classes[i]][pred[i]];
                    if(active[i]){total_ng+=cost[classes[i]][pred[i]];total_bare+=bits[rank[0]*K+classes[i]][symbols[rank[0]][i]];}
                }
                prefer_ng[level]=total_ng<total_bare;
            }
        }
        Bytes fine(count*8),coarse(count);std::array<unsigned,M+2> hf{},hc{};
        auto choose=[&](const Costs&v){unsigned m=rank[0];if(preset==1)for(unsigned k=1;k<3;k++)if(v[rank[k]]<v[m])m=rank[k];return m;};
        auto f=raw;
        for(unsigned q=0;q<count;q++){
            Costs parent{};bool all=true;double small=0,zeros=0;
            for(unsigned k=0;k<8;k++){
                auto&s=sums[q*8+k];for(unsigned m=0;m<M;m++)parent[m]+=s[m];
                unsigned c=empty[q*8+k]?M:choose(s);
                if(c<M&&fit_aware&&(preset==3?prefer_ng[0]:ngcost[0][q*8+k]<s[c]))c=M+1;
                fine[q*8+k]=c;hf[c]++;
                all&=empty[q*8+k];if(c<M)small+=s[c];else if(c==M+1)small+=ngcost[0][q*8+k];else zeros+=1./8;
            }
            unsigned c=all?M:choose(parent);double large=all?0:parent[c];
            if(!all&&fit_aware){double ng=0;for(unsigned k=0;k<8;k++)ng+=ngcost[1][q*8+k];if(preset==3?prefer_ng[1]:ng<large){c=M+1;large=ng;}}
            coarse[q]=c;hc[c]++;
            f[q][0]=(small-large)/(8*V); // small minus large residual bytes/voxel
            f[q][2]=zeros-double(all);f[q][3]=large/(8*V);
            double jumps=0;for(unsigned k=0;k<8;k++)for(unsigned axis=1;axis<=4;axis*=2)if(!(k&axis))jumps+=fine[q*8+k]!=fine[q*8+(k|axis)];
            f[q][8]=jumps/(8*V);f[q][9]=1;
        }
        for(unsigned q=0;q<count;q++){
            double sf=0;for(unsigned k=0;k<8;k++)sf+=-std::log2(double(hf[fine[q*8+k]])/(count*8));
            double sc=-std::log2(double(hc[coarse[q]])/count);
            f[q][1]=(sf-sc)/(8*V);
        }
        std::fill_n(out,F+F*F,0.);
        for(auto&v:f)for(unsigned j=0;j<F;j++)out[j]+=v[j]/count;
        // Covariance of the sample mean. Treat whole parent blocks, not
        // correlated voxels, as observations. This is a sampling diagnostic;
        // calibration error must be accounted for separately by the caller.
        for(auto&v:f)for(unsigned j=0;j<F;j++)for(unsigned k=0;k<F;k++)
            out[F+j*F+k]+=(v[j]-out[j])*(v[k]-out[k])/(count*(count-1.));
        double finite=double(order.size()-count)/(order.size()-1.);
        for(unsigned i=F;i<F+F*F;i++)out[i]*=finite;
    }
};
}
extern "C" void* ng_pair_size_create(const uint8_t*s,unsigned d,unsigned preset,unsigned larger,unsigned seed){
    try{return new paired_size::State(s,d,preset,larger,seed);}catch(...){return nullptr;}
}
extern "C" int ng_pair_size_extend(void*h,unsigned count,double*out){
    try{require(h&&out);auto&s=*static_cast<paired_size::State*>(h);s.extend(count);s.evidence(out);return 0;}catch(...){return -1;}
}
extern "C" void* ng_pair_size_create_fit(const uint8_t*s,unsigned d,unsigned preset,unsigned larger,unsigned seed){
    try{auto*p=new paired_size::State(s,d,preset,larger,seed);p->fit_aware=true;return p;}catch(...){return nullptr;}
}
extern "C" void ng_pair_size_destroy(void*h){delete static_cast<paired_size::State*>(h);}
// Diagnostic subset payload, not a standalone source-volume frame: decoding
// predictions requires original context outside the selected target blocks.
// Native wavefront/lane order is retained. Output: estimated full bytes,
// actual sampled frame bytes, target count, full header bytes, state bytes.
extern "C" int ng_pair_subset_score(const uint8_t*src,unsigned d,unsigned preset,unsigned larger,unsigned count,unsigned edge,unsigned seed,unsigned sample_fit,uint64_t*out){
    struct Restore{unsigned edge=choice_grid::edge();~Restore(){ng_screen_work=nullptr;ng_screen_targets=nullptr;ng_screen_size=0;subset_targets::active.clear();subset_targets::source=nullptr;subset_targets::fit_sample=false;ng_codec_choice_edge(edge);}} restore;
    try{
        require(out&&(edge==larger||edge==larger/2));paired_size::State plan(src,d,preset,larger,seed);
        require(count&&count<=plan.order.size());unsigned nr=d/larger;
        {
            subset_targets::active.assign(size_t(d)*d*d,0);
            for(unsigned q=0;q<count;q++){
                unsigned b=plan.order[q],ox=b%nr*larger,oy=b/nr%nr*larger,oz=b/(nr*nr)*larger;
                for(unsigned z=0;z<larger;z++)for(unsigned y=0;y<larger;y++)
                    std::fill_n(subset_targets::active.data()+(size_t(oz+z)*d+oy+y)*d+ox,larger,1);
            }
        }
        subset_targets::source=src;subset_targets::fit_sample=sample_fit!=0;
        Bytes work;paired_size::last_work=size_t(d)*d*d;
        if(paired_size::sparse_halo>=0){
            work.resize(size_t(d)*d*d);unsigned h=paired_size::sparse_halo;
            for(unsigned q=0;q<count;q++){
                unsigned b=plan.order[q];unsigned o[3]={b%nr*larger,b/nr%nr*larger,b/(nr*nr)*larger};
                unsigned lo[3],hi[3];for(unsigned j=0;j<3;j++){
                    lo[j]=o[j]>h?o[j]-h:0;
                    hi[j]=std::min(d,o[j]+larger+h);
                }
                for(unsigned z=lo[2];z<hi[2];z++)for(unsigned y=lo[1];y<hi[1];y++)
                    std::fill(work.begin()+(size_t(z)*d+y)*d+lo[0],work.begin()+(size_t(z)*d+y)*d+hi[0],1);
            }
            paired_size::last_work=std::count(work.begin(),work.end(),uint8_t(1));
            ng_screen_work=work.data();ng_screen_targets=subset_targets::active.data();ng_screen_size=work.size();
        }
        ng_codec_choice_edge(edge);Bytes frame(size_t(d)*d*d*4+131072);size_t n=0;
        require(ng_codec_encode(src,d,preset,frame.data(),frame.size(),&n)==0);
        require(n>=12);size_t header=12+uint32_t(frame[8])+(uint32_t(frame[9])<<8)+(uint32_t(frame[10])<<16)+(uint32_t(frame[11])<<24);
        require(header<=n);size_t states=header<n?ng_rans_state_bytes():0;require(header+states<=n);
        out[2]=size_t(count)*larger*larger*larger;
        out[0]=header+states+((n-header-states)*size_t(d)*d*d+out[2]-1)/out[2];
        out[1]=n;out[3]=header;out[4]=states;
        if(count==plan.order.size()){
            ng_screen_work=nullptr;ng_screen_targets=nullptr;ng_screen_size=0;
            subset_targets::active.clear();subset_targets::fit_sample=false;Bytes control(frame.size());size_t cn=0;
            require(ng_codec_encode(src,d,preset,control.data(),control.size(),&cn)==0);
            require(cn==n&&std::equal(control.begin(),control.begin()+cn,frame.begin()));
        }
        return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"subset encode: %s\n",e.what());return -1;}
}
extern "C" int ng_pair_sparse_score(const uint8_t*s,unsigned d,unsigned preset,unsigned larger,unsigned count,unsigned edge,unsigned seed,unsigned halo,unsigned verify,uint64_t*out){
    if(halo>d||!out)return -1;
    struct Scope{int h=paired_size::sparse_halo;bool v=subset_targets::verify;~Scope(){paired_size::sparse_halo=h;subset_targets::verify=v;}}scope;
    paired_size::sparse_halo=halo;subset_targets::verify=verify!=0;
    int rc=ng_pair_subset_score(s,d,preset,larger,count,edge,seed,1,out);
    if(!rc)out[5]=paired_size::last_work;return rc;
}
// Per-input lifetime. Reuse the existing exact-key fit/prepared-history cache;
// no approximate matching, cross-volume state, or production default change.
extern "C" int ng_pair_context_begin(){
    if(small_bank::memo::enabled)return -1;
    small_bank::memo::entries.clear();small_bank::memo::used=small_bank::memo::hits=small_bank::memo::misses=0;
    small_bank::memo::enabled=true;return 0;
}
// Never export estimated placeholder bytes; production encode remains untouched.
extern "C" int ng_full_size_estimate(const uint8_t*s,unsigned d,unsigned preset,unsigned edge,uint64_t*out){
    unsigned old=choice_frame::estimate_dimension;
    struct Reset {unsigned old;~Reset(){choice_frame::estimate_dimension=old;}} reset{old};
    try{
        choice_grid::Scope grid(edge);choice_frame::estimate_dimension=d;
        Bytes frame(size_t(d)*d*d*4+131072);size_t n=0;
        require(ng_codec_encode(s,d,preset,frame.data(),frame.size(),&n)==0);*out=n;return 0;
    }catch(...){return -1;}
}
extern "C" void ng_size_prepared_begin(){choice_frame::prepared.clear();choice_frame::retain_estimates=true;}
extern "C" void ng_size_prepared_end(){choice_frame::retain_estimates=false;choice_frame::prepared.clear();}
extern "C" int ng_size_prepared_finish(unsigned edge,uint8_t*out,size_t cap,size_t*n){
    try{
        auto&p=choice_frame::prepared.at(edge);GeometryScope geom(p.d,p.d,p.d);
        choice_grid::Scope grid(edge);adaptive_grid::Scope adaptive(0);
        Bytes result=p.complete;
        if(result.empty()){
            Bytes entropy(size_t(p.d)*p.d*p.d*3+131072);size_t en=0,hn=0;
            require(ng_wave_encode(p.params.data(),p.params.size(),p.symbols.data(),p.classes.data(),entropy.data(),entropy.size(),&en)==0);
            require(ng_entropy_header_size(entropy.data(),en,&hn)==0);
            result=choice_frame::finish_entropy(p.header,entropy,en,hn);
        }
        require(result.size()<=cap);std::copy(result.begin(),result.end(),out);*n=result.size();return 0;
    }catch(...){return -1;}
}
extern "C" void ng_pair_context_end(uint64_t*out){
    if(out){out[0]=small_bank::memo::hits;out[1]=small_bank::memo::misses;out[2]=small_bank::memo::used;}
    small_bank::memo::enabled=false;small_bank::memo::entries.clear();small_bank::memo::used=0;
}
extern "C" int ng_pair_context_score(const uint8_t*s,unsigned d,unsigned preset,unsigned larger,unsigned count,unsigned edge,unsigned seed,uint64_t*out){
    struct Scope{bool v=subset_targets::verify;~Scope(){subset_targets::verify=v;}}scope;
    subset_targets::verify=false;
    return ng_pair_subset_score(s,d,preset,larger,count,edge,seed,0,out);
}
namespace size_search {thread_local unsigned selected=4;}
extern "C" int ng_rc_encode_fixed(const uint8_t*,unsigned,unsigned,uint8_t*,size_t,size_t*);
extern "C" unsigned ng_size_search_choice(){return size_search::selected;}
// Experimental Fast-Size/Slow-Size. Flags:1 span costs,2 distribution reuse,
// 4 exact fitted-model reuse. Format and fixed-size preset IDs are unchanged.
extern "C" int ng_codec_encode_size_search(const uint8_t*src,unsigned d,unsigned preset,unsigned flags,uint8_t*out,size_t cap,size_t*n){
    if(!src||!out||!n||(preset!=1&&preset!=3&&(preset<8||preset>10))||flags>7||!ng_geometry_valid(d,d,d)||
       small_bank::memo::enabled||three_candidate::distribution_cache::enabled||
       choice_frame::estimate_dimension||subset_targets::enabled())return -1;
    struct Reset {
        unsigned flags;bool old=three_candidate::span_selection;
        ~Reset(){three_candidate::span_selection=old;if(flags&2)ng_three_distribution_cache(0);if(flags&4)ng_pair_context_end(nullptr);}
    }reset{flags};
    try{
        choice_grid::Scope grid(4);adaptive_grid::Scope adaptive(0);
        three_candidate::span_selection=(flags&1)!=0;
        if(flags&2)ng_three_distribution_cache(1);
        if(flags&4)require(ng_pair_context_begin()==0);
        Bytes buffer(size_t(d)*d*d*4+131072),winner;unsigned chosen=4;
        auto trial=[&](unsigned edge){
            ng_codec_choice_edge(edge);size_t bytes=0;
            require((preset>=8?ng_rc_encode_fixed(src,d,preset-8,buffer.data(),buffer.size(),&bytes):ng_codec_encode(src,d,preset,buffer.data(),buffer.size(),&bytes))==0);
            if(winner.empty()||bytes<winner.size()){winner.assign(buffer.begin(),buffer.begin()+bytes);chosen=edge;}
            return bytes;
        };
        size_t s4=trial(4),s8=trial(8);
        if(s4<=s8)trial(2);
        else {size_t previous=s8;for(unsigned edge:{16u,32u}){size_t bytes=trial(edge);if(bytes>=previous)break;previous=bytes;}}
        require(winner.size()<=cap);std::copy(winner.begin(),winner.end(),out);*n=winner.size();size_search::selected=chosen;return 0;
    }catch(...){return -1;}
}
extern "C" int ng_codec_encode_legacy_slow_size(const uint8_t*s,unsigned d,uint8_t*out,size_t cap,size_t*n){
    three_candidate::LegacySlowScope legacy;return ng_codec_encode_size_search(s,d,1,3,out,cap,n);
}
// S2 exact-size research. Start from the historical Slow-Size winner, reproduce
// its fit/choices byte-for-byte, and retain it through all adaptive trials.
extern "C" int ng_codec_encode_exact_partition(const uint8_t*src,unsigned d,unsigned budget,unsigned sweeps,unsigned contexts,uint8_t*out,size_t cap,size_t*n){
    three_candidate::LegacySlowScope legacy;
    if(!src||!out||!n||d<32||d>128||d%32||budget>1024||!sweeps||sweeps>8||contexts>3||exact_partition::enabled||!choice_frame::context_map.empty())return -1;
    struct Reset{
        bool enabled=choice_frame::enabled,span=three_candidate::span_selection;
        unsigned root=adaptive_grid::s2_root,opts=adaptive_grid::s2_options,minimum=adaptive_grid::s2_min;
        ~Reset(){choice_frame::enabled=enabled;three_candidate::span_selection=span;adaptive_grid::s2_root=root;adaptive_grid::s2_options=opts;adaptive_grid::s2_min=minimum;exact_partition::enabled=false;exact_partition::reference.clear();choice_frame::context_map.clear();}
    }reset;
    try{
        adaptive_grid::s2_root=adaptive_grid::s2_options=0;choice_grid::Scope grid(4);adaptive_grid::Scope tree(0);
        require(ng_codec_encode_size_search(src,d,1,3,out,cap,n)==0);
        exact_partition::reference.assign(out,out+*n);unsigned edge=size_search::selected;
        std::fill_n(exact_partition::stats,10,0);exact_partition::stats[0]=*n;exact_partition::stats[2]=edge;
        exact_partition::accepted={*n};exact_partition::enabled=true;exact_partition::budget=budget;exact_partition::sweeps=sweeps;exact_partition::contexts=contexts&1;exact_partition::coordinated=contexts&2;
        choice_frame::enabled=true;three_candidate::span_selection=true;adaptive_grid::s2_root=32;adaptive_grid::s2_min=2;ng_choice_edge=edge;
        require(three_candidate_encode_impl(src,d,1,7,1,0,0,out,cap,n)==0);
        require(*n<=exact_partition::reference.size());exact_partition::stats[1]=*n;
        return 0;
    }catch(...){return -1;}
}
extern "C" void ng_exact_partition_stats(uint64_t*out){std::copy_n(exact_partition::stats,10,out);}
extern "C" unsigned ng_exact_partition_history(uint64_t*out,unsigned cap){unsigned n=exact_partition::accepted.size();if(out)std::copy_n(exact_partition::accepted.begin(),std::min(n,cap),out);return n;}
