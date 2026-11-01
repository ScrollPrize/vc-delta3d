/* Encoder-only incremental search. NMB1 framing and decoder are unchanged. */
namespace small_bank {
// Shared deterministic residual/gradient split used by exhaustive and bounded
// expert searches. Membership only seeds fits; later assignments remain free.
template<class Prediction>
bool correlation_split(const uint8_t*src,unsigned d,const std::vector<uint32_t>&block,
                       const std::vector<size_t>&ids,Prediction prediction,Bytes&group){
    if(ids.size()<32)return false;size_t blocks=N/64;
    Bytes member(blocks);for(auto b:ids)member[b]=1;
    std::vector<std::array<double,3>>desc(blocks),den(blocks);std::vector<double>energy(blocks);
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
        size_t i=(size_t(z)*d+y)*d+x,b=block[i];if(!member[b])continue;
        uint16_t q=prediction(i,b);int e=int(src[i])-int(uint8_t(uint16_t(q+128u)>>8));e=(e+128)&255;e-=128;
        const int g[3]={x?int(src[i])-src[i-1]:0,y?int(src[i])-src[i-d]:0,z?int(src[i])-src[i-size_t(d)*d]:0};
        energy[b]+=double(e)*e;for(unsigned j=0;j<3;j++){desc[b][j]+=double(e)*g[j];den[b][j]+=double(g[j])*g[j];}
    }
    for(auto b:ids)for(unsigned j=0;j<3;j++)desc[b][j]/=std::sqrt(1.+energy[b]*den[b][j]);
    auto dist=[](const std::array<double,3>&a,const std::array<double,3>&b){double v=0;for(unsigned j=0;j<3;j++)v+=(a[j]-b[j])*(a[j]-b[j]);return v;};
    size_t a=ids[0],b=a;for(auto i:ids)if(dist(desc[i],desc[a])>dist(desc[b],desc[a]))b=i;
    a=b;for(auto i:ids)if(dist(desc[i],desc[a])>dist(desc[b],desc[a]))b=i;
    if(dist(desc[a],desc[b])<1e-12)return false;
    std::array<double,3>centers[2]={desc[a],desc[b]};group.assign(blocks,0);size_t counts[2]={};
    for(unsigned it=0;it<8;it++){
        std::array<double,3>sum[2]={};counts[0]=counts[1]=0;
        for(auto i:ids){unsigned k=dist(desc[i],centers[1])<dist(desc[i],centers[0]);group[i]=k;counts[k]++;for(unsigned j=0;j<3;j++)sum[k][j]+=desc[i][j];}
        if(!counts[0]||!counts[1])return false;
        for(unsigned k=0;k<2;k++)for(unsigned j=0;j<3;j++)centers[k][j]=sum[k][j]/counts[k];
    }return counts[0]>=16&&counts[1]>=16;
}
struct SearchState {
    std::vector<std::shared_ptr<const Fit>> fits;
    Bytes sel,encoded;
};
struct Incremental {
    const uint8_t*src;unsigned d,nb;size_t blocks;
    const int cfg[6]={3,3,1,16,40,128};
    std::vector<Bytes> candidate;
    std::vector<uint32_t> block;
    Bytes symbols,raw,classes,feature,mask;
    std::vector<uint16_t>activity;
    unsigned evaluations=0,refits=0,proposals=0,merges=0;
    Incremental(const uint8_t*s,unsigned dim):src(s),d(dim),nb(d/4),blocks(N/64),
        candidate(60,Bytes(N)),block(N),symbols(N),raw(N),classes(N),feature(N),mask(N),activity(N){
        bank_fixed_candidates(src,candidate,d);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++)
            block[(size_t(z)*d+y)*d+x]=((z/4)*nb+y/4)*nb+x/4;
    }
    void evaluate(SearchState&s){
        const unsigned count=s.fits.size()*12,P=12+count;
        std::vector<const uint8_t*>params(count);std::vector<size_t>sizes(count);
        std::vector<const uint16_t*>base(count);std::vector<uint8_t*>out(P);
        for(unsigned m=0;m<count;m++){
            auto&f=*s.fits[m/12];params[m]=f.params.data();sizes[m]=f.params.size();base[m]=f.base[m%12].data();
        }
        for(unsigned m=0;m<P;m++)out[m]=candidate[m].data();
        require(ng_bank_symbols(count,params.data(),sizes.data(),base.data(),s.sel.data(),src,out.data(),symbols.data(),activity.data())==0);
        for(size_t i=0;i<N;i++)raw[i]=unpyramid(symbols[i]);
        auto thr=activity_quantiles(activity,16),tuned=activity_refine(raw,activity,thr,block,s.sel,1);
        ActivityHistogram hist(raw,activity,true);if(hist.estimate(tuned)<hist.estimate(thr))thr=tuned;
        classes=activity_classify(activity,thr);
        // Unused models cost no bytes. Remapping changes no prediction/history.
        std::vector<unsigned>used(s.fits.size()),mapping(s.fits.size());
        for(auto c:s.sel)if(c>=12)used[(c-12)/12]++;
        std::vector<Bytes>models;std::vector<Weight>weights;
        for(unsigned k=0;k<s.fits.size();k++)if(used[k]){
            mapping[k]=models.size();models.push_back(s.fits[k]->params);weights.push_back(s.fits[k]->weights);
        }
        // NMB1 requires at least one model even for an all-fixed selection.
        if(models.empty()){models.push_back(s.fits[0]->params);weights.push_back(s.fits[0]->weights);}
        Bytes compact=s.sel;
        for(auto&c:compact)if(c>=12)c=12+12*mapping[(c-12)/12]+(c-12)%12;
        s.encoded=frame(d,models,weights,compact,raw,thr,classes);evaluations++;
    }
    SearchState select(SearchState s){
        evaluate(s);SearchState best=s;
        const unsigned P=12+12*s.fits.size();std::vector<const uint8_t*>ptr(P);
        for(unsigned m=0;m<P;m++)ptr[m]=candidate[m].data();
        // Recompute causal activity and entropy costs after every reassignment.
        for(unsigned pass=0;pass<6;pass++){
            auto before=s.sel;activity_reselect(symbols,classes,block,ptr,s.sel,1);
            if(s.sel==before)break;
            evaluate(s);if(s.encoded.size()<best.encoded.size())best=s;
        }
        return best;
    }
    void refit(SearchState&s,const std::vector<unsigned>&which){
        for(size_t i=0;i<N;i++){
            unsigned c=s.sel[block[i]],m=c<12?c:(c-12)%12;
            feature[i]=uint8_t(src[i]-unpyramid(candidate[m][i]));
        }
        for(unsigned k:which){
            size_t members=0;
            for(size_t i=0;i<N;i++){unsigned c=s.sel[block[i]];mask[i]=c>=12&&(c-12)/12==k;members+=mask[i];}
            // Do not replace a small model with an unrelated global fit.
            if(members<1024)continue;
            s.fits[k]=std::make_shared<Fit>(fit(src,d,cfg,feature,mask.data(),candidate));refits++;
        }
    }
    SearchState refine(SearchState s){
        SearchState best=select(std::move(s));
        for(unsigned round=0;round<6;round++){
            SearchState next=best;std::vector<unsigned>which(next.fits.size());
            for(unsigned k=0;k<which.size();k++)which[k]=k;
            refit(next,which);next=select(std::move(next));
            if(next.encoded.size()>=best.encoded.size())break;
            best=std::move(next);
        }
        return best;
    }
    SearchState initial(){
        // A global plain NG seed; all choices become dynamic immediately.
        std::fill(feature.begin(),feature.end(),0);
        SearchState s;s.fits.push_back(std::make_shared<Fit>(fit(src,d,cfg,feature,nullptr,candidate)));refits++;
        s.sel.assign(blocks,12);return refine(std::move(s));
    }
    bool split(SearchState&s,unsigned parent){
        std::vector<size_t>ids;for(size_t b=0;b<blocks;b++)if(s.sel[b]>=12&&(s.sel[b]-12)/12==parent)ids.push_back(b);
        if(ids.size()<32)return false;
        Bytes group;
        if(!correlation_split(src,d,block,ids,[&](size_t i,size_t b){return s.fits[parent]->base[(s.sel[b]-12)%12][i];},group))return false;
        unsigned child=s.fits.size();s.fits.push_back(s.fits[parent]);
        for(auto i:ids)if(group[i])s.sel[i]=12+12*child+(s.sel[i]-12)%12;
        refit(s,{parent,child});proposals++;return true;
    }
    SearchState grow(const SearchState&previous){
        SearchState best;bool found=false;
        for(unsigned k=0;k<previous.fits.size();k++){
            auto next=previous;if(!split(next,k))continue;
            next=refine(std::move(next));
            if(!found||next.encoded.size()<best.encoded.size()){best=std::move(next);found=true;}
        }
        if(!found){best=previous;best.fits.push_back(best.fits.back());best=refine(std::move(best));}
        // Trial merging near-identical models. All blocks can subsequently
        // migrate again; this is not a permanent assignment restriction.
        for(unsigned a=0;a<best.fits.size();a++)for(unsigned b=a+1;b<best.fits.size();b++){
            double diff=0;size_t n=0;
            for(size_t i=0;i<N;i+=97){int va=uint8_t(uint16_t(best.fits[a]->base[0][i]+128u)>>8),vb=uint8_t(uint16_t(best.fits[b]->base[0][i]+128u)>>8);int v=std::abs(va-vb);diff+=std::min(v,256-v);n++;}
            if(diff>n)continue;
            auto merged=best;bool changed=false;
            for(auto&c:merged.sel)if(c>=12&&(c-12)/12==b){c=12+12*a+(c-12)%12;changed=true;}
            if(!changed)continue;
            refit(merged,{a});merged.fits[b]=merged.fits[a];merged=refine(std::move(merged));merges++;
            if(merged.encoded.size()<best.encoded.size())best=std::move(merged);
        }
        return best;
    }
};
}
extern "C" int ng_incremental_bank_encode(const uint8_t*src,unsigned d,unsigned maximum,
        uint8_t*const*out,size_t cap,size_t*sizes,double*seconds,unsigned*stats){
    try{
        require(src&&out&&sizes&&seconds&&stats&&maximum>=1&&maximum<=4&&ng_geometry_valid(d,d,d));GeometryScope scope(d,d,d);
        const auto start=std::chrono::steady_clock::now();small_bank::Incremental search(src,d);
        auto state=search.initial();small_bank::SearchState best=state;
        for(unsigned k=0;k<maximum;k++){
            if(k){state=search.grow(best);if(state.encoded.size()<best.encoded.size())best=state;}
            // A rejected split still leaves room for another proposal at the
            // next budget without changing the smaller-bank retained frame.
            if(best.fits.size()<state.fits.size())best.fits.push_back(best.fits.back());
            require(out[k]&&state.encoded.size()<=cap);memcpy(out[k],state.encoded.data(),state.encoded.size());sizes[k]=state.encoded.size();
            seconds[k]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            stats[4*k]=search.evaluations;stats[4*k+1]=search.refits;stats[4*k+2]=search.proposals;stats[4*k+3]=search.merges;
        }
        return 0;
    }catch(...){return -1;}
}
