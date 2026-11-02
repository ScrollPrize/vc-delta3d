/* Shared 4³ occupancy pass. Runtime masks are never serialized separately:
 * zero is represented by an ordinary NCF1 choice. */
namespace subset_targets {
thread_local Bytes active;
thread_local const uint8_t* source=nullptr;
thread_local bool fit_sample=false;
thread_local bool verify=true;
bool enabled(){return !active.empty();}
bool applies(size_t n){return active.size()==n;}
const uint8_t* fit_mask(const uint8_t*src,const uint8_t*mask,Bytes&combined){
    if(!fit_sample||src!=source||!enabled())return mask;
    combined=active;if(mask)for(size_t i=0;i<combined.size();i++)combined[i]&=mask[i];
    // Preserve the ordinary fitting kernel when membership is unchanged.
    // In particular an all-ones sampling mask must not select a different
    // masked-regression implementation merely because a pointer is non-null.
    bool unchanged=true;for(size_t i=0;i<combined.size();i++)if(bool(combined[i])!=bool(mask?mask[i]:1)){unchanged=false;break;}
    if(unchanged)return mask;
    return combined.data();
}
}
namespace zero_blocks {
thread_local bool enabled=true;
// Predetermined zero choices have no predictor residuals to fit. Keep their
// decoded values available as causal inputs, but exclude their target samples.
thread_local bool mask_fits=true;
struct Scan {
    unsigned d;Bytes empty,mask;size_t count=0;
    Scan(const uint8_t*src,unsigned dimension,bool active):d(dimension){
        if(!active||!enabled)return;
        unsigned e=choice_grid::edge(),nb=d/e;empty.resize(size_t(nb)*nb*nb);
        count=vc_ng_zero_scan(src,d,e,empty.data());
        if(!count){empty.clear();return;}
        mask.resize(size_t(d)*d*d);
        vc_ng_zero_mask(d,e,empty.data(),mask.data());
    }
    bool any()const{return count!=0;}
    bool filtered()const{return any()||(ng_screen_targets&&ng_screen_size==size_t(d)*d*d);}
    bool all()const{return any()&&count==empty.size();}
    const uint8_t* fit_mask(bool screening=false)const{return mask.empty()||(!screening&&!mask_fits)?nullptr:mask.data();}
    template<class T> std::vector<T> take(const std::vector<T>&v)const{
        if(!filtered())return v;
        require(v.size()==size_t(d)*d*d);std::vector<T> a;a.reserve(v.size()-choice_grid::volume()*count);
        const bool sparse=ng_screen_targets&&ng_screen_size==v.size();
        for(size_t i=0;i<v.size();i++)if((mask.empty()||mask[i])&&(!sparse||ng_screen_targets[i]))a.push_back(v[i]);return a;
    }
};
}
