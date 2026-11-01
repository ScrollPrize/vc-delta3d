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
        for(unsigned z=0;z<d;z+=e)for(unsigned y=0;y<d;y+=e)for(unsigned x=0;x<d;x+=e){
            uint32_t bits=0;
            for(unsigned dz=0;dz<e;dz++)for(unsigned dy=0;dy<e;dy++){
                if(e==4){uint32_t v;memcpy(&v,src+(size_t(z+dz)*d+y+dy)*d+x,4);bits|=v;}
                else for(unsigned dx=0;dx<e;dx++)bits|=src[(size_t(z+dz)*d+y+dy)*d+x+dx];
            }
            size_t b=(size_t(z/e)*nb+y/e)*nb+x/e;empty[b]=bits==0;count+=empty[b];
        }
        if(!count){empty.clear();return;}
        mask.resize(size_t(d)*d*d);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x+=e)
            std::fill_n(mask.data()+(size_t(z)*d+y)*d+x,e,!empty[(size_t(z/e)*nb+y/e)*nb+x/e]);
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
