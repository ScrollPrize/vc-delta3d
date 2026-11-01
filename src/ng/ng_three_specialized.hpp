/* Bounded plain-NG specialization: one split seed, a small number of refits,
 * three freely selectable slots, and complete-byte retention. */
namespace three_specialized {thread_local uint64_t stats[16]={};}
extern "C" void ng_three_specialized_stats(uint64_t*out){std::copy_n(three_specialized::stats,16,out);}
extern "C" int ng_three_specialized_encode(const uint8_t*src,unsigned d,unsigned experts,unsigned rounds,uint8_t*out,size_t cap,size_t*written){
    try{
        require(src&&out&&written&&(experts==2||experts==3)&&rounds<=2&&ng_geometry_valid(d,d,d));GeometryScope scope(d,d,d);
        auto*st=three_specialized::stats;std::fill_n(st,16,0);st[2]=experts;st[3]=rounds;
        Bytes best(N*3+131072);size_t n=0;
        require(ng_three_candidate_encode_abc(src,d,3,best.data(),best.size(),&n)==0);best.resize(n);st[0]=n;
        const size_t voxels=N,blocks=N/64;unsigned nb=d/4;
        int cfg[6]={3,3,1,16,40,128};if(d<=32)cfg[5]=4;
        auto global=small_bank::plain_fit(src,cfg);st[5]++;
        std::vector<uint32_t>block(N);for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++)
            block[(size_t(z)*d+y)*d+x]=(size_t(z/4)*nb+y/4)*nb+x/4;
        Bytes members(blocks,0);
        for(unsigned child=1;child<experts;child++){
            size_t count[3]={};for(auto c:members)count[c]++;unsigned parent=std::max_element(count,count+child)-count;
            std::vector<size_t>ids;for(size_t b=0;b<blocks;b++)if(members[b]==parent)ids.push_back(b);Bytes group;
            if(!small_bank::correlation_split(src,d,block,ids,[&](size_t i,size_t){return global.base[0][i];},group))break;
            for(auto b:ids)if(group[b])members[b]=child;
        }
        unsigned bare=0;std::vector<Bytes>fixed(12);
        if(experts==2){
            auto proxy=fast_split::screen_proxy(src,d);unsigned sd=std::min(d,32u);
            {GeometryScope screen(sd,sd,sd);std::vector<Bytes>sample(12,Bytes(N));bank_fixed_candidates(proxy.data(),sample,sd);bare=fast_split::screen(sample,sd,12)[0];}
            fixed[bare].resize(N);bank_fixed_candidates(src,fixed,d,1u<<bare);
        }
        st[9]=bare;
        std::array<unsigned,3>menu={12,12,experts==3?12:bare};std::array<small_bank::Fit,3>fits;
        auto fit_members=[&](unsigned k,const Bytes&assignment,const small_bank::Fit&previous){
            Bytes mask(voxels);size_t count=0;for(size_t i=0;i<voxels;i++){mask[i]=assignment[block[i]]==k;count+=mask[i];}
            if(count<1024)return previous;
            int c[6];std::copy_n(cfg,6,c);c[5]=std::max(1u,unsigned(count/16384));st[5]++;
            return small_bank::plain_fit(src,c,mask.data());
        };
        for(unsigned k=0;k<experts;k++)fits[k]=fit_members(k,members,global);
        auto encode=[&](const std::array<small_bank::Fit,3>&f,Bytes&assignment){
            std::array<const small_bank::Fit*,3>ptr;for(unsigned k=0;k<3;k++)ptr[k]=k<experts?&f[k]:&global;
            return three_free::encode_slots(src,d,block,fixed,menu,ptr,&assignment,true,true);
        };
        auto current=encode(fits,members);st[6]=current.size();
        if(current.size()<best.size()){best=current;st[4]=1;}
        for(unsigned round=0;round<rounds;round++){
            std::array<small_bank::Fit,3>next;for(unsigned k=0;k<experts;k++)next[k]=fit_members(k,members,fits[k]);
            auto next_members=members;auto proposed=encode(next,next_members);st[7+round]=proposed.size();
            if(proposed.size()>=current.size())break;
            fits=std::move(next);members=std::move(next_members);current=std::move(proposed);
            if(current.size()<best.size()){best=current;st[4]=2+round;}
        }
        st[1]=best.size();require(best.size()<=cap);std::copy(best.begin(),best.end(),out);*written=best.size();return 0;
    }catch(...){return -1;}
}
extern "C" int ng_three_specialized_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    if(!src||n<4)return -1;
    return !memcmp(src,"N3C1",4)?ng_three_candidate_decode(src,n,out,cap):ng_three_free_decode(src,n,out,cap);
}
