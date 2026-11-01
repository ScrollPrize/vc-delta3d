// Experimental indexed NX/NY container. Reuses the existing leaf decoders.
// No global pool, no hidden threads: requested worker count includes the caller.
#include "ng_leaf.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
extern "C" int ng_lifting_trial_decode(const uint8_t*,size_t,uint8_t*,uint8_t*,size_t,int,uint64_t*);
namespace {
void require(bool b){if(!b)throw std::runtime_error("invalid JSC1");}
uint32_t read(const uint8_t*s,size_t n,size_t&p){require(p<=n&&n-p>=4);uint32_t v=0;for(int k=0;k<4;k++)v|=uint32_t(s[p++])<<(8*k);return v;}
struct Leaf{uint32_t kind,a,b;size_t offset;};
}
extern "C" int ng_joint_subchunks_decode(const uint8_t*s,size_t n,uint8_t*x,uint8_t*y,size_t cap,int workers){
    try{
        require(s&&x&&y&&n>=16&&!std::memcmp(s,"JSC1",4)&&workers>=1&&workers<=128);
        size_t p=4;unsigned d=read(s,n,p),b=read(s,n,p),count=read(s,n,p);
        require(d>=32&&d<=256&&d%32==0&&b>=32&&b<=d&&d%b==0);
        unsigned side=d/b;require(count==side*side*side&&cap>=size_t(d)*d*d);
        std::vector<Leaf> leaves(count);for(auto&l:leaves){l.kind=read(s,n,p);l.a=read(s,n,p);l.b=read(s,n,p);require(l.kind<=1&&l.a>0&&(l.kind?l.b==0:l.b>0));}
        for(auto&l:leaves){l.offset=p;require(l.a<=n-p);p+=l.a;require(l.b<=n-p);p+=l.b;
            // Reject shape mismatches before allowing a leaf to touch output.
            size_t q=l.offset+4;require(read(s,n,q)==b);
            if(!l.kind){require(read(s,n,q)==b&&read(s,n,q)==b);q=l.offset+l.a+4;require(read(s,n,q)==b&&read(s,n,q)==b&&read(s,n,q)==b);}}
        require(p==n);std::atomic<unsigned> next{0};std::atomic<bool> failed{false};
        auto work=[&](){try{
            std::vector<uint8_t> a(size_t(b)*b*b),c(a.size());
            for(unsigned k;(k=next.fetch_add(1))<count&&!failed.load();){const auto&l=leaves[k];
                int rc=l.kind?ng_lifting_trial_decode(s+l.offset,l.a,a.data(),c.data(),a.size(),25,nullptr):
                    ng_leaf_decode(s+l.offset,l.a,a.data(),a.size(),0)|ng_leaf_decode(s+l.offset+l.a,l.b,c.data(),c.size(),0);
                require(rc==0);unsigned bx=(k%side)*b,by=(k/side%side)*b,bz=(k/side/side)*b;
                for(unsigned z=0;z<b;z++)for(unsigned yy=0;yy<b;yy++){
                    size_t out=(size_t(bz+z)*d+by+yy)*d+bx,in=(size_t(z)*b+yy)*b;
                    std::memcpy(x+out,a.data()+in,b);std::memcpy(y+out,c.data()+in,b);}
            }
        }catch(...){failed=true;}};
        // Thread creation/join is deliberately part of the measured API cost.
        std::vector<std::thread> threads;
        try{for(int k=1;k<std::min<int>(workers,count);k++)threads.emplace_back(work);}
        catch(...){failed=true;for(auto&t:threads)t.join();throw;}
        work();for(auto&t:threads)t.join();require(!failed);return 0;
    }catch(...){return -1;}
}
