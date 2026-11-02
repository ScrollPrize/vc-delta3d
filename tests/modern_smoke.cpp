#include <vc_delta3d/codec.hpp>
#include <vc_delta3d/indexed.hpp>
#include <cassert>
#include <iostream>
extern "C" int ng_set_isa_limit(int);
int main() {
    using namespace vc_delta3d;
    std::vector<std::byte> a(32*32*32);
    for(size_t i=0;i<a.size();++i)a[i]=std::byte((i*17+i/32)%256);
    for(auto p:{Predictor::Delta,Predictor::Fast,Predictor::Medium,
                Predictor::Slow,Predictor::Slower}){
        auto f=compress(a,{32,32,32},1,1,p);
        auto d=decompress(f,a.size());assert(d&&*d==a);
        ng_set_isa_limit(256);d=decompress(f,a.size());assert(d&&*d==a);ng_set_isa_limit(0);
        if(p>=Predictor::Fast)assert(f[0]==std::byte{'N'}||f[0]==std::byte{'H'});
        else assert(f[4]==std::byte{1});
        assert(!decompress(std::span(f).first(19),a.size()));
        assert(!decompress(f,a.size()-1));
        std::cout<<int(p)<<" "<<f.size()<<" bytes OK\n";
    }
    assert(compress(a,{32,32,32},1)[0]==std::byte{'N'});
    std::vector<std::byte> both=a;both.insert(both.end(),a.begin(),a.end());
    for(auto layout:{ChannelLayout::First,ChannelLayout::Last}){
        auto f=compressChannels(both,{32,32,32},2,layout,1);
        auto d=decompress(f,both.size());assert(d&&*d==both&&channelCount(f)==2);
        auto hn=indexedHeaderSize(f);assert(hn&&*hn==63);
        auto index=readIndex(std::span(f).first(*hn),f.size());assert(index.leaves.size()==2);
        for(const auto&leaf:index.leaves){
            std::vector<std::byte> out(a.size());
            auto payload=std::span(f).subspan(leaf.offset,leaf.length);
            ng_set_isa_limit(256);assert(decodeIndexedLeaf(payload,leaf,out));ng_set_isa_limit(0);
        }
    }
    auto f=compress(a,{16,32,64},1);assert(f[4]==std::byte{1});
    assert(f==compress(a,{16,32,64},1,1,Predictor::Delta));
    assert(decompress(f,a.size())==a);
    assert(compress(a,{32,32,32},1,3)==compress(a,{32,32,32},1,3,Predictor::Delta));
    assert(compress(both,{32,32,32},2)==compress(both,{32,32,32},2,1,Predictor::Delta));
    for(auto layout:{ChannelLayout::First,ChannelLayout::Last})
        assert(compressChannels(both,{16,32,64},2,layout,1)==
               compressChannels(both,{16,32,64},2,layout,1,1,Predictor::Delta));
    auto q=compress(a,{32,32,32},1,3);quantize(a,1,3);
    assert(decompress(q,a.size())==a&&quantization(q)==3);
    std::cout<<"PASS default, all presets, legacy fallback, indexed channels, quantization, AVX2\n";
}
