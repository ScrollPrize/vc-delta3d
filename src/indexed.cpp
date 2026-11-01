#include <vc_delta3d/indexed.hpp>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

extern "C" int vc_ng_preset_encode(const unsigned char*,unsigned,unsigned,unsigned char*,size_t,size_t*);
extern "C" int vc_ng_preset_decode(const unsigned char*,size_t,unsigned char*,size_t);
extern "C" unsigned vc_ng_frame_dimension(const unsigned char*,size_t);
extern "C" int vc_ng_rect_encode(const unsigned char*,unsigned,unsigned,unsigned,unsigned char*,size_t,size_t*);
extern "C" int vc_ng_rect_decode(const unsigned char*,size_t,unsigned,unsigned,unsigned,unsigned char*,size_t);

namespace vc_delta3d {
namespace {
using Bytes=std::vector<std::byte>;
constexpr size_t prefixBytes=24;
void check(bool ok){if(!ok)throw std::invalid_argument("invalid NGC1 index/partition");}
std::uint64_t mul(std::uint64_t a,std::uint64_t b){check(!b||a<=UINT64_MAX/b);return a*b;}
std::uint64_t voxels(const std::array<std::uint32_t,3>&s){return mul(mul(s[0],s[1]),s[2]);}
void shape(const std::array<std::uint32_t,3>&s){for(auto d:s)check(d&&d%32==0);}
bool leafShape(const std::array<std::uint32_t,3>&s){
    return s[0]>=32&&s[1]>=32&&s[2]>=32&&s[0]%32==0&&s[1]%32==0&&s[2]%32==0
        &&s[2]<=1024&&std::uint64_t(s[0])*s[1]<=65536;
}
bool cube(const std::array<std::uint32_t,3>&s){return s[0]==s[1]&&s[1]==s[2]&&leafShape(s);}
std::uint64_t read(std::span<const std::byte>b,size_t&pos,unsigned n){
    check(pos<=b.size()&&n<=b.size()-pos);std::uint64_t v=0;
    for(unsigned i=0;i<n;++i)v|=std::uint64_t(b[pos++])<<(8*i);return v;
}
void write(Bytes&b,size_t pos,std::uint64_t v,unsigned n){
    check(pos<=b.size()&&n<=b.size()-pos);
    for(unsigned i=0;i<n;++i)b[pos+i]=std::byte((v>>(8*i))&255);
}
void append(Bytes&b,std::uint64_t v,unsigned n){auto p=b.size();b.resize(p+n);write(b,p,v,n);}
struct Domain {
    std::array<std::uint32_t,3> origin{},extent{};
    std::uint32_t channel=0,channels=1;
};
std::pair<Domain,Domain> split(Domain a,unsigned axis,std::uint32_t cut){
    check(axis<4);Domain b=a;
    if(axis==3){check(cut&&cut<a.channels);a.channels=cut;b.channel+=cut;b.channels-=cut;}
    else{check(cut&&cut%32==0&&cut<a.extent[axis]);a.extent[axis]=cut;b.origin[axis]+=cut;b.extent[axis]-=cut;}
    return {a,b};
}
bool full(const IndexedLeaf&l,const std::array<std::uint32_t,3>&s){
    return l.originZYX==std::array<std::uint32_t,3>{}&&l.shapeZYX==s;
}
// Shared gather/scatter for either memory layout. Contiguous channel rows
// are copied together; no per-voxel work in the channel-first path.
void transfer(std::span<const std::byte>source,std::span<std::byte>dest,
    const IndexedLeaf&l,const std::array<std::uint32_t,3>&root,unsigned channels,
    ChannelLayout layout,bool gather){
    const auto&sz=l.shapeZYX;const auto&o=l.originZYX;size_t field=voxels(root),p=0;
    for(unsigned z=0;z<sz[0];++z)for(unsigned y=0;y<sz[1];++y){
        size_t q=(size_t(o[0]+z)*root[1]+o[1]+y)*root[2]+o[2];
        if(layout==ChannelLayout::First){
            q+=size_t(l.channel)*field;
            if(gather)std::memcpy(dest.data()+p,source.data()+q,sz[2]);
            else std::memcpy(dest.data()+q,source.data()+p,sz[2]);
        }else{
            q=q*channels+l.channel;
            for(unsigned x=0;x<sz[2];++x){
                if(gather)dest[p+x]=source[q+size_t(x)*channels];
                else dest[q+size_t(x)*channels]=source[p+x];
            }
        }
        p+=sz[2];
    }
}
}

std::optional<std::uint32_t> indexedHeaderSize(std::span<const std::byte> b){
    if(b.size()<8||std::memcmp(b.data(),"NGC1",4))return std::nullopt;
    size_t p=4;auto n=read(b,p,4);if(n<prefixBytes+17)return std::nullopt;return std::uint32_t(n);
}

ChunkIndex readIndex(std::span<const std::byte> bytes,std::uint64_t objectBytes){
    auto hn=indexedHeaderSize(bytes);check(hn&&*hn<=bytes.size()&&*hn<=objectBytes);
    auto b=bytes.first(*hn);size_t p=8;ChunkIndex index;index.headerBytes=*hn;
    index.encodedBytes=objectBytes; // also implied by the final leaf's end
    for(auto&d:index.shapeZYX)d=std::uint32_t(read(b,p,4));shape(index.shapeZYX);
    auto c=read(b,p,4);index.layout=(c>>31)?ChannelLayout::Last:ChannelLayout::First;
    index.channels=c&0x7fffffff;check(index.channels);
    index.decodedBytes=mul(voxels(index.shapeZYX),index.channels);check(index.decodedBytes<=SIZE_MAX);
    std::vector<Domain> pending{{{},index.shapeZYX,0,index.channels}};
    std::uint64_t next=*hn;
    while(!pending.empty()){
        auto d=pending.back();pending.pop_back();auto tag=read(b,p,1);
        if(tag<4){auto cut=read(b,p,4);auto [left,right]=split(d,unsigned(tag),std::uint32_t(cut));pending.push_back(right);pending.push_back(left);continue;}
        check(tag>=16&&tag<=18&&d.channels==1&&leafShape(d.extent));
        if(tag!=18)check(cube(d.extent));
        IndexedLeaf leaf{d.channel,d.origin,d.extent,LeafCodec(tag),read(b,p,8),read(b,p,8)};
        // Canonical tightly packed leaves: no overlap, padding, gaps or tail.
        check(leaf.offset==next&&leaf.length>=4&&next<=objectBytes&&leaf.length<=objectBytes-next);
        next+=leaf.length;index.leaves.push_back(leaf);
    }
    check(p==b.size()&&next==objectBytes);return index;
}

std::vector<std::byte> compressIndexed(std::span<const std::byte> input,
    std::array<int,3>s,int channels,ChannelLayout layout,const IndexedOptions&opt){
    check(channels>0&&(layout==ChannelLayout::First||layout==ChannelLayout::Last));
    check(opt.preset>=Predictor::Fast&&opt.preset<=Predictor::TooSlow);
    std::array<std::uint32_t,3> root{};
    for(unsigned i=0;i<3;++i){check(s[i]>0);root[i]=s[i];}shape(root);
    check(mul(voxels(root),channels)==input.size());
    // First validate and materialize the complete tree before fitting anything.
    Bytes tree;std::vector<IndexedLeaf> leaves;std::vector<size_t> records;
    std::vector<Domain> pending{{{},root,0,unsigned(channels)}};size_t plan=0;
    while(!pending.empty()){
        auto d=pending.back();pending.pop_back();PartitionNode node;
        if(!opt.partition.empty()){check(plan<opt.partition.size());node=opt.partition[plan++];}
        else if(d.channels>1){node.axis=SplitAxis::Channel;node.cut=d.channels/2;}
        else if(!cube(d.extent)){
            auto axis=std::max_element(d.extent.begin(),d.extent.end())-d.extent.begin();
            node.axis=SplitAxis(axis);node.cut=(d.extent[axis]/64)*32;check(node.cut);
        }
        auto axis=unsigned(node.axis);
        if(axis<4){check(!node.rectangularBaseline);auto [left,right]=split(d,axis,node.cut);
            append(tree,axis,1);append(tree,node.cut,4);pending.push_back(right);pending.push_back(left);continue;}
        check(node.axis==SplitAxis::Leaf&&node.cut==0&&d.channels==1&&leafShape(d.extent));
        check(node.rectangularBaseline||cube(d.extent));
        auto codec=node.rectangularBaseline?LeafCodec::Rectangular:
            opt.preset==Predictor::TooSlow?LeafCodec::Tree:LeafCodec::Choice;
        append(tree,unsigned(codec),1);records.push_back(prefixBytes+tree.size());
        append(tree,0,8);append(tree,0,8);leaves.push_back({d.channel,d.origin,d.extent,codec,0,0});
    }
    check(opt.partition.empty()||plan==opt.partition.size());
    check(tree.size()<=UINT32_MAX-prefixBytes);Bytes out(prefixBytes);
    std::memcpy(out.data(),"NGC1",4);write(out,4,prefixBytes+tree.size(),4);
    for(unsigned i=0;i<3;++i)write(out,8+4*i,root[i],4);
    write(out,20,unsigned(channels)|(layout==ChannelLayout::Last?0x80000000u:0),4);
    out.insert(out.end(),tree.begin(),tree.end());
    for(size_t i=0;i<leaves.size();++i){
        auto&l=leaves[i];size_t count=voxels(l.shapeZYX);Bytes gathered;
        std::span<const std::byte> src;
        if(layout==ChannelLayout::First&&full(l,root))src=input.subspan(size_t(l.channel)*count,count);
        else{gathered.resize(count);transfer(input,gathered,l,root,channels,layout,true);src=gathered;}
        Bytes payload(count*4+131072);size_t n=0;int rc;
        if(l.codec==LeafCodec::Rectangular)rc=vc_ng_rect_encode(
            reinterpret_cast<const unsigned char*>(src.data()),l.shapeZYX[2],l.shapeZYX[1],l.shapeZYX[0],
            reinterpret_cast<unsigned char*>(payload.data()),payload.size(),&n);
        else rc=vc_ng_preset_encode(reinterpret_cast<const unsigned char*>(src.data()),l.shapeZYX[0],unsigned(opt.preset)-3,
            reinterpret_cast<unsigned char*>(payload.data()),payload.size(),&n);
        if(rc)throw std::runtime_error("indexed leaf encoding failed");
        check(n<=payload.size()&&n<=SIZE_MAX-out.size());
        write(out,records[i],out.size(),8);write(out,records[i]+8,n,8);
        out.insert(out.end(),payload.begin(),payload.begin()+n);
    }
    return out;
}

bool decodeIndexedLeaf(std::span<const std::byte> payload,const IndexedLeaf&l,std::span<std::byte> output){
    try{
        check(leafShape(l.shapeZYX)&&payload.size()==l.length&&payload.size()>=4&&output.size()==voxels(l.shapeZYX));
        const char*magic=l.codec==LeafCodec::Choice?"NCF1":l.codec==LeafCodec::Tree?"HGT3":l.codec==LeafCodec::Rectangular?"NDB2":nullptr;
        check(magic&&!std::memcmp(payload.data(),magic,4));
        auto src=reinterpret_cast<const unsigned char*>(payload.data());auto dst=reinterpret_cast<unsigned char*>(output.data());
        if(l.codec==LeafCodec::Rectangular)return vc_ng_rect_decode(src,payload.size(),l.shapeZYX[2],l.shapeZYX[1],l.shapeZYX[0],dst,output.size())==0;
        check(cube(l.shapeZYX)&&vc_ng_frame_dimension(src,payload.size())==l.shapeZYX[0]);
        return vc_ng_preset_decode(src,payload.size(),dst,output.size())==0;
    }catch(...){return false;}
}

bool decompressIndexedInto(std::span<const std::byte> input,std::span<std::byte> output){
    try{
        auto index=readIndex(input,input.size());check(index.decodedBytes==output.size());
        for(const auto&l:index.leaves){
            auto payload=input.subspan(size_t(l.offset),size_t(l.length));size_t count=voxels(l.shapeZYX);
            if(index.layout==ChannelLayout::First&&full(l,index.shapeZYX)){
                if(!decodeIndexedLeaf(payload,l,output.subspan(size_t(l.channel)*count,count)))return false;
            }else{
                Bytes scratch(count);if(!decodeIndexedLeaf(payload,l,scratch))return false;
                transfer(scratch,output,l,index.shapeZYX,index.channels,index.layout,false);
            }
        }
        return true;
    }catch(...){return false;}
}
} // namespace vc_delta3d
