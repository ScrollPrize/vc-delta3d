#include <vc_delta3d/indexed.hpp>
#include <vc_delta3d/indexed.h>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vc_delta3d {
namespace {
void check(bool ok){if(!ok)throw std::invalid_argument("invalid NGC1 index/partition");}
}

std::optional<std::uint32_t> indexedHeaderSize(std::span<const std::byte> b){
    auto n=vc_indexed_header_size(reinterpret_cast<const uint8_t*>(b.data()),b.size());
    return n?std::optional<std::uint32_t>(n):std::nullopt;
}
ChunkIndex readIndex(std::span<const std::byte> b,std::uint64_t objectBytes){
    vc_chunk_index raw{};
    check(vc_index_read(reinterpret_cast<const uint8_t*>(b.data()),b.size(),objectBytes,&raw)!=0);
    std::unique_ptr<vc_chunk_index,decltype(&vc_index_free)> guard(&raw,&vc_index_free);
    ChunkIndex out;
    std::copy_n(raw.shape,3,out.shapeZYX.begin());
    out.channels=raw.channels;out.headerBytes=raw.header_bytes;out.layout=ChannelLayout(raw.layout);
    out.encodedBytes=raw.encoded_bytes;out.decodedBytes=raw.decoded_bytes;
    out.leaves.reserve(raw.leaf_count);
    for(size_t i=0;i<raw.leaf_count;++i){
        const auto&l=raw.leaves[i];
        IndexedLeaf leaf;
        leaf.channel=l.channel;leaf.codec=LeafCodec(l.codec);leaf.offset=l.offset;leaf.length=l.length;
        std::copy_n(l.origin,3,leaf.originZYX.begin());std::copy_n(l.shape,3,leaf.shapeZYX.begin());
        out.leaves.push_back(leaf);
    }
    return out;
}

std::vector<std::byte> compressIndexed(std::span<const std::byte> input,
    std::array<int,3> shape,int channels,ChannelLayout layout,const IndexedOptions&opt){
    check(channels>0);
    uint32_t dims[3];
    for(unsigned a=0;a<3;++a){check(shape[a]>0);dims[a]=unsigned(shape[a]);}
    std::vector<vc_partition_node> plan;
    plan.reserve(opt.partition.size());
    for(const auto&n:opt.partition)plan.push_back({unsigned(n.axis),n.cut,unsigned(n.rectangularBaseline)});
    uint8_t *raw;size_t size;
    check(vc_indexed_encode(reinterpret_cast<const uint8_t*>(input.data()),input.size(),dims,
        unsigned(channels),unsigned(layout),unsigned(opt.preset),plan.data(),plan.size(),&raw,&size)!=0);
    std::unique_ptr<uint8_t,decltype(&std::free)> guard(raw,&std::free);
    const auto*p=reinterpret_cast<const std::byte*>(raw);
    return {p,p+size};
}

bool decodeIndexedLeaf(std::span<const std::byte> payload,const IndexedLeaf&l,std::span<std::byte> output){
    vc_indexed_leaf raw{};
    raw.channel=l.channel;raw.codec=unsigned(l.codec);raw.offset=l.offset;raw.length=l.length;
    std::copy(l.originZYX.begin(),l.originZYX.end(),raw.origin);
    std::copy(l.shapeZYX.begin(),l.shapeZYX.end(),raw.shape);
    return vc_indexed_leaf_decode(reinterpret_cast<const uint8_t*>(payload.data()),payload.size(),&raw,
        reinterpret_cast<uint8_t*>(output.data()),output.size())!=0;
}
bool decompressIndexedInto(std::span<const std::byte> input,std::span<std::byte> output){
    return vc_indexed_decode(reinterpret_cast<const uint8_t*>(input.data()),input.size(),
        reinterpret_cast<uint8_t*>(output.data()),output.size())!=0;
}
} // namespace vc_delta3d
