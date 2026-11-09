#pragma once
#include <vc_delta3d/codec.hpp>
#include <cstdint>

namespace vc_delta3d {
// Preorder partition. A split's first subtree immediately follows it,
// then its second subtree. Spatial cuts are relative voxel extents (32-grid);
// channel cuts are relative channel counts. No padding or overlapping leaves.
// Repeats have one child template and cut stores the repetition count.
enum class SplitAxis : unsigned char { Z=0, Y=1, X=2, Channel=3, Leaf=4,
    RepeatZ=5, RepeatY=6, RepeatX=7, RepeatChannel=8 };
struct PartitionNode {
    SplitAxis axis=SplitAxis::Leaf;
    std::uint32_t cut=0;
};
struct IndexedOptions {
    Predictor preset=Predictor::Fast;
    std::vector<PartitionNode> partition; // empty: channel repeat + cubic leaves
};
enum class LeafCodec : unsigned char { Choice=16 };
struct IndexedLeaf {
    std::uint32_t channel=0;
    std::array<std::uint32_t,3> originZYX{},shapeZYX{};
    LeafCodec codec=LeafCodec::Choice;
    std::uint64_t offset=0,length=0;
};
struct ChunkIndex {
    std::array<std::uint32_t,3> shapeZYX{};
    std::uint32_t channels=0,headerBytes=0;
    ChannelLayout layout=ChannelLayout::First;
    std::uint64_t encodedBytes=0,decodedBytes=0;
    std::vector<IndexedLeaf> leaves;
};

// NGI1 index. Read >=8 bytes; nullopt means invalid/incomplete.
// Read the rest of this many header bytes, then inspect without any payload.
std::optional<std::uint32_t> indexedHeaderSize(std::span<const std::byte> prefix);
// Validates tree coverage, geometry, codecs, offsets and exact object length.
// Throws invalid_argument on malformed input. Does NOT fetch/decode payloads.
ChunkIndex readIndex(std::span<const std::byte> header,std::uint64_t objectBytes);
// Input is uint8 only. Spatial extents/cuts are positive multiples of32.
// Encoder leaves are cubic and use the selected preset; decoding also accepts rectangles.
std::vector<std::byte> compressIndexed(std::span<const std::byte> input,
    std::array<int,3> shapeZYX,int channels,ChannelLayout layout,
    const IndexedOptions& options={});
// Only this leaf's byte range is needed; output is contiguous Z/Y/X uint8.
// Safe for concurrent independent calls. No access to other leaves required.
bool decodeIndexedLeaf(std::span<const std::byte> payload,const IndexedLeaf& leaf,
    std::span<std::byte> output);
bool decompressIndexedInto(std::span<const std::byte> input,std::span<std::byte> output);
} // namespace vc_delta3d

// Header-only ownership adapter; codec implementation and library are pure C.
#include <vc_delta3d/detail/indexed_adapter.hpp>
