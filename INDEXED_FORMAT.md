# NGC1 indexed channel/spatial container

NGC1 implements the agreed NG/Delta tree container. It is **not** a D3D1
extension, and contains no D3D1 children. Preset and legacy encoding paths
are separate. All integers are fixed-width little-endian, with no alignment
padding. The complete index is uncompressed and precedes all payloads.

## Prefix — 24 bytes

|Offset|Bytes|Meaning|
|---|---:|---|
|0|4|`NGC1` (magic and version)|
|4|4|Total header length, including prefix|
|8|12|Logical Z, Y, X dimensions (three uint32)|
|20|4|Channel count in low31 bits; high bit means channels-last|

Total object length is implied by the last leaf's end and checked against
the storage object's length; it is not stored a second time in the prefix.

Channels-first memory order is C/Z/Y/X, channels-last is Z/Y/X/C. The data
type is uint8, lossless, by definition. Spatial extents are positive multiples
of32; no padding, clipping, or partial boundary leaves.

## Preorder binary partition tree

The root covers the full channel/spatial domain. Every internal node has
exactly two children, serialized first-child subtree then second-child subtree.
No child pointers, node counts, parent IDs, or repeated leaf coordinates.

* Split record: **5 bytes** — axis tag(uint8), cut(uint32).
  Tags0/1/2 mean Z/Y/X; tag3 means channels. The cut is the first child's
  extent relative to the parent, in voxels or channel count. Spatial cuts
  must be multiples of32, and every cut must be strictly inside its parent.
* Leaf record: **17 bytes** — codec/version tag(uint8), absolute object
  payload offset(uint64), payload length(uint64).
  Tag16 is NCF1; tag17 is retired and rejected; tag18 is the documented rectangular NDB2
  mode504 grammar. Each leaf covers exactly one channel. Shape, location and
  channel identity are derived from the tree.

Payloads immediately follow the header, in leaf traversal order, without
gaps or alignment padding. Offsets and lengths are validated against the full
object length. The last payload must end exactly at the object end.
Each leaf owns its complete prediction and entropy state; no sibling references.
NCF1 are unchanged native frames. Rectangular NDB2 stores its dimensions
only in the index, not an additional NDL1 wrapper.

Overhead for L leaves is **24 + 17L + 5(L−1) = 22L + 19 bytes**:
41 for1 leaf,63 for2,107 for4,195 for8,371 for16. For8 channels of128³ this
is0.001162% of raw size. There is no reserved4KiB header slab. These are
container/index bytes; the leaf's own model/entropy metadata remains necessary.
Ordinary scalar cubic encoding emits NCF1 directly, with **zero**
additional container overhead; `compressIndexed` explicitly requests an index.

## Shapes, encoding policy and validation

Logical dimensions can exceed a single leaf's limits. Rectangular leaves
require32<=X<=1024, Y/Z>=32, all dimensions32-aligned, and Y*Z<=65536.
The default encoder separates channels and partitions spatially into cubes
that fit the existing four preset kernels. No adaptive partition optimization
is implied. Caller-provided preorder partitions allow different spatial trees
per channel. Rectangular mode504 leaves are an **explicit** per-leaf option,
not a silent replacement for fast/medium/slow/slower.

The decoder validates header lengths, integer products, geometry, split
alignment, exact tree coverage, leaf codec IDs and payload ranges before
decoding. The traversal uses a heap stack, not recursive calls on untrusted
tree depth. Invalid partitions are rejected before encoder model fitting.
The regular `compress`/`compressChannels` APIs automatically choose the complete
legacy Delta3D path for 16-bit data, quantized data, or positive dimensions not
divisible by 32. This produces D3D1, never an NGC1 container with legacy leaves.
Invalid types, shapes, sizes and quantization settings still raise errors.
The explicit `compressIndexed` API does not fall back: incompatible geometry
or partition requests are errors. Explicit `delta` behavior is unchanged.
D3D1 version2 accepts only legacy D3D1 scalar children; mixed old/new containers
and the discarded D3D1 version3 envelope are rejected.

## APIs and byte-range access

Existing `compressChannels` / Python `Delta3D(channel_axis=...)` select NGC1
for presets and the unchanged D3D1 container for Delta. Existing
`decompress`, `decompressInto`, `decodedSize`, and `channelCount` recognize NGC1.
All format/shape/channel/partition information needed to decode is in the
object; no per-chunk index information is added to Zarr metadata.

Additional C++ APIs in `include/vc_delta3d/indexed.hpp`:

* `compressIndexed(input, shapeZYX, channels, layout, options)`.
* `indexedHeaderSize(prefix)` needs only8 bytes.
* `readIndex(header, objectBytes)` needs the header only, not any payload.
* `decodeIndexedLeaf(payload, leafDescriptor, output)` needs only that leaf's
  byte range. Calls on independent leaves can run concurrently.

Python equivalents are native bindings, not Python codec implementations:

```python
from vc_delta3d import indexed_header_size, read_index, decode_indexed_leaf

# fetch is the application's byte-range reader; object_size comes from storage.
prefix = fetch(0, min(4096, object_size))
header_size = indexed_header_size(prefix)
header = prefix[:header_size]
if len(header) < header_size:
    header += fetch(len(header), header_size - len(header))
index = read_index(header, object_size)
leaf = index.leaves[0]
pixels = decode_indexed_leaf(fetch(leaf.offset, leaf.length), leaf)
```

This is two reads when the first range contains the whole header; otherwise
one header-extension read is needed. Cached headers remove repeat index reads.
The codec does not itself implement S3/Zarr range requests; the APIs make them
possible without fetching/decompressing siblings.

Python `compress_indexed` also accepts `partition=[(axis,cut,rectangular),...]`.
Axis4 denotes a leaf, with cut0; rectangular1 explicitly selects mode504.
Example: channel0 one64³ preset leaf; channel1 two32×64×64 rectangular leaves:
`[(3,1,0),(4,0,0),(0,32,0),(4,0,1),(4,0,1)]`.
