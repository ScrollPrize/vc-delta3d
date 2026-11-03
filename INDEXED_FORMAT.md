# Compact / extended indexed channel and spatial container

One tree grammar, with two fixed-width representations. No varints, explicit
payload offsets, alignment padding, historical container readers, or D3D1
children. Only original Delta3D retains compatibility. The magic selects every
field width in the container; it does not select a different codec or preset.

## Prefix

| Field | Compact NGC2 | Extended NGX2 |
|---|---:|---:|
| Magic/version |4|4|
| Total header length including prefix |2|4|
| Flags: bit0 channels-last, others zero |1|1|
| Root Z, Y, X: each dimension/32 minus one |3 ×1|3 ×4|
| Channel count minus one |1|4|
| **Total prefix** |**11 B**|**25 B**|

All integers are unsigned little-endian. Spatial dimensions must be multiples
of32. Compact root dimensions cover32..8192 and channels1..256. Extended dimensions
are also in units32, but reconstructed dimensions must fit uint32. Current APIs
limit channels to INT32_MAX and decoded size to SIZE_MAX.

The encoder selects extended if any root dimension, channel count, tree operand,
leaf length, or the total compact header length exceeds the compact field.
Compact header length is at most65535. Extended header length is at mostUINT32_MAX.
Reading8bytes suffices to discover either header length (compact needs only6).
No fixed-size reserved header slab.

## Tree template

The prefix is followed by one preorder tree template:

| Wire tag | Meaning | Following operand |
|---|---|---|
|0/1/2|Binary split Z/Y/X|First-child extent/32 minus one|
|3|Binary split channels|First-child channel count minus one|
|4/5/6|Repeat Z/Y/X|Repetition count minus one|
|7|Repeat channels|Repetition count minus one|
|8|Leaf|None|

Operands occupy1byte compact or4bytes extended. Spatial split values thus
cover32..8192 in compact mode, including byte255 representing8192.
Repeat counts cover2..256 in compact mode; stored zero (count1) is rejected.
Other tags are invalid. A binary split has two child templates. A repeat has
exactly one child template, instantiated into equal disjoint regions along its
axis. The parent extent must divide exactly; spatial children remain32-aligned.
Every leaf has exactly one channel. All other extents are inherited.

The template ends after its complete root subtree. No explicit template length
or node count is necessary. It is followed by one length per **expanded leaf**,
4bytes compact or8bytes extended. Payloads follow immediately in the same order.
Offsets are the cumulative sum of lengths starting at the header end, always
using64-bit arithmetic. The final leaf must end exactly at the object end.

Traversal order: first child before second for splits; repetition0,1,... for
repeats, each visiting the full child template. Each occurrence has its own
length, payload, fitted models and entropy state. Repetition shares geometry
only; it never shares compressed data or causal prediction state.

Default multi-channel encoding uses a channel-repeat node around one canonical
spatial partition. Explicit plans can use arbitrary binary splits and repeats,
including different spatial trees for different channels. Existing explicit
binary plans are not automatically deduplicated; callers can express repetition
directly. There is no automatic compression-driven partition search.

## Leaf representation and overhead

All leaves use the NCF1 codec. Their redundant four-byte magic is omitted inside
the container. They retain their four-byte compressed-header length, followed by
the same compact header, selector payload and residual payload as a standalone
NCF1 frame. Decode uses the shared native implementation with an expected ZYX
shape; no payload copy or reconstructed magic is needed. Embedded leaves must
be passed to decodeIndexedLeaf, not the standalone-frame decoder.

For a tree with B binary nodes, R repeat nodes, T template leaf nodes and L
expanded payload leaves:

- Compact index:11 +2B +2R +T +4L bytes.
- Extended index:25 +5B +5R +T +8L bytes.
- Embedded leaf prefixes:another4L bytes.
- Model/entropy headers and selector/residual data are additional.

Eight full-volume channels: one repeat, one template leaf, eight lengths:
**46-byte index +32-byte leaf prefixes =78 bytes total framing**.
The corresponding extended framing is127bytes.
Previous framing used259bytes; original Delta3D uses216bytes for the same
channel count (excluding its probability tables and states).

Standalone scalar NCF1 retains its8-byte magic/length prefix, with no container.
Data is uint8/lossless. Rectangular leaf limits remain X<=1024 and Y*Z<=65536,
all extents32-aligned. Logical chunks may exceed leaf limits by subdivision.
Explicit rectangular baseline encoding is an encoder option, not another format.

## APIs and random range access

C vc_partition_node and Python tuples use API axes:
0/1/2 split Z/Y/X,3 split channels,4 leaf,
5/6/7 repeat Z/Y/X,8 repeat channels.
For repeats, cut is the repetition count. Rectangular is permitted only on
leaves. These API axis values are deliberately distinct from wire tags.

Example: three channels each with two Z regions and two X regions:
`[(8,3,0),(5,2,0),(7,2,0),(4,0,0)]`.
This has four template nodes but twelve independently encoded leaves.

The existing C/C++/Python index APIs recognize both magics:
indexed_header_size, read_index, decode_indexed_leaf.
Fetch an initial range (at least8bytes), extend it if necessary to header length,
then fetch only the selected leaf. All dimensions, layout, lengths and
partitions are stored in the object; Zarr metadata needs no extra information.
Independent leaf decoding can run concurrently.

Parsing is iterative. Expansion is bounded by the number of physical length
entries, so a tiny malicious repeat template cannot force unbounded expansion.
Cuts, repeats, overflow, geometry, total coverage and exact object length are
checked before payload decoding. No checksum is implied.

Regular compression falls back to complete original Delta3D for uint16,
quantization or non32-aligned positive dimensions. Explicit indexed compression
rejects these inputs instead. Old experimental containers are not supported.

## Build and validation

From this checkout, using the existing configured Release build:

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
/data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/vc_delta3d_modern_smoke
```

Python regression coverage is in tests/test_indexed.py, alongside the existing
test_python.py, test_modern.py and test_fallback.py suites. With the built native
module loaded, all135tests pass. Native CTest6/6 passes; the sanitizer build
exercises truncated/mutated indexes, repeat expansion and64-bit ranges.
Framing-byte counts above are exact structural sizes, not throughput claims.
No macOS/ARM hardware test was performed in this session.
