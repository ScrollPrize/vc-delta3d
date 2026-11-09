# NGI1 indexed channel and spatial container

One tree grammar with one representation. The total header length is a fixed
u32 so a reader learns it from the first 8 bytes; every other integer is an
unsigned LEB128 varint (7 bits per byte, low group first). No payload offsets,
alignment padding, historical container readers, or D3D1 children. Only original
Delta3D retains compatibility. The magic does not select a codec or preset.

## Prefix

| Field | Encoding |
|---|---|
| Magic `NGI1` | 4 bytes |
| Total header length including prefix | u32 LE |
| Flags: bit0 channels-last, others zero | 1 byte |
| Root Z, Y, X: each dimension/32 minus one | 3 varints |
| Channel count minus one | varint |

Spatial dimensions must be multiples of 32 and fit uint32; channels are
1..INT32_MAX. Current APIs limit decoded size to SIZE_MAX. Reading 8 bytes
suffices to discover the header length. No fixed-size reserved header slab.

## Tree template

The prefix is followed by one preorder tree template:

| Wire tag | Meaning | Following operand |
|---|---|---|
|0/1/2|Binary split Z/Y/X|First-child extent/32 minus one|
|3|Binary split channels|First-child channel count minus one|
|4/5/6|Repeat Z/Y/X|Repetition count minus one|
|7|Repeat channels|Repetition count minus one|
|8|Leaf|None|

Operands are varints. Spatial split values are stored /32 minus one.
Repeat counts are stored minus one; stored zero (count 1) is rejected.
Other tags are invalid. A binary split has two child templates. A repeat has
exactly one child template, instantiated into equal disjoint regions along its
axis. The parent extent must divide exactly; spatial children remain32-aligned.
Every leaf has exactly one channel. All other extents are inherited.

The template ends after its complete root subtree. No explicit template length
or node count is necessary. It is followed by one varint length per **expanded leaf**
(up to 64 bits). Payloads follow immediately in the same order.
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

- Index: 13 + 2B + 2R + T bytes plus one varint per leaf length (3 bytes for
  leaves of 16 KiB to 2 MiB), assuming dimensions below 4096, fewer than 129
  channels and split/repeat operands below 128; larger values take more varint bytes.
- Embedded leaf prefixes: another 4L bytes.
- Model/entropy headers and selector/residual data are additional.

Eight full-volume 128³ channels (leaves of 16 KiB to 2 MiB): one repeat, one
template leaf, eight lengths: **40-byte index + 32-byte leaf prefixes = 72 bytes
total framing**. The former NGC2/NGX2 layouts needed 78/127 bytes; original
Delta3D uses 216 bytes for the same channel count (excluding its probability
tables and states).

Standalone scalar NCF1 retains its8-byte magic/length prefix, with no container.
Data is uint8/lossless. Rectangular leaf limits remain X<=1024 and Y*Z<=65536,
all extents32-aligned. Logical chunks may exceed leaf limits by subdivision.
Encoding uses only fast/medium/slow/slower presets on cubic leaves. Rectangular
input volumes use the canonical cubic partition. Explicit plans must also
specify cubic leaves. The reader still accepts rectangular leaves; no wire
format capability has been removed.

## APIs and random range access

C vc_partition_node and Python tuples use API axes:
0/1/2 split Z/Y/X,3 split channels,4 leaf,
5/6/7 repeat Z/Y/X,8 repeat channels.
For repeats, cut is the repetition count. Nodes are `(axis, cut)` pairs;
the former rectangular-baseline policy flag has been removed. These API axis
values are deliberately distinct from wire tags.

Example: three channels each with two Z regions and two X regions:
`[(8,3),(5,2),(7,2),(4,0)]`.
This has four template nodes but twelve independently encoded leaves.

The C/C++/Python index APIs are indexed_header_size, read_index and
decode_indexed_leaf. The former NGC2/NGX2 containers are not readable.
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

From the repository root, after configuring a Release build with native tests
and `VC_DELTA3D_MODERN_SMOKE=ON`:

```sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/vc_delta3d_modern_smoke
```

Python regression coverage is in tests/test_indexed.py, alongside the existing
test_python.py, test_modern.py and test_fallback.py suites. The sanitizer build
exercises truncated/mutated indexes, repeat expansion and64-bit ranges.
Framing-byte counts above are exact structural sizes, not throughput claims.
No macOS/ARM hardware test was performed in this session.
