# VC-Delta3D wire formats

Every encoded object starts with a four-byte magic that identifies it.
`vc_decompress` dispatches on that magic alone; decoding never needs the
encoder preset or any Zarr-side metadata.

| Magic | What it is | Written for |
|---|---|---|
| `NCF1` | One preset-coded uint8 field (the NG/Delta3D frame) | A single cubic uint8 chunk, 32-aligned, edge ≤ 256, preset fast…slower |
| `NGI1` | Index over several NCF1 leaves (channels and/or spatial partitions) | Multi-channel, rectangular or larger uint8 chunks, preset fast…slower |
| `D3D1` | Original Delta3D: delta filter + rANS, scalar (v1) or per-channel (v2) | `preset=delta`, uint16/int16, quantization > 1, or dimensions not divisible by 32 |

How they relate:

```
NGI1  prefix | tree template | leaf lengths | leaf | leaf | ...
                                              │
                                              └─ NCF1 frame without its 4-byte magic

NCF1  magic | header length | compact header | selector payload | residual payload

D3D1  v1: 20-byte header | rANS payload
      v2: 24-byte header | channel lengths | D3D1 v1 frame | D3D1 v1 frame | ...
```

- **NCF1 is the codec.** All preset compression happens in NCF1 frames.
- **NGI1 is only an index.** It holds geometry and byte lengths so a reader can
  fetch and decode one channel or region by byte range. It never changes how a
  leaf is coded, and it only contains NCF1 leaves.
- **D3D1 is the separate original Delta3D codec.** It covers inputs that NCF1
  cannot represent (16-bit, quantized, unaligned shapes) and stays readable for
  data written by earlier releases. D3D1 and NCF1/NGI1 are never nested inside
  each other.

Conventions used below: integers are little-endian; a *varint* is an unsigned
LEB128 value (7 bits per byte, low group first, continuation bit `0x80`).
Shapes are given as Z, Y, X in elements unless stated otherwise.

---

## 1. NCF1 frame

```
"NCF1" | header_bytes:u32 | compact_header | selector_payload | residual_payload
```

`header_bytes` is the length of `compact_header`. The selector payload length
is stored inside the header; the residual payload is the rest of the frame.
The same frame describes cubic and rectangular fields.

The format fixes what can be expressed, not how an encoder chooses it. Menus,
model reuse, entropy-bank sharing, fit geometry, class counts and selector
block size are encoder decisions within the bounds below; the current preset
policies (see `src/ng/README.md`) are not part of the format.

### 1.1 Compact header: controls

Unsigned varints, in this order:

1. `X`, `Y`, `Z`, `mode | block_edge << 8`, model count, menu count,
   selector payload byte count.
2. Per model: five configuration values (three neighborhood parameters, class
   count, entropy precision). Signed neighborhood values are stored as their
   uint8 representation. The coefficient layout follows from these values.
3. Per menu entry: predictor ID, model ID (255 = no model), entropy-bank ID,
   and two signed input weights as their exact uint32 representation.
   The bank count is `max(bank ID) + 1`.
4. Entropy-present flag (0/1). If 1: the entropy geometry/class/precision
   configuration. Entropy carries thresholds and frequency tables only;
   predictor weights come from the model records.

`mode` 0 is correction on a fixed base; `mode` 1 uses learned input weights.

### 1.2 Compact header: numerical pool

After the controls follows one pool of numerical records, in this order:

1. model parameter bodies: two coefficient arrays and biases, class thresholds
   and class biases, in NG order; sizes follow from each model's configuration;
2. entropy thresholds: `classes_per_bank − 1` values, shared by all banks;
3. entropy frequency tables: one 256-entry table per bank and class.

Fields that the decoder does not use (unused predictor fields, unused entropy
threshold/bias slots) are not transmitted and are reconstructed as zero.

Every record is compacted the same way:

- Its full length follows from the schema. A varint gives the retained-prefix
  count; the omitted suffix is zero.
- Values 0–15 are their own label. A larger 16-bit value `v` gets label
  `12 + floor(log2(v))`, and its low exponent bits go to a bypass stream.
- All label records share one eight-lane rANS state group. Per record the
  encoder chooses literal labels or wrapping 16-bit deltas of label-byte pairs
  (an odd final byte is left unchanged).
- Coding byte: bit 2 = delta on/off; bits 0–1 = raw (0), fixed-table rANS (1),
  transmitted-table rANS (2). Other values are invalid. It is followed by the
  encoded label byte count and the bytes.
- Fixed-table mode uses one of two constant frequency tables in
  `src/ng/header_tables.h`, selected by bit 2. These tables are part of the
  format; changing them changes the format.
- The bypass bits follow, low bit first; their count follows from the labels.
  Final padding bits must be zero.

### 1.3 Bounds

- Geometry: X, Y, Z each a multiple of 32 and ≥ 32; `X ≤ 1024`, `Y·Z ≤ 65536`.
- 0–4 models; 1–60 menu entries; predictor IDs 0–12 (12 = zero predictor,
  which takes no model and zero weights).
- Up to 16 entropy banks, 2–32 classes per bank, at most 128 tables in total.
- Selector block edge: 2, 4, 8, 16 or 32 (the current presets use 4).
- Two menu entries with the same model and predictor must use the same two
  input weights; a different fitted combination needs its own model record.

### 1.4 Selector payload

One uint8 menu index per selector block. The number of IDs follows from the
frame geometry and block edge.

A tag byte selects the map coding:

| Tag | Payload |
|---|---|
| 0 | One constant ID, repeated for every block |
| 1 | Previous-ID context coding |
| 2 | Packing width byte (1–8), then packed-LZ coding |

#### Small-stream coding

Both coded modes encode their streams with the same stream format, which
starts with a stream tag:

- 0: raw bytes.
- 1: one constant byte.
- 2: sparse normalized frequencies: used-symbol count, increasing symbol gaps
  (first from 0, then from previous + 1), frequencies of all but the last
  symbol; the last is `4096 − sum`. All values are varints.
- 3–6: alphabet size (varint); frequencies derived from integer weights
  `4096`, `floor(65536/(s+1))`, `floor(65536/(s+1)²)` or `65536 >> min(s,16)`
  respectively (each at least 1), normalized with `vc_rans_normalize`.

Tags 2–6 are followed by the encoded size (varint), a 4-byte final state and
the byte-renormalized rANS payload (scale 12, total 4096, lower bound 2²³).
The final decoded state must equal 2²³ and all bytes must be consumed.
The decoded length of each stream comes from the surrounding structure.

#### Context mode (tag 1)

Alphabet size, used-context count (varints), then for each context in
ascending order: context ID (varint), decoded count (varint), one stream.
The context is the previous ID, starting at 0. Symbols are grouped by context
and the decoder rebuilds the original order. Every ID must be below the
alphabet size and every stream must be consumed exactly.

#### Packed-LZ mode (tag 2)

IDs are bit-packed, low bits first, at the given width; the packed size is
`ceil(count · width / 8)` and unused final bits are zero. Then:

- match count, literal count (varints);
- three streams: literal-run codes (matches + 1), match-length codes (matches),
  distance codes (matches); zero-length streams are omitted;
- extra-bit byte count (varint) and the extra bits;
- the literal stream, if non-empty.

A value `v` is coded as `floor(log2(v + 1))` with the low bits of `v + 1` as
extra bits. Match lengths are stored minus 4. Each sequence reads literal
length, match length, then distance extra bits; the last sequence has
literals only. Distance codes 0–2 refer to a move-to-front cache initialised
to `[1, 4, 8]`; code `3 + k` means distance `2^k + extra` (k bits). Extra bits
are low-first and zero-padded. Matches copy forward and may overlap; maximum
distance 65536, minimum match length 4.

### 1.5 Residual payload

Voxel residuals coded with the bulk eight-lane rANS coder, using the entropy
banks and classes defined in the header and the per-block predictor choices
from the selector map.

---

## 2. NGI1 indexed container

Holds several NCF1 leaves: channels, spatial partitions, or both. The whole
index is at the start of the object, so a reader can fetch it with one small
range request and then fetch individual leaves.

### 2.1 Prefix

| Field | Encoding |
|---|---|
| Magic `NGI1` | 4 bytes |
| Total header length, including the prefix | u32 |
| Flags: bit 0 = channels-last, other bits zero | 1 byte |
| Root Z, Y, X, each as `dimension/32 − 1` | 3 varints |
| Channel count − 1 | varint |

Spatial dimensions are multiples of 32; channel count is 1…INT32_MAX.
The first 8 bytes are enough to learn how much more to fetch for the index.

### 2.2 Tree template

One preorder tree follows the prefix:

| Tag | Node | Operand (varint) |
|---|---|---|
| 0 / 1 / 2 | Binary split along Z / Y / X | first-child extent / 32 − 1 |
| 3 | Binary split of channels | first-child channel count − 1 |
| 4 / 5 / 6 | Repeat along Z / Y / X | repetition count − 1 |
| 7 | Repeat channels | repetition count − 1 |
| 8 | Leaf | none |

Other tags are invalid. A split has two child subtrees; a repeat has one child
subtree, applied to equal, disjoint regions along its axis. The parent extent
must divide exactly, spatial children stay 32-aligned, and a stored repeat
count of 0 (count 1) is invalid. Every leaf covers exactly one channel. The
template ends with its root subtree; no length or node count is stored.

Expanded traversal order: first child before second for splits; repetitions
0, 1, … for repeats, each visiting the full child subtree. Repetition shares
geometry only: every expanded leaf has its own payload, models and entropy
state.

### 2.3 Lengths and payloads

After the template: one varint byte length per expanded leaf, in traversal
order, then the payloads in the same order. Leaf offsets are the running sum
of lengths from the end of the header (64-bit arithmetic); the last leaf must
end exactly at the end of the object.

Each leaf is an NCF1 frame without the `NCF1` magic: `header_bytes:u32`,
compact header, selector payload, residual payload. Leaves are decoded with
`decode_indexed_leaf` (which passes the expected shape), not with the
standalone frame decoder.

### 2.4 Size

For B binary nodes, R repeat nodes, T template leaves and L expanded leaves,
the index takes `13 + 2B + 2R + T` bytes plus one varint per leaf length
(3 bytes for leaves of 16 KiB–2 MiB), assuming dimensions below 4096, fewer
than 129 channels and operands below 128. Leaves add 4L bytes for their
`header_bytes`. Example: eight 128³ channels (one repeat, one template leaf)
take 40 index bytes + 32 leaf-prefix bytes = 72 bytes of framing.

### 2.5 Encoder and API notes

- Default multi-channel encoding is a channel repeat around one spatial
  partition. Rectangular or large inputs are split into cubic leaves.
  Explicit plans may use any splits and repeats but must produce cubic leaves;
  readers accept rectangular leaves.
- The API describes plans as `(axis, cut)` pairs with axis numbers that differ
  from the wire tags: 0/1/2 split Z/Y/X, 3 split channels, 4 leaf,
  5/6/7 repeat Z/Y/X, 8 repeat channels; for repeats, `cut` is the count.
  Example, three channels each split in two along Z and X:
  `[(8,3),(5,2),(7,2),(4,0)]`: four template nodes, twelve leaves.
- Byte-range reading: fetch at least 8 bytes, extend to the header length,
  call `read_index`, then fetch and decode only the needed leaves (in parallel
  if desired). `indexed_header_size`, `read_index` and `decode_indexed_leaf`
  are available in C, C++ and Python.
- Readers check cuts, repeat expansion (bounded by the number of length
  entries), overflow, geometry, coverage and the exact object length before
  decoding any payload. There is no checksum.

---

## 3. D3D1 (original Delta3D)

A reversible delta filter followed by one eight-lane rANS stream. Supports
uint8/int8 and uint16/int16 data, any positive shape and optional quantization.
Signed data is coded as its bytes and must stay lossless.

### 3.1 Version 1: one field

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | Magic `D3D1` |
| 4 | 1 | Version = 1 |
| 5 | 1 | Element size (1 or 2) |
| 6 | 1 | Quantization step (0 is read as 1) |
| 7 | 1 | Flags: bits 0–3 = 1; bit 7 = mask present; bits 4–6 = delta mask |
| 8 | 12 | Shape Z, Y, X (u32 each) |
| 20 | … | rANS payload |

Delta mask bits: 4 = along Z, 2 = along Y, 1 = along X. Without bit 7 the mask
is 7 (all axes) and bits 4–6 must be zero. The encoder picks the mask for
uint8 fields of at least 2³; uint16 always uses 7.

rANS payload: a 512-byte frequency table (256 × u16, summing to 4096), eight
64-bit lane states, then the 32-bit renormalization words. Decoding applies
the inverse filter in X, Y, Z order.

### 3.2 Version 2: channels

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | Magic `D3D1` |
| 4 | 1 | Version = 2 |
| 5 | 1 | Element size |
| 6 | 1 | Quantization step |
| 7 | 1 | Layout: 0 = channels first, 1 = channels last |
| 8 | 12 | Per-channel shape Z, Y, X |
| 20 | 4 | Channel count C |
| 24 | 4·C | Byte length of each channel frame |
| … | … | C version-1 frames with the same element size and shape |

### 3.3 Alternate magic

`vc_decompress_with_magic` reads D3D1-layout frames written under a different
magic (for example `VCZ1`). It applies only to this layout, not to NCF1/NGI1.

---

## 4. Compatibility

- Decoding depends only on the bytes: magic, header fields and lengths.
- D3D1 v1/v2 frames from all releases remain readable.
- NCF1 and NGI1 are the 0.3.0 formats. Formats that 0.3.0 dropped (LPC frames
  from 0.2.0, earlier experimental NCF1 variants, NGC2/NGX2 containers) are
  rejected explicitly, not misread.
