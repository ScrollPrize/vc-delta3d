# Current compact NCF1 header

This is the **one evolving new format**, not a compatibility layer. Earlier
experimental NCF1, NDB2, selector markers and model representations are not
accepted. Only original Delta3D D3D1 decoding/fallback remains supported.

All four presets emit this grammar. The format does not prescribe their menus:
future encoders can choose different menus, model reuse, entropy-bank sharing,
fit geometry, class counts and fixed block sizes within the supported bounds.
The current encoder policies are not part of the wire format.

## Frame

`NCF1 | header_bytes:u32le | compact_header | selector_payload | residual_payload`

Eight-byte prefix. No uncompressed header length, extra envelope, alignment
padding, historical version switch or Zarr-side per-chunk metadata. Selector
length is in the header; residual length is the remaining frame size. Cubic
and rectangular leaves use the same frame. NGC2 supplies the optional
channel/spatial random-access index (NGC2 compact / NGX2 extended), not an
alternative leaf codec. Indexed leaves omit only the four-byte NCF1 magic;
their header length and subsequent payload use the same parser and decoder.

## Semantic fields

Controls are unsigned base-128 varints, in this order:

1. X, Y, Z, `mode | block_edge<<8`, model count, menu count, selector byte count.
2. Each model's five configuration values: three neighborhood parameters,
   class count, entropy precision. Signed neighborhood values retain their
   uint8 representation. One standard coefficient layout follows logically;
   no model version or model length is transmitted.
3. Each menu entry: predictor ID, model ID (255=no model), entropy-bank ID,
   and two exact uint32 representations of signed input weights. No reserved
   byte. Bank count is `max(bank ID)+1`, not repeated.
4. Entropy-present flag. If present: its geometry/class/precision configuration.
   It has thresholds and frequency tables, **no second set of predictor
   coefficients**. Predictor weights come only from model records.

After these controls comes the one numerical pool. Record order is model
parameter bodies, entropy thresholds, entropy frequency tables. Model bodies
are two coefficient arrays and biases, class thresholds and class biases in
native NG order. Neighborhood and class configuration determine their sizes.
Entropy has `classes_per_bank-1` thresholds (shared across banks), followed by
one 256-entry frequency array per bank/class. The runtime adapter reconstructs
unused predictor fields and unused entropy threshold/bias slots as zero; none
of those zeros are serialized as fields.

Bounds: 0–4 models; 1–60 menu entries; up to16 entropy banks, 2–32 classes per
bank and at most128 total tables. Fixed selector edges2,4,8,16,32. Geometry is
32-aligned, X<=1024 and Y*Z<=65536. Models may have different neighborhoods and
weights; menu entries reference models and entropy banks independently.
Two entries with the same model/predictor must agree on its two input weights
(the reconstruction palette is indexed by that pair). Use another model
record for a different fitted combination. Mode0 means correction on a fixed
base; mode1 means learned input weight. Both are functional, not legacy modes.

## Generic numerical compaction

One implementation treats every numerical array alike:

- Its full size is derived from the semantic schema.
- Transmit a retained-prefix count (varint); the omitted suffix is zero.
- Values0–15 are singleton labels. Larger unsigned16-bit values use label
  `12+floor(log2(value))`, with the exact low exponent bits in a bypass stream.
- All label records share **one numerical rANS state group**. Before
  coding, choose literal labels or wrapping16-bit deltas within each record
  (pairs of label bytes, an odd last byte unchanged). This is the retained
  measured delta transform, not a claim to delta-code the original integers.
- One byte: bit2 specifies delta/no delta; low two bits specify raw(0),
  fixed-table rANS(1), or transmitted-table rANS(2). Other values are invalid.
  It is followed by the encoded label byte count and bytes.
  Two global frequency tables in `header_tables.h` are format constants:
  bit2 selects the appropriate plain/delta table. No table ID or frequencies
  are transmitted in fixed mode. The shared eight-lane coder still stores its
  64-byte state group. These constants must not be retrained independently by
  encoders; changing them changes the format.
  The default encoder only chooses between these two fixed tables and raw
  labels. Optional transmitted-table coding remains available to future
  encoders, using the existing compact frequency-table representation.
  There are no per-preset or predictor-specific tables.
- Append low-first bypass bits; their count follows from the labels. No
  separate bit count. Final padding must be zero.

The implementation has no XOR, reference search, RLE, signed mapping, alternative
integer-compaction formats, standalone packing or field-specific coding menus.
Controls are plain varints; there is no separate control entropy table.
The selector payload has its own constant/context/packed-LZ codec described
in `SELECTOR_FORMAT.md`; it is payload, not a numerical header record.

`header.c` shares one schema walk between encoding, shape derivation and
reconstruction. C callers retain the straightforward native model/table
layout, while wire compression omits redundant fields. No Zstd is linked.

## Validation and comparison

See `experiments/header_cleanup.md`. The native frozen-frame importer exists
only in the benchmark: it is not linked into the production library. Old
experimental bytes need conversion/re-encoding, not a compatibility decoder.
