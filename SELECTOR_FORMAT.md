# Retained selector payload

Selector IDs are uint8. Their decoded count is known from the containing
frame's dimensions/block edge, not repeated in this payload. The internal
API accepts counts from1 through INT32_MAX-1 (also bounded for size_t arithmetic).
All multi-byte fixed integers are little-endian. Varints are unsigned7-bit
groups, low group first, continuation bit0x80.

## Map tag

One byte, followed by:

|Tag|Payload|
|---|---|
|0|One constant ID; decoded count copies.|
|1|Previous-ID context representation.|
|2|One packing-width byte (1..8), then LZ representation of packed IDs.|

Encoder trials context and LZ, retains the smaller complete payload (context
wins ties). Constant maps are detected first. There are no mode-search settings,
local switches, higher-order contexts, literal contexts or alternate LZ formats.
The decoder only executes the selected mode.

## Shared small-stream entropy representation

Both map modes use precisely the same backend. Stream decoded count is supplied
by their structural fields. One stream tag:

-0: raw bytes.
-1: constant byte.
-2: sparse normalized frequencies: used-symbol count(varint), increasing symbol
  gaps(varints, first gap from0, later from previous+1), normalized frequencies
  for all but the last symbol(varints). Final frequency=4096-sum(previous).
-3..6: alphabet size(varint), then reconstruct frequencies from respectively
  uniform, reciprocal, reciprocal-square, or geometric integer weights.

For rANS tags2..6, append encoded size(varint), four-byte final state, byte
renormalization payload. Scale12, total4096, lower bound2^23; final decode state
must equal2^23 and all bytes must be consumed. Raw/constant have no ANS state.
Predefined integer weights and normalization use the same deterministic shared
`vc_rans_normalize` algorithm as the existing bulk coder. Exact weights:
4096; floor(65536/(s+1)); floor(65536/(s+1)^2); 65536>>min(s,16), each floored
to at least1 before normalization. No learned or per-channel parameters.

## Context mode

Alphabet size(varint), used-context count(varint), followed by ascending records:
context ID(varint), decoded stream count(varint), shared entropy representation.
Previous ID initially0. Symbols are gathered by previous ID; decode rebuilds
the causal original sequence. All streams must be exactly consumed; all IDs
must be below alphabet size. No per-record alternate coding paths beyond the
shared stream tags above.

## Packed LZ mode

IDs packed low bits first using minimum width for the maximum ID. Packed byte
count is ceil(ID count*width/8); unused final bits must be zero.

Write match count(varint), literal count(varint), three entropy streams:
literal-run codes (matches+1), match-length codes(matches), distance codes(matches).
Zero-length streams are absent. Then extra-bit byte count(varint), extra bits,
and the literal entropy stream if nonempty.

Length magnitude: value v maps to code floor(log2(v+1)), with code low bits of
v+1 as extra bits. Match lengths store length-4. Each sequence consumes literal
length, match length, then distance extra bits. Final sequence has only literals.
Distance codes0..2 use a move-to-front cache initially[1,4,8]; code3+k stores
distance=2^k+extra(k bits). Extra bits are low-first, zero-padded at the end.
Matches copy forward, allowing overlap; maximum distance65536, minimum match4.

Encoder-only: four-byte multiplicative hash, chain search up to256 candidates,
greedy longest match, nearest match on ties. This search is not needed by decode.

## NCF1 integration

`NCF1 + uint32(compact_header_size) + compact_header + selector_payload + voxel_payload`

The header contains the selector payload's byte length; geometry determines its
decoded ID count. Every map uses this one codec, including constant maps. There
is no separate singleton field or marker253/254/255 compatibility path.
The compact model/entropy header uses the shared rANS implementation, not Zstd.
See [header format](HEADER_FORMAT.md). Earlier experimental NCF1 variants are
intentionally unsupported; only legacy Delta3D compatibility is retained.

Implementation: `src/ng/selector.c`, only C and shared storage/normalization.
All public/internal routines return failure on malformed input/allocation
failure; no asserts or process exits in codec paths. No globals or thread-local
state. Tests exercise malformed/truncated streams plus complete API decoding.
