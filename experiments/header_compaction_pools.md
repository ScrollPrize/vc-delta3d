# Compaction before transforms and shared entropy pools

2026-11-02. Native C experiment; production framing/presets unchanged.

## Matched result

Same 576 frozen headers (18 groups × 8 chunks × 4 presets), identical volume
residual payloads. Sizes include all pool references, compacted lengths, codec
tags, entropy tables and states. No timing was measured in these runs.

| Variant | Mean complete header bytes | Header-compressor rANS table bytes | rANS state bytes |
|---|---:|---:|---:|
| Zstd control (0) | 6173.17 | — | — |
| Earlier simplified control (88) | 6172.32 | 433.16 | 108.89 |
| Corrected typed control (106) | 6337.04 | 398.57 | 110.89 |
| Generic compaction first (107) | 6090.13 | 388.77 | 108.44 |
| Shared pools only (108) | 6333.81 | 398.69 | 110.89 |
| Compaction + shared pools (109) | **6088.50** | **388.77** | **108.44** |

All 576 typed-control sizes exactly match the preceding run. Relative to that
control, compaction improves 505 headers, leaves 71 unchanged, regresses none.
Pooling alone improves 172, leaves 404 unchanged, regresses none. Combined
improves 521, leaves 55 unchanged, regresses none. Shared pools are selected on
188 combined headers; the additional mean saving over compaction alone is 1.63 B.

Combined saves 83.82 B/header (1.36%) over the earlier simplified control, or
approximately 0.0123% of complete compressed data. This is a header improvement,
not a comparable percentage reduction in volume payload size. It is not a win
on every individual header or group against Zstd/the earlier control.

There IS entropy-table overhead. The previous regression was not caused mainly
by increased table overhead: the worse typed control already stored fewer table
bytes than the smaller simplified control. Pooling remains useful, but here its
incremental benefit is small. Table bytes above refer to the compressor's own
tables, not the volume frequency-count arrays being compressed.

## Generic compaction

`header_compaction.h` receives record shapes and element widths, not channel or
frequency-table identities. It can trim trailing zero elements and encode the
remaining elements as fixed-width, unsigned varints, or zigzag varints. The decoder
knows the output record size and fills the missing tail with zeros.

Only the variable compacted byte lengths are stored; original array lengths and
retained element counts are not repeated. Length controls are compressed separately
from numerical bytes. After compaction, optional previous/first-record references
and byte/word deltas act on the COMPACT representation. The full encoded candidate
is measured, rather than choosing that global transform from a proxy cost.

Existing raw-integer transforms and local-reference choices are retained. These
are additions to the candidate set, not a change to the frozen control. Local
reference optimization within the older pooled candidate still has its documented
standalone-record objective; this experiment does not claim to solve that separate
optimization issue.

## Pool references / framing

Compatible equal-width numerical arrays can reference one shared pool (a schema
array bit mask). The pool carries its transform/codec once. Its values are concatenated
in schema order and reconstructed into the original arrays using known shapes.
There is one table/state set per selected entropy stream, not one copy per array.
This is stream pooling, rather than independent payloads that merely reuse a table.

Current schema candidates are the model-parameter and frequency-count families.
Controls and selector IDs are not forced into those numerical pools. RLE run controls
remain separate from RLE values inside the pooled codec. Width compatibility and
the existing 128-record bound are generic checks; no channel-specific merge rules.

The shared family encoder is used for both standalone and pooled candidates. The
merge compares complete wire sizes, including its reference/codec bytes, against
separate encodings. Thus pooling is available without forcing an unfavorable shared
distribution. It can also select raw/packed coding when an entropy table is wasteful.

## Per-group header sizes

Mean bytes/header over 8 chunks × 4 presets. Full frame ratios and field accounting
are in the machine-readable run and generated report linked below.

| Group | Zstd | Earlier simplified | Compaction + pools |
|---|---:|---:|---:|
| CT L0 | 932.97 | 1201.22 | 1013.28 |
| CT L1 | 1225.75 | 1412.66 | 1167.94 |
| nx L3 | 8200.81 | 8063.28 | 8019.53 |
| nx L4 | 7283.34 | 6928.97 | 6925.72 |
| ny L3 | 7828.25 | 7737.69 | 7670.75 |
| ny L4 | 7326.03 | 6914.25 | 6831.84 |
| width L3 | 8785.47 | 8708.28 | 8700.69 |
| width L4 | 4389.78 | 4637.12 | 4638.97 |
| compound presence L3 | 7842.97 | 8115.62 | 8010.44 |
| compound presence L4 | 5146.28 | 5392.22 | 5260.12 |
| compound sheet_grad_mag L3 | 2527.97 | 2686.12 | 2598.88 |
| compound sheet_grad_mag L4 | 2973.88 | 3108.06 | 2982.75 |
| compound sheet_nx L3 | 8259.91 | 8200.59 | 8138.34 |
| compound sheet_nx L4 | 8102.78 | 7867.03 | 7799.69 |
| compound sheet_ny L3 | 8277.94 | 8204.03 | 8138.59 |
| compound sheet_ny L4 | 8203.28 | 7999.06 | 7925.19 |
| compound width L3 | 8458.06 | 8364.25 | 8314.97 |
| compound width L4 | 5351.56 | 5561.28 | 5455.38 |

## Reproduction / validation

From the delta3d checkout:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_pools
/data/hendrik/compression_experiments/villa4/builds/header_pools --context-self-test
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_pools \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_compaction_pools_20261102 \
  --repeats 0 --cpu 99 --variants 0,88,106,107,108,109
```

Zstd is only the historical-frame reader and comparison control; replacement
encoding/decoding is native C. Python only orchestrates and reports.

- All six modes round-trip all 576 headers exactly.
- 558 synthetic round-trips pass in release and ASan/UBSan, including all 27
  compaction/reference/delta configurations across 8/16/32-bit elements, zero
  tails, constants, empty records, repeated values and pseudorandom values.
- ASan/UBSan full-header checks pass for all three new modes on one slower-preset
  frame per group (54 checks). This is not a full-corpus sanitizer claim.
- Four CTest tests and API/AVX2 smoke pass. No production path changed this turn.

Data: `/data/hendrik/compression_experiments/villa4/encoder_speed/header_compaction_pools_20261102/`
(`results.jsonl`, `report.md`). Pool bytes are attributed to the first member family
in the existing field-size telemetry; zero bytes for another member mean shared
storage, not missing values. Encoder/decoder speed and cross-platform performance
have not been measured for these experimental paths.

## Selected-compaction audit (2026-11-02)

Decoder-only telemetry over the same 576 headers reproduces 6088.503472 B/header
exactly. Counts below are selected groups, including groups within shared pools;
they are NOT candidate-trial counts or percentages of all header fields.

583 groups use compact-first coding; 11/27 combinations occur. Integer forms:
fixed-width 14 (2.40%), unsigned varints 477 (81.82%), zigzag varints 92 (15.78%).
The dominant complete combinations are unsigned-varint + word-delta / no reference
(364 groups, 62.44%), zigzag-varint + previous reference / no delta (89, 15.27%),
and unsigned-varint + previous reference + byte delta (60, 10.29%).

These groups contain 12,768,036 dense bytes (6,384,018 uint16 elements), of which
3,837,851 elements (60.12%) are implicit zero tails. Compact values occupy 2,658,331
bytes before later transforms/entropy, or 1.044 bytes per retained element. This
does not include other header sections, compact-length controls or framing.

There is no end-of-array symbol in codec 34. Each record's unknown compact byte
length is encoded in the separate length stream. After parsing values to that
boundary, the decoder restores zeros up to the schema-known element count.

Possible simplification (not implemented/tested): direct integer entropy coding
with exact small-value symbols and progressively wider magnitude buckets, plus
exact offset bits inside each bucket. This compresses the probability model,
not the integer values. A full-array version needs no early-end signaling; a
zero-tail version must still signal the shorter payload/count or an explicit end.
No claim that dropping current transforms will retain the measured ratio.

Audit data/report: `/data/hendrik/compression_experiments/villa4/encoder_speed/header_compaction_usage_20261102/`.
Run the command above with output changed to that directory; native JSON includes
`compaction_selected` and the runner reports the selected mode counts.
