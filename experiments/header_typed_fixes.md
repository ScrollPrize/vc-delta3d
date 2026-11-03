# Known shapes, separated RLE contexts, local decisions

2026-11-02. Experimental native C header codecs, not a production change.

## What was corrected

- Array lengths/counts are derived from the model configuration, entropy class
  count, and chunk/block dimensions. Reserved zero fields, derived bank counts,
  absent-entropy sentinels, and uniform selector payloads are omitted.
- Raw and constant substream lengths are also derived, including nested streams
  (variant 106). Encoded lengths remain only where the payload size is unknown.
  The control bootstrap size remains necessary before those controls are parsed.
- Generic element RLE has separate control and value streams, each with its own
  entropy model. The replacement excludes the old mixed-token RLE+rANS choices.
  A zero-tail token finishes the known-size record: no tail count or zero value.
  Middle zero runs store a length but no value. This applies to every field family,
  not just frequency tables. Value counts are derived from the control stream.
- Record-local decisions try every earlier reference and optional within-record
  wrapping subtraction. Choices can be independently coded or pooled with a
  separately coded reference/transform stream. Family-wide choices remain available.

The generic codec receives element widths and record shapes. It does not contain
channel-specific or frequency-table-specific trimming rules. Schema parsing still
necessarily knows which fields determine each shape. The existing parameter-size
formula was extracted into a shared helper rather than duplicated.

## Controlled ablations

Same 576 frozen headers: 18 groups, eight chunks, four presets. Complete header
sizes include all tags, lengths, reference choices, entropy tables and states.
Residual payloads are unchanged. Verification-only runs; no speed claims.

| Variant | Change relative to typed baseline | Mean header bytes |
|---|---|---:|
| 0 | Zstd control | 6173.17 |
| 88 | Existing simplified word-delta control | 6172.32 |
| 98 | Previous typed baseline | 6532.01 |
| 99 | Derived shapes / lean framing only | 6429.76 |
| 100 | Separate RLE roles and implicit tails only | 6482.81 |
| 101 | Record-local decisions only | 6466.53 |
| 103 | Framing + separate RLE roles | 6432.47 |
| 104 | Framing + local decisions | 6363.70 |
| 102 | All three fixes | 6338.72 |
| 105 | Also consider role-RLE when choosing local references | 6344.70 |
| 106 | Variant 102, also derive nested raw/constant lengths | **6337.04** |

Variant 105 is not retained as the preferred candidate. Local decisions currently
minimize each record's standalone encoded cost before pooling; this is not a joint
optimum for the pooled distribution. Trying additional record encodings there did
not improve the complete result. Historical variants remain for reproducibility.

## Remaining gap (not hidden)

The corrected candidate is still 164.72 bytes/header above the simplified control,
approximately +0.024% of complete compressed data. Do not promote it as a better
replacement. It fixes the structural issues but has not recovered the old ratio.

| Part of variant 106 | Mean bytes |
|---|---:|
| Controls | 36.49 |
| Model parameters | 75.56 |
| Frequency counts | 2264.34 |
| Selector IDs | 3948.59 |
| Outer header-section framing | 12.06 |
| Total | 6337.04 |

Parameters + frequency counts are 2339.90 bytes, versus 2190.46 in the earlier
simplified control. Thus most of the remaining gap is that coding/grouping, not
redundant array sizes. Splitting RLE contexts is not automatically a win everywhere:
without local decisions its frequency section gets larger, although controls and
parameters shrink. All three changes were therefore measured separately.

## Reproduction and validation

From the delta3d checkout:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_codecs
/data/hendrik/compression_experiments/villa4/builds/header_codecs --context-self-test
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_codecs \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_typed_known_lengths_20261102 \
  --repeats 0 --cpu 99 --variants 0,88,106
```

Zstd is used only to read historical headers and as the comparison control.
Replacement compression/decompression is C; Python only orchestrates/reports.

Additional matched outputs under the same `encoder_speed` directory:
`header_typed_fixes_20261102` (99–104 ablations),
`header_typed_context_fit_20261102` (105 comparison).

All release variants round-trip all 576 headers exactly. The 72 synthetic tests
cover 8/16/32-bit elements, empty records, constants, zero tails, repeated runs,
random values, separated contexts and local references. All four CTest tests,
modern API/AVX2 smoke test, and 576 frozen-frame raw hashes pass after the shared
parameter-size helper extraction. The full sanitizer ablation run is tracked
separately; do not interpret release verification as sanitizer coverage.
The 72 synthetic tests also pass with ASan/UBSan for the final variant 106.
