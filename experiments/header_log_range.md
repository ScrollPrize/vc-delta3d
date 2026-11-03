# Log-range integer compaction with zero-tail skipping

2026-11-02. Native C experiment; no production format/default change.

## Matched complete-header results

576 frozen headers: 18 groups × 8 chunks × 4 presets. Same arrays, residual payloads,
earlier compaction choices, reference/delta choices and family-pooling options.
All sizes include retained-count controls, range labels, exact within-range bits,
codec/transform tags, rANS tables and states. These are actual round-tripped complete
headers, not entropy estimates. No timing measurements were made.

| Configuration | Mean bytes/header | Saving vs baseline |
|---|---:|---:|
| Current compact/pool baseline (109) | 6088.5035 | — |
| Add unsigned log-range compaction (111) | **5975.0747** | **113.4288 B / 1.863%** |
| Also allow zigzag log-range compaction (112) | 5975.0747 | Same |

Both additions improve 420 headers, leave 156 unchanged, and regress none.
All 576 baseline sizes exactly match the previous saved result. Every individual
header has identical size in 111 and 112; no signed log mode is selected. The saved
bytes are 0.016612% of the complete compressed data, not 1.863% of the volume payload.

Keep unsigned log compaction as the simpler promising experimental candidate.
This does not establish that earlier varint/zigzag/fixed-width alternatives can
be removed: they remain available and are used when smaller.

## Exact compaction

For each schema-known integer array:

1. Skip its trailing zero elements, as before.
2. Store the unknown retained-element count in the separate control stream.
   Original array lengths are derived from the schema; no duplicate length list.
3. Emit one range label per retained integer. Values 0–15 have individual labels;
   16–31, 32–63, ..., 32768–65535 each have one label for uint16 (28 labels total).
4. Encode the exact offset inside each larger range as bypass bits, packed across
   array boundaries without per-array byte padding. No values are approximated.
5. Existing byte/word delta and previous/first-record references remain available
   on the range-label arrays before the existing entropy/raw/bit-packing choice.
   Offset bits are literal bypass bits, not independently predicted or compressed.

The decoder reads retained counts, recovers range labels, derives the bypass-bit
count from those labels, reconstructs each exact integer, then restores the omitted
zero suffix. No end-of-array symbol or redundant bypass length is needed.

Signed trial 112 first zigzags values before finding their ranges; unsigned trial
111 does not. Earlier signed/unsigned compaction modes are untouched in both.
The new candidate is compared using complete section-wire size, including codec
and length fields. There is no per-channel special case.

Shared transform helpers in `header_compaction.h` serve both the existing compact
bytes and new range-label arrays. `header_log_range.h` contains the new compaction;
the existing rANS implementation is reused.

## Which log modes won?

Each of the 420 improved headers selects one log-coded group. These are selected
decoder paths, not encoder trial counts.

| Delta on labels | Reference | Selected groups |
|---|---|---:|
| None | None | 25 |
| Byte | None | 13 |
| Word | None | 345 |
| None | Previous record | 18 |
| Word | Previous record | 1 |
| Byte | First record | 1 |
| Word | First record | 17 |

## Per-group means (bytes/header)

| Group | Baseline | Add log-range |
|---|---:|---:|
| CT L0 | 1013.28 | 1006.47 |
| CT L1 | 1167.94 | 1158.78 |
| nx L3 | 8019.53 | 7924.22 |
| nx L4 | 6925.72 | 6885.59 |
| ny L3 | 7670.75 | 7579.38 |
| ny L4 | 6831.84 | 6822.25 |
| width L3 | 8700.69 | 8498.91 |
| width L4 | 4638.97 | 4432.94 |
| compound presence L3 | 8010.44 | 7765.44 |
| compound presence L4 | 5260.12 | 5019.03 |
| compound sheet_grad_mag L3 | 2598.88 | 2598.88 |
| compound sheet_grad_mag L4 | 2982.75 | 2982.75 |
| compound sheet_nx L3 | 8138.34 | 7988.59 |
| compound sheet_nx L4 | 7799.69 | 7695.81 |
| compound sheet_ny L3 | 8138.59 | 8012.81 |
| compound sheet_ny L4 | 7925.19 | 7809.16 |
| compound width L3 | 8314.97 | 8128.16 |
| compound width L4 | 5455.38 | 5242.19 |

## Validation / reproduction

All 1728 full-header round-trips pass. The 1230 synthetic cases pass release and
ASan/UBSan: all byte/uint16 values, uint32 range boundaries, every new mode, empty
records, zero-only records, zero tails, signed cases and padding. Thirty-nine
representative full-header sanitizer checks cover both new modes on one slower
frame per group, plus the empty-reference regression on all three modes. All four
CTest tests pass. This is not a claim of a full-corpus sanitizer sweep.

The first run (`header_log_range_20261102`) is incomplete and invalid for results.
It caught an empty-reference bug introduced during shared-helper extraction:
a NULL reference means adjacent-value delta in the existing primitive, not an
empty all-zero reference. A non-NULL empty-reference sentinel now preserves the
original semantics. Regression tests include empty records before nonempty ones,
and the full baseline sizes/bytes reconstruct correctly after the fix.

From the delta3d checkout:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_log_range_v2
/data/hendrik/compression_experiments/villa4/builds/header_log_range_v2 --context-self-test
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_log_range_v2 \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_log_range_v2_20261102 \
  --repeats 0 --cpu 99 --variants 109,111,112
```

Data and generated full-channel report:
`/data/hendrik/compression_experiments/villa4/encoder_speed/header_log_range_v2_20261102/`.
Python only orchestrates/reports; compression and reconstruction are native C.
Zstd is only the historical frame reader/control, not part of the replacement.
