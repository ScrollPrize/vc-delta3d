# Minimal LZ selector-payload trial

2026-11-03. Same576 saved frames:18groups ×8chunks × current fast/slow ×2³/4³.
No volume re-encoding or changes to selectors/models/residuals. Native C only.
All tables below are complete selector payload bytes/chunk, not whole-volume
compression. Zstd level3 is a comparison, not a new production dependency.

## Algorithm

`benchmark/selector_lz.h`: greedy four-byte matches, one previous position per
16-bit hash,64KiB maximum distance, minimum match length4. All skipped positions
update the hash table. Copy overlap is allowed and decoded by a forward copy.
No dictionary, chain search, lazy matching, repeat-distance shortcuts or cost
model. Literal-run lengths, match lengths and distances are three separate
unsigned-varint byte streams, each entropy-coded through the shared native blob
helper. Literals have their own stream. Constant and raw fallbacks are shared
with the previous context prototype; no duplicate rANS implementation.

Four variants: raw IDs/packed IDs × ordinary rANS/previous-byte context literals.
Raw context means preceding selector ID; packed context means preceding packed
byte, NOT the preceding unpacked ID. Context comes from the preceding byte of
the full reconstructed sequence, including bytes produced by copies. The
literal context is therefore not reset/skipped at a copy. Decoder pre-expands
context streams, then replays literal/copy sequences.

Outer mode tags, packing width, counts, stream sizes, tables and states are
included. Uniform maps keep the existing common2-byte representation. These are
standalone trials without automatic fallback to a non-LZ candidate; an optional
best-of selector is separately quantified below. No production change.

## CT L0 (eight chunks per configuration)

| Preset | Edge | Packed rANS | Prev1 | Packed Zstd | LZ raw | LZ packed | LZ raw+prev1 | LZ packed+prev1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fast | 2 | 790.38 | 622.75 | **61.00** | 848.75 | 138.75 | 850.25 | 140.50 |
| fast | 4 | 160.50 | 139.88 | **24.62** | 168.12 | 60.38 | 168.88 | 61.38 |
| slow | 2 | 1755.75 | 1634.12 | **1304.00** | 2717.75 | 1723.38 | 2721.88 | 1781.88 |
| slow | 4 | 357.50 | 315.38 | **178.62** | 502.50 | 335.88 | 505.62 | 346.50 |

Packed LZ captures much of fast CT L0's repetition, but does not reach Zstd.
It also does not outperform the earlier prev1 model on slow CT L0. More contexts
on the remaining literals do not help this trial. No claim that the remaining
gap is exclusively framing: this minimal parser also chooses matches without
considering their encoded cost and has only one hash candidate.

## Combined across all18 groups

| Preset | Edge | Packed rANS | Prev1 | Packed Zstd | LZ raw | LZ packed | LZ raw+prev1 | LZ packed+prev1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fast | 2 | 19645.84 | **18494.35** | 19887.41 | 33327.10 | 21542.56 | 33331.85 | 24309.06 |
| fast | 4 | 1809.94 | **1665.24** | 1779.34 | 3399.07 | 2278.44 | 3403.71 | 2741.78 |
| slow | 2 | 39773.60 | **37928.81** | 39540.81 | 57550.51 | 40955.85 | 57582.41 | 47223.32 |
| slow | 4 | 4884.31 | **4722.82** | 4789.95 | 7229.53 | 5168.14 | 7256.33 | 6549.48 |

Do not replace the existing selector codec with unconditional LZ. If selecting
the smallest already measured stream per chunk (existing mode byte suffices):

| Preset | Edge | Best byte/packed/prev1 | Also allow four LZ choices | Strict LZ wins /144 |
|---|---:|---:|---:|---:|
| fast | 2 | 18479.62 | 18435.92 | 8 |
| fast | 4 | 1653.44 | 1649.03 | 1 |
| slow | 2 | 37921.97 | 37898.46 | 6 |
| slow | 4 | 4716.28 | 4712.26 | 1 |

This is exact size selection from complete measured streams, not a measured
speed result for searching all options. It shows the small aggregate additional
value once the earlier choices already compete. Keep as an experiment.

## Speed and validation

Packed-LZ prototype selector throughput, MiB of unpacked IDs/s:

| Preset | Edge | Encode | Decode |
|---|---:|---:|---:|
| fast | 2 | 417.0 | 561.8 |
| fast | 4 | 527.0 | 674.5 |
| slow | 2 | 304.2 | 505.4 |
| slow | 4 | 402.0 | 602.9 |

Same methodology as previous tests: Release C11/O3, three groups of three
repetitions, allocation/checking included,8independent pinned workers; external
load recorded but not rejected. Indicative selector-only speed, not end-to-end
codec throughput or an optimized limit. Full per-channel/method tables and raw
timings are in the generated report. Unchanged5184 baseline sizes confirm that
factoring shared context-bank helpers did not alter previous wire results.

7488 exact stream round-trips plus timed validation;936 ASan/UBSan corpus checks
(18groups ×4configs ×13methods);320 synthetic context/run/LZ round-trips,
including short inputs, full-byte alphabets, runs and overlapping copies.
Production CTest unchanged. No new platform-specific codec implementation.

## Reproduction

From the delta3d checkout:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/selector_payload.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/selector_lz
/data/hendrik/compression_experiments/villa4/builds/selector_lz --self-test
python3 benchmark/compare_selector_payload.py \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/selector_lz_20261103 \
  --library /data/hendrik/compression_experiments/villa4/builds/hybrid-smoothed-pyramid-v1/libvolcodec_ng_hybrid.so \
  --binary /data/hendrik/compression_experiments/villa4/builds/selector_lz \
  --reuse-results /data/hendrik/compression_experiments/villa4/encoder_speed/selector_payload_20261103/results.jsonl \
  --workers 8 --repetitions 3
```

All outputs under `/data/hendrik/compression_experiments/villa4/encoder_speed/selector_lz_20261103/`.
Code is in `benchmark/selector_lz.h`, integrated through `selector_payload.c`;
generic context-bank helpers extracted in `selector_context.h` are shared by
the old context codecs and the new LZ literal coding.
