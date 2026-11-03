# Minimal header features — 2026-11-02

Scope: native C header-compression experiments in `benchmark/header_*.h` and
`header_codecs.c`; Python only orchestration/reporting. No production changes.
Same 576 frozen headers: 18 groups × 8 chunks × four presets. Release C11/O3,
no fast-math. Actual complete encoded sizes and exact round-trips, not estimated
entropy. Verification-only runs: no speed claims. Portability: no new SIMD or
platform dependencies. Zstd reads the frozen originals, not the replacement.

## Factorial test

All 16 combinations of signed mapping, packing, delta and RLE were tested.
Common: schema-derived sizes, unsigned control varints, raw/rANS/constant
fallbacks, log-range with zero-tail skipping. RLE has separate control/value
streams. Delta+RLE admits the existing delta-RLE mode; no local references,
pooling or old numerical compaction alternatives are available.

Packing here is fixed-width packing of byte streams (range labels/selector
IDs), optionally followed by rANS. It is NOT a competing compaction of the
original uint16 arrays. Log-range bypass bits remain exact packed bits in
every log variant, including the ones with optional packing disabled.

| ID | Extra features above unsigned log + common backend | Mean B/header |
|---|---|---:|
| 128 | None | 7044.219 |
| 129 | Sign | 7016.191 |
| 130 | Pack | 6469.389 |
| 131 | Sign, pack | 6441.361 |
| 132 | Delta | 6643.507 |
| 133 | Sign, delta | 6615.479 |
| 134 | Pack, delta | 6066.431 |
| 135 | Sign, pack, delta | 6038.392 |
| 136 | RLE | 6981.689 |
| 137 | Sign, RLE | 6953.797 |
| 138 | Pack, RLE | 6462.861 |
| 139 | Sign, pack, RLE | 6434.969 |
| 140 | Delta, RLE | 6578.559 |
| 141 | Sign, delta, RLE | 6550.667 |
| 142 | Pack, delta, RLE | 6059.297 |
| 143 | Sign, pack, delta, RLE | 6031.405 |

Controls without log-range: RLE-only 7524.028 B; RLE+delta 7496.866 B;
raw/rANS 8576.773 B. RLE itself retains implicit zero tails. Raw/rANS is a
diagnostic lower-complexity control without explicit tail skipping, not the
recommended replacement. These comparisons do not establish that RLE is
inherently inferior to all possible log-free designs, only these implementations.

## Further removals from 134

| ID | Change | Mean B/header |
|---|---|---:|
| 147 | Remove constant-value codec | 6066.431 |
| 148 | Remove control-array varint codec | 6260.323 |
| 149 | Remove both | 6260.323 |
| 150 | Both removed; always word-delta log labels | 6331.618 |
| 151 | From 147: remove standalone packed mode | 6076.809 |
| 152 | From 147: remove packing followed by rANS | 6632.872 |
| 153 | From 147: always word-delta log labels | 6138.667 |
| 154 | From 151: always word-delta log labels | **6148.250** |

For minimum features, choose **154**: no signed mapping, no RLE, no constant
codec, no standalone packed mode, and no choice of delta transform. Keep
zero-tail skipping + unsigned log ranges, fixed wrapping word-delta of the
label stream, packing before rANS, rANS/raw backend, and control varints.
This remains 24.918 B/header smaller than Zstd in aggregate (not every channel).
151 is the modestly larger-feature alternative: the optional delta transform
saves 71.441 B/header (6076.809 B total). Both remain experimental.

The fixed-delta variant uses the existing word-delta helper on the byte label
stream, exactly as before; it does not claim to delta-code the original integers.
Wire mode tags remain in the experimental framing for compatibility with the
other trials; a production-only fixed-mode format could later omit that tag.

The native wire still uses ordinary varints for variable payload lengths;
148–150 remove the competing control-array codec, not all format varints.
Controls are schema/count fields, not original numerical arrays.

Historical controls: full option search 5975.075 B/header, previous recommended
126 6038.392 B/header, old pre-log baseline 6088.503 B/header, Zstd 6173.168 B/header.

147 uses only tags rANS (816), literal pack (468), pack+rANS (437), control
varints (1688), log-range (1152) across all headers. Constant coding is never
selected and can go. Raw fallback remains valid for other inputs.
Its min/median/max header sizes are 555 / 6665 / 11489 B.

## Reproduce

From `local/codec_sources/delta3d-v020`:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_minimal_last
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_minimal_last \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_minimal_pack_20261102 \
  --repeats 0 --cpu 99 --variants 151,152,153
```

All outputs under that encoder_speed directory, including full per-group
tables and per-header native selection telemetry:

- `header_minimal_0_20261102`: 128–132
- `header_minimal_1_20261102`: 133–137
- `header_minimal_2_20261102`: 138–142
- `header_minimal_3_20261102`: 143–146
- `header_minimal_final_20261102`: 147–150
- `header_minimal_pack_20261102`: 151–153
- `header_minimal_last_20261102`: 154

All variants have 576 exact round-trips. ASan/UBSan: 1230 synthetic tests and
486 full-header checks (18 groups × variants128–154). Production CTest 4/4.
Historical alternatives stay in the research harness for reproduction; these
restricted candidates do not search/emit them. No production format is changed.
