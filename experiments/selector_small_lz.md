# Small-state LZ and predefined probabilities (2026-11-03)

Experimental only. No production/default/frame change. Native C implementation;
Python only launches tests and produces reports. Same saved 576 selector maps:
18 groups × 8 chunks × current fast/slow (historical RC1) × block edges 2/4.
Only selector storage changes, not predictor decisions or voxel residuals.

## What changed, independently

1. Replace the bulk eight-lane 64-byte rANS state with a single 4-byte state,
   byte renormalization, 12-bit probability precision. Transmitted tables list
   only present symbols (gaps) and their counts; final count is implied.
2. Replace varint length/distance byte streams with magnitude classes and
   separately packed exact extra bits. Three recently used distances have
   codes 0..2; other distances use magnitude classes. Unlike Zstd, this is a
   simple move-to-front cache, with no literal-length-dependent repeat rules.
3. Allow four reconstructed distributions: uniform, reciprocal, reciprocal
   squared, and geometric. Integer weights are normalized by the shared rANS
   normalization routine. Wire cost is just the mode and alphabet size (usually
   two bytes), plus a coded-stream length and four-byte rANS state. No frequency
   counts or learned channel parameters are sent. Raw/constant are also allowed.
4. Search 32 or 256 previous hash matches instead of only the latest one.
   Same minimum match 4 and 64KiB window. Greedy longest match; no optimal
   parser or corpus-specific tuning. Refactored the existing match finder so
   old/new encoders share it; all 7,488 old per-frame sizes remain unchanged.

This copies the useful Zstd *mechanisms*, not its bitstream: compact sequence
codes, recent distances, small predefined distributions, and a better match
search. Earlier LZ copied only the basic back-reference idea.

## CT L0: mean selector bytes per chunk

All eight chunks included, including uniform maps (common two-byte encoding).
These are selector bytes, NOT compressed voxel bytes.

|Method|Fast 2³|Fast 4³|Slow 2³|Slow 4³|
|---|---:|---:|---:|---:|
|Zstd level 3, packed|61.00|24.625|1,304.00|178.625|
|Previous simple LZ|138.75|60.375|1,723.375|335.875|
|Smaller state/table only|79.75|35.625|1,642.75|226.875|
|Then sequence/repeat codes, transmitted tables|89.875|37.50|1,652.125|223.75|
|Same, predefined tables only|96.625|37.125|1,799.75|221.00|
|Same, optional predefined tables|89.25|37.00|1,640.00|217.00|
|32 candidates, transmitted tables|74.125|20.25|1,283.875|178.75|
|32 candidates, predefined tables only|78.25|16.00|1,293.25|160.50|
|32 candidates, optional predefined tables|73.75|16.00|1,265.875|160.375|
|256 candidates, optional predefined tables|**33.625**|**14.25**|**1,179.00**|**147.875**|

Sequence recoding alone is not sufficient: its advantages depend on the match
choices. Predefined-only is not universally best. Actual-byte selection between
transmitted and predefined tables avoids that regression within a fixed parse.

Hard fast chunk ab40c1a9: 256-candidate version 179 bytes vs Zstd 395 at 2³,
100 vs183 at4³. Both use three predefined streams and one raw stream: **zero
transmitted frequency counts**. Their three rANS states total12 bytes. Seven
other fast4 maps remain2 bytes each, so the mean is14.25 bytes.

Across CT L0, the 256-candidate codec selects 20 raw streams, 9 transmitted
tables, 22 uniform, 1 reciprocal and 8 reciprocal-squared tables. Across all
groups: 49 raw,26 constant,1387 transmitted,257 uniform,145 reciprocal,
113 reciprocal-squared,135 geometric. These are entropy substreams, not maps.

## Not an overall winner on predicted channels

Mean selector bytes over all 18 groups:

|Method|Fast 2³|Fast 4³|Slow 2³|Slow 4³|
|---|---:|---:|---:|---:|
|Previous-symbol context|18,494.35|1,665.24|37,928.81|4,722.82|
|Packed Zstd|19,887.41|1,779.34|39,540.81|4,789.95|
|Previous LZ|21,542.56|2,278.44|40,955.85|5,168.14|
|New 32-candidate LZ|20,574.14|1,878.69|40,310.28|5,029.89|
|New 256-candidate LZ|20,503.74|1,869.34|40,292.28|5,026.15|

Keep this as an experimental CT/structured-map candidate, not a replacement
for all selector coding. A future exact-byte choice could retain context coding;
no such selection/integration is claimed here.

## Indicative native speed

MiB/s of **unpacked selector IDs**, not voxel data. Aggregate bytes divided by
sum of per-map median time, three batches of three repeats, pinned workers8.
Allocation/validation included; loaded host, not an isolated speed guarantee.
Uniform maps included identically for every method.

|Method|Fast2 enc/dec|Fast4 enc/dec|Slow2 enc/dec|Slow4 enc/dec|
|---|---:|---:|---:|---:|
|Packed Zstd|472.5 / 784.7|579.7 / 788.6|389.5 / 688.6|460.8 / 661.9|
|32-candidate new|149.8 / 486.0|207.7 / 579.6|118.4 / 395.8|137.6 / 413.7|
|256-candidate new|111.0 / 491.1|190.6 / 581.2|111.5 / 395.9|135.7 / 415.0|

No speed optimization undertaken. The single-state codec prioritizes small
streams and simplicity, not bulk SIMD throughput. More search affects encoding
only; decode uses the same elementary code.

## Reproduce / artifacts

From the delta3d checkout:

```sh
cc -std=c11 -O3 -Werror=incompatible-pointer-types -ffp-contract=off -Isrc \
  benchmark/selector_payload.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/selector-small-lz
/data/hendrik/compression_experiments/villa4/builds/selector-small-lz --self-test
python3 benchmark/compare_selector_payload.py \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/selector_small_lz_v2_20261103 \
  --library /data/hendrik/compression_experiments/villa4/builds/hybrid-smoothed-pyramid-v1/libvolcodec_ng_hybrid.so \
  --binary /data/hendrik/compression_experiments/villa4/builds/selector-small-lz \
  --reuse-results /data/hendrik/compression_experiments/villa4/encoder_speed/selector_payload_20261103/results.jsonl \
  --workers 8 --repetitions 3
```

Full per-channel table, per-frame wire sizes, mode counts, three timing samples
and host loads: `encoder_speed/selector_small_lz_v2_20261103/{report.md,results.jsonl}`
under the experiments root. Build uses existing Zstd solely for frozen frame
reading/comparison; the new codec itself does not use Zstd.

Validation: 12,672 exact corpus round-trips (22 codecs ×576 frames), deterministic
re-encodes, 850 synthetic cases including full byte alphabets and >64KiB streams.
ASan/UBSan variant uses the existing sanitizer static library, `-O1 -g
-fsanitize=address,undefined -fno-omit-frame-pointer`, stack limit32768KiB,
and repetitions0. Output: `encoder_speed/selector_small_lz_sanitize_20261103`.
