# Spatial selector payload, separate from general metadata

2026-11-03. Current **fast** and **slow** (historical Fast and RC1, not
historical Slow/current medium), each at 2³ and 4³ predictor-choice blocks.
128³ chunks, 18 channel/scale groups × eight frozen inputs × four configs.
The chunk size remains 128³; only the selector grid changes.

## Outcome

Packing is now restricted to selector payloads in experimental header mode158.
General numerical arrays/controls cannot select packing. All 576 frozen headers
produce exactly the same sizes as156 (6061.875 B/header), with exact round-trips.
Model/frequency arrays still share one numerical pool. Production formats/defaults
are unchanged; this is a tested separation in the candidate implementation.

Mean **selector payload bytes per chunk**, including its mode tag, probability
tables and state, but excluding general metadata and voxel residuals:

| Preset | Choice edge | Byte rANS | Packed rANS | Byte Zstd3 | Packed Zstd3 | Separate-control RLE |
|---|---:|---:|---:|---:|---:|---:|
| fast | 2 | 26726.56 | 19645.84 | 32095.46 | 19887.41 | 38040.54 |
| fast | 4 | 2286.95 | 1809.94 | 2720.44 | 1779.34 | 3228.45 |
| slow | 2 | 47555.27 | 39773.60 | 54027.96 | 39540.81 | 65054.22 |
| slow | 4 | 5457.47 | 4884.31 | 6478.72 | 4789.95 | 7930.05 |

Packed-rANS saves26.49%,20.86%,16.36%,10.50% respectively versus byte-rANS.
RLE is not a good aggregate replacement. Packed-Zstd is close overall, with
large benefits on some structured CT maps, but is only a comparison: the aim
remains removing the Zstd dependency. Keep byte-rANS as a possible fallback:
packing is not better on every map (slow CT L1 at4³ is93.00 vs129.38 B).

CT details (same units; CT L1 fast and slow2³ maps are uniform):

| Group | Preset | Edge | Byte rANS | Packed rANS | Packed Zstd3 |
|---|---|---:|---:|---:|---:|
| CT L0 | fast | 2 | 2965.50 | 790.38 | 61.00 |
| CT L0 | fast | 4 | 374.75 | 160.50 | 24.62 |
| CT L0 | slow | 2 | 3990.50 | 1755.75 | 1304.00 |
| CT L0 | slow | 4 | 528.75 | 357.50 | 178.62 |
| CT L1 | fast | 2 | 2.00 | 2.00 | 2.00 |
| CT L1 | fast | 4 | 2.00 | 2.00 | 2.00 |
| CT L1 | slow | 2 | 2.00 | 2.00 | 2.00 |
| CT L1 | slow | 4 | 93.00 | 129.38 | 94.75 |

Smaller blocks have eight times as many IDs:262144 vs32768. Map storage alone
cannot decide the best block size: smaller blocks also change voxel residuals.
Adaptive split-tree signaling and irregular traversal are NOT modeled here.

## Method and verification

- Native `benchmark/selector_payload.c` reuses the existing header parser,
  bit packer, rANS, separated-role RLE and Zstd control. No Python compression.
- The native variable-edge encoder is the existing frozen research library.
  Every one of288 newly encoded4³ frames must byte-match its frozen release
  counterpart; all matched. Thus fast/slow definitions were not silently changed.
  The same fitting/settings encode2³. All576 full frames decode exactly and
  decoded inputs match their stored SHA256 hashes.
- Selector IDs are extracted without changing their values, predictors, models,
  entropy tables or voxel residuals. Maps follow their existing raster order.
- All methods have a one-byte codec tag. Packed methods add the bit width.
  Uniform maps use the same two-byte tag/value representation for every method;
  no full array is transmitted. Count/grid shape is derived from chunk geometry.
- Packed-rANS packs IDs using ceil(log2(maxID+1)) bits then byte-rANS. No signed
  mapping or delta. RLE separates run controls from stored values and uses the
  existing backend. Zstd level3 is strictly a comparison, not a dependency added
  to the proposed format.
- 2880 initial selector round-trips plus timed repetitions; 360 ASan/UBSan
  selector round-trips (all18 groups × fourconfigs × fivemethods). CTest4/4.
- Release C11/O3, no numeric changes. Eight independent pinned worker processes;
  each codec is single-core. Three timing groups, three repetitions/group.
  Timings include allocation and output checking; external load was recorded,
  not rejected. They are indicative selector-only speeds, not release codec
  speed claims. Report includes per-group sizes and aggregate selector throughput.

## Files and reproduction

Code: `benchmark/selector_payload.c`, `benchmark/compare_selector_payload.py`;
general-header restriction: `header_typed.h`, variant158 in `header_codecs.c`.
Historical experiments remain reproducible. No public API change.

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/selector_payload.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/selector_payload
python3 benchmark/compare_selector_payload.py \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/selector_payload_20261103 \
  --library /data/hendrik/compression_experiments/villa4/builds/hybrid-smoothed-pyramid-v1/libvolcodec_ng_hybrid.so \
  --binary /data/hendrik/compression_experiments/villa4/builds/selector_payload \
  --workers 8 --repetitions 3
```

Outputs (full per-channel report, raw timings, all generated2³/4³ frames):
`/data/hendrik/compression_experiments/villa4/encoder_speed/selector_payload_20261103/`.
General-header restriction verification:
`/data/hendrik/compression_experiments/villa4/encoder_speed/header_payload_only_20261103/`.
The Python runner's affinity pinning is Linux-specific; codec code remains portable C.
