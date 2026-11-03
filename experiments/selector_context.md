# Conditional rANS and run-length selector payloads

2026-11-03. Follow-up to `selector_payload.md`. Same576 verified full frames,
current fast and slow (formerlyRC1), choice edges2³/4³,18groups ×8chunks.
Reused saved frames: no new fits, selector changes, residual encoding or volume
work. All compression native C; Python only scheduling/reporting.

## What changed

- `context-prev1`: table conditioned on the immediately preceding selector ID.
- `context-prev2`: table conditioned on the previous two IDs (a triple-symbol
  model). Initial history is zero; state continues across raster row boundaries.
- `runs-context-prev1/2`: replace each run with one ID; code those IDs with the
  respective conditional model. Independently code run_length−1 varints with
  their own entropy table. Run lengths never share probabilities with IDs.

The context implementation gathers one stream per observed context. It stores
each context ID, count, and its rANS payload with table/state. Constant streams
store one symbol; raw is the fallback for streams smaller than their rANS encoding.
Decoder first expands context streams, then reconstructs the selector sequence
using the already reconstructed preceding IDs. This is a fully lossless working
prototype, not an optimized integrated causal rANS kernel. All metadata included.

No predictors, channel-specific rules or fitted classifier. Alphabet/context
sizes come from the observed IDs, with only used contexts serialized. The
implementation handles general byte IDs; no fixed three-choice assumption.

## Complete selector-payload bytes/chunk, averaged over all groups

| Preset | Edge | Byte rANS | Packed rANS | Byte Zstd3 | Packed Zstd3 | Prev1 | Prev2 | Runs+prev1 | Runs+prev2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fast | 2 | 26726.56 | 19645.84 | 32095.46 | 19887.41 | 18494.35 | **18423.24** | 18751.22 | 18715.38 |
| fast | 4 | 2286.95 | 1809.94 | 2720.44 | 1779.34 | **1665.24** | 1757.22 | 1874.94 | 1830.17 |
| slow | 2 | 47555.27 | 39773.60 | 54027.96 | 39540.81 | **37928.81** | 38144.10 | 39191.88 | 39445.44 |
| slow | 4 | 5457.47 | 4884.31 | 6478.72 | 4789.95 | **4722.82** | 5206.67 | 4942.60 | 5235.65 |

Prev1 saves5.86%,8.00%,4.64%,3.31% versus packed-rANS. It also wins versus
packed-Zstd on the aggregate in all four configurations, not necessarily every
channel/map. Prev2 has more context tables/states and is only slightly better
for fast2³. RLE+context helps CT but not the aggregate. These are **selector-only**
savings, not equivalent percentage improvements of the whole compressed volume.
No adaptive tree signaling modeled; no production default change.

## CT comparison (same units)

| Group | Preset | Edge | Packed rANS | Byte Zstd3 | Packed Zstd3 | Prev1 | Prev2 | Runs+prev1 | Runs+prev2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CT L0 | fast | 2 | 790.38 | 427.88 | **61.00** | 622.75 | 621.50 | 247.88 | 248.75 |
| CT L0 | fast | 4 | 160.50 | 84.75 | **24.62** | 139.88 | 143.62 | 88.00 | 88.50 |
| CT L0 | slow | 2 | 1755.75 | 1981.88 | **1304.00** | 1634.12 | 1720.12 | 1461.62 | 1441.12 |
| CT L0 | slow | 4 | 357.50 | 271.12 | **178.62** | 315.38 | 360.62 | 376.25 | 366.50 |
| CT L1 | slow | 4 | 129.38 | 133.38 | 94.75 | 111.88 | 133.00 | 164.25 | 148.25 |

CT L1 fast2/4 and slow2 are uniform:2B for every method. Byte-rANS wins the
remaining slow4 CT L1 comparison at93B. Thus no universal unconditional winner.
CT L0 still favors Zstd strongly, suggesting longer repeated patterns not
captured by these one/two-symbol contexts or simple runs. This is an inference,
not a measured decomposition of its LZ matches.

## Indicative speed tradeoff

MiB of unpacked selector IDs/s, not voxel throughput. Allocation and validation
included, three groups of three repetitions, single-core codec calls across
eight independent pinned workers. External load recorded but not rejected;
these are prototype comparisons, not an optimized speed limit.

| Preset | Edge | Packed rANS enc/dec | Prev1 enc/dec | Prev2 enc/dec |
|---|---:|---:|---:|---:|
| fast | 2 | 603.6 / 740.2 | 284.0 / 300.3 | 288.3 / 287.4 |
| fast | 4 | 651.8 / 764.4 | 278.1 / 290.0 | 276.1 / 275.4 |
| slow | 2 | 454.8 / 592.7 | 286.2 / 292.4 | 282.4 / 275.5 |
| slow | 4 | 508.5 / 638.1 | 292.3 / 293.3 | 268.6 / 255.4 |

Prev1 is a promising generic higher-compression candidate, but not yet a free
speed improvement. No additional rANS core is introduced: existing native
kernel handles every context. Optimizing dispatch/stream assembly is future work.

## Reproduction and validation

`benchmark/selector_context.h` implements the new native paths, included by
`selector_payload.c`. `compare_selector_payload.py --reuse-results` reuses frames
instead of rerunning full volume codecs. Run from the delta3d checkout:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/selector_payload.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/selector_context
python3 benchmark/compare_selector_payload.py \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/selector_context_20261103 \
  --library /data/hendrik/compression_experiments/villa4/builds/hybrid-smoothed-pyramid-v1/libvolcodec_ng_hybrid.so \
  --binary /data/hendrik/compression_experiments/villa4/builds/selector_context \
  --reuse-results /data/hendrik/compression_experiments/villa4/encoder_speed/selector_payload_20261103/results.jsonl \
  --workers 8 --repetitions 3
```

5184 exact stream round-trips plus timing-loop checks. All2880 baseline sizes
match the previous run. ASan/UBSan:648 corpus checks (18groups ×4configs ×9modes),
plus160 synthetic cases spanning alphabets1/2/3/17/256, lengths1/2/257/4096,
pseudorandom symbols and long runs. CTest4/4 passes. No platform-specific codec
code added; Linux affinity is in the orchestration script only.
Full per-channel sizes, raw timings and report live at the output directory above.
