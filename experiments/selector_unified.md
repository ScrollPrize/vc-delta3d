# Unified small-state selector entropy backend (2026-11-03)

Test replacing only the context coder's bulk rANS stream backend with the
same small-state entropy backend used by LZ. Context assignment, map inputs,
LZ parser and whole-map exact-byte selection are unchanged. Shared context
construction/reconstruction now accept entropy callbacks; no duplicated codec.
No production integration/default change. The old backend remains the control.

Small backend: single32-bit state, byte renormalization, sparse transmitted
counts or four predefined distributions, raw/constant alternatives. Both
context and LZ use exactly the same encode/decode functions. The encoder
tries all those entropy representations; this is not a speed optimization.

## Results

Same576 saved maps (18 groups ×8 chunks ×fast/slow ×2³/4³ blocks).
Mean selector bytes/chunk, all framing included. Not full-chunk sizes.

|Configuration|Zstd|Existing context/LZ choice|Unified choice|Change|
|---|---:|---:|---:|---:|
|Fast2³|19887.41|18428.01|18349.24|-0.43%|
|Fast4³|1779.34|1619.90|1545.78|-4.58%|
|Slow2³|39540.81|37881.72|37740.88|-0.37%|
|Slow4³|4789.95|4706.35|4570.16|-2.89%|

491 maps smaller,85 equal,0 larger. CT L0 unchanged (mostly LZ/uniform),
including means fast4=14.25,slow4=147.875. Examples for4³:

|Channel|Fast old/new|Slow old/new|
|---|---:|---:|
|nx L3|2096.75 /2004|6572.25 /6405.25|
|nx L4|1342.75 /1264.875|5479 /5308.5|
|width L4|1342.75 /1264.875|3021.375 /2891.875|
|Compound presence L3|2293.75 /2198.625|6280.625 /6106.625|
|Compound grad_mag L3|2293 /2198.5|2293 /2198.5|

## Speed tradeoff

MiB/s of **unpacked selector IDs**, not voxel data. O3, three timing batches
of three repetitions, per-map median; aggregate bytes divided by summed median
times. Eight pinned workers; loaded host, indicative not isolated timings.
Includes allocation and exact-output validation. Sanitizer corpus launched
after the timed run completed.

|Configuration|Old encode|New encode|Old decode|New decode|
|---|---:|---:|---:|---:|
|Fast2³|76.4|33.7|312.3|162.9|
|Fast4³|107.1|39.1|316.8|174.8|
|Slow2³|76.8|31.8|307.3|152.3|
|Slow4³|85.9|32.9|308.2|153.7|

Standalone context decode: old294–305 vs new147–160MiB/s. The new backend
has one serial state and byte renormalization, whereas the existing backend
uses eight states/word renormalization. It also uses scalar per-symbol buffer
output; speed loss cannot be attributed exclusively to state count without
profiling. Encoder adds trials of predefined distributions. No attempt here
to optimize these paths or prune the distributions.

Conclusion: genuine simplification and modest size gain, substantial current
speed regression. Keep experimental; do not silently replace the existing
combination. Unifying selector entropy does not remove bulk rANS needed by
the rest of the volume codec.

## Validation / reproduce

3456 corpus round-trips (six measured configurations ×576), ASan/UBSan corpus,
1400 synthetic checks. All2304 old control sizes unchanged. Full per-channel
sizes and timings in `encoder_speed/selector_unified_20261103/report.md` and
`results.jsonl` under `/data/hendrik/compression_experiments/villa4`.

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/selector_payload.c \
 /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
 -lzstd -lm -pthread -o /data/hendrik/compression_experiments/villa4/builds/selector-unified
/data/hendrik/compression_experiments/villa4/builds/selector-unified --self-test
SELECTOR_UNIFIED_ONLY=1 python3 benchmark/compare_selector_payload.py \
 --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
 --output /data/hendrik/compression_experiments/villa4/encoder_speed/selector_unified_20261103 \
 --library /data/hendrik/compression_experiments/villa4/builds/hybrid-smoothed-pyramid-v1/libvolcodec_ng_hybrid.so \
 --binary /data/hendrik/compression_experiments/villa4/builds/selector-unified \
 --reuse-results /data/hendrik/compression_experiments/villa4/encoder_speed/selector_payload_20261103/results.jsonl \
 --workers 8 --repetitions 3
```

`SELECTOR_UNIFIED_ONLY` skips unrelated historical modes, not any data. Native
encode/decode only; Python orchestration. Sanitizer: existing sanitizer library,
O1/debug, address/undefined sanitizers, stack32768KiB, repetitions0; output
`encoder_speed/selector_unified_sanitize_20261103`.
