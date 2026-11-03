# Matched improvements to the 6,181-byte header baseline

2026-11-02. Experimental native C only; production framing is unchanged.

## Scope and invariant

Start from variant 11 (`struct-xor-delta-rans`), not the later fixed-transform
experiments. Preserve its compact controls, adaptive section codecs, XOR
references, and unsplit entropy section. New representations are optional and
selected only when their complete encoded size is smaller, including tags and
lengths. There is no extra whole-header wrapper or compulsory overhead.

All 576 baseline sizes, section sizes and non-model encoded-byte hashes match
the previous `header_all_compact_delta_20261102` baseline records exactly.
All 18 groups × eight frozen 128³ chunks × four presets are included.

## Results

| Addition to baseline | Mean header bytes | Saved bytes/header | Smaller / equal / larger |
|---|---:|---:|---:|
| None (baseline) | 6181.02 | 0 | 0 / 576 / 0 |
| Byte/word delta references alongside XOR | 6179.76 | 1.26 | 119 / 457 / 0 |
| Delta→RLE→rANS and double-delta→rANS section candidates | 6180.47 | 0.55 | 21 / 555 / 0 |
| Optional separate probability-table transforms | 6170.60 | 10.42 | 256 / 320 / 0 |
| All additions | **6169.36** | **11.66** | **328 / 248 / 0** |

Zstd control: 6173.17 bytes/header. The combined variant saves 0.189% of
header bytes and 0.001707% of complete compressed bytes against the 6181.02
baseline. This is a real but tiny improvement, not the earlier apparent
449-byte component saving from a different serialization.

Combined parts: models 77.887, selectors 3958.750, entropy 2109.616,
controls 23.111 bytes/header. Selectors and controls are unchanged in size.

The semantic table candidate splits the entropy parameter record from its
compact serialized frequency tables. Parameters retain the baseline reference
codec. Table candidates include direct, between-table byte/word wrapping
delta, within-table delta, both, and within-first/between-rest. Table lengths,
scheme and an unused reference tag (254) are counted. The existing whole
entropy representation wins whenever splitting does not pay.

## Reproduction and validation

From the delta3d checkout:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_codecs
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_codecs \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_best_additive_20261102 \
  --repeats 3 --cpu 97 --variants 0,11,64,65,66,67
```

Every encoded header is decoded and compared byte-for-byte with its input.
All five baseline/additive variants also passed the full 576-header corpus
under AddressSanitizer and UndefinedBehaviorSanitizer (one repeat per batch,
CPU 98; output directory `header_best_additive_20261102_sanitize`).
Residual payloads are untouched. The report asserts no complete-header size
regressions and unchanged control sizes for every additive variant.
Zstd is linked to read the frozen input frames and measure the control, not
used by the proposed replacement codecs.

Header-only native timings (three batches × three repeats, CPU 97) were
212/15.8 µs encode/decode for baseline, 363/16.2 µs for the table addition,
875/16.9 µs for all additions. These are exploratory search implementations,
not full-codec speed measurements or production recommendations. Raw timing
samples and per-channel size results are in results.jsonl/report.md alongside
this run; the report includes timing distribution summaries.

Earlier fixed whole-header/group experiments are not matched ablations of
this baseline and must not be used to claim incremental savings from it.
