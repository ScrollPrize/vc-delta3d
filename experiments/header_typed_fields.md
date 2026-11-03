# Typed logical header experiment

2026-11-02. Experimental C implementation in `benchmark/header_typed.h`;
not a production frame/preset. All 576 frozen headers, all four retained
presets, unchanged volume residual payloads. Zstd/current simplified controls
are rerun on exactly the same corpus.

## Representation

Four field families, with three storage widths (8, 16, 32 bits):

1. Controls, stored as plain 32-bit values in variant 93: dimensions, mode,
   model/choice/bank counts, model configuration bytes widened to 32 bits,
   predictor palette IDs and 32-bit weights, uniform-selector marker,
   parameter lengths and family record counts/lengths. Signed weights retain
   their exact 32-bit representation.
2. Model parameter records, plain 16-bit values: coefficients, bias, optional
   context coefficients, thresholds, class biases. Includes the entropy
   parameter record. Signed and unsigned fields retain their original bits.
3. Frequency records, exactly 256 unsigned 16-bit counts per entropy table,
   including trailing zeros. No pre-serialization table trimming or varints.
4. Selectors, one unsigned byte ID per block, including uniform maps.

The logical-header layer does no varint/RLE/bit packing. A shared family
compressor tries literal/within-record/previous-record/first-record wrapping
subtraction (and reference followed by within-record subtraction), with
generic raw/RLE/bit-pack/rANS alternatives. Variant 93 additionally tries
unsigned and zigzag LEB128 integer compaction, direct or followed by rANS or
RLE+rANS. Integer compaction is a compressor choice, not the logical layout.
No data-channel-specific choices or tuned tables are hardcoded.

Four fixed directory entries each contain a 32-bit plain size, 32-bit coded
size, transform byte, codec byte: 40 bytes total. Record lengths live in the
compressed control family. Different families have independent entropy
streams when rANS is chosen; the encoder can avoid rANS for small families.

Variants 91/92 use literal mixed-width control structs as the first prototype;
93 widens them consistently to 32 bits. Variant 92 adds optional low/high byte
contexts over word arrays; 93 retains that option. Each variant is reported
separately; representation changes are not conflated with context ablations.

## Complete size comparison

| Variant | Mean header bytes | Complete compressed size / raw |
|---|---:|---:|
| Zstd | 6173.17 | 32.563130% |
| Current simplified word-delta | **6172.32** | **32.563090%** |
| Typed families, shared bytes (91) | 7429.09 | 32.623017% |
| Typed families, optional low/high contexts (92) | 7327.95 | 32.618195% |
| Typed families + integer compaction (93) | **6585.67** | **32.582800%** |

Best typed version is 413.35 bytes/header larger (+6.70% of headers,
+0.06053% of complete compressed data). This is not a compression improvement;
retain the current production/experimental baseline rather than adopting it.

## Field breakdown, variant 93

All compressed payload sizes include each family's entropy tables, states and
internal coding metadata. The bootstrap directory is separate.

| Family | Plain bytes/header | Compressed bytes/header | Share of compressed header |
|---|---:|---:|---:|
| Controls and record descriptors | 363.99 | 111.14 | 1.688% |
| Model parameters | 507.18 | 107.01 | 1.625% |
| Volume-residual frequency counts | 24192.00 | 2376.88 | 36.092% |
| Block selector IDs | 32768.00 | 3950.63 | 59.988% |
| Bootstrap directory | — | 40.00 | 0.607% |
| **Total** | **57831.17** | **6585.67** | **100%** |

These are four semantic families, not four scalar fields or four guaranteed
rANS contexts. The existing model schema contains multiple arrays within its
parameter record; this experiment pools them as one numerical family.

## Is the regression extra rANS tables?

No. Measured selected-stream overhead:

| Component | Current simplified | Typed integer | Change |
|---|---:|---:|---:|
| Header compressor frequency tables | 433.16 | 411.89 | -21.27 |
| rANS states | 108.89 | 112.67 | +3.78 |
| Everything else | 5630.27 | 6061.11 | +430.84 |
| **Total** | **6172.32** | **6585.67** | **+413.35** |

Grouping comparable semantic content (the old entropy section also contains
parameter data) gives this exact additive accounting:

- Parameters + frequency sections: +293.44 bytes/header.
- Controls + bootstrap directory: +128.03 bytes/header.
- Selectors: -8.12 bytes/header.

The old table serialization omits trailing zero symbols and writes compact
counts. The new compressor operates on all 256 counts and does not yet have
generic trailing-zero omission. Also, separate control/parameter/frequency
families no longer reproduce the exact old combined-section choices. These
are representation/coding differences, not extra rANS table overhead. The
exact attribution among those individual mechanisms needs further ablation;
the measured family totals alone do not establish it.

## Reproduction and verification

From the delta3d checkout:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_codecs
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_codecs \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_typed_integer_20261102 \
  --repeats 2 --cpu 97 --variants 0,88,91,92,93
```

Three batches × two native repetitions per header; file I/O and verification
outside timing. Mean batch-median header encode/decode: baseline 182/18 µs,
typed shared 1068/114 µs, typed split 1458/119 µs, typed integer 2977/128 µs.
These are unoptimized experiment costs, including conversion back to the old
header for verification, not a production direct-to-typed decoder benchmark.
Per-channel tables and min/median/max timing summaries are in report.md; all
samples/load in results.jsonl. No full-codec speed claim.

Validation completed:

- All variants reconstruct the exact original header bytes, including the
  old compact frequency tables and packed selector stream.
- ASan/UBSan: variants 91/92/93 ×576 headers passed in
  `header_typed_sanitize_20261102` (one repeat, CPU 98).
- Selector bit packing/unpacking was mechanically extracted from `ng_frame.c`
  into shared `src/ng/selector_bits.h`; production and experiment reuse it.
  No codec behavior/format change. Four CTest suites and native modern smoke
  passed; all 576 frozen volume frames decoded to exact expected raw hashes.
- The new typed codec remains experiment-only. Production still reads its
  existing header formats; no dependency removal is claimed.

Shared-selector validation commands:

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j 8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
/data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/vc_delta3d_modern_smoke
/home/hendrik/.venv_las/bin/python benchmark/verify_port.py \
  --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --workers 8 --decode-only
```
