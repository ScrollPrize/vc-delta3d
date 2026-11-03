# Header encoding experiment — 2026-11-02

**Objective clarification:** removing the external dependency is the goal, not
beating Zstd. The recommendation below is historical and superseded by the
[delta+rANS follow-up](header_delta_rans.md), which finds practical Zstd-free
candidates with nearly identical aggregate size.

Production baseline: C port `81657ed`. **No production format or codec changed.**
This experiment is native C in `benchmark/header_codecs.c`; Python only
orchestrates processes and aggregates results.

## Question and conclusion

Can compact, independently encoded headers replace the existing Zstd-compressed
metadata, saving dependency/decoding work without losing compression?

On the complete frozen corpus, **keep Zstd**. It reduces mean header bytes
from 11,710 to 6,173. Raw headers increase total file size by 0.811%.
The closest custom candidate (structured-XOR) increases total size by 0.066%
but costs more encode and decode time. Structured plain fields avoid entropy
coding entirely but increase total size by 0.802%.

This is not a proof that no better custom format exists. In particular these
tests do not design a new specialized frequency-table coding algorithm.
The existing compact NG frequency-table bytes are retained and processed as
part of the header. Models, thresholds, symbols, block choices and payload
remain fixed: search quality cannot confound the comparison.

## What is encoded

The existing NCF1 header holds dimension/configuration fields, fitted NG models,
the predictor palette, bit-packed block selectors, entropy-model parameters,
thresholds and frequency tables. The residual bitstream is separate.

Variants:

- **zstd3:** current complete metadata block with Zstd level 3.
- **raw:** the identical original metadata, stored directly.
- **byte-rle:** PackBits-style runs/literals, up to 128 bytes/run.
- **word-pack:** 32-word blocks, signed 16-bit zigzag and a shared bit width.
- **delta-word-pack:** same, after modulo-16-bit first differences.
- **rans:** existing native eight-lane rANS with compact frequency tables.
- **structured:** unsigned variable-length control fields, zigzag palette
  weights, shared model-prefix references, and independently encoded sections
  (smallest of raw/run/word/delta-word/rANS). Reserved zero fields are implied.
- **structured-rans:** rANS over the whole structured representation.
- **structured-xor:** structured format, but compare models and entropy metadata
  against preceding models by XOR before section coding. This handles differences
  inside a model rather than only its prefix.
- **structured-plain:** variable-length fields and model-prefix references;
  sections stored directly, **no Zstd or other entropy codec**.

Each alternative has a fixed representation known to its decoder. No uncounted
per-chunk winner selection is used. The existing 12-byte outer framing is retained
in total size calculations. These are experimental header payloads, not new
production NCF1 versions. Decoders reconstruct the exact original header before
the unchanged semantic parser; a native redesigned parser could avoid that
intermediate materialization.

## Timing interpretation

GCC 13.3, C11, -O3, AMD EPYC 9475F, one logical CPU (97).
Three batches of 20 encodes and decodes per header, with allocation/free included,
I/O and verification excluded. Host one-minute load was 0.81–1.14.
576 headers = 18 groups × eight chunks × four presets; all represent 128³ uint8
chunks. All ten variants round-trip byte-exactly; all 5,760 header round-trips also
passed the native ASan/UBSan run on CPU 98.

The final-header Zstd encode is approximately 0.10% of complete fast encode time
and 0.008–0.026% for the other presets, using the earlier full-codec benchmark.
This excludes Zstd calls used in candidate search/metadata-size estimates.
One header decompression costs about 0.2% of full decode time. The current
public scalar decoding path reads it twice (dimension check and decode), or three
times when Python first queries decoded_size: roughly 0.4–0.6% together.
These fractions are **estimates from separate measurements**, not measured
end-to-end replacement-codec speedups. Avoiding repeat metadata parsing would
be a smaller independent improvement and need not change the format.

## Reproduce / locations

Code remains in this checkout. Data/results:
`/data/hendrik/compression_experiments/villa4/encoder_speed/header_formats_20261102_v2/`
(`results.jsonl` contains every timing batch and size).
The first eight-variant run is in `header_formats_20261102/`.
Sanitizer verification is in `header_formats_20261102_sanitize/`.

```sh
cc -std=c11 -O3 -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_codecs
/home/hendrik/.venv_las/bin/python benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_codecs \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_formats_20261102_v2 \
  --repeats 20 --cpu 97
```

Use `--report-only` to regenerate tables from saved timings.
The runner's CPU pinning is Linux-specific; the native harness itself uses POSIX
clock_gettime. This experiment is not installed or linked into the library.

# Header representation experiments

All eight chunks × 18 groups × four presets; residual bytes unchanged. Every header reconstructs exactly. Existing 12-byte outer framing retained in size accounting; codecs are alternatives, not selected per chunk.

Times are native header-only microseconds/chunk (median of three batches), including allocations. These are NOT full-codec throughput measurements. Original header parsing/reconstruction remains after unpacking.

## Combined

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 6173 | 32.5631% | +0.000% | 22.8 | 7.9 |
| raw | 11710 | 32.8272% | +0.811% | 0.1 | 0.1 |
| byte-rle | 9344 | 32.7143% | +0.464% | 7.7 | 4.2 |
| word-pack | 10474 | 32.7682% | +0.630% | 12.3 | 9.3 |
| delta-word-pack | 10162 | 32.7533% | +0.584% | 16.4 | 9.4 |
| rans | 7232 | 32.6136% | +0.155% | 13.8 | 9.3 |
| structured | 6641 | 32.5855% | +0.069% | 54.3 | 10.6 |
| structured-rans | 6808 | 32.5934% | +0.093% | 63.0 | 16.7 |
| structured-xor | 6624 | 32.5847% | +0.066% | 77.1 | 10.6 |
| structured-plain | 11648 | 32.8242% | +0.802% | 0.4 | 0.3 |

### Combined timing spread (µs/header, min / median / max)

Each value aggregates one of the three timing batches over all headers.

| Variant | Encode | Decode |
|---|---:|---:|
| zstd3 | 22.5 / 22.7 / 25.1 | 7.9 / 8.0 / 8.0 |
| raw | 0.1 / 0.1 / 0.1 | 0.1 / 0.1 / 0.1 |
| byte-rle | 7.6 / 7.7 / 8.5 | 4.3 / 4.3 / 4.4 |
| word-pack | 12.2 / 12.3 / 12.4 | 9.2 / 9.3 / 9.4 |
| delta-word-pack | 16.4 / 16.4 / 16.8 | 9.4 / 9.4 / 9.5 |
| rans | 13.7 / 13.8 / 14.4 | 9.3 / 9.3 / 9.7 |
| structured | 53.8 / 54.1 / 57.2 | 10.5 / 10.6 / 11.2 |
| structured-rans | 62.8 / 63.1 / 63.3 | 16.7 / 16.7 / 17.3 |
| structured-xor | 76.7 / 76.9 / 78.2 | 10.6 / 10.6 / 10.9 |
| structured-plain | 0.4 / 0.4 / 0.4 | 0.3 / 0.3 / 0.3 |

## fast

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 3091 | 33.6504% | +0.000% | 17.5 | 6.3 |
| raw | 8109 | 33.8896% | +0.711% | 0.1 | 0.1 |
| byte-rle | 5771 | 33.7782% | +0.380% | 5.3 | 4.0 |
| word-pack | 7331 | 33.8526% | +0.601% | 8.6 | 6.5 |
| delta-word-pack | 7056 | 33.8395% | +0.562% | 11.0 | 6.5 |
| rans | 3765 | 33.6825% | +0.096% | 10.1 | 6.8 |
| structured | 3535 | 33.6715% | +0.063% | 38.5 | 7.9 |
| structured-rans | 3659 | 33.6775% | +0.081% | 43.6 | 11.7 |
| structured-xor | 3536 | 33.6716% | +0.063% | 53.4 | 7.9 |
| structured-plain | 8047 | 33.8867% | +0.702% | 0.3 | 0.3 |

## medium

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7051 | 32.3715% | +0.000% | 24.8 | 8.4 |
| raw | 12902 | 32.6505% | +0.862% | 0.1 | 0.1 |
| byte-rle | 10500 | 32.5360% | +0.508% | 8.3 | 4.3 |
| word-pack | 11492 | 32.5833% | +0.654% | 13.5 | 10.3 |
| delta-word-pack | 11188 | 32.5688% | +0.609% | 18.3 | 10.4 |
| rans | 8414 | 32.4365% | +0.201% | 15.0 | 10.2 |
| structured | 7443 | 32.3902% | +0.058% | 59.4 | 11.5 |
| structured-rans | 7607 | 32.3981% | +0.082% | 69.0 | 18.3 |
| structured-xor | 7403 | 32.3883% | +0.052% | 83.7 | 11.6 |
| structured-plain | 12843 | 32.6477% | +0.853% | 0.4 | 0.3 |

## slow

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7270 | 32.1316% | +0.000% | 24.5 | 8.5 |
| raw | 12908 | 32.4005% | +0.837% | 0.1 | 0.1 |
| byte-rle | 10533 | 32.2872% | +0.484% | 8.5 | 4.3 |
| word-pack | 11536 | 32.3351% | +0.633% | 13.4 | 10.2 |
| delta-word-pack | 11201 | 32.3191% | +0.583% | 18.3 | 10.4 |
| rans | 8358 | 32.1835% | +0.161% | 15.0 | 10.2 |
| structured | 7787 | 32.1563% | +0.077% | 59.7 | 11.5 |
| structured-rans | 7973 | 32.1652% | +0.104% | 69.6 | 18.5 |
| structured-xor | 7777 | 32.1558% | +0.075% | 85.6 | 11.5 |
| structured-plain | 12850 | 32.3977% | +0.828% | 0.4 | 0.3 |

## slower

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7281 | 32.0990% | +0.000% | 24.5 | 8.5 |
| raw | 12923 | 32.3680% | +0.838% | 0.1 | 0.1 |
| byte-rle | 10570 | 32.2559% | +0.489% | 8.5 | 4.3 |
| word-pack | 11538 | 32.3020% | +0.632% | 13.4 | 10.2 |
| delta-word-pack | 11204 | 32.2861% | +0.583% | 18.2 | 10.4 |
| rans | 8390 | 32.1519% | +0.165% | 15.0 | 10.2 |
| structured | 7801 | 32.1238% | +0.077% | 59.7 | 11.5 |
| structured-rans | 7991 | 32.1329% | +0.106% | 69.7 | 18.4 |
| structured-xor | 7782 | 32.1229% | +0.074% | 85.7 | 11.5 |
| structured-plain | 12850 | 32.3645% | +0.827% | 0.4 | 0.4 |

## Per-group total size / raw (all four presets combined)

| Group | zstd3 | raw | byte-rle | word-pack | delta-word-pack | rans | structured | structured-rans | structured-xor | structured-plain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CT-L0 | 21.9300% | 22.0641% | 21.9847% | 22.0067% | 22.0031% | 21.9586% | 21.9456% | 21.9466% | 21.9429% | 22.0522% |
| CT-L1 | 45.1002% | 45.1793% | 45.1353% | 45.1423% | 45.1435% | 45.1207% | 45.1093% | 45.1093% | 45.1095% | 45.1632% |
| nx-L3 | 38.1239% | 38.4250% | 38.2842% | 38.3613% | 38.3264% | 38.2047% | 38.1557% | 38.1649% | 38.1552% | 38.4233% |
| nx-L4 | 44.8130% | 45.1083% | 44.9922% | 45.0421% | 45.0144% | 44.8833% | 44.8481% | 44.8557% | 44.8478% | 45.1067% |
| width-L3 | 28.0760% | 28.4580% | 28.2602% | 28.3589% | 28.3418% | 28.1518% | 28.1009% | 28.1107% | 28.1000% | 28.4562% |
| width-L4 | 35.6834% | 35.9934% | 35.8497% | 35.9256% | 35.9174% | 35.7103% | 35.7053% | 35.7126% | 35.7050% | 35.9920% |
| ny-L3 | 35.2784% | 35.5711% | 35.4496% | 35.5129% | 35.4857% | 35.3492% | 35.3049% | 35.3139% | 35.3043% | 35.5693% |
| ny-L4 | 48.7956% | 49.0809% | 48.9877% | 49.0234% | 48.9990% | 48.8634% | 48.8272% | 48.8363% | 48.8268% | 49.0792% |
| compound/presence-L3 | 26.1481% | 26.3779% | 26.3027% | 26.3357% | 26.3308% | 26.1893% | 26.1718% | 26.1819% | 26.1680% | 26.3762% |
| compound/presence-L4 | 36.6658% | 36.9601% | 36.8565% | 36.8953% | 36.8872% | 36.6955% | 36.6867% | 36.6944% | 36.6855% | 36.9585% |
| compound/sheet_grad_mag-L3 | 3.9510% | 4.0467% | 4.0206% | 4.0450% | 4.0453% | 3.9566% | 3.9586% | 3.9677% | 3.9586% | 4.0455% |
| compound/sheet_grad_mag-L4 | 6.1495% | 6.2762% | 6.2408% | 6.2703% | 6.2709% | 6.1550% | 6.1559% | 6.1644% | 6.1559% | 6.2749% |
| compound/sheet_nx-L3 | 30.0346% | 30.3512% | 30.2039% | 30.2802% | 30.2613% | 30.1032% | 30.0573% | 30.0661% | 30.0566% | 30.3494% |
| compound/sheet_nx-L4 | 44.1654% | 44.4768% | 44.3498% | 44.4077% | 44.3839% | 44.2360% | 44.1972% | 44.2067% | 44.1966% | 44.4751% |
| compound/sheet_ny-L3 | 32.4321% | 32.7470% | 32.6047% | 32.6776% | 32.6537% | 32.5041% | 32.4568% | 32.4659% | 32.4562% | 32.7452% |
| compound/sheet_ny-L4 | 43.4085% | 43.7165% | 43.5929% | 43.6468% | 43.6208% | 43.4809% | 43.4384% | 43.4479% | 43.4379% | 43.7148% |
| compound/width-L3 | 27.0824% | 27.4447% | 27.2679% | 27.3587% | 27.3453% | 27.1497% | 27.1015% | 27.1112% | 27.1005% | 27.4429% |
| compound/width-L4 | 38.2985% | 38.6117% | 38.4741% | 38.5377% | 38.5297% | 38.3326% | 38.3170% | 38.3249% | 38.3165% | 38.6103% |

