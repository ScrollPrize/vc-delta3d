# Strict wrapping model deltas + rANS — 2026-11-02

## Precisely tested

No XOR, RLE, bit-packing alternatives, reference search, raw fallback or codec
selection occurs in these new strict variants. The earlier adaptive experiments
are included only as explicitly separate controls.

The stored NG models remain in original order. Each model's six configuration
bytes are literal; its remaining little-endian coefficient/bias/threshold slots
form the model body. Each transformed body has one plain native rANS stream
and explicit lengths. Previous-model prediction uses the **previous original,
reconstructed model**, not its encoded delta. Missing reference slots predict
zero; unused extra reference slots are ignored.

For each scheme, test modulo-256 byte deltas and modulo-65536 word deltas:

1. **Between only:** first body plain rANS; subsequent bodies subtract the
   corresponding slots of the preceding model, then rANS.
2. **Both:** first body within-model delta + rANS; subsequent bodies subtract
   the preceding model, then difference the residuals within the body, then rANS.
3. **First within, rest between:** within-model delta only on the first model;
   subsequent bodies get between-model delta only, then rANS.
4. **Independent within:** every model body independently gets within-model
   delta + rANS, without between-model prediction.
5. **Direct control:** all bodies plain rANS without either delta.

Within-model prediction starts at zero each body. For words, an odd final byte
uses an eight-bit residual; synthetic round-trip tests cover this as well.

All non-model bytes are held identical and coded in a common **plain rANS**
stream for every strict variant. This includes the remaining entropy metadata
and block-choice information. Unlike the previous adaptive structured format,
it does not search for a representation of this common stream. Therefore
whole-header differences versus that previous format are not an isolated
measurement of model-delta quality; use the direct strict control.

## Corpus finding and conclusion

Of 576 frozen headers:

- 32 have no stored NG model.
- 540 have one.
- Only 4 have two (two CT L0 medium frames and two CT L1 medium frames).

Consequently between-model prediction can affect only four headers. Their
mean encoded NG-model storage is 878 bytes with plain rANS, 835 with byte-wise
between-model delta, or 849 with word-wise between-model delta.

Mean model storage over the full corpus drops only from 193.14 to 192.84 bytes.
The common non-model stream is 7,149.16 bytes on average and is identical across
all strict schemes. This explains the nearly unchanged whole-header ratios.
Within-model differentiation is counterproductive for these fitted coefficients:
it raises average model storage to roughly 199 bytes (word) or 215–216 (byte).

**Best of the requested strict schemes: byte-wise between-model delta only.**
There is no evidence here to retain the extra within-model pass.
The four multi-model examples are insufficient to claim general transfer.

A distinct future experiment could apply these schemes to the much more
numerous rANS **frequency tables**. Those are not the NG predictor models tested
here, and silently conflating them would make this comparison misleading.

## Validation, measurements and reproduction

All 9 strict schemes × 576 headers reconstructed the original header exactly,
also under ASan/UBSan. An additional 135 synthetic tests exercise zero models,
empty/growing/shrinking bodies, zero values, modulo-wrap boundaries and odd tails.
Production codec and dependency configuration are unchanged.

Native C, GCC13.3 -O3, AMD EPYC9475F, CPU97. Three batches of20 repetitions per
header; allocations included, input reads and verification excluded. Timings
are header-only, not complete volume encode/decode throughput. Encoding is
about15 µs/header and decoding about11 µs/header across the strict schemes.
Raw timing repetitions are retained in the result JSONL.

Code:
- `benchmark/header_codecs.c`: shared native transform helpers and strict
  encoder/decoder; the strict path does not call adaptive section/reference code.
- `benchmark/compare_headers.py`: orchestration and reporting only.

Results:
`/data/hendrik/compression_experiments/villa4/encoder_speed/header_strict_model_delta_20261102/`
and the corresponding `_sanitize/` directory.

```sh
cc -std=c11 -O3 -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_codecs
/data/hendrik/compression_experiments/villa4/builds/header_codecs --self-test
/home/hendrik/.venv_las/bin/python benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_codecs \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_strict_model_delta_20261102 \
  --repeats 20 --cpu 97 --variant-mask 0x0ff80901
```

The mask includes Zstd and two earlier structured controls plus all nine strict
schemes. Use `0x0ff80000` for only the strict schemes. Sanitizer runs compile with
`-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`, link the sanitizer
build of the library, set `ulimit -s 32768` on this host, and use CPU98/repeats1.
Zstd is linked by this experimental harness solely to read original frozen
frames and measure the old control; none of the strict transforms uses it.

# Header representation experiments

All eight chunks × 18 groups × four presets; residual bytes unchanged. Every header reconstructs exactly. Existing 12-byte outer framing retained in size accounting; codecs are alternatives, not selected per chunk.

Times are native header-only microseconds/chunk (median of three batches), including allocations. These are NOT full-codec throughput measurements. Original header parsing/reconstruction remains after unpacking.

## Combined

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 6173 | 32.5631% | +0.000% | 22.9 | 8.0 |
| structured-xor | 6624 | 32.5847% | +0.066% | 76.4 | 11.0 |
| struct-xor-delta-rans | 6181 | 32.5635% | +0.001% | 184.4 | 13.1 |
| strict-direct-rans | 7342 | 32.6189% | +0.171% | 14.8 | 11.0 |
| strict-between8 | 7342 | 32.6189% | +0.171% | 14.7 | 10.9 |
| strict-between16 | 7342 | 32.6189% | +0.171% | 14.6 | 10.9 |
| strict-both8 | 7365 | 32.6199% | +0.174% | 14.7 | 11.0 |
| strict-both16 | 7348 | 32.6192% | +0.172% | 14.7 | 11.0 |
| strict-first-within8 | 7364 | 32.6199% | +0.174% | 14.7 | 11.0 |
| strict-first-within16 | 7348 | 32.6192% | +0.172% | 14.7 | 11.0 |
| strict-independent8 | 7365 | 32.6200% | +0.175% | 14.7 | 11.0 |
| strict-independent16 | 7348 | 32.6192% | +0.172% | 14.7 | 11.0 |

## Strict NG-model experiment

Stored NG-model counts: 0 models: 32 headers, 1 models: 540 headers, 2 models: 4 headers.

The common non-model bytes use identical plain rANS coding in all strict variants. Prior adaptive structured controls differ in non-model representation too; their whole-header differences must not be attributed to model deltas alone.

| Variant | Mean NG-model bytes | Mean common bytes | NG-model bytes on multi-model subset |
|---|---:|---:|---:|
| strict-direct-rans | 193.14 | 7149.16 | 878.00 |
| strict-between8 | 192.84 | 7149.16 | 835.00 |
| strict-between16 | 192.94 | 7149.16 | 849.00 |
| strict-both8 | 215.54 | 7149.16 | 965.75 |
| strict-both16 | 199.01 | 7149.16 | 892.75 |
| strict-first-within8 | 215.14 | 7149.16 | 907.50 |
| strict-first-within16 | 198.87 | 7149.16 | 872.50 |
| strict-independent8 | 215.94 | 7149.16 | 1023.50 |
| strict-independent16 | 199.22 | 7149.16 | 922.50 |

### Combined timing spread (µs/header, min / median / max)

Each value aggregates one of the three timing batches over all headers.

| Variant | Encode | Decode |
|---|---:|---:|
| zstd3 | 22.8 / 22.9 / 25.3 | 8.0 / 8.0 / 8.1 |
| structured-xor | 75.6 / 76.2 / 80.1 | 10.9 / 11.0 / 11.7 |
| struct-xor-delta-rans | 182.8 / 184.4 / 188.6 | 13.0 / 13.0 / 13.6 |
| strict-direct-rans | 14.7 / 14.7 / 15.2 | 11.0 / 11.0 / 11.4 |
| strict-between8 | 14.7 / 14.7 / 14.7 | 11.0 / 11.0 / 11.0 |
| strict-between16 | 14.7 / 14.7 / 14.7 | 11.0 / 11.0 / 11.0 |
| strict-both8 | 14.8 / 14.8 / 14.8 | 11.0 / 11.1 / 11.1 |
| strict-both16 | 14.8 / 14.8 / 14.8 | 11.0 / 11.1 / 11.1 |
| strict-first-within8 | 14.8 / 14.8 / 14.8 | 11.1 / 11.1 / 11.1 |
| strict-first-within16 | 14.8 / 14.8 / 14.8 | 11.0 / 11.0 / 11.0 |
| strict-independent8 | 14.8 / 14.8 / 14.8 | 11.1 / 11.1 / 11.1 |
| strict-independent16 | 14.8 / 14.8 / 14.8 | 11.0 / 11.1 / 11.1 |

## fast

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 3091 | 33.6504% | +0.000% | 17.6 | 6.3 |
| structured-xor | 3536 | 33.6716% | +0.063% | 53.9 | 8.2 |
| struct-xor-delta-rans | 3192 | 33.6552% | +0.014% | 113.8 | 9.3 |
| strict-direct-rans | 3850 | 33.6866% | +0.108% | 10.9 | 8.5 |
| strict-between8 | 3850 | 33.6866% | +0.108% | 10.8 | 8.4 |
| strict-between16 | 3850 | 33.6866% | +0.108% | 10.8 | 8.4 |
| strict-both8 | 3858 | 33.6870% | +0.109% | 10.8 | 8.4 |
| strict-both16 | 3853 | 33.6867% | +0.108% | 10.8 | 8.4 |
| strict-first-within8 | 3858 | 33.6870% | +0.109% | 10.8 | 8.4 |
| strict-first-within16 | 3853 | 33.6867% | +0.108% | 10.8 | 8.4 |
| strict-independent8 | 3858 | 33.6870% | +0.109% | 10.8 | 8.4 |
| strict-independent16 | 3853 | 33.6867% | +0.108% | 10.8 | 8.4 |

## medium

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7051 | 32.3715% | +0.000% | 24.7 | 8.5 |
| structured-xor | 7403 | 32.3883% | +0.052% | 81.7 | 12.0 |
| struct-xor-delta-rans | 6935 | 32.3660% | -0.017% | 220.1 | 14.2 |
| strict-direct-rans | 8599 | 32.4453% | +0.228% | 16.1 | 11.8 |
| strict-between8 | 8598 | 32.4453% | +0.228% | 16.0 | 11.7 |
| strict-between16 | 8598 | 32.4453% | +0.228% | 16.0 | 11.7 |
| strict-both8 | 8649 | 32.4477% | +0.235% | 16.1 | 11.8 |
| strict-both16 | 8613 | 32.4460% | +0.230% | 16.1 | 11.8 |
| strict-first-within8 | 8648 | 32.4477% | +0.235% | 16.2 | 11.8 |
| strict-first-within16 | 8612 | 32.4460% | +0.230% | 16.1 | 11.8 |
| strict-independent8 | 8651 | 32.4478% | +0.236% | 16.2 | 11.8 |
| strict-independent16 | 8614 | 32.4460% | +0.230% | 16.1 | 11.8 |

## slow

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7270 | 32.1316% | +0.000% | 24.7 | 8.6 |
| structured-xor | 7777 | 32.1558% | +0.075% | 84.6 | 11.8 |
| struct-xor-delta-rans | 7297 | 32.1329% | +0.004% | 201.6 | 14.4 |
| strict-direct-rans | 8445 | 32.1877% | +0.174% | 16.0 | 11.9 |
| strict-between8 | 8445 | 32.1877% | +0.174% | 16.0 | 11.8 |
| strict-between16 | 8445 | 32.1877% | +0.174% | 15.9 | 11.8 |
| strict-both8 | 8457 | 32.1883% | +0.176% | 16.0 | 11.9 |
| strict-both16 | 8450 | 32.1879% | +0.175% | 16.0 | 11.9 |
| strict-first-within8 | 8457 | 32.1883% | +0.176% | 16.0 | 11.9 |
| strict-first-within16 | 8450 | 32.1879% | +0.175% | 16.1 | 11.9 |
| strict-independent8 | 8457 | 32.1883% | +0.176% | 16.0 | 11.9 |
| strict-independent16 | 8450 | 32.1879% | +0.175% | 16.0 | 11.9 |

## slower

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7281 | 32.0990% | +0.000% | 24.7 | 8.6 |
| structured-xor | 7782 | 32.1229% | +0.074% | 85.3 | 12.0 |
| struct-xor-delta-rans | 7300 | 32.0999% | +0.003% | 202.2 | 14.4 |
| strict-direct-rans | 8475 | 32.1559% | +0.177% | 16.1 | 11.9 |
| strict-between8 | 8475 | 32.1559% | +0.177% | 15.9 | 11.8 |
| strict-between16 | 8475 | 32.1559% | +0.177% | 15.9 | 11.8 |
| strict-both8 | 8494 | 32.1568% | +0.180% | 16.0 | 11.9 |
| strict-both16 | 8478 | 32.1561% | +0.178% | 16.0 | 11.8 |
| strict-first-within8 | 8494 | 32.1568% | +0.180% | 15.9 | 11.9 |
| strict-first-within16 | 8478 | 32.1561% | +0.178% | 16.0 | 11.8 |
| strict-independent8 | 8494 | 32.1568% | +0.180% | 16.0 | 11.9 |
| strict-independent16 | 8478 | 32.1561% | +0.178% | 16.0 | 11.9 |

## Per-group total size / raw (all four presets combined)

| Group | zstd3 | structured-xor | struct-xor-delta-rans | strict-direct-rans | strict-between8 | strict-between16 | strict-both8 | strict-both16 | strict-first-within8 | strict-first-within16 | strict-independent8 | strict-independent16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CT-L0 | 21.9300% | 21.9429% | 21.9429% | 21.9719% | 21.9718% | 21.9719% | 21.9754% | 21.9729% | 21.9753% | 21.9728% | 21.9756% | 21.9729% |
| CT-L1 | 45.1002% | 45.1095% | 45.1092% | 45.1342% | 45.1340% | 45.1340% | 45.1382% | 45.1352% | 45.1381% | 45.1351% | 45.1384% | 45.1353% |
| nx-L3 | 38.1239% | 38.1552% | 38.1180% | 38.2085% | 38.2085% | 38.2085% | 38.2091% | 38.2087% | 38.2091% | 38.2087% | 38.2091% | 38.2087% |
| nx-L4 | 44.8130% | 44.8478% | 44.7972% | 44.8868% | 44.8868% | 44.8868% | 44.8873% | 44.8869% | 44.8873% | 44.8869% | 44.8873% | 44.8869% |
| width-L3 | 28.0760% | 28.1000% | 28.0727% | 28.1568% | 28.1568% | 28.1568% | 28.1576% | 28.1570% | 28.1576% | 28.1570% | 28.1576% | 28.1570% |
| width-L4 | 35.6834% | 35.7050% | 35.6953% | 35.7134% | 35.7134% | 35.7134% | 35.7136% | 35.7135% | 35.7136% | 35.7135% | 35.7136% | 35.7135% |
| ny-L3 | 35.2784% | 35.3043% | 35.2747% | 35.3532% | 35.3532% | 35.3532% | 35.3539% | 35.3533% | 35.3539% | 35.3533% | 35.3539% | 35.3533% |
| ny-L4 | 48.7956% | 48.8268% | 48.7770% | 48.8668% | 48.8668% | 48.8668% | 48.8674% | 48.8670% | 48.8674% | 48.8670% | 48.8674% | 48.8670% |
| compound/presence-L3 | 26.1481% | 26.1680% | 26.1613% | 26.1987% | 26.1987% | 26.1987% | 26.2016% | 26.1995% | 26.2016% | 26.1995% | 26.2016% | 26.1995% |
| compound/presence-L4 | 36.6658% | 36.6855% | 36.6779% | 36.7007% | 36.7007% | 36.7007% | 36.7018% | 36.7010% | 36.7018% | 36.7010% | 36.7018% | 36.7010% |
| compound/sheet_grad_mag-L3 | 3.9510% | 3.9586% | 3.9586% | 3.9589% | 3.9589% | 3.9589% | 3.9589% | 3.9589% | 3.9589% | 3.9589% | 3.9589% | 3.9589% |
| compound/sheet_grad_mag-L4 | 6.1495% | 6.1559% | 6.1559% | 6.1572% | 6.1572% | 6.1572% | 6.1572% | 6.1572% | 6.1572% | 6.1572% | 6.1572% | 6.1572% |
| compound/sheet_nx-L3 | 30.0346% | 30.0566% | 30.0321% | 30.1074% | 30.1074% | 30.1074% | 30.1082% | 30.1076% | 30.1082% | 30.1076% | 30.1082% | 30.1076% |
| compound/sheet_nx-L4 | 44.1654% | 44.1966% | 44.1551% | 44.2404% | 44.2404% | 44.2404% | 44.2411% | 44.2406% | 44.2411% | 44.2406% | 44.2411% | 44.2406% |
| compound/sheet_ny-L3 | 32.4321% | 32.4562% | 32.4290% | 32.5084% | 32.5084% | 32.5084% | 32.5091% | 32.5086% | 32.5091% | 32.5086% | 32.5091% | 32.5086% |
| compound/sheet_ny-L4 | 43.4085% | 43.4379% | 43.3991% | 43.4851% | 43.4851% | 43.4851% | 43.4858% | 43.4853% | 43.4858% | 43.4853% | 43.4858% | 43.4853% |
| compound/width-L3 | 27.0824% | 27.1005% | 27.0781% | 27.1547% | 27.1547% | 27.1547% | 27.1555% | 27.1549% | 27.1555% | 27.1549% | 27.1555% | 27.1549% |
| compound/width-L4 | 38.2985% | 38.3165% | 38.3091% | 38.3368% | 38.3368% | 38.3368% | 38.3373% | 38.3369% | 38.3373% | 38.3369% | 38.3373% | 38.3369% |

