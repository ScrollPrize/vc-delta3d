# Dependency-free header coding: delta + rANS — 2026-11-02

Follow-up: [strict model-delta tests](header_strict_model_delta.md) remove XOR
and adaptive codec selection completely and isolate the requested delta orders.

## Objective correction

The objective is **removing Zstd as an external dependency**, not beating it on
speed or size. The earlier report's recommendation to keep Zstd addressed the
wrong objective. These experiments establish good replacement candidates;
they do not yet change production framing or dependencies.

## Results and recommendation

On all 576 frozen headers (18 groups × eight chunks × four presets):

- Prior structured-XOR: 6,624 mean header bytes, +0.066% total file size vs Zstd.
- Add adjacent-byte/word delta transforms before rANS: 6,195 bytes, +0.003%.
- Also allow run-length preprocessing before rANS: 6,181 bytes, +0.001%.
- Search XOR vs byte/word subtraction between models as well: 6,180 bytes.
  This saves only about one byte over XOR while increasing search time.
- Simplify sections to raw/rANS/byte-delta-rANS/word-delta-rANS only:
  6,341 bytes, +0.025%; about 75 µs header encode versus 136–178 µs for
  the broader choices.

Thus **structured XOR references plus delta+rANS is a strong Zstd-free
candidate**. Delta is the main improvement, not RLE: RLE alone gives 6,445
bytes; delta alone gives 6,195. Arithmetic model-reference deltas do not bring
a meaningful further benefit over XOR references. Keep those out of a compact
production implementation unless another corpus demonstrates a benefit.

These are aggregate results, not a universal per-channel improvement. The
XOR+delta/RLE version changes total size by +0.059% on CT L0, +0.020% on CT L1,
and up to +0.191% on gradient-mask L3. Several normal channels improve slightly.
Full per-group and per-preset tables follow.

## Algorithm

Keep the previous structured fields (varints, signed palette-weight zigzag,
model references, separate selector and entropy-metadata sections).

Additional section representations:

1. Byte differences modulo 256, then the existing compact-table rANS coder.
2. Word differences modulo 65536, then rANS on the resulting little-endian
   bytes. An odd final byte is handled separately.
3. PackBits-style runs/literals followed by rANS.

Each section carries a mode byte, uncompressed length and coded length.
The encoder selects the smallest exact representation, including old/raw
choices; the decoder follows that mode and reverses the transformation.
Reference modes optionally subtract each prior model byte-wise or word-wise,
rather than XOR. Decoding only uses models already reconstructed.
No training, approximation, value changes, or external codec is required by
these new header representations. rANS is reused from the retained C codec.

The native experiment still links Zstd to read the old frozen NCF1 inputs and
measure the control. That is **not** a dependency of the candidate algorithms.

## Verification / measurements

Every candidate reconstructs the original header byte-for-byte on the entire
corpus; residual payloads and encoder model/block decisions remain unchanged.
All new modes were also exercised across all 576 headers with ASan/UBSan.
Timing uses the same native C harness, GCC13.3 -O3, CPU97, three batches of20
repetitions/header. Reported µs are header-only encode/decode including allocation,
not whole-codec throughput. Sanitizers use CPU98 and one repetition.

Data under `/data/hendrik/compression_experiments/villa4/encoder_speed/`:

- `header_delta_rans_20261102`: model-reference and section-transform comparisons.
- `header_delta_rans_ablation_20261102`: delta versus RLE contribution.
- `header_delta_rans_minimal_20261102`: fewer section coding choices.
- Corresponding `_sanitize` directories: verification results.

Use the commands in [the initial report](header_formats.md), changing output
directory and adding `--variant-mask 0x7d01`, `0x3a101`, or `0x48101`,
respectively. The mask chooses harness variants only, not a codec option.
Source: `benchmark/header_codecs.c`; orchestration:
`benchmark/compare_headers.py`.

## Integration caveats

Production still uses Zstd. Removing it requires replacing both final header
coding and the Zstd-based metadata-size estimates used during preset searches.
This test deliberately does not refit/reselect those models; sizes here isolate
the header representation.

Existing NCF1 frames contain Zstd. Retaining their decoding while eliminating
the external dependency requires either a bundled compatible decoder or an
optional legacy-Zstd build feature. A new encoder format alone cannot remove
that compatibility requirement. No compatibility was dropped in this experiment.


## Main experiment

# Header representation experiments

All eight chunks × 18 groups × four presets; residual bytes unchanged. Every header reconstructs exactly. Existing 12-byte outer framing retained in size accounting; codecs are alternatives, not selected per chunk.

Times are native header-only microseconds/chunk (median of three batches), including allocations. These are NOT full-codec throughput measurements. Original header parsing/reconstruction remains after unpacking.

## Combined

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 6173 | 32.5631% | +0.000% | 22.8 | 7.9 |
| structured-xor | 6624 | 32.5847% | +0.066% | 75.1 | 10.7 |
| struct-delta8-rans | 6181 | 32.5635% | +0.001% | 183.7 | 14.1 |
| struct-xor-delta-rans | 6181 | 32.5635% | +0.001% | 177.8 | 12.7 |
| struct-delta16-rans | 6180 | 32.5635% | +0.001% | 178.4 | 14.2 |
| struct-mixed-delta-rans | 6180 | 32.5634% | +0.001% | 283.8 | 13.0 |
| struct-prefix-delta-rans | 6200 | 32.5644% | +0.004% | 117.8 | 12.6 |

### Combined timing spread (µs/header, min / median / max)

Each value aggregates one of the three timing batches over all headers.

| Variant | Encode | Decode |
|---|---:|---:|
| zstd3 | 22.5 / 22.7 / 25.1 | 7.9 / 8.0 / 8.0 |
| structured-xor | 74.4 / 74.8 / 78.9 | 10.6 / 10.7 / 11.5 |
| struct-delta8-rans | 182.0 / 183.8 / 187.7 | 14.0 / 14.1 / 14.3 |
| struct-xor-delta-rans | 177.0 / 177.8 / 180.0 | 12.7 / 12.7 / 12.8 |
| struct-delta16-rans | 178.0 / 178.6 / 178.9 | 14.2 / 14.2 / 14.3 |
| struct-mixed-delta-rans | 281.8 / 284.1 / 284.6 | 13.1 / 13.1 / 13.1 |
| struct-prefix-delta-rans | 115.4 / 117.8 / 125.6 | 12.6 / 12.7 / 12.8 |

## fast

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 3091 | 33.6504% | +0.000% | 17.3 | 6.3 |
| structured-xor | 3536 | 33.6716% | +0.063% | 52.6 | 8.0 |
| struct-delta8-rans | 3192 | 33.6552% | +0.014% | 112.5 | 9.8 |
| struct-xor-delta-rans | 3192 | 33.6552% | +0.014% | 111.1 | 9.0 |
| struct-delta16-rans | 3192 | 33.6552% | +0.014% | 112.0 | 9.9 |
| struct-mixed-delta-rans | 3192 | 33.6552% | +0.014% | 174.7 | 9.1 |
| struct-prefix-delta-rans | 3192 | 33.6552% | +0.014% | 79.2 | 8.9 |

## medium

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7051 | 32.3715% | +0.000% | 24.7 | 8.4 |
| structured-xor | 7403 | 32.3883% | +0.052% | 80.6 | 11.7 |
| struct-delta8-rans | 6934 | 32.3659% | -0.017% | 220.8 | 15.7 |
| struct-xor-delta-rans | 6935 | 32.3660% | -0.017% | 216.1 | 13.9 |
| struct-delta16-rans | 6934 | 32.3659% | -0.017% | 217.2 | 16.0 |
| struct-mixed-delta-rans | 6932 | 32.3658% | -0.018% | 351.0 | 14.9 |
| struct-prefix-delta-rans | 6981 | 32.3682% | -0.010% | 138.5 | 13.9 |

## slow

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7270 | 32.1316% | +0.000% | 24.7 | 8.5 |
| structured-xor | 7777 | 32.1558% | +0.075% | 83.6 | 11.6 |
| struct-delta8-rans | 7296 | 32.1329% | +0.004% | 200.6 | 15.4 |
| struct-xor-delta-rans | 7297 | 32.1329% | +0.004% | 191.8 | 14.0 |
| struct-delta16-rans | 7296 | 32.1329% | +0.004% | 192.2 | 15.5 |
| struct-mixed-delta-rans | 7296 | 32.1329% | +0.004% | 304.6 | 14.1 |
| struct-prefix-delta-rans | 7308 | 32.1334% | +0.006% | 126.9 | 13.9 |

## slower

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7281 | 32.0990% | +0.000% | 24.6 | 8.4 |
| structured-xor | 7782 | 32.1229% | +0.074% | 83.5 | 11.6 |
| struct-delta8-rans | 7300 | 32.0999% | +0.003% | 201.0 | 15.3 |
| struct-xor-delta-rans | 7300 | 32.0999% | +0.003% | 192.3 | 14.0 |
| struct-delta16-rans | 7300 | 32.0999% | +0.003% | 192.3 | 15.5 |
| struct-mixed-delta-rans | 7299 | 32.0999% | +0.003% | 305.0 | 14.1 |
| struct-prefix-delta-rans | 7319 | 32.1008% | +0.006% | 126.6 | 13.9 |

## Per-group total size / raw (all four presets combined)

| Group | zstd3 | structured-xor | struct-delta8-rans | struct-xor-delta-rans | struct-delta16-rans | struct-mixed-delta-rans | struct-prefix-delta-rans |
|---|---:|---:|---:|---:|---:|---:|---:|
| CT-L0 | 21.9300% | 21.9429% | 21.9429% | 21.9429% | 21.9428% | 21.9428% | 21.9456% |
| CT-L1 | 45.1002% | 45.1095% | 45.1093% | 45.1092% | 45.1091% | 45.1091% | 45.1090% |
| nx-L3 | 38.1239% | 38.1552% | 38.1180% | 38.1180% | 38.1179% | 38.1179% | 38.1187% |
| nx-L4 | 44.8130% | 44.8478% | 44.7971% | 44.7972% | 44.7972% | 44.7971% | 44.7977% |
| width-L3 | 28.0760% | 28.1000% | 28.0727% | 28.0727% | 28.0727% | 28.0727% | 28.0738% |
| width-L4 | 35.6834% | 35.7050% | 35.6953% | 35.6953% | 35.6953% | 35.6953% | 35.6956% |
| ny-L3 | 35.2784% | 35.3043% | 35.2747% | 35.2747% | 35.2746% | 35.2746% | 35.2754% |
| ny-L4 | 48.7956% | 48.8268% | 48.7770% | 48.7770% | 48.7770% | 48.7770% | 48.7776% |
| compound/presence-L3 | 26.1481% | 26.1680% | 26.1611% | 26.1613% | 26.1612% | 26.1611% | 26.1655% |
| compound/presence-L4 | 36.6658% | 36.6855% | 36.6779% | 36.6779% | 36.6779% | 36.6778% | 36.6790% |
| compound/sheet_grad_mag-L3 | 3.9510% | 3.9586% | 3.9586% | 3.9586% | 3.9586% | 3.9586% | 3.9586% |
| compound/sheet_grad_mag-L4 | 6.1495% | 6.1559% | 6.1559% | 6.1559% | 6.1559% | 6.1559% | 6.1559% |
| compound/sheet_nx-L3 | 30.0346% | 30.0566% | 30.0320% | 30.0321% | 30.0321% | 30.0320% | 30.0330% |
| compound/sheet_nx-L4 | 44.1654% | 44.1966% | 44.1551% | 44.1551% | 44.1551% | 44.1550% | 44.1558% |
| compound/sheet_ny-L3 | 32.4321% | 32.4562% | 32.4289% | 32.4290% | 32.4289% | 32.4289% | 32.4297% |
| compound/sheet_ny-L4 | 43.4085% | 43.4379% | 43.3991% | 43.3991% | 43.3991% | 43.3991% | 43.3997% |
| compound/width-L3 | 27.0824% | 27.1005% | 27.0780% | 27.0781% | 27.0781% | 27.0780% | 27.0791% |
| compound/width-L4 | 38.2985% | 38.3165% | 38.3091% | 38.3091% | 38.3091% | 38.3091% | 38.3097% |

## Delta versus RLE ablation

# Header representation experiments

All eight chunks × 18 groups × four presets; residual bytes unchanged. Every header reconstructs exactly. Existing 12-byte outer framing retained in size accounting; codecs are alternatives, not selected per chunk.

Times are native header-only microseconds/chunk (median of three batches), including allocations. These are NOT full-codec throughput measurements. Original header parsing/reconstruction remains after unpacking.

## Combined

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 6173 | 32.5631% | +0.000% | 22.7 | 7.9 |
| structured-xor | 6624 | 32.5847% | +0.066% | 76.6 | 10.7 |
| struct-mixed-delta-rans | 6180 | 32.5634% | +0.001% | 290.9 | 13.7 |
| struct-xor-delta-only | 6195 | 32.5642% | +0.003% | 135.9 | 13.1 |
| struct-xor-rle-only | 6445 | 32.5761% | +0.040% | 103.5 | 12.1 |
| struct-prefix-rle-only | 6470 | 32.5773% | +0.043% | 72.9 | 11.7 |

### Combined timing spread (µs/header, min / median / max)

Each value aggregates one of the three timing batches over all headers.

| Variant | Encode | Decode |
|---|---:|---:|
| zstd3 | 22.5 / 22.6 / 25.1 | 7.9 / 8.0 / 8.0 |
| structured-xor | 75.7 / 76.1 / 80.3 | 10.6 / 10.7 / 11.5 |
| struct-mixed-delta-rans | 289.8 / 291.0 / 293.8 | 13.6 / 13.7 / 13.9 |
| struct-xor-delta-only | 134.3 / 135.7 / 140.5 | 13.1 / 13.1 / 13.3 |
| struct-xor-rle-only | 102.9 / 103.5 / 105.3 | 11.9 / 12.0 / 12.6 |
| struct-prefix-rle-only | 72.5 / 72.7 / 73.8 | 11.7 / 11.7 / 12.1 |

## fast

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 3091 | 33.6504% | +0.000% | 17.4 | 6.3 |
| structured-xor | 3536 | 33.6716% | +0.063% | 53.5 | 8.0 |
| struct-mixed-delta-rans | 3192 | 33.6552% | +0.014% | 174.9 | 9.5 |
| struct-xor-delta-only | 3200 | 33.6556% | +0.015% | 91.0 | 9.3 |
| struct-xor-rle-only | 3424 | 33.6662% | +0.047% | 72.0 | 8.6 |
| struct-prefix-rle-only | 3422 | 33.6662% | +0.047% | 51.9 | 8.5 |

## medium

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7051 | 32.3715% | +0.000% | 24.6 | 8.4 |
| structured-xor | 7403 | 32.3883% | +0.052% | 82.5 | 11.8 |
| struct-mixed-delta-rans | 6932 | 32.3658% | -0.018% | 358.4 | 15.5 |
| struct-xor-delta-only | 6955 | 32.3669% | -0.014% | 162.6 | 14.6 |
| struct-xor-rle-only | 7167 | 32.3770% | +0.017% | 111.4 | 13.3 |
| struct-prefix-rle-only | 7233 | 32.3802% | +0.027% | 78.8 | 12.8 |

## slow

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7270 | 32.1316% | +0.000% | 24.6 | 8.5 |
| structured-xor | 7777 | 32.1558% | +0.075% | 85.1 | 11.6 |
| struct-mixed-delta-rans | 7296 | 32.1329% | +0.004% | 314.6 | 14.9 |
| struct-xor-delta-only | 7312 | 32.1336% | +0.006% | 144.8 | 14.3 |
| struct-xor-rle-only | 7594 | 32.1471% | +0.048% | 115.2 | 13.2 |
| struct-prefix-rle-only | 7605 | 32.1476% | +0.050% | 80.3 | 12.8 |

## slower

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7281 | 32.0990% | +0.000% | 24.5 | 8.4 |
| structured-xor | 7782 | 32.1229% | +0.074% | 85.2 | 11.6 |
| struct-mixed-delta-rans | 7299 | 32.0999% | +0.003% | 315.6 | 14.9 |
| struct-xor-delta-only | 7315 | 32.1006% | +0.005% | 144.9 | 14.3 |
| struct-xor-rle-only | 7598 | 32.1141% | +0.047% | 115.5 | 13.1 |
| struct-prefix-rle-only | 7619 | 32.1151% | +0.050% | 80.4 | 12.8 |

## Per-group total size / raw (all four presets combined)

| Group | zstd3 | structured-xor | struct-mixed-delta-rans | struct-xor-delta-only | struct-xor-rle-only | struct-prefix-rle-only |
|---|---:|---:|---:|---:|---:|---:|
| CT-L0 | 21.9300% | 21.9429% | 21.9428% | 21.9429% | 21.9429% | 21.9456% |
| CT-L1 | 45.1002% | 45.1095% | 45.1091% | 45.1095% | 45.1092% | 45.1090% |
| nx-L3 | 38.1239% | 38.1552% | 38.1179% | 38.1196% | 38.1388% | 38.1401% |
| nx-L4 | 44.8130% | 44.8478% | 44.7971% | 44.7999% | 44.8308% | 44.8317% |
| width-L3 | 28.0760% | 28.1000% | 28.0727% | 28.0752% | 28.0847% | 28.0860% |
| width-L4 | 35.6834% | 35.7050% | 35.6953% | 35.6988% | 35.6981% | 35.6984% |
| ny-L3 | 35.2784% | 35.3043% | 35.2746% | 35.2747% | 35.2921% | 35.2933% |
| ny-L4 | 48.7956% | 48.8268% | 48.7770% | 48.7770% | 48.8156% | 48.8165% |
| compound/presence-L3 | 26.1481% | 26.1680% | 26.1611% | 26.1616% | 26.1668% | 26.1714% |
| compound/presence-L4 | 36.6658% | 36.6855% | 36.6778% | 36.6779% | 36.6853% | 36.6867% |
| compound/sheet_grad_mag-L3 | 3.9510% | 3.9586% | 3.9586% | 3.9586% | 3.9586% | 3.9586% |
| compound/sheet_grad_mag-L4 | 6.1495% | 6.1559% | 6.1559% | 6.1559% | 6.1559% | 6.1559% |
| compound/sheet_nx-L3 | 30.0346% | 30.0566% | 30.0320% | 30.0326% | 30.0442% | 30.0454% |
| compound/sheet_nx-L4 | 44.1654% | 44.1966% | 44.1550% | 44.1551% | 44.1809% | 44.1821% |
| compound/sheet_ny-L3 | 32.4321% | 32.4562% | 32.4289% | 32.4290% | 32.4424% | 32.4436% |
| compound/sheet_ny-L4 | 43.4085% | 43.4379% | 43.3991% | 43.3991% | 43.4240% | 43.4251% |
| compound/width-L3 | 27.0824% | 27.1005% | 27.0780% | 27.0787% | 27.0876% | 27.0889% |
| compound/width-L4 | 38.2985% | 38.3165% | 38.3091% | 38.3095% | 38.3120% | 38.3127% |

## Minimal section menu

# Header representation experiments

All eight chunks × 18 groups × four presets; residual bytes unchanged. Every header reconstructs exactly. Existing 12-byte outer framing retained in size accounting; codecs are alternatives, not selected per chunk.

Times are native header-only microseconds/chunk (median of three batches), including allocations. These are NOT full-codec throughput measurements. Original header parsing/reconstruction remains after unpacking.

## Combined

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 6173 | 32.5631% | +0.000% | 22.9 | 7.9 |
| structured-xor | 6624 | 32.5847% | +0.066% | 77.4 | 10.8 |
| struct-xor-delta-only | 6195 | 32.5642% | +0.003% | 136.3 | 12.7 |
| struct-xor-rans-only | 6341 | 32.5711% | +0.025% | 74.9 | 13.0 |

### Combined timing spread (µs/header, min / median / max)

Each value aggregates one of the three timing batches over all headers.

| Variant | Encode | Decode |
|---|---:|---:|
| zstd3 | 22.6 / 22.8 / 25.1 | 7.9 / 8.0 / 8.0 |
| structured-xor | 76.9 / 77.2 / 81.6 | 10.7 / 10.7 / 11.5 |
| struct-xor-delta-only | 134.7 / 136.3 / 140.7 | 12.5 / 12.7 / 13.0 |
| struct-xor-rans-only | 74.9 / 74.9 / 75.5 | 13.0 / 13.0 / 13.2 |

## fast

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 3091 | 33.6504% | +0.000% | 17.5 | 6.3 |
| structured-xor | 3536 | 33.6716% | +0.063% | 53.6 | 8.0 |
| struct-xor-delta-only | 3200 | 33.6556% | +0.015% | 91.5 | 9.0 |
| struct-xor-rans-only | 3320 | 33.6613% | +0.032% | 54.7 | 10.0 |

## medium

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7051 | 32.3715% | +0.000% | 24.9 | 8.4 |
| structured-xor | 7403 | 32.3883% | +0.052% | 83.5 | 11.8 |
| struct-xor-delta-only | 6955 | 32.3669% | -0.014% | 157.9 | 14.1 |
| struct-xor-rans-only | 7139 | 32.3757% | +0.013% | 80.3 | 13.2 |

## slow

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7270 | 32.1316% | +0.000% | 24.7 | 8.5 |
| structured-xor | 7777 | 32.1558% | +0.075% | 86.3 | 11.6 |
| struct-xor-delta-only | 7312 | 32.1336% | +0.006% | 147.7 | 13.8 |
| struct-xor-rans-only | 7442 | 32.1398% | +0.026% | 82.3 | 14.2 |

## slower

| Variant | Mean header bytes | Total size / raw | Total change vs Zstd | Encode µs | Decode µs |
|---|---:|---:|---:|---:|---:|
| zstd3 | 7281 | 32.0990% | +0.000% | 24.6 | 8.5 |
| structured-xor | 7782 | 32.1229% | +0.074% | 86.4 | 11.6 |
| struct-xor-delta-only | 7315 | 32.1006% | +0.005% | 148.2 | 13.8 |
| struct-xor-rans-only | 7461 | 32.1076% | +0.027% | 82.4 | 14.4 |

## Per-group total size / raw (all four presets combined)

| Group | zstd3 | structured-xor | struct-xor-delta-only | struct-xor-rans-only |
|---|---:|---:|---:|---:|
| CT-L0 | 21.9300% | 21.9429% | 21.9429% | 21.9634% |
| CT-L1 | 45.1002% | 45.1095% | 45.1095% | 45.1257% |
| nx-L3 | 38.1239% | 38.1552% | 38.1196% | 38.1245% |
| nx-L4 | 44.8130% | 44.8478% | 44.7999% | 44.8046% |
| width-L3 | 28.0760% | 28.1000% | 28.0752% | 28.0803% |
| width-L4 | 35.6834% | 35.7050% | 35.6988% | 35.7021% |
| ny-L3 | 35.2784% | 35.3043% | 35.2747% | 35.2797% |
| ny-L4 | 48.7956% | 48.8268% | 48.7770% | 48.7814% |
| compound/presence-L3 | 26.1481% | 26.1680% | 26.1616% | 26.1701% |
| compound/presence-L4 | 36.6658% | 36.6855% | 36.6779% | 36.6830% |
| compound/sheet_grad_mag-L3 | 3.9510% | 3.9586% | 3.9586% | 3.9673% |
| compound/sheet_grad_mag-L4 | 6.1495% | 6.1559% | 6.1559% | 6.1654% |
| compound/sheet_nx-L3 | 30.0346% | 30.0566% | 30.0326% | 30.0375% |
| compound/sheet_nx-L4 | 44.1654% | 44.1966% | 44.1551% | 44.1600% |
| compound/sheet_ny-L3 | 32.4321% | 32.4562% | 32.4290% | 32.4340% |
| compound/sheet_ny-L4 | 43.4085% | 43.4379% | 43.3991% | 43.4040% |
| compound/width-L3 | 27.0824% | 27.1005% | 27.0787% | 27.0837% |
| compound/width-L4 | 38.2985% | 38.3165% | 38.3095% | 38.3136% |

