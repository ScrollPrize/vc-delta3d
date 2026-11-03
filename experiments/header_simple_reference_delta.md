# Remove XOR from the simplified header candidate

2026-11-02. Native C, experimental only; no production format changes.

## Matched results

Same 576 frozen headers (18 groups × eight chunks × four presets), same
section encodings and within-word delta. Only the reference transform changes.
All complete sizes include signaling, rANS frequency tables and stream states.

| Configuration | Mean header bytes | Bytes vs simplified XOR | Bytes vs Zstd |
|---|---:|---:|---:|
| Zstd | 6173.1684 | +0.4583 | 0 |
| Complex dependency-free best | 6169.3646 | -3.3455 | -3.8038 |
| Simplified XOR reference (84) | 6172.7101 | 0 | -0.4583 |
| Simplified byte subtraction reference (87) | 6172.4705 | -0.2396 | -0.6979 |
| **Simplified word subtraction reference (88)** | **6172.3194** | **-0.3906** | **-0.8490** |

Recommend variant 88 as the updated simple candidate. Both reference
subtraction and within-record delta now call the same `differences()` helper
with 16-bit wrapping arithmetic; inverse decoding shares it too. An odd tail
byte is handled modulo 256. References use corresponding positions in an
earlier record, whereas within-record deltas use the previous word. Missing
reference positions predict zero. Unreferenced encoding remains available.
There is no XOR candidate or XOR inverse operation on this variant's path.
The experiment harness retains XOR solely for reproducible historical controls.

The two replacements are not selected together per header: variant 88 uses
word reference subtraction only, not an extra byte-vs-word search.

Compared with simplified XOR, word subtraction improves 92 headers, leaves
452 identical in size, and worsens 32; the aggregate is smaller. Byte
subtraction improves 45, ties 521 and worsens 10. This is not a universal
per-header win, but does remove XOR while improving aggregate compression.

The simplified XOR control exactly reproduces all 576 prior sizes, component
sizes and non-model encoded-byte hashes. Selector and control sizes remain
identical in both replacement variants. Word-reference parts average:
models 78.0330, selectors 3958.7500, entropy 2112.4253, controls 23.1111 bytes.
The header compressor's own rANS tables occupy 433.1580 bytes/header; states
remain 108.8889 bytes/header (1.7014 streams/header).

## Verification and reproduction

Every header is decoded and checked byte-for-byte before timing. The XOR
control and both new variants also passed the full corpus under ASan/UBSan
(1,728 headers). `git diff --check` passes. Production codecs are unchanged.

Build from the delta3d checkout with the same command as
[the simplification report](header_simplification.md), then:

```sh
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_codecs \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_simple_reference_delta_20261102 \
  --repeats 3 --cpu 97 --variants 0,67,84,87,88
```

Three batches × three native repetitions, CPU 97, `-O3 -ffp-contract=off`:
mean per-header median encode/decode times are 163.3/14.3 µs (XOR),
153.3/14.3 µs (byte reference), 149.7/14.0 µs (word reference). These are
header-only exploratory timings, not whole-codec speed claims. JSONL stores
all samples; report.md includes min/median/max and all per-group sizes.

Sanitizer build uses the flags/library in the simplification report; run with
`--variants 84,87,88 --repeats 1 --cpu 98` and output directory
`header_simple_reference_delta_20261102_sanitize` under the same experiment root.
