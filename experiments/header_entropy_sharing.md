# Post-transform entropy-table sharing

2026-11-02. Native C experiment; no new production frame/default.

## Result

576 frozen headers, same 18 groups × 8 chunks × 4 presets. Source is variant 109
from `header_compaction_pools.md`, 6088.50 mean bytes/header. Its transforms and
compaction choices remain fixed. Every trial actually encodes and decodes the
prepared byte streams, rather than substituting entropy estimates.

| Eligible streams | Table-only total saving | Table-only B/header | Shared table + states total saving | Shared table + states B/header |
|---|---:|---:|---:|---:|
| Already using rANS | 0 B | 0.0000 | 66 B | 0.1146 |
| Also raw / bit-packed / integer-compacted literals | 5 B | 0.0087 | 5652 B | 9.8125 |

**These savings exclude new bank references/framing. They are NOT achieved new
header sizes.** The complete source header remains unchanged. A practical format
would save less; no new sharing format was integrated given the tiny result.
The combined-payload result additionally changes stream/state grouping, whereas
table-only leaves each payload's eight rANS states independent.

The broader literal trial improves table-only cost on one header (5 bytes), and
combined-payload cost on 315 headers (at most 69 bytes/header). It is not a global
optimality claim for the three larger-search cases described below.

## Why the opportunity is small

Selected rANS leaf counts in the existing best header:

| Control streams | Numerical streams | Headers |
|---:|---:|---:|
| 0 | 0 | 47 |
| 1 | 0 | 87 |
| 1 | 1 | 429 |
| 2 | 0 | 8 |
| 2 | 1 | 5 |

There are **no headers with two separate numerical rANS streams**. Most eligible
merges are therefore control/control, not parameter-value/frequency-value. Many
small components already choose raw, constant or packed representations instead
of paying for an rANS table. Controls and numerical values are deliberately not
mixed. The original average table overhead was real, but generally there is not
a second compatible table to remove.

## Implementation and scope

- `basic_decode` collects only the selected leaf byte streams during the final
  verification decode, after all chosen transforms/compaction. Candidate trials
  during fitting are not included. Role tags follow the decoding context: header
  controls, lengths, run controls and selectors versus numerical values.
- Literal follow-up additionally admits raw bytes, constant values, bit-packed
  values and direct compact integers at their terminal representation. Mixed
  byte-RLE control/value payloads are deliberately excluded to preserve the rule
  against entropy-coding those together.
- `vc_rans_encode_table` reuses the existing rANS kernel with supplied frequencies.
  The old wrapper still normalizes its own histogram. No duplicate encoder/kernel
  implementation and no new Python codec. The shared table is normalized from
  pooled histograms and counted once in compact form; every independent payload's
  state and renormalization bytes are counted.
- Combined-payload trials concatenate prepared bytes and use the existing native
  stream encoder/decoder. This shares both tables and states, without changing the
  per-array predictions/transforms.
- All compatible partitions are searched for up to ten leaves. This covers all
  576 existing-rANS cases and 573/576 literal-expanded cases. The remaining three
  (up to 34 leaves) use repeated best pair/all-same-role merges. No leaves are
  dropped or sampled. Their search is not exhaustive.
- Literal mode 110 deliberately emits the same source header as 109: its extra
  JSON `entropy_audit` fields report the separate stream experiments. The normal
  header-size columns must not be mistaken for a new encoded sharing format.

## Validation and reproduction

The source header round-trips all 576 cases for both tests. Each attempted shared
table/payload is also decoded and compared byte-for-byte inside the native test.
Unit tests cover supplied-table normalization validation, missing-symbol rejection,
and byte-identical encoding with the original table. Release CTest passes;
ASan/UBSan CTest passes with bounded stack (unlimited stack can collide with ASan
shadow mappings on this machine). API/AVX2 smoke and all 576 frozen raw hashes pass.
Twenty representative full-header entropy audits also pass ASan/UBSan, covering
all 18 groups and all three larger-search cases. Both full-corpus audit runs
exactly reproduce the 576 original source-header sizes.

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j 8
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_entropy_literals_v2
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_entropy_literals_v2 \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_entropy_literals_v2_20261102 \
  --repeats 0 --cpu 99 --variants 110
```

Use variant 109 for already-rANS leaves only. Original audit outputs:

- `/data/hendrik/compression_experiments/villa4/encoder_speed/header_entropy_sharing_20261102/`
- `/data/hendrik/compression_experiments/villa4/encoder_speed/header_entropy_literals_v2_20261102/`

The first literal attempt (`header_entropy_literals_20261102`) stopped on a
leaf-count assertion; it is incomplete and superseded by v2. No performance claim
is made: these are verification/size experiments, not timed runs.
