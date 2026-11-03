# Restore one numerical pool — 2026-11-02

Correction: the earlier observation of one numerical rANS table described
the older pooled baseline, NOT the later simplification trials that removed
pooling. Reusing that observation for those trials was incorrect.

Same 576 frozen headers, native C, complete round-tripped wire sizes. Model
parameter records and frequency-count records now ALWAYS share one numerical
pool (not a conditional pooling search). Reuses the existing grouping/codec
implementation. All retained counts for those records share their control
stream. Schema controls and selector IDs retain their existing treatment;
this is not a claim that the entire header has only one entropy table.
No RLE is enabled. No production change.

## Matched results

| Configuration | Separate numerical families | One numerical pool |
|---|---:|---:|
| Standalone packing allowed | 6066.4306 | 6058.8750 |
| No standalone packing; pack+rANS retained | 6076.8090 | **6061.8750** |
| Neither standalone packing nor pack+rANS | 6643.5069 | 6628.5729 |

Units: complete bytes/header, mean over all 576. No timings collected.
One pooled numerical stream avoids both duplicate tables and duplicate rANS
states. This tests the actual pooled wire format, not an entropy-only bound.
Standalone packing saves only 3.0000 B/header once numerical pooling is restored;
discarding it is a reasonable simplification. Packing followed by rANS remains
material (566.6979 B/header versus no packing in the pooled comparison).
The no-pack pair retains the unused constant codec from its matched prior
control132; both packed pairs remove it. No extra model/refinement changes.

## Actual table counts

| Configuration | 0 numerical tables | 1 numerical table | 2 numerical tables |
|---|---:|---:|---:|
| Separate, packing allowed | 64 | 340 | 172 |
| Separate, standalone packing removed | 64 | 332 | 180 |
| Shared, packing allowed | 64 | 512 | 0 |
| Shared, standalone packing removed | 60 | 516 | 0 |

An assertion checks the one-numerical-table maximum in every shared trial.
Zero tables means no numerical rANS stream was selected, not missing data.
With shared/no-standalone-packing, control/selector table counts are:
47 headers have zero, 527 have one, 2 have two. These are distinct from numerical
tables. Post-transform label packing can improve byte-rANS compression; this
does not imply it reduces the information-theoretic entropy of the source.

## Reproduce / verification

From `local/codec_sources/delta3d-v020`:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_shared_numeric
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_shared_numeric \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_shared_numeric_20261102 \
  --repeats 0 --cpu 99 --variants 147,151,132,155,156,157
```

All 3456 exact round-trips pass. ASan/UBSan checks cover all three new variants
on one header per group (54 checks). Known-size parsing and zero-tail restoration
unchanged. The numerical-record capacity is asserted, never silently falls back
to multiple pools. Zstd only reads frozen originals. Production remains unchanged.
