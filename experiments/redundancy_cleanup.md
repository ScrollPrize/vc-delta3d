# Follow-up redundancy cleanup

Baseline: `6910777` on `refactor/c-only-presets`. No preset, format, API or
predictor-kernel additions. Six audit findings addressed:

1. Public standalone NCF1 decoding dispatches directly to its decoder rather
   than decompressing the header for dimensions and immediately repeating that
   work. The leaf parser enforces the exact decoded byte count; both undersized
   and oversized public output counts remain errors.
2. Fast and slow/slower no longer transmit an empty model when no choice uses a
   fitted model. Existing zero-model grammar suffices. Internal dummy geometry
   needed for entropy traversal remains; it is not serialized as a fitted model.
3. Fast uses the shared masked threshold preparation. An optional already-built
   histogram preserves its fused activity path. Masked collection builds the
   histogram in the same pass, rather than scanning the compacted arrays again.
4. All screens use one stable top-two ranking helper, preserving ID tie breaks.
5. All-zero frame construction is shared. No redundant selector allocation in
   fast/medium's zero case.
6. A single C macro list supplies fixed-size declarations and dispatch cases.
   Numerical kernels and geometry specializations themselves are unchanged.

Production source: **7,540 -> 7,530 physical lines, 81 files unchanged**. This
is a small line reduction, chiefly a removal of duplicated behavior and runtime
work, not a large architectural rewrite. Tests/benchmark tooling are additional
and excluded from this production count.

## Compatibility and correctness

All **576 cases** (18 groups × eight 128³ uint8 chunks × four presets) pass:

- old decoder on newly encoded frames;
- new decoder on previously encoded frames;
- exact voxel round trip;
- byte-identical selector/residual payloads, following the compressed header.

364 frames save exactly six header bytes; 212 frames are byte-identical. Savings
total 2,184 bytes: fast 768, medium 0, slow 708, slower 708. The framing grammar
and magic numbers are unchanged; no compatibility code was added.

Release, portable and ASan/UBSan CTest: 7/7 each. Python suite: 139 passed. C++
smoke covers presets, legacy fallback, indexed channels, quantization and AVX2.
Generic-format tests retain all selector sizes, rectangular decoder geometry,
multiple models/banks and varying weights. New assertions cover oversized
output counts, zero frames across presets and stable ranking ties.

The sanitizer initially could not start because this shell's unlimited stack
collided with ASan shadow memory. `ulimit -s 8192` resolves that environmental
startup failure; no source workaround was needed.

## Measurement protocol

Release build, native C API, fixed CPU affinity, full 576-case corpus. Alternating
before/after order reverses each repetition. Encoding uses two repetitions per
build/case; decoding uses eight. All timed encodes also receive bidirectional
decode checks outside the timed interval. Python only orchestrates native code.

A noninterleaved baseline encode run was interrupted when external load rose;
its partial `speed_before.jsonl` is preserved but is not used to claim speedups.
Paired measurements and CPU/SMT monitoring are used instead. Encoding is pinned
to CPU64; decoding to CPU8. Do not compare absolute throughput across cores.

### Decoding (MiB/s)

Means are total uncompressed bytes divided by total native time, not an
arithmetic mean of per-chunk speeds. Eight repetitions per case:

| Preset | Before | After | Change | After min / median / max across repetitions |
|---|---:|---:|---:|---:|
| fast | 664.57 | 665.44 | +0.13% | 661.32 / 665.40 / 669.75 |
| medium | 479.66 | 479.73 | +0.01% | 477.29 / 480.09 / 481.32 |
| slow | 500.03 | 499.96 | -0.01% | 496.86 / 500.55 / 501.62 |
| slower | 483.19 | 484.99 | +0.37% | 483.74 / 484.98 / 486.19 |

Repeated complete run (no code change):

| Preset | Before | After | Change | After min / median / max |
|---|---:|---:|---:|---:|
| fast | 664.91 | 672.07 | +1.08% | 669.51 / 672.01 / 673.80 |
| medium | 480.08 | 485.72 | +1.17% | 484.74 / 485.79 / 486.15 |
| slow | 498.89 | 504.83 | +1.19% | 503.82 / 504.90 / 505.59 |
| slower | 482.66 | 488.27 | +1.16% | 486.83 / 488.45 / 489.26 |

Machine mean busy: 2.83%, then 2.08%; decoding core: 99.21%, then 99.75%;
SMT sibling: 2.33%, then 0.10%. Both runs satisfy the 0.5% regression tolerance.
The speedup magnitude is not stable enough to claim an exact gain.

### Encoding

Full corpus, two alternating repetitions per build/case, native C API MiB/s:

| Preset | Before mean | After mean | Change | After min / median / max |
|---|---:|---:|---:|---:|
| fast | 108.644 | 112.730 | +3.76% | 112.220 / 112.733 / 113.245 |
| medium | 20.439 | 20.834 | +1.94% | 20.819 / 20.834 / 20.850 |
| slow | 7.167 | 7.204 | +0.51% | 7.199 / 7.204 / 7.208 |
| slower | 6.346 | 6.368 | +0.35% | 6.360 / 6.368 / 6.376 |

Before min / median / max: fast 106.744 / 108.679 / 110.614; medium
20.396 / 20.439 / 20.482; slow 7.166 / 7.167 / 7.168; slower
6.340 / 6.346 / 6.352. Machine mean busy 1.84%, benchmark core 99.93%, SMT
sibling 0.93%. These are gains relative to the interleaved baseline, not claims
that absolute throughput matches older measurements under other conditions.

All validation is complete; no pending run. Changes are uncommitted for review.

## Reproduction and artifacts

From the codec checkout:

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
/home/hendrik/.venv_las/bin/python benchmark/source_sizes.py
```

Results root:
`/data/hendrik/compression_experiments/villa4/encoder_speed/redundancy_cleanup_20261103/`.
`parity_final.jsonl` is the cross-compatibility/payload check;
`encode.jsonl` and `decode.jsonl` hold paired native times.
Exact timing commands are the first records of `contention_encode.jsonl` and
`contention_decode.jsonl`. Baseline binary is preserved in
`/data/hendrik/compression_experiments/villa4/builds/cleanup-6910777/`.

Tests ran on Linux x86; macOS/ARM runtime behavior was not measured. Portable
and AVX2 paths were tested locally. No ISA-specific arithmetic changed.
