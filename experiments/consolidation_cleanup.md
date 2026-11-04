# Measured helper consolidation

Baseline: `3b1c266`. Work started 2026-11-03; follow-up measurements 2026-11-04.

## Changes and actual savings

* Replace four private fitting preparation wrappers and their caller dispatch
  with one `ng_prepare`. Preserve masked/plain, cached-feature and fused-activity
  paths, their validation, and the existing native kernel. Removes 31 lines
  across `ng_native_encode.c`, `ng_fit.c`, and `residual_bridge.h`.
* Share block-edge and menu-entry validity predicates across the frame encoder,
  frame decoder and header schema. Adds six net lines but puts each rule in one
  place instead of three. All existing format capabilities remain supported.
* No new dependencies, public API changes, wire changes, compatibility code,
  predictor/threshold/search changes, or SIMD changes. In particular, no new
  `vc_reserve` usage; indexed scratch allocation is untouched. The existing
  helper lives inside this independent library, not in Villa.

Total production physical lines (including comments and SIMD fragments):
**7,530 -> 7,505**, still 81 files. The release Python extension's text section
is 1,038,905 -> 1,036,370 bytes; data is 5,244 -> 5,220 bytes; BSS unchanged.
These are modest reductions, not a large architectural simplification.

Rejected/deferred consolidations:

* Implemented a common owned screening object across all three encoders, then
  removed it: it added 11 lines and another lifetime abstraction. No timing
  claim is made for this discarded trial.
* The two identical block-map loops are only two lines per caller. A helper,
  declaration and calls would increase total source rather than simplify it.
* Header and selector bit handling have different ownership and padding/error
  checks. Moving the short header loops behind the selector's reader/writer
  plumbing offers little reduction and adds coupling. Left unchanged.
* Fixed geometry, scalar/AVX2/AVX-512 and distinct rANS stream implementations
  remain intentional, supported specializations, not abandoned experiments.

## Correctness

All 576 encodes are lossless and **byte-identical** to `3b1c266`: 18 groups x
eight 128-cubed uint8 chunks x four presets. Compression ratios are unchanged.
Paired timings also cross-decode each newly encoded frame with both libraries.

* Release CTest: 7/7.
* Portable CTest: 7/7.
* ASan/UBSan CTest: 7/7 (use `ulimit -s 8192` on this machine).
* Python API/indexed/fallback suites: 139 passed.
* C++ adapter smoke: default, four presets, legacy fallback, indexed channels,
  quantization and AVX2 passed.
* Existing generic-format tests exercise rectangular leaves, variable block
  edges, models/banks and per-predictor weights.

Ubuntu/GCC 13 Release (`-O3 -DNDEBUG`). No new macOS/ARM runtime test performed;
portable tests pass and architecture-specific arithmetic is untouched.

## Measurement protocol

Frozen before binary: `builds/cleanup-3b1c266`; after: `builds/delta3d-c-only`.
All experiment paths below are relative to
`/data/hendrik/compression_experiments/villa4`.

Results: `encoder_speed/consolidation_20261103/`. Native C API calls are timed;
Python only drives the benchmark. Identical input/order, builds alternate and
reverse order on successive repetitions. Encoder: two repetitions per build
and case in each run. Decoder: eight. Throughput is total raw bytes divided by
sum of mean native call times, not the arithmetic average of channel speeds.
Cross-decoding/checks run outside the timed region. CPU/SMT contention is logged.
The first encode/decode runs overlapped on distinct physical cores; the second
encode run overlapped part of the first. The confirmation run is separate.

Initial full-run measurements (MiB/s):

| Preset | Encode CPU64 before | After | Change | Encode CPU8 before | After | Change | Decode before | After | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fast | 112.862 | 114.485 | +1.438% | 113.574 | 114.757 | +1.041% | 667.078 | 666.855 | -0.034% |
| medium | 21.057 | 21.037 | -0.096% | 20.989 | 20.879 | -0.522% | 469.242 | 469.479 | +0.051% |
| slow | 7.335 | 7.319 | -0.222% | 7.384 | 7.360 | -0.326% | 487.937 | 488.850 | +0.187% |
| slower | 6.471 | 6.454 | -0.256% | 6.539 | 6.494 | -0.684% | 472.215 | 473.540 | +0.281% |

The second encoder run marginally exceeds the 0.5% regression threshold for
medium/slower. The separate full CPU64 confirmation measured:

| Preset | Before MiB/s | After MiB/s | Change |
|---|---:|---:|---:|
| fast | 112.099 | 113.203 | +0.984% |
| medium | 21.011 | 20.972 | -0.185% |
| slow | 7.355 | 7.345 | -0.145% |
| slower | 6.478 | 6.454 | -0.360% |

Pooling the equally sized three encoder runs by total native time gives:
fast 112.842 -> 114.144 (+1.154%); medium 21.019 -> 20.963 (-0.268%);
slow 7.358 -> 7.341 (-0.231%); slower 6.496 -> 6.467 (-0.433%).
Retained: full confirmation and pooled results meet the agreed 0.5% tolerance.
This is a small encoder regression for three presets, not an across-the-board
speedup. The CPU8 outlier remains reported above; no timing was discarded.

Whole-corpus repetition min/median/max MiB/s (six encoder repetitions across
the three runs, eight decoder repetitions). Encoder ranges include core/run
variability; do not compare absolute encoder and decoder speed across CPUs.

| Preset | Encode before | Encode after | Decode before | Decode after |
|---|---|---|---|---|
| fast | 110.150 / 112.827 / 115.689 | 113.169 / 114.485 / 114.912 | 663.794 / 667.585 / 669.192 | 662.840 / 667.105 / 669.043 |
| medium | 20.965 / 21.005 / 21.117 | 20.862 / 20.972 / 21.067 | 468.856 / 469.276 / 469.802 | 468.919 / 469.597 / 470.097 |
| slow | 7.328 / 7.355 / 7.385 | 7.311 / 7.345 / 7.363 | 486.593 / 488.237 / 488.905 | 487.594 / 489.097 / 489.540 |
| slower | 6.462 / 6.482 / 6.540 | 6.450 / 6.457 / 6.495 | 471.449 / 472.270 / 473.071 | 472.779 / 473.660 / 474.288 |

Mean whole-machine / measured-core / SMT-sibling utilization:

* Encode CPU64: 1.408% / 99.925% / 0.572%.
* Encode repeat CPU8: 1.321% / 99.875% / 0.396%.
* Encode confirmation CPU64: 1.079% / 99.551% / 0.237%.
* Decode CPU8: 1.780% / 99.643% / 0.570%.

All runs completed; no unfinished performance test remains.

## Reproduction

From the codec checkout:

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
/home/hendrik/.venv_las/bin/python benchmark/source_sizes.py
/home/hendrik/.venv_las/bin/python benchmark/check_core_cleanup.py \
  --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/consolidation_20261103/parity.jsonl \
  --compare /data/hendrik/compression_experiments/villa4/encoder_speed/redundancy_cleanup_20261103/parity_final.jsonl \
  --workers 8 --repetitions 1
```

Exact paired timing commands are in the first record of each
`contention_*.jsonl` file. They use `benchmark/monitor_contention.py` around
`benchmark/check_decode_cleanup.py` with `--before builds/cleanup-3b1c266`,
`--after builds/delta3d-c-only`, and cached frames in
`encoder_speed/final_cleanup_20261103/frames` (all paths supplied as absolute
paths). `--encode --repetitions 2 --cpu 64` selects the encoder; decoder uses
`--repetitions 8 --cpu 8` without `--encode`.
