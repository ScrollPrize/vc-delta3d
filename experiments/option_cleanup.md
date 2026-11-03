# Retained-path cleanup — 2026-11-03

## Scope and checkpoints

This follows `core_cleanup.md`. No wire-format change and no new compatibility
reader. Old Delta3D remains supported. Preserve flexible predictor/model/bank
menus and selector block sizes independently of what today's presets emit.

1. Remove unused encoder targets, centered/nonlinear fitting features, weighted
   Gram/refit support, damping, frozen fits, raster entropy order and partial
   entropy sampling. Keep ordinary and additional-input NG fits with their
   exact original statistics, tap selection, rounding and regularization.
   All 576 corpus cases remain byte-identical (`encoder_pruned.jsonl`).
2. Remove unused old recurrence kernels (scalar, AVX2, AVX-512, NEON); these
   were not the active wavefront kernels. Replace decoder TLS override chains
   with an explicit per-call context. All 576 cases pass (`decoder_context.jsonl`).
3. Remove the large trial bitmask and unavailable reconstruction/preparation
   alternatives. Preserve the measured common-model, independent-model,
   homogeneous-block, fused-base, AVX2/AVX-512 and portable paths. Share the
   remaining recurrence template; fixed geometry still instantiates the same
   source. All 576 cases pass (`decoder_pruned.jsonl`).
4. Share the bounded cursor, fixed integer and limited varint readers across
   metadata, selectors and leaf framing. Preserve 32-bit metadata limits versus
   size_t selector limits. Remove diagnostic switches/no-op timing hooks.

The more invasive menu-descriptor rewrite is intentionally a separate next
step: it must preserve generic menus and the now-measured SIMD paths, not merely
reduce source lines by putting dynamic dispatch in the inner loop.
That follow-up is now completed in `final_cleanup.md`: simplify construction
and ownership while retaining compact numeric selectors inside SIMD kernels.

## Line counts

Physical source lines, including comments, blanks and SIMD .inc fragments;
each file counted once. Tests/docs/benchmarks excluded. SIMD belongs to the
owning encoder/decoder/shared category.

| Category | Before | After |
|---|---:|---:|
| Encoder core | 2,862 | 2,688 |
| Decoder core | 1,675 | 1,320 |
| Shared core | 1,444 | 1,419 |
| Header/container encoder | 207 | 207 |
| Header/container decoder | 210 | 200 |
| Header/selector/container mixed | 759 | 751 |
| Legacy Delta3D | 349 | 349 |
| C API | 174 | 174 |
| C++ adapters | 378 | 378 |
| Python bindings/API | 394 | 394 |
| Total | 8,452 | 7,880 |

Overall -572 lines (-6.8%); decoder core -355 (-21.2%). Reproduce with
`/home/hendrik/.venv_las/bin/python benchmark/source_sizes.py`.

## Tests and measurements

Added `tests/test_format.c`: 684 synthetic generic-format cases including
2/4/8/16/32 block edges, 0–4 independent models, 1/3 entropy banks, 2/16/32
classes, both prediction modes, independent per-choice feature weights,
rectangular chunks and automatic/AVX2-only dispatch. This avoids relying only
on preset-selected cases. A build with `-DVCN_NO_AVX2` exercises portable NG
reconstruction (other independently guarded encoder SIMD remains available).

Before/after extension builds:

- `/data/hendrik/compression_experiments/villa4/builds/pre_option_cleanup`
- `/data/hendrik/compression_experiments/villa4/builds/delta3d-c-only`

The benchmark uses the same 18-group × 8-chunk × 4-preset corpus, Release,
three encode/decode repetitions per case, one worker pinned to CPU64. Compressed
SHA256 and exact decoded bytes are checked, not just compression size. Decode
time includes the complete native decoder, not only its reconstruction kernel.
No compression algorithm is implemented in the Python harness.

Final results: all 576 cases produced byte-identical compressed frames and
exact decoded bytes. Throughput below is total raw bytes divided by total
elapsed native time across three repetitions (MiB/s).

| Preset | Encode before | Encode after | Decode before | Decode after |
|---|---:|---:|---:|---:|
| fast | 119.1 | 122.0 | 589.5 | 627.5 |
| medium | 21.8 | 22.0 | 430.2 | 447.4 |
| slow | 7.4 | 7.6 | 423.1 | 456.1 |
| slower | 6.5 | 6.7 | 412.9 | 445.0 |

Combined decode min/median/max over the three repetitions:

| Preset | Before | After |
|---|---|---|
| fast | 582.0 / 591.0 / 595.5 | 623.6 / 625.4 / 633.4 |
| medium | 429.4 / 430.5 / 430.8 | 445.0 / 448.1 / 449.0 |
| slow | 421.8 / 422.5 / 425.0 | 453.7 / 456.6 / 458.0 |
| slower | 411.7 / 411.7 / 415.4 | 443.8 / 445.3 / 445.9 |

CT L0 decode before/after: fast 436.5/456.2, medium 328.1/337.3,
slow 300.8/306.8, slower 287.6/292.6 MiB/s. CT L1: fast 426.3/448.7,
medium 326.8/334.5, slow 418.8/431.1, slower 343.0/354.2 MiB/s.
All 72 group/preset mean decode rates improved; smallest improvement 1.23%.
Host load was low, but the whole machine was not exclusively reserved; these
are measured results on this x86 host, not guarantees for other hardware.

Full per-group results and repetition statistics are in
`/data/hendrik/compression_experiments/villa4/encoder_speed/core_cleanup_20261103/timing_summary.json`,
alongside `speed_before.jsonl` and `speed_after.jsonl`.

Validation passed: Release, ASan/UBSan and portable NG reconstruction CTest
suites (7/7 each), Python tests (136/136), C++ adapter smoke tests, and
the final expanded 684 generic-format cases on all three native builds.
The generic tests cover full format flexibility, not just preset outputs.

Commands (run the two measurements sequentially, not concurrently):

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j 8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
/home/hendrik/.venv_las/bin/python benchmark/check_core_cleanup.py --build /data/hendrik/compression_experiments/villa4/builds/pre_option_cleanup --output /data/hendrik/compression_experiments/villa4/encoder_speed/core_cleanup_20261103/speed_before.jsonl --workers 1 --repetitions 3 --cpu 64
/home/hendrik/.venv_las/bin/python benchmark/check_core_cleanup.py --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output /data/hendrik/compression_experiments/villa4/encoder_speed/core_cleanup_20261103/speed_after.jsonl --compare /data/hendrik/compression_experiments/villa4/encoder_speed/core_cleanup_20261103/speed_before.jsonl --workers 1 --repetitions 3 --cpu 64
/home/hendrik/.venv_las/bin/python benchmark/report_cleanup.py /data/hendrik/compression_experiments/villa4/encoder_speed/core_cleanup_20261103
```

Portable build:

```sh
cmake -S . -B /data/hendrik/compression_experiments/villa4/builds/option-cleanup-portable -DCMAKE_BUILD_TYPE=Release -DVC_DELTA3D_BUILD_TESTS=OFF -DVC_DELTA3D_BUILD_PYTHON=OFF -DVC_DELTA3D_MODERN_SMOKE=ON -DCMAKE_C_FLAGS=-DVCN_NO_AVX2
cmake --build /data/hendrik/compression_experiments/villa4/builds/option-cleanup-portable -j 8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/option-cleanup-portable --output-on-failure
```

Sanitizers use the existing `delta3d-c-sanitize` build with `ulimit -s 32768`.
No ARM/macOS runtime claim: those require their own hardware/CI checks.
