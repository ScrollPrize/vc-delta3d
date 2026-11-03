# Preset-only encoder cleanup — 2026-11-03

Baseline: `c158df7` (production code identical to `e0c855e`).
Branch: `refactor/c-only-presets`. This completes the additional source-audit
findings in `task.md`; earlier reports remain historical measurements.

## Removed and simplified

- Deleted `ng_rect.c` and its separate fixed-menu encoder, plus its private
  `ng_block_symbols` implementation. The encoder now has only fast, medium,
  slow and slower, besides the explicitly retained legacy Delta3D fallback.
- Rectangular logical chunks still encode through the automatic cubic
  partitioner. Every leaf uses the requested preset. Explicit encoder leaves
  must be cubic. Removed the rectangular-baseline policy flag from C/C++ and
  Python partition nodes: Python nodes are now `(axis, cut)`, not triples.
  This is an intentional API change, not a compatibility shim.
- Kept rectangular decoding and generic low-level frame serialization. Tests
  construct rectangular indexed frames independently of the preset encoder;
  full decode and individual range-read leaf decode both remain covered.
- Removed dead `delta_predict_xyz`, `vcn_u8`, frame-dimension probing and
  redundant declarations. Removed private C++ compatibility branches; public
  C++ adapters and public C headers' C++ linkage remain supported.
- Removed unused histogram modes, raw-residual threshold conversion, alternate
  threshold candidate grids and non-4-cubed occupancy branches. The retained
  global/coordinate searches are both live. Decoder block-size support is
  unchanged.
- Encoder symbol evaluation parses each fitted model once, not twelve times.
  Candidate order, causal history, arithmetic and chosen outputs are unchanged.
  Removed unused empty-model initialization, fields and ignored arguments.
- Removed redundant CMake options/flags and unused Threads dependency. Exact
  floating-point contraction settings remain on the main and specialized cores.

No LPC, tree coder, streaming adaptation, historical new-format decoder or
optional experimental encoder remains in production. ISA kernels and geometry
specializations remain because they serve real supported execution paths;
they instantiate shared source. Delta3D compatibility is deliberately retained.

## Size and correctness

Production source: **7,759 → 7,540 lines**, **82 → 81 files** (physical lines,
including comments and SIMD fragments; tests/docs excluded).

| Category | Lines |
|---|---:|
| Encoder core | 2,526 |
| Decoder core | 1,259 |
| Shared core | 1,337 |
| Header/container encoder | 206 |
| Header/container decoder | 192 |
| Header/selector/container shared | 740 |
| Legacy Delta3D | 337 |
| C API | 173 |
| C++ adapters | 376 |
| Python adapter/API | 394 |

Release, ASan/UBSan and portable CTest: **7/7 each**. Python: **139 passed**.
All **576** corpus encodes (18 groups ×8 chunks ×4 presets, uint8 128³)
are lossless and byte-identical to the baseline. Compression ratios therefore
do not change. Rectangular custom-plan API rejection and preset-correct
automatic partitioning are explicitly tested.

## Decode performance

Release, same host, single CPU64, balanced alternating order, eight repetitions
on all 576 cached frames; no simultaneous build/test workload. Rates below
aggregate per-frame median native C decode times, in MiB/s.

| Preset | Before | After | Change |
|---|---:|---:|---:|
| fast | 674.2 | 676.4 | +0.33% |
| medium | 477.0 | 477.8 | +0.16% |
| slow | 498.0 | 497.3 | −0.15% |
| slower | 482.1 | 481.6 | −0.10% |

All differences are below the user's 0.5% tolerance. Across whole-corpus
repetitions, after mean / min / median / max MiB/s:

| Preset | Mean | Min | Median | Max |
|---|---:|---:|---:|---:|
| fast | 675.84 | 672.13 | 676.56 | 677.79 |
| medium | 477.65 | 476.83 | 477.72 | 478.32 |
| slow | 497.19 | 496.18 | 497.37 | 497.87 |
| slower | 481.47 | 480.67 | 481.48 | 482.01 |

Contention monitoring: machine mean busy 1.15%, CPU64 mean 99.0%, its SMT
sibling CPU160 mean 0.29%. No significant decoder slowdown demonstrated.

## Encoder performance

Same Release build/corpus and CPU64, three native encode repetitions per case,
serial baseline run followed by serial cleaned-build run. Exact frames checked
on every repetition. Aggregate MiB/s (total bytes / total time):

| Preset | Before mean | After mean | Change | After min / median / max |
|---|---:|---:|---:|---:|
| fast | 121.29 | 126.57 | +4.35% | 126.06 / 126.78 / 126.86 |
| medium | 21.85 | 22.49 | +2.94% | 22.47 / 22.48 / 22.52 |
| slow | 7.571 | 7.572 | +0.00% | 7.529 / 7.590 / 7.597 |
| slower | 6.678 | 6.681 | +0.05% | 6.664 / 6.676 / 6.704 |

Mean machine utilization: baseline 0.99%, after 0.97%; SMT sibling about 0.20%
for both. Per-channel and all samples are in `timing_summary.json` and
`speed_{before,after}.jsonl`. These are not claims of uniform per-channel gains.

The encode harness also times decode immediately after each encode, with Python
wrapper/allocation overhead. That non-interleaved measurement reports -0.3%,
-0.7%, -2.2%, -1.9% decode changes, respectively. It is not interchangeable with
the cached-frame paired native test above: both wrapper overhead and preceding
encoder/cache state differ. Preserve these results rather than hiding them;
the repeated isolated paired check gives the following results:

| Preset | Before MiB/s | After MiB/s | Change |
|---|---:|---:|---:|
| fast | 673.2 | 675.1 | +0.28% |
| medium | 477.5 | 476.7 | −0.17% |
| slow | 497.7 | 496.2 | −0.30% |
| slower | 482.3 | 480.8 | −0.32% |

Both isolated passes stay within 0.5% for every preset. The post-encode result
is not reproduced as a >0.5% standalone native decoder slowdown. Raw repeat
data and contention logs are `decode_repeat.jsonl` and
`contention_decode_repeat.jsonl`. No diagnostic alternate code was retained.

## Reproduction

From this repository, using the existing configured builds (no install):

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
python benchmark/source_sizes.py
```

Benchmark scripts: `benchmark/check_core_cleanup.py` (exact parity and
three-repeat encoder timings), `benchmark/check_decode_cleanup.py` (balanced
native decoder comparison), `benchmark/monitor_contention.py` (CPU/SMT/pressure
monitor). Exact benchmark commands are stored in the first record of each
`contention_*.jsonl`. The Python scripts orchestrate native C codecs only.

Results: `/data/hendrik/compression_experiments/villa4/encoder_speed/preset_only_cleanup_20261103/`.
`parity_final.jsonl` holds all 576 exact frame hashes; `decode.jsonl` holds all
paired native samples. Baseline binary: `builds/cleanup-e0c855e` under the same
experiment root. Tests also used `builds/delta3d-c-sanitize` and
`builds/option-cleanup-portable`.

Portability limitation: this machine validates Linux x86 scalar/AVX paths;
macOS/ARM execution is not measured here. No ARM numerical kernel was changed.
