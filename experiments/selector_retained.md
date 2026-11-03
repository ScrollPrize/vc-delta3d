# Retained selector integration / experiment cleanup (2026-11-03)

Selected the unified small-state context/LZ implementation. Replaced the
experimental inclusion chain with299 lines of C (`src/ng/selector.c`), a small
header, tests, and one native benchmark. This is shared code, not a parallel
production copy: the old prototype implementations have been removed.

Retained: constant map, previous-ID contexts, bit-packed greedy LZ with three
recent distances, one shared scalar rANS backend and exact whole-map choice.
Removed: order2/run contexts, raw/context-literal LZ, local switching, optional
parsers/search depths, old table experiments, unused custom bulk-rANS-table API.
Historical general-header experiments are archived too, not implicitly installed
as a new general-header codec. Reports remain in this directory for provenance.

New nonuniform NCF1 maps use marker253 and an external selector payload;
uniform/old254/255 forms remain readable. Models/entropy-header Zstd remains
unchanged. See `SELECTOR_FORMAT.md` for exact layout/compatibility.

## Verification / full-frame comparison

576 saved frames,18 groups×8×fast/slow×2³/4³. Every selector size exactly equals
the selected unified experiment. Frozen voxel residuals and fitted parameters
unchanged; both original and rewritten full frames decode to identical voxels.
Both release and ASan/UBSan corpus pass. CTest: five suites pass in both builds,
including all four presets, C API, independent leaves and legacy Delta3D.

Full frame, aggregate raw MiB/s and mean encoded bytes:

|Configuration|Old bytes|New bytes|Old decode MiB/s|New decode MiB/s|
|---|---:|---:|---:|---:|
|Fast2³|695374.56|693516.20|374.3|310.9|
|Fast4³|705699.47|705408.58|471.2|462.0|
|Slow2³|648604.97|646350.00|305.1|257.3|
|Slow4³|673849.07|673404.92|367.6|359.5|

Old is the original packed-selector-in-Zstd full frame, **not** the earlier
context/LZ combination. Three groups of three timed decodes, median per frame,
sum bytes/sum medians; allocation and validation included,8pinned workers.
Old/new order alternates. Loaded host: indicative timing, not a release guarantee.
Standard4³ slows~2%;2³ slows~16–17%. No voxel math/prediction changes.

## Reproduction

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j 8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/selector_check.c \
 /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
 -lzstd -lm -pthread -o /data/hendrik/compression_experiments/villa4/builds/selector-retained
python3 benchmark/check_selectors.py \
 --reference /data/hendrik/compression_experiments/villa4/encoder_speed/selector_unified_20261103/results.jsonl \
 --binary /data/hendrik/compression_experiments/villa4/builds/selector-retained \
 --output /data/hendrik/compression_experiments/villa4/encoder_speed/selector_retained_20261103 \
 --workers 8 --repetitions 3
```

Sanitizer: existing `delta3d-c-sanitize` library/build, O1/debug and
`-fsanitize=address,undefined -fno-omit-frame-pointer`; stack32768KiB,
repetitions0. Output `encoder_speed/selector_retained_sanitize_20261103`.
All corpus/report data remain outside the checkout. The runner does not encode
or decode in Python. macOS/arm64 not exercised here; scalar code uses no
architecture-specific intrinsics. Linux CPU pinning is guarded in the runner.

## Recoverable cleanup

Removed15 prototype source/script files from `benchmark/` after archiving them:
`/data/hendrik/compression_experiments/villa4/archives/selector_cleanup_20261103_v1/experiment_sources.tar.gz`.
`experimental_support.tar.gz` preserves experiment-only bulk-rANS changes.
Historical report reproduction commands refer to this archived snapshot;
they are not commands against the current lean checkout. No test data or
measurement results were deleted. No commit made by this task.
