# Encoder/decoder source separation — 2026-11-03

Removed unused original VCNG framing, Huffman mode and standalone decoder
paths. These were not the retained Delta3D compatibility reader. Live NG-derived
arithmetic and SIMD kernels remain in use, with no new copies of kernels.

Split native entrypoints, kernel fitting/encoding/decoding, leaf framing and
indexed-container encoding/decoding. Shared prediction, scheduling, tree geometry
and schema walks remain shared. The small header and selector codecs remain
mixed rather than duplicating their shared machinery across files.

## Physical source lines

Includes comments, blank lines and SIMD `.inc` fragments; excludes docs, tests,
benchmarks and build scripts. Each file is assigned to one category. SIMD is
included in its owning role, not counted a second time.

| Category | Files | Lines |
|---|---:|---:|
| Encoder core | 25 | 2,862 |
| Decoder core | 16 | 1,675 |
| Mixed/shared core | 14 | 1,444 |
| Header/container encoder | 2 | 207 |
| Header/container decoder | 2 | 210 |
| Header/selector/container mixed | 10 | 759 |
| Legacy Delta3D (enc+dec) | 4 | 349 |
| C API/dispatch | 3 | 174 |
| C++ adapters | 4 | 378 |
| Python bindings/API | 3 | 394 |
| Total | 83 | 8,452 |

Previous reported total 9,490 omitted two SIMD `.inc` fragments (93 lines).
Like-for-like total: 9,583 -> 8,452, down 1,131 lines (11.8%). File count rises
because mixed files have been split, not because codec paths were duplicated.

Reproduce counts from repository root:

```sh
/home/hendrik/.venv_las/bin/python benchmark/source_sizes.py
```

## Validation

- 576 cases: 18 channel/scale groups × 8 chunks × 4 presets. All decoded bytes
  and compressed SHA256 hashes exactly match the pre-cleanup native extension.
- Release CTest 6/6, ASan/UBSan CTest 6/6, Python 136/136.
- C++ smoke passes all presets, legacy fallback, indexed channels, quantization
  and AVX2 dispatch. No ARM/macOS hardware test performed.
- No algorithm, numerical flags, stream format or SIMD implementation changed.
  Corpus timings are single repetitions with eight workers, not a controlled
  throughput benchmark; no speedup claim is made from them.

Exact build/native validation commands:

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j 8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
/data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/vc_delta3d_modern_smoke
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-sanitize -j 8
ulimit -s 32768
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-sanitize --output-on-failure
```

Corpus before/after records are under
`/data/hendrik/compression_experiments/villa4/encoder_speed/core_cleanup_20261103/`.
The untouched extension is in `builds/pre_core_cleanup` under that experiment
root. Reproduce the after comparison with:

```sh
/home/hendrik/.venv_las/bin/python benchmark/check_core_cleanup.py \
  --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/core_cleanup_20261103/after.jsonl \
  --compare /data/hendrik/compression_experiments/villa4/encoder_speed/core_cleanup_20261103/before.jsonl
```
