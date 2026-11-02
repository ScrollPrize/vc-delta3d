# C-only codec reduction

## Contract

- Branch: `refactor/c-only-presets`, based on merged main `98bb019`.
- Preserve `feature/ng-presets` unchanged as the reference implementation.
- Keep fast, medium, slow, slower and their fixed-preset framing, including
  indexed spatial/channel payloads. Keep legacy Delta3D decoding and encoding
  fallback for 16-bit, irregular dimensions and quantization.
- Remove LPC, tooslow/tree coding, streaming/search experiments and unrelated
  optional encoders. Unsupported retired formats must fail explicitly.
- The codec implementation and primary API must be C. Only thin C++/Python
  adapters may use C++; retain the existing consumer-facing API where applicable.
- Preserve arithmetic, predictor choices, emitted frames and SIMD dispatch while
  porting. Measure encoding and decoding; do not exchange compression for speed.

## Sequence

1. Freeze baseline frames/results; identify the retained call graph.
2. Extract legacy filters/rANS into shared C, with original wrappers calling C.
3. Port retained preset search/framing orchestration to C, reusing C SIMD kernels.
4. Port indexed containers and expose the checked C API; thin consumer adapters.
5. Remove unreachable experimental code and retired formats/presets.
6. Test old-frame decoding, exact new-frame parity, malformed inputs, native C
   consumers, Python adapters, portability and all-channel speed/size comparisons.

Do not describe the intermediate mixed-language build as a finished C port.

## Baselines

- Original source: `feature/ng-presets` (`4533974`, tree identical to merge).
- Frozen 144-chunk / five-preset frames:
  `/data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json`.
  Four retained presets give 576 frames. Retired tooslow is not a compatibility target.
- Original timing/compression report: `BENCHMARKS.md`.
- Existing regression tests: scalar, indexed, fallback and Python tests.

## Progress

- Created separate branch; original branch preserved. No consumer pins changed.
- Inspection: numerical kernels already reside mostly in C; the largest C++
  reduction is experimental search orchestration and container ownership.
- Starting with shared C delta/quantization primitives; retain existing operation
  order and SSE2/NEON prefix-sum implementations.
- Extracted those primitives into `src/delta_filter.c`; C++ orchestration now calls
  the shared C functions. No duplicate filter implementation remains.
- Validation: C11 standalone tests cover all 8 masks, 6 shapes including SIMD
  tails, uint8/uint16 and all 255 quantizers; native smoke and 145 Python tests
  pass. `benchmark/verify_port.py` checks all 576 retained-preset frozen frames:
  decoded SHA256 and re-encoded bytes match exactly.
- First performance check is NOT final acceptance. GCC 13.3 Release, CPU 97,
  six real 128³ Paris4 channel chunks, alternating baseline/port, 3 runs each,
  each run best of 9 repetitions with the existing `benchmark/bench.cpp`:

  | Native Delta path | Encode MB/s min/median/max | Decode MB/s min/median/max |
  |---|---:|---:|
  | Original | 424.3 / 424.9 / 425.0 | 868.2 / 876.8 / 877.7 |
  | C filters | 426.5 / 426.7 / 426.9 | 849.4 / 849.5 / 855.5 |

  The roughly 3% decode regression needs resolving before final acceptance;
  do not describe performance as unchanged. Compression was identical.

### Commands for this milestone

```sh
cmake -S . -B <build> -DCMAKE_BUILD_TYPE=Release \
  -DVC_DELTA3D_BUILD_TESTS=OFF -DVC_DELTA3D_MODERN_SMOKE=ON \
  -DVC_DELTA3D_BUILD_PYTHON=ON -DPython_EXECUTABLE=<python>
cmake --build <build> -j 12
ctest --test-dir <build> --output-on-failure
<build>/vc_delta3d_modern_smoke
python benchmark/verify_port.py --build <build> --frames <frozen-results.json>
```

Timing harness linked unchanged against each static library (C++20, `-O3`);
`taskset -c 97 <bench> <six-raw-chunk-dir> 9 delta`. Source raw chunks and
`shape.txt` are under `/data/hendrik/compression_experiments/villa4/runs/villa-delta3d-integration`.
