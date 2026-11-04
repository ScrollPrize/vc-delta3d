# Pure-C port acceptance — 2026-11-02

## Scope and outcome

Branch: `refactor/c-only-presets`. The original `feature/ng-presets`
branch is preserved at `45339743b450ff707d21fb742dc1b3b2bf99306d`.
The original port retained fast, medium, slow, slower and NCF1/NGC1/NDB2 framing
(NDB2 and historical NCF1 readers have since been removed; see HEADER_FORMAT.md),
and legacy Delta3D encoding, decoding and automatic fallback. LPC, tooslow,
tree routing and experimental streaming/adaptive encoders are removed.

All codec implementation and the primary API are C11. C++ headers and the
nanobind Python binding are thin adapters. A standalone C consumer links
without libstdc++. No compression arithmetic, predictor decisions or wire bytes
changed. Roughly 18,000 lines were removed and 4,200 added versus the merged base.

## Validation

- 576 frozen frames (144 chunks × four presets): exact decoded SHA256 and
  byte-for-byte identical re-encoding. Also passed all 576 with AVX-512 disabled
  to exercise the AVX2 path.
- 384 historical frames with different predictor-block sizes: decoded SHA256
  matches the historical corpus.
- 96 legacy/indexed cases: byte-identical frames and decoded hashes versus the
  untouched build, including uint16, quantization, channels and rectangles.
- 124 Python tests, 18 C++ adapter test cases, four native C test executables,
  native preset smoke, and four ASan/UBSan test executables passed.
- Specialized 64³/256³ and generic 96³ geometry tests passed.
- C-only CMake configuration builds with Python/C++ tests disabled.
- Not executed here: macOS, ARM/NEON and Windows runtime testing. Their guards
  and fallbacks remain; those platforms still need CI validation.

## Measurement protocol

AMD EPYC 9475F, Linux x86-64, GCC 13.3 Release (`-O3`, FP contraction off).
Single process pinned to logical CPU 97. Each preset/build measured on all
18 groups, eight 128³ chunks/group, three complete repetitions. Python only
loads, times and validates native calls; compression is entirely native.
Baseline and port run in separate processes to avoid symbol interposition.
Combined throughput is aggregate raw bytes divided by summed times, not the
arithmetic mean of channel speeds. Min/median/max are reported below.
Per-group differences of a few percent can include clock/load noise; the
table does not assert every individual decode became faster.

The frozen input corpus and timing artifacts are external benchmark data,
not bundled with this repository. Reproduction requires a compatible input
manifest and saved baseline build; the commands below take these explicitly.

## Legacy performance regression caught and fixed

The intermediate C encoder regressed to about 276 MB/s versus 425 MB/s.
A gprof run attributed 48% of combined encode/decode time to rANS encoding.
Assembly inspection found FORTIFY dynamic-object-size checked memcpy calls in
the backwards-growing four-byte stream stores. Explicit unaligned byte stores
preserve native-endian bytes and compile to ordinary word stores, without
disabling FORTIFY globally. The eight-lane loop is explicitly unrolled on
GCC/Clang. All bounds remain protected by the existing worst-case allocation.

Final alternating baseline/port runs, six real 128³ chunks, three runs each,
best of nine repetitions per run; **decimal MB/s** as printed by bench.cpp:

| Build | Encode min / median / max | Decode min / median / max |
|---|---:|---:|
| Original legacy Delta3D | 422.4 / 423.1 / 423.2 | 868.2 / 877.2 / 885.7 |
| C legacy Delta3D | 424.6 / 427.7 / 432.1 | 853.9 / 859.0 / 874.2 |

Encoding recovered (+1.1% median); legacy decode is within about 2.1% in this
alternating measurement (earlier runs favored the C decoder). Compression
remains identical. The four-preset table below is independent of this legacy
rANS implementation.

## Reproduction

From the repository root, with the documented build dependencies installed.
Replace the angle-bracket placeholders with your own benchmark paths:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DVC_DELTA3D_BUILD_TESTS=OFF \
  -DVC_DELTA3D_MODERN_SMOKE=ON -DVC_DELTA3D_BUILD_PYTHON=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 benchmark/verify_port.py --build build --frames <frozen-results.json>
python3 benchmark/compare_port.py \
  --baseline <baseline-build> --build build \
  --frames <frozen-results.json> --output <results-directory> --repetitions 3
```

Add `--isa-limit 256` to verify_port for AVX2 validation. Historical sizes use
`--decode-only --frame-directory <choice_grid_v1/frames>` and that experiment's
results.json. Use compare_port's `--report-only` to regenerate tables without
rerunning codecs. Sanitizer tests on this host require a bounded stack
(`ulimit -s 32768`) to avoid an ASan shadow mapping collision before main.

# C port: frozen corpus native timings

MiB/s; three complete-group repetitions unless configured otherwise.

| Group | Preset | Size % | Original enc | C enc | Original dec | C dec |
|---|---|---:|---:|---:|---:|---:|
| CT-L0 | fast | 22.121 | 65.6 | 101.5 | 436.6 | 439.3 |
| CT-L0 | medium | 22.099 | 14.0 | 22.0 | 333.0 | 341.6 |
| CT-L0 | slow | 21.845 | 3.5 | 4.5 | 463.0 | 460.1 |
| CT-L0 | slower | 21.654 | 2.6 | 3.1 | 376.6 | 363.3 |
| CT-L1 | slow | 45.164 | 3.4 | 4.4 | 454.0 | 447.0 |
| CT-L1 | slower | 44.856 | 2.4 | 2.9 | 364.0 | 360.2 |
| CT-L1 | fast | 45.192 | 83.9 | 115.0 | 452.0 | 455.7 |
| CT-L1 | medium | 45.189 | 14.5 | 22.2 | 332.0 | 340.7 |
| nx-L3 | fast | 40.697 | 52.8 | 103.8 | 498.4 | 534.6 |
| nx-L3 | medium | 37.621 | 13.6 | 20.4 | 431.1 | 453.7 |
| nx-L3 | slow | 37.089 | 4.6 | 7.9 | 488.2 | 494.5 |
| nx-L3 | slower | 37.089 | 4.5 | 7.7 | 476.6 | 504.2 |
| nx-L4 | medium | 44.877 | 13.4 | 20.6 | 491.5 | 506.8 |
| nx-L4 | slow | 44.416 | 4.6 | 7.8 | 398.6 | 416.3 |
| nx-L4 | slower | 44.416 | 4.6 | 7.6 | 395.6 | 417.3 |
| nx-L4 | fast | 45.544 | 77.0 | 109.6 | 685.8 | 729.2 |
| width-L3 | slower | 27.237 | 4.6 | 8.1 | 396.6 | 403.7 |
| width-L3 | fast | 30.203 | 49.7 | 112.2 | 538.2 | 560.7 |
| width-L3 | medium | 27.626 | 13.3 | 20.5 | 427.3 | 441.6 |
| width-L3 | slow | 27.237 | 4.7 | 8.3 | 400.0 | 422.6 |
| width-L4 | fast | 35.844 | 77.6 | 119.5 | 684.5 | 740.4 |
| width-L4 | medium | 35.692 | 13.5 | 20.7 | 629.7 | 660.6 |
| width-L4 | slow | 35.599 | 4.6 | 7.6 | 567.4 | 570.9 |
| width-L4 | slower | 35.599 | 4.4 | 7.1 | 562.2 | 565.1 |
| ny-L3 | slow | 34.614 | 4.6 | 7.9 | 409.0 | 413.2 |
| ny-L3 | slower | 34.614 | 4.5 | 7.6 | 403.5 | 417.8 |
| ny-L3 | fast | 37.085 | 52.2 | 113.4 | 570.2 | 600.3 |
| ny-L3 | medium | 34.800 | 13.1 | 20.2 | 431.2 | 439.5 |
| ny-L4 | fast | 49.614 | 74.9 | 109.3 | 674.0 | 723.5 |
| ny-L4 | medium | 48.770 | 12.9 | 20.1 | 476.3 | 504.1 |
| ny-L4 | slow | 48.399 | 4.6 | 7.8 | 483.7 | 497.5 |
| ny-L4 | slower | 48.399 | 4.5 | 7.6 | 489.6 | 507.5 |
| compound/presence-L3 | medium | 25.736 | 13.5 | 20.2 | 362.9 | 373.2 |
| compound/presence-L3 | slow | 25.620 | 3.8 | 5.6 | 375.6 | 389.6 |
| compound/presence-L3 | slower | 25.533 | 3.1 | 4.2 | 319.5 | 329.5 |
| compound/presence-L3 | fast | 27.703 | 61.0 | 116.5 | 613.1 | 653.2 |
| compound/presence-L4 | slower | 36.571 | 3.5 | 5.0 | 496.8 | 503.1 |
| compound/presence-L4 | fast | 36.887 | 78.1 | 112.0 | 695.3 | 748.0 |
| compound/presence-L4 | medium | 36.632 | 13.4 | 20.2 | 495.7 | 486.5 |
| compound/presence-L4 | slow | 36.573 | 4.0 | 6.2 | 514.4 | 528.1 |
| compound/sheet_grad_mag-L3 | fast | 3.951 | 81.4 | 130.1 | 729.1 | 807.5 |
| compound/sheet_grad_mag-L3 | medium | 3.951 | 14.0 | 20.6 | 734.7 | 798.9 |
| compound/sheet_grad_mag-L3 | slow | 3.951 | 5.4 | 9.6 | 730.5 | 769.5 |
| compound/sheet_grad_mag-L3 | slower | 3.951 | 5.3 | 9.3 | 734.3 | 745.1 |
| compound/sheet_grad_mag-L4 | slow | 6.127 | 5.4 | 9.7 | 589.6 | 619.6 |
| compound/sheet_grad_mag-L4 | slower | 6.127 | 5.3 | 9.4 | 587.9 | 615.2 |
| compound/sheet_grad_mag-L4 | fast | 6.172 | 81.5 | 130.4 | 726.5 | 790.2 |
| compound/sheet_grad_mag-L4 | medium | 6.172 | 13.9 | 21.0 | 740.7 | 789.3 |
| compound/sheet_nx-L3 | fast | 31.815 | 53.1 | 113.0 | 548.6 | 590.5 |
| compound/sheet_nx-L3 | medium | 29.568 | 13.5 | 20.6 | 453.6 | 483.0 |
| compound/sheet_nx-L3 | slow | 29.378 | 4.7 | 8.1 | 441.9 | 444.6 |
| compound/sheet_nx-L3 | slower | 29.378 | 4.6 | 8.0 | 438.2 | 441.9 |
| compound/sheet_nx-L4 | medium | 44.079 | 13.3 | 19.9 | 451.8 | 449.9 |
| compound/sheet_nx-L4 | slow | 43.537 | 4.6 | 7.8 | 482.6 | 491.7 |
| compound/sheet_nx-L4 | slower | 43.537 | 4.5 | 7.5 | 483.5 | 517.2 |
| compound/sheet_nx-L4 | fast | 45.509 | 76.6 | 112.5 | 691.1 | 724.7 |
| compound/sheet_ny-L3 | slower | 31.608 | 4.5 | 7.7 | 417.4 | 443.6 |
| compound/sheet_ny-L3 | fast | 34.724 | 49.8 | 106.6 | 546.4 | 566.6 |
| compound/sheet_ny-L3 | medium | 31.788 | 13.5 | 20.5 | 465.9 | 471.4 |
| compound/sheet_ny-L3 | slow | 31.608 | 4.6 | 7.8 | 423.0 | 429.7 |
| compound/sheet_ny-L4 | fast | 45.059 | 76.6 | 111.0 | 696.3 | 740.0 |
| compound/sheet_ny-L4 | medium | 43.282 | 13.2 | 20.4 | 457.4 | 456.2 |
| compound/sheet_ny-L4 | slow | 42.646 | 4.6 | 7.8 | 440.7 | 472.8 |
| compound/sheet_ny-L4 | slower | 42.646 | 4.6 | 7.6 | 439.3 | 470.1 |
| compound/width-L3 | slow | 26.422 | 4.7 | 8.2 | 418.6 | 414.8 |
| compound/width-L3 | slower | 26.422 | 4.7 | 8.0 | 422.0 | 424.5 |
| compound/width-L3 | fast | 28.956 | 52.9 | 117.1 | 567.2 | 599.3 |
| compound/width-L3 | medium | 26.529 | 13.4 | 20.3 | 436.3 | 457.9 |
| compound/width-L4 | fast | 38.631 | 77.7 | 117.4 | 684.1 | 718.3 |
| compound/width-L4 | medium | 38.278 | 13.1 | 20.4 | 511.2 | 526.6 |
| compound/width-L4 | slow | 38.143 | 4.7 | 8.1 | 548.9 | 558.2 |
| compound/width-L4 | slower | 38.143 | 4.6 | 7.9 | 543.9 | 561.0 |

## Combined throughput (min / median / max MiB/s)

Aggregate bytes divided by summed group times for each repetition.

| Build | Preset | Encode | Decode |
|---|---|---:|---:|
| original | fast | 65.3 / 65.4 / 65.6 | 593.5 / 597.8 / 598.8 |
| original | medium | 13.5 / 13.5 / 13.5 | 458.4 / 458.9 / 459.6 |
| original | slow | 4.4 / 4.4 / 4.5 | 463.6 / 464.5 / 468.0 |
| original | slower | 4.1 / 4.1 / 4.1 | 443.8 / 445.7 / 450.0 |
| c_port | fast | 113.1 / 113.3 / 113.8 | 624.4 / 631.4 / 633.7 |
| c_port | medium | 20.6 / 20.6 / 20.6 | 471.3 / 474.5 / 477.4 |
| c_port | slow | 7.2 / 7.2 / 7.2 | 473.5 / 476.9 / 479.7 |
| c_port | slower | 6.2 / 6.3 / 6.3 | 455.5 / 457.1 / 463.1 |

