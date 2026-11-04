# Decoder simplification trials — 2026-11-04

Baseline is the uncommitted, fully tested consolidation state after `3b1c266`:
7,505 production physical lines; decoder core 1,255, decoder-only container
and frame code 191. Frozen binary: `builds/decoder-pass-baseline`.
All data paths below are relative to
`/data/hendrik/compression_experiments/villa4`.

## Trial decisions

| Trial | Net production line change | Decision |
|---|---:|---|
| Inline single-use published-wave wrapper | -21, all decoder | Reject: repeated slowdown exceeds 0.5% |
| Shared scalar metadata preparation | +4 decoder | Reject: extra temporary representation and no line saving |
| Direct dummy-model construction | +3 overall (+5 decoder, -2 frame decoder) | Reject: adds another default-model initialization path; no line saving |
| Dense internal entropy-table handoff | -5 overall (shared code) | Keep; removes redundant compact serialization/parsing; wire format unchanged |
| Also fuse dense validation/table construction | Additional -2 shared | Reject: medium/slower measured beyond slowdown tolerance |

No SIMD transpose scaffolding or ISA implementations were consolidated.
No external dependency, allocator abstraction, public API, preset policy or
legacy Delta3D change was made. Rejected source changes were removed.

The dummy model still has an encoder user, so its serialized constructor cannot
simply be deleted. The trial removed the decoder's serialize/parse round trip
but duplicated initialization details and increased source size.

The scalar trial used one eleven-value helper for row-major and block-major
metadata, plus an offset table for copying to the latter. Different layouts
meant the helper added plumbing rather than reducing it. It was tested, not
merely assumed to be unhelpful.

## Paired decode measurements

576 cases: 18 groups x eight uint8 128-cubed chunks x four presets. Release GCC
13, native C API, alternating before/after calls, reversing order each repetition.
Each measured frame is verified against original bytes. Eight repetitions per
case; wrapper repeat has twelve and uses CPU8 instead of CPU64.
Rates are total raw bytes / summed mean native call time (MiB/s).

| Trial | fast before -> after | medium | slow | slower |
|---|---:|---:|---:|---:|
| Wrapper | 672.749 -> 670.473 | 473.243 -> 469.220 | 490.252 -> 485.726 | 475.618 -> 471.193 |
| Wrapper repeat | 670.065 -> 666.720 | 482.055 -> 477.745 | 502.076 -> 497.807 | 486.065 -> 481.957 |
| Dummy initializer (vs wrapper) | 663.834 -> 668.164 | 475.861 -> 480.319 | 496.215 -> 500.624 | 480.376 -> 484.344 |
| Scalar metadata (vs initial header trial) | 671.872 -> 668.627 | 479.991 -> 478.786 | 501.702 -> 501.390 | 485.176 -> 485.362 |
| Initial header handoff (vs baseline) | 669.046 -> 667.293 | 482.719 -> 482.322 | 504.040 -> 503.079 | 487.346 -> 487.310 |

Dummy and scalar trials were sequential, with their specific comparison binaries
shown above; their gains/losses must not be added to the baseline comparison.
The dummy speed change alone does not establish that serialization dominated:
code layout also changes and this was only one measurement.

Wrapper slowdown reproduced on medium/slow/slower: approximately 0.85–0.93%.
The initial header trial is within 0.27% of baseline for every preset.
All four trials passed Release CTest 7/7 and full-corpus lossless decode.
Initial header handoff also produced 576 byte-identical encoded frames.

Contention was recorded by `benchmark/monitor_contention.py`. Mean whole-machine
utilization for wrapper/repeat/dummy/metadata/header: 1.775/1.638/1.923/2.462/2.608%.
Corresponding SMT sibling utilization: 0.313/0.854/0.414/0.838/0.148%.
Some compile/test work overlapped the scalar trial and encoder measurement;
timings are not claimed to be perfectly isolated. No failed trial is hidden.

Results: `encoder_speed/decoder_pass/`, including `*_contention.jsonl`.
Each contention file's first record contains the exact command. Frozen wrapper,
initial header and scalar binaries are `builds/decoder-pass-{wrapper,header,metadata}`.
The header trial uses dense little-endian 16-bit frequencies internally, while
the existing compressed header still carries the same pooled numerical arrays.
An additional refinement validated/built decode tables directly from dense
values, without a temporary frequency array. It measured fast +0.597%, medium
-0.797%, slow -0.490%, slower -0.513% against baseline; it was conservatively
reverted. Its completed decode measurement is `header_final.jsonl`. The
associated `final_encode.jsonl` was interrupted after rejecting the candidate;
it is incomplete and is not used as an encoder result.

## Retained result

Production lines: **7,505 -> 7,500**. Decoder-only lines remain **1,446**
(1,255 core + 191 frame/container). Shared header code is 750 -> 743;
other shared core is 1,337 -> 1,339. This is a simplification of the internal
data path, not a substantial decoder source reduction.

The Release extension text grows 1,036,370 -> 1,041,610 bytes; data and BSS
remain 5,220 and 32,992. Source reduction is not a binary-size improvement.
The final retained library is byte-identical to the frozen initial header
trial binary (SHA256 `a5defd30abdeb0b870d694495946e42151afd338c4b3932f0b8fabab0c1f8978`).

Header decode repeat (12 repetitions, total-time rates):

| Preset | Before MiB/s | After MiB/s | Change | Before min/median/max | After min/median/max |
|---|---:|---:|---:|---|---|
| fast | 674.913 | 678.294 | +0.501% | 671.677 / 675.193 / 676.685 | 674.173 / 678.360 / 680.529 |
| medium | 475.144 | 477.278 | +0.449% | 471.067 / 475.584 / 476.486 | 473.798 / 477.651 / 478.394 |
| slow | 494.189 | 499.407 | +1.056% | 492.785 / 494.312 / 494.990 | 498.165 / 499.419 / 500.467 |
| slower | 478.319 | 482.575 | +0.890% | 477.864 / 478.193 / 478.927 | 480.967 / 482.802 / 483.393 |

Both retained-version decode measurements meet the 0.5% regression criterion,
but their differing signs mean a small speedup is not established reliably.
Native/portable/ASan+UBSan suites pass 7/7 each; Python 139 tests and the C++
adapter smoke pass. Header tests include all truncations, mutations, optional
transmitted-table mode, and explicit rejection of incorrect frequency sums.
All 576 retained-version encodes match the baseline bytes; format and compressed
sizes are unchanged. ARM/macOS runtime performance was not measured.

### Encoder check

Full 576-case paired encoder run, two repetitions, CPU8:

| Preset | Before MiB/s | After MiB/s | Change | Before min/median/max | After min/median/max |
|---|---:|---:|---:|---|---|
| fast | 113.3261 | 114.6662 | +1.1826% | 111.8475 / 113.3459 / 114.8442 | 114.4252 / 114.6667 / 114.9083 |
| medium | 20.8286 | 20.8349 | +0.0302% | 20.8167 / 20.8286 / 20.8406 | 20.8218 / 20.8349 / 20.8480 |
| slow | 7.3174 | 7.2755 | -0.5725% | 7.3123 / 7.3174 / 7.3224 | 7.2713 / 7.2755 / 7.2796 |
| slower | 6.4683 | 6.4641 | -0.0640% | 6.4682 / 6.4683 / 6.4683 | 6.4628 / 6.4641 / 6.4654 |

The borderline slow regression was repeated over all 144 slow cases on CPU64,
again twice: **7.3271 -> 7.3001 MiB/s (-0.3681%)**. Before min/median/max:
7.3231 / 7.3271 / 7.3311; after: 7.2922 / 7.3001 / 7.3081.
Pooled by native time over both runs: **7.3222 -> 7.2878 (-0.4705%)**.
Kept within the agreed 0.5% tolerance, but this is a small encoder regression,
not a speed improvement for slow. The initial -0.5725% result remains reported.

Mean machine / measured core / sibling utilization: full encode
1.7908 / 99.9812 / 0.4089%; slow repeat 1.1395 / 99.7262 / 0.3409%.
The repeat ran after builds and other benchmarks finished.
The benchmark now accepts `--preset slow` for this targeted confirmation;
without that option it still runs every preset. This is test tooling only.
All required trials and validation are complete; no process remains running.

## Reproduction

From `local/codec_sources/delta3d-v020`:

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
/home/hendrik/.venv_las/bin/python benchmark/source_sizes.py
```

Paired timing uses `benchmark/check_decode_cleanup.py` under the contention
monitor. `--before` is `builds/decoder-pass-baseline`; `--after` is the trial
build; `--frames` is `encoder_speed/final_cleanup_20261103/frames`; all these
paths are absolute under the data root. Use `--repetitions 8 --cpu 64` for
decode, or `--encode --repetitions 2 --cpu 8` for encode. Python only orchestrates;
the codec path is native C. Full commands are preserved in the result logs.
