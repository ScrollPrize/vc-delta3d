# Final retained-path cleanup — 2026-11-03

Follow-up: [attribution checks](cleanup_attribution.md) found variable repeat
timings and did not establish a specific cause for the remaining small loss.
Treat the performance table below as one measurement, not an isolated trade-off.

Checkpoint before this pass: `22e2583` on `refactor/c-only-presets`.
This completes the remaining cleanup items from `option_cleanup.md` without
changing frame bytes, compression choices, decoded values, or public APIs.

## Completed work

- Menu translation now passes one pointer per live fitted model, rather than
  twelve identical pointers per model. It uses the shared menu descriptor,
  records live choices during validation, constructs models directly, and
  remaps selectors in place. A bank map is allocated only for multiple banks.
- Removed the unused model prediction buffer, unreachable activity-bias table
  path, obsolete residual-only decoder branch, unused scalar compact recurrence,
  unused AVX-512 full-column recurrence, and unused NEON forward transpose.
  The live portable full-column recurrence and NEON inverse transpose remain.
- Removed the non-fused class-lookup alternative: the sole caller always uses
  the fused implementation. Removed the unreachable paired-classification
  fallback and its extra byte/word variants. Retained homogeneous, compact, full-column
  and fused-base paths; they serve different real model/ISA combinations.
- Removed the always-false force-scalar preparation argument; hardware dispatch
  and the portable fallback remain.
- Removed unused encoder feature-weight/export variants from symbol evaluation,
  raw-residual activity mode, unused fit-session SIMD override, redundant wrappers,
  and rectangular encoder block-size options not used by the four presets.
  Generic frame serialization and decoding still support all selector sizes.
- Replaced the encoder's nested thread-local selected-only override with an
  explicit call argument. The decoder already uses a per-call context.
- Header decoding now calls the shared rANS payload decoder with decoded or
  fixed frequencies directly. It no longer allocates/copies buffers to fabricate
  a legacy table prefix. The legacy reader calls the same payload implementation.

The earlier steps already unified bounded parsing and shared recurrence math.
The compact numeric selector representation remains inside the SIMD kernels:
replacing it with per-voxel descriptor indirection is not required to simplify
menu construction and would add work to the hot loop. Geometry-specialized
kernels are instantiations of shared source, not copied implementations.

Historical experiment reports remain documentation, not compiled alternatives.
No experimental runtime toggles, old new-format compatibility readers, streaming
search, LPC or tree coder were restored. Full current-format model/menu/bank
flexibility and legacy Delta3D compatibility remain required functionality.

## Source counts

Physical lines including comments, blanks and SIMD include fragments, excluding
tests/docs/benchmarks. Reproduce with `benchmark/source_sizes.py`.

| Category | Before this pass | After |
|---|---:|---:|
| Encoder core | 2,688 | 2,658 |
| Decoder core | 1,320 | 1,267 |
| Shared core | 1,419 | 1,392 |
| Header/container encoder | 207 | 207 |
| Header/container decoder | 200 | 196 |
| Header/selector/container shared | 751 | 744 |
| Legacy Delta3D | 349 | 349 |
| C API | 174 | 174 |
| C++ adapters | 378 | 378 |
| Python bindings/API | 394 | 394 |
| **Total** | **7,880** | **7,759** |

82 production files. Across the two option-cleanup passes: 8,452 → 7,759 total
lines (−8.2%), and 1,675 → 1,267 decoder-core lines (−24.4%).

## Performance-sensitive structure: measured, not removed blindly

An initial full cleanup regressed decode by 5–9%. Hardware `perf` is not
installed here; isolate with alternating C-API timings on identical cached
frames, pinned CPU64, seven repetitions. Both builds run in the same process,
with alternating order and exact byte checks; nanobind modules are not
initialized twice. Data and frame exports are under `final_cleanup_20261103`.

Restoring replicated model-pointer palettes did not help. Restoring the old
header-copying implementation made it worse. Restoring the unused model buffer
or frame allocation pattern also failed to recover CT-L0 throughput. Restoring
the old reconstruction loop did recover most of it. The minimal fix is retaining
the four entropy phase guards in that loop, without the unused residual-only
fallback. The public/native entrypoint already requires entropy: these guards
preserve favorable compiler-generated loop structure, not an optional format.
Do not remove them merely because they look logically redundant.

Final paired **C API** decode MiB/s (total raw bytes / summed native times):

| Preset | Before mean | After mean | Before min / median / max | After min / median / max |
|---|---:|---:|---|---|
| fast | 666.8 | 662.2 | 665.4 / 667.0 / 667.9 | 661.3 / 662.3 / 662.9 |
| medium | 481.1 | 476.0 | 479.9 / 481.3 / 481.9 | 475.2 / 476.1 / 476.9 |
| slow | 501.3 | 494.6 | 500.2 / 501.4 / 502.4 | 493.6 / 494.8 / 495.3 |
| slower | 485.4 | 479.2 | 484.5 / 485.2 / 486.2 | 478.5 / 479.1 / 479.8 |

Final throughput is within 0.7–1.4% of the checkpoint. These warmed, preallocated
C API timings are not directly comparable to the Python adapter timing below.
The 5–9% regression was rejected; the remaining small trade-off is retained.
Isolation results are `paired_initial.jsonl`, `paired_palette.jsonl`,
`paired_header_control.jsonl`, `paired_state_control.jsonl`,
`paired_frame_control.jsonl`, `paired_core_control.jsonl`, `paired_guards.jsonl`
and `paired_final.jsonl`. State/frame/core/guard controls use only CT-L0;
initial/palette/header/final measurements cover all 18 groups.
The final argument cleanup is measured in `paired_final_no_scalar.jsonl`,
which supplies the table above; the earlier `paired_final.jsonl` is retained
as the preceding checkpoint. Correctness is checked separately over all 576
latest-build cases in `parity_final.jsonl`.

```sh
/home/hendrik/.venv_las/bin/python benchmark/check_decode_cleanup.py --before /data/hendrik/compression_experiments/villa4/builds/pre_menu_cleanup --after /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --frames /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103/frames --output /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103/paired_final.jsonl --export
/home/hendrik/.venv_las/bin/python benchmark/check_decode_cleanup.py --before /data/hendrik/compression_experiments/villa4/builds/pre_menu_cleanup --after /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --frames /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103/frames --output /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103/paired_final.jsonl
```

## Verification

- Release, ASan/UBSan and portable NG reconstruction CTest suites: 7/7 each.
- 684 generic-format cases in each build: block edges 2–32, independent models,
  model-specific feature weights, entropy banks, classes, rectangular geometry,
  automatic ISA and AVX2 ceiling.
- Python tests: 136/136; C++ adapter smoke passed.
- Added direct rANS payload parity, truncation and invalid-frequency tests.
- Intermediate menu cleanup: all 576 corpus frames byte-identical.
- Final latest-build parity: all 576 encoded frames byte-identical and lossless.

The sequential Python-adapter timing run (three repetitions, before the final
unused preparation-argument removal) gave the following combined MiB/s. Encoding
was unaffected by that final decoder-only edit; use the paired C API table above
for final decoder measurements.

| Preset | Encode before | Encode after | Decode before | Decode after |
|---|---:|---:|---:|---:|
| fast | 121.1 | 122.4 | 623.6 | 621.7 |
| medium | 22.0 | 22.1 | 442.4 | 441.4 |
| slow | 7.6 | 7.6 | 452.3 | 449.5 |
| slower | 6.7 | 6.7 | 439.6 | 438.6 |

Per-channel timing and min/median/max statistics are in `timing_summary.json`,
generated by `benchmark/report_cleanup.py`. These sequential measurements and
the alternating native test use different allocation/timing protocols.

No ARM/macOS runtime measurements were made. Portable reconstruction is tested
on this x86 host; architecture guards and live NEON code remain intact.

## Reproduction

Run from the codec checkout. Native builds and test data stay under `/data`.
The two timing commands run sequentially, pinned to CPU64, three repetitions
per case, 18 groups × 8 chunks × 4 presets, Release. Python only orchestrates
native calls, verifies raw output and compares complete compressed SHA256s.

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j 8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
/data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/vc_delta3d_modern_smoke
/home/hendrik/.venv_las/bin/python benchmark/check_core_cleanup.py --build /data/hendrik/compression_experiments/villa4/builds/pre_menu_cleanup --output /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103/speed_before.jsonl --workers 1 --repetitions 3 --cpu 64
/home/hendrik/.venv_las/bin/python benchmark/check_core_cleanup.py --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103/speed_after.jsonl --compare /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103/speed_before.jsonl --workers 1 --repetitions 3 --cpu 64
/home/hendrik/.venv_las/bin/python benchmark/report_cleanup.py /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103
```

Sanitizer and portable builds use the commands in `option_cleanup.md`.
