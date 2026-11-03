# Cleanup slowdown attribution checks — 2026-11-03

Production checkpoint: `e0c855e`; comparison checkpoint: `22e2583`.
No diagnostic source changes were retained. The user accepted keeping the
cleanup after the variable repeat results, with **0.5%** as the future
acceptable regression threshold. This is not proof that every measurement is
within 0.5%, nor that contention explains all differences.

## Method and results

Release native C API, CPU64, identical saved 128³ frames, 18 groups × 8 chunks ×
4 presets = 576 cases. Alternate calls between libraries, verify every decode,
and aggregate per-case median times. Isolation runs use eight repetitions;
reverse-order and identical-binary controls use twelve. Python only orchestrates
native codec calls. All 576 cases passed for each completed experiment.

Each reversion below changes only the named part of `e0c855e`. Positive numbers
mean restoring the older implementation was faster than the cleanup.

| Diagnostic reversion | Fast | Medium | Slow | Slower |
|---|---:|---:|---:|---:|
| Old activity classifier | −0.01% | −0.24% | −0.07% | −0.08% |
| Scalar override argument | +0.15% | −0.14% | −0.30% | −0.32% |
| Lookup-table guard | −0.36% | −0.62% | −0.61% | −0.58% |
| Payload guard | +0.04% | +0.13% | +0.26% | +0.27% |
| Old state layout (padding only) | +0.24% | +0.37% | +0.53% | +0.41% |
| Expanded model pointer palette | +0.34% | +0.33% | +0.16% | +0.09% |
| 32-bit local header frequencies | +0.05% | −0.05% | −0.03% | +0.05% |

These are individual runs, not confidence intervals; small apparent benefits
were not sufficient grounds to retain extra code. The 32-bit frequency test
would add two lines, state padding and the scalar argument save no physical
lines, and the two restored guards add two and zero lines respectively.

The unchanged saved before/after binaries gave inconsistent estimates:

| Comparison | Fast | Medium | Slow | Slower |
|---|---:|---:|---:|---:|
| Balanced repeat: cleanup vs old | −0.00% | −0.31% | −0.03% | −0.06% |
| Reversed load order: old vs cleanup | +0.84% | +2.10% | +2.52% | +2.43% |
| Identical cleanup binaries in separate mappings | +0.04% | +0.17% | +0.14% | +0.12% |
| Monitored repeat: cleanup vs old | −0.55% | −1.29% | −1.04% | −1.18% |

The reversed row reports the old binary's advantage, not the cleanup's.
The remaining difference was **not attributed to a specific source edit**.
The earlier report's implication of a known 121-lines-for-1%-speed trade-off
was unjustified. The previously isolated 5–9% phase-guard regression remains
separate: those guards were already restored in the committed cleanup.

## Contention monitoring

Overall load alone was insufficient. Added `benchmark/monitor_contention.py`
to record `/proc/stat` per-core counters, the benchmark's SMT sibling, CPU
frequency, CPU/memory/I/O pressure, process faults and context switches at 0.5s
intervals. The monitor runs away from the measured physical core. The paired
benchmark now defaults to an even repetition count and records monotonic case
intervals for correlation. No root privileges or installed dependencies needed.

During the monitored baseline run, total CPU utilization averaged 2.74% of
192 logical CPUs and peaked at 10.36% (about 20 busy logical CPUs). CPU160,
the sibling of benchmark CPU64, averaged 0.20% and peaked at 6%. Sampled CPU64
frequency ranged from 3.65 to 4.85 GHz. Pressure averages were zero, but those
facts do not establish an uncontended or fixed-frequency timing environment.
Historical runs lack these logs, so their contention cannot be reconstructed.

Results and raw monitoring logs:
`/data/hendrik/compression_experiments/villa4/encoder_speed/cleanup_cause_20261103/`.
Saved production binary:
`/data/hendrik/compression_experiments/villa4/builds/cleanup-e0c855e/`.

Example command (from codec checkout; supply the existing benchmark arguments):

```sh
/home/hendrik/.venv_las/bin/python benchmark/monitor_contention.py \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/run-contention.jsonl \
  --cpu 64 -- /home/hendrik/.venv_las/bin/python benchmark/check_decode_cleanup.py \
  --before /data/hendrik/compression_experiments/villa4/builds/pre_menu_cleanup \
  --after /data/hendrik/compression_experiments/villa4/builds/cleanup-e0c855e \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/final_cleanup_20261103/frames \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/run-paired.jsonl \
  --repetitions 12
```

The contention monitor is Linux-only measurement tooling, not a codec dependency.
