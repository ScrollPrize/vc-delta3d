# VC-Delta3D benchmark harness

Benchmarks the codec against real chunks pulled from a zarr v2 store on S3
(anonymous access). Chunks are cached under `benchmark/.cache/`, so only
the first run for a store touches the network.

```sh
python3 run.py s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr \
    --chunks 50 --levels 6
```

Per level it samples `--chunks` full, nonzero chunks (≥5% nonzero voxels by
default, mid-volume first), verifies exact roundtrips, then reports the
compression ratio and best-of-`--reps` single-threaded encode/decode
throughput.

Requires `numpy` and `s3fs` (plus `numcodecs` if the store is compressed),
and `g++` to build the C++ harness on first run. Supports 3D uint8/uint16
arrays, either a bare zarr array or a `0..N` multiscale pyramid group.

Keep `--seed` fixed (default) so reruns reuse the cached chunk sample.

## C-port regression and timing checks

These drivers only orchestrate native codec calls. They never implement a
predictor, fitting or entropy coder in Python. Keep frozen frames and generated
reports in an external experiment directory, not the source checkout.

```sh
python benchmark/verify_port.py --build <port-build> --frames <frozen-results.json>
python benchmark/verify_legacy.py --baseline <original-build> --build <port-build>
python benchmark/compare_port.py --baseline <original-build> --build <port-build> \
  --frames <frozen-results.json> --output <experiment-dir> --cpu 97 --repetitions 3
```

The comparison decodes the same frozen frames, verifies original raw SHA256 and
byte-identical re-encoding, then measures each group/preset on one CPU. Timings
exclude file reads and verification. Separate processes prevent baseline/port
symbol interposition. JSON records every repetition and load average; the report
uses MiB/s and includes combined min/median/max. `--cpu` is Linux-only; omit it on
macOS. Use `--report-only` with the same output directory to regenerate the report
without any codec work. Do not rebuild or overwrite the original build.
