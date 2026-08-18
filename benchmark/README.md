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
