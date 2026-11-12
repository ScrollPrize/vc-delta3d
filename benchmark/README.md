# VC-Delta3D benchmark harness

Benchmarks the codec against real chunks pulled from a zarr v2 store on S3
(anonymous access). Chunks and the benchmark build are cached under
`benchmark/.cache/` (tagged with `CACHEDIR.TAG`), so only the first run for a
store touches the network.

```sh
python3 run.py s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr \
    --chunks 50 --levels 6 --preset fast
```

Per level it samples `--chunks` full, nonzero chunks (≥5% nonzero voxels by
default, mid-volume first), verifies exact roundtrips, then reports the
compression ratio and best-of-`--reps` single-threaded encode/decode
throughput.

Requires `numpy` and `s3fs` (plus `numcodecs` if the store is compressed),
`cmake` and `g++` to build the library and C++ harness. Supports 3D uint8/uint16
arrays, either a bare zarr array or a `0..N` multiscale pyramid group.

Keep `--seed` fixed (default) so reruns reuse the cached chunk sample.

## Encoder ladder

`ladder.c` measures size, encode time and decode time of encoder settings and
presets on 128³ uint8 chunks, relative to each other in one run. It links the
static library to reach the internal tuning entry point:

```sh
cc -O2 -Iinclude benchmark/ladder.c build/libvc_delta3d.a -lm -o ladder
./ladder 7 1,40,128,0,1,0 2,56,128,0,1,0 delta slower -- chunk1.raw chunk2.raw
```

A setting is `reach,taps,stride,intercept,models,rounds` (`vc_ng_tuning` in
`src/ng/ng_frame.h`); the first two above are `faster` and `fast`.

## Header table generator

`header_static.c` generated the fixed header frequency tables in
`src/ng/header_tables.h` from a list of encoded frames. It includes
`src/ng/header.c` directly and is not part of the library build.

The one-off C-port validation scripts were removed after the port was
merged; see commit 93469eb for them and `C_PORT_RESULTS.md`.
