# Fixed header probability tables — 2026-11-03

Experiment only; production header format and selection are unchanged.
Native C reuses the actual header schema, transforms and shared eight-lane rANS
kernel. A small internal explicit-frequency entry point shares the existing
encoder body; the ordinary path does not acquire extra histogram/validation scans.

Dataset: same 576 frozen frames, 18 channel/scale groups ×8 source chunks ×4
presets, block size4. Two-fold cross-validation with four training/four test
chunks per group. All presets of a source chunk stay in the same fold.
Also tested leave-one-channel/scale-group-out over all18groups. This latter
test can retain other scales of the same physical channel; it is not a new-volume test.

Each group has two distributions, one for plain log labels and one for the
existing wrapping-delta transform. Add-one smoothing covers unseen symbols.
All counts use the existing 4096-frequency normalization. Unseen predictor
groups fall back to global training statistics. Predictor group means the
most frequently selected predictor in a chunk, not arbitrary selector ID.

Sizes use actual rANS encodes, including64state bytes, renormalization bytes,
length varints, unchanged schema/bypass data. Transmitted512-byte tables are
omitted from fixed-table payloads. Global transform selection fits the existing
one-byte mode; grouped variants charge one extra table-ID byte per coded header.
Table constants reside in decoder code, not per chunk: global2tables=1024bytes
at uint16, per-preset8tables=4096bytes, predictor26tables=13312bytes,
combined104tables=53248bytes before deduplicating unused groups.
Raw labels remain available. Hybrid additionally retains dynamic-table encoding;
that selection fits one mode byte and guarantees no size regression by choosing
the smallest actual representation. Selector payload is unchanged and excluded.

| Distribution grouping | Current | Fixed/raw | Fixed/raw + dynamic fallback |
|---|---:|---:|---:|
| Global |2049.510|1916.863|1914.958|
| Preset |2049.510|1912.677|1910.580|
| Dominant predictor |2049.510|1918.219|1916.627|
| Preset + dominant predictor |2049.510|1913.524|1911.837|

All values average bytes/header, cross-validated. Global fixed/raw saves6.47%;
per-preset adds only4.19bytes saving. Global fixed/raw increases34/576individual
headers; hybrid avoids these. Entire held-out channel/scale results:
global1917.13B, preset1912.79B, predictor1921.89B, combined1917.17B (fixed/raw).
Thus the savings transfer, but predictor-specialized tables are not justified.
Recommend the small global pair, optionally with dynamic fallback, pending
approval to integrate. No claim of encode/decode speed improvement yet.

## Reproduce

From the delta3d checkout (existing Release build):

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j8
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_check.c /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a -lzstd -lm -pthread -o /data/hendrik/compression_experiments/villa4/builds/header-export
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_static.c /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a -lm -pthread -o /data/hendrik/compression_experiments/villa4/builds/header-static
python3 benchmark/check_header_static.py
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
```

Results, inputs, exported headers, exact per-case sizes and full channel table:
`/data/hendrik/compression_experiments/villa4/encoder_speed/header_static_20261103/`.
Every fixed-table trial is decoded and compared byte-for-byte; frequency tables
are supplied to the existing decoder as constants would be, not transmitted.
Python only orchestrates and aggregates; encoding/decoding is native C.
