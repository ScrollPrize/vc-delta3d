# Why Zstd wins on CT L0 selector maps

2026-11-03. Diagnostic sequence extraction and cross-coding, not a codec change.
32 saved CT L0 frames:8chunks ×fast/slow ×2³/4³; inspect raw and packed maps.
Uniform maps use the existing2-byte special case and are not passed through an
artificial LZ/Zstd comparison. Fast4 has only one nonuniform map; fast2 has3,
slow4 has4, slow2 has7. This matters when interpreting averaged results.

Native `ZSTD_generateSequences` at level3 exposes actual literal lengths,
match lengths, distances and repeat-distance indicators. Sequences are verified
against source bytes. Block delimiters are merged using Zstd's own API.
Cross tests hold the matches fixed while replacing their encoder:

1. Our parser → our current LZ serialization.
2. The same matches → `ZSTD_compressSequences`, validation enabled.
3. Zstd's matches → our current LZ serialization (where representable).
4. Zstd's matches → Zstd sequence API, versus normal level3 compression.

All emitted streams are decoded and compared exactly. Sequence-API replay can
differ slightly from regular Zstd (e.g.391 vs395 B); it is not assumed identical.
Our min-match4/window64KiB cannot represent some raw-map Zstd sequences; those
cross tests are explicitly unsupported, never silently modified. Packed-map
cross tests all work.

## Concrete nonuniform example: input hash `ab40c1a9…`, fast

Complete packed selector payload bytes, including outer tag/width:

| Codec/parser combination | 2³ | 4³ |
|---|---:|---:|
| Our matches + our coder | 1009 | 469 |
| **Our unchanged matches + Zstd coder** | **577** | **249** |
| Zstd matches + our coder | 1148 | 301 |
| Zstd matches + sequence-API coder | 391 | 182 |
| Normal Zstd level3 | 395 | 183 |

Sequence statistics:

| Metric | Our2³ | Zstd2³ | Our4³ | Zstd4³ |
|---|---:|---:|---:|---:|
| Input packed bytes | 65536 | 65536 | 8192 | 8192 |
| Literal bytes remaining | 5 | 8 | 5 | 7 |
| Matches | 369 | 372 | 166 | 79 |
| Matches length≤8 | 143 | 48 | 92 | 18 |
| Longest match | 976 | 7164 | 240 | 968 |
| Largest distance | 9005 | 49968 | 1232 | 7160 |
| Repeat-distance codes | not implemented | 341 | not implemented | 40 |

Thus finding more literal coverage is not the answer: both already copy nearly
everything. At2³ Zstd has almost the same match count but far more reusable
distances:1008 occurs157times,16 occurs112times,208 occurs47times. Its repeat
codes reuse recently seen distances instead of transmitting their integer
representations each time. At4³ Zstd additionally halves the match count.

The counters are measured from Zstd's sequence `rep` field; its meaning is
documented in the installed `/usr/include/zstd.h` ZSTD_Sequence declaration.
No inference about repeat coding is based solely on the compressed size.

## Our actual byte breakdown on that example

| Part | 2³ | 4³ |
|---|---:|---:|
| Outer mode/packing width | 2 | 2 |
| Match/literal counts | 3 | 3 |
| Literal-run lengths, encoded | 75 | 74 |
| Match lengths, encoded | 520 | 187 |
| Distances, encoded | 403 | 197 |
| Literal data, encoded | 6 | 6 |
| **Total** | **1009** | **469** |
| Of that: rANS probability tables | 193 | 6 |
| Of that: rANS states | 128 | 64 |

Last two rows are subsets, not additional bytes. State size is the current
shared kernel's8×uint64 states per entropy stream, verified in `src/rans.c`.
Table bytes are parsed with the actual compact-table reader. Tiny streams can
select raw/constant rather than rANS, so not every part pays that state cost.

Conclusions: not just framing overhead, not just a better match finder. The
copy-description coder is a major contributor; the parser must also favor
cheap sequences rather than only consuming as many source bytes as possible.
Switching to Zstd matches alone can even increase size with our byte-varint
coding (1148>1009 at2³). This is a measured interaction, not additive savings.

## Mean packed payload over the original8 CT L0 chunks

| Preset | Edge | Our coder | Our matches/Zstd coder | Zstd matches/our coder | Normal Zstd |
|---|---:|---:|---:|---:|---:|
| fast | 2 | 138.75 | 83.75 | 155.38 | 61.00 |
| fast | 4 | 60.38 | 32.88 | 39.38 | 24.62 |
| slow | 2 | 1723.38 | 1533.88 | 2733.88 | 1304.00 |
| slow | 4 | 335.88 | 196.38 | 373.00 | 178.62 |

Measured Zstd repeat-code usage on nonuniform packed maps:fast2 350/384matches,
fast4 40/79, slow2 7216/8105, slow4 730/988. Large benefit is not exclusively
a fast-preset phenomenon, although parser/coder interactions differ.

## Sensible next targets (not implemented here)

- A small recent-distance cache and repeat-distance symbols.
- Dedicated integer magnitude symbols + exact extra bits for lengths/distances,
  rather than treating varint bytes as independent entropy symbols.
- Smaller/shared entropy state overhead for these tiny control streams.
- Then test a few match candidates and choose using estimated coded cost.

These target the observed costs. They do not require cloning all of Zstd, and
are not proof that a small implementation will match its compression.

## Reproduction / validation

Code: `benchmark/selector_zstd_audit.c`, `benchmark/audit_selector_zstd.py`.
Uses native installed Zstd sequence API only for diagnostics, plus the existing
selector parser and LZ/rANS helpers. No Python compression, no dependencies
installed, no production/default changes or performance claims.

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/selector_zstd_audit.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/selector_zstd_audit_v2
python3 benchmark/audit_selector_zstd.py \
  --results /data/hendrik/compression_experiments/villa4/encoder_speed/selector_lz_20261103/results.jsonl \
  --binary /data/hendrik/compression_experiments/villa4/builds/selector_zstd_audit_v2 \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/selector_zstd_audit_v2_20261103
```

32frames audited,64representation records;30nonuniform streams cross-tested.
Original sizes must match the previous run, including uniform wrappers. All
supported crosses verified lossless. Full audit passes ASan/UBSan; fixed an
initial diagnostic-only leak on unsupported-cross early return by prechecking
eligibility before allocation. Saved sequence lists, table/state counters and
per-case results are in `results.jsonl` beside `report.md` in the output directory.
The Zstd experimental sequence API may differ across installed library versions;
it is not introduced into the production codec.
