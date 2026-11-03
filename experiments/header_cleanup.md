# Compact header integration — 2026-11-03

## Scope and retained flexibility

Target: standalone Delta3D C library (`local/codec_sources/delta3d-v020`),
branch `refactor/c-only-presets`. Native C codec, CMake Release and sanitizer
builds; existing C++/Python ownership adapters. No dependency installs.
Keep lossless voxel values, old Delta3D compatibility/fallback, four presets,
portable kernels, fixed block sizes and indexed channel/spatial access.

One new format, no compatibility with experimental NCF1/NDB2 variants. Menu
entries, fitted model reuse, entropy-bank sharing, block size and fitting
configuration remain independent of encoder presets. See `HEADER_FORMAT.md`.

Changes:

- `header.c`: single shared schema walk, derived array shapes, omitted zero
  suffixes, exact unsigned log ranges, optional within-record wrapping word
  delta of label pairs, raw/rANS pooled numerical stream, control varints.
- `ng_frame.c`: one selector codec (including constants), eight-byte frame
  prefix, no old selectors/Zstd/raw-length fields. Entropy coefficients are
  unused and no longer stored; thresholds and tables remain.
- Removed experimental model-v4 nonlinear-context encoding/decoding.
- `ng_rect.c` now emits the same NCF1 as cubic leaves. One NGC1 leaf tag.
  Fixed rectangular plane stride in shared directional reconstruction; no
  duplicate rectangular decoder or old frame reader retained.
- Screening uses the same compact-header codec for model costs, not Zstd.
  This intentionally changes estimated metadata costs, not fitted arithmetic.
- Zstd removed from library links, CMake and wheel installation steps. The
  standalone test importer alone uses Zstd to read archived fixtures/control.
- Obsolete selector-only frame-rewrap harness replaced by the full header
  harness. Historical reports/data remain; no external datasets removed.

## Test first: size comparisons

576 frozen 128³ frames: 18 groups ×8 source chunks × fast/slow ×2³/4³ blocks.
Same stored choices, models, residual bytes and selector payload for comparisons.
The matched Zstd control compresses the same canonical model/entropy header.
It excludes selectors on **both** sides. These numbers must not be compared
directly with historical 6,061.875-byte totals that included selector data.

|Native candidate|Mean compact header B|Matched Zstd B|Status|
|---|---:|---:|---|
|Initial generic schema/log pool|1686.165|1664.639|Not accepted|
|Add previous/first word references|1687.165|1664.639|Never selected; removed|
|Remove duplicate entropy coefficients, with reference tag|1671.128|1665.292|Intermediate|
|Then trial control entropy coding|1672.128|1665.292|Not retained|
|Derive unused entropy fields too|1666.304|1665.292|Intermediate|
|Final: plain self-delimiting controls|**1664.304**|1665.292|Retained|

Thus effectively parity with Zstd (0.988 B/header smaller, not a major win).
Including the 8-byte versus12-byte prefix:1672.304 versus1677.292 B.
Individual groups differ: CT-L0 +46.500 B, CT-L1 +1.438 B, width-L4 +96.750 B,
ny-L4 −201.781 B. The simplification is NOT claimed to win every header.

Including the separately improved selector codec, total old original frame
mean680882.014 B becomes679662.594 B (−1219.420 B). This larger saving is mostly
selector storage, **not** the header replacement. Complete group breakdown:

`/data/hendrik/compression_experiments/villa4/encoder_speed/header_clean_final_verified_20261103/report.md`

## Verification

- 576 exact canonical-header round trips and exact retained selector sizes.
- All576 new-frame decoded voxel hashes equal the saved original decoder;
  original encoder source SHA256 is additionally checked in fresh-encode test.
- Same576 cases under ASan/UBSan, identical sizes/hashes, no findings.
- CTest6/6 in Release and ASan/UBSan: header truncation/mutation tests, selector
  tests, all four presets, explicit rectangular shapes, indexed channels,
  legacy Delta3D, rANS and filters. C++ smoke also passes including AVX2 cap.
- Python/API suite124/124, including large/noncubic leaves and range decoding.
- Fresh144 chunks ×4 presets ×3 deterministic encode/decode repetitions pass.
- Static-library undefined-symbol audit has no ZSTD references.

## Fresh encoder corpus

Full18-group corpus, eight128³ chunks/group, four presets. Compression includes
all bytes. Original column is the frozen release result, not a new baseline
timing. New includes both header/selector changes and revised screening costs.

|Preset|Original % raw|New % raw|
|---|---:|---:|
|fast|33.6504|33.6367|
|medium|32.3715|32.3407|
|slow|32.1316|32.1088|
|slower|32.0990|32.0768|

Full per-channel ratios and new timings:
`/data/hendrik/compression_experiments/villa4/encoder_speed/header_clean_presets_20261103/report.md`.
Small CT exceptions: fast CT-L0 22.1215→22.1226%; fast CT-L1
45.1923→45.1927%; slower CT-L1 44.8558→44.8566%. No broad regression hidden
behind the combined averages.

Release/O3, no fast math; eight independent workers, three repetitions each.
Load-average min/median/max2.69/4.27/5.40. These are per-operation throughput
summaries, not aggregate eight-worker throughput or a same-session speedup
claim against an old decoder. Combined repeat min/median/max MiB/s:

|Preset|Encode min/median/max|Decode min/median/max|
|---|---:|---:|
|fast|110.50 /110.87 /111.27|589.60 /590.96 /600.91|
|medium|20.68 /20.69 /20.69|432.52 /433.81 /436.25|
|slow|7.13 /7.14 /7.17|421.43 /425.53 /425.94|
|slower|6.25 /6.25 /6.26|411.59 /414.15 /414.67|

## Reproduce

From the Delta3D checkout; data/builds stay outside the checkout:

```sh
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only -j8
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only --output-on-failure
cmake --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-sanitize -j8
ulimit -s 32768
ctest --test-dir /data/hendrik/compression_experiments/villa4/builds/delta3d-c-sanitize --output-on-failure
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_check.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread -o /data/hendrik/compression_experiments/villa4/builds/header-clean
python3 benchmark/check_headers.py \
  --reference /data/hendrik/compression_experiments/villa4/encoder_speed/selector_unified_20261103/results.jsonl \
  --binary /data/hendrik/compression_experiments/villa4/builds/header-clean \
  --baseline /data/hendrik/compression_experiments/villa4/encoder_speed/header_clean_baseline_20261103/results.jsonl \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_clean_final_verified_20261103
/home/hendrik/.venv_las/bin/python benchmark/check_presets.py \
  --build /data/hendrik/compression_experiments/villa4/builds/delta3d-modern \
  --manifest /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_clean_presets_20261103 --export
/home/hendrik/.venv_las/bin/python benchmark/check_presets.py \
  --build /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only \
  --manifest /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_clean_presets_20261103
```

For the frozen baseline hash importer, the saved binary is `builds/header-baseline`.
Its old decoder is statically contained there, never linked into the new library.
ASan native benchmark adds `-O1 -g -fsanitize=address,undefined
-fno-omit-frame-pointer` and links `delta3d-c-sanitize/libvc_delta3d.a`.

No macOS/ARM64 run on this host; changes are portable C with existing SIMD
dispatch. Experimental frames from earlier revisions intentionally require
offline conversion or re-encoding. Old Delta3D wire format is unchanged.
