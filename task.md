# C-only codec reduction

## Contract

- Branch: `refactor/c-only-presets`, based on merged main `98bb019`.
- Preserve `feature/ng-presets` unchanged as the reference implementation.
- Keep fast, medium, slow, slower and their fixed-preset framing, including
  indexed spatial/channel payloads. Keep legacy Delta3D decoding and encoding
  fallback for 16-bit, irregular dimensions and quantization.
- Remove LPC, tooslow/tree coding, streaming/search experiments and unrelated
  optional encoders. Unsupported retired formats must fail explicitly.
- The codec implementation and primary API must be C. Only thin C++/Python
  adapters may use C++; retain the existing consumer-facing API where applicable.
- Preserve arithmetic, predictor choices, emitted frames and SIMD dispatch while
  porting. Measure encoding and decoding; do not exchange compression for speed.

## Sequence

### Strict model wrapping-delta comparison (2026-11-02)

- Implemented exactly between-only, both passes, first-within/rest-between,
  independent-within and direct controls. Byte and 16-bit modulo variants;
  no XOR, RLE, reference search or adaptive section coding on this path.
- All non-model bytes use the same plain rANS stream. Nine schemes ×576 frames
  plus135 synthetic edge tests pass exact reconstruction and ASan/UBSan.
- Corpus has 32 zero-model, 540 one-model and only4 two-model headers. On the
  latter, model bytes fall from878 to835 with between-byte deltas; adding
  within-model deltas is worse. Full-corpus model mean only193.14 →192.84 bytes.
- Report, full timings, limitations and commands:
  experiments/header_strict_model_delta.md. No production changes.

### Zstd-dependency removal follow-up (2026-11-02)

- Corrected objective: eliminate the dependency; do not reject replacements
  merely because Zstd remains slightly faster or smaller.
- Native structured XOR + delta-rANS cuts mean metadata from 6,624 to 6,195
  bytes; adding RLE-rANS reaches 6,181, versus Zstd 6,173. All 576 headers
  reconstruct exactly; all new variants pass the same corpus under sanitizers.
- Model-reference subtraction versus XOR is essentially tied. The extra model
  method search is not worth retaining. A minimal raw/rANS/delta-rANS menu
  gives 6,341 bytes at roughly half the header encode time of the broad menu.
- Production unchanged. Full data, comparisons and compatibility/dependency
  integration caveats: experiments/header_delta_rans.md.

### Header format experiment (2026-11-02)

- Tested ten native C encodings on all 576 frozen headers, leaving model choices
  and residual payloads unchanged. Exact round-trips also pass ASan/UBSan.
- Zstd remains best overall: 6,173 mean bytes vs raw 11,710. The closest custom
  structured-XOR format makes total files 0.066% larger and is slower. Pure
  structured fields without entropy coding are fast but make files 0.802% larger.
- No production changes adopted. Commands and full results are recorded in
  experiments/header_formats.md; external JSON timings remain under
  /data/hendrik/compression_experiments/villa4/encoder_speed/header_formats_20261102_v2.

### Completed acceptance (2026-11-02)

- All retained implementation is C11; only public C++/Python adapters use C++.
  Original feature branch is unchanged. The earlier unfinished milestones below
  are historical, not the current state.
- Passed 576 exact-frame checks on native and AVX2-only paths, 384 historical
  variable-block frames, 96 legacy/indexed parity cases, 124 Python tests,
  18 C++ adapter cases and four C suites (also ASan/UBSan).
- Full 18-group, three-repetition single-core measurements: combined encode
  MiB/s original → C is fast 65.4 → 113.3, medium 13.5 → 20.6,
  slow 4.4 → 7.2, slower 4.1 → 6.3. Combined decode is
  597.8 → 631.4, 458.9 → 474.5, 464.5 → 476.9, 445.7 → 457.1.
  All compressed bytes remain identical.
- C legacy rANS initially regressed because fortified memcpy inserted dynamic
  checks into four-byte stream writes. Assembly plus gprof identified this;
  byte-preserving unaligned stores and explicit lane unrolling restore encode
  speed. Legacy decode varies by about 2% between alternating measurements.
- Full results, profiling explanation, limitations and commands are in
  C_PORT_RESULTS.md. No ARM/macOS runtime claim; those remain CI checks.

### Cleanup and validation checkpoint (2026-11-02)

- Removed unused tree routing, experimental entropy/fitting controls, tap-swap,
  dense-tap experiments, standalone NG exports and obsolete residual adapters.
  Rectangular leaves directly reuse the legacy C filter. No C++ implementation
  or templates remain under `src/`. Native kernels now compile as C11.
- Public/docs describe only fast/medium/slow/slower and legacy Delta. Historical
  measurements remain explicitly labeled, not advertised as supported modes.
- Rechecked all 576 frozen frames and 96 legacy/indexed configurations; exact
  bytes and decoded hashes match. 121 Python cases and 18 C++ adapter cases pass.
  The latter used Villa's existing doctest compatibility header (no install).
- Standalone `cc` consumer passes, with ldd showing libc/libm/libzstd only.
  Four ASan/UBSan regression suites pass, including malformed/truncated frames.
  This host's unlimited stack causes ASan shadow mapping collisions; run the
  sanitizer tests with `ulimit -s 32768` (a per-process limit, no system change).
- Full original/port single-core 18-group timings are running via
  `benchmark/compare_port.py`; do not substitute the six-chunk preliminary
  numbers for the final result. Output:
  `/data/hendrik/compression_experiments/villa4/encoder_speed/c_only_port_20261102`.
- Linux x86-64 and forced AVX2 paths were exercised here. ARM/macOS/Windows
  execution remains a CI validation limitation; their guarded kernels remain.

### Checkpoint: all retained encoders in C (2026-11-02)

- Medium, slow, slower and rectangular NDB2 orchestration now execute in C.
  The static library links with the C compiler; C++ ownership adapters are
  header-only and Python calls through those thin adapters into the C API.
- Removed the obsolete C++ encoder/search implementation, unused experiment
  entrypoints, sparse-bank/context/coarse prototypes and tap-swap/entropy-fit
  experiments. Preserved the production fixed-geometry SIMD instantiations.
- All 576 frozen frames still decode and re-encode byte-identically after the
  C-only transition; all 96 legacy/indexed compatibility cases also passed.
  Four native regression executables pass after pruning unused entrypoints.
- Remaining: finish internal experimental-control cleanup, final Python and
  sanitizer/portability checks, performance measurements and public docs.
  Earlier checkpoints below describe historical intermediate states.

### Checkpoint: native C frames and Fast (2026-11-02)

- NCF1 writer/decoder, threshold DP/coordinate search, occupancy/screening and
  fixed predictor SIMD kernels are now C. Fast's complete encoder orchestration
  is C, with the original screening, fit, zero-block handling and DP64 choices.
- All 576 frozen corpus frames decode and re-encode byte-identically, including
  Fast using the new C path. Four native regression executables pass.
- Removed the retired tree leaf from C++/Python enums; it remains rejected by
  the C indexed parser. Original reference branch/build remain untouched.
- Remaining: medium/slow/slower orchestration, rectangular payload orchestration,
  removal of their obsolete experimental dependencies, C-only linking and final
  performance/portability validation. This is not yet a complete pure-C port.

1. Freeze baseline frames/results; identify the retained call graph.
2. Extract legacy filters/rANS into shared C, with original wrappers calling C.
3. Port retained preset search/framing orchestration to C, reusing C SIMD kernels.
4. Port indexed containers and expose the checked C API; thin consumer adapters.
5. Remove unreachable experimental code and retired formats/presets.
6. Test old-frame decoding, exact new-frame parity, malformed inputs, native C
   consumers, Python adapters, portability and all-channel speed/size comparisons.

Do not describe the intermediate mixed-language build as a finished C port.

## Baselines

- Original source: `feature/ng-presets` (`4533974`, tree identical to merge).
- Frozen 144-chunk / five-preset frames:
  `/data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json`.
  Four retained presets give 576 frames. Retired tooslow is not a compatibility target.
- Original timing/compression report: `BENCHMARKS.md`.
- Existing regression tests: scalar, indexed, fallback and Python tests.

## Progress

### Second port stage (in progress)

- Public `codec.h` and `indexed.h` now provide checked C entrypoints. C++
  codec/container APIs are thin ownership/error adapters. Actual legacy filtering,
  rANS, D3D1 scalar/channel framing, NGC1 indexing/gather/scatter and top-level
  dispatch now execute in C. Python still uses its existing thin C++ adapter.
- Removed LPC implementation and exposed LPC/tooslow preset options; added
  explicit rejection tests. Retired HGT3 is rejected by public dispatch.
- Removed four unused experiment translation units from the build. Slow/slower
  no longer pass through the tree capture/replay wrapper: its retained mode
  returned the exact NCF1 frame unchanged. Removed 45 now-unreferenced experiment
  source/header files and their standalone experimental build/CABAC notes.
  Everything remains recoverable on `feature/ng-presets`.
- Verification: 576 frozen preset frames retain identical bytes and raw hashes;
  96 legacy/indexed configurations also match the unmodified native build exactly.
  121 Python tests and four native test executables pass. Legacy C tests pass
  ASan/UBSan. New C translation units pass C11 `-Wall -Wextra -Werror` syntax checks.
- Still unfinished: preset orchestration/framing retains C++ vectors and search
  code; remaining experimental branches must be pruned, then ported to C. The
  complete library still links C++ and must NOT be called a pure-C implementation.
  Final all-channel speed comparisons and portability checks remain outstanding.

Additional reproducible check:

```sh
python benchmark/verify_legacy.py --baseline <unchanged-build> --build <port-build>
```

- Created separate branch; original branch preserved. No consumer pins changed.
- Inspection: numerical kernels already reside mostly in C; the largest C++
  reduction is experimental search orchestration and container ownership.
- Starting with shared C delta/quantization primitives; retain existing operation
  order and SSE2/NEON prefix-sum implementations.
- Extracted those primitives into `src/delta_filter.c`; C++ orchestration now calls
  the shared C functions. No duplicate filter implementation remains.
- Validation: C11 standalone tests cover all 8 masks, 6 shapes including SIMD
  tails, uint8/uint16 and all 255 quantizers; native smoke and 145 Python tests
  pass. `benchmark/verify_port.py` checks all 576 retained-preset frozen frames:
  decoded SHA256 and re-encoded bytes match exactly.
- First performance check is NOT final acceptance. GCC 13.3 Release, CPU 97,
  six real 128³ Paris4 channel chunks, alternating baseline/port, 3 runs each,
  each run best of 9 repetitions with the existing `benchmark/bench.cpp`:

  | Native Delta path | Encode MB/s min/median/max | Decode MB/s min/median/max |
  |---|---:|---:|
  | Original | 424.3 / 424.9 / 425.0 | 868.2 / 876.8 / 877.7 |
  | C filters | 426.5 / 426.7 / 426.9 | 849.4 / 849.5 / 855.5 |

  The roughly 3% decode regression needs resolving before final acceptance;
  do not describe performance as unchanged. Compression was identical.

### Commands for this milestone

```sh
cmake -S . -B <build> -DCMAKE_BUILD_TYPE=Release \
  -DVC_DELTA3D_BUILD_TESTS=OFF -DVC_DELTA3D_MODERN_SMOKE=ON \
  -DVC_DELTA3D_BUILD_PYTHON=ON -DPython_EXECUTABLE=<python>
cmake --build <build> -j 12
ctest --test-dir <build> --output-on-failure
<build>/vc_delta3d_modern_smoke
python benchmark/verify_port.py --build <build> --frames <frozen-results.json>
```

Timing harness linked unchanged against each static library (C++20, `-O3`);
`taskset -c 97 <bench> <six-raw-chunk-dir> 9 delta`. Source raw chunks and
`shape.txt` are under `/data/hendrik/compression_experiments/villa4/runs/villa-delta3d-integration`.

# Matched header improvement experiment (2026-11-02)

- Preserved the 6181.02-byte baseline and added optional representations only.
- Full 576-header corpus: best 6169.36 bytes, 328 improved, zero regressions.
- Results/reproduction: `experiments/header_best_additive.md`.
- Production framing remains unchanged; these small savings are experimental.

# Header simplification (2026-11-02)

- Tested candidate removals and cumulative simplifications on the full corpus.
- Recommended simple candidate: 6172.71 bytes vs best 6169.36 and Zstd 6173.17.
- Header compressor tables: 433.16 bytes/header (7.02%); states another 108.89.
- Details and reproduction: `experiments/header_simplification.md`.

## Remove XOR from the simple candidate (2026-11-02)

- Variant 88 uses shared wrapping-word subtraction for references and within
  records; no XOR in this candidate. Historical controls remain reproducible.
- 6172.32 bytes/header vs simplified XOR 6172.71 and Zstd 6173.17.
- Full 576-header release and ASan/UBSan comparisons passed for both byte/word
  replacements and the simplified XOR control.
- Details: `experiments/header_simple_reference_delta.md`.

## Low/high byte contexts (2026-11-02)

- Optional two-stream contexts never beat shared-byte coding on 576 headers.
- Forced diagnostic: 6482.43 vs 6172.32 bytes; keep current shared-byte candidate.
- Held-out global-distribution analysis is logged separately from actual sizes;
  fixed embedded tables are not implemented or ruled out by this test.
- Full release and ASan/UBSan round-trips passed; see `experiments/header_byte_contexts.md`.

## Typed header proposal (2026-11-02)

- Implemented four plain field families and a shared family compressor;
  selectors expanded to bytes, frequency tables to 256 uint16 counts.
- Best generic integer-compaction trial: 6585.67 bytes/header, worse than
  simplified baseline 6172.32 and Zstd 6173.17. Do not promote.
- Regression is NOT extra rANS tables: tables shrink by 21.27 bytes, states
  grow 3.78; parameters/frequencies and framing account for the larger result.
- Full 576-header comparisons and ASan/UBSan passed. Shared selector helpers
  extracted without behavior changes; CTest, smoke, frozen decode tests pass.
- Detailed field shares/commands: `experiments/header_typed_fields.md`.

## Generic element-RLE trials (2026-11-02)

- Implemented RLE on whole 8/16/32-bit elements before/after delta, followed
  by direct rANS or integer compaction + rANS, preserving previous candidates.
- All 576 headers: combined typed size 6532.01 vs typed baseline 6585.67;
  still worse than simplified 6172.32 / Zstd 6173.17. Direct-rANS new choices
  never selected. Savings almost entirely in frequency counts.
- 504 synthetic edge-case tests pass. Full matched corpus round-trips pass.
- ASan/UBSan: all 504 synthetic cases and 2,880 new-variant corpus cases pass.
- Details and exact commands: `experiments/header_element_rle.md`.

## Correct typed framing / RLE / local choices (2026-11-02)

- Separate ablations for derived array shapes, separate RLE control/value
  contexts with implicit zero tails, and generic record-local reference choices.
- Final 106 also derives nested raw/constant encoded lengths. 6337.04 bytes/header
  vs typed 6532.01; still above simplified 6172.32 / Zstd 6173.17. Not promoted.
- 576 release round-trips per variant; 72 final synthetic tests pass release and
  ASan/UBSan. Four CTest tests, API/AVX2 smoke, 576 frozen raw hashes pass.
- Full ASan/UBSan 99–105 corpus sweep remains running separately; not claimed
  completed. Results directory: header_typed_fixes_sanitize_20261102.
- Details, field sizes and commands: `experiments/header_typed_fixes.md`.

## Generic compaction before transforms + shared pools (2026-11-02)

- Shared family helper for standalone/pooled coding; schema-array pool references
  share encoded data/table/states. RLE controls remain separate from values.
- Generic zero-tail/integer compaction precedes optional references/deltas.
- Full 576-header matched ablations: compaction 6090.13 B, pools only 6333.81 B,
  combined 6088.50 B vs typed 6337.04, earlier simplified 6172.32, Zstd 6173.17.
- 558 synthetic release/ASan/UBSan checks and 54 representative full-header
  sanitizer checks pass; all corpus round-trips, CTest and API/AVX2 smoke pass.
- No production/default change. No speed claim. Detailed all-group results and
  commands in `experiments/header_compaction_pools.md`.

## Post-transform entropy sharing (2026-11-02)

- Freeze compaction/transforms; test shared fitted rANS tables and combined
  payloads separately. Controls/values remain separate. Add a supplied-table
  entry point reusing the existing native rANS kernel; original API unchanged.
- 576-header existing-rANS test: 0 B total saving for table sharing, 66 B total
  for shared states too, before any new references/framing. No header has two
  numerical rANS streams. The common case is one control + one numerical stream.
- Admit raw/packed literals as well: 5 B total table-only saving; 9.8125 B/header
  combined-payload saving before references. Three larger cases use greedy
  pair/all-role search, all others exhaustive. No new framing/default promoted.
- Details/reproduction: `experiments/header_entropy_sharing.md`.

## Selected compaction modes (2026-11-02)

- Added decoder-only telemetry; 576 unchanged header sizes and exact round-trips.
- 583 compact-first groups: 477 unsigned varint, 92 zigzag, 14 fixed-width;
  11/27 transform/reference combinations selected. 60.12% of their elements
  are implicit zero tails; retained elements average 1.044 compact bytes.
- Explain length-based early end (not an EOF symbol), and propose direct
  integer magnitude-bucket rANS as a simpler alternative; no codec change made.
- Detailed audit appended to `experiments/header_compaction_pools.md`.

## Log-range compaction, preserving zero tails (2026-11-02)

- Added exact small-value/log-range labels plus literal offset bits; retained
  counts signal skipped zero tails. Existing choices, references/deltas and
  pooling retained. Shared byte-transform helpers avoid duplicate implementations.
- 576-header means: baseline 6088.50 B, unsigned log 5975.07 B, optional zigzag
  log also 5975.07 B. 420 improve / 156 equal / 0 worse; signed log never selected.
- Baseline exactly matches saved results; full 1728 round-trips pass. 1230
  synthetic release/ASan/UBSan checks and 39 full-header sanitizer checks pass.
- Corrected helper-extraction bug for empty references; initial failed run is
  superseded by header_log_range_v2_20261102. No production/default change.
- Details and all-group results: `experiments/header_log_range.md`.

## Header simplification (2026-11-02)

- Measured 11 independent removals, then four combined profiles on all 576
  headers. Added actual nested-codec telemetry; no Python compression.
- Recommend experimental 126: log-range numerical compaction, zero-tail skipping,
  no RLE/old compaction/reference search/pooling. 6038.39 B/header versus
  5975.07 full alternatives and 6173.17 Zstd. RLE restores only 6.99 B/header.
- Retain signed log mapping: now selected in 182/1152 numerical groups after
  older alternatives are removed. Historical experiments stay reproducible;
  production defaults unchanged. Full results: `experiments/header_simplification.md`.
- All corpus round-trips pass, 1230 sanitizer synthetic cases, 72 representative
  full-header sanitizer checks, and 4/4 production CTests pass.

## Minimal header feature set (2026-11-02)

- Tested the full 16-way sign/packing/delta/RLE factorial, three log-free
  controls, and eight further removals on the same 576 frozen headers.
- Minimum-feature candidate154: unsigned log + zero tails, fixed word delta
  on labels, packing+rANS, control varints and raw/rANS fallback. No signed
  mapping, RLE, constant coding or standalone packed mode. 6148.25 B/header
  versus Zstd6173.17. Candidate151 retains optional delta: 6076.81 B/header.
- Signed mapping saves28 B, RLE7 B in the compact setting; packing+rANS is
  important (about566 B). Constant codec is unused. No production changes.
- 15552 exact corpus round-trips, 1230 sanitizer synthetic cases, 486 full-header
  sanitizer checks; 4/4 CTests. `experiments/header_minimal_features.md` has
  complete combinations, methodology and reproduction commands.

## Corrected shared numerical table test (2026-11-02)

- Earlier one-table statement did not apply after removing pooling: 172/576
  headers in147 actually had two numerical tables. Added explicit counts.
- 155–157 force model/frequency records into one numerical pool using existing
  implementation; assert at most one numerical rANS table. Controls remain
  separate. All 3456 matched round-trips pass; 54 full-header sanitizer checks.
- Shared packing allowed6058.875 B/header; standalone packing removed6061.875;
  no packing6628.573. Standalone choice now saves only3 B/header: drop it.
- Details: `experiments/header_shared_numeric.md`. No production format change.

## Selector payload separation and block sizes (2026-11-03)

- Restricted packing to selector payload in candidate158; all576 header sizes
  unchanged versus156, exact round-trips. Production defaults unchanged.
- Native five-way selector storage trial over18groups ×8chunks × current
  fast/slow ×2³/4³. All2884³ frames reproduce frozen release exactly; all
  generated2³ frames lossless. Python orchestration only; no codec reimplementation.
- Packed-rANS savings versus byte-rANS:26.5/20.9% fast2/4;16.4/10.5% slow2/4.
  Separated RLE loses overall. Packed Zstd close overall; control only.
- 2880 initial selector round-trips,360 sanitizer checks, CTest4/4.
  Full report/reproduction: `experiments/selector_payload.md`.

## Generic selector contexts and RLE (2026-11-03)

- Added prev1/prev2 rANS contexts and run-symbol contexts with independent run
  lengths. Native C only, reuse existing rANS kernel; no production changes.
- Same576 frames, all fourconfigs and18groups. No voxel re-encoding. Prev1
  improves packed-rANS selector bytes by5.86/8.00/4.64/3.31%; beats packedZstd
  aggregate but not CT. Prev2 only slightly wins fast2. RLE helps CT, not total.
- Prototype prev1 decode290–300MiB IDs/s versus593–764 packed-rANS in this run;
  not a free speed improvement. 5184 exact stream checks, baseline sizes identical.
- Full results/reproduction: `experiments/selector_context.md`; all-channel
  raw/report data under encoder_speed/selector_context_20261103.

## Minimal selector LZ trial (2026-11-03)

- Four native variants: raw/packed × rANS/previous-byte literals; greedy4-byte
  hash matches,64KiB window, separate literal-run/match-length/distance streams.
  Reused rANS and factored generic context-bank helpers instead of duplicating.
- Same576 frames; no voxel re-encoding. Packed LZ helps fast CT L0 substantially
  (790→139B at2³,161→60B at4³), but Zstd still61/25B. All LZ variants lose on
  aggregate; context literals worse. Keep experimental, no default promotion.
- 7488 exact round-trips;5184 previous sizes identical;320 synthetic tests and
  936 full corpus sanitizer checks. Details: `experiments/selector_lz.md`.

## Zstd CT L0 sequence audit (2026-11-03)

- Native sequence extraction and parser/coder cross tests on32saved CT L0 frames,
  raw and packed. Original sizes unchanged; supported cross streams lossless.
- Hard fast2³ map: ours1009B, same matches with Zstd577B, original Zstd395B.
  Only5 vs8 literals remain. Zstd341/372matches use repeat-distance codes.
  Our tables193B +states128B are substantial, but not the whole explanation.
- Fast4³ hard map: ours166matches vsZstd79, sizes469 vs183B; own table/state
  overhead70B. Faster entropy coding alone is not the target; improve sequence
  representation/repcodes and then match choice. No implementation promotion.
- Full ASan/UBSan audit passes; fixed unsupported-cross diagnostic leak.
  Results/reproduction: `experiments/selector_zstd_audit.md`.

## Small-state LZ / predefined tables (2026-11-03)

- Single32-bit rANS state, sparse transmitted tables or four deterministic
  distributions. Magnitude-coded sequences, three repeat distances, shared
  match parser with32/256-candidate search. No production/default changes.
- CT L0 selector means fast2/4,slow2/4:33.625/14.25/1179/147.875B versus Zstd
  61/24.625/1304/178.625B. Hard fast map uses no transmitted probability counts.
- Not an all-channel winner: previous-symbol context still wins aggregate.
  Fixed-only probabilities lose; optional exact-byte choice is useful.
- Full576-frame corpus,22 codecs, all7488 previous control sizes unchanged.
  Details and reproduction: `experiments/selector_small_lz.md`.

## Context + LZ and local switching (2026-11-03)

- Tested whole-map exact-byte choice, raw/packed context-coded LZ literals,
  and independent1/4/16/64Ki-ID switching regions. Native C, existing shared
  bank helpers refactored for small/bulk entropy callbacks; production unchanged.
- Whole choice selects context483,LZ46,uniform47 maps. Combined selector savings
  vs packed Zstd7.34/8.96/4.20/1.75% for fast2/4,slow2/4. Literal-context candidates
  never selected. Local choice wins only5 maps, all64Ki-ID, all-context regions:
  negligible gain from probability adaptation, not actual mode switching.
- Recommendation: whole-map choice only. Keep other tests experimental.
  Full results, caveats, commands: `experiments/selector_switch.md`.

## Unified selector entropy backend (2026-11-03)

- Tested identical small-state rANS backend for context and LZ, shared context
  callback helpers. 491maps smaller,85equal,0worse; combined savings vs prior
  choice0.43/4.58/0.37/2.89% (fast2/4,slow2/4).
- Decode roughly halves (152–175 vs307–317MiB selector IDs/s); encode also slower.
  Keep experimental, production unchanged. Details: `experiments/selector_unified.md`.
- 3456 corpus checks,1400synthetic, ASan/UBSan;2304 old controls unchanged.

## Retain unified codec / cleanup (2026-11-03)

- User selected the simplified codec. Promoted to `src/ng/selector.c` with
  checked failure returns, shared context/LZ entropy; no experimental switches.
- Integrated external selector payload via NCF1 marker253, retaining old
  packed/rANS selector reads and legacy Delta3D. No double Zstd compression.
  Remaining general model/header Zstd explicitly unchanged.
- Archived then removed15 prototype files and unused bulk-table API; historical
  results/data retained. Archive path and commands: `experiments/selector_retained.md`.
-576 frames match selected experiment sizes exactly and decode unchanged voxels;
  release and sanitizer corpus/tests pass. Full decode overhead~2% at4³,
  ~16–17% at2³ versus original packed-Zstd frames. Details in integration report.

# 2026-11-03 — one compact evolving frame, no experimental compatibility

Completed the requested simplification; see `HEADER_FORMAT.md` and
`experiments/header_cleanup.md` for format, tests, exact commands and results.
Preserved generic menus/models/banks/fixed block sizes and indexed access.
Removed old selector branches, model-v4 path, rectangular alternative framing,
and Zstd dependency. Old Delta3D alone retains compatibility/fallback.

Test-first header result:1664.304 B vs matched Zstd1665.292 B (selectors excluded
equally; near parity). Whole-frame saving also includes the already-selected
selector codec and must not be attributed solely to the header. Failed extra
reference/control-coding trials were discarded, not left as selectable modes.
576 frozen cases pass Release and ASan;144 source chunks ×4 presets ×3 fresh
repetitions pass; CTest6/6 and Python124/124. No commit requested this turn.

## Fixed header table transfer experiment

Completed native cross-validation over576headers, global/preset/predictor/joint
groupings, plus18leave-channel/scale-out tests. Global transform-specific pair
saves6.47% of header size; per-preset saves only another4.2B/header.
Production header unchanged. See `experiments/header_static.md` for results,
limitations and reproducible commands. Explicit-frequency encoder shares the
existing rANS body, with byte-parity regression tests for ordinary frequencies.

## Fixed header tables integrated; compact repeat containers

- Global two-table header encoding selected; raw alternative retained. Optional
  transmitted-table wire mode is decoded/tested, not searched by default.
  Four-preset frozen corpus:2049.51 ->1916.16bytes/header, all576voxel hashes match.
  Independent fast/slow2/4corpus576cases also passes. Header integration Release
  and ASan/UBSan CTest6/6; explicit dynamic-mode decoder regression covered.
- Implement compact NGC1/extended NGX1 fixed-width tree, repeated subtrees,
  cumulative64-bit offsets, 16/32-bit header lengths, no varints. Reuse leaf
  decode without a copied prefix. See INDEXED_FORMAT.md for grammar and APIs.
  Eight full-volume channels:78bytes total framing, previously259bytes.
- Final widths:11/25-byte container prefixes, 2/4-byte header lengths,
  1/4-byte root and tree fields, 4/8-byte leaf lengths. Distinct NGC1/NGX1
  magics. Release CTest6/6, C++/AVX2 smoke, Python135/135 passed. Tests include
  256 repetitions, nested spatial/channel repeats, >64KiB indexes, >4GiB
  cumulative ranges, expanded root dimensions, malformed templates and
  independent leaf range decoding. Legacy Delta3D remains unchanged.

## Container magic version correction

Verified remote main98bb019 uses NGC1 (src/indexed.cpp). Renamed the new compact
and extended formats to NGC2 / NGX2 throughout active code, tests and docs.
No historical container reader added; explicit rejection tests cover both v1
magic strings. Earlier task entries retain their historical names.

## Source cleanup and encoder/decoder separation

Removed unused original VCNG/Huffman frame paths, retaining live NG-derived
kernels and original Delta3D compatibility. Split native entrypoints, kernels,
leaf framing and indexed-container code by role; shared helpers are not copied.
See experiments/core_cleanup.md and benchmark/source_sizes.py for exact counts,
classification, commands and validation. Production source is 8,452 lines,
including 93 SIMD .inc lines omitted in the previous count (corrected previous
total 9,583). All 576 corpus frames byte-identical, Release and sanitizer CTest
6/6, Python136/136, C++/AVX2 smoke passed. No commit requested.

## Retained-path cleanup, stage two

Removed experiment-only encoder fitting/entropy modes and old unused recurrence
kernels. Decoder settings are per-call instead of nested TLS overrides; trial
bitmask dispatch has been replaced with the reachable production paths. Shared
bounded parsing helpers serve header, selector and leaf decoding. Generic
format tests explicitly cover choices not selected by the current presets.
See experiments/option_cleanup.md for checkpoints, counts and commands.
Code total 8,452 -> 7,880; decoder core 1,675 -> 1,320. Corpus checks pass at
each checkpoint; final serial before/after timing completed with all 576 frames
byte-identical. Combined decode MiB/s: fast 589.5 -> 627.5, medium 430.2 ->
447.4, slow 423.1 -> 456.1, slower 412.9 -> 445.0. Release/sanitizer/portable
CTest, 136 Python tests, C++ smoke and 684 generic-format cases passed.
No commit requested. Menu-to-kernel descriptor redesign remains a separate
next step.
