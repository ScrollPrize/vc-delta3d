# Experimental NG fork: Delta3D / LPC fallback

Experiment naming: **Fast** and **Slow** mean the historical fixed and
three-choice policies below. **Slower-RC configs 1/2/3** mean the broader
menu search, expanded all-coefficient search, and expanded+tree search.
See [configuration names](../../../experiments/ng_config_names.md); these
labels do not renumber the public preset API.

## Current native API: NCF1

Use `ng_codec_encode(src, d, preset, dst, cap, written)` and
`ng_codec_decode(frame, bytes, dst, cap)` from `ng_packed_bank.h`.
`NG_CODEC_FAST` / `NG_CODEC_DEFAULT` (**3**) is the fast/default policy:
screening followed by a single predictor for occupied blocks and DP64 residual
thresholds. Completely zero4³ blocks use an ordinary zero selector (no separate
map). The block screener avoids empty regions; full NG fits exclude zero-block
targets while retaining the original sparse sample budget, redistributed over
eligible targets (capped by availability). Decoded zeros remain causal inputs.
`NG_CODEC_SLOW` (**1**) is adaptive: three choices, initial fits, selection and
one reselection, then final entropy tables, without iterative NG refitting.
The API requires an explicit preset; use these named constants. Numeric0
retains the old quantile-only policy for reproduction; its meaning is unchanged.

NCF1 independently references predictor models and residual entropy banks from
its choice table. Models can be shared or independent, including multiple
models of the same kind. Uniform chunks omit the block-selector payload.
Search policy is not part of the decoding contract. Both paths are native;
Python is only a benchmark/test driver. Older APIs below remain experimental
reproduction interfaces, not the new preset defaults.

Format bounds, validation and measured performance:
[`experiments/ng_unified_codec.md`](../../../experiments/ng_unified_codec.md).
Preset3 keeps the fixed predictor but globally optimizes entropy thresholds
on a64-bin candidate grid. Other search ablations and all measurements:
[`experiments/ng_threshold_search.md`](../../../experiments/ng_threshold_search.md).
The adaptive independent-model decoder is not yet at the 250–400 MiB/s target;
do not infer its speed from the older shared-model NWP1 results.

Experimental `ng_codec_encode_fast_config` and `ng_codec_encode_slow_config`
expose independent NG/entropy class counts, tap budgets, causal neighborhoods,
and fitting strides without changing the two default policies. See
[`CT L0 Slow tuning`](../../../experiments/ng_ct_l0_slow_hyperparameters.md).

L1 fitting was tested and removed; its negative results remain documented in
[`joint L1 LP results`](../../../experiments/ng_lad_lp.md). There is no GLPK
build/runtime dependency. Native entropy-aware coefficient refinement is an
encoder-only experiment, disabled by default; it does not change the format.

### Retained decoder scheduling improvement

The NWP1 default now decodes each prepared row's entropy symbols immediately,
with cache-line-aligned rolling directional history. The row width is a
multiple of32 rANS states, so this preserves the state/symbol order and every
encoded byte. Predictor fitting, choices, and integer reconstruction are
unchanged. Delta-only frames retain their existing group path. This is not
full entropy/reconstruction pipelining and introduces no threads.

Trials and limitations: `../../../experiments/ng_history_followups.md`.
The no-argument `local/scripts/run_compression_experiment` validates and
benchmarks the selected default against frozen prior/native-NG libraries.
Failed homogeneous-bypass and NG-tap-history implementations are archived,
not retained in production loops. Scalar/ISA fallbacks remain; this round's
timings are x86 only.

## Default shared-fit bank format (NWP1)

`ng_packed_bank_wave_encode(src, d, mode, dst, cap, written)` is a complete
native encoder for shared correction (mode2) and extra-input (mode3).
`ng_packed_bank_wave_repack` converts an existing shared bank without refitting
or changing any block choices. Use ordinary `ng_packed_bank_decode` to decode.
Capacity remains `3*d*d*d + 131072`. No Python codec operations are involved.

This version follows original NG's entropy execution: its wavefront scheduler,
native rANS encoder, SIMD rANS decoder, and row reconstruction are integrated.
It replaces the old full-chunk class-stream expansion and raster demultiplexing.
The new encoding is explicitly identified as NWP1 (not an old-frame speedup).
Its little-endian header stores the decompressed/compressed metadata byte counts,
then Zstd3 metadata, then native NG rANS states and payload. Metadata expands
to an NBK3 bank prefix and the original native entropy parameter/table header.
Both predictor and entropy thresholds are retained, and all24 choices remain.
Delta-only chunks use a zero-weight history model and can bypass NG reconstruction.

See `../../benchmarks/ng_shared_decode_speed_v1/integrated_entropy*.md` for
matched single-core speeds, complete frame sizes and phase profiling. Original
NG and both shared fits are reported separately. The approved no-argument
`local/scripts/run_compression_experiment` builds Release, tests, and measures
the locked48 inputs. x86-64 AVX2 measured; C/NEON fallbacks are provided but
ARM/macOS performance has not been measured. Existing frame decoding remains.

## Current recommended bank configuration

Native packed bank **mode 3**: shared extra-input NG fit, full 24-choice bank,
4³ block selectors, compact NG parameters and joint Zstd probability tables.
The legacy-frame default decoder is compact shared-model SIMD (control 14), with exact
raster fallback for nonshared fits or unsupported CPUs. Both metadata
preparation and row reconstruction are vectorized on AVX2, with the original
NG scheduling and SIMD transposes. Shared coefficients stay per row; five
metadata fields replace eleven. The branch-free activity gather remains.
Control 4 is the previous raster baseline. Earlier failed trials remain
diagnostic-only. Control 1 is
the previous raster reference; control 0 is the earlier wavefront experiment.
Legacy-frame optimizations do not change encoded bytes.

The default `ng_packed_bank_encode` now writes NWP1 for shared-fit modes2/3
(mode3 remains the recommended baseline). `ng_packed_bank_wave_encode` is
an explicit equivalent entry point. `ng_packed_bank_legacy_encode` retains
the old format for reproducing prior experiments or older decoders; modes0/1
remain legacy-only because they do not share their fitted coefficients.
Its NWP1 decoder integrates original NG wavefront SIMD entropy and reconstruction,
fuses predictor/entropy classifiers, avoids raster residual gathering, and uses
non-aliasing residual/history updates. Delta-only banks bypass unnecessary NG
reconstruction. This is the selected integrated implementation (diagnostic
control21), with an exact NG+X fast path automatically selected when at least
25% of saved block choices use NG+X (diagnostic control27). That fast path
folds the X feature coefficient into NG's first causal tap, avoiding generic
directional-base handling without changing arithmetic or compressed bytes.
NWP1 requires
the updated decoder. The metadata inspection API now handles NWP1 too.

The decoder additionally uses four metadata planes (the residual byte shares
the flag word), an exact combined activity-to-NG-bias/entropy-class lookup,
and a uniform clampedXYZ shortcut. These are decoder-only changes; old NWP1
frames are byte-identical and remain valid. The lookup is derived once from
the stored thresholds and biases, not fitted or transmitted. It is skipped
for Delta-only chunks. AVX2 paths are guarded; portable fallbacks remain.
Diagnostic control41 selects this previous combination.
Default14 (previous control57) additionally fuses SIMD activity directly into the lookup,
interleaves the two reconstruction lane groups every eight positions, and
reuses the entropy-symbol buffer for residual handoff instead of routing it
through the complete output chunk. All are exact decoder-only changes.
Rejected tiling, coefficient caching and workspace reuse implementations are
archived in `../../../experiments/patches/nwp_six_decoder_trials.patch` against
commit4e6d251a4, rather than retained in the active decoder. Full ablations:
`../../benchmarks/ng_shared_decode_speed_v1/nwp_six_targets*.md` and
`nwp_tiled_reconstruction.md`. Controls47/49/51 isolate the retained changes;
55 combines lookup/interleaving,57 adds local handoff. No extra decoding threads.

Default14 (previous control65) bypasses generic metadata for wavefront row groups
containing only NG+X blocks. It omits unused constant/bounds planes and their
transposes, using the same integer SIMD recurrence. At x=0 the Delta-X feature
is zero while the NG causal history retains its original boundary value.
Mixed groups and non-AVX2 CPUs retain the generic path. Encoded bytes and all
decoded values are unchanged. Control57 retains the preceding implementation.
The five-target experiment also tested selector runs, direct fixed331 taps,
10/11-bit entropy tables and zero-residual block bypass; none was selected.
Full results and archived prototypes: experiments/compression_log.md and
experiments/patches/nwp_five_decoder_trials.patch. NWP1 remains the format.
Control27 retains the previous NG+X baseline for ablations.

Default14 (also69) uses AVX2 hardware gathers for native entropy table loads.
The shared entropy implementation is constant-specialized; original NG keeps
its scalar-load gather wrapper. Previous bank control65 remains available.
Empty-refill branching, lookup pipelining and AVX512 masked expand were tested
and rejected, archived in experiments/patches/nwp_entropy_trials.patch against
fbbfa6d32. This does not change frames, probabilities or reconstructed values.
Portable C/NEON paths are unchanged. Measured on Intel Xeon Platinum8480+;
performance on other AVX2 implementations still needs measurement.

Current measurements: `../../benchmarks/ng_shared_decode_speed_v1/nwp_entropy_selected.md`
and its JSON with raw timings, mean/p50/p95, sizes and native phase profiles.
This retains all24 block choices and fitted parameters. Entropy ordering changes
compressed bytes and slightly improves size; decoded voxels remain byte-exact.

Shared correction (mode 2) was tested too: modestly faster CT decoding, worse
NX4 decoding, slightly worse compression. Keep mode 3 as the baseline.
See `../../benchmarks/ng_shared_decode_speed_v1/compact_simd.md` and
`compact_simd_confirm.md` for matched serial speed/phase results; JSON includes
mean/p50/p95. Reproduce via the approved no-argument
`/home/hendrik/villa4/local/scripts/run_compression_experiment`.

## rANS table-storage experiment (RPK1)

`scripts/ng_rans_table_storage.py` wraps model-packed NBK2/NPK1 frames without
changing their models,choices,probabilities,rANS states or residual payloads.
Current compact frequency headers use uint16 symbol count and unsigned varints,
trimming trailing zero frequencies;sum is4096. The experiment extracts the
selector table and up to16 nonempty residual-class tables. An RPK1 header holds
packing ID,table count,decoded/stored lengths,and varint byte gaps locating the
removed tables in the otherwise untouched original frame. Decode reinstates
the canonical tables and passes the byte-identical original frame to the
existing model-header/native voxel decoder. Wrapper bytes are included in costs.

Trials:infer final frequency;nonzero bitmap;nonzero index gaps;Zstd3 of joint
compact tables;bitmap+Zstd3;sparse signed differences from the previous nonempty
residual-class table+Zstd3;differences with inferred final frequency+Zstd3.
Selector and first residual table are absolute,not treated as neighboring
difficulty classes. Each trial retains the original frame if larger,but logs
forced sizes separately. Python header prototype only,no production/default
or speed changes. Reproduce with `local/scripts/run_compression_experiment`;
outputs `benchmarks/ng_rans_table_storage_128_v1`.

## Model-bank storage experiments (NBK2/NPK1)

`scripts/ng_model_storage.py` changes only parameter representation of existing
NBK1 results. NBK2 inserts a uint16 active-model mask after the selector, stores
only present model records, and uses the shared native bank decoder with absent
states skipped. Missing models referenced by a selector are rejected. NBK1
remains supported. All absent models are genuinely unnecessary for decoding;
selected-residual history is shared, not dependent on unused predictions.

NPK1 retains dimension/mode/selector and stores mask,packing ID,decoded length,
stored length,and packed active parameters before the unchanged residual stream.
Packing choices: sparse nonzero bitmap+signed zigzag varints (thresholds as
successive differences),Zstd3 on raw records,sparse+Zstd3,and differences from
one active model at matching coefficient positions followed by sparse+Zstd3.
For differences,try all active reference IDs plus absolute representation and
retain smallest measured bytes; the reference ID is included. Native parameter
bytes,including Q8 weights and unsigned bias bit patterns,are reconstructed
exactly before native decoding. Python is only an experimental header codec.
No decode-speed claims or production metadata changes.

All options omit unused models first and fall back to NBK2 raw parameters when
their complete frame is not smaller. Masks and packing wrappers are included
in the reported model overhead. The benchmark independently checks byte-exact
active records,selectors,residuals,source hashes,prior full frame sizes/block
choices,and all source voxel roundtrips. Run the unchanged no-argument
`local/scripts/run_compression_experiment`; results in
`benchmarks/ng_model_storage_128_v1`.

## Directional bank trial (NBK1)

`ng_direction_bank_encode/decode` tests24 choices per4³ block on a uint8 cube:
direct;7 unclamped Delta masks;clamped XY/XZ/YZ/XYZ;plain NG;11 separate NG
models attached to those bases. Mode0 fits signed-modulo target-minus-base
against source neighbors, then adds the base. Mode1 fits target against source
neighbors and the base as an extra learned input. Each uses the native fitting
code, independent main/first-slice spatial banks and difficulty corrections.
Single-axis clamping is identical and omitted. At boundaries with a missing
selected axis, clamped variants retain the legacy available-axis Delta fallback.

All candidates share actual selected-residual causal activity, but each NG
model retains its own fixed prediction thresholds/biases. Entropy thresholds
are independently refined. Shared native class/symbol kernels generate encoder
candidates; the shared native causal decoder reconstructs each model. The
experimental decoder evaluates all models: this is a compression trial, not a
decode-speed result. Three selector/entropy passes per seed retain smallest
actual frame. Seeds include prior clamped Delta and Lorenzo-input NG choices;
mode1 checks exact old model and residual payload preservation at that seed.

NBK1 stores dimension,mode,compact rANS selector,12 length-prefixed native
parameter sets (plus11 pairs of int32 input weights for mode1),and the shared
16-class activity rANS payload. All models are stored, even unused ones; no
whole-frame fallback or hidden omission of overhead. Stats28 entries contain
model bytes (including lengths/input weights),selector bytes,residual payload
bytes,outer header bytes,and24 block counts. These four sizes sum to frame size.
No production/default codec changes. Reproduce via the no-argument approved
`local/scripts/run_compression_experiment`; results in
`benchmarks/ng_direction_bank_128_v1` with strict128³ inputs and roundtrips.

## Lorenzo inside the fit, and direct choice (NAL2)

`ng_lorenzo_encode` additionally accepts16=centered fit,32=extra input (mutually
exclusive),64=direct values as last choice. Thus64/68 are direct-enabled NG/Delta
plain/refined;65/69 also include explicit Lorenzo;67/71 also include the separate
correction model;80/84 use centered NG;96/100 use extra-input NG. Existing8
still forces trial frames for tests. Stats now require14 uint64 slots: baseline
bytes, trial bytes, won flag, and11 selector counts. Direct's ID is the last
candidate (8,9,or10), depending on enabled predictors.

Let L be causal clamped Lorenzo. Centered fit trains target-L against each
neighbor-L, then predicts L+bias+sum(w*(neighbor-L)). Zero weights give Lorenzo;
the effective Q8 Lorenzo coefficient is256-sum(neighbor weights). Extra-input
fit instead trains target on neighbors plus L with a free coefficient. Both
reuse original integer Gram statistics, selection, ridge, coefficient rounding,
first-slice/main spatial banks and class correction fitting. Centered targets
are actual signed differences, not modulo-wrapped training targets. Final
prediction/residual arithmetic remains native modulo256. Features are int16,
including negative contrasts, with exact integer sufficient statistics.

The two effective Lorenzo coefficients (one per bank) are serialized as int32
in NAL2. Native parameter blocks retain the neighbor weights and class biases.
Encoder prediction export includes the feature contribution, and decoder adds
the same coefficient times causal L before rounding. New fits replace the NG
candidate; they do not implicitly add an independent Lorenzo choice. Direct is
available in every new benchmark variant. Prediction thresholds stay separate
from entropy refinement. Old NAL1 and baseline fallback semantics are retained.

Results: `benchmarks/ng_lorenzo_fit_direct_v1`; run with the approved no-argument
`local/scripts/run_compression_experiment`. No default promotion or speed claim.

## NG/Delta4 + Lorenzo (+ NG correction) experiment

`ng_lorenzo_encode/decode` use4³ blocks with original full NG as choice0,
seven Delta masks1..7, optional clamped3D Lorenzo8, and optional Lorenzo+NG9.
Options:1 adds Lorenzo,2 additionally adds the correction,4 refines,8 forces
the experimental frame for tests. No direct-only choice is added. Correction
is a separate original NG fit (spatial banks and class biases) to signed modulo
residuals relative to Lorenzo, using original source neighbors as features.
Both NG predictors retain native fractional-Q8 residual sign mapping.

The shared native selected-symbol path regenerates classes/predictions/residuals
after each selector update. Both predictors see selected-residual activity.
Original NG prediction thresholds/biases remain fixed, while independent rANS
thresholds may be quantile-fitted and coordinate-refined. The shared selector
cost helper is also used by earlier activity experiments. Plain evaluates two
seeds once; refined evaluates three passes per seed, retaining the smallest
actual encoded frame rather than trusting estimated cost.

NAL1 contains dimension, choice count, compact rANS selector stream, native
model parameter block(s),15 entropy thresholds and16 compact rANS streams.
These residual streams are raster-ordered, unlike baseline interleaved native
wavefront payload. A no-extra matched control isolates that coding/refinement
change. Full original NDL1 frame remains available as fallback with exact-byte
checks. Output stats record baseline bytes, trial bytes, trial won, and10 trial
selector counts even when the baseline wins. Forced-frame tests validate NAL1
decoding without fallback hiding errors. Current defaults are unchanged.

Reproduce via the no-argument `local/scripts/run_compression_experiment`.
Results and trial-before-fallback tables: `benchmarks/ng_lorenzo_addition_v1`.

## Soft native class corrections (NSC1 experiment)

`ng_soft_class_encode/decode` keep native NG tap fitting, thresholds, per-class
bias estimates and hard rANS activity classes. Prediction alone interpolates
Q8 biases at nonempty class interval midpoints, based on causal numerical
activity. Outside the anchors clamp; the last bin uses the preceding bin width
instead of the unbounded tail. Blend1 uses the interpolated bias, blend2 averages
it with the hard-class bias, blend0 preserves the original. No new fit or model
coefficients. Shared scalar class application handles the blend on all platforms;
ordinary hard-class paths retain existing SIMD. This experiment measures sizes,
not decoding speed.

Self-describing NSC1 frame: magic, uint32LE cube dimension, codec kind (0 NG,
1 NG/Delta4), blend mode, then existing payload. Its16-byte header replaces the
NDL1 header for hybrid leaves, or adds16 bytes for NG alone. Hard payloads are
verified identical to the original. Scoped thread-local dispatch is restored
on success/failure; only the NSC1 wrapper activates blending. Production defaults
and existing bitstream interpretation are unchanged. Encoder and decoder both
use the final selected residuals to update activity. Hybrid block selection is
rerun using the softened initial NG residuals and unchanged Delta candidates.

Reproduce via `local/scripts/run_compression_experiment`; results in
`benchmarks/ng_soft_classes_v1`.

## Difficulty-conditioned prediction (NAF2)

The activity experiment's flags32/64/128 (mutually exclusive, requiring16)
select three models:32 difficulty as an extra scalar input;64 two independently
weighted-fitted models blended by difficulty;128 jointly fitted neighbor×
difficulty interactions (varying coefficients). Flag256 increases the new-term
ridge from0.001 to0.1 times mean covariance diagonal. Existing4/8 control
entropy-threshold refinement; no model is selected using unknown current data.

Difficulty uses the same NG causal residual-activity stencil, but computed on
residuals of the stored shared fitted reference predictor. This reference is
fixed independently of block choices and entropy thresholds. Normalize/clip
approximately p10–p90 into Q8 [0,256], storing both bounds. Selected taps come
from the reference NG fit; features are centered bytes plus an intercept.
Shared native solver/Q8 coefficient quantization are reused. The joint model
fits all features at once; the blend fits two separate weighted least-squares
systems with weights1-t and t. This distinction matters even though both are
linear functions of t at decode time. New fits use the native-style spatial
sample lattice with stride128 (32 for tiny regression-test chunks).

NAF2 extends NAE1 by another length-prefixed model after the reference params.
It contains mode, tap count, activity bounds (u16LE), tap IDs, and two signed
Q8 coefficient arrays including intercept slots. Additive-input mode uses only
one extra coefficient in its second array; padding is counted in compressed
size. Every voxel advances the reference predictor using its own residual,
regardless of the chosen predictor. Existing causal padding/state is reused.
Stable difficulty avoids circular block-choice/refitting dependencies; the
conditional model is fitted once, entropy classes/selectors may be refined.
Experimental only; no production/default changes or throughput claims.
Flag512 adds the matched no-difficulty refit control. NAF2 removes the constant
from the normal equations and recovers the intercept after quantizing neighbor
coefficients, retaining native1/256-voxel intercept precision. Sampling now
matches the native NG lattice. The earlier NAF1 trial used a coarse intercept
and different sampling; its results are archived in ng_activity_conditional_v1
and superseded by ng_activity_conditional_v2. No NAF1 payload files were saved.
If one weighted blend model has zero training mass, it copies the other model.

## Single fitted predictor correction (NAE1)

Activity experiment flag16 adds one fitted NG predictor instead of the fixed
clamp (flag1); they are mutually exclusive. One native fitted coefficient bank
and intercept covers all voxels, including the first slice. No class-dependent
bias corrections. Sixteen independent difficulty rANS contexts remain. Original
NG already shares main coefficients across difficulty classes, but normally
has class biases and two banks (first slice versus remaining slices).

NAE1 is NAD1 plus a uint32 parameter length and native parameter block after
the selectors. To reuse the existing parser/reconstruction unmodified, both
serialized boundary banks contain the SAME fitted coefficients; all class
biases are zero. These redundant header bytes are included in the benchmark.
Prediction uses native fitting, padding, fixed-point arithmetic and rounding;
the residual uses the activity experiment's modulo-byte pyramid mapping.
Decode uses the existing incremental native predictor, advancing it for every
voxel even when a Delta/direct choice wins. This is a correctness prototype,
not a tuned decoder. Flags20/28 add final/iterative thresholds;18/30 additionally
split rANS contexts by choice. No production/default change.
Results: `benchmarks/ng_activity_fitted_v1`.

## Activity refinement experiment (NAD1)

`ng_activity_refined_encode/decode` keeps the previous NAC1/NAB1 frame as a
size fallback, and optionally emits NAD1. Predictor choices are direct +7 Delta
masks and optionally one parameter-free clamped 3D Lorenzo predictor. Clamp
uses the min/max of immediate left/up/back values; boundary handling uses
available Delta axes. No coefficients are fitted.

Flags:1 extra predictor;2 rANS tables per difficulty × predictor;4 threshold
refinement on final selector iteration;8 (with4) threshold refinement at each
selector iteration. All combinations use two selector re-scoring rounds from
old and local seeds, retaining the smallest actual complete frame. Threshold
refinement uses two coordinate-descent sweeps over64 activity quantiles plus
old boundaries, estimated normalized histogram/table cost, then actual-size
comparison against quantiles. Not a global optimum. Decode is unchanged by
how boundaries were selected; thresholds are stored explicitly.

NAD1: magic, cube dimension, predictor count, group count, selector byte length
(all uint32LE); compact rANS selectors for4³ blocks;15 uint16LE thresholds;
16×group-count streams with logical count, encoded length, compact rANS bytes.
Groups are shared1 or predictor-count8/9. Activities still depend only on the
causal residuals, not reconstructed values. Shared activity extraction and
payload serialization/parsing are reused rather than copied. No production
registration/default change. `benchmarks/ng_activity_refine_v1` records the
CT L0/L1, NX/width L3/L4 comparison and exact-roundtrip validation.

## Direct/Delta + difficulty block experiment

`ng_activity_blocks_encode/decode` adds direct+7 Delta choices without fitted
NG prediction. B=dimension selects smallest actual whole-chunk NAC1 frame;
B4/8/16 additionally tries spatial selector frames NAB1 with two cost-refinement
rounds and actual-size fallback. Neighborhoods cross block boundaries. NAB1:
magic,dimension,B,K,count,selector-byte-size (uint32LE), compact rANS selector,
then the same thresholds/class-stream payload as NAC1. Activity payload parsing
and writing are shared with the prior ablation. Experimental, no registration
or default change. Results and per-core speed: `benchmarks/ng_activity_blocks_v1`.

## New activity-only ablation (2026-10-25)

`ng_activity_only_encode/decode` (NAC1) isolates existing NG causal difficulty
classes and class-specialized rANS from all predictor fitting. Cubes32..256
in32 increments, uint8, fixed Delta mask0..7, K1 or16. Only class thresholds
are fitted; no per-voxel class labels. Uses the shared NG activity helpers and
existing compact rANS table/Delta-filter implementations. No default change.
Raw activity16 compresses especially well at L4, but the scalar callback-based
prototype decoder is only111–115MiB/s/core. Full matched results are in
`benchmarks/ng_activity_only_v1`. The approved runner currently reproduces this
ablation and its tests, superseding the baseline-only recipe described below.

## Current recommendation (2026-10-25)

Use the established4³ NG/Delta selector, rANS selector coding and shared native
residual streams. Keep the byte-preserving stack-local rANS table and shared
32/64/128/256 kernel specializations (generic fallback for other valid shapes).

Joint NX/NY semantic/lifting/CABAC formats and their speed experiments below
are **archived**, not defaults or current recommendations. They missed the
150MiB/s **per-core** target. Historical “best” labels describe compression-only
or experimental comparisons, not the recommended codec. Source/results remain
for reproducibility; no new joint-format development is queued.
See `experiments/compression_log.md` at the repository root for the final decision.
The approved no-argument runner now validates native leaves only.

## Extended semantic-rANS trials

`ng_lifting_trial_decode` additionally accepts experimental trials14–22:
14 stages one row's significance, width, sign and remainder streams before
assembling symbols;15 uses exact AVX2 masked byte gathers for the selected
predictor;16 combines them;17 bypasses slot-table construction and state updates
for a fully validated constant-symbol rANS stream;18 reuses decoded stream
storage per thread;19 combines14/17/18. All preserve existing LFP4 frames.
The predictor has an exact scalar fallback; no new instruction requirement.

20/21/22 are explicit **format-changing experiments**, not production codecs.
`ng_lifting_simple_transcode` decodes a mode78 LFP4 frame and retains its model,
block selectors, traversal, activity classes and residual symbols, changing
only entropy grammar. LFP5 adds a little-endian grammar ID after the order:
1 uses four significance streams plus eight nonzero-byte streams;2 uses four
interleaved pair-byte streams;3 uses four joint-tag streams plus twelve
width-specific remainder streams. Each channel's four-bit tag represents zero,
signed widths1–7, or the unique -128 residual; two tags form one byte. These use
the shared static rANS coder and count
all metadata. Only the corresponding trial decoder accepts LFP5. No production
registry/default is changed. Transcoding is benchmark preparation, excluded
from decode timing. The native and original semantic baselines remain present.

Output storage reuse retains at most the latest per-thread stream capacities;
it is an opt-in experiment, not an unbounded decoded-data cache. Constant-stream
decode validates the frequency table, exact frame length and every terminal
rANS state; it does not skip corruption checks.

The standalone CMake build mechanically adapts the pinned upstream decoder's
`slotEntry` declaration from thread-local to a16KiB stack array. GCC's PIC build
was calling `__tls_get_addr` inside the table-fill loop and decoding accesses.
The table is rebuilt completely on every call, so retaining its storage across
calls provides no data reuse. Stack storage preserves exact math, validation,
thread safety and format. This is the same upstream implementation, not a new
decoder: the build-generated `delta_adapter/codec.cpp` changes only that one
declaration, is shared by both bridge targets, and leaves the external checkout
untouched. Configuration fails if the expected declaration disappears.

Final matrix: `volume-cartographer/benchmarks/ng_rans_optimization_v6/`.
For the current experimental joint rANS candidate use LFP5 grammar3 / trial22:
55.01/62.75MiB/s at41.2307/35.6916% raw on S3/S4. Existing LFP4 uses trial13
(48.13/55.58 with the shared stack-table fix). Bulk semantic/SIMD predictor
variants are retained solely as reproducible failed trials. The native control
is155.07/158.63MiB/s; the150MiB/s joint goal has not been reached.

### Same-frame preparation/fusion experiments

Trials23–30 all decode the identical LFP5 grammar3 bytes:
23 tag descriptor LUT;24 four-X-block predictor/rotation preparation;25 both;
26 fully fused scalar rows;27 prevalidated remainder streams;28 paired Delta
arithmetic in biased16-bit integer lanes;29 SIMD-preserving assembly fusion;
30 additionally computes activity-error bytes in the SIMD reconstruction kernel.

The paired predictor uses the same extracted C++ Delta stencil as the scalar
predictor. Both byte values fit independently in16-bit lanes with a32768 bias;
no carry/borrow crosses lanes. This is exact integer arithmetic, not precision
reduction. Original C callers retain the identical scalar stencil.

Trial29 reserves each independent row's remainder slices from its tag counts,
then assembles residuals directly in the prediction loop, without the full
intermediate symbol buffer. Tag reads and slice reservations are bounds checked;
remainder value ranges are validated once, and final per-row/end-of-stream
consumption is checked. Trial27/28 instead validate global counts beforehand.
Tests also re-encode malformed remainder streams with valid rANS payloads, so
the optimized paths must reject bad logical counts/ranges independently of
entropy integrity. No safety checks are merely disabled.

Profiles label fused work explicitly: trial26's reconstruction column contains
the whole scalar loop;29/30's prediction column includes residual assembly.
Compare combined stages, not individual columns as though the work vanished.

Measured in `ng_rans_optimization_v9`: optional trial25 reaches56.60/63.46MiB/s
vs55.00/62.76 for22, only2.9%/1.1% better. All sizes unchanged. Trials26–30 do
not improve throughput; retained as explicit failed controls, not selected by
default.307 regressions and every timed real decode pass exactly. Prediction
and assembly remain substantial; no claim of a major speedup or150MiB/s.

## Joint NX/NY decoder acceleration (2026-10-24)

Experimental LFP3 modes61/78/79 now share a fast raster decoder for32/64/128/256
cubes. All other modes and32-increment shapes retain the generic path. The
encoded bytes, fitted classes, probability updates and traversal are unchanged.

- These three modes select only Delta-family predictors, so the fast decoder
  skips unused NG prediction. `ng_causal_state.h` exposes the existing native
  row activity/padding kernels to both runtime and fixed-geometry builds.
  The internal row API must not be mixed with incremental per-voxel calls on
  the same state. Class kernels retain AVX2/NEON/scalar dispatch.
- Joint decoding specializes dimensions/modes, fuses the flip correction into
  reconstruction, skips exact identity work for zero residual pairs, and caches
  finite normalized direction lookups. Delta inclusion/exclusion is shared
  through `delta_predict_xyz`, with exact bounded integer arithmetic.
- Adaptive binary decoding uses the identical interval split directly. A
  corrected reciprocal computes exact integer quotients on compilers with
  128-bit integer support; others retain division. Encoder intervals/model
  updates are unchanged. The grammar has separately instantiated encode/decode
  paths, removing dynamic per-bin dispatch without copying the grammar.

This is **not** the production decoder's multi-row wavefront schedule: the
existing joint entropy streams consume contexts in raster order. Reordering
them would require additional indexing or a changed stream traversal/format.

Use the approved no-argument `local/scripts/run_compression_experiment` from
the repository root. It builds Release, runs regressions, then compares the
frozen pre-optimization library and current library on identical real frames.
The runner requires the retained `/tmp/paris4-codec-hybrid/libjoint_before_speed.so`;
do not replace it with the new library. Reports are under
`volume-cartographer/benchmarks/ng_joint_decode_speed_v2` through `v6`.
The final comparison includes the established NG/Delta codec on the same inputs.
No data download, environment installation or production registry changes.

Regression coverage includes exact generic/specialized arithmetic intervals,
all binary probability totals2..32768 at varied ranges, old/new decoder
compatibility, unchanged joint encoder bytes on locked real chunks, and the
existing codec suite. Linux x86_64 measured; macOS/arm64 not measured. Sampled
instrumentation is separate from throughput and includes timer overhead.

Local fork of PR #2232's `volcodec_ng`, pinned at
`e1770003ba0534d02459e09cb5764087141d8dbc`. The four vendored files
`volcodec_ng_lib.c`, `volcodec_ng_lib.h`, `volcodec_ng_lossless.h`, and `LICENSE`
were initially unchanged from that revision (MIT, Chase Hendrick). The CABAC
experiment now adds entropy-backend handoff hooks to the C implementation/header;
the original NG numerical behavior and payloads are preserved. The later selector
experiment refactors reconstruction kernels to share native SIMD with Delta.
New whole-chunk selection and
dispatch live in `hybrid.{h,cpp}`; no predictor implementation is duplicated.
Delta3D is linked from the existing v0.2.0 source checkout, commit
`103b502ad784df14c876100d87f601c0b6c46888`, not copied or modified.

## Selection and format

`volcodec_ng_hybrid_encode` supports four policies:

- 0: original NG only (including its own constant/raw choices).
- 1: smallest of NG and Delta3D.
- 2: smallest of NG, Delta3D and LPC. Upstream LPC already calculates the
  Delta3D candidate, so it is not recomputed separately.
- 3: smallest of NG/Delta rANS and eight CABAC predictor/context candidates.
  See [CABAC experiment](CABAC.md) for context definitions and wire format.

Selection uses actual complete encoded byte lengths, not an entropy proxy.
Ties prefer NG. The optional `winner` output reports the predictor actually
stored. The NG raw bound remains sufficient: adding choices cannot increase
the size. Every chunk is a scalar lossless uint8 128³ volume; neither 64³ nor
multi-channel chunks are introduced by this experiment.

The payload retains its original `VCNG` or `D3D1` magic (or `NCAB` for CABAC). Hybrid decode dispatches
by magic, without an additional header, copy, or selector. It also reads existing
NG chunks. **Original NG-only decoders cannot read the added D3D1 payloads.**
Use the new codec identity `volcodec_ng_hybrid`, not `volcodec_ng`, in Zarr
metadata. The benchmark registers this identity with Numcodecs; this is not
yet installed or enabled in VC3D's production codec registry. Original NG
entrypoints remain unchanged for callers that require its original format.

## Build and test

From the villa4 root, using already available dependencies (no network/install):

```sh
cmake -S volume-cartographer/libs/volcodec_ng_hybrid -B /tmp/paris4-codec-hybrid -DCMAKE_BUILD_TYPE=Release -DVC_DELTA3D_SOURCE=/tmp/paris4-codec-delta3d-v020
cmake --build /tmp/paris4-codec-hybrid -j 8
/home/hendrik/.venv_las/bin/python -m pytest -q volume-cartographer/scripts/test_hybrid_codec.py volume-cartographer/scripts/test_prediction_codecs.py volume-cartographer/scripts/test_paris4_codec_ladder.py
```

Tests cover byte-for-byte minimum selection, original-format compatibility,
constant/noisy/ramp/real CT chunks, invalid inputs and existing native bindings.
Release GCC 11.4, no fast-math or architecture-specific build flags. The vendored
NG code retains its runtime SIMD dispatch. Linux amd64 tested; macOS/arm64 not
tested here. `VC_DELTA3D_SOURCE` must point to the stated upstream version.

## Recorded benchmarks

- [Fiber/Lasagna](../../benchmarks/predictions_ng_hybrid_pr2232/README.md)
- [CT ladder](../../benchmarks/paris4_ng_hybrid_pr2232/table.md)

These are experimental alternatives, not a change to production defaults.

Completed validation: 11 tests passed; 710 matched prediction measurements
(48 actual one-channel CZYX Zarr roundtrips) and 1,840 CT codec trials over all
368 saved chunks. Every hybrid size equals its exact per-chunk minimum and
every native decode is byte-identical. For that rANS-only comparison the
recommended policy was **1 (NG + Delta)**. Policy 2 adds no prediction-data savings and
only 110 bytes of CT savings, while substantially increasing encoder work.

The subsequent CABAC experiment adds policy 3 for a size-focused comparison.
It improves the Fiber pool from 21.9272% to 17.5126%, and Lasagna from
36.6273% to 34.9780%; decode speed has not been evaluated. See
[CABAC results](../../benchmarks/predictions_ng_cabac_pr2232/README.md).

## Integrated selector presets: native-optimized implementation

The separate `ng_delta_experiment.cpp` experimental interface implements fixed
NG with a per-class NG/Delta choice. Fast/Balanced/Best modes 22/23/24 use
native wavefront rANS, transposes and reconstruction (AVX2/NEON/scalar), with
vectorized Delta setup. Their NDC2 frame identifies wavefront entropy ordering;
the experimental decoder also reads earlier raster NDC1 frames. These modes are
not yet wired into the whole-chunk hybrid policies or production Zarr registry.

Encoding caches the seven Delta candidates and bias-free NG prediction, shares
native class/symbol kernels, caches entropy logarithms and uses candidate-local
four-way histogram counters. Quality presets still perform different amounts of
class/threshold refinement; they share the same decoder. Parameters and selector
decisions match old modes 12/13/14 exactly. Changing entropy traversal can change
a few final rANS bytes, not the decoded values.

Run the approved no-argument `local/scripts/run_compression_experiment` for the
Release build, 25 regression tests, full 361-input correctness/size comparison,
phase profiles and serial timing. See
[protocol and results](../../benchmarks/ng_wavefront_v3/README.md).

## Spatial block selectors

Modes 34/38/46 choose NG or one of seven Delta predictors independently on
4³/8³/16³ spatial blocks. Activity classes still select entropy contexts and
NG biases; they no longer determine the predictor. Selectors are raw 3-bit
packed; NDB1 identifies this experimental frame. Selection uses initial-class
chunk-wide smoothed histogram costs, followed by one causal regeneration.
Block boundaries do not reset prediction. Native SIMD kernels are shared with
the class selector; no duplicate reconstruction implementation is introduced.
See [experiment protocol/results](../../benchmarks/ng_spatial_blocks_v2/README.md).

NDB2 extends the experiment with optional rANS selectors and direction-separated
residual groups (retaining activity-class substreams). Compact frequency tables
reuse native NG's table writer/reader, with existing Delta3D rANS for the stream
payload. Shared-class controls isolate entropy framing from direction grouping.
The native wavefront reconstruction also accepts decoded substream symbols;
this experimental callback path is slower than fused native rANS decoding.
See [entropy experiment](../../benchmarks/ng_block_entropy_v2/README.md).

NJB1 adds 4³-block choices for ordinary NG, pure Delta and Delta plus learned
NG correction. Modes900–908 cross three NG parameter-sharing layouts with
three residual-rANS layouts independently;910–912 are matched NG/Delta-only
controls. Shared models are fitted from provisional baseline block assignments,
then all15 predictor candidates are rescored. Additional parameters and entropy
tables are fully charged. The causal predictor traversal/fitter are shared with
the earlier correction experiments; this research path is not a SIMD in-row
joint decoder. [Protocol/results](../../benchmarks/ng_joint_sharing_v1/README.md).

## Current baseline speed improvements

Mode **504** remains the recommended experimental baseline: 4³ NG/Delta blocks,
compact rANS selectors, native shared residual rANS. The joint-model sharing
matrix above did not provide a worthwhile compression gain.

The baseline now skips redundant NG/Delta prediction on spans wholly assigned
to the other predictor, uses four-way histogram counters and block-row scoring,
and avoids unused fitting allocations and symbol/class copies in entropy-only
encoding. Cached NG predictions are moved rather than copied. Delta candidate
storage is reused per worker thread (14 MiB at128³;1.75 MiB at64³), released at
thread exit. No shared mutable workspace or locking is introduced.

The format, choices and complete encoded bytes are unchanged. Direct fused
candidate generation and explicit SIMD gather scoring were tested and removed
because they did not improve overall speed. The scalar/AVX2/NEON decoder shares
the existing native kernels; no fast-math, precision or entropy-order change.

Run the same approved no-argument `local/scripts/run_compression_experiment`.
[Full protocol, raw trials and final results](../../benchmarks/ng_baseline_speed/README.md).
[Progress/decisions](../../../experiments/task.md). These changes are still
experimental, not a production codec registry change.

## Decoder-oriented alternative (NDF1)

Modes1000–1005 freeze504's predictor choices/taps but signal a block entropy
context instead of reconstructing causal voxel activity. They omit causal class
bias and remain lossless. Direct, predecoded native SIMD and fused native paths
share existing predictor/entropy kernels. These are separate experimental
formats, not defaults: the best measured gain is only3–4%, with worse total
compression. [Full comparison](../../benchmarks/ng_fast_decode_v3/README.md).

## Rectangular independent leaves

`ng_leaf.h` exposes baseline504 for uint8 X-fastest arrays with dimensions
divisible by32,32<=X<=1024, Y,Z>=32 and Y*Z<=65536. No boundary padding.
NDL1 stores X/Y/Z ahead of the unchanged NDB2 payload; leaves are independently
decodable.64³,128³ and256³ use constant-size instantiations of the shared kernels
(`ng_fixed.c`, compiled three times); other shapes use the same source with
runtime geometry. This dispatch does not change encoded bytes or format.
The indexed multichannel tree is now implemented as NGC1 in standalone
Delta3D; see [wire format and range APIs](../../INDEXED_FORMAT.md).
Zarr/storage byte-range fetching remains the caller's responsibility.
[Format, limits, API and validation](../../../experiments/ng_delta_container_format.md).

## Wavefront/CT semantic experiments

`ng_lifting_order_encode` adds an experimental `LFP4` frame for native NG
wavefront row ordering (order1), with unchanged fitted models, block choices,
per-voxel contexts and reversible NX/NY transform. Order0 retains the exact
existing frame. Both encode and causal decode reuse `ng_wave_rows`; this is
not yet a SIMD joint reconstruction optimization.

Scalar `NSC1` experiments use the same semantic binary grammar with one
coordinate: significance, width, sign and remainder. Native NG/Delta4 supplies
unchanged fitted parameters, selectors, residual symbols and activity classes
through `ng_block_semantic_prepare`, and performs actual reconstruction through
`ng_block_stream_decode`. No normal rotation or flip logic applies to CT.
These are experimental complete frames, not production format/default changes.

`ng_lifting_wave_decode` experiments with actual multi-row mode78 reconstruction:
`ng_wave_groups` shares the native schedule (up to32 independent rows), existing
SIMD activity kernels classify each row, semantic residuals are assembled for
the group, then X positions are interleaved across rows during reconstruction.
The ordinary decoder remains a row-at-a-time control on identical LFP4 bytes.
This is a scalar interleaved reconstruction prototype, NOT a SIMD implementation
of the joint predictors and inverse lifting. No production dispatch changes.
`benchmark_ng_wave_speed.py` times full decodes with native/raster controls,
one warmup and10 measured repeats, all byte-exact. See benchmark output v1/v2.

## Optional semantic rANS speed trials

`ng_lifting_trial_decode` keeps the current decoder as a control. Trials1/2/3
test temporary unpack-buffer reuse, direct semantic stream pointers plus width
lookup, and their combination. Trials10/11/12 use a staged wavefront pipeline:
native activity, semantic assembly, scalar predictor preparation with cached
block reference, then scalar/AVX2/AVX2+both earlier changes respectively.
The AVX2 kernel is `joint_reconstruct_simd.hpp`; it handles exact modular
lifting, conditional flip adjustment, residual addition and inverse lifting
for eight independent lanes. No floating point, approximate math or reduced
precision. GCC/Clang x86 runtime AVX2 dispatch; shared scalar fallback otherwise
and for tails. Predictors themselves remain scalar in this trial.

Optional timing output separates setup/unpack, activity, semantic assembly,
predictor/reference preparation, inverse lifting and stores/error updates.
Instrumentation runs separately from throughput and includes clock overhead.
`ng_joint_simd_audit` compares320k synthetic lanes against the scalar kernel;
tests also cover random/constant/seam data and real cached chunks. No new format.

Confirmation: variant13 (AVX2 + semantic pointers, without scratch reuse) is
the recommended experimental wavefront path on AVX2 machines:44.89/52.27MiB/s
on L3/L4 versus raster37.90/45.16. It decodes identical LFP4 bytes. Scratch reuse
did not help; scalar staged processing remains slower. Native codec/default
dispatch is unchanged. Non-AVX2 fallback is exact but not benchmarked for speed.
See `benchmarks/ng_rans_optimization_v2/summary.md` relative to volume-cartographer.

## Structural decode experiments (2026-10-25)

Historical recipe used the approved no-argument runner to build Release,
validate roundtrips and measure locked16 NX/NY pairs. The current runner no
longer launches these archived experiments. Their benchmark scripts and reports
remain available for an explicitly requested reproduction.

* LFP5 grammar4 keeps the grammar3 joint tags, but stores each remainder stream
  in its known1..6 bits, LSB-first with zero tail padding. Trial25 accepts this
  explicit grammar; it expands packed tails before the shared reconstruction.
* JSC1 (`joint_subchunks.cpp`) wraps independent native or joint leaves. Header:
  magic, cube dimension, leaf dimension, leaf count (uint32 LE), followed by
  `kind, first_bytes, second_bytes` for every leaf. Payloads follow in Z/Y/X
  leaf order. Kind0 is two native leaves, kind1 one joint frame. All sizes and
  shapes are checked. The benchmark includes the uncompressed directory,
  output assembly, allocations and thread creation/join. This is an experimental
  uniform grid, not the proposed general tree/Zarr container.
* Mode82 is a deliberate contiguous-X SIMD tradeoff: only Y/Z/YZ Delta choices
  and Y/Z rotation references, but the same fitted NG activity,4³ selection,
  modular lifting, conditional flip and joint-tag rANS. It does not pretend to
  preserve the compression or predictions of mode78. AVX2 has a scalar fallback;
  `ng_lifting_contiguous_scalar_decode` validates the same frames without SIMD.

The independent32³ test and contiguous-X test are separate: this is not a new
8³ transposed microblock codec. Hybrid thresholds select by actual total leaf
bytes, not by an estimated entropy. Results are in `benchmarks/ng_structural_v1`
(generic32), `benchmarks/ng_structural_v2` (shared-kernel32 specialization),
and `benchmarks/ng_contiguous_v1`; scalar and multiworker speeds are distinguished.
## Native-fit difficulty-input experiment

`ng_activity_refined_encode` flag1024 (with16) extends the original NG fit
with one activity input. Flag2048 is an audited disabled-input control: exact
native coefficients/predictions and original NAE1 frame bytes. Optional12
retains iterative entropy-threshold refinement. No default format changes.

The shared `vcn_lc_fit_feature` is the native fit, with an optional extra column;
the existing `vcn_lc_fit_target` calls it without that column. Complete tap
candidates, finalized padding, integer Gram statistics, stride128, selection
budget40, ridge and quantization remain native. Activity is rounded after
division by10 (0..255), not mapped to classes. Initial training features use
the ordinary single-fit residuals. Final causal encode/decode activity uses
the newly fitted predictor's residuals, independent of block selector choices.

NAF2 model5 stores all native weights, bias and activity coefficient. The
experimental wrapper still carries the native parameter header as well;
reported sizes include that redundant metadata. Mode6 is an internal audit
only and emits the original frame. This trial does not replace the archived
separate-refit models1–4 or promote new defaults. Reproduce with the approved
no-argument `local/scripts/run_compression_experiment`.

### Complete native-fit variants (V2)

Flags4096/8192 (each requiring16, mutually exclusive with other conditional
models) add NAF2 model7/8. Model7 includes all native neighbor taps, their
rounded products with activity/255, and activity as candidate features for the
same40-coefficient native selection. Fit-only arrays support321 features; the
native physical tap capacity and existing bitstream remain unchanged. Model8
runs the same native fit twice with integer sample weights255-activity/activity,
selecting40 coefficients per bank. The integer weighted Gram matrix is divided
by total weight before the ordinary covariance/ridge/solve/quantization steps.
Insufficient or empty banks copy the successful bank, or use native W fallback
if neither fits. Predictions blend the unwrapped Q8 sums, with explicit signed
rounding; only then are they wrapped. Encoding and decoding use the model's own
causal residual activity. Fitting activity is initialized from native single-fit
residuals, as in model5; no repeated refit is performed.

Optional flags12 refine rANS thresholds. Both complete coefficient banks and
the existing native parameter header are counted in sizes. The comparison
retains the scalar model and exact disabled-feature audit, and additionally
tests the model decoder directly (so Delta fallback cannot conceal a broken
model path). Results: `benchmarks/ng_activity_native_difficulty_v2`.
# Incumbent-preserving NG clamp trial

`ng_lorenzo_encode` option256 retains the previous flags13 clamp/Delta/direct
winner as a complete-frame fallback and imports its block choices and entropy
thresholds if the trial palette can express them. Its residual payload is
asserted byte-identical before any search updates. Legacy options remain for
matched historical controls. This is an experimental API, not a default change.

Option128 adds an NG clamp toggle to each4³ NG block. On disk the selector still
names NG and a separate packed bit (one per NG block, padded to a byte) selects
clamping. No flag is stored for other predictors. The rounded modulo-byte NG
prediction, after native class correction, is clamped to the minimum/maximum
available left/up/back samples. Q8 fractional prediction is preserved when
unchanged; a changed prediction is the exact integer bound. At the origin there
is no clamp. Encoder and decoder use the same helper. This does not change the
fitter or residual activity calculation. Option1 independently adds standalone
clamped Lorenzo; option64 adds direct values. Options16/32 select the earlier
native Lorenzo-aware fits. NAL2 stores the clamp feature flag and packed bits.

The no-Lorenzo ablation cannot import incumbent Lorenzo blocks: it retains the
incumbent only as a whole-frame fallback. Always inspect pre-fallback trial
sizes as well as final sizes. Search is heuristic, not a global optimum.
Experimental statistics now require15 uint64 entries (three size/status fields
and12 candidate counts). Run via `local/scripts/run_compression_experiment`
without arguments. Locked matrix: `benchmarks/ng_clamp_seed_v1/`.

## Complete native packed-bank entry points

`ng_packed_bank.h` exposes native encode/decode and phase profiling. Mode 1 is
the directional extra-input NG baseline with 4³ block choices, omitted unused
models, sparse model differences + Zstd3, and joint Zstd3 rANS table storage.
Both wrappers retain their raw-storage fallback when compression is larger.
There is no Python implementation in this encode/decode path. Link libzstd.
Use the approved no-argument `local/scripts/run_compression_experiment` runner
for regression tests and the locked 48-input Release benchmark.

### Bank wavefront reconstruction experiment

`ng_bank_reconstruct.h` shares the implementation across runtime geometry and
32/64/128/256 cube specializations. It reuses NG's dependency scheduler and row
padding, evaluates selected four-value previous-row prediction spans, and
uses one shared activity calculation per row. A lane is one independent row;
the x dimension remains causal. Lane-specific coefficients, clamped/unclamped
Delta bases and residual sign reversal retain exact modulo-16-bit arithmetic.
Preparation and vectorized reconstruction operate on four x positions across
the independent rows at a time, keeping lane metadata in L1. AVX2 dispatch is
runtime-guarded; portable C remains available. ARM/macOS performance is not measured.
No bitstream change, extra threads or Python codec operations are introduced.

`ng_bank_set_scalar(1)` selects the retained reference raster implementation
on the calling thread (default); `0` selects experimental wavefront. The
regression harness compares both against the original input. Benchmark results
and unsuccessful variants are recorded in `benchmarks/ng_bank_decode_speed_v1/`.
Wavefront is NOT the default: the tiled trial measured 29.95 MiB/s on CT L1
versus 49.04 MiB/s for the matched raster path. SIMD arithmetic is present,
but the complete reconstruction stage is slower. No compression changes.

### Full-choice shared NG fit experiment (compression only)

Packed encode modes 2/3 are the shared-fit counterparts of correction/input
modes 0/1. All 24 choices stay available, including the plain NG choice and
each of the 11 NG/base combinations. The main/first-slice parameter sets,
difficulty thresholds and 16 class biases are shared across NG choices; mode 3
also shares the two base-feature coefficients. Fixed Delta/direct choices and
the 16 selected-residual entropy classes are unchanged.

Start with the original plain-NG fit, search choices, then fit one common NG
model against the selected spatially varying base and search again, twice.
Use the native existing fitter and choice/threshold machinery throughout.
Retain the best raw-bank trial across these rounds; never fall back to separate
weights. Packing counts model/header/selector/residual/table bytes. NPK1 codec
4 optionally stores the shared model once (Zstd3); existing sparse-difference
and raw representations remain available if smaller. Codec 4 is experimental
and requires the updated decoder. Existing mode 0/1 framing is unchanged.

`ng_packed_bank_inspect` reports model-header size, active model IDs, distinct
parameter-set count and all 24 block-choice counts. The compression benchmark
asserts at most one distinct parameter set in shared outputs and exact voxel
roundtrips. Run the approved no-argument experiment runner; results go to
`benchmarks/ng_shared_fit_128_v1/`. No speed claims or default changes.
