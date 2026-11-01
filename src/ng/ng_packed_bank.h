#ifndef NG_PACKED_BANK_H
#define NG_PACKED_BANK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Public speed policies. Preserve numeric experimental IDs for reproducibility. */
enum { NG_CODEC_FAST = 3, NG_CODEC_SLOW = 1, NG_CODEC_DEFAULT = NG_CODEC_FAST };
/* Fixed selectors default to bit-packed IDs inside joint Zstd metadata.
 * Marker254 is packed,255 is legacy rANS,0..59 are uniform. Older decoders
 * reject254; current decoder retains legacy support. Adaptive NCA1 unchanged. */
void ng_codec_packed_selectors(unsigned enabled);
/* Native sampled evidence: five edges x seven doubles; no fits/encodes. */
int ng_codec_size_features(const uint8_t*,unsigned,unsigned,unsigned,unsigned,double*);
/* Incremental paired evidence. larger=4/8/16/32; each observation compares
 * exactly that cube to its eight children, distributed across octants.
 * Source memory must outlive handle. Extend count is cumulative. Output:
 * ten means then10x10 covariance of mean; no sample fit or encoding. */
void* ng_pair_size_create(const uint8_t*,unsigned,unsigned,unsigned,unsigned);
/* Experimental fit-aware evidence; native sample-only NG fits, no payload
 * encoding. Separate entry point; never enabled implicitly by production. */
void* ng_pair_size_create_fit(const uint8_t*,unsigned,unsigned,unsigned,unsigned);
int ng_pair_size_extend(void*,unsigned,double*);
void ng_pair_size_destroy(void*);
/* Unified NCF1 codec. preset0 (legacy): screened uniform choice;
 * preset1: screened three choices, selection + one reselection, refined entropy.
 * preset2 (experimental): preset0 + residual-only iterative entropy thresholds.
 * preset3 (fast/default): fixed predictor + DP64 residual thresholds.
 * Experimental thresholds: 4/5 global DP on128/256 quantile grids;
 * 6 DP128 + fine local refinement; 7 exact-frame best of0/2/6.
 * No iterative model refits. Both use native wavefront decoding.
 * Same capacities and cube geometry as the experimental APIs below. */
int ng_codec_encode(const uint8_t*,unsigned,unsigned,uint8_t*,size_t,size_t*);
int ng_codec_decode(const uint8_t*,size_t,uint8_t*,size_t);
/* Opt-in choice-grid experiments; default grid remains4. Fixed grids accept
 * 2/4/8/16/32 and write the size into NCF1. encode_grid accepts Fast/Slow only:
 * policy0 screens the fixed size;8/16/32 use adaptive NCA1 roots with2³ leaves;
 * add256 for a2-bit rather than8-bit estimated tree-node cost. Decoding uses
 * ng_codec_decode; older decoders reject these experimental extensions. */
int ng_codec_choice_edge(unsigned);
int ng_codec_encode_grid(const uint8_t*,unsigned,unsigned,unsigned,uint8_t*,size_t,size_t*);
/* encode_grid policy512: old proxy/encoder, only metadata-aware size ranking. */
/* Five proxy-frame sizes, six leaf counts (2,4,8,16,32,64), chosen fixed edge.
 * A uniform frame omits the tree and reports zero leaf counts. Thread-local. */
void ng_grid_experiment_stats(double*,uint64_t*,unsigned*);
/* Independent contiguous samples, unchanged production encoders. side32 uses
 * eight octant centers; side64 two diagonal centers. Twenty doubles: per size
 * scaled total,payload,selector contribution,unscaled remaining metadata. */
int ng_codec_screen_fixed_samples(const uint8_t*,unsigned,unsigned,unsigned,double*);
/* Shared full-coordinate cost experiment. preset Fast/Slow; policy4 fixes4³,
 * policy0 screens2–32,8/16/32 select adaptive maximum root sizes. Fits retain
 * canonical4³ sampling. Same NCF1/NCA1 decoders, no production default change. */
int ng_codec_encode_shared_grid(const uint8_t*,unsigned,unsigned,unsigned,uint8_t*,size_t,size_t*);
void ng_shared_grid_stats(double*,unsigned*);
/* Six u64: total, prefix+compressed metadata, residual payload, raw metadata,
 * raw model bytes, model count. Read-only diagnostic. */
int ng_codec_inspect_sizes(const uint8_t*,size_t,uint64_t*);
/* Zero-block experiment control: bit0 enables the ordinary zero selector;
 * bit1 preserves unmasked full fits (recommended when enabling bit0).
 * Screening excludes empty blocks whenever bit0 is set. No extra side map. */
int ng_codec_encode_zero_control(const uint8_t*,unsigned,unsigned,unsigned,uint8_t*,size_t,size_t*);
/* Experimental Fast+DP64 config: r0,r1,r2,NG bias classes,tap budget,fit stride;
 * entropy classes are independent (2..32), one entropy bank only. */
int ng_codec_encode_fast_config(const uint8_t*,unsigned,const int*,unsigned,uint8_t*,size_t,size_t*);
/* Same hyperparameter array as fast_config, but baseline Slow policy:
 * three independent entropy banks, two selections, zero refits, greedy64. */
int ng_codec_encode_slow_config(const uint8_t*,unsigned,const int*,unsigned,uint8_t*,size_t,size_t*);
/* Three-choice threshold experiment: selections1..16, refits<selections;
 * grid0=greedy,64/256=DP; feedback0=final-only,1=between selections too.
 * First `refits` reselections update NG fits on their selected blocks. */
int ng_codec_encode_three_thresholds(const uint8_t*,unsigned,unsigned,unsigned,unsigned,unsigned,uint8_t*,size_t,size_t*);
/* NMB1 -> NCF1, retaining fitted models, block choices and thresholds. */
int ng_codec_repack_model_bank(const uint8_t*,size_t,uint8_t*,size_t,size_t*);
/* Experimental uint8 cube codec, dimensions in multiples of 32.
 * Mode 0: fitted correction; mode 1: separate fitted base input.
 * Modes 2/3 retain ALL choices but share the correction/extra-input fit.
 * Mode 3 is the recommended compression baseline. Modes 2/3 write NWP1 by
 * default; experimental nonshared modes 0/1 retain their legacy format.
 * Caller owns buffers. Returns zero on success; negative on failure.
 * Encode capacity: 3*d*d*d + 131072 bytes. Decode capacity: d*d*d bytes.
 * All compression, metadata and framing operations are native. */
int ng_packed_bank_encode(const uint8_t*, unsigned, unsigned, uint8_t*, size_t, size_t*);
/* Optional bounded search, same lossless NWP1 format and predictor palette.
 * Effort1: initial plain-NG fit only, one seed/pass. Effort2/3: one/two shared
 * feature refits, two passes per round. Compression differs; NOT the default. */
int ng_packed_bank_fast_encode(const uint8_t*, unsigned dimension, unsigned effort,
                              uint8_t*, size_t, size_t*);
/* Per-call encoder-only search controls; unchanged NWP1 decoder/format.
 * seed: 0 cheap histogram, 1 plain NG, 2 clamped XYZ, 3 direct,
 *       4 existing comprehensive initialization; 5 NG+clamped XYZ,
 *       6 NG+XYZ, 7 NG+X (nonzero-input seeds for the first refit).
 *       8 bounded fixed shortlist (NG,X,XYZ,clamped XYZ), 9 causal neighbor
 *       shortlist (NG,up to2distinct left/up/back choices,one seeded random).
 *       10 fills to exactly4 with uniform random proposals; 11 fills to4 with
 *       25%uniform +75%past mean-positive-bit-saving weighted proposals.
 *       12 weighted second sweep/frozen costs, 13 second sweep/refreshed costs,
 *       14 weighted then shared NG refit only, 15 refit then refreshed second
 *       sweep. Later sweeps retain incumbents and proposal history. These keep
 *       the same13 predictor choices, not the comprehensive24-choice palette.
 *       16 expands to24 choices after refit;17 adds another expanded sweep.
 *       18 fits all12 NG models independently, then one24-choice sparse sweep.
 *       Seeds8..18 require rounds=passes=1 and thresholds=1 or2; score using
 *       initial NG costs, rebuild selected symbols once, encode only once.
 * rounds: 1..6 total fits; passes: 1..6 block-choice evaluations per fit.
 * thresholds: 0 existing full trials, 1 histogram-selected refined candidate,
 *             2 quantiles only. Re-selection considers all24 block choices;
 *             passes=1 leaves the initial block selection unchanged.
 * Work is option-dependent: unchanged initial NG reuses prepared symbols and
 * activity; fixed NG skips palette/index generation; a sole estimated/quantile
 * candidate goes directly to final wavefront encoding (no trial payload).
 * These are experimental controls, not accepted default channel profiles. */
typedef struct { unsigned seed,rounds,passes,thresholds; } ng_bank_search_options;
int ng_packed_bank_search_encode(const uint8_t*,unsigned,const ng_bank_search_options*,uint8_t*,size_t,size_t*);
/* Experimental NFA1 wrapper; mode1 XYZ Lorenzo, mode2 W/N/B mean context.
 * Same independent predictor bank (seed18). Decoded by ng_packed_bank_decode. */
int ng_packed_bank_fixed_activity_encode(const uint8_t*,unsigned,unsigned,const ng_bank_search_options*,uint8_t*,size_t,size_t*);
/* Fit/calibrate on samples; evaluate only proposed blocks and final choices.
 * models=2 (plain+fixed XYZ extra-input) or12, sample stride16..8192. NWP1. */
int ng_packed_bank_sampled_encode(const uint8_t*,unsigned,unsigned,unsigned,const ng_bank_search_options*,uint8_t*,size_t,size_t*);
/* Matched plain+XYZ-input menu, original full-volume model preparation.
 * stride128 is the original regression grid; stride1 samples every eligible
 * interior x (first-plane fit still uses the original stride2). */
int ng_packed_bank_dense_two_encode(const uint8_t*,unsigned,unsigned,const ng_bank_search_options*,uint8_t*,size_t,size_t*);
/* Experimental metadata-only repacking: method0 raw Zstd,1 sparse fields,
 * 2 best sparse reference deltas,3 best including unchanged frame; level1..19.
 * NWM1 records transformed metadata, with byte-identical residual payload. */
int ng_packed_bank_metadata_repack(const uint8_t*,size_t,unsigned,int,uint8_t*,size_t,size_t*);
int ng_packed_bank_metadata_unwrap(const uint8_t*,size_t,uint8_t*,size_t,size_t*);
/* Experiment-only diagnostics. mode bit0 always proposes input-X in large
 * menus; bit1 fits the shared extra input against X instead of chosen bases.
 * get:24 trial counts,24 win counts,24 sums of estimated cost minus plain NG.
 * reset/end are thread-local; callers must end before ordinary encoding. */
void ng_bank_diagnostic_reset(unsigned mode);
void ng_bank_diagnostic_get(double*);
void ng_bank_diagnostic_end(void);
/* Optional thread-local encoder phase profiling. Enabling resets counters;
 * get writes eight doubles: fixed candidates, fitting, seeds, symbols, entropy
 * trials, threshold refinement, reselection, initial block selection.
 * Times are seconds. */
void ng_bank_encode_profile_enable(int enabled);
void ng_bank_encode_profile_get(double* eight_seconds);
/* Native fit subphases: padded source, regression, prediction/activity,
 * class calibration, biased symbols. Seconds; inclusive in fitting above. */
void ng_bank_fit_profile_get(double* five_seconds);
/* Independent seed18:0 uncached,1 shared statistics/scalar dot products,
 * 2 shared statistics/runtime-gated SIMD. Scoped per thread, default2. */
void ng_bank_fit_acceleration(unsigned mode);
/* Inclusive work counters (also nested searches), fourteen doubles:
 * rANS calls/bytes/seconds, activity payload calls/bytes/seconds,
 * reselection calls/candidate bytes, bank passes/candidate bytes,
 * lower-bound checks/rejections, final framing seconds, random proposals.
 * Final framing consumes selected symbols (no predictor computation).
 * Disabled outside profile calls. */
void ng_bank_encode_work_get(double* fourteen_values);
/* Thread-local diagnostic controls. Bound pruning defaults ON and is exact.
 * Policy0 is the unchanged full-search control. Policy1 (recommended default)
 * keeps only the prior winner after refits; policy2 keeps winner+clamp seed.
 * Policy1's worst measured size regression is0.0793% across160 corpus inputs;
 * this is NOT a mathematical guarantee of a per-input size budget. */
void ng_bank_encode_bound_enable(int enabled);
void ng_bank_encode_search_policy(unsigned policy);
/* Experimental policies3/4/5: cheap joint initialization; estimated/adaptive
 * checkpoints; coarse spatial seeds. None passes the0.1% pilot budget on all
 * tested inputs. Default remains1. */
/* Separate experimental NMB1 model-bank format, one-bank matched control or
 * two to four fitted models selectable per4^3 block. Scalar experimental decoder;
 * production ng_packed_bank_decode does not accept NMB1. Same buffer bounds. */
int ng_small_bank_encode(const uint8_t*,unsigned dimension,unsigned banks,uint8_t*,size_t,size_t*);
int ng_small_bank_decode(const uint8_t*,size_t,uint8_t*,size_t);
/* Fast residual-energy initialized model families, free model/base choice.
 * NFS1 uses shared wavefront entropy/SIMD decode, not legacy raster NMB1.
 * banks1..2, rounds1..3, sparse sweeps1..3 per fit. */
int ng_fast_split_encode(const uint8_t*,unsigned,unsigned,unsigned,unsigned,uint8_t*,size_t,size_t*);
/* Same solver with legacy histogram base-choice initialization. */
int ng_fast_split_hist_encode(const uint8_t*,unsigned,unsigned,unsigned,unsigned,uint8_t*,size_t,size_t*);
int ng_fast_split_full_encode(const uint8_t*,unsigned,unsigned,unsigned,unsigned,uint8_t*,size_t,size_t*);
int ng_fast_split_decode(const uint8_t*,size_t,uint8_t*,size_t);
int ng_fast_split_inspect(const uint8_t*,size_t,uint64_t*); /* 40 counters */
int ng_fast_split_seeded_encode(const uint8_t*,unsigned,unsigned,unsigned,uint8_t*,size_t,size_t*);
void ng_fast_split_search_trace(double*); /* 96 entries: screen costs, seeds, checkpoints */
/* Incremental split/refit search, 1..maximum (<=4) model budgets in one run.
 * Caller supplies maximum output buffers, sizes/seconds[maximum] and
 * stats[maximum*4] (cumulative evaluation/refit/split/merge counts).
 * Seconds are cumulative. Take the smallest returned frame for up-to-N. */
int ng_incremental_bank_encode(const uint8_t*,unsigned,unsigned,uint8_t*const*,size_t,size_t*,double*,unsigned*);
/* Explicit legacy encoder for reproducibility/older decoder compatibility. */
int ng_packed_bank_legacy_encode(const uint8_t*, unsigned, unsigned, uint8_t*, size_t, size_t*);
int ng_packed_bank_decode(const uint8_t*, size_t, uint8_t*, size_t);
/* Optional size/speed tradeoff, NOT the default encoder. Encodes both mode3
 * NWP1 and ordinary NG; selects NG only if its size including the 8-byte NWP0
 * wrapper is within permille (0..100) of NWP1. 5 means 0.5%, 10 means 1%.
 * Same buffer capacities as packed_bank. Decoder accepts NWP0 and NWP1.
 * The extra encoder search is intentionally explicit. */
int ng_speed_bank_encode(const uint8_t*, unsigned dimension, unsigned permille,
                         uint8_t*, size_t, size_t*);
int ng_speed_bank_decode(const uint8_t*, size_t, uint8_t*, size_t);
/* Five phase durations in seconds: table/model unpack, setup, entropy, prediction. */
int ng_packed_bank_profile(const uint8_t*, size_t, uint8_t*, size_t, double*);
int ng_packed_bank_repack(const uint8_t*, size_t, uint8_t*, size_t, size_t*);
/* Default shared-fit NWP1: retain fitted models/choices, repack entropy with
 * original NG wavefront rANS and joint compressed headers. Decode through the
 * ordinary packed-bank decoder. Delta-only frames use a zero-weight history
 * model and keep the original Delta/direct choices. */
int ng_packed_bank_wave_repack(const uint8_t*, size_t, uint8_t*, size_t, size_t*);
/* Native full encoder, same capacity and modes 2/3 as above. */
int ng_packed_bank_wave_encode(const uint8_t*, unsigned, unsigned, uint8_t*, size_t, size_t*);
/* Per-thread diagnostic: 0 experimental SIMD wavefront, 1 reference raster,
 * 2 raster with identical-model preparation reuse; 3 shared wavefront;
 * 4 raster with branch-free activity gathering; 5 Q8 mask trial;
 * 6 block specialization trial; 7 full-row SIMD; 8 skewed field planes;
 * 9 vector metadata preparation; 10 compact shared-model SIMD;
 * 14 cached work masks/direct transpose (default for legacy frames).
 * NWP1 defaults to integrated wavefront SIMD entropy/reconstruction with
 * non-aliasing residual/history updates and lightweight Delta-only handling.
 * It selects the exact NG+X shortcut when >=25% of block choices use NG+X.
 * It also uses four metadata planes, exact combined class lookup and a
 * uniform clampedXYZ shortcut. Control41 selects this previous combination;
 * control57 additionally fuses activity into lookup, interleaves lane groups,
 * and uses a local residual handoff. Controls47/49/51 isolate
 * these additions;55 combines lookup and interleaving without local handoff.
 * control27 retains the prior automatic NG+X baseline,21 its generic kernel.
 * Default14/109 additionally use runtime-gated wide prediction, activity and
 * recurrence, native predicates, unrolled entropy, borrowed compressed payload
 * and shared parsed model state. No format or numerical changes. Control111
 * exercises scalar entropy. Unsupported ISAs retain narrower/scalar paths;
 * nonshared fits retain raster decoding. */
void ng_bank_set_scalar(int);
/* 28 uint64 outputs: model-header bytes, active IDs, unique parameter sets,
 * choice alphabet size, followed by 24 block-choice counts. */
int ng_packed_bank_inspect(const uint8_t*, size_t, uint64_t*);
#ifdef __cplusplus
}
#endif
#endif
