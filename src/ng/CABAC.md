# CABAC-style residual backend experiment

## Scope and provenance

No predictor replacement, no new predictor search, no width/data conversion
changes. Original NG fit, first-slice fit, thresholds, class biases, rounding
and sign flip are retained. Delta uses the existing axis-mask choice and
modulo differences. Each predictor produces exactly its original residuals.

The arithmetic engine (`adaptive_arithmetic.hpp`) was extracted unchanged from
`fiber-comp` at `eb7f68b65`, file
`volume-cartographer/python/vc/compression/vcz2_adaptive_arithmetic.inc`.
The earlier semantic zero/width/remainder/sign grammar and medium adaptation
(increment 4; halve counts above total 4096) are adapted to 128³ chunks.
This is CABAC-style adaptive binary arithmetic, not standardized H.264 CABAC.

NG's original implementation now offers a handoff immediately after its
predictor parameters and residual symbols are produced, before rANS tables.
Its decoder shares the original parameter parser, prediction/class kernels and
reconstruction kernel, but calls the new entropy decoder in raster order.
No classes are transmitted: the decoder derives them from decoded residuals.

The external Delta checkout is not edited or copied. The local CMake target
compiles its source once through `delta_bridge.cpp`, exposing the private
forward/inverse filters in the same translation unit. This avoids duplicating
the filter math; a future upstream public residual API would replace this bridge.

## Eight candidates per chunk

Each of NG and Delta is tested with these four context layouts:

| Variant | Zero/width/remainder context | Sign context |
|---|---|---|
| v0 | Global | Global |
| v1 | Native context | Native context |
| v2 | Native context | Native × neighbor sign agreement × current magnitude |
| v3 | Native × neighbor magnitude | That combined context × sign agreement × current magnitude |

NG native context is its existing **16-way activity class**, including its
causal magnitude padding rules. Delta has no original activity model, so its
native context is the four-way maximum residual-magnitude bucket of decoded
left/up/back neighbors. For v3 Delta this repeats the same spatial feature;
it has identical effective conditioning to v2 (the redundant table IDs transmit
no state, and serve as an identity check, not an independent gain).

Spatial magnitude buckets: 0, 1, 2–3, ≥4. Sign agreement: no nonzero neighbor,
positive only, negative only, mixed. Current magnitude buckets: 1, 2–3, 4–15,
≥16. NG sign refers to its original **rounding-flipped** residual sign; the
predictor inversion restores that sign exactly. Boundary neighbors are omitted
for these additional left/up/back contexts (NG's native activity padding remains
unchanged). All models reset per chunk and adapt during decoding. No transmitted
frequency tables, noncausal feature lookups, or predictor signals per voxel.

Residual grammar: zero/nonzero; unary magnitude bit width; remainder bits;
sign. Magnitude 128 is the unique signed-modulo-byte value -128 and needs no
remainder/sign bits. Small-symbol priors are fixed 3:1 for zero/width bins,
1:1 for remainder/sign bins.

## Stream and accounting

`NCAB` frame: 24-byte fixed header, predictor parameters, arithmetic body.
Header fields: magic, version 1, predictor ID, context ID, delta mask, uint32
parameter length, uint32 logical bit count, uint64 decoded-data FNV checksum.
All multibyte fields are little-endian. NG stores exactly its existing fitted
weights/biases/activity thresholds; Delta needs only the header mask. Full
headers, predictor parameters and termination bits count in reported sizes.

Hybrid policy 3 tries all eight candidates and the existing NG/Delta streams,
then writes the smallest complete stream. Unchanged rANS wins ties. Decoder
dispatches by magic; the per-chunk context ID is explicitly charged. Decoder
uses no reference source data. Every benchmark candidate is decoded and checked,
not just the winner. Arithmetic implementation/stream layout is experimental,
not production hardened. Decode speed is intentionally not a target yet.

## Reproduction

```sh
/home/hendrik/villa4/local/scripts/run_compression_experiment
```

No arguments, package installation, GPU use or input-cache mutation. Native
C++ evaluates eight candidates concurrently, reusing each predictor's fitted
residuals across contexts. Python handles input loading, experiment scheduling
and reports only. The existing prediction loader preserves the established
width conversion and masks. The saved CT ladder is also reused verbatim.

Results: `volume-cartographer/benchmarks/predictions_ng_cabac_pr2232/` and
`volume-cartographer/benchmarks/paris4_ng_cabac_pr2232/`. Context-only, per-channel,
per-scale and fallback-inclusive percentages are reported separately. No
compression sizes are estimated from entropy; all are actual stream lengths.

## Completed outcome

14 regression tests pass, including unchanged original rANS streams, real CT,
constants, random bytes, all eight context variants and complete hybrid
minimum selection. Full benchmarks verified 2,648 prediction and 2,944 CT
candidate payloads. All decode checks pass. Relative to NG+Delta rANS, the new
hybrid reduces Fiber from 21.9272% to 17.5126%, Lasagna from 36.6273% to
34.9780%, and CT from 39.9955% to 39.7553%. Production defaults are unchanged.
