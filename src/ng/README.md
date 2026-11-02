# Retained NG/Delta kernels

Derived from Chase Hendrick's MIT-licensed NG implementation; see LICENSE.
Delta3D integration and directional prediction retain Sean Johnson's attribution
in the repository AUTHORS.md. The pre-reduction implementation is preserved on
the `feature/ng-presets` branch.

## Encoder policies

Public IDs are defined in `include/vc_delta3d/codec.h`:

| Preset | Policy |
|---|---|
| fast | Occupancy-aware screening; one selected predictor for nonzero blocks; DP64 entropy thresholds. |
| medium | Direct, NG and a screened third candidate with independent residual statistics; one reselection; compare the screened singleton. |
| slow | Broader independent-model menu search, selected-tap fits and dense regularized sufficient statistics. |
| slower | Same menu policy with expanded predictor neighborhood/coefficient budget. |

All emit NCF1 and use 4³ selector blocks. The decoder retains the frame's
2/4/8/16/32 selector sizes; block size is a frame parameter, not global metadata.
Zero blocks use the existing selector alphabet. Models and residual banks are
referenced separately; a choice may share either without duplicating it.

There is no streaming state, adaptive partition search, tree entropy coder,
LPC, tooslow, alternative fitting/refinement or user-selectable decoder trial.
Fixed-geometry instantiations share the SAME C kernels rather than copies.

## Source map

- `ng_fast.c`, `ng_medium.c`, `ng_slow.c`: retained search policies.
- `ng_fit.c`, `ng_screen.c`, `ng_search.c`, `ng_select.c`,
  `ng_thresholds.c`: shared fitting, occupancy, scoring and threshold primitives.
- `ng_predict.c`: directional candidates, including SIMD row implementations.
- `ng_frame.c`: NCF1 validation, model/entropy metadata and selector framing.
- `ng_rect.c`, `ng_stream.c`: rectangular NDB2 leaves and compact rANS streams.
- `volcodec_ng_lossless.h`: original arithmetic, tap fitting, scheduling and
  entropy/reconstruction kernels.
- `volcodec_ng_lib.c`: internal C entrypoints and scoped kernel state.
- `ng_fixed.c`: compile-time geometry specializations (32/64/128/256).
- `ng_bank_reconstruct.h` and vector headers: integrated wavefront decoder.

Public framing/dispatch lives in `../api.c`; NGC1 indexing is in `../indexed.c`.
Legacy Delta3D uses shared `../delta_filter.c`, `../rans.c`, `../legacy.c`.
The public C++ headers adapt ownership and errors; they contain no codec kernels.

## Invariants

- Keep exact integer residual/sign mapping, predictor rounding and rANS order.
- No fast math; floating-point contraction is disabled for fitted weights.
- Geometry and fitting configuration are thread-local and restored by callers.
- Runtime ISA dispatch retains portable/NEON fallbacks and AVX2 without AVX-512.
- Changes must pass the frozen 144-chunk × four-preset byte-parity harness.
  See repository `task.md` for commands and performance measurements.
