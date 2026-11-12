# How VC-Delta3D compresses

This document explains the compression methods: what the encoder predicts, how
it codes the residuals, and what the presets `fast`, `medium`, `slow`,
`slower` and `delta` do differently. The byte layout is in
[FORMAT.md](FORMAT.md); measured sizes and speeds are in
[BENCHMARKS.md](BENCHMARKS.md).

All methods are lossless (quantization in `delta` is an explicit, optional
pre-step) and work the same way at the top level:

1. **Predict** each voxel from voxels that the decoder has already decoded.
2. **Map** the prediction error to a small unsigned symbol.
3. **Entropy-code** the symbols with rANS, using statistics that the decoder
   can reproduce.

The presets differ in how good the prediction is and how much the encoder
searches to find it. Decoding never searches: it reads the chosen
configuration from the frame and replays it.

## The preset codec (NCF1: fast, medium, slow, slower)

Used for uint8 chunks whose dimensions are multiples of 32. It combines Chase
Hendrick's NG context modelling with Delta3D's directional predictors.

### Predictors

Every voxel `v` is predicted from causal voxels, that is, voxels the decoder
has already reconstructed. There are three kinds of predictor.

**Fixed predictors (IDs 0–11).** Closed-form predictions from the seven
nearest causal neighbours: `a` (x − 1), `b` (y − 1), `c` (z − 1) and the
diagonals between them (`ab`, `ac`, `bc`, `abc`):

| ID | Prediction | ID | Prediction |
|---|---|---|---|
| 0 | 0 (store the value itself) | 6 | `b + c − bc` (YZ plane) |
| 1 | `a` (previous in X) | 7 | 3-D Lorenzo: `a+b+c − ab − ac − bc + abc` |
| 2 | `b` (previous in Y) | 8 | 3, clamped to `[min(a,b), max(a,b)]` |
| 3 | `a + b − ab` (XY plane) | 9 | 5, clamped to `[min(a,c), max(a,c)]` |
| 4 | `c` (previous in Z) | 10 | 6, clamped to `[min(b,c), max(b,c)]` |
| 5 | `a + c − ac` (XZ plane) | 11 | 7, clamped to the range of `a, b, c` |

The clamped forms (8–11) behave like the median edge detector of LOCO-I: they
avoid overshooting at edges.

**Learned linear predictors (models).** A weighted sum over a much larger
causal neighbourhood: up to 3 voxels back in the current row, 7 voxels in each
of the three rows above, a 7×7 window in the previous slice, and a 3×3 window
two slices back (82 candidate voxels; `slower` uses 7×7 there too, 122
candidates). The encoder fits one set of weights per chunk by least squares,
keeps the 40 most useful taps (`slower`: all), and rounds the weights to
multiples of 1/256. The neighbourhood is independent of the 4³ blocks
described below and reads across their boundaries. One fixed predictor's
output is an extra input with its own learned weight, so a model refines a
fixed predictor rather than replacing it. The first slice, which has no
previous slice, gets its own weight set. The weights travel in the frame
header.

**Zero predictor (ID 12).** For blocks that are entirely zero (for example
masked regions); they cost almost nothing.

### Per-block choice

The volume is divided into 4³ blocks. A frame carries a small *menu* of up to
60 predictor/model combinations, and every block stores which menu entry it
uses. This selector map is compressed on its own (constant, context-coded or
LZ), so a uniform choice costs a few bytes.

### Residuals

The prediction error `v − p` is taken modulo 256 and folded into a symbol:
small positive and negative errors become small symbols (0, −1, 1, −2, … →
0, 1, 2, 3, …).

### Context modelling and entropy coding

The symbols are coded with rANS, but not with one global distribution:

- **Activity:** for every voxel the encoder and decoder compute a weighted sum
  of the residual magnitudes of already decoded neighbours. High activity means
  the local prediction is poor, so larger symbols are likely.
- **Classes:** thresholds split the activity range into classes (16 in the
  current presets). Each class has its own 256-entry frequency table and, for
  models, a small bias correction. The encoder chooses the thresholds to
  minimize the coded size.
- **Banks:** different menu entries can use separate sets of class tables
  (*entropy banks*), so blocks that are predicted differently also get their
  own statistics.
- 32 interleaved rANS states let the decoder rebuild many voxels in parallel;
  the SIMD decoder processes rows in a wavefront.

### What the presets do

| Preset | Encoder search |
|---|---|
| `fast` | Tries all 12 fixed predictors on a 32³ sample of the chunk, fits a model for the two best, and keeps the cheapest single option (fixed or learned) for all non-empty blocks. All-zero blocks use the zero predictor. |
| `medium` | Keeps up to three candidates (a fixed predictor, a learned model and a third screened candidate), each with its own entropy bank, and chooses per block; one reselection pass, and the best single candidate is tried too. |
| `slow` | Beam search over menus of one to three entries from 25 candidates (fixed predictors and up to 13 models) on a sample, then fits the finalists at full resolution on all voxels and keeps the smallest frame. |
| `slower` | Like `slow`, with a larger neighbourhood (more previous-slice taps) and up to 160 instead of 40 model taps. |

`fast` and `medium` fit their models on a subsample of voxels; `slow` and
`slower` fit on every voxel. How much the slower presets gain depends on the
data: on scroll CT they are within 1.5% of `fast`, on smoother derived volumes
up to about 5%, at about 4× (`medium`) to 40–60× (`slower`) the encode time of `fast`
(see [BENCHMARKS.md](BENCHMARKS.md)). Decoding does the
same kind of work for every preset, but menus with more entries and models
with more taps cost more per voxel. On x86, `medium` takes about 15–30% longer
to decode than `fast`, `slow` 25–50% and `slower` 40–60%; on ARM the gaps are
smaller. See [BENCHMARKS.md](BENCHMARKS.md) for measured numbers.

The encoder's choices are not part of the format: a future encoder can pick
other menus, models, class counts or block sizes (2–32) without changing the
decoder.

### Larger, rectangular and multi-channel chunks

Encoding works on cubic leaves of up to 256³. Larger or rectangular chunks are
split into cubic leaves, and channels are coded one at a time; the leaves are
stored in an NGI1 container. Each leaf is compressed independently, with its
own models and statistics, which is what makes byte-range reads of single
leaves possible.

## The original Delta3D codec (D3D1: delta)

Used for `predictor="delta"`, for uint16/int16 data, for quantization and for
dimensions not divisible by 32.

1. **Quantization (optional, lossy):** with `quant > 1`, every value is
   rounded to the nearest multiple of `quant` before coding. The decoder
   returns the rounded values.
2. **Delta filter:** the volume is replaced by differences along a chosen set
   of axes (X, Y, Z or any combination). Filtering along all three axes is the
   3-D Lorenzo predictor. For uint8 the encoder picks the combination whose
   result has the lowest entropy; uint16 always uses all three axes.
3. **Entropy coding:** one order-0 rANS stream with a single frequency table
   for the whole field and eight interleaved states.

There is no context modelling, so `delta` compresses less than the presets but
encodes and decodes very quickly. Multi-channel data stores one such stream
per channel.
