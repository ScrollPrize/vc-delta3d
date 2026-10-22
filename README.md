# VC-Delta3D

VC-Delta3D compresses three-dimensional `uint8`, `uint16`, `int8` and `int16`
scalar fields, and multi-channel chunks whose channels are such fields.
It combines optional bounded scalar quantization, a choice of two predictors
(an axis-adaptive delta filter or a per-chunk linear predictor), and order-0
rANS entropy coding.

Encoded chunks use the self-describing `D3D1` wire format and the Zarr
compressor identifier `vc-delta3d`.

Since 0.2.0, multi-channel chunks (a channel axis first or last) are stored as
a format-version-2 container of one ordinary payload per channel, so each
channel's spatial field is coded on its own. Signed integers are coded as
their bytes (the delta filter wraps modulo 2^8/2^16), so they must stay
lossless (`quant=1`). Decoders before 0.2.0 cannot read version-2 containers;
single-field payloads are unchanged in both directions.

## Compression methods

Each chunk is predicted from its already decoded voxels, and the prediction
residuals are entropy-coded with order-0 rANS. The predictor is chosen when
encoding and recorded in every chunk, so decoding needs no configuration.

- **Delta** (default, `predictor="delta"`, C++ `vc_delta3d::Predictor::Delta`):
  a Lorenzo delta filter. Each voxel is coded as its difference from its
  neighbours along the z, y and x axes; for 8-bit fields the encoder picks the
  subset of axes per chunk. Works for every supported type. The faster method.
- **LPC** (`predictor="lpc"`, C++ `vc_delta3d::Predictor::Lpc`): a per-chunk
  least-squares linear predictor. The encoder picks 40 of the 62 causal
  neighbours in the 5x5x5 box and fits their weights. The method that
  compresses best. LPC can be used for uint8 data in chunks of at least 16^3
  voxels; other chunks are stored with the delta filter. Introduced in version
  0.2.0.

Either method can be preceded by quantization (`quant`, unsigned types only):
a bin width of 2k+1 bounds the per-voxel error by +-k, and 1 is lossless. In
multi-channel chunks the method applies to every channel. In zarr metadata the
codec reads e.g. `{"id": "vc-delta3d", "quant": 1, "predictor": "lpc"}`; the
predictor key is omitted for delta.

## C++

```cpp
#include <vc_delta3d/codec.hpp>

auto encoded = vc_delta3d::compress(bytes, {z, y, x}, element_size);
auto decoded = vc_delta3d::decompress(encoded, bytes.size());

// Best compression (uint8):
auto smaller = vc_delta3d::compress(bytes, {z, y, x}, 1, vc_delta3d::kQuantizationLossless,
                                    vc_delta3d::Predictor::Lpc);

// Multi-channel, e.g. a (z, y, x, c) chunk of embeddings:
auto packed = vc_delta3d::compressChannels(bytes, {z, y, x}, c,
                                           vc_delta3d::ChannelLayout::Last, element_size);
auto restored = vc_delta3d::decompress(packed, bytes.size());
```

CMake consumers can link `vc_delta3d::vc_delta3d`.

## Python

```python
from vc_delta3d import Delta3D

codec = Delta3D(quant=1)
encoded = codec.encode(array)
decoded = codec.decode(encoded)

# Best compression (uint8):
codec = Delta3D(predictor="lpc")

# 4D chunks with a channel axis ("first" or "last"); int8/int16 must stay lossless.
codec = Delta3D(channel_axis="last")
```

On real 64- and 32-channel int8 embedding chunks (unit vectors with
per-vector scales), the channel container reaches 4.0x and 3.9x versus 1.1x for
Zstd on the interleaved bytes.

Importing `vc_delta3d` registers `vc-delta3d` with numcodecs.

## Build

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ctest --test-dir build --output-on-failure
```

## Performance

### Size

Full PHercMANBp volume (2.399um 78keV masked, 17148x12577x12577 uint8, 128^3
chunks, lossless):

| | level 0 | all 6 levels |
|---|---|---|
| uncompressed | 85.60 GB | 102.49 GB |
| delta filter | 27.12 GB (3.16x) | 33.22 GB (3.09x) |
| LPC | 22.11 GB (3.87x) | 27.54 GB (3.72x) |

Uncompressed sizes are the stored chunks of the published volume; the mask
leaves chunks outside the scroll unstored.

On 2.4um level-0 chunks (63 from PHercParis4, 14 from PHerc0841), the delta
filter compresses 2.72x and 2.93x, LPC 3.43x and 3.91x.

Delta filter on the whole of
`s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/5`:
2.30 GiB raw to 471.70 MiB, 5.00x (19.99% of raw).

### Speed

Single core, MB/s of uint8 voxels, 63 PHercParis4 2.4um level-0 chunks (128^3),
on AWS `.2xlarge` instances (GCC 12) and one laptop:

| instance | CPU | decode delta | decode LPC | encode delta | encode LPC |
|---|---|---|---|---|---|
| c6g | Graviton2 | 408 | 151 | 193 | 46 |
| c7g | Graviton3 | 575 | 253 | 288 | 73 |
| c8g | Graviton4 | 730 | 299 | 344 | 90 |
| c9g | Graviton5 | 943 | 392 | 441 | 114 |
| c6a | Zen3 | 524 | 289 | 210 | 70 |
| c7a | Zen4 | 556 | 318 | 223 | 75 |
| c5 | Cascade Lake | 395 | 212 | 180 | 53 |
| c6i | Ice Lake | 442 | 247 | 186 | 60 |
| c7i | Sapphire Rapids | 556 | 334 | 252 | 82 |
| c8i | Granite Rapids | 588 | 344 | 281 | 89 |
| laptop | Ryzen 7 4800H (Zen2), fixed 2.9 GHz | 415 | 215 | 175 | 52 |

x86 decoding uses AVX2 when the CPU has it (selected at run time) and SSE2
otherwise (LPC decoding about 1.4x slower than with AVX2). Encoded bytes are
identical on every platform.
