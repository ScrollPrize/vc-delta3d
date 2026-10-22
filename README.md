# VC-Delta3D

VC-Delta3D compresses three-dimensional `uint8`, `uint16`, `int8` and `int16`
scalar fields, and multi-channel chunks whose channels are such fields.
It combines optional bounded scalar quantization, an axis-adaptive Lorenzo
delta transform, and order-0 rANS entropy coding.

Encoded chunks use the self-describing `D3D1` wire format and the Zarr
compressor identifier `vc-delta3d`.

Since 0.2.0, multi-channel chunks (a channel axis first or last) are stored as
a format-version-2 container of one ordinary payload per channel, so each
channel's spatial field is delta-coded on its own. Signed integers are coded as
their bytes (the delta filter wraps modulo 2^8/2^16), so they must stay
lossless (`quant=1`). Decoders before 0.2.0 cannot read version-2 containers;
single-field payloads are unchanged in both directions.

## C++

```cpp
#include <vc_delta3d/codec.hpp>

auto encoded = vc_delta3d::compress(bytes, {z, y, x}, element_size);
auto decoded = vc_delta3d::decompress(encoded, bytes.size());

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

# 4D chunks with a channel axis ("first" or "last"); int8/int16 must stay lossless.
codec = Delta3D(channel_axis="last")
```

On real 64- and 32-channel int8 embedding chunks (unit vectors with
per-vector scales), the channel container reaches 4.0x and 3.9x versus 1.1x for
Zstd on the interleaved bytes.

Importing `vc_delta3d` registers `vc-delta3d` with numcodecs.

### Linear prediction (LPC)

`Delta3D(predictor="lpc")` (C++: `vc_delta3d::Predictor::Lpc`) replaces the
delta filter for uint8 chunks of at least 16 voxels per axis with a per-chunk
least-squares linear predictor over the 62 causal neighbours in the 5x5x5 box.
On scroll CT it makes full-resolution chunks ~20-25% smaller, at roughly 4x
slower encode and decode. The encoder keeps the delta payload when it is
smaller, and decoders need no configuration: the predictor is recorded in
each chunk. In zarr metadata the codec reads
`{"id": "vc-delta3d", "quant": 1, "predictor": "lpc"}`; older releases cannot
decode LPC chunks.

## Build

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ctest --test-dir build --output-on-failure
```

## Performance
Measured on `s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/5` :   

```
start size:       2.30 GiB (2,474,382,596 bytes)
end size:         471.70 MiB (494,617,928 bytes)
compression ratio: 5.00x (19.99% of raw)
encode speed:     316.5 MiB/s
decode speed:     449.8 MiB/s
copy throughput:  85.5 MiB/s
```

Size is for the full array , enc/dec speed are calculated from 128 randomly selected nonzero chunks 
