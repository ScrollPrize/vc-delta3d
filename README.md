# VC-Delta3D

A native C volume codec with four presets: **fast** (default), **medium**,
**slow**, and **slower**. It combines Chase Hendrick's NG prediction with
Sean Johnson's Delta3D directional predictors. Both MIT notices are retained;
see [AUTHORS.md](AUTHORS.md).

The codec, fitting, SIMD kernels, entropy coding, framing and indexed containers
are C. The C++ headers are ownership/error adapters over the C API. Python uses
a thin nanobind adapter; it does not perform prediction or compression in Python.

[C-port validation and measurements](C_PORT_RESULTS.md) includes the complete
18-group before/after tables, compatibility checks and reproduction commands.

## Formats and compatibility

- Lossless uint8 chunks with dimensions divisible by 32 use the preset encoder.
  Scalar cubes emit **NCF1 directly**, without another envelope.
- Multiple channels and spatial partitions use **NGC2** compact or **NGX2**
  extended fixed-width indexes. Repeated subtrees share partition geometry.
  Leaves use NCF1 coding without redundant leaf magic. Eight full-volume
  channels cost46index bytes plus32leaf-prefix bytes. See `INDEXED_FORMAT.md`.
- `predictor="delta"` explicitly selects the original Delta3D algorithm and
  **D3D1** frame. Regular APIs also fall back to this complete legacy path for
  uint16/int16 data, quantization, or positive dimensions not divisible by 32.
  Legacy and preset frames are never nested inside each other's containers.
- Legacy Delta3D scalar/channel decoding and custom legacy magic (such as VCZ1)
  remain supported. Signed data must remain lossless.
- LPC, tooslow/HGT3, adaptive/streaming searches and experimental entropy routes
  are deliberately removed on this branch. Retired formats fail explicitly.
  The original implementation remains on `feature/ng-presets`.

Decoding reads its configuration from the frame, not from the encoder preset.
[INDEXED_FORMAT.md](INDEXED_FORMAT.md) documents shapes, partitions and independent
byte-range reads. Zarr metadata needs no additional per-chunk fields.

## C API

Include `vc_delta3d/codec.h` (and `indexed.h` for indexed leaf access).
Functions return 1 on success and 0 on invalid input or allocation failure.

```c
#include <vc_delta3d/codec.h>

/* input and output are caller-owned buffers containing 128^3 bytes. */
vc_encode_options options = vc_default_options();
options.shape[0] = options.shape[1] = options.shape[2] = 128;
uint8_t *encoded = NULL;
size_t encoded_size = 0;
if (vc_compress(input, 128u*128u*128u, &options,
                &encoded, &encoded_size)) {
    int ok = vc_decompress(encoded, encoded_size, output, 128u*128u*128u);
    /* Check ok before using output. */
    vc_free(encoded);
}
```

`vc_frame_inspect` reports the decoded size before allocation. The indexed C API
reads the index separately and decodes any leaf without accessing its siblings.
The library has no C++ runtime dependency.

## C++ and Python adapters

```cpp
#include <vc_delta3d/codec.hpp>

auto encoded = vc_delta3d::compress(bytes, {z, y, x}, element_size);
auto decoded = vc_delta3d::decompress(encoded, bytes.size());
auto packed = vc_delta3d::compressChannels(bytes, {z, y, x}, channels,
    vc_delta3d::ChannelLayout::Last, element_size);
```

```python
from vc_delta3d import Delta3D

codec = Delta3D(predictor="fast")
encoded = codec.encode(array)
decoded = codec.decode(encoded)
legacy = Delta3D(predictor="delta")
multichannel = Delta3D(channel_axis="last")
```

Importing the Python package registers `vc-delta3d` with numcodecs.

## Build and test

Requires a C compiler; the codec has no Zstd dependency.
The optional C++ adapters require C++20; Python additionally requires nanobind.
No dependency installation is performed by these commands.

C-only library, without a C++ compiler or downloaded test dependency:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVC_DELTA3D_BUILD_TESTS=OFF
cmake --build build -j
```

Native regression tests and C++ adapter smoke test:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVC_DELTA3D_BUILD_TESTS=OFF -DVC_DELTA3D_MODERN_SMOKE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
build/vc_delta3d_modern_smoke
```

CMake consumers link `vc_delta3d::vc_delta3d`. For Python, also configure
`VC_DELTA3D_BUILD_PYTHON=ON` and `Python_EXECUTABLE`.
`VC_DELTA3D_BUILD_TESTS=ON` enables the additional doctest suite and may fetch
its header dependency. Windows builds use ClangCL because the kernels use VLAs.
AVX2/AVX-512 are runtime-dispatched; NEON and portable implementations are retained.
Floating-point contraction remains disabled to preserve fitted weights.

## Measurements and development record

[BENCHMARKS.md](BENCHMARKS.md) contains the historical 18-group comparison
against original NG, Delta3D, LPC, Blosc and Zstd. Historical rows describing
removed presets are reference measurements, not supported options.
[task.md](task.md) records the C port, byte-parity checks, commands and remaining
validation work. All experimental data/build output belongs outside the checkout.
