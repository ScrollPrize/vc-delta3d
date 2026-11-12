# VC-Delta3D

A native C volume codec with six presets, from fastest to smallest: **delta**,
**faster**, **fast** (default), **balanced**, **small** and **slower**; the
**medium** and **slow** presets of 0.3.x are kept as well. It combines Chase Hendrick's NG prediction with
Sean Johnson's Delta3D directional predictors. Both MIT notices are retained;
see [AUTHORS.md](AUTHORS.md).

The codec, fitting, SIMD kernels, entropy coding, framing and indexed containers
are C. The C++ headers are ownership/error adapters over the C API. Python uses
a thin nanobind adapter; it does not perform prediction or compression in Python.

## Performance

Scroll CT, 95 masked level-0 128³ chunks from five scrolls, one core
(full tables for 10 EC2 instance types in [BENCHMARKS.md](BENCHMARKS.md)):

|Preset|Size (% raw)|Encode MiB/s (c7i / c8g)|Decode MiB/s (c7i / c8g)|
|---|---:|---:|---:|
|delta|37.0|240 / 279|512 / 703|
|faster|27.0|64.7 / 46.7|306 / 117|
|fast|26.8|64.0 / 43.6|291 / 109|
|balanced|26.7|47.6 / 27.2|279 / 102|
|small|26.6|11.8 / 8.17|217 / 79.8|
|slower|26.7|1.86 / 0.86|237 / 75.7|
|medium|27.0|16.2 / 13.0|262 / 104|
|slow|26.9|2.57 / 1.36|250 / 93.4|

c7i: Intel Sapphire Rapids; c8g: Graviton4. `faster` is the `fast` preset of
0.3.x; `medium` and `slow` are kept from 0.3.x for some derived volumes. How
much the smaller presets gain depends on the data; see
[BENCHMARKS.md](BENCHMARKS.md). [COMPRESSION.md](COMPRESSION.md) explains what
the presets do.

## Formats and compatibility

- Lossless uint8 chunks with dimensions divisible by 32 use the preset encoder.
  A single cubic chunk is one **NCF1** frame.
- Multiple channels and spatial partitions use the **NGI1** index around NCF1
  leaves, so single channels or regions can be read by byte range.
- `predictor="delta"`, uint16/int16 data, quantization, or dimensions not
  divisible by 32 use the original Delta3D **D3D1** format. D3D1 data from all
  earlier releases, including alternate magics such as VCZ1, stays readable.
- Decoding reads everything from the encoded bytes; Zarr metadata needs no
  per-chunk fields.

[COMPRESSION.md](COMPRESSION.md) explains how the presets and Delta3D
compress; [FORMAT.md](FORMAT.md) specifies all three formats and how they relate.

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

Importing the Python package registers `vc-delta3d` with numcodecs, for Zarr v2
arrays (as `compressor`). `predictor` names the encoder preset (`delta`,
`faster`, `fast`, `balanced`, `small`, `slower`, and `medium` and `slow` from
0.3.x). It only affects encoding: arrays written with any preset open and
decode with any setting.

## Zarr v3 and sharding

Install with `pip install 'vc-delta3d[zarr]'`. In Zarr v3, `vc-delta3d` is a
bytes-to-bytes codec after `bytes`, normally inside `sharding_indexed`:

```python
import zarr
from zarr.codecs import BytesCodec
from vc_delta3d.zarr3 import Delta3DCodec

array = zarr.create_array(
    "volume.zarr", shape=(4096, 4096, 4096), dtype="uint8",
    shards=(1024, 1024, 1024), chunks=(128, 128, 128),
    serializer=BytesCodec(endian="little"),
    compressors=Delta3DCodec(predictor="fast"),
)
```

This writes the codec chain `[bytes, vc-delta3d]` that the neuroglancer and
VC3D readers expect. zarr-python finds the codec through a `zarr.codecs` entry
point, so reading needs no import. The configuration (`predictor`, `quant`,
`channel_axis`) only affects encoding. 16-bit payloads hold little-endian
elements, so keep `endian="little"`.

Pick inner chunks that are cubic, divisible by 32 and at most 256 on a side
(e.g. 64³ or 128³). Each inner chunk is then one NCF1 frame, and the shard index
lets a reader fetch just the inner chunks it needs. Larger inner chunks are
coded as NGI1, which zarr reads only as a whole. Chunk shapes not divisible by 32
fall back to D3D1, and creating such an array warns.

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
See [benchmark/README.md](benchmark/README.md) for the benchmark harness.
The C-port validation record (`C_PORT_RESULTS.md`) and its one-off scripts
are in git history (commit 93469eb).

## Releasing

Publishing a GitHub release uploads wheels (Linux x86_64/aarch64, macOS arm64,
Windows AMD64) and the sdist to [PyPI](https://pypi.org/p/vc-delta3d) via
`.github/workflows/release.yml` (trusted publishing, `pypi` environment).

1. Set the version in **both** `pyproject.toml` (`version`) and
   `CMakeLists.txt` (`project(... VERSION ...)`), merge to `main`, and check
   that the wheel builds on `main` are green.
2. Create the release from `main` with tag `vX.Y.Z` matching that version,
   e.g. `gh release create v0.3.0 --target main --title v0.3.0 --notes-file notes.md`
   (or *Draft a new release* on GitHub). Note format and decoding
   compatibility changes in the release notes. `--target` only applies when
   the tag does not exist yet: if an earlier attempt already created the tag,
   check that it points at the intended commit
   (`git ls-remote --tags origin vX.Y.Z`) before publishing.
3. The *Publish to PyPI* job runs once the build jobs pass; check the Actions run
   and https://pypi.org/p/vc-delta3d.
