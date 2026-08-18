# VC-Delta3D

VC-Delta3D compresses three-dimensional `uint8` and `uint16` scalar fields.
It combines optional bounded scalar quantization, an axis-adaptive Lorenzo
delta transform, and either order-0 rANS or zstd entropy coding.

The project preserves the original `VCZ1` wire format used by Volume
Cartographer. Existing payloads and Zarr arrays with compressor ID `vcz1`
remain supported; new Zarr metadata can use `vc-delta3d`.

## C++

```cpp
#include <vc_delta3d/codec.hpp>

auto encoded = vc_delta3d::compress(bytes, {z, y, x}, element_size);
auto decoded = vc_delta3d::decompress(encoded, bytes.size());
```

CMake consumers can link `vc_delta3d::vc_delta3d`.

## Python

```python
from vc_delta3d import Delta3D

codec = Delta3D(codec="rans", quant=1)
encoded = codec.encode(array)
decoded = codec.decode(encoded)
```

Importing `vc_delta3d` registers both `vc-delta3d` and the legacy `vcz1`
identifier with numcodecs.

## Build

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ctest --test-dir build --output-on-failure
```
