# VC-Delta3D

VC-Delta3D compresses three-dimensional `uint8` and `uint16` scalar fields.
It combines optional bounded scalar quantization, an axis-adaptive Lorenzo
delta transform, and order-0 rANS entropy coding.

Encoded chunks use the self-describing `D3D1` wire format and the Zarr
compressor identifier `vc-delta3d`.

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

codec = Delta3D(quant=1)
encoded = codec.encode(array)
decoded = codec.decode(encoded)
```

Importing `vc_delta3d` registers `vc-delta3d` with numcodecs.

## Build

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ctest --test-dir build --output-on-failure
```
