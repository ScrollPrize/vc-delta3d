# vc-delta3d-sys

Raw Rust FFI bindings to [VC-Delta3D](https://github.com/ScrollPrize/vc-delta3d),
a fast codec for three-dimensional (optionally multi-channel) 8- and 16-bit
integer fields. The crate bundles the C sources and builds them with the `cc`
crate, so it needs a C compiler but neither CMake nor an installed library.

```toml
[dependencies]
vc-delta3d-sys = "0.3"
# or straight from the repository:
# vc-delta3d-sys = { git = "https://github.com/ScrollPrize/vc-delta3d" }
```

```rust
use vc_delta3d_sys::*;

fn compress(volume: &[u8], shape: [u32; 3]) -> Option<Vec<u8>> {
    unsafe {
        let mut options = vc_default_options(); // fast, uint8, lossless, scalar
        options.shape = shape;                  // Z, Y, X
        let (mut frame, mut len) = (core::ptr::null_mut(), 0);
        if vc_compress(volume.as_ptr().cast(), volume.len(), &options, &mut frame, &mut len) != 1 {
            return None;
        }
        let bytes = core::slice::from_raw_parts(frame, len).to_vec();
        vc_free(frame.cast());
        Some(bytes)
    }
}

fn decompress(frame: &[u8]) -> Option<Vec<u8>> {
    unsafe {
        let mut info = core::mem::zeroed::<vc_frame_info>();
        if vc_frame_inspect(frame.as_ptr(), frame.len(), &mut info) != 1 {
            return None;
        }
        let mut out = vec![0u8; info.decoded_bytes];
        (vc_decompress(frame.as_ptr(), frame.len(), out.as_mut_ptr().cast(), out.len()) == 1).then_some(out)
    }
}
```

The bindings mirror `include/vc_delta3d/codec.h` and
`include/vc_delta3d/indexed.h`; those headers and `FORMAT.md` are the
reference.

Every function returns 1 on success and 0 on failure. A failed decode may
already have written to the output buffer, so discard it unless the call
returned 1.

## Build notes

- The C code is always compiled at `-O3`, also in debug profiles.
- Windows (MSVC targets) needs `clang-cl` on `PATH`: the kernels use C99
  variable-length arrays, which MSVC's C compiler does not support.
- Dependent build scripts get the header directory as
  `DEP_VC_DELTA3D_INCLUDE`.

Licensed under MIT (`LICENSE`, and `src/ng/LICENSE` for the NG kernels).
