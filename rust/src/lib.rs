//! Raw FFI bindings to [VC-Delta3D](https://github.com/ScrollPrize/vc-delta3d),
//! a codec for three-dimensional (optionally multi-channel) 8- and 16-bit
//! integer fields. The C sources are bundled and built by this crate.
//!
//! The declarations mirror `include/vc_delta3d/codec.h` and
//! `include/vc_delta3d/indexed.h`. Every function returns 1 on success and 0
//! on invalid input or allocation failure, unless noted otherwise.
//!
//! A failed decode may already have written to the output buffer: treat its
//! contents as undefined unless the call returned 1.
#![no_std]
#![allow(non_camel_case_types)]

use core::ffi::{c_int, c_uint, c_void};

/// Encoder preset (`vc_preset` in C). A preset only steers the encoder;
/// every frame decodes without it. From fastest to smallest: delta, faster,
/// fast, balanced, small, slower. Medium and slow are the menu-search presets.
pub type vc_preset = c_uint;
pub const VC_DELTA: vc_preset = 1;
pub const VC_FAST: vc_preset = 3;
pub const VC_MEDIUM: vc_preset = 4;
pub const VC_SLOW: vc_preset = 5;
pub const VC_SLOWER: vc_preset = 6;
pub const VC_FASTER: vc_preset = 8;
pub const VC_BALANCED: vc_preset = 9;
pub const VC_SMALL: vc_preset = 10;

/// Whether this version encodes with `preset`.
pub const fn vc_preset_valid(preset: c_uint) -> bool {
    matches!(
        preset,
        VC_DELTA | VC_FAST | VC_MEDIUM | VC_SLOW | VC_SLOWER | VC_FASTER | VC_BALANCED | VC_SMALL
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct vc_encode_options {
    /// Z, Y, X in elements.
    pub shape: [u32; 3],
    /// 0: scalar frame; >= 1: channel container.
    pub channels: u32,
    /// 0: channels first; 1: channels last.
    pub layout: c_uint,
    pub element_size: c_uint,
    pub quantization: c_uint,
    pub preset: vc_preset,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct vc_frame_info {
    pub decoded_bytes: usize,
    pub channels: u32,
    pub quantization: c_uint,
    /// -1 when not explicitly recorded.
    pub delta_mask: c_int,
    pub legacy_scalar: c_uint,
}

/// `codec` value of a leaf that picks its codec per block.
pub const VC_LEAF_CHOICE: c_uint = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct vc_indexed_leaf {
    pub channel: u32,
    pub origin: [u32; 3],
    pub shape: [u32; 3],
    pub codec: c_uint,
    pub offset: u64,
    pub length: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct vc_chunk_index {
    pub shape: [u32; 3],
    pub channels: u32,
    pub header_bytes: u32,
    pub layout: c_uint,
    pub encoded_bytes: u64,
    pub decoded_bytes: u64,
    /// Owned by the index; released by [`vc_index_free`].
    pub leaves: *mut vc_indexed_leaf,
    pub leaf_count: usize,
}

/// Preorder partition node: axes 0/1/2 spatial, 3 channel, 4 leaf; 5/6/7/8
/// repeat Z/Y/X/channel. See `indexed.h` for the cut semantics.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct vc_partition_node {
    pub axis: c_uint,
    pub cut: u32,
}

extern "C" {
    /// Default: fast, uint8, lossless, scalar. The caller supplies the shape.
    pub fn vc_default_options() -> vc_encode_options;
    /// On success `*out` holds `*out_len` bytes; release with [`vc_free`].
    pub fn vc_compress(
        src: *const c_void,
        src_len: usize,
        options: *const vc_encode_options,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int;
    pub fn vc_free(ptr: *mut c_void);
    pub fn vc_frame_inspect(frame: *const u8, frame_len: usize, info: *mut vc_frame_info) -> c_int;
    /// On failure (0) the contents of `dst` are undefined.
    pub fn vc_decompress(frame: *const u8, frame_len: usize, dst: *mut c_void, dst_len: usize) -> c_int;
    /// The alternate magic applies only to legacy D3D1 framing (e.g. `VCZ1`).
    pub fn vc_decompress_with_magic(
        frame: *const u8,
        frame_len: usize,
        magic: *const [u8; 4],
        dst: *mut c_void,
        dst_len: usize,
    ) -> c_int;
    pub fn vc_quantize(data: *mut c_void, len: usize, element_size: c_uint, width: c_uint) -> c_int;

    /// An empty plan (`nodes == 0`) selects the canonical channel/cubic
    /// partition. On success release `*out` with [`vc_free`].
    pub fn vc_indexed_encode(
        src: *const u8,
        src_len: usize,
        shape: *const [u32; 3],
        channels: u32,
        layout: c_uint,
        preset: c_uint,
        plan: *const vc_partition_node,
        nodes: usize,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int;
    /// Given the first >= 8 bytes, returns the full header size (0 on error).
    pub fn vc_indexed_header_size(data: *const u8, len: usize) -> u32;
    /// On success the index owns its leaves; release with [`vc_index_free`].
    pub fn vc_index_read(header: *const u8, header_len: usize, object_bytes: u64, index: *mut vc_chunk_index) -> c_int;
    pub fn vc_index_free(index: *mut vc_chunk_index);
    pub fn vc_indexed_leaf_decode(
        payload: *const u8,
        payload_len: usize,
        leaf: *const vc_indexed_leaf,
        dst: *mut u8,
        dst_len: usize,
    ) -> c_int;
    pub fn vc_indexed_decode(data: *const u8, len: usize, dst: *mut u8, dst_len: usize) -> c_int;
}
