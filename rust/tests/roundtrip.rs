use core::ffi::c_void;
use core::ptr;
use vc_delta3d_sys::*;

fn volume(dim: usize, channels: usize) -> Vec<u8> {
    let mut state = 12345u32;
    let mut out = Vec::with_capacity(dim * dim * dim * channels);
    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                for c in 0..channels {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let mut v = 2 * x + y + 3 * z / 2 + 7 * c + (state >> 29) as usize;
                    if z < 8 {
                        v = 200;
                    }
                    if x >= 40 && y >= 40 {
                        v = 0;
                    }
                    out.push(v as u8);
                }
            }
        }
    }
    out
}

fn compress(src: &[u8], options: &vc_encode_options) -> Vec<u8> {
    let mut frame = ptr::null_mut();
    let mut len = 0usize;
    unsafe {
        assert_eq!(vc_compress(src.as_ptr() as *const c_void, src.len(), options, &mut frame, &mut len), 1);
        let bytes = core::slice::from_raw_parts(frame, len).to_vec();
        vc_free(frame as *mut c_void);
        bytes
    }
}

#[test]
fn scalar_roundtrip_all_presets() {
    const D: usize = 64;
    let src = volume(D, 1);
    for preset in [VC_DELTA, VC_FASTER, VC_FAST, VC_BALANCED, VC_SMALL, VC_SLOWER, VC_MEDIUM, VC_SLOW] {
        assert!(vc_preset_valid(preset));
        let mut options = unsafe { vc_default_options() };
        assert_eq!((options.preset, options.element_size, options.channels), (VC_FAST, 1, 0));
        options.shape = [D as u32; 3];
        options.preset = preset;
        let frame = compress(&src, &options);
        assert!(frame.len() < src.len(), "preset {preset}");

        let mut info = vc_frame_info { decoded_bytes: 0, channels: 0, quantization: 0, delta_mask: 0, legacy_scalar: 0 };
        assert_eq!(unsafe { vc_frame_inspect(frame.as_ptr(), frame.len(), &mut info) }, 1);
        assert_eq!((info.decoded_bytes, info.channels), (src.len(), 1));

        let mut decoded = vec![0x5Au8; src.len()];
        unsafe {
            assert_eq!(vc_decompress(frame.as_ptr(), frame.len(), decoded.as_mut_ptr() as *mut c_void, decoded.len()), 1);
        }
        assert_eq!(decoded, src, "preset {preset}");
        // Wrong output size and truncated input are rejected (a failed
        // decode may leave partial output behind).
        unsafe {
            assert_eq!(vc_decompress(frame.as_ptr(), frame.len(), decoded.as_mut_ptr() as *mut c_void, decoded.len() - 1), 0);
            assert_eq!(vc_decompress(frame.as_ptr(), frame.len() - 1, decoded.as_mut_ptr() as *mut c_void, decoded.len()), 0);
        }
    }
    assert!(!vc_preset_valid(2) && !vc_preset_valid(7));
}

#[test]
fn indexed_leaves_decode_independently() {
    const D: usize = 64;
    const C: usize = 2;
    let n = D * D * D;
    let src = volume(D, C); // channels last
    let mut options = unsafe { vc_default_options() };
    options.shape = [D as u32; 3];
    options.channels = C as u32;
    options.layout = 1;
    let frame = compress(&src, &options);

    unsafe {
        let header = vc_indexed_header_size(frame.as_ptr(), 8);
        assert!(header >= 8 && header as usize <= frame.len());
        let mut index = core::mem::zeroed::<vc_chunk_index>();
        assert_eq!(vc_index_read(frame.as_ptr(), header as usize, frame.len() as u64, &mut index), 1);
        assert_eq!((index.shape, index.channels, index.layout), ([D as u32; 3], C as u32, 1));
        assert_eq!(index.decoded_bytes, (n * C) as u64);
        let leaves = core::slice::from_raw_parts(index.leaves, index.leaf_count);
        assert_eq!(leaves.len(), C);
        let mut decoded = vec![0u8; n];
        for leaf in leaves {
            assert_eq!(leaf.codec, VC_LEAF_CHOICE);
            let payload = &frame[leaf.offset as usize..(leaf.offset + leaf.length) as usize];
            assert_eq!(vc_indexed_leaf_decode(payload.as_ptr(), payload.len(), leaf, decoded.as_mut_ptr(), n), 1);
            assert!(decoded.iter().enumerate().all(|(i, &v)| v == src[C * i + leaf.channel as usize]));
        }
        vc_index_free(&mut index);

        let mut all = vec![0u8; src.len()];
        assert_eq!(vc_indexed_decode(frame.as_ptr(), frame.len(), all.as_mut_ptr(), all.len()), 1);
        assert_eq!(all, src);
    }
}
