//! Compiles the bundled C sources the same way CMakeLists.txt and
//! cmake/modern.cmake do. Keep the source list in step with those files.
use std::{env, fs, path::PathBuf, process::Command};

const SOURCES: &[&str] = &[
    "src/delta_filter.c",
    "src/rans.c",
    "src/legacy.c",
    "src/api.c",
    "src/indexed_encode.c",
    "src/indexed_decode.c",
    "src/ng/ng_native_common.c",
    "src/ng/ng_native_encode.c",
    "src/ng/ng_native_decode.c",
    "src/ng/ng_presets.c",
    "src/ng/ng_frame_encode.c",
    "src/ng/ng_frame_decode.c",
    "src/ng/ng_thresholds.c",
    "src/ng/ng_screen.c",
    "src/ng/ng_predict.c",
    "src/ng/ng_fit.c",
    "src/ng/ng_fast.c",
    "src/ng/ng_search.c",
    "src/ng/ng_medium.c",
    "src/ng/ng_stream.c",
    "src/ng/ng_select.c",
    "src/ng/ng_split.c",
    "src/ng/ng_slow.c",
    "src/ng/selector.c",
    "src/ng/header.c",
];

/// src/ng/ng_fixed.c is compiled once per block size.
const FIXED_DIMS: &[u32] = &[32, 64, 128, 256];

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let msvc = env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");

    let mut build = cc::Build::new();
    // The kernels use C99 variable-length arrays, which MSVC's C compiler
    // does not support: build with clang-cl there unless CC says otherwise.
    println!("cargo:rerun-if-env-changed=CC");
    if msvc && env::var_os("CC").is_none() {
        build.compiler("clang-cl");
    }
    build
        .include(root.join("include"))
        .include(root.join("src"))
        .include(root.join("src/ng"))
        .define("VC_DELTA3D_MODERN", "1")
        .define("NDEBUG", None)
        // The codec is unusably slow unoptimized, so it is always built at
        // -O3, also in debug profiles.
        .opt_level(3)
        .warnings(false);
    // No fused multiply-adds: the NG fit must produce identical weights, and
    // so identical payload bytes, on every platform.
    if msvc {
        build.flag("/clang:-std=gnu11").flag("/clang:-ffp-contract=off");
    } else {
        build.flag("-std=gnu11").flag("-ffp-contract=off");
    }

    for src in SOURCES {
        build.file(root.join(src));
    }
    for dim in FIXED_DIMS {
        let wrapper = out.join(format!("ng_fixed{dim}.c"));
        fs::write(
            &wrapper,
            format!(
                "#define NG_FIXED_GEOMETRY {dim}\n#define NG_BLOCK_DIM {dim}\n#include \"ng_fixed.c\"\n"
            ),
        )
        .unwrap();
        build.file(wrapper);
    }
    for dir in ["include", "src"] {
        println!("cargo:rerun-if-changed={}", root.join(dir).display());
    }

    if msvc {
        link_clang_builtins(&build);
    }
    build.compile("vc_delta3d");

    if env::var("CARGO_CFG_UNIX").is_ok() {
        println!("cargo:rustc-link-lib=m");
    }
    // For dependents' build scripts: DEP_VC_DELTA3D_INCLUDE.
    println!("cargo:include={}", root.join("include").display());
}

/// clang-cl does not link compiler-rt's CPU detection state, which
/// `__builtin_cpu_supports` in the SIMD dispatch needs.
fn link_clang_builtins(build: &cc::Build) {
    let compiler = build.get_compiler();
    let dir = Command::new(compiler.path())
        .arg("/clang:--print-runtime-dir")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    let Some(dir) = dir else {
        println!("cargo:warning=vc-delta3d-sys: could not locate the clang runtime directory");
        return;
    };
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=static=clang_rt.builtins-{arch}");
}
