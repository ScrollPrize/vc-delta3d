"""Count physical production lines exactly once, including SIMD .inc fragments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
groups = {}

def add(group, names, prefix="src/ng/"):
    for name in names.split():
        path = prefix + name
        assert path not in groups, path
        groups[path] = group

add("Encoder core", """
ng_fast.c ng_medium.c ng_slow.c ng_rect.c ng_presets.c
ng_fit.c ng_fit.h ng_fit_cache.h ng_screen.c ng_screen.h ng_search.c
ng_search.h ng_select.c ng_select.h ng_thresholds.c ng_thresholds.h
ng_predict.c ng_predict.h ng_bank_symbols.h ng_activity_values.h
ng_kernel_fit.h ng_kernel_encode.h ng_native_encode.c
ng_gram_vector.h ng_symbol_mapping.h
""")
add("Decoder core", """
ng_bank_reconstruct.h ng_causal_state.h ng_kernel_decode.h ng_native_decode.c
ng_entropy_vector.h ng_recur_vector.h
ng_recur_wide.h ng_recur_wide_ops.h ng_recur_undef.h ng_gather.h
ng_wave_column.h ng_entropy_permutation.h
ng_entropy_step.inc ng_recur_math.inc
""")
add("Mixed/shared core", """
ng_kernel_common.h ng_kernels.h ng_native_common.c ng_fixed.c
ng_activity_vector.h ng_pre_vector.h ng_pre_wide.h ng_class_lookup.h
ng_entropy_state.h ng_geometry.h residual_bridge.h
""")
add("Mixed/shared core", "rans.c rans.h buffer.h", "src/")
add("Header/container encoder", "ng_frame_encode.c")
add("Header/container encoder", "indexed_encode.c", "src/")
add("Header/container decoder", "ng_frame_decode.c")
add("Header/container decoder", "indexed_decode.c", "src/")
add("Header/selector/container mixed", """
header.c header.h header_tables.h selector.c selector.h selector_bits.h
ng_stream.c ng_stream.h ng_frame.h
""")
add("Header/selector/container mixed", "indexed_internal.h reader.h", "src/")
add("Legacy Delta3D (enc+dec)", "delta_filter.c delta_filter.h legacy.c legacy.h", "src/")
add("C API/dispatch", "api.c", "src/")
add("C API/dispatch", "codec.h indexed.h", "include/vc_delta3d/")
add("C++ adapters", "codec.hpp indexed.hpp detail/codec_adapter.hpp detail/indexed_adapter.hpp", "include/vc_delta3d/")
add("Python bindings/API", "bindings.cpp vc_delta3d/__init__.py vc_delta3d/numcodecs.py", "python/")

files = {str(p.relative_to(ROOT)) for root in ("src", "include", "python")
         for p in (ROOT / root).rglob("*")
         if p.is_file() and p.suffix in (".c", ".h", ".cpp", ".hpp", ".py", ".inc")}
assert files == set(groups), (files - set(groups), set(groups) - files)
counts = {}
for path, group in groups.items():
    rows = counts.setdefault(group, [0, 0])
    rows[0] += 1
    rows[1] += len((ROOT / path).read_bytes().splitlines())
print("| Category | Files | Lines |\n|---|---:|---:|")
for group, (n, lines) in counts.items():
    print(f"| {group} | {n} | {lines:,} |")
print(f"| Total | {sum(n for n, _ in counts.values())} | {sum(lines for _, lines in counts.values()):,} |")
