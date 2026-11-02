"""Compare legacy scalar/channel/fallback frames with a frozen native build.

Separate processes prevent the two shared libraries interposing their symbols.
Only test orchestration is Python; both codecs execute their native paths.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def worker(build):
    package = Path(__file__).resolve().parents[1] / "python/vc_delta3d"
    spec = importlib.util.spec_from_file_location(
        "vc_delta3d", package / "__init__.py",
        submodule_search_locations=[str(build.resolve()), str(package)],
    )
    codec = importlib.util.module_from_spec(spec)
    sys.modules["vc_delta3d"] = codec
    spec.loader.exec_module(codec)
    rows = {}
    for shape in [(1, 7, 19), (7, 11, 19), (32, 32, 32)]:
        for dtype in [np.uint8, np.uint16]:
            for layout in [None, "first", "last"]:
                dims = shape if layout is None else ((3,) + shape if layout == "first" else shape + (3,))
                source = np.random.default_rng(318).integers(0, np.iinfo(dtype).max + 1, size=dims, dtype=dtype)
                for quant in [1, 3, 100, 255]:
                    c = codec.Delta3D(predictor="delta", quant=quant, channel_axis=layout)
                    frame = c.encode(source)
                    raw = c.decode(frame)
                    # Quantization must be idempotent and container layout stable.
                    assert c.encode(np.frombuffer(raw, dtype=dtype).reshape(dims)) == frame
                    key = str((shape, np.dtype(dtype).str, layout, quant))
                    rows[key] = [hashlib.sha256(frame).hexdigest(), hashlib.sha256(raw).hexdigest()]
    for preset in ["fast", "medium", "slow", "slower"]:
        for shape in [(32, 64, 32), (64, 32, 64)]:
            for layout in [None, "first", "last"]:
                dims = shape if layout is None else ((2,) + shape if layout == "first" else shape + (2,))
                source = np.random.default_rng(319).integers(0, 256, size=dims, dtype=np.uint8)
                c = codec.Delta3D(predictor=preset, channel_axis=layout)
                frame = c.encode(source)
                raw = c.decode(frame)
                assert raw == source.tobytes()
                key = str((preset, shape, layout))
                rows[key] = [hashlib.sha256(frame).hexdigest(), hashlib.sha256(raw).hexdigest()]
    print(json.dumps(rows, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--build", type=Path)
    p.add_argument("--worker", type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not args.baseline or not args.build:
        p.error("--baseline and --build are required")
    def run(build):
        return json.loads(subprocess.check_output([sys.executable, __file__, "--worker", str(build)], text=True))
    before, after = run(args.baseline), run(args.build)
    assert before == after, [k for k in before if before[k] != after.get(k)]
    print(f"PASS: {len(before)} legacy/indexed configurations; exact encoded and decoded hashes")


if __name__ == "__main__":
    main()
