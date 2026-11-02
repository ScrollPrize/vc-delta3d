"""Compare an in-place native build with the frozen release corpus.

Python only orchestrates native codec calls. Decode frozen bytes, verify their
original raw hash, then require byte-identical re-encoding for retained presets.
"""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--decode-only", action="store_true",
                        help="validate historical frames with other block sizes")
    parser.add_argument("--frame-directory", type=Path,
                        help="locate historical frames by their recorded SHA256")
    parser.add_argument("--isa-limit", type=int, choices=[0, 256], default=0,
                        help="256 exercises AVX2 with AVX-512 disabled")
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1] / "python/vc_delta3d"
    spec = importlib.util.spec_from_file_location(
        "vc_delta3d", package / "__init__.py",
        submodule_search_locations=[str(args.build.resolve()), str(package)],
    )
    codec = importlib.util.module_from_spec(spec)
    sys.modules["vc_delta3d"] = codec
    spec.loader.exec_module(codec)
    native = importlib.import_module("vc_delta3d._codec")
    library = ctypes.CDLL(native.__file__)
    isa_limit = library.ng_set_isa_limit
    isa_limit.argtypes = [ctypes.c_int]
    isa_limit.restype = ctypes.c_int
    presets = {"Fast": "fast", "Slow": "medium", "RC1": "slow", "RC2": "slower"}
    presets.update({"Slower-RC1": "slow", "Slower-RC2": "slower"})
    by_hash = {}
    if args.frame_directory:
        for path in args.frame_directory.glob("*.bin"):
            by_hash[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    rows = [r for r in json.loads(args.frames.read_text()) if r["name"] in presets]
    if not rows:
        raise ValueError("No retained-preset frames in manifest")

    def verify(row):
        # This setting is thread-local; apply in every worker, not only main.
        if isa_limit(args.isa_limit):
            raise RuntimeError("ISA ceiling not supported")
        path = (by_hash[row["frame_sha256"]] if args.frame_directory
                else Path(row["frame_path"]))
        frame = path.read_bytes()
        c = codec.Delta3D(predictor=presets[row["name"]])
        raw = c.decode(frame)
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise AssertionError((row["group"], row["name"], "raw hash changed"))
        if not args.decode_only:
            encoded = c.encode(np.frombuffer(raw, dtype=np.uint8).reshape((128,) * 3))
            if encoded != frame:
                raise AssertionError((row["group"], row["name"], "encoded bytes changed"))
        return 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        count = sum(pool.map(verify, rows))
    checks = "raw hashes" if args.decode_only else "raw hashes and encoded bytes"
    print(f"PASS: {count} frozen frames; exact {checks}; ISA limit {args.isa_limit}")


if __name__ == "__main__":
    main()
