"""Compare an in-place native build with the frozen release corpus.

Python only orchestrates native codec calls. Decode frozen bytes, verify their
original raw hash, then require byte-identical re-encoding for retained presets.
"""
import argparse
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
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1] / "python/vc_delta3d"
    spec = importlib.util.spec_from_file_location(
        "vc_delta3d", package / "__init__.py",
        submodule_search_locations=[str(args.build.resolve()), str(package)],
    )
    codec = importlib.util.module_from_spec(spec)
    sys.modules["vc_delta3d"] = codec
    spec.loader.exec_module(codec)
    presets = {"Fast": "fast", "Slow": "medium", "RC1": "slow", "RC2": "slower"}
    rows = [r for r in json.loads(args.frames.read_text()) if r["name"] in presets]
    if not rows:
        raise ValueError("No retained-preset frames in manifest")

    def verify(row):
        frame = Path(row["frame_path"]).read_bytes()
        c = codec.Delta3D(predictor=presets[row["name"]])
        raw = c.decode(frame)
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise AssertionError((row["group"], row["name"], "raw hash changed"))
        encoded = c.encode(np.frombuffer(raw, dtype=np.uint8).reshape((128,) * 3))
        if encoded != frame:
            raise AssertionError((row["group"], row["name"], "encoded bytes changed"))
        return 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        count = sum(pool.map(verify, rows))
    print(f"PASS: {count} frozen frames; exact raw hashes and encoded bytes")


if __name__ == "__main__":
    main()
