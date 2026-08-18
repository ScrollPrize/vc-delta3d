#!/usr/bin/env python3
"""Benchmark VC-Delta3D against real chunks from a zarr on S3.

Downloads nonzero chunks from each pyramid level of a zarr v2 store
(anonymous S3 access), caches them locally, builds the C++ benchmark
harness, and reports encode/decode throughput and compression ratio per
level. Cached chunks are reused on subsequent runs, so only the first
run for a given store hits the network.

Usage:
    python3 run.py s3://bucket/path/to/volume.zarr [--chunks 50]
                   [--levels N] [--reps 5]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
CACHE_ROOT = BENCH_DIR / ".cache"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("s3_path", help="s3://... path to a zarr v2 store")
    ap.add_argument("--chunks", type=int, default=50,
                    help="nonzero chunks to sample per level (default 50)")
    ap.add_argument("--levels", type=int, default=None,
                    help="benchmark the first N levels (default: all)")
    ap.add_argument("--reps", type=int, default=5,
                    help="timing repetitions, best-of is reported (default 5)")
    ap.add_argument("--min-nonzero", type=float, default=0.05,
                    help="minimum nonzero-voxel fraction per chunk "
                         "(default 0.05)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="sampling seed, keep fixed for a stable cache")
    return ap.parse_args()


def find_levels(fs, base: str, max_levels: int | None) -> list[tuple[str, str]]:
    """Return [(name, array_path)] for each pyramid level (or the bare array)."""
    if fs.exists(f"{base}/.zarray"):
        return [("L0", base)]
    levels = []
    lvl = 0
    while max_levels is None or lvl < max_levels:
        if not fs.exists(f"{base}/{lvl}/.zarray"):
            break
        levels.append((f"L{lvl}", f"{base}/{lvl}"))
        lvl += 1
    if not levels:
        sys.exit(f"error: no .zarray found under {base} (zarr v2 only)")
    return levels


def load_array_meta(fs, array_path: str) -> dict:
    meta = json.loads(fs.cat(f"{array_path}/.zarray"))
    dtype = meta["dtype"].lstrip("<>|=")
    if dtype not in ("u1", "u2"):
        sys.exit(f"error: {array_path} has dtype {meta['dtype']}; "
                 "vc-delta3d supports uint8/uint16 only")
    if len(meta["shape"]) != 3:
        sys.exit(f"error: {array_path} is not 3D")
    meta["_elem"] = 1 if dtype == "u1" else 2
    meta["_sep"] = meta.get("dimension_separator", ".")
    return meta


def make_decoder(meta: dict):
    """Return bytes -> raw chunk bytes for one stored chunk object."""
    if meta.get("filters"):
        sys.exit("error: zarr filters are not supported by this harness")
    comp = meta.get("compressor")
    if comp is None:
        return lambda data: data
    import numcodecs
    codec = numcodecs.get_codec(comp)
    return lambda data: bytes(codec.decode(data))


def fetch_level(fs, meta: dict, array_path: str, out_dir: Path,
                n_wanted: int, min_nonzero: float, seed: int) -> None:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    cz, cy, cx = meta["chunks"]
    elem = meta["_elem"]
    (out_dir / "shape.txt").write_text(f"{cz} {cy} {cx} {elem}\n")

    have = len(list(out_dir.glob("*.raw")))
    if have >= n_wanted:
        print(f"  {out_dir.name}: {have} chunks cached")
        return

    decode = make_decoder(meta)
    chunk_bytes = cz * cy * cx * elem
    Z, Y, X = meta["shape"]
    nz, ny, nx = Z // cz, Y // cy, X // cx
    if min(nz, ny, nx) < 1:
        sys.exit(f"error: {array_path} has no full chunks")
    sep = meta["_sep"]
    rng = random.Random(seed)

    def try_fetch(coord):
        z, y, x = coord
        key = (f"{array_path}/{z}/{y}/{x}" if sep == "/"
               else f"{array_path}/{z}.{y}.{x}")
        try:
            data = decode(fs.cat(key))
        except FileNotFoundError:
            return None
        if len(data) != chunk_bytes:
            return None  # partial edge chunk
        a = np.frombuffer(data, dtype=np.uint8)
        if (a != 0).mean() < min_nonzero:
            return None
        return (z, y, x, data)

    got = have
    tried: set[tuple[int, int, int]] = set()
    # Start in the middle of the volume where masked scans have data;
    # widen to the full volume if the middle runs dry.
    windows = [(0.2, 0.8), (0.0, 1.0)]
    for lo, hi in windows:
        misses = 0
        while got < n_wanted and misses < 20 * n_wanted:
            batch = []
            while len(batch) < 40 and len(tried) < nz * ny * nx:
                c = (rng.randint(int(nz * lo), max(int(nz * hi) - 1, 0)),
                     rng.randint(int(ny * lo), max(int(ny * hi) - 1, 0)),
                     rng.randint(int(nx * lo), max(int(nx * hi) - 1, 0)))
                if c in tried:
                    continue
                tried.add(c)
                batch.append(c)
            if not batch:
                break
            with ThreadPoolExecutor(16) as ex:
                results = list(ex.map(try_fetch, batch))
            for r in results:
                if r is None:
                    misses += 1
                    continue
                if got >= n_wanted:
                    break
                z, y, x, data = r
                (out_dir / f"{z}_{y}_{x}.raw").write_bytes(data)
                got += 1
            print(f"  {out_dir.name}: {got}/{n_wanted} "
                  f"(tried {len(tried)})", flush=True)
        if got >= n_wanted:
            break
    if got < n_wanted:
        print(f"  {out_dir.name}: WARNING only found {got}/{n_wanted} "
              "nonzero chunks")


def build_bench() -> Path:
    exe = CACHE_ROOT / "bench"
    sources = [BENCH_DIR / "bench.cpp", REPO_ROOT / "src" / "codec.cpp"]
    headers = [REPO_ROOT / "include" / "vc_delta3d" / "codec.hpp"]
    if exe.exists() and all(
            exe.stat().st_mtime > s.stat().st_mtime
            for s in sources + headers):
        return exe
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-O3", "-std=c++20", "-I", str(REPO_ROOT / "include"),
           *map(str, sources), "-o", str(exe)]
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return exe


def main() -> None:
    args = parse_args()
    if not args.s3_path.startswith("s3://"):
        sys.exit("error: expected an s3:// path")
    base = args.s3_path[len("s3://"):].rstrip("/")
    store_cache = CACHE_ROOT / base.rsplit("/", 1)[-1]

    import s3fs
    fs = s3fs.S3FileSystem(anon=True)

    levels = find_levels(fs, base, args.levels)
    print(f"{len(levels)} level(s), caching to {store_cache}")
    for name, array_path in levels:
        meta = load_array_meta(fs, array_path)
        fetch_level(fs, meta, array_path, store_cache / name,
                    args.chunks, args.min_nonzero, args.seed)

    exe = build_bench()
    for name, _ in levels:
        subprocess.run([str(exe), str(store_cache / name), str(args.reps)],
                       check=True)


if __name__ == "__main__":
    main()
