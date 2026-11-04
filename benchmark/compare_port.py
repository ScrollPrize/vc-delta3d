"""Single-core native before/after benchmark on the frozen 18-group corpus.

Python handles I/O, timing and verification only. Every encode/decode is native.
Separate processes avoid symbol interposition between the two codec libraries.
Reports MiB/s and min/median/max throughput over complete group repetitions.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


def worker(args):
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    package = Path(__file__).resolve().parents[1] / "python/vc_delta3d"
    spec = importlib.util.spec_from_file_location(
        "vc_delta3d", package / "__init__.py",
        submodule_search_locations=[str(args.worker.resolve()), str(package)])
    codec = importlib.util.module_from_spec(spec)
    sys.modules["vc_delta3d"] = codec
    spec.loader.exec_module(codec)
    names = {"Fast": "fast", "Slow": "medium", "RC1": "slow", "RC2": "slower"}
    groups = {}
    for row in json.loads(args.frames.read_text()):
        if row["name"] in names:
            groups.setdefault((row["group"], names[row["name"]]), []).append(row)
    results = []
    for (group, preset), entries in groups.items():
        c = codec.Delta3D(predictor=preset)
        frames = [Path(r["frame_path"]).read_bytes() for r in entries]
        arrays = []
        for row, frame in zip(entries, frames):
            raw = c.decode(frame)
            assert hashlib.sha256(raw).hexdigest() == row["sha256"]
            arrays.append(np.frombuffer(raw, dtype=np.uint8).reshape((128,) * 3))
        enc, dec = [], []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            encoded = [c.encode(a) for a in arrays]
            enc.append(time.perf_counter() - start)
            assert encoded == frames, (group, preset, "bytes changed")
            start = time.perf_counter()
            decoded = [c.decode(f) for f in frames]
            dec.append(time.perf_counter() - start)
            assert all(r == a.tobytes() for r, a in zip(decoded, arrays))
        raw_bytes = sum(a.nbytes for a in arrays)
        result = dict(group=group, preset=preset, raw_bytes=raw_bytes,
                      encoded_bytes=sum(map(len, frames)), encode_seconds=enc,
                      decode_seconds=dec, load_average=os.getloadavg())
        results.append(result)
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--build", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--cpu", type=int)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--report-only", action="store_true", help="reuse saved timings without rerunning codecs")
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if not args.output or (not args.report_only and (not args.baseline or not args.build)):
        p.error("--baseline, --build and --output required")
    results = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, build in [("original", args.baseline), ("c_port", args.build)]:
        if args.report_only:
            results[name] = json.loads((args.output / (name + ".json")).read_text())
            continue
        command = [sys.executable, __file__, "--worker", str(build),
                   "--frames", str(args.frames), "--repetitions", str(args.repetitions)]
        if args.cpu is not None:
            command += ["--cpu", str(args.cpu)]
        rows = []
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                row = json.loads(line)
                rows.append(row)
                print(name, row["group"], row["preset"], flush=True)
                (args.output / (name + ".json")).write_text(json.dumps(rows, indent=2))
            if proc.wait():
                raise RuntimeError(name + " benchmark failed")
        results[name] = rows
    table = ["# C port: frozen corpus native timings", "",
             "MiB/s; three complete-group repetitions unless configured otherwise.", "",
             "| Group | Preset | Size % | Original enc | C enc | Original dec | C dec |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for old, new in zip(results["original"], results["c_port"]):
        assert (old["group"], old["preset"], old["encoded_bytes"]) == (new["group"], new["preset"], new["encoded_bytes"])
        speeds = [r["raw_bytes"] / statistics.median(r[k]) / 2**20
                  for k in ["encode_seconds", "decode_seconds"] for r in [old, new]]
        table.append(f'| {old["group"]} | {old["preset"]} | {100*old["encoded_bytes"]/old["raw_bytes"]:.3f} | ' + " | ".join(f"{s:.1f}" for s in speeds) + " |")
    table += ["", "## Combined throughput (min / median / max MiB/s)", "",
              "Aggregate bytes divided by summed group times for each repetition.", "",
              "| Build | Preset | Encode | Decode |", "|---|---|---:|---:|"]
    for name, rows in results.items():
        for preset in ["fast", "medium", "slow", "slower"]:
            selected = [r for r in rows if r["preset"] == preset]
            raw_bytes = sum(r["raw_bytes"] for r in selected)
            cells = []
            for key in ["encode_seconds", "decode_seconds"]:
                speeds = [raw_bytes / sum(r[key][i] for r in selected) / 2**20
                          for i in range(args.repetitions)]
                cells.append(" / ".join(f"{v:.1f}" for v in
                             (min(speeds), statistics.median(speeds), max(speeds))))
            table.append(f"| {name} | {preset} | " + " | ".join(cells) + " |")
    (args.output / "report.md").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    main()
