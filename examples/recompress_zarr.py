"""Recompress one or more resolution levels of a public Zarr pyramid.

The source store is read anonymously from S3. The output may be a local path
or a writable fsspec URL such as ``s3://my-bucket/output.zarr``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

import zarr
from vc_delta3d import Delta3D

SOURCE_URL = (
    "s3://vesuvius-challenge-open-data/PHercParis4/volumes/"
    "20260411134726-2.400um-0.2m-78keV-masked.zarr"
)


@dataclass
class LevelReport:
    level: str
    start_bytes: int
    end_bytes: int
    copy_seconds: float
    benchmark_bytes: int
    encode_seconds: float
    decode_seconds: float


def block_indices(cdata_shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Yield every chunk-grid index without materializing the full grid."""

    return itertools.product(*(range(size) for size in cdata_shape))


def copy_blocks(
    source: zarr.Array,
    target: zarr.Array,
    *,
    workers: int,
) -> float:
    """Read and recompress an array one chunk at a time with bounded memory."""

    indices = iter(block_indices(source.cdata_shape))
    total = math.prod(source.cdata_shape)
    max_pending = max(workers * 2, 1)
    started = time.monotonic()
    completed = 0

    def copy_one(index: tuple[int, ...]) -> None:
        target.blocks[index] = source.blocks[index]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future[None], tuple[int, ...]] = {}

        for index in itertools.islice(indices, max_pending):
            pending[executor.submit(copy_one, index)] = index

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    future.result()
                except Exception as error:
                    raise RuntimeError(f"failed to copy chunk {index}") from error

                completed += 1
                if completed == total or completed % 100 == 0:
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed if elapsed else 0.0
                    print(
                        f"  {completed:,}/{total:,} chunks "
                        f"({completed / total:.1%}, {rate:.1f} chunks/s)",
                        flush=True,
                    )

                try:
                    next_index = next(indices)
                except StopIteration:
                    continue
                pending[executor.submit(copy_one, next_index)] = next_index

    return time.monotonic() - started


def flat_block_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a flat chunk-grid offset to a multidimensional block index."""

    result = []
    for size in reversed(shape):
        flat_index, coordinate = divmod(flat_index, size)
        result.append(coordinate)
    return tuple(reversed(result))


def benchmark_codec(
    source: zarr.Array,
    *,
    quant: int,
    sample_chunks: int,
) -> tuple[int, float, float]:
    """Benchmark codec-only encode/decode time on evenly spaced chunks."""

    if sample_chunks <= 0:
        return 0, 0.0, 0.0

    total_chunks = math.prod(source.cdata_shape)
    count = min(sample_chunks, total_chunks)
    flat_indices = [(index * total_chunks) // count for index in range(count)]
    codec = Delta3D(quant=quant)
    payloads: list[tuple[bytes, int]] = []
    benchmark_bytes = 0
    encode_seconds = 0.0

    for flat_index in flat_indices:
        block = source.blocks[flat_block_index(flat_index, source.cdata_shape)]
        started = time.perf_counter()
        payload = codec.encode(block)
        encode_seconds += time.perf_counter() - started
        payloads.append((payload, block.nbytes))
        benchmark_bytes += block.nbytes

    decode_seconds = 0.0
    for payload, expected_size in payloads:
        started = time.perf_counter()
        decoded = codec.decode(payload)
        decode_seconds += time.perf_counter() - started
        if len(decoded) != expected_size:
            raise RuntimeError(
                f"codec benchmark decoded {len(decoded)} bytes, "
                f"expected {expected_size}"
            )

    return benchmark_bytes, encode_seconds, decode_seconds


def selected_group_attributes(
    source_attributes: dict,
    levels: Iterable[str],
) -> dict:
    """Keep OME-Zarr multiscale metadata consistent with selected levels."""

    selected = set(levels)
    attributes = copy.deepcopy(source_attributes)
    for multiscale in attributes.get("multiscales", []):
        multiscale["datasets"] = [
            dataset
            for dataset in multiscale.get("datasets", [])
            if dataset.get("path") in selected
        ]
    return attributes


def recompress_level(
    source_group: zarr.Group,
    target_group: zarr.Group,
    level: str,
    *,
    quant: int,
    workers: int,
    benchmark_chunks: int,
) -> LevelReport:
    """Create and populate one output pyramid level."""

    source = source_group[level]
    if not isinstance(source, zarr.Array):
        raise TypeError(f"source member {level!r} is not an array")
    if source.ndim != 3 or source.dtype.name not in {"uint8", "uint16"}:
        raise TypeError(
            f"level {level} must be a 3D uint8 or uint16 array, got "
            f"shape={source.shape}, dtype={source.dtype}"
        )

    print(
        f"level {level}: shape={source.shape}, chunks={source.chunks}, "
        f"dtype={source.dtype}",
        flush=True,
    )
    target = target_group.create_array(
        level,
        shape=source.shape,
        chunks=source.chunks,
        dtype=source.dtype,
        compressor=Delta3D(quant=quant),
        fill_value=source.fill_value,
        order=source.order,
    )
    target.attrs.update(dict(source.attrs))
    copy_seconds = copy_blocks(source, target, workers=workers)

    print(f"  measuring level {level} stored size", flush=True)
    end_bytes = target.nbytes_stored()
    benchmark_bytes, encode_seconds, decode_seconds = benchmark_codec(
        source,
        quant=quant,
        sample_chunks=benchmark_chunks,
    )
    return LevelReport(
        level=level,
        start_bytes=source.nbytes,
        end_bytes=end_bytes,
        copy_seconds=copy_seconds,
        benchmark_bytes=benchmark_bytes,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
    )


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def mib_per_second(byte_count: int, seconds: float) -> str:
    if byte_count == 0 or seconds <= 0:
        return "n/a"
    return f"{byte_count / (1024 * 1024) / seconds:.1f} MiB/s"


def print_report(reports: list[LevelReport], benchmark_chunks: int) -> None:
    start_bytes = sum(report.start_bytes for report in reports)
    end_bytes = sum(report.end_bytes for report in reports)
    copy_seconds = sum(report.copy_seconds for report in reports)
    benchmark_bytes = sum(report.benchmark_bytes for report in reports)
    encode_seconds = sum(report.encode_seconds for report in reports)
    decode_seconds = sum(report.decode_seconds for report in reports)

    print("\nRecompression report")
    print("--------------------")
    for report in reports:
        ratio = report.start_bytes / report.end_bytes
        print(
            f"level {report.level}: {human_bytes(report.start_bytes)} -> "
            f"{human_bytes(report.end_bytes)} ({ratio:.2f}x), "
            f"encode {mib_per_second(report.benchmark_bytes, report.encode_seconds)}, "
            f"decode {mib_per_second(report.benchmark_bytes, report.decode_seconds)}, "
            f"copy {mib_per_second(report.start_bytes, report.copy_seconds)}"
        )

    ratio = start_bytes / end_bytes
    print(f"start size:       {human_bytes(start_bytes)} ({start_bytes:,} bytes)")
    print(f"end size:         {human_bytes(end_bytes)} ({end_bytes:,} bytes)")
    print(f"compression ratio: {ratio:.2f}x ({end_bytes / start_bytes:.2%} of raw)")
    print(f"encode speed:     {mib_per_second(benchmark_bytes, encode_seconds)}")
    print(f"decode speed:     {mib_per_second(benchmark_bytes, decode_seconds)}")
    print(f"copy throughput:  {mib_per_second(start_bytes, copy_seconds)}")
    if benchmark_chunks > 0:
        print(
            f"codec speeds sampled from up to {benchmark_chunks} evenly spaced "
            "chunks per level; copy throughput includes source reads and "
            "output writes"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        help="New local path or writable fsspec URL for the output Zarr group",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        metavar="LEVEL",
        default=["5"],
        help="Levels to copy (default: 5)",
    )
    parser.add_argument(
        "--quant",
        type=int,
        default=1,
        help="Quantization bin width (default: 1, lossless)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum concurrent chunk transfers (default: 8)",
    )
    parser.add_argument(
        "--benchmark-chunks",
        type=int,
        default=128,
        help="Chunks per level used for codec speed measurements (default: 128)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output store instead of failing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.quant <= 255:
        raise SystemExit("--quant must be in [1, 255]")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.benchmark_chunks < 0:
        raise SystemExit("--benchmark-chunks must be nonnegative")

    # This public AWS store requires unsigned requests. Zarr passes anon=True
    # through to s3fs via its fsspec store implementation.
    source_group = zarr.open_group(
        SOURCE_URL,
        mode="r",
        storage_options={"anon": True},
    )

    available_levels = sorted(source_group.array_keys(), key=int)
    levels = args.levels
    missing = sorted(set(levels) - set(available_levels))
    if missing:
        raise SystemExit(
            f"levels not found: {', '.join(missing)}; available levels: "
            f"{', '.join(available_levels)}"
        )

    target_group = zarr.open_group(
        args.output,
        mode="w" if args.overwrite else "w-",
        zarr_format=2,
    )
    target_group.attrs.update(
        selected_group_attributes(dict(source_group.attrs), levels)
    )

    print(f"source: {SOURCE_URL}")
    print(f"output: {args.output}")
    print(f"levels: {', '.join(levels)}")
    print(f"codec: Delta3D(quant={args.quant})")
    reports = []
    for level in levels:
        reports.append(
            recompress_level(
                source_group,
                target_group,
                level,
                quant=args.quant,
                workers=args.workers,
                benchmark_chunks=args.benchmark_chunks,
            )
        )

    zarr.consolidate_metadata(target_group.store)
    print_report(reports, args.benchmark_chunks)


if __name__ == "__main__":
    main()
