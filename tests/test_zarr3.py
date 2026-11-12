import json
import subprocess
import sys

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from zarr.codecs import BytesCodec  # noqa: E402
from zarr.storage import MemoryStore  # noqa: E402

from vc_delta3d.zarr3 import Delta3DCodec  # noqa: E402


def volume(shape, dtype=np.uint8, seed=0):
    """A smooth field with some noise, roughly like a CT volume."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*(np.linspace(0, 3, n) for n in shape), indexing="ij")
    field = sum(np.sin(g * (i + 1)) for i, g in enumerate(grids))
    top = np.iinfo(dtype).max if np.dtype(dtype).kind == "u" else np.iinfo(dtype).max // 2
    scaled = (field - field.min()) / (np.ptp(field) or 1) * (top * 0.8)
    scaled += rng.normal(0, top * 0.02, shape)
    return np.clip(scaled, 0, top).astype(dtype)


def create(store, data, chunks, shards=None, **codec_args):
    array = zarr.create_array(
        store,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        shards=shards,
        serializer=BytesCodec(endian="little"),
        compressors=Delta3DCodec(**codec_args),
        fill_value=0,
    )
    array[...] = data
    return array


class RangeLog(MemoryStore):
    """A memory store that records every read as (key, number of bytes returned)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads = []

    def _log(self, key, result):
        self.reads.append((key, 0 if result is None else len(result)))
        return result

    def get_sync(self, key, *, prototype=None, byte_range=None):
        return self._log(key, super().get_sync(key, prototype=prototype, byte_range=byte_range))

    async def get(self, key, prototype=None, byte_range=None):
        return self._log(key, await super().get(key, prototype, byte_range))


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.uint16, np.int16])
def test_unsharded_roundtrip(dtype):
    data = volume((64, 64, 64), dtype)
    store = MemoryStore()
    create(store, data, chunks=(32, 32, 32))
    np.testing.assert_array_equal(zarr.open_array(store, mode="r")[...], data)


def test_unsharded_chunk_is_a_plain_frame():
    data = volume((64, 64, 64))
    store = MemoryStore()
    create(store, data, chunks=(64, 64, 64))
    raw = store._store_dict["c/0/0/0"].to_bytes()
    assert raw[:4] == b"NCF1"


@pytest.mark.parametrize("predictor", ["fast", "slower", "delta"])
def test_sharded_roundtrip_with_partial_shards(predictor):
    # 200 x 160 x 96 leaves partial shards and partial inner chunks on every axis.
    data = volume((200, 160, 96))
    store = MemoryStore()
    create(store, data, chunks=(64, 64, 64), shards=(128, 128, 128), predictor=predictor)
    array = zarr.open_array(store, mode="r")
    np.testing.assert_array_equal(array[...], data)
    np.testing.assert_array_equal(array[130:190, 5:150, 70:90], data[130:190, 5:150, 70:90])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_sharded_metadata_matches_readers(dtype):
    """neuroglancer and VC3D read [bytes, vc-delta3d] inside sharding_indexed."""
    store = MemoryStore()
    create(store, volume((64, 64, 64), dtype), chunks=(32, 32, 32), shards=(64, 64, 64))
    meta = json.loads(store._store_dict["zarr.json"].to_bytes())
    (sharding,) = meta["codecs"]
    assert sharding["name"] == "sharding_indexed"
    # zarr leaves out the byte order for single-byte types.
    serializer = {"name": "bytes"}
    if np.dtype(dtype).itemsize > 1:
        serializer["configuration"] = {"endian": "little"}
    assert sharding["configuration"]["codecs"] == [
        serializer,
        {"name": "vc-delta3d", "configuration": {"predictor": "fast", "quant": 1}},
    ]


def test_reading_one_inner_chunk_fetches_only_that_chunk():
    data = volume((256, 256, 256))
    log = RangeLog()
    create(log, data, chunks=(64, 64, 64), shards=(256, 256, 256))
    shard_size = len(log._store_dict["c/0/0/0"])

    # mode="r" would open a read-only copy of the store without the log.
    array = zarr.open_array(log, mode="r+")
    log.reads.clear()
    np.testing.assert_array_equal(array[64:128, 0:64, 128:192], data[64:128, 0:64, 128:192])

    shard_reads = [n for key, n in log.reads if key == "c/0/0/0"]
    assert len(shard_reads) >= 2  # the index, then the chunk
    assert sum(shard_reads) < shard_size / 16


@pytest.mark.parametrize("channel_axis", ["first", "last"])
def test_channels(channel_axis):
    spatial = volume((64, 64, 64))
    stack = np.stack([spatial, 255 - spatial, spatial // 2], axis=0 if channel_axis == "first" else -1)
    chunks = (3, 32, 32, 32) if channel_axis == "first" else (32, 32, 32, 3)
    store = MemoryStore()
    create(store, stack, chunks=chunks, shards=stack.shape, channel_axis=channel_axis)
    np.testing.assert_array_equal(zarr.open_array(store, mode="r")[...], stack)


def test_quantized_roundtrip_is_lossy_but_close():
    data = volume((64, 64, 64), np.uint16)
    store = MemoryStore()
    create(store, data, chunks=(32, 32, 32), quant=4)
    restored = zarr.open_array(store, mode="r")[...]
    assert np.abs(restored.astype(int) - data.astype(int)).max() <= 4


def test_config_roundtrip_and_unknown_keys():
    codec = Delta3DCodec(predictor="slower", quant=2, channel_axis="last")
    assert Delta3DCodec.from_dict(codec.to_dict()) == codec
    newer = {"name": "vc-delta3d", "configuration": {"predictor": "fast", "future_knob": 3}}
    assert Delta3DCodec.from_dict(newer) == Delta3DCodec()
    assert Delta3DCodec.from_dict({"name": "vc-delta3d"}) == Delta3DCodec()


def test_validation():
    with pytest.raises(ValueError, match="supports"):
        zarr.create_array(MemoryStore(), shape=(32,) * 3, dtype="float32", compressors=Delta3DCodec())
    with pytest.raises(ValueError, match="4D"):
        zarr.create_array(MemoryStore(), shape=(32,) * 3, dtype="uint8", compressors=Delta3DCodec(channel_axis="first"))
    with pytest.raises(ValueError, match="3D"):
        zarr.create_array(MemoryStore(), shape=(32,) * 2, dtype="uint8", compressors=Delta3DCodec())
    with pytest.raises(ValueError, match="signed"):
        zarr.create_array(MemoryStore(), shape=(32,) * 3, dtype="int16", compressors=Delta3DCodec(quant=2))
    with pytest.warns(UserWarning, match="divisible by 32"):
        zarr.create_array(MemoryStore(), shape=(100,) * 3, chunks=(50,) * 3, dtype="uint8", compressors=Delta3DCodec())


def test_entry_point_opens_without_import(tmp_path):
    path = tmp_path / "volume.zarr"
    data = volume((64, 64, 64))
    create(zarr.storage.LocalStore(path), data, chunks=(32, 32, 32), shards=(64, 64, 64))
    np.save(tmp_path / "expected.npy", data)
    script = (
        "import sys, numpy as np, zarr\n"
        "a = zarr.open_array(sys.argv[1], mode='r')[...]\n"
        "np.testing.assert_array_equal(a, np.load(sys.argv[2]))\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(path), str(tmp_path / "expected.npy")],
        check=True,
    )
