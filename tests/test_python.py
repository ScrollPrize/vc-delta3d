import numcodecs
import numpy as np
import pytest

from vc_delta3d import (
    Delta3D,
    decompress_into_with_magic,
    decompress_with_magic,
)

@pytest.mark.parametrize("preset", ["lpc", "tooslow"])
def test_retired_presets_rejected(preset):
    with pytest.raises(ValueError):
        Delta3D(predictor=preset)


def test_retired_lpc_frame_rejected():
    source = np.zeros((4, 5, 6), np.uint8)
    frame = bytearray(Delta3D(predictor="delta").encode(source))
    frame[7] = 2
    with pytest.raises((ValueError, RuntimeError)):
        Delta3D().decode(frame)


def test_roundtrip_and_registration():
    source = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)

    codec = Delta3D(predictor="delta")
    encoded = codec.encode(source)
    assert encoded[:4] == b"D3D1"
    decoded = np.frombuffer(codec.decode(encoded), dtype=np.uint16)
    np.testing.assert_array_equal(decoded.reshape(source.shape), source)
    assert numcodecs.get_codec(codec.get_config()).codec_id == codec.codec_id


def test_decode_into():
    source = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    codec = Delta3D(predictor="delta")
    output = np.empty_like(source)
    assert codec.decode(codec.encode(source), out=output) is output
    np.testing.assert_array_equal(output, source)


def test_alternate_wire_magic():
    source = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    encoded = Delta3D(predictor="delta").encode(source)
    alternate = b"ALT1" + encoded[4:]

    decoded = decompress_with_magic(alternate, source.nbytes, b"ALT1")
    np.testing.assert_array_equal(
        np.frombuffer(decoded, dtype=np.uint8).reshape(source.shape), source
    )

    output = np.empty_like(source)
    decompress_into_with_magic(alternate, output.view(np.uint8), b"ALT1")
    np.testing.assert_array_equal(output, source)


def test_concurrent_roundtrip():
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(0)
    chunks = [
        rng.integers(0, 4096, size=(16, 32, 32), dtype=np.uint16)
        for _ in range(32)
    ]
    codec = Delta3D(predictor="delta")

    def roundtrip(chunk):
        decoded = np.frombuffer(codec.decode(codec.encode(chunk)), np.uint16)
        return np.array_equal(decoded.reshape(chunk.shape), chunk)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(roundtrip, chunks))




def _scroll_like(shape, seed):
    """Smooth structure plus noise and a masked corner, like scroll CT.

    Integer arithmetic only, so the input is bit-identical on every platform.
    """
    rng = np.random.default_rng(seed)
    # Spatially correlated texture: uniform noise box-blurred over 3^3.
    v = rng.integers(0, 256, size=tuple(n + 2 for n in shape), dtype=np.int64)
    for axis in range(3):
        v = sum(np.take(v, range(k, k + v.shape[axis] - 2), axis=axis) for k in range(3))
    v = 60 + v // 27 // 2 + rng.integers(-2, 3, size=shape)
    z, y, x = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    v[x + y > shape[1] + shape[2] - 12] = 0
    return np.clip(v, 0, 255).astype(np.uint8)





def test_signed_chunks_roundtrip_losslessly():
    source = (np.arange(4 * 5 * 6) % 37 - 18).astype(np.int8).reshape(4, 5, 6)
    codec = Delta3D(predictor="delta")
    decoded = np.frombuffer(codec.decode(codec.encode(source)), dtype=np.int8).reshape(source.shape)
    np.testing.assert_array_equal(decoded, source)
    try:
        Delta3D(quant=3).encode(source)
    except ValueError:
        pass
    else:
        raise AssertionError("lossy quantization of signed data must be refused")


def test_channel_axis_roundtrip_and_config():
    rng = np.random.default_rng(0)
    base = np.cumsum(np.cumsum(rng.integers(-2, 3, size=(6, 7, 8)), axis=1), axis=2)
    for axis in ("first", "last"):
        for dtype in (np.int8, np.uint8, np.int16, np.uint16):
            fields = np.stack([(base + 9 * c) for c in range(5)], axis=0 if axis == "first" else -1)
            source = fields.astype(dtype)
            codec = Delta3D(predictor="delta", channel_axis=axis)
            encoded = codec.encode(source)
            assert encoded[4] == 2
            output = np.empty_like(source)
            codec.decode(encoded, out=output)
            np.testing.assert_array_equal(output, source)
            config = codec.get_config()
            assert config["channel_axis"] == axis
            assert numcodecs.get_codec(config).channel_axis == axis
    assert "channel_axis" not in Delta3D().get_config()



def test_channel_axis_compresses_smooth_channels_better_than_interleaving():
    rng = np.random.default_rng(1)
    base = np.cumsum(np.cumsum(np.cumsum(rng.integers(-1, 2, size=(32, 32, 32)), 0), 1), 2)
    fields = np.stack([(base * (c + 1)) % 200 for c in range(16)], axis=-1).astype(np.uint8)
    per_channel = len(Delta3D(predictor="delta", channel_axis="last").encode(fields))
    interleaved = len(Delta3D(predictor="delta").encode(np.ascontiguousarray(fields.reshape(32, 32, 32 * 16))))
    assert per_channel < interleaved
