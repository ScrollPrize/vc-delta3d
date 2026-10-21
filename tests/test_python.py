import numcodecs
import numpy as np

from vc_delta3d import (
    Delta3D,
    decompress_into_with_magic,
    decompress_with_magic,
)


def test_roundtrip_and_registration():
    source = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)

    codec = Delta3D()
    encoded = codec.encode(source)
    assert encoded[:4] == b"D3D1"
    decoded = np.frombuffer(codec.decode(encoded), dtype=np.uint16)
    np.testing.assert_array_equal(decoded.reshape(source.shape), source)
    assert numcodecs.get_codec(codec.get_config()).codec_id == codec.codec_id


def test_decode_into():
    source = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    codec = Delta3D()
    output = np.empty_like(source)
    assert codec.decode(codec.encode(source), out=output) is output
    np.testing.assert_array_equal(output, source)


def test_alternate_wire_magic():
    source = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    encoded = Delta3D().encode(source)
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
    codec = Delta3D()

    def roundtrip(chunk):
        decoded = np.frombuffer(codec.decode(codec.encode(chunk)), np.uint16)
        return np.array_equal(decoded.reshape(chunk.shape), chunk)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(roundtrip, chunks))


# sha256 of the LPC payload of _scroll_like((32, 40, 36), 1).
GOLDEN_LPC_SHA256 = "f118a1397c61557b85b45b2a47a2f9c75ee38bdf128eb78ec12ad02a1f7d818c"


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


def test_lpc_roundtrip_and_config():
    source = _scroll_like((32, 40, 36), 1)
    codec = Delta3D(predictor="lpc")
    encoded = codec.encode(source)
    assert encoded[7] == 2  # LPC predictor marker
    assert len(encoded) < len(Delta3D().encode(source))
    decoded = np.frombuffer(codec.decode(encoded), dtype=np.uint8)
    np.testing.assert_array_equal(decoded.reshape(source.shape), source)

    config = codec.get_config()
    assert config == {"id": "vc-delta3d", "quant": 1, "predictor": "lpc"}
    assert numcodecs.get_codec(config).predictor == "lpc"
    # The default keeps the metadata existing arrays carry.
    assert Delta3D().get_config() == {"id": "vc-delta3d", "quant": 1}


def test_lpc_payload_is_identical_on_every_platform():
    # The LPC fit is exact integer accumulation plus an FMA-free double
    # solve, and the SIMD sums are exact, so every platform must produce
    # these bytes. A different digest means the encoder diverged from the
    # build that recorded it (or the format changed).
    import hashlib

    encoded = Delta3D(predictor="lpc").encode(_scroll_like((32, 40, 36), 1))
    assert encoded[7] == 2
    assert hashlib.sha256(encoded).hexdigest() == GOLDEN_LPC_SHA256


def test_lpc_concurrent_roundtrip():
    from concurrent.futures import ThreadPoolExecutor

    chunks = [_scroll_like((24, 24, 24), seed) for seed in range(16)]
    codec = Delta3D(predictor="lpc")

    def roundtrip(chunk):
        decoded = np.frombuffer(codec.decode(codec.encode(chunk)), np.uint8)
        return np.array_equal(decoded.reshape(chunk.shape), chunk)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(roundtrip, chunks))
