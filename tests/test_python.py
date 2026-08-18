import numcodecs
import numpy as np

from vc_delta3d import Delta3D, Vcz1


def test_roundtrip_and_legacy_registration():
    source = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)

    for codec_type in (Delta3D, Vcz1):
        codec = codec_type()
        encoded = codec.encode(source)
        assert encoded[:4] == b"VCZ1"
        decoded = np.frombuffer(codec.decode(encoded), dtype=np.uint16)
        np.testing.assert_array_equal(decoded.reshape(source.shape), source)
        assert numcodecs.get_codec(codec.get_config()).codec_id == codec.codec_id


def test_decode_into():
    source = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    codec = Delta3D(codec="zstd")
    output = np.empty_like(source)
    assert codec.decode(codec.encode(source), out=output) is output
    np.testing.assert_array_equal(output, source)
