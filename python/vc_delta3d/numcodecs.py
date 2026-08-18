"""numcodecs adapters for VC-Delta3D and its legacy VCZ1 identifier."""

from __future__ import annotations

import numcodecs
import numpy as np

from . import _codec


class Delta3D(numcodecs.abc.Codec):
    """Compress a three-dimensional uint8 or uint16 chunk."""

    codec_id = "vc-delta3d"

    def __init__(self, codec: str = "rans", quant: int = 1):
        if codec not in {"rans", "zstd"}:
            raise ValueError("codec must be 'rans' or 'zstd'")
        if not 1 <= int(quant) <= 255:
            raise ValueError("quant must be in [1, 255]")
        self.codec = codec
        self.quant = int(quant)

    def encode(self, buf):
        array = np.asarray(buf)
        if array.ndim != 3:
            raise ValueError("vc-delta3d expects 3D chunks")
        if array.dtype not in (np.uint8, np.uint16):
            raise ValueError("vc-delta3d supports uint8 and uint16 chunks")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return _codec.compress_array(array, self.quant, self.codec)

    def decode(self, buf, out=None):
        payload = buf if isinstance(buf, bytes) else bytes(memoryview(buf))
        z, y, x = _shape(payload)
        expected_size = z * y * x * payload[5]
        if out is not None:
            out_bytes = np.frombuffer(out, dtype=np.uint8)
            if out_bytes.size != expected_size:
                raise ValueError(
                    f"output buffer has {out_bytes.size} bytes, "
                    f"expected {expected_size}"
                )
            _codec.decompress_into(payload, out_bytes)
            return out
        return _codec.decompress(payload, expected_size)

    def get_config(self):
        return {"id": self.codec_id, "codec": self.codec, "quant": self.quant}


class Vcz1(Delta3D):
    """Compatibility adapter for Zarr metadata using the original codec ID."""

    codec_id = "vcz1"


def register() -> None:
    """Register both the current and legacy codec identifiers."""

    numcodecs.register_codec(Delta3D)
    numcodecs.register_codec(Vcz1)


def _shape(payload) -> tuple[int, int, int]:
    if len(payload) < 20 or payload[:4] != b"VCZ1":
        raise ValueError("not a VC-Delta3D payload")
    return (
        int.from_bytes(payload[8:12], "little"),
        int.from_bytes(payload[12:16], "little"),
        int.from_bytes(payload[16:20], "little"),
    )


register()
