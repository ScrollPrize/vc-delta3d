"""numcodecs adapter for VC-Delta3D."""

from __future__ import annotations

import numcodecs
import numpy as np

from . import _codec


class Delta3D(numcodecs.abc.Codec):
    """Compress a three-dimensional uint8 or uint16 chunk.

    ``predictor="lpc"`` fits a per-chunk linear predictor to uint8 chunks of
    at least 16 voxels along every axis: smaller payloads, slower encode and
    decode. Chunks it does not apply to, or where it would not be smaller,
    use the default delta filter. Decoding needs no configuration either way.
    """

    codec_id = "vc-delta3d"

    def __init__(self, codec: str = "rans", quant: int = 1, predictor: str = "delta"):
        if codec != "rans":
            raise ValueError("vc-delta3d only supports rANS entropy coding")
        if not 1 <= int(quant) <= 255:
            raise ValueError("quant must be in [1, 255]")
        if predictor not in ("delta", "lpc"):
            raise ValueError("predictor must be 'delta' or 'lpc'")
        self.quant = int(quant)
        self.predictor = predictor

    def encode(self, buf):
        array = np.asarray(buf)
        if array.ndim != 3:
            raise ValueError("vc-delta3d expects 3D chunks")
        if array.dtype not in (np.uint8, np.uint16):
            raise ValueError("vc-delta3d supports uint8 and uint16 chunks")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return _codec.compress_array(array, self.quant, self.predictor)

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
        config = {"id": self.codec_id, "quant": self.quant}
        # Omitted for the default so existing array metadata stays unchanged.
        if self.predictor != "delta":
            config["predictor"] = self.predictor
        return config


def register() -> None:
    """Register the VC-Delta3D codec identifier."""

    numcodecs.register_codec(Delta3D)


def _shape(payload) -> tuple[int, int, int]:
    if len(payload) < 20 or payload[:4] != b"D3D1":
        raise ValueError("not a VC-Delta3D payload")
    return (
        int.from_bytes(payload[8:12], "little"),
        int.from_bytes(payload[12:16], "little"),
        int.from_bytes(payload[16:20], "little"),
    )


register()
