"""numcodecs adapter for VC-Delta3D."""

from __future__ import annotations

import numcodecs
import numpy as np

from . import _codec


class Delta3D(numcodecs.abc.Codec):
    """Compress a 3D uint8/uint16/int8/int16 chunk, or a 4D one with a channel axis.

    ``channel_axis`` ("first" or "last") codes each channel's z, y, x field on its
    own (a version-2 container); leave it unset for 3D chunks. Signed values are
    coded as their bytes and must stay lossless (``quant=1``).

    ``predictor="lpc"`` fits a per-chunk linear predictor to 8-bit fields of
    at least 16 voxels along every axis: smaller payloads, slower encode and
    decode. Fields it does not apply to, or where it would not be smaller,
    use the default delta filter. Decoding needs no configuration either way.
    """

    codec_id = "vc-delta3d"
    _UNSIGNED = {1: np.uint8, 2: np.uint16}

    def __init__(
        self,
        codec: str = "rans",
        quant: int = 1,
        predictor: str = "delta",
        channel_axis: str | None = None,
    ):
        if codec != "rans":
            raise ValueError("vc-delta3d only supports rANS entropy coding")
        if not 1 <= int(quant) <= 255:
            raise ValueError("quant must be in [1, 255]")
        if predictor not in ("delta", "lpc"):
            raise ValueError("predictor must be 'delta' or 'lpc'")
        if channel_axis not in (None, "first", "last"):
            raise ValueError('channel_axis must be None, "first" or "last"')
        self.quant = int(quant)
        self.predictor = predictor
        self.channel_axis = channel_axis

    def encode(self, buf):
        array = np.asarray(buf)
        if array.dtype not in (np.uint8, np.uint16, np.int8, np.int16):
            raise ValueError("vc-delta3d supports uint8, uint16, int8 and int16 chunks")
        if array.dtype.kind == "i" and self.quant > 1:
            raise ValueError("quantization is defined for unsigned data only; signed chunks need quant=1")
        array = np.ascontiguousarray(array).view(self._UNSIGNED[array.dtype.itemsize])
        if self.channel_axis is None:
            if array.ndim != 3:
                raise ValueError("vc-delta3d expects 3D chunks (set channel_axis for 4D chunks)")
            return _codec.compress_array(array, self.quant, self.predictor)
        if array.ndim != 4:
            raise ValueError("channel_axis needs 4D chunks")
        last = self.channel_axis == "last"
        channels = array.shape[-1] if last else array.shape[0]
        z, y, x = array.shape[:3] if last else array.shape[1:]
        return _codec.compress_channels(
            array.tobytes(), z, y, x, channels, last, array.itemsize, self.quant, self.predictor
        )

    def decode(self, buf, out=None):
        payload = buf if isinstance(buf, bytes) else bytes(memoryview(buf))
        expected_size = _decoded_size(payload)
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
        # Omitted for the defaults so existing array metadata stays unchanged.
        if self.predictor != "delta":
            config["predictor"] = self.predictor
        if self.channel_axis is not None:
            config["channel_axis"] = self.channel_axis
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


def _decoded_size(payload) -> int:
    """Bytes a payload decodes to: channels * z * y * x * element size."""
    z, y, x = _shape(payload)
    channels = _codec.channel_count(payload)
    if channels is None:
        raise ValueError("not a VC-Delta3D payload")
    return channels * z * y * x * payload[5]


register()
