"""Zarr format 3 codec for VC-Delta3D.

The codec is a bytes-to-bytes codec placed after ``bytes``, usually as the inner
codec of ``sharding_indexed``::

    zarr.create_array(
        store, shape=..., dtype="uint8",
        shards=(1024, 1024, 1024), chunks=(128, 128, 128),
        serializer=zarr.codecs.BytesCodec(endian="little"),
        compressors=Delta3DCodec(),
    )

Readers such as neuroglancer expect exactly this chain (``[bytes, vc-delta3d]``).
Encoding reshapes the bytes to the chunk shape from the chunk spec; decoding
needs nothing but the frame, which records its own shape. 16-bit payloads hold
little-endian elements, so ``bytes`` must use ``endian: "little"``.

Inner chunks that are cubic, divisible by 32 and at most 256 on a side are coded
as single NCF1 frames, which the shard index lets readers fetch one at a time.
Other shapes still work: larger or rectangular uint8 chunks become NGI1
containers, and shapes not divisible by 32 fall back to D3D1.
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import parse_named_configuration

from .numcodecs import Delta3D

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

CODEC_NAME = "vc-delta3d"
_DTYPES = (np.dtype("uint8"), np.dtype("uint16"), np.dtype("int8"), np.dtype("int16"))


@dataclass(frozen=True)
class Delta3DCodec(BytesBytesCodec):
    """VC-Delta3D as a Zarr v3 bytes-to-bytes codec.

    The configuration (``predictor``, ``quant``, ``channel_axis``) has the same
    meaning as for the numcodecs :class:`~vc_delta3d.Delta3D` codec. It only
    affects encoding.
    """

    is_fixed_size = False

    predictor: str = "fast"
    quant: int = 1
    channel_axis: str | None = None

    def __init__(
        self, *, predictor: str = "fast", quant: int = 1, channel_axis: str | None = None
    ) -> None:
        # The numcodecs codec checks the arguments and does the work.
        codec = Delta3D(quant=quant, predictor=predictor, channel_axis=channel_axis)
        object.__setattr__(self, "predictor", codec.predictor)
        object.__setattr__(self, "quant", codec.quant)
        object.__setattr__(self, "channel_axis", codec.channel_axis)
        object.__setattr__(self, "_codec", codec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _, configuration = parse_named_configuration(data, CODEC_NAME, require_configuration=False)
        # Keys this version does not know (from a newer encoder) only steered
        # encoding; frames describe themselves, so they never block reading.
        known = {"predictor", "quant", "channel_axis"}
        return cls(**{k: v for k, v in (configuration or {}).items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        configuration: dict[str, Any] = {"predictor": self.predictor, "quant": self.quant}
        if self.channel_axis is not None:
            configuration["channel_axis"] = self.channel_axis
        return {"name": CODEC_NAME, "configuration": configuration}

    def validate(self, *, shape, dtype, chunk_grid) -> None:
        native = dtype.to_native_dtype()
        if native.newbyteorder("=") not in _DTYPES:
            raise ValueError(f"vc-delta3d supports uint8, uint16, int8 and int16, not {native}")
        if native.kind == "i" and self.quant > 1:
            raise ValueError("quantization is defined for unsigned data only; signed data needs quant=1")
        ndim = 3 if self.channel_axis is None else 4
        if len(shape) != ndim:
            hint = "" if self.channel_axis else " (set channel_axis for 4D arrays)"
            raise ValueError(f"vc-delta3d needs {ndim}D arrays here, got {len(shape)}D{hint}")
        chunk_shape = getattr(chunk_grid, "chunk_shape", None)
        if (
            chunk_shape is not None
            and native.itemsize == 1
            and self.quant == 1
            and self.predictor != "delta"
        ):
            _warn_if_legacy(tuple(chunk_shape), self.channel_axis)

    def _encode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        native = chunk_spec.dtype.to_native_dtype()
        data = chunk_bytes.as_numpy_array()
        array = np.frombuffer(data, dtype=native.newbyteorder("<")).reshape(chunk_spec.shape)
        encoded = self._codec.encode(array)
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        expected = prod(chunk_spec.shape) * chunk_spec.dtype.to_native_dtype().itemsize
        decoded = self._codec.decode(chunk_bytes.to_bytes())
        if len(decoded) != expected:
            raise ValueError(
                f"vc-delta3d chunk holds {len(decoded)} bytes, expected {expected} "
                f"for shape {chunk_spec.shape}"
            )
        return chunk_spec.prototype.buffer.from_bytes(decoded)

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._encode_sync, chunk_bytes, chunk_spec)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


def _warn_if_legacy(chunk_shape: tuple[int, ...], channel_axis: str | None) -> None:
    """Warn when a preset cannot be used because of the chunk shape."""
    if channel_axis == "first":
        spatial = chunk_shape[1:]
    elif channel_axis == "last":
        spatial = chunk_shape[:3]
    else:
        spatial = chunk_shape
    if any(edge % 32 for edge in spatial):
        warnings.warn(
            f"vc-delta3d chunk shape {chunk_shape} is not divisible by 32, so it is coded "
            "with the legacy D3D1 codec; use 32-aligned (inner) chunks",
            stacklevel=2,
        )


def register() -> None:
    """Register the codec with zarr (the package entry point does this on demand)."""
    from zarr.registry import register_codec

    register_codec(CODEC_NAME, Delta3DCodec)
