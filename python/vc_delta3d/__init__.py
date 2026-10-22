"""VC-Delta3D compression for three-dimensional scalar fields (optionally multi-channel)."""

from ._codec import (
    channel_count,
    compress,
    compress_array,
    compress_channels,
    decompress,
    decompress_into,
    decompress_into_with_magic,
    decompress_with_magic,
)
from .numcodecs import Delta3D, register

__all__ = [
    "Delta3D",
    "channel_count",
    "compress",
    "compress_array",
    "compress_channels",
    "decompress",
    "decompress_into",
    "decompress_into_with_magic",
    "decompress_with_magic",
    "register",
]
