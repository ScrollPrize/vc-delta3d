"""VC-Delta3D compression for three-dimensional scalar fields."""

from ._codec import (
    compress,
    compress_array,
    decompress,
    decompress_into,
    decompress_into_with_magic,
    decompress_with_magic,
)
from .numcodecs import Delta3D, register

__all__ = [
    "Delta3D",
    "compress",
    "compress_array",
    "decompress",
    "decompress_into",
    "decompress_into_with_magic",
    "decompress_with_magic",
    "register",
]
