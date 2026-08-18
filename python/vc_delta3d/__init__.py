"""VC-Delta3D compression for three-dimensional scalar fields."""

from ._codec import compress, compress_array, decompress, decompress_into
from .numcodecs import Delta3D, Vcz1, register

__all__ = [
    "Delta3D",
    "Vcz1",
    "compress",
    "compress_array",
    "decompress",
    "decompress_into",
    "register",
]
