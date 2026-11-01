import struct

import numpy as np
import pytest

from vc_delta3d import Delta3D


@pytest.mark.parametrize('preset', ['fast', 'medium', 'slow', 'slower', 'tooslow'])
@pytest.mark.parametrize('layout', [None, 'first', 'last'])
@pytest.mark.parametrize('dtype,shape,quant', [
    (np.uint16, (32,32,32), 1),
    (np.int16, (32,32,32), 1),
    (np.uint8, (33,32,32), 1),
    (np.uint8, (8,16,32), 1),
    (np.uint8, (32,32,32), 3),
    (np.uint16, (32,32,32), 3),
])
def test_incompatible_inputs_use_entire_legacy_path(preset, layout, dtype, shape, quant):
    spatial = np.arange(np.prod(shape), dtype=np.uint64).reshape(shape)
    data = ((spatial * 17 + spatial // 7) % 251).astype(dtype)
    if np.issubdtype(dtype, np.signedinteger):
        data -= 125
    if layout is not None:
        data = np.stack([data, np.flip(data).copy()])
        if layout == 'last':
            data = np.ascontiguousarray(np.moveaxis(data, 0, -1))
    options = dict(quant=quant, channel_axis=layout)
    legacy = Delta3D(predictor='delta', **options)
    frame = Delta3D(predictor=preset, **options).encode(data)
    assert frame == legacy.encode(data)
    assert frame[:5] == b'D3D1' + bytes([1 if layout is None else 2])
    if layout is not None:
        # Legacy multi-channel fallback contains only legacy scalar frames.
        count = struct.unpack_from('<I', frame, 20)[0]
        sizes = struct.unpack_from('<' + 'I' * count, frame, 24)
        offset = 24 + 4 * count
        for size in sizes:
            assert frame[offset:offset+5] == b'D3D1\x01'
            offset += size
        assert offset == len(frame)
    decoded = Delta3D().decode(frame)
    assert decoded == legacy.decode(frame)
    if quant == 1:
        assert decoded == data.tobytes()


def test_default_falls_back_but_invalid_inputs_still_fail():
    data = np.zeros((7,9,11), np.uint16)
    assert Delta3D().encode(data) == Delta3D(predictor='delta').encode(data)
    for bad in [np.zeros((0,32,32), np.uint8), np.zeros((32,32,32), np.float32)]:
        with pytest.raises(ValueError):
            Delta3D().encode(bad)
    with pytest.raises(ValueError):
        Delta3D(quant=3).encode(data.astype(np.int16))
