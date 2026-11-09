from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from vc_delta3d import Delta3D, read_index, indexed_header_size


@pytest.mark.parametrize("dimension", [64, 96, 256])
def test_c_geometry_kernels(dimension):
    # Cover fixed-size kernels and a 32-aligned non-power-of-two runtime kernel.
    z, y, x = np.ogrid[:dimension, :dimension, :dimension]
    data = ((7 * z + 3 * y + x) % 256).astype(np.uint8)
    codec = Delta3D(predictor="fast")
    frame = codec.encode(data)
    assert codec.decode(frame) == data.tobytes()

@pytest.mark.parametrize('preset',['fast','medium','slow','slower'])
def test_presets(preset):
    a=np.random.default_rng(718).integers(0,256,(32,32,32),dtype=np.uint8)
    a[:16]=0
    c=Delta3D(predictor=preset);f=c.encode(a)
    assert f[:4]==b'NCF1'
    np.testing.assert_array_equal(np.frombuffer(c.decode(f),np.uint8).reshape(a.shape),a)
    for bad in (f[:19],f[:-1]):
        with pytest.raises((ValueError,RuntimeError)):c.decode(bad)

def test_modern_thread_safety():
    a=np.random.default_rng(719).integers(0,256,(32,32,32),dtype=np.uint8)
    def run(p):return Delta3D(predictor=p).encode(a)
    names=['fast','medium','slow','slower']*2
    expected=list(map(run,names))
    with ThreadPoolExecutor(max_workers=4) as pool:assert list(pool.map(run,names))==expected

def test_explicit_legacy():
    a=np.zeros((32,32,32),np.uint8)
    c=Delta3D();legacy=Delta3D(predictor='delta').encode(a)
    assert legacy[:5]==b'D3D1\x01' and c.encode(a)[:4]==b'NCF1'
    np.testing.assert_array_equal(np.frombuffer(c.decode(legacy),np.uint8),a.ravel())

@pytest.mark.parametrize('layout',['first','last'])
@pytest.mark.parametrize('preset',['fast'])
def test_channels_contain_unwrapped_native_frames(layout,preset):
    a=np.random.default_rng(811).integers(0,256,(2,32,32,32),dtype=np.uint8)
    if layout=='last':a=np.ascontiguousarray(np.moveaxis(a,0,-1))
    c=Delta3D(predictor=preset,channel_axis=layout);f=c.encode(a)
    assert f[:4]==b'NGI1'
    index=read_index(f[:indexed_header_size(f[:8])],len(f))
    assert index.header_bytes==22 and len(index.leaves)==2
    # Embedded payload begins with header length, not a repeated codec magic.
    assert all(f[l.offset:l.offset+4]!=b'NCF1' for l in index.leaves)
    np.testing.assert_array_equal(np.frombuffer(c.decode(f),np.uint8).reshape(a.shape),a)

def test_discarded_envelope_rejected():
    import struct
    a=np.zeros((32,32,32),np.uint8);c=Delta3D()
    wrapped=struct.pack('<4sBBBBIII',b'D3D1',3,1,1,3,32,32,32)+c.encode(a)
    with pytest.raises((ValueError,RuntimeError)):c.decode(wrapped)
