import struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from vc_delta3d import (Delta3D,compress_indexed,indexed_header_size,read_index,
                        decode_indexed_leaf,compress_array)

def encode(a,**kwargs):
    return compress_indexed(a.tobytes(),*a.shape[-3:],**kwargs)

@pytest.mark.parametrize('last',[False,True])
@pytest.mark.parametrize('preset',['fast','medium','slow','slower','tooslow'])
def test_independent_ranges_and_compact_header(last,preset):
    a=np.random.default_rng(781).integers(0,256,(3,32,32,32),dtype=np.uint8)
    raw=np.ascontiguousarray(np.moveaxis(a,0,-1)) if last else a
    f=compress_indexed(raw.tobytes(),32,32,32,3,last,preset)
    # Prefix read + exact header extension + any single payload range.
    assert indexed_header_size(f[:7]) is None
    hn=indexed_header_size(f[:8]);assert hn==24+3*17+2*5
    index=read_index(f[:hn],len(f));assert index.channels_last==last
    assert index.decoded_bytes==a.size and index.encoded_bytes==len(f)
    assert index.shape==[32,32,32] and len(index.leaves)==3
    def decode(l):
        payload=f[l.offset:l.offset+l.length] # no other leaf or header provided
        out=np.frombuffer(decode_indexed_leaf(payload,l),np.uint8).reshape(l.shape)
        np.testing.assert_array_equal(out,a[l.channel])
        assert payload==compress_array(a[l.channel],1,preset)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(decode,index.leaves))
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8),raw.ravel())

@pytest.mark.parametrize('last',[False,True])
def test_different_channel_trees_and_rectangular_leaves(last):
    a=np.random.default_rng(782).integers(0,256,(2,64,64,64),dtype=np.uint8)
    raw=np.ascontiguousarray(np.moveaxis(a,0,-1)) if last else a
    # Channel0: one64³ preset leaf. Channel1: two32x64x64 rectangular baseline leaves.
    plan=[(3,1,0),(4,0,0),(0,32,0),(4,0,1),(4,0,1)]
    f=compress_indexed(raw.tobytes(),64,64,64,2,last,'fast',plan)
    hn=indexed_header_size(f[:8]);assert hn==85
    index=read_index(f[:hn],len(f));assert len(index.leaves)==3
    for l in index.leaves:
        z,y,x=l.origin;dz,dy,dx=l.shape
        out=np.frombuffer(decode_indexed_leaf(f[l.offset:l.offset+l.length],l),np.uint8).reshape(l.shape)
        np.testing.assert_array_equal(out,a[l.channel,z:z+dz,y:y+dy,x:x+dx])
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8),raw.ravel())

@pytest.mark.parametrize('shape',[(32,64,96),(96,160,32),(32,32,1024),(2048,32,32),(256,256,32)])
def test_rectangular_geometry(shape):
    a=np.zeros(shape,np.uint8);a[...,::7]=123
    f=encode(a,partition=[(4,0,1)])
    index=read_index(f[:41],len(f));assert index.header_bytes==41
    assert index.leaves[0].shape==list(shape)
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8).reshape(shape),a)

def test_auto_spatial_subdivision():
    a=np.arange(32*64*96,dtype=np.uint8).reshape(32,64,96)
    f=Delta3D().encode(a);index=read_index(f,len(f))
    assert len(index.leaves)==6
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8).reshape(a.shape),a)

@pytest.mark.parametrize('shape',[(32,32,1056),(288,256,32)])
def test_leaf_limits_are_not_logical_chunk_limits(shape):
    a=np.zeros(shape,np.uint8)
    with pytest.raises(ValueError):encode(a,partition=[(4,0,1)])
    f=encode(a);index=read_index(f,len(f));assert len(index.leaves)>1
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8),a.ravel())

@pytest.mark.parametrize('plan',[
    [(4,0,0)], # multi-channel leaf
    [(3,0,0)],[(3,2,0)],[(3,1,0)], # invalid/truncated splits
    [(3,1,0),(4,0,0),(4,0,0),(4,0,0)], # unreachable extra node
    [(0,16,0)],[(0,32,0)], # unaligned/out-of-domain
])
def test_invalid_partitions(plan):
    a=np.zeros((2,32,32,32),np.uint8)
    with pytest.raises(ValueError):encode(a,channels=2,partition=plan)

def test_malformed_index_and_range_payload():
    f=encode(np.zeros((2,32,32,32),np.uint8),channels=2)
    hn=indexed_header_size(f);h=f[:hn]
    def bad(offset,data):return h[:offset]+data+h[offset+len(data):]
    corrupt=[h[:7],h[:-1],bad(0,b'BAD1'),bad(4,struct.pack('<I',hn-1)),
             bad(8,struct.pack('<I',31)),
             bad(20,bytes(4)),bad(24,b'\x04'),bad(25,struct.pack('<I',2)),
             bad(29,b'\xff'),bad(30,struct.pack('<Q',hn+1)),
             bad(38,struct.pack('<Q',2**64-1))]
    for b in corrupt:
        with pytest.raises(ValueError):read_index(b,len(f))
    with pytest.raises(ValueError):read_index(h,len(f)+1)
    index=read_index(h,len(f));l=index.leaves[0];p=f[l.offset:l.offset+l.length]
    for b in (p[:-1],b'D3D1'+p[4:]):
        with pytest.raises(RuntimeError):decode_indexed_leaf(b,l)
    with pytest.raises((ValueError,RuntimeError)):Delta3D().decode(f[:-1])

def test_no_mixed_legacy_container():
    a=np.zeros((32,32,32),np.uint8)
    native=Delta3D().encode(a)
    mixed=struct.pack('<4sBBBBIIII',b'D3D1',2,1,1,0,32,32,32,1)+struct.pack('<I',len(native))+native
    with pytest.raises((ValueError,RuntimeError)):Delta3D().decode(mixed)

def test_explicit_indexed_api_still_requires_valid_partition_geometry():
    # Explicit indexing cannot silently discard the requested tree or emit D3D1.
    with pytest.raises(ValueError):encode(np.zeros((16,32,32),np.uint8))

def test_header_extension_and_64bit_ranges():
    f=compress_indexed(bytes(256*32**3),32,32,32,256)
    ranges=[]
    def fetch(offset,length):
        ranges.append((offset,length));return f[offset:offset+length]
    prefix=fetch(0,4096);hn=indexed_header_size(prefix);assert hn==5651
    h=prefix+fetch(len(prefix),hn-len(prefix));index=read_index(h,len(f))
    l=index.leaves[-1];assert decode_indexed_leaf(fetch(l.offset,l.length),l)==bytes(32**3)
    assert len(ranges)==3 and sum(n for _,n in ranges)==hn+l.length
    # Exercise offsets above4GiB without allocating/fetching those payloads.
    f=compress_indexed(bytes(2*32**3),32,32,32,2)
    h=bytearray(f[:63]);length=(1<<32)+11;total=63+length+4
    struct.pack_into('<Q',h,38,length)
    struct.pack_into('<QQ',h,47,63+length,4)
    index=read_index(bytes(h),total)
    assert index.leaves[1].offset==63+length
