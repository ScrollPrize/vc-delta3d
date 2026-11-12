import struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from vc_delta3d import (Delta3D,compress_indexed,indexed_header_size,read_index,
                        decode_indexed_leaf,compress_array)

def encode(a,**kwargs):
    return compress_indexed(a.tobytes(),*a.shape[-3:],**kwargs)

def varint(v):
    out=bytearray()
    while True:
        out.append((v&127)|(128 if v>127 else 0));v>>=7
        if not v:return bytes(out)

def index_bytes(shape,channels,template,lengths):
    """NGI1 header size: prefix, flags, varint root/channels, template, varint lengths."""
    return (8+1+sum(len(varint(s//32-1)) for s in shape)+len(varint(channels-1))
            +template+sum(len(varint(n)) for n in lengths))

def raw_index(shape,channels,template,lengths,last=False):
    h=bytearray(b'NGI1'+bytes(4)+bytes([int(last)]))
    for s in shape:h+=varint(s//32-1)
    h+=varint(channels-1)+template
    for n in lengths:h+=varint(n)
    struct.pack_into('<I',h,4,len(h));return bytes(h)

@pytest.mark.parametrize('last',[False,True])
@pytest.mark.parametrize('preset',['faster','fast','balanced','small','medium','slow','slower'])
def test_independent_ranges_and_compact_header(last,preset):
    a=np.random.default_rng(781).integers(0,256,(3,32,32,32),dtype=np.uint8)
    raw=np.ascontiguousarray(np.moveaxis(a,0,-1)) if last else a
    f=compress_indexed(raw.tobytes(),32,32,32,3,last,preset)
    # Prefix read + exact header extension + any single payload range.
    assert indexed_header_size(f[:5]) is None
    hn=indexed_header_size(f[:8])
    index=read_index(f[:hn],len(f));assert index.channels_last==last
    assert hn==index_bytes([32]*3,3,3,[l.length for l in index.leaves])
    assert index.decoded_bytes==a.size and index.encoded_bytes==len(f)
    assert index.shape==[32,32,32] and len(index.leaves)==3
    def decode(l):
        payload=f[l.offset:l.offset+l.length] # no other leaf or header provided
        out=np.frombuffer(decode_indexed_leaf(payload,l),np.uint8).reshape(l.shape)
        np.testing.assert_array_equal(out,a[l.channel])
        assert payload==compress_array(a[l.channel],1,preset)[4:]
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(decode,index.leaves))
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8),raw.ravel())

@pytest.mark.parametrize('last',[False,True])
def test_different_channel_trees(last):
    a=np.random.default_rng(782).integers(0,256,(2,64,64,64),dtype=np.uint8)
    raw=np.ascontiguousarray(np.moveaxis(a,0,-1)) if last else a
    # Channel0: one64³ preset leaf. Channel1: eight32³ leaves with the same preset.
    plan=[(3,1),(4,0),(5,2),(6,2),(7,2),(4,0)]
    f=compress_indexed(raw.tobytes(),64,64,64,2,last,'fast',plan)
    hn=indexed_header_size(f[:8])
    index=read_index(f[:hn],len(f));assert len(index.leaves)==9
    assert hn==index_bytes([64]*3,2,10,[l.length for l in index.leaves])
    for l in index.leaves:
        z,y,x=l.origin;dz,dy,dx=l.shape
        out=np.frombuffer(decode_indexed_leaf(f[l.offset:l.offset+l.length],l),np.uint8).reshape(l.shape)
        np.testing.assert_array_equal(out,a[l.channel,z:z+dz,y:y+dy,x:x+dx])
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8),raw.ravel())

@pytest.mark.parametrize('shape',[(32,64,96),(96,160,32),(32,32,1024),(2048,32,32),(256,256,32)])
def test_rectangular_geometry(shape):
    a=np.zeros(shape,np.uint8);a[...,::7]=123
    f=encode(a)
    hn=indexed_header_size(f);index=read_index(f[:hn],len(f))
    assert len(index.leaves)>1
    assert all(len(set(l.shape))==1 for l in index.leaves)
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8).reshape(shape),a)

def test_auto_spatial_subdivision():
    a=np.arange(32*64*96,dtype=np.uint8).reshape(32,64,96)
    f=Delta3D().encode(a);index=read_index(f,len(f))
    assert len(index.leaves)==6
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8).reshape(a.shape),a)

@pytest.mark.parametrize('preset',['faster','fast','balanced','small','medium','slow','slower'])
def test_rectangular_chunk_uses_requested_preset(preset):
    a=np.random.default_rng(785).integers(0,256,(32,64,32),dtype=np.uint8)
    f=encode(a,predictor=preset);index=read_index(f,len(f))
    for l in index.leaves:
        z,y,x=l.origin;dz,dy,dx=l.shape
        leaf=np.ascontiguousarray(a[z:z+dz,y:y+dy,x:x+dx])
        assert f[l.offset:l.offset+l.length]==compress_array(leaf,1,preset)[4:]
    with pytest.raises(ValueError):encode(a,predictor=preset,partition=[(4,0)])
    with pytest.raises((ValueError,TypeError)):encode(a,partition=[(4,0,1)])

@pytest.mark.parametrize('shape',[(32,32,1056),(288,256,32)])
def test_leaf_limits_are_not_logical_chunk_limits(shape):
    a=np.zeros(shape,np.uint8)
    with pytest.raises(ValueError):encode(a,partition=[(4,0)])
    f=encode(a);index=read_index(f,len(f));assert len(index.leaves)>1
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8),a.ravel())

@pytest.mark.parametrize('plan',[
    [(4,0)], # multi-channel leaf
    [(3,0)],[(3,2)],[(3,1)], # invalid/truncated splits
    [(3,1),(4,0),(4,0),(4,0)], # unreachable extra node
    [(0,16)],[(0,32)], # unaligned/out-of-domain
])
def test_invalid_partitions(plan):
    a=np.zeros((2,32,32,32),np.uint8)
    with pytest.raises(ValueError):encode(a,channels=2,partition=plan)

def test_malformed_index_and_range_payload():
    f=encode(np.zeros((2,32,32,32),np.uint8),channels=2)
    hn=indexed_header_size(f);h=f[:hn]
    def bad(offset,data):return h[:offset]+data+h[offset+len(data):]
    # NGI1 | u32 length | flags@8 | dims@9-11 | channels@12 | repeat@13,14 | leaf@15 | lengths@16
    assert h[13:16]==b'\x07\x01\x08'
    corrupt=[h[:7],h[:-1],bad(0,b'BAD1'),bad(4,struct.pack('<I',hn-1)),
             bad(8,b'\x04'),bad(9,b'\xff'),bad(12,b'\x02'),
             bad(13,b'\xff'),bad(14,b'\x00'),bad(15,b'\xff'),
             bad(16,b'\xff\xff\xff\xff\x0f')]
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
    f=compress_indexed(bytes(1024*32**3),32,32,32,1024)
    ranges=[]
    def fetch(offset,length):
        ranges.append((offset,length));return f[offset:offset+length]
    prefix=fetch(0,1024);hn=indexed_header_size(prefix)
    assert f[:4]==b'NGI1' and hn>len(prefix)
    h=prefix+fetch(len(prefix),hn-len(prefix));index=read_index(h,len(f))
    assert hn==index_bytes([32]*3,1024,4,[l.length for l in index.leaves])
    l=index.leaves[-1];assert decode_indexed_leaf(fetch(l.offset,l.length),l)==bytes(32**3)
    assert len(ranges)==3 and sum(n for _,n in ranges)==hn+l.length
    # Expanded repeat template with >4GiB leaf; no payload allocation.
    length=(1<<32)+11;h=raw_index([32]*3,2,b'\x07\x01\x08',[length,4])
    index=read_index(h,len(h)+length+4)
    assert index.leaves[1].offset==len(h)+length
    # Leaves below 4GiB can still produce >4GiB cumulative offsets.
    h=raw_index([32]*3,3,b'\x07\x02\x08',[2**32-1,16,4])
    index=read_index(h,len(h)+2**32-1+20)
    assert index.leaves[2].offset>2**32

@pytest.mark.parametrize('last',[False,True])
def test_repeated_spatial_template(last):
    a=np.random.default_rng(99).integers(0,256,(3,64,32,64),dtype=np.uint8)
    raw=np.ascontiguousarray(np.moveaxis(a,0,-1)) if last else a
    plan=[(8,3),(5,2),(7,2),(4,0)]
    f=compress_indexed(raw.tobytes(),64,32,64,3,last,'fast',plan)
    idx=read_index(f,len(f));assert len(idx.leaves)==12
    assert idx.header_bytes==index_bytes([64,32,64],3,7,[l.length for l in idx.leaves])
    np.testing.assert_array_equal(np.frombuffer(Delta3D().decode(f),np.uint8),raw.ravel())
    for l in idx.leaves:
        z,y,x=l.origin;dz,dy,dx=l.shape
        expected=a[l.channel,z:z+dz,y:y+dy,x:x+dx].tobytes()
        assert decode_indexed_leaf(f[l.offset:l.offset+l.length],l)==expected

@pytest.mark.parametrize('plan',[
    [(8,1),(4,0)],[(8,3),(4,0)],[(5,2),(4,0)],
    [(8,2)],[(9,2),(4,0)]])
def test_invalid_repeats(plan):
    with pytest.raises(ValueError):compress_indexed(bytes(2*32**3),32,32,32,2,False,'fast',plan)

def test_repeat_256_and_eight_channel_overhead():
    f=compress_indexed(bytes(256*32**3),32,32,32,256)
    assert f[:4]==b'NGI1'
    idx=read_index(f,len(f));assert len(idx.leaves)==256
    assert idx.header_bytes==index_bytes([32]*3,256,4,[l.length for l in idx.leaves])
    f=compress_indexed(bytes(8*32**3),32,32,32,8)
    idx=read_index(f,len(f))
    assert idx.header_bytes==index_bytes([32]*3,8,3,[l.length for l in idx.leaves])

def test_expanded_root_dimensions_and_large_header():
    f=compress_indexed(bytes(257*32**3),8224,32,32,1,False,'fast',[(5,257),(4,0)])
    assert f[:4]==b'NGI1'
    idx=read_index(f,len(f));assert idx.shape==[8224,32,32] and len(idx.leaves)==257
    assert decode_indexed_leaf(f[idx.leaves[-1].offset:],idx.leaves[-1])==bytes(32**3)
    # Large index, but no large payload allocation: 1024 channels x8 z-leaves.
    h=raw_index([256,32,32],1024,b'\x07'+varint(1023)+b'\x04\x07\x08',[2**35]*8192)
    assert len(h)>32768
    idx=read_index(h,len(h)+8192*2**35);assert len(idx.leaves)==8192
    assert idx.leaves[-1].origin==[224,0,0] and idx.leaves[-1].channel==1023

def test_repeat_truncation_and_expansion_bounds():
    f=compress_indexed(bytes(2*64*32*32),64,32,32,2,False,'fast',[(8,2),(5,2),(4,0)])
    hn=indexed_header_size(f)
    for cut in range(hn):
        with pytest.raises(ValueError):read_index(f[:cut],len(f))
    # Claim far more repeats than the physically present length table permits.
    h=bytearray(f[:hn]);h[10]=255;h[12]=255
    with pytest.raises(ValueError):read_index(bytes(h),len(f))

def test_retired_container_magics_rejected():
    f=compress_indexed(bytes(2*32**3),32,32,32,2)
    for magic in (b'NGC1',b'NGX1'):
        old=magic+f[4:]
        assert indexed_header_size(old) is None
        with pytest.raises(ValueError):read_index(old,len(old))
        with pytest.raises((ValueError,RuntimeError)):Delta3D().decode(old)
