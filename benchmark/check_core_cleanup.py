"""Native encode/decode byte-parity check over all 18 groups and four presets."""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--build',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--compare',type=Path)
p.add_argument('--workers',type=int,default=8)
p.add_argument('--repetitions',type=int,default=1)
p.add_argument('--cpu',type=int)
p.add_argument('--compatible-build',type=Path,help='Old decoder for new frames; allow header-only changes')
p.add_argument('--frames',type=Path,help='Cached previous frames for payload comparison')
a=p.parse_args()
if a.cpu is not None:
    os.sched_setaffinity(0,{a.cpu})
root=Path('/data/hendrik/compression_experiments/villa4')
package=Path(__file__).resolve().parents[1]/'python/vc_delta3d'
s=importlib.util.spec_from_file_location('vc_delta3d',package/'__init__.py',submodule_search_locations=[str(a.build.resolve()),str(package)])
codec=importlib.util.module_from_spec(s);sys.modules['vc_delta3d']=codec;s.loader.exec_module(codec)
rows=json.loads((root/'encoder_speed/release_fixed_presets_20261101/results.json').read_text())
names={'Fast':'fast','Slow':'medium','RC1':'slow','RC2':'slower'}
rows=[r for r in rows if r['name'] in names]
previous={(r['sha256'],r['preset']):r for r in map(json.loads,a.compare.open())} if a.compare else {}
old_decode=None
if a.compatible_build:
    assert a.compare and a.frames
    library=ctypes.CDLL(str(next(a.compatible_build.glob('_codec*.so'))))
    old_decode=library.vc_decompress
    old_decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
    old_decode.restype=ctypes.c_int
def run(r):
    raw=(root/'encoder_speed/header_clean_presets_20261103/raw'/(r['sha256']+'.raw')).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==r['sha256']
    x=np.frombuffer(raw,np.uint8).reshape((128,)*3);c=codec.Delta3D(predictor=names[r['name']])
    enc=[];dec=[];first=None
    for _ in range(a.repetitions):
        t=time.perf_counter();f=c.encode(x);enc.append(time.perf_counter()-t)
        t=time.perf_counter();out=c.decode(f);dec.append(time.perf_counter()-t)
        assert out==raw
        if first is not None: assert f==first
        first=f
    result=dict(group=r['group'],preset=names[r['name']],sha256=r['sha256'],encoded_hash=hashlib.sha256(f).hexdigest(),bytes=len(f),encode=sum(enc)/len(enc),decode=sum(dec)/len(dec),encode_samples=enc,decode_samples=dec,load=os.getloadavg())
    if previous:
        old=previous[r['sha256'],result['preset']]
        if old_decode:
            original=(a.frames/(old['encoded_hash']+'.bin')).read_bytes()
            assert c.decode(original)==raw
            output=ctypes.create_string_buffer(len(raw))
            assert old_decode(f,len(f),output,len(raw)) and output.raw==raw
            # NCF1 compressed-header length: everything after it is unchanged.
            assert original[:4]==f[:4]==b'NCF1'
            assert original[8+int.from_bytes(original[4:8],'little'):]==f[8+int.from_bytes(f[4:8],'little'):],result
            result['header_delta']=len(f)-len(original)
        else:
            assert result['encoded_hash']==old['encoded_hash'] and result['bytes']==old['bytes'],result
    return result
a.output.parent.mkdir(parents=True,exist_ok=True)
with ThreadPoolExecutor(a.workers) as pool,a.output.open('w') as f:
    for i,r in enumerate(pool.map(run,rows)):
        f.write(json.dumps(r)+'\n');f.flush()
        if (i+1)%32==0:print(f'{i+1}/{len(rows)} lossless'+(' and cross-compatible, payload-identical' if old_decode else ' and byte-identical' if previous else ''),flush=True)
