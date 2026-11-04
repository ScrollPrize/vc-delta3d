"""Paired native decode timings, alternating builds on cached identical frames.
Python performs orchestration and verification only, never codec algorithms.
"""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--before',type=Path,required=True)
p.add_argument('--after',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--frames',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True,help='JSONL records with group/preset/sha256/encoded_hash')
p.add_argument('--raw-dir',type=Path,required=True,help='Directory containing <sha256>.raw chunks')
p.add_argument('--export',action='store_true')
p.add_argument('--repetitions',type=int,default=8)
p.add_argument('--cpu',type=int,help='Optional Linux CPU affinity')
p.add_argument('--group',help='Optional isolation subset; final validation uses all groups')
p.add_argument('--preset',choices=['fast','medium','slow','slower'],help='Isolate a preset when checking a borderline regression')
p.add_argument('--encode',action='store_true',help='Alternate native encoder calls instead of decoder calls')
a=p.parse_args()
package=Path(__file__).resolve().parents[1]/'python/vc_delta3d'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(path.resolve()),str(package)])
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
rows=[json.loads(s) for s in a.manifest.read_text().splitlines()]
if a.group:rows=[r for r in rows if r['group']==a.group]
if a.preset:rows=[r for r in rows if r['preset']==a.preset]
def frame_path(r):return a.frames/(r['encoded_hash']+'.bin')
def raw(r):return (a.raw_dir/(r['sha256']+'.raw')).read_bytes()
if a.export:
    before=load('codec_before',a.before)
    a.frames.mkdir(parents=True,exist_ok=True)
    def export(r):
        path=frame_path(r)
        if path.exists():return
        frame=before.Delta3D(predictor=r['preset']).encode(np.frombuffer(raw(r),np.uint8).reshape((128,)*3))
        assert hashlib.sha256(frame).hexdigest()==r['encoded_hash'];path.write_bytes(frame)
    with ThreadPoolExecutor(8) as pool:list(pool.map(export,rows))
    print(f'{len(rows)} frames exported',flush=True)
else:
    # Load the C API without initializing two nanobind modules/types.
    libraries=[ctypes.CDLL(str(next(path.glob('_codec*.so')))) for path in [a.before,a.after]]
    decoders=[lib.vc_decompress for lib in libraries]
    for fn in decoders:
        fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
        fn.restype=ctypes.c_int
    class Options(ctypes.Structure):
        _fields_=[('shape',ctypes.c_uint32*3),('channels',ctypes.c_uint32),
                  ('layout',ctypes.c_uint),('element_size',ctypes.c_uint),
                  ('quantization',ctypes.c_uint),('preset',ctypes.c_int)]
    encoders=[lib.vc_compress for lib in libraries]
    for lib,fn in zip(libraries,encoders):
        fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(Options),
                     ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_size_t)]
        fn.restype=ctypes.c_int
        lib.vc_free.argtypes=[ctypes.c_void_p]
    if a.cpu is not None:os.sched_setaffinity(0,{a.cpu})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for r in rows:
            frame=frame_path(r).read_bytes();expected=raw(r)
            times=[[],[]];out=ctypes.create_string_buffer(len(expected))
            for decoder in decoders:assert decoder(frame,len(frame),out,len(expected)) and out.raw==expected
            options=Options((128,128,128),0,0,1,1,3+['fast','medium','slow','slower'].index(r['preset']))
            started=time.monotonic()
            for rep in range(a.repetitions):
                for k in ([0,1] if rep%2==0 else [1,0]):
                    if a.encode:
                        ptr=ctypes.c_void_p();size=ctypes.c_size_t()
                        t=time.perf_counter();ok=encoders[k](expected,len(expected),ctypes.byref(options),ctypes.byref(ptr),ctypes.byref(size));times[k].append(time.perf_counter()-t)
                        assert ok
                        for decoder in decoders:assert decoder(ptr,size.value,out,len(expected)) and out.raw==expected
                        libraries[k].vc_free(ptr)
                    else:
                        t=time.perf_counter();ok=decoders[k](frame,len(frame),out,len(expected));times[k].append(time.perf_counter()-t)
                        assert ok and out.raw==expected
            f.write(json.dumps(dict(group=r['group'],preset=r['preset'],sha256=r['sha256'],before=times[0],after=times[1],started=started,finished=time.monotonic()))+'\n');f.flush()
    data=[json.loads(s) for s in a.output.read_text().splitlines()]
    for preset in ['fast','medium','slow','slower']:
        rs=[r for r in data if r['preset']==preset]
        if not rs:continue
        speeds=[2*len(rs)/sum(statistics.median(r[k]) for r in rs) for k in ['before','after']]
        print(preset,*(round(x,1) for x in speeds),f'{100*(speeds[1]/speeds[0]-1):+.2f}%')
