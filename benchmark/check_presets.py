"""Format-change regression: separate old-data export from new native encodes.
The old decoder is loaded only in the explicit export process, never linked
into the current library. Python performs I/O/timing/hashes, not compression.
"""
import argparse
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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export',action='store_true')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rawdir=a.output/'raw';rawdir.mkdir(exist_ok=True)
    package=Path(__file__).resolve().parents[1]/'python/vc_delta3d'
    spec=importlib.util.spec_from_file_location('vc_delta3d',package/'__init__.py',submodule_search_locations=[str(a.build.resolve()),str(package)])
    codec=importlib.util.module_from_spec(spec);sys.modules['vc_delta3d']=codec;spec.loader.exec_module(codec)
    names={'Fast':'fast','Slow':'medium','RC1':'slow','RC2':'slower'}
    rows=[r for r in json.loads(a.manifest.read_text()) if r['name'] in names]
    if a.export:
        seen=set()
        for r in rows:
            if r['sha256'] in seen:continue
            seen.add(r['sha256']);path=rawdir/(r['sha256']+'.raw')
            if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];continue
            raw=codec.Delta3D().decode(Path(r['frame_path']).read_bytes())
            assert hashlib.sha256(raw).hexdigest()==r['sha256'];path.write_bytes(raw)
        print(f'{len(seen)} exact source chunks available');return
    def run(r):
        raw=(rawdir/(r['sha256']+'.raw')).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
        arr=np.frombuffer(raw,np.uint8).reshape((128,)*3);c=codec.Delta3D(predictor=names[r['name']])
        enc=[];dec=[];encoded=None
        for _ in range(3):
            t=time.perf_counter();f=c.encode(arr);enc.append(time.perf_counter()-t)
            if encoded is not None:assert encoded==f
            encoded=f;t=time.perf_counter();out=c.decode(f);dec.append(time.perf_counter()-t);assert out==raw
        return dict(group=r['group'],preset=names[r['name']],sha256=r['sha256'],old_bytes=Path(r['frame_path']).stat().st_size,
                    new_bytes=len(encoded),raw_bytes=len(raw),encode=enc,decode=dec,load=os.getloadavg())
    results=[]
    with ThreadPoolExecutor(a.workers) as pool,(a.output/'results.jsonl').open('w') as f:
        for i,r in enumerate(pool.map(run,rows)):
            results.append(r);f.write(json.dumps(r)+'\n');f.flush()
            if (i+1)%32==0:print(f'{i+1}/{len(rows)} deterministic native encode/decode checks',flush=True)
    lines=['|Group|Preset|Old % raw|New % raw|Encode MiB/s|Decode MiB/s|','|---|---|---:|---:|---:|---:|']
    for g in list(dict.fromkeys(r['group'] for r in results))+['Combined']:
        for p in names.values():
            rs=[r for r in results if r['preset']==p and (r['group']==g or g=='Combined')];n=sum(r['raw_bytes'] for r in rs)
            if not n:continue
            old=100*sum(r['old_bytes'] for r in rs)/n;new=100*sum(r['new_bytes'] for r in rs)/n
            e=n/1048576/sum(statistics.median(r['encode']) for r in rs);d=n/1048576/sum(statistics.median(r['decode']) for r in rs)
            lines.append(f'|{g}|{p}|{old:.4f}|{new:.4f}|{e:.1f}|{d:.1f}|')
    report='\n'.join(lines)+'\n';(a.output/'report.md').write_text(report);print(report)
if __name__=='__main__':main()
