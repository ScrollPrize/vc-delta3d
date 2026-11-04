"""Frozen-corpus native header comparison; Python does no codec work."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import statistics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--binary',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    rows=[r for r in map(json.loads,a.reference.open()) if r['codec']=='choose-unified-context-lz']
    baseline={r['frame_path']:r for r in map(json.loads,a.baseline.open())} if a.baseline else {}
    def run(r):
        x=json.loads(subprocess.check_output([a.binary,r['frame_path']],text=True))
        assert x['selector_bytes']==r['payload_bytes'],r['frame_path']
        if baseline: assert x['voxel_hash']==baseline[r['frame_path']]['voxel_hash'], r['frame_path']
        x.update({k:r[k] for k in ('frame_path','group','preset','sha256')})
        return x
    a.output.mkdir(parents=True,exist_ok=True)
    results=[]
    with ThreadPoolExecutor(8) as pool,(a.output/'results.jsonl').open('w') as f:
        for i,r in enumerate(pool.map(run,rows)):
            results.append(r);f.write(json.dumps(r)+'\n');f.flush()
            if (i+1)%64==0: print(f'{i+1}/{len(rows)} exact header round trips',flush=True)
    lines=['|Group|Compact header B|Zstd same header B|Difference B|','|---|---:|---:|---:|']
    for group in list(dict.fromkeys(r['group'] for r in results))+['Combined']:
        sub=[r for r in results if group=='Combined' or r['group']==group]
        c=statistics.mean(r['compact_header'] for r in sub);z=statistics.mean(r['zstd_header'] for r in sub)
        lines.append(f'|{group}|{c:.3f}|{z:.3f}|{c-z:+.3f}|')
    report='\n'.join(lines)+'\n';(a.output/'report.md').write_text(report);print(report)

if __name__=='__main__':main()
