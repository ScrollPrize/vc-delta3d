"""Orchestrate native header tests; no entropy coding in Python."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True, help='JSONL frame records: variant, frame_path, group, preset')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--header-export', type=Path, required=True)
p.add_argument('--header-static', type=Path, required=True)
a = p.parse_args()
OUT = a.output
OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(s) for s in a.manifest.read_text().splitlines()]
rows = [r for r in rows if r['variant'] == rows[0]['variant']]
presets = ['fast', 'medium', 'slow', 'slower']
# Chunk hash, not preset, determines fold. Balance 4/4 within each channel.
groups = {}
for r in rows:
    r['chunk'] = Path(r['frame_path']).name.split('_')[1]
    groups.setdefault(r['group'], set()).add(r['chunk'])
folds = {(g, c): i % 2 for g, cs in groups.items() for i, c in enumerate(sorted(cs))}

def export(pair):
    i, r = pair
    path = OUT / f'{i}.header'
    p = subprocess.run([str(a.header_export.resolve()), r['frame_path']],
                       env={**os.environ, 'HEADER_EXPORT': str(path)}, capture_output=True, text=True, check=True)
    predictor = int(p.stderr.strip().split('=')[-1])
    r.update(index=i, fold=folds[r['group'], r['chunk']], predictor=predictor)
    return f'{path} {presets.index(r["preset"])} {predictor} {r["fold"]} {sorted(groups).index(r["group"])}\n'

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    manifest = list(ex.map(export, enumerate(rows)))
(OUT / 'manifest.txt').write_text(''.join(manifest))
(OUT / 'inputs.json').write_text(json.dumps(rows, indent=2))
p = subprocess.run([str(a.header_static.resolve()), str(OUT / 'manifest.txt')], capture_output=True, text=True, check=True)
(OUT / 'results.jsonl').write_text(p.stdout)
results = [json.loads(s) for s in p.stdout.splitlines()]
names = ['global', 'preset', 'dominant predictor', 'preset + dominant predictor']
lines = ['# Cross-validated static header tables', '',
         '576 frames; 18 groups × 8 chunks × 4 presets. Two-fold cross-validation, 4 training/4 test chunks per group; all encodings of a source chunk share a fold. Tables include a pseudocount of one for every symbol. Two transform-specific distributions per group. Grouped schemes charge one byte per coded header for table ID. Raw and adaptive fallback retain their existing mode byte. Sizes include actual rANS states/renormalization and stream-length varints; selectors excluded. Every trial is losslessly decoded.', '',
         '| Scope | Scheme | Current B | Fixed/raw B | Fixed/raw change | With adaptive fallback B | Worse fixed/raw |',
         '|---|---|---:|---:|---:|---:|---:|']
for scope in ['ALL'] + presets + sorted(groups):
    for mode, name in enumerate(names):
        rr = [r for r in results if r['fold'] < 2 and r['mode'] == mode and (scope == 'ALL' or rows[r['index']]['preset'] == scope or rows[r['index']]['group'] == scope)]
        n = len(rr)
        means = {k: sum(r[k] for r in rr)/n for k in ['baseline', 'fixed', 'hybrid']}
        lines.append(f'| {scope} | {name} | {means["baseline"]:.2f} | {means["fixed"]:.2f} | {100*(means["fixed"]/means["baseline"]-1):+.2f}% | {means["hybrid"]:.2f} | {sum(r["fixed"]>r["baseline"] for r in rr)}/{n} |')
lines += ['', '## Entire channel/scale held out', '', '| Scheme | Current B | Fixed/raw B | With adaptive fallback B |', '|---|---:|---:|---:|']
for mode,name in enumerate(names):
    rr=[r for r in results if r['fold']>=2 and r['mode']==mode]
    means=[sum(r[k] for r in rr)/len(rr) for k in ['baseline','fixed','hybrid']]
    lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' for v in means)+' |')
(OUT / 'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:25]))
print('\n'.join(lines[-8:]))
