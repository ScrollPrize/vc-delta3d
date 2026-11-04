"""Aggregate paired cleanup measurements; MiB/s from summed native times."""
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1])
before=[json.loads(s) for s in (root/'speed_before.jsonl').read_text().splitlines()]
after=[json.loads(s) for s in (root/'speed_after.jsonl').read_text().splitlines()]
assert len(before)==len(after)==576
key=lambda r:(r['group'],r['preset'],r['sha256'])
before={key(r):r for r in before};after={key(r):r for r in after}
assert before.keys()==after.keys()
assert all(before[k]['encoded_hash']==after[k]['encoded_hash'] for k in before)
def stats(rows,kind):
    size=2*len(rows)
    rates=[size/sum(r[kind+'_samples'][i] for r in rows) for i in range(3)]
    return dict(mean=statistics.harmonic_mean(rates),median=statistics.median(rates),min=min(rates),max=max(rates))
result=[]
for group in ['combined']+sorted({k[0] for k in before}):
    for preset in ['fast','medium','slow','slower']:
        keys=[k for k in before if k[1]==preset and (group=='combined' or k[0]==group)]
        r=dict(group=group,preset=preset)
        for label,data in [('before',before),('after',after)]:
            for kind in ['encode','decode']:r[label+'_'+kind]=stats([data[k] for k in keys],kind)
        result.append(r)
(root/'timing_summary.json').write_text(json.dumps(result,indent=2)+'\n')
for group in ['combined']+sorted({r['group'] for r in result if 'ct' in r['group'].lower()}):
    print(group)
    print('| Preset | Encode before | Encode after | Decode before | Decode after | Decode change |')
    print('|---|---:|---:|---:|---:|---:|')
    for r in result:
        if r['group']!=group:continue
        values=[r[x]['mean'] for x in ['before_encode','after_encode','before_decode','after_decode']]
        print('| '+r['preset']+' | '+' | '.join(f'{v:.1f}' for v in values)+f' | {(values[3]/values[2]-1)*100:+.1f}% |')
print('Decode aggregate min/median/max MiB/s:')
for r in result:
    if r['group']=='combined':
        print(r['preset'],[(label,*(round(r[label+'_decode'][k],1) for k in ['min','median','max'])) for label in ['before','after']])
