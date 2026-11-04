"""Aggregate complete-frame results from check_core_cleanup/check_controls."""
import argparse
import json
from pathlib import Path
import statistics as st

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--presets', type=Path, required=True)
p.add_argument('--controls', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--decode-manifest', type=Path,
               help='Optional isolated native decode timings from check_decode_cleanup')
a = p.parse_args()
rows = [json.loads(line) for path in (a.presets, a.controls)
        for line in path.read_text().splitlines()]
if a.decode_manifest:
    dec = {(r['sha256'], r['preset']): r['before'] + r['after']
           for r in map(json.loads, a.decode_manifest.read_text().splitlines())}
    for r in rows:
        if (r['sha256'], r['preset']) in dec:
            r['decode_samples'] = dec[r['sha256'], r['preset']]
names = ['Delta3D', 'Original NG', 'LPC', 'Blosc-LZ4', 'Blosc-Zstd',
         'Zstd', 'fast', 'medium', 'slow', 'slower']
groups = list(dict.fromkeys(r['group'] for r in rows))
assert len(groups) == 18
assert len(rows) == 18 * 8 * len(names)
assert len({(r['group'], r['sha256'], r['preset']) for r in rows}) == len(rows)
lines = ['# Complete-frame benchmark', '',
         '144 hash-checked uint8 128³ chunks; 18 groups × 8 chunks. '
         'Three native encode/decode repetitions, one core per codec. '
         'Sizes include headers and selector/residual streams. '
         'Each channel is encoded independently (no additional multi-channel index).', '',
         'Combined predictions includes all 16 non-CT groups: fiber and sheet '
         'channels, including the gradient mask, at L3/L4. '
         'Rates are total raw bytes / total time; central rates use the median '
         'time per chunk. Mean/min/median/max repeat rates are also given.', '']
if a.decode_manifest:
    lines += ['Preset decode timings are a separate full-frame native C API '
              'pass (16 calls per chunk, preallocated output); encode timings '
              'include allocation and Python binding dispatch. Controls use '
              'their native fixture APIs, including their normal wrapper '
              'allocation/copy costs, or numcodecs. These API differences '
              'are not subtracted from reported time.', '']
for group in ['CT-L0', 'Combined predictions'] + [g for g in groups if g != 'CT-L0'] + ['All']:
    lines += ['## ' + group, '',
              '| Codec | Size % raw | Encode MiB/s | Decode MiB/s |',
              '|---|---:|---:|---:|']
    selected = {}
    for name in names:
        rr = [r for r in rows if r['preset'] == name and
              (group == 'All' or (not r['group'].startswith('CT-') if
               group == 'Combined predictions' else r['group'] == group))]
        selected[name] = rr
        raw = len(rr) * 128**3
        speed = [raw / 2**20 / sum(st.median(r[k]) for r in rr)
                 for k in ['encode_samples', 'decode_samples']]
        lines.append(f'| {name} | {100*sum(r["bytes"] for r in rr)/raw:.3f} | '
                     f'{speed[0]:.1f} | {speed[1]:.1f} |')
    lines += ['', '| Codec | Phase | Mean-time MiB/s | Repeat min | Repeat median | Repeat max |',
              '|---|---|---:|---:|---:|---:|']
    for name, rr in selected.items():
        mib = len(rr)*2
        for phase in ['encode', 'decode']:
            key = phase + '_samples'
            rates = [mib/sum(r[key][i] for r in rr)
                     for i in range(min(len(r[key]) for r in rr))]
            mean = mib/sum(st.mean(r[key]) for r in rr)
            lines.append(f'| {name} | {phase} | {mean:.1f} | {min(rates):.1f} | '
                         f'{st.median(rates):.1f} | {max(rates):.1f} |')
    lines.append('')
a.output.write_text('\n'.join(lines))
print(a.output)
