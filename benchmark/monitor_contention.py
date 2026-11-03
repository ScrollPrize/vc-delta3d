"""Run a benchmark while logging Linux CPU/SMT and pressure counters.

Example: python benchmark/monitor_contention.py --output /data/run.jsonl \
    --cpu 64 -- python benchmark/check_decode_cleanup.py ...
This is measurement orchestration, not codec implementation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cpu', type=int, default=64)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command:
    p.error('benchmark command required')
siblings = []
for part in Path(f'/sys/devices/system/cpu/cpu{a.cpu}/topology/thread_siblings_list').read_text().strip().split(','):
    bounds = [int(x) for x in part.split('-')]
    siblings.extend(range(bounds[0], bounds[-1] + 1))
monitor_cpu = next(c for c in sorted(os.sched_getaffinity(0)) if c not in siblings)
# Keep the monitor off the measured physical core. The benchmark sets its own
# affinity explicitly; do not silently pin arbitrary supplied commands.
process = subprocess.Popen(command)
os.sched_setaffinity(0, {monitor_cpu})

def read_sample():
    cpus = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith('cpu'):
            cpus[fields[0]] = list(map(int, fields[1:9]))
    pressure = {kind: Path(f'/proc/pressure/{kind}').read_text().strip()
                for kind in ('cpu', 'memory', 'io')}
    result = dict(time=time.time(), monotonic=time.monotonic(), cpu=cpus,
                  pressure=pressure, load=Path('/proc/loadavg').read_text().strip())
    frequency = Path(f'/sys/devices/system/cpu/cpu{a.cpu}/cpufreq/scaling_cur_freq')
    if frequency.exists():
        result['frequency_khz'] = int(frequency.read_text())
    try:
        fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        result['process'] = dict(minflt=int(fields[7]), majflt=int(fields[9]),
                                 utime=int(fields[11]), stime=int(fields[12]),
                                 processor=int(fields[36]))
        result['switches'] = [line for line in Path(f'/proc/{process.pid}/status').read_text().splitlines()
                              if 'ctxt_switches' in line]
    except FileNotFoundError:
        pass
    return result

a.output.parent.mkdir(parents=True, exist_ok=True)
previous = None
with a.output.open('w') as f:
    f.write(json.dumps(dict(command=command, pid=process.pid, cpu=a.cpu,
                            siblings=siblings, monitor_cpu=monitor_cpu)) + '\n')
    while True:
        sample = read_sample()
        if previous:
            busy = {}
            for name, values in sample['cpu'].items():
                delta = [x-y for x,y in zip(values, previous['cpu'][name])]
                total = sum(delta)
                busy[name] = 100*(total-delta[3]-delta[4])/total if total else 0
            sample['busy_percent'] = busy
        f.write(json.dumps(sample) + '\n')
        f.flush()
        previous = sample
        if process.poll() is not None:
            break
        time.sleep(0.5)
raise SystemExit(process.wait())
