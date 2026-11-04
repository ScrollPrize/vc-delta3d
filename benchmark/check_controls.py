"""Time full native baseline frames on an explicit hash-checked raw corpus.

The supplied control library exposes codec_encode_buffer/codec_decode_buffer
for legacy Delta3D, LPC and original NG (IDs 0/1/2). It is a benchmark fixture,
not a production dependency. Python only orchestrates native codec calls.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import time

from numcodecs import Blosc, Zstd
from numcodecs.blosc import set_nthreads

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--raw-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cpu', type=int)
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--decode-repetitions', type=int, default=1,
               help='Repeated full-frame decodes per timing sample')
a = p.parse_args()
if a.cpu is not None:
    os.sched_setaffinity(0, {a.cpu})
set_nthreads(1)
lib = ct.CDLL(str(a.library.resolve()))
lib.codec_encode_buffer.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int,
                                   ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t)]
lib.codec_decode_buffer.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int,
                                   ct.c_void_p, ct.c_size_t]
standard = {
    'Blosc-LZ4': Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
    'Blosc-Zstd': Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE),
    'Zstd': Zstd(level=3),
}
names = ['Delta3D', 'LPC', 'Original NG', *standard]
rows = [r for r in json.loads(a.manifest.read_text()) if r['name'] == 'Fast']
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('w') as output:
    for i, r in enumerate(rows):
        raw = (a.raw_dir / (r['sha256'] + '.raw')).read_bytes()
        assert len(raw) == 128**3 and hashlib.sha256(raw).hexdigest() == r['sha256']
        dst = ct.create_string_buffer(len(raw))
        frame = ct.create_string_buffer(len(raw) * 4 + 131072)
        n = ct.c_size_t()
        for name in names[i % len(names):] + names[:i % len(names)]:
            enc, dec = [], []
            digest = None
            for repeat in range(-1, a.repetitions):
                start = time.perf_counter()
                if name in standard:
                    payload = standard[name].encode(raw)
                    n.value = len(payload)
                else:
                    assert lib.codec_encode_buffer(raw, names.index(name), 128, 0,
                                                   frame, len(frame), ct.byref(n)) == 0
                elapsed = time.perf_counter() - start
                if repeat >= 0:
                    enc.append(elapsed)
                if name not in standard:
                    payload = frame.raw[:n.value]
                new_digest = hashlib.sha256(payload).hexdigest()
                assert digest is None or digest == new_digest
                digest = new_digest
                start = time.perf_counter()
                for _ in range(a.decode_repetitions):
                    if name in standard:
                        restored = standard[name].decode(payload)
                    else:
                        assert lib.codec_decode_buffer(frame, n.value, names.index(name),
                                                       dst, len(raw)) == 0
                elapsed = (time.perf_counter() - start) / a.decode_repetitions
                if repeat >= 0:
                    dec.append(elapsed)
                assert (restored if name in standard else dst.raw) == raw
            output.write(json.dumps(dict(group=r['group'], preset=name,
                sha256=r['sha256'], bytes=n.value, raw_bytes=len(raw),
                encode_samples=enc, decode_samples=dec)) + '\n')
            output.flush()
        if (i + 1) % 8 == 0:
            print(f'{i + 1}/{len(rows)} chunks: six full-frame controls verified', flush=True)
