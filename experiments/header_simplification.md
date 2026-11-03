# Header option removal — 2026-11-02

Native C, same 576 frozen headers (18 groups × 8 chunks × 4 presets).
Complete encoded-header bytes, including all metadata; exact round-trip checks.
These are compression experiments, not production format changes. No timings
were collected (`--repeats 0`). Python only orchestrates the C implementation.

## Individual removals

Each row starts independently from log-range baseline 111; savings/costs cannot
be added because the remaining choices can substitute for each other.

| Removed option | Mean bytes/header | Increase |
|---|---:|---:|
| None (111) | 5975.0747 | — |
| Old compact-first representation (113) | 5981.0000 | 5.9253 |
| Local reference search (114) | 5977.2812 | 2.2066 |
| All RLE (115) | 6008.5556 | 33.4809 |
| Cross-family pooling (116) | 5976.7743 | 1.6997 |
| References in log labels (117) | 5977.2396 | 2.1649 |
| All log transforms except unsigned word delta (118) | 5980.5174 | 5.4427 |
| Word bit-packing (119) | 5977.4340 | 2.3594 |
| Separate low/high byte contexts (120) | 5975.0747 | 0 |
| Primitive zigzag varints (121) | 5975.4236 | 0.3490 |
| Primitive varint+rANS (122) | 5977.1476 | 2.0729 |
| Byte RLE only (123) | 5978.4913 | 3.4167 |

## Combined simplifications

| Configuration | Mean bytes/header |
|---|---:|
| Historical Zstd control | 6173.1684 |
| Old compact/pool baseline | 6088.5035 |
| All log-range alternatives (111) | 5975.0747 |
| Remove expensive alternative searches (124) | 6020.7899 |
| Also restrict primitive backend (125) | 6023.2049 |
| Log-range-only numerical compaction (126) | **6038.3924** |
| Log-range plus element RLE (127) | 6031.4045 |

Recommend **126 as the simpler experimental basis**: +63.3177 B/header
(1.060%) versus the full option set, but 134.7760 B/header (2.183%) smaller
than Zstd overall. This is an aggregate claim, not a win on every channel.
The increase versus 111 is about 0.0093% of complete compressed data.
Restoring RLE saves only 6.9878 B/header in this simplified setting; discard it.

126 removes old compact-first fixed/varint alternatives, all RLE, local
references, inter-family pooling, family-wide transforms, byte-delta log
transforms, word-packing, split byte contexts and varint+rANS alternatives.
It retains zero-tail skipping, exact log-range labels/bypass bits, optional
word delta and signed/unsigned mapping, and the small raw/constant/bit-pack/rANS
backend. Unsigned varints still serve controls/lengths, not a competing
numerical-array representation. Counts/known-size reconstruction are unchanged.

Actual log mode usage among 1152 numerical groups in 126:

| Mode | Selected groups |
|---|---:|
| Unsigned, no delta | 257 |
| Signed, no delta | 180 |
| Unsigned, word delta | 713 |
| Signed, word delta | 2 |

Signed mode was unused when the older alternatives remained available, but
becomes useful after removing them. Keep it; it shares the existing mapping.
Actual nested codec tags in 126 are only rANS, bit-pack, bit-pack+rANS,
control varints and log-range. Raw/constant fallbacks remain valid for other data.
Decoder telemetry now counts nested selected tags, not stale pre-pool choices.

Historical alternatives remain in the experiment harness for reproducibility;
126 does not search or emit them. This is not a claim that their historical
decoder code has been deleted from the repository, or that production has
switched to the new header format.

## Per-group comparison (bytes/header)

| Group | Simplified 126 | Zstd |
|---|---:|---:|
| CT L0 | 1184.72 | 932.97 |
| CT L1 | 1322.69 | 1225.75 |
| nx L3 | 7978.03 | 8200.81 |
| nx L4 | 6950.09 | 7283.34 |
| width L3 | 8539.88 | 8785.47 |
| width L4 | 4459.84 | 4389.78 |
| ny L3 | 7622.91 | 7828.25 |
| ny L4 | 6895.31 | 7326.03 |
| compound presence L3 | 7881.78 | 7842.97 |
| compound presence L4 | 5082.56 | 5146.28 |
| compound gradient L3 | 2638.88 | 2527.97 |
| compound gradient L4 | 3022.81 | 2973.88 |
| compound nx L3 | 8045.53 | 8259.91 |
| compound nx L4 | 7733.12 | 8102.78 |
| compound ny L3 | 8051.06 | 8277.94 |
| compound ny L4 | 7846.94 | 8203.28 |
| compound width L3 | 8162.75 | 8458.06 |
| compound width L4 | 5272.16 | 5351.56 |

## Reproduce

From the delta3d checkout:

```sh
cc -std=c11 -O3 -ffp-contract=off -Isrc benchmark/header_codecs.c \
  /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
  -lzstd -lm -pthread \
  -o /data/hendrik/compression_experiments/villa4/builds/header_simplify_combined
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_simplify_combined \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_simplify_combined_b_20261102 \
  --repeats 0 --cpu 100 --variants 126,127
```

Other saved runs under the same encoder_speed directory:
`header_simplify_log_{1,2,3,4}_20261102` for 111,113–123;
`header_simplify_combined_a_20261102` for 124/125.
Every variant has 576 exact round-trips. Baseline 111 matches saved sizes.
Release/ASan/UBSan synthetic self-test: 1230 cases; representative full-header
sanitizer checks: 18 groups × 4 combined variants. Production CTest: 4/4 pass.
Zstd is only used to read historical frames/control, not by the replacement.
