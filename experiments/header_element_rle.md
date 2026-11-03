# Generic element RLE before/after delta

2026-11-02. Native C experiment; no production changes or new default.

## Matched complete sizes

Same 576 frozen headers (18 groups × eight chunks × four presets), same
typed logical representation and 40-byte directory as variant 93. Each new
mode adds candidates to that baseline, keeping its existing choices. Sizes
include every tag, encoded token length, rANS table and state. Direct-rANS
means fixed-width RLE tokens are entropy-coded without integer compaction;
it does not disable compaction in the baseline alternatives.

| New candidates available | Mean header bytes | Saved vs typed baseline | Headers improved / equal / worse |
|---|---:|---:|---:|
| Zstd control | 6173.17 | — | — |
| Current simplified word-delta control | 6172.32 | — | — |
| Typed baseline (93) | 6585.67 | 0 | 0 / 576 / 0 |
| Optional delta → element RLE → direct rANS (94) | 6585.67 | 0 | 0 / 576 / 0 |
| Optional delta → element RLE → integer compaction → rANS (95) | 6552.65 | 33.02 | 464 / 112 / 0 |
| Element RLE → value delta → direct rANS (96) | 6585.67 | 0 | 0 / 576 / 0 |
| Element RLE → value delta → integer compaction → rANS (97) | 6553.21 | 32.46 | 179 / 397 / 0 |
| All above candidates (98) | **6532.01** | **53.66** | **481 / 95 / 0** |

The typed baseline reproduces all 576 earlier sizes and family-size vectors
exactly. No new variant increases any header size because the baseline is
still available. These guarantees apply relative to the typed baseline, not
relative to the smaller old simplified representation.

Element RLE helps but does not close the gap: combined remains 359.69 bytes
larger than the simplified control (+0.05267% of complete compressed data).
Neither direct-rANS RLE candidate wins on any family. Do not promote the typed
prototype as a better compression default.

## Generic implementation

`header_typed.h` implements one RLE token format for 8/16/32-bit elements.
A 32-bit tag contains `(count << 1) | repeated`. A repeated run stores one
element; a literal run stores `count` elements. Runs of three or more equal
elements are repeated runs. There is no zero-specific or probability-table-
specific operation: zeros, including a trailing zero region, are ordinary
repeated values. Runs can span record boundaries within a family.

- Direct-rANS option: tags remain uint32 and values retain their element
  width. The byte stream is sent to the existing shared native rANS coder.
- Integer-compaction option: tags use unsigned LEB128, values use either
  unsigned or zigzag LEB128; the resulting stream is then rANS-coded.
- Delta-before-RLE can use the existing plain, within-record, previous-record,
  first-record and combined transforms; delta is optional.
- RLE-before-delta tokenizes plain values first. Only stored values are
  differenced, modulo their width, against the previous stored value;
  run-length tags are not differenced. State starts at zero per family.
  This branch does not add reference subtraction between variable token lists.

The no-element-RLE paths remain in the encoder's choice set. This is an
additive comparison, not a forced-RLE size claim. Existing byte-level RLE
alternatives are retained too, unchanged.

## What is chosen and where bytes go

Combined candidate, mean bytes/header:

| Family | Before | With element RLE | New share |
|---|---:|---:|---:|
| Controls/descriptors | 111.14 | 111.14 | 1.70% |
| Parameters | 107.01 | 107.01 | 1.64% |
| Frequency counts | 2376.88 | 2323.24 | 35.57% |
| Selectors | 3950.63 | 3950.61 | 60.48% |
| Directory | 40.00 | 40.00 | 0.61% |
| **Total** | **6585.67** | **6532.01** | **100%** |

New frequency choices: 226 headers use RLE + unsigned compaction after
literal/first-reference input, 97 use RLE + zigzag after reference/within
transforms, and 158 use RLE then value delta + zigzag. One selector family
also benefits from element RLE. Controls and parameters retain their prior
choices. All new selected paths include integer compaction.

Header-compressor frequency tables: 411.89 → 415.44 bytes; rANS states:
112.67 → 113.44 bytes. The savings therefore come from the coded data rather
than reduced metadata overhead.

## Reproduce

Build command is the same as `header_typed_fields.md`, then:

```sh
/data/hendrik/compression_experiments/villa4/builds/header_codecs --rle-self-test
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_codecs \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_element_rle_20261102 \
  --repeats 1 --cpu 97 --variants 0,88,93,94,95,96,97,98
```

Every native header is reconstructed byte-for-byte before timing; residual
payloads never change. The 504 synthetic round-trips cover element widths
1/2/4, empty inputs, short inputs, repeated zeros/maxima, mixed runs, and
wrapping pseudo-random values for all six new coding paths.

ASan/UBSan also passed the 504 synthetic tests and all five new variants on
all 576 headers (2,880 exact full-header round-trips). Use the sanitizer build
flags/library documented in `header_simplification.md`, run with
`ulimit -s 32768`, and the same command above with the sanitizer binary,
`--cpu 98 --variants 94,95,96,97,98` and output directory
`header_element_rle_sanitize_20261102`. `git diff --check` passes.

Native release build: C11 `-O3 -ffp-contract=off`, CPU 97, three batches × one
repeat per header. Exploratory header-only mean median encode/decode times:
typed baseline 3183/155 µs; combined 5626/141 µs. These include conversion
back to the legacy header for validation; not whole-codec throughput claims.
All timing samples and distribution summaries are in results.jsonl/report.md.

Results directory:
`/data/hendrik/compression_experiments/villa4/encoder_speed/header_element_rle_20261102/`.
