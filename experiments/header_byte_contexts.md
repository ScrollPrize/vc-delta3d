# Separate low/high byte contexts

2026-11-02. Native C experiment on the current simplified word-reference
candidate (88), preserving all other choices. Production unchanged.

## Actual encoded sizes

Same 576 frozen headers, 18 groups × eight chunks × four presets:

| Variant | Mean header bytes | Own rANS table bytes | Own rANS states |
|---|---:|---:|---:|
| Zstd control | 6173.17 | — | — |
| Current shared-byte candidate (88) | 6172.32 | 433.16 | 108.89 |
| Optional low/high contexts (89) | 6172.32 | 433.16 | 108.89 |
| Forced split rANS candidates (90) | 6482.43 | 612.92 | 155.33 |

The optional split never wins: all 576 headers retain the existing choices.
The forced diagnostic is 310.11 bytes/header larger. Its table and state
overhead increases by 179.76 and 46.44 bytes respectively. The remaining
change also includes payload/choice effects; do not attribute it solely to
the context distributions.

The implementation uses two existing native rANS streams, one for each byte
position, including both serialized tables and eight-lane states. Section
candidates cover raw/reference-transformed words and within-word delta.
Probability-table groups reset byte parity at every record boundary, including
odd lengths. All original non-split choices remain in variant 89, with complete
signaling counted. Variant 90 removes the unsplit plain/delta-rANS candidates
but retains raw/RLE/bit-packing and the existing RLE→rANS candidate; it is a
diagnostic, not a recommended configuration.

Two streams are a simple prototype, not a requirement of context coding:
an integrated rANS stream could share states while switching frequency tables.
Fixed embedded tables would eliminate transmitted table overhead, so these
results do not settle the fixed-table option.

## Held-out transfer diagnostic (not encoded fixed-table results)

Native decode records actual split-stream byte histograms. Fit two global
distributions on four chunks per group and test on the other four, then swap.
All presets of a source chunk stay in the same fold. Add-one smoothing supports
unseen bytes. No Python compression is performed; the Python script only
calculates cross-entropy from the native histograms.

| Fold | Test headers | Shared fixed bits/byte | Split fixed bits/byte | Split saving bytes/header | Split mismatch vs test-pooled distribution bytes/header |
|---|---:|---:|---:|---:|---:|
| 1 | 288 | 4.0161 | 4.0096 | 3.99 | 14.79 |
| 2 | 288 | 3.9242 | 3.9139 | 6.82 | 15.29 |

On these selected streams, globally pooled low/high separation contributes
only about 4–7 theoretical bytes/header compared with a single globally pooled
distribution. These are unquantized cross-entropy diagnostics, NOT actual
fixed-rANS sizes or savings relative to the current adaptive codec. They do
not account for new candidate choices under fixed tables. No fixed tables
have been embedded or recommended based on this diagnostic.

## Reproduction and verification

Build `header_codecs` using the command in `header_simplification.md`, then:

```sh
python3 benchmark/compare_headers.py \
  --binary /data/hendrik/compression_experiments/villa4/builds/header_codecs \
  --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_byte_contexts_forced_20261102 \
  --repeats 2 --cpu 97 --variants 0,88,89,90
python3 benchmark/header_context_transfer.py \
  /data/hendrik/compression_experiments/villa4/encoder_speed/header_byte_contexts_forced_20261102/results.jsonl \
  --output /data/hendrik/compression_experiments/villa4/encoder_speed/header_byte_contexts_forced_20261102/transfer.md
```

Release build: C11 `-O3 -ffp-contract=off`, shared native rANS library, three
timing batches per frame. Every frame round-trips exactly. Variants 88/89/90
also pass all 576 headers under ASan/UBSan (one repeat per batch, CPU 98):
`header_byte_contexts_sanitize_20261102`. Native timings and per-channel sizes
are in the run JSONL/report; this experiment makes no full-codec speed claim.
