# Combining context and small LZ (2026-11-03)

## Scope / implementation

Same 576 frozen selector maps:18 groups ×8 chunks ×current fast/slow
(historical RC1) ×2³/4³ spatial predictor blocks. Native C encode/decode only.
No production defaults or public formats changed.

- Whole-map choice: run previous-symbol context and chain256 packed LZ; keep
  the smaller complete representation. Reuse its existing mode tag rather
  than wrapping it in a second frame. No extra overhead or double decoding.
- Contextual LZ literals: same LZ matches/sequence coding, but literals are
  independently coded by preceding reconstructed byte (including matched
  bytes). Test both raw IDs and packed bytes, using the new small-state coder
  and optional predefined distributions. Generic bank helpers are shared with
  the old context code through callbacks, not copied.
- Local switching: partition the serialized ID stream into regions of
  1024/4096/16384/65536 IDs. At each region choose among context, packed LZ,
  and raw/packed context-literal LZ by actual complete bytes. Regions have a
  mode and encoded-byte-length varint; decoded length is implied. Context
  history and LZ dictionary reset at each region. These are **1D ID regions**,
  not spatial cubes, and not changes to the underlying2³/4³ predictor blocks.
- Final experimental policy chooses the smallest complete whole-map or local
  representation. This is exhaustive and intentionally not fast.

## Combined mean selector bytes/chunk

Includes all states, tables, tags and region lengths. NOT full volume/chunk
compression ratios. Equal eight-chunk weighting for each of18 groups.

|Method|Fast2³|Fast4³|Slow2³|Slow4³|
|---|---:|---:|---:|---:|
|Packed Zstd3|19,887.41|1,779.34|39,540.81|4,789.95|
|Previous-symbol context|18,494.35|1,665.24|37,928.81|4,722.82|
|Chain256 packed LZ|20,503.74|1,869.34|40,292.28|5,026.15|
|Packed LZ + contextual literals|22,627.98|2,292.54|46,138.86|6,220.00|
|Raw LZ + contextual literals|25,009.07|2,229.34|48,669.66|6,012.24|
|**Whole-map context/LZ choice**|**18,428.01**|**1,619.90**|**37,881.72**|**4,706.35**|
|Also offer contextual literals|18,428.01|1,619.90|37,881.72|4,706.35|
|Local choice,1Ki IDs|31,096.22|3,032.56|53,512.51|6,383.28|
|Local choice,4Ki IDs|24,746.38|2,188.92|49,013.58|5,762.21|
|Local choice,16Ki IDs|20,010.85|1,716.40|40,844.86|4,887.01|
|Local choice,64Ki IDs|18,709.28|1,622.58|38,376.35|4,709.15|
|Best whole or local|18,422.70|1,619.90|37,880.26|4,706.35|

Whole-map choice saves7.34/8.96/4.20/1.75% versus packed Zstd selectors.
It picks context483 times, LZ46, uniform47. Contextual literals never win
when offered alongside them, including ties resolved toward existing choices.

## Selected channels, standard4³

|Channel|Fast Zstd|Fast context|Fast choice|Slow Zstd|Slow context|Slow choice|
|---|---:|---:|---:|---:|---:|---:|
|CT L0|24.625|139.875|14.25|178.625|315.375|147.875|
|CT L1|2|2|2|94.75|111.875|77.625|
|nx L3|2367.625|2096.75|2096.75|6569.625|6572.25|6572.25|
|nx L4|1426.5|1383.625|1342.75|5558|5479|5479|
|width L3|2413.5|2097|2097|6682.375|6685.25|6685.25|
|width L4|1426.5|1383.625|1342.75|3223.75|3023.875|3021.375|
|Compound presence L3|2541|2293.75|2293.75|6335.5|6280.625|6280.625|
|Compound presence L4|1378.5|1431.25|1319.875|3486.875|3267.375|3242|
|Compound grad_mag L3|2389|2293|2293|2389|2293|2293|
|Compound grad_mag L4|1979.25|1949.25|1897.25|3589.25|3516.125|3464.125|

All18 channels/all4 configurations are in the generated report. Selection is
not guaranteed to beat Zstd on every channel; Zstd is not an offered candidate.

## Why local switching/context literals did not help

- Packed literal contexts expand the small ID alphabet into up to256 packed
  byte contexts. Measured average entropy substreams/nonuniform map rises
  from3.99 to158.14. Tables, states, counts and data fragmentation are material.
  Raw-ID literal contexts need only5.76 streams but still lose: preserving
  exact raw substring repetition differs from packing, and removing matched
  data changes the literal statistics. We did not isolate every contribution.
- Smaller independent regions repeat tables/states, fragment training data,
  and lose cross-region matches. The old direct context candidate still uses
  its bulk rANS states. This does NOT rule out shared-table/shared-dictionary
  switching; that would be a different, more complex format/experiment.
- Only five maps prefer local coding, all2³ with64Ki-ID regions. Crucially,
  **all four regions choose context in every winning map**: there is no actual
  mode switch in these winners, only local probability adaptation.

Winning map byte changes (whole ->local): nx L4 slow ea0b8150:48038->47956;
nx L4 fast ea0b8150:19169->18787; width L4 slow d6863d4f:45662->45646;
width L4 fast d6863d4f:19169->18787; compound ny L4 slow feeb86b7:53871->53759.
No standard4³ map benefits. Do not recommend local switching on this evidence.

## Indicative speed

MiB/s of **unpacked selector IDs**, not voxel data. Median of3 timed runs per
map, aggregate bytes/sum medians. O3,8 pinned workers; loaded host including
overlapping sanitizer checks, so not a clean performance claim. Exact sizes
are unaffected. Every decode reconstructs and verifies all original IDs.

|Method|Fast2 enc/dec|Fast4 enc/dec|Slow2 enc/dec|Slow4 enc/dec|
|---|---:|---:|---:|---:|
|Context|237.7 /269.8|276.4 /294.5|221.2 /291.1|286.8 /295.3|
|LZ|83.1 /436.9|172.5 /491.5|83.5 /306.6|126.2 /384.5|
|Whole context/LZ choice|56.3 /281.0|97.5 /312.7|60.1 /269.4|81.1 /296.0|
|Also try contextual literals|8.1 /294.0|12.2 /310.5|5.5 /262.6|7.1 /289.5|
|Try all local sizes too|1.5 /262.9|2.2 /307.8|1.0 /232.0|1.2 /288.7|

Encoding choice is expensive because both codecs run. Decoder runs only the
chosen codec. Discarding the unsuccessful extra candidates avoids their cost.

## Validation and reproduction

31 codecs ×576 maps =17,856 exact corpus checks; all12,672 previous sizes
unchanged. 1,300 synthetic cases (alphabet1..256, short/long/runs/random,
non-divisible lengths and >64KiB). ASan/UBSan corpus uses repetitions0.

```sh
cc -std=c11 -O3 -Werror=incompatible-pointer-types -ffp-contract=off -Isrc \
 benchmark/selector_payload.c \
 /data/hendrik/compression_experiments/villa4/builds/delta3d-c-only/libvc_delta3d.a \
 -lzstd -lm -pthread -o /data/hendrik/compression_experiments/villa4/builds/selector-switch
/data/hendrik/compression_experiments/villa4/builds/selector-switch --self-test
python3 benchmark/compare_selector_payload.py \
 --frames /data/hendrik/compression_experiments/villa4/encoder_speed/release_fixed_presets_20261101/results.json \
 --output /data/hendrik/compression_experiments/villa4/encoder_speed/selector_switch_20261103 \
 --library /data/hendrik/compression_experiments/villa4/builds/hybrid-smoothed-pyramid-v1/libvolcodec_ng_hybrid.so \
 --binary /data/hendrik/compression_experiments/villa4/builds/selector-switch \
 --reuse-results /data/hendrik/compression_experiments/villa4/encoder_speed/selector_payload_20261103/results.jsonl \
 --workers 8 --repetitions 1
```

Reports/data: `encoder_speed/selector_switch_20261103/{report.md,results.jsonl}`
under the experiment root. Sanitizer equivalent:
`encoder_speed/selector_switch_sanitize_20261103`, `selector-switch-sanitize`,
existing sanitizer static library, stack limit32768KiB, O1/debug and
`-fsanitize=address,undefined -fno-omit-frame-pointer`.
