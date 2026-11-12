# Benchmarks

## Current

Measured 2026-11-12. EC2 rows on `522ac4d`, the local row on `50a905d`; the
codec code is the same except for a run-time switch added for tests.

**Data:** 95 level-0 128³ uint8 chunks from masked scroll CT volumes of five
scrolls, and 32 chunks from downsampled pyramid levels:

- PHercParis4 `20260411134726-2.400um-0.2m-78keV-masked.zarr` (63 chunks; 12 from level 1, 10 from level 2)
- PHerc0841 `20260319124803-2.403um-0.2m-77keV-masked.zarr` (14 chunks; 10 from level 1)
- PHerc1667 `20260323082859-1.129um-0.2m-59keV-masked.zarr` (6 chunks)
- PHerc0172 `20241024131839-7.910um-53keV-masked.zarr` (5 chunks)
- PHerc0139 `20250728140407-9.362um-1.2m-113keV-masked.zarr` (7 chunks)

**Method:** one core (`taskset`), each chunk encoded once per preset and
round-trip checked, then all frames decoded in nine interleaved rounds; decode
speed is from the fastest round per chunk, averaged over chunks. Timings cover
in-memory buffers. AWS machines are `*.2xlarge` on-demand EC2 instances, GCC
12, CMake Release builds; the local machine is a laptop with turbo boost off.
Encoded bytes are identical on every machine. The tool is
`benchmark/ladder.c`; the AWS boxes are `tests/vc-delta3d-bench/arm-dev.yaml`
in the ScrollPrize infra repository. Speeds are for the 95 level-0 chunks.

### Size (% of raw; lower is better)

|Data|Chunks|delta|faster|fast|balanced|small|slower|medium|slow|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PHercParis4 2.4 µm|63|36.78|27.60|27.45|27.37|27.30|27.30|27.60|27.52|
|PHerc0841 2.4 µm|14|34.14|23.16|22.91|22.75|22.54|22.63|23.09|22.97|
|PHerc1667 1.1 µm|6|22.20|12.02|11.90|11.86|11.65|11.76|12.00|11.95|
|PHerc0172 7.9 µm|5|63.33|49.40|49.32|49.23|49.14|49.18|49.40|49.37|
|PHerc0139 9.4 µm|7|39.08|26.25|25.88|25.65|24.99|25.53|26.18|26.06|
|All level 0|95|37.04|27.01|26.84|26.73|26.59|26.65|26.99|26.91|
|PHercParis4 level 1|12|56.93|45.16|45.02|44.92|44.72|44.88|45.15|45.12|
|PHercParis4 level 2|10|70.10|59.27|59.20|59.10|58.95|59.06|59.27|59.21|
|PHerc0841 level 1|10|56.18|44.84|44.68|44.56|44.07|44.53|44.84|44.80|

### Encode (MiB/s, one core; higher is better)

|Machine|delta|faster|fast|balanced|small|slower|medium|slow|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|c5 · Xeon 8275CL (Cascade Lake)|159|40.1|38.8|29.0|7.21|1.19|9.79|1.61|
|c6i · Xeon 8375C (Ice Lake)|197|51.6|49.6|35.3|9.29|1.37|12.9|1.94|
|c7i · Xeon 8488C (Sapphire Rapids)|240|64.7|64.0|47.6|11.8|1.86|16.2|2.57|
|c8i · Xeon 6975P (Granite Rapids)|277|68.5|68.9|50.3|11.9|1.93|16.5|2.65|
|c6a · EPYC 7R13 (Zen 3)|216|47.2|46.8|34.8|9.31|1.54|11.3|1.92|
|c7a · EPYC 9R14 (Zen 4)|230|58.4|57.5|44.7|11.0|2.01|14.5|2.66|
|c6g · Graviton2 (Neoverse N1)|167|23.3|21.4|12.9|4.00|0.40|6.35|0.65|
|c7g · Graviton3 (Neoverse V1)|243|38.7|36.6|23.4|6.92|0.77|10.4|1.20|
|c8g · Graviton4 (Neoverse V2)|279|46.7|43.6|27.2|8.17|0.86|13.0|1.36|
|c9g · Graviton5 (Neoverse V3)|356|58.9|55.1|33.4|10.0|1.03|16.1|1.63|
|local · Ryzen 7 4800H (Zen 2, AVX2)|165|36.3|35.5|26.2|6.56|1.02|8.48|1.44|

### Decode (MiB/s, one core; higher is better)

|Machine|delta|faster|fast|balanced|small|slower|medium|slow|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|c5 · Xeon 8275CL (Cascade Lake)|361|212|204|197|158|167|186|177|
|c6i · Xeon 8375C (Ice Lake)|434|264|247|235|184|193|228|217|
|c7i · Xeon 8488C (Sapphire Rapids)|512|306|291|279|217|237|262|250|
|c8i · Xeon 6975P (Granite Rapids)|583|347|330|315|237|264|292|277|
|c6a · EPYC 7R13 (Zen 3)|506|202|191|183|168|160|185|184|
|c7a · EPYC 9R14 (Zen 4)|541|277|265|254|226|226|254|253|
|c6g · Graviton2 (Neoverse N1)|399|56.6|51.6|47.6|37.5|36.6|50.8|48.4|
|c7g · Graviton3 (Neoverse V1)|571|92.3|86.2|80.1|62.9|60.5|81.7|74.8|
|c8g · Graviton4 (Neoverse V2)|703|117|109|102|79.8|75.7|104|93.4|
|c9g · Graviton5 (Neoverse V3)|932|146|136|127|102|94.5|130|117|
|local · Ryzen 7 4800H (Zen 2, AVX2)|362|140|133|127|111|111|129|127|

Notes:

- `faster` is the `fast` preset of 0.3.x, byte for byte. How much the smaller
  presets gain depends on the scan: `small` is 1.1% smaller than `faster` on
  PHercParis4 and 4.8% on PHerc0139. On these volumes `medium` and `slow` are
  not worth their encode time.
- x86 decoders use AVX2, plus AVX-512 where available. Frames with several
  models (`small`, and some `slower`, `medium` and `slow` frames) use the AVX2
  recurrence on every x86 machine, which is why `small` loses more against
  `fast` on Intel than on Zen 3.
- Frames with several models decode about twice as slowly with 0.3.x
  decoders as with this version; the frame format is unchanged.
- ARM decoders use NEON. NG decoding on Graviton is about 2.5 times slower
  than on comparable x86 machines: NEON vectors are 128 bits wide (AVX2 256,
  AVX-512 512) and there is no gather instruction for the rANS table lookups.
  `delta` is as fast as on x86.
- Single runs on shared cloud machines vary by a few percent.

### Derived volumes: size (% of raw)

256 cubes of 128³ uint8, 32 per row, cut from PHercParis4 prediction volumes:
3D ink (`ink-3d`, levels 0 and 1), surface predictions (`surfaces`, levels 0
and 2) and the lasagna volumes nx, ny, cos and grad_mag. Cubes with fewer than
5% nonzero voxels were skipped.

|Data|Cubes|delta|faster|fast|balanced|small|slower|medium|slow|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|3D ink, level 0|32|15.28|9.89|9.92|9.87|8.53|8.87|9.21|8.85|
|3D ink, level 1|32|24.11|11.97|11.97|11.97|11.97|11.65|11.95|11.65|
|Surface, level 0|32|1.13|0.59|0.59|0.59|0.59|0.58|0.59|0.58|
|Surface, level 2|32|1.87|1.40|1.40|1.40|1.40|1.31|1.38|1.31|
|Lasagna nx, level 4|32|24.52|22.57|22.60|22.48|20.52|18.48|20.09|18.31|
|Lasagna ny, level 4|32|23.61|21.54|21.53|21.46|20.39|17.33|18.85|17.36|
|Lasagna cos, level 3|32|44.89|23.86|23.22|22.60|19.41|19.96|21.21|21.07|
|Lasagna grad_mag, level 4|32|31.81|18.79|18.45|18.29|17.74|18.15|18.71|18.62|

- 3D ink level 1 and the surface volumes are sparse; a fixed predictor codes
  them best, so `faster` to `small` write the same frame. `slower`, `medium`
  and `slow` gain there by letting every block choose among fixed predictors.
- On nx and ny `medium` is ahead of `small`; on 3D ink level 0, cos and
  grad_mag `small` is ahead of `medium` and `slow`.

## Historical comparison (before the C port)

Measured before the C port, on a different machine and data set (18
channel/scale groups including derived volumes). It includes LPC and the
`tooslow` preset, which 0.3.0 no longer has, and the then-current `NGC1`
container. Its `fast` column is the preset now called `faster`. Speeds are not
comparable with the tables above.

144 identical hash-checked128³ uint8 chunks across18 channel/scale groups. Native codecs, single-core, three load-filtered pinned repetitions. Combined speed is total raw bytes divided by summed median chunk times. Complete frames, not residual-only sizes.

New preset names map to the completed five-preset release suite; those measurements are reused, not rerun. Baselines were freshly measured. Preset scalar ratios are complete native frames, with no added envelope. NGC1 index bytes apply only to channel/spatial containers. Integrated API timings are not substituted for the earlier native timings.

Delta3D/LPC: pinned0.2.0 (LPC includes upstream Delta fallback). Original NG: pinned original control. Blosc1 via numcodecs0.16.5: level5, byte shuffle, automatic internal block size, explicitly one thread; both LZ4 and Zstd backends. Standalone Zstd: level3. Python is orchestration only; compression/decompression is native. Baseline wrappers include output allocation/copy, as in the existing control harness.

### Size (% raw; lower is better)

|Channel|Delta3D|Original NG|LPC|Blosc-LZ4|Blosc-Zstd|Zstd|fast|medium|slow|slower|tooslow|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|CT-L0|31.112|22.123|24.390|86.789|59.209|64.530|22.121|22.099|21.845|21.654|**21.632**|
|CT-L1|57.439|45.321|47.454|100.001|84.544|86.408|45.192|45.189|45.164|44.856|**44.798**|
|nx-L3|47.830|64.561|47.830|64.678|48.981|50.454|40.697|37.621|37.089|37.089|**33.388**|
|nx-L4|52.656|75.948|52.656|70.313|51.764|52.767|45.544|44.877|44.416|44.416|**39.500**|
|width-L3|38.203|55.971|38.203|61.015|37.279|38.988|30.203|27.626|27.237|27.237|**23.170**|
|width-L4|42.420|69.097|42.420|67.411|42.193|43.277|35.844|35.692|35.599|35.599|**30.291**|
|ny-L3|44.399|62.630|44.399|62.479|45.813|47.561|37.085|34.800|34.614|34.614|**30.714**|
|ny-L4|57.063|77.042|57.063|73.958|56.136|57.046|49.614|48.770|48.399|48.399|**43.092**|
|compound/presence-L3|38.612|30.725|38.118|56.564|36.397|37.770|27.703|25.736|25.620|25.533|**23.999**|
|compound/presence-L4|44.040|48.914|44.040|68.925|43.985|45.084|36.887|36.632|36.573|36.571|**33.767**|
|compound/sheet_grad_mag-L3|6.488|65.144|6.488|28.923|8.529|11.047|3.951|3.951|3.951|3.951|**3.113**|
|compound/sheet_grad_mag-L4|9.242|74.592|9.242|32.740|11.473|13.859|6.172|6.172|6.127|6.127|**5.253**|
|compound/sheet_nx-L3|39.574|64.899|39.574|59.965|40.396|42.442|31.815|29.568|29.378|29.378|**25.875**|
|compound/sheet_nx-L4|52.927|76.515|52.927|73.214|52.475|53.644|45.509|44.079|43.537|43.537|**37.897**|
|compound/sheet_ny-L3|42.018|62.970|42.018|62.597|42.765|44.790|34.724|31.788|31.608|31.608|**27.621**|
|compound/sheet_ny-L4|51.938|75.715|51.938|72.867|51.238|52.511|45.059|43.282|42.646|42.646|**37.084**|
|compound/width-L3|35.666|56.908|35.666|60.542|36.666|38.402|28.956|26.529|26.422|26.422|**22.495**|
|compound/width-L4|45.615|70.849|45.615|71.174|44.827|45.946|38.631|38.278|38.143|38.143|**32.272**|
|Combined|40.958|61.107|40.002|65.231|44.148|45.918|33.650|32.372|32.132|32.099|**28.665**|

### Encode (MiB/s)

|Channel|Delta3D|Original NG|LPC|Blosc-LZ4|Blosc-Zstd|Zstd|fast|medium|slow|slower|tooslow|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|CT-L0|359.851|233.933|134.844|**1890.344**|65.691|264.986|74.518|14.059|3.446|2.602|1.727|
|CT-L1|315.222|247.270|121.935|**5143.167**|141.473|568.299|84.539|14.391|3.401|2.393|1.541|
|nx-L3|381.924|223.883|123.589|**949.932**|73.228|291.066|52.491|13.546|4.599|4.568|2.058|
|nx-L4|403.865|252.268|123.217|**945.058**|65.236|275.604|77.241|13.402|4.684|4.606|1.979|
|width-L3|380.392|249.049|126.309|**859.956**|64.366|286.769|49.642|13.659|4.687|4.671|2.186|
|width-L4|451.119|246.260|129.964|**866.188**|60.484|272.643|79.468|13.597|4.591|4.456|2.013|
|ny-L3|372.410|238.745|123.146|**887.770**|71.154|294.625|51.628|13.419|4.572|4.510|2.011|
|ny-L4|391.320|250.235|121.990|**952.167**|66.524|272.753|76.409|13.213|4.615|4.546|1.905|
|compound/presence-L3|418.326|239.609|137.506|**779.268**|61.148|297.490|61.645|13.566|3.769|3.070|1.789|
|compound/presence-L4|440.786|245.154|135.086|**824.143**|56.874|266.297|78.846|13.400|4.047|3.483|1.737|
|compound/sheet_grad_mag-L3|461.599|246.834|130.113|**1165.451**|67.610|732.012|81.810|13.855|5.439|5.341|2.894|
|compound/sheet_grad_mag-L4|493.521|241.805|129.448|**1133.216**|50.994|681.605|82.371|13.857|5.421|5.357|2.826|
|compound/sheet_nx-L3|385.802|240.044|124.030|**836.855**|66.654|294.460|52.951|13.566|4.650|4.582|2.144|
|compound/sheet_nx-L4|396.590|251.277|122.025|**937.574**|63.987|274.408|76.423|13.238|4.626|4.580|1.969|
|compound/sheet_ny-L3|383.652|238.819|124.519|**883.606**|68.088|295.720|49.365|13.494|4.581|4.546|2.141|
|compound/sheet_ny-L4|378.472|242.136|120.455|**922.106**|62.224|269.357|76.147|13.296|4.677|4.614|1.958|
|compound/width-L3|369.683|246.945|125.123|**842.748**|63.583|291.703|52.760|13.553|4.711|4.654|2.158|
|compound/width-L4|443.136|250.488|129.042|**900.043**|60.744|272.928|78.879|13.304|4.751|4.691|2.053|
|Combined|397.284|243.394|126.608|**981.420**|65.610|310.402|66.009|13.572|4.446|4.085|2.015|

### Decode (MiB/s)

|Channel|Delta3D|Original NG|LPC|Blosc-LZ4|Blosc-Zstd|Zstd|fast|medium|slow|slower|tooslow|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|CT-L0|742.781|504.405|427.145|**10498.071**|1340.712|1467.403|453.658|334.468|456.493|370.800|83.090|
|CT-L1|585.408|495.715|354.062|**37968.408**|1352.412|1692.826|462.955|332.933|448.265|363.737|79.924|
|nx-L3|681.443|478.548|681.918|**3422.698**|1537.845|1382.902|487.401|425.468|486.063|484.170|73.896|
|nx-L4|670.138|473.860|671.994|**3334.755**|1429.986|1363.272|700.341|505.755|398.539|394.882|74.781|
|width-L3|746.026|477.376|746.325|**3215.645**|1645.337|1541.465|542.161|427.757|394.783|390.331|74.752|
|width-L4|749.878|479.731|752.496|**3262.344**|1622.009|1557.762|706.595|630.896|556.516|554.799|80.514|
|ny-L3|684.745|480.009|684.876|**3188.341**|1520.422|1376.524|564.670|444.107|395.233|396.870|70.901|
|ny-L4|641.648|473.692|640.266|**3288.041**|1374.563|1307.599|700.552|476.499|481.936|478.354|73.127|
|compound/presence-L3|759.736|507.587|501.416|**2996.864**|1777.533|1550.139|606.415|365.267|369.627|318.796|67.353|
|compound/presence-L4|733.985|495.508|733.849|**3063.781**|1587.776|1486.969|703.123|496.691|505.951|489.412|75.271|
|compound/sheet_grad_mag-L3|1132.178|482.246|1128.915|**2111.091**|1865.939|1326.648|723.181|732.786|731.836|728.337|87.464|
|compound/sheet_grad_mag-L4|1099.075|482.314|1099.279|**2072.036**|1882.219|1287.683|729.466|729.191|589.032|579.368|87.199|
|compound/sheet_nx-L3|732.216|483.694|732.355|**3160.439**|1560.331|1417.485|562.087|456.415|429.955|432.881|72.964|
|compound/sheet_nx-L4|667.435|477.681|665.738|**3240.660**|1394.022|1344.101|693.590|446.020|489.414|484.358|75.027|
|compound/sheet_ny-L3|707.160|479.307|706.150|**3147.127**|1511.539|1412.619|543.940|444.530|416.543|420.539|72.314|
|compound/sheet_ny-L4|646.357|472.745|646.060|**3206.712**|1397.399|1354.317|703.946|454.941|444.293|443.668|72.969|
|compound/width-L3|741.074|486.553|742.046|**3047.220**|1648.676|1529.393|549.699|443.009|421.836|412.352|74.251|
|compound/width-L4|729.185|502.355|729.448|**3180.385**|1578.284|1530.256|706.580|530.751|556.631|558.309|78.074|
|Combined|728.012|484.952|654.227|**3295.207**|1541.155|1432.912|603.192|460.318|463.241|444.221|75.979|

### Repeat spread (aggregate MiB/s)

|Preset|Direction|Mean-time|Slowest repeats|Median repeats|Fastest repeats|
|---|---|---:|---:|---:|---:|
|Delta3D|encode|396.446|390.799|397.284|401.399|
|Delta3D|decode|727.586|723.656|728.012|731.128|
|Original NG|encode|240.889|233.605|243.394|246.028|
|Original NG|decode|484.945|478.777|484.952|491.265|
|LPC|encode|126.462|125.460|126.608|127.333|
|LPC|decode|653.507|649.163|654.227|657.180|
|Blosc-LZ4|encode|976.583|956.773|981.420|992.237|
|Blosc-LZ4|decode|3292.676|3261.085|3295.207|3322.308|
|Blosc-Zstd|encode|65.518|64.971|65.610|65.981|
|Blosc-Zstd|decode|1540.857|1527.684|1541.155|1553.955|
|Zstd|encode|309.893|304.709|310.402|314.730|
|Zstd|decode|1432.449|1421.447|1432.912|1443.152|
|fast|encode|65.862|65.111|66.009|66.480|
|fast|decode|602.522|594.732|603.192|609.831|
|medium|encode|13.560|13.466|13.572|13.643|
|medium|decode|459.228|450.537|460.318|467.133|
|slow|encode|4.438|4.392|4.446|4.477|
|slow|decode|463.243|454.778|463.241|472.032|
|slower|encode|4.083|4.048|4.085|4.115|
|slower|decode|443.787|435.373|444.221|452.082|
|tooslow|encode|2.014|1.974|2.015|2.054|
|tooslow|decode|76.000|74.968|75.979|77.084|
