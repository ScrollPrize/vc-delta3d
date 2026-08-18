# Zarr recompression example

Install VC-Delta3D with the dependencies needed for S3-backed Zarr stores:

```sh
python -m pip install 'vc-delta3d[examples]'
```

From a repository checkout, use `python -m pip install -e '.[examples]'`
instead.

The example reads the public PHercParis4 multiscale group with anonymous S3
access and writes a new Zarr v2 group whose arrays use `Delta3D` compression.
The output may be a local directory or an S3 URL for which your normal AWS
credential chain has write permission.

Try the smallest resolution level first:

```sh
python examples/recompress_zarr.py ./PHercParis4-delta3d.zarr --levels 5
```

Recompress every pyramid level to S3:

```sh
python examples/recompress_zarr.py \
  s3://my-output-bucket/PHercParis4-delta3d.zarr
```

The source contains levels `0` through `5`, totaling roughly 92.6 TB of raw
voxels. Copying every level is a substantial data-transfer and compute job;
the `--levels 5` run is about 2.5 GB before compression and is a useful check
of permissions, throughput, and the resulting codec metadata. The command
refuses to replace an existing output unless `--overwrite` is supplied.

By default the copy is lossless (`--quant 1`) and uses eight concurrent chunk
transfers. `--quant 3` enables bounded-error quantization with a maximum error
of one intensity unit; `--workers` controls transfer concurrency.

After recompression, the script reports the logical starting size, actual
stored output size, compression ratio, codec encode/decode speed, and the
end-to-end copy throughput. Codec speed is measured from 128 evenly distributed
chunks per level by default, avoiding an expensive second read of the complete
92.6 TB pyramid. Change the sample size with `--benchmark-chunks`; use
`--benchmark-chunks 0` to skip the codec-only speed measurement.
