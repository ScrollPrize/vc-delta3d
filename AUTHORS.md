# Authors and provenance

The original **volcodec_ng** implementation is by **Chase Hendrick**,
MIT-licensed. Its original copyright and license are preserved in
`src/ng/LICENSE`. The imported base revision is
`e1770003ba0534d02459e09cb5764087141d8dbc` from ScrollPrize/villa PR #2232.

The original **VC-Delta3D** implementation is by **Sean Johnson**.
Repository history also records contributions by **Johannes Rudolph**,
including the MIT relicensing commit
`103b502ad784df14c876100d87f601c0b6c46888` used as this integration's base.
The project license remains in `LICENSE`.

The hybrid predictor selection, frame formats, presets, and subsequent
optimization experiments were developed in the Villa checkout, branch
`compression/geometry-prediction-trials`. This integration relocates that
implementation into `src/ng`; Villa's former library path is a compatibility
symlink, not a second source copy. Experimental entrypoints remain present
for reproducibility but are not additional public presets.
