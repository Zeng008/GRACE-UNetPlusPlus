# Reproducibility scope

This is a curated release of the model and experiment pipeline, extracted from
the original implementation rather than a newly trained experiment.

## Included

1. GFM and CGRR, retaining the original state-dict module names.
2. The EfficientNet-B0 / SMP UNet++ wrapper.
3. The recorded 100-patient ACDC and 41-volume MSD five-fold identifiers.
4. The original preprocessing, sampling, validation selection, loss, training
   stages and evaluation formulas.
5. The paper's own component and refinement-protocol variants.

The public entry points remove dataset-scanning and GPU/environment side effects
at import time, accept explicit data roots and devices, and omit historical
experimental architectures and external comparison networks. The Fourier kernel
cache now also checks device changes. Cached MSD data loading uses a top-level
class for multiprocessing compatibility. Resume restores CPU RNG tensors even
when a checkpoint is loaded on a CUDA device. These are packaging fixes.

## Limits of this release

- No five-fold retraining was performed during packaging.
- Trained research checkpoints are not bundled.
- The original raw ACDC NIfTI-to-HDF5 conversion is not available in this package.
- The release provides runnable method/protocol code. It does not claim that
  every numerical entry in the manuscript has been independently reproduced.
  Exact table replication also requires the matching checkpoints, data
  preprocessing provenance and the precise aggregation endpoint.
- External baseline experiments, comparisons and their authors' code are outside
  this repository. They should be obtained from the respective original sources.

Synthetic CPU tests establish mathematical invariants and software integration;
they do not establish GPU determinism or the paper's benchmark accuracy.
The complete release check record is written in `VALIDATION.md` after testing.
