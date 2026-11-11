# Experiment protocol

This repository provides the GRACE-UNet++ model, training pipeline, recorded
five-fold partitions and evaluation tools.

## Included

1. GFM and CGRR, retaining the original state-dict module names.
2. The EfficientNet-B0 / SMP UNet++ wrapper.
3. The recorded 100-patient ACDC and 41-volume MSD five-fold identifiers.
4. The original preprocessing, sampling, validation selection, loss, training
   stages and evaluation formulas.
5. The paper's own component and refinement-protocol variants.

## Data and configuration

The entry points accept explicit data roots and devices. ACDC uses HDF5 slices;
MSD Spleen uses NIfTI volumes and `dataset.json`. Dataset identifiers are checked
against the recorded manifests in `splits/`. Hyperparameters are in `configs/`.

Stage 1 trains the backbone and GFM. Stage 2 starts from the validation-selected
checkpoint and trains CGRR with the decoder and segmentation head. The fold
training seeds are 731--735, and checkpoint selection uses pooled foreground
validation Dice. Training stages save checkpoints and histories for resuming.

## Evaluation

Evaluation writes per-slice and per-case metrics. `summarize.py` selects the
aggregation endpoint explicitly and reports the mean and sample standard
deviation across folds. HD95_grid and ASD_grid use pixel spacing on the
256 x 256 evaluation grid, and ASD equally weights both directed surface means.

Software checks are documented in [VALIDATION.md](VALIDATION.md).
