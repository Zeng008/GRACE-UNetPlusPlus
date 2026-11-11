# GRACE-UNet++

PyTorch implementation of **GRACE-UNet++: Gravitational Modulation with
Confidence-Guided Refinement for Biomedical Image Segmentation**.

This repository contains our model and the code needed to train and evaluate it
on ACDC and MSD Spleen. It includes the component/refinement ablations used in the
manuscript. Implementations of external comparison methods are outside this release.

## Model

- EfficientNet-B0 encoder and nested UNet++ decoder, provided by
  [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch).
- **GFM:** non-negative learned source, periodic Fourier inverse Laplacian with
  its DC mode removed, and residual bottleneck modulation
  `F' = F + gamma * F * sigmoid(-phi)`. `gamma` starts at zero.
- **CGRR:** `u = 1 - max_c softmax(base_logits)_c`; concatenate decoder features
  with `u`, then apply Conv-BN-ReLU, depthwise Conv-BN-ReLU and a zero-initialized
  pointwise projection. Refined logits are `base_logits + delta * (0.5 + u)`.

```python
import torch
from grace_unetpp import GRACEUNetPlusPlus

model = GRACEUNetPlusPlus(in_channels=1, classes=4, encoder_weights=None).eval()
with torch.no_grad():
    logits = model(torch.randn(1, 1, 256, 256))
assert logits.shape == (1, 4, 256, 256)
```

## Installation

The recorded experiment uses Python 3.10/3.11, PyTorch 2.1.2, SMP 0.5.0 and
Albumentations 1.4.3. Create an environment, install a PyTorch build suitable for
your device, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Other dependency versions may run but do not establish reproduction of the
recorded GPU experiment. No training data or pretrained research checkpoints
are distributed in this repository. ImageNet encoder weights are downloaded by
SMP when training with the default `--weights imagenet`.

## Data

Obtain ACDC and MSD Spleen from their original providers. Follow their access and
reuse terms. The repository only includes the existing benchmark identifiers
needed for our recorded five-fold splits.

ACDC input is one HDF5 file per slice, with `image` and `label` arrays, named like
`patient001_frame01_slice_0.h5`. Both frames and all slices of a patient must
retain the same patient prefix. Labels are background=0, RV=1, Myo=2 and LV=3.
Spacing may be stored in `spacing`, `pixel_spacing`, `voxel_spacing` or `pixdim`
datasets/attributes. Without recorded spacing, evaluation uses unit spacing.
The original raw-NIfTI-to-HDF5 converter is not included; use the same HDF5
preprocessing as the experiments to compare exact results.

MSD input is the original `Task09_Spleen` folder containing `dataset.json`,
`imagesTr/` and `labelsTr/`. CT intensities are clipped to [-160, 240], scaled to
[0,1], cropped to an image-derived body region and replicated to three channels.

Both datasets are resized to 256 x 256 and normalized with mean/std 0.5 on a
unit intensity range. Training augmentation uses horizontal/vertical flips,
rotation within 20 degrees and brightness/contrast changes. All validation/test
preprocessing is deterministic. MSD training retains all positive slices and up
to one empty slice per positive slice; evaluation retains all slices.

## Five-fold training

```bash
python train.py --dataset acdc --data-root /path/to/ACDC_training_slices
python train.py --dataset msd --data-root /path/to/Task09_Spleen
```

The recorded splits are in `splits/`; the configuration is in `configs/`.
The split seed is 20261027 and fold training seeds are 731--735. The code checks
that the input subject identifiers agree with the recorded manifest.

Stage 1 trains the encoder, GFM, decoder and segmentation head while CGRR is
not optimized. Stage 2 starts from the best validation checkpoint and optimizes
the decoder, segmentation head and CGRR with a fresh optimizer. Encoder and GFM
parameters are frozen. As in the original experiment, `model.train()` remains
active during optimization, so frozen modules' BatchNorm running buffers may
still update. Selection uses pooled foreground validation Dice only.

Choose `--fold-indices 0` for one fold. Outputs include best/last checkpoints,
training history, run settings, per-slice and per-case metrics, confusion counts
and probability histograms. Interrupted stages resume from `last.pth`.
Use a new `--output` directory when changing settings. For Windows data loading,
the entry points are guarded and the cached MSD dataset is a module-level class.

The following protocols contain only variants of our own model:

```bash
python train.py --dataset acdc --data-root /path/to/slices --protocol without_gfm
python train.py --dataset acdc --data-root /path/to/slices --protocol without_cgrr
python train.py --dataset acdc --data-root /path/to/slices --protocol cgrr_only
python train.py --dataset acdc --data-root /path/to/slices --protocol cgrr_unified
```

`full` is joint decoder/head/CGRR refinement from the outset of stage 2.
`cgrr_only` trains only CGRR in stage 2. `cgrr_unified` first trains CGRR alone
and then jointly with the decoder/head. `without_gfm` pretrains the same backbone
without gravitational modulation before joint CGRR refinement. The w/o-CGRR
variant uses stage 1 directly.

## Evaluation and inference

```bash
python evaluate.py --dataset acdc --data-root /path/to/slices --fold 0 \
  --checkpoint runs/acdc/fold_0/full/stage_cgrr/best.pth --output evaluation/acdc_fold0
python predict.py --dataset acdc --input normalized_input.npy \
  --checkpoint runs/acdc/fold_0/full/stage_cgrr/best.pth --output prediction.npz
python summarize.py --root runs --aggregation slice
python summarize.py --root runs --aggregation case
```

`predict.py` expects **already preprocessed**, normalized CHW/NCHW input; it does
not accept raw clinical images. It exports predictions, logits, base logits,
gated corrections and ambiguity maps. For the w/o-GFM checkpoint, use
`evaluate.py --without-gfm`.

HD95 and ASD are spacing-weighted distances on the resized/cropped evaluation
grid, **not restored physical millimetres**. ASD equally weights the two directed
surface-distance means. Both-empty masks give NaN (excluded from means);
one-empty masks receive the spacing-weighted grid diagonal as a penalty.
The Dice loss includes background through SMP's multiclass loss; checkpoint
selection averages foreground classes only. Slice and case aggregation are
different endpoints; `summarize.py` requires the choice explicitly.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for release scope and validation.
See [README_zh.md](README_zh.md) for a Chinese overview.

## Attribution

UNet++, EfficientNet and their reusable implementation are dependencies rather
than new architectures claimed by this repository. Their original authors and
licenses remain applicable. This release does not redistribute their source,
ImageNet weights, medical datasets or other authors' comparison networks.
