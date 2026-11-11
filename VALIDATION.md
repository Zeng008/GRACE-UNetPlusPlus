# Release validation

Validation performed on 11 November 2026, on CPU. No benchmark retraining was
performed for this release.

- Nine automated tests passed: zero DC response, spectral response to a known
  sinusoid, GFM identity initialization and gradients, CGRR initialization and
  confidence gradients, boundary metric edge cases, both dataset input/output
  shapes, trainable stage parameters, checkpoint round-trip, dataset transforms,
  MSD sampling/cache serialization, and subject-level fold isolation.
- With the same state dict and nonzero GFM/CGRR contributions, all four diagnostic
  tensors were bitwise identical to the original implementation for both
  1-channel/4-class ACDC and 3-channel/2-class MSD inputs at 256 x 256.
- Parameter counts with SMP 0.5.0: 6,571,157 (ACDC) and 6,571,425 (MSD).
- A synthetic two-stage CPU training run saved and reloaded the selected
  checkpoints successfully. The w/o-GFM variant completed a backward step.
- Python source compilation and a release-file check passed. The release contains
  no raw images, datasets, trained weights, machine-specific data paths or tokens.

The CPU check environment used Python 3.13.15 and PyTorch 2.15.0+cpu. Model checks
passed with SMP 0.5.0 / timm 0.9.16, and also with SMP 0.3.3. These checks are
software integration evidence; they do not validate numerical reproduction of the
recorded PyTorch 2.1.2 / CUDA 12.1 experiments.

The nine-test suite also passed with the recorded Albumentations 1.4.3 version.
