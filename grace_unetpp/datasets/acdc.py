"""ACDC HDF5 slice pipeline used by the five-fold experiments."""
import h5py

import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

IMG_SIZE = 256
ENCODER = "efficientnet-b0"
EARLY_STOPPING_PATIENCE = 8
_spacing_warning_emitted = False
criterion_ce = nn.CrossEntropyLoss()
criterion_dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)


IN_CHANNELS = 1
NUM_CLASSES = 4
BATCH_SIZE = 16
MIN_EPOCHS_BEFORE_EARLY_STOP = 8

def extract_patient_id(filename: str):
    """
    ACDC slice filename example:
    patient001_frame01_slice_0.h5
    -> patient001
    """
    return filename.split("_")[0]


def scan_acdc_cases(root_dir):
    """
    返回:
    cases = {
        patient_id: [h5_path1, h5_path2, ...]
    }
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root_dir}")

    cases = {}
    all_files = sorted([p for p in root.iterdir() if p.suffix.lower() == ".h5"])

    if not all_files:
        raise RuntimeError(f"No .h5 files found in {root_dir}")

    for h5_path in all_files:
        pid = extract_patient_id(h5_path.name)
        cases.setdefault(pid, []).append(str(h5_path))

    if not cases:
        raise RuntimeError("No valid ACDC patients found.")
    return cases


def normalize_inplane_spacing(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]

    if arr.size == 0:
        return None
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    elif arr.size >= 3:
        arr = arr[-2:]
    else:
        arr = arr[:2]
    return arr.astype(np.float32)


def extract_inplane_spacing(h5_file):
    global _spacing_warning_emitted

    candidate_keys = [
        "spacing",
        "pixel_spacing",
        "voxel_spacing",
        "pixdim",
    ]

    for key in candidate_keys:
        if key in h5_file:
            spacing = normalize_inplane_spacing(h5_file[key][()])
            if spacing is not None:
                return spacing

    for key in candidate_keys:
        if key in h5_file.attrs:
            spacing = normalize_inplane_spacing(h5_file.attrs[key])
            if spacing is not None:
                return spacing

    if not _spacing_warning_emitted:
        print("Warning: spacing metadata not found in H5. HD95/ASD will fall back to unit spacing.")
        _spacing_warning_emitted = True
    return np.array([1.0, 1.0], dtype=np.float32)


class ACDCH5Dataset(Dataset):
    def __init__(self, file_list, img_size=256, is_train=True, return_spacing=False):
        self.file_list = file_list
        self.img_size = img_size
        self.return_spacing = return_spacing

        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        h5_path = self.file_list[idx]

        try:
            with h5py.File(h5_path, "r") as f:
                image = f["image"][:]
                mask = f["label"][:]
                spacing = extract_inplane_spacing(f) if self.return_spacing else None
        except Exception as e:
            raise RuntimeError(f"Failed to read H5 file: {h5_path}\n{e}")

        # normalize raw image to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # ensure image is HWC for albumentations
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        mask = mask.astype(np.uint8)

        augmented = self.transform(image=image, mask=mask)
        img = augmented["image"]           # [1, H, W]
        msk = augmented["mask"].long()     # [H, W]

        if msk.max().item() >= NUM_CLASSES or msk.min().item() < 0:
            raise ValueError(
                f"Mask label out of range in {h5_path}: "
                f"min={msk.min().item()}, max={msk.max().item()}"
            )

        if self.return_spacing:
            spacing_tensor = torch.tensor(spacing, dtype=torch.float32)
            return img, msk, spacing_tensor
        return img, msk


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def combined_loss(logits, target):
    return 0.5 * criterion_ce(logits, target) + 0.5 * criterion_dice(logits, target)
