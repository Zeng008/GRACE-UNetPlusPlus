"""MSD Spleen NIfTI slice pipeline, including image-only ROI cropping."""
import json
import nibabel as nib

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


IN_CHANNELS = 3
NUM_CLASSES = 2
BATCH_SIZE = 32
MIN_EPOCHS_BEFORE_EARLY_STOP = 25
TRAIN_EMPTY_TO_POSITIVE_RATIO = 1.0
ROI_CROP_PADDING = 16
BODY_THRESHOLD = 0.05

def resolve_path(base_dir, relative_path):
    clean_rel = relative_path.replace("./", "")
    full_path = os.path.join(base_dir, clean_rel)

    if os.path.exists(full_path):
        return full_path

    if full_path.endswith(".nii.gz"):
        alt_path = full_path[:-3]
        if os.path.exists(alt_path):
            return alt_path
    elif full_path.endswith(".nii"):
        alt_path = full_path + ".gz"
        if os.path.exists(alt_path):
            return alt_path

    return None


def extract_case_id(img_path: str):
    name = os.path.basename(img_path)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name


def scan_msd_cases(root_dir):
    """
    返回:
    cases = {
        case_id: (img_path, mask_path)
    }
    """
    json_path = os.path.join(root_dir, "dataset.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"dataset.json not found in {root_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    training_items = data.get("training", [])
    if not training_items:
        raise RuntimeError("No training items found in dataset.json")

    cases = {}
    for item in training_items:
        img_path = resolve_path(root_dir, item["image"])
        mask_path = resolve_path(root_dir, item["label"])

        if img_path is None or mask_path is None:
            raise FileNotFoundError(f"Failed to resolve paths:\nimage={item['image']}\nlabel={item['label']}")

        case_id = extract_case_id(img_path)
        cases[case_id] = (img_path, mask_path)

    if not cases:
        raise RuntimeError("No valid MSD cases found.")

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
        arr = arr[:2]
    else:
        arr = arr[:2]
    return arr.astype(np.float32)


def extract_inplane_spacing(nii_obj):
    global _spacing_warning_emitted

    spacing = None
    try:
        spacing = normalize_inplane_spacing(nii_obj.header.get_zooms())
    except Exception:
        spacing = None

    if spacing is None:
        if not _spacing_warning_emitted:
            print("Warning: spacing metadata not found in NIfTI. HD95/ASD will fall back to unit spacing.")
            _spacing_warning_emitted = True
        spacing = np.array([1.0, 1.0], dtype=np.float32)
    return spacing


class MSDNiiDataset(Dataset):
    def __init__(self, volume_pairs, img_size=256, is_train=True, return_spacing=False, sampling_seed=42):
        self.volume_pairs = volume_pairs
        self.img_size = img_size
        self.is_train = is_train
        self.return_spacing = return_spacing
        self.sampling_seed = sampling_seed

        print(f"Building 3D slice index ({'Train' if is_train else 'Val/Test'})...")
        positive_slice_map = []
        empty_slice_map = []

        for vol_idx, (img_path, mask_path) in enumerate(tqdm(volume_pairs)):
            try:
                img_proxy = nib.load(img_path)
                mask_proxy = nib.load(mask_path)
                shape = img_proxy.shape
                mask_data = mask_proxy.get_fdata()

                if len(shape) == 3:
                    num_slices = shape[2]
                elif len(shape) == 4:
                    num_slices = shape[2]
                else:
                    print(f"Skipping weird shape {shape}: {img_path}")
                    continue

                for s in range(num_slices):
                    if mask_data.ndim == 4:
                        mask_slice = mask_data[:, :, s, 0]
                    else:
                        mask_slice = mask_data[:, :, s]

                    if np.any(mask_slice > 0):
                        positive_slice_map.append((vol_idx, s))
                    else:
                        empty_slice_map.append((vol_idx, s))
            except Exception as e:
                print(f"Error reading header {img_path}: {e}")

        if is_train:
            rng = random.Random(self.sampling_seed)
            max_empty_slices = int(round(len(positive_slice_map) * TRAIN_EMPTY_TO_POSITIVE_RATIO))
            keep_empty_slices = min(len(empty_slice_map), max_empty_slices)
            sampled_empty_slice_map = (
                rng.sample(empty_slice_map, keep_empty_slices)
                if keep_empty_slices < len(empty_slice_map)
                else list(empty_slice_map)
            )
            self.slice_map = list(positive_slice_map) + sampled_empty_slice_map
            rng.shuffle(self.slice_map)
            print(
                f"Indexed {len(self.slice_map)} training slices "
                f"(positive={len(positive_slice_map)}, kept_empty={len(sampled_empty_slice_map)}, "
                f"dropped_empty={max(0, len(empty_slice_map) - len(sampled_empty_slice_map))})."
            )
        else:
            self.slice_map = list(positive_slice_map) + list(empty_slice_map)
            print(
                f"Indexed {len(self.slice_map)} eval slices "
                f"(positive={len(positive_slice_map)}, empty={len(empty_slice_map)})."
            )

        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1.0),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1.0),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.slice_map)

    def preprocess_ct(self, img_slice):
        # MSD Spleen 常用腹部软组织窗
        min_hu = -160
        max_hu = 240
        img_slice = np.clip(img_slice, min_hu, max_hu)
        img_slice = (img_slice - min_hu) / (max_hu - min_hu + 1e-8)
        return img_slice.astype(np.float32)

    @staticmethod
    def compute_body_bbox(image, threshold=BODY_THRESHOLD):
        body_mask = image > threshold
        if not np.any(body_mask):
            return None

        ys, xs = np.where(body_mask)
        return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    @staticmethod
    def expand_to_square_bbox(bbox, image_shape, padding=ROI_CROP_PADDING):
        y1, y2, x1, x2 = bbox
        height, width = image_shape

        y1 = max(0, y1 - padding)
        y2 = min(height, y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(width, x2 + padding)

        box_h = y2 - y1
        box_w = x2 - x1
        side = max(box_h, box_w)
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0

        new_y1 = int(round(cy - side / 2.0))
        new_y2 = new_y1 + side
        new_x1 = int(round(cx - side / 2.0))
        new_x2 = new_x1 + side

        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        if new_y2 > height:
            shift = new_y2 - height
            new_y1 = max(0, new_y1 - shift)
            new_y2 = height
        if new_x2 > width:
            shift = new_x2 - width
            new_x1 = max(0, new_x1 - shift)
            new_x2 = width

        return new_y1, new_y2, new_x1, new_x2

    def crop_to_body_roi(self, image, mask):
        bbox = self.compute_body_bbox(image, threshold=BODY_THRESHOLD)
        if bbox is None:
            return image, mask

        y1, y2, x1, x2 = self.expand_to_square_bbox(bbox, image.shape, padding=ROI_CROP_PADDING)
        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.slice_map[idx]
        img_path, mask_path = self.volume_pairs[vol_idx]

        try:
            img_nii = nib.load(img_path)
            mask_nii = nib.load(mask_path)

            image = img_nii.get_fdata()
            mask = mask_nii.get_fdata()
            spacing = extract_inplane_spacing(img_nii) if self.return_spacing else None
        except Exception as e:
            raise RuntimeError(f"Failed to read NIfTI pair:\n{img_path}\n{mask_path}\n{e}")

        if image.ndim == 4:
            image = image[:, :, slice_idx, 0]
        else:
            image = image[:, :, slice_idx]

        if mask.ndim == 4:
            mask = mask[:, :, slice_idx, 0]
        else:
            mask = mask[:, :, slice_idx]

        image = self.preprocess_ct(image)

        # binary spleen
        mask = (mask > 0).astype(np.uint8)

        # image-derived ROI crop to remove surrounding air without using labels
        image, mask = self.crop_to_body_roi(image, mask)

        # 单通道复制为3通道，沿用 KiTS / ImageNet encoder 设定
        image = np.stack([image, image, image], axis=-1)

        augmented = self.transform(image=image, mask=mask)
        img = augmented["image"]
        msk = augmented["mask"].long()

        if msk.max().item() >= NUM_CLASSES or msk.min().item() < 0:
            raise ValueError(
                f"Mask label out of range in {mask_path}: "
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


class CachedMSDDataset(MSDNiiDataset):

    def _cached_volume(self, volume_index):
        cache = getattr(self, '_volume_cache', None)
        if cache is None:
            cache = self._volume_cache = {}
        if volume_index not in cache:
            image_path, mask_path = self.volume_pairs[volume_index]
            image_nii, mask_nii = (nib.load(image_path), nib.load(mask_path))
            image = np.asarray(image_nii.dataobj, dtype=np.float32)
            mask = np.asarray(mask_nii.dataobj, dtype=np.uint8)
            if image.ndim == 4:
                image = image[..., 0]
            if mask.ndim == 4:
                mask = mask[..., 0]
            spacing = extract_inplane_spacing(image_nii)
            cache[volume_index] = (image, mask, spacing)
        return cache[volume_index]

    def __getitem__(self, index):
        volume_index, slice_index = self.slice_map[index]
        image_volume, mask_volume, spacing = self._cached_volume(volume_index)
        image = self.preprocess_ct(image_volume[..., slice_index])
        mask = (mask_volume[..., slice_index] > 0).astype(np.uint8)
        image, mask = self.crop_to_body_roi(image, mask)
        image = np.stack([image] * 3, axis=-1)
        augmented = self.transform(image=image, mask=mask)
        image, mask = (augmented['image'], augmented['mask'].long())
        if self.return_spacing:
            spacing = torch.tensor(spacing, dtype=torch.float32)
            return (image, mask, spacing)
        return (image, mask)
