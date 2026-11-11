from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _sample_metadata(dataset_name, dataset, index):
    if dataset_name == "acdc":
        path = Path(dataset.file_list[index])
        match = re.match(r"(.+?_frame\d+)_slice_(\d+)$", path.stem)
        volume_id = match.group(1) if match else path.stem
        slice_index = int(match.group(2)) if match else index
        case_id = volume_id.split("_frame")[0]
        return case_id, volume_id, slice_index, str(path)
    volume_index, slice_index = dataset.slice_map[index]
    image_path, _mask_path = dataset.volume_pairs[volume_index]
    name = Path(image_path).name
    case_id = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    return case_id, case_id, int(slice_index), str(image_path)


def _surface(mask):
    if not mask.any():
        return mask
    return np.logical_xor(mask, binary_erosion(mask, border_value=0))


def boundary_metrics_grid(prediction, target, spacing=None):
    """Return symmetric HD95/ASD on the spacing-weighted evaluation grid.

    HD95 is the 95th percentile of the union of both directed surface
    distances.  ASD follows the equation in the paper: the two directed
    surface means receive equal weight.  The previous implementation took a
    point-count-weighted mean after concatenation, which is not equivalent
    when the two surfaces contain different numbers of pixels.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    spacing = (np.ones(2, dtype=np.float64) if spacing is None
               else np.asarray(spacing, dtype=np.float64).reshape(-1)[-2:])
    if spacing.size != 2 or not np.isfinite(spacing).all() or (spacing <= 0).any():
        spacing = np.ones(2, dtype=np.float64)
    if not prediction.any() and not target.any():
        return np.nan, np.nan
    if not prediction.any() or not target.any():
        extent = (np.asarray(prediction.shape, dtype=np.float64) - 1.0) * spacing
        penalty = float(np.sqrt(np.square(extent).sum()))
        return penalty, penalty
    pred_surface = _surface(prediction)
    true_surface = _surface(target)
    pred_to_true = distance_transform_edt(
        ~true_surface, sampling=spacing)[pred_surface]
    true_to_pred = distance_transform_edt(
        ~pred_surface, sampling=spacing)[true_surface]
    distances = np.concatenate([pred_to_true, true_to_pred])
    asd = 0.5 * (float(pred_to_true.mean()) + float(true_to_pred.mean()))
    return float(np.percentile(distances, 95)), asd


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


@torch.no_grad()
def evaluate_model(source, dataset_name: str, model, test_loader, output_dir,
                   fold: int, method: str, probability_bins=1000):
    """Save out-of-fold, case-level metrics and curve sufficient statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(source.DEVICE)
    model.to(device).eval()
    classes = source.NUM_CLASSES
    confusion = np.zeros((classes, classes), dtype=np.int64)
    positive_hist = np.zeros((classes - 1, probability_bins), dtype=np.int64)
    negative_hist = np.zeros_like(positive_hist)
    histogram_edges = np.linspace(0.0, 1.0, probability_bins + 1)
    cases = defaultdict(lambda: {
        "intersection": np.zeros(classes - 1, dtype=np.float64),
        "denominator": np.zeros(classes - 1, dtype=np.float64),
        "hd95": [[] for _ in range(classes - 1)],
        "asd": [[] for _ in range(classes - 1)],
        "slices": 0,
    })
    slice_rows = []

    for index, batch in enumerate(test_loader):
        image, target = batch[:2]
        spacing = (batch[2][0].cpu().numpy() if len(batch) >= 3
                   else np.ones(2, dtype=np.float32))
        image = image.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=source.USE_AMP):
            logits = model(image)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        prediction = probabilities.argmax(axis=0).astype(np.int64)
        target_np = target[0].numpy().astype(np.int64)
        case_id, volume_id, slice_index, source_path = _sample_metadata(
            dataset_name, test_loader.dataset, index)
        case = cases[case_id]
        case["slices"] += 1

        indices = target_np.ravel() * classes + prediction.ravel()
        confusion += np.bincount(indices, minlength=classes * classes).reshape(classes, classes)
        row = {
            "dataset": dataset_name,
            "fold": fold,
            "method": method,
            "case_id": case_id,
            "volume_id": volume_id,
            "slice_index": slice_index,
            "dataset_index": index,
            "source_path": source_path,
            "foreground_pixels": int(np.count_nonzero(target_np)),
        }
        slice_dice, slice_hd95, slice_asd = [], [], []
        for offset, label in enumerate(range(1, classes)):
            pred_mask = prediction == label
            true_mask = target_np == label
            intersection = float(np.logical_and(pred_mask, true_mask).sum())
            denominator = float(pred_mask.sum() + true_mask.sum())
            case["intersection"][offset] += 2.0 * intersection
            case["denominator"][offset] += denominator
            dice = (2.0 * intersection / denominator) if denominator else np.nan
            hd95, asd = boundary_metrics_grid(pred_mask, true_mask, spacing=spacing)
            if np.isfinite(hd95):
                case["hd95"][offset].append(hd95)
                case["asd"][offset].append(asd)
            row[f"Dice_class_{label}"] = dice
            row[f"HD95_grid_class_{label}"] = hd95
            row[f"ASD_grid_class_{label}"] = asd
            slice_dice.append(dice)
            slice_hd95.append(hd95)
            slice_asd.append(asd)
            labels = true_mask.ravel()
            values = probabilities[label].ravel()
            positive_hist[offset] += np.histogram(
                values[labels], bins=histogram_edges)[0]
            negative_hist[offset] += np.histogram(
                values[~labels], bins=histogram_edges)[0]
        row["Mean_Dice"] = _safe_mean(slice_dice)
        row["Mean_HD95_grid"] = _safe_mean(slice_hd95)
        row["Mean_ASD_grid"] = _safe_mean(slice_asd)
        slice_rows.append(row)

    case_rows = []
    for case_id, values in sorted(cases.items()):
        row = {
            "dataset": dataset_name,
            "fold": fold,
            "method": method,
            "case_id": case_id,
            "slices": values["slices"],
        }
        dices, hd95s, asds = [], [], []
        for offset, label in enumerate(range(1, classes)):
            denominator = values["denominator"][offset]
            dice = values["intersection"][offset] / denominator if denominator else np.nan
            hd95 = _safe_mean(values["hd95"][offset])
            asd = _safe_mean(values["asd"][offset])
            row[f"Dice_class_{label}"] = dice
            row[f"HD95_grid_class_{label}"] = hd95
            row[f"ASD_grid_class_{label}"] = asd
            dices.append(dice)
            hd95s.append(hd95)
            asds.append(asd)
        row["Mean_Dice"] = _safe_mean(dices)
        row["Mean_HD95_grid"] = _safe_mean(hd95s)
        row["Mean_ASD_grid"] = _safe_mean(asds)
        case_rows.append(row)

    pd.DataFrame(slice_rows).to_csv(output_dir / "slice_metrics.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output_dir / "case_metrics.csv", index=False)
    np.save(output_dir / "confusion_matrix.npy", confusion)
    np.savez_compressed(
        output_dir / "probability_histograms.npz",
        positive=positive_hist, negative=negative_hist, edges=histogram_edges)
    return pd.DataFrame(case_rows)
