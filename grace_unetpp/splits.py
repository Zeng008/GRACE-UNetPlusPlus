from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from grace_unetpp.data import correct_normalization


def _balanced_rank_bins(values, bins: int):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(len(values), dtype=np.int64)
    labels[order] = np.arange(len(values)) * bins // len(values)
    return labels.tolist()


def foreground_burdens(dataset: str, source):
    """Compute a label-derived balance proxy without using it in training."""
    values = {}
    if dataset == "acdc":
        for case_id, files in sorted(source.cases.items()):
            foreground = total = 0
            for filename in files:
                with source.h5py.File(filename, "r") as handle:
                    label = np.asarray(handle["label"])
                foreground += int(np.count_nonzero(label))
                total += int(label.size)
            values[case_id] = foreground / max(total, 1)
    else:
        for case_id, (_image_path, mask_path) in sorted(source.cases.items()):
            nii = source.nib.load(mask_path)
            label = np.asanyarray(nii.dataobj)
            values[case_id] = float(np.count_nonzero(label)) / max(int(label.size), 1)
    return values


def make_fold_manifest(dataset: str, source, output: Path, folds=5, cv_seed=20261027):
    """Create patient/volume-level outer folds and a balanced inner validation set."""
    output = Path(output)
    if output.exists():
        manifest = json.loads(output.read_text())
        expected_ids = sorted(source.cases)
        recorded_ids = sorted(manifest.get("burdens", {}))
        if (manifest["dataset"] != dataset or manifest["folds"] != folds
                or manifest.get("cv_seed") != cv_seed or recorded_ids != expected_ids):
            raise RuntimeError(f"Existing incompatible fold manifest: {output}")
        return manifest

    ids = sorted(source.cases)
    burdens = foreground_burdens(dataset, source)
    bins = min(4, max(2, len(ids) // folds))
    strata = _balanced_rank_bins([burdens[case_id] for case_id in ids], bins)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    fold_rows = []
    indices = np.arange(len(ids))
    for fold, (trainval_idx, test_idx) in enumerate(splitter.split(indices, strata)):
        trainval_strata = np.asarray(strata)[trainval_idx]
        val_count = max(bins, int(round(0.10 * len(ids))))
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=val_count, random_state=cv_seed + 100 + fold)
        train_sub, val_sub = next(inner.split(trainval_idx, trainval_strata))
        train_idx = trainval_idx[train_sub]
        val_idx = trainval_idx[val_sub]
        row = {
            "fold": fold,
            "train_ids": [ids[i] for i in train_idx],
            "validation_ids": [ids[i] for i in val_idx],
            "test_ids": [ids[i] for i in test_idx],
        }
        assert not (set(row["train_ids"]) & set(row["validation_ids"]))
        assert not (set(row["train_ids"]) & set(row["test_ids"]))
        assert not (set(row["validation_ids"]) & set(row["test_ids"]))
        fold_rows.append(row)
    flattened = [case for row in fold_rows for case in row["test_ids"]]
    if sorted(flattened) != ids:
        raise RuntimeError("Outer-fold test sets do not cover every case exactly once")
    manifest = {
        "dataset": dataset,
        "folds": folds,
        "cv_seed": cv_seed,
        "stratification": "balanced rank bins of foreground voxel fraction",
        "burdens": burdens,
        "strata": dict(zip(ids, strata)),
        "fold_assignments": fold_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output)
    return manifest


def _flatten_acdc(cases, ids):
    return [filename for case_id in ids for filename in cases[case_id]]


def loaders_for_fold(dataset: str, source, fold_row, run_seed: int,
                     batch_size=None, workers=4, use_augmentation=True):
    """Build loaders from an explicit fold; no module-level split state is used."""
    train_ids = fold_row["train_ids"]
    val_ids = fold_row["validation_ids"]
    test_ids = fold_row["test_ids"]
    if dataset == "acdc":
        train_items = _flatten_acdc(source.cases, train_ids)
        val_items = _flatten_acdc(source.cases, val_ids)
        test_items = _flatten_acdc(source.cases, test_ids)
        train_ds = source.ACDCH5Dataset(
            train_items, img_size=source.IMG_SIZE, is_train=use_augmentation)
        val_ds = source.ACDCH5Dataset(val_items, img_size=source.IMG_SIZE, is_train=False)
        test_ds = source.ACDCH5Dataset(
            test_items, img_size=source.IMG_SIZE, is_train=False, return_spacing=True)
    else:
        train_items = [source.cases[case_id] for case_id in train_ids]
        val_items = [source.cases[case_id] for case_id in val_ids]
        test_items = [source.cases[case_id] for case_id in test_ids]
        train_ds = source.MSDNiiDataset(
            train_items, img_size=source.IMG_SIZE, is_train=use_augmentation,
            sampling_seed=run_seed)
        val_ds = source.MSDNiiDataset(
            val_items, img_size=source.IMG_SIZE, is_train=False, sampling_seed=run_seed)
        test_ds = source.MSDNiiDataset(
            test_items, img_size=source.IMG_SIZE, is_train=False,
            return_spacing=True, sampling_seed=run_seed)

    generator = torch.Generator().manual_seed(run_seed)
    kwargs = {
        "num_workers": workers,
        "pin_memory": source.PIN_MEMORY,
        "worker_init_fn": source.seed_worker,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_ds, batch_size=batch_size or source.BATCH_SIZE, shuffle=True,
        generator=generator, **kwargs)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size or source.BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **kwargs)
    correct_normalization(source, [train_loader, val_loader, test_loader])
    return train_loader, val_loader, test_loader

