from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _atomic_json(path: Path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(model, path, device="cpu", strict=True):
    checkpoint = torch_load(path, map_location=device)
    result = model.load_state_dict(_model_state(checkpoint), strict=strict)
    return checkpoint, result


@torch.no_grad()
def validation_dice(model, loader, classes: int, device):
    """Foreground Dice pooled over the complete validation set."""
    model.eval()
    intersections = torch.zeros(classes - 1, dtype=torch.float64)
    denominators = torch.zeros(classes - 1, dtype=torch.float64)
    for batch in loader:
        image, target = batch[:2]
        prediction = model(image.to(device, non_blocking=True)).argmax(dim=1).cpu()
        target = target.cpu()
        for offset, label in enumerate(range(1, classes)):
            pred_mask = prediction == label
            true_mask = target == label
            intersections[offset] += 2 * torch.logical_and(pred_mask, true_mask).sum()
            denominators[offset] += pred_mask.sum() + true_mask.sum()
    valid = denominators > 0
    if not valid.any():
        return 0.0
    return float((intersections[valid] / denominators[valid]).mean())


def _save_training_checkpoint(path, model, optimizer, scheduler, epoch, best_score,
                              run_seed, history, train_loader, extra=None):
    generator = getattr(train_loader, "generator", None)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": best_score,
        "run_seed": run_seed,
        "history": history,
        "extra": extra or {},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state() if generator is not None else None,
        },
    }, path)


def _restore_rng(checkpoint, train_loader):
    state = checkpoint.get("rng_state")
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    generator = getattr(train_loader, "generator", None)
    if generator is not None and state["loader"] is not None:
        generator.set_state(state["loader"].cpu())


def train_stage(source, model, train_loader, validation_loader, output_dir,
                epochs: int, learning_rate: float, run_seed: int,
                weight_decay=1e-4, patience=8, min_epochs=8,
                initialize_best=False, extra=None, optimizer_name="adamw",
                scheduler_name="cosine_warm_restarts"):
    """Train one restartable stage and select on foreground validation Dice."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "complete.json"
    best_path = output_dir / "best.pth"
    last_path = output_dir / "last.pth"
    history_path = output_dir / "history.json"
    if complete_path.exists() and best_path.exists():
        load_checkpoint(model, best_path, device=source.DEVICE)
        return json.loads(history_path.read_text())

    seed_everything(run_seed)
    device = torch.device(source.DEVICE)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters in this stage")
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            trainable, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if scheduler_name == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, min(10, epochs)), T_mult=2)
    elif scheduler_name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=source.USE_AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=source.USE_AMP)

    start_epoch = 0
    best_score = -float("inf")
    best_epoch = -1
    history = []
    if last_path.exists():
        checkpoint = torch_load(last_path, map_location=device)
        model.load_state_dict(_model_state(checkpoint))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        history = list(checkpoint.get("history", []))
        if history:
            best_epoch = max(history, key=lambda row: row["validation_dice"])["epoch"]
        _restore_rng(checkpoint, train_loader)
    elif initialize_best:
        best_score = validation_dice(model, validation_loader, source.NUM_CLASSES, device)
        best_epoch = -1
        _save_training_checkpoint(
            best_path, model, optimizer, scheduler, -1, best_score,
            run_seed, history, train_loader, extra)

    bad_epochs = max(0, start_epoch - 1 - best_epoch)
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        for image, target, *_ in train_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                    device_type=device.type, enabled=source.USE_AMP):
                logits = model(image)
                loss = source.combined_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step(epoch + 1)
        score = validation_dice(model, validation_loader, source.NUM_CLASSES, device)
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation_dice": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if score > best_score + 1e-6:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            _save_training_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best_score,
                run_seed, history, train_loader, extra)
        else:
            bad_epochs += 1
        _save_training_checkpoint(
            last_path, model, optimizer, scheduler, epoch, best_score,
            run_seed, history, train_loader, extra)
        _atomic_json(history_path, history)
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"loss={row['training_loss']:.4f} | val_fg_dice={score:.4f}")
        if epoch + 1 >= min_epochs and bad_epochs >= patience:
            break

    if not best_path.exists():
        raise RuntimeError(f"No checkpoint was saved in {output_dir}")
    load_checkpoint(model, best_path, device=device)
    _atomic_json(complete_path, {
        "best_epoch": best_epoch,
        "best_validation_dice": best_score,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
    })
    return history
