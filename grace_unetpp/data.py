import importlib
from types import SimpleNamespace
import torch
import segmentation_models_pytorch as smp
from .model import GravityPhysics2D

def make_source(dataset, data_root, device=None, scan=True):
    if dataset not in {"acdc", "msd"}:
        raise ValueError(dataset)
    module = importlib.import_module(f"grace_unetpp.datasets.{dataset}")
    source = SimpleNamespace(**{k:v for k,v in vars(module).items() if not k.startswith("__")})
    source.smp = smp
    source.GravityPhysics2D = GravityPhysics2D
    source.DEVICE = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source.USE_AMP = source.DEVICE.startswith("cuda")
    source.PIN_MEMORY = source.USE_AMP
    if dataset == "msd":
        source.MSDNiiDataset = module.CachedMSDDataset
    if scan:
        source.cases = (source.scan_acdc_cases(data_root) if dataset == "acdc"
                        else source.scan_msd_cases(data_root))
    return source

def correct_normalization(source, loaders):
    # Kept for compatibility with the original shared loader construction.
    for loader in loaders:
        transforms = loader.dataset.transform.transforms
        for i, transform in enumerate(transforms):
            if isinstance(transform, source.A.Normalize):
                transforms[i] = source.A.Normalize(
                    mean=(0.5,) * source.IN_CHANNELS,
                    std=(0.5,) * source.IN_CHANNELS, max_pixel_value=1.0)
