"""Paper model. State-dict names match the original GRACE training implementation."""
import inspect
import math
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class GravityPhysics2D(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.green_kernel = None

    def _create_green_kernel(self, size, device):
        H, W = size
        ky = torch.fft.fftfreq(H, device=device).view(-1, 1)
        kx = torch.fft.fftfreq(W, device=device).view(1, -1)
        k_sq = ky ** 2 + kx ** 2
        k_sq[0, 0] = 1.0
        green_kernel = -1.0 / (k_sq * 4 * (math.pi ** 2))
        green_kernel[0, 0] = 0.0
        return green_kernel

    def forward(self, mass_map):
        spatial_size = mass_map.shape[2:]
        if self.green_kernel is None or self.green_kernel.shape != spatial_size or self.green_kernel.device != mass_map.device:
            self.green_kernel = self._create_green_kernel(spatial_size, mass_map.device)

        rho_hat = torch.fft.fft2(mass_map)
        phi_hat = rho_hat * self.green_kernel
        phi = torch.fft.ifft2(phi_hat).real
        return phi

class GravityMLPGate(nn.Module):
    def __init__(self, source, channels):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.BatchNorm2d(1))
        self.activation = nn.Sequential(nn.Conv2d(1, 4, 1), nn.Tanh(),
                                        nn.Conv2d(4, 1, 1), nn.Softplus())
        self.solver = source.GravityPhysics2D()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            mass = self.activation(self.project(x.float()))
            attention = torch.sigmoid(-self.solver(mass))
        return x + self.gamma.to(x.dtype) * x * attention.to(x.dtype)


class ConfidenceGuidedRefiner(nn.Module):
    def __init__(self, decoder_channels, classes):
        super().__init__()
        hidden = max(8, decoder_channels // 2)
        self.block = nn.Sequential(
            nn.Conv2d(decoder_channels + 1, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, classes, 1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, decoded, base_logits):
        delta, uncertainty = self.correction(decoded, base_logits)
        return delta * (0.5 + uncertainty.to(delta.dtype))

    def correction(self, decoded, base_logits):
        """Return the ungated correction and ambiguity for diagnostics."""
        probs = F.softmax(base_logits.float(), dim=1)
        uncertainty = 1.0 - probs.max(dim=1, keepdim=True).values
        delta = self.block(torch.cat([decoded, uncertainty.to(decoded.dtype)], dim=1))
        return delta, uncertainty


class GravityMLPRefineUNetPP(nn.Module):
    def __init__(self, source, weights="imagenet"):
        super().__init__()
        self.base_model = source.smp.UnetPlusPlus(
            encoder_name=source.ENCODER,
            encoder_weights=weights,
            in_channels=source.IN_CHANNELS,
            classes=source.NUM_CLASSES,
            encoder_depth=5,
        )
        self.gravity = GravityMLPGate(source, self.base_model.encoder.out_channels[-1])
        decoder_channels = self.base_model.segmentation_head[0].in_channels
        self.refiner = ConfidenceGuidedRefiner(decoder_channels, source.NUM_CLASSES)
        self.unpack_decoder = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in inspect.signature(self.base_model.decoder.forward).parameters.values()
        )

    def forward(self, x):
        return self.forward_with_diagnostics(x)[0]

    def forward_with_diagnostics(self, x):
        features = list(self.base_model.encoder(x))
        features[-1] = self.gravity(features[-1])
        decoded = (self.base_model.decoder(*features) if self.unpack_decoder
                   else self.base_model.decoder(features))
        logits = self.base_model.segmentation_head(decoded)
        delta, ambiguity = self.refiner.correction(decoded, logits)
        gated_delta = delta * (0.5 + ambiguity.to(delta.dtype))
        return logits + gated_delta, logits, gated_delta, ambiguity

    def set_trainable(self, mode):
        for p in self.parameters():
            p.requires_grad = False
        if mode == "head":
            modules = [self.refiner]
        elif mode == "tail":
            modules = [self.base_model.decoder, self.base_model.segmentation_head, self.refiner]
        elif mode == "all":
            modules = [self]
        else:
            raise ValueError(mode)
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True


class GRACEUNetPlusPlus(GravityMLPRefineUNetPP):
    """EfficientNet-B0 / UNet++ with GFM and CGRR.

    ACDC: in_channels=1, classes=4. MSD Spleen: in_channels=3, classes=2.
    encoder_weights=None avoids downloads; "imagenet" matches paper training.
    forward() returns logits; forward_with_diagnostics() additionally returns
    base logits, gated residual correction and the ambiguity map.
    """
    def __init__(self, in_channels=1, classes=4, encoder_weights=None):
        if classes < 2:
            raise ValueError("CGRR uses multiclass softmax; use at least two classes.")
        source = SimpleNamespace(smp=smp, ENCODER="efficientnet-b0",
                                 IN_CHANNELS=in_channels, NUM_CLASSES=classes,
                                 GravityPhysics2D=GravityPhysics2D)
        super().__init__(source, weights=encoder_weights)

    def set_stage(self, phase):
        if phase == "gfm":
            for p in self.parameters():
                p.requires_grad = True
            for p in self.refiner.parameters():
                p.requires_grad = False
        else:
            self.set_trainable(phase)
