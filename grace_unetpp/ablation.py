"""Only the paper's own w/o-GFM component ablation; no comparison networks."""
import inspect
from torch import nn
from .model import ConfidenceGuidedRefiner

class NoGFMRefineUNetPP(nn.Module):
    """UNet++ plus CGRR, used for the w/o GFM component ablation."""

    def __init__(self, source, weights="imagenet"):
        super().__init__()
        self.base_model = source.smp.UnetPlusPlus(
            encoder_name=source.ENCODER,
            encoder_weights=weights,
            in_channels=source.IN_CHANNELS,
            classes=source.NUM_CLASSES,
            encoder_depth=5,
        )
        decoder_channels = self.base_model.segmentation_head[0].in_channels
        self.refiner = ConfidenceGuidedRefiner(decoder_channels, source.NUM_CLASSES)
        self.unpack_decoder = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in inspect.signature(self.base_model.decoder.forward).parameters.values())

    def forward(self, x):
        features = list(self.base_model.encoder(x))
        decoded = (self.base_model.decoder(*features) if self.unpack_decoder
                   else self.base_model.decoder(features))
        logits = self.base_model.segmentation_head(decoded)
        return logits + self.refiner(decoded, logits)

    def set_trainable(self, phase):
        for parameter in self.parameters():
            parameter.requires_grad = False
        if phase == "head":
            modules = [self.refiner]
        elif phase == "tail":
            modules = [self.base_model.decoder, self.base_model.segmentation_head, self.refiner]
        else:
            raise ValueError(phase)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
