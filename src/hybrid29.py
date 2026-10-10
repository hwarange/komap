"""MiT-B2 plus a small four-scale CNN; features fuse before the U-Net decoder.

The shared trainer and submitted inference sources are kept unchanged.
Install the factory hook in a dedicated process before loading hybrid checkpoints.
"""
import torch
from torch import nn
from torch.nn import functional as F

ARCH = 'HybridMiTCNN29'
CNN_CHANNELS = (16, 24, 32, 48)
MIT_CHANNELS = (64, 128, 320, 512)


def block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out_channels), nn.GELU(),
    )


class ParallelLocalEncoder(nn.Module):
    def __init__(self, mit):
        super().__init__()
        assert list(mit.out_channels) == [1, 0, *MIT_CHANNELS]
        self.mit = mit
        # Extra initialization must not change base initialization or crop RNG.
        with torch.random.fork_rng(devices=[]):
            # Parameters are created on CPU. torch.manual_seed also reseeds
            # CUDA generators; seed the CPU generator alone here.
            torch.default_generator.manual_seed(2900)
            self.stem = block(1, 8, stride=2)
            self.stages = nn.ModuleList()
            previous = 8
            for channels in CNN_CHANNELS:
                self.stages.append(nn.Sequential(block(previous, channels, stride=2), block(channels, channels)))
                previous = channels
            self.projections = nn.ModuleList(nn.Conv2d(c, m, 1, bias=False) for c, m in zip(CNN_CHANNELS, MIT_CHANNELS))
            # Start as the exact base network; the residual branch learns gradually.
            for projection in self.projections:
                nn.init.zeros_(projection.weight)

    @property
    def out_channels(self):
        return self.mit.out_channels

    @property
    def output_stride(self):
        return self.mit.output_stride

    def forward(self, x):
        features = self.mit(x)
        local = self.stem(x)
        for index, (stage, projection) in enumerate(zip(self.stages, self.projections)):
            local = stage(local)
            reference = features[index + 2]
            aligned = local if local.shape[-2:] == reference.shape[-2:] else F.interpolate(
                local, size=reference.shape[-2:], mode='bilinear', align_corners=False)
            features[index + 2] = reference + projection(aligned)
        return features


def install():
    """Install only in this interpreter; old checkpoint behavior is preserved."""
    import train
    if getattr(train.build_model, '_hybrid29_factory', False):
        return train
    original = train.build_model

    def factory(arch, encoder, weights='imagenet', attention=None, hires=False, n_classes=4, feats=False):
        if arch != ARCH:
            return original(arch, encoder, weights, attention, hires, n_classes, feats)
        assert encoder == 'mit_b2' and not hires and not feats and attention is None and n_classes == 4
        model = original('Unet', encoder, weights=weights, n_classes=n_classes)
        model.encoder = ParallelLocalEncoder(model.encoder)
        return model

    factory._hybrid29_factory = True
    train.build_model = factory
    return train


def architecture_spec():
    return dict(arch=ARCH, transformer='MiT-B2 pretrained ImageNet', cnn_channels=list(CNN_CHANNELS),
                fusion_strides=[4, 8, 16, 32], fusion='MiT features + zero-initialized1x1 projected CNN features',
                cnn_normalization='GroupNorm8', decoder='unchanged U-Net decoder', input_channels=1,
                same_crop_both_branches=True, whole_image_context_added=False,
                previous_hires_difference='Four encoder-scale features fused before decoding, rather than full/half-scale features after decoding')
