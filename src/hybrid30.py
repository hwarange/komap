"""Isolated gated and whole-context variants of the running Hybrid29 model."""
import torch
from torch import nn

from hybrid29 import ParallelLocalEncoder, CNN_CHANNELS, MIT_CHANNELS, block
from context30 import ContextInput, ContextCropDataset, aligned_features, predict_context, CONTEXT_SIDE
from infer import predict_probs as original_predict

GATE = 'GatedMiTCNN30'
GLOBAL = 'GlobalContextMiTCNN30'


class GatedEncoder(ParallelLocalEncoder):
    def __init__(self, mit):
        super().__init__(mit)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(3001)
            self.gates = nn.ModuleList(nn.Sequential(nn.Conv2d(2, 8, 1), nn.GELU(), nn.Conv2d(8, 1, 1)) for _ in CNN_CHANNELS)
            for gate in self.gates:
                nn.init.zeros_(gate[-1].weight); nn.init.zeros_(gate[-1].bias)

    def forward(self, x):
        features = self.mit(x); local = self.stem(x)
        for index, (stage, projection, gate) in enumerate(zip(self.stages, self.projections, self.gates)):
            local = stage(local); reference = features[index+2]
            assert local.shape[-2:] == reference.shape[-2:]
            summary = torch.cat([reference.mean(1, keepdim=True), local.mean(1, keepdim=True)], 1)
            weight = 2*torch.sigmoid(gate(summary))
            features[index+2] = reference + weight*projection(local)
        return features


class GlobalEncoder(ParallelLocalEncoder):
    requires_context = True

    def __init__(self, mit):
        super().__init__(mit)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(3002)
            self.whole_stem = block(1, 8, stride=2)
            previous = 8; self.whole_stages = nn.ModuleList()
            for channels in CNN_CHANNELS:
                self.whole_stages.append(nn.Sequential(block(previous, channels, stride=2), block(channels, channels)))
                previous = channels
            self.context_projections = nn.ModuleList(nn.Conv2d(c+CNN_CHANNELS[-1], m, 1, bias=False)
                for c, m in zip(CNN_CHANNELS[1:], MIT_CHANNELS[1:]))
            for projection in self.context_projections: nn.init.zeros_(projection.weight)

    def forward(self, inputs):
        assert isinstance(inputs, ContextInput)
        features = super().forward(inputs.crop)
        whole = self.whole_stem(inputs.whole); levels = []
        for stage in self.whole_stages:
            whole = stage(whole); levels.append(whole)
        pooled = levels[-1].mean((-2, -1), keepdim=True)
        for index, (level, projection, stride) in enumerate(zip(levels[1:], self.context_projections, (8, 16, 32))):
            reference = features[index+3]
            roi = aligned_features(level, inputs.crop_to_whole, reference.shape[-2:], stride)
            contextual = torch.cat([roi, pooled.expand(-1, -1, *roi.shape[-2:])], 1)
            features[index+3] = reference + projection(contextual)
        return features


def predict_probs(model, *args, **kwargs):
    if getattr(model.encoder, 'requires_context', False):
        return predict_context(model, *args, **kwargs)
    return original_predict(model, *args, **kwargs)


def install(arch=None):
    from hybrid29 import install as install29
    train = install29()
    if not getattr(train.build_model, '_hybrid30_factory', False):
        original = train.build_model
        def factory(name, encoder, weights='imagenet', attention=None, hires=False, n_classes=4, feats=False):
            if name not in (GATE, GLOBAL):
                return original(name, encoder, weights, attention, hires, n_classes, feats)
            assert encoder == 'mit_b2' and not hires and not feats and attention is None and n_classes == 4
            model = original('HybridMiTCNN29', encoder, weights=weights)
            model.encoder = (GatedEncoder if name == GATE else GlobalEncoder)(model.encoder.mit)
            return model
        factory._hybrid30_factory = True
        # Retain the earlier hook marker so subsequent install29 calls do not wrap again.
        factory._hybrid29_factory = True
        train.build_model = factory
    train.predict_probs = predict_probs
    if arch == GLOBAL:
        import data
        data.CropDataset = ContextCropDataset
        train.CropDataset = ContextCropDataset
    return train


def architecture_spec(arch):
    if arch == GATE:
        return dict(arch=arch, base='HybridMiTCNN29', fusion='MiT + 2*sigmoid(spatial_gate)*projected_CNN',
                    gate_input='channel means of MiT and local CNN', initial_gate=1., strides=[4,8,16,32])
    if arch == GLOBAL:
        return dict(arch=arch, base='HybridMiTCNN29', whole_view=[CONTEXT_SIDE,CONTEXT_SIDE],
                    preserve_aspect_ratio=True, whole_cnn_channels=list(CNN_CHANNELS),
                    fusion='aligned whole-image CNN ROI + global average descriptor, zero-projected residual',
                    context_fusion_strides=[8,16,32], local_crops_and_rng_unchanged=True,
                    photometry='cloned scalar augmentation draws; illumination/noise in each view coordinates',
                    labels_used_in_context_input=False, whole_image_attention=False,
                    limitation='Adds context input and small CNN capacity together; not a parameter-matched context-only ablation')
    from hybrid29 import architecture_spec as spec29
    return spec29()
