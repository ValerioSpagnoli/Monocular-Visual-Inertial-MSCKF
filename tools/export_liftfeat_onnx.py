"""
Export the LiftFeat network (Liu et al., "LiftFeat: 3D Geometry-Aware Local Feature Matching", ICRA 2025) to ONNX with
the same interface as models/xfeat.onnx, so that the C++ XFeatExtractor runs it unchanged:
  image        (1, 1, H, W)       grayscale, H and W multiples of 32 (the network starts with an instance norm, so
                                  the pixel scale does not matter)
  descriptors  (1, 64, H/8, W/8)  L2-normalized dense descriptors after the 3D-aware lifting (FeatureBooster)
  heatmap      (1, 1, H, W)       keypoint probability (softmax over the 8x8 cells, rearranged to full resolution)
  reliability  (1, 1, H/8, W/8)   constant 1: LiftFeat scores keypoints with the heatmap only
This reproduces LiftFeat.extract (models/liftfeat_wrapper.py); NMS (threshold 0.1), scores and descriptor sampling
are done in C++. The LiftFeat repository has no license, so the model is generated locally and not committed.

    pixi run -e export export-liftfeat      # -> models/liftfeat.onnx
"""
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'LiftFeat')
sys.path.insert(0, ROOT)
sys.modules.setdefault('cv2', types.ModuleType('cv2'))  # imported by models/model.py, used only in its __main__
from models.model import LiftFeatSPModel  # noqa: E402
from utils.config import featureboost_config  # noqa: E402


class LiftFeatDense(nn.Module):
    def __init__(self, net: LiftFeatSPModel):
        super().__init__()
        self.net = net

    def forward(self, image):
        M1, K1, D1 = self.net.forward1(image)
        refine_M = self.net.forward2(M1, K1, D1)
        refine_M = refine_M.reshape(M1.shape[0], M1.shape[2], M1.shape[3], -1).permute(0, 3, 1, 2)
        descriptors = F.normalize(refine_M, 2, dim=1)

        scores = torch.softmax(K1, dim=1)[:, :64]
        B, _, h8, w8 = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, h8, w8, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 1, h8 * 8, w8 * 8)
        return descriptors, heatmap, torch.ones_like(K1[:, :1])


def unfold2d_exportable(x, ws=2):
    """LiftFeatSPModel._unfold2d: identical to pixel_unshuffle, which (unlike Tensor.unfold) can be exported to ONNX
    with dynamic height and width."""
    return F.pixel_unshuffle(x, ws)


if __name__ == '__main__':
    output = os.path.join(os.path.dirname(__file__), '..', 'models', 'liftfeat.onnx')
    net = LiftFeatSPModel(featureboost_config).eval()
    weights = torch.load(os.path.join(ROOT, 'weights', 'LiftFeat.pth'), map_location='cpu')
    missing, unexpected = net.load_state_dict(weights, strict=False)
    print(f'missing keys: {missing}, unexpected keys: {unexpected}')

    sample = torch.rand(1, 3, 64, 96)
    assert torch.equal(net._unfold2d(sample, ws=8), unfold2d_exportable(sample, ws=8))
    net._unfold2d = unfold2d_exportable
    model = LiftFeatDense(net).eval()

    example = torch.rand(1, 1, 480, 736) * 255
    torch.onnx.export(model, (example,), output, input_names=['image'], output_names=['descriptors', 'heatmap', 'reliability'],
                      dynamic_axes={'image': {2: 'height', 3: 'width'}, 'descriptors': {2: 'h8', 3: 'w8'},
                                    'heatmap': {2: 'height', 3: 'width'}, 'reliability': {2: 'h8', 3: 'w8'}},
                      opset_version=17, dynamo=False)

    # Check against PyTorch on a different resolution
    import onnxruntime
    session = onnxruntime.InferenceSession(output, providers=['CPUExecutionProvider'])
    test = torch.rand(1, 1, 480, 640) * 255
    with torch.no_grad(): reference = [t.numpy() for t in model(test)]
    exported = session.run(None, {'image': test.numpy()})
    for name, a, b in zip(['descriptors', 'heatmap', 'reliability'], reference, exported):
        print(f'{name:12s} {a.shape}  max |difference| {np.abs(a - b).max():.2e}')
    print(f'Saved {os.path.abspath(output)} ({os.path.getsize(output) / 1e6:.1f} MB)')
